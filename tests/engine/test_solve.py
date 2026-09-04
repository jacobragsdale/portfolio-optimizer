"""Tier 1: solving — the three hand-checkable order flows over the example's first account, binding limits with known answers, typed constraints and kinds through the shipped step, and every failure path."""

from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config.resolve import ConfigResolutionError, ResolvedConfig
from portfolio_optimizer.domain.objective import TermSpecError
from portfolio_optimizer.domain.order_flow import INFLOW
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.engine.build import StandardParams, standard
from portfolio_optimizer.engine.check import constraints_of, verify
from portfolio_optimizer.engine.solve import InfeasibleError, SolveSetupError
from portfolio_optimizer.engine.solve import solve as engine_solve
from tests.conftest import ALPHA, BUY_TERMS, CASH_CAP, CASH_FLOOR, SELL_TERMS, SHIPPED_CONSTRAINTS, TRANSACTION_COST, TURNOVER, Factories, Frames, Row, constraint_frame, resolved_example, typed_row

CASH: list[Row] = [CASH_FLOOR, CASH_CAP]
SECTOR_FLOOR: Row = typed_row("group_limit", "sector_floor", direction=">=", column="sector", bounds={"TECH": "0.5"})
"""The example's ``TECH`` floor: what stops the outflow from harvesting the whole of a loss."""


def resolved_with(terms: Sequence[object], **overrides: object) -> ResolvedConfig:
    """The example config over ``terms`` — records, or the example's term names — with any other section replaced; the inflow unless ``order_flow`` says otherwise."""
    named = {str(term["name"]): term for term in SELL_TERMS}
    return resolved_example(objective=[named[term] if isinstance(term, str) else term for term in terms], **overrides)


def solve(spec: ProblemSpec, chain: ChainState, resolved: ResolvedConfig, rows: Sequence[Row] = tuple(SHIPPED_CONSTRAINTS)) -> Solution:
    """Solve under the shipped constraint rows unless a case names its own; the engine reads them off the frame."""
    return engine_solve(spec, chain, resolved, constraint_frame(rows))


def first_account(make: Factories, frames: Frames, **details: object) -> ProblemSpec:
    """P1 exactly as the example book has it: A 3,000 @100 at cost, B 6,000 @60 against a price of 50, 400,000 of cash, a 40% name cap, a quarter of ADV, cash allowed up to 60%."""
    holdings = frames.holdings({"security_id": "A", "quantity": 3000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 6000, "avg_cost": Decimal(60)})
    account = make.details(**{"cash": Decimal(400_000), "max_weight": Decimal("0.4"), "max_adv_participation": Decimal("0.25"), "cash_ub": Decimal("0.6"), **details})
    return standard(make.portfolio_data(holdings=holdings, details=account))


def test_the_inflow_over_the_first_account_matches_the_analytic_optimum(make: Factories, frames: Frames) -> None:
    """C to its ADV budget, a quarter of NAV; then A to its cap; B has turned negative, so the last 5% of cash stays cash."""
    spec = first_account(make, frames)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(BUY_TERMS))
    assert solution.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(solution.w, [0.4, 0.3, 0.25], atol=1e-6)
    np.testing.assert_allclose(solution.buy, [0.1, 0.0, 0.25], atol=1e-6)
    assert solution.sell.tolist() == [0.0, 0.0, 0.0]
    alpha, cost = 0.03 * 0.4 - 0.01 * 0.3 + 0.05 * 0.25, 0.0005 * 0.1 + 0.002 * 0.25
    assert solution.objective == pytest.approx(-alpha + cost, abs=1e-6)
    assert solution.solver == "CLARABEL"
    assert solution.spec_hash == spec.content_hash()
    assert [record["kind"] for record in solution.constraints] == ["cash_limit", "cash_limit", "turnover_limit", "participation_limit"], "what the step applied travels with the answer"
    assert solution.duals["adv"] > 0.0 and solution.duals["cash_floor"] == pytest.approx(0.0, abs=1e-6), "the ADV budget binds and its shadow price says so; the cash floor is not reached"


def test_the_outflow_over_the_first_account_matches_the_analytic_optimum(make: Factories, frames: Frames) -> None:
    """B is held at a loss its long-term rate turns into 4 cents of tax refund per dollar sold, so it is harvested — down to where the ``TECH`` floor stops it; A is at cost and worth holding."""
    spec = first_account(make, frames)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(SELL_TERMS, order_flow="outflow"), [*SHIPPED_CONSTRAINTS, SECTOR_FLOOR])
    assert solution.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(solution.w, [0.3, 0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(solution.sell, [0.0, 0.1, 0.0], atol=1e-6)
    assert solution.buy.tolist() == [0.0, 0.0, 0.0]
    alpha, tax, cost = 0.03 * 0.3 - 0.01 * 0.2, -0.04 * 0.1, 0.0005 * 0.1
    assert solution.objective == pytest.approx(-alpha + tax + cost, abs=1e-6)
    assert solution.duals["sector_floor"] > 0.0, "the floor is what holds the rest of the loss unharvested, and its shadow price is the refund per dollar it costs"


def test_solving_twice_is_bitwise_identical(make: Factories, frames: Frames) -> None:
    spec = first_account(make, frames)
    resolved = resolved_with(BUY_TERMS)
    first = solve(spec, ChainState.empty(spec.security_ids), resolved)
    second = solve(spec, ChainState.empty(spec.security_ids), resolved)
    assert first.w.tobytes() == second.w.tobytes()
    assert first.objective == second.objective


def test_turnover_cap_binds_with_the_known_answer(make: Factories) -> None:
    spec = make.spec(n=2, w0=np.array([0.5, 0.0]), quantity_held=np.array([5_000.0, 0.0]), max_turnover=0.2, cash_ub=0.5)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"]), [*CASH, TURNOVER])
    np.testing.assert_allclose(solution.w, [0.5, 0.2], atol=1e-6, err_msg="S1 is the name worth buying, and the cap stops it at a fifth of NAV with cash to spare")
    assert float(solution.buy.sum()) == pytest.approx(0.2, abs=1e-6)


def test_a_portfolio_whose_predecessors_spent_a_names_budget_cannot_buy_it(make: Factories, frames: Frames) -> None:
    spec = first_account(make, frames, max_weight=Decimal("0.6"))  # the example's second account: room to take A alone when C is unbuyable
    spent = ChainState(security_ids=spec.security_ids, traded_quantity=np.array([1000.0, 0.0, 25_000.0]), predecessors=("P1",))
    solution = solve(spec, spent, resolved_with(BUY_TERMS))
    assert solution.buy[2] == pytest.approx(0.0, abs=1e-6), "C's whole ADV budget went to the predecessor"
    np.testing.assert_allclose(solution.w, [0.6, 0.3, 0.0], atol=1e-6, err_msg="so the cash goes to A, up to the wider cap")
    fresh = solve(spec, ChainState.empty(spec.security_ids), resolved_with(BUY_TERMS))
    assert fresh.buy[2] == pytest.approx(0.25, abs=1e-6)


def test_an_inflow_solve_only_buys_and_verifies(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 2500, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 5000, "avg_cost": Decimal(50)})
    details = make.details(cash=Decimal(500_000), max_weight=Decimal("0.5"), max_adv_participation=Decimal("0.25"))
    spec = standard(make.portfolio_data(holdings=holdings, details=details))
    resolved = resolved_with(BUY_TERMS)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, [0.5, 0.25, 0.25], atol=1e-6, err_msg="C takes its whole ADV budget, the rest of the cash goes to A, the best of what is left, up to its cap")
    assert (solution.w >= spec.w0 - 1e-9).all() and solution.sell.tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(solution.buy, [0.25, 0.0, 0.25], atol=1e-6)
    report = verify(spec, solution, ChainState.empty(spec.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_sells", "trade_balance", "nonneg_buy", "sell_absent", "lb", "ub"}


def test_an_inflow_run_cannot_sell_its_way_back_inside_a_cap_and_says_so(make: Factories, frames: Frames) -> None:
    resolved = resolved_with(BUY_TERMS)
    over_cap = standard(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 8000, "avg_cost": Decimal(100)}), details=make.details(max_weight=Decimal("0.6"))))
    with pytest.raises(InfeasibleError, match=r"names whose cap is below their holding, which this order flow cannot trade out of: \['A'\]"):
        solve(over_cap, ChainState.empty(over_cap.security_ids), resolved)
    fully_invested = standard(
        make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 10000, "avg_cost": Decimal(100)}), details=make.details(cash_lb=Decimal("0.1"), cash_ub=Decimal("0.2")))
    )
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 below cash_lb 0\.100000, and an inflow can only lower cash"):
        solve(fully_invested, ChainState.empty(fully_invested.security_ids), resolved)


def test_an_inflow_holds_an_over_cap_name_where_it_is_when_the_build_says_so(make: Factories, frames: Frames) -> None:
    """The over-cap book above under ``hold_breached_starts``: A's cap moves to its 80% weight, so the start is feasible, A is neither bought nor sold, and the cash goes to the rest."""
    resolved = resolved_with(BUY_TERMS)
    data = make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 8000, "avg_cost": Decimal(100)}), details=make.details(max_weight=Decimal("0.6")))
    held = standard(data, StandardParams(hold_breached_starts=True))
    assert held.ub[0] == pytest.approx(0.8) and not held.buyable[0], "the cap is the weight, so A is outside the buyable set the schedule couples through"
    solution = solve(held, ChainState.empty(held.security_ids), resolved)
    assert solution.w[0] == pytest.approx(0.8, abs=1e-6) and solution.buy[0] == pytest.approx(0.0, abs=1e-6)
    assert solution.buy[1:].sum() > 0.0, "the cash is put to work elsewhere"
    report = verify(held, solution, ChainState.empty(held.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, report.violated


def test_a_rebalance_trades_out_of_the_starts_an_inflow_cannot(make: Factories, frames: Frames) -> None:
    """The two books the inflow refuses above: the over-cap name is sold down to its cap, and the fully invested book raises the cash its floor asks for — which is why a failed inflow is retried as a rebalance."""
    resolved = resolved_with(BUY_TERMS, order_flow="rebalance")
    over_cap = standard(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 8000, "avg_cost": Decimal(100)}), details=make.details(max_weight=Decimal("0.6"))))
    solution = solve(over_cap, ChainState.empty(over_cap.security_ids), resolved)
    assert solution.w[0] <= 0.6 + 1e-6 and solution.sell[0] >= 0.2 - 1e-6, "A is sold down inside its cap — and past it, to fund C, whose alpha is better"
    report = verify(over_cap, solution, ChainState.empty(over_cap.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"trade_balance", "nonneg_buy", "nonneg_sell", "no_round_trip", "lb", "ub"}
    fully_invested = standard(
        make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 10000, "avg_cost": Decimal(100)}), details=make.details(cash_lb=Decimal("0.1"), cash_ub=Decimal("0.2")))
    )
    raised = solve(fully_invested, ChainState.empty(fully_invested.security_ids), resolved)
    assert 1.0 - raised.w.sum() >= 0.1 - 1e-6 and raised.sell[0] >= 0.1 - 1e-6, "the cash floor is met by selling A"
    assert (np.minimum(raised.buy, raised.sell) == 0).all(), "one variable per name: nothing is on both sides"


def test_a_term_that_rewards_a_side_is_refused_under_a_rebalance_by_name(make: Factories, frames: Frames) -> None:
    """``tax_cost`` rewards selling B, held at a loss; under a rebalance ``sell`` is convex, so the reward is not, and the term says which names rather than leaving the solver's DCP error to explain it."""
    spec = first_account(make, frames)
    with pytest.raises(TermSpecError, match=r"tax_cost: rewards sell on 1 name\(s\), e.g. \['B'\]; under a rebalance sell is convex rather than affine"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(SELL_TERMS, order_flow="rebalance"))


def test_an_outflow_solve_only_sells_and_couples_through_sells(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    universe = frames.three_security_universe().assign(adv_quantity=pd.Series([4000, 1_000_000, 100_000], dtype="Int64"), alpha=pd.Series([-0.03, -0.01, 0.05], dtype="Float64"))
    spec = standard(make.portfolio_data(holdings=holdings, universe=universe, details=make.details(max_adv_participation=Decimal("0.25"), cash_lb=Decimal(0), cash_ub=Decimal(1))))
    resolved = resolved_with(["alpha"], order_flow="outflow")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, [0.4, 0.0, 0.0], atol=1e-6, err_msg="both alphas are negative, so A sells its whole ADV budget (0.1) and B, which has budget to spare, sells out")
    assert (solution.w <= spec.w0 + 1e-9).all() and solution.buy.tolist() == [0.0, 0.0, 0.0]
    report = verify(spec, solution, ChainState.empty(spec.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_buys", "trade_balance", "nonneg_sell", "buy_absent", "lb", "ub"}
    spent = ChainState(security_ids=spec.security_ids, traded_quantity=np.array([1000.0, 0.0, 0.0]), predecessors=("P0",))
    np.testing.assert_allclose(solve(spec, spent, resolved).w, [0.5, 0.0, 0.0], atol=1e-6, err_msg="a predecessor's sells of A consumed the budget this portfolio would have sold into")
    report = verify(spec, _with_sell(solve(spec, spent, resolved), spec, np.array([0.05, 0.0, 0.0])), spent, resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.violated == ("adv/cumulative_participation",), "a sell past what the chain left of A's budget is caught: the verifier checks the chain against the sells, not the absent buys"


def _with_sell(solution: Solution, spec: ProblemSpec, sell: np.ndarray) -> Solution:
    """The same solution with ``sell`` replaced and ``w`` moved to match, which the chain check must catch."""
    return replace(solution, w=spec.w0 - sell, sell=sell)


def test_an_outflow_run_cannot_buy_its_way_back_above_a_floor_and_says_so(make: Factories) -> None:
    resolved = resolved_with(["alpha"], order_flow="outflow")
    too_much_cash = make.spec(cash_lb=-0.2, cash_ub=-0.1)
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 above cash_ub -0\.100000, and an outflow can only raise cash"):
        solve(too_much_cash, ChainState.empty(too_much_cash.security_ids), resolved)
    under_floor = make.spec(lb=np.array([0.0, 0.0, 0.5]), cash_ub=1.0)
    with pytest.raises(InfeasibleError, match=r"names whose floor is above their holding, which this order flow cannot trade out of: \['S2'\]"):
        solve(under_floor, ChainState.empty(under_floor.security_ids), resolved)


def reflect(spec: ProblemSpec) -> ProblemSpec:
    """The book seen in a mirror, ``w' = 1 - w``: a buy in the original is a sell of the same size here, and every constraint and term maps across.

    ``alpha`` is the one term that is not symmetric on its own — a reward for holding becomes a reward
    for not holding — so the mirror carries its negative, and the two objectives then differ by the
    constant ``alpha.sum()``.
    """
    ones = np.ones(spec.n)
    return replace(
        spec,
        w0=ones - spec.w0,
        lb=ones - spec.ub,
        ub=ones - spec.lb,
        columns={**spec.columns, "alpha": -spec.column("alpha")},
        scalars={**spec.scalars, "cash_lb": 2.0 - spec.n - spec.scalar("cash_ub"), "cash_ub": 2.0 - spec.n - spec.scalar("cash_lb")},
    )


def test_an_outflow_run_over_the_reflected_book_is_the_inflow_run_over_the_original(make: Factories) -> None:
    spec = make.spec(
        w0=np.array([0.3, 0.2, 0.1]),
        columns={"alpha": np.array([0.05, 0.04, 0.0])},
        ub=np.array([0.5, 0.35, 0.5]),
        adv_capacity=np.array([0.05, 1.0, 1.0]),
        tcost_per_dollar=np.array([0.001, 0.002, 0.0005]),
        cash_lb=0.0,
        cash_ub=0.2,
    )
    mirrored = reflect(spec)
    chain = ChainState(spec.security_ids, np.array([200.0, 0.0, 0.0]), predecessors=("P0",))  # 200 shares at 100 on 1,000,000 is 0.02 of ADV's 0.05
    terms = ["alpha", TRANSACTION_COST]
    buying = solve(spec, chain, resolved_with(terms, order_flow="inflow"))
    selling = solve(mirrored, chain, resolved_with(terms, order_flow="outflow"))
    assert buying.buy[0] == pytest.approx(0.03, abs=1e-6), "A is bound by what the chain left of its ADV budget, so the reflection exercises the coupling too"
    np.testing.assert_allclose(selling.w, 1.0 - buying.w, atol=1e-6)
    np.testing.assert_allclose(selling.sell, buying.buy, atol=1e-6)
    assert buying.objective is not None and selling.objective is not None
    assert selling.objective == pytest.approx(buying.objective + float(spec.column("alpha").sum()), rel=1e-6), (
        "the mirror's alpha is the negative of the original's, so the two objectives differ by its sum"
    )


def test_a_rebalance_over_the_reflected_book_is_the_mirror_of_the_rebalance_over_the_original(make: Factories) -> None:
    """A rebalance is its own mirror image: the reflection swaps its buys and sells, chain included, and B — sold to its floor here, which an inflow could not do — is bought to its cap there."""
    spec = make.spec(
        w0=np.array([0.3, 0.2, 0.1]),
        columns={"alpha": np.array([0.05, -0.04, 0.0])},
        lb=np.array([0.1, 0.0, 0.0]),
        ub=np.array([0.5, 0.35, 0.5]),
        adv_capacity=np.array([0.05, 1.0, 1.0]),
        tcost_per_dollar=np.array([0.001, 0.002, 0.0005]),
        cash_lb=0.0,
        cash_ub=1.0,
    )
    mirrored = reflect(spec)
    chain = ChainState(spec.security_ids, np.array([200.0, 0.0, 0.0]), predecessors=("P0",))
    terms = ["alpha", TRANSACTION_COST]
    original = solve(spec, chain, resolved_with(terms, order_flow="rebalance"))
    mirror = solve(mirrored, chain, resolved_with(terms, order_flow="rebalance"))
    assert original.buy[0] == pytest.approx(0.03, abs=1e-6), "A is bound by what the chain left of its ADV budget"
    assert original.sell[1] == pytest.approx(0.2, abs=1e-6), "B is sold to its floor in the same solve"
    np.testing.assert_allclose(mirror.w, 1.0 - original.w, atol=1e-6)
    np.testing.assert_allclose(mirror.sell, original.buy, atol=1e-6)
    np.testing.assert_allclose(mirror.buy, original.sell, atol=1e-6)
    assert original.objective is not None and mirror.objective is not None
    assert mirror.objective == pytest.approx(original.objective + float(spec.column("alpha").sum()), rel=1e-6)


def test_infeasible_problem_raises_with_an_arithmetic_diagnosis(make: Factories) -> None:
    spec = make.spec(w0=np.zeros(3), ub=np.array([0.3, 0.3, 0.3]))
    with pytest.raises(InfeasibleError, match=r"upper bounds sum to 0\.900000 < required investment 1\.000000"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"]), CASH)


def test_empty_universe_solves_trivially(make: Factories) -> None:
    spec = make.spec(n=0)
    solution = solve(spec, ChainState.empty(()), resolved_with(["alpha"]))
    assert solution.w.shape == (0,)
    assert solution.status is SolveStatus.OPTIMAL


def test_a_solve_step_that_is_not_an_optimizer_returns_weights_and_no_objective(make: Factories) -> None:
    spec = make.spec()
    chain = ChainState.empty(spec.security_ids)
    resolved = resolved_with(["alpha"], solve="tests.steps:hold_still")
    solution = solve(spec, chain, resolved)
    np.testing.assert_array_equal(solution.w, spec.w0)
    assert solution.buy.tolist() == solution.sell.tolist() == [0.0, 0.0, 0.0]
    assert (solution.objective, solution.iterations) == (None, None)
    assert (solution.solver, solution.solver_version) == ("tests.steps:hold_still", "unknown"), "the step's qualified name, and the version of its distribution — which a test module has none of"
    assert [record["kind"] for record in solution.constraints] == ["cash_limit", "cash_limit", "turnover_limit", "participation_limit"], (
        "the rows are the engine's to record, whatever the step made of them"
    )
    assert solution.duals == {}
    report = verify(spec, solution, chain, resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed and report.objective_passed and report.solver_objective is None
    assert report.objective_terms == (("alpha", pytest.approx(-float((spec.column("alpha") * spec.w0).sum()))),), "the configured terms are still evaluated, as a report line"


def test_a_solve_step_that_ignores_a_constraint_row_cannot_pass_verification(make: Factories) -> None:
    """The rows are what the engine verifies, not what the step says it applied: a step that leaves cash above the cap fails on the cap it never rendered."""
    spec = make.spec(w0=np.full(3, 0.3))  # a tenth of NAV in cash, and the example's cash_cap of zero
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved_with(["alpha"], solve="tests.steps:hold_still"))
    report = verify(spec, solution, chain, (), constraints_of(solution), profile=INFLOW)
    assert report.violated == ("cash_cap/cash_limit",)


def test_a_solve_step_returning_the_wrong_shape_is_a_setup_error(make: Factories) -> None:
    spec = make.spec()
    with pytest.raises(SolveSetupError, match="returned weights of shape"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], solve="tests.steps:wrong_shape"))


def test_a_solver_this_process_cannot_run_is_refused_when_the_config_resolves_not_when_it_solves() -> None:
    with pytest.raises(ConfigResolutionError, match="solve: solver 'NOPE' is not one the adapter knows"):
        resolved_with(["alpha"], solve={"name": "cvxpy", "params": {"solver": "NOPE"}})


def test_an_outflow_run_harvests_a_loss_exactly(make: Factories) -> None:
    """A rewarded sale is a real sale: one variable per name leaves nothing to rebuy with, so the refund the objective books is the refund the orders earn."""
    spec = make.spec(tax_per_dollar=np.array([-0.05, 0.0, 0.0]), cash_ub=1.0)
    resolved = resolved_with(["alpha", "tax_cost"], order_flow="outflow")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    assert solution.sell[0] == pytest.approx(spec.w0[0], abs=1e-6), "the whole loss lot goes: nothing else charges for selling it"
    assert solution.buy.tolist() == [0.0, 0.0, 0.0]
    report = verify(spec, solution, ChainState.empty(spec.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed and report.objective_gap <= 1e-9, "the recomputed objective is the solver's: no round trip inflated it"


def test_tax_and_transaction_costs_discourage_selling_gains(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    universe = frames.three_security_universe().assign(alpha=pd.Series([-0.01, 0.02, 0.05], dtype="Float64"))  # A has turned negative, and is held at a doubling
    details = make.details(max_adv_participation=Decimal("0.25"), cash_ub=Decimal(1))
    spec = standard(make.portfolio_data(holdings=holdings, universe=universe, details=details))
    taxed = solve(spec, ChainState.empty(spec.security_ids), resolved_with(SELL_TERMS, order_flow="outflow"))
    untaxed = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], order_flow="outflow"))
    assert untaxed.sell[0] > 0.0 and taxed.sell[0] == pytest.approx(0.0, abs=1e-6), "A is worth leaving on its alpha alone, but the tax on the gain holds the position"


# --- typed constraints and kinds through the shipped step ---


def test_typed_rows_are_rendered_and_verified_through_their_own_models(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.1, 0.1, 0.1]))
    rows = [typed_row("weight_limit", "cap", direction="<=", bounds="0.4"), *CASH]
    resolved = resolved_with(["alpha"])
    solution = engine_solve(spec, ChainState.empty(spec.security_ids), resolved, constraint_frame(rows))
    np.testing.assert_allclose(solution.w, [0.2, 0.4, 0.4], atol=1e-6, err_msg="alpha wants S2 then S1, the typed cap stops both at 0.4, S0 takes the rest of the cash")
    assert [record["name"] for record in solution.constraints] == ["cap", "cash_floor", "cash_cap"]
    assert solution.constraints[0]["kind"] == "weight_limit"
    report = verify(spec, solution, ChainState.empty(spec.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, report.violated
    assert "cap/weight_limit" in report.active, "the cap binds on S1 and S2, and the report says so"


def test_allow_current_weight_holds_a_breached_start_an_inflow_run_cannot_trade_out_of(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.5, 0.3, 0.2]))
    resolved = resolved_with(["alpha"])
    strict = [typed_row("weight_limit", "cap", direction="<=", bounds="0.4"), *CASH]
    with pytest.raises(InfeasibleError):
        engine_solve(spec, ChainState.empty(spec.security_ids), resolved, constraint_frame(strict))
    held = [typed_row("weight_limit", "cap", direction="<=", bounds="0.4", allow_current_weight=True), *CASH]
    solution = engine_solve(spec, ChainState.empty(spec.security_ids), resolved, constraint_frame(held))
    np.testing.assert_allclose(solution.w, spec.w0, atol=1e-6, err_msg="the breached name is held where it is, not worsened, and the portfolio solves")
    report = verify(spec, solution, ChainState.empty(spec.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, "the residual applies the same start policy, so holding the breach verifies"


def test_a_scoped_participation_limit_binds_its_scope_and_leaves_the_rest_alone(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), flags={"is_thin": np.array([False, False, True])}, adv_capacity=np.array([0.05, 0.05, 0.05]), price=np.full(3, 100.0))
    spent = ChainState(security_ids=spec.security_ids, traded_quantity=np.array([0.0, 0.0, 300.0]), predecessors=("P0",))  # 0.03 of S2's 0.05 budget
    rows = [typed_row("participation_limit", "adv", direction="<=", scope="is_thin"), *CASH]
    solution = engine_solve(spec, spent, resolved_with(["alpha"]), constraint_frame(rows))
    assert solution.buy[2] == pytest.approx(0.02, abs=1e-6), "S2 gets what the predecessor left of its budget"
    assert solution.buy[1] > 0.05, "S1 is outside the scope: its trade is not participation-bound at all, and takes the rest of the cash"


def test_a_bound_read_from_the_accounts_scalars_or_a_spec_column(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.0, 0.1, 0.1]), scalars={"max_single": 0.45}, columns={"cap": np.array([0.2, 0.5, 0.5])})
    from_scalar = engine_solve(
        spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"]), constraint_frame([typed_row("weight_limit", "cap", direction="<=", bounds={"scalar": "max_single"}), *CASH])
    )
    np.testing.assert_allclose(from_scalar.w, [0.1, 0.45, 0.45], atol=1e-6)
    from_column = engine_solve(
        spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"]), constraint_frame([typed_row("weight_limit", "cap", direction="<=", bounds={"column": "cap"}), *CASH])
    )
    np.testing.assert_allclose(from_column.w, [0.0, 0.5, 0.5], atol=1e-6)


def test_a_convex_kind_a_package_registers_solves_and_verifies_like_a_shipped_one(make: Factories) -> None:
    spec = make.spec(w0=np.zeros(3), columns={"variance": np.array([0.01, 0.04, 0.09])})
    resolved = resolved_with([ALPHA, {"kind": "quadratic", "name": "risk", "column": "variance", "weight": "2"}])
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved, CASH)
    assert solution.status is SolveStatus.OPTIMAL
    assert solution.w[0] > 0.0 and solution.w[2] < 1.0, "the penalty spreads the cash across names the linear objective alone would put into S2"
    report = verify(spec, solution, ChainState.empty(spec.security_ids), resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, (report.violated, report.objective_gap)
    assert dict(report.objective_terms)["risk"] == pytest.approx(2.0 * float((spec.column("variance") * solution.w**2).sum()))


def test_rows_in_another_vocabulary_are_refused_by_the_shipped_step(make: Factories) -> None:
    spec = make.spec()
    rows = pd.DataFrame({"portfolio_id": pd.Series(["P1"], dtype="string"), "rule": pd.Series(["max 40%"], dtype="string")})
    with pytest.raises(SolveSetupError, match="carry no `kind` column; the cvxpy step interprets typed rows only"):
        engine_solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"]), rows)
