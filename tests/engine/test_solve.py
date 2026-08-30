"""Tier 1: solving — a hand-checkable optimum, a binding cap with a known answer, and every failure path."""

import json
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config.resolve import ConfigResolutionError, ResolvedConfig
from portfolio_optimizer.cvx.adapter import WashTradeError
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.domain.sides import TWO_SIDED
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.solve import InfeasibleError, SolveSetupError
from portfolio_optimizer.engine.solve import solve as engine_solve
from portfolio_optimizer.engine.tasks import step_refs
from tests.conftest import BUY_ONLY_TERMS, EXAMPLE_TERMS, SHIPPED_CONSTRAINTS, Factories, Frames, constraint_frame, resolved_example
from tests.engine.support import HAND_OPTIMUM


def resolved_with(terms: Sequence[object], constraints: Sequence[object], **overrides: object) -> ResolvedConfig:
    del constraints  # the set is data now, passed to solve(); kept here so each case still reads as "these terms, those constraints"
    return resolved_example(objective={"terms": list(terms)}, **overrides)


def solve(spec: ProblemSpec, chain: ChainState, resolved: ResolvedConfig, names: Sequence[str] = tuple(SHIPPED_CONSTRAINTS)) -> Solution:
    """Solve with the shipped constraints unless a case names its own; the engine reads them off the frame."""
    return engine_solve(spec, chain, resolved, constraint_frame(names))


def hand_case(make: Factories, frames: Frames) -> ProblemSpec:
    """P1 exactly as the example book has it: A 5000 @100 at no gain, B 10000 @40 at a fifth gain, a 40% name cap, a quarter of ADV."""
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(40)})
    details = make.details(max_weight=Decimal("0.4"), max_adv_participation=Decimal("0.25"))
    return build_problem_spec(make.portfolio_data(holdings=holdings, details=details)).spec


def test_hand_case_matches_the_analytic_optimum(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(EXAMPLE_TERMS, SHIPPED_CONSTRAINTS))
    assert solution.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(solution.w, HAND_OPTIMUM, atol=1e-6)
    np.testing.assert_allclose(solution.sell, [0.15, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(solution.buy, [0.0, 0.0, 0.25], atol=1e-6)
    alpha, tax, cost = 0.03 * 0.35 + 0.01 * 0.4 + 0.05 * 0.25, 0.04 * 0.1, 0.0005 * 0.15 + 0.0005 * 0.1 + 0.002 * 0.25
    assert solution.objective == pytest.approx(-alpha + tax + cost, abs=1e-6)
    assert solution.solver == "CLARABEL"
    assert solution.spec_hash == spec.content_hash()


def test_solving_twice_is_bitwise_identical(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    resolved = resolved_with(EXAMPLE_TERMS, SHIPPED_CONSTRAINTS)
    first = solve(spec, ChainState.empty(spec.security_ids), resolved)
    second = solve(spec, ChainState.empty(spec.security_ids), resolved)
    assert first.w.tobytes() == second.w.tobytes()
    assert first.objective == second.objective


def test_turnover_cap_binds_with_the_known_answer(make: Factories) -> None:
    spec = make.spec(n=2, w0=np.array([1.0, 0.0]), shares_held=np.array([10_000.0, 0.0]), max_turnover=0.2)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], []), ["long_only", "max_weight", "cash_bounds", "turnover_cap"])
    np.testing.assert_allclose(solution.w, [0.9, 0.1], atol=1e-6)
    assert float((solution.buy + solution.sell).sum()) == pytest.approx(0.2, abs=1e-6)


def test_a_portfolio_whose_predecessors_spent_a_names_budget_cannot_buy_it(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(40)})
    details = make.details(max_weight=Decimal("0.6"), max_adv_participation=Decimal("0.25"))  # the example's second account: room to hold A and B alone when C is unbuyable
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, details=details)).spec
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([0.0, 0.0, 25_000.0]), predecessors=("P0",))
    solution = solve(spec, spent, resolved_with(EXAMPLE_TERMS, SHIPPED_CONSTRAINTS))
    assert solution.buy[2] == pytest.approx(0.0, abs=1e-6), "C's whole ADV budget went to the predecessor"
    fresh = solve(spec, ChainState.empty(spec.security_ids), resolved_with(EXAMPLE_TERMS, SHIPPED_CONSTRAINTS))
    assert fresh.buy[2] == pytest.approx(0.25, abs=1e-6)


def test_a_buy_only_solve_only_buys_and_verifies(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 2500, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 5000, "avg_cost": Decimal(50)})
    details = make.details(cash=Decimal(500_000), max_adv_participation=Decimal("0.25"))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, details=details)).spec
    resolved = resolved_with(BUY_ONLY_TERMS, SHIPPED_CONSTRAINTS, sides="buy")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, [0.5, 0.25, 0.25], atol=1e-6, err_msg="C takes its whole ADV budget, the rest of the cash goes to A, the best of what is left")
    assert (solution.w >= spec.w0 - 1e-9).all() and solution.sell.tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(solution.buy, [0.25, 0.0, 0.25], atol=1e-6)
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), solution.constraints, profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_sells", "trade_balance", "nonneg_buy", "sell_absent"}
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([0.0, 0.0, 25_000.0]), predecessors=("P0",))
    np.testing.assert_allclose(solve(spec, spent, resolved).w, [0.75, 0.25, 0.0], atol=1e-6, err_msg="C's whole ADV budget went to the predecessor, so the cash goes to A")


def test_a_buy_only_run_cannot_sell_its_way_back_inside_a_cap_and_says_so(make: Factories, frames: Frames) -> None:
    resolved = resolved_with(BUY_ONLY_TERMS, SHIPPED_CONSTRAINTS, sides="buy")
    over_cap = build_problem_spec(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 8000, "avg_cost": Decimal(100)}), details=make.details(max_weight=Decimal("0.6")))).spec
    with pytest.raises(InfeasibleError, match=r"names whose cap is below their holding, which this side cannot trade out of: \['A'\]"):
        solve(over_cap, ChainState.empty(over_cap.security_ids), resolved)
    fully_invested = build_problem_spec(
        make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 10000, "avg_cost": Decimal(100)}), details=make.details(cash_lb=Decimal("0.1"), cash_ub=Decimal("0.2")))
    ).spec
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 below cash_lb 0\.100000, and a buy-only run can only lower cash"):
        solve(fully_invested, ChainState.empty(fully_invested.security_ids), resolved)


def test_a_sell_only_solve_only_sells_and_couples_through_sells(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    universe = frames.three_security_universe().assign(adv_shares=pd.Series([4000, 1_000_000, 100_000], dtype="Int64"), alpha=pd.Series([-0.03, -0.01, 0.05], dtype="Float64"))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, universe=universe, details=make.details(max_adv_participation=Decimal("0.25"), cash_lb=Decimal(0), cash_ub=Decimal(1)))).spec
    resolved = resolved_with(["alpha"], SHIPPED_CONSTRAINTS, sides="sell")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, [0.4, 0.0, 0.0], atol=1e-6, err_msg="both alphas are negative, so A sells its whole ADV budget (0.1) and B, which has budget to spare, sells out")
    assert (solution.w <= spec.w0 + 1e-9).all() and solution.buy.tolist() == [0.0, 0.0, 0.0]
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), solution.constraints, profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_buys", "trade_balance", "nonneg_sell", "buy_absent"}
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([1000.0, 0.0, 0.0]), predecessors=("P0",))
    np.testing.assert_allclose(solve(spec, spent, resolved).w, [0.5, 0.0, 0.0], atol=1e-6, err_msg="a predecessor's sells of A consumed the budget this portfolio would have sold into")
    report = verify(spec, _with_sell(solve(spec, spent, resolved), spec, np.array([0.05, 0.0, 0.0])), spent, step_refs(resolved.terms), solution.constraints, profile=resolved.profile)
    assert report.violated == ("cumulative_adv_participation",), "a sell past what the chain left of A's budget is caught: the verifier checks the chain against the sells, not the absent buys"


def _with_sell(solution: Solution, spec: ProblemSpec, sell: np.ndarray) -> Solution:
    """The same solution with ``sell`` replaced and ``w`` moved to match, which the chain check must catch."""
    return replace(solution, w=spec.w0 - sell, sell=sell)


def test_a_sell_only_run_cannot_buy_its_way_back_above_a_floor_and_says_so(make: Factories) -> None:
    resolved = resolved_with(["alpha"], SHIPPED_CONSTRAINTS, sides="sell")
    too_much_cash = make.spec(cash_lb=-0.2, cash_ub=-0.1)
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 above cash_ub -0\.100000, and a sell-only run can only raise cash"):
        solve(too_much_cash, ChainState.empty(too_much_cash.security_ids), resolved)
    under_floor = make.spec(lb=np.array([0.0, 0.0, 0.5]), cash_ub=1.0)
    with pytest.raises(InfeasibleError, match=r"names whose floor is above their holding, which this side cannot trade out of: \['S2'\]"):
        solve(under_floor, ChainState.empty(under_floor.security_ids), resolved)


def reflect(spec: ProblemSpec) -> ProblemSpec:
    """The book seen in a mirror, ``w' = 1 - w``: a buy in the original is a sell of the same size here, and every constraint and term maps across.

    ``alpha`` is the one term that is not symmetric on its own — a reward for holding becomes a reward
    for not holding — so the mirror carries its negative, and the two objectives then differ by the
    constant ``alpha.sum()``.
    """
    ones = np.ones(spec.n)
    return ProblemSpec(
        portfolio_id=spec.portfolio_id,
        as_of_date=spec.as_of_date,
        security_ids=spec.security_ids,
        sector_names=spec.sector_names,
        nav=spec.nav,
        w0=ones - spec.w0,
        price=spec.price,
        shares_held=spec.shares_held,
        lot_size=spec.lot_size,
        tax_per_dollar=spec.tax_per_dollar,
        tcost_per_dollar=spec.tcost_per_dollar,
        lb=ones - spec.ub,
        ub=ones - spec.lb,
        adv_capacity=spec.adv_capacity,
        sector_matrix=spec.sector_matrix,
        max_turnover=spec.max_turnover,
        cash_lb=2.0 - spec.n - spec.cash_ub,
        cash_ub=2.0 - spec.n - spec.cash_lb,
        min_trade_notional=spec.min_trade_notional,
        columns={**spec.columns, "alpha": -spec.column("alpha")},
        flags=spec.flags,
    )


def test_a_sell_only_run_over_the_reflected_book_is_the_buy_only_run_over_the_original(make: Factories) -> None:
    spec = make.spec(
        w0=np.array([0.3, 0.2, 0.1]),
        columns={"alpha": np.array([0.05, 0.04, 0.0])},
        ub=np.array([0.5, 0.35, 0.5]),
        adv_capacity=np.array([0.05, 1.0, 1.0]),
        tcost_per_dollar=np.array([0.001, 0.002, 0.0]),
        cash_lb=0.0,
        cash_ub=0.2,
    )
    mirrored = reflect(spec)
    chain = ChainState(spec.security_ids, np.array([200.0, 0.0, 0.0]), predecessors=("P0",))  # 200 shares at 100 on 1,000,000 is 0.02 of ADV's 0.05
    terms = ["alpha", {"name": "transaction_cost", "params": {"cost_bps": "5"}}]
    buying = solve(spec, chain, resolved_with(terms, SHIPPED_CONSTRAINTS, sides="buy"))
    selling = solve(mirrored, chain, resolved_with(terms, SHIPPED_CONSTRAINTS, sides="sell"))
    assert buying.buy[0] == pytest.approx(0.03, abs=1e-6), "A is bound by what the chain left of its ADV budget, so the reflection exercises the coupling too"
    np.testing.assert_allclose(selling.w, 1.0 - buying.w, atol=1e-6)
    np.testing.assert_allclose(selling.sell, buying.buy, atol=1e-6)
    assert buying.objective is not None and selling.objective is not None
    assert selling.objective == pytest.approx(buying.objective + float(spec.column("alpha").sum()), rel=1e-6), (
        "the mirror's alpha is the negative of the original's, so the two objectives differ by its sum"
    )


def test_infeasible_problem_raises_with_an_arithmetic_diagnosis(make: Factories) -> None:
    spec = make.spec(ub=np.array([0.3, 0.3, 0.3]))
    with pytest.raises(InfeasibleError, match=r"upper bounds sum to 0\.900000 < required investment 1\.000000"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], []), ["long_only", "max_weight", "cash_bounds"])


def test_empty_universe_solves_trivially(make: Factories) -> None:
    spec = make.spec(n=0)
    solution = solve(spec, ChainState.empty(()), resolved_with(["alpha"], SHIPPED_CONSTRAINTS))
    assert solution.w.shape == (0,)
    assert solution.status is SolveStatus.OPTIMAL


def test_a_solve_step_that_is_not_an_optimizer_returns_weights_and_no_objective(make: Factories) -> None:
    spec = make.spec()
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved_with(["alpha"], SHIPPED_CONSTRAINTS, solve="tests.steps:hold_still"))
    np.testing.assert_array_equal(solution.w, spec.w0)
    assert solution.buy.tolist() == solution.sell.tolist() == [0.0, 0.0, 0.0]
    assert (solution.objective, solution.iterations) == (None, None)
    assert (solution.solver, solution.solver_version) == ("tests.steps:hold_still", "unknown"), "the step's qualified name, and the version of its distribution — which a test module has none of"
    report = verify(spec, solution, chain, step_refs(resolved_with(["alpha"], SHIPPED_CONSTRAINTS).terms), [], profile=TWO_SIDED)
    assert report.passed and report.objective_passed and report.solver_objective is None
    assert report.objective_terms == (("portfolio_optimizer.terms:alpha", pytest.approx(-float((spec.column("alpha") * spec.w0).sum()))),), "the configured terms are still evaluated, as a report line"


def test_a_solve_step_returning_the_wrong_shape_is_a_setup_error(make: Factories) -> None:
    spec = make.spec()
    with pytest.raises(SolveSetupError, match="returned weights of shape"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], SHIPPED_CONSTRAINTS, solve="tests.steps:wrong_shape"))


def test_a_solver_this_process_cannot_run_is_refused_when_the_config_resolves_not_when_it_solves() -> None:
    with pytest.raises(ConfigResolutionError, match="solver: solver 'NOPE' is not one the adapter knows"):
        resolved_with(["alpha"], SHIPPED_CONSTRAINTS, solver={"name": "NOPE"})


def test_tax_cost_refuses_a_name_whose_loss_pays_for_its_round_trip(make: Factories) -> None:
    spec = make.spec(tax_per_dollar=np.array([-0.05, 0.0, 0.0]), tcost_per_dollar=np.array([0.001, 0.001, 0.001]))
    with pytest.raises(ValueError, match=r"wash trade in 1 name\(s\) held at a loss \(worst offenders \['S0'\]\)"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha", "tax_cost"], SHIPPED_CONSTRAINTS))


def test_tax_cost_allows_a_loss_whose_transaction_costs_exceed_the_tax_saving(make: Factories) -> None:
    spec = make.spec(tax_per_dollar=np.array([-0.001, 0.0, 0.0]), tcost_per_dollar=np.array([0.002, 0.002, 0.002]))
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha", "tax_cost", "transaction_cost"], SHIPPED_CONSTRAINTS))
    assert solution.status is SolveStatus.OPTIMAL


def test_tax_cost_lets_a_sell_only_run_harvest_the_loss(make: Factories) -> None:
    spec = make.spec(tax_per_dollar=np.array([-0.05, 0.0, 0.0]), cash_ub=1.0)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha", "tax_cost"], SHIPPED_CONSTRAINTS, sides="sell"))
    assert solution.sell[0] > 0.0, "selling the loss earns, and a sell-only run has no rebuy to wash against"


def test_a_round_trip_the_objective_rewards_is_refused_and_names_the_trade(make: Factories) -> None:
    """The tax guard reads unweighted per-security costs, so a heavily weighted tax term can still make a round trip pay; the solve step's refusal is the backstop that names it.

    Alpha prefers S0 outright, so the optimum shifts the book into it — and then sells and rebuys the
    S0 it already held, because the weighted tax saving on its loss beats the two transaction costs.
    """
    spec = make.spec(columns={"alpha": np.array([0.0, -0.01, -0.01])}, tax_per_dollar=np.array([-0.001, 0.0, 0.0]), tcost_per_dollar=np.array([0.00075, 0.00075, 0.00075]), max_turnover=4.0)
    terms = ["alpha", {"name": "tax_cost", "params": {"weight": "10"}}, "transaction_cost"]
    with pytest.raises(WashTradeError, match=r"improve the objective by .*worst \['S0'\]"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(terms, SHIPPED_CONSTRAINTS))


def test_tax_and_transaction_costs_discourage_selling_gains(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    universe = frames.three_security_universe().assign(alpha=pd.Series([0.0, 0.02, 0.05], dtype="Float64"))  # A is the name to leave: no expected return, and held at a doubling
    details = make.details(max_adv_participation=Decimal("0.25"))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, universe=universe, details=details)).spec
    resolved = resolved_with(
        [{"name": "alpha", "params": {"weight": "1"}}, {"name": "tax_cost", "params": {"weight": "1"}}, {"name": "transaction_cost", "params": {"weight": "1", "cost_bps": "10"}}], SHIPPED_CONSTRAINTS
    )
    taxed = solve(spec, ChainState.empty(spec.security_ids), resolved)
    untaxed = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], SHIPPED_CONSTRAINTS))
    assert taxed.sell[0] < untaxed.sell[0], "A is held at a gain, so the tax on selling it holds the position where the cap allows"


# --- typed constraints through the shipped step ---


def _typed_rows(*records: dict[str, object]) -> pd.DataFrame:
    """A constraints frame the way the loader delivers it: a ``kind`` column, ``params`` as JSON text; a record without ``kind`` is an opaque function row."""
    frame = pd.DataFrame.from_records([{"portfolio_id": "P1", **record} for record in records])
    if "params" in frame.columns:
        frame["params"] = frame["params"].map(lambda value: json.dumps(value) if isinstance(value, dict) else value)
    return frame


def test_typed_constraints_solve_beside_function_rows_and_verify_through_their_own_residuals(make: Factories) -> None:
    spec = make.spec()
    rows = _typed_rows({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.4"}}, {"name": "long_only"}, {"name": "cash_bounds"})
    resolved = resolved_with(["alpha"], [])
    solution = engine_solve(spec, ChainState.empty(spec.security_ids), resolved, rows)
    np.testing.assert_allclose(solution.w, [0.2, 0.4, 0.4], atol=1e-6, err_msg="alpha wants S2 then S1, the typed cap stops both at 0.4, S0 takes the rest")
    assert [ref.label for ref in solution.constraints] == ["cap", "long_only", "cash_bounds"]
    assert solution.constraints[0].qualname == "portfolio_optimizer.domain.constraints:weight_limit"
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), solution.constraints, profile=TWO_SIDED)
    assert report.passed, report.violated
    assert report.unverified == (), "the typed model is its own twin, so nothing goes unchecked"


def test_allow_current_weight_holds_a_breached_start_a_buy_only_run_cannot_trade_out_of(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.5, 0.3, 0.2]))
    resolved = resolved_with(["alpha"], [], sides="buy")
    strict = _typed_rows({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.4"}}, {"name": "cash_bounds"})
    with pytest.raises(InfeasibleError):
        engine_solve(spec, ChainState.empty(spec.security_ids), resolved, strict)
    held = _typed_rows({"kind": "weight_limit", "label": "cap", "params": {"direction": "<=", "bounds": "0.4", "allow_current_weight": True}}, {"name": "cash_bounds"})
    solution = engine_solve(spec, ChainState.empty(spec.security_ids), resolved, held)
    np.testing.assert_allclose(solution.w, spec.w0, atol=1e-6, err_msg="the breached name is held where it is, not worsened, and the portfolio solves")
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), solution.constraints, profile=resolved.profile)
    assert report.passed, "the residual applies the same start policy, so holding the breach verifies"


def test_a_scoped_participation_limit_binds_its_scope_and_leaves_the_rest_alone(make: Factories) -> None:
    spec = make.spec(flags={"is_thin": np.array([False, False, True])}, adv_capacity=np.array([0.05, 0.05, 0.05]), price=np.full(3, 100.0))
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([0.0, 0.0, 300.0]), predecessors=("P0",))  # 0.03 of S2's 0.05 budget
    rows = _typed_rows({"kind": "participation_limit", "label": "adv", "params": {"direction": "<=", "scope": "is_thin"}}, {"name": "long_only"}, {"name": "cash_bounds"})
    solution = engine_solve(spec, spent, resolved_with(["alpha"], []), rows)
    assert solution.buy[2] == pytest.approx(0.02, abs=1e-6), "S2 gets what the predecessor left of its budget"
    assert solution.buy[1] > 0.05, "S1 is outside the scope: its trade is not participation-bound at all"


def test_a_typed_and_a_function_constraint_may_not_share_a_name(make: Factories) -> None:
    spec = make.spec()
    rows = _typed_rows({"kind": "weight_limit", "label": "max_weight", "params": {"direction": "<=", "bounds": "0.4"}}, {"name": "max_weight"})
    with pytest.raises(SolveSetupError, match="used by both a typed row and a function row"):
        engine_solve(spec, ChainState.empty(spec.security_ids), resolved_with(["alpha"], []), rows)
