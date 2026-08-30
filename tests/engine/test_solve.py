"""Tier 1: solving — a hand-checkable optimum, a binding cap with a known KKT answer, and every failure path."""

from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config.resolve import ConfigResolutionError, ResolvedConfig
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.domain.sides import TWO_SIDED
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.solve import InfeasibleError, SolveSetupError, solve
from portfolio_optimizer.engine.tasks import constraint_refs, step_refs
from tests.conftest import SHIPPED_CONSTRAINTS, Factories, Frames, resolved_example
from tests.engine.support import HAND_OPTIMUM


def resolved_with(terms: Sequence[object], constraints: Sequence[object], **overrides: object) -> ResolvedConfig:
    return resolved_example(objective={"terms": list(terms)}, constraints=list(constraints), **overrides)


def hand_case(make: Factories, frames: Frames) -> ProblemSpec:
    """P1 holds A 5000 @100 and B 10000 @50 (no gains), targets a third each, may trade a quarter of ADV."""
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    return build_problem_spec(make.portfolio_data(holdings=holdings, style=make.style(max_adv_participation=Decimal("0.25")))).spec


def test_hand_case_matches_the_analytic_optimum(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with([{"name": "tracking_error", "params": {"weight": "1"}}], SHIPPED_CONSTRAINTS))
    assert solution.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(solution.w, HAND_OPTIMUM, atol=1e-6)
    np.testing.assert_allclose(solution.sell, [0.125, 0.125, 0.0], atol=1e-6)
    np.testing.assert_allclose(solution.buy, [0.0, 0.0, 0.25], atol=1e-6)
    assert solution.objective == pytest.approx(2 * (0.375 - 1 / 3) ** 2 + (0.25 - 1 / 3) ** 2, abs=1e-6)
    assert solution.solver == "CLARABEL"
    assert solution.spec_hash == spec.content_hash()


def test_solving_twice_is_bitwise_identical(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    resolved = resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS)
    first = solve(spec, ChainState.empty(spec.security_ids), resolved)
    second = solve(spec, ChainState.empty(spec.security_ids), resolved)
    assert first.w.tobytes() == second.w.tobytes()
    assert first.objective == second.objective


def test_turnover_cap_binds_with_the_known_answer(make: Factories) -> None:
    spec = make.spec(n=2, w0=np.array([1.0, 0.0]), w_target=np.array([0.5, 0.5]), shares_held=np.array([10_000.0, 0.0]), max_turnover=0.2)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], ["long_only", "max_weight", "cash_bounds", "turnover_cap"]))
    np.testing.assert_allclose(solution.w, [0.9, 0.1], atol=1e-6)
    assert float((solution.buy + solution.sell).sum()) == pytest.approx(0.2, abs=1e-6)


def test_a_portfolio_whose_predecessors_spent_a_names_budget_cannot_buy_it(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([0.0, 0.0, 25_000.0]), predecessors=("P0",))
    solution = solve(spec, spent, resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS))
    assert solution.buy[2] == pytest.approx(0.0, abs=1e-6), "C's whole ADV budget went to the predecessor"
    fresh = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS))
    assert fresh.buy[2] == pytest.approx(0.25, abs=1e-6)


def test_a_buy_only_solve_only_buys_and_verifies(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 2500, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 5000, "avg_cost": Decimal(50)})
    details = make.details(cash=Decimal(500_000))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, details=details, style=make.style(max_adv_participation=Decimal("0.25")))).spec
    resolved = resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, sides="buy")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, HAND_OPTIMUM, atol=1e-6)
    assert (solution.w >= spec.w0 - 1e-9).all() and solution.sell.tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(solution.buy, [0.125, 0.125, 0.25], atol=1e-6)
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), constraint_refs(resolved.constraints), profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_sells", "trade_balance", "nonneg_buy", "sell_absent"}
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([0.0, 0.0, 25_000.0]), predecessors=("P0",))
    np.testing.assert_allclose(solve(spec, spent, resolved).w, [0.5, 0.5, 0.0], atol=1e-6, err_msg="C's whole ADV budget went to the predecessor, so the cash goes to A and B")


def test_a_buy_only_run_cannot_sell_its_way_back_inside_a_cap_and_says_so(make: Factories, frames: Frames) -> None:
    resolved = resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, sides="buy")
    over_cap = build_problem_spec(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 8000, "avg_cost": Decimal(100)}), style=make.style(max_weight=Decimal("0.6")))).spec
    with pytest.raises(InfeasibleError, match=r"names whose cap is below their holding, which this side cannot trade out of: \['A'\]"):
        solve(over_cap, ChainState.empty(over_cap.security_ids), resolved)
    fully_invested = build_problem_spec(
        make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 10000, "avg_cost": Decimal(100)}), style=make.style(cash_bounds=(Decimal("0.1"), Decimal("0.2"))))
    ).spec
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 below cash_lb 0\.100000, and a buy-only run can only lower cash"):
        solve(fully_invested, ChainState.empty(fully_invested.security_ids), resolved)


def test_a_sell_only_solve_only_sells_and_couples_through_sells(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    universe = frames.three_security_universe().assign(adv_shares=pd.Series([4000, 1_000_000, 100_000], dtype="Int64"))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, universe=universe, style=make.style(max_adv_participation=Decimal("0.25"), cash_bounds=(Decimal(0), Decimal(1))))).spec
    resolved = resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, sides="sell")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, [0.4, 1 / 3, 0.0], atol=1e-6, err_msg="A sells its whole ADV budget (0.1), B to target, C is not held")
    assert (solution.w <= spec.w0 + 1e-9).all() and solution.buy.tolist() == [0.0, 0.0, 0.0]
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), constraint_refs(resolved.constraints), profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_buys", "trade_balance", "nonneg_sell", "buy_absent"}
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([1000.0, 0.0, 0.0]), predecessors=("P0",))
    np.testing.assert_allclose(solve(spec, spent, resolved).w, [0.5, 1 / 3, 0.0], atol=1e-6, err_msg="a predecessor's sells of A consumed the budget this portfolio would have sold into")
    report = verify(spec, _with_sell(solve(spec, spent, resolved), spec, np.array([0.05, 0.0, 0.0])), spent, step_refs(resolved.terms), constraint_refs(resolved.constraints), profile=resolved.profile)
    assert report.violated == ("cumulative_adv_participation",), "a sell past what the chain left of A's budget is caught: the verifier checks the chain against the sells, not the absent buys"


def _with_sell(solution: Solution, spec: ProblemSpec, sell: np.ndarray) -> Solution:
    """The same solution with ``sell`` replaced and ``w`` moved to match, which the chain check must catch."""
    return replace(solution, w=spec.w0 - sell, sell=sell)


def test_a_sell_only_run_cannot_buy_its_way_back_above_a_floor_and_says_so(make: Factories) -> None:
    resolved = resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, sides="sell")
    too_much_cash = make.spec(cash_lb=-0.2, cash_ub=-0.1)
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 above cash_ub -0\.100000, and a sell-only run can only raise cash"):
        solve(too_much_cash, ChainState.empty(too_much_cash.security_ids), resolved)
    under_floor = make.spec(lb=np.array([0.0, 0.0, 0.5]), cash_ub=1.0)
    with pytest.raises(InfeasibleError, match=r"names whose floor is above their holding, which this side cannot trade out of: \['S2'\]"):
        solve(under_floor, ChainState.empty(under_floor.security_ids), resolved)


def reflect(spec: ProblemSpec) -> ProblemSpec:
    """The book seen in a mirror, ``w' = 1 - w``: a buy in the original is a sell of the same size here, and every constraint maps across."""
    ones = np.ones(spec.n)
    rowsum = np.asarray(spec.sector_matrix @ ones, dtype=np.float64)
    return ProblemSpec(
        portfolio_id=spec.portfolio_id,
        as_of=spec.as_of,
        security_ids=spec.security_ids,
        sector_names=spec.sector_names,
        nav=spec.nav,
        w0=ones - spec.w0,
        price=spec.price,
        shares_held=spec.shares_held,
        lot_size=spec.lot_size,
        w_target=ones - spec.w_target,
        tax_per_dollar=spec.tax_per_dollar,
        tcost_per_dollar=spec.tcost_per_dollar,
        lb=ones - spec.ub,
        ub=ones - spec.lb,
        adv_capacity=spec.adv_capacity,
        sector_matrix=spec.sector_matrix,
        sector_lb=rowsum - spec.sector_ub,
        sector_ub=rowsum - spec.sector_lb,
        max_turnover=spec.max_turnover,
        cash_lb=2.0 - spec.n - spec.cash_ub,
        cash_ub=2.0 - spec.n - spec.cash_lb,
        min_trade_notional=spec.min_trade_notional,
        columns=spec.columns,
        flags=spec.flags,
    )


def test_a_sell_only_run_over_the_reflected_book_is_the_buy_only_run_over_the_original(make: Factories) -> None:
    spec = make.spec(
        w0=np.array([0.3, 0.2, 0.1]),
        w_target=np.array([0.45, 0.4, 0.1]),
        ub=np.array([0.5, 0.35, 0.5]),
        adv_capacity=np.array([0.05, 1.0, 1.0]),
        tcost_per_dollar=np.array([0.001, 0.002, 0.0]),
        cash_lb=0.0,
        cash_ub=0.2,
        sector_ub=np.array([0.95]),
    )
    mirrored = reflect(spec)
    chain = ChainState(spec.security_ids, np.array([200.0, 0.0, 0.0]), predecessors=("P0",))  # 200 shares at 100 on 1,000,000 is 0.02 of ADV's 0.05
    buying = solve(spec, chain, resolved_with(["tracking_error", {"name": "transaction_cost", "params": {"cost_bps": "5"}}], SHIPPED_CONSTRAINTS, sides="buy"))
    selling = solve(mirrored, chain, resolved_with(["tracking_error", {"name": "transaction_cost", "params": {"cost_bps": "5"}}], SHIPPED_CONSTRAINTS, sides="sell"))
    assert buying.buy[0] == pytest.approx(0.03, abs=1e-6), "A is bound by what the chain left of its ADV budget, so the reflection exercises the coupling too"
    np.testing.assert_allclose(selling.w, 1.0 - buying.w, atol=1e-6)
    np.testing.assert_allclose(selling.sell, buying.buy, atol=1e-6)
    assert selling.objective == pytest.approx(buying.objective, rel=1e-6)


def test_infeasible_problem_raises_with_an_arithmetic_diagnosis(make: Factories) -> None:
    spec = make.spec(ub=np.array([0.3, 0.3, 0.3]))
    with pytest.raises(InfeasibleError, match=r"upper bounds sum to 0\.900000 < required investment 1\.000000"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], ["long_only", "max_weight", "cash_bounds"]))


def test_empty_universe_solves_trivially(make: Factories) -> None:
    spec = make.spec(n=0)
    solution = solve(spec, ChainState.empty(()), resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS))
    assert solution.w.shape == (0,)
    assert solution.status is SolveStatus.OPTIMAL


def test_a_solve_step_that_is_not_an_optimizer_returns_weights_and_no_objective(make: Factories) -> None:
    spec = make.spec()
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, solve="tests.steps:hold_still"))
    np.testing.assert_array_equal(solution.w, spec.w0)
    assert solution.buy.tolist() == solution.sell.tolist() == [0.0, 0.0, 0.0]
    assert (solution.objective, solution.iterations) == (None, None)
    assert (solution.solver, solution.solver_version) == ("tests.steps:hold_still", "unknown"), "the step's qualified name, and the version of its distribution — which a test module has none of"
    report = verify(spec, solution, chain, step_refs(resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS).terms), [], profile=TWO_SIDED)
    assert report.passed and report.objective_passed and report.solver_objective is None
    assert report.objective_terms == (("portfolio_optimizer.terms:tracking_error", pytest.approx(float(((spec.w0 - spec.w_target) ** 2).sum()))),), (
        "the configured terms are still evaluated, as a report line"
    )


def test_a_solve_step_returning_the_wrong_shape_is_a_setup_error(make: Factories) -> None:
    spec = make.spec()
    with pytest.raises(SolveSetupError, match="returned weights of shape"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, solve="tests.steps:wrong_shape"))


def test_a_solver_this_process_cannot_run_is_refused_when_the_config_resolves_not_when_it_solves() -> None:
    with pytest.raises(ConfigResolutionError, match="solver: solver 'NOPE' is not one the adapter knows"):
        resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS, solver={"name": "NOPE"})


def test_tax_cost_refuses_a_free_loss_harvest(make: Factories) -> None:
    spec = make.spec(tax_per_dollar=np.array([-0.05, 0.0, 0.0]))
    with pytest.raises(ValueError, match="loss-harvest incentive but no transaction cost"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error", "tax_cost"], SHIPPED_CONSTRAINTS))


def test_tax_and_transaction_costs_discourage_selling_gains(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    spec = build_problem_spec(make.portfolio_data(holdings=holdings)).spec
    resolved = resolved_with(
        [{"name": "tracking_error", "params": {"weight": "1"}}, {"name": "tax_cost", "params": {"weight": "1"}}, {"name": "transaction_cost", "params": {"weight": "1", "cost_bps": "10"}}],
        SHIPPED_CONSTRAINTS,
    )
    taxed = solve(spec, ChainState.empty(spec.security_ids), resolved)
    untaxed = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], SHIPPED_CONSTRAINTS))
    assert taxed.sell[0] < untaxed.sell[0]
