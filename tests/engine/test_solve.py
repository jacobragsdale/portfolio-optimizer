"""Tier 1: solving — a hand-checkable optimum, a binding cap with a known KKT answer, and every failure path."""

from collections.abc import Sequence
from decimal import Decimal

import numpy as np
import pytest

from portfolio_optimizer.config.resolve import ConfigResolutionError, ResolvedConfig
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, SolveStatus, derive_chain_state
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.solve import InfeasibleError, SolveSetupError, solve
from portfolio_optimizer.engine.tasks import constraint_refs, step_refs
from tests.conftest import Factories, Frames, resolved_example

CORE_CONSTRAINTS = ["long_only", "max_weight", "cash_bounds", "turnover_cap", "sector_bounds", "cumulative_adv_participation"]


def resolved_with(terms: Sequence[object], constraints: Sequence[object], **overrides: object) -> ResolvedConfig:
    return resolved_example(objective={"terms": list(terms)}, constraints=list(constraints), **overrides)


def hand_case(make: Factories, frames: Frames) -> ProblemSpec:
    """P1 holds A 5000 @100 and B 10000 @50 (no gains), targets a third each, may trade a quarter of ADV."""
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    return build_problem_spec(make.portfolio_data(holdings=holdings, style=make.style(max_adv_participation=Decimal("0.25")))).spec


def test_hand_case_matches_the_analytic_optimum(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved_with([{"name": "tracking_error", "params": {"weight": "1"}}], CORE_CONSTRAINTS))
    assert solution.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(solution.w, [0.375, 0.375, 0.25], atol=1e-6)
    np.testing.assert_allclose(solution.sell, [0.125, 0.125, 0.0], atol=1e-6)
    np.testing.assert_allclose(solution.buy, [0.0, 0.0, 0.25], atol=1e-6)
    assert solution.objective == pytest.approx(2 * (0.375 - 1 / 3) ** 2 + (0.25 - 1 / 3) ** 2, abs=1e-6)
    assert solution.solver == "CLARABEL"
    assert solution.spec_hash == spec.content_hash()


def test_solving_twice_is_bitwise_identical(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    resolved = resolved_with(["tracking_error"], CORE_CONSTRAINTS)
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
    solution = solve(spec, spent, resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert solution.buy[2] == pytest.approx(0.0, abs=1e-6), "C's whole ADV budget went to the predecessor"
    fresh = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert fresh.buy[2] == pytest.approx(0.25, abs=1e-6)


def test_a_buy_only_solve_only_buys_and_verifies(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 2500, "avg_cost": Decimal(100)}, {"security_id": "B", "quantity": 5000, "avg_cost": Decimal(50)})
    details = make.details(cash=Decimal(500_000))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, details=details, style=make.style(max_adv_participation=Decimal("0.25")))).spec
    resolved = resolved_with(["tracking_error"], CORE_CONSTRAINTS, sides="buy")
    solution = solve(spec, ChainState.empty(spec.security_ids), resolved)
    np.testing.assert_allclose(solution.w, [0.375, 0.375, 0.25], atol=1e-6)
    assert (solution.w >= spec.w0 - 1e-9).all() and solution.sell.tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(solution.buy, [0.125, 0.125, 0.25], atol=1e-6)
    report = verify(spec, solution, ChainState.empty(spec.security_ids), step_refs(resolved.terms), constraint_refs(resolved.constraints), profile=resolved.profile)
    assert report.passed, report.violated
    assert {check.name for check in report.checks if check.label == "identity"} == {"no_sells", "trade_balance", "nonneg_buy", "sell_absent"}
    spent = ChainState(security_ids=spec.security_ids, traded_shares=np.array([0.0, 0.0, 25_000.0]), predecessors=("P0",))
    np.testing.assert_allclose(solve(spec, spent, resolved).w, [0.5, 0.5, 0.0], atol=1e-6, err_msg="C's whole ADV budget went to the predecessor, so the cash goes to A and B")


def test_a_buy_only_run_cannot_sell_its_way_back_inside_a_cap_and_says_so(make: Factories, frames: Frames) -> None:
    resolved = resolved_with(["tracking_error"], CORE_CONSTRAINTS, sides="buy")
    over_cap = build_problem_spec(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 8000, "avg_cost": Decimal(100)}), style=make.style(max_weight=Decimal("0.6")))).spec
    with pytest.raises(InfeasibleError, match=r"names whose cap is below their holding, which this side cannot trade out of: \['A'\]"):
        solve(over_cap, ChainState.empty(over_cap.security_ids), resolved)
    fully_invested = build_problem_spec(
        make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 10000, "avg_cost": Decimal(100)}), style=make.style(cash_bounds=(Decimal("0.1"), Decimal("0.2"))))
    ).spec
    with pytest.raises(InfeasibleError, match=r"the book starts with cash 0\.000000 below cash_lb 0\.100000, and a buy-only run can only lower cash"):
        solve(fully_invested, ChainState.empty(fully_invested.security_ids), resolved)


def test_infeasible_problem_raises_with_an_arithmetic_diagnosis(make: Factories) -> None:
    spec = make.spec(ub=np.array([0.3, 0.3, 0.3]))
    with pytest.raises(InfeasibleError, match=r"upper bounds sum to 0\.900000 < required investment 1\.000000"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], ["long_only", "max_weight", "cash_bounds"]))


def test_empty_universe_solves_trivially(make: Factories) -> None:
    spec = make.spec(n=0)
    solution = solve(spec, ChainState.empty(()), resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert solution.w.shape == (0,)
    assert solution.status is SolveStatus.OPTIMAL


def test_a_solve_step_that_is_not_an_optimizer_returns_weights_and_no_objective(make: Factories) -> None:
    spec = make.spec()
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved_with(["tracking_error"], CORE_CONSTRAINTS, solve="tests.conftest:hold_still"))
    np.testing.assert_array_equal(solution.w, spec.w0)
    assert solution.buy.tolist() == solution.sell.tolist() == [0.0, 0.0, 0.0]
    assert (solution.objective, solution.iterations, solution.cvxpy_version) == (None, None, "n/a")
    assert (solution.solver, solution.solver_version) == ("tests.conftest:hold_still", "unknown"), "the step's qualified name, and the version of its distribution — which a test module has none of"
    report = verify(spec, solution, chain, step_refs(resolved_with(["tracking_error"], CORE_CONSTRAINTS).terms), [])
    assert report.passed and report.objective_passed and report.solver_objective is None
    assert report.objective_terms == (("portfolio_optimizer.terms:tracking_error", pytest.approx(float(((spec.w0 - spec.w_target) ** 2).sum()))),), (
        "the configured terms are still evaluated, as a report line"
    )


def test_a_solve_step_returning_the_wrong_shape_is_a_setup_error(make: Factories) -> None:
    spec = make.spec()
    with pytest.raises(SolveSetupError, match="returned weights of shape"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], CORE_CONSTRAINTS, solve="tests.conftest:wrong_shape"))


def test_a_solver_this_process_cannot_run_is_refused_when_the_config_resolves_not_when_it_solves() -> None:
    with pytest.raises(ConfigResolutionError, match="solver: solver 'NOPE' is not one the adapter knows"):
        resolved_with(["tracking_error"], CORE_CONSTRAINTS, solver={"name": "NOPE"})


def test_tax_cost_refuses_a_free_loss_harvest(make: Factories) -> None:
    spec = make.spec(tax_per_dollar=np.array([-0.05, 0.0, 0.0]))
    with pytest.raises(ValueError, match="loss-harvest incentive but no transaction cost"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error", "tax_cost"], CORE_CONSTRAINTS))


def test_tax_and_transaction_costs_discourage_selling_gains(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    spec = build_problem_spec(make.portfolio_data(holdings=holdings)).spec
    resolved = resolved_with(
        [{"name": "tracking_error", "params": {"weight": "1"}}, {"name": "tax_cost", "params": {"weight": "1"}}, {"name": "transaction_cost", "params": {"weight": "1", "cost_bps": "10"}}],
        CORE_CONSTRAINTS,
    )
    taxed = solve(spec, ChainState.empty(spec.security_ids), resolved)
    untaxed = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert taxed.sell[0] < untaxed.sell[0]


def test_a_term_that_returns_the_wrong_type_is_rejected(make: Factories) -> None:
    spec = make.spec()
    with pytest.raises(SolveSetupError, match="returned ConstraintSet, expected ObjectiveTerm"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tests.conftest:lying_term"], CORE_CONSTRAINTS))


def test_chain_state_derives_from_prior_orders_for_the_next_solve(make: Factories, frames: Frames) -> None:
    spec = hand_case(make, frames)
    assert derive_chain_state(spec.security_ids, spec.buyable, ()).traded_shares.tolist() == [0.0, 0.0, 0.0]
