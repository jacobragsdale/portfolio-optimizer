"""Tier 1: solving — a hand-checkable optimum, a binding cap with a known KKT answer, and every failure path."""

from collections.abc import Sequence
from decimal import Decimal

import numpy as np
import pytest

from portfolio_optimizer.config.resolve import ConfigResolutionError, ResolvedConfig
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, SolveStatus, derive_chain_state
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.solve import InfeasibleError, SolveSetupError, solve
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
    spent = ChainState(security_ids=spec.security_ids, bought_shares=np.array([0.0, 0.0, 25_000.0]), predecessors=("P0",))
    solution = solve(spec, spent, resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert solution.buy[2] == pytest.approx(0.0, abs=1e-6), "C's whole ADV budget went to the predecessor"
    fresh = solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert fresh.buy[2] == pytest.approx(0.25, abs=1e-6)


def test_infeasible_problem_raises_with_an_arithmetic_diagnosis(make: Factories) -> None:
    spec = make.spec(ub=np.array([0.3, 0.3, 0.3]))
    with pytest.raises(InfeasibleError, match=r"upper bounds sum to 0\.900000 < required investment 1\.000000"):
        solve(spec, ChainState.empty(spec.security_ids), resolved_with(["tracking_error"], ["long_only", "max_weight", "cash_bounds"]))


def test_empty_universe_solves_trivially(make: Factories) -> None:
    spec = make.spec(n=0)
    solution = solve(spec, ChainState.empty(()), resolved_with(["tracking_error"], CORE_CONSTRAINTS))
    assert solution.w.shape == (0,)
    assert solution.status is SolveStatus.OPTIMAL


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
    assert derive_chain_state(spec.security_ids, spec.buyable, ()).bought_shares.tolist() == [0.0, 0.0, 0.0]
