"""Tier 1/2: the cvxpy-free verifier catches every perturbation and agrees with the solver on true optima."""

import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from portfolio_optimizer.domain.results import ChainState, ProblemSpec, Solution, SolveStatus, StepRef, Tolerances
from portfolio_optimizer.domain.sides import TWO_SIDED
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import CONSTRAINT_TWINS, TERM_TWINS, verify
from portfolio_optimizer.engine.solve import solve
from portfolio_optimizer.engine.tasks import constraint_refs, step_refs
from tests.conftest import Factories, Frames, resolved_example

CONSTRAINTS = [
    StepRef(qualname=f"portfolio_optimizer.terms:{name}", params={}, label=name) for name in ("long_only", "max_weight", "cash_bounds", "sector_bounds", "turnover_cap", "cumulative_adv_participation")
]
TERMS = [StepRef(qualname="portfolio_optimizer.terms:tracking_error", params={"weight": "1"}, label="tracking_error")]


def rest_solution(spec: ProblemSpec, **overrides: object) -> Solution:
    base: dict[str, object] = {
        "w": spec.w0,
        "buy": np.zeros(spec.n),
        "sell": np.zeros(spec.n),
        "objective": float(((spec.w0 - spec.w_target) ** 2).sum()),
        "status": SolveStatus.OPTIMAL,
        "solver": "X",
        "solver_version": "0",
        "solve_time_s": 0.0,
        "iterations": 1,
        "spec_hash": spec.content_hash(),
    }
    return Solution(**(base | overrides))  # ty: ignore[invalid-argument-type]  # merged mapping; Solution validates on construction


def test_the_resting_portfolio_verifies_when_it_is_feasible(make: Factories) -> None:
    spec = make.spec()
    report = verify(spec, rest_solution(spec), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert report.passed
    assert report.violated == ()
    assert report.unverified == ()
    assert report.objective_gap == 0.0
    assert {name for name, _ in report.objective_terms} == {"portfolio_optimizer.terms:tracking_error"}


Perturbation = Callable[[ProblemSpec, np.ndarray], dict[str, object]]

PERTURBATIONS: list[tuple[str, Perturbation]] = [
    ("trade_balance", lambda _spec, w: {"w": w + np.array([2e-6, 0, 0])}),
    ("nonneg_buy", lambda _spec, _w: {"buy": np.array([-2e-6, 0, 0]), "sell": np.array([-2e-6, 0, 0])}),
    ("sell_le_w0", lambda spec, _w: {"sell": spec.w0 + 2e-6, "buy": np.full(3, 2e-6)}),
    ("complementarity", lambda _spec, _w: {"buy": np.full(3, 1e-3), "sell": np.full(3, 1e-3)}),
    ("long_only", lambda _spec, w: {"w": w - np.array([w[0] + 2e-6, 0, 0]), "sell": np.array([w[0] + 2e-6, 0, 0])}),
    ("max_weight", lambda _spec, w: {"w": np.array([1.0 + 2e-6, 0, 0]), "buy": np.array([1.0 + 2e-6 - w[0], 0, 0]), "sell": np.array([0, w[1], w[2]])}),
    ("cash_ub", lambda _spec, w: {"w": w * 0.999, "sell": w * 0.001}),
    ("cash_lb", lambda _spec, w: {"w": w * 1.001, "buy": w * 0.001}),
    ("turnover_cap", lambda _spec, w: {"w": w, "buy": np.full(3, 0.5), "sell": np.full(3, 0.5)}),
    ("adv_participation", lambda _spec, w: {"w": w, "buy": np.zeros(3), "sell": np.full(3, 11.0)}),
    ("cumulative_adv_participation", lambda _spec, w: {"w": w, "buy": np.full(3, 11.0), "sell": np.zeros(3)}),
]


@pytest.mark.parametrize(("name", "perturb"), PERTURBATIONS, ids=[name for name, _ in PERTURBATIONS])
def test_each_violation_is_detected(make: Factories, name: str, perturb: Perturbation) -> None:
    spec = make.spec()
    solution = rest_solution(spec, **perturb(spec, spec.w0))
    report = verify(spec, solution, ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert name in report.violated, report.violated
    assert report.max_violation > 0


def test_sector_bounds_use_the_configured_tolerance(make: Factories) -> None:
    spec = make.spec(sector_ub=np.array([0.5]))
    tight = verify(spec, rest_solution(spec), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert "sector_ub" in tight.violated
    loose = verify(
        spec,
        rest_solution(spec),
        ChainState.empty(spec.security_ids),
        TERMS,
        [StepRef(qualname="portfolio_optimizer.terms:sector_bounds", params={"tolerance": "0.5"}, label="sector_bounds")],
        profile=TWO_SIDED,
    )
    assert loose.passed


def test_a_violation_exactly_at_tolerance_passes(make: Factories) -> None:
    spec = make.spec()
    solution = rest_solution(spec, w=spec.w0 + np.array([1e-6, 0, 0]))
    report = verify(spec, solution, ChainState.empty(spec.security_ids), [], [CONSTRAINTS[0]], profile=TWO_SIDED, tolerances=Tolerances(violation=1e-6))
    trade_balance = next(check for check in report.checks if check.name == "trade_balance")
    assert trade_balance.label == "identity"
    assert next(check for check in report.checks if check.name == "long_only").label == "long_only"
    assert trade_balance.violation == pytest.approx(1e-6)
    assert trade_balance.passed
    assert trade_balance.worst_security == "S0"


def test_hash_mismatch_and_non_finite_values_fail(make: Factories) -> None:
    spec = make.spec()
    stale = verify(spec, rest_solution(spec, spec_hash="0" * 64), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert "spec_hash_matches" in stale.violated
    broken = verify(spec, rest_solution(spec, w=np.array([np.nan, 0.5, 0.5])), ChainState.empty(spec.security_ids), TERMS, [], profile=TWO_SIDED)
    assert "finite" in broken.violated


def test_objective_gap_is_checked_and_custom_steps_are_reported_unverified(make: Factories) -> None:
    spec = make.spec()
    wrong_objective = verify(spec, rest_solution(spec, objective=0.5), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert not wrong_objective.objective_passed
    custom = verify(
        spec,
        rest_solution(spec, objective=0.5),
        ChainState.empty(spec.security_ids),
        [*TERMS, StepRef(qualname="my_firm.terms:esg", params={}, label="esg")],
        [*CONSTRAINTS, StepRef(qualname="my_firm.terms:beta", params={}, label="beta")],
        profile=TWO_SIDED,
    )
    assert custom.unverified == ("my_firm.terms:beta", "my_firm.terms:esg")
    assert custom.objective_passed  # the total cannot be compared when a term is unknown


def test_every_shipped_term_and_constraint_has_a_twin() -> None:
    from portfolio_optimizer import terms  # imported here so this module's header stays cvxpy-free

    shipped_terms = {f"portfolio_optimizer.terms:{name}" for name in ("tracking_error", "alpha", "tax_cost", "transaction_cost")}
    shipped_constraints = {f"portfolio_optimizer.terms:{name}" for name in ("long_only", "max_weight", "cash_bounds", "sector_bounds", "turnover_cap", "cumulative_adv_participation")}
    assert shipped_terms == set(TERM_TWINS)
    assert shipped_constraints == set(CONSTRAINT_TWINS)
    for qualname in shipped_terms | shipped_constraints:
        assert callable(getattr(terms, qualname.split(":")[1]))


def test_true_optimum_verifies_including_the_objective(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, style=make.style(max_adv_participation=Decimal("0.25")))).spec
    terms = [{"name": "tracking_error", "params": {"weight": "1"}}, {"name": "tax_cost", "params": {"weight": "1"}}, {"name": "transaction_cost", "params": {"weight": "1", "cost_bps": "10"}}]
    resolved = resolved_example(objective={"terms": terms}, constraints=[ref.qualname.split(":")[1] for ref in CONSTRAINTS])
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved)
    refs_terms = step_refs(resolved.terms)
    refs_constraints = constraint_refs(resolved.constraints)
    report = verify(spec, solution, chain, refs_terms, refs_constraints, profile=TWO_SIDED)
    assert report.passed, (report.violated, report.objective_gap)
    assert report.objective_gap <= 1e-9 + 1e-5 * abs(report.recomputed_objective)


def test_verification_works_from_persisted_files(make: Factories, tmp_path: Path) -> None:
    spec = make.spec()
    solution = rest_solution(spec)
    spec.to_npz(tmp_path / "spec.npz")
    solution.to_npz(tmp_path / "solution.npz")
    loaded_spec = ProblemSpec.from_npz(tmp_path / "spec.npz")
    loaded_solution = Solution.from_npz(tmp_path / "solution.npz")
    assert verify(loaded_spec, loaded_solution, ChainState.empty(loaded_spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED).passed


def test_check_module_never_imports_cvxpy() -> None:
    code = "import sys; import portfolio_optimizer.engine.check; assert 'cvxpy' not in sys.modules, 'check imported cvxpy'"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
