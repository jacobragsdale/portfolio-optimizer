"""Tier 1/2: the cvxpy-free verifier catches every perturbation and agrees with the solver on true optima."""

import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from portfolio_optimizer.domain.results import ChainState, MissingSpecColumnError, ProblemSpec, Solution, StepRef, Tolerances
from portfolio_optimizer.domain.sides import TWO_SIDED
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import CONSTRAINT_TWINS, TERM_TWINS, verify
from portfolio_optimizer.engine.solve import solve
from portfolio_optimizer.engine.tasks import step_refs
from tests.conftest import SHIPPED_CONSTRAINTS, Factories, Frames, constraint_frame, resolved_example, step_refs_for

CONSTRAINTS = step_refs_for(SHIPPED_CONSTRAINTS)
TERMS = [StepRef(qualname="portfolio_optimizer.terms:alpha", params={"weight": "1"}, label="alpha")]


def resting_objective(spec: ProblemSpec) -> float:
    """What ``TERMS`` scores at ``w0``: the objective a solution that trades nothing has to report for the verifier to agree with it."""
    return -float((spec.column("alpha") * spec.w0).sum())


def test_the_resting_portfolio_verifies_when_it_is_feasible(make: Factories) -> None:
    spec = make.spec()
    report = verify(spec, make.solution(spec, objective=resting_objective(spec)), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert report.passed
    assert report.violated == ()
    assert report.unverified == ()
    assert report.objective_gap == 0.0, "the twin recomputes exactly the alpha the resting book earns"
    assert {name for name, _ in report.objective_terms} == {"portfolio_optimizer.terms:alpha"}


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
    solution = make.solution(spec, **perturb(spec, spec.w0))
    report = verify(spec, solution, ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert name in report.violated, report.violated
    assert report.max_violation > 0


def test_a_sector_bound_reads_its_band_and_tolerance_from_the_row(make: Factories) -> None:
    spec = make.spec(sector_names=("TECH",))  # every security in one sector, so its exposure is the whole invested weight
    capped = StepRef(qualname="portfolio_optimizer.terms:sector_bound", params={"sector": "TECH", "upper": "0.5"}, label="tech")
    tight = verify(spec, make.solution(spec), ChainState.empty(spec.security_ids), TERMS, [capped], profile=TWO_SIDED)
    assert "sector_ub" in tight.violated, "the whole book is TECH, so its exposure of 1 is twice the band the row allows"
    loose = verify(
        spec,
        make.solution(spec, objective=resting_objective(spec)),
        ChainState.empty(spec.security_ids),
        TERMS,
        [StepRef(qualname="portfolio_optimizer.terms:sector_bound", params={"sector": "TECH", "upper": "0.5", "tolerance": "0.5"}, label="tech")],
        profile=TWO_SIDED,
    )
    assert loose.passed


def test_a_sector_bound_naming_a_sector_the_universe_lacks_is_refused(make: Factories) -> None:
    spec = make.spec(sector_names=("TECH",))
    unknown = StepRef(qualname="portfolio_optimizer.terms:sector_bound", params={"sector": "ENERGY"}, label="energy")
    with pytest.raises(MissingSpecColumnError, match=r"spec has no sector 'ENERGY'"):
        verify(spec, make.solution(spec), ChainState.empty(spec.security_ids), TERMS, [unknown], profile=TWO_SIDED)


def test_a_violation_exactly_at_tolerance_passes(make: Factories) -> None:
    spec = make.spec()
    solution = make.solution(spec, w=spec.w0 + np.array([1e-6, 0, 0]))
    report = verify(spec, solution, ChainState.empty(spec.security_ids), [], step_refs_for(["long_only"]), profile=TWO_SIDED, tolerances=Tolerances(violation=1e-6))
    trade_balance = next(check for check in report.checks if check.name == "trade_balance")
    assert trade_balance.label == "identity"
    assert next(check for check in report.checks if check.name == "long_only").label == "long_only"
    assert trade_balance.violation == pytest.approx(1e-6)
    assert trade_balance.passed
    assert trade_balance.worst_security == "S0"


def test_hash_mismatch_and_non_finite_values_fail(make: Factories) -> None:
    spec = make.spec()
    stale = verify(spec, make.solution(spec, spec_hash="0" * 64), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert "spec_hash_matches" in stale.violated
    broken = verify(spec, make.solution(spec, w=np.array([np.nan, 0.5, 0.5])), ChainState.empty(spec.security_ids), TERMS, [], profile=TWO_SIDED)
    assert "finite" in broken.violated


def test_objective_gap_is_checked_and_custom_steps_are_reported_unverified(make: Factories) -> None:
    spec = make.spec()
    wrong_objective = verify(spec, make.solution(spec, objective=0.5), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED)
    assert not wrong_objective.objective_passed
    custom = verify(
        spec,
        make.solution(spec, objective=0.5),
        ChainState.empty(spec.security_ids),
        [*TERMS, StepRef(qualname="my_firm.terms:esg", params={}, label="esg")],
        [*CONSTRAINTS, StepRef(qualname="my_firm.terms:beta", params={}, label="beta")],
        profile=TWO_SIDED,
    )
    assert custom.unverified == ("my_firm.terms:beta", "my_firm.terms:esg")
    assert custom.objective_passed  # the total cannot be compared when a term is unknown


def test_every_shipped_term_and_constraint_has_a_twin() -> None:
    from portfolio_optimizer import terms  # imported here so this module's header stays cvxpy-free

    shipped_terms = {f"portfolio_optimizer.terms:{name}" for name in ("alpha", "tax_cost", "transaction_cost")}
    shipped_constraints = {f"portfolio_optimizer.terms:{name}" for name in (*SHIPPED_CONSTRAINTS, "sector_bound")}
    typed_kinds = {f"portfolio_optimizer.domain.constraints:{kind}" for kind in ("group_limit", "exposure_limit", "weight_limit", "participation_limit")}
    assert shipped_terms == set(TERM_TWINS)
    assert shipped_constraints | typed_kinds == set(CONSTRAINT_TWINS)
    for qualname in shipped_terms | shipped_constraints:
        assert callable(getattr(terms, qualname.split(":")[1]))
    assert len({CONSTRAINT_TWINS[qualname] for qualname in typed_kinds}) == 1, "every typed kind shares one twin: the model's own residual"


def test_true_optimum_verifies_including_the_objective(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(50)})
    spec = build_problem_spec(make.portfolio_data(holdings=holdings, details=make.details(max_adv_participation=Decimal("0.25")))).spec
    terms = [{"name": "alpha", "params": {"weight": "1"}}, {"name": "tax_cost", "params": {"weight": "1"}}, {"name": "transaction_cost", "params": {"weight": "1", "cost_bps": "10"}}]
    resolved = resolved_example(objective={"terms": terms})
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved, constraint_frame(SHIPPED_CONSTRAINTS))
    report = verify(spec, solution, chain, step_refs(resolved.terms), solution.constraints, profile=TWO_SIDED)
    assert report.passed, (report.violated, report.objective_gap)
    assert report.objective_gap <= 1e-9 + 1e-5 * abs(report.recomputed_objective)


def test_verification_works_from_persisted_files(make: Factories, tmp_path: Path) -> None:
    spec = make.spec()
    solution = make.solution(spec, objective=resting_objective(spec))
    spec.to_npz(tmp_path / "spec.npz")
    solution.to_npz(tmp_path / "solution.npz")
    loaded_spec = ProblemSpec.from_npz(tmp_path / "spec.npz")
    loaded_solution = Solution.from_npz(tmp_path / "solution.npz")
    assert verify(loaded_spec, loaded_solution, ChainState.empty(loaded_spec.security_ids), TERMS, CONSTRAINTS, profile=TWO_SIDED).passed


def test_check_module_never_imports_cvxpy() -> None:
    code = "import sys; import portfolio_optimizer.engine.check; assert 'cvxpy' not in sys.modules, 'check imported cvxpy'"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
