"""Tier 1/2: the cvxpy-free verifier catches every perturbation, names what binds, and agrees with the solver on true optima."""

import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from portfolio_optimizer.domain.constraints import ConstraintSpecError, GroupLimit, TypedConstraint, parse_constraints
from portfolio_optimizer.domain.objective import TypedTerm
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, Solution, Tolerances
from portfolio_optimizer.domain.sides import BUY_ONLY
from portfolio_optimizer.engine.build import standard
from portfolio_optimizer.engine.check import constraints_of, verify
from portfolio_optimizer.engine.solve import solve
from tests.conftest import SELL_TERMS, SHIPPED_CONSTRAINTS, Factories, Frames, Row, constraint_frame, resolved_example, terms_of, typed_row


def parsed(*rows: Row) -> tuple[TypedConstraint, ...]:
    """Typed rows as models, the way the build reads them."""
    result = parse_constraints(constraint_frame(list(rows)))
    assert result is not None
    return result.typed


CONSTRAINTS = parsed(*SHIPPED_CONSTRAINTS)
TERMS: tuple[TypedTerm, ...] = terms_of("alpha")


def resting_objective(spec: ProblemSpec) -> float:
    """What ``TERMS`` scores at ``w0``: the objective a solution that trades nothing has to report for the verifier to agree with it."""
    return -float((spec.column("alpha") * spec.w0).sum())


def test_the_resting_portfolio_verifies_when_it_is_feasible(make: Factories) -> None:
    spec = make.spec()
    report = verify(spec, make.solution(spec, objective=resting_objective(spec)), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=BUY_ONLY)
    assert report.passed
    assert report.violated == ()
    assert report.objective_gap == 0.0, "the twin recomputes exactly the alpha the resting book earns"
    assert {name for name, _ in report.objective_terms} == {"alpha"}
    assert report.active == ("cash_floor/cash_limit", "cash_cap/cash_limit"), "a fully invested book sits on both cash bounds, and nothing else binds at rest"


Perturbation = Callable[[ProblemSpec, np.ndarray], dict[str, object]]

PERTURBATIONS: list[tuple[str, Perturbation]] = [
    ("trade_balance", lambda _spec, w: {"w": w + np.array([2e-6, 0, 0])}),
    ("no_sells", lambda _spec, w: {"w": w - np.array([2e-6, 0, 0])}),
    ("nonneg_buy", lambda _spec, _w: {"buy": np.array([-2e-6, 0, 0])}),
    ("sell_absent", lambda _spec, _w: {"buy": np.full(3, 1e-3), "sell": np.full(3, 1e-3)}),
    ("lb", lambda _spec, w: {"w": w - np.array([w[0] + 2e-6, 0, 0]), "sell": np.array([w[0] + 2e-6, 0, 0])}),
    ("ub", lambda _spec, w: {"w": np.array([1.0 + 2e-6, 0, 0]), "buy": np.array([1.0 + 2e-6 - w[0], 0, 0]), "sell": np.array([0, w[1], w[2]])}),
    ("cash_cap/cash_limit", lambda _spec, w: {"w": w * 0.999, "sell": w * 0.001}),
    ("cash_floor/cash_limit", lambda _spec, w: {"w": w * 1.001, "buy": w * 0.001}),
    ("turnover/turnover_limit", lambda _spec, w: {"w": w, "buy": np.full(3, 0.5), "sell": np.full(3, 0.5)}),
    ("adv/participation", lambda _spec, w: {"w": w, "buy": np.zeros(3), "sell": np.full(3, 11.0)}),
    ("adv/cumulative_participation", lambda _spec, w: {"w": w, "buy": np.full(3, 11.0), "sell": np.zeros(3)}),
]


@pytest.mark.parametrize(("name", "perturb"), PERTURBATIONS, ids=[name for name, _ in PERTURBATIONS])
def test_each_violation_is_detected(make: Factories, name: str, perturb: Perturbation) -> None:
    spec = make.spec()
    solution = make.solution(spec, **perturb(spec, spec.w0))
    report = verify(spec, solution, ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=BUY_ONLY)
    assert name in report.violated, report.violated
    assert report.max_violation > 0


def test_a_group_limit_reads_its_band_and_tolerance_from_its_row(make: Factories) -> None:
    spec = make.spec()  # every security in one sector, so its exposure is the whole invested weight
    capped = parsed(typed_row("group_limit", "tech", direction="<=", column="sector", bounds={"TECH": "0.5"}))
    tight = verify(spec, make.solution(spec), ChainState.empty(spec.security_ids), TERMS, capped, profile=BUY_ONLY)
    assert "tech/group_limit" in tight.violated, "the whole book is TECH, so its exposure of 1 is twice the band the row allows"
    loose = parsed(typed_row("group_limit", "tech", direction="<=", column="sector", bounds={"TECH": "0.5"}, tolerance="0.5"))
    assert verify(spec, make.solution(spec, objective=resting_objective(spec)), ChainState.empty(spec.security_ids), TERMS, loose, profile=BUY_ONLY).passed


def test_a_group_limit_naming_a_group_the_universe_lacks_is_refused(make: Factories) -> None:
    spec = make.spec()
    unknown = GroupLimit(name="energy", direction="<=", column="sector", bounds={"ENERGY": Decimal("0.5")})
    with pytest.raises(ConstraintSpecError, match=r"group\(s\) \['ENERGY'\]"):
        verify(spec, make.solution(spec), ChainState.empty(spec.security_ids), TERMS, [unknown], profile=BUY_ONLY)


def test_a_violation_exactly_at_tolerance_passes_and_the_box_reports_where_it_binds(make: Factories) -> None:
    spec = make.spec()
    solution = make.solution(spec, w=spec.w0 + np.array([1e-6, 0, 0]))
    report = verify(spec, solution, ChainState.empty(spec.security_ids), (), (), profile=BUY_ONLY, tolerances=Tolerances(violation=1e-6))
    trade_balance = next(check for check in report.checks if check.name == "trade_balance")
    assert trade_balance.label == "identity" and trade_balance.display == "trade_balance"
    assert trade_balance.violation == pytest.approx(1e-6)
    assert trade_balance.passed
    assert trade_balance.worst_security == "S0"
    all_cash = make.spec(w0=np.zeros(3))
    at_cap = make.solution(all_cash, w=np.array([1.0, 0.0, 0.0]), buy=np.array([1.0, 0.0, 0.0]))
    binding = verify(all_cash, at_cap, ChainState.empty(spec.security_ids), (), (), profile=BUY_ONLY)
    assert binding.passed and binding.active == ("lb", "ub"), "S0 sits on its cap and S1 and S2 on their floor: what a reader asking why the solver stopped wants to see"


def test_hash_mismatch_and_non_finite_values_fail(make: Factories) -> None:
    spec = make.spec()
    stale = verify(spec, make.solution(spec, spec_hash="0" * 64), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=BUY_ONLY)
    assert "spec_hash_matches" in stale.violated
    broken = verify(spec, make.solution(spec, w=np.array([np.nan, 0.5, 0.5])), ChainState.empty(spec.security_ids), TERMS, (), profile=BUY_ONLY)
    assert "finite" in broken.violated


def test_the_objective_gap_is_checked_and_a_registered_kind_is_recomputed_like_a_shipped_one(make: Factories) -> None:
    spec = make.spec(columns={"variance": np.array([0.01, 0.02, 0.03])})
    wrong_objective = verify(spec, make.solution(spec, objective=0.5), ChainState.empty(spec.security_ids), TERMS, CONSTRAINTS, profile=BUY_ONLY)
    assert not wrong_objective.objective_passed
    terms = terms_of("alpha", {"kind": "quadratic", "name": "risk", "column": "variance"})
    report = verify(spec, make.solution(spec, objective=resting_objective(spec) + 0.02 / 3.0), ChainState.empty(spec.security_ids), terms, CONSTRAINTS, profile=BUY_ONLY)
    assert dict(report.objective_terms)["risk"] == pytest.approx(0.02 / 3.0), "the kind's own numpy half: sum of variance times a third squared"
    assert report.objective_passed


def test_true_optimum_verifies_including_the_objective(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(50)}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(60)})
    spec = standard(make.portfolio_data(holdings=holdings, details=make.details(max_adv_participation=Decimal("0.25"), cash_ub=Decimal(1))))
    resolved = resolved_example(sides="sell", objective=SELL_TERMS)
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved, constraint_frame(SHIPPED_CONSTRAINTS))
    report = verify(spec, solution, chain, resolved.terms, constraints_of(solution), profile=resolved.profile)
    assert report.passed, (report.violated, report.objective_gap)
    assert solution.sell[1] > 0.0, "the loss on B is harvested, so the tax term has something to recompute"
    assert report.objective_gap <= 1e-9 + 1e-5 * abs(report.recomputed_objective)


def test_verification_works_from_persisted_files(make: Factories, tmp_path: Path) -> None:
    spec = make.spec()
    solution = make.solution(spec, objective=resting_objective(spec), constraints=tuple(constraint.record() for constraint in CONSTRAINTS), duals={"adv": 0.1})
    spec.to_npz(tmp_path / "spec.npz")
    solution.to_npz(tmp_path / "solution.npz")
    loaded_spec = ProblemSpec.from_npz(tmp_path / "spec.npz")
    loaded_solution = Solution.from_npz(tmp_path / "solution.npz")
    assert constraints_of(loaded_solution) == CONSTRAINTS, "the records a solution carries parse back into the very models"
    assert loaded_solution.duals == {"adv": 0.1}
    assert verify(loaded_spec, loaded_solution, ChainState.empty(loaded_spec.security_ids), TERMS, constraints_of(loaded_solution), profile=BUY_ONLY).passed


def test_check_module_never_imports_cvxpy() -> None:
    code = "import sys; import portfolio_optimizer.engine.check; assert 'cvxpy' not in sys.modules, 'check imported cvxpy'"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
