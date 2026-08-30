"""Independent, cvxpy-free re-verification of a solution against its spec.

Every shipped constraint and objective term has a numpy twin here, keyed by the step's qualified
name, so an auditor can re-run this over a persisted spec and solution without the solver stack.
Custom steps are reported as unverified rather than silently trusted.

This module must never import cvxpy; a test enforces it.
"""

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal

import numpy as np

from portfolio_optimizer.domain.results import F64, ChainState, ConstraintCheck, ConstraintReport, ProblemSpec, Solution, StepRef, Tolerances
from portfolio_optimizer.domain.sides import SideProfile

DEFAULT_TOLERANCES = Tolerances()

SOLUTION_LABEL = "solution"
"""Label of the checks on the solution itself: finiteness and the spec hash."""
IDENTITY_LABEL = "identity"
"""Label of the side profile's trade-identity checks."""

type ConstraintTwin = Callable[[ProblemSpec, Solution, ChainState, Mapping[str, object], SideProfile], list[tuple[str, F64]]]
"""A shipped constraint's numpy twin: the spec, the solution in place of ``x``, the chain, the params, and the profile that made ``x``."""
type TermTwin = Callable[[ProblemSpec, Solution, Mapping[str, object]], float]


def param(params: Mapping[str, object], name: str, default: float) -> float:
    """Read a numeric parameter that may have arrived as Decimal, str, int, or float."""
    value = params.get(name, default)
    if isinstance(value, bool):
        msg = f"parameter {name!r} is a bool"
        raise TypeError(msg)
    if isinstance(value, int | float | Decimal):
        return float(value)
    if isinstance(value, str):
        return float(Decimal(value))
    msg = f"parameter {name!r} has unsupported type {type(value).__name__}"
    raise TypeError(msg)


def _long_only(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object], profile: SideProfile) -> list[tuple[str, F64]]:
    del chain, params, profile
    return [("long_only", spec.lb - sol.w)]


def _max_weight(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object], profile: SideProfile) -> list[tuple[str, F64]]:
    del chain, params, profile
    return [("max_weight", sol.w - spec.ub)]


def _cash_bounds(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object], profile: SideProfile) -> list[tuple[str, F64]]:
    del chain, params, profile
    cash = 1.0 - float(sol.w.sum())
    return [("cash_lb", np.array([spec.cash_lb - cash])), ("cash_ub", np.array([cash - spec.cash_ub]))]


def _sector_bound(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object], profile: SideProfile) -> list[tuple[str, F64]]:
    del chain, profile
    sector = params.get("sector")
    if not isinstance(sector, str):
        msg = f"sector_bound needs a 'sector' parameter naming one of {list(spec.sector_names)}, got {sector!r}"
        raise TypeError(msg)
    tolerance = param(params, "tolerance", 0.0)
    exposure = np.asarray(spec.sector(sector) @ sol.w, dtype=np.float64)
    return [("sector_lb", param(params, "lower", 0.0) - tolerance - exposure), ("sector_ub", exposure - param(params, "upper", 1.0) - tolerance)]


def _turnover_cap(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object], profile: SideProfile) -> list[tuple[str, F64]]:
    del chain, params, profile
    return [("turnover_cap", np.array([float((sol.buy + sol.sell).sum()) - spec.max_turnover]))]


def _adv_participation(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object], profile: SideProfile) -> list[tuple[str, F64]]:
    del params
    consumed = chain.traded_shares * spec.price / spec.nav
    remaining = np.maximum(0.0, spec.adv_capacity - consumed)
    return [("adv_participation", sol.buy + sol.sell - spec.adv_capacity), ("cumulative_adv_participation", profile.coupled(sol) - remaining)]


def _alpha(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    column = params.get("column", "alpha")
    return -param(params, "weight", 1.0) * float((spec.column(str(column)) * sol.w).sum())


def _tax_cost(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    return param(params, "weight", 1.0) * float((spec.tax_per_dollar * sol.sell).sum())


def _transaction_cost(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    cost = spec.tcost_per_dollar + param(params, "cost_bps", 0.0) / 10_000.0
    return param(params, "weight", 1.0) * float((cost * (sol.buy + sol.sell)).sum())


CONSTRAINT_TWINS: Mapping[str, ConstraintTwin] = {
    "portfolio_optimizer.terms:long_only": _long_only,
    "portfolio_optimizer.terms:max_weight": _max_weight,
    "portfolio_optimizer.terms:cash_bounds": _cash_bounds,
    "portfolio_optimizer.terms:sector_bound": _sector_bound,
    "portfolio_optimizer.terms:turnover_cap": _turnover_cap,
    "portfolio_optimizer.terms:cumulative_adv_participation": _adv_participation,
}
"""Numpy twin of every shipped constraint, keyed by the qualified name the manifest records."""

TERM_TWINS: Mapping[str, TermTwin] = {"portfolio_optimizer.terms:alpha": _alpha, "portfolio_optimizer.terms:tax_cost": _tax_cost, "portfolio_optimizer.terms:transaction_cost": _transaction_cost}
"""Numpy twin of every shipped objective term."""


def verify(
    spec: ProblemSpec, solution: Solution, chain: ChainState, terms: Sequence[StepRef], constraints: Sequence[StepRef], *, profile: SideProfile, tolerances: Tolerances = DEFAULT_TOLERANCES
) -> ConstraintReport:
    """Recompute the trade identity, every verifiable constraint's violation, and the objective, and compare with the solver.

    ``profile`` is the side the run traded: it supplies the identity checks and the coupled quantity the twins read.
    """
    checks: list[ConstraintCheck] = []
    unverified: list[str] = []
    checks.append(_check("finite", 0.0 if all(np.isfinite(a).all() for a in (solution.w, solution.buy, solution.sell)) else float("inf"), tolerances.violation, None, SOLUTION_LABEL))
    checks.append(_check("spec_hash_matches", 0.0 if solution.spec_hash == spec.content_hash() else float("inf"), 0.0, None, SOLUTION_LABEL))
    checks.extend(_residual_check(name, residual, spec, tolerances, IDENTITY_LABEL) for name, residual in profile.identity_residuals(spec, solution))
    for ref in constraints:
        twin = CONSTRAINT_TWINS.get(ref.qualname)
        if twin is None:
            unverified.append(ref.qualname)
            continue
        checks.extend(_residual_check(name, residual, spec, tolerances, ref.label) for name, residual in twin(spec, solution, chain, ref.params, profile))
    objective_terms: list[tuple[str, float]] = []
    for ref in terms:
        twin_term = TERM_TWINS.get(ref.qualname)
        if twin_term is None:
            unverified.append(ref.qualname)
            continue
        objective_terms.append((ref.qualname, twin_term(spec, solution, ref.params)))
    recomputed = float(sum(value for _, value in objective_terms))
    comparable = solution.objective is not None and all(ref.qualname in TERM_TWINS for ref in terms)  # a step that minimized nothing has no objective to agree with
    gap = abs(recomputed - solution.objective) if comparable and solution.objective is not None else 0.0
    objective_passed = (gap <= tolerances.obj_abs + tolerances.obj_rel * abs(recomputed)) if comparable else True
    return ConstraintReport(
        checks=tuple(checks),
        objective_terms=tuple(objective_terms),
        recomputed_objective=recomputed,
        solver_objective=solution.objective,
        objective_gap=gap,
        objective_passed=objective_passed,
        unverified=tuple(unverified),
    )


def _residual_check(name: str, residual: F64, spec: ProblemSpec, tolerances: Tolerances, label: str) -> ConstraintCheck:
    """The worst entry of a residual vector against the violation tolerance, naming the security it sits at when the residual is per security."""
    worst = int(np.argmax(residual)) if residual.size and residual.size == spec.n else None
    return _check(name, float(residual.max(initial=0.0)), tolerances.violation, spec.security_ids[worst] if worst is not None else None, label)


def _check(name: str, violation: float, tolerance: float, worst: str | None, label: str) -> ConstraintCheck:
    return ConstraintCheck(name=name, violation=violation, tolerance=tolerance, passed=violation <= tolerance, worst_security=worst, label=label)
