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

DEFAULT_TOLERANCES = Tolerances()

type ConstraintTwin = Callable[[ProblemSpec, Solution, ChainState, Mapping[str, object]], list[tuple[str, F64]]]
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


def _trade_balance(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del chain, params
    return [
        ("trade_balance", np.abs(sol.w - spec.w0 - sol.buy + sol.sell)),
        ("nonneg_buy", -sol.buy),
        ("nonneg_sell", -sol.sell),
        ("sell_le_w0", sol.sell - spec.w0),
        ("complementarity", np.minimum(sol.buy, sol.sell)),
    ]


def _long_only(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del chain, params
    return [("long_only", spec.lb - sol.w)]


def _max_weight(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del chain, params
    return [("max_weight", sol.w - spec.ub)]


def _cash_bounds(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del chain, params
    cash = 1.0 - float(sol.w.sum())
    return [("cash_lb", np.array([spec.cash_lb - cash])), ("cash_ub", np.array([cash - spec.cash_ub]))]


def _sector_bounds(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del chain
    if len(spec.sector_names) == 0:
        return []
    tolerance = param(params, "tolerance", 0.0)
    exposure = spec.sector_matrix @ sol.w
    return [("sector_lb", spec.sector_lb - tolerance - exposure), ("sector_ub", exposure - spec.sector_ub - tolerance)]


def _turnover_cap(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del chain, params
    return [("turnover_cap", np.array([float((sol.buy + sol.sell).sum()) - spec.max_turnover]))]


def _adv_participation(spec: ProblemSpec, sol: Solution, chain: ChainState, params: Mapping[str, object]) -> list[tuple[str, F64]]:
    del params
    consumed = chain.cumulative_shares * spec.price / spec.nav
    remaining = np.maximum(0.0, spec.adv_capacity - consumed)
    return [("cumulative_adv_participation", sol.buy + sol.sell - remaining)]


def _tracking_error(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    return param(params, "weight", 1.0) * float(((sol.w - spec.w_target) ** 2).sum())


def _alpha(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    column = params.get("column", "alpha")
    return -param(params, "weight", 1.0) * float((spec.column(str(column)) * sol.w).sum())


def _tax_cost(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    return param(params, "weight", 1.0) * float((spec.tax_per_dollar * sol.sell).sum())


def _transaction_cost(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    cost = spec.tcost_per_dollar + param(params, "cost_bps", 0.0) / 10_000.0
    return param(params, "weight", 1.0) * float((cost * (sol.buy + sol.sell)).sum())


CONSTRAINT_TWINS: Mapping[str, ConstraintTwin] = {
    "portfolio_optimizer.terms:trade_balance": _trade_balance,
    "portfolio_optimizer.terms:long_only": _long_only,
    "portfolio_optimizer.terms:max_weight": _max_weight,
    "portfolio_optimizer.terms:cash_bounds": _cash_bounds,
    "portfolio_optimizer.terms:sector_bounds": _sector_bounds,
    "portfolio_optimizer.terms:turnover_cap": _turnover_cap,
    "portfolio_optimizer.terms:cumulative_adv_participation": _adv_participation,
}
"""Numpy twin of every shipped constraint, keyed by the qualified name the manifest records."""

TERM_TWINS: Mapping[str, TermTwin] = {
    "portfolio_optimizer.terms:tracking_error": _tracking_error,
    "portfolio_optimizer.terms:alpha": _alpha,
    "portfolio_optimizer.terms:tax_cost": _tax_cost,
    "portfolio_optimizer.terms:transaction_cost": _transaction_cost,
}
"""Numpy twin of every shipped objective term."""


def verify(spec: ProblemSpec, solution: Solution, chain: ChainState, terms: Sequence[StepRef], constraints: Sequence[StepRef], tolerances: Tolerances = DEFAULT_TOLERANCES) -> ConstraintReport:
    """Recompute every verifiable constraint's violation and the objective, and compare with the solver."""
    checks: list[ConstraintCheck] = []
    unverified: list[str] = []
    checks.append(_check("finite", 0.0 if all(np.isfinite(a).all() for a in (solution.w, solution.buy, solution.sell)) else float("inf"), tolerances.eq, None))
    checks.append(_check("spec_hash_matches", 0.0 if solution.spec_hash == spec.content_hash() else float("inf"), 0.0, None))
    for ref in constraints:
        twin = CONSTRAINT_TWINS.get(ref.qualname)
        if twin is None:
            unverified.append(ref.qualname)
            continue
        for name, residual in twin(spec, solution, chain, ref.params):
            tolerance = tolerances.eq if name == "trade_balance" else tolerances.ineq
            worst = int(np.argmax(residual)) if residual.size and residual.size == spec.n else None
            checks.append(_check(name, float(residual.max(initial=0.0)), tolerance, spec.security_ids[worst] if worst is not None else None))
    objective_terms: list[tuple[str, float]] = []
    for ref in terms:
        twin_term = TERM_TWINS.get(ref.qualname)
        if twin_term is None:
            unverified.append(ref.qualname)
            continue
        objective_terms.append((ref.qualname, twin_term(spec, solution, ref.params)))
    recomputed = float(sum(value for _, value in objective_terms))
    all_terms_verified = all(ref.qualname in TERM_TWINS for ref in terms)
    gap = abs(recomputed - solution.objective) if all_terms_verified else 0.0
    objective_passed = (gap <= tolerances.obj_abs + tolerances.obj_rel * abs(recomputed)) if all_terms_verified else True
    return ConstraintReport(
        checks=tuple(checks),
        objective_terms=tuple(objective_terms),
        recomputed_objective=recomputed,
        solver_objective=solution.objective,
        objective_gap=gap,
        objective_passed=objective_passed,
        unverified=tuple(unverified),
    )


def _check(name: str, violation: float, tolerance: float, worst: str | None) -> ConstraintCheck:
    return ConstraintCheck(name=name, violation=violation, tolerance=tolerance, passed=violation <= tolerance, worst_security=worst)
