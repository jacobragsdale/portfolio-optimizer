"""Independent, cvxpy-free re-verification of a solution against its spec.

Every typed constraint row the portfolio carries and every configured term is a typed model with its
own numpy half — ``residual`` for a constraint, ``value`` for a term — so an auditor can re-run this
over a persisted spec and solution without the solver stack, and a kind a package ships is checked
exactly like a shipped one. The rows are the engine's reading, stamped on the solution at solve, so a
solve step cannot narrow what is checked by leaving a row out. The same check runs twice per
portfolio: over the solved weights, and over the weights the rounded orders leave the book at.

This module must never import cvxpy; a test enforces it.
"""

from collections.abc import Sequence

import numpy as np

from portfolio_optimizer.domain.constraints import TypedConstraint, parse_constraint
from portfolio_optimizer.domain.objective import TypedTerm
from portfolio_optimizer.domain.order_flow import OrderFlowProfile
from portfolio_optimizer.domain.results import F64, ChainState, ConstraintCheck, ConstraintReport, ProblemSpec, Solution, Tolerances

DEFAULT_TOLERANCES = Tolerances()

SOLUTION_LABEL = "solution"
"""Label of the checks on the solution itself: finiteness and the spec hash."""
IDENTITY_LABEL = "identity"
"""Label of the order-flow profile's trade-identity and box checks."""

BOX: frozenset[str] = frozenset({"lb", "ub"})
"""The identity checks that can bind in a way worth reporting: the spec's own bounds. The rest hold as equalities or at zero on every name, so "binding" means nothing for them."""


def constraints_of(solution: Solution) -> tuple[TypedConstraint, ...]:
    """The typed constraint rows the engine stamped on a solution, parsed back into their models through the registry."""
    return tuple(parse_constraint(record, f"constraints[{index}]") for index, record in enumerate(solution.constraints))


def verify(
    spec: ProblemSpec,
    solution: Solution,
    chain: ChainState,
    terms: Sequence[TypedTerm],
    constraints: Sequence[TypedConstraint],
    *,
    profile: OrderFlowProfile,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> ConstraintReport:
    """Recompute the trade identity, every constraint's violation, and the objective, and compare with the solver.

    ``profile`` is the side the run traded: it supplies the identity checks and the coupled quantity
    the residuals read. A check is *active* when its residual sits within the tolerance of its bound —
    the constraint bound, or was breached — which is what says where the answer stopped.
    """
    checks: list[ConstraintCheck] = []
    checks.append(_check("finite", 0.0 if all(np.isfinite(a).all() for a in (solution.w, solution.buy, solution.sell)) else float("inf"), tolerances.violation, None, SOLUTION_LABEL))
    checks.append(_check("spec_hash_matches", 0.0 if solution.spec_hash == spec.content_hash() else float("inf"), 0.0, None, SOLUTION_LABEL))
    checks.extend(_residual_check(name, residual, spec, tolerances, IDENTITY_LABEL, binding=name in BOX) for name, residual in profile.identity_residuals(spec, solution))
    for constraint in constraints:
        checks.extend(_residual_check(name, residual, spec, tolerances, constraint.name, binding=True) for name, residual in constraint.residual(spec, solution, chain, profile))
    objective_terms = tuple((term.name, term.value(spec, solution, chain, profile)) for term in terms)
    recomputed = float(sum(value for _, value in objective_terms))
    comparable = solution.objective is not None  # a step that minimized nothing has no objective to agree with
    gap = abs(recomputed - solution.objective) if solution.objective is not None else 0.0
    objective_passed = (gap <= tolerances.obj_abs + tolerances.obj_rel * abs(recomputed)) if comparable else True
    return ConstraintReport(
        checks=tuple(checks), objective_terms=objective_terms, recomputed_objective=recomputed, solver_objective=solution.objective, objective_gap=gap, objective_passed=objective_passed
    )


def _residual_check(name: str, residual: F64, spec: ProblemSpec, tolerances: Tolerances, label: str, *, binding: bool) -> ConstraintCheck:
    """The worst entry of a residual vector against the violation tolerance, naming the security it sits at when the residual is per security."""
    worst = int(np.argmax(residual)) if residual.size and residual.size == spec.n else None
    largest = float(residual.max()) if residual.size else float("-inf")
    active = binding and largest >= -tolerances.violation
    return _check(name, max(largest, 0.0), tolerances.violation, spec.security_ids[worst] if worst is not None else None, label, active=active, residual=largest if residual.size else None)


def _check(name: str, violation: float, tolerance: float, worst: str | None, label: str, *, active: bool = False, residual: float | None = None) -> ConstraintCheck:
    return ConstraintCheck(name=name, violation=violation, tolerance=tolerance, passed=violation <= tolerance, worst_security=worst, label=label, active=active, residual=residual)
