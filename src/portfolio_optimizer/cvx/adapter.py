"""The only module that imports cvxpy.

Terms and constraints are written against the small typed surface here — decision variables,
a handful of DCP atoms, and the result wrappers — so the rest of the engine never touches
cvxpy objects and the post-solve verifier never needs cvxpy at all.
"""

import importlib.metadata
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError

from portfolio_optimizer.domain.results import F64, SolveStatus

type Expr = cp.Expression
type Constraint = cp.Constraint

_SOLVER_PACKAGES: Mapping[str, str] = {"CLARABEL": "clarabel", "OSQP": "osqp", "SCS": "scs", "ECOS": "ecos", "HIGHS": "highspy", "PIQP": "piqp"}
_TIME_LIMIT_OPTION: Mapping[str, str] = {"CLARABEL": "time_limit", "OSQP": "time_limit", "SCS": "time_limit_secs", "HIGHS": "time_limit"}
_STATUS: Mapping[str, SolveStatus] = {
    cp.OPTIMAL: SolveStatus.OPTIMAL,
    cp.OPTIMAL_INACCURATE: SolveStatus.OPTIMAL_INACCURATE,
    cp.INFEASIBLE: SolveStatus.INFEASIBLE,
    cp.INFEASIBLE_INACCURATE: SolveStatus.INFEASIBLE,
    cp.UNBOUNDED: SolveStatus.UNBOUNDED,
    cp.UNBOUNDED_INACCURATE: SolveStatus.UNBOUNDED,
}


@dataclass(frozen=True, slots=True, eq=False)
class DecisionVars:
    """Portfolio weights and the non-negative buy/sell split, all as fractions of NAV."""

    w: cp.Variable
    buy: cp.Variable
    sell: cp.Variable
    n: int


@dataclass(frozen=True, slots=True, eq=False)
class ObjectiveTerm:
    """One named, DCP-convex contribution to the objective."""

    name: str
    expression: Expr


@dataclass(frozen=True, slots=True, eq=False)
class ConstraintSet:
    """A named group of constraints."""

    name: str
    constraints: tuple[Constraint, ...]


def _expr(value: object) -> Expr:
    """Narrow a cvxpy result to ``Expression``; the adapter is where cvxpy's partial typing stops."""
    if not isinstance(value, cp.Expression):
        msg = f"expected a cvxpy Expression, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _constraint(value: object) -> Constraint:
    if not isinstance(value, cp.Constraint):
        msg = f"expected a cvxpy Constraint, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def variables(n: int) -> DecisionVars:
    """Create the decision variables for ``n`` securities."""
    return DecisionVars(w=cp.Variable(n, name="w"), buy=cp.Variable(n, name="buy"), sell=cp.Variable(n, name="sell"), n=n)


def sum_squares(expr: Expr) -> Expr:
    """``‖expr‖²``, convex."""
    return _expr(cp.sum_squares(expr))


def norm1(expr: Expr) -> Expr:
    """``‖expr‖₁``, convex."""
    return _expr(cp.norm1(expr))


def total(expr: Expr) -> Expr:
    """Sum of all entries, affine."""
    return _expr(cp.sum(expr))


def dot(vector: F64, expr: Expr) -> Expr:
    """``vectorᵀ · expr``, affine."""
    return _expr(cp.sum(cp.multiply(vector, expr)))


def matvec(matrix: F64, expr: Expr) -> Expr:
    """``matrix @ expr``, affine."""
    return _expr(cp.matmul(matrix, expr))


def scale(factor: float, expr: Expr) -> Expr:
    """``factor * expr``; a negative factor flips convexity, so callers keep factors non-negative."""
    return _expr(factor * expr)


def minus(left: Expr, right: Expr) -> Expr:
    """``left - right``."""
    return _expr(left - right)


def shifted(expr: Expr, offset: F64) -> Expr:
    """``expr - offset`` for a constant vector."""
    return _expr(expr - offset)


def plus(left: Expr, right: Expr) -> Expr:
    """``left + right``."""
    return _expr(left + right)


def equals(left: Expr, right: Expr) -> Constraint:
    """``left == right``."""
    return _constraint(left == right)


def at_most(expr: Expr, bound: F64 | float) -> Constraint:
    """``expr <= bound``."""
    return _constraint(expr <= bound)


def at_least(expr: Expr, bound: F64 | float) -> Constraint:
    """``expr >= bound``."""
    return _constraint(expr >= bound)


@dataclass(frozen=True, slots=True, eq=False)
class RawSolve:
    """What came back from cvxpy, before the engine decides what it means."""

    status: SolveStatus
    objective: float | None
    w: F64 | None
    buy: F64 | None
    sell: F64 | None
    iterations: int | None
    solve_time_s: float
    solver: str
    solver_version: str
    cvxpy_version: str
    detail: str


class UnavailableSolverError(RuntimeError):
    """The configured solver is not installed in this environment."""


def installed_solvers() -> tuple[str, ...]:
    """Names of the solvers cvxpy can use here."""
    return tuple(str(name) for name in cp.installed_solvers())


def solver_version(solver: str) -> str:
    """Installed version of the package backing ``solver``, or ``"unknown"``."""
    package = _SOLVER_PACKAGES.get(solver)
    if package is None:
        return "unknown"
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def solve_problem(
    x: DecisionVars, terms: Sequence[ObjectiveTerm], constraints: Sequence[ConstraintSet], *, solver: str, options: Mapping[str, float | int | bool | str], time_limit_s: float | None, verbose: bool
) -> RawSolve:
    """Build the cvxpy problem from the given terms and constraints and solve it once."""
    if solver not in installed_solvers():
        msg = f"solver {solver!r} is not installed; available: {list(installed_solvers())}"
        raise UnavailableSolverError(msg)
    if not terms:
        msg = "an objective needs at least one term"
        raise ValueError(msg)
    objective = terms[0].expression
    for term in terms[1:]:
        objective = objective + term.expression
    problem = cp.Problem(cp.Minimize(objective), [constraint for group in constraints for constraint in group.constraints])
    if not problem.is_dcp():
        msg = "the objective and constraints are not DCP-compliant; every term must be convex and every constraint affine or convex"
        raise ValueError(msg)
    kwargs: dict[str, float | int | bool | str] = dict(options)
    if time_limit_s is not None:
        option = _TIME_LIMIT_OPTION.get(solver)
        if option is None:
            msg = f"solver {solver!r} has no known time-limit option; remove solver.time_limit_s or add the option name to the adapter"
            raise ValueError(msg)
        kwargs[option] = time_limit_s
    started = time.perf_counter()
    detail = ""
    try:
        problem.solve(solver=solver, verbose=verbose, **kwargs)
    except SolverError as error:
        detail = str(error)
    elapsed = time.perf_counter() - started
    status = _STATUS.get(str(problem.status), SolveStatus.SOLVER_ERROR) if not detail else SolveStatus.SOLVER_ERROR
    stats = problem.solver_stats
    iterations = None if stats is None or stats.num_iters is None else int(stats.num_iters)
    value = problem.value
    return RawSolve(
        status=status,
        objective=None if value is None or not np.isfinite(float(value)) else float(value),
        w=_value(x.w),
        buy=_value(x.buy),
        sell=_value(x.sell),
        iterations=iterations,
        solve_time_s=elapsed,
        solver=solver,
        solver_version=solver_version(solver),
        cvxpy_version=str(cp.__version__),
        detail=detail or str(problem.status),
    )


def _value(variable: cp.Variable) -> F64 | None:
    value = variable.value
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)
