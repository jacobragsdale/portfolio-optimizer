"""The only module that imports cvxpy.

Terms and constraints are written against the small typed surface here — decision variables,
a handful of DCP atoms, and the result wrappers — so the rest of the engine never touches
cvxpy objects and the post-solve verifier never needs cvxpy at all.
"""

import importlib.metadata
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError
from scipy.sparse import csr_array

from portfolio_optimizer.domain.results import F64, SolveStatus

type Expr = cp.Expression
type Constraint = cp.Constraint


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """What the engine knows about one cvxpy solver: the distribution that versions it and how it spells a time limit.

    A solver cvxpy can see but this table does not name is refused when the config resolves: without
    its distribution the environment fingerprint would record ``unknown`` for its version on every
    process, and two different builds of it would compare equal.
    """

    name: str
    distribution: str
    time_limit_option: str | None


SOLVERS: Mapping[str, SolverSpec] = {
    spec.name: spec
    for spec in (
        SolverSpec("CLARABEL", "clarabel", "time_limit"),
        SolverSpec("OSQP", "osqp", "time_limit"),
        SolverSpec("SCS", "scs", "time_limit_secs"),
        SolverSpec("HIGHS", "highspy", "time_limit"),
        SolverSpec("PIQP", "piqp", None),
    )
}
"""Every solver a config may name. cvxpy installs the first four; ``PIQP`` is the ``piqp`` extra. Adding one is a row here and an extra in ``pyproject.toml``."""

_STATUS: Mapping[str, SolveStatus] = {
    cp.OPTIMAL: SolveStatus.OPTIMAL,
    cp.OPTIMAL_INACCURATE: SolveStatus.OPTIMAL_INACCURATE,
    cp.INFEASIBLE: SolveStatus.INFEASIBLE,
    cp.INFEASIBLE_INACCURATE: SolveStatus.INFEASIBLE,
    cp.UNBOUNDED: SolveStatus.UNBOUNDED,
    cp.UNBOUNDED_INACCURATE: SolveStatus.UNBOUNDED,
}


class SideUnavailableError(LookupError):
    """A term or constraint reached for a decision vector the run's side does not have."""

    def __init__(self, side: str, sides: str) -> None:
        self.side = side
        self.sides = sides
        super().__init__(f"a {sides!r} run has no {side!r} vector; this term or constraint reads x.{side}, so it cannot run under sides={sides!r}")


@dataclass(frozen=True, slots=True, eq=False)
class DecisionVars:
    """The decision variables of one solve, as fractions of NAV; what each is depends on the run's side.

    ``w`` is always a variable, the target weight. ``buy`` and ``sell`` are what the side profile
    made them — a variable each under ``both``, an affine expression of ``w`` on the side a one-sided
    run has, and absent on the side it lacks, where reading them raises :class:`SideUnavailableError`
    (dry construction at ``validate-config`` is where that surfaces). A term that means "the amount
    traded" reads ``trade`` — ``buy + sell``, ``buy``, or ``sell`` — and one that means "the amount
    traded on the side the run couples through" reads ``coupled``; both exist under every side.
    """

    w: cp.Variable
    n: int
    sides: str
    trade: Expr
    coupled: Expr
    _buy: Expr | None = field(default=None, repr=False)
    _sell: Expr | None = field(default=None, repr=False)

    @property
    def buy(self) -> Expr:
        """The non-negative buy, as a fraction of NAV; absent in a sell-only run."""
        if self._buy is None:
            raise SideUnavailableError(side="buy", sides=self.sides)
        return self._buy

    @property
    def sell(self) -> Expr:
        """The non-negative sell, as a fraction of NAV; absent in a buy-only run."""
        if self._sell is None:
            raise SideUnavailableError(side="sell", sides=self.sides)
        return self._sell


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


def variable(n: int, name: str) -> cp.Variable:
    """A fresh vector variable of length ``n``; the side profile decides which ones a solve has."""
    return cp.Variable(n, name=name)


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


def matvec(matrix: F64 | csr_array, expr: Expr) -> Expr:
    """``matrix @ expr``, affine; a sparse matrix is passed through as is."""
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


def shortfall(offset: F64, expr: Expr) -> Expr:
    """``offset - expr`` for a constant vector: how far ``expr`` sits below ``offset``."""
    return _expr(offset - expr)


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
    iterations: int | None
    solve_time_s: float
    solver: str
    solver_version: str
    cvxpy_version: str
    detail: str


class UnavailableSolverError(RuntimeError):
    """The configured solver cannot run in this environment."""


def installed_solvers() -> tuple[str, ...]:
    """Names of the solvers cvxpy can use here."""
    return tuple(str(name) for name in cp.installed_solvers())


def solver_failures(name: str, time_limit_s: float | None, installed: Sequence[str]) -> list[str]:
    """Why solver ``name`` cannot run against ``installed``, if it cannot; empty when it can.

    Unknown to :data:`SOLVERS`, not installed, or asked for a time limit it has no option for. The
    resolver runs this in every process that will solve, so a run fails before any data loads.
    """
    spec = SOLVERS.get(name)
    if spec is None:
        return [f"solver {name!r} is not one the adapter knows; known: {sorted(SOLVERS)}"]
    failures: list[str] = []
    if name not in installed:
        failures.append(f"solver {name!r} is not installed in this environment; installed: {sorted(set(installed) & set(SOLVERS))}")
    if time_limit_s is not None and spec.time_limit_option is None:
        failures.append(f"solver {name!r} has no time-limit option; remove solver.time_limit_s")
    return failures


def solver_version(solver: str) -> str:
    """Installed version of the distribution behind ``solver``, or ``"unknown"``."""
    spec = SOLVERS.get(solver)
    if spec is None:
        return "unknown"
    try:
        return importlib.metadata.version(spec.distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def solve_problem(
    x: DecisionVars, terms: Sequence[ObjectiveTerm], constraints: Sequence[ConstraintSet], *, solver: str, options: Mapping[str, float | int | bool | str], time_limit_s: float | None, verbose: bool
) -> RawSolve:
    """Build the cvxpy problem from the given terms and constraints and solve it once."""
    failures = solver_failures(solver, time_limit_s, installed_solvers())
    if failures:
        raise UnavailableSolverError("; ".join(failures))
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
    option = SOLVERS[solver].time_limit_option
    if time_limit_s is not None and option is not None:
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
