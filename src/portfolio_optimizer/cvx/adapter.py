"""The only module that imports cvxpy.

Terms and constraints are written against the small typed surface here — decision variables,
a handful of DCP atoms, and the result wrappers — so the rest of the engine never touches
cvxpy objects and the post-solve verifier never needs cvxpy at all. The atoms are affine or
convex; a term that scales a convex atom by a negative weight is not DCP, which the dry
construction at ``validate-config`` and the solve itself both refuse.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError
from scipy.sparse import csr_array

from portfolio_optimizer.domain.constraints import Vector
from portfolio_optimizer.domain.order_flow import OrderFlow
from portfolio_optimizer.domain.results import F64, Flags, SolveStatus
from portfolio_optimizer.solving import SOLVERS, SolveResult, solver_failures, solver_version

type Expr = cp.Expression
type Constraint = cp.Constraint


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

    def __init__(self, side: str, order_flow: OrderFlow) -> None:
        self.side = side
        self.order_flow = order_flow
        super().__init__(f"order flow {order_flow!r} has no {side!r} vector; this term or constraint reads x.{side}, so it cannot run under order_flow={order_flow!r}")


@dataclass(frozen=True, slots=True, eq=False)
class DecisionVars:
    """The decision variables of one solve, as fractions of NAV; what each is depends on the run's order flow.

    ``w`` is always a variable, the target weight. ``buy`` or ``sell`` is what the order-flow profile made
    it — an affine expression of ``w`` on the side the run has — and absent on the side it lacks,
    where reading it raises :class:`SideUnavailableError` (dry construction at ``validate-config`` is
    where that surfaces). A term that means "the amount traded" reads ``trade``, and one that means
    "the amount traded on the side the run couples through" reads ``coupled``; both exist under
    every order flow, and under either they are the one side's vector.
    """

    w: cp.Variable
    n: int
    order_flow: OrderFlow
    trade: Expr
    coupled: Expr
    _buy: Expr | None = field(default=None, repr=False)
    _sell: Expr | None = field(default=None, repr=False)

    @property
    def buy(self) -> Expr:
        """The non-negative buy, as a fraction of NAV; absent in an outflow."""
        if self._buy is None:
            raise SideUnavailableError(side="buy", order_flow=self.order_flow)
        return self._buy

    @property
    def sell(self) -> Expr:
        """The non-negative sell, as a fraction of NAV; absent in an inflow."""
        if self._sell is None:
            raise SideUnavailableError(side="sell", order_flow=self.order_flow)
        return self._sell

    def vector(self, name: Vector) -> Expr:
        """The decision quantity a typed term or constraint names: ``w``, ``buy``, ``sell``, or ``trade``."""
        if name == "w":
            return self.w
        if name == "buy":
            return self.buy
        if name == "sell":
            return self.sell
        return self.trade


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
    """A fresh vector variable of length ``n``; the order-flow profile decides which ones a solve has."""
    return cp.Variable(n, name=name)


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
    """``factor * expr``; a negative factor on a convex atom is not DCP, which the dry construction refuses."""
    return _expr(factor * expr)


def weighted(vector: F64, expr: Expr) -> Expr:
    """``vector ∘ expr`` elementwise for a constant vector, affine; ``sum_squares(weighted(√d, w))`` is a diagonal penalty."""
    return _expr(cp.multiply(vector, expr))


def masked(flags: Flags, expr: Expr) -> Expr:
    """``expr`` where ``flags`` is set and zero elsewhere — how a scoped constraint touches only its scope; affine."""
    return _expr(cp.multiply(flags.astype(np.float64), expr))


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


def sum_squares(expr: Expr) -> Expr:
    """``Σ exprᵢ²``, convex: a diagonal risk penalty is ``sum_squares(multiply(√d, w))``, a factor one ``sum_squares(matvec(F½B, w))``."""
    return _expr(cp.sum_squares(expr))


def norm1(expr: Expr) -> Expr:
    """``Σ |exprᵢ|``, convex: the shape of a tracking penalty against a target, ``norm1(shifted(w, target))``."""
    return _expr(cp.norm1(expr))


def absolute(expr: Expr) -> Expr:
    """``|expr|`` elementwise, convex."""
    return _expr(cp.abs(expr))


def pos(expr: Expr) -> Expr:
    """``max(expr, 0)`` elementwise, convex: the shape of a one-sided penalty."""
    return _expr(cp.pos(expr))


def equals(left: Expr, right: Expr) -> Constraint:
    """``left == right``."""
    return _constraint(left == right)


def at_most(expr: Expr, bound: F64 | float) -> Constraint:
    """``expr <= bound``."""
    return _constraint(expr <= bound)


def at_least(expr: Expr, bound: F64 | float) -> Constraint:
    """``expr >= bound``."""
    return _constraint(expr >= bound)


class UnavailableSolverError(RuntimeError):
    """The configured solver cannot run in this environment."""


def installed_solvers() -> tuple[str, ...]:
    """Names of the solvers cvxpy can use here."""
    return tuple(str(name) for name in cp.installed_solvers())


def build_problem(terms: Sequence[ObjectiveTerm], constraints: Sequence[ConstraintSet]) -> cp.Problem:
    """The cvxpy problem: minimize the terms' sum subject to every constraint set; refused when it is not DCP."""
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
    return problem


def solve_problem(
    x: DecisionVars, terms: Sequence[ObjectiveTerm], constraints: Sequence[ConstraintSet], *, solver: str, options: Mapping[str, float | int | bool | str], time_limit_s: float | None, verbose: bool
) -> SolveResult:
    """Build the cvxpy problem from the given terms and constraints and solve it once; what comes back is the solve step's result as is.

    The result carries, per constraint set, the largest dual value the solver reported: the shadow
    price of each limit, zero where it did not bind.
    """
    failures = solver_failures(solver, time_limit_s, installed_solvers())
    if failures:
        raise UnavailableSolverError("; ".join(failures))
    problem = build_problem(terms, constraints)
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
    duals: dict[str, float] = {}
    if status in (SolveStatus.OPTIMAL, SolveStatus.OPTIMAL_INACCURATE):
        duals = {group.name: _largest_dual(group) for group in constraints}
    stats = problem.solver_stats
    iterations = None if stats is None or stats.num_iters is None else int(stats.num_iters)
    value = problem.value
    return SolveResult(
        w=_value(x.w),
        status=status,
        objective=None if value is None or not np.isfinite(float(value)) else float(value),
        iterations=iterations,
        solve_time_s=elapsed,
        solver=solver,
        solver_version=solver_version(solver),
        detail=detail or str(problem.status),
        duals=duals,
    )


def _largest_dual(group: ConstraintSet) -> float:
    """The largest dual value across a set's constraints — how hard the limit bound; zero when the solver reported none."""
    largest = 0.0
    for constraint in group.constraints:
        dual = constraint.dual_value
        if dual is not None:
            largest = max(largest, float(np.abs(np.asarray(dual, dtype=np.float64)).max(initial=0.0)))
    return largest


def _value(variable: cp.Variable) -> F64 | None:
    value = variable.value
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)
