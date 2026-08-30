"""Shipped solve steps: functions a run's ``solve`` may name.

A solve step takes a :class:`~portfolio_optimizer.solving.SolveRequest` and returns a
:class:`~portfolio_optimizer.solving.SolveResult`. ``cvxpy`` is the default and the only one shipped:
it builds the problem from the configured terms and constraints and the side profile's trade identity,
and solves it with the ``solver`` block's solver. Write your own to plug in a library that builds the
problem its own way, or a plain numpy function for a side that needs no optimizer at all.
"""

from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm, solve_problem, variables
from portfolio_optimizer.cvx.sides import identity_constraints
from portfolio_optimizer.domain.results import ChainState, ProblemSpec
from portfolio_optimizer.solving import SolveRequest, SolveResult, SolveSetupError


def cvxpy(request: SolveRequest) -> SolveResult:
    """Build the cvxpy problem from the terms, the constraints, and the profile's identity, and solve it with the configured solver."""
    spec, chain, solver = request.spec, request.chain, request.solver
    x = variables(spec.n)
    terms = [_term(step, x, spec, chain) for step in request.terms]
    constraints = [identity_constraints(request.profile.sides, x, spec.w0), *(_constraint_set(constraint.step, x, spec, chain) for constraint in request.constraints)]
    raw = solve_problem(x, terms, constraints, solver=solver.name, options=solver.options, time_limit_s=solver.time_limit_s, verbose=solver.verbose)
    return SolveResult(
        w=raw.w,
        status=raw.status,
        objective=raw.objective,
        iterations=raw.iterations,
        solve_time_s=raw.solve_time_s,
        solver=raw.solver,
        solver_version=raw.solver_version,
        cvxpy_version=raw.cvxpy_version,
        detail=raw.detail,
    )


def _term(step: ResolvedStep, x: object, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
    result = step.invoke(x=x, spec=spec, context=chain if step.needs_context else None)
    if not isinstance(result, ObjectiveTerm):
        msg = f"term {step.qualname!r} returned {type(result).__name__}, expected ObjectiveTerm"
        raise SolveSetupError(msg)
    return result


def _constraint_set(step: ResolvedStep, x: object, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:
    result = step.invoke(x=x, spec=spec, context=chain if step.needs_context else None)
    if not isinstance(result, ConstraintSet):
        msg = f"constraint {step.qualname!r} returned {type(result).__name__}, expected ConstraintSet"
        raise SolveSetupError(msg)
    return result
