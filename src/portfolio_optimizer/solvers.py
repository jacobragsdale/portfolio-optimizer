"""Shipped solve steps: functions a run's ``solve`` may name.

A solve step takes a :class:`~portfolio_optimizer.solving.SolveRequest` and returns a
:class:`~portfolio_optimizer.solving.SolveResult`. ``cvxpy`` is the default: it builds the problem
from the configured terms and constraints and the side profile's trade identity, and solves it with
the ``solver`` block's solver. ``pro_rata_fill`` is the other shipped step and the shape to copy for
a side that needs no optimizer: a numpy function that reads the spec and the chain and returns
weights, verified afterwards like any solve. A firm's own library that builds the problem its own way
fits the same contract.
"""

import numpy as np

from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm, solve_problem
from portfolio_optimizer.cvx.sides import decision_variables, identity_constraints
from portfolio_optimizer.domain.results import F64, ChainState, ProblemSpec
from portfolio_optimizer.solving import SolveRequest, SolveResult, SolveSetupError
from portfolio_optimizer.terms import adv_remaining


def cvxpy(request: SolveRequest) -> SolveResult:
    """Build the cvxpy problem from the terms, the constraints, and the profile's variables and identity, and solve it with the configured solver."""
    spec, chain, solver = request.spec, request.chain, request.solver
    x = decision_variables(request.profile.sides, spec.w0)
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


def pro_rata_fill(request: SolveRequest) -> SolveResult:
    """Invest the cash above the style's floor into the underweights, pro rata to how far below target each is — no optimizer.

    A name's buy is capped by its upper bound and by the ADV budget left after higher-priority
    portfolios' buys; a cap's excess is spread over the names still open. The verifier checks the
    result like any solve: this step honours bounds, the cash floor, and the ADV budget by
    construction, and sector limits not at all — a book with binding ones is a job for the optimizer.
    """
    spec, chain = request.spec, request.chain
    budget = (1.0 - float(spec.w0.sum())) - spec.cash_lb
    if budget < -CASH_TOLERANCE:
        msg = f"cash is {budget:+.6f} of NAV below the floor {spec.cash_lb:.6f}; a fill only buys, so nothing can be done"
        raise ValueError(msg)
    room = np.maximum(np.minimum(spec.ub - spec.w0, adv_remaining(spec, chain)), 0.0)
    want = np.maximum(spec.w_target - spec.w0, 0.0)
    buy = _water_fill(want, room, max(budget, 0.0))
    return SolveResult(w=spec.w0 + buy, detail=f"invested {float(buy.sum()):.6f} of {max(budget, 0.0):.6f} of NAV across {int((buy > 0).sum())} names")


CASH_TOLERANCE = 1e-9
"""How far below the cash floor a book may start before a fill refuses it: float noise, not policy."""


def _water_fill(want: F64, cap: F64, budget: float) -> F64:
    """Spread ``budget`` over names in proportion to ``want``, never past ``cap``; what a capped name cannot take goes to the rest."""
    allocated = np.zeros_like(want)
    open_names = (want > 0.0) & (cap > 0.0)
    remaining = budget
    while remaining > 1e-15 and open_names.any():
        weights = want[open_names] / want[open_names].sum()
        take = np.minimum(remaining * weights, cap[open_names] - allocated[open_names])
        if take.sum() <= 1e-15:
            break
        allocated[open_names] += take
        remaining -= float(take.sum())
        open_names &= allocated < cap - 1e-15
    return allocated
