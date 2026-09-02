"""Shipped solve steps — yours to edit.

A solve step takes a :class:`~portfolio_optimizer.solving.SolveRequest` and returns a
:class:`~portfolio_optimizer.solving.SolveResult`. ``cvxpy`` is the default: it renders the
portfolio's typed constraint rows and the configured typed terms through their own ``to_cvxpy``,
adds the order-flow profile's identity, and solves with the solver its ``params`` name. ``pro_rata_fill``
is the other shipped step and the shape to copy for a side that needs no optimizer: a numpy
function that reads the spec and the chain and returns weights, verified afterwards like any solve.

A desk with its own constraint vocabulary writes a step beside these that reads its own rows; the
engine hands every step the same request and verifies every answer the same way.
"""

import contextlib

import numpy as np
from pydantic import Field

from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm, solve_problem
from portfolio_optimizer.cvx.order_flow import decision_variables, identity_constraints
from portfolio_optimizer.domain.constraints import adv_remaining, parse_constraints
from portfolio_optimizer.domain.results import F64, MissingSpecColumnError
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.solving import DEFAULT_SOLVER, SolveRequest, SolveResult, SolveSetupError


class CvxpyParams(Params):
    """Which cvxpy solver runs and with what options.

    ``solver`` must be one the adapter knows and cvxpy has installed — `CLARABEL`, `OSQP`, `SCS`,
    `HIGHS` (installed with cvxpy) or `PIQP` (the `piqp` extra) — checked when the config resolves,
    by `validate-config`, at the start of `run`, and on every worker before it does any work; there is
    no automatic fallback.
    """

    solver: str = Field(default=DEFAULT_SOLVER, min_length=1, description="The cvxpy solver: `CLARABEL` (default), `OSQP`, `SCS`, `HIGHS`, or `PIQP`.")
    options: dict[str, float | int | bool | str] = Field(default_factory=dict, description='Passed verbatim to `Problem.solve(**options)`, e.g. `{"max_iter": 200}`.')
    time_limit_s: float | None = Field(
        default=None,
        gt=0,
        description="Wall-clock limit per solve in seconds, translated to the solver's own option; omit for no limit. Rejected at resolve for a solver with no such option (`PIQP`).",
    )
    verbose: bool = Field(default=False, description="Let the solver print its iteration log.")


def cvxpy(request: SolveRequest, params: CvxpyParams) -> SolveResult:
    """Render the typed terms and constraint rows, add the profile's identity, and solve the cvxpy problem.

    The rows must speak the typed vocabulary — a ``kind`` column naming a model — since this step
    interprets nothing else; a desk's own syntax is a step of its own. What was applied is reported
    back as constraint records, and the verifier re-checks each through its model's own residual.
    """
    spec, chain = request.spec, request.chain
    parsed = parse_constraints(request.constraints)
    if parsed is None and not request.constraints.empty:
        msg = f"the constraint rows carry no `kind` column; the cvxpy step interprets typed rows only, and these {len(request.constraints)} row(s) are in another vocabulary"
        raise SolveSetupError(msg)
    typed = parsed.typed if parsed is not None else ()
    x = decision_variables(request.profile.order_flow, spec)
    terms = [_term(term.to_cvxpy(x, spec, chain), term.name) for term in request.terms]
    constraints = [identity_constraints(request.profile.order_flow, x, spec), *(_constraint_set(model.to_cvxpy(x, spec, chain), model.name) for model in typed)]
    result = solve_problem(x, terms, constraints, solver=params.solver, options=params.options, time_limit_s=params.time_limit_s, verbose=params.verbose)
    return SolveResult(
        w=result.w,
        status=result.status,
        objective=result.objective,
        iterations=result.iterations,
        solve_time_s=result.solve_time_s,
        solver=result.solver,
        solver_version=result.solver_version,
        detail=result.detail,
        constraints=tuple(model.record() for model in typed),
        duals=result.duals,
    )


def _term(result: object, name: str) -> ObjectiveTerm:
    if not isinstance(result, ObjectiveTerm):
        msg = f"term {name!r} rendered {type(result).__name__}, expected ObjectiveTerm"
        raise SolveSetupError(msg)
    return result


def _constraint_set(result: object, name: str) -> ConstraintSet:
    if not isinstance(result, ConstraintSet):
        msg = f"constraint {name!r} rendered {type(result).__name__}, expected ConstraintSet"
        raise SolveSetupError(msg)
    return result


def pro_rata_fill(request: SolveRequest) -> SolveResult:
    """Spread the cash above the style's floor evenly over the names the portfolio may buy — no optimizer, and no view on which name is better.

    A name's room is the smaller of what its upper bound allows and what is left of its ADV budget
    after higher-priority portfolios' buys (every name, where the universe carries no ``adv_shares``);
    a name that fills up passes its share to the rest. The verifier checks the result like any solve:
    this step honours bounds, the cash floor, and the ADV budget by construction, and sector limits
    not at all — a book with binding ones is a job for the optimizer.
    """
    spec, chain = request.spec, request.chain
    floor = spec.scalars.get("cash_lb", 0.0)
    budget = (1.0 - float(spec.w0.sum())) - floor
    if budget < -CASH_TOLERANCE:
        msg = f"cash is {budget:+.6f} of NAV below the floor {floor:.6f}; a fill only buys, so nothing can be done"
        raise ValueError(msg)
    room = spec.ub - spec.w0
    with contextlib.suppress(MissingSpecColumnError):  # no ADV column, no participation budget: the bounds alone limit the fill
        room = np.minimum(room, adv_remaining(spec, chain))
    room = np.maximum(room, 0.0)
    buy = _water_fill(np.where(room > 0.0, 1.0, 0.0), room, max(budget, 0.0))
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
