"""Shipped solve steps, and the constraint convention they read — yours to edit.

A solve step takes a :class:`~portfolio_optimizer.solving.SolveRequest` and returns a
:class:`~portfolio_optimizer.solving.SolveResult`. ``cvxpy`` is the default: it interprets the
portfolio's constraint rows, builds the problem from them, the configured terms, and the side
profile's trade identity, and solves it with the ``solver`` block's solver. ``pro_rata_fill`` is the
other shipped step and the shape to copy for a side that needs no optimizer: a numpy function that
reads the spec and the chain and returns weights, verified afterwards like any solve.

**The constraint convention lives here, not in the engine.** ``request.constraints`` is this
portfolio's rows exactly as the loader returned them and the rules left them; the engine knows only
which portfolio they belong to. :func:`cvxpy` reads the shipped convention — a ``name`` column naming
a step in ``terms.py`` or an importable module, an optional ``label``, and optional ``params`` as JSON
text — and any desk with its own constraint syntax replaces this one function, or writes a step beside
it, without the engine changing at all. A step reports what it made of the rows on
``SolveResult.constraints`` so the verifier can re-check what it recognizes.
"""

import json
from typing import Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from portfolio_optimizer.config.models import STEP_NAME_PATTERN, StepSpec
from portfolio_optimizer.config.resolve import resolve_step
from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm, solve_problem
from portfolio_optimizer.cvx.sides import decision_variables, identity_constraints
from portfolio_optimizer.domain.results import F64, ChainState, ProblemSpec, StepRef
from portfolio_optimizer.domain.types import StrictModel
from portfolio_optimizer.solving import SolveRequest, SolveResult, SolveSetupError
from portfolio_optimizer.terms import adv_remaining

CONSTRAINT_COLUMNS: tuple[str, ...] = ("name", "label", "params")
"""The columns :func:`cvxpy` reads. Only ``name`` is required; the engine reads none of them."""


class ConstraintRow(StrictModel):
    """One constraint row under the shipped convention.

    ``name`` is a step in ``terms.py`` or a qualified ``package.module:function``; ``params`` is the
    JSON object its ``Params`` model validates, written as text because a frame column holds one dtype
    and money must stay exact. ``label`` names the constraint in the verifier's report and the manifest,
    and defaults to the bare name — set it when one function is used twice for one portfolio.
    """

    name: str = Field(pattern=STEP_NAME_PATTERN)
    label: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_params_text(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        params = value.get("params")
        if isinstance(params, str):
            return {**value, "params": json.loads(params) if params.strip() else {}}
        return value

    @model_validator(mode="after")
    def _reject_the_trade_identity(self) -> Self:
        if self.name == "trade_balance":
            msg = "'trade_balance' is not a configurable constraint; the trade identity comes from `sides` — remove the row"
            raise ValueError(msg)
        return self

    @property
    def effective_label(self) -> str:
        """The row's label, or the step's bare name."""
        return self.label if self.label is not None else self.name.rpartition(":")[2]


def cvxpy(request: SolveRequest) -> SolveResult:
    """Interpret the constraint rows, build the cvxpy problem with the terms and the profile's identity, and solve it."""
    spec, chain, solver = request.spec, request.chain, request.solver
    x = decision_variables(request.profile.sides, spec.w0)
    terms = [_term(step, x, spec, chain) for step in request.terms]
    interpreted = interpret_constraints(request.constraints)
    constraints = [identity_constraints(request.profile.sides, x, spec.w0), *(_constraint_set(step, x, spec, chain) for step, _ in interpreted)]
    result = solve_problem(x, terms, constraints, solver=solver.name, options=solver.options, time_limit_s=solver.time_limit_s, verbose=solver.verbose)
    return replace_constraints(result, tuple(ref for _, ref in interpreted))


def interpret_constraints(frame: pd.DataFrame) -> tuple[tuple[ResolvedStep, StepRef], ...]:
    """Read the shipped convention off one portfolio's constraint rows: a resolved step and its ref for each.

    Rows are taken in the order the frame carries them. A malformed row, an unimportable name, or params
    the function's own model refuses is raised here, which the engine records as this portfolio's failure
    at stage ``solve`` — one account, not the book.
    """
    if frame.empty:
        return ()
    missing = [column for column in ("name",) if column not in frame.columns]
    if missing:
        msg = f"constraints frame is missing column(s) {missing}; the shipped convention reads {list(CONSTRAINT_COLUMNS)}"
        raise SolveSetupError(msg)
    interpreted: list[tuple[ResolvedStep, StepRef]] = []
    labels: dict[str, int] = {}
    for position, record in enumerate(frame.to_dict("records")):
        row = _row(position, {column: record[column] for column in CONSTRAINT_COLUMNS if column in record})
        if row.effective_label in labels:
            msg = f"constraints[{position}]: label {row.effective_label!r} is also used by constraints[{labels[row.effective_label]}]; give one of them a label"
            raise SolveSetupError(msg)
        labels[row.effective_label] = position
        step = _resolve(position, row)
        interpreted.append((step, StepRef(qualname=step.qualname, params=step.params_json, label=row.effective_label)))
    return tuple(interpreted)


def replace_constraints(result: SolveResult, refs: tuple[StepRef, ...]) -> SolveResult:
    """Record on the result what the step made of the constraint rows, for the verifier and the manifest."""
    return SolveResult(
        w=result.w,
        status=result.status,
        objective=result.objective,
        iterations=result.iterations,
        solve_time_s=result.solve_time_s,
        solver=result.solver,
        solver_version=result.solver_version,
        detail=result.detail,
        constraints=refs,
    )


def _row(position: int, record: dict[str, object]) -> ConstraintRow:
    try:
        return ConstraintRow.model_validate({key: value for key, value in record.items() if not _is_null(value)})
    except ValueError as error:
        msg = f"constraints[{position}]: {error}"
        raise SolveSetupError(msg) from error


def _resolve(position: int, row: ConstraintRow) -> ResolvedStep:
    try:
        return resolve_step(StepSpec(name=row.name, params=row.params), "constraint")
    except ValueError as error:
        msg = f"constraints[{position}]: {error}"
        raise SolveSetupError(msg) from error


def _is_null(value: object) -> bool:
    """True for the several ways a missing optional column arrives from a frame: ``None``, ``pd.NA``, or a float ``NaN``."""
    return value is None or value is pd.NA or (isinstance(value, float) and bool(np.isnan(value)))


def _term(step: ResolvedStep, x: object, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
    result = step.invoke(x=x, spec=spec, chain=chain)
    if not isinstance(result, ObjectiveTerm):
        msg = f"term {step.qualname!r} returned {type(result).__name__}, expected ObjectiveTerm"
        raise SolveSetupError(msg)
    return result


def _constraint_set(step: ResolvedStep, x: object, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:
    result = step.invoke(x=x, spec=spec, chain=chain)
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
