"""Typed constraints: the declaration the engine reads, the numbers the solve step renders, and the residual the verifier checks.

A constraint row used to be opaque to the engine — a function name only the solve step could
interpret — so the dependency graph had to assume the widest coupling: any two portfolios sharing a
tradable name might affect each other. A *typed* constraint is a strict, hashable pydantic model the
engine can read without understanding the mathematics: its ``kind`` says whether it reads the chain,
and its ``scope`` says which securities it can couple through. That is all the schedule needs, and it
is what lets a portfolio whose constraints read no chain solve with no predecessors, and a scoped
participation cap couple through its scope alone.

**The typed spec is optional, at every level.** A run may declare no constraints dataset, a portfolio
may have no rows, and a desk may write rows in its own vocabulary that this module does not know —
:func:`parse_constraints` returns ``None`` when the frame does not speak this spec at all, and counts
any non-typed row as *opaque*. Opaque rows keep exactly today's behaviour: the engine does not
interpret them, and they couple the portfolio through its whole tradable set, because an opaque
reader cannot declare a narrower scope. The same conservatism applies one level up: only the shipped
``cvxpy`` solve step confines chain access to the configured terms and constraints, so under any
other solve step — which receives ``request.chain`` and may read it however it likes — the coupling
stays the full tradable set whatever the rows say.

The contract is behavioural, three-sided:

- **The engine** reads the declaration: :func:`parse_constraints` at build time,
  :func:`consumed_securities` for the graph. It never looks inside for a matrix.
- **The solve step** renders the model however it likes; the shipped cvxpy step's renderers live in
  ``solvers.py``. A step that cannot render a kind refuses it there.
- **The verifier** re-checks every typed constraint through :meth:`TypedConstraint.residual` — plain
  numpy, this module never imports cvxpy — so ``verify`` still runs without the solver stack.

Rows keep the loaded-data convention: a ``kind`` column selects the typed model and ``params`` holds
its fields as JSON; the row's ``label`` (or ``name``) column may carry the constraint's name.

``allow_current_weight`` is the start policy, per constraint: a book that already breaches the bound
(a name over its cap, a sector over its band) either loosens the bound to the current value — hold
it, do not worsen it — or fails the portfolio as infeasible. The loosening applies to ``w``-shaped
constraints only; a bound on ``buy``, ``sell``, or ``trade`` starts at zero and cannot be breached.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from typing import Literal, Self, override

import numpy as np
import pandas as pd
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from portfolio_optimizer.domain.results import F64, ChainState, Flags, MissingSpecColumnError, ProblemSpec, Solution
from portfolio_optimizer.domain.sides import SideProfile
from portfolio_optimizer.domain.types import StrictModel

type Direction = Literal["le", "ge"]
"""``le``: the expression stays at or below the bound; ``ge``: at or above. An equality is two rows."""

type Vector = Literal["w", "buy", "sell", "trade"]
"""The decision quantity the constraint bounds. ``trade`` is ``buy + sell``, or the one side a one-sided run has."""

type GroupBounds = tuple[tuple[str, Decimal], ...]
"""Per-group bounds as sorted ``(group, bound)`` pairs — the hashable form a bounds dictionary is canonicalized to."""

CONSTRAINT_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


class ConstraintSpecError(ValueError):
    """A typed constraint row is malformed, or names a column, flag, or group the spec does not carry."""


class TypedConstraint(StrictModel):
    """What every typed constraint declares; frozen and hashable, so a set of constraints is a set.

    ``name`` is the label the report, the manifest, and the acceptance vocabulary key on — unique among
    one portfolio's typed constraints. ``scope`` optionally names a boolean flag column of the
    universe; the constraint then touches only flagged securities, and — for a chain-reading kind —
    couples only through them, which is what narrows the dependency graph.
    """

    name: str = Field(pattern=CONSTRAINT_NAME_PATTERN)
    direction: Direction
    scope: str | None = Field(default=None, min_length=1)
    allow_current_weight: bool = False
    tolerance: Decimal = Field(default=Decimal(0), ge=0)

    @property
    def reads_chain(self) -> bool:
        """Whether this constraint reads what higher-priority portfolios traded; the schedule is derived from this."""
        return False

    def coupling_securities(self, spec: ProblemSpec, profile: SideProfile) -> Flags:
        """The securities this constraint couples its portfolio through: none unless it reads the chain, its scope of the tradable set when it does."""
        if not self.reads_chain:
            return np.zeros(spec.n, dtype=np.bool_)
        return profile.tradable(spec) & self.scope_mask(spec)

    def scope_mask(self, spec: ProblemSpec) -> Flags:
        """The securities the constraint touches: the scope flag's mask, or every security."""
        if self.scope is None:
            return np.ones(spec.n, dtype=np.bool_)
        return spec.flag(self.scope)

    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Violation vectors for the verifier, positive where breached beyond ``tolerance``; plain numpy, shared by every consumer."""
        raise NotImplementedError

    def _signed(self, values: F64, bounds: F64) -> F64:
        """Residual of ``values`` against ``bounds`` under the direction, net of tolerance: positive is a violation either way."""
        slack = float(self.tolerance)
        if self.direction == "le":
            return values - bounds - slack
        return bounds - values - slack

    def _effective_bounds(self, bounds: F64, current: F64) -> F64:
        """The bounds this constraint holds, after the start policy."""
        return effective_bounds(self.direction, self.allow_current_weight, bounds, current)


def effective_bounds(direction: Direction, allow_current: bool, bounds: F64, current: F64) -> F64:
    """The bounds after the start policy: a breached start loosens the bound to the current value instead of failing.

    Shared by the residuals here and the shipped renderers in ``solvers.py``, so the two cannot
    disagree about what the policy means.
    """
    if not allow_current:
        return bounds
    return np.maximum(bounds, current) if direction == "le" else np.minimum(bounds, current)


def vector_values(solution: Solution, vector: Vector) -> F64:
    """The solved values of a decision quantity; ``trade`` is the two sides' sum, which is the one side's value in a one-sided run."""
    if vector == "w":
        return solution.w
    if vector == "buy":
        return solution.buy
    if vector == "sell":
        return solution.sell
    return solution.buy + solution.sell


def starting_values(spec: ProblemSpec, vector: Vector) -> F64:
    """The same quantity before any trade: the held weights for ``w``, zero for anything traded."""
    if vector == "w":
        return spec.w0
    return np.zeros(spec.n)


class GroupLimit(TypedConstraint):
    """Bound the summed ``vector`` over each group of a categorical ``column`` — sector bands, country caps, issuer caps.

    ``bounds`` is one bound for every group, or per-group pairs (a dictionary in JSON); a group the
    pairs do not name is unbounded by this row. The grouping must be one the spec carries as a
    membership matrix — today that is ``sector``; generalizing the column is the open half of the
    constraints thread in ``IDEAS.md``.
    """

    kind: Literal["group_limit"] = "group_limit"
    column: str = Field(min_length=1)
    vector: Vector = "w"
    bounds: Decimal | GroupBounds

    @model_validator(mode="before")
    @classmethod
    def _canonical_bounds(cls, value: object) -> object:
        """A bounds mapping becomes sorted pairs, so the model is hashable and two spellings of one bound compare equal."""
        if not isinstance(value, dict):
            return value
        bounds = value.get("bounds")
        if not isinstance(bounds, Mapping):
            return value
        return {**value, "bounds": tuple(sorted((str(group), bound) for group, bound in bounds.items()))}

    def groups(self, spec: ProblemSpec) -> tuple[tuple[str, Decimal], ...]:
        """The ``(group, bound)`` pairs this row bounds, resolved against the spec's grouping."""
        if self.column != "sector":
            msg = f"{self.name}: group_limit column {self.column!r} is not a grouping the spec carries as a membership matrix; today that is 'sector'"
            raise ConstraintSpecError(msg)
        if isinstance(self.bounds, Decimal):
            return tuple((group, self.bounds) for group in spec.sector_names)
        unknown = sorted(group for group, _ in self.bounds if group not in spec.sector_names)
        if unknown:
            msg = f"{self.name}: bounds name group(s) {unknown} the universe's {self.column!r} does not carry; it has {list(spec.sector_names)}"
            raise ConstraintSpecError(msg)
        return self.bounds

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Per bounded group: the scoped membership row times the vector, against the (possibly start-loosened) bound."""
        del chain, profile
        pairs = self.groups(spec)
        mask = self.scope_mask(spec).astype(np.float64)
        values = vector_values(solution, self.vector) * mask
        start = starting_values(spec, self.vector) * mask
        exposure = np.array([float((spec.sector(group) @ values).sum()) for group, _ in pairs])
        current = np.array([float((spec.sector(group) @ start).sum()) for group, _ in pairs])
        bounds = self._effective_bounds(np.array([float(bound) for _, bound in pairs]), current)
        return [("group_limit", self._signed(exposure, bounds))]


class ExposureLimit(TypedConstraint):
    """Bound the portfolio's exposure to a numeric per-security ``column`` — beta, duration, a score: ``direction`` on ``column · vector``."""

    kind: Literal["exposure_limit"] = "exposure_limit"
    column: str = Field(min_length=1)
    vector: Vector = "w"
    bounds: Decimal

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """The scoped dot product against the bound, start-loosened where the policy allows."""
        del chain, profile
        loadings = spec.column(self.column) * self.scope_mask(spec)
        exposure = float((loadings * vector_values(solution, self.vector)).sum())
        current = float((loadings * starting_values(spec, self.vector)).sum())
        bounds = self._effective_bounds(np.array([float(self.bounds)]), np.array([current]))
        return [("exposure_limit", self._signed(np.array([exposure]), bounds))]


class WeightLimit(TypedConstraint):
    """Bound the ``vector`` per security over the scope — a cap on every flagged name, a floor on every held one, ``buy ≤ 0`` for no new positions."""

    kind: Literal["weight_limit"] = "weight_limit"
    vector: Vector = "w"
    bounds: Decimal

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Per scoped security: the vector against the bound; outside the scope the residual is zero by construction."""
        del chain, profile
        mask = self.scope_mask(spec).astype(np.float64)
        values = vector_values(solution, self.vector) * mask
        bounds = self._effective_bounds(np.full(spec.n, float(self.bounds)), starting_values(spec, self.vector)) * mask
        return [("weight_limit", self._signed(values, bounds))]


class ParticipationLimit(TypedConstraint):
    """The chain-aware ADV cap, typed: this portfolio's trade in each scoped name stays inside its share of the day's volume, *after* what higher-priority portfolios already took.

    ``bounds`` scales the spec's ``adv_capacity`` (the style's ``max_adv_participation`` times the
    day's volume): ``1`` is the style's own budget, ``0.5`` half of it. Because the kind declares
    ``reads_chain`` and the scope declares which names the budget can bind on, the engine couples the
    portfolio through ``scope ∩ tradable`` alone — the narrowing that gives a single-universe book a
    schedule better than the line. ``direction`` must be ``le`` (a floor on participation is not a
    thing), and the start policy does not apply: a trade starts at zero and cannot begin breached.
    """

    kind: Literal["participation_limit"] = "participation_limit"
    bounds: Decimal = Field(default=Decimal(1), gt=0)

    @model_validator(mode="after")
    def _shape_is_meaningful(self) -> Self:
        if self.direction != "le":
            msg = f"{self.name}: participation_limit only bounds from above; direction must be 'le'"
            raise ValueError(msg)
        if self.allow_current_weight:
            msg = f"{self.name}: allow_current_weight does not apply to participation_limit; a trade starts at zero and cannot begin breached"
            raise ValueError(msg)
        return self

    @property
    @override
    def reads_chain(self) -> bool:
        """The budget is cumulative across the book: this is what makes a portfolio wait at all."""
        return True

    def capacity(self, spec: ProblemSpec) -> F64:
        """Each name's budget as a fraction of NAV: the spec's capacity scaled by ``bounds``; the scope says where the constraint applies it."""
        return float(self.bounds) * spec.adv_capacity

    def remaining(self, spec: ProblemSpec, chain: ChainState) -> F64:
        """The budget left after predecessors' trades on the coupled side; shared by the residual and the shipped renderer."""
        if chain.security_ids != spec.security_ids:
            msg = f"{self.name}: chain state is not aligned to this spec's securities"
            raise ConstraintSpecError(msg)
        consumed = chain.traded_shares * spec.price / spec.nav
        return np.maximum(0.0, self.capacity(spec) - consumed)

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Two scoped checks, the twins of the rendered constraints: own trade within the budget, and the coupled side within what predecessors left."""
        mask = self.scope_mask(spec).astype(np.float64)
        trade = (solution.buy + solution.sell) * mask
        coupled = profile.coupled(solution) * mask
        slack = float(self.tolerance)
        return [("participation", trade - self.capacity(spec) * mask - slack), ("cumulative_participation", coupled - self.remaining(spec, chain) * mask - slack)]


type AnyTypedConstraint = GroupLimit | ExposureLimit | WeightLimit | ParticipationLimit
"""Every typed kind; :data:`TYPED_CONSTRAINT` discriminates on ``kind``."""

TYPED_CONSTRAINT: TypeAdapter[AnyTypedConstraint] = TypeAdapter(AnyTypedConstraint)
"""Validator for one typed constraint of any kind, discriminated on ``kind`` by the literal fields."""


class ParsedConstraints(StrictModel):
    """One portfolio's constraint rows as the engine reads them: the typed models, and how many opaque rows remain.

    An opaque row is anything this spec does not type — the function convention's step names, or a
    desk's own vocabulary — and the engine treats it as it always has: uninterpreted, and coupling
    the portfolio through its whole tradable set, because an opaque reader cannot declare a narrower
    scope. ``None`` in place of this object means the frame does not speak the typed spec at all.
    """

    typed: tuple[AnyTypedConstraint, ...]
    opaque_rows: int = Field(ge=0)

    @property
    def reads_chain(self) -> bool:
        """Whether anything here can read the chain: a chain-aware typed kind, or any opaque row."""
        return self.opaque_rows > 0 or any(constraint.reads_chain for constraint in self.typed)


def parse_constraints(frame: pd.DataFrame) -> ParsedConstraints | None:
    """Read one portfolio's constraint rows into the typed union, or ``None`` when the frame does not speak this spec.

    ``None`` — no rows, or no ``kind`` column — leaves everything exactly as it was before typed
    constraints existed. Where the column exists, a row whose ``kind`` names a typed model is
    validated (its fields in ``params`` as JSON text or an object, its ``name`` from ``params``, the
    ``label`` column, or the ``name`` column); any other row is counted as opaque. A malformed typed
    row raises :class:`ConstraintSpecError` naming the row, which the engine records as the
    portfolio's failure at stage ``build`` — before any solve is scheduled on it.
    """
    if frame.empty or "kind" not in frame.columns:
        return None
    typed: list[AnyTypedConstraint] = []
    opaque_rows = 0
    names: dict[str, int] = {}
    for position, record in enumerate(frame.to_dict("records")):
        kind = record.get("kind")
        if _is_missing(kind) or kind == "function":
            opaque_rows += 1
            continue
        constraint = _typed_row(position, {str(key): value for key, value in record.items()})
        if constraint.name in names:
            msg = f"constraints[{position}]: name {constraint.name!r} is also used by constraints[{names[constraint.name]}]; names are what reports key on, so they must be unique"
            raise ConstraintSpecError(msg)
        names[constraint.name] = position
        typed.append(constraint)
    return ParsedConstraints(typed=tuple(typed), opaque_rows=opaque_rows)


def consumed_securities(parsed: ParsedConstraints | None, spec: ProblemSpec, profile: SideProfile, *, chain_aware_terms: bool, opaque_solve: bool, opaque_rows: int) -> tuple[str, ...]:
    """The securities predecessors' trades can reach this portfolio through, sorted — the dependency graph's consume side.

    Empty when nothing can read the chain: the portfolio then waits for no one. The narrowed union of
    the typed chain constraints' scopes when they are the only readers. The whole tradable set the
    moment anything opaque might read it — an opaque constraint row, a configured objective term that
    declares ``chain``, or a solve step other than the shipped cvxpy one (every step receives
    ``request.chain`` and an unknown step may read it however it likes). ``opaque_rows`` is the raw
    frame's row count when ``parsed`` is ``None``, since a frame in another vocabulary is all opaque.
    """
    rows = parsed.opaque_rows if parsed is not None else opaque_rows
    if opaque_solve or chain_aware_terms or rows > 0:
        mask = profile.tradable(spec)
    else:
        mask = np.zeros(spec.n, dtype=np.bool_)
        for constraint in parsed.typed if parsed is not None else ():
            mask |= constraint.coupling_securities(spec, profile)
    return tuple(sorted(security for security, flag in zip(spec.security_ids, mask, strict=True) if flag))


def opaque_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows this spec does not type, exactly as loaded — what a solve step's own convention still interprets."""
    if "kind" not in frame.columns:
        return frame
    keep = pd.Series([_is_missing(kind) or kind == "function" for kind in frame["kind"]], index=frame.index)
    return frame[keep].drop(columns=["kind"]).reset_index(drop=True)


def check_against_spec(constraints: Sequence[AnyTypedConstraint], spec: ProblemSpec) -> Iterator[str]:
    """Every reason a typed constraint cannot apply to this spec — a missing column, flag, or group — for one collected failure."""
    for constraint in constraints:
        try:
            constraint.scope_mask(spec)
            if isinstance(constraint, GroupLimit):
                constraint.groups(spec)
            if isinstance(constraint, ExposureLimit):
                spec.column(constraint.column)
        except (ConstraintSpecError, MissingSpecColumnError) as error:
            yield f"{constraint.name}: {error}"


def _typed_row(position: int, record: dict[str, object]) -> AnyTypedConstraint:
    params = record.get("params")
    if isinstance(params, str) and params.strip():
        body: dict[str, object] = dict(json.loads(params))
    elif isinstance(params, Mapping):
        body = {str(key): value for key, value in params.items()}
    else:
        body = {}
    body.setdefault("kind", record.get("kind"))
    for alias in ("label", "name"):
        value = record.get(alias)
        if "name" not in body and isinstance(value, str) and value:
            body["name"] = value
    try:
        return TYPED_CONSTRAINT.validate_json(json.dumps(body))
    except ValidationError as error:
        details = "; ".join(f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}" for item in error.errors())
        msg = f"constraints[{position}]: {details}"
        raise ConstraintSpecError(msg) from error
    except (TypeError, ValueError) as error:
        msg = f"constraints[{position}]: {error}"
        raise ConstraintSpecError(msg) from error


def _is_missing(value: object) -> bool:
    """The ways an absent optional column arrives from a frame: ``None``, ``pd.NA``, a float ``NaN``, or blank text."""
    return value is None or value is pd.NA or (isinstance(value, float) and bool(np.isnan(value))) or (isinstance(value, str) and not value.strip())
