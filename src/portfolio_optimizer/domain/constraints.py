"""Typed constraints: the declaration the engine reads, the cvxpy the shipped step renders, and the residual the verifier checks.

A constraint is a strict, hashable pydantic model the engine can read without understanding the
mathematics: its ``kind`` says whether it reads the chain, and its ``scope`` says which securities
it can couple through. That is all the schedule needs, and it is what lets a portfolio whose
constraints read no chain solve with no predecessors, and a scoped participation cap couple through
its scope alone. The model also carries both halves of what a solve needs: :meth:`TypedConstraint.to_cvxpy`
renders it for the shipped cvxpy step, and :meth:`TypedConstraint.residual` re-checks the answer
in plain numpy for the verifier, so the two cannot drift and a kind a package ships is verified
exactly like a shipped one. This module never imports cvxpy at import time — the renderers reach
for the adapter when called — so ``verify`` runs without the solver stack.

Rows keep the loaded-data convention: a ``kind`` column selects the model and ``params`` holds its
fields as JSON; the row's ``label`` (or ``name``) column may carry the constraint's name. A frame
without a ``kind`` column is written in a vocabulary this module does not know — a custom solve
step's business — and :func:`parse_constraints` returns ``None`` for it.

A bound is a literal, a per-account scalar the spec carries (``{"scalar": "cash_ub"}``), or a
per-security column (``{"column": "ub"}``); the numbers stay in the data, and the row says where.
``allow_current_weight`` is the start policy, per constraint: a book that already breaches the bound
(a name over its cap, a sector over its band) either loosens the bound to the current value — hold
it, do not worsen it — or fails the portfolio as infeasible. The loosening applies to ``w``-shaped
constraints only; a bound on ``buy``, ``sell``, or ``trade`` starts at zero and cannot be breached.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, ClassVar, Literal, Self, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from scipy.sparse import csr_array, vstack

from portfolio_optimizer.domain.registry import KindError, kinds_from, parse_kind
from portfolio_optimizer.domain.results import F64, ChainState, Flags, MissingSpecColumnError, ProblemSpec, Solution
from portfolio_optimizer.domain.sides import SideProfile
from portfolio_optimizer.domain.types import StrictModel

if TYPE_CHECKING:
    from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, Expr

type Direction = Literal["<=", "<", ">=", ">"]
"""Which side of the bound the expression must stay on; an equality is two rows.

The strict and non-strict spellings bind identically: a convex solver enforces the closed bound, and
the verifier's ``tolerance`` is the only real slack. Both are accepted because desks write both.
"""


def bounds_above(direction: Direction) -> bool:
    """True for the ``<=``/``<`` spellings: the expression is held at or below the bound."""
    return direction in ("<=", "<")


type Vector = Literal["w", "buy", "sell", "trade"]
"""The decision quantity a constraint or term bounds. ``trade`` is ``buy + sell``, or the one side a one-sided run has."""

type GroupBounds = tuple[tuple[str, Decimal], ...]
"""Per-group bounds as sorted ``(group, bound)`` pairs — the hashable form a bounds dictionary is canonicalized to."""

CONSTRAINT_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
CONSTRAINT_GROUP = "portfolio_optimizer.constraint"
"""The entry-point group a package publishes constraint kinds under."""


class ConstraintSpecError(ValueError):
    """A typed constraint row is malformed, or names a column, flag, scalar, or group the spec does not carry."""


class ScalarRef(StrictModel):
    """A per-account number the spec carries: ``{"scalar": "cash_ub"}`` reads ``spec.scalar("cash_ub")``."""

    scalar: str = Field(min_length=1, description="The name of a per-account scalar the spec carries: a style limit such as `cash_ub`, or any numeric column of the account's details row.")


class ColumnRef(StrictModel):
    """A per-security vector the spec carries: ``{"column": "ub"}`` reads ``spec.column("ub")``."""

    column: str = Field(min_length=1, description="The name of a per-security vector the spec carries: one of its own (`lb`, `ub`, `w0`, ...) or an exported universe column.")


type ScalarBound = Decimal | ScalarRef
"""A bound that is one number: written literally, or read from the account's scalars."""

type Bound = Decimal | ScalarRef | ColumnRef
"""A bound that may vary per security: a literal or a scalar broadcast to every name, or a column of the spec."""


def scalar_bound(bound: ScalarBound, spec: ProblemSpec) -> float:
    """Resolve a one-number bound against the spec."""
    return spec.scalar(bound.scalar) if isinstance(bound, ScalarRef) else float(bound)


def vector_bound(bound: Bound, spec: ProblemSpec) -> F64:
    """Resolve a per-security bound against the spec, broadcasting a number to every name."""
    if isinstance(bound, ColumnRef):
        return spec.column(bound.column)
    return np.full(spec.n, scalar_bound(bound, spec))


def bound_requirements(bound: Bound, spec: ProblemSpec) -> Iterator[str]:
    """What the spec must carry for ``bound`` to resolve, as failure messages."""
    try:
        vector_bound(bound, spec)
    except MissingSpecColumnError as error:
        yield str(error)


def effective_bounds(direction: Direction, allow_current: bool, bounds: F64, current: F64) -> F64:
    """The bounds after the start policy: a breached start loosens the bound to the current value instead of failing.

    Shared by the residuals and the renderers, so the two cannot disagree about what the policy means.
    """
    if not allow_current:
        return bounds
    return np.maximum(bounds, current) if bounds_above(direction) else np.minimum(bounds, current)


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


def adv_remaining(spec: ProblemSpec, chain: ChainState, scale: float = 1.0) -> F64:
    """The per-name budget left on the side the run couples through, after predecessors' trades there, as a fraction of NAV: ``scale`` times the spec's ``adv_capacity`` column less what the chain consumed."""
    if chain.security_ids != spec.security_ids:
        msg = "chain state is not aligned to this spec's securities"
        raise ConstraintSpecError(msg)
    consumed = chain.traded_shares * spec.price / spec.nav
    return np.maximum(0.0, scale * spec.column("adv_capacity") - consumed)


def _exact(value: object) -> object:
    """A bound written as JSON text or number becomes an exact ``Decimal``; anything else is left for the field to refuse."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return value
    try:
        return Decimal(value.strip() if isinstance(value, str) else repr(value))
    except (InvalidOperation, ValueError):
        return value


def _compare(expr: "Expr", bound: F64 | float, direction: Direction) -> "ConstraintSet | object":
    from portfolio_optimizer.cvx.adapter import at_least, at_most  # the adapter is the one module that imports cvxpy; reached only when rendering

    return at_most(expr, bound) if bounds_above(direction) else at_least(expr, bound)


class TypedConstraint(StrictModel):
    """What every typed constraint declares; frozen and hashable, so a set of constraints is a set.

    ``name`` is the label the report, the manifest, and the acceptance vocabulary key on — unique among
    one portfolio's constraints. ``scope`` optionally names a boolean flag column of the universe;
    the constraint then touches only flagged securities, and — for a chain-reading kind — couples
    only through them, which is what narrows the dependency graph. A subclass narrows ``kind`` to
    the literal that names it, sets ``reads_chain`` when it reads what predecessors traded, and
    implements :meth:`residual`, :meth:`to_cvxpy`, and — for what the spec must carry — :meth:`requirements`.
    """

    name: str = Field(pattern=CONSTRAINT_NAME_PATTERN, description="What the report and the manifest key on; unique among one portfolio's constraints.")
    direction: Direction = Field(description="Which side of the bound the expression must stay on: `<=` or `>=` (`<` and `>` bind identically).")
    scope: str | None = Field(default=None, min_length=1, description="A boolean flag column of the universe; set, the constraint touches only flagged securities and couples only through them.")
    allow_current_weight: bool = Field(
        default=False, description="The start policy: a bound the book already breaches loosens to the current value — hold it, do not worsen it — instead of failing the portfolio."
    )
    tolerance: Decimal = Field(default=Decimal(0), ge=0, description="Slack the verifier allows on the bound; the solver is held to the bound itself.")

    reads_chain: ClassVar[bool] = False
    """Whether this kind reads what higher-priority portfolios traded; the schedule is derived from this."""

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

    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        """Every reason this constraint cannot apply to ``spec`` — a missing column, flag, scalar, or group; empty when it can."""
        try:
            self.scope_mask(spec)
        except MissingSpecColumnError as error:
            yield str(error)

    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Violation vectors for the verifier, positive where breached beyond ``tolerance``; plain numpy."""
        raise NotImplementedError

    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        """This constraint over the decision variables, for the shipped cvxpy step; strict, since ``tolerance`` is verification slack."""
        raise NotImplementedError

    def record(self) -> dict[str, object]:
        """The constraint as JSON-safe data — what a solution carries and the manifest records; :func:`parse_constraint` reads it back."""
        return {str(key): value for key, value in self.model_dump(mode="json").items()}

    def _signed(self, values: F64, bounds: F64) -> F64:
        """Residual of ``values`` against ``bounds`` under the direction, net of tolerance: positive is a violation either way."""
        slack = float(self.tolerance)
        if bounds_above(self.direction):
            return values - bounds - slack
        return bounds - values - slack

    def _effective(self, bounds: F64, current: F64) -> F64:
        """The bounds this constraint holds, after the start policy."""
        return effective_bounds(self.direction, self.allow_current_weight, bounds, current)


class WeightLimit(TypedConstraint):
    """Bound the ``vector`` per security over the scope: a cap on every flagged name, ``w`` inside the spec's own ``lb``/``ub``, ``buy ≤ 0`` for no new positions.

    ``bounds`` may be a column of the spec, so ``{"direction": ">=", "bounds": {"column": "lb"}}`` is
    the long-only floor and ``{"direction": "<=", "bounds": {"column": "ub"}}`` the style cap.
    """

    kind: Literal["weight_limit"] = Field(default="weight_limit", description="The kind: a per-security bound on a decision vector.")
    vector: Vector = Field(default="w", description="The decision vector bounded per security.")
    bounds: Bound = Field(description='One number for every scoped name, a spec scalar (`{"scalar": "max_weight"}`), or a spec column (`{"column": "ub"}`).')

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        yield from super().requirements(spec)
        yield from bound_requirements(self.bounds, spec)

    def _bounds(self, spec: ProblemSpec) -> F64:
        mask = self.scope_mask(spec).astype(np.float64)
        return self._effective(vector_bound(self.bounds, spec), starting_values(spec, self.vector)) * mask

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Per scoped security: the vector against the bound; outside the scope the residual is zero by construction."""
        del chain, profile
        values = vector_values(solution, self.vector) * self.scope_mask(spec).astype(np.float64)
        return [("weight_limit", self._signed(values, self._bounds(spec)))]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, masked

        del chain
        return ConstraintSet(self.name, (_compare(masked(self.scope_mask(spec), x.vector(self.vector)), self._bounds(spec), self.direction),))  # ty: ignore[invalid-argument-type]  # _compare returns a Constraint


class GroupLimit(TypedConstraint):
    """Bound the summed ``vector`` over each group of a categorical ``column``: sector bands, country caps, issuer caps.

    ``bounds`` is one bound for every group, or per-group pairs (a dictionary in JSON); a group the
    pairs do not name is unbounded by this row. The column must be one the spec carries as a
    grouping — every string column of the universe is.
    """

    kind: Literal["group_limit"] = Field(default="group_limit", description="The kind: a bound on each group of a categorical column.")
    column: str = Field(min_length=1, description="A string column of the universe, carried by the spec as a grouping: `sector`, `country`, `issuer`, ...")
    vector: Vector = Field(default="w", description="The decision vector summed over each group.")
    bounds: Decimal | GroupBounds = Field(description='One bound for every group, or a mapping of group to bound (`{"TECH": "0.5"}`); a group the mapping does not name is unbounded by this row.')

    @field_validator("bounds", mode="before")
    @classmethod
    def _canonical_bounds(cls, value: object) -> object:
        """A mapping or a list of pairs becomes sorted ``(group, Decimal)`` pairs and a number an exact ``Decimal``, so the model is hashable and two spellings of one bound compare equal.

        A before-validator's field is validated as Python rather than JSON afterwards, so the values
        it returns are the exact types the field declares.
        """
        if isinstance(value, Mapping):
            return tuple(sorted((str(group), _exact(bound)) for group, bound in value.items()))
        if isinstance(value, list | tuple):
            return tuple((str(group), _exact(bound)) for group, bound in cast("Iterable[tuple[object, object]]", value))
        return _exact(value)

    def groups(self, spec: ProblemSpec) -> tuple[tuple[str, Decimal], ...]:
        """The ``(group, bound)`` pairs this row bounds, resolved against the spec's grouping."""
        grouping = spec.group(self.column)
        if isinstance(self.bounds, Decimal):
            return tuple((group, self.bounds) for group in grouping.names)
        unknown = sorted(group for group, _ in self.bounds if group not in grouping.names)
        if unknown:
            msg = f"{self.name}: bounds name group(s) {unknown} the universe's {self.column!r} does not carry; it has {list(grouping.names)}"
            raise ConstraintSpecError(msg)
        return self.bounds

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        yield from super().requirements(spec)
        try:
            self.groups(spec)
        except (ConstraintSpecError, MissingSpecColumnError) as error:
            yield str(error)

    def _membership(self, spec: ProblemSpec) -> csr_array:
        """One row per bounded group, restricted to the scope."""
        grouping = spec.group(self.column)
        mask = self.scope_mask(spec).astype(np.float64)
        return csr_array(vstack([grouping.row(group) for group, _ in self.groups(spec)], format="csr").multiply(mask))

    def _bounds(self, spec: ProblemSpec, membership: csr_array) -> F64:
        current = np.asarray(membership @ starting_values(spec, self.vector), dtype=np.float64)
        return self._effective(np.array([float(bound) for _, bound in self.groups(spec)]), current)

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Per bounded group: the scoped membership row times the vector, against the (possibly start-loosened) bound."""
        del chain, profile
        membership = self._membership(spec)
        exposure = np.asarray(membership @ vector_values(solution, self.vector), dtype=np.float64)
        return [("group_limit", self._signed(exposure, self._bounds(spec, membership)))]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, matvec

        del chain
        membership = self._membership(spec)
        return ConstraintSet(self.name, (_compare(matvec(membership, x.vector(self.vector)), self._bounds(spec, membership), self.direction),))  # ty: ignore[invalid-argument-type]  # _compare returns a Constraint


class ExposureLimit(TypedConstraint):
    """Bound the portfolio's exposure to a numeric per-security ``column`` — beta, duration, a score: ``direction`` on ``column · vector``."""

    kind: Literal["exposure_limit"] = Field(default="exposure_limit", description="The kind: a bound on a column's dot product with a decision vector.")
    column: str = Field(min_length=1, description="A numeric per-security column of the spec: a beta, a duration, a score.")
    vector: Vector = Field(default="w", description="The decision vector the column multiplies.")
    bounds: ScalarBound = Field(description='The bound: a number, or a spec scalar (`{"scalar": "max_beta"}`).')

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        yield from super().requirements(spec)
        try:
            spec.column(self.column)
            scalar_bound(self.bounds, spec)
        except MissingSpecColumnError as error:
            yield str(error)

    def _loadings(self, spec: ProblemSpec) -> F64:
        return spec.column(self.column) * self.scope_mask(spec)

    def _bound(self, spec: ProblemSpec) -> float:
        current = float((self._loadings(spec) * starting_values(spec, self.vector)).sum())
        return float(self._effective(np.array([scalar_bound(self.bounds, spec)]), np.array([current]))[0])

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """The scoped dot product against the bound, start-loosened where the policy allows."""
        del chain, profile
        exposure = float((self._loadings(spec) * vector_values(solution, self.vector)).sum())
        return [("exposure_limit", self._signed(np.array([exposure]), np.array([self._bound(spec)])))]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, dot

        del chain
        return ConstraintSet(self.name, (_compare(dot(self._loadings(spec), x.vector(self.vector)), self._bound(spec), self.direction),))  # ty: ignore[invalid-argument-type]  # _compare returns a Constraint


class CashLimit(TypedConstraint):
    """Bound the cash left after the run, ``1 - sum(w)``: ``{"direction": ">=", "bounds": {"scalar": "cash_lb"}}`` is the account's floor, ``<=`` on ``cash_ub`` its cap; both at ``0`` is full investment.

    Scope does not apply: cash is what every name together leaves uninvested.
    """

    kind: Literal["cash_limit"] = Field(default="cash_limit", description="The kind: a bound on the cash left after the run.")
    bounds: ScalarBound = Field(description='The bound on `1 - sum(w)`: a number, or a spec scalar (`{"scalar": "cash_lb"}`).')

    @model_validator(mode="after")
    def _unscoped(self) -> Self:
        if self.scope is not None:
            msg = f"{self.name}: cash_limit takes no scope; cash is what the whole book leaves uninvested"
            raise ValueError(msg)
        return self

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        try:
            scalar_bound(self.bounds, spec)
        except MissingSpecColumnError as error:
            yield str(error)

    def _bound(self, spec: ProblemSpec) -> float:
        current = 1.0 - float(spec.w0.sum())
        return float(self._effective(np.array([scalar_bound(self.bounds, spec)]), np.array([current]))[0])

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        del chain, profile
        cash = 1.0 - float(solution.w.sum())
        return [("cash_limit", self._signed(np.array([cash]), np.array([self._bound(spec)])))]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, shortfall, total

        del chain
        cash = shortfall(np.ones(1), total(x.w))
        return ConstraintSet(self.name, (_compare(cash, self._bound(spec), self.direction),))  # ty: ignore[invalid-argument-type]  # _compare returns a Constraint


class TurnoverLimit(TypedConstraint):
    """Bound the summed ``vector`` over the scope: ``trade`` against ``{"scalar": "max_turnover"}`` is the style's two-way turnover cap."""

    kind: Literal["turnover_limit"] = Field(default="turnover_limit", description="The kind: a bound on a decision vector's sum over the scope.")
    vector: Vector = Field(default="trade", description="The decision vector summed; `trade` is two-way turnover.")
    bounds: ScalarBound = Field(description='The bound: a number, or a spec scalar (`{"scalar": "max_turnover"}`).')

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        yield from super().requirements(spec)
        try:
            scalar_bound(self.bounds, spec)
        except MissingSpecColumnError as error:
            yield str(error)

    def _bound(self, spec: ProblemSpec) -> float:
        current = float((starting_values(spec, self.vector) * self.scope_mask(spec)).sum())
        return float(self._effective(np.array([scalar_bound(self.bounds, spec)]), np.array([current]))[0])

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        del chain, profile
        summed = float((vector_values(solution, self.vector) * self.scope_mask(spec)).sum())
        return [("turnover_limit", self._signed(np.array([summed]), np.array([self._bound(spec)])))]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, masked, total

        del chain
        return ConstraintSet(self.name, (_compare(total(masked(self.scope_mask(spec), x.vector(self.vector))), self._bound(spec), self.direction),))  # ty: ignore[invalid-argument-type]  # _compare returns a Constraint


class ParticipationLimit(TypedConstraint):
    """The chain-aware ADV cap: this portfolio's trade in each scoped name stays inside its share of the day's volume, *after* what higher-priority portfolios already took.

    ``bounds`` scales the spec's ``adv_capacity`` column (the style's ``max_adv_participation`` times
    the day's volume): ``1`` is the style's own budget, ``0.5`` half of it. Because the kind declares
    ``reads_chain`` and the scope declares which names the budget can bind on, the engine couples the
    portfolio through ``scope ∩ tradable`` alone — the narrowing that gives a single-universe book a
    schedule better than the line. ``direction`` must be ``<=`` (a floor on participation is not a
    thing), and the start policy does not apply: a trade starts at zero and cannot begin breached.
    """

    kind: Literal["participation_limit"] = Field(default="participation_limit", description="The kind: the chain-aware cap on each name's share of its daily volume.")
    bounds: Decimal = Field(default=Decimal(1), gt=0, description="A multiple of the spec's `adv_capacity` column: `1` is the style's own participation, `0.5` half of it.")

    reads_chain: ClassVar[bool] = True

    @model_validator(mode="after")
    def _shape_is_meaningful(self) -> Self:
        if not bounds_above(self.direction):
            msg = f"{self.name}: participation_limit only bounds from above; direction must be '<='"
            raise ValueError(msg)
        if self.allow_current_weight:
            msg = f"{self.name}: allow_current_weight does not apply to participation_limit; a trade starts at zero and cannot begin breached"
            raise ValueError(msg)
        return self

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        yield from super().requirements(spec)
        try:
            spec.column("adv_capacity")
        except MissingSpecColumnError as error:
            yield f"{error}; a participation limit needs the universe's adv_shares"

    def capacity(self, spec: ProblemSpec) -> F64:
        """Each name's budget as a fraction of NAV: the spec's capacity scaled by ``bounds``; the scope says where the constraint applies it."""
        return float(self.bounds) * spec.column("adv_capacity")

    def remaining(self, spec: ProblemSpec, chain: ChainState) -> F64:
        """The budget left after predecessors' trades on the coupled side; shared by the residual and the renderer."""
        return adv_remaining(spec, chain, float(self.bounds))

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: SideProfile) -> list[tuple[str, F64]]:
        """Two scoped checks, the twins of the rendered constraints: own trade within the budget, and the coupled side within what predecessors left."""
        mask = self.scope_mask(spec).astype(np.float64)
        trade = (solution.buy + solution.sell) * mask
        coupled = profile.coupled(solution) * mask
        slack = float(self.tolerance)
        return [("participation", trade - self.capacity(spec) * mask - slack), ("cumulative_participation", coupled - self.remaining(spec, chain) * mask - slack)]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, at_most, masked

        flags = self.scope_mask(spec)
        mask = flags.astype(np.float64)
        return ConstraintSet(self.name, (at_most(masked(flags, x.trade), self.capacity(spec) * mask), at_most(masked(flags, x.coupled), self.remaining(spec, chain) * mask)))


SHIPPED_CONSTRAINT_KINDS: tuple[type[TypedConstraint], ...] = (WeightLimit, GroupLimit, ExposureLimit, CashLimit, TurnoverLimit, ParticipationLimit)
"""The kinds this package ships; a package adds its own under the :data:`CONSTRAINT_GROUP` entry-point group."""

_REGISTERED: dict[str, type[TypedConstraint]] = {}


def register_constraint_kind[T: TypedConstraint](model: type[T]) -> type[T]:
    """Make ``model`` a known kind in this process — what loading an entry point does, for a notebook or a test; usable as a decorator."""
    _REGISTERED[model.__name__] = model
    return model


def constraint_kinds() -> Mapping[str, type[TypedConstraint]]:
    """Every constraint kind known here, by name: shipped, published by installed packages, or registered."""
    return kinds_from(CONSTRAINT_GROUP, TypedConstraint, SHIPPED_CONSTRAINT_KINDS, _REGISTERED.values())


def parse_constraint(body: Mapping[str, object], where: str = "constraint") -> TypedConstraint:
    """Validate one constraint record as the kind it names; a failure names ``where``."""
    try:
        return parse_kind(constraint_kinds(), body, where)
    except KindError as error:
        raise ConstraintSpecError(str(error)) from error


@dataclass(frozen=True, slots=True)
class ParsedConstraints:
    """One portfolio's constraint rows as the engine reads them."""

    typed: tuple[TypedConstraint, ...]

    @property
    def reads_chain(self) -> bool:
        """Whether any of them reads the chain."""
        return any(constraint.reads_chain for constraint in self.typed)


def parse_constraints(frame: pd.DataFrame) -> ParsedConstraints | None:
    """Read one portfolio's constraint rows into their models, or ``None`` when the frame does not speak this vocabulary.

    ``None`` — no rows, or no ``kind`` column — is a frame a custom solve step interprets its own way.
    Where the column exists every row must name a kind: its fields come from ``params`` (JSON text or
    an object), and its ``name`` from ``params``, the ``label`` column, or the ``name`` column. A
    malformed row raises :class:`ConstraintSpecError` naming the row, which the engine records as the
    portfolio's failure at stage ``build`` — before any solve is scheduled on it.
    """
    if frame.empty or "kind" not in frame.columns:
        return None
    typed: list[TypedConstraint] = []
    names: dict[str, int] = {}
    for position, record in enumerate(frame.to_dict("records")):
        constraint = _typed_row(position, {str(key): value for key, value in record.items()})
        if constraint.name in names:
            msg = f"constraints[{position}]: name {constraint.name!r} is also used by constraints[{names[constraint.name]}]; names are what reports key on, so they must be unique"
            raise ConstraintSpecError(msg)
        names[constraint.name] = position
        typed.append(constraint)
    return ParsedConstraints(typed=tuple(typed))


def frame_reads_chain(frame: pd.DataFrame) -> bool:
    """Whether any row of a whole book's constraints can read what others traded: a chain-reading kind, or rows in a vocabulary the engine cannot read."""
    if frame.empty:
        return False
    if "kind" not in frame.columns:
        return True
    chain_kinds = {name for name, model in constraint_kinds().items() if model.reads_chain}
    return any(is_missing(kind) or kind in chain_kinds for kind in frame["kind"])


def consumed_securities(parsed: ParsedConstraints | None, spec: ProblemSpec, profile: SideProfile, *, chain_aware_terms: bool, opaque_solve: bool) -> tuple[str, ...]:
    """The securities predecessors' trades can reach this portfolio through, sorted — the dependency graph's consume side.

    Empty when nothing can read the chain: the portfolio then waits for no one. The union of the
    chain-reading constraints' scopes when they are the only readers. The whole tradable set the
    moment anything opaque might read it — a configured objective term that declares ``reads_chain``,
    or a solve step other than the shipped cvxpy one (every step receives ``request.chain`` and an
    unknown step may read it however it likes).
    """
    if opaque_solve or chain_aware_terms:
        mask = profile.tradable(spec)
    else:
        mask = np.zeros(spec.n, dtype=np.bool_)
        for constraint in parsed.typed if parsed is not None else ():
            mask |= constraint.coupling_securities(spec, profile)
    return tuple(sorted(security for security, flag in zip(spec.security_ids, mask, strict=True) if flag))


def check_against_spec(constraints: Sequence[TypedConstraint], spec: ProblemSpec) -> Iterator[str]:
    """Every reason a typed constraint cannot apply to this spec — a missing column, flag, scalar, or group — for one collected failure."""
    for constraint in constraints:
        for problem in constraint.requirements(spec):
            yield f"{constraint.name}: {problem}"


def _typed_row(position: int, record: dict[str, object]) -> TypedConstraint:
    import json

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
    if is_missing(body.get("kind")):
        msg = f"constraints[{position}]: the row names no kind; known kinds: {sorted(constraint_kinds())}"
        raise ConstraintSpecError(msg)
    return parse_constraint(body, f"constraints[{position}]")


def is_missing(value: object) -> bool:
    """The ways an absent optional column arrives from a frame: ``None``, ``pd.NA``, a float ``NaN``, or blank text."""
    return value is None or value is pd.NA or (isinstance(value, float) and bool(np.isnan(value))) or (isinstance(value, str) and not value.strip())
