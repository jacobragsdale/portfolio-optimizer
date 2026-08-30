"""The named datasets between loading and slicing, the per-portfolio bundle, and the typed non-frame inputs."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Self, override

import pandas as pd
from pydantic import Field, model_validator

from portfolio_optimizer.domain.frames import FrameSchemaError, validate_frame
from portfolio_optimizer.domain.optimizer_frame import column_dtype_conflicts, stack_frames
from portfolio_optimizer.domain.schemas import CONSTRAINTS, HOLDINGS, RESERVED_DATASET_NAMES, SECTOR_BOUNDS, TARGETS, UNIVERSE
from portfolio_optimizer.domain.types import Clock, PortfolioId, StrictModel
from portfolio_optimizer.ratelimit import RateLimiter


class Frames(Mapping[str, pd.DataFrame]):
    """Every loaded dataset by name, as assembly steps see them: an immutable mapping of name to frame.

    An assembly step receives one and returns a new one built with :meth:`with_frame` and
    :meth:`without`; the engine validates the engine-known frames after the last step. Looking up a
    name that is not present raises ``KeyError`` naming what is.
    """

    __slots__ = ("_frames",)

    def __init__(self, frames: Mapping[str, pd.DataFrame] | None = None) -> None:
        self._frames: dict[str, pd.DataFrame] = dict(frames or {})
        for name, frame in self._frames.items():
            if not isinstance(frame, pd.DataFrame):
                msg = f"dataset {name!r} is a {type(frame).__name__}, expected DataFrame"
                raise TypeError(msg)

    @override
    def __getitem__(self, name: str) -> pd.DataFrame:
        try:
            return self._frames[name]
        except KeyError:
            msg = f"no dataset {name!r}; available: {sorted(self._frames)}"
            raise KeyError(msg) from None

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._frames)

    @override
    def __len__(self) -> int:
        return len(self._frames)

    @override
    def __repr__(self) -> str:
        return f"Frames({', '.join(f'{name}={len(frame)} rows' for name, frame in self._frames.items())})"

    def with_frame(self, name: str, frame: pd.DataFrame) -> "Frames":
        """Return a copy in which ``name`` is ``frame``, added or replaced."""
        return Frames({**self._frames, name: frame})

    def without(self, *names: str) -> "Frames":
        """Return a copy without ``names``; every name must be present."""
        missing = [name for name in names if name not in self._frames]
        if missing:
            msg = f"no dataset(s) {missing}; available: {sorted(self._frames)}"
            raise KeyError(msg)
        return Frames({name: frame for name, frame in self._frames.items() if name not in names})

    def row_counts(self) -> dict[str, int]:
        """Rows per dataset, for audit records."""
        return {name: len(frame) for name, frame in self._frames.items()}


class PortfolioDetails(StrictModel):
    """One row of the ``details`` dataset: the account's facts and its management-style limits.

    The limits are fractions of NAV and are what every bounded constraint reads: ``max_weight`` is
    the single-name cap, ``max_turnover`` is two-way (buys plus sells), ``cash_lb``/``cash_ub`` bound
    the uninvested fraction, and ``max_adv_participation`` caps a day's share of each name's volume.
    ``min_trade_notional`` is not a constraint at all — the order step drops trades below it after the
    solve. Per-sector limits do not fit a row and live in the ``sector_bounds`` dataset instead.
    """

    portfolio_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    state: str = Field(pattern=r"^[A-Z]{2}$")
    st_tax_rate: Decimal = Field(ge=0, lt=1)
    lt_tax_rate: Decimal = Field(ge=0, lt=1)
    cash: Decimal = Field(ge=0)
    nav: Decimal = Field(gt=0)
    benchmark_id: str = Field(min_length=1)
    max_weight: Decimal = Field(gt=0, le=1)
    max_turnover: Decimal = Field(ge=0, le=2)
    max_adv_participation: Decimal = Field(ge=0, le=1)
    min_trade_notional: Decimal = Field(ge=0)
    cash_lb: Decimal = Field(ge=0, le=1)
    cash_ub: Decimal = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _cash_bounds_are_ordered(self) -> Self:
        if self.cash_lb > self.cash_ub:
            msg = f"cash_lb must not exceed cash_ub, got {self.cash_lb} > {self.cash_ub}"
            raise ValueError(msg)
        return self


def json_default(value: object) -> str:
    """``json.dumps(default=...)`` for the engine's values: a ``Decimal`` is written as plain digits, nothing else is serializable."""
    if isinstance(value, Decimal):
        return format(value, "f")
    msg = f"object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def details_from_frame(frame: pd.DataFrame, portfolio_id: PortfolioId) -> PortfolioDetails:
    """Pick ``portfolio_id``'s row from a validated ``details`` frame and type it."""
    rows = frame[frame["portfolio_id"] == portfolio_id]
    if len(rows) != 1:
        msg = f"details for portfolio {portfolio_id!r}: expected exactly one row, found {len(rows)}"
        raise PortfolioDataError(portfolio_id, [msg])
    record = {str(column): value for column, value in rows.iloc[0].items()}
    return PortfolioDetails.model_validate(record)


PREVALIDATED_FRAMES: frozenset[str] = frozenset({"holdings", "universe", "targets", "sector_bounds", "constraints"})
"""The bundle's frames that have a schema: the only names ``prevalidated`` may carry, and what the engine marks when it slices from assembled datasets."""


class PortfolioDataError(ValueError):
    """The per-portfolio bundle is internally inconsistent."""

    def __init__(self, portfolio_id: str, failures: list[str]) -> None:
        self.portfolio_id = portfolio_id
        self.failures = tuple(failures)
        super().__init__(f"portfolio {portfolio_id!r}: " + "; ".join(failures))


@dataclass(frozen=True, slots=True, eq=False)
class PortfolioData:
    """Everything the engine knows about one portfolio, validated on construction.

    ``holdings`` is what the portfolio owns and ``universe`` what it may buy; a held name need not be
    buyable. Both may carry any analytics columns beyond their schemas, and :meth:`optimizer_frame`
    stacks them into the single frame an optimizer consumes — so every construction checks that
    the columns the two share agree on dtype. ``extras`` are the other datasets the run carried
    through assembly, already reduced to this portfolio's rows where they have a ``portfolio_id``.

    Rules return a new instance via :meth:`with_changes`; every construction re-validates each frame
    against its schema and the cross-frame invariants, so a rule cannot hand the optimizer an
    inconsistent bundle. The one exception is deliberate: ``prevalidated`` names frames the engine
    already validated against the same schema before slicing (the shared universe, a row subset of
    validated holdings or targets), and those are not checked again — a run over a thousand
    portfolios would otherwise validate the same universe a thousand times per rule. Only the
    engine sets it, and :meth:`with_changes` drops the name of any frame that is replaced, so a
    rule's output is always validated.
    """

    details: PortfolioDetails
    holdings: pd.DataFrame
    universe: pd.DataFrame
    targets: pd.DataFrame
    sector_bounds: pd.DataFrame
    constraints: pd.DataFrame
    as_of_date: datetime
    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    applied_rules: tuple[str, ...] = field(default=())
    prevalidated: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", dict(self.extras))
        failures: list[str] = []
        unknown = sorted(self.prevalidated - PREVALIDATED_FRAMES)
        if unknown:
            failures.append(f"prevalidated names unknown frames {unknown}")
        for name, frame, schema in (
            ("holdings", self.holdings, HOLDINGS),
            ("universe", self.universe, UNIVERSE),
            ("targets", self.targets, TARGETS),
            ("sector_bounds", self.sector_bounds, SECTOR_BOUNDS),
            ("constraints", self.constraints, CONSTRAINTS),
        ):
            if name in self.prevalidated:
                continue
            try:
                validate_frame(frame, schema)
            except FrameSchemaError as error:
                failures.extend(f"{name}: {failure}" for failure in error.failures)
        failures.extend(self._extras_failures())
        if failures:
            raise PortfolioDataError(self.details.portfolio_id, failures)
        failures.extend(self._cross_frame_failures())
        if failures:
            raise PortfolioDataError(self.details.portfolio_id, failures)

    def _extras_failures(self) -> list[str]:
        failures: list[str] = []
        for name, frame in self.extras.items():
            if name in RESERVED_DATASET_NAMES:
                failures.append(f"extras: {name!r} is an engine-known dataset name")
            if not isinstance(frame, pd.DataFrame):
                failures.append(f"extras: {name!r} is a {type(frame).__name__}, expected DataFrame")
        return failures

    def _cross_frame_failures(self) -> list[str]:
        failures: list[str] = []
        if self.as_of_date.tzinfo is None or self.as_of_date.utcoffset() != UTC.utcoffset(self.as_of_date):
            failures.append(f"as_of_date must be timezone-aware UTC, got {self.as_of_date!r}")
        own = {self.details.portfolio_id}
        foreign = sorted({str(p) for p in self.holdings["portfolio_id"]} - own)
        if foreign:
            failures.append(f"holdings contain other portfolios {foreign}")
        foreign_benchmarks = sorted({str(b) for b in self.targets["benchmark_id"]} - {self.details.benchmark_id})
        if foreign_benchmarks:
            failures.append(f"targets contain other benchmarks {foreign_benchmarks}")
        known = {str(s) for s in self.universe["security_id"]} | {str(s) for s in self.holdings["security_id"]}
        missing_targets = sorted({str(s) for s in self.targets["security_id"]} - known)
        if missing_targets:
            failures.append(f"target securities in neither holdings nor universe {missing_targets}")
        sectors = {str(s) for s in self.universe["sector"]}
        unknown_sectors = sorted({str(s) for s in self.sector_bounds["sector"]} - sectors)
        if unknown_sectors:
            failures.append(f"sector_bounds reference sectors absent from universe {unknown_sectors}")
        foreign_bounds = sorted({str(p) for p in self.sector_bounds["portfolio_id"]} - own)
        if foreign_bounds:
            failures.append(f"sector_bounds contain other portfolios {foreign_bounds}")
        foreign_constraints = sorted({str(p) for p in self.constraints["portfolio_id"]} - own)
        if foreign_constraints:
            failures.append(f"constraints contain other portfolios {foreign_constraints}")
        failures.extend(f"holdings and universe disagree on {conflict}" for conflict in column_dtype_conflicts({"holdings": self.holdings, "universe": self.universe}))
        for name, frame in self.extras.items():
            if "portfolio_id" in frame.columns:
                foreign_rows = sorted({str(p) for p in frame["portfolio_id"]} - own)
                if foreign_rows:
                    failures.append(f"extras[{name!r}] contain other portfolios {foreign_rows}")
        return failures

    @property
    def portfolio_id(self) -> PortfolioId:
        """Identifier of this portfolio."""
        return PortfolioId(self.details.portfolio_id)

    def optimizer_frame(self, *, source_column: str | None = "source") -> pd.DataFrame:
        """Holdings and universe stacked into the one frame an optimizer consumes.

        Rows are the holdings followed by the universe; a name that is both held and buyable appears
        twice, told apart by ``source_column`` (``"holdings"`` or ``"universe"``; pass ``None`` to
        omit it). Columns are the union of both frames' columns, holdings' first. A column one side
        lacks is null on that side, promoted to its nullable dtype where needed (``bool`` to
        ``boolean``, ``int64`` to ``Int64``); shared columns keep their common dtype, which the
        bundle's own validation has already checked.
        """
        return stack_frames({"holdings": self.holdings, "universe": self.universe}, source_column=source_column)

    def with_changes(
        self,
        *,
        details: PortfolioDetails | None = None,
        holdings: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        targets: pd.DataFrame | None = None,
        sector_bounds: pd.DataFrame | None = None,
        constraints: pd.DataFrame | None = None,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> "PortfolioData":
        """Return a re-validated copy with the given details, frames, or extras replaced.

        A replaced frame loses its ``prevalidated`` standing and is checked against its schema.
        Adjusting ``constraints`` from what the holdings or the universe say is ordinary rule work:
        the frame carries the desk's own columns, and only the solve step interprets them.
        """
        replaced = {name for name, frame in (("holdings", holdings), ("universe", universe), ("targets", targets), ("sector_bounds", sector_bounds), ("constraints", constraints)) if frame is not None}
        return replace(
            self,
            details=self.details if details is None else details,
            holdings=self.holdings if holdings is None else holdings,
            universe=self.universe if universe is None else universe,
            targets=self.targets if targets is None else targets,
            sector_bounds=self.sector_bounds if sector_bounds is None else sector_bounds,
            constraints=self.constraints if constraints is None else constraints,
            extras=self.extras if extras is None else extras,
            prevalidated=self.prevalidated - replaced,
        )

    def with_rule_applied(self, qualname: str) -> "PortfolioData":
        """Record that ``qualname`` ran; called by the pipeline, not by rules."""
        return replace(self, applied_rules=(*self.applied_rules, qualname))


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """What a loader is asked for: which dataset, for which portfolios, as of when, and where data lives.

    ``rate_limiter`` is the pool the dataset's config names, or an unlimited one. A loader that
    makes many calls wraps each in ``async with request.rate_limiter:`` (or
    ``with request.rate_limiter.sync:`` from a sync loader) so large runs stay inside the
    backend's limits.
    """

    dataset: str
    portfolio_ids: tuple[PortfolioId, ...]
    as_of_date: datetime
    data_root: Path
    run_id: str
    rate_limiter: RateLimiter = field(default_factory=RateLimiter.unlimited)


@dataclass(frozen=True, slots=True)
class IoContext:
    """The single seam loaders and sinks receive for filesystem and time access."""

    data_root: Path
    output_dir: Path
    run_id: str
    clock: Clock
