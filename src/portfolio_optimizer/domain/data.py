"""The per-portfolio data bundle and the typed forms of the non-frame inputs."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from pydantic import Field, model_validator

from portfolio_optimizer.domain.frames import FrameSchemaError, validate_frame
from portfolio_optimizer.domain.schemas import COVARIANCE, HOLDINGS, TARGETS, UNIVERSE
from portfolio_optimizer.domain.types import Clock, PortfolioId, StrictModel
from portfolio_optimizer.ratelimit import RateLimiter


class PortfolioDetails(StrictModel):
    """One row of the ``details`` dataset."""

    portfolio_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    state: str = Field(pattern=r"^[A-Z]{2}$")
    st_tax_rate: Decimal = Field(ge=0, lt=1)
    lt_tax_rate: Decimal = Field(ge=0, lt=1)
    cash: Decimal = Field(ge=0)
    nav: Decimal = Field(gt=0)
    benchmark_id: str = Field(min_length=1)


class StyleConstraints(StrictModel):
    """Management-style limits for one portfolio, typed from the ``constraints`` dict.

    All fractions are of NAV. ``max_turnover`` is two-way (buys plus sells).
    """

    max_weight: Decimal = Field(gt=0, le=1)
    max_turnover: Decimal = Field(ge=0, le=2)
    min_trade_notional: Decimal = Field(ge=0)
    cash_bounds: tuple[Decimal, Decimal]
    max_adv_participation: Decimal = Field(ge=0, le=1)
    sector_bounds: dict[str, tuple[Decimal, Decimal]] = Field(default_factory=dict)
    long_only: Literal[True] = True

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        low, high = self.cash_bounds
        if not (Decimal(0) <= low <= high <= Decimal(1)):
            msg = f"cash_bounds must satisfy 0 <= low <= high <= 1, got {self.cash_bounds}"
            raise ValueError(msg)
        for sector, (sector_low, sector_high) in self.sector_bounds.items():
            if not (Decimal(0) <= sector_low <= sector_high <= Decimal(1)):
                msg = f"sector_bounds[{sector!r}] must satisfy 0 <= low <= high <= 1, got {(sector_low, sector_high)}"
                raise ValueError(msg)
        return self


def style_constraints_from_mapping(mapping: Mapping[str, object]) -> StyleConstraints:
    """Validate a raw constraints dict (as loaded from JSON or a database row).

    The mapping is round-tripped through JSON so that money written as strings becomes
    ``Decimal`` exactly; ``Decimal`` values already present are serialized as strings first.
    """
    return StyleConstraints.model_validate_json(json.dumps(mapping, default=_json_default))


def _json_default(value: object) -> str:
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


class PortfolioDataError(ValueError):
    """The per-portfolio bundle is internally inconsistent."""

    def __init__(self, portfolio_id: str, failures: list[str]) -> None:
        self.portfolio_id = portfolio_id
        self.failures = tuple(failures)
        super().__init__(f"portfolio {portfolio_id!r}: " + "; ".join(failures))


@dataclass(frozen=True, slots=True, eq=False)
class PortfolioData:
    """Everything the engine knows about one portfolio, validated on construction.

    Rules return a new instance via :meth:`with_frames`; every construction re-validates each
    frame against its schema and the cross-frame invariants, so a rule cannot hand the
    optimizer an inconsistent bundle.
    """

    details: PortfolioDetails
    holdings: pd.DataFrame
    universe: pd.DataFrame
    targets: pd.DataFrame
    style: StyleConstraints
    as_of: datetime
    covariance: pd.DataFrame | None = None
    applied_rules: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        failures: list[str] = []
        for name, frame, schema in (("holdings", self.holdings, HOLDINGS), ("universe", self.universe, UNIVERSE), ("targets", self.targets, TARGETS)):
            try:
                validate_frame(frame, schema)
            except FrameSchemaError as error:
                failures.extend(f"{name}: {failure}" for failure in error.failures)
        if self.covariance is not None:
            try:
                validate_frame(self.covariance, COVARIANCE)
            except FrameSchemaError as error:
                failures.extend(f"covariance: {failure}" for failure in error.failures)
        if failures:
            raise PortfolioDataError(self.details.portfolio_id, failures)
        failures.extend(self._cross_frame_failures())
        if failures:
            raise PortfolioDataError(self.details.portfolio_id, failures)

    def _cross_frame_failures(self) -> list[str]:
        failures: list[str] = []
        if self.as_of.tzinfo is None or self.as_of.utcoffset() != UTC.utcoffset(self.as_of):
            failures.append(f"as_of must be timezone-aware UTC, got {self.as_of!r}")
        foreign = sorted({str(p) for p in self.holdings["portfolio_id"]} - {self.details.portfolio_id})
        if foreign:
            failures.append(f"holdings contain other portfolios {foreign}")
        universe_ids = {str(s) for s in self.universe["security_id"]}
        missing_held = sorted({str(s) for s in self.holdings["security_id"]} - universe_ids)
        if missing_held:
            failures.append(f"held securities missing from universe {missing_held}")
        foreign_benchmarks = sorted({str(b) for b in self.targets["benchmark_id"]} - {self.details.benchmark_id})
        if foreign_benchmarks:
            failures.append(f"targets contain other benchmarks {foreign_benchmarks}")
        missing_targets = sorted({str(s) for s in self.targets["security_id"]} - universe_ids)
        if missing_targets:
            failures.append(f"target securities missing from universe {missing_targets}")
        if self.covariance is not None:
            covered = {str(s) for s in self.covariance["security_id_a"]} & {str(s) for s in self.covariance["security_id_b"]}
            uncovered = sorted(universe_ids - covered)
            if uncovered:
                failures.append(f"covariance does not cover universe securities {uncovered}")
        sectors = {str(s) for s in self.universe["sector"]}
        unknown_sectors = sorted(set(self.style.sector_bounds) - sectors)
        if unknown_sectors:
            failures.append(f"sector_bounds reference sectors absent from universe {unknown_sectors}")
        return failures

    @property
    def portfolio_id(self) -> PortfolioId:
        """Identifier of this portfolio."""
        return PortfolioId(self.details.portfolio_id)

    def with_changes(
        self,
        *,
        holdings: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        targets: pd.DataFrame | None = None,
        covariance: pd.DataFrame | None = None,
        style: StyleConstraints | None = None,
    ) -> "PortfolioData":
        """Return a re-validated copy with the given frames or style replaced."""
        return replace(
            self,
            holdings=self.holdings if holdings is None else holdings,
            universe=self.universe if universe is None else universe,
            targets=self.targets if targets is None else targets,
            covariance=self.covariance if covariance is None else covariance,
            style=self.style if style is None else style,
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
    as_of: datetime
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
