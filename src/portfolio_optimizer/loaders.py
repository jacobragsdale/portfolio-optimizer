"""Dataset loaders — yours to edit.

A loader is an ordinary function ``(request: LoadRequest, params: P) -> pd.DataFrame`` named in the
run config; it may be ``async def``. Every dataset is a frame, so that is the only shape. Loaders are
the only place a network call, a database query, or a file read belongs; everything downstream is pure.

The engine loads the datasets as the dependency DAG the config declares: each starts the moment the
datasets its entry depends on have loaded, one with no dependencies starts immediately, and an
``async def`` loader runs as a task on the event loop while a plain one runs in a worker thread, so a
blocking driver never stalls the loop. Every call to a source goes through that input's rate limit — ``async with
request.rate_limiter:`` from a coroutine, ``with request.rate_limiter.sync:`` from a thread — and the
pool is named per input in the run config.

The six loaders here stand in for the services a desk actually has: a book of record, a custodian's
position service, a security master, the account-master database, a compliance service, and a
parameter store. Each waits as long as its own source would, drawn from the bands below, and then
answers from a CSV table under ``request.data_root``, so the template runs against no infrastructure
at all. Replacing a mock with the real thing changes the middle line and nothing around it: the wait
becomes the call, ``request.portfolio_ids`` becomes the query's id list, and the frame still comes
back cast to the dataset's schema.
"""

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import Field, model_validator

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.frames import ColumnSpec, FrameSchema, coerce_frame, empty_frame
from portfolio_optimizer.domain.schemas import CONSTRAINTS, DETAILS, HOLDINGS, PORTFOLIOS, UNIVERSE
from portfolio_optimizer.domain.types import Params, PortfolioId
from portfolio_optimizer.ratelimit import fan_out


@dataclass(frozen=True, slots=True)
class Latency:
    """How long one call to a source takes: a draw from ``[low_s, high_s]``, seeded so a run reproduces its own timings."""

    low_s: float
    high_s: float

    def draw(self, seed: str) -> float:
        """Seconds this call takes."""
        return random.Random(seed).uniform(self.low_s, self.high_s)


BOOK_OF_RECORD = Latency(0.5, 3.0)
"""Which accounts are in scope today: a small query, but nothing else can start until it answers."""

CUSTODIAN = Latency(0.5, 2.0)
"""One account's positions. Fast per call, and there is one call per account."""

SECURITY_MASTER = Latency(8.0, 30.0)
"""Every tradable name with its analytics. The slowest input in the run by an order of magnitude."""

ACCOUNT_MASTER = Latency(1.0, 6.0)
"""A batch of accounts from the firm's own database: one query however many ids it is given."""

COMPLIANCE = Latency(2.0, 9.0)
"""Every account's mandate rules, in the vocabulary the desk writes them in."""

PARAMETER_STORE = Latency(0.5, 1.5)
"""A named set of runtime settings. Small, cached, and asked for more than once a run."""

PARAMETERS = FrameSchema(name="parameters", columns=(ColumnSpec("name", "string"), ColumnSpec("value", "decimal")), key=("name",))
"""The shape of a parameter set: an extra dataset the engine does not know, typed by the loader that fetches it."""


class ServiceParams(Params):
    """What every shipped loader accepts: how long its source is pretended to take.

    Each loader draws a wait from its own band — :data:`SECURITY_MASTER` is tens of seconds, a
    parameter store is under one — seeded on the run id, the dataset, and the ids in the call, so two
    runs of one config wait the same amounts. Setting both fields overrides that band; setting both to
    zero removes the wait, which is what a test does. A real loader has none of this: delete the field
    and the wait, and keep the rest.
    """

    min_latency_s: float | None = Field(default=None, ge=0, description="Shortest this source may take to answer, in seconds. Omit for the loader's own band.")
    max_latency_s: float | None = Field(default=None, ge=0, description="Longest this source may take to answer, in seconds. Omit for the loader's own band.")

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if self.min_latency_s is not None and self.max_latency_s is not None and self.min_latency_s > self.max_latency_s:
            msg = f"min_latency_s must not exceed max_latency_s, got {self.min_latency_s} > {self.max_latency_s}"
            raise ValueError(msg)
        return self

    def latency(self, default: Latency) -> Latency:
        """The band this call draws from: the loader's own, with either end the config overrode.

        The end the config set wins and the other moves out of its way, so ``max_latency_s: 0`` is no
        wait at all whatever the loader's own floor is.
        """
        low = default.low_s if self.min_latency_s is None else self.min_latency_s
        high = default.high_s if self.max_latency_s is None else self.max_latency_s
        return Latency(low if self.min_latency_s is not None else min(low, high), high if self.max_latency_s is not None else max(low, high))


class ParametersParams(ServiceParams):
    """Parameters for :func:`load_parameters`."""

    set_name: str | None = Field(default=None, min_length=1, description="Which parameter set to fetch. Omit to fetch the one named by the dataset itself, which is the usual arrangement.")


async def load_portfolios(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """The accounts in scope for this run, and the order they solve in.

    Nothing waits on it but the inputs that ask for the book's ids — in the example the per-account
    datasets and ``constraints`` — so the security-master scan runs beside this call rather than
    behind it. ``solve_order`` is a priority — lower solves first, ties break on the id — and may be
    omitted. A run over a fixed book can skip the loader and write the ids inline in the config.
    """
    await _call(request, params.latency(BOOK_OF_RECORD))
    return _table(request, "portfolios.csv", PORTFOLIOS)


async def load_holdings(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """What each account in the request owns: shares, average cost, and the date the lot was acquired.

    The shape of a custodian that answers one account per call. :func:`~portfolio_optimizer.ratelimit.fan_out`
    starts every call at once and the rate limiter decides how many actually run, so the loader is
    right however the engine batches it — under the example's ``batch_size: 1`` it fans out over one
    id, and over the whole book if that key were removed.
    """
    latency = params.latency(CUSTODIAN)
    positions = _table(request, "holdings.csv", HOLDINGS)  # the mock's store; a real client holds a connection here instead

    async def one(portfolio_id: PortfolioId) -> pd.DataFrame:
        await asyncio.sleep(latency.draw(f"{request.run_id}:{request.dataset}:{portfolio_id}"))
        return _rows_for(positions, (portfolio_id,))

    parts = await fan_out(request.portfolio_ids, one, limiter=request.rate_limiter)
    return pd.concat(parts, ignore_index=True) if parts else empty_frame(HOLDINGS)


async def load_universe(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """Every security the book may trade, with its price, sector, liquidity, and analytics.

    One call for the whole run, and the long pole of the load stage: a security-master scan is tens of
    seconds and every other input finishes inside it. That is the argument for loading concurrently —
    the per-account calls cost nothing while this one is outstanding.
    """
    await _call(request, params.latency(SECURITY_MASTER))
    return _table(request, "universe.csv", UNIVERSE)


def load_details(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """The account master for the accounts in the request: NAV, cash, tax rates, and the style limits.

    A plain ``def`` because the driver blocks: the engine runs it in a worker thread, and it takes the
    limiter's synchronous form, which draws from the same pool an async loader would. One query
    however many ids it is given — the shape ``batch_size`` in the config exists for.
    """
    _call_blocking(request, params.latency(ACCOUNT_MASTER))
    return _rows_for(_table(request, "details.csv", DETAILS), request.portfolio_ids)


async def load_constraints(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """Which constraints bind each account and how tight they are, in the desk's own vocabulary.

    The example declares ``depends_on: ["portfolios"]`` on this input, so this one call receives the
    book's ids and fetches the book rather than the firm. The engine reads only ``portfolio_id``;
    every other column travels untouched to the solve step, which is the only thing that interprets it.
    """
    await _call(request, params.latency(COMPLIANCE))
    return _rows_for(_table(request, "constraints.csv", CONSTRAINTS), request.portfolio_ids)


async def load_parameters(request: LoadRequest, params: ParametersParams) -> pd.DataFrame:
    """One named set of runtime settings as ``name``/``value`` rows.

    The engine knows nothing about a dataset like this: it is an extra, carried through assembly to
    every portfolio's ``data.extras``, where a rule or the solve step reads the numbers it needs. The
    set fetched is the dataset's own name unless ``set_name`` says otherwise, so a run that wants two
    sets names this loader twice.
    """
    await _call(request, params.latency(PARAMETER_STORE))
    return _table(request, f"{params.set_name or request.dataset}.csv", PARAMETERS)


async def _call(request: LoadRequest, latency: Latency, key: str = "") -> None:
    """Hold the input's rate limit for as long as the source would take to answer."""
    async with request.rate_limiter:
        await asyncio.sleep(latency.draw(f"{request.run_id}:{request.dataset}:{key}"))


def _call_blocking(request: LoadRequest, latency: Latency, key: str = "") -> None:
    """The same wait from a worker thread; the bridge hands the acquisition to the engine's event loop."""
    with request.rate_limiter.sync:
        time.sleep(latency.draw(f"{request.run_id}:{request.dataset}:{key}"))


def _rows_for(frame: pd.DataFrame, portfolio_ids: Sequence[PortfolioId]) -> pd.DataFrame:
    """The rows belonging to the accounts the request names: the mock's ``where portfolio_id in (...)``."""
    return frame[frame["portfolio_id"].isin(list(portfolio_ids))].reset_index(drop=True)


def _table(request: LoadRequest, name: str, schema: FrameSchema) -> pd.DataFrame:
    """Read one table from under ``request.data_root``, cast to ``schema``.

    Every dtype is declared at the boundary — money as ``Decimal``, timestamps with a zone — because
    this is the last place raw input can be typed. A real loader casts what the client returned the
    same way, with :func:`~portfolio_optimizer.domain.frames.coerce_frame`.
    """
    return _read_csv(request.data_root / name, schema)


def _read_csv(path: Path, schema: FrameSchema) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=_read_dtypes(schema))
    return coerce_frame(_parse_utc(raw, [c.name for c in schema.columns if c.kind == "datetime_utc" and c.name in raw.columns]), schema)


def _read_dtypes(schema: FrameSchema) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for column in schema.columns:
        if column.kind in ("decimal", "datetime_utc"):
            dtypes[column.name] = "string"
        elif column.kind == "bool":
            dtypes[column.name] = "boolean"
        else:
            dtypes[column.name] = column.dtype
    return dtypes


def _parse_utc(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame
    for name in columns:
        result = result.assign(**{name: pd.to_datetime(result[name], utc=True).astype("datetime64[ns, UTC]")})
    return result
