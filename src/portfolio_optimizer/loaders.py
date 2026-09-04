"""Dataset loaders — yours to edit.

A loader is an ordinary function ``(request: LoadRequest, params: P) -> pd.DataFrame`` named in the
run config; it may be ``async def``. Every dataset is a frame, so that is the only shape. Loaders are
the only place a network call, a database query, or a file read belongs; everything downstream is pure.

The engine loads the datasets as the dependency DAG the config declares: each starts the moment the
datasets its entry depends on have loaded, one with no dependencies starts immediately, and an
``async def`` loader runs as a task on the event loop while a plain one runs in a worker thread, so a
blocking driver never stalls the loop. A loader never counts its own requests: the config's
``batch_size`` cuts the book into calls and ``max_in_flight`` bounds how many of them run at once.

The loaders here stand in for the services a desk actually has: a book of record, a custodian's
position service, a security master, a research store, the account-master database, a compliance
service, a trade blotter, and a parameter store. Each waits as long as its own source would, drawn from the bands below, and then
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
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from pydantic import Field, model_validator

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.frames import ColumnSpec, FrameSchema, coerce_frame, empty_frame
from portfolio_optimizer.domain.schemas import CONSTRAINTS, DETAILS, HOLDINGS, ORDER_SIDES, ORDERS, PORTFOLIOS, UNIVERSE
from portfolio_optimizer.domain.types import Params, PortfolioId


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
"""Every tradable name with its reference data. The slowest input in the run by an order of magnitude."""

RESEARCH = Latency(2.0, 6.0)
"""What the desk's models say about each name today: one file, published once a day, read once a run."""

ACCOUNT_MASTER = Latency(1.0, 6.0)
"""A batch of accounts from the firm's own database: one query however many ids it is given."""

COMPLIANCE = Latency(2.0, 9.0)
"""Every account's mandate rules, in the vocabulary the desk writes them in."""

PARAMETER_STORE = Latency(0.5, 1.5)
"""A named set of runtime settings. Small, cached, and asked for more than once a run."""

BLOTTER = Latency(1.0, 4.0)
"""What the book traded recently: one query over the accounts asked for."""

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
    """What each account in the request owns, one tax lot per row: quantity, average cost, and the date the lot was acquired.

    The shape of a custodian that answers one account per call: one request per id in the batch, run
    together, so the loader is right however the engine batches it — under the example's
    ``batch_size: 1`` and ``max_in_flight: 8`` that is eight accounts in flight across the book.
    """
    latency = params.latency(CUSTODIAN)
    positions = _table(request, "holdings.csv", HOLDINGS)  # the mock's store; a real client holds a connection here instead

    async def one(portfolio_id: PortfolioId) -> pd.DataFrame:
        await asyncio.sleep(latency.draw(f"{request.run_id}:{request.dataset}:{portfolio_id}"))
        return _rows_for(positions, (portfolio_id,))

    parts = await asyncio.gather(*(one(portfolio_id) for portfolio_id in request.portfolio_ids))
    return pd.concat(parts, ignore_index=True) if parts else empty_frame(HOLDINGS)


async def load_universe(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """Every security the book may trade, with its price, sector, liquidity, lot size, and restricted flag.

    One call for the whole run, and the long pole of the load stage: a security-master scan is tens of
    seconds and every other input finishes inside it. That is the argument for loading concurrently —
    the per-account calls cost nothing while this one is outstanding.
    """
    await _call(request, params.latency(SECURITY_MASTER))
    return _table(request, "universe.csv", UNIVERSE)


SIGNALS = FrameSchema(name="signals", columns=(ColumnSpec("security_id", "string"), ColumnSpec("alpha", "Float64"), ColumnSpec("tcost_bps", "decimal", ge=Decimal(0))), key=("security_id",))
"""The shape of the research store's answer: expected return and cost per security, typed by the loader that fetches it."""


async def load_signals(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """What research says about every name: its expected return and the cost of trading it.

    A security master does not know a name's alpha, so the universe arrives in two parts from two
    services. This one is an extra dataset the engine does not interpret: the example's assembly
    ``join`` puts its columns on the universe, on ``security_id`` under ``one_to_one`` with every
    universe name required to match, then drops it. The join is where the run learns a name has no
    signal, before any account is built.
    """
    await _call(request, params.latency(RESEARCH))
    return _table(request, "signals.csv", SIGNALS)


def load_details(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """The account master for the accounts in the request: NAV, cash, tax rates, and the style limits.

    A plain ``def`` because the driver blocks: the engine runs it in a worker thread, and bounds how
    many of those threads its calls occupy. One query however many ids it is given — the shape
    ``batch_size`` in the config exists for.
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


MANDATES = FrameSchema(name="mandates", columns=(ColumnSpec("portfolio_id", "string"), ColumnSpec("sector", "string")), key=("portfolio_id", "sector"))
"""The shape of a mandate: which sectors each account may trade, one row per allowed sector — an extra dataset typed by the loader that fetches it."""


async def load_mandates(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """Which sectors each account may trade, from the same compliance service the constraints come from.

    An extra dataset the engine does not interpret: it is carried to each portfolio's bundle, where
    :func:`~portfolio_optimizer.rules.restrict_to_mandate` freezes every universe name outside the
    account's sectors — the restriction-list shape that partitions a book into independent components
    of the dependency graph. Declare ``depends_on: ["portfolios"]`` so the call receives the book's ids.
    """
    await _call(request, params.latency(COMPLIANCE))
    return _rows_for(_table(request, "mandates.csv", MANDATES), request.portfolio_ids)


TRADES = FrameSchema(
    name="trades",
    columns=(
        ColumnSpec("portfolio_id", "string"),
        ColumnSpec("security_id", "string"),
        ColumnSpec("side", "string", allowed=ORDER_SIDES),
        ColumnSpec("traded_on", "datetime_utc"),
        ColumnSpec("quantity", "Int64", required=False, gt=Decimal(0)),
    ),
    key=(),
)
"""The shape of a trade record: which account traded which name, which way, when, and — where the source says — how much; an extra dataset typed by the loader that fetches it. No key: an account may trade a name twice in a day."""


async def load_trades(request: LoadRequest, params: ServiceParams) -> pd.DataFrame:
    """What each account traded recently, from the desk's blotter: one row per fill.

    An extra dataset the engine does not interpret: it is carried to each portfolio's bundle, where
    :func:`~portfolio_optimizer.rules.restrict_recent_trades` freezes every name the account traded
    inside the wash-sale window, so a loss the outflow harvested is not bought straight back. Declare
    ``depends_on: ["portfolios"]`` so the call receives the book's ids.
    """
    await _call(request, params.latency(BLOTTER))
    return _rows_for(_table(request, "trades.csv", TRADES), request.portfolio_ids)


class RunOrdersParams(ServiceParams):
    """Parameters for :func:`load_run_orders`: which run's orders, and in which of two shapes."""

    path: str = Field(
        min_length=1,
        description="The orders file a previous run's sink wrote (`orders.csv` or `orders.parquet`), relative to the data root. It names the predecessor, so it is part of the config hash: a run fed by a different run is a different run.",
    )
    emit: Literal["trades", "adv_consumed"] = Field(
        default="trades",
        description='`trades`: the orders as blotter rows (`portfolio_id`, `security_id`, `side`, `quantity`, `traded_on`), the shape `restrict_recent_trades` reads. `adv_consumed`: one row per universe security with `adv_consumed_quantity`, the quantity the run traded in it on either side, for an assembly `join` into the universe, where the standard build takes it off each name\'s participation budget; needs `depends_on: ["universe"]`.',
    )


async def load_run_orders(request: LoadRequest, params: RunOrdersParams) -> pd.DataFrame:
    """A previous run's orders as this run's input: the blotter the wash-sale rule reads, or the ADV its trades consumed.

    The handoff between an outflow and the inflow that follows it, or between a run and the retry of
    part of it, is data and nothing else: the orders the first run's sink wrote are loaded, hashed,
    and recorded like any dataset, so the second run stays a pure function of a snapshot and
    ``diff-manifests`` names the predecessor's orders as the input that changed. ``trades`` is the
    orders frame as trade rows — a run's ``as_of_date`` is when it traded — for the accounts asked
    for; ``adv_consumed`` is the quantity traded per security, either side, over every universe security
    (zero where none). Both are sorted, since an extra dataset hashes by row order.
    """
    await _call(request, params.latency(BLOTTER))
    orders = _read_orders(request.data_root / params.path)
    if params.emit == "trades":
        trades = orders.rename(columns={"as_of_date": "traded_on"})[["portfolio_id", "security_id", "side", "quantity", "traded_on"]]
        if request.portfolio_ids:
            trades = _rows_for(trades, request.portfolio_ids)
        return coerce_frame(trades.sort_values(["portfolio_id", "security_id", "side"], kind="stable").reset_index(drop=True), TRADES)
    try:
        universe = request.inputs["universe"]
    except KeyError:
        msg = 'load_run_orders with emit: adv_consumed needs the universe to name every security; declare depends_on: ["universe"] on the dataset'
        raise ValueError(msg) from None
    consumed = orders.groupby("security_id")["quantity"].sum()
    ids = sorted(str(value) for value in universe["security_id"])
    return pd.DataFrame({"security_id": pd.Series(ids, dtype="string"), "adv_consumed_quantity": pd.Series([int(consumed.get(security, 0)) for security in ids], dtype="Int64")})


def _read_orders(path: Path) -> pd.DataFrame:
    """An orders file as the ``ORDERS`` frame, whichever of the shipped sinks wrote it."""
    if path.suffix == ".parquet":
        return coerce_frame(pd.read_parquet(path), ORDERS)
    return _read_csv(path, ORDERS)


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
    """Wait as long as the source would take to answer."""
    await asyncio.sleep(latency.draw(f"{request.run_id}:{request.dataset}:{key}"))


def _call_blocking(request: LoadRequest, latency: Latency, key: str = "") -> None:
    """The same wait from a worker thread, where a blocking driver's call would be."""
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
