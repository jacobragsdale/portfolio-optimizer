"""Functions a JSON config names as steps: no-ops that let a config resolve, liars the engine must catch, and loaders that prove the load stage's plumbing.

Nothing here is a test. A config refers to these as ``tests.steps:<name>``.
"""

import asyncio
import threading
from decimal import Decimal

import numpy as np
import pandas as pd

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, scale, total
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.results import Artifact, ProblemSpec
from portfolio_optimizer.loaders import CsvPerPortfolioParams, csv_per_portfolio
from portfolio_optimizer.solving import SolveRequest, SolveResult

# --- steps that satisfy the resolver's contracts, for tests that need a resolvable config ---


def noop_term(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:
    """A zero objective: lets a config resolve and solve without exercising a real term."""
    del spec
    return ObjectiveTerm("noop", scale(0.0, total(x.w)))


def lying_term(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:
    """Annotated as a term but returns a constraint set; solve must catch it."""
    del x, spec
    return ConstraintSet("lie", ())  # ty: ignore[invalid-return-type]  # the lie is the case under test


def hold_still(request: SolveRequest) -> SolveResult:
    """A solve step that is not an optimizer: the resting portfolio is the answer, and the terms are never touched."""
    return SolveResult(w=request.spec.w0)


def wrong_shape(request: SolveRequest) -> SolveResult:
    """A solve step whose weights are not aligned to the spec; the engine must refuse it."""
    return SolveResult(w=np.zeros(request.spec.n + 1))


def noop_sink(orders: pd.DataFrame, io: IoContext) -> tuple[Artifact, ...]:
    """Publish nothing."""
    del orders, io
    return ()


def lying_loader(request: LoadRequest) -> pd.DataFrame:
    """Annotated as a frame loader but returns a dict; the engine must catch it."""
    del request
    return {}  # ty: ignore[invalid-return-type]  # the lie is the case under test


def lying_rule(data: PortfolioData) -> PortfolioData:
    """Annotated correctly but returns a frame; the pipeline must catch it."""
    return data.universe  # ty: ignore[invalid-return-type]  # the lie is the case under test


def lying_assembly_step(frames: Frames) -> Frames:
    """Annotated as an assembly step but returns a frame; the engine must catch it."""
    return frames["universe"]  # ty: ignore[invalid-return-type]  # the lie is the case under test


def score_by_price(frames: Frames) -> Frames:
    """A custom assembly step: attach a ``Float64`` analytics column to both holdings and universe from the prices dataset."""
    scores = frames["prices"].assign(score=frames["prices"]["price"].map(float).astype("Float64")).drop(columns=["price"])
    holdings = frames["holdings"].merge(scores, on="security_id", how="left", validate="many_to_one")
    universe = frames["universe"].merge(scores, on="security_id", how="left", validate="one_to_one")
    return frames.with_frame("holdings", holdings).with_frame("universe", universe)


def refuse_assembly(frames: Frames) -> Frames:
    """An assembly step whose precondition fails."""
    del frames
    msg = "vendor scores are stale"
    raise ValueError(msg)


def buy_only_listed(data: PortfolioData) -> PortfolioData:
    """Cap every security not on the portfolio's ``buy_list`` extras dataset at its current weight: the shape of a real buy-universe filter."""
    listed = {str(security) for security in data.extras["buy_list"]["security_id"]}
    prices = {str(security): price for security, price in zip(data.universe["security_id"], data.universe["price"], strict=True)}
    held = {str(security): int(quantity) for security, quantity in zip(data.holdings["security_id"], data.holdings["quantity"], strict=True)}
    caps = [None if security in listed else Decimal(held.get(security, 0)) * prices[security] / data.details.nav for security in (str(value) for value in data.universe["security_id"])]
    return data.with_changes(universe=data.universe.assign(max_weight=pd.Series(caps, index=data.universe.index, dtype="object")))


# --- loaders that prove the load stage's concurrency and plumbing ---

_SYNC_BARRIER = threading.Barrier(2, timeout=5)
_ASYNC_BARRIER: tuple[asyncio.AbstractEventLoop, asyncio.Barrier] | None = None


def barrier_loader(request: LoadRequest) -> pd.DataFrame:
    """Returns only once a second barrier loader is running at the same time — proof that sync loaders overlap."""
    _SYNC_BARRIER.wait()
    return pd.DataFrame({"portfolio_id": pd.Series(list(request.portfolio_ids), dtype="string")})


async def async_barrier_loader(request: LoadRequest) -> pd.DataFrame:
    """The async twin of ``barrier_loader``; two of these must be in flight together."""
    global _ASYNC_BARRIER  # noqa: PLW0603  # one barrier per event loop, shared by the two loaders of a test
    loop = asyncio.get_running_loop()
    if _ASYNC_BARRIER is None or _ASYNC_BARRIER[0] is not loop:
        _ASYNC_BARRIER = (loop, asyncio.Barrier(2))
    await asyncio.wait_for(_ASYNC_BARRIER[1].wait(), timeout=5)
    return pd.DataFrame({"portfolio_id": pd.Series(list(request.portfolio_ids), dtype="string")})


def pool_reporting_loader(request: LoadRequest) -> pd.DataFrame:
    """Reports which rate-limit pool the engine handed it, and takes one turn from it."""
    with request.rate_limiter.sync:
        return pd.DataFrame({"pool": pd.Series([request.rate_limiter.name], dtype="string"), "limited": [request.rate_limiter.is_limited]})


async def async_pool_reporting_loader(request: LoadRequest) -> pd.DataFrame:
    """The async twin of ``pool_reporting_loader``."""
    async with request.rate_limiter:
        return pd.DataFrame({"pool": pd.Series([request.rate_limiter.name], dtype="string"), "limited": [request.rate_limiter.is_limited]})


def invalid_input_loader(request: LoadRequest) -> pd.DataFrame:
    """A loader whose source rejected the request."""
    msg = f"{request.dataset}: no rows as of {request.as_of:%Y-%m-%d}"
    raise ValueError(msg)


def unreachable_loader(request: LoadRequest) -> pd.DataFrame:
    """A loader whose backend is down; an infrastructure failure, not an input problem."""
    msg = f"{request.dataset}: connection refused"
    raise ConnectionError(msg)


def limiter_naming_portfolios_loader(request: LoadRequest) -> pd.DataFrame:
    """A portfolio list whose single id is the name of the limiter the engine handed this input."""
    with request.rate_limiter.sync:
        return pd.DataFrame({"portfolio_id": pd.Series([request.rate_limiter.name], dtype="string"), "solve_order": pd.Series([0], dtype="Int64")})


def last_portfolio_id_first(data: PortfolioData) -> Decimal:
    """A solve-order key that reverses the portfolios frame's order, so a test can tell the two apart."""
    return -Decimal(data.portfolio_id.removeprefix("P"))


BATCHES: list[tuple[str, ...]] = []
"""Every batch of ids the engine handed :func:`recording_csv`, in call order; a test clears it first."""


class RecordingCsvParams(CsvPerPortfolioParams):
    """:class:`CsvPerPortfolioParams` plus the portfolios this source has no data for."""

    fails_for: tuple[str, ...] = ()


async def recording_csv(request: LoadRequest, params: RecordingCsvParams) -> pd.DataFrame:
    """A per-portfolio loader that records the batch it was given, reads that batch's rows, and refuses a batch holding a portfolio it has no data for."""
    BATCHES.append(tuple(request.portfolio_ids))
    refused = sorted(set(request.portfolio_ids) & set(params.fails_for))
    if refused:
        msg = f"{request.dataset}: no data for {refused}"
        raise ValueError(msg)
    return await csv_per_portfolio(request, params)
