"""Functions a JSON config names as steps: no-ops that let a config resolve, liars the engine must catch, and loaders that prove the load stage's plumbing.

Nothing here is a test. A config refers to these as ``tests.steps:<name>``.
"""

import asyncio
import threading
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, scale, total
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.frames import ColumnSpec, FrameSchema, coerce_frame
from portfolio_optimizer.domain.results import Artifact, ProblemSpec
from portfolio_optimizer.loaders import ServiceParams, load_holdings
from portfolio_optimizer.rules import parameter
from portfolio_optimizer.solvers import cvxpy
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
    """A custom assembly step: attach a ``Float64`` analytics column to both holdings and universe, derived from the universe's price."""
    universe_prices = frames["universe"][["security_id", "price"]]
    scores = universe_prices.assign(score=universe_prices["price"].map(float).astype("Float64")).drop(columns=["price"])
    holdings = frames["holdings"].merge(scores, on="security_id", how="left", validate="many_to_one")
    universe = frames["universe"].merge(scores, on="security_id", how="left", validate="one_to_one")
    return frames.with_frame("holdings", holdings).with_frame("universe", universe)


def cvxpy_reporting_a_runtime_parameter(request: SolveRequest) -> SolveResult:
    """The shipped cvxpy step, naming the runtime parameter it read as its solver: proof an extra dataset reaches this seam, and the shape of a step driven by one."""
    value = parameter(request.extras["global_parameters"], "risk_aversion")
    return replace(cvxpy(request), solver=f"risk_aversion={value}")


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


def barrier_portfolios_loader(request: LoadRequest) -> pd.DataFrame:
    """A book of record that answers only once another barrier loader is in flight — proof the book no longer loads alone."""
    del request
    _SYNC_BARRIER.wait()
    return pd.DataFrame({"portfolio_id": pd.Series(["P1", "P2"], dtype="string"), "solve_order": pd.Series([0, 1], dtype="Int64")})


def inputs_reporting_loader(request: LoadRequest) -> pd.DataFrame:
    """Report what the engine handed this loader: each input frame's name and row count, and the ids of the call."""
    names = sorted(request.inputs)
    return pd.DataFrame(
        {
            "input": pd.Series(names, dtype="string"),
            "rows": pd.Series([len(request.inputs[name]) for name in names], dtype="Int64"),
            "ids": pd.Series([",".join(request.portfolio_ids)] * len(names), dtype="string"),
        }
    )


INPUT_VIEWS: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []
"""Per call of ``recording_inputs_holdings``: the batch's ids, the ids in its ``portfolios`` input, and its ``universe`` input's rows. Cleared by each test that uses it."""


async def recording_inputs_holdings(request: LoadRequest) -> pd.DataFrame:
    """The custodian mock, recording the view of its inputs each batch received."""
    portfolios = request.inputs["portfolios"]
    INPUT_VIEWS.append((tuple(str(value) for value in request.portfolio_ids), tuple(str(value) for value in portfolios["portfolio_id"]), len(request.inputs["universe"])))
    return await load_holdings(request, ServiceParams(min_latency_s=0, max_latency_s=0))


async def async_barrier_loader(request: LoadRequest) -> pd.DataFrame:
    """The async twin of ``barrier_loader``; two of these must be in flight together."""
    global _ASYNC_BARRIER  # noqa: PLW0603  # one barrier per event loop, shared by the two loaders of a test
    loop = asyncio.get_running_loop()
    if _ASYNC_BARRIER is None or _ASYNC_BARRIER[0] is not loop:
        _ASYNC_BARRIER = (loop, asyncio.Barrier(2))
    await asyncio.wait_for(_ASYNC_BARRIER[1].wait(), timeout=5)
    return pd.DataFrame({"portfolio_id": pd.Series(list(request.portfolio_ids), dtype="string")})


PEAK_IN_FLIGHT: dict[str, int] = {}
"""The most calls of each dataset the engine had running at once; a test clears it first."""

_IN_FLIGHT: dict[str, int] = {}


async def in_flight_recording_holdings(request: LoadRequest) -> pd.DataFrame:
    """Records how many of this dataset's calls the engine ran at once, and yields so they can overlap."""
    dataset = request.dataset
    _IN_FLIGHT[dataset] = _IN_FLIGHT.get(dataset, 0) + 1
    PEAK_IN_FLIGHT[dataset] = max(PEAK_IN_FLIGHT.get(dataset, 0), _IN_FLIGHT[dataset])
    try:
        await asyncio.sleep(0.01)
        return await load_holdings(request, ServiceParams(min_latency_s=0, max_latency_s=0))
    finally:
        _IN_FLIGHT[dataset] -= 1


def invalid_input_loader(request: LoadRequest) -> pd.DataFrame:
    """A loader whose source rejected the request."""
    msg = f"{request.dataset}: no rows as of {request.as_of_date:%Y-%m-%d}"
    raise ValueError(msg)


def unreachable_loader(request: LoadRequest) -> pd.DataFrame:
    """A loader whose backend is down; an infrastructure failure, not an input problem."""
    msg = f"{request.dataset}: connection refused"
    raise ConnectionError(msg)


BUY_LIST = FrameSchema(name="buy_list", columns=(ColumnSpec("portfolio_id", "string"), ColumnSpec("security_id", "string")), key=("portfolio_id", "security_id"))
"""The shape of an extra dataset a desk loads for itself: the engine knows nothing about it, so its loader types it."""


def load_buy_list(request: LoadRequest) -> pd.DataFrame:
    """The securities each account may buy, from its own source."""
    return coerce_frame(pd.read_csv(request.data_root / "buy_list.csv", dtype={"portfolio_id": "string", "security_id": "string"}), BUY_LIST)


def last_portfolio_id_first(data: PortfolioData) -> Decimal:
    """A solve-order key that reverses the portfolios frame's order, so a test can tell the two apart."""
    return -Decimal(data.portfolio_id.removeprefix("P"))


BATCHES: list[tuple[str, ...]] = []
"""Every batch of ids the engine handed :func:`recording_holdings`, in call order; a test clears it first."""


class RecordingLoaderParams(ServiceParams):
    """:class:`ServiceParams` plus the portfolios this source has no data for."""

    fails_for: tuple[str, ...] = ()


async def recording_holdings(request: LoadRequest, params: RecordingLoaderParams) -> pd.DataFrame:
    """A per-portfolio loader that records the batch it was given, fetches that batch's rows, and refuses a batch holding a portfolio it has no data for."""
    BATCHES.append(tuple(request.portfolio_ids))
    refused = sorted(set(request.portfolio_ids) & set(params.fails_for))
    if refused:
        msg = f"{request.dataset}: no data for {refused}"
        raise ValueError(msg)
    return await load_holdings(request, params)
