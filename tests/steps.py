"""Functions and kinds a JSON config names: no-ops that let a config resolve, liars the engine must catch, loaders that prove the load stage's plumbing, and custom term kinds.

Nothing here is a test. A config refers to the steps as ``tests.steps:<name>``; the kinds are
registered in the process, which is what an installed package's entry point would do.
"""

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal
from typing import Literal, override

import numpy as np
import pandas as pd
from pydantic import Field

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, dot, scale, sum_squares, weighted
from portfolio_optimizer.domain.constraints import Vector, vector_values
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.frames import ColumnSpec, FrameSchema, coerce_frame
from portfolio_optimizer.domain.objective import TypedTerm, register_term_kind
from portfolio_optimizer.domain.order_flow import OrderFlowProfile
from portfolio_optimizer.domain.results import Artifact, ChainState, MissingSpecColumnError, ProblemSpec, Solution
from portfolio_optimizer.engine.build import standard
from portfolio_optimizer.loaders import ServiceParams, load_holdings
from portfolio_optimizer.rules import parameter
from portfolio_optimizer.solvers import CvxpyParams, cvxpy
from portfolio_optimizer.solving import SolveRequest, SolveResult

# --- term kinds a package might publish: a convex one, one that reads the chain, and liars ---


@register_term_kind
class Quadratic(TypedTerm):
    """``weight · Σ dᵢ vᵢ²`` for a per-security column ``d``: the shape of a diagonal risk penalty, and proof a convex kind verifies like a linear one."""

    kind: Literal["quadratic"] = "quadratic"
    column: str = Field(min_length=1, description="The per-security column of penalties.")
    vector: Vector = Field(default="w", description="The decision vector the penalty is over.")

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        try:
            spec.column(self.column)
        except MissingSpecColumnError as error:
            yield str(error)

    @override
    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        del chain, profile
        return float(self.weight) * float((spec.column(self.column) * vector_values(solution, self.vector) ** 2).sum())

    @override
    def to_cvxpy(self, x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
        del chain
        return ObjectiveTerm(self.name, scale(float(self.weight), sum_squares(weighted(np.sqrt(spec.column(self.column)), x.vector(self.vector)))))


@register_term_kind
class ChainPenalty(TypedTerm):
    """``weight · (traded_quantity · w)``: a term that reads the chain, so a run configuring it couples every portfolio through its whole tradable set."""

    kind: Literal["chain_penalty"] = "chain_penalty"
    reads_chain = True

    @override
    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        del spec, profile
        return float(self.weight) * float((chain.traded_quantity * solution.w).sum())

    @override
    def to_cvxpy(self, x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
        del spec
        return ObjectiveTerm(self.name, scale(float(self.weight), dot(chain.traded_quantity, x.w)))


@register_term_kind
class Lying(TypedTerm):
    """Renders a constraint set where a term is expected; the resolver must catch it."""

    kind: Literal["lying"] = "lying"

    @override
    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        del spec, solution, chain, profile
        return 0.0

    @override
    def to_cvxpy(self, x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
        del x, spec, chain
        return ConstraintSet("lie", ())  # ty: ignore[invalid-return-type]  # the lie is the case under test


@register_term_kind
class Raising(TypedTerm):
    """Raises when rendered: what dry construction at resolve exists to report."""

    kind: Literal["raising"] = "raising"

    @override
    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        del spec, solution, chain, profile
        return 0.0

    @override
    def to_cvxpy(self, x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
        del x, spec, chain
        msg = "no such column 'beta' in the risk model"
        raise RuntimeError(msg)


# --- steps that satisfy the resolver's contracts, for tests that need a resolvable config ---


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


def add_a_participation_row(data: PortfolioData) -> PortfolioData:
    """Append a chain-reading row a rule has no business adding: what a run planned without a chain must refuse at build."""
    row = pd.DataFrame(
        {
            "portfolio_id": pd.Series([data.details.portfolio_id], dtype="string"),
            "kind": pd.Series(["participation_limit"], dtype="string"),
            "label": pd.Series(["adv_from_rule"], dtype="string"),
            "params": pd.Series(['{"direction": "<="}'], dtype="string"),
        }
    )
    return data.with_changes(constraints=pd.concat([data.constraints, row], ignore_index=True))


def lying_rule(data: PortfolioData) -> PortfolioData:
    """Annotated correctly but returns a frame; the pipeline must catch it."""
    return data.universe  # ty: ignore[invalid-return-type]  # the lie is the case under test


def lying_assembly_step(frames: Frames) -> Frames:
    """Annotated as an assembly step but returns a frame; the engine must catch it."""
    return frames["universe"]  # ty: ignore[invalid-return-type]  # the lie is the case under test


def lying_build(data: PortfolioData) -> ProblemSpec:
    """Annotated as a build step but returns the bundle; the engine must catch it."""
    return data  # ty: ignore[invalid-return-type]  # the lie is the case under test


def build_with_a_scalar(data: PortfolioData) -> ProblemSpec:
    """The standard build plus one account scalar a constraint row can bound against: the shape of a custom build."""
    spec = standard(data)
    return replace(spec, scalars={**spec.scalars, "max_names": 2.0})


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
    return replace(cvxpy(request, CvxpyParams()), solver=f"risk_aversion={value}")


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


# --- checks: what a breached rule, an unexercised one, and a broken one look like to the run ---


def failing_check(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame) -> pd.DataFrame:
    """A check every order fails: what a breached business rule looks like to the run."""
    del frames, solved
    return orders[["portfolio_id", "security_id", "side"]].assign(ok=False)


def empty_check(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame) -> pd.DataFrame:
    """A check the book never reaches: it examines nothing, so it proves nothing."""
    del frames, orders, solved
    return pd.DataFrame({"portfolio_id": pd.Series(dtype="string"), "ok": pd.Series(dtype="bool")})


def raising_check(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame) -> pd.DataFrame:
    """A check that is itself broken; the run records that, not a verdict on the orders."""
    del frames, orders, solved
    msg = "the check itself is broken"
    raise RuntimeError(msg)


def lying_check(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame) -> pd.DataFrame:
    """Annotated as returning examined rows but returns a dict; the engine must catch it."""
    del frames, orders, solved
    return {}  # ty: ignore[invalid-return-type]  # the lie is the case under test
