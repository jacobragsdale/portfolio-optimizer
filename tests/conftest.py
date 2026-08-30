"""Frame builders derived from the boundary schemas, and small domain-object factories.

Every builder takes partial rows and fills unstated fields with valid defaults, so a test
states only what it is about. Dtypes come from the schema, so fixtures cannot drift from it.
Builders are exposed as fixtures because ``--import-mode=importlib`` keeps ``tests/`` off
``sys.path``.
"""

import asyncio
import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from distributed import LocalCluster

from portfolio_optimizer.config.models import RunConfig, load_run_config
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, scale, total
from portfolio_optimizer.domain.data import Frames as DatasetFrames
from portfolio_optimizer.domain.data import IoContext, LoadRequest, PortfolioData, PortfolioDetails, StyleConstraints
from portfolio_optimizer.domain.frames import FrameSchema
from portfolio_optimizer.domain.results import F64, Artifact, ProblemSpec
from portfolio_optimizer.domain.schemas import DETAILS, HOLDINGS, ORDERS, PORTFOLIOS, TARGETS, UNIVERSE
from portfolio_optimizer.settings import ExecutionSettings
from portfolio_optimizer.solving import SolveRequest, SolveResult

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_run.json"
EXAMPLE_DATA = REPO_ROOT / "examples" / "data"

AS_OF = datetime(2026, 8, 28, tzinfo=UTC)
ACQUIRED = datetime(2024, 1, 15, tzinfo=UTC)

type Builder = Callable[..., pd.DataFrame]
type Row = Mapping[str, object]


def _frame(schema: FrameSchema, defaults: Row, rows: tuple[Row, ...]) -> pd.DataFrame:
    records = [dict(defaults) | dict(row) for row in rows or ({},)]
    frame = pd.DataFrame.from_records(records)
    return frame.astype({name: dtype for name, dtype in schema.dtypes.items() if name in frame.columns})


def empty_frame(schema: FrameSchema) -> pd.DataFrame:
    """A zero-row frame with the schema's required columns and dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=schema.column(name).dtype) for name in schema.required_columns})


_DEFAULTS: dict[str, Row] = {
    PORTFOLIOS.name: {"portfolio_id": "P1", "solve_order": 0},
    DETAILS.name: {
        "portfolio_id": "P1",
        "name": "Portfolio One",
        "state": "NY",
        "st_tax_rate": Decimal("0.40"),
        "lt_tax_rate": Decimal("0.20"),
        "cash": Decimal(0),
        "nav": Decimal(1000000),
        "benchmark_id": "B1",
    },
    HOLDINGS.name: {"portfolio_id": "P1", "security_id": "A", "quantity": 5000, "avg_cost": Decimal(90), "acquired_on": ACQUIRED},
    UNIVERSE.name: {"security_id": "A", "price": Decimal(100), "sector": "TECH", "adv_shares": 1_000_000, "lot_size": 1, "restricted": False},
    TARGETS.name: {"benchmark_id": "B1", "security_id": "A", "weight": Decimal(1)},
    ORDERS.name: {
        "portfolio_id": "P1",
        "security_id": "A",
        "side": "BUY",
        "quantity": 10,
        "reference_price": Decimal(100),
        "notional": Decimal(1000),
        "target_weight": 0.1,
        "unrounded_shares": 10.0,
        "spec_hash": "0" * 64,
        "run_id": "run-1",
        "as_of": AS_OF,
    },
}


def build(schema: FrameSchema, *rows: Row) -> pd.DataFrame:
    """Build a frame for ``schema`` from partial rows."""
    return _frame(schema, _DEFAULTS[schema.name], rows)


@dataclass(frozen=True, slots=True)
class Frames:
    """Per-schema frame builders."""

    def for_schema(self, schema: FrameSchema) -> Builder:
        """The builder for ``schema``."""
        return lambda *rows: build(schema, *rows)

    def portfolios(self, *rows: Row) -> pd.DataFrame:
        """Build a ``portfolios`` frame."""
        return build(PORTFOLIOS, *rows)

    def details(self, *rows: Row) -> pd.DataFrame:
        """Build a ``details`` frame."""
        return build(DETAILS, *rows)

    def holdings(self, *rows: Row) -> pd.DataFrame:
        """Build a ``holdings`` frame."""
        return build(HOLDINGS, *rows)

    def universe(self, *rows: Row) -> pd.DataFrame:
        """Build a ``universe`` frame."""
        return build(UNIVERSE, *rows)

    def targets(self, *rows: Row) -> pd.DataFrame:
        """Build a ``targets`` frame."""
        return build(TARGETS, *rows)

    def orders(self, *rows: Row) -> pd.DataFrame:
        """Build an ``orders`` frame."""
        return build(ORDERS, *rows)

    def three_security_universe(self) -> pd.DataFrame:
        """Securities A, B, C at 100, 50, 10 in one sector; C has limited ADV."""
        return self.universe(
            {"security_id": "A", "price": Decimal(100), "adv_shares": 1_000_000},
            {"security_id": "B", "price": Decimal(50), "adv_shares": 1_000_000},
            {"security_id": "C", "price": Decimal(10), "adv_shares": 100_000},
        )

    def equal_weight_targets(self) -> pd.DataFrame:
        """Benchmark B1: one third in each of A, B, C."""
        third = Decimal(1) / Decimal(3)
        return self.targets({"security_id": "A", "weight": third}, {"security_id": "B", "weight": third}, {"security_id": "C", "weight": Decimal(1) - 2 * third})


def make_details(**overrides: object) -> PortfolioDetails:
    """A valid ``PortfolioDetails`` with optional field overrides."""
    return PortfolioDetails.model_validate(dict(_DEFAULTS[DETAILS.name]) | overrides)


def make_style(**overrides: object) -> StyleConstraints:
    """A permissive ``StyleConstraints`` with optional field overrides."""
    base: dict[str, object] = {
        "max_weight": Decimal(1),
        "max_turnover": Decimal(2),
        "min_trade_notional": Decimal(0),
        "cash_bounds": (Decimal(0), Decimal(0)),
        "max_adv_participation": Decimal(1),
        "sector_bounds": {},
    }
    return StyleConstraints.model_validate(base | overrides)


def make_portfolio_data(
    *,
    details: PortfolioDetails | None = None,
    holdings: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
    style: StyleConstraints | None = None,
    as_of: datetime = AS_OF,
    extras: Mapping[str, pd.DataFrame] | None = None,
    prevalidated: frozenset[str] = frozenset(),
) -> PortfolioData:
    """The canonical small bundle: P1 holds A 5000 and B 10000 against the three-security universe."""
    frames = Frames()
    return PortfolioData(
        details=details if details is not None else make_details(),
        holdings=holdings if holdings is not None else frames.holdings({"security_id": "A", "quantity": 5000}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(60)}),
        universe=universe if universe is not None else frames.three_security_universe(),
        targets=targets if targets is not None else frames.equal_weight_targets(),
        style=style if style is not None else make_style(),
        as_of=as_of,
        extras=extras if extras is not None else {},
        prevalidated=prevalidated,
    )


def make_spec(n: int = 3, **overrides: object) -> ProblemSpec:
    """A feasible spec with ``n`` securities, identity-like data, and no binding limits."""
    ids = tuple(f"S{i}" for i in range(n))
    zeros: F64 = np.zeros(n)
    base: dict[str, object] = {
        "portfolio_id": "P1",
        "as_of": AS_OF,
        "security_ids": ids,
        "sector_names": ("TECH",),
        "nav": 1_000_000.0,
        "w0": np.full(n, 1.0 / n) if n else zeros,
        "price": np.full(n, 100.0),
        "shares_held": np.full(n, 1_000_000.0 / n / 100.0) if n else zeros,
        "lot_size": np.ones(n),
        "w_target": np.full(n, 1.0 / n) if n else zeros,
        "tax_per_dollar": zeros,
        "tcost_per_dollar": zeros,
        "lb": zeros,
        "ub": np.ones(n),
        "adv_capacity": np.full(n, 10.0),
        "sector_matrix": np.ones((1, n)),
        "sector_lb": np.zeros(1),
        "sector_ub": np.ones(1),
        "max_turnover": 2.0,
        "cash_lb": 0.0,
        "cash_ub": 0.0,
        "min_trade_notional": 0.0,
    }
    return ProblemSpec(**(base | overrides))  # ty: ignore[invalid-argument-type]  # the merged mapping is typed `object`; the dataclass validates every field on construction


@dataclass(frozen=True, slots=True)
class Factories:
    """Domain-object factories handed to tests as one fixture."""

    details: Callable[..., PortfolioDetails]
    style: Callable[..., StyleConstraints]
    portfolio_data: Callable[..., PortfolioData]
    spec: Callable[..., ProblemSpec]


@pytest.fixture
def frames() -> Frames:
    """Per-schema frame builders."""
    return Frames()


@pytest.fixture
def make() -> Factories:
    """Domain-object factories."""
    return Factories(details=make_details, style=make_style, portfolio_data=make_portfolio_data, spec=make_spec)


NO_CHAIN_CONSTRAINTS = ["long_only", "max_weight", "cash_bounds", "turnover_cap", "sector_bounds"]
"""The example's constraints without the chain-aware ADV cap: nothing reads the chain, so no portfolio waits for another."""
BUY_ONLY_OBJECTIVE: dict[str, object] = {"terms": [{"name": "tracking_error", "params": {"weight": "1.0"}}, {"name": "transaction_cost", "params": {"weight": "1.0"}}]}
"""The example's objective without ``tax_cost``, which reads ``sell`` and so cannot run in a buy-only run."""


def half_cash_book(tmp_path: Path) -> Path:
    """The example data with each portfolio holding A 2500 @100 and B 5000 @50 and half its NAV in cash: what a buy-only run invests.

    Targets are a third each and C's ADV budget is 25,000 shares, so the hand answer for P1 is buy
    1,250 A, 2,500 B, and 25,000 C (C is capped at 0.25, the rest splits evenly), and P2 — C's budget
    spent by P1 — buys 2,500 A and 5,000 B.
    """
    root = tmp_path / "half-cash"
    shutil.copytree(EXAMPLE_DATA, root)
    (root / "details.csv").write_text(
        "portfolio_id,name,state,st_tax_rate,lt_tax_rate,cash,nav,benchmark_id\nP1,Alpha Growth,NY,0.40,0.20,500000,1000000,B1\nP2,Beta Income,CA,0.37,0.20,500000,1000000,B1\n"
    )
    (root / "holdings.csv").write_text(
        "portfolio_id,security_id,quantity,avg_cost,acquired_on\n"
        "P1,A,2500,100,2024-01-15T00:00:00Z\nP1,B,5000,50,2024-01-15T00:00:00Z\nP2,A,2500,100,2025-11-01T00:00:00Z\nP2,B,5000,50,2025-11-01T00:00:00Z\n"
    )
    return root


def sell_book(tmp_path: Path) -> Path:
    """The example data allowed to raise cash (``cash_bounds`` ``[0, 1]``) with A's ADV budget cut to 1,000 shares: what a sell-only run trims.

    Each portfolio holds A 0.5 and B 0.5 against a target of a third each, so the hand answer for P1 is
    sell 1,000 A (its whole ADV budget, a 0.1 weight) and 3,333 B (to a third); P2, with A's budget spent
    by P1, sells 3,333 B alone.
    """
    root = tmp_path / "sell-book"
    shutil.copytree(EXAMPLE_DATA, root)
    constraints = json.loads((root / "constraints.json").read_text())
    for style in constraints.values():
        style["cash_bounds"] = ["0", "1"]
    (root / "constraints.json").write_text(json.dumps(constraints))
    (root / "universe.csv").write_text("security_id,sector,adv_shares,lot_size,restricted\nA,TECH,4000,1,false\nB,TECH,1000000,1,false\nC,TECH,100000,1,false\n")
    return root


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


def lying_assembly_step(frames: DatasetFrames) -> DatasetFrames:
    """Annotated as an assembly step but returns a frame; the engine must catch it."""
    return frames["universe"]  # ty: ignore[invalid-return-type]  # the lie is the case under test


def score_by_price(frames: DatasetFrames) -> DatasetFrames:
    """A custom assembly step: attach a ``Float64`` analytics column to both holdings and universe from the prices dataset."""
    scores = frames["prices"].assign(score=frames["prices"]["price"].map(float).astype("Float64")).drop(columns=["price"])
    holdings = frames["holdings"].merge(scores, on="security_id", how="left", validate="many_to_one")
    universe = frames["universe"].merge(scores, on="security_id", how="left", validate="one_to_one")
    return frames.with_frame("holdings", holdings).with_frame("universe", universe)


def refuse_assembly(frames: DatasetFrames) -> DatasetFrames:
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


def _example_body(real_steps: bool) -> dict[str, object]:
    body = json.loads(EXAMPLE_CONFIG.read_text())
    if not real_steps:
        body["objective"] = {"terms": ["tests.conftest:noop_term"]}
        body["constraints"] = []
    body["sink"] = "tests.conftest:noop_sink"
    return {str(key): value for key, value in body.items()}


def example_config(**overrides: object) -> RunConfig:
    """The shipped example config with no-op objective, constraints, and sink; sections replaced by ``overrides``."""
    return load_run_config(json.dumps(_example_body(real_steps=False) | overrides))


def example_config_real(**overrides: object) -> RunConfig:
    """The shipped example config with its real terms and constraints and a no-op sink."""
    return load_run_config(json.dumps(_example_body(real_steps=True) | overrides))


def resolved_example(**overrides: object) -> ResolvedConfig:
    """``example_config`` resolved."""
    return resolve_config(example_config(**overrides))


def resolved_example_real(**overrides: object) -> ResolvedConfig:
    """``example_config_real`` resolved."""
    return resolve_config(example_config_real(**overrides))


class FixedClock:
    """Always the same instant, so manifests are reproducible in tests."""

    def __init__(self, at: datetime = AS_OF) -> None:
        self.at = at

    def now(self) -> datetime:
        """The fixed instant."""
        return self.at


class FixedIds:
    """Deterministic run ids."""

    def __init__(self, run_id: str = "run-test") -> None:
        self.run_id = run_id

    def new_run_id(self) -> str:
        """The fixed id."""
        return self.run_id


def io_context(output_dir: Path, data_root: Path = EXAMPLE_DATA, run_id: str = "run-test") -> IoContext:
    """An ``IoContext`` with a fixed clock."""
    return IoContext(data_root=data_root, output_dir=output_dir, run_id=run_id, clock=FixedClock())


# --- one local Dask cluster for the whole session; runs connect to it by address so no test pays a cluster start ---


@pytest.fixture(scope="session")
def scheduler_address() -> Iterator[str]:
    """A two-worker ``LocalCluster`` shared by every test that runs portfolios through workers."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0", silence_logs=logging.WARNING)
    try:
        yield str(cluster.scheduler_address)
    finally:
        cluster.close()


def execution_on(scheduler_address: str, *, max_workers: int = 2) -> ExecutionSettings:
    """Execution settings that use the session cluster."""
    return ExecutionSettings(cluster=scheduler_address, min_workers=1, max_workers=max_workers, cluster_timeout_s=120.0)


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
