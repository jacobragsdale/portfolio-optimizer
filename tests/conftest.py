"""Frame builders derived from the boundary schemas, domain-object factories, and the shipped example config.

Every builder takes partial rows and fills unstated fields with valid defaults, so a test
states only what it is about. Dtypes come from the schema, so fixtures cannot drift from it.
Builders are exposed as fixtures because ``--import-mode=importlib`` keeps ``tests/`` off
``sys.path``. Steps a config names live in ``tests/steps.py``; run helpers in ``tests/engine/support.py``.
"""

import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from distributed import LocalCluster

from portfolio_optimizer.config.models import RunConfig, StepSpec, load_run_config
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDetails
from portfolio_optimizer.domain.frames import FrameSchema
from portfolio_optimizer.domain.results import F64, ProblemSpec, Solution, SolveStatus, StepRef
from portfolio_optimizer.domain.schemas import CONSTRAINTS, DETAILS, HOLDINGS, ORDERS, PORTFOLIOS, UNIVERSE

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
        "max_weight": Decimal(1),
        "max_turnover": Decimal(2),
        "max_adv_participation": Decimal(1),
        "min_trade_notional": Decimal(0),
        "cash_lb": Decimal(0),
        "cash_ub": Decimal(0),
    },
    CONSTRAINTS.name: {"portfolio_id": "P1", "name": "long_only", "label": None, "params": None},
    HOLDINGS.name: {"portfolio_id": "P1", "security_id": "A", "quantity": 5000, "avg_cost": Decimal(90), "acquired_on": ACQUIRED},
    UNIVERSE.name: {"security_id": "A", "price": Decimal(100), "sector": "TECH", "adv_shares": 1_000_000, "lot_size": 1, "restricted": False, "alpha": 0.0},
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
        "as_of_date": AS_OF,
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

    def constraints(self, *names: str, portfolio_id: str = "P1") -> pd.DataFrame:
        """Build a ``constraints`` frame under the shipped convention; with no names, the empty frame that constrains nothing."""
        return build(CONSTRAINTS, *({"portfolio_id": portfolio_id, "name": name} for name in names)) if names else empty_frame(CONSTRAINTS)

    def orders(self, *rows: Row) -> pd.DataFrame:
        """Build an ``orders`` frame."""
        return build(ORDERS, *rows)

    def three_security_universe(self) -> pd.DataFrame:
        """The example's securities: A, B, C at 100, 50, 10 in one sector, C thin and dear to trade but worth the most."""
        return self.universe(
            {"security_id": "A", "price": Decimal(100), "adv_shares": 1_000_000, "alpha": 0.03, "tcost_bps": Decimal(5)},
            {"security_id": "B", "price": Decimal(50), "adv_shares": 1_000_000, "alpha": 0.01, "tcost_bps": Decimal(5)},
            {"security_id": "C", "price": Decimal(10), "adv_shares": 100_000, "alpha": 0.05, "tcost_bps": Decimal(20)},
        )


def make_details(**overrides: object) -> PortfolioDetails:
    """A valid ``PortfolioDetails`` with optional field overrides."""
    return PortfolioDetails.model_validate(dict(_DEFAULTS[DETAILS.name]) | overrides)


def make_portfolio_data(
    *,
    details: PortfolioDetails | None = None,
    holdings: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    constraints: pd.DataFrame | None = None,
    as_of_date: datetime = AS_OF,
    extras: Mapping[str, pd.DataFrame] | None = None,
    prevalidated: frozenset[str] = frozenset(),
) -> PortfolioData:
    """The canonical small bundle: P1 holds A 5000 and B 10000 against the three-security universe."""
    frames = Frames()
    return PortfolioData(
        details=details if details is not None else make_details(),
        holdings=holdings if holdings is not None else frames.holdings({"security_id": "A", "quantity": 5000}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(60)}),
        universe=universe if universe is not None else frames.three_security_universe(),
        constraints=constraints if constraints is not None else frames.constraints(*SHIPPED_CONSTRAINTS),
        as_of_date=as_of_date,
        extras=extras if extras is not None else {},
        prevalidated=prevalidated,
    )


def make_spec(n: int = 3, **overrides: object) -> ProblemSpec:
    """A feasible spec with ``n`` securities, identity-like data, and no binding limits.

    ``alpha`` rises with the index, so the shipped ``alpha`` term has a strict preference and every
    optimum below is a single vertex rather than a face the solver may pick any point of.
    """
    ids = tuple(f"S{i}" for i in range(n))
    zeros: F64 = np.zeros(n)
    base: dict[str, object] = {
        "portfolio_id": "P1",
        "as_of_date": AS_OF,
        "security_ids": ids,
        "sector_names": ("TECH",),
        "nav": 1_000_000.0,
        "w0": np.full(n, 1.0 / n) if n else zeros,
        "price": np.full(n, 100.0),
        "shares_held": np.full(n, 1_000_000.0 / n / 100.0) if n else zeros,
        "lot_size": np.ones(n),
        "columns": {"alpha": np.arange(n, dtype=np.float64) * 0.01},
        "tax_per_dollar": zeros,
        "tcost_per_dollar": zeros,
        "lb": zeros,
        "ub": np.ones(n),
        "adv_capacity": np.full(n, 10.0),
        "sector_matrix": np.ones((1, n)),
        "max_turnover": 2.0,
        "cash_lb": 0.0,
        "cash_ub": 0.0,
        "min_trade_notional": 0.0,
    }
    return ProblemSpec(**(base | overrides))  # ty: ignore[invalid-argument-type]  # the merged mapping is typed `object`; the dataclass validates every field on construction


def make_solution(spec: ProblemSpec, *, w: F64 | None = None, buy: F64 | None = None, sell: F64 | None = None, **overrides: object) -> Solution:
    """A ``Solution`` for ``spec``: at rest with no trades unless told otherwise, hashed against the spec, with placeholder provenance."""
    base: dict[str, object] = {
        "w": spec.w0 if w is None else w,
        "buy": np.zeros(spec.n) if buy is None else buy,
        "sell": np.zeros(spec.n) if sell is None else sell,
        "objective": 0.0,
        "status": SolveStatus.OPTIMAL,
        "solver": "X",
        "solver_version": "0",
        "solve_time_s": 0.0,
        "iterations": 1,
        "spec_hash": spec.content_hash(),
    }
    return Solution(**(base | overrides))  # ty: ignore[invalid-argument-type]  # merged mapping; Solution validates on construction


@dataclass(frozen=True, slots=True)
class Factories:
    """Domain-object factories handed to tests as one fixture."""

    details: Callable[..., PortfolioDetails]
    portfolio_data: Callable[..., PortfolioData]
    spec: Callable[..., ProblemSpec]
    solution: Callable[..., Solution]


@pytest.fixture
def frames() -> Frames:
    """Per-schema frame builders."""
    return Frames()


@pytest.fixture
def make() -> Factories:
    """Domain-object factories."""
    return Factories(details=make_details, portfolio_data=make_portfolio_data, spec=make_spec, solution=make_solution)


# --- the shipped constraints, and the example config with its steps swapped for no-ops or kept real ---

SHIPPED_CONSTRAINTS = ["long_only", "max_weight", "cash_bounds", "turnover_cap", "cumulative_adv_participation"]
"""Every constraint the template ships that needs no params, in the example's order; ``sector_bound`` takes a band and is named per case."""
NO_CHAIN_CONSTRAINTS = [name for name in SHIPPED_CONSTRAINTS if name != "cumulative_adv_participation"]
"""The shipped constraints without the chain-aware ADV cap; a run over them can declare `dependencies: none`."""
UNCOUPLED: dict[str, object] = {"on_error": "continue", "dependencies": "none"}
"""An `execution` block in which nothing waits for anything: coupling is declared, not inferred from the constraints."""
EXAMPLE_TERMS: list[object] = [{"name": "alpha", "params": {"weight": "1.0"}}, {"name": "tax_cost", "params": {"weight": "1.0"}}, {"name": "transaction_cost", "params": {"weight": "1.0"}}]
"""The example's objective terms, in its order: buy the expected return, pay the tax and the trading cost."""
BUY_ONLY_TERMS: list[object] = [term for term in EXAMPLE_TERMS if term != EXAMPLE_TERMS[1]]
"""The example's terms without ``tax_cost``, which reads ``sell`` and so cannot run in a buy-only run."""
BUY_ONLY_OBJECTIVE: dict[str, object] = {"terms": BUY_ONLY_TERMS}
"""The example's objective without ``tax_cost``."""


def constraint_frame(names: Sequence[str] = tuple(SHIPPED_CONSTRAINTS), portfolio_id: str = "P1") -> pd.DataFrame:
    """A ``constraints`` frame naming shipped steps under the convention ``solvers.cvxpy`` reads."""
    return Frames().constraints(*names, portfolio_id=portfolio_id)


def step_refs_for(names: list[str]) -> list[StepRef]:
    """``StepRef``s for shipped constraints named bare, with no params and the bare name as label."""
    return [StepRef(qualname=f"portfolio_optimizer.terms:{name}", params={}, label=name) for name in names]


def step_spec(name: str, **params: object) -> StepSpec:
    """A ``StepSpec`` as the config would carry it."""
    return StepSpec.model_validate_json(json.dumps({"name": name, "params": params}))


def example_body() -> dict[str, object]:
    """The shipped example config as a JSON object."""
    loaded = json.loads(EXAMPLE_CONFIG.read_text())
    assert isinstance(loaded, dict)
    return {str(key): value for key, value in loaded.items()}


def _example_body(real_steps: bool) -> dict[str, object]:
    body = example_body()
    if not real_steps:
        body["objective"] = {"terms": ["tests.steps:noop_term"]}
        datasets = body["datasets"]
        assert isinstance(datasets, dict)
        body["datasets"] = {name: spec for name, spec in datasets.items() if name != "constraints"}
    body["sink"] = "tests.steps:noop_sink"
    return body


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


# --- one local Dask cluster for the whole session; runs connect to it by address so no test pays a cluster start ---


@pytest.fixture(scope="session")
def scheduler_address() -> Iterator[str]:
    """A two-worker ``LocalCluster`` shared by every test that runs portfolios through workers."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0", silence_logs=logging.WARNING)
    try:
        yield str(cluster.scheduler_address)
    finally:
        cluster.close()
