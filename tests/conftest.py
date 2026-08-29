"""Frame builders derived from the boundary schemas, and small domain-object factories.

Every builder takes partial rows and fills unstated fields with valid defaults, so a test
states only what it is about. Dtypes come from the schema, so fixtures cannot drift from it.
Builders are exposed as fixtures because ``--import-mode=importlib`` keeps ``tests/`` off
``sys.path``.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config.models import RunConfig, load_run_config
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm
from portfolio_optimizer.domain.data import IoContext, LoadRequest, PortfolioData, PortfolioDetails, StyleConstraints
from portfolio_optimizer.domain.frames import FrameSchema
from portfolio_optimizer.domain.results import F64, Artifact, ProblemSpec
from portfolio_optimizer.domain.schemas import COVARIANCE, DETAILS, HOLDINGS, ORDERS, PORTFOLIOS, TARGETS, UNIVERSE
from portfolio_optimizer.domain.types import Clock, IdFactory

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
    COVARIANCE.name: {"security_id_a": "A", "security_id_b": "A", "covariance": 0.04},
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
_SCHEMAS: dict[str, FrameSchema] = {schema.name: schema for schema in (PORTFOLIOS, DETAILS, HOLDINGS, UNIVERSE, TARGETS, COVARIANCE, ORDERS)}


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

    def covariance(self, *rows: Row) -> pd.DataFrame:
        """Build a long-form ``covariance`` frame."""
        return build(COVARIANCE, *rows)

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
    covariance: pd.DataFrame | None = None,
    style: StyleConstraints | None = None,
    as_of: datetime = AS_OF,
) -> PortfolioData:
    """The canonical small bundle: P1 holds A 5000 and B 10000 against the three-security universe."""
    frames = Frames()
    return PortfolioData(
        details=details if details is not None else make_details(),
        holdings=holdings if holdings is not None else frames.holdings({"security_id": "A", "quantity": 5000}, {"security_id": "B", "quantity": 10000, "avg_cost": Decimal(60)}),
        universe=universe if universe is not None else frames.three_security_universe(),
        targets=targets if targets is not None else frames.equal_weight_targets(),
        covariance=covariance,
        style=style if style is not None else make_style(),
        as_of=as_of,
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
    schemas: Mapping[str, FrameSchema]


@pytest.fixture
def frames() -> Frames:
    """Per-schema frame builders."""
    return Frames()


@pytest.fixture
def make() -> Factories:
    """Domain-object factories."""
    return Factories(details=make_details, style=make_style, portfolio_data=make_portfolio_data, spec=make_spec, schemas=_SCHEMAS)


# --- steps that satisfy the resolver's contracts, for tests that need a resolvable config ---


def noop_term(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:
    """Never invoked; exists so a config can resolve before real terms are exercised."""
    raise NotImplementedError


def lying_term(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:
    """Annotated as a term but returns a constraint set; solve must catch it."""
    del x, spec
    return ConstraintSet("lie", ())  # ty: ignore[invalid-return-type]  # the lie is the case under test


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
    return resolve_config(example_config(**overrides), config_sha256="example")


def resolved_example_real(**overrides: object) -> ResolvedConfig:
    """``example_config_real`` resolved."""
    return resolve_config(example_config_real(**overrides), config_sha256="example")


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


def failing_sink(orders: pd.DataFrame, io: IoContext) -> tuple[Artifact, ...]:
    """A sink whose destination is down."""
    del orders, io
    msg = "trading gateway unreachable"
    raise OSError(msg)


def io_context(output_dir: Path, data_root: Path = EXAMPLE_DATA, run_id: str = "run-test") -> IoContext:
    """An ``IoContext`` with a fixed clock."""
    return IoContext(data_root=data_root, output_dir=output_dir, run_id=run_id, clock=FixedClock())


def _protocols_hold(clock: Clock, ids: IdFactory) -> None:
    del clock, ids


_protocols_hold(FixedClock(), FixedIds())
