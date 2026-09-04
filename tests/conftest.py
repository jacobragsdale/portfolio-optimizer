"""Frame builders derived from the boundary schemas, domain-object factories, and the shipped example config.

Every builder takes partial rows and fills unstated fields with valid defaults, so a test
states only what it is about. Dtypes come from the schema, so fixtures cannot drift from it.
Builders are exposed as fixtures because ``--import-mode=importlib`` keeps ``tests/`` off
``sys.path``. Steps and kinds a config names live in ``tests/steps.py``; run helpers in ``tests/engine/support.py``.
"""

import json
import logging
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from distributed import LocalCluster

from portfolio_optimizer.config.models import RunConfig, StepSpec, load_run_config
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDetails
from portfolio_optimizer.domain.frames import FrameSchema
from portfolio_optimizer.domain.objective import TypedTerm, parse_terms
from portfolio_optimizer.domain.results import F64, Grouping, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.domain.schemas import CONSTRAINTS, DETAILS, HOLDINGS, ORDERS, PORTFOLIOS, UNIVERSE

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_inflow.json"
SELL_CONFIG = REPO_ROOT / "configs" / "example_outflow.json"
REBALANCE_CONFIG = REPO_ROOT / "configs" / "example_rebalance.json"
HANDOFF_CONFIG = REPO_ROOT / "configs" / "example_inflow_after_outflow.json"
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


# --- the typed constraint rows the example ships, as the loader delivers them ---

CASH_FLOOR: Row = {"kind": "cash_limit", "label": "cash_floor", "params": {"direction": ">=", "bounds": {"scalar": "cash_lb"}}}
CASH_CAP: Row = {"kind": "cash_limit", "label": "cash_cap", "params": {"direction": "<=", "bounds": {"scalar": "cash_ub"}}}
TURNOVER: Row = {"kind": "turnover_limit", "label": "turnover", "params": {"direction": "<=", "bounds": {"scalar": "max_turnover"}}}
ADV: Row = {"kind": "participation_limit", "label": "adv", "params": {"direction": "<="}}
SHIPPED_CONSTRAINTS: list[Row] = [CASH_FLOOR, CASH_CAP, TURNOVER, ADV]
"""The example's constraint rows for an account without sector bands, in its order: the cash bounds, the turnover cap, and the chain-aware ADV cap."""
NO_CHAIN_CONSTRAINTS: list[Row] = [CASH_FLOOR, CASH_CAP, TURNOVER]
"""The shipped rows without the chain-aware ADV cap; a run over them reads no chain, so nothing waits."""


def typed_row(kind: str, label: str, **params: object) -> Row:
    """One typed constraint row: ``kind``, ``label``, and the model's fields as ``params``."""
    return {"kind": kind, "label": label, "params": params}


def constraint_frame(rows: Sequence[Row] = tuple(SHIPPED_CONSTRAINTS), portfolio_id: str = "P1") -> pd.DataFrame:
    """A ``constraints`` frame of typed rows the way the loader delivers them: ``params`` as JSON text; with no rows, the empty frame that constrains nothing."""
    if not rows:
        return empty_frame(CONSTRAINTS)
    records = [{"portfolio_id": portfolio_id, **row, "params": json.dumps(row["params"]) if isinstance(row.get("params"), Mapping) else row.get("params")} for row in rows]
    return pd.DataFrame.from_records(records).astype({"portfolio_id": "string", "kind": "string", "label": "string", "params": "string"})


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

    def constraints(self, *rows: Row, portfolio_id: str = "P1") -> pd.DataFrame:
        """Build a ``constraints`` frame of typed rows; with none, the empty frame that constrains nothing."""
        return constraint_frame(rows, portfolio_id=portfolio_id)

    def orders(self, *rows: Row) -> pd.DataFrame:
        """Build an ``orders`` frame."""
        return build(ORDERS, *rows)

    def parameters(self, **values: object) -> pd.DataFrame:
        """A ``name``/``value`` extra dataset of runtime settings, typed the way ``load_parameters`` produces it."""
        return pd.DataFrame({"name": pd.Series(list(values), dtype="string"), "value": pd.Series([Decimal(str(value)) for value in values.values()], dtype="object")})

    def trades(self, *rows: tuple[str, str, str, int]) -> pd.DataFrame:
        """A ``trades`` frame of ``(portfolio_id, security_id, side, days_ago)`` rows against the test's as-of instant, typed the way ``load_trades`` produces it."""
        return pd.DataFrame(
            {
                "portfolio_id": pd.Series([portfolio for portfolio, _, _, _ in rows], dtype="string"),
                "security_id": pd.Series([security for _, security, _, _ in rows], dtype="string"),
                "side": pd.Series([side for _, _, side, _ in rows], dtype="string"),
                "traded_on": pd.Series([AS_OF - timedelta(days=days) for _, _, _, days in rows], dtype="datetime64[ns, UTC]"),
            }
        )

    def three_security_universe(self) -> pd.DataFrame:
        """The example's securities: A, B, C at 100, 50, 10 in one sector, C thin and dear to trade but worth the most."""
        return self.universe(
            {"security_id": "A", "price": Decimal(100), "adv_shares": 1_000_000, "alpha": 0.03, "tcost_bps": Decimal(5)},
            {"security_id": "B", "price": Decimal(50), "adv_shares": 1_000_000, "alpha": -0.01, "tcost_bps": Decimal(5)},
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


_SPEC_COLUMNS: tuple[str, ...] = ("tax_per_dollar", "tcost_per_dollar", "adv_capacity")
_SPEC_SCALARS: tuple[str, ...] = ("max_turnover", "cash_lb", "cash_ub", "min_trade_notional", "max_weight", "max_adv_participation")


def make_spec(n: int = 3, **overrides: object) -> ProblemSpec:
    """A feasible spec with ``n`` securities, identity-like data, and no binding limits.

    ``alpha`` rises with the index, so the shipped ``alpha`` term has a strict preference and every
    optimum below is a single vertex rather than a face the solver may pick any point of. The derived
    columns (``tax_per_dollar``, ``tcost_per_dollar``, ``adv_capacity``), the account scalars
    (``max_turnover``, ``cash_lb``, ...), and ``sector_names``/``sector_matrix`` may be given as bare
    keywords and are folded into ``columns``, ``scalars``, and ``groups``.
    """
    ids = tuple(f"S{i}" for i in range(n))
    zeros: F64 = np.zeros(n)
    columns: dict[str, object] = {"alpha": np.arange(n, dtype=np.float64) * 0.01, "tax_per_dollar": zeros, "tcost_per_dollar": zeros, "adv_capacity": np.full(n, 10.0)}
    scalars: dict[str, object] = {"max_turnover": 2.0, "cash_lb": 0.0, "cash_ub": 0.0, "min_trade_notional": 0.0, "max_weight": 1.0, "max_adv_participation": 1.0}
    sector_names = overrides.pop("sector_names", ("TECH",))
    sector_matrix = overrides.pop("sector_matrix", np.ones((len(sector_names), n)))  # ty: ignore[invalid-argument-type]  # a name tuple by construction
    for name in _SPEC_COLUMNS:
        if name in overrides:
            columns[name] = overrides.pop(name)
    for name in _SPEC_SCALARS:
        if name in overrides:
            scalars[name] = overrides.pop(name)
    columns.update(cast("Mapping[str, object]", overrides.pop("columns", {})))
    scalars.update(cast("Mapping[str, object]", overrides.pop("scalars", {})))
    base: dict[str, object] = {
        "portfolio_id": "P1",
        "as_of_date": AS_OF,
        "security_ids": ids,
        "nav": 1_000_000.0,
        "w0": np.full(n, 1.0 / n) if n else zeros,
        "price": np.full(n, 100.0),
        "shares_held": np.full(n, 1_000_000.0 / n / 100.0) if n else zeros,
        "lot_size": np.ones(n),
        "lb": zeros,
        "ub": np.ones(n),
        "columns": columns,
        "scalars": scalars,
        "groups": {"sector": Grouping(tuple(sector_names), sector_matrix)},  # ty: ignore[invalid-argument-type]  # see above
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


# --- the shipped terms, and the example config with its steps swapped for no-ops or kept real ---

ALPHA: dict[str, object] = {"kind": "linear", "name": "alpha", "column": "alpha", "weight": "-1"}
TAX_COST: dict[str, object] = {"kind": "linear", "name": "tax_cost", "column": "tax_per_dollar", "vector": "sell"}
TRANSACTION_COST: dict[str, object] = {"kind": "linear", "name": "transaction_cost", "column": "tcost_per_dollar", "vector": "trade"}
BUY_TERMS: list[dict[str, object]] = [ALPHA, TRANSACTION_COST]
"""The inflow's objective, as ``configs/example_inflow.json`` has it: buy the expected return, pay the trading cost."""
SELL_TERMS: list[dict[str, object]] = [ALPHA, TAX_COST, TRANSACTION_COST]
"""The outflow's objective, as ``configs/example_outflow.json`` has it: the inflow's terms plus the tax on what is sold — ``tax_cost`` reads ``sell`` and so cannot run in an inflow."""
NOOP_TERMS: list[dict[str, object]] = [{"kind": "linear", "name": "noop", "weight": "0"}]
"""A zero objective: lets a config resolve and solve without exercising a real term."""


def terms_of(*items: object) -> tuple[TypedTerm, ...]:
    """Term records — or bare term names from the example — as models."""
    named = {str(term["name"]): term for term in SELL_TERMS}
    return parse_terms([named[item] if isinstance(item, str) else item for item in items])


def step_spec(name: str, **params: object) -> StepSpec:
    """A ``StepSpec`` as the config would carry it."""
    return StepSpec.model_validate_json(json.dumps({"name": name, "params": params}))


NO_LATENCY: dict[str, object] = {"min_latency_s": 0, "max_latency_s": 0}
"""What every shipped loader's `params` gets in a test: the mock services stand in for real ones and wait accordingly, and no test can afford that."""


def as_object(value: object) -> dict[str, object]:
    """A JSON object read from a config file, typed."""
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def instant(entry: object) -> dict[str, object]:
    """One `datasets` entry with its loader's simulated wait removed; an inline book passes through untouched."""
    spec = as_object({"loader": entry} if isinstance(entry, str) else entry)
    if "ids" in spec:
        return spec
    step = as_object({"name": spec["loader"]} if isinstance(spec["loader"], str) else spec["loader"])
    return spec | {"loader": step | {"params": as_object(step.get("params", {})) | NO_LATENCY}}


def example_body(*, latency: bool = False) -> dict[str, object]:
    """The shipped example config as a JSON object, by default with every loader's simulated latency removed."""
    body = as_object(json.loads(EXAMPLE_CONFIG.read_text()))
    if latency:
        return body
    datasets = as_object(body["datasets"])
    return body | {"datasets": {name: instant(spec) for name, spec in datasets.items()}}


def example_datasets(**overrides: object) -> dict[str, object]:
    """The example's `datasets` block with entries replaced; what a test that swaps one input starts from."""
    return as_object(example_body()["datasets"]) | overrides


def _example_body(real_steps: bool) -> dict[str, object]:
    body = example_body()
    if not real_steps:
        body["objective"] = NOOP_TERMS
        datasets = body["datasets"]
        assert isinstance(datasets, dict)
        body["datasets"] = {name: spec for name, spec in datasets.items() if name != "constraints"}
    body["sink"] = "tests.steps:noop_sink"
    return body


def example_config(**overrides: object) -> RunConfig:
    """The shipped example config with a zero objective, no constraints dataset, and a no-op sink; sections replaced by ``overrides``."""
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


# --- the book the golden tests run over: the first two of the shipped hundred accounts ---


TWO_ACCOUNTS = "portfolio_id,solve_order\nP1,0\nP2,1\n"
"""A portfolio list of two accounts. Every other table is the shipped one: a loader returns only the rows the portfolio list asks for, so cutting the list cuts the book."""


def two_account_book(root: Path, **files: str) -> Path:
    """The example data copied to ``root``, cut to two accounts, with the named tables replaced."""
    shutil.copytree(EXAMPLE_DATA, root, dirs_exist_ok=True)
    for name, content in {"portfolios.csv": TWO_ACCOUNTS, **files}.items():
        (root / name).write_text(content)
    return root


@pytest.fixture(scope="session")
def book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The two-account book, built once for the session; no test writes to it."""
    return two_account_book(tmp_path_factory.mktemp("books") / "book")


# --- one local Dask cluster for the whole session; runs connect to it by address so no test pays a cluster start ---


@pytest.fixture(scope="session")
def scheduler_address() -> Iterator[str]:
    """A two-worker ``LocalCluster`` shared by every test that runs portfolios through workers."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0", silence_logs=logging.WARNING)
    try:
        yield str(cluster.scheduler_address)
    finally:
        cluster.close()
