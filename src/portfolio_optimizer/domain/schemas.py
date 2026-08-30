"""The concrete frame schemas every dataset and output must satisfy.

Column conventions: identifiers are ``string``; share counts are ``Int64``; money, prices, and
weights are ``decimal``; statistical estimates (alpha, scores, loadings) are ``Float64``.
"""

from decimal import Decimal

import pandas as pd

from portfolio_optimizer.domain.frames import ColumnSpec, FrameCheck, FrameSchema

ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
TARGET_WEIGHT_SUM_TOLERANCE = Decimal("1e-8")


def _targets_sum_to_one(frame: pd.DataFrame) -> str | None:
    sums = frame.groupby("benchmark_id", sort=True)["weight"].agg(lambda weights: sum(weights, ZERO))
    off = {str(benchmark): total for benchmark, total in sums.items() if abs(total - ONE) > TARGET_WEIGHT_SUM_TOLERANCE}
    if off:
        return f"weights do not sum to 1 for benchmark(s) {off}"
    return None


def _cash_bounds_ordered(frame: pd.DataFrame) -> str | None:
    off = [str(portfolio) for portfolio, low, high in zip(frame["portfolio_id"], frame["cash_lb"], frame["cash_ub"], strict=True) if low > high]
    if off:
        return f"cash_lb exceeds cash_ub for portfolio(s) {off}"
    return None


def _sector_bounds_ordered(frame: pd.DataFrame) -> str | None:
    off = [f"{portfolio}/{sector}" for portfolio, sector, low, high in zip(frame["portfolio_id"], frame["sector"], frame["lower"], frame["upper"], strict=True) if low > high]
    if off:
        return f"lower exceeds upper for {off}"
    return None


def _orders_notional_matches(frame: pd.DataFrame) -> str | None:
    mismatched = [
        str(security)
        for security, quantity, price, notional in zip(frame["security_id"], frame["quantity"], frame["reference_price"], frame["notional"], strict=True)
        if Decimal(int(quantity)) * price != notional
    ]
    if mismatched:
        return f"notional != quantity * reference_price for {mismatched}"
    return None


PORTFOLIOS = FrameSchema(name="portfolios", columns=(ColumnSpec("portfolio_id", "string"), ColumnSpec("solve_order", "Int64", required=False, ge=ZERO)), key=("portfolio_id",))
"""The portfolio list. ``solve_order`` is a priority — lower solves first, ties break on ``portfolio_id`` — and may be omitted or repeated."""

DETAILS = FrameSchema(
    name="details",
    columns=(
        ColumnSpec("portfolio_id", "string"),
        ColumnSpec("name", "string"),
        ColumnSpec("state", "string"),
        ColumnSpec("st_tax_rate", "decimal", ge=ZERO, lt=ONE),
        ColumnSpec("lt_tax_rate", "decimal", ge=ZERO, lt=ONE),
        ColumnSpec("cash", "decimal", ge=ZERO),
        ColumnSpec("nav", "decimal", gt=ZERO),
        ColumnSpec("benchmark_id", "string"),
        # The account's management-style limits. Every constraint reads its bounds from here or from
        # `sector_bounds`, so what a run permits is data that changes daily, not config.
        ColumnSpec("max_weight", "decimal", gt=ZERO, le=ONE),
        ColumnSpec("max_turnover", "decimal", ge=ZERO, le=TWO),
        ColumnSpec("max_adv_participation", "decimal", ge=ZERO, le=ONE),
        ColumnSpec("min_trade_notional", "decimal", ge=ZERO),
        ColumnSpec("cash_lb", "decimal", ge=ZERO, le=ONE),
        ColumnSpec("cash_ub", "decimal", ge=ZERO, le=ONE),
    ),
    key=("portfolio_id",),
    checks=(FrameCheck("cash_bounds_ordered", _cash_bounds_ordered),),
)

SECTOR_BOUNDS = FrameSchema(
    name="sector_bounds",
    columns=(ColumnSpec("portfolio_id", "string"), ColumnSpec("sector", "string"), ColumnSpec("lower", "decimal", ge=ZERO, le=ONE), ColumnSpec("upper", "decimal", ge=ZERO, le=ONE)),
    key=("portfolio_id", "sector"),
    checks=(FrameCheck("bounds_ordered", _sector_bounds_ordered),),
)
"""Per-account sector limits, one row per portfolio and sector. Optional: a run that declares no such dataset bounds no sector."""

HOLDINGS = FrameSchema(
    name="holdings",
    columns=(
        ColumnSpec("portfolio_id", "string"),
        ColumnSpec("security_id", "string"),
        ColumnSpec("quantity", "Int64", ge=ZERO),
        ColumnSpec("avg_cost", "decimal", ge=ZERO),
        ColumnSpec("acquired_on", "datetime_utc"),
    ),
    key=("portfolio_id", "security_id"),
    allow_extra=True,  # analytics columns joined or computed per position ride along; the shipped build ignores them
)

UNIVERSE = FrameSchema(
    name="universe",
    columns=(
        ColumnSpec("security_id", "string"),
        ColumnSpec("price", "decimal", gt=ZERO),
        ColumnSpec("sector", "string"),
        ColumnSpec("adv_shares", "Int64", ge=ZERO),
        ColumnSpec("lot_size", "Int64", ge=ONE),
        ColumnSpec("restricted", "bool"),
        ColumnSpec("alpha", "Float64", required=False),
        ColumnSpec("tcost_bps", "decimal", required=False, ge=ZERO),
        ColumnSpec("min_weight", "decimal", required=False, nullable=True, ge=ZERO, le=ONE),
        ColumnSpec("max_weight", "decimal", required=False, nullable=True, ge=ZERO, le=ONE),
    ),
    key=("security_id",),
    allow_extra=True,  # analytics columns joined or computed per security; build exports every numeric extra by name
)

TARGETS = FrameSchema(
    name="targets",
    columns=(ColumnSpec("benchmark_id", "string"), ColumnSpec("security_id", "string"), ColumnSpec("weight", "decimal", ge=ZERO, le=ONE)),
    key=("benchmark_id", "security_id"),
    checks=(FrameCheck("weights_sum_to_one", _targets_sum_to_one),),
)

ORDER_SIDES = frozenset({"BUY", "SELL"})

ORDERS = FrameSchema(
    name="orders",
    columns=(
        ColumnSpec("portfolio_id", "string"),
        ColumnSpec("security_id", "string"),
        ColumnSpec("side", "string", allowed=ORDER_SIDES),
        ColumnSpec("quantity", "Int64", gt=ZERO),
        ColumnSpec("reference_price", "decimal", gt=ZERO),
        ColumnSpec("notional", "decimal", gt=ZERO),
        ColumnSpec("target_weight", "Float64"),
        ColumnSpec("unrounded_shares", "Float64"),
        ColumnSpec("spec_hash", "string"),
        ColumnSpec("run_id", "string"),
        ColumnSpec("as_of_date", "datetime_utc"),
    ),
    key=("portfolio_id", "security_id"),
    checks=(FrameCheck("notional_matches", _orders_notional_matches),),
)

CONSTRAINTS = FrameSchema(name="constraints", columns=(ColumnSpec("portfolio_id", "string"),), key=(), allow_extra=True)
"""One portfolio's constraints, in whatever shape the desk writes them.

The engine knows exactly one thing about a constraint row: which portfolio it belongs to. Everything
else is the desk's own vocabulary, carried opaquely from the loader, through the rules that adjust it,
to the solve step that interprets it — which is the only thing that understands the rest of the
columns. There is no key, because the engine does not know what identifies a row.

Optional: a run whose solve step needs no constraints declares no such dataset, and every portfolio
gets the empty frame.
"""

DATASET_SCHEMAS: dict[str, FrameSchema] = {"holdings": HOLDINGS, "universe": UNIVERSE, "details": DETAILS, "targets": TARGETS, "sector_bounds": SECTOR_BOUNDS, "constraints": CONSTRAINTS}
"""Engine-known frames and the schema each must satisfy after assembly. Any other dataset name is an extra."""

REQUIRED_DATASETS: tuple[str, ...] = ("holdings", "universe", "details", "targets")
"""Datasets a run cannot do without, loaded directly or produced by an assembly step. ``sector_bounds`` and ``constraints`` are engine-known but optional."""

REQUIRED_FRAMES: tuple[str, ...] = REQUIRED_DATASETS
"""Every dataset is a frame, so these are the same names; both spellings are kept because each reads better in its own place."""

RESERVED_DATASET_NAMES: frozenset[str] = frozenset({*DATASET_SCHEMAS, "portfolios"})
"""Names an extra dataset may not use."""
