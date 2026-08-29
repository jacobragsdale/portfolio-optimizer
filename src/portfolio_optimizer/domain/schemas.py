"""The concrete frame schemas every dataset and output must satisfy.

Column conventions: identifiers are ``string``; share counts are ``Int64``; money, prices, and
weights are ``decimal``; statistical estimates (alpha, scores, loadings) are ``Float64``.
"""

from decimal import Decimal

import pandas as pd

from portfolio_optimizer.domain.frames import ColumnSpec, FrameCheck, FrameSchema

ZERO = Decimal(0)
ONE = Decimal(1)
TARGET_WEIGHT_SUM_TOLERANCE = Decimal("1e-8")


def _targets_sum_to_one(frame: pd.DataFrame) -> str | None:
    sums = frame.groupby("benchmark_id", sort=True)["weight"].agg(lambda weights: sum(weights, ZERO))
    off = {str(benchmark): total for benchmark, total in sums.items() if abs(total - ONE) > TARGET_WEIGHT_SUM_TOLERANCE}
    if off:
        return f"weights do not sum to 1 for benchmark(s) {off}"
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


PORTFOLIOS = FrameSchema(
    name="portfolios",
    columns=(ColumnSpec("portfolio_id", "string"), ColumnSpec("solve_order", "Int64", ge=ZERO)),
    key=("portfolio_id",),
    checks=(FrameCheck("solve_order_unique", lambda frame: "solve_order values are not unique" if bool(frame["solve_order"].duplicated().any()) else None),),
)

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
    ),
    key=("portfolio_id",),
)

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
        ColumnSpec("as_of", "datetime_utc"),
    ),
    key=("portfolio_id", "security_id"),
    checks=(FrameCheck("notional_matches", _orders_notional_matches),),
)

DATASET_SCHEMAS: dict[str, FrameSchema] = {"holdings": HOLDINGS, "universe": UNIVERSE, "details": DETAILS, "targets": TARGETS}
"""Engine-known frames and the schema each must satisfy after assembly. Any other dataset name is an extra."""

REQUIRED_FRAMES: tuple[str, ...] = ("holdings", "universe", "details", "targets")
"""Frames that must exist after assembly, loaded directly or produced by an assembly step."""

REQUIRED_DATASETS: tuple[str, ...] = (*REQUIRED_FRAMES, "constraints")
"""Datasets a run cannot do without. ``constraints`` is a dict, not a frame, and must always be declared."""

RESERVED_DATASET_NAMES: frozenset[str] = frozenset({*DATASET_SCHEMAS, "constraints", "portfolios"})
"""Names an extra dataset may not use."""
