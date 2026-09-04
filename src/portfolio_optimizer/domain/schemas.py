"""The concrete frame schemas every dataset and output must satisfy.

Column conventions: identifiers are ``string``; share counts are ``Int64``; money, prices, and
weights are ``decimal``; statistical estimates (alpha, scores, loadings) are ``Float64``. The
engine-known frames declare what the shipped build reads and leave the rest open: ``holdings``,
``universe``, and ``details`` all accept columns beyond their schemas, and the build exports every
one of them by name — a universe column as a per-security column, flag, or grouping; a details
column as a per-account scalar.
"""

from decimal import Decimal

import pandas as pd

from portfolio_optimizer.domain.frames import ColumnSpec, FrameCheck, FrameSchema

ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)


def _cash_bounds_ordered(frame: pd.DataFrame) -> str | None:
    if "cash_lb" not in frame.columns or "cash_ub" not in frame.columns:
        return None
    off = [str(portfolio) for portfolio, low, high in zip(frame["portfolio_id"], frame["cash_lb"], frame["cash_ub"], strict=True) if low > high]
    if off:
        return f"cash_lb exceeds cash_ub for portfolio(s) {off}"
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
        # What the engine itself reads: the id, the NAV every weight is a fraction of, the single-name
        # cap the build folds into the box, and the dust threshold the order step drops trades below.
        ColumnSpec("portfolio_id", "string"),
        ColumnSpec("nav", "decimal", gt=ZERO),
        ColumnSpec("max_weight", "decimal", gt=ZERO, le=ONE),
        ColumnSpec("min_trade_notional", "decimal", ge=ZERO),
        # What the shipped build derives from when present: the tax rates give `tax_per_dollar`, the
        # participation gives `adv_capacity`; a term or row that needs the missing column is refused by
        # name at build. The rest are facts and style limits that reach the spec only as scalars a typed
        # constraint row reads by name, so a desk whose account master lacks one leaves it out.
        ColumnSpec("name", "string", required=False),
        ColumnSpec("state", "string", required=False),
        ColumnSpec("st_tax_rate", "decimal", required=False, ge=ZERO, lt=ONE),
        ColumnSpec("lt_tax_rate", "decimal", required=False, ge=ZERO, lt=ONE),
        ColumnSpec("cash", "decimal", required=False, ge=ZERO),
        ColumnSpec("max_turnover", "decimal", required=False, ge=ZERO, le=TWO),
        ColumnSpec("max_adv_participation", "decimal", required=False, ge=ZERO, le=ONE),
        ColumnSpec("cash_lb", "decimal", required=False, ge=ZERO, le=ONE),
        ColumnSpec("cash_ub", "decimal", required=False, ge=ZERO, le=ONE),
    ),
    key=("portfolio_id",),
    checks=(FrameCheck("cash_bounds_ordered", _cash_bounds_ordered),),
    allow_extra=True,  # any further column rides along on the account's details; a numeric one becomes a spec scalar a constraint can bound against
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
        ColumnSpec("sector", "string", required=False),
        ColumnSpec("adv_shares", "Int64", required=False, ge=ZERO),
        ColumnSpec("adv_consumed_shares", "Int64", required=False, ge=ZERO),
        ColumnSpec("lot_size", "Int64", required=False, ge=ONE),
        ColumnSpec("restricted", "bool", required=False),
        ColumnSpec("alpha", "Float64", required=False),
        ColumnSpec("tcost_bps", "decimal", required=False, ge=ZERO),
        ColumnSpec("min_weight", "decimal", required=False, nullable=True, ge=ZERO, le=ONE),
        ColumnSpec("max_weight", "decimal", required=False, nullable=True, ge=ZERO, le=ONE),
    ),
    key=("security_id",),
    allow_extra=True,  # analytics columns joined or computed per security; the build exports every extra by name
)
"""Every security the book may trade. Only ``security_id`` and ``price`` are required: ``sector`` is one grouping among any string column, ``adv_shares`` enables the participation constraints and ``adv_consumed_shares`` is what an earlier run already took of it, ``lot_size`` defaults to one share, ``restricted`` to false."""

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

CHECK_RESULTS = FrameSchema(name="check_results", columns=(ColumnSpec("portfolio_id", "string"), ColumnSpec("ok", "bool")), key=(), allow_extra=True)
"""What a check step returns: every row the business rule examined, ``ok`` false where it was breached, and any further column the check's own detail.

There is no key: the engine does not know what identifies a case, only which portfolio it belongs to.
"""

CONSTRAINTS = FrameSchema(name="constraints", columns=(ColumnSpec("portfolio_id", "string"),), key=(), allow_extra=True)
"""One portfolio's constraints, one typed row each.

The engine knows two things about a constraint row: which portfolio it belongs to, and — through
its ``kind`` column, which names a typed model (``domain/constraints.py``) — the declaration it
schedules by: whether the row reads the chain, and what it couples through. The model's fields
travel in ``params`` as JSON, its name in ``label``, and the shipped cvxpy step renders it from the
model. A frame in another vocabulary, with no ``kind`` column, is a custom solve step's to read.
There is no key, because the engine does not know what identifies a row.

Optional: a run whose solve step needs no constraints declares no such dataset, and every portfolio
gets the empty frame.
"""

DATASET_SCHEMAS: dict[str, FrameSchema] = {"holdings": HOLDINGS, "universe": UNIVERSE, "details": DETAILS, "constraints": CONSTRAINTS}
"""Engine-known frames and the schema each must satisfy after assembly. Any other dataset name is an extra."""

REQUIRED_DATASETS: tuple[str, ...] = ("holdings", "universe", "details")
"""Datasets a run cannot do without, loaded directly or produced by an assembly step. ``constraints`` is engine-known but optional."""

RESERVED_DATASET_NAMES: frozenset[str] = frozenset({*DATASET_SCHEMAS, "portfolios"})
"""Names an extra dataset may not use."""
