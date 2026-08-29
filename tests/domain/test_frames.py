"""Tier 3: what each frame schema refuses, and that the builders satisfy them."""

from collections.abc import Mapping
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_optimizer.domain.frames import ColumnSpec, FrameSchema, FrameSchemaError, coerce_frame, validate_frame
from portfolio_optimizer.domain.schemas import DETAILS, HOLDINGS, ORDERS, PORTFOLIOS, TARGETS, UNIVERSE
from tests.conftest import Frames, empty_frame

ALL_SCHEMAS = [PORTFOLIOS, DETAILS, HOLDINGS, UNIVERSE, TARGETS, ORDERS]


@pytest.mark.parametrize("schema", ALL_SCHEMAS, ids=[schema.name for schema in ALL_SCHEMAS])
def test_builder_produces_a_frame_the_schema_accepts(frames: Frames, schema: FrameSchema) -> None:
    validate_frame(frames.for_schema(schema)(), schema)


@pytest.mark.parametrize("schema", ALL_SCHEMAS, ids=[schema.name for schema in ALL_SCHEMAS])
def test_empty_frame_with_declared_dtypes_is_accepted(schema: FrameSchema) -> None:
    validate_frame(empty_frame(schema), schema)


REJECT_CASES: list[tuple[str, FrameSchema, Mapping[str, object], str]] = [
    ("negative quantity", HOLDINGS, {"quantity": -1}, "quantity"),
    ("null quantity", HOLDINGS, {"quantity": None}, "null"),
    ("float where Decimal", HOLDINGS, {"avg_cost": 90.0}, "finite Decimals"),
    ("string where Decimal", HOLDINGS, {"avg_cost": "90"}, "finite Decimals"),
    ("non-finite Decimal", HOLDINGS, {"avg_cost": Decimal("Infinity")}, "finite Decimals"),
    ("zero price", UNIVERSE, {"price": Decimal(0)}, "price"),
    ("lot_size zero", UNIVERSE, {"lot_size": 0}, "lot_size"),
    ("NaN alpha", UNIVERSE, {"alpha": float("nan")}, "alpha"),
    ("tax rate of one", DETAILS, {"st_tax_rate": Decimal(1)}, "st_tax_rate"),
    ("zero nav", DETAILS, {"nav": Decimal(0)}, "nav"),
    ("target weight above one", TARGETS, {"weight": Decimal("1.5")}, "weight"),
    ("order side unknown", ORDERS, {"side": "HOLD"}, "side"),
    ("order quantity zero", ORDERS, {"quantity": 0}, "quantity"),
    ("order notional mismatch", ORDERS, {"notional": Decimal(999)}, "notional"),
]


@pytest.mark.parametrize(("schema", "row", "fragment"), [case[1:] for case in REJECT_CASES], ids=[case[0] for case in REJECT_CASES])
def test_out_of_domain_rows_are_rejected(frames: Frames, schema: FrameSchema, row: Mapping[str, object], fragment: str) -> None:
    with pytest.raises(FrameSchemaError) as info:
        validate_frame(frames.for_schema(schema)(row), schema)
    assert any(fragment in failure for failure in info.value.failures), info.value.failures


def test_duplicate_key_is_rejected(frames: Frames) -> None:
    with pytest.raises(FrameSchemaError, match="duplicate"):
        validate_frame(frames.holdings({"security_id": "A"}, {"security_id": "A"}), HOLDINGS)


def test_wrong_dtype_is_rejected_before_value_checks(frames: Frames) -> None:
    frame = frames.holdings().astype({"quantity": "int64"})
    with pytest.raises(FrameSchemaError, match="dtype 'int64', expected 'Int64'"):
        validate_frame(frame, HOLDINGS)


def test_missing_and_unexpected_columns_are_both_reported(frames: Frames) -> None:
    frame = frames.details().drop(columns=["cash"]).assign(extra=1)
    with pytest.raises(FrameSchemaError) as info:
        validate_frame(frame, DETAILS)
    assert len(info.value.failures) == 2
    assert "missing columns ['cash']" in info.value.failures[0]
    assert "unexpected columns ['extra']" in info.value.failures[1]


def test_universe_keeps_extra_columns_added_by_rules(frames: Frames) -> None:
    frame = frames.universe().assign(signal=pd.Series([0.5], dtype="Float64"))
    assert "signal" in validate_frame(frame, UNIVERSE).columns


def test_every_failure_is_listed_at_once(frames: Frames) -> None:
    frame = frames.holdings({"quantity": -1, "avg_cost": Decimal(-1)})
    with pytest.raises(FrameSchemaError) as info:
        validate_frame(frame, HOLDINGS)
    assert len(info.value.failures) == 2


def test_target_weights_must_sum_to_one_per_benchmark(frames: Frames) -> None:
    frame = frames.targets({"security_id": "A", "weight": Decimal("0.6")}, {"security_id": "B", "weight": Decimal("0.3")})
    with pytest.raises(FrameSchemaError, match="do not sum to 1"):
        validate_frame(frame, TARGETS)


@pytest.mark.parametrize("schema", [HOLDINGS, UNIVERSE], ids=["holdings", "universe"])
def test_analytics_tables_accept_columns_beyond_their_schema(frames: Frames, schema: FrameSchema) -> None:
    validate_frame(frames.for_schema(schema)().assign(score=pd.Series([0.5], dtype="Float64")), schema)


@pytest.mark.parametrize("schema", [DETAILS, TARGETS], ids=["details", "targets"])
def test_other_engine_frames_reject_unexpected_columns(frames: Frames, schema: FrameSchema) -> None:
    with pytest.raises(FrameSchemaError, match="unexpected columns \\['score'\\]"):
        validate_frame(frames.for_schema(schema)().assign(score=pd.Series([0.5], dtype="Float64")), schema)


def test_solve_order_must_be_unique(frames: Frames) -> None:
    frame = frames.portfolios({"portfolio_id": "P1"}, {"portfolio_id": "P2"})
    with pytest.raises(FrameSchemaError, match="solve_order"):
        validate_frame(frame, PORTFOLIOS)


def test_coerce_frame_builds_decimals_exactly_and_casts_declared_dtypes() -> None:
    raw = pd.DataFrame({"portfolio_id": ["P1"], "security_id": ["A"], "quantity": [5], "avg_cost": ["90.10"], "acquired_on": pd.to_datetime(["2024-01-15"]).tz_localize("UTC")})
    coerced = validate_frame(coerce_frame(raw, HOLDINGS), HOLDINGS)
    assert coerced["avg_cost"].iloc[0] == Decimal("90.10")
    assert str(coerced["quantity"].dtype) == "Int64"


def test_coerce_frame_reads_floats_by_shortest_repr() -> None:
    coerced = coerce_frame(pd.DataFrame({"weight": [0.1]}), FrameSchema("t", (ColumnSpec("weight", "decimal"),), ()))
    assert coerced["weight"].iloc[0] == Decimal("0.1")


def test_coerce_frame_rejects_unparseable_decimal_text() -> None:
    with pytest.raises(ValueError, match="cannot coerce"):
        coerce_frame(pd.DataFrame({"weight": ["abc"]}), FrameSchema("t", (ColumnSpec("weight", "decimal"),), ()))


def test_schema_rejects_key_column_that_is_optional() -> None:
    with pytest.raises(ValueError, match="key column"):
        FrameSchema("t", (ColumnSpec("id", "string", required=False),), ("id",))


def test_column_spec_rejects_bounds_on_strings() -> None:
    with pytest.raises(ValueError, match="bounds apply only"):
        ColumnSpec("name", "string", ge=Decimal(0))
