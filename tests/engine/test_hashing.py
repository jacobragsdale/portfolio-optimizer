"""Tier 1: content hashes are canonical — order-, index-, and representation-independent, dtype-sensitive."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd

from portfolio_optimizer.engine.hashing import frame_sha256, json_sha256
from tests.conftest import Frames


def test_frame_hash_ignores_row_order_column_order_and_index(frames: Frames) -> None:
    a = frames.holdings({"security_id": "A"}, {"security_id": "B"})
    b = frames.holdings({"security_id": "B"}, {"security_id": "A"})
    shuffled_columns = b[list(reversed(b.columns))]
    reindexed = shuffled_columns.set_axis([10, 20])
    key = ("portfolio_id", "security_id")
    assert frame_sha256(a, key) == frame_sha256(b, key) == frame_sha256(shuffled_columns, key) == frame_sha256(reindexed, key)


def test_frame_hash_changes_with_a_value_or_a_dtype(frames: Frames) -> None:
    base = frames.holdings()
    assert frame_sha256(base) != frame_sha256(frames.holdings({"quantity": 5001}))
    assert frame_sha256(base) != frame_sha256(base.astype({"quantity": "int64"}))


def test_frame_hash_normalizes_decimal_representation(frames: Frames) -> None:
    assert frame_sha256(frames.holdings({"avg_cost": Decimal("90.50")})) == frame_sha256(frames.holdings({"avg_cost": Decimal("90.5")}))
    assert frame_sha256(frames.holdings({"avg_cost": Decimal("90.50")})) != frame_sha256(frames.holdings({"avg_cost": Decimal("90.51")}))


def test_frame_hash_treats_equal_instants_in_different_zones_equally(frames: Frames) -> None:
    utc = frames.holdings({"acquired_on": datetime(2024, 1, 15, 12, tzinfo=UTC)})
    shifted = utc.assign(acquired_on=utc["acquired_on"].dt.tz_convert("America/New_York").astype("datetime64[ns, America/New_York]"))
    assert frame_sha256(utc) == frame_sha256(shifted)


def test_frame_hash_handles_nulls_and_empty_frames() -> None:
    with_null = pd.DataFrame({"x": pd.Series([1, None], dtype="Int64")})
    without = pd.DataFrame({"x": pd.Series([1, 2], dtype="Int64")})
    assert frame_sha256(with_null) != frame_sha256(without)
    assert frame_sha256(pd.DataFrame({"x": pd.Series([], dtype="Int64")})) == frame_sha256(pd.DataFrame({"x": pd.Series([], dtype="Int64")}))


def test_json_hash_is_canonical() -> None:
    assert json_sha256({"b": 1, "a": [1, 2]}) == json_sha256({"a": [1, 2], "b": 1})
    assert json_sha256({"a": Decimal("0.5")}) == json_sha256({"a": "0.5"})
    assert json_sha256({"a": 1}) != json_sha256({"a": 2})
