"""Tier 1: stacking holdings and universe into one frame — column union, typed nulls, dtype promotion, and what is refused."""

from decimal import Decimal

import pandas as pd
import pytest

from portfolio_optimizer.domain.optimizer_frame import FrameStackError, column_dtype_conflicts, stack_frames


def held() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": pd.Series(["A"], dtype="string"),
            "quantity": pd.Series([5], dtype="Int64"),
            "avg_cost": pd.Series([Decimal(90)], dtype="object"),
            "lot_flag": pd.Series([True], dtype="bool"),
            "days_held": pd.Series([10], dtype="int64"),
        }
    )


def buyable() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": pd.Series(["A", "B"], dtype="string"),
            "price": pd.Series([Decimal(100), Decimal(50)], dtype="object"),
            "restricted": pd.Series([False, True], dtype="bool"),
            "raw_score": pd.Series([0.5, 0.7], dtype="float64"),
        }
    )


def test_columns_are_the_union_in_first_seen_order_with_typed_nulls() -> None:
    frame = stack_frames({"holdings": held(), "universe": buyable()})
    assert list(frame.columns) == ["security_id", "quantity", "avg_cost", "lot_flag", "days_held", "price", "restricted", "raw_score"]
    assert frame["security_id"].tolist() == ["A", "A", "B"]
    assert frame["quantity"].tolist() == [5, pd.NA, pd.NA] and str(frame["quantity"].dtype) == "Int64"
    assert frame["avg_cost"].tolist() == [Decimal(90), None, None] and str(frame["avg_cost"].dtype) == "object"
    assert frame["price"].tolist() == [None, Decimal(100), Decimal(50)]


@pytest.mark.parametrize(("column", "promoted"), [("lot_flag", "boolean"), ("days_held", "Int64"), ("raw_score", "Float64"), ("restricted", "boolean")])
def test_one_sided_numpy_columns_are_promoted_to_their_nullable_form(column: str, promoted: str) -> None:
    frame = stack_frames({"holdings": held(), "universe": buyable()})
    assert str(frame[column].dtype) == promoted
    assert frame[column].isna().sum() == (2 if column in held().columns else 1)


def test_a_column_present_on_both_sides_keeps_its_dtype() -> None:
    both = {"holdings": held().assign(flag=pd.Series([True], dtype="bool")), "universe": buyable().assign(flag=pd.Series([False, True], dtype="bool"))}
    assert str(stack_frames(both)["flag"].dtype) == "bool"


def test_source_column_tags_each_row_and_must_not_collide() -> None:
    frame = stack_frames({"holdings": held(), "universe": buyable()}, source_column="source")
    assert next(iter(frame.columns)) == "source"
    assert frame["source"].tolist() == ["holdings", "universe", "universe"]
    assert str(frame["source"].dtype) == "string"
    with pytest.raises(FrameStackError, match=r"source column 'price' is already a column of \['universe'\]"):
        stack_frames({"holdings": held(), "universe": buyable()}, source_column="price")


def test_shared_columns_with_different_dtypes_are_named_and_refused() -> None:
    frames = {"holdings": held().assign(score=pd.Series([1.0], dtype="Float64")), "universe": buyable().assign(score=pd.Series([1.0, 2.0], dtype="float64"))}
    assert column_dtype_conflicts(frames) == ["column 'score': holdings has dtype 'Float64', universe has 'float64'"]
    with pytest.raises(FrameStackError, match="column 'score'"):
        stack_frames(frames)
    assert column_dtype_conflicts({"holdings": held(), "universe": buyable()}) == []


def test_a_dtype_that_cannot_hold_a_null_is_refused_when_it_must() -> None:
    frames = {"holdings": held().assign(bucket=pd.Series([1], dtype="int32")), "universe": buyable()}
    with pytest.raises(FrameStackError, match="column 'bucket' has dtype 'int32', which cannot hold a null"):
        stack_frames(frames)


def test_empty_inputs() -> None:
    frame = stack_frames({"holdings": held().iloc[0:0], "universe": buyable()})
    assert frame["security_id"].tolist() == ["A", "B"]
    with pytest.raises(FrameStackError, match="at least one frame"):
        stack_frames({})
