"""Tier 1: the shipped assembly steps on small frames — join claims, union dtype rules, select, drop — and ``Frames`` itself."""

from decimal import Decimal

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.assembly import DropParams, JoinParams, SelectParams, UnionParams, drop, join, select, union
from portfolio_optimizer.domain.data import Frames
from portfolio_optimizer.domain.optimizer_frame import FrameStackError


def securities(*ids: str, **columns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"security_id": pd.Series(list(ids), dtype="string"), **columns})


def prices(*rows: tuple[str, str]) -> pd.DataFrame:
    return securities(*(security for security, _ in rows), price=pd.Series([Decimal(price) for _, price in rows], dtype="object"))


@pytest.fixture
def datasets() -> Frames:
    universe = securities("A", "B", sector=pd.Series(["TECH", "ENERGY"], dtype="string"))
    vendor = securities("A", "B", "C", score=pd.Series([0.1, 0.2, 0.3], dtype="Float64"), rank=pd.Series([1, 2, 3], dtype="Int64"))
    return Frames({"universe": universe, "vendor": vendor, "prices": prices(("A", "100"), ("B", "50"))})


# --- Frames ---


def test_frames_is_an_immutable_mapping_that_names_what_it_has(datasets: Frames) -> None:
    assert set(datasets) == {"universe", "vendor", "prices"}
    assert datasets.row_counts() == {"universe": 2, "vendor": 3, "prices": 2}
    with pytest.raises(KeyError, match=r"no dataset 'sectors'; available: \['prices', 'universe', 'vendor'\]"):
        datasets["sectors"]
    added = datasets.with_frame("sectors", securities("A"))
    assert "sectors" in added and "sectors" not in datasets
    assert set(added.without("vendor", "prices")) == {"universe", "sectors"}
    with pytest.raises(KeyError, match=r"no dataset\(s\) \['nope'\]"):
        datasets.without("nope")
    with pytest.raises(TypeError, match="is a dict, expected DataFrame"):
        Frames({"bad": {}})  # ty: ignore[invalid-argument-type]  # the wrong value type is the case under test


# --- join ---


def test_join_brings_every_non_key_column_by_default(datasets: Frames) -> None:
    result = join(datasets, JoinParams(into="universe", source="vendor", on=("security_id",), cardinality="one_to_one"))
    assert list(result["universe"].columns) == ["security_id", "sector", "score", "rank"]
    assert result["universe"]["rank"].tolist() == [1, 2]
    assert result["vendor"] is datasets["vendor"]


def test_join_can_pick_and_rename_the_columns_it_brings(datasets: Frames) -> None:
    params = JoinParams(into="universe", source="vendor", on=("security_id",), cardinality="one_to_one", columns=("score",), rename={"score": "vendor_score"})
    result = join(datasets, params)
    assert list(result["universe"].columns) == ["security_id", "sector", "vendor_score"]
    with pytest.raises(ValueError, match=r"columns \['security_id', 'beta'\] are not non-key columns of vendor"):
        join(datasets, JoinParams(into="universe", source="vendor", on=("security_id",), cardinality="one_to_one", columns=("security_id", "beta")))
    with pytest.raises(ValueError, match=r"rename refers to columns \['rank'\]"):
        join(datasets, JoinParams(into="universe", source="vendor", on=("security_id",), cardinality="one_to_one", columns=("score",), rename={"rank": "r"}))


def test_join_aligns_key_dtypes_so_a_text_key_still_matches(datasets: Frames) -> None:
    vendor = datasets["vendor"].astype({"security_id": "object"})
    result = join(datasets.with_frame("vendor", vendor), JoinParams(into="universe", source="vendor", on=("security_id",), cardinality="one_to_one"))
    assert result["universe"]["score"].tolist() == [0.1, 0.2]
    assert str(result["universe"]["security_id"].dtype) == "string"


def test_join_left_keeps_unmatched_rows_with_nulls_and_inner_drops_them(datasets: Frames) -> None:
    into = datasets.with_frame("universe", securities("A", "Z"))
    left = join(into, JoinParams(into="universe", source="prices", on=("security_id",), cardinality="one_to_one"))
    assert left["universe"]["price"].tolist() == [Decimal(100), None]
    inner = join(into, JoinParams(into="universe", source="prices", on=("security_id",), cardinality="one_to_one", how="inner"))
    assert inner["universe"]["security_id"].tolist() == ["A"]
    with pytest.raises(ValueError, match=r"1 row\(s\) of universe had no match in prices, e.g. \['Z'\]"):
        join(into, JoinParams(into="universe", source="prices", on=("security_id",), cardinality="one_to_one", how="inner", require_all_matched=True))


def test_join_enforces_the_declared_cardinality(datasets: Frames) -> None:
    duplicated = pd.concat([datasets["prices"], datasets["prices"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="cardinality 'one_to_one' violated"):
        join(datasets.with_frame("prices", duplicated), JoinParams(into="universe", source="prices", on=("security_id",), cardinality="one_to_one"))
    many = join(datasets.with_frame("prices", duplicated), JoinParams(into="universe", source="prices", on=("security_id",), cardinality="one_to_many"))
    assert len(many["universe"]) == 3


def test_join_overwrites_only_when_told_to(datasets: Frames) -> None:
    vendor = datasets["vendor"].assign(sector=pd.Series(["X", "Y", "Z"], dtype="string"))
    params = JoinParams(into="universe", source="vendor", on=("security_id",), cardinality="one_to_one", columns=("sector",))
    with pytest.raises(ValueError, match=r"would overwrite columns \['sector'\] already present in universe; set overwrite"):
        join(datasets.with_frame("vendor", vendor), params)
    result = join(datasets.with_frame("vendor", vendor), params.model_copy(update={"overwrite": True}))
    assert result["universe"]["sector"].tolist() == ["X", "Y"]


def test_join_rejects_self_joins_and_missing_keys(datasets: Frames) -> None:
    with pytest.raises(ValueError, match="cannot join 'universe' into itself"):
        join(datasets, JoinParams(into="universe", source="universe", on=("security_id",), cardinality="one_to_one"))
    with pytest.raises(ValueError, match=r"join columns missing: universe lacks \['isin'\], vendor lacks \['isin'\]"):
        join(datasets, JoinParams(into="universe", source="vendor", on=("isin",), cardinality="one_to_one"))


# --- union ---


def test_union_stacks_sources_with_typed_nulls_and_drops_them(datasets: Frames) -> None:
    result = union(datasets, UnionParams(into="all", sources=("universe", "vendor")))
    stacked = result["all"]
    assert set(result) == {"all", "prices"}
    assert stacked["security_id"].tolist() == ["A", "B", "A", "B", "C"]
    assert str(stacked["sector"].dtype) == "string" and stacked["sector"].isna().tolist() == [False, False, True, True, True]
    assert str(stacked["rank"].dtype) == "Int64" and stacked["rank"].tolist()[:2] == [pd.NA, pd.NA]


def test_union_can_keep_sources_and_tag_rows(datasets: Frames) -> None:
    result = union(datasets, UnionParams(into="universe", sources=("universe", "vendor"), source_column="origin", keep_sources=True))
    assert set(result) == {"universe", "vendor", "prices"}
    assert result["universe"]["origin"].tolist() == ["universe", "universe", "vendor", "vendor", "vendor"]


def test_union_refuses_to_replace_a_dataset_that_is_not_a_source(datasets: Frames) -> None:
    with pytest.raises(ValueError, match="'prices' already exists and is not among the sources"):
        union(datasets, UnionParams(into="prices", sources=("universe", "vendor")))


def test_union_refuses_conflicting_dtypes(datasets: Frames) -> None:
    vendor = datasets["vendor"].astype({"security_id": "object"})
    with pytest.raises(FrameStackError, match="column 'security_id': universe has dtype 'string', vendor has 'object'"):
        union(datasets.with_frame("vendor", vendor), UnionParams(into="all", sources=("universe", "vendor")))


# --- select and drop ---


def test_select_keeps_drops_and_renames(datasets: Frames) -> None:
    kept = select(datasets, SelectParams(dataset="vendor", columns=("security_id", "rank"), rename={"rank": "vendor_rank"}))
    assert list(kept["vendor"].columns) == ["security_id", "vendor_rank"]
    dropped = select(datasets, SelectParams(dataset="vendor", drop=("rank",)))
    assert list(dropped["vendor"].columns) == ["security_id", "score"]
    assert_frame_equal(select(datasets, SelectParams(dataset="vendor"))["vendor"], datasets["vendor"])


def test_select_rejects_contradictions_and_unknown_columns(datasets: Frames) -> None:
    with pytest.raises(ValueError, match="columns and drop are mutually exclusive"):
        select(datasets, SelectParams(dataset="vendor", columns=("score",), drop=("rank",)))
    with pytest.raises(ValueError, match=r"vendor has no columns \['beta'\]"):
        select(datasets, SelectParams(dataset="vendor", rename={"beta": "b"}))


def test_drop_removes_datasets(datasets: Frames) -> None:
    assert set(drop(datasets, DropParams(datasets=("vendor", "prices")))) == {"universe"}
    with pytest.raises(KeyError, match=r"no dataset\(s\) \['sectors'\]"):
        drop(datasets, DropParams(datasets=("sectors",)))
