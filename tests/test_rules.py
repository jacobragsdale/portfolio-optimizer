"""Tier 1: the shipped rules — boundary, empty, normal — and that restricting is idempotent."""

from decimal import Decimal

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from portfolio_optimizer.rules import AttachUniverseColumnsParams, CapSingleNameParams, LiquidityParams, add_zero_alpha, attach_universe_columns, cap_single_name, restrict_low_liquidity
from tests.conftest import Factories, Frames, make_portfolio_data

# --- cap_single_name ---


@pytest.mark.parametrize(("style_limit", "param", "expected"), [("0.10", "0.05", "0.05"), ("0.10", "0.10", "0.10"), ("0.10", "0.20", "0.10")])
def test_cap_single_name_only_tightens(make: Factories, style_limit: str, param: str, expected: str) -> None:
    data = make.portfolio_data(details=make.details(max_weight=Decimal(style_limit)))
    result = cap_single_name(data, CapSingleNameParams(max_weight=Decimal(param)))
    assert result.details.max_weight == Decimal(expected)
    assert data.details.max_weight == Decimal(style_limit)


# --- add_zero_alpha ---


def test_add_zero_alpha_adds_a_float_column_only_when_missing(make: Factories, frames: Frames) -> None:
    added = add_zero_alpha(make.portfolio_data(universe=frames.three_security_universe().drop(columns=["alpha"])))
    assert str(added.universe["alpha"].dtype) == "Float64"
    assert added.universe["alpha"].tolist() == [0.0, 0.0, 0.0]
    kept = add_zero_alpha(make.portfolio_data())
    assert kept.universe["alpha"].tolist() == [0.03, 0.01, 0.05]


def test_add_zero_alpha_on_an_empty_universe(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data(holdings=frames.holdings().iloc[0:0], universe=frames.universe().iloc[0:0])
    assert add_zero_alpha(data).universe["alpha"].tolist() == []


# --- restrict_low_liquidity ---


@pytest.mark.parametrize(("threshold", "restricted"), [(100_000, [False, False, False]), (100_001, [False, False, True]), (1_000_001, [True, True, True])])
def test_restrict_low_liquidity_freezes_names_strictly_below_the_threshold(make: Factories, threshold: int, restricted: list[bool]) -> None:
    result = restrict_low_liquidity(make.portfolio_data(), LiquidityParams(min_adv_shares=threshold))
    assert result.universe["restricted"].tolist() == restricted
    assert str(result.universe["restricted"].dtype) == "bool"


def test_restrict_low_liquidity_keeps_names_already_restricted(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[0, "restricted"] = True
    result = restrict_low_liquidity(make.portfolio_data(universe=universe), LiquidityParams(min_adv_shares=0))
    assert result.universe["restricted"].tolist() == [True, False, False]


@given(threshold=st.integers(min_value=0, max_value=2_000_000))
@settings(deadline=None, max_examples=25)
def test_restrict_low_liquidity_is_idempotent(threshold: int) -> None:
    params = LiquidityParams(min_adv_shares=threshold)
    once = restrict_low_liquidity(make_portfolio_data(), params)  # hypothesis tests cannot take fixtures
    twice = restrict_low_liquidity(once, params)
    assert_frame_equal(once.universe, twice.universe)


# --- attach_universe_columns ---


def _scored(frames: Frames, dtype: str = "Float64") -> pd.DataFrame:
    """The three-security universe carrying a `score` analytic."""
    universe = frames.three_security_universe()
    return universe.assign(score=pd.Series([1.0, 2.0, 3.0], dtype=dtype))


def test_attach_universe_columns_copies_every_analytic_by_default(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data(universe=_scored(frames))
    attached = attach_universe_columns(data, AttachUniverseColumnsParams())
    assert attached.holdings["score"].tolist() == [1.0, 2.0], "A and B take the universe's score for their own security"
    assert str(attached.holdings["score"].dtype) == "Float64", "the dtype comes across too, which is what stacking the two tables requires"
    assert attached.optimizer_frame()["score"].tolist() == [1.0, 2.0, 1.0, 2.0, 3.0]


def test_attach_universe_columns_leaves_schema_columns_where_they_are(make: Factories, frames: Frames) -> None:
    attached = attach_universe_columns(make.portfolio_data(universe=_scored(frames)), AttachUniverseColumnsParams())
    assert "price" not in attached.holdings.columns and "sector" not in attached.holdings.columns, "the universe's own schema columns are not analytics"


def test_attach_universe_columns_copies_only_what_it_is_asked_for(make: Factories, frames: Frames) -> None:
    universe = _scored(frames).assign(momentum=pd.Series([0.1, 0.2, 0.3], dtype="Float64"))
    attached = attach_universe_columns(make.portfolio_data(universe=universe), AttachUniverseColumnsParams(columns=("score",)))
    assert "score" in attached.holdings.columns and "momentum" not in attached.holdings.columns


def test_attach_universe_columns_is_a_no_op_when_holdings_already_agree(make: Factories, frames: Frames) -> None:
    once = attach_universe_columns(make.portfolio_data(universe=_scored(frames)), AttachUniverseColumnsParams())
    twice = attach_universe_columns(once, AttachUniverseColumnsParams())
    assert_frame_equal(once.holdings, twice.holdings)


def test_attach_universe_columns_gives_a_held_name_outside_the_universe_a_null(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000}, {"security_id": "Z", "quantity": 10, "avg_cost": Decimal(1)})
    attached = attach_universe_columns(make.portfolio_data(holdings=holdings, universe=_scored(frames)), AttachUniverseColumnsParams())
    assert attached.holdings["score"].tolist()[0] == 1.0
    assert pd.isna(attached.holdings["score"].tolist()[1]), "Z is held but not in the universe; the build refuses it later, by name"


def test_attach_universe_columns_refuses_a_bool_analytic_it_cannot_fill(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(vendor_flag=pd.Series([True, False, True], dtype="bool"))
    holdings = frames.holdings({"security_id": "Z", "quantity": 10, "avg_cost": Decimal(1)})
    data = make.portfolio_data(holdings=holdings, universe=universe)
    with pytest.raises(ValueError, match=r"\['vendor_flag'\] are bool and cannot hold the null"):
        attach_universe_columns(data, AttachUniverseColumnsParams())


def test_attach_universe_columns_names_a_column_the_universe_does_not_have(make: Factories) -> None:
    with pytest.raises(ValueError, match=r"the universe has no column\(s\) \['score'\]"):
        attach_universe_columns(make.portfolio_data(), AttachUniverseColumnsParams(columns=("score",)))


def test_attach_universe_columns_refuses_to_overwrite_a_column_holdings_has(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000}).assign(score=pd.Series([9.0], dtype="Float64"))
    data = make.portfolio_data(holdings=holdings, universe=_scored(frames))
    with pytest.raises(ValueError, match=r"holdings already has column\(s\) \['score'\]"):
        attach_universe_columns(data, AttachUniverseColumnsParams(columns=("score",)))
