"""Tier 1: the shipped rules — boundary, empty, normal — and that restricting is idempotent."""

from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from portfolio_optimizer.rules import CapSingleNameParams, LiquidityParams, add_zero_alpha, cap_single_name, restrict_low_liquidity
from tests.conftest import Factories, Frames, make_portfolio_data

# --- cap_single_name ---


@pytest.mark.parametrize(("style_limit", "param", "expected"), [("0.10", "0.05", "0.05"), ("0.10", "0.10", "0.10"), ("0.10", "0.20", "0.10")])
def test_cap_single_name_only_tightens(make: Factories, style_limit: str, param: str, expected: str) -> None:
    data = make.portfolio_data(style=make.style(max_weight=Decimal(style_limit)))
    result = cap_single_name(data, CapSingleNameParams(max_weight=Decimal(param)))
    assert result.style.max_weight == Decimal(expected)
    assert data.style.max_weight == Decimal(style_limit)


# --- add_zero_alpha ---


def test_add_zero_alpha_adds_a_float_column_only_when_missing(make: Factories, frames: Frames) -> None:
    added = add_zero_alpha(make.portfolio_data())
    assert str(added.universe["alpha"].dtype) == "Float64"
    assert added.universe["alpha"].tolist() == [0.0, 0.0, 0.0]
    universe = frames.three_security_universe().assign(alpha=np.array([0.1, 0.2, 0.3]))
    universe["alpha"] = universe["alpha"].astype("Float64")
    kept = add_zero_alpha(make.portfolio_data(universe=universe))
    assert kept.universe["alpha"].tolist() == [0.1, 0.2, 0.3]


def test_add_zero_alpha_on_an_empty_universe(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data(holdings=frames.holdings().iloc[0:0], universe=frames.universe().iloc[0:0], targets=frames.targets().iloc[0:0])
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
