"""Tier 1: building the spec — alignment, the single Decimal→float64 conversion, and bounds."""

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.engine.build import LONG_TERM_HOLDING, BuildError, build_problem_spec, to_float64
from tests.conftest import AS_OF, Factories, Frames


def test_spec_aligns_to_the_sorted_universe(make: Factories) -> None:
    output = build_problem_spec(make.portfolio_data())
    spec = output.spec
    assert spec.security_ids == ("A", "B", "C")
    np.testing.assert_allclose(spec.w0, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(spec.shares_held, [5000.0, 10000.0, 0.0])
    np.testing.assert_array_equal(spec.price, [100.0, 50.0, 10.0])
    np.testing.assert_allclose(spec.w_target, [1 / 3, 1 / 3, 1 / 3], atol=1e-15)
    np.testing.assert_array_equal(spec.lb, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(spec.ub, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(spec.adv_capacity, [100.0, 50.0, 1.0])
    assert spec.sector_names == ("TECH",)
    np.testing.assert_array_equal(spec.sector_matrix, [[1.0, 1.0, 1.0]])
    assert output.order_inputs.price == (Decimal(100), Decimal(50), Decimal(10))
    assert output.order_inputs.shares_held == (5000, 10000, 0)
    assert output.order_inputs.nav == Decimal(1_000_000)


def test_current_weights_sum_to_one_minus_cash(make: Factories) -> None:
    details = make.details(nav=Decimal(1_250_000), cash=Decimal(250_000))
    spec = build_problem_spec(make.portfolio_data(details=details)).spec
    assert spec.w0.sum() == pytest.approx(0.8, abs=1e-12)


@pytest.mark.parametrize(("held_for", "rate"), [(LONG_TERM_HOLDING, 0.40), (LONG_TERM_HOLDING + timedelta(seconds=1), 0.20)])
def test_tax_per_dollar_switches_to_the_long_term_rate_strictly_after_a_year(make: Factories, frames: Frames, held_for: timedelta, rate: float) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(90), "acquired_on": AS_OF - held_for})
    spec = build_problem_spec(make.portfolio_data(holdings=holdings)).spec
    assert spec.tax_per_dollar[0] == pytest.approx(0.1 * rate, abs=1e-15)
    assert spec.tax_per_dollar[1:].tolist() == [0.0, 0.0]


def test_a_loss_gives_a_negative_tax_per_dollar(make: Factories, frames: Frames) -> None:
    spec = build_problem_spec(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "avg_cost": Decimal(110)}))).spec
    assert spec.tax_per_dollar[0] == pytest.approx(-0.1 * 0.20, abs=1e-15)  # held since 2024: long-term rate


def test_transaction_cost_comes_from_the_optional_bps_column(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(tcost_bps=pd.Series([Decimal(5), Decimal(0), Decimal("2.5")], dtype="object"))
    spec = build_problem_spec(make.portfolio_data(universe=universe)).spec
    np.testing.assert_allclose(spec.tcost_per_dollar, [0.0005, 0.0, 0.00025])


def test_restricted_names_are_frozen_at_their_current_weight(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[0, "restricted"] = True
    spec = build_problem_spec(make.portfolio_data(universe=universe)).spec
    assert spec.lb[0] == spec.ub[0] == 0.5


def test_per_security_bound_columns_only_tighten_the_style_limits(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(max_weight=pd.Series([Decimal("0.3"), None, None], dtype="object"), min_weight=pd.Series([None, Decimal("0.1"), None], dtype="object"))
    spec = build_problem_spec(make.portfolio_data(universe=universe, style=make.style(max_weight=Decimal("0.6")))).spec
    np.testing.assert_array_equal(spec.ub, [0.3, 0.6, 0.6])
    np.testing.assert_array_equal(spec.lb, [0.0, 0.1, 0.0])


def test_crossed_per_security_bounds_are_rejected(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(max_weight=pd.Series([Decimal("0.1"), None, None], dtype="object"), min_weight=pd.Series([Decimal("0.2"), None, None], dtype="object"))
    with pytest.raises(BuildError, match=r"A: lower bound 0\.2 exceeds upper bound 0\.1"):
        build_problem_spec(make.portfolio_data(universe=universe))


def test_extra_numeric_columns_are_exported_by_name(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(alpha=pd.Series([0.1, 0.2, 0.3], dtype="Float64"), momentum=pd.Series([1, 2, 3], dtype="Int64"), note=pd.Series(["x", "y", "z"], dtype="string"))
    spec = build_problem_spec(make.portfolio_data(universe=universe)).spec
    assert set(spec.columns) == {"alpha", "momentum"}
    np.testing.assert_array_equal(spec.column("momentum"), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_boolean_columns_are_exported_as_real_boolean_flags(make: Factories, frames: Frames, dtype: str) -> None:
    universe = frames.three_security_universe().assign(esg=pd.Series([True, False, True], dtype=dtype))
    spec = build_problem_spec(make.portfolio_data(universe=universe)).spec
    assert "esg" not in spec.columns
    assert spec.flag("esg").dtype == np.bool_
    np.testing.assert_array_equal(spec.flag("esg"), [True, False, True])


def test_null_in_a_flag_column_is_rejected(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(esg=pd.Series([True, None, True], dtype="boolean"))
    with pytest.raises(BuildError, match="flag column 'esg' has null values"):
        build_problem_spec(make.portfolio_data(universe=universe))


def test_null_in_an_exported_column_is_rejected(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(momentum=pd.Series([1, None, 3], dtype="Int64"))
    with pytest.raises(BuildError, match="column 'momentum' has null values"):
        build_problem_spec(make.portfolio_data(universe=universe))


def test_a_held_name_outside_the_universe_cannot_be_built(make: Factories, frames: Frames) -> None:
    with pytest.raises(BuildError, match="held securities missing from universe \\['Z'\\]"):
        build_problem_spec(make.portfolio_data(holdings=frames.holdings({"security_id": "Z"})))


def test_holdings_analytics_columns_do_not_reach_the_spec(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A"}).assign(lot_score=pd.Series([0.5], dtype="Float64"))
    spec = build_problem_spec(make.portfolio_data(holdings=holdings)).spec
    assert spec.columns == {}


@pytest.mark.parametrize("bad", [[1.5], [True], [Decimal("NaN")], [Decimal("Infinity")], ["1"]])
def test_to_float64_rejects_anything_but_finite_decimals_and_ints(bad: list[object]) -> None:
    with pytest.raises(BuildError):
        to_float64(bad, "x")  # ty: ignore[invalid-argument-type]  # the wrong element type is the case under test


def test_to_float64_is_correctly_rounded_and_handles_empty() -> None:
    assert to_float64([Decimal("0.1"), 3], "x").tolist() == [0.1, 3.0]
    assert to_float64([], "x").shape == (0,)


def test_empty_holdings_give_zero_weights(make: Factories, frames: Frames) -> None:
    spec = build_problem_spec(make.portfolio_data(holdings=frames.holdings().iloc[0:0])).spec
    np.testing.assert_array_equal(spec.w0, [0.0, 0.0, 0.0])
    assert spec.tax_per_dollar.tolist() == [0.0, 0.0, 0.0]


def test_two_builds_of_the_same_bundle_hash_equal(make: Factories) -> None:
    data = make.portfolio_data()
    assert build_problem_spec(data).spec.content_hash() == build_problem_spec(data).spec.content_hash()
