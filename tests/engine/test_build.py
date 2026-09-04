"""Tier 1: the standard build — alignment, the single Decimal→float64 conversion, bounds, and what it exports by name; and the order inputs derived from any spec."""

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from portfolio_optimizer.domain.constraints import adv_remaining
from portfolio_optimizer.domain.data import PortfolioDataError
from portfolio_optimizer.domain.results import ChainState
from portfolio_optimizer.engine.build import LONG_TERM_HOLDING, BuildError, StandardParams, order_inputs, standard, to_float64
from tests.conftest import AS_OF, Factories, Frames


def test_spec_aligns_to_the_sorted_universe(make: Factories) -> None:
    data = make.portfolio_data()
    spec = standard(data)
    assert spec.security_ids == ("A", "B", "C")
    np.testing.assert_allclose(spec.w0, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(spec.quantity_held, [5000.0, 10000.0, 0.0])
    np.testing.assert_array_equal(spec.price, [100.0, 50.0, 10.0])
    np.testing.assert_array_equal(spec.lb, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(spec.ub, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(spec.column("adv_capacity"), [100.0, 50.0, 1.0])
    assert spec.group("sector").names == ("TECH",)
    np.testing.assert_array_equal(spec.group("sector").matrix.toarray(), [[1.0, 1.0, 1.0]])
    inputs = order_inputs(data, spec)
    assert inputs.price == (Decimal(100), Decimal(50), Decimal(10))
    assert inputs.quantity_held == (5000, 10000, 0)
    assert inputs.nav == Decimal(1_000_000)
    assert inputs.ub == (Decimal(1), Decimal(1), Decimal(1)), "the bounds the order step clamps to come from the spec, whichever build made it"


def test_the_accounts_numbers_are_exported_as_scalars_and_every_extra_details_column_too(make: Factories) -> None:
    spec = standard(make.portfolio_data(details=make.details(max_turnover=Decimal("0.5"), cash_ub=Decimal("0.02"), extra={"max_issuer_weight": Decimal("0.1"), "benchmark": "SPX", "active": True})))
    assert spec.scalar("max_turnover") == 0.5 and spec.scalar("cash_ub") == 0.02 and spec.scalar("cash_lb") == 0.0
    assert spec.scalar("max_issuer_weight") == 0.1, "a numeric column the schema does not declare is a scalar a constraint row can bound against"
    assert "benchmark" not in spec.scalars and "active" not in spec.scalars, "text and flags are not numbers"


def test_every_string_column_is_a_grouping_and_restricted_is_a_flag(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(sector=pd.Series(["TECH", "FIN", "TECH"], dtype="string"), country=pd.Series(["US", "US", "GB"], dtype="string"))
    spec = standard(make.portfolio_data(universe=universe))
    assert spec.group("sector").names == ("FIN", "TECH")
    assert isinstance(spec.group("sector").matrix, csr_array) and spec.group("sector").matrix.nnz == 3
    np.testing.assert_array_equal(spec.group("sector").matrix.toarray(), [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(spec.group("country").row("GB").toarray(), [[0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(spec.flag("restricted"), [False, False, False])


def test_a_universe_of_only_ids_and_prices_builds(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().drop(columns=["sector", "adv_quantity", "increment", "restricted", "tcost_bps"])
    spec = standard(make.portfolio_data(universe=universe))
    assert set(spec.columns) == {"alpha", "tax_per_dollar"} and spec.groups == {} and spec.flags == {}, (
        "what the universe does not carry the spec does not have, and a constraint that needs it says so"
    )
    assert order_inputs(make.portfolio_data(universe=universe), spec).increment == (1, 1, 1)


def test_current_weights_sum_to_one_minus_cash(make: Factories) -> None:
    details = make.details(nav=Decimal(1_250_000), cash=Decimal(250_000))
    spec = standard(make.portfolio_data(details=details))
    assert spec.w0.sum() == pytest.approx(0.8, abs=1e-12)


@pytest.mark.parametrize(("held_for", "rate"), [(LONG_TERM_HOLDING, 0.40), (LONG_TERM_HOLDING + timedelta(seconds=1), 0.20)])
def test_tax_per_dollar_switches_to_the_long_term_rate_strictly_after_a_year(make: Factories, frames: Frames, held_for: timedelta, rate: float) -> None:
    holdings = frames.holdings({"security_id": "A", "quantity": 5000, "avg_cost": Decimal(90), "acquired_on": AS_OF - held_for})
    spec = standard(make.portfolio_data(holdings=holdings))
    assert spec.column("tax_per_dollar")[0] == pytest.approx(0.1 * rate, abs=1e-15)
    assert spec.column("tax_per_dollar")[1:].tolist() == [0.0, 0.0]


def test_lots_of_one_name_are_summed_and_taxed_pro_rata(make: Factories, frames: Frames) -> None:
    """Two lots of A against a price of 100: 1,000 bought at 90 two years ago (a long-term gain) and 1,000 at 110 last month (a short-term loss). The build holds 2,000 and taxes a pro-rata sale: (10 at 20% less 10 at 40%) / 2 per dollar of proceeds."""
    holdings = frames.holdings(
        {"security_id": "A", "lot_id": "old", "quantity": 1000, "avg_cost": Decimal(90)},
        {"security_id": "A", "lot_id": "new", "quantity": 1000, "avg_cost": Decimal(110), "acquired_on": AS_OF - timedelta(days=30)},
    )
    spec = standard(make.portfolio_data(holdings=holdings))
    assert spec.quantity_held[0] == 2000.0
    assert spec.column("tax_per_dollar")[0] == pytest.approx(-0.01, abs=1e-15)
    with pytest.raises(PortfolioDataError, match="lot_id"):
        make.portfolio_data(holdings=frames.holdings({"security_id": "A", "quantity": 1000}, {"security_id": "A", "quantity": 1000}))


def test_accrued_interest_values_the_book_and_the_tax_is_on_the_clean_price(make: Factories, frames: Frames) -> None:
    """A at 100 clean carries 2 of accrued: the 5,000 held are worth 510,000, and the gain on a cost of 90 is 10 per unit taxed at the long-term rate over 102 of proceeds."""
    universe = frames.three_security_universe().assign(accrued_interest=pd.Series([Decimal(2), Decimal(0), Decimal(0)], dtype="object"))
    spec = standard(make.portfolio_data(universe=universe))
    assert spec.price[0] == 102.0 and spec.w0[0] == pytest.approx(0.51)
    assert spec.column("tax_per_dollar")[0] == pytest.approx(10 * 0.2 / 102, abs=1e-15)


def test_a_loss_gives_a_negative_tax_per_dollar(make: Factories, frames: Frames) -> None:
    spec = standard(make.portfolio_data(holdings=frames.holdings({"security_id": "A", "avg_cost": Decimal(110)})))
    assert spec.column("tax_per_dollar")[0] == pytest.approx(-0.1 * 0.20, abs=1e-15)  # held since 2024: long-term rate


def test_transaction_cost_comes_from_the_optional_bps_column(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(tcost_bps=pd.Series([Decimal(5), Decimal(0), Decimal("2.5")], dtype="object"))
    spec = standard(make.portfolio_data(universe=universe))
    np.testing.assert_allclose(spec.column("tcost_per_dollar"), [0.0005, 0.0, 0.00025])


def test_the_derived_columns_need_the_accounts_rates_and_participation(make: Factories) -> None:
    """Without tax rates there is no `tax_per_dollar`, and without a participation no `adv_capacity` however liquid the universe: a term or row that needs either is refused by name at build rather than fed a zero."""
    spec = standard(make.portfolio_data(details=make.details(st_tax_rate=None, lt_tax_rate=None, max_adv_participation=None)))
    assert "tax_per_dollar" not in spec.columns and "adv_capacity" not in spec.columns
    assert {"st_tax_rate", "lt_tax_rate", "max_adv_participation"}.isdisjoint(spec.scalars)
    assert "tax_per_dollar" in standard(make.portfolio_data(details=make.details(max_adv_participation=None))).columns, "the rates alone give the tax column"


def test_adv_capacity_is_net_of_what_an_earlier_run_consumed(make: Factories, frames: Frames) -> None:
    """A's day is half spent, B's whole day is, and C's is oversubscribed: the budget is the participation times what is left, never negative."""
    universe = frames.three_security_universe().assign(adv_consumed_quantity=pd.Series([500_000, 1_000_000, 200_000], dtype="Int64"))
    spec = standard(make.portfolio_data(universe=universe))
    np.testing.assert_allclose(spec.column("adv_capacity"), [50.0, 0.0, 0.0])
    assert "adv_consumed_quantity" not in spec.columns, "folded into the capacity, like adv_quantity, not exported again"


def test_an_earlier_runs_consumption_and_a_predecessors_leave_the_same_budget(make: Factories, frames: Frames) -> None:
    """ADV 1,000, participation 10%, 50 shares already traded: 50 shares of budget left, whether the 50 came from an earlier run's orders or from a predecessor in this one."""
    universe = frames.universe({"security_id": "A", "price": Decimal(100), "adv_quantity": 1000, "adv_consumed_quantity": 50})
    data = make.portfolio_data(holdings=frames.holdings({"security_id": "A"}), universe=universe, details=make.details(max_adv_participation=Decimal("0.1")))
    after_earlier_run = standard(data)
    fresh = standard(data.with_changes(universe=universe.drop(columns=["adv_consumed_quantity"])))
    after_predecessor = adv_remaining(fresh, ChainState(security_ids=fresh.security_ids, traded_quantity=np.array([50.0])))
    budget_left = 50 * 100 / 1_000_000
    np.testing.assert_allclose(after_earlier_run.column("adv_capacity"), [budget_left])
    np.testing.assert_allclose(after_predecessor, [budget_left])


def test_restricted_names_are_frozen_at_their_current_weight(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[0, "restricted"] = True
    spec = standard(make.portfolio_data(universe=universe))
    assert spec.lb[0] == spec.ub[0] == 0.5


def test_per_security_bound_columns_only_tighten_the_style_limits(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(max_weight=pd.Series([Decimal("0.3"), None, None], dtype="object"), min_weight=pd.Series([None, Decimal("0.1"), None], dtype="object"))
    spec = standard(make.portfolio_data(universe=universe, details=make.details(max_weight=Decimal("0.6"))))
    np.testing.assert_array_equal(spec.ub, [0.3, 0.6, 0.6])
    np.testing.assert_array_equal(spec.lb, [0.0, 0.1, 0.0])


def test_crossed_per_security_bounds_are_rejected(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(max_weight=pd.Series([Decimal("0.1"), None, None], dtype="object"), min_weight=pd.Series([Decimal("0.2"), None, None], dtype="object"))
    with pytest.raises(BuildError, match=r"A: lower bound 0\.2 exceeds upper bound 0\.1"):
        standard(make.portfolio_data(universe=universe))


def test_hold_breached_starts_moves_the_breached_bound_to_the_current_weight(make: Factories, frames: Frames) -> None:
    """A and B start at half the book against a 40% cap, C at nothing against a 20% floor: held, each breached bound moves to the weight, so the name leaves the buyable (or sellable) set instead of failing the start; the untouched side and the other bounds are as they were, and the default build is unchanged."""
    universe = frames.three_security_universe().assign(min_weight=pd.Series([None, None, Decimal("0.2")], dtype="object"))
    data = make.portfolio_data(universe=universe, details=make.details(max_weight=Decimal("0.4")))
    strict = standard(data)
    np.testing.assert_array_equal(strict.ub, [0.4, 0.4, 0.4])
    np.testing.assert_array_equal(strict.lb, [0.0, 0.0, 0.2])
    held = standard(data, StandardParams(hold_breached_starts=True))
    np.testing.assert_array_equal(held.ub, [0.5, 0.5, 0.4])
    np.testing.assert_array_equal(held.lb, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(held.buyable, [False, False, True])
    np.testing.assert_array_equal(held.sellable, [True, True, False])
    assert held.content_hash() != strict.content_hash(), "the policy changes the problem the solver sees, so it changes the spec hash"
    assert standard(data, StandardParams()).content_hash() == strict.content_hash()


def test_extra_numeric_columns_are_exported_by_name(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(alpha=pd.Series([0.1, 0.2, 0.3], dtype="Float64"), momentum=pd.Series([1, 2, 3], dtype="Int64"), note=pd.Series(["x", "y", "z"], dtype="string"))
    spec = standard(make.portfolio_data(universe=universe))
    assert set(spec.columns) == {"alpha", "momentum", "tax_per_dollar", "tcost_per_dollar", "adv_capacity"}
    np.testing.assert_array_equal(spec.column("momentum"), [1.0, 2.0, 3.0])
    assert spec.group("note").names == ("x", "y", "z"), "a string column is a grouping, however many groups it has"


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_boolean_columns_are_exported_as_real_boolean_flags(make: Factories, frames: Frames, dtype: str) -> None:
    universe = frames.three_security_universe().assign(esg=pd.Series([True, False, True], dtype=dtype))
    spec = standard(make.portfolio_data(universe=universe))
    assert "esg" not in spec.columns
    assert spec.flag("esg").dtype == np.bool_
    np.testing.assert_array_equal(spec.flag("esg"), [True, False, True])


def test_null_in_a_flag_column_is_rejected(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe().assign(esg=pd.Series([True, None, True], dtype="boolean"))
    with pytest.raises(BuildError, match="flag column 'esg' has null values"):
        standard(make.portfolio_data(universe=universe))


def test_null_in_an_exported_column_or_grouping_is_rejected(make: Factories, frames: Frames) -> None:
    with pytest.raises(BuildError, match="column 'momentum' has null values"):
        standard(make.portfolio_data(universe=frames.three_security_universe().assign(momentum=pd.Series([1, None, 3], dtype="Int64"))))
    with pytest.raises(BuildError, match="grouping column 'country' has null values"):
        standard(make.portfolio_data(universe=frames.three_security_universe().assign(country=pd.Series(["US", None, "GB"], dtype="string"))))


def test_a_held_name_outside_the_universe_cannot_be_built(make: Factories, frames: Frames) -> None:
    with pytest.raises(BuildError, match="held securities missing from universe \\['Z'\\]"):
        standard(make.portfolio_data(holdings=frames.holdings({"security_id": "Z"})))


def test_holdings_analytics_columns_do_not_reach_the_spec(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings({"security_id": "A"}).assign(lot_score=pd.Series([0.5], dtype="Float64"))
    spec = standard(make.portfolio_data(holdings=holdings))
    assert "lot_score" not in spec.columns, "the universe's own analytics column reaches the spec; the holdings' does not"


@pytest.mark.parametrize("bad", [[1.5], [True], [Decimal("NaN")], [Decimal("Infinity")], ["1"]])
def test_to_float64_rejects_anything_but_finite_decimals_and_ints(bad: list[object]) -> None:
    with pytest.raises(BuildError):
        to_float64(bad, "x")  # ty: ignore[invalid-argument-type]  # the wrong element type is the case under test


def test_to_float64_is_correctly_rounded_and_handles_empty() -> None:
    assert to_float64([Decimal("0.1"), 3], "x").tolist() == [0.1, 3.0]
    assert to_float64([], "x").shape == (0,)


def test_empty_holdings_give_zero_weights(make: Factories, frames: Frames) -> None:
    spec = standard(make.portfolio_data(holdings=frames.holdings().iloc[0:0]))
    np.testing.assert_array_equal(spec.w0, [0.0, 0.0, 0.0])
    assert spec.column("tax_per_dollar").tolist() == [0.0, 0.0, 0.0]


def test_two_builds_of_the_same_bundle_hash_equal(make: Factories) -> None:
    data = make.portfolio_data()
    assert standard(data).content_hash() == standard(data).content_hash()


def test_order_inputs_refuse_a_spec_that_names_securities_the_universe_lacks(make: Factories) -> None:
    data = make.portfolio_data()
    spec = make.spec(n=2)  # S0 and S1 are not in the universe
    with pytest.raises(BuildError, match=r"the spec names securities the universe does not carry \['S0', 'S1'\]"):
        order_inputs(data, spec)
