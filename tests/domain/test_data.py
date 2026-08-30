"""Tier 1/3: the per-portfolio bundle's cross-frame invariants and the typed style constraints."""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import ValidationError

from portfolio_optimizer.domain.data import PortfolioData, PortfolioDataError, details_from_frame
from portfolio_optimizer.domain.types import PortfolioId
from tests.conftest import AS_OF, Factories, Frames


def test_canonical_bundle_is_valid(make: Factories) -> None:
    data = make.portfolio_data()
    assert data.portfolio_id == "P1"
    assert data.applied_rules == ()


def test_a_held_name_need_not_be_buyable(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data(holdings=frames.holdings({"security_id": "Z"}))
    assert data.holdings["security_id"].tolist() == ["Z"]


def test_holdings_of_another_portfolio_are_rejected(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="other portfolios \\['P2'\\]"):
        make.portfolio_data(holdings=frames.holdings({"portfolio_id": "P2"}))


def test_sector_bound_for_unknown_sector_is_rejected(make: Factories, frames: Frames) -> None:
    bounds = frames.sector_bounds({"sector": "ENERGY", "lower": Decimal(0), "upper": Decimal("0.5")})
    with pytest.raises(PortfolioDataError, match="sectors absent from universe \\['ENERGY'\\]"):
        make.portfolio_data(sector_bounds=bounds)


def test_sector_bounds_for_another_portfolio_are_rejected(make: Factories, frames: Frames) -> None:
    bounds = frames.sector_bounds({"portfolio_id": "P9", "sector": "TECH"})
    with pytest.raises(PortfolioDataError, match="sector_bounds contain other portfolios \\['P9'\\]"):
        make.portfolio_data(sector_bounds=bounds)


def test_unordered_sector_bounds_are_rejected(make: Factories, frames: Frames) -> None:
    bounds = frames.sector_bounds({"sector": "TECH", "lower": Decimal("0.5"), "upper": Decimal("0.1")})
    with pytest.raises(PortfolioDataError, match="lower exceeds upper"):
        make.portfolio_data(sector_bounds=bounds)


def test_shared_analytics_column_must_agree_on_dtype_between_holdings_and_universe(make: Factories, frames: Frames) -> None:
    holdings = frames.holdings().assign(score=pd.Series([1.0], dtype="Float64"))
    universe = frames.three_security_universe().assign(score=pd.Series([1.0, 2.0, 3.0], dtype="float64"))
    with pytest.raises(PortfolioDataError, match="disagree on column 'score': holdings has dtype 'Float64', universe has 'float64'"):
        make.portfolio_data(holdings=holdings, universe=universe)
    agreed = make.portfolio_data(holdings=holdings, universe=universe.astype({"score": "Float64"}))
    assert agreed.optimizer_frame()["score"].tolist() == [1.0, 1.0, 2.0, 3.0]


def test_extras_are_carried_and_must_belong_to_this_portfolio(make: Factories) -> None:
    mine = pd.DataFrame({"portfolio_id": pd.Series(["P1"], dtype="string"), "note": pd.Series(["x"], dtype="string")})
    shared = pd.DataFrame({"security_id": pd.Series(["A"], dtype="string"), "beta": pd.Series([1.1], dtype="Float64")})
    data = make.portfolio_data(extras={"notes": mine, "betas": shared})
    assert set(data.extras) == {"notes", "betas"}
    with pytest.raises(PortfolioDataError, match="extras\\['notes'\\] contain other portfolios \\['P2'\\]"):
        make.portfolio_data(extras={"notes": mine.assign(portfolio_id=pd.Series(["P2"], dtype="string"))})
    with pytest.raises(PortfolioDataError, match="'holdings' is an engine-known dataset name"):
        make.portfolio_data(extras={"holdings": shared})


def test_optimizer_frame_stacks_holdings_then_universe_with_typed_nulls(make: Factories) -> None:
    frame = make.portfolio_data().optimizer_frame()
    assert frame["source"].tolist() == ["holdings", "holdings", "universe", "universe", "universe"]
    assert frame["security_id"].tolist() == ["A", "B", "A", "B", "C"]
    assert list(frame.columns[:3]) == ["source", "portfolio_id", "security_id"]
    assert str(frame["quantity"].dtype) == "Int64" and frame["quantity"].isna().tolist() == [False, False, True, True, True]
    assert str(frame["restricted"].dtype) == "boolean" and frame["restricted"].isna().tolist() == [True, True, False, False, False]
    assert frame["price"].tolist()[:2] == [None, None] and frame["price"].tolist()[2] == Decimal(100)
    assert "source" not in make.portfolio_data().optimizer_frame(source_column=None).columns


def test_frame_schema_failures_are_reported_with_their_frame_name(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="universe: column 'price'"):
        make.portfolio_data(universe=frames.universe({"price": Decimal(0)}))


def test_naive_as_of_date_is_rejected(make: Factories) -> None:
    with pytest.raises(PortfolioDataError, match="timezone-aware UTC"):
        make.portfolio_data(as_of_date=datetime(2026, 8, 28))  # noqa: DTZ001  # the naive datetime is the case under test


def test_with_changes_revalidates_and_records_the_rule(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data()
    updated = data.with_changes(holdings=data.holdings[data.holdings["security_id"] != "B"]).with_rule_applied("drop_b")
    assert updated.applied_rules == ("drop_b",)
    assert list(updated.holdings["security_id"]) == ["A"]
    assert list(data.holdings["security_id"]) == ["A", "B"]
    with pytest.raises(PortfolioDataError):
        data.with_changes(holdings=frames.holdings({"portfolio_id": "P2"}))
    assert data.with_changes(extras={"tags": frames.universe()[["security_id"]]}).extras.keys() == {"tags"}


def test_prevalidated_frames_are_trusted_until_a_rule_replaces_them(make: Factories, frames: Frames) -> None:
    invalid = frames.universe({"price": Decimal(0)})  # fails the universe schema; the bundle would normally refuse it
    trusted = PortfolioData(
        details=make.details(),
        holdings=frames.holdings(),
        universe=invalid,
        sector_bounds=frames.sector_bounds(),
        constraints=frames.constraints(),
        as_of_date=AS_OF,
        prevalidated=frozenset({"universe"}),
    )
    assert trusted.universe is invalid
    assert trusted.with_changes(details=make.details()).prevalidated == frozenset({"universe"})  # untouched frames keep their standing
    with pytest.raises(PortfolioDataError, match="universe: column 'price'"):
        trusted.with_changes(universe=invalid)  # a replaced frame is validated even when it is the same object
    with pytest.raises(PortfolioDataError, match="prevalidated names unknown frames \\['details'\\]"):
        make.portfolio_data(prevalidated=frozenset({"details"}))


def test_details_from_frame_types_the_matching_row(frames: Frames) -> None:
    frame = frames.details({"portfolio_id": "P1"}, {"portfolio_id": "P2", "nav": Decimal(5)})
    details = details_from_frame(frame, PortfolioId("P2"))
    assert details.nav == Decimal(5)
    assert details.state == "NY"


def test_details_from_frame_requires_exactly_one_row(frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="expected exactly one row, found 0"):
        details_from_frame(frames.details(), PortfolioId("P9"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_weight", Decimal(0)), ("max_weight", Decimal("1.0001")), ("max_turnover", Decimal("2.01")), ("max_adv_participation", Decimal("-0.01")), ("min_trade_notional", Decimal(-1))],
)
def test_style_limits_reject_values_just_past_their_limits(make: Factories, field: str, value: Decimal) -> None:
    with pytest.raises(ValidationError):
        make.details(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("max_weight", Decimal(1)), ("max_turnover", Decimal(2)), ("max_adv_participation", Decimal(0)), ("min_trade_notional", Decimal(0))])
def test_style_limits_accept_values_at_their_limits(make: Factories, field: str, value: Decimal) -> None:
    assert getattr(make.details(**{field: value}), field) == value


def test_details_reject_unordered_cash_bounds(make: Factories) -> None:
    with pytest.raises(ValidationError, match="cash_lb must not exceed cash_ub"):
        make.details(cash_lb=Decimal("0.5"), cash_ub=Decimal("0.1"))


def test_details_from_frame_reads_money_exactly_and_rejects_unknown_columns(frames: Frames) -> None:
    details = details_from_frame(frames.details({"max_weight": Decimal("0.05"), "cash_lb": Decimal("0.01"), "cash_ub": Decimal("0.03")}), PortfolioId("P1"))
    assert details.max_weight == Decimal("0.05")
    assert (details.cash_lb, details.cash_ub) == (Decimal("0.01"), Decimal("0.03"))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        details_from_frame(frames.details().assign(max_leverage=pd.Series([Decimal(2)], dtype="object")), PortfolioId("P1"))
