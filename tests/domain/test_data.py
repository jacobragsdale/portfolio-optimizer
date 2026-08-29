"""Tier 1/3: the per-portfolio bundle's cross-frame invariants and the typed style constraints."""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from portfolio_optimizer.domain.data import PortfolioDataError, details_from_frame, style_constraints_from_mapping
from portfolio_optimizer.domain.types import PortfolioId
from tests.conftest import Factories, Frames


def test_canonical_bundle_is_valid(make: Factories) -> None:
    data = make.portfolio_data()
    assert data.portfolio_id == "P1"
    assert data.applied_rules == ()


def test_holding_absent_from_universe_is_rejected(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="held securities missing from universe \\['Z'\\]"):
        make.portfolio_data(holdings=frames.holdings({"security_id": "Z"}))


def test_holdings_of_another_portfolio_are_rejected(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="other portfolios \\['P2'\\]"):
        make.portfolio_data(holdings=frames.holdings({"portfolio_id": "P2"}))


def test_targets_for_another_benchmark_are_rejected(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="other benchmarks \\['B9'\\]"):
        make.portfolio_data(targets=frames.targets({"benchmark_id": "B9"}))


def test_sector_bound_for_unknown_sector_is_rejected(make: Factories) -> None:
    style = make.style(sector_bounds={"ENERGY": (Decimal(0), Decimal("0.5"))})
    with pytest.raises(PortfolioDataError, match="sectors absent from universe \\['ENERGY'\\]"):
        make.portfolio_data(style=style)


def test_covariance_must_cover_the_universe(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="does not cover universe securities \\['B', 'C'\\]"):
        make.portfolio_data(covariance=frames.covariance())


def test_frame_schema_failures_are_reported_with_their_frame_name(make: Factories, frames: Frames) -> None:
    with pytest.raises(PortfolioDataError, match="universe: column 'price'"):
        make.portfolio_data(universe=frames.universe({"price": Decimal(0)}))


def test_naive_as_of_is_rejected(make: Factories) -> None:
    with pytest.raises(PortfolioDataError, match="timezone-aware UTC"):
        make.portfolio_data(as_of=datetime(2026, 8, 28))  # noqa: DTZ001  # the naive datetime is the case under test


def test_with_frames_revalidates_and_records_the_rule(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data()
    updated = data.with_frames(rule="drop_b", holdings=data.holdings[data.holdings["security_id"] != "B"])
    assert updated.applied_rules == ("drop_b",)
    assert list(updated.holdings["security_id"]) == ["A"]
    assert list(data.holdings["security_id"]) == ["A", "B"]
    with pytest.raises(PortfolioDataError):
        data.with_frames(rule="bad", holdings=frames.holdings({"security_id": "Z"}))


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
def test_style_constraints_reject_values_just_past_their_limits(make: Factories, field: str, value: Decimal) -> None:
    with pytest.raises(ValidationError):
        make.style(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("max_weight", Decimal(1)), ("max_turnover", Decimal(2)), ("max_adv_participation", Decimal(0)), ("min_trade_notional", Decimal(0))])
def test_style_constraints_accept_values_at_their_limits(make: Factories, field: str, value: Decimal) -> None:
    assert getattr(make.style(**{field: value}), field) == value


def test_style_constraints_reject_unordered_bounds(make: Factories) -> None:
    with pytest.raises(ValidationError, match="cash_bounds"):
        make.style(cash_bounds=(Decimal("0.5"), Decimal("0.1")))
    with pytest.raises(ValidationError, match="sector_bounds"):
        make.style(sector_bounds={"TECH": (Decimal("0.5"), Decimal("0.1"))})


def test_style_constraints_from_mapping_reads_money_as_strings_and_rejects_unknown_keys() -> None:
    raw: dict[str, object] = {
        "max_weight": "0.05",
        "max_turnover": "0.2",
        "min_trade_notional": "100",
        "cash_bounds": ["0.01", "0.03"],
        "max_adv_participation": "0.1",
        "sector_bounds": {"TECH": ["0", "0.4"]},
    }
    style = style_constraints_from_mapping(raw)
    assert style.max_weight == Decimal("0.05")
    assert style.sector_bounds == {"TECH": (Decimal(0), Decimal("0.4"))}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        style_constraints_from_mapping(raw | {"max_leverage": "2"})
