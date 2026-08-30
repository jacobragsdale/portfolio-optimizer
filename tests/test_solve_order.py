"""Tier 1: the shipped solve-order step — exact Decimal, the least invested portfolio first, ties at equal investment."""

from decimal import Decimal

from portfolio_optimizer.solve_order import most_uninvested_first
from tests.conftest import Factories, Frames


def test_a_fully_invested_portfolio_has_key_zero_and_one_holding_nothing_sorts_first(make: Factories, frames: Frames) -> None:
    invested = make.portfolio_data()  # A 5000 at 100 and B 10000 at 50 against a NAV of 1,000,000
    in_cash = make.portfolio_data(holdings=frames.holdings().iloc[0:0])
    assert most_uninvested_first(invested) == Decimal(0)
    assert most_uninvested_first(in_cash) == Decimal(-1)
    assert most_uninvested_first(in_cash) < most_uninvested_first(invested), "lower solves first"


def test_the_key_is_exact_decimal_arithmetic(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data(holdings=frames.holdings({"security_id": "C", "quantity": 33333}))  # 333,330 of a 1,000,000 NAV
    key = most_uninvested_first(data)
    assert isinstance(key, Decimal)
    assert key == Decimal(333330) / Decimal(1000000) - 1, "no float rounds through the key the schedule sorts on"
