"""Tier 1: the shipped solve-order step — exact Decimal, the furthest-from-target portfolio first, ties at equal distance."""

from decimal import Decimal

from portfolio_optimizer.solve_order import furthest_from_target_first
from tests.conftest import Factories, Frames


def test_a_portfolio_at_its_target_has_key_zero_and_one_far_from_it_sorts_first(make: Factories, frames: Frames) -> None:
    at_target = make.portfolio_data(targets=frames.targets({"security_id": "A", "weight": Decimal("0.5")}, {"security_id": "B", "weight": Decimal("0.5")}, {"security_id": "C", "weight": Decimal(0)}))
    all_in_c = make.portfolio_data(targets=frames.targets({"security_id": "A", "weight": Decimal(0)}, {"security_id": "B", "weight": Decimal(0)}, {"security_id": "C", "weight": Decimal(1)}))
    assert furthest_from_target_first(at_target) == Decimal(0)
    assert furthest_from_target_first(all_in_c) == Decimal(-1)
    assert furthest_from_target_first(all_in_c) < furthest_from_target_first(at_target), "lower solves first"


def test_the_key_is_exact_decimal_arithmetic(make: Factories) -> None:
    data = make.portfolio_data()  # A and B held at half the NAV each against equal-weight targets
    key = furthest_from_target_first(data)
    assert isinstance(key, Decimal)
    weights = {str(security): weight for security, weight in zip(data.targets["security_id"], data.targets["weight"], strict=True)}
    assert key == -(abs(Decimal("0.5") - weights["A"]) + abs(Decimal("0.5") - weights["B"]) + weights["C"]) / 2
