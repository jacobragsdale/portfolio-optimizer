"""Tier 1: solution → whole-share orders, nearest-share rounding, the buy clamp, and the drift bound."""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.constraints import parse_constraints
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.order_flow import INFLOW
from portfolio_optimizer.domain.results import ChainState, OrderInputs, ProblemSpec, Solution
from portfolio_optimizer.domain.schemas import ORDERS
from portfolio_optimizer.engine.build import order_inputs, standard
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.orders import executed_solution, rounding_drift, solution_to_orders
from tests.conftest import CASH_FLOOR, Factories, Frames, Row, constraint_frame, make_portfolio_data, make_solution, typed_row

HAND_OPTIMUM = np.array([0.35, 0.4, 0.25])
"""A target for the default bundle — A 5,000 @100 and B 10,000 @50 on 1,000,000 — whose deltas are whole shares: sell 1,500 A, sell 2,000 B, buy 25,000 C."""
EXACT_ORDERS = [{"security_id": "A", "side": "SELL", "quantity": 1500}, {"security_id": "B", "side": "SELL", "quantity": 2000}, {"security_id": "C", "side": "BUY", "quantity": 25000}]


def solution_at(spec: ProblemSpec, w: np.ndarray) -> Solution:
    """A solution at ``w`` with the minimal buy/sell split of the move from ``w0``."""
    delta = w - spec.w0
    return make_solution(spec, w=w, buy=np.maximum(delta, 0.0), sell=np.maximum(-delta, 0.0))


@dataclass(frozen=True, slots=True)
class Built:
    """The spec and the exact order inputs the standard build derives for a bundle."""

    spec: ProblemSpec
    order_inputs: OrderInputs


def built(make: Factories, **kwargs: object) -> Built:
    data = make.portfolio_data(**kwargs)
    spec = standard(data)
    return Built(spec, order_inputs(data, spec))


def test_exact_deltas_become_exact_orders(make: Factories) -> None:
    output = built(make)
    orders = solution_to_orders(output.spec, solution_at(output.spec, HAND_OPTIMUM), output.order_inputs, run_id="r1")
    assert orders[["security_id", "side", "quantity"]].to_dict("records") == EXACT_ORDERS
    assert orders["notional"].tolist() == [Decimal(150000), Decimal(100000), Decimal(250000)]
    assert orders["run_id"].tolist() == ["r1"] * 3
    assert orders["spec_hash"].iloc[0] == output.spec.content_hash()


def test_fractional_shares_round_to_the_nearest_share(make: Factories) -> None:
    output = built(make)
    w = output.spec.w0 + np.array([0.4 * 100 / 1e6, -0.6 * 50 / 1e6, 1.5 * 10 / 1e6])  # +0.4, -0.6, +1.5 shares
    orders = solution_to_orders(output.spec, solution_at(output.spec, w), output.order_inputs, run_id="r")
    assert orders["security_id"].tolist() == ["B", "C"]
    assert orders["side"].tolist() == ["SELL", "BUY"]
    assert orders["quantity"].tolist() == [1, 2]
    assert orders["unrounded_quantity"].iloc[1] == pytest.approx(1.5)


def test_lots_round_down_to_a_multiple(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[2, "increment"] = 100
    output = built(make, universe=universe)
    w = output.spec.w0 + np.array([0.0, 0.0, 25_050 * 10 / 1e6])
    orders = solution_to_orders(output.spec, solution_at(output.spec, w), output.order_inputs, run_id="r")
    assert orders["quantity"].tolist() == [25000]


def test_trades_below_the_minimum_notional_are_dropped(make: Factories) -> None:
    output = built(make, details=make.details(min_trade_notional=Decimal(200_000)))
    orders = solution_to_orders(output.spec, solution_at(output.spec, HAND_OPTIMUM), output.order_inputs, run_id="r")
    assert orders["security_id"].tolist() == ["C"]


def test_sells_never_exceed_the_quantity_held(make: Factories) -> None:
    output = built(make)
    w = np.array([-0.2, 0.7, 0.5])
    orders = solution_to_orders(output.spec, solution_at(output.spec, w), output.order_inputs, run_id="r")
    sell_a = orders[orders["security_id"] == "A"]
    assert sell_a["side"].tolist() == ["SELL"]
    assert sell_a["quantity"].tolist() == [5000]


def test_no_change_yields_an_empty_but_valid_frame(make: Factories) -> None:
    output = built(make)
    orders = solution_to_orders(output.spec, solution_at(output.spec, output.spec.w0), output.order_inputs, run_id="r")
    assert len(orders) == 0
    validate_frame(orders, ORDERS)
    assert str(orders["as_of_date"].dtype) == "datetime64[ns, UTC]"


def test_orders_are_deterministic(make: Factories) -> None:
    output = built(make)
    solution = solution_at(output.spec, HAND_OPTIMUM)
    assert_frame_equal(solution_to_orders(output.spec, solution, output.order_inputs, run_id="r"), solution_to_orders(output.spec, solution, output.order_inputs, run_id="r"))


def test_misaligned_inputs_are_rejected(make: Factories) -> None:
    output = built(make)
    inputs = OrderInputs(
        security_ids=("A", "B"),
        price=(Decimal(1), Decimal(1)),
        accrued_interest=(Decimal(0), Decimal(0)),
        quantity_held=(0, 0),
        increment=(1, 1),
        min_denomination=(1, 1),
        w0=(Decimal(0), Decimal(0)),
        ub=(Decimal(1), Decimal(1)),
        nav=Decimal(1),
        min_trade_notional=Decimal(0),
    )
    with pytest.raises(ValueError, match="not aligned"):
        solution_to_orders(output.spec, solution_at(output.spec, HAND_OPTIMUM), inputs, run_id="r")


def test_a_buy_never_exceeds_the_room_under_the_upper_bound(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[0, "restricted"] = True  # A is frozen at its current weight: not buyable
    output = built(make, universe=universe)
    w = output.spec.w0 + np.array([3e-7, 0.0, 0.0])  # solver noise the verifier tolerates, but three whole shares of A
    solution = solution_at(output.spec, w)
    orders = solution_to_orders(output.spec, solution, output.order_inputs, run_id="r")
    assert orders.empty, "the clamp keeps the buyable set structural"
    assert rounding_drift(output.spec, solution, orders, output.order_inputs, violation_tol=1e-6).passed


def test_drift_is_zero_for_exact_orders_and_bounded_for_lots(make: Factories, frames: Frames) -> None:
    exact = built(make)
    solution = solution_at(exact.spec, HAND_OPTIMUM)
    orders = solution_to_orders(exact.spec, solution, exact.order_inputs, run_id="r")
    report = rounding_drift(exact.spec, solution, orders, exact.order_inputs)
    assert report.max_weight_error == 0.0
    assert report.passed
    universe = frames.three_security_universe()
    universe.loc[2, "increment"] = 100
    lots = built(make, universe=universe)
    w = lots.spec.w0 + np.array([0.0, 0.0, 25_050 * 10 / 1e6])
    lot_solution = solution_at(lots.spec, w)
    lot_report = rounding_drift(lots.spec, lot_solution, solution_to_orders(lots.spec, lot_solution, lots.order_inputs, run_id="r"), lots.order_inputs)
    assert 0 < lot_report.max_weight_error <= lot_report.tolerance
    assert lot_report.passed


def test_orders_that_round_into_a_breach_fail_verification_unless_the_row_allows_the_slack(make: Factories) -> None:
    """Seven dollars of cash buys 0.7 of a share of C; the nearest share costs ten, so the executed book is three dollars overdrawn — a breach of the cash floor the solved weights never showed."""
    output = built(make, details=make.details(nav=Decimal(1_000_007), cash=Decimal(7)))
    solution = solution_at(output.spec, output.spec.w0 + np.array([0.0, 0.0, 7 / 1_000_007]))
    orders = solution_to_orders(output.spec, solution, output.order_inputs, run_id="r")
    assert orders["quantity"].tolist() == [1]
    executed = executed_solution(output.spec, solution, orders, INFLOW)
    np.testing.assert_allclose(executed.w.sum() - 1.0, 3 / 1_000_007)
    assert executed.objective is None, "an executed book has no solver objective to agree with"
    chain = ChainState.empty(output.spec.security_ids)

    def report_under(row: Row) -> tuple[str, ...]:
        rows = parse_constraints(constraint_frame([row]))
        assert rows is not None
        return verify(output.spec, executed, chain, (), rows.typed, profile=INFLOW).violated

    assert report_under(CASH_FLOOR) == ("cash_floor/cash_limit",)
    assert report_under(typed_row("cash_limit", "cash_floor", direction=">=", bounds={"scalar": "cash_lb"}, tolerance="0.00001")) == ()


def test_bond_quantities_honour_the_increment_the_minimum_piece_and_accrued_interest(make: Factories, frames: Frames) -> None:
    """A bond at 0.98 clean with 0.01 accrued, traded in 1,000s with a 5,000 minimum piece, on a book of 100,000 that holds 6,000 of it."""
    universe = frames.universe({"security_id": "A", "price": Decimal("0.98"), "accrued_interest": Decimal("0.01"), "increment": 1000, "min_denomination": 5000})
    output = built(make, universe=universe, holdings=frames.holdings({"quantity": 6000, "avg_cost": Decimal("0.95")}), details=make.details(nav=Decimal(100_000), cash=Decimal(94_060)))
    assert output.spec.price.tolist() == [0.99] and output.spec.w0.tolist() == [pytest.approx(0.0594)], "the spec values a unit dirty"

    def orders_for(units: float) -> pd.DataFrame:
        return solution_to_orders(output.spec, solution_at(output.spec, output.spec.w0 + np.array([units * 0.99 / 100_000])), output.order_inputs, run_id="r")

    buy = orders_for(2400)
    assert buy[["side", "quantity", "reference_price", "accrued_interest", "notional"]].to_dict("records") == [
        {"side": "BUY", "quantity": 2000, "reference_price": Decimal("0.98"), "accrued_interest": Decimal("0.01"), "notional": Decimal("1980.00")}
    ], "2,400 rounds down to the increment, and the notional is at the dirty price"
    assert orders_for(-3000)["quantity"].tolist() == [1000], "selling 3,000 would leave 3,000, below the minimum piece; the sell shrinks to leave exactly 5,000"
    assert orders_for(-6000)[["side", "quantity"]].to_dict("records") == [{"side": "SELL", "quantity": 6000}], "a full liquidation goes out whole"
    empty = built(make, universe=universe, holdings=frames.holdings({"quantity": 0, "avg_cost": Decimal("0.95")}), details=make.details(nav=Decimal(100_000), cash=Decimal(100_000)))
    assert solution_to_orders(empty.spec, solution_at(empty.spec, np.array([4000 * 0.99 / 100_000])), empty.order_inputs, run_id="r").empty, (
        "4,000 from nothing cannot reach the minimum piece, so no order"
    )


def test_drift_counts_orders_dropped_by_the_dust_filter(make: Factories) -> None:
    output = built(make, details=make.details(min_trade_notional=Decimal(200_000)))
    solution = solution_at(output.spec, HAND_OPTIMUM)
    report = rounding_drift(output.spec, solution, solution_to_orders(output.spec, solution, output.order_inputs, run_id="r"), output.order_inputs)
    assert report.dropped_orders == 2
    assert report.passed


@given(weights=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3))
@settings(deadline=None, max_examples=40)
def test_orders_respect_holdings_lots_and_the_drift_bound(weights: list[float]) -> None:
    total = sum(weights) or 1.0
    w = np.array([weight / total for weight in weights])
    universe = make_portfolio_data().universe
    universe.loc[2, "increment"] = 7
    data = make_portfolio_data(universe=universe)
    output = Built(standard(data), order_inputs(data, standard(data)))
    solution = solution_at(output.spec, w)
    orders = solution_to_orders(output.spec, solution, output.order_inputs, run_id="r")
    held = dict(zip(output.order_inputs.security_ids, output.order_inputs.quantity_held, strict=True))
    lots = dict(zip(output.order_inputs.security_ids, output.order_inputs.increment, strict=True))
    for security, side, quantity in zip(orders["security_id"], orders["side"], orders["quantity"], strict=True):
        assert int(quantity) % lots[str(security)] == 0
        if str(side) == "SELL":
            assert int(quantity) <= held[str(security)]
    assert rounding_drift(output.spec, solution, orders, output.order_inputs).passed
