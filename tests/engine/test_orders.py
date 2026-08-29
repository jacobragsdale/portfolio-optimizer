"""Tier 1: solution → whole-share orders, nearest-share rounding, the buy clamp, and the drift bound."""

from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.results import OrderInputs, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.domain.schemas import ORDERS
from portfolio_optimizer.engine.build import BuildOutput, build_problem_spec
from portfolio_optimizer.engine.orders import rounding_drift, solution_to_orders
from tests.conftest import Factories, Frames, make_portfolio_data

HAND_OPTIMUM = np.array([0.375, 0.375, 0.25])


def solution_at(spec: ProblemSpec, w: np.ndarray) -> Solution:
    delta = w - spec.w0
    return Solution(
        w=w,
        buy=np.maximum(delta, 0.0),
        sell=np.maximum(-delta, 0.0),
        objective=0.0,
        status=SolveStatus.OPTIMAL,
        solver="X",
        solver_version="0",
        cvxpy_version="0",
        solve_time_s=0.0,
        iterations=1,
        spec_hash=spec.content_hash(),
    )


def built(make: Factories, **kwargs: object) -> BuildOutput:
    return build_problem_spec(make.portfolio_data(**kwargs))


def test_exact_deltas_become_exact_orders(make: Factories) -> None:
    output = built(make)
    orders = solution_to_orders(output.spec, solution_at(output.spec, HAND_OPTIMUM), output.order_inputs, run_id="r1")
    assert orders["security_id"].tolist() == ["A", "B", "C"]
    assert orders["side"].tolist() == ["SELL", "SELL", "BUY"]
    assert orders["quantity"].tolist() == [1250, 2500, 25000]
    assert orders["notional"].tolist() == [Decimal(125000), Decimal(125000), Decimal(250000)]
    assert orders["run_id"].tolist() == ["r1"] * 3
    assert orders["spec_hash"].iloc[0] == output.spec.content_hash()


def test_fractional_shares_round_to_the_nearest_share(make: Factories) -> None:
    output = built(make)
    w = output.spec.w0 + np.array([0.4 * 100 / 1e6, -0.6 * 50 / 1e6, 1.5 * 10 / 1e6])  # +0.4, -0.6, +1.5 shares
    orders = solution_to_orders(output.spec, solution_at(output.spec, w), output.order_inputs, run_id="r")
    assert orders["security_id"].tolist() == ["B", "C"]
    assert orders["side"].tolist() == ["SELL", "BUY"]
    assert orders["quantity"].tolist() == [1, 2]
    assert orders["unrounded_shares"].iloc[1] == pytest.approx(1.5)


def test_lots_round_down_to_a_multiple(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[2, "lot_size"] = 100
    output = built(make, universe=universe)
    w = output.spec.w0 + np.array([0.0, 0.0, 25_050 * 10 / 1e6])
    orders = solution_to_orders(output.spec, solution_at(output.spec, w), output.order_inputs, run_id="r")
    assert orders["quantity"].tolist() == [25000]


def test_trades_below_the_minimum_notional_are_dropped(make: Factories) -> None:
    output = built(make, style=make.style(min_trade_notional=Decimal(200_000)))
    orders = solution_to_orders(output.spec, solution_at(output.spec, HAND_OPTIMUM), output.order_inputs, run_id="r")
    assert orders["security_id"].tolist() == ["C"]


def test_sells_never_exceed_the_shares_held(make: Factories) -> None:
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
    assert str(orders["as_of"].dtype) == "datetime64[ns, UTC]"


def test_orders_are_deterministic(make: Factories) -> None:
    output = built(make)
    solution = solution_at(output.spec, HAND_OPTIMUM)
    assert_frame_equal(solution_to_orders(output.spec, solution, output.order_inputs, run_id="r"), solution_to_orders(output.spec, solution, output.order_inputs, run_id="r"))


def test_misaligned_inputs_are_rejected(make: Factories) -> None:
    output = built(make)
    inputs = OrderInputs(
        security_ids=("A", "B"),
        price=(Decimal(1), Decimal(1)),
        shares_held=(0, 0),
        lot_size=(1, 1),
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
    universe.loc[2, "lot_size"] = 100
    lots = built(make, universe=universe)
    w = lots.spec.w0 + np.array([0.0, 0.0, 25_050 * 10 / 1e6])
    lot_solution = solution_at(lots.spec, w)
    lot_report = rounding_drift(lots.spec, lot_solution, solution_to_orders(lots.spec, lot_solution, lots.order_inputs, run_id="r"), lots.order_inputs)
    assert 0 < lot_report.max_weight_error <= lot_report.tolerance
    assert lot_report.passed


def test_drift_counts_orders_dropped_by_the_dust_filter(make: Factories) -> None:
    output = built(make, style=make.style(min_trade_notional=Decimal(200_000)))
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
    universe.loc[2, "lot_size"] = 7
    output = build_problem_spec(make_portfolio_data(universe=universe))
    solution = solution_at(output.spec, w)
    orders = solution_to_orders(output.spec, solution, output.order_inputs, run_id="r")
    held = dict(zip(output.order_inputs.security_ids, output.order_inputs.shares_held, strict=True))
    lots = dict(zip(output.order_inputs.security_ids, output.order_inputs.lot_size, strict=True))
    for security, side, quantity in zip(orders["security_id"], orders["side"], orders["quantity"], strict=True):
        assert int(quantity) % lots[str(security)] == 0
        if str(side) == "SELL":
            assert int(quantity) <= held[str(security)]
    assert rounding_drift(output.spec, solution, orders, output.order_inputs).passed
