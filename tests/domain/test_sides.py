"""Tier 1: the side profiles — the split, the identity residuals, the tradable set, what a dependent receives, and the starts a side cannot trade out of."""

from decimal import Decimal
from typing import get_args

import numpy as np
import pytest

from portfolio_optimizer.cvx.sides import SIDES
from portfolio_optimizer.domain.results import Contribution
from portfolio_optimizer.domain.sides import BUY_ONLY, PROFILES, SELL_ONLY, Sides, profile_for
from tests.conftest import Factories, Frames


def test_every_profile_has_its_cvxpy_half_and_the_config_can_select_it() -> None:
    assert set(PROFILES) == set(SIDES) == set(get_args(Sides.__value__)) == {"buy", "sell"}
    assert profile_for("buy") is BUY_ONLY
    assert profile_for("sell") is SELL_ONLY
    with pytest.raises(ValueError, match="sides 'both' is not one the engine knows"):
        profile_for("both")


def test_every_profile_holds_the_specs_box(make: Factories) -> None:
    spec = make.spec(lb=np.array([0.1, 0.0, 0.0]), ub=np.array([0.5, 0.5, 0.5]))
    for profile in (BUY_ONLY, SELL_ONLY):
        residuals = dict(profile.identity_residuals(spec, make.solution(spec, w=np.array([0.05, 0.6, 0.35]))))
        np.testing.assert_allclose(residuals["lb"], [0.05, -0.6, -0.35], err_msg="S0 sits below its floor")
        np.testing.assert_allclose(residuals["ub"], [-0.45, 0.1, -0.15], err_msg="S1 sits above its cap")


# --- buy ---


def test_buy_only_split_is_the_clipped_delta_and_no_sell() -> None:
    buy, sell = BUY_ONLY.split(np.array([0.5, 0.3 - 1e-12, 0.3]), np.array([0.4, 0.3, 0.3]))
    np.testing.assert_allclose(buy, [0.1, 0.0, 0.0])
    assert sell.tolist() == [0.0, 0.0, 0.0], "solver noise below w0 is not a sell"


@pytest.mark.parametrize(
    ("w", "buy", "sell", "violated"),
    [
        (np.array([0.5, 0.3, 0.3]), np.array([0.1, 0.0, 0.0]), np.zeros(3), set()),
        (np.array([0.5, 0.2, 0.3]), np.array([0.1, 0.0, 0.0]), np.zeros(3), {"no_sells", "trade_balance"}),
        (np.array([0.5, 0.3, 0.3]), np.array([0.2, 0.0, 0.0]), np.zeros(3), {"trade_balance"}),
        (np.array([0.5, 0.3, 0.3]), np.array([0.1, 0.0, -0.01]), np.zeros(3), {"nonneg_buy", "trade_balance"}),
        (np.array([0.5, 0.3, 0.3]), np.array([0.1, 0.1, 0.0]), np.array([0.0, 0.1, 0.0]), {"sell_absent"}),
    ],
    ids=["buy", "a-sell", "buy-off-w", "negative-buy", "hidden-round-trip"],
)
def test_buy_only_identity_residuals_name_what_is_violated(make: Factories, w: np.ndarray, buy: np.ndarray, sell: np.ndarray, violated: set[str]) -> None:
    spec = make.spec(w0=np.array([0.4, 0.3, 0.3]))
    residuals = dict(BUY_ONLY.identity_residuals(spec, make.solution(spec, w=w, buy=buy, sell=sell)))
    assert set(residuals) == {"no_sells", "trade_balance", "nonneg_buy", "sell_absent", "lb", "ub"}
    assert {name for name, residual in residuals.items() if residual.max() > 1e-9} == violated


def test_buy_only_couples_through_buys_and_reports_only_buys(make: Factories, frames: Frames) -> None:
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), ub=np.array([0.5, 1.0, 1.0]))
    assert BUY_ONLY.tradable(spec).tolist() == [False, True, True], "S0 is capped at its holding, so it is not buyable"
    orders = frames.orders({"security_id": "S1", "side": "BUY", "quantity": 10, "notional": Decimal(1000)}, {"security_id": "S2", "side": "BUY", "quantity": 7, "notional": Decimal(700)})
    contribution = BUY_ONLY.contribution("P1", orders)
    assert (contribution.security_ids, contribution.traded_shares.tolist()) == (("S1", "S2"), [10.0, 7.0])
    state = BUY_ONLY.chain_state(spec, [Contribution("P0", ("S0", "S2"), np.array([5.0, 3.0]))], np.ones(3, dtype=np.bool_))
    assert state.traded_shares.tolist() == [0.0, 0.0, 3.0], "a predecessor's buy of a name this portfolio cannot buy is masked out"
    assert BUY_ONLY.order_sides == {"BUY"}
    assert BUY_ONLY.coupled(make.solution(spec, buy=np.array([0.0, 0.1, 0.2]))).tolist() == [0.0, 0.1, 0.2]


def test_buy_only_names_the_starts_it_cannot_trade_out_of(make: Factories) -> None:
    fine = make.spec(w0=np.array([0.3, 0.3, 0.3]), cash_lb=0.0, cash_ub=0.1)
    assert BUY_ONLY.infeasible_starts(fine) == []
    assert BUY_ONLY.infeasible_starts(make.spec(w0=np.array([0.3, 0.3, 0.3]), scalars={})) == [], "a spec without cash bounds has no cash start to be wrong"
    low_cash = make.spec(w0=np.array([0.3, 0.3, 0.3]), cash_lb=0.2, cash_ub=0.5)
    assert BUY_ONLY.infeasible_starts(low_cash) == ["the book starts with cash 0.100000 below cash_lb 0.200000, and a buy-only run can only lower cash"]
    over_cap = make.spec(w0=np.array([0.6, 0.3, 0.1]), ub=np.array([0.5, 0.3, 1.0]))
    assert BUY_ONLY.infeasible_starts(over_cap) == ["names whose cap is below their holding, which this side cannot trade out of: ['S0']"], "S1 sits exactly at its cap, which is allowed"


# --- sell ---


def test_sell_only_split_is_the_clipped_deficit_and_no_buy() -> None:
    buy, sell = SELL_ONLY.split(np.array([0.3, 0.3 + 1e-12, 0.3]), np.array([0.4, 0.3, 0.3]))
    np.testing.assert_allclose(sell, [0.1, 0.0, 0.0])
    assert buy.tolist() == [0.0, 0.0, 0.0], "solver noise above w0 is not a buy"


@pytest.mark.parametrize(
    ("w", "buy", "sell", "violated"),
    [
        (np.array([0.3, 0.3, 0.3]), np.zeros(3), np.array([0.1, 0.0, 0.0]), set()),
        (np.array([0.3, 0.4, 0.3]), np.zeros(3), np.array([0.1, 0.0, 0.0]), {"no_buys", "trade_balance"}),
        (np.array([0.3, 0.3, 0.3]), np.zeros(3), np.array([0.2, 0.0, 0.0]), {"trade_balance"}),
        (np.array([0.3, 0.3, 0.3]), np.zeros(3), np.array([0.1, 0.0, -0.01]), {"nonneg_sell", "trade_balance"}),
        (np.array([0.3, 0.3, 0.3]), np.array([0.0, 0.1, 0.0]), np.array([0.1, 0.1, 0.0]), {"buy_absent"}),
    ],
    ids=["sell", "a-buy", "sell-off-w", "negative-sell", "hidden-round-trip"],
)
def test_sell_only_identity_residuals_name_what_is_violated(make: Factories, w: np.ndarray, buy: np.ndarray, sell: np.ndarray, violated: set[str]) -> None:
    spec = make.spec(w0=np.array([0.4, 0.3, 0.3]))
    residuals = dict(SELL_ONLY.identity_residuals(spec, make.solution(spec, w=w, buy=buy, sell=sell)))
    assert set(residuals) == {"no_buys", "trade_balance", "nonneg_sell", "buy_absent", "lb", "ub"}
    assert {name for name, residual in residuals.items() if residual.max() > 1e-9} == violated


def test_sell_only_couples_through_sells_and_reports_only_sells(make: Factories, frames: Frames) -> None:
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), lb=np.array([0.5, 0.0, 0.0]))
    assert SELL_ONLY.tradable(spec).tolist() == [False, True, False], "S0 is floored at its holding and S2 is not held"
    orders = frames.orders({"security_id": "S0", "side": "SELL", "quantity": 10, "notional": Decimal(1000)}, {"security_id": "S1", "side": "SELL", "quantity": 7, "notional": Decimal(700)})
    contribution = SELL_ONLY.contribution("P1", orders)
    assert (contribution.security_ids, contribution.traded_shares.tolist()) == (("S0", "S1"), [10.0, 7.0])
    state = SELL_ONLY.chain_state(spec, [Contribution("P0", ("S0", "S1"), np.array([5.0, 3.0]))], np.ones(3, dtype=np.bool_))
    assert state.traded_shares.tolist() == [0.0, 3.0, 0.0], "a predecessor's sell of a name this portfolio cannot sell is masked out"
    assert SELL_ONLY.order_sides == {"SELL"}
    assert SELL_ONLY.coupled(make.solution(spec, sell=np.array([0.0, 0.1, 0.2]))).tolist() == [0.0, 0.1, 0.2]


def test_sell_only_names_the_starts_it_cannot_trade_out_of(make: Factories) -> None:
    fine = make.spec(w0=np.array([0.3, 0.3, 0.3]), cash_lb=0.0, cash_ub=0.1)
    assert SELL_ONLY.infeasible_starts(fine) == []
    high_cash = make.spec(w0=np.array([0.3, 0.3, 0.3]), cash_lb=0.0, cash_ub=0.05)
    assert SELL_ONLY.infeasible_starts(high_cash) == ["the book starts with cash 0.100000 above cash_ub 0.050000, and a sell-only run can only raise cash"]
    under_floor = make.spec(w0=np.array([0.1, 0.3, 0.6]), lb=np.array([0.2, 0.3, 0.0]))
    assert SELL_ONLY.infeasible_starts(under_floor) == ["names whose floor is above their holding, which this side cannot trade out of: ['S0']"], "S1 sits exactly on its floor, which is allowed"


def test_chain_state_masks_to_the_consume_set_as_well_as_the_tradable_set(make: Factories) -> None:
    """A predecessor's trade in a name this portfolio *could* buy but whose chain readers never consult is zeroed — which is what keeps the chain hash identical however many predecessors were folded."""
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), ub=np.array([0.5, 1.0, 1.0]))
    contributions = [Contribution("P0", ("S1", "S2"), np.array([5.0, 3.0]))]
    narrowed = BUY_ONLY.chain_state(spec, contributions, np.array([False, False, True]))
    assert narrowed.traded_shares.tolist() == [0.0, 0.0, 3.0], "S1 is buyable but outside the consume set"
    everything = BUY_ONLY.chain_state(spec, contributions, np.ones(3, dtype=np.bool_))
    assert everything.traded_shares.tolist() == [0.0, 5.0, 3.0]
