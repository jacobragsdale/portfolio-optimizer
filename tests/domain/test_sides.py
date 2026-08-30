"""Tier 1: the side profiles — the split, the identity residuals, the tradable set, what a dependent receives, and the starts a side cannot trade out of."""

from decimal import Decimal
from typing import get_args

import numpy as np
import pytest

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.cvx.adapter import SideUnavailableError
from portfolio_optimizer.cvx.sides import SIDES, decision_variables
from portfolio_optimizer.domain.results import ChainState, Contribution, Solution, SolveStatus
from portfolio_optimizer.domain.sides import BUY_ONLY, PROFILES, TWO_SIDED, profile_for
from tests.conftest import Factories, Frames


def _solution(w: np.ndarray, buy: np.ndarray, sell: np.ndarray) -> Solution:
    return Solution(w=w, buy=buy, sell=sell, objective=0.0, status=SolveStatus.OPTIMAL, solver="X", solver_version="0", cvxpy_version="0", solve_time_s=0.0, iterations=1, spec_hash="h")


def test_every_profile_has_its_cvxpy_half_and_the_config_can_select_it() -> None:
    assert set(PROFILES) == set(SIDES) == set(get_args(RunConfig.model_fields["sides"].annotation)) == {"both", "buy"}
    assert profile_for("both") is TWO_SIDED
    assert profile_for("buy") is BUY_ONLY
    with pytest.raises(ValueError, match="sides 'short' is not one the engine knows"):
        profile_for("short")


# --- both ---


def test_two_sided_split_is_the_minimal_one() -> None:
    buy, sell = TWO_SIDED.split(np.array([0.5, 0.2, 0.3]), np.array([0.4, 0.3, 0.3]))
    np.testing.assert_allclose(buy, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(sell, [0.0, 0.1, 0.0])


@pytest.mark.parametrize(
    ("buy", "sell", "violated"),
    [
        (np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), set()),
        (np.array([0.2, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), {"trade_balance"}),
        (np.array([0.1, 0.0, -0.01]), np.array([0.0, 0.1, -0.01]), {"nonneg_buy", "nonneg_sell"}),
        (np.array([0.1, 0.0, 0.5]), np.array([0.0, 0.1, 0.5]), {"sell_le_w0", "complementarity"}),
    ],
    ids=["minimal", "balance", "negative", "round-trip-beyond-holding"],
)
def test_two_sided_identity_residuals_name_what_is_violated(make: Factories, buy: np.ndarray, sell: np.ndarray, violated: set[str]) -> None:
    spec = make.spec(w0=np.array([0.4, 0.3, 0.3]))
    solution = _solution(np.array([0.5, 0.2, 0.3]), buy, sell)
    residuals = dict(TWO_SIDED.identity_residuals(spec, solution))
    assert set(residuals) == {"trade_balance", "nonneg_buy", "nonneg_sell", "sell_le_w0", "complementarity"}
    assert {name for name, residual in residuals.items() if residual.max() > 1e-9} == violated


def test_two_sided_couples_through_buys_only(make: Factories, frames: Frames) -> None:
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), ub=np.array([0.5, 1.0, 1.0]))
    assert TWO_SIDED.tradable(spec).tolist() == [False, True, True], "S0 is capped at its holding, so it is not buyable"
    orders = frames.orders({"security_id": "S1", "side": "SELL", "quantity": 10, "notional": Decimal(1000)}, {"security_id": "S2", "side": "BUY", "quantity": 7, "notional": Decimal(700)})
    contribution = TWO_SIDED.contribution("P1", orders)
    assert (contribution.security_ids, contribution.traded_shares.tolist()) == (("S2",), [7.0])
    state = TWO_SIDED.chain_state(spec, [Contribution("P0", ("S0", "S2"), np.array([5.0, 3.0]))])
    assert isinstance(state, ChainState) and state.traded_shares.tolist() == [0.0, 0.0, 3.0], "a predecessor's buy of a name this portfolio cannot buy is masked out"
    assert TWO_SIDED.order_sides == {"BUY", "SELL"}
    assert TWO_SIDED.coupled(_solution(spec.w0, np.array([0.0, 0.1, 0.2]), np.array([0.3, 0.0, 0.0]))).tolist() == [0.0, 0.1, 0.2]
    assert TWO_SIDED.infeasible_starts(make.spec(w0=np.array([0.6, 0.3, 0.3]), ub=np.array([0.5, 1.0, 1.0]))) == [], "a two-sided run can sell its way back inside a cap"


def test_two_sided_variables_are_three_vectors_and_couple_through_buy() -> None:
    x = decision_variables("both", np.array([0.4, 0.3, 0.3]))
    assert (x.sides, x.n) == ("both", 3)
    assert x.buy is not x.w and x.sell is not x.w and x.buy.is_affine() and x.sell.is_affine()
    assert x.trade.shape == (3,) and x.coupled is x.buy


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
    residuals = dict(BUY_ONLY.identity_residuals(spec, _solution(w, buy, sell)))
    assert set(residuals) == {"no_sells", "trade_balance", "nonneg_buy", "sell_absent"}
    assert {name for name, residual in residuals.items() if residual.max() > 1e-9} == violated


def test_buy_only_couples_through_buys_and_reports_only_buys(make: Factories, frames: Frames) -> None:
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), ub=np.array([0.5, 1.0, 1.0]))
    assert BUY_ONLY.tradable(spec).tolist() == [False, True, True]
    orders = frames.orders({"security_id": "S1", "side": "BUY", "quantity": 10, "notional": Decimal(1000)}, {"security_id": "S2", "side": "BUY", "quantity": 7, "notional": Decimal(700)})
    contribution = BUY_ONLY.contribution("P1", orders)
    assert (contribution.security_ids, contribution.traded_shares.tolist()) == (("S1", "S2"), [10.0, 7.0])
    state = BUY_ONLY.chain_state(spec, [Contribution("P0", ("S0", "S2"), np.array([5.0, 3.0]))])
    assert state.traded_shares.tolist() == [0.0, 0.0, 3.0]
    assert BUY_ONLY.order_sides == {"BUY"}
    assert BUY_ONLY.coupled(_solution(spec.w0, np.array([0.0, 0.1, 0.2]), np.zeros(3))).tolist() == [0.0, 0.1, 0.2]


def test_buy_only_names_the_starts_it_cannot_trade_out_of(make: Factories) -> None:
    fine = make.spec(w0=np.array([0.3, 0.3, 0.3]), cash_lb=0.0, cash_ub=0.1)
    assert BUY_ONLY.infeasible_starts(fine) == []
    low_cash = make.spec(w0=np.array([0.3, 0.3, 0.3]), cash_lb=0.2, cash_ub=0.5)
    assert BUY_ONLY.infeasible_starts(low_cash) == ["the book starts with cash 0.100000 below cash_lb 0.200000, and a buy-only run can only lower cash"]
    over_cap = make.spec(w0=np.array([0.6, 0.3, 0.1]), ub=np.array([0.5, 0.3, 1.0]))
    assert BUY_ONLY.infeasible_starts(over_cap) == ["names whose cap is below their holding, which this side cannot trade out of: ['S0']"], "S1 sits exactly at its cap, which is allowed"


def test_buy_only_variables_are_one_vector_and_sell_is_unavailable() -> None:
    w0 = np.array([0.4, 0.3, 0.3])
    x = decision_variables("buy", w0)
    assert (x.sides, x.n) == ("buy", 3)
    assert x.buy.is_affine() and x.trade is x.buy and x.coupled is x.buy
    x.w.value = np.array([0.5, 0.3, 0.3])
    np.testing.assert_allclose(np.asarray(x.buy.value), [0.1, 0.0, 0.0])
    with pytest.raises(SideUnavailableError, match="a 'buy' run has no 'sell' vector"):
        _ = x.sell
    assert x.trade.variables() == [x.w], "no second variable: buy is an expression of w"
