"""Tier 1: the cvxpy half of each side profile — how many variables, which vector is the trade, and the side a one-sided run refuses to hand out."""

import numpy as np
import pytest

from portfolio_optimizer.cvx.adapter import SideUnavailableError
from portfolio_optimizer.cvx.sides import decision_variables


def test_two_sided_variables_are_three_vectors_and_couple_through_buy() -> None:
    x = decision_variables("both", np.array([0.4, 0.3, 0.3]))
    assert (x.sides, x.n) == ("both", 3)
    assert x.buy is not x.w and x.sell is not x.w and x.buy.is_affine() and x.sell.is_affine()
    assert x.trade.shape == (3,) and x.coupled is x.buy


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


def test_sell_only_variables_are_one_vector_and_buy_is_unavailable() -> None:
    w0 = np.array([0.4, 0.3, 0.3])
    x = decision_variables("sell", w0)
    assert (x.sides, x.n) == ("sell", 3)
    assert x.sell.is_affine() and x.trade is x.sell and x.coupled is x.sell
    x.w.value = np.array([0.3, 0.3, 0.3])
    np.testing.assert_allclose(np.asarray(x.sell.value), [0.1, 0.0, 0.0])
    with pytest.raises(SideUnavailableError, match="a 'sell' run has no 'buy' vector"):
        _ = x.buy
    assert x.trade.variables() == [x.w]
