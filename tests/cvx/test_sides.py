"""Tier 1: the cvxpy half of each side profile — one variable, the trade an expression of it, the side a run refuses to hand out, and the box in the identity."""

import cvxpy as cp
import numpy as np
import pytest

from portfolio_optimizer.cvx.adapter import SideUnavailableError
from portfolio_optimizer.cvx.sides import decision_variables, identity_constraints
from portfolio_optimizer.domain.sides import Sides
from tests.conftest import Factories

W0 = np.array([0.4, 0.3, 0.3])


@pytest.mark.parametrize(("sides", "direction", "expected"), [("sell", cp.Minimize, [0.1, 0.0, 0.0]), ("buy", cp.Maximize, [0.5, 0.5, 0.5])])
def test_the_identity_of_every_side_holds_the_specs_box(make: Factories, sides: Sides, direction: type[cp.Minimize | cp.Maximize], expected: list[float]) -> None:
    """Pushed as far as the identity alone allows, ``w`` stops at the floor selling and at the cap buying: the box is in the identity, not a configurable row."""
    spec = make.spec(w0=W0, lb=np.array([0.1, 0.0, 0.0]), ub=np.array([0.5, 0.5, 0.5]))
    x = decision_variables(sides, spec)
    problem = cp.Problem(direction(cp.sum(x.w)), list(identity_constraints(sides, x, spec).constraints))
    problem.solve(solver="CLARABEL")
    assert problem.status == "optimal" and x.w.value is not None
    np.testing.assert_allclose(x.w.value, expected, atol=1e-6)


def test_buy_only_variables_are_one_vector_and_sell_is_unavailable(make: Factories) -> None:
    x = decision_variables("buy", make.spec(w0=W0))
    assert (x.sides, x.n) == ("buy", 3)
    assert x.buy.is_affine() and x.trade is x.buy and x.coupled is x.buy, "the trade is w - w0, an expression of the one variable"
    assert x.vector("trade") is x.trade and x.vector("w") is x.w and x.vector("buy") is x.buy
    x.w.value = W0 + np.array([0.1, 0.0, 0.0])
    assert x.buy.value is not None
    np.testing.assert_allclose(x.buy.value, [0.1, 0.0, 0.0])
    with pytest.raises(SideUnavailableError, match="a 'buy' run has no 'sell' vector") as info:
        _ = x.sell
    assert (info.value.side, info.value.sides) == ("sell", "buy")
    with pytest.raises(SideUnavailableError):
        x.vector("sell")


def test_sell_only_variables_are_one_vector_and_buy_is_unavailable(make: Factories) -> None:
    x = decision_variables("sell", make.spec(w0=W0))
    assert (x.sides, x.n) == ("sell", 3)
    assert x.sell.is_affine() and x.trade is x.sell and x.coupled is x.sell, "the trade is w0 - w, an expression of the one variable"
    x.w.value = W0 - np.array([0.1, 0.0, 0.0])
    assert x.sell.value is not None
    np.testing.assert_allclose(x.sell.value, [0.1, 0.0, 0.0])
    with pytest.raises(SideUnavailableError, match="a 'sell' run has no 'buy' vector"):
        _ = x.buy
