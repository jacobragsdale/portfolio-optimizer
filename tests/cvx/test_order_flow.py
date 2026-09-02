"""Tier 1: the cvxpy half of each order-flow profile — one variable, the trade an expression of it, the side a run refuses to hand out, and the box in the identity."""

import cvxpy as cp
import numpy as np
import pytest

from portfolio_optimizer.cvx.adapter import SideUnavailableError
from portfolio_optimizer.cvx.order_flow import decision_variables, identity_constraints
from portfolio_optimizer.domain.order_flow import OrderFlow
from tests.conftest import Factories

W0 = np.array([0.4, 0.3, 0.3])


@pytest.mark.parametrize(("order_flow", "direction", "expected"), [("outflow", cp.Minimize, [0.1, 0.0, 0.0]), ("inflow", cp.Maximize, [0.5, 0.5, 0.5])])
def test_the_identity_of_every_side_holds_the_specs_box(make: Factories, order_flow: OrderFlow, direction: type[cp.Minimize | cp.Maximize], expected: list[float]) -> None:
    """Pushed as far as the identity alone allows, ``w`` stops at the floor selling and at the cap buying: the box is in the identity, not a configurable row."""
    spec = make.spec(w0=W0, lb=np.array([0.1, 0.0, 0.0]), ub=np.array([0.5, 0.5, 0.5]))
    x = decision_variables(order_flow, spec)
    problem = cp.Problem(direction(cp.sum(x.w)), list(identity_constraints(order_flow, x, spec).constraints))
    problem.solve(solver="CLARABEL")
    assert problem.status == "optimal" and x.w.value is not None
    np.testing.assert_allclose(x.w.value, expected, atol=1e-6)


def test_inflow_variables_are_one_vector_and_sell_is_unavailable(make: Factories) -> None:
    x = decision_variables("inflow", make.spec(w0=W0))
    assert (x.order_flow, x.n) == ("inflow", 3)
    assert x.buy.is_affine() and x.trade is x.buy and x.coupled is x.buy, "the trade is w - w0, an expression of the one variable"
    assert x.vector("trade") is x.trade and x.vector("w") is x.w and x.vector("buy") is x.buy
    x.w.value = W0 + np.array([0.1, 0.0, 0.0])
    assert x.buy.value is not None
    np.testing.assert_allclose(x.buy.value, [0.1, 0.0, 0.0])
    with pytest.raises(SideUnavailableError, match="order flow 'inflow' has no 'sell' vector") as info:
        _ = x.sell
    assert (info.value.side, info.value.order_flow) == ("sell", "inflow")
    with pytest.raises(SideUnavailableError):
        x.vector("sell")


def test_outflow_variables_are_one_vector_and_buy_is_unavailable(make: Factories) -> None:
    x = decision_variables("outflow", make.spec(w0=W0))
    assert (x.order_flow, x.n) == ("outflow", 3)
    assert x.sell.is_affine() and x.trade is x.sell and x.coupled is x.sell, "the trade is w0 - w, an expression of the one variable"
    x.w.value = W0 - np.array([0.1, 0.0, 0.0])
    assert x.sell.value is not None
    np.testing.assert_allclose(x.sell.value, [0.1, 0.0, 0.0])
    with pytest.raises(SideUnavailableError, match="order flow 'outflow' has no 'buy' vector"):
        _ = x.buy
