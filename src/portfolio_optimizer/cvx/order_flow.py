"""The cvxpy half of the order-flow profiles: the decision variables each order flow has, and the identity over them.

Neither is a configurable constraint — together they define what ``buy`` or ``sell`` *means*, and
every cost term, the turnover cap, the ADV constraint, and the verifier rely on that definition — so
both come from the run's ``order_flow``, and their numpy twins live on the profile in
``domain/order_flow.py``. The identity includes the spec's own box, ``lb ≤ w ≤ ub``, which the
schedule and the order rounding already assume. Every order flow has one variable vector, ``w``,
with its trade a function of it — affine under an inflow or an outflow, the convex ``pos`` of the
change under a rebalance: no name can be on both sides of one solve, so no term can be paid for a
round trip.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, Expr, absolute, at_least, at_most, pos, shifted, shortfall, variable
from portfolio_optimizer.domain.order_flow import OrderFlow
from portfolio_optimizer.domain.results import ProblemSpec


@dataclass(frozen=True, slots=True)
class CvxOrderFlow:
    """What an order flow is to cvxpy: how its variables are made, and the identity that binds them."""

    variables: Callable[[ProblemSpec], DecisionVars]
    identity: Callable[[DecisionVars, ProblemSpec], ConstraintSet]


def _inflow_variables(spec: ProblemSpec) -> DecisionVars:
    """``w`` alone; ``buy = w - w0`` is an expression, ``sell`` does not exist."""
    w = variable(spec.n, "w")
    buy: Expr = shifted(w, spec.w0)
    return DecisionVars(w=w, n=spec.n, order_flow="inflow", trade=buy, coupled=buy, _buy=buy)


def _inflow_identity(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``w ≥ w0`` — nothing is sold — and the box ``lb ≤ w ≤ ub``."""
    return ConstraintSet("no_sells", (at_least(x.w, spec.w0), at_most(x.w, spec.ub), at_least(x.w, spec.lb)))


def _outflow_variables(spec: ProblemSpec) -> DecisionVars:
    """``w`` alone; ``sell = w0 - w`` is an expression, ``buy`` does not exist."""
    w = variable(spec.n, "w")
    sell: Expr = shortfall(spec.w0, w)
    return DecisionVars(w=w, n=spec.n, order_flow="outflow", trade=sell, coupled=sell, _sell=sell)


def _outflow_identity(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``w ≤ w0`` — nothing is bought — and the box; the mirror of the inflow identity."""
    return ConstraintSet("no_buys", (at_most(x.w, spec.w0), at_least(x.w, spec.lb), at_most(x.w, spec.ub)))


def _rebalance_variables(spec: ProblemSpec) -> DecisionVars:
    """``w`` alone; ``buy = pos(w - w0)``, ``sell = pos(w0 - w)``, and the trade is ``|w - w0|`` — convex, so a cost on any of them is fine and a reward is refused."""
    w = variable(spec.n, "w")
    change: Expr = shifted(w, spec.w0)
    trade: Expr = absolute(change)
    return DecisionVars(w=w, n=spec.n, order_flow="rebalance", trade=trade, coupled=trade, _buy=pos(change), _sell=pos(shortfall(spec.w0, w)))


def _rebalance_identity(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """The box alone: ``w`` may move either way inside ``lb ≤ w ≤ ub``."""
    return ConstraintSet("box", (at_least(x.w, spec.lb), at_most(x.w, spec.ub)))


ORDER_FLOWS: Mapping[OrderFlow, CvxOrderFlow] = {
    "inflow": CvxOrderFlow(_inflow_variables, _inflow_identity),
    "outflow": CvxOrderFlow(_outflow_variables, _outflow_identity),
    "rebalance": CvxOrderFlow(_rebalance_variables, _rebalance_identity),
}
"""The cvxpy half for every ``order_flow`` value; a test holds this in step with ``domain.order_flow.PROFILES``."""


def decision_variables(order_flow: OrderFlow, spec: ProblemSpec) -> DecisionVars:
    """The decision variables an ``order_flow`` run has, over ``spec``."""
    return ORDER_FLOWS[order_flow].variables(spec)


def identity_constraints(order_flow: OrderFlow, x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """The trade identity and the box for ``order_flow`` over the variables ``x`` and the spec."""
    return ORDER_FLOWS[order_flow].identity(x, spec)
