"""The cvxpy half of the side profiles: the decision variables each side has, and the identity over them.

Neither is a configurable constraint — together they define what ``buy`` or ``sell`` *means*, and
every cost term, the turnover cap, the ADV constraint, and the verifier rely on that definition — so
both come from the run's ``sides``, and their numpy twins live on the profile in ``domain/sides.py``.
The identity includes the spec's own box, ``lb ≤ w ≤ ub``, which the schedule and the order rounding
already assume. Either side has one variable vector, ``w``, with its trade an affine expression of
it: no name can be on both sides of one solve, so no term can be paid for a round trip.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, Expr, at_least, at_most, shifted, shortfall, variable
from portfolio_optimizer.domain.results import ProblemSpec
from portfolio_optimizer.domain.sides import Sides


@dataclass(frozen=True, slots=True)
class CvxSide:
    """What a side is to cvxpy: how its variables are made, and the identity that binds them."""

    variables: Callable[[ProblemSpec], DecisionVars]
    identity: Callable[[DecisionVars, ProblemSpec], ConstraintSet]


def _buy_only_variables(spec: ProblemSpec) -> DecisionVars:
    """``w`` alone; ``buy = w - w0`` is an expression, ``sell`` does not exist."""
    w = variable(spec.n, "w")
    buy: Expr = shifted(w, spec.w0)
    return DecisionVars(w=w, n=spec.n, sides="buy", trade=buy, coupled=buy, _buy=buy)


def _buy_only_identity(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``w ≥ w0`` — nothing is sold — and the box ``lb ≤ w ≤ ub``."""
    return ConstraintSet("no_sells", (at_least(x.w, spec.w0), at_most(x.w, spec.ub), at_least(x.w, spec.lb)))


def _sell_only_variables(spec: ProblemSpec) -> DecisionVars:
    """``w`` alone; ``sell = w0 - w`` is an expression, ``buy`` does not exist."""
    w = variable(spec.n, "w")
    sell: Expr = shortfall(spec.w0, w)
    return DecisionVars(w=w, n=spec.n, sides="sell", trade=sell, coupled=sell, _sell=sell)


def _sell_only_identity(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``w ≤ w0`` — nothing is bought — and the box; the mirror of the buy-only identity."""
    return ConstraintSet("no_buys", (at_most(x.w, spec.w0), at_least(x.w, spec.lb), at_most(x.w, spec.ub)))


SIDES: Mapping[Sides, CvxSide] = {"buy": CvxSide(_buy_only_variables, _buy_only_identity), "sell": CvxSide(_sell_only_variables, _sell_only_identity)}
"""The cvxpy half for every ``sides`` value; a test holds this in step with ``domain.sides.PROFILES``."""


def decision_variables(sides: Sides, spec: ProblemSpec) -> DecisionVars:
    """The decision variables a ``sides`` run has, over ``spec``."""
    return SIDES[sides].variables(spec)


def identity_constraints(sides: Sides, x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """The trade identity and the box for ``sides`` over the variables ``x`` and the spec."""
    return SIDES[sides].identity(x, spec)
