"""The cvxpy half of the side profiles: the decision variables each side has, and the trade identity over them.

Neither is a configurable constraint — together they define what ``buy`` and ``sell`` *mean*, and
every cost term, the turnover cap, the ADV constraint, and the verifier rely on that definition — so
both come from the run's ``sides``, and their numpy twins live on the profile in ``domain/sides.py``.
A one-sided run has one variable vector, ``w``, with its trade an affine expression of it: a third of
the two-sided KKT system, and no wash trade possible by construction.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, Expr, at_least, at_most, equals, minus, plus, shifted, variable
from portfolio_optimizer.domain.results import F64


@dataclass(frozen=True, slots=True)
class CvxSide:
    """What a side is to cvxpy: how its variables are made, and the identity that binds them."""

    variables: Callable[[F64], DecisionVars]
    identity: Callable[[DecisionVars, F64], ConstraintSet]


def _two_sided_variables(w0: F64) -> DecisionVars:
    """``w``, ``buy``, ``sell`` all variables; the trade is ``buy + sell`` and the run couples through ``buy``."""
    n = len(w0)
    buy, sell = variable(n, "buy"), variable(n, "sell")
    return DecisionVars(w=variable(n, "w"), n=n, sides="both", trade=plus(buy, sell), coupled=buy, _buy=buy, _sell=sell)


def _two_sided_identity(x: DecisionVars, w0: F64) -> ConstraintSet:
    """``w - w0 = buy - sell``, ``buy, sell ≥ 0``, ``sell ≤ w0`` — the definition of the buy/sell split."""
    return ConstraintSet("trade_balance", (equals(shifted(x.w, w0), minus(x.buy, x.sell)), at_least(x.buy, 0.0), at_least(x.sell, 0.0), at_most(x.sell, w0)))


def _buy_only_variables(w0: F64) -> DecisionVars:
    """``w`` alone; ``buy = w - w0`` is an expression, ``sell`` does not exist."""
    n = len(w0)
    w = variable(n, "w")
    buy: Expr = shifted(w, w0)
    return DecisionVars(w=w, n=n, sides="buy", trade=buy, coupled=buy, _buy=buy)


def _buy_only_identity(x: DecisionVars, w0: F64) -> ConstraintSet:
    """``w ≥ w0`` — nothing is sold, which is all the identity has to say when ``buy`` is ``w - w0`` by construction."""
    return ConstraintSet("no_sells", (at_least(x.w, w0),))


SIDES: Mapping[str, CvxSide] = {"both": CvxSide(_two_sided_variables, _two_sided_identity), "buy": CvxSide(_buy_only_variables, _buy_only_identity)}
"""The cvxpy half for every ``sides`` value; a test holds this in step with ``domain.sides.PROFILES``."""


def decision_variables(sides: str, w0: F64) -> DecisionVars:
    """The decision variables a ``sides`` run has, over the starting weights ``w0``."""
    return SIDES[sides].variables(w0)


def identity_constraints(sides: str, x: DecisionVars, w0: F64) -> ConstraintSet:
    """The trade identity for ``sides`` over the variables ``x`` and the starting weights ``w0``."""
    return SIDES[sides].identity(x, w0)
