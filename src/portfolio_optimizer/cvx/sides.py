"""The cvxpy half of the side profiles: the trade identity each side imposes on the decision variables.

The identity is not a configurable constraint — it defines what ``buy`` and ``sell`` *mean*, and
every cost term, the turnover cap, the ADV constraint, and the verifier rely on that definition — so
it comes from the run's ``sides``, and its numpy twin lives on the profile in ``domain/sides.py``.
"""

from collections.abc import Callable, Mapping

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, at_least, at_most, equals, minus, shifted
from portfolio_optimizer.domain.results import F64


def _two_sided(x: DecisionVars, w0: F64) -> ConstraintSet:
    """``w - w0 = buy - sell``, ``buy, sell ≥ 0``, ``sell ≤ w0`` — the definition of the buy/sell split."""
    return ConstraintSet("trade_balance", (equals(shifted(x.w, w0), minus(x.buy, x.sell)), at_least(x.buy, 0.0), at_least(x.sell, 0.0), at_most(x.sell, w0)))


IDENTITIES: Mapping[str, Callable[[DecisionVars, F64], ConstraintSet]] = {"both": _two_sided}
"""The identity for every ``sides`` value; a test holds this in step with ``domain.sides.PROFILES``."""


def identity_constraints(sides: str, x: DecisionVars, w0: F64) -> ConstraintSet:
    """The trade identity for ``sides`` over the variables ``x`` and the starting weights ``w0``."""
    return IDENTITIES[sides](x, w0)
