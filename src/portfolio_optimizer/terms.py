"""Objective terms and constraints — yours to edit.

An objective term is ``(x: DecisionVars, spec: ProblemSpec, params: P) -> ObjectiveTerm``; a
constraint is the same signature returning ``ConstraintSet``. Add ``chain: ChainState`` to read
what higher-priority portfolios in the run have already *bought* among the names this portfolio
may buy; declaring it is what makes this portfolio wait for those with overlapping buy universes.
Everything is expressed through the typed atoms in :mod:`portfolio_optimizer.cvx.adapter`, so the
post-solve verifier can recompute each shipped term and constraint without cvxpy.

Decision variables are fractions of NAV: ``w`` is the target weight, ``buy`` and ``sell`` its
non-negative split against the current weight ``spec.w0``.
"""

from decimal import Decimal

import numpy as np
from pydantic import Field

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, at_least, at_most, dot, matvec, plus, scale, shifted, sum_squares, total
from portfolio_optimizer.domain.results import ChainState, ProblemSpec
from portfolio_optimizer.domain.types import Params


class WeightedParams(Params):
    """A non-negative weight multiplying the term."""

    weight: Decimal = Field(default=Decimal(1), ge=0)


def tracking_error(x: DecisionVars, spec: ProblemSpec, params: WeightedParams) -> ObjectiveTerm:
    """``weight · |w - w_target|^2`` — squared deviation from the target weights."""
    return ObjectiveTerm("tracking_error", scale(float(params.weight), sum_squares(shifted(x.w, spec.w_target))))


class AlphaParams(WeightedParams):
    """Which per-security column holds the expected return."""

    column: str = "alpha"


def alpha(x: DecisionVars, spec: ProblemSpec, params: AlphaParams) -> ObjectiveTerm:
    """``-weight · alpha^T w`` — reward expected return; ``alpha`` comes from a universe column exported into the spec."""
    return ObjectiveTerm("alpha", scale(float(params.weight), dot(-spec.column(params.column), x.w)))


def tax_cost(x: DecisionVars, spec: ProblemSpec, params: WeightedParams) -> ObjectiveTerm:
    """``weight · tau^T sell`` — tax owed on realized gains; losses reduce the objective.

    A loss-harvest incentive with zero transaction cost would let the solver sell and rebuy a
    name for free, so that combination is refused here rather than caught after the fact.
    """
    if np.any(spec.tax_per_dollar < 0.0) and not np.any(spec.tcost_per_dollar > 0.0):
        msg = "tax_cost has a loss-harvest incentive but no transaction cost anywhere; add 'transaction_cost' with a positive cost_bps or a tcost_bps column"
        raise ValueError(msg)
    return ObjectiveTerm("tax_cost", scale(float(params.weight), dot(spec.tax_per_dollar, x.sell)))


class TransactionCostParams(WeightedParams):
    """A flat cost in basis points added to any per-security ``tcost_bps`` column."""

    cost_bps: Decimal = Field(default=Decimal(0), ge=0)


def transaction_cost(x: DecisionVars, spec: ProblemSpec, params: TransactionCostParams) -> ObjectiveTerm:
    """``weight · c^T(buy + sell)`` with ``c = tcost_per_dollar + cost_bps / 10^4``."""
    cost = spec.tcost_per_dollar + float(params.cost_bps) / 10_000.0
    return ObjectiveTerm("transaction_cost", scale(float(params.weight), dot(cost, plus(x.buy, x.sell))))


def long_only(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``w ≥ lb`` — no shorts, plus any per-security floor; restricted names are frozen at ``w0``."""
    return ConstraintSet("long_only", (at_least(x.w, spec.lb),))


def max_weight(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``w ≤ ub`` — the style's single-name cap, tightened by any per-security ``max_weight`` column."""
    return ConstraintSet("max_weight", (at_most(x.w, spec.ub),))


def cash_bounds(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``cash_lb ≤ 1 - sum(w) ≤ cash_ub``; full investment is ``cash_bounds = [0, 0]``."""
    invested = total(x.w)
    return ConstraintSet("cash_bounds", (at_most(invested, 1.0 - spec.cash_lb), at_least(invested, 1.0 - spec.cash_ub)))


class SectorBoundsParams(Params):
    """Slack applied symmetrically to every sector bound."""

    tolerance: Decimal = Field(default=Decimal(0), ge=0)


def sector_bounds(x: DecisionVars, spec: ProblemSpec, params: SectorBoundsParams) -> ConstraintSet:
    """``sector_lb - tol ≤ G w ≤ sector_ub + tol`` for the style's sector limits."""
    if len(spec.sector_names) == 0:
        return ConstraintSet("sector_bounds", ())
    exposure = matvec(spec.sector_matrix, x.w)
    tolerance = float(params.tolerance)
    return ConstraintSet("sector_bounds", (at_least(exposure, spec.sector_lb - tolerance), at_most(exposure, spec.sector_ub + tolerance)))


def turnover_cap(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``sum(buy + sell) ≤ max_turnover`` — two-way turnover as a fraction of NAV."""
    return ConstraintSet("turnover_cap", (at_most(total(plus(x.buy, x.sell)), spec.max_turnover),))


def cumulative_adv_participation(x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:
    """``buy + sell ≤ adv_capacity`` for this portfolio's own participation, and ``buy ≤ remaining`` where higher-priority portfolios' buys have already consumed part of each name's budget.

    Sells are the portfolio's own business: what others bought never limits them.
    """
    return ConstraintSet("cumulative_adv_participation", (at_most(plus(x.buy, x.sell), spec.adv_capacity), at_most(x.buy, adv_remaining(spec, chain))))


def adv_remaining(spec: ProblemSpec, chain: ChainState) -> np.ndarray:
    """The per-name buy budget left for this portfolio after predecessors' buys, as a fraction of its NAV; shared with the verifier."""
    if chain.security_ids != spec.security_ids:
        msg = "chain state is not aligned to this spec's securities"
        raise ValueError(msg)
    consumed = chain.bought_shares * spec.price / spec.nav
    return np.maximum(0.0, spec.adv_capacity - consumed)
