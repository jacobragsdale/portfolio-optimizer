"""Objective terms and constraints — yours to edit.

An objective term is ``(x: DecisionVars, spec: ProblemSpec, params: P) -> ObjectiveTerm``; a
constraint is the same signature returning ``ConstraintSet``. Add ``chain: ChainState`` to read
what higher-priority portfolios in the run have already *traded*, on the side the run couples
through, among the names this portfolio can trade there; declaring it is what makes this portfolio
wait for those with overlapping tradable sets. Everything is expressed through the typed atoms in
:mod:`portfolio_optimizer.cvx.adapter`, so the post-solve verifier can recompute each shipped term
and constraint without cvxpy.

Decision variables are fractions of NAV: ``w`` is the target weight; ``buy`` and ``sell`` its
non-negative split against the current weight ``spec.w0``, each present only on a side the run
has; ``trade`` the amount traded on the sides it has, and ``coupled`` the amount traded on the side
it couples through. A term that reads a side the run lacks fails at ``validate-config``.
"""

from decimal import Decimal
from typing import Self

import numpy as np
from pydantic import Field, model_validator

from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, at_least, at_most, dot, matvec, scale, total
from portfolio_optimizer.domain.results import ChainState, ProblemSpec
from portfolio_optimizer.domain.types import Params


class WeightedParams(Params):
    """A non-negative weight multiplying the term."""

    weight: Decimal = Field(default=Decimal(1), ge=0)


class AlphaParams(WeightedParams):
    """Which per-security column holds the expected return."""

    column: str = "alpha"


def alpha(x: DecisionVars, spec: ProblemSpec, params: AlphaParams) -> ObjectiveTerm:
    """``-weight · alpha^T w`` — reward expected return; ``alpha`` comes from a universe column exported into the spec."""
    return ObjectiveTerm("alpha", scale(float(params.weight), dot(-spec.column(params.column), x.w)))


def tax_cost(x: DecisionVars, spec: ProblemSpec, params: WeightedParams) -> ObjectiveTerm:
    """``weight · tau^T sell`` — tax owed on realized gains; losses reduce the objective. Reads ``sell``, so a buy-only run refuses it.

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
    """``weight · c^T trade`` with ``c = tcost_per_dollar + cost_bps / 10^4``; ``trade`` is ``buy + sell``, or the one side the run has."""
    cost = spec.tcost_per_dollar + float(params.cost_bps) / 10_000.0
    return ObjectiveTerm("transaction_cost", scale(float(params.weight), dot(cost, x.trade)))


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


class SectorBoundParams(Params):
    """One sector's exposure band, and the slack the verifier allows on it.

    The numbers live on the constraint row, not in the spec: a run bounds a sector by giving the
    account a row naming it, and bounds another by adding a second row with its own label.
    """

    sector: str = Field(min_length=1)
    lower: Decimal = Field(default=Decimal(0), ge=0, le=1)
    upper: Decimal = Field(default=Decimal(1), ge=0, le=1)
    tolerance: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if self.lower > self.upper:
            msg = f"lower must not exceed upper, got {self.lower} > {self.upper}"
            raise ValueError(msg)
        return self


def sector_bound(x: DecisionVars, spec: ProblemSpec, params: SectorBoundParams) -> ConstraintSet:
    """``lower - tol ≤ sum of w over one sector ≤ upper + tol``; the sector's membership comes from the spec, its band from this row."""
    exposure = matvec(spec.sector(params.sector), x.w)
    tolerance = float(params.tolerance)
    return ConstraintSet("sector_bound", (at_least(exposure, float(params.lower) - tolerance), at_most(exposure, float(params.upper) + tolerance)))


def turnover_cap(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:
    """``sum(trade) ≤ max_turnover`` — turnover as a fraction of NAV, two-way where the run has two sides."""
    return ConstraintSet("turnover_cap", (at_most(total(x.trade), spec.max_turnover),))


def cumulative_adv_participation(x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:
    """``trade ≤ adv_capacity`` for this portfolio's own participation, and ``coupled ≤ remaining`` where higher-priority portfolios' trades on the side the run couples through have already consumed part of each name's budget.

    The other side, where the run has one, is the portfolio's own business: what others traded never limits it.
    """
    return ConstraintSet("cumulative_adv_participation", (at_most(x.trade, spec.adv_capacity), at_most(x.coupled, adv_remaining(spec, chain))))


def adv_remaining(spec: ProblemSpec, chain: ChainState) -> np.ndarray:
    """The per-name budget left for this portfolio on the side the run couples through, after predecessors' trades there, as a fraction of its NAV; shared with the verifier."""
    if chain.security_ids != spec.security_ids:
        msg = "chain state is not aligned to this spec's securities"
        raise ValueError(msg)
    consumed = chain.traded_shares * spec.price / spec.nav
    return np.maximum(0.0, spec.adv_capacity - consumed)
