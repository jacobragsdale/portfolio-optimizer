"""Business-logic rules — yours to edit.

A rule is an ordinary function that takes the portfolio's data bundle and returns a new one.
Name it in the run config's ``rules`` list and it runs, in order, between loading and the
optimizer. Declare parameters by annotating a ``params`` argument with a ``Params`` subclass. Rules
must be pure — same bundle in, same bundle out, no I/O — and they never see other portfolios. A
rule that shrinks the *buy* universe (marking a name ``restricted``, capping ``max_weight`` at its
current weight) is what lets the engine solve portfolios concurrently: two portfolios wait on each
other only when they can both buy the same security.
"""

from decimal import Decimal

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.types import Params


class CapSingleNameParams(Params):
    """Parameters for :func:`cap_single_name`."""

    max_weight: Decimal = Field(gt=0, le=1)


def cap_single_name(data: PortfolioData, params: CapSingleNameParams) -> PortfolioData:
    """Tighten the style's single-name limit to ``max_weight`` when it is looser."""
    if params.max_weight >= data.style.max_weight:
        return data.with_changes()
    return data.with_changes(style=data.style.model_copy(update={"max_weight": params.max_weight}))


def add_zero_alpha(data: PortfolioData) -> PortfolioData:
    """Add an ``alpha`` column of zeros when the universe has none, so alpha-reading terms can run."""
    if "alpha" in data.universe.columns:
        return data.with_changes()
    universe = data.universe.assign(alpha=pd.Series(0.0, index=data.universe.index, dtype="Float64"))
    return data.with_changes(universe=universe)


class LiquidityParams(Params):
    """Parameters for :func:`restrict_low_liquidity`."""

    min_adv_shares: int = Field(ge=0)


def restrict_low_liquidity(data: PortfolioData, params: LiquidityParams) -> PortfolioData:
    """Freeze names whose average daily volume is below ``min_adv_shares``; they keep their current weight."""
    illiquid = data.universe["adv_shares"] < params.min_adv_shares
    universe = data.universe.assign(restricted=(data.universe["restricted"] | illiquid).astype("bool"))
    return data.with_changes(universe=universe)
