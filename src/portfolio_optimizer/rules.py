"""Business-logic rules — yours to edit.

A rule is an ordinary function that takes the portfolio's data bundle and returns a new one.
Name it in the run config's ``rules`` list and it runs, in order, between loading and the
optimizer. Declare parameters by annotating a ``params`` argument with a ``Params`` subclass;
declare a dependency on earlier portfolios' results with a ``ctx: SolveContext`` argument
(sequential mode only). Rules must be pure: same bundle in, same bundle out, no I/O.
"""

from decimal import Decimal

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.results import SolveContext
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


def avoid_cross_portfolio_wash_sales(data: PortfolioData, ctx: SolveContext) -> PortfolioData:
    """Do not buy a name an earlier portfolio in this run sold: cap it at its current weight.

    Uses the optional ``max_weight`` universe column, which build turns into a per-security
    upper bound. Runs only in ``sequential`` mode because it needs prior results.
    """
    sold: set[str] = set()
    for result in ctx.results:
        sold.update(str(security) for security, side in zip(result.orders["security_id"], result.orders["side"], strict=True) if str(side) == "SELL")
    if not sold:
        return data.with_changes()
    nav = data.details.nav
    held = data.holdings.set_index("security_id")["quantity"]
    prices = data.universe.set_index("security_id")["price"]
    current: dict[str, Decimal] = {security: Decimal(int(held[security])) * prices[security] / nav if security in held.index else Decimal(0) for security in sold if security in prices.index}
    existing = data.universe["max_weight"] if "max_weight" in data.universe.columns else pd.Series([None] * len(data.universe), index=data.universe.index, dtype="object")
    capped = [
        min(current[security], limit) if security in current and limit is not None else current.get(security, limit)
        for security, limit in zip(data.universe["security_id"].astype(str), existing, strict=True)
    ]
    universe = data.universe.assign(max_weight=pd.Series(capped, index=data.universe.index, dtype="object"))
    return data.with_changes(universe=universe)
