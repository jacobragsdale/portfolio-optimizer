"""Business-logic rules — yours to edit.

A rule is an ordinary function that takes the portfolio's data bundle and returns a new one.
Name it in the run config's ``rules`` list and it runs, in order, between loading and the
optimizer. Declare parameters by annotating a ``params`` argument with a ``Params`` subclass. Rules
must be pure — same bundle in, same bundle out, no I/O — and they never see other portfolios. A
rule that shrinks a portfolio's *tradable set* — the securities it can trade on the side the run
couples through: marking a name ``restricted`` freezes it on both sides, capping ``max_weight`` at
its current weight takes it out of the buyable set — is what lets the engine solve portfolios
concurrently: two portfolios wait on each other only when they can both trade the same security on
that side.
"""

from decimal import Decimal

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.schemas import UNIVERSE
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


class AttachUniverseColumnsParams(Params):
    """Parameters for :func:`attach_universe_columns`."""

    columns: tuple[str, ...] = Field(
        default=(), description="Universe columns to copy onto holdings, matched on `security_id`. Default: every column the universe carries beyond its schema — the analytics columns."
    )


def attach_universe_columns(data: PortfolioData, params: AttachUniverseColumnsParams) -> PortfolioData:
    """Copy per-security columns from the universe onto holdings, so both tables carry the same analytics.

    An assembly ``join`` attaches analytics to `holdings` and `universe` in one place, but only while
    `holdings` is a global dataset: a `per_portfolio` one is never passed to assembly. This is the
    per-portfolio counterpart — it runs on the bundle, where both tables are already this portfolio's,
    and leaves them agreeing on column and dtype, which is what :meth:`PortfolioData.optimizer_frame`
    requires. The build reads analytics from the universe alone, so this is for what consumes that
    stacked frame: a custom solve step, or a later rule.

    A held name the universe does not carry gets a null, which is the honest answer and which the build
    refuses later by name; a ``bool`` column cannot hold one, so that combination is refused here
    instead. Copying the other way — a per-position analytic onto the universe, where a term can read
    it — is a rule of the same shape, and yours to write, because only you know what it means for a
    name nobody holds.
    """
    declared = {column.name for column in UNIVERSE.columns}
    named = params.columns or tuple(str(column) for column in data.universe.columns if str(column) not in declared)
    missing = [name for name in named if name not in data.universe.columns]
    if missing:
        msg = f"attach_universe_columns: the universe has no column(s) {missing}; it has {sorted(str(column) for column in data.universe.columns)}"
        raise ValueError(msg)
    present = [name for name in named if name in data.holdings.columns]
    if params.columns and present:
        msg = f"attach_universe_columns: holdings already has column(s) {present}; rename or drop them first rather than overwriting"
        raise ValueError(msg)
    attaching = [name for name in named if name not in data.holdings.columns]
    if not attaching:
        return data.with_changes()
    source = data.universe[["security_id", *attaching]]
    merged = data.holdings.merge(source, on="security_id", how="left", validate="many_to_one")
    unmatched = [name for name in attaching if merged[name].isna().any() and str(data.universe[name].dtype) == "bool"]
    if unmatched:
        msg = f"attach_universe_columns: column(s) {unmatched} are bool and cannot hold the null a held name outside the universe would take; make them nullable or restrict the copy"
        raise ValueError(msg)
    holdings = merged.astype({name: data.universe[name].dtype for name in attaching if not merged[name].isna().any()})
    return data.with_changes(holdings=holdings)
