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
    """Tighten the account's single-name limit to ``max_weight`` when it is looser.

    The limits are ordinary fields of ``data.details``, loaded with the rest of the account's row, so
    a rule adjusts one the same way it would adjust any other input: by returning a bundle that holds
    the new value.
    """
    if params.max_weight >= data.details.max_weight:
        return data.with_changes()
    return data.with_changes(details=data.details.model_copy(update={"max_weight": params.max_weight}))


def add_zero_alpha(data: PortfolioData) -> PortfolioData:
    """Add an ``alpha`` column of zeros when the universe has none, so alpha-reading terms can run."""
    if "alpha" in data.universe.columns:
        return data.with_changes()
    universe = data.universe.assign(alpha=pd.Series(0.0, index=data.universe.index, dtype="Float64"))
    return data.with_changes(universe=universe)


class LiquidityParams(Params):
    """Where :func:`restrict_low_liquidity` reads its threshold: an extra dataset of named values, not a number in the config."""

    dataset: str = Field(default="buy_universe_parameters", min_length=1)
    key: str = Field(default="min_adv_shares", min_length=1)


def restrict_low_liquidity(data: PortfolioData, params: LiquidityParams) -> PortfolioData:
    """Freeze names whose average daily volume is below the threshold the parameter frame names; they keep their current weight.

    The threshold is loaded at runtime rather than written into the config, which is what lets it change
    daily without changing the run's identity — the frame is content-hashed and recorded in the manifest
    like any other input, so two runs that used different thresholds are visibly different runs.
    """
    try:
        frame = data.extras[params.dataset]
    except KeyError:
        msg = f"no extra dataset {params.dataset!r} to read {params.key!r} from; the run carries {sorted(data.extras)}"
        raise ValueError(msg) from None
    minimum = parameter(frame, params.key)
    illiquid = data.universe["adv_shares"] < minimum
    universe = data.universe.assign(restricted=(data.universe["restricted"] | illiquid).astype("bool"))
    return data.with_changes(universe=universe)


class MandateParams(Params):
    """Where :func:`restrict_to_mandate` reads the account's mandate: an extra dataset of ``portfolio_id``/``sector`` rows."""

    dataset: str = Field(default="mandates", min_length=1)


def restrict_to_mandate(data: PortfolioData, params: MandateParams) -> PortfolioData:
    """Freeze every universe name whose sector is outside the account's mandate; held names keep their current weight.

    The mandate is loaded data — a compliance service's answer for this account, one row per allowed
    sector (:func:`~portfolio_optimizer.loaders.load_mandates` is the shipped source) — so which names
    an account may trade changes daily without changing the run's identity. Shrinking the tradable set
    this way is also what gives the derived dependency graph real components: two accounts whose
    mandates share no sector cannot affect each other, however large the shared universe is, so their
    solves never wait on one another. An account with no mandate rows is refused rather than silently
    frozen out of every name, which is more likely missing data than intent.
    """
    try:
        frame = data.extras[params.dataset]
    except KeyError:
        msg = f"no extra dataset {params.dataset!r} to read the mandate from; the run carries {sorted(data.extras)}"
        raise ValueError(msg) from None
    if "sector" not in frame.columns:
        msg = f"mandate dataset {params.dataset!r} needs a 'sector' column; it has {sorted(str(column) for column in frame.columns)}"
        raise ValueError(msg)
    allowed = {str(sector) for sector in frame["sector"]}
    if not allowed:
        msg = f"portfolio {data.portfolio_id!r} has no rows in {params.dataset!r}: an empty mandate would freeze every name"
        raise ValueError(msg)
    outside = ~data.universe["sector"].isin(sorted(allowed))
    universe = data.universe.assign(restricted=(data.universe["restricted"] | outside).astype("bool"))
    return data.with_changes(universe=universe)


def parameter(frame: pd.DataFrame, key: str) -> Decimal:
    """Read one value out of a ``name``/``value`` parameter frame, the narrowest shape a runtime setting can take.

    The engine knows nothing about such a frame — it is an ordinary extra dataset — so this is a
    convention of the template layer, and a desk with a wider one writes its own reader.
    """
    missing = [column for column in ("name", "value") if column not in frame.columns]
    if missing:
        msg = f"parameter frame is missing column(s) {missing}; it should carry 'name' and 'value'"
        raise ValueError(msg)
    rows = frame.loc[frame["name"] == key, "value"]
    if len(rows) != 1:
        msg = f"parameter {key!r}: expected exactly one row, found {len(rows)} among {sorted(str(name) for name in frame['name'])}"
        raise ValueError(msg)
    value = rows.iloc[0]
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        msg = f"parameter {key!r} is {type(value).__name__}, expected an exact Decimal — declare its kind in the loader's dtypes"
        raise TypeError(msg)
    return Decimal(value)


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
