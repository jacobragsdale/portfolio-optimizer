"""Business-rule checks — yours to edit.

A check is an ordinary function that takes every assembled dataset — as the rules first saw them,
before any rule ran — the orders the run published, and the portfolios that solved, and returns every
row the rule it proves *examined*, with a boolean ``ok`` that is false where the rule was breached. Name it in the run
config's ``checks`` list under a ``label`` and it runs once, after the sink, over the whole batch;
the manifest records it under that label as ``passed``, ``failed``, or — when it examined nothing,
because the book never put the rule to the test — ``not_exercised``, and the rows that failed go to
``checks/<label>.csv`` beside the manifest.

What "examined" means is the check's own honesty: one row per case the rule *applies to*, so that a
book that never reaches the rule is reported as not having proven it rather than as passing it. The
population is ``solved`` — every portfolio that produced an answer — not the portfolios that happened
to trade: a rule that stopped a trade is proven on the account that traded nothing. A
check reads the pre-rule data on purpose — the rules are what it proves — and it is the second half
of the run's proof: the verifier (``engine/check.py``) re-checks every typed constraint row on the
solved weights, a check proves a Python rule on the orders that went out.
"""

from datetime import timedelta

import pandas as pd

from portfolio_optimizer.domain.data import Frames
from portfolio_optimizer.rules import RecentTradesParams, restricted_flags


def restricted_never_traded(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame) -> pd.DataFrame:
    """No order in a name the universe marks ``restricted``.

    Examined: every (solved portfolio, restricted name) pair, so a universe with no restricted name is
    ``not_exercised``. Detail: the offending order's ``side`` and ``quantity``.
    """
    universe = frames["universe"]
    restricted = universe.loc[restricted_flags(universe), ["security_id"]]
    return _no_order_in(solved[["portfolio_id"]].merge(restricted, how="cross"), orders)


def no_trades_inside_wash_window(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame, params: RecentTradesParams) -> pd.DataFrame:
    """No order in a name the account traded inside the window: what ``restrict_recent_trades`` promises, proven on the orders.

    Reads the same dataset under the same params as the rule, so the two cannot drift apart; the
    as-of instant is the orders' own. Examined: every (solved portfolio, name it traded inside the
    window) pair — its latest such trade — so a book with no recent trade is ``not_exercised``, and
    an account the rule kept out of a name is examined whether or not it traded anything else. Detail: when the name was traded, and the offending order's ``side`` and ``quantity``.
    """
    trades = frames[params.dataset]
    missing = [column for column in ("portfolio_id", "security_id", "traded_on") if column not in trades.columns]
    if missing:
        msg = f"trades dataset {params.dataset!r} needs column(s) {missing}; it has {sorted(str(column) for column in trades.columns)}"
        raise ValueError(msg)
    as_of = orders["as_of_date"].max()
    recent = trades.loc[trades["traded_on"] >= as_of - timedelta(days=params.window_days), ["portfolio_id", "security_id", "traded_on"]]
    latest = recent.sort_values(["portfolio_id", "security_id", "traded_on"], kind="stable").drop_duplicates(["portfolio_id", "security_id"], keep="last")
    return _no_order_in(latest[latest["portfolio_id"].isin(solved["portfolio_id"])], orders)


def _no_order_in(forbidden: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """``forbidden`` — one row per (portfolio, name) the rule allows no order in — with the order found there, if any, and ``ok`` where none was."""
    found = forbidden.merge(orders[["portfolio_id", "security_id", "side", "quantity"]], on=["portfolio_id", "security_id"], how="left", validate="one_to_one")
    return found.assign(ok=found["side"].isna().astype("bool")).sort_values(["portfolio_id", "security_id"], kind="stable").reset_index(drop=True)
