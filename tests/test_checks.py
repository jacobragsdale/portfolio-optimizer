"""Tier 1: the shipped checks prove the shipped rules on the orders — the window's edge, a book that never reaches the rule, and what a violation row carries."""

import pandas as pd
import pytest

from portfolio_optimizer.checks import no_trades_inside_wash_window, restricted_never_traded
from portfolio_optimizer.domain.data import Frames as Datasets
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.schemas import CHECK_RESULTS
from portfolio_optimizer.rules import RecentTradesParams
from tests.conftest import Frames


def datasets(frames: Frames, universe: pd.DataFrame | None = None, **extras: pd.DataFrame) -> Datasets:
    """The assembled datasets a check receives: the engine-known four over the example's securities, plus ``extras``."""
    return Datasets(
        {"holdings": frames.holdings(), "universe": frames.three_security_universe() if universe is None else universe, "details": frames.details(), "constraints": frames.constraints(), **extras}
    )


def solved(*portfolio_ids: str) -> pd.DataFrame:
    """The portfolios that solved, as the runner hands them to a check."""
    return pd.DataFrame({"portfolio_id": pd.Series(list(portfolio_ids), dtype="string")})


def examined(result: pd.DataFrame) -> list[tuple[str, str, bool]]:
    validate_frame(result, CHECK_RESULTS)
    return list(result[["portfolio_id", "security_id", "ok"]].itertuples(index=False, name=None))


# --- restricted_never_traded ---


def test_restricted_never_traded_examines_every_portfolio_with_orders_against_every_restricted_name(frames: Frames) -> None:
    universe = frames.three_security_universe().assign(restricted=pd.Series([False, True, False], dtype="bool"))
    orders = frames.orders({"portfolio_id": "P1", "security_id": "A"}, {"portfolio_id": "P2", "security_id": "B", "side": "SELL", "quantity": 7}, {"portfolio_id": "P2", "security_id": "C"})
    result = restricted_never_traded(datasets(frames, universe), orders, solved("P1", "P2", "P3"))
    assert examined(result) == [("P1", "B", True), ("P2", "B", False), ("P3", "B", True)], (
        "one row per (solved portfolio, restricted name): P1 and P3 stayed away from B, P2 traded it; P3 traded nothing at all and is proven all the same"
    )
    violation = result.loc[~result["ok"]].iloc[0]
    assert (violation["side"], violation["quantity"]) == ("SELL", 7), "the row carries the order that broke the rule"
    assert result.loc[result["ok"], "side"].isna().all()


def test_restricted_never_traded_is_not_exercised_without_a_restricted_name(frames: Frames) -> None:
    orders = frames.orders({"portfolio_id": "P1", "security_id": "A"})
    assert examined(restricted_never_traded(datasets(frames), orders, solved("P1"))) == [], "no name is restricted, so there is nothing to prove"
    without_column = frames.three_security_universe().drop(columns=["restricted"])
    assert examined(restricted_never_traded(datasets(frames, without_column), orders, solved("P1"))) == []
    restricted = frames.three_security_universe().assign(restricted=pd.Series([True, True, True], dtype="bool"))
    assert examined(restricted_never_traded(datasets(frames, restricted), orders.iloc[0:0], solved())) == [], "no portfolio solved, so none was examined"


# --- no_trades_inside_wash_window ---


def test_no_trades_inside_wash_window_holds_at_the_windows_edge_and_proves_only_portfolios_with_orders(frames: Frames) -> None:
    trades = frames.trades(("P1", "A", "SELL", 30), ("P1", "C", "BUY", 31), ("P1", "A", "SELL", 60), ("P2", "B", "SELL", 1))
    orders = frames.orders({"portfolio_id": "P1", "security_id": "A", "quantity": 1000}, {"portfolio_id": "P1", "security_id": "C"})
    result = no_trades_inside_wash_window(datasets(frames, trades=trades), orders, solved("P1"), RecentTradesParams())
    assert examined(result) == [("P1", "A", False)], "A sold exactly thirty days ago is inside the window, C's trade a day earlier is not, and P2 did not solve"
    assert result.loc[0, "quantity"] == 1000 and result.loc[0, "traded_on"] == trades.loc[0, "traded_on"], "the latest trade inside the window, with the order that followed it"
    wider = no_trades_inside_wash_window(datasets(frames, trades=trades), orders, solved("P1", "P2"), RecentTradesParams(window_days=31))
    assert examined(wider) == [("P1", "A", False), ("P1", "C", False), ("P2", "B", True)], "P2 solved without an order and its recent B is proven left alone"
    clean = no_trades_inside_wash_window(datasets(frames, trades=trades), frames.orders({"portfolio_id": "P1", "security_id": "B"}), solved("P1"), RecentTradesParams())
    assert examined(clean) == [("P1", "A", True)], "P1 traded A recently and left it alone: examined, and ok"


def test_no_trades_inside_wash_window_is_not_exercised_without_a_recent_trade_or_a_solved_portfolio(frames: Frames) -> None:
    orders = frames.orders({"portfolio_id": "P1", "security_id": "A"})
    old = frames.trades(("P1", "A", "SELL", 200))
    assert examined(no_trades_inside_wash_window(datasets(frames, trades=old), orders, solved("P1"), RecentTradesParams())) == []
    assert examined(no_trades_inside_wash_window(datasets(frames, trades=frames.trades()), orders, solved("P1"), RecentTradesParams())) == []
    recent = frames.trades(("P1", "A", "SELL", 1))
    assert examined(no_trades_inside_wash_window(datasets(frames, trades=recent), orders.iloc[0:0], solved(), RecentTradesParams())) == [], "nothing solved, nothing examined"


def test_no_trades_inside_wash_window_reads_the_rules_dataset_and_refuses_a_missing_one(frames: Frames) -> None:
    orders = frames.orders({"portfolio_id": "P1", "security_id": "A"})
    blotter = frames.trades(("P1", "A", "SELL", 1))
    assert examined(no_trades_inside_wash_window(datasets(frames, blotter=blotter), orders, solved("P1"), RecentTradesParams(dataset="blotter"))) == [("P1", "A", False)]
    with pytest.raises(KeyError, match="no dataset 'trades'; available: "):
        no_trades_inside_wash_window(datasets(frames), orders, solved("P1"), RecentTradesParams())
    with pytest.raises(ValueError, match=r"trades dataset 'trades' needs column\(s\) \['traded_on'\]"):
        no_trades_inside_wash_window(datasets(frames, trades=blotter.drop(columns=["traded_on"])), orders, solved("P1"), RecentTradesParams())
