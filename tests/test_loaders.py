"""Tier 3/5: the shipped loaders fetch what the request asks for, type it at the boundary, and pace themselves against the input's rate limit."""

import asyncio
import json
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer.domain.data import Frames, LoadRequest
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.schemas import CONSTRAINTS, DETAILS, HOLDINGS, PORTFOLIOS, UNIVERSE
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.loaders import (
    CUSTODIAN,
    SECURITY_MASTER,
    SIGNALS,
    TRADES,
    Latency,
    ParametersParams,
    RunOrdersParams,
    ServiceParams,
    load_constraints,
    load_details,
    load_holdings,
    load_mandates,
    load_parameters,
    load_portfolios,
    load_run_orders,
    load_signals,
    load_trades,
    load_universe,
)
from tests.conftest import AS_OF, EXAMPLE_DATA
from tests.conftest import Frames as FrameFactory

INSTANT = ServiceParams(min_latency_s=0, max_latency_s=0)
"""Every shipped loader's simulated wait turned off; a test pays for the read and nothing else."""


def request(dataset: str, *portfolio_ids: str, root: Path = EXAMPLE_DATA) -> LoadRequest:
    ids = tuple(PortfolioId(portfolio_id) for portfolio_id in portfolio_ids)
    return LoadRequest(dataset=dataset, portfolio_ids=ids, as_of_date=AS_OF, data_root=root, run_id="test")


# --- what each service answers, and how it is typed on the way in ---


def test_the_book_of_record_returns_every_account_and_its_priority() -> None:
    portfolios = asyncio.run(load_portfolios(request("portfolios"), INSTANT))
    validate_frame(portfolios, PORTFOLIOS)
    assert len(portfolios) == 100
    assert portfolios["portfolio_id"].tolist()[:2] == ["P1", "P2"]
    assert portfolios["solve_order"].tolist()[:2] == [0, 1]


def test_the_custodian_answers_only_for_the_accounts_asked_for_with_money_as_decimal() -> None:
    holdings = asyncio.run(load_holdings(request("holdings", "P2"), INSTANT))
    validate_frame(holdings, HOLDINGS)
    assert holdings["portfolio_id"].unique().tolist() == ["P2"]
    assert holdings["avg_cost"].tolist() == [Decimal(100), Decimal(60)]
    assert str(holdings["acquired_on"].dtype) == "datetime64[ns, UTC]"


def test_the_security_master_answers_for_the_book_rather_than_per_account() -> None:
    universe = asyncio.run(load_universe(request("universe"), INSTANT))
    validate_frame(universe, UNIVERSE)
    assert universe["security_id"].tolist() == ["A", "B", "C"]
    assert universe["price"].tolist() == [Decimal(100), Decimal(50), Decimal(10)]
    assert universe["restricted"].tolist() == [False, False, False], "a boolean column is read strictly, not by truthiness"


def test_the_research_store_answers_for_the_book_with_one_signal_row_per_name() -> None:
    signals = asyncio.run(load_signals(request("signals"), INSTANT))
    validate_frame(signals, SIGNALS)
    assert signals["security_id"].tolist() == ["A", "B", "C"]
    assert signals["alpha"].tolist() == [0.03, -0.01, 0.05]
    assert signals["tcost_bps"].tolist() == [Decimal(5), Decimal(5), Decimal(20)], "a cost in basis points is money and arrives exact"


def test_the_account_master_takes_a_batch_of_ids_and_returns_a_row_each() -> None:
    details = load_details(request("details", "P1", "P2"), INSTANT)
    validate_frame(details, DETAILS)
    assert details["portfolio_id"].tolist() == ["P1", "P2"]
    assert details["name"].tolist() == ["Alpha Growth", "Beta Income"]
    assert details["max_weight"].tolist() == [Decimal("0.4"), Decimal("0.6")]


def test_compliance_returns_typed_rows_the_engine_reads_the_declaration_of() -> None:
    constraints = asyncio.run(load_constraints(request("constraints", "P1"), INSTANT))
    validate_frame(constraints, CONSTRAINTS)
    assert set(constraints.columns) == {"portfolio_id", "kind", "label", "params"}, "the schema declares only portfolio_id; the rest is the typed-row convention"
    band = constraints.loc[constraints["label"] == "sector_floor", "params"].iloc[0]
    assert json.loads(str(band)) == {"direction": ">=", "column": "sector", "bounds": {"TECH": "0.5", "HEALTH": "0"}}


def test_compliance_answers_the_mandate_for_the_accounts_asked_for(tmp_path: Path) -> None:
    (tmp_path / "mandates.csv").write_text("portfolio_id,sector\nP1,TECH\nP1,HEALTH\nP2,TECH\n")
    mandates = asyncio.run(load_mandates(request("mandates", "P1", root=tmp_path), INSTANT))
    assert mandates["portfolio_id"].unique().tolist() == ["P1"]
    assert mandates["sector"].tolist() == ["TECH", "HEALTH"]


def test_the_blotter_answers_the_trades_for_the_accounts_asked_for(tmp_path: Path) -> None:
    (tmp_path / "trades.csv").write_text("portfolio_id,security_id,side,traded_on\nP1,A,SELL,2026-08-19T00:00:00Z\nP2,B,BUY,2026-08-20T00:00:00Z\nP1,C,BUY,2026-06-01T00:00:00Z\n")
    trades = asyncio.run(load_trades(request("trades", "P1", root=tmp_path), INSTANT))
    assert trades["portfolio_id"].unique().tolist() == ["P1"] and trades["side"].tolist() == ["SELL", "BUY"]
    assert str(trades["traded_on"].dtype) == "datetime64[ns, UTC]", "a trade is dated with a zone, like a holding"


RUN_ORDERS = (
    "portfolio_id,security_id,side,quantity,reference_price,notional,target_weight,unrounded_quantity,spec_hash,run_id,as_of_date\n"
    "P2,B,SELL,200,50,10000,0.1,200.0,abc,run-1,2026-08-27T00:00:00Z\n"
    "P1,C,BUY,50,10,500,0.2,50.0,abc,run-1,2026-08-27T00:00:00Z\n"
    "P1,A,SELL,100,100,10000,0.0,100.0,abc,run-1,2026-08-27T00:00:00Z\n"
)
"""An orders file as the CSV sink writes it: three orders in two accounts, out of order."""


def test_a_previous_runs_orders_load_as_the_blotter_for_the_accounts_asked_for(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(RUN_ORDERS)
    trades = asyncio.run(load_run_orders(request("trades", "P1", root=tmp_path), RunOrdersParams(path="orders.csv", min_latency_s=0, max_latency_s=0)))
    validate_frame(trades, TRADES)
    assert list(trades.columns) == ["portfolio_id", "security_id", "side", "quantity", "traded_on"]
    assert trades[["portfolio_id", "security_id", "side", "quantity"]].to_dict("records") == [
        {"portfolio_id": "P1", "security_id": "A", "side": "SELL", "quantity": 100},
        {"portfolio_id": "P1", "security_id": "C", "side": "BUY", "quantity": 50},
    ], "P1's rows alone, sorted: an extra dataset hashes by row order"
    assert str(trades["traded_on"].dtype) == "datetime64[ns, UTC]" and trades["traded_on"].iloc[0] == pd.Timestamp("2026-08-27", tz="UTC"), "the run's as-of instant is when it traded"
    everyone = asyncio.run(load_run_orders(request("trades", root=tmp_path), RunOrdersParams(path="orders.csv", min_latency_s=0, max_latency_s=0)))
    assert everyone["portfolio_id"].tolist() == ["P1", "P1", "P2"], "a dataset that declares no dependency on the book gets every account"


def test_a_previous_runs_orders_load_as_the_volume_each_universe_security_lost(tmp_path: Path, frames: FrameFactory) -> None:
    (tmp_path / "orders.csv").write_text(RUN_ORDERS)
    universe = frames.three_security_universe().assign(security_id=pd.Series(["C", "A", "D"], dtype="string"))
    params = RunOrdersParams(path="orders.csv", emit="adv_consumed", min_latency_s=0, max_latency_s=0)
    consumed = asyncio.run(load_run_orders(LoadRequest(dataset="adv_consumed", portfolio_ids=(), as_of_date=AS_OF, data_root=tmp_path, run_id="test", inputs=Frames({"universe": universe})), params))
    assert consumed.to_dict("records") == [{"security_id": "A", "adv_consumed_quantity": 100}, {"security_id": "C", "adv_consumed_quantity": 50}, {"security_id": "D", "adv_consumed_quantity": 0}], (
        "every universe security, sorted, either side summed, zero where nothing traded; B is not in this universe"
    )
    assert str(consumed["adv_consumed_quantity"].dtype) == "Int64" and str(consumed["security_id"].dtype) == "string"
    with pytest.raises(ValueError, match=r'needs the universe to name every security; declare depends_on: \["universe"\]'):
        asyncio.run(load_run_orders(request("adv_consumed", root=tmp_path), params))


def test_a_previous_runs_orders_load_from_parquet_as_from_csv(tmp_path: Path, frames: FrameFactory) -> None:
    (tmp_path / "orders.csv").write_text(RUN_ORDERS)
    frames.orders(
        {"portfolio_id": "P2", "security_id": "B", "side": "SELL", "quantity": 200, "reference_price": Decimal(50), "notional": Decimal(10000), "as_of_date": pd.Timestamp("2026-08-27", tz="UTC")},
        {"portfolio_id": "P1", "security_id": "C", "side": "BUY", "quantity": 50, "reference_price": Decimal(10), "notional": Decimal(500), "as_of_date": pd.Timestamp("2026-08-27", tz="UTC")},
        {"portfolio_id": "P1", "security_id": "A", "side": "SELL", "quantity": 100, "reference_price": Decimal(100), "notional": Decimal(10000), "as_of_date": pd.Timestamp("2026-08-27", tz="UTC")},
    ).to_parquet(tmp_path / "orders.parquet", index=False)
    from_csv = asyncio.run(load_run_orders(request("trades", root=tmp_path), RunOrdersParams(path="orders.csv", min_latency_s=0, max_latency_s=0)))
    from_parquet = asyncio.run(load_run_orders(request("trades", root=tmp_path), RunOrdersParams(path="orders.parquet", min_latency_s=0, max_latency_s=0)))
    pd.testing.assert_frame_equal(from_parquet, from_csv)


def test_the_parameter_store_fetches_the_set_the_dataset_names() -> None:
    parameters = asyncio.run(load_parameters(request("global_parameters"), ParametersParams(min_latency_s=0, max_latency_s=0)))
    assert dict(zip(parameters["name"], parameters["value"], strict=True)) == {"risk_aversion": Decimal("2.5"), "max_names": Decimal(150)}
    named = ParametersParams(min_latency_s=0, max_latency_s=0, set_name="buy_universe_parameters")
    other = asyncio.run(load_parameters(request("whatever_the_desk_calls_it"), named))
    assert other["name"].tolist() == ["min_adv_quantity"]


def test_a_global_input_still_fetches_the_book_and_not_the_firm() -> None:
    constraints = asyncio.run(load_constraints(request("constraints", "P1", "P2"), INSTANT))
    assert constraints["portfolio_id"].unique().tolist() == ["P1", "P2"], "a global input receives every id, so the query is filtered to the accounts in the run"


def test_an_account_the_source_has_no_rows_for_comes_back_empty_rather_than_raising() -> None:
    details = load_details(request("details", "P404"), INSTANT)
    assert len(details) == 0, "a missing account is a coverage problem the engine fails alone, not a dataset failure"


def test_no_accounts_asked_for_returns_an_empty_frame_with_the_schema_columns() -> None:
    holdings = asyncio.run(load_holdings(request("holdings"), INSTANT))
    assert len(holdings) == 0
    assert list(holdings.columns) == [column.name for column in HOLDINGS.columns]


# --- fan-out and the simulated wait ---


def test_the_custodian_fans_out_over_its_batch_and_keeps_the_requests_order() -> None:
    holdings = asyncio.run(load_holdings(request("holdings", "P2", "P1"), INSTANT))
    assert [str(value) for value in holdings["portfolio_id"]] == ["P2", "P2", "P1", "P1"], "results come back in the order the request listed the accounts, not the order the calls finished"


def test_a_loader_waits_as_long_as_its_service_would() -> None:
    slow = ServiceParams(min_latency_s=0.05, max_latency_s=0.05)
    started = time.perf_counter()
    asyncio.run(load_universe(request("universe"), slow))
    assert time.perf_counter() - started >= 0.05


def test_the_wait_is_drawn_from_the_loaders_own_band_unless_the_config_overrides_it() -> None:
    assert ServiceParams().latency(CUSTODIAN) == CUSTODIAN
    assert ServiceParams(min_latency_s=4).latency(CUSTODIAN) == Latency(4, 4), "an override below the band's floor cannot outlive its ceiling"
    assert ServiceParams(max_latency_s=0).latency(SECURITY_MASTER) == Latency(0, 0)
    with pytest.raises(ValueError, match="min_latency_s must not exceed max_latency_s"):
        ServiceParams(min_latency_s=2, max_latency_s=1)


def test_the_wait_is_the_same_every_time_a_run_makes_the_same_call() -> None:
    band = Latency(0.5, 30.0)
    assert band.draw("run-1:universe:") == band.draw("run-1:universe:")
    assert band.draw("run-1:universe:") != band.draw("run-2:universe:"), "two runs of one config wait different amounts; one run reproduces its own"
    assert all(band.low_s <= band.draw(f"run-1:holdings:P{i}") <= band.high_s for i in range(20))
