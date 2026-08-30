"""Tier 3/5: the shipped file loaders declare dtypes at the boundary and reject malformed input."""

import asyncio
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.schemas import HOLDINGS, UNIVERSE
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.loaders import CsvParams, CsvPerPortfolioParams, JsonConstraintsParams, ParquetParams, csv, csv_per_portfolio, json_constraints, parquet
from portfolio_optimizer.ratelimit import RateLimit, RateLimiter
from tests.conftest import AS_OF, EXAMPLE_DATA, Frames


def request(dataset: str, root: Path = EXAMPLE_DATA) -> LoadRequest:
    return LoadRequest(dataset=dataset, portfolio_ids=(PortfolioId("P1"),), as_of=AS_OF, data_root=root, run_id="test")


def test_csv_loads_an_engine_dataset_with_schema_dtypes() -> None:
    holdings = csv(request("holdings"), CsvParams(path="holdings.csv"))
    validate_frame(holdings, HOLDINGS)
    assert holdings["avg_cost"].iloc[0] == Decimal(100)
    assert str(holdings["acquired_on"].dtype) == "datetime64[ns, UTC]"


def test_csv_loads_an_extra_dataset_with_the_kinds_its_dtypes_declare() -> None:
    prices = csv(request("prices"), CsvParams(path="prices.csv", dtypes={"security_id": "string", "price": "decimal"}))
    assert prices["price"].tolist() == [Decimal(100), Decimal(50), Decimal(10)]
    assert str(prices["security_id"].dtype) in ("str", "string", "object")


def test_csv_types_an_extra_dataset_timestamp_from_its_declared_kind(tmp_path: Path) -> None:
    (tmp_path / "signals.csv").write_text("security_id,published_at,live\nA,2026-08-28T00:00:00Z,true\n")
    signals = csv(request("signals", tmp_path), CsvParams(path="signals.csv", dtypes={"security_id": "string", "published_at": "datetime_utc", "live": "bool"}))
    assert str(signals["published_at"].dtype) == "datetime64[ns, UTC]"
    assert str(signals["live"].dtype) == "bool"


def test_csv_leaves_a_column_no_kind_is_declared_for_to_pandas(tmp_path: Path) -> None:
    (tmp_path / "signals.csv").write_text("security_id,rank\nA,3\n")
    signals = csv(request("signals", tmp_path), CsvParams(path="signals.csv", dtypes={"security_id": "string"}))
    assert signals["rank"].iloc[0] == 3


def test_csv_reads_booleans_strictly(tmp_path: Path) -> None:
    (tmp_path / "universe.csv").write_text("security_id,price,sector,adv_shares,lot_size,restricted\nA,10,TECH,1,1,maybe\n")
    with pytest.raises(ValueError, match="maybe"):
        csv(request("universe", tmp_path), CsvParams(path="universe.csv"))


def test_parquet_round_trips_decimal_columns(tmp_path: Path, frames: Frames) -> None:
    universe = frames.universe({"security_id": "A", "price": Decimal("12.34")})
    universe.to_parquet(tmp_path / "universe.parquet", index=False)
    loaded = parquet(request("universe", tmp_path), ParquetParams(path="universe.parquet"))
    validate_frame(loaded, UNIVERSE)
    assert loaded["price"].iloc[0] == Decimal("12.34")


def test_parquet_extra_dataset_converts_float_columns_to_decimal(tmp_path: Path) -> None:
    pd.DataFrame({"security_id": ["A"], "price": [0.1]}).to_parquet(tmp_path / "prices.parquet", index=False)
    loaded = parquet(request("prices", tmp_path), ParquetParams(path="prices.parquet", dtypes={"price": "decimal"}))
    assert loaded["price"].iloc[0] == Decimal("0.1")


def test_json_constraints_loads_the_example_file() -> None:
    loaded = json_constraints(request("constraints"), JsonConstraintsParams(path="constraints.json"))
    assert set(loaded) == {"P1", "P2"}
    assert loaded["P1"]["max_adv_participation"] == "0.25"


def test_json_constraints_rejects_a_list(tmp_path: Path) -> None:
    (tmp_path / "constraints.json").write_text('[{"max_weight": "1"}]')
    with pytest.raises(ValueError, match="expected an object mapping portfolio ids"):
        json_constraints(request("constraints", tmp_path), JsonConstraintsParams(path="constraints.json"))


def _per_portfolio_files(root: Path) -> None:
    (root / "holdings").mkdir()
    (root / "holdings" / "P1.csv").write_text("portfolio_id,security_id,quantity,avg_cost,acquired_on\nP1,A,100,90.5,2024-01-15T00:00:00Z\n")
    (root / "holdings" / "P2.csv").write_text("portfolio_id,security_id,quantity,avg_cost,acquired_on\nP2,B,200,40,2025-06-01T00:00:00Z\nP2,C,300,9.25,2025-06-01T00:00:00Z\n")


def test_csv_per_portfolio_fans_out_under_the_rate_limit_and_keeps_portfolio_order(tmp_path: Path) -> None:
    _per_portfolio_files(tmp_path)

    async def scenario() -> tuple[pd.DataFrame, RateLimiter]:
        limiter = RateLimiter(RateLimit(max_in_flight=1), name="files")
        request = LoadRequest(dataset="holdings", portfolio_ids=(PortfolioId("P2"), PortfolioId("P1")), as_of=AS_OF, data_root=tmp_path, run_id="test", rate_limiter=limiter)
        return await csv_per_portfolio(request, CsvPerPortfolioParams(directory="holdings")), limiter

    holdings, limiter = asyncio.run(scenario())
    validate_frame(holdings, HOLDINGS)
    assert holdings["portfolio_id"].tolist() == ["P2", "P2", "P1"]
    assert holdings["avg_cost"].tolist() == [Decimal(40), Decimal("9.25"), Decimal("90.5")]
    assert limiter.acquired == 2


def test_csv_per_portfolio_with_no_portfolios_returns_an_empty_frame_with_the_schema_columns(tmp_path: Path) -> None:
    request = LoadRequest(dataset="holdings", portfolio_ids=(), as_of=AS_OF, data_root=tmp_path, run_id="test")
    holdings = asyncio.run(csv_per_portfolio(request, CsvPerPortfolioParams(directory="holdings")))
    assert len(holdings) == 0
    assert list(holdings.columns) == [column.name for column in HOLDINGS.columns]


def test_csv_per_portfolio_names_the_portfolio_whose_file_is_missing(tmp_path: Path) -> None:
    _per_portfolio_files(tmp_path)
    request = LoadRequest(dataset="holdings", portfolio_ids=(PortfolioId("P1"), PortfolioId("P9")), as_of=AS_OF, data_root=tmp_path, run_id="test")
    with pytest.raises(ExceptionGroup) as info:
        asyncio.run(csv_per_portfolio(request, CsvPerPortfolioParams(directory="holdings")))
    assert info.group_contains(FileNotFoundError, match="P9.csv")
