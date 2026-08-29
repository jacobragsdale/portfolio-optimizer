"""Tier 3/5: the shipped file loaders declare dtypes at the boundary and reject malformed input."""

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.schemas import HOLDINGS, UNIVERSE
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.loaders import CsvParams, JsonConstraintsParams, ParquetParams, csv, json_constraints, parquet
from tests.conftest import AS_OF, EXAMPLE_DATA, Frames


def request(dataset: str, root: Path = EXAMPLE_DATA) -> LoadRequest:
    return LoadRequest(dataset=dataset, portfolio_ids=(PortfolioId("P1"),), as_of=AS_OF, data_root=root, run_id="test")


def test_csv_loads_an_engine_dataset_with_schema_dtypes() -> None:
    holdings = csv(request("holdings"), CsvParams(path="holdings.csv"))
    validate_frame(holdings, HOLDINGS)
    assert holdings["avg_cost"].iloc[0] == Decimal(100)
    assert str(holdings["acquired_on"].dtype) == "datetime64[ns, UTC]"


def test_csv_loads_an_extra_dataset_with_declared_decimal_columns() -> None:
    prices = csv(request("prices"), CsvParams(path="prices.csv", decimal_columns=("price",)))
    assert prices["price"].tolist() == [Decimal(100), Decimal(50), Decimal(10)]
    assert str(prices["security_id"].dtype) in ("str", "string", "object")


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
    loaded = parquet(request("prices", tmp_path), ParquetParams(path="prices.parquet", decimal_columns=("price",)))
    assert loaded["price"].iloc[0] == Decimal("0.1")


def test_json_constraints_loads_the_example_file() -> None:
    loaded = json_constraints(request("constraints"), JsonConstraintsParams(path="constraints.json"))
    assert set(loaded) == {"P1", "P2"}
    assert loaded["P1"]["max_adv_participation"] == "0.25"


def test_json_constraints_rejects_a_list(tmp_path: Path) -> None:
    (tmp_path / "constraints.json").write_text('[{"max_weight": "1"}]')
    with pytest.raises(ValueError, match="expected an object mapping portfolio ids"):
        json_constraints(request("constraints", tmp_path), JsonConstraintsParams(path="constraints.json"))
