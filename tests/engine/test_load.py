"""Tier 2: loading, assembly joins with declared cardinality, and per-portfolio isolation."""

from decimal import Decimal

import pandas as pd
import pytest

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.load import AssemblyError, LoadedDatasets, LoadError, assemble, load_datasets, slice_portfolio
from tests.conftest import AS_OF, EXAMPLE_DATA, Frames, resolved_example


@pytest.fixture
def example_loaded() -> LoadedDatasets:
    return load_datasets(resolved_example(), data_root=EXAMPLE_DATA, run_id="test")


@pytest.fixture
def example_config() -> RunConfig:
    return resolved_example().config


def test_example_data_loads_in_solve_order_with_audit_records(example_loaded: LoadedDatasets) -> None:
    assert example_loaded.portfolio_ids == ("P1", "P2")
    assert set(example_loaded.frames) == {"holdings", "universe", "details", "targets", "prices"}
    audit = {record.name: record for record in example_loaded.audits}
    assert audit["holdings"].rows == 3
    assert audit["constraints"].rows == 2
    assert len(audit["prices"].content_sha256) == 64
    assert audit["universe"].loader_qualname == "portfolio_optimizer.loaders:csv"


def test_assembly_joins_prices_into_universe_and_slices_each_portfolio(example_loaded: LoadedDatasets, example_config: RunConfig) -> None:
    assembled = assemble(example_loaded, example_config)
    assert assembled.universe["price"].tolist() == [Decimal(100), Decimal(50), Decimal(10)]
    p1 = slice_portfolio(assembled, PortfolioId("P1"))
    p2 = slice_portfolio(assembled, PortfolioId("P2"))
    assert isinstance(p1, PortfolioData)
    assert p1.holdings["security_id"].tolist() == ["A", "B"]
    assert p2.holdings["security_id"].tolist() == ["C"]
    assert p1.details.state == "NY"
    assert p2.details.name == "Beta Income"
    assert len(p1.targets) == 3
    assert p1.style.max_adv_participation == Decimal("0.25")
    assert p1.as_of == AS_OF


def _loaded_with(example_loaded: LoadedDatasets, **frames: object) -> LoadedDatasets:
    merged = {**example_loaded.frames, **frames}
    return LoadedDatasets(portfolio_ids=example_loaded.portfolio_ids, frames=merged, constraints=example_loaded.constraints, audits=example_loaded.audits)  # ty: ignore[invalid-argument-type]  # frames are DataFrames by construction in every caller


def test_duplicate_join_key_violates_declared_cardinality(example_loaded: LoadedDatasets, example_config: RunConfig) -> None:
    prices = example_loaded.frames["prices"]
    duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssemblyError, match="cardinality 'one_to_one' violated"):
        assemble(_loaded_with(example_loaded, prices=duplicated), example_config)


def test_unmatched_rows_are_named_when_all_must_match(example_loaded: LoadedDatasets, example_config: RunConfig, frames: Frames) -> None:
    universe = example_loaded.frames["universe"]
    extra = frames.universe({"security_id": "D", "price": Decimal(1)}).drop(columns=["price"])
    with_d = pd.concat([universe, extra], ignore_index=True)
    with pytest.raises(AssemblyError, match=r"1 row\(s\) of universe had no match in prices, e.g. \['D'\]"):
        assemble(_loaded_with(example_loaded, universe=with_d), example_config)


def test_join_refuses_to_overwrite_existing_columns(example_loaded: LoadedDatasets, example_config: RunConfig) -> None:
    prices = example_loaded.frames["prices"].assign(sector="X")
    with pytest.raises(AssemblyError, match="would overwrite columns \\['sector'\\]"):
        assemble(_loaded_with(example_loaded, prices=prices), example_config)


def test_schema_failures_after_assembly_name_the_dataset(example_loaded: LoadedDatasets, example_config: RunConfig) -> None:
    holdings = example_loaded.frames["holdings"].assign(quantity=example_loaded.frames["holdings"]["quantity"] * -1)
    with pytest.raises(LoadError, match="holdings: column 'quantity'"):
        assemble(_loaded_with(example_loaded, holdings=holdings), example_config)


def test_missing_constraints_for_a_portfolio_fail_loudly(example_loaded: LoadedDatasets, example_config: RunConfig) -> None:
    partial = LoadedDatasets(portfolio_ids=example_loaded.portfolio_ids, frames=example_loaded.frames, constraints={"P1": example_loaded.constraints["P1"]}, audits=example_loaded.audits)
    with pytest.raises(LoadError, match="constraints missing for portfolios \\['P2'\\]"):
        assemble(partial, example_config)


def test_loader_returning_the_wrong_type_is_rejected() -> None:
    config = resolved_example(datasets={**resolved_example().config.model_dump(mode="json")["datasets"], "holdings": {"loader": "tests.conftest:lying_loader"}})
    with pytest.raises(LoadError, match="returned dict, expected DataFrame"):
        load_datasets(config, data_root=EXAMPLE_DATA, run_id="test")


def test_portfolio_list_must_satisfy_its_schema() -> None:
    config = resolved_example(portfolios={"name": "csv", "params": {"path": "prices.csv"}})
    with pytest.raises(LoadError, match="portfolios: portfolios"):
        load_datasets(config, data_root=EXAMPLE_DATA, run_id="test")
