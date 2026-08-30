"""Tier 2: loading, assembly steps with declared cardinality, extras that flow into bundles, and per-portfolio isolation."""

import asyncio
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDataError
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.load import AssemblyError, LoadedDatasets, LoadError, assemble, load_datasets, load_datasets_async, slice_portfolio
from tests.conftest import AS_OF, EXAMPLE_DATA, Frames, resolved_example

JOIN_PRICES_PARAMS: dict[str, object] = {"into": "universe", "source": "prices", "on": ["security_id"], "cardinality": "one_to_one", "require_all_matched": True}
JOIN_PRICES: dict[str, object] = {"name": "join", "params": JOIN_PRICES_PARAMS}


@pytest.fixture
def example_loaded() -> LoadedDatasets:
    return load_datasets(resolved_example(), data_root=EXAMPLE_DATA, run_id="test")


@pytest.fixture
def example_resolved() -> ResolvedConfig:
    return resolved_example()


def _with_assembly(*steps: object) -> ResolvedConfig:
    return resolved_example(assembly=list(steps))


def _loaded_with(example_loaded: LoadedDatasets, **frames: object) -> LoadedDatasets:
    merged = {**example_loaded.frames, **frames}
    return LoadedDatasets(portfolio_ids=example_loaded.portfolio_ids, solve_orders=example_loaded.solve_orders, frames=merged, constraints=example_loaded.constraints, audits=example_loaded.audits)  # ty: ignore[invalid-argument-type]  # frames are DataFrames by construction in every caller


def test_example_data_loads_in_solve_order_with_audit_records(example_loaded: LoadedDatasets) -> None:
    assert example_loaded.portfolio_ids == ("P1", "P2")
    assert example_loaded.solve_orders == {"P1": 0, "P2": 1}
    assert set(example_loaded.frames) == {"holdings", "universe", "details", "targets", "prices"}
    audit = {record.name: record for record in example_loaded.audits}
    assert audit["holdings"].rows == 4
    assert audit["constraints"].rows == 2
    assert len(audit["prices"].content_sha256) == 64
    assert audit["universe"].loader_qualname == "portfolio_optimizer.loaders:csv"


def test_assembly_joins_prices_into_universe_drops_them_and_slices_each_portfolio(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    assembled = assemble(example_loaded, example_resolved, run_id="test")
    assert assembled.universe["price"].tolist() == [Decimal(100), Decimal(50), Decimal(10)]
    assert assembled.extras == {}
    assert [audit.qualname for audit in assembled.audits] == ["portfolio_optimizer.assembly:join", "portfolio_optimizer.assembly:drop"]
    assert assembled.audits[0].columns_added == {"universe": ("price",)}
    assert assembled.audits[1].rows_in["prices"] == 3
    assert assembled.audits[1].rows_out == {"holdings": 4, "universe": 3, "details": 2, "targets": 3}
    p1 = slice_portfolio(assembled, PortfolioId("P1"))
    p2 = slice_portfolio(assembled, PortfolioId("P2"))
    assert isinstance(p1, PortfolioData)
    assert p1.holdings["security_id"].tolist() == ["A", "B"]
    assert p2.holdings["security_id"].tolist() == ["A", "B"]
    assert p1.details.state == "NY"
    assert p2.details.name == "Beta Income"
    assert len(p1.targets) == 3
    assert p1.style.max_adv_participation == Decimal("0.25")
    assert p1.as_of == AS_OF


def test_a_custom_step_attaches_analytics_to_holdings_and_universe_and_is_audited(example_loaded: LoadedDatasets) -> None:
    assembled = assemble(example_loaded, _with_assembly(JOIN_PRICES, "tests.conftest:score_by_price", {"name": "drop", "params": {"datasets": ["prices"]}}), run_id="test")
    assert assembled.audits[1].columns_added == {"holdings": ("score",), "universe": ("score",)}
    assert len(assembled.audits[1].source_sha256) == 64
    frame = slice_portfolio(assembled, PortfolioId("P1")).optimizer_frame()
    assert frame["score"].tolist() == [100.0, 50.0, 100.0, 50.0, 10.0]
    assert str(frame["score"].dtype) == "Float64"


def test_extras_not_dropped_are_carried_into_every_bundle_reduced_to_its_portfolio(example_loaded: LoadedDatasets) -> None:
    notes = pd.DataFrame({"portfolio_id": pd.Series(["P1", "P2", "P2"], dtype="string"), "note": pd.Series(["a", "b", "c"], dtype="string")})
    assembled = assemble(_loaded_with(example_loaded, notes=notes), _with_assembly(JOIN_PRICES), run_id="test")
    assert set(assembled.extras) == {"prices", "notes"}
    p1 = slice_portfolio(assembled, PortfolioId("P1"))
    p2 = slice_portfolio(assembled, PortfolioId("P2"))
    assert p1.extras["notes"]["note"].tolist() == ["a"]
    assert p2.extras["notes"]["note"].tolist() == ["b", "c"]
    assert len(p1.extras["prices"]) == len(p2.extras["prices"]) == 3  # no portfolio_id column: passed whole


def test_duplicate_join_key_violates_declared_cardinality(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    prices = example_loaded.frames["prices"]
    duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssemblyError, match=r"assembly\[0\] portfolio_optimizer.assembly:join: cardinality 'one_to_one' violated"):
        assemble(_loaded_with(example_loaded, prices=duplicated), example_resolved, run_id="test")


def test_unmatched_rows_are_named_when_all_must_match(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig, frames: Frames) -> None:
    universe = example_loaded.frames["universe"]
    extra = frames.universe({"security_id": "D", "price": Decimal(1)}).drop(columns=["price"])
    with_d = pd.concat([universe, extra], ignore_index=True)
    with pytest.raises(AssemblyError, match=r"1 row\(s\) of universe had no match in prices, e.g. \['D'\]"):
        assemble(_loaded_with(example_loaded, universe=with_d), example_resolved, run_id="test")


def test_join_refuses_to_overwrite_existing_columns_unless_told_to(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    prices = example_loaded.frames["prices"].assign(sector=pd.Series(["X", "X", "X"], dtype="string"))
    with pytest.raises(AssemblyError, match=r"would overwrite columns \['sector'\] already present in universe"):
        assemble(_loaded_with(example_loaded, prices=prices), example_resolved, run_id="test")
    overwriting = _with_assembly({"name": "join", "params": {**JOIN_PRICES_PARAMS, "overwrite": True}})
    assert assemble(_loaded_with(example_loaded, prices=prices), overwriting, run_id="test").universe["sector"].tolist() == ["X", "X", "X"]


def test_a_step_naming_an_unknown_dataset_is_told_what_exists(example_loaded: LoadedDatasets) -> None:
    resolved = _with_assembly({"name": "join", "params": {**JOIN_PRICES_PARAMS, "source": "sectors"}})
    with pytest.raises(AssemblyError, match=r"assembly\[0\] portfolio_optimizer.assembly:join: no dataset 'sectors'; available: \['details', 'holdings', 'prices', 'targets', 'universe'\]"):
        assemble(example_loaded, resolved, run_id="test")


def test_a_step_that_refuses_its_input_rejects_the_run_with_its_own_message(example_loaded: LoadedDatasets) -> None:
    with pytest.raises(AssemblyError, match=r"assembly\[0\] tests.conftest:refuse_assembly: vendor scores are stale"):
        assemble(example_loaded, _with_assembly("tests.conftest:refuse_assembly"), run_id="test")


def test_a_step_returning_the_wrong_type_is_rejected(example_loaded: LoadedDatasets) -> None:
    with pytest.raises(AssemblyError, match="returned DataFrame, expected Frames"):
        assemble(example_loaded, _with_assembly("tests.conftest:lying_assembly_step"), run_id="test")


def test_dropping_an_engine_frame_is_caught_after_the_last_step(example_loaded: LoadedDatasets) -> None:
    with pytest.raises(LoadError, match=r"required datasets are missing \['universe'\]"):
        assemble(example_loaded, _with_assembly(JOIN_PRICES, {"name": "drop", "params": {"datasets": ["universe"]}}), run_id="test")


def test_schema_failures_after_assembly_name_the_dataset(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    holdings = example_loaded.frames["holdings"].assign(quantity=example_loaded.frames["holdings"]["quantity"] * -1)
    with pytest.raises(LoadError, match="holdings: column 'quantity'"):
        assemble(_loaded_with(example_loaded, holdings=holdings), example_resolved, run_id="test")


def test_analytics_dtype_conflicts_between_holdings_and_universe_fail_at_slice(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    holdings = example_loaded.frames["holdings"].assign(score=pd.Series([1.0, 2.0, 3.0], dtype="Float64"))
    universe = example_loaded.frames["universe"].assign(score=pd.Series([1.0, 2.0, 3.0], dtype="float64"))
    assembled = assemble(_loaded_with(example_loaded, holdings=holdings, universe=universe), example_resolved, run_id="test")
    with pytest.raises(PortfolioDataError, match="holdings and universe disagree on column 'score'"):
        slice_portfolio(assembled, PortfolioId("P1"))


def test_missing_constraints_for_a_portfolio_fail_loudly(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    partial = LoadedDatasets(
        portfolio_ids=example_loaded.portfolio_ids,
        solve_orders=example_loaded.solve_orders,
        frames=example_loaded.frames,
        constraints={"P1": example_loaded.constraints["P1"]},
        audits=example_loaded.audits,
    )
    with pytest.raises(LoadError, match="constraints missing for portfolios \\['P2'\\]"):
        assemble(partial, example_resolved, run_id="test")


def test_loader_returning_the_wrong_type_is_rejected() -> None:
    config = resolved_example(datasets={**resolved_example().config.model_dump(mode="json")["datasets"], "holdings": {"loader": "tests.conftest:lying_loader"}})
    with pytest.raises(LoadError, match="returned dict, expected DataFrame"):
        load_datasets(config, data_root=EXAMPLE_DATA, run_id="test")


def test_portfolio_list_must_satisfy_its_schema() -> None:
    config = resolved_example(portfolios={"name": "csv", "params": {"path": "prices.csv"}})
    with pytest.raises(LoadError, match="portfolios: portfolios"):
        load_datasets(config, data_root=EXAMPLE_DATA, run_id="test")


# --- the load stage runs loaders concurrently and hands each its pool ---


def _with_extra_datasets(**extra: object) -> ResolvedConfig:
    body = resolved_example().config.model_dump(mode="json", by_alias=True, exclude_none=True)
    return resolved_example(datasets={**body["datasets"], **extra}, rate_limits=body.get("rate_limits", {}) | {"vendor": {"requests_per_second": 50, "max_in_flight": 2}})


@pytest.mark.parametrize("loader", ["barrier_loader", "async_barrier_loader"], ids=["sync loaders in threads", "async loaders on the loop"])
def test_dataset_loaders_run_at_the_same_time(loader: str) -> None:
    resolved = _with_extra_datasets(left={"loader": f"tests.conftest:{loader}"}, right={"loader": f"tests.conftest:{loader}"})
    loaded = load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test")  # would deadlock or time out if left and right ran one after another
    assert loaded.frames["left"]["portfolio_id"].tolist() == ["P1", "P2"]
    assert loaded.frames["right"]["portfolio_id"].tolist() == ["P1", "P2"]


def test_each_dataset_receives_the_pool_its_config_names() -> None:
    resolved = _with_extra_datasets(
        left={"loader": "tests.conftest:pool_reporting_loader", "rate_limit": "vendor"},
        right={"loader": "tests.conftest:async_pool_reporting_loader", "rate_limit": "vendor"},
        free={"loader": "tests.conftest:pool_reporting_loader"},
    )
    loaded = load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test")
    assert loaded.frames["left"].iloc[0].tolist() == ["vendor", True]
    assert loaded.frames["right"].iloc[0].tolist() == ["vendor", True]
    assert loaded.frames["free"].iloc[0].tolist() == ["unlimited", False]


def test_every_failed_dataset_is_reported_together_as_rejected_input() -> None:
    resolved = _with_extra_datasets(left={"loader": "tests.conftest:invalid_input_loader"}, right={"loader": "tests.conftest:invalid_input_loader"})
    with pytest.raises(LoadError, match=r"left: left: no rows as of 2026-08-28; right: right: no rows"):
        load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test")


def test_an_unreachable_backend_keeps_its_exception_type_so_the_exit_code_is_infrastructure() -> None:
    resolved = _with_extra_datasets(left={"loader": "tests.conftest:invalid_input_loader"}, right={"loader": "tests.conftest:unreachable_loader"})
    with pytest.raises(ConnectionError, match="right: connection refused"):
        load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test")


def test_audits_record_how_long_each_dataset_took(example_loaded: LoadedDatasets) -> None:
    assert all(record.load_time_s >= 0.0 for record in example_loaded.audits)
    assert {record.name for record in example_loaded.audits} == {"portfolios", "holdings", "universe", "details", "constraints", "targets", "prices"}


def test_the_sync_entry_point_refuses_to_nest_inside_a_running_loop() -> None:
    async def inside() -> None:
        load_datasets(resolved_example(), data_root=EXAMPLE_DATA, run_id="test")

    with pytest.raises(RuntimeError, match="await load_datasets_async instead"):
        asyncio.run(inside())


def test_the_async_entry_point_can_be_awaited_directly() -> None:
    loaded = asyncio.run(load_datasets_async(resolved_example(), data_root=EXAMPLE_DATA, run_id="test"))
    assert loaded.portfolio_ids == ("P1", "P2")


def test_an_inline_bound_is_private_to_its_input_while_a_named_pool_is_shared() -> None:
    resolved = _with_extra_datasets(
        left={"loader": "tests.conftest:pool_reporting_loader", "rate_limit": "vendor"},
        right={"loader": "tests.conftest:async_pool_reporting_loader", "rate_limit": "vendor"},
        slow={"loader": "tests.conftest:pool_reporting_loader", "rate_limit": {"max_in_flight": 1}},
        fast={"loader": "tests.conftest:async_pool_reporting_loader", "rate_limit": {"requests_per_second": 100, "max_in_flight": 32}},
    )
    loaded = load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test")
    assert loaded.frames["left"].iloc[0].tolist() == ["vendor", True]
    assert loaded.frames["right"].iloc[0].tolist() == ["vendor", True]
    assert loaded.frames["slow"].iloc[0].tolist() == ["slow", True]
    assert loaded.frames["fast"].iloc[0].tolist() == ["fast", True]


def test_the_portfolio_list_input_can_be_bounded_too() -> None:
    resolved = resolved_example(portfolios={"loader": "tests.conftest:limiter_naming_portfolios_loader", "rate_limit": {"max_in_flight": 1}})
    assert load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test").portfolio_ids == ("portfolios",)
    resolved = resolved_example(portfolios={"loader": "tests.conftest:limiter_naming_portfolios_loader", "rate_limit": "shared"}, rate_limits={"shared": {"max_in_flight": 1}})
    assert load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="test").portfolio_ids == ("shared",)
