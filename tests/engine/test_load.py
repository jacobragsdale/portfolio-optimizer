"""Tier 2: loading, assembly steps with declared cardinality, extras that flow into bundles, and per-portfolio isolation."""

import asyncio
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDataError
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.load import AssemblyError, LoadedDatasets, LoadError, assemble, load_datasets, load_datasets_async, slice_portfolio
from tests import steps
from tests.conftest import AS_OF, Frames, example_datasets, instant, resolved_example, two_account_book

JOIN_SCORES_PARAMS: dict[str, object] = {"into": "universe", "source": "scores", "on": ["security_id"], "cardinality": "one_to_one", "require_all_matched": True}
JOIN_SCORES: dict[str, object] = {"name": "join", "params": JOIN_SCORES_PARAMS}
P2 = PortfolioId("P2")


def scores_frame(*values: float) -> pd.DataFrame:
    """An extra dataset with one analytics column, the shape an assembly `join` brings into the universe."""
    return pd.DataFrame({"security_id": pd.Series(["A", "B", "C"], dtype="string"), "score": pd.Series(values, dtype="Float64")})


SCORES = scores_frame(1.0, 2.0, 3.0)


@pytest.fixture
def example_loaded(book: Path) -> LoadedDatasets:
    return load_datasets(resolved_example(), data_root=book, run_id="test", as_of_date=AS_OF)


@pytest.fixture
def example_resolved() -> ResolvedConfig:
    return resolved_example()


def _with_assembly(*steps: object) -> ResolvedConfig:
    return resolved_example(assembly=list(steps))


GLOBAL_HOLDINGS: dict[str, object] = instant("load_holdings") | {"depends_on": ["portfolios"]}
"""The same fan-out loader at `global` scope — one call for every id, the loader owning its partition: for a test about assembly, which per-account datasets never reach. Global, so the ids must be asked for in `depends_on`."""


def _loaded_with(example_loaded: LoadedDatasets, **frames: object) -> LoadedDatasets:
    """The loaded datasets with each named frame replaced in the mapping it came from, global or per-account."""
    sharded = {name: frame for name, frame in frames.items() if name in example_loaded.per_portfolio}
    return replace(  # frames are DataFrames by construction in every caller
        example_loaded, frames={**example_loaded.frames, **{name: frame for name, frame in frames.items() if name not in sharded}}, per_portfolio={**example_loaded.per_portfolio, **sharded}
    )


def test_example_data_loads_in_solve_order_with_audit_records(example_loaded: LoadedDatasets) -> None:
    assert example_loaded.portfolio_ids == ("P1", "P2")
    assert example_loaded.solve_orders == {"P1": 0, "P2": 1}
    assert set(example_loaded.frames) == {"universe", "global_parameters", "buy_universe_parameters"}
    assert set(example_loaded.per_portfolio) == {"details", "holdings"}, "the example loads both per account, and assembly never sees either"
    audit = {record.name: record for record in example_loaded.audits}
    assert (audit["holdings"].rows, audit["holdings"].batches) == (4, 2), "batch_size 1 is one call per portfolio"
    assert (audit["details"].rows, audit["details"].batches) == (2, 1), "batch_size 25 puts this two-account book in a single call"
    assert len(audit["universe"].content_sha256) == 64
    assert audit["universe"].loader_qualname == "portfolio_optimizer.loaders:load_universe"
    assert audit["holdings"].depends_on == ("portfolios",), "per_portfolio implies the dependency without declaring it"
    assert audit["universe"].depends_on == (), "nothing declared, so the security master started at once"


def test_the_example_assembles_without_a_step_and_slices_each_portfolio(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    assembled = assemble(example_loaded, example_resolved, run_id="test", as_of_date=AS_OF)
    assert assembled.universe["price"].tolist() == [Decimal(100), Decimal(50), Decimal(10)], "the universe carries its own price; nothing has to supply it"
    assert set(assembled.extras) == {"global_parameters", "buy_universe_parameters"}, "the example's two parameter frames are extras and survive to every bundle"
    assert assembled.audits == (), "the example configures no assembly step"
    assert (len(assembled.holdings), len(assembled.details)) == (4, 2)
    p1 = slice_portfolio(assembled, PortfolioId("P1"))
    p2 = slice_portfolio(assembled, PortfolioId("P2"))
    assert isinstance(p1, PortfolioData)
    assert p1.holdings["security_id"].tolist() == ["A", "B"]
    assert p2.holdings["security_id"].tolist() == ["A", "B"]
    assert p1.details.state == "NY"
    assert p2.details.name == "Beta Income"
    assert p1.details.max_adv_participation == Decimal("0.25")
    assert len(p1.universe) == len(p2.universe) == 3, "the universe is book-wide and passed whole to both"
    assert p1.as_of_date == AS_OF


def test_a_join_brings_a_column_across_records_what_it_added_and_a_drop_frees_the_source(example_loaded: LoadedDatasets) -> None:
    steps = _with_assembly(JOIN_SCORES, {"name": "drop", "params": {"datasets": ["scores"]}})
    assembled = assemble(_loaded_with(example_loaded, scores=SCORES), steps, run_id="test", as_of_date=AS_OF)
    assert assembled.universe["score"].tolist() == [1.0, 2.0, 3.0]
    assert "scores" not in assembled.extras, "a dataset that has done its job is dropped rather than carried into every bundle"
    assert [audit.qualname for audit in assembled.audits] == ["portfolio_optimizer.assembly:join", "portfolio_optimizer.assembly:drop"]
    assert assembled.audits[0].columns_added == {"universe": ("score",)}
    assert assembled.audits[1].rows_in["scores"] == 3
    assert assembled.audits[1].rows_out == {"universe": 3, "global_parameters": 2, "buy_universe_parameters": 1}, "the per-account holdings and details are merged back in after the steps have run"


def test_a_custom_step_attaches_analytics_to_holdings_and_universe_and_is_audited(book: Path) -> None:
    resolved = resolved_example(datasets=example_datasets(holdings=GLOBAL_HOLDINGS), assembly=["tests.steps:score_by_price"])
    loaded = load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)
    assembled = assemble(loaded, resolved, run_id="test", as_of_date=AS_OF)
    assert assembled.audits[0].columns_added == {"holdings": ("score",), "universe": ("score",)}
    assert len(assembled.audits[0].source_sha256) == 64
    frame = slice_portfolio(assembled, PortfolioId("P1")).optimizer_frame()
    assert frame["score"].tolist() == [100.0, 50.0, 100.0, 50.0, 10.0]
    assert str(frame["score"].dtype) == "Float64"


def test_extras_not_dropped_are_carried_into_every_bundle_reduced_to_its_portfolio(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    notes = pd.DataFrame({"portfolio_id": pd.Series(["P1", "P2", "P2"], dtype="string"), "note": pd.Series(["a", "b", "c"], dtype="string")})
    assembled = assemble(_loaded_with(example_loaded, notes=notes, scores=SCORES), example_resolved, run_id="test", as_of_date=AS_OF)
    assert set(assembled.extras) == {"scores", "notes", "global_parameters", "buy_universe_parameters"}
    p1 = slice_portfolio(assembled, PortfolioId("P1"))
    p2 = slice_portfolio(assembled, PortfolioId("P2"))
    assert p1.extras["notes"]["note"].tolist() == ["a"]
    assert p2.extras["notes"]["note"].tolist() == ["b", "c"]
    assert len(p1.extras["scores"]) == len(p2.extras["scores"]) == 3  # no portfolio_id column: passed whole


def test_duplicate_join_key_violates_declared_cardinality(example_loaded: LoadedDatasets) -> None:
    duplicated = pd.concat([SCORES, SCORES.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssemblyError, match=r"assembly\[0\] portfolio_optimizer.assembly:join: cardinality 'one_to_one' violated"):
        assemble(_loaded_with(example_loaded, scores=duplicated), _with_assembly(JOIN_SCORES), run_id="test", as_of_date=AS_OF)


def test_unmatched_rows_are_named_when_all_must_match(example_loaded: LoadedDatasets, frames: Frames) -> None:
    universe = example_loaded.frames["universe"]
    with_d = pd.concat([universe, frames.universe({"security_id": "D"})], ignore_index=True)
    with pytest.raises(AssemblyError, match=r"1 row\(s\) of universe had no match in scores, e.g. \['D'\]"):
        assemble(_loaded_with(example_loaded, universe=with_d, scores=SCORES), _with_assembly(JOIN_SCORES), run_id="test", as_of_date=AS_OF)


def test_join_refuses_to_overwrite_existing_columns_unless_told_to(example_loaded: LoadedDatasets) -> None:
    scores = SCORES.assign(sector=pd.Series(["X", "X", "X"], dtype="string"))
    with pytest.raises(AssemblyError, match=r"would overwrite columns \['sector'\] already present in universe"):
        assemble(_loaded_with(example_loaded, scores=scores), _with_assembly(JOIN_SCORES), run_id="test", as_of_date=AS_OF)
    overwriting = _with_assembly({"name": "join", "params": {**JOIN_SCORES_PARAMS, "overwrite": True}})
    assert assemble(_loaded_with(example_loaded, scores=scores), overwriting, run_id="test", as_of_date=AS_OF).universe["sector"].tolist() == ["X", "X", "X"]


def test_a_step_naming_an_unknown_dataset_is_told_what_exists(example_loaded: LoadedDatasets) -> None:
    resolved = _with_assembly({"name": "join", "params": {**JOIN_SCORES_PARAMS, "source": "sectors"}})
    with pytest.raises(AssemblyError, match=r"assembly\[0\] portfolio_optimizer.assembly:join: no dataset 'sectors'; available: \['buy_universe_parameters', 'global_parameters', 'universe'\]"):
        assemble(example_loaded, resolved, run_id="test", as_of_date=AS_OF)


def test_a_step_that_refuses_its_input_rejects_the_run_with_its_own_message(example_loaded: LoadedDatasets) -> None:
    with pytest.raises(AssemblyError, match=r"assembly\[0\] tests.steps:refuse_assembly: vendor scores are stale"):
        assemble(example_loaded, _with_assembly("tests.steps:refuse_assembly"), run_id="test", as_of_date=AS_OF)


def test_a_step_returning_the_wrong_type_is_rejected(example_loaded: LoadedDatasets) -> None:
    with pytest.raises(AssemblyError, match="returned DataFrame, expected Frames"):
        assemble(example_loaded, _with_assembly("tests.steps:lying_assembly_step"), run_id="test", as_of_date=AS_OF)


def test_dropping_an_engine_frame_is_caught_after_the_last_step(example_loaded: LoadedDatasets) -> None:
    with pytest.raises(LoadError, match=r"required datasets are missing \['universe'\]"):
        assemble(example_loaded, _with_assembly({"name": "drop", "params": {"datasets": ["universe"]}}), run_id="test", as_of_date=AS_OF)


def test_schema_failures_after_assembly_name_the_dataset(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    holdings = example_loaded.per_portfolio["holdings"].assign(quantity=example_loaded.per_portfolio["holdings"]["quantity"] * -1)
    with pytest.raises(LoadError, match="holdings: column 'quantity'"):
        assemble(_loaded_with(example_loaded, holdings=holdings), example_resolved, run_id="test", as_of_date=AS_OF)


def test_analytics_dtype_conflicts_between_holdings_and_universe_fail_at_slice(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    holdings = example_loaded.per_portfolio["holdings"].assign(score=pd.Series([1.0, 2.0, 3.0], dtype="Float64"))
    universe = example_loaded.frames["universe"].assign(score=pd.Series([1.0, 2.0, 3.0], dtype="float64"))
    assembled = assemble(_loaded_with(example_loaded, holdings=holdings, universe=universe), example_resolved, run_id="test", as_of_date=AS_OF)
    with pytest.raises(PortfolioDataError, match="holdings and universe disagree on column 'score'"):
        slice_portfolio(assembled, PortfolioId("P1"))


def test_a_portfolio_without_details_is_rejected_alone(example_loaded: LoadedDatasets, example_resolved: ResolvedConfig) -> None:
    details = example_loaded.per_portfolio["details"]
    partial = replace(example_loaded, per_portfolio={**example_loaded.per_portfolio, "details": details[details["portfolio_id"] == "P1"]})
    assembled = assemble(partial, example_resolved, run_id="test", as_of_date=AS_OF)
    assert set(assembled.rejected) == {P2}, "a portfolio the inputs do not cover fails alone; P1 still has everything it needs"
    assert (assembled.rejected[P2].stage, assembled.rejected[P2].error_type) == ("load", "MissingInput")
    assert "no details for this portfolio" in assembled.rejected[P2].message
    assert assembled.portfolio_ids == ("P1", "P2"), "the book is unchanged; the rejection travels beside it"


def test_loader_returning_the_wrong_type_is_rejected(book: Path) -> None:
    config = resolved_example(datasets={**resolved_example().config.model_dump(mode="json")["datasets"], "holdings": {"loader": "tests.steps:lying_loader"}})
    with pytest.raises(LoadError, match="returned dict, expected DataFrame"):
        load_datasets(config, data_root=book, run_id="test", as_of_date=AS_OF)


def test_portfolio_list_must_satisfy_its_schema(book: Path) -> None:
    config = resolved_example(datasets=example_datasets(portfolios={"loader": {"name": "load_universe", "params": {"min_latency_s": 0, "max_latency_s": 0}}}))
    with pytest.raises(LoadError, match="portfolios: portfolios"):
        load_datasets(config, data_root=book, run_id="test", as_of_date=AS_OF)


# --- the load stage runs loaders concurrently and hands each its pool ---


def _with_extra_datasets(**extra: object) -> ResolvedConfig:
    body = resolved_example().config.model_dump(mode="json", by_alias=True, exclude_none=True)
    return resolved_example(datasets={**body["datasets"], **extra})


@pytest.mark.parametrize("loader", ["barrier_loader", "async_barrier_loader"], ids=["sync loaders in threads", "async loaders on the loop"])
def test_dataset_loaders_run_at_the_same_time(loader: str, book: Path) -> None:
    resolved = _with_extra_datasets(left={"loader": f"tests.steps:{loader}", "depends_on": ["portfolios"]}, right={"loader": f"tests.steps:{loader}", "depends_on": ["portfolios"]})
    loaded = load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)  # would deadlock or time out if left and right ran one after another
    assert loaded.frames["left"]["portfolio_id"].tolist() == ["P1", "P2"]
    assert loaded.frames["right"]["portfolio_id"].tolist() == ["P1", "P2"]


def test_the_book_and_an_independent_dataset_load_at_the_same_time(book: Path) -> None:
    resolved = resolved_example(datasets=example_datasets(portfolios={"loader": "tests.steps:barrier_portfolios_loader"}, universe={"loader": "tests.steps:barrier_loader"}))
    loaded = load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)  # would time out at the barrier if the security master still waited for the book of record
    assert loaded.portfolio_ids == ("P1", "P2")
    assert loaded.frames["universe"].empty, "universe declared no dependency on the book, so its request carried no ids"


def test_a_loader_receives_the_frames_of_the_datasets_it_depends_on(book: Path) -> None:
    resolved = _with_extra_datasets(enriched={"loader": "tests.steps:inputs_reporting_loader", "depends_on": ["universe", "global_parameters"]})
    loaded = load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)
    enriched = loaded.frames["enriched"]
    assert dict(zip(enriched["input"], enriched["rows"], strict=True)) == {"universe": 3, "global_parameters": 2}, "each dependency arrives whole, by name"
    assert enriched["ids"].tolist() == ["", ""], "the book is not among its dependencies, so the request carries no ids"


def test_a_per_portfolio_batch_sees_its_inputs_cut_to_its_own_accounts(book: Path) -> None:
    steps.INPUT_VIEWS.clear()
    holdings: dict[str, object] = {"loader": "tests.steps:recording_inputs_holdings", "scope": "per_portfolio", "batch_size": 1, "depends_on": ["universe"]}
    load_datasets(resolved_example(datasets=example_datasets(holdings=holdings)), data_root=book, run_id="test", as_of_date=AS_OF)
    assert sorted(steps.INPUT_VIEWS) == [(("P1",), ("P1",), 3), (("P2",), ("P2",), 3)], "the portfolios input is cut to the batch's rows; the universe, with no portfolio_id column, is passed whole"


def test_a_dependent_of_a_failed_dataset_is_skipped_and_named_beside_the_failure(book: Path) -> None:
    resolved = _with_extra_datasets(left={"loader": "tests.steps:invalid_input_loader"}, right={"loader": "tests.steps:unreachable_loader", "depends_on": ["left"]})
    with pytest.raises(LoadError, match=r"left: left: no rows as of 2026-08-28; not loaded because left failed: right"):
        load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)  # `right` raising ConnectionError would prove it was called; being skipped, the failure stays input-shaped


def test_a_dependent_of_a_per_portfolio_dataset_gets_the_combined_frame_for_the_surviving_accounts(book: Path) -> None:
    steps.BATCHES.clear()
    datasets = _sharded(1, fails_for=["P2"]) | {"enriched": {"loader": "tests.steps:inputs_reporting_loader", "depends_on": ["holdings", "portfolios"]}}
    loaded = load_datasets(resolved_example(datasets=datasets), data_root=book, run_id="test", as_of_date=AS_OF)
    enriched = loaded.frames["enriched"]
    assert dict(zip(enriched["input"], enriched["rows"], strict=True)) == {"holdings": 2, "portfolios": 2}, "P1's two positions arrived as one frame once every batch had reported"
    assert enriched["ids"].tolist() == ["P1", "P1"], "a portfolio an upstream batch rejected is not asked for again"
    assert set(loaded.rejected) == {P2}


def test_an_inline_book_costs_nothing_and_keeps_its_written_order(book: Path) -> None:
    loaded = load_datasets(resolved_example(datasets=example_datasets(portfolios=["P2", "P1"])), data_root=book, run_id="test", as_of_date=AS_OF)
    assert loaded.portfolio_ids == ("P2", "P1"), "the written order is the solve order"
    assert loaded.solve_orders == {"P2": 0, "P1": 1}
    audit = {record.name: record for record in loaded.audits}["portfolios"]
    assert (audit.loader_qualname, audit.batches, audit.load_time_s) == ("config", 0, 0.0), "no loader ran; the audit names the config as the source"
    assert len(audit.content_sha256) == 64


def test_every_failed_dataset_is_reported_together_as_rejected_input(book: Path) -> None:
    resolved = _with_extra_datasets(left={"loader": "tests.steps:invalid_input_loader"}, right={"loader": "tests.steps:invalid_input_loader"})
    with pytest.raises(LoadError, match=r"left: left: no rows as of 2026-08-28; right: right: no rows"):
        load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)


def test_an_unreachable_backend_keeps_its_exception_type_so_the_exit_code_is_infrastructure(book: Path) -> None:
    resolved = _with_extra_datasets(left={"loader": "tests.steps:invalid_input_loader"}, right={"loader": "tests.steps:unreachable_loader"})
    with pytest.raises(ConnectionError, match="right: connection refused"):
        load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)


def test_audits_record_how_long_each_dataset_took(example_loaded: LoadedDatasets) -> None:
    assert all(record.load_time_s >= 0.0 for record in example_loaded.audits)
    assert all(record.started_s >= 0.0 for record in example_loaded.audits), "each audit records how long the dataset waited on its dependencies before starting"
    assert {record.name for record in example_loaded.audits} == {"portfolios", "holdings", "universe", "details", "global_parameters", "buy_universe_parameters"}


def test_the_sync_entry_point_refuses_to_nest_inside_a_running_loop(book: Path) -> None:
    async def inside() -> None:
        load_datasets(resolved_example(), data_root=book, run_id="test", as_of_date=AS_OF)

    with pytest.raises(RuntimeError, match="await load_datasets_async instead"):
        asyncio.run(inside())


def test_the_async_entry_point_can_be_awaited_directly(book: Path) -> None:
    loaded = asyncio.run(load_datasets_async(resolved_example(), data_root=book, run_id="test", as_of_date=AS_OF))
    assert loaded.portfolio_ids == ("P1", "P2")


def test_max_in_flight_bounds_the_batches_the_engine_runs_at_once(book: Path) -> None:
    def peak(**bound: object) -> int:
        steps.PEAK_IN_FLIGHT.clear()
        holdings = {"loader": "tests.steps:in_flight_recording_holdings", "scope": "per_portfolio", "batch_size": 1, **bound}
        load_datasets(resolved_example(datasets=example_datasets(holdings=holdings)), data_root=book, run_id="test", as_of_date=AS_OF)
        return steps.PEAK_IN_FLIGHT["holdings"]

    assert peak(max_in_flight=1) == 1, "one slot serialises the batches the engine cut"
    assert peak() == 2, "an unbounded dataset runs every batch at once"


# --- per-portfolio datasets: the engine owns the fan-out ---


def _sharded(batch_size: int | None, **overrides: object) -> dict[str, object]:
    """The example datasets with ``holdings`` loaded per portfolio by the recording loader."""
    holdings: dict[str, object] = {"loader": {"name": "tests.steps:recording_holdings", "params": {"min_latency_s": 0, "max_latency_s": 0, **overrides}}, "scope": "per_portfolio"}
    if batch_size is not None:
        holdings["batch_size"] = batch_size
    return example_datasets(holdings=holdings)


def test_a_per_portfolio_dataset_is_called_once_per_batch(book: Path) -> None:
    steps.BATCHES.clear()
    loaded = load_datasets(resolved_example(datasets=_sharded(1)), data_root=book, run_id="test", as_of_date=AS_OF)
    assert sorted(steps.BATCHES) == [("P1",), ("P2",)], (
        "batch_size 1 is one call per portfolio, driven by the engine rather than inside the loader; the calls run concurrently, so their order is the sources'"
    )
    audit = {record.name: record for record in loaded.audits}["holdings"]
    assert (audit.batches, audit.rejected, audit.rows) == (2, 0, 4), "the batches are concatenated back into one dataset, and the audit records the partition"
    assert set(loaded.per_portfolio) == {"details", "holdings"} and "holdings" not in loaded.frames, "assembly sees the global datasets only"


def test_the_whole_book_goes_in_one_call_when_no_batch_size_is_given(book: Path) -> None:
    steps.BATCHES.clear()
    load_datasets(resolved_example(datasets=_sharded(None)), data_root=book, run_id="test", as_of_date=AS_OF)
    assert steps.BATCHES == [("P1", "P2")], "a source that takes an id list is still one call; the scope only says the dataset is not assembly's to see"


def test_a_failed_batch_rejects_its_own_portfolios_and_no_others(book: Path) -> None:
    steps.BATCHES.clear()
    resolved = resolved_example(datasets=_sharded(1, fails_for=["P2"]))
    loaded = load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)
    assert set(loaded.rejected) == {P2}
    assert (loaded.rejected[P2].stage, loaded.rejected[P2].error_type) == ("load", "ValueError")
    assert "dataset 'holdings' did not load for this portfolio" in loaded.rejected[P2].message
    assert {record.name: record for record in loaded.audits}["holdings"].rejected == 1
    assembled = assemble(loaded, resolved, run_id="test", as_of_date=AS_OF)
    assert set(assembled.rejected) == {P2} and assembled.portfolio_ids == ("P1", "P2")


def test_a_per_portfolio_dataset_no_batch_of_which_loads_rejects_the_run(book: Path) -> None:
    steps.BATCHES.clear()
    with pytest.raises(LoadError, match="no data for"):
        load_datasets(resolved_example(datasets=_sharded(1, fails_for=["P1", "P2"])), data_root=book, run_id="test", as_of_date=AS_OF)


def test_a_per_portfolio_source_that_is_down_raises_the_failure_not_the_group(tmp_path: Path) -> None:
    root = two_account_book(tmp_path / "book")
    (root / "holdings.csv").unlink()  # every batch fails, which is the source being down rather than an account being bad
    with pytest.raises(FileNotFoundError, match=r"holdings\.csv") as info:
        load_datasets(resolved_example(), data_root=root, run_id="test", as_of_date=AS_OF)
    assert "TaskGroup" not in str(info.value), "the group a fan-out loader raises is unwrapped to the one failure inside it"


def test_assembly_is_told_why_a_per_portfolio_dataset_is_not_there(book: Path) -> None:
    resolved = resolved_example(datasets=_sharded(1), assembly=[{"name": "drop", "params": {"datasets": ["holdings"]}}])
    loaded = load_datasets(resolved, data_root=book, run_id="test", as_of_date=AS_OF)
    with pytest.raises(AssemblyError, match="'holdings' is a per_portfolio dataset, which assembly never sees"):
        assemble(loaded, resolved, run_id="test", as_of_date=AS_OF)
