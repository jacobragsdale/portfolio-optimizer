"""Tier 3: what the run-config models refuse, and that the shipped example validates."""

import json

import pytest
from pydantic import ValidationError

from portfolio_optimizer.config.models import DatasetConfig, InlinePortfolios, RunConfig, StepSpec, config_sha256, load_run_config
from tests.conftest import EXAMPLE_CONFIG, example_body


@pytest.fixture
def example_text() -> str:
    return EXAMPLE_CONFIG.read_text()


@pytest.fixture
def example_dict() -> dict[str, object]:
    """The shipped file as it ships: this module is about the config, so nothing here is adjusted for test speed."""
    return example_body(latency=True)


def section(config: dict[str, object], key: str) -> dict[str, object]:
    value = config[key]
    assert isinstance(value, dict)
    return {str(inner_key): inner for inner_key, inner in value.items()}


def test_shipped_example_validates(example_text: str) -> None:
    config = load_run_config(example_text)
    assert config.run.name == "example_rebalance"
    assert config.execution.dependencies == "overlap"
    assert config.solve_order is None
    assert config.rules[0].params == {}, "the liquidity threshold is loaded at runtime, not written into the config"


def test_step_spec_accepts_bare_names_and_objects() -> None:
    bare = StepSpec.model_validate_json('"exclude_restricted"')
    full = StepSpec.model_validate_json('{"name": "my_firm.rules:tilt", "params": {"strength": "0.5"}}')
    assert bare.name == "exclude_restricted"
    assert bare.params == {}
    assert not bare.is_qualified
    assert full.is_qualified
    assert full.params == {"strength": "0.5"}


@pytest.mark.parametrize("name", ["cap", "cap_single_name", "pkg.mod:fn", "a.b.c:fn_2"])
def test_well_formed_step_names(name: str) -> None:
    assert StepSpec.model_validate_json(json.dumps(name)).name == name


@pytest.mark.parametrize("name", ["", "1abc", "pkg:", ":fn", "pkg.mod", "pkg mod:fn", "fn()", "a:b:c"])
def test_malformed_step_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        StepSpec.model_validate_json(json.dumps(name))


def test_unknown_keys_are_rejected(example_dict: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_run_config(json.dumps(example_dict | {"parallelism": 4}))


def test_missing_required_dataset_is_rejected(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets")
    del datasets["universe"]
    with pytest.raises(ValidationError, match="missing \\['universe'\\]"):
        load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": []}))


def test_engine_frames_may_come_from_assembly_steps(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets")
    del datasets["holdings"]
    with pytest.raises(ValidationError, match="missing \\['holdings'\\]; a run without assembly steps has nothing else to produce them"):
        load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": []}))
    config = load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": [{"name": "union", "params": {"into": "holdings", "sources": ["custody"]}}]}))
    assert [step.name for step in config.assembly] == ["union"]


def test_constraints_is_optional(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets")
    del datasets["constraints"]
    config = load_run_config(json.dumps(example_dict | {"datasets": datasets}))
    assert "constraints" not in config.datasets, "a run constrained only by the trade identity simply does not declare the dataset"


@pytest.mark.parametrize("mechanic", [{"executor": "thread"}, {"max_workers": 2}])
def test_execution_mechanics_are_settings_not_config(example_dict: dict[str, object], mechanic: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_run_config(json.dumps(example_dict | {"execution": section(example_dict, "execution") | mechanic}))


def test_naive_as_of_date_is_rejected(example_dict: dict[str, object]) -> None:
    run = section(example_dict, "run") | {"as_of_date": "2026-08-28T00:00:00"}
    with pytest.raises(ValidationError, match="timezone"):
        load_run_config(json.dumps(example_dict | {"run": run}))


@pytest.mark.parametrize(("field", "value"), [("time_limit_s", 0), ("violation_tol", 0)])
def test_numeric_limits_just_past_their_bounds_are_rejected(example_dict: dict[str, object], field: str, value: int) -> None:
    key = "solver" if field == "time_limit_s" else "post_solve"
    patched = section(example_dict, key) | {field: value}
    with pytest.raises(ValidationError):
        load_run_config(json.dumps(example_dict | {key: patched}))


def test_objective_needs_at_least_one_term(example_dict: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        load_run_config(json.dumps(example_dict | {"objective": {"terms": []}}))


def test_config_hash_ignores_source_whitespace_but_not_values(example_text: str, example_dict: dict[str, object]) -> None:
    compact = json.dumps(example_dict, separators=(",", ":"))
    assert config_sha256(load_run_config(example_text)) == config_sha256(load_run_config(compact))
    changed = load_run_config(json.dumps(example_dict | {"run": section(example_dict, "run") | {"name": "other"}}))
    assert config_sha256(changed) != config_sha256(load_run_config(example_text))


def test_defaults_fill_optional_sections() -> None:
    minimal = {
        "run": {"name": "r", "as_of_date": "2026-01-01T00:00:00Z"},
        "datasets": {name: {"loader": "csv"} for name in ("portfolios", "holdings", "universe", "details", "constraints")},
        "objective": {"terms": ["alpha"]},
        "sink": "orders_to_parquet",
    }
    config = RunConfig.model_validate_json(json.dumps(minimal))
    assert config.solver.name == "CLARABEL"
    assert config.execution.on_error == "fail_fast"
    assert config.execution.dependencies == "overlap"
    assert config.assembly == ()
    assert config.rules == ()


@pytest.mark.parametrize(
    ("holdings", "fragment"),
    [({"scope": "per_portfolio", "max_in_flight": 0}, "greater than or equal to 1"), ({"max_in_flight": 4}, "apply only to a per_portfolio dataset")],
    ids=["zero in-flight", "a bound on a dataset loaded by one call"],
)
def test_a_meaningless_in_flight_bound_is_rejected(example_dict: dict[str, object], holdings: dict[str, object], fragment: str) -> None:
    datasets = section(example_dict, "datasets") | {"holdings": {"loader": "load_holdings"} | holdings}
    with pytest.raises(ValidationError, match=fragment):
        load_run_config(json.dumps(example_dict | {"datasets": datasets}))


@pytest.mark.parametrize("portfolios", [["P7", "P2", "P9"], {"ids": ["P7", "P2", "P9"]}], ids=["bare array", "object form"])
def test_the_portfolio_list_may_be_written_inline(example_dict: dict[str, object], portfolios: object) -> None:
    config = load_run_config(json.dumps(example_dict | {"datasets": section(example_dict, "datasets") | {"portfolios": portfolios}}))
    book = config.datasets["portfolios"]
    assert isinstance(book, InlinePortfolios)
    assert book.ids == ("P7", "P2", "P9"), "the written order is kept: it is the solve order"


def test_a_bare_inline_list_hashes_the_same_as_its_object_form(example_dict: dict[str, object]) -> None:
    bare = load_run_config(json.dumps(example_dict | {"datasets": section(example_dict, "datasets") | {"portfolios": ["P1", "P2"]}}))
    wrapped = load_run_config(json.dumps(example_dict | {"datasets": section(example_dict, "datasets") | {"portfolios": {"ids": ["P1", "P2"]}}}))
    assert config_sha256(bare) == config_sha256(wrapped)


@pytest.mark.parametrize(("portfolios", "fragment"), [(["P1", "P1"], "ids repeat"), ([""], "non-empty"), ([], "at least 1")], ids=["repeated id", "empty id", "empty list"])
def test_a_malformed_inline_book_is_rejected(example_dict: dict[str, object], portfolios: object, fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        load_run_config(json.dumps(example_dict | {"datasets": section(example_dict, "datasets") | {"portfolios": portfolios}}))


def test_only_the_portfolio_list_may_be_written_inline(example_dict: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="only 'portfolios' may be written inline"):
        load_run_config(json.dumps(example_dict | {"datasets": section(example_dict, "datasets") | {"holdings": {"ids": ["P1"]}}}))


def test_the_portfolio_list_is_required_and_global(example_dict: dict[str, object]) -> None:
    without = {name: spec for name, spec in section(example_dict, "datasets").items() if name != "portfolios"}
    with pytest.raises(ValidationError, match="datasets must declare 'portfolios'"):
        load_run_config(json.dumps(example_dict | {"datasets": without}))
    per_account = section(example_dict, "datasets") | {"portfolios": {"loader": "load_portfolios", "scope": "per_portfolio"}}
    with pytest.raises(ValidationError, match="portfolios must be a global dataset"):
        load_run_config(json.dumps(example_dict | {"datasets": per_account}))


def test_a_per_portfolio_dataset_depends_on_the_book_implicitly(example_dict: dict[str, object]) -> None:
    config = load_run_config(json.dumps(example_dict))
    holdings, constraints, universe = config.datasets["holdings"], config.datasets["constraints"], config.datasets["universe"]
    assert isinstance(holdings, DatasetConfig) and holdings.dependencies() == ("portfolios",), "per_portfolio implies the book without declaring it"
    assert isinstance(constraints, DatasetConfig) and constraints.depends_on == ("portfolios",) == constraints.dependencies(), "a global input that wants the ids declares the dependency"
    assert isinstance(universe, DatasetConfig) and universe.dependencies() == (), "nothing declared, nothing waited on"


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        ({"loader": "load_holdings", "depends_on": ["univrse"]}, r"datasets.holdings: depends_on names unknown dataset\(s\) \['univrse'\]"),
        ({"loader": "load_holdings", "depends_on": ["holdings"]}, "cannot depend on itself"),
        ({"loader": "load_holdings", "depends_on": ["portfolios", "portfolios"]}, "depends_on repeats"),
    ],
    ids=["unknown name", "self-dependency", "repeated name"],
)
def test_malformed_dependencies_are_rejected(example_dict: dict[str, object], entry: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        load_run_config(json.dumps(example_dict | {"datasets": section(example_dict, "datasets") | {"holdings": entry}}))


def test_dependency_cycles_are_rejected_naming_the_cycle(example_dict: dict[str, object]) -> None:
    looped = section(example_dict, "datasets") | {"universe": {"loader": "load_universe", "depends_on": ["constraints"]}, "constraints": {"loader": "load_constraints", "depends_on": ["universe"]}}
    with pytest.raises(ValidationError, match="cycle: universe -> constraints -> universe"):
        load_run_config(json.dumps(example_dict | {"datasets": looped}))
    implicit = section(example_dict, "datasets") | {"portfolios": {"loader": "load_portfolios", "depends_on": ["holdings"]}}
    with pytest.raises(ValidationError, match=r"cycle: portfolios -> holdings -> portfolios \(a per_portfolio dataset depends on 'portfolios' implicitly\)"):
        load_run_config(json.dumps(example_dict | {"datasets": implicit}))


def test_each_input_carries_its_own_bound(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets") | {
        "holdings": {"loader": {"name": "csv", "params": {"path": "holdings.csv"}}, "scope": "per_portfolio", "batch_size": 1, "max_in_flight": 2},
        "details": {"loader": "csv", "scope": "per_portfolio"},
    }
    config = load_run_config(json.dumps(example_dict | {"datasets": datasets}))
    holdings, details = config.datasets["holdings"], config.datasets["details"]
    assert isinstance(holdings, DatasetConfig) and holdings.max_in_flight == 2
    assert isinstance(details, DatasetConfig) and details.max_in_flight is None, "omitted is unbounded"
