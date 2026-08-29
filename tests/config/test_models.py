"""Tier 3: what the run-config models refuse, and that the shipped example validates."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_optimizer.config.models import RunConfig, StepSpec, config_sha256, is_step_name, load_run_config

EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "example_run.json"


@pytest.fixture
def example_text() -> str:
    return EXAMPLE.read_text()


@pytest.fixture
def example_dict(example_text: str) -> dict[str, object]:
    loaded = json.loads(example_text)
    assert isinstance(loaded, dict)
    return {str(key): value for key, value in loaded.items()}


def section(config: dict[str, object], key: str) -> dict[str, object]:
    value = config[key]
    assert isinstance(value, dict)
    return {str(inner_key): inner for inner_key, inner in value.items()}


def test_shipped_example_validates(example_text: str) -> None:
    config = load_run_config(example_text)
    assert config.run.name == "example_rebalance"
    assert config.execution.mode == "parallel_build_sequential_solve"
    assert config.rules[0].params == {"min_adv_shares": 1000}


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
    assert is_step_name(name)
    assert StepSpec.model_validate_json(json.dumps(name)).name == name


@pytest.mark.parametrize("name", ["", "1abc", "pkg:", ":fn", "pkg.mod", "pkg mod:fn", "fn()", "a:b:c"])
def test_malformed_step_names_are_rejected(name: str) -> None:
    assert not is_step_name(name)
    with pytest.raises(ValidationError):
        StepSpec.model_validate_json(json.dumps(name))


def test_unknown_keys_are_rejected(example_dict: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_run_config(json.dumps(example_dict | {"parallelism": 4}))


def test_missing_required_dataset_is_rejected(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets")
    del datasets["targets"]
    with pytest.raises(ValidationError, match="missing \\['targets'\\]"):
        load_run_config(json.dumps(example_dict | {"datasets": datasets}))


def test_join_must_reference_a_declared_dataset(example_dict: dict[str, object]) -> None:
    assembly = {"joins": [{"into": "universe", "source": "sectors", "on": ["security_id"], "cardinality": "one_to_one"}]}
    with pytest.raises(ValidationError, match="source dataset 'sectors' is not declared"):
        load_run_config(json.dumps(example_dict | {"assembly": assembly}))


def test_join_cannot_target_itself_or_use_constraints(example_dict: dict[str, object]) -> None:
    assembly = {"joins": [{"into": "universe", "source": "universe", "on": ["security_id"], "cardinality": "one_to_one"}]}
    with pytest.raises(ValidationError, match="cannot join 'universe' into 'universe'"):
        load_run_config(json.dumps(example_dict | {"assembly": assembly}))


def test_join_into_must_be_an_engine_frame(example_dict: dict[str, object]) -> None:
    assembly = {"joins": [{"into": "prices", "source": "universe", "on": ["security_id"], "cardinality": "one_to_one"}]}
    with pytest.raises(ValidationError, match="into"):
        load_run_config(json.dumps(example_dict | {"assembly": assembly}))


def test_parallel_mode_cannot_use_threads(example_dict: dict[str, object]) -> None:
    execution = {"mode": "parallel", "executor": "thread", "max_workers": 2}
    with pytest.raises(ValidationError, match="not thread-safe"):
        load_run_config(json.dumps(example_dict | {"execution": execution}))


def test_naive_as_of_is_rejected(example_dict: dict[str, object]) -> None:
    run = section(example_dict, "run") | {"as_of": "2026-08-28T00:00:00"}
    with pytest.raises(ValidationError, match="timezone"):
        load_run_config(json.dumps(example_dict | {"run": run}))


@pytest.mark.parametrize(("field", "value"), [("max_workers", 0), ("time_limit_s", 0), ("violation_tol", 0)])
def test_numeric_limits_just_past_their_bounds_are_rejected(example_dict: dict[str, object], field: str, value: int) -> None:
    key = "execution" if field == "max_workers" else "solver" if field == "time_limit_s" else "post_solve"
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
        "run": {"name": "r", "as_of": "2026-01-01T00:00:00Z"},
        "portfolios": "csv",
        "datasets": {name: {"loader": "csv"} for name in ("holdings", "universe", "details", "constraints", "targets")},
        "objective": {"terms": ["tracking_error"]},
        "sink": "orders_to_parquet",
        "execution": {"mode": "sequential"},
    }
    config = RunConfig.model_validate_json(json.dumps(minimal))
    assert config.solver.name == "CLARABEL"
    assert config.execution.executor == "process"
    assert config.assembly.joins == ()
    assert config.rules == ()
