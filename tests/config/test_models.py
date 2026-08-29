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
        load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": []}))


def test_engine_frames_may_come_from_assembly_steps_but_constraints_must_be_loaded(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets")
    del datasets["holdings"]
    with pytest.raises(ValidationError, match="missing \\['holdings'\\]; a run without assembly steps has nothing else to produce them"):
        load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": []}))
    config = load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": [{"name": "union", "params": {"into": "holdings", "sources": ["prices"]}}]}))
    assert [step.name for step in config.assembly] == ["union"]
    del datasets["constraints"]
    with pytest.raises(ValidationError, match="missing \\['constraints'\\]"):
        load_run_config(json.dumps(example_dict | {"datasets": datasets, "assembly": ["my_pkg.assembly:everything"]}))


@pytest.mark.parametrize("mechanic", [{"executor": "thread"}, {"max_workers": 2}])
def test_execution_mechanics_are_settings_not_config(example_dict: dict[str, object], mechanic: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_run_config(json.dumps(example_dict | {"execution": section(example_dict, "execution") | mechanic}))


def test_naive_as_of_is_rejected(example_dict: dict[str, object]) -> None:
    run = section(example_dict, "run") | {"as_of": "2026-08-28T00:00:00"}
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
        "run": {"name": "r", "as_of": "2026-01-01T00:00:00Z"},
        "portfolios": "csv",
        "datasets": {name: {"loader": "csv"} for name in ("holdings", "universe", "details", "constraints", "targets")},
        "objective": {"terms": ["tracking_error"]},
        "sink": "orders_to_parquet",
        "execution": {"mode": "sequential"},
    }
    config = RunConfig.model_validate_json(json.dumps(minimal))
    assert config.solver.name == "CLARABEL"
    assert config.execution.on_error == "fail_fast"
    assert config.assembly == ()
    assert config.rules == ()


def test_rate_limit_pool_named_by_a_dataset_must_be_declared(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets") | {"holdings": {"loader": {"name": "csv", "params": {"path": "holdings.csv"}}, "rate_limit": "vendor"}}
    with pytest.raises(ValidationError, match="rate_limit 'vendor' is not declared in rate_limits \\[\\]"):
        load_run_config(json.dumps(example_dict | {"datasets": datasets}))
    config = load_run_config(json.dumps(example_dict | {"datasets": datasets, "rate_limits": {"vendor": {"requests_per_second": 20, "max_in_flight": 4}}}))
    assert config.datasets["holdings"].rate_limit == "vendor"
    assert config.rate_limits["vendor"].to_limit().burst == 20


@pytest.mark.parametrize(
    ("pool", "fragment"),
    [
        ({}, "requests_per_second, max_in_flight, or both"),
        ({"burst": 5, "max_in_flight": 2}, "burst only applies with requests_per_second"),
        ({"requests_per_second": 0}, "greater than 0"),
        ({"max_in_flight": 0}, "greater than or equal to 1"),
    ],
    ids=["no bound", "burst without a rate", "zero rate", "zero in-flight"],
)
def test_meaningless_rate_limits_are_rejected(example_dict: dict[str, object], pool: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        load_run_config(json.dumps(example_dict | {"rate_limits": {"vendor": pool}}))


def test_rate_limit_burst_defaults_to_the_rate_rounded_up_and_never_below_one() -> None:
    from portfolio_optimizer.config.models import RateLimitConfig

    assert RateLimitConfig.model_validate({"requests_per_second": 2.5}).to_limit().burst == 3
    assert RateLimitConfig.model_validate({"requests_per_second": 0.1}).to_limit().burst == 1
    assert RateLimitConfig.model_validate({"requests_per_second": 4, "burst": 1}).to_limit().burst == 1
    assert RateLimitConfig.model_validate({"max_in_flight": 8}).to_limit().max_in_flight == 8


@pytest.mark.parametrize(
    "portfolios",
    ["csv", {"name": "csv", "params": {"path": "portfolios.csv"}}, {"loader": {"name": "csv", "params": {"path": "portfolios.csv"}}}, {"loader": "csv", "rate_limit": {"max_in_flight": 1}}],
    ids=["bare name", "bare step", "loader object", "loader with its own bound"],
)
def test_portfolios_accepts_a_bare_step_or_the_full_input_form(example_dict: dict[str, object], portfolios: object) -> None:
    config = load_run_config(json.dumps(example_dict | {"portfolios": portfolios}))
    assert config.portfolios.loader.name == "csv"


def test_a_bare_portfolios_step_hashes_the_same_as_its_wrapped_form(example_dict: dict[str, object]) -> None:
    bare = load_run_config(json.dumps(example_dict | {"portfolios": {"name": "csv", "params": {"path": "portfolios.csv"}}}))
    wrapped = load_run_config(json.dumps(example_dict | {"portfolios": {"loader": {"name": "csv", "params": {"path": "portfolios.csv"}}}}))
    assert config_sha256(bare) == config_sha256(wrapped)


def test_each_input_may_carry_its_own_inline_bound(example_dict: dict[str, object]) -> None:
    datasets = section(example_dict, "datasets") | {
        "holdings": {"loader": {"name": "csv", "params": {"path": "holdings.csv"}}, "rate_limit": {"requests_per_second": 5, "max_in_flight": 2}},
        "universe": {"loader": {"name": "csv", "params": {"path": "universe.csv"}}, "rate_limit": {"max_in_flight": 16}},
    }
    config = load_run_config(json.dumps(example_dict | {"datasets": datasets, "portfolios": {"loader": "csv", "rate_limit": "slow"}, "rate_limits": {"slow": {"requests_per_second": 1}}}))
    holdings = config.datasets["holdings"].rate_limit
    universe = config.datasets["universe"].rate_limit
    assert not isinstance(holdings, str | None) and holdings.to_limit().max_in_flight == 2
    assert not isinstance(universe, str | None) and universe.to_limit().requests_per_second is None
    assert config.portfolios.rate_limit == "slow"
    with pytest.raises(ValidationError, match="portfolios: rate_limit 'fast' is not declared"):
        load_run_config(json.dumps(example_dict | {"portfolios": {"loader": "csv", "rate_limit": "fast"}}))
