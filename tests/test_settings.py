"""Tier 3: settings load from an explicit environment, default to a laptop run, and refuse invalid or unknown variables."""

import re
from pathlib import Path

import pytest

from portfolio_optimizer.settings import SettingsError, load_settings

BASE_ENV: dict[str, str] = {
    "PORTFOLIO_OPTIMIZER_OUTPUT_DIR": "/tmp/out",
    "PORTFOLIO_OPTIMIZER_DATA_ROOT": "/tmp/data",
    "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "DEBUG",
    "PORTFOLIO_OPTIMIZER_CLUSTER": "local",
    "PORTFOLIO_OPTIMIZER_MIN_WORKERS": "2",
    "PORTFOLIO_OPTIMIZER_MAX_WORKERS": "4",
    "PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S": "60",
}

GATEWAY_ENV: dict[str, str] = {
    "PORTFOLIO_OPTIMIZER_CLUSTER": "https://dask.example",
    "PORTFOLIO_OPTIMIZER_WORKER_IMAGE": "registry/optimizer:1.2",
    "PORTFOLIO_OPTIMIZER_GATEWAY_PASSWORD": "hunter2",
    "PORTFOLIO_OPTIMIZER_GATEWAY_PROXY_ADDRESS": "tls://scheduler.example:8786",
}


def test_settings_load_from_an_explicit_environment() -> None:
    settings = load_settings(BASE_ENV | {"HOME": "/x"})
    assert settings.output_dir == Path("/tmp/out")
    assert settings.log_level == "DEBUG"
    execution = settings.execution()
    assert (execution.cluster, execution.cluster_kind, execution.min_workers, execution.max_workers, execution.cluster_timeout_s) == ("local", "local", 2, 4, 60.0)
    assert settings.shown() == {"output_dir": "/tmp/out", "data_root": "/tmp/data", "log_level": "DEBUG", "cluster": "local", "min_workers": "2", "max_workers": "4", "cluster_timeout_s": "60.0"}


def test_every_setting_has_a_default_a_laptop_can_run_with() -> None:
    settings = load_settings({"HOME": "/x"})
    execution = settings.execution()
    assert (settings.output_dir, settings.data_root, settings.log_level) == (Path("out"), Path(), "INFO")
    assert (execution.cluster, execution.cluster_kind, execution.min_workers, execution.max_workers) == ("inline", "inline", 1, 1)
    assert execution.step_packages is None, "unset, a qualified step name may import from any installed module"


def test_step_packages_are_a_comma_separated_allowlist() -> None:
    execution = load_settings({"PORTFOLIO_OPTIMIZER_STEP_PACKAGES": "my_firm_quant, desk_tools"}).execution()
    assert execution.step_packages == ("my_firm_quant", "desk_tools")


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_LOG_LEVEL": "LOUD"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_TYPO": "1"}, "PORTFOLIO_OPTIMIZER_TYPO"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_MAX_WORKERS": "0"}, "PORTFOLIO_OPTIMIZER_MAX_WORKERS"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "https://dask.example"}, "https://dask.example is a Dask Gateway address and requires PORTFOLIO_OPTIMIZER_WORKER_IMAGE"),
        ({**BASE_ENV, **GATEWAY_ENV, "PORTFOLIO_OPTIMIZER_GATEWAY_PASSWORD": ""}, "PORTFOLIO_OPTIMIZER_GATEWAY_PASSWORD"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "somewhere"}, "PORTFOLIO_OPTIMIZER_CLUSTER"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_MIN_WORKERS": "8"}, "PORTFOLIO_OPTIMIZER_MIN_WORKERS (8) exceeds PORTFOLIO_OPTIMIZER_MAX_WORKERS (4)"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S": "0"}, "PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S"),
    ],
    ids=["invalid level", "unknown variable", "zero workers", "gateway without image", "gateway without password", "malformed cluster", "min above max", "zero timeout"],
)
def test_settings_refuse_invalid_or_unknown_variables(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(SettingsError, match=re.escape(fragment)):
        load_settings(env)


def test_cluster_kinds_and_what_a_gateway_requires() -> None:
    assert load_settings(BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "tcp://scheduler:8786"}).execution().cluster_kind == "address"
    assert load_settings(BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "inline"}).execution().cluster_kind == "inline"
    settings = load_settings(BASE_ENV | GATEWAY_ENV)
    gateway = settings.execution()
    assert (gateway.cluster_kind, gateway.worker_image, gateway.gateway_proxy_address) == ("gateway", "registry/optimizer:1.2", "tls://scheduler.example:8786")
    assert gateway.gateway_password is not None and gateway.gateway_password.get_secret_value() == "hunter2"
    assert settings.shown()["gateway_password"] == "**********"  # the manifest records that a password was given, never which
