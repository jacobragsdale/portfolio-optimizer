"""Tier 3: settings load from an explicit environment and refuse incomplete, invalid, or unknown variables."""

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


def test_settings_load_from_an_explicit_environment() -> None:
    settings = load_settings(BASE_ENV | {"HOME": "/x", "PORTFOLIO_OPTIMIZER_IMAGE_DIGEST": "sha256:abc"})
    assert settings.output_dir == Path("/tmp/out")
    assert settings.log_level == "DEBUG"
    execution = settings.execution()
    assert (execution.cluster, execution.cluster_kind, execution.min_workers, execution.max_workers, execution.cluster_timeout_s) == ("local", "local", 2, 4, 60.0)
    assert execution.image_digest == "sha256:abc"
    assert settings.shown() == {
        "output_dir": "/tmp/out",
        "data_root": "/tmp/data",
        "log_level": "DEBUG",
        "cluster": "local",
        "min_workers": "2",
        "max_workers": "4",
        "cluster_timeout_s": "60.0",
        "image_digest": "sha256:abc",
    }


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({key: value for key, value in BASE_ENV.items() if key != "PORTFOLIO_OPTIMIZER_LOG_LEVEL"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL: Field required"),
        ({key: value for key, value in BASE_ENV.items() if key != "PORTFOLIO_OPTIMIZER_CLUSTER"}, "PORTFOLIO_OPTIMIZER_CLUSTER: Field required"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_LOG_LEVEL": "LOUD"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_TYPO": "1"}, "PORTFOLIO_OPTIMIZER_TYPO"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_MAX_WORKERS": "0"}, "PORTFOLIO_OPTIMIZER_MAX_WORKERS"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "kubernetes"}, "PORTFOLIO_OPTIMIZER_CLUSTER=kubernetes requires PORTFOLIO_OPTIMIZER_WORKER_IMAGE"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "somewhere"}, "PORTFOLIO_OPTIMIZER_CLUSTER"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_MIN_WORKERS": "8"}, "PORTFOLIO_OPTIMIZER_MIN_WORKERS (8) exceeds PORTFOLIO_OPTIMIZER_MAX_WORKERS (4)"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S": "0"}, "PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S"),
    ],
    ids=["missing level", "missing cluster", "invalid level", "unknown variable", "zero workers", "kubernetes without image", "malformed cluster", "min above max", "zero timeout"],
)
def test_settings_refuse_missing_invalid_or_unknown_variables(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(SettingsError, match=re.escape(fragment)):
        load_settings(env)


def test_cluster_kinds_and_the_kubernetes_image() -> None:
    assert load_settings(BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "tcp://scheduler:8786"}).execution().cluster_kind == "address"
    kubernetes = load_settings(BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "kubernetes", "PORTFOLIO_OPTIMIZER_WORKER_IMAGE": "registry/optimizer:1.2"}).execution()
    assert (kubernetes.cluster_kind, kubernetes.worker_image) == ("kubernetes", "registry/optimizer:1.2")


def test_auto_cluster_resolves_on_the_kubernetes_marker_and_is_recorded_resolved() -> None:
    laptop = load_settings(BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "auto"})
    assert laptop.cluster == "local"
    assert laptop.shown()["cluster"] == "local"
    pod_env = BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "auto", "PORTFOLIO_OPTIMIZER_WORKER_IMAGE": "registry/optimizer:1.2", "KUBERNETES_SERVICE_HOST": "10.0.0.1"}
    assert load_settings(pod_env).cluster == "kubernetes"
    with pytest.raises(SettingsError, match="kubernetes requires PORTFOLIO_OPTIMIZER_WORKER_IMAGE"):
        load_settings(BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "auto", "KUBERNETES_SERVICE_HOST": "10.0.0.1"})
