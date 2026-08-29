"""Tier 2/3: sinks publish atomically, settings refuse incomplete environments, and every shipped step follows the convention."""

import inspect
import re
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from portfolio_optimizer import loaders, rules, sinks, terms
from portfolio_optimizer.config.models import StepSpec
from portfolio_optimizer.config.resolve import StepKind, resolve_step
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm
from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.results import Artifact
from portfolio_optimizer.domain.schemas import ORDERS
from portfolio_optimizer.engine.hashing import file_sha256
from portfolio_optimizer.settings import SettingsError, load_settings
from portfolio_optimizer.sinks import FileSinkParams, orders_to_csv, orders_to_parquet
from tests.conftest import Frames, io_context

# --- sinks ---


def test_parquet_sink_writes_atomically_and_reports_the_artifact(tmp_path: Path, frames: Frames) -> None:
    orders = frames.orders({"security_id": "A"}, {"security_id": "B", "quantity": 7, "notional": Decimal(700)})
    (artifact,) = orders_to_parquet(orders, io_context(tmp_path), FileSinkParams())
    path = Path(artifact.path)
    assert path == tmp_path / "run-test" / "orders" / "orders.parquet"
    assert artifact.sha256 == file_sha256(path)
    assert artifact.size_bytes == path.stat().st_size
    assert sorted(p.name for p in path.parent.iterdir()) == ["orders.parquet"]
    validate_frame(pd.read_parquet(path), ORDERS)


def test_csv_sink_writes_a_readable_file(tmp_path: Path, frames: Frames) -> None:
    (artifact,) = orders_to_csv(frames.orders(), io_context(tmp_path), FileSinkParams(subdir="human"))
    assert Path(artifact.path).read_text().startswith("portfolio_id,security_id,side,quantity")


# --- settings ---


BASE_ENV: dict[str, str] = {
    "PORTFOLIO_OPTIMIZER_OUTPUT_DIR": "/tmp/out",
    "PORTFOLIO_OPTIMIZER_DATA_ROOT": "/tmp/data",
    "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "DEBUG",
    "PORTFOLIO_OPTIMIZER_EXECUTOR": "process",
    "PORTFOLIO_OPTIMIZER_MAX_WORKERS": "4",
}
DASK_ENV: dict[str, str] = BASE_ENV | {
    "PORTFOLIO_OPTIMIZER_EXECUTOR": "dask",
    "PORTFOLIO_OPTIMIZER_CLUSTER": "local",
    "PORTFOLIO_OPTIMIZER_MIN_WORKERS": "2",
    "PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S": "60",
}


def test_settings_load_from_an_explicit_environment() -> None:
    settings = load_settings(BASE_ENV | {"HOME": "/x"})
    assert settings.output_dir == Path("/tmp/out")
    assert settings.log_level == "DEBUG"
    execution = settings.execution()
    assert (execution.executor, execution.max_workers, execution.cluster, execution.window) == ("process", 4, None, 8)
    assert settings.shown() == {"output_dir": "/tmp/out", "data_root": "/tmp/data", "log_level": "DEBUG", "executor": "process", "max_workers": "4"}


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({key: value for key, value in BASE_ENV.items() if key != "PORTFOLIO_OPTIMIZER_LOG_LEVEL"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL: Field required"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_LOG_LEVEL": "LOUD"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_TYPO": "1"}, "PORTFOLIO_OPTIMIZER_TYPO"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_MAX_WORKERS": "0"}, "PORTFOLIO_OPTIMIZER_MAX_WORKERS"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_EXECUTOR": "ray"}, "PORTFOLIO_OPTIMIZER_EXECUTOR"),
        (BASE_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "local"}, "PORTFOLIO_OPTIMIZER_CLUSTER only apply with PORTFOLIO_OPTIMIZER_EXECUTOR=dask"),
        (
            BASE_ENV | {"PORTFOLIO_OPTIMIZER_EXECUTOR": "dask"},
            "PORTFOLIO_OPTIMIZER_EXECUTOR=dask requires PORTFOLIO_OPTIMIZER_CLUSTER, PORTFOLIO_OPTIMIZER_MIN_WORKERS, PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S",
        ),
        (DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "kubernetes"}, "PORTFOLIO_OPTIMIZER_CLUSTER=kubernetes requires PORTFOLIO_OPTIMIZER_WORKER_IMAGE"),
        (DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "somewhere"}, "PORTFOLIO_OPTIMIZER_CLUSTER"),
        (DASK_ENV | {"PORTFOLIO_OPTIMIZER_MIN_WORKERS": "8"}, "PORTFOLIO_OPTIMIZER_MIN_WORKERS (8) exceeds PORTFOLIO_OPTIMIZER_MAX_WORKERS (4)"),
    ],
    ids=[
        "missing",
        "invalid level",
        "unknown variable",
        "zero workers",
        "unknown executor",
        "cluster without dask",
        "dask without cluster",
        "kubernetes without image",
        "malformed cluster",
        "min above max",
    ],
)
def test_settings_refuse_missing_invalid_or_unknown_variables(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(SettingsError, match=re.escape(fragment)):
        load_settings(env)


def test_dask_settings_carry_the_cluster_into_the_execution_settings() -> None:
    execution = load_settings(DASK_ENV | {"PORTFOLIO_OPTIMIZER_IMAGE_DIGEST": "sha256:abc"}).execution()
    assert (execution.cluster, execution.cluster_kind, execution.min_workers, execution.cluster_timeout_s, execution.image_digest) == ("local", "local", 2, 60.0, "sha256:abc")
    assert load_settings(DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "tcp://scheduler:8786"}).execution().cluster_kind == "address"
    kubernetes = load_settings(DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "kubernetes", "PORTFOLIO_OPTIMIZER_WORKER_IMAGE": "registry/optimizer:1.2"}).execution()
    assert (kubernetes.cluster_kind, kubernetes.worker_image) == ("kubernetes", "registry/optimizer:1.2")


def test_auto_cluster_resolves_on_the_kubernetes_marker_and_is_recorded_resolved() -> None:
    laptop = load_settings(DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "auto"})
    assert laptop.cluster == "local"
    assert laptop.shown()["cluster"] == "local"
    pod_env = DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "auto", "PORTFOLIO_OPTIMIZER_WORKER_IMAGE": "registry/optimizer:1.2", "KUBERNETES_SERVICE_HOST": "10.0.0.1"}
    assert load_settings(pod_env).cluster == "kubernetes"
    with pytest.raises(SettingsError, match="kubernetes requires PORTFOLIO_OPTIMIZER_WORKER_IMAGE"):
        load_settings(DASK_ENV | {"PORTFOLIO_OPTIMIZER_CLUSTER": "auto", "KUBERNETES_SERVICE_HOST": "10.0.0.1"})


# --- every shipped step follows the convention ---


def _kind_for(fn: object, module: ModuleType) -> StepKind:
    hints = inspect.get_annotations(fn, eval_str=True)  # ty: ignore[invalid-argument-type]  # every public attribute checked here is a function
    returns = hints.get("return")
    if module is loaders:
        return "constraints_loader" if returns is not pd.DataFrame else "loader"
    if module is rules:
        return "rule"
    if module is sinks:
        return "sink"
    if returns is ObjectiveTerm:
        return "term"
    if returns is ConstraintSet:
        return "constraint"
    msg = f"{fn!r} in {module.__name__} has an unexpected return annotation {returns!r}"
    raise AssertionError(msg)


def _public_step_functions(module: ModuleType) -> list[str]:
    names: list[str] = []
    for name, value in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(value) or value.__module__ != module.__name__:
            continue
        if module is terms and name == "adv_remaining":
            continue  # a helper shared with the verifier, not a step
        names.append(name)
    return names


@pytest.mark.parametrize("module", [loaders, rules, terms, sinks], ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_shipped_function_resolves_under_the_convention(module: ModuleType) -> None:
    names = _public_step_functions(module)
    assert names, f"{module.__name__} ships no steps"
    for name in names:
        fn = getattr(module, name)
        kind = _kind_for(fn, module)
        params_model = inspect.get_annotations(fn, eval_str=True).get("params")
        params = {field: (1 if info.annotation is int else "1") for field, info in params_model.model_fields.items() if info.is_required()} if params_model is not None else {}
        if params_model is not None and "path" in params:
            params["path"] = "x"
        step = resolve_step(StepSpec.model_validate({"name": name, "params": params}), kind)
        assert step.qualname == f"{module.__name__}:{name}"
        assert not step.is_external


def test_extension_modules_import_without_side_effects() -> None:
    for module in (loaders, rules, terms, sinks):
        assert module.__doc__ is not None
        assert "yours to edit" in module.__doc__
    assert PortfolioData is not None
    assert Artifact is not None
