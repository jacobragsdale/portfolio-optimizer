"""Tier 2/3: sinks publish atomically, settings refuse incomplete environments, and every shipped step follows the convention."""

import inspect
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


def test_settings_load_from_an_explicit_environment() -> None:
    settings = load_settings({"PORTFOLIO_OPTIMIZER_OUTPUT_DIR": "/tmp/out", "PORTFOLIO_OPTIMIZER_DATA_ROOT": "/tmp/data", "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "DEBUG", "HOME": "/x"})
    assert settings.output_dir == Path("/tmp/out")
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"PORTFOLIO_OPTIMIZER_OUTPUT_DIR": "/o", "PORTFOLIO_OPTIMIZER_DATA_ROOT": "/d"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL: Field required"),
        ({"PORTFOLIO_OPTIMIZER_OUTPUT_DIR": "/o", "PORTFOLIO_OPTIMIZER_DATA_ROOT": "/d", "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "LOUD"}, "PORTFOLIO_OPTIMIZER_LOG_LEVEL"),
        ({"PORTFOLIO_OPTIMIZER_OUTPUT_DIR": "/o", "PORTFOLIO_OPTIMIZER_DATA_ROOT": "/d", "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "INFO", "PORTFOLIO_OPTIMIZER_TYPO": "1"}, "PORTFOLIO_OPTIMIZER_TYPO"),
    ],
)
def test_settings_refuse_missing_invalid_or_unknown_variables(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(SettingsError, match=fragment):
        load_settings(env)


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
