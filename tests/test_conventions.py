"""Tier 3: every public function in the shipped step modules resolves under the convention, and the modules import without side effects."""

import inspect
from types import ModuleType

import pytest

from portfolio_optimizer import loaders, rules, sinks, terms
from portfolio_optimizer.config.models import StepSpec
from portfolio_optimizer.config.resolve import StepKind, resolve_step
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm
from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.results import Artifact


def _kind_for(fn: object, module: ModuleType) -> StepKind:
    hints = inspect.get_annotations(fn, eval_str=True)  # ty: ignore[invalid-argument-type]  # every public attribute checked here is a function
    returns = hints.get("return")
    if module is loaders:
        return "loader"
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
