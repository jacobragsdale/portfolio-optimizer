"""Tier 3: every public step in the shipped step modules resolves under the convention, every shipped kind is registered, and the modules import without side effects."""

import inspect
from types import ModuleType

import pytest

from portfolio_optimizer import checks, loaders, rules, sinks, solve_order, solvers
from portfolio_optimizer.config.models import StepSpec
from portfolio_optimizer.config.resolve import CONTRACTS, TEMPLATE_MODULES, StepKind, resolve_step
from portfolio_optimizer.config.schema import shipped_steps
from portfolio_optimizer.domain.constraints import SHIPPED_CONSTRAINT_KINDS, constraint_kinds
from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.objective import SHIPPED_TERM_KINDS, term_kinds
from portfolio_optimizer.domain.registry import kind_name
from portfolio_optimizer.domain.results import Artifact
from portfolio_optimizer.engine import build

MODULES: dict[str, StepKind] = {
    loaders.__name__: "loader",
    rules.__name__: "rule",
    solve_order.__name__: "solve_order",
    build.__name__: "build",
    solvers.__name__: "solve",
    sinks.__name__: "sink",
    checks.__name__: "check",
}
HELPERS: dict[str, frozenset[str]] = {rules.__name__: frozenset({"parameter", "restricted_flags"}), build.__name__: frozenset({"to_float64", "order_inputs"})}
"""Public functions in a step module that are not steps: helpers the shipped steps, the engine, and a desk's own steps share."""


def _public_functions(module: ModuleType) -> list[str]:
    return [name for name, value in vars(module).items() if not name.startswith("_") and inspect.isfunction(value) and value.__module__ == module.__name__]


@pytest.mark.parametrize("module", [loaders, rules, solve_order, build, solvers, sinks, checks], ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_shipped_function_resolves_under_the_convention_or_is_a_declared_helper(module: ModuleType) -> None:
    kind = MODULES[module.__name__]
    assert TEMPLATE_MODULES[kind] == module.__name__
    helpers = HELPERS.get(module.__name__, frozenset())
    names = [name for name in _public_functions(module) if name not in helpers]
    assert names, f"{module.__name__} ships no steps"
    assert set(shipped_steps(module, kind)) == set(names), "the schema generator sees exactly the steps, and none of the helpers"
    for name in names:
        fn = getattr(module, name)
        params_model = inspect.get_annotations(fn, eval_str=True).get("params")
        params = {field: (1 if info.annotation is int else "1") for field, info in params_model.model_fields.items() if info.is_required()} if params_model is not None else {}
        step = resolve_step(StepSpec.model_validate({"name": name, "params": params}), kind)
        assert step.qualname == f"{module.__name__}:{name}"
        assert not step.is_external
    for helper in helpers:
        returns = inspect.get_annotations(getattr(module, helper), eval_str=True).get("return")
        assert not any(returns == allowed for allowed in CONTRACTS[kind].returns), f"{helper} returns what a {kind} step returns; it is a step or the helper list is stale"


def test_every_shipped_kind_is_registered_under_its_own_name() -> None:
    assert {kind_name(model) for model in SHIPPED_CONSTRAINT_KINDS} <= set(constraint_kinds())
    assert {kind_name(model) for model in SHIPPED_TERM_KINDS} <= set(term_kinds())
    for model in (*SHIPPED_CONSTRAINT_KINDS, *SHIPPED_TERM_KINDS):
        assert model.__doc__, f"{model.__name__} needs a docstring: the schema and the GUI show it"
        assert all(field.description for name, field in model.model_fields.items()), f"{model.__name__}: every field needs a description"


def test_extension_modules_import_without_side_effects() -> None:
    for module in (loaders, rules, solvers, sinks, checks):
        assert module.__doc__ is not None
        assert "yours to edit" in module.__doc__
    assert PortfolioData is not None
    assert Artifact is not None
