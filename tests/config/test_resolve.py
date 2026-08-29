"""Tier 1: the step-resolution convention, table-tested against every way a function can violate it."""

import asyncio
import json
import sys
import types
from collections.abc import Iterator, Mapping
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pydantic import Field

from portfolio_optimizer.config.models import RunConfig, StepSpec
from portfolio_optimizer.config.resolve import ConfigResolutionError, ResolvedStep, StepKind, resolve_config, resolve_step
from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm
from portfolio_optimizer.domain.data import IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.results import Artifact, ChainState, ProblemSpec, SolveContext
from portfolio_optimizer.domain.types import Params
from tests.conftest import AS_OF

# --- functions that follow the convention (and ones that break it), registered as module "fake_steps" ---


class TiltParams(Params):
    strength: Decimal = Field(ge=0)
    column: str = "alpha"


def plain_rule(data: PortfolioData) -> PortfolioData:
    return data


def rule_with_params(data: PortfolioData, params: TiltParams) -> PortfolioData:
    return data.with_rule_applied(f"tilt:{params.strength}")


def chained_rule(data: PortfolioData, ctx: SolveContext) -> PortfolioData:
    return data.with_rule_applied(f"chained:{ctx.portfolios_done}")


def rule_wrong_data_annotation(data: pd.DataFrame) -> PortfolioData:  # the annotation is the case under test
    raise NotImplementedError


def rule_missing_data(params: TiltParams) -> PortfolioData:  # see above
    raise NotImplementedError


def rule_extra_parameter(data: PortfolioData, universe: pd.DataFrame) -> PortfolioData:  # see above
    raise NotImplementedError


def rule_var_kwargs(data: PortfolioData, **kwargs: object) -> PortfolioData:  # see above
    raise NotImplementedError


def rule_wrong_return(data: PortfolioData) -> pd.DataFrame:  # see above
    raise NotImplementedError


def rule_no_return_annotation(data: PortfolioData):  # noqa: ANN201  # the missing annotation is the case under test
    raise NotImplementedError


def rule_untyped_params(data: PortfolioData, params: dict[str, object]) -> PortfolioData:  # see above
    raise NotImplementedError


def rule_wrong_ctx(data: PortfolioData, ctx: ChainState) -> PortfolioData:  # see above
    raise NotImplementedError


def term(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:  # never invoked here
    raise NotImplementedError


def chained_constraint(x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:  # never invoked here
    raise NotImplementedError


def loader(request: LoadRequest) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [request.dataset]})


async def async_loader(request: LoadRequest) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [request.dataset]})


async def async_rule(data: PortfolioData) -> PortfolioData:  # async is the case under test
    return data


def constraints_loader(request: LoadRequest) -> dict[str, dict[str, object]]:  # noqa: ARG001  # never invoked here
    return {}


def sink(orders: pd.DataFrame, io: IoContext) -> tuple[Artifact, ...]:  # noqa: ARG001  # never invoked here
    return ()


NOT_A_FUNCTION = 42


@pytest.fixture
def fake_steps() -> Iterator[str]:
    module = types.ModuleType("fake_steps")
    for name, value in globals().items():
        if callable(value) or name == "NOT_A_FUNCTION":
            setattr(module, name, value)
    sys.modules["fake_steps"] = module
    yield "fake_steps"
    del sys.modules["fake_steps"]


def spec(name: str, **params: object) -> StepSpec:
    return StepSpec.model_validate_json(json.dumps({"name": name, "params": params}))


# --- resolve_step ---


def test_bare_name_resolves_in_the_template_module_and_types_its_params() -> None:
    step = resolve_step(spec("cap_single_name", max_weight="0.05"), "rule")
    assert step.qualname == "portfolio_optimizer.rules:cap_single_name"
    assert not step.is_external
    assert step.params is not None
    assert step.params.model_dump() == {"max_weight": Decimal("0.05")}
    assert step.context_name is None
    assert len(step.source_sha256) == 64


def test_qualified_name_resolves_an_external_module(fake_steps: str) -> None:
    step = resolve_step(spec(f"{fake_steps}:rule_with_params", strength="0.5"), "rule")
    assert step.is_external
    assert step.qualname == "fake_steps:rule_with_params"


def test_context_parameter_is_detected_by_name_and_type(fake_steps: str) -> None:
    assert resolve_step(spec(f"{fake_steps}:chained_rule"), "rule").context_name == "ctx"
    assert resolve_step(spec(f"{fake_steps}:chained_constraint"), "constraint").context_name == "chain"
    assert resolve_step(spec(f"{fake_steps}:term"), "term").context_name is None


VIOLATIONS: list[tuple[str, str, StepKind, Mapping[str, object], str]] = [
    ("unknown module", "no_such_module:fn", "rule", {}, "cannot import module"),
    ("unknown function", "fake_steps:no_such_function", "rule", {}, "has no function"),
    ("attribute is not a function", "fake_steps:NOT_A_FUNCTION", "rule", {}, "has no function"),
    ("wrong data annotation", "fake_steps:rule_wrong_data_annotation", "rule", {}, "'data' must be annotated PortfolioData, got DataFrame"),
    ("missing data parameter", "fake_steps:rule_missing_data", "rule", {}, "missing required parameter 'data'"),
    ("unexpected parameter", "fake_steps:rule_extra_parameter", "rule", {}, "unexpected parameter 'universe'; allowed: ['data', 'params', 'ctx']"),
    ("**kwargs", "fake_steps:rule_var_kwargs", "rule", {}, "no *args, **kwargs"),
    ("wrong return", "fake_steps:rule_wrong_return", "rule", {}, "return annotation must be PortfolioData, got DataFrame"),
    ("no return annotation", "fake_steps:rule_no_return_annotation", "rule", {}, "return annotation must be PortfolioData, got nothing"),
    ("params not a Params model", "fake_steps:rule_untyped_params", "rule", {}, "'params' must be annotated with a Params subclass"),
    ("wrong ctx annotation", "fake_steps:rule_wrong_ctx", "rule", {}, "'ctx' must be annotated SolveContext, got ChainState"),
    ("params given to a rule without params", "fake_steps:plain_rule", "rule", {"strength": "1"}, "does not take params, but the config supplies ['strength']"),
    ("params missing a required field", "fake_steps:rule_with_params", "rule", {}, "strength: Field required"),
    ("params with an unknown field", "fake_steps:rule_with_params", "rule", {"strength": "1", "tilt": "2"}, "tilt: Extra inputs are not permitted"),
    ("params with the wrong type", "fake_steps:rule_with_params", "rule", {"strength": "-1"}, "strength: Input should be greater than or equal to 0"),
    ("term used as a constraint", "fake_steps:term", "constraint", {}, "return annotation must be ConstraintSet, got ObjectiveTerm"),
    ("rule used as a loader", "fake_steps:plain_rule", "loader", {}, "unexpected parameter 'data'"),
    ("async rule", "fake_steps:async_rule", "rule", {}, "`async def` is only allowed for loaders"),
]


@pytest.mark.parametrize(("name", "kind", "params", "fragment"), [case[1:] for case in VIOLATIONS], ids=[case[0] for case in VIOLATIONS])
def test_convention_violations_are_reported(fake_steps: str, name: str, kind: StepKind, params: Mapping[str, object], fragment: str) -> None:
    del fake_steps
    with pytest.raises(ConfigResolutionError) as info:
        resolve_step(spec(name, **params), kind)
    assert any(fragment in failure for failure in info.value.failures), info.value.failures


def test_every_contract_kind_accepts_its_canonical_signature(fake_steps: str) -> None:
    pairs: list[tuple[str, StepKind]] = [
        ("loader", "portfolios"),
        ("loader", "loader"),
        ("constraints_loader", "constraints_loader"),
        ("term", "term"),
        ("chained_constraint", "constraint"),
        ("sink", "sink"),
    ]
    for name, kind in pairs:
        assert resolve_step(spec(f"{fake_steps}:{name}"), kind).kind == kind


def test_loaders_may_be_async_and_invoke_async_runs_both_styles(fake_steps: str) -> None:
    asynchronous = resolve_step(spec(f"{fake_steps}:async_loader"), "loader")
    synchronous = resolve_step(spec(f"{fake_steps}:loader"), "loader")
    assert asynchronous.is_async
    assert not synchronous.is_async
    request = LoadRequest(dataset="holdings", portfolio_ids=(), as_of=AS_OF, data_root=Path(), run_id="r")

    async def both() -> tuple[object, object]:
        return await asynchronous.invoke_async(request=request), await synchronous.invoke_async(request=request)

    from_async, from_thread = asyncio.run(both())
    assert isinstance(from_async, pd.DataFrame)
    assert isinstance(from_thread, pd.DataFrame)
    assert from_async["dataset"].tolist() == from_thread["dataset"].tolist() == ["holdings"]


def test_invoke_supplies_params_and_context(fake_steps: str, make: object) -> None:
    del make
    step = resolve_step(spec(f"{fake_steps}:rule_with_params", strength="0.5"), "rule")
    chained = resolve_step(spec(f"{fake_steps}:chained_rule"), "rule")
    plain_loader = resolve_step(spec(f"{fake_steps}:loader"), "loader")
    from tests.conftest import make_portfolio_data  # local import keeps the header about the unit under test

    data = make_portfolio_data()
    result = step.invoke(data=data)
    assert isinstance(result, PortfolioData)
    assert result.applied_rules == ("tilt:0.5",)
    chained_result = chained.invoke(data=data, context=SolveContext())
    assert isinstance(chained_result, PortfolioData)
    assert chained_result.applied_rules == ("chained:0",)
    with pytest.raises(ValueError, match="requires 'ctx'"):
        chained.invoke(data=data)
    frame = plain_loader.invoke(request=LoadRequest(dataset="holdings", portfolio_ids=(), as_of=data.as_of, data_root=Path(), run_id="r"))
    assert isinstance(frame, pd.DataFrame)


def test_source_hash_is_stable_and_function_specific(fake_steps: str) -> None:
    first = resolve_step(spec(f"{fake_steps}:plain_rule"), "rule")
    second = resolve_step(spec(f"{fake_steps}:plain_rule"), "rule")
    other = resolve_step(spec(f"{fake_steps}:chained_rule"), "rule")
    assert first.source_sha256 == second.source_sha256
    assert first.source_sha256 != other.source_sha256


# --- resolve_config ---


def fake_config(fake_steps: str, *, mode: str = "sequential", on_error: str = "fail_fast", rules: list[str] | None = None, constraints: list[str] | None = None) -> RunConfig:
    body = {
        "run": {"name": "r", "as_of": "2026-01-01T00:00:00Z"},
        "portfolios": f"{fake_steps}:loader",
        "datasets": {name: {"loader": f"{fake_steps}:loader"} for name in ("holdings", "universe", "details", "targets")} | {"constraints": {"loader": f"{fake_steps}:constraints_loader"}},
        "rules": rules if rules is not None else [f"{fake_steps}:plain_rule"],
        "objective": {"terms": [f"{fake_steps}:term"]},
        "constraints": constraints if constraints is not None else [],
        "sink": f"{fake_steps}:sink",
        "execution": {"mode": mode, "on_error": on_error},
    }
    return RunConfig.model_validate_json(json.dumps(body))


def test_resolve_config_resolves_every_step(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, constraints=[f"{fake_steps}:chained_constraint"]), config_sha256="abc")
    assert [step.kind for step in resolved.all_steps] == ["portfolios", "loader", "loader", "loader", "loader", "constraints_loader", "rule", "term", "constraint", "sink"]
    assert resolved.loaders["constraints"].kind == "constraints_loader"
    assert [step.qualname for step in resolved.chain_aware_steps] == ["fake_steps:chained_constraint"]


def test_resolve_config_reports_every_failing_step_at_once(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, rules=[f"{fake_steps}:rule_wrong_return", "no_such_rule"]), config_sha256="abc")
    assert len(info.value.failures) == 2
    assert info.value.failures[0].startswith("rules[0]: ")
    assert info.value.failures[1].startswith("rules[1]: ")


@pytest.mark.parametrize(
    ("mode", "on_error", "rules", "constraints", "fragment"),
    [
        ("parallel", "fail_fast", [], ["chained_constraint"], "'parallel' cannot run chain-aware steps"),
        ("parallel", "fail_fast", ["chained_rule"], [], "'parallel' cannot run chain-aware steps"),
        ("parallel_build_sequential_solve", "fail_fast", ["chained_rule"], [], "rules cannot take 'ctx'"),
        ("sequential", "continue", [], ["chained_constraint"], "'continue' is ambiguous with chain-aware steps"),
    ],
)
def test_execution_mode_is_checked_against_chain_aware_steps(fake_steps: str, mode: str, on_error: str, rules: list[str], constraints: list[str], fragment: str) -> None:
    config = fake_config(fake_steps, mode=mode, on_error=on_error, rules=[f"{fake_steps}:{r}" for r in rules], constraints=[f"{fake_steps}:{c}" for c in constraints])
    with pytest.raises(ConfigResolutionError, match=fragment):
        resolve_config(config, config_sha256="abc")


def test_chain_aware_steps_are_allowed_where_the_mode_supports_them(fake_steps: str) -> None:
    sequential = resolve_config(fake_config(fake_steps, rules=[f"{fake_steps}:chained_rule"], constraints=[f"{fake_steps}:chained_constraint"]), config_sha256="abc")
    assert len(sequential.chain_aware_steps) == 2
    pbss = resolve_config(fake_config(fake_steps, mode="parallel_build_sequential_solve", constraints=[f"{fake_steps}:chained_constraint"]), config_sha256="abc")
    assert len(pbss.chain_aware_steps) == 1


def test_resolved_step_is_a_plain_frozen_record(fake_steps: str) -> None:
    step = resolve_step(spec(f"{fake_steps}:plain_rule"), "rule")
    assert isinstance(step, ResolvedStep)
    with pytest.raises(AttributeError):
        step.name = "other"  # ty: ignore[invalid-assignment]  # assigning to a frozen field is the case under test
