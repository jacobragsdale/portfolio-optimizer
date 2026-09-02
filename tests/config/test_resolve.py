"""Tier 1: the step-resolution convention, table-tested against every way a function can violate it, and the terms parsed beside it."""

import asyncio
import importlib.metadata
import json
import sys
import types
from collections.abc import Iterator, Mapping
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pydantic import Field

from portfolio_optimizer.config import resolve as resolve_module
from portfolio_optimizer.config.models import RunConfig, config_sha256
from portfolio_optimizer.config.resolve import ConfigResolutionError, published_steps, resolve_config, resolve_step
from portfolio_optimizer.config.steps import ResolvedStep, StepKind
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.results import Artifact, ProblemSpec
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.engine.build import standard
from portfolio_optimizer.solving import SolveRequest, SolveResult
from tests import steps
from tests.conftest import AS_OF, NOOP_TERMS, SELL_TERMS, resolved_example_real, step_spec

# --- functions that follow the convention (and ones that break it), registered as module "fake_steps" ---


class TiltParams(Params):
    strength: Decimal = Field(ge=0)
    column: str = "alpha"


def plain_rule(data: PortfolioData) -> PortfolioData:
    return data


def rule_with_params(data: PortfolioData, params: TiltParams) -> PortfolioData:
    return data.with_rule_applied(f"tilt:{params.strength}")


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


def solve_step(request: SolveRequest) -> SolveResult:
    return SolveResult(w=request.spec.w0)


def solve_wrong_args(spec: ProblemSpec) -> SolveResult:  # the missing `request` is the case under test
    raise NotImplementedError


def build_step(data: PortfolioData) -> ProblemSpec:
    return standard(data)


def build_wrong_return(data: PortfolioData) -> PortfolioData:  # the annotation is the case under test
    raise NotImplementedError


def solve_order_step(data: PortfolioData) -> Decimal:
    return Decimal(len(data.holdings))


def solve_order_wrong_return(data: PortfolioData) -> float:  # the annotation is the case under test
    raise NotImplementedError


def loader(request: LoadRequest) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [request.dataset]})


async def async_loader(request: LoadRequest) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [request.dataset]})


async def async_rule(data: PortfolioData) -> PortfolioData:  # async is the case under test
    return data


def assembly_step(frames: Frames) -> Frames:
    return frames


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


# --- resolve_step ---


def test_bare_name_resolves_in_the_template_module_and_types_its_params() -> None:
    step = resolve_step(step_spec("cap_single_name", max_weight="0.05"), "rule")
    assert step.qualname == "portfolio_optimizer.rules:cap_single_name"
    assert not step.is_external
    assert step.params is not None
    assert step.params.model_dump() == {"max_weight": Decimal("0.05")}
    assert len(step.source_sha256) == 64


def test_qualified_name_resolves_an_external_module(fake_steps: str) -> None:
    step = resolve_step(step_spec(f"{fake_steps}:rule_with_params", strength="0.5"), "rule")
    assert step.is_external
    assert step.qualname == "fake_steps:rule_with_params"


def test_the_build_step_resolves_bare_in_the_engine_module_and_qualified_elsewhere(fake_steps: str) -> None:
    assert resolve_step(step_spec("standard"), "build").qualname == "portfolio_optimizer.engine.build:standard"
    assert resolve_step(step_spec(f"{fake_steps}:build_step"), "build").is_external


def test_a_bare_name_resolves_through_the_steps_installed_packages_publish(fake_steps: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point in the kind's group is how a package makes a step nameable without its module path; the template module still wins a name both have."""
    published = {
        "portfolio_optimizer.rule": [
            importlib.metadata.EntryPoint("tilt", f"{fake_steps}:rule_with_params", "portfolio_optimizer.rule"),
            importlib.metadata.EntryPoint("cap_single_name", f"{fake_steps}:plain_rule", "portfolio_optimizer.rule"),
        ]
    }
    monkeypatch.setattr(resolve_module.importlib.metadata, "entry_points", lambda group: published.get(group, []))
    assert published_steps("rule") == {"tilt": ("fake_steps", "rule_with_params"), "cap_single_name": ("fake_steps", "plain_rule")}
    tilt = resolve_step(step_spec("tilt", strength="1"), "rule")
    assert (tilt.qualname, tilt.is_external) == ("fake_steps:rule_with_params", True)
    assert resolve_step(step_spec("cap_single_name", max_weight="0.1"), "rule").qualname == "portfolio_optimizer.rules:cap_single_name", "the template module is looked up first"


def test_a_qualified_name_outside_the_allowed_packages_is_refused(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError, match=r"package 'fake_steps' is not among the step packages the settings allow \['my_firm'\]"):
        resolve_step(step_spec(f"{fake_steps}:plain_rule"), "rule", packages=("my_firm",))
    assert resolve_step(step_spec(f"{fake_steps}:plain_rule"), "rule", packages=("fake_steps",)).qualname == "fake_steps:plain_rule"
    assert resolve_step(step_spec("portfolio_optimizer.rules:add_zero_alpha"), "rule", packages=("my_firm",)).qualname == "portfolio_optimizer.rules:add_zero_alpha", (
        "the template's own package is always allowed"
    )


VIOLATIONS: list[tuple[str, str, StepKind, Mapping[str, object], str]] = [
    ("unknown module", "no_such_module:fn", "rule", {}, "cannot import module"),
    ("unknown function", "fake_steps:no_such_function", "rule", {}, "has no function"),
    ("unknown bare name", "no_such_rule", "rule", {}, "'portfolio_optimizer.rules' has no function 'no_such_rule'"),
    ("attribute is not a function", "fake_steps:NOT_A_FUNCTION", "rule", {}, "has no function"),
    ("wrong data annotation", "fake_steps:rule_wrong_data_annotation", "rule", {}, "'data' must be annotated PortfolioData, got DataFrame"),
    ("missing data parameter", "fake_steps:rule_missing_data", "rule", {}, "missing required parameter 'data'"),
    ("unexpected parameter", "fake_steps:rule_extra_parameter", "rule", {}, "unexpected parameter 'universe'; allowed: ['data', 'params']"),
    ("**kwargs", "fake_steps:rule_var_kwargs", "rule", {}, "no *args, **kwargs"),
    ("wrong return", "fake_steps:rule_wrong_return", "rule", {}, "return annotation must be PortfolioData, got DataFrame"),
    ("no return annotation", "fake_steps:rule_no_return_annotation", "rule", {}, "return annotation must be PortfolioData, got nothing"),
    ("params not a Params model", "fake_steps:rule_untyped_params", "rule", {}, "'params' must be annotated with a Params subclass"),
    ("solve-order step returning float", "fake_steps:solve_order_wrong_return", "solve_order", {}, "return annotation must be Decimal, got float"),
    ("build step returning the bundle", "fake_steps:build_wrong_return", "build", {}, "return annotation must be ProblemSpec, got PortfolioData"),
    ("params given to a rule without params", "fake_steps:plain_rule", "rule", {"strength": "1"}, "does not take params, but the config supplies ['strength']"),
    ("params missing a required field", "fake_steps:rule_with_params", "rule", {}, "strength: Field required"),
    ("params with an unknown field", "fake_steps:rule_with_params", "rule", {"strength": "1", "tilt": "2"}, "tilt: Extra inputs are not permitted"),
    ("params with the wrong type", "fake_steps:rule_with_params", "rule", {"strength": "-1"}, "strength: Input should be greater than or equal to 0"),
    ("a loader param outside its bound", "load_universe", "loader", {"min_latency_s": -1}, "min_latency_s: Input should be greater than or equal to 0"),
    ("rule used as a loader", "fake_steps:plain_rule", "loader", {}, "unexpected parameter 'data'"),
    ("async rule", "fake_steps:async_rule", "rule", {}, "`async def` is only allowed for loaders"),
]


@pytest.mark.parametrize(("name", "kind", "params", "fragment"), [case[1:] for case in VIOLATIONS], ids=[case[0] for case in VIOLATIONS])
def test_convention_violations_are_reported(fake_steps: str, name: str, kind: StepKind, params: Mapping[str, object], fragment: str) -> None:
    del fake_steps
    with pytest.raises(ConfigResolutionError) as info:
        resolve_step(step_spec(name, **params), kind)
    assert any(fragment in failure for failure in info.value.failures), info.value.failures


def test_every_contract_kind_accepts_its_canonical_signature(fake_steps: str) -> None:
    pairs: list[tuple[str, StepKind]] = [
        ("loader", "loader"),
        ("assembly_step", "assembly"),
        ("plain_rule", "rule"),
        ("solve_order_step", "solve_order"),
        ("build_step", "build"),
        ("solve_step", "solve"),
        ("sink", "sink"),
    ]
    for name, kind in pairs:
        assert resolve_step(step_spec(f"{fake_steps}:{name}"), kind).kind == kind


def test_loaders_may_be_async_and_invoke_async_runs_both_styles(fake_steps: str) -> None:
    asynchronous = resolve_step(step_spec(f"{fake_steps}:async_loader"), "loader")
    synchronous = resolve_step(step_spec(f"{fake_steps}:loader"), "loader")
    assert asynchronous.is_async
    assert not synchronous.is_async
    request = LoadRequest(dataset="holdings", portfolio_ids=(), as_of_date=AS_OF, data_root=Path(), run_id="r")

    async def both() -> tuple[object, object]:
        return await asynchronous.invoke_async(request=request), await synchronous.invoke_async(request=request)

    from_async, from_thread = asyncio.run(both())
    assert isinstance(from_async, pd.DataFrame)
    assert isinstance(from_thread, pd.DataFrame)
    assert from_async["dataset"].tolist() == from_thread["dataset"].tolist() == ["holdings"]


def test_invoke_supplies_params(fake_steps: str) -> None:
    step = resolve_step(step_spec(f"{fake_steps}:rule_with_params", strength="0.5"), "rule")
    plain_loader = resolve_step(step_spec(f"{fake_steps}:loader"), "loader")
    from tests.conftest import make_portfolio_data  # local import keeps the header about the unit under test

    data = make_portfolio_data()
    result = step.invoke(data=data)
    assert isinstance(result, PortfolioData)
    assert result.applied_rules == ("tilt:0.5",)
    frame = plain_loader.invoke(request=LoadRequest(dataset="holdings", portfolio_ids=(), as_of_date=data.as_of_date, data_root=Path(), run_id="r"))
    assert isinstance(frame, pd.DataFrame)


def test_source_hash_is_stable_and_function_specific(fake_steps: str) -> None:
    first = resolve_step(step_spec(f"{fake_steps}:plain_rule"), "rule")
    second = resolve_step(step_spec(f"{fake_steps}:plain_rule"), "rule")
    other = resolve_step(step_spec(f"{fake_steps}:rule_with_params", strength="1"), "rule")
    assert first.source_sha256 == second.source_sha256
    assert first.source_sha256 != other.source_sha256


# --- resolve_config ---


def fake_config(
    fake_steps: str, *, on_error: str = "fail_fast", rules: list[str] | None = None, solve_order: str | None = None, solve: object = None, objective: list[object] | None = None
) -> RunConfig:
    body: dict[str, object] = {
        "run": {"name": "r"},
        "order_flow": "inflow",
        "datasets": {name: {"loader": f"{fake_steps}:loader"} for name in ("portfolios", "holdings", "universe", "details")},
        "rules": rules if rules is not None else [f"{fake_steps}:plain_rule"],
        "objective": objective if objective is not None else NOOP_TERMS,
        "sink": f"{fake_steps}:sink",
        "execution": {"on_error": on_error},
    }
    if solve_order is not None:
        body["solve_order"] = solve_order
    if solve is not None:
        body["solve"] = solve
    return RunConfig.model_validate_json(json.dumps(body))


def test_resolve_config_resolves_every_step_and_parses_every_term(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, objective=[*NOOP_TERMS, {"kind": "chain_penalty", "name": "crowding"}], solve_order=f"{fake_steps}:solve_order_step"))
    assert [step.kind for step in resolved.all_steps] == ["loader", "loader", "loader", "loader", "rule", "solve_order", "build", "solve", "sink"]
    assert resolved.config_sha256 == config_sha256(resolved.config)
    assert resolved.solve.qualname == "portfolio_optimizer.solvers:cvxpy" and resolved.shipped_solve, "the default solve step is the shipped cvxpy one"
    assert resolved.build.qualname == "portfolio_optimizer.engine.build:standard"
    assert {step.kind for step in resolved.loaders.values()} == {"loader"}, "every dataset is a frame, so every loader resolves under the one contract"
    assert resolved.solve_order is not None and resolved.solve_order.qualname == "fake_steps:solve_order_step"
    assert [type(term).__name__ for term in resolved.terms] == ["Linear", "ChainPenalty"]
    assert [term.name for term in resolved.chain_aware_terms] == ["crowding"], "a kind that declares it reads the chain is what makes the run couple"
    assert resolve_config(fake_config(fake_steps)).solve_order is None


def test_resolve_config_reports_every_failing_step_at_once(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, rules=[f"{fake_steps}:rule_wrong_return", "no_such_rule"]))
    assert len(info.value.failures) == 2
    assert info.value.failures[0].startswith("rules[0]: ")
    assert info.value.failures[1].startswith("rules[1]: ")


def test_continue_is_allowed_with_chain_aware_terms(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, on_error="continue", objective=[{"kind": "chain_penalty", "name": "crowding"}]))
    assert len(resolved.chain_aware_terms) == 1
    assert resolved.config.execution.dependencies == "overlap"


@pytest.mark.parametrize(
    ("params", "installed", "failure"),
    [
        ({"solver": "CLARABEL", "time_limit_s": 5.0}, ("CLARABEL", "SCIPY"), None),
        ({"solver": "SCIPY"}, ("CLARABEL", "SCIPY"), "solve: solver 'SCIPY' is not one the adapter knows; known: ['CLARABEL', 'HIGHS', 'OSQP', 'PIQP', 'SCS']"),
        ({"solver": "OSQP"}, ("CLARABEL", "SCIPY"), "solve: solver 'OSQP' is not installed in this environment; installed: ['CLARABEL']"),
        ({"solver": "PIQP", "time_limit_s": 5.0}, ("PIQP",), "solve: solver 'PIQP' has no time-limit option; remove time_limit_s"),
    ],
    ids=["installed", "cvxpy-has-it-but-the-adapter-does-not", "not-installed", "no-time-limit-option"],
)
def test_resolve_config_checks_the_cvxpy_steps_solver_against_what_this_process_has_installed(fake_steps: str, params: dict[str, object], installed: tuple[str, ...], failure: str | None) -> None:
    config = fake_config(fake_steps, solve={"name": "cvxpy", "params": params})
    if failure is None:
        assert resolve_config(config, installed=lambda: installed).solve.params_json["solver"] == params["solver"]
        return
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(config, installed=lambda: installed)
    assert info.value.failures == (failure,)


def test_a_solve_step_that_is_not_cvxpy_needs_no_solver_and_no_objective(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, solve=f"{fake_steps}:solve_step", objective=[]), installed=lambda: ())
    assert resolved.terms == () and not resolved.shipped_solve, "a pure function minimizes nothing, and no cvxpy solver is asked for"
    with pytest.raises(ConfigResolutionError, match="objective: the cvxpy solve step minimizes the terms' sum and needs at least one"):
        resolve_config(fake_config(fake_steps, objective=[]))


def test_a_term_reading_a_side_the_run_lacks_fails_dry_rendering_naming_the_side() -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolved_example_real(objective=SELL_TERMS)
    assert info.value.failures == (
        "objective[1]: tax_cost: rendering failed: SideUnavailableError: order flow 'inflow' has no 'sell' vector; this term or constraint reads x.sell, so it cannot run under order_flow='inflow'",
    )
    assert resolved_example_real(order_flow="outflow", objective=SELL_TERMS).profile.order_flow == "outflow"


def test_dry_rendering_surfaces_a_term_that_raises_and_skips_one_that_needs_data(fake_steps: str) -> None:
    assert steps.Raising.reads_chain is False  # imported for its side effect: the kinds below are registered
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, objective=[{"kind": "raising", "name": "risk"}, {"kind": "linear", "name": "momentum", "column": "momentum"}]))
    assert info.value.failures == ("objective[0]: risk: rendering failed: RuntimeError: no such column 'beta' in the risk model",)


def test_a_term_rendering_the_wrong_type_fails_dry_rendering() -> None:
    with pytest.raises(ConfigResolutionError, match=r"objective\[0\]: lie: rendered ConstraintSet, expected ObjectiveTerm"):
        resolved_example_real(objective=[{"kind": "lying", "name": "lie"}])


def test_a_term_that_is_not_convex_fails_dry_rendering(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError, match="objective: the objective and constraints are not DCP-compliant"):
        resolve_config(fake_config(fake_steps, objective=[{"kind": "quadratic", "name": "risk", "column": "w0", "weight": "-1"}]))


@pytest.mark.parametrize(
    ("objective", "fragment"),
    [
        ([{"kind": "no_such_kind", "name": "x"}], "objective[0]: unknown kind 'no_such_kind'; known kinds: ["),
        ([{"kind": "linear", "name": "alpha", "colour": "alpha"}], "objective[0]: linear: colour: Extra inputs are not permitted"),
        ([{"kind": "linear", "name": "twice"}, {"kind": "linear", "name": "twice"}], "objective[1]: name 'twice' is also used by objective[0]"),
    ],
    ids=["unknown kind", "unknown field", "duplicate name"],
)
def test_malformed_terms_are_reported_by_position(fake_steps: str, objective: list[object], fragment: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, objective=objective))
    assert any(fragment in failure for failure in info.value.failures), info.value.failures


def test_the_solve_step_is_resolved_against_its_contract(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, solve=f"{fake_steps}:solve_step"))
    assert resolved.solve.qualname == "fake_steps:solve_step" and resolved.solve.is_external
    with pytest.raises(ConfigResolutionError, match=r"solve: fake_steps:solve_wrong_args: unexpected parameter 'spec'.*missing required parameter 'request'"):
        resolve_config(fake_config(fake_steps, solve=f"{fake_steps}:solve_wrong_args"))


def test_a_config_cannot_name_constraints_at_all(fake_steps: str) -> None:
    body = json.loads(fake_config(fake_steps).model_dump_json(by_alias=True, exclude_none=True))
    with pytest.raises(ValueError, match="extra_forbidden"):
        RunConfig.model_validate_json(json.dumps({**body, "constraints": ["long_only"]}))
    assert resolve_config(fake_config(fake_steps)).profile.order_flow == "inflow", "the trade identity still comes from order_flow"


def test_a_solver_failure_is_reported_together_with_the_step_failures(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, rules=["no_such_rule"], solve={"name": "cvxpy", "params": {"solver": "OSQP"}}), installed=lambda: ("CLARABEL",))
    assert [failure.partition(":")[0] for failure in info.value.failures] == ["rules[0]", "solve"]


def test_a_failing_solve_order_step_is_reported_under_its_own_key(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError, match=r"solve_order: .*return annotation must be Decimal"):
        resolve_config(fake_config(fake_steps, solve_order=f"{fake_steps}:solve_order_wrong_return"))


def test_resolved_step_is_a_plain_frozen_record(fake_steps: str) -> None:
    step = resolve_step(step_spec(f"{fake_steps}:plain_rule"), "rule")
    assert isinstance(step, ResolvedStep)
    with pytest.raises(AttributeError):
        step.name = "other"  # ty: ignore[invalid-assignment]  # assigning to a frozen field is the case under test
