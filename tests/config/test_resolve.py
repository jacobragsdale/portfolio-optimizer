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

from portfolio_optimizer.config.models import RunConfig, StepSpec, config_sha256
from portfolio_optimizer.config.resolve import ConfigResolutionError, resolve_config, resolve_step
from portfolio_optimizer.config.steps import ResolvedStep, StepKind
from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, at_most, scale, total
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.results import Artifact, ChainState, ProblemSpec
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.solving import SolveRequest, SolveResult
from tests.conftest import AS_OF, BUY_ONLY_OBJECTIVE, resolved_example_real

# --- functions that follow the convention (and ones that break it), registered as module "fake_steps" ---


class TiltParams(Params):
    strength: Decimal = Field(ge=0)
    column: str = "alpha"


def plain_rule(data: PortfolioData) -> PortfolioData:
    return data


def rule_with_params(data: PortfolioData, params: TiltParams) -> PortfolioData:
    return data.with_rule_applied(f"tilt:{params.strength}")


def chained_rule(data: PortfolioData, ctx: ChainState) -> PortfolioData:  # rules never see the chain; the parameter is the case under test
    raise NotImplementedError


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


def constraint_that_raises(x: DecisionVars, spec: ProblemSpec) -> ConstraintSet:  # noqa: ARG001  # raising at construction is the case under test
    msg = "no such column 'beta' in the risk model"
    raise RuntimeError(msg)


def term_needing_a_column(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:
    from portfolio_optimizer.cvx.adapter import dot  # local: the header stays about the resolver

    return ObjectiveTerm("needs_column", dot(spec.column("momentum"), x.w))


def solve_order_step(data: PortfolioData) -> Decimal:
    return Decimal(len(data.holdings))


def solve_order_wrong_return(data: PortfolioData) -> float:  # the annotation is the case under test
    raise NotImplementedError


def term(x: DecisionVars, spec: ProblemSpec) -> ObjectiveTerm:
    del spec
    return ObjectiveTerm("term", scale(0.0, total(x.w)))


def chained_constraint(x: DecisionVars, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:
    del spec
    return ConstraintSet("chained", (at_most(x.coupled, chain.traded_shares + 1.0),))


def loader(request: LoadRequest) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [request.dataset]})


async def async_loader(request: LoadRequest) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [request.dataset]})


async def async_rule(data: PortfolioData) -> PortfolioData:  # async is the case under test
    return data


def constraints_loader(request: LoadRequest) -> dict[str, dict[str, object]]:  # noqa: ARG001  # never invoked here
    return {}


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


def spec(name: str, **params: object) -> StepSpec:
    return StepSpec.model_validate_json(json.dumps({"name": name, "params": params}))


# --- resolve_step ---


def test_bare_name_resolves_in_the_template_module_and_types_its_params() -> None:
    step = resolve_step(spec("cap_single_name", max_weight="0.05"), "rule")
    assert step.qualname == "portfolio_optimizer.rules:cap_single_name"
    assert not step.is_external
    assert step.params is not None
    assert step.params.model_dump() == {"max_weight": Decimal("0.05")}
    assert not step.reads_chain
    assert len(step.source_sha256) == 64


def test_qualified_name_resolves_an_external_module(fake_steps: str) -> None:
    step = resolve_step(spec(f"{fake_steps}:rule_with_params", strength="0.5"), "rule")
    assert step.is_external
    assert step.qualname == "fake_steps:rule_with_params"


def test_the_chain_parameter_is_detected_by_name_and_type(fake_steps: str) -> None:
    assert not resolve_step(spec(f"{fake_steps}:plain_rule"), "rule").reads_chain
    assert resolve_step(spec(f"{fake_steps}:chained_constraint"), "constraint").reads_chain
    assert not resolve_step(spec(f"{fake_steps}:term"), "term").reads_chain


VIOLATIONS: list[tuple[str, str, StepKind, Mapping[str, object], str]] = [
    ("unknown module", "no_such_module:fn", "rule", {}, "cannot import module"),
    ("unknown function", "fake_steps:no_such_function", "rule", {}, "has no function"),
    ("attribute is not a function", "fake_steps:NOT_A_FUNCTION", "rule", {}, "has no function"),
    ("wrong data annotation", "fake_steps:rule_wrong_data_annotation", "rule", {}, "'data' must be annotated PortfolioData, got DataFrame"),
    ("missing data parameter", "fake_steps:rule_missing_data", "rule", {}, "missing required parameter 'data'"),
    ("unexpected parameter", "fake_steps:rule_extra_parameter", "rule", {}, "unexpected parameter 'universe'; allowed: ['data', 'params']"),
    ("**kwargs", "fake_steps:rule_var_kwargs", "rule", {}, "no *args, **kwargs"),
    ("wrong return", "fake_steps:rule_wrong_return", "rule", {}, "return annotation must be PortfolioData, got DataFrame"),
    ("no return annotation", "fake_steps:rule_no_return_annotation", "rule", {}, "return annotation must be PortfolioData, got nothing"),
    ("params not a Params model", "fake_steps:rule_untyped_params", "rule", {}, "'params' must be annotated with a Params subclass"),
    ("rule asking for the chain", "fake_steps:chained_rule", "rule", {}, "unexpected parameter 'ctx'; allowed: ['data', 'params']"),
    ("solve-order step returning float", "fake_steps:solve_order_wrong_return", "solve_order", {}, "return annotation must be Decimal, got float"),
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
        ("loader", "loader"),
        ("constraints_loader", "constraints_loader"),
        ("assembly_step", "assembly"),
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


def test_invoke_supplies_params_and_the_chain(fake_steps: str) -> None:
    step = resolve_step(spec(f"{fake_steps}:rule_with_params", strength="0.5"), "rule")
    chained = resolve_step(spec(f"{fake_steps}:chained_constraint"), "constraint")
    plain_loader = resolve_step(spec(f"{fake_steps}:loader"), "loader")
    from tests.conftest import make_portfolio_data  # local import keeps the header about the unit under test

    data = make_portfolio_data()
    result = step.invoke(data=data)
    assert isinstance(result, PortfolioData)
    assert result.applied_rules == ("tilt:0.5",)
    with pytest.raises(ValueError, match="reads the chain but none was supplied"):
        chained.invoke(x=None, spec=None)
    frame = plain_loader.invoke(request=LoadRequest(dataset="holdings", portfolio_ids=(), as_of=data.as_of, data_root=Path(), run_id="r"))
    assert isinstance(frame, pd.DataFrame)


def test_source_hash_is_stable_and_function_specific(fake_steps: str) -> None:
    first = resolve_step(spec(f"{fake_steps}:plain_rule"), "rule")
    second = resolve_step(spec(f"{fake_steps}:plain_rule"), "rule")
    other = resolve_step(spec(f"{fake_steps}:rule_with_params", strength="1"), "rule")
    assert first.source_sha256 == second.source_sha256
    assert first.source_sha256 != other.source_sha256


# --- resolve_config ---


def fake_config(
    fake_steps: str,
    *,
    on_error: str = "fail_fast",
    rules: list[str] | None = None,
    constraints: list[object] | None = None,
    solve_order: str | None = None,
    solver: dict[str, object] | None = None,
    solve: str | None = None,
    terms: list[str] | None = None,
) -> RunConfig:
    body: dict[str, object] = {
        "run": {"name": "r", "as_of": "2026-01-01T00:00:00Z"},
        "portfolios": f"{fake_steps}:loader",
        "datasets": {name: {"loader": f"{fake_steps}:loader"} for name in ("holdings", "universe", "details", "targets")} | {"constraints": {"loader": f"{fake_steps}:constraints_loader"}},
        "rules": rules if rules is not None else [f"{fake_steps}:plain_rule"],
        "objective": {"terms": terms if terms is not None else [f"{fake_steps}:term"]},
        "constraints": constraints if constraints is not None else [],
        "sink": f"{fake_steps}:sink",
        "execution": {"on_error": on_error},
    }
    if solve_order is not None:
        body["solve_order"] = solve_order
    if solver is not None:
        body["solver"] = solver
    if solve is not None:
        body["solve"] = solve
    return RunConfig.model_validate_json(json.dumps(body))


def test_resolve_config_resolves_every_step(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, constraints=[f"{fake_steps}:chained_constraint"], solve_order=f"{fake_steps}:solve_order_step"))
    assert [step.kind for step in resolved.all_steps] == ["loader", "loader", "loader", "loader", "loader", "constraints_loader", "rule", "solve_order", "term", "constraint", "solve", "sink"]
    assert resolved.config_sha256 == config_sha256(resolved.config)
    assert resolved.solve.qualname == "portfolio_optimizer.solvers:cvxpy", "the default solve step is the shipped cvxpy one"
    assert resolved.loaders["constraints"].kind == "constraints_loader"
    assert resolved.solve_order is not None and resolved.solve_order.qualname == "fake_steps:solve_order_step"
    assert [step.qualname for step in resolved.chain_aware_steps] == ["fake_steps:chained_constraint"]
    assert resolve_config(fake_config(fake_steps)).solve_order is None


def test_resolve_config_reports_every_failing_step_at_once(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, rules=[f"{fake_steps}:rule_wrong_return", "no_such_rule"]))
    assert len(info.value.failures) == 2
    assert info.value.failures[0].startswith("rules[0]: ")
    assert info.value.failures[1].startswith("rules[1]: ")


def test_continue_is_allowed_with_chain_aware_steps(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, on_error="continue", constraints=[f"{fake_steps}:chained_constraint"]))
    assert len(resolved.chain_aware_steps) == 1
    assert resolved.config.execution.dependencies == "overlap"


@pytest.mark.parametrize(
    ("solver", "installed", "failure"),
    [
        ({"name": "CLARABEL", "time_limit_s": 5.0}, ("CLARABEL", "SCIPY"), None),
        ({"name": "SCIPY"}, ("CLARABEL", "SCIPY"), "solver: solver 'SCIPY' is not one the adapter knows; known: ['CLARABEL', 'HIGHS', 'OSQP', 'PIQP', 'SCS']"),
        ({"name": "OSQP"}, ("CLARABEL", "SCIPY"), "solver: solver 'OSQP' is not installed in this environment; installed: ['CLARABEL']"),
        ({"name": "PIQP", "time_limit_s": 5.0}, ("PIQP",), "solver: solver 'PIQP' has no time-limit option; remove solver.time_limit_s"),
    ],
    ids=["installed", "cvxpy-has-it-but-the-adapter-does-not", "not-installed", "no-time-limit-option"],
)
def test_resolve_config_checks_the_solver_against_what_this_process_has_installed(fake_steps: str, solver: dict[str, object], installed: tuple[str, ...], failure: str | None) -> None:
    config = fake_config(fake_steps, solver=solver)
    if failure is None:
        assert resolve_config(config, installed=lambda: installed).config.solver.name == solver["name"]
        return
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(config, installed=lambda: installed)
    assert info.value.failures == (failure,)


def test_a_term_reading_a_side_the_run_lacks_fails_dry_construction_naming_the_side() -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolved_example_real(sides="buy")
    assert info.value.failures == (
        "objective.terms[1]: portfolio_optimizer.terms:tax_cost: construction failed: SideUnavailableError: a 'buy' run has no 'sell' vector; this term or constraint reads x.sell, so it cannot run under sides='buy'",
    )
    assert resolved_example_real(sides="buy", objective=BUY_ONLY_OBJECTIVE).profile.sides == "buy", "every shipped constraint constructs on one side: they read trade and coupled, not sell"


def test_dry_construction_surfaces_a_step_that_raises_and_skips_one_that_needs_data(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, constraints=[f"{fake_steps}:constraint_that_raises"], terms=[f"{fake_steps}:term_needing_a_column"]))
    assert info.value.failures == ("constraints[0]: fake_steps:constraint_that_raises: construction failed: RuntimeError: no such column 'beta' in the risk model",)
    assert resolve_config(fake_config(fake_steps, terms=["tracking_error"], constraints=["long_only", "cumulative_adv_participation"])).chain_aware_steps


def test_a_term_returning_the_wrong_type_fails_dry_construction() -> None:
    with pytest.raises(ConfigResolutionError, match=r"lying_term: returned ConstraintSet, expected ObjectiveTerm"):
        resolved_example_real(objective={"terms": ["tests.conftest:lying_term"]})


def test_the_solve_step_is_resolved_against_its_contract(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, solve=f"{fake_steps}:solve_step"))
    assert resolved.solve.qualname == "fake_steps:solve_step" and resolved.solve.is_external
    with pytest.raises(ConfigResolutionError, match=r"solve: fake_steps:solve_wrong_args: unexpected parameter 'spec'.*missing required parameter 'request'"):
        resolve_config(fake_config(fake_steps, solve=f"{fake_steps}:solve_wrong_args"))


def test_constraints_resolve_under_unique_labels_that_default_to_the_bare_name(fake_steps: str) -> None:
    resolved = resolve_config(fake_config(fake_steps, constraints=[f"{fake_steps}:chained_constraint", {"name": f"{fake_steps}:chained_constraint", "label": "adv_again"}]))
    assert [(constraint.label, constraint.reads_chain, constraint.qualname) for constraint in resolved.constraints] == [
        ("chained_constraint", True, "fake_steps:chained_constraint"),
        ("adv_again", True, "fake_steps:chained_constraint"),
    ]
    assert resolved.constraints[1].spec.kind == "function"
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, constraints=[f"{fake_steps}:chained_constraint", f"{fake_steps}:chained_constraint"]))
    assert info.value.failures == ("constraints[1]: label 'chained_constraint' is also used by constraints[0]; give one of them a `label`",)


def test_trade_balance_is_refused_by_name_because_sides_supplies_the_identity(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, constraints=["trade_balance"]))
    assert info.value.failures == ("constraints[0]: 'trade_balance' is not a configurable constraint; the trade identity comes from `sides` ('both') — remove it",)
    assert resolve_config(fake_config(fake_steps)).profile.sides == "both"


def test_a_solver_failure_is_reported_together_with_the_step_failures(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError) as info:
        resolve_config(fake_config(fake_steps, rules=["no_such_rule"], solver={"name": "OSQP"}), installed=lambda: ("CLARABEL",))
    assert [failure.partition(":")[0] for failure in info.value.failures] == ["solver", "rules[0]"]


def test_a_failing_solve_order_step_is_reported_under_its_own_key(fake_steps: str) -> None:
    with pytest.raises(ConfigResolutionError, match=r"solve_order: .*return annotation must be Decimal"):
        resolve_config(fake_config(fake_steps, solve_order=f"{fake_steps}:solve_order_wrong_return"))


def test_resolved_step_is_a_plain_frozen_record(fake_steps: str) -> None:
    step = resolve_step(spec(f"{fake_steps}:plain_rule"), "rule")
    assert isinstance(step, ResolvedStep)
    with pytest.raises(AttributeError):
        step.name = "other"  # ty: ignore[invalid-assignment]  # assigning to a frozen field is the case under test
