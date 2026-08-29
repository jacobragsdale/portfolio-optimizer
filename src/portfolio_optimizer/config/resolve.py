"""Resolve step names in the run config to functions, and check them before any data loads.

The convention: a step is an ordinary function whose signature carries engine-provided
arguments by fixed names (``data``, ``request``, ``x``, ``spec``, ``orders``, ``io``), an optional
``params`` argument annotated with a :class:`~portfolio_optimizer.domain.types.Params` subclass,
and an optional context argument (``ctx`` for rules, ``chain`` for terms and constraints). The
engine calls steps with keyword arguments, so the order does not matter.
"""

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_type_hints

import pandas as pd
from pydantic import ValidationError

from portfolio_optimizer.config.models import RunConfig, StepSpec
from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm
from portfolio_optimizer.domain.data import IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.results import Artifact, ChainState, ProblemSpec, SolveContext
from portfolio_optimizer.domain.types import Params

type StepKind = Literal["portfolios", "loader", "constraints_loader", "rule", "term", "constraint", "sink"]

TEMPLATE_MODULES: Mapping[StepKind, str] = {
    "portfolios": "portfolio_optimizer.loaders",
    "loader": "portfolio_optimizer.loaders",
    "constraints_loader": "portfolio_optimizer.loaders",
    "rule": "portfolio_optimizer.rules",
    "term": "portfolio_optimizer.terms",
    "constraint": "portfolio_optimizer.terms",
    "sink": "portfolio_optimizer.sinks",
}

type ConstraintsMapping = Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class Contract:
    """The signature a step of one kind must have."""

    engine_args: Mapping[str, type]
    context: tuple[str, type] | None
    returns: tuple[object, ...]


CONTRACTS: Mapping[StepKind, Contract] = {
    "portfolios": Contract({"request": LoadRequest}, None, (pd.DataFrame,)),
    "loader": Contract({"request": LoadRequest}, None, (pd.DataFrame,)),
    "constraints_loader": Contract({"request": LoadRequest}, None, (ConstraintsMapping.__value__, dict[str, dict[str, object]])),
    "rule": Contract({"data": PortfolioData}, ("ctx", SolveContext), (PortfolioData,)),
    "term": Contract({"x": DecisionVars, "spec": ProblemSpec}, ("chain", ChainState), (ObjectiveTerm,)),
    "constraint": Contract({"x": DecisionVars, "spec": ProblemSpec}, ("chain", ChainState), (ConstraintSet,)),
    "sink": Contract({"orders": pd.DataFrame, "io": IoContext}, None, (tuple[Artifact, ...],)),
}


class ConfigResolutionError(ValueError):
    """One or more steps could not be resolved; ``failures`` lists every problem found."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__(f"{len(failures)} config resolution failure(s): " + "; ".join(failures))


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    """A step whose function, parameters, and provenance have been checked."""

    kind: StepKind
    name: str
    qualname: str
    fn: Callable[..., object]
    params: Params | None
    context_name: str | None
    source_sha256: str
    module_sha256: str
    params_sha256: str
    is_external: bool

    @property
    def needs_context(self) -> bool:
        """True when the function declares the optional context argument for its kind."""
        return self.context_name is not None

    def invoke(self, *, context: object | None = None, **engine_args: object) -> object:
        """Call the function with the engine arguments, its validated params, and its context."""
        kwargs: dict[str, object] = dict(engine_args)
        if self.params is not None:
            kwargs["params"] = self.params
        if self.context_name is not None:
            if context is None:
                msg = f"step {self.qualname!r} requires {self.context_name!r} but none was supplied"
                raise ValueError(msg)
            kwargs[self.context_name] = context
        return self.fn(**kwargs)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """The run config with every step resolved; the only form the engine consumes."""

    config: RunConfig
    config_sha256: str
    portfolios: ResolvedStep
    loaders: Mapping[str, ResolvedStep]
    rules: tuple[ResolvedStep, ...]
    terms: tuple[ResolvedStep, ...]
    constraints: tuple[ResolvedStep, ...]
    sink: ResolvedStep

    @property
    def chain_aware_steps(self) -> tuple[ResolvedStep, ...]:
        """Rules, terms, and constraints that depend on prior portfolios' results."""
        return tuple(step for step in (*self.rules, *self.terms, *self.constraints) if step.needs_context)

    @property
    def all_steps(self) -> tuple[ResolvedStep, ...]:
        """Every resolved step, in pipeline order."""
        return (self.portfolios, *self.loaders.values(), *self.rules, *self.terms, *self.constraints, self.sink)


def resolve_config(config: RunConfig, config_sha256: str) -> ResolvedConfig:
    """Resolve every step in ``config`` and check the execution mode against the steps' needs."""
    failures: list[str] = []

    def resolve(spec: StepSpec, kind: StepKind, where: str) -> ResolvedStep | None:
        try:
            return resolve_step(spec, kind)
        except ConfigResolutionError as error:
            failures.extend(f"{where}: {failure}" for failure in error.failures)
            return None

    portfolios = resolve(config.portfolios, "portfolios", "portfolios")
    loaders = {name: resolve(dataset.loader, "constraints_loader" if name == "constraints" else "loader", f"datasets.{name}") for name, dataset in config.datasets.items()}
    rules = [resolve(spec, "rule", f"rules[{i}]") for i, spec in enumerate(config.rules)]
    terms = [resolve(spec, "term", f"objective.terms[{i}]") for i, spec in enumerate(config.objective.terms)]
    constraints = [resolve(spec, "constraint", f"constraints[{i}]") for i, spec in enumerate(config.constraints)]
    sink = resolve(config.sink, "sink", "sink")
    resolved_loaders = {name: step for name, step in loaders.items() if step is not None}
    if failures or portfolios is None or sink is None or len(resolved_loaders) != len(loaders):
        raise ConfigResolutionError(failures)
    resolved = ResolvedConfig(
        config=config,
        config_sha256=config_sha256,
        portfolios=portfolios,
        loaders=resolved_loaders,
        rules=tuple(step for step in rules if step is not None),
        terms=tuple(step for step in terms if step is not None),
        constraints=tuple(step for step in constraints if step is not None),
        sink=sink,
    )
    failures.extend(_mode_failures(resolved))
    if failures:
        raise ConfigResolutionError(failures)
    return resolved


def _mode_failures(resolved: ResolvedConfig) -> list[str]:
    execution = resolved.config.execution
    chain_aware = [step.qualname for step in resolved.chain_aware_steps]
    ctx_rules = [step.qualname for step in resolved.rules if step.needs_context]
    failures: list[str] = []
    if execution.mode == "parallel" and chain_aware:
        failures.append(f"execution.mode 'parallel' cannot run chain-aware steps {chain_aware}; use a sequential mode or remove them")
    if execution.mode == "parallel_build_sequential_solve" and ctx_rules:
        failures.append(f"execution.mode 'parallel_build_sequential_solve' builds in parallel, so rules cannot take 'ctx': {ctx_rules}")
    if execution.on_error == "continue" and chain_aware:
        failures.append(f"execution.on_error 'continue' is ambiguous with chain-aware steps {chain_aware}: a skipped portfolio would silently change later solves")
    return failures


def resolve_step(spec: StepSpec, kind: StepKind) -> ResolvedStep:
    """Resolve one step: import it, check its signature against the kind's contract, validate params."""
    contract = CONTRACTS[kind]
    module_name, function_name = _split_name(spec.name, kind)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ConfigResolutionError([f"{spec.name!r}: cannot import module {module_name!r} ({error})"]) from error
    candidate = getattr(module, function_name, None)
    if candidate is None or not inspect.isfunction(candidate):
        raise ConfigResolutionError([f"{spec.name!r}: {module_name!r} has no function {function_name!r}"])
    fn: Callable[..., object] = candidate
    qualname = f"{module_name}:{function_name}"
    try:
        hints = get_type_hints(fn)
    except (NameError, TypeError) as error:
        raise ConfigResolutionError([f"{qualname}: annotations could not be evaluated ({error})"]) from error
    failures = list(_signature_failures(fn, hints, contract, qualname))
    if failures:
        raise ConfigResolutionError(failures)
    params_model = hints.get("params")
    params = _validate_params(spec, params_model, qualname)
    context_name = contract.context[0] if contract.context is not None and contract.context[0] in inspect.signature(fn).parameters else None
    return ResolvedStep(
        kind=kind,
        name=spec.name,
        qualname=qualname,
        fn=fn,
        params=params,
        context_name=context_name,
        source_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest(),
        module_sha256=_module_sha256(fn),
        params_sha256=hashlib.sha256(json.dumps(spec.params, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        is_external=module_name != TEMPLATE_MODULES[kind],
    )


def _split_name(name: str, kind: StepKind) -> tuple[str, str]:
    if ":" in name:
        module_name, function_name = name.split(":", 1)
        return module_name, function_name
    return TEMPLATE_MODULES[kind], name


def _signature_failures(fn: Callable[..., object], hints: Mapping[str, object], contract: Contract, qualname: str) -> list[str]:
    failures: list[str] = []
    parameters = inspect.signature(fn).parameters
    for name, parameter in parameters.items():
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            failures.append(f"{qualname}: parameter {name!r} must be a plain keyword-capable parameter (no *args, **kwargs, or positional-only)")
            continue
        annotation = hints.get(name)
        if name in contract.engine_args:
            expected = contract.engine_args[name]
            if annotation is not expected:
                failures.append(f"{qualname}: parameter {name!r} must be annotated {expected.__name__}, got {_describe(annotation)}")
        elif name == "params":
            if not (inspect.isclass(annotation) and issubclass(annotation, Params)):
                failures.append(f"{qualname}: 'params' must be annotated with a Params subclass, got {_describe(annotation)}")
        elif contract.context is not None and name == contract.context[0]:
            if annotation is not contract.context[1]:
                failures.append(f"{qualname}: parameter {name!r} must be annotated {contract.context[1].__name__}, got {_describe(annotation)}")
        else:
            failures.append(f"{qualname}: unexpected parameter {name!r}; allowed: {_allowed_names(contract)}")
    failures.extend(f"{qualname}: missing required parameter {name!r}" for name in contract.engine_args if name not in parameters)
    returns = hints.get("return")
    if returns is None or not any(returns == allowed for allowed in contract.returns):
        failures.append(f"{qualname}: return annotation must be {' or '.join(_describe(r) for r in contract.returns)}, got {_describe(returns)}")
    return failures


def _allowed_names(contract: Contract) -> list[str]:
    names = [*contract.engine_args, "params"]
    if contract.context is not None:
        names.append(contract.context[0])
    return names


def _describe(annotation: object) -> str:
    if annotation is None:
        return "nothing"
    if inspect.isclass(annotation):
        return annotation.__name__
    return str(annotation)


def _validate_params(spec: StepSpec, params_model: object, qualname: str) -> Params | None:
    if params_model is None:
        if spec.params:
            raise ConfigResolutionError([f"{qualname}: does not take params, but the config supplies {sorted(spec.params)}"])
        return None
    if not (inspect.isclass(params_model) and issubclass(params_model, Params)):  # pragma: no cover - guarded by _signature_failures
        raise ConfigResolutionError([f"{qualname}: 'params' annotation is not a Params subclass"])
    try:
        return params_model.model_validate_json(json.dumps(spec.params))
    except ValidationError as error:
        details = "; ".join(f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}" for item in error.errors())
        raise ConfigResolutionError([f"{qualname}: invalid params ({details})"]) from error


def _module_sha256(fn: Callable[..., object]) -> str:
    source_file = inspect.getsourcefile(fn)
    if source_file is None:  # pragma: no cover - functions resolved here always come from files
        return ""
    return hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
