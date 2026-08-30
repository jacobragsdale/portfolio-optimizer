"""Resolve step names in the run config to functions, and check them before any data loads.

The convention: a step is an ordinary function whose signature carries engine-provided
arguments by fixed names (``request``, ``frames``, ``data``, ``x``, ``spec``, ``orders``, ``io``), an optional
``params`` argument annotated with a :class:`~portfolio_optimizer.domain.types.Params` subclass,
and — for terms and constraints only — an optional ``chain`` argument that reads what higher-priority
portfolios bought. The engine calls steps with keyword arguments, so the order does not matter.
Loaders may be ``async def``; every other kind runs synchronously.

The solver is checked here too — known to the adapter, installed, able to honor ``time_limit_s`` —
because every process that will solve resolves the config first: the client at ``validate-config``
and at the start of ``run``, and each worker before it does any work.
"""

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import get_type_hints

import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy.sparse import csr_array

from portfolio_optimizer.config.models import RunConfig, StepSpec
from portfolio_optimizer.config.steps import ResolvedConstraint, ResolvedStep, StepKind
from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars, ObjectiveTerm, installed_solvers, solver_failures, variables
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.results import Artifact, ChainState, MissingSpecColumnError, ProblemSpec
from portfolio_optimizer.domain.sides import SideProfile, profile_for
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.solving import SolveRequest, SolveResult

__all__ = [
    "CONTRACTS",
    "TEMPLATE_MODULES",
    "ConfigResolutionError",
    "Contract",
    "ResolvedConfig",
    "ResolvedConstraint",
    "ResolvedStep",
    "StepKind",
    "construction_failures",
    "resolve_config",
    "resolve_step",
]

TEMPLATE_MODULES: Mapping[StepKind, str] = {
    "portfolios": "portfolio_optimizer.loaders",
    "loader": "portfolio_optimizer.loaders",
    "constraints_loader": "portfolio_optimizer.loaders",
    "assembly": "portfolio_optimizer.assembly",
    "rule": "portfolio_optimizer.rules",
    "solve_order": "portfolio_optimizer.solve_order",
    "term": "portfolio_optimizer.terms",
    "constraint": "portfolio_optimizer.terms",
    "solve": "portfolio_optimizer.solvers",
    "sink": "portfolio_optimizer.sinks",
}

type ConstraintsMapping = Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class Contract:
    """The signature a step of one kind must have."""

    engine_args: Mapping[str, type]
    context: tuple[str, type] | None
    returns: tuple[object, ...]
    allows_async: bool = False


CONTRACTS: Mapping[StepKind, Contract] = {
    "portfolios": Contract({"request": LoadRequest}, None, (pd.DataFrame,), allows_async=True),
    "loader": Contract({"request": LoadRequest}, None, (pd.DataFrame,), allows_async=True),
    "constraints_loader": Contract({"request": LoadRequest}, None, (ConstraintsMapping.__value__, dict[str, dict[str, object]]), allows_async=True),
    "assembly": Contract({"frames": Frames}, None, (Frames,)),
    "rule": Contract({"data": PortfolioData}, None, (PortfolioData,)),
    "solve_order": Contract({"data": PortfolioData}, None, (Decimal,)),
    "term": Contract({"x": DecisionVars, "spec": ProblemSpec}, ("chain", ChainState), (ObjectiveTerm,)),
    "constraint": Contract({"x": DecisionVars, "spec": ProblemSpec}, ("chain", ChainState), (ConstraintSet,)),
    "solve": Contract({"request": SolveRequest}, None, (SolveResult,)),
    "sink": Contract({"orders": pd.DataFrame, "io": IoContext}, None, (tuple[Artifact, ...],)),
}


class ConfigResolutionError(ValueError):
    """One or more steps could not be resolved; ``failures`` lists every problem found."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__(f"{len(failures)} config resolution failure(s): " + "; ".join(failures))


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """The run config with every step resolved; the only form the engine consumes."""

    config: RunConfig
    config_sha256: str
    portfolios: ResolvedStep
    loaders: Mapping[str, ResolvedStep]
    assembly: tuple[ResolvedStep, ...]
    rules: tuple[ResolvedStep, ...]
    solve_order: ResolvedStep | None
    terms: tuple[ResolvedStep, ...]
    constraints: tuple[ResolvedConstraint, ...]
    solve: ResolvedStep
    sink: ResolvedStep
    profile: SideProfile

    @property
    def chain_aware_steps(self) -> tuple[ResolvedStep, ...]:
        """Terms and constraints that read what higher-priority portfolios traded; if there are none, no portfolio waits for another."""
        return tuple(step for step in (*self.terms, *(constraint.step for constraint in self.constraints)) if step.needs_context)

    @property
    def all_steps(self) -> tuple[ResolvedStep, ...]:
        """Every resolved step, in pipeline order."""
        ordering = () if self.solve_order is None else (self.solve_order,)
        return (self.portfolios, *self.loaders.values(), *self.assembly, *self.rules, *ordering, *self.terms, *(constraint.step for constraint in self.constraints), self.solve, self.sink)


def resolve_config(config: RunConfig, config_sha256: str, *, installed: Callable[[], Sequence[str]] = installed_solvers) -> ResolvedConfig:
    """Resolve every step in ``config`` and check its solver can run in this process; every failure is collected and reported together.

    ``installed`` names the solvers cvxpy can use here; it is a parameter so the check can be exercised
    against any environment.
    """
    failures: list[str] = [f"solver: {failure}" for failure in solver_failures(config.solver.name, config.solver.time_limit_s, installed())]

    def resolve(spec: StepSpec, kind: StepKind, where: str) -> ResolvedStep | None:
        try:
            return resolve_step(spec, kind)
        except ConfigResolutionError as error:
            failures.extend(f"{where}: {failure}" for failure in error.failures)
            return None

    portfolios = resolve(config.portfolios.loader, "portfolios", "portfolios")
    loaders = {name: resolve(dataset.loader, "constraints_loader" if name == "constraints" else "loader", f"datasets.{name}") for name, dataset in config.datasets.items()}
    assembly = [resolve(spec, "assembly", f"assembly[{i}]") for i, spec in enumerate(config.assembly)]
    rules = [resolve(spec, "rule", f"rules[{i}]") for i, spec in enumerate(config.rules)]
    solve_order = resolve(config.solve_order, "solve_order", "solve_order") if config.solve_order is not None else None
    terms = [resolve(spec, "term", f"objective.terms[{i}]") for i, spec in enumerate(config.objective.terms)]
    constraints: list[ResolvedConstraint | None] = []
    labels: dict[str, int] = {}
    for i, spec in enumerate(config.constraints):
        if spec.name == "trade_balance":
            failures.append(f"constraints[{i}]: 'trade_balance' is not a configurable constraint; the trade identity comes from `sides` ({config.sides!r}) — remove it")
            continue
        label = spec.effective_label
        if label in labels:
            failures.append(f"constraints[{i}]: label {label!r} is also used by constraints[{labels[label]}]; give one of them a `label`")
        labels.setdefault(label, i)
        step = resolve(spec, "constraint", f"constraints[{i}]")
        constraints.append(ResolvedConstraint(label=label, spec=spec, step=step) if step is not None else None)
    solve = resolve(config.solve, "solve", "solve")
    sink = resolve(config.sink, "sink", "sink")
    resolved_loaders = {name: step for name, step in loaders.items() if step is not None}
    if failures or portfolios is None or solve is None or sink is None or len(resolved_loaders) != len(loaders):
        raise ConfigResolutionError(failures)
    return ResolvedConfig(
        config=config,
        config_sha256=config_sha256,
        portfolios=portfolios,
        loaders=resolved_loaders,
        assembly=tuple(step for step in assembly if step is not None),
        rules=tuple(step for step in rules if step is not None),
        solve_order=solve_order,
        terms=tuple(step for step in terms if step is not None),
        constraints=tuple(constraint for constraint in constraints if constraint is not None),
        solve=solve,
        sink=sink,
        profile=profile_for(config.sides),
    )


def construction_failures(resolved: ResolvedConfig) -> list[str]:
    """Construct every term and constraint once against a one-security dummy spec, under the run's side profile.

    What resolution cannot see — a term that raises when called, a constraint reaching for a decision
    vector the side does not have — surfaces here instead of on a worker. A step that asks for a spec
    column or flag the dummy does not carry is skipped rather than failed: whether the universe has it
    is a question for the data, not the config. The solve step is not run; a firm's step may reach a
    service, and the dummy is not a problem worth solving.
    """
    spec = _dry_run_spec()
    chain = ChainState.empty(spec.security_ids)
    failures: list[str] = []
    for where, step, expected in (
        *((f"objective.terms[{i}]", step, ObjectiveTerm) for i, step in enumerate(resolved.terms)),
        *((f"constraints[{i}]", c.step, ConstraintSet) for i, c in enumerate(resolved.constraints)),
    ):
        try:
            result = step.invoke(x=variables(spec.n), spec=spec, context=chain if step.needs_context else None)
        except MissingSpecColumnError:
            continue
        except Exception as error:  # noqa: BLE001  # any construction failure is what this check exists to report
            failures.append(f"{where}: {step.qualname}: construction failed: {type(error).__name__}: {error}")
            continue
        if not isinstance(result, expected):
            failures.append(f"{where}: {step.qualname}: returned {type(result).__name__}, expected {expected.__name__}")
    return failures


def _dry_run_spec() -> ProblemSpec:
    one = np.ones(1)
    return ProblemSpec(
        portfolio_id="dry-run",
        as_of=datetime(2000, 1, 1, tzinfo=UTC),
        security_ids=("DRY",),
        sector_names=("DRY",),
        nav=1.0,
        w0=one * 0.5,
        price=one,
        shares_held=one * 0.5,
        lot_size=one,
        w_target=one * 0.5,
        tax_per_dollar=np.zeros(1),
        tcost_per_dollar=np.zeros(1),
        lb=np.zeros(1),
        ub=one,
        adv_capacity=one,
        sector_matrix=csr_array(np.ones((1, 1))),
        sector_lb=np.zeros(1),
        sector_ub=one,
        max_turnover=2.0,
        cash_lb=0.0,
        cash_ub=1.0,
        min_trade_notional=0.0,
    )


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
        params_sha256=hashlib.sha256(json.dumps(spec.params, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        is_external=module_name != TEMPLATE_MODULES[kind],
        is_async=inspect.iscoroutinefunction(fn),
    )


def _split_name(name: str, kind: StepKind) -> tuple[str, str]:
    if ":" in name:
        module_name, function_name = name.split(":", 1)
        return module_name, function_name
    return TEMPLATE_MODULES[kind], name


def _signature_failures(fn: Callable[..., object], hints: Mapping[str, object], contract: Contract, qualname: str) -> list[str]:
    failures: list[str] = []
    if inspect.iscoroutinefunction(fn) and not contract.allows_async:
        failures.append(f"{qualname}: `async def` is only allowed for loaders; this step kind runs synchronously")
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
