"""Resolve step names in the run config to functions, parse the objective's terms, and check both before any data loads.

The convention: a step is an ordinary function whose signature carries engine-provided arguments
by fixed names (``request``, ``frames``, ``data``, ``orders``, ``io``) and an optional ``params``
argument annotated with a :class:`~portfolio_optimizer.domain.types.Params` subclass. The engine
calls steps with keyword arguments, so the order does not matter. Loaders may be ``async def``;
every other kind runs synchronously.

A bare name is looked up in the template module for its kind, then among the steps installed
packages publish as entry points in the group ``portfolio_optimizer.<kind>``; a qualified
``package.module:function`` is imported from anywhere the engine can import — or, when the settings
name an allowlist of step packages, from those alone. Objective terms are typed models rather than
steps: each ``objective`` entry is validated against the kind it names.

Resolution is every check a config can pass without data: the solver the shipped cvxpy step names —
known to the adapter, installed, able to honor ``time_limit_s`` — and, under that step, one dry
rendering of every term against a one-security dummy spec under the run's side profile, so a term
that raises, reads a side the run lacks, or is not convex is refused here rather than on a worker.
Every process that will solve resolves the config first: the client at ``validate-config`` and at
the start of ``run``, and each worker before it does any work, so all of them apply the same checks.
"""

import hashlib
import importlib
import importlib.metadata
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

from portfolio_optimizer.config.models import DatasetConfig, RunConfig, StepSpec, config_sha256
from portfolio_optimizer.config.steps import ResolvedStep, StepKind
from portfolio_optimizer.domain.data import Frames, IoContext, LoadRequest, PortfolioData
from portfolio_optimizer.domain.objective import TermSpecError, TypedTerm, parse_terms
from portfolio_optimizer.domain.results import Artifact, ChainState, MissingSpecColumnError, ProblemSpec
from portfolio_optimizer.domain.sides import SideProfile, profile_for
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.solving import SHIPPED_CVXPY_SOLVE, SolveRequest, SolveResult, solver_failures

TEMPLATE_PACKAGE = "portfolio_optimizer"

TEMPLATE_MODULES: Mapping[StepKind, str] = {
    "loader": "portfolio_optimizer.loaders",
    "assembly": "portfolio_optimizer.assembly",
    "rule": "portfolio_optimizer.rules",
    "solve_order": "portfolio_optimizer.solve_order",
    "build": "portfolio_optimizer.engine.build",
    "solve": "portfolio_optimizer.solvers",
    "sink": "portfolio_optimizer.sinks",
}


def entry_point_group(kind: StepKind) -> str:
    """The entry-point group a package publishes steps of ``kind`` under: ``portfolio_optimizer.rule``, ``portfolio_optimizer.loader``, ..."""
    return f"{TEMPLATE_PACKAGE}.{kind}"


@dataclass(frozen=True, slots=True)
class Contract:
    """The signature a step of one kind must have."""

    engine_args: Mapping[str, type]
    returns: tuple[object, ...]
    allows_async: bool = False


CONTRACTS: Mapping[StepKind, Contract] = {
    "loader": Contract({"request": LoadRequest}, (pd.DataFrame,), allows_async=True),
    "assembly": Contract({"frames": Frames}, (Frames,)),
    "rule": Contract({"data": PortfolioData}, (PortfolioData,)),
    "solve_order": Contract({"data": PortfolioData}, (Decimal,)),
    "build": Contract({"data": PortfolioData}, (ProblemSpec,)),
    "solve": Contract({"request": SolveRequest}, (SolveResult,)),
    "sink": Contract({"orders": pd.DataFrame, "io": IoContext}, (tuple[Artifact, ...],)),
}


class ConfigResolutionError(ValueError):
    """One or more steps could not be resolved; ``failures`` lists every problem found."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__(f"{len(failures)} config resolution failure(s): " + "; ".join(failures))


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """The run config with every step resolved and every term parsed; the only form the engine consumes."""

    config: RunConfig
    config_sha256: str
    loaders: Mapping[str, ResolvedStep]
    assembly: tuple[ResolvedStep, ...]
    rules: tuple[ResolvedStep, ...]
    solve_order: ResolvedStep | None
    build: ResolvedStep
    terms: tuple[TypedTerm, ...]
    solve: ResolvedStep
    sink: ResolvedStep
    profile: SideProfile

    @property
    def chain_aware_terms(self) -> tuple[TypedTerm, ...]:
        """Objective terms that read what higher-priority portfolios traded; any one of them couples every portfolio through its whole tradable set."""
        return tuple(term for term in self.terms if term.reads_chain)

    @property
    def shipped_solve(self) -> bool:
        """Whether the solve step is the shipped cvxpy one, the only step whose chain access is exactly the configured terms and constraints."""
        return self.solve.qualname == SHIPPED_CVXPY_SOLVE

    @property
    def all_steps(self) -> tuple[ResolvedStep, ...]:
        """Every resolved step, in pipeline order."""
        ordering = () if self.solve_order is None else (self.solve_order,)
        return (*self.loaders.values(), *self.assembly, *self.rules, *ordering, self.build, self.solve, self.sink)


def resolve_config(config: RunConfig, *, installed: Callable[[], Sequence[str]] | None = None, packages: Sequence[str] | None = None) -> ResolvedConfig:
    """Resolve every step in ``config``, parse every term, check the shipped solver can run in this process, and dry-render the objective once.

    Every failure resolution can see is collected and reported together; the dry rendering runs
    only on a config whose steps and terms all resolved. ``installed`` names the solvers cvxpy can
    use here (a parameter so the check can be exercised against any environment; unset, cvxpy is
    asked, and only when the shipped step is configured). ``packages`` is the allowlist a qualified
    step name may import from; ``None`` allows any importable module.
    """
    failures: list[str] = []

    def resolve(spec: StepSpec, kind: StepKind, where: str) -> ResolvedStep | None:
        try:
            return resolve_step(spec, kind, packages=packages)
        except ConfigResolutionError as error:
            failures.extend(f"{where}: {failure}" for failure in error.failures)
            return None

    loaders = {name: resolve(dataset.loader, "loader", f"datasets.{name}") for name, dataset in config.datasets.items() if isinstance(dataset, DatasetConfig)}
    assembly = [resolve(spec, "assembly", f"assembly[{i}]") for i, spec in enumerate(config.assembly)]
    rules = [resolve(spec, "rule", f"rules[{i}]") for i, spec in enumerate(config.rules)]
    solve_order = resolve(config.solve_order, "solve_order", "solve_order") if config.solve_order is not None else None
    build = resolve(config.build, "build", "build")
    solve = resolve(config.solve, "solve", "solve")
    sink = resolve(config.sink, "sink", "sink")
    try:
        terms = parse_terms(config.objective)
    except TermSpecError as error:
        failures.append(str(error))
        terms = ()
    if solve is not None and solve.qualname == SHIPPED_CVXPY_SOLVE and solve.params is not None:
        params = solve.params.model_dump()
        available = installed() if installed is not None else _installed_solvers()
        failures.extend(f"solve: {failure}" for failure in solver_failures(str(params["solver"]), params["time_limit_s"], available))
    resolved_loaders = {name: step for name, step in loaders.items() if step is not None}
    if failures or build is None or solve is None or sink is None or len(resolved_loaders) != len(loaders):
        raise ConfigResolutionError(failures)
    resolved = ResolvedConfig(
        config=config,
        config_sha256=config_sha256(config),
        loaders=resolved_loaders,
        assembly=tuple(step for step in assembly if step is not None),
        rules=tuple(step for step in rules if step is not None),
        solve_order=solve_order,
        build=build,
        terms=terms,
        solve=solve,
        sink=sink,
        profile=profile_for(config.sides),
    )
    construction = _construction_failures(resolved)
    if construction:
        raise ConfigResolutionError(construction)
    return resolved


def _installed_solvers() -> tuple[str, ...]:
    from portfolio_optimizer.cvx.adapter import installed_solvers  # cvxpy is reached only when the shipped step is configured

    return installed_solvers()


def _construction_failures(resolved: ResolvedConfig) -> list[str]:
    """Render every objective term once against a one-security dummy spec, under the run's side profile, through the shipped cvxpy step's own machinery.

    What parsing cannot see — a term that raises when rendered, reaches for a decision vector the
    side does not have, or is not convex — surfaces here instead of on a worker. A term that asks for
    a spec column, scalar, or flag the dummy does not carry is skipped rather than failed: whether the
    universe has it is a question for the data, not the config. Only the shipped step renders terms
    this way, so only under it is the check made; constraints are loaded per portfolio and are checked
    against the real spec at build.
    """
    if not resolved.shipped_solve:
        return []
    if not resolved.terms:
        return ["objective: the cvxpy solve step minimizes the terms' sum and needs at least one; a run that minimizes nothing wants a solve step that is not an optimizer"]
    from portfolio_optimizer.cvx.adapter import ObjectiveTerm, build_problem
    from portfolio_optimizer.cvx.sides import decision_variables, identity_constraints

    spec = _dry_run_spec()
    chain = ChainState.empty(spec.security_ids)
    x = decision_variables(resolved.profile.sides, spec)
    failures: list[str] = []
    rendered: list[ObjectiveTerm] = []
    for index, term in enumerate(resolved.terms):
        where = f"objective[{index}]"
        try:
            result = term.to_cvxpy(x, spec, chain)
        except MissingSpecColumnError:
            continue
        except Exception as error:  # noqa: BLE001  # any rendering failure is what this check exists to report
            failures.append(f"{where}: {term.name}: rendering failed: {type(error).__name__}: {error}")
            continue
        if not isinstance(result, ObjectiveTerm):
            failures.append(f"{where}: {term.name}: rendered {type(result).__name__}, expected ObjectiveTerm")
            continue
        rendered.append(result)
    if not failures and rendered:
        try:
            build_problem(rendered, [identity_constraints(resolved.profile.sides, x, spec)])
        except ValueError as error:
            failures.append(f"objective: {error}")
    return failures


def _dry_run_spec() -> ProblemSpec:
    """One security carrying the columns the standard build always derives, so the shipped terms render in full; an exported column is the data's business."""
    one = np.ones(1)
    derived = {"tax_per_dollar": np.zeros(1), "tcost_per_dollar": np.zeros(1), "adv_capacity": one}
    return ProblemSpec(
        portfolio_id="dry-run",
        as_of_date=datetime(2000, 1, 1, tzinfo=UTC),
        security_ids=("DRY",),
        nav=1.0,
        w0=one * 0.5,
        price=one,
        shares_held=one * 0.5,
        lot_size=one,
        lb=np.zeros(1),
        ub=one,
        columns=derived,
    )


def resolve_step(spec: StepSpec, kind: StepKind, *, packages: Sequence[str] | None = None) -> ResolvedStep:
    """Resolve one step: find it, import it, check its signature against the kind's contract, validate params."""
    contract = CONTRACTS[kind]
    module_name, function_name = _locate(spec.name, kind, packages)
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
    return ResolvedStep(
        kind=kind,
        name=spec.name,
        qualname=qualname,
        fn=fn,
        params=params,
        source_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest(),
        params_sha256=hashlib.sha256(json.dumps(spec.params, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        is_external=module_name != TEMPLATE_MODULES[kind],
        is_async=inspect.iscoroutinefunction(fn),
    )


def published_steps(kind: StepKind) -> Mapping[str, tuple[str, str]]:
    """The steps installed packages publish for ``kind``, by bare name: ``(module, function)`` each."""
    found: dict[str, tuple[str, str]] = {}
    for entry_point in importlib.metadata.entry_points(group=entry_point_group(kind)):
        if not entry_point.attr:
            msg = f"entry point {entry_point.name!r} in {entry_point_group(kind)!r} must name a function as module:function"
            raise ConfigResolutionError([msg])
        found[entry_point.name] = (entry_point.module, entry_point.attr)
    return found


def _locate(name: str, kind: StepKind, packages: Sequence[str] | None) -> tuple[str, str]:
    """Where a step name points: a qualified name as written, a bare name in the template module or among the published steps."""
    if ":" in name:
        module_name, function_name = name.split(":", 1)
        top = module_name.partition(".")[0]
        if packages is not None and top != TEMPLATE_PACKAGE and top not in packages:
            msg = f"{name!r}: package {top!r} is not among the step packages the settings allow {sorted(packages)}; add it to PORTFOLIO_OPTIMIZER_STEP_PACKAGES or publish the step as an entry point"
            raise ConfigResolutionError([msg])
        return module_name, function_name
    template = TEMPLATE_MODULES[kind]
    module = importlib.import_module(template)
    if hasattr(module, name):
        return template, name
    published = published_steps(kind).get(name)
    if published is not None:
        return published
    return template, name  # reported as "has no function" by the caller, naming the template module


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
        else:
            failures.append(f"{qualname}: unexpected parameter {name!r}; allowed: {[*contract.engine_args, 'params']}")
    failures.extend(f"{qualname}: missing required parameter {name!r}" for name in contract.engine_args if name not in parameters)
    returns = hints.get("return")
    if returns is None or not any(returns == allowed for allowed in contract.returns):
        failures.append(f"{qualname}: return annotation must be {' or '.join(_describe(r) for r in contract.returns)}, got {_describe(returns)}")
    return failures


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
