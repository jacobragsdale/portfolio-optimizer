"""Strict models for the JSON run config.

Money and weights are written as JSON strings and validated in JSON mode so they become exact
``Decimal`` values; solver tolerances are floats because that is the solver's domain.
"""

import hashlib
import re
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from portfolio_optimizer.domain.schemas import REQUIRED_DATASETS
from portfolio_optimizer.domain.types import StrictModel

STEP_NAME_PATTERN = r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:)?[A-Za-z_][A-Za-z0-9_]*$"
_STEP_NAME = re.compile(STEP_NAME_PATTERN)

JOINABLE_DATASETS: tuple[str, ...] = ("holdings", "universe", "targets")
"""Datasets an assembly join may enrich; each is validated against its schema afterwards."""

type ExecutionMode = Literal["sequential", "parallel_build_sequential_solve", "parallel"]
type ExecutorKind = Literal["process", "thread"]
type OnError = Literal["fail_fast", "continue"]
type JoinHow = Literal["left", "inner"]
type JoinCardinality = Literal["one_to_one", "one_to_many", "many_to_one"]


class StepSpec(StrictModel):
    """A reference to a function plus its parameters.

    Accepts a bare string (``"exclude_restricted"``) or an object (``{"name": ..., "params": {...}}``).
    A name is either bare — resolved in the template module for its kind — or qualified as
    ``package.module:function``.
    """

    name: str = Field(pattern=STEP_NAME_PATTERN)
    params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: object) -> object:
        if isinstance(value, str):
            return {"name": value}
        return value

    @property
    def is_qualified(self) -> bool:
        """True when the name carries an explicit module."""
        return ":" in self.name


def is_step_name(value: str) -> bool:
    """True when ``value`` is a well-formed bare or qualified step name."""
    return _STEP_NAME.match(value) is not None


class RunMeta(StrictModel):
    """Identity of a run: a human name, the as-of timestamp, and free-form tags for the manifest."""

    name: str = Field(min_length=1)
    as_of: AwareDatetime
    tags: dict[str, str] = Field(default_factory=dict)


class DatasetConfig(StrictModel):
    """How one named dataset is loaded."""

    loader: StepSpec


class JoinSpec(StrictModel):
    """Enrich an engine-known frame with columns from another dataset."""

    into: Literal["holdings", "universe", "targets"]
    source: str = Field(min_length=1)
    on: tuple[str, ...] = Field(min_length=1)
    how: JoinHow = "left"
    cardinality: JoinCardinality
    require_all_matched: bool = False


class AssemblyConfig(StrictModel):
    """Key column names and the joins that combine datasets."""

    portfolio_key: str = "portfolio_id"
    security_key: str = "security_id"
    joins: tuple[JoinSpec, ...] = ()


class ObjectiveConfig(StrictModel):
    """The objective as a list of named terms; the engine minimizes their sum."""

    sense: Literal["minimize"] = "minimize"
    terms: tuple[StepSpec, ...] = Field(min_length=1)


class SolverConfig(StrictModel):
    """Which cvxpy solver runs and with what options."""

    name: str = Field(default="CLARABEL", min_length=1)
    options: dict[str, float | int | bool | str] = Field(default_factory=dict)
    time_limit_s: float | None = Field(default=None, gt=0)
    verbose: bool = False


class PostSolveConfig(StrictModel):
    """Tolerances for the independent post-solve verification."""

    violation_tol: float = Field(default=1e-6, gt=0)
    objective_rel_tol: float = Field(default=1e-5, gt=0)
    objective_abs_tol: float = Field(default=1e-9, gt=0)


class ExecutionConfig(StrictModel):
    """How portfolios are scheduled across the build and solve phases."""

    mode: ExecutionMode
    executor: ExecutorKind = "process"
    max_workers: int = Field(default=1, ge=1)
    on_error: OnError = "fail_fast"

    @model_validator(mode="after")
    def _threads_cannot_solve_concurrently(self) -> Self:
        if self.mode == "parallel" and self.executor == "thread":
            msg = "mode 'parallel' requires executor 'process': cvxpy solves are not thread-safe"
            raise ValueError(msg)
        return self


class RunConfig(StrictModel):
    """The whole run config."""

    run: RunMeta
    portfolios: StepSpec
    datasets: dict[str, DatasetConfig]
    assembly: AssemblyConfig = Field(default_factory=AssemblyConfig)
    rules: tuple[StepSpec, ...] = ()
    objective: ObjectiveConfig
    constraints: tuple[StepSpec, ...] = ()
    solver: SolverConfig = Field(default_factory=SolverConfig)
    post_solve: PostSolveConfig = Field(default_factory=PostSolveConfig)
    sink: StepSpec
    execution: ExecutionConfig

    @model_validator(mode="after")
    def _datasets_and_joins_are_consistent(self) -> Self:
        missing = [name for name in REQUIRED_DATASETS if name not in self.datasets]
        if missing:
            msg = f"datasets must declare {list(REQUIRED_DATASETS)}; missing {missing}"
            raise ValueError(msg)
        for index, join in enumerate(self.assembly.joins):
            if join.source not in self.datasets:
                msg = f"assembly.joins[{index}]: source dataset {join.source!r} is not declared"
                raise ValueError(msg)
            if join.source in (join.into, "constraints"):
                msg = f"assembly.joins[{index}]: cannot join {join.source!r} into {join.into!r}"
                raise ValueError(msg)
        return self


def load_run_config(text: str) -> RunConfig:
    """Validate the JSON text of a run config."""
    return RunConfig.model_validate_json(text)


def config_sha256(config: RunConfig) -> str:
    """Hash of the validated config's canonical JSON form; whitespace in the source file does not matter."""
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()
