"""Strict models for the JSON run config.

Money and weights are written as JSON strings and validated in JSON mode so they become exact
``Decimal`` values; solver tolerances are floats because that is the solver's domain. Every field's
``description`` is emitted into the published JSON Schema (``configs/run-config.schema.json``), so
this module is also the schema's documentation.
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

STEP_NAME_DESCRIPTION = (
    "A bare function name (`cap_single_name`), resolved in the template module for this kind of step, or a qualified `package.module:function` importable by the engine and by any worker process."
)


class StepSpec(StrictModel):
    """A reference to a function plus its parameters.

    Written either as a bare string (a step without parameters) or as an object with `name` and `params`.
    The engine imports the function before any data loads, checks its signature against the contract for
    its kind, and validates `params` against the function's own `Params` model.
    """

    name: str = Field(pattern=STEP_NAME_PATTERN, description=STEP_NAME_DESCRIPTION)
    params: dict[str, object] = Field(
        default_factory=dict,
        description='Keyword parameters validated against the function\'s `params` model. Money, weights, and rates are strings such as "0.05". A function without a `params` argument accepts none.',
    )

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
    """Identity of the run, recorded in the manifest."""

    name: str = Field(min_length=1, description="Human-readable run name, e.g. `daily_rebalance`.")
    as_of: AwareDatetime = Field(description="The timestamp the run is as of, timezone-aware (`2026-08-28T00:00:00Z`). Holding periods and the manifest use it; naive timestamps are rejected.")
    tags: dict[str, str] = Field(default_factory=dict, description='Free-form string labels copied into the manifest, e.g. `{"desk": "tax-aware"}`.')


class DatasetConfig(StrictModel):
    """How one named dataset is loaded."""

    loader: StepSpec = Field(description="The loader step. For frame datasets it returns a DataFrame; for `constraints` it returns a mapping of portfolio id to style-constraint object.")


class JoinSpec(StrictModel):
    """Enrich an engine-known frame with columns from another dataset."""

    into: Literal["holdings", "universe", "targets"] = Field(description="The engine-known frame that receives the columns; it is validated against its schema after every join.")
    source: str = Field(min_length=1, description="A declared dataset other than `into` and `constraints`; typically an extra dataset such as `prices`.")
    on: tuple[str, ...] = Field(min_length=1, description="Join key columns present in both frames. Their dtypes are aligned to `into` before merging so a text key never silently becomes `object`.")
    how: JoinHow = Field(default="left", description="`left` keeps every row of `into`; `inner` keeps only matched rows.")
    cardinality: JoinCardinality = Field(
        description="Expected key cardinality, enforced by pandas `merge(validate=...)`: a duplicate key on the wrong side aborts the run instead of multiplying rows."
    )
    require_all_matched: bool = Field(default=False, description="When true, every row of `into` must find a match in `source`; unmatched keys are reported and the run is rejected.")


class AssemblyConfig(StrictModel):
    """Key column names and the joins that combine datasets."""

    portfolio_key: str = Field(default="portfolio_id", description="Column identifying a portfolio in per-portfolio frames.")
    security_key: str = Field(default="security_id", description="Column identifying a security.")
    joins: tuple[JoinSpec, ...] = Field(default=(), description="Joins applied in order after loading and before schema validation.")


class ObjectiveConfig(StrictModel):
    """The objective as a list of named terms; the engine minimizes their sum."""

    sense: Literal["minimize"] = Field(default="minimize", description="Only minimization is supported; express a reward as a negative term (see `alpha`).")
    terms: tuple[StepSpec, ...] = Field(min_length=1, description='Objective-term steps from `terms.py`; every shipped term takes a non-negative `weight` (default "1").')


class SolverConfig(StrictModel):
    """Which cvxpy solver runs and with what options."""

    name: str = Field(
        default="CLARABEL",
        min_length=1,
        description="A solver installed in cvxpy (`CLARABEL`, `OSQP`, `SCS`, `HIGHS`, ...). The run fails before solving if it is missing; there is no automatic fallback.",
    )
    options: dict[str, float | int | bool | str] = Field(default_factory=dict, description='Passed verbatim to `Problem.solve(**options)`, e.g. `{"max_iter": 200}`.')
    time_limit_s: float | None = Field(default=None, gt=0, description="Wall-clock limit per solve in seconds, translated to the solver's own option; omit for no limit.")
    verbose: bool = Field(default=False, description="Let the solver print its iteration log.")


class PostSolveConfig(StrictModel):
    """Tolerances for the independent, cvxpy-free verification of every solution."""

    violation_tol: float = Field(default=1e-6, gt=0, description="Maximum constraint violation accepted by the verifier; deliberately looser than the solver's own tolerance.")
    objective_rel_tol: float = Field(default=1e-5, gt=0, description="Relative tolerance between the recomputed and the solver-reported objective.")
    objective_abs_tol: float = Field(default=1e-9, gt=0, description="Absolute tolerance between the recomputed and the solver-reported objective.")


class ExecutionConfig(StrictModel):
    """How portfolios are scheduled across the build and solve phases."""

    mode: ExecutionMode = Field(
        description="`sequential`: build and solve one after another with a live chain context. `parallel_build_sequential_solve`: build in workers, solve in order (constraints may use `chain`, rules may not use `ctx`). `parallel`: everything in workers; no chain-aware steps allowed."
    )
    executor: ExecutorKind = Field(default="process", description="`process` (spawned workers; required for `parallel` because cvxpy solves are not thread-safe) or `thread` (for I/O-bound builds).")
    max_workers: int = Field(default=1, ge=1, description="Worker count. Results are consumed in solve order, so this never changes the output.")
    on_error: OnError = Field(
        default="fail_fast", description="`fail_fast` stops after the first failed portfolio and records the rest as skipped; `continue` isolates failures. Chain-aware steps require `fail_fast`."
    )

    @model_validator(mode="after")
    def _threads_cannot_solve_concurrently(self) -> Self:
        if self.mode == "parallel" and self.executor == "thread":
            msg = "mode 'parallel' requires executor 'process': cvxpy solves are not thread-safe"
            raise ValueError(msg)
        return self


class RunConfig(StrictModel):
    """A portfolio-optimizer run: what to load, how to combine it, which rules and terms apply, and how to execute.

    Validated strictly: unknown keys are errors, money is written as strings, timestamps carry a zone.
    Step names are resolved and their parameters validated before any data is loaded.
    """

    schema_ref: str | None = Field(default=None, alias="$schema", description="Optional pointer to the JSON Schema for editor validation; ignored by the engine.")
    run: RunMeta = Field(description="Run identity.")
    portfolios: StepSpec = Field(description="Loader returning the portfolio list (`portfolio_id`, `solve_order`); portfolios are processed in ascending `solve_order`.")
    datasets: dict[str, DatasetConfig] = Field(
        description="Named datasets. `holdings`, `universe`, `details`, `constraints`, and `targets` are required; `covariance` is optional and enables the `risk` term; any other name is an extra dataset for `assembly.joins`."
    )
    assembly: AssemblyConfig = Field(default_factory=AssemblyConfig, description="How datasets are combined.")
    rules: tuple[StepSpec, ...] = Field(default=(), description="Business-logic rule steps from `rules.py`, applied in order to each portfolio's bundle.")
    objective: ObjectiveConfig = Field(description="What the optimizer minimizes.")
    constraints: tuple[StepSpec, ...] = Field(default=(), description="Constraint steps from `terms.py`. Keep `trade_balance`; it defines the buy/sell split the other steps rely on.")
    solver: SolverConfig = Field(default_factory=SolverConfig, description="Solver selection and options.")
    post_solve: PostSolveConfig = Field(default_factory=PostSolveConfig, description="Verification tolerances.")
    sink: StepSpec = Field(description="Sink step from `sinks.py`, called once with every solved portfolio's orders.")
    execution: ExecutionConfig = Field(description="Scheduling and failure semantics.")

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
    """Hash of the validated config's canonical JSON form; source whitespace and the `$schema` pointer do not matter."""
    return hashlib.sha256(config.model_dump_json(exclude={"schema_ref"}).encode()).hexdigest()
