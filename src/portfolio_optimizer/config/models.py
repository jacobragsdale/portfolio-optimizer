"""Strict models for the JSON run config.

Money and weights are written as JSON strings and validated in JSON mode so they become exact
``Decimal`` values; solver tolerances are floats because that is the solver's domain. Every field's
``description`` is emitted into the published JSON Schema (``configs/run-config.schema.json``), so
this module is also the schema's documentation.
"""

import hashlib
import math
import re
from typing import Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from portfolio_optimizer.domain.schemas import REQUIRED_DATASETS
from portfolio_optimizer.domain.types import StrictModel
from portfolio_optimizer.ratelimit import RateLimit

STEP_NAME_PATTERN = r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:)?[A-Za-z_][A-Za-z0-9_]*$"
_STEP_NAME = re.compile(STEP_NAME_PATTERN)

type ExecutionMode = Literal["sequential", "parallel_build_sequential_solve", "parallel"]
type OnError = Literal["fail_fast", "continue"]

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


class RateLimitConfig(StrictModel):
    """Bounds shared by every dataset that names this pool.

    At least one of `requests_per_second` and `max_in_flight` is required. The loader receives the pool as
    `request.rate_limiter` and wraps each call to the backend in it.
    """

    requests_per_second: float | None = Field(default=None, gt=0, description="Sustained request rate the pool allows, refilled continuously (a token bucket). Omit for no rate bound.")
    burst: int | None = Field(
        default=None, ge=1, description="Requests that may be made at once before the rate applies; defaults to `requests_per_second` rounded up. Requires `requests_per_second`."
    )
    max_in_flight: int | None = Field(default=None, ge=1, description="Maximum simultaneous requests across every loader in the pool. Omit for no concurrency bound.")

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> Self:
        if self.requests_per_second is None and self.max_in_flight is None:
            msg = "a rate limit needs requests_per_second, max_in_flight, or both"
            raise ValueError(msg)
        if self.burst is not None and self.requests_per_second is None:
            msg = "burst only applies with requests_per_second"
            raise ValueError(msg)
        return self

    def to_limit(self) -> RateLimit:
        """The runtime form the engine builds a limiter from."""
        default_burst = 1 if self.requests_per_second is None else max(1, math.ceil(self.requests_per_second))
        return RateLimit(requests_per_second=self.requests_per_second, burst=self.burst if self.burst is not None else default_burst, max_in_flight=self.max_in_flight)


class DatasetConfig(StrictModel):
    """How one input is loaded, and how hard its source may be pushed."""

    loader: StepSpec = Field(description="The loader step. For frame datasets it returns a DataFrame; for `constraints` it returns a mapping of portfolio id to style-constraint object.")
    rate_limit: str | RateLimitConfig | None = Field(
        default=None,
        description="How hard this input's source may be pushed: the name of a shared pool in the top-level `rate_limits`, or an inline bound private to this input. The loader receives it as `request.rate_limiter`. Omit for no limit.",
    )


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
    """How portfolios are scheduled across the build and solve phases, and what a failure means.

    Which cluster the run provisions for itself and how many workers it has are settings
    (`PORTFOLIO_OPTIMIZER_CLUSTER`, `PORTFOLIO_OPTIMIZER_MAX_WORKERS`, ...), not config, so the same config
    hashes the same on a laptop and on a cluster. Results are consumed in solve order whatever the
    cluster, so neither changes the output.
    """

    mode: ExecutionMode = Field(
        description="`sequential`: build and solve one after another with a live chain context. `parallel_build_sequential_solve`: build in workers, solve in order (constraints may use `chain`, rules may not use `ctx`). `parallel`: everything in workers; no chain-aware steps allowed."
    )
    on_error: OnError = Field(
        default="fail_fast", description="`fail_fast` stops after the first failed portfolio and records the rest as skipped; `continue` isolates failures. Chain-aware steps require `fail_fast`."
    )


class RunConfig(StrictModel):
    """A portfolio-optimizer run: what to load, how to combine it, which rules and terms apply, and how to execute.

    Validated strictly: unknown keys are errors, money is written as strings, timestamps carry a zone.
    Step names are resolved and their parameters validated before any data is loaded.
    """

    schema_ref: str | None = Field(default=None, alias="$schema", description="Optional pointer to the JSON Schema for editor validation; ignored by the engine.")
    run: RunMeta = Field(description="Run identity.")
    portfolios: DatasetConfig = Field(
        description='The portfolio list (`portfolio_id`, `solve_order`): a bare loader step, or `{"loader": step, "rate_limit": ...}` to bound its source. Portfolios are processed in ascending `solve_order`.'
    )
    datasets: dict[str, DatasetConfig] = Field(
        description="Named datasets. `constraints` is always required; `holdings`, `universe`, `details`, and `targets` must be declared here unless an assembly step produces them. Any other name is an extra dataset: available to every assembly step by name and carried into each portfolio's bundle as `data.extras` (reduced to the portfolio's rows when it has a `portfolio_id` column). Every dataset loader runs concurrently once the portfolio list is known."
    )
    rate_limits: dict[str, RateLimitConfig] = Field(
        default_factory=dict,
        description='Named rate-limit pools for loaders, e.g. `{"vendor_api": {"requests_per_second": 20, "max_in_flight": 8}}`. A dataset opts in with `rate_limit`; datasets naming the same pool share its budget.',
    )
    assembly: tuple[StepSpec, ...] = Field(
        default=(),
        description="Assembly steps from `assembly.py`, applied in order to the loaded datasets before the engine-known frames are validated: `join`, `union`, `select`, `drop`, or any custom `(frames: Frames[, params]) -> Frames` function. This is where analytics columns are attached to `holdings` and `universe`.",
    )
    rules: tuple[StepSpec, ...] = Field(default=(), description="Business-logic rule steps from `rules.py`, applied in order to each portfolio's bundle.")
    objective: ObjectiveConfig = Field(description="What the optimizer minimizes.")
    constraints: tuple[StepSpec, ...] = Field(default=(), description="Constraint steps from `terms.py`. Keep `trade_balance`; it defines the buy/sell split the other steps rely on.")
    solver: SolverConfig = Field(default_factory=SolverConfig, description="Solver selection and options.")
    post_solve: PostSolveConfig = Field(default_factory=PostSolveConfig, description="Verification tolerances.")
    sink: StepSpec = Field(description="Sink step from `sinks.py`, called once with every solved portfolio's orders.")
    execution: ExecutionConfig = Field(description="Scheduling and failure semantics.")

    @field_validator("portfolios", mode="before")
    @classmethod
    def _accept_bare_portfolios_step(cls, value: object) -> object:
        if isinstance(value, str) or (isinstance(value, dict) and "name" in value):
            return {"loader": value}
        return value

    @model_validator(mode="after")
    def _datasets_are_consistent(self) -> Self:
        required = REQUIRED_DATASETS if not self.assembly else ("constraints",)
        missing = [name for name in required if name not in self.datasets]
        if missing:
            hint = "" if self.assembly else "; a run without assembly steps has nothing else to produce them"
            msg = f"datasets must declare {list(required)}; missing {missing}{hint}"
            raise ValueError(msg)
        inputs = [("portfolios", self.portfolios), *((f"datasets.{name}", dataset) for name, dataset in self.datasets.items())]
        for where, dataset in inputs:
            if isinstance(dataset.rate_limit, str) and dataset.rate_limit not in self.rate_limits:
                msg = f"{where}: rate_limit {dataset.rate_limit!r} is not declared in rate_limits {sorted(self.rate_limits)}"
                raise ValueError(msg)
        return self


def load_run_config(text: str) -> RunConfig:
    """Validate the JSON text of a run config."""
    return RunConfig.model_validate_json(text)


def config_sha256(config: RunConfig) -> str:
    """Hash of the validated config's canonical JSON form; source whitespace and the `$schema` pointer do not matter."""
    return hashlib.sha256(config.model_dump_json(exclude={"schema_ref"}).encode()).hexdigest()
