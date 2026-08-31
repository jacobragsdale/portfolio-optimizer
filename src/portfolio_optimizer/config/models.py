"""Strict models for the JSON run config.

Money and weights are written as JSON strings and validated in JSON mode so they become exact
``Decimal`` values; solver tolerances are floats because that is the solver's domain. Every field's
``description`` is emitted into the published JSON Schema (``configs/run-config.schema.json``), so
this module is also the schema's documentation.
"""

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from portfolio_optimizer.domain.schemas import REQUIRED_DATASETS
from portfolio_optimizer.domain.sides import Sides
from portfolio_optimizer.domain.types import PortfolioId, StrictModel
from portfolio_optimizer.ratelimit import RateLimit

STEP_NAME_PATTERN = r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:)?[A-Za-z_][A-Za-z0-9_]*$"

type OnError = Literal["fail_fast", "continue"]
type Dependencies = Literal["overlap", "all", "none"]
type DatasetScope = Literal["global", "per_portfolio"]

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


class RunMeta(StrictModel):
    """Identity of the run, recorded in the manifest."""

    name: str = Field(min_length=1, description="Human-readable run name, e.g. `daily_rebalance`.")
    as_of_date: AwareDatetime = Field(description="The timestamp the run is as of, timezone-aware (`2026-08-28T00:00:00Z`). Holding periods and the manifest use it; naive timestamps are rejected.")
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
    """How one input is loaded, what it depends on, how its portfolios are partitioned across calls, and how hard its source may be pushed."""

    loader: StepSpec = Field(description="The loader step: `(request: LoadRequest[, params]) -> DataFrame`, plain or `async def`.")
    depends_on: tuple[str, ...] = Field(
        default=(),
        description="Names of other `datasets` entries this one needs. The engine starts every dataset the moment its dependencies have loaded and hands their frames to the loader as `request.inputs`. Declare `portfolios` to receive the book's ids as `request.portfolio_ids`; a `per_portfolio` dataset depends on `portfolios` implicitly. Default: no dependencies, so the loader starts the moment the load stage does.",
    )
    scope: DatasetScope = Field(
        default="global",
        description="`global` (default): one call, and the dataset is visible to every assembly step. `per_portfolio`: the engine partitions the portfolio ids into batches and calls the loader once per batch, so a source that answers per account is driven by the engine rather than by the loader; a batch that fails, or that comes back without a portfolio's rows, fails those portfolios alone at stage `load` and the run carries on. A per-portfolio dataset is not passed to assembly — attach its columns in a rule instead.",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="How many portfolios one call of a `per_portfolio` loader is given, as `request.portfolio_ids`. `1` is a call per portfolio; a larger number batches a source that takes an id list; omitted, every portfolio goes in one call. Ignored for a `global` dataset, which is loaded by one call.",
    )
    rate_limit: str | RateLimitConfig | None = Field(
        default=None,
        description="How hard this input's source may be pushed: the name of a shared pool in the top-level `rate_limits`, or an inline bound private to this input. The loader receives it as `request.rate_limiter`. Omit for no limit.",
    )

    @model_validator(mode="after")
    def _shape_is_consistent(self) -> Self:
        if self.batch_size is not None and self.scope != "per_portfolio":
            msg = "batch_size applies only to a per_portfolio dataset; a global dataset is loaded by one call"
            raise ValueError(msg)
        duplicates = sorted(name for name, count in Counter(self.depends_on).items() if count > 1)
        if duplicates:
            msg = f"depends_on repeats {duplicates}"
            raise ValueError(msg)
        return self

    def dependencies(self) -> tuple[str, ...]:
        """The effective dependencies: the declared ones, with `portfolios` prepended for a `per_portfolio` dataset that does not declare it."""
        if self.scope == "per_portfolio" and "portfolios" not in self.depends_on:
            return ("portfolios", *self.depends_on)
        return self.depends_on

    def batches(self, portfolio_ids: tuple[PortfolioId, ...]) -> tuple[tuple[PortfolioId, ...], ...]:
        """How this dataset's calls are partitioned: one batch of every id when global, otherwise `batch_size` at a time."""
        if self.scope == "global":
            return (portfolio_ids,)
        size = self.batch_size or max(len(portfolio_ids), 1)
        return tuple(portfolio_ids[start : start + size] for start in range(0, len(portfolio_ids), size)) or ((),)


class InlinePortfolios(StrictModel):
    """The portfolio list written directly in the config instead of loaded: `{"ids": ["P7", "P2"]}`, or the bare array as shorthand.

    The written order is the solve order — the first id solves first. Only the `portfolios` dataset may
    be written inline; it costs nothing to load, so every dataset that depends on it starts at once.
    A book too large to write out, or one whose priorities come from data, uses a loader instead.
    """

    ids: tuple[str, ...] = Field(min_length=1, description="The portfolio ids, in solve order: the first id solves first, and the engine records `solve_order` as the position.")

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_list(cls, value: object) -> object:
        """A bare array is shorthand for the object form; containers become tuples because a before-validator forfeits JSON-mode coercion."""
        if isinstance(value, list | tuple):
            return {"ids": tuple(value)}
        if isinstance(value, dict):
            ids = value.get("ids")
            if isinstance(ids, list):
                return {**value, "ids": tuple(ids)}
        return value

    @field_validator("ids")
    @classmethod
    def _ids_are_unique_and_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not portfolio_id for portfolio_id in value):
            msg = "ids must be non-empty strings"
            raise ValueError(msg)
        duplicates = sorted(portfolio_id for portfolio_id, count in Counter(value).items() if count > 1)
        if duplicates:
            msg = f"ids repeat {duplicates}"
            raise ValueError(msg)
        return value

    def dependencies(self) -> tuple[str, ...]:
        """An inline book depends on nothing; it is the run's first fact."""
        return ()


type DatasetSpec = DatasetConfig | InlinePortfolios
"""One entry of `datasets`: a loaded input, or — for `portfolios` only — the inline list."""


def dataset_order(datasets: Mapping[str, DatasetSpec]) -> tuple[str, ...]:
    """Every dataset after the ones it depends on; raises ``ValueError`` naming the cycle when there is one.

    The config validator runs this to refuse a cyclic config, and the engine runs it again as a guard —
    a cycle that reached the scheduler would deadlock silently — and for a deterministic outcome order.
    """
    order: list[str] = []
    done: set[str] = set()
    path: list[str] = []

    def visit(name: str) -> None:
        if name in done:
            return
        if name in path:
            cycle = [*path[path.index(name) :], name]
            hint = " (a per_portfolio dataset depends on 'portfolios' implicitly)" if "portfolios" in cycle else ""
            msg = f"datasets depend on each other in a cycle: {' -> '.join(cycle)}{hint}"
            raise ValueError(msg)
        path.append(name)
        for dependency in datasets[name].dependencies():
            visit(dependency)
        path.pop()
        done.add(name)
        order.append(name)

    for name in datasets:
        visit(name)
    return tuple(order)


class ObjectiveConfig(StrictModel):
    """The objective as a list of named terms; the engine minimizes their sum."""

    sense: Literal["minimize"] = Field(default="minimize", description="Only minimization is supported; express a reward as a negative term (see `alpha`).")
    terms: tuple[StepSpec, ...] = Field(min_length=1, description='Objective-term steps from `terms.py`; every shipped term takes a non-negative `weight` (default "1").')


class SolverConfig(StrictModel):
    """Which cvxpy solver runs and with what options."""

    name: str = Field(
        default="CLARABEL",
        min_length=1,
        description="A solver the adapter knows and cvxpy has installed: `CLARABEL`, `OSQP`, `SCS`, `HIGHS` (installed with cvxpy) or `PIQP` (the `piqp` extra). Checked when the config resolves — by `validate-config`, at the start of `run`, and on every worker before it does any work; there is no automatic fallback.",
    )
    options: dict[str, float | int | bool | str] = Field(default_factory=dict, description='Passed verbatim to `Problem.solve(**options)`, e.g. `{"max_iter": 200}`.')
    time_limit_s: float | None = Field(
        default=None,
        gt=0,
        description="Wall-clock limit per solve in seconds, translated to the solver's own option; omit for no limit. Rejected at resolve for a solver with no such option (`PIQP`).",
    )
    verbose: bool = Field(default=False, description="Let the solver print its iteration log.")


class PostSolveConfig(StrictModel):
    """Tolerances for the independent, cvxpy-free verification of every solution."""

    violation_tol: float = Field(default=1e-6, gt=0, description="Maximum constraint violation accepted by the verifier; deliberately looser than the solver's own tolerance.")
    objective_rel_tol: float = Field(default=1e-5, gt=0, description="Relative tolerance between the recomputed and the solver-reported objective.")
    objective_abs_tol: float = Field(default=1e-9, gt=0, description="Absolute tolerance between the recomputed and the solver-reported objective.")


class ExecutionConfig(StrictModel):
    """What a failure means, and which portfolios wait for which.

    The schedule itself is derived: every portfolio builds in parallel, then solves once the
    higher-priority portfolios it depends on have solved. Which cluster the run provisions for itself
    and how many workers it has are settings (`PORTFOLIO_OPTIMIZER_CLUSTER`,
    `PORTFOLIO_OPTIMIZER_MAX_WORKERS`, ...), not config, so the same config hashes the same on a laptop
    and on a cluster — and nothing here changes a portfolio's answer, only how long the run takes.
    """

    on_error: OnError = Field(
        default="fail_fast",
        description="`fail_fast` stops at the first failed portfolio: every lower-priority portfolio is recorded as skipped. `continue` isolates the failure: only the portfolios that depended on it are skipped.",
    )
    dependencies: Dependencies = Field(
        default="overlap",
        description="Whether portfolios wait for each other, and which. `overlap` (default): a portfolio waits only for higher-priority portfolios that can trade a security it can trade too, on the side the run couples through (buys under `both` and `buy`, sells under `sell`). `all`: every higher-priority portfolio is a predecessor, one line — the same answer, for diagnosis. `none`: nothing waits and the whole book solves at once, which is right when no constraint reads what others traded. Under `overlap` the edge test also reads what each portfolio's constraints *declare*: a typed constraint row (a `kind` column) says whether it reads the chain and — through its `scope` — which securities it couples through, so a portfolio whose rows read no chain waits for nobody and a scoped `participation_limit` couples only through its scope. An opaque row (the function convention, or a desk's own vocabulary), a chain-aware objective term, or a solve step other than the shipped `cvxpy` one couples conservatively through the whole tradable set.",
    )


class RunConfig(StrictModel):
    """A portfolio-optimizer run: what to load, how to combine it, which rules and terms apply, and how to execute.

    Validated strictly: unknown keys are errors, money is written as strings, timestamps carry a zone.
    Step names are resolved and their parameters validated before any data is loaded.
    """

    schema_ref: str | None = Field(default=None, alias="$schema", description="Optional pointer to the JSON Schema for editor validation; ignored by the engine.")
    run: RunMeta = Field(description="Run identity.")
    datasets: dict[str, DatasetSpec] = Field(
        description="Named datasets, every one a frame. `portfolios` is required — the list of portfolio ids and their `solve_order` priorities (lower solves first, ties break on `portfolio_id`), loaded like any dataset or written inline as a list of ids — and is consumed by the engine rather than passed to assembly. `holdings`, `universe`, and `details` must be declared unless an assembly step produces them; `constraints` is engine-known but optional, and a run that omits it constrains nothing beyond the trade identity. Any other name is an extra dataset: available to every assembly step by name and carried into each portfolio's bundle as `data.extras` (reduced to the portfolio's rows when it has a `portfolio_id` column). Each dataset's loader starts the moment its `depends_on` dependencies have loaded — with none, the moment the load stage does."
    )
    rate_limits: dict[str, RateLimitConfig] = Field(
        default_factory=dict,
        description='Named rate-limit pools for loaders, e.g. `{"vendor_api": {"requests_per_second": 20, "max_in_flight": 8}}`. A dataset opts in with `rate_limit`; datasets naming the same pool share its budget.',
    )
    assembly: tuple[StepSpec, ...] = Field(
        default=(),
        description="Assembly steps from `assembly.py`, applied in order to the loaded datasets before the engine-known frames are validated: `join`, `union`, `select`, `drop`, or any custom `(frames: Frames[, params]) -> Frames` function. This is where analytics columns are attached to `holdings` and `universe`.",
    )
    rules: tuple[StepSpec, ...] = Field(default=(), description="Business-logic rule steps from `rules.py`, applied in order to each portfolio's bundle. Rules never see other portfolios.")
    solve_order: StepSpec | None = Field(
        default=None,
        description="Optional solve-order step from `solve_order.py`: `(data: PortfolioData[, params]) -> Decimal`, evaluated on each portfolio's ruled bundle. Lower keys solve first; ties break on `portfolio_id`. Replaces the portfolios frame's `solve_order` column.",
    )
    sides: Sides = Field(
        default="both",
        description="Which side the run trades. `both`: buys and sells in one problem, portfolios coupling through buys only. `buy`: buys alone — one variable per name, `w >= w0`, no sell vector, so a term that reads `sell` is refused at validate-config; portfolios couple through buys. `sell`: the mirror — `w <= w0`, no buy vector, portfolios couple through sells. The value selects the side profile that supplies the decision variables, the trade identity, the tradable set the dependency graph is built from, and the chain.",
    )
    objective: ObjectiveConfig = Field(description="What the optimizer minimizes.")
    solve: StepSpec = Field(
        default_factory=lambda: StepSpec(name="cvxpy"),
        description="The solve step from `solvers.py`: `(request: SolveRequest[, params]) -> SolveResult`. `cvxpy` (default) builds and solves the cvxpy problem from the terms and constraints; a qualified name plugs in a firm's own library or a pure function for one side.",
    )
    solver: SolverConfig = Field(default_factory=SolverConfig, description="cvxpy solver selection and options, read by the `cvxpy` solve step.")
    post_solve: PostSolveConfig = Field(default_factory=PostSolveConfig, description="Verification tolerances.")
    sink: StepSpec = Field(description="Sink step from `sinks.py`, called once with every solved portfolio's orders.")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Failure semantics and how dependencies between portfolios are derived.")

    @model_validator(mode="after")
    def _datasets_are_consistent(self) -> Self:
        portfolios = self.datasets.get("portfolios")
        if portfolios is None:
            msg = "datasets must declare 'portfolios': the list of portfolios the run is over"
            raise ValueError(msg)
        if isinstance(portfolios, DatasetConfig) and portfolios.scope != "global":
            msg = "portfolios must be a global dataset: it produces the ids a per_portfolio dataset is partitioned by"
            raise ValueError(msg)
        if not self.assembly:
            missing = [name for name in REQUIRED_DATASETS if name not in self.datasets]
            if missing:
                msg = f"datasets must declare {list(REQUIRED_DATASETS)}; missing {missing}; a run without assembly steps has nothing else to produce them"
                raise ValueError(msg)
        for name, dataset in self.datasets.items():
            if isinstance(dataset, InlinePortfolios):
                if name != "portfolios":
                    msg = f"datasets.{name}: only 'portfolios' may be written inline as a list of ids; every other dataset needs a loader"
                    raise ValueError(msg)
                continue
            if isinstance(dataset.rate_limit, str) and dataset.rate_limit not in self.rate_limits:
                msg = f"datasets.{name}: rate_limit {dataset.rate_limit!r} is not declared in rate_limits {sorted(self.rate_limits)}"
                raise ValueError(msg)
            self._check_dependencies(name, dataset)
        dataset_order(self.datasets)
        return self

    def _check_dependencies(self, name: str, dataset: DatasetConfig) -> None:
        unknown = [dependency for dependency in dataset.depends_on if dependency not in self.datasets]
        if unknown:
            msg = f"datasets.{name}: depends_on names unknown dataset(s) {unknown}; declared: {sorted(self.datasets)}"
            raise ValueError(msg)
        if name in dataset.dependencies():
            msg = f"datasets.{name}: a dataset cannot depend on itself"
            raise ValueError(msg)


def load_run_config(text: str) -> RunConfig:
    """Validate the JSON text of a run config."""
    return RunConfig.model_validate_json(text)


def config_sha256(config: RunConfig) -> str:
    """Hash of the validated config's canonical JSON form; source whitespace and the `$schema` pointer do not matter."""
    return hashlib.sha256(config.model_dump_json(exclude={"schema_ref"}).encode()).hexdigest()
