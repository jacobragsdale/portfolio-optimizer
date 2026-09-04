"""Strict models for the JSON run config.

Money and weights are written as JSON strings and validated in JSON mode so they become exact
``Decimal`` values; solver tolerances are floats because that is the solver's domain. Every field's
``description`` is emitted into the published JSON Schema (``configs/run-config.schema.json``), so
this module is also the schema's documentation.

The config is the *wiring* of a run: which inputs, combined how, filtered by which rules, minimized
against which terms, solved with what, checked how tightly, delivered where. What it is *as of* is a
run argument, and the run's name and tags are identity for people — none of that is in the config
hash, so two runs of one wiring hash the same whatever day they ran.
"""

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from portfolio_optimizer.domain.order_flow import OrderFlow
from portfolio_optimizer.domain.schemas import REQUIRED_DATASETS
from portfolio_optimizer.domain.types import PortfolioId, StrictModel

STEP_NAME_PATTERN = r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:)?[A-Za-z_][A-Za-z0-9_]*$"

type OnError = Literal["fail_fast", "continue"]
type Dependencies = Literal["overlap", "all"]
type DatasetScope = Literal["global", "per_portfolio"]

STEP_NAME_DESCRIPTION = (
    "A bare function name (`cap_single_name`), resolved in the template module for this kind of step or among the steps installed packages publish, "
    "or a qualified `package.module:function` importable by the engine and by any worker process."
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


CHECK_LABEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class CheckSpec(StepSpec):
    """A check step and the label its outcome is recorded under.

    Always an object: a check needs a `label` beside its `name`, because the manifest's `checks[]`
    and the file its violations are written to are keyed by it, and two checks of one function
    under different params are told apart by it.
    """

    label: str = Field(
        pattern=CHECK_LABEL_PATTERN,
        description="What this check's outcome is recorded as, in the manifest's `checks[]` and as `checks/<label>.csv` for the rows that failed; unique across the run's checks. Letters, digits, `_`, `.`, `-`.",
    )


class RunMeta(StrictModel):
    """Identity of the run, recorded in the manifest and kept out of the config hash."""

    name: str = Field(min_length=1, description="Human-readable run name, e.g. `daily_rebalance`.")
    tags: dict[str, str] = Field(default_factory=dict, description='Free-form string labels copied into the manifest, e.g. `{"desk": "tax-aware"}`.')


class DatasetConfig(StrictModel):
    """How one input is loaded, what it depends on, and how its portfolios are partitioned across calls."""

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
    max_in_flight: int | None = Field(
        default=None,
        ge=1,
        description="How many of this dataset's calls the engine keeps in flight at once. A slot is held for the whole call, so `batch_size: 1` with `max_in_flight: 8` is eight concurrent per-portfolio calls to the source. Omit for no bound.",
    )

    @model_validator(mode="after")
    def _shape_is_consistent(self) -> Self:
        if self.scope != "per_portfolio" and (self.batch_size is not None or self.max_in_flight is not None):
            msg = "batch_size and max_in_flight apply only to a per_portfolio dataset; a global dataset is loaded by one call"
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
        description="Which higher-priority portfolios a portfolio waits for. `overlap` (default): those that can trade a security it can trade too, on the side the run couples through (buys under `inflow`, sells under `outflow`, either under `rebalance`) — and only what its own constraints *declare* they read: a typed constraint row says whether it reads the chain and, through its `scope`, which securities it couples through, so a portfolio whose rows read no chain waits for nobody. A chain-aware objective term or a solve step other than the shipped `cvxpy` one couples conservatively through the whole tradable set; with nothing anywhere reading the chain, nothing waits. `all`: every higher-priority portfolio is a predecessor, one line — the same answer, for diagnosis.",
    )


TERM_DESCRIPTION = (
    'Objective terms, each an object whose `kind` names a typed term model — the shipped `linear` (`{"kind": "linear", "name": "alpha", "column": "alpha", "weight": "-1"}`), '
    "or a kind an installed package publishes. The engine minimizes their sum; a reward is a negative weight. Every term is validated against its model when the config resolves, "
    "and its `name` must be unique. Empty is a run whose solve step minimizes nothing."
)


class RunConfig(StrictModel):
    """A portfolio-optimizer run: what to load, how to combine it, which rules and terms apply, and how to execute.

    Validated strictly: unknown keys are errors, money is written as strings. Step names are
    resolved and their parameters validated — and every objective term is validated against its kind —
    before any data is loaded.
    """

    schema_ref: str | None = Field(default=None, alias="$schema", description="Optional pointer to the JSON Schema for editor validation; ignored by the engine.")
    run: RunMeta = Field(description="Run identity: a name and tags for people, recorded in the manifest and kept out of the config hash.")
    datasets: dict[str, DatasetSpec] = Field(
        description="Named datasets, every one a frame. `portfolios` is required — the list of portfolio ids and their `solve_order` priorities (lower solves first, ties break on `portfolio_id`), loaded like any dataset or written inline as a list of ids — and is consumed by the engine rather than passed to assembly. `holdings`, `universe`, and `details` must be declared unless an assembly step produces them; `constraints` is engine-known but optional, and a run that omits it constrains nothing beyond the trade identity and the spec's own bounds. Any other name is an extra dataset: available to every assembly step by name and carried into each portfolio's bundle as `data.extras` (reduced to the portfolio's rows when it has a `portfolio_id` column). Each dataset's loader starts the moment its `depends_on` dependencies have loaded — with none, the moment the load stage does."
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
    order_flow: OrderFlow = Field(
        description="The run's order flow: `inflow` — cash comes into the book, so the run buys: one variable per name, `w >= w0`, no sell vector, portfolios coupling through buys; `outflow`, the mirror: cash goes out, the run sells, coupling through sells; or `rebalance`: no cash moves on purpose, `w` may go either way inside its bounds with `buy` and `sell` the positive and negative parts of the change, portfolios coupling through every trade. A term or row that reads a side an inflow or an outflow lacks is refused, and so is a term that rewards `buy`, `sell`, or `trade` under a rebalance, where they are convex rather than affine. A desk's order flows are separate runs over one snapshot."
    )
    build: StepSpec = Field(
        default_factory=lambda: StepSpec(name="standard"),
        description="The build step: `(data: PortfolioData[, params]) -> ProblemSpec`, run per portfolio after its rules. `standard` (default, in `engine/build.py`) aligns the bundle to the sorted universe, computes weights and tax per dollar exactly, derives the bounds, and exports every extra column, flag, grouping, and account scalar by name; a qualified name plugs in a build that reads the bundle its own way — tax lots, a factor block, a different bounds policy.",
    )
    objective: tuple[dict[str, object], ...] = Field(default=(), description=TERM_DESCRIPTION)
    solve: StepSpec = Field(
        default_factory=lambda: StepSpec(name="cvxpy"),
        description='The solve step from `solvers.py`: `(request: SolveRequest[, params]) -> SolveResult`. `cvxpy` (default) renders the terms and the typed constraint rows and solves with the solver its params name (`{"name": "cvxpy", "params": {"solver": "CLARABEL", "time_limit_s": 60}}`); a qualified name plugs in a firm\'s own library or a pure function for one side.',
    )
    post_solve: PostSolveConfig = Field(default_factory=PostSolveConfig, description="Verification tolerances.")
    sink: StepSpec = Field(description="Sink step from `sinks.py`, called once with every solved portfolio's orders.")
    checks: tuple[CheckSpec, ...] = Field(
        default=(),
        description="Check steps from `checks.py`: `(frames: Frames, orders: DataFrame, solved: DataFrame[, params]) -> DataFrame`, each run once after the sink over every assembled dataset as the rules first saw it, the orders the run published, and the portfolios that solved, returning the rows the business rule examined with an `ok` flag. Recorded under its `label` as `passed`, `failed`, or `not_exercised` (it examined nothing: the book never put the rule to the test); a failed check fails the run.",
    )
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Failure semantics and how dependencies between portfolios are derived.")

    @field_validator("checks")
    @classmethod
    def _check_labels_are_unique(cls, value: tuple[CheckSpec, ...]) -> tuple[CheckSpec, ...]:
        duplicates = sorted(label for label, count in Counter(check.label for check in value).items() if count > 1)
        if duplicates:
            msg = f"checks repeat label(s) {duplicates}; every check is recorded under its own label"
            raise ValueError(msg)
        return value

    @field_validator("objective")
    @classmethod
    def _terms_name_a_kind(cls, value: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
        for index, term in enumerate(value):
            kind = term.get("kind")
            if not isinstance(kind, str) or not kind:
                msg = f"objective[{index}]: a term is an object with a `kind` naming its model"
                raise ValueError(msg)
        return value

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


HASH_EXCLUDES: frozenset[str] = frozenset({"schema_ref", "run"})
"""What the config hash leaves out: the schema pointer, and the run's name and tags, which are identity for people rather than wiring."""


def config_sha256(config: RunConfig) -> str:
    """Hash of the validated config's canonical JSON form: the wiring, so whitespace, the `$schema` pointer, and the `run` block do not matter."""
    return hashlib.sha256(config.model_dump_json(exclude=set(HASH_EXCLUDES)).encode()).hexdigest()
