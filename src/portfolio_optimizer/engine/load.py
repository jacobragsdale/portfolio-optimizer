"""Run the configured loaders as the config's dependency graph, apply the assembly steps, validate, and slice per portfolio.

Loading is the slow part of a real run — API calls and database queries, not files — so it is
asynchronous and driven by the DAG the config declares: every dataset is a task that starts the
moment the datasets its ``depends_on`` names (plus ``portfolios``, for a ``per_portfolio`` dataset)
have loaded, and one with no dependencies starts the moment the stage does. ``async def`` loaders run
as tasks on the event loop; plain ``def`` loaders run in worker threads so a blocking driver never
stalls the loop. Each dataset's rate-limit pool is created here and shared by every dataset that
names it. A loader receives its dependencies' frames as ``request.inputs``.

``portfolios`` is engine-known but scheduled like any other node: once it loads — or is read straight
from an inline config list, which costs nothing — its frame is validated, sorted by ``solve_order``
then ``portfolio_id``, and the ids still in the run reach its direct dependents as
``request.portfolio_ids``.

A dataset's ``scope`` says how it is partitioned. A ``global`` dataset is one call and is what the
assembly steps see. A ``per_portfolio`` dataset is the engine's own fan-out: the ids are cut into
batches of ``batch_size`` and the loader is called once per batch, so a source that answers one
account at a time is driven by the engine — schedulable, throttled by one limiter, and overlapping
every dataset that does not depend on it — rather than by a loader that fans out privately.

Failure is split along the same line. A **structural** problem rejects the run: a required dataset
missing, a schema violated, a global loader that raised, or a per-portfolio dataset no batch of which
came back — and a dataset downstream of one of those is skipped, never called, and named beside the
failure it was blocked on. A **coverage** problem fails only the portfolios it touches — one batch
that raised, or a portfolio with no ``details`` row — which are recorded as failures at stage
``load`` and carried into the run so every other portfolio still solves; a dependent dataset simply
loads for the surviving ids.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.models import DatasetConfig, InlinePortfolios, dataset_order
from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.domain.data import PREVALIDATED_FRAMES, Frames, LoadRequest, PortfolioData, details_from_frame
from portfolio_optimizer.domain.frames import FrameSchemaError, empty_frame, validate_frame
from portfolio_optimizer.domain.results import AssemblyAuditRecord, PortfolioFailure
from portfolio_optimizer.domain.schemas import CONSTRAINTS, DATASET_SCHEMAS, PORTFOLIOS, REQUIRED_FRAMES
from portfolio_optimizer.domain.types import PortfolioId, StrictModel
from portfolio_optimizer.engine.hashing import frame_sha256
from portfolio_optimizer.ratelimit import RateLimiter

log = logging.getLogger(__name__)


type _BatchResult = pd.DataFrame
"""What one call of a loader returns. Every dataset is a frame, so there is only the one shape."""


class LoadError(ValueError):
    """A loader returned something other than its contract promises, or a dataset failed its schema."""


class AssemblyError(ValueError):
    """An assembly step refused its input or returned something other than ``Frames``."""


class DatasetAudit(StrictModel):
    """Provenance of one loaded dataset, recorded in the manifest as is: which loader, its hashes, what came back, and how long it took."""

    name: str
    loader_qualname: str
    loader_source_sha256: str
    params_sha256: str
    rows: int
    columns: tuple[str, ...]
    content_sha256: str
    load_time_s: float
    batches: int = 1
    """Calls the engine made to this dataset's loader: one for a global dataset, one per batch for a per-portfolio one, zero for a book written inline in the config."""

    rejected: int = 0
    """Portfolios whose batch failed, and which are therefore failed at stage ``load``."""

    depends_on: tuple[str, ...] = ()
    """The dataset's effective dependencies: what its config declared, plus ``portfolios`` for a per-portfolio dataset."""

    started_s: float = 0.0
    """Seconds after the load stage began that this dataset started: its wait on dependencies. Beside ``load_time_s``, it shows how the stage overlapped."""


@dataclass(frozen=True, slots=True)
class LoadedDatasets:
    """Everything the loaders returned, before assembly.

    ``frames`` are the global datasets, the only ones assembly sees; ``per_portfolio`` are the batched
    ones, concatenated back into a frame each and merged in after assembly has run. ``rejected`` names
    the portfolios a batch failed to load, which never reach a build.
    """

    portfolio_ids: tuple[PortfolioId, ...]
    solve_orders: Mapping[PortfolioId, int]
    frames: Mapping[str, pd.DataFrame]
    per_portfolio: Mapping[str, pd.DataFrame]
    rejected: Mapping[PortfolioId, PortfolioFailure]
    audits: tuple[DatasetAudit, ...]


@dataclass(frozen=True, slots=True)
class AssembledDatasets:
    """Engine-known frames after the assembly steps and schema validation, ready to slice per portfolio.

    ``portfolio_ids`` are in ascending ``solve_order`` then ``portfolio_id``; ``solve_orders`` keeps
    the column's values, the solve-order key when no step computes one. ``extras`` are the remaining
    datasets — every one that is not engine-known — carried into each portfolio's bundle. ``audits``
    record what each assembly step did, for the manifest.
    """

    portfolio_ids: tuple[PortfolioId, ...]
    solve_orders: Mapping[PortfolioId, int]
    holdings: pd.DataFrame
    universe: pd.DataFrame
    details: pd.DataFrame
    constraints: pd.DataFrame
    extras: Mapping[str, pd.DataFrame]
    rejected: Mapping[PortfolioId, PortfolioFailure]
    """Portfolios that could not be loaded — a failed batch or a missing ``details`` row — failed at stage ``load`` so the rest of the book still runs."""

    as_of_date: datetime
    audits: tuple[AssemblyAuditRecord, ...]


@dataclass(frozen=True, slots=True)
class _Book:
    """What the ``portfolios`` dataset resolves to: the run's ids in solve order, and each id's priority."""

    ids: tuple[PortfolioId, ...]
    solve_orders: Mapping[PortfolioId, int]


@dataclass(frozen=True, slots=True)
class _Loaded:
    name: str
    frame: pd.DataFrame
    rejected: Mapping[PortfolioId, PortfolioFailure]
    audit: DatasetAudit
    per_portfolio: bool
    book: _Book | None = None
    """Set on the ``portfolios`` outcome alone, once its frame has been validated and ordered."""


@dataclass(frozen=True, slots=True)
class _Failed:
    name: str
    error: Exception


@dataclass(frozen=True, slots=True)
class _Skipped:
    """A dataset never loaded because something upstream failed; ``blocked_on`` names the root failures, transitively."""

    name: str
    blocked_on: tuple[str, ...]


type _Outcome = _Loaded | _Failed | _Skipped
"""How one dataset's task ended. Outcomes are values, never raised, so awaiting a shared dependency task cannot blow up its awaiters."""


@dataclass(frozen=True, slots=True)
class _Plan:
    """One dataset's schedule entry: its loader (``None`` for an inline book), its config, and its effective dependencies."""

    name: str
    step: ResolvedStep | None
    spec: DatasetConfig | InlinePortfolios
    dependencies: tuple[str, ...]


type _RequestFactory = Callable[[str, DatasetConfig, tuple[PortfolioId, ...], Frames], LoadRequest]
"""Builds one loader call's request: dataset name, its config, the ids this call is for, and its view of the input frames."""


def load_datasets(resolved: ResolvedConfig, *, data_root: Path, run_id: str) -> LoadedDatasets:
    """Run every loader on a fresh event loop; the entry point for the synchronous runner.

    Code that already runs an event loop must ``await`` :func:`load_datasets_async` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(load_datasets_async(resolved, data_root=data_root, run_id=run_id))
    msg = "load_datasets cannot be called from inside an event loop; await load_datasets_async instead"
    raise RuntimeError(msg)


async def load_datasets_async(resolved: ResolvedConfig, *, data_root: Path, run_id: str) -> LoadedDatasets:
    """Run every dataset's loader as a task that starts the moment its dependencies have loaded.

    The tasks are the config's dependency DAG: each awaits the outcomes of the datasets its entry
    depends on and then calls its own loader, so a dataset with no dependencies starts immediately.
    Every dataset's outcome is collected — a failure in one does not cancel the others, and a
    dependent of a failed dataset is skipped rather than called — and all failures are reported
    together. Input problems (``ValueError``, ``KeyError``) become a :class:`LoadError`; any other
    failure keeps its type so the caller's exit code stays right.
    """
    config = resolved.config
    order = dataset_order(config.datasets)  # the model validator already refused a cycle; guard again so a hand-built config cannot deadlock the scheduler
    pools = {name: RateLimiter(pool.to_limit(), name=name) for name, pool in config.rate_limits.items()}
    private: dict[str, RateLimiter] = {}

    def limiter_for(dataset: str, spec: DatasetConfig) -> RateLimiter:
        """The shared pool the input names, a private limiter from its inline bound, or no limit.

        An inline bound is built once per input and kept, because a ``per_portfolio`` input asks for one
        limiter per batch and a bound that is not shared between them is not a bound at all.
        """
        bound = spec.rate_limit
        if bound is None:
            return RateLimiter.unlimited()
        if isinstance(bound, str):
            return pools[bound]
        if dataset not in private:
            private[dataset] = RateLimiter(bound.to_limit(), name=dataset)
        return private[dataset]

    def request(name: str, spec: DatasetConfig, ids: tuple[PortfolioId, ...], inputs: Frames) -> LoadRequest:
        return LoadRequest(dataset=name, portfolio_ids=ids, as_of_date=config.run.as_of_date, data_root=data_root, run_id=run_id, rate_limiter=limiter_for(name, spec), inputs=inputs)

    plans = {name: _Plan(name=name, step=resolved.loaders.get(name), spec=spec, dependencies=spec.dependencies()) for name, spec in config.datasets.items()}
    started = time.perf_counter()
    tasks: dict[str, asyncio.Task[_Outcome]] = {}
    async with asyncio.TaskGroup() as group:
        for name in order:
            tasks[name] = group.create_task(_run_dataset(plans[name], tasks, request, stage_started=started, run_id=run_id))
    outcomes = {name: tasks[name].result() for name in order}
    for limiter in (*pools.values(), *private.values()):
        log.info("rate limit %r: %d request(s), %.2fs spent waiting", limiter.name, limiter.acquired, limiter.waited_s, extra={"run_id": run_id, "stage": "load"})
    return _collect(outcomes, run_id=run_id)


async def _run_dataset(plan: _Plan, tasks: Mapping[str, "asyncio.Task[_Outcome]"], request: _RequestFactory, *, stage_started: float, run_id: str) -> _Outcome:
    """Wait for the dataset's dependencies, then load it; a failed dependency skips it instead of calling its loader.

    Dependencies are awaited one at a time rather than gathered: their tasks all run regardless, so
    the total wait is the slowest one's, and ``gather`` would propagate one dependent's cancellation
    into a task other datasets share. Outcomes are values, never raised — the only exception an await
    here can see is the group unwinding, which is the caller's business.
    """
    upstream: dict[str, _Loaded] = {}
    causes: list[str] = []
    for dependency in plan.dependencies:
        outcome = await tasks[dependency]
        if isinstance(outcome, _Loaded):
            upstream[dependency] = outcome
        elif isinstance(outcome, _Failed):
            if dependency not in causes:
                causes.append(dependency)
        else:
            causes.extend(cause for cause in outcome.blocked_on if cause not in causes)
    if causes:
        log.warning("dataset %r not loaded: %s failed", plan.name, ", ".join(causes), extra={"run_id": run_id, "stage": "load"})
        return _Skipped(plan.name, tuple(causes))
    started_s = time.perf_counter() - stage_started
    if isinstance(plan.spec, InlinePortfolios):
        return _inline_portfolios(plan.spec, started_s, run_id=run_id)
    outcome = await _load_dataset(plan, plan.spec, _alive_ids(upstream), Frames({name: loaded.frame for name, loaded in upstream.items()}), request, started_s=started_s, run_id=run_id)
    if plan.name == "portfolios" and isinstance(outcome, _Loaded):
        return _finish_portfolios(outcome, run_id=run_id)
    return outcome


def _alive_ids(upstream: Mapping[str, _Loaded]) -> tuple[PortfolioId, ...]:
    """The ids this dataset's calls are for: the book in solve order, minus portfolios a dependency already rejected.

    Empty when the dataset did not declare ``portfolios`` among its dependencies — the ids are data
    from another dataset, and a loader that wants them says so in ``depends_on``.
    """
    portfolios = upstream.get("portfolios")
    if portfolios is None or portfolios.book is None:
        return ()
    dead = {portfolio_id for loaded in upstream.values() for portfolio_id in loaded.rejected}
    return tuple(portfolio_id for portfolio_id in portfolios.book.ids if portfolio_id not in dead)


def _inputs_for(inputs: Frames, ids: tuple[PortfolioId, ...]) -> Frames:
    """A per-portfolio batch's view of its inputs: a frame with a ``portfolio_id`` column cut to the batch's rows, any other passed whole.

    The same convention :func:`slice_portfolio` applies to extras. O(batches x rows) per input; if a
    book ever has thousands of batches, pre-split each input once with ``groupby("portfolio_id")``.
    """
    wanted = [str(portfolio_id) for portfolio_id in ids]
    return Frames({name: frame[frame["portfolio_id"].isin(wanted)].reset_index(drop=True) if "portfolio_id" in frame.columns else frame for name, frame in inputs.items()})


async def _load_dataset(plan: _Plan, dataset: DatasetConfig, ids: tuple[PortfolioId, ...], inputs: Frames, request: _RequestFactory, *, started_s: float, run_id: str) -> _Loaded | _Failed:
    """Call the loader once per batch, concurrently, and combine what came back.

    A global dataset has one batch and any failure is the dataset's. A per-portfolio dataset fails only
    the portfolios of the batches that raised — unless no batch survived, which is the source being
    down rather than a portfolio being bad, and rejects the run like any other dataset failure.
    """
    step = plan.step
    if step is None:  # pragma: no cover - only an inline book has no loader, and it never reaches here
        msg = f"dataset {plan.name!r} has no loader"
        raise LoadError(msg)
    started = time.perf_counter()
    batches = dataset.batches(ids)
    per_portfolio = dataset.scope == "per_portfolio"
    requests = [request(plan.name, dataset, batch, _inputs_for(inputs, batch) if per_portfolio else inputs) for batch in batches]
    results: list[_BatchResult | BaseException] = list(await asyncio.gather(*(_load_frame(step, batch_request) for batch_request in requests), return_exceptions=True))
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result  # cancellation and the like are the caller's, not a dataset's failure to report
    errors = [(batch, _unwrap(result)) for batch, result in zip(batches, results, strict=True) if isinstance(result, Exception)]
    good = [result for result in results if not isinstance(result, BaseException)]
    if errors and (dataset.scope == "global" or not good):
        first = errors[0][1]
        log.error("dataset %r failed after %.2fs: %s: %s", plan.name, time.perf_counter() - started, type(first).__name__, _describe(first), extra={"run_id": run_id, "stage": "load"})
        return _Failed(plan.name, first)
    rejected = {portfolio_id: _load_failure(portfolio_id, plan.name, error) for batch, error in errors for portfolio_id in batch}
    for batch, error in errors:
        log.error("dataset %r: batch of %d portfolio(s) failed: %s: %s", plan.name, len(batch), type(error).__name__, _describe(error), extra={"run_id": run_id, "stage": "load"})
    elapsed = time.perf_counter() - started
    frame = _combine([_as_frame(batch) for batch in good])
    schema = PORTFOLIOS if plan.name == "portfolios" else DATASET_SCHEMAS.get(plan.name)
    key = (
        schema.key if schema is not None and all(column in frame.columns for column in schema.key) else ()
    )  # a frame missing its key will fail its schema later; hash it unkeyed rather than crash here
    log.info("dataset %r loaded: %d row(s) in %d batch(es), %.2fs", plan.name, len(frame), len(batches), elapsed, extra={"run_id": run_id, "stage": "load"})
    audit = _audit(plan.name, step, frame, key, elapsed, len(batches), len(rejected), depends_on=plan.dependencies, started_s=started_s)
    return _Loaded(plan.name, frame, rejected, audit, per_portfolio)


def _finish_portfolios(outcome: _Loaded, *, run_id: str) -> _Loaded | _Failed:
    """Validate the loaded book against its schema and derive the run's ids and priorities.

    Dependents receive the sorted frame with ``solve_order`` filled in, so every consumer sees the
    canonical book whatever shape the loader returned it in.
    """
    try:
        validate_frame(outcome.frame, PORTFOLIOS)
    except FrameSchemaError as error:
        return _Failed("portfolios", error)
    keyed = outcome.frame if "solve_order" in outcome.frame.columns else outcome.frame.assign(solve_order=pd.Series(0, index=outcome.frame.index, dtype="Int64"))
    ordered = keyed.assign(portfolio_id=keyed["portfolio_id"].astype(str)).sort_values(["solve_order", "portfolio_id"], kind="stable").reset_index(drop=True)
    ids = tuple(PortfolioId(str(value)) for value in ordered["portfolio_id"])
    solve_orders = {PortfolioId(str(portfolio_id)): int(value) for portfolio_id, value in zip(ordered["portfolio_id"], ordered["solve_order"], strict=True)}
    log.info("portfolio list loaded: %d portfolio(s)", len(ids), extra={"run_id": run_id, "stage": "load"})
    return replace(outcome, frame=ordered, book=_Book(ids=ids, solve_orders=solve_orders))


def _inline_portfolios(spec: InlinePortfolios, started_s: float, *, run_id: str) -> _Loaded:
    """The book written in the config, materialised as the frame a loader would have returned.

    The written order is the solve order, recorded as each id's position. The audit names ``config``
    as the loader and hashes the literal ids where a loader's source and params would be hashed.
    """
    frame = pd.DataFrame({"portfolio_id": pd.Series(list(spec.ids), dtype="string"), "solve_order": pd.Series(range(len(spec.ids)), dtype="Int64")})
    ids = tuple(PortfolioId(portfolio_id) for portfolio_id in spec.ids)
    digest = hashlib.sha256(json.dumps(list(spec.ids), separators=(",", ":")).encode()).hexdigest()
    audit = DatasetAudit(
        name="portfolios",
        loader_qualname="config",
        loader_source_sha256=digest,
        params_sha256=digest,
        rows=len(frame),
        columns=("portfolio_id", "solve_order"),
        content_sha256=frame_sha256(frame, PORTFOLIOS.key),
        load_time_s=0.0,
        batches=0,
        started_s=started_s,
    )
    log.info("portfolio list read from the config: %d portfolio(s)", len(ids), extra={"run_id": run_id, "stage": "load"})
    return _Loaded("portfolios", frame, {}, audit, per_portfolio=False, book=_Book(ids=ids, solve_orders={portfolio_id: index for index, portfolio_id in enumerate(ids)}))


def _collect(outcomes: Mapping[str, _Outcome], *, run_id: str) -> LoadedDatasets:
    """Combine every dataset's outcome into the loaded datasets, or raise naming every failure and what it blocked."""
    failures = [outcome for outcome in outcomes.values() if isinstance(outcome, _Failed)]
    if failures:
        _raise_load_failures(failures, [outcome for outcome in outcomes.values() if isinstance(outcome, _Skipped)])
    loaded = {name: outcome for name, outcome in outcomes.items() if isinstance(outcome, _Loaded)}
    portfolios = loaded["portfolios"]
    if portfolios.book is None:  # pragma: no cover - the portfolios outcome always carries its book
        msg = "the portfolios dataset resolved to no book"
        raise LoadError(msg)
    frames = {name: outcome.frame for name, outcome in loaded.items() if not outcome.per_portfolio and name != "portfolios"}
    per_portfolio = {name: outcome.frame for name, outcome in loaded.items() if outcome.per_portfolio}
    rejected = {portfolio_id: failure for outcome in loaded.values() for portfolio_id, failure in outcome.rejected.items()}
    if rejected:
        log.warning("%d portfolio(s) could not be loaded and will not be solved", len(rejected), extra={"run_id": run_id, "stage": "load"})
    return LoadedDatasets(
        portfolio_ids=portfolios.book.ids,
        solve_orders=portfolios.book.solve_orders,
        frames=frames,
        per_portfolio=per_portfolio,
        rejected=rejected,
        audits=tuple(outcome.audit for outcome in loaded.values()),
    )


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One batch's frame as is, several concatenated; the content hash sorts by key, so batch order never reaches the manifest."""
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _as_frame(batch: _BatchResult) -> pd.DataFrame:
    if not isinstance(batch, pd.DataFrame):  # pragma: no cover - _load_frame already refuses anything else
        msg = f"expected a DataFrame, got {type(batch).__name__}"
        raise LoadError(msg)
    return batch


def _load_failure(portfolio_id: PortfolioId, dataset: str, error: Exception) -> PortfolioFailure:
    """A portfolio whose batch of ``dataset`` did not come back; it never reaches a build."""
    return PortfolioFailure(portfolio_id=portfolio_id, stage="load", error_type=type(error).__name__, message=f"dataset {dataset!r} did not load for this portfolio: {_describe(error)}")


def _leaves(error: BaseException) -> tuple[BaseException, ...]:
    """Every real failure inside a possibly nested exception group; the error itself when it is not one."""
    if isinstance(error, BaseExceptionGroup):
        return tuple(leaf for inner in error.exceptions for leaf in _leaves(inner))
    return (error,)


def _unwrap(error: Exception) -> Exception:
    """The one failure a loader's exception group wraps, or the group itself when it holds more than one.

    A loader that fans out privately runs its calls in a ``TaskGroup`` — :func:`~portfolio_optimizer.ratelimit.fan_out`
    does — and the first failure cancels the others, so what reaches the engine is almost always a single
    real error inside a group. Unwrapping keeps the type the exit code is chosen from and the message that
    names the missing file, instead of the group's ``unhandled errors in a TaskGroup``.
    """
    leaves = _leaves(error)
    if len(leaves) == 1 and isinstance(leaves[0], Exception):
        return leaves[0]
    return error


def _describe(error: Exception) -> str:
    """How a failure reads in a message: a group with several failures spells them out, anything else prints itself."""
    leaves = _leaves(error)
    if len(leaves) == 1:
        return str(error)
    return "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in leaves)


def _raise_load_failures(failures: list[_Failed], skipped: list[_Skipped]) -> None:
    """Raise one error for every failed dataset, naming what each blocked; an infrastructure error keeps its own type.

    A batch that failed more than once over — a fan-out loader against a source that is down — arrives as
    an exception group :func:`_unwrap` cannot collapse, so classification looks *inside* it: the first
    failure that is not an input error is raised as itself (skipped datasets attached as notes, so its
    message and the exit code stay the source's), and every failure is in the log line and in each
    portfolio's record.
    """
    blocked: dict[tuple[str, ...], list[str]] = {}
    for outcome in skipped:
        blocked.setdefault(outcome.blocked_on, []).append(outcome.name)
    notes = [f"not loaded because {', '.join(causes)} failed: {', '.join(names)}" for causes, names in blocked.items()]
    hard = [leaf for failure in failures for leaf in _leaves(failure.error) if not isinstance(leaf, ValueError | KeyError)]
    if hard:
        for note in notes:
            hard[0].add_note(note)
        raise hard[0]
    raise LoadError("; ".join([*(f"{failure.name}: {_describe(failure.error)}" for failure in failures), *notes]))


async def _load_frame(step: ResolvedStep, request: LoadRequest) -> pd.DataFrame:
    result = await step.invoke_async(request=request)
    if not isinstance(result, pd.DataFrame):
        msg = f"loader {step.qualname!r} for {request.dataset!r} returned {type(result).__name__}, expected DataFrame"
        raise LoadError(msg)
    return result


def _audit(
    name: str, step: ResolvedStep, frame: pd.DataFrame, key: tuple[str, ...], load_time_s: float, batches: int = 1, rejected: int = 0, *, depends_on: tuple[str, ...] = (), started_s: float = 0.0
) -> DatasetAudit:
    return DatasetAudit(
        name=name,
        loader_qualname=step.qualname,
        loader_source_sha256=step.source_sha256,
        params_sha256=step.params_sha256,
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        content_sha256=frame_sha256(frame, key),
        load_time_s=load_time_s,
        batches=batches,
        rejected=rejected,
        depends_on=depends_on,
        started_s=started_s,
    )


def assemble(loaded: LoadedDatasets, resolved: ResolvedConfig, *, run_id: str) -> AssembledDatasets:
    """Run the assembly steps in order over the global datasets, then validate every engine-known frame against its schema.

    A step that raises ``ValueError`` or ``KeyError`` (a missing dataset, a violated cardinality, a
    column conflict) becomes an :class:`AssemblyError` naming the step; any other exception keeps its
    type. Per-portfolio datasets are not in ``frames`` at all — a step that names one is told so —
    and are merged back in once the steps have run, because a step sees whole datasets and those
    arrive a batch at a time; attach their columns in a rule instead.

    After the last step the four required frames must exist and each engine-known frame that is
    present must satisfy its schema — structural problems, which reject the run. A portfolio with no
    ``details`` row is a coverage problem: it joins ``rejected`` and the rest of the book runs.
    """
    frames = Frames(loaded.frames)
    audits: list[AssemblyAuditRecord] = []
    for index, step in enumerate(resolved.assembly):
        where = f"assembly[{index}] {step.qualname}"
        before = frames
        try:
            result = step.invoke(frames=before)
        except (ValueError, KeyError) as error:
            msg = f"{where}: {_message(error)}{_sharded_hint(error, loaded.per_portfolio)}"
            raise AssemblyError(msg) from error
        if not isinstance(result, Frames):
            msg = f"{where}: returned {type(result).__name__}, expected Frames"
            raise AssemblyError(msg)
        frames = result
        audits.append(
            AssemblyAuditRecord(
                qualname=step.qualname,
                source_sha256=step.source_sha256,
                params_sha256=step.params_sha256,
                rows_in=before.row_counts(),
                rows_out=frames.row_counts(),
                columns_added=_columns_added(before, frames),
            )
        )
        log.info("assembly step %r applied: %s", step.qualname, ", ".join(f"{name}={rows}" for name, rows in frames.row_counts().items()), extra={"run_id": run_id, "stage": "assembly"})
    frames = Frames({**frames, **loaded.per_portfolio})
    missing_frames = [name for name in REQUIRED_FRAMES if name not in frames]
    if missing_frames:
        msg = f"after assembly, required datasets are missing {missing_frames}; declare a loader for each or produce it in an assembly step"
        raise LoadError(msg)
    failures: list[str] = []
    for name, schema in DATASET_SCHEMAS.items():
        if name not in frames:
            continue  # only REQUIRED_FRAMES must exist; constraints is engine-known but optional
        try:
            validate_frame(frames[name], schema)
        except FrameSchemaError as error:
            failures.extend(f"{name}: {failure}" for failure in error.failures)
    if failures:
        raise LoadError("; ".join(failures))
    details = frames["details"]
    rejected = _rejections(loaded, details, run_id=run_id)
    return AssembledDatasets(
        portfolio_ids=loaded.portfolio_ids,
        solve_orders=loaded.solve_orders,
        holdings=frames["holdings"],
        universe=frames["universe"],
        details=details,
        constraints=frames.get("constraints", empty_frame(CONSTRAINTS)),
        extras={name: frame for name, frame in frames.items() if name not in DATASET_SCHEMAS},
        rejected=rejected,
        as_of_date=resolved.config.run.as_of_date,
        audits=tuple(audits),
    )


def _rejections(loaded: LoadedDatasets, details: pd.DataFrame, *, run_id: str) -> dict[PortfolioId, PortfolioFailure]:
    """Portfolios that cannot be built: a batch that failed, or no ``details`` row to build from."""
    rejected = dict(loaded.rejected)
    covered = {str(value) for value in details["portfolio_id"]}
    for portfolio_id in loaded.portfolio_ids:
        if portfolio_id in rejected or portfolio_id in covered:
            continue
        rejected[portfolio_id] = PortfolioFailure(portfolio_id=portfolio_id, stage="load", error_type="MissingInput", message="no details for this portfolio")
    if len(rejected) == len(loaded.portfolio_ids) and rejected:
        log.error("no portfolio has the inputs to be built", extra={"run_id": run_id, "stage": "assembly"})
    return rejected


def _sharded_hint(error: Exception, per_portfolio: Mapping[str, pd.DataFrame]) -> str:
    """Say why a per-portfolio dataset the step asked for is not among the frames, which is otherwise a bare missing name."""
    named = sorted(name for name in per_portfolio if name in _message(error))
    if not named:
        return ""
    plural = "are per_portfolio datasets" if len(named) > 1 else "is a per_portfolio dataset"
    return f" ({', '.join(repr(name) for name in named)} {plural}, which assembly never sees; attach their columns in a rule instead)"


def _message(error: Exception) -> str:
    """``KeyError`` quotes its message on ``str()``; take the message itself."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def _columns_added(before: Frames, after: Frames) -> dict[str, tuple[str, ...]]:
    added: dict[str, tuple[str, ...]] = {}
    for name, frame in after.items():
        previous = {str(column) for column in before[name].columns} if name in before else set()
        new = tuple(str(column) for column in frame.columns if str(column) not in previous)
        if new:
            added[name] = new
    return added


def slice_portfolio(assembled: AssembledDatasets, portfolio_id: PortfolioId) -> PortfolioData:
    """Build the validated per-portfolio bundle: its own holdings, constraint rows, and extras rows, and the whole universe.

    The schema frames are marked prevalidated: :func:`assemble` validated them, the universe is
    passed whole, and the holdings slice is a row subset, which keeps every per-column check and the
    key's uniqueness.
    """
    details = details_from_frame(assembled.details, portfolio_id)
    holdings = assembled.holdings[assembled.holdings["portfolio_id"] == portfolio_id].reset_index(drop=True)
    extras = {name: _rows_for(frame, portfolio_id) for name, frame in assembled.extras.items()}
    return PortfolioData(
        details=details,
        holdings=holdings,
        universe=assembled.universe.reset_index(drop=True),
        constraints=_rows_for(assembled.constraints, portfolio_id),
        as_of_date=assembled.as_of_date,
        extras=extras,
        prevalidated=PREVALIDATED_FRAMES,
    )


def _rows_for(frame: pd.DataFrame, portfolio_id: PortfolioId) -> pd.DataFrame:
    """A per-portfolio dataset (one with a ``portfolio_id`` column) reduced to this portfolio; a global one passed whole."""
    if "portfolio_id" in frame.columns:
        return frame[frame["portfolio_id"] == portfolio_id].reset_index(drop=True)
    return frame.reset_index(drop=True)
