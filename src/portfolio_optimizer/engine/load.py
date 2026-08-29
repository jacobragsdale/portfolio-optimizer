"""Run the configured loaders once, apply the assembly steps, validate, and slice per portfolio.

Loading is the slow part of a real run — API calls and database queries, not files — so it is
asynchronous: the portfolio list loads first (its ids are part of every other request), and then
every dataset loader starts at once. ``async def`` loaders run as tasks on the event loop; plain
``def`` loaders run in worker threads so a blocking driver never stalls the loop. Each dataset's
rate-limit pool is created here and shared by every dataset that names it.
"""

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.models import DatasetConfig
from portfolio_optimizer.config.resolve import ResolvedConfig, ResolvedStep
from portfolio_optimizer.domain.data import Frames, LoadRequest, PortfolioData, details_from_frame, style_constraints_from_mapping
from portfolio_optimizer.domain.frames import FrameSchemaError, validate_frame
from portfolio_optimizer.domain.results import AssemblyAuditRecord
from portfolio_optimizer.domain.schemas import DATASET_SCHEMAS, PORTFOLIOS, REQUIRED_FRAMES
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.hashing import frame_sha256, json_sha256
from portfolio_optimizer.ratelimit import RateLimiter

log = logging.getLogger(__name__)


class LoadError(ValueError):
    """A loader returned something other than its contract promises, or a dataset failed its schema."""


class AssemblyError(ValueError):
    """An assembly step refused its input or returned something other than ``Frames``."""


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Provenance of one loaded dataset for the manifest."""

    name: str
    loader_qualname: str
    loader_source_sha256: str
    params_sha256: str
    rows: int
    columns: tuple[str, ...]
    content_sha256: str
    load_time_s: float


@dataclass(frozen=True, slots=True)
class LoadedDatasets:
    """Everything the loaders returned, before assembly."""

    portfolio_ids: tuple[PortfolioId, ...]
    frames: Mapping[str, pd.DataFrame]
    constraints: Mapping[str, Mapping[str, object]]
    audits: tuple[DatasetAudit, ...]
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class AssembledDatasets:
    """Engine-known frames after the assembly steps and schema validation, ready to slice per portfolio.

    ``extras`` are the remaining datasets — every one that is not engine-known — carried into each
    portfolio's bundle. ``audits`` record what each assembly step did, for the manifest.
    """

    portfolio_ids: tuple[PortfolioId, ...]
    holdings: pd.DataFrame
    universe: pd.DataFrame
    details: pd.DataFrame
    targets: pd.DataFrame
    extras: Mapping[str, pd.DataFrame]
    constraints: Mapping[str, Mapping[str, object]]
    as_of: datetime
    audits: tuple[AssemblyAuditRecord, ...]


@dataclass(frozen=True, slots=True)
class _Loaded:
    name: str
    frame: pd.DataFrame | None
    constraints: Mapping[str, Mapping[str, object]] | None
    audit: DatasetAudit


@dataclass(frozen=True, slots=True)
class _Failed:
    name: str
    error: Exception


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
    """Invoke the portfolio-list loader, then every dataset loader concurrently, each exactly once.

    Every dataset's outcome is collected — a failure in one does not cancel the others — and all
    failures are reported together. Input problems (``ValueError``, ``KeyError``) become a
    :class:`LoadError`; any other failure keeps its type so the caller's exit code stays right.
    """
    config = resolved.config
    as_of = config.run.as_of
    pools = {name: RateLimiter(pool.to_limit(), name=name) for name, pool in config.rate_limits.items()}
    private: list[RateLimiter] = []

    def limiter_for(dataset: str, spec: DatasetConfig) -> RateLimiter:
        """The shared pool the input names, a private limiter from its inline bound, or no limit."""
        bound = spec.rate_limit
        if bound is None:
            return RateLimiter.unlimited()
        if isinstance(bound, str):
            return pools[bound]
        limiter = RateLimiter(bound.to_limit(), name=dataset)
        private.append(limiter)
        return limiter

    started = time.perf_counter()
    portfolios_request = LoadRequest(dataset="portfolios", portfolio_ids=(), as_of=as_of, data_root=data_root, run_id=run_id, rate_limiter=limiter_for("portfolios", config.portfolios))
    portfolios = await _load_frame(resolved.portfolios, portfolios_request)
    try:
        validate_frame(portfolios, PORTFOLIOS)
    except FrameSchemaError as error:
        msg = f"portfolios: {error}"
        raise LoadError(msg) from error
    ordered = portfolios.sort_values("solve_order", kind="stable")
    portfolio_ids = tuple(PortfolioId(str(value)) for value in ordered["portfolio_id"])
    audits = [_audit("portfolios", resolved.portfolios, portfolios, PORTFOLIOS.key, time.perf_counter() - started)]
    log.info("portfolio list loaded: %d portfolio(s); loading %d dataset(s) concurrently", len(portfolio_ids), len(resolved.loaders), extra={"run_id": run_id, "stage": "load"})

    def request(name: str) -> LoadRequest:
        return LoadRequest(dataset=name, portfolio_ids=portfolio_ids, as_of=as_of, data_root=data_root, run_id=run_id, rate_limiter=limiter_for(name, config.datasets[name]))

    async with asyncio.TaskGroup() as group:
        tasks = {name: group.create_task(_load_dataset(name, step, request(name))) for name, step in resolved.loaders.items()}
    outcomes = [tasks[name].result() for name in resolved.loaders]
    for limiter in (*pools.values(), *private):
        log.info("rate limit %r: %d request(s), %.2fs spent waiting", limiter.name, limiter.acquired, limiter.waited_s, extra={"run_id": run_id, "stage": "load"})
    failures = [outcome for outcome in outcomes if isinstance(outcome, _Failed)]
    if failures:
        _raise_load_failures(failures, run_id)
    loaded = [outcome for outcome in outcomes if isinstance(outcome, _Loaded)]
    frames = {outcome.name: outcome.frame for outcome in loaded if outcome.frame is not None}
    constraints: Mapping[str, Mapping[str, object]] = next((outcome.constraints for outcome in loaded if outcome.constraints is not None), {})
    audits.extend(outcome.audit for outcome in loaded)
    return LoadedDatasets(portfolio_ids=portfolio_ids, frames=frames, constraints=constraints, audits=tuple(audits), run_id=run_id)


async def _load_dataset(name: str, step: ResolvedStep, request: LoadRequest) -> _Loaded | _Failed:
    started = time.perf_counter()
    try:
        if name == "constraints":
            constraints = await _load_constraints(step, request)
            elapsed = time.perf_counter() - started
            audit = DatasetAudit(name, step.qualname, step.source_sha256, step.params_sha256, len(constraints), (), json_sha256(constraints), elapsed)
            log.info("dataset %r loaded: %d portfolio(s) in %.2fs", name, len(constraints), elapsed, extra={"run_id": request.run_id, "stage": "load"})
            return _Loaded(name, None, constraints, audit)
        frame = await _load_frame(step, request)
        elapsed = time.perf_counter() - started
        schema = DATASET_SCHEMAS.get(name)
        log.info("dataset %r loaded: %d row(s) in %.2fs", name, len(frame), elapsed, extra={"run_id": request.run_id, "stage": "load"})
        return _Loaded(name, frame, None, _audit(name, step, frame, schema.key if schema is not None else (), elapsed))
    except Exception as error:  # noqa: BLE001  # every dataset's outcome is collected so all failures are reported together
        log.error("dataset %r failed after %.2fs: %s: %s", name, time.perf_counter() - started, type(error).__name__, error, extra={"run_id": request.run_id, "stage": "load"})
        return _Failed(name, error)


def _raise_load_failures(failures: list[_Failed], run_id: str) -> None:
    """Raise one error for every failed dataset; an infrastructure error keeps its own type."""
    del run_id
    hard = [failure for failure in failures if not isinstance(failure.error, ValueError | KeyError)]
    if hard:
        raise hard[0].error
    raise LoadError("; ".join(f"{failure.name}: {failure.error}" for failure in failures))


async def _load_frame(step: ResolvedStep, request: LoadRequest) -> pd.DataFrame:
    result = await step.invoke_async(request=request)
    if not isinstance(result, pd.DataFrame):
        msg = f"loader {step.qualname!r} for {request.dataset!r} returned {type(result).__name__}, expected DataFrame"
        raise LoadError(msg)
    return result


async def _load_constraints(step: ResolvedStep, request: LoadRequest) -> Mapping[str, Mapping[str, object]]:
    result = await step.invoke_async(request=request)
    msg = f"loader {step.qualname!r} for 'constraints' must return a mapping of portfolio id to constraints mapping"
    if not isinstance(result, Mapping):
        raise LoadError(msg)
    constraints: dict[str, dict[str, object]] = {}
    for portfolio_id, mapping in result.items():
        if not isinstance(mapping, Mapping):
            raise LoadError(msg)
        constraints[str(portfolio_id)] = {str(key): value for key, value in mapping.items()}
    return constraints


def _audit(name: str, step: ResolvedStep, frame: pd.DataFrame, key: tuple[str, ...], load_time_s: float) -> DatasetAudit:
    return DatasetAudit(name, step.qualname, step.source_sha256, step.params_sha256, len(frame), tuple(str(column) for column in frame.columns), frame_sha256(frame, key), load_time_s)


def assemble(loaded: LoadedDatasets, resolved: ResolvedConfig) -> AssembledDatasets:
    """Run the assembly steps in order, then validate every engine-known frame against its schema.

    A step that raises ``ValueError`` or ``KeyError`` (a missing dataset, a violated cardinality, a
    column conflict) becomes an :class:`AssemblyError` naming the step; any other exception keeps its
    type. After the last step the four required frames must exist, each engine-known frame must
    satisfy its schema, and ``details`` and ``constraints`` must cover every portfolio.
    """
    frames = Frames(loaded.frames)
    audits: list[AssemblyAuditRecord] = []
    for index, step in enumerate(resolved.assembly):
        where = f"assembly[{index}] {step.qualname}"
        before = frames
        try:
            result = step.invoke(frames=before)
        except (ValueError, KeyError) as error:
            msg = f"{where}: {_message(error)}"
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
        log.info("assembly step %r applied: %s", step.qualname, ", ".join(f"{name}={rows}" for name, rows in frames.row_counts().items()), extra={"run_id": loaded.run_id, "stage": "assembly"})
    missing_frames = [name for name in REQUIRED_FRAMES if name not in frames]
    if missing_frames:
        msg = f"after assembly, required datasets are missing {missing_frames}; declare a loader for each or produce it in an assembly step"
        raise LoadError(msg)
    failures: list[str] = []
    for name, schema in DATASET_SCHEMAS.items():
        try:
            validate_frame(frames[name], schema)
        except FrameSchemaError as error:
            failures.extend(f"{name}: {failure}" for failure in error.failures)
    if failures:
        raise LoadError("; ".join(failures))
    details = frames["details"]
    missing_details = sorted(set(loaded.portfolio_ids) - {str(value) for value in details["portfolio_id"]})
    if missing_details:
        msg = f"details missing for portfolios {missing_details}"
        raise LoadError(msg)
    missing_constraints = sorted(set(loaded.portfolio_ids) - set(loaded.constraints))
    if missing_constraints:
        msg = f"constraints missing for portfolios {missing_constraints}"
        raise LoadError(msg)
    return AssembledDatasets(
        portfolio_ids=loaded.portfolio_ids,
        holdings=frames["holdings"],
        universe=frames["universe"],
        details=details,
        targets=frames["targets"],
        extras={name: frame for name, frame in frames.items() if name not in DATASET_SCHEMAS},
        constraints=loaded.constraints,
        as_of=resolved.config.run.as_of,
        audits=tuple(audits),
    )


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


PREVALIDATED_AT_SLICE: frozenset[str] = frozenset({"holdings", "universe", "targets"})
"""Frames the bundle need not re-validate when sliced from assembled datasets.

:func:`assemble` validated all three against their schemas. The universe is passed whole; the
holdings and targets slices are row subsets, and a row subset keeps every per-column check, the
key's uniqueness, and — because targets are sliced by whole benchmark — the sum-to-one invariant.
"""


def slice_portfolio(assembled: AssembledDatasets, portfolio_id: PortfolioId) -> PortfolioData:
    """Build the validated per-portfolio bundle: its own holdings, constraints, and extras rows; its benchmark's targets; the whole universe."""
    details = details_from_frame(assembled.details, portfolio_id)
    holdings = assembled.holdings[assembled.holdings["portfolio_id"] == portfolio_id].reset_index(drop=True)
    targets = assembled.targets[assembled.targets["benchmark_id"] == details.benchmark_id].reset_index(drop=True)
    extras = {name: _rows_for(frame, portfolio_id) for name, frame in assembled.extras.items()}
    return PortfolioData(
        details=details,
        holdings=holdings,
        universe=assembled.universe.reset_index(drop=True),
        targets=targets,
        style=style_constraints_from_mapping(assembled.constraints[portfolio_id]),
        as_of=assembled.as_of,
        extras=extras,
        prevalidated=PREVALIDATED_AT_SLICE,
    )


def _rows_for(frame: pd.DataFrame, portfolio_id: PortfolioId) -> pd.DataFrame:
    """A per-portfolio dataset (one with a ``portfolio_id`` column) reduced to this portfolio; a global one passed whole."""
    if "portfolio_id" in frame.columns:
        return frame[frame["portfolio_id"] == portfolio_id].reset_index(drop=True)
    return frame.reset_index(drop=True)
