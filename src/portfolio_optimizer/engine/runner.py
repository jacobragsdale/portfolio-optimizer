"""Schedule portfolios through build and solve on the run's cluster, then publish and record.

Every portfolio builds at once, chain-free. The builds' summaries give the main process each
portfolio's solve-order key and tradable securities; from those it derives the schedule
(``engine/schedule.py``) — who solves after whom — and submits every solve with its predecessors'
contributions as dependencies, so the cluster enforces the order and each solve folds only the buys
that could affect it. Outcomes are classified in solve order whatever finished first, so the worker
count and completion order never change a record, and the manifest is written whatever happens. The
backend (``engine/backends.py``) is started right after config resolution so a cluster warms up under
the load stage, scaled and waited on only after assembly, and closed in a ``finally``.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.cvx.adapter import solver_version
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import Artifact, AssemblyAuditRecord, Contribution, PortfolioFailure, PortfolioResult
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import Backend, BackendFactory, ClusterError, Pending, SharedRunData, TaskOutput, WorkerEnvironmentError, WorkersReady
from portfolio_optimizer.engine.dask_backend import DaskBackend
from portfolio_optimizer.engine.environment import GitInfo, WorkerEnvironment, environment_for, package_versions
from portfolio_optimizer.engine.hashing import file_sha256
from portfolio_optimizer.engine.load import DatasetAudit, assemble, load_datasets
from portfolio_optimizer.engine.manifest import (
    ClusterRecord,
    ConfigInfo,
    RunManifest,
    WorkerRecord,
    artifact_records,
    assembly_records,
    created_at,
    dataset_records,
    failed_record,
    finalize,
    schedule_record,
    solved_record,
    step_records,
    versions,
    write_manifest,
)
from portfolio_optimizer.engine.schedule import Coupling, Schedule, dependency_graph, order_portfolios
from portfolio_optimizer.engine.tasks import BuildResult, BuildSummary, Outcome, build_task, contribution, probe_task, skipped, solve_task, step_refs, summarize
from portfolio_optimizer.settings import ExecutionSettings

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_PORTFOLIO_FAILED = 1
EXIT_INPUT_REJECTED = 2
EXIT_INFRASTRUCTURE = 3

SKIPPED_BY_POSITION = "not processed because a higher-priority portfolio failed and on_error is fail_fast"


class InputRejectedError(ValueError):
    """The configured inputs could not be loaded or assembled; nothing was solved."""


@dataclass(frozen=True, slots=True)
class RunReport:
    """What a run produced."""

    run_id: str
    outcomes: tuple[Outcome, ...]
    manifest: RunManifest
    manifest_path: Path
    artifacts: tuple[Artifact, ...]
    exit_code: int

    @property
    def solved(self) -> tuple[PortfolioResult, ...]:
        """Portfolios that produced orders."""
        return tuple(outcome for outcome in self.outcomes if isinstance(outcome, PortfolioResult))

    @property
    def failed(self) -> tuple[PortfolioFailure, ...]:
        """Portfolios that did not."""
        return tuple(outcome for outcome in self.outcomes if isinstance(outcome, PortfolioFailure))


@dataclass(frozen=True, slots=True)
class Executed:
    """What the scheduling stage produced: every outcome, the schedule it followed, each portfolio's key, and the files it persisted."""

    outcomes: Mapping[PortfolioId, Outcome]
    schedule: Schedule
    keys: Mapping[PortfolioId, Decimal]
    artifacts: tuple[Artifact, ...]


@dataclass(slots=True)
class _Session:
    """The cluster's lifetime, and what the manifest records about it: timestamps, size, and every environment that did work."""

    execution: ExecutionSettings
    io: IoContext
    backend_factory: BackendFactory
    backend: Backend | None = None
    provision_started_at: datetime | None = None
    first_worker_ready_at: datetime | None = None
    closed_at: datetime | None = None
    ready: WorkersReady | None = None
    sightings: dict[WorkerEnvironment, dict[str, int]] = field(default_factory=dict)

    def start(self) -> None:
        """Ask for the backend now; the cluster then warms up while data loads."""
        self.backend = self.backend_factory(self.execution, run_id=self.io.run_id)
        self.provision_started_at = self.io.clock.now()
        self.backend.start()
        log.info("backend %s starting", self.backend.kind, extra={"run_id": self.io.run_id, "stage": "cluster"})

    def wait(self) -> Backend:
        """Scale to the full size and block until one worker can take a task."""
        if self.backend is None:
            msg = "no backend was started"
            raise ClusterError(msg)
        self.backend.scale(self.execution.max_workers)
        self.ready = self.backend.ready(1, self.execution.cluster_timeout_s)
        self.first_worker_ready_at = self.io.clock.now()
        log.info("backend %s ready with %d worker(s)", self.backend.kind, self.ready.workers, extra={"run_id": self.io.run_id, "stage": "cluster"})
        return self.backend

    def close(self) -> None:
        """Release the backend; always called."""
        if self.backend is not None:
            self.backend.close()
            self.closed_at = self.io.clock.now()

    def saw(self, environment: WorkerEnvironment, host: str, *, solved: bool) -> None:
        """Record that ``host``, running ``environment``, did work; only solves count toward its portfolio total."""
        hosts = self.sightings.setdefault(environment, {})
        hosts[host] = hosts.get(host, 0) + (1 if solved else 0)

    def cluster_record(self) -> ClusterRecord | None:
        """The manifest's view of the backend, or ``None`` when the run had none."""
        if self.backend is None:
            return None
        return ClusterRecord(
            kind=self.backend.kind,
            min_workers=self.execution.min_workers,
            max_workers=self.execution.max_workers,
            workers_ready=self.ready.workers if self.ready is not None else None,
            scheduler_address=self.ready.scheduler_address if self.ready is not None else None,
            provision_started_at=self.provision_started_at,
            first_worker_ready_at=self.first_worker_ready_at,
            closed_at=self.closed_at,
        )

    def worker_records(self) -> tuple[WorkerRecord, ...]:
        """Every distinct environment that did work, with its hosts and the portfolios it solved."""
        return tuple(WorkerRecord(environment=environment, hosts=tuple(sorted(hosts)), portfolios=sum(hosts.values())) for environment, hosts in self.sightings.items())


# --- the run ---


def run(
    resolved: ResolvedConfig, io: IoContext, *, execution: ExecutionSettings, git: GitInfo, config_path: str, settings: Mapping[str, str], backend_factory: BackendFactory = DaskBackend
) -> RunReport:
    """Execute the run end to end and write its manifest. Raises only when nothing could start."""
    config = resolved.config
    log.info("run starting", extra={"run_id": io.run_id, "stage": "load"})
    session = _Session(execution, io, backend_factory)
    order: tuple[PortfolioId, ...] = ()
    dataset_audits: tuple[DatasetAudit, ...] = ()
    assembly_audits: tuple[AssemblyAuditRecord, ...] = ()
    executed: Executed | None = None
    outcomes: Mapping[PortfolioId, Outcome] = {}
    cluster_error: PortfolioFailure | None = None
    try:
        session.start()
        try:
            loaded = load_datasets(resolved, data_root=io.data_root, run_id=io.run_id)
            assembled = assemble(loaded, resolved)
        except (ValueError, KeyError) as error:
            msg = f"inputs rejected: {error}"
            raise InputRejectedError(msg) from error
        dataset_audits, assembly_audits, order = loaded.audits, assembled.audits, assembled.portfolio_ids
        executed = _execute(SharedRunData(assembled=assembled, config=config, config_sha256=resolved.config_sha256, run_id=io.run_id), resolved, session, run_dir=io.output_dir / io.run_id)
        outcomes, order = executed.outcomes, executed.schedule.order
    except ClusterError as error:
        log.error("cluster unavailable", extra={"run_id": io.run_id, "stage": "cluster", "error": type(error).__name__})
        cluster_error = PortfolioFailure("*", "cluster", type(error).__name__, str(error))
        reason = "a worker failed its environment check" if isinstance(error, WorkerEnvironmentError) else "the cluster did not come up"
        outcomes = {portfolio_id: PortfolioFailure(portfolio_id, "skipped", "ClusterUnavailable", f"not processed because {reason}") for portfolio_id in order}
    finally:
        session.close()
    ordered = tuple(outcomes[portfolio_id] for portfolio_id in order)
    persisted = executed.artifacts if executed is not None else ()
    published, publish_error = _publish(ordered, resolved, io)
    artifacts = (*persisted, *published)
    infrastructure_error = publish_error if publish_error is not None else cluster_error
    exit_code = _exit_code(ordered, infrastructure_error)
    manifest = _manifest(resolved, io, git, config_path, settings, dataset_audits, assembly_audits, ordered, executed, artifacts, exit_code, infrastructure_error, session)
    manifest_path = write_manifest(manifest, io.output_dir / io.run_id)
    log.info("run finished", extra={"run_id": io.run_id, "stage": "manifest", "exit_code": exit_code})
    return RunReport(run_id=io.run_id, outcomes=ordered, manifest=manifest, manifest_path=manifest_path, artifacts=artifacts, exit_code=exit_code)


@dataclass(frozen=True, slots=True)
class _Dispatch:
    """What every submission needs: the backend, the shared-data handle, and how to gate what comes back."""

    backend: Backend
    handle: object
    run_id: str
    total: int
    session: _Session
    expected: WorkerEnvironment

    def key(self, portfolio_id: PortfolioId, name: str) -> str:
        return f"{self.run_id}/{portfolio_id}/{name}"


def _execute(shared: SharedRunData, resolved: ResolvedConfig, session: _Session, *, run_dir: Path) -> Executed:
    """Build everything, derive the schedule, solve along it, and classify every outcome in solve order."""
    config = resolved.config
    fail_fast = config.execution.on_error == "fail_fast"
    backend = session.wait()
    expected = environment_for(config, cwd=Path.cwd(), image_digest=session.execution.image_digest)
    _check_workers(backend, shared, session, expected)
    dispatch = _Dispatch(backend, backend.share(shared), shared.run_id, len(shared.assembled.portfolio_ids), session, expected)
    builds, failed, keys, tradable = _build_all(dispatch, shared)
    order = order_portfolios(keys)
    coupling: Coupling = "none" if not resolved.chain_aware_steps else config.execution.dependencies
    schedule = dependency_graph(order, tradable, frozenset(failed), coupling)
    shape = schedule.summary()
    log.info(
        "schedule derived: %d portfolio(s), %d edge(s), %d component(s), critical path %d",
        shape.portfolios,
        shape.edges,
        shape.components,
        shape.critical_path,
        extra={"run_id": shared.run_id, "stage": "schedule", "coupling": coupling, "largest_component": shape.largest_component},
    )
    outcomes: dict[PortfolioId, Outcome] = dict(failed)
    to_solve = list(order)
    if fail_fast and failed:
        first = min(order.index(portfolio_id) for portfolio_id in failed)
        to_solve = list(order[: first + 1])
        for portfolio_id in order[first + 1 :]:
            outcomes[portfolio_id] = skipped(portfolio_id, SKIPPED_BY_POSITION)
    solves = _submit_solves(dispatch, schedule, to_solve, builds, outcomes)
    artifacts = _gather_solves(dispatch, schedule, solves, outcomes, fail_fast=fail_fast, run_dir=run_dir)
    return Executed(outcomes=outcomes, schedule=schedule, keys=keys, artifacts=artifacts)


def _check_workers(backend: Backend, shared: SharedRunData, session: _Session, expected: WorkerEnvironment) -> None:
    """Every worker the run starts with must resolve the config and match the run's fingerprint before any data is shared.

    A worker that cannot — the solver or a step package missing from its image, a stale image — would
    fail every portfolio it touched at stage ``worker``; one round trip here catches it before the run
    has done any work. Workers that join later are gated per result by :func:`_accept`.
    """
    problems: list[str] = []
    probes = backend.probe(probe_task, shared.config, shared.config_sha256)
    for address, output in probes.items():
        session.saw(output.environment, output.host, solved=False)
        if isinstance(output.outcome, PortfolioFailure):
            problems.append(f"worker {output.host} ({address}) cannot resolve the config: {output.outcome.message}")
            continue
        differences = expected.differences(output.environment)
        if differences:
            problems.append(f"worker {output.host} ({address}) runs a different environment: {'; '.join(differences)}")
    if problems:
        raise WorkerEnvironmentError("; ".join(problems))
    log.info("%d worker(s) checked: config resolves and fingerprints match", len(probes), extra={"run_id": shared.run_id, "stage": "cluster"})


def _build_all(
    dispatch: _Dispatch, shared: SharedRunData
) -> tuple[dict[PortfolioId, Pending[TaskOutput[BuildResult]]], dict[PortfolioId, PortfolioFailure], dict[PortfolioId, Decimal], dict[PortfolioId, tuple[str, ...]]]:
    """Submit every build at once and gather the summaries: each portfolio's key and tradable securities, or its failure."""
    ids = shared.assembled.portfolio_ids
    builds: dict[PortfolioId, Pending[TaskOutput[BuildResult]]] = {}
    summaries: dict[PortfolioId, Pending[TaskOutput[BuildSummary]]] = {}
    for rank, portfolio_id in enumerate(ids):
        builds[portfolio_id] = dispatch.backend.submit(build_task, dispatch.handle, portfolio_id, key=dispatch.key(portfolio_id, "build"), priority=dispatch.total - rank)
        summaries[portfolio_id] = dispatch.backend.submit(summarize, builds[portfolio_id], key=dispatch.key(portfolio_id, "summary"), priority=dispatch.total - rank + 1)
    failed: dict[PortfolioId, PortfolioFailure] = {}
    keys: dict[PortfolioId, Decimal] = {}
    tradable: dict[PortfolioId, tuple[str, ...]] = {}
    for portfolio_id in ids:
        keys[portfolio_id] = Decimal(shared.assembled.solve_orders[portfolio_id])
        summary = _accept(_result_or_error(summaries[portfolio_id]), portfolio_id, dispatch.session, dispatch.expected, solved=False)
        if isinstance(summary, PortfolioFailure):
            failed[portfolio_id] = summary
        else:
            keys[portfolio_id] = summary.solve_order
            tradable[portfolio_id] = summary.tradable
    return builds, failed, keys, tradable


def _submit_solves(
    dispatch: _Dispatch, schedule: Schedule, to_solve: Sequence[PortfolioId], builds: dict[PortfolioId, Pending[TaskOutput[BuildResult]]], outcomes: dict[PortfolioId, Outcome]
) -> dict[PortfolioId, Pending[TaskOutput[PortfolioResult]]]:
    """Submit each solve with its predecessors' contributions as dependencies; a portfolio behind a failed build is skipped here, never submitted."""
    heights = schedule.heights()
    solves: dict[PortfolioId, Pending[TaskOutput[PortfolioResult]]] = {}
    contributions: dict[PortfolioId, Pending[Contribution | PortfolioFailure]] = {}
    for portfolio_id in to_solve:
        if portfolio_id in outcomes:
            continue
        blocked = next((earlier for earlier in schedule.predecessors[portfolio_id] if earlier in outcomes), None)
        if blocked is not None:
            outcomes[portfolio_id] = _skipped_after(portfolio_id, blocked, outcomes[blocked])
            continue
        priority = dispatch.total + heights[portfolio_id]
        dependencies = [contributions[earlier] for earlier in schedule.predecessors[portfolio_id]]
        solves[portfolio_id] = dispatch.backend.submit(solve_task, dispatch.handle, builds.pop(portfolio_id), *dependencies, key=dispatch.key(portfolio_id, "solve"), priority=priority)
        contributions[portfolio_id] = dispatch.backend.submit(contribution, solves[portfolio_id], key=dispatch.key(portfolio_id, "contribution"), priority=priority + 1)
    builds.clear()  # a build nothing waits for is released by the scheduler before it runs
    return solves


def _gather_solves(
    dispatch: _Dispatch, schedule: Schedule, solves: Mapping[PortfolioId, Pending[TaskOutput[PortfolioResult]]], outcomes: dict[PortfolioId, Outcome], *, fail_fast: bool, run_dir: Path
) -> tuple[Artifact, ...]:
    """Collect solves as they complete, classify them in solve order, and persist each result as it is classified.

    Under ``fail_fast`` the first failure *in solve order* cancels every solve behind it; those are
    recorded as skipped by position, whatever they had finished, so the manifest never depends on timing.
    """
    artifacts: list[Artifact] = []
    raw: dict[PortfolioId, TaskOutput[PortfolioResult] | Exception] = {}
    waiting = [portfolio_id for portfolio_id in schedule.order if portfolio_id in solves]
    cursor = 0
    stopped = False
    for portfolio_id in dispatch.backend.as_completed(solves):
        raw[portfolio_id] = _result_or_error(solves[portfolio_id])
        while not stopped and cursor < len(waiting) and waiting[cursor] in raw:
            current = waiting[cursor]
            outcome = _classify(current, raw[current], schedule, outcomes, dispatch.session, dispatch.expected)
            outcomes[current] = outcome
            cursor += 1
            if isinstance(outcome, PortfolioResult):
                artifacts.extend(_persist_result(outcome, run_dir))
            elif fail_fast:
                stopped = True
        if stopped:
            break
    if stopped:
        rest = waiting[cursor:]
        dispatch.backend.cancel([solves[portfolio_id] for portfolio_id in rest])
        for portfolio_id in rest:
            outcomes[portfolio_id] = skipped(portfolio_id, SKIPPED_BY_POSITION)
    return tuple(artifacts)


def _result_or_error[T](pending: Pending[T]) -> T | Exception:
    """A task's output, or the exception Dask raised for it — its own crash or a dependency's."""
    try:
        return pending.result()
    except Exception as error:  # noqa: BLE001  # a worker that died (e.g. unpicklable result) is a per-portfolio failure
        return error


def _classify(
    portfolio_id: PortfolioId, raw: TaskOutput[PortfolioResult] | Exception, schedule: Schedule, outcomes: Mapping[PortfolioId, Outcome], session: _Session, expected: WorkerEnvironment
) -> Outcome:
    """A raised solve is a failed predecessor's doing when one exists, else a worker failure; a returned output is gated on its environment."""
    if isinstance(raw, Exception):
        blamed = next((earlier for earlier in schedule.predecessors[portfolio_id] if isinstance(outcomes.get(earlier), PortfolioFailure)), None)
        if blamed is not None:
            return _skipped_after(portfolio_id, blamed, outcomes[blamed])
        return PortfolioFailure(portfolio_id, "worker", type(raw).__name__, _stable_message(raw))
    return _accept(raw, portfolio_id, session, expected, solved=True)


def _accept[T](raw: TaskOutput[T] | Exception, portfolio_id: PortfolioId, session: _Session, expected: WorkerEnvironment, *, solved: bool) -> T | PortfolioFailure:
    """Record who did the work and refuse a result from an environment that differs from this run's."""
    if isinstance(raw, Exception):
        return PortfolioFailure(portfolio_id, "worker", type(raw).__name__, _stable_message(raw))
    session.saw(raw.environment, raw.host, solved=solved)
    differences = expected.differences(raw.environment)
    if differences:
        return PortfolioFailure(portfolio_id, "worker", "EnvironmentMismatch", f"worker {raw.host} runs a different environment: {'; '.join(differences)}")
    return raw.outcome


def _stable_message(error: Exception) -> str:
    """A worker death names the task it blames, never the worker address, so the manifest does not depend on placement."""
    task = getattr(error, "task", None)
    if isinstance(task, str):
        return f"task {task!r} killed its worker"
    return str(error)


def _skipped_after(portfolio_id: PortfolioId, blamed: PortfolioId, cause: Outcome) -> PortfolioFailure:
    stage = cause.stage if isinstance(cause, PortfolioFailure) else "solve"
    return skipped(portfolio_id, f"not solved because predecessor {blamed!r} failed at stage {stage!r}")


# --- persistence, publication, manifest ---


def _publish(outcomes: Sequence[Outcome], resolved: ResolvedConfig, io: IoContext) -> tuple[tuple[Artifact, ...], PortfolioFailure | None]:
    """Call the sink once with every solved portfolio's orders; a sink failure is infrastructure, and the manifest is still written."""
    solved = [outcome for outcome in outcomes if isinstance(outcome, PortfolioResult)]
    if not solved:
        log.error("no portfolio solved; nothing published", extra={"run_id": io.run_id, "stage": "sink"})
        return (), None
    artifacts: list[Artifact] = []
    try:
        orders = pd.concat([result.orders for result in solved], ignore_index=True).sort_values(["portfolio_id", "security_id"], kind="stable").reset_index(drop=True)
        published = resolved.sink.invoke(orders=orders, io=io)
        if not isinstance(published, tuple):
            msg = f"sink {resolved.sink.qualname!r} returned {type(published).__name__}, expected a tuple of Artifact"
            raise TypeError(msg)  # the sink contract violation is reported like any other sink failure
        for item in published:
            if not isinstance(item, Artifact):
                msg = f"sink {resolved.sink.qualname!r} returned a {type(item).__name__}, expected Artifact"
                raise TypeError(msg)  # see above
            artifacts.append(item)
    except Exception as error:  # noqa: BLE001  # the manifest must still be written; the exit code carries the failure
        log.error("publishing failed", extra={"run_id": io.run_id, "stage": "sink", "error": type(error).__name__})
        return tuple(artifacts), PortfolioFailure("*", "sink", type(error).__name__, str(error))
    return tuple(artifacts), None


def _persist_result(result: PortfolioResult, run_dir: Path) -> list[Artifact]:
    written: list[Artifact] = []
    for subdir, writer in (("problem_specs", result.spec.to_npz), ("solutions", result.solution.to_npz), ("chain", result.chain_state.to_npz)):
        directory = run_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{result.portfolio_id}.npz"
        writer(path)
        written.append(Artifact(path=str(path), sha256=file_sha256(path), size_bytes=path.stat().st_size))
    return written


def _exit_code(outcomes: Sequence[Outcome], infrastructure_error: PortfolioFailure | None) -> int:
    if infrastructure_error is not None:
        return EXIT_INFRASTRUCTURE
    if not outcomes or any(isinstance(outcome, PortfolioFailure) for outcome in outcomes):
        return EXIT_PORTFOLIO_FAILED
    return EXIT_OK


def _manifest(
    resolved: ResolvedConfig,
    io: IoContext,
    git: GitInfo,
    config_path: str,
    settings: Mapping[str, str],
    audits: Sequence[DatasetAudit],
    assembly_audits: Sequence[AssemblyAuditRecord],
    outcomes: Sequence[Outcome],
    executed: Executed | None,
    artifacts: Sequence[Artifact],
    exit_code: int,
    infrastructure_error: PortfolioFailure | None,
    session: _Session,
) -> RunManifest:
    config = resolved.config
    post = config.post_solve
    keys: Mapping[PortfolioId, Decimal] = executed.keys if executed is not None else {}
    predecessors: Mapping[PortfolioId, tuple[PortfolioId, ...]] = executed.schedule.predecessors if executed is not None else {}
    records = [
        solved_record(o, o.report, o.drift, post.violation_tol, post.violation_tol, solve_order=_key_text(keys, o.portfolio_id))
        if isinstance(o, PortfolioResult)
        else failed_record(o, solve_order=_key_text(keys, o.portfolio_id), predecessors=len(predecessors[PortfolioId(o.portfolio_id)]) if PortfolioId(o.portfolio_id) in predecessors else None)
        for o in outcomes
    ]
    if infrastructure_error is not None:
        records.append(failed_record(infrastructure_error))
    solved = [o for o in outcomes if isinstance(o, PortfolioResult)]
    solver_ver = solved[0].solution.solver_version if solved else solver_version(config.solver.name)
    packages = package_versions(step.qualname.partition(":")[0] for step in resolved.all_steps if step.is_external)
    manifest = RunManifest(
        run_id=io.run_id,
        run_name=config.run.name,
        created_at_utc=created_at(io.clock.now()),
        as_of=config.run.as_of,
        git_sha=git.sha,
        git_dirty=git.dirty,
        schedule=schedule_record(executed.schedule.summary()) if executed is not None else None,
        cluster=session.cluster_record(),
        versions=versions(config.solver.name, solver_ver, packages, session.worker_records()),
        config=ConfigInfo(path=config_path, sha256=resolved.config_sha256, resolved=config.model_dump(mode="json")),
        settings=dict(settings),
        terms=step_records(step_refs(resolved.terms)),
        constraints=step_records(step_refs(resolved.constraints)),
        datasets=dataset_records(audits),
        assembly=assembly_records(assembly_audits),
        portfolios=tuple(records),
        artifacts=artifact_records(artifacts),
        exit_code=exit_code,
    )
    return finalize(manifest)


def _key_text(keys: Mapping[PortfolioId, Decimal], portfolio_id: str) -> str | None:
    key = keys.get(PortfolioId(portfolio_id))
    return None if key is None else str(key)
