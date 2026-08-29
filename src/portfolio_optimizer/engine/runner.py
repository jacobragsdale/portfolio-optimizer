"""Schedule portfolios through build and solve according to the execution mode, then publish and record.

The three modes differ only in *where* build and solve happen; every portfolio goes through the same
``slice_and_build`` → ``finish_or_fail`` functions (``engine/tasks.py``), results are consumed in
configured solve order regardless of completion order, and the manifest is written whatever happens.
The backend (``engine/backends.py``) is started right after config resolution so a cluster warms up
under the load stage, scaled and waited on only after assembly, and closed in a ``finally``.
"""

import logging
from collections import deque
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.cvx.adapter import solver_version
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import Artifact, AssemblyAuditRecord, PortfolioFailure, PortfolioResult, SolveContext
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import Backend, BackendFactory, ClusterError, Pending, SharedRunData, Task, TaskOutput, WorkersReady
from portfolio_optimizer.engine.dask_backend import DaskBackend
from portfolio_optimizer.engine.environment import GitInfo, WorkerEnvironment, environment_for, host_name, package_versions
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
    solved_record,
    step_records,
    versions,
    write_manifest,
)
from portfolio_optimizer.engine.tasks import BuildResult, Outcome, build_task, failure, finish_or_fail, full_task, slice_and_build, step_refs
from portfolio_optimizer.settings import ExecutionSettings

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_PORTFOLIO_FAILED = 1
EXIT_INPUT_REJECTED = 2
EXIT_INFRASTRUCTURE = 3


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

    def start(self, *, needs_backend: bool) -> None:
        """Ask for the backend now; a cluster then warms up while data loads."""
        if not needs_backend:
            return
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

    def saw(self, environment: WorkerEnvironment, host: str) -> None:
        """Record that ``host``, running ``environment``, produced one outcome."""
        hosts = self.sightings.setdefault(environment, {})
        hosts[host] = hosts.get(host, 0) + 1

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
        """Every distinct environment that did work, with its hosts and portfolio count."""
        return tuple(WorkerRecord(environment=environment, hosts=tuple(sorted(hosts)), portfolios=sum(hosts.values())) for environment, hosts in self.sightings.items())


# --- the run ---


def run(
    resolved: ResolvedConfig, io: IoContext, *, execution: ExecutionSettings, git: GitInfo, config_path: str, settings: Mapping[str, str], backend_factory: BackendFactory = DaskBackend
) -> RunReport:
    """Execute the run end to end and write its manifest. Raises only when nothing could start."""
    config = resolved.config
    log.info("run starting", extra={"run_id": io.run_id, "mode": config.execution.mode, "stage": "load"})
    session = _Session(execution, io, backend_factory)
    ids: tuple[PortfolioId, ...] = ()
    dataset_audits: tuple[DatasetAudit, ...] = ()
    assembly_audits: tuple[AssemblyAuditRecord, ...] = ()
    outcomes: dict[PortfolioId, Outcome] = {}
    cluster_error: PortfolioFailure | None = None
    try:
        session.start(needs_backend=config.execution.mode != "sequential")
        try:
            loaded = load_datasets(resolved, data_root=io.data_root, run_id=io.run_id)
            assembled = assemble(loaded, resolved)
        except (ValueError, KeyError) as error:
            msg = f"inputs rejected: {error}"
            raise InputRejectedError(msg) from error
        dataset_audits, assembly_audits, ids = loaded.audits, assembled.audits, assembled.portfolio_ids
        outcomes = _execute(SharedRunData(assembled=assembled, config=config, config_sha256=resolved.config_sha256, run_id=io.run_id), resolved, session)
    except ClusterError as error:
        log.error("cluster unavailable", extra={"run_id": io.run_id, "stage": "cluster", "error": type(error).__name__})
        cluster_error = failure("*", "cluster", error)
        outcomes = {portfolio_id: PortfolioFailure(portfolio_id, "skipped", "ClusterUnavailable", "not processed because the cluster did not come up") for portfolio_id in ids}
    finally:
        session.close()
    ordered = tuple(outcomes[portfolio_id] for portfolio_id in ids)
    artifacts, publish_error = _persist_and_publish(ordered, resolved, io)
    infrastructure_error = publish_error if publish_error is not None else cluster_error
    exit_code = _exit_code(ordered, infrastructure_error)
    manifest = _manifest(resolved, io, git, config_path, settings, dataset_audits, assembly_audits, ordered, artifacts, exit_code, infrastructure_error, session)
    manifest_path = write_manifest(manifest, io.output_dir / io.run_id)
    log.info("run finished", extra={"run_id": io.run_id, "stage": "manifest", "exit_code": exit_code})
    return RunReport(run_id=io.run_id, outcomes=ordered, manifest=manifest, manifest_path=manifest_path, artifacts=artifacts, exit_code=exit_code)


def _execute(shared: SharedRunData, resolved: ResolvedConfig, session: _Session) -> dict[PortfolioId, Outcome]:
    config = resolved.config
    ids = shared.assembled.portfolio_ids
    fail_fast = config.execution.on_error == "fail_fast"
    expected = environment_for(config, cwd=Path.cwd(), image_digest=session.execution.image_digest)
    if config.execution.mode == "sequential":
        return _run_sequential(ids, shared, resolved, session, expected, fail_fast=fail_fast)
    backend = session.wait()
    handle = backend.share(shared)
    if config.execution.mode == "parallel":
        return dict(_collect(ids, backend, handle, full_task, session, expected, fail_fast=fail_fast))
    built = _collect(ids, backend, handle, build_task, session, expected, fail_fast=fail_fast)
    return _solve_sequentially(ids, built, resolved, shared.run_id, fail_fast=fail_fast)


def _run_sequential(ids: Sequence[PortfolioId], shared: SharedRunData, resolved: ResolvedConfig, session: _Session, expected: WorkerEnvironment, *, fail_fast: bool) -> dict[PortfolioId, Outcome]:
    outcomes: dict[PortfolioId, Outcome] = {}
    ctx = SolveContext()
    failed = False
    host = host_name()
    for portfolio_id in ids:
        if failed and fail_fast:
            outcomes[portfolio_id] = _skipped(portfolio_id, "not processed because an earlier portfolio failed and on_error is fail_fast")
            continue
        built = slice_and_build(shared, resolved, portfolio_id, ctx)
        outcome = built if isinstance(built, PortfolioFailure) else finish_or_fail(built, resolved, ctx, shared.run_id)
        session.saw(expected, host)
        outcomes[portfolio_id] = outcome
        if isinstance(outcome, PortfolioResult):
            ctx = ctx.with_result(outcome)
        else:
            failed = True
    return outcomes


def _solve_sequentially(
    ids: Sequence[PortfolioId], built: Generator[tuple[PortfolioId, BuildResult | PortfolioFailure]], resolved: ResolvedConfig, run_id: str, *, fail_fast: bool
) -> dict[PortfolioId, Outcome]:
    """Solve each build as it arrives, in solve order, with a live chain; builds keep arriving from the workers meanwhile."""
    outcomes: dict[PortfolioId, Outcome] = {}
    ctx = SolveContext()
    for portfolio_id, build_outcome in built:
        if isinstance(build_outcome, PortfolioFailure):
            outcomes[portfolio_id] = build_outcome
            continue
        outcome = finish_or_fail(build_outcome, resolved, ctx, run_id)
        outcomes[portfolio_id] = outcome
        if isinstance(outcome, PortfolioResult):
            ctx = ctx.with_result(outcome)
        elif fail_fast:
            built.close()
            break
    for portfolio_id in ids:
        outcomes.setdefault(portfolio_id, _skipped(portfolio_id, "not solved because an earlier portfolio failed and on_error is fail_fast"))
    return outcomes


def _collect[T](
    ids: Sequence[PortfolioId], backend: Backend, handle: object, task: Task[T], session: _Session, expected: WorkerEnvironment, *, fail_fast: bool
) -> Generator[tuple[PortfolioId, T | PortfolioFailure]]:
    """Keep a window of tasks outstanding and yield outcomes in configured order, so completion order never matters.

    Under ``fail_fast`` the first failure stops further submission, cancels what is queued, and marks
    the rest skipped; closing the iterator early cancels whatever is still outstanding.
    """
    queue = deque(ids)
    pending: dict[PortfolioId, Pending[TaskOutput[T]]] = {}

    def top_up() -> None:
        while queue and len(pending) < session.execution.window:
            next_id = queue.popleft()
            pending[next_id] = backend.submit(task, handle, next_id)

    top_up()
    failed = False
    try:
        for portfolio_id in ids:
            if failed and fail_fast:
                queued = pending.pop(portfolio_id, None)
                if queued is not None:
                    queued.cancel()
                yield portfolio_id, _skipped(portfolio_id, "not processed because an earlier portfolio failed and on_error is fail_fast")
                continue
            future = pending.pop(portfolio_id)
            try:
                output = future.result()
            except Exception as error:  # noqa: BLE001  # a worker that died (e.g. unpicklable result) is a per-portfolio failure
                outcome: T | PortfolioFailure = failure(portfolio_id, "worker", error)
            else:
                outcome = _accept(output, session, expected)
            if isinstance(outcome, PortfolioFailure):
                failed = True
            if not (failed and fail_fast):
                top_up()
            yield portfolio_id, outcome
    finally:
        for future in pending.values():
            future.cancel()


def _accept[T](output: TaskOutput[T], session: _Session, expected: WorkerEnvironment) -> T | PortfolioFailure:
    """Record who did the work and refuse a result from an environment that differs from this run's."""
    session.saw(output.environment, output.host)
    differences = expected.differences(output.environment)
    if differences:
        portfolio_id = output.outcome.portfolio_id if isinstance(output.outcome, PortfolioFailure | PortfolioResult | BuildResult) else "?"
        return PortfolioFailure(portfolio_id, "worker", "EnvironmentMismatch", f"worker {output.host} runs a different environment: {'; '.join(differences)}")
    return output.outcome


def _skipped(portfolio_id: PortfolioId, message: str) -> PortfolioFailure:
    return PortfolioFailure(portfolio_id, "skipped", "SkippedAfterFailure", message)


# --- persistence, publication, manifest ---


def _persist_and_publish(outcomes: Sequence[Outcome], resolved: ResolvedConfig, io: IoContext) -> tuple[tuple[Artifact, ...], PortfolioFailure | None]:
    artifacts: list[Artifact] = []
    solved = [outcome for outcome in outcomes if isinstance(outcome, PortfolioResult)]
    run_dir = io.output_dir / io.run_id
    try:
        for result in solved:
            artifacts.extend(_persist_result(result, run_dir))
        if not solved:
            log.error("no portfolio solved; nothing published", extra={"run_id": io.run_id, "stage": "sink"})
            return tuple(artifacts), None
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
        return tuple(artifacts), failure("*", "sink", error)
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
    artifacts: Sequence[Artifact],
    exit_code: int,
    infrastructure_error: PortfolioFailure | None,
    session: _Session,
) -> RunManifest:
    config = resolved.config
    post = config.post_solve
    records = [solved_record(o, o.report, o.drift, post.violation_tol, post.violation_tol) if isinstance(o, PortfolioResult) else failed_record(o) for o in outcomes]
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
        execution_mode=config.execution.mode,
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
