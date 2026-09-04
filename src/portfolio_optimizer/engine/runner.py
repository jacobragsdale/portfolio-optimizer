"""Schedule portfolios through build and solve on the run's cluster, then publish and record.

Every portfolio builds at once, chain-free. A build's summary tells the main process that portfolio's
solve-order key and tradable securities; from those it derives the schedule (``engine/schedule.py``) —
who solves after whom — and submits every solve with its predecessors' contributions as dependencies,
so the cluster enforces the order and each solve folds only the trades — on the side the run couples
through — that could affect it.

Submission does not wait for the build wave to finish. A portfolio only ever waits for portfolios
earlier in the solve order, so the runner walks that order and submits each solve as the build it has
reached reports (:func:`_stream_solves`): the head of the book is solving while the tail is still
building. When nothing reads the chain there is no order to respect at all, and every solve goes in
behind its own build (:func:`_plan_uncoupled`). The one case that still needs every build first is a
configured solve-order step, because then the order is itself a build output (:func:`_solve_order`).

Outcomes are classified in solve order whatever finished first, so the worker count and completion
order never change a record, and the manifest is written whatever happens. The backend
(``engine/backends.py``) is started right after config resolution so a cluster warms up under the load
stage, scaled and waited on only after assembly, and closed in a ``finally``.
"""

import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.domain.constraints import frame_reads_chain
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.results import RUN_SCOPED, Artifact, AssemblyAuditRecord, Contribution, PortfolioFailure, PortfolioResult
from portfolio_optimizer.domain.schemas import CHECK_RESULTS
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import Backend, BackendFactory, ClusterError, InlineBackend, Pending, SharedRunData, TaskOutput, WorkerEnvironmentError, WorkersReady
from portfolio_optimizer.engine.dask_backend import DaskBackend
from portfolio_optimizer.engine.environment import IMAGE_DIGEST_VARIABLE, GitInfo, WorkerEnvironment, environment_for, package_versions
from portfolio_optimizer.engine.files import write_atomically
from portfolio_optimizer.engine.hashing import file_sha256
from portfolio_optimizer.engine.load import AssembledDatasets, DatasetAudit, assemble, load_datasets
from portfolio_optimizer.engine.manifest import (
    CHECKS_SUBDIR,
    CheckStatus,
    CheckStepRecord,
    ClusterRecord,
    ConfigInfo,
    RunManifest,
    WorkerRecord,
    created_at,
    failed_record,
    finalize,
    solved_record,
    versions,
    write_failure_reports,
    write_manifest,
)
from portfolio_optimizer.engine.schedule import Coupling, OverlapIndex, Schedule, order_portfolios
from portfolio_optimizer.engine.tasks import BuildResult, BuildSummary, Outcome, build_task, contribution, probe_task, skipped, solve_task, summarize
from portfolio_optimizer.engine.timing import Span, SpanRecorder, sort_spans, write_trace
from portfolio_optimizer.settings import ExecutionSettings
from portfolio_optimizer.solving import configured_solver, solver_version

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


@dataclass(frozen=True, slots=True)
class RunContext:
    """The run's surroundings, beyond its config: the instant it is as of, where it reads and writes, its id and clock, the cluster it provisions, the code revision, and the settings the manifest records."""

    io: IoContext
    as_of_date: datetime
    """The timezone-aware instant the run is *as of*: every loader receives it, the build decides holding periods against it, and the manifest records it. A run argument, not config, so one wiring runs every day under one hash."""

    execution: ExecutionSettings
    git: GitInfo
    config_path: str
    settings: Mapping[str, str]


def default_backend(execution: ExecutionSettings, *, run_id: str) -> Backend:
    """The backend the settings ask for: this process under ``inline``, otherwise a Dask cluster the run owns."""
    if execution.cluster_kind == "inline":
        return InlineBackend()
    return DaskBackend(execution, run_id=run_id)


@dataclass(slots=True)
class _Session:
    """The cluster's lifetime, and what the manifest records about it: timestamps, size, and every environment that did work."""

    context: RunContext
    backend_factory: BackendFactory
    backend: Backend | None = None
    provision_started_at: datetime | None = None
    first_worker_ready_at: datetime | None = None
    closed_at: datetime | None = None
    ready: WorkersReady | None = None
    sightings: dict[WorkerEnvironment, dict[str, int]] = field(default_factory=dict)
    provision_started_s: float | None = None
    """Wall-clock twin of ``provision_started_at`` for the timing spans, which must share one axis with the workers' ``time.time`` readings rather than the injected clock's."""

    ready_s: float | None = None
    spans: list[Span] = field(default_factory=list)

    def start(self) -> None:
        """Ask for the backend now; the cluster then warms up while data loads."""
        self.backend = self.backend_factory(self.context.execution, run_id=self.context.io.run_id)
        self.provision_started_at = self.context.io.clock()
        self.provision_started_s = time.time()
        self.backend.start()
        log.info("backend %s starting", self.backend.kind, extra={"run_id": self.context.io.run_id, "stage": "cluster"})

    def wait(self) -> Backend:
        """Scale to the full size and block until one worker can take a task."""
        if self.backend is None:
            msg = "no backend was started"
            raise ClusterError(msg)
        execution = self.context.execution
        self.backend.scale(execution.max_workers)
        self.ready = self.backend.ready(1, execution.cluster_timeout_s)
        self.first_worker_ready_at = self.context.io.clock()
        self.ready_s = time.time()
        log.info("backend %s ready with %d worker(s)", self.backend.kind, self.ready.workers, extra={"run_id": self.context.io.run_id, "stage": "cluster"})
        return self.backend

    def close(self) -> None:
        """Release the backend; always called."""
        if self.backend is not None:
            self.backend.close()
            self.closed_at = self.context.io.clock()

    def saw(self, environment: WorkerEnvironment, host: str, *, solved: bool) -> None:
        """Record that ``host``, running ``environment``, did work; only solves count toward its portfolio total."""
        hosts = self.sightings.setdefault(environment, {})
        hosts[host] = hosts.get(host, 0) + (1 if solved else 0)

    def absorb(self, spans: Sequence[Span]) -> None:
        """Keep the spans a task reported, for the manifest's timing block."""
        self.spans.extend(spans)

    def cluster_record(self) -> ClusterRecord | None:
        """The manifest's view of the backend, or ``None`` when the run had none."""
        if self.backend is None:
            return None
        return ClusterRecord(
            kind=self.backend.kind,
            min_workers=self.context.execution.min_workers,
            max_workers=self.context.execution.max_workers,
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


def run(resolved: ResolvedConfig, context: RunContext, *, backend_factory: BackendFactory = default_backend) -> RunReport:
    """Execute the run end to end and write its manifest. Raises only when nothing could start."""
    config = resolved.config
    io = context.io
    log.info("run starting", extra={"run_id": io.run_id, "stage": "load"})
    session = _Session(context, backend_factory)
    recorder = SpanRecorder()
    order: tuple[PortfolioId, ...] = ()
    dataset_audits: tuple[DatasetAudit, ...] = ()
    assembly_audits: tuple[AssemblyAuditRecord, ...] = ()
    executed: Executed | None = None
    assembled: AssembledDatasets | None = None
    outcomes: Mapping[PortfolioId, Outcome] = {}
    cluster_error: PortfolioFailure | None = None
    try:
        session.start()
        load_started_s = time.time()
        try:
            with recorder.span("load"):
                loaded = load_datasets(resolved, data_root=io.data_root, run_id=io.run_id, as_of_date=context.as_of_date)
            with recorder.span("assembly"):
                assembled = assemble(loaded, resolved, run_id=io.run_id, as_of_date=context.as_of_date)
        except (ValueError, KeyError) as error:
            msg = f"inputs rejected: {error}"
            raise InputRejectedError(msg) from error
        dataset_audits, assembly_audits, order = loaded.audits, assembled.audits, assembled.portfolio_ids
        for audit in dataset_audits:
            recorder.add(f"dataset:{audit.name}", started_at_s=load_started_s + audit.started_s, duration_s=audit.load_time_s)
        shared = SharedRunData(assembled=assembled, config=config, config_sha256=resolved.config_sha256, run_id=io.run_id, packages=context.execution.step_packages)
        executed = _execute(shared, resolved, session, run_dir=io.output_dir / io.run_id)
        outcomes, order = executed.outcomes, executed.schedule.order
    except ClusterError as error:
        log.error("cluster unavailable", extra={"run_id": io.run_id, "stage": "cluster", "error": type(error).__name__})
        cluster_error = PortfolioFailure.from_exception(RUN_SCOPED, "cluster", error)
        reason = "a worker failed its environment check" if isinstance(error, WorkerEnvironmentError) else "the cluster did not come up"
        outcomes = {portfolio_id: PortfolioFailure(portfolio_id, "skipped", "ClusterUnavailable", f"not processed because {reason}") for portfolio_id in order}
    finally:
        session.close()
    ordered = tuple(outcomes[portfolio_id] for portfolio_id in order)
    persisted = executed.artifacts if executed is not None else ()
    orders = _orders(ordered, run_id=io.run_id)
    with recorder.span("sink"):
        published, publish_error = _publish(orders, resolved, io)
    with recorder.span("checks"):
        check_records, check_artifacts, check_error = _run_checks(orders, assembled, ordered, resolved, io.output_dir / io.run_id, run_id=io.run_id)
    if session.provision_started_s is not None and session.ready_s is not None:
        recorder.add("cluster", started_at_s=session.provision_started_s, duration_s=session.ready_s - session.provision_started_s)
    spans = sort_spans((*recorder.spans, *session.spans))
    trace_path = write_trace(spans, io.output_dir / io.run_id)
    infrastructure_error = publish_error if publish_error is not None else cluster_error
    run_failures = tuple(failure for failure in (infrastructure_error, check_error) if failure is not None)
    reports = write_failure_reports([*(outcome for outcome in ordered if isinstance(outcome, PortfolioFailure)), *run_failures], io.output_dir / io.run_id, run_id=io.run_id)
    artifacts = (*persisted, *published, *check_artifacts, *reports, Artifact(path=str(trace_path), sha256=file_sha256(trace_path), size_bytes=trace_path.stat().st_size))
    exit_code = _exit_code(ordered, infrastructure_error, check_records, check_error)
    manifest = _manifest(resolved, session, dataset_audits, assembly_audits, ordered, executed, artifacts, exit_code, run_failures, check_records, spans)
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

    def solve_priority(self, position: int) -> int:
        """Solves are ordered by their place in the solve order, and every solve outranks every build.

        A predecessor always precedes its dependents, so this never puts a blocked solve in front of what
        blocks it, which is what the priority is for. It is a weaker hint than the longest chain starting
        at each portfolio: when a solve is submitted, who will depend on it has not been read yet.
        """
        return 2 * self.total + 1 - position


@dataclass(frozen=True, slots=True)
class _Planned:
    """What submission produced: the schedule it followed, every portfolio's key, the solves in flight, and the portfolios it settled itself."""

    schedule: Schedule
    keys: Mapping[PortfolioId, Decimal]
    solves: Mapping[PortfolioId, Pending[TaskOutput[PortfolioResult]]]
    outcomes: dict[PortfolioId, Outcome]


@dataclass(slots=True)
class _Builds:
    """Every portfolio's build and the summary that describes it; a summary is read once and remembered.

    A build's result never leaves the worker that produced it: :meth:`report` blocks on one portfolio's
    build and returns only what the main process needs to place it in the schedule. A portfolio the
    load stage rejected has no build at all: its failure is already in ``reports``, so it places in the
    schedule like any other portfolio whose tradable set is unknown, and no solve is submitted for it.
    """

    dispatch: _Dispatch
    fallback: Mapping[PortfolioId, Decimal]
    pending: dict[PortfolioId, Pending[TaskOutput[BuildResult]]]
    summaries: dict[PortfolioId, Pending[TaskOutput[BuildSummary]]]
    reports: dict[PortfolioId, BuildSummary | PortfolioFailure] = field(default_factory=dict)
    """Seeded with the load stage's rejections, which have no build to report and are never submitted."""

    def submitted(self, portfolio_id: PortfolioId) -> bool:
        """Whether this portfolio was built at all; one the load stage rejected was not."""
        return portfolio_id in self.summaries

    def report(self, portfolio_id: PortfolioId) -> BuildSummary | PortfolioFailure:
        """What this portfolio's build reported, blocking on that build alone; a portfolio the load stage rejected reports that instead."""
        if portfolio_id not in self.reports:
            read: BuildSummary | PortfolioFailure = _accept(_result_or_error(self.summaries[portfolio_id]), portfolio_id, self.dispatch.session, self.dispatch.expected, solved=False)
            self.reports[portfolio_id] = read
        return self.reports[portfolio_id]

    def key(self, portfolio_id: PortfolioId) -> Decimal:
        """The portfolio's solve-order key: the one its build computed, or the portfolios frame's when no build reported."""
        report = self.reports.get(portfolio_id)
        return report.solve_order if isinstance(report, BuildSummary) else self.fallback[portfolio_id]

    def take(self, portfolio_id: PortfolioId) -> Pending[TaskOutput[BuildResult]]:
        """Hand the build over to the solve that consumes it."""
        return self.pending.pop(portfolio_id)

    def release(self) -> None:
        """Let go of every build nothing waits for; the scheduler frees it before it runs."""
        self.pending.clear()


def _execute(shared: SharedRunData, resolved: ResolvedConfig, session: _Session, *, run_dir: Path) -> Executed:
    """Build everything, submit each solve as soon as the schedule allows it, and classify every outcome in solve order."""
    config = resolved.config
    fail_fast = config.execution.on_error == "fail_fast"
    backend = session.wait()
    expected = environment_for(config, cwd=Path.cwd(), image_digest=os.environ.get(IMAGE_DIGEST_VARIABLE))
    _check_workers(backend, shared, session, expected)
    dispatch = _Dispatch(backend, backend.share(shared), shared.run_id, len(shared.assembled.portfolio_ids), session, expected)
    builds = _submit_builds(dispatch, shared)
    if _nothing_reads_the_chain(resolved, shared):
        coupling: Coupling = "none"
        planned = _plan_uncoupled(dispatch, shared, builds)
    else:
        coupling = config.execution.dependencies
        planned = _stream_solves(dispatch, builds, _solve_order(shared, resolved, builds), coupling, fail_fast=fail_fast, securities=_universe_securities(shared))
    builds.release()
    shape = planned.schedule.summary()
    log.info(
        "schedule derived: %d portfolio(s), %d edge(s), %d component(s), critical path %d",
        shape.portfolios,
        shape.edges,
        shape.components,
        shape.critical_path,
        extra={"run_id": shared.run_id, "stage": "schedule", "coupling": coupling, "largest_component": shape.largest_component},
    )
    artifacts = _gather_solves(dispatch, planned.schedule, planned.solves, planned.outcomes, fail_fast=fail_fast, run_dir=run_dir)
    return Executed(outcomes=planned.outcomes, schedule=planned.schedule, keys=planned.keys, artifacts=artifacts)


def _nothing_reads_the_chain(resolved: ResolvedConfig, shared: SharedRunData) -> bool:
    """Whether the run can be told, before any build, that no portfolio waits for another: no chain-aware term, the shipped solve step, and no constraint row that reads the chain.

    Derived, never declared: a config cannot switch the chain off, only the data and the steps can
    make it moot. When they do, every solve goes in behind its own build. The rows read here are the
    loaded ones, before any rule runs, so a rule that *adds* a chain reader would be invisible to the
    plan; :func:`_plan_uncoupled` fails such a portfolio at build rather than solving it blind.
    """
    return not resolved.chain_aware_terms and resolved.shipped_solve and not frame_reads_chain(shared.assembled.constraints)


def _check_workers(backend: Backend, shared: SharedRunData, session: _Session, expected: WorkerEnvironment) -> None:
    """Every worker the run starts with must resolve the config and match the run's fingerprint before any data is shared.

    A worker that cannot — the solver or a step package missing from its image, a stale image — would
    fail every portfolio it touched at stage ``worker``; one round trip here catches it before the run
    has done any work. Workers that join later are gated per result by :func:`_accept`.
    """
    problems: list[str] = []
    probes = backend.probe(probe_task, shared.config, shared.packages)
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


def _submit_builds(dispatch: _Dispatch, shared: SharedRunData) -> _Builds:
    """Submit every portfolio's build and its summary at once; a build reads no other portfolio, so none of them waits.

    A portfolio the load stage rejected has no inputs to build from and is not submitted; its failure
    is what it reports instead.
    """
    rejected = shared.assembled.rejected
    pending: dict[PortfolioId, Pending[TaskOutput[BuildResult]]] = {}
    summaries: dict[PortfolioId, Pending[TaskOutput[BuildSummary]]] = {}
    for rank, portfolio_id in enumerate(shared.assembled.portfolio_ids):
        if portfolio_id in rejected:
            continue
        pending[portfolio_id] = dispatch.backend.submit(build_task, dispatch.handle, portfolio_id, key=dispatch.key(portfolio_id, "build"), priority=dispatch.total - rank)
        summaries[portfolio_id] = dispatch.backend.submit(summarize, pending[portfolio_id], key=dispatch.key(portfolio_id, "summary"), priority=dispatch.total - rank + 1)
    fallback = {portfolio_id: Decimal(value) for portfolio_id, value in shared.assembled.solve_orders.items()}
    return _Builds(dispatch, fallback, pending, summaries, dict(rejected))


def _universe_securities(shared: SharedRunData) -> Iterable[str]:
    """Every security the assembled universe names, to seed the overlap index.

    A tradable set is drawn from the portfolio's universe, so this covers it unless a rule added a
    security the loaders never returned; one that did costs the index a repack, nothing more.
    """
    return (str(value) for value in shared.assembled.universe["security_id"])


def _solve_order(shared: SharedRunData, resolved: ResolvedConfig, builds: _Builds) -> tuple[PortfolioId, ...]:
    """The order a coupled run solves in.

    Without a solve-order step the key is the portfolios frame's column, which every portfolio carries
    before it is built: ``assembled.portfolio_ids`` is already that order. A configured step computes
    the key from the post-rules bundle instead, so the order is itself a build output and a coupled run
    has to wait for every build to learn who precedes whom — the one place the schedule cannot be
    streamed, and the reason to prefer the column when the priority can be expressed there.
    """
    ids = shared.assembled.portfolio_ids
    if resolved.solve_order is None:
        return ids
    for portfolio_id in ids:
        builds.report(portfolio_id)
    return order_portfolios({portfolio_id: builds.key(portfolio_id) for portfolio_id in ids})


def _plan_uncoupled(dispatch: _Dispatch, shared: SharedRunData, builds: _Builds) -> _Planned:
    """Nothing reads the chain, so no solve waits for another: each is submitted behind its own build.

    The order still decides how outcomes are classified and what the manifest records, but it is read
    back only once the solves are in flight — the summaries resolve while the cluster is already
    solving rather than in front of it, and no portfolio's build holds up another's solve. A portfolio
    whose build failed has its solve cancelled as soon as that is known, and so does one whose build
    reports a consume set: the plan was made from the loaded rows, so a chain reader can only have
    been added by a rule, and a solve with an empty chain would be silently wrong.
    """
    ids = shared.assembled.portfolio_ids
    solves = {
        portfolio_id: dispatch.backend.submit(solve_task, dispatch.handle, builds.take(portfolio_id), key=dispatch.key(portfolio_id, "solve"), priority=dispatch.solve_priority(position))
        for position, portfolio_id in enumerate(ids)
        if builds.submitted(portfolio_id)
    }
    outcomes: dict[PortfolioId, Outcome] = {}
    for portfolio_id in ids:
        report = builds.report(portfolio_id)
        if isinstance(report, PortfolioFailure):
            outcomes[portfolio_id] = report
        elif report.consumes:
            outcomes[portfolio_id] = _added_a_chain_reader(portfolio_id, report.consumes)
    dispatch.backend.cancel([solves.pop(portfolio_id) for portfolio_id in outcomes if portfolio_id in solves])
    keys = {portfolio_id: builds.key(portfolio_id) for portfolio_id in ids}
    order = order_portfolios(keys)
    return _Planned(Schedule(order, dict.fromkeys(order, ()), "none"), keys, solves, outcomes)


def _added_a_chain_reader(portfolio_id: PortfolioId, consumes: Sequence[str]) -> PortfolioFailure:
    """A build that consumes the chain in a run planned without one: its rules added a chain-reading row the plan could not see."""
    shown = f"{list(consumes[:3])}{'...' if len(consumes) > 3 else ''}"
    message = (
        f"its rules added a chain-reading constraint, coupling through {shown}, to a run the engine planned without a chain from the loaded rows; "
        "a rule may remove chain readers, never add them — load the row instead"
    )
    return PortfolioFailure(portfolio_id, "build", "ChainInvariantError", message)


def _stream_solves(dispatch: _Dispatch, builds: _Builds, order: tuple[PortfolioId, ...], coupling: Coupling, *, fail_fast: bool, securities: Iterable[str]) -> _Planned:
    """One pass down the solve order: read a portfolio's build, place it in the graph, submit its solve.

    Every predecessor is earlier in the order, so the pass never waits on a build it has not reached —
    the head of the order is solving while the tail of the book is still building. A portfolio whose
    build failed, or that waits on one that did, is settled here and never submitted; under
    ``fail_fast`` the first such failure stops submission, and the builds behind it are read only to
    finish the graph the manifest records.

    A *failed build* is treated as overlapping everything: it got far enough to have a tradable set and
    the run could not read it, so no later portfolio can be shown to be independent of it. A portfolio
    the *load stage* rejected is a different thing — it never entered the run, so it traded nothing,
    and it couples to nobody. It is recorded as failed and blocks no one, which is what lets a book
    with one uncovered account still solve the rest.
    """
    overlaps = OverlapIndex(len(order), securities)
    positions = {portfolio_id: position for position, portfolio_id in enumerate(order)}
    predecessors: dict[PortfolioId, tuple[PortfolioId, ...]] = {}
    outcomes: dict[PortfolioId, Outcome] = {}
    solves: dict[PortfolioId, Pending[TaskOutput[PortfolioResult]]] = {}
    contributions: dict[PortfolioId, Pending[Contribution | PortfolioFailure]] = {}

    def contribution_of(portfolio_id: PortfolioId) -> Pending[Contribution | PortfolioFailure]:
        """The predecessor's trades on the coupled side, reduced on its worker; submitted the first time a dependent asks, so a portfolio nothing waits for never pays for one."""
        pending = contributions.get(portfolio_id)
        if pending is None:
            priority = dispatch.solve_priority(positions[portfolio_id]) + 1
            pending = contributions[portfolio_id] = dispatch.backend.submit(contribution, solves[portfolio_id], key=dispatch.key(portfolio_id, "contribution"), priority=priority)
        return pending

    stopped = False
    for position, portfolio_id in enumerate(order):
        report = builds.report(portfolio_id)
        failed = isinstance(report, PortfolioFailure)
        earlier = overlaps.add((), (), unknown=builds.submitted(portfolio_id)) if failed else overlaps.add(report.tradable, report.consumes)
        predecessors[portfolio_id] = order[:position] if coupling == "all" else tuple(order[index] for index in earlier)
        if stopped:
            outcomes[portfolio_id] = skipped(portfolio_id, SKIPPED_BY_POSITION)
            continue
        blocked = next((other for other in predecessors[portfolio_id] if other in outcomes), None)
        if failed:
            outcomes[portfolio_id] = report
        elif blocked is not None:
            outcomes[portfolio_id] = _skipped_after(portfolio_id, blocked, outcomes[blocked])
        else:
            dependencies = [contribution_of(other) for other in predecessors[portfolio_id]]
            solves[portfolio_id] = dispatch.backend.submit(
                solve_task, dispatch.handle, builds.take(portfolio_id), *dependencies, key=dispatch.key(portfolio_id, "solve"), priority=dispatch.solve_priority(position)
            )
        stopped = fail_fast and portfolio_id in outcomes
    keys = {portfolio_id: builds.key(portfolio_id) for portfolio_id in order}
    return _Planned(Schedule(order, predecessors, coupling), keys, solves, outcomes)


def _gather_solves(
    dispatch: _Dispatch, schedule: Schedule, solves: Mapping[PortfolioId, Pending[TaskOutput[PortfolioResult]]], outcomes: dict[PortfolioId, Outcome], *, fail_fast: bool, run_dir: Path
) -> tuple[Artifact, ...]:
    """Collect solves as they complete, classify every portfolio in solve order, and persist each result as it is classified.

    Submission settles some portfolios itself — a failed build, a portfolio behind one — and the walk
    steps over those in place, so the first failure *in solve order* is found whether a build or a solve
    produced it. Under ``fail_fast`` it cancels every solve behind it; those are recorded as skipped by
    position, whatever they had finished, so the manifest never depends on timing.
    """
    artifacts: list[Artifact] = []
    raw: dict[PortfolioId, TaskOutput[PortfolioResult] | Exception] = {}
    settled = dict(outcomes)
    order = schedule.order
    cursor = 0
    stopped = False

    def advance() -> None:
        """Classify every portfolio the walk can now reach, in solve order, stopping at a failure under ``fail_fast``."""
        nonlocal cursor, stopped
        while not stopped and cursor < len(order) and (order[cursor] in settled or order[cursor] in raw):
            current = order[cursor]
            outcome = settled[current] if current in settled else _classify(current, raw[current], schedule, outcomes, dispatch.session, dispatch.expected)
            outcomes[current] = outcome
            cursor += 1
            if isinstance(outcome, PortfolioResult):
                artifacts.extend(_persist_result(outcome, run_dir))
            elif fail_fast:
                stopped = True

    advance()
    for portfolio_id in dispatch.backend.as_completed(solves):
        raw[portfolio_id] = _result_or_error(solves[portfolio_id])
        advance()
        if stopped:
            break
    if stopped:
        rest = [portfolio_id for portfolio_id in order[cursor:] if portfolio_id not in settled]
        dispatch.backend.cancel([solves[portfolio_id] for portfolio_id in rest if portfolio_id in solves])
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
        return PortfolioFailure.from_exception(portfolio_id, "worker", raw, message=_stable_message(raw))
    return _accept(raw, portfolio_id, session, expected, solved=True)


def _accept[T](raw: TaskOutput[T] | Exception, portfolio_id: PortfolioId, session: _Session, expected: WorkerEnvironment, *, solved: bool) -> T | PortfolioFailure:
    """Record who did the work and refuse a result from an environment that differs from this run's."""
    if isinstance(raw, Exception):
        return PortfolioFailure.from_exception(portfolio_id, "worker", raw, message=_stable_message(raw))
    session.saw(raw.environment, raw.host, solved=solved)
    session.absorb(raw.spans)
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


def _orders(outcomes: Sequence[Outcome], *, run_id: str) -> pd.DataFrame | None:
    """Every solved portfolio's orders as one frame, sorted — what the sink publishes and the checks prove — or ``None`` when nothing solved."""
    solved = [outcome for outcome in outcomes if isinstance(outcome, PortfolioResult)]
    if not solved:
        log.error("no portfolio solved; nothing published", extra={"run_id": run_id, "stage": "sink"})
        return None
    return pd.concat([result.orders for result in solved], ignore_index=True).sort_values(["portfolio_id", "security_id"], kind="stable").reset_index(drop=True)


def _publish(orders: pd.DataFrame | None, resolved: ResolvedConfig, io: IoContext) -> tuple[tuple[Artifact, ...], PortfolioFailure | None]:
    """Call the sink once with every solved portfolio's orders; a sink failure is infrastructure, and the manifest is still written."""
    if orders is None:
        return (), None
    artifacts: list[Artifact] = []
    try:
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
        return tuple(artifacts), PortfolioFailure.from_exception(RUN_SCOPED, "sink", error)
    return tuple(artifacts), None


def _run_checks(
    orders: pd.DataFrame | None, assembled: AssembledDatasets | None, outcomes: Sequence[Outcome], resolved: ResolvedConfig, run_dir: Path, *, run_id: str
) -> tuple[tuple[CheckStepRecord, ...], tuple[Artifact, ...], PortfolioFailure | None]:
    """Run every configured check once over the assembled datasets, the published orders, and the portfolios that solved, and write each one's failing rows.

    ``solved`` is the population a rule applies to — every portfolio that produced an answer, whether or
    not it produced an order, since a rule that stopped a trade is proven on the account that traded
    nothing. A check proves a business rule on what went out, so it runs only when something did. Every check
    runs whatever the others found; a check that raises, or returns something other than examined
    rows, is the run's own failure at stage ``check`` — a bug in the check, not a verdict on the
    orders — and the first such failure is the one recorded.
    """
    if orders is None or assembled is None or not resolved.checks:
        return (), (), None
    frames = assembled.frames()
    solved = pd.DataFrame({"portfolio_id": pd.Series(_solved_ids(outcomes), dtype="string")})
    records: list[CheckStepRecord] = []
    artifacts: list[Artifact] = []
    error_recorded: PortfolioFailure | None = None
    for label, step in resolved.checks.items():
        try:
            result = step.invoke(frames=frames, orders=orders, solved=solved)
            if not isinstance(result, pd.DataFrame):
                msg = f"check {label!r} ({step.qualname}) returned {type(result).__name__}, expected a DataFrame of the rows it examined"
                raise TypeError(msg)  # a contract violation is reported like any other failure of the check
            examined = validate_frame(result, CHECK_RESULTS)
        except Exception as error:  # noqa: BLE001  # the manifest must still be written; the exit code carries the failure
            log.error("check %r could not run", label, extra={"run_id": run_id, "stage": "check", "error": type(error).__name__})
            if error_recorded is None:
                error_recorded = PortfolioFailure.from_exception(RUN_SCOPED, "check", error, message=f"{label}: {error}")
            continue
        failed = examined.loc[~examined["ok"].astype("bool")]
        status: CheckStatus = "failed" if len(failed) else ("passed" if len(examined) else "not_exercised")
        records.append(
            CheckStepRecord(
                label=label,
                qualname=step.qualname,
                source_sha256=step.source_sha256,
                params_sha256=step.params_sha256,
                examined=len(examined),
                violations=len(failed),
                portfolios_affected=int(failed["portfolio_id"].nunique()),
                status=status,
                passed=status != "failed",
            )
        )
        log.log(
            logging.ERROR if status == "failed" else logging.INFO, "check %r %s: %d examined, %d violation(s)", label, status, len(examined), len(failed), extra={"run_id": run_id, "stage": "check"}
        )
        if len(failed):
            artifacts.append(_write_csv(run_dir / CHECKS_SUBDIR / f"{label}.csv", failed))
    return tuple(records), tuple(artifacts), error_recorded


def _solved_ids(outcomes: Sequence[Outcome]) -> list[str]:
    return [outcome.portfolio_id for outcome in outcomes if isinstance(outcome, PortfolioResult)]


def _write_csv(target: Path, frame: pd.DataFrame) -> Artifact:
    write_atomically(target, lambda path: frame.to_csv(path, index=False))
    return Artifact(path=str(target), sha256=file_sha256(target), size_bytes=target.stat().st_size)


def _persist_result(result: PortfolioResult, run_dir: Path) -> list[Artifact]:
    written: list[Artifact] = []
    for subdir, writer in (("problem_specs", result.spec.to_npz), ("solutions", result.solution.to_npz), ("chain", result.chain_state.to_npz)):
        directory = run_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{result.portfolio_id}.npz"
        writer(path)
        written.append(Artifact(path=str(path), sha256=file_sha256(path), size_bytes=path.stat().st_size))
    return written


def _exit_code(outcomes: Sequence[Outcome], infrastructure_error: PortfolioFailure | None, checks: Sequence[CheckStepRecord], check_error: PortfolioFailure | None) -> int:
    if infrastructure_error is not None:
        return EXIT_INFRASTRUCTURE
    if not outcomes or any(isinstance(outcome, PortfolioFailure) for outcome in outcomes) or check_error is not None or any(not check.passed for check in checks):
        return EXIT_PORTFOLIO_FAILED
    return EXIT_OK


def _manifest(
    resolved: ResolvedConfig,
    session: _Session,
    audits: Sequence[DatasetAudit],
    assembly_audits: Sequence[AssemblyAuditRecord],
    outcomes: Sequence[Outcome],
    executed: Executed | None,
    artifacts: Sequence[Artifact],
    exit_code: int,
    run_failures: Sequence[PortfolioFailure],
    checks: Sequence[CheckStepRecord],
    spans: Sequence[Span],
) -> RunManifest:
    config = resolved.config
    context = session.context
    post = config.post_solve
    keys: Mapping[PortfolioId, Decimal] = executed.keys if executed is not None else {}
    predecessors: Mapping[PortfolioId, tuple[PortfolioId, ...]] = executed.schedule.predecessors if executed is not None else {}
    records = [
        solved_record(o, post.violation_tol, solve_order=_key_text(keys, o.portfolio_id))
        if isinstance(o, PortfolioResult)
        else failed_record(o, solve_order=_key_text(keys, o.portfolio_id), predecessors=len(predecessors[PortfolioId(o.portfolio_id)]) if PortfolioId(o.portfolio_id) in predecessors else None)
        for o in outcomes
    ]
    records.extend(failed_record(failure) for failure in run_failures)
    solved = [o for o in outcomes if isinstance(o, PortfolioResult)]
    solver = configured_solver(config)
    solver_ver = solved[0].solution.solver_version if solved else solver_version(solver)
    packages = package_versions(step.qualname.partition(":")[0] for step in resolved.all_steps if step.is_external)
    manifest = RunManifest(
        run_id=context.io.run_id,
        run_name=config.run.name,
        tags=dict(config.run.tags),
        created_at_utc=created_at(context.io.clock()),
        as_of_date=context.as_of_date,
        git_sha=context.git.sha,
        git_dirty=context.git.dirty,
        schedule=executed.schedule.summary() if executed is not None else None,
        cluster=session.cluster_record(),
        versions=versions(solver, solver_ver, packages, session.worker_records()),
        config=ConfigInfo(path=context.config_path, sha256=resolved.config_sha256, resolved=config.model_dump(mode="json")),
        settings=dict(context.settings),
        terms=tuple(term.record() for term in resolved.terms),
        datasets=tuple(audits),
        assembly=tuple(assembly_audits),
        portfolios=tuple(records),
        checks=tuple(checks),
        artifacts=tuple(artifacts),
        timing=tuple(spans),
        exit_code=exit_code,
    )
    return finalize(manifest)


def _key_text(keys: Mapping[PortfolioId, Decimal], portfolio_id: str) -> str | None:
    key = keys.get(PortfolioId(portfolio_id))
    return None if key is None else str(key)
