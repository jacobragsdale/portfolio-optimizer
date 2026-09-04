"""Command-line entry points. ``main`` wires real collaborators; ``run_cli`` takes them as arguments."""

import argparse
import os
import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from portfolio_optimizer.config.models import InlinePortfolios, RunConfig, load_run_config
from portfolio_optimizer.config.resolve import TEMPLATE_MODULES, ConfigResolutionError, ResolvedConfig, published_steps, resolve_config
from portfolio_optimizer.config.schema import installed_steps, run_config_schema, schema_json
from portfolio_optimizer.config.steps import StepKind
from portfolio_optimizer.domain.constraints import constraint_kinds, parse_constraint
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.objective import TermSpecError, parse_terms, term_kinds
from portfolio_optimizer.domain.order_flow import profile_for
from portfolio_optimizer.domain.results import ChainState, PortfolioResult, ProblemSpec, Solution, Tolerances
from portfolio_optimizer.domain.types import Clock
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.environment import read_git_info
from portfolio_optimizer.engine.logging import configure_logging
from portfolio_optimizer.engine.manifest import MANIFEST_FILENAME, PortfolioRecord, RunManifest, diff_manifests, failure_report_path, load_manifest
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_INPUT_REJECTED, EXIT_OK, EXIT_PORTFOLIO_FAILED, InputRejectedError, RunContext, run
from portfolio_optimizer.settings import Settings, SettingsError, load_settings


def system_clock() -> datetime:
    """The real clock."""
    return datetime.now(tz=UTC)


def new_uuid_run_id() -> str:
    """A fresh random run id."""
    return f"run-{uuid.uuid4().hex[:12]}"


def main() -> int:
    """Console-script entry point."""
    return run_cli(sys.argv[1:], env=os.environ, clock=system_clock, new_run_id=new_uuid_run_id, stdout=sys.stdout, stderr=sys.stderr)


def run_cli(argv: Sequence[str], *, env: Mapping[str, str], clock: Clock, new_run_id: Callable[[], str], stdout: TextIO, stderr: TextIO) -> int:
    """Parse ``argv`` and dispatch. Exit codes: 0 ok, 1 a portfolio failed, 2 inputs rejected, 3 infrastructure."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return EXIT_OK if exit_.code == 0 else EXIT_INPUT_REJECTED
    command = str(args.command)
    if command == "run":
        return _run(args, env=env, clock=clock, new_run_id=new_run_id, stdout=stdout, stderr=stderr)
    if command == "validate-config":
        return _validate_config(args, env=env, stdout=stdout, stderr=stderr)
    if command == "verify":
        return _verify(args, stdout=stdout, stderr=stderr)
    if command == "schema":
        stdout.write(schema_json(run_config_schema()))
        return EXIT_OK
    if command == "steps":
        return _steps(stdout=stdout)
    return _diff_manifests(args, stdout=stdout, stderr=stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-optimizer", description="JSON-driven, auditable portfolio optimization.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run every portfolio in a config")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--as-of", required=True, metavar="DATETIME", help="the timezone-aware instant the run is as of, e.g. 2026-09-01T00:00:00Z; every loader receives it")
    run_parser.add_argument("--data-root", type=Path, default=None, help="override PORTFOLIO_OPTIMIZER_DATA_ROOT")
    run_parser.add_argument("--output", type=Path, default=None, help="override PORTFOLIO_OPTIMIZER_OUTPUT_DIR")
    run_parser.add_argument("--max-workers", type=int, default=None, help="override PORTFOLIO_OPTIMIZER_MAX_WORKERS")
    run_parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help="the run's id, and so the name of its output directory (default: a fresh random one); letters, digits, '.', '_', '-', and not one the output directory already holds a manifest for",
    )
    run_parser.add_argument(
        "--retry-of",
        type=Path,
        default=None,
        metavar="MANIFEST",
        help="retry the portfolios that run recorded as failed — at stage solve unless --retry-stages says otherwise — under this config: exactly those ids, in their solve order, written inline as the book and tagged retry_of",
    )
    run_parser.add_argument(
        "--retry-stages",
        default="solve",
        metavar="STAGES",
        help=f"with --retry-of: the failure stages to retry, comma-separated, from {sorted(RETRY_STAGES)}; skipped is what fail_fast left behind a failure (default: solve)",
    )
    run_parser.add_argument(
        "--retry-errors", default=None, metavar="ERRORS", help="with --retry-of: retry only failures of these exception types, comma-separated, e.g. InfeasibleError,VerificationError (default: any)"
    )
    validate = commands.add_parser("validate-config", help="validate and resolve a config without loading data")
    validate.add_argument("config", type=Path)
    verify_parser = commands.add_parser("verify", help="re-verify a persisted solution without cvxpy")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--portfolio", required=True)
    commands.add_parser("schema", help="print the JSON Schema for run configs (redirect to configs/run-config.schema.json)")
    commands.add_parser("steps", help="list every step and every term and constraint kind this environment can name")
    diff = commands.add_parser("diff-manifests", help="name the first stage at which two runs diverge")
    diff.add_argument("left", type=Path)
    diff.add_argument("right", type=Path)
    return parser


RETRY_STAGES: frozenset[str] = frozenset({"load", "slice", "build", "solve", "worker", "skipped"})
"""The stages a portfolio can fail at, and so the ones ``--retry-stages`` may name."""

DEFAULT_RETRY_STAGES: frozenset[str] = frozenset({"solve"})


class RetryError(ValueError):
    """``--retry-of`` matched nothing: no portfolio in the manifest failed at the stages, and with the errors, asked for."""


@dataclass(frozen=True, slots=True)
class RetrySelection:
    """What ``--retry-of`` retries: the manifest, the failure stages, and — optionally — the exception types."""

    manifest: Path
    stages: frozenset[str]
    errors: frozenset[str] | None


def retry_of(config: RunConfig, manifest: RunManifest, *, stages: frozenset[str] = DEFAULT_RETRY_STAGES, errors: frozenset[str] | None = None) -> RunConfig:
    """``config`` over exactly the portfolios ``manifest`` recorded as failed at one of ``stages`` — and, when ``errors`` is given, with one of those exception types — in their solve order, tagged with the run it retries.

    The config is whatever the desk wants the second attempt to be: the same wiring with the build's
    ``hold_breached_starts`` on for a start the order flow could not trade out of, a looser
    ``post_solve`` or another solver for a solve that hit its limit or failed verification, a
    rebalance, or the original config unchanged over the portfolios ``fail_fast`` skipped behind one
    failure. A retry is a clean run: nothing from the failed run reaches it but the ids — no cash
    carried forward, no chain, no state; what a run traded reaches a later one only as data it
    loads. The ids are written inline, so the retry's config hash differs from the original's: it is
    a different run over a different book, and the manifest's ``retry_of`` tag says which.
    """
    failed = [record for record in manifest.portfolios if record.status == "failed" and record.failure_stage in stages and (errors is None or _error_type(record) in errors)]
    if not failed:
        recorded = Counter((record.failure_stage or "unknown", _error_type(record)) for record in manifest.portfolios if record.status == "failed")
        detail = ", ".join(f"{count} at {stage} ({error})" for (stage, error), count in sorted(recorded.items())) or "every portfolio solved"
        wanted = f"at {', '.join(sorted(stages))}" + ("" if errors is None else f" with {', '.join(sorted(errors))}")
        msg = f"no portfolio in run {manifest.run_id} failed {wanted}; the run recorded {detail}"
        raise RetryError(msg)
    ordered = sorted(failed, key=lambda record: (Decimal(record.solve_order) if record.solve_order is not None else Decimal(0), record.portfolio_id))
    portfolios = InlinePortfolios(ids=tuple(record.portfolio_id for record in ordered))
    run = config.run.model_copy(update={"tags": {**config.run.tags, "retry_of": manifest.run_id}})
    return config.model_copy(update={"datasets": {**config.datasets, "portfolios": portfolios}, "run": run})


def _error_type(record: PortfolioRecord) -> str:
    """The exception type a failed record names; the manifest writes ``error`` as ``Type: message``."""
    return record.error.partition(":")[0] if record.error else "unknown"


def _names(text: str) -> frozenset[str]:
    """A comma-separated flag value as a set of names, blanks dropped."""
    return frozenset(part.strip() for part in text.split(",") if part.strip())


def parse_as_of(text: str) -> datetime:
    """An ISO 8601 instant with a zone, normalized to UTC; a naive one is refused because a holding period compared against it would be off by hours."""
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as error:
        msg = f"--as-of must be an ISO 8601 instant such as 2026-09-01T00:00:00Z, got {text!r}"
        raise ValueError(msg) from error
    if instant.tzinfo is None:
        msg = f"--as-of must carry a time zone (2026-09-01T00:00:00Z), got the naive {text!r}"
        raise ValueError(msg)
    return instant.astimezone(UTC)


def _settings(env: Mapping[str, str], stderr: TextIO) -> Settings | None:
    try:
        return load_settings(env)
    except SettingsError as error:
        stderr.write(f"{error}\n")
        return None


def _run(args: argparse.Namespace, *, env: Mapping[str, str], clock: Clock, new_run_id: Callable[[], str], stdout: TextIO, stderr: TextIO) -> int:
    settings = _settings(env, stderr)
    if settings is None:
        return EXIT_INPUT_REJECTED
    configure_logging(settings.log_level, stderr)
    try:
        as_of_date = parse_as_of(str(args.as_of))
    except ValueError as error:
        stderr.write(f"{error}\n")
        return EXIT_INPUT_REJECTED
    config_path = Path(args.config)
    retry = _retry_selection(args, stderr)
    if isinstance(retry, int):
        return retry
    resolved = _resolved_config(config_path, retry, settings, stderr)
    if isinstance(resolved, int):
        return resolved
    data_root = Path(args.data_root) if args.data_root is not None else settings.data_root
    output_dir = Path(args.output) if args.output is not None else settings.output_dir
    execution = settings.execution()
    if args.max_workers is not None:
        if int(args.max_workers) < execution.min_workers:
            stderr.write(f"--max-workers must be at least PORTFOLIO_OPTIMIZER_MIN_WORKERS ({execution.min_workers})\n")
            return EXIT_INPUT_REJECTED
        execution = replace(execution, max_workers=int(args.max_workers))
    run_id = _run_id(args, output_dir, new_run_id, stderr)
    if run_id is None:
        return EXIT_INPUT_REJECTED
    context = RunContext(
        io=IoContext(data_root=data_root, output_dir=output_dir, run_id=run_id, clock=clock),
        as_of_date=as_of_date,
        execution=execution,
        git=read_git_info(Path.cwd()),
        config_path=str(config_path),
        settings=settings.shown() | {"data_root": str(data_root), "output_dir": str(output_dir), "max_workers": str(execution.max_workers)},
    )
    try:
        report = run(resolved, context)
    except InputRejectedError as error:
        stderr.write(f"{error}\n")
        return EXIT_INPUT_REJECTED
    except OSError as error:
        stderr.write(f"infrastructure failure: {error}\n")
        return EXIT_INFRASTRUCTURE
    stdout.write(f"run {report.run_id}: manifest {report.manifest_path}\n")
    run_dir = report.manifest_path.parent
    for outcome in report.outcomes:
        if isinstance(outcome, PortfolioResult):
            binding = f"; binding: {', '.join(outcome.report.active)}" if outcome.report.active else ""
            stdout.write(f"  {outcome.portfolio_id}: solved, {len(outcome.orders)} order(s){binding}\n")
        else:
            traceback_hint = "" if outcome.traceback is None else f" (traceback: {failure_report_path(run_dir, outcome)})"
            stdout.write(f"  {outcome.portfolio_id}: FAILED at {outcome.stage}: {outcome.error_type}: {outcome.message}{traceback_hint}\n")
    stdout.writelines(f"  check {check.label}: {check.status}, {check.examined} examined, {check.violations} violation(s)\n" for check in report.manifest.checks)
    stdout.write(f"exit code {report.exit_code}\n")
    return report.exit_code


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
"""What ``--run-id`` may be: it names the run's output directory."""


def _run_id(args: argparse.Namespace, output_dir: Path, new_run_id: Callable[[], str], stderr: TextIO) -> str | None:
    """The run's id: ``--run-id`` when given and usable, else a fresh one; ``None`` with the refusal written when it is not."""
    if args.run_id is None:
        return new_run_id()
    run_id = str(args.run_id)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        stderr.write(f"--run-id {run_id!r} must be letters, digits, '.', '_', or '-': it names the run's output directory\n")
        return None
    if (output_dir / run_id / MANIFEST_FILENAME).exists():
        stderr.write(f"--run-id {run_id!r} already has a manifest under {output_dir}; a run id is used once\n")
        return None
    return run_id


def _retry_selection(args: argparse.Namespace, stderr: TextIO) -> RetrySelection | int | None:
    """What ``--retry-of`` and its selectors ask for, ``None`` without a manifest, or the exit code that refuses them."""
    stages = _names(str(args.retry_stages))
    errors = None if args.retry_errors is None else _names(str(args.retry_errors))
    if args.retry_of is None:
        if stages != DEFAULT_RETRY_STAGES or errors is not None:
            stderr.write("--retry-stages and --retry-errors select what --retry-of retries; give it a manifest\n")
            return EXIT_INPUT_REJECTED
        return None
    unknown = sorted(stages - RETRY_STAGES)
    if unknown or not stages:
        stderr.write(f"--retry-stages names {unknown or 'no stage'}; a failure stage is one of {sorted(RETRY_STAGES)}\n")
        return EXIT_INPUT_REJECTED
    return RetrySelection(Path(args.retry_of), stages, errors)


def _resolved_config(config_path: Path, retry: RetrySelection | None, settings: Settings, stderr: TextIO) -> ResolvedConfig | int:
    """The config at ``config_path`` resolved — over the failed portfolios ``retry`` selects from its manifest, when given — or the exit code that stops the run."""
    try:
        config = load_run_config(config_path.read_text())
    except OSError as error:
        stderr.write(f"cannot read config: {error}\n")
        return EXIT_INFRASTRUCTURE
    except ValidationError as error:
        stderr.write(f"config rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    if retry is not None:
        try:
            manifest = load_manifest(retry.manifest.read_text())
        except OSError as error:
            stderr.write(f"cannot read manifest: {error}\n")
            return EXIT_INFRASTRUCTURE
        except (ValidationError, ValueError) as error:
            stderr.write(f"manifest rejected: {error}\n")
            return EXIT_INPUT_REJECTED
        try:
            config = retry_of(config, manifest, stages=retry.stages, errors=retry.errors)
        except RetryError as error:
            stderr.write(f"{error}\n")
            return EXIT_INPUT_REJECTED
    try:
        return resolve_config(config, packages=settings.packages())
    except ConfigResolutionError as error:
        stderr.write(f"config rejected: {error}\n")
        return EXIT_INPUT_REJECTED


def _validate_config(args: argparse.Namespace, *, env: Mapping[str, str], stdout: TextIO, stderr: TextIO) -> int:
    settings = _settings(env, stderr)
    if settings is None:
        return EXIT_INPUT_REJECTED
    try:
        resolved = resolve_config(load_run_config(Path(args.config).read_text()), packages=settings.packages())
    except OSError as error:
        stderr.write(f"cannot read config: {error}\n")
        return EXIT_INFRASTRUCTURE
    except (ValidationError, ConfigResolutionError) as error:
        stderr.write(f"config rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    stdout.write(f"config ok (sha256 {resolved.config_sha256[:12]}): {len(resolved.rules)} rule(s), {len(resolved.terms)} term(s), dependencies {resolved.config.execution.dependencies}\n")
    stdout.writelines(f"  {step.kind:19} {step.qualname}{' [external]' if step.is_external else ''}\n" for step in resolved.all_steps)
    stdout.writelines(f"  {'term':19} {term.name} ({type(term).__name__}){' [chain]' if term.reads_chain else ''}\n" for term in resolved.terms)
    return EXIT_OK


def _steps(*, stdout: TextIO) -> int:
    """Every step a bare name can resolve to, by kind, and every term and constraint kind: the template's, and what installed packages publish."""
    for kind in TEMPLATE_MODULES:
        step_kind: StepKind = kind
        stdout.write(f"{step_kind} ({TEMPLATE_MODULES[step_kind]})\n")
        published = published_steps(step_kind)
        for name, params in installed_steps(step_kind).items():
            source = f" [{published[name][0]}]" if name in published else ""
            fields = ", ".join(params.model_fields) if params is not None else ""
            stdout.write(f"  {name}{source}{f' ({fields})' if fields else ''}\n")
    for title, kinds in (("term kinds", term_kinds()), ("constraint kinds", constraint_kinds())):
        stdout.write(f"{title}\n")
        for name, model in sorted(kinds.items()):
            fields = ", ".join(field for field in model.model_fields if field != "kind")
            stdout.write(f"  {name} ({fields})\n")
    return EXIT_OK


def _verify(args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO) -> int:
    manifest_path = Path(args.manifest)
    try:
        manifest = load_manifest(manifest_path.read_text())
    except OSError as error:
        stderr.write(f"cannot read manifest: {error}\n")
        return EXIT_INFRASTRUCTURE
    except (ValidationError, ValueError) as error:
        stderr.write(f"manifest rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    portfolio_id = str(args.portfolio)
    record = next((p for p in manifest.portfolios if p.portfolio_id == portfolio_id), None)
    if record is None or record.status != "solved":
        stderr.write(f"portfolio {portfolio_id!r} was not solved in this run\n")
        return EXIT_INPUT_REJECTED
    run_dir = manifest_path.parent
    try:
        spec = ProblemSpec.from_npz(run_dir / "problem_specs" / f"{portfolio_id}.npz")
        solution = Solution.from_npz(run_dir / "solutions" / f"{portfolio_id}.npz")
        chain = ChainState.from_npz(run_dir / "chain" / f"{portfolio_id}.npz")
    except OSError as error:
        stderr.write(f"cannot read persisted problem: {error}\n")
        return EXIT_INFRASTRUCTURE
    if record.check is None or spec.content_hash() != record.problem_spec_sha256:
        stderr.write("persisted spec does not match the manifest's spec hash\n")
        return EXIT_PORTFOLIO_FAILED
    if chain.content_hash() != record.chain_inputs_sha256:
        stderr.write("persisted chain state does not match the manifest's chain hash\n")
        return EXIT_PORTFOLIO_FAILED
    try:
        terms = parse_terms(manifest.terms)
        constraints = tuple(parse_constraint(constraint, f"constraints[{index}]") for index, constraint in enumerate(record.constraints))
    except (TermSpecError, ValueError) as error:
        stderr.write(f"the manifest names a kind this environment does not know: {error}\n")
        return EXIT_INPUT_REJECTED
    profile = profile_for(str(manifest.config.resolved["order_flow"]))
    report = verify(spec, solution, chain, terms, constraints, profile=profile, tolerances=Tolerances(violation=record.check.tolerance))
    stdout.writelines(
        f"  {'ok  ' if check.passed else 'FAIL'} {check.display:32} violation {check.violation:.3e} (tol {check.tolerance:.1e}){' worst ' + check.worst_security if check.worst_security else ''}{' [binding]' if check.active else ''}\n"
        for check in report.checks
    )
    solver_objective = "none (the solve step minimized nothing)" if report.solver_objective is None else f"{report.solver_objective:.9g}"
    stdout.write(f"  objective recomputed {report.recomputed_objective:.9g} vs solver {solver_objective} (gap {report.objective_gap:.3e})\n")
    stdout.write(f"{'VERIFIED' if report.passed else 'VERIFICATION FAILED'} {portfolio_id} (spec {spec.content_hash()[:12]})\n")
    return EXIT_OK if report.passed else EXIT_PORTFOLIO_FAILED


def _diff_manifests(args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO) -> int:
    try:
        left = load_manifest(Path(args.left).read_text())
        right = load_manifest(Path(args.right).read_text())
    except OSError as error:
        stderr.write(f"cannot read manifest: {error}\n")
        return EXIT_INFRASTRUCTURE
    except (ValidationError, ValueError) as error:
        stderr.write(f"manifest rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    divergences = diff_manifests(left, right)
    if not divergences:
        stdout.write("no differences\n")
        return EXIT_OK
    stdout.write("\n".join(divergence.line for divergence in divergences) + "\n")
    return EXIT_PORTFOLIO_FAILED
