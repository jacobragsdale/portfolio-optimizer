"""Command-line entry points. ``main`` wires real collaborators; ``run_cli`` takes them as arguments."""

import argparse
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from portfolio_optimizer.config.models import load_run_config
from portfolio_optimizer.config.resolve import TEMPLATE_MODULES, ConfigResolutionError, published_steps, resolve_config
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
from portfolio_optimizer.engine.manifest import diff_manifests, failure_report_path, load_manifest
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
    try:
        resolved = resolve_config(load_run_config(config_path.read_text()), packages=settings.packages())
    except OSError as error:
        stderr.write(f"cannot read config: {error}\n")
        return EXIT_INFRASTRUCTURE
    except (ValidationError, ConfigResolutionError) as error:
        stderr.write(f"config rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    data_root = Path(args.data_root) if args.data_root is not None else settings.data_root
    output_dir = Path(args.output) if args.output is not None else settings.output_dir
    execution = settings.execution()
    if args.max_workers is not None:
        if int(args.max_workers) < execution.min_workers:
            stderr.write(f"--max-workers must be at least PORTFOLIO_OPTIMIZER_MIN_WORKERS ({execution.min_workers})\n")
            return EXIT_INPUT_REJECTED
        execution = replace(execution, max_workers=int(args.max_workers))
    context = RunContext(
        io=IoContext(data_root=data_root, output_dir=output_dir, run_id=new_run_id(), clock=clock),
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
    stdout.write(f"exit code {report.exit_code}\n")
    return report.exit_code


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
    lines = diff_manifests(left, right)
    if not lines:
        stdout.write("no differences\n")
        return EXIT_OK
    stdout.write("\n".join(lines) + "\n")
    return EXIT_PORTFOLIO_FAILED
