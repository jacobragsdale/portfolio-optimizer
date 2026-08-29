"""Command-line entry points. ``main`` wires real collaborators; ``run_cli`` takes them as arguments."""

import argparse
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from portfolio_optimizer.config.models import config_sha256, load_run_config
from portfolio_optimizer.config.resolve import ConfigResolutionError, resolve_config
from portfolio_optimizer.config.schema import run_config_schema, schema_json
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import ChainState, PortfolioResult, ProblemSpec, Solution, StepRef, Tolerances
from portfolio_optimizer.domain.types import Clock, IdFactory
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.logging import configure_logging
from portfolio_optimizer.engine.manifest import diff_manifests, load_manifest, read_git_info
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_INPUT_REJECTED, EXIT_OK, EXIT_PORTFOLIO_FAILED, InputRejectedError, run
from portfolio_optimizer.settings import SettingsError, load_settings


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        """Current UTC time."""
        return datetime.now(tz=UTC)


class UuidIdFactory:
    """Random run ids."""

    def new_run_id(self) -> str:
        """A fresh run id."""
        return f"run-{uuid.uuid4().hex[:12]}"


def main() -> int:
    """Console-script entry point."""
    return run_cli(sys.argv[1:], env=os.environ, clock=SystemClock(), ids=UuidIdFactory(), stdout=sys.stdout, stderr=sys.stderr)


def run_cli(argv: Sequence[str], *, env: Mapping[str, str], clock: Clock, ids: IdFactory, stdout: TextIO, stderr: TextIO) -> int:
    """Parse ``argv`` and dispatch. Exit codes: 0 ok, 1 a portfolio failed, 2 inputs rejected, 3 infrastructure."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return EXIT_OK if exit_.code == 0 else EXIT_INPUT_REJECTED
    command = str(args.command)
    if command == "run":
        return _run(args, env=env, clock=clock, ids=ids, stdout=stdout, stderr=stderr)
    if command == "validate-config":
        return _validate_config(args, stdout=stdout, stderr=stderr)
    if command == "verify":
        return _verify(args, stdout=stdout, stderr=stderr)
    if command == "schema":
        stdout.write(schema_json(run_config_schema()))
        return EXIT_OK
    return _diff_manifests(args, stdout=stdout, stderr=stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-optimizer", description="JSON-driven, auditable portfolio optimization.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run every portfolio in a config")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--data-root", type=Path, default=None, help="override PORTFOLIO_OPTIMIZER_DATA_ROOT")
    run_parser.add_argument("--output", type=Path, default=None, help="override PORTFOLIO_OPTIMIZER_OUTPUT_DIR")
    validate = commands.add_parser("validate-config", help="validate and resolve a config without loading data")
    validate.add_argument("config", type=Path)
    verify_parser = commands.add_parser("verify", help="re-verify a persisted solution without cvxpy")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--portfolio", required=True)
    commands.add_parser("schema", help="print the JSON Schema for run configs (redirect to configs/run-config.schema.json)")
    diff = commands.add_parser("diff-manifests", help="name the first stage at which two runs diverge")
    diff.add_argument("left", type=Path)
    diff.add_argument("right", type=Path)
    return parser


def _run(args: argparse.Namespace, *, env: Mapping[str, str], clock: Clock, ids: IdFactory, stdout: TextIO, stderr: TextIO) -> int:
    try:
        settings = load_settings(env)
    except SettingsError as error:
        stderr.write(f"{error}\n")
        return EXIT_INPUT_REJECTED
    configure_logging(settings.log_level, stderr)
    config_path = Path(args.config)
    try:
        config = load_run_config(config_path.read_text())
        resolved = resolve_config(config, config_sha256(config))
    except OSError as error:
        stderr.write(f"cannot read config: {error}\n")
        return EXIT_INFRASTRUCTURE
    except (ValidationError, ConfigResolutionError) as error:
        stderr.write(f"config rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    data_root = Path(args.data_root) if args.data_root is not None else settings.data_root
    output_dir = Path(args.output) if args.output is not None else settings.output_dir
    io = IoContext(data_root=data_root, output_dir=output_dir, run_id=ids.new_run_id(), clock=clock)
    shown_settings = {"data_root": str(data_root), "output_dir": str(output_dir), "log_level": settings.log_level}
    try:
        report = run(resolved, io, git=read_git_info(Path.cwd()), config_path=str(config_path), settings=shown_settings)
    except InputRejectedError as error:
        stderr.write(f"{error}\n")
        return EXIT_INPUT_REJECTED
    except OSError as error:
        stderr.write(f"infrastructure failure: {error}\n")
        return EXIT_INFRASTRUCTURE
    stdout.write(f"run {report.run_id}: manifest {report.manifest_path}\n")
    for outcome in report.outcomes:
        if isinstance(outcome, PortfolioResult):
            stdout.write(f"  {outcome.portfolio_id}: solved, {len(outcome.orders)} order(s)\n")
        else:
            stdout.write(f"  {outcome.portfolio_id}: FAILED at {outcome.stage}: {outcome.error_type}: {outcome.message}\n")
    stdout.write(f"exit code {report.exit_code}\n")
    return report.exit_code


def _validate_config(args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO) -> int:
    try:
        config = load_run_config(Path(args.config).read_text())
        resolved = resolve_config(config, config_sha256(config))
    except OSError as error:
        stderr.write(f"cannot read config: {error}\n")
        return EXIT_INFRASTRUCTURE
    except (ValidationError, ConfigResolutionError) as error:
        stderr.write(f"config rejected: {error}\n")
        return EXIT_INPUT_REJECTED
    stdout.write(
        f"config ok (sha256 {resolved.config_sha256[:12]}): {len(resolved.rules)} rule(s), {len(resolved.terms)} term(s), {len(resolved.constraints)} constraint(s), mode {config.execution.mode}\n"
    )
    stdout.writelines(f"  {step.kind:19} {step.qualname}{' [external]' if step.is_external else ''}{' [' + step.context_name + ']' if step.context_name else ''}\n" for step in resolved.all_steps)
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
    terms = [StepRef(t.qualname, t.params) for t in manifest.terms]
    constraints = [StepRef(c.qualname, c.params) for c in manifest.constraints]
    report = verify(spec, solution, chain, terms, constraints, Tolerances(eq=record.check.tolerance_eq, ineq=record.check.tolerance_ineq))
    stdout.writelines(
        f"  {'ok  ' if check.passed else 'FAIL'} {check.name:32} violation {check.violation:.3e} (tol {check.tolerance:.1e}){' worst ' + check.worst_security if check.worst_security else ''}\n"
        for check in report.checks
    )
    stdout.write(
        f"  objective recomputed {report.recomputed_objective:.9g} vs solver {report.solver_objective:.9g} (gap {report.objective_gap:.3e}){' unverified: ' + ', '.join(report.unverified) if report.unverified else ''}\n"
    )
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
