"""The half of a book benchmark that is not the book: the run config it executes, the cluster it runs on, and what it reports.

``run_book.py`` and ``run_state_book.py`` differ only in the book they generate — one partitions a
universe into interchangeable mandate groups, the other gives a municipal desk's three sleeves over one
universe of bonds. Everything downstream is the same run: the shipped loaders with their latency turned
off, ``restrict_to_mandate``, the example's objective, a local Dask cluster, and a report read back out
of the manifest the run wrote.

A sibling module rather than a package, because the benchmarks are scripts run by path and the
directory they live in is what Python puts on the path first.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from portfolio_optimizer.cli import system_clock
from portfolio_optimizer.config.models import load_run_config
from portfolio_optimizer.config.resolve import resolve_config
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import PortfolioFailure
from portfolio_optimizer.engine.environment import read_git_info
from portfolio_optimizer.engine.runner import RunContext, RunReport, run
from portfolio_optimizer.settings import ExecutionSettings

AS_OF = datetime(2026, 8, 28, tzinfo=UTC)
NO_LATENCY = {"min_latency_s": 0, "max_latency_s": 0}
CONSTRAINT_COLUMNS = ("portfolio_id", "kind", "label", "params")
SHIPPED_CONSTRAINTS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("cash_limit", "cash_floor", {"direction": ">=", "bounds": {"scalar": "cash_lb"}}),
    ("cash_limit", "cash_cap", {"direction": "<=", "bounds": {"scalar": "cash_ub"}}),
    ("turnover_limit", "turnover", {"direction": "<=", "bounds": {"scalar": "max_turnover"}}),
    ("participation_limit", "adv", {"direction": "<="}),
)
"""The example's typed constraint rows for an account without bands: ``(kind, label, params)`` — the cash bounds, the turnover cap, and the chain-aware ADV cap."""
ALPHA: dict[str, object] = {"kind": "linear", "name": "alpha", "weight": "-1", "column": "alpha"}
TAX_COST: dict[str, object] = {"kind": "linear", "name": "tax_cost", "weight": "1", "column": "tax_per_dollar", "vector": "sell"}
TRANSACTION_COST: dict[str, object] = {"kind": "linear", "name": "transaction_cost", "weight": "1", "column": "tcost_per_dollar", "vector": "trade"}
OBJECTIVE: dict[str, list[dict[str, object]]] = {"buy": [ALPHA, TRANSACTION_COST], "sell": [ALPHA, TAX_COST, TRANSACTION_COST]}
"""The example's objective for each side, as the typed records the config carries; ``tax_cost`` reads ``sell`` and so belongs to the sell program."""


def constraint_row(portfolio_id: str, kind: str, label: str, params: dict[str, object]) -> tuple[str, str, str, str]:
    """One typed constraint row the way the CSV loader reads it: ``params`` as JSON text."""
    return (portfolio_id, kind, label, json.dumps(params))


def config_body(dependencies: str) -> dict[str, object]:
    """The run config a book benchmark executes: the buy program over the shipped loaders with zero latency, the mandate rule, and the example's terms."""
    loader = lambda name: {"loader": {"name": name, "params": dict(NO_LATENCY)}}  # noqa: E731
    with_book = lambda name: {**loader(name), "depends_on": ["portfolios"]}  # noqa: E731
    return {
        "run": {"name": "book_benchmark", "tags": {"purpose": "benchmark"}},
        "sides": "buy",
        "datasets": {
            "portfolios": loader("load_portfolios"),
            "holdings": with_book("load_holdings"),
            "universe": loader("load_universe"),
            "details": with_book("load_details"),
            "constraints": with_book("load_constraints"),
            "mandates": with_book("load_mandates"),
        },
        "rules": ["restrict_to_mandate"],
        "objective": OBJECTIVE["buy"],
        "solve": {"name": "cvxpy", "params": {"solver": "CLARABEL", "options": {"max_iter": 200}}},
        "sink": {"name": "orders_to_parquet"},
        "execution": {"on_error": "continue", "dependencies": dependencies},
    }


def execute(book: Path, out: Path, run_id: str, dependencies: str, workers: int, benchmark: str) -> RunReport:
    """The real runner over the generated book, on a local cluster of ``workers`` processes; ``benchmark`` names the generator in the manifest's settings."""
    config_path = out / "run.json"
    config_path.write_text(json.dumps(config_body(dependencies), indent=2) + "\n")
    resolved = resolve_config(load_run_config(config_path.read_text()))
    execution = ExecutionSettings(cluster="local", min_workers=workers, max_workers=workers, cluster_timeout_s=300.0)
    context = RunContext(
        io=IoContext(data_root=book, output_dir=out, run_id=run_id, clock=system_clock),
        as_of_date=AS_OF,
        execution=execution,
        git=read_git_info(Path.cwd()),
        config_path=str(config_path),
        settings={"benchmark": benchmark},
    )
    return run(resolved, context)


def report_lines(report: RunReport) -> list[str]:
    """What a book benchmark says about the run: outcome counts, the derived schedule, and the solve spans the manifest recorded."""
    manifest = report.manifest
    failed = [outcome for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)]
    lines = [f"run {report.run_id}: exit {report.exit_code}, {len(report.solved)} solved, {len(failed)} failed"]
    lines.extend(f"  FAILED {failure.portfolio_id} at {failure.stage}: {failure.message[:160]}" for failure in failed[:5])
    if manifest.schedule is not None:
        shape = manifest.schedule
        lines.append(
            f"schedule: coupling {shape.coupling}, {shape.portfolios} portfolio(s), {shape.edges} edge(s), {shape.components} component(s), largest {shape.largest_component}, critical path {shape.critical_path}"
        )
    solves = [span for span in manifest.timing if span.name == "solve"]
    if solves:
        total = sum(span.duration_s for span in solves)
        lines.append(
            f"solve spans: {len(solves)}, total {total:.1f}s, mean {total / len(solves):.2f}s; critical-path lower bound at that mean: {manifest.schedule.critical_path * total / len(solves):.1f}s"
            if manifest.schedule
            else ""
        )
    lines.append(f"manifest: {report.manifest_path}")
    return lines
