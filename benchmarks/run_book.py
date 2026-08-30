"""Run a synthetic book of *N* portfolios through the real engine on a local cluster and report how the derived schedule behaved.

Where ``profile_portfolio.py`` times one portfolio's pipeline in one process, this measures the *run*:
the book builds and solves on the same Dask cluster a real run provisions, the engine derives the
dependency graph from the mandates' overlap structure, and the report is the manifest's own record —
the schedule summary (edges, components, critical path) and the timing spans, rendered the way
``portfolio-optimizer timeline`` renders them.

The book is generated from a seed. Every portfolio holds and may buy only names inside its mandate
(``--groups`` mandate groups over ``--sectors`` sectors, portfolio *i* in group ``i % groups``), so the
overlap structure — and therefore the schedule — is a parameter: ``--groups 1`` is the degenerate
single-universe book where every portfolio couples to every earlier one, larger values partition the
book into that many independent components, and ``--mandate-overlap M`` extends each mandate over the
next *M* groups' sectors as well, ring-wise — disjoint components become one connected component whose
critical path sits between the partitioned book's and the line's, which is what a real book of
overlapping mandates looks like. ``--dependencies all`` runs the same book as a strict line for
comparison.

    uv run python benchmarks/run_book.py --portfolios 100 --groups 10
    uv run python benchmarks/run_book.py --portfolios 100 --groups 10 --mandate-overlap 1
    uv run python benchmarks/run_book.py --portfolios 100 --groups 1 --dependencies all
"""

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from portfolio_optimizer.cli import SystemClock
from portfolio_optimizer.config.models import load_run_config
from portfolio_optimizer.config.resolve import resolve_config
from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import PortfolioFailure
from portfolio_optimizer.engine.environment import read_git_info
from portfolio_optimizer.engine.runner import RunContext, RunReport, run
from portfolio_optimizer.engine.timing import render_timeline
from portfolio_optimizer.settings import ExecutionSettings

AS_OF = "2026-08-28T00:00:00Z"
NO_LATENCY = {"min_latency_s": 0, "max_latency_s": 0}
SHIPPED_CONSTRAINTS = ("long_only", "max_weight", "cash_bounds", "turnover_cap", "cumulative_adv_participation")


def write_book(root: Path, rng: np.random.Generator, *, portfolios: int, securities: int, sectors: int, groups: int, held: int, mandate_overlap: int) -> None:
    """Write the CSV tables of one synthetic book: a shared universe, and per account a mandate, holdings inside it, details, and constraint rows.

    ``mandate_overlap`` widens each group's mandate over the next that many groups' sectors, ring-wise,
    so neighbouring groups share names and the graph connects without collapsing to the complete DAG.
    """
    root.mkdir(parents=True, exist_ok=True)
    sector_names = [f"SEC{index:03d}" for index in range(sectors)]
    sector_of = [index % sectors for index in range(securities)]
    prices = [Decimal(int(value)) / 100 for value in rng.integers(500, 50_000, size=securities)]
    adv = np.exp(rng.uniform(np.log(5_000), np.log(2_000_000), size=securities)).astype(np.int64)
    alphas = rng.uniform(-0.05, 0.05, size=securities)
    universe = ["security_id,price,sector,adv_shares,lot_size,restricted,alpha,tcost_bps"]
    universe.extend(f"S{index:06d},{prices[index]},{sector_names[sector_of[index]]},{int(adv[index])},1,false,{alphas[index]:.6f},5" for index in range(securities))
    (root / "universe.csv").write_text("\n".join(universe) + "\n")

    portfolio_ids = [f"P{index:04d}" for index in range(portfolios)]
    (root / "portfolios.csv").write_text("\n".join(["portfolio_id,solve_order", *(f"{portfolio_id},{index}" for index, portfolio_id in enumerate(portfolio_ids))]) + "\n")

    mandates = ["portfolio_id,sector"]
    holdings = ["portfolio_id,security_id,quantity,avg_cost,acquired_on"]
    details = ["portfolio_id,name,state,st_tax_rate,lt_tax_rate,cash,nav,max_weight,max_turnover,max_adv_participation,min_trade_notional,cash_lb,cash_ub"]
    constraints = ["portfolio_id,name,label,params"]
    nav = Decimal(50_000_000)
    cap = max(Decimal("0.05"), (Decimal("1.8") / Decimal(held)).quantize(Decimal("0.0001")))  # twice the starting position weight, so no start sits above its cap
    for index, portfolio_id in enumerate(portfolio_ids):
        allowed = {(index + step) % groups for step in range(mandate_overlap + 1)}
        mandates.extend(f"{portfolio_id},{name}" for sector, name in enumerate(sector_names) if sector % groups in allowed)
        eligible = [security for security in range(securities) if sector_of[security] % groups in allowed]
        positions = sorted(rng.choice(len(eligible), size=min(held, len(eligible)), replace=False).tolist())
        per_position = nav * Decimal("0.9") / Decimal(len(positions))
        for position in positions:
            security = eligible[position]
            quantity = int(per_position / prices[security])
            cost = (prices[security] * Decimal(int(rng.integers(50, 100))) / 100).quantize(
                Decimal("0.01")
            )  # gains only: a harvestable loss is the tax term's wash-trade refusal, not a scheduling question
            holdings.append(f"{portfolio_id},S{security:06d},{quantity},{cost},2025-07-01T00:00:00Z")
        details.append(f"{portfolio_id},{portfolio_id} book,NY,0.40,0.20,{nav // 10},{nav},{cap},2,0.25,0,0,0.15")
        constraints.extend(f"{portfolio_id},{name}," for name in SHIPPED_CONSTRAINTS)
    (root / "mandates.csv").write_text("\n".join(mandates) + "\n")
    (root / "holdings.csv").write_text("\n".join(holdings) + "\n")
    (root / "details.csv").write_text("\n".join(details) + "\n")
    (root / "constraints.csv").write_text("\n".join(constraints) + "\n")


def config_body(dependencies: str) -> dict[str, object]:
    """The run config the benchmark executes: the shipped loaders with zero latency, the mandate rule, and the example's terms."""
    loader = lambda name: {"loader": {"name": name, "params": dict(NO_LATENCY)}}  # noqa: E731
    with_book = lambda name: {**loader(name), "depends_on": ["portfolios"]}  # noqa: E731
    return {
        "run": {"name": "book_benchmark", "as_of_date": AS_OF, "tags": {"purpose": "benchmark"}},
        "datasets": {
            "portfolios": loader("load_portfolios"),
            "holdings": with_book("load_holdings"),
            "universe": loader("load_universe"),
            "details": with_book("load_details"),
            "constraints": with_book("load_constraints"),
            "mandates": with_book("load_mandates"),
        },
        "rules": ["restrict_to_mandate"],
        "objective": {"terms": [{"name": "alpha"}, {"name": "tax_cost"}, {"name": "transaction_cost"}]},
        "solver": {"name": "CLARABEL", "options": {"max_iter": 200}},
        "sink": {"name": "orders_to_parquet"},
        "execution": {"on_error": "continue", "dependencies": dependencies},
    }


def execute(book: Path, out: Path, run_id: str, dependencies: str, workers: int) -> RunReport:
    """The real runner over the generated book, on a local cluster of ``workers`` processes."""
    config_path = out / "run.json"
    config_path.write_text(json.dumps(config_body(dependencies), indent=2) + "\n")
    resolved = resolve_config(load_run_config(config_path.read_text()))
    execution = ExecutionSettings(cluster="local", min_workers=workers, max_workers=workers, cluster_timeout_s=300.0)
    context = RunContext(
        io=IoContext(data_root=book, output_dir=out, run_id=run_id, clock=SystemClock()),
        execution=execution,
        git=read_git_info(Path.cwd()),
        config_path=str(config_path),
        settings={"benchmark": "run_book"},
    )
    return run(resolved, context)


def report_lines(report: RunReport, *, limit: int) -> list[str]:
    """What the benchmark says about the run: outcome counts, the derived schedule, and the manifest's own timeline."""
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
    lines.append("")
    lines.append(render_timeline(manifest.timing, limit=limit))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the book, run it, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portfolios", type=int, default=100)
    parser.add_argument("--securities", type=int, default=2_000)
    parser.add_argument("--sectors", type=int, default=40)
    parser.add_argument("--groups", type=int, default=10, help="mandate groups; 1 is the degenerate single-universe book")
    parser.add_argument("--mandate-overlap", type=int, default=0, help="how many neighbouring groups' sectors each mandate also covers, ring-wise; 0 keeps the groups disjoint")
    parser.add_argument("--held", type=int, default=50, help="positions per portfolio")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dependencies", default="overlap", choices=("overlap", "all", "none"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="directory for the book, config, and run output (default: a fresh temp directory, kept)")
    parser.add_argument("--limit", type=int, default=24, help="timeline row limit before collapsing to occupancy lanes")
    args = parser.parse_args(argv)
    if args.groups < 1 or args.groups > args.sectors:
        parser.error("--groups must be between 1 and --sectors")
    if not 0 <= args.mandate_overlap < args.groups:
        parser.error("--mandate-overlap must be between 0 and --groups - 1")
    out = Path(tempfile.mkdtemp(prefix="run-book-")) if args.out is None else Path(args.out)
    if args.out is not None and out.exists():
        shutil.rmtree(out)
    book = out / "book"
    started = datetime.now(tz=UTC)
    write_book(
        book,
        np.random.default_rng(int(args.seed)),
        portfolios=int(args.portfolios),
        securities=int(args.securities),
        sectors=int(args.sectors),
        groups=int(args.groups),
        held=int(args.held),
        mandate_overlap=int(args.mandate_overlap),
    )
    sys.stdout.write(
        f"book: {args.portfolios} portfolios x {args.securities} securities, {args.groups} mandate group(s) at overlap {args.mandate_overlap}, "
        f"under {out} (generated in {(datetime.now(tz=UTC) - started).total_seconds():.1f}s)\n"
    )
    run_id = f"book-p{args.portfolios}-g{args.groups}-m{args.mandate_overlap}-{args.dependencies}"
    report = execute(book, out, run_id, str(args.dependencies), int(args.workers))
    sys.stdout.write("\n".join(report_lines(report, limit=int(args.limit))))
    return 0 if report.exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
