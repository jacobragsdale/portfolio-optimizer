"""Time one portfolio through the shipped pipeline at a chosen book size, stage by stage.

Runs the functions a worker runs — bundle validation, rules, the spec build, the content hash, the
cvxpy expression tree, canonicalization, the solve, verification, orders, persistence — over a
synthetic book of *N* securities in *K* sectors, in one process, and prints a table of what each
stage cost and the size of what it produced. The point is where the time goes *inside* a task, which
is what decides whether the build path or the solver needs work; scheduling is measured elsewhere.

    uv run python benchmarks/profile_portfolio.py --securities 100000 --sectors 11
    uv run python benchmarks/profile_portfolio.py --securities 100000 --sectors 160 --solver OSQP
    uv run python benchmarks/profile_portfolio.py --securities 100000 --order-flow outflow

The book is generated from a seed, so two runs of one command time the same problem. ``--order-flow``
selects the order flow: the inflow invests the tenth of NAV the book starts with in cash under
the example's inflow terms; the outflow, under its terms with ``tax_cost`` added, lets the
book raise cash (a cash cap of one), since it can only add to it; the rebalance, under the inflow's
terms, may do either and gets the same cap.
"""

import argparse
import json
import pickle
import resource
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from harness import CONSTRAINT_COLUMNS, OBJECTIVE, SHIPPED_CONSTRAINTS, constraint_row
from scipy.sparse import issparse

from portfolio_optimizer.config.models import load_run_config
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.cvx.adapter import build_problem
from portfolio_optimizer.cvx.order_flow import decision_variables, identity_constraints
from portfolio_optimizer.domain.constraints import parse_constraints
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDetails
from portfolio_optimizer.domain.frames import empty_frame
from portfolio_optimizer.domain.results import VECTOR_FIELDS, ChainState, PortfolioResult, ProblemSpec
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.build import order_inputs, standard
from portfolio_optimizer.engine.check import constraints_of, verify
from portfolio_optimizer.engine.orders import executed_solution, rounding_drift, solution_to_orders
from portfolio_optimizer.engine.pipeline import apply_rules
from portfolio_optimizer.engine.solve import solve
from portfolio_optimizer.engine.tasks import BuildResult
from portfolio_optimizer.loaders import TRADES

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_inflow.json"
AS_OF = datetime(2026, 8, 28, tzinfo=UTC)
NAV = Decimal(1_000_000_000)
MB = 1e6


@dataclass(frozen=True, slots=True)
class Book:
    """A synthetic single-portfolio book: the frames and the account row the engine slices for one portfolio."""

    details: PortfolioDetails
    holdings: pd.DataFrame
    universe: pd.DataFrame
    constraints: pd.DataFrame
    extras: dict[str, pd.DataFrame]


def synthetic_book(rng: np.random.Generator, *, securities: int, sectors: int, held: int, order_flow: str) -> Book:
    """``securities`` names across ``sectors``, ``held`` of them owned, an alpha on every name, and a cap on every sector."""
    ids = [f"S{index:07d}" for index in range(securities)]
    prices = [Decimal(int(value)) / 100 for value in rng.integers(500, 50_000, size=securities)]
    adv = [int(value) for value in np.exp(rng.uniform(np.log(500), np.log(10_000_000), size=securities)).astype(np.int64)]
    sector_names = [f"SEC{index:03d}" for index in range(sectors)]
    sector_of = rng.integers(0, sectors, size=securities)
    universe = pd.DataFrame(
        {
            "security_id": pd.Series(ids, dtype="string"),
            "price": pd.Series(prices, dtype="object"),
            "sector": pd.Series([sector_names[int(index)] for index in sector_of], dtype="string"),
            "adv_quantity": pd.Series(adv, dtype="Int64"),
            "increment": pd.Series([1] * securities, dtype="Int64"),
            "restricted": pd.Series([False] * securities, dtype="bool"),
            "tcost_bps": pd.Series([Decimal(5)] * securities, dtype="object"),
            "alpha": pd.Series(rng.uniform(-0.05, 0.05, size=securities), dtype="Float64"),
        }
    )
    held_indexes = np.sort(rng.choice(securities, size=held, replace=False))
    per_position = NAV * Decimal("0.9") / Decimal(held)
    holdings = pd.DataFrame(
        {
            "portfolio_id": pd.Series(["P1"] * held, dtype="string"),
            "security_id": pd.Series([ids[int(index)] for index in held_indexes], dtype="string"),
            "lot_id": pd.Series(["L1"] * held, dtype="string"),
            "quantity": pd.Series([int(per_position / prices[int(index)]) for index in held_indexes], dtype="Int64"),
            "avg_cost": pd.Series(
                [(prices[int(index)] * Decimal(int(factor)) / 100).quantize(Decimal("0.01")) for index, factor in zip(held_indexes, rng.integers(50, 100, size=held), strict=True)], dtype="object"
            ),  # gains only: a harvestable loss makes sell-and-rebuy optimal, which is a modelling question, not a timing one
            "acquired_on": pd.Series([AS_OF - timedelta(days=int(days)) for days in rng.integers(1, 1000, size=held)], dtype="datetime64[ns, UTC]"),
        }
    )
    details = PortfolioDetails(
        portfolio_id="P1",
        name="Benchmark Book",
        state="NY",
        st_tax_rate=Decimal("0.40"),
        lt_tax_rate=Decimal("0.20"),
        cash=NAV * Decimal("0.1"),
        nav=NAV,
        max_weight=Decimal("0.05"),
        max_turnover=Decimal(2),
        max_adv_participation=Decimal("0.25"),
        min_trade_notional=Decimal(0),
        cash_lb=Decimal(0),
        cash_ub=Decimal(0) if order_flow == "inflow" else Decimal(1),
    )
    rows = [constraint_row("P1", kind, label, params) for kind, label, params in SHIPPED_CONSTRAINTS]
    rows.append(
        constraint_row("P1", "group_limit", "sector_caps", {"direction": "<=", "column": "sector", "bounds": dict.fromkeys(sector_names, "0.5")})
    )  # K bands in one row: what the block costs at K is part of what this measures
    constraints = pd.DataFrame({column: pd.Series([row[index] for row in rows], dtype="string") for index, column in enumerate(CONSTRAINT_COLUMNS)})
    extras = {"buy_universe_parameters": pd.DataFrame({"name": pd.Series(["min_adv_quantity"], dtype="string"), "value": pd.Series([Decimal(1000)], dtype="object")}), "trades": empty_frame(TRADES)}
    return Book(details=details, holdings=holdings, universe=universe, constraints=constraints, extras=extras)


@dataclass(slots=True)
class Row:
    """One timed stage."""

    stage: str
    seconds: float
    peak_rss_mb: float
    note: str = ""


@dataclass(slots=True)
class Report:
    """The stages in order, with a note each stage may fill in after it ran."""

    rows: list[Row] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, note: str = "") -> Iterator[Row]:
        """Time the block and record the peak RSS after it; the yielded row's ``note`` may be set inside."""
        row = Row(stage=name, seconds=0.0, peak_rss_mb=0.0, note=note)
        started = time.perf_counter()
        try:
            yield row
        finally:
            row.seconds = time.perf_counter() - started
            row.peak_rss_mb = _peak_rss_mb()
            self.rows.append(row)

    def render(self) -> str:
        """A markdown table."""
        lines = ["| Stage | Seconds | Peak RSS (MB) | Note |", "|---|---:|---:|---|"]
        lines.extend(f"| {row.stage} | {row.seconds:.3f} | {row.peak_rss_mb:,.0f} | {row.note} |" for row in self.rows)
        return "\n".join(lines)


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / MB if sys.platform == "darwin" else peak * 1024 / MB


def _config(solver: str, order_flow: str, *, verbose: bool) -> ResolvedConfig:
    """The shipped example's rules with the solver and order flow replaced and the order flow's own terms; its loaders and sink are never invoked."""
    body = json.loads(EXAMPLE_CONFIG.read_text())
    body["solve"] = {"name": "cvxpy", "params": {"solver": solver, "verbose": verbose}}
    body["order_flow"] = order_flow
    body["objective"] = OBJECTIVE[order_flow]
    return resolve_config(load_run_config(json.dumps(body)))


def _groups_mb(spec: ProblemSpec) -> float:
    return sum(int(np.asarray(part).nbytes) for group in spec.groups.values() for part in (group.matrix.data, group.matrix.indices, group.matrix.indptr)) / MB


def _spec_mb(spec: ProblemSpec) -> float:
    vectors = sum(int(getattr(spec, name).nbytes) for name in VECTOR_FIELDS)
    extras = sum(int(array.nbytes) for array in spec.columns.values()) + sum(int(array.nbytes) for array in spec.flags.values())
    return (vectors + extras) / MB + _groups_mb(spec)


def _matrix_note(name: str, matrix: object) -> str:
    if issparse(matrix):
        return f"{name} {matrix.shape} nnz {matrix.nnz:,}"  # ty: ignore[unresolved-attribute]  # narrowed by issparse
    if isinstance(matrix, np.ndarray):
        return f"{name} {matrix.shape} dense"
    return f"{name} absent"


def profile(args: argparse.Namespace) -> Report:  # one straight line through the pipeline, timed stage by stage
    """Run every stage once over a synthetic book and return the timings."""
    rng = np.random.default_rng(int(args.seed))
    securities, sectors, held = int(args.securities), int(args.sectors), int(args.held)
    order_flow = str(args.order_flow)
    resolved = _config(str(args.solver), order_flow, verbose=bool(args.verbose))
    book = synthetic_book(rng, securities=securities, sectors=sectors, held=held, order_flow=order_flow)
    report = Report()
    with report.stage("validate bundle", f"{securities:,} securities in {sectors} sectors, {held:,} held; order flow {order_flow}, terms {', '.join(term.name for term in resolved.terms)}"):
        data = PortfolioData(details=book.details, holdings=book.holdings, universe=book.universe, constraints=book.constraints, extras=book.extras, as_of_date=AS_OF)
    with report.stage("rules", ", ".join(step.name for step in resolved.rules)):
        ruled, _ = apply_rules(data, resolved.rules)
    with report.stage("spec build") as row:
        spec = standard(ruled)
        inputs = order_inputs(ruled, spec)
    sector = spec.group("sector")
    row.note = f"spec arrays {_spec_mb(spec):.1f} MB, of which groupings {_groups_mb(spec):.1f} MB ({sector.matrix.nnz:,} nonzeros); {int(resolved.profile.tradable(spec).sum()):,} tradable"
    with report.stage("content hash"):
        spec.content_hash()
    chain = ChainState.empty(spec.security_ids)
    parsed = parse_constraints(book.constraints)
    typed = () if parsed is None else parsed.typed
    with report.stage("expression tree") as row:
        x = decision_variables(resolved.profile.order_flow, spec)
        terms = [term.to_cvxpy(x, spec, chain) for term in resolved.terms]
        sets = [identity_constraints(resolved.profile.order_flow, x, spec), *(constraint.to_cvxpy(x, spec, chain) for constraint in typed)]
    flat = [constraint for group in sets for constraint in group.constraints]
    row.note = f"{len(terms)} terms, {len(flat)} constraint objects"
    with report.stage("cvxpy Problem + is_dcp"):
        problem = build_problem(terms, sets)
    with report.stage("canonicalization (get_problem_data)") as row:
        solver_data, solving_chain, inverse = problem.get_problem_data(str(args.solver), solver_opts={})
    if isinstance(solver_data, dict):
        row.note = "; ".join(_matrix_note(name, solver_data.get(name)) for name in ("P", "A", "F", "G"))
    with report.stage("solve via data") as row:
        raw = solving_chain.solve_via_data(problem, solver_data, warm_start=False, verbose=bool(args.verbose), solver_opts={})
    with report.stage("unpack results"):
        problem.unpack_results(raw, solving_chain, inverse)
    stats = problem.solver_stats
    row.note = (
        f"status {problem.status}; solver-reported {float(stats.solve_time):.3f}s, {stats.num_iters} iterations" if stats is not None and stats.solve_time is not None else f"status {problem.status}"
    )
    with report.stage("engine solve() end to end", "tree + canonicalization + solve + classify, the way solve_task does it"):
        solution = solve(spec, chain, resolved, book.constraints)
    with report.stage("verify") as row:
        checked = verify(spec, solution, chain, resolved.terms, constraints_of(solution), profile=resolved.profile)
    row.note = f"passed {checked.passed}, max violation {checked.max_violation:.2e}, objective gap {checked.objective_gap:.2e}; binding {', '.join(checked.active) or 'nothing'}"
    with report.stage("orders") as row:
        orders = solution_to_orders(spec, solution, inputs, run_id="benchmark")
    row.note = f"{len(orders):,} orders"
    with report.stage("rounding drift"):
        drift = rounding_drift(spec, solution, orders, inputs)
    with report.stage("verify executed orders") as row:
        executed = executed_solution(spec, solution, orders, resolved.profile)
        executed_report = verify(spec, executed, chain, resolved.terms, constraints_of(solution), profile=resolved.profile)
    row.note = f"passed {executed_report.passed}, max violation {executed_report.max_violation:.2e}"
    with tempfile.TemporaryDirectory() as directory, report.stage("persist spec + solution (.npz)") as row:
        spec_path, solution_path = Path(directory) / "spec.npz", Path(directory) / "solution.npz"
        spec.to_npz(spec_path)
        solution.to_npz(solution_path)
        row.note = f"spec {spec_path.stat().st_size / MB:.1f} MB, solution {solution_path.stat().st_size / MB:.1f} MB on disk"
    with report.stage("pickle sizes") as row:
        built = BuildResult(
            portfolio_id=PortfolioId(spec.portfolio_id), spec=spec, order_inputs=inputs, rule_audit=(), solve_order=Decimal(0), tradable=(), consumes=(), constraints=book.constraints, extras={}
        )
        result = PortfolioResult(
            portfolio_id=spec.portfolio_id,
            spec=spec,
            solution=solution,
            report=checked,
            orders=orders,
            rule_audit=(),
            chain_state=chain,
            drift=drift,
            contribution=resolved.profile.contribution(spec.portfolio_id, orders),
            executed=executed,
            executed_report=executed_report,
        )
        row.note = f"BuildResult {len(pickle.dumps(built)) / MB:.1f} MB (stays on the worker), PortfolioResult {len(pickle.dumps(result)) / MB:.1f} MB (returns to the client)"
    return report


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--securities", type=int, default=100_000)
    parser.add_argument("--sectors", type=int, default=11)
    parser.add_argument("--held", type=int, default=25_000)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--order-flow", default="inflow", choices=sorted(OBJECTIVE), help="the order flow to time: the inflow, the outflow, or the rebalance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="let the solver print its iteration log")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one portfolio and print the table."""
    args = _parse(argv)
    report = profile(args)
    total = sum(row.seconds for row in report.rows)
    sys.stdout.write(f"{report.render()}\n\nsum of stages {total:.1f}s; solver {args.solver}, order flow {args.order_flow}, seed {args.seed}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
