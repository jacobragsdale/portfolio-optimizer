"""Time one portfolio through the shipped pipeline at a chosen book size, stage by stage.

Runs the functions a worker runs — bundle validation, rules, the spec build, the content hash, the
cvxpy expression tree, canonicalization, the solve, verification, orders, persistence — over a
synthetic book of *N* securities in *K* sectors, in one process, and prints a table of what each
stage cost and the size of what it produced. The point is where the time goes *inside* a task, which
is what decides whether the build path or the solver needs work; scheduling is measured elsewhere.

    uv run python benchmarks/profile_portfolio.py --securities 100000 --sectors 11
    uv run python benchmarks/profile_portfolio.py --securities 100000 --sectors 160 --solver OSQP
    uv run python benchmarks/profile_portfolio.py --securities 100000 --sides buy

The book is generated from a seed, so two runs of one command time the same problem. ``--sides``
selects the side profile; a one-sided run drops the example's terms that read the side it lacks
(``tax_cost`` under ``buy``), which the table's first row says, and a sell-only run lets the book
raise cash (``cash_bounds`` ``[0, 1]``), since it starts a tenth in cash and can only add to it.
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

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from portfolio_optimizer.config.models import StepSpec, load_run_config
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm
from portfolio_optimizer.cvx.sides import decision_variables
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDetails
from portfolio_optimizer.domain.results import ChainState, PortfolioResult, ProblemSpec
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.orders import rounding_drift, solution_to_orders
from portfolio_optimizer.engine.pipeline import apply_rules
from portfolio_optimizer.engine.solve import solve
from portfolio_optimizer.engine.tasks import BuildResult, step_refs
from portfolio_optimizer.solvers import interpret_constraints

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_run.json"
AS_OF = datetime(2026, 8, 28, tzinfo=UTC)
NAV = Decimal(1_000_000_000)
MB = 1e6
SHIPPED_CONSTRAINTS: tuple[str, ...] = ("long_only", "max_weight", "cash_bounds", "turnover_cap", "cumulative_adv_participation")
"""Every constraint the template ships that needs no params, as the shipped convention names them in the constraints frame; `sector_bound` is added once per sector, with its band."""


@dataclass(frozen=True, slots=True)
class Book:
    """A synthetic single-portfolio book: the frames and the account row the engine slices for one portfolio."""

    details: PortfolioDetails
    holdings: pd.DataFrame
    universe: pd.DataFrame
    constraints: pd.DataFrame
    extras: dict[str, pd.DataFrame]


def synthetic_book(rng: np.random.Generator, *, securities: int, sectors: int, held: int, sides: str) -> Book:
    """``securities`` names across ``sectors``, ``held`` of them owned, an alpha on every name, and a band on every sector."""
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
            "adv_shares": pd.Series(adv, dtype="Int64"),
            "lot_size": pd.Series([1] * securities, dtype="Int64"),
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
        cash_ub=Decimal(1) if sides == "sell" else Decimal(0),
    )
    names = [*SHIPPED_CONSTRAINTS, *["sector_bound"] * len(sector_names)]  # one row per sector: what the row block costs at K bands is part of what this measures
    labels = [*SHIPPED_CONSTRAINTS, *(name.lower() for name in sector_names)]
    params = [*[""] * len(SHIPPED_CONSTRAINTS), *(json.dumps({"sector": name, "upper": "0.5"}) for name in sector_names)]
    constraints = pd.DataFrame(
        {
            "portfolio_id": pd.Series(["P1"] * len(names), dtype="string"),
            "name": pd.Series(names, dtype="string"),
            "label": pd.Series(labels, dtype="string"),
            "params": pd.Series(params, dtype="string"),
        }
    )
    extras = {"buy_universe_parameters": pd.DataFrame({"name": pd.Series(["min_adv_shares"], dtype="string"), "value": pd.Series([Decimal(1000)], dtype="object")})}
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


SIDE_BLIND_TERMS: dict[str, frozenset[str]] = {"both": frozenset(), "buy": frozenset({"tax_cost"}), "sell": frozenset()}
"""The example's terms that read a side each profile lacks, dropped so the problem constructs."""


def _config(solver: str, sides: str, *, verbose: bool) -> ResolvedConfig:
    """The shipped example's rules, terms, and constraints with the solver and side replaced; its loaders and sink are never invoked."""
    body = json.loads(EXAMPLE_CONFIG.read_text())
    body["solver"] = {"name": solver, "options": {}, "verbose": verbose}
    body["sides"] = sides
    objective = dict(body["objective"])
    objective["terms"] = [term for term in objective["terms"] if _term_name(term) not in SIDE_BLIND_TERMS[sides]]
    body["objective"] = objective
    return resolve_config(load_run_config(json.dumps(body)))


def _term_name(term: object) -> str:
    return StepSpec.model_validate(term).name


def _sector_matrix_mb(spec: ProblemSpec) -> float:
    matrix = spec.sector_matrix
    return sum(int(np.asarray(part).nbytes) for part in (matrix.data, matrix.indices, matrix.indptr)) / MB


def _spec_mb(spec: ProblemSpec) -> float:
    names = ("w0", "price", "shares_held", "lot_size", "tax_per_dollar", "tcost_per_dollar", "lb", "ub", "adv_capacity")
    vectors = sum(int(getattr(spec, name).nbytes) for name in names)
    extras = sum(int(array.nbytes) for array in spec.columns.values()) + sum(int(array.nbytes) for array in spec.flags.values())
    return (vectors + extras) / MB + _sector_matrix_mb(spec)


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
    sides = str(args.sides)
    resolved = _config(str(args.solver), sides, verbose=bool(args.verbose))
    book = synthetic_book(rng, securities=securities, sectors=sectors, held=held, sides=sides)
    report = Report()
    with report.stage("validate bundle", f"{securities:,} securities in {sectors} sectors, {held:,} held; sides {sides}, terms {', '.join(step.name for step in resolved.terms)}") as row:
        data = PortfolioData(details=book.details, holdings=book.holdings, universe=book.universe, constraints=book.constraints, extras=book.extras, as_of_date=AS_OF)
    with report.stage("rules", ", ".join(step.name for step in resolved.rules)):
        ruled, _ = apply_rules(data, resolved.rules)
    with report.stage("spec build") as row:
        output = build_problem_spec(ruled)
    spec = output.spec
    row.note = (
        f"spec arrays {_spec_mb(spec):.1f} MB, of which sector matrix {_sector_matrix_mb(spec):.1f} MB ({spec.sector_matrix.nnz:,} nonzeros); {int(resolved.profile.tradable(spec).sum()):,} tradable"
    )
    with report.stage("content hash"):
        spec.content_hash()
    chain = ChainState.empty(spec.security_ids)
    with report.stage("expression tree") as row:
        x = decision_variables(resolved.profile.sides, spec.w0)
        terms = [step.invoke(x=x, spec=spec, chain=chain) for step in resolved.terms]
        sets = [step.invoke(x=x, spec=spec, chain=chain) for step, _ in interpret_constraints(book.constraints)]
    constraint_sets = [item for item in sets if isinstance(item, ConstraintSet)]
    expressions = [item.expression for item in terms if isinstance(item, ObjectiveTerm)]
    flat = [constraint for group in constraint_sets for constraint in group.constraints]
    row.note = f"{len(expressions)} terms, {len(flat)} constraint objects"
    with report.stage("cvxpy Problem + is_dcp"):
        objective = expressions[0]
        for expression in expressions[1:]:
            objective = objective + expression
        problem = cp.Problem(cp.Minimize(objective), flat)
        if not problem.is_dcp():
            msg = "the benchmark problem is not DCP"
            raise RuntimeError(msg)
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
        checked = verify(spec, solution, chain, step_refs(resolved.terms), solution.constraints, profile=resolved.profile)
    row.note = f"passed {checked.passed}, max violation {checked.max_violation:.2e}, objective gap {checked.objective_gap:.2e}"
    with report.stage("orders") as row:
        orders = solution_to_orders(spec, solution, output.order_inputs, run_id="benchmark")
    row.note = f"{len(orders):,} orders"
    with report.stage("rounding drift"):
        rounding_drift(spec, solution, orders, output.order_inputs)
    with tempfile.TemporaryDirectory() as directory, report.stage("persist spec + solution (.npz)") as row:
        spec_path, solution_path = Path(directory) / "spec.npz", Path(directory) / "solution.npz"
        spec.to_npz(spec_path)
        solution.to_npz(solution_path)
        row.note = f"spec {spec_path.stat().st_size / MB:.1f} MB, solution {solution_path.stat().st_size / MB:.1f} MB on disk"
    with report.stage("pickle sizes") as row:
        built = BuildResult(
            portfolio_id=PortfolioId(spec.portfolio_id), spec=spec, order_inputs=output.order_inputs, rule_audit=(), solve_order=Decimal(0), tradable=(), constraints=book.constraints, extras={}
        )
        result = PortfolioResult(
            portfolio_id=spec.portfolio_id,
            spec=spec,
            solution=solution,
            report=checked,
            orders=orders,
            rule_audit=(),
            chain_state=chain,
            drift=rounding_drift(spec, solution, orders, output.order_inputs),
            contribution=resolved.profile.contribution(spec.portfolio_id, orders),
        )
        row.note = f"BuildResult {len(pickle.dumps(built)) / MB:.1f} MB (stays on the worker), PortfolioResult {len(pickle.dumps(result)) / MB:.1f} MB (returns to the client)"
    return report


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--securities", type=int, default=100_000)
    parser.add_argument("--sectors", type=int, default=11)
    parser.add_argument("--held", type=int, default=25_000)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--sides", default="both", choices=sorted(SIDE_BLIND_TERMS), help="the side profile to solve under")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="let the solver print its iteration log")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one portfolio and print the table."""
    args = _parse(argv)
    report = profile(args)
    total = sum(row.seconds for row in report.rows)
    sys.stdout.write(f"{report.render()}\n\nsum of stages {total:.1f}s; solver {args.solver}, sides {args.sides}, seed {args.seed}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
