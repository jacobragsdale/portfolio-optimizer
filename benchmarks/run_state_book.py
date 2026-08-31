"""Run a book of state-mandated municipal accounts through the real engine and report how the derived schedule behaved.

Where ``run_book.py`` partitions a book into *G* interchangeable mandate groups, this one has the shape a
municipal desk actually has: three sleeves over one universe of bonds, each bond issued by a state.

* **NY** accounts may only buy New York paper — in-state accounts hold their own state for the state
  income-tax exemption, and compliance freezes everything else.
* **CA** accounts, the same, for California.
* **National** accounts have no state restriction and buy anywhere, so they also carry a concentration
  band on every issuer state — the rule a single-state sleeve cannot have and does not need.

That is a *star*, not a partition: NY and CA share no buyable name, so no NY account ever waits on a CA
account; but every national account overlaps both sleeves, so it couples them. The derived critical path
is therefore about ``nationals + max(NY, CA)`` rather than the book's length, whatever order the accounts
solve in — and ``--dependencies all`` runs the same book as the strict line for comparison.

    uv run python benchmarks/run_state_book.py --portfolios 750 --securities 12000
    uv run python benchmarks/run_state_book.py --portfolios 750 --securities 12000 --dependencies all

``--territories N`` is the counterweight: it moves *N* of the universe's states into a triple-exempt
territory block (Puerto Rico, Guam) that *every* sleeve may buy. One shared block is enough to give NY
and CA accounts a name in common, and the star collapses back towards the line.
"""

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
from harness import SHIPPED_CONSTRAINTS, execute, report_lines

from portfolio_optimizer.engine.runner import RunReport

MARKET: tuple[tuple[str, float], ...] = (
    ("CA", 13.0),
    ("NY", 12.5),
    ("TX", 9.0),
    ("FL", 4.5),
    ("IL", 4.5),
    ("PA", 4.0),
    ("NJ", 3.5),
    ("MA", 3.5),
    ("WA", 3.0),
    ("OH", 3.0),
    ("MI", 2.5),
    ("VA", 2.5),
    ("GA", 2.5),
    ("NC", 2.5),
    ("MD", 2.0),
    ("CO", 2.0),
    ("AZ", 2.0),
    ("MN", 2.0),
    ("WI", 2.0),
    ("MO", 1.5),
    ("IN", 1.5),
    ("TN", 1.5),
    ("CT", 1.5),
    ("OR", 1.5),
    ("PR", 1.0),
    ("GU", 0.5),
)
"""Issuer states and their rough share of the municipal market; the last entries are the territories ``--territories`` opens to every sleeve."""


@dataclass(frozen=True, slots=True)
class Sleeve:
    """One kind of account: what it may buy, how it is taxed, and whether it carries a state band."""

    code: str
    mandate: tuple[str, ...]
    st_tax_rate: Decimal
    lt_tax_rate: Decimal
    state_band: Decimal | None


def sleeves(states: Sequence[str], territories: int) -> tuple[Sleeve, ...]:
    """The three sleeves over ``states``; the last ``territories`` of them are triple-exempt and open to all three."""
    shared = tuple(states[len(states) - territories :]) if territories else ()
    return (
        Sleeve("NY", ("NY", *shared), Decimal("0.4776"), Decimal("0.3176"), None),
        Sleeve("CA", ("CA", *shared), Decimal("0.5030"), Decimal("0.3430"), None),
        Sleeve("US", tuple(states), Decimal("0.4080"), Decimal("0.2380"), Decimal("0.20")),
    )


def write_book(root: Path, rng: np.random.Generator, *, portfolios: int, securities: int, states: int, held: int, territories: int) -> tuple[Sleeve, ...]:
    """Write the CSV tables of one municipal book: a universe of bonds by issuer state, and per account a mandate, in-mandate holdings, details, and typed constraint rows.

    Accounts are dealt round-robin across the sleeves, so the solve order interleaves them the way a
    book ordered by account number would; the schedule is derived from what each account may buy, not
    from how the ids happen to sort.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = [state for state, _ in MARKET[:states]]
    shares = np.array([weight for _, weight in MARKET[:states]], dtype=np.float64)
    counts = _allocate(securities, shares / shares.sum())
    state_of = np.repeat(np.arange(states), counts)

    prices = [Decimal(int(value)) / 100 for value in rng.integers(90_000, 112_000, size=securities)]  # a $1,000-face bond quoted between 90 and 112
    adv = np.exp(rng.uniform(np.log(100), np.log(20_000), size=securities)).astype(np.int64)  # bonds a day: the thin tail is where the chain actually binds
    alphas = rng.uniform(-0.02, 0.02, size=securities)  # relative-value scores, tighter than an equity book's
    universe = ["security_id,price,sector,adv_shares,lot_size,restricted,alpha,tcost_bps"]
    universe.extend(f"S{index:06d},{prices[index]},{names[state_of[index]]},{int(adv[index])},5,false,{alphas[index]:.6f},25" for index in range(securities))
    (root / "universe.csv").write_text("\n".join(universe) + "\n")

    book = sleeves(names, territories)
    eligible = {sleeve.code: [security for security in range(securities) if names[state_of[security]] in sleeve.mandate] for sleeve in book}
    portfolio_ids = [f"P{index:04d}" for index in range(portfolios)]
    (root / "portfolios.csv").write_text("\n".join(["portfolio_id,solve_order", *(f"{portfolio_id},{index}" for index, portfolio_id in enumerate(portfolio_ids))]) + "\n")

    mandates = ["portfolio_id,sector"]
    holdings = ["portfolio_id,security_id,quantity,avg_cost,acquired_on"]
    details = ["portfolio_id,name,state,st_tax_rate,lt_tax_rate,cash,nav,max_weight,max_turnover,max_adv_participation,min_trade_notional,cash_lb,cash_ub"]
    rows: list[tuple[str, str, str, str]] = [("portfolio_id", "name", "label", "params")]
    nav = Decimal(10_000_000)
    cap = max(Decimal("0.02"), (Decimal("1.8") / Decimal(held)).quantize(Decimal("0.0001")))  # twice the starting position weight, so no start sits above its cap
    for index, portfolio_id in enumerate(portfolio_ids):
        sleeve = book[index % len(book)]
        mandates.extend(f"{portfolio_id},{state}" for state in sleeve.mandate)
        pool = eligible[sleeve.code]
        positions = sorted(rng.choice(len(pool), size=min(held, len(pool)), replace=False).tolist())
        per_position = nav * Decimal("0.9") / Decimal(len(positions))
        for position in positions:
            security = pool[position]
            quantity = int(per_position / prices[security] / 5) * 5  # whole $5,000 pieces, the smallest a muni desk trades
            cost = (prices[security] * Decimal(int(rng.integers(80, 100))) / 100).quantize(
                Decimal("0.01")
            )  # gains only: a harvestable loss is the tax term's wash-trade refusal, not a scheduling question
            holdings.append(f"{portfolio_id},S{security:06d},{quantity},{cost},2025-07-01T00:00:00Z")
        details.append(f"{portfolio_id},{portfolio_id} {sleeve.code} muni,{sleeve.code},{sleeve.st_tax_rate},{sleeve.lt_tax_rate},{nav // 8},{nav},{cap},1,0.25,0,0,0.15")
        rows.extend(_constraint_rows(portfolio_id, sleeve))
    (root / "mandates.csv").write_text("\n".join(mandates) + "\n")
    (root / "holdings.csv").write_text("\n".join(holdings) + "\n")
    (root / "details.csv").write_text("\n".join(details) + "\n")
    with (root / "constraints.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return book


def _constraint_rows(portfolio_id: str, sleeve: Sleeve) -> list[tuple[str, str, str, str]]:
    """One account's constraint rows, in the shipped function convention the example uses.

    ``long_only`` and ``max_weight`` are what hold ``w`` inside the spec's own per-security bounds, and
    those are where the mandate lands: :func:`~portfolio_optimizer.rules.restrict_to_mandate` marks every
    out-of-state bond ``restricted``, which the build freezes at ``lb = ub = w0``. A national account adds
    a band per issuer state — the concentration rule an unrestricted muni account carries and a
    single-state one cannot. ``cumulative_adv_participation`` is the only row that reads the chain, and
    no row here declares a narrower scope, so each account couples through its whole buyable set, which
    is exactly its mandate's states.
    """
    rows = [(portfolio_id, name, "", "") for name in SHIPPED_CONSTRAINTS]
    if sleeve.state_band is not None:
        rows.extend((portfolio_id, "sector_bound", state.lower(), json.dumps({"sector": state, "upper": str(sleeve.state_band)})) for state in sleeve.mandate)
    return rows


def _allocate(total: int, shares: np.ndarray) -> np.ndarray:
    """Split ``total`` securities across the states by market share, largest-remainder, at least one each."""
    exact = shares * total
    counts = np.maximum(1, np.floor(exact).astype(np.int64))
    for position in np.argsort(-(exact - np.floor(exact)))[: total - int(counts.sum())]:
        counts[position] += 1
    return counts


def sleeve_lines(report: RunReport, book: Sequence[Sleeve]) -> list[str]:
    """Solve cost per sleeve: the accounts are dealt round-robin, so the id's number says which sleeve an account is in.

    Worth reporting separately because the sleeves do not cost the same. Every account solves over the
    whole universe — the mandate freezes names, it does not remove them — but a single-state account has
    a fraction of it buyable and no state bands, so it is the cheaper solve, and the mean over the book
    hides that.
    """
    spans: dict[str, list[float]] = {sleeve.code: [] for sleeve in book}
    for span in report.manifest.timing:
        if span.name == "solve" and span.portfolio_id is not None:
            spans[book[int(span.portfolio_id[1:]) % len(book)].code].append(span.duration_s)
    lines = ["solve seconds by sleeve:"]
    for code, seconds in spans.items():
        if seconds:
            ordered = sorted(seconds)
            lines.append(
                f"  {code:<4} n {len(ordered):>4}  mean {sum(ordered) / len(ordered):.2f}s  p50 {ordered[len(ordered) // 2]:.2f}s  p95 {ordered[int(len(ordered) * 0.95)]:.2f}s  max {ordered[-1]:.2f}s"
            )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the municipal book, run it, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portfolios", type=int, default=750, help="accounts, dealt round-robin across the NY, CA, and national sleeves")
    parser.add_argument("--securities", type=int, default=12_000, help="bonds in the universe; this is what sets the solve time")
    parser.add_argument("--states", type=int, default=len(MARKET), help=f"issuer states drawn from the market table, at most {len(MARKET)}")
    parser.add_argument("--territories", type=int, default=0, help="how many of the smallest states are triple-exempt and buyable by every sleeve; 1 is enough to couple NY to CA")
    parser.add_argument("--held", type=int, default=200, help="positions per account")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dependencies", default="overlap", choices=("overlap", "all", "none"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="directory for the book, config, and run output (default: a fresh temp directory, kept)")
    args = parser.parse_args(argv)
    if not 2 < args.states <= len(MARKET):
        parser.error(f"--states must be between 3 and {len(MARKET)}")
    if not 0 <= args.territories < args.states - 2:
        parser.error("--territories must leave NY and CA outside the shared block")
    out = Path(tempfile.mkdtemp(prefix="run-state-book-")) if args.out is None else Path(args.out)
    if args.out is not None and out.exists():
        shutil.rmtree(out)
    book = out / "book"
    started = datetime.now(tz=UTC)
    written = write_book(
        book, np.random.default_rng(int(args.seed)), portfolios=int(args.portfolios), securities=int(args.securities), states=int(args.states), held=int(args.held), territories=int(args.territories)
    )
    sys.stdout.write(
        f"book: {args.portfolios} accounts over {args.securities} bonds in {args.states} issuer states, "
        f"sleeves {', '.join(f'{sleeve.code}({len(sleeve.mandate)} states)' for sleeve in written)}, "
        f"under {out} (generated in {(datetime.now(tz=UTC) - started).total_seconds():.1f}s)\n"
    )
    run_id = f"state-p{args.portfolios}-s{args.securities}-t{args.territories}-{args.dependencies}"
    report = execute(book, out, run_id, str(args.dependencies), int(args.workers), "run_state_book")
    sys.stdout.write("\n".join([*sleeve_lines(report, written), "", *report_lines(report)]))
    return 0 if report.exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
