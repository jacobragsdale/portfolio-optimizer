# Tutorial: your first run

In this tutorial you will run the shipped example — a buy program and a sell program over one book of a
hundred accounts and three securities — inspect the orders each produces, prove a run is reproducible,
and re-verify a solution without the solver.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed and a Python 3.12+ interpreter available to it.
- A clone of this repository, with the working directory at its root.

## 1. Install the locked environment

```bash
uv sync --locked
```

This creates `.venv` with the exact dependency versions in `uv.lock`, including cvxpy and the Clarabel
solver. Nothing else needs to be installed, and nothing needs to be configured: every setting has a
default, and the default cluster is `inline` — every task runs in this process, one after another.
`.env.example` lists the settings with their defaults; copy it to `.env` and run with
`uv run --env-file .env ...` only when you want to change one, such as where runs are written (`./out`)
or which cluster the run provisions for itself.

## 2. Read the config, then check it before touching any data

Open `configs/example_buy.json`. That one file is the whole run: the side it trades, the data to load,
the rules to apply, the terms to minimize, the solve step and its solver, the verifier's tolerances, and
where the orders go. Each step is named by an ordinary Python function in `src/portfolio_optimizer/` —
the `load_holdings` loader, the `restrict_low_liquidity` rule, the `cvxpy` solve — and each objective
term by a typed kind, `linear`. The README walks through the file
[block by block](../README.md#the-run-config-block-by-block); keep it beside you for the rest of this
tutorial. One thing is not in the file: the instant the run is *as of*. That is an argument of `run`,
so one config runs every day under one hash.

Beside it is `configs/example_sell.json`: the same wiring with the run's name, `sides`, and one more
objective term changed. A run trades one side — a desk's buy program and its sell program are two runs
over one snapshot — and the sell program's extra term, the tax on what is sold, reads a vector the buy
program does not have.

Now ask the engine to check the buy program:

```bash
uv run portfolio-optimizer validate-config configs/example_buy.json
```

You should see `config ok` followed by one line per step and one per term:

```text
config ok (sha256 eb156c9bdfec): 1 rule(s), 2 term(s), dependencies overlap
  loader              portfolio_optimizer.loaders:load_portfolios
  ...
  rule                portfolio_optimizer.rules:restrict_low_liquidity
  build               portfolio_optimizer.engine.build:standard
  solve               portfolio_optimizer.solvers:cvxpy
  sink                portfolio_optimizer.sinks:orders_to_parquet
  term                alpha (Linear)
  term                transaction_cost (Linear)
```

The solve is itself a configured step, and this run uses the default: build a cvxpy problem from the
terms and each account's constraint rows and solve it with Clarabel. No constraint is listed, because
constraints are data. Open `examples/data/constraints.csv`: each account has one typed row per limit —
`cash_limit`, `turnover_limit`, `group_limit` — and one `participation_limit`, the kind that reads what
higher-priority portfolios in the run have already *traded* on the side the run trades. That row is why
P2, which can buy the same securities as P1, will wait for P1, and why every account in this book waits
for the ones ahead of it. `dependencies overlap` on the first line says a portfolio waits only for
portfolios it shares a tradable security with. `uv run portfolio-optimizer steps` lists every step and
every term and constraint kind this environment can name.

## 3. Run the buy program

```bash
uv run portfolio-optimizer run configs/example_buy.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

`--as-of` is the instant the run is as of — every loader receives it, and the build decides each lot's
holding period against it — and it must carry a time zone. `--data-root` is where the shipped loaders
read their tables; without it they look in the working directory. The run takes about half a minute,
nearly all of it the load stage. Expected output (the run id will differ, and there is a line per
account):

```text
run run-b3a1c61b6f8c: manifest out/run-b3a1c61b6f8c/manifest.json
  P1: solved, 2 order(s); binding: ub, adv/participation, adv/cumulative_participation
  P2: solved, 1 order(s); binding: lb, ub, sector_floor/group_limit, adv/cumulative_participation
  P3: solved, 2 order(s); binding: ub, adv/cumulative_participation
  ...
exit code 0
```

Each line names the constraints that *bind* — where the answer sits against a limit, as the verifier
found it — which is the first answer to "why did the solver stop here".

Above that summary, the log has one line per dataset, and they are the reason the run takes as long as
it does:

```text
dataset 'constraints' loaded: 530 row(s) in 1 batch(es), 2.28s
dataset 'holdings' loaded: 200 row(s) in 100 batch(es), 15.34s
dataset 'universe' loaded: 3 row(s) in 1 batch(es), 24.21s
```

None of that is file reading, and the seconds differ every run: the shipped loaders stand in for the
services a desk actually has — a custodian, a security master, an account master — and each waits a
draw from its own band, seeded on the run id, before answering from a CSV table. `load_universe` is a
security-master scan and takes tens of seconds; `load_holdings` is a custodian that answers one account
at a time. They overlap, so the whole stage costs about what its slowest input costs rather than the sum.

The universe is book-wide, so its loader was called once. `holdings` is not: the config asks for it with
`"scope": "per_portfolio", "batch_size": 1`, so the engine called its loader once per account — a
hundred calls, eight of them open at a time under the input's `max_in_flight`. That is how a source that
answers one account per call is wired up, and it is why a single account whose data is missing fails on
its own instead of stopping the book.

`details` is per-account too but asks for `"batch_size": 25`, which is the other shape: a source that
takes a *list* of accounts. The engine hands it twenty-five ids per call, so a hundred accounts is four
calls. Both numbers are in the manifest:

```bash
uv run python -c "import json, glob; print(*[(d['name'], d['batches']) for d in json.load(open(glob.glob('out/*/manifest.json')[0]))['datasets']], sep='\n')"
```

Expected: `('holdings', 100)` and `('details', 4)`, the rest 1 — every dataset's call count, recorded
beside its content hash.

The first two accounts are the ones to read by hand. P1 and P2 each have a NAV of $1,000,000, hold
3,000 A at 100 and 6,000 B at 50, and have $400,000 of cash to invest; each may trade at most a
quarter of a name's daily volume. C has the best expected return of the three and the worst liquidity:
100,000 shares a day at 10, so a portfolio can buy at most 25,000 shares — a 0.25 weight. A is worth
buying too; B has turned negative. The accounts differ in one thing that decides the answer: P1's style
caps any one name at 40%, P2's at 60%.

P1 has first pick. It buys the 25,000 shares of C its budget allows, then A up to its cap — 1,000
shares — and leaves the last $50,000 in cash rather than put it into B, since the cash floor is a
floor, not a target. P2 wants C too, but P1 has used the whole budget for the day, so its cash goes to
A instead: 3,000 shares, up to its wider cap. The output says so: `adv/participation` and
`adv/cumulative_participation` bind on P1 — C's budget is what stopped it — and
`adv/cumulative_participation` binds on P2, the budget P1 used. P3 is the exception that proves the
rule: its style allows 30% of a day's volume, so it finds 5,000 shares of C still inside its own budget
after P1's 25,000. Behind them the other accounts buy A up to their caps; about half have room.

That is the chain at work, and the manifest's `schedule` block records it: every account in this book
can buy all three securities, so every one of them waits for all the accounts ahead of it —
`"edges": 4950`, one component, a critical path of 100. A book whose accounts trade *different*
securities is where that graph opens up and the solves run side by side.

## 4. Look at the orders

```bash
uv run python -c "import pandas as pd, glob; print(pd.read_parquet(glob.glob('out/*/orders/orders.parquet')[0]).to_string())"
```

You should see 55 orders across 53 accounts, every one a `BUY`. P1's two are the ones to check: buy
1,000 A and buy 25,000 C — the hand-computed optimum, to the share. P2's one is buy 3,000 A.

## 5. Run the sell program over the same book

```bash
uv run portfolio-optimizer run configs/example_sell.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

Same data, same instant, the other side. Expected:

```text
  P1: solved, 1 order(s); binding: lb, sector_floor/group_limit
  P2: solved, 1 order(s); binding: lb, sector_floor/group_limit
  P3: solved, 0 order(s); binding: lb
```

Both P1 and P2 hold B at a cost of 60 against a price of 50. That loss is worth money: at P1's
long-term rate it is 4 cents of tax refund on every dollar sold, far more than the alpha given up, so
the sell program harvests it — down to where the account's `TECH` floor of 50% stops it, which is
2,000 shares each. A is at cost and worth holding, so nothing else moves. The output names the floor as
what bound. Down the book, accounts holding B at a gain keep it, because the tax on the gain outweighs
its small negative alpha, and from the thirty-fourth account on `adv/cumulative_participation` binds
instead: the accounts ahead have sold B's day.

The orders file for this run holds 38 orders across 37 accounts, every one a `SELL`, and its manifest
records `"sides": "sell"`. Nothing crosses between the two programs inside the engine: each is a pure
function of the snapshot it was given. A desk that runs the buy program after the sells settle runs it
over the new snapshot, and [how to run the buy program and the sell program](how-to-run-one-side.md)
covers what carries across, including keeping the buy program off what the sell program just harvested.

## 6. Prove the run is reproducible

Run the buy program a second time, as of the same instant, and compare the two manifests:

```bash
uv run portfolio-optimizer run configs/example_buy.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
uv run portfolio-optimizer diff-manifests out/<first-run-id>/manifest.json out/<second-run-id>/manifest.json
```

Expected: `no differences`. Every dataset hash, rule source hash, problem-spec hash, objective value,
and orders hash matched. If anything had drifted, the output would name the portfolio and the first
stage at which it diverged.

## 7. Re-verify a solution without the solver

```bash
uv run portfolio-optimizer verify --manifest out/<first-run-id>/manifest.json --portfolio P1
```

This reloads the persisted problem, solution, and chain state and recomputes every constraint the
solve applied and every term in plain numpy — cvxpy is never imported. Expected: a list of `ok` lines,
`[binding]` after the ones the answer sits against, and `VERIFIED P1`. Notice that the recomputed
objective equals the solver's to nine digits.

## What you accomplished

You ran the engine end to end on both sides, saw a cross-portfolio constraint change a second
portfolio's answer, confirmed two runs are byte-for-byte equivalent where it matters, and audited a
solution independently.

Next: [How to add a rule](how-to-add-a-rule.md) shows how to put your own business logic into that pipeline.
