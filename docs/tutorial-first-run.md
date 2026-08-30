# Tutorial: your first run

In this tutorial you will run the shipped example — a hundred accounts over three securities — inspect
the orders it produces, prove the run is reproducible, and re-verify a solution without the solver.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed and a Python 3.12+ interpreter available to it.
- A clone of this repository, with the working directory at its root.

## 1. Install the locked environment

```bash
uv sync --locked
```

This creates `.venv` with the exact dependency versions in `uv.lock`, including cvxpy and the Clarabel
solver. Nothing else needs to be installed.

## 2. Point the engine at the example data

Every run needs seven environment variables. The example file sets them to the shipped data:

```bash
cp .env.example .env
```

`.env` now says where data is read from (`examples/data`), where runs are written (`./out`), how
loudly to log, and the cluster the run provisions for itself — `local`, two worker processes on this
machine, torn down when the run ends. There are no defaults: a missing variable stops the run before it
starts.

## 3. Read the config, then check it before touching any data

Open `configs/example_run.json`. That one file is the whole run: the data to load, the steps that
combine it, the rules to apply, the terms to minimize, the constraints to hold, the solver, the
verifier's tolerances, and where the orders go. Each block is named by an ordinary Python function in
`src/portfolio_optimizer/` — the `load_holdings` loader, the `restrict_low_liquidity` rule, the `alpha`
term. The README walks through the file
[block by block](../README.md#the-run-config-block-by-block); keep it beside you for the rest of this
tutorial.

Now ask the engine to check it:

```bash
uv run --env-file .env portfolio-optimizer validate-config configs/example_run.json
```

You should see `config ok` followed by one line per step. Notice the line

```text
  constraint          portfolio_optimizer.terms:cumulative_adv_participation [chain]
```

The `[chain]` marker means this constraint reads what higher-priority portfolios in the run have already
*traded* — the example is a two-sided run, which couples through buys, so it is why P2, which can buy the
same securities as P1, will wait for P1, and why every account in this book waits for the ones ahead of
it. The line above the steps says `dependencies overlap`: a
portfolio waits only for portfolios it shares a tradable security with. Notice also the line

```text
  solve               portfolio_optimizer.solvers:cvxpy
```

The solve is itself a configured step, and this run uses the default: build a cvxpy problem from the
terms and constraints and solve it with Clarabel.

## 4. Run it

```bash
uv run --env-file .env portfolio-optimizer run configs/example_run.json
```

It takes about half a minute, nearly all of it the load stage. Expected output (the run id will
differ, and there is a line per account):

```text
run run-e204cb5a1ea9: manifest out/run-e204cb5a1ea9/manifest.json
  P1: solved, 3 order(s)
  P2: solved, 0 order(s)
  P3: solved, 3 order(s)
  ...
  P100: solved, 0 order(s)
exit code 0
```

Above that summary, the log has one line per dataset, and they are the reason the run takes as long as
it does:

```text
dataset 'constraints' loaded: 634 row(s) in 1 batch(es), 3.11s
dataset 'universe' loaded: 3 row(s) in 1 batch(es), 13.65s
dataset 'holdings' loaded: 200 row(s) in 100 batch(es), 16.36s
```

None of that is file reading, and the seconds differ every run: the shipped loaders stand in for the
services a desk actually has — a custodian, a security master, an account master — and each waits a
draw from its own band, seeded on the run id, before answering from a CSV table. `load_universe` is a security-master scan and takes tens of seconds;
`load_holdings` is a custodian that answers one account at a time. They overlap, so the whole stage
costs about what its slowest input costs rather than the sum.

The universe is book-wide, so its loader was called once. `holdings` is not: the config asks for it with
`"scope": "per_portfolio", "batch_size": 1`, so the engine called its loader once per account — a
hundred calls, paced by the `custodian` rate-limit pool the input names. That is how a source that
answers one account per call is wired up, and it is why a single account whose data is missing fails on
its own instead of stopping the book.

`details` is per-account too but asks for `"batch_size": 25`, which is the other shape: a source that
takes a *list* of accounts. The engine hands it twenty-five ids per call, so a hundred accounts is four
calls. Both numbers are in the manifest:

```bash
uv run --env-file .env python -c "import json, glob; print(*[(d['name'], d['batches']) for d in json.load(open(glob.glob('out/*/manifest.json')[0]))['datasets']], sep='\n')"
```

Expected: `('holdings', 100)` and `('details', 4)`, the rest 1 — every dataset's call count, recorded
beside its content hash.

The first two accounts are the ones to read by hand. P1 and P2 each hold $500,000 of A and $500,000 of
B, and each may trade at most a quarter of a name's daily volume. C has the best expected return of the three and the worst liquidity: 100,000 shares a day
at 10, so a portfolio can buy at most 25,000 shares — a 0.25 weight. The two accounts differ in two
ways that decide the answer. P1's style caps any one name at 40%, and it holds two at 50%, so it *must*
trim; P2's cap is 60%, so it must do nothing. And both hold B at a fifth of unrealized gain, which P1
would realize at its long-term rate and P2 at its short-term one.

P1 has first pick. It raises the quarter of NAV that C's budget allows, selling A before B because A is
at cost and selling B would realize a gain, then buys the 25,000 shares of C. P2 wants C too, but P1 has
used the whole budget for the day; nothing else is worth doing at its short-term rate, so P2 correctly
produces no orders. The other ninety-eight accounts are the same story with different limits: about half
find something worth trading.

That is the chain at work, and the manifest's `schedule` block records it: every account in this book
can trade all three securities, so every one of them waits for all the accounts ahead of it —
`"edges": 4950`, one component, a critical path of 100. A book whose accounts trade *different*
securities is where that graph opens up and the solves run side by side.

## 5. Look at the orders

```bash
uv run --env-file .env python -c "import pandas as pd, glob; print(pd.read_parquet(glob.glob('out/*/orders/orders.parquet')[0]).to_string())"
```

You should see 98 orders across 48 accounts. P1's three are the ones to check: sell 1,500 A, sell
2,000 B, buy 25,000 C — the hand-computed optimum, to the share.

## 6. Prove the run is reproducible

Run it a second time and compare the two manifests:

```bash
uv run --env-file .env portfolio-optimizer run configs/example_run.json
uv run --env-file .env portfolio-optimizer diff-manifests out/<first-run-id>/manifest.json out/<second-run-id>/manifest.json
```

Expected: `no differences`. Every dataset hash, rule source hash, problem-spec hash, objective value,
and orders hash matched. If anything had drifted, the output would name the portfolio and the first
stage at which it diverged.

## 7. Re-verify a solution without the solver

```bash
uv run --env-file .env portfolio-optimizer verify --manifest out/<first-run-id>/manifest.json --portfolio P1
```

This reloads the persisted problem, solution, and chain state and recomputes every constraint and the
objective in plain numpy — cvxpy is never imported. Expected: a list of `ok` lines and
`VERIFIED P1`. Notice that the recomputed objective equals the solver's to nine digits.

## What you accomplished

You ran the engine end to end, saw a cross-portfolio constraint change a second portfolio's answer,
confirmed two runs are byte-for-byte equivalent where it matters, and audited a solution independently.

Next: [How to add a rule](how-to-add-a-rule.md) shows how to put your own business logic into that pipeline.
