# Tutorial: your first run

In this tutorial you will run the shipped example — two portfolios over three securities — inspect the
orders it produces, prove the run is reproducible, and re-verify a solution without the solver.

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
`src/portfolio_optimizer/` — the `csv` loader, the `restrict_low_liquidity` rule, the `tracking_error`
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
same securities as P1, will wait for P1. The line above the steps says `dependencies overlap`: a
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

Expected output (the run id will differ):

```text
run run-4d9cb20e40db: manifest out/run-4d9cb20e40db/manifest.json
  P1: solved, 3 order(s)
  P2: solved, 0 order(s)
exit code 0
```

Above that summary, the log the run wrote to your terminal has one line per dataset. Two of them are
worth reading side by side:

```text
dataset 'universe' loaded: 3 row(s) in 1 batch(es), 0.01s
dataset 'details' loaded: 2 row(s) in 2 batch(es), 0.02s
```

The universe is book-wide, so its loader was called once. `details` is not: `examples/data/details/`
holds one file per account — `P1.csv` and `P2.csv` — and the config asks for that dataset with
`"scope": "per_portfolio", "batch_size": 1`, so the engine called its loader once per portfolio. That
is how a source that answers one account at a time — an account master, a custodian — is wired up, and
it is why a single account whose data is missing fails on its own instead of stopping the book.

P1 and P2 each hold $500,000 of A and $500,000 of B against an equal-weight target and may trade at
most a quarter of each name's daily volume. C's daily volume is 100,000 shares at 10, so a portfolio can
buy at most 25,000 shares (a 0.25 weight). P1 has first pick: the optimizer buys those 25,000 shares of
C and puts the remaining weight equally into A and B. P2 wants exactly the same trade, but P1 has already
used C's whole buying budget for the day, and with no cash allowed and A and B already balanced against
each other there is nothing else worth doing — so P2 correctly produces no orders. That is the chain at
work, and the manifest's `schedule` block records that P2 waited for P1 (`"edges": 1`).

## 5. Look at the orders

```bash
uv run --env-file .env python -c "import pandas as pd, glob; print(pd.read_parquet(glob.glob('out/*/orders/orders.parquet')[0]).to_string())"
```

You should see exactly three orders: sell 1,250 A, sell 2,500 B, buy 25,000 C — the hand-computed
optimum, to the share.

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
