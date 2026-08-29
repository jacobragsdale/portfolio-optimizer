# Explanation: the life of a run

This page walks through one run of the engine from the command line to the manifest, in the order the
code executes. It is the companion to [how the engine is built and why](explanation-architecture.md):
that page explains the design decisions; this one shows where each of them takes effect. Read it once
end to end and the module layout under `src/portfolio_optimizer/` will feel inevitable.

The short version: **read the config → prove every named function exists and has the right shape →
load data through loaders → join and validate → slice per portfolio → apply rules → build a
pure-numpy problem → solve with cvxpy → re-check the answer without cvxpy → round to whole shares →
publish the orders once → write a manifest.** Money is `Decimal` everywhere except inside the solver,
and there are exactly two conversion points. Every stage validates its own output, so a bad input
fails at the earliest stage that can detect it, with a message naming what is wrong.

## 1. Startup: nothing touches data yet

**Settings** (`settings.py`) come from three environment variables: `PORTFOLIO_OPTIMIZER_DATA_ROOT`,
`PORTFOLIO_OPTIMIZER_OUTPUT_DIR`, and `PORTFOLIO_OPTIMIZER_LOG_LEVEL`. There are no defaults, `.env`
files are read only when you pass `--env-file`, and an *unknown* `PORTFOLIO_OPTIMIZER_*` variable is an
error — a typo fails loudly instead of being ignored.

**The config** (`config/models.py`) is a strict pydantic model: unknown keys are errors, money is written
as strings (`"0.05"`) so it becomes exact `Decimal`, and `as_of` must carry a timezone. The validated
config is hashed on its canonical JSON form, so whitespace and the `$schema` pointer never change the
hash.

**Resolution** (`config/resolve.py`) is where the one convention is enforced. Every step in the config —
loaders, rules, objective terms, constraints, the sink — is a function name such as `"tracking_error"`
or `"mypkg.mod:my_rule"`. Before any data loads, the resolver:

- imports the function;
- checks its signature against the **contract for its kind**, by argument name and annotation — a rule
  must take `data: PortfolioData` and return `PortfolioData`, a term must take `x: DecisionVars,
  spec: ProblemSpec` and return `ObjectiveTerm`, and so on;
- validates the JSON `params` object against the function's own `Params` model;
- notes whether the function declares the optional context argument for its kind (`ctx` for rules,
  `chain` for terms and constraints), which makes it *chain-aware*;
- records three hashes — the function's source text, its whole module file, and its params.

It then checks the **execution mode against what the steps need**: `parallel` cannot run chain-aware
steps; `parallel_build_sequential_solve` cannot run rules that take `ctx`, because builds happen in
workers; and `on_error: continue` is refused alongside chain-aware steps, because a skipped portfolio
would silently change what later solves see. Every failure is collected and reported together.

`portfolio-optimizer validate-config` stops here and prints one line per resolved step.

## 2. Loading and assembly: validation, first layer

Loading is the slow part of a real run — API calls and database queries, not files — so `engine/load.py`
is asynchronous. It calls the **portfolios loader first**, validates the result against the `PORTFOLIOS`
schema, and sorts it by `solve_order`. That sorted tuple of ids is the order for the rest of the run,
and it is part of every other request, which is why this one loader cannot overlap with the rest. Then
**every dataset loader starts at once**, each called exactly once with a `LoadRequest` (dataset name,
portfolio ids, `as_of`, `data_root`, `run_id`, and a rate limiter): an `async def` loader runs as a task
on the event loop, a plain `def` loader in a worker thread so a blocking driver never stalls the loop.
Every outcome is collected — one failing dataset does not cancel the others — and all failures are
reported together. Every loaded frame gets an audit record — loader name, source hash, params hash,
row count, columns, how long it took, and a **content hash** that ignores row order, column order, and
index.

A source that answers one portfolio per call needs a rate limit to survive a large run, and sources
scale differently, so every input — the portfolio list and each dataset — can carry its own
`rate_limit`: a token bucket (`requests_per_second`, `burst`) and an in-flight bound (`max_in_flight`),
written inline and private to that input, or the name of a shared pool from the config's `rate_limits`
section for inputs that hit the same backend. The loader wraps each call in
`async with request.rate_limiter:` (or the `.sync` form from a thread); `fan_out` packages the
one-call-per-portfolio pattern with results in portfolio order.

The shipped loaders (`loaders.py`) are the only place I/O happens. The CSV loader deliberately reads
decimal and timestamp columns as strings, then `coerce_frame` turns them into `Decimal` exactly; a float
only ever becomes `Decimal(repr(value))`, the shortest round-tripping form. `csv_per_portfolio` is the
fan-out pattern with files in place of a client. The `constraints` dataset is different: its loader
returns a mapping of portfolio id to style-constraint object, not a frame.

`assemble` applies the config's **joins** in order — the example joins a `prices` dataset into
`universe`. Each join is defensive: key dtypes are aligned so a `str` key never silently joins to a
`string` key as `object`; overlapping non-key columns are refused rather than suffixed `_x`/`_y`; the
declared cardinality is enforced by pandas' `merge(validate=...)`; and `require_all_matched` uses the
merge indicator to report unmatched keys by example. After the joins, **every engine-known frame is
validated against its schema** — column set, dtype, nullability, bounds, unique key, and frame-level
invariants such as "target weights sum to one" — and all failures across all frames are reported at
once. Finally, `details` and `constraints` must have an entry for every portfolio.

Anything failing here raises `InputRejectedError` and the run exits with code 2. Nothing was solved.

## 3. Slice per portfolio: validation, second layer

`slice_portfolio` builds a `PortfolioData` bundle: this portfolio's `details` row typed into a
`PortfolioDetails` model, its holdings, the full universe, its benchmark's targets, the optional
covariance, and its style constraints typed into `StyleConstraints` (round-tripped through JSON so money
strings become `Decimal`).

`PortfolioData.__post_init__` (`domain/data.py`) holds the cross-frame invariants: `as_of` is UTC,
holdings contain only this portfolio, every held security is in the universe, targets belong to this
benchmark and are all in the universe, the covariance covers the universe, and every sector named in
`sector_bounds` exists. A failure here is a per-portfolio failure at stage `slice`, not a run-level
rejection; other portfolios proceed according to `on_error`.

## 4. Rules: validation, third layer

`engine/pipeline.py` runs each rule in config order. A rule is a pure function
`PortfolioData → PortfolioData`. The only way to return a modified bundle is `with_changes(...)`, which
constructs a new `PortfolioData` and therefore **re-runs every check from stage 3**. A rule cannot hand
the optimizer a broken bundle. Each rule gets an audit record of row counts before and after.

The shipped rules (`rules.py`) show the patterns: `restrict_low_liquidity` freezes names below an ADV
threshold, `add_zero_alpha` adds a column, `cap_single_name` tightens the style, and
`avoid_cross_portfolio_wash_sales` takes `ctx: SolveContext` and caps names that earlier portfolios
sold — a chain-aware rule, available in `sequential` mode only.

## 5. Build: `Decimal` becomes float64, once

`engine/build.py` sorts the universe by `security_id` and aligns everything to that order. In exact
`Decimal` it computes current weights (`shares × price / nav`), tax per dollar sold (gain fraction times
the short- or long-term rate, long-term when held longer than 365 days; losses come out negative),
transaction cost from `tcost_bps`, per-name bounds (restricted names frozen at their current weight,
optional `min_weight`/`max_weight` columns tightening the style cap), the sector indicator matrix, and
ADV capacity as a fraction of NAV. Only then does `to_float64` convert — refusing bools, non-Decimals,
and anything non-finite.

If there is a covariance, it is symmetrized, eigendecomposed under `threadpool_limits(1)` (multithreaded
BLAS can change the last bits and with them the spec hash), clipped to the PSD cone, and factored so the
risk term is a plain sum of squares. Too much clipping is a `BuildError`.

Any numeric universe column the schema does not declare is exported into `spec.columns` by name, so a
custom term can read `spec.column("my_signal")`.

The result is a `ProblemSpec` (`domain/results.py`): pure numpy, read-only arrays, no cvxpy. Its own
`__post_init__` checks shapes, sortedness, finiteness, `lb ≤ ub`, positive prices, and so on, and it
carries a content hash. Alongside it, `OrderInputs` keeps the *exact* `Decimal` prices and share counts
for stage 8.

## 6. Solve

`engine/solve.py` creates the decision variables `w`, `buy`, and `sell` — all fractions of NAV —
invokes each configured term and constraint function to obtain expressions, and hands them to the
adapter. `cvx/adapter.py` is the **only module that imports cvxpy**; terms are written against a dozen
typed atoms (`dot`, `matvec`, `sum_squares`, `at_most`, ...) so that the verifier can mirror each one in
numpy. The adapter checks that the solver is installed (there is no fallback), that the problem is
DCP-compliant, maps `time_limit_s` to the solver's own option, and returns the raw outcome.

Classification decides what the outcome means:

- **Optimal.** The buy/sell split is **canonicalized** to `buy = max(w − w0, 0)` and
  `sell = max(w0 − w, 0)`. With no term charging for trading, an interior-point solver can return a
  wash trade — buy 0.3 and sell 0.3 of the same name — that nets to the right weights but is not
  minimal.
- **Infeasible.** `InfeasibleError` carries an arithmetic diagnosis computed without another solve: do
  the upper bounds even sum to the required investment? Do the lower bounds exceed it? Can each sector
  reach its floor? Does moving the current weights inside their bounds already exceed `max_turnover`?
  Is a name that must trade out of ADV budget?
- **Unbounded.** Impossible with the shipped constraints, so it is reported as a bug in a custom step.
- **Anything else** is a `SolverFailureError` with the solver's own detail.

There is deliberately no path that returns the current portfolio as a fallback answer.

## 7. Verify: the second opinion

`engine/check.py` recomputes everything in numpy and never imports cvxpy (a test enforces this). For
each configured constraint it looks up a **numpy twin by qualified name** and computes the violation
vector; for each term it recomputes the value, sums them, and compares with the solver's reported
objective. It also checks finiteness, that the solution's `spec_hash` matches the spec it is being
checked against, and **complementarity** (`min(buy, sell) ≈ 0`), which is what proves the canonical
split held.

Tolerances come from the config's `post_solve` block and default to `1e-6` — about a hundred times
looser than the solver's own — so a pass is a genuine statement about the solution rather than a
restatement of the solver's convergence check. Custom steps with no twin are listed as `unverified` in
the manifest, and the objective comparison is skipped only when some term cannot be recomputed. A
failed check raises `VerificationError`; the portfolio fails at stage `solve`.

## 8. Orders: float64 becomes `Decimal`, once

`engine/orders.py` converts each weight delta to shares: **nearest share** (half-even), then down to a
lot multiple, then sells are clamped to what is held, then trades below `min_trade_notional` are
dropped. `notional = quantity × reference_price` is computed in `Decimal` from `OrderInputs`, never from
the float copies inside the spec, and the `ORDERS` schema's `notional_matches` invariant confirms it.
Each order also carries the spec hash, run id, `as_of`, the float target weight, and the unrounded share
count for audit.

`rounding_drift` rebuilds the executed weights from the orders and measures the worst deviation from the
solved weights. The tolerance is derived, not configured: one lot of the priciest name plus one
dust-filtered trade, both as fractions of NAV. Exceeding it raises `DriftError`.

## 9. The chain and the three execution modes

Portfolios in one run can depend on each other. The example's `cumulative_adv_participation` constraint
says "buy plus sell in each name may not exceed the ADV budget that earlier portfolios have not already
consumed."

The mechanism is small. `SolveContext` is an immutable tuple of completed results in solve order. Before
each solve, `derive_chain_state` sums the absolute shares ordered so far per security and projects them
into a `ChainState` aligned to *this* portfolio's securities. Terms and constraints that declare
`chain: ChainState` receive it; rules that declare `ctx: SolveContext` receive the full context. The
chain state's hash is recorded per portfolio in the manifest.

`engine/runner.py` dispatches on `execution.mode`; all three modes funnel through the same
`build_portfolio` → `finish_portfolio` functions.

- **`sequential`** — one loop in the main process. Rules see `ctx`, constraints see `chain`. Slowest,
  most expressive.
- **`parallel_build_sequential_solve`** (the example) — every portfolio's payload (its bundle, the
  config, the config hash) is submitted to a pool. Workers **re-resolve the config themselves** —
  function objects are never pickled, only their names — apply rules with no `ctx`, build the spec, and
  return pure numpy. The main process then solves in order with a live chain. Rules cannot be
  chain-aware; constraints can.
- **`parallel`** — the whole pipeline runs in the worker with an empty context. No chain-aware steps
  are permitted, and the resolver already refused them in stage 1.

Process workers use the `spawn` start method. The `thread` executor is allowed only where nothing is
solved in the worker, because cvxpy solves are not thread-safe; the config model rejects `parallel`
with `thread`. The scheduler submits every task and then **consumes results in configured solve order**,
so the worker count and the order in which workers finish can never change the output. Under
`fail_fast`, once any consumed outcome is a failure the remaining futures are cancelled and recorded as
`skipped`. A worker that dies — for instance on an unpicklable result — becomes a per-portfolio failure
at stage `worker`.

## 10. Failure semantics and exit codes

Every portfolio ends as a `PortfolioResult` or a `PortfolioFailure` naming its stage — `slice`, `build`,
`solve`, `worker`, or `skipped` — with the exception type and message. Under `fail_fast`, later
portfolios are `skipped`; under `continue`, each failure is isolated. The process exit code is **0**
when every portfolio solved, **1** when any failed, **2** when the inputs were rejected before anything
was solved, and **3** for infrastructure: a sink failure, invalid settings, an unreadable config.

## 11. Persist, publish, record

For each solved portfolio, the spec, solution, and chain state are written as `.npz` files under
`<output_dir>/<run_id>/{problem_specs,solutions,chain}/`. These are what `portfolio-optimizer verify`
reloads to re-check a solution without the solver stack.

The **sink** is called exactly once, with every solved portfolio's orders concatenated and sorted, and
only when at least one portfolio solved. The shipped sinks write Parquet (Decimals as Arrow decimals) or
CSV, atomically, via a temp file and rename. A sink failure is exit code 3, but the manifest is still
written.

The **manifest** (`engine/manifest.py`) records the run id, name, and timestamps; the git revision and
whether the tree was dirty; the Python, cvxpy, numpy, pandas, and solver versions; the resolved config
and its hash; the settings; every term and constraint with its params; every dataset's provenance and
content hash; and, per portfolio, the rule audits, spec hash, chain-input hash, solver statistics,
verification outcome, drift, and the orders' count, hash, and gross notional. The manifest is then
self-hashed, and `load_manifest` refuses one whose hash does not match its content.

`portfolio-optimizer diff-manifests` compares two manifests and names the **first stage at which they
diverge**: config, code, versions, datasets, and then per portfolio status → rules → spec → solve →
orders. "Did the data change, or did the solver?" is a one-command question. The
[reference page](reference-manifest.md) documents every field.

## The example, stage by stage

`configs/example_run.json` declares two portfolios over three securities, a `prices` dataset joined into
the universe, two rules, three terms (tracking error, tax cost, transaction cost), seven constraints,
the Clarabel solver, and `parallel_build_sequential_solve` with two workers.

P1 holds 500,000 shares each of A and B against an equal-weight target. C trades 100,000 shares a day at
10, and the style allows 25% participation, so P1 may buy at most 25,000 shares of C. The optimizer
sells 1,250 A and 2,500 B and buys 25,000 C — the hand-computable optimum, to the share. P2 holds only C
and would like to diversify, but by the time it solves the chain state says C's budget is spent, so P2
produces no orders. Run the config twice and `diff-manifests` reports `no differences`. The
[tutorial](tutorial-first-run.md) walks through exactly this.

## Where validation happens, in one list

1. **Settings** — unknown or missing environment variables.
2. **Config** — strict models; money as strings; timestamps with a zone.
3. **Resolver** — the function exists, its signature matches the contract, its params validate, and the
   execution mode is compatible with the steps.
4. **Loaders** — dtypes declared up front; exact `Decimal` coercion.
5. **Assembly** — join keys, cardinality, and coverage; then every frame schema; then details and
   constraints for every portfolio.
6. **`PortfolioData`** — cross-frame invariants, re-run after every rule.
7. **`ProblemSpec`** — shapes, finiteness, bound ordering, read-only arrays.
8. **Solver** — installed, DCP-compliant, status classified, infeasibility diagnosed.
9. **Verifier** — every shipped constraint and the objective, independently of cvxpy.
10. **Orders** — the `ORDERS` schema including `notional = quantity × price`, then the drift bound.
11. **Manifest** — self-hash checked on load.
