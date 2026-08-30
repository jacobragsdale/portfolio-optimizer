# Explanation: the life of a run

This page walks through one run of the engine from the command line to the manifest, in the order the
code executes. It is the companion to [how the engine is built and why](explanation-architecture.md):
that page explains the design decisions; this one shows where each of them takes effect. For the same
machinery seen from the config file's side — block by block, what each one tells the engine — see
[reading a run config](explanation-run-config.md). Read this page once end to end and the module
layout under `src/portfolio_optimizer/` will feel inevitable.

The short version: **read the config → prove every named function exists and has the right shape →
load data through loaders → assemble and validate → build every portfolio at once (slice, rules,
solve-order key, a pure-numpy problem) → derive who waits for whom from what each may trade on the side
the run couples through → solve along that graph with the configured solve step (cvxpy by default) →
re-check each answer without cvxpy → round to whole shares → publish the orders
once → write a manifest.** Money is `Decimal` everywhere except inside the solver,
and there are exactly two conversion points. Every stage validates its own output, so a bad input
fails at the earliest stage that can detect it, with a message naming what is wrong.

## 1. Startup: nothing touches data yet

**Settings** (`settings.py`) come from environment variables: where data is read from
(`PORTFOLIO_OPTIMIZER_DATA_ROOT`), where runs are written (`PORTFOLIO_OPTIMIZER_OUTPUT_DIR`), how loudly
to log, and — deliberately here rather than in the config — which Dask cluster the run provisions for
itself and how big it is (`PORTFOLIO_OPTIMIZER_CLUSTER`, how many workers to provision up front, how
many after assembly, and how long to wait for the first). There are no defaults, `.env` files are read
only when you pass `--env-file`, and an *unknown* `PORTFOLIO_OPTIMIZER_*` variable is an error — a typo
fails loudly instead of being ignored. `PORTFOLIO_OPTIMIZER_CLUSTER=auto` is resolved right here, to
`kubernetes` inside a pod and `local` anywhere else, so what the manifest records is what happened.

**The config** (`config/models.py`) is a strict pydantic model: unknown keys are errors, money is written
as strings (`"0.05"`) so it becomes exact `Decimal`, and `as_of_date` must carry a timezone. The validated
config is hashed on its canonical JSON form, so whitespace and the `$schema` pointer never change the
hash.

**Resolution** (`config/resolve.py`) is where the one convention is enforced. Every step in the config —
loaders, assembly steps, rules, the solve-order step, objective terms, constraints, the solve step, the
sink — is a function name such as `"transaction_cost"` or `"mypkg.mod:my_rule"`. Before any data loads,
the resolver:

- imports the function;
- checks its signature against the **contract for its kind**, by argument name and annotation — a rule
  must take `data: PortfolioData` and return `PortfolioData`, a term must take `x: DecisionVars,
  spec: ProblemSpec` and return `ObjectiveTerm`, a solve-order step must take `data: PortfolioData` and
  return `Decimal`, a solve step must take `request: SolveRequest` and return `SolveResult`, and so on;
- validates the JSON `params` object against the function's own `Params` model;
- notes whether a term or constraint declares the optional `chain: ChainState` argument, which makes it
  *chain-aware* (`ResolvedStep.reads_chain`): it reads what higher-priority portfolios traded on the side
  the run couples through, and its presence is what makes one portfolio wait for another. A rule cannot
  declare it — rules never see other portfolios;
- records two hashes — the function's source text and its params;
- checks the solver: known to the adapter, installed in this process, and able to honor `time_limit_s`;
- and, once every step has resolved, constructs every term and constraint once against a one-security
  dummy spec under the run's side profile, so a term that raises when called, or reads a decision vector
  the side does not have, is refused here. The solve step is not run: a firm's step may reach a service,
  and the dummy is not a problem worth solving.

Every failure is collected and reported together. The same resolution runs in every process that will
solve — here, and again on every worker before it does any work (§9) — so all of them apply identical
checks. There is no execution mode to check the steps against: the schedule is derived later, from the
steps and the data (§9).

`portfolio-optimizer validate-config` stops here and prints one line per resolved step. `run` then,
**before any data loads, asks for its cluster** — local worker processes or Kubernetes pods. The call
does not block; the point is that the cluster comes up underneath the slow stage that follows.

## 2. Loading and assembly: validation, first layer

Loading is the slow part of a real run — API calls and database queries, not files — so `engine/load.py`
is asynchronous and runs the **dependency DAG the config declares**: every dataset is a task that
starts the moment the datasets its `depends_on` names (plus `portfolios`, for a `per_portfolio`
dataset) have loaded, and one with no dependencies starts immediately, so the stage costs its longest
chain rather than its sum. Each loader is called with a `LoadRequest` (dataset name, portfolio ids,
its dependencies' frames as `inputs`, `as_of_date`, `data_root`, `run_id`, and a rate limiter): an
`async def` loader runs as a task on the event loop, a plain `def` loader in a worker thread so a
blocking driver never stalls the loop.

`portfolios` is engine-known but scheduled like any other node — loaded by a loader, or read at no
cost from an inline list in the config, whose written order is the solve order. Once it loads, the
engine validates it against the `PORTFOLIOS` schema and sorts it by `solve_order` then `portfolio_id`.
`solve_order` is a *priority* — lower solves first, ties break on the id, and the column may be
omitted or repeated — and this loaded order is only the order builds are submitted in and the fallback
key when no `solve_order` step is configured. The ids reach the entries that declare a dependency on
`portfolios` as `request.portfolio_ids`; an entry that declares nothing — the example's `universe`,
the slowest source in the run — is already loading by then.

How many times each loader is called is the dataset's `scope`. A **global** dataset is one call for the
whole book, and is what the assembly steps see. A **per-portfolio** dataset is the engine's own
fan-out: the ids are cut into batches of `batch_size` and the loader is called once per batch, so a
source that answers one account at a time is driven by the engine rather than privately inside a
loader. The batches share the dataset's one rate limiter and run alongside the global loaders, so on a
book whose global stage is the long pole they cost nothing.

Failure is split along the same line. A **structural** problem rejects the whole run — a required
dataset missing, a schema violated, a global loader that raised, a per-portfolio dataset no batch of
which came back — and a dataset downstream of one of those is skipped, never called, and named beside
the failure. A **coverage** problem fails only the portfolios it touches — one batch that raised,
a portfolio with no `details` row — recorded as a failure at stage `load` so
that every other portfolio still solves. A portfolio rejected here never entered the run, so it traded
nothing and couples to nobody: it appears in the manifest as failed and blocks no one in the schedule.
Otherwise every outcome is collected — one failing dataset does not cancel the others — and all
failures are reported together. Every loaded frame gets an audit record — loader name, source hash,
params hash, row count, columns, how many batches it took, how many portfolios a failed batch cost,
how long it took, and a **content hash** that ignores row order, column order, and index, so which
batch returned first never reaches the manifest.

Every loaded dataset — the portfolio list included — may carry a `rate_limit`, inline or as the name of
a pool shared with other inputs on the same backend; the loader receives it as `request.rate_limiter`
and wraps each call to its source in it, and `fan_out` packages the one-call-per-portfolio pattern.
Why the bound is per input is in [the architecture explanation](explanation-architecture.md#loading-is-the-slow-part-so-it-is-concurrent-and-metered);
the keys are in [the reference](reference-run-config.md#rate-limits).

The shipped loaders (`loaders.py`) are the only place I/O happens. Each stands in for a service — a
custodian, a security master, an account master — waiting as long as that source would and then
answering from a table under the data root. They read decimal and timestamp columns as strings and let
`coerce_frame` turn them into `Decimal` exactly; a float only ever becomes `Decimal(repr(value))`, the
shortest round-tripping form. `load_holdings` is the fan-out pattern, one call per account;
`load_details` is its blocking twin, one query per batch of ids in a worker thread. Every dataset is a
frame, so every loader has the one shape.

`assemble` runs the config's **assembly steps** in order. Each is a function `Frames → Frames` over
every loaded dataset by name — a vendor's analytics joined into `universe`, then dropped once it has
done its job. The shipped `join` is defensive: key dtypes are aligned so a `str` key never
silently joins to a `string` key as `object`; a brought column the target already has is refused unless
`overwrite` is set; the declared cardinality is enforced by pandas' `merge(validate=...)`; and
`require_all_matched` uses the merge indicator to report unmatched keys by example. A custom step gets
the same treatment: a `ValueError` it raises rejects the run under the step's name, and its source hash,
row counts, and the columns it added go into the manifest. After the last step, `holdings`, `universe`,
and `details` must exist, and **every engine-known frame is validated against its schema** — column
set, dtype, nullability, bounds, unique key, and frame-level invariants such as "cash_lb does not
exceed cash_ub" — with all failures across all frames reported at once. `holdings` and `universe` may
carry any further columns; that is where security analytics live. `constraints` is engine-known but
optional — a run that omits it is bound by nothing but the trade identity. Finally, `details` must have a row for every
portfolio. Whatever datasets remain that the engine does not know become the run's extras.

Anything failing here raises `InputRejectedError` and the run exits with code 2. Nothing was solved.

## 3. Slice per portfolio: validation, second layer

`slice_portfolio` builds a `PortfolioData` bundle: this portfolio's `details` row typed into a
`PortfolioDetails` model — which carries the account's style limits alongside its facts — its holdings,
the full universe, its own rows of `constraints`, and its share of the
extras — a dataset with a `portfolio_id` column reduced to this portfolio's rows, one without passed
whole. This happens *in the worker*, which received the assembled
datasets once: a task carries a portfolio id and nothing else.

The frames the slice produces are marked `prevalidated`: assembly already checked them against their
schemas, the universe is passed whole, and a row subset of validated holdings or sector bounds keeps
every per-column check, the key's uniqueness, and the bounds' ordering. So the bundle does not check
them again, here or after a rule that leaves them untouched. A rule that returns a *new* frame loses that standing for it, and the new frame is validated.

`PortfolioData.__post_init__` (`domain/data.py`) holds the cross-frame invariants: `as_of_date` is UTC,
holdings contain only this portfolio, every constraint row and every extra with a `portfolio_id` column
belongs to this portfolio, and — because the two tables will be stacked into one optimizer frame — every column
that `holdings` and `universe` share has the same dtype on both. A held name need not be in the
universe; that is the shipped build's requirement, not the bundle's. A failure here is a per-portfolio
failure at stage `slice`, not a run-level rejection; other portfolios proceed according to `on_error`.

`PortfolioData.optimizer_frame()` is the bundle's view for an optimizer that wants one table: the
holdings rows followed by the universe rows, tagged by a `source` column, over the union of both tables'
columns with typed nulls where a side lacks a column. The shipped build does not use it — it aligns
everything to the universe — but a custom build that takes "one optimizer frame plus the account's
limits" gets exactly that from `data.optimizer_frame()` and `data.details`.

## 4. Rules: validation, third layer

`engine/pipeline.py` runs each rule in config order. A rule is a pure function
`PortfolioData → PortfolioData`. The only way to return a modified bundle is `with_changes(...)`, which
constructs a new `PortfolioData` and therefore **re-runs every check from stage 3**. A rule cannot hand
the optimizer a broken bundle. Each rule gets an audit record of row counts before and after.

The shipped rules (`rules.py`) show the patterns: `restrict_low_liquidity` freezes names below an ADV
threshold it reads from an extra dataset rather than from the config, `add_zero_alpha` fills in a column the objective needs, `attach_universe_columns` copies the universe's analytics
onto holdings for a book that loads holdings per account, `cap_single_name` tightens the style. A rule never sees other
portfolios. What it *can* do is shrink the portfolio's tradable set — freeze a name, or cap it at its
current weight in a run that couples through buys — and that is what lets portfolios solve
concurrently (§9): two portfolios wait on each other only when they can both trade the same security on
the side the run couples through.

Between the rules and the build, the optional **solve-order step** (`solve_order.py`) reads the ruled
bundle and returns the portfolio's solve-order key, a finite `Decimal`; lower solves first. It replaces
the portfolios frame's column and answers "who gets first pick of a shared budget" from the data — the
shipped `most_uninvested_first` puts the account with the most left to put to work first.

## 5. Build: `Decimal` becomes float64, once

`engine/build.py` sorts the universe by `security_id` and aligns everything to that order, which is
why it requires every held name to be in the universe (a `BuildError` otherwise). In exact `Decimal` it
computes current weights (`shares × price / nav`), tax per dollar sold (gain fraction times
the short- or long-term rate, long-term when held longer than 365 days; losses come out negative),
transaction cost from `tcost_bps`, per-name bounds (restricted names frozen at their current weight,
optional `min_weight`/`max_weight` columns tightening the style cap), the sector indicator matrix, and
ADV capacity as a fraction of NAV. Only then does `to_float64` convert — refusing bools, non-Decimals,
and anything non-finite.

Any universe column the schema does not declare is exported by name — numeric ones into
`spec.columns` for `spec.column("my_signal")`, boolean ones into `spec.flags` as real boolean masks for
`spec.flag("excluded")` — so a custom term can read either. Holdings' extra columns are not exported:
the build has no row for a name that is not in the universe.

The result is a `ProblemSpec` (`domain/results.py`): pure numpy, read-only arrays, no cvxpy. Its own
`__post_init__` checks shapes, sortedness, finiteness, `lb ≤ ub`, positive prices, and so on, and it
carries a content hash. Alongside it, `OrderInputs` keeps the *exact* `Decimal` prices and share counts
for stage 8.

## 6. Solve

`engine/solve.py` hands the configured **solve step** a `SolveRequest` — the spec, the chain, the
side profile, the resolved terms and constraints, the run's extra datasets as the rules left them, the
`solver` block — and takes back a `SolveResult`: weights aligned to the spec and, if the step minimized
one, an objective. The constraints and the extras cross the build unchanged for the same reason: the
engine does not know what either means, so it carries rather than interprets them. The default
step, `solvers.cvxpy`, creates the side profile's decision variables — `w`, `buy`, and `sell` under
`both`; `w` alone under `buy` or `sell`, with the trade an expression of it — all fractions of NAV,
adds the profile's trade identity, invokes each configured term and constraint function to obtain
expressions, and hands them to the adapter. `cvx/adapter.py` is the **only module that imports
cvxpy**; terms are written against a dozen typed atoms (`dot`, `matvec`, `sum_squares`, `at_most`,
...) so that the verifier can mirror each one in numpy. The adapter checks that the problem is
DCP-compliant, maps `time_limit_s` to the solver's own option, solves once, and returns the
`SolveResult` as is — status, weights, objective, iterations, solve time, solver and version. A step
that is not cvxpy — the shipped `pro_rata_fill`, a firm's library, a function of your own — returns
weights the same way and is verified the same way; see
[how to replace the cvxpy solve](how-to-write-a-solve-step.md).

Classification decides what the result means:

- **Optimal.** The side profile decides the buy/sell split the engine reports for the weights: for the
  two-sided profile, the minimal one, `buy = max(w − w0, 0)` and `sell = max(w0 − w, 0)`. With no term
  charging for trading, an interior-point solver can return a wash trade — buy 0.3 and sell 0.3 of the
  same name — that nets to the right weights but is not minimal.
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
checked against, and the side profile's **identity checks** — under `both`, the trade balance and
complementarity (`min(buy, sell) ≈ 0`), which is what proves the reported split held; under a
one-sided profile, that `w` never crossed `w0` and the absent side is zero. Every check is reported
under the label of the constraint it belongs to (`identity` and `solution` for the engine's own).

Tolerances come from the config's `post_solve` block: `violation_tol`, one tolerance for every residual,
defaults to `1e-6` — about a hundred times looser than the solver's own — so a pass is a genuine
statement about the solution rather than a restatement of the solver's convergence check, and the two
objective tolerances bound the gap. Custom steps with no twin are listed as `unverified` in the
manifest, and the objective comparison is skipped when some term cannot be recomputed or when the solve
step reported no objective. A failed check raises `VerificationError`; the portfolio fails at stage
`solve`.

## 8. Orders: float64 becomes `Decimal`, once

`engine/orders.py` converts each weight delta to shares: **nearest share** (half-even), then down to a
lot multiple, then sells are clamped to what is held, then trades below `min_trade_notional` are
dropped. `notional = quantity × reference_price` is computed in `Decimal` from `OrderInputs`, never from
the float copies inside the spec, and the `ORDERS` schema's `notional_matches` invariant confirms it.
Each order also carries the spec hash, run id, `as_of_date`, the float target weight, and the unrounded share
count for audit.

`rounding_drift` rebuilds the executed weights from the orders and measures the worst deviation from the
solved weights. The tolerance is derived, not configured: one lot of the priciest name plus one
dust-filtered trade, both as fractions of NAV. Exceeding it raises `DriftError`.

## 9. The dependency graph, and where the work runs

Portfolios in one run can compete for the same trades. The example's `cumulative_adv_participation`
constraint says "trade in each name no more than the ADV budget that higher-priority portfolios' trades
on the side the run couples through have not already consumed" (and, chain-free, "trade no more than
your own participation"). That is the only kind of coupling the engine has: **a run couples through its
one side** — buys under `both` and `buy`, sells under `sell`. What a two-sided run's portfolio *sold*
never changes what a later one may do — a product decision, and everything in this section leans on it.

The mechanism is small. A build reports its portfolio's solve-order key and its **tradable set** — the
securities the side profile lets it trade on that side: buyable (`ub > w0`; a name frozen or capped at
its current weight is outside it) or sellable (held and `lb < w0`). Portfolios sort by
`(key, portfolio_id)`, and that order is the graph (`engine/schedule.py`): portfolio *j* depends on
every earlier *i* whose tradable set intersects its own, and on nothing else. If no term or constraint
declares `chain`, there are no edges at all. Under `execution.dependencies: "all"` every earlier
portfolio is a predecessor — one line, the same answer, for diagnosis. A build that failed has an
unknown tradable set and is treated as overlapping everything after it. The graph is never transitively
reduced: a solve folds its *direct* predecessors' own trades, so every overlapping earlier portfolio
stays a direct dependency.

Because every predecessor is *earlier*, the graph is grown a portfolio at a time rather than derived
all at once: the runner walks the order and places each portfolio as its build reports, so the head of
the book is solving while the tail is still building. Nothing waits for the build wave to finish.

Each solve folds its predecessors' orders on that side into a `ChainState`: `traded_shares`, whole
shares per security, projected onto this spec's securities and **zeroed wherever this portfolio cannot
trade the name on that side**. That mask is why the schedule never changes the answer — the argument is
in [the architecture explanation](explanation-architecture.md#a-run-couples-through-its-one-side-so-the-schedule-is-a-graph).
Order rounding makes the tradable set structural (a BUY is clamped to the room under `ub` and a SELL to
the shares held, so solver noise at a bound never produces a trade the graph could not have seen), and
`finish_portfolio` asserts it, along with every order being on a side the run trades. The chain state's
hash — the ids and the shares, never who traded them — is recorded per portfolio.

`engine/runner.py` drives one schedule, on the cluster:

- **build** every portfolio at once — slice, rules, the solve-order key, the spec — chain-free and in
  parallel. Each build stays on the worker that made it; a `summarize` task sends back only the key, the
  tradable ids, the spec hash, and the rule audit, stamped with the worker's environment.
- **place** each portfolio in the graph as its build reports, walking the solve order, and log the
  shape it ends up with: portfolios, edges, components, the longest chain of solves.
- **solve** each portfolio where its build lives, submitted as it is placed with its predecessors'
  *contributions* — their order rows on the coupled side, a few kilobytes each — as Dask dependencies,
  so the scheduler enforces the order and starts a solve the moment its last predecessor finishes. A
  portfolio nothing waits for is never asked to contribute. Solves run in solve order and every solve
  outranks every build.
- **classify** outcomes in solve order as they complete, persisting each result as it arrives, so the
  worker count and completion order never change a record.

Only two things make a solve wait for a build it does not depend on. A configured `solve_order` step
computes the key from the *ruled* bundle, so the order is itself a build output and the walk cannot
start until every build has reported — the reason to put the priority in the portfolios frame's column
when it can go there. And under `fail_fast`, a failure stops submission, after which the builds behind
it are read only to finish the graph the manifest records.

![Where each stage runs](images/execution-stages.svg)

Which cluster the run provisions is a *setting*: a `LocalCluster` of worker processes on a laptop, a
`KubeCluster` of pods on Kubernetes, or a scheduler address (`engine/dask_backend.py`), behind a seam
the runner can also be tested against with a fake (`engine/backends.py`). The runner drives it through
one lifetime: **start** it before the load stage, so worker processes import the solver stack and pods
come up while the loaders wait on their sources; **scale** it and **wait** for the first worker after
assembly; **probe** every worker that has joined — each resolves the config itself, which is where a
missing solver or step package surfaces, and reports its fingerprint — and stop the run as an
infrastructure failure if any cannot; **share** the assembled datasets and the config with it once —
scattered, and replicated between workers on demand — so a task carries a portfolio id and nothing
else; **submit** every build, then every solve with its dependencies; **close** it in a `finally`.
Workers re-resolve the config themselves — function objects are never pickled, only their names.

![The run owns its cluster: provisioning overlaps the load stage](images/cluster-lifecycle.svg)

Every task returns the **fingerprint of the process that ran it** — interpreter, numerical libraries,
solver, the versions of the packages behind external steps, the git revision, the image digest — and
the runner compares it with its own. A worker running different code fails its portfolio at stage
`worker` rather than answering, and the manifest lists every environment that did work. A local
cluster's workers are spawned from the run's own environment, so the fingerprints agree by construction;
on Kubernetes this is what makes sharing machines safe.

Under `fail_fast`, the first failure *in solve order* cancels every solve behind it — a running task
finishes and is discarded — and every lower-priority portfolio is recorded `skipped`, whatever it had
finished, so the manifest never depends on timing. Under `continue`, a failure skips only the portfolios
that depended on it: their solve returns `skipped` naming the predecessor, on the cluster, with no
bookkeeping in the main process. A worker that dies — for instance on an unpicklable result — becomes a
per-portfolio failure at stage `worker`, and the exception Dask then raises for every dependent is
classified the same way: `skipped` when a predecessor failed, `worker` otherwise. A cluster that never
produces a worker within its timeout is an infrastructure failure, exit code 3, with a `cluster` record
in the manifest and every portfolio skipped. [How to run on a cluster](how-to-run-on-a-cluster.md) has
the settings.

## 10. Failure semantics and exit codes

Every portfolio ends as a `PortfolioResult` or a `PortfolioFailure` naming its stage — `slice`, `build`,
`solve`, `worker`, or `skipped` — with the exception type and message. A skipped portfolio's message
names what it was waiting for: the predecessor that failed, or, under `fail_fast`, any higher-priority
failure. A build that fails has an unknown tradable set and is treated as overlapping everything after
it, so under `continue` it skips every lower-priority portfolio whenever a step reads the chain. The
process exit code is **0**
when every portfolio solved, **1** when any failed, **2** when the inputs were rejected before anything
was solved — invalid settings, a config that does not validate or resolve, datasets that fail loading
or assembly — and **3** for infrastructure: a sink failure, a cluster that never came up, a config file
that cannot be read.

## 11. Persist, publish, record

For each solved portfolio, the spec, solution, and chain state — its predecessors' trades on the side
the run couples through, and which predecessors — are written as `.npz` files under
`<output_dir>/<run_id>/{problem_specs,solutions,chain}/` as each result is classified, while the cluster
is still up. These are what `portfolio-optimizer verify` reloads to re-check a solution without the
solver stack.

The **sink** is called exactly once, with every solved portfolio's orders concatenated and sorted, and
only when at least one portfolio solved. The shipped sinks write Parquet (Decimals as Arrow decimals) or
CSV, atomically, via a temp file and rename. A sink failure is exit code 3, but the manifest is still
written.

The **manifest** (`engine/manifest.py`) records the run id, name, and timestamps; the git revision and
whether the tree was dirty; the schedule the run derived — coupling, edges, components, the critical
path; the Python, cvxpy, numpy, pandas, and solver versions, and every worker environment that executed
a task; the backend's lifetime — what was asked for, when the first worker answered, when it was
released; the resolved config and its hash; the settings, with the cluster and worker counts; every term
and constraint with its params and label; every dataset's provenance and content hash; and, per portfolio, the
solve-order key, the number of predecessors, the rule audits, spec hash, chain-input hash, solver
statistics, verification outcome, drift, and the orders' count, hash, and gross notional. The manifest is then self-hashed, and
`load_manifest` refuses one whose hash does not match its content.

`portfolio-optimizer diff-manifests` compares two manifests and names the **first stage at which they
diverge**: config, code, versions, datasets, assembly, and then per portfolio status → rules → spec →
solve → orders. "Did the data change, or did the solver?" is a one-command question. The
[reference page](reference-manifest.md) documents every field.

## The example, stage by stage

`configs/example_run.json` declares a hundred accounts over three securities, four global datasets and
two loaded per account (`holdings`, a call each under a shared rate-limit pool, and `details`,
twenty-five ids a call), no assembly steps, one rule, three terms (alpha, tax cost, transaction cost),
up to seven constraints an account, the Clarabel solver, and `fail_fast`; how many workers the run has
is the `PORTFOLIO_OPTIMIZER_MAX_WORKERS` setting. The first two accounts are the ones to follow.

P1 and P2 each hold $500,000 of A and $500,000 of B (5,000 A at 100, 10,000 B at 50 against a cost of
40, so B carries a fifth of unrealized gain). C has the best expected return and the worst liquidity: it
trades 100,000 shares a day at 10, and the style allows 25% participation, so a portfolio may buy at
most 25,000 shares — a quarter of its NAV. A and B are `TECH`, C is `HEALTH`, and the sector bands —
`TECH` in `[0.5, 1]`, `HEALTH` in `[0, 0.5]` — do not bind.

P1 solves first. Its 40% single-name cap puts both holdings over the line, so it must trim; it raises
the quarter of NAV that C's budget allows and takes A down first, because A is at cost and B is not:
sell 1,500 A, sell 2,000 B to the cap, buy 25,000 C — the hand-computable optimum, to the share. P2 can
buy the same securities, so it waits for P1; when it solves, the chain state says C's budget is spent,
and with a 60% cap forcing nothing and a short-term rate on B's gain making every A/B swap uneconomic,
P2 produces no orders. The other ninety-eight accounts follow behind them, each waiting on all the
ones ahead because every account in this book trades the same three names; about half find something
worth trading. Run the config twice and `diff-manifests` reports `no differences`; run it with
`"dependencies": "all"` and every portfolio record is the same again (the diff names only the config).
The [tutorial](tutorial-first-run.md) walks through exactly this.

## Where validation happens, in one list

1. **Settings** — unknown or missing environment variables; a Kubernetes cluster without a worker image.
2. **Config** — strict models; money as strings; timestamps with a zone.
3. **Resolver** — the function exists, its signature matches the contract, its params validate; the
   solver is known and installed; constraint labels are unique; then every term and constraint is
   constructed once against a one-security dummy spec under the run's side profile, so a step that
   raises, reads a side the run lacks, or returns the wrong type is refused. Run by `validate-config`,
   at the start of `run` before a cluster is asked for, and again on every worker before it does any
   work — the same checks in every process.
4. **Loaders** — dtypes declared up front; exact `Decimal` coercion.
5. **Assembly** — each step's own claims (join keys, cardinality, coverage, dtype agreement on a
   union); then the required frames exist and every frame schema holds; then details and constraints
   for every portfolio.
6. **`PortfolioData`** — cross-frame invariants including holdings/universe dtype agreement, re-run
   after every rule; a frame a rule replaces is re-validated against its schema.
7. **`ProblemSpec`** — shapes, finiteness, bound ordering, read-only arrays.
8. **Solve step** — weights of the right shape, status classified, infeasibility diagnosed; for the
   cvxpy step, DCP-compliant first.
9. **Verifier** — every shipped constraint and the objective, independently of cvxpy.
10. **Orders** — the `ORDERS` schema including `notional = quantity × price`, every order on a side the
    run trades and inside the tradable set, then the drift bound.
11. **Manifest** — self-hash checked on load.
