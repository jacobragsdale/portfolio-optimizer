# Explanation: the life of a run

This page walks through one run of the engine from the command line to the manifest, in the order the
code executes. It is the companion to [how the engine is built and why](explanation-architecture.md):
that page explains the design decisions; this one shows where each of them takes effect. For the same
machinery seen from the config file's side — block by block, what each one tells the engine — see
[reading a run config](explanation-run-config.md). Read this page once end to end and the module
layout under `src/portfolio_optimizer/` will feel inevitable.

The short version: **read the config → prove every named function exists and has the right shape,
and every term is a kind the engine knows → load data through loaders → assemble and validate → build
every portfolio at once (slice, rules, solve-order key, a pure-numpy problem, its constraint rows
parsed) → derive who waits for whom from what each may trade on the side the run couples through and
what its rows declare they read → solve along that graph with the configured solve step (cvxpy by
default) → re-check each answer without cvxpy → round to whole shares → publish the orders once →
prove the business rules on what went out → write a manifest.** Money is `Decimal` everywhere except inside the solver, and there are exactly two
conversion points. Every stage validates its own output, so a bad input fails at the earliest stage
that can detect it, with a message naming what is wrong.

## 1. Startup: nothing touches data yet

**Settings** (`settings.py`) come from environment variables: where data is read from
(`PORTFOLIO_OPTIMIZER_DATA_ROOT`), where runs are written (`PORTFOLIO_OPTIMIZER_OUTPUT_DIR`), how loudly
to log, which packages a qualified step name may import from, and — deliberately here rather than in
the config — where the work runs (`PORTFOLIO_OPTIMIZER_CLUSTER`: this process under `inline`, the
default, or a Dask cluster the run provisions for itself) and how big that cluster is: how many
workers to provision up front, how many after assembly, and how long to wait for the first. Every
setting has a default a laptop can run with, `.env` files are read only when you pass `--env-file`,
and an *unknown* `PORTFOLIO_OPTIMIZER_*` variable is an error — a typo fails loudly instead of being
ignored. A cluster the run cannot actually ask for is refused right here: a gateway address without
the image its pods run, or without the password it authenticates with.

**The as-of instant** is a run argument, `--as-of 2026-08-28T00:00:00Z`, and it must carry a time
zone; a naive one is refused, because a holding period compared against it would be off by hours. It
is not in the config, so one wiring runs every day under one hash.

**The config** (`config/models.py`) is a strict pydantic model: unknown keys are errors, and money is
written as strings (`"0.05"`) so it becomes exact `Decimal`. The validated config is hashed on its
canonical JSON form with the `run` block and the `$schema` pointer left out, so whitespace, a rename,
and a relabel never change the hash.

**Resolution** (`config/resolve.py`) is where the one convention is enforced. Every step in the config —
loaders, assembly steps, rules, the solve-order step, the build step, the solve step, the sink — is a
function name such as `"restrict_low_liquidity"` or `"mypkg.mod:my_rule"`: a bare name is looked up
in the template module for its kind and then among the steps installed packages publish as entry
points; a qualified name is imported from anywhere, or from the packages the settings allow. Before any
data loads, the resolver:

- imports the function;
- checks its signature against the **contract for its kind**, by argument name and annotation — a rule
  must take `data: PortfolioData` and return `PortfolioData`, a build step must take the same and
  return `ProblemSpec`, a solve-order step must return `Decimal`, a solve step must take
  `request: SolveRequest` and return `SolveResult`, and so on;
- validates the JSON `params` object against the function's own `Params` model;
- records two hashes — the function's source text and its params;
- parses every record in `objective` as the typed term kind its `kind` names — the shipped `linear`,
  or one an installed package publishes — and insists the names are unique. A kind declares on its
  class whether it reads the chain (`reads_chain`), which is what makes one portfolio wait for
  another; a rule cannot read it — rules never see other portfolios;
- under the shipped `cvxpy` step, checks the solver its params name: known to the adapter, installed
  in this process, and able to honor `time_limit_s`;
- and, once every step has resolved, renders every term once against a one-security dummy spec under
  the run's order-flow profile and checks the problem is convex, so a term that raises when rendered, reads a
  decision vector the order flow does not have, or is not DCP is refused here. The solve step is not run: a
  firm's step may reach a service, and the dummy is not a problem worth solving.

Every failure is collected and reported together. The same resolution runs in every process that will
solve — here, and again on every worker before it does any work (§9) — so all of them apply identical
checks. There is no execution mode to check the steps against: the schedule is derived later, from the
steps and the data (§9).

`portfolio-optimizer validate-config` stops here and prints one line per resolved step and one per
term. `run` then, **before any data loads, asks for its backend** — this process, or local worker
processes, or a gateway's pods. The call does not block; the point is that a cluster comes up
underneath the slow stage that follows.

## 2. Loading and assembly: validation, first layer

Loading is the slow part of a real run — API calls and database queries, not files — so `engine/load.py`
is asynchronous and runs the **dependency DAG the config declares**: every dataset is a task that
starts the moment the datasets its `depends_on` names (plus `portfolios`, for a `per_portfolio`
dataset) have loaded, and one with no dependencies starts immediately, so the stage costs its longest
chain rather than its sum. Each loader is called with a `LoadRequest` (dataset name, portfolio ids,
its dependencies' frames as `inputs`, `as_of_date`, `data_root`, and `run_id`): an
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
loader. At most `max_in_flight` of those batches run at once, and they run alongside the global
loaders, so on a book whose global stage is the long pole they cost nothing.

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

Why the bound on those calls is one number per input, and the engine's rather than the loader's, is in
[the architecture explanation](explanation-architecture.md#loading-is-the-slow-part-so-it-is-concurrent-and-metered);
how `max_in_flight` behaves at load time is in [the reference](reference-run-config.md#max_in_flight).

The shipped loaders (`loaders.py`) are the only place I/O happens. Each stands in for a service — a
custodian, a security master, an account master, a compliance service — waiting as long as that source
would and then answering from a table under the data root. They read decimal and timestamp columns as
strings and let `coerce_frame` turn them into `Decimal` exactly; a float only ever becomes
`Decimal(repr(value))`, the shortest round-tripping form. `load_holdings` is the fan-out pattern, one
call per account; `load_details` is its blocking twin, one query per batch of ids in a worker thread.
Every dataset is a frame, so every loader has the one shape.

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
exceed cash_ub" — with all failures across all frames reported at once. All three may carry any
further columns: that is where security analytics and the desk's own account limits live, and the
build exports every one. Only `security_id` and `price` are required of the universe; `sector`,
`adv_shares`, `lot_size`, and `restricted` are optional and a build or a rule that needs one says so.
`constraints` is engine-known but optional — a run that omits it is bound by nothing but the trade
identity and the spec's own bounds. Finally, `details` must have a row for every portfolio. Whatever
datasets remain that the engine does not know become the run's extras.

Anything failing here raises `InputRejectedError` and the run exits with code 2. Nothing was solved.

## 3. Slice per portfolio: validation, second layer

`slice_portfolio` builds a `PortfolioData` bundle: this portfolio's `details` row typed into a
`PortfolioDetails` model — which carries the account's style limits alongside its facts, and every
further column of the row as `extra` — its holdings, the full universe, its own rows of `constraints`,
and its share of the extras — a dataset with a `portfolio_id` column reduced to this portfolio's rows,
one without passed whole. This happens *in the worker*, which received the assembled datasets once: a
task carries a portfolio id and nothing else.

The frames the slice produces are marked `prevalidated`: assembly already checked them against their
schemas, the universe is passed whole, and a row subset of validated holdings or constraint rows keeps
every per-column check and the key's uniqueness. So the bundle does not check them again, here or
after a rule that leaves them untouched. A rule that returns a *new* frame loses that standing for it,
and the new frame is validated.

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
threshold it reads from an extra dataset rather than from the config, `restrict_to_mandate` freezes
every name whose sector is outside the account's mandate rows, `add_zero_alpha` fills in a column the
objective needs, `attach_universe_columns` copies the universe's analytics onto holdings for a book that
loads holdings per account, `cap_single_name` tightens the style. A rule that freezes names starts
from `restricted_flags(universe)`, the `restricted` column or all-false where the universe has none,
and a rule that needs `adv_shares` or `sector` refuses a universe without it by name. A rule never sees
other portfolios. What it *can* do is shrink the portfolio's tradable set — freeze a name, or cap it at
its current weight in a run that couples through buys — and that is what lets portfolios solve
concurrently (§9): two portfolios wait on each other only when they can both trade the same security on
the side the run couples through.

Between the rules and the build, the optional **solve-order step** (`solve_order.py`) reads the ruled
bundle and returns the portfolio's solve-order key, a finite `Decimal`; lower solves first. It replaces
the portfolios frame's column and answers "who gets first pick of a shared budget" from the data — the
shipped `most_uninvested_first` puts the account with the most left to put to work first.

## 5. Build: `Decimal` becomes float64, once

The **build step** is configured like any other, `(data: PortfolioData[, params]) -> ProblemSpec`, and
`standard` (`engine/build.py`) is the default. It sorts the universe by `security_id` and aligns
everything to that order, which is why it requires every held name to be in the universe (a
`BuildError` otherwise). In exact `Decimal` it computes current weights (`shares × price / nav`), tax
per dollar sold (gain fraction times the short- or long-term rate, long-term when held longer than 365
days as of `--as-of`; losses come out negative), transaction cost from `tcost_bps` where the universe
carries it, per-name bounds (restricted names frozen at their current weight, optional
`min_weight`/`max_weight` columns tightening the style cap), and ADV capacity as a fraction of NAV where
it carries `adv_shares`. Only then does `to_float64` convert — refusing bools, non-Decimals, and
anything non-finite.

Everything the bundle carries beyond the schemas is exported by name: each numeric universe column
into `spec.columns` for `spec.column("my_signal")`, each boolean one into `spec.flags` as a real
boolean mask for `spec.flag("excluded")`, each string one — `sector`, `country`, an issuer — into
`spec.groups` as a sparse membership matrix for `spec.group("sector")`, and every number on the
account's `details` row, declared or extra, into `spec.scalars` for `spec.scalar("cash_ub")`. That is
what lets a constraint row name a limit the engine has never heard of. Holdings' extra columns are not
exported: the build has no row for a name that is not in the universe.

The result is a `ProblemSpec` (`domain/results.py`): pure numpy, read-only arrays, no cvxpy — six fixed
vectors (`w0`, `price`, `shares_held`, `lot_size`, `lb`, `ub`) and the named columns, flags, groups,
and scalars. Its own `__post_init__` checks shapes, sortedness, finiteness, `lb ≤ ub`, positive prices,
and so on, and it carries a content hash. Alongside it, `order_inputs` keeps the *exact* `Decimal`
prices and share counts for stage 8, derived from whatever spec the build returned.

The build task then reads the portfolio's **constraint rows** (`domain/constraints.py`): where the
frame has a `kind` column, every row is parsed as the typed model it names — its fields from `params`,
its name from `label` — and asked what it needs of the spec; a malformed row, or one naming a column,
flag, scalar, or group the spec does not carry, fails the portfolio here, at stage `build`, before any
solve is scheduled on it, as does a term that reads a column this spec lacks. From the parsed rows the
build derives the portfolio's **consume set**, the schedule's other half (§9): empty when no row reads
the chain, the scopes of the chain-reading rows when they are the only readers, and the whole tradable
set when anything opaque might read it — a chain-aware term, a solve step other than the shipped one, a
frame with no `kind` column at all.

## 6. Solve

`engine/solve.py` hands the configured **solve step** a `SolveRequest` — the spec, the chain, the
order-flow profile, the typed terms, the portfolio's constraint rows, and the run's extra datasets as the
rules left them — and takes back a `SolveResult`: weights aligned to the spec and, if the step
minimized one, an objective. The constraints and the extras cross the build unchanged for the same
reason: the engine does not interpret what they mean, so it carries them. The default step,
`solvers.cvxpy`, creates the order-flow profile's decision variable — `w` alone, a fraction of NAV, with the
trade an expression of it: `buy = w − w0` under `inflow`, `sell = w0 − w` under `outflow` — adds the
profile's trade identity (`w ≥ w0` or `w ≤ w0`) and the spec's box `lb ≤ w ≤ ub`, renders each
configured term and each typed
constraint row through the model's own `to_cvxpy`, and hands the expressions to the adapter.
`cvx/adapter.py` is the **only module that imports cvxpy**; kinds are written against a small set of
typed atoms — affine `dot`, `matvec`, `masked`, `scale`, `weighted`, ...; convex `sum_squares`,
`norm1`, `absolute`, `pos`; the comparisons `at_most`, `at_least`, `equals` — so that each kind's numpy
half can mirror them. The adapter checks that the problem is DCP-compliant, maps `time_limit_s` to the
solver's own option, solves once, reads back the largest dual value of every constraint set — the
shadow price of each limit — and returns the `SolveResult` as is: status, weights, objective,
iterations, solve time, solver and version, the records of the constraints it applied, the duals. A
step that is not cvxpy — the shipped `pro_rata_fill`, a firm's library, a function of your own — returns
weights the same way and is verified the same way; see
[how to replace the cvxpy solve](how-to-write-a-solve-step.md).

Classification decides what the result means:

- **Optimal.** The order-flow profile turns the weights into the trade the engine reports:
  `buy = max(w − w0, 0)` under `inflow`, `sell = max(w0 − w, 0)` under `outflow`, the clip keeping solver
  noise past `w0` from becoming a few shares on the side the run does not have. One variable per name
  means there is no split to choose and no round trip to strip; a term that rewards a sale is priced
  exactly.
- **Infeasible.** `InfeasibleError` carries an arithmetic diagnosis computed without another solve,
  from what the spec carries: does the book start where the order flow cannot take it? Do the upper bounds
  even sum to the required investment? Do the lower bounds exceed it? Does moving the current weights
  inside their bounds already exceed `max_turnover`? Is a name that must trade out of ADV budget?
- **Unbounded.** Impossible inside the box, so it is reported as a bug in a custom term or constraint.
- **Anything else** is a `SolverFailureError` with the solver's own detail.

There is deliberately no path that returns the current portfolio as a fallback answer.

## 7. Verify: the second opinion

`engine/check.py` recomputes everything in numpy and never imports cvxpy (a test enforces this). Every
constraint the solve step reported is parsed back through the kind registry and re-checked through the
model's own `residual`; every term is recomputed through its own `value`, the values summed and
compared with the solver's reported objective. So a kind a package published is checked exactly like a
shipped one, and nothing typed is ever unverified — a step that reports no constraints has none
checked, which is the honest answer rather than a silent pass. It also checks finiteness, that the
solution's `spec_hash` matches the spec it is being checked against, and the order-flow profile's **identity
checks** — under `inflow`, `no_sells` (`w ≥ w0`), `trade_balance` (the reported buy is `w − w0`),
`nonneg_buy`, and `sell_absent`; under `outflow`, `no_buys`, `trade_balance`, `nonneg_sell`, and
`buy_absent`; under either, the box as `lb` and `ub`. Every check is reported under the label of the constraint
it belongs to (`identity` and `solution` for the engine's own), as `label/residual` where two rows of
one kind produce residuals of the same name.

Tolerances come from the config's `post_solve` block: `violation_tol`, one tolerance for every residual,
defaults to `1e-6` — about a hundred times looser than the solver's own — so a pass is a genuine
statement about the solution rather than a restatement of the solver's convergence check, and the two
objective tolerances bound the gap. The same tolerance decides which checks are **active**: a residual
within it of its bound is a constraint the answer sits against, which the run prints per portfolio as
`binding:` and the manifest records as `check.active`. The objective comparison is skipped when the
solve step reported no objective. A failed check raises `VerificationError`; the portfolio fails at
stage `solve`.

## 8. Orders: float64 becomes `Decimal`, once

`engine/orders.py` converts each weight delta to shares: **nearest share** (half-even), then down to a
lot multiple, then a sell is clamped to what is held and a buy to the room under the security's upper
bound, then trades below `min_trade_notional` are dropped. `notional = quantity × reference_price` is
computed in `Decimal` from `OrderInputs`, never from the float copies inside the spec, and the `ORDERS`
schema's `notional_matches` invariant confirms it. Each order also carries the spec hash, run id,
`as_of_date`, the float target weight, and the unrounded share count for audit.

`rounding_drift` rebuilds the executed weights from the orders and measures the worst deviation from the
solved weights. The tolerance is derived, not configured: one lot of the priciest name plus one
dust-filtered trade, both as fractions of NAV, plus the bound overshoot the verifier accepts. Exceeding
it raises `DriftError`.

## 9. The dependency graph, and where the work runs

Portfolios in one run can compete for the same trades. The example's `participation_limit` rows say
"trade in each name no more than the ADV budget that higher-priority portfolios' trades on the side the
run couples through have not already consumed" (and, chain-free, "trade no more than your own
participation"). That is the only kind of coupling the engine has: **a run couples through its one
side** — buys under `inflow`, sells under `outflow` — and a run has no other side, so nothing else a
portfolio did can change what a later one may do: a product decision, and everything in this section
leans on it. What an outflow sold reaches an inflow only as data the desk hands it, never
through the engine: the shipped `load_run_orders` loader reads the outflow's orders file as the
inflow's blotter and as the volume each name's ADV budget has already lost (§2).

The mechanism is small. A build reports its portfolio's solve-order key, its **tradable set** — the
securities the order-flow profile lets it trade on that side: buyable (`ub > w0`; a name frozen or capped at
its current weight is outside it) or sellable (held and `lb < w0`) — and its **consume set**, what its
own chain readers can see (§5). Portfolios sort by `(key, portfolio_id)`, and that order is the graph
(`engine/schedule.py`): portfolio *j* depends on every earlier *i* whose tradable set intersects *j*'s
consume set, and on nothing else. A portfolio whose consume set is empty waits for no one. When the
run can be told before any build that nothing reads the chain — no chain-reading row in any account,
no chain-aware term, the shipped solve step — there is no order to respect at all: every solve goes in
behind its own build, and the manifest records `coupling: "none"`. Under `execution.dependencies:
"all"` every earlier portfolio is a predecessor — one line, the same answer, for diagnosis. A build that
failed has an unknown tradable set and is treated as overlapping everything after it. The graph is
never transitively reduced: a solve folds its *direct* predecessors' own trades, so every overlapping
earlier portfolio stays a direct dependency.

Because every predecessor is *earlier*, the graph is grown a portfolio at a time rather than derived
all at once: the runner walks the order and places each portfolio as its build reports, so the head of
the book is solving while the tail is still building. Nothing waits for the build wave to finish.

Each solve folds its predecessors' orders on that side into a `ChainState`: `traded_shares`, whole
shares per security, projected onto this spec's securities and **zeroed wherever this portfolio cannot
trade the name on that side, or its own readers cannot see it**. That mask is why the schedule never
changes the answer — the argument is in
[the architecture explanation](explanation-architecture.md#a-run-couples-through-its-one-side-so-the-schedule-is-a-graph).
Order rounding makes the tradable set structural (a BUY is clamped to the room under `ub` and a SELL to
the shares held, so solver noise at a bound never produces a trade the graph could not have seen), and
`finish_portfolio` asserts it, along with every order being on a side the run trades. The chain state's
hash — the ids and the shares, never who traded them — is recorded per portfolio.

`engine/runner.py` drives one schedule, on the backend:

- **build** every portfolio at once — slice, rules, the solve-order key, the spec, the constraint rows
  — chain-free and in parallel. Each build stays on the worker that made it; a `summarize` task sends
  back only the key, the tradable and consume sets, the spec hash, and the rule audit, stamped with the
  worker's environment.
- **place** each portfolio in the graph as its build reports, walking the solve order, and log the
  shape it ends up with: portfolios, edges, components, the longest chain of solves.
- **solve** each portfolio where its build lives, submitted as it is placed with its predecessors'
  *contributions* — their order rows on the coupled side, a few kilobytes each — as dependencies, so
  the scheduler enforces the order and starts a solve the moment its last predecessor finishes. A
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

Where the work runs is a *setting*. Under `inline`, the default, the backend is this process
(`engine/backends.py`): every task runs the moment it is submitted, one after another, with exactly the
seam a cluster gives the runner — a dependency's failure reaches its dependents as a raised handle —
and no worker in between, which is where a rule is stepped through under a debugger. Otherwise it is a
Dask cluster the run owns (`engine/dask_backend.py`): a `LocalCluster` of worker processes on a laptop,
a `GatewayCluster` of pods a Dask Gateway creates, or a scheduler address. The runner drives either
through one lifetime: **start** it before the load stage, so worker processes import the solver stack
and pods come up while the loaders wait on their sources; **scale** it and **wait** for the first
worker after assembly; **probe** every worker that has joined — each resolves the config itself, which
is where a missing solver, step package, or kind surfaces, and reports its fingerprint — and stop the
run as an infrastructure failure if any cannot; **share** the assembled datasets and the config with it
once — scattered, and replicated between workers on demand — so a task carries a portfolio id and
nothing else; **submit** every build, then every solve with its dependencies; **close** it in a
`finally`. Workers re-resolve the config themselves, under the run's own step-package allowlist —
function objects are never pickled, only their names.

![The run owns its cluster: provisioning overlaps the load stage](images/cluster-lifecycle.svg)

Every task returns the **fingerprint of the process that ran it** — interpreter, numerical libraries,
solver, the versions of the packages behind external steps, the git revision, the image digest — and
the runner compares it with its own. A worker running different code fails its portfolio at stage
`worker` rather than answering, and the manifest lists every environment that did work. Under `inline`
and a local cluster the fingerprints agree by construction; on a gateway's cluster this is what makes
sharing machines safe.

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

Every portfolio ends as a `PortfolioResult` or a `PortfolioFailure` naming its stage — `load`,
`slice`, `build`, `solve`, `worker`, or `skipped` — with the exception type, its message, and its
traceback. The traceback is what makes a failure debuggable once the run is over: `slice`, `build`,
and `solve` all run in a worker process whose own stderr goes to a pod that outlives nothing, so the
formatted frames travel home on the failure and the run writes them to `failures/<portfolio_id>.txt`
beside the manifest, hashed like every other artifact. A skipped portfolio's message names what it was
waiting for: the predecessor that failed, or, under `fail_fast`, any higher-priority failure. A build
that fails has an unknown tradable set and is treated as overlapping everything after it, so under
`continue` it skips every lower-priority portfolio that reads the chain. The process exit code is
**0** when every portfolio solved and every check passed, **1** when any portfolio or check failed, **2** when the inputs were rejected before
anything was solved — invalid settings, a bad `--as-of`, a config that does not validate or resolve,
datasets that fail loading or assembly — and **3** for infrastructure: a sink failure, a cluster that
never came up, a config file that cannot be read. A failure has one command behind it:
`run CONFIG --retry-of MANIFEST` runs *this* config over exactly the portfolios the manifest
recorded as failed at the stages `--retry-stages` names (`solve` by default; `skipped` is what
`fail_fast` left behind a failure; `--retry-errors` narrows to exception types) — written inline as
the book, in their recorded solve order, the run tagged `retry_of` — with nothing carried from the
failed run but the ids. Which config is the desk's call: the same wiring with the build's
`hold_breached_starts` on for a start the order flow could not trade out of, a looser `post_solve`
or another solver for a solve that hit its limit or failed verification, a rebalance, or the
original config unchanged over the skipped tail. The retry refuses a manifest in which nothing
matches, naming what did fail.

## 11. Persist, publish, record

For each solved portfolio, the spec, solution, and chain state — its predecessors' trades on the side
the run couples through, and which predecessors — are written as `.npz` files under
`<output_dir>/<run_id>/{problem_specs,solutions,chain}/` as each result is classified, while the cluster
is still up. The solution carries the records of the constraints the step applied and the solver's
duals, so these files are what `portfolio-optimizer verify` reloads to re-check a solution without the
solver stack.

The **sink** is called exactly once, with every solved portfolio's orders concatenated and sorted, and
only when at least one portfolio solved. The shipped sinks write Parquet (Decimals as Arrow decimals) or
CSV, atomically, via a temp file and rename. A sink failure is exit code 3, but the manifest is still
written.

The **checks** run next, once, on the client, over the assembled datasets as the rules first saw them
and the orders the sink received — only when something solved, since there is nothing to prove
otherwise. Each is a configured step, `(frames, orders, solved[, params]) -> DataFrame` — the solved
portfolios being the population a rule applies to, whether or not they traded — returning one row per
case the business rule applies to with a boolean `ok`; the manifest records it under its label as
`passed`, `failed`, or `not_exercised`, and the rows that failed go to `checks/<label>.csv`. A failed
check is exit code 1; a check that raises is the run's own failure at stage `check`, with its traceback
in `failures/check.txt`, and the checks after it still run. This is the second half of the run's
proof: the verifier (§7) holds the typed constraint rows on the solved weights; a check holds a Python
rule — which only shaped the problem — on the orders that actually went out.

The **manifest** (`engine/manifest.py`) records the run id, name, tags, and timestamps, and the instant
the run was as of; the git revision and whether the tree was dirty; the schedule the run derived —
coupling, edges, components, the critical path; the Python, cvxpy, numpy, pandas, and solver versions,
and every worker environment that executed a task; the backend's lifetime — its kind, what was asked
for, when the first worker answered, when it was released; the resolved config and its hash; the
settings; every objective term as its record; every dataset's provenance and content hash; and, per
portfolio, the solve-order key, the number of predecessors, the rule audits, the constraint records it
solved under, the spec hash, the chain-input hash, the solver statistics and duals, the verification
outcome with the checks that bound and every residual signed, drift, and the orders' count, hash, and gross notional;
and every check step's outcome. The
manifest is then self-hashed, and `load_manifest` refuses one whose hash does not match its content.

`portfolio-optimizer diff-manifests` compares two manifests and names the **first stage at which they
diverge**: config, code, versions, datasets, assembly, and then per portfolio status → rules → spec →
solve → orders. "Did the data change, or did the solver?" is a one-command question. The
[reference page](reference-manifest.md) documents every field.

## The example, stage by stage

The shipped example is three configs over one book: `configs/example_inflow.json`, the desk's inflow
(`order_flow: inflow`, two `linear` terms — alpha and transaction cost), `configs/example_outflow.json`, its
outflow (`order_flow: outflow`, three — alpha, tax cost, transaction cost), and
`configs/example_rebalance.json`, the rebalance (`order_flow: rebalance`, the inflow's two terms). Each declares a hundred
accounts over three securities, seven global datasets and two loaded per account (`holdings`, a call
each with eight in flight — 200 rows in 100 batches — and `details`, twenty-five ids a call, four
batches), two assembly steps (the research store's `signals` joined into the universe, then dropped),
two rules, up to six typed constraint rows an account (530 rows in all),
the Clarabel solver under the `cvxpy` step, and `fail_fast`; where the work runs is the
`PORTFOLIO_OPTIMIZER_CLUSTER` setting, this process by default. Each run is a pure function of the
snapshot with its own manifest; nothing crosses between them. The first two accounts are the ones to
follow.

P1 and P2 each have a NAV of 1,000,000 with 400,000 in cash, and hold 3,000 A at 100 (at cost) and
6,000 B at 60 against a price of 50 — a loss — P1's lots long-term, P2's short-term. A (alpha 0.03)
and B (alpha −0.01) are `TECH` and trade a million shares a day; C (alpha 0.05, `HEALTH`, 20 bps to
trade) has the best expected return and the worst liquidity: 100,000 shares a day at 10, and the style
allows 25% participation, so a portfolio may buy at most 25,000 shares — a quarter of its NAV. P1's
single-name cap is 40%, P2's 60%; the cash band is `[0, 0.6]` and the sector bands `TECH ≥ 0.5`,
`HEALTH ≤ 0.5`.

In the inflow P1 solves first: it takes the quarter of NAV that C's budget allows, 25,000 C, and
buys 1,000 A up to its 40% cap; B's alpha has turned negative, so the remaining 50,000 stays cash — the
hand-computable optimum, to the share. P2 can buy the same securities, so it waits for P1; when it
solves, the chain state says C's budget is spent, so it buys 3,000 A to its 60% cap and the run says
why C is closed: `adv/cumulative_participation` binds. Across the book that is 54 orders in 52
accounts — 52 in A (P4 has room but sold A nine days earlier, and the blotter rule freezes it), and
C only for P1 and P3, whose 30% participation leaves 5,000 shares inside its
own budget. In the outflow neither account waits on C; each harvests B down to the `TECH` floor,
2,000 shares — P1 at the long-term rate, four cents of refund per dollar sold, P2 at the short-term
rate — and the term that rewards the sale is exact because nothing can rebuy B in the same solve.
Across the book: 38 orders in 37 accounts, 23 in B and 15 short-term loss harvests of A, with the
cumulative ADV cap on B binding from P34 on, in 45 accounts. Either way the other ninety-eight
accounts follow behind, each waiting on all the ones ahead because every account in this book can
trade the same three names — 4,950 edges, one component, critical path 100. Run a config twice and
`diff-manifests` reports `no differences`; run it with `"dependencies": "all"` and every portfolio
record is the same again (the diff names only the config). The [tutorial](tutorial-first-run.md) walks
through exactly this.

## Where validation happens, in one list

1. **Settings** — unknown environment variables; a gateway without a worker image or a password.
   Every setting has a default.
2. **Run argument** — `--as-of` is an ISO 8601 instant with a zone.
3. **Config** — strict models; money as strings; every objective record names a `kind`.
4. **Resolver** — the function exists, its signature matches the contract, its params validate; every
   term is a known kind with a unique name; the solver is known and installed; then every term is
   rendered once against a one-security dummy spec under the run's order-flow profile and the problem is
   checked for convexity, so a term that raises, reads a side the run lacks, or is not DCP is refused.
   Run by `validate-config`, at the start of `run` before a backend is asked for, and again on every
   worker before it does any work — the same checks in every process.
5. **Loaders** — dtypes declared up front; exact `Decimal` coercion.
6. **Assembly** — each step's own claims (join keys, cardinality, coverage, dtype agreement on a
   union); then the required frames exist and every frame schema holds; then details for every
   portfolio.
7. **`PortfolioData`** — cross-frame invariants including holdings/universe dtype agreement, re-run
   after every rule; a frame a rule replaces is re-validated against its schema.
8. **`ProblemSpec`** — shapes, finiteness, bound ordering, read-only arrays; then every constraint row
   parses as its kind and every row and term finds what it reads on the spec.
9. **Solve step** — weights of the right shape, status classified, infeasibility diagnosed; for the
   cvxpy step, DCP-compliant first.
10. **Verifier** — every reported constraint through its own residual, every term through its own
    value, and the objective, independently of cvxpy.
11. **Orders** — the `ORDERS` schema including `notional = quantity × price`, every order on a side the
    run trades and inside the tradable set, then the drift bound.
12. **Checks** — every configured check over the assembled datasets and the published orders, one
    row per case the rule applies to; `not_exercised` when it examined nothing.
13. **Manifest** — self-hash checked on load.
