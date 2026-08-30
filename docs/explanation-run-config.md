# Explanation: reading a run config

A run config is one JSON document that tells the engine what to load, how to combine it, which rules
and terms apply, and how to execute. This page reads through that document block by block, in the
order the shipped `configs/example_run.json` lists them, and for each block answers three questions:
what is it telling the engine, when does the engine consume it, and what changes if you set it
differently. It is the companion to two other pages: the [reference](reference-run-config.md) lists
every key with its type and default and says nothing about why, and
[the life of a run](explanation-run-lifecycle.md) follows the engine stage by stage. The README gives
[a sentence or two per block](../README.md#the-run-config-block-by-block); this page is the long
version. Read it when you have a config in front of you and want it to make sense.

## The document at a glance

| Block | What it tells the engine | When the engine consumes it |
|---|---|---|
| [`$schema`](#schema) | Nothing — it points your editor at the generated schema | Never; excluded from the config hash |
| [`run`](#run) | The run's name and tags, and the instant it is *as of* | `as_of` at load and at build (tax-lot terms) |
| [`portfolios`](#portfolios) | How to load the list of portfolio ids and their priorities | First, alone, before any other loader |
| [`datasets`](#datasets) | How to load every other input, engine-known or extra, and how its calls are partitioned | All at once, after the portfolio list |
| [`rate_limits`](#rate_limit-on-a-dataset-and-rate_limits) | Named request budgets that inputs on one backend share | During loading |
| [`assembly`](#assembly) | Steps that turn loaded datasets into the tables the build expects | Once, after all loaders return, before schema validation |
| [`rules`](#rules) | Business logic applied to each portfolio's bundle, in order | Per portfolio, on a worker, before the build |
| [`solve_order`](#solve_order) | A step that computes each portfolio's priority from its data | Per portfolio, after its rules |
| [`sides`](#sides) | Which side the run trades, and so what a trade means | At resolve, then at every build, solve, and verification |
| [`objective`](#objective) | The terms whose sum is minimized | Constructed once at resolve, then at every solve and verification |
| [`constraints`](#constraints) | Which constraints apply; their limits come from the data | Constructed once at resolve, then at every solve and verification |
| [`solve`](#solve) | The step that turns a built problem into weights | At every solve |
| [`solver`](#solver) | Which cvxpy solver, with what options and time limit | Checked at resolve, used at every solve |
| [`post_solve`](#post_solve) | How tightly the cvxpy-free verifier holds each solution | After every solve |
| [`sink`](#sink) | Where the orders go | Once, at the end, if any portfolio solved |
| [`execution`](#execution) | What one failure does to the rest, and how predecessors are chosen | When the dependency graph is derived and when a portfolio fails |

## Two passes over one document

The engine reads the config twice, and the split explains most of what follows.

The **first pass** happens before any data is touched. `config/models.py` parses the JSON strictly —
an unknown key anywhere is an error, money and weights must be strings so they become exact
`Decimal`, and `as_of` must carry a time zone. Then `config/resolve.py` takes every *step* in the
document (the loaders, the assembly steps, the rules, the solve-order step, the terms, the constraints,
the solve step, and the sink), imports the function it names, checks that its signature matches the
contract for its kind, and validates its `params` against the function's own `Params` model. It checks
the `solver` block too — the solver is known to the adapter, installed in this process, and able to
honor `time_limit_s` — and, once every step has resolved, it constructs every term and constraint once
against a one-security dummy spec under the run's side profile, so a term that raises or reads a side
the run lacks is refused here rather than on a worker. Every failure across the whole document is
collected and reported together. `portfolio-optimizer validate-config` runs exactly this pass and
stops; `run` runs it before asking for a cluster, and every worker runs it before it does any work, so
all three apply identical checks.

The **second pass** is the run itself: each block is consumed at the stage that needs it. `run.as_of`
goes to every loader and to the tax calculation; `assembly` runs once after loading; `rules` run per
portfolio; `solve`, `solver`, and `post_solve` are read once per solve; `sink` runs once at the end. So the
config is not a script the engine executes top to bottom — it is a description of a pipeline, and the
order of blocks in the file is for the reader, not the engine.

One shape recurs in nine of the sixteen top-level keys: a **step**. A step is either a bare string
naming a function, or an object with `name` and `params`:

```json
"add_zero_alpha"
{"name": "restrict_low_liquidity", "params": {"min_adv_shares": 1000}}
```

A bare name is looked up in the template module for that kind of step — `loaders.py` for loaders,
`assembly.py` for assembly steps, `rules.py` for rules, `solve_order.py` for the solve-order step,
`terms.py` for terms and constraints, `solvers.py` for the solve step, `sinks.py` for sinks. A qualified name
such as `mypkg.rules:my_rule` is imported from anywhere the engine (and any worker process) can
import. Because the resolver reads the function's `params` annotation, the JSON Schema knows the exact
parameter shape of every shipped step and rejects a typo before the engine ever runs.

## `$schema`

```json
"$schema": "./run-config.schema.json"
```

This is for your editor, not the engine. Editors that honor `$schema` validate the file as you type
and complete key names and step names from the generated schema. The engine accepts the key and
ignores it, and the config hash recorded in the manifest excludes it, so adding or removing the
pointer never makes two runs look different.

## `run`

```json
"run": {"name": "example_rebalance", "as_of": "2026-08-28T00:00:00Z", "tags": {"desk": "template"}}
```

`name` and `tags` are identity: they are copied into the manifest and used for nothing else. Pick a
name that will still mean something when you are comparing two manifests a month later.

`as_of` is the one field here that changes results. It is the moment the run is *as of*, and it is
threaded through the whole pipeline: every loader receives it as `request.as_of` so a source can be
asked for data at that instant; the build uses it to decide whether each tax lot is long- or
short-term (held longer than 365 days as of this timestamp means the long-term rate applies); every
order row carries it; and the manifest records it. It must be timezone-aware, because a naive
timestamp compared against a lot's `acquired_on` would be a silent off-by-hours bug.

## `portfolios`

```json
"portfolios": {"name": "csv", "params": {"path": "portfolios.csv"}}
```

This is the one loader that runs alone, before everything else. Its frame has a `portfolio_id` column
and an optional `solve_order`, and the engine sorts it by `solve_order` then `portfolio_id` to produce
the tuple of ids the rest of the run starts from: it is included in every other dataset's request so a
loader for a per-portfolio source knows exactly which ids to fetch. That dependency is why this loader
cannot overlap with the others.

The block is written here as a bare step because a file needs no throttling. A source that does can
use the longer form, `{"loader": step, "rate_limit": ...}`, which is the same shape every entry in
`datasets` has; the bare form is a convenience the model expands.

`solve_order` is a *priority*, not a sequence: lower solves first, ties break on `portfolio_id`, and it
matters only when a term or constraint is chain-aware — when a later portfolio's problem depends on what
higher-priority ones already *traded* on the side the run couples through. A portfolio waits only for
higher-priority portfolios that can trade a security it can trade too, on that side; everything else
solves concurrently. In the example, P2 solves after P1 and
finds that P1 has consumed the ADV budget for security C. Swap the `solve_order` values in the data and
P2 gets the budget instead. A [`solve_order` step](#solve_order) computes the key from the data instead
of reading this column.

## `datasets`

```json
"datasets": {
  "holdings":    {"loader": {"name": "csv", "params": {"path": "holdings.csv"}}},
  "universe":    {"loader": {"name": "csv", "params": {"path": "universe.csv"}}},
  "details":     {"loader": {"name": "csv", "params": {"path": "details.csv"}}},
  "constraints": {"loader": {"name": "json_constraints", "params": {"path": "constraints.json"}}},
  "targets":     {"loader": {"name": "csv", "params": {"path": "targets.csv"}}},
  "prices":      {"loader": {"name": "csv", "params": {"path": "prices.csv", "decimal_columns": ["price"]}}}
}
```

Each key is a dataset name and each value says how to load it. The names fall into three groups, and
the engine treats them differently.

**Five names are required**, because the build cannot produce a problem without them: `holdings`
(what each portfolio owns, with cost basis and acquisition date), `universe` (every security the
portfolio may buy, with its sector, ADV, lot size, and restricted flag), `details` (per-portfolio NAV,
cash, tax rates, and benchmark), `targets` (per-benchmark target weights), and `constraints`
(per-portfolio style limits). `constraints` must always be declared here. The four frames must be
declared here too unless the config has assembly steps, in which case a step may produce them — two
custodians' files stacked into one `holdings`, say — and their presence is checked after assembly
instead. Each frame is validated against a fixed schema after assembly — column set, dtypes,
nullability, bounds, unique key, and invariants such as "target weights sum to one" — with one
deliberate opening: `holdings` and `universe` accept any columns beyond their schemas, because that is
where security analytics go. `constraints` is the odd one out: its loader returns a mapping of
portfolio id to a style-constraint object rather than a frame, which is why it has its own step kind
and its own shipped loader, `json_constraints`.

**Any other name is an extra dataset.** The engine knows nothing about its columns. It is visible to
every assembly step by name, and whatever is still present after the last step is carried into each
portfolio's bundle as `data.extras` — reduced to that portfolio's rows when it has a `portfolio_id`
column, passed whole otherwise — where a rule can use it. The example's `prices` is one: the universe
schema requires a `price` column, but the example's universe file does not carry it, so prices arrive
as a separate file, are joined in by the first assembly step, and are dropped by the second so they are
not carried further. Because the engine cannot type an extra frame from a schema, the loader has to be
told: `dtypes` makes `security_id` a `string` key and `decimal_columns` makes `price` an exact
`Decimal` rather than a float.

Once the portfolio list is known, **every dataset loader starts at once**, called with a `LoadRequest`
carrying the dataset name, portfolio ids, `as_of`, the data root, the run id, and a rate limiter. An
`async def` loader runs on the event loop; a plain `def` loader runs in a worker thread so a blocking
driver cannot stall the others. Each loaded frame is recorded in the manifest with its loader, params
hash, row count, and an order-insensitive content hash, which is what lets `diff-manifests` say "the
data changed" rather than "something changed".

**How many times each loader is called is `scope`.** A `global` dataset — the default, and every
dataset above — is one call for the whole book, and is what the assembly steps see. A `per_portfolio`
dataset is the engine's fan-out: the ids are cut into batches of `batch_size` and the loader is called
once per batch, sharing the dataset's one rate limiter. That is the arrangement for a source that
answers one account at a time, and it buys three things a loader that fans out privately cannot give
you — the batches are visible in the manifest, they overlap the global loaders, and a failure is
isolated. The cost is that assembly, which runs over whole datasets, never sees such a dataset; attach
its columns in a rule instead.

**A structural problem rejects the run; a coverage problem fails a portfolio.** A required dataset
missing, a schema violated, a global loader that raised, or a per-portfolio dataset no batch of which
came back — the run stops, because nothing can be built. One batch that raised, or a portfolio with no
`details` row or no `constraints` entry — that portfolio alone is recorded as failed at stage `load`
and the rest of the book runs. A portfolio rejected here never entered the run, so it traded nothing
and couples to nobody in the schedule. Loading failures are otherwise collected rather than raced: one
failing dataset does not cancel the others, and all of them are reported together.

The shipped loaders show the two shapes a source can take. `csv` and `parquet` read one file for the
whole dataset. `csv_per_portfolio` reads `<directory>/<portfolio_id>.csv` for every id in the request,
concurrently, under the dataset's rate limiter — the shape of a loader for an API that answers one
portfolio per call, with a file read standing in for the HTTP request. It works under either scope: as
a `global` dataset it owns its fan-out, and with `"scope": "per_portfolio", "batch_size": 1` the engine
owns it and each call reads one file.

### `rate_limit` on a dataset, and `rate_limits`

Every entry in `datasets` (and `portfolios`) accepts an optional `rate_limit`, which the loader
receives as `request.rate_limiter` and wraps around each call to its backend: either an inline bound
private to that input, or the name of a pool declared under the top-level `rate_limits` and shared by
every input that names it. Omit it and the loader never waits; the example sets none because it reads
local files, and has no `rate_limits` block for the same reason. The choice between the two spellings
is about sharing — a pool is a property of the *backend*, not of any one input, which is why pools are
declared at the top level — and the reasoning is in
[the architecture explanation](explanation-architecture.md#loading-is-the-slow-part-so-it-is-concurrent-and-metered).
The keys of a bound (`requests_per_second`, `burst`, `max_in_flight`) and their defaults are in
[the reference](reference-run-config.md#rate-limits); wiring a fan-out loader to one is in
[how to add a loader](how-to-add-a-loader-or-sink.md#async-loaders-fan-out-and-rate-limits).

## `assembly`

```json
"assembly": [
  {"name": "join", "params": {"into": "universe", "source": "prices", "on": ["security_id"],
                              "cardinality": "one_to_one", "require_all_matched": true}},
  {"name": "drop", "params": {"datasets": ["prices"]}}
]
```

`assembly` is how separately loaded datasets become the tables the build expects, and it is a list of
steps like `rules` — the same convention, applied once per run to all the data rather than once per
portfolio to one bundle. Each step is a function `(frames: Frames[, params]) -> Frames` that sees every
loaded dataset by name and returns the new set; the shipped ones live in `assembly.py`, and a desk's
own live in its package. The list runs after every loader has returned and before the engine-known
frames are validated against their schemas, which is what lets a step *supply* a required column, as
the example's join supplies `price`.

The four shipped steps are the shapes that recur: `join` brings columns from one dataset into another,
`union` stacks datasets with the same meaning into one, `select` trims and renames, `drop` discards.
Read the example's join as a set of claims about the data, each of which the engine checks:

- `cardinality: one_to_one` claims each security appears once on both sides; pandas enforces it, so a
  duplicated price row aborts the run instead of silently doubling a universe row.
- `require_all_matched: true` claims every universe security has a price; an unmatched key is reported
  by example and the run is rejected.
- A brought column that the target already has is refused unless `overwrite` is set, so a stale
  column is never silently replaced — or silently kept.

The key columns' dtypes are aligned to the target before merging so a `str` key never joins to a
`string` key as `object` and matches nothing. The `drop` afterwards is a courtesy to memory: any dataset
still present after the last step is carried into every portfolio's bundle, which is exactly right for
a per-portfolio exclusion list a rule will read, and wasteful for a price file that has done its job.

Most real assembly lists are longer than the example's and mostly about one thing: attaching
per-security analytics to `holdings` and `universe`. Both tables accept any columns beyond their
schemas, and the two are later stacked into a single optimizer frame, so a column attached to both must
have the same dtype on both — the bundle refuses otherwise, naming the column.
[How to add security analytics](how-to-add-security-analytics.md) walks through that work; the
manifest records every step's source hash, parameters, row counts, and the columns it added.

## `rules`

```json
"rules": [{"name": "restrict_low_liquidity", "params": {"min_adv_shares": 1000}}, "add_zero_alpha"]
```

Rules are the business-logic layer: functions that take one portfolio's assembled data bundle and
return a modified bundle. They run per portfolio, in the order listed, after slicing and before the
build. The bundle they receive is already validated, and the only way to return a changed one is
`with_changes(...)`, which re-runs every cross-frame check — so a rule can tighten a cap, freeze a
name, or add a column, but cannot hand the optimizer something inconsistent.

The example's two rules show the spectrum. `restrict_low_liquidity` reads its threshold from `params`
and marks names below it restricted, which the build then freezes at their current weight.
`add_zero_alpha` takes no parameters and adds an `alpha` column of zeros when the universe has none, so
the `alpha` term could be enabled without changing the data. The shipped `cap_single_name` tightens the
style's `max_weight` when the style's own is looser.

Ordering is meaningful: a rule sees the output of the one before it. A rule never sees other
portfolios — it runs in a worker, on one bundle, before anything is solved — and that is what lets
every portfolio build at once. A rule that shrinks the portfolio's *tradable set* — freezing a name,
or, in a run that couples through buys, capping it at its current weight — also shrinks the set of
portfolios this one has to wait for; see `execution`.

## `solve_order`

```json
"solve_order": "furthest_from_target_first"
```

The example does not set this — its portfolios file carries the priority — but a real book usually
should. A solve-order step is `(data: PortfolioData[, params]) -> Decimal`, run on each portfolio's
ruled bundle in the worker that built it; lower keys solve first and ties break on `portfolio_id`. It
answers "who gets first pick of a shared budget" from the data — the shipped step puts the portfolio
furthest from its target first — instead of from a hand-maintained column, and it is part of the
config hash, so two runs with different priorities are visibly different runs.

## `sides`

```json
"sides": "both"
```

Which side the run trades: `both`, the default, is the two-sided problem; `buy` and `sell` are
one-sided, a third the size, with the trade an expression of the one variable `w`. The value selects
the *side profile*, the one object in the engine that knows what a side means, and it fixes which side
portfolios couple through — buys under `both` and `buy`, sells under `sell`. Two things follow for the
rest of the config: a term or constraint that reads a side the run lacks (the shipped `tax_cost` reads
`sell`) is refused at `validate-config`, and `cash_bounds` keeps its meaning as the cash *after* the
run while the side fixes the direction cash can move. [How to run one side](how-to-run-one-side.md)
walks through both; [the architecture explanation](explanation-architecture.md#the-side-a-run-trades-is-one-object)
covers what the profile owns and why the side is a config value rather than a pair of bounds.

## `objective`

```json
"objective": {
  "sense": "minimize",
  "terms": [
    {"name": "tracking_error", "params": {"weight": "1.0"}},
    {"name": "tax_cost", "params": {"weight": "1.0"}},
    {"name": "transaction_cost", "params": {"weight": "1.0"}}
  ]
}
```

The objective is the sum of the listed terms, and the engine only ever minimizes. That single sense is
deliberate: mixing rewards and costs in one list is easy to get wrong, so a reward is written as a
negative term (the shipped `alpha` term is `−weight · alphaᵀw`) and everything else is a cost. Every
shipped term takes a `weight`, written as a string so it is an exact `Decimal` in the manifest even
though the solver ultimately sees a float.

Each term is called once per solve with the decision variables (`w`, and `buy` and `sell` on the sides
the run has, all fractions of NAV; `trade` is the amount traded on the sides it has) and the numeric
`ProblemSpec` the build produced, and returns a convex expression. The verifier
later recomputes every shipped term in numpy and compares the sum with what the solver reported, which
is why the weights here are the only tuning knobs on the objective: the shape of each term is fixed in
code so that its numpy twin stays in step with it.

The example's three terms make the solver trade off closeness to the benchmark against the taxes and
trading costs of getting there. One term carries a condition worth knowing: `tax_cost` refuses to run
when losses could be harvested but nothing charges for trading (no `transaction_cost` term with a
positive `cost_bps` and no `tcost_bps` column), because that combination lets the solver sell and
rebuy a name for free.

## `constraints`

```json
"constraints": ["long_only", "max_weight", "cash_bounds", "turnover_cap",
                "sector_bounds", "cumulative_adv_participation"]
```

This list says *which* constraints apply; *how tight* they are comes from the data. `max_weight`,
`cash_bounds`, `turnover_cap`, `sector_bounds`, and `cumulative_adv_participation` each read their
limits from the portfolio's style object in the `constraints` dataset (`max_weight`, `cash_bounds`,
`max_turnover`, `sector_bounds`, `max_adv_participation`), tightened where the universe carries
per-security `min_weight`/`max_weight` columns or a restricted flag. That split is the reason the
config almost never changes between daily runs while the numbers inside the data do: the config is
wiring, the data is policy.

The trade identity is not on this list, and cannot be. What `buy` and `sell` *mean* — for a two-sided
run, `w − w0 = buy − sell`, both non-negative, `sell ≤ w0`; for a one-sided run, `w ≥ w0` or `w ≤ w0`
with the trade an expression of `w` — is what every cost term, the turnover cap, the ADV constraint,
and the verifier's identity checks rely on, so it comes from `sides` and is added to every solve; a
config that still names `trade_balance` is refused at resolve with a message saying so.

Each constraint may carry a `label`, unique among the run's constraints and defaulting to the bare
name; the verifier's report and the manifest key on it, which is what tells two instances of one
function apart. A `kind` key names what sort of constraint it is — `function`, a step, is the only
kind today; it is the seam on which constraint models that are not functions will be added.

Most constraints take no parameters, and the JSON Schema enforces that: `{"name": "long_only",
"params": {"x": 1}}` is rejected by your editor. `sector_bounds` is the exception with a `tolerance`
that loosens every sector band symmetrically; with an empty `sector_bounds` map in the style, it
contributes nothing. `cumulative_adv_participation` declares `chain: ChainState` — it needs to know how
much of each name's ADV budget higher-priority portfolios have already *traded* on the side the run
couples through — which makes it chain-aware, and its presence is the only reason any portfolio waits
for another. It writes two rows: `trade ≤ adv_capacity` for the portfolio's own participation, and
`coupled ≤ remaining` where predecessors' trades on the coupled side have consumed part of the budget.
The other side, where the run has one, is the portfolio's own business.

Every shipped constraint has a numpy twin in the verifier, looked up by qualified name, so the
post-solve check is a genuine second opinion for each one you list. A custom constraint without a twin
is accepted and reported as `unverified` in the manifest rather than refused.

## `solve`

```json
"solve": "cvxpy"
```

Which step decides the weights. The engine builds each portfolio's problem as data, folds the chain,
and hands one function everything it may use — the spec, the chain, the side profile, the resolved
terms and constraints, the `solver` block — and takes back weights. `cvxpy`, the default, is the
optimizer: it builds the cvxpy problem from the terms and constraints and solves it. `pro_rata_fill`
is the other shipped step and is not an optimizer at all — a numpy function that spends the cash on
the underweights — which is the point of the key: what a desk does on one side is sometimes not an
optimization, and dressing it as one costs seconds of solver time for an answer a function computes in
milliseconds. A qualified name plugs in a firm's own library that builds the problem its own way.

Whatever the step is, the engine treats its answer the same way: the side profile turns the weights
into a trade, the verifier re-checks every shipped constraint against them, rounding and drift run
unchanged, and the manifest records the step and its version where it records the solver's. A step
that minimized nothing reports no objective; the verifier then skips the objective comparison and
still evaluates the configured terms as a report line, so a heuristic and the optimizer can be
compared on one book. The guarantees are the verifier's, not the step's — see
[how to replace the cvxpy solve](how-to-write-a-solve-step.md).

## `solver`

```json
"solver": {"name": "CLARABEL", "options": {"max_iter": 200}, "time_limit_s": 60.0, "verbose": false}
```

`name` selects a solver. It must be one the adapter has a record for — `CLARABEL`, `OSQP`, `SCS`,
`HIGHS`, which cvxpy installs, or `PIQP`, the `piqp` extra — and it must be installed, and both are
checked when the config *resolves*, not when the first portfolio solves: `validate-config` rejects a
typo, `run` rejects it before asking for a cluster, and every worker checks its own image before it
does any work. A solver cvxpy can see but the adapter has no record for is refused too, because the
record is what names the distribution whose version goes into the environment fingerprint; without it
two different builds of the solver would compare equal. There is no automatic fallback: a run
configured for one solver never silently produces answers from another, because the manifest records
the solver and its version as part of what makes a run reproducible. `options` is passed verbatim to
`Problem.solve`; the engine does not interpret it, so what is valid depends entirely on the solver.
`time_limit_s` is the one option the engine does translate, because every solver spells it differently
— `time_limit` for Clarabel, OSQP, and HiGHS, `time_limit_secs` for SCS — and a solver with no such
option (`PIQP`) rejects the setting at resolve rather than guessing. `verbose` turns on the solver's
own iteration log, which is the first thing to enable when a solve is slow or hits its limit.

Adding a solver is one row in the adapter's table (name, distribution, time-limit option) and one
extra in `pyproject.toml`, so that "install the solver" is the same `uv sync --extra` on a laptop and
in the worker image.

## `post_solve`

```json
"post_solve": {"violation_tol": 1e-6, "objective_rel_tol": 1e-5, "objective_abs_tol": 1e-9}
```

After every solve the engine re-checks the solution in numpy without cvxpy: the side profile's identity
checks (under `both`, the trade balance and the complementarity of `buy` and `sell`), each constraint's
violation, the recomputed objective against the solver's reported one, and finiteness. These three
numbers are its tolerances — `violation_tol` bounds every residual, identity and constraint alike, and
the other two bound the objective gap — and they are JSON numbers rather than strings
because they are float tolerances on float arithmetic. The defaults are deliberately about a hundred
times looser than a solver's convergence tolerance so that a pass says something about the solution
rather than restating the solver's own stopping criterion. Tighten them and a solver that is merely
converged can start failing verification; loosen them and the check stops meaning much. The example
writes the defaults out explicitly.

## `sink`

```json
"sink": {"name": "orders_to_parquet", "params": {"subdir": "orders"}}
```

The sink is where the orders go, and it is called exactly once per run, after every portfolio has
finished, with all solved portfolios' orders concatenated and sorted — and only if at least one
portfolio solved. Everything before it is pure; this is the one step with side effects downstream of
the engine, which is why it is a single step and not a list. The shipped sinks write one file under
`<output_dir>/<run_id>/<subdir>/`, atomically via a temporary file and rename, and return the path as
an artifact the manifest records. A sink that raises is exit code 3 and the manifest is still written,
so the run's evidence survives a failed handoff.

## `execution`

```json
"execution": {"on_error": "fail_fast"}
```

This block answers two questions: what one failed portfolio does to the rest, and which higher-priority
portfolios a portfolio waits for. It deliberately does *not* say where the work runs or how many workers
there are — those are settings (`PORTFOLIO_OPTIMIZER_CLUSTER`, `PORTFOLIO_OPTIMIZER_MAX_WORKERS`, and the
other cluster variables), so a laptop run and a cluster run of one config hash identically and differ
only in the manifest's `settings` block, where `diff-manifests` can name the difference. The block may
be omitted entirely; both keys have defaults.

There is no schedule to choose. The engine derives who waits for whom from each portfolio's solve-order
key and its *tradable set* on the side the run couples through, and the answer never depends on the
schedule: each solve sees only what its overlapping predecessors traded there, which is a function of
the data. [The architecture explanation](explanation-architecture.md#a-run-couples-through-its-one-side-so-the-schedule-is-a-graph)
makes that argument; [the life of a run](explanation-run-lifecycle.md#9-the-dependency-graph-and-where-the-work-runs)
shows the mechanism in execution order.

`dependencies` has one non-default value, `"all"`, under which every higher-priority portfolio is a
predecessor — one line. It gives the same orders and the same chain hashes and exists for diagnosis:
rerun a suspicious batch as a line and `diff-manifests` the two (it names the config, because the
field is part of it, and nothing else).

`on_error` decides what one failed portfolio does to the rest. `fail_fast` records every lower-priority
portfolio as `skipped` — whatever it had finished, so the manifest never depends on timing. `continue`
isolates the failure: only the portfolios that depended on it are skipped, each naming the predecessor
that failed, and a portfolio that shared no tradable security with it is unaffected. A build that fails
has an unknown tradable set and is treated as overlapping every lower-priority portfolio. The cluster
and its worker count are about throughput, never about output;
[how to run on a cluster](how-to-run-on-a-cluster.md) covers the settings.

## What the config does not decide

Seeing what is absent sharpens the picture of what the config is for.

**Numbers live in the data.** Position sizes, prices, tax rates, target weights, single-name caps,
turnover limits, sector bands, ADV participation — none of these appear in the config. It names the
datasets that carry them and the steps that read them. The practical consequence is that the same
config can run every day against fresh data, and `diff-manifests` can tell you whether a change in
orders came from the data or from the wiring.

**Behavior lives in code.** The config chooses functions by name and tunes them through `params`; it
cannot express a new constraint or a different cost model. That is the boundary the resolver enforces:
if a step needs a shape the shipped ones do not have, you write a function with the right signature and
name it, and the [how-to guides](how-to-add-a-term.md) cover that.

**Some tolerances are derived, not configured.** The rounding-drift bound — how far executed weights
may deviate from solved weights after shares are rounded — is computed from the priciest lot and the
dust threshold in the data, so there is no knob for it. The only tolerances you set are the verifier's,
in `post_solve`.

**The schedule is derived, not configured.** Which portfolios wait for which follows from the steps
(does anything read the chain?) and the data (which securities can each portfolio trade on the side the
run couples through?). The config has no execution mode; the only schedule knob is the diagnostic
`dependencies: "all"`.

Put together: the config is the wiring of a pipeline — which inputs, combined how, filtered by which
rules, prioritized how, optimized against which terms and constraints, solved with what, checked how
tightly, delivered where. Everything in it is either a name the resolver can check before data loads or a
value with a declared type and range, and that is what lets `validate-config` promise that a config
which passes will not fail for a configuration reason later.
