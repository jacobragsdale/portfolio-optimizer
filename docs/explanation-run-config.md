# Explanation: reading a run config

A run config is one JSON document that tells the engine what to load, how to combine it, which rules
and terms apply, and how to execute. This page reads through that document block by block, in the
order the shipped `configs/example_inflow.json` lists them — its sibling `configs/example_outflow.json` is
the same document with the other `order_flow`, its own `run.name`, and one more objective term — and for
each block answers three questions: what is it telling the engine, when does the engine consume it,
and what changes if you set it differently. It is the companion to two other pages: the [reference](reference-run-config.md) carries what the
generated JSON Schema cannot — step signatures, load-time behaviour, the constraint rows and style
limits that live in data — and says nothing about why, and
[the life of a run](explanation-run-lifecycle.md) follows the engine stage by stage. The README annotates
[the example block by block](../README.md#the-run-config-block-by-block); this page is the long
version. Read it when you have a config in front of you and want it to make sense.

## The document at a glance

| Block | What it tells the engine | When the engine consumes it |
|---|---|---|
| [`run`](#run) | The run's name and tags — identity for people | Copied into the manifest |
| [`datasets`](#datasets) | How to load every input — the portfolio list included — what each depends on, and how its calls are partitioned | Each dataset the moment its dependencies are loaded |
| [`assembly`](#assembly) | Steps that turn loaded datasets into the tables the build expects | Once, after all loaders return, before schema validation |
| [`rules`](#rules) | Business logic applied to each portfolio's bundle, in order | Per portfolio, on a worker, before the build |
| [`solve_order`](#solve_order) | A step that computes each portfolio's priority from its data | Per portfolio, after its rules |
| [`order_flow`](#order_flow) | Whether the run is an inflow, an outflow, or a rebalance, and so what a trade means | At resolve, then at every build, solve, and verification |
| [`build`](#build) | The step that turns a ruled bundle into a problem | Per portfolio, after its rules |
| [`objective`](#objective) | The typed terms whose sum is minimized | Parsed and rendered once at resolve, then at every solve and verification |
| [constraints](#constraints-are-not-a-config-block-at-all) | *Not a config block* — a loaded per-portfolio dataset of typed rows | Sliced per portfolio, adjusted by rules, parsed at build, rendered at every solve |
| [`solve`](#solve) | The step that turns a built problem into weights, and its own parameters — the cvxpy solver among them | At every solve |
| [`post_solve`](#post_solve) | How tightly the cvxpy-free verifier holds each solution | After every solve |
| [`sink`](#sink) | Where the orders go | Once, at the end, if any portfolio solved |
| [`execution`](#execution) | What one failure does to the rest, and how predecessors are chosen | When the dependency graph is derived and when a portfolio fails |

One thing the document does not say is *when* the run is: the instant it is as of is an argument,
`run --as-of 2026-08-28T00:00:00Z`, threaded through the whole pipeline — every loader receives it as
`request.as_of_date`, the build uses it to decide whether each lot is long- or short-term, every order
row carries it, and the manifest records it. It must carry a time zone, because a naive timestamp
compared against a lot's `acquired_on` would be a silent off-by-hours bug. Keeping it out of the
document is what lets one wiring run every day under one config hash, so `diff-manifests` compares
Monday with Tuesday and blames the data, not the config.

## Two passes over one document

The engine reads the config twice, and the split explains most of what follows.

The **first pass** happens before any data is touched. `config/models.py` parses the JSON strictly —
an unknown key anywhere is an error, and money and weights must be strings so they become exact
`Decimal`. Then `config/resolve.py` takes every *step* in the document (the loaders, the assembly
steps, the rules, the solve-order step, the build step, the solve step, and the sink), imports the
function it names, checks that its signature matches the contract for its kind, and validates its
`params` against the function's own `Params` model; it parses every objective term as the kind its
record names; under the shipped `cvxpy` step it checks the solver named in that step's params — known
to the adapter, installed in this process, and able to honor `time_limit_s` — and, once every step
has resolved, it renders every term once against a one-security dummy spec under the run's side
profile and checks the problem is convex, so a term that raises, reads a side the run lacks, or is not
DCP is refused here rather than on a worker. Constraints are not checked in this pass at all: they
are loaded data, and each portfolio's rows are parsed when it builds. Every failure across the whole
document is collected and reported together. `portfolio-optimizer validate-config` runs exactly this
pass and stops; `run` runs it before asking for a cluster, and every worker runs it before it does
any work, so all three apply identical checks.

The **second pass** is the run itself: each block is consumed at the stage that needs it. `assembly`
runs once after loading; `rules` and `build` run per portfolio; `solve` and `post_solve` are read once
per solve; `sink` runs once at the end. So the config is not a script the engine executes top to bottom
— it is a description of a pipeline, and the order of blocks in the file is for the reader, not the
engine.

One shape recurs in eight of the top-level keys: a **step**. A step is either a bare string naming a
function, or an object with `name` and `params`:

```json
"add_zero_alpha"
{"name": "restrict_low_liquidity", "params": {"dataset": "buy_universe_parameters", "key": "min_adv_shares"}}
```

A bare name is looked up in the template module for that kind of step — `loaders.py` for loaders,
`assembly.py` for assembly steps, `rules.py` for rules, `solve_order.py` for the solve-order step,
`engine/build.py` for the build step, `solvers.py` for the solve step, `sinks.py` for sinks — and then
among the steps installed packages publish as entry points in the group `portfolio_optimizer.<kind>`,
which is how a firm shares a loader or a rule across desks and names it bare. A qualified name such as
`mypkg.rules:my_rule` is imported from anywhere the engine (and any worker process) can import — or,
when the `PORTFOLIO_OPTIMIZER_STEP_PACKAGES` setting names an allowlist, from those packages alone.
Because the resolver reads the function's `params` annotation, the JSON Schema knows the exact
parameter shape of every step the environment can name and rejects a typo before the engine ever runs.
Terms and constraints are the other shape: not functions but *kinds*, strict pydantic models a record
names by `kind`, covered below.

## `run`

```json
"run": {"name": "example_inflow", "tags": {"desk": "template"}}
```

`name` and `tags` are identity: they are copied into the manifest and used for nothing else, and they
are kept out of the config hash, so renaming or relabelling a run does not make it a different wiring.
Pick a name that will still mean something when you are comparing two manifests a month later. The
shipped outflow is `example_outflow`: one book, the other order flow, its own manifest.

## `datasets`

```json
"datasets": {
  "portfolios":  {"loader": "load_portfolios"},
  "holdings":    {"loader": "load_holdings", "scope": "per_portfolio", "batch_size": 1,
                  "max_in_flight": 8},
  "universe":    {"loader": "load_universe"},
  "details":     {"loader": "load_details", "scope": "per_portfolio", "batch_size": 25,
                  "max_in_flight": 4},
  "constraints": {"loader": "load_constraints", "depends_on": ["portfolios"]}
}
```

### `portfolios`: the book of record

`portfolios` is the one entry every run must declare. Its frame has a `portfolio_id` column and an
optional `solve_order`, and the engine sorts it by `solve_order` then `portfolio_id` to produce the
tuple of ids the run is over. It is consumed by the engine — assembly never sees it — but scheduled
like any other dataset: nothing waits on it except the entries that ask for its ids by naming it in
`depends_on`. A manager who keeps a fixed book can skip the loader and write the ids straight into
the config:

```json
"portfolios": ["P7", "P2", "P9"]
```

The written order is the solve order — the engine records `solve_order` as each id's position — the
list costs nothing to load, so every dependent starts at once, and the manifest hashes the literal
ids where it would hash a loader's source.

`solve_order` is a *priority*, not a sequence: lower solves first, ties break on `portfolio_id`, and it
matters only when something reads the chain — when a later portfolio's problem depends on what
higher-priority ones already *traded* on the side the run couples through. A portfolio waits only for
higher-priority portfolios that can trade a security it can trade too, on that side, and only where
its own constraint rows say they read the chain; everything else solves concurrently. In the inflow,
P2 solves after P1 and finds that P1 has consumed the ADV budget for security C. Swap the
`solve_order` values in the data and P2 gets the budget instead. A [`solve_order` step](#solve_order)
computes the key from the data instead of reading this column.

### The other names, and what each entry says

Each key is a dataset name and each value says how to load it. Beyond the book, the names fall into
three groups, and the engine treats them differently.

**Three names are required**, because the build cannot produce a problem without them: `holdings`
(what each portfolio owns, with cost basis and acquisition date), `universe` (every security the
portfolio may buy, with its price and — optionally — sector, ADV, lot size, restricted flag, alpha,
transaction cost, and whatever per-security analytics the terms read), and `details` (per-portfolio
NAV, the single-name cap, and the dust threshold, which the engine reads; optionally the cash, the
tax rates, the participation, the turnover cap and cash bounds, which reach the spec only as
scalars a constraint row names, so an account master that lacks one leaves it out; and any further
column the desk keeps on an account). They must be declared here unless the config has assembly steps, in which case a step may
produce them — two custodians' files stacked into one `holdings`, say — and their presence is checked
after assembly instead. Each frame is validated against a fixed schema after assembly — column set,
dtypes, nullability, bounds, unique key, and cross-column invariants — with one deliberate opening: all
three accept any columns beyond their schemas, because that is where security analytics and account
limits go, and the build exports every one of them by name.

**`constraints` is engine-known but optional**, and unlike the three above the engine knows only which
portfolio each row belongs to and, through its `kind`, what the row declares. A run that declares no
such dataset is constrained by nothing beyond the trade identity its side implies and the spec's own
bounds. The section below explains why it is data rather than a config block.

**Any other name is an extra dataset.** The engine knows nothing about its columns. It is visible to
every assembly step by name, and whatever is still present after the last step is carried into each
portfolio's bundle as `data.extras` — reduced to that portfolio's rows when it has a `portfolio_id`
column, passed whole otherwise — where a rule can use it, and on past the build to the solve step as
`request.extras`. The engine cannot type an extra frame from a schema it does not have, so its loader
types it: the shipped `load_parameters` declares a two-column `FrameSchema` of its own — `name` a
`string`, `value` a `decimal` so it arrives as an exact `Decimal` rather than a float — in the same
vocabulary the engine's own schemas are written in, and casts what it fetched to it.

Two shapes recur. A vendor's **per-security analytics** file is declared here, joined onto the universe
by an assembly step, and then dropped so it is not carried into every bundle for nothing; from the
universe the build exports it as a spec column a term can read. **Runtime parameters** are the other:
a narrow `name`/`value` frame of numbers that change without the config changing.

```json
"trades": {"loader": "load_trades", "depends_on": ["portfolios"]},
"global_parameters": {"loader": "load_parameters"},
"buy_universe_parameters": {"loader": "load_parameters"}
```

The example declares three. `trades` is a third shape, a **per-account record** with a `portfolio_id`
column: the desk's blotter, which the engine reduces to each account's rows on the way into its
bundle, where `restrict_recent_trades` reads it. The parameter sets are two, both served by one loader that fetches the set named by the dataset itself.
`buy_universe_parameters` holds the `min_adv_shares` that `restrict_low_liquidity` reads, which is why
that rule takes no number in the config; `global_parameters` holds settings for a solve step, and
nothing shipped reads it — the `cvxpy` step has no business interpreting a desk's own settings. Both
are content-hashed and recorded in the manifest like every other input, which is the point: a run
driven by parameters is still a pure function of a snapshot, and `diff-manifests` says so when one
changes.

**Every dataset loads as early as its dependencies allow.** An entry that declares nothing starts the
moment the load stage does — in the example that is `portfolios`, `universe`, and both parameter sets,
so the security-master scan, the slowest input by an order of magnitude, no longer waits behind the
book of record. An entry that names other datasets in `depends_on` starts when they have loaded and
receives their frames as `request.inputs`; declaring `portfolios` is what fills
`request.portfolio_ids`, which is why `constraints` declares it — its loader fetches the book, not the
firm. The whole stage costs its longest chain rather than its sum. Each loader is called with a
`LoadRequest` carrying the dataset name, portfolio ids, the input frames, `as_of_date`, the data root,
and the run id. An `async def` loader runs on the event loop; a plain `def` loader
runs in a worker thread so a blocking driver cannot stall the others. Each loaded frame is recorded in
the manifest with its loader, params hash, row count, dependencies, start offset, and an
order-insensitive content hash, which is what lets `diff-manifests` say "the data changed" rather than
"something changed".

**How many times each loader is called is `scope`.** A `global` dataset — the default, so every entry
above that says nothing about scope — is one call for the whole book, and is what the assembly steps
see. A `per_portfolio` dataset is the engine's fan-out: the ids are cut into batches of `batch_size`
and the loader is called once per batch, at most `max_in_flight` of them at a time. That is the
arrangement for a source that answers one account at a time, and it buys three things a loader that
fans out privately cannot give you — the batches are visible in the manifest, they overlap the global
loaders, and a failure is isolated. The cost is that assembly, which runs over whole datasets, never
sees such a dataset; attach its columns in a rule instead.

The example is deliberately mixed, because a real book is. `holdings` and `details` are the two
account-shaped inputs — a custodian says what an account owns, an account master says its NAV, cash,
tax rates, and style limits — so both are `per_portfolio`, and their `batch_size` records the difference
between the two kinds of source. `holdings` asks for `1`: a call per account, for a backend that
answers one at a time, which on the shipped hundred-account book is a hundred calls. `details` asks for
`25`: the engine hands its loader twenty-five ids per call, for a backend that takes a list — four calls
on that book, twenty on a book of five hundred, and the number to tune when a source has a maximum
request size or charges per call. `universe` and `constraints` are book-wide by nature, so both are
global.

Making `holdings` per-account has a consequence worth understanding before you copy it, because
`holdings` is also one of the two tables security analytics are attached to, and assembly never sees a
per-portfolio dataset. The attachment moves rather than disappears: an assembly `join` puts the
analytics on the `universe`, and the shipped `attach_universe_columns` rule copies them onto
`holdings` per portfolio. Keep `holdings` global instead and one `join` does both tables at once —
that is the trade, and [how to add security analytics](how-to-add-security-analytics.md) walks both
ways through it. Watch the split in the run's log —

```text
dataset 'constraints' loaded: 534 row(s) in 1 batch(es), 2.28s
dataset 'details' loaded: 100 row(s) in 4 batch(es), 3.06s
dataset 'holdings' loaded: 200 row(s) in 100 batch(es), 15.34s
dataset 'universe' loaded: 3 row(s) in 1 batch(es), 24.21s
```

— and in the manifest, where each dataset's audit records its `batches` and how many portfolios it
`rejected`.

**A structural problem rejects the run; a coverage problem fails a portfolio.** A required dataset
missing, a schema violated, a global loader that raised, or a per-portfolio dataset no batch of which
came back — the run stops, because nothing can be built. One batch that raised, or a portfolio with no
`details` row — that portfolio alone is recorded as failed at stage `load` and the rest of the book
runs. In the example that is the difference between deleting
`examples/data/universe.csv`, which stops the run, and deleting P2's row from
`examples/data/details.csv`, after which every other account still solves and P2 alone is reported as
failed at `load`. A portfolio rejected here never
entered the run, so it traded nothing and couples to nobody in the schedule. Loading failures are
otherwise collected rather than raced: one failing dataset does not cancel the others, and all of them
are reported together.

The shipped loaders show the two shapes a source can take. `load_universe`, `load_constraints`, and
`load_parameters` answer for the whole book in one call. `load_holdings` answers one account per call:
it runs one request per id in the batch together — the shape of a loader for an API with a per-account
endpoint. It works under either scope: as a `global` dataset it owns its own fan-out, unbounded, and —
as the example configures it — with `"scope": "per_portfolio", "batch_size": 1` the engine owns the
partition, so `max_in_flight` bounds it. `load_details` is the third shape and a plain `def`: a
blocking database driver, run by the engine in a worker thread, issuing one query per batch of ids.

None of them is really a file loader. Each waits as long as its own source would and then answers from a
CSV table under the data root, so the template runs against no infrastructure; replacing one with the
real client changes the line that waits and nothing around it.

### `max_in_flight`

A `per_portfolio` entry may add `max_in_flight`: how many of the batches `batch_size` cut the book into
the engine keeps open at once. The example bounds both account-shaped inputs — `holdings` to 8, because
a hundred calls at once is more than the custodian's API allows, and `details` to 4 queries against the
firm's database. Omit it and every batch runs at once; a `global` dataset is one call and may not carry
it at all.

There is one number and it belongs to one input: no shared pools, no request rate, no burst. Why the
bound is the engine's rather than the loader's is in
[the architecture explanation](explanation-architecture.md#loading-is-the-slow-part-so-it-is-concurrent-and-metered);
how it behaves at load time is in [the reference](reference-run-config.md#max_in_flight), and wiring a
per-account source to it is in
[how to add a loader](how-to-add-a-loader-or-sink.md#async-loaders-and-fan-out).

## `assembly`

```json
"assembly": [
  {"name": "join", "params": {"into": "universe", "source": "analytics", "on": ["security_id"],
                              "cardinality": "one_to_one", "require_all_matched": true}},
  {"name": "drop", "params": {"datasets": ["analytics"]}}
]
```

`assembly` is how separately loaded datasets become the tables the build expects, and it is a list of
steps like `rules` — the same convention, applied once per run to all the data rather than once per
portfolio to one bundle. Each step is a function `(frames: Frames[, params]) -> Frames` that sees every
loaded dataset by name and returns the new set; the shipped ones live in `assembly.py`, and a desk's
own live in its package. The list runs after every loader has returned and before the engine-known
frames are validated against their schemas, which is what lets a step *supply* a required column that
no single loader produced.

The example configures none: each of its datasets arrives in the shape the engine wants. The snippet
above is the shape most real lists take, and the four shipped steps are the shapes that recur: `join`
brings columns from one dataset into another, `union` stacks datasets with the same meaning into one,
`select` trims and renames, `drop` discards. Read that join as a set of claims about the data, each of
which the engine checks:

- `cardinality: one_to_one` claims each security appears once on both sides; pandas enforces it, so a
  duplicated analytics row aborts the run instead of silently doubling a universe row.
- `require_all_matched: true` claims every universe security has a row; an unmatched key is reported
  by example and the run is rejected.
- A brought column that the target already has is refused unless `overwrite` is set, so a stale
  column is never silently replaced — or silently kept.

The key columns' dtypes are aligned to the target before merging so a `str` key never joins to a
`string` key as `object` and matches nothing. The `drop` afterwards is a courtesy to memory: any dataset
still present after the last step is carried into every portfolio's bundle, which is exactly right for
a per-portfolio exclusion list a rule will read, and wasteful for a vendor file that has done its job.

Most real assembly lists are mostly about one thing: attaching
per-security analytics to `holdings` and `universe`. Both tables accept any columns beyond their
schemas, and the two are later stacked into a single optimizer frame, so a column attached to both must
have the same dtype on both — the bundle refuses otherwise, naming the column.
[How to add security analytics](how-to-add-security-analytics.md) walks through that work; the
manifest records every step's source hash, parameters, row counts, and the columns it added.

## `rules`

```json
"rules": ["restrict_low_liquidity", "restrict_recent_trades", "add_zero_alpha", "attach_universe_columns"]
```

Rules are the business-logic layer: functions that take one portfolio's assembled data bundle and
return a modified bundle. They run per portfolio, in the order listed, after slicing and before the
build. The bundle they receive is already validated, and the only way to return a changed one is
`with_changes(...)`, which re-runs every cross-frame check — so a rule can tighten a cap, freeze a
name, or add a column, but cannot hand the optimizer something inconsistent.

The example configures the first two, and together the four show the spectrum.
`restrict_low_liquidity` reads its threshold from the `buy_universe_parameters` extra dataset — its
`params` name the dataset and the key, and default to exactly those — and marks names below it
restricted, which the build then freezes at their current weight. `restrict_recent_trades` reads the
account's rows of the `trades` extra and freezes every name traded within `window_days` of the run's
as-of instant: the wash-sale rule, as data the desk loads rather than state the engine keeps. `add_zero_alpha` takes no
parameters and adds an `alpha` column of zeros when the universe has none, so the `alpha` term could
be enabled without changing the data. `attach_universe_columns` copies that column — and any other
analytic the universe carries beyond its schema — onto `holdings`, so the two tables stack into one
optimizer frame with the same columns and dtypes; it is the per-portfolio counterpart of an assembly
`join` into `holdings`, for a book that loads holdings per account. The shipped `cap_single_name`
tightens the style's `max_weight` when the style's own is looser, and `restrict_to_mandate` freezes
every name whose sector is outside the account's mandate — the restriction-list shape that partitions
a book into independent components of the schedule.

Ordering is meaningful: a rule sees the output of the one before it. A rule never sees other
portfolios — it runs in a worker, on one bundle, before anything is solved — and that is what lets
every portfolio build at once. A rule that shrinks the portfolio's *tradable set* — freezing a name,
or, in a run that couples through buys, capping it at its current weight — also shrinks the set of
portfolios this one has to wait for; see `execution`.

## `solve_order`

```json
"solve_order": "most_uninvested_first"
```

The example does not set this — its portfolios file carries the priority — but a real book usually
should. A solve-order step is `(data: PortfolioData[, params]) -> Decimal`, run on each portfolio's
ruled bundle in the worker that built it; lower keys solve first and ties break on `portfolio_id`. It
answers "who gets first pick of a shared budget" from the data — the shipped step puts the account with
the most left to invest first — instead of from a hand-maintained column, and it is part of the
config hash, so two runs with different priorities are visibly different runs.

## `order_flow`

```json
"order_flow": "inflow"
```

The run's order flow, `inflow`, `outflow`, or `rebalance` — cash into the book, so the run buys; cash
out, so it sells; or neither on purpose, so it may do either; it is required and has no default. Every
one has one variable per name, `w`, with the trade a function of it alone — `buy = w − w0` under
`w ≥ w0`, `sell = w0 − w` under `w ≤ w0`, or `max(w − w0, 0)` and `max(w0 − w, 0)` with `w` free in
its bounds — so no name can be bought and sold in one solve. The value selects the *order-flow
profile*, the one object in the engine that knows what the order flow means, and it fixes which trades
portfolios couple through — buys under `inflow`, sells under `outflow`, both under `rebalance`. Three
things follow for the rest of the config: a term that reads a side the run lacks (`example_outflow`'s
`tax_cost` reads `sell`, and is absent from `example_inflow`) is refused at `validate-config`; under
`rebalance` both sides exist but are convex, so a term that *rewards* one — `tax_cost` on a name held
at a loss, a negative `weight` on `trade` — is refused too, by name (`example_rebalance` keeps the
inflow's terms); and the cash bounds keep their meaning as the cash *after* the run while an inflow or
an outflow fixes the direction cash can move and a rebalance moves it either way. A desk's order
flows are separate runs over one snapshot — `configs/example_inflow.json`, `configs/example_outflow.json`,
and `configs/example_rebalance.json` — each a pure function of its inputs with its own manifest;
nothing crosses between them inside the engine. [How to run an order flow](how-to-run-an-order-flow.md) walks through
both; [the architecture explanation](explanation-architecture.md#a-runs-order-flow-is-one-object)
covers what the profile owns, and why the rebalance is not the two-sided profile that was removed.

## `build`

```json
"build": "standard"
```

The step that turns a ruled bundle into the problem the solver sees, `(data: PortfolioData[, params])
-> ProblemSpec`, run per portfolio after its rules. The example leaves it at the default, `standard`
(`engine/build.py`): align every input to the sorted universe, compute the starting weights and the
tax per dollar sold exactly in `Decimal`, derive the per-security bounds from the style's `max_weight`,
the universe's optional `min_weight`/`max_weight` columns, and the `restricted` flag, and export
everything the bundle carries beyond the schemas by name — each numeric universe column as a spec
column, each boolean one as a flag, each string one as a grouping, and every number on the account's
`details` row as a scalar. That export is what lets a constraint row name a column the engine has
never heard of. Its one parameter is the box's start policy, `hold_breached_starts`: a name already
past a bound is held where it is — the bound moves to the current weight — instead of failing the
portfolio as a start the order flow cannot trade out of; it is in the config hash, so a run that
holds and a run that refuses are visibly different runs. A qualified name plugs in a build that reads the bundle its own way — tax lots, a
factor block, a different bounds policy — and returns a spec the rest of the engine consumes
unchanged; the engine derives the exact order inputs from whatever spec it returns.

## `objective`

```json
"objective": [
  {"kind": "linear", "name": "alpha", "column": "alpha", "weight": "-1"},
  {"kind": "linear", "name": "transaction_cost", "column": "tcost_per_dollar", "vector": "trade"}
]
```

The objective is the sum of the listed terms, and the engine only ever minimizes. That single sense is
deliberate: mixing rewards and costs in one list is easy to get wrong, so a reward is written with a
negative `weight` and everything else is a cost. Every term is a *kind* — a strict pydantic model the
record names by `kind` — with a `name` the report and the manifest key on and a `weight`, written as
a string so it is an exact `Decimal` in the manifest even though the solver ultimately sees a float.
The record is the whole term: there is no function to look up, and the manifest carries the record as
it is.

The shipped kind is `linear`: `weight · columnᵀvector` over any per-security column the spec carries
and one of the decision vectors — `w`, the target weight; `buy` or `sell`, the non-negative trade on
the side the run has; `trade`, the same amount under either name. The inflow config's two terms make the
solver trade the expected return of a name off against the cost of the trade itself: the exported
`alpha` column against `w` with a negative weight, the derived `tcost_per_dollar` against `trade`. The
outflow config adds a third, the derived `tax_per_dollar` against `sell` — the tax on realising a gain,
or the refund on harvesting a loss, which the run prices exactly because a name cannot be sold and
rebought in one solve; in the buy config the same term would be refused, since that run has no
`sell`. `column` is
any numeric universe column the build exported, which is how a signal a desk computes elsewhere
reaches the objective without the engine knowing anything about it; omit it and every name counts
once, so `trade` alone is a turnover penalty.

Every kind carries both halves of itself: `to_cvxpy`, which the shipped solve step renders once per
solve into a convex expression over the decision variables, and `value`, which the verifier later
recomputes in plain numpy to compare the sum with what the solver reported. A shape `linear` cannot
say — a diagonal risk penalty, a one-sided tracking cost — is a kind of its own, in this repository or
published by a package in the entry-point group `portfolio_optimizer.term`, known to the resolver,
the solve step, the verifier, and the JSON Schema alike; [how to add a term](how-to-add-a-term.md)
writes one.

## Constraints are not a config block at all

There is no `constraints` key. Which constraints bind an account is *data*, loaded per portfolio like
`holdings`, one typed row per limit:

```json
"constraints": {"loader": "load_constraints", "depends_on": ["portfolios"]}
```

```csv
portfolio_id,kind,label,params
P1,cash_limit,cash_floor,"{""direction"": "">="", ""bounds"": {""scalar"": ""cash_lb""}}"
P1,turnover_limit,turnover,"{""direction"": ""<="", ""bounds"": {""scalar"": ""max_turnover""}}"
P1,group_limit,sector_cap,"{""direction"": ""<="", ""column"": ""sector"", ""bounds"": {""TECH"": ""1"", ""HEALTH"": ""0.5""}}"
P1,participation_limit,adv,"{""direction"": ""<=""}"
```

A row's `kind` names a typed constraint model (`domain/constraints.py`), the same shape as a term:
`label` is its name, `params` its fields. The engine reads two things from a row — which portfolio it
belongs to, and the *declaration* the model makes: whether the kind reads the chain, and through its
`scope`, which securities it couples through. That is all the schedule needs, and it is why a
portfolio whose rows read no chain waits for nobody. What a row *does* is the solve step's business:
the shipped `cvxpy` step renders each model through its own `to_cvxpy`, and reports back on
`SolveResult.constraints` the records it applied. A book where every account has different
constraints needs one config rather than one per combination, and a desk with its own constraint
vocabulary writes rows without a `kind` column and replaces one function — the solve step — without
the engine changing.

**Where the numbers come from is the row's to say.** A bound is a literal (`"0.05"`), a per-account
scalar the spec carries (`{"scalar": "cash_ub"}` — any numeric column of the account's `details` row,
so `cash_limit` and `turnover_limit` read the style's `cash_lb`, `cash_ub`, and `max_turnover`), or,
for a per-security bound, a column of the spec (`{"column": "ub"}`). A limit that is not one scalar per
account carries its numbers on the row: `group_limit` names a string universe column — the spec carries
every one as a sparse membership matrix — and a bound per group, so bounding a second sector is a
second entry and bounding by country is a row naming `country`, not a schema change. Every kind takes
a `direction`, an optional boolean-flag `scope`, a verifier `tolerance`, and `allow_current_weight`,
the start policy: a bound the book already breaches is held where it is rather than failing the
portfolio. The full grammar is in
[the reference](reference-run-config.md#constraints-the-constraints-dataset).

**Rules may change them**, and that is ordinary rule work rather than a special power: the frame is on
the bundle, so a rule that tightens a cap because of what the holdings say returns
`data.with_changes(constraints=...)` like any other change, and the result is re-validated. The set a
portfolio actually solved is therefore recorded per portfolio in the manifest, after its rules, and
the rule audit counts the rows before and after. One edit is refused: adding a chain-reading row to
a run whose loaded rows read no chain. The engine plans such a run without a chain before any build,
so a build that then declares one fails at `build` rather than solving blind; a rule may tighten or
remove a chain reader, never add one.

The trade identity is not a constraint and cannot be. What `buy` or `sell` *means* — `w ≥ w0` with
`buy = w − w0`, or `w ≤ w0` with `sell = w0 − w`, the trade an expression of the one variable — and
the spec's own box `lb ≤ w ≤ ub` are what every cost term,
the turnover cap, the ADV constraint, the order rounding, and the verifier's identity checks rely on,
so they come from `order_flow` and the build and are added to every solve.

`participation_limit` is the one shipped kind that reads the chain — it needs to know how much of each
name's ADV budget higher-priority portfolios have already *traded* on the side the run couples
through. It writes two constraints: own trade within the budget, and the coupled side within what
predecessors left of it; and because it declares that, and its `scope` says where, the engine couples
the portfolio through `scope ∩ tradable` and nothing more. The rows are parsed when the portfolio
builds, after its rules: a malformed row, or one naming a column, flag, scalar, or group the spec does
not carry, fails that portfolio at stage `build` with the row's index in the message, before any solve
is scheduled on it, and the rest of the book runs.

Every kind carries its own numpy `residual`, so the post-solve check is a genuine second opinion for
each row the solve reported, shipped or published alike; nothing typed is ever unverified. A solve
step that reports no constraints has none checked, which is the honest answer rather than a silent
pass; the identity and solution checks still run for every portfolio.

## `solve`

```json
"solve": {
  "name": "cvxpy",
  "params": {"solver": "CLARABEL", "options": {"max_iter": 200}, "time_limit_s": 60.0}
}
```

Which step decides the weights, and its own parameters. The engine builds each portfolio's problem as
data, folds the chain, and hands one function everything it may use — the spec, the chain, the side
profile, the typed terms, the portfolio's constraint rows, the run's extra datasets — and takes back
weights. `cvxpy`, the default, is the optimizer: it renders the terms and the typed rows through their
own `to_cvxpy`, adds the profile's identity, and solves. `pro_rata_fill` is the other shipped step and
is not an optimizer at all — a numpy function that spends the cash on the underweights — which is the
point of the key: what a desk does on one side is sometimes not an optimization, and dressing it as
one costs seconds of solver time for an answer a function computes in milliseconds. A qualified name
plugs in a firm's own library that builds the problem its own way.

The cvxpy solver is the shipped step's parameter, and leaves with it. `solver` must be one the adapter
has a record for — `CLARABEL`, `OSQP`, `SCS`, `HIGHS`, which cvxpy installs, or `PIQP`, the `piqp`
extra — and it must be installed, and both are checked when the config *resolves*, not when the first
portfolio solves: `validate-config` rejects a typo, `run` rejects it before asking for a cluster, and
every worker checks its own image before it does any work. A solver cvxpy can see but the adapter has
no record for is refused too, because the record is what names the distribution whose version goes
into the environment fingerprint; without it two different builds of the solver would compare equal.
There is no automatic fallback: a run configured for one solver never silently produces answers from
another, because the manifest records the solver and its version as part of what makes a run
reproducible. `options` is passed verbatim to `Problem.solve`; the engine does not interpret it, so
what is valid depends entirely on the solver. `time_limit_s` is the one option the engine does
translate, because every solver spells it differently — `time_limit` for Clarabel, OSQP, and HiGHS,
`time_limit_secs` for SCS — and a solver with no such option (`PIQP`) rejects the setting at resolve
rather than guessing. `verbose` turns on the solver's own iteration log, which is the first thing to
enable when a solve is slow or hits its limit. Adding a solver is one row in the adapter's table
(name, distribution, time-limit option) and one extra in `pyproject.toml`, so that "install the
solver" is the same `uv sync --extra` on a laptop and in the worker image.

Whatever the step is, the engine treats its answer the same way: the order-flow profile turns the weights
into a trade, the verifier re-checks every constraint the step reported against them, rounding and
drift run unchanged, and the manifest records the step and its version where it records the solver's.
A step that minimized nothing reports no objective; the verifier then skips the objective comparison
and still evaluates the configured terms as a report line, so a heuristic and the optimizer can be
compared on one book. One thing a step of your own gives up: the engine cannot see whether it reads
the chain, so every portfolio couples through its whole tradable set under it. The guarantees are the
verifier's, not the step's — see [how to replace the cvxpy solve](how-to-write-a-solve-step.md).

## `post_solve`

```json
"post_solve": {"violation_tol": 1e-6, "objective_rel_tol": 1e-5, "objective_abs_tol": 1e-9}
```

After every solve the engine re-checks the solution in numpy without cvxpy: the order-flow profile's identity
checks (`no_sells`, `trade_balance`, `nonneg_buy`, and `sell_absent` under `inflow`; `no_buys`,
`trade_balance`, `nonneg_sell`, and `buy_absent` under `outflow`; the spec's box as `lb` and `ub` under
either),
each reported constraint's residual, the recomputed objective against the solver's reported one, and
finiteness. These three numbers are its tolerances — `violation_tol` bounds every residual, identity
and constraint alike, and the other two bound the objective gap — and they are JSON numbers rather
than strings because they are float tolerances on float arithmetic. The defaults are deliberately about
a hundred times looser than a solver's convergence tolerance so that a pass says something about the
solution rather than restating the solver's own stopping criterion. Tighten them and a solver that is
merely converged can start failing verification; loosen them and the check stops meaning much. The
example writes the defaults out explicitly. The same tolerance is what decides that a check is
*binding* — its residual sits within it of the bound — which the run prints per portfolio and the
manifest records as `check.active`.

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
"execution": {"on_error": "fail_fast", "dependencies": "overlap"}
```

This block answers two questions: what one failed portfolio does to the rest, and which higher-priority
portfolios a portfolio waits for. It deliberately does *not* say where the work runs or how many workers
there are — those are settings (`PORTFOLIO_OPTIMIZER_CLUSTER`, `PORTFOLIO_OPTIMIZER_MAX_WORKERS`, and the
other cluster variables), so a laptop run and a cluster run of one config hash identically and differ
only in the manifest's `settings` block, where `diff-manifests` can name the difference. The block may
be omitted entirely; both keys have defaults, and the example writes them out.

There is no schedule to choose. The engine derives who waits for whom from each portfolio's solve-order
key, its *tradable set* on the side the run couples through, and what its own constraint rows declare
they consume, and the answer never depends on the schedule: each solve sees only what its overlapping
predecessors traded there, which is a function of the data.
[The architecture explanation](explanation-architecture.md#a-run-couples-through-its-one-side-so-the-schedule-is-a-graph)
makes that argument; [the life of a run](explanation-run-lifecycle.md#9-the-dependency-graph-and-where-the-work-runs)
shows the mechanism in execution order.

`dependencies` has one non-default value, `"all"`, under which every higher-priority portfolio is a
predecessor — one line. It gives the same orders and the same chain hashes and exists for diagnosis:
rerun a suspicious batch as a line and `diff-manifests` the two (it names the config, because the
field is part of it, and nothing else). There is no `"none"`, because nothing a config says can switch
the chain off: when the data and the steps make it moot — no chain-reading row in any account, no
chain-aware term, the shipped solve step — the engine sees that before any build, no portfolio waits,
and the manifest's `schedule` block records `coupling: "none"`.

`on_error` decides what one failed portfolio does to the rest. `fail_fast` records every lower-priority
portfolio as `skipped` — whatever it had finished, so the manifest never depends on timing. `continue`
isolates the failure: only the portfolios that depended on it are skipped, each naming the predecessor
that failed, and a portfolio that shared no tradable security with it is unaffected. A build that fails
has an unknown tradable set and is treated as overlapping every lower-priority portfolio. The cluster
and its worker count are about throughput, never about output;
[how to run on a cluster](how-to-run-on-a-cluster.md) covers the settings.

## What the config does not decide

Seeing what is absent sharpens the picture of what the config is for.

**Numbers live in the data.** Position sizes, prices, tax rates, single-name caps, turnover limits,
sector bands, ADV participation, a liquidity threshold — none of these appear in the config. It names
the datasets that carry them and the kinds and steps that read them. The practical consequence is that
the same config can run every day against fresh data, and `diff-manifests` can tell you whether a
change in orders came from the data or from the wiring.

**The instant is an argument.** `--as-of` is not in the document, so one wiring has one hash however
many days it runs.

**Behavior lives in code.** The config chooses functions by name and tunes them through `params`, and
names term kinds by `kind`; it cannot express a new cost model or a new constraint shape. That is the
boundary the resolver enforces: if a run needs a shape the shipped kinds do not have, you write a kind
or a step with the right contract and name it, and the [how-to guides](how-to-add-a-term.md) cover
that.

**Some tolerances are derived, not configured.** The rounding-drift bound — how far executed weights
may deviate from solved weights after shares are rounded — is computed from the priciest lot and the
dust threshold in the data, so there is no knob for it. The only tolerances you set are the verifier's,
in `post_solve`.

**The schedule is derived, not configured.** Which portfolios wait for which follows from the steps
(does anything read the chain?), the constraint rows (which ones, and through which securities?), and
the data (which securities can each portfolio trade on the side the run couples through?). The config
has no execution mode; the only schedule knob is the diagnostic `dependencies: "all"`.

Put together: the config is the wiring of a pipeline — which inputs, combined how, filtered by which
rules, prioritized how, built how, optimized against which terms, solved with what, checked how
tightly, delivered where. Everything in it is either a name the resolver can check before data loads or
a value with a declared type and range, and that is what lets `validate-config` promise that a config
which passes will not fail for a configuration reason later.
