# Ideas

Threads for expanding the template that are not yet decisions. Each one states the problem as the
engine has it today, the options, and a leaning; none is a commitment. When a thread becomes a
decision it moves into the code and [the architecture explanation](docs/explanation-architecture.md)
and leaves this file. The last section is the exception: known defects and trailing work, decided
already and waiting only to be done. Numbers below are for a book of *N* = 100,000 unique securities,
which a business unit can exceed — measured where the text says so, estimated otherwise.

## Where the time goes at 100k names: the solver, not the build

Measured 2026-08-29 with `benchmarks/profile_portfolio.py` — one portfolio, the shipped rules, terms,
and constraints, a synthetic book of 100,000 names with 25,000 held, in one process, Clarabel:

| Stage | Seconds | Where it runs |
|---|---:|---|
| validate the bundle | 0.4 | build task, parallel |
| rules (`restrict_low_liquidity`, `add_zero_alpha`) | 1.5 | build task, parallel |
| spec build | 0.7 | build task, parallel |
| content hash | 0.01 | build task |
| expression tree + `is_dcp` | 0.003 | solve task, critical path |
| canonicalization (`get_problem_data`) | 0.6 | solve task, critical path |
| **Clarabel**, 29 iterations | **7.5** | solve task, critical path |
| unpack, classify | 0.1 | solve task |
| verify | 0.01 | solve task |
| orders + rounding drift | 0.55 | solve task |
| persist spec + solution (`.npz`) | 0.01 | client |

The canonical form is *P* 400k × 400k with 100k nonzeros and *A* 900k × 400k with 1.9M nonzeros; the
process peaks at 1.4–1.8 GB, canonicalization and the solve each adding about half a gigabyte, which
is what a worker needs per concurrent solve at this size.

So the premise this section used to open with — that building the cvxpy problem is the expensive
half — is wrong by an order of magnitude. Canonicalization is 7% of the solve task; Clarabel is 85%.
The build side is two and a half seconds and runs for every portfolio at once, most of it bundle
validation (0.4 s per pass, and every rule's output is validated again). Verification, hashing, and
orders are noise. Two conclusions follow and one thread stays open:

- **The direct assembler is off the table.** It would save at most 0.6 s of a 9 s critical path at a
  real cost in readability. Removed from this file; the numbers above are why.
- **The sector matrix was the size problem, and it is fixed.** Dense, at 160 sub-industries, it was 128
  of the spec's 137 MB, 142 MB per portfolio in `problem_specs/`, and — the part the old estimate
  missed — 147 MB in every `PortfolioResult` returning to the client, since the result carries the
  spec. Built in numpy from category codes and carried as CSR (one nonzero per security) it is 1.6 MB
  whatever *K* is: spec 10 MB, `.npz` 15 MB, result 20 MB, spec build 1.5 → 0.7 s, hash 0.08 → 0.01 s.
- **What is left is the solver**, which is the thread below.

### The solver thread

Clarabel spends about 0.26 s per interior-point iteration here, in a KKT factorization over *A*. *P* is
diagonal — the shipped `tracking_error` is a plain sum of squares — so this is very nearly an LP with
300k variables and 900k rows, and the per-iteration cost is the sparse factorization, not the
objective. Things to measure, in the order they are cheap:

1. **Clarabel's linear solver.** The default is QDLDL, single-threaded. Clarabel also offers `faer`
   (multi-threaded) and, where the wheel carries it, MKL, through `direct_solve_method` in
   `solver.options` — no engine change, one config key. Workers run with `--nthreads 1` and
   `OMP_NUM_THREADS=1`, which is right for many small portfolios and wrong for a few enormous ones;
   the worker thread count may want to be a setting rather than a constant.
2. **Other solvers, same problem — measured, and Clarabel is the only one that works at defaults.**
   OSQP stops at its iteration limit (`user_limit`) and the engine refuses the answer. HiGHS fails
   outright (its QP method is not built for 400k variables). SCS reports `optimal` after 260 s and
   12,750 iterations, and the verifier rejects it: max violation 6e-7 is inside `violation_tol`, but
   the objective gap is 1.05e-5 against a 1e-5 relative tolerance — the first-order optimum is looser
   than the verifier's definition of agreement. Any of the three would need its tolerances and
   iteration cap set deliberately, and the verifier's tolerances loosened to match, before it could
   even be compared on time. Clarabel stays the default; the thread is making *it* faster.
3. **Warm starts** move from "pointless" to worth trying, but only for the solvers that use them
   (OSQP, SCS); Clarabel does not. See the thread under *Other threads*.
4. **The formulation.** Three variables per name (`w`, `buy`, `sell`) plus the slack every inequality
   adds is what makes *A* 900k rows. A formulation that lets the solver see `buy` and `sell` as the
   positive and negative parts of one trade vector without an explicit equality row would shrink the
   KKT system; whether cvxpy's reductions already do this is a question for the canonical data, not
   for reasoning.

### The result carries the spec back

Every `PortfolioResult` returns to the client with its `ProblemSpec` inside — 20 MB at 100k names now
that the sector matrix is sparse, 10 MB of it the spec's own vectors — because the client persists
`problem_specs/<portfolio>.npz` for `verify` and `diff-manifests`. A thousand portfolios is 20 GB into
one process over one NIC, held until each is written. The spec is also exactly what the worker already
has. Options, none decided: write the `.npz` from the worker when the run directory is a shared or
object-store path and return only the hash; or return the spec lazily, as a Dask future the client
pulls while persisting, so the transfer overlaps the solves instead of following them. The
`Contribution` a dependent solve receives is a few kilobytes and is unaffected either way.

### Considered and rejected: build the problem elsewhere and ship it

Every variant — pickle the `cp.Problem` back, ship `get_problem_data`'s output and unpack in the
client, the DPP split with the chain as `Parameter`s — moves canonicalization, and canonicalization
is 0.6 s. What the variants would move instead is data: the canonical form is a 3–5× expansion of the
spec, `unpack_results` needs the same `Problem` object on both sides (`inverse_data` refers to
per-process variable ids), and a private canonicalization cache is not part of any pickle contract.
Dropped 2026-08-29 on the numbers above.

### Three things are called "build"

For the record, since the word is overloaded:

1. **The spec build** (`engine/build.py`): rules, Decimal arithmetic, alignment to the sorted universe,
   the one Decimal→float64 conversion. Pure numpy out. Runs in workers, every portfolio at once.
2. **The expression tree** (`engine/solve.py` → the terms and constraints → `cvx/adapter.py`): a few
   dozen cvxpy nodes holding references to the spec's arrays. Milliseconds.
3. **Canonicalization** — inside `problem.solve()`: DCP verification, the reduction chain to the
   solver's conic or QP form, coefficient extraction into sparse matrices. 0.6 s at 100k names with
   the shipped terms.

### The chain is a graph, not a line — done

Decided 2026-08-29 and landed: portfolios couple through buys only, every portfolio builds at once, the
runner derives who waits for whom from each build's buyable set, and Dask enforces the graph. The
design, the exactness argument, and the failure contract are in
[the architecture explanation](docs/explanation-architecture.md).

### Selling, if it ever comes

Not planned, and possibly never; recorded so the buy-only guarantee is not silently load-bearing.
Everything selling would add couples through sells, per security:

| Effect | Produced by | Consumed by |
|---|---|---|
| ADV budget spent by sells | sells | buys and sells |
| Wash sales: do not buy what an earlier account sold at a loss | sells | buys |
| Wash sales, mirrored: do not sell at a loss what an earlier account bought | buys | sells |
| Internal crossing: an earlier sell of *X* makes a later buy of *X* cheaper — a *term* | sells | buys |

Each simplification the buy-only guarantee bought (see the architecture explanation) would need to un-simplify, in this order:

1. `ChainState` gains `cumulative_sold`; the fold reads both sides.
2. A chain-aware step declares which sides it produces on and consumes on, defaulting to both
   (conservative, always sound). The resolver already inspects signatures; this is one more attribute.
3. The edge test becomes side-aware: *j* depends on *i* iff, for some configured step, *i*'s set on the
   step's produce side intersects *j*'s set on its consume side. Sellable is the held securities the
   spec allows a positive sell in, after rules.
4. Keep rules chain-free even then. The wash-sale rule the template once shipped capped `max_weight`
   from earlier sells; that is a constraint with a chain right-hand side, not a rule, and writing it as
   one preserves the single build pass. If a rule genuinely must read the chain, the pipeline becomes two
   passes — rules with an empty chain to derive the graph, rules with the real chain to build — and needs
   the invariant that a chain-aware rule may tighten a portfolio's buyable or sellable set but never
   expand it, checked after the real build.

The cost to expect: the moment any sell-side step is configured, every held security is a potential
edge again — the bonds the buy filter removed re-couple accounts through the sell side — and components
grow to match. The manifest's derived-graph record is how to watch that happen.

### Cheap things to do first, whatever else happens

- **Build the sector matrix in numpy and carry it sparse.** Done 2026-08-29; numbers above.
- **Keep the factor risk term structured when it returns.** `sum_squares(F½ · B · w) + sum_squares(√D ∘ w)`,
  never a dense *N* × *N* covariance (80 GB at 100k). The 50 × *N* loadings are the one genuinely dense
  block, 40 MB, and they set the floor on every size above the moment the term exists.
- **Re-profile when a term changes.** `uv run python benchmarks/profile_portfolio.py --securities 100000`
  prints the table above for the shipped config; the split between canonicalization and solve is
  solver- and structure-dependent, and a factor term will not look like the diagonal *P* measured here.

## Acceptance scenarios the business writes, and a harness that runs them

The suite today is tiered — pure functions, invariants no schema can express, boundary rejects, one
smoke test per entry point — and every tier is Python, written by whoever wrote the engine, asserting on
values only that person can check. Nothing in the repo states a *business* expectation — "no account
buys more than a quarter of a name's daily volume", "a restricted name is never traded", "the
higher-priority account gets the scarce liquidity" — in a form a portfolio manager or a QA analyst can
approve before the code exists and point at after a bad run. The gap is not coverage. It is that the
suite has one audience.

Little of it has to be built. `engine/check.py` is already a cvxpy-free numpy twin of every shipped
constraint; `verify` re-runs it over a persisted `.npz` with no solver installed; `diff-manifests` names
the stage at which two runs diverge; the manifest records every step, hash, and environment fingerprint.
An acceptance harness needs a *vocabulary*, a *runner*, and a *report* — not new engine machinery.

### The scenario is a file, not a test function

Three forms were considered.

**Gherkin** (`pytest-bdd`, `behave`). Reads beautifully in a demo and rots in a repo: every phrase needs
a step definition, the step definitions become a second untested codebase, and the assertions here are
tabular, not narrative. Rejected.

**Golden order files.** Business cannot review a parquet of orders, and a flat optimum makes the goldens
churn for reasons nobody can explain. Keep them, but as a *secondary* regression check the harness
generates and a human diffs — never as the statement of intent.

**A declarative scenario file.** Leaning. One JSON (or TOML) file per scenario: a `given` naming a tiny
book — a handful of securities, two or three portfolios, their holdings, targets, and style constraints
— a run config, and an `expect` list drawn from a fixed vocabulary:

```json
{
  "name": "priority_wins_the_scarce_liquidity",
  "intent": "When two accounts want the same thin name, the higher-priority one fills first.",
  "given": {"data": "./data", "config": "./run.json"},
  "expect": [
    {"portfolio": "P1", "buys_at_most_adv_fraction": {"security": "THIN", "fraction": "0.25"}},
    {"portfolio": "P2", "does_not_buy": "THIN"},
    {"portfolio": "P2", "bound_by": "cumulative_adv_participation"},
    {"every_portfolio": true, "satisfies": "all_constraints", "within": 1e-6}
  ]
}
```

The `intent` string is the point of the file. It is what business signs off on and what the report
prints beside pass or fail; the `expect` list is the machine-checkable spelling of it, and a review that
finds the two disagree has found the bug the harness exists for.

Two entry points over the same files: a CLI subcommand (`portfolio-optimizer qa qa/scenarios/`) so QA
runs it without knowing pytest exists, and a pytest collector so CI runs the identical files as a
parametrized test. One file, two runners, no second implementation.

### The assertions that are stable, and the ones that are flaky in disguise

This is where a naive harness fails. A QP with a flat direction has many optimal solutions, and the one
the solver returns depends on its version, its tolerances, and the platform's BLAS. An expectation on an
individual weight is a coin flip dressed as a requirement. The vocabulary has to be built around what is
actually unique:

| Kind | Example | Unique? |
|---|---|---|
| Constraint residual | `satisfies: all_constraints, within: 1e-6` | Yes — holds for every optimal solution |
| Objective value | `objective_at_most: "0.0042"` | Yes — the argmin may not be unique, the minimum is |
| A binding constraint | `bound_by: turnover_cap` | Yes when the active set is unique; check it |
| Aggregate flow | `turnover_at_most`, `cash_between` | Usually — these are constrained quantities |
| Sign or absence | `does_not_buy: THIN` | Only when a constraint forces it, not a preference |
| A single weight or share count | `buys_shares: {"AAPL": 500}` | **Only when the optimum is unique** |

So: assert on residuals and on the objective by default; assert on individual orders only in scenarios
deliberately constructed to have a strictly convex objective and a unique optimum, and make the harness
*prove* that rather than assume it — re-solve with the objective perturbed by ±ε and with a second
solver, and fail the scenario if the orders move. A scenario that cannot pass its own uniqueness check
may only carry the stable kinds. That check is the difference between a harness QA trusts and one they
learn to re-run until it goes green.

Tolerances belong per expectation, not globally. The run's environment fingerprint — the one `_accept`
already computes — goes in the report, because "the same scenario passed on a different solver build" is
a claim the report should be able to substantiate.

### The report is the deliverable, and margin is what makes it useful

The harness writes a table per scenario: the `intent` sentence, then every expectation with expected,
actual, **margin**, and pass or fail, above the run id and config hash. Margin is what turns the report
from a green tick into a review artifact — "passed with 0.4% of headroom" and "passed with 0.00003" are
different facts, and the second is the one a portfolio manager wants to see before a limit is loosened.

The other thing business asks first, every time, is *why*. "Why did P2 not get the name?" is answerable
today and printed nowhere: the largest active residual per portfolio names the binding constraint, and
the chain state names who consumed the budget ahead of it. The harness should print, per portfolio, the
constraints that were tight and — for a chain-bound portfolio — which higher-priority portfolios
consumed what. Cheap to compute, and probably the highest-value line in the whole report.

### One scenario that cannot be checked by hand

Most scenarios should be ten securities and three portfolios, small enough that a person can work the
answer out on paper — that is what makes them reviewable. But the behavior nobody can check by
inspection, and the one business most wants demonstrated, is the chain: twenty portfolios with a
deliberate overlap pattern, a binding ADV budget, and a priority order, asserted against the same run
executed as a strict total order with `dependencies: "all"`. Identical orders and identical chain hashes
from two schedules is the claim the derived-DAG design rests on, and it belongs in the QA suite where
business can see it pass, not only in an engine test.

### What it needs, and what it must not become

Needed: a scenario model (strict, like the run config), an expectation vocabulary evaluated against
`PortfolioResult` and `ConstraintReport`, a runner over a `tmp_path`, and the report writer. The
expectation names should be the names the verifier already reports, so a failure points at a
`ConstraintCheck` rather than at a numpy line.

Not needed, and worth refusing: a way for a scenario to reach into the engine. A scenario that patches,
stubs, or imports anything from `portfolio_optimizer` has stopped being an acceptance test. It runs the
CLI over a directory and reads the artifacts, or it is a unit test in the wrong place.

## Constraints written in the config, not only in Python

A constraint today is a *function* — named in `constraints`, found in `terms.py` or an importable module,
signature-checked at resolve time. Its *numbers* live somewhere else entirely: the `constraints` dataset,
per portfolio, typed by `StyleConstraints`. Adding a limit that is structurally identical to one that
exists — country bounds where there are sector bounds, an issuer cap where there is a single-name cap —
means editing `terms.py`, writing its twin in `check.py`, extending `StyleConstraints`, and shipping a
release. The shape is code, the numbers are data, and the two are joined only by convention.

Two asks, and they are not the same ask.

### Builder functions: one function, many instances

A parametrized constraint instantiated more than once in a run — `group_bounds(column="country", ...)`
alongside `group_bounds(column="rating", ...)`. Most of this works already; three gaps:

- **Instances need labels.** `constraints` is a tuple of `StepSpec` and the model permits the same name
  twice, but the manifest and `ConstraintReport` key on `qualname`, so two instances of one builder
  collide in the report. An optional `label` on `StepSpec`, defaulting to the bare name and checked
  unique at resolve, is the whole fix; `params_sha256` already tells the instances apart for provenance.
- **Params need to carry tables.** A builder wants a grouping column and a mapping of group → bounds.
  `Params` is a pydantic model, so `dict[str, tuple[Decimal, Decimal]]` validates for free. The real work
  is in the spec: `sector_matrix` / `sector_lb` / `sector_ub` generalize to `groups: Mapping[str,
  GroupBlock]` keyed by the universe column they were built from, and `sector_bounds` becomes
  `group_bounds(column="sector")` — the shipped constraint is one instantiation of the general one. Build
  it once, sparse, for every grouping, the way the sector matrix already is.
- **The numbers have two homes.** A builder's params are per-run; `sector_bounds` gets its numbers per
  portfolio from the style. So a param must be able to say *where* rather than *what*: a literal table,
  or `{"from_style": "sector_bounds"}`, with the style overriding the config default per portfolio. This
  is the one genuine fork in the design, and it wants deciding before anything is written — a run-wide
  default with per-portfolio override is the guess, because that is how desks actually describe limits.

### Declarative constraints: the algebra in the config

The constraint itself written as data, no Python:

```json
"constraints": [
  "trade_balance",
  {"label": "country_caps", "of": {"sum": "w", "by": "country"}, "at_most": {"from_style": "country_bounds"}},
  {"label": "no_new_tobacco", "of": "buy", "where": {"flag": "is_tobacco"}, "at_most": "0"},
  {"label": "issuer_cap", "of": {"sum": "w", "by": "issuer"}, "at_most": "0.05"},
  {"label": "adv", "of": "buy", "at_most": {"chain": "adv_remaining"}}
]
```

**The grammar stops at affine.** Every expression is affine in `(w, buy, sell)` with constant data, over
a closed set of aggregations — elementwise, sum, sum-by-group, a boolean-flag subset — compared against a
constant, a style value, a spec column, or a named chain quantity. Every shipped constraint fits; nothing
in the objective does, which is why this is a constraint grammar and not a term grammar.

Holding that line buys three things at once:

- **The verifier twin is generated, not written.** This is the strongest argument for the declarative
  path and it inverts the usual expectation. A custom Python constraint is reported as `unverified`
  today — the auditor is told to trust it. A declarative one never can be: one interpreter walks the tree
  emitting cvxpy atoms, another walks it emitting numpy residuals, and `check.py` gets a single entry for
  the whole grammar. Constraints authored in config end up *more* verifiable than constraints authored in
  Python.
- **DCP never comes up.** Affine ≤ constant is convex by construction, so there is no convexity check to
  write, no confusing failure to explain, and no temptation to grow the grammar until there is.
- **Provenance is free.** `config_sha256` already covers it. No module to import on a worker, no
  `source_sha256` to drift between the client image and the cluster's.

Chain-awareness has to be declared, not inferred, because the buy-only dependency graph derives its edges
from which steps read the chain. `{"chain": "adv_remaining"}` names one of a closed set of quantities the
engine computes in numpy and shares with the verifier — `adv_remaining` is already exactly that function
today. A closed set, never an expression: the verifier needs a numpy twin of every chain quantity, and
a closed set is the only way to have one for each.

### One implementation, two spellings

The failure to design against is two ways to say the same limit that drift apart — a JSON `group_bounds`
and a Python `group_bounds` that disagree at the third decimal after someone fixes one of them. So the
declarative form should **compile to the builders**, not live beside them: JSON → validated model → a
call to `group_bounds` or `bound_on`, with the same params model, the same twin, the same report label.
The Python builder stays the extension point for what the grammar cannot say; the grammar is what
business and QA read. And it is the same vocabulary the acceptance scenarios above assert in — one says
what the optimizer must enforce, the other says what must hold afterward, and a QA analyst should not
have to learn two syntaxes to write both.

Two things this changes elsewhere. `constraints` becomes a heterogeneous list — bare name, step object,
declarative object — so the published JSON Schema grows a discriminated union and the docs have to say
plainly when to reach for which. And a declarative constraint naming a column the universe does not carry
cannot be caught by `resolve_config`, which runs before any data loads; the check moves to a new gate
just after assembly and before the first build, with the same collected-failures error shape, so the run
still dies before it does any real work.

### Where the line has to hold

Someone will ask for a conditional, then for arithmetic between columns, then for a soft version with a
penalty in the objective. Each is reasonable alone and together they are a worse cvxpy expressed in JSON,
with a hand-written twin and a DCP checker of our own. The rule to write down now: **anything that is not
affine in `(w, buy, sell)` against constant data is a Python builder, no exceptions.** A soft limit is a
term, and terms stay in Python.

Integer constraints are outside both paths and should be said so out loud: minimum trade notional, round
lots, and a cap on the number of holdings are not convex, they are not what `orders.py` rounding already
approximates, and putting them in the grammar would promise something the solver will not deliver.

## Order aggregation as a step: from per-portfolio orders to what goes to the street

The run ends abruptly today. Every solved portfolio's orders are concatenated, sorted, and handed to the
sink in one frame (`engine/runner.py:390`), and that frame is simultaneously two things it should not be:
the audit record of what the optimizer decided for each account, and the instruction set a trading system
is expected to act on. Real desks do not send a thousand separate 300-share buys of the same name. They
net, they block, they drop what is too small to work, they split what is too big, and they cancel or
amend what an earlier run already put in the market. All of that is between the last solve and the sink,
and none of it exists.

It should be a configured step kind, for the same reason rules and terms are: the arithmetic of netting is
common, and every desk's version of it differs — a street minimum here, a child-order policy there, a
broker that wants one block per sector. `sinks.py` is the wrong home; a sink that nets is a sink that
cannot be swapped for a different destination without carrying the netting logic with it.

### The grain changes, so the frame has to

This is the design decision everything else follows from. `ORDERS` is keyed `("portfolio_id",
"security_id")`. A block has no `portfolio_id` — it has many — so an aggregator that rewrites the orders
frame in place has to invent a synthetic id and, in doing so, destroys the one thing that must survive:
which account gets which shares. That is not a nicety, it is what makes the run auditable and what the
allocation back from a fill depends on.

So aggregation **derives, never mutates**. The per-portfolio orders frame stays exactly as it is — tied to
`spec_hash`, drift-checked, the record of what was decided — and the step produces a second structure
beside it:

| Frame | Grain | What it is |
|---|---|---|
| `orders` | portfolio × security | Unchanged. What the optimizer decided, already tied to a spec hash. |
| `blocks` | block | What goes to the street: security, side, total quantity, and a policy label. |
| `allocations` | block × portfolio | The link. Every block's rows sum to the orders that composed it. |

Carrying `allocations` explicitly, rather than recomputing it from a shared key, is what lets a partial
fill be allocated back later, and what lets the engine check the aggregator's arithmetic at all.

### A list of steps, not one step

Desks describe this as a sequence — "net by security, then drop anything under the street minimum, then
split whatever exceeds a quarter of the day's volume into child blocks" — so `aggregation` should be a
tuple of steps in the config, like `assembly` and `rules`, not a single hook.

That composes only if every step has the same input and output type, which means the grain conversion
cannot be one of the steps. The engine converts first: an internal `to_blocks(orders)` produces the
trivial blocking — one block per order, one allocation each — and every configured step is then
`(blocks: Blocks[, frames: Frames][, params]) -> Blocks`. A run that configures nothing passes the trivial
blocks straight through, so the sink's input is uniform and there is no implicit behavior to remember.

The `frames` argument matters more than it looks. Netting to a lot boundary needs `lot_size`; a
participation cap needs `adv_shares`; a sector-blocked broker needs `sector`. Those live in the assembled
universe, not in the orders frame, and the alternative — growing `ORDERS` a column for every aggregator
anyone might write — is how a schema dies. Hand the step the assembled frames read-only and let it take
what it needs. What it should *not* get is the specs: `PortfolioResult` carries one per portfolio, a
thousand of them are ten gigabytes in the main process, and nothing an aggregator legitimately does needs
the problem the solver saw.

### Cancels are two features wearing one word

Worth separating before either is built, because only one of them is easy.

**Dropping within the run.** The aggregator decides an order should not go out — netted to zero, under the
street minimum, a name that halted. Mechanically this is just absence from the block frame, and the only
requirement is that absence be *recorded with a reason* rather than silent. There is precedent:
`rounding_drift` already counts `dropped_orders` for dust-filtered trades. Same shape, one field wider.

**Cancel or replace against a live market.** An order from an earlier run is still working and must be
pulled or amended. This is a different animal, and it breaks the property the whole engine is built on: a
run is a pure function of its inputs, rerunnable, diffable with `diff-manifests`. A cancel refers to an
order id that exists because of a previous run and a market that has moved since.

The property is recoverable, and cheaply, if the design is right from the start: **the working-order book
is an input dataset**, loaded by a loader like everything else, content-hashed, and recorded in the
manifest. Then the run is once again a pure function — of its holdings, its universe, and a *snapshot* of
what was working at a moment — and rerunning it against the same snapshot gives the same instructions. Get
this wrong (a sink that queries the OMS for live state at publish time) and the run stops being
reproducible, quietly, and no amount of manifest detail brings it back.

Given that input, the block frame's rows carry an intent — `NEW`, `REPLACE`, `CANCEL` — against a
`client_order_id`, and a shipped `reconcile_working_orders` step does the diff. The engine should not
invent the id scheme; that is the OMS's, and it belongs in the step's params.

### The engine checks the aggregator, the way it checks the solver

An aggregation step is arbitrary user code that decides what a trading system is told to do. The posture
the repo already takes with custom steps applies with more force here, not less: `check.py` re-verifies
every solution without cvxpy and reports custom constraints as `unverified` rather than trusting them. The
equivalent is an `AggregationReport`, computed by the engine over whatever the steps returned, and the run
refuses to publish when it fails:

- Every block's allocations sum exactly to the orders they claim, per security and side.
- No allocation exceeds the portfolio's own order for that security.
- No block names a security and side that no portfolio ordered.
- Quantities stay positive whole numbers on lot multiples.
- Nothing is allocated to a portfolio that did not solve, and every order is either in exactly one block or
  in the dropped list with a reason.

These are cheap — a couple of grouped sums over a frame that is orders of magnitude smaller than anything
else in the run — and they are exactly the invariants a business reader would state if asked what
aggregation must never do. Which makes them the natural first entries in the acceptance harness's
vocabulary: *the blocks add up*.

### What the block sees that no portfolio can

There is a reason for this step beyond plumbing, and it is the argument for building it.

`cumulative_adv_participation` bounds each portfolio's buy against the shared budget, and the chain makes
the portfolios' sum respect it — but that check happens on the *solved weights*, before rounding, and
rounding is per portfolio and not chain-aware. Two hundred accounts each rounding a fraction of a lot in
the same direction can push the aggregate over a participation cap that every individual account
respected, and nothing in the engine would see it: verification ran pre-rounding, per portfolio, and the
sink does not know what a cap is.

The block is the first place the run can look at the total it is actually about to send. Aggregate ADV
participation, aggregate notional, an issuer or sector total across the whole run — these are constraints
that only exist at the block grain, and the `AggregationReport` is where they belong. That is also the
honest framing of what a per-portfolio ADV constraint buys: it makes the aggregate approximately right,
and the block check is what makes it true.

### Where the line is

Aggregation nets and reshapes what the solver already decided. It does not change any account's target
weights, and it must not: the moment an aggregation step is allowed to move a portfolio's allocation to
make a block work, the orders frame stops matching the verified solution and the whole post-solve audit
chain is fiction.

That is the answer to internal crossing, too. P1 buying 10,000 of a name while P2 sells 6,000 of it is a
crossing opportunity, and the temptation is to net it here. But crossing changes what each account pays,
which is a decision about the objective — it is listed as a *term* in the selling table above, not as a
post-processing step. Aggregation may report that the opportunity existed; it may not price it. And
because it runs after every solve, on one process, over a small frame, it never touches the dependency
graph — nothing an aggregator does can feed back into which portfolio waits for which.

## Other threads

- **Tax lots.** Holdings are security-level with an average cost. Lot-level tax needs one sell variable
  per lot (*N* + *L* variables, lot-to-security aggregation as one more sparse matrix), changes the
  holdings schema and `orders.py`, and is the first extension that makes the build-placement question
  above unavoidable.
- **Warm starts.** OSQP and SCS benefit from starting at `w0` or at yesterday's solution; Clarabel, an
  interior-point method, does not. The solutions of every run are already persisted as `.npz`; a loader
  could hand the previous run's in as a `warm_start` column and the adapter could pass it through.
  The solver *is* the bottleneck (measured above), so this is worth trying — but only once OSQP or SCS
  is made to converge on a 100k book at all, which at their defaults they do not.
- **Re-solve from the persisted spec.** `problem_specs/<portfolio>.npz` plus `chain/<portfolio>.npz` is
  everything the solver saw. A `resolve` CLI subcommand that rebuilds the problem from those files and
  compares the result with `solutions/<portfolio>.npz` turns the audit artifacts into a reproducibility
  test that needs no data sources at all.
- **Solver fallback stays out.** A silent second solver changes what the manifest says was solved. A
  visible retry — same solver, relaxed tolerances, recorded in the manifest as a second attempt — is the
  acceptable variant if one is ever needed.

## Bugs and cleanup

Not threads: known, decided, and only waiting for someone to do them.

### The canonical split can move the objective, and the verifier then refuses the portfolio

`_classify` in `engine/solve.py` replaces the solver's buy/sell pair with the minimal split
`buy = max(w − w0, 0)`, `sell = max(w0 − w, 0)`, on the grounds that the minimal split cannot make any
shipped term worse. That is false for `tax_cost` on a loss: `tax_per_dollar` is negative there, so
selling *earns*, and a sell-and-rebuy of *x* dollars in such a name changes the objective by
`(τ + 2c)·x` — profitable whenever the tax saving beats two transaction costs, which at 20–40% tax
rates and single-digit-bps costs is every loss position in a book. The solver's optimum is then full of
round trips (the guard in `tax_cost` only checks that *some* transaction cost is positive, not that a
round trip is unprofitable), the canonical split strips them, the objective the twins recompute is
higher than the one the solver reported, and `verify` fails the portfolio with an objective gap:
`VerificationError`, "objective gap 2.8e-03", nothing about why. The orders would have been right
regardless — they derive from `w` alone (`engine/orders.py`), and a round trip does not change `w`.

Found on 2026-08-29 by the profile's synthetic book (`benchmarks/profile_portfolio.py`): 500 held
names, half at a loss, 5 bps `tcost_bps`. The example data cannot show it: no losses, no cost column.

Two fixes, both small:

- **The refusal has to name the round trip.** In `_classify`, measure `min(raw.buy, raw.sell)` before
  canonicalizing; where it exceeds the verifier's tolerance, fail the solve with an error that lists the
  names the solver round-tripped and says a term rewards a wash trade — the same refusal, with its cause
  in the message. Keep the canonical split for the other case, where it is a harmless tidy-up of
  interior-point slack.
- **The shipped tax term's guard should be the real condition.** Refuse per security when
  `−tax_per_dollar > 2 · (tcost_per_dollar + cost_bps / 10⁴)` rather than when no cost is positive at
  all. The honest modelling fix is a wash-sale-aware term or constraint, which belongs to the selling
  thread above; until then the tighter guard turns a cryptic verification failure into an error that
  says what to change.

### Smaller things noticed in passing

- **The example never exercises `sector_bounds`.** `configs/example_run.json` lists the constraint, but
  every style in `examples/data/constraints.json` has `"sector_bounds": {}`, so `spec.sector_names` is
  empty and the constraint returns an empty `ConstraintSet`. The universe carries a `sector` column, but
  all three securities are `TECH`, so even populated bounds would be one group. A config that names a
  constraint doing nothing is worse than one that omits it: the manifest records it as configured and the
  verifier reports nothing about it. Give the example a second sector and real bounds, or drop it from the
  constraint list — and note that this is precisely the class of hole the acceptance harness above exists
  to make visible.
