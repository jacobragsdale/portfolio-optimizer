# Ideas

Threads for expanding the template that are not yet decisions. Each one states the problem as the
engine has it today, the options, and a leaning; none is a commitment. When a thread becomes a
decision it moves into the code and [the architecture explanation](docs/explanation-architecture.md)
and leaves this file. The last section is the exception: known defects and trailing work, decided
already and waiting only to be done. Numbers below are for a book of *N* = 100,000 unique securities,
which a business unit can exceed — measured where the text says so, estimated otherwise.

## The derived solve graph is the headline feature, and nothing here says so

The state of the art in production rebalancing is parallel builds and a sequential solve. The
sequential half is not laziness: it is what a chain-aware constraint forces. `cumulative_adv_participation`
is the ordinary example — an account may take its share of a name's daily volume only after the
accounts ahead of it have taken theirs — so account *j*'s feasible set depends on what *1…j−1*
actually traded, and the safe reading of that dependency is a line. *N* solves end to end, wall clock
*N* × solve. At the 9.8 s the measured 100k book takes, five hundred accounts is eighty minutes of
solving on a cluster that is idle for all but one core of it.

The line is a worst case, not the truth. Two accounts that cannot trade a security in common cannot
affect each other's feasible set, whatever their constraints read. The real object is a DAG, and the
engine derives it rather than being told it:

- `order_portfolios` fixes the priority order from the data (`solve_order`, or a step that computes it).
- `OverlapIndex` takes one portfolio's tradable set — the securities the side profile couples through,
  buys under `both` and `buy` — and answers which *earlier* portfolios share one, without knowing the
  portfolios still to come. Predecessors are direct and the graph is never transitively reduced,
  because a solve folds its direct predecessors' own contributions.
- Because every predecessor is earlier in the order, the graph can be grown a portfolio at a time:
  `_stream_solves` walks the order and submits each solve as the build it has reached reports, so the
  head of the book is solving while the tail is still building.
- The manifest records the shape it derived: `coupling`, `edges`, `components`, `largest_component`,
  `critical_path`.

So the wall clock is `critical_path` × solve, not *N* × solve, and independent components solve at
once, bounded only by workers. What makes that a claim rather than an assertion is `dependencies: all`,
the line, kept deliberately: a test runs the same book both ways and asserts the spec hashes, the chain
hashes, and the orders are identical. The fast schedule and the safe one agree, and anyone can check it
on their own book with one config key.

Nothing in the repository leads with this. The README's `execution` bullet describes `dependencies` in
terms of what happens when a portfolio fails; `how-to-set-the-solve-order.md` is about the key, not the
graph; the schedule's own docstring is the only place the mechanism is explained, and it is in the
engine. A reader has to reach the layout table to learn a dependency graph exists at all.

Three pieces of work, in order:

1. **Lead with it.** A paragraph at the top of the README and a section in the architecture explanation
   that states the problem (chain-aware constraints serialize a book), the derivation (overlap on the
   coupled side), and the check (`all` gives the same answer). It wants the one diagram this repository
   does not have: a book of accounts, the edges overlap actually produces, and the critical path
   through it against the line beside it.
2. **Measure it.** Nothing measures the schedule at scale — `benchmarks/profile_portfolio.py` times
   *one* portfolio, and the schedule the shipped example derives is the degenerate one (below). The
   number that carries the argument is a book-level benchmark: *N* accounts over a shared universe with
   a realistic overlap structure, reporting edges, components, critical path, and wall clock against
   `dependencies: all` on the same cluster. Until that exists the feature is a design claim, not a
   result.
3. **Say where it degenerates.** Overlap is on *any* shared tradable security, so a book whose accounts
   all buy from one universe is a complete DAG and the critical path is the line again. The shipped
   example is exactly that case and says so in its own manifest — 100 accounts over three securities:
   `edges 4950, components 1, largest_component 100, critical_path 100`, every solve waiting on every
   earlier one. That is the honest case, and the one a reader will have. The win as it stands belongs
   to books partitioned by strategy, universe, or restriction list, where the components are real.

Which points at the thread that would make it bite everywhere. The engine couples two portfolios when
their tradable sets intersect, because a constraint is opaque data and that is the widest safe reading.
But a chain-aware constraint knows more than the engine does: `cumulative_adv_participation` couples
only through names whose budget can actually bind, and on a 100k universe rebalanced at ordinary
turnover that is a small fraction of the tradable set. If the constraint contract — the behavioural
union in the constraints thread above, which already declares *whether* a constraint reads the chain —
also declared the securities it couples *through*, `OverlapIndex` would index those instead, and a
single-universe book would stop being one component. That is the same declaration the engine already
asks for, made one field more specific, and it is what turns the graph from a structural win into a
numerical one.

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

The same book under each side profile (`--sides`, measured 2026-08-29 after the one-sided profiles
landed; `both` re-measured the same day, bitwise the same orders as before them):

| `sides` | Variables (*P*) | Clarabel | Iterations | Solve task end to end | Peak RSS | Note |
|---|---|---:|---:|---:|---:|---|
| `both` | 400k | 7.5 s | 29 | 8.0 s | 2.1 GB | tracking, tax, tcost |
| `buy` | 200k | 2.1 s | 12 | 2.8 s | 1.5 GB | `tax_cost` dropped: it reads `sell` |
| `sell` | 200k | 3.6 s | 20 | 5.7 s | 1.3 GB | `cash_bounds` `[0, 1]`; the gains-only book sells nothing |

One variable per name instead of three and no trade identity: Clarabel 3.5× faster under `buy`, 2×
under `sell`, in the 2–4× band the design predicted, with a third less memory. (The benchmark's own
`get_problem_data` + `solve_via_data` stage reports `unbounded` after 9 iterations under `both` while
`problem.solve` on the identical problem is optimal in 29; that was so before the profiles too, so it
is a quirk of driving cvxpy by hand in the harness, not of the engine's solve — the end-to-end row is
the one to read.)

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

Clarabel spends about 0.26 s per interior-point iteration here, in a KKT factorization over *A*. Every
shipped term is now linear, so *P* is absent altogether and this is an LP with 300k variables and 900k
rows; the per-iteration cost is the sparse factorization, not the objective. (The numbers in this
section were measured while `tracking_error` still shipped and made *P* diagonal, which is a smaller
difference than it sounds — re-measure before leaning on them.) Things to measure, in the order they are cheap:

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

The formulation was the largest solve-time win on this list, and it came with the design rather than
from solver tuning: the one-sided profiles (table above) have one variable per name and no trade
identity. What remains of it is `both`, which is deferred.

### The result carries the spec back

Every `PortfolioResult` returns to the client with its `ProblemSpec` inside — 20 MB at 100k names now
that the sector matrix is sparse, 10 MB of it the spec's own vectors — because the client persists
`problem_specs/<portfolio>.npz` for `verify` and `diff-manifests`. A thousand portfolios is 20 GB into
one process over one NIC, held until each is written. The spec is also exactly what the worker already
has. Options, none decided: write the `.npz` from the worker when the run directory is a shared or
object-store path and return only the hash; or return the spec lazily, as a Dask future the client
pulls while persisting, so the transfer overlaps the solves instead of following them. The
`Contribution` a dependent solve receives is a few kilobytes and is unaffected either way.

### Three things are called "build"

For the record, since the word is overloaded:

1. **The spec build** (`engine/build.py`): rules, Decimal arithmetic, alignment to the sorted universe,
   the one Decimal→float64 conversion. Pure numpy out. Runs in workers, every portfolio at once.
2. **The expression tree** (`engine/solve.py` → the terms and constraints → `cvx/adapter.py`): a few
   dozen cvxpy nodes holding references to the spec's arrays. Milliseconds.
3. **Canonicalization** — inside `problem.solve()`: DCP verification, the reduction chain to the
   solver's conic or QP form, coefficient extraction into sparse matrices. 0.6 s at 100k names with
   the shipped terms.

### Two-sided coupling, if it ever comes

Sell-only *runs* are built: within one they couple through sells only, the exact mirror of the
buy-only run (see *Sides*, below, for what is still open). What is not planned, and possibly never, is a run in which buys and
sells couple *with each other* across portfolios — the deferred two-sided profile made chain-aware on
both sides. Recorded so the one-side guarantee is not silently load-bearing. Everything such a run
would add couples across the sides, per security:

| Effect | Produced by | Consumed by |
|---|---|---|
| ADV budget spent by sells | sells | buys and sells |
| Wash sales: do not buy what an earlier account sold at a loss | sells | buys |
| Wash sales, mirrored: do not sell at a loss what an earlier account bought | buys | sells |
| Internal crossing: an earlier sell of *X* makes a later buy of *X* cheaper — a *term* | sells | buys |

Each simplification the one-side guarantee buys (see the architecture explanation) would need to un-simplify, in this order:

1. `ChainState` carries one side today (`traded_shares`, the side the run couples through); it would
   need both, and the fold would read both.
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

- **Keep the factor risk term structured when it returns.** `sum_squares(F½ · B · w) + sum_squares(√D ∘ w)`,
  never a dense *N* × *N* covariance (80 GB at 100k). The 50 × *N* loadings are the one genuinely dense
  block, 40 MB, and they set the floor on every size above the moment the term exists.
- **Re-profile when a term changes.** `uv run python benchmarks/profile_portfolio.py --securities 100000`
  prints the table above for the shipped config; the split between canonicalization and solve is
  solver- and structure-dependent, and a factor term will not look like the diagonal *P* measured here.

## Sides: what is still open after the one-sided profiles

`sides: "buy" | "sell" | "both"` is built (2026-08-29) and documented: what the profile owns, why the
side is a config value, and what it costs the solver are in
[the architecture explanation](docs/explanation-architecture.md#the-side-a-run-trades-is-one-object);
the operating guide is [how to run one side](docs/how-to-run-one-side.md); the numbers are in the
table above. Two things are not decided:

1. **A constraint the starting point already violates** — a name over `max_weight` in a buy-only run,
   a holding under a floor in a sell-only run — is each constraint's own call: it declares `accept`
   (hold the name where it is and do not worsen it; the `ub = max(w0, cap)` shape) or `infeasible`
   (fail the portfolio). The declaration is per constraint, never per run. Today every such start is
   infeasible, and `diagnose_infeasibility` lists the names (`SideProfile.infeasible_starts`), which is
   where the accept policy lands.
2. **Sells do not feed buys today**, so nothing crosses between a sell run and a buy run; see below.

`both` is what existed before the profiles, extracted and not extended; the wash-trade defect under
*Bugs and cleanup* belongs to it alone.

### Future enhancement: the sell run feeds the buy run

Not needed now — the sell process and the buy run do not exchange anything today. When they do, the
handoff is three inputs, all content-hashed like every other dataset, so the buy run stays a pure
function of a snapshot and `diff-manifests` works across the boundary: holdings *after* the sells
(already just "holdings as of"), the cash raised as the cash the buy run invests, and the sell run's
ADV usage as an `adv_consumed` column that `adv_remaining` subtracts alongside predecessors' buys —
one line in the build, no chain machinery. Two runs, two manifests, one program.

## Solving without cvxpy

Done 2026-08-29: `solve` is a configured step and `pro_rata_fill` the shipped pure function — see
[how to replace the cvxpy solve](docs/how-to-write-a-solve-step.md). What remains of the thread is in
*Constraints: one contract, three ways to author it*.

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
book — a handful of securities, two or three portfolios, their holdings, alphas, and style constraints
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

## Constraints: one contract, three ways to author it

A constraint today is a `ConstraintStep` in the config (`config/models.py`): a strict model with a
`kind` — `function`, the only kind so far — and a `label`, unique among the run's constraints and
defaulting to the step's bare name. The engine resolves it to a `ResolvedConstraint` (`config/steps.py`)
and asks it three things only: its label, whether it reads the chain (`reads_chain`, what the
dependency graph is derived from), and — through the twin table in `check.py`, keyed by qualified name
— how to verify it; every check and manifest record carries the label. What the function *does* is
the solve step's business, and the shipped `cvxpy` step calls it for atoms. Its numbers still live
somewhere else: the `constraints` dataset, per portfolio, typed by a fixed `StyleConstraints` model.
So adding a limit that is structurally identical to one that exists — country bounds where there are
sector bounds, an issuer cap where there is a single-name cap — still means editing `terms.py`, writing
its twin, extending `StyleConstraints`, and shipping a release. The shape is code, the numbers are data,
and the two are joined only by convention.

Three groups need to author constraints, and (2026-08-29) none of them is well served by that: portfolio
managers, who are not in the code and will edit the template through a GUI; quants, who want a Python
function for what no grammar says; and other optimizer types, which want a *constraints builder* — one
function that turns loaded tables into the whole constraint set. The design that serves all three is not
three features. It is one contract with three authoring surfaces.

### The contract is behavioural, not structural

The first draft of this thread (2026-08-29) fixed a *shape* — an affine row block with a sparse matrix
and bounds — as the contract, so that one cvxpy interpreter and one numpy verifier could share it.
That over-commits the engine to affine constraints at the moment they are becoming pydantic objects
consumed by libraries the engine does not own (the desk has a class that takes constraints as a
dictionary and builds the cvxpy problem itself), and it makes an optimization with no constraints a
special case. So the contract is what the *engine* — build, schedule, dispatch, verify, manifest — has
to know about a constraint, and nothing a solver does with it:

1. **It is a strict pydantic model.** Serializable, hashed into the config, JSON-schema-able: all the
   GUI and the manifest need. `constraints` becomes `tuple[ConstraintSpec, ...]`, a discriminated union
   on `kind`; today's `StepSpec` — a name and params — is its first member.
2. **It declares whether it reads the chain, and which quantities.** The dependency graph is derived
   from that declaration alone; the engine never looks inside for a matrix. This is the one clause
   that cannot be relaxed: the graph is the engine's, and it cannot be inferred from an opaque object.
3. **It has a unique label**, checked at resolve; the report, the manifest, and the acceptance
   harness key on it. `params_sha256` tells two instances apart for provenance; the label, for people.
4. **It *may* offer `residual(spec, solution, chain) -> F64`.** If it does, the verifier checks it; if
   not, it is `unverified` — today's posture for custom constraints, made uniform. Agreement between the
   residual and what the solver was told is the author's responsibility, as it is for the shipped
   twins today.

Nothing about vectors, matrices, or affineness. `constraints: []` is a valid run whose verifier has only
the side profile's identity checks to make, not a special case.

### The solve step is the interpreter

Everything shape-specific belongs to the *consumer*. The solve step receives a `SolveRequest` —
`spec`, `chain`, `profile`, `terms`, `constraints`, and the `solver` block (`solving.py`) — plus its own
`params`, and returns a `SolveResult` whose `w` is aligned to the spec; the side profile turns `w` into
the trade; the verifier is what makes the answer trustworthy; the manifest records the step where it
records the solver and its version. Three interpreters, and the engine treats them identically:

| Solve step | Consumes constraints how | Verification |
|---|---|---|
| The shipped cvxpy step (today's adapter) | a registry of `to_cvxpy` per model kind; an unknown kind is a resolve-time error *for this step* | each model's `residual`, plus the objective against the terms' twins |
| The desk's library, `solve: "firm.optim:solve"` | as dicts (`model_dump`), building cvxpy itself | each model's `residual`; no objective comparison unless the step reports one |
| A pure function — a pro-rata fill, a cash raise | reads them or ignores them | each model's `residual`; the configured terms evaluated after the fact as a report line, which is how a heuristic is compared with the optimizer on one book |

Side compatibility (a model naming `sell` in a buy-only run), the start policy (`accept` or
`infeasible` is a *field* on the model; the logic that applies it lives in whichever step consumes the
model), DCP: all the interpreter's business. What is *not* the interpreter's business, and has to be
said plainly: the step returns `w` and nothing else — it writes no files, reads no clock, sees no other
portfolio; infeasibility is an exception with a message, which the engine records as a failure at stage
`solve` without trying to explain. A `Solution` from a step that named no solver records the step's
bare qualified name as `solver` and its package version as `solver_version`, with no iterations and no
objective, and the verifier skips the objective comparison for it.

That makes the solve step the engine's principal extension seam, and its contract has the care the
term and constraint contracts got — signature checked at resolve, engine arguments by fixed name,
params validated. It is deliberately *not* dry-constructed at `validate-config`: a firm's step may
reach a service, and the one-security dummy is not a problem worth solving.

### What this leaves of the first draft

- **The row block is demoted** from the contract to one convenient family of shipped models: a few
  `kind`s that carry a matrix and a bound — `bound_on`, `group_bound`, a flagged subset — with
  `residual` and `to_cvxpy` implemented once for the family. Useful, not load-bearing.
- **The declarative JSON grammar *is* those models.** No intermediate representation to compile to: a
  `{"kind": "group_bound", ...}` object in the config is validated as that model, rendered by the GUI
  from that model's schema, hashed as that model, and interpreted by whichever step consumes it.
- **"What the solver saw" is persisted as the models' JSON** beside the spec, not as matrices. Reading
  it needs no code; verifying it needs the models' own `residual`, which is the shipped package's or
  the firm's.
- **Property 7 survives unchanged and is still the fork to decide.** Where the *numbers* come from —
  a literal, the per-portfolio style, a spec column or chain quantity — is independent of shape. The
  leaning stands: the run config is the schema of the style; a model saying
  `"at_most": {"from_style": "country_bounds"}` makes `country_bounds` a required, typed key in every
  portfolio's style, validated at assembly before any build, with a run-wide default and a
  per-portfolio override. It is what lets a PM add a limit without a release, and what makes a
  missing number a config error rather than a silent zero.

### The declarative grammar

```json
"constraints": [
  {"label": "country_caps", "of": {"sum": "w", "by": "country"}, "at_most": {"from_style": "country_bounds"}},
  {"label": "no_new_tobacco", "of": "buy", "where": {"flag": "is_tobacco"}, "at_most": "0"},
  {"label": "issuer_cap", "of": {"sum": "w", "by": "issuer"}, "at_most": "0.05", "at_start": "accept"},
  {"label": "adv", "of": "buy", "at_most": {"chain": "adv_remaining"}},
  "long_only",
  {"name": "mypkg.limits:risk_budget", "params": {"sigma": "0.04"}}
]
```

**The shipped grammar stops at affine.** An expression is one decision vector, optionally summed,
summed by a categorical column, or restricted to a boolean flag, compared against a literal, a style
value, a spec column, or a named chain quantity. Each spelling is one model kind with one `residual`
and one `to_cvxpy`; a Python function that says the same limit is refused, so the two cannot drift.
The published JSON Schema grows a discriminated union, which is also what a GUI renders forms from. A
model naming a column the universe does not carry cannot be caught at resolve, which runs before data
loads; the check moves to a gate just after assembly, with the same collected-failures shape, so the
run still dies before it does any work.

The spec generalizes `sector_matrix` to `groups`: one sparse membership block per categorical universe
column (a megabyte each at 100k names, however many groups), built the way the sector matrix already
is, and `sector_bound` becomes `group_bound(column="sector")` — the shipped constraint as one instance
of the general one. Half of this landed on 2026-08-30: the *numbers* left the spec, so `sector_bound`
is one row per sector carrying its own band and reading only the membership (`spec.sector(name)`, one
sparse row). What is left is generalizing the column.

### Where the line has to hold

Someone will ask for a conditional, then for arithmetic between columns, then for a soft version with a
penalty in the objective. Each is reasonable alone and together they are a worse cvxpy expressed in JSON,
with a DCP checker of our own. The rule: **anything that is not affine in one decision vector against
constant data is a Python function, no exceptions** — rows if it can be, atoms if it must. A soft limit
is a term, and terms stay in Python. Integer constraints — minimum trade notional, round lots, a cap on
the number of holdings — are outside both paths and should be said so out loud: they are not convex,
they are not what `orders.py` rounding already approximates, and a grammar that admitted them would
promise something the solver will not deliver.

### Order of work, and what it composes with

1. `residual` *on* the model. The first kind, the labels, and the solve seam are built (above); the
   shipped constraints' twins are still the table in `check.py`, keyed by qualname, and move onto the
   model when the second kind arrives and needs it.
2. `groups` on the spec and the row-block family (`bound_on`, `group_bound`), with `sector_bound` as
   an instance — its band already lives on the row, so what remains is the column.
3. The style schema derived from the config (property 7).
4. The GUI, which by then is a form renderer over the schema plus a call to `validate-config`.

It composes with the acceptance harness, whose expectation vocabulary is the labels and residuals the
models already carry, and it absorbs the pure-function solve: not a mode, the third interpreter.

## A GUI edits the template

Raised 2026-08-29: portfolio managers will edit the run template through a GUI rather than a text editor,
and PMs are the people who change constraints. What that asks of the engine, so the GUI is a client of it
and not a second implementation:

- **The JSON Schema is the GUI's contract.** Forms are rendered from it — which is why every property
  must be documented (a test enforces that already), why the declarative constraint grammar must be a
  discriminated union rather than a free object, and why step names and parameter models are the stable
  surface a release must not break casually.
- **Validation stays in the engine.** The GUI calls `validate-config` on save; it does not re-implement
  rules. `validate-config` wants a `--json` mode with structured failures (path, message) so a GUI can
  put each one beside the field it concerns, and an exit-code contract it already has.
- **Two things get edited, not one.** The template — which constraints apply and their run-wide
  defaults — and the per-portfolio style — the numbers. The style form is generated from the template's
  declarations (property 7 above), so adding a constraint in the template is what makes its number
  appear in every portfolio's form.
- **The GUI never touches Python.** A constraint the grammar cannot say is a quant's job; the GUI shows
  it by name and params and does not offer to edit its body.

## Order aggregation as a step: from per-portfolio orders to what goes to the street

The run ends abruptly today. Every solved portfolio's orders are concatenated, sorted, and handed to the
sink in one frame (`_publish` in `engine/runner.py`), and that frame is simultaneously two things it should not be:
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

## Timing every stage, so a run can be drawn

What a finished run says about its own time today: each dataset's `load_time_s` and `batches`, the
cluster's `provision_started_at`, `first_worker_ready_at`, and `closed_at`, and per portfolio the
`solve_time_s` the solver reports. Everything else has to be reconstructed from log lines, which carry
a timestamp and a stage but no duration and no host. So the question a slow run actually raises — where
did the wall clock go, and what was waiting on what — has no answer in the artifact: the sum of every
`solve_time_s` is well under the run's elapsed time, and nothing accounts for the difference, which is
some mix of queueing behind predecessors, builds, a worker that had not joined yet, and the sink.

What it takes is small. A start and end instant on every task the engine already submits, the host it
ran on, and the stage names the code already uses (`load`, `assembly`, `build`, `solve`, `verify`,
`orders`, `sink`), recorded per portfolio the way `solve_time_s` is now. The build and solve tasks
already measure their own phases when `benchmarks/profile_portfolio.py` drives them — validate, rules,
spec build, content hash, canonicalization, solve, unpack, verify, orders — so recording the same
breakdown from a real run would put the table at the top of this file within reach of any book, rather
than of one synthetic one, and would say whether a per-portfolio dataset's batches really do overlap
the global loaders on a book where it matters.

Two constraints shape where it lands:

1. **Timing must not touch identity.** Two runs of one config are byte-identical where it matters and
   `diff-manifests` says so; a duration differs every run by definition. The manifest already carries
   fields nobody diffs — the cluster's timestamps, the settings block — because `diff_manifests`
   compares an explicit list of identity fields rather than the whole document. Timing joins them under
   that rule: in the manifest, never in the config hash, never in the diff.
2. **It is timing, not profiling.** Wall-clock spans around work the engine already delimits, a few
   dozen per portfolio, a `perf_counter` call each. Not a sampling profiler, not per-function
   attribution, nothing that changes how the work runs, nothing to switch on.

The leaning on the visual: write the spans beside the manifest in the Chrome trace format
(`{"name", "ph": "X", "ts", "dur", "pid", "tid"}`), because it is a list of flat objects to produce and
it opens in `chrome://tracing` or Perfetto with a row per worker and every portfolio's build and solve
where it happened — a picture of the run for the price of a JSON file, with no viewer to write. A
`portfolio-optimizer timeline <manifest>` subcommand then prints the same spans as an ASCII waterfall
with per-stage totals, which is what a terminal and a CI log can use. Both read the recorded spans;
neither is a second source of truth.

What it makes answerable, none of which the manifest can support today: whether the workers were busy
or idle (so `PORTFOLIO_OPTIMIZER_MAX_WORKERS` can be set on evidence), whether the chain's critical
path — which the `schedule` block already counts in edges and depth — is where the wall clock actually
went, and which stage a regression landed in when a run that took nine minutes last week takes twenty
today.

## A live dashboard: watching a run rather than reading about it afterwards

Everything a run says about itself today it says when it is over. The manifest is written in the last
stage, `diff-manifests` compares two finished runs, and the timeline above would be read from a file
that only exists once the run has ended. While a run is going, the only channel is the log: one line
per stage transition, carrying a `run_id` and a `stage` but no notion of how much is left. On the
example book that is fine. On a book of five hundred accounts where the chain's critical path is deep
and one solve takes minutes, "is this progressing or is it wedged, and on what" is unanswerable
without tailing a log and reconstructing the schedule in your head.

What makes this tractable is that the engine already knows the shape of the answer before it starts.
The config names every stage a portfolio will pass through; the portfolio list is known after the very
first loader; and the dependency DAG — which portfolio waits on which, and the critical path through
it — is derived at `schedule`, before any build runs. So the denominator exists early: *N* portfolios ×
the stages this config declares, with the edges between them. A dashboard is not inferring progress, it
is filling in a grid the run has already described.

Three ways to serve it, in rising order of what they cost:

1. **Lean on Dask.** The distributed scheduler already publishes a dashboard with task streams and
   worker occupancy, and the run provisions its own cluster, so the URL is knowable. Nearly free, and
   it answers "are the workers busy". It cannot answer anything in the engine's own vocabulary — its
   tasks are `build`/`solve` futures, not portfolios and stages, and a run whose backend is not Dask
   has nothing.
2. **Structured events on a bus, rendered by a terminal UI.** The log already emits at exactly the
   right moments with `run_id` and `stage`; making those emissions structured events (a portfolio id, a
   stage, a transition, an instant) and giving the runner a pluggable sink is a small change to code
   that exists. A `portfolio-optimizer watch <run_id>` then draws the grid live. The same event stream
   is what the timeline section above wants to persist, so the two should be one mechanism with two
   consumers — live and recorded — rather than two.
3. **A served web page.** The grid, the DAG with the critical path highlighted, per-stage durations
   filling in, failures in place. Genuinely the best artifact, and the one that needs a server, a
   transport, and a front end to maintain — none of which the template has today.

The leaning is (2) built on the same spans (1) already implies: define the event, give the runner a
sink for it, and let the recorded stream feed the timeline while a live stream feeds a terminal grid.
That keeps one source of truth, keeps the web page as a later consumer of an interface that already
exists rather than a rewrite, and holds to the rule the timing thread sets — **observability is never
identity**: no event, duration, or progress reading may reach the config hash or `diff-manifests`.

Two things to settle before any of it. **What a stage means for a portfolio that is waiting.** A
portfolio blocked on a predecessor is not the same as one being built, and a grid that shows both as
"not done" hides the thing worth seeing; the schedule knows which, so the event vocabulary has to carry
`blocked` as a state distinct from `pending`. And **what the live channel is when the run is on a
cluster.** Events originate on workers, and the obvious answers — the scheduler as a relay, or workers
writing to shared storage the watcher polls — differ in whether a dashboard can attach to a run already
in flight, which is exactly the case that matters most.

## Other threads

- **Batch-level pipelining in the load DAG.** A dataset that depends on a `per_portfolio` dataset
  waits for every batch today and receives the concatenated frame; letting batch *i* of a dependent
  start when batch *i* of its dependency lands would overlap the two fan-outs. Worth building only
  when a real book shows the wait matters — the manifest's `started_s` per dataset is where it would show.
- **Tax lots.** Holdings are security-level with an average cost. Lot-level tax needs one sell variable
  per lot (*N* + *L* variables, lot-to-security aggregation as one more sparse matrix), changes the
  holdings schema and `orders.py`, and is the first extension that would reopen the question of where
  the problem is built — settled for now on the numbers in the architecture explanation's "left out"
  section.
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
names, half at a loss, 5 bps `tcost_bps`. The example data cannot show it, and not by accident — every
lot in `examples/data/holdings.csv` is at a gain or flat, because an account holding a loss makes the
shipped run fail this way.
Only the two-sided profile can have this problem — a one-sided run has one vector and no round trip to
strip — so it belongs to a deferred path, and the first fix below is the one still worth doing.

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
