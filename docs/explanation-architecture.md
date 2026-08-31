# Explanation: how the engine is built and why

This page explains the design decisions behind the template — what the engine promises, where those
promises are enforced, and what was deliberately left out. For the same engine narrated in execution
order, stage by stage, see [the life of a run](explanation-run-lifecycle.md).

## One convention instead of a framework

Quant developers who clone this repository want to write Python, not learn a plugin system. So the only
mechanism for extending the engine is: write an ordinary function — in a designated module of this repository or in any installed
package — and name it in JSON. The engine's resolver (`config/resolve.py`) does the work a framework would normally push onto the
author — it imports the function, checks its signature against the contract for its kind (nine kinds:
loader, assembly step, rule, solve-order step, term, constraint, solve step, sink),
validates the JSON parameters against the function's own `Params` model, and detects the optional
`chain` argument by name — and it does all of this before any data is loaded. It then checks the
solver and constructs every term and constraint once, against a one-security dummy spec, under the
run's side profile, so a step that raises or reads a side the run lacks is refused too. The same
resolution runs in every process that will solve — at `validate-config`, at the start of `run`, and
on every worker before it does any work — so all of them apply identical checks. A mistake surfaces
as a config error with the function's qualified name, not as a traceback halfway through a run.

The convention has a second purpose: auditability. Because every step is a named function, the manifest
can record its qualified name and the hash of its source text. A run can be traced to the exact business
logic that produced it, and two runs can be compared to find whether code, data, or the solver changed.

## Loading is the slow part, so it is concurrent and metered

In production the datasets come from APIs and databases, and waiting on them dominates a run. The load
stage is therefore asynchronous and runs the dependency DAG the config declares: each dataset starts
the moment the datasets its entry depends on have loaded — with no dependencies, immediately — with
`async def` loaders on the event loop and plain ones in worker threads. The DAG is declared rather
than inferred, because the engine cannot see inside a loader: only the config can say that compliance
needs the book's ids while the security master needs nothing. `portfolios` is just the node the
per-account inputs depend on, so the stage costs its longest chain — in the example the security
master's scan — rather than the book of record plus everything behind it. Loaders are the only step
kind that may be async; everything downstream is pure and synchronous.

A backend has limits, and a source that answers one portfolio per call will hit them on a large run.
Backends also differ: a vendor API may tolerate eight concurrent requests where a warehouse takes
thirty-two. So the input that is cut into per-account calls carries `max_in_flight`, and the engine —
which already owns that partition — holds a slot for the length of each call. One number per input,
and no loader counts its own requests: a bound the loader enforces is a bound the engine cannot
schedule around, and one shared between inputs is budget arithmetic in a config file. A source that
needs pacing rather than a concurrency cap gets it in the client, beside its retry and backoff. The
manifest records each input's load time and its batch count, so "why was this run slow" has an answer.

## Assembly is a step kind, and the bundle is two tables

Combining datasets is business logic — which vendor's price wins, how two custodians' files become one
holdings table, what a z-score is normalized against — so it follows the same convention as everything
else: an ordinary function, `(frames: Frames[, params]) -> Frames`, named in the config's `assembly`
list, run once per run over every loaded dataset, and recorded in the manifest with its source hash,
its parameters, the row count of every dataset before and after, and the columns it added. The
shipped `join`, `union`, `select`, and `drop` cover the recurring shapes; anything else is a function
of the same shape in the desk's package. Making assembly a step kind rather than a fixed join
vocabulary is what lets a run's data preparation be audited the same way its rules are, and it is why
a dataset the engine does not know can still be used: it stays visible to every step by name and is
carried into each portfolio's bundle as an extra for a rule to read.

The bundle itself is deliberately two tables, not one: `holdings` (owned, with cost basis) and
`universe` (buyable, with price and liquidity). Both accept any analytics columns beyond their schemas,
a held name need not be buyable, and `PortfolioData.optimizer_frame()` stacks them into the single
frame an optimizer wants — holdings rows then universe rows, over the union of columns, with typed
nulls where one side lacks a column. The one invariant this needs is that a column present on both
tables has the same dtype on both, so the bundle checks it on every construction and names the column
when it fails. That check is what makes "attach a score to both tables" safe to do in a loader, a step,
or a rule: whichever produced a mismatch is the one that fails.

## Two conversions, in two places

Money enters the engine as `Decimal` and stays that way through loading, assembly, and rules; frame
schemas reject a float where a Decimal belongs. The optimizer needs float64, so `engine/build.py` is the
one place the conversion happens — after current weights and tax per dollar have been computed exactly.
Orders need exact quantities and notionals again, so `engine/orders.py` is the one place float64 becomes
shares and `Decimal`, using the exact prices carried alongside the spec rather than the float copies
inside it. Keeping both conversions in single, tested functions is what makes "the notional in the
order is exactly quantity times price" a guarantee rather than a hope.

## The spec is pure data

`ProblemSpec` holds every input the solver will see as plain numpy arrays — and the sector membership as a
sparse matrix, one nonzero per security, since dense it was most of every large spec — aligned to a sorted list of
securities, plus a content hash. cvxpy objects are created inside the `solvers.cvxpy` step — and, once,
during the dry construction at resolve — and never leave the process that made them. This
buys three things: the spec is built on a worker and stays there until it is solved; it can be persisted as an `.npz` file that an auditor can open without the solver stack; and the hash pins down
exactly what was optimized, so a change in any input changes the hash and shows up in `diff-manifests`.

## The side a run trades is one object

A run's `sides` selects a *side profile* (`domain/sides.py`, with its cvxpy half in `cvx/sides.py`):
the one place in the engine that knows what the side means. The profile makes the decision variables
and supplies the trade identity to every solve, turns the solver's weights into the reported buy/sell
split, names the tradable set the dependency graph and the chain state are built from, reduces a
solved portfolio to what a dependent receives, says which starting books the side cannot trade out of,
and hands the verifier the identity's numpy twin. Nothing else branches on the side.

There are three profiles. `both` is the two-sided problem: `w`, `buy`, and `sell` are all variables,
bound by `w = w0 + buy − sell`, and portfolios couple through buys. `buy` and `sell` are one-sided:
`w` is the only variable, the trade is an affine expression of it (`buy = w − w0` under `w ≥ w0`, or
`sell = w0 − w` under `w ≤ w0`), and the other side does not exist — a term that reads it fails at
`validate-config`, not on a worker. That is why the side is a config value rather than a pair of
bounds: bounding `sell` to zero would keep three vectors and the identity in the KKT system, while
removing it makes the problem a third the size and leaves no way to express a wash trade. Measured at
100,000 names, Clarabel takes 2.1 s under `buy` and 3.6 s under `sell` against 7.5 s under `both`
(`IDEAS.md`). The shipped terms that mean "the amount traded" read `x.trade` — `buy + sell`, `buy`, or
`sell` — and the ADV constraint's chain half reads `x.coupled`, the amount traded on the side the run
couples through; both exist under every profile, so a term written against them runs anywhere.
`coupled` exists because without it that constraint would have to say `buy ≤ remaining` under `both`
and `sell ≤ remaining` under `sell`, and its numpy twin would need the same quantity — which is why the
verifier's twins receive the profile and read `profile.coupled(solution)`.

The two one-sided profiles are exact mirrors, and that is testable: the reflection `w' = 1 − w` maps a
buy-only book onto a sell-only one — bounds, cash, sector bounds, ADV budget, the chain, all of it —
so `tests/engine/test_solve.py` solves a book under `buy` and its reflection under `sell` and asserts
the answers coincide. That symmetry test is what the design asked for, and it is cheap because the
profile is one object.

A one-sided run can move cash one way only — a buy-only run lowers it, a sell-only run raises it — and
The cash bounds keep their meaning as the cash after the run, so a book that starts on the wrong side of
its bound is infeasible, and the infeasibility diagnosis says so in words. The same holds for a name
held past a bound the side cannot move it back inside (over its cap in a buy-only run, under its floor
in a sell-only one): the profile lists the names, and a per-constraint policy for accepting such a
start is the recorded next step.

## The solve step is the interpreter

What the engine knows about a constraint is deliberately little: it is a strict model in the config
with a unique label, it declares whether it reads the chain (which is what the dependency graph is
derived from), and the verifier may know how to check it. What a *solver* does with it is not the
engine's business. So the solve is a configured step — `solve`, `cvxpy` by default — that receives
everything a solver could want (the spec, the chain, the side profile, the resolved terms and
constraints, the cvxpy options) and returns weights. The shipped cvxpy step builds and solves the
problem; the shipped `pro_rata_fill` is a numpy function with no objective at all; a firm's own
library that builds the problem from the constraint models its own way fits the same contract. The
engine treats every answer identically: the side profile turns weights into a trade, the verifier
re-checks the shipped constraints, and the manifest records what solved it. This is what keeps the
engine agnostic to the shape of a constraint and to whether cvxpy is involved — the guarantees are
the verifier's, not the step's — and it is the seam the next constraint models will be interpreted
through.

## Verification is independent of the solver

After every solve, `engine/check.py` recomputes each constraint's violation and each objective term in
numpy and compares the total with the solver's reported objective. It does not import cvxpy — a test
enforces that — and its tolerances are a hundred times looser than the solver's, so a pass is a genuine
statement about the solution rather than a restatement of the solver's own convergence check. One
violation tolerance bounds every residual, the side profile's identity checks and the constraints alike;
the objective comparison has its own pair. Custom terms and constraints have no numpy twin unless the
author adds one; they are listed as `unverified` in the manifest rather than silently trusted, and a
solve step that minimized nothing reports no objective, so that comparison is skipped for it.

One consequence surfaced during development: with no term charging for trading, an interior-point solver
may return a buy/sell split that nets to the right trade but is not minimal — a free "wash trade". The
weights were right; the split was not. So under `both` the engine reports the minimal split for the
solver's weights — `buy = max(w − w0, 0)`, `sell = max(w0 − w, 0)` — and the verifier's complementarity
check confirms it. That tidy-up is harmless only while no term *rewards* a round trip; the shipped
`tax_cost` does, on a loss position, and then the solver's optimum carries round trips the minimal
split strips out, the recomputed objective no longer matches the reported one, and the verifier fails
the portfolio without saying why. That is a known defect of the two-sided profile, recorded with its fix
in [`IDEAS.md`](../IDEAS.md#the-canonical-split-can-move-the-objective-and-the-verifier-then-refuses-the-portfolio);
the one-sided profiles have one vector and cannot contain a round trip.

## Rounding

Whole-share rounding is the point where a mathematically feasible answer meets reality. The engine rounds
to the nearest share, then down to lot multiples, then clamps sells to what is held. Rounding toward zero
was considered — it can never breach a cap — but solver noise of about 1e-8 in weight space turns an
exact 1,250-share answer into 1,249, which is worse in practice. Instead the rounding drift is measured
against the solved weights and bounded by what one lot and one dust-filtered trade can cost; exceeding the
bound fails the portfolio.

## A run couples through its one side, so the schedule is a graph

Portfolios in one run often depend on each other: the second portfolio to trade a thinly traded name
should see how much of its daily volume the first already used. The engine allows exactly this kind of
dependency and no other — **what a higher-priority portfolio traded on the side the run couples
through may limit what a later one trades there; nothing else reaches anyone.** Under `both` and `buy`
that side is buys, under `sell` it is sells; a two-sided run's sells reach no one. That is a product
decision (2026-08-29), and it is what makes the schedule derivable instead of configured.

![Eight accounts: the line dependencies-all forces, and the graph overlap derives — three components, critical path three, identical orders either way](images/derived-schedule.svg)

Every portfolio builds at once, chain-free: rules never see other portfolios. A build reports its
**tradable set** — the securities the profile lets it trade on that side: buyable (`ub > w0`) or
sellable (held, `lb < w0`) — and its solve-order key, a priority from an optional `solve_order` step or
the portfolios frame's column, ties broken on `portfolio_id`. From those the engine derives the
dependency graph (`engine/schedule.py`), and the edge test is directional: portfolio *j* depends on
every higher-priority *i* whose tradable set intersects what *j*'s own chain readers *consume* — the
scopes of *j*'s typed chain constraints (`domain/constraints.py`), the whole tradable set when
anything opaque might read the chain (a function-convention row, a chain-aware term, a solve step
other than the shipped one), and nothing at all when nothing does, in which case *j* waits for
nobody; with no chain-aware step anywhere there are no edges. The
graph is never transitively reduced — a solve folds its *direct* predecessors' own trades, so every
overlapping earlier portfolio stays a direct dependency. Every predecessor is earlier in the order, so
the graph is grown a portfolio at a time as builds report rather than derived once they all have: a
solve goes in while the rest of the book is still building. There is no execution mode to choose: the
graph replaces one, and the manifest records its shape — edges, components, the longest chain of
solves — so a slow batch explains itself.

Each solve folds its predecessors' orders on that side into a `ChainState` — `traded_shares`, aligned
to its own securities and **masked to its own tradable set**. The mask is load-bearing: a predecessor's
trades lie inside *its* tradable set, the mask keeps only *this* portfolio's, so what a solve sees is a
function of the overlapping predecessors alone — the same array whether every higher-priority
portfolio was folded or only those sharing a tradable name. That is the exactness argument, and it
carries over from buys to sells verbatim; a property test asserts it: the graph schedule and the total
order produce identical orders and identical chain hashes. Order rounding clamps a buy to the room
under its upper bound and a sell to the shares held, so solver noise can never produce a trade the
graph could not have seen, and the pipeline asserts every order is on a side the run trades and inside
the tradable set. `execution.dependencies: "all"` keeps the total order available for diagnosis.

Dask enforces the graph: a solve is submitted with its predecessors' contributions — their order rows
on the coupled side — as dependencies and runs where its build lives. Outcomes are classified in solve
order whatever finished first, so the number of workers and the order in which they finish never affect
the output. Coupling across sides in one run — a two-sided run's sells limiting someone's buys, wash
sales, internal crossing — is a recorded non-goal; `IDEAS.md` says what it would cost.

The shape this produces is measured, not assumed (`benchmarks/run_book.py`, 2026-08-30, 8 local
workers). A book of 100 accounts across 10 disjoint mandates derives 450 edges, 10 components,
critical path 10, and finishes in 6.9s where the same book as a line takes 14.9s — with byte-identical
orders and chain hashes both ways; at ~2s solves (30,000 names, 12 accounts, 4 mandates) the ratio is
14.0s against 24.6s, and at 1,000 accounts the graph stops being the constraint at all: 34s,
capacity-bound at 6.1× parallelism on 8 workers. The same harness shows where the graph degenerates:
overlap is on *any* shared tradable name, so one sector shared between neighbouring mandates
reconnects the book and the critical path is the line again — 1,450 edges instead of 4,950, critical
path still 100 — and the shipped example, 100 accounts over three securities, is deliberately the
degenerate case (its manifest records `edges 4950, critical_path 100`). Two consequences worth
stating. The win belongs to books partitioned by mandate, universe, or restriction list — and to
books whose constraints *declare* their coupling: a typed constraint row says whether it reads the
chain and, through its scope, which securities it couples through, so a portfolio with no chain
reader waits for nobody and only opaque rows keep the widest reading (generalizing the grouping
column and demonstrating the narrowing at scale are what remain open in `IDEAS.md`). And the graph's per-link cost — a contribution round-trip of ~40–120ms under
load — means it pays once solves dominate that, which a 100,000-name book's multi-second solves do
by two orders of magnitude; a book of many sub-second solves is bounded by the scheduler, not the
chain, whatever the graph looks like.

## Where the work runs is a setting, and the run owns its cluster

The graph says which portfolios wait for which; it says nothing about machines. Which
cluster the run provisions — worker processes on this machine, pods a Dask Gateway creates, or a scheduler
someone else runs — and how many workers, are settings, so the same config hashes identically on a
laptop and on a cluster and `diff-manifests` never blames the wiring for where a run happened to
execute. There is one execution path whatever the answer: the runner starts the cluster before the load
stage so its start-up hides under the slow part, waits for the first worker only after assembly, hands
it the assembled datasets once, submits tasks that carry a portfolio id and nothing else, and closes it
in a `finally`. A laptop run and a gateway run differ in two settings and exercise the same code.

![Where each stage runs](images/execution-stages.svg)

The cluster is the run's own. It is provisioned when the config resolves — a `LocalCluster` on a
laptop, a cluster a Dask Gateway creates running the run's own image — sized up after assembly, and shut
down when the run ends. That is deliberate: a shared, long-lived cluster has to be operated, pushes
fairness between runs onto the scheduler's priorities, and can only prove which code solved a portfolio
through a fingerprint check. Per-run clusters use the run's image, let the gateway's own limits and the
namespace's quota arbitrate between runs, and cost start-up latency that the load stage mostly hides.

![The run owns its cluster: provisioning overlaps the load stage](images/cluster-lifecycle.svg)

The fingerprint is kept anyway, because it is cheap and it is what makes any shared machine safe: every
task returns the environment of the process that ran it — interpreter, libraries, solver, the versions of
the packages behind external steps, git revision, image digest — and a worker whose environment differs
from the run's fails its portfolio at stage `worker` rather than answering. The workers the run starts
with are checked before it shares any data with them: each resolves the config — the solver and every
step package must be present — and reports its fingerprint, and one that cannot stops the run before
it has done any work. The manifest records the backend's lifetime and every environment that did work.

## Timing is recorded, and observability is never identity

A finished run can be drawn. Every task the engine submits times itself — `build` with its
`build:slice`/`build:rules`/`build:spec` phases, `solve` with `solve:chain`/`solve:solve`/`solve:verify`/`solve:orders` —
and the runner times the client-side stages: `load`, each `dataset:<name>` (derived from the load
audits), `assembly`, `cluster` (provisioning to first worker), `sink`. A span is a wall-clock start, a duration, and the
`host:pid` that ran it; spans ride each task's output back beside the environment stamp and land in
the manifest's `timing` block. `trace.json`, written beside the manifest in the Chrome trace format,
is what renders them: it opens in `chrome://tracing` or Perfetto with a row per worker process and
every build and solve where it actually ran.

Two rules shape it. **Observability is never identity**: spans live in the manifest but are not in the
config hash and are never compared by `diff-manifests` — two runs of one config are identical where it
matters and differ here by definition. And **it is timing, not profiling**: a `perf_counter` pair
around work the engine already delimits, a few dozen spans per portfolio, nothing to switch on. What
it answers is what the manifest alone could not: whether the workers were busy or idle, whether the
schedule's critical path is where the wall clock went, and which stage a regression landed in.

## Failure semantics

Every portfolio ends as a `PortfolioResult` or a `PortfolioFailure` naming the stage that failed, and
carrying the traceback of the exception behind it — observability, never identity, like the timing
spans: the run writes it to `failures/<portfolio_id>.txt` and `diff-manifests` compares neither it nor
that file. A failure follows the edges: the portfolios that depended on it are skipped, each naming the predecessor,
and the rest are untouched under `continue`; under `fail_fast` every lower-priority portfolio is skipped
by position, whatever it had finished, so the manifest never depends on timing. A failed build has an
unknown tradable set and is treated as overlapping everything after it. The sink is called once, only when at least one portfolio
solved, and a sink failure is an infrastructure exit code with the manifest still written. There is no
path on which the engine returns the current portfolio as a default answer: an infeasible problem raises,
with an arithmetic diagnosis of why.

## What was left out on purpose

Tax lots (holdings are security-level with an average cost), shorting, fractional shares, a covariance
risk term, and automatic solver fallback. Each is a real extension, and each would have made the
template harder to read without changing the shape of the engine. The manifest, the hashes, and the
verifier are designed so that adding them leaves the audit story intact.

Also left out, on the numbers: building the cvxpy problem somewhere other than the worker that solves
it — pickling the `Problem` back, shipping canonicalized data, a DPP split with the chain as
parameters. Every variant moves canonicalization, which is 0.6 s of a 9 s critical path at 100,000
names, and would move a canonical form three to five times the spec's size in its place; `IDEAS.md`
has the measurements.
