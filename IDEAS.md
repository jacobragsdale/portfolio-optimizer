# Ideas

Threads for expanding the template that are not yet decisions. Each one states the problem as the
engine has it today, the options, and a leaning; none is a commitment. When a thread becomes a
decision it moves into the code and [the architecture explanation](docs/explanation-architecture.md)
and leaves this file. Numbers below are estimates for a book of *N* = 100,000 unique securities, which
a business unit can exceed.

## Building the cvxpy problem is the expensive half, and it runs in the wrong place

### Three things are called "build"

1. **The spec build** (`engine/build.py`): rules, Decimal arithmetic, alignment to the sorted universe,
   the one Decimal→float64 conversion. Pure numpy out. Runs in workers in
   `parallel_build_sequential_solve` and `parallel`.
2. **The expression tree** (`engine/solve.py` → the terms and constraints → `cvx/adapter.py`): a few
   dozen cvxpy nodes holding references to the spec's arrays. Cheap — cvxpy does no numeric work here.
3. **Canonicalization** — inside `problem.solve()`: DCP verification, the reduction chain to the solver's
   conic or QP form, coefficient extraction into sparse matrices. This is what people mean when they say
   "building the problem is slower than solving it": for a few hundred thousand variables and a few
   million nonzeros it is seconds to tens of seconds in pure Python and scipy, often several times the
   solver's own time on a well-conditioned QP.

`parallel_build_sequential_solve` parallelizes (1), which at *N* = 100k is a second or two of Decimal
loops, and runs (2) and (3) sequentially in the main process, once per portfolio, on the critical path.
The mode's name promises more than it delivers at scale. The question is where (3) should run.

### Why "build in the worker and pickle the problem back" does not work

It is the obvious fix and it fails on three counts.

**The pickle carries the wrong thing.** A `cp.Problem` pickled before `.solve()` is the expression tree:
the spec's constants plus node overhead. Canonicalization has not happened, so the main process pays
it anyway; the pickle bought nothing. A `Problem` pickled *after* a solve may carry cvxpy's
canonicalization cache, but that cache is a private attribute, not part of the pickle contract, and
nothing promises it survives a round trip or a version bump. Designing around it means designing around
an accident.

**Size.** What is actually in a 100 MB problem pickle at *N* = 100k:

| Object | Size | Note |
|---|---|---|
| One float64 vector | 0.8 MB | |
| `ProblemSpec`, shipped fields | ~10 MB | ten vectors, ids, flags; already what workers return today |
| Dense sector matrix, 11 sectors | 9 MB | `build_problem_spec` builds it dense, by Python comprehension |
| Dense sector matrix, ~160 sub-industries | 128 MB | same code; *this alone is a 100 MB pickle* |
| Same matrix, sparse | ~1 MB | one nonzero per security |
| Factor loadings, 50 factors, dense | 40 MB | genuinely dense data; unavoidable if the term exists |
| Canonical solver data, shipped terms only | ~20–25 MB | ~16 nonzeros per security in *A*, CSC |
| Canonical solver data with a 50-factor risk term | ~80–90 MB | the factor block expands into *A* or *P* |
| `ChainState` projected onto the spec | 0.8 MB | one float per security |
| Cumulative shares in `SolveContext` | ≤ 0.8 MB | one float per security ever traded |

The spec is the minimal representation of the problem; the expression tree is the spec plus overhead;
the canonical form is a 3–5× expansion of it. Every representation that skips canonicalization on the
main side is bigger than the one we already ship.

**Bandwidth on one process.** Even if the pickle carried the cache, every portfolio's result funnels
into the client. At 100 MB and 1 Gb/s that is ~0.8 s each, faster than re-canonicalizing — but 1,000
portfolios is 100 GB of ingress on one NIC, 13 minutes of serialized transfer on the critical path, and
`window = 2 × max_workers` of them held in worker memory waiting to be gathered in order. Canonicalization
would be replaced by a transfer bottleneck, not removed.

What is *not* an objection: pickle fragility across environments. `_accept` in `engine/runner.py`
already rejects any result from a process whose fingerprint (interpreter, cvxpy, solver, image digest)
differs from the run's, so a pickled cvxpy object would only ever cross between identical images.

### Move the chain to the problem, not the problem to the chain

The only reason the solve happens in the main process is the chain: a constraint may read what earlier
portfolios ordered (`cumulative_adv_participation`), and that is known only in solve order. The chain
is tiny — one float per security — and the problem is huge. So send the chain to the worker that built
the problem, and let the problem never leave it.

Per portfolio, two tasks:

- **build** — slice, rules, spec, expression tree, canonicalization; runs as soon as the shared data
  is on a worker, fully parallel. Every chain-dependent input is a cvxpy `Parameter` so the
  canonicalization is valid for any chain value. The task returns a handle to the built problem held
  in worker memory, plus the small things the main process wants now (spec hash, rule audit).
- **solve** — takes the build handle and the *previous portfolio's solve output*, sets the parameters
  from the chain, solves, verifies, rounds, and returns a `PortfolioResult` (a few MB: three vectors and
  the orders) together with the updated cumulative shares for the next solve.

Dask expresses this directly: `solve_i = submit(solve_task, build_i, solve_{i-1})`, and the scheduler
places `solve_i` on the worker that holds `build_i` because it weighs dependency sizes when placing
tasks. The critical path becomes Σ(parameter apply + solve + verify + orders) plus a chain hop of under
a megabyte per portfolio; canonicalization overlaps with the solves ahead of it in the order. The
runner's contract does not change — results are still consumed in configured order, `fail_fast` still
cancels everything queued behind the first failure, `continue` is still refused for chain-aware steps —
and the config does not change either: this is what `parallel_build_sequential_solve` should have meant.

What it changes in the engine:

- The adapter gains a `parameter(n)` atom. A chain-aware constraint splits into the shape the verifier
  already uses: a *structure* `(x, spec, p) -> ConstraintSet` that references the parameter, and a
  *value* `(spec, chain) -> F64` that computes it in numpy once the chain arrives. `adv_remaining` is
  already the value half of `cumulative_adv_participation`. DPP forbids `maximum(0, capacity − consumed)`
  as an expression in the parameter, which is why the value must be computed outside cvxpy and handed
  in as data — the same reason the verifier has to be able to compute it.
- The worker-side chain is the cumulative totals only. `SolveContext.results` retains every prior
  `PortfolioResult` including its spec, which `avoid_cross_portfolio_wash_sales` reads; rules cannot see
  the context in this mode anyway, so nothing on the worker needs it. (Sequential mode keeps every spec
  of the run in the main process — 1,000 portfolios × 10 MB — which is worth revisiting on its own; the
  wash-sale rule needs orders, not specs.)
- `engine/solve.py` separates "construct and canonicalize" from "set parameters and solve", and
  `engine/tasks.py` gets the two entry points.

DPP is the load-bearing assumption. cvxpy caches the canonicalization the first time a parametrized
problem is solved and applies later parameter values as a sparse product. For a parameter that appears
only as a constraint right-hand side the parameter tensor has ~*N* nonzeros and the re-apply is
negligible. If a parameter ever multiplies a variable — a chain-dependent cost vector in the objective,
say — the tensor grows with the product and cvxpy's own advice is `ignore_dpp=True`, which puts
canonicalization back on the critical path. Measure the first version with the shipped ADV constraint
before generalizing.

Memory is bounded: each worker holds `window / workers` = 2 built problems.

### The chain as a DAG, not a line

Two portfolios that trade no name in common do not depend on each other. The solve order can be a
dependency graph built from universe overlap in configured order — portfolio *j* depends on every
earlier *i* whose universe intersects its own — and Dask will run independent solves concurrently. The
answer is identical to the sequential one, because *j* still sees every earlier *i* that could have
consumed its ADV budget; only the wall clock changes. Overlap on universes is conservative (overlap on
*traded* names is what matters, and that is unknown until solved), and on a book where most portfolios
share one large universe the graph degenerates to a line. Worth it for desks with disjoint universes,
and essentially free once the chain is a task dependency rather than a loop in the runner.

### Later, if canonicalization still dominates: assemble the solver data directly

The verifier already has a numpy twin of every shipped term and constraint. A direct assembler that
produces the QP or conic data as scipy sparse matrices from a spec — for the shipped structure only —
would be one to two orders of magnitude faster than cvxpy's general reduction chain, and the chain's
right-hand-side rows are then a plain index write rather than a parameter. cvxpy stays for custom terms
and as the cross-check on small problems (assemble both ways, solve both, compare). It is a large
scope with a real cost in readability — terms would no longer be "just write the cvxpy atoms" — so it
is only justified once the two changes above are in and a profile still shows canonicalization on top.

### Considered and rejected: ship canonical data, unpack in the main process

`problem.get_problem_data(solver)` in the worker, `chain.solve_via_data(...)` and
`problem.unpack_results(...)` in the main process. Unpacking needs the same `Problem` object on both
sides — `inverse_data` refers to variable ids, which are a per-process counter — so the main process
would have to rebuild the tree and hope the ids match. And the canonical data is the largest
representation in the table. It answers the wrong half of the question.

### Cheap things to do first, whatever else happens

- **Build the sector matrix in numpy and carry it sparse.** `build_problem_spec` builds a dense
  *K* × *N* matrix by a nested Python comprehension: at 160 sub-industries and 100k names that is 16
  million Python iterations and 128 MB per portfolio — plausibly most of any "100 MB pickle". A
  broadcast comparison of two arrays builds it in milliseconds; a `scipy.sparse` matrix holds it in a
  megabyte; cvxpy's `matmul` and the verifier's `@` both accept it unchanged. The content hash and
  `to_npz` need a sparse encoding (`indptr`, `indices`, `data`), which is the only real work.
- **Keep the factor risk term structured when it returns.** `sum_squares(F½ · B · w) + sum_squares(√D ∘ w)`,
  never a dense *N* × *N* covariance (80 GB at 100k). The 50 × *N* loadings are the one genuinely dense
  block and they set the floor on every size above.
- **Profile before deciding.** One portfolio at *N* = 100k with the shipped terms, timed as spec build /
  expression tree / `get_problem_data` / `solve_via_data` / verify / orders. Everything in this section
  is reasoning from the code; the split between canonicalization and solve is solver- and
  structure-dependent, and Clarabel on a QP with a factor term may not look like OSQP on one without.

## Other threads

- **Tax lots.** Holdings are security-level with an average cost. Lot-level tax needs one sell variable
  per lot (*N* + *L* variables, lot-to-security aggregation as one more sparse matrix), changes the
  holdings schema and `orders.py`, and is the first extension that makes the build-placement question
  above unavoidable.
- **Warm starts.** OSQP and SCS benefit from starting at `w0` or at yesterday's solution; Clarabel, an
  interior-point method, does not. The solutions of every run are already persisted as `.npz`; a loader
  could hand the previous run's in as a `warm_start` column and the adapter could pass it through.
  Pointless until the solver is the bottleneck, which the section above argues it currently is not.
- **Re-solve from the persisted spec.** `problem_specs/<portfolio>.npz` plus `chain/<portfolio>.npz` is
  everything the solver saw. A `resolve` CLI subcommand that rebuilds the problem from those files and
  compares the result with `solutions/<portfolio>.npz` turns the audit artifacts into a reproducibility
  test that needs no data sources at all.
- **Solver fallback stays out.** A silent second solver changes what the manifest says was solved. A
  visible retry — same solver, relaxed tolerances, recorded in the manifest as a second attempt — is the
  acceptable variant if one is ever needed.
