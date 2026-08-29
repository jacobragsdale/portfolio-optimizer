# Explanation: how the engine is built and why

This page explains the design decisions behind the template — what the engine promises, where those
promises are enforced, and what was deliberately left out.

## One convention instead of a framework

Quant developers who clone this repository want to write Python, not learn a plugin system. So the only
mechanism for extending the engine is: write an ordinary function in a designated module and name it in
JSON. The engine's resolver (`config/resolve.py`) does the work a framework would normally push onto the
author — it imports the function, checks its signature against the contract for its kind, validates the
JSON parameters against the function's own `Params` model, and detects optional context arguments by
name — and it does all of this before any data is loaded. A mistake surfaces as a config error with the
function's qualified name, not as a traceback halfway through a run.

The convention has a second purpose: auditability. Because every step is a named function, the manifest
can record its qualified name and the hash of its source text. A run can be traced to the exact business
logic that produced it, and two runs can be compared to find whether code, data, or the solver changed.

## Two conversions, in two places

Money enters the engine as `Decimal` and stays that way through loading, assembly, and rules; frame
schemas reject a float where a Decimal belongs. The optimizer needs float64, so `engine/build.py` is the
one place the conversion happens — after current weights and tax per dollar have been computed exactly.
Orders need exact quantities and notionals again, so `engine/orders.py` is the one place float64 becomes
shares and `Decimal`, using the exact prices carried alongside the spec rather than the float copies
inside it. Keeping both conversions in single, tested functions is what makes "the notional in the
order is exactly quantity times price" a guarantee rather than a hope.

## The spec is pure data

`ProblemSpec` holds every input the solver will see as plain numpy arrays aligned to a sorted list of
securities, plus a content hash. cvxpy objects are created only inside `solve()` and never leave it. This
buys three things: the spec can cross a process boundary in `parallel_build_sequential_solve` mode; it can
be persisted as an `.npz` file that an auditor can open without the solver stack; and the hash pins down
exactly what was optimized, so a change in any input changes the hash and shows up in `diff-manifests`.

## Verification is independent of the solver

After every solve, `engine/check.py` recomputes each constraint's violation and each objective term in
numpy and compares the total with the solver's reported objective. It does not import cvxpy — a test
enforces that — and its tolerances are a hundred times looser than the solver's, so a pass is a genuine
statement about the solution rather than a restatement of the solver's own convergence check. Custom
terms and constraints have no numpy twin unless the author adds one; they are listed as `unverified` in
the manifest rather than silently trusted.

One consequence surfaced during development: with no term charging for trading, an interior-point solver
may return a buy/sell split that nets to the right trade but is not minimal — a free "wash trade". The
weights were right; the split was not. The engine now canonicalizes the split to the minimal one after
solving, which satisfies every constraint the solver's did and cannot increase any shipped term, and the
verifier's complementarity check confirms it.

## Rounding

Whole-share rounding is the point where a mathematically feasible answer meets reality. The engine rounds
to the nearest share, then down to lot multiples, then clamps sells to what is held. Rounding toward zero
was considered — it can never breach a cap — but solver noise of about 1e-8 in weight space turns an
exact 1,250-share answer into 1,249, which is worse in practice. Instead the rounding drift is measured
against the solved weights and bounded by what one lot and one dust-filtered trade can cost; exceeding the
bound fails the portfolio.

## Execution modes and the chain

Portfolios in one run often depend on each other: the second portfolio to trade a thinly traded name
should see how much of its daily volume the first already used. The engine models this as an immutable
`SolveContext` that accumulates results in solve order and is projected, per solve, into a `ChainState`
aligned to that portfolio's securities.

The three execution modes differ only in where build and solve happen. `sequential` gives rules and
constraints the live context. `parallel_build_sequential_solve` builds every portfolio's spec in workers
(rules cannot see the context there) and solves in order in the main process (constraints can). `parallel`
runs everything in workers and therefore permits no chain-aware steps at all — the resolver rejects such a
config rather than silently dropping the dependency. Whatever the mode, results are consumed in configured
solve order, so the number of workers and the order in which they finish never affect the output.

## Failure semantics

Every portfolio ends as a `PortfolioResult` or a `PortfolioFailure` naming the stage that failed. Under
`fail_fast` the first failure stops further processing and later portfolios are recorded as `skipped`;
under `continue` the failure is isolated. Chain-aware steps forbid `continue`, because a skipped portfolio
would silently change what later solves see. The sink is called once, only when at least one portfolio
solved, and a sink failure is an infrastructure exit code with the manifest still written. There is no
path on which the engine returns the current portfolio as a default answer: an infeasible problem raises,
with an arithmetic diagnosis of why.

## What was left out on purpose

Tax lots (holdings are security-level with an average cost), shorting, fractional shares, and automatic
solver fallback. Each is a real extension, and each would have made the template harder to read without
changing the shape of the engine. The manifest, the hashes, and the verifier are designed so that adding
them leaves the audit story intact.
