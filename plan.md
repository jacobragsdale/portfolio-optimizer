# Focused fixed-income optimization framework

Status: proposed implementation plan; no product changes implemented by this document.

This plan consolidates the engineering, adoption, and simplicity reviews. It supersedes their broader feature suggestions where they conflict with this scope. Parallel multiprocessing is retained as an explicit requirement. Implementation should use coherent changes with no compatibility shims for the current first-draft interfaces.

## 1. Product contract

Build a transparent, reproducible optimization framework for taxable and municipal bond portfolios and their cash balances. Researchers own investment objectives, constraints, analytics, tax assumptions, and allocation priorities. The framework owns validated inputs, explicit execution, numerical verification, executable quantity construction, and evidence.

The primary users are senior quant researchers, with a short path for less experienced programmers and portfolio managers to run and modify an existing strategy. The framework must make the final effective problem understandable before execution and the resulting proposal explainable afterward.

Every core feature must help define the bond problem, execute it, or verify and explain its output.

### Retain

- Strict boundary validation, exact monetary inputs, explicit units, typed Python interfaces, sparse numerical representations, and reproducible artifacts.
- Parallel execution through a bounded local process pool, with an inline mode for debugging.
- Researcher-defined objectives and constraints under one problem and verification contract.
- Independent portfolios and explicitly ordered portfolios sharing execution capacity.
- Buy-only, sell-only, and two-sided trade permissions.
- One CLI, a notebook-friendly API, a small read-only result viewer, and a few runnable bond recipes.
- Ordinary versioned Python packages for shared preparation functions, analytics, constraints, and external publishers.

### Remove from the engine

- Equity and multi-asset positioning, examples, and extension plans.
- Dask, Gateway, external scheduler connections, cluster provisioning, autoscaling, and distributed worker lifecycle management.
- Inferred overlap graphs, chain-reader declarations, speculative solve submission, and multiple dependency scheduling modes.
- The general-purpose dataset DAG, per-dataset async scheduling and batching, and configurable assembly language.
- Opaque solve steps with their own constraint vocabulary or optional verification coverage.
- Framework-owned investment/tax policies, implicit analytics exports, and scattered bound-loosening mechanisms.
- Live trading-system submission and selective retry orchestration inside optimization runs.

### Defer

Marketplace services, visual strategy builders, graphical data mapping, built-in backtesting, pricing and tax engines, cloud deployment management, automated promotion platforms, joint book optimization, and solver-template caching. Reconsider only against a concrete user need and measured cost. These are not prerequisites for this plan.

## 2. One lifecycle and a few explicit boundaries

The public lifecycle is:

**Prepare a snapshot → construct the effective problem → solve → construct executable quantities → verify → save a proposal.**

Preparation is a separate Python boundary. It may call shared loaders or external analytics libraries, but the numerical engine receives a completed, validated snapshot and performs no data fetching.

Model construction is researcher-owned Python that produces one typed problem contract. It declares the selected analytics, objective terms, constraints, effective bounds, trade permissions, and any required shared-capacity inputs. It performs no publication and cannot omit required verification.

Execution is framework-owned. The same functions run inline, in local workers, through the CLI, and from a notebook. A solver adapter executes the declared problem; it cannot reinterpret constraint records into a different model.

Publication is a separate consumer of an eligible proposal. An external publisher owns credentials, acknowledgements, submission idempotency, and retries. The engine supports writing local artifacts; it does not submit trades.

Use one canonical serialized strategy definition for parameters and component references. Avoid configuration inheritance and parallel JSON/Python definitions with different meanings. The typed Python API and CLI resolve to the same definition and explicit research package.

## 3. Bond data and model contracts

### Snapshot

Define a compact validated snapshot containing accounts, positions or tax lots, bond reference data, explicitly selected analytics, and provenance. Include cash and external cash movements explicitly. Distinguish security tax classification from account tax treatment.

Specify face quantity, quotation convention, valuation price, accrued interest treatment, valuation/settlement dates, currency, minimum denomination, and quantity increment wherever the supported calculation needs them. Accept external analytics with declared units and timestamps. Do not implement instrument pricing or infer tax treatment from a municipal label.

The initial valuation contract should use a declared account currency and reject unsupported currency mismatches rather than infer FX conversions. Cash is part of the portfolio accounting, not an opening for a general multi-asset model.

Validate identifiers, uniqueness, joins, nullability, finite values, provenance, and accounting reconciliation at the preparation boundary. Missing required data fails with affected accounts and securities. Reconcile holdings plus cash and declared adjustments to NAV using an explicit valuation policy and tolerance.

Preserve lot identity for tax-aware inputs. An aggregate-position model must explicitly declare that it does not perform lot selection. Do not silently use average cost as a replacement for a requested lot model.

### Effective problem

- Freeze the aligned security index, numeric arrays, parameter mappings, and constraint definitions after construction.
- Keep Decimal/integer values at monetary and quantity boundaries; convert explicitly to float64 for numerical solving.
- Use bond quantities rather than equity-oriented share terminology throughout public interfaces and artifacts.
- Declare analytics by name, dtype, units, and role. Remove automatic conversion of every extra column into a coefficient, flag, or grouping.
- Resolve bounds once. Record requested and effective values and the researcher-specified reason for any difference.
- Preserve sparse exposure matrices and explicit alignment checks.
- Keep one supported production solver initially: CLARABEL, the current default. Unsupported model classes fail with a clear capability error. Additional adapters require a demonstrated research requirement and the same verification contract.

Researcher freedom covers signals, exposure definitions, penalties, tax coefficients, eligibility, and priorities. Supported mathematical classes remain explicit; unrestricted arbitrary optimization algorithms are outside the initial contract.

### Extension API

Replace universal `Callable[..., object]` dispatch with typed preparation, model, term, constraint, and adapter interfaces after registration. Use one registration mechanism based on installed package entry points. Reject ambiguous names. Capture exact package versions in runs.

Keep unknown boundary data behind validation. Model solver success and failure as distinct complete states. Constraint verification definitions are owned by the effective problem and persisted independently of what the solver reports.

Avoid public validation-bypass flags. Mutable frames used during preparation must not invalidate a previously validated snapshot or leak changes between worker tasks. Choose immutable storage or controlled ownership with measured copying costs; do not add blanket deep copies.

## 4. Parallel multiprocessing without distributed orchestration

Provide an inline executor and one bounded local process-pool executor. Expose a small execution configuration: worker count, bounded in-flight work, and documented timeout/cancellation behavior. Use explicit process initialization and importable research packages so spawn-based execution works on macOS and other supported platforms.

### Independent work

Run independent portfolio solves and independent research scenarios concurrently. Each task receives an immutable input identity and produces a result independent of completion order. Collect and display results in a stable order. A portfolio failure does not prevent unrelated portfolios from completing.

### Shared execution capacity

For a coupled book, require a unique, explicit portfolio allocation order supplied by the researcher. Prepare chain-independent inputs and models in parallel. Solve in that allocation order against one ledger owned by the coordinator. Workers never mutate the ledger.

Update capacity only from the preceding account's accepted, rounded, verified proposal quantities. If a coupled account fails, stop that coupled sequence and mark the remaining accounts unprocessed; restart from an explicit input snapshot. Do not silently skip it and redistribute capacity.

Serial dependency is unavoidable when a later account's model reads an earlier account's accepted trades. Multiprocessing remains useful for preparation, independent books, and scenarios. Do not claim coupled solves are all parallel. Do not infer disjoint subgroups or resurrect an overlap graph in this implementation.

### Resource management

- Bound task submission and result buffering; do not submit the entire book eagerly.
- Index portfolio rows once during preparation instead of scanning all holdings and analytics for every portfolio.
- Initialize common read-only worker data once where practical. Measure serialization and memory before introducing shared-memory machinery.
- Persist completed results and retain compact summaries and required order/ledger data, rather than every full numerical result in coordinator memory.
- Set and record solver/numerical-library thread limits to prevent worker-count times solver-thread-count oversubscription.
- Define worker-crash, interruption, and timeout behavior explicitly. A future timeout alone does not imply that a running solver process has stopped. Clean up workers before declaring execution terminated.
- Require inline and multiprocessing agreement within declared numeric tolerances, with identical input identities, ordering, coverage, and proposal eligibility. Do not promise cross-platform bitwise numerical equality.

## 5. Correctness and explanation requirements

### Shared capacity

Replace the inconsistent prior-run and within-run consumption calculations with one formula. For a declared total capacity, remaining capacity is total capacity minus explicitly consumed quantities, floored at zero. Where capacity is derived from participation times market volume, subtract prior consumption after applying participation.

Record the capacity unit, scope, and side/netting convention explicitly. Cross-run continuation supplies cumulative prior consumption as snapshot data. There must be no hidden replay state or double subtraction. Splitting the same ordered book across runs must preserve allocation behavior when the same cumulative snapshot is supplied.

### Verification coverage

Every declared hard constraint must have a required check, whether built in or supplied by a research package. Missing or unsupported checks prevent proposal eligibility. Solver-returned records cannot narrow verification coverage.

Maintain separate results for continuous feasibility, numerical objective agreement, and executable-order feasibility. Objective agreement is not proof of global optimality, and rounded orders are not required to preserve the continuous objective value.

### Executable quantities

Use one deterministic bond-quantity construction policy honoring denominations, increments, position availability, trade permissions, and declared bounds. Rebuild exposures and cash from rounded quantities and verify every applicable hard constraint.

Keep rounding drift as a diagnostic, not an eligibility substitute. Initially reject a proposal that cannot meet its declared constraints after rounding. Do not add an automatic repair optimizer or silently relax constraints. Researchers may explicitly specify economically meaningful slack as part of their model.

Consolidate buy-only, sell-only, and two-sided handling into one lifecycle with an explicit permission field. Retain different expressions where mathematically necessary. Do not introduce simultaneous buying and selling or alter tax incentives merely to unify code paths.

### Evidence and failures

Persist one compact run bundle: snapshot references/hashes, exact component and solver versions, effective model, constraint coverage, solver status, objective contributions, continuous and executed exposures, proposed quantities, residuals, failures, and stage timings. Input references must resolve to retained snapshots; hashes alone are insufficient for replay.

Produce a proposal only after required checks pass. Diagnostic artifacts may still be saved for failed accounts. Include the expected and completed account population so consumers cannot confuse a partial run with a complete book. Coupled-book publication requires complete accepted execution; independent proposals carry explicit per-account eligibility.

Reports answer which inputs, bounds, permissions, and preceding allocations applied. They must distinguish observed binding constraints from causal explanations. Counterfactual questions require an explicitly labelled new solve with a stated change.

Keep inspect, verify, and compare operations over saved artifacts. Remove selective failure-stage retry machinery. A rerun has a new identity and explicit inputs.

## 6. Adoption and sharing within the smaller scope

Distribute the engine as a versioned package with a small public SDK. Strategy projects depend on it and contain their own research code, parameters, example data, and tests. Users should not edit engine modules to create a strategy.

Ship two small bond examples covering taxable and municipal data, without presenting their investment assumptions as recommended policy. Start the tutorial with one portfolio and a small synthetic universe. Let the user run it, modify one limit, and inspect the effect before learning multi-account execution or packaging.

The initial adoption surface is a typed Python API, one example notebook, and one CLI. Do not build a visual editor or graphical mapping workbench. Offer a small validated tabular import example and useful errors naming missing columns, units, accounts, and securities.

Keep a small shared Python component package with named maintainers, input/output contracts, compatible engine versions, example usage, and limitations. Retain standard package entry points, without a marketplace service. Shared publishers operate outside the optimization core.

Provide a minimal strategy/component scaffold and meaningful contract tests: boundary rejection, alignment, model/residual agreement, quantity conservation, and process serialization where applicable. A notebook prototype must become importable code with explicit parameters before multiprocessing or deployment.

Organize documentation around actual tasks: first run, modify a model, prepare data, share a component, inspect a failure, and consume a proposal. Keep API reference and mathematical explanations separate from the short tutorial. Remove obsolete cluster, assembly-DAG, and retry documentation as their features disappear.

Pilot with researchers and less experienced users. Measure time to first meaningful modification, first own-data run, repeat independent use, and component reuse by a second user. A 15-minute first modification is a proposed onboarding target, not a measured result. Defer automated telemetry infrastructure; observation and lightweight feedback are sufficient initially.

## 7. Implementation sequence and acceptance

All phases below are pending. Preserve unrelated existing work. Update code, tests, configuration schemas, examples, and documentation together as contracts change.

| Phase | Work | Acceptance |
|---|---|---|
| 1. Pin guarantees | Reproduce review findings and establish representative fixtures and workload measurements. | Failing regression cases cover capacity inconsistency, missing verification coverage, rounding breaches, and mutable validated state. Record current inline/local behavior without claiming production certification. |
| 2. Define the domain | Introduce the bond snapshot, effective problem, typed interfaces, explicit units, and researcher-owned policy construction. | Taxable and muni examples validate; malformed/reconciled inputs behave as declared; effective constraints are inspectable; exact boundary values and immutable state survive serialization. |
| 3. Unify execution | Implement the bounded local process pool and explicit ordered capacity ledger. Replace Dask and inferred scheduling. | Parallel independent work is exercised with real processes; coupled order is invariant to preparation timing; worker failure/cancellation is tested; prior-run and same-run capacity are equivalent. |
| 4. Produce verified proposals | Consolidate trade permissions and quantity conversion; verify executed constraints; separate publication; simplify artifacts and reruns. | Omitted checks and invalid rounded proposals cannot be eligible; failed runs retain useful diagnostics; coupled failures stop the sequence; publisher-side retries are outside engine scope. |
| 5. Delete superseded machinery | Remove old loaders/assembly orchestration, permissive custom solve paths, policy defaults, aliases, old backends/settings, and retry branches. | One supported path per responsibility; no compatibility shims; dependencies, exports, schemas, tests, and docs contain no live references to removed features. |
| 6. Package and teach | Deliver the stable SDK, project/example scaffold, short notebook/tutorial, small shared library, and updated read-only viewer. | A fresh environment runs both examples; a new user changes a limit without editing engine code; a second project installs and uses a shared component; CLI/notebook results agree. |
| 7. Measure and finish | Benchmark representative books and perform the focused usability pilot. | Record throughput, latency distributions, peak worker/coordinator memory, and serialization costs; compare inline and multiprocessing; document limitations and remove remaining unnecessary interfaces. |

Deletion can occur during earlier phases once replacement acceptance checks pass; phase 5 is the final removal audit, not a reason to maintain parallel legacy paths.

### Required regression cases

- ADV 1,000, participation 10%, prior consumption 50 leaves 50 units, identically inside one run and across an explicit continuation snapshot.
- A continuous fully invested answer that rounds to residual cash fails a required full-investment bound unless the model explicitly permits that slack.
- A solver omitting a required constraint cannot produce an eligible proposal.
- Mutating preparation inputs cannot alter an already validated effective problem or another task's inputs.
- Denomination, increment, sell availability, and buy/sell permissions hold for both taxable and municipal examples.
- Required tax-lot identity is preserved; aggregated input cannot silently claim lot-level tax support.
- Coupled ledger consumption uses accepted executable quantities and is independent of worker completion order.
- Real multiprocessing covers package imports, serialization, bounded work, worker crashes, interruption, and cleanup.
- Independent-account partial results and coupled-book incomplete results are reported and gated distinctly.
- Saved proposals can be verified without invoking the solver or publisher.

### Verification policy

Run the repository's Python tests and hooks for implementation changes. Regenerate the published schema when configuration models change. Update the README example equivalence test and fixture data with the new examples.

Use seeded benchmark workloads that vary accounts, securities, holdings/lots, constraint count, shared-capacity use, and worker count. Report sample counts and meaningful percentiles; do not claim P99.9 from a tiny sample. The earlier single-run profile is a baseline observation, not a performance target or capacity guarantee.

Do not optimize CVXPY caching, shared memory, or specialized representations until these measurements identify a material bottleneck. Preserve bounded parallel multiprocessing regardless; optimize its data movement before adding execution modes.

## 8. Completion criteria

The work is complete when the taxable and muni examples run through one transparent lifecycle, independent work runs in parallel with bounded resources, shared capacity follows a documented explicit order, every eligible proposal passes executed-order verification, and a researcher can package a model without modifying the engine.

The result should have fewer extension contracts, execution modes, implicit transformations, settings, and recovery branches than the current system. Reduced complexity must be demonstrated by deleted responsibilities and a shorter path to explaining a run, rather than by moving old machinery behind a new facade.
