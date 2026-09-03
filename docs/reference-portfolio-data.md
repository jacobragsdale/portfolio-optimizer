# Reference: the per-portfolio bundle and the optimizer frame

`portfolio_optimizer.domain.data.PortfolioData` is the validated bundle every rule receives and every
build consumes; `portfolio_optimizer.domain.results.ProblemSpec` is what the build returns and every
term, constraint, and solve step reads. `Frames` is what assembly steps receive. This page states their
contracts; for how to use them see [how to add security analytics](how-to-add-security-analytics.md),
[how to add a rule](how-to-add-a-rule.md), and [how to add a term or constraint kind](how-to-add-a-term.md).

## `Frames`

An immutable `Mapping[str, pd.DataFrame]` of every dataset by name, handed to each assembly step and
returned by it.

| Member | Description |
|---|---|
| `frames["name"]` | The dataset. A missing name raises `KeyError` listing the names that exist. |
| `"name" in frames`, `len(frames)`, iteration, `.items()` | Standard mapping behavior. |
| `with_frame(name, frame)` | A copy in which `name` is `frame`, added or replaced. |
| `without(*names)` | A copy without `names`; every name must exist. |
| `row_counts()` | `{name: rows}`, as recorded in the manifest. |

After the last step, `holdings`, `universe`, and `details` must exist and satisfy their
schemas. Every other dataset still present becomes an extra.

## `PortfolioData`

Constructed by `slice_portfolio` and by `with_changes`; every construction runs every check below and
raises `PortfolioDataError` listing all failures.

| Field | Type | Description |
|---|---|---|
| `details` | `PortfolioDetails` | This portfolio's `details` row. Required, because the engine reads them: `portfolio_id`, `nav`, `max_weight` (the single-name cap the build folds into the box), `min_trade_notional` (the order step's dust threshold). Optional: `name`, `state`, `st_tax_rate` and `lt_tax_rate` (together they give the build `tax_per_dollar`), `max_adv_participation` (with the universe's `adv_shares`, `adv_capacity`), `cash`, `max_turnover`, `cash_lb`, `cash_ub` (scalars a constraint row names), and `extra` — every further column of the row, as `Decimal`, `int`, `str`, `bool`, or `None`. `details.scalars()` is every number the row carries, declared or extra, which the build exports as the spec's scalars; a term or row that names one the row lacks is refused by name at build. |
| `holdings` | frame | This portfolio's rows of `holdings`. Schema columns `portfolio_id`, `security_id`, `quantity`, `avg_cost`, `acquired_on`; any further columns allowed. |
| `universe` | frame | The whole `universe`, identical for every portfolio. Required columns `security_id`, `price`; optional `sector`, `adv_shares`, `adv_consumed_shares` (what an earlier run already traded of the day's volume), `lot_size` (default one share), `restricted` (default false), `alpha`, `tcost_bps`, `min_weight`, `max_weight`; any further columns allowed. |
| `constraints` | frame | This portfolio's rows of `constraints`: one typed row each — `portfolio_id`, `kind`, `label`, `params` — or a desk's own shape. The engine validates that `portfolio_id` is present and names this portfolio, and reads `kind` for the declaration it schedules by; the rest reaches the solve step. Empty when the run declares no such dataset. A rule may replace it. |
| `as_of_date` | `datetime` | The run's `--as-of`, timezone-aware UTC. |
| `extras` | `Mapping[str, frame]` | Every non-engine dataset that remained after assembly. A dataset with a `portfolio_id` column is reduced to this portfolio's rows; one without is passed whole. Carried past the build to the solve step as `request.extras`, which is where runtime parameters reach a solver. Default `{}`. |
| `applied_rules` | `tuple[str, ...]` | Qualified names of the rules applied so far; maintained by the pipeline. |
| `portfolio_id` | property | `details.portfolio_id`. |

### Checks on construction

1. `holdings`, `universe`, and `constraints` satisfy their frame schemas (columns, dtypes,
   nullability, bounds, unique key). `constraints` declares only `portfolio_id`, so only that is
   checked here; its rows are parsed as their kinds when the portfolio builds.
2. `extras` values are frames, and no extra is named `holdings`, `universe`, `details`,
   `constraints`, or `portfolios`.
3. `as_of_date` is timezone-aware UTC.
4. `holdings.portfolio_id` contains only this portfolio.
5. `constraints.portfolio_id` contains only this portfolio.
6. Every column present in both `holdings` and `universe` has the same dtype in both.
7. Every extra with a `portfolio_id` column contains only this portfolio.

A held security need not be in the universe. The shipped build requires it (a `BuildError` at stage
`build`); a custom build consuming the optimizer frame need not.

### Methods

| Method | Description |
|---|---|
| `with_changes(*, details=None, holdings=None, universe=None, constraints=None, extras=None)` | A re-validated copy with the given parts replaced. |
| `with_rule_applied(qualname)` | A copy recording that a rule ran; called by the pipeline. |
| `optimizer_frame(*, source_column="source")` | Holdings and universe stacked into one frame; see below. |

## The optimizer frame

`PortfolioData.optimizer_frame()` returns one `pd.DataFrame` built by
`portfolio_optimizer.domain.optimizer_frame.stack_frames({"holdings": ..., "universe": ...})`.

| Aspect | Contract |
|---|---|
| Rows | Every `holdings` row, then every `universe` row, in their original orders; fresh integer index. A security in both appears twice. |
| `source` column | First column, dtype `string`, value `"holdings"` or `"universe"`. Renamed by `source_column`; omitted when `source_column=None`. Must not collide with an existing column. |
| Columns | The union of both tables' columns; `holdings` columns first in their order, then `universe` columns not already seen. |
| Shared columns | Keep their common dtype. A dtype mismatch is refused (`FrameStackError`; already refused on bundle construction). |
| One-sided columns | Null on the side that lacks them. A dtype that cannot hold a null is promoted in the whole column: `bool` → `boolean`, `int64` → `Int64`, `float64` → `Float64`. `object` (Decimal) columns are filled with `None`; `string`, `Int64`, `Float64`, `boolean`, and `datetime64[ns, UTC]` with `pd.NA`/`NaT`. Any other dtype that rejects a null (`int32`, `uint8`, …) is refused. |
| Empty input | A table with zero rows contributes no rows and still contributes its columns. |

`stack_frames` accepts any mapping of named frames and is what the shipped `union` assembly step uses.

## `ProblemSpec`

What the build step returns — `standard` by default, or any `(data: PortfolioData[, params]) ->
ProblemSpec` named as `build` in the config — and what every term, constraint, solve step, and the
verifier read: pure numpy, read-only arrays, aligned to the sorted `security_ids`, every vector a
fraction of NAV, no cvxpy. It is persisted as `problem_specs/<portfolio_id>.npz` and its content hash
goes into the manifest.

| Field | Type | Description |
|---|---|---|
| `portfolio_id`, `as_of_date`, `nav` | | Identity; `nav` as float64. |
| `security_ids` | `tuple[str, ...]` | Sorted and unique; every array is aligned to it. |
| `w0`, `price`, `shares_held`, `lot_size`, `lb`, `ub` | `float64[n]` | The fixed vectors every spec carries: starting weights, price, shares held, lot size, and the per-security box the build derived. |
| `columns` | `Mapping[str, float64[n]]` | Named per-security numbers: the derived `tax_per_dollar`, `tcost_per_dollar` (where the universe carries `tcost_bps`), `adv_capacity` (where it carries `adv_shares`), and every exported numeric universe column, `alpha` included. |
| `flags` | `Mapping[str, bool[n]]` | Named per-security masks: every boolean universe column, `restricted` included. What a constraint's `scope` names. |
| `groups` | `Mapping[str, Grouping]` | Every string universe column as a sparse membership matrix: `names`, the sorted distinct values, and `matrix`, *K*-by-*n* CSR with one nonzero per security. What `group_limit` reads. |
| `scalars` | `Mapping[str, float]` | Every number the account's `details` row carries, declared or extra: always `nav`, `max_weight`, and `min_trade_notional`; `cash_lb`, `cash_ub`, `max_turnover`, `max_adv_participation`, `cash`, and the tax rates where the row has them; and any numeric extra column. What a `{"scalar": ...}` bound names. |
| `buyable`, `sellable` | properties, `bool[n]` | `ub > w0`; held and `lb < w0`. The sets a order-flow profile couples the portfolio through. |

Accessors — `spec.column(name)` (a fixed vector or an exported column; `spec.column_names` lists both),
`spec.flag(name)`, `spec.scalar(name)`, `spec.group(column)` — raise `MissingSpecColumnError` naming
what is available, which is how a term or constraint refuses, at build, a spec that lacks what it
reads. Construction checks shapes, sortedness, finiteness, `lb ≤ ub`, positive `nav` and `price`,
`lot_size ≥ 1`, that no name is both a column and a flag, and that no column shadows a fixed vector.

## What the shipped build exports

`engine/build.py`'s `standard` aligns every array to the sorted `universe`, computes the starting
weights, the tax per dollar sold, and the bounds exactly in `Decimal` before the one conversion to
float64, and exports each universe column other than `security_id` by name: **numeric** columns beyond
the schema (`Int64`, `Float64`, `float64`, `int64`, Decimal-valued `object`), plus `alpha`, into
`columns`; **boolean** columns (`bool`, `boolean`) into `flags`; **string** columns into `groups`. The
schema's other numeric columns (`price`, `adv_shares`, `adv_consumed_shares`, `lot_size`, `tcost_bps`,
`min_weight`, `max_weight`) are folded into the fixed vectors and the derived columns and not exported
again (`adv_capacity` is the participation times `adv_shares` less `adv_consumed_shares`, floored at zero). Every
number on the `details` row becomes a scalar. A null in an exported column, flag, or grouping is a
`BuildError`, and a name cannot be both a column and a flag. Columns on `holdings` beyond its schema
are never exported: the build has no row for a name that is not in the universe.

`engine.build.order_inputs(data, spec)` derives the exact `Decimal` prices, share counts, and bounds the
order step needs from any spec aligned to the universe's securities, so a custom build never has to
reconstruct money from float64.
