# Reference: the per-portfolio bundle and the optimizer frame

`portfolio_optimizer.domain.data.PortfolioData` is the validated bundle every rule receives and every
build consumes. `Frames` is what assembly steps receive. This page states their contracts; for how to
use them see [how to add security analytics](how-to-add-security-analytics.md) and
[how to add a rule](how-to-add-a-rule.md).

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
| `details` | `PortfolioDetails` | This portfolio's `details` row: `portfolio_id`, `name`, `state`, `st_tax_rate`, `lt_tax_rate`, `cash`, `nav`, and the style limits `max_weight`, `max_turnover`, `max_adv_participation`, `min_trade_notional`, `cash_lb`, `cash_ub`. |
| `holdings` | frame | This portfolio's rows of `holdings`. Schema columns `portfolio_id`, `security_id`, `quantity`, `avg_cost`, `acquired_on`; any further columns allowed. |
| `universe` | frame | The whole `universe`, identical for every portfolio. Schema columns `security_id`, `price`, `sector`, `adv_shares`, `lot_size`, `restricted`; optional `alpha`, `tcost_bps`, `min_weight`, `max_weight`; any further columns allowed. |
| `constraints` | frame | This portfolio's rows of `constraints`, in the desk's own shape. The engine validates only that `portfolio_id` is present and names this portfolio; every other column is carried untouched to the solve step. Empty when the run declares no such dataset. A rule may replace it. |
| `as_of_date` | `datetime` | The run's `as_of_date`, timezone-aware UTC. |
| `extras` | `Mapping[str, frame]` | Every non-engine dataset that remained after assembly. A dataset with a `portfolio_id` column is reduced to this portfolio's rows; one without is passed whole. Default `{}`. |
| `applied_rules` | `tuple[str, ...]` | Qualified names of the rules applied so far; maintained by the pipeline. |
| `portfolio_id` | property | `details.portfolio_id`. |

### Checks on construction

1. `holdings`, `universe`, and `constraints` satisfy their frame schemas (columns, dtypes,
   nullability, bounds, unique key). `constraints` declares only `portfolio_id`, so in practice only
   that is checked.
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

## What the shipped build exports

`engine/build.py` aligns every array to the sorted `universe` and exports each universe column the
schema does not declare (plus `alpha`) by name: **numeric** columns (`Int64`, `Float64`, `float64`,
`int64`, Decimal-valued `object`) into `ProblemSpec.columns` as float64, read with `spec.column(name)`;
**boolean** columns (`bool`, `boolean`) into `ProblemSpec.flags` as real `np.bool_` masks, read with
`spec.flag(name)`. String columns are not exported. A null in an exported column or flag is a
`BuildError`, and a name cannot be both a column and a flag. Columns on `holdings` beyond its schema
are never exported.
