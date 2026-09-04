# How to add a check

A check is a business rule proven on the orders that went out: it takes every assembled dataset as the
rules first saw it, the orders the run published, and the portfolios that solved, and returns the cases
the rule applies to with a flag on each. The verifier proves the typed constraint rows on the solved weights; a check proves a
Python rule — a wash-sale window, a mandate, a signal floor — on what was actually sent. This guide
adds a check with parameters, wires it into a run config, and tests it.

## Prerequisites

- The environment is installed (`uv sync --locked`) and you can run the example.
- You know which rule the check proves and what a *case* of it is: the check reports one row per
  case, so a book with no case is reported as not having exercised the rule rather than as passing it.

## 1. Write the function in `src/portfolio_optimizer/checks.py` — or in your package

A check shared across desks belongs in an installed package and is named `my_firm.checks:no_buys_below_floor`
in the config, or published as an entry point in the group `portfolio_optimizer.check` and named bare;
everything below applies unchanged.

```python
class NoBuysBelowFloorParams(Params):
    column: str = Field(default="alpha", min_length=1)
    floor: Decimal = Field(default=Decimal(0))


def no_buys_below_floor(frames: Frames, orders: pd.DataFrame, solved: pd.DataFrame, params: NoBuysBelowFloorParams) -> pd.DataFrame:
    """No BUY in a name whose ``column`` is below ``floor``: no new money goes into a name research dislikes.

    Examined: every (solved portfolio, disliked name) pair; ``ok`` where no BUY was found there. A
    universe with no name below the floor is ``not_exercised``.
    """
    universe = frames["universe"]
    if params.column not in universe.columns:
        msg = f"no_buys_below_floor needs the universe's {params.column!r} column, and this universe has none"
        raise ValueError(msg)
    disliked = universe.loc[universe[params.column] < float(params.floor), ["security_id"]]
    forbidden = solved[["portfolio_id"]].merge(disliked, how="cross")
    buys = orders.loc[orders["side"] == "BUY", ["portfolio_id", "security_id", "quantity"]]
    found = forbidden.merge(buys, on=["portfolio_id", "security_id"], how="left", validate="one_to_one")
    return found.assign(ok=found["quantity"].isna().astype("bool"))
```

The signature is the whole contract:

- `frames: Frames` — required, exactly this name and type: every assembled dataset by name, *before
  any rule ran*. The rules are what the check proves, so it must not read their output.
- `orders: pd.DataFrame` — required: every solved portfolio's orders as the sink received them, in the
  [orders frame](reference-manifest.md#orders-frame); every row carries `as_of_date`, so a check that
  needs the instant reads it there.
- `solved: pd.DataFrame` — required: one `portfolio_id` per portfolio that solved, whether or not it
  produced an order. This is the population a rule applies to: an account the rule kept out of a name
  traded nothing, and is exactly the case that proves it.
- `params: <Params subclass>` — optional; the engine validates the JSON `params` object against it
  before any data loads.
- Return a `DataFrame` with a `portfolio_id` column and a boolean `ok` column; every other column is
  the check's own detail and is written out for the rows that fail. One row per case the rule applies
  to. Zero rows means the book never put the rule to the test, and the manifest says so.

What "examined" means is the check's honesty. `restricted_never_traded` examines every
(solved portfolio, restricted name) pair, not every order, so a book with no restricted name is
`not_exercised` rather than `passed`; a check that examined only the orders it found would call an
untested rule proven. Read the same dataset under the same params as the rule you prove where you
can — `no_trades_inside_wash_window` shares `RecentTradesParams` with `restrict_recent_trades` — so
the two cannot drift apart. Keep checks pure: no I/O, no clock, no randomness; the manifest records the
function's source hash beside its outcome.

## 2. Name it in the run config, under a label

```json
"checks": [
  {"name": "no_buys_below_floor", "label": "no_buys_in_disliked_names", "params": {"floor": "0"}},
  {"name": "no_trades_inside_wash_window", "label": "wash_sale_window", "params": {"window_days": 30}}
]
```

A check is always the object form, because it needs a `label`: the manifest's `checks[]` records the
outcome under it, the rows that failed go to `checks/<label>.csv`, and two checks of one function under
different params are told apart by it. Labels are unique across the run's checks and may use letters,
digits, `_`, `.`, and `-`. Checks run in list order, once, on the client, after the sink — only when at
least one portfolio solved, since there is nothing to prove otherwise — and every check runs whatever
the others found.

## 3. Check it resolves

```bash
uv run portfolio-optimizer validate-config configs/my_run.json
```

A typo in the name, a missing `label`, a parameter the model does not declare, or a wrong annotation
is reported here with the function's qualified name. `uv run portfolio-optimizer steps` lists every
check a bare name can resolve to, with its parameters.

## 4. Read the outcome

The run prints one line per check after the portfolios — `check wash_sale_window: passed, 3 examined,
0 violation(s)` — and its exit code is 1 when any check `failed`. The manifest's `checks[]` block
carries the same, with the function's source and params hashes; `checks/<label>.csv` beside it holds
every examined row whose `ok` was false, with the check's detail columns. `diff-manifests` names a
check whose status changed between two runs — [how to QA a deployment](how-to-qa-a-deployment.md).

A check that raises, or returns something other than examined rows, is the run's own failure at stage
`check` — a bug in the check, not a verdict on the orders — recorded on the `*` record with its
traceback in `failures/check.txt`, and the run exits 1. The orders were published before the check ran:
a check is a proof, not a gate.

## 5. Test it

Add a table-driven test to `tests/test_checks.py`, building the datasets with the frame builders in
`tests/conftest.py` and the orders with `frames.orders(...)`: one case at the boundary of the rule,
one just past it, a book with no case (the result is empty), and a portfolio the rule applies to that
produced no order (not examined). Validate the result against `CHECK_RESULTS` in the test, as the
runner will.

```bash
uv run pytest tests/test_checks.py
```

## 6. Verify the whole pipeline still passes

```bash
uv run pre-commit run --all-files
uv run pytest
```

The convention test in `tests/test_conventions.py` resolves every public function in `checks.py`; a
helper that returns a `DataFrame` must be private (`_`-prefixed), or the schema generator will list it
as a step.
