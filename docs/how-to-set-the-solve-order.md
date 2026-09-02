# How to set the solve order

Solve order decides who gets first pick when two portfolios compete for the same trades on the side the
run couples through — the same buys under `order_flow: inflow`, the same sells under `outflow`. It is a
*priority*, not a sequence: a portfolio waits only for higher-priority portfolios that can trade a
security it can trade too, on that side, and everything else solves concurrently. This guide sets it from the data
with a solve-order step; the alternative is the `solve_order` column of the portfolios frame.

## 1. Decide what "first" means

The shipped step puts the account with the most left to invest first, on the theory that whoever has
the most to buy should get first pick of the scarce liquidity:

```python
def most_uninvested_first(data: PortfolioData) -> Decimal:
    """Minus the fraction of NAV the portfolio has yet to invest, so the account with the most to put to work solves first."""
```

Lower keys solve first, so a "priority" that should go first returns a *smaller* number — the shipped
step returns the invested fraction minus one. Equal keys tie, and ties break on `portfolio_id`, so the order is
deterministic whatever the data.

## 2. Write the function in `solve_order.py`

```python
from decimal import Decimal

from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.types import Params


class CashFirstParams(Params):
    weight: Decimal = Decimal(1)


def most_cash_first(data: PortfolioData, params: CashFirstParams) -> Decimal:
    """Deploy the largest cash balances first."""
    return -data.details.cash / data.details.nav * params.weight
```

The contract is `(data: PortfolioData[, params]) -> Decimal`; the value must be finite. The step sees
the portfolio's bundle *after* the rules, in the worker that built it, and never sees another
portfolio. Keep it pure and exact — `Decimal`, not `float` — because the key is sorted and recorded in
the manifest.

There is a cost to configuring a step at all. The key then comes out of the build, so on a coupled run
the engine cannot tell who precedes whom until *every* portfolio has built, and no solve starts before
the last build finishes. The portfolios frame's `solve_order` column is known before anything is built,
so a run that keys off it starts solving the head of the book while the tail is still building. Prefer
the column when the priority can be expressed there, and reach for a step when it genuinely needs the
ruled bundle.

## 3. Name it in the run config

```json
"solve_order": {"name": "most_cash_first", "params": {"weight": "1"}}
```

A bare string names a step without params; a qualified `package.module:function` names one outside
this module. When a step is configured the portfolios frame's `solve_order` column is ignored; without
one the column is the key, and without either every portfolio ties and solves in `portfolio_id` order.
The step is part of the config hash, so two runs with different priorities are visibly different runs.

## 4. Check it resolves, then read the schedule back

```bash
uv run portfolio-optimizer validate-config configs/my_run.json
uv run portfolio-optimizer run configs/my_run.json --as-of 2026-08-28T00:00:00Z
```

Each portfolio's record in the manifest carries `solve_order` (the key as a string) and
`predecessors` (how many higher-priority portfolios it waited for); the run-level `schedule` record
says how many independent components the book split into and how long the longest chain of solves
was. If the critical path is close to the number of portfolios, the priority is not the lever —
the tradable sets are: see [how to add a rule](how-to-add-a-rule.md), step 4.

## 5. Test it

```python
def test_most_cash_first_puts_the_largest_cash_balance_first(make: Factories) -> None:
    rich = make.portfolio_data(details=make.details(cash=Decimal(500_000)))
    poor = make.portfolio_data(details=make.details(cash=Decimal(0)))
    assert most_cash_first(rich, CashFirstParams()) < most_cash_first(poor, CashFirstParams())
```

Assert on the ordering and on exactness (`Decimal`, `==`), not on how the number was computed.
