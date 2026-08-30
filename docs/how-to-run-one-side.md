# How to run one side: a buy-only or sell-only run

A desk that decides sells in a separate process wants the optimizer to do one side at a time. This
guide turns a working two-sided config into a buy-only or sell-only run: it sets `sides`, removes what
the missing side takes with it, sets the cash bounds for the direction cash can move, and shows what the
run refuses and why. A one-sided run has one variable per name instead of three, so it is also the
faster problem — Clarabel is 2–3.5× quicker at 100,000 names (`IDEAS.md`).

## Prerequisites

- A run config that works with `sides: both` ([the tutorial](tutorial-first-run.md) gets you one).
- You know which side this run trades and where the cash it raises or spends goes: a buy-only run can
  only lower cash, a sell-only run can only raise it.

## 1. Set `sides`

```json
"sides": "buy"
```

or `"sell"`. The default is `both`. The value selects the side profile that every side-dependent
decision goes through, and it fixes which side portfolios couple through: buys under `both` and
`buy`, sells under `sell`. Under `buy` the solve has `w` alone with `w ≥ w0` and `buy = w − w0`;
under `sell`, `w ≤ w0` and `sell = w0 − w`. The other side does not exist.

## 2. Remove what reads the missing side

A term or constraint that reads `x.sell` cannot run in a buy-only run, and `validate-config` says so.
The shipped `tax_cost` reads `sell`, so a buy-only config that still lists it is rejected:

```bash
uv run portfolio-optimizer validate-config configs/my_buy_run.json
```

```text
config rejected: 1 config resolution failure(s): objective.terms[1]: portfolio_optimizer.terms:tax_cost: construction failed: SideUnavailableError: a 'buy' run has no 'sell' vector; this term or constraint reads x.sell, so it cannot run under sides='buy'
```

Drop it from `objective.terms`. A custom step that reads `x.buy` in a sell-only run is refused with
the mirror message. Every other shipped term and constraint is written against `x.trade` (the amount
traded on the sides the run has) or `x.coupled` (the amount on the side it couples through), and runs
under every `sides` unchanged — `transaction_cost`, `turnover_cap`, and `cumulative_adv_participation`
included. When you write your own, do the same unless the term means one side specifically; see
[how to add a term](how-to-add-a-term.md).

The shipped `pro_rata_fill` solve step spends cash into the underweights and so fits `buy` (or `both`);
under `sell` its answer fails the profile's `no_buys` check.

## 3. Set `cash_lb` and `cash_ub` for the direction cash moves

The account's `cash_lb` and `cash_ub` columns keep their meaning — bounds on the cash *after* the
run — but the side fixes which way cash can go, so the starting cash has to be on the right side of
the bound:

| Run | Cash can only | So the start must satisfy | What the bounds mean |
|---|---|---|---|
| `buy` | fall | starting cash ≥ `cash_lb` | `cash_lb` is the floor the run may spend down to; `cash_ub` above the start is irrelevant. |
| `sell` | rise | starting cash ≤ `cash_ub` | `cash_ub` is the most it may raise; `cash_lb` above the start is the least it must raise. |

Two shapes are common. A buy run that invests the cash it starts with: `cash_lb` and `cash_ub` both
`0`, or `0` and `0.02` to leave a small buffer. A sell run that raises cash to a target: `0.1` and
`0.2` raises at least 10% of NAV and at most 20%; `0` and `1` lets the objective decide. A sell run
with both at `0` on a fully invested book is feasible and trades nothing — every sell would raise cash
above the cap.

## 4. Validate, then run

```bash
uv run --env-file .env portfolio-optimizer validate-config configs/my_sell_run.json
uv run --env-file .env portfolio-optimizer run configs/my_sell_run.json
```

`validate-config` lists the same steps as before, minus what you removed; the `[chain]` marker on
`cumulative_adv_participation` now means it reads what higher-priority portfolios *sold*. Every order
the run produces is on the one side (`side` is `SELL` throughout the orders frame), the manifest's
`config.resolved.sides` records the side, and `verify` reads it back to pick the identity checks:

```text
  ok   no_buys                          violation 0.000e+00 (tol 1.0e-06) worst C
  ok   trade_balance                    violation 0.000e+00 (tol 1.0e-06) worst A
  ok   nonneg_sell                      violation 0.000e+00 (tol 1.0e-06) worst C
  ok   buy_absent                       violation 0.000e+00 (tol 1.0e-06) worst A
```

(`no_sells`, `nonneg_buy`, and `sell_absent` for a buy-only run.)

## 5. Read an infeasible start

A book that starts where the side cannot take it is reported as the infeasibility it is, at stage
`solve`, in words. A buy-only book below its cash floor:

```text
  P1: FAILED at solve: InfeasibleError: infeasible problem (spec 42af64bf5893): the book starts with cash 0.000000 below cash_lb 0.100000, and a buy-only run can only lower cash
```

The messages the side profile can add (`domain/sides.py`, `infeasible_starts`):

| Message | Cause | What to change |
|---|---|---|
| `the book starts with cash C below cash_lb L, and a buy-only run can only lower cash` | The floor is above the starting cash. | Lower `cash_lb`, or raise the cash first with a sell run. |
| `the book starts with cash C above cash_ub U, and a sell-only run can only raise cash` | The cap is below the starting cash. | Raise `cash_ub`, or spend the cash first with a buy run. |
| `names whose cap is below their holding, which this side cannot trade out of: [...]` | A name is held above `max_weight` (the style's, or a universe `max_weight` column) and a buy-only run cannot sell it down. | Loosen the cap, or have a rule mark the name `restricted` so it is frozen where it is. |
| `names whose floor is above their holding, which this side cannot trade out of: [...]` | A name is held below a universe `min_weight` floor and a sell-only run cannot buy it up. | Loosen the floor, or mark the name `restricted`. |

Today every such start is infeasible; a per-constraint policy for accepting one (hold the name where it
is and do not worsen it) is an open thread in `IDEAS.md`. The arithmetic diagnoses that apply to any
run — bounds that cannot sum to the required investment, a sector that cannot reach its floor, a name
with no ADV budget left — follow the profile's on the same line.

## What couples in a one-sided run

The chain couples through the side chosen. Under `sell`, a portfolio's tradable set is what it can
sell — held, with `lb < w0` — and it waits for every higher-priority portfolio that can sell a name it
can sell too; `cumulative_adv_participation` then limits its sell in each name to the ADV budget
predecessors' sells left. Expect more coupling than the same book showed under `buy`: every held
name is a potential edge, and the bonds a buy filter took out of the buyable set re-couple accounts
through the sell side. The manifest's `schedule` record shows the difference. Nothing crosses between
a sell run and a buy run; the reasoning, and what a two-sided coupling would cost, are in
[the architecture explanation](explanation-architecture.md#the-side-a-run-trades-is-one-object).
