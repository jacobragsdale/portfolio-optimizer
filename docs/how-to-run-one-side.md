# How to run the buy program and the sell program

A run trades one side. A desk that decides its sells in one process and its buys in another runs two
configs over one snapshot, and this guide sets them up: what `sides` fixes, what each program's config
must leave out, how the cash bounds read under each, what a run refuses and why, and what carries
across between the two. The shipped `configs/example_buy.json` and `configs/example_sell.json` are the
worked pair.

## Prerequisites

- A working run config ([the tutorial](tutorial-first-run.md) gets you one).
- You know which side each program trades and where the cash it raises or spends goes: a buy program
  can only lower cash, a sell program can only raise it.

## 1. Set `sides` in each config

```json
"sides": "buy"
```

or `"sell"`. The key is required. It selects the side profile that every side-dependent decision goes
through, and it fixes which side portfolios couple through: buys under `buy`, sells under `sell`. Under
`buy` the solve has `w` alone with `w ≥ w0` and `buy = w − w0`; under `sell`, `w ≤ w0` and
`sell = w0 − w`. The other side does not exist in the problem, so no name can be bought and sold in one
solve, and a term that rewards selling — a harvestable loss — is exact. The two shipped configs are one
wiring with three keys changed: the run's name, `sides`, and the objective.

## 2. Leave out what reads the missing side

A term that reads `sell` cannot run in a buy program, and `validate-config` says so. The sell program's
`tax_cost` term is `linear` over `sell`, so a buy config that lists it is rejected:

```bash
uv run portfolio-optimizer validate-config configs/my_buy_run.json
```

```text
config rejected: 1 config resolution failure(s): objective[1]: tax_cost: rendering failed: SideUnavailableError: a 'buy' run has no 'sell' vector; this term or constraint reads x.sell, so it cannot run under sides='buy'
```

Drop it from `objective`. A kind of your own that reads `x.buy` in a sell program is refused with the
mirror message. The example's other terms read `w` and `trade` (the amount traded on the side the run
has), and the shipped constraint kinds are written against `w`, `trade`, or `x.coupled` (the amount on
the side the run couples through — `participation_limit`), so they run under either `sides` unchanged.
A constraint *row* that names `buy` or `sell` as its `vector` in a run without that side is data, not
config, so it is not caught here: it fails its own portfolio at stage `solve`. When you write your own
kind, read `x.trade` or `x.coupled` unless it means one side specifically; see
[how to add a term or constraint kind](how-to-add-a-term.md).

The shipped `pro_rata_fill` solve step spends cash into the underweights and so belongs to a buy
program; under `sell` its answer fails the profile's `no_buys` check.

## 3. Set `cash_lb` and `cash_ub` for the direction cash moves

The account's `cash_lb` and `cash_ub` columns keep their meaning — bounds on the cash *after* the
run — but the side fixes which way cash can go, so the starting cash has to be on the right side of
the bound:

| Run | Cash can only | So the start must satisfy | What the bounds mean |
|---|---|---|---|
| `buy` | fall | starting cash ≥ `cash_lb` | `cash_lb` is the floor the run may spend down to; `cash_ub` above the start is irrelevant. |
| `sell` | rise | starting cash ≤ `cash_ub` | `cash_ub` is the most it may raise; `cash_lb` above the start is the least it must raise. |

Two shapes are common. A buy program that invests the cash it starts with: `cash_lb` at `0` and the
objective decides how much of it is worth putting to work, or `cash_lb` and `cash_ub` both `0` to force
every dollar in. A sell program that raises cash to a target: `0.1` and `0.2` raises at least 10% of NAV
and at most 20%; `0` and `1` lets the objective decide. The shipped book gives every account a band of
`0` to somewhere above its starting cash, so one `details` table serves both programs: the buy program
spends down towards the floor, the sell program raises up towards the cap. A sell program with both
bounds at the starting cash is feasible and trades nothing — every sell would raise cash above the cap.

## 4. Validate, then run

```bash
uv run portfolio-optimizer validate-config configs/example_sell.json
uv run portfolio-optimizer run configs/example_sell.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

`validate-config` lists the same steps as the buy program and one more term; each account's
`participation_limit` row now reads what higher-priority portfolios *sold*. Every order the run
produces is on the one side (`side` is `SELL` throughout the orders frame), the manifest's
`config.resolved.sides` records the side, and `verify` reads it back to pick the identity checks:

```text
  ok   no_buys                          violation 6.539e-11 (tol 1.0e-06) worst C
  ok   trade_balance                    violation 6.539e-11 (tol 1.0e-06) worst C
  ok   nonneg_sell                      violation -0.000e+00 (tol 1.0e-06) worst C
  ok   buy_absent                       violation 0.000e+00 (tol 1.0e-06) worst A
  ok   lb                               violation 0.000e+00 (tol 1.0e-06) worst C [binding]
  ok   ub                               violation 0.000e+00 (tol 1.0e-06) worst A
```

(`no_sells`, `nonneg_buy`, and `sell_absent` for a buy program; the box `lb`/`ub` under either side.)

## 5. Read an infeasible start

A book that starts where the side cannot take it is reported as the infeasibility it is, at stage
`solve`, in words. A buy program below its cash floor:

```text
  P1: FAILED at solve: InfeasibleError: infeasible problem (spec 42af64bf5893): the book starts with cash 0.000000 below cash_lb 0.100000, and a buy-only run can only lower cash
```

The messages the side profile can add (`domain/sides.py`, `infeasible_starts`):

| Message | Cause | What to change |
|---|---|---|
| `the book starts with cash C below cash_lb L, and a buy-only run can only lower cash` | The floor is above the starting cash. | Lower `cash_lb`, or raise the cash first with the sell program. |
| `the book starts with cash C above cash_ub U, and a sell-only run can only raise cash` | The cap is below the starting cash. | Raise `cash_ub`, or spend the cash first with the buy program. |
| `names whose cap is below their holding, which this side cannot trade out of: [...]` | A name is held above `max_weight` (the style's, or a universe `max_weight` column) and a buy program cannot sell it down. | Sell it down in the sell program first, loosen the cap, or have a rule mark the name `restricted` so it is frozen where it is. |
| `names whose floor is above their holding, which this side cannot trade out of: [...]` | A name is held below a universe `min_weight` floor and a sell program cannot buy it up. | Buy it up in the buy program first, loosen the floor, or mark the name `restricted`. |

The box `lb ≤ w ≤ ub` is part of every solve's identity, so a start outside it is the data's to fix:
the other program, a looser bound, or a rule that freezes the name. A typed constraint *row* — a
`weight_limit`, a `group_limit`, a `cash_limit` — can instead carry `allow_current_weight`, which holds
a bound the book already breaches where it is (do not worsen it) rather than failing the portfolio. The
arithmetic diagnoses that apply to any run — bounds that cannot sum to the required investment, a
turnover cap the start already exceeds, a name with no ADV budget left — follow the profile's on the
same line.

## 6. Keep the buy program off what the sell program harvested

Nothing crosses between the two programs inside the engine, and that includes a wash-sale rule: a name
the sell program sold at a loss must not be bought straight back, but the buy program only knows what
its inputs say. The mechanism is data, the same as every other limit. Give the universe a boolean
column naming the names to stay out of — from the sell program's orders, a trade-history dataset, or
whatever rule your jurisdiction sets — and the build exports it as a flag; one constraint row closes
buys on the flagged names:

```json
{"kind": "weight_limit", "vector": "buy", "direction": "<=", "bounds": "0", "scope": "sold_at_loss"}
```

Who computes the flag and under which rule is the desk's business; the row is the engine's whole part
in it. A rule that attaches the column is written like any other
([how to add a rule](how-to-add-a-rule.md)), and a flag column is also what a `participation_limit`'s
`scope` reads, so the same column can narrow the chain coupling to those names.

## What couples in a one-sided run

The chain couples through the side chosen. Under `sell`, a portfolio's tradable set is what it can
sell — held, with `lb < w0` — and it waits for every higher-priority portfolio that can sell a name its
own `participation_limit` rows can see; the row then limits its sell in each name to the ADV budget
predecessors' sells left. Expect a different coupling from the same book under `buy`: every held name
is a potential edge, and the names a buy filter took out of the buyable set re-couple accounts through
the sell side. The manifest's `schedule` record shows the difference. What carries between a sell run
and a buy run — the holdings after the sells, the cash raised, the ADV the sells consumed — is a new
snapshot for the next run, and [the architecture explanation](explanation-architecture.md#the-side-a-run-trades-is-one-object)
covers why the engine holds that line.
