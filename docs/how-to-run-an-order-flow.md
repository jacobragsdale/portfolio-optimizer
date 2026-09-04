# How to run an order flow: inflow, outflow, rebalance

A run is one order flow: an inflow buys, an outflow sells, a rebalance may do either. A desk that
decides its sells in one process and its buys in another runs two configs over one snapshot, and
rebalances a book that starts outside its bounds with a third; this guide sets them up: what
`order_flow` fixes, what each order flow's config must leave out, how the cash bounds read under each,
what a run refuses and why, and what carries across between them. The shipped
`configs/example_inflow.json`, `configs/example_outflow.json`, and `configs/example_rebalance.json` are
the worked set.

## Prerequisites

- A working run config ([the tutorial](tutorial-first-run.md) gets you one).
- You know which order flow each run is and where the cash it raises or spends goes: an inflow
  can only lower cash, an outflow can only raise it, a rebalance moves it whichever way the bounds
  need.

## 1. Set `order_flow` in each config

```json
"order_flow": "inflow"
```

or `"outflow"` or `"rebalance"`. The key is required. It selects the order-flow profile that every
order-flow-dependent decision goes through, and it fixes which trades portfolios couple through: buys
under `inflow`, sells under `outflow`, both under `rebalance`. Every order flow has `w` alone as its
variable. Under `inflow`, `w ≥ w0` and `buy = w − w0`; under `outflow`, `w ≤ w0` and `sell = w0 − w`;
the other side does not exist in the problem, and a term that rewards selling — a harvestable loss —
is exact. Under `rebalance`, `w` moves anywhere inside its bounds and `buy = max(w − w0, 0)`,
`sell = max(w0 − w, 0)`: both sides exist, both are convex rather than affine, and still no name can
be bought and sold in one solve. The three shipped configs are one wiring with three keys changed at
most: the run's name, `order_flow`, and the objective.

## 2. Leave out what reads the missing side, and what rewards a side under a rebalance

A term that reads `sell` cannot run in an inflow, and `validate-config` says so. The outflow's
`tax_cost` term is `linear` over `sell`, so an inflow config that lists it is rejected:

```bash
uv run portfolio-optimizer validate-config configs/my_inflow.json
```

```text
config rejected: 1 config resolution failure(s): objective[1]: tax_cost: rendering failed: SideUnavailableError: order flow 'inflow' has no 'sell' vector; this term or constraint reads x.sell, so it cannot run under order_flow='inflow'
```

Drop it from `objective`. A kind of your own that reads `x.buy` in an outflow is refused with the
mirror message. The example's other terms read `w` and `trade` (the amount traded on the side the run
has), and the shipped constraint kinds are written against `w`, `trade`, or `x.coupled` (the amount on
the side the run couples through — `participation_limit`), so they run under either `order_flow` unchanged.
A constraint *row* that names `buy` or `sell` as its `vector` in a run without that side is data, not
config, so it is not caught here: it fails its own portfolio at stage `solve`. When you write your own
kind, read `x.trade` or `x.coupled` unless it means one side specifically; see
[how to add a term or constraint kind](how-to-add-a-term.md).

A rebalance has both sides, so `tax_cost` renders — but as a reward on `sell` for every name held at
a loss, and `sell` is `max(w0 − w, 0)` there, convex, so the reward is not convex. The `linear` kind
refuses it by name, per portfolio at stage `solve`, since which names are at a loss is the data's:

```text
  P1: FAILED at solve: TermSpecError: tax_cost: rewards sell on 1 name(s), e.g. ['B']; under a rebalance sell is convex rather than affine, so a reward on it is not convex — a rewarded side belongs to an inflow or an outflow
```

A reward the config itself writes — a negative `weight` on `buy`, `sell`, or `trade` — is caught at
`validate-config` the same way. Costs on any of the three are fine under every order flow, and so is a
`<=` row on them; a `>=` row on `buy`, `sell`, or `trade` under a rebalance is a floor on a convex
quantity, which cvxpy refuses at solve. Harvesting is the outflow's job: `configs/example_rebalance.json`
keeps the inflow's two terms.

The shipped `pro_rata_fill` solve step spends cash into the underweights and so belongs to an
inflow; under `outflow` its answer fails the profile's `no_buys` check.

## 3. Set `cash_lb` and `cash_ub` for the direction cash moves

The account's `cash_lb` and `cash_ub` columns keep their meaning — bounds on the cash *after* the
run — but the order flow fixes which way cash can go, so the starting cash has to be on the right side of
the bound:

| Run | Cash can only | So the start must satisfy | What the bounds mean |
|---|---|---|---|
| `inflow` | fall | starting cash ≥ `cash_lb` | `cash_lb` is the floor the run may spend down to; `cash_ub` above the start is irrelevant. |
| `outflow` | rise | starting cash ≤ `cash_ub` | `cash_ub` is the most it may raise; `cash_lb` above the start is the least it must raise. |
| `rebalance` | move either way | nothing | The bounds mean what they say: the run ends with cash between `cash_lb` and `cash_ub`, wherever it started. |

Two shapes are common. An inflow that invests the cash it starts with: `cash_lb` at `0` and the
objective decides how much of it is worth putting to work, or `cash_lb` and `cash_ub` both `0` to force
every dollar in. An outflow that raises cash to a target: `0.1` and `0.2` raises at least 10% of NAV
and at most 20%; `0` and `1` lets the objective decide. The shipped book gives every account a band of
`0` to somewhere above its starting cash, so one `details` table serves every order flow: the inflow
spends down towards the floor, the outflow raises up towards the cap, the rebalance lands anywhere in
the band. An outflow with both
bounds at the starting cash is feasible and trades nothing — every sell would raise cash above the cap.

## 4. Validate, then run

```bash
uv run portfolio-optimizer validate-config configs/example_outflow.json
uv run portfolio-optimizer run configs/example_outflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

`validate-config` lists the same steps as the inflow and one more term; each account's
`participation_limit` row now reads what higher-priority portfolios *sold*. Every order the run
produces is on the one side (`side` is `SELL` throughout the orders frame), the manifest's
`config.resolved.order_flow` records the order flow, and `verify` reads it back to pick the identity checks:

```text
  ok   no_buys                          violation 6.539e-11 (tol 1.0e-06) worst C
  ok   trade_balance                    violation 6.539e-11 (tol 1.0e-06) worst C
  ok   nonneg_sell                      violation -0.000e+00 (tol 1.0e-06) worst C
  ok   buy_absent                       violation 0.000e+00 (tol 1.0e-06) worst A
  ok   lb                               violation 0.000e+00 (tol 1.0e-06) worst C [binding]
  ok   ub                               violation 0.000e+00 (tol 1.0e-06) worst A
```

(`no_sells`, `nonneg_buy`, and `sell_absent` for an inflow; `trade_balance`, `nonneg_buy`, `nonneg_sell`,
and `no_round_trip` for a rebalance; the box `lb`/`ub` under every order flow.)

## 5. Read an infeasible start

A book that starts where the order flow cannot take it is reported as the infeasibility it is, at stage
`solve`, in words. An inflow below its cash floor:

```text
  P1: FAILED at solve: InfeasibleError: infeasible problem (spec 42af64bf5893): the book starts with cash 0.000000 below cash_lb 0.100000, and an inflow can only lower cash
```

The messages the order-flow profile can add (`domain/order_flow.py`, `infeasible_starts`):

| Message | Cause | What to change |
|---|---|---|
| `the book starts with cash C below cash_lb L, and an inflow can only lower cash` | The floor is above the starting cash. | Lower `cash_lb`, or raise the cash first with the outflow. |
| `the book starts with cash C above cash_ub U, and an outflow can only raise cash` | The cap is below the starting cash. | Raise `cash_ub`, or spend the cash first with the inflow. |
| `names whose cap is below their holding, which this order flow cannot trade out of: [...]` | A name is held above `max_weight` (the style's, or a universe `max_weight` column) and an inflow cannot sell it down. | Set the build's `hold_breached_starts` so the name is held where it is, sell it down in the outflow first, loosen the cap, or have a rule mark the name `restricted`. |
| `names whose floor is above their holding, which this order flow cannot trade out of: [...]` | A name is held below a universe `min_weight` floor and an outflow cannot buy it up. | Set `hold_breached_starts`, buy it up in the inflow first, loosen the floor, or mark the name `restricted`. |

A rebalance adds no message of its own: it can move any weight either way, so every start above is
its ordinary business — the over-cap name is sold down, the cash floor is raised to — which is why a
failed inflow or outflow can be retried as one. The box `lb ≤ w ≤ ub` is part of every solve's
identity, and its start policy is the build's: `{"name": "standard", "params":
{"hold_breached_starts": true}}` moves the breached bound to the current weight, so the name is
held — outside the buyable or sellable set, bought or sold not at all — and the run goes on. Off,
the default, a start outside the box is the data's to fix: the other order flow, a rebalance, a
looser bound, or a rule that freezes the name. A typed constraint *row* — a
`weight_limit`, a `group_limit`, a `cash_limit` — can instead carry `allow_current_weight`, which holds
a bound the book already breaches where it is (do not worsen it) rather than failing the portfolio. The
arithmetic diagnoses that apply to any run — bounds that cannot sum to the required investment, a
turnover cap the start already exceeds, a name with no ADV budget left — follow the profile's on the
same line.

## 6. Hand the outflow's orders to the inflow: the blotter, and the volume it used

Nothing crosses between order flows inside the engine. What the outflow did reaches the inflow as
data — loaded, hashed, and recorded like every other input — and two things about it matter to the
inflow: which names each account just traded, so a loss the outflow harvested is not bought straight
back (the wash-sale rule), and how much of each name's daily volume the outflow's sells already
took, so the inflow's participation budget is what is left. The shipped `load_run_orders` reads the
orders file the outflow's sink wrote in both shapes, and `configs/example_inflow_after_outflow.json`
is the inflow wired through it:

```json
"trades": {
  "loader": {"name": "load_run_orders", "params": {"path": "outflow_orders.csv"}},
  "depends_on": ["portfolios"]
},
"adv_consumed": {
  "loader": {"name": "load_run_orders", "params": {"path": "outflow_orders.csv", "emit": "adv_consumed"}},
  "depends_on": ["universe"]
},
...
"assembly": [
  {"name": "join", "params": {"into": "universe", "source": "signals", "on": ["security_id"], "cardinality": "one_to_one", "require_all_matched": true}},
  {"name": "join", "params": {"into": "universe", "source": "adv_consumed", "on": ["security_id"], "cardinality": "one_to_one", "require_all_matched": true}},
  {"name": "drop", "params": {"datasets": ["signals", "adv_consumed"]}}
],
"rules": ["restrict_low_liquidity", "restrict_recent_trades"]
```

`path` is the orders file, relative to the data root — `examples/data/outflow_orders.csv` is what
`configs/example_outflow.json` writes, checked in so the example runs against no earlier run. As
`trades` it is the blotter: one row per order for the accounts asked for (`portfolio_id`,
`security_id`, `side`, `quantity`, and the run's `as_of_date` as `traded_on`), and the
`restrict_recent_trades` rule freezes every universe name the account traded within `window_days`
of the run's `--as-of` (thirty by default, the US window) by marking it `restricted`, so it keeps its
current weight under every order flow. As `adv_consumed` it is one row per universe security with
`adv_consumed_shares`, the shares the outflow traded in it; the second `join` puts the column on the
universe beside the signals the first brought, and the standard build derives `adv_capacity` as the
participation times the day's volume less those shares, floored at zero — off the budget, exactly as a
predecessor's trades in the same run come off it. Run the outflow, then this config:

```bash
uv run portfolio-optimizer run configs/example_outflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
uv run portfolio-optimizer run configs/example_inflow_after_outflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

The outflow harvests B in most accounts, so the inflow after it buys no B for those — which it
would not have anyway, B's alpha being negative — and, the day's volume in B being spent, every
`participation_limit` on B starts from less. The example keeps the book's holdings as they were; a
desk's custodian answers with the positions after the sells, and the cash the outflow raised is the
cash the inflow's `details` row shows. Three things to know before you copy it:

- **The path names the predecessor, so it is in the config hash.** A run fed by a different run is
  a different run, exactly as a retry's inline ids make it one; `diff-manifests` between two inflows
  fed by two outflows names the `trades` and `adv_consumed` datasets first. A desk whose blotter is
  a service writes a loader that asks it, and keeps `emit`'s two shapes.
- **The same file feeds a retry.** A retry over part of a book (step 7) sees nothing of what the
  first run's solved portfolios traded unless the data says so; point `load_run_orders` at the first
  run's orders and the retry's participation budgets are net of them.
- **An extra dataset with a `portfolio_id` column is cut to each account's own rows.** That is what
  makes `trades` per-account for free, and what a household-level convention must avoid: a blotter
  that should show an account its spouse's fills carries a `household_id` and no `portfolio_id`, and
  the rule matches on the household.

The rule freezes both sides — an inflow cannot rebuy, an outflow cannot sell, a rebalance does
neither — which is the conservative reading; a desk that wants to keep trimming what it just bought,
or to bar only names sold at a loss (the run's `problem_specs/<portfolio>.npz` carries
`tax_per_dollar` per name, so a loader can say which sells were losses), writes the directional
version in the same dozen lines. The trades are loaded rather than remembered across runs, so the
inflow stays a pure function of its own snapshot and the blotter, not the engine, is the record.

The other shape is a constraint row. Give the universe a boolean column naming the names to stay out
of — from a rule, a loader, or whatever your jurisdiction sets — and the build exports it as a flag;
one row closes buys on the flagged names and leaves sells open:

```json
{"kind": "weight_limit", "vector": "buy", "direction": "<=", "bounds": "0", "scope": "sold_at_loss"}
```

A flag column is also what a `participation_limit`'s `scope` reads, so the same column can narrow the
chain coupling to those names.

## 7. Retry a failed run

A run's failures are the manifest's to name and `--retry-of`'s to re-run: `run CONFIG --retry-of
MANIFEST` runs *this* config over exactly the portfolios that manifest recorded as failed, written
inline as the book in their recorded solve order, the run tagged `retry_of`. Which portfolios is
`--retry-stages` (default `solve`; comma-separated from `load, slice, build, solve, worker, skipped`)
and, within those, `--retry-errors` (exception types; a portfolio `fail_fast` skipped is a
`SkippedAfterFailure`). Which config is yours. The three shapes:

| The failure | The retry |
|---|---|
| `InfeasibleError` at `solve`: a start the order flow cannot trade out of (step 5) | The same config with the build's `hold_breached_starts` on — the name is held, the rest of the book trades — or `configs/example_rebalance.json`, which trades the name back inside its bound. |
| `SolverFailureError` or `VerificationError` at `solve`: the solver hit its limit, or the verifier did not agree with it | The same config with a looser `post_solve`, a higher `max_iter`, or another solver: a visible second attempt with its own manifest, never a silent fallback. |
| `skipped`: what `fail_fast` left behind the first failure | The original config, unchanged, over `--retry-stages skipped` — or `solve,skipped`, once the failure's remedy is in the config, to take the failure and its tail in one run. |

```bash
uv run portfolio-optimizer run configs/example_rebalance.json --data-root examples/data --as-of 2026-08-28T00:00:00Z --retry-of out/<failed-run-id>/manifest.json
uv run portfolio-optimizer run configs/my_inflow_held.json --data-root examples/data --as-of 2026-08-28T00:00:00Z --retry-of out/<failed-run-id>/manifest.json --retry-stages solve,skipped
```

The retry is clean: nothing from the failed run reaches it but the ids — no cash carried forward, no
chain, no state; what a run *traded* reaches a later run only as data it loads (step 6). The ids
are written into the retry's config as the inline `portfolios` list, so the retry's config hash
differs from the original's (it is a different run over a different book) and `diff-manifests` says
so; the manifest's `tags.retry_of` names the run it retries. A manifest in which nothing matches is
refused before any data loads, naming what did fail:

```text
no portfolio in run run-b3a1c61b6f8c failed at load; the run recorded 1 at skipped (SkippedAfterFailure), 1 at solve (InfeasibleError)
```

## What couples under each order flow

The chain couples through the trades the order flow makes. Under `outflow`, a portfolio's tradable set is what it can
sell — held, with `lb < w0` — and it waits for every higher-priority portfolio that can sell a name its
own `participation_limit` rows can see; the row then limits its sell in each name to the ADV budget
predecessors' sells left. Expect a different coupling from the same book under `inflow`: every held name
is a potential edge, and the names a buy filter took out of the buyable set re-couple accounts through
the sell side. Under `rebalance` the tradable set is both sets together and a contribution is every
order row, BUY and SELL alike: a predecessor's sell spends a name's daily volume exactly as its buy
does, and `participation_limit` reads what is left either way. The manifest's `schedule` record shows
the difference. What carries between an outflow
and an inflow — the holdings after the sells, the cash raised, the ADV the sells consumed — is a new
snapshot for the next run, and [the architecture explanation](explanation-architecture.md#a-runs-order-flow-is-one-object)
covers why the engine holds that line.
