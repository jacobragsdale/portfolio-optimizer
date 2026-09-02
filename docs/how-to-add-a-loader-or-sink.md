# How to add a loader or a sink

Loaders and sinks are the engine's only I/O. This guide adds a loader that reads a dataset from a
database and a sink that submits orders to a trading system, keeping each testable without the real
dependency.

## Prerequisites

- You know the dataset's shape. Engine-known datasets (`portfolios`, `holdings`, `universe`, `details`,
  `constraints`) must satisfy the schemas in `src/portfolio_optimizer/domain/schemas.py`
  after assembly (`holdings` and `universe` may carry any further columns); any other dataset only
  needs the columns your assembly steps and rules use, typed the way you want them to arrive — give its
  loader a `FrameSchema` of its own, as the shipped `load_parameters` does, so a join never has to guess
  a key's dtype.

## Add a loader

### 1. Write the function in `src/portfolio_optimizer/loaders.py` — or in your package

A loader shared across desks belongs in a package installed in the environment and is named
`my_firm.loaders:holdings_from_sql` in the config — or published as an entry point in the group
`portfolio_optimizer.loader` and named bare; the manifest records the package's version either way.
When `PORTFOLIO_OPTIMIZER_STEP_PACKAGES` names an allowlist, a qualified name must come from one of
those packages.

```python
class SqlParams(Params):
    query: str = Field(min_length=1)
    dsn_env: str = "PORTFOLIO_OPTIMIZER_DSN"


def holdings_from_sql(request: LoadRequest, params: SqlParams) -> pd.DataFrame:
    """Read holdings for the requested portfolios as of ``request.as_of_date``."""
    frame = my_gateway.query(params.query, portfolio_ids=request.portfolio_ids, as_of_date=request.as_of_date)
    return coerce_frame(frame, DATASET_SCHEMAS[request.dataset])
```

- `request: LoadRequest` carries `dataset` (the config key being loaded), `portfolio_ids` in solve
  order (filled when the entry's `depends_on` names `portfolios`; `per_portfolio` implies it),
  `inputs` (the frames of the datasets named in `depends_on`), `as_of_date`, `data_root`, and
  `run_id`. How many calls run at once is the config's, not the loader's (see below).
- Return `pd.DataFrame` with every dtype declared. `coerce_frame` casts to the dataset's schema and turns
  money written as strings, ints, or floats into `Decimal` — do this at the read boundary, not later.
- Every dataset loader returns a DataFrame, `constraints` included; money inside a frame may
  be strings (`"0.05"`), which the engine validates into `Decimal`.

Pass the database client in as a parameter of your gateway object rather than reaching for a global,
so a tier-4 contract test can call the real query and validate its shape with the production schema.

### Async loaders and fan-out

Every dataset loader starts the moment the datasets its entry depends on have loaded — with no
`depends_on`, the moment the run does: an `async def` loader runs on the engine's event loop, a plain
`def` loader in a worker thread. Use `async def` for a source with an async client; a blocking driver
is fine as a plain function and still overlaps with the other loaders. A loader that needs another
dataset's rows — a vendor whose query wants the universe's tickers, say — names it in `depends_on` and
reads `request.inputs["universe"]` rather than loading it again.

A source that answers one portfolio per call needs fan-out and a bound on it, and both belong to the
engine rather than to your loader:

```json
"datasets": {
  "portfolios": {"loader": "portfolios_from_api"},
  "holdings": {"loader": "holdings_from_api", "scope": "per_portfolio", "batch_size": 1, "max_in_flight": 8},
  "universe": {"loader": "universe_from_api"},
  "details": {"loader": "details_from_sql", "scope": "per_portfolio", "batch_size": 25, "max_in_flight": 32}
}
```

`scope: "per_portfolio"` says the ids are the engine's to cut up, `batch_size` says how finely, and
`max_in_flight` says how many of those calls may be open at once — 8 against the fragile vendor, 32
against a database that takes concurrent queries happily. A per-portfolio dataset implies
`depends_on: ["portfolios"]`, which is what fills its `request.portfolio_ids`; `universe` declares
nothing and starts immediately. The loader is then written for the batch it was handed and counts
nothing:

```python
async def holdings_from_api(request: LoadRequest, params: ApiParams) -> pd.DataFrame:
    client = build_client(params)
    frame = await client.holdings(request.portfolio_ids, as_of_date=request.as_of_date)
    return coerce_frame(frame, DATASET_SCHEMAS[request.dataset])
```

Under `batch_size: 1` that is one account per call and the engine keeps 8 of them running; under
`batch_size: 25` it is one query per 25 ids. A loader may still fan out privately — `asyncio.gather`
over the batch's ids, as the shipped `load_holdings` does — but nothing bounds those calls, so keep
the fan-out the engine's whenever the source needs bounding. One failure inside a private fan-out
surfaces as an `ExceptionGroup`, which the engine unwraps when it holds a single failure so the log
and the manifest record that error's own type and message rather than the group's.

Letting the engine cut the book buys three things a private fan-out cannot: a failed account fails
alone, the batches are visible in the manifest, and the whole stage overlaps the global loaders. The
shipped example does exactly this over a hundred accounts — `holdings` with `batch_size: 1` and
`max_in_flight: 8`, `details` with `batch_size: 25` and `max_in_flight: 4` — while its five other
datasets stay global. `load_details` is the blocking twin, a plain `def` the engine runs in a worker
thread; copy whichever matches your source.

Two things follow from a per-portfolio dataset being loaded in pieces:

- **Assembly never sees it.** Assembly steps run over whole datasets, before the batches are back.
  Attach its columns in a [rule](how-to-add-a-rule.md) instead, which already runs per portfolio.
- **A batch that fails fails only its own portfolios.** They are recorded as failures at stage `load`
  and the rest of the book runs; `on_error` decides whether the run stops there. If *no* batch comes
  back the source is down rather than an account being bad, and the run is rejected like any other
  dataset failure.

The manifest records how long each dataset took (`load_time_s`), when it started and what it waited on
(`started_s`, `depends_on`), how many calls it was cut into
(`batches`), and how many portfolios a failed batch cost (`rejected`), and the `timing` block times
each dataset, so a slow run can be traced to the dataset and the bound that paced it.

### 2. Name it in the config

```json
"datasets": {
  "holdings": {"loader": {"name": "holdings_from_sql", "params": {"query": "EXEC dbo.holdings_asof ?"}}},
  ...
}
```

Datasets that are not engine-known are combined by the `assembly` steps — the shipped `join`
declares its cardinality (`one_to_one`, `one_to_many`, `many_to_one`), can require every row to match,
and never silently overwrites a column the target frame already has — or carried into each portfolio's
bundle as `data.extras`. See [how to add security analytics](how-to-add-security-analytics.md).

### 3. Check and test

`validate-config` confirms the signature. For the loader itself, write one contract test marked
`integration` that runs the real query and validates the result with the same schema production
uses — shape only, never values.

## Add a sink

### 1. Write the function in `src/portfolio_optimizer/sinks.py` — or in your package

A sink in a package is named `my_firm.sinks:orders_to_gateway`, or published in the group
`portfolio_optimizer.sink` and named bare.

```python
class GatewaySinkParams(Params):
    account: str = Field(min_length=1)


def orders_to_gateway(orders: pd.DataFrame, io: IoContext, params: GatewaySinkParams) -> tuple[Artifact, ...]:
    """Submit the run's orders and return the gateway's acknowledgement as an artifact."""
    gateway = build_gateway(params.account)  # your own client, behind your own seam
    return gateway.submit(orders, io.run_id)
```

- The engine calls the sink **once per run**, after every portfolio has been processed, with the orders
  of every solved portfolio concatenated and sorted by `(portfolio_id, security_id)`. It is not called
  when nothing solved.
- Return a tuple of `Artifact(path, sha256, size_bytes)` describing what was written or acknowledged;
  the manifest records them. For a network destination, write the acknowledgement to a file under
  `io.output_dir / io.run_id` and return that.
- Keep the network client behind a seam of your own — a `Protocol` in your module — so the sink can be
  exercised against a fake in tests.
- A sink that raises is recorded in the manifest as a `sink` failure and the run exits with code 3.

### 2. Name it in the config

```json
"sink": {"name": "orders_to_gateway", "params": {"account": "SMA-TAX-01"}}
```

### 3. Test

Write the sink's test against a fake gateway; assert on the artifacts returned and on what the fake
received. The shipped `orders_to_parquet` and `orders_to_csv` go through
`engine.files.write_atomically` (temp file plus rename) — use it for any file destination so a crash
never leaves a partial file behind.
