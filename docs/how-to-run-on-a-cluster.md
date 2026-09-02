# How to run on a cluster

By default a run executes every task in its own process, one after another
(`PORTFOLIO_OPTIMIZER_CLUSTER=inline`), which needs nothing beyond the locked environment and is where
a rule is stepped through under a debugger. To parallelize over per-portfolio tasks it provisions a
Dask cluster for itself and tears it down when it ends: worker processes on this machine, or pods a
Dask Gateway creates for it. This guide sets that up. It changes settings only: the run config is the
same file on a laptop and on the cluster, and hashes the same, so `diff-manifests` never blames the
config for where a run happened to execute.

![The run owns its cluster: provisioning overlaps the load stage](images/cluster-lifecycle.svg)

## Prerequisites

- For a cluster a gateway creates, the `gateway` extra in the image:
  `uv sync --locked --extra gateway`. A local cluster needs nothing beyond the locked environment.
- A reachable [Dask Gateway](https://gateway.dask.org/) and the password its authenticator accepts. The
  gateway owns everything about the pods it creates except the options it chooses to declare; this run
  sets one of them, `image`, so a gateway that does not declare it cannot run this.
- An image that contains this package, the firm's step packages and any term or constraint kinds it
  publishes, the solver the config names (cvxpy installs `CLARABEL`, `OSQP`, `SCS`, and `HIGHS`;
  `PIQP` is `--extra piqp`), and the same locked environment, in a registry the gateway's cluster can
  pull from. The run's own image is the worker image; every worker is checked before the run shares
  any data with it — the config must resolve there, under the run's own
  `PORTFOLIO_OPTIMIZER_STEP_PACKAGES` allowlist, and its fingerprint must equal the run's — and a
  worker that joins later and differs fails its portfolio at stage `worker`.

## 1. Choose the cluster and size it

Which cluster the run provisions is a setting, never a config key:

```bash
PORTFOLIO_OPTIMIZER_CLUSTER=local            # inline (default) | local | https://gateway | tcp://host:8786 | tls://host:8786
PORTFOLIO_OPTIMIZER_MIN_WORKERS=8            # provisioned before the load stage
PORTFOLIO_OPTIMIZER_MAX_WORKERS=48           # scaled to after assembly
PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S=300    # for the first worker to appear
PORTFOLIO_OPTIMIZER_WORKER_IMAGE=registry/optimizer@sha256:...   # gateway only
PORTFOLIO_OPTIMIZER_GATEWAY_PASSWORD=...     # gateway only
PORTFOLIO_OPTIMIZER_GATEWAY_PROXY_ADDRESS=tls://host:8786   # gateway only, and only when it is not the gateway's own host and port
```

| Cluster | Workers | When |
|---|---|---|
| `inline` (default) | this process; every task runs the moment it is submitted, one after another, and the worker counts are moot | the tutorial, debugging a rule, a book small enough that provisioning would cost more than it saves |
| `local` | one worker process per worker on this machine, one thread each | laptops, tests, and books that fit one node |
| `https://gateway`, `http://gateway` | a cluster the gateway creates for this run, its scheduler and workers running `WORKER_IMAGE`, shut down when the run ends | many portfolios or several machines |
| `tcp://host:port`, `tls://host:port` | a scheduler someone else runs; the run connects, submits, and disconnects | when a shared scheduler exists anyway |

- `MIN_WORKERS` is requested as soon as the config resolves and sits idle while data loads;
  `MAX_WORKERS` is requested right after assembly. If node warm-up is what takes long, set the floor
  high and accept the idle pod-minutes; if pods start fast on warm nodes, keep it at one and scale late.
- `MAX_WORKERS` is a ceiling on concurrency, not a promise of it: every build is submitted at once and
  each solve as soon as its own predecessors are known, and the scheduler runs whatever is ready. A book whose
  portfolios all compete for the same buys is one chain of solves however many workers it has; the
  manifest's `schedule` record says how long that chain was.
- `CLUSTER_TIMEOUT_S` bounds how long the run waits, after assembly, for the first worker. A cluster
  that never answers is exit code 3 with a `cluster` failure record in the manifest and every
  portfolio marked skipped.
- `GATEWAY_PASSWORD` is a `SecretStr`: the manifest records that a password was given, never which.
- A gateway has two endpoints, and they are often not the same one. `CLUSTER` is its REST API, ordinary
  HTTPS that proxies happily. Scheduler traffic is raw TLS routed by SNI, which an HTTP proxy cannot
  carry, so deployments usually publish it separately — and then `GATEWAY_PROXY_ADDRESS` is required,
  because unset, `dask-gateway` assumes the REST endpoint's host and port and the client waits for a
  scheduler that is not listening there.

## 2. Try it locally first

`CLUSTER=local` provisions a `LocalCluster` from this process, one worker process per worker with one
thread each, and exercises exactly the code path the gateway's cluster will:

```bash
PORTFOLIO_OPTIMIZER_CLUSTER=local PORTFOLIO_OPTIMIZER_MAX_WORKERS=2 \
  uv run portfolio-optimizer run configs/example_buy.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

The orders are the ones the tutorial produced, and the manifest gains a `cluster` block:

```json
"cluster": {"kind": "local", "min_workers": 1, "max_workers": 2, "workers_ready": 1,
            "scheduler_address": "tcp://127.0.0.1:53211", "provision_started_at": "...", "first_worker_ready_at": "...", "closed_at": "..."}
```

`first_worker_ready_at − provision_started_at` is the start-up the load stage hid.
`tests/engine/test_dask_backend.py` runs this same comparison in the ordinary test suite.

## 3. Run it on a gateway

Set `CLUSTER` to the gateway's address, `GATEWAY_PASSWORD` to the password it accepts, and
`WORKER_IMAGE` to the image the run itself is running — by digest, so a re-tag cannot change what
workers execute.

The run and its workers each read `PORTFOLIO_OPTIMIZER_IMAGE_DIGEST` from their own environment, so
bake it into the image: a client cannot set the environment of pods it did not create. **Run the client
from the worker image too.** A client running somewhere else carries neither that digest nor the same
`git_sha`, and the fingerprint check stops the run rather than answer with two environments — which is
the check working, not a bug in it.

What happens, in order:

1. The config resolves. The gateway is asked for a cluster running `WORKER_IMAGE`, scaled to
   `MIN_WORKERS`.
2. The loaders run. The gateway's scheduler pod and the first workers come up underneath them.
3. Assembly finishes. The run asks for `MAX_WORKERS`, waits for the first worker, checks every worker
   that has joined — the config resolves there, so the solver and every step package are present, and
   its fingerprint equals the run's — and stops with exit code 3 if one cannot. It then scatters the
   assembled datasets and the config once and starts submitting tasks; workers that join later receive
   the data from their peers.
4. Every build runs at once; the pod derives the dependency graph from what the builds report and
   submits each solve with its predecessors' contributions as dependencies, so a solve runs on the
   worker that holds its build the moment its predecessors finish. Outcomes are classified in solve
   order, and each solved portfolio's spec, solution, and chain state are written as `.npz` files the
   moment it is classified, while the cluster is still up.
5. The cluster is shut down in a `finally` — also when inputs are rejected — and then the sink is called
   with every solved portfolio's orders and the manifest is written.

Fairness between runs is the gateway's job: its own `cluster_max_cores` and `cluster_max_memory` cap
what one run can take, and a `ResourceQuota` caps the namespace they all share. Nothing in the engine
arbitrates between runs.

## 4. Point at a scheduler someone else runs

`CLUSTER=tcp://scheduler:8786` connects to an existing scheduler instead of creating one. The run still
scatters its data once and closes its client at the end, but it does not scale or
tear anything down. Every task's fingerprint is compared with the run's, so a shared cluster running an
older image fails its portfolios rather than answering with different code; the manifest's
`versions.workers` shows what it was running.

## What the manifest records

| Field | Meaning |
|---|---|
| `settings` | Every setting the run used, with `cluster` resolved. |
| `cluster` | Kind (`inline`, `local`, `gateway`, `address`), requested sizes, workers joined when the first task could run, scheduler address (`null` under `inline`), and the three timestamps. |
| `versions.workers[]` | Each distinct environment that executed a task — normally one, equal to the run's own — with its hosts and portfolio count. |
| `portfolios[].failure_stage` | `worker` for a task whose environment differed or whose worker died; `cluster` (on the `*` record) when no worker ever came up. |

## Operating notes

- One thread per worker. cvxpy is not thread-safe, and a solve that spins up BLAS threads beside seven
  others is slower, not faster. A local cluster is started that way by the run itself. On a gateway it
  cannot be: threads per worker and the workers' environment are the gateway's configuration, not the
  client's, so `--nthreads 1` and `OMP_NUM_THREADS=1` have to come from the gateway's backend settings
  or be baked into the image. A gateway that hands its workers two threads runs two solves in one
  process.
- The run scales with `scale()`, never `adapt()`; adaptive sizing oscillates on a thousand short tasks.
- `CLUSTER_TIMEOUT_S` bounds only how long the run waits for its first worker. What reaps a cluster
  orphaned by a client killed before its `finally` is the gateway's own `idle_timeout`; set it, because
  the engine has nothing else to offer here.
- `max_in_flight` is per run. Several runs hitting one vendor at the same time each respect their own
  bound and together exceed it; if that is your situation, the limit has to live outside the run.
- Only the options a gateway declares can be set from the client, and this run sets `image`. Worker size
  is whatever that gateway's defaults or its own options say. `engine/dask_backend.py` is the only
  module that touches `dask-gateway`; check its constructor call against the installed version when
  upgrading.
