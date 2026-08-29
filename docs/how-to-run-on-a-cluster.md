# How to run on a cluster

A run parallelizes over per-portfolio tasks on a Dask cluster it provisions for itself and tears down
when it ends: worker processes on this machine, or pods on Kubernetes through the Dask operator. This
guide sets that up. It changes settings only: the run config is the same file on a laptop and on the
cluster, and hashes the same, so `diff-manifests` never blames the config for where a run happened to
execute.

![The run owns its cluster: provisioning overlaps the load stage](images/cluster-lifecycle.svg)

## Prerequisites

- For a cluster the run creates on Kubernetes, the `kubernetes` extra in the image:
  `uv sync --locked --extra kubernetes`. A local cluster needs nothing beyond the locked environment.
- On Kubernetes: the [Dask Kubernetes operator](https://kubernetes.dask.org/) installed in the cluster,
  and a service account for the run's pod that may create, watch, and delete
  `daskclusters.kubernetes.dask.org` in its namespace.
- An image that contains this package, the firm's step packages, the solver the config names (cvxpy
  installs `CLARABEL`, `OSQP`, `SCS`, and `HIGHS`; `PIQP` is `--extra piqp`), and the same locked
  environment. The run's own image is the worker image; every worker is checked before the run shares
  any data with it — the config must resolve there and its fingerprint must equal the run's — and a
  worker that joins later and differs fails its portfolio at stage `worker`.

## 1. Choose the cluster and size it

Which cluster the run provisions is a setting, never a config key:

```bash
PORTFOLIO_OPTIMIZER_CLUSTER=auto             # local | kubernetes | auto | tcp://host:8786
PORTFOLIO_OPTIMIZER_MIN_WORKERS=8            # provisioned before the load stage
PORTFOLIO_OPTIMIZER_MAX_WORKERS=48           # scaled to after assembly
PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S=300    # for the first worker to appear
PORTFOLIO_OPTIMIZER_WORKER_IMAGE=registry/optimizer@sha256:...   # kubernetes only
```

| Cluster | Workers | When |
|---|---|---|
| `local` | one worker process per worker on this machine, one thread each | laptops, tests, and books that fit one node |
| `kubernetes` | a `DaskCluster` of pods the run creates through the operator and deletes when it ends | many portfolios or several machines |
| `tcp://host:port` | a scheduler someone else runs; the run connects, submits, and disconnects | when a shared scheduler exists anyway |
| `auto` | `kubernetes` inside a pod, `local` anywhere else | one setting for both places |

- `MIN_WORKERS` is requested as soon as the config resolves and sits idle while data loads;
  `MAX_WORKERS` is requested right after assembly. If node warm-up is what takes long, set the floor
  high and accept the idle pod-minutes; if pods start fast on warm nodes, keep it at one and scale late.
- `MAX_WORKERS` is a ceiling on concurrency, not a promise of it: every build is submitted at once and
  every solve as soon as the schedule is known, and the scheduler runs whatever is ready. A book whose
  portfolios all compete for the same buys is one chain of solves however many workers it has; the
  manifest's `schedule` record says how long that chain was.
- `CLUSTER_TIMEOUT_S` bounds how long the run waits, after assembly, for the first worker. A cluster
  that never answers is exit code 3 with a `cluster` failure record in the manifest and every
  portfolio marked skipped.
- `auto` resolves to `kubernetes` when `KUBERNETES_SERVICE_HOST` is set — every pod has it and no
  laptop does — and to `local` otherwise. The manifest records the resolved value, never `auto`.

## 2. Try it locally first

`CLUSTER=local` — what `.env.example` sets — provisions a `LocalCluster` in this process, one worker
process per worker with one thread each, and exercises exactly the code path the Kubernetes cluster
will:

```bash
uv run --env-file .env portfolio-optimizer run configs/example_run.json
```

The orders are the ones the tutorial produced, and the manifest gains a `cluster` block:

```json
"cluster": {"kind": "local", "min_workers": 1, "max_workers": 2, "workers_ready": 1,
            "scheduler_address": "tcp://127.0.0.1:53211", "provision_started_at": "...", "first_worker_ready_at": "...", "closed_at": "..."}
```

`first_worker_ready_at − provision_started_at` is the start-up the load stage hid.
`tests/engine/test_dask_backend.py` runs this same comparison in the ordinary test suite.

## 3. Run it in a pod

Set `CLUSTER=auto` (or `kubernetes`) and `WORKER_IMAGE` to the image the run itself is running — by
digest, so a re-tag cannot change what workers execute. If the platform exposes the image digest to the
pod, pass it as `PORTFOLIO_OPTIMIZER_IMAGE_DIGEST`; the run forwards it into the worker pods' environment
and every fingerprint carries it. What happens, in order:

1. The config resolves. A `DaskCluster` resource named after the run id is created with `MIN_WORKERS`
   workers, running `WORKER_IMAGE` with `--nthreads 1` and `OMP_NUM_THREADS=1`.
2. The loaders run. The scheduler pod and the first workers come up underneath them.
3. Assembly finishes. The run asks for `MAX_WORKERS`, waits for the first worker, checks every worker
   that has joined — the config resolves there, so the solver and every step package are present, and
   its fingerprint equals the run's — and stops with exit code 3 if one cannot. It then scatters the
   assembled datasets and the config once and starts submitting tasks; workers that join later receive
   the data from their peers.
4. Every build runs at once; the pod derives the dependency graph from what the builds report and
   submits each solve with its predecessors' contributions as dependencies, so a solve runs on the
   worker that holds its build the moment its predecessors finish. Outcomes are classified in solve order.
5. The cluster is deleted in a `finally` — also when inputs are rejected — and then the orders are
   persisted, the sink is called, and the manifest is written.

Fairness between runs is Kubernetes' job: give each run's namespace a `ResourceQuota` and an urgent run
a `PriorityClass`. Nothing in the engine arbitrates between runs.

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
| `cluster` | Kind, requested sizes, workers joined when the first task could run, scheduler address, and the three timestamps. |
| `versions.workers[]` | Each distinct environment that executed a task — normally one, equal to the run's own — with its hosts and portfolio count. |
| `portfolios[].failure_stage` | `worker` for a task whose environment differed or whose worker died; `cluster` (on the `*` record) when no worker ever came up. |

## Operating notes

- One thread per worker. cvxpy is not thread-safe, and a solve that spins up BLAS threads beside seven
  others is slower, not faster. The run sets `--nthreads 1` and `OMP_NUM_THREADS=1` itself.
- The run scales with `scale()`, never `adapt()`; adaptive sizing oscillates on a thousand short tasks.
- The scheduler is started with an idle timeout equal to `CLUSTER_TIMEOUT_S`, so a cluster orphaned by a
  client pod that was killed before its `finally` exits on its own. For belt and braces, give the
  `DaskCluster` resource an owner reference to the client's Job so Kubernetes garbage-collects it; that
  is a deployment concern, not an engine one.
- Rate limits are per run. Several runs hitting one vendor at the same time each respect their own pool
  and together exceed it; if that is your situation, the limiter has to live outside the run.
- The `dask-kubernetes` API has changed more than once. `engine/dask_backend.py` is the only module that
  touches it; check its constructor call against the installed version when upgrading.
