# How to run on a cluster

A run parallelizes over per-portfolio tasks. On a laptop those tasks run in a pool of spawned
interpreters; for a large book, or when several runs share a machine pool, the run can provision its
own Dask cluster — on Kubernetes through the Dask operator, or locally for testing — and tear it down
when it ends. This guide sets that up. It changes settings only: the run config is the same file on a
laptop and on the cluster, and hashes the same, so `diff-manifests` never blames the config for where a
run happened to execute.

![The run owns its cluster: provisioning overlaps the load stage](images/cluster-lifecycle.svg)

## Prerequisites

- The optional extras in the environment, and in the image: `uv sync --locked --extra dask` for a local
  cluster or an existing scheduler, `--extra dask --extra kubernetes` for a cluster the run creates on
  Kubernetes.
- On Kubernetes: the [Dask Kubernetes operator](https://kubernetes.dask.org/) installed in the cluster,
  and a service account for the run's pod that may create, watch, and delete
  `daskclusters.kubernetes.dask.org` in its namespace.
- An image that contains this package, the firm's step packages, and the same locked environment. The
  run's own image is the worker image; every task carries the fingerprint of the process that ran it,
  and a worker whose fingerprint differs from the run's fails its portfolio at stage `worker`.

## 1. Choose the executor and size the cluster

Where work runs is a setting, never a config key. `PORTFOLIO_OPTIMIZER_EXECUTOR` is one of:

| Executor | Workers | When |
|---|---|---|
| `process` | a pool of spawned interpreters on this machine | the default for laptops and small books; needs no Dask |
| `thread` | threads in this process | stepping through a build in one process; cannot solve, so `execution.mode` must not be `parallel` |
| `dask` | a Dask cluster the run provisions and tears down | many portfolios, several machines, or many runs sharing a pool |

With `dask`, three more settings are required and one is conditional:

```bash
PORTFOLIO_OPTIMIZER_EXECUTOR=dask
PORTFOLIO_OPTIMIZER_CLUSTER=auto             # local | kubernetes | auto | tcp://host:8786
PORTFOLIO_OPTIMIZER_MIN_WORKERS=8            # provisioned before the load stage
PORTFOLIO_OPTIMIZER_MAX_WORKERS=48           # scaled to after assembly
PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S=300    # for the first worker to appear
PORTFOLIO_OPTIMIZER_WORKER_IMAGE=registry/optimizer@sha256:...   # kubernetes only
```

- `MIN_WORKERS` is requested as soon as the config resolves and sits idle while data loads;
  `MAX_WORKERS` is requested right after assembly. If node warm-up is what takes long, set the floor
  high and accept the idle pod-minutes; if pods start fast on warm nodes, keep it at one and scale late.
- `MAX_WORKERS` also sets the run's *window*: it keeps twice that many tasks outstanding, whatever the
  executor, so every worker has one task queued behind the one it is running and a run never holds
  its whole book in flight.
- `CLUSTER_TIMEOUT_S` bounds how long the run waits, after assembly, for the first worker. A cluster
  that never answers is exit code 3 with a `cluster` failure record in the manifest and every
  portfolio marked skipped.
- `auto` resolves to `kubernetes` when `KUBERNETES_SERVICE_HOST` is set — every pod has it and no
  laptop does — and to `local` otherwise. The manifest records the resolved value, never `auto`.

A developer who is not touching the cluster needs only `EXECUTOR=process` and `MAX_WORKERS`; the
cluster variables are refused unless the executor is `dask`, so a stale one cannot linger unnoticed.

## 2. Try it locally first

`CLUSTER=local` provisions a `LocalCluster` in this process — one worker process per worker, one
thread each — and exercises exactly the code path the Kubernetes cluster will:

```bash
PORTFOLIO_OPTIMIZER_EXECUTOR=dask PORTFOLIO_OPTIMIZER_CLUSTER=local PORTFOLIO_OPTIMIZER_MIN_WORKERS=1 \
PORTFOLIO_OPTIMIZER_MAX_WORKERS=2 PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S=120 \
uv run --env-file .env portfolio-optimizer run configs/example_run.json
```

The orders are the ones the tutorial produced, and the manifest gains a `cluster` block:

```json
"cluster": {"executor": "dask", "kind": "local", "min_workers": 1, "max_workers": 2, "workers_ready": 1,
            "scheduler_address": "tcp://127.0.0.1:53211", "provision_started_at": "...", "first_worker_ready_at": "...", "closed_at": "..."}
```

`first_worker_ready_at − provision_started_at` is the start-up the load stage hid. The integration test
`tests/engine/test_dask_backend.py` runs this same comparison; `uv run pytest -m integration` runs it.

## 3. Run it in a pod

Set `CLUSTER=auto` (or `kubernetes`) and `WORKER_IMAGE` to the image the run itself is running — by
digest, so a re-tag cannot change what workers execute. If the platform exposes the image digest to the
pod, pass it as `PORTFOLIO_OPTIMIZER_IMAGE_DIGEST`; the run forwards it into the worker pods' environment
and every fingerprint carries it. What happens, in order:

1. The config resolves. A `DaskCluster` resource named after the run id is created with `MIN_WORKERS`
   workers, running `WORKER_IMAGE` with `--nthreads 1` and `OMP_NUM_THREADS=1`.
2. The loaders run. The scheduler pod and the first workers come up underneath them.
3. Assembly finishes. The run asks for `MAX_WORKERS`, waits for the first worker, scatters the assembled
   datasets and the config to it once, and starts submitting tasks; workers that join later receive the
   data from their peers.
4. Results are consumed in solve order; in `parallel_build_sequential_solve` mode the solve chain runs in
   the pod as each build arrives.
5. The cluster is deleted in a `finally` — also when inputs are rejected — and then the orders are
   persisted, the sink is called, and the manifest is written.

Fairness between runs is Kubernetes' job: give each run's namespace a `ResourceQuota` and an urgent run
a `PriorityClass`. Nothing in the engine arbitrates between runs.

## 4. Point at a scheduler someone else runs

`CLUSTER=tcp://scheduler:8786` connects to an existing scheduler instead of creating one. The run still
scatters its data once, keeps its window, and closes its client at the end, but it does not scale or
tear anything down. Every task's fingerprint is compared with the run's, so a shared cluster running an
older image fails its portfolios rather than answering with different code; the manifest's
`versions.workers` shows what it was running.

## What the manifest records

| Field | Meaning |
|---|---|
| `settings` | Every setting the run used, with `cluster` resolved. |
| `cluster` | Executor, kind, requested sizes, workers joined when the first task could run, scheduler address, and the three timestamps. |
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
