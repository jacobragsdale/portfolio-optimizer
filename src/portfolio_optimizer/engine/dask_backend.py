"""A Dask cluster the run owns: ``LocalCluster`` on a laptop, a cluster a Dask Gateway creates for it, or a scheduler address.

Provisioning is issued from a helper thread so :meth:`DaskBackend.start` returns at once and the
scheduler and its first workers come up under the load stage; :meth:`DaskBackend.ready` is where the
run first blocks, and only until one worker can take a task. The run's shared data is scattered once
and every task receives the resulting future, which Dask resolves on whichever worker runs it,
replicating the data between workers on demand; a pending handle passed as an argument is likewise a
Dask future, so the scheduler runs the task where its largest input already is and only once every
input exists. Everything here is the adapter around a partly typed dependency; nothing else in the
engine imports ``distributed``.
"""

import importlib
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import dask
from distributed import Client, LocalCluster, as_completed

from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import ClusterError, Pending, SharedRunData, WorkersReady
from portfolio_optimizer.settings import ExecutionSettings

log = logging.getLogger(__name__)


@runtime_checkable
class _ClusterLike(Protocol):
    """What any cluster object Dask hands back must offer; the one place its untyped surface is typed."""

    def scale(self, n: int) -> object: ...

    def close(self) -> object: ...


def _cluster_like(cluster: object) -> _ClusterLike:
    """Narrow whatever Dask constructed to the surface this backend calls."""
    if not isinstance(cluster, _ClusterLike):
        msg = f"{type(cluster).__name__} does not expose scale() and close()"
        raise ClusterError(msg)
    return cluster


class _DaskPending[T]:
    """A typed handle on a ``distributed.Future``; satisfies :class:`Pending`."""

    def __init__(self, future: object) -> None:
        self.future = future

    def result(self, timeout: float | None = None) -> T:
        """The task's return value, or its exception re-raised — including a dependency's."""
        value: T = self.future.result(timeout=timeout)  # ty: ignore[unresolved-attribute]  # distributed.Future is untyped; the runner only ever passes what submit() returned
        return value


def _unwrap(argument: object) -> object:
    """A pending handle becomes the future Dask resolves on the worker; anything else is passed as is."""
    return argument.future if isinstance(argument, _DaskPending) else argument


class DaskBackend:
    """The run's own cluster, sized from the execution settings and torn down with the run."""

    def __init__(self, execution: ExecutionSettings, *, run_id: str) -> None:
        self._kind = execution.cluster_kind
        self._cluster_setting = execution.cluster
        self._execution = execution
        self._run_id = run_id
        self._min_workers = execution.min_workers
        self._timeout_s = execution.cluster_timeout_s
        self._desired: int | None = None
        self._cluster: _ClusterLike | None = None
        self._client: Client | None = None
        self._provisioner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dask-provision")
        self._connecting: Future[Client] | None = None

    @property
    def kind(self) -> str:
        """``local``, ``gateway``, or ``address``."""
        kind: str = self._kind
        return kind

    def start(self) -> None:
        """Issue provisioning from a helper thread and return."""
        self._connecting = self._provisioner.submit(self._connect)

    def scale(self, workers: int) -> None:
        """Ask for ``workers`` in total; applied now if the cluster exists, else as soon as it does."""
        self._desired = workers
        if self._cluster is not None:
            self._cluster.scale(workers)

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        """Block until the scheduler is reachable and ``workers`` have joined, or raise :class:`ClusterError`."""
        if self._connecting is None:
            msg = "backend was not started"
            raise ClusterError(msg)
        try:
            client = self._connecting.result(timeout=timeout_s)
        except TimeoutError as error:
            msg = f"{self._kind} cluster did not come up within {timeout_s:.0f}s"
            raise ClusterError(msg) from error
        except Exception as error:  # whatever provisioning raised is the cluster's failure to report
            msg = f"{self._kind} cluster could not be provisioned: {type(error).__name__}: {error}"
            raise ClusterError(msg) from error
        if self._cluster is not None and self._desired is not None:
            self._cluster.scale(self._desired)
        try:
            client.wait_for_workers(workers, timeout=timeout_s)
        except TimeoutError as error:
            msg = f"{self._kind} cluster had no worker within {timeout_s:.0f}s"
            raise ClusterError(msg) from error
        info = client.scheduler_info()
        joined = len(info["workers"]) if isinstance(info, dict) and isinstance(info.get("workers"), dict) else workers
        return WorkersReady(workers=joined, scheduler_address=str(client.scheduler.address) if client.scheduler is not None else None)

    def probe[T](self, fn: Callable[..., T], /, *args: object) -> Mapping[str, T]:
        """``fn(*args)`` on every connected worker at once, keyed by worker address; a worker's exception is re-raised here."""
        results: dict[str, T] = {}
        for address, result in self._require_client().run(fn, *args).items():
            results[str(address)] = result
        return results

    def share(self, data: SharedRunData) -> object:
        """Scatter the run's shared data once; the future is what every task receives."""
        return self._require_client().scatter(data, hash=False)

    def submit[T](self, fn: Callable[..., T], /, *args: object, key: str, priority: int) -> Pending[T]:
        """Schedule ``fn`` under a readable key; ``pure=False`` so two runs never share a result, and pending arguments become dependencies."""
        pending: _DaskPending[T] = _DaskPending(self._require_client().submit(fn, *(_unwrap(argument) for argument in args), key=key, priority=priority, pure=False))
        return pending

    def as_completed(self, pendings: Mapping[PortfolioId, Pending[object]]) -> Iterator[PortfolioId]:
        """Keys in the order the scheduler reports them done; a dependency's failure counts as done."""
        by_future = {_unwrap(pending): portfolio_id for portfolio_id, pending in pendings.items()}  # distributed.Future hashes by key
        for future in as_completed(list(by_future), loop=self._require_client().loop):  # the run's client is never the global default
            yield by_future[future]

    def cancel(self, pendings: Sequence[Pending[object]]) -> None:
        """One scheduler round trip for every handle; Dask cancels their dependents with them."""
        if pendings:
            self._require_client().cancel([_unwrap(pending) for pending in pendings])

    def close(self) -> None:
        """Close the client, tear the cluster down, and stop the helper thread; safe to call twice."""
        client = self._client
        cluster = self._cluster
        self._client = None
        self._cluster = None
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
        self._provisioner.shutdown(wait=False, cancel_futures=True)

    def _connect(self) -> Client:
        # A task that kills its worker is not retried across the fleet: a build that OOMs would otherwise evict every other build on three more workers before failing.
        dask.config.set({"distributed.scheduler.allowed-failures": 1})
        if self._kind == "address":
            client = Client(self._cluster_setting, timeout=self._timeout_s, set_as_default=False)
        else:
            cluster = self._local_cluster() if self._kind == "local" else self._gateway_cluster()
            self._cluster = cluster
            if self._desired is not None:
                cluster.scale(self._desired)
            client = Client(cluster, set_as_default=False)
        self._client = client
        log.info("dask %s cluster connected", self._kind, extra={"run_id": self._run_id, "stage": "cluster"})
        return client

    def _local_cluster(self) -> _ClusterLike:
        # An ephemeral dashboard port: `None` falls back to distributed's default 8787, which a second cluster in the same process cannot bind.
        return _cluster_like(LocalCluster(n_workers=self._min_workers, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0", silence_logs=logging.WARNING))

    def _gateway_cluster(self) -> _ClusterLike:
        """A cluster a Dask Gateway creates for this run, its scheduler and workers running this run's image.

        Only the options the gateway declares can be set from here, and ``image`` is the one that
        matters: a scheduler or worker without this package cannot run the run's tasks. One thread per
        worker, ``OMP_NUM_THREADS``, the image digest, and the idle timeout belong to the gateway's own
        configuration or to the image, because a client cannot reach into pods it does not create.

        ``gateway_proxy_address`` is separate from the gateway's own address because the two endpoints
        need not share a host or a port: scheduler traffic is raw TLS routed by SNI, so a deployment
        that terminates the REST API behind an HTTP proxy usually publishes the scheduler proxy
        elsewhere. Left unset, ``dask-gateway`` assumes the gateway's own host and port, which is right
        only when they are in fact the same endpoint.
        """
        gateway = importlib.import_module("dask_gateway")
        if self._execution.gateway_password is None:
            msg = "a gateway cluster needs a gateway password"
            raise ClusterError(msg)
        cluster = _cluster_like(
            gateway.GatewayCluster(
                address=self._cluster_setting,
                proxy_address=self._execution.gateway_proxy_address,
                auth=gateway.BasicAuth(password=self._execution.gateway_password.get_secret_value()),
                image=self._execution.worker_image,
                shutdown_on_close=True,
            )
        )
        cluster.scale(self._min_workers)  # a gateway cluster starts empty: it takes no worker count at construction
        return cluster

    def _require_client(self) -> Client:
        if self._client is None:
            msg = "backend is not ready; call ready() after start()"
            raise ClusterError(msg)
        return self._client
