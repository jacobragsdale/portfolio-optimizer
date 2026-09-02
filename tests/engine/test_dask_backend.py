"""Tier 2: the run provisions, scales, and tears down its own local cluster, produces the hand-checked answer on it, and asks a gateway for the remote one."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from portfolio_optimizer.engine.backends import ClusterError
from portfolio_optimizer.engine.dask_backend import DaskBackend
from portfolio_optimizer.engine.runner import EXIT_OK, RunContext, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import AS_OF, resolved_example_real
from tests.engine.support import BUY_ORDERS_P1, BUY_ORDERS_P2, GIT, example_book, io_context


def test_a_run_owned_local_cluster_solves_the_example(tmp_path: Path) -> None:
    context = RunContext(
        io=io_context(tmp_path / "dask", data_root=example_book(tmp_path), run_id="dask"),
        as_of_date=AS_OF,
        execution=ExecutionSettings(cluster="local", min_workers=1, max_workers=2, cluster_timeout_s=180.0),
        git=GIT,
        config_path="c.json",
        settings={},
    )
    report = run(resolved_example_real(sink="orders_to_parquet"), context)
    assert report.exit_code == EXIT_OK, [str(o) for o in report.outcomes]
    p1, p2 = report.solved
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == BUY_ORDERS_P1
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == BUY_ORDERS_P2
    cluster = report.manifest.cluster
    assert cluster is not None
    assert (cluster.kind, cluster.min_workers, cluster.max_workers) == ("local", 1, 2)
    assert cluster.scheduler_address is not None and cluster.scheduler_address.startswith("tcp://")
    assert cluster.workers_ready is not None and cluster.workers_ready >= 1
    (worker,) = report.manifest.versions.workers
    assert worker.portfolios == 2
    assert report.manifest.schedule is not None and report.manifest.schedule.edges == 1


def test_a_gateway_is_asked_for_this_runs_image_and_the_cluster_is_scaled_to_min_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway is a network service, so what is checkable here is the request the backend makes of it."""
    asked: dict[str, object] = {}

    class _Cluster:
        def scale(self, workers: int) -> None:
            asked["scaled_to"] = workers

        def close(self) -> None:
            """Part of the cluster surface the backend narrows to; this one is never connected."""

    def _gateway_cluster(**options: object) -> _Cluster:
        asked.update(options)
        return _Cluster()

    monkeypatch.setitem(sys.modules, "dask_gateway", SimpleNamespace(GatewayCluster=_gateway_cluster, BasicAuth=lambda password: f"basic:{password}"))
    execution = ExecutionSettings(
        cluster="https://dask.example",
        min_workers=3,
        max_workers=5,
        cluster_timeout_s=60.0,
        worker_image="registry/optimizer@sha256:abc",
        gateway_password=SecretStr("hunter2"),
        gateway_proxy_address="tls://scheduler.example:8786",
    )
    backend = DaskBackend(execution, run_id="r1")
    assert backend.kind == "gateway"
    backend._gateway_cluster()
    assert asked == {
        "address": "https://dask.example",
        "proxy_address": "tls://scheduler.example:8786",
        "auth": "basic:hunter2",  # the secret is unwrapped once, here, and nowhere else
        "image": "registry/optimizer@sha256:abc",
        "shutdown_on_close": True,
        "scaled_to": 3,  # a gateway cluster starts empty, so min_workers is a scale() the local cluster does not need
    }
    with pytest.raises(ClusterError, match="gateway password"):
        DaskBackend(replace(execution, gateway_password=None), run_id="r1")._gateway_cluster()
