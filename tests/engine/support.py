"""Run helpers shared by the engine and CLI tests: a fixed clock and ids, the example book with files swapped, the hand-checked answers, and one ``execute``."""

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.engine.backends import BackendFactory
from portfolio_optimizer.engine.environment import GitInfo
from portfolio_optimizer.engine.runner import RunContext, RunReport, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import AS_OF, EXAMPLE_DATA, resolved_example_real

GIT = GitInfo(sha="0123456789abcdef", dirty=False)

type Orders = list[dict[str, object]]


class FixedClock:
    """Always the same instant, so manifests are reproducible in tests."""

    def __init__(self, at: datetime = AS_OF) -> None:
        self.at = at

    def now(self) -> datetime:
        """The fixed instant."""
        return self.at


class FixedIds:
    """Deterministic run ids."""

    def __init__(self, run_id: str = "run-test") -> None:
        self.run_id = run_id

    def new_run_id(self) -> str:
        """The fixed id."""
        return self.run_id


def io_context(output_dir: Path, data_root: Path = EXAMPLE_DATA, run_id: str = "run-test") -> IoContext:
    """An ``IoContext`` with a fixed clock."""
    return IoContext(data_root=data_root, output_dir=output_dir, run_id=run_id, clock=FixedClock())


def execution_on(scheduler_address: str, *, max_workers: int = 2) -> ExecutionSettings:
    """Execution settings that connect to ``scheduler_address``."""
    return ExecutionSettings(cluster=scheduler_address, min_workers=1, max_workers=max_workers, cluster_timeout_s=120.0)


# --- the example book, and variations of it with a hand-checked answer each ---

HAND_OPTIMUM = np.array([0.375, 0.375, 0.25])
"""P1's optimal weights on the example book: C is capped by ADV at a quarter, the rest splits evenly between A and B."""
EXAMPLE_ORDERS_P1: Orders = [{"security_id": "A", "side": "SELL", "quantity": 1250}, {"security_id": "B", "side": "SELL", "quantity": 2500}, {"security_id": "C", "side": "BUY", "quantity": 25000}]
"""P1's orders on the example book: from a half-and-half book at 100 and 50 to ``HAND_OPTIMUM`` on a NAV of 1,000,000."""


def example_book(tmp_path: Path, **files: str) -> Path:
    """A copy of the example data under ``tmp_path`` with the named files replaced by the given text."""
    root = tmp_path / "book"
    shutil.copytree(EXAMPLE_DATA, root)
    for name, content in files.items():
        (root / name).write_text(content)
    return root


def details_csv(portfolio_id: str, **overrides: str) -> str:
    """The example's ``details`` row for one portfolio with named columns overridden."""
    header, row = (EXAMPLE_DATA / "details" / f"{portfolio_id}.csv").read_text().splitlines()[:2]
    names = header.split(",")
    values = dict(zip(names, row.split(","), strict=True)) | overrides
    return f"{header}\n" + ",".join(values[name] for name in names) + "\n"


def no_details_csv(portfolio_id: str) -> str:
    """A ``details`` file with its header and no row: the dataset loads, but this portfolio has no row in it."""
    return (EXAMPLE_DATA / "details" / f"{portfolio_id}.csv").read_text().splitlines()[0] + "\n"


HALF_CASH_ORDERS_P1: Orders = [{"security_id": "A", "side": "BUY", "quantity": 1250}, {"security_id": "B", "side": "BUY", "quantity": 2500}, {"security_id": "C", "side": "BUY", "quantity": 25000}]
HALF_CASH_ORDERS_P2: Orders = [{"security_id": "A", "side": "BUY", "quantity": 2500}, {"security_id": "B", "side": "BUY", "quantity": 5000}]


def half_cash_book(tmp_path: Path) -> Path:
    """The example data with each portfolio holding A 2500 @100 and B 5000 @50 and half its NAV in cash: what a buy-only run invests.

    Targets are a third each and C's ADV budget is 25,000 shares, so the hand answer for P1 is buy
    1,250 A, 2,500 B, and 25,000 C (C is capped at 0.25, the rest splits evenly), and P2 — C's budget
    spent by P1 — buys 2,500 A and 5,000 B: ``HALF_CASH_ORDERS_P1`` and ``HALF_CASH_ORDERS_P2``.
    """
    details = {f"details/{pid}.csv": details_csv(pid, cash="500000") for pid in ("P1", "P2")}
    header = "portfolio_id,security_id,quantity,avg_cost,acquired_on\n"
    holdings = {
        "holdings/P1.csv": header + "P1,A,2500,100,2024-01-15T00:00:00Z\nP1,B,5000,50,2024-01-15T00:00:00Z\n",
        "holdings/P2.csv": header + "P2,A,2500,100,2025-11-01T00:00:00Z\nP2,B,5000,50,2025-11-01T00:00:00Z\n",
    }
    return example_book(tmp_path, **details, **holdings)


SELL_BOOK_ORDERS_P1: Orders = [{"security_id": "A", "side": "SELL", "quantity": 1000}, {"security_id": "B", "side": "SELL", "quantity": 3333}]
SELL_BOOK_ORDERS_P2: Orders = [{"security_id": "B", "side": "SELL", "quantity": 3333}]


def sell_book(tmp_path: Path) -> Path:
    """The example data allowed to raise cash (``cash_ub`` of ``1``) with A's ADV budget cut to 1,000 shares: what a sell-only run trims.

    Each portfolio holds A 0.5 and B 0.5 against a target of a third each, so the hand answer for P1 is
    sell 1,000 A (its whole ADV budget, a 0.1 weight) and 3,333 B (to a third); P2, with A's budget spent
    by P1, sells 3,333 B alone: ``SELL_BOOK_ORDERS_P1`` and ``SELL_BOOK_ORDERS_P2``.
    """
    raise_cash = {f"details/{pid}.csv": details_csv(pid, cash_ub="1") for pid in ("P1", "P2")}
    universe = "security_id,sector,adv_shares,lot_size,restricted\nA,TECH,4000,1,false\nB,TECH,1000000,1,false\nC,HEALTH,100000,1,false\n"
    return example_book(tmp_path, **raise_cash, **{"universe.csv": universe})


# --- one run helper: a cluster run connects by address, a fake run supplies its backend ---


def execute(
    tmp_path: Path,
    *,
    backend_factory: BackendFactory | None = None,
    scheduler_address: str | None = None,
    data_root: Path = EXAMPLE_DATA,
    run_id: str = "run-test",
    max_workers: int = 2,
    sink: str = "orders_to_parquet",
    on_error: str = "fail_fast",
    dependencies: str = "overlap",
    **config_overrides: object,
) -> RunReport:
    """Run the real example config over ``data_root``, writing under ``tmp_path / run_id``.

    ``on_error`` and ``dependencies`` fill the config's ``execution`` section; any other section is
    replaced by ``config_overrides``. A run on a real cluster passes ``scheduler_address``; a run through
    a fake passes ``backend_factory`` and the address is never dialled.
    """
    resolved = resolved_example_real(execution={"on_error": on_error, "dependencies": dependencies}, sink=sink, **config_overrides)
    execution = execution_on(scheduler_address if scheduler_address is not None else "tcp://fake:8786", max_workers=max_workers)
    context = RunContext(io=io_context(tmp_path / run_id, data_root=data_root, run_id=run_id), execution=execution, git=GIT, config_path="c.json", settings={})
    if backend_factory is None:
        return run(resolved, context)
    return run(resolved, context, backend_factory=backend_factory)
