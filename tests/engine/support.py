"""Run helpers shared by the engine and CLI tests: a fixed clock and ids, the example book with files swapped, the hand-checked answers, and one ``execute``."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.engine.backends import BackendFactory
from portfolio_optimizer.engine.environment import GitInfo
from portfolio_optimizer.engine.runner import RunContext, RunReport, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import AS_OF, EXAMPLE_DATA, resolved_example_real, two_account_book

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

HAND_OPTIMUM = np.array([0.35, 0.4, 0.25])
"""P1's optimal weights on the example book.

C has the best alpha and is bought to its ADV budget, a quarter of NAV. The quarter is raised by
selling A and B, and A goes first: B is at a 20% unrealized gain, so selling it costs 4 cents of tax
per dollar against A's nothing. B still falls to P1's 40% single-name cap, which it starts above.
"""
EXAMPLE_ORDERS_P1: Orders = [{"security_id": "A", "side": "SELL", "quantity": 1500}, {"security_id": "B", "side": "SELL", "quantity": 2000}, {"security_id": "C", "side": "BUY", "quantity": 25000}]
"""P1's orders on the example book: from a half-and-half book at 100 and 50 to ``HAND_OPTIMUM`` on a NAV of 1,000,000."""


def example_book(tmp_path: Path, **files: str) -> Path:
    """The two-account book under ``tmp_path`` with the named tables replaced by the given text."""
    return two_account_book(tmp_path / "book", **files)


def _table(name: str) -> tuple[str, list[str]]:
    header, *rows = (EXAMPLE_DATA / name).read_text().splitlines()
    return header, rows


def details_csv(**overrides: Mapping[str, str]) -> str:
    """The example's ``details`` table with named columns overridden for the named accounts."""
    header, rows = _table("details.csv")
    names = header.split(",")
    written = []
    for row in rows:
        values = dict(zip(names, row.split(","), strict=True))
        values |= overrides.get(values["portfolio_id"], {})
        written.append(",".join(values[name] for name in names))
    return "\n".join([header, *written]) + "\n"


def details_without(*portfolio_ids: str) -> str:
    """The ``details`` table with the named accounts left out: the dataset loads, but they have no row in it."""
    header, rows = _table("details.csv")
    return "\n".join([header, *(row for row in rows if row.split(",")[0] not in portfolio_ids)]) + "\n"


def holdings_csv(**positions: Sequence[tuple[str, int, str, str]]) -> str:
    """The ``holdings`` table with the named accounts' positions replaced by ``(security_id, quantity, avg_cost, acquired_on)`` rows."""
    header, rows = _table("holdings.csv")
    replaced = [f"{portfolio_id},{security},{quantity},{cost},{acquired}" for portfolio_id, lots in positions.items() for security, quantity, cost, acquired in lots]
    return "\n".join([header, *replaced, *(row for row in rows if row.split(",")[0] not in positions)]) + "\n"


HALF_CASH_ORDERS_P1: Orders = [{"security_id": "A", "side": "BUY", "quantity": 1500}, {"security_id": "B", "side": "BUY", "quantity": 2000}, {"security_id": "C", "side": "BUY", "quantity": 25000}]
HALF_CASH_ORDERS_P2: Orders = [{"security_id": "A", "side": "BUY", "quantity": 3500}, {"security_id": "B", "side": "BUY", "quantity": 3000}]


def half_cash_book(tmp_path: Path) -> Path:
    """The example data with each portfolio holding A 2500 @100 and B 5000 @50 and half its NAV in cash: what a buy-only run invests.

    Half of NAV goes to work, best name first, net of what each costs to trade. For P1 that is C to its
    ADV budget of 25,000 shares, then A and B to P1's 40% cap: buy 1,500 A, 2,000 B, 25,000 C. P2, with
    C's budget spent by P1 and a 60% cap, puts the same half into A first: 3,500 A and 3,000 B.
    ``HALF_CASH_ORDERS_P1`` and ``HALF_CASH_ORDERS_P2``.
    """
    holdings = holdings_csv(
        P1=[("A", 2500, "100", "2024-01-15T00:00:00Z"), ("B", 5000, "50", "2024-01-15T00:00:00Z")], P2=[("A", 2500, "100", "2025-11-01T00:00:00Z"), ("B", 5000, "50", "2025-11-01T00:00:00Z")]
    )
    half = {"cash": "500000"}
    return example_book(tmp_path, **{"details.csv": details_csv(P1=half, P2=half), "holdings.csv": holdings})


SELL_BOOK_ORDERS_P1: Orders = [{"security_id": "A", "side": "SELL", "quantity": 1000}, {"security_id": "B", "side": "SELL", "quantity": 8000}]
SELL_BOOK_ORDERS_P2: Orders = [{"security_id": "B", "side": "SELL", "quantity": 10000}]


def sell_book(tmp_path: Path) -> Path:
    """The example data allowed to raise cash (``cash_ub`` of ``1``) over a universe whose alphas have turned negative, with A's ADV cut to 4,000 shares: what a sell-only run trims.

    Both held names are worth less than nothing now, so both are sold; B far enough underwater that even
    P2's short-term rate on its gain does not hold it. A's ADV budget is 1,000 shares and P1 takes all of
    it, which leaves A at the 40% cap and the ``TECH`` floor of 0.5 holding 0.1 of B back: P1 sells
    1,000 A and 8,000 B. P2, with no A budget left, keeps A at 0.5 and so has the whole floor covered:
    it sells all 10,000 B. ``SELL_BOOK_ORDERS_P1`` and ``SELL_BOOK_ORDERS_P2``.
    """
    raise_cash = {"cash_ub": "1"}
    universe = "security_id,price,sector,adv_shares,lot_size,restricted,alpha,tcost_bps\nA,100,TECH,4000,1,false,-0.03,5\nB,50,TECH,1000000,1,false,-0.10,5\nC,10,HEALTH,100000,1,false,0.05,20\n"
    return example_book(tmp_path, **{"details.csv": details_csv(P1=raise_cash, P2=raise_cash), "universe.csv": universe})


# --- one run helper: a cluster run connects by address, a fake run supplies its backend ---


def execute(
    tmp_path: Path,
    *,
    backend_factory: BackendFactory | None = None,
    scheduler_address: str | None = None,
    data_root: Path | None = None,
    run_id: str = "run-test",
    max_workers: int = 2,
    sink: str = "orders_to_parquet",
    on_error: str = "fail_fast",
    dependencies: str = "overlap",
    **config_overrides: object,
) -> RunReport:
    """Run the real example config over ``data_root``, writing under ``tmp_path / run_id``.

    The book is the two-account one unless ``data_root`` names another. ``on_error`` and ``dependencies``
    fill the config's ``execution`` section; any other section is replaced by ``config_overrides``. A run on a real cluster passes ``scheduler_address``; a run through
    a fake passes ``backend_factory`` and the address is never dialled.
    """
    resolved = resolved_example_real(execution={"on_error": on_error, "dependencies": dependencies}, sink=sink, **config_overrides)
    execution = execution_on(scheduler_address if scheduler_address is not None else "tcp://fake:8786", max_workers=max_workers)
    root = example_book(tmp_path) if data_root is None else data_root
    context = RunContext(io=io_context(tmp_path / run_id, data_root=root, run_id=run_id), execution=execution, git=GIT, config_path="c.json", settings={})
    if backend_factory is None:
        return run(resolved, context)
    return run(resolved, context, backend_factory=backend_factory)
