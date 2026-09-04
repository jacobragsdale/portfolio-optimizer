"""Run helpers shared by the engine and CLI tests: a fixed clock and id, the example book with files swapped, the hand-checked answers, and one ``execute``."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.types import Clock
from portfolio_optimizer.engine.backends import BackendFactory
from portfolio_optimizer.engine.environment import GitInfo
from portfolio_optimizer.engine.runner import RunContext, RunReport, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import AS_OF, EXAMPLE_DATA, resolved_example_real, two_account_book

GIT = GitInfo(sha="0123456789abcdef", dirty=False)

type Orders = list[dict[str, object]]


def fixed_clock(at: datetime = AS_OF) -> Clock:
    """Always the same instant, so manifests are reproducible in tests."""
    return lambda: at


def io_context(output_dir: Path, data_root: Path = EXAMPLE_DATA, run_id: str = "run-test") -> IoContext:
    """An ``IoContext`` with a fixed clock."""
    return IoContext(data_root=data_root, output_dir=output_dir, run_id=run_id, clock=fixed_clock())


def execution_on(scheduler_address: str, *, max_workers: int = 2) -> ExecutionSettings:
    """Execution settings that connect to ``scheduler_address``."""
    return ExecutionSettings(cluster=scheduler_address, min_workers=1, max_workers=max_workers, cluster_timeout_s=120.0)


# --- the example book, and variations of it with a hand-checked answer each ---

BUY_ORDERS_P1: Orders = [{"security_id": "A", "side": "BUY", "quantity": 1000}, {"security_id": "C", "side": "BUY", "quantity": 25000}]
"""P1 under the inflow: 400,000 of cash on a NAV of 1,000,000, A 3,000 @100 and B 6,000 @50 held.

C has the best alpha and is bought to its ADV budget, a quarter of NAV: 25,000 shares. A is next and
goes to P1's 40% cap: 1,000 shares. B has turned negative, so the last 50,000 stays cash — the cash
floor is 0, not a target.
"""
BUY_ORDERS_P2: Orders = [{"security_id": "A", "side": "BUY", "quantity": 3000}]
"""P2 under the inflow, behind P1: the same book with a 60% cap. P1 spent C's budget for the day, so the cash goes to A, up to the cap: 3,000 shares, and 100,000 stays cash."""
SELL_ORDERS_P1: Orders = [{"security_id": "B", "side": "SELL", "quantity": 2000}]
"""P1 under the outflow: B is held at 60 against a price of 50, a loss its long-term rate turns into 4 cents of tax refund per dollar sold, so it is harvested — down to where the ``TECH`` floor of 0.5 stops it, 2,000 shares. A is at cost and worth holding."""
SELL_ORDERS_P2: Orders = [{"security_id": "B", "side": "SELL", "quantity": 2000}]
"""P2 under the outflow: the same harvest at its short-term rate, to the same floor; B's budget of 250,000 shares a day is nowhere near spent by P1."""
THIN_B_ORDERS_P2: Orders = [{"security_id": "B", "side": "SELL", "quantity": 1000}]
"""P2 under the outflow over ``thin_b_book``: B trades 12,000 shares a day, a quarter of which is 3,000; P1 sold 2,000, so P2 may sell the 1,000 left."""
REBALANCE_ORDERS_P1: Orders = [{"security_id": "A", "side": "BUY", "quantity": 1000}, {"security_id": "B", "side": "SELL", "quantity": 4000}, {"security_id": "C", "side": "BUY", "quantity": 25000}]
"""P1 under the rebalance: the inflow's buys — C to its ADV budget, A to its cap — and, selling now allowed, B sold down to the ``TECH`` floor of 50%: 4,000 shares, leaving 100,000 of it beside A's 400,000."""
REBALANCE_ORDERS_P2: Orders = [{"security_id": "A", "side": "BUY", "quantity": 3000}, {"security_id": "B", "side": "SELL", "quantity": 6000}]
"""P2 under the rebalance, behind P1: A to its 60% cap keeps ``TECH`` above the floor on its own, so all of B goes; C's budget for the day was P1's."""


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


def constraints_without(*kinds: str) -> str:
    """The example's ``constraints`` table with every row of the named kinds left out; without ``participation_limit``, nothing in the book reads the chain."""
    header, rows = _table("constraints.csv")
    return "\n".join([header, *(row for row in rows if row.split(",")[1] not in kinds)]) + "\n"


def holdings_csv(**positions: Sequence[tuple[str, int, str, str]]) -> str:
    """The ``holdings`` table with the named accounts' positions replaced by ``(security_id, quantity, avg_cost, acquired_on)`` rows."""
    header, rows = _table("holdings.csv")
    replaced = [f"{portfolio_id},{security},L1,{quantity},{cost},{acquired}" for portfolio_id, lots in positions.items() for security, quantity, cost, acquired in lots]
    return "\n".join([header, *replaced, *(row for row in rows if row.split(",")[0] not in positions)]) + "\n"


def uncoupled_book(tmp_path: Path, **files: str) -> Path:
    """The two-account book with the chain-aware ADV rows removed: a run over it has nothing reading the chain, so nothing waits."""
    return example_book(tmp_path, **{"constraints.csv": constraints_without("participation_limit"), **files})


def thin_b_book(tmp_path: Path) -> Path:
    """The example data with B's daily volume cut to 12,000 shares: what makes the outflow's two accounts compete for a name's ADV budget (``THIN_B_ORDERS_P2``)."""
    universe = "security_id,price,sector,adv_quantity,increment,restricted\nA,100,TECH,1000000,1,false\nB,50,TECH,12000,1,false\nC,10,HEALTH,100000,1,false\n"
    return example_book(tmp_path, **{"universe.csv": universe})


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
    as_of_date: datetime = AS_OF,
    **config_overrides: object,
) -> RunReport:
    """Run the real example config over ``data_root``, writing under ``tmp_path / run_id``.

    The book is the two-account one unless ``data_root`` names another. ``on_error`` and ``dependencies``
    fill the config's ``execution`` section; any other section is replaced by ``config_overrides``. A run
    on a real cluster passes ``scheduler_address``; a run through a fake passes ``backend_factory`` and the
    address is never dialled; with neither, the run is inline in this process.
    """
    resolved = resolved_example_real(execution={"on_error": on_error, "dependencies": dependencies}, sink=sink, **config_overrides)
    execution = execution_on(scheduler_address if scheduler_address is not None else "inline", max_workers=max_workers)
    root = example_book(tmp_path) if data_root is None else data_root
    context = RunContext(io=io_context(tmp_path / run_id, data_root=root, run_id=run_id), as_of_date=as_of_date, execution=execution, git=GIT, config_path="c.json", settings={})
    if backend_factory is None:
        return run(resolved, context)
    return run(resolved, context, backend_factory=backend_factory)
