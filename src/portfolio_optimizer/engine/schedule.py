"""Derive the run's schedule: the order portfolios solve in, and which ones each waits for.

Pure functions over what the builds reported; nothing here knows about Dask. Portfolios couple
across a run through one side only — the side profile's tradable set — so portfolio *j* depends on
every higher-priority *i* that can trade a security *j* can trade too, and on nothing else. The graph is never transitively reduced: a solve
folds its *direct* predecessors' own contributions, so every overlapping earlier portfolio must stay a direct
dependency.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np

from portfolio_optimizer.domain.types import PortfolioId, StrictModel

type Coupling = Literal["none", "overlap", "all"]
"""``none``: nothing reads the chain, so nothing waits. ``overlap``: wait for higher-priority portfolios with a shared tradable security. ``all``: wait for every higher-priority portfolio."""


def order_portfolios(keys: Mapping[PortfolioId, Decimal]) -> tuple[PortfolioId, ...]:
    """Ascending solve-order key, ties broken on ``portfolio_id``; the priority order of the run."""
    return tuple(sorted(keys, key=lambda portfolio_id: (keys[portfolio_id], portfolio_id)))


class ScheduleSummary(StrictModel):
    """The shape of a schedule, for the manifest and the log.

    ``coupling`` is ``none`` when no step read the chain, ``overlap`` when portfolios waited only for
    higher-priority portfolios with a shared tradable security, ``all`` when every higher-priority
    portfolio was a predecessor. ``critical_path`` counts solves that had to run one after another.
    """

    coupling: Coupling
    portfolios: int
    edges: int
    components: int
    largest_component: int
    critical_path: int


@dataclass(frozen=True, slots=True)
class Schedule:
    """Who solves after whom. ``predecessors[j]`` lists, in solve order, every portfolio ``j`` folds into its chain."""

    order: tuple[PortfolioId, ...]
    predecessors: Mapping[PortfolioId, tuple[PortfolioId, ...]]
    coupling: Coupling

    def __post_init__(self) -> None:
        if set(self.predecessors) != set(self.order):
            msg = "predecessors must be listed for exactly the portfolios in order"
            raise ValueError(msg)
        position = {portfolio_id: index for index, portfolio_id in enumerate(self.order)}
        for portfolio_id, earlier in self.predecessors.items():
            if any(position[other] >= position[portfolio_id] for other in earlier):
                msg = f"portfolio {portfolio_id!r} depends on a portfolio that does not precede it"
                raise ValueError(msg)

    def successors(self) -> dict[PortfolioId, tuple[PortfolioId, ...]]:
        """The inverse of ``predecessors``, in solve order."""
        following: dict[PortfolioId, list[PortfolioId]] = {portfolio_id: [] for portfolio_id in self.order}
        for portfolio_id in self.order:
            for earlier in self.predecessors[portfolio_id]:
                following[earlier].append(portfolio_id)
        return {portfolio_id: tuple(later) for portfolio_id, later in following.items()}

    def heights(self) -> dict[PortfolioId, int]:
        """Longest chain of solves that starts at each portfolio, itself included; what to solve first."""
        following = self.successors()
        height: dict[PortfolioId, int] = {}
        for portfolio_id in reversed(self.order):
            height[portfolio_id] = 1 + max((height[later] for later in following[portfolio_id]), default=0)
        return height

    def summary(self) -> ScheduleSummary:
        """Edge count, connected components, and the critical path in solves."""
        parent = {portfolio_id: portfolio_id for portfolio_id in self.order}

        def root(portfolio_id: PortfolioId) -> PortfolioId:
            while parent[portfolio_id] != portfolio_id:
                parent[portfolio_id] = parent[parent[portfolio_id]]
                portfolio_id = parent[portfolio_id]
            return portfolio_id

        edges = 0
        for portfolio_id in self.order:
            for earlier in self.predecessors[portfolio_id]:
                edges += 1
                parent[root(earlier)] = root(portfolio_id)
        sizes: dict[PortfolioId, int] = {}
        for portfolio_id in self.order:
            sizes[root(portfolio_id)] = sizes.get(root(portfolio_id), 0) + 1
        depth: dict[PortfolioId, int] = {}
        for portfolio_id in self.order:
            depth[portfolio_id] = 1 + max((depth[earlier] for earlier in self.predecessors[portfolio_id]), default=0)
        return ScheduleSummary(
            coupling=self.coupling, portfolios=len(self.order), edges=edges, components=len(sizes), largest_component=max(sizes.values(), default=0), critical_path=max(depth.values(), default=0)
        )


def dependency_graph(order: Sequence[PortfolioId], tradable: Mapping[PortfolioId, Iterable[str]], unknown: frozenset[PortfolioId], coupling: Coupling) -> Schedule:
    """Which higher-priority portfolios each portfolio waits for.

    ``tradable`` is each built portfolio's tradable securities; ``unknown`` names portfolios whose build
    failed, so their tradable set is unknown and they are treated as overlapping every later portfolio
    (and every earlier one). Under ``all`` every earlier portfolio is a predecessor; under ``none``
    nothing is.
    """
    ordered = tuple(order)
    if coupling == "none":
        return Schedule(ordered, dict.fromkeys(ordered, ()), coupling)
    if coupling == "all":
        return Schedule(ordered, {portfolio_id: tuple(ordered[:position]) for position, portfolio_id in enumerate(ordered)}, coupling)
    return Schedule(ordered, _overlap_predecessors(ordered, tradable, unknown), coupling)


def _overlap_predecessors(order: tuple[PortfolioId, ...], tradable: Mapping[PortfolioId, Iterable[str]], unknown: frozenset[PortfolioId]) -> dict[PortfolioId, tuple[PortfolioId, ...]]:
    """Packed-bit incidence over a sorted security index; one AND per portfolio against every earlier row."""
    code = {security: index for index, security in enumerate(sorted({security for portfolio_id in order for security in tradable.get(portfolio_id, ())}))}
    incidence = np.zeros((len(order), max(len(code), 1)), dtype=np.bool_)
    for row, portfolio_id in enumerate(order):
        for security in tradable.get(portfolio_id, ()):
            incidence[row, code[security]] = True
    packed = np.packbits(incidence, axis=1)
    is_unknown = np.array([portfolio_id in unknown for portfolio_id in order], dtype=np.bool_)
    predecessors: dict[PortfolioId, tuple[PortfolioId, ...]] = {}
    for position, portfolio_id in enumerate(order):
        if is_unknown[position]:
            predecessors[portfolio_id] = order[:position]
            continue
        overlaps = np.bitwise_and(packed[:position], packed[position]).any(axis=1) | is_unknown[:position]
        predecessors[portfolio_id] = tuple(order[earlier] for earlier in np.flatnonzero(overlaps))
    return predecessors
