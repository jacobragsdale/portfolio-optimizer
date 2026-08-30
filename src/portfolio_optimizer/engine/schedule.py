"""Derive the run's schedule: the order portfolios solve in, and which ones each waits for.

Pure functions over what the builds reported; nothing here knows about Dask. Portfolios couple
across a run through one side only, and the edge test is directional: portfolio *j* depends on every
higher-priority *i* whose *tradable* set (what *i* may trade there — the produce side) intersects
*j*'s *consume* set — the securities *j*'s own chain readers can see, which its build derives from
its typed constraints and which is the whole tradable set when anything opaque might read the chain.
A portfolio whose consume set is empty reads nothing and waits for no one. The graph is never
transitively reduced: a solve folds its *direct* predecessors' own contributions, so every
overlapping earlier portfolio must stay a direct dependency.

Every predecessor is earlier in the order, so the graph can be grown a portfolio at a time:
:class:`OverlapIndex` takes one portfolio's tradable set and answers which earlier ones it overlaps,
without knowing the portfolios still to come. That is what lets the runner submit a solve while the
tail of the book is still building; :func:`dependency_graph` is the same index driven over a book
whose builds have all reported.
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


def dependency_graph(
    order: Sequence[PortfolioId], tradable: Mapping[PortfolioId, Iterable[str]], consumes: Mapping[PortfolioId, Iterable[str]], unknown: frozenset[PortfolioId], coupling: Coupling
) -> Schedule:
    """Which higher-priority portfolios each portfolio waits for.

    ``tradable`` is each built portfolio's tradable securities — what its trades can reach others
    through; ``consumes`` is each portfolio's consume set — what its own chain readers can see, at
    most its tradable set and absent (or empty) when nothing it runs reads the chain. ``unknown``
    names portfolios whose build failed: both sets are unknown, so they are treated as overlapping
    every other portfolio. Under ``all`` every earlier portfolio is a predecessor; under ``none``
    nothing is, and both mappings are ignored.
    """
    ordered = tuple(order)
    if coupling == "none":
        return Schedule(ordered, dict.fromkeys(ordered, ()), coupling)
    if coupling == "all":
        return Schedule(ordered, {portfolio_id: tuple(ordered[:position]) for position, portfolio_id in enumerate(ordered)}, coupling)
    return Schedule(ordered, _overlap_predecessors(ordered, tradable, consumes, unknown), coupling)


def _overlap_predecessors(
    order: tuple[PortfolioId, ...], tradable: Mapping[PortfolioId, Iterable[str]], consumes: Mapping[PortfolioId, Iterable[str]], unknown: frozenset[PortfolioId]
) -> dict[PortfolioId, tuple[PortfolioId, ...]]:
    """One :class:`OverlapIndex` driven over the whole order, seeded with every security it will see."""
    index = OverlapIndex(len(order), (security for portfolio_id in order for security in tradable.get(portfolio_id, ())))
    predecessors: dict[PortfolioId, tuple[PortfolioId, ...]] = {}
    for portfolio_id in order:
        produced = tuple(tradable.get(portfolio_id, ()))
        earlier = index.add(produced, tuple(consumes[portfolio_id]) if portfolio_id in consumes else produced, unknown=portfolio_id in unknown)
        predecessors[portfolio_id] = tuple(order[position] for position in earlier)
    return predecessors


class OverlapIndex:
    """Packed-bit incidence over a security index, one row per portfolio in solve order.

    Rows are added as their builds report, and each add answers which earlier portfolios' *tradable*
    sets the newcomer's *consume* set intersects — so a caller can place a portfolio in the graph
    without waiting for the portfolios behind it. The stored row is always the tradable set: that is
    what later consumers test against, whatever this portfolio itself consumes. ``securities`` seeds
    the index — the assembled universe covers every tradable set a rule has not added to — and a
    security outside it is coded on arrival, which costs a repack only when it widens the row.
    """

    def __init__(self, portfolios: int, securities: Iterable[str] = ()) -> None:
        self._code = {security: position for position, security in enumerate(sorted(set(securities)))}
        self._rows: list[tuple[str, ...]] = []
        self._unknown = np.zeros(portfolios, dtype=np.bool_)
        self._packed = np.zeros((portfolios, self._width()), dtype=np.uint8)

    def add(self, tradable: Iterable[str], consumes: Iterable[str], *, unknown: bool = False) -> tuple[int, ...]:
        """Append a portfolio's tradable row and return the positions of the earlier rows its ``consumes`` set overlaps.

        ``consumes`` is what this portfolio's chain readers can see — at most its tradable set, and
        empty when nothing it runs reads the chain, in which case it waits for nobody, failed builds
        included: a portfolio that provably reads nothing cannot be affected by anyone. ``unknown`` is
        a portfolio whose build failed or was never read: both of its sets are unknown, so it overlaps
        every row on both sides.
        """
        members = tuple(tradable)
        consumed = tuple(consumes)
        position = len(self._rows)
        self._rows.append(members)
        self._unknown[position] = unknown
        width = self._width()
        for security in (*members, *consumed):
            if security not in self._code:
                self._code[security] = len(self._code)
        if self._width() != width:
            self._repack()
        else:
            self._packed[position] = self._pack(members)
        if unknown:
            return tuple(range(position))
        if not consumed:
            return ()
        overlaps = np.bitwise_and(self._packed[:position], self._pack(consumed)).any(axis=1) | self._unknown[:position]
        return tuple(int(earlier) for earlier in np.flatnonzero(overlaps))

    def _width(self) -> int:
        """Bytes a packed row occupies; at least one, so an index over no securities is still a matrix."""
        return max(-(-len(self._code) // 8), 1)

    def _pack(self, members: tuple[str, ...]) -> np.ndarray:
        bits = np.zeros(max(len(self._code), 1), dtype=np.bool_)
        if members:
            bits[[self._code[security] for security in members]] = True
        return np.packbits(bits)

    def _repack(self) -> None:
        """Re-pack every row at the current width; only a security the seed did not name gets here."""
        packed = np.zeros((len(self._unknown), self._width()), dtype=np.uint8)
        for position, members in enumerate(self._rows):
            packed[position] = self._pack(members)
        self._packed = packed
