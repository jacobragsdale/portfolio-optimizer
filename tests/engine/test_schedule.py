"""Tier 1: the derived schedule — priority order with ties, overlap edges, unknown builds, and the shape summary."""

from collections.abc import Iterable, Mapping
from decimal import Decimal

import pytest

from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.schedule import OverlapIndex, Schedule, order_portfolios


def ids(*names: str) -> tuple[PortfolioId, ...]:
    return tuple(PortfolioId(name) for name in names)


def overlap_graph(
    order: tuple[PortfolioId, ...], tradable: Mapping[PortfolioId, Iterable[str]], consumes: Mapping[PortfolioId, Iterable[str]] | None = None, unknown: frozenset[PortfolioId] = frozenset()
) -> Schedule:
    """The graph an ``overlap`` run derives, driving the index exactly as ``runner._stream_solves`` does."""
    index = OverlapIndex(len(order), (security for members in tradable.values() for security in members))
    rows = [index.add(tradable.get(portfolio_id, ()), (consumes or {}).get(portfolio_id, tradable.get(portfolio_id, ())), unknown=portfolio_id in unknown) for portfolio_id in order]
    return Schedule(order, {portfolio_id: tuple(order[position] for position in row) for portfolio_id, row in zip(order, rows, strict=True)}, "overlap")


BUYABLE: dict[PortfolioId, tuple[str, ...]] = {PortfolioId("P1"): ("A", "B"), PortfolioId("P2"): ("C",), PortfolioId("P3"): ("B", "D"), PortfolioId("P4"): ("D",)}
ORDER = ids("P1", "P2", "P3", "P4")


def test_order_is_ascending_key_then_portfolio_id() -> None:
    keys = {PortfolioId("b"): Decimal(1), PortfolioId("a"): Decimal(1), PortfolioId("c"): Decimal("-0.5")}
    assert order_portfolios(keys) == ids("c", "a", "b")


def test_overlap_edges_only_where_buyable_sets_intersect() -> None:
    schedule = overlap_graph(ORDER, BUYABLE)
    assert dict(schedule.predecessors) == {"P1": (), "P2": (), "P3": ("P1",), "P4": ("P3",)}
    summary = schedule.summary()
    assert (summary.coupling, summary.portfolios, summary.edges, summary.components, summary.largest_component, summary.critical_path) == ("overlap", 4, 2, 2, 3, 3)


def test_the_summary_of_a_line_and_of_an_uncoupled_book() -> None:
    line = Schedule(ORDER, {portfolio_id: ORDER[:position] for position, portfolio_id in enumerate(ORDER)}, "all")
    assert (line.summary().edges, line.summary().components, line.summary().critical_path) == (6, 1, 4)
    free = Schedule(ORDER, dict.fromkeys(ORDER, ()), "none")
    assert (free.summary().edges, free.summary().components, free.summary().critical_path) == (0, 4, 1)


def test_an_unknown_build_overlaps_every_other_portfolio() -> None:
    schedule = overlap_graph(ORDER, BUYABLE, unknown=frozenset({PortfolioId("P2")}))
    assert schedule.predecessors[PortfolioId("P2")] == ids("P1")
    assert schedule.predecessors[PortfolioId("P3")] == ids("P1", "P2")
    assert schedule.predecessors[PortfolioId("P4")] == ids("P2", "P3")


def test_a_portfolio_with_nothing_buyable_waits_for_nobody() -> None:
    schedule = overlap_graph(ORDER, {**BUYABLE, PortfolioId("P3"): ()})
    assert schedule.predecessors[PortfolioId("P3")] == ()
    assert schedule.predecessors[PortfolioId("P4")] == ()


def test_a_schedule_rejects_a_dependency_on_a_later_portfolio() -> None:
    with pytest.raises(ValueError, match="does not precede"):
        Schedule(ids("P1", "P2"), {PortfolioId("P1"): ids("P2"), PortfolioId("P2"): ()}, "overlap")
    with pytest.raises(ValueError, match="exactly the portfolios"):
        Schedule(ids("P1", "P2"), {PortfolioId("P1"): ()}, "overlap")


def test_the_overlap_index_answers_a_row_without_the_rows_behind_it() -> None:
    index = OverlapIndex(4, ("A", "B", "C", "D"))
    assert index.add(("A", "B"), ("A", "B")) == ()
    assert index.add(("C",), ("C",)) == ()
    assert index.add(("B", "D"), ("B", "D")) == (0,)
    assert index.add(("D",), ("D",)) == (2,)


def test_the_overlap_index_codes_a_security_its_seed_never_named() -> None:
    seeded = OverlapIndex(3, ("A",))
    assert seeded.add(("A", *(f"S{index}" for index in range(20))), ("A",)) == ()
    assert seeded.add(("S19",), ("S19",)) == (0,), "a security past the seeded width is coded on arrival"
    assert seeded.add(("A",), ("A",)) == (0,)


def test_the_edge_test_is_directional_produce_against_consume() -> None:
    """P3 can trade B and D but its chain readers see only D: it waits for the D-producer alone, while later portfolios still test against everything P3 can trade."""
    consumes = {PortfolioId("P1"): ("A", "B"), PortfolioId("P2"): ("C",), PortfolioId("P3"): ("D",), PortfolioId("P4"): ("D",)}
    schedule = overlap_graph(ORDER, BUYABLE, consumes)
    assert dict(schedule.predecessors) == {"P1": (), "P2": (), "P3": (), "P4": ("P3",)}, "P3 no longer waits for P1: B reaches only portfolios whose readers consume B"


def test_an_empty_consume_set_waits_for_nobody_even_behind_a_failed_build() -> None:
    """A portfolio that provably reads no chain cannot be affected by anyone — an unknown build included — but its own trades still reach later consumers."""
    index = OverlapIndex(3, ("A",))
    assert index.add(("A",), ("A",), unknown=True) == ()
    assert index.add(("A",), ()) == (), "nothing consumed: even the unknown producer casts no edge here"
    assert index.add(("A",), ("A",)) == (0, 1), "a real consumer waits for the unknown build and for the reader-free producer alike"
