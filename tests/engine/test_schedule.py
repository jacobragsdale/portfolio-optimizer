"""Tier 1: the derived schedule — priority order with ties, overlap edges, unknown builds, the diagnostic line, and the shape summary."""

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.schedule import Schedule, dependency_graph, order_portfolios


def ids(*names: str) -> tuple[PortfolioId, ...]:
    return tuple(PortfolioId(name) for name in names)


BUYABLE: dict[PortfolioId, tuple[str, ...]] = {PortfolioId("P1"): ("A", "B"), PortfolioId("P2"): ("C",), PortfolioId("P3"): ("B", "D"), PortfolioId("P4"): ("D",)}
ORDER = ids("P1", "P2", "P3", "P4")


def test_order_is_ascending_key_then_portfolio_id() -> None:
    keys = {PortfolioId("b"): Decimal(1), PortfolioId("a"): Decimal(1), PortfolioId("c"): Decimal("-0.5")}
    assert order_portfolios(keys) == ids("c", "a", "b")


def test_overlap_edges_only_where_buyable_sets_intersect() -> None:
    schedule = dependency_graph(ORDER, BUYABLE, frozenset(), "overlap")
    assert dict(schedule.predecessors) == {"P1": (), "P2": (), "P3": ("P1",), "P4": ("P3",)}
    summary = schedule.summary()
    assert (summary.coupling, summary.portfolios, summary.edges, summary.components, summary.largest_component, summary.critical_path) == ("overlap", 4, 2, 2, 3, 3)
    assert schedule.heights() == {"P1": 3, "P2": 1, "P3": 2, "P4": 1}


def test_all_is_a_line_and_none_is_nothing() -> None:
    line = dependency_graph(ORDER, BUYABLE, frozenset(), "all")
    assert line.predecessors[PortfolioId("P4")] == ids("P1", "P2", "P3")
    assert (line.summary().edges, line.summary().components, line.summary().critical_path) == (6, 1, 4)
    free = dependency_graph(ORDER, BUYABLE, frozenset(), "none")
    assert (free.summary().edges, free.summary().components, free.summary().critical_path) == (0, 4, 1)


def test_an_unknown_build_overlaps_every_other_portfolio() -> None:
    schedule = dependency_graph(ORDER, BUYABLE, frozenset({PortfolioId("P2")}), "overlap")
    assert schedule.predecessors[PortfolioId("P2")] == ids("P1")
    assert schedule.predecessors[PortfolioId("P3")] == ids("P1", "P2")
    assert schedule.predecessors[PortfolioId("P4")] == ids("P2", "P3")


def test_a_portfolio_with_nothing_buyable_waits_for_nobody() -> None:
    schedule = dependency_graph(ORDER, {**BUYABLE, PortfolioId("P3"): ()}, frozenset(), "overlap")
    assert schedule.predecessors[PortfolioId("P3")] == ()
    assert schedule.predecessors[PortfolioId("P4")] == ()


def test_a_schedule_rejects_a_dependency_on_a_later_portfolio() -> None:
    with pytest.raises(ValueError, match="does not precede"):
        Schedule(ids("P1", "P2"), {PortfolioId("P1"): ids("P2"), PortfolioId("P2"): ()}, "overlap")
    with pytest.raises(ValueError, match="exactly the portfolios"):
        Schedule(ids("P1", "P2"), {PortfolioId("P1"): ()}, "overlap")


@given(sets=st.lists(st.frozensets(st.sampled_from("ABCDE")), min_size=1, max_size=6))
@settings(deadline=None, max_examples=100)
def test_every_edge_has_a_witness_security_and_every_overlap_has_an_edge(sets: list[frozenset[str]]) -> None:
    order = ids(*(f"P{index}" for index in range(len(sets))))
    schedule = dependency_graph(order, {portfolio_id: tuple(sorted(members)) for portfolio_id, members in zip(order, sets, strict=True)}, frozenset(), "overlap")
    for position, portfolio_id in enumerate(order):
        assert schedule.predecessors[portfolio_id] == tuple(order[earlier] for earlier in range(position) if sets[earlier] & sets[position])
    summary = schedule.summary()
    assert summary.edges == sum(len(earlier) for earlier in schedule.predecessors.values())
    assert 1 <= summary.critical_path <= len(sets)
