"""Property: over random buyable sets, every overlap between an earlier and a later portfolio is an edge, and every edge has a witness security."""

from hypothesis import given, settings
from hypothesis import strategies as st

from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.schedule import dependency_graph


def ids(*names: str) -> tuple[PortfolioId, ...]:
    return tuple(PortfolioId(name) for name in names)


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
