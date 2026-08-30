"""Tier 1: spans record what ran where, and the Chrome trace and the ASCII waterfall render the same recorded spans."""

import json
from pathlib import Path

from portfolio_optimizer.engine.timing import Span, SpanRecorder, render_timeline, sort_spans, write_trace


def span(name: str, start: float, duration: float = 1.0, portfolio_id: str | None = None, worker: str = "host:1") -> Span:
    return Span(name=name, portfolio_id=portfolio_id, worker=worker, started_at_s=start, duration_s=duration)


def test_recorder_times_blocks_and_stamps_its_process() -> None:
    recorder = SpanRecorder("P1")
    with recorder.span("build"), recorder.span("build:rules"):
        pass
    named = {recorded.name: recorded for recorded in recorder.spans}
    assert set(named) == {"build", "build:rules"}
    assert named["build"].duration_s >= named["build:rules"].duration_s, "the outer span contains the inner one"
    assert all(recorded.portfolio_id == "P1" and ":" in recorded.worker for recorded in recorder.spans)


def test_recorder_keeps_the_span_of_a_block_that_raises() -> None:
    recorder = SpanRecorder()
    try:
        with recorder.span("solve"):
            msg = "boom"
            raise RuntimeError(msg)
    except RuntimeError:
        pass
    assert [recorded.name for recorded in recorder.spans] == ["solve"], "a failed stage's time was still spent"


def test_sort_is_by_start_then_name_never_completion_order() -> None:
    spans = [span("solve", 2.0), span("build", 1.0), span("assembly", 1.0)]
    assert [recorded.name for recorded in sort_spans(spans)] == ["assembly", "build", "solve"]


def test_trace_is_chrome_json_with_a_process_per_worker_and_microsecond_offsets(tmp_path: Path) -> None:
    spans = [span("load", 10.0, 2.0), span("build", 12.0, 1.0, "P1", "hostA:1"), span("solve", 13.0, 1.5, "P1", "hostA:1"), span("build", 12.0, 1.0, "P2", "hostB:2")]
    body = json.loads(write_trace(spans, tmp_path).read_text())
    events = body["traceEvents"]
    assert {event["args"]["name"] for event in events if event["name"] == "process_name"} == {"host:1", "hostA:1", "hostB:2"}
    complete = [event for event in events if event["ph"] == "X"]
    assert {event["name"] for event in complete} == {"load", "build", "solve"}
    assert min(event["ts"] for event in complete) == 0.0, "timestamps count from the earliest span"
    build_p1 = next(event for event in complete if event["name"] == "build" and event["args"]["portfolio_id"] == "P1")
    assert build_p1["dur"] == 1e6, "durations are microseconds"


def test_timeline_draws_run_rows_and_portfolio_rows_within_the_limit() -> None:
    spans = [span("load", 0.0, 2.0), span("build", 2.0, 1.0, "P1"), span("build:rules", 2.0, 0.5, "P1")]
    text = render_timeline(spans, limit=5)
    totals, waterfall = text.split("waterfall")
    assert "build P1" in waterfall
    assert "build:rules" in totals, "sub-phases count in the totals"
    assert "build:rules" not in waterfall, "and are not drawn as rows"


def test_timeline_collapses_a_large_book_into_worker_occupancy_lanes() -> None:
    spans = [span("build", float(index), 1.0, f"P{index}", worker=f"host:{index % 2}") for index in range(10)]
    text = render_timeline(spans, limit=3)
    assert "occupancy lane" in text
    assert "host:0" in text
    assert "host:1" in text


def test_timeline_with_no_spans_says_so() -> None:
    assert render_timeline([]) == "no timing recorded\n"
