"""Wall-clock spans over the work a run already delimits, and the two readers of them.

A span is a start instant, a duration, the worker process that ran it, and the stage name the code
already uses (``load``, ``dataset:<name>``, ``assembly``, ``cluster:provision``, ``build``, ``solve``,
``sink``, with sub-phases as ``build:rules`` or ``solve:verify``). Tasks record their own spans with a
:class:`SpanRecorder` and return them on their ``TaskOutput``; the runner records the client-side
stages and folds everything into the manifest's ``timing`` block, then writes the same spans beside
the manifest in the Chrome trace format (``trace.json``, opened in ``chrome://tracing`` or Perfetto).
The ``timeline`` CLI subcommand renders them as an ASCII waterfall with per-stage totals. Both readers
consume the recorded spans; neither is a second source of truth.

Two rules shape everything here. **Timing never touches identity**: spans live in the manifest but are
not part of the config hash and are never compared by ``diff-manifests``. **It is timing, not
profiling**: a ``perf_counter`` pair around work the engine already delimits, a few dozen spans per
portfolio, nothing that changes how the work runs. Instants are wall clock (``time.time``) so spans
from different hosts land on one axis; clock skew between hosts is visible in the picture rather than
corrected, which is the honest reading of a distributed run.
"""

import json
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from portfolio_optimizer.domain.types import StrictModel
from portfolio_optimizer.engine.environment import host_name

TRACE_FILENAME = "trace.json"

BLOCKS = " ▁▂▃▄▅▆▇█"
"""Occupancy glyphs for the per-worker lanes, blank through full."""


class Span(StrictModel):
    """One timed stage: what ran, for which portfolio, where, and when.

    ``name`` is the stage, with sub-phases spelled ``stage:phase``; ``portfolio_id`` is ``None`` for a
    run-scoped stage. ``worker`` is ``host:pid``, so two worker processes on one machine are two lanes.
    ``started_at_s`` is Unix wall-clock seconds; ``duration_s`` is measured with ``perf_counter``.
    """

    name: str
    portfolio_id: str | None = None
    worker: str
    started_at_s: float
    duration_s: float

    @property
    def stage(self) -> str:
        """The top-level stage name: ``build`` for ``build:rules``."""
        return self.name.partition(":")[0]

    @property
    def is_phase(self) -> bool:
        """True for a sub-phase of a stage, which totals count but lanes do not draw."""
        return ":" in self.name


class SpanRecorder:
    """Collects the spans of one process, stamped with its ``host:pid``.

    Time a block with :meth:`span`; record a span measured elsewhere — a dataset audit's
    ``started_s``/``load_time_s`` pair — with :meth:`add`.
    """

    def __init__(self, portfolio_id: str | None = None) -> None:
        self.portfolio_id = portfolio_id
        self.worker = f"{host_name()}:{os.getpid()}"
        self._spans: list[Span] = []

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time the block and record it, even when it raises — a failed stage's time is still spent."""
        started_at = time.time()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, started_at_s=started_at, duration_s=time.perf_counter() - started)

    def add(self, name: str, *, started_at_s: float, duration_s: float, portfolio_id: str | None = None) -> None:
        """Record one span measured by the caller; ``portfolio_id`` defaults to the recorder's."""
        self._spans.append(Span(name=name, portfolio_id=portfolio_id if portfolio_id is not None else self.portfolio_id, worker=self.worker, started_at_s=started_at_s, duration_s=duration_s))

    @property
    def spans(self) -> tuple[Span, ...]:
        """Everything recorded so far, in recording order."""
        return tuple(self._spans)


def sort_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
    """The canonical order the manifest records: by start, then name, then portfolio — never by completion order."""
    return tuple(sorted(spans, key=lambda span: (span.started_at_s, span.name, span.portfolio_id or "")))


def write_trace(spans: Sequence[Span], directory: Path) -> Path:
    """Write the spans as a Chrome trace (``trace.json``) beside the manifest, atomically.

    A row (``pid``) per worker process and a lane (``tid``) per portfolio within it, named through
    metadata events, so the file opens in ``chrome://tracing`` or Perfetto as a picture of the run.
    Timestamps are microseconds from the earliest span.
    """
    origin = min((span.started_at_s for span in spans), default=0.0)
    workers: dict[str, int] = {}
    for span in spans:
        workers.setdefault(span.worker, len(workers) + 1)
    lanes: dict[tuple[str, str], int] = {}
    events: list[dict[str, object]] = [{"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": worker}} for worker, pid in workers.items()]
    for span in spans:
        lane_key = (span.worker, span.portfolio_id or "")
        if lane_key not in lanes:
            lanes[lane_key] = sum(1 for existing in lanes if existing[0] == span.worker)
            events.append({"name": "thread_name", "ph": "M", "pid": workers[span.worker], "tid": lanes[lane_key], "args": {"name": span.portfolio_id or "run"}})
        events.append(
            {
                "name": span.name,
                "cat": span.stage,
                "ph": "X",
                "ts": round((span.started_at_s - origin) * 1e6, 1),
                "dur": round(span.duration_s * 1e6, 1),
                "pid": workers[span.worker],
                "tid": lanes[lane_key],
                "args": {"portfolio_id": span.portfolio_id},
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / TRACE_FILENAME
    temp = target.with_name(f".{TRACE_FILENAME}.tmp")
    try:
        temp.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}, separators=(",", ":")) + "\n")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def render_timeline(spans: Sequence[Span], *, limit: int = 40, width: int = 60) -> str:
    """The recorded spans as text: per-stage totals, then a waterfall.

    Run-scoped spans are always drawn one row each. Per-portfolio stages are drawn one row per span up
    to ``limit`` portfolios; past that each worker becomes one occupancy lane — how busy it was across
    the run — because a thousand rows tell a terminal nothing. Sub-phases (``build:rules``) appear in
    the totals only.
    """
    if not spans:
        return "no timing recorded\n"
    ordered = sort_spans(spans)
    origin = min(span.started_at_s for span in ordered)
    end = max(span.started_at_s + span.duration_s for span in ordered)
    wall = max(end - origin, 1e-9)
    lines = [f"wall clock {wall:.2f}s across {len({span.worker for span in ordered})} process(es); {len(ordered)} span(s)", "", _totals_table(ordered), "", f"waterfall (0s to {wall:.2f}s):"]
    run_scoped = [span for span in ordered if span.portfolio_id is None and not span.is_phase]
    lines.extend(_bar(span, span.name, origin, wall, width) for span in run_scoped)
    per_portfolio = [span for span in ordered if span.portfolio_id is not None and not span.is_phase]
    portfolios = {span.portfolio_id for span in per_portfolio}
    if len(portfolios) <= limit:
        lines.extend(_bar(span, f"{span.name} {span.portfolio_id}", origin, wall, width) for span in per_portfolio)
    elif per_portfolio:
        lines.append(f"{len(portfolios)} portfolios exceed --limit {limit}; one occupancy lane per worker (build+solve):")
        lines.extend(_occupancy_lanes(per_portfolio, origin, wall, width))
    return "\n".join(lines) + "\n"


def _totals_table(spans: Sequence[Span]) -> str:
    totals: dict[str, tuple[int, float, float]] = {}
    for span in spans:
        count, total, worst = totals.get(span.name, (0, 0.0, 0.0))
        totals[span.name] = (count + 1, total + span.duration_s, max(worst, span.duration_s))
    rows = [f"  {name:<24} {count:>6} {total:>10.2f}s {worst:>9.3f}s" for name, (count, total, worst) in sorted(totals.items())]
    return "\n".join([f"  {'stage':<24} {'count':>6} {'total':>11} {'max':>10}", *rows])


def _bar(span: Span, label: str, origin: float, wall: float, width: int) -> str:
    start = round((span.started_at_s - origin) / wall * width)
    length = max(round(span.duration_s / wall * width), 1)
    start = min(start, width - 1)
    length = min(length, width - start)
    return f"  {label:<28} |{'.' * start}{'#' * length}{'.' * (width - start - length)}| {span.duration_s:.2f}s"


def _occupancy_lanes(spans: Sequence[Span], origin: float, wall: float, width: int) -> list[str]:
    by_worker: dict[str, list[Span]] = {}
    for span in spans:
        by_worker.setdefault(span.worker, []).append(span)
    lanes = []
    for worker in sorted(by_worker):
        slices = [0.0] * width
        for span in by_worker[worker]:
            first, last = (span.started_at_s - origin) / wall * width, (span.started_at_s + span.duration_s - origin) / wall * width
            for index in range(max(int(first), 0), min(int(last) + 1, width)):
                slices[index] += max(min(last, index + 1) - max(first, index), 0.0)
        busy = sum(span.duration_s for span in by_worker[worker])
        lanes.append(f"  {worker:<28} |{''.join(BLOCKS[min(int(fraction * (len(BLOCKS) - 1) + 0.999), len(BLOCKS) - 1)] for fraction in slices)}| busy {busy:.1f}s ({busy / wall:.0%})")
    return lanes
