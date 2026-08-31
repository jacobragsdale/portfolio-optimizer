"""Wall-clock spans over the work a run already delimits.

A span is a start instant, a duration, the worker process that ran it, and the stage name the code
already uses (``load``, ``dataset:<name>``, ``assembly``, ``cluster:provision``, ``build``, ``solve``,
``sink``, with sub-phases as ``build:rules`` or ``solve:verify``). Tasks record their own spans with a
:class:`SpanRecorder` and return them on their ``TaskOutput``; the runner records the client-side
stages and folds everything into the manifest's ``timing`` block, then writes the same spans beside
the manifest in the Chrome trace format (``trace.json``, opened in ``chrome://tracing`` or Perfetto),
which is where a run's timing is read.

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
from portfolio_optimizer.engine.files import write_atomically

TRACE_FILENAME = "trace.json"


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
    return write_atomically(directory / TRACE_FILENAME, json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}, separators=(",", ":")) + "\n")
