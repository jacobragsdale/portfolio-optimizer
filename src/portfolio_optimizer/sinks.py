"""Order sinks — yours to edit.

A sink is an ordinary function ``(orders: pd.DataFrame, io: IoContext, params: P) -> tuple[Artifact, ...]``
that publishes the run's orders somewhere — a file, a queue, a trading system — and returns what
it wrote so the manifest can record it.
"""
