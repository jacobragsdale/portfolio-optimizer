"""JSON-driven, auditable portfolio-optimization engine built on pandas and cvxpy.

Quant developers extend the engine by writing ordinary functions in ``loaders.py``, ``assembly.py``,
``rules.py``, ``solve_order.py``, ``terms.py``, ``solvers.py``, and ``sinks.py`` and naming them in a
run config. The ``engine`` package is the part that rarely changes.
"""

__all__: list[str] = []
