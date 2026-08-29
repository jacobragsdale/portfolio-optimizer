"""Dataset loaders — yours to edit.

A loader is an ordinary function ``(request: LoadRequest, params: P) -> pd.DataFrame`` named in
the run config. The ``constraints`` dataset's loader returns ``dict[str, dict[str, object]]``
keyed by portfolio id instead. Loaders are the only place file, database, or network access
belongs; everything downstream is pure.
"""
