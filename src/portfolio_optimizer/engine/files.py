"""Writing a run's output files, so a crash never leaves a half-written one behind."""

from collections.abc import Callable
from pathlib import Path


def write_atomically(target: Path, content: str | Callable[[Path], None]) -> Path:
    """Write to a sibling temp file and rename, creating ``target``'s directory; return ``target``.

    ``content`` is the text to write, or a writer that takes the path — what a Parquet or CSV
    serializer needs. The temp file is removed whatever happens, so a failed write leaves nothing.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        if isinstance(content, str):
            temp.write_text(content)
        else:
            content(temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target
