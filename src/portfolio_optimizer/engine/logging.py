"""Structured logging: the identifiers that make an incident searchable travel in ``extra=``."""

import json
import logging
from typing import TextIO, override

CONTEXT_FIELDS: tuple[str, ...] = ("run_id", "portfolio_id", "stage")


class ContextFormatter(logging.Formatter):
    """Render the message followed by any known context fields as a JSON object."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = {name: getattr(record, name) for name in CONTEXT_FIELDS if hasattr(record, name)}
        return f"{base} {json.dumps(context, sort_keys=True, default=str)}" if context else base


def configure_logging(level: str, stream: TextIO) -> logging.Logger:
    """Install one handler on the package logger, replacing any previous configuration."""
    logger = logging.getLogger("portfolio_optimizer")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ContextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
