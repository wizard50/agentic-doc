"""Stdlib logging setup for the API process."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the API process.

    Safe to call multiple times (e.g. tests); only adjusts level if handlers
    already exist.
    """
    root = logging.getLogger()
    normalized = level.upper()
    if root.handlers:
        root.setLevel(normalized)
        return

    logging.basicConfig(
        level=normalized,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
