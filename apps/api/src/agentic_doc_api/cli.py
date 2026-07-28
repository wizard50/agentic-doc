"""CLI entry: ``uv run api`` starts uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from agentic_doc_api.config import get_api_settings
from agentic_doc_core.config import Environment


def main(argv: list[str] | None = None) -> None:
    """Launch the API with uvicorn using process settings."""
    settings = get_api_settings()
    parser = argparse.ArgumentParser(description="Agentic Doc API server")
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"Bind host (default: {settings.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Bind port (default: {settings.port})",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=settings.environment == Environment.DEV,
        help="Enable auto-reload (default: on in dev, off in prod)",
    )
    args = parser.parse_args(argv)

    uvicorn.run(
        "agentic_doc_api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
