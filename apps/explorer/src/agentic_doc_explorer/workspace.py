import sys
from pathlib import Path


def require_workspace_root(command: str) -> None:
    cwd = Path.cwd()
    if (cwd / "packages").is_dir() and (cwd / "apps").is_dir():
        return
    sys.exit(
        f"{command} must be run from the workspace root (agentic-doc/), not a subfolder.\n\n"
        "  cd path/to/agentic-doc\n"
        f"  uv run explorer{'' if command == 'explorer' else ' ' + command}\n"
    )
