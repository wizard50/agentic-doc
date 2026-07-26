"""CLI entry for Doc Agent (Streamlit UI)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agentic_doc_studio.workspace import require_workspace_root


def main() -> None:
    """Launch the Streamlit Doc Agent UI from the workspace root."""
    require_workspace_root("studio")
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
