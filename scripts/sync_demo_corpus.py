#!/usr/bin/env python
"""Vendor the Rust book demo corpus from a local upstream clone.

Copies license files and ``src/`` from ``data/download/rust-book`` (full git
clone, gitignored) into ``corpora/rust-book`` (intended for git + Streamlit).

Workflow:
  1. python scripts/download.py          # clone or pull upstream
  2. python scripts/sync_demo_corpus.py  # refresh corpora/rust-book from that clone
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("data/download/rust-book")
DEFAULT_DEST = Path("corpora/rust-book")
LICENSE_NAMES = ("LICENSE-MIT", "LICENSE-APACHE", "COPYRIGHT")


def _upstream_revision(source: Path) -> str | None:
    if not (source / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def sync_demo_corpus(source: Path, dest: Path) -> None:
    src_dir = source / "src"
    if not src_dir.is_dir():
        msg = (
            f"Missing {src_dir}. Clone or update the book first:\n"
            "  python scripts/download.py"
        )
        raise FileNotFoundError(msg)

    dest.mkdir(parents=True, exist_ok=True)

    for name in LICENSE_NAMES:
        license_path = source / name
        if license_path.is_file():
            shutil.copy2(license_path, dest / name)
        else:
            print(f"warning: {license_path} not found; skipping", file=sys.stderr)

    dest_src = dest / "src"
    if dest_src.exists():
        shutil.rmtree(dest_src)
    shutil.copytree(src_dir, dest_src)

    revision = _upstream_revision(source)
    pin_path = dest / "UPSTREAM"
    lines = [
        "Upstream: https://github.com/rust-lang/book",
        "Licenses: MIT OR Apache-2.0 (see LICENSE-* and COPYRIGHT in this directory)",
        f"Synced from: {source.resolve()}",
    ]
    if revision:
        lines.append(f"Git revision: {revision}")
    pin_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_count = len(list(dest_src.glob("**/*.md")))
    print(f"Synced demo corpus → {dest}")
    print(f"  Markdown files: {md_count}")
    if revision:
        print(f"  Upstream HEAD:  {revision}")
    print("  Demo ingest path: corpora/rust-book/src")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Full book clone (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Vendored demo corpus (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    try:
        sync_demo_corpus(args.source, args.dest)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
