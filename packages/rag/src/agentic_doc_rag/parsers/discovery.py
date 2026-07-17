from pathlib import Path


def discover_files(
    source_dir: Path,
    *,
    skip_files: frozenset[str],
    extensions: frozenset[str],
) -> list[Path]:
    """Return sorted source files under root matching extensions, excluding skip names."""
    allowed = {ext.casefold() if ext.startswith(".") else f".{ext.casefold()}" for ext in extensions}
    files: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in skip_files:
            continue
        if path.suffix.casefold() not in allowed:
            continue
        files.append(path)
    return files
