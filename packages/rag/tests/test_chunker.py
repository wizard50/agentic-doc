from pathlib import Path

from agentic_doc_rag.chunk.chunker import (
    chunk_markdown_dir,
    chunk_markdown_text,
    make_chunk_id,
    split_by_headers,
    split_with_overlap,
)


def test_split_by_headers_respects_code_fences() -> None:
    text = """## Real Section

```rust
# not a heading
## also not a heading
```

### Subsection
"""
    sections = split_by_headers(text, levels={2, 3})
    assert len(sections) == 2
    assert "## Real Section" in sections[0].text
    assert "# not a heading" in sections[0].text
    assert sections[1].text.startswith("### Subsection")


def test_split_by_headers_tracks_hierarchy() -> None:
    text = """# Chapter

## Section A

content a

### Sub A1

content a1
"""
    sections = split_by_headers(text, levels={2, 3})
    assert len(sections) == 3
    assert sections[1].headers["h2"] == "Section A"
    assert sections[2].headers["h3"] == "Sub A1"
    assert sections[2].section_path == "Chapter > Section A > Sub A1"


def test_split_with_overlap_produces_windows() -> None:
    text = "a" * 100
    parts = split_with_overlap(text, chunk_size=40, chunk_overlap=10)
    assert len(parts) == 3
    assert parts[0] == "a" * 40
    assert len(parts[1]) == 40
    assert parts[-1].endswith("a")


def test_make_chunk_id_is_stable_and_short() -> None:
    first = make_chunk_id("lib.rs", "lib.rs::main", 0)
    second = make_chunk_id("lib.rs", "lib.rs::main", 0)
    other = make_chunk_id("lib.rs", "lib.rs::main", 1)

    assert first == second
    assert first != other
    assert len(first) == 16



def test_chunk_markdown_text_adds_metadata_and_ids() -> None:
    text = """## Ownership

Rust uses ownership rules.

### Rules

Each value has one owner.
"""
    chunks = chunk_markdown_text(text, "ch04-01.md")
    assert len(chunks) == 2
    assert chunks[0].metadata["h2"] == "Ownership"
    assert chunks[0].metadata["source"] == "ch04-01.md"
    assert chunks[0].metadata["file_type"] == "markdown"
    assert chunks[0].text.startswith("## Ownership")
    assert chunks[1].text.startswith("### Rules")
    assert chunks[1].metadata["h3"] == "Rules"
    assert chunks[0].id != chunks[1].id


def test_chunk_markdown_dir_skip_files(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("## Keep\n\ncontent\n", encoding="utf-8")
    (tmp_path / "skip.md").write_text("## Skip\n\ncontent\n", encoding="utf-8")

    chunks = chunk_markdown_dir(tmp_path, skip_files={"skip.md"})
    sources = {chunk.metadata["source"] for chunk in chunks}

    assert len(sources) == 1
    assert str(tmp_path / "keep.md") in sources
