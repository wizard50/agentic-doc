from pathlib import Path

from support.paths import SAMPLE_PYTHON_PATH, SAMPLE_RUST_PATH

from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.parsers import (
    CodeParser,
    default_parsers,
    parser_for_path,
    supported_extensions,
)


def test_supported_extensions_include_code_languages() -> None:
    extensions = supported_extensions(default_parsers())
    assert ".rs" in extensions
    assert ".py" in extensions
    assert ".ts" in extensions
    assert ".go" in extensions


def test_parser_for_path_returns_code_parser() -> None:
    parsers = default_parsers()
    assert isinstance(parser_for_path(Path("lib.rs"), parsers), CodeParser)
    assert isinstance(parser_for_path(Path("app.py"), parsers), CodeParser)


def test_code_parser_rust_fixture_metadata() -> None:
    parser = CodeParser()
    settings = IngestSettings(source_dir=SAMPLE_RUST_PATH.parent)

    chunks = parser.parse(SAMPLE_RUST_PATH, settings)

    assert chunks
    assert all(chunk.metadata["file_type"] == "code" for chunk in chunks)
    assert all(chunk.metadata["language"] == "rust" for chunk in chunks)
    symbols = {chunk.metadata.get("symbol") for chunk in chunks}
    assert "take_ownership" in symbols
    assert "Owner" in symbols
    assert "main" in symbols
    take = next(chunk for chunk in chunks if chunk.metadata.get("symbol") == "take_ownership")
    assert "/// Ownership helper" in take.text
    assert take.metadata["start_line"] < take.metadata["end_line"]
    assert "sample.rs::take_ownership" in take.metadata["section_path"]


def test_code_parser_python_fixture() -> None:
    chunks = CodeParser().parse(
        SAMPLE_PYTHON_PATH,
        IngestSettings(source_dir=SAMPLE_PYTHON_PATH.parent),
    )

    symbols = {chunk.metadata.get("symbol") for chunk in chunks}
    assert "Config" in symbols
    assert "load_config" in symbols
    assert any(chunk.metadata.get("symbol") is None for chunk in chunks)  # preamble/imports


def test_code_parser_windows_large_function(tmp_path: Path) -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    path = tmp_path / "big.rs"
    path.write_text(f"fn big()\n{{\n{body}\n}}\n", encoding="utf-8")

    chunks = CodeParser().parse(
        path,
        IngestSettings(source_dir=tmp_path, chunk_size=200, chunk_overlap=20),
    )

    big_chunks = [chunk for chunk in chunks if chunk.metadata.get("symbol") == "big"]
    assert len(big_chunks) > 1
    assert all(chunk.metadata["section_path"] == "big.rs::big" for chunk in big_chunks)
    assert all(chunk.metadata["start_line"] == big_chunks[0].metadata["start_line"] for chunk in big_chunks)
    assert len({chunk.id for chunk in big_chunks}) == len(big_chunks)


def test_code_parser_keeps_preamble_as_separate_unit() -> None:
    chunks = CodeParser().parse(
        SAMPLE_RUST_PATH,
        IngestSettings(source_dir=SAMPLE_RUST_PATH.parent),
    )

    preamble = next(chunk for chunk in chunks if chunk.metadata.get("symbol") is None)
    assert "use std::io" in preamble.text
    assert "fn " not in preamble.text.split("\n")[0]
