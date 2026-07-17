from collections.abc import Sequence
from pathlib import Path

from agentic_doc_rag.parsers.code import CodeParser
from agentic_doc_rag.parsers.markdown import MarkdownParser
from agentic_doc_rag.parsers.pdf import PdfParser
from agentic_doc_rag.parsers.protocols import DocumentParser


def default_parsers() -> list[DocumentParser]:
    return [MarkdownParser(), PdfParser(), CodeParser()]


def supported_extensions(parsers: Sequence[DocumentParser]) -> frozenset[str]:
    extensions: set[str] = set()
    for parser in parsers:
        extensions.update(parser.extensions)
    return frozenset(extensions)


def parser_for_path(path: Path, parsers: Sequence[DocumentParser]) -> DocumentParser | None:
    for parser in parsers:
        if parser.can_parse(path):
            return parser
    return None
