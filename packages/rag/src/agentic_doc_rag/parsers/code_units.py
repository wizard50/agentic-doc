from __future__ import annotations

from dataclasses import dataclass

from agentic_doc_rag.parsers.language_profiles import LanguageProfile


@dataclass(frozen=True, slots=True)
class CodeUnit:
    """A contiguous region of source text after structure-aware splitting."""

    text: str
    start_line: int
    end_line: int
    symbol: str | None = None


def split_code_into_units(text: str, profile: LanguageProfile) -> list[CodeUnit]:
    """Split source text into structural units using a language profile.

    Definition-based profiles cut on top-level declaration lines and attach
    contiguous leading line-comments. Generic profiles split on blank lines.
    """
    if not text.strip():
        return []

    lines = text.splitlines()
    if profile.use_blank_line_units or not profile.definition_patterns:
        return _split_on_blank_lines(lines)

    starts = _definition_starts(lines, profile)
    if not starts:
        return _single_unit(lines)

    units: list[CodeUnit] = []
    first_start, _symbol = starts[0]
    if first_start > 0:
        preamble = _unit_from_range(lines, 0, first_start - 1, symbol=None)
        if preamble is not None:
            units.append(preamble)

    for index, (start, symbol) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines) - 1
        unit = _unit_from_range(lines, start, end, symbol=symbol)
        if unit is not None:
            units.append(unit)

    return units


def _definition_starts(
    lines: list[str],
    profile: LanguageProfile,
) -> list[tuple[int, str | None]]:
    """Return (unit_start_line_index, symbol) pairs for each top-level definition."""
    starts: list[tuple[int, str | None]] = []
    for line_index, line in enumerate(lines):
        matched, symbol = _match_definition(line, profile)
        if not matched:
            continue
        content_start = _leading_comment_start(lines, line_index, profile)
        starts.append((content_start, symbol))
    return starts


def _match_definition(line: str, profile: LanguageProfile) -> tuple[bool, str | None]:
    code = _top_level_code(line, profile.max_indent)
    if code is None:
        return False, None
    for pattern in profile.definition_patterns:
        match = pattern.match(code)
        if match is None:
            continue
        if "name" in match.re.groupindex:
            name = match.group("name")
            if name:
                return True, name
        token = code.split(None, 1)[0].rstrip("{")
        return True, token or None
    return False, None


def _top_level_code(line: str, max_indent: int) -> str | None:
    if not line.strip():
        return None
    indent = len(line) - len(line.lstrip(" \t"))
    if indent > max_indent:
        return None
    return line.lstrip(" \t")


def _leading_comment_start(
    lines: list[str],
    definition_index: int,
    profile: LanguageProfile,
) -> int:
    start = definition_index
    index = definition_index - 1
    while index >= 0 and _is_line_comment(lines[index], profile):
        start = index
        index -= 1
    return start


def _is_line_comment(line: str, profile: LanguageProfile) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(stripped.startswith(prefix) for prefix in profile.line_comment_prefixes)


def _split_on_blank_lines(lines: list[str]) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    block_start: int | None = None

    for index, line in enumerate(lines):
        if line.strip():
            if block_start is None:
                block_start = index
            continue
        if block_start is not None:
            unit = _unit_from_range(lines, block_start, index - 1, symbol=None)
            if unit is not None:
                units.append(unit)
            block_start = None

    if block_start is not None:
        unit = _unit_from_range(lines, block_start, len(lines) - 1, symbol=None)
        if unit is not None:
            units.append(unit)

    return units if units else _single_unit(lines)


def _single_unit(lines: list[str]) -> list[CodeUnit]:
    unit = _unit_from_range(lines, 0, len(lines) - 1, symbol=None)
    return [unit] if unit is not None else []


def _unit_from_range(
    lines: list[str],
    start: int,
    end: int,
    *,
    symbol: str | None,
) -> CodeUnit | None:
    if start > end or start < 0 or end >= len(lines):
        return None
    text = "\n".join(lines[start : end + 1]).strip("\n")
    if not text.strip():
        return None
    return CodeUnit(
        text=text,
        start_line=start + 1,
        end_line=end + 1,
        symbol=symbol,
    )
