from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Light, regex-based profile for structure-aware code splitting."""

    language: str
    extensions: frozenset[str]
    definition_patterns: tuple[re.Pattern[str], ...]
    line_comment_prefixes: tuple[str, ...] = ()
    max_indent: int = 0
    use_blank_line_units: bool = False


def _patterns(*raw: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in raw)


RUST = LanguageProfile(
    language="rust",
    extensions=frozenset({".rs"}),
    line_comment_prefixes=("//",),
    definition_patterns=_patterns(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?P<name>\w+)",
        r"^(?:pub(?:\([^)]*\))?\s+)?struct\s+(?P<name>\w+)",
        r"^(?:pub(?:\([^)]*\))?\s+)?enum\s+(?P<name>\w+)",
        r"^(?:pub(?:\([^)]*\))?\s+)?trait\s+(?P<name>\w+)",
        r"^(?:pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>\w+)",
        r"^impl\b",
    ),
)

PYTHON = LanguageProfile(
    language="python",
    extensions=frozenset({".py"}),
    line_comment_prefixes=("#",),
    definition_patterns=_patterns(
        r"^(?:async\s+)?def\s+(?P<name>\w+)",
        r"^class\s+(?P<name>\w+)",
    ),
)

TYPESCRIPT = LanguageProfile(
    language="typescript",
    extensions=frozenset({".ts", ".tsx", ".js", ".jsx"}),
    line_comment_prefixes=("//",),
    definition_patterns=_patterns(
        r"^(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)",
        r"^(?:export\s+)?class\s+(?P<name>\w+)",
        r"^(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=",
    ),
)

GO = LanguageProfile(
    language="go",
    extensions=frozenset({".go"}),
    line_comment_prefixes=("//",),
    definition_patterns=_patterns(
        r"^func\s+(?:\([^)]+\)\s+)?(?P<name>\w+)",
        r"^type\s+(?P<name>\w+)\s+(?:struct|interface)\b",
    ),
)

GENERIC = LanguageProfile(
    language="generic",
    extensions=frozenset(),
    definition_patterns=(),
    use_blank_line_units=True,
)

PROFILES: tuple[LanguageProfile, ...] = (RUST, PYTHON, TYPESCRIPT, GO)

_EXTENSION_TO_PROFILE: dict[str, LanguageProfile] = {
    extension: profile
    for profile in PROFILES
    for extension in profile.extensions
}


def profile_for_path(path: Path) -> LanguageProfile:
    """Return a language profile for the path suffix, or the generic blank-line profile."""
    return _EXTENSION_TO_PROFILE.get(path.suffix.casefold(), GENERIC)


def profile_for_language(language: str) -> LanguageProfile | None:
    for profile in PROFILES:
        if profile.language == language:
            return profile
    if language == GENERIC.language:
        return GENERIC
    return None
