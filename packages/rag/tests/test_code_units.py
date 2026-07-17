from pathlib import Path

from agentic_doc_rag.parsers.code_units import split_code_into_units
from agentic_doc_rag.parsers.language_profiles import (
    GENERIC,
    GO,
    PYTHON,
    RUST,
    TYPESCRIPT,
    profile_for_path,
)


def test_profile_for_path_maps_known_extensions() -> None:
    assert profile_for_path(Path("lib.rs")).language == "rust"
    assert profile_for_path(Path("app.py")).language == "python"
    assert profile_for_path(Path("main.ts")).language == "typescript"
    assert profile_for_path(Path("main.go")).language == "go"
    assert profile_for_path(Path("notes.txt")).language == "generic"


def test_split_rust_functions_and_struct() -> None:
    source = """\
use std::io;

/// Docs for foo
fn foo() {
    println!("foo");
}

pub struct Bar {
    x: i32,
}

fn baz() {}
"""
    units = split_code_into_units(source, RUST)

    assert len(units) >= 3
    assert units[0].symbol is None
    assert "use std::io" in units[0].text
    symbols = [unit.symbol for unit in units if unit.symbol]
    assert "foo" in symbols
    assert "Bar" in symbols
    assert "baz" in symbols
    foo = next(unit for unit in units if unit.symbol == "foo")
    assert "/// Docs for foo" in foo.text
    assert foo.start_line < foo.end_line


def test_split_python_classes_and_functions() -> None:
    source = """\
import os

class Owner:
    def method(self):
        pass

def helper():
    return 1
"""
    units = split_code_into_units(source, PYTHON)

    assert units[0].symbol is None
    assert "import os" in units[0].text
    symbols = [unit.symbol for unit in units if unit.symbol]
    assert symbols == ["Owner", "helper"]
    # Nested methods are not top-level definitions.
    assert all("def method" not in (unit.symbol or "") for unit in units)
    owner = next(unit for unit in units if unit.symbol == "Owner")
    assert "def method" in owner.text


def test_split_typescript_functions() -> None:
    source = """\
export function greet(name: string) {
  return name;
}

const answer = 42;

export class Service {}
"""
    units = split_code_into_units(source, TYPESCRIPT)
    symbols = [unit.symbol for unit in units if unit.symbol]
    assert "greet" in symbols
    assert "answer" in symbols
    assert "Service" in symbols


def test_split_go_funcs() -> None:
    source = """\
package main

func main() {}

type Config struct {
	Port int
}
"""
    units = split_code_into_units(source, GO)
    symbols = [unit.symbol for unit in units if unit.symbol]
    assert "main" in symbols
    assert "Config" in symbols


def test_generic_blank_line_fallback() -> None:
    source = """\
first block
still first

second block
"""
    units = split_code_into_units(source, GENERIC)

    assert len(units) == 2
    assert "first block" in units[0].text
    assert "second block" in units[1].text
    assert units[0].symbol is None


def test_no_definitions_returns_single_unit() -> None:
    source = "const x = 1;\nconst y = 2;\n"
    units = split_code_into_units(source, RUST)

    assert len(units) == 1
    assert "const x" in units[0].text


def test_empty_text_returns_no_units() -> None:
    assert split_code_into_units("", RUST) == []
    assert split_code_into_units("   \n\n", PYTHON) == []
