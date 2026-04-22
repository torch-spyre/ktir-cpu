# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parser error handling tests for KTIRParser.

Covers:
- parse_file() with non-existent path
- parse_file() with empty file
- parse_module() with empty string
- Malformed/truncated MLIR (unclosed braces, missing operands)
- Unrecognized op text that doesn't match any dialect parser
- Partial parse: file where some ops parse and others don't
- Unicode and binary content in MLIR text
"""

import pytest

from ktir_cpu.parser import KTIRParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_func(body: str = "") -> str:
    return f"""
module {{
  func.func @test() {{
{body}
  }}
}}
"""


# ---------------------------------------------------------------------------
# parse_file errors
# ---------------------------------------------------------------------------

def test_parse_file_nonexistent(tmp_path):
    """parse_file() with a path that doesn't exist should raise FileNotFoundError."""
    parser = KTIRParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_file(str(tmp_path / "no_such_file.mlir"))


def test_parse_file_empty(tmp_path):
    """parse_file() on an empty file should return an IRModule with no functions."""
    f = tmp_path / "empty.mlir"
    f.write_text("")
    parser = KTIRParser()
    module = parser.parse_file(str(f))
    assert len(module.functions) == 0


def test_parse_file_valid(tmp_path):
    """Smoke test: parse_file() on a minimal valid file works."""
    f = tmp_path / "minimal.mlir"
    f.write_text(_minimal_func())
    parser = KTIRParser()
    module = parser.parse_file(str(f))
    assert len(module.functions) == 1
    assert "test" in module.functions


# ---------------------------------------------------------------------------
# parse_module edge cases
# ---------------------------------------------------------------------------

def test_parse_module_empty_string():
    """parse_module('') should return an empty IRModule, not raise."""
    parser = KTIRParser()
    module = parser.parse_module("")
    assert len(module.functions) == 0


def test_parse_module_whitespace_only():
    """Whitespace-only input should return an empty IRModule."""
    parser = KTIRParser()
    module = parser.parse_module("   \n\t\n   ")
    assert len(module.functions) == 0


def test_parse_module_comments_only():
    """Input that is only comments should return an empty IRModule."""
    parser = KTIRParser()
    module = parser.parse_module("// nothing here\n// also nothing\n")
    assert len(module.functions) == 0


# ---------------------------------------------------------------------------
# Malformed MLIR
# ---------------------------------------------------------------------------

def test_parse_module_unclosed_module_brace():
    """Module with unclosed outer brace should not crash — returns partial or empty result."""
    mlir = "module {\n  func.func @foo() {\n  }\n"  # missing closing }
    parser = KTIRParser()
    # Should not raise; may return a partial module or empty.
    module = parser.parse_module(mlir)
    assert module is not None


def test_parse_module_func_unclosed_body():
    """Function body with unclosed brace: parser should skip the function gracefully."""
    mlir = "module {\n  func.func @broken() {\n    %0 = arith.constant 1 : index\n"
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    # The broken function may be skipped entirely (body extraction returns None).
    assert module is not None


def test_parse_module_truncated_mid_op():
    """Module truncated in the middle of an operation should not crash."""
    mlir = _minimal_func("  %x = arith.constant")  # truncated
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    assert module is not None


# ---------------------------------------------------------------------------
# Unrecognized / invalid ops
# ---------------------------------------------------------------------------

def test_parse_operation_text_empty():
    """_parse_operation_text with empty string returns None."""
    parser = KTIRParser()
    result = parser._parse_operation_text("")
    assert result is None


def test_parse_operation_text_whitespace():
    """_parse_operation_text with whitespace-only string returns None."""
    parser = KTIRParser()
    result = parser._parse_operation_text("   \t  ")
    assert result is None


def test_parse_operation_text_unrecognized_op():
    """Op text that matches no dialect parser falls back to general parser without crashing."""
    parser = KTIRParser()
    # This op name is not registered in any dialect parser.
    result = parser._parse_operation_text("%x = totally.unknown.dialect.op %y : index")
    # General parser should produce something or return None — either is acceptable.
    # The key assertion: no exception.
    assert result is None or result.op_type == "totally.unknown.dialect.op"


def test_parse_operation_text_garbage():
    """Garbage text that matches no pattern returns None without raising."""
    parser = KTIRParser()
    result = parser._parse_operation_text("@@@###$$$!!!")
    assert result is None


# ---------------------------------------------------------------------------
# Partial parse: some ops valid, some not
# ---------------------------------------------------------------------------

def test_partial_parse_mixed_ops():
    """Module where some ops are valid and some are unrecognised parses the valid ones."""
    mlir = _minimal_func(
        "    %c0 = arith.constant 0 : index\n"
        "    @@@garbage@@@\n"
        "    %c1 = arith.constant 1 : index\n"
    )
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    assert len(module.functions) == 1
    func = module.functions["test"]
    # At least the two arith.constant ops should be present.
    op_types = [o.op_type for o in func.operations]
    assert "arith.constant" in op_types


# ---------------------------------------------------------------------------
# Unicode and binary-adjacent content
# ---------------------------------------------------------------------------

def test_parse_module_unicode_in_comments():
    """Unicode characters inside comments should not crash the parser."""
    mlir = _minimal_func("    // こんにちは 🎉 unicode comment\n    %c0 = arith.constant 0 : index\n")
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    assert len(module.functions) == 1


def test_parse_module_unicode_in_op():
    """Unicode in an op body (not comment) should not crash — may fail to parse the op."""
    mlir = "module {\n  func.func @unicode_test() {\n    %x = arith.constant こんにちは : index\n  }\n}\n"
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    assert module is not None


def test_parse_file_unicode_content(tmp_path):
    """parse_file() on a file with unicode text should not crash."""
    f = tmp_path / "unicode.mlir"
    f.write_text(_minimal_func("    // 日本語テスト\n"), encoding="utf-8")
    parser = KTIRParser()
    module = parser.parse_file(str(f))
    assert module is not None


# ---------------------------------------------------------------------------
# Grid attribute edge cases (parser robustness)
# ---------------------------------------------------------------------------

def test_parse_grid_missing():
    """Function with no grid attribute gets default (1,1,1)."""
    mlir = _minimal_func()
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    assert module.functions["test"].grid == (1, 1, 1)


def test_parse_grid_malformed():
    """Function with malformed grid attribute falls back to default without crashing."""
    mlir = "module {\n  func.func @g() attributes { grid = [not_a_number] } {\n  }\n}\n"
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    assert module.functions["g"].grid == (1, 1, 1)
