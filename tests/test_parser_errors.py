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


# ---------------------------------------------------------------------------
# Preprocessing: comment stripping
# ---------------------------------------------------------------------------


def test_preprocess_strips_inline_comments():
    """_preprocess_text removes // comments, preserving line structure."""
    text = "  %x = arith.addf %a, %b : f32  // add\n  return\n"
    result = KTIRParser._preprocess_text(text)
    assert "//" not in result
    assert "%x = arith.addf" in result
    assert "return" in result
    # Line count preserved (newlines intact)
    assert result.count("\n") == text.count("\n")


def test_preprocess_full_line_comment_becomes_blank():
    """A full-line comment becomes a blank line (op separator for tokenizer)."""
    result = KTIRParser._preprocess_text("    // this is a comment\n")
    assert result.strip() == ""


# ---------------------------------------------------------------------------
# Region detection: % in comments must not trigger false positive
# ---------------------------------------------------------------------------


def test_percent_in_comment_not_misclassified_as_region():
    """Attribute block with % in a comment must not be treated as a region.

    _line_opens_region peeks into { } blocks for SSA references (%<id>).
    A bare % in a comment (e.g. '// 90% of time') must not match.
    Regression test for reviewer comment on PR #5.
    """
    mlir = """
module {
  func.func @test() attributes { grid = [1] } {
    %c0 = arith.constant 0 : index
    %x = arith.constant {
        value = 42 : i32
    } : index
    return %c0 : index
  }
}
"""
    # Baseline: attribute block without comment — no regions
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    func = module.get_function("test")
    x_op = [op for op in func.operations if op.result == "%x"][0]
    assert len(x_op.regions) == 0, "attribute block should not produce regions"


def test_percent_in_comment_inside_attribute_block():
    """Same as above but with a // comment containing % inside the block."""
    mlir = """
module {
  func.func @test() attributes { grid = [1] } {
    %c0 = arith.constant 0 : index
    %x = arith.constant {
        // use 100% of capacity
        value = 42 : i32
    } : index
    return %c0 : index
  }
}
"""
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    func = module.get_function("test")
    x_op = [op for op in func.operations if op.result == "%x"][0]
    assert len(x_op.regions) == 0, (
        "% in comment inside attribute block must not trigger region detection"
    )


def test_region_with_percent_in_comment_still_detected():
    """A real region with SSA refs + a comment containing % is still a region."""
    mlir = """
module {
  func.func @test(%lb: index, %ub: index, %step: index) attributes { grid = [1] } {
    scf.for %i = %lb to %ub step %step {
        // iterate over 100% of elements
        %c0 = arith.constant 0 : index
        scf.yield
    }
    return
  }
}
"""
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    func = module.get_function("test")
    for_op = [op for op in func.operations if op.op_type == "scf.for"][0]
    assert len(for_op.regions) == 1, "scf.for body should be detected as a region"


# ---------------------------------------------------------------------------
# Operation boundary detection
#
# The tokenizer detects where one MLIR operation ends and the next begins
# using three structural signals (no dialect-specific knowledge):
#
# 1. Type-terminal completeness (_is_op_complete)
# 2. Blank line (or full-line comment) after balanced braces
# 3. SSA assignment (%name = ) starting a new op
#
# Void terminators (return, scf.yield) are special-cased in _is_op_complete.
# ---------------------------------------------------------------------------


# -- Signal 1: type-terminal completeness --
# An op is complete when its text ends with `: <type>` or `-> <type>`,
# where the type ends with `>`, `index`, or a scalar like `f32`/`i32`.

def test_type_terminal_simple_types():
    """_is_op_complete matches scalar and index type terminals."""
    parser = KTIRParser()
    assert parser._is_op_complete("%x = arith.addf %a, %b : f32")
    assert parser._is_op_complete("%x = arith.addi %a, %b : index")
    assert parser._is_op_complete("%x = arith.addi %a, %b : i32")


def test_type_terminal_tensor():
    """_is_op_complete matches tensor<...> type terminals."""
    parser = KTIRParser()
    assert parser._is_op_complete("%t = tensor.empty() : tensor<128x256xf16>")


def test_type_terminal_nested_angle_brackets():
    """_is_op_complete handles nested <> in complex type annotations.

    Types like tensor<..., #encoding<{...}>> have nested <> but the
    heuristic only checks that the line ends with >, so nesting depth
    doesn't matter.
    """
    parser = KTIRParser()
    complex_type = (
        "%t = tensor.empty() : tensor<128x256xf16, "
        "#sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>>"
    )
    assert parser._is_op_complete(complex_type)


def test_type_terminal_arrow():
    """_is_op_complete matches -> return type terminals."""
    parser = KTIRParser()
    assert parser._is_op_complete(
        "%acc = ktdp.construct_access_tile %ref[%c0, %c0]"
        " : memref<128x256xf16> -> !ktdp.access_tile<128x256xindex>"
    )


def test_type_terminal_void_terminators():
    """_is_op_complete recognises return and scf.yield as complete."""
    parser = KTIRParser()
    assert parser._is_op_complete("return")
    assert parser._is_op_complete("return %c0 : index")
    assert parser._is_op_complete("scf.yield")
    assert parser._is_op_complete("scf.yield %acc : tensor<8xf32>")


def test_type_terminal_no_match():
    """_is_op_complete returns False when op has no type terminal.

    Ops like linalg.reduce that end with `dimensions = [1]` are flushed
    by a blank line or SSA assignment instead.
    """
    parser = KTIRParser()
    assert not parser._is_op_complete(
        "%r = linalg.reduce { arith.maxnumf }"
        " ins(%x : tensor<4xf16>)"
        " outs(%init : tensor<1xf16>)"
        " dimensions = [1]"
    )


# -- Signal 2: blank line flush --
# A blank line (or full-line comment) after balanced braces flushes
# the accumulated op. This handles ops with no type terminal.

def test_blank_line_flushes_op():
    """An op without a type terminal is flushed by a following blank line."""
    mlir = """
module {
  func.func @test(%x: tensor<4xf16>, %init: tensor<1xf16>) attributes { grid = [1] } {
    %r = linalg.reduce { arith.maxnumf } ins(%x : tensor<4xf16>) outs(%init : tensor<1xf16>) dimensions = [0]

    return
  }
}
"""
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    func = module.get_function("test")
    op_types = [op.op_type for op in func.operations]
    assert "linalg.reduce" in op_types, "reduce op should be flushed by blank line"
    assert "return" in op_types


# -- Signal 3: SSA assignment flush --
# An incoming `%name = ` line unambiguously starts a new op,
# flushing the previous one even without a blank line.

def test_ssa_assignment_flushes_previous_op():
    """Adjacent ops without blank line: SSA assignment starts new op."""
    mlir = """
module {
  func.func @test() attributes { grid = [1] } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    return
  }
}
"""
    parser = KTIRParser()
    module = parser.parse_module(mlir)
    func = module.get_function("test")
    constants = [op for op in func.operations if op.op_type == "arith.constant"]
    assert len(constants) == 2, "two adjacent constants should be parsed as separate ops"


