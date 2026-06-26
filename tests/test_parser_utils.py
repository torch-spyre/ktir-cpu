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

"""Unit tests for parser_utils helpers.

Covers element types whose name contains the dimension separator ``x``
(``index``, ``complex``); the previous ``inner.split('x')`` implementation
mis-tokenised these by splitting on the ``x`` inside the dtype.
"""

import pytest

from ktir_cpu.parser_utils import parse_multi_result_lhs, parse_tensor_or_memref_type, extract_outs_operands


# ---------------------------------------------------------------------------
# Basic shape/dtype combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_str, expected_shape, expected_dtype",
    [
        # Floats
        ("tensor<256xf16>", (256,), "f16"),
        ("tensor<1024xf32>", (1024,), "f32"),
        ("tensor<8xbf16>", (8,), "bf16"),
        # Signless integers
        ("tensor<10xi32>", (10,), "i32"),
        ("tensor<3xi64>", (3,), "i64"),
        ("tensor<7xi1>", (7,), "i1"),
        # 2D and higher rank
        ("tensor<1x64xf32>", (1, 64), "f32"),
        ("tensor<128x16xf16>", (128, 16), "f16"),
        ("tensor<1x16x1x128xf16>", (1, 16, 1, 128), "f16"),
    ],
)
def test_parse_tensor_or_memref_type_basic(type_str, expected_shape, expected_dtype):
    """Plain numeric/float dtypes round-trip cleanly across rank 1–4."""
    info = parse_tensor_or_memref_type(type_str)
    assert info == {"shape": expected_shape, "dtype": expected_dtype}


# ---------------------------------------------------------------------------
# Regression: dtypes whose name contains 'x'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_str, expected_shape",
    [
        ("tensor<2xindex>", (2,)),
        ("tensor<3xindex>", (3,)),
        ("tensor<2x3xindex>", (2, 3)),
        ("tensor<1x16x1xindex>", (1, 16, 1)),
    ],
)
def test_parse_tensor_or_memref_type_index_dtype(type_str, expected_shape):
    """``index`` dtype is preserved despite the ``x`` inside its name.

    Pins the regression where ``inner.split('x')`` on ``"2xindex"`` produced
    ``["2", "inde", ""]``, taking ``""`` as the dtype. ``arith.constant
    dense<[1, N]> : tensor<2xindex>`` is the shape operand emitted by
    ``tensor.reshape`` lowerings; mis-parsing it broke any KTIR containing
    ``tl.reshape`` from a 3D descriptor load.
    """
    info = parse_tensor_or_memref_type(type_str)
    assert info == {"shape": expected_shape, "dtype": "index"}


# ---------------------------------------------------------------------------
# Non-matching inputs return None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_str",
    [
        "f32",                  # Bare element type (no dims)
        "not a tensor",         # Random text
        "",                     # Empty
        "tensor<>",             # Malformed: empty body
        "tensor<f32>",          # Rank-0 tensor — no dim tokens
        "tensor<?xf16>",        # All-dynamic dims — dropped → empty shape
    ],
)
def test_parse_tensor_or_memref_type_rejects_invalid(type_str):
    """Inputs with no parseable dim tokens return ``None``."""
    assert parse_tensor_or_memref_type(type_str) is None


# ---------------------------------------------------------------------------
# Real-MLIR forms: dynamic dims, encoding attribute, whitespace,
# trailing context after the closing '>'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_str, expected_shape, expected_dtype",
    [
        # Dynamic dims ('?') are dropped by default.
        ("tensor<?x4xf32>", (4,), "f32"),
        ("tensor<2x?x4xindex>", (2, 4), "index"),
        # Encoding attribute (RFC-allowed second positional).
        ("tensor<4x4xf32, #my_enc>", (4, 4), "f32"),
        ("tensor<8xf16, dense<0> : tensor<8xi1>>", (8,), "f16"),
        # MLIR pretty-printer whitespace tolerance.
        ("tensor< 2 x f32 >", (2,), "f32"),
        ("tensor<1 x 64 x i32>", (1, 64), "i32"),
        # Trailing context after the closing '>' — ignored.
        ("tensor<4xf32> loc(unknown)", (4,), "f32"),
        ("tensor<4xf32>, %arg0", (4,), "f32"),
        # memref wrapper
        ("memref<128x32xf16>", (128, 32), "f16"),
        ("memref<4x4xi32, #ktdp.spyre_memory_space<LX>>", (4, 4), "i32"),
        # Bare inner content (no wrapper)
        ("64x32xindex", (64, 32), "index"),
        ("256xbf16", (256,), "bf16"),
    ],
)
def test_parse_tensor_or_memref_type_real_mlir_forms(type_str, expected_shape, expected_dtype):
    """Forms that the upstream MLIR pretty-printer can produce."""
    info = parse_tensor_or_memref_type(type_str)
    assert info == {"shape": expected_shape, "dtype": expected_dtype}


@pytest.mark.parametrize(
    "type_str, expected_shape, expected_dtype",
    [
        ("tensor<?x4xf32>", (None, 4), "f32"),
        ("tensor<?xf16>", (None,), "f16"),
        ("memref<?x32xf32>", (None, 32), "f32"),
        ("128x?xindex", (128, None), "index"),
    ],
)
def test_parse_tensor_or_memref_type_keep_dynamic_dims(type_str, expected_shape, expected_dtype):
    """With keep_dynamic_dims=True, '?' dims appear as None."""
    info = parse_tensor_or_memref_type(type_str, keep_dynamic_dims=True)
    assert info == {"shape": expected_shape, "dtype": expected_dtype}


# ---------------------------------------------------------------------------
# Known limitation: nested-bracket element types
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Nested-bracket element types (complex<f32>, !tt.ptr<f32>, "
        "vector<4xf32>) cannot be parsed with a single Python regex — "
        "``re`` does not count balanced ``<>`` so the outer ``>`` is "
        "indistinguishable from a nested one. These types do not appear "
        "in lowered KTIR reaching this parser today (only in TTIR). "
        "Promote to a real test if a depth-counting parser replaces the "
        "regex, or if the parser starts to consume TTIR fragments."
    ),
    strict=True,
)
@pytest.mark.parametrize(
    "type_str, expected_shape, expected_dtype",
    [
        ("tensor<4xcomplex<f32>>", (4,), "complex<f32>"),
        ("tensor<4x!tt.ptr<f32>>", (4,), "!tt.ptr<f32>"),
        ("tensor<4xvector<4xf32>>", (4,), "vector<4xf32>"),
    ],
)
def test_parse_tensor_or_memref_type_nested_bracket_dtype(type_str, expected_shape, expected_dtype):
    """xfail: dtypes whose own form contains ``<...>`` are not supported.

    The regex stops at the first ``>``, so ``complex<f32>`` is parsed
    with the inner ``>`` mistaken for the closing bracket of the tensor
    type, leaving the dtype truncated to ``complex<f32`` (no closing
    bracket) and the trailing ``>`` unconsumed. A depth-counting parser
    is needed to fix this — out of scope for the current change.
    """
    info = parse_tensor_or_memref_type(type_str)
    assert info == {"shape": expected_shape, "dtype": expected_dtype}


# ---------------------------------------------------------------------------
# extract_outs_operands
# ---------------------------------------------------------------------------

def test_extract_outs_operands_single():
    assert extract_outs_operands(
        "linalg.matmul ins(%a, %b : tensor<4x4xf16>, tensor<4x4xf16>) "
        "outs(%c : tensor<4x4xf16>) -> tensor<4x4xf16>"
    ) == ["%c"]


def test_extract_outs_operands_multi():
    assert extract_outs_operands(
        "linalg.generic ins(%a : tensor<4xf16>) "
        "outs(%c : tensor<4xf16>, %d : tensor<4xf16>)"
    ) == ["%c", "%d"]


def test_extract_outs_operands_none():
    assert extract_outs_operands("arith.addf %x, %y : f16") == []

# ---------------------------------------------------------------------------
# parse_multi_result_lhs for supporting parsing result LHS containing a mix 
#                        of bundled and split forms.
# ---------------------------------------------------------------------------

def test_parse_multi_result_lhs_single():
    assert parse_multi_result_lhs("%x") == ["%x"]


def test_parse_multi_result_lhs_comma():
    assert parse_multi_result_lhs("%a, %b, %c") == ["%a", "%b", "%c"]


def test_parse_multi_result_lhs_bundled():
    assert parse_multi_result_lhs("%g:3") == ["%g#0", "%g#1", "%g#2"]


def test_parse_multi_result_lhs_bundled_one(): assert parse_multi_result_lhs("%r:1") == ["%r#0"]


def test_parse_multi_result_lhs_mixed():
    assert parse_multi_result_lhs("%a:2, %b") == ["%a#0", "%a#1", "%b"]


def test_parse_multi_result_lhs_mixed_complex():
    assert parse_multi_result_lhs("%x:2, %y, %z:3") == [
        "%x#0", "%x#1", "%y", "%z#0", "%z#1", "%z#2"
    ]


def test_parse_multi_result_lhs_malformed():
    with pytest.raises(ValueError, match="cannot parse multi-result LHS"):
        parse_multi_result_lhs("not_valid")
