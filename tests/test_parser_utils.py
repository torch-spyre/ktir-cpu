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

from ktir_cpu.parser_utils import parse_tensor_type


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
def test_parse_tensor_type_basic(type_str, expected_shape, expected_dtype):
    """Plain numeric/float dtypes round-trip cleanly across rank 1–4."""
    info = parse_tensor_type(type_str)
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
def test_parse_tensor_type_index_dtype(type_str, expected_shape):
    """``index`` dtype is preserved despite the ``x`` inside its name.

    Pins the regression where ``inner.split('x')`` on ``"2xindex"`` produced
    ``["2", "inde", ""]``, taking ``""`` as the dtype. ``arith.constant
    dense<[1, N]> : tensor<2xindex>`` is the shape operand emitted by
    ``tensor.reshape`` lowerings; mis-parsing it broke any KTIR containing
    ``tl.reshape`` from a 3D descriptor load.
    """
    info = parse_tensor_type(type_str)
    assert info == {"shape": expected_shape, "dtype": "index"}


# ---------------------------------------------------------------------------
# Non-matching inputs return None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_str",
    [
        "memref<10xf32>",       # Different aggregate type
        "f32",                  # Bare element type
        "not a tensor",         # Random text
        "",                     # Empty
        "tensor<>",             # Malformed: empty body
        "tensor<f32>",          # Rank-0 tensor — unsupported by the regex parser
    ],
)
def test_parse_tensor_type_rejects_non_tensor(type_str):
    """Inputs that are not a ranked tensor type return ``None``."""
    assert parse_tensor_type(type_str) is None
