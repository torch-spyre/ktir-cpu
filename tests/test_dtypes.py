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

"""Tests for ktir_cpu.dtypes — the canonical dtype mapping module."""

import numpy as np
import pytest

from ktir_cpu.dtypes import bytes_per_elem, to_ktir_dtype, to_np_dtype


@pytest.mark.parametrize("ktir_dtype, expected_np, expected_bytes", [
    ("f16",     np.float16, 2),
    ("fp16",    np.float16, 2),
    ("float16", np.float16, 2),
    ("f32",     np.float32, 4),
    ("float32", np.float32, 4),
    ("i32",     np.int32,   4),
    ("si32",    np.int32,   4),
    ("index",   np.int32,   4),
    ("i64",     np.int64,   8),
    ("si64",    np.int64,   8),
])
def test_to_np_dtype(ktir_dtype, expected_np, expected_bytes):
    assert to_np_dtype(ktir_dtype) == np.dtype(expected_np)
    assert bytes_per_elem(ktir_dtype) == expected_bytes


@pytest.mark.parametrize("bad_dtype", ["bf16", "i8", "unknown", ""])
def test_unknown_dtype_raises(bad_dtype):
    with pytest.raises(ValueError, match="Unsupported"):
        to_np_dtype(bad_dtype)


@pytest.mark.parametrize("placeholder_dtype", ["fp8", "mxfp8"])
def test_placeholder_dtype_raises(placeholder_dtype):
    with pytest.raises(NotImplementedError):
        to_np_dtype(placeholder_dtype)


@pytest.mark.parametrize("np_dtype, expected_ktir", [
    (np.float16, "f16"),
    (np.float32, "f32"),
    (np.int32,   "i32"),
    (np.int64,   "i64"),
])
def test_to_ktir_dtype(np_dtype, expected_ktir):
    assert to_ktir_dtype(np.dtype(np_dtype)) == expected_ktir


def test_to_ktir_dtype_unknown_raises():
    with pytest.raises(ValueError, match="No KTIR dtype"):
        to_ktir_dtype(np.dtype(np.float64))
