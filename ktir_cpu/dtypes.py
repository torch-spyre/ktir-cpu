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

"""Canonical KTIR dtype mappings.

Single source of truth for all dtype-related logic.  Every module that
needs to convert between KTIR dtype strings and NumPy dtypes should
import from here.
"""

import numpy as np

SUPPORTED_DTYPES: dict[str, np.dtype] = {
    # floating point
    "f16":     np.dtype(np.float16),
    "fp16":    np.dtype(np.float16),
    "float16": np.dtype(np.float16),
    "f32":     np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    # integer
    "i32":     np.dtype(np.int32),
    "si32":    np.dtype(np.int32),
    "index":   np.dtype(np.int32),
    "i64":     np.dtype(np.int64),
    "si64":    np.dtype(np.int64),
}

# Placeholder dtypes: not yet exercised by any example.  to_np_dtype raises
# NotImplementedError for these so that any future example that uses them
# fails test_examples immediately, forcing a proper implementation first.
# fp8/mxfp8 are registered as placeholders that raise NotImplementedError. 
# RFC 0682 doesn't define them; torch-spyre uses "fp8" at 1 byte/elem on hardware. 
# NOTE: What should the simulation representation be?
_PLACEHOLDER_DTYPES: frozenset[str] = frozenset({"fp8", "mxfp8"})

_REVERSE_MAP: dict[np.dtype, str] = {
    np.dtype(np.float16): "f16",
    np.dtype(np.float32): "f32",
    np.dtype(np.int32):   "i32",
    np.dtype(np.int64):   "i64",
}


def to_np_dtype(dtype: str) -> np.dtype:
    """Convert a KTIR dtype string to a NumPy dtype.

    Raises NotImplementedError for placeholder dtypes (fp8, mxfp8).
    Raises ValueError for unrecognised strings.
    """
    if dtype in _PLACEHOLDER_DTYPES:
        raise NotImplementedError(
            f"dtype {dtype!r} is a placeholder pending hardware confirmation; "
            "update SUPPORTED_DTYPES before adding examples that use it"
        )
    try:
        return SUPPORTED_DTYPES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported KTIR dtype: {dtype!r}")


def bytes_per_elem(dtype: str) -> int:
    """Return element size in bytes for a KTIR dtype string."""
    return int(to_np_dtype(dtype).itemsize)


def to_ktir_dtype(np_dtype: np.dtype) -> str:
    """Map a NumPy dtype to a canonical KTIR dtype string.

    Raises ValueError for unrecognised NumPy dtypes.
    """
    np_dtype = np.dtype(np_dtype)
    try:
        return _REVERSE_MAP[np_dtype]
    except KeyError:
        raise ValueError(f"No KTIR dtype for NumPy dtype: {np_dtype!r}")
