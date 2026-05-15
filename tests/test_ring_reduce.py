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

"""End-to-end test for ring_reduce.mlir.

4-core ring reduce: each core holds a 1×128 f16 row in HBM.  After
``ktdp.reduce`` (reduce_to_core<0>, sum across grid axis 0) core 0's
output row must equal the element-wise sum of all 4 input rows.

Marked xfail until PR-B lands the generator-based scheduler that
fixes K1/K2 (multi-round communication / cyclic communication
correctness — see docs/gap_analysis.md).

The parser also needs updating for the new ktdp.reduce attribute
syntax (#ktdp.reduce_kind, #ktdp.reduce_mode, #ktdp.grid_axis)
introduced in torch-spyre/ktir-mlir-frontend#21.
"""

import os

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter

_MLIR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "ktir", "ring_reduce.mlir"
)

_NUM_CORES = 4
_COLS = 128
_DTYPE = np.float16

# HBM layout:
#   in_ptr  = 0             : 4 rows × 128 elems × 2 bytes = 1024 bytes
#   out_ptr = 1024          : 1 row  × 128 elems × 2 bytes = 256  bytes
_IN_PTR = 0
_OUT_PTR = _NUM_CORES * _COLS * np.dtype(_DTYPE).itemsize


@pytest.mark.xfail(
    reason=(
        "Requires generator-based scheduler (PR-B, gap_analysis.md K1/K2) "
        "and parser support for #ktdp.reduce_kind / reduce_mode / grid_axis "
        "attributes (torch-spyre/ktir-mlir-frontend#21)."
    )
)
def test_ring_reduce_sum():
    """4-core ring reduce: core 0's output equals sum of all 4 input rows."""
    rng = np.random.default_rng(42)
    rows = rng.uniform(1.0, 2.0, size=(_NUM_CORES, _COLS)).astype(_DTYPE)

    interp = KTIRInterpreter()
    interp.load(_MLIR_PATH)
    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        # Write each core's input row contiguously starting at _IN_PTR.
        hbm.write(_IN_PTR, rows.flatten())
        # Zero-initialise output region.
        hbm.write(_OUT_PTR, np.zeros(_COLS, dtype=_DTYPE))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function("ring_reduce", in_ptr=_IN_PTR, out_ptr=_OUT_PTR)

    result = interp.memory.hbm.read(_OUT_PTR, _COLS, "f16")
    expected = rows.sum(axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-2,
                               err_msg="Core 0 output does not match element-wise sum")
