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

"""Unit tests for ktdp.construct_indirect_access_tile.

Tests indirect (gather) memory access patterns with data correctness
verification, complementing the xfail-promoted spec-gap tests.
"""

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from conftest import get_test_params


# ---------------------------------------------------------------------------
# RFC example: indirect-access-copy.mlir (moved from test_spec_gaps.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,func_name,entry", get_test_params("indirect_access_copy"))
def test_indirect_access_tile_rfc(path, func_name, entry):
    """construct_indirect_access_tile (2-D gather) — RFC example file."""
    interp = KTIRInterpreter()
    interp.load(path)
    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        hbm.write(0, np.zeros(64 * 64, dtype=np.float16))       # X (stick 0)
        hbm.write(64, np.zeros(64 * 64, dtype=np.int32))        # IDX1 (stick 64)
        hbm.write(128, np.zeros(64 * 64, dtype=np.int32))       # IDX2 (stick 128)
        hbm.write(192, np.zeros(64 * 64, dtype=np.float16))     # Y (stick 192)
    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)


# ---------------------------------------------------------------------------
# Small 4x4 indirect gather with data verification
# ---------------------------------------------------------------------------
# X  = 4x4 matrix at stick 0, values X[i,j] = i*4+j  (0..15)
# IDX1 = 4x4 matrix at stick 1, each row = [3, 2, 1, 0]  (reversed row indices)
# IDX2 = 4x4 matrix at stick 2, each row = [0, 1, 2, 3]  (identity col indices)
# Y  = 4x4 matrix at stick 3
#
# Kernel: Y[m,k] = X[ IDX1[m,k], IDX2[m,k] ]
# Expected: Y[m,k] = X[3-k, k] ... wait, IDX1[m,k] = 3-k for each row.
# Actually IDX1 has shape 4x4 and each row is [3,2,1,0], so:
#   IDX1[m,0]=3, IDX1[m,1]=2, IDX1[m,2]=1, IDX1[m,3]=0
#   IDX2[m,k]=k
# So Y[m,k] = X[3-k, k] for all m.  (All rows of Y are the same.)
#   Y[m,0] = X[3,0] = 12
#   Y[m,1] = X[2,1] = 9
#   Y[m,2] = X[1,2] = 6
#   Y[m,3] = X[0,3] = 3

SMALL_INDIRECT_MLIR = """
#coord_set_4x4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_order = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @small_indirect_gather() attributes {grid = [1, 1]} {
    %X_addr    = arith.constant 0 : index
    %IDX1_addr = arith.constant 1 : index
    %IDX2_addr = arith.constant 2 : index
    %Y_addr    = arith.constant 3 : index

    %X = ktdp.construct_memory_view %X_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %IDX1 = ktdp.construct_memory_view %IDX1_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xi32>

    %IDX2 = ktdp.construct_memory_view %IDX2_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xi32>

    %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %X_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k)
        %X[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
            variables_space_set = #var_space_set,
            variables_space_order = #var_space_order
        } : memref<4x4xf16>, memref<4x4xi32>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>

    %c0 = arith.constant 0 : index
    %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0] {
        access_tile_set = #coord_set_4x4,
        access_tile_order = #var_space_order
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>

    %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<4x4xindex> -> tensor<4x4xf16>
    ktdp.store %X_data_tile, %Y_access_tile : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>

    return
  }
}
"""


def test_small_indirect_gather():
    """Verify indirect gather Y[m,k] = X[IDX1[m,k], IDX2[m,k]] produces correct data."""
    interp = KTIRInterpreter()
    interp.load(SMALL_INDIRECT_MLIR)

    # Seed HBM after _prepare_execution allocates memory but before ops run.
    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        # X: 4x4, values 0..15 as f16 (stick 0)
        hbm.write(0, np.arange(16, dtype=np.float16))
        # IDX1: 4x4, each row = [3,2,1,0] as i32 (stick 1)
        hbm.write(1, np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 4))
        # IDX2: 4x4, each row = [0,1,2,3] as i32 (stick 2)
        hbm.write(2, np.tile(np.array([0, 1, 2, 3], dtype=np.int32), 4))
        # Y: 4x4, zeros (stick 3)
        hbm.write(3, np.zeros(16, dtype=np.float16))
    interp._prepare_execution = _prepare_and_seed

    interp.execute_function("small_indirect_gather")

    # Read Y output from HBM (stick 3)
    y_data = interp.memory.hbm.read(3, 16, "f16").reshape(4, 4)

    # Expected: Y[m,k] = X[IDX1[m,k], IDX2[m,k]] = X[3-k, k]
    expected = np.array([
        [12, 9, 6, 3],  # X[3,0]=12, X[2,1]=9, X[1,2]=6, X[0,3]=3
        [12, 9, 6, 3],
        [12, 9, 6, 3],
        [12, 9, 6, 3],
    ], dtype=np.float16)

    np.testing.assert_array_equal(y_data, expected)


# ---------------------------------------------------------------------------
# Validation: outer SSA as intermediate variable with non-zero range must fail
# ---------------------------------------------------------------------------

# Minimal MLIR where %c2 (an outer SSA constant = 2) is listed as the first
# intermediate variable while that dimension's range in variables_space_set
# spans [0, 3] (non-zero).  The interpreter should reject this at construction
# time because iterating over that dimension would override the SSA value with
# the iterator position, silently producing wrong results.
_SSA_NONZERO_RANGE_MLIR = """
module {
  func.func @bad_indirect() attributes {grid = [1, 1]} {
    %X_addr   = arith.constant 0   : index
    %IDX_addr = arith.constant 64  : index
    %c2       = arith.constant 2   : index

    %X = ktdp.construct_memory_view %X_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %IDX = ktdp.construct_memory_view %IDX_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xi32>

    %tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%c2, %k)
        %X[ind(%IDX[%c2, %k]), (%k)] {
            variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
            variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<4x4xf16>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>

    return
  }
}
"""

# Same kernel but dimension 0's range is exactly [0, 0] (a constant range):
# this is the old-style usage that is still valid — %c2 acts as a constant
# because the iterator never moves off point (0,).
_SSA_ZERO_RANGE_MLIR = """
module {
  func.func @ok_indirect() attributes {grid = [1, 1]} {
    %X_addr   = arith.constant 0   : index
    %IDX_addr = arith.constant 64  : index
    %c2       = arith.constant 2   : index

    %X = ktdp.construct_memory_view %X_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %IDX = ktdp.construct_memory_view %IDX_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xi32>

    %tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%c2, %k)
        %X[ind(%IDX[%c2, %k]), (%k)] {
            variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
            variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<4x4xf16>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>

    return
  }
}
"""


def test_ssa_intermediate_var_nonzero_range_raises():
    """Outer SSA value in intermediate_variables with non-zero range must raise ValueError.

    Listing an outer SSA scalar (e.g. %c2) as an intermediate variable only
    produces correct results when that dimension's range in variables_space_set
    is [0, 0] (the iterator never moves).  A non-zero range silently overwrites
    the SSA value with the iterator position.  The interpreter must reject this
    at construction time.
    """
    interp = KTIRInterpreter()
    interp.load(_SSA_NONZERO_RANGE_MLIR)

    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        hbm.write(0, np.zeros(16, dtype=np.float16))
        hbm.write(64, np.zeros(16, dtype=np.int32))
    interp._prepare_execution = _prepare_and_seed

    with pytest.raises(ValueError, match="outer SSA value"):
        interp.execute_function("bad_indirect")


def test_ssa_intermediate_var_zero_range_ok():
    """Outer SSA value in intermediate_variables with zero range must NOT raise.

    When the coordinate range for the SSA dimension is exactly [0, 0], the
    iterator is pinned to that single point and the SSA value drives the
    coordinate correctly — the old-style usage is still valid.
    """
    interp = KTIRInterpreter()
    interp.load(_SSA_ZERO_RANGE_MLIR)

    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        hbm.write(0, np.zeros(16, dtype=np.float16))
        hbm.write(64, np.zeros(16, dtype=np.int32))
    interp._prepare_execution = _prepare_and_seed

    # Should complete without error (result correctness is not the focus here)
    interp.execute_function("ok_indirect")


# ---------------------------------------------------------------------------
# ktdp.store with IndirectAccessTile (scatter)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,func_name,entry", get_test_params("indirect_scatter"))
def test_indirect_scatter_rfc(path, func_name, entry):
    """ktdp.store with IndirectAccessTile (2-D scatter) — RFC-sized fixture.

    Smoke test only: all inputs are zero-seeded, so this verifies the kernel
    parses and executes end-to-end at production size, not data correctness.
    Correctness is covered by `test_small_indirect_scatter` (bijection) and
    `test_small_indirect_scatter_collision` on a 4x4 fixture.
    """
    interp = KTIRInterpreter()
    interp.load(path)
    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        hbm.write(0, np.zeros(64 * 64, dtype=np.float16))    # X (stick 0)
        hbm.write(64, np.zeros(64 * 64, dtype=np.int32))     # IDX1 (stick 64)
        hbm.write(128, np.zeros(64 * 64, dtype=np.int32))    # IDX2 (stick 128)
        hbm.write(192, np.zeros(64 * 64, dtype=np.float16))  # Y (stick 192)
    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)


# ---------------------------------------------------------------------------
# Small 4x4 indirect scatter with data verification (bijection)
# ---------------------------------------------------------------------------
# X    = 4x4 matrix at stick 0, values X[i,j] = i*4+j  (0..15)
# IDX1 = 4x4 matrix at stick 1, IDX1[m,k] = 3-m  (rows: [3,3,3,3],[2,2,2,2],...)
# IDX2 = 4x4 matrix at stick 2, IDX2[m,k] = 3-k  (each row [3,2,1,0])
# Y    = 4x4 matrix at stick 3
#
# Kernel: Y[IDX1[m,k], IDX2[m,k]] = X[m,k]  (scatter)
# Mapping (m,k) → (3-m, 3-k) is a bijection, so every source element lands at
# a distinct destination — the result is X rotated 180°: Y[3-m, 3-k] = X[m, k].
#   Y[3,3] = X[0,0] = 0      Y[3,0] = X[0,3] = 3
#   Y[2,3] = X[1,0] = 4      Y[0,0] = X[3,3] = 15
#   etc.

SMALL_INDIRECT_SCATTER_MLIR = """
#coord_set_4x4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_order = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @small_indirect_scatter() attributes {grid = [1, 1]} {
    %X_addr    = arith.constant 0 : index
    %IDX1_addr = arith.constant 1 : index
    %IDX2_addr = arith.constant 2 : index
    %Y_addr    = arith.constant 3 : index

    %X_view = ktdp.construct_memory_view %X_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %IDX1 = ktdp.construct_memory_view %IDX1_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xi32>

    %IDX2 = ktdp.construct_memory_view %IDX2_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xi32>

    %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %c0 = arith.constant 0 : index
    %X_access_tile = ktdp.construct_access_tile %X_view[%c0, %c0] {
        access_tile_set   = #coord_set_4x4,
        access_tile_order = #var_space_order
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>

    %Y_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k)
        %Y_view[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
            variables_space_set = #var_space_set,
            variables_space_order = #var_space_order
        } : memref<4x4xf16>, memref<4x4xi32>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>

    %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<4x4xindex> -> tensor<4x4xf16>
    ktdp.store %X_data_tile, %Y_access_tile : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>

    return
  }
}
"""


def test_small_indirect_scatter():
    """Verify scatter Y[IDX1[m,k], IDX2[m,k]] = X[m,k] produces correct data
    when (IDX1, IDX2) form a bijection (no coordinate collisions)."""
    interp = KTIRInterpreter()
    interp.load(SMALL_INDIRECT_SCATTER_MLIR)

    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        # X: 4x4, values 0..15 as f16
        hbm.write(0, np.arange(16, dtype=np.float16))
        # IDX1[m,k] = 3-m: rows are [3,3,3,3], [2,2,2,2], [1,1,1,1], [0,0,0,0]
        idx1 = np.repeat(np.array([3, 2, 1, 0], dtype=np.int32), 4)
        hbm.write(1, idx1)
        # IDX2[m,k] = 3-k: each row is [3,2,1,0]
        idx2 = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 4)
        hbm.write(2, idx2)
        # Y: 4x4, zeros
        hbm.write(3, np.zeros(16, dtype=np.float16))
    interp._prepare_execution = _prepare_and_seed

    interp.execute_function("small_indirect_scatter")

    y_data = interp.memory.hbm.read(3, 16, "f16").reshape(4, 4)

    # Y[3-m, 3-k] = X[m, k] = m*4+k → Y[r, c] = (3-r)*4 + (3-c)
    expected = np.array(
        [[(3 - r) * 4 + (3 - c) for c in range(4)] for r in range(4)],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(y_data, expected)


def test_small_indirect_scatter_collision():
    """All source elements map to Y[0,0] (IDX1 and IDX2 are all zeros).

    Locks the current implementation behavior: last source element written
    in row-major (vss.enumerate) order wins. This is *implementation-defined*
    behavior inherited from NumPy fancy-index assignment — it is **not** a
    spec guarantee. Future RFC changes (e.g. add/max reduction) may invalidate
    this test, in which case the assertion should be revised, not removed.
    """
    interp = KTIRInterpreter()
    interp.load(SMALL_INDIRECT_SCATTER_MLIR)

    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        # X: 4x4, values 0..15 as f16
        hbm.write(0, np.arange(16, dtype=np.float16))
        # IDX1, IDX2 all zeros → every (m,k) writes to Y[0,0]
        hbm.write(1, np.zeros(16, dtype=np.int32))
        hbm.write(2, np.zeros(16, dtype=np.int32))
        # Y: 4x4 sentinel value (-1) so we can verify untouched cells stay put
        hbm.write(3, np.full(16, -1, dtype=np.float16))
    interp._prepare_execution = _prepare_and_seed

    interp.execute_function("small_indirect_scatter")

    y_data = interp.memory.hbm.read(3, 16, "f16").reshape(4, 4)

    # Last writer in vss.enumerate order is (m=3, k=3), value X[3,3] = 15.
    assert y_data[0, 0] == 15.0
    # All other cells should remain at the sentinel value (untouched).
    for r in range(4):
        for c in range(4):
            if (r, c) == (0, 0):
                continue
            assert y_data[r, c] == -1.0, f"Y[{r},{c}] = {y_data[r, c]}, expected -1"


# ---------------------------------------------------------------------------
# Negative-index guard (covers both indirect_load and indirect_store)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mlir,func_name",
    [
        (SMALL_INDIRECT_MLIR, "small_indirect_gather"),
        (SMALL_INDIRECT_SCATTER_MLIR, "small_indirect_scatter"),
    ],
    ids=["indirect_load", "indirect_store"],
)
def test_negative_indirect_index_raises(mlir, func_name):
    """A negative entry in the index tensor must raise IndexError, not silently wrap."""
    interp = KTIRInterpreter()
    interp.load(mlir)

    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        hbm.write(0, np.zeros(16, dtype=np.float16))
        idx1 = np.zeros(16, dtype=np.int32)
        idx1[0] = -1
        hbm.write(1, idx1)
        hbm.write(2, np.zeros(16, dtype=np.int32))
        hbm.write(3, np.zeros(16, dtype=np.float16))
    interp._prepare_execution = _prepare_and_seed

    with pytest.raises(IndexError, match="is negative"):
        interp.execute_function(func_name)
