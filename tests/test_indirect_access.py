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
# Helpers: interpreter scaffolding
# ---------------------------------------------------------------------------

def _prep_with_hbm(mlir, seed_fn):
    """Load MLIR, patch _prepare_execution with seed_fn(hbm), return interpreter."""
    interp = KTIRInterpreter()
    interp.load(mlir)
    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        seed_fn(interp.memory.hbm)
    interp._prepare_execution = _prepare_and_seed
    return interp


def _run_with_hbm(mlir, func_name, seed_fn):
    """Load MLIR, seed HBM via seed_fn(hbm), execute func_name, return interpreter."""
    interp = _prep_with_hbm(mlir, seed_fn)
    interp.execute_function(func_name)
    return interp


def _seed_rfc_zeros(hbm):
    """64x64 zero tensors at sticks 0/64/128/192 — X, IDX1, IDX2, Y."""
    hbm.write(0, np.zeros(64 * 64, dtype=np.float16))       # X (stick 0)
    hbm.write(64, np.zeros(64 * 64, dtype=np.int32))        # IDX1 (stick 64)
    hbm.write(128, np.zeros(64 * 64, dtype=np.int32))       # IDX2 (stick 128)
    hbm.write(192, np.zeros(64 * 64, dtype=np.float16))     # Y (stick 192)


# ---------------------------------------------------------------------------
# MLIR templates: parameterised on indirect_side ("load" or "store")
# ---------------------------------------------------------------------------

def _small_indirect_4x4_mlir(func_name, indirect_side="load"):
    """4x4 indirect access fixture.

    indirect_side='load': IAT on X (gather from X via indices into Y)
    indirect_side='store': IAT on Y (scatter from X into Y via indices)
    """
    header = f"""
#coord_set_4x4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_order = affine_map<(d0, d1) -> (d0, d1)>

module {{
  func.func @{func_name}() attributes {{grid = [1, 1]}} {{
    %X_addr    = arith.constant 0 : index
    %IDX1_addr = arith.constant 32 : index
    %IDX2_addr = arith.constant 64 : index
    %Y_addr    = arith.constant 192 : index

    %X_view = ktdp.construct_memory_view %X_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xf16>

    %IDX1 = ktdp.construct_memory_view %IDX1_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xi32>

    %IDX2 = ktdp.construct_memory_view %IDX2_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xi32>

    %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xf16>

"""
    if indirect_side == "load":
        tiles = """    %X_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k)
        %X_view[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
            variables_space_set = #var_space_set,
            variables_space_order = #var_space_order
        } : memref<4x4xf16>, memref<4x4xi32>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>

    %c0 = arith.constant 0 : index
    %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0] {
        access_tile_set = #coord_set_4x4,
        access_tile_order = #var_space_order
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>
"""
    else:
        tiles = """    %c0 = arith.constant 0 : index
    %X_access_tile = ktdp.construct_access_tile %X_view[%c0, %c0] {
        access_tile_set = #coord_set_4x4,
        access_tile_order = #var_space_order
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>

    %Y_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k)
        %Y_view[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
            variables_space_set = #var_space_set,
            variables_space_order = #var_space_order
        } : memref<4x4xf16>, memref<4x4xi32>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>
"""
    footer = """
    %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<4x4xindex> -> tensor<4x4xf16>
    ktdp.store %X_data_tile, %Y_access_tile : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>

    return
  }
}
"""
    return header + tiles + footer


def _indirect_vso_mlir_4x4(vso_str, func_name, indirect_side="load"):
    """4x4 indirect access fixture parameterised on vso and which side is indirect.

    indirect_side='load': IAT on X with vso, direct AT on Y with identity order
    indirect_side='store': direct AT on X with identity order, IAT on Y with vso
    """
    header = f"""
#coord_set_4x4   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#var_space_order = affine_map<{vso_str}>
#cso_identity    = affine_map<(d0, d1) -> (d0, d1)>

module {{
  func.func @{func_name}() attributes {{grid = [1, 1]}} {{
    %X_addr    = arith.constant 0 : index
    %IDX1_addr = arith.constant 32 : index
    %IDX2_addr = arith.constant 64 : index
    %Y_addr    = arith.constant 192 : index

    %X_view = ktdp.construct_memory_view %X_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xf16>

    %IDX1 = ktdp.construct_memory_view %IDX1_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xi32>

    %IDX2 = ktdp.construct_memory_view %IDX2_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xi32>

    %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [4, 4], strides: [4, 1] {{
        coordinate_set = #coord_set_4x4,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<4x4xf16>

"""
    if indirect_side == "load":
        tiles = """    %X_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k)
        %X_view[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
            variables_space_set = #coord_set_4x4,
            variables_space_order = #var_space_order
        } : memref<4x4xf16>, memref<4x4xi32>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>

    %c0 = arith.constant 0 : index
    %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0] {
        access_tile_set   = #coord_set_4x4,
        access_tile_order = #cso_identity
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>
"""
    else:
        tiles = """    %c0 = arith.constant 0 : index
    %X_access_tile = ktdp.construct_access_tile %X_view[%c0, %c0] {
        access_tile_set   = #coord_set_4x4,
        access_tile_order = #cso_identity
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>

    %Y_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k)
        %Y_view[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
            variables_space_set = #coord_set_4x4,
            variables_space_order = #var_space_order
        } : memref<4x4xf16>, memref<4x4xi32>, memref<4x4xi32> -> !ktdp.access_tile<4x4xindex>
"""
    footer = """
    %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<4x4xindex> -> tensor<4x4xf16>
    ktdp.store %X_data_tile, %Y_access_tile : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>

    return
  }
}
"""
    return header + tiles + footer


def _small_3d_indirect_3cycle_mlir(func_name, indirect_side="load"):
    """2x2x2 indirect access with 3-cycle vso (d0,d1,d2)->(d2,d0,d1).

    indirect_side='load': IAT on X, direct AT on Y
    indirect_side='store': direct AT on X, IAT on Y
    """
    header = f"""
#coord_set_2x2x2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 1 >= 0)>
#vso_3cycle      = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#cso_identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {{
  func.func @{func_name}() attributes {{grid = [1, 1]}} {{
    %X_addr   = arith.constant 0 : index
    %IDX_addr = arith.constant 32 : index
    %Y_addr   = arith.constant 128 : index

    %X_view = ktdp.construct_memory_view %X_addr, sizes: [2, 2, 2], strides: [4, 2, 1] {{
        coordinate_set = #coord_set_2x2x2,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<2x2x2xf16>

    %IDX = ktdp.construct_memory_view %IDX_addr, sizes: [2, 2, 2], strides: [4, 2, 1] {{
        coordinate_set = #coord_set_2x2x2,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<2x2x2xi32>

    %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [2, 2, 2], strides: [4, 2, 1] {{
        coordinate_set = #coord_set_2x2x2,
        memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<2x2x2xf16>

"""
    if indirect_side == "load":
        tiles = """    %X_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k, %l)
        %X_view[ind(%IDX[%m, %k, %l]), (%k), (%l)] {
            variables_space_set = #coord_set_2x2x2,
            variables_space_order = #vso_3cycle
        } : memref<2x2x2xf16>, memref<2x2x2xi32> -> !ktdp.access_tile<2x2x2xindex>

    %c0 = arith.constant 0 : index
    %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0, %c0] {
        access_tile_set   = #coord_set_2x2x2,
        access_tile_order = #cso_identity_3d
    } : memref<2x2x2xf16> -> !ktdp.access_tile<2x2x2xindex>
"""
    else:
        tiles = """    %c0 = arith.constant 0 : index
    %X_access_tile = ktdp.construct_access_tile %X_view[%c0, %c0, %c0] {
        access_tile_set   = #coord_set_2x2x2,
        access_tile_order = #cso_identity_3d
    } : memref<2x2x2xf16> -> !ktdp.access_tile<2x2x2xindex>

    %Y_access_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%m, %k, %l)
        %Y_view[ind(%IDX[%m, %k, %l]), (%k), (%l)] {
            variables_space_set = #coord_set_2x2x2,
            variables_space_order = #vso_3cycle
        } : memref<2x2x2xf16>, memref<2x2x2xi32> -> !ktdp.access_tile<2x2x2xindex>
"""
    footer = """
    %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<2x2x2xindex> -> tensor<2x2x2xf16>
    ktdp.store %X_data_tile, %Y_access_tile : tensor<2x2x2xf16>, !ktdp.access_tile<2x2x2xindex>

    return
  }
}
"""
    return header + tiles + footer


# ---------------------------------------------------------------------------
# RFC example: indirect-access-copy.mlir (moved from test_spec_gaps.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,func_name,entry", get_test_params("indirect_access_copy"))
def test_indirect_access_tile_rfc(path, func_name, entry):
    """construct_indirect_access_tile (2-D gather) — RFC example file."""
    _run_with_hbm(path, func_name, _seed_rfc_zeros)


# ---------------------------------------------------------------------------
# Small 4x4 indirect gather with data verification
# ---------------------------------------------------------------------------
# X  = 4x4 matrix at stick 0, values X[i,j] = i*4+j  (0..15)
# IDX1 = 4x4 matrix at stick 1, each row = [3, 2, 1, 0]  (reversed row indices)
# IDX2 = 4x4 matrix at stick 2, each row = [0, 1, 2, 3]  (identity col indices)
# Y  = 4x4 matrix at stick 3
#
# Kernel: Y[m,k] = X[ IDX1[m,k], IDX2[m,k] ]
# Expected: Y[m,k] = X[3-k, k] for all m.  (All rows of Y are the same.)
#   Y[m,0] = X[3,0] = 12
#   Y[m,1] = X[2,1] = 9
#   Y[m,2] = X[1,2] = 6
#   Y[m,3] = X[0,3] = 3


def test_small_indirect_gather():
    """Verify indirect gather Y[m,k] = X[IDX1[m,k], IDX2[m,k]] produces correct data."""
    def seed(hbm):
        hbm.write(0, np.arange(16, dtype=np.float16))                        # X: 4x4, values 0..15
        hbm.write(1, np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 4))     # IDX1: each row = [3,2,1,0]
        hbm.write(2, np.tile(np.array([0, 1, 2, 3], dtype=np.int32), 4))     # IDX2: each row = [0,1,2,3]
        hbm.write(3, np.zeros(16, dtype=np.float16))                         # Y: zeros
    interp = _run_with_hbm(
        _small_indirect_4x4_mlir("small_indirect_gather", "load"),
        "small_indirect_gather", seed)

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
    %IDX_addr = arith.constant 2048 : index
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
    %IDX_addr = arith.constant 2048 : index
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
    def seed(hbm):
        hbm.write(0, np.zeros(16, dtype=np.float16))
        hbm.write(64, np.zeros(16, dtype=np.int32))
    interp = _prep_with_hbm(_SSA_NONZERO_RANGE_MLIR, seed)
    with pytest.raises(ValueError, match="outer SSA value"):
        interp.execute_function("bad_indirect")


def test_ssa_intermediate_var_zero_range_ok():
    """Outer SSA value in intermediate_variables with zero range must NOT raise.

    When the coordinate range for the SSA dimension is exactly [0, 0], the
    iterator is pinned to that single point and the SSA value drives the
    coordinate correctly — the old-style usage is still valid.
    """
    def seed(hbm):
        hbm.write(0, np.zeros(16, dtype=np.float16))
        hbm.write(64, np.zeros(16, dtype=np.int32))
    _run_with_hbm(_SSA_ZERO_RANGE_MLIR, "ok_indirect", seed)


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
    _run_with_hbm(path, func_name, _seed_rfc_zeros)


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


def test_small_indirect_scatter():
    """Verify scatter Y[IDX1[m,k], IDX2[m,k]] = X[m,k] produces correct data
    when (IDX1, IDX2) form a bijection (no coordinate collisions)."""
    def seed(hbm):
        hbm.write(0, np.arange(16, dtype=np.float16))                        # X: 4x4, values 0..15
        idx1 = np.repeat(np.array([3, 2, 1, 0], dtype=np.int32), 4)          # IDX1[m,k] = 3-m
        hbm.write(1, idx1)
        idx2 = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 4)            # IDX2[m,k] = 3-k
        hbm.write(2, idx2)
        hbm.write(3, np.zeros(16, dtype=np.float16))                         # Y: zeros
    interp = _run_with_hbm(
        _small_indirect_4x4_mlir("small_indirect_scatter", "store"),
        "small_indirect_scatter", seed)

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
    def seed(hbm):
        hbm.write(0, np.arange(16, dtype=np.float16))                        # X: 4x4, values 0..15
        hbm.write(1, np.zeros(16, dtype=np.int32))                           # IDX1, IDX2 all zeros → every (m,k) writes to Y[0,0]
        hbm.write(2, np.zeros(16, dtype=np.int32))
        hbm.write(3, np.full(16, -1, dtype=np.float16))                      # Y: sentinel -1 to verify untouched cells
    interp = _run_with_hbm(
        _small_indirect_4x4_mlir("small_indirect_scatter", "store"),
        "small_indirect_scatter", seed)

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
    "indirect_side,func_name",
    [
        ("load", "small_indirect_gather"),   # IAT on X
        ("store", "small_indirect_scatter"),  # IAT on Y
    ],
    ids=["indirect_load", "indirect_store"],
)
def test_negative_indirect_index_raises(indirect_side, func_name):
    """A negative entry in the index tensor must raise IndexError, not silently wrap."""
    def seed(hbm):
        hbm.write(0, np.zeros(16, dtype=np.float16))
        idx1 = np.zeros(16, dtype=np.int32)
        idx1[0] = -1
        hbm.write(1, idx1)
        hbm.write(2, np.zeros(16, dtype=np.int32))
        hbm.write(3, np.zeros(16, dtype=np.float16))
    interp = _prep_with_hbm(
        _small_indirect_4x4_mlir(func_name, indirect_side), seed)
    with pytest.raises(IndexError, match="is negative"):
        interp.execute_function(func_name)


# ---------------------------------------------------------------------------
# Non-identity variables_space_order
# ---------------------------------------------------------------------------
# RFC 0682 §473: vso defines lexicographic traversal order over the variable
# space (rightmost output = innermost iteration).  These fixtures exercise the
# non-identity branch by templating the vso affine_map into a 4x4 IAT shell.


def _seed_swap_4x4(hbm):
    """X = 0..15 row-major; IDX1[m,k]=m; IDX2[m,k]=k; Y zeroed."""
    hbm.write(0, np.arange(16, dtype=np.float16))                            # X[m,k] = m*4+k
    hbm.write(1, np.repeat(np.arange(4, dtype=np.int32), 4))                 # IDX1[m,k] = m
    hbm.write(2, np.tile(np.arange(4, dtype=np.int32), 4))                   # IDX2[m,k] = k
    hbm.write(3, np.zeros(16, dtype=np.float16))                             # Y zeroed


@pytest.mark.parametrize(
    "indirect_side,func_name",
    [
        ("load",  "gather_swap_vso"),   # IAT on X, vso swaps iteration order
        ("store", "scatter_swap_vso"),   # IAT on Y, vso swaps iteration order
    ],
    ids=["indirect_load_4x4_swap", "indirect_store_4x4_swap"],
)
def test_swap_vso(indirect_side, func_name):
    """A swap vso (d0,d1)->(d1,d0) reorders iteration to (d1,d0); with
    identity-coord IDX (IDX1[m,k]=m, IDX2[m,k]=k) the result Y equals X
    transposed.

    The same expected Y holds for both load (gather X via IAT then store
    direct) and store (load X direct then scatter via IAT) — vso is applied
    on the IAT side in both directions.

    Subscript resolution must use the variable-space point (not vso(pt)):
    IDX values are sensitive to which point is passed in, so a wrong
    implementation that fed vso(pt) to the subscripts would produce X
    itself rather than X^T at non-symmetric positions.
    """
    # MLIR: @{func_name}, vso=(d0,d1)->(d1,d0), 4x4 f16 with 2 i32 index views
    mlir = _indirect_vso_mlir_4x4("(d0, d1) -> (d1, d0)", func_name, indirect_side)
    interp = _run_with_hbm(mlir, func_name, _seed_swap_4x4)

    y_data = interp.memory.hbm.read(3, 16, "f16").reshape(4, 4)
    # Y[r, c] = X[c, r] = c*4 + r
    expected = np.array(
        [[c * 4 + r for c in range(4)] for r in range(4)],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(y_data, expected)


# ---------------------------------------------------------------------------
# 3-D 2x2x2 fixture for the canonical 3-cycle vso example (RFC 0682 §473
# worked through in issue #58 discussion).  The 3-cycle is non-involution,
# so this case distinguishes the sort-then-gather implementation from any
# apply-then-load alternative that happens to coincide on involutions.
# ---------------------------------------------------------------------------


def _seed_3cycle(hbm):
    """X = 0..7, IDX[m,k,l]=m, Y zeroed."""
    hbm.write(0, np.arange(8, dtype=np.float16))                             # X
    hbm.write(1, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32))        # IDX[m,k,l]=m
    hbm.write(2, np.zeros(8, dtype=np.float16))                              # Y zeroed


def test_indirect_load_with_3cycle_vso():
    """3-D non-involution vso (d0,d1,d2)->(d2,d0,d1) — the canonical
    worked example from RFC 0682 §473 / issue #58.

    Setup: X[m,k,l] = m*4+k*2+l (values 0..7), IDX[m,k,l] = m, dims 1+2 are
    direct (k, l) — so the gather coordinate is just (m, k, l) and each
    point reads X[pt].

    Sort key under the 3-cycle is (l, m, k); enumerated points sorted by
    this key visit (in lex sort-key order):
        (0,0,0)→X=0, (0,1,0)→X=2, (1,0,0)→X=4, (1,1,0)→X=6,
        (0,0,1)→X=1, (0,1,1)→X=3, (1,0,1)→X=5, (1,1,1)→X=7
    Reshaped to (2,2,2) row-major:
        [[[0, 2], [4, 6]],
         [[1, 3], [5, 7]]]

    A non-3-cycle implementation (e.g. apply-then-load) would land at a
    different layout for this input — distinguishing this from the
    involution case caught by test_swap_vso.
    """
    mlir = _small_3d_indirect_3cycle_mlir("small_3d_indirect_gather_3cycle", "load")
    interp = _run_with_hbm(mlir, "small_3d_indirect_gather_3cycle", _seed_3cycle)

    y_data = interp.memory.hbm.read(2, 8, "f16").reshape(2, 2, 2)
    expected = np.array(
        [[[0, 2], [4, 6]],
         [[1, 3], [5, 7]]],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(y_data, expected)


def test_indirect_store_with_3cycle_vso():
    """Mirror of :func:`test_indirect_load_with_3cycle_vso` on the store
    side.  Read X identity-direct, scatter through Y with vso=3-cycle.

    Sort key under the 3-cycle is (l, m, k); enumerated points sort to:
        [(0,0,0), (0,1,0), (1,0,0), (1,1,0),
         (0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    With IDX[m,k,l]=m the resolved write coord equals the variable-space
    point itself, so ``Y[sorted_pts[i]] = X.flatten()[i]``.

    Crucial vs the involution case in :func:`test_swap_vso`: a 3-cycle is
    non-self-inverse, so any implementation that writes ``Y[vso(pt)]``
    instead of using vso only as an iteration sort key would land at a
    different layout — distinguishing the spec'd sort-then-scatter
    semantics from any apply-then-store alternative.
    """
    mlir = _small_3d_indirect_3cycle_mlir("small_3d_indirect_scatter_3cycle", "store")
    interp = _run_with_hbm(mlir, "small_3d_indirect_scatter_3cycle", _seed_3cycle)

    y_data = interp.memory.hbm.read(2, 8, "f16").reshape(2, 2, 2)
    expected = np.array(
        [[[0, 4], [1, 5]],
         [[2, 6], [3, 7]]],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(y_data, expected)


@pytest.mark.parametrize(
    "indirect_side,func_name",
    [
        ("load",  "gather_bad_vso"),    # IAT on X, non-perm vso
        ("store", "scatter_bad_vso"),   # IAT on Y, non-perm vso
    ],
    ids=["indirect_load_non_perm", "indirect_store_non_perm"],
)
def test_non_permutation_vso_raises(indirect_side, func_name):
    """A non-permutation vso (here (d0,d1)->(d0,d0), which collapses two
    inputs to the same output) must be rejected at op-execution time, not
    silently ordered.  Stable sort on duplicate keys would cluster
    same-key points together — that is well-defined Python but produces
    semantically meaningless output for the spec."""
    # MLIR: @{func_name}, vso=(d0,d1)->(d0,d0), 4x4 — intentionally invalid
    mlir = _indirect_vso_mlir_4x4("(d0, d1) -> (d0, d0)", func_name, indirect_side)
    def seed(hbm):
        hbm.write(0, np.zeros(16, dtype=np.float16))
        hbm.write(1, np.zeros(16, dtype=np.int32))
        hbm.write(2, np.zeros(16, dtype=np.int32))
        hbm.write(3, np.zeros(16, dtype=np.float16))
    interp = _prep_with_hbm(mlir, seed)
    with pytest.raises(ValueError, match="must permute its input dimensions"):
        interp.execute_function(func_name)


# ---------------------------------------------------------------------------
# parse_subscript_expr / eval_subscript_expr — unit tests
# ---------------------------------------------------------------------------

def _resolve_ssa(node, val=32):
    """Replace ("ssa", ...) leaves with ("const", val) — test-only tree walker."""
    tag = node[0]
    if tag == "ssa":
        return ("const", val)
    if tag in ("add", "sub"):
        return (tag, _resolve_ssa(node[1], val), _resolve_ssa(node[2], val))
    if tag == "mul":
        return (tag, node[1], _resolve_ssa(node[2], val))
    if tag in ("floordiv", "mod"):
        return (tag, _resolve_ssa(node[1], val), node[2])
    return node


class TestSubscriptExpr:
    """Unit tests for parse_subscript_expr and eval_subscript_expr.

    These exercise the helpers in isolation, independent of any MLIR module.
    var_names mirrors the intermediate_variables list; SSA refs are pre-resolved
    to ("const", v) before eval (as _resolve_node does at construct time).
    """

    from ktir_cpu.dialects.ktdp_helpers import parse_subscript_expr, eval_subscript_expr

    # --- single variable ---

    def test_bare_dim(self):
        from ktir_cpu.dialects.ktdp_helpers import parse_subscript_expr, eval_subscript_expr
        expr = parse_subscript_expr("%d0", ["d0", "d1"])
        assert expr == ("dim", 0)
        assert eval_subscript_expr(expr, (3, 7)) == 3

    # --- floordiv ---

    def test_floordiv_single_var(self):
        from ktir_cpu.dialects.ktdp_helpers import parse_subscript_expr, eval_subscript_expr
        expr = parse_subscript_expr("%d0 floordiv 4", ["d0"])
        assert expr == ("floordiv", ("dim", 0), 4)
        assert eval_subscript_expr(expr, (8,)) == 2
        assert eval_subscript_expr(expr, (7,)) == 1

    def test_mod_single_var(self):
        from ktir_cpu.dialects.ktdp_helpers import parse_subscript_expr, eval_subscript_expr
        expr = parse_subscript_expr("%d0 mod 64", ["d0"])
        assert expr == ("mod", ("dim", 0), 64)
        assert eval_subscript_expr(expr, (65,)) == 1
        assert eval_subscript_expr(expr, (64,)) == 0

    # --- compound LHS with SSA ref (pre-resolved to const) ---

    def test_compound_floordiv_with_ssa(self):
        # (%y_offset + %d_stick * 64 + %d_lane) floordiv 64
        # y_offset is an outer SSA value; d_stick, d_lane are iteration vars.
        from ktir_cpu.dialects.ktdp_helpers import parse_subscript_expr, eval_subscript_expr
        var_names = ["d_stick", "d_lane"]
        expr = parse_subscript_expr(
            "(%y_offset + %d_stick * 64 + %d_lane) floordiv 64",
            var_names,
        )
        expr = _resolve_ssa(expr)
        # (32 + d_stick*64 + d_lane) floordiv 64
        assert eval_subscript_expr(expr, (0, 0))  == 0   # (32+0+0)//64 == 0
        assert eval_subscript_expr(expr, (0, 32)) == 1   # (32+0+32)//64 == 1
        assert eval_subscript_expr(expr, (1, 0))  == 1   # (32+64+0)//64 == 1
        assert eval_subscript_expr(expr, (1, 32)) == 2   # (32+64+32)//64 == 2

    def test_compound_mod_with_ssa(self):
        from ktir_cpu.dialects.ktdp_helpers import parse_subscript_expr, eval_subscript_expr
        var_names = ["d_stick", "d_lane"]
        expr = parse_subscript_expr(
            "(%y_offset + %d_stick * 64 + %d_lane) mod 64",
            var_names,
        )
        expr = _resolve_ssa(expr)
        # (32 + d_stick*64 + d_lane) mod 64
        assert eval_subscript_expr(expr, (0, 0))  == 32  # (32+0+0) % 64
        assert eval_subscript_expr(expr, (0, 32)) == 0   # (32+0+32) % 64
        assert eval_subscript_expr(expr, (1, 0))  == 32  # (32+64+0) % 64
        assert eval_subscript_expr(expr, (1, 31)) == 63  # (32+64+31) % 64


# ---------------------------------------------------------------------------
# Integration: construct_indirect_access_tile with compound floordiv/mod
# direct subscripts — exercises the ktdp_ops.py parser path (line 792),
# which the TestSubscriptExpr unit tests bypass.
# ---------------------------------------------------------------------------

# Kernel: copy src[i] -> dst[i floordiv 64][i mod 64]
# src is a flat 128-element view (stick 0); dst is a 2x64 view (stick 100).
# Sticks are spaced far apart so the two allocations don't collide
# (STICK_BYTES=128, f16=2 bytes → 64 f16 per stick; 128 f16 spans 2 sticks).
# intermediate_variables(%page, %lane) iterate over [0,2) x [0,64).
# src subscript: (%page * 64 + %lane)   — direct, compound mul+add
# dst dim 0:     (%page * 64 + %lane) floordiv 64  — compound LHS floordiv
# dst dim 1:     (%page * 64 + %lane) mod 64        — compound LHS mod
# After the copy dst[p][l] == src[p*64+l] == p*64+l for all p,l.
_COMPOUND_FLOORDIV_MOD_MLIR = """
#src_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#dst_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_order = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @compound_floordiv_mod() attributes {grid = [1, 1]} {
    %src_addr = arith.constant 0 : index
    %dst_addr = arith.constant 6400 : index

    %src = ktdp.construct_memory_view %src_addr, sizes: [128], strides: [1] {
        coordinate_set = #src_set,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128xf16>

    %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 64], strides: [64, 1] {
        coordinate_set = #dst_set,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x64xf16>

    %src_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%page, %lane)
        %src[(%page * 64 + %lane)] {
            variables_space_set = #var_set,
            variables_space_order = #var_order
        } : memref<128xf16> -> !ktdp.access_tile<2x64xindex>

    %dst_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%page, %lane)
        %dst[(%page * 64 + %lane) floordiv 64, (%page * 64 + %lane) mod 64] {
            variables_space_set = #var_set,
            variables_space_order = #var_order
        } : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>

    %data = ktdp.load %src_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
    ktdp.store %data, %dst_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>

    return
  }
}
"""


def test_compound_floordiv_mod_direct_subscript():
    """Compound-LHS floordiv/mod in construct_indirect_access_tile direct subscripts.

    Exercises ktdp_ops.py parse path (strip('()') bug site) with a non-trivial
    expression as the floordiv/mod LHS.  Without the fix, floordiv/mod would be
    silently dropped and dst would not be written correctly.
    """
    def seed(hbm):
        hbm.write(0, np.arange(128, dtype=np.float16))                       # src: stick 0
        hbm.write(100, np.zeros(128, dtype=np.float16))                      # dst: stick 100
    interp = _run_with_hbm(_COMPOUND_FLOORDIV_MOD_MLIR, "compound_floordiv_mod", seed)

    dst = interp.memory.hbm.read(100, 128, "f16").reshape(2, 64)
    expected = np.arange(128, dtype=np.float16).reshape(2, 64)
    np.testing.assert_array_equal(dst, expected)
