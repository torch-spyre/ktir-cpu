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

"""Spec-gap tests: one xfail(strict=True) per known RFC conformance gap.

Each test here documents a KTDP/KTIR dialect feature that is present in the
RFC example files but not yet implemented in the interpreter.  When the
interpreter gains support for a feature, the corresponding test will begin
passing — at which point ``strict=True`` causes pytest to report it as an
unexpected pass (XPASS), reminding the developer to promote it to a real
passing test and remove the xfail marker.

Tests are grouped by the implementing issue they gate, in phase order.
All tests carry the ``spec_gap`` marker declared in pyproject.toml.
"""

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter

from conftest import get_test_params


# ===========================================================================
# First-class access tile IR
# Affine attributes (base_map, access_tile_set, access_tile_order,
# coordinate_set) are currently dropped at parse time.
# ===========================================================================

# No standalone RFC example exercises this gap in isolation — it is a
# prerequisite for later work rather than a directly observable
# behaviour.  Tests will be added in test_ktir_simple.py once the IR is
# extended (see issue).


# ===========================================================================
# Minimal affine / integer-set evaluator
# Covers: named affine aliases outside module {}, affine_map evaluation,
# affine_set enumeration.
# ===========================================================================

# test_affine_aliases_outside_module — implemented and moved to
# test_tile.py::TestTileOps::test_affine_attrs_preserved
# (complete)


# ===========================================================================
# load / store over coordinate sets
# ktdp.load / ktdp.store currently use a rectangular-slice shortcut; they
# need to iterate over the affine coordinate set from the access tile.
# ===========================================================================

# No RFC example file isolates this gap without also requiring later ops.
# Covered implicitly by the indirect-access and distributed-view tests below. 
# Will add more tests in test_examples.py, and test_latency.py


# ===========================================================================
# ktdp.construct_indirect_access_tile
# RFC §C.5: indirect gather Y[m,k] = X[IDX1[m,k], IDX2[m,k]] and
#           paged-attention style 4-D indirect gather.
# ===========================================================================

# test_indirect_access_tile — implemented and moved to
# test_indirect_access.py::test_indirect_access_tile_rfc
# (2-D indirect gather complete)


@pytest.mark.spec_gap
@pytest.mark.xfail(strict=True, reason="ktdp.load of 4x8x2048x128 f16 tile (16 MB) exceeds 2 MB LX scratchpad")
@pytest.mark.parametrize("path,func_name,entry", get_test_params("paged_tensor_copy_1core"))
def test_paged_tensor_indirect_access(path, func_name, entry):
    """paged-tensor-copy.mlir uses production-sized tensors; the single ktdp.load
    of the full 4x8x2048x128 output tile (16 MB) overflows the 2 MB LX scratchpad.
    The interpreter correctly parses and partially executes the kernel — the
    failure is a capacity limit of the example, not a missing feature.
    See test_indirect_access.py for a correctly-scaled 4-D paged indirect test.
    """
    interp = KTIRInterpreter()
    interp.load(path)
    _orig = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        # Idx tensor at 20000000: shape 4x32 i32, all zeros → every page ID = 0
        hbm.write(20000000, np.zeros(4 * 32, dtype=np.int32))
        # X tensor at 30000000: seed page 0 only (65536 f16 elements)
        # All accesses land on page 0 because Idx is all zeros.
        hbm.write(30000000, np.zeros(65536, dtype=np.float16))
    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)


# test_paged_attention — implemented and moved to
# test_examples.py::TestPagedAttentionExecution::test_paged_attention
# (Phase 2: 4-D paged indirect access with linalg.generic complete)


# ===========================================================================
# ktdp.construct_distributed_memory_view
# RFC §C.3: compose per-partition views (HBM + LX) into one logical view.
# ===========================================================================

@pytest.mark.spec_gap
@pytest.mark.xfail(strict=True, reason="ktdp.construct_distributed_memory_view not implemented")
@pytest.mark.parametrize("path,func_name,entry", get_test_params("distributed_view_copy"))
def test_distributed_memory_view(path, func_name, entry):
    """construct_distributed_memory_view is not yet supported."""
    interp = KTIRInterpreter()
    interp.load(path)
    interp.execute_function(func_name)


# ===========================================================================
# RFC-explicit non-ktdp ops
# add-with-control-flow.mlir uses linalg.add and tensor.empty inside scf.for.
# scf.for itself is already implemented; the gap is linalg.add / tensor.empty.
# ===========================================================================

@pytest.mark.spec_gap
@pytest.mark.xfail(reason="linalg.add not implemented")
@pytest.mark.parametrize("path,func_name,entry", get_test_params("add"))
def test_linalg_add_tensor_empty(path, func_name, entry):
    """linalg.add is not yet supported (tensor.empty is now implemented)."""
    interp = KTIRInterpreter()
    interp.load(path)
    interp.execute_function(func_name)


@pytest.mark.spec_gap
@pytest.mark.xfail(strict=True, reason="tensor.extract_slice not implemented")
def test_tensor_extract_slice():
    """tensor.extract_slice is not yet supported.

    No RFC fixture file exists yet for this op; inline MLIR is used until
    an examples/rfc/ file is added.
    The slice result is stored back to memory so an unknown-op skip causes a
    KeyError on %slice, ensuring the test cannot pass silently.
    """
    interp = KTIRInterpreter()
    interp.load("""
module {
  func.func @extract_slice_kernel() attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %src = arith.constant 0 : index
    %dst = arith.constant 256 : index
    %src_view = ktdp.construct_memory_view %src, sizes: [8, 8], strides: [8, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x8xf16>
    %dst_view = ktdp.construct_memory_view %dst, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>
    %src_access = ktdp.construct_access_tile %src_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8x8xf16> -> !ktdp.access_tile<8x8xindex>
    %dst_access = ktdp.construct_access_tile %dst_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>
    %tile = ktdp.load %src_access : !ktdp.access_tile<8x8xindex> -> tensor<8x8xf16>
    %slice = tensor.extract_slice %tile[0, 0][4, 4][1, 1] : tensor<8x8xf16> to tensor<4x4xf16>
    ktdp.store %slice, %dst_access : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>
    return
  }
}
""")
    interp.execute_function("extract_slice_kernel")


@pytest.mark.spec_gap
@pytest.mark.xfail(strict=True, reason="scf.parallel / scf.forall not implemented")
def test_scf_parallel():
    """scf.parallel and scf.forall are not yet supported.

    No RFC fixture file exists yet for these ops; inline MLIR is used until
    an examples/rfc/ file is added.
    The loop result is consumed by a store so an unknown-op skip causes a
    KeyError on %result, ensuring the test cannot pass silently.
    """
    interp = KTIRInterpreter()
    interp.load("""
module {
  func.func @parallel_kernel() attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %dst = arith.constant 0 : index
    %dst_view = ktdp.construct_memory_view %dst, sizes: [4, 1], strides: [1, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x1xf16>
    %dst_access = ktdp.construct_access_tile %dst_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<4x1xf16> -> !ktdp.access_tile<4x1xindex>
    %result = scf.parallel (%i) = (%c0) to (%c4) step (%c1) -> tensor<4x1xf16> {
      scf.reduce
    }
    ktdp.store %result, %dst_access : tensor<4x1xf16>, !ktdp.access_tile<4x1xindex>
    return
  }
}
""")
    interp.execute_function("parallel_kernel")


@pytest.mark.spec_gap
@pytest.mark.xfail(strict=True, reason="scf.reduce / scf.reduce.return not implemented")
def test_scf_reduce():
    """scf.reduce and scf.reduce.return are not yet supported.

    No RFC fixture file exists yet for these ops; inline MLIR is used until
    an examples/rfc/ file is added.
    The reduction result is used in an arith op so an unknown-op skip causes
    a KeyError on %result, ensuring the test cannot pass silently.
    """
    interp = KTIRInterpreter()
    interp.load("""
module {
  func.func @reduce_kernel() attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0.0 : f16
    %result = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%init) -> f16 {
      %val = arith.constant 1.0 : f16
      scf.reduce(%val : f16) {
        ^bb0(%lhs: f16, %rhs: f16):
          %sum = arith.addf %lhs, %rhs : f16
          scf.reduce.return %sum : f16
      }
    }
    %check = arith.addf %result, %init : f16
    return
  }
}
""")
    interp.execute_function("reduce_kernel")
