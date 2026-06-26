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

"""Unit tests for CoreContext scope stack and LX tracking.

Tests that region-scoped SSA values correctly manage LX scratchpad
memory via push_scope/pop_scope and set_value auto-tracking.
"""

import numpy as np
import pytest

from ktir_cpu.grid import CoreContext
from ktir_cpu.memory import LXOptions, LXScratchpad, HBMSimulator
from ktir_cpu.ir_types import Tile
from ktir_cpu.ops.memory_ops import MemoryOps

# ---------------------------------------------------------------------------
# LXOptions presets used throughout
# ---------------------------------------------------------------------------

_LX_BASELINE = LXOptions(alias_dedup=False, consume_last_use=False)
_LX_DEDUP    = LXOptions(alias_dedup=True,  consume_last_use=False)
_LX_FULL     = LXOptions(alias_dedup=True,  consume_last_use=True)

# ---------------------------------------------------------------------------
# MLIR kernels used by parse-level and end-to-end tests
# ---------------------------------------------------------------------------

# Small 8×8 matmul — used by TestUseCountsParsed to verify _build_use_counts.
MATMUL_UC_MLIR = """
module {
  func.func @matmul_uc_test(
      %a_ptr: index, %b_ptr: index, %c_ptr: index, %K: index,
      %BM: index, %BN: index, %BK: index
  ) attributes {grid = [1]} {
    %pid_m, %pid_n = ktdp.get_compute_tile_id : index, index
    %a_view = ktdp.construct_memory_view %a_ptr, sizes: [8, 8], strides: [8, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x8xf16>
    %b_view = ktdp.construct_memory_view %b_ptr, sizes: [8, 8], strides: [8, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x8xf16>
    %c_view = ktdp.construct_memory_view %c_ptr, sizes: [8, 8], strides: [8, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x8xf16>
    %c0 = arith.constant 0 : index
    %accum_zero = arith.constant dense<0.0> : tensor<8x8xf16>
    %c = scf.for %off_k = %c0 to %K step %BK iter_args(%accum_itr = %accum_zero) -> (tensor<8x8xf16>) {
      %a_acc = ktdp.construct_access_tile %a_view[%pid_m, %off_k] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<8x8xf16> -> !ktdp.access_tile<8x8xindex>
      %b_acc = ktdp.construct_access_tile %b_view[%off_k, %pid_n] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<8x8xf16> -> !ktdp.access_tile<8x8xindex>
      %a = ktdp.load %a_acc : !ktdp.access_tile<8x8xindex> -> tensor<8x8xf16>
      %b = ktdp.load %b_acc : !ktdp.access_tile<8x8xindex> -> tensor<8x8xf16>
      %c_init = tensor.empty() : tensor<8x8xf16>
      %a_dot_b = linalg.matmul ins(%a, %b : tensor<8x8xf16>, tensor<8x8xf16>)
                               outs(%c_init : tensor<8x8xf16>) -> tensor<8x8xf16>
      %accum_next = arith.addf %accum_itr, %a_dot_b : tensor<8x8xf16>
      scf.yield %accum_next : tensor<8x8xf16>
    }
    %c_acc = ktdp.construct_access_tile %c_view[%pid_m, %pid_n] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<8x8xf16> -> !ktdp.access_tile<8x8xindex>
    ktdp.store %c, %c_acc : tensor<8x8xf16>, !ktdp.access_tile<8x8xindex>
    return
  }
}
"""

# Large-tile matmul — used by TestLastUseMatmulKernel end-to-end test.
# BM=512, BN=256, BK=64: accumulator tile = 512*256*2 = 262144 bytes (256 KB).
# Peak live set with _LX_BASELINE exceeds 1 MB; with _LX_FULL it fits.
MATMUL_LX_MLIR = """
module {
  func.func @matmul_lx_test(
      %a_ptr: index,
      %b_ptr: index,
      %c_ptr: index,
      %K: index,
      %BM: index,
      %BN: index,
      %BK: index
  ) attributes {grid = [1]} {
    %pid_m, %pid_n = ktdp.get_compute_tile_id : index, index

    %a_view = ktdp.construct_memory_view %a_ptr, sizes: [512, 256], strides: [256, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<512x256xf16>

    %b_view = ktdp.construct_memory_view %b_ptr, sizes: [256, 256], strides: [256, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    %c_view = ktdp.construct_memory_view %c_ptr, sizes: [512, 256], strides: [256, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<512x256xf16>

    %c0 = arith.constant 0 : index
    %accum_zero = arith.constant dense<0.0> : tensor<512x256xf16>

    %c = scf.for %off_k = %c0 to %K step %BK iter_args(%accum_itr = %accum_zero) -> (tensor<512x256xf16>) {

      %a_acc = ktdp.construct_access_tile %a_view[%pid_m, %off_k] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<512x256xf16> -> !ktdp.access_tile<512x64xindex>

      %b_acc = ktdp.construct_access_tile %b_view[%off_k, %pid_n] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<256x256xf16> -> !ktdp.access_tile<64x256xindex>

      %a = ktdp.load %a_acc : !ktdp.access_tile<512x64xindex> -> tensor<512x64xf16>
      %b = ktdp.load %b_acc : !ktdp.access_tile<64x256xindex> -> tensor<64x256xf16>

      %c_init = tensor.empty() : tensor<512x256xf16>
      %a_dot_b = linalg.matmul ins(%a, %b : tensor<512x64xf16>, tensor<64x256xf16>)
                               outs(%c_init : tensor<512x256xf16>) -> tensor<512x256xf16>

      %accum_next = arith.addf %accum_itr, %a_dot_b : tensor<512x256xf16>

      scf.yield %accum_next : tensor<512x256xf16>
    }

    %c_acc = ktdp.construct_access_tile %c_view[%pid_m, %pid_n] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<512x256xf16> -> !ktdp.access_tile<512x256xindex>

    ktdp.store %c, %c_acc : tensor<512x256xf16>, !ktdp.access_tile<512x256xindex>

    return
  }
}
"""


def _make_context(lx_size_mb: int = 2, lx_options: LXOptions = None) -> CoreContext:
    """Create a CoreContext with a fresh LX scratchpad and HBM."""
    lx = LXScratchpad(size_mb=lx_size_mb, core_id=0)
    hbm = HBMSimulator()
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm,
                       lx_options=lx_options if lx_options is not None else _LX_FULL)


def _make_tile(shape: tuple, dtype: str = "f16") -> Tile:
    """Create a Tile backed by zeros."""
    np_dtype = np.float16 if dtype == "f16" else np.float32
    data = np.zeros(shape, dtype=np_dtype)
    return Tile(data=data, dtype=dtype, shape=shape)


# ---------------------------------------------------------------------------
# Basic scope push/pop
# ---------------------------------------------------------------------------

class TestScopeStack:
    """Test that push_scope / pop_scope correctly isolate SSA values."""

    def test_function_scope_is_always_present(self):
        """_scope_stack starts with one scope (function body) and cannot be popped."""
        ctx = _make_context()
        assert len(ctx._scope_stack) == 1
        with pytest.raises(RuntimeError, match="Cannot pop the function-body scope"):
            ctx.pop_scope()

    def test_inner_scope_sees_outer_values(self):
        """Values set in the function scope are visible inside a pushed scope."""
        ctx = _make_context()
        ctx.set_value("%x", 42)
        ctx.push_scope()
        assert ctx.get_value("%x") == 42

    def test_outer_scope_does_not_see_inner_values(self):
        """Values set in an inner scope are gone after pop_scope."""
        ctx = _make_context()
        ctx.push_scope()
        ctx.set_value("%body_local", 99)
        assert ctx.get_value("%body_local") == 99
        ctx.pop_scope()
        with pytest.raises(KeyError):
            ctx.get_value("%body_local")

    def test_has_value_searches_all_scopes(self):
        ctx = _make_context()
        ctx.set_value("%outer", 1)
        ctx.push_scope()
        ctx.set_value("%inner", 2)
        assert ctx.has_value("%outer")
        assert ctx.has_value("%inner")
        ctx.pop_scope()
        assert ctx.has_value("%outer")
        assert not ctx.has_value("%inner")

    def test_nested_scopes(self):
        """Three levels: function -> for -> nested for."""
        ctx = _make_context()
        ctx.set_value("%func_val", "f")
        ctx.push_scope()  # outer for
        ctx.set_value("%outer_val", "o")
        ctx.push_scope()  # inner for
        ctx.set_value("%inner_val", "i")

        # All visible from innermost
        assert ctx.get_value("%func_val") == "f"
        assert ctx.get_value("%outer_val") == "o"
        assert ctx.get_value("%inner_val") == "i"

        ctx.pop_scope()  # exit inner for
        assert ctx.has_value("%outer_val")
        assert not ctx.has_value("%inner_val")

        ctx.pop_scope()  # exit outer for
        assert ctx.has_value("%func_val")
        assert not ctx.has_value("%outer_val")


# ---------------------------------------------------------------------------
# LX tracking
# ---------------------------------------------------------------------------

class TestLXTracking:
    """set_value auto-tracking: every Tile binding charges lx.used exactly once.

    Tile size used in alias tests: tensor<32x1024xf16> = 32*1024*2 = 65536 bytes.

    Alias dedup (_LX_DEDUP vs _LX_BASELINE)
    ----------------------------------------
    Two SSA names can point to the same Python Tile object (e.g. linalg.reduce
    binds %result and %outs to the same buffer).  Without dedup, each binding
    charges separately:

      _LX_BASELINE — no id() tracking:
        set_value("%a", tile)  →  lx.used = 65536   (charged)
        set_value("%b", tile)  →  lx.used = 131072  (charged again — wrong)

      _LX_DEDUP — refcount by id():
        set_value("%a", tile)  →  lx.used = 65536   (first binding, charged)
        set_value("%b", tile)  →  lx.used = 65536   (same id(), refcount 1→2, no charge)
    """

    def test_set_value_tile_increments_used(self):
        ctx = _make_context()
        tile = _make_tile((32, 1024))  # 32*1024*2 = 65536 bytes
        ctx.set_value("%tile", tile)
        assert ctx.lx.used == 65536

    def test_overwrite_name_decrements_used(self):
        ctx = _make_context()
        tile = _make_tile((32, 1024))
        ctx.set_value("%tile", tile)
        assert ctx.lx.used == 65536
        ctx.set_value("%tile", 0)  # overwrite with a non-Tile — refcount drops to 0
        assert ctx.lx.used == 0

    def test_alias_charges_once_with_dedup(self):
        """With alias_dedup, two names for the same Tile charge LX once (N)."""
        ctx = _make_context(lx_options=_LX_DEDUP)
        tile = _make_tile((32, 1024))  # 65536 bytes
        ctx.set_value("%a", tile)
        ctx.set_value("%b", tile)  # same id(tile)
        assert ctx.lx.used == 65536

    def test_alias_double_charges_without_dedup(self):
        """Without alias_dedup, two names for the same Tile charge LX twice (2N)."""
        ctx = _make_context(lx_options=_LX_BASELINE)
        tile = _make_tile((32, 1024))  # 65536 bytes
        ctx.set_value("%a", tile)
        ctx.set_value("%b", tile)
        assert ctx.lx.used == 65536 * 2

    def test_pop_scope_frees_lx(self):
        """Popping a scope frees LX for all Tiles defined in that scope."""
        ctx = _make_context()
        ctx.push_scope()
        tile = _make_tile((32, 1024))  # 65536 bytes
        ctx.set_value("%tile", tile)
        assert ctx.lx.used == 65536

        ctx.pop_scope()
        assert ctx.lx.used == 0

    def test_pop_scope_does_not_free_outer_lx(self):
        """Popping inner scope preserves LX tracked in the outer scope."""
        ctx = _make_context()
        outer_tile = _make_tile((4, 64))  # 512 bytes
        ctx.set_value("%outer", outer_tile)

        ctx.push_scope()
        inner_tile = _make_tile((32, 1024))  # 65536 bytes
        ctx.set_value("%inner", inner_tile)
        assert ctx.lx.used == 512 + 65536

        ctx.pop_scope()
        assert ctx.lx.used == 512  # only outer remains

    def test_lx_overflow_raises(self):
        """Exceeding LX capacity raises MemoryError."""
        ctx = _make_context(lx_size_mb=1)  # 1 MB = 1048576 bytes
        # Two 512 KB tiles just fill capacity
        tile_a = _make_tile((512, 512))  # 512*512*2 = 524288 bytes = 512 KB
        tile_b = _make_tile((512, 512))
        ctx.set_value("%a", tile_a)
        ctx.set_value("%b", tile_b)
        assert ctx.lx.used == 1048576
        # One more byte should overflow
        tile_c = _make_tile((1,))  # 2 bytes
        with pytest.raises(MemoryError, match="LX scratchpad overflow"):
            ctx.set_value("%c", tile_c)

    def test_clear_values_resets_everything(self):
        ctx = _make_context()
        ctx.set_value("%x", 1)
        ctx.push_scope()
        ctx.set_value("%y", 2)
        tile = _make_tile((8, 64))
        ctx.set_value("%tile", tile)

        ctx.clear_values()
        assert ctx._scope_stack == [{}]
        assert ctx._tile_refcount == {}
        assert ctx.lx.used == 0


# ---------------------------------------------------------------------------
# iter_args simulation
# ---------------------------------------------------------------------------

class TestForOpLXTracking:
    """Regression tests for double-tracking in scf.for iter_args (issue #109)."""

    def test_iter_arg_init_does_not_double_count(self):
        """iter_arg init is a name alias, not a new allocation.

        In a pattern like:
            %accum_zero = arith.constant dense<0.0> : tensor<768x1024xf16>
            scf.for %k = ... iter_args(%accum_itr = %accum_zero) { ... }

        %accum_itr and %accum_zero are two SSA names for the same Python Tile
        object — no new buffer is allocated.  _execute_operation tracks the
        Tile under %accum_zero when arith.constant runs.  for_op must not
        track it again under %accum_itr or lx.used will be double the actual
        footprint, triggering a spurious MemoryError for tiles over half the
        LX capacity.

        LX state:
            after arith.constant:
              [%accum_zero: 1.5MB]   lx.used = 1.5MB

            after iter_arg binding (correct):
              [%accum_zero: 1.5MB]   lx.used = 1.5MB   ← same object, no new entry

            after iter_arg binding (wrong):
              [%accum_zero: 1.5MB, %accum_itr: 1.5MB]  lx.used = 3.0MB → MemoryError
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        # 768*1024*2 = 1_572_864 bytes — just under the 2 MB LX cap on its own,
        # but over cap if double-counted.
        tile = _make_tile((768, 1024))
        ctx.set_value("%accum_zero", tile)
        assert ctx.lx.used == 1_572_864

        # for_op binds %accum_itr = %accum_zero (same object).
        # The body just yields the same tile back unchanged each iteration.
        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=4, step=1,
            iter_var_name="%k",
            body_region=[],
            region_executor=lambda c, ops: _YieldResult([c.get_value("%accum_itr")]),
            iter_arg_names=["%accum_itr"],
            iter_init_values=[tile],
        )

        # lx.used must still reflect exactly one copy of the tile.
        assert ctx.lx.used == 1_572_864, (
            f"double-count: expected 1_572_864, got {ctx.lx.used}"
        )

    def test_iter_arg_does_not_leak_after_loop(self):
        """lx.used is the same before and after for_op runs.

        iter_arg names are scoped to the loop — they have no meaning once
        for_op returns.  Any _lx_bytes entry left behind under an iter_arg
        name inflates lx.used for the rest of the function, causing false
        overflow errors in subsequent ops that have nothing to do with the loop.

        LX state:
            before loop:   [%init: N]           lx.used = N
            inside loop:   [%init: N]           lx.used = N   ← %itr is just a name
            after loop:    [%init: N]           lx.used = N   ← no dangling %itr entry
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        tile = _make_tile((32, 64))
        ctx.set_value("%init", tile)
        used_before = ctx.lx.used

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=3, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=lambda c, ops: _YieldResult([c.get_value("%itr")]),
            iter_arg_names=["%itr"],
            iter_init_values=[tile],
        )

        # After the loop, lx.used must be exactly what it was before.
        # Any delta means for_op left a dangling refcount entry.
        assert ctx.lx.used == used_before, (
            f"leak: expected {used_before}, got {ctx.lx.used}"
        )

    def test_iter_arg_readable_inside_body(self):
        """The iter_arg value is visible to the body on every iteration.

        for_op binds the iter_arg name in the parent scope before entering
        the loop.  The body reads it via get_value, which searches the scope
        stack top-to-bottom and finds it in the parent scope.  This must hold
        regardless of whether for_op does any LX tracking — the name binding
        and the LX accounting are independent concerns.

        Scope stack during body execution:
            [0] function scope: {%acc: Tile, %itr: Tile}   ← iter_arg lives here
            [1] body scope:     {%i: 0, ...}
            get_value("%itr") finds it in scope[0]
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        tile = _make_tile((4, 4))
        ctx.set_value("%acc", tile)

        seen = []

        def body(c, ops):
            # Record what the body sees as the current iter_arg value.
            seen.append(c.get_value("%itr"))
            # Yield the same tile back (no-op accumulation).
            return _YieldResult([c.get_value("%itr")])

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=3, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=body,
            iter_arg_names=["%itr"],
            iter_init_values=[tile],
        )

        # Body ran 3 times and saw the tile each time.
        assert len(seen) == 3
        assert all(s is tile for s in seen)

    def test_two_iter_args_neither_double_counted(self):
        """Multiple iter_args from distinct already-tracked Tiles must not add to lx.used.

        Each Tile is already tracked under its own SSA name.  for_op binds two
        iter_arg names to those two objects.  lx.used must remain the sum of
        the two original tiles, not double that sum.

        LX state:
            after setup:        [%a: 2048, %b: 512]                     lx.used = 2560
            after iter binding: [%a: 2048, %b: 512]                     lx.used = 2560
            wrong behavior:     [%a: 2048, %b: 512, %itr_a: 2048, %itr_b: 512]  lx.used = 5120
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        tile_a = _make_tile((32, 32))   # 32*32*2 = 2048 bytes
        tile_b = _make_tile((16, 16))   # 16*16*2 = 512 bytes
        ctx.set_value("%a", tile_a)
        ctx.set_value("%b", tile_b)
        assert ctx.lx.used == 2048 + 512

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=2, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=lambda c, ops: _YieldResult([
                c.get_value("%itr_a"), c.get_value("%itr_b"),
            ]),
            iter_arg_names=["%itr_a", "%itr_b"],
            iter_init_values=[tile_a, tile_b],
        )

        assert ctx.lx.used == 2048 + 512

    def test_zero_iterations_lx_unchanged(self):
        """A loop with zero iterations must not change lx.used at all.

        lower_bound == upper_bound means the body never executes and no
        yielded values are produced.  The iter_arg init binding is still
        performed (the loop variable is valid in the enclosing scope) but
        no LX should be charged or leaked.

        LX state:
            before loop:  [%init: N]   lx.used = N
            (body never runs)
            after loop:   [%init: N]   lx.used = N
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        tile = _make_tile((8, 8))
        ctx.set_value("%init", tile)
        used_before = ctx.lx.used

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=0, step=1,  # zero iterations
            iter_var_name="%i",
            body_region=[],
            region_executor=lambda c, ops: _YieldResult([c.get_value("%itr")]),
            iter_arg_names=["%itr"],
            iter_init_values=[tile],
        )

        assert ctx.lx.used == used_before

    def test_scalar_iter_arg_no_lx_effect(self):
        """A scalar (non-Tile) iter_arg must have no effect on lx.used.

        Index counters and integer accumulators are common iter_args that
        carry no LX backing.  for_op must handle them without error and
        without touching lx.used.

        LX state:
            iter 0: %counter = 0   lx.used = 0
            iter 1: %counter = 1   lx.used = 0
            iter 2: %counter = 2   lx.used = 0
            ...                    (scalars never enter _lx_bytes)
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        assert ctx.lx.used == 0

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=4, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=lambda c, ops: _YieldResult([c.get_value("%counter") + 1]),
            iter_arg_names=["%counter"],
            iter_init_values=[0],
        )

        assert ctx.lx.used == 0

    def test_lx_stable_across_iterations(self):
        """lx.used must not grow with each iteration when iter_args are Tiles.

        If for_op were to track the iter_arg on every re-bind after yield,
        lx.used would accumulate one tile's worth of bytes per iteration.
        Measure lx.used at the end of each body invocation and confirm it
        is the same every time.

        LX state per iteration (correct):
            iter 0 body: lx.used = N   (just %init)
            iter 1 body: lx.used = N
            iter 2 body: lx.used = N
            ...

        LX state per iteration (wrong — re-tracking on each yield):
            iter 0 body: lx.used = N
            iter 1 body: lx.used = 2N
            iter 2 body: lx.used = 3N  → eventual MemoryError
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        tile = _make_tile((32, 32))
        ctx.set_value("%init", tile)
        baseline = ctx.lx.used

        used_per_iter = []

        def body(c, ops):
            used_per_iter.append(c.lx.used)
            return _YieldResult([c.get_value("%itr")])

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=5, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=body,
            iter_arg_names=["%itr"],
            iter_init_values=[tile],
        )

        assert len(used_per_iter) == 5
        assert all(u == baseline for u in used_per_iter), (
            f"lx.used grew across iterations: {used_per_iter}"
        )

    def test_new_tile_yielded_from_body_not_leaked(self):
        """A new Tile yielded from the body is live in the iter_arg after the loop.

        for_op() owns the loop-body scope (push_scope/pop_scope around each
        iteration).  The body callback receives the already-pushed scope and
        should NOT call push_scope/pop_scope itself — that would add an extra
        level not present in real _execute_operation execution.

        LX state per iteration (3 total, N = tile_size):
            iter 0 init:   parent={%init: tile, %itr: tile}   lx.used = N  (same object, rc=2)
            iter 0 body:   ... | body={%i:0, %new0: new0}     lx.used = 2N
            pop_scope:     parent={%init: tile, %itr: tile}   lx.used = N  (new0 freed, rc=0)
            rebind %itr:   parent={%init: tile, %itr: new0}   lx.used = 2N (tile rc 2→1, new0 rc 0→1)

            iter 1 body:   ... | body={%i:1, %new1: new1}     lx.used = 3N (init+new0+new1)
            pop_scope:     parent={%init: tile, %itr: new0}   lx.used = 2N (new1 freed)
            rebind %itr:   parent={%init: tile, %itr: new1}   lx.used = 2N (new0 freed, new1 charged)

        After the loop: lx.used = 2N (init + last yielded tile held by %itr).
        The interpreter then binds the scf.for result to the same last tile
        (refcount 1→2, no new charge) — total stays 2N.
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        tile = _make_tile((4, 4))
        ctx.set_value("%init", tile)
        tile_size = tile.size_bytes()
        used_before = ctx.lx.used  # = tile_size

        def body(c, ops):
            # for_op already pushed the body scope before calling this.
            # Simulate _execute_operation placing a new Tile directly in it.
            new_tile = _make_tile((4, 4))
            c.set_value("%new", new_tile)
            return _YieldResult([new_tile])

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=3, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=body,
            iter_arg_names=["%itr"],
            iter_init_values=[tile],
        )

        # After the loop: %init tile (N) + last yielded tile held by %itr (N) = 2N.
        # All previous iterations' tiles were freed when %itr was overwritten.
        assert ctx.lx.used == used_before + tile_size, (
            f"expected {used_before + tile_size}, got {ctx.lx.used}"
        )

    def test_nested_loops_inner_does_not_leak_into_outer(self):
        """Inner loop iter_args must not inflate lx.used seen by the outer loop.

        In a tiled matmul, the outer loop carries the accumulator and the inner
        loop iterates over K-tiles.  The inner loop's iter_args are scoped to
        the inner loop and must not affect the outer loop's LX accounting.

        LX state during outer iteration 1:
            outer body entry:  [%outer_init: A]                  lx.used = A
            inner loop runs:   [%outer_init: A, %inner_init: B]  lx.used = A+B
            inner loop exits:  [%outer_init: A, %inner_init: B]  lx.used = A+B
            after untrack:     [%outer_init: A]                  lx.used = A

        outer iteration 2 must see lx.used = A, same as iteration 1.
        """
        from ktir_cpu.ops.control_ops import ControlOps, _YieldResult

        ctx = _make_context()
        outer_tile = _make_tile((32, 32))
        ctx.set_value("%outer_init", outer_tile)
        used_at_outer_start = ctx.lx.used

        outer_used_per_iter = []

        def outer_body(c, ops):
            outer_used_per_iter.append(c.lx.used)

            # Inner loop with its own iter_arg.
            inner_tile = _make_tile((8, 8))
            c.set_value("%inner_init", inner_tile)

            ControlOps.for_op(
                context=c,
                lower_bound=0, upper_bound=2, step=1,
                iter_var_name="%j",
                body_region=[],
                region_executor=lambda c2, ops: _YieldResult([c2.get_value("%inner_itr")]),
                iter_arg_names=["%inner_itr"],
                iter_init_values=[inner_tile],
            )

            # Release the inner tile by overwriting the name with a non-Tile.
            c.set_value("%inner_init", None)
            return _YieldResult([c.get_value("%outer_itr")])

        ControlOps.for_op(
            context=ctx,
            lower_bound=0, upper_bound=3, step=1,
            iter_var_name="%i",
            body_region=[],
            region_executor=outer_body,
            iter_arg_names=["%outer_itr"],
            iter_init_values=[outer_tile],
        )

        # lx.used seen at the start of each outer iteration must be stable —
        # the inner loop must not have left any LX entries behind.
        assert all(u == used_at_outer_start for u in outer_used_per_iter), (
            f"inner loop leaked into outer: {outer_used_per_iter}"
        )


class TestIterArgsPersistence:
    """Simulate the for_op iter_arg flow: Tiles survive across iterations
    while body-local Tiles are freed each iteration."""

    def test_iter_arg_tiles_persist_body_local_freed(self):
        """Mimics a scf.for with one Tile iter_arg and one body-local Tile.

        Each iteration:
          1. push_scope
          2. create body-local Tile via set_value (auto-tracked)
          3. create new iter_arg Tile via set_value (auto-tracked)
          4. pop_scope (frees both body-local and new iter_arg Tile via refcount)
          5. set_value("%acc", new_acc) — decrements old tile, increments new tile

        After each iteration, lx.used == iter_arg size only.
        """
        ctx = _make_context()

        # Initial iter_arg: tensor<4x1xf16> = 8 bytes
        iter_tile = _make_tile((4, 1))
        ctx.set_value("%acc", iter_tile)
        assert ctx.lx.used == 8

        for i in range(3):
            ctx.push_scope()

            # Body-local: tensor<4x256xf16> = 2048 bytes
            body_tile = _make_tile((4, 256))
            ctx.set_value("%body_tile", body_tile)

            # New iter_arg value (created in body, will be yielded)
            new_acc = _make_tile((4, 1))
            ctx.set_value("%new_acc", new_acc)

            assert ctx.lx.used == 8 + 2048 + 8  # old acc + body + new acc

            # pop_scope frees body-local LX (%body_tile AND %new_acc) via refcount
            ctx.pop_scope()
            assert ctx.lx.used == 8  # only old %acc remains

            # Re-bind iter_arg: set_value decrements old tile, increments new tile
            ctx.set_value("%acc", new_acc)
            assert ctx.lx.used == 8  # back to steady state


# ---------------------------------------------------------------------------
# next_ptr rewind on pop_scope (issue #26)
# ---------------------------------------------------------------------------

class TestNextPtrRewind:
    """Regression tests for issue #26: lx.next_ptr must be rewound on
    pop_scope so it stays in lockstep with lx.used.
    """

    def test_next_ptr_returns_to_zero_each_iteration(self):
        """The exact failure mode from issue #26.

        Before the fix, next_ptr advanced by 2048 (tile size, stick-aligned)
        every iteration and crossed the 128 KB LX capacity at iter 64 while
        lx.used stayed 0. After the fix, both return to 0 on every pop.
        """
        lx = LXScratchpad(size_mb=0.125, core_id=0)  # 128 KB
        hbm = HBMSimulator()
        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)

        for i in range(100):
            ctx.push_scope()

            # Mirrors the interpreter's _execute_operation sequence for a
            # load op: write into LX, set the SSA value in the scope.
            # set_value auto-tracks; pop_scope auto-frees via refcount.
            tile_data = np.zeros((4, 256), dtype=np.float16)  # 2 KB
            MemoryOps._write_to_lx(ctx, tile_data)
            name = f"%tile_iter{i}"
            ctx.set_value(name, Tile(tile_data, "f16", tile_data.shape))

            ctx.pop_scope()

            # Strong invariant: both accountants return to their pre-push
            # values. Catches divergence at iter 0, not just overflow at 64.
            assert ctx.lx.used == 0, f"iter {i}: lx.used={ctx.lx.used}"
            assert ctx.lx.next_ptr == 0, (
                f"iter {i}: lx.next_ptr={ctx.lx.next_ptr}"
            )

    def test_next_ptr_restored_on_pop_single_level(self):
        """A single push/allocate/pop cycle restores next_ptr exactly."""
        ctx = _make_context()
        assert ctx.lx.next_ptr == 0

        ctx.push_scope()
        MemoryOps._write_to_lx(ctx, np.zeros((128,), dtype=np.float16))  # 256 B
        assert ctx.lx.next_ptr == 256
        ctx.pop_scope()
        assert ctx.lx.next_ptr == 0

    def test_next_ptr_restored_on_pop_with_outer_allocation(self):
        """Outer-scope allocations are preserved; only inner ones are reclaimed.

        This is the nested-scope discipline: push saves the current watermark,
        pop restores to it, so whatever the outer scope allocated survives.
        """
        ctx = _make_context()

        # Outer-scope allocation at the function level.
        MemoryOps._write_to_lx(ctx, np.zeros((128,), dtype=np.float16))  # 256 B
        outer_ptr = ctx.lx.next_ptr
        assert outer_ptr == 256

        ctx.push_scope()
        MemoryOps._write_to_lx(ctx, np.zeros((1024,), dtype=np.float16))  # 2 KB
        assert ctx.lx.next_ptr == 256 + 2048
        ctx.pop_scope()

        # Inner allocation reclaimed; outer watermark preserved.
        assert ctx.lx.next_ptr == outer_ptr

    def test_nested_scopes_rewind_lifo(self):
        """Three-level nesting (fn / for-body / if-body) restores each level."""
        ctx = _make_context()

        # Function-level allocation.
        MemoryOps._write_to_lx(ctx, np.zeros((128,), dtype=np.float16))  # A
        wm_fn = ctx.lx.next_ptr

        ctx.push_scope()                                                   # for-body
        MemoryOps._write_to_lx(ctx, np.zeros((512,), dtype=np.float16))   # B
        wm_for = ctx.lx.next_ptr

        ctx.push_scope()                                                   # if-body
        MemoryOps._write_to_lx(ctx, np.zeros((1024,), dtype=np.float16))  # C
        assert ctx.lx.next_ptr > wm_for

        ctx.pop_scope()                                                    # pop if
        assert ctx.lx.next_ptr == wm_for   # C reclaimed, B/A live

        ctx.pop_scope()                                                    # pop for
        assert ctx.lx.next_ptr == wm_fn    # B reclaimed, A live

    def test_legitimate_overflow_still_raises(self):
        """The fix must not hide genuine LX exhaustion within a single scope."""
        ctx = _make_context(lx_size_mb=0.125)  # 128 KB cap

        # set_value on a Tile whose size would exceed capacity raises MemoryError.
        with pytest.raises(MemoryError, match="LX scratchpad overflow"):
            for i in range(100):
                data = np.zeros((1024,), dtype=np.float16)  # 2 KB each
                tile = Tile(data, "f16", data.shape)
                MemoryOps._write_to_lx(ctx, data)
                ctx.set_value(f"%t{i}", tile)  # eventually exceeds 128 KB

    def test_clear_values_resets_watermark_stack(self):
        """clear_values must also clear the watermark stack."""
        ctx = _make_context()
        ctx.push_scope()
        ctx.push_scope()
        assert len(ctx._lx_next_ptr_stack) == 2
        ctx.clear_values()
        assert ctx._lx_next_ptr_stack == []
        assert ctx.lx.next_ptr == 0
        assert ctx.lx.used == 0


# ---------------------------------------------------------------------------
# Consume-on-last-use
# ---------------------------------------------------------------------------

class TestConsumeOnLastUse:
    """consume_last_use: free a Tile at its last fetch instead of at scope exit.

    Motivating pattern — a binary op where both inputs are single-use:

        %a = ktdp.load ...          # tensor<16x16xf16> = 512 bytes
        %b = ktdp.load ...          # tensor<16x16xf16> = 512 bytes
        %r = arith.addf %a, %b     # handler fetches %a then %b, then registers %r

    Without consume_last_use (_LX_DEDUP):
        after set_value("%a"):       lx.used = 512
        after set_value("%b"):       lx.used = 1024
        get_value("%a") — no free:   lx.used = 1024
        get_value("%b") — no free:   lx.used = 1024
        set_value("%r", result):     lx.used = 1536   ← all three live simultaneously

    With consume_last_use (_LX_FULL), use_counts = {%a:1, %b:1}:
        after set_value("%a"):       lx.used = 512
        after set_value("%b"):       lx.used = 1024
        get_value("%a") — freed:     lx.used =  512   ← %a gone before handler returns
        get_value("%b") — freed:     lx.used =    0   ← %b gone before result lands
        set_value("%r", result):     lx.used =  512   ← net: replaced two inputs with one output

    Guard: only the topmost scope is eligible for early-free.  An outer-scope
    name with use_count==1 that is read N times inside a loop is NOT consumed
    on first fetch — the topmost-scope check (scope is _scope_stack[-1]) blocks it.
    """

    def test_single_use_tile_lx_used(self):
        """Single-use tile: with consume on, LX is freed at fetch; with consume off, it stays."""
        tile = _make_tile((32, 64))  # 4096 bytes

        ctx_off = _make_context(lx_options=_LX_DEDUP)
        ctx_off.set_value("%a", tile)
        ctx_off._use_counts = {"%a": 1}
        ctx_off.get_value("%a")
        assert ctx_off.lx.used == 4096  # still held — consume off
        assert ctx_off.has_value("%a")

        ctx_on = _make_context(lx_options=_LX_FULL)
        ctx_on.set_value("%a", tile)
        ctx_on._use_counts = {"%a": 1}
        ctx_on.get_value("%a")
        assert ctx_on.lx.used == 0     # freed at fetch — consume on
        assert not ctx_on.has_value("%a")

    def test_peek_does_not_consume(self):
        """peek=True reads without consuming; subsequent normal get_value still consumes.

        Sequence:
            set_value("%a", tile)    lx.used = 512
            get_value("%a", peek=T)  lx.used = 512  — no consume, tile still in scope
            get_value("%a")          lx.used = 0    — consume fires here, not earlier
        """
        tile = _make_tile((16, 16))  # 512 bytes

        ctx = _make_context(lx_options=_LX_FULL)
        ctx.set_value("%a", tile)
        ctx._use_counts = {"%a": 1}

        result = ctx.get_value("%a", peek=True)
        assert result is tile
        assert ctx.lx.used == 512        # peek did not free
        assert ctx.has_value("%a")       # still in scope

        result2 = ctx.get_value("%a")
        assert result2 is tile
        assert ctx.lx.used == 0          # consumed on normal fetch
        assert not ctx.has_value("%a")

    def test_multi_use_tile_not_consumed(self):
        """Multi-use tile (count > 1) is never consumed regardless of consume_last_use."""
        tile = _make_tile((32, 64))  # 4096 bytes

        for opts in (_LX_DEDUP, _LX_FULL):
            ctx = _make_context(lx_options=opts)
            ctx.set_value("%a", tile)
            ctx._use_counts = {"%a": 2}
            ctx.get_value("%a")
            assert ctx.lx.used == 4096
            assert ctx.has_value("%a")

    def test_non_tile_not_consumed(self):
        """Scalars are never consumed regardless of use_count."""
        for opts in (_LX_DEDUP, _LX_FULL):
            ctx = _make_context(lx_options=opts)
            ctx.set_value("%s", 42)
            ctx._use_counts = {"%s": 1}
            assert ctx.get_value("%s") == 42
            assert ctx.has_value("%s")

    def test_net_zero_with_consume_on(self):
        """With consume on: fetch two single-use tiles (lx → 0), register result (lx = N)."""
        ctx = _make_context(lx_options=_LX_FULL)
        tile_a = _make_tile((16, 16))  # 512 bytes
        tile_b = _make_tile((16, 16))
        ctx.set_value("%a", tile_a)
        ctx.set_value("%b", tile_b)
        ctx._use_counts = {"%a": 1, "%b": 1}

        assert ctx.lx.used == 1024
        ctx.get_value("%a")
        assert ctx.lx.used == 512
        ctx.get_value("%b")
        assert ctx.lx.used == 0
        ctx.set_value("%r", _make_tile((16, 16)))
        assert ctx.lx.used == 512

    def test_net_accumulation_with_consume_off(self):
        """With consume off: fetching does not free — lx stays at 1024 after both fetches."""
        ctx = _make_context(lx_options=_LX_DEDUP)
        tile_a = _make_tile((16, 16))  # 512 bytes
        tile_b = _make_tile((16, 16))
        ctx.set_value("%a", tile_a)
        ctx.set_value("%b", tile_b)
        ctx._use_counts = {"%a": 1, "%b": 1}

        ctx.get_value("%a")
        ctx.get_value("%b")
        assert ctx.lx.used == 1024  # both still live


class TestUseCountsParsed:
    """_build_use_counts correctly counts operand occurrences across the whole function.

    The use-count map is a flat dict: SSA name → number of times it appears
    as an operand across all ops and nested regions.  A name with count == 1
    is eligible for consume-on-last-use; count > 1 must never be consumed early.

    Verified against MATMUL_UC_MLIR (8×8 tiles, one scf.for loop):

        // function body
        %accum_zero = arith.constant dense<0.0>            ← defined here
        %c = scf.for ... iter_args(%accum_itr = %accum_zero) {
            // loop body (nested region)
            %c_init  = tensor.empty()                      ← defined here
            %a_dot_b = linalg.matmul ... outs(%c_init)     ← %c_init used here (count=1)
            %accum_next = arith.addf %accum_itr, %a_dot_b  ← %accum_itr used here (count=1)
                                                           ← %a_dot_b used here (count=1)
            scf.yield %accum_next
        }                                                  ← %accum_zero used as iter_init (count=1)

        // after loop
        ktdp.store %c, %c_acc                              ← %pid_m used here

        // %pid_m also appears inside the loop as index for %a_acc
        // → two occurrences across function body + loop region → count >= 2

    Name          Where used                         count   eligible?
    ----------    --------------------------------   -----   ---------
    %accum_zero   scf.for iter_init                    1     yes — freed when loop starts
    %c_init       linalg.matmul outs                   1     yes — freed before %a_dot_b lands
    %accum_itr    arith.addf operand                   1     yes (count) but blocked by
                                                             topmost-scope guard at runtime
    %pid_m        %a_acc index, %c_acc index            2     no  — must not be consumed early
    """

    def _get_use_counts(self):
        from ktir_cpu.parser import KTIRParser
        module = KTIRParser().parse_module(MATMUL_UC_MLIR)
        return module.get_function("matmul_uc_test").use_counts

    def test_gap1_accum_zero_single_use(self):
        """%accum_zero appears once — as the iter_init of scf.for."""
        uc = self._get_use_counts()
        assert uc.get("%accum_zero", 0) == 1

    def test_gap2_c_init_single_use(self):
        """%c_init appears once — as the outs buffer of linalg.matmul."""
        uc = self._get_use_counts()
        assert uc.get("%c_init", 0) == 1

    def test_gap3_accum_itr_single_use(self):
        """%accum_itr appears once — as the left operand of arith.addf."""
        uc = self._get_use_counts()
        assert uc.get("%accum_itr", 0) == 1

    def test_multi_use_names_not_single(self):
        """%pid_m appears twice: once as index for %a_acc inside the loop, once for %c_acc after it."""
        uc = self._get_use_counts()
        assert uc.get("%pid_m", 0) >= 2, (
            f"Expected %pid_m count >= 2, got {uc.get('%pid_m', 0)}"
        )


class TestLastUseMatmulKernel:
    """End-to-end LX accounting for MATMUL_LX_MLIR (issue #130).

    Kernel: BM=512, BN=256, BK=64, K=256 (4 iterations).

    Tile sizes (f16 = 2 bytes/elem):
        %accum_itr / %accum_next  512×256×2 = 262144 bytes  (256 KB)
        %c_init / %a_dot_b        512×256×2 = 262144 bytes  (256 KB)
        %a                        512×64×2  =  65536 bytes  ( 64 KB)
        %b                         64×256×2 =  32768 bytes  ( 32 KB)

    %accum_zero is an arith.constant (no_lx_charge) — a 0-LX literal. The
    scf.for binding materializes it into a fresh, charged %accum_itr buffer
    (256 KB), so the accumulator cost appears at the loop, not the constant.
    (%a_view/%b_view/%c_view are memory-view descriptors, not LX tiles — 0 LX.)

    Per-iteration trace, _LX_DEDUP (consume off):
        parent: %accum_itr  (materialized from %accum_zero)   lx = 262144
        body:
          ktdp.load    → %a            +65536                 lx = 327680
          ktdp.load    → %b            +32768                 lx = 360448
          tensor.empty → %c_init      +262144                 lx = 622592
          linalg.matmul→ %a_dot_b     +262144 (reuses %c_init id, outs)  lx = 622592
          arith.addf   → %accum_next  +262144                 lx = 884736  ← peak
        pop body scope frees %a, %b, %c_init/%a_dot_b, %accum_next; rebind
        %accum_itr = %accum_next for the next iteration.
        peak = 884736 bytes (864 KB)

    Per-iteration trace, _LX_FULL (consume on), use_counts all 1:
        parent: %accum_itr alive                              lx = 262144
        body:
          ktdp.load    → %a            +65536                 lx = 327680
          ktdp.load    → %b            +32768                 lx = 360448
          tensor.empty → %c_init      +262144                 lx = 622592  ← peak
          linalg.matmul fetches %a (last use, topmost) → freed   lx = 557056
                        fetches %b (last use, topmost) → freed    lx = 524288
                        fetches %c_init (last use, topmost)→ freed lx = 262144
                        → %a_dot_b registered (outs id)        lx = 524288
          arith.addf   fetches %a_dot_b (last use)→ freed      lx = 262144
                       fetches %accum_itr (parent scope — guard blocks)
                        → %accum_next registered               lx = 524288
        peak = 622592 bytes (608 KB)

    consume_last_use lowers the peak (864 KB → 608 KB) by freeing %a / %b /
    %c_init at their last fetch instead of at scope exit.

    Remaining limitation — simultaneous window for %accum_itr:
        %accum_itr lives in the parent scope, not the body scope, so the
        topmost-scope guard blocks its early-free; it and %accum_next are both
        live inside each iteration body (hence the 524288 tail in the consume-on
        trace). The set_value rebind after each iteration frees the old
        %accum_itr, but only after %accum_next is charged.
    """

    LX_CAP = 1 * 1024 * 1024  # 1 MB — both modes fit under the const=0 model;
                              # the test asserts the measured peaks, not overflow.

    def _run(self, lx_options):
        """Run the kernel and return the max LX peak observed across cores."""
        from ktir_cpu.interpreter import KTIRInterpreter

        peak = {"v": 0}

        class _PatchedInterpreter(KTIRInterpreter):
            """Applies lx_options and LX cap, and tracks peak lx.used."""
            def _prepare_execution(self, grid_shape):
                super()._prepare_execution(grid_shape)
                for core in self.grid_executor.cores:
                    core.lx.capacity = TestLastUseMatmulKernel.LX_CAP
                    core.lx_options = lx_options
                    _orig = core._charge_lx

                    def _charge(value, _c=core, _o=_orig):
                        _o(value)
                        peak["v"] = max(peak["v"], _c.lx.used)
                    core._charge_lx = _charge

        BM, BN, BK, K = 512, 256, 64, 256
        a = np.random.rand(BM, K).astype(np.float16)
        b = np.random.rand(K, BN).astype(np.float16)
        c = np.zeros((BM, BN), dtype=np.float16)
        interp = _PatchedInterpreter()
        interp.load(MATMUL_LX_MLIR)
        interp.execute_function(
            "matmul_lx_test", a_ptr=a, b_ptr=b, c_ptr=c,
            K=K, BM=BM, BN=BN, BK=BK,
        )
        return peak["v"]

    def test_consume_on_peak(self):
        """consume_last_use on: peak LX = 608 KB (within 1 MB), no overflow."""
        assert self._run(_LX_FULL) == 622592

    def test_consume_off_higher_peak(self):
        """consume_last_use off: peak LX = 864 KB — higher than consume-on, still fits."""
        assert self._run(_LX_DEDUP) == 884736

    def test_addf_loop_body_lx_accounting(self):
        """Unit: fetch two single-use tiles (lx → 0 each step), register result (lx = N)."""
        N = 393216  # 384*512*2 bytes
        ctx = _make_context(lx_options=_LX_FULL)
        ctx.set_value("%accum_itr", _make_tile((384, 512)))
        ctx.set_value("%a_dot_b",   _make_tile((384, 512)))
        ctx._use_counts = {"%accum_itr": 1, "%a_dot_b": 1}

        assert ctx.lx.used == N * 2
        ctx.get_value("%accum_itr")
        assert ctx.lx.used == N
        ctx.get_value("%a_dot_b")
        assert ctx.lx.used == 0
        ctx.set_value("%accum_next", _make_tile((384, 512)))
        assert ctx.lx.used == N


# ---------------------------------------------------------------------------
# Fused matmul / batch_matmul: outs = accumulator iter_arg (issue #135)
# ---------------------------------------------------------------------------

# Fused matmul kernel: linalg.matmul outs(%accum_itr) directly yields the
# result as %accum_next. No separate arith.addf — the accumulator IS the outs.
# BM=64, BN=32, BK=16, K=64 (4 iterations).
# Accumulator = 64*32*2 = 4096 bytes.
# If the handler creates a new Tile, peak = 3*4096 = 12288 bytes (overcounted).
# With in-place fix, peak = 1*4096 + loads = 4096 + 2048 + 1024 = 7168 bytes.
FUSED_MATMUL_MLIR = """
module {
  func.func @fused_matmul_test(
      %a_ptr: index, %b_ptr: index, %c_ptr: index,
      %K: index, %BM: index, %BN: index, %BK: index
  ) attributes {grid = [1]} {
    %pid_m, %pid_n = ktdp.get_compute_tile_id : index, index

    %a_view = ktdp.construct_memory_view %a_ptr, sizes: [64, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x64xf16>

    %b_view = ktdp.construct_memory_view %b_ptr, sizes: [64, 32], strides: [32, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x32xf16>

    %c_view = ktdp.construct_memory_view %c_ptr, sizes: [64, 32], strides: [32, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x32xf16>

    %c0 = arith.constant 0 : index
    %accum_zero = arith.constant dense<0.0> : tensor<64x32xf16>

    %c = scf.for %off_k = %c0 to %K step %BK iter_args(%accum_itr = %accum_zero) -> (tensor<64x32xf16>) {
      %a_acc = ktdp.construct_access_tile %a_view[%pid_m, %off_k] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 15 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<64x64xf16> -> !ktdp.access_tile<64x16xindex>

      %b_acc = ktdp.construct_access_tile %b_view[%off_k, %pid_n] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<64x32xf16> -> !ktdp.access_tile<16x32xindex>

      %a = ktdp.load %a_acc : !ktdp.access_tile<64x16xindex> -> tensor<64x16xf16>
      %b = ktdp.load %b_acc : !ktdp.access_tile<16x32xindex> -> tensor<16x32xf16>

      %accum_next = linalg.matmul ins(%a, %b : tensor<64x16xf16>, tensor<16x32xf16>)
                                  outs(%accum_itr : tensor<64x32xf16>) -> tensor<64x32xf16>

      scf.yield %accum_next : tensor<64x32xf16>
    }

    %c_acc = ktdp.construct_access_tile %c_view[%pid_m, %pid_n] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>

    ktdp.store %c, %c_acc : tensor<64x32xf16>, !ktdp.access_tile<64x32xindex>
    return
  }
}
"""



class TestFusedMatmulLX:
    """Issue #135: fused matmul outs(%accum_itr) must not overcount LX.

    When the handler returns the same Tile object as outs, alias_dedup sees
    one refcount key for %accum_itr and %accum_next — LX is charged once.

    Tile sizes (f16 = 2 bytes/elem):
        accumulator  64×32×2 = 4096 bytes
        %a           64×16×2 = 2048 bytes
        %b           16×32×2 = 1024 bytes

    With in-place fix (dedup on): peak = accum + a + b = 7168 bytes.
    Without fix (3x overcount): peak = 3*4096 + 2048 + 1024 = 15360 bytes.

    LX cap set to 10240 bytes — passes with fix, overflows without.
    """

    LX_CAP = 10240  # 10 KB — fits if accum charged once, overflows if 3x

    def _run_fused_matmul(self, lx_options):
        from ktir_cpu.interpreter import KTIRInterpreter

        cap = TestFusedMatmulLX.LX_CAP

        class _Patched(KTIRInterpreter):
            def _prepare_execution(self, grid_shape):
                super()._prepare_execution(grid_shape)
                for core in self.grid_executor.cores:
                    core.lx.capacity = cap
                    core.lx_options = lx_options

        BM, BN, BK, K = 64, 32, 16, 64
        a = np.random.rand(BM, K).astype(np.float16)
        b = np.random.rand(K, BN).astype(np.float16)
        c = np.zeros((BM, BN), dtype=np.float16)
        interp = _Patched()
        interp.load(FUSED_MATMUL_MLIR)
        interp.execute_function(
            "fused_matmul_test", a_ptr=a, b_ptr=b, c_ptr=c,
            K=K, BM=BM, BN=BN, BK=BK,
        )

    def test_fused_matmul_fits_with_dedup(self):
        """With alias_dedup + in-place fix, fused matmul fits in tight LX."""
        self._run_fused_matmul(_LX_DEDUP)  # must not raise

    def test_fused_matmul_fits_with_full(self):
        """With full LX tracking + in-place fix, fused matmul fits."""
        self._run_fused_matmul(_LX_FULL)  # must not raise


class TestFusedBatchMatmulLX:
    """Issue #135: batch_matmul in-place fix — unit-level LX verification.

    Directly exercises the handler's alias behavior through CoreContext
    without full HBM simulation (3D access tiles are complex to set up).

    Simulates 2 iterations of:
        %accum_next = linalg.batch_matmul ins(%a, %b) outs(%accum_itr)
        scf.yield %accum_next

    With in-place fix: %accum_next is the same object as %accum_itr,
    so alias_dedup charges LX only once for the accumulator.
    """

    def test_batch_matmul_returns_same_object(self):
        """Handler returns the same Tile as outs — alias chain preserved."""
        from ktir_cpu.dialects.linalg_ops import linalg__batch_matmul
        from ktir_cpu.ir_types import Operation

        ctx = _make_context(lx_options=_LX_DEDUP)
        a_tile = Tile(np.ones((2, 4, 3), dtype=np.float16), "f16", (2, 4, 3))
        b_tile = Tile(np.ones((2, 3, 5), dtype=np.float16), "f16", (2, 3, 5))
        acc_tile = Tile(np.zeros((2, 4, 5), dtype=np.float16), "f16", (2, 4, 5))

        ctx.set_value("%a", a_tile)
        ctx.set_value("%b", b_tile)
        ctx.set_value("%acc", acc_tile)

        op = Operation(
            result="%r", op_type="linalg.batch_matmul",
            operands=["%a", "%b", "%acc"], attributes={},
            result_type="tensor<2x4x5xf16>",
            outs_operands=["%acc"],
        )

        result = linalg__batch_matmul(op, ctx, None)
        assert result is acc_tile, "batch_matmul must return same object as outs"

    def test_batch_matmul_lx_not_overcounted(self):
        """Simulated loop: alias_dedup charges accumulator once, not 3x."""
        ctx = _make_context(lx_options=_LX_DEDUP)

        # Simulate: %accum_zero bound in parent scope
        accum = Tile(np.zeros((2, 4, 5), dtype=np.float16), "f16", (2, 4, 5))
        accum_bytes = accum.size_bytes()  # 2*4*5*2 = 80
        ctx.set_value("%accum_itr", accum)
        assert ctx.lx.used == accum_bytes

        # Iteration 1: bind %accum_next to same object (in-place result)
        ctx.set_value("%accum_next", accum)  # same id → refcount++, no new charge
        assert ctx.lx.used == accum_bytes, (
            f"LX overcounted: {ctx.lx.used} != {accum_bytes}"
        )

        # Rebind iter_arg: %accum_itr = %accum_next (same object)
        ctx.set_value("%accum_itr", accum)
        assert ctx.lx.used == accum_bytes

    def test_batch_matmul_correctness(self):
        """batch_matmul produces correct numerical result."""
        from ktir_cpu.dialects.linalg_ops import linalg__batch_matmul
        from ktir_cpu.ir_types import Operation

        ctx = _make_context(lx_options=_LX_DEDUP)
        a_data = np.ones((2, 4, 3), dtype=np.float16) * 2
        b_data = np.ones((2, 3, 5), dtype=np.float16) * 3
        acc_data = np.ones((2, 4, 5), dtype=np.float16)

        ctx.set_value("%a", Tile(a_data, "f16", (2, 4, 3)))
        ctx.set_value("%b", Tile(b_data, "f16", (2, 3, 5)))
        acc_tile = Tile(acc_data, "f16", (2, 4, 5))
        ctx.set_value("%acc", acc_tile)

        op = Operation(
            result="%r", op_type="linalg.batch_matmul",
            operands=["%a", "%b", "%acc"], attributes={},
            result_type="tensor<2x4x5xf16>",
            outs_operands=["%acc"],
        )

        result = linalg__batch_matmul(op, ctx, None)
        # Expected: acc + a @ b = 1 + 2*3*3 = 19 (each element)
        expected = np.full((2, 4, 5), 19.0, dtype=np.float16)
        np.testing.assert_array_equal(result.data, expected)


class TestOutsStructuralAssertion:
    """Verify the structural outs-invariant assertion fires on violations."""

    def test_assertion_catches_new_tile(self):
        """A handler that returns a new Tile instead of mutating outs triggers assert."""
        from ktir_cpu.interpreter import KTIRInterpreter
        from ktir_cpu.ir_types import Operation
        from ktir_cpu.dialects.registry import register, temp_registry

        with temp_registry():
            @register("linalg.batch_matmul", latency_category=None)
            def broken_handler(op, context, env):
                acc = context.get_value(op.operands[2])
                if isinstance(acc, Tile):
                    return Tile(acc.data.copy(), acc.dtype, acc.shape)  # NEW object!
                return Tile(np.zeros((4, 4), dtype=np.float16), "f16", (4, 4))

            op = Operation(
                result="%r",
                op_type="linalg.batch_matmul",
                operands=["%a", "%b", "%c"],
                attributes={},
                result_type="tensor<4x4xf16>",
                outs_operands=["%c"],
            )

            ctx = _make_context(lx_options=_LX_DEDUP)
            ctx.set_value("%a", _make_tile((4, 4)))
            ctx.set_value("%b", _make_tile((4, 4)))
            ctx.set_value("%c", _make_tile((4, 4)))

            interp = KTIRInterpreter()
            interp.load('module { func.func @dummy() attributes {grid = [1]} { return } }')

            with pytest.raises(AssertionError, match="handler returned new Tile"):
                interp._execute_op(op, ctx)

    def test_no_assertion_for_correct_handler(self):
        """A handler that returns the same outs object does not trigger."""
        from ktir_cpu.interpreter import KTIRInterpreter
        from ktir_cpu.ir_types import Operation
        from ktir_cpu.dialects.registry import register, temp_registry

        with temp_registry():
            @register("linalg.matmul", latency_category=None)
            def correct_handler(op, context, env):
                acc = context.get_value(op.operands[2])
                if isinstance(acc, Tile):
                    acc.data += 1.0  # in-place
                    return acc       # same object
                return Tile(np.zeros((4, 4), dtype=np.float16), "f16", (4, 4))

            op = Operation(
                result="%r",
                op_type="linalg.matmul",
                operands=["%a", "%b", "%c"],
                attributes={},
                result_type="tensor<4x4xf16>",
                outs_operands=["%c"],
            )

            ctx = _make_context(lx_options=_LX_DEDUP)
            ctx.set_value("%a", _make_tile((4, 4)))
            ctx.set_value("%b", _make_tile((4, 4)))
            ctx.set_value("%c", _make_tile((4, 4)))

            interp = KTIRInterpreter()
            interp.load('module { func.func @dummy() attributes {grid = [1]} { return } }')
            interp._execute_op(op, ctx)  # must not raise


# Real round-tripped KTIR from the matmul__bmm Triton fixture (dump_round_trip.py).
# Compiled with: B=4, M=128, K=32, N=64, BLOCK_B=1, BLOCK_M=16, BLOCK_K=16, BLOCK_N=16
# %cst is defined once outside all loops; outer scf.for loops over m-tiles and
# n-tiles each run a K-reduction with iter_args(%arg6 = %cst).
# Without the fix, %cst is mutated after the first K-reduction and subsequent
# N-tile iterations accumulate on the corrupted value, producing 64/96/128
# instead of 32 everywhere.
MULTI_TILE_BMM_MLIR = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 + 15 >= 0)>
module {
  func.func @bmm_matmul_kernel(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [32]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x16x16xf32>
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %m_blocks = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32

    %pid = ktdp.get_compute_tile_id : index
    %pid_0 = arith.index_cast %pid : index to i32
    %bm_end = arith.addi %pid_0, %c1_i32 : i32
    %bm_end_1 = arith.minsi %bm_end, %c32_i32 : i32

    %a_desc = ktdp.construct_memory_view %arg0, sizes: [4, 128, 32], strides: [4096, 32, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x128x32xf32>
    %b_desc = ktdp.construct_memory_view %arg1, sizes: [4, 32, 64], strides: [2048, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x32x64xf32>
    %c_desc = ktdp.construct_memory_view %arg2, sizes: [4, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<4x128x64xf32>

    scf.for %arg3 = %pid_0 to %bm_end_1 step %c1_i32  : i32 {
      %b = arith.divsi %arg3, %m_blocks : i32
      %m = arith.remsi %arg3, %m_blocks : i32

      scf.for %arg4 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %acc = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1x16x16xf32>)  : i32 {
          %a_tile = arith.muli %m, %c16_i32 : i32
          %a_tile_2 = arith.muli %arg5, %c16_i32 : i32
          %a_tile_3 = arith.index_cast %b : i32 to index
          %a_tile_4 = arith.index_cast %a_tile : i32 to index
          %a_tile_5 = arith.index_cast %a_tile_2 : i32 to index

          %a_tile_6 = ktdp.construct_access_tile %a_desc[%a_tile_3, %a_tile_4, %a_tile_5] {access_tile_order = #map, access_tile_set = #set3} : memref<4x128x32xf32> -> !ktdp.access_tile<1x16x16xindex>
          %a_tile_7 = ktdp.load %a_tile_6 : <1x16x16xindex> -> tensor<1x16x16xf32>
          %b_tile = arith.muli %arg4, %c16_i32 : i32
          %b_tile_8 = arith.index_cast %b_tile : i32 to index

          %b_tile_9 = ktdp.construct_access_tile %b_desc[%a_tile_3, %a_tile_5, %b_tile_8] {access_tile_order = #map, access_tile_set = #set3} : memref<4x32x64xf32> -> !ktdp.access_tile<1x16x16xindex>
          %b_tile_10 = ktdp.load %b_tile_9 : <1x16x16xindex> -> tensor<1x16x16xf32>
          %acc_11 = linalg.batch_matmul ins(%a_tile_7, %b_tile_10 : tensor<1x16x16xf32>, tensor<1x16x16xf32>) outs(%arg6 : tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
          scf.yield %acc_11 : tensor<1x16x16xf32>
        }
        %0 = arith.muli %m, %c16_i32 : i32
        %1 = arith.muli %arg4, %c16_i32 : i32
        %2 = arith.index_cast %b : i32 to index
        %3 = arith.index_cast %0 : i32 to index
        %4 = arith.index_cast %1 : i32 to index

        %5 = ktdp.construct_access_tile %c_desc[%2, %3, %4] {access_tile_order = #map, access_tile_set = #set3} : memref<4x128x64xf32> -> !ktdp.access_tile<1x16x16xindex>
        ktdp.store %acc, %5 : tensor<1x16x16xf32>, <1x16x16xindex>
      }
    }
    return
  }
}
"""


class TestConstantIterArgMutation:
    """Regression: batch_matmul must not mutate %cst shared across outer loop iterations.

    The real generated bmm KTIR defines %cst once outside all loops and uses it
    as the iter_arg init for every K-reduction loop.  Without the fix, %cst is
    mutated after the first N-tile iteration and subsequent iterations accumulate
    on the corrupted value, producing 64/96/128 instead of 32 everywhere.

    Fix: %cst is an arith.constant (no_lx_charge) — a 0-LX literal. The scf.for
    iter_arg binding materializes it into a fresh, charged accumulator buffer
    per loop entry, so the in-place matmul accumulation never touches the shared
    literal and each outer iteration restarts from a clean zero.
    """

    def test_multi_tile_bmm_correctness(self):
        """Every output tile must equal K=32, not multiples of 32 (cst corrupted)."""
        from ktir_cpu.interpreter import KTIRInterpreter

        # B=4, M=128, K=32, N=64, ones everywhere.
        # Each output tile: A_tile(1x16x16) @ B_tile(1x16x16) with K=2 steps of 16
        # = 16 + 16 = 32.0 everywhere.
        # If %cst is corrupted, tile (m=0,n=1) starts from 32 and gives 64, etc.
        B, M, K, N = 4, 128, 32, 64
        a = np.ones((B, M, K), dtype=np.float32)
        b = np.ones((B, K, N), dtype=np.float32)
        c = np.zeros((B, M, N), dtype=np.float32)

        interp = KTIRInterpreter()
        interp.load(MULTI_TILE_BMM_MLIR)
        result = interp.execute_function("bmm_matmul_kernel", arg0=a, arg1=b, arg2=c)
        out = result["arg2"]

        expected = np.full_like(out, 32.0)
        np.testing.assert_array_equal(out, expected,
            err_msg=f"unique values {np.unique(out)} — expected all 32.0 (cst corrupted?)")

    def test_iter_arg_init_is_distinct_object_from_constant(self):
        """Unit: scf.for must hand the loop a *copy* of a constant init.

        Directly exercises _materialize_iter_inits — the aliasing fix. A
        constant Tile bound with charge=False (an arith.constant literal) must
        be materialized into a *distinct* Tile object for the iter_arg, so a
        later in-place mutation of the iter_arg cannot corrupt the literal.
        """
        from ktir_cpu.dialects.scf_ops import _materialize_iter_inits

        ctx = _make_context()
        cst = _make_tile((16, 16), dtype="f32")
        cst.data[:] = 7.0
        ctx.set_value("%cst", cst, charge=False)   # 0-LX literal, like arith.constant
        assert ctx.lx.used == 0

        init_values = _materialize_iter_inits(ctx, ["%cst"])
        (iter_arg,) = init_values

        # The iter_arg is a different object than the constant ...
        assert iter_arg is not cst, "iter_arg aliases the constant — in-place mutation will corrupt it"
        assert id(iter_arg) not in ctx._uncharged_tiles, "the materialized copy must be a real (chargeable) buffer"

        # ... so mutating it in place (as linalg.matmul outs does) leaves %cst clean.
        iter_arg.data += 1.0
        assert np.all(cst.data == 7.0), "constant literal was corrupted by iter_arg mutation"

    def test_constant_init_is_uncharged_before_materialize(self):
        """Sanity: the constant init really is an uncharged literal pre-copy."""
        from ktir_cpu.dialects.scf_ops import _materialize_iter_inits

        ctx = _make_context()
        cst = _make_tile((32, 32), dtype="f32")  # 4 KB if it were charged
        ctx.set_value("%cst", cst, charge=False)
        assert id(cst) in ctx._uncharged_tiles
        assert ctx.lx.used == 0

        # Materializing charges the fresh copy (a real working buffer), not the literal.
        ctx.set_value("%itr", _materialize_iter_inits(ctx, ["%cst"])[0])
        assert ctx.lx.used == cst.size_bytes()


class TestConstantNoLXCharge:
    """arith.constant tensors are 0-LX literals; consumers pay for real buffers."""

    def test_constant_tensor_costs_zero_lx(self):
        """Binding an arith.constant tensor result must not charge LX."""
        from ktir_cpu.interpreter import KTIRInterpreter

        mlir = """
        module {
          func.func @k(%arg0: index) attributes {grid = [1]} {
            %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
            return
          }
        }
        """
        interp = KTIRInterpreter()
        interp.load(mlir)
        interp.execute_function("k", arg0=np.zeros((1,), dtype=np.float32))
        # The 64x64xf32 constant (16 KB) must not have occupied LX.
        for core in interp.grid_executor.cores:
            assert core.lx.used == 0, (
                f"constant charged LX: core {core.core_id} lx.used={core.lx.used}"
            )

