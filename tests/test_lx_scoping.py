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
memory via push_scope/pop_scope and track_lx/untrack_lx.
"""

import numpy as np
import pytest

from ktir_cpu.grid import CoreContext
from ktir_cpu.memory import LXScratchpad, HBMSimulator
from ktir_cpu.ir_types import Tile
from ktir_cpu.ops.memory_ops import MemoryOps


def _make_context(lx_size_mb: int = 2) -> CoreContext:
    """Create a CoreContext with a fresh LX scratchpad and HBM."""
    lx = LXScratchpad(size_mb=lx_size_mb, core_id=0)
    hbm = HBMSimulator()
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)


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
    """Test that track_lx / untrack_lx correctly manage lx.used."""

    def test_track_increments_used(self):
        ctx = _make_context()
        tile = _make_tile((32, 1024))  # 32*1024*2 = 65536 bytes
        ctx.track_lx("%tile", tile.size_bytes())
        assert ctx.lx.used == 65536

    def test_untrack_decrements_used(self):
        ctx = _make_context()
        tile = _make_tile((32, 1024))
        ctx.track_lx("%tile", tile.size_bytes())
        ctx.untrack_lx("%tile")
        assert ctx.lx.used == 0

    def test_untrack_nonexistent_is_noop(self):
        ctx = _make_context()
        ctx.untrack_lx("%does_not_exist")  # should not raise
        assert ctx.lx.used == 0

    def test_pop_scope_frees_lx(self):
        """Popping a scope frees LX for all Tiles defined in that scope."""
        ctx = _make_context()
        ctx.push_scope()
        tile = _make_tile((32, 1024))  # 65536 bytes
        ctx.set_value("%tile", tile)
        ctx.track_lx("%tile", tile.size_bytes())
        assert ctx.lx.used == 65536

        ctx.pop_scope()
        assert ctx.lx.used == 0

    def test_pop_scope_does_not_free_outer_lx(self):
        """Popping inner scope preserves LX tracked in the outer scope."""
        ctx = _make_context()
        outer_tile = _make_tile((4, 64))  # 512 bytes
        ctx.set_value("%outer", outer_tile)
        ctx.track_lx("%outer", outer_tile.size_bytes())

        ctx.push_scope()
        inner_tile = _make_tile((32, 1024))  # 65536 bytes
        ctx.set_value("%inner", inner_tile)
        ctx.track_lx("%inner", inner_tile.size_bytes())
        assert ctx.lx.used == 512 + 65536

        ctx.pop_scope()
        assert ctx.lx.used == 512  # only outer remains

    def test_lx_overflow_raises(self):
        """Exceeding LX capacity raises MemoryError."""
        ctx = _make_context(lx_size_mb=1)  # 1 MB = 1048576 bytes
        # Two 512KB tiles fit
        ctx.track_lx("%a", 512 * 1024)
        ctx.track_lx("%b", 512 * 1024)
        assert ctx.lx.used == 1048576
        # One more byte should overflow
        with pytest.raises(MemoryError, match="LX scratchpad overflow"):
            ctx.track_lx("%c", 1)

    def test_clear_values_resets_everything(self):
        ctx = _make_context()
        ctx.set_value("%x", 1)
        ctx.push_scope()
        ctx.set_value("%y", 2)
        tile = _make_tile((8, 64))
        ctx.track_lx("%tile", tile.size_bytes())

        ctx.clear_values()
        assert ctx._scope_stack == [{}]
        assert ctx._lx_bytes == {}
        assert ctx.lx.used == 0


# ---------------------------------------------------------------------------
# iter_args simulation
# ---------------------------------------------------------------------------

class TestIterArgsPersistence:
    """Simulate the for_op iter_arg flow: Tiles survive across iterations
    while body-local Tiles are freed each iteration."""

    def test_iter_arg_tiles_persist_body_local_freed(self):
        """Mimics a scf.for with one Tile iter_arg and one body-local Tile.

        Each iteration:
          1. push_scope
          2. create body-local Tile (tracked by _execute_operation)
          3. create new iter_arg Tile (tracked by _execute_operation)
          4. pop_scope (frees both body-local and new iter_arg Tile)
          5. untrack old iter_arg, re-bind + re-track new iter_arg

        After each iteration, lx.used == iter_arg size only.
        """
        ctx = _make_context()

        # Initial iter_arg: tensor<4x1xf16> = 8 bytes
        iter_tile = _make_tile((4, 1))
        ctx.set_value("%acc", iter_tile)
        ctx.track_lx("%acc", iter_tile.size_bytes())
        assert ctx.lx.used == 8

        for i in range(3):
            ctx.push_scope()

            # Body-local: tensor<4x256xf16> = 2048 bytes
            body_tile = _make_tile((4, 256))
            ctx.set_value("%body_tile", body_tile)
            ctx.track_lx("%body_tile", body_tile.size_bytes())

            # New iter_arg value (created in body, will be yielded)
            new_acc = _make_tile((4, 1))
            ctx.set_value("%new_acc", new_acc)
            ctx.track_lx("%new_acc", new_acc.size_bytes())

            assert ctx.lx.used == 8 + 2048 + 8  # old acc + body + new acc

            # pop_scope frees body-local LX (%body_tile AND %new_acc)
            ctx.pop_scope()
            assert ctx.lx.used == 8  # only old %acc remains

            # Re-bind iter_arg: untrack old, set + track new
            ctx.untrack_lx("%acc")
            ctx.set_value("%acc", new_acc)
            ctx.track_lx("%acc", new_acc.size_bytes())
            assert ctx.lx.used == 8  # back to steady state


# ---------------------------------------------------------------------------
# next_ptr rewind on pop_scope (issue #26)
# ---------------------------------------------------------------------------

class TestNextPtrRewind:
    """Regression tests for issue #26: lx.next_ptr must be rewound on
    pop_scope so it stays in lockstep with lx.used.
    """

    def test_issue_26_reproducer_next_ptr_bounded_in_loop(self):
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
            # load op: write into LX, set the SSA value in the scope, track
            # its bytes. pop_scope then frees it via untrack_lx.
            tile_data = np.zeros((4, 256), dtype=np.float16)  # 2 KB
            MemoryOps._write_to_lx(ctx, tile_data)
            name = f"%tile_iter{i}"
            ctx.set_value(name, Tile(tile_data, "f16", tile_data.shape))
            ctx.track_lx(name, tile_data.nbytes)

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

        # Try to track more than capacity in one scope — track_lx should raise.
        with pytest.raises(MemoryError, match="LX scratchpad overflow"):
            for i in range(100):
                data = np.zeros((1024,), dtype=np.float16)  # 2 KB each
                MemoryOps._write_to_lx(ctx, data)
                ctx.track_lx(f"%t{i}", data.nbytes)  # eventually exceeds 128 KB

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
