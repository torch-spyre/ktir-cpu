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
    """Test that set_value auto-tracking correctly manages lx.used."""

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

    def test_alias_does_not_double_count(self):
        """Two names bound to the same Tile object charge LX only once."""
        ctx = _make_context()
        tile = _make_tile((32, 1024))
        ctx.set_value("%a", tile)
        ctx.set_value("%b", tile)  # alias — same id(tile)
        assert ctx.lx.used == 65536

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
