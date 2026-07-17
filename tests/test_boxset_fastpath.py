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

"""BoxSet rectangular fast-path tests for ktdp.load/store.

Covers issue #52: parity between the BoxSet sub-TileRef shortcut on
direct ktdp.load/store and the per-coord scatter that the AffineSet
slow path uses.
"""

from unittest.mock import patch

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from ktir_cpu.affine import BoxSet
from ktir_cpu.dialects.ktdp_ops import ktdp__load
from ktir_cpu.dtypes import bytes_per_elem
from ktir_cpu.grid import CoreContext
from ktir_cpu.ir_types import AccessTile, MemRef, Tile
from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.ops.memory_ops import MemoryOps
from ktir_cpu.parser_ast import parse_affine_map


def _make_ctx():
    hbm = HBMSimulator()
    lx = LXScratchpad(core_id=0)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm), hbm


# ---------------------------------------------------------------------------
# Sub-TileRef shortcut: _subtile_ref + MemoryOps.load/store
# ---------------------------------------------------------------------------

class TestSubTileRefShortcut:
    """The BoxSet branch in ktdp.load/store builds a sub-TileRef via
    ``MemoryOps._subtile_ref`` and delegates to ``MemoryOps.load`` /
    ``MemoryOps.store`` — the same composition ``MemoryOps.distributed_load``
    /``distributed_store`` use for partition shards.  These tests pin
    that composition at the helper level for the non-distributed parent.

    Row-major no-trample is omitted — column-packed strictly contains it
    (stride-inheritance bugs cannot fail col-packed and pass row-major).
    Full-rectangle correctness lives in ``TestKtdpDispatchTakesBoxSetPath``
    to avoid duplicating the happy-path assertion at two layers.
    """

    def test_sub_rectangle_load_returns_box_extent(self):
        """``_subtile_ref + MemoryOps.load`` on a translated sub-box
        returns just the box contents (in row-major order), not the
        full parent.  Sole unit-level coverage for the read direction;
        the store direction is covered by the col-packed no-trample
        test."""
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        elem_idx = stick * HBMSimulator.STICK_BYTES // bytes_per_elem("f16")
        tile_ref = MemRef(
            base_ptr=elem_idx, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        box = BoxSet(lo=(1, 1), hi=(3, 3))
        sub_ref = MemoryOps._subtile_ref(tile_ref, box)
        loaded = MemoryOps.load(ctx, sub_ref)
        assert loaded.shape == (2, 2)
        np.testing.assert_array_equal(loaded.data, data[1:3, 1:3])

    def test_store_col_packed_does_not_trample(self):
        """Column-packed strides=[1, R]: stride inheritance must come
        from the parent so the sub-tile lands at the right byte
        positions in column-major memory.  Bug here would scatter the
        write across columns and trample sentinels."""
        ctx, hbm = _make_ctx()
        ROWS, COLS = 4, 4
        SENTINEL = np.float16(-3.0)
        elems = ROWS * COLS
        stick = hbm.allocate(elems * bytes_per_elem("f16"))
        hbm.write(stick, np.full(elems, SENTINEL, dtype=np.float16))
        elem_idx = stick * HBMSimulator.STICK_BYTES // bytes_per_elem("f16")

        # Column-packed: strides=[1, ROWS] means consecutive elements
        # along axis 0 are 1 element apart in memory, and consecutive
        # elements along axis 1 are ROWS elements apart.
        tile_ref = MemRef(
            base_ptr=elem_idx, shape=(ROWS, COLS), strides=[1, ROWS],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        box = BoxSet(lo=(1, 1), hi=(3, 3))
        payload = np.arange(1, 5, dtype=np.float16).reshape(2, 2)
        sub_ref = MemoryOps._subtile_ref(tile_ref, box)
        MemoryOps.store(ctx, Tile(payload, "f16", (2, 2)), sub_ref)

        # Reconstruct logical view from column-packed bytes.
        flat = hbm.read(stick, elems, "f16")
        logical = np.empty((ROWS, COLS), dtype=np.float16)
        for r in range(ROWS):
            for c in range(COLS):
                logical[r, c] = flat[r * 1 + c * ROWS]
        np.testing.assert_array_equal(logical[1:3, 1:3], payload)
        mask = np.zeros((ROWS, COLS), dtype=bool)
        mask[1:3, 1:3] = True
        outside = logical[~mask]
        assert np.all(outside == SENTINEL), (
            "Column-packed sub-tile store wrote outside the box — "
            "stride inheritance bug."
        )

    def test_empty_box(self):
        """``hi[d] <= lo[d]`` on any axis: ``_subtile_ref`` produces a
        TileRef with a zero-extent axis; ``MemoryOps.load`` /
        ``MemoryOps.store`` then handle the degenerate shape without
        fabricating an offset pointer or trampling the parent.  One
        test, two assertions — same failure mode."""
        ctx, hbm = _make_ctx()
        SENTINEL = np.float16(-1.0)
        elems = 16
        stick = hbm.allocate(elems * bytes_per_elem("f16"))
        hbm.write(stick, np.full(elems, SENTINEL, dtype=np.float16))
        elem_idx = stick * HBMSimulator.STICK_BYTES // bytes_per_elem("f16")

        tile_ref = MemRef(
            base_ptr=elem_idx, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        box = BoxSet(lo=(2, 2), hi=(2, 4))   # empty on axis 0
        sub_ref = MemoryOps._subtile_ref(tile_ref, box)
        loaded = MemoryOps.load(ctx, sub_ref)
        assert loaded.data.size == 0
        assert loaded.data.shape == (0, 2)

        empty_tile = Tile(np.zeros((0, 2), dtype=np.float16), "f16", (0, 2))
        sticks = MemoryOps.store(ctx, empty_tile, sub_ref)
        assert sticks == 0
        assert np.all(hbm.read(stick, elems, "f16") == SENTINEL)


# ---------------------------------------------------------------------------
# ktdp.load / ktdp.store wiring + structural fast-path verification
# ---------------------------------------------------------------------------

_VECTOR_ADD_MLIR = """
module {
  func.func @add() attributes {grid = [1, 1]} {
    %X_addr = arith.constant 0 : index
    %Y_addr = arith.constant 64 : index
    %Z_addr = arith.constant 128 : index

    %X = ktdp.construct_memory_view %X_addr, sizes: [8], strides: [1] {
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %Y = ktdp.construct_memory_view %Y_addr, sizes: [8], strides: [1] {
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %Z = ktdp.construct_memory_view %Z_addr, sizes: [8], strides: [1] {
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>

    %c0 = arith.constant 0 : index
    %x_at = ktdp.construct_access_tile %X[%c0] {
        access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %y_at = ktdp.construct_access_tile %Y[%c0] {
        access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %z_at = ktdp.construct_access_tile %Z[%c0] {
        access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>

    %x = ktdp.load %x_at : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %y = ktdp.load %y_at : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %z = arith.addf %x, %y : tensor<8xf16>
    ktdp.store %z, %z_at : tensor<8xf16>, !ktdp.access_tile<8xindex>
    return
  }
}
"""


class TestKtdpDispatchTakesBoxSetPath:
    """End-to-end tests via ktdp.load/store handlers."""

    def test_full_rectangle_via_ktdp_load(self):
        """End-to-end vector add: BoxSet([0, shape)) (the new sentinel)
        must (a) take the structural sub-TileRef shortcut — verified
        by spying on ``_flat_memory_offsets`` — and (b) compute the
        right values.  Two assertions, one fixture: structural fast
        path + happy-path correctness.  Replaces a separate unit-level
        full-rectangle parity test (the helper-level layer is
        sufficiently exercised by sub-rectangle and col-packed tests).
        """
        interp = KTIRInterpreter()
        interp.load(_VECTOR_ADD_MLIR)

        x_in = np.arange(8, dtype=np.float16)
        y_in = np.arange(8, 16, dtype=np.float16)

        seen = []
        real = MemoryOps._flat_memory_offsets

        def _spy(*args, **kwargs):
            seen.append(args)
            return real(*args, **kwargs)

        with patch.object(MemoryOps, "_flat_memory_offsets",
                          staticmethod(_spy)):
            _orig = interp._prepare_execution
            def _seed(grid_shape):
                _orig(grid_shape)
                hbm = interp.memory.hbm
                hbm.write(0, x_in)   # stick 0 = elem idx 0
                hbm.write(1, y_in)   # stick 1 = elem idx 64
                hbm.write(2, np.zeros(8, dtype=np.float16))  # stick 2 = elem idx 128
            interp._prepare_execution = _seed
            interp.execute_function("add")

        assert seen == [], (
            f"BoxSet fast path was bypassed: _flat_memory_offsets was "
            f"called {len(seen)} time(s) — expected 0 because every "
            f"access tile in this fixture is rectangular."
        )
        z_out = interp.memory.hbm.read(2, 8, "f16")  # read back from stick 2
        np.testing.assert_array_equal(z_out, x_in + y_in)

    def test_parser_lowers_full_rect_affine_set_to_boxset(self):
        """``parse_affine_set`` lowers an axis-aligned full-rectangle
        affine set to ``BoxSet`` automatically, so explicit
        ``access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>``
        in the fixture comes out as ``BoxSet([0, 8))`` and routes
        through the fast path — no separate normalisation needed."""
        interp = KTIRInterpreter()
        interp.load(_VECTOR_ADD_MLIR)
        func = interp.module.functions["add"]
        access_tile_ops = [
            op for op in func.operations
            if op.op_type == "ktdp.construct_access_tile"
        ]
        assert access_tile_ops, "no construct_access_tile op in fixture"
        for op in access_tile_ops:
            cs = op.attributes.get("coordinate_set")
            assert isinstance(cs, BoxSet), (
                f"coordinate_set should be BoxSet (parse_affine_set "
                f"lowers axis-aligned sets), got {type(cs).__name__}"
            )
            assert cs.lo == (0,) * len(op.attributes["shape"])
            assert tuple(cs.hi) == tuple(op.attributes["shape"])


class TestSlowPathFallbackOnNonIdentityCoordinateOrder:
    """All other fast-path tests verify that the BoxSet branch is
    taken; this one verifies the negative direction — when the guard
    rejects the access (non-identity ``coordinate_order``), execution
    must fall through to the coord-list slow path that calls
    ``_flat_memory_offsets``.  Without this, the guard could silently
    widen (e.g. ``cso is None or True``) and no test would catch it.
    """

    def test_non_identity_order_calls_flat_memory_offsets(self):
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        elem_idx = stick * HBMSimulator.STICK_BYTES // bytes_per_elem("f16")
        parent_ref = MemRef(
            base_ptr=elem_idx, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        # Transposing AffineMap — non-identity, so the guard rejects
        # the BoxSet fast path and must use the coord-list scatter.
        transpose = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert not transpose.is_identity()
        access_tile = AccessTile(
            parent_ref=parent_ref, shape=(4, 4),
            base_map=parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>"),
            coordinate_set=BoxSet(lo=(0, 0), hi=(4, 4)),
            coordinate_order=transpose,
        )


        class _Op:
            attributes = {"_result_shape": (4, 4)}
            operands = ["%a"]

        class _Ctx:
            def __init__(self, inner, at):
                self._inner, self._at = inner, at
            def get_value(self, _):
                return self._at
            def __getattr__(self, n):
                return getattr(self._inner, n)

        seen = []
        real = MemoryOps._flat_memory_offsets

        def _spy(*args, **kwargs):
            seen.append(args)
            return real(*args, **kwargs)

        with patch.object(MemoryOps, "_flat_memory_offsets",
                          staticmethod(_spy)):
            ktdp__load(_Op(), _Ctx(ctx, access_tile), env=None)

        assert len(seen) >= 1, (
            "Non-identity coordinate_order must route to the slow path "
            "and call _flat_memory_offsets — guard symmetry check."
        )


class TestExplicitIdentityCoordinateOrder:
    """The dialect handler's guard is ``cso is None or cso.is_identity()``,
    not ``cso is None`` alone — so a non-None identity AffineMap that
    bypasses the parser-side normalisation still triggers the BoxSet
    fast path.  Verifies the guard symmetry across the parser↔handler
    boundary."""

    def test_identity_affine_map_takes_fast_path(self):
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        elem_idx = stick * HBMSimulator.STICK_BYTES // bytes_per_elem("f16")
        parent_ref = MemRef(
            base_ptr=elem_idx, shape=(4, 4), strides=[4, 1],
            memory_space="HBM", dtype="f16",
        ).to_tile_ref()

        # Construct an AccessTile directly with a non-None identity map,
        # bypassing the parser's identity → None normalisation.
        identity_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        assert identity_map.is_identity()
        access_tile = AccessTile(
            parent_ref=parent_ref,
            shape=(4, 4),
            base_map=identity_map,
            coordinate_set=BoxSet(lo=(0, 0), hi=(4, 4)),
            coordinate_order=identity_map,
        )

        # Drive the load handler directly.

        class _Op:
            attributes = {"_result_shape": (4, 4)}
            operands = ["%a"]

        seen = []
        real = MemoryOps._flat_memory_offsets

        def _spy(*args, **kwargs):
            seen.append(args)
            return real(*args, **kwargs)

        class _Ctx:
            def __init__(self, inner, at):
                self._inner = inner
                self._at = at
            def get_value(self, _ssa):
                return self._at
            def __getattr__(self, n):
                return getattr(self._inner, n)

        with patch.object(MemoryOps, "_flat_memory_offsets",
                          staticmethod(_spy)):
            tile = ktdp__load(_Op(), _Ctx(ctx, access_tile), env=None)

        np.testing.assert_array_equal(tile.data, data)
        assert seen == [], (
            "Explicit identity coordinate_order should still trigger the "
            "BoxSet fast path (guard: cso is None or cso.is_identity())."
        )


