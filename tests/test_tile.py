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

"""Tests for AccessTile and TileRef IR structure.

Verifies that affine attributes are correctly parsed and preserved on
AccessTile and TileRef objects produced by construct_access_tile and
construct_memory_view, and that tile_access computes correct offsets
via AffineMap evaluation.
"""

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from ktir_cpu.affine import AffineMap
from ktir_cpu.grid import CoreContext
from ktir_cpu.ir_types import TileRef
from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.ops.memory_ops import MemoryOps
from ktir_cpu.parser_ast import parse_affine_map

from conftest import get_test_params


def _make_ctx():
    hbm = HBMSimulator()
    lx = LXScratchpad(core_id=0)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm), hbm


class TestTileAccess:
    """Tests for MemoryOps (ktir_cpu/ops/memory_ops.py): tile_access offset
    computation, load/store with contiguous, strided, and coordinate-set paths.

    How TileRef shape/strides are obtained
    ---------------------------------------
    In the MLIR dialect (see examples/rfc/indirect-access-copy.mlir), a
    memory view is constructed as:

        %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #Y_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xf16>

    The parser copies sizes → shape and strides verbatim into a TileRef.
    So TileRef.shape = (64, 64) and TileRef.strides = [64, 1].

    How tile_access is called
    -------------------------
    A direct access tile is constructed as:

        %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0] {
            access_tile_set = #Y_coord_set,
            access_tile_order = #var_space_order
        } : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>

    The runtime handler calls:

        tile_access(ctx, parent_ref=%Y_view,
                    indices=[0, 0],
                    access_shape=(64, 64),   # from !ktdp.access_tile<64x64xindex>
                    base_map=identity_map)   # synthesised; no base_map in MLIR above

    base_map.eval([0, 0]) → (0, 0), so offset = 0*64 + 0*1 = 0 — the
    returned TileRef has base_ptr = Y_addr and shape = (64, 64).
    """

    def test_identity_map_matches_direct_stride(self):
        """Identity base_map produces same offset as the old sum(idx*stride) formula.

        Mirrors the Y_access_tile in indirect-access-copy.mlir but with a
        smaller 4×4 parent and a non-zero starting index so the offset is
        non-trivial:

            parent: sizes [4, 4], strides [4, 1]   (row-major 4×4 f16 block)
            indices: [1, 2]                          (row 1, col 2)
            base_map: (d0, d1) -> (d0, d1)           (identity — synthesised)

            base_coords = (1, 2)
            offset      = 1*4 + 2*1 = 6 elements = 12 bytes (f16 = 2 bytes each)
        """
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        parent = TileRef(base_ptr=ptr, shape=(4, 4), strides=[4, 1], memory_space="HBM", dtype="f16")
        identity = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")

        ref = MemoryOps.tile_access(ctx, parent, indices=[1, 2], access_shape=(2, 2), base_map=identity)
        # row 1, col 2 → offset = 1*4 + 2*1 = 6 elements = 12 bytes
        assert ref.base_ptr == ptr + 6 * 2

    def test_non_identity_map_transposed_access(self):
        """Swapped-dimension map (d0,d1)->(d1,d0) gives a transposed base pointer.

        A non-identity base_map can express a coordinate permutation.  With
        indices [1, 2] and the swapped map the base coordinates become (2, 1)
        instead of (1, 2), so the resulting pointer lands at a different row:

            parent: sizes [4, 4], strides [4, 1]
            indices: [1, 2]
            base_map: (d0, d1) -> (d1, d0)           (swap row/col)

            base_coords = (2, 1)
            offset      = 2*4 + 1*1 = 9 elements = 18 bytes
        """
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        parent = TileRef(base_ptr=ptr, shape=(4, 4), strides=[4, 1], memory_space="HBM", dtype="f16")
        swapped = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")

        ref = MemoryOps.tile_access(ctx, parent, indices=[1, 2], access_shape=(1, 1), base_map=swapped)
        # base_coords = (d1, d0) = (2, 1) → offset = 2*4 + 1*1 = 9 elements = 18 bytes
        assert ref.base_ptr == ptr + 9 * 2

    def test_scaled_map(self):
        """Affine map with a scale factor (tiling stride) computes the correct offset.

        A scaled map is used when tile indices address non-unit-stride blocks
        in the parent — for example, when each tile covers every other row.
        The access tile index (1, 3) refers to row 2 (= 1*2) in the parent:

            parent: sizes [8, 8], strides [8, 1]
            indices: [1, 3]
            base_map: (d0, d1) -> (d0 * 2, d1)      (outer dim scaled by 2)

            base_coords = (2, 3)
            offset      = 2*8 + 3*1 = 19 elements = 38 bytes
        """
        ctx, hbm = _make_ctx()
        data = np.arange(64, dtype=np.float16)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        parent = TileRef(base_ptr=ptr, shape=(8, 8), strides=[8, 1], memory_space="HBM", dtype="f16")
        scaled = parse_affine_map("affine_map<(d0, d1) -> (d0 * 2, d1)>")

        ref = MemoryOps.tile_access(ctx, parent, indices=[1, 3], access_shape=(1, 1), base_map=scaled)
        # base_coords = (2, 3) → offset = 2*8 + 3*1 = 19 elements = 38 bytes
        assert ref.base_ptr == ptr + 19 * 2

    def test_access_shape_preserved(self):
        """tile_access returns a TileRef whose shape is the requested access_shape.

        access_shape comes from the result type of the construct_access_tile op,
        e.g. !ktdp.access_tile<2x3xindex> → access_shape = (2, 3).  The
        returned TileRef inherits this shape so that the subsequent load knows
        how many elements to read.
        """
        ctx, hbm = _make_ctx()
        ptr = hbm.allocate(32)
        parent = TileRef(base_ptr=ptr, shape=(4, 4), strides=[4, 1], memory_space="HBM", dtype="f16")
        identity = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")

        ref = MemoryOps.tile_access(ctx, parent, indices=[0, 0], access_shape=(2, 3), base_map=identity)
        assert ref.shape == (2, 3)

    def test_load_after_tile_access(self):
        """tile_access + load retrieves the correct sub-tile from HBM.

        End-to-end analogue of the Y_access_tile load in indirect-access-copy.mlir:

            %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0] {...}
            %Y_data = ktdp.load %Y_access_tile

        Here we use a 4×4 parent and access a 2×2 sub-tile starting at row 1:

            parent layout in HBM:
              [[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],   ← row 1 (start of sub-tile)
               [ 8,  9, 10, 11],   ← row 2
               [12, 13, 14, 15]]

            indices=[1, 0], access_shape=(2, 2), identity base_map
            → base_ptr offset = 1*4 + 0*1 = 4 elements
            → loaded sub-tile = [[4, 5], [8, 9]]
        """
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        parent = TileRef(base_ptr=ptr, shape=(4, 4), strides=[4, 1], memory_space="HBM", dtype="f16")
        identity = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")

        sub_ref = MemoryOps.tile_access(ctx, parent, indices=[1, 0], access_shape=(2, 2), base_map=identity)
        tile = MemoryOps.load(ctx, sub_ref)

        expected = np.array([[4, 5], [8, 9]], dtype=np.float16)
        assert np.array_equal(tile.data, expected)

    def test_is_contiguous(self):
        """_is_contiguous returns True for row-major strides, False otherwise."""
        assert MemoryOps._is_contiguous((3, 4), [4, 1])
        assert MemoryOps._is_contiguous((2, 3, 4), [12, 4, 1])
        assert MemoryOps._is_contiguous((5,), [1])
        assert not MemoryOps._is_contiguous((3, 4), [8, 1])   # row stride too large (sub-tile)
        assert not MemoryOps._is_contiguous((3, 4), [4, 2])   # col stride > 1

    def test_strided_load_no_coords(self):
        """Load from a non-contiguous (row-strided) tile without coords uses gather path."""
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        # 2×2 sub-tile with parent row stride 4 — not contiguous
        tile_ref = TileRef(base_ptr=ptr, shape=(2, 2), strides=[4, 1], memory_space="HBM", dtype="f16")
        assert not MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides)

        tile = MemoryOps.load(ctx, tile_ref)
        expected = np.array([[0, 1], [4, 5]], dtype=np.float16)
        assert np.array_equal(tile.data, expected), (
            f"Strided load mismatch:\n  got {tile.data}\n  expected {expected}"
        )

    def test_strided_store_no_coords(self):
        """Store to a non-contiguous (row-strided) tile without coords uses scatter path."""
        from ktir_cpu.ir_types import Tile
        ctx, hbm = _make_ctx()
        data = np.zeros((4, 4), dtype=np.float16)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        tile_ref = TileRef(base_ptr=ptr, shape=(2, 2), strides=[4, 1], memory_space="HBM", dtype="f16")
        assert not MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides)

        patch = np.array([[1, 2], [3, 4]], dtype=np.float16)
        MemoryOps.store(ctx, Tile(patch, "f16", (2, 2)), tile_ref)

        result = hbm.read(ptr, 16, "f16").reshape(4, 4)
        expected = np.array([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.float16)
        assert np.array_equal(result, expected), (
            f"Strided store mismatch:\n  got\n{result}\n  expected\n{expected}"
        )

    def test_non_rectangular_load_store(self):
        """Upper-triangular gather from a 4×4 HBM tile, then scatter back doubled."""
        from ktir_cpu.parser_ast import parse_affine_set
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        tile_ref = TileRef(base_ptr=ptr, shape=(4, 4), strides=[4, 1], memory_space="HBM", dtype="f16")

        # Upper-triangular coordinates: d1 >= d0
        # (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3) — 10 elements
        css = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d1 - d0 >= 0)>"
        )
        coords = css.enumerate((4, 4))
        assert len(coords) == 10

        tile = MemoryOps.load(ctx, tile_ref, coords=coords, result_shape=(10,))
        expected_vals = np.array([data[r, c] for r, c in coords], dtype=np.float16)
        assert np.array_equal(tile.data, expected_vals), (
            f"Gathered values mismatch:\n  got {tile.data}\n  expected {expected_vals}"
        )

        doubled = tile.data * np.float16(2.0)
        doubled_tile = type(tile)(doubled, tile.dtype, tile.shape)
        MemoryOps.store(ctx, doubled_tile, tile_ref, coords=coords)

        result = hbm.read(ptr, 16, "f16").reshape(4, 4)
        for r in range(4):
            for c in range(4):
                if c >= r:
                    assert result[r, c] == np.float16(data[r, c] * 2), (
                        f"Upper-tri [{r},{c}]: expected {data[r,c]*2}, got {result[r,c]}"
                    )
                else:
                    assert result[r, c] == data[r, c], (
                        f"Lower-tri [{r},{c}] should be unchanged: expected {data[r,c]}, got {result[r,c]}"
                    )

    def test_access_tile_order_inverted(self):
        """coordinate_order with d0↔d1 swapped transposes the gather order.

        For a 3×3 tile with values [[0,1,2],[3,4,5],[6,7,8]], applying
        access_tile_order = affine_map<(d0,d1) -> (d1,d0)> means output
        position i gets the element at coord (d1,d0) instead of (d0,d1).

        Row-major coords:  (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
        After cso (swap):  (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)
        Expected values:     0,   3,   6,   1,   4,   7,   2,   5,   8
        (column-major traversal of the original data)
        """
        ctx, hbm = _make_ctx()
        data = np.arange(9, dtype=np.float16).reshape(3, 3)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        tile_ref = TileRef(base_ptr=ptr, shape=(3, 3), strides=[3, 1], memory_space="HBM", dtype="f16")

        coords = list(map(tuple, np.ndindex(3, 3)))  # (0,0)..(2,2)
        cso = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert not cso.is_identity()

        remapped = [cso.eval(pt) for pt in coords]
        tile = MemoryOps.load(ctx, tile_ref, coords=remapped, result_shape=(9,))

        expected = np.array([data[c[0], c[1]] for c in remapped], dtype=np.float16)
        assert np.array_equal(tile.data, expected), (
            f"Transposed gather mismatch:\n  got {tile.data}\n  expected {expected}"
        )


class TestTileOps:
    """Verify AccessTile and TileRef carry their affine attributes after parsing."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("indirect_access_copy"))
    def test_affine_attrs_preserved(self, path, func_name, entry):
        """construct_access_tile ops should carry coordinate_set and
        construct_memory_view ops should carry coordinate_set after parsing.

        indirect-access-copy.mlir uses top-level affine aliases (#X_coord_set,
        #Y_coord_set, etc.) referenced as coordinate_set / access_tile_set
        attributes. Moved from test_spec_gaps.py once affine attribute support was
        implemented.
        """
        interp = KTIRInterpreter()
        interp.load(path)
        func = interp.module.functions[func_name]

        access_tile_ops = [
            op for op in func.operations if op.op_type == "ktdp.construct_access_tile"
        ]
        assert access_tile_ops, "no construct_access_tile op found"
        op0_attrs = access_tile_ops[0].attributes
        # coordinate_set is None when the set covers the full tile (is_full
        # normalises it away for the fast path).  Either way the attr block was
        # parsed, which we verify via base_map being a concrete AffineMap.
        from ktir_cpu.affine import AffineMap
        assert isinstance(op0_attrs.get("base_map"), AffineMap), (
            "base_map missing or not an AffineMap — attr block was not parsed"
        )
        coord_set = op0_attrs.get("coordinate_set")
        assert coord_set is None or hasattr(coord_set, "enumerate"), (
            "coordinate_set present but not an AffineSet"
        )

        memory_view_ops = [
            op for op in func.operations if op.op_type == "ktdp.construct_memory_view"
        ]
        assert memory_view_ops, "no construct_memory_view op found"
        assert memory_view_ops[0].attributes.get("coordinate_set") is not None

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("indirect_access_copy"))
    def test_base_map_always_present(self, path, func_name, entry):
        """Every construct_access_tile op should have a base_map attribute —
        either from MLIR or synthesized as an identity map by the parser."""
        interp = KTIRInterpreter()
        interp.load(path)
        func = interp.module.functions[func_name]

        access_tile_ops = [
            op for op in func.operations if op.op_type == "ktdp.construct_access_tile"
        ]
        assert access_tile_ops, "no construct_access_tile op found"
        for op in access_tile_ops:
            base_map = op.attributes.get("base_map")
            assert base_map is not None, f"base_map missing on {op}"
            from ktir_cpu.affine import AffineMap
            assert isinstance(base_map, AffineMap), f"expected AffineMap, got {type(base_map)}: {base_map}"


class TestTileAccessEdgeCases:
    """Edge-case tests for tile access: 3D tiles, non-contiguous strides,
    and zero-element shapes."""

    def test_3d_tile_access_identity(self):
        """tile_access with a 3D parent and identity base_map computes the correct offset.

        A 3D parent (2x3x4) with row-major strides [12, 4, 1] and indices [1, 1, 2]:

            base_coords = (1, 1, 2)
            offset = 1*12 + 1*4 + 2*1 = 18 elements = 36 bytes (f16)
        """
        ctx, hbm = _make_ctx()
        data = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        parent = TileRef(base_ptr=ptr, shape=(2, 3, 4), strides=[12, 4, 1],
                         memory_space="HBM", dtype="f16")
        identity = parse_affine_map("affine_map<(d0, d1, d2) -> (d0, d1, d2)>")

        ref = MemoryOps.tile_access(ctx, parent, indices=[1, 1, 2],
                                    access_shape=(1, 1, 1), base_map=identity)
        # offset = 1*12 + 1*4 + 2*1 = 18 elements = 36 bytes
        assert ref.base_ptr == ptr + 18 * 2

    def test_3d_tile_load(self):
        """Load a contiguous 3D sub-tile from a 3D parent and verify values.

        Parent is a 2x3x4 block. We access starting at [0, 1, 0] and load a
        1x2x4 sub-tile (rows 1-2 of the second "slice").
        """
        ctx, hbm = _make_ctx()
        data = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        parent = TileRef(base_ptr=ptr, shape=(2, 3, 4), strides=[12, 4, 1],
                         memory_space="HBM", dtype="f16")
        identity = parse_affine_map("affine_map<(d0, d1, d2) -> (d0, d1, d2)>")

        sub_ref = MemoryOps.tile_access(ctx, parent, indices=[0, 1, 0],
                                        access_shape=(1, 2, 4), base_map=identity)
        tile = MemoryOps.load(ctx, sub_ref)

        # Starting at element [0,1,0] = offset 4, contiguous 8 elements
        expected = np.array([[[4, 5, 6, 7], [8, 9, 10, 11]]], dtype=np.float16)
        assert np.array_equal(tile.data, expected)

    def test_is_contiguous_3d(self):
        """_is_contiguous correctly identifies row-major 3D strides."""
        assert MemoryOps._is_contiguous((2, 3, 4), [12, 4, 1])
        # Wrong outermost stride
        assert not MemoryOps._is_contiguous((2, 3, 4), [24, 4, 1])

    def test_non_contiguous_stride_larger_than_extent(self):
        """Load with stride > shape extent gathers non-adjacent rows.

        A 2x2 sub-tile with strides [8, 1] in a 4x4 parent selects every
        other row (rows 0 and 2):

            parent:
              [[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11],
               [12, 13, 14, 15]]

            strides [8, 1] means row stride = 8 (= 2 parent rows apart)
            Gathered: row 0 cols [0,1] and row 2 cols [0,1] = [[0, 1], [8, 9]]
        """
        ctx, hbm = _make_ctx()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        # shape (2, 2) with stride [8, 1]: every-other-row gather
        tile_ref = TileRef(base_ptr=ptr, shape=(2, 2), strides=[8, 1],
                           memory_space="HBM", dtype="f16")
        assert not MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides)

        tile = MemoryOps.load(ctx, tile_ref)
        expected = np.array([[0, 1], [8, 9]], dtype=np.float16)
        assert np.array_equal(tile.data, expected), (
            f"Non-contiguous stride load mismatch:\n  got {tile.data}\n  expected {expected}"
        )

    def test_non_contiguous_store_stride_larger_than_extent(self):
        """Store into non-contiguous locations (stride > extent) scatters correctly."""
        from ktir_cpu.ir_types import Tile
        ctx, hbm = _make_ctx()
        data = np.zeros((4, 4), dtype=np.float16)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        # shape (2, 2) with stride [8, 1]: rows 0 and 2 in the parent
        tile_ref = TileRef(base_ptr=ptr, shape=(2, 2), strides=[8, 1],
                           memory_space="HBM", dtype="f16")
        patch = np.array([[10, 20], [30, 40]], dtype=np.float16)
        MemoryOps.store(ctx, Tile(patch, "f16", (2, 2)), tile_ref)

        result = hbm.read(ptr, 16, "f16").reshape(4, 4)
        expected = np.array([
            [10, 20, 0, 0],
            [0, 0, 0, 0],
            [30, 40, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.float16)
        assert np.array_equal(result, expected)
