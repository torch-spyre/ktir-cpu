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

"""Tests for the block-gather fast path in indirect_load / indirect_store.

Covers:
  - MoE pattern: X[IDX[e], m, n] with small index into large data
  - Paged attention pattern: key_cache[BT[0, bt_idx], d1, d2, d3]
  - 1D indirect with full column gather
  - Correctness: fast path produces identical results to general path
  - Gate condition: _is_block_gather correctly accepts/rejects patterns
  - Functional: end-to-end load and store via the fast path
"""

import numpy as np
import pytest

from ktir_cpu.affine import BoxSet
from ktir_cpu.ir_types import MemRef, IndirectAccessTile, Tile
from ktir_cpu.grid import CoreContext
from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.ops.memory_ops import (
    MemoryOps,
    _expr_dependent_vars,
    _is_block_gather,
    _block_gather_load,
    _resolve_idx_reads,
    _build_indirect_coords,
)
from ktir_cpu.dtypes import bytes_per_elem


# ---------------------------------------------------------------------------
# Helper: build a CoreContext with fresh HBM/LX
# ---------------------------------------------------------------------------

def _make_context():
    hbm = HBMSimulator()
    lx = LXScratchpad(size_mb=64)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)


# ---------------------------------------------------------------------------
# Unit tests: _expr_dependent_vars
# ---------------------------------------------------------------------------

class TestExprDependentVars:
    def test_simple_dim(self):
        assert _expr_dependent_vars(("dim", 0)) == {0}
        assert _expr_dependent_vars(("dim", 2)) == {2}

    def test_const(self):
        assert _expr_dependent_vars(("const", 42)) == set()

    def test_ssa(self):
        assert _expr_dependent_vars(("ssa", "%grid0")) == set()

    def test_add_two_dims(self):
        expr = ("add", ("dim", 0), ("dim", 1))
        assert _expr_dependent_vars(expr) == {0, 1}

    def test_add_dim_const(self):
        expr = ("add", ("ssa", "%c0"), ("dim", 0))
        assert _expr_dependent_vars(expr) == {0}

    def test_floordiv(self):
        expr = ("floordiv", ("dim", 2), 64)
        assert _expr_dependent_vars(expr) == {2}

    def test_mod(self):
        expr = ("mod", ("dim", 2), 64)
        assert _expr_dependent_vars(expr) == {2}

    def test_mul(self):
        expr = ("mul", 4, ("dim", 1))
        assert _expr_dependent_vars(expr) == {1}

    def test_compound_paged_attn(self):
        # block_tables[%b, %tkv floordiv 64] → idx_exprs = [("const",0), ("add", ssa, ("dim",0))]
        expr1 = ("const", 0)
        expr2 = ("add", ("ssa", "%bt_idx"), ("dim", 0))
        assert _expr_dependent_vars(expr1) == set()
        assert _expr_dependent_vars(expr2) == {0}


# ---------------------------------------------------------------------------
# Unit tests: _is_block_gather gate condition
# ---------------------------------------------------------------------------

class TestIsBlockGather:
    def test_moe_pattern(self):
        """MoE: X[IDX[e], m, n] — 8 experts from 128, shape (8, 64, 128)"""
        x_memref = MemRef(base_ptr=0, shape=(128, 64, 128), strides=[8192, 128, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=10000, shape=(8,), strides=[1],
                            memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
            {"kind": "direct", "var_index": 2},
        ]
        vss = BoxSet(lo=(0, 0, 0), hi=(8, 64, 128))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(8, 64, 128),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        # unique=8, total=8*64*128=65536, ratio=8192x → qualifies
        assert _is_block_gather(iat) is True

    def test_paged_attention_pattern(self):
        """Paged attn: cache[BT[0, bt+d0], d1, d2, d3] — 1 page from 64"""
        cache_memref = MemRef(base_ptr=0, shape=(64, 16, 8, 128),
                              strides=[16384, 1024, 128, 1],
                              memory_space="HBM", dtype="f16")
        bt_memref = MemRef(base_ptr=50000, shape=(1, 16), strides=[16, 1],
                           memory_space="HBM", dtype="i32")
        # idx_exprs for paged attn: depends on dim 0 only (but dim 0 range is 1)
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("const", 0), ("dim", 0)]},
            {"kind": "direct", "var_index": 1},
            {"kind": "direct", "var_index": 2},
            {"kind": "direct", "var_index": 3},
        ]
        vss = BoxSet(lo=(0, 0, 0, 0), hi=(1, 16, 8, 128))
        iat = IndirectAccessTile(
            parent_ref=cache_memref, shape=(1, 16, 8, 128),
            dim_subscripts=dim_subscripts, index_views=[bt_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        # unique=1, total=1*16*8*128=16384, ratio=16384x → qualifies
        assert _is_block_gather(iat) is True

    def test_all_indirect_rejected(self):
        """X[IDX1[m,k], IDX2[m,k]] — 2 indirect dims, must be rejected"""
        x_memref = MemRef(base_ptr=0, shape=(4, 4), strides=[4, 1],
                          memory_space="HBM", dtype="f16")
        idx1_memref = MemRef(base_ptr=1000, shape=(4, 4), strides=[4, 1],
                             memory_space="HBM", dtype="i32")
        idx2_memref = MemRef(base_ptr=2000, shape=(4, 4), strides=[4, 1],
                             memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0), ("dim", 1)]},
            {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 0), ("dim", 1)]},
        ]
        vss = BoxSet(lo=(0, 0), hi=(4, 4))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4, 4),
            dim_subscripts=dim_subscripts, index_views=[idx1_memref, idx2_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat) is False

    def test_small_ratio_rejected(self):
        """When unique lookups ≈ total points, general path is correct."""
        x_memref = MemRef(base_ptr=0, shape=(16, 4), strides=[4, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=1000, shape=(16,), strides=[1],
                            memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        # unique=16, total=16*4=64, ratio=4x < 16x threshold → rejected
        vss = BoxSet(lo=(0, 0), hi=(16, 4))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(16, 4),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat) is False


# ---------------------------------------------------------------------------
# Functional tests: MoE gather (load + store)
# ---------------------------------------------------------------------------

class TestMoEGather:
    """MoE pattern: Y[e, m] = X[IDX[e], m] — select 4 rows from a 32×8 matrix."""

    def _setup(self):
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        # X: 32×64 matrix, values = row*64 + col
        n_rows, n_cols = 32, 64
        x_data = np.arange(n_rows * n_cols, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # IDX: [3, 7, 15, 31] — select 4 rows from 32
        idx_data = np.array([3, 7, 15, 31], dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(n_rows, n_cols),
                          strides=[n_cols, 1], memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(4,), strides=[1],
                            memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        vss = BoxSet(lo=(0, 0), hi=(4, n_cols))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4, n_cols),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        return ctx, iat, x_data

    def test_indirect_load_correctness(self):
        """Fast-path load produces correct gathered data."""
        ctx, iat, x_data = self._setup()
        # unique=4, total=4*64=256, ratio=64x → qualifies
        assert _is_block_gather(iat)

        tile = MemoryOps.indirect_load(ctx, iat)

        # Expected: rows 3, 7, 15, 31 from X (32×64)
        expected = x_data.reshape(32, 64)[[3, 7, 15, 31], :]
        np.testing.assert_array_equal(tile.data, expected)

    def test_indirect_load_matches_general_path(self):
        """Fast-path result is identical to what the general path would produce."""
        ctx, iat, x_data = self._setup()

        # Run fast path
        tile_fast = MemoryOps.indirect_load(ctx, iat)

        # Run general path manually (bypass the gate)
        ctx2, iat2, _ = self._setup()
        idx_values, idx_sticks = _resolve_idx_reads(ctx2, iat2)
        coords = _build_indirect_coords(iat2, idx_values)
        tile_general = MemoryOps.load(
            ctx2, iat2.parent_ref.to_tile_ref(),
            coords=coords, result_shape=iat2.shape,
        )

        np.testing.assert_array_equal(tile_fast.data, tile_general.data)

    def test_indirect_store_correctness(self):
        """Fast-path store scatters data correctly."""
        ctx, iat, x_data = self._setup()

        # Create a tile with known data to store (4 rows × 64 cols)
        store_data = np.arange(4 * 64, dtype=np.float16) + 100
        tile = Tile(store_data.reshape(4, 64), "f16", (4, 64))

        sticks = MemoryOps.indirect_store(ctx, tile, iat)

        # Verify: rows 3, 7, 15, 31 in X should now contain store_data
        x_base_byte = iat.parent_ref.byte_address
        x_stick, x_intra = divmod(x_base_byte, HBMSimulator.STICK_BYTES)
        x_out = ctx.hbm.read(x_stick, 32 * 64, "f16", intra_byte=x_intra).reshape(32, 64)

        for i, row_idx in enumerate([3, 7, 15, 31]):
            np.testing.assert_array_equal(
                x_out[row_idx, :], store_data.reshape(4, 64)[i, :],
                err_msg=f"Row {row_idx} mismatch"
            )


# ---------------------------------------------------------------------------
# Functional tests: 1D indirect load (simplest block-gather)
# ---------------------------------------------------------------------------

class TestSimple1DIndirect:
    """Y[e] = X[IDX[e]] — 1D gather from a large vector."""

    def test_1d_gather(self):
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        # X: 1024-element vector
        x_data = np.arange(1024, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # IDX: [10, 500, 999, 0] — 4 elements
        idx_data = np.array([10, 500, 999, 0], dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(1024,), strides=[1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(4,), strides=[1],
                            memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
        ]
        vss = BoxSet(lo=(0,), hi=(4,))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4,),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        # This has unique=4, total=4, ratio=1 → does NOT qualify for fast path
        # (ratio < 16x). Need a bigger ratio.
        assert _is_block_gather(iat) is False

    def test_1d_gather_with_direct_dims(self):
        """Y[e, col] = X[IDX[e], col] — 4 rows from 256, 64 cols."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        x_data = np.arange(256 * 64, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        idx_data = np.array([0, 100, 200, 255], dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(256, 64), strides=[64, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(4,), strides=[1],
                            memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        vss = BoxSet(lo=(0, 0), hi=(4, 64))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4, 64),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        # unique=4, total=4*64=256, ratio=64x → qualifies
        assert _is_block_gather(iat) is True

        tile = MemoryOps.indirect_load(ctx, iat)
        expected = x_data.reshape(256, 64)[[0, 100, 200, 255], :]
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Functional tests: paged attention pattern
# ---------------------------------------------------------------------------

class TestPagedAttention:
    """key_cache[BT[0, bt_idx+d0], d1, d2, d3] — 1 page selected, sub-slice read."""

    def test_paged_attn_load(self):
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        # key_cache: 8 pages × 4 heads × 2 block_size × 16 head_dim
        n_pages, n_heads, block_size, head_dim = 8, 4, 2, 16
        cache_data = np.arange(n_pages * n_heads * block_size * head_dim, dtype=np.float16)
        cache_stick = hbm.allocate(cache_data.nbytes)
        hbm.write(cache_stick, cache_data)
        cache_base_ptr = (cache_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # block_tables: shape (1, 4), values [5, 2, 7, 0] → page mapping
        bt_data = np.array([5, 2, 7, 0], dtype=np.int32)
        bt_stick = hbm.allocate(bt_data.nbytes)
        hbm.write(bt_stick, bt_data)
        bt_base_ptr = (bt_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        cache_memref = MemRef(
            base_ptr=cache_base_ptr,
            shape=(n_pages, n_heads, block_size, head_dim),
            strides=[n_heads * block_size * head_dim, block_size * head_dim, head_dim, 1],
            memory_space="HBM", dtype="f16",
        )
        bt_memref = MemRef(
            base_ptr=bt_base_ptr, shape=(1, 4), strides=[4, 1],
            memory_space="HBM", dtype="i32",
        )

        # Access pattern: cache[BT[0, d0], d1, d2, d3]
        # idx_exprs: [("const", 0), ("dim", 0)] → index into bt[0, d0]
        # We select bt_idx=0..3 (all 4 block_tables entries)
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("const", 0), ("dim", 0)]},
            {"kind": "direct", "var_index": 1},
            {"kind": "direct", "var_index": 2},
            {"kind": "direct", "var_index": 3},
        ]
        # Iterate over (bt_idx=0..3, head=0..3, bs=0..1, hd=0..15)
        vss = BoxSet(lo=(0, 0, 0, 0), hi=(4, n_heads, block_size, head_dim))
        iat = IndirectAccessTile(
            parent_ref=cache_memref,
            shape=(4, n_heads, block_size, head_dim),
            dim_subscripts=dim_subscripts,
            index_views=[bt_memref],
            variables_space_set=vss,
            variables_space_order=None,
        )

        # unique=4, total=4*4*2*16=512, ratio=128x → qualifies
        assert _is_block_gather(iat) is True

        tile = MemoryOps.indirect_load(ctx, iat)

        # Verify: tile[i, h, b, d] = cache_data[bt_data[i], h, b, d]
        cache_arr = cache_data.reshape(n_pages, n_heads, block_size, head_dim)
        expected = cache_arr[bt_data, :, :, :]  # (4, 4, 2, 16)
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Functional test: 3D MoE — 8 experts from 128
# ---------------------------------------------------------------------------

class TestMoE3D:
    """X[IDX[e], M, N] — 8 expert blocks from 128×64×32 weight tensor."""

    def test_large_moe_load(self):
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        num_experts, M, N = 128, 8, 16
        x_data = np.random.randn(num_experts * M * N).astype(np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # Select experts [0, 15, 33, 64, 77, 99, 111, 127]
        selected = np.array([0, 15, 33, 64, 77, 99, 111, 127], dtype=np.int32)
        idx_stick = hbm.allocate(selected.nbytes)
        hbm.write(idx_stick, selected)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(
            base_ptr=x_base_ptr,
            shape=(num_experts, M, N),
            strides=[M * N, N, 1],
            memory_space="HBM", dtype="f16",
        )
        idx_memref = MemRef(
            base_ptr=idx_base_ptr, shape=(8,), strides=[1],
            memory_space="HBM", dtype="i32",
        )

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
            {"kind": "direct", "var_index": 2},
        ]
        vss = BoxSet(lo=(0, 0, 0), hi=(8, M, N))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(8, M, N),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        # unique=8, total=8*8*16=1024, ratio=128x → qualifies
        assert _is_block_gather(iat) is True

        tile = MemoryOps.indirect_load(ctx, iat)

        x_arr = x_data.reshape(num_experts, M, N)
        expected = x_arr[selected, :, :]
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Functional test: direct_expr dim (e.g. paged-tensor with floordiv/mod)
# ---------------------------------------------------------------------------

class TestDirectExprDim:
    """Pattern with direct_expr dims alongside an indirect dim."""

    def test_indirect_with_direct_expr(self):
        """X[IDX[e], (2*m+1)] — indirect dim 0, direct_expr dim 1."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        # X: 16×8
        x_data = np.arange(16 * 8, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # IDX: [2, 5, 10]
        idx_data = np.array([2, 5, 10], dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(16, 8), strides=[8, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(3,), strides=[1],
                            memory_space="HBM", dtype="i32")

        # dim 0: indirect, idx_exprs=[("dim", 0)]
        # dim 1: direct_expr, subscript = 2*m + 1 = ("add", ("mul", 2, ("dim", 1)), ("const", 1))
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct_expr", "subscript": ("add", ("mul", 2, ("dim", 1)), ("const", 1))},
        ]
        # Variables: (e in [0,3), m in [0,3))
        vss = BoxSet(lo=(0, 0), hi=(3, 3))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(3, 3),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        # unique=3, total=3*3=9, ratio=3x < 16x → does NOT qualify
        # But let's verify the general path still works correctly
        assert _is_block_gather(iat) is False

        tile = MemoryOps.indirect_load(ctx, iat)

        # Expected: tile[e, m] = X[IDX[e], 2*m+1]
        x_arr = x_data.reshape(16, 8)
        expected = np.zeros((3, 3), dtype=np.float16)
        for e in range(3):
            for m in range(3):
                expected[e, m] = x_arr[idx_data[e], 2 * m + 1]
        np.testing.assert_array_equal(tile.data, expected)

    def test_indirect_with_direct_expr_fast_path(self):
        """Same pattern but with enough ratio to trigger fast path."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        # X: 64×128
        x_data = np.arange(64 * 128, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # IDX: [3, 7, 50, 63]
        idx_data = np.array([3, 7, 50, 63], dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(64, 128), strides=[128, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(4,), strides=[1],
                            memory_space="HBM", dtype="i32")

        # dim 0: indirect, dim 1: direct_expr = 2*m + 1
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct_expr", "subscript": ("add", ("mul", 2, ("dim", 1)), ("const", 1))},
        ]
        # Variables: (e in [0,4), m in [0,60))
        vss = BoxSet(lo=(0, 0), hi=(4, 60))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4, 60),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        # unique=4, total=4*60=240, ratio=60x → qualifies
        assert _is_block_gather(iat) is True

        tile = MemoryOps.indirect_load(ctx, iat)

        # Verify
        x_arr = x_data.reshape(64, 128)
        expected = np.zeros((4, 60), dtype=np.float16)
        for e in range(4):
            for m in range(60):
                expected[e, m] = x_arr[idx_data[e], 2 * m + 1]
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Edge case: empty iteration space
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_iteration_space(self):
        """Zero-extent variable space should not crash."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        x_data = np.arange(64, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        idx_data = np.array([], dtype=np.int32)
        idx_stick = hbm.allocate(max(idx_data.nbytes, 4))
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(64, 4), strides=[4, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(0,), strides=[1],
                            memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        # Empty range on dim 0
        vss = BoxSet(lo=(0, 0), hi=(0, 4))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(0, 4),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        tile = MemoryOps.indirect_load(ctx, iat)
        assert tile.data.size == 0
