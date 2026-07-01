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

The fast path handles any indirect access with ≥1 indirect dim + ≥1 direct dim
("block") where the unique-lookup-to-total-points ratio exceeds 16×.

Covers:
  - Gate condition: accepts qualifying patterns, rejects others
  - Load correctness: 1-indirect, 2-indirect, compound idx_exprs, direct_expr
  - Store correctness
  - Equivalence with general path
  - Edge cases
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
    _block_gather_store,
    _resolve_idx_reads,
    _build_indirect_coords,
)
from ktir_cpu.dtypes import bytes_per_elem
from ktir_cpu.parser_ast import parse_affine_map


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
        expr1 = ("const", 0)
        expr2 = ("add", ("ssa", "%bt_idx"), ("dim", 0))
        assert _expr_dependent_vars(expr1) == set()
        assert _expr_dependent_vars(expr2) == {0}


# ---------------------------------------------------------------------------
# Gate condition: _is_block_gather
# ---------------------------------------------------------------------------

class TestBlockGatherGating:
    def test_accepted_1_indirect(self):
        """X[IDX[e], m, n] — 1 indirect + 2 direct, ratio=8192× → accepted."""
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
        assert _is_block_gather(iat) is True

    def test_accepted_2_indirect(self):
        """W[E[e], H[h], m, n] — 2 indirect + 2 direct → accepted."""
        data_memref = MemRef(base_ptr=0, shape=(8, 4, 16, 32),
                             strides=[2048, 512, 32, 1],
                             memory_space="HBM", dtype="f16")
        e_memref = MemRef(base_ptr=5000, shape=(3,), strides=[1],
                          memory_space="HBM", dtype="i32")
        h_memref = MemRef(base_ptr=6000, shape=(2,), strides=[1],
                          memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
            {"kind": "direct", "var_index": 2},
            {"kind": "direct", "var_index": 3},
        ]
        vss = BoxSet(lo=(0, 0, 0, 0), hi=(3, 2, 16, 32))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(3, 2, 16, 32),
            dim_subscripts=dim_subscripts, index_views=[e_memref, h_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        # unique=3*2=6, total=3*2*16*32=3072, ratio=512× → qualifies
        assert _is_block_gather(iat) is True

    def test_rejected_no_direct_dims(self):
        """X[IDX1[i], IDX2[j]] — all indirect, no block → rejected."""
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

    def test_rejected_low_ratio(self):
        """X[IDX[e], col] with ratio=4× < 16× threshold → rejected."""
        x_memref = MemRef(base_ptr=0, shape=(16, 4), strides=[4, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=1000, shape=(16,), strides=[1],
                            memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        vss = BoxSet(lo=(0, 0), hi=(16, 4))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(16, 4),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat) is False


# ---------------------------------------------------------------------------
# Load correctness
# ---------------------------------------------------------------------------

class TestBlockGatherLoad:
    """Fast-path load produces correct data across all supported patterns."""

    def test_moe_1i_2d(self):
        """X[IDX[e], M, N] — 8 experts from 128×8×16 weight tensor."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        num_experts, M, N = 128, 8, 16
        x_data = np.random.randn(num_experts * M * N).astype(np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        selected = np.array([0, 15, 33, 64, 77, 99, 111, 127], dtype=np.int32)
        idx_stick = hbm.allocate(selected.nbytes)
        hbm.write(idx_stick, selected)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(num_experts, M, N),
                          strides=[M * N, N, 1], memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(8,), strides=[1],
                            memory_space="HBM", dtype="i32")

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
        assert _is_block_gather(iat)
        tile = MemoryOps.indirect_load(ctx, iat)

        expected = x_data.reshape(num_experts, M, N)[selected, :, :]
        np.testing.assert_array_equal(tile.data, expected)
        assert tile.index_unique_sticks == 1  # 8 i32 elements = 32 bytes < STICK_BYTES

    def test_paged_attn_compound_idx(self):
        """cache[BT[0, d0], d1, d2, d3] — compound idx_exprs, 1i + 3d."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        n_pages, n_heads, block_size, head_dim = 8, 4, 2, 16
        cache_data = np.arange(n_pages * n_heads * block_size * head_dim, dtype=np.float16)
        cache_stick = hbm.allocate(cache_data.nbytes)
        hbm.write(cache_stick, cache_data)
        cache_base_ptr = (cache_stick * HBMSimulator.STICK_BYTES) // bpe_f16

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
        bt_memref = MemRef(base_ptr=bt_base_ptr, shape=(1, 4), strides=[4, 1],
                           memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("const", 0), ("dim", 0)]},
            {"kind": "direct", "var_index": 1},
            {"kind": "direct", "var_index": 2},
            {"kind": "direct", "var_index": 3},
        ]
        vss = BoxSet(lo=(0, 0, 0, 0), hi=(4, n_heads, block_size, head_dim))
        iat = IndirectAccessTile(
            parent_ref=cache_memref,
            shape=(4, n_heads, block_size, head_dim),
            dim_subscripts=dim_subscripts, index_views=[bt_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat)
        tile = MemoryOps.indirect_load(ctx, iat)

        cache_arr = cache_data.reshape(n_pages, n_heads, block_size, head_dim)
        expected = cache_arr[bt_data, :, :, :]
        np.testing.assert_array_equal(tile.data, expected)

    def test_sparse_attn_2i_1d(self):
        """cache[page_idx[b], token_idx[t], d] — 2 indirect + 1 direct."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe = bytes_per_elem("f16")
        bpe_idx = bytes_per_elem("i32")

        n_pages, n_tokens, hidden = 8, 6, 32
        n_sel_p, n_sel_t = 4, 3

        data = np.arange(n_pages * n_tokens * hidden, dtype=np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        base_ptr = (stick * HBMSimulator.STICK_BYTES) // bpe

        page_sel = np.sort(np.random.choice(n_pages, n_sel_p, replace=False)).astype(np.int32)
        ps = hbm.allocate(page_sel.nbytes)
        hbm.write(ps, page_sel)
        page_ptr = (ps * HBMSimulator.STICK_BYTES) // bpe_idx

        token_sel = np.sort(np.random.choice(n_tokens, n_sel_t, replace=False)).astype(np.int32)
        ts = hbm.allocate(token_sel.nbytes)
        hbm.write(ts, token_sel)
        token_ptr = (ts * HBMSimulator.STICK_BYTES) // bpe_idx

        data_memref = MemRef(base_ptr=base_ptr, shape=(n_pages, n_tokens, hidden),
                             strides=[n_tokens * hidden, hidden, 1],
                             memory_space="HBM", dtype="f16")
        page_memref = MemRef(base_ptr=page_ptr, shape=(n_sel_p,), strides=[1],
                             memory_space="HBM", dtype="i32")
        token_memref = MemRef(base_ptr=token_ptr, shape=(n_sel_t,), strides=[1],
                              memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
            {"kind": "direct", "var_index": 2},
        ]
        vss = BoxSet(lo=(0, 0, 0), hi=(n_sel_p, n_sel_t, hidden))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(n_sel_p, n_sel_t, hidden),
            dim_subscripts=dim_subscripts, index_views=[page_memref, token_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat)
        tile = _block_gather_load(ctx, iat)

        full = data.reshape(n_pages, n_tokens, hidden)
        expected = full[np.ix_(page_sel, token_sel, np.arange(hidden))]
        np.testing.assert_array_equal(tile.data, expected)

    def test_multi_head_2i_2d(self):
        """W[E[e], H[h], m, n] — 2 indirect + 2 direct."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe = bytes_per_elem("f16")
        bpe_idx = bytes_per_elem("i32")
        n_exp, n_h, M, N = 8, 4, 16, 32
        n_sel_e, n_sel_h = 3, 2

        data = np.arange(n_exp * n_h * M * N, dtype=np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        base = (stick * HBMSimulator.STICK_BYTES) // bpe

        e_sel = np.array([1, 3, 7], dtype=np.int32)
        es = hbm.allocate(e_sel.nbytes)
        hbm.write(es, e_sel)
        e_ptr = (es * HBMSimulator.STICK_BYTES) // bpe_idx

        h_sel = np.array([0, 3], dtype=np.int32)
        hs = hbm.allocate(h_sel.nbytes)
        hbm.write(hs, h_sel)
        h_ptr = (hs * HBMSimulator.STICK_BYTES) // bpe_idx

        data_memref = MemRef(base_ptr=base, shape=(n_exp, n_h, M, N),
                             strides=[n_h * M * N, M * N, N, 1],
                             memory_space="HBM", dtype="f16")
        e_memref = MemRef(base_ptr=e_ptr, shape=(n_sel_e,), strides=[1],
                          memory_space="HBM", dtype="i32")
        h_memref = MemRef(base_ptr=h_ptr, shape=(n_sel_h,), strides=[1],
                          memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
            {"kind": "direct", "var_index": 2},
            {"kind": "direct", "var_index": 3},
        ]
        vss = BoxSet(lo=(0, 0, 0, 0), hi=(n_sel_e, n_sel_h, M, N))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(n_sel_e, n_sel_h, M, N),
            dim_subscripts=dim_subscripts, index_views=[e_memref, h_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        tile = _block_gather_load(ctx, iat)

        full = data.reshape(n_exp, n_h, M, N)
        expected = full[np.ix_(e_sel, h_sel, np.arange(M), np.arange(N))]
        np.testing.assert_array_equal(tile.data, expected)
        assert tile.index_unique_sticks == 2  # e_sel: 3 i32 → 1 stick; h_sel: 2 i32 → 1 stick

    def test_direct_expr(self):
        """X[IDX[e], (2*m+1)] — indirect + direct_expr, ratio=60× → qualifies."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        x_data = np.arange(64 * 128, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data)
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        idx_data = np.array([3, 7, 50, 63], dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(64, 128), strides=[128, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(4,), strides=[1],
                            memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct_expr", "subscript": ("add", ("mul", 2, ("dim", 1)), ("const", 1))},
        ]
        vss = BoxSet(lo=(0, 0), hi=(4, 60))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4, 60),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat)
        tile = MemoryOps.indirect_load(ctx, iat)

        x_arr = x_data.reshape(64, 128)
        expected = np.zeros((4, 60), dtype=np.float16)
        for e in range(4):
            for m in range(60):
                expected[e, m] = x_arr[idx_data[e], 2 * m + 1]
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Store correctness
# ---------------------------------------------------------------------------

class TestBlockGatherStore:
    def test_scatter_write(self):
        """W[E[e], H[h], m, n] = tile — verifies scatter writes back correctly."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe = bytes_per_elem("f16")
        bpe_idx = bytes_per_elem("i32")
        n_exp, n_h, M, N = 8, 4, 16, 32
        n_sel_e, n_sel_h = 3, 2

        data = np.zeros(n_exp * n_h * M * N, dtype=np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        base = (stick * HBMSimulator.STICK_BYTES) // bpe

        e_sel = np.array([1, 3, 7], dtype=np.int32)
        es = hbm.allocate(e_sel.nbytes)
        hbm.write(es, e_sel)
        e_ptr = (es * HBMSimulator.STICK_BYTES) // bpe_idx

        h_sel = np.array([0, 3], dtype=np.int32)
        hs = hbm.allocate(h_sel.nbytes)
        hbm.write(hs, h_sel)
        h_ptr = (hs * HBMSimulator.STICK_BYTES) // bpe_idx

        data_memref = MemRef(base_ptr=base, shape=(n_exp, n_h, M, N),
                             strides=[n_h * M * N, M * N, N, 1],
                             memory_space="HBM", dtype="f16")
        e_memref = MemRef(base_ptr=e_ptr, shape=(n_sel_e,), strides=[1],
                          memory_space="HBM", dtype="i32")
        h_memref = MemRef(base_ptr=h_ptr, shape=(n_sel_h,), strides=[1],
                          memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
            {"kind": "direct", "var_index": 2},
            {"kind": "direct", "var_index": 3},
        ]
        vss = BoxSet(lo=(0, 0, 0, 0), hi=(n_sel_e, n_sel_h, M, N))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(n_sel_e, n_sel_h, M, N),
            dim_subscripts=dim_subscripts, index_views=[e_memref, h_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        write_data = np.ones((n_sel_e, n_sel_h, M, N), dtype=np.float16) * 42.0
        write_tile = Tile(write_data, "f16", write_data.shape, 0)
        _block_gather_store(ctx, write_tile, iat)

        full = hbm.read(stick, n_exp * n_h * M * N, "f16").reshape(n_exp, n_h, M, N)
        for ei in e_sel:
            for hi in h_sel:
                np.testing.assert_array_equal(full[ei, hi], 42.0)
        for ei in range(n_exp):
            for hi in range(n_h):
                if ei not in e_sel or hi not in h_sel:
                    np.testing.assert_array_equal(full[ei, hi], 0.0)


# ---------------------------------------------------------------------------
# Equivalence: fast path == general path
# ---------------------------------------------------------------------------

class TestBlockGatherMatchesGeneral:
    def test_fast_equals_general(self):
        """Fast path result is bit-exact with general inspector-executor."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe = bytes_per_elem("f16")
        bpe_idx = bytes_per_elem("i32")

        n_pages, n_tokens, hidden = 8, 6, 32
        n_sel_p, n_sel_t = 4, 3

        data = np.arange(n_pages * n_tokens * hidden, dtype=np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        base_ptr = (stick * HBMSimulator.STICK_BYTES) // bpe

        page_sel = np.sort(np.random.choice(n_pages, n_sel_p, replace=False)).astype(np.int32)
        ps = hbm.allocate(page_sel.nbytes)
        hbm.write(ps, page_sel)
        page_ptr = (ps * HBMSimulator.STICK_BYTES) // bpe_idx

        token_sel = np.sort(np.random.choice(n_tokens, n_sel_t, replace=False)).astype(np.int32)
        ts = hbm.allocate(token_sel.nbytes)
        hbm.write(ts, token_sel)
        token_ptr = (ts * HBMSimulator.STICK_BYTES) // bpe_idx

        data_memref = MemRef(base_ptr=base_ptr, shape=(n_pages, n_tokens, hidden),
                             strides=[n_tokens * hidden, hidden, 1],
                             memory_space="HBM", dtype="f16")
        page_memref = MemRef(base_ptr=page_ptr, shape=(n_sel_p,), strides=[1],
                             memory_space="HBM", dtype="i32")
        token_memref = MemRef(base_ptr=token_ptr, shape=(n_sel_t,), strides=[1],
                              memory_space="HBM", dtype="i32")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
            {"kind": "direct", "var_index": 2},
        ]
        vss = BoxSet(lo=(0, 0, 0), hi=(n_sel_p, n_sel_t, hidden))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(n_sel_p, n_sel_t, hidden),
            dim_subscripts=dim_subscripts, index_views=[page_memref, token_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        fast_tile = _block_gather_load(ctx, iat)

        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        idx_values, _ = _resolve_idx_reads(ctx, iat)
        coords = _build_indirect_coords(iat, idx_values)
        general_tile = MemoryOps.load(ctx, iat.parent_ref.to_tile_ref(),
                                       coords=coords, result_shape=iat.shape)

        np.testing.assert_array_equal(fast_tile.data, general_tile.data)


# ---------------------------------------------------------------------------
# Non-identity variables_space_order: gate still accepts, fallback is correct
# ---------------------------------------------------------------------------

class TestBlockGatherPermutedVSO:
    """_is_block_gather returns True for a permuted variables_space_order,
    and the load result matches the general inspector-executor path.

    Issue #96 notes that permuted-VSO block-gathers miss the broadcast fast
    path (they fall through to _block_gather_offsets_fallback via the guard at
    memory_ops.py:396-402). This class confirms:

      (a) _is_block_gather still returns True — _block_gather_analyze does not
          gate on VSO, so the dispatch guard in indirect_load routes permuted-
          VSO IATs into the block-gather branch rather than the general path.
      (b) _block_gather_load produces bit-exact results vs the general
          inspector-executor path, confirming the fallback enumerates points in
          the correct permuted order.
    """

    def test_permuted_vso_qualifies_and_matches_general(self):
        """W[E[e], n] with vso=(d1,d0) swaps iteration order; result matches general path."""
        ctx = _make_context()
        hbm = ctx.hbm
        bpe = bytes_per_elem("f16")
        bpe_idx = bytes_per_elem("i32")

        n_exp, N, n_sel_e = 64, 128, 4  # ratio=128× > 16× ✓
        data = np.arange(n_exp * N, dtype=np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)
        base = (stick * HBMSimulator.STICK_BYTES) // bpe

        e_sel = np.array([5, 17, 42, 63], dtype=np.int32)
        es = hbm.allocate(e_sel.nbytes)
        hbm.write(es, e_sel)
        e_ptr = (es * HBMSimulator.STICK_BYTES) // bpe_idx

        data_memref = MemRef(base_ptr=base, shape=(n_exp, N), strides=[N, 1],
                             memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=e_ptr, shape=(n_sel_e,), strides=[1],
                            memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        vss = BoxSet(lo=(0, 0), hi=(n_sel_e, N))
        vso = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        # double-check we're set up the right way
        assert not vso.is_identity()
        assert vso.is_permutation()

        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(n_sel_e, N),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=vso,
        )

        # (a) gate: _block_gather_analyze does not check VSO, so this must be True
        assert _is_block_gather(iat) is True

        # (b) fast path (routes through _block_gather_offsets_fallback via VSO guard)
        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        fast_tile = _block_gather_load(ctx, iat)

        # general inspector-executor path (also uses _enumerate_in_vso_order)
        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        idx_values, _ = _resolve_idx_reads(ctx, iat)
        coords = _build_indirect_coords(iat, idx_values)
        general_tile = MemoryOps.load(ctx, iat.parent_ref.to_tile_ref(),
                                      coords=coords, result_shape=iat.shape)

        np.testing.assert_array_equal(fast_tile.data, general_tile.data)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBlockGatherEdgeCases:
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
        vss = BoxSet(lo=(0, 0), hi=(0, 4))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(0, 4),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        tile = MemoryOps.indirect_load(ctx, iat)
        assert tile.data.size == 0


# ---------------------------------------------------------------------------
# index_unique_sticks: populated by _block_gather_load
# ---------------------------------------------------------------------------

class TestBlockGatherIndexUniqueSticks:
    """_block_gather_load sets Tile.index_unique_sticks with the HBM stick
    count for index-tensor reads, which the latency estimator uses to account
    for index-side memory traffic separately from data-side traffic.
    """

    def test_multi_stick_index_read(self):
        """33 i32 index elements (132 bytes) cross a stick boundary → index_unique_sticks == 2.

        With STICK_BYTES=128 and bpe_i32=4: addresses e*4 for e in 0..32 span
        bytes 0..128. Byte 128 lands on the next stick, so the set has 2 entries.

        Note: STICK_BYTES=128 is a fixed hardware constant
        on HBMSimulator (not exposed in HardwareConfig), and index views are unconstrained in
        size — there is no spec rule or interpreter limit that caps them at one stick.
        MoE top-k routing with k > 32 selected experts, for example, produces an i32 index
        tensor larger than 128 bytes. The block-gather 16× threshold actually favours such
        large index sets, so this path is exercised by production-scale kernels.
        """
        ctx = _make_context()
        hbm = ctx.hbm
        bpe_f16 = bytes_per_elem("f16")
        bpe_i32 = bytes_per_elem("i32")

        # 33 indirect * 32 direct = 1056 total, unique=33, ratio=32× > 16× → qualifies
        num_experts, M = 256, 32
        x_data = np.arange(num_experts * M, dtype=np.float16)
        x_stick = hbm.allocate(x_data.nbytes)
        hbm.write(x_stick, x_data.ravel())
        x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_f16

        # 33 * 4 = 132 bytes: elements 0-31 in stick N, element 32 in stick N+1
        idx_data = np.arange(33, dtype=np.int32)
        idx_stick = hbm.allocate(idx_data.nbytes)
        hbm.write(idx_stick, idx_data)
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_i32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(num_experts, M),
                          strides=[M, 1], memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(33,), strides=[1],
                            memory_space="HBM", dtype="i32")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
            {"kind": "direct", "var_index": 1},
        ]
        vss = BoxSet(lo=(0, 0), hi=(33, M))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(33, M),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        assert _is_block_gather(iat)
        tile = _block_gather_load(ctx, iat)

        assert tile.index_unique_sticks == 2
