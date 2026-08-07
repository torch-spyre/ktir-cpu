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

"""Tests for the blocked-indirect fast path in indirect_load / indirect_store.

Notation: W[E[e], m, n] means "for each e, look up row E[e] from W across
all m, n."  See _analyze_blocked_indirect docstring for full terminology
(index view, dep_var, blocking factor, fast vs general path).

Covers:
  - Classifier gating: accepts qualifying patterns, rejects others
  - Load correctness: 1-indirect, 2-indirect, compound idx_exprs, direct_expr
  - Store correctness
  - Equivalence with general path
  - Shared-view patterns (multiple indirect subs on the same index array)
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
    _analyze_blocked_indirect,
    _resolve_idx_reads,
    _build_indirect_coords,
)
from ktir_cpu.dtypes import bytes_per_elem
from ktir_cpu.parser_ast import parse_affine_map


# ---------------------------------------------------------------------------
# Test helpers
#
# _isub(view_idx, *dims) — build an indirect subscript referencing a view
# _dsub(var_idx)         — build a direct subscript for a variables-space dim
# _make_iat(...)         — assemble a full IAT from parts (parent, shape, subs)
# _alloc_idx(hbm, vals)  — write an i32 index array to HBM, return its MemRef
# ---------------------------------------------------------------------------

_BPE_F16 = bytes_per_elem("f16")
_BPE_I32 = bytes_per_elem("i32")


def _make_context():
    hbm = HBMSimulator()
    lx = LXScratchpad(size_mb=64)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)


def _alloc_hbm(hbm, data, dtype):
    """Allocate in HBM, write data, return element-index base_ptr."""
    bpe = bytes_per_elem(dtype)
    stick = hbm.allocate(data.nbytes)
    hbm.write(stick, data)
    return (stick * HBMSimulator.STICK_BYTES) // bpe, stick


def _alloc_idx(hbm, indices):
    """Allocate i32 index array in HBM, return MemRef."""
    data = np.asarray(indices, dtype=np.int32)
    ptr, _ = _alloc_hbm(hbm, data, "i32")
    return MemRef(base_ptr=ptr, shape=(len(data),), strides=[1],
                  memory_space="HBM", dtype="i32")


def _isub(view_idx, *dims):
    """Indirect subscript: read index_views[view_idx] at given iteration dims."""
    return {"kind": "indirect", "index_view_idx": view_idx,
            "idx_exprs": [("dim", d) for d in dims]}


def _dsub(var_idx):
    """Direct subscript: contiguous range over variables-space dim var_idx."""
    return {"kind": "direct", "var_index": var_idx}


def _make_iat(parent_ref, shape, dim_subscripts, index_views, vso=None):
    """Build an IndirectAccessTile with BoxSet(lo=0, hi=shape)."""
    vss = BoxSet(lo=tuple(0 for _ in shape), hi=shape)
    return IndirectAccessTile(
        parent_ref=parent_ref, shape=shape,
        dim_subscripts=dim_subscripts, index_views=index_views,
        variables_space_set=vss, variables_space_order=vso,
    )


# ---------------------------------------------------------------------------
# _expr_dependent_vars: identifies which iteration dims an index expression
# depends on.  This determines K (number of unique lookups) for the fast path.
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
# Classifier gating: _analyze_blocked_indirect decides whether an IAT
# qualifies for the fast path (returns info tuple) or must fall through
# to the general per-point path (returns None).
# ---------------------------------------------------------------------------

class TestBlockedIndirectGating:
    def test_accepted_1_indirect(self):
        """X[IDX[e], m, n] — 1 indirect + 2 direct, ratio=8192× → accepted."""
        x_memref = MemRef(base_ptr=0, shape=(128, 64, 128), strides=[8192, 128, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=10000, shape=(8,), strides=[1],
                            memory_space="HBM", dtype="i32")
        iat = _make_iat(
            x_memref, (8, 64, 128),
            [_isub(0, 0), _dsub(1), _dsub(2)],
            [idx_memref],
        )
        assert _analyze_blocked_indirect(iat) is not None

    def test_accepted_2_indirect(self):
        """W[E[e], H[h], m, n] — 2 indirect + 2 direct → accepted."""
        data_memref = MemRef(base_ptr=0, shape=(8, 4, 16, 32),
                             strides=[2048, 512, 32, 1],
                             memory_space="HBM", dtype="f16")
        e_memref = MemRef(base_ptr=5000, shape=(3,), strides=[1],
                          memory_space="HBM", dtype="i32")
        h_memref = MemRef(base_ptr=6000, shape=(2,), strides=[1],
                          memory_space="HBM", dtype="i32")
        iat = _make_iat(
            data_memref, (3, 2, 16, 32),
            [_isub(0, 0), _isub(1, 1), _dsub(2), _dsub(3)],
            [e_memref, h_memref],
        )
        # unique=3*2=6, total=3*2*16*32=3072, ratio=512× → qualifies
        assert _analyze_blocked_indirect(iat) is not None

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
        iat = _make_iat(x_memref, (4, 4), dim_subscripts, [idx1_memref, idx2_memref])
        assert _analyze_blocked_indirect(iat) is None

    @pytest.mark.parametrize("unique_rows,direct_cols,expected", [
        (16,  4, False),   # ratio=4×,  well below threshold
        ( 2, 15, False),   # ratio=7.5×, just below threshold (2×16=32 > 30)
        ( 1, 16, True),    # ratio=16×, exactly at threshold (1×16=16 ≤ 16)
    ])
    def test_ratio_threshold(self, unique_rows, direct_cols, expected):
        """X[IDX[e], col] — varies the unique:total ratio around the 16× threshold."""
        x_memref = MemRef(base_ptr=0, shape=(unique_rows, direct_cols),
                          strides=[direct_cols, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=1000, shape=(unique_rows,), strides=[1],
                            memory_space="HBM", dtype="i32")
        iat = _make_iat(
            x_memref, (unique_rows, direct_cols),
            [_isub(0, 0), _dsub(1)],
            [idx_memref],
        )
        assert (_analyze_blocked_indirect(iat) is not None) is expected


# ---------------------------------------------------------------------------
# Load correctness: each test constructs an IAT representing a real workload
# pattern, runs indirect_load (which takes the fast path), and compares the
# result against a manually-computed expected array.
# ---------------------------------------------------------------------------

class TestBlockedIndirectLoad:
    """Fast-path load produces correct data across all supported patterns.

    Pattern notation (test docstrings):
      X[IDX[e], m, n]       → 1 indirect dim (e selects rows), 2 direct (m, n)
      cache[BT[0,d0], ...]  → compound idx_expr (2D view, constant + dim)
      W[E[e], H[h], m, n]   → 2 independent indirect dims + 2 direct
    """

    def test_moe_1i_2d(self):
        """X[IDX[e], M, N] — 8 experts from 128×8×16 weight tensor."""
        ctx = _make_context()
        hbm = ctx.hbm

        num_experts, M, N = 128, 8, 16
        x_data = np.random.randn(num_experts * M * N).astype(np.float16)
        x_base_ptr, _ = _alloc_hbm(hbm, x_data, "f16")

        selected = np.array([0, 15, 33, 64, 77, 99, 111, 127], dtype=np.int32)
        idx_memref = _alloc_idx(hbm, selected)

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(num_experts, M, N),
                          strides=[M * N, N, 1], memory_space="HBM", dtype="f16")
        iat = _make_iat(
            x_memref, (8, M, N),
            [_isub(0, 0), _dsub(1), _dsub(2)],
            [idx_memref],
        )
        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        expected = x_data.reshape(num_experts, M, N)[selected, :, :]
        np.testing.assert_array_equal(tile.data, expected)
        assert tile.index_unique_sticks == 1  # 8 i32 elements = 32 bytes < STICK_BYTES

    def test_paged_attn_compound_idx(self):
        """cache[BT[0, d0], d1, d2, d3] — compound idx_exprs, 1i + 3d."""
        ctx = _make_context()
        hbm = ctx.hbm

        n_pages, n_heads, block_size, head_dim = 8, 4, 2, 16
        cache_data = np.arange(n_pages * n_heads * block_size * head_dim, dtype=np.float16)
        cache_base_ptr, _ = _alloc_hbm(hbm, cache_data, "f16")

        bt_data = np.array([5, 2, 7, 0], dtype=np.int32)
        bt_base_ptr, _ = _alloc_hbm(hbm, bt_data, "i32")

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
            _dsub(1), _dsub(2), _dsub(3),
        ]
        iat = _make_iat(
            cache_memref, (4, n_heads, block_size, head_dim),
            dim_subscripts, [bt_memref],
        )
        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        cache_arr = cache_data.reshape(n_pages, n_heads, block_size, head_dim)
        expected = cache_arr[bt_data, :, :, :]
        np.testing.assert_array_equal(tile.data, expected)

    def test_sparse_attn_2i_1d(self):
        """cache[page_idx[b], token_idx[t], d] — 2 indirect + 1 direct."""
        ctx = _make_context()
        hbm = ctx.hbm

        n_pages, n_tokens, hidden = 8, 6, 32
        n_sel_p, n_sel_t = 4, 3

        data = np.arange(n_pages * n_tokens * hidden, dtype=np.float16)
        base_ptr, _ = _alloc_hbm(hbm, data, "f16")

        page_sel = np.sort(np.random.choice(n_pages, n_sel_p, replace=False)).astype(np.int32)
        page_memref = _alloc_idx(hbm, page_sel)

        token_sel = np.sort(np.random.choice(n_tokens, n_sel_t, replace=False)).astype(np.int32)
        token_memref = _alloc_idx(hbm, token_sel)

        data_memref = MemRef(base_ptr=base_ptr, shape=(n_pages, n_tokens, hidden),
                             strides=[n_tokens * hidden, hidden, 1],
                             memory_space="HBM", dtype="f16")
        iat = _make_iat(
            data_memref, (n_sel_p, n_sel_t, hidden),
            [_isub(0, 0), _isub(1, 1), _dsub(2)],
            [page_memref, token_memref],
        )
        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        full = data.reshape(n_pages, n_tokens, hidden)
        expected = full[np.ix_(page_sel, token_sel, np.arange(hidden))]
        np.testing.assert_array_equal(tile.data, expected)

    def test_multi_head_2i_2d(self):
        """W[E[e], H[h], m, n] — 2 indirect + 2 direct."""
        ctx = _make_context()
        hbm = ctx.hbm
        n_exp, n_h, M, N = 8, 4, 16, 32
        n_sel_e, n_sel_h = 3, 2

        data = np.arange(n_exp * n_h * M * N, dtype=np.float16)
        base, _ = _alloc_hbm(hbm, data, "f16")

        e_sel = np.array([1, 3, 7], dtype=np.int32)
        e_memref = _alloc_idx(hbm, e_sel)

        h_sel = np.array([0, 3], dtype=np.int32)
        h_memref = _alloc_idx(hbm, h_sel)

        data_memref = MemRef(base_ptr=base, shape=(n_exp, n_h, M, N),
                             strides=[n_h * M * N, M * N, N, 1],
                             memory_space="HBM", dtype="f16")
        iat = _make_iat(
            data_memref, (n_sel_e, n_sel_h, M, N),
            [_isub(0, 0), _isub(1, 1), _dsub(2), _dsub(3)],
            [e_memref, h_memref],
        )
        tile = MemoryOps.indirect_load(ctx, iat)

        full = data.reshape(n_exp, n_h, M, N)
        expected = full[np.ix_(e_sel, h_sel, np.arange(M), np.arange(N))]
        np.testing.assert_array_equal(tile.data, expected)
        assert tile.index_unique_sticks == 2  # e_sel: 1 stick; h_sel: 1 stick

    def test_direct_expr(self):
        """X[IDX[e], (2*m+1)] — indirect + direct_expr: rejected by classifier,
        falls through to general path, result still correct."""
        ctx = _make_context()
        hbm = ctx.hbm

        x_data = np.arange(64 * 128, dtype=np.float16)
        x_base_ptr, _ = _alloc_hbm(hbm, x_data, "f16")

        idx_data = np.array([3, 7, 50, 63], dtype=np.int32)
        idx_memref = _alloc_idx(hbm, idx_data)

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(64, 128), strides=[128, 1],
                          memory_space="HBM", dtype="f16")

        dim_subscripts = [
            _isub(0, 0),
            {"kind": "direct_expr", "subscript": ("add", ("mul", 2, ("dim", 1)), ("const", 1))},
        ]
        vss = BoxSet(lo=(0, 0), hi=(4, 60))
        iat = IndirectAccessTile(
            parent_ref=x_memref, shape=(4, 60),
            dim_subscripts=dim_subscripts, index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        # direct_expr causes classifier rejection
        assert _analyze_blocked_indirect(iat) is None
        tile = MemoryOps.indirect_load(ctx, iat)

        x_arr = x_data.reshape(64, 128)
        expected = np.zeros((4, 60), dtype=np.float16)
        for e in range(4):
            for m in range(60):
                expected[e, m] = x_arr[idx_data[e], 2 * m + 1]
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Store correctness: the fast path also handles indirect_store (scatter).
# Writes should land only at the indirectly-selected positions.
# ---------------------------------------------------------------------------

class TestBlockedIndirectStore:
    def test_scatter_write(self):
        """W[E[e], H[h], m, n] = tile — verifies scatter writes back correctly."""
        ctx = _make_context()
        hbm = ctx.hbm
        n_exp, n_h, M, N = 8, 4, 16, 32
        n_sel_e, n_sel_h = 3, 2

        data = np.zeros(n_exp * n_h * M * N, dtype=np.float16)
        base, stick = _alloc_hbm(hbm, data, "f16")

        e_sel = np.array([1, 3, 7], dtype=np.int32)
        e_memref = _alloc_idx(hbm, e_sel)

        h_sel = np.array([0, 3], dtype=np.int32)
        h_memref = _alloc_idx(hbm, h_sel)

        data_memref = MemRef(base_ptr=base, shape=(n_exp, n_h, M, N),
                             strides=[n_h * M * N, M * N, N, 1],
                             memory_space="HBM", dtype="f16")
        iat = _make_iat(
            data_memref, (n_sel_e, n_sel_h, M, N),
            [_isub(0, 0), _isub(1, 1), _dsub(2), _dsub(3)],
            [e_memref, h_memref],
        )

        write_data = np.ones((n_sel_e, n_sel_h, M, N), dtype=np.float16) * 42.0
        write_tile = Tile(write_data, "f16", write_data.shape, 0)
        MemoryOps.indirect_store(ctx, write_tile, iat)

        full = hbm.read(stick, n_exp * n_h * M * N, "f16").reshape(n_exp, n_h, M, N)
        for ei in e_sel:
            for hi in h_sel:
                np.testing.assert_array_equal(full[ei, hi], 42.0)
        for ei in range(n_exp):
            for hi in range(n_h):
                if ei not in e_sel or hi not in h_sel:
                    np.testing.assert_array_equal(full[ei, hi], 0.0)


# ---------------------------------------------------------------------------
# Sparse write primitives: unit tests for the underlying HBM/LX write(offsets=)
# ops (these are the building blocks that indirect_store's fast path uses).
# ---------------------------------------------------------------------------

class TestSparseWrite:
    """write(offsets=) writes only the targeted offsets, leaving the rest untouched."""

    def test_hbm_write_sparse(self):
        """Write into a few elements of a larger HBM allocation via offsets."""
        hbm = HBMSimulator()
        data = np.zeros(64, dtype=np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)

        offsets = np.array([5, 17, 42, 63], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        hbm.write(stick, values, offsets=offsets)

        result = hbm.read(stick, 64, "f16")
        for o, v in zip(offsets, values):
            assert result[o] == v
        untouched = np.delete(np.arange(64), offsets)
        np.testing.assert_array_equal(result[untouched], 0.0)

    def test_lx_write_sparse(self):
        """Write into LX scratchpad allocation via offsets."""
        lx = LXScratchpad(size_mb=1)
        data = np.zeros(32, dtype=np.float16)
        lx.write(0, data)

        offsets = np.array([0, 15, 31], dtype=np.int64)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float16)
        lx.write(0, values, offsets=offsets)

        result = lx.read(0, 32, "f16", offsets=np.arange(32, dtype=np.int64))
        for o, v in zip(offsets, values):
            assert result[o] == v
        untouched = np.delete(np.arange(32), offsets)
        np.testing.assert_array_equal(result[untouched], 0.0)

    def test_write_read_sparse_roundtrip(self):
        """write(offsets=) then read(offsets=) at same offsets roundtrips."""
        hbm = HBMSimulator()
        data = np.random.randn(128).astype(np.float16)
        stick = hbm.allocate(data.nbytes)
        hbm.write(stick, data)

        offsets = np.array([10, 50, 100, 127], dtype=np.int64)
        new_vals = np.array([99.0, 88.0, 77.0, 66.0], dtype=np.float16)
        hbm.write(stick, new_vals, offsets=offsets)

        gathered = hbm.read(stick, len(offsets), "f16", offsets=offsets)
        np.testing.assert_array_equal(gathered, new_vals)


# ---------------------------------------------------------------------------
# Equivalence: the fast path must produce bit-exact results compared to the
# general per-point path.  This is the primary correctness oracle — if the
# fast path ever diverges, this test catches it.
# ---------------------------------------------------------------------------

class TestBlockedIndirectMatchesGeneral:
    def test_fast_equals_general(self):
        """Fast path result is bit-exact with general inspector-executor."""
        ctx = _make_context()
        hbm = ctx.hbm

        n_pages, n_tokens, hidden = 8, 6, 32
        n_sel_p, n_sel_t = 4, 3

        data = np.arange(n_pages * n_tokens * hidden, dtype=np.float16)
        base_ptr, _ = _alloc_hbm(hbm, data, "f16")

        page_sel = np.sort(np.random.choice(n_pages, n_sel_p, replace=False)).astype(np.int32)
        page_memref = _alloc_idx(hbm, page_sel)

        token_sel = np.sort(np.random.choice(n_tokens, n_sel_t, replace=False)).astype(np.int32)
        token_memref = _alloc_idx(hbm, token_sel)

        data_memref = MemRef(base_ptr=base_ptr, shape=(n_pages, n_tokens, hidden),
                             strides=[n_tokens * hidden, hidden, 1],
                             memory_space="HBM", dtype="f16")
        iat = _make_iat(
            data_memref, (n_sel_p, n_sel_t, hidden),
            [_isub(0, 0), _isub(1, 1), _dsub(2)],
            [page_memref, token_memref],
        )

        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        fast_tile = MemoryOps.indirect_load(ctx, iat)

        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        idx_values, _ = _resolve_idx_reads(ctx, iat)
        coords = _build_indirect_coords(iat, idx_values)
        general_tile = MemoryOps.load(ctx, iat.parent_ref.to_tile_ref(),
                                       coords=coords, result_shape=iat.shape)

        np.testing.assert_array_equal(fast_tile.data, general_tile.data)


# ---------------------------------------------------------------------------
# Non-identity VSO (variables_space_order): when the iteration order differs
# from the natural dimension order (e.g., iterating columns-first), the
# meshgrid broadcast would produce wrong results → classifier rejects.
# The general path handles this correctly via explicit per-point evaluation.
# ---------------------------------------------------------------------------

class TestBlockedIndirectPermutedVSO:
    """Permuted VSO is rejected by _analyze_blocked_indirect, falling
    through to the general inspector-executor path.
    """

    def test_permuted_vso_rejected_and_general_path_correct(self):
        """W[E[e], n] with vso=(d1,d0): rejected by classifier, general path correct."""
        ctx = _make_context()
        hbm = ctx.hbm

        n_exp, N, n_sel_e = 64, 128, 4
        data = np.arange(n_exp * N, dtype=np.float16)
        base, _ = _alloc_hbm(hbm, data, "f16")

        e_sel = np.array([5, 17, 42, 63], dtype=np.int32)
        idx_memref = _alloc_idx(hbm, e_sel)

        data_memref = MemRef(base_ptr=base, shape=(n_exp, N), strides=[N, 1],
                             memory_space="HBM", dtype="f16")
        vso = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert not vso.is_identity()

        vss = BoxSet(lo=(0, 0), hi=(n_sel_e, N))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(n_sel_e, N),
            dim_subscripts=[_isub(0, 0), _dsub(1)],
            index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=vso,
        )

        # classifier rejects non-identity VSO
        assert _analyze_blocked_indirect(iat) is None

        # indirect_load falls through to general path
        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        result_tile = MemoryOps.indirect_load(ctx, iat)

        # explicit general path for reference
        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        idx_values, _ = _resolve_idx_reads(ctx, iat)
        coords = _build_indirect_coords(iat, idx_values)
        general_tile = MemoryOps.load(ctx, iat.parent_ref.to_tile_ref(),
                                      coords=coords, result_shape=iat.shape)

        np.testing.assert_array_equal(result_tile.data, general_tile.data)


# ---------------------------------------------------------------------------
# Edge cases: degenerate inputs that should not crash
# ---------------------------------------------------------------------------

class TestBlockedIndirectEdgeCases:
    def test_empty_iteration_space(self):
        """Zero-extent variable space should not crash."""
        ctx = _make_context()
        hbm = ctx.hbm

        x_data = np.arange(64, dtype=np.float16)
        x_base_ptr, _ = _alloc_hbm(hbm, x_data, "f16")

        idx_data = np.array([], dtype=np.int32)
        idx_stick = hbm.allocate(max(idx_data.nbytes, 4))
        idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // _BPE_I32

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(64, 4), strides=[4, 1],
                          memory_space="HBM", dtype="f16")
        idx_memref = MemRef(base_ptr=idx_base_ptr, shape=(0,), strides=[1],
                            memory_space="HBM", dtype="i32")

        iat = _make_iat(
            x_memref, (0, 4),
            [_isub(0, 0), _dsub(1)],
            [idx_memref],
        )
        tile = MemoryOps.indirect_load(ctx, iat)
        assert tile.data.size == 0


# ---------------------------------------------------------------------------
# index_unique_sticks: the fast path counts how many HBM "sticks" (128-byte
# aligned cache lines) the index reads touch.  The latency estimator uses
# this to model index-side memory traffic separately from data traffic.
# ---------------------------------------------------------------------------

class TestBlockedIndirectIndexUniqueSticks:
    """indirect_load populates Tile.index_unique_sticks for the estimator."""

    def test_multi_stick_index_read(self):
        """33 i32 index elements (132 bytes) cross a stick boundary → index_unique_sticks == 2.

        With STICK_BYTES=128 and bpe_i32=4: addresses e*4 for e in 0..32 span
        bytes 0..128. Byte 128 lands on the next stick, so the set has 2 entries.
        """
        ctx = _make_context()
        hbm = ctx.hbm

        # 33 indirect * 32 direct = 1056 total, unique=33, ratio=32× > 16× → qualifies
        num_experts, M = 256, 32
        x_data = np.arange(num_experts * M, dtype=np.float16)
        x_base_ptr, _ = _alloc_hbm(hbm, x_data, "f16")

        # 33 * 4 = 132 bytes: elements 0-31 in stick N, element 32 in stick N+1
        idx_data = np.arange(33, dtype=np.int32)
        idx_memref = _alloc_idx(hbm, idx_data)

        x_memref = MemRef(base_ptr=x_base_ptr, shape=(num_experts, M),
                          strides=[M, 1], memory_space="HBM", dtype="f16")
        iat = _make_iat(
            x_memref, (33, M),
            [_isub(0, 0), _dsub(1)],
            [idx_memref],
        )
        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        assert tile.index_unique_sticks == 2


# ---------------------------------------------------------------------------
# OOB handling: when indirect indices point outside the parent allocation,
# read(offsets=) returns zero (safe default) and write(offsets=) silently
# drops the write.  This prevents crashes from stale or out-of-range index
# arrays.
# ---------------------------------------------------------------------------

class TestSparseOOB:
    """Verify that read(offsets=) zero-pads OOB and write(offsets=) drops them."""

    def test_read_oob_returns_zero(self):
        """Offsets past allocation end return zero."""
        from ktir_cpu.memory import _read_flat
        memory = {0x1000: np.arange(10, dtype=np.float16)}
        offsets = np.array([0, 5, 9, 10, 11], dtype=np.int64)
        result = _read_flat(memory, 0x1000, len(offsets), np.float16, 2, offsets=offsets)
        expected = np.array([0, 5, 9, 0, 0], dtype=np.float16)
        np.testing.assert_array_equal(result, expected)

    def test_read_all_inbounds(self):
        """All-inbounds path returns correct values (no OOB branch)."""
        from ktir_cpu.memory import _read_flat
        memory = {0x1000: np.arange(10, dtype=np.float16)}
        offsets = np.array([0, 3, 7, 9], dtype=np.int64)
        result = _read_flat(memory, 0x1000, len(offsets), np.float16, 2, offsets=offsets)
        expected = np.array([0, 3, 7, 9], dtype=np.float16)
        np.testing.assert_array_equal(result, expected)

    def test_write_oob_dropped(self):
        """OOB offsets are silently dropped; inbounds writes land."""
        from ktir_cpu.memory import _write_flat
        memory = {0x1000: np.zeros(10, dtype=np.float16)}
        data = np.array([99, 88, 77], dtype=np.float16)
        offsets = np.array([0, 10, 5], dtype=np.int64)
        _write_flat(memory, 0x1000, data, offsets=offsets)
        assert memory[0x1000][0] == 99
        assert memory[0x1000][5] == 77
        assert memory[0x1000][1] == 0  # untouched

    def test_write_all_inbounds(self):
        """All-inbounds sparse write lands correctly."""
        from ktir_cpu.memory import _write_flat
        memory = {0x1000: np.zeros(10, dtype=np.float16)}
        data = np.array([11, 22, 33], dtype=np.float16)
        offsets = np.array([1, 4, 8], dtype=np.int64)
        _write_flat(memory, 0x1000, data, offsets=offsets)
        assert memory[0x1000][1] == 11
        assert memory[0x1000][4] == 22
        assert memory[0x1000][8] == 33


# ---------------------------------------------------------------------------
# Shared view: two or more dim_subscripts entries that reference the same
# index_view_idx — i.e., multiple subscriptions reading from one index array,
# possibly with different idx_exprs.  Real examples:
#   - Coordinate table: B is (K,3), access B[i,0], B[i,1], B[i,2] for 3D coords
#   - Shifted window: B is 1D, access B[e] and B[e+1] for adjacent pairs
#   - Diagonal: B[e], B[e] (degenerate — same value used in two parent dims)
#
# These patterns require per-subscription-expression re-keying (not per-view)
# so that each subscript gets its own K-element value array for broadcast.
# ---------------------------------------------------------------------------

class TestBlockedIndirectSharedView:
    """Fast path handles shared views correctly via per-sub re-keying."""

    def test_shared_view_shifted(self):
        """A[B[e], B[e+1], n] — shifted window into 1D index array."""
        ctx = _make_context()
        hbm = ctx.hbm

        n_rows, n_cols, N = 16, 16, 32
        data = np.arange(n_rows * n_cols * N, dtype=np.float16)
        data_ptr, _ = _alloc_hbm(hbm, data, "f16")

        K = 4
        idx_data = np.array([2, 5, 9, 13, 7], dtype=np.int32)
        idx_ptr, _ = _alloc_hbm(hbm, idx_data, "i32")
        idx_memref = MemRef(base_ptr=idx_ptr, shape=(5,), strides=[1],
                            memory_space="HBM", dtype="i32")

        data_memref = MemRef(base_ptr=data_ptr, shape=(n_rows, n_cols, N),
                             strides=[n_cols * N, N, 1],
                             memory_space="HBM", dtype="f16")

        # VSS has 2 dims: dim 0 = dep_var e (K), dim 1 = direct n (N)
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("add", ("dim", 0), ("const", 1))]},
            _dsub(1),
        ]
        iat = _make_iat(data_memref, (K, N), dim_subscripts, [idx_memref])

        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        arr = data.reshape(n_rows, n_cols, N)
        expected = np.zeros((K, N), dtype=np.float16)
        for e in range(K):
            expected[e, :] = arr[idx_data[e], idx_data[e + 1], :]
        np.testing.assert_array_equal(tile.data, expected)

    def test_shared_view_2d_columns(self):
        """A[B[i,0], B[i,1], B[i,2], n] — 2D coord table, different columns."""
        ctx = _make_context()
        hbm = ctx.hbm

        d0, d1, d2, N = 8, 8, 8, 32
        data = np.arange(d0 * d1 * d2 * N, dtype=np.float16)
        data_ptr, _ = _alloc_hbm(hbm, data, "f16")

        K = 3
        coord_table = np.array([[1, 3, 5], [2, 7, 0], [4, 1, 6]], dtype=np.int32)
        coord_ptr, _ = _alloc_hbm(hbm, coord_table, "i32")
        coord_memref = MemRef(base_ptr=coord_ptr, shape=(K, 3), strides=[3, 1],
                              memory_space="HBM", dtype="i32")

        data_memref = MemRef(base_ptr=data_ptr, shape=(d0, d1, d2, N),
                             strides=[d1 * d2 * N, d2 * N, N, 1],
                             memory_space="HBM", dtype="f16")

        # VSS has 2 dims: dim 0 = dep_var i (K), dim 1 = direct n (N)
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0), ("const", 0)]},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0), ("const", 1)]},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0), ("const", 2)]},
            _dsub(1),
        ]
        iat = _make_iat(data_memref, (K, N), dim_subscripts, [coord_memref])

        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        arr = data.reshape(d0, d1, d2, N)
        expected = np.zeros((K, N), dtype=np.float16)
        for i in range(K):
            expected[i, :] = arr[coord_table[i, 0], coord_table[i, 1], coord_table[i, 2], :]
        np.testing.assert_array_equal(tile.data, expected)

    def test_shared_view_same_expr(self):
        """A[B[e], B[e], m, n] — degenerate same-expr diagonal access."""
        ctx = _make_context()
        hbm = ctx.hbm

        n_rows, n_cols, M, N = 8, 8, 4, 16
        data = np.arange(n_rows * n_cols * M * N, dtype=np.float16)
        data_ptr, _ = _alloc_hbm(hbm, data, "f16")

        K = 3
        idx_data = np.array([1, 5, 7], dtype=np.int32)
        idx_memref = _alloc_idx(hbm, idx_data)

        data_memref = MemRef(base_ptr=data_ptr, shape=(n_rows, n_cols, M, N),
                             strides=[n_cols * M * N, M * N, N, 1],
                             memory_space="HBM", dtype="f16")

        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0)]},
            _dsub(1), _dsub(2),
        ]
        iat = _make_iat(data_memref, (K, M, N), dim_subscripts, [idx_memref])

        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        arr = data.reshape(n_rows, n_cols, M, N)
        expected = np.zeros((K, M, N), dtype=np.float16)
        for e in range(K):
            expected[e, :, :] = arr[idx_data[e], idx_data[e], :, :]
        np.testing.assert_array_equal(tile.data, expected)


# ---------------------------------------------------------------------------
# Correctness gaps between fast path and general path.
#
# Three areas where the two paths diverge in capability or behavior:
# 1. Non-zero vss.lo — fast path starts dep-var iteration at lo[d], not 0
# 2. Negative indices — _read_flat(offsets=) wraps silently; upstream guard prevents
# 3. Shared-view below threshold — general path crashes (StopIteration)
# ---------------------------------------------------------------------------

class TestBlockedIndirectCorrectnessGaps:
    """Tests for edge cases where fast path and general path diverge."""

    @staticmethod
    def _build_1d_iat(hbm, num_experts, N, idx_data, lo=None, hi=None):
        """Helper: allocate W[E[e], n] with optional non-zero lo."""
        data = np.arange(num_experts * N, dtype=np.float16)
        base, _ = _alloc_hbm(hbm, data, "f16")
        idx_memref = _alloc_idx(hbm, idx_data)
        data_memref = MemRef(base_ptr=base, shape=(num_experts, N),
                             strides=[N, 1], memory_space="HBM", dtype="f16")
        if lo is None:
            return _make_iat(data_memref, (len(idx_data), N),
                             [_isub(0, 0), _dsub(1)], [idx_memref]), data
        K = hi[0] - lo[0]
        vss = BoxSet(lo=lo, hi=hi)
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(K, N),
            dim_subscripts=[_isub(0, 0), _dsub(1)],
            index_views=[idx_memref],
            variables_space_set=vss, variables_space_order=None,
        )
        return iat, data

    # --- Non-zero vss.lo ---

    def test_nonzero_lo_1d_indirect(self):
        """W[E[e], n] with lo=(2,0) reads indices 2..5, not 0..3."""
        ctx = _make_context()
        N = 16
        idx_data = np.array([10, 3, 7, 20, 15, 1, 28, 5], dtype=np.int32)
        iat, data = self._build_1d_iat(ctx.hbm, 32, N, idx_data,
                                       lo=(2, 0), hi=(6, N))

        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        arr = data.reshape(32, N)
        expected = np.stack([arr[idx_data[e]] for e in range(2, 6)])
        np.testing.assert_array_equal(tile.data, expected)

    def test_nonzero_lo_2d_indirect(self):
        """cache[P[p], T[t], h] with lo=(1,2,0) — two indirect dims with non-zero lo."""
        ctx = _make_context()
        n_pages, n_tokens, H = 16, 8, 16  # H≥16 passes gate 3
        data = np.arange(n_pages * n_tokens * H, dtype=np.float16)
        base, _ = _alloc_hbm(ctx.hbm, data, "f16")

        page_idx = np.array([0, 9, 3, 14, 7, 11], dtype=np.int32)
        token_idx = np.array([1, 5, 0, 7, 2, 6, 3, 4], dtype=np.int32)
        page_memref = _alloc_idx(ctx.hbm, page_idx)
        token_memref = _alloc_idx(ctx.hbm, token_idx)

        data_memref = MemRef(base_ptr=base, shape=(n_pages, n_tokens, H),
                             strides=[n_tokens * H, H, 1],
                             memory_space="HBM", dtype="f16")
        vss = BoxSet(lo=(1, 2, 0), hi=(4, 6, H))
        iat = IndirectAccessTile(
            parent_ref=data_memref, shape=(3, 4, H),
            dim_subscripts=[_isub(0, 0), _isub(1, 1), _dsub(2)],
            index_views=[page_memref, token_memref],
            variables_space_set=vss, variables_space_order=None,
        )

        assert _analyze_blocked_indirect(iat) is not None
        tile = MemoryOps.indirect_load(ctx, iat)

        arr = data.reshape(n_pages, n_tokens, H)
        expected = np.zeros((3, 4, H), dtype=np.float16)
        for pi, p in enumerate(range(1, 4)):
            for ti, t in enumerate(range(2, 6)):
                expected[pi, ti, :] = arr[page_idx[p], token_idx[t], :]
        np.testing.assert_array_equal(tile.data, expected)

    def test_nonzero_lo_store_roundtrip(self):
        """Store with non-zero lo, then load — verifies store uses lo correctly."""
        ctx = _make_context()
        N = 16  # ≥16 passes gate 3
        idx_data = np.array([10, 3, 7, 20, 15, 1], dtype=np.int32)
        iat, _ = self._build_1d_iat(ctx.hbm, 32, N, idx_data,
                                    lo=(2, 0), hi=(5, N))

        assert _analyze_blocked_indirect(iat) is not None
        write_data = np.arange(3 * N, dtype=np.float16).reshape(3, N) + 100
        write_tile = Tile(data=write_data, shape=(3, N), dtype="f16")
        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        MemoryOps.indirect_store(ctx, write_tile, iat)

        ctx.lx.memory.clear()
        ctx.lx.next_ptr = 0
        result = MemoryOps.indirect_load(ctx, iat)
        np.testing.assert_array_equal(result.data, write_data)

    # --- Negative index guard ---

    def test_negative_index_in_idx_array_raises(self):
        """_runtime_read_and_expand_sub_space raises IndexError on idx < 0."""
        ctx = _make_context()
        N = 16  # ≥16 passes gate 3
        idx_data = np.array([3, -1, 7], dtype=np.int32)
        iat, _ = self._build_1d_iat(ctx.hbm, 16, N, idx_data)

        assert _analyze_blocked_indirect(iat) is not None
        with pytest.raises(IndexError, match="negative"):
            MemoryOps.indirect_load(ctx, iat)

    def test_sparse_read_wraps_negative(self):
        """_read_flat(offsets=) wraps negative offsets (NumPy behavior) — no guard."""
        from ktir_cpu.memory import _read_flat
        memory = {0x1000: np.array([10, 20, 30, 40, 50], dtype=np.float16)}
        offsets = np.array([0, -1, 2], dtype=np.int64)
        result = _read_flat(memory, 0x1000, len(offsets), np.float16, 2, offsets=offsets)
        assert result[0] == 10
        assert result[1] == 50  # wrap-around: flat[-1] = last element
        assert result[2] == 30

    # --- Shared-view below threshold (general path handles it correctly) ---

    def test_shared_view_below_threshold_general_path(self):
        """A[B[e], B[e], n]: shared view falls to general path, produces diagonal."""
        ctx = _make_context()
        n_rows, n_cols, N = 8, 8, 2  # N=2 → blocking factor < 16
        data = np.arange(n_rows * n_cols * N, dtype=np.float16)
        data_ptr, _ = _alloc_hbm(ctx.hbm, data, "f16")

        K = 3
        idx_data = np.array([1, 5, 7], dtype=np.int32)
        idx_memref = _alloc_idx(ctx.hbm, idx_data)
        data_memref = MemRef(base_ptr=data_ptr, shape=(n_rows, n_cols, N),
                             strides=[n_cols * N, N, 1],
                             memory_space="HBM", dtype="f16")
        dim_subscripts = [
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0)]},
            {"kind": "indirect", "index_view_idx": 0,
             "idx_exprs": [("dim", 0)]},
            _dsub(1),
        ]
        iat = _make_iat(data_memref, (K, N), dim_subscripts, [idx_memref])

        assert _analyze_blocked_indirect(iat) is None
        tile = MemoryOps.indirect_load(ctx, iat)

        arr = data.reshape(n_rows, n_cols, N)
        expected = np.stack([arr[idx_data[e], idx_data[e]] for e in range(K)])
        np.testing.assert_array_equal(tile.data, expected)

    def test_distinct_view_below_threshold_uses_general_path(self):
        """A[B[e], C[e], n]: distinct views work fine on general path."""
        ctx = _make_context()
        d0, d1, N = 8, 8, 4  # N=4 → blocking factor < 16
        data = np.arange(d0 * d1 * N, dtype=np.float16)
        data_ptr, _ = _alloc_hbm(ctx.hbm, data, "f16")

        idx_b = np.array([1, 5, 7], dtype=np.int32)
        idx_c = np.array([3, 0, 6], dtype=np.int32)
        b_memref = _alloc_idx(ctx.hbm, idx_b)
        c_memref = _alloc_idx(ctx.hbm, idx_c)
        data_memref = MemRef(base_ptr=data_ptr, shape=(d0, d1, N),
                             strides=[d1 * N, N, 1],
                             memory_space="HBM", dtype="f16")
        iat = _make_iat(data_memref, (3, N),
                        [_isub(0, 0), _isub(1, 0), _dsub(1)],
                        [b_memref, c_memref])

        assert _analyze_blocked_indirect(iat) is None
        tile = MemoryOps.indirect_load(ctx, iat)
        arr = data.reshape(d0, d1, N)
        expected = np.stack([arr[idx_b[e], idx_c[e]] for e in range(3)])
        np.testing.assert_array_equal(tile.data, expected)
