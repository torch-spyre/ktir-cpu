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

"""Unit tests for ktdp.construct_distributed_memory_view.

Structure
---------
- test_distributed_view_copy_rfc   : RFC §C.3 reference example (xfail, per-core LX gap)
- test_distributed_copy            : parametrized table of 2-partition copy cases

The parametrized suite covers all combinations of:
  - memory spaces: HBM/HBM, HBM/LX, LX/HBM
  - partition strides: row-major [R,1] and column-packed [1,C]
  - access shapes: full tile, partial tile (one partition pruned), sub-tile
  - access indices: zero and non-zero (sub-tile spanning both partitions)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from ktir_cpu.dtypes import stick_to_elem_idx
from conftest import get_test_params


# ---------------------------------------------------------------------------
# Shared memory helpers
# ---------------------------------------------------------------------------

def _write_strided(mem, base_ptr: int, block: np.ndarray, strides: List[int]):
    """Write *block* (f16) into *mem* at *base_ptr* (element index) using *strides* (element units).

    Element (i, j) lands at byte offset ``(base_ptr + (i*strides[0] + j*strides[1])) * 2``.
    Holes from non-contiguous layouts are left as zero.
    For HBM, *base_ptr* is converted to a stick index before writing.
    For LX, *base_ptr* is converted to a byte address before writing.
    """
    from ktir_cpu.memory import HBMSimulator
    assert block.dtype == np.float16
    ndim = block.ndim
    assert len(strides) == ndim
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in block.shape], indexing='ij'), axis=-1)
    coords = coords.reshape(-1, ndim)
    offsets = coords @ np.array(strides, dtype=np.int64)
    span = int(offsets.max()) + 1 if offsets.size else 1
    buf = np.zeros(span, dtype=np.float16)
    buf[offsets] = block.flatten()
    if isinstance(mem, HBMSimulator):
        byte_addr = base_ptr * 2  # f16
        stick, intra = divmod(byte_addr, HBMSimulator.STICK_BYTES)
        mem.write(stick, buf, intra_byte=intra)
    else:
        mem.write(base_ptr * 2, buf)


def _get_mem(interp, space: str):
    """Return the memory object for *space* ('HBM' or 'LX')."""
    return interp.memory.hbm if space == "HBM" else interp.memory.get_lx(0)


# ---------------------------------------------------------------------------
# MLIR builder
# ---------------------------------------------------------------------------

def _set_box(r0: int, r1: int, c0: int, c1: int) -> str:
    """affine_set covering [r0,r1] x [c0,c1] (all inclusive)."""
    parts = []
    parts.append(f"d0 - {r0} >= 0" if r0 > 0 else "d0 >= 0")
    parts.append(f"-d0 + {r1} >= 0")
    parts.append(f"d1 - {c0} >= 0" if c0 > 0 else "d1 >= 0")
    parts.append(f"-d1 + {c1} >= 0")
    return f"affine_set<(d0, d1) : ({', '.join(parts)})>"


@dataclass
class PartitionSpec:
    """Describes one rectangular partition of the distributed view.

    Attributes:
        rows         : (first_row, last_row) inclusive in global coords
        cols         : (first_col, last_col) inclusive in global coords
        memory_space : "HBM" or "LX"
        strides      : element strides [row_stride, col_stride]
        base_ptr     : byte address of element [0,0] of this partition
    """
    rows: Tuple[int, int]
    cols: Tuple[int, int]
    memory_space: str
    strides: List[int]
    base_ptr: int

    @property
    def nrows(self) -> int:
        return self.rows[1] - self.rows[0] + 1

    @property
    def ncols(self) -> int:
        return self.cols[1] - self.cols[0] + 1


@dataclass
class DistCopySpec:
    """Full specification for a 2-partition distributed copy test.

    The kernel loads an access tile from distributed A and stores it to
    contiguous HBM output B.  Both are 2-D.

    Attributes:
        global_shape  : logical shape of the full distributed tensor
        p0, p1        : partition specs (disjoint rectangular regions in global coords)
        access_shape  : shape of the access tile
        indices       : [row, col] base indices for construct_access_tile
        out_ptr       : byte address of the B output on HBM
        id            : short human-readable label used in pytest ids
    """
    global_shape: Tuple[int, int]
    p0: PartitionSpec
    p1: PartitionSpec
    access_shape: Tuple[int, int]
    indices: List[int]
    out_ptr: int
    id: str


def _build_mlir(spec: DistCopySpec) -> str:
    """Generate a single-function MLIR module from *spec*."""
    G = spec.global_shape
    p0, p1 = spec.p0, spec.p1
    ac = spec.access_shape
    idx = spec.indices

    p0_set = _set_box(p0.rows[0], p0.rows[1], p0.cols[0], p0.cols[1])
    p1_set = _set_box(p1.rows[0], p1.rows[1], p1.cols[0], p1.cols[1])
    ac_set = _set_box(0, ac[0] - 1, 0, ac[1] - 1)

    idx_decls = "\n".join(
        f"    %idx{i} = arith.constant {v} : index" for i, v in enumerate(idx)
    )
    idx_refs = ", ".join(f"%idx{i}" for i in range(len(idx)))

    p0_ms = f"#ktdp.spyre_memory_space<{p0.memory_space}>"
    p1_ms = f"#ktdp.spyre_memory_space<{p1.memory_space}>"

    return f"""
#P0_set   = {p0_set}
#P1_set   = {p1_set}
#ac_set   = {ac_set}
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {{
  func.func @dist_copy() {{
    %c0 = arith.constant 0 : index
{idx_decls}
    %A0_addr = arith.constant {p0.base_ptr} : index
    %A1_addr = arith.constant {p1.base_ptr} : index
    %B_addr  = arith.constant {spec.out_ptr} : index

    %A0 = ktdp.construct_memory_view %A0_addr, sizes: [{p0.nrows}, {p0.ncols}], strides: [{p0.strides[0]}, {p0.strides[1]}] {{
        coordinate_set = #P0_set, memory_space = {p0_ms}
    }} : memref<{p0.nrows}x{p0.ncols}xf16>

    %A1 = ktdp.construct_memory_view %A1_addr, sizes: [{p1.nrows}, {p1.ncols}], strides: [{p1.strides[0]}, {p1.strides[1]}] {{
        coordinate_set = #P1_set, memory_space = {p1_ms}
    }} : memref<{p1.nrows}x{p1.ncols}xf16>

    %A = ktdp.construct_distributed_memory_view
        (%A0, %A1 : memref<{p0.nrows}x{p0.ncols}xf16>, memref<{p1.nrows}x{p1.ncols}xf16>)
        : memref<{G[0]}x{G[1]}xf16>

    %B = ktdp.construct_memory_view %B_addr, sizes: [{ac[0]}, {ac[1]}], strides: [{ac[1]}, 1] {{
        coordinate_set = #ac_set, memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{ac[0]}x{ac[1]}xf16>

    %A_at = ktdp.construct_access_tile %A[{idx_refs}] {{
        access_tile_set = #ac_set, access_tile_order = #identity
    }} : memref<{G[0]}x{G[1]}xf16> -> !ktdp.access_tile<{ac[0]}x{ac[1]}xindex>

    %B_at = ktdp.construct_access_tile %B[%c0, %c0] {{
        access_tile_set = #ac_set, access_tile_order = #identity
    }} : memref<{ac[0]}x{ac[1]}xf16> -> !ktdp.access_tile<{ac[0]}x{ac[1]}xindex>

    %data = ktdp.load %A_at : !ktdp.access_tile<{ac[0]}x{ac[1]}xindex> -> tensor<{ac[0]}x{ac[1]}xf16>
    ktdp.store %data, %B_at : tensor<{ac[0]}x{ac[1]}xf16>, !ktdp.access_tile<{ac[0]}x{ac[1]}xindex>
    return
  }}
}}
"""


# ---------------------------------------------------------------------------
# Test case table
# ---------------------------------------------------------------------------
#
# Reference tensor: np.arange(GR*GC, dtype=f16).reshape(GR, GC)
# All cases use a 4×4 global tensor unless noted.
#
# Three partition layouts are used:
#   Row-band    : P0 rows 0..r, P1 rows r+1..3, each spanning all 4 cols
#   Col-band    : P0 cols 0..c, P1 cols c+1..3, each spanning all 4 rows
#   Mixed       : P0 is a 2-row block, P1 is a 2-row×2-col block beside it
#
# Memory pointer map (non-overlapping, 4096-byte spacing):
_P0_PTR  = 0
_P1_PTR  = 4096
_OUT_PTR = 8192

_CASES: List[DistCopySpec] = [
    # -----------------------------------------------------------------------
    # Row-band partitioning
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0  ---- ]
    # r1 [ ----  P0  ---- ]   P0: rows 0..1, all cols
    # r2 [ ----  P1  ---- ]   P1: rows 2..3, all cols
    # r3 [ ----  P1  ---- ]
    # -----------------------------------------------------------------------

    # Case 1: full 4×4 access — both partitions loaded
    #
    #      c0   c1   c2   c3
    # r0 [ ░░░  P0  ░░░░░ ]   ░ = access tile
    # r1 [ ░░░  P0  ░░░░░ ]
    # r2 [ ░░░  P1  ░░░░░ ]
    # r3 [ ░░░  P1  ░░░░░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_full",
    ),
    # Case 2: 2×4 access at [0,0] — P1 pruned (access doesn't reach rows 2..3)
    #
    #      c0   c1   c2   c3
    # r0 [ ░░░  P0  ░░░░░ ]   ░ = access tile
    # r1 [ ░░░  P0  ░░░░░ ]
    # r2 [ ----  P1  ---- ]   P1 pruned
    # r3 [ ----  P1  ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(2, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_partial_p1_pruned",
    ),
    # Case 3: 2×2 sub-tile at [1,1] — spans both partitions (nonzero indices)
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0  ---- ]
    # r1 [  P0 | ░░░ |    ]   ░ = access tile  global[1:3, 1:3]
    # r2 [  P1 | ░░░ |    ]
    # r3 [ ----  P1  ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[1, 1], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_subtile_nonzero",
    ),
    # Case 4: full 4×4, P1 in LX with col-packed strides [1,4]
    #
    #      c0   c1   c2   c3
    # r0 [ ░░  P0(HBM)  ░ ]   ░ = access tile
    # r1 [ ░░  P0(HBM)  ░ ]   P0 row-major:  elem[r,c] at offset r*4+c
    # r2 [ ░░  P1(LX)   ░ ]   P1 col-packed: elem[r,c] at offset r+c*2
    # r3 [ ░░  P1(LX)   ░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_lx_col_packed_full",
    ),
    # Case 5: full 4×4, P0 in LX col-packed, P1 in HBM — reversed memory spaces
    #
    #      c0   c1   c2   c3
    # r0 [ ░░  P0(LX)   ░ ]   P0 col-packed: elem[r,c] at offset r+c*2
    # r1 [ ░░  P0(LX)   ░ ]   P1 row-major:  elem[r,c] at offset r*4+c
    # r2 [ ░░  P1(HBM)  ░ ]
    # r3 [ ░░  P1(HBM)  ░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_lx_hbm_col_packed_full",
    ),
    # Case 6: 2×2 sub-tile at [1,1], P1 in LX col-packed — mixed spaces + nonzero indices
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0(HBM)  ---- ]
    # r1 [  P0(HBM) | ░░░ |    ]   ░ = access tile  global[1:3, 1:3]
    # r2 [  P1(LX)  | ░░░ |    ]
    # r3 [ ----  P1(LX)   ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[1, 1], out_ptr=_OUT_PTR,
        id="row_hbm_lx_subtile_nonzero",
    ),
    # Case 7: unequal row bands — P0 is 1 row, P1 is 3 rows; full 4×4 access
    #
    #      c0   c1   c2   c3
    # r0 [ ░░░  P0  ░░░░░ ]   P0: 1 row
    # r1 [ ░░░  P1  ░░░░░ ]   P1: 3 rows
    # r2 [ ░░░  P1  ░░░░░ ]
    # r3 [ ░░░  P1  ░░░░░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 0), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(1, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_unequal_full",
    ),
    # Case 8: unequal row bands, 2×4 at [1,0] — P0 pruned (access starts at row 1)
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0  ---- ]   P0 pruned
    # r1 [ ░░░  P1  ░░░░░ ]   ░ = access tile
    # r2 [ ░░░  P1  ░░░░░ ]
    # r3 [ ----  P1  ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 0), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(1, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(2, 4), indices=[1, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_unequal_partial_p0_pruned",
    ),

    # -----------------------------------------------------------------------
    # Col-band partitioning
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0  |  --  P1  ]
    # r1 [  --  P0  |  --  P1  ]   P0: all rows, cols 0..1
    # r2 [  --  P0  |  --  P1  ]   P1: all rows, cols 2..3
    # r3 [  --  P0  |  --  P1  ]
    # -----------------------------------------------------------------------

    # Case 9: full 4×4 access — both col partitions loaded
    #
    #      c0   c1 | c2   c3
    # r0 [ ░░░  P0 | ░░░  P1 ]   ░ = access tile
    # r1 [ ░░░  P0 | ░░░  P1 ]
    # r2 [ ░░░  P0 | ░░░  P1 ]
    # r3 [ ░░░  P0 | ░░░  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="col_hbm_hbm_full",
    ),
    # Case 10: 4×2 access at [0,0] — P1 pruned (access stays in left cols 0..1)
    #
    #      c0   c1 | c2   c3
    # r0 [ ░░░  P0 |  --  P1 ]   P1 pruned
    # r1 [ ░░░  P0 |  --  P1 ]
    # r2 [ ░░░  P0 |  --  P1 ]
    # r3 [ ░░░  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(4, 2), indices=[0, 0], out_ptr=_OUT_PTR,
        id="col_hbm_hbm_partial_p1_pruned",
    ),
    # Case 11: 2×2 sub-tile at [1,1] — straddles col boundary (col 1 in P0, col 2 in P1)
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0 |  --  P1 ]
    # r1 [  P0 |░░ | ░░| P1  ]   ░ = access tile  global[1:3, 1:3]
    # r2 [  P0 |░░ | ░░| P1  ]
    # r3 [  --  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[1, 1], out_ptr=_OUT_PTR,
        id="col_hbm_hbm_subtile_nonzero",
    ),
    # Case 12: full 4×4, P0 HBM row-major, P1 LX col-packed — mixed spaces, col-band
    #
    #      c0     c1   |   c2     c3
    # r0 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]   P0 row-major:  elem[r,c] at offset r*2+c
    # r1 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]   P1 col-packed: elem[r,c] at offset r+c*4
    # r2 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]
    # r3 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="col_hbm_lx_col_packed_full",
    ),

    # -----------------------------------------------------------------------
    # Mixed layout: P0 is a tall left block (4 rows × 2 cols),
    #               P1 is a small bottom-right block (2 rows × 2 cols).
    # Top-right corner (rows 0..1, cols 2..3) is uncovered — access tiles
    # in these cases are designed to avoid it.
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0 |  --  -- ]
    # r1 [  --  P0 |  --  -- ]   P0: rows 0..3, cols 0..1
    # r2 [  --  P0 |  --  P1 ]   P1: rows 2..3, cols 2..3
    # r3 [  --  P0 |  --  P1 ]
    # -----------------------------------------------------------------------

    # Case 13: 4×2 access at [0,0] — spans full left block P0; P1 pruned
    #
    #      c0   c1 | c2   c3
    # r0 [ ░░░  P0 |  --  -- ]
    # r1 [ ░░░  P0 |  --  -- ]
    # r2 [ ░░░  P0 |  --  P1 ]
    # r3 [ ░░░  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(4, 2), indices=[0, 0], out_ptr=_OUT_PTR,
        id="mixed_left_block_only",
    ),
    # Case 14: 2×2 at [2,0] — bottom-left corner, only P0 contributes; P1 pruned
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0 |  --  -- ]
    # r1 [  --  P0 |  --  -- ]
    # r2 [ ░░░  P0 |  --  P1 ]   ░ = access tile  global[2:4, 0:2]
    # r3 [ ░░░  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[2, 0], out_ptr=_OUT_PTR,
        id="mixed_bottom_left_only",
    ),
    # Case 15: 2×2 at [2,0], P0 in LX col-packed, P1 in HBM — mixed spaces, mixed layout
    #
    #      c0   c1 | c2   c3
    # r0 [  -- P0(LX) |  --  --     ]
    # r1 [  -- P0(LX) |  --  --     ]
    # r2 [ ░░ P0(LX) ░|  --  P1(HBM)]   ░ = access tile  global[2:4, 0:2]
    # r3 [ ░░ P0(LX) ░|  --  P1(HBM)]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="LX",  strides=[1, 4], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[2, 0], out_ptr=_OUT_PTR,
        id="mixed_lx_left_hbm_right_bottom_only",
    ),
]


def _seed_and_run(spec: DistCopySpec) -> Tuple[np.ndarray, np.ndarray]:
    """Seed memory from spec, execute the generated kernel, return (expected, actual).

    The reference tensor is np.arange(16, dtype=f16).reshape(4,4).
    The expected slice is full[indices[0]:indices[0]+access_shape[0],
                               indices[1]:indices[1]+access_shape[1]].
    """
    full = np.arange(16, dtype=np.float16).reshape(spec.global_shape)
    mlir = _build_mlir(spec)
    interp = KTIRInterpreter()
    interp.load(mlir)
    _orig = interp._prepare_execution

    p0, p1 = spec.p0, spec.p1
    ac = spec.access_shape
    idx = spec.indices

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        p0_block = full[p0.rows[0]:p0.rows[1] + 1, p0.cols[0]:p0.cols[1] + 1]
        p1_block = full[p1.rows[0]:p1.rows[1] + 1, p1.cols[0]:p1.cols[1] + 1]
        p0_mem = _get_mem(interp, p0.memory_space)
        p1_mem = _get_mem(interp, p1.memory_space)
        _write_strided(p0_mem, p0.base_ptr, p0_block.copy(), p0.strides)
        _write_strided(p1_mem, p1.base_ptr, p1_block.copy(), p1.strides)
        out_stick = (spec.out_ptr * 2) // interp.memory.hbm.STICK_BYTES
        interp.memory.hbm.write(out_stick, np.zeros(ac[0] * ac[1], dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function("dist_copy")

    r0, c0 = idx[0], idx[1]
    expected = full[r0:r0 + ac[0], c0:c0 + ac[1]]
    n_out = ac[0] * ac[1]
    out_stick = (spec.out_ptr * 2) // interp.memory.hbm.STICK_BYTES
    actual = interp.memory.hbm.read(out_stick, n_out, "f16").reshape(ac)
    return expected, actual


# ---------------------------------------------------------------------------
# RFC example: per-core LX routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,func_name,entry", get_test_params("distributed_view_copy"))
def test_distributed_view_copy_rfc(path, func_name, entry):
    """construct_distributed_memory_view — RFC §C.3 example file.

    A is a 192×64 logical tensor distributed across three regions:
      A_HBM (96×64,  HBM,          row-major strides [64, 1])  rows   0..95
      A_LX0 (32×64,  LX core=0,    col-packed strides [1, 64]) rows  96..127
      A_LX1 (64×64,  LX core=1,    row-major strides [64, 1])  rows 128..191
    The kernel copies A into contiguous HBM output B, also 192×64.
    """
    interp = KTIRInterpreter()
    interp.load(path)
    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        lx0 = interp.memory.get_lx(0)
        lx1 = interp.memory.get_lx(1)
        full = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
        # MLIR constants are element indices (f16, 2 bytes/elem).
        # A_HBM_addr=0  → byte 0   → stick 0
        # A_LX0_addr=12288 → byte 24576 (via _write_strided element-index path)
        # A_LX1_addr=16384 → byte 32768
        # B_addr=24576 → byte 49152 → stick 384
        hbm.write(0, full[0:96, :].flatten())
        _write_strided(lx0, 12288, full[96:128, :].copy(), strides=[1, 64])
        lx1.write(16384 * 2, full[128:192, :].flatten())
        hbm.write(24576 * 2 // hbm.STICK_BYTES, np.zeros(192 * 64, dtype=np.float16))
        # Advance each LX next_ptr past its seeded region.
        # lx0 seeded at byte 24576, col-packed span = 31 + 63*64 + 1 = 4064 elems = 8128 bytes
        # lx1 seeded at byte 32768, row-major span = 64*64 = 4096 elems = 8192 bytes
        lx0.next_ptr = 16384 * 2 + 8128
        lx1.next_ptr = 16384 * 2 + 8192

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)

    expected = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
    b = interp.memory.hbm.read(24576 * 2 // interp.memory.hbm.STICK_BYTES, 192 * 64, "f16").reshape(192, 64)
    np.testing.assert_array_equal(b, expected)


# ---------------------------------------------------------------------------
# Parametrized 2-partition copy suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", _CASES, ids=[c.id for c in _CASES])
def test_distributed_copy(spec: DistCopySpec):
    """2-partition distributed copy — data-correctness check.

    Generates a single-function MLIR kernel from *spec*, seeds the partitions
    in the appropriate memory spaces and strides, runs the kernel, and asserts
    that the output matches the corresponding slice of the reference tensor.
    """
    expected, actual = _seed_and_run(spec)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Structural: fast path produces BoxSet in surviving partitions
# ---------------------------------------------------------------------------

def test_distributed_tile_access_fast_path_emits_box_set():
    """When B_i and A are both boxes, C_i must be stored as a BoxSet.

    Confirms that the BoxSet refactor wires end-to-end: parse-time
    lowering turns the partition coordinate_set into BoxSet, and
    distributed_tile_access's fast path stores the intersection as a
    BoxSet (not a pre-enumerated point list).  A regression here would
    silently drop us to the slow path and re-introduce the per-partition
    enumeration cost.
    """
    from ktir_cpu.affine import AffineMap, BoxSet
    from ktir_cpu.ir_types import DistributedMemRef, MemRef
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map, parse_affine_set

    # Build 2 row-band partitions of a 4×4 tensor: P0 rows 0..1, P1 rows 2..3.
    B0 = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
    B1 = parse_affine_set("affine_set<(d0, d1) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
    assert isinstance(B0, BoxSet) and isinstance(B1, BoxSet)  # sanity

    P0 = MemRef(base_ptr=0, shape=(2, 4), strides=[4, 1], memory_space="HBM", coordinate_set=B0)
    P1 = MemRef(base_ptr=64, shape=(2, 4), strides=[4, 1], memory_space="HBM", coordinate_set=B1)
    dist = DistributedMemRef(partitions=[P0, P1], shape=(4, 4), dtype="f16")

    # Full-tile access from the origin: both partitions survive with their
    # own extents as C_i.
    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    assert isinstance(base_map, AffineMap)
    out = MemoryOps.distributed_tile_access(
        dist_ref=dist,
        access_shape=(4, 4),
        base_map=base_map,
        indices=[0, 0],
        access_tile_set=None,
    )
    assert len(out.partitions) == 2
    for part in out.partitions:
        assert isinstance(part.coordinate_set, BoxSet), (
            f"fast path must store BoxSet in coordinate_set, got "
            f"{type(part.coordinate_set).__name__}"
        )
    # P0 survives with rows 0..1 × cols 0..3; P1 with rows 2..3 × cols 0..3.
    assert out.partitions[0].coordinate_set == BoxSet(lo=(0, 0), hi=(2, 4))
    assert out.partitions[1].coordinate_set == BoxSet(lo=(2, 0), hi=(4, 4))
    # partition_origin == min(B_i)
    assert out.partitions[0].partition_origin == (0, 0)
    assert out.partitions[1].partition_origin == (2, 0)


# ---------------------------------------------------------------------------
# Symbolic-shape variant of the structural fast-path assertion
#
# Pure ops-layer test: each partition's row extent is given symbolically
# (``[0, s_0)`` and ``[s_0, 2*s_0)``); the test specialises both manually
# before handing the partitions to ``distributed_tile_access``.  This
# verifies the GEOMETRY ``distributed_tile_access`` produces on already-
# concrete symbolic-derived partitions plus the
# ``survivor.coordinate_set is BoxSet`` regression guard the issue test
# plan asks for.
#
# It does NOT exercise the dialect handler's symbol-binding step — the
# ``specialize(P0_shape)`` call here passes the full per-partition shape
# tuple, which only happens to coincide with the right symbol values
# because the symbolic dim is leading (``s_0`` == ``shape[0]``).  The
# binding-source verification (and a regression for the trailing-``?``
# case where ``shape[0]`` is the wrong source) lives in
# ``test_dialects_exec.py::TestKtdp``.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("partition_rows", [2, 4, 8])
def test_distributed_tile_access_dynamic_shape_emits_box_set(partition_rows):
    """Geometry + ``is BoxSet`` survivor check on symbolic-derived partitions.

    Parametrised over partition row counts to exercise the lowering and
    intersect on a few concrete values.  Pure ops-layer — see the module
    comment for why the binding source isn't verified here.
    """
    from ktir_cpu.affine import AffineMap, BoxSet
    from ktir_cpu.ir_types import DistributedMemRef, MemRef
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map, parse_affine_set

    # Symbolic partition extents:
    #   B0 = rows [0, s0), cols [0, 3]
    #   B1 = rows [s0, 2*s0), cols [0, 3]
    B0_sym = parse_affine_set(
        "affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>"
    )
    B1_sym = parse_affine_set(
        "affine_set<(d0, d1)[s0] : (d0 - s0 >= 0, -d0 + 2*s0 - 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>"
    )
    assert isinstance(B0_sym, BoxSet) and not B0_sym._all_concrete
    assert isinstance(B1_sym, BoxSet) and not B1_sym._all_concrete

    # Specialise manually before constructing the DistributedMemRef.  This
    # leans on the per-partition row count being the leading dim so a
    # plain ``shape`` tuple happens to be the right symbols tuple — see
    # module comment for the caveat.
    P0_shape = (partition_rows, 4)
    P1_shape = (partition_rows, 4)
    B0 = B0_sym.specialize(P0_shape)
    B1 = B1_sym.specialize(P1_shape)
    assert B0._all_concrete and B1._all_concrete

    P0 = MemRef(
        base_ptr=0, shape=P0_shape, strides=[4, 1],
        memory_space="HBM", coordinate_set=B0,
    )
    P1 = MemRef(
        base_ptr=P0_shape[0] * 4 * 2, shape=P1_shape, strides=[4, 1],
        memory_space="HBM", coordinate_set=B1,
    )
    total_rows = 2 * partition_rows
    dist = DistributedMemRef(
        partitions=[P0, P1], shape=(total_rows, 4), dtype="f16",
    )

    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    assert isinstance(base_map, AffineMap)
    out = MemoryOps.distributed_tile_access(
        dist_ref=dist,
        access_shape=(total_rows, 4),
        base_map=base_map,
        indices=[0, 0],
        access_tile_set=None,
    )

    # Both partitions survive a full-tile access; each survivor's
    # coordinate_set must be BoxSet — the regression guard the issue
    # test plan asks for.
    assert len(out.partitions) == 2
    for part in out.partitions:
        assert isinstance(part.coordinate_set, BoxSet), (
            f"dynamic-shape fast path must store BoxSet in coordinate_set, "
            f"got {type(part.coordinate_set).__name__}"
        )
    # Geometry derived from the resolved sizes — no hand-coded constants.
    assert out.partitions[0].coordinate_set == BoxSet(
        lo=(0, 0), hi=(partition_rows, 4)
    )
    assert out.partitions[1].coordinate_set == BoxSet(
        lo=(partition_rows, 0), hi=(2 * partition_rows, 4)
    )
    assert out.partitions[0].partition_origin == (0, 0)
    assert out.partitions[1].partition_origin == (partition_rows, 0)


# ---------------------------------------------------------------------------
# distributed_tile_access fast/slow path: verified against a static fixture
#
# Scenario (shared by both tests below):
#   - 256×512 distributed view, 4 row-band partitions of 64×512
#     (partition_rows scaled down from the 2048×8192 Triton matmul view
#     so the slow path's brute-force enumeration stays CI-tractable).
#   - Access tile shape 32×128 — matches the A-tile size in
#     examples/triton-ktir/matmul_fwd_ktir.mlir.
#   - base_map = identity; access_tile_set = None (full box A).
#
# Fixture: for each indices value, the expected list of surviving partitions,
# each described by (C_i extent as (lo, hi), expected partition_origin).
# C_i = [max(r, r0_i), min(r+32, r1_i)) × [c, c+128) where x = (r, c).
# ---------------------------------------------------------------------------

_SHAPE = (256, 512)
_PARTITION_ROWS = [(0, 64), (64, 128), (128, 192), (192, 256)]
_ACCESS_SHAPE = (32, 128)

# Each fixture entry: (id, indices, [(C_i_lo, C_i_hi, partition_origin), ...])
# partition_origin = (r0_i, 0) — the lower corner of the partition's own extent.
_FIXTURE = [
    # Access origin inside P0, no boundary crossing.
    ("single_partition", (10, 0), [
        ((10, 0), (42, 128), (0, 0)),
    ]),
    # Access rows 50..81 span P0 (rows 50..63) and P1 (rows 64..81).
    ("cross_boundary", (50, 64), [
        ((50, 64), (64, 192), (0, 0)),
        ((64, 64), (82, 192), (64, 0)),
    ]),
    # Access rows 200..231, cols 256..383 — lands inside P3 only.
    ("last_partition", (200, 256), [
        ((200, 256), (232, 384), (192, 0)),
    ]),
    # Access from origin: first 32 rows of P0, first 128 cols.
    ("origin", (0, 0), [
        ((0, 0), (32, 128), (0, 0)),
    ]),
]


def _build_partitions(parser):
    """Build 4 row-band partitions with coordinate_set parsed by *parser*.

    Passing ``parse_affine_set`` lowers axis-aligned sets to BoxSet
    (fast path); ``parse_affine_set_raw`` keeps them as AffineSet (slow path).
    """
    from ktir_cpu.ir_types import MemRef
    _, ncols = _SHAPE
    parts: List[MemRef] = []
    for r0, r1 in _PARTITION_ROWS:
        src = (
            f"affine_set<(d0, d1) : "
            f"(d0 - {r0} >= 0, -d0 + {r1 - 1} >= 0, "
            f"d1 >= 0, -d1 + {ncols - 1} >= 0)>"
        )
        parts.append(MemRef(
            base_ptr=r0 * ncols * 2,   # f16 = 2 bytes; placeholder addr
            shape=(r1 - r0, ncols),
            strides=[ncols, 1],
            memory_space="HBM",
            coordinate_set=parser(src),
        ))
    return parts


def _run_and_collect(partitions, indices):
    """Run distributed_tile_access and return [(C_i_pts_sorted, origin), ...]."""
    from ktir_cpu.affine import AffineMap, BoxSet
    from ktir_cpu.ir_types import DistributedMemRef
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map

    dist = DistributedMemRef(
        partitions=partitions, shape=_SHAPE, dtype="f16",
    )
    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    assert isinstance(base_map, AffineMap)
    out = MemoryOps.distributed_tile_access(
        dist_ref=dist, access_shape=_ACCESS_SHAPE,
        base_map=base_map, indices=list(indices), access_tile_set=None,
    )
    collected = []
    for part in out.partitions:
        cs = part.coordinate_set
        pts = cs.enumerate() if isinstance(cs, BoxSet) else cs
        collected.append((sorted(pts), part.partition_origin, type(cs)))
    return collected


def _expected_points(lo, hi):
    """Expand a box [lo, hi) into a sorted row-major point list."""
    import itertools as _it
    return sorted(_it.product(*(range(lo[d], hi[d]) for d in range(len(lo)))))


@pytest.mark.parametrize("case_id, indices, expected", _FIXTURE, ids=[c[0] for c in _FIXTURE])
def test_distributed_tile_access_fast_path(case_id, indices, expected):
    """Fast path (BoxSet) produces the expected C_i extents and origins."""
    from ktir_cpu.affine import BoxSet
    from ktir_cpu.parser_ast import parse_affine_set

    got = _run_and_collect(_build_partitions(parse_affine_set), indices)
    assert len(got) == len(expected), f"{case_id}: partition count mismatch"
    for (pts_got, origin_got, cs_type), (exp_lo, exp_hi, exp_origin) in zip(got, expected):
        assert cs_type is BoxSet, f"{case_id}: fast path must emit BoxSet, got {cs_type.__name__}"
        assert origin_got == exp_origin, f"{case_id}: origin {origin_got} != {exp_origin}"
        assert pts_got == _expected_points(exp_lo, exp_hi), (
            f"{case_id}: C_i mismatch for partition at origin {origin_got}"
        )


@pytest.mark.parametrize("case_id, indices, expected", _FIXTURE, ids=[c[0] for c in _FIXTURE])
def test_distributed_tile_access_slow_path(case_id, indices, expected):
    """Slow path (AffineSet brute-force enumerate) produces the same C_i.

    Parses partition coordinate_sets via parse_affine_set_raw, which skips
    the BoxSet lowering, forcing distributed_tile_access onto the
    AffineSet.enumerate path.  Exercises ~130k AST walks per partition at
    this fixture size — slow but tractable for CI, and confirms the two
    paths agree on real-scale inputs.
    """
    from ktir_cpu.affine import AffineSet
    from ktir_cpu.parser_ast import parse_affine_set_raw

    got = _run_and_collect(_build_partitions(parse_affine_set_raw), indices)
    assert len(got) == len(expected), f"{case_id}: partition count mismatch"
    for (pts_got, origin_got, cs_type), (exp_lo, exp_hi, exp_origin) in zip(got, expected):
        # Slow path stores a point list in coordinate_set, not a BoxSet.
        assert cs_type is list, f"{case_id}: slow path must emit list, got {cs_type.__name__}"
        assert origin_got == exp_origin, f"{case_id}: origin {origin_got} != {exp_origin}"
        assert pts_got == _expected_points(exp_lo, exp_hi), (
            f"{case_id}: C_i mismatch for partition at origin {origin_got}"
        )


# ---------------------------------------------------------------------------
# distributed_store fast path: writes touch ONLY the C_i rectangle
#
# The sub-TileRef construction in distributed_store inherits the parent's
# strides verbatim and shifts base_ptr to the box's local origin.  A bug in
# either of those (wrong stride, wrong byte offset, wrong sub-shape) could
# silently overwrite memory adjacent to C_i — outside the rectangle but
# still inside the partition's allocation.  test_distributed_copy doesn't
# catch this: it only reads back the access-tile region, so trampled data
# elsewhere goes unnoticed.
#
# These tests seed the WHOLE partition with a sentinel pattern, run a
# distributed_store that touches only a small C_i sub-rectangle, and then
# inspect every byte: C_i must hold the new data; every other byte must
# still be the sentinel.  Two cases exercise the row-major and
# column-packed stride layouts.
# ---------------------------------------------------------------------------

def test_distributed_store_does_not_trample_outside_C_i():
    """Row-major partition: distributed_store must not write outside C_i."""
    from ktir_cpu.affine import BoxSet
    from ktir_cpu.dtypes import bytes_per_elem
    from ktir_cpu.grid import CoreContext
    from ktir_cpu.ir_types import DistributedMemRef, MemRef, Tile
    from ktir_cpu.memory import HBMSimulator, LXScratchpad
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map, parse_affine_set

    PART_SHAPE = (8, 16)
    NCOLS = 16
    DTYPE = "f16"
    bpe = bytes_per_elem(DTYPE)
    elems_per_part = PART_SHAPE[0] * PART_SHAPE[1]
    bytes_per_part = elems_per_part * bpe

    hbm = HBMSimulator(size_gb=1)
    # Allocate two contiguous partition regions; HBM is stick-addressed
    # (upstream PR #32) so MemRef.base_ptr is a stick index.
    P0_STICK = hbm.allocate(bytes_per_part)
    P1_STICK = hbm.allocate(bytes_per_part)
    SENTINEL = np.float16(-7.0)

    hbm.write(P0_STICK, np.full(elems_per_part, SENTINEL, dtype=np.float16))
    hbm.write(P1_STICK, np.full(elems_per_part, SENTINEL, dtype=np.float16))
    ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                      lx=LXScratchpad(size_mb=1, core_id=0), hbm=hbm)

    # Box-form coordinate sets → BoxSet via parse-time lowering.
    B0 = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 15 >= 0)>")
    B1 = parse_affine_set("affine_set<(d0, d1) : (d0 - 8 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 15 >= 0)>")
    assert isinstance(B0, BoxSet) and isinstance(B1, BoxSet)

    P0 = MemRef(base_ptr=stick_to_elem_idx(P0_STICK, DTYPE), shape=PART_SHAPE, strides=[NCOLS, 1],
                memory_space="HBM", dtype=DTYPE, coordinate_set=B0)
    P1 = MemRef(base_ptr=stick_to_elem_idx(P1_STICK, DTYPE), shape=PART_SHAPE, strides=[NCOLS, 1],
                memory_space="HBM", dtype=DTYPE, coordinate_set=B1)
    dist = DistributedMemRef(partitions=[P0, P1], shape=(16, 16), dtype=DTYPE)

    # 4×4 access at (2, 4) — fully inside P0; C_i = [2,6) × [4,8).
    access_shape = (4, 4)
    indices = (2, 4)
    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    resolved = MemoryOps.distributed_tile_access(
        dist_ref=dist, access_shape=access_shape, base_map=base_map,
        indices=list(indices), access_tile_set=None,
    )
    assert len(resolved.partitions) == 1, "P1 should be pruned"
    assert isinstance(resolved.partitions[0].coordinate_set, BoxSet)
    assert resolved.partitions[0].coordinate_set == BoxSet(lo=(2, 4), hi=(6, 8))

    payload_values = np.arange(1, 17, dtype=np.float16).reshape(4, 4)
    MemoryOps.distributed_store(ctx, Tile(payload_values, DTYPE, access_shape), resolved)

    p0_full = hbm.read(P0_STICK, elems_per_part, DTYPE).reshape(PART_SHAPE)
    p1_full = hbm.read(P1_STICK, elems_per_part, DTYPE).reshape(PART_SHAPE)

    # P1 entirely untouched (was pruned, must not have been visited).
    assert np.all(p1_full == SENTINEL), \
        "P1 was trampled — distributed_store wrote outside the surviving partition"

    # Inside C_i: payload data.  Outside C_i: still sentinel.
    c_i_slice = (slice(2, 6), slice(4, 8))
    np.testing.assert_array_equal(
        p0_full[c_i_slice], payload_values,
        err_msg="C_i sub-rectangle of P0 has wrong values",
    )
    mask = np.zeros(PART_SHAPE, dtype=bool)
    mask[c_i_slice] = True
    outside = p0_full[~mask]
    assert np.all(outside == SENTINEL), (
        f"P0 has {(outside != SENTINEL).sum()} cell(s) outside C_i that "
        f"differ from the sentinel — distributed_store trampled neighbouring "
        f"memory."
    )


def test_distributed_store_col_packed_does_not_trample_outside_C_i():
    """Column-packed partition (strides=[1, R]): same trample check.

    Stresses the sub-TileRef stride-inheritance claim: the sub-tile must
    reuse the parent's strides=[1, R], not synthesise row-major strides
    for its own sub-shape.  A bug there would scatter the written data
    across column-packed memory and trample sentinels outside C_i.
    """
    from ktir_cpu.affine import BoxSet
    from ktir_cpu.dtypes import bytes_per_elem
    from ktir_cpu.grid import CoreContext
    from ktir_cpu.ir_types import DistributedMemRef, MemRef, Tile
    from ktir_cpu.memory import HBMSimulator, LXScratchpad
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map, parse_affine_set

    PART_SHAPE = (8, 16)
    NROWS = 8
    DTYPE = "f16"
    bpe = bytes_per_elem(DTYPE)
    elems_per_part = PART_SHAPE[0] * PART_SHAPE[1]
    bytes_per_part = elems_per_part * bpe

    hbm = HBMSimulator(size_gb=1)
    P0_STICK = hbm.allocate(bytes_per_part)
    P1_STICK = hbm.allocate(bytes_per_part)
    SENTINEL = np.float16(99.0)

    hbm.write(P0_STICK, np.full(elems_per_part, SENTINEL, dtype=np.float16))
    hbm.write(P1_STICK, np.full(elems_per_part, SENTINEL, dtype=np.float16))
    ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                      lx=LXScratchpad(size_mb=1, core_id=0), hbm=hbm)

    B0 = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 15 >= 0)>")
    B1 = parse_affine_set("affine_set<(d0, d1) : (d0 - 8 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 15 >= 0)>")
    assert isinstance(B0, BoxSet) and isinstance(B1, BoxSet)

    # strides=[1, NROWS] → column-packed: element (r, c) at offset r + c*NROWS.
    P0 = MemRef(base_ptr=stick_to_elem_idx(P0_STICK, DTYPE), shape=PART_SHAPE, strides=[1, NROWS],
                memory_space="HBM", dtype=DTYPE, coordinate_set=B0)
    P1 = MemRef(base_ptr=stick_to_elem_idx(P1_STICK, DTYPE), shape=PART_SHAPE, strides=[1, NROWS],
                memory_space="HBM", dtype=DTYPE, coordinate_set=B1)
    dist = DistributedMemRef(partitions=[P0, P1], shape=(16, 16), dtype=DTYPE)

    access_shape = (4, 4)
    indices = (2, 4)
    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    resolved = MemoryOps.distributed_tile_access(
        dist_ref=dist, access_shape=access_shape, base_map=base_map,
        indices=list(indices), access_tile_set=None,
    )
    assert len(resolved.partitions) == 1
    assert isinstance(resolved.partitions[0].coordinate_set, BoxSet)

    payload_values = np.arange(1, 17, dtype=np.float16).reshape(4, 4)
    MemoryOps.distributed_store(ctx, Tile(payload_values, DTYPE, access_shape), resolved)

    p0_flat = hbm.read(P0_STICK, elems_per_part, DTYPE)
    p1_flat = hbm.read(P1_STICK, elems_per_part, DTYPE)

    # P1 entirely untouched.
    assert np.all(p1_flat == SENTINEL), "P1 was trampled (col-packed case)"

    # Reconstruct P0's logical grid via the column-packed strides.
    p0_logical = np.empty(PART_SHAPE, dtype=np.float16)
    for r in range(PART_SHAPE[0]):
        for c in range(PART_SHAPE[1]):
            p0_logical[r, c] = p0_flat[r * 1 + c * NROWS]

    c_i_slice = (slice(2, 6), slice(4, 8))
    np.testing.assert_array_equal(
        p0_logical[c_i_slice], payload_values,
        err_msg="C_i sub-rectangle has wrong values (col-packed)",
    )
    mask = np.zeros(PART_SHAPE, dtype=bool)
    mask[c_i_slice] = True
    outside = p0_logical[~mask]
    assert np.all(outside == SENTINEL), (
        f"P0 col-packed: {(outside != SENTINEL).sum()} cell(s) outside C_i "
        f"differ from sentinel — strides inheritance is broken."
    )


# ---------------------------------------------------------------------------
# distributed_load / distributed_store SLOW path (non-BoxSet C_i)
#
# When a partition's coordinate_set is a plain List[Tuple] (parsed via
# parse_affine_set_raw, which skips BoxSet lowering), distributed_tile_access
# stores an enumerated point list on each survivor, forcing distributed_load
# and distributed_store onto their slow paths.  Those slow paths now use NumPy
# fancy-indexing (one vectorized gather/scatter per partition) instead of a
# per-coordinate Python loop.  These tests cross a partition boundary so the
# fancy-index write lands in different access-local regions for each survivor,
# and assert the survivors are genuinely on the slow path (coordinate_set is a
# list, not a BoxSet) so a regression to BoxSet wouldn't silently bypass them.
# ---------------------------------------------------------------------------

def _build_raw_row_band_partitions(p0_stick, p1_stick, part_shape, ncols, dtype):
    """Two row-band partitions with raw (non-BoxSet) coordinate sets.

    P0 covers rows ``[0, R)``, P1 rows ``[R, 2R)``; both span all *ncols*
    columns with row-major strides.  Parsing via ``parse_affine_set_raw``
    keeps the sets as ``AffineSet`` so the surviving C_i is enumerated to a
    point list — the slow path.
    """
    from ktir_cpu.dtypes import stick_to_elem_idx
    from ktir_cpu.ir_types import MemRef
    from ktir_cpu.parser_ast import parse_affine_set_raw

    R = part_shape[0]
    B0 = parse_affine_set_raw(
        f"affine_set<(d0, d1) : (d0 >= 0, -d0 + {R - 1} >= 0, "
        f"d1 >= 0, -d1 + {ncols - 1} >= 0)>"
    )
    B1 = parse_affine_set_raw(
        f"affine_set<(d0, d1) : (d0 - {R} >= 0, -d0 + {2 * R - 1} >= 0, "
        f"d1 >= 0, -d1 + {ncols - 1} >= 0)>"
    )
    P0 = MemRef(base_ptr=stick_to_elem_idx(p0_stick, dtype), shape=part_shape, strides=[ncols, 1],
                memory_space="HBM", dtype=dtype, coordinate_set=B0)
    P1 = MemRef(base_ptr=stick_to_elem_idx(p1_stick, dtype), shape=part_shape, strides=[ncols, 1],
                memory_space="HBM", dtype=dtype, coordinate_set=B1)
    return P0, P1


def test_distributed_load_slow_path_cross_boundary():
    """distributed_load slow path (list C_i) gathers correctly across partitions."""
    from ktir_cpu.affine import BoxSet
    from ktir_cpu.dtypes import bytes_per_elem
    from ktir_cpu.grid import CoreContext
    from ktir_cpu.ir_types import DistributedMemRef
    from ktir_cpu.memory import HBMSimulator, LXScratchpad
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map

    PART_SHAPE = (8, 16)
    NCOLS = 16
    DTYPE = "f16"
    GLOBAL = (16, 16)
    bpe = bytes_per_elem(DTYPE)
    elems_per_part = PART_SHAPE[0] * PART_SHAPE[1]

    full = np.arange(GLOBAL[0] * GLOBAL[1], dtype=np.float16).reshape(GLOBAL)

    hbm = HBMSimulator(size_gb=1)
    P0_STICK = hbm.allocate(elems_per_part * bpe)
    P1_STICK = hbm.allocate(elems_per_part * bpe)
    # Row-major contiguous → flatten writes the block verbatim.
    hbm.write(P0_STICK, full[0:8, :].copy().flatten())
    hbm.write(P1_STICK, full[8:16, :].copy().flatten())
    ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                      lx=LXScratchpad(size_mb=1, core_id=0), hbm=hbm)

    P0, P1 = _build_raw_row_band_partitions(P0_STICK, P1_STICK, PART_SHAPE, NCOLS, DTYPE)
    dist = DistributedMemRef(partitions=[P0, P1], shape=GLOBAL, dtype=DTYPE)

    # 4×4 access at (6, 4): rows 6,7 in P0 and rows 8,9 in P1 → both survive.
    access_shape = (4, 4)
    indices = (6, 4)
    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    resolved = MemoryOps.distributed_tile_access(
        dist_ref=dist, access_shape=access_shape, base_map=base_map,
        indices=list(indices), access_tile_set=None,
    )
    assert len(resolved.partitions) == 2, "access must straddle both partitions"
    for part in resolved.partitions:
        assert not isinstance(part.coordinate_set, BoxSet), \
            "slow path requires a non-BoxSet (list) coordinate_set"
        assert isinstance(part.coordinate_set, list)

    tile = MemoryOps.distributed_load(ctx, resolved, result_shape=access_shape)
    np.testing.assert_array_equal(tile.data, full[6:10, 4:8])


def test_distributed_store_slow_path_cross_boundary():
    """distributed_store slow path (list C_i) scatters across partitions without trampling."""
    from ktir_cpu.affine import BoxSet
    from ktir_cpu.dtypes import bytes_per_elem
    from ktir_cpu.grid import CoreContext
    from ktir_cpu.ir_types import DistributedMemRef, Tile
    from ktir_cpu.memory import HBMSimulator, LXScratchpad
    from ktir_cpu.ops.memory_ops import MemoryOps
    from ktir_cpu.parser_ast import parse_affine_map

    PART_SHAPE = (8, 16)
    NCOLS = 16
    DTYPE = "f16"
    GLOBAL = (16, 16)
    bpe = bytes_per_elem(DTYPE)
    elems_per_part = PART_SHAPE[0] * PART_SHAPE[1]
    SENTINEL = np.float16(-7.0)

    hbm = HBMSimulator(size_gb=1)
    P0_STICK = hbm.allocate(elems_per_part * bpe)
    P1_STICK = hbm.allocate(elems_per_part * bpe)
    hbm.write(P0_STICK, np.full(elems_per_part, SENTINEL, dtype=np.float16))
    hbm.write(P1_STICK, np.full(elems_per_part, SENTINEL, dtype=np.float16))
    ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                      lx=LXScratchpad(size_mb=1, core_id=0), hbm=hbm)

    P0, P1 = _build_raw_row_band_partitions(P0_STICK, P1_STICK, PART_SHAPE, NCOLS, DTYPE)
    dist = DistributedMemRef(partitions=[P0, P1], shape=GLOBAL, dtype=DTYPE)

    # 4×4 access at (6, 4): rows 6,7 → P0, rows 8,9 → P1.
    access_shape = (4, 4)
    indices = (6, 4)
    base_map = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
    resolved = MemoryOps.distributed_tile_access(
        dist_ref=dist, access_shape=access_shape, base_map=base_map,
        indices=list(indices), access_tile_set=None,
    )
    assert len(resolved.partitions) == 2
    for part in resolved.partitions:
        assert isinstance(part.coordinate_set, list), "slow path requires list C_i"

    payload = np.arange(1, 17, dtype=np.float16).reshape(4, 4)
    MemoryOps.distributed_store(ctx, Tile(payload, DTYPE, access_shape), resolved)

    p0_full = hbm.read(P0_STICK, elems_per_part, DTYPE).reshape(PART_SHAPE)
    p1_full = hbm.read(P1_STICK, elems_per_part, DTYPE).reshape(PART_SHAPE)

    # P0 receives the top two access rows (global rows 6,7 → local 6,7) at cols 4..7.
    # P1 receives the bottom two access rows (global rows 8,9 → local 0,1) at cols 4..7.
    np.testing.assert_array_equal(p0_full[6:8, 4:8], payload[0:2, :])
    np.testing.assert_array_equal(p1_full[0:2, 4:8], payload[2:4, :])

    # Everything outside the two written sub-rectangles stays sentinel.
    p0_mask = np.zeros(PART_SHAPE, dtype=bool)
    p0_mask[6:8, 4:8] = True
    p1_mask = np.zeros(PART_SHAPE, dtype=bool)
    p1_mask[0:2, 4:8] = True
    assert np.all(p0_full[~p0_mask] == SENTINEL), "P0 trampled outside C_i (slow path)"
    assert np.all(p1_full[~p1_mask] == SENTINEL), "P1 trampled outside C_i (slow path)"
