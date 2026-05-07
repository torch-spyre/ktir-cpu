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

"""
Memory compute helpers.

Tile view construction, sub-tile access, and HBM/LX load/store
primitives used by dialect handlers in ``ktir_cpu.dialects``.
"""

from typing import List, Optional, Tuple
import numpy as np
from ..affine import AffineMap
from ..dialects.ktdp_helpers import eval_subscript_expr
from ..dtypes import bytes_per_elem as _bytes_per_elem
from ..ir_types import MemRef, Tile, TileRef
from ..grid import CoreContext
from ..memory import HBMSimulator


class MemoryOps:
    """Tile memory helpers — view, access, load, store."""

    @staticmethod
    def tile_view(
        context: CoreContext,
        ptr: int,
        shape: Tuple[int, ...],
        strides: List[int],
        memory_space: str,
        dtype: str = "f16",
        coordinate_set: Optional[str] = None,
    ) -> MemRef:
        """Create a hardware-aware memory view (MemRef).

        Builds a MemRef describing a contiguous region in HBM or LX.

        Args:
            context: Core execution context
            ptr: Base pointer (stick index for HBM, byte address for LX)
            shape: Tile shape
            strides: Memory strides (element counts)
            memory_space: "HBM" or "LX"
            dtype: Data type
            coordinate_set: Verbatim affine_set string, no evaluation

        Returns:
            MemRef describing the memory layout
        """
        return MemRef(
            base_ptr=ptr,
            shape=shape,
            strides=strides,
            memory_space=memory_space,
            dtype=dtype,
            coordinate_set=coordinate_set,
        )

    @staticmethod
    def tile_access(
        context: CoreContext,
        parent_ref: MemRef,
        indices: List[int],
        access_shape: Tuple[int, ...],
        base_map: AffineMap,
    ) -> TileRef:
        """Extract a sub-tile from a parent MemRef.

        Evaluates *base_map* with *indices* to obtain the base coordinates
        in the parent memref, then computes a byte offset using the parent
        strides.  The resulting byte address falls within the same physical
        allocation as parent_ref — this invariant is relied upon by load/store.

        Args:
            context: Core execution context
            parent_ref: Parent MemRef (from construct_memory_view)
            indices: Access indices (one per base_map input dim)
            access_shape: Shape of the accessed sub-tile
            base_map: AffineMap mapping indices → base coordinates

        Returns:
            TileRef (byte-addressed) for the sub-tile
        """
        base_coords = base_map.eval(indices)
        bpe = _bytes_per_elem(parent_ref.dtype)
        offset_elems = sum(coord * stride for coord, stride in zip(base_coords, parent_ref.strides))
        byte_pos = parent_ref.byte_address + offset_elems * bpe

        return TileRef(
            base_ptr=byte_pos,
            shape=access_shape,
            strides=parent_ref.strides,
            dtype=parent_ref.dtype,
            memref=parent_ref,
        )

    @staticmethod
    def _is_contiguous(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> bool:
        """Check if a shape/strides pair describes contiguous (row-major) memory."""
        expected_stride = 1
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if stride != expected_stride:
                return False
            expected_stride *= dim
        return True

    @staticmethod
    def _write_to_lx(context: CoreContext, data: np.ndarray):
        """Write data into the core-local LX scratchpad.

        Advances ``next_ptr`` so subsequent writes don't collide.
        LX capacity accounting is handled by ``CoreContext.track_lx()``
        in ``_execute_operation`` — we only reserve address space here.
        All loaded Tiles always land in LX regardless of source memory space.
        """
        size = data.nbytes
        lx_ptr = context.lx.next_ptr
        context.lx.next_ptr += size
        context.lx.next_ptr = (context.lx.next_ptr + HBMSimulator.STICK_BYTES - 1) & ~(HBMSimulator.STICK_BYTES - 1)
        context.lx.write(lx_ptr, data)

    @staticmethod
    def _flat_memory_offsets(
        base_ptr: int,
        shape: Tuple[int, ...],
        strides: List[int],
        dtype: str,
        coords: Optional[List[Tuple[int, ...]]] = None,
        stick_bytes: Optional[int] = None,
    ) -> Tuple[List[int], Optional[int]]:
        """Linearize N-d coordinates to flat element offsets and optionally count sticks.

        Args:
            base_ptr: Byte address of tile start.
            shape: Tile shape.
            strides: Element strides.
            dtype: Element dtype (for bytes_per_elem).
            coords: Optional coordinate list; if None, enumerates full shape.
            stick_bytes: If set (HBM), count distinct sticks touched. None skips.

        Returns:
            (offsets, unique_sticks) — element offsets and stick count (None for LX).
        """
        offsets = []
        sticks = set() if stick_bytes else None
        bpe = _bytes_per_elem(dtype)
        for coord in (coords if coords is not None else np.ndindex(*shape)):
            o = sum(c * s for c, s in zip(coord, strides))
            offsets.append(o)
            if sticks is not None:
                sticks.add((base_ptr + o * bpe) // stick_bytes)
        return offsets, len(sticks) if sticks is not None else None

    @staticmethod
    def load(
        context: CoreContext,
        tile_ref: TileRef,
        coords: Optional[List[Tuple[int, ...]]] = None,
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load data from HBM or LX into LX and return a Tile.

        All loaded Tiles always land in LX regardless of source memory space:
        - HBM source → DMA read from HBM, write into LX scratchpad.
        - LX source  → logical copy within LX (no physical movement).

        When *coords* is given (coordinate-set path), gathers only the
        elements at those local coordinates and reshapes to *result_shape*.
        When *coords* is None, loads the full tile described by tile_ref
        (contiguous or strided).

        A single ``mem.read`` covers the entire element footprint; no
        per-element dict scans occur.

        Example — loading column 2 of a 4×4 f16 matrix (strided, coords=None)::

            # Parent 4×4 allocation at base_ptr=0x1000, values 0..15
            # tile_ref for column 2: base_ptr=0x1004, shape=(4,), strides=[4]
            #   flat offsets: [0*4, 1*4, 2*4, 3*4] = [0, 4, 8, 12]
            #   span = 13  (max offset + 1)
            #   mem.read(0x1004, 13) -> [2,3,4,5,6,7,8,9,10,11,12,13,14]
            #   gathered = flat[[0,4,8,12]] = [2, 6, 10, 14]  ✓

        Example — upper-triangular load from a 4×4 tile (coords provided)::

            # tile_ref: base_ptr=0x1000, shape=(4,4), strides=[4,1]
            # coords = [(0,0),(0,1),...,(3,3)]  — 10 upper-tri tuples
            #   flat offsets = [0*4+0, 0*4+1, ..., 3*4+3] = [0,1,2,3,5,6,7,10,11,15]
            #   span = 16
            #   mem.read(0x1000, 16) -> flat 0..15
            #   gathered = flat[[0,1,2,3,5,6,7,10,11,15]] = [0,1,2,3,5,6,7,10,11,15]

        Args:
            context: Core execution context
            tile_ref: Tile reference (memref) describing source
            coords: Optional list of local coordinate tuples to gather.
                    Each tuple is 0-based within tile_ref.shape.
            result_shape: Output shape when coords is given; defaults to
                          tile_ref.shape when coords is None.

        Returns:
            Tile value (tensor) loaded into LX
        """
        memref = tile_ref.memref
        is_hbm = memref.memory_space == "HBM"
        stick_bytes = HBMSimulator.STICK_BYTES if is_hbm else None
        if is_hbm:
            stick, intra = tile_ref.hbm_addr
            def _read(n, dtype): return context.hbm.read(stick, n, dtype, intra_byte=intra)
        else:
            ptr = tile_ref.base_ptr
            def _read(n, dtype): return context.lx.read(ptr, n, dtype)

        # Fast path: contiguous tile, no coord filtering — single dict-key read.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            n = int(np.prod(tile_ref.shape))
            data = _read(n, tile_ref.dtype).reshape(tile_ref.shape)
            MemoryOps._write_to_lx(context, data)
            if stick_bytes:
                bpe = _bytes_per_elem(tile_ref.dtype)
                end = tile_ref.base_ptr + n * bpe
                unique_sticks = (
                    (end + stick_bytes - 1) // stick_bytes
                    - tile_ref.base_ptr // stick_bytes
                )
            else:
                unique_sticks = None
            return Tile(data, tile_ref.dtype, tile_ref.shape, unique_sticks)

        # Strided or coord-set path: linearize coords, single read, numpy fancy-index.
        offsets, unique_sticks = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype,
            coords, stick_bytes=stick_bytes
        )
        span = max(offsets) + 1 if offsets else 1
        flat = _read(span, tile_ref.dtype)

        gathered = flat[offsets]
        out_shape = result_shape if result_shape is not None else tile_ref.shape
        data = gathered.reshape(out_shape)

        MemoryOps._write_to_lx(context, data)
        return Tile(data, tile_ref.dtype, out_shape, unique_sticks)

    @staticmethod
    def store(
        context: CoreContext,
        tile: Tile,
        tile_ref: TileRef,
        coords: Optional[List[Tuple[int, ...]]] = None,
    ):
        """Store tile data to HBM or LX.

        - HBM target → DMA write from LX to HBM.
        - LX target  → write directly to LX.

        When *coords* is given (coordinate-set path), scatters tile elements
        to those local coordinates via a read-modify-write on the allocation.
        When *coords* is None, stores the full tile (contiguous or strided).

        A single ``mem.read`` + ``mem.write`` covers the entire footprint;
        no per-element dict scans occur.

        Args:
            context: Core execution context
            tile: Tile value (tensor data) to store
            tile_ref: Tile reference (memref) describing destination
            coords: Optional list of local coordinate tuples to scatter into.
        """
        memref = tile_ref.memref
        if memref.memory_space == "HBM":
            stick, intra = tile_ref.hbm_addr
            def _read(n, dtype): return context.hbm.read(stick, n, dtype, intra_byte=intra)
            def _write(data):    context.hbm.write(stick, data, intra_byte=intra)
        else:
            ptr = tile_ref.base_ptr
            def _read(n, dtype): return context.lx.read(ptr, n, dtype)
            def _write(data):    context.lx.write(ptr, data)

        # Fast path: contiguous tile, no coord filtering — single dict-key write.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            _write(tile.data.flatten())
            return

        # Strided or coord-set path: read-modify-write via scatter offsets.
        offsets, _ = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype, coords
        )
        span = max(offsets) + 1 if offsets else 1
        flat = _read(span, tile_ref.dtype)
        flat[offsets] = tile.data.flatten()
        _write(flat)

    @staticmethod
    def indirect_load(
        context: CoreContext,
        iat: "IndirectAccessTile",
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load data using an indirect access tile (gather pattern).

        Enumerates the variable space, resolves each coordinate tuple
        (direct dims use the variable value, indirect dims look up the
        index in an index memref), then delegates to :meth:`load`.
        """
        vss = iat.variables_space_set
        vso = iat.variables_space_order

        # Enumerate all points in the variable space
        points = vss.enumerate(iat.shape)
        if vso is not None:
            points = [vso.eval(pt) for pt in points]

        # For each point, resolve the actual coordinates in the parent memref
        coords = []
        for pt in points:
            coord = []
            for sub in iat.dim_subscripts:
                if sub["kind"] == "indirect":
                    # Look up the index from the index memref
                    idx_view = iat.index_views[sub["index_view_idx"]]
                    # Compute address into the index tensor
                    idx_coords = tuple(
                        eval_subscript_expr(e, pt) for e in sub["idx_exprs"]
                    )
                    offset = sum(c * s for c, s in zip(idx_coords, idx_view.strides))
                    addr = idx_view.byte_address + offset * _bytes_per_elem(idx_view.dtype)
                    if idx_view.memory_space == "HBM":
                        stick = addr // HBMSimulator.STICK_BYTES
                        intra = addr % HBMSimulator.STICK_BYTES
                        raw = context.hbm.read(stick, 1, idx_view.dtype, intra_byte=intra)
                    else:
                        raw = context.lx.read(addr, 1, idx_view.dtype)
                    coord.append(int(raw[0]))
                elif sub["kind"] == "direct":
                    coord.append(pt[sub["var_index"]])
                elif sub["kind"] == "direct_expr":
                    coord.append(eval_subscript_expr(sub["subscript"], pt))
            coords.append(tuple(coord))

        out_shape = result_shape if result_shape is not None else iat.shape
        return MemoryOps.load(context, iat.parent_ref.to_tile_ref(), coords=coords, result_shape=out_shape)
