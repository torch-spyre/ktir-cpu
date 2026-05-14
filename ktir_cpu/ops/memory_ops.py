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

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..affine import AffineMap, AffineSet, BoxSet
from ..dialects.ktdp_helpers import eval_subscript_expr
from ..dtypes import bytes_per_elem as _bytes_per_elem, to_np_dtype as _to_np_dtype
from ..ir_types import (
    CoordinateSet, DistributedMemRef, DistributedTileRef, MemRef, Tile, TileRef,
)
from ..grid import CoreContext
from ..memory import HBMSimulator


class _MemAccessor:
    """Resolves a (context, memory_space, byte_addr) triple into simulator
    read/write calls.

    This is the single place in the codebase that manages the intra-stick byte
    offset abstraction: HBMSimulator requires a (stick, intra_byte) address
    pair while LXScratchpad uses a plain byte address.  The accessor consumes
    only ``memory_space`` (for simulator dispatch) and an absolute
    ``byte_addr``; the byte_addr must live in physical memory matching the
    given memory_space.  Callers do not need to manufacture a MemRef.

    ``stick_bytes`` is exposed for callers that need to count distinct HBM
    sticks touched by an access (latency accounting); it is None for LX.

    To extend to a new memory space, add a branch in ``__init__`` that
    populates ``_args`` and ``_kwargs`` appropriately — ``read`` and ``write``
    require no changes.
    """

    def __init__(self, context: CoreContext, memory_space: str, byte_addr: int):
        if memory_space == "HBM":
            self.stick_bytes: Optional[int] = HBMSimulator.STICK_BYTES
            self._sim = context.hbm
            stick, intra = divmod(byte_addr, HBMSimulator.STICK_BYTES)
            self._args = (stick,)
            self._kwargs = {"intra_byte": intra}
        else:
            self.stick_bytes = None
            self._sim = context.lx
            self._args = (byte_addr,)
            self._kwargs = {}

    def read(self, n: int, dtype: str) -> np.ndarray:
        return self._sim.read(*self._args, n, dtype, **self._kwargs)

    def write(self, data: np.ndarray) -> None:
        self._sim.write(*self._args, data, **self._kwargs)

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
        mgr = _MemAccessor(context, tile_ref.memref.memory_space, tile_ref.base_ptr)
        stick_bytes = mgr.stick_bytes

        # Fast path: contiguous tile, no coord filtering — single dict-key read.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            n = int(np.prod(tile_ref.shape))
            data = mgr.read(n, tile_ref.dtype).reshape(tile_ref.shape)
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
        flat = mgr.read(span, tile_ref.dtype)

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
        mgr = _MemAccessor(context, tile_ref.memref.memory_space, tile_ref.base_ptr)

        # Fast path: contiguous tile, no coord filtering — single dict-key write.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            mgr.write(tile.data.flatten())
            return

        # Strided or coord-set path: read-modify-write via scatter offsets.
        offsets, _ = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype, coords
        )
        span = max(offsets) + 1 if offsets else 1
        flat = mgr.read(span, tile_ref.dtype)
        flat[offsets] = tile.data.flatten()
        mgr.write(flat)

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
                    raw = _MemAccessor(context, idx_view.memory_space, addr).read(1, idx_view.dtype)
                    coord.append(int(raw[0]))
                elif sub["kind"] == "direct":
                    coord.append(pt[sub["var_index"]])
                elif sub["kind"] == "direct_expr":
                    coord.append(eval_subscript_expr(sub["subscript"], pt))
            coords.append(tuple(coord))

        out_shape = result_shape if result_shape is not None else iat.shape
        return MemoryOps.load(context, iat.parent_ref.to_tile_ref(), coords=coords, result_shape=out_shape)

    # ------------------------------------------------------------------
    # Distributed memory views (RFC 0682 §3.3)
    #
    # Naming used throughout:
    #   x   = global_base = base_map.eval(indices) — global origin of
    #         the access tile
    #   A   = access_tile_set, in local coords 0..access_shape-1; None
    #         means the full box [0, access_shape)
    #   x+A = global footprint of the access tile
    #   B_i = partition i's coordinate_set, in global coords
    #   C_i = (x + A) ∩ B_i — global coords covered by both the access
    #         tile and partition i; per-survivor coordinate_set
    #   p_i = min(B_i) — partition i's origin in global coords
    #
    # distributed_load consumes C_i and p_i directly:
    #   load coords (partition-local) = C_i - p_i
    #   output coords (access-local)  = C_i - x
    # ------------------------------------------------------------------

    @staticmethod
    def distributed_tile_access(
        dist_ref: DistributedMemRef,
        access_shape: Tuple[int, ...],
        base_map: AffineMap,
        indices: List[int],
        access_tile_set: Optional[Union[BoxSet, AffineSet]] = None,
    ) -> DistributedTileRef:
        """Resolve partition routing once, return a DistributedTileRef.

        Fast path (BoxSet): when both ``B_i`` and the access set ``A``
        (or the implicit full-box A) are :class:`BoxSet`, compute
        ``C_i = B_i ∩ (x + A)`` in O(ndim) via ``translate`` +
        ``intersect`` and store ``C_i`` as a ``BoxSet``.  Skip empty
        intersections.

        Slow path (AffineSet on either side): enumerate B_i over the
        global shape, filter by membership in ``x + A``, store C_i as
        a ``List[Tuple[int, ...]]``.

        Each survivor inherits ``memref = P_i``, ``base_ptr =
        P_i.byte_address``, and ``strides = P_i.strides``.  Load/store
        translate per-coord via ``C_i - p_i``.
        """
        global_base = tuple(base_map.eval(indices))
        x = global_base
        ndim = len(dist_ref.shape)

        # Pre-compute (x + A) as a BoxSet when possible.  None ⇒ A is
        # the implicit full box [0, access_shape).
        xA_box: Optional[BoxSet] = None
        if access_tile_set is None:
            xA_box = BoxSet(
                lo=tuple(x),
                hi=tuple(x[d] + access_shape[d] for d in range(ndim)),
            )
        elif isinstance(access_tile_set, BoxSet):
            xA_box = access_tile_set.translate(x)

        def _in_xA(p: Tuple[int, ...]) -> bool:
            """Slow-path membership test: point ∈ x + A."""
            if access_tile_set is None:
                return all(0 <= p[d] - x[d] < access_shape[d] for d in range(ndim))
            return access_tile_set.contains(tuple(p[d] - x[d] for d in range(ndim)))

        survivors: List[TileRef] = []
        for part in dist_ref.partitions:
            B_i = part.coordinate_set
            if isinstance(B_i, BoxSet) and xA_box is not None:
                # Fast path: O(ndim) intersect
                C_i = B_i.intersect(xA_box)
                if C_i.is_empty():
                    continue
                p_i = B_i.lower_bounds()
                coordinate_set_out: CoordinateSet = C_i
            else:
                # Slow path: brute-force enumerate + filter
                B_i_pts = B_i.enumerate(dist_ref.shape)
                if not B_i_pts:
                    continue
                p_i = tuple(min(pt[d] for pt in B_i_pts) for d in range(ndim))
                C_i_pts = [pt for pt in B_i_pts if _in_xA(pt)]
                if not C_i_pts:
                    continue
                coordinate_set_out = C_i_pts

            survivors.append(TileRef(
                base_ptr=part.byte_address,
                shape=part.shape,
                strides=list(part.strides),
                memref=part,
                dtype=part.dtype,
                coordinate_set=coordinate_set_out,
                partition_origin=p_i,
            ))

        if not survivors:
            raise ValueError(
                f"distributed_tile_access: no partition covers access region "
                f"global_base={global_base} shape={access_shape}"
            )
        return DistributedTileRef(
            partitions=survivors,
            shape=dist_ref.shape,
            dtype=dist_ref.dtype,
            global_base=global_base,
        )

    @staticmethod
    def _subtile_ref(survivor: TileRef, box: BoxSet) -> TileRef:
        """Build a TileRef covering exactly *box* (in global coords) within *survivor*.

        Inherits the survivor's strides verbatim; only ``shape`` shrinks
        to the box extent and ``base_ptr`` shifts to the box's local
        origin (``box.lo - p_i``, in element units, scaled by bpe).  The
        resulting sub-TileRef plugs into :meth:`load` / :meth:`store`,
        whose strided iteration lands each element at the byte offset
        the parent layout dictates — row-major and column-packed
        partitions both work uniformly without caller-side transposes.
        """
        ndim = len(survivor.shape)
        p_i = survivor.partition_origin or (0,) * ndim
        local_lo = tuple(box.lo[d] - p_i[d] for d in range(ndim))
        sub_shape = tuple(box.hi[d] - box.lo[d] for d in range(ndim))
        bpe = _bytes_per_elem(survivor.dtype)
        byte_offset = sum(local_lo[d] * survivor.strides[d] for d in range(ndim)) * bpe
        return TileRef(
            base_ptr=survivor.base_ptr + byte_offset,
            shape=sub_shape,
            strides=list(survivor.strides),
            memref=survivor.memref,
            dtype=survivor.dtype,
        )

    @staticmethod
    def distributed_load(
        context: CoreContext,
        dist_tile_ref: DistributedTileRef,
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Gather elements across surviving partitions into a single LX-resident Tile.

        Fast path (BoxSet C_i): build a sub-TileRef of the partition
        covering exactly C_i, delegate the read to :meth:`load`, and
        slot the returned tile into a rectangular slice of the output
        buffer.  One NumPy slice assignment per partition.

        Slow path (List[Tuple] C_i): per-coord scatter — translate C_i
        to partition-local coords, issue one batched read, write each
        element into the access-local position of the output buffer.
        """
        x = dist_tile_ref.global_base or (0,) * len(dist_tile_ref.shape)
        ndim = len(dist_tile_ref.shape)
        out_shape = (
            tuple(result_shape) if result_shape is not None else tuple(dist_tile_ref.shape)
        )
        out = np.zeros(out_shape, dtype=_to_np_dtype(dist_tile_ref.dtype))

        total_unique_sticks = 0
        for survivor in dist_tile_ref.partitions:
            cs = survivor.coordinate_set
            if isinstance(cs, BoxSet):
                # Fast path: rectangular sub-tile.
                sub = MemoryOps._subtile_ref(survivor, cs)
                tile = MemoryOps.load(context, sub)
                # access-local rectangle = C_i - x
                slc = tuple(
                    slice(cs.lo[d] - x[d], cs.hi[d] - x[d]) for d in range(ndim)
                )
                out[slc] = tile.data
                if tile.unique_sticks is not None:
                    total_unique_sticks += tile.unique_sticks
                continue

            # Slow path: List[Tuple[int, ...]] enumeration.
            C_i = cs or []
            p_i = survivor.partition_origin or (0,) * ndim
            local_coords = [
                tuple(c[d] - p_i[d] for d in range(ndim)) for c in C_i
            ]
            access_coords = [
                tuple(c[d] - x[d] for d in range(ndim)) for c in C_i
            ]
            mgr = _MemAccessor(context, survivor.memref.memory_space, survivor.base_ptr)
            offsets, unique_sticks = MemoryOps._flat_memory_offsets(
                survivor.base_ptr, survivor.shape, survivor.strides, survivor.dtype,
                local_coords, stick_bytes=mgr.stick_bytes,
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mgr.read(span, survivor.dtype)
            for ac, off in zip(access_coords, offsets):
                out[ac] = flat[off]
            if unique_sticks is not None:
                total_unique_sticks += unique_sticks

        MemoryOps._write_to_lx(context, out)
        return Tile(
            out,
            dist_tile_ref.dtype,
            out_shape,
            total_unique_sticks if total_unique_sticks else None,
        )

    @staticmethod
    def distributed_store(
        context: CoreContext,
        tile: Tile,
        dist_tile_ref: DistributedTileRef,
    ) -> None:
        """Scatter a tile to surviving partitions, symmetric to :meth:`distributed_load`.

        Fast path (BoxSet C_i): slice the source tile rectangularly at
        ``C_i - x``, wrap in a Tile, write through a sub-TileRef built
        on C_i.  np.ascontiguousarray covers the case where the slice
        is a non-contiguous view.

        Slow path (List[Tuple] C_i): per-coord gather/write via one
        read-modify-write.
        """
        x = dist_tile_ref.global_base or (0,) * len(dist_tile_ref.shape)
        ndim = len(dist_tile_ref.shape)

        for survivor in dist_tile_ref.partitions:
            cs = survivor.coordinate_set
            if isinstance(cs, BoxSet):
                sub = MemoryOps._subtile_ref(survivor, cs)
                slc = tuple(
                    slice(cs.lo[d] - x[d], cs.hi[d] - x[d]) for d in range(ndim)
                )
                src = np.ascontiguousarray(tile.data[slc])
                sub_tile = Tile(src, survivor.dtype, src.shape)
                MemoryOps.store(context, sub_tile, sub)
                continue

            C_i = cs or []
            p_i = survivor.partition_origin or (0,) * ndim
            local_coords = [
                tuple(c[d] - p_i[d] for d in range(ndim)) for c in C_i
            ]
            access_coords = [
                tuple(c[d] - x[d] for d in range(ndim)) for c in C_i
            ]
            mgr = _MemAccessor(context, survivor.memref.memory_space, survivor.base_ptr)
            offsets, _ = MemoryOps._flat_memory_offsets(
                survivor.base_ptr, survivor.shape, survivor.strides, survivor.dtype,
                local_coords,
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mgr.read(span, survivor.dtype)
            for ac, off in zip(access_coords, offsets):
                flat[off] = tile.data[ac]
            mgr.write(flat)
