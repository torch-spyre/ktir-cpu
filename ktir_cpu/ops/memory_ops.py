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

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..affine import AffineMap, AffineSet
from ..dialects.ktdp_helpers import eval_subscript_expr
from ..dtypes import bytes_per_elem as _bytes_per_elem, to_np_dtype as _to_np_dtype
from ..ir_types import DistributedMemRef, DistributedTileRef, MemRef, Tile, TileRef
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
    # ------------------------------------------------------------------

    @staticmethod
    def distributed_tile_access(
        dist_ref: DistributedMemRef,
        access_shape: Tuple[int, ...],
        base_map: AffineMap,
        indices: List[int],
        access_tile_set: Optional[AffineSet] = None,
    ) -> DistributedTileRef:
        """Construct a DistributedTileRef from a DistributedMemRef + access tile.

        v1 (C1): pass-through partition resolution.  Each partition becomes
        a survivor TileRef pointing to that partition's full allocation;
        per-coord routing happens at load/store time via
        :meth:`DistributedMemRef.find_partition`.  C2 will narrow each
        survivor to its own ``C_i = (x + A) ∩ B_i``.
        """
        global_base = tuple(base_map.eval(indices))
        survivors: List[TileRef] = []
        for part in dist_ref.partitions:
            survivors.append(TileRef(
                base_ptr=part.byte_address,
                shape=part.shape,
                strides=list(part.strides),
                memref=part,
                dtype=part.dtype,
            ))
        return DistributedTileRef(
            partitions=survivors,
            shape=dist_ref.shape,
            dtype=dist_ref.dtype,
            global_base=global_base,
        )

    @staticmethod
    def _enumerate_dist_coords(
        dist_tile_ref: DistributedTileRef, result_shape: Tuple[int, ...]
    ) -> List[Tuple[int, ...]]:
        """Enumerate the global coords covered by a DistributedTileRef access.

        v1 (C1): the access region is the full result_shape, anchored at
        ``global_base``.  C2 will replace this with per-survivor C_i
        enumeration.
        """
        x = dist_tile_ref.global_base
        if x is None:
            x = (0,) * len(result_shape)
        ndim = len(result_shape)
        return [
            tuple(x[d] + idx[d] for d in range(ndim))
            for idx in np.ndindex(*result_shape)
        ]

    @staticmethod
    def _route_coords_to_partitions(
        dist_tile_ref: DistributedTileRef,
        coords: List[Tuple[int, ...]],
    ) -> Dict[int, Tuple[List[int], List[Tuple[int, ...]]]]:
        """Group global *coords* by owning partition and translate to local.

        Returns ``{partition_idx: (orig_positions, local_coords)}``.
        ``orig_positions[k]`` is the index into *coords* that produced
        ``local_coords[k]`` — used to scatter per-partition results back
        into the flat output buffer.

        Per-partition origin (= ``min(B_i)``) is computed lazily and once.
        """
        ndim = len(dist_tile_ref.shape)
        origins: Dict[int, Tuple[int, ...]] = {}
        groups: Dict[int, Tuple[List[int], List[Tuple[int, ...]]]] = {}
        # Build a virtual DistributedMemRef-like view from the survivor
        # MemRefs so we can reuse find_partition semantics without the
        # dataclass validation.  In C1 the survivors' memref fields are
        # exactly the partitions of the original DistributedMemRef.
        partition_memrefs = [s.memref for s in dist_tile_ref.partitions]
        for pos, coord in enumerate(coords):
            part_idx = None
            for i, p_memref in enumerate(partition_memrefs):
                if p_memref.coordinate_set.contains(coord):
                    part_idx = i
                    break
            if part_idx is None:
                raise IndexError(
                    f"distributed access: no partition contains global coord {coord}"
                )
            if part_idx not in origins:
                p_set = partition_memrefs[part_idx].coordinate_set
                pts = p_set.enumerate(dist_tile_ref.shape)
                if not pts:
                    raise ValueError(
                        f"distributed access: partition {part_idx} coordinate_set "
                        f"is empty over shape {dist_tile_ref.shape}"
                    )
                origins[part_idx] = tuple(min(p[d] for p in pts) for d in range(ndim))
            origin = origins[part_idx]
            local = tuple(int(coord[d] - origin[d]) for d in range(ndim))
            pos_list, local_list = groups.setdefault(part_idx, ([], []))
            pos_list.append(pos)
            local_list.append(local)
        return groups

    @staticmethod
    def distributed_load(
        context: CoreContext,
        dist_tile_ref: DistributedTileRef,
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Gather elements across partitions and return a single LX-resident Tile.

        For each global coord in the access region, route to the owning
        partition, translate to that partition's local coord, issue one
        batched read per partition group, and scatter the values back into
        a flat output buffer indexed by the coord's original position.
        Writes the result into LX so the caller sees the same contract as
        :meth:`load`.
        """
        out_shape = (
            tuple(result_shape) if result_shape is not None else tuple(dist_tile_ref.shape)
        )
        coords = MemoryOps._enumerate_dist_coords(dist_tile_ref, out_shape)
        n_total = len(coords)
        out_flat = np.zeros(n_total, dtype=_to_np_dtype(dist_tile_ref.dtype))
        groups = MemoryOps._route_coords_to_partitions(dist_tile_ref, coords)

        total_unique_sticks = 0
        for part_idx, (orig_positions, local_coords) in groups.items():
            survivor = dist_tile_ref.partitions[part_idx]
            mgr = _MemAccessor(context, survivor.memref.memory_space, survivor.base_ptr)
            offsets, unique_sticks = MemoryOps._flat_memory_offsets(
                survivor.base_ptr, survivor.shape, survivor.strides, survivor.dtype,
                local_coords, stick_bytes=mgr.stick_bytes,
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mgr.read(span, survivor.dtype)
            out_flat[orig_positions] = flat[offsets]
            if unique_sticks is not None:
                total_unique_sticks += unique_sticks

        data = out_flat.reshape(out_shape)
        MemoryOps._write_to_lx(context, data)
        return Tile(
            data,
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
        """Scatter a tile across partitions, symmetric to :meth:`distributed_load`.

        For each global coord covered by the access region, route to the
        owning partition and translate to that partition's local coord;
        per partition, do one read-modify-write covering the partition's
        scatter targets.
        """
        coords = MemoryOps._enumerate_dist_coords(dist_tile_ref, tile.shape)
        groups = MemoryOps._route_coords_to_partitions(dist_tile_ref, coords)
        flat_values = tile.data.flatten()

        for part_idx, (orig_positions, local_coords) in groups.items():
            survivor = dist_tile_ref.partitions[part_idx]
            mgr = _MemAccessor(context, survivor.memref.memory_space, survivor.base_ptr)
            offsets, _ = MemoryOps._flat_memory_offsets(
                survivor.base_ptr, survivor.shape, survivor.strides, survivor.dtype,
                local_coords,
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mgr.read(span, survivor.dtype)
            flat[offsets] = flat_values[orig_positions]
            mgr.write(flat)
