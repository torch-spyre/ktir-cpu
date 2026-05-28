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

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
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

    def __init__(
        self,
        context: CoreContext,
        memory_space: str,
        byte_addr: int,
        lx_core_id: Optional[int] = None,
    ):
        self._memory_space = memory_space
        if memory_space == "HBM":
            self.stick_bytes: Optional[int] = HBMSimulator.STICK_BYTES
            self._sim = context.hbm
            stick, intra = divmod(byte_addr, HBMSimulator.STICK_BYTES)
            self._args = (stick,)
            self._kwargs = {"intra_byte": intra}
        else:
            self.stick_bytes = None
            # Per-core routing: when lx_core_id is None or matches the
            # executing core, context.lx is used directly; otherwise we
            # route to a remote LX scratchpad via the ring backend.
            self._sim = context.get_lx(lx_core_id)
            self._args = (byte_addr,)
            self._kwargs = {}

    @classmethod
    def count_sticks(
        cls, memory_space: str, byte_addresses: Iterable[int],
    ) -> Optional[int]:
        """Distinct HBM sticks touched by ``byte_addresses``.

        HBM returns ``len({addr // STICK_BYTES})``; LX returns ``None``
        (the address space has no stick concept). An empty input on the
        HBM path returns ``0`` — a defined "no stick traffic" answer,
        kept distinct from ``None`` which is reserved for "not computed".

        Single source of truth for stick counting: callers route through
        here so ``addr // STICK_BYTES`` arithmetic stays encapsulated.
        """
        if memory_space != "HBM":
            return None
        return len({a // HBMSimulator.STICK_BYTES for a in byte_addresses})

    def read(self, n: int, dtype: str) -> np.ndarray:
        return self._sim.read(*self._args, n, dtype, **self._kwargs)

    def read_scattered(
        self, byte_addresses: List[int], dtype: str,
    ) -> Tuple[np.ndarray, Optional[int]]:
        """Run-batched scatter read; returns ``(values, unique_sticks)``.

        Sorts unique addresses (set-deduped) and merges adjacent ones
        (``diff == bpe``) into contiguous runs. Each run becomes a single
        ``self._sim.read(start, len(run), dtype)`` call — one DMA
        descriptor's worth, matching real hardware behavior. Values are
        then assembled in the caller's order.

        ``unique_sticks`` comes from :meth:`count_sticks` over
        ``byte_addresses`` (HBM: ``int``; LX: ``None``).

        Number of ``sim.read`` calls = run count, bounded by
        ``unique_sticks`` (HBM) or by ``len(set(byte_addresses))`` (LX).
        Best case (dense access) collapses to ``1`` call; worst case
        (fully scattered) issues one call per unique address.

        Reads are addressed by elements of ``byte_addresses`` directly;
        the accessor's ``byte_addr`` (used by :meth:`read`) is unused on
        this path.

        Raises ``ValueError`` on empty ``byte_addresses`` — empty is
        ambiguous in a read context (zero-traffic vs caller bug); use
        :meth:`count_sticks` directly for the pure query.

        Caller invariant: all ``byte_addresses`` must lie within a single
        HBM allocation. Cross-allocation calls are silently wrong —
        physically-adjacent addresses from two allocations merge into one
        run, and ``HBMSimulator._read_flat`` reads only the allocation
        containing the run's start address, zero-filling the rest instead
        of reading from the second allocation. No error is raised.
        Hard-guarding this requires the simulator to expose allocation
        extent; tracked as a follow-up.
        """
        if not byte_addresses:
            raise ValueError("read_scattered called with empty address list")
        unique_sticks = type(self).count_sticks(self._memory_space, byte_addresses)
        np_dtype = _to_np_dtype(dtype)
        bpe = _bytes_per_elem(dtype)

        sorted_unique = sorted(set(byte_addresses))
        # sorted_unique non-empty: byte_addresses guarded above.
        runs: List[List[int]] = [[sorted_unique[0]]]
        for a in sorted_unique[1:]:
            if a - runs[-1][-1] == bpe:
                runs[-1].append(a)
            else:
                runs.append([a])

        cache: Dict[int, Any] = {}
        for run in runs:
            start = run[0]
            n = len(run)
            if self.stick_bytes is not None:
                stick, intra = divmod(start, self.stick_bytes)
                block = self._sim.read(stick, n, dtype, intra_byte=intra)
            else:
                block = self._sim.read(start, n, dtype)
            for i, addr in enumerate(run):
                cache[addr] = block[i]

        values = np.fromiter(
            (cache[a] for a in byte_addresses), dtype=np_dtype,
            count=len(byte_addresses),
        )
        return values, unique_sticks

    def write(self, data: np.ndarray) -> None:
        self._sim.write(*self._args, data, **self._kwargs)


def _enumerate_in_vso_order(iat: "IndirectAccessTile") -> List[Tuple[int, ...]]:
    """Enumerate variable-space points in ``variables_space_order``-permuted order.

    Identity (or absent) ``vso`` returns the natural row-major enumeration;
    otherwise points are sorted by ``vso.eval(pt)`` per RFC 0682 §473.

    Both :func:`_resolve_idx_reads` and :func:`_build_indirect_coords` route
    through this so their pt iteration stays in lockstep — they consume
    ``idx_values`` positionally, so any divergence would silently mismatch
    indirect dims to coords.  Callers are expected to have already rejected
    non-permutation ``vso`` upstream; this function trusts the guard.
    """
    points = iat.variables_space_set.enumerate(iat.shape)
    vso = iat.variables_space_order
    if vso is not None and not vso.is_identity():
        points = sorted(points, key=lambda pt: vso.eval(pt))
    return points


def _resolve_idx_reads(
    context: CoreContext, iat: "IndirectAccessTile",
) -> Tuple[Dict[int, np.ndarray], int]:
    """Read every idx-tensor value the IAT enumeration needs.

    For each indirect dimension, enumerates its address in pt order, then
    issues one ``_MemAccessor.read_scattered`` per index view (so all
    reads to one view share a single accessor and a single dedup pass).

    Returns ``(per_view_values, total_idx_unique_sticks)``:

    * ``per_view_values[idx_view_idx]`` is an ``np.ndarray`` whose ``i``-th
      entry is the idx value resolved for the ``i``-th enumerated point's
      use of that view.  Indirect dims sharing the same view share the
      array (consumed in pt-major, dim-minor order).
    * ``total_idx_unique_sticks`` is the sum across HBM views; ``0`` when
      every idx view lives in LX (LX has no stick concept). The return
      type is always ``int``: callers receiving ``None`` would have to
      special-case it, and the LX-only case is a defined "zero HBM
      traffic" answer, so the function returns the integer directly.

    Per-view loop-invariants (``bpe``, ``strides``, ``byte_address``)
    are hoisted out of the pt loop for million-point scale.

    This is the canonical idx-side resolver: ``indirect_load`` and
    ``indirect_store`` both call it so their stick accounting stays in
    sync (guard symmetry).
    """
    points = _enumerate_in_vso_order(iat)
    indirect_subs = [s for s in iat.dim_subscripts if s.get("kind") == "indirect"]

    # Hoist per-view loop-invariants once before enumerating points.
    per_view_consts: Dict[int, Tuple[int, List[int], int]] = {}
    per_view_addrs: Dict[int, List[int]] = {}
    for sub in indirect_subs:
        iv_idx = sub["index_view_idx"]
        if iv_idx in per_view_consts:
            continue
        iv = iat.index_views[iv_idx]
        per_view_consts[iv_idx] = (
            _bytes_per_elem(iv.dtype), list(iv.strides), iv.byte_address,
        )
        per_view_addrs[iv_idx] = []

    for pt in points:
        for sub in indirect_subs:
            iv_idx = sub["index_view_idx"]
            bpe, strides, base = per_view_consts[iv_idx]
            offset = sum(
                eval_subscript_expr(e, pt) * s
                for e, s in zip(sub["idx_exprs"], strides)
            )
            per_view_addrs[iv_idx].append(base + offset * bpe)

    per_view_values: Dict[int, np.ndarray] = {}
    total_sticks = 0
    for iv_idx, addrs in per_view_addrs.items():
        # Zero-extent enumeration: no points, no addresses, no read.
        # _build_indirect_coords iterates the same enumeration, so it
        # also produces zero coords and never consumes from this view.
        if not addrs:
            continue
        idx_view = iat.index_views[iv_idx]
        accessor = _MemAccessor(
            context, idx_view.memory_space, idx_view.byte_address,
            idx_view.lx_core_id,
        )
        values, sticks = accessor.read_scattered(addrs, idx_view.dtype)
        per_view_values[iv_idx] = values
        if sticks is not None:
            total_sticks += sticks

    return per_view_values, total_sticks


def _build_indirect_coords(
    iat: "IndirectAccessTile", idx_values: Dict[int, np.ndarray],
) -> List[Tuple[int, ...]]:
    """Materialize the parent-tensor coordinate list for an IAT.

    For each enumerated point of ``iat.variables_space_set``, walks
    ``dim_subscripts`` to fill the coordinate tuple:

    * ``direct`` dims read directly from the variable-space point.
    * ``direct_expr`` dims evaluate a quasi-affine expression over the point.
    * ``indirect`` dims consume the next pre-resolved value from
      ``idx_values[iv_idx]`` (set up by :func:`_resolve_idx_reads` in the
      same pt-major, dim-minor order).

    Raises ``IndexError`` on a negative idx value — NumPy fancy-indexing
    silently wraps negatives, so we reject them here.  The check survives
    ``python -O`` (uses ``raise``, not ``assert``).

    Shared by ``indirect_load`` and ``indirect_store`` so their coord
    construction stays in lockstep (guard symmetry).
    """
    points = _enumerate_in_vso_order(iat)
    idx_iters = {iv_idx: iter(values) for iv_idx, values in idx_values.items()}

    coords: List[Tuple[int, ...]] = []
    for pt in points:
        coord: List[int] = []
        for sub in iat.dim_subscripts:
            kind = sub["kind"]
            if kind == "direct":
                coord.append(pt[sub["var_index"]])
            elif kind == "direct_expr":
                coord.append(eval_subscript_expr(sub["subscript"], pt))
            elif kind == "indirect":
                iv_idx = sub["index_view_idx"]
                raw_idx = int(next(idx_iters[iv_idx]))
                if raw_idx < 0:
                    raise IndexError(
                        f"indirect index {raw_idx} from "
                        f"{iat.index_views[iv_idx]} is negative"
                    )
                coord.append(raw_idx)
            else:
                raise ValueError(f"Unknown indirect subscript kind: {kind}")
        coords.append(tuple(coord))
    return coords


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
        lx_core_id: Optional[int] = None,
    ) -> MemRef:
        """Create a hardware-aware memory view (MemRef).

        Builds a MemRef describing a contiguous region in HBM or LX.
        ``lx_core_id``, when set, identifies which core's LX scratchpad
        the data lives in (parsed from #ktdp.spyre_memory_space<LX, core=N>);
        load/store use it to route via context.get_lx().
        """
        return MemRef(
            base_ptr=ptr,
            shape=shape,
            strides=strides,
            memory_space=memory_space,
            dtype=dtype,
            coordinate_set=coordinate_set,
            lx_core_id=lx_core_id,
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
        mgr = _MemAccessor(context, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
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
    ) -> int:
        """Store tile data to HBM or LX.

        - HBM target → DMA write from LX to HBM.
        - LX target  → write directly to LX.

        When *coords* is given (coordinate-set path), scatters tile elements
        to those local coordinates via a read-modify-write on the allocation.
        When *coords* is None, stores the full tile (contiguous or strided).

        Source data layout: ``tile.data`` is read in C-order via
        ``numpy.ndarray.flatten()``, which always returns a contiguous copy.
        Non-contiguous source arrays are handled internally — callers do not
        need to pre-``ascontiguousarray`` the tile. When *coords* is supplied,
        ``coords[i]`` receives the i-th element of ``tile.data`` in C-order.

        A single ``mem.read`` + ``mem.write`` covers the entire footprint;
        no per-element dict scans occur.

        Args:
            context: Core execution context
            tile: Tile value (tensor data) to store
            tile_ref: Tile reference (memref) describing destination
            coords: Optional list of local coordinate tuples to scatter into.

        Returns:
            ``unique_sticks`` (int) — the number of distinct 128-byte HBM
            sticks the write touches. ``0`` for LX destinations (no stick
            concept; LX HBM traffic is zero by definition). The dialect
            handler returns this value so :meth:`LatencyTracker._data_size`
            charges HBM traffic at stick granularity
            (``unique_sticks * STICK_BYTES``) instead of the source tile's
            logical ``nbytes``, which would undercount scatter writes.
        """
        mgr = _MemAccessor(context, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
        stick_bytes = mgr.stick_bytes

        # Fast path: contiguous tile, no coord filtering — single dict-key write.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            mgr.write(tile.data.flatten())
            if not stick_bytes:
                return 0
            n = int(np.prod(tile_ref.shape))
            bpe = _bytes_per_elem(tile_ref.dtype)
            end = tile_ref.base_ptr + n * bpe
            return (
                (end + stick_bytes - 1) // stick_bytes
                - tile_ref.base_ptr // stick_bytes
            )

        # Strided or coord-set path: read-modify-write via scatter offsets.
        offsets, unique_sticks = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype,
            coords, stick_bytes=stick_bytes,
        )
        span = max(offsets) + 1 if offsets else 1
        flat = mgr.read(span, tile_ref.dtype)
        flat[offsets] = tile.data.flatten()
        mgr.write(flat)
        return unique_sticks if unique_sticks is not None else 0

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

        ``variables_space_order``, when non-identity, sets a permuted
        iteration order over the variable space: enumerated points are
        sorted by the map's image and visited in that order.  Subscript
        resolution evaluates each ``idx_exprs`` against the variable-space
        point.  The map must be a coordinate permutation; non-permutation
        maps are rejected with ``ValueError``.  See RFC 0682 §473.
        """
        vso = iat.variables_space_order
        if vso is not None and not vso.is_permutation():
            raise ValueError(
                f"indirect_load: variables_space_order must permute its input "
                f"dimensions; got non-permutation map: {vso.source}"
            )

        # Resolve every idx-tensor read up front: one accessor per index
        # view, one read_scattered call, sticks deduped inside the accessor.
        # Both helpers route their pt enumeration through
        # _enumerate_in_vso_order, so non-identity vso permutes the
        # iteration order consistently across idx reads and coord build
        # (RFC 0682 §473).
        idx_values, idx_unique_sticks = _resolve_idx_reads(context, iat)
        coords = _build_indirect_coords(iat, idx_values)

        out_shape = result_shape if result_shape is not None else iat.shape
        result = MemoryOps.load(
            context, iat.parent_ref.to_tile_ref(),
            coords=coords, result_shape=out_shape,
        )
        result.index_unique_sticks = idx_unique_sticks
        return result

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

        ``p_i = min(B_i)`` (per-axis) is the partition's origin in
        global coords.  This is correct because per-axis ``strides`` on
        ``MemRef`` can only describe a strided rectangle, so any
        non-rectangular ``B_i`` is stored BB-padded inside the
        partition's ``shape`` (see ``MemRef.coordinate_set``).

        Contract on dynamic shapes: callers must supply concrete
        coordinate sets — symbol resolution happens upstream at
        ``construct_memory_view`` (per partition) and
        ``construct_access_tile`` boundaries.  A symbolic ``BoxSet``
        leaking through here will surface as ``IndexError`` from
        ``eval_bound`` rather than a silently wrong answer.  Keeping
        symbol handling out of this function makes the specialise
        boundary single-layer and avoids dead-code on the integration
        path.
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
            return access_tile_set.contains(
                tuple(p[d] - x[d] for d in range(ndim))
            )

        survivors: List[TileRef] = []
        for part in dist_ref.partitions:
            B_i = part.coordinate_set
            if isinstance(B_i, BoxSet) and xA_box is not None:
                # Fast path: O(ndim) intersect on concrete bounds.
                C_i = B_i.intersect(xA_box)
                if C_i.is_empty():
                    continue
                p_i = B_i.lower_bounds()
                coordinate_set_out: CoordinateSet = C_i
            else:
                # Slow path: brute-force enumerate + filter.
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
            mgr = _MemAccessor(context, survivor.memref.memory_space, survivor.base_ptr, survivor.memref.lx_core_id)
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
    ) -> int:
        """Scatter a tile to surviving partitions, symmetric to :meth:`distributed_load`.

        Fast path (BoxSet C_i): slice the source tile rectangularly at
        ``C_i - x``, wrap in a Tile, write through a sub-TileRef built
        on C_i.  np.ascontiguousarray covers the case where the slice
        is a non-contiguous view.

        Slow path (List[Tuple] C_i): per-coord gather/write via one
        read-modify-write.

        Returns:
            Sum of ``unique_sticks`` across all surviving HBM partitions
            (``0`` when every partition lives in LX). Mirrors
            :meth:`distributed_load`'s ``total_unique_sticks`` aggregation
            so :meth:`LatencyTracker._data_size` charges HBM at stick
            granularity instead of the source tile's ``nbytes``.
        """
        x = dist_tile_ref.global_base or (0,) * len(dist_tile_ref.shape)
        ndim = len(dist_tile_ref.shape)

        total_unique_sticks = 0
        for survivor in dist_tile_ref.partitions:
            cs = survivor.coordinate_set
            if isinstance(cs, BoxSet):
                sub = MemoryOps._subtile_ref(survivor, cs)
                slc = tuple(
                    slice(cs.lo[d] - x[d], cs.hi[d] - x[d]) for d in range(ndim)
                )
                src = np.ascontiguousarray(tile.data[slc])
                sub_tile = Tile(src, survivor.dtype, src.shape)
                total_unique_sticks += MemoryOps.store(context, sub_tile, sub)
                continue

            C_i = cs or []
            p_i = survivor.partition_origin or (0,) * ndim
            local_coords = [
                tuple(c[d] - p_i[d] for d in range(ndim)) for c in C_i
            ]
            access_coords = [
                tuple(c[d] - x[d] for d in range(ndim)) for c in C_i
            ]
            mgr = _MemAccessor(context, survivor.memref.memory_space, survivor.base_ptr, survivor.memref.lx_core_id)
            offsets, unique_sticks = MemoryOps._flat_memory_offsets(
                survivor.base_ptr, survivor.shape, survivor.strides, survivor.dtype,
                local_coords, stick_bytes=mgr.stick_bytes,
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mgr.read(span, survivor.dtype)
            for ac, off in zip(access_coords, offsets):
                flat[off] = tile.data[ac]
            mgr.write(flat)
            if unique_sticks is not None:
                total_unique_sticks += unique_sticks

        return total_unique_sticks

    @staticmethod
    def indirect_store(
        context: CoreContext,
        tile: Tile,
        iat: "IndirectAccessTile",
    ) -> int:
        """Store data using an indirect access tile (scatter pattern).

        Mirror of :meth:`indirect_load`. Enumerates the variable space,
        resolves each coordinate tuple (direct dims use the variable value,
        indirect dims look up the index in an index memref), then delegates
        to :meth:`store`.

        Coordinate collisions (multiple source elements mapping to the same
        destination coordinate) are *implementation-defined*; the current
        behavior is last-writer-wins via NumPy fancy-index assignment.

        Returns:
            Total ``unique_sticks`` touched on HBM — sum of the parent
            tile's destination sticks (from :meth:`store`) and the
            idx-side sticks (from :func:`_resolve_idx_reads`). ``0`` when
            both the parent and every idx view live in LX (no HBM
            traffic). Returned via the dialect handler as the op result
            so :meth:`LatencyTracker._data_size` can charge stick-granular
            HBM cost — guard symmetry with :meth:`indirect_load`, which
            stamps the same totals on the result Tile.
        """
        # MLIR type system should already enforce shape match; raise here so a
        # mismatch surfaces clearly instead of as an opaque NumPy shape error.
        if tuple(tile.shape) != tuple(iat.shape):
            raise ValueError(
                f"indirect_store: source tile shape {tuple(tile.shape)} does not "
                f"match IAT shape {tuple(iat.shape)}"
            )

        vso = iat.variables_space_order
        if vso is not None and not vso.is_permutation():
            raise ValueError(
                f"indirect_store: variables_space_order must permute its input "
                f"dimensions; got non-permutation map: {vso.source}"
            )

        # Resolve idx reads (returns idx_unique_sticks: int, 0 for all-LX
        # views) and delegate the data write to MemoryOps.store (returns
        # int: HBM stick count, 0 for LX).  Both helpers enumerate via
        # _enumerate_in_vso_order so non-identity vso permutes the
        # iteration order consistently with indirect_load (RFC 0682 §473).
        idx_values, idx_unique_sticks = _resolve_idx_reads(context, iat)
        coords = _build_indirect_coords(iat, idx_values)
        data_sticks = MemoryOps.store(
            context, tile, iat.parent_ref.to_tile_ref(), coords=coords,
        )
        return data_sticks + idx_unique_sticks
