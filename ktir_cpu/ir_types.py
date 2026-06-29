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
Core IR data types.

Types are ordered by the data-flow pipeline:

- MemRef: Hardware-aware memory view (construct_memory_view result; stick-indexed for HBM)
- DistributedMemRef: Distributed analogue of MemRef — list of per-partition MemRefs
  (construct_distributed_memory_view result)
- TileRef: Byte-addressed sub-tile view (construct_access_tile result; produced from MemRef)
- DistributedTileRef: Distributed analogue of TileRef — list of per-partition TileRef
  survivors after access-set intersection (distributed_tile_access result)
- Tile: Data value backed by a NumPy array (load result; tensor in MLIR)
- AccessTile / IndirectAccessTile: Coordinate descriptors for load/store
- Operation: Single IR operation
- IRFunction / IRModule: Top-level IR containers
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .affine import AffineMap, AffineSet, BoxSet

# Type alias for TileRef.coordinate_set: parsed affine sets lower to BoxSet
# at parse time when axis-aligned; non-box sets stay as AffineSet; the
# distributed slow path stores a pre-enumerated list of points; None means
# the field hasn't been set (ordinary single-allocation TileRefs).
CoordinateSet = Union[BoxSet, AffineSet, List[Tuple[int, ...]]]


@dataclass
class MemRef:
    """Hardware-aware memory view (result of construct_memory_view).

    Represents a logical view over allocated memory.  ``base_ptr`` is an
    element index — the number of elements from the start of the address
    space, matching what MLIR pointer operands carry.  Use ``byte_address``
    to get the absolute byte position for load/store.
    """
    base_ptr: int              # element index (both HBM and LX)
    shape: Tuple[int, ...]
    strides: List[int]         # element counts
    memory_space: str          # "HBM" or "LX"
    dtype: str = "f16"
    # ``coordinate_set`` is the set of global coords this MemRef owns.
    # Per-axis ``strides`` only describes a strided rectangle, so a
    # non-rectangular set is stored BB-padded inside ``shape`` (slots
    # outside the set are unused but addressable).  Hence the partition
    # origin in global coords is ``min(coordinate_set)`` (= BB lower
    # corner) — relied on by ``distributed_tile_access`` for ``p_i``.
    coordinate_set: Optional[AffineSet] = None
    # Set when memory_space="LX" and a core index was specified via
    # #ktdp.spyre_memory_space<LX, core = N>.  None means "the executing
    # core's own LX scratchpad" (default routing).
    lx_core_id: Optional[int] = None

    def __post_init__(self):
        valid = ("HBM", "LX")
        if self.memory_space not in valid:
            raise ValueError(
                f"Invalid memory_space {self.memory_space!r}. Must be one of {valid}."
            )
        if self.lx_core_id is not None and self.memory_space != "LX":
            raise ValueError(
                f"lx_core_id may only be set when memory_space is 'LX', "
                f"got memory_space={self.memory_space!r}"
            )

    @property
    def byte_address(self) -> int:
        """Absolute byte address of this memref's base in its memory space."""
        from .dtypes import bytes_per_elem
        return self.base_ptr * bytes_per_elem(self.dtype)

    def to_tile_ref(self) -> 'TileRef':
        """Convert to a byte-addressed TileRef for load/store operations."""
        return TileRef(
            base_ptr=self.byte_address,
            shape=self.shape,
            strides=self.strides,
            dtype=self.dtype,
            memref=self,
        )

    def split_addr(self, byte_addr: int) -> Tuple[int, int]:
        """Split a byte address into a (main, intra) pair for memory reads/writes.

        The meaning of the pair depends on memory space:
        - HBM: ``(stick_index, intra_byte_offset)``
        - LX:  ``(byte_addr, 0)`` — LX is byte-addressed with no sub-unit concept.

        New memory spaces add a branch here; callers are unchanged.
        """
        if self.memory_space == "HBM":
            from .memory import HBMSimulator
            return byte_addr // HBMSimulator.STICK_BYTES, byte_addr % HBMSimulator.STICK_BYTES
        return byte_addr, 0

    def size_bytes(self) -> int:
        """Calculate size in bytes."""
        from .dtypes import bytes_per_elem
        return int(np.prod(self.shape) * bytes_per_elem(self.dtype))


@dataclass
class DistributedMemRef:
    """Distributed memory view: composition of N per-partition MemRefs.

    Produced by ``ktdp.construct_distributed_memory_view``.  Each partition
    is a plain :class:`MemRef` carrying its own ``coordinate_set`` (= B_i
    in global coords), ``memory_space``, ``base_ptr``, and ``strides``;
    this wrapper records the global logical shape and can dispatch a
    global coordinate to the first partition whose set contains it.

    The op does not allocate or move data — this is bookkeeping only.
    Access-time partition resolution (intersect with the access tile and
    return the survivors) is performed by
    ``MemoryOps.distributed_tile_access`` and yields a
    :class:`DistributedTileRef`.
    """
    partitions: List[MemRef]
    shape: Tuple[int, ...]   # global logical shape (coordinate_sets use these coords)
    dtype: str

    def __post_init__(self):
        if not self.partitions:
            raise ValueError("DistributedMemRef requires at least one partition")
        for i, p in enumerate(self.partitions):
            if p.coordinate_set is None:
                raise ValueError(
                    f"DistributedMemRef partition {i} must have a coordinate_set"
                )
            if p.dtype != self.dtype:
                raise ValueError(
                    f"DistributedMemRef partition {i} dtype {p.dtype!r} "
                    f"does not match view dtype {self.dtype!r}"
                )

    def find_partition(self, coord: Tuple[int, ...]) -> Tuple[int, MemRef]:
        """Return ``(index, partition)`` whose coordinate_set contains *coord*.

        Returns the first match; per RFC 0682 §3.3, overlapping coordinate
        sets produce unspecified behavior, and "first match" is a legal
        (and conveniently deterministic) resolution.
        """
        for i, p in enumerate(self.partitions):
            if p.coordinate_set.contains(coord):
                return i, p
        raise IndexError(
            f"No partition of DistributedMemRef contains global coord {coord}"
        )


@dataclass
class TileRef:
    """Byte-addressed tile view for load/store operations.

    Produced by ``tile_access()`` from a parent MemRef.  ``base_ptr`` is
    always an absolute byte address regardless of memory space.  Holds a
    reference to the parent ``memref`` for memory-space-aware queries
    (e.g. ``unique_sticks``).
    """
    base_ptr: int              # always byte address
    shape: Tuple[int, ...]
    strides: List[int]         # element counts
    memref: 'MemRef'           # parent MemRef — always set; owns memory_space and hw address conversion
    dtype: str = "f16"
    # Per-survivor metadata set by distributed_tile_access (None on
    # ordinary single-allocation TileRefs).  ``coordinate_set`` is one of:
    #   - BoxSet: axis-aligned C_i, produced by the BoxSet fast path in
    #     distributed_tile_access (B_i and A both BoxSet).  O(ndim) ops.
    #   - List[Tuple[int,...]]: pre-enumerated C_i points from the slow
    #     path (B_i or A is AffineSet — non-box).
    coordinate_set: Optional[CoordinateSet] = None
    partition_origin: Optional[Tuple[int, ...]] = None      # p_i = min(B_i) in global coords

    def size_bytes(self) -> int:
        """Calculate size in bytes."""
        from .dtypes import bytes_per_elem
        return int(np.prod(self.shape) * bytes_per_elem(self.dtype))


@dataclass
class DistributedTileRef:
    """Distributed analogue of TileRef: per-partition survivors of an access.

    Produced by ``MemoryOps.distributed_tile_access``.  Each survivor is a
    plain :class:`TileRef` whose ``memref`` points to the partition's
    :class:`MemRef` (so memory-space dispatch and per-core LX routing
    work without any DistributedMemRef-aware code at the load/store call
    site).

    The wrapper records the global logical shape (inherited from the
    DistributedMemRef the access was issued against) and ``global_base``
    — the origin of the access tile in global coords (= ``base_map.eval(indices)``).
    distributed_load and distributed_store consume this object directly.
    """
    partitions: List['TileRef']
    shape: Tuple[int, ...]                           # global logical shape
    dtype: str
    global_base: Optional[Tuple[int, ...]] = None    # x = base_map.eval(indices); set by
                                                     # distributed_tile_access, None on
                                                     # construct_distributed_memory_view

    def __post_init__(self):
        if not self.partitions:
            raise ValueError("DistributedTileRef requires at least one partition")
        for i, p in enumerate(self.partitions):
            if p.dtype != self.dtype:
                raise ValueError(
                    f"DistributedTileRef partition {i} dtype {p.dtype!r} "
                    f"does not match view dtype {self.dtype!r}"
                )


@dataclass
class Tile:
    """Data value backed by a NumPy array.

    Represents an actual tensor of element data (e.g. ``tensor<1024xf16>``
    in MLIR).  Produced by load operations and consumed by compute ops.
    """
    data: np.ndarray
    dtype: str  # see dtypes.SUPPORTED_DTYPES
    shape: Tuple[int, ...]
    # Number of distinct HBM sticks touched by the load that produced this
    # tile.  Set by ``MemoryOps.load`` for all access patterns.  None only
    # for tiles produced by compute ops (not loaded from memory).
    unique_sticks: Optional[int] = None
    # Number of distinct HBM sticks touched by index-tensor reads during
    # an indirect load/store.  Tracked separately from ``unique_sticks``
    # so that ``coalescing_efficiency`` retains its data-layout meaning.
    # None for direct loads (no index tensors) and compute-produced tiles.
    index_unique_sticks: Optional[int] = None
    # Total bytes moved across cores by the comm op that produced this
    # tile.  Set by ``ktdp.inter_tile_reduce`` (and future delivery ops)
    # from the transport backend's accumulated send total.  None for any
    # tile not produced by a comm op.  The latency tracker reads this
    # off the result for stick-granular comm bandwidth accounting,
    # mirroring how ``unique_sticks`` carries memory provenance.  Not
    # propagated by ``copy()`` — the bytes belong to the production
    # event, not to copies of the data.
    comm_bytes: Optional[int] = None

    def copy(self) -> 'Tile':
        """Create a deep copy of this tile.

        Used by comm ops for message buffers / reduce accumulators.
        When cross-core communication is fixed (gap analysis K1/K2),
        reconsider whether unique_sticks should be recomputed for the
        target device's base_ptr — currently we propagate the source
        value since the memory layout is abstracted away from Tile.
        """
        return Tile(
            self.data.copy(), self.dtype, self.shape,
            self.unique_sticks, self.index_unique_sticks,
        )

    def size_bytes(self) -> int:
        """Return size in bytes."""
        return self.data.nbytes

    @property
    def coalescing_efficiency(self) -> Optional[float]:
        """Ratio of packed traffic to actual stick traffic.

        Defined as ``data.nbytes / (unique_sticks * HBMSimulator.STICK_BYTES)``.
        Ranges from ``bytes_per_elem / STICK_BYTES`` (worst: every
        element on its own stick) to ``1.0`` (best: sticks fully packed).

        Returns ``None`` when ``unique_sticks`` is not set (compute-produced tiles).
        """
        if self.unique_sticks is None or self.unique_sticks == 0:
            return None
        from .memory import HBMSimulator
        return self.data.nbytes / (self.unique_sticks * HBMSimulator.STICK_BYTES)


@dataclass
class AccessTile:
    """Coordinate access tile referencing a sub-region of a MemRef.

    Holds the affine attributes that describe which coordinates of the
    parent memref to access.  Load and store operations use these to
    find the actual memory location.

    ``parent_ref`` is a :class:`TileRef` for the single-allocation case
    or a :class:`DistributedTileRef` when the access was constructed
    against a distributed memory view (in which case partition routing
    has already been resolved by ``MemoryOps.distributed_tile_access``).
    """
    parent_ref: Union[TileRef, 'DistributedTileRef']
    shape: Tuple[int, ...]
    base_map: AffineMap                                  # always present; synthesized as identity if absent in MLIR
    coordinate_set: Optional[AffineSet] = None     # parsed access_tile_set; None if omitted
    coordinate_order: Optional[AffineMap] = None   # parsed access_tile_order; None if omitted


@dataclass
class IndirectAccessTile:
    """Indirect access tile for gather/scatter patterns.

    Each dimension of the target tensor is indexed either:
    - Directly via an intermediate variable value, or
    - Indirectly via a lookup into an index memory view.
    """
    parent_ref: MemRef                          # primary memory view (e.g. X)
    shape: Tuple[int, ...]                      # output access tile shape
    dim_subscripts: List[Dict[str, Any]]        # per-dim descriptor (kind, var_indices, etc.)
    index_views: List[MemRef]                   # index memrefs for indirect dims (used for byte_address)
    variables_space_set: AffineSet              # domain of intermediate variables
    variables_space_order: Optional[AffineMap]  # iteration order; None = default


@dataclass
class TileFuture:
    """Per-core handle produced by ``ktdp.inter_tile_produce``.

    Holds *this* core's contribution to the inter-tile op.  SPMD:
    each core has its own ``TileFuture`` instance bound to its local
    ``%fut`` SSA value; cross-core data movement happens via the
    scheduler's mailbox once the matching delivery op runs, not by
    reading other cores' futures.

    Def-use simulation
    ------------------
    The KTIR spec uses the SSA def-use edge ``%fut → consume(%fut)``
    to (1) order produce before consume on each core and (2) pair a
    delivery op with its matching produce.  This simulator does not
    walk the IR's def-use graph for either role.  Both are absorbed
    by ``TileFuture`` itself:
      - ordering: ordinary SSA scoping (``ctx.get_value`` only
        succeeds after produce has bound the name);
      - pairing: the ``TileFuture`` instance the consume handler
        receives *is* the edge — the producer-side data already
        lives on the future the consumer holds.

    Fields:
      ``partial_tensor_types``: declared T_p_1, ..., T_p_N (verbatim).
      ``local_partial``: this core's yielded tile(s) — the seed for
          the transport.  ``None`` when the core is in
          ``groups_set`` but outside ``producer_set`` (non-producer
          cores still run the delivery op so they participate in the
          workgroup-wide protocol; the backend substitutes
          ``identity`` for them).
      ``producer_set`` / ``groups_set``: parsed IR sets, kept on the
          future so the consumer handler can build a ``CommPlan``
          without re-parsing.
      ``group_idx``: the group this core belongs to, computed once
          at produce time.
    """
    partial_tensor_types: Tuple[str, ...]
    local_partial: Optional[Tuple['Tile', ...]]
    producer_set: AffineSet
    groups_set: AffineSet
    group_idx: int


@dataclass
class Operation:
    """Single IR operation with result, operands, attributes, and optional regions."""
    result: Optional[str]  # Result SSA value name (e.g., "%x")
    op_type: str  # Operation type (e.g., "ktdp.get_compute_tile_id", "arith.addf")
    operands: List[str]  # Operand SSA value names
    attributes: Dict[str, Any]  # Operation attributes
    result_type: Optional[str]  # Result type string
    regions: List[List['Operation']] = field(default_factory=list)  # For control flow
    outs_operands: List[str] = field(default_factory=list)  # SSA names from outs(...)

    def __repr__(self):
        if self.result:
            return f"{self.result} = {self.op_type}({', '.join(self.operands)})"
        else:
            return f"{self.op_type}({', '.join(self.operands)})"


def _iter_ops(operations: List['Operation']):
    """Yield all operations recursively, descending into regions."""
    for op in operations:
        yield op
        for region in op.regions:
            yield from _iter_ops(region)


@dataclass
class IRFunction:
    """IR function with arguments, body operations, and grid shape."""
    name: str
    arguments: List[Tuple[str, str]]  # [(name, type), ...]
    operations: List[Operation]
    grid: Tuple[int, int, int]  # Grid shape from function attributes
    use_counts: Dict[str, int] = field(default_factory=dict)  # SSA name -> use count for function body
    return_type: Optional[str] = None
    tensor_sizes: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def arg_names(self) -> List[str]:
        return [name.lstrip("%") for name, _ in self.arguments]

    def __post_init__(self):
        for op in _iter_ops(self.operations):
            if op.op_type == "ktdp.construct_memory_view" and op.operands:
                arg_name = op.operands[0].lstrip("%")
                if arg_name not in self.tensor_sizes:
                    self.tensor_sizes[arg_name] = {
                        "shape": tuple(op.attributes.get("shape", ())),
                        "dtype": op.attributes.get("dtype", "f16"),
                    }

    def __repr__(self):
        tensors = ", ".join(
            f"{name}: {info['shape']}x{info['dtype']}"
            for name, info in self.tensor_sizes.items()
        )
        return f"IRFunction({self.name}, grid={self.grid}, tensors=[{tensors}])"


@dataclass
class IRModule:
    """Top-level IR module containing functions."""
    functions: Dict[str, IRFunction] = field(default_factory=dict)
    # Named attribute aliases declared at module scope before module {}.
    # Maps "#name" -> verbatim value string (e.g. "#X_coord_set" -> "affine_set<...>").
    # Populated by the parser; used by construct handlers to resolve references.
    aliases: Dict[str, str] = field(default_factory=dict)

    def get_function(self, name: str) -> IRFunction:
        """Get function by name."""
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not found in module")
        return self.functions[name]

    def add_function(self, func: IRFunction):
        """Add function to module."""
        self.functions[func.name] = func
