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
Execution latency simulation for KTIR CPU backend.

Provides cycle-approximate latency estimation for Spyre hardware.
When a HardwareConfig is passed to KTIRInterpreter, each operation records
its estimated cycle cost. When disabled (default), zero overhead.

Cycle model: sequential within each core (total = compute + memory + comm).
Kernel latency = max across all cores (critical path).
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .ir_types import (
    AccessTile, DistributedTileRef, IndirectAccessTile, MemRef, Tile, TileRef,
)
from .dtypes import bytes_per_elem
from .memory import HBMSimulator


from .dialects.registry import get_latency_category


class LatencyCategory(StrEnum):
    """Categories used to classify op latency cost."""
    ZERO = "zero"
    MEMORY = "memory"
    COMPUTE_FLOAT = "compute_float"
    COMPUTE_TRANSCENDENTAL = "compute_transcendental"
    COMPUTE_INT = "compute_int"
    COMPUTE_MATMUL = "compute_matmul"
    COMM = "comm"


# ---------------------------------------------------------------------------
# Execution model — how the hardware is programmed
# ---------------------------------------------------------------------------

@dataclass
class ExecutionModel:
    """Modeling choices that can change independently of the silicon.

    Attributes:
        unit_categories: Which LatencyCategories each compute unit claims.
        pipeline: How compute, memory, and comm overlap in time.
            "serial" — no overlap: total = compute + memory + comm.
            "overlapped" — compute overlaps with data movement (memory + comm
                share the data-movement pipe and do NOT overlap each other):
                total = max(compute, memory + comm).
            "overlapped_3way" — all three stages have independent pipes and
                overlap freely: total = max(compute, memory, comm).
        bw_sharing: How HBM bandwidth is partitioned among cores. The model
            implements contended sharing; static partitioning is reserved as a
            future modeling choice and is not yet wired into the bandwidth path.
            "contended" — per-core BW = hbm_bw_chip / cores_active: the active
                cores share the whole chip's bandwidth (the implemented model).
            "static" — per-core BW = hbm_bw_chip / num_cores: a fixed per-core
                slice regardless of how many cores are active (reserved).
        fallback_unit: Dominant unit when no compute cycles ran.
    """
    unit_categories: Dict[str, set] = field(default_factory=lambda: {
        "systolic": {"compute_matmul"},
        "simd": {"compute_float", "compute_transcendental", "compute_int"},
    })
    pipeline: str = "serial"
    bw_sharing: str = "contended"
    fallback_unit: str = "simd"

    def __post_init__(self):
        # Reject unimplemented / unknown bw_sharing so it cannot silently fall
        # through to the contended path (see attribute docstring above).
        if self.bw_sharing == "static":
            raise NotImplementedError(
                "bw_sharing='static' (fixed per-core partition) is reserved "
                "and not yet implemented; only 'contended' is available."
            )
        if self.bw_sharing != "contended":
            raise ValueError(
                f"bw_sharing must be 'contended' (got {self.bw_sharing!r})"
            )


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Tunable hardware parameters for latency estimation.

    Defaults are chosen to be reasonable approximations. Parameters marked
    "estimated" have no authoritative source — users should override them
    if real Spyre specs are known.

    Attributes:
        num_cores: Number of cores (default 32).
        clock_ghz: Clock frequency in GHz (default 1.0, so 1 cycle = 1 ns).
        hbm_bandwidth_tb_s: Aggregate HBM bandwidth in TB/s (estimated).
        ring_bandwidth_tb_s: Ring bandwidth per direction in TB/s. The ring has two
            directions:  clock-wise and anti-clock-wise.
        simd_elements_per_cycle: SIMD throughput in f16 elements/cycle (estimated).
        systolic_rows: Number of rows in the systolic array.  The systolic
            peak is derived as ``2 * simd_elements_per_cycle * systolic_rows``
            FLOPs/cycle — each row performs one FMA (= 2 FLOP) on a SIMD-width
            vector per cycle, and rows execute in parallel.  A ``linalg.matmul``
            with dimensions M×N×K costs ``2·M·N·K / systolic_flops_per_cycle``
            cycles.
        transcendental_penalty: Multiplier for transcendental ops (aka. complex ops like 
            exp, log, sin, cos, tanh, rsqrt) vs simple elementwise ops (add, mul, max). 
            Default 4 means a transcendental on a SIMD-width vector costs 4× the cycles 
            of an elementwise op (estimated).
    """
    # TODO: add gather_bandwidth_tb_s if Spyre scatter/gather BW differs from
    # sequential HBM BW. Until confirmed by hardware team, both share hbm_bandwidth_tb_s.

    # The values used here are example, with some level of alignment to Spyre related
    # technical publications: ISCA'21. 
    num_cores: int = 32
    clock_ghz: float = 1.0
    hbm_bandwidth_tb_s: float = 0.128
    ring_bandwidth_tb_s: float = 0.064
    simd_elements_per_cycle: int = 64
    systolic_rows: int = 8
    transcendental_penalty: int = 4
    lx_size_mb: int = 2

    @property
    def clock_hz(self) -> float:
        """Clock frequency in Hz."""
        return self.clock_ghz * 1e9

    @property
    def hbm_bw_chip(self) -> float:
        """Aggregate HBM bandwidth in B/s."""
        return self.hbm_bandwidth_tb_s * 1e12

    @property
    def systolic_flops_per_cycle(self) -> int:
        """Systolic throughput in FLOPs/cycle = 2 * simd_elements * systolic_rows (FMA = 2 FLOP)."""
        return 2 * self.simd_elements_per_cycle * self.systolic_rows

    @property
    def systolic_peak(self) -> float:
        """Per-core systolic peak in FLOP/s."""
        return self.systolic_flops_per_cycle * self.clock_hz

    @property
    def simd_peak(self) -> float:
        """Per-core SIMD peak in FLOP/s."""
        return self.simd_elements_per_cycle * self.clock_hz

    def lx_bytes_per_cycle(self) -> float:
        """LX bandwidth in bytes/cycle (derived: ring_bandwidth_tb_s * 1)."""
        return self.ring_bandwidth_tb_s * 1e12 / (self.clock_ghz * 1e9)

    def lx_bytes_per_cycle_per_core(self) -> float:
        """LX bytes/cycle for one core (private scratchpad, same as lx_bytes_per_cycle)."""
        return self.lx_bytes_per_cycle()

    @property
    def ring_bytes_per_cycle(self) -> float:
        """Ring network bytes per cycle (one direction)."""
        return self.ring_bandwidth_tb_s * 1e12 / self.clock_hz


# ---------------------------------------------------------------------------
# Per-core latency counters
# ---------------------------------------------------------------------------

@dataclass
class _TraceEntry:
    """Single operation trace entry."""
    op_type: str
    cycles: float
    category: str


@dataclass
class CoreLatencyCounters:
    """Per-core cycle counters."""
    memory_cycles: float = 0.0
    comm_cycles: float = 0.0
    # Per-category flops, cycles, and bytes — keys are LatencyCategory string values.
    # Compute categories: "compute_matmul", "compute_float", "compute_transcendental", "compute_int".
    flops_by_category: Dict[str, float] = field(default_factory=dict)
    cycles_by_category: Dict[str, float] = field(default_factory=dict)
    # Bytes split by transport: the roofline uses DRAM-only traffic for arithmetic
    # intensity, while comm/ring bytes stay separately readable (dram_bytes / comm_bytes).
    bytes_by_category: Dict[str, int] = field(default_factory=dict)
    trace: Optional[List[_TraceEntry]] = None

    @property
    def total_cycles(self) -> float:
        return self.compute_cycles + self.memory_cycles + self.comm_cycles

    @property
    def compute_cycles(self) -> float:
        return sum(self.cycles_by_category.values())

    @property
    def total_flops(self) -> float:
        return sum(self.flops_by_category.values())

    @property
    def total_bytes(self) -> int:
        """All bytes moved by this core, across every transport (HBM + comm/ring)."""
        return sum(self.bytes_by_category.values())

    @property
    def comm_bytes(self) -> int:
        """Bytes this core moved over the cross-core comm transport (ring)."""
        return self.bytes_by_category.get("comm", 0)

    @property
    def dram_bytes(self) -> int:
        """Bytes crossing the HBM/DRAM boundary — the ``"memory"`` category.

        The traffic the roofline's HBM bandwidth ceiling governs, hence the correct
        denominator for arithmetic intensity. It sums only the ``"memory"`` category
        (HBM load/store bytes): a per-transport whitelist, so any other transport —
        comm/ring, or a future interconnect — contributes only if explicitly
        categorised ``"memory"``. On-chip LX ops record 0 bytes, so they never
        enter here.
        """
        return self.bytes_by_category.get("memory", 0)

    def record(self, category: str, cycles: float, op_type: str = "",
               flops: float = 0.0, nbytes: int = 0):
        if category.startswith("compute_"):
            self.cycles_by_category[category] = self.cycles_by_category.get(category, 0.0) + cycles
            self.flops_by_category[category] = self.flops_by_category.get(category, 0.0) + flops
        elif category == "memory":
            self.memory_cycles += cycles
        elif category == "comm":
            self.comm_cycles += cycles

        # Bucket bytes by transport so the roofline can isolate HBM/DRAM traffic
        # (dram_bytes) from comm/ring traffic (comm_bytes); total_bytes sums both.
        if nbytes:
            self.bytes_by_category[category] = self.bytes_by_category.get(category, 0) + nbytes

        if self.trace is not None:
            self.trace.append(_TraceEntry(op_type=op_type, cycles=cycles, category=category))


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Records per-operation cycle costs across all cores.

    Created by KTIRInterpreter when a HardwareConfig is provided.
    Counters are created lazily on first record_op for each core_id,
    so the tracker does not need to know the grid shape up front.
    """

    def __init__(self, config: HardwareConfig, trace: bool = False):
        self.config = config
        self._trace = trace
        self.counters: Dict[int, CoreLatencyCounters] = {}
        self._cores_active: int = config.num_cores

    def set_cores_active(self, cores_active: int):
        """Set the number of active cores (from grid shape) for BW sharing."""
        self._cores_active = cores_active

    def reset(self):
        """Clear all accumulated counters."""
        self.counters.clear()

    def record_op(self, core_id: int, op_type: str, result: Any, operands: List[Any]):
        """Estimate and record cycle cost for an operation.

        Args:
            core_id: Core that executed the operation.
            op_type: MLIR operation type string.
            result: The result value produced by the operation.
            operands: Resolved operand values.
        """
        if core_id not in self.counters:
            self.counters[core_id] = CoreLatencyCounters(
                trace=[] if self._trace else None
            )
        category, cycles, flops, nbytes = self._estimate(op_type, result, operands)
        self.counters[core_id].record(category, cycles, op_type, flops=flops, nbytes=nbytes)

    def report(self) -> "LatencyReport":
        """Build a LatencyReport from accumulated counters."""
        return LatencyReport(config=self.config, counters=dict(self.counters))

    # -- private helpers -----------------------------------------------------

    def _estimate(self, op_type: str, result: Any, operands: List[Any]) -> Tuple[str, float, float, int]:
        """Return (category, cycles, flops, nbytes) for a single operation."""

        LC = LatencyCategory
        category = get_latency_category(op_type)

        # Metadata-only ops (tensor.splat, scf.yield, …): no compute,
        # no memory traffic, no cycles.
        if category == LC.ZERO:
            return ("zero", 0.0, 0.0, 0)

        if category == LC.MEMORY:
            # LX (on-chip scratchpad) ops are free — the tile already
            # lives in LX as an SSA value, so no DMA occurs.
            if self._memory_space(operands) == "LX":
                return ("memory", 0.0, 0.0, 0)
            # Distributed view with mixed HBM + LX partitions:
            # decompose cost per memory space, take max (overlapped).
            if self._is_distributed_mixed(operands):
                hbm_bytes = self._data_size(result, operands)
                lx_bytes = self._lx_partition_bytes(operands)
                hbm_bw = self.config.hbm_bw_chip / self.config.clock_hz / self._cores_active
                lx_bw = self.config.lx_bytes_per_cycle_per_core()
                hbm_cycles = hbm_bytes / hbm_bw if hbm_bw > 0 else 0.0
                lx_cycles = lx_bytes / lx_bw if lx_bw > 0 else 0.0
                cycles = max(hbm_cycles, lx_cycles)
                return ("memory", cycles, 0.0, hbm_bytes + lx_bytes)
            # Uniform HBM load/store: cycles = bytes / per-core bandwidth.
            # BW per core = chip BW / cores_active (contended).
            nbytes = self._data_size(result, operands)
            bw = self.config.hbm_bw_chip / self.config.clock_hz / self._cores_active
            cycles = nbytes / bw if bw > 0 else 0.0
            return ("memory", cycles, 0.0, nbytes)

        if category == LC.COMPUTE_MATMUL:
            # Systolic matmul: 2*M*N*K FLOPs (one multiply + one add
            # per output element per K step).  No HBM traffic — operand
            # tiles are already in LX.
            m, n, k = self._matmul_dims(operands)
            flops = 2.0 * m * n * k
            cycles = flops / self.config.systolic_flops_per_cycle
            return (str(LC.COMPUTE_MATMUL), cycles, flops, 0)

        if category == LC.COMPUTE_TRANSCENDENTAL:
            # Transcendentals (exp, log, …): 1 FLOP per element, same
            # as elementwise, but the penalty multiplier models the
            # higher *latency* of the function unit — it does not
            # increase the FLOP count.
            n_elems = self._num_elements(result, operands)
            cycles = (n_elems / self.config.simd_elements_per_cycle) * self.config.transcendental_penalty
            return (str(LC.COMPUTE_TRANSCENDENTAL), cycles, float(n_elems), 0)

        if category == LC.COMPUTE_FLOAT:
            # Elementwise float (addf, mulf, …): 1 FLOP per element,
            # one SIMD-width per cycle.  No memory traffic.
            n_elems = self._num_elements(result, operands)
            cycles = n_elems / self.config.simd_elements_per_cycle
            return (str(LC.COMPUTE_FLOAT), cycles, float(n_elems), 0)

        if category == LC.COMPUTE_INT:
            # Integer ops (addi, muli, index casts, …): 1 FLOP per element.
            n_elems = self._num_elements(result, operands)
            if n_elems <= 1:
                # Scalar index arithmetic (e.g. address/offset computation) is
                # resolved at compile time and has no runtime cost.
                return (str(LC.COMPUTE_INT), 0.0, 0.0, 0)
            cycles = n_elems / self.config.simd_elements_per_cycle
            return (str(LC.COMPUTE_INT), cycles, float(n_elems), 0)

        if category == LC.COMM:
            # Ring/transport bytes for this core's contribution to the
            # comm op.  When the dialect handler stamps ``comm_bytes`` on
            # the result Tile (as ``ktdp.inter_tile_reduce`` does), use
            # that exact per-core total — it reflects what the transport
            # actually moved, including any per-tile sync subset.  The
            # operand-based fallback is for legacy/test paths only.
            nbytes = self._comm_size(result)
            bw = self.config.ring_bytes_per_cycle
            cycles = nbytes / bw if bw > 0 else 0.0
            return ("comm", cycles, 0.0, nbytes)

        # Unknown category
        raise NotImplementedError(f"Unknown category {category}")

    @staticmethod
    def _memory_space(operands: List[Any]) -> str:
        """Return the memory space of the memory op's TileRef target.

        The TileRef's memory_space determines the bandwidth bottleneck:
        - "HBM": data crosses the HBM <-> LX boundary (DMA).
        - "LX": data stays on-chip (local copy).

        Returns "HBM" when no TileRef is found (e.g. tt.load which always
        reads from HBM via pointer arithmetic).
        """
        for v in operands:
            if isinstance(v, MemRef):
                return v.memory_space
            if isinstance(v, TileRef):
                return v.memref.memory_space
            if isinstance(v, AccessTile):
                if isinstance(v.parent_ref, DistributedTileRef):
                    if any(p.memref.memory_space == "HBM"
                           for p in v.parent_ref.partitions):
                        return "HBM"
                    return v.parent_ref.partitions[0].memref.memory_space
                return v.parent_ref.memref.memory_space
            if isinstance(v, IndirectAccessTile):
                all_lx = (v.parent_ref.memory_space == "LX" and
                          all(iv.memory_space == "LX" for iv in v.index_views))
                return "LX" if all_lx else "HBM"
        return "HBM"

    @staticmethod
    def _is_distributed_mixed(operands: List[Any]) -> bool:
        """True if operands contain a DistributedTileRef with heterogeneous memory spaces."""
        for v in operands:
            if isinstance(v, AccessTile) and isinstance(v.parent_ref, DistributedTileRef):
                spaces = {p.memref.memory_space for p in v.parent_ref.partitions}
                return len(spaces) > 1
        return False

    @staticmethod
    def _lx_partition_bytes(operands: List[Any]) -> int:
        """Sum bytes from LX partitions in a distributed view."""
        for v in operands:
            if isinstance(v, AccessTile) and isinstance(v.parent_ref, DistributedTileRef):
                return sum(
                    p.size_bytes() for p in v.parent_ref.partitions
                    if p.memref.memory_space == "LX"
                )
        return 0

    @staticmethod
    def _data_size(result: Any, operands: List[Any]) -> int:
        """Estimate bytes transferred by a memory operation.

        HBM traffic is always charged at stick granularity:
        ``unique_sticks * HBMSimulator.STICK_BYTES``.

        Two carriers convey ``unique_sticks`` from the op handler:

        * **Loads** stamp ``unique_sticks`` (data) and
          ``index_unique_sticks`` (idx, when an IAT is involved) on the
          result :class:`Tile`. ``_data_size`` reads them off the result.
        * **Stores** have no result Tile — the dialect handler instead
          returns the int from ``MemoryOps.store`` /
          ``indirect_store`` / ``distributed_store`` as the op result.
          ``_data_size`` consumes it via ``isinstance(result, int)``.
          For an indirect store, the int already aggregates both the
          parent destination's sticks and the idx-side sticks.
        """
        # Store sideband: the handler propagated MemoryOps.{store,
        # indirect_store, distributed_store}'s int return as op result.
        if isinstance(result, int):
            return result * HBMSimulator.STICK_BYTES

        total = 0
        if isinstance(result, Tile):
            if result.unique_sticks is None:
                raise RuntimeError(
                    "Tile result on HBM path must populate unique_sticks; "
                    "got None. Load handlers must set unique_sticks for "
                    "stick-granular HBM accounting."
                )
            total += result.unique_sticks * HBMSimulator.STICK_BYTES
            if result.index_unique_sticks is not None:
                total += result.index_unique_sticks * HBMSimulator.STICK_BYTES
        for v in operands:
            if isinstance(v, IndirectAccessTile):
                if not isinstance(result, Tile):
                    raise RuntimeError(
                        "IAT operand without Tile result and without int "
                        f"sideband; got result={type(result).__name__}. "
                        "Store handlers must return MemoryOps.indirect_store's "
                        "int as the op result for stick-granular accounting."
                    )
                if result.index_unique_sticks is None:
                    raise RuntimeError(
                        "IAT operand with Tile result must populate "
                        "index_unique_sticks; got None. This indicates "
                        "the op handler skipped _resolve_idx_reads."
                    )
                continue
            elif isinstance(v, Tile):
                if result is not None:
                    raise ValueError(
                        f"_data_size: Tile in operands but result is also "
                        f"{type(result).__name__}; no ktdp op should produce both"
                    )
                raise RuntimeError(
                    "Tile operand with None result: store handler must "
                    "propagate MemoryOps.store's int return as op result "
                    "for stick-granular HBM accounting."
                )
        return total

    @staticmethod
    def _num_elements(result: Any, operands: List[Any]) -> int:
        """Count number of data elements processed."""
        if isinstance(result, Tile):
            return int(np.prod(result.shape))
        # For scalar results, check operands for tiles
        for v in operands:
            if isinstance(v, Tile):
                return int(np.prod(v.shape))
        return 1

    @staticmethod
    def _matmul_dims(operands: List[Any]) -> Tuple[int, int, int]:
        """Extract (M, N, K) from matmul operands."""
        tiles = [v for v in operands if isinstance(v, Tile)]
        if len(tiles) >= 2:
            a, b = tiles[0], tiles[1]
            # a is (M, K), b is (K, N)
            m = a.shape[0] if len(a.shape) >= 2 else 1
            k = a.shape[1] if len(a.shape) >= 2 else a.shape[0]
            n = b.shape[1] if len(b.shape) >= 2 else 1
            return (m, n, k)
        return (1, 1, 1)

    @staticmethod
    def _comm_size(result: Any) -> int:
        """Bytes transferred by a communication operation.

        Comm ops must stamp the per-core wire total onto the result
        ``Tile.comm_bytes`` from inside the handler.
        ``ktdp.inter_tile_reduce`` does this by reading
        ``RingReduceBackend.bytes_moved`` after ``yield from`` returns
        and assigning to ``final.comm_bytes``.  Future delivery ops
        (``inter_tile_consume``, ``inter_tile_reduce_scatter``) follow
        the same pattern.

        Raises if the carrier is missing — it's a contract violation,
        not a fallback case.
        """
        if not isinstance(result, Tile):
            raise RuntimeError(
                f"_comm_size: comm op result must be a Tile, got "
                f"{type(result).__name__}"
            )
        if result.comm_bytes is None:
            raise RuntimeError(
                "_comm_size: Tile result on comm path must populate "
                "comm_bytes; got None.  Comm-op handlers must stamp "
                "comm_bytes from the transport backend's send total."
            )
        return result.comm_bytes


# ---------------------------------------------------------------------------
# Latency report
# ---------------------------------------------------------------------------

@dataclass
class LatencyReport:
    """Summary of estimated execution latency."""
    config: HardwareConfig
    counters: Dict[int, CoreLatencyCounters]
    model: ExecutionModel = field(default_factory=ExecutionModel)

    def _effective_cycles(self, core: "CoreLatencyCounters") -> float:
        """Apply the pipeline overlap model to get a core's wall-clock cycles."""
        if self.model.pipeline == "overlapped":
            return max(core.compute_cycles, core.memory_cycles + core.comm_cycles)
        elif self.model.pipeline == "overlapped_3way":
            return max(core.compute_cycles, core.memory_cycles, core.comm_cycles)
        # serial (default)
        return core.compute_cycles + core.memory_cycles + core.comm_cycles

    @property
    def critical_core(self) -> "CoreLatencyCounters":
        """The core with the longest wall-clock time (the chip finishes when it finishes)."""
        return max(self.counters.values(), key=self._effective_cycles)

    @property
    def kernel_cycles(self) -> float:
        """Kernel latency = max effective cycles across all cores."""
        if not self.counters:
            return 0.0
        return self._effective_cycles(self.critical_core)

    @property
    def kernel_time_us(self) -> float:
        """Kernel time in microseconds."""
        return self.kernel_cycles / (self.config.clock_hz / 1e6)

    @property
    def bottleneck(self) -> str:
        """Identify the bottleneck category on the critical-path core."""
        if not self.counters:
            return "none"
        critical = self.critical_core
        cats = {
            "compute": critical.compute_cycles,
            "memory": critical.memory_cycles,
            "comm": critical.comm_cycles,
        }
        return max(cats, key=cats.get)

    def per_core_summary(self) -> List[Dict[str, Any]]:
        """Return per-core breakdown as list of dicts."""
        summaries = []
        for core_id in sorted(self.counters):
            c = self.counters[core_id]
            summaries.append({
                "core_id": core_id,
                "compute_cycles": c.compute_cycles,
                "memory_cycles": c.memory_cycles,
                "comm_cycles": c.comm_cycles,
                "total_cycles": self._effective_cycles(c),
            })
        return summaries

    # ------------------------------------------------------------------
    # Roofline — unified formulation
    # ------------------------------------------------------------------

    @staticmethod
    def _roofline_at(scope_flops: float, scope_bytes: float,
                     peak_compute: float, peak_bw: float,
                     elapsed_s: float) -> Dict[str, Any]:
        """Universal roofline from 5 inputs. Both chip and per-core call this
        with different parameterizations — the formulas are identical."""
        ai = scope_flops / scope_bytes if scope_bytes > 0 else float("inf")
        achieved = scope_flops / elapsed_s if elapsed_s > 0 else 0.0
        ceiling = min(peak_compute, peak_bw * ai) if ai != float("inf") else peak_compute
        return {
            "AI": ai,
            "compute_throughput": (achieved / peak_compute
                                   if peak_compute > 0 else 0.0),
            "dram_throughput": (scope_bytes / (peak_bw * elapsed_s)
                                if elapsed_s > 0 and peak_bw > 0 else 0.0),
            "achieved_gflops": achieved / 1e9,
            "ceiling_gflops": ceiling / 1e9,
            "attainment": achieved / ceiling if ceiling > 0 else 0.0,
            "ridge": peak_compute / peak_bw if peak_bw > 0 else float("inf"),
            "peak_gflops": peak_compute / 1e9,
        }

    def _roofline_common(self) -> Dict[str, Any]:
        """Shared raw data for both granularities. Dominant unit is decided
        once here (chip-wide cycles) so both methods always agree."""
        critical = self.critical_core
        elapsed_s = self.kernel_cycles / self.config.clock_hz
        unit_cats = self.model.unit_categories
        num_cores = self.config.num_cores

        # Cores with nonzero effective cycles (idle grid slots excluded).
        cores_active = sum(1 for c in self.counters.values()
                          if self._effective_cycles(c) > 0)

        # Chip-wide cycles per category → dominant unit decision.
        chip_cycles_by_cat: Dict[str, float] = {}
        for c in self.counters.values():
            for cat, cyc in c.cycles_by_category.items():
                chip_cycles_by_cat[cat] = chip_cycles_by_cat.get(cat, 0.0) + cyc

        totals = {unit: sum(chip_cycles_by_cat.get(cat, 0.0) for cat in cats)
                  for unit, cats in unit_cats.items()}
        dominant = (max(totals, key=totals.get) if any(totals.values())
                    else self.model.fallback_unit)

        return {
            "critical": critical,
            "elapsed_s": elapsed_s,
            "num_cores": num_cores,
            "cores_active": cores_active,
            "unit_peaks": {
                "systolic": self.config.systolic_peak,
                "simd": self.config.simd_peak,
            },
            "unit_cats": unit_cats,
            "dominant_unit": dominant,
            "hbm_bw_chip": self.config.hbm_bw_chip,
        }

    def chip_roofline(self) -> Dict[str, Any]:
        """Chip-level roofline — whole chip as one unit."""
        if not self.counters:
            return {}
        k = self._roofline_common()
        dom = k["dominant_unit"]
        dom_cats = k["unit_cats"][dom]
        num_cores = k["num_cores"]
        elapsed_s = k["elapsed_s"]

        chip_flops = sum(c.flops_by_category.get(cat, 0.0)
                         for c in self.counters.values() for cat in dom_cats)
        chip_dram_bytes = sum(c.dram_bytes for c in self.counters.values())
        sum_core_cycles = sum(self._effective_cycles(c)
                              for c in self.counters.values())
        elapsed_cycles = self.kernel_cycles

        chip_peak = k["unit_peaks"][dom] * num_cores
        hbm_bw = k["hbm_bw_chip"]

        rf = self._roofline_at(chip_flops, chip_dram_bytes,
                               chip_peak, hbm_bw, elapsed_s)
        rf.update({
            "mean_core_active_frac": (
                sum_core_cycles / (num_cores * elapsed_cycles)
                if num_cores > 0 and elapsed_cycles > 0 else 0.0),
            "grid_coverage": (k["cores_active"] / num_cores
                              if num_cores > 0 else 0.0),
            "dominant_unit": dom,
            "num_cores": num_cores,
            "cores_active": k["cores_active"],
        })
        return rf

    def core_roofline(self) -> Dict[str, Any]:
        """Per-core roofline — critical-path core, per-unit breakdown."""
        if not self.counters:
            return {}
        k = self._roofline_common()
        critical = k["critical"]
        elapsed_s = k["elapsed_s"]
        cores_active = k["cores_active"]
        dom = k["dominant_unit"]

        # Per-core available BW = chip BW / cores_active (contended).
        peak_bw = k["hbm_bw_chip"] / cores_active

        units: Dict[str, Any] = {}
        for unit_name, unit_peak in k["unit_peaks"].items():
            cats = k["unit_cats"][unit_name]
            unit_flops = sum(critical.flops_by_category.get(c, 0.0)
                             for c in cats)
            rf = self._roofline_at(unit_flops, critical.dram_bytes,
                                   unit_peak, peak_bw, elapsed_s)
            units[unit_name] = rf

        return {
            "core_AI": units[dom]["AI"],
            "core_attainment": units[dom]["attainment"],
            "core_dominant_unit": dom,
            "peak_bw_gb_s": peak_bw / 1e9,
            "ridge": units[dom]["ridge"],
            "units": units,
        }

    def roofline(self) -> Dict[str, Any]:
        """Combined roofline = chip-level + per-core, merged into one dict."""
        if not self.counters:
            return {}
        return {**self.core_roofline(), **self.chip_roofline()}

    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as a dictionary."""
        return {
            "kernel_cycles": self.kernel_cycles,
            "kernel_time_us": self.kernel_time_us,
            "bottleneck": self.bottleneck,
            "grid_cores": len(self.counters),
            "num_cores": self.config.num_cores,
            "per_core": self.per_core_summary(),
        }

    def __str__(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("KTIR Latency Estimation Report")
        lines.append("=" * 60)
        lines.append(f"  Kernel cycles : {self.kernel_cycles:,.1f}")
        lines.append(f"  Kernel time   : {self.kernel_time_us:.3f} us")
        lines.append(f"  Bottleneck    : {self.bottleneck}")
        lines.append(f"  Cores         : {len(self.counters)}")
        lines.append("-" * 60)
        lines.append(f"  {'Core':>4}  {'Compute':>12}  {'Memory':>12}  {'Comm':>12}  {'Total':>12}")
        lines.append("-" * 60)
        for core_id in sorted(self.counters):
            c = self.counters[core_id]
            lines.append(
                f"  {core_id:>4}  {c.compute_cycles:>12.1f}  "
                f"{c.memory_cycles:>12.1f}  {c.comm_cycles:>12.1f}  "
                f"{self._effective_cycles(c):>12.1f}"
            )
        lines.append("=" * 60)

        critical = self.critical_core
        if critical.total_flops > 0 or critical.total_bytes > 0:
            chip = self.chip_roofline()
            core = self.core_roofline()

            def _fmt_ai(ai: float) -> str:
                if ai != float('inf'):
                    return f"{ai:.2f} FLOP/B"
                if critical.total_bytes > 0:
                    return "inf (no HBM traffic)"
                return "inf (no memory traffic)"

            # --- CHIP-LEVEL block ---
            lines.append("")
            lines.append("Roofline: CHIP-LEVEL  (whole chip as one unit)")
            lines.append("-" * 60)
            lines.append(f"  AI                    : {_fmt_ai(chip['AI'])}")
            lines.append(f"  compute_throughput    : {chip['compute_throughput']:.1%}")
            lines.append(f"  dram_throughput       : {chip['dram_throughput']:.1%}")
            lines.append(f"  attainment            : {chip['attainment']:.1%}")
            lines.append(f"  mean_core_active_frac : {chip['mean_core_active_frac']:.1%}")
            lines.append(
                f"  grid_coverage         : {chip['cores_active']}/{chip['num_cores']}  "
                f"({chip['grid_coverage']:.1%})"
            )
            lines.append(f"  dominant_unit         : {chip['dominant_unit']}")

            # --- PER-CORE block ---
            dom = core["core_dominant_unit"]
            lines.append("")
            lines.append("Roofline: PER-CORE  (critical-path core)")
            lines.append("-" * 60)
            lines.append(f"  core_AI               : {_fmt_ai(core['core_AI'])}")
            lines.append(f"  core_attainment       : {core['core_attainment']:.1%}")
            lines.append(f"  core_dominant_unit    : {dom}")
            lines.append(f"  peak_bw (per-core)    : {core['peak_bw_gb_s']:.2f} GB/s")
            lines.append("")
            lines.append(
                f"  {'Unit':>10}  {'Achieved':>12}  {'Ceiling':>12}  "
                f"{'Ridge':>10}  {'Attain':>7}"
            )
            lines.append(
                f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*7}"
            )
            for unit_name, u in core["units"].items():
                marker = " *" if unit_name == dom else "  "
                lines.append(
                    f"{marker} {unit_name:>10}  "
                    f"{u['achieved_gflops']:>10.2f} G  "
                    f"{u['ceiling_gflops']:>10.2f} G  "
                    f"{u['ridge']:>8.1f} F/B  "
                    f"{u['attainment']:>6.1%}"
                )
            lines.append("=" * 60)

        return "\n".join(lines)
