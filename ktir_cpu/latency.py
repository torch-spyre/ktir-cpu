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

from .ir_types import AccessTile, IndirectAccessTile, MemRef, Tile, TileRef
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
        ring_bandwidth_tb_s: Ring bandwidth per direction in TB/s.
        simd_elements_per_cycle: SIMD throughput in f16 elements/cycle (estimated).
        systolic_flops_per_cycle: Peak throughput of the systolic array in
            FLOPs per cycle (estimated).  A systolic array is a grid of
            processing elements (PEs) that perform multiply-accumulate in
            lock-step.  For an N×N array, each PE does 1 fused multiply-add
            (= 2 FLOPs) per cycle, giving 2×N×N FLOPs/cycle per outer-product
            step.  The default assumes a 64×64 array executing one K-step per
            cycle: ``2 × 64 × 64 = 8192`` FLOPs/cycle per step, times 64
            K-steps pipelined = ``2 × 64 × 64 × 64 = 524288`` FLOPs/cycle
            effective throughput.  A ``linalg.matmul`` with dimensions M×N×K
            costs ``2·M·N·K / systolic_flops_per_cycle`` cycles.
        transcendental_penalty: Multiplier for transcendental ops vs elementwise (estimated).
    """
    # TODO: add gather_bandwidth_tb_s if Spyre scatter/gather BW differs from
    # sequential HBM BW. Until confirmed by hardware team, both share hbm_bandwidth_tb_s.
    num_cores: int = 32
    clock_ghz: float = 1.0
    hbm_bandwidth_tb_s: float = 1.0
    ring_bandwidth_tb_s: float = 4.0
    simd_elements_per_cycle: int = 64
    systolic_flops_per_cycle: int = 2 * 64 * 64 * 64
    transcendental_penalty: int = 4

    @property
    def hbm_bytes_per_cycle_per_core(self) -> float:
        """HBM bytes per cycle available to each core."""
        bytes_per_cycle_total = self.hbm_bandwidth_tb_s * 1e12 / (self.clock_ghz * 1e9)
        return bytes_per_cycle_total / self.num_cores

    @property
    def ring_bytes_per_cycle(self) -> float:
        """Ring network bytes per cycle (one direction)."""
        return self.ring_bandwidth_tb_s * 1e12 / (self.clock_ghz * 1e9)


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
            # HBM load/store: cycles = bytes / per-core bandwidth.
            # Pure data movement — no FLOPs, only bytes transferred.
            nbytes = self._data_size(result, operands)
            bw = self.config.hbm_bytes_per_cycle_per_core
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
                return v.parent_ref.memref.memory_space
            if isinstance(v, IndirectAccessTile):
                all_lx = (v.parent_ref.memory_space == "LX" and
                          all(iv.memory_space == "LX" for iv in v.index_views))
                return "LX" if all_lx else "HBM"
        return "HBM"

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

    @property
    def kernel_cycles(self) -> float:
        """Kernel latency = max total cycles across all cores."""
        if not self.counters:
            return 0.0
        return max(c.total_cycles for c in self.counters.values())

    @property
    def kernel_time_us(self) -> float:
        """Kernel time in microseconds (cycles / clock_ghz / 1e3)."""
        return self.kernel_cycles / (self.config.clock_ghz * 1e3)

    @property
    def bottleneck(self) -> str:
        """Identify the bottleneck category on the critical-path core."""
        if not self.counters:
            return "none"
        critical = max(self.counters.values(), key=lambda c: c.total_cycles)
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
                "total_cycles": c.total_cycles,
            })
        return summaries

    def roofline(self) -> Dict[str, Any]:
        """Compute per-unit roofline metrics for the critical-path core.

        Two compute units are modelled (systolic for matmul, SIMD for everything
        else).  The dominant unit — whichever consumed the most compute cycles —
        sets the headline.  The chart shows one roof::

            GFLOP/s
              ^
              |         peak (dominant unit)
              |        .----------------------------  compute ceiling
              |       /
              |      /    * kernel dot
              |     /
              |    /
              |   /  BW ceiling = peak_bw × AI
              |  /
              | /
              +-----------------------------------> AI (FLOP/B)
                       ^
                  ridge point



        - **systolic**: ``linalg.matmul`` ops, peak = ``systolic_flops_per_cycle × clock``
        - **simd**: all other compute ops (float, transcendental, int),
          peak = ``simd_elements_per_cycle × clock``

        The **dominant unit** is whichever consumed the most compute cycles in
        the kernel trace — the real bottleneck.  FLOPs are not comparable across
        units (matmul ``2*M*N*K`` vs SIMD per-element), so cycles, not FLOPs,
        identify the busy unit.  The summary ``arithmetic_intensity`` and
        ``efficiency`` are reported for the dominant unit.

        For each unit, ``achieved_gflops`` is ``unit_flops / total_wall_time``
        (not unit_flops / unit_cycles) so the achieved rate reflects true
        end-to-end throughput including memory stalls.

        .. note:: The roofline covers compute and HBM bandwidth only.
           Communication cycles (ring allgather/reduce) are not modelled.

        Returns a dict with:
            arithmetic_intensity: the dominant unit's per-unit AI
                (``dominant_unit FLOPs / total bytes``, FLOP/B).
            peak_bw_gb_s: Per-core HBM bandwidth in GB/s.
            dominant_unit: ``"systolic"`` or ``"simd"`` (most compute cycles).
            efficiency: achieved / ceiling for the dominant unit (0..1).
            cores_active: number of cores that consumed any cycle (nonzero
                total_cycles). A core left idle by an oversized grid produces a
                counter entry but spends zero cycles, so it is excluded and an
                under-filled grid does not inflate coverage. A core busy only
                on communication still counts as active — dispatched is not the
                same as doing useful compute.
            num_cores: chip-wide hardware core count from ``HardwareConfig``.
            grid_coverage: ``cores_active / num_cores`` — fraction of chip
                cores dispatched any work. Spatial dispatch coverage, not how
                busy each core is (a core running a single cycle counts fully),
                so it is NOT Nsight's time-based "SM Active %"; pair with
                chip_throughput / efficiency to read actual utilization.
            units: per-unit dict, each with:
                achieved_gflops, ceiling_gflops, ridge_point, efficiency,
                arithmetic_intensity (this unit's own FLOPs / total bytes),
                peak_gflops, chip_peak_gflops, chip_throughput.

        Per-unit chip-level fields (peak-based, Nsight SOL analogue):
            peak_gflops: per-core flat hardware peak (independent of AI).
            chip_peak_gflops: ``peak_gflops × num_cores`` — chip-wide flat
                peak. Distinct from ``ceiling_gflops`` which is the
                roofline ceiling at this kernel's AI.
            chip_throughput: ``sum(core FLOPs over all cores) / elapsed
                / (peak × num_cores)`` — Nsight "Compute (SM) Throughput %"
                analogue. The numerator is the actual total FLOPs summed
                across every core over the elapsed (wall) time, so idle and
                lighter cores correctly pull the figure down — the same way
                Nsight aggregates per-SM counters across all SMs over elapsed
                cycles. This is exact under any work distribution, including
                split-K and uneven tiling, where extrapolating the critical
                core's rate to all cores would overstate utilization.

                Distinct from ``efficiency`` (per-active-core, ceiling-based)
                and from ``grid_coverage`` (dispatched-core fraction):
                chip_throughput is peak-based and chip-wide. The three
                coincide only when work is evenly distributed and the kernel
                is compute-bound at its flat peak.
        """
        if not self.counters:
            return {}
        critical = max(self.counters.values(), key=lambda c: c.total_cycles)

        clock = self.config.clock_ghz * 1e9
        elapsed_s = critical.total_cycles / clock
        peak_bw = self.config.hbm_bytes_per_cycle_per_core * clock

        # Count cores that consumed any cycle, not every grid core that produced
        # a counter entry: an oversized grid leaves some cores with zero loop
        # iterations (0 cycles), and those must not inflate utilization. Use
        # total_cycles (compute+memory+comm) so memory-only kernels with 0 FLOPs
        # (e.g. embedding gather) still count their active cores. A comm-only
        # core also counts as active here (dispatched, not necessarily computing).
        cores_active = sum(1 for c in self.counters.values() if c.total_cycles > 0)
        num_cores = self.config.num_cores
        grid_coverage = cores_active / num_cores if num_cores > 0 else 0.0

        # Per-unit hardware ceilings (hardware constants, not kernel-derived).
        unit_ceilings = {
            "systolic": self.config.systolic_flops_per_cycle * clock,
            "simd": self.config.simd_elements_per_cycle * clock,
        }

        # Which LatencyCategory strings belong to each unit.
        _LC = LatencyCategory
        unit_categories = {
            "systolic": {str(_LC.COMPUTE_MATMUL)},
            "simd": {str(_LC.COMPUTE_FLOAT), str(_LC.COMPUTE_TRANSCENDENTAL), str(_LC.COMPUTE_INT)},
        }

        units: Dict[str, Any] = {}
        for unit_name, peak in unit_ceilings.items():
            cats = unit_categories[unit_name]
            flops = sum(critical.flops_by_category.get(c, 0.0) for c in cats)
            achieved = flops / elapsed_s if elapsed_s > 0 else 0.0
            # Per-unit arithmetic intensity: this unit's own FLOPs over the kernel's
            # DRAM bytes (NCU convention — numerator split per pipeline, denominator
            # the shared byte traffic). The denominator is dram_bytes: the HBM
            # bandwidth ceiling governs HBM traffic only, so comm/ring bytes (a
            # different interconnect) are excluded — otherwise a comm kernel's
            # byte-rate could exceed HBM peak and put the point above the roof. Each
            # unit gets its own ceiling, so the other unit's FLOPs don't inflate it.
            unit_ai = (flops / critical.dram_bytes
                       if critical.dram_bytes > 0 else float('inf'))
            # Roofline ceiling at this unit's own AI.
            ceiling = min(peak, peak_bw * unit_ai)
            # Cores are homogeneous: every core shares the same clock and
            # compute rates from HardwareConfig, so the chip peak is
            # peak * num_cores, which equals summing identical per-core peaks.
            # Heterogeneity lives on other axes (per functional unit, captured
            # separately in unit_ceilings; per precision/generation, captured by
            # the config values), never core-to-core.
            chip_peak = peak * num_cores
            # Chip throughput uses the actual FLOPs summed across every core,
            # not the critical core's rate extrapolated to all cores. Under
            # uneven tiling the lighter cores do fewer FLOPs in the same wall
            # time; extrapolating `achieved * cores_active` would count them as
            # if they matched the critical core and overstate utilization. The
            # real per-core sum over the same elapsed time gives the true
            # chip-wide figure, matching Nsight's SM Throughput (per-SM counters
            # aggregated across all SMs over elapsed cycles).
            chip_flops = sum(c.flops_by_category.get(cat, 0.0)
                             for c in self.counters.values() for cat in cats)
            chip_achieved = chip_flops / elapsed_s if elapsed_s > 0 else 0.0
            chip_throughput = chip_achieved / chip_peak if chip_peak > 0 else 0.0
            units[unit_name] = {
                "achieved_gflops": achieved / 1e9,
                "ceiling_gflops": ceiling / 1e9,
                "ridge_point": peak / peak_bw,
                "efficiency": achieved / ceiling if ceiling > 0 else 0.0,
                "arithmetic_intensity": unit_ai,
                "peak_gflops": peak / 1e9,
                "chip_peak_gflops": chip_peak / 1e9,
                "chip_throughput": chip_throughput,
            }

        # Dominant unit = the unit that consumed the most compute cycles — the
        # real bottleneck. Per-category cycles are already attributed to each
        # unit upstream (LatencyTracker._estimate); read that conclusion rather
        # than re-deriving from FLOPs, which are not comparable across units
        # (matmul 2*M*N*K vs SIMD per-element). Fall back to "simd" when no
        # compute ran at all (every category zero → both flops and cycles zero).
        dominant = max(
            unit_ceilings,
            key=lambda u: sum(critical.cycles_by_category.get(c, 0.0)
                              for c in unit_categories[u]),
        )
        if all(units[u]["achieved_gflops"] == 0.0 for u in units):
            dominant = "simd"

        return {
            "arithmetic_intensity": units[dominant]["arithmetic_intensity"],
            "peak_bw_gb_s": peak_bw / 1e9,
            "dominant_unit": dominant,
            "efficiency": units[dominant]["efficiency"],
            "cores_active": cores_active,
            "num_cores": num_cores,
            "grid_coverage": grid_coverage,
            "units": units,
        }

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
        lines.append(f"  Kernel cycles : {self.kernel_cycles:,.0f}")
        lines.append(f"  Kernel time   : {self.kernel_time_us:.3f} us")
        lines.append(f"  Bottleneck    : {self.bottleneck}")
        lines.append(f"  Cores         : {len(self.counters)}")
        lines.append("-" * 60)
        lines.append(f"  {'Core':>4}  {'Compute':>12}  {'Memory':>12}  {'Comm':>12}  {'Total':>12}")
        lines.append("-" * 60)
        for core_id in sorted(self.counters):
            c = self.counters[core_id]
            lines.append(
                f"  {core_id:>4}  {c.compute_cycles:>12.0f}  "
                f"{c.memory_cycles:>12.0f}  {c.comm_cycles:>12.0f}  "
                f"{c.total_cycles:>12.0f}"
            )
        lines.append("=" * 60)

        # Roofline section — only if there are flops or bytes
        critical = max(self.counters.values(), key=lambda c: c.total_cycles)
        if critical.total_flops > 0 or critical.total_bytes > 0:
            rf = self.roofline()
            lines.append("")
            lines.append("Roofline Analysis (critical-path core)")
            lines.append("-" * 60)
            ai = rf["arithmetic_intensity"]
            # AI == inf means dram_bytes == 0 (no HBM). Split by total_bytes, not by
            # naming a transport, so a comm-only kernel (or a future non-HBM
            # interconnect) reads "no HBM traffic" instead of the misleading "no
            # memory traffic" — it did move bytes, just not over HBM.
            if ai != float('inf'):
                ai_str = f"{ai:.2f} FLOP/B"
            elif critical.total_bytes > 0:
                ai_str = "inf (no HBM traffic)"
            else:
                ai_str = "inf (no memory traffic)"
            dom_unit = rf["units"][rf["dominant_unit"]]
            lines.append(f"  Arithmetic intensity : {ai_str}")
            lines.append(f"  Peak bandwidth       : {rf['peak_bw_gb_s']:.2f} GB/s")
            lines.append(f"  Dominant unit        : {rf['dominant_unit']}")
            lines.append(
                f"  Grid coverage        : {rf['cores_active']}/{rf['num_cores']}  "
                f"(grid_coverage {rf['grid_coverage']:.1%})"
            )
            lines.append(
                f"  Efficiency           : {rf['efficiency']:.1%}  "
                f"(per-active core, achieved/ceiling)"
            )
            lines.append(
                f"  Chip throughput      : {dom_unit['chip_throughput']:.1%}  "
                f"(chip-wide, achieved/peak with idle cores)"
            )
            lines.append("")
            lines.append(
                f"  {'Unit':>10}  {'Achieved':>12}  {'Ceiling':>12}  "
                f"{'Ridge':>10}  {'Eff':>7}  {'ChipThru':>9}"
            )
            lines.append(
                f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*7}  {'-'*9}"
            )
            for unit_name, u in rf["units"].items():
                marker = " *" if unit_name == rf["dominant_unit"] else "  "
                lines.append(
                    f"{marker} {unit_name:>10}  "
                    f"{u['achieved_gflops']:>10.2f} G  "
                    f"{u['ceiling_gflops']:>10.2f} G  "
                    f"{u['ridge_point']:>8.1f} F/B  "
                    f"{u['efficiency']:>6.1%}  "
                    f"{u['chip_throughput']:>8.2%}"
                )
            lines.append("=" * 60)

        return "\n".join(lines)
