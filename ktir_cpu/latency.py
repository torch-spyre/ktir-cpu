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

    # ------------------------------------------------------------------
    # Roofline — shared base, then chip / per-core fork
    # ------------------------------------------------------------------
    # Two compute units are modelled: systolic (``linalg.matmul``) and SIMD
    # (float / transcendental / int). Their FLOPs are not comparable (matmul
    # ``2*M*N*K`` vs SIMD per-element) and their peaks differ by orders of
    # magnitude, so every roofline reports the *dominant* unit — the one that
    # consumed the most cycles (the bottleneck). Cycles, not FLOPs, pick it.

    def _roofline_common(self) -> Dict[str, Any]:
        """Granularity-agnostic quantities shared by :meth:`chip_roofline` and
        :meth:`core_roofline`, gathered in one place so neither method
        re-derives them (each call builds this dict once; :meth:`roofline`,
        which calls both, builds it twice — negligible for report sizes).

        Nothing here mixes granularity: hardware peak rates, the wall-clock
        elapsed (= the critical core's total cycles, i.e. when the whole chip
        finishes), and core counts.

        The **dominant unit is decided here, once**, from cycles summed over
        ALL cores, and both :meth:`chip_roofline` and :meth:`core_roofline`
        consume that single value — they never re-derive it, so the two can
        never disagree. Dominant is a workload property (every core runs the
        same op mix in data-parallel execution), not a per-core one, so the
        chip-wide aggregate is the right — and only — place to decide it.

        ``peak_bw_core`` (per-core HBM) feeds the per-core roofline only;
        ``hbm_bw_chip`` (aggregate HBM, read directly) feeds the chip roofline
        only — the two never cross, so chip is never fed a ``÷ num_cores``
        bandwidth.
        """
        critical = max(self.counters.values(), key=lambda c: c.total_cycles)
        clock = self.config.clock_ghz * 1e9
        _LC = LatencyCategory
        unit_categories = {
            "systolic": {str(_LC.COMPUTE_MATMUL)},
            "simd": {str(_LC.COMPUTE_FLOAT),
                     str(_LC.COMPUTE_TRANSCENDENTAL),
                     str(_LC.COMPUTE_INT)},
        }
        # Chip-wide cycles per category (Σ over all cores) → the single
        # dominant-unit decision, shared by both granularities.
        chip_cycles_by_cat: Dict[str, float] = {}
        for c in self.counters.values():
            for cat, cyc in c.cycles_by_category.items():
                chip_cycles_by_cat[cat] = chip_cycles_by_cat.get(cat, 0.0) + cyc
        return {
            "critical": critical,
            "clock": clock,
            "elapsed_cycles": critical.total_cycles,          # wall clock
            "elapsed_s": critical.total_cycles / clock if clock > 0 else 0.0,
            "num_cores": self.config.num_cores,
            # Cores that consumed any cycle — an oversized grid leaves some
            # cores at 0 cycles; those must not inflate coverage. total_cycles
            # (compute+memory+comm) so memory-only cores still count.
            "cores_active": sum(1 for c in self.counters.values()
                                if c.total_cycles > 0),
            # Per-core hardware peak FLOP-rate per compute unit (hardware consts).
            "unit_ceilings": {
                "systolic": self.config.systolic_flops_per_cycle * clock,
                "simd": self.config.simd_elements_per_cycle * clock,
            },
            "unit_categories": unit_categories,
            # Single dominant unit (bottleneck), chip-wide, shared by both forks.
            "dominant_unit": self._dominant_unit(chip_cycles_by_cat,
                                                 unit_categories),
            # Per-core HBM bandwidth (bytes/s) — CORE roofline only.
            "peak_bw_core": self.config.hbm_bytes_per_cycle_per_core * clock,
            # Aggregate chip HBM bandwidth (bytes/s), read directly — CHIP only.
            # NOT the per-core hbm_bytes_per_cycle_per_core (= this / num_cores).
            "hbm_bw_chip": self.config.hbm_bandwidth_tb_s * 1e12,
        }

    @staticmethod
    def _dominant_unit(cycles_by_category: Dict[str, float],
                       unit_categories: Dict[str, Any]) -> str:
        """Bottleneck unit = the one with the most cycles. Called once, from
        :meth:`_roofline_common`, on cycles summed over all cores (dominant is
        a chip-wide workload property). FLOPs are not comparable across units,
        so cycles — not FLOPs — pick the bottleneck. Falls back to ``"simd"``
        when no compute ran (every category zero).
        """
        totals = {
            unit: sum(cycles_by_category.get(cat, 0.0) for cat in cats)
            for unit, cats in unit_categories.items()
        }
        if not any(totals.values()):
            return "simd"
        return max(totals, key=totals.get)

    def chip_roofline(self) -> Dict[str, Any]:
        """Chip (device) level roofline — the whole chip as ONE unit (NCU
        device view).

        Aggregates the existing per-core counters over ALL cores; no new
        inputs, and no per-core (critical-core) quantity except the wall-clock
        elapsed. FLOPs are the dominant unit's, summed over all cores; the
        dominant unit is chosen chip-wide (Σ cycles over all cores), never from
        the critical core. The HBM roof is the aggregate bandwidth read
        directly, never per-core ``÷ num_cores``.

        Returns:
            AI: ``Σ FLOP / Σ HBM bytes`` over all cores — roofline x-axis (FLOP/B).
            compute_throughput: ``Σ FLOP / elapsed / chip_peak`` — compute SOL
                (Nsight "Compute (SM) Throughput %"); ``chip_peak`` = dominant
                unit's per-core peak × ``num_cores``.
            dram_throughput: ``Σ HBM bytes / (hbm_bw × elapsed)`` — memory SOL
                (Nsight "DRAM Throughput %"); ``hbm_bw`` the aggregate bandwidth.
            mean_core_active_frac: ``Σ core.total_cycles / (num_cores ×
                elapsed_cycles)`` — time-based core occupancy over all cores
                (Nsight "SM Active %" analogue; active = compute+memory+comm).
            grid_coverage: ``cores_active / num_cores`` — spatial dispatch coverage.
            ridge: ``chip_peak / hbm_bw`` (== per-core ridge — same dominant
                unit for both, num_cores cancels).
            dominant_unit, num_cores, cores_active.
        """
        if not self.counters:
            return {}
        k = self._roofline_common()
        # Single dominant unit, decided chip-wide in _roofline_common.
        dom = k["dominant_unit"]
        dom_cats = k["unit_categories"][dom]

        chip_flops = sum(c.flops_by_category.get(cat, 0.0)
                         for c in self.counters.values() for cat in dom_cats)
        chip_dram_bytes = sum(c.dram_bytes for c in self.counters.values())
        chip_cycles = sum(c.total_cycles for c in self.counters.values())

        elapsed_s = k["elapsed_s"]
        elapsed_cycles = k["elapsed_cycles"]
        num_cores = k["num_cores"]
        chip_peak = k["unit_ceilings"][dom] * num_cores
        hbm_bw = k["hbm_bw_chip"]

        return {
            "AI": (chip_flops / chip_dram_bytes
                   if chip_dram_bytes > 0 else float("inf")),
            "compute_throughput": (chip_flops / elapsed_s / chip_peak
                                   if elapsed_s > 0 and chip_peak > 0 else 0.0),
            "dram_throughput": (chip_dram_bytes / (hbm_bw * elapsed_s)
                                if elapsed_s > 0 and hbm_bw > 0 else 0.0),
            "mean_core_active_frac": (
                chip_cycles / (num_cores * elapsed_cycles)
                if num_cores > 0 and elapsed_cycles > 0 else 0.0),
            "grid_coverage": (k["cores_active"] / num_cores
                              if num_cores > 0 else 0.0),
            "ridge": chip_peak / hbm_bw if hbm_bw > 0 else float("inf"),
            "dominant_unit": dom,
            "num_cores": num_cores,
            "cores_active": k["cores_active"],
        }

    def core_roofline(self) -> Dict[str, Any]:
        """Per-core roofline — the critical-path core only.

        For each compute unit, ``achieved_gflops`` is ``unit_flops /
        total_wall_time`` (not unit_flops / unit_cycles) so it reflects true
        end-to-end throughput including memory stalls. The dominant unit sets
        the ``core_*`` headline. HBM bandwidth here is the per-core figure.

        .. note:: Covers compute and HBM bandwidth only. Communication cycles
           (ring allgather/reduce) are not modelled.

        Returns:
            core_AI: dominant unit's FLOPs / HBM ``dram_bytes`` on the critical
                core (FLOP/B; comm/ring bytes excluded).
            core_efficiency: achieved / ceiling for the dominant unit (0..1).
            core_dominant_unit: ``"systolic"`` or ``"simd"`` (most cycles).
            peak_bw_gb_s: per-core HBM bandwidth (GB/s).
            ridge: dominant unit's ridge point (peak / peak_bw_core; == chip
                ridge — same dominant unit, num_cores cancels).
            units: per-unit dict, each with achieved_gflops, ceiling_gflops,
                ridge_point, efficiency, arithmetic_intensity, peak_gflops,
                chip_peak_gflops, compute_throughput.
        """
        if not self.counters:
            return {}
        k = self._roofline_common()
        critical = k["critical"]
        elapsed_s = k["elapsed_s"]
        peak_bw = k["peak_bw_core"]
        num_cores = k["num_cores"]
        cats_of = k["unit_categories"]

        units: Dict[str, Any] = {}
        for unit_name, peak in k["unit_ceilings"].items():
            cats = cats_of[unit_name]
            flops = sum(critical.flops_by_category.get(c, 0.0) for c in cats)
            achieved = flops / elapsed_s if elapsed_s > 0 else 0.0
            # Per-unit AI: this unit's FLOPs over HBM dram_bytes (comm/ring
            # excluded — the HBM ceiling governs HBM traffic only).
            unit_ai = (flops / critical.dram_bytes
                       if critical.dram_bytes > 0 else float("inf"))
            ceiling = min(peak, peak_bw * unit_ai)
            # Per-unit chip-wide throughput (shipped #120 metric, renamed from
            # chip_throughput): real per-core FLOP sum over elapsed, vs the
            # unit's chip peak. Lighter/idle cores correctly pull it down.
            chip_peak = peak * num_cores
            chip_flops = sum(c.flops_by_category.get(cat, 0.0)
                             for c in self.counters.values() for cat in cats)
            compute_throughput = (chip_flops / elapsed_s / chip_peak
                                  if elapsed_s > 0 and chip_peak > 0 else 0.0)
            units[unit_name] = {
                "achieved_gflops": achieved / 1e9,
                "ceiling_gflops": ceiling / 1e9,
                "ridge_point": peak / peak_bw if peak_bw > 0 else float("inf"),
                "efficiency": achieved / ceiling if ceiling > 0 else 0.0,
                "arithmetic_intensity": unit_ai,
                "peak_gflops": peak / 1e9,
                "chip_peak_gflops": chip_peak / 1e9,
                "compute_throughput": compute_throughput,
            }

        # Same single dominant unit as chip_roofline (decided chip-wide in
        # _roofline_common) so the two granularities never disagree.
        dom = k["dominant_unit"]

        return {
            "core_AI": units[dom]["arithmetic_intensity"],
            "core_efficiency": units[dom]["efficiency"],
            "core_dominant_unit": dom,
            "peak_bw_gb_s": peak_bw / 1e9,
            "ridge": units[dom]["ridge_point"],
            "units": units,
        }

    def roofline(self) -> Dict[str, Any]:
        """Combined roofline = chip-level + per-core, merged into one flat dict.

        Prefer the granularity-specific methods for clarity:
        :meth:`chip_roofline` (whole chip as one unit) and :meth:`core_roofline`
        (critical-path core). This merges both for a single-call combined view.
        The two key sets are disjoint — chip uses plain NCU names
        (``AI``, ``compute_throughput``, …), per-core uses ``core_*`` / ``units``
        / ``peak_bw_gb_s`` — except ``ridge``, which is identical at both
        granularities: the dominant unit is decided once (chip-wide) and shared,
        and ``num_cores`` cancels within a unit.
        """
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

        # Roofline section — two clearly-separated blocks (chip, then per-core),
        # each sourced from its own method. Only if there are flops or bytes.
        critical = max(self.counters.values(), key=lambda c: c.total_cycles)
        if critical.total_flops > 0 or critical.total_bytes > 0:
            chip = self.chip_roofline()
            core = self.core_roofline()

            # AI == inf means no HBM bytes moved. Split by total_bytes, not by
            # naming a transport, so a comm-only kernel (or a future non-HBM
            # interconnect) reads "no HBM traffic" instead of the misleading "no
            # memory traffic" — it did move bytes, just not over HBM.
            def _fmt_ai(ai: float) -> str:
                if ai != float('inf'):
                    return f"{ai:.2f} FLOP/B"
                if critical.total_bytes > 0:
                    return "inf (no HBM traffic)"
                return "inf (no memory traffic)"

            # --- CHIP-LEVEL block (whole chip as one unit) ---
            lines.append("")
            lines.append("Roofline: CHIP-LEVEL  (whole chip as one unit)")
            lines.append("-" * 60)
            lines.append(f"  AI                    : {_fmt_ai(chip['AI'])}")
            lines.append(f"  compute_throughput    : {chip['compute_throughput']:.1%}")
            lines.append(f"  dram_throughput       : {chip['dram_throughput']:.1%}")
            lines.append(f"  mean_core_active_frac : {chip['mean_core_active_frac']:.1%}")
            lines.append(
                f"  grid_coverage         : {chip['cores_active']}/{chip['num_cores']}  "
                f"({chip['grid_coverage']:.1%})"
            )
            lines.append(f"  dominant_unit         : {chip['dominant_unit']}")

            # --- PER-CORE block (critical-path core) ---
            dom = core["core_dominant_unit"]
            lines.append("")
            lines.append("Roofline: PER-CORE  (critical-path core)")
            lines.append("-" * 60)
            lines.append(f"  core_AI               : {_fmt_ai(core['core_AI'])}")
            lines.append(f"  core_efficiency       : {core['core_efficiency']:.1%}")
            lines.append(f"  core_dominant_unit    : {dom}")
            lines.append(f"  peak_bw (per-core)    : {core['peak_bw_gb_s']:.2f} GB/s")
            lines.append("")
            lines.append(
                f"  {'Unit':>10}  {'Achieved':>12}  {'Ceiling':>12}  "
                f"{'Ridge':>10}  {'Eff':>7}  {'CompThru':>9}"
            )
            lines.append(
                f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*7}  {'-'*9}"
            )
            for unit_name, u in core["units"].items():
                marker = " *" if unit_name == dom else "  "
                lines.append(
                    f"{marker} {unit_name:>10}  "
                    f"{u['achieved_gflops']:>10.2f} G  "
                    f"{u['ceiling_gflops']:>10.2f} G  "
                    f"{u['ridge_point']:>8.1f} F/B  "
                    f"{u['efficiency']:>6.1%}  "
                    f"{u['compute_throughput']:>8.2%}"
                )
            lines.append("=" * 60)

        return "\n".join(lines)
