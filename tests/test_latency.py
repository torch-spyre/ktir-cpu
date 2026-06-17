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

#!/usr/bin/env python3
"""Tests for execution latency simulation."""

from collections import Counter
import math
import numpy as np
import pytest
from pathlib import Path

from ktir_cpu import KTIRInterpreter, HardwareConfig, LatencyReport

from conftest import EXAMPLES_DIR, get_test_params, parse_example

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_vector_add(path, func_name, entry, cfg, trace=False):
    """Run vector_add and return (report, outputs)."""
    interp = KTIRInterpreter(latency_config=cfg, trace_latency=trace)
    interp.load(path)

    sizes = interp.tensor_input_output_sizes(func_name)
    n = sizes["x_ptr"]["shape"][0]
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n).astype(np.float16)
    y = rng.standard_normal(n).astype(np.float16)
    out = np.zeros(n, dtype=np.float16)
    kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
    outputs = interp.execute_function(
        func_name, x_ptr=x, y_ptr=y, output_ptr=out, **kwargs
    )
    return interp.get_latency_report(), outputs


def _run_softmax(path, func_name, entry, cfg, trace=False):
    """Run softmax on 32 cores and return report."""
    interp = KTIRInterpreter(latency_config=cfg, trace_latency=trace)
    interp.load(path)

    sizes = interp.tensor_input_output_sizes(func_name)
    n_rows, n_padded_cols = sizes["input_ptr"]["shape"]

    n_real_cols = int(n_padded_cols * 0.76)  # ~76% real data, rest is -inf
    rng = np.random.default_rng(42)
    inp = np.full((n_rows, n_padded_cols), float('-inf'), dtype=np.float16)
    inp[:, :n_real_cols] = rng.standard_normal(
        (n_rows, n_real_cols)
    ).astype(np.float16)
    out = np.zeros((n_rows, n_padded_cols), dtype=np.float16)
    kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
    kwargs["n_cols"] = n_real_cols  # fill dynamic kwarg from actual sizes
    interp.execute_function(
        func_name,
        output_ptr=out, input_ptr=inp,
        **kwargs,
    )
    return interp.get_latency_report()


def _run_matmul(path, func_name, entry, cfg, trace=False):
    """Run matmul on the full grid and return report."""
    interp = KTIRInterpreter(latency_config=cfg, trace_latency=trace)
    interp.load(path)

    kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
    M, N, K = kwargs["M"], kwargs["N"], kwargs["K"]
    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np.float16)
    B = rng.standard_normal((K, N)).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    interp.execute_function(
        func_name,
        a_ptr=A, b_ptr=B, c_ptr=C,
        **kwargs,
    )
    return interp.get_latency_report()


def _run_vector_reduce(path, func_name, entry, cfg, trace=False):
    """Run a vector reduce (per-core tile) and return report."""
    interp = KTIRInterpreter(latency_config=cfg, trace_latency=trace)
    interp.load(path)
    arg_names = interp.arg_names(func_name)
    sizes = interp.tensor_input_output_sizes(func_name)

    arg0 = arg_names[0]
    shape = sizes[arg0]["shape"]
    rng = np.random.default_rng(42)
    inp = rng.standard_normal(tuple(shape)).astype(np.float16)

    scalars = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
    overlap = set(scalars.keys()) & {arg0}
    assert not overlap, f"duplicate keys in tensors and scalars: {overlap}"

    kwargs = {**scalars, **{arg0: inp}}
    interp.execute_function(func_name, **kwargs)
    return interp.get_latency_report()


def _run_ring_reduce(path, func_name, entry, cfg, trace=False):
    """Run ring_reduce on its 4-core grid and return ``(report, rows, n_cols)``.

    Differs from the other ``_run_*`` helpers because ``ring_reduce.mlir``
    takes raw stick-index pointers (``in_ptr`` / ``out_ptr``) rather than
    ndarray kwargs.  We patch ``_prepare_execution`` to seed the input
    rows after HBM allocation; this mirrors
    ``test_examples.py::TestRingReduceExecution::test_ring_reduce_sum``.
    Returning ``rows`` and ``n_cols`` lets callers compute the expected
    ring-traffic byte count without re-deriving it.
    """
    meta = parse_example(path, func_name)
    num_cores = math.prod(meta.grid)
    n_cols = entry["n_cols"]
    in_ptr = entry["execute_kwargs"]["in_ptr"]
    out_ptr = entry["execute_kwargs"]["out_ptr"]

    rng = np.random.default_rng(42)
    rows = rng.uniform(1.0, 2.0, size=(num_cores, n_cols)).astype(np.float16)

    interp = KTIRInterpreter(latency_config=cfg, trace_latency=trace)
    interp.load(path)

    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        interp.memory.hbm.write(in_ptr,  rows.flatten())
        interp.memory.hbm.write(out_ptr, np.zeros(n_cols, dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name, **entry["execute_kwargs"])
    return interp.get_latency_report(), rows, n_cols


def _run_ring_reduce_multi_group(path, func_name, entry, cfg, trace=False):
    """Run the multi-group ring_reduce on its 16-core grid and return
    ``(report, rows, n_cols, n_groups, group_size)``.

    Mirrors ``_run_ring_reduce`` but for the 16-core, 4-group kernel
    in ``examples/latency/ring_reduce_multi_group.mlir``.  Every core
    holds its own 1×128 partial; the kernel runs four concurrent
    in-group all-reduces and writes one output row per group.
    """
    meta = parse_example(path, func_name)
    num_cores = math.prod(meta.grid)
    n_cols = entry["n_cols"]
    n_groups = entry["n_groups"]
    group_size = entry["group_size"]
    in_ptr = entry["execute_kwargs"]["in_ptr"]
    out_ptr = entry["execute_kwargs"]["out_ptr"]

    rng = np.random.default_rng(42)
    rows = rng.uniform(1.0, 2.0, size=(num_cores, n_cols)).astype(np.float16)

    interp = KTIRInterpreter(latency_config=cfg, trace_latency=trace)
    interp.load(path)

    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        interp.memory.hbm.write(in_ptr,  rows.flatten())
        interp.memory.hbm.write(out_ptr, np.zeros(n_groups * n_cols, dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name, **entry["execute_kwargs"])
    return (
        interp.get_latency_report(),
        rows,
        n_cols,
        n_groups,
        group_size,
        interp.memory.hbm.read(out_ptr, n_groups * n_cols, "f16").reshape(n_groups, n_cols),
    )


# ---------------------------------------------------------------------------
# Vector add latency — memory-dominated
# ---------------------------------------------------------------------------

class TestVectorAddLatency:

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_memory_dominated(self, path, func_name, entry):
        """2 loads + 1 store should dominate over 1 addf."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        core0 = report.counters[0]
        assert report.bottleneck == "memory"
        assert core0.memory_cycles > core0.compute_cycles

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    @pytest.mark.parametrize("hbm_bw", [0.5, 1.0, 2.0, 4.0])
    def test_memory_scales_with_bandwidth(self, path, func_name, entry, hbm_bw):
        """Memory cycles should scale inversely with HBM bandwidth."""
        baseline_cfg = HardwareConfig(hbm_bandwidth_tb_s=1.0)
        scaled_cfg = HardwareConfig(hbm_bandwidth_tb_s=hbm_bw)

        baseline, _ = _run_vector_add(path, func_name, entry, baseline_cfg)
        scaled, _ = _run_vector_add(path, func_name, entry, scaled_cfg)

        baseline_mem = baseline.counters[0].memory_cycles
        scaled_mem = scaled.counters[0].memory_cycles

        # Memory cycles should scale as 1/bandwidth
        expected_ratio = 1.0 / hbm_bw
        actual_ratio = scaled_mem / baseline_mem
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    @pytest.mark.parametrize("simd", [32, 64, 128])
    def test_compute_scales_with_simd(self, path, func_name, entry, simd):
        """The addf component of compute cycles should scale inversely with SIMD width.

        The ktir/ vector_add has a scalar muli (offset calculation) that adds
        a fixed 1 cycle regardless of SIMD width.  We verify using traces that
        the tensor addf cycles scale correctly.
        """
        cfg = HardwareConfig(simd_elements_per_cycle=simd)
        report, _ = _run_vector_add(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        # Extract addf cycles from trace
        addf_cycles = sum(e.cycles for e in core0.trace
                          if e.op_type == "arith.addf" and e.category.startswith("compute_"))
        # Per-core tile is 128 elements (BLOCK_SIZE=128), addf costs tile_size / simd
        assert addf_cycles == pytest.approx(128.0 / simd)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    @pytest.mark.parametrize("num_cores", [1, 8, 32])
    def test_per_core_latency_scales_with_num_cores(self, path, func_name, entry, num_cores):
        """Per-core memory cycles scale with num_cores (shared bus),
        but the per-core tile size is fixed by the MLIR, so total bytes
        per core stays constant.  Memory cycles = bytes / (BW / num_cores)."""
        baseline_cfg = HardwareConfig(num_cores=32)
        scaled_cfg = HardwareConfig(num_cores=num_cores)

        baseline, _ = _run_vector_add(path, func_name, entry, baseline_cfg)
        scaled, _ = _run_vector_add(path, func_name, entry, scaled_cfg)

        baseline_mem = baseline.counters[0].memory_cycles
        scaled_mem = scaled.counters[0].memory_cycles

        # Per-core BW = total_BW / num_cores, so memory_cycles scales as num_cores.
        # Ratio relative to baseline (32 cores):
        expected_ratio = num_cores / 32.0
        actual_ratio = scaled_mem / baseline_mem
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_report_formatting(self, path, func_name, entry):
        """Report __str__ should contain key fields."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        text = str(report)
        assert "Kernel cycles" in text
        assert "Bottleneck" in text
        assert "memory" in text

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_summary_dict(self, path, func_name, entry):
        meta = parse_example(path, func_name)
        num_cores = meta.grid[0]

        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        d = report.summary_dict()
        assert "kernel_cycles" in d
        assert "kernel_time_us" in d
        assert "bottleneck" in d
        assert "per_core" in d
        assert len(d["per_core"]) == num_cores
        # grid_cores = counter entries (grid product); num_cores = hardware total.
        assert d["grid_cores"] == num_cores
        assert d["num_cores"] == report.config.num_cores


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------

class TestRoofline:
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_vector_add_flops_and_bytes(self, path, func_name, entry):
        """vector_add: per-core tile → tile FLOPs (one addf), 3×tile×2 bytes."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        core0 = report.counters[0]

        # The per-core access tile is 128 f16 elements (BLOCK_SIZE=128).
        # 3 memory ops (2 loads + 1 store) × 128 elements × 2 bytes = 768
        assert core0.total_bytes == 768
        # addf on 128 elements = 128 FLOPs
        assert core0.total_flops >= 128

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_roofline_returns_sane_values(self, path, func_name, entry):
        """roofline() should return valid metrics with efficiency in [0, 1]."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        rf = report.roofline()

        assert "arithmetic_intensity" in rf
        assert "peak_bw_gb_s" in rf
        assert "dominant_unit" in rf
        assert "efficiency" in rf
        assert "units" in rf

        assert 0 < rf["efficiency"] <= 1.0
        for unit in rf["units"].values():
            assert unit["achieved_gflops"] <= unit["ceiling_gflops"]

        # Chip-level fields (peak-based, Nsight SOL analogue).
        assert "cores_active" in rf
        assert "num_cores" in rf
        assert "grid_coverage" in rf
        assert rf["cores_active"] >= 1
        assert rf["num_cores"] == report.config.num_cores
        # Note: grid_coverage > 1.0 is possible if the kernel grid
        # oversubscribes the modelled core count (e.g. running a 64-core
        # kernel against HardwareConfig(num_cores=32)).
        assert rf["grid_coverage"] > 0
        for unit in rf["units"].values():
            assert unit["peak_gflops"] > 0
            assert unit["chip_peak_gflops"] == pytest.approx(
                unit["peak_gflops"] * rf["num_cores"]
            )
            assert unit["chip_throughput"] >= 0

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_chip_throughput_even_tiling_matches_extrapolation(self, path, func_name, entry):
        """For evenly-tiled work, the chip-wide aggregate equals the
        critical-core extrapolation ``grid_coverage × (achieved / peak)``.

        chip_throughput is defined as the real per-core FLOP sum over elapsed
        time (see test_chip_throughput_uneven_split_uses_aggregate); when every
        active core does equal work the two coincide. This anchors that
        equivalence so the uneven-split divergence is meaningful, not an
        artifact of an unrelated formula change.
        """
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        rf = report.roofline()

        for unit_name, unit in rf["units"].items():
            extrapolation = (rf["grid_coverage"]
                             * unit["achieved_gflops"]
                             / unit["peak_gflops"])
            assert unit["chip_throughput"] == pytest.approx(extrapolation, abs=1e-9), (
                f"even-tiling equivalence broken for unit {unit_name!r}: "
                f"got {unit['chip_throughput']}, expected {extrapolation}"
            )

    def test_chip_throughput_uneven_split_uses_aggregate(self):
        """Under uneven tiling chip_throughput must aggregate the real FLOPs of
        every core, not extrapolate the critical (heaviest) core's rate to all.

        Three cores run for the same wall time but do 300/200/100 matmul FLOPs.
        The aggregate (600) is the true chip work; extrapolating the critical
        core (300 × 3 = 900) would overstate utilization by 1.5×.
        """
        from ktir_cpu.latency import CoreLatencyCounters

        cfg = HardwareConfig(num_cores=32)
        counters = {}
        for cid, flops in enumerate((300.0, 200.0, 100.0)):
            c = CoreLatencyCounters()
            c.record("compute_matmul", cycles=100.0, flops=flops)
            counters[cid] = c

        report = LatencyReport(config=cfg, counters=counters)
        rf = report.roofline()
        unit = rf["units"]["systolic"]

        clock = cfg.clock_ghz * 1e9
        elapsed_s = max(c.total_cycles for c in counters.values()) / clock
        chip_peak = unit["peak_gflops"] * 1e9 * rf["num_cores"]
        total_flops = sum(c.flops_by_category.get("compute_matmul", 0.0)
                          for c in counters.values())  # 600

        expected_aggregate = (total_flops / elapsed_s) / chip_peak
        assert unit["chip_throughput"] == pytest.approx(expected_aggregate)

        # The critical-core extrapolation would be strictly larger (overstated).
        extrapolation = (unit["achieved_gflops"] * 1e9 * rf["cores_active"]) / chip_peak
        assert unit["chip_throughput"] < extrapolation
        assert extrapolation == pytest.approx(1.5 * unit["chip_throughput"])

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_vector_add_is_memory_bound(self, path, func_name, entry):
        """vector_add has low arithmetic intensity → memory-bound on roofline."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        rf = report.roofline()
        dominant = rf["dominant_unit"]

        # AI should be well below the dominant unit's ridge point (memory-bound)
        assert rf["arithmetic_intensity"] < rf["units"][dominant]["ridge_point"]

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("reduce_explicit_region"))
    def test_vector_reduce_is_memory_bound(self, path, func_name, entry):
        """A simple vector reduce should be memory-bound on the roofline."""
        report = _run_vector_reduce(path, func_name, entry, HardwareConfig())
        rf = report.roofline()
        assert "arithmetic_intensity" in rf
        dominant = rf["dominant_unit"]
        assert rf["arithmetic_intensity"] < rf["units"][dominant]["ridge_point"]

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_roofline_in_report_str(self, path, func_name, entry):
        """Roofline section should appear in report __str__."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        text = str(report)
        assert "Roofline Analysis" in text
        assert "Arithmetic intensity" in text
        assert "Efficiency" in text

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("matmul_kernel_small"))
    def test_matmul_flops(self, path, func_name, entry):
        """matmul should report 2*BLOCK_SIZE_M*BLOCK_SIZE_N*K FLOPs across all loop iterations."""
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        bm = kwargs["BLOCK_SIZE_M"]
        bn = kwargs["BLOCK_SIZE_N"]
        bk = kwargs["BLOCK_SIZE_K"]
        K = kwargs["K"]
        n_iters = K // bk  # number of scf.for iterations per core

        report = _run_matmul(path, func_name, entry, HardwareConfig())
        core0 = report.counters[0]

        # Each iteration does one linalg.matmul of shape (bm × bk) × (bk × bn)
        assert core0.total_flops >= 2.0 * bm * bn * bk * n_iters

    def test_empty_report_roofline(self):
        """roofline() on empty report returns empty dict."""
        from ktir_cpu.latency import LatencyReport
        report = LatencyReport(config=HardwareConfig(), counters={})
        assert report.roofline() == {}

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_memory_bound_roofline_matches_bottleneck(self, path, func_name, entry):
        """When bottleneck is memory, roofline AI should be below the dominant unit's ridge point."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        assert report.bottleneck == "memory"
        rf = report.roofline()
        dominant = rf["dominant_unit"]
        assert rf["arithmetic_intensity"] < rf["units"][dominant]["ridge_point"]

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel_small"))
    def test_compute_bound_roofline_matches_bottleneck(self, path, func_name, entry):
        """When bottleneck is compute, roofline AI should be above the dominant unit's ridge point."""
        # Single core + high HBM BW → compute-dominated
        cfg = HardwareConfig(num_cores=1, hbm_bandwidth_tb_s=100.0)
        report = _run_softmax(path, func_name, entry, cfg)
        assert report.bottleneck == "compute"
        rf = report.roofline()
        dominant = rf["dominant_unit"]
        assert rf["arithmetic_intensity"] > rf["units"][dominant]["ridge_point"]

    def test_oversized_grid_does_not_inflate_cores_active(self):
        """An over-large grid leaves some cores with zero loop iterations; those
        idle cores still get a counter entry but must not count toward
        cores_active / grid_coverage (else coverage is overstated)."""
        from ktir_cpu.latency import CoreLatencyCounters

        cfg = HardwareConfig(num_cores=32)
        counters = {}
        # 4 active cores: real matmul work + HBM traffic.
        for cid in range(4):
            c = CoreLatencyCounters()
            c.record("compute_matmul", cycles=100.0, flops=2048.0)
            c.record("memory", cycles=50.0, nbytes=1024)
            counters[cid] = c
        # 28 idle grid cores: counter entry exists, but zero cycles.
        for cid in range(4, 32):
            counters[cid] = CoreLatencyCounters()

        rf = LatencyReport(config=cfg, counters=counters).roofline()
        assert rf["cores_active"] == 4          # not 32 (idle cores excluded)
        assert rf["grid_coverage"] == pytest.approx(4 / 32)


# ---------------------------------------------------------------------------
# Softmax latency — 32 cores
# ---------------------------------------------------------------------------

class TestSoftmaxLatency:
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel_small"))
    def test_softmax_cycle_breakdown(self, path, func_name, entry):
        """Softmax on N cores: math.exp is the dominant compute op (4× penalty),
        memory and compute are roughly balanced, all cores are active."""
        meta = parse_example(path, func_name)
        num_cores = meta.grid[0]
        padded_cols = meta.tensor_sizes["input_ptr"]["shape"][1]

        cfg = HardwareConfig()
        report = _run_softmax(path, func_name, entry, cfg, trace=True)
        assert len(report.counters) == num_cores

        # All cores should have both compute and memory cycles
        for core_id, counters in report.counters.items():
            assert counters.compute_cycles > 0, f"Core {core_id} has zero compute"
            assert counters.memory_cycles > 0, f"Core {core_id} has zero memory"

        # math.exp should be the single largest compute contributor by op type
        core0 = report.counters[0]
        compute_by_op = Counter()
        for e in core0.trace:
            if e.category.startswith("compute_"):
                compute_by_op[e.op_type] += e.cycles
        top_op = compute_by_op.most_common(1)[0]
        assert top_op[0] == "math.exp"

        # Total math.exp cycles per core should be at least
        # core_rows * padded_cols / simd * penalty (one full pass).
        # Two-pass online softmax (rowcolchunk) has ~2× exp due to
        # computing exp in both the stats and output passes.
        n_rows = meta.tensor_sizes["input_ptr"]["shape"][0]
        core_rows = math.ceil(n_rows / num_cores)
        exp_entries = [e for e in core0.trace if e.op_type == "math.exp"]
        total_exp_cycles = sum(e.cycles for e in exp_entries)
        one_pass_exp = (core_rows * padded_cols / cfg.simd_elements_per_cycle) * cfg.transcendental_penalty
        assert total_exp_cycles >= one_pass_exp * 0.99
        assert total_exp_cycles <= one_pass_exp * 2.5  # allow up to ~2× + correction exps

        # Memory and compute should be roughly balanced (within 2× of each other)
        ratio = core0.memory_cycles / core0.compute_cycles
        assert 0.5 < ratio < 2.0

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel_small"))
    @pytest.mark.parametrize("hbm_bw", [0.5, 1.0, 2.0])
    def test_memory_scales_with_bandwidth(self, path, func_name, entry, hbm_bw):
        """Softmax memory cycles should scale inversely with HBM bandwidth."""
        baseline_cfg = HardwareConfig(hbm_bandwidth_tb_s=1.0)
        scaled_cfg = HardwareConfig(hbm_bandwidth_tb_s=hbm_bw)

        baseline = _run_softmax(path, func_name, entry, baseline_cfg)
        scaled = _run_softmax(path, func_name, entry, scaled_cfg)

        baseline_mem = baseline.counters[0].memory_cycles
        scaled_mem = scaled.counters[0].memory_cycles

        # Memory cycles should scale as 1/bandwidth
        expected_ratio = 1.0 / hbm_bw
        actual_ratio = scaled_mem / baseline_mem
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel_small"))
    @pytest.mark.parametrize("penalty", [1, 4, 8])
    def test_transcendental_scales_with_penalty(self, path, func_name, entry, penalty):
        """math.exp cycles should scale linearly with transcendental_penalty."""
        baseline_cfg = HardwareConfig(transcendental_penalty=1)
        scaled_cfg = HardwareConfig(transcendental_penalty=penalty)

        baseline = _run_softmax(path, func_name, entry, baseline_cfg, trace=True)
        scaled = _run_softmax(path, func_name, entry, scaled_cfg, trace=True)

        baseline_exp = sum(e.cycles for e in baseline.counters[0].trace if e.op_type == "math.exp")
        scaled_exp = sum(e.cycles for e in scaled.counters[0].trace if e.op_type == "math.exp")

        # exp cycles should scale linearly with penalty
        assert scaled_exp == pytest.approx(baseline_exp * penalty, rel=1e-3)


# ---------------------------------------------------------------------------
# Reduce latency
# ---------------------------------------------------------------------------

class TestReduceLatency:
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel_small"))
    def test_reduce_defers_cost_to_combiner(self, path, func_name, entry):
        """linalg.reduce is a zero-cost orchestrator (like linalg.generic); the
        reduction cost is charged to the combiner ops executed by the tree fold.
        Folding N elements pairwise processes N/2 + N/4 + … = N-1 elements total,
        so the combiner's *summed* cost scales with input size (~N/simd_width),
        not the reduced output shape.  Holds for the shorthand form (softmax
        uses ``linalg.reduce { arith.maximumf }`` and ``{ arith.addf }``)."""
        cfg = HardwareConfig()
        report = _run_softmax(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        # The orchestrator op itself charges nothing.
        reduce_entries = [e for e in core0.trace if e.op_type == "linalg.reduce"]
        assert len(reduce_entries) > 0, "expected at least one linalg.reduce in softmax"
        for e in reduce_entries:
            assert e.category == "zero" and e.cycles == 0.0, (
                f"linalg.reduce must be zero-cost, got {e.category}/{e.cycles}"
            )

        # Combiner ops carry the cost. Core 0 processes 2 rows; per row a max
        # (arith.maximumf) and a sum (arith.addf) each fold a 1×64 tile → the
        # tree fold processes N-1 elements, plus one final combine with the outs
        # accumulator (1 element). Total per reduce: N-1+1 = N elements.
        per_reduce = 64 / cfg.simd_elements_per_cycle  # N-1 tree fold + 1 outs combine
        for combiner in ("arith.maximumf", "arith.addf"):
            total = sum(e.cycles for e in core0.trace if e.op_type == combiner)
            assert total == pytest.approx(2 * per_reduce), (
                f"{combiner} should total {2 * per_reduce} cyc over 2 rows "
                f"(tree fold of 1×64 + outs combine each); got {total}"
            )

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("reduce_explicit_region"))
    def test_reduce_explicit_region_charges_combiner(self, path, func_name, entry):
        """Explicit-region form routes through the same tree-fold path: the
        combiner op in the region (arith.addf) carries the cycles and
        linalg.reduce itself is zero-cost."""
        cfg = HardwareConfig()
        report = _run_vector_reduce(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        reduce_entries = [e for e in core0.trace if e.op_type == "linalg.reduce"]
        assert reduce_entries and all(
            e.category == "zero" and e.cycles == 0.0 for e in reduce_entries
        ), "linalg.reduce must be zero-cost in the explicit-region form too"

        # reduce_generic.mlir folds a 1×4 input tile with arith.addf. A pairwise
        # tree fold of 4 elements processes N-1=3 elements, plus one final
        # combine with the outs accumulator (1 element). Total: 4 elements.
        expected = 4 / cfg.simd_elements_per_cycle  # N-1 tree fold + 1 outs combine
        total = sum(e.cycles for e in core0.trace if e.op_type == "arith.addf")
        assert total == pytest.approx(expected), (
            f"combiner arith.addf should total {expected} cyc "
            f"(tree fold of 1×4 + outs combine); got {total}"
        )

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("reduce_explicit_region"))
    def test_reduce_kernel_is_memory_bound(self, path, func_name, entry):
        """Vector reduce per-core tile should be memory-dominated."""
        report = _run_vector_reduce(path, func_name, entry, HardwareConfig())
        core0 = report.counters[0]
        assert report.bottleneck == "memory"
        assert core0.memory_cycles > core0.compute_cycles

    @pytest.mark.parametrize("path,func_name,entry",
                             get_test_params("softmax_kernel_small_explicit"))
    def test_explicit_region_softmax_matches_shorthand(self, path, func_name, entry):
        """Softmax with explicit (%in,%out){...} combiner regions charges the
        same combiner cost as the shorthand softmax — proving both forms feed
        the identical tree-fold path. linalg.reduce stays zero-cost."""
        cfg = HardwareConfig()
        report = _run_softmax(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        assert all(e.cycles == 0.0 for e in core0.trace if e.op_type == "linalg.reduce")
        per_reduce = 64 / cfg.simd_elements_per_cycle  # N-1 tree fold + 1 outs combine
        for combiner in ("arith.maximumf", "arith.addf"):
            total = sum(e.cycles for e in core0.trace if e.op_type == combiner)
            assert total == pytest.approx(2 * per_reduce), (
                f"{combiner} should total {2 * per_reduce} cyc (explicit-region "
                f"softmax, 2 rows of 1×64 tree fold + outs combine); got {total}"
            )

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("reduce_multiop"))
    def test_multiop_combiner_charges_all_region_ops(self, path, func_name, entry):
        """A multi-op combiner (max via cmpf+select) charges EVERY op in the
        region — there is no single-combiner-name shortcut. Both arith.cmpf and
        arith.select carry the tree-fold cost; linalg.reduce is zero."""
        cfg = HardwareConfig()
        report = _run_vector_reduce(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        assert all(e.cycles == 0.0 for e in core0.trace if e.op_type == "linalg.reduce")
        # 1×8 tree fold → 7 elements processed + 1 outs combine = 8 total.
        expected = 8 / cfg.simd_elements_per_cycle  # N-1 tree fold + 1 outs combine
        for region_op in ("arith.cmpf", "arith.select"):
            total = sum(e.cycles for e in core0.trace if e.op_type == region_op)
            assert total == pytest.approx(expected), (
                f"{region_op} should total {expected} cyc (1×8 tree fold + outs combine); "
                f"got {total}"
            )


# ---------------------------------------------------------------------------
# Matmul latency
# ---------------------------------------------------------------------------

class TestMatmulLatency:
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("matmul_kernel_small"))
    def test_matmul_cycle_breakdown(self, path, func_name, entry):
        """Matmul: linalg.matmul cost derived from block shape,
        memory dominates from loading A, B tiles and storing C."""
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        bm = kwargs["BLOCK_SIZE_M"]   # 32
        bn = kwargs["BLOCK_SIZE_N"]   # 512
        bk = kwargs["BLOCK_SIZE_K"]   # 128
        K = kwargs["K"]               # 2048
        n_iters = K // bk            # 16 scf.for iterations per core

        cfg = HardwareConfig()
        report = _run_matmul(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        # Each scf.for iteration produces one linalg.matmul entry
        matmul_entries = [e for e in core0.trace if e.op_type == "linalg.matmul"]
        assert len(matmul_entries) == n_iters
        # Each matmul: 2 * bm * bn * bk FLOPs / systolic_flops_per_cycle
        expected_per_iter = (2.0 * bm * bn * bk) / cfg.systolic_flops_per_cycle
        for entry_e in matmul_entries:
            assert entry_e.cycles == pytest.approx(expected_per_iter)

        # Memory should dominate (tile loads >> matmul cycles)
        assert core0.memory_cycles > core0.compute_cycles
        assert report.bottleneck == "memory"

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("matmul_kernel_small"))
    @pytest.mark.parametrize("systolic", [
        2 * 32 * 32 * 32,   # smaller array → more compute cycles
        2 * 64 * 64 * 64,   # default
        2 * 128 * 128 * 128, # larger array → fewer compute cycles
    ])
    def test_matmul_scales_with_systolic_throughput(self, path, func_name, entry, systolic):
        """linalg.matmul cycles should scale inversely with systolic throughput."""
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        bm = kwargs["BLOCK_SIZE_M"]
        bn = kwargs["BLOCK_SIZE_N"]
        bk = kwargs["BLOCK_SIZE_K"]

        cfg = HardwareConfig(systolic_flops_per_cycle=systolic)
        report = _run_matmul(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        matmul_entries = [e for e in core0.trace if e.op_type == "linalg.matmul"]
        expected_per_iter = (2.0 * bm * bn * bk) / systolic
        for entry_e in matmul_entries:
            assert entry_e.cycles == pytest.approx(expected_per_iter)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("matmul_kernel_small"))
    @pytest.mark.parametrize("hbm_bw", [0.5, 1.0, 4.0])
    def test_memory_scales_with_bandwidth(self, path, func_name, entry, hbm_bw):
        """Matmul memory cycles should scale inversely with HBM bandwidth."""
        baseline_cfg = HardwareConfig(hbm_bandwidth_tb_s=1.0)
        scaled_cfg = HardwareConfig(hbm_bandwidth_tb_s=hbm_bw)

        baseline = _run_matmul(path, func_name, entry, baseline_cfg)
        scaled = _run_matmul(path, func_name, entry, scaled_cfg)

        baseline_mem = baseline.counters[0].memory_cycles
        scaled_mem = scaled.counters[0].memory_cycles

        expected_ratio = 1.0 / hbm_bw
        actual_ratio = scaled_mem / baseline_mem
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3)


# ---------------------------------------------------------------------------
# Latency disabled — default behavior unchanged
# ---------------------------------------------------------------------------

class TestLatencyDisabled:
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_none_report(self, path, func_name, entry):
        """Default interpreter (no config) returns None report."""
        meta = parse_example(path, func_name)
        n = meta.tensor_sizes["x_ptr"]["shape"][0]

        interp = KTIRInterpreter()
        interp.load(path)

        x = np.zeros(n, dtype=np.float16)
        y = np.zeros(n, dtype=np.float16)
        output = np.zeros(n, dtype=np.float16)

        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        interp.execute_function(
            func_name, x_ptr=x, y_ptr=y, output_ptr=output, **kwargs
        )

        assert interp.get_latency_report() is None

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_identical_results(self, path, func_name, entry):
        """Functional results should be identical with and without latency tracking."""
        meta = parse_example(path, func_name)
        n = meta.tensor_sizes["x_ptr"]["shape"][0]

        rng = np.random.default_rng(42)
        x = rng.standard_normal(n).astype(np.float16)
        y = rng.standard_normal(n).astype(np.float16)

        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}

        # Without latency
        interp1 = KTIRInterpreter()
        interp1.load(path)
        out1 = interp1.execute_function(
            func_name, x_ptr=x.copy(), y_ptr=y.copy(),
            output_ptr=np.zeros(n, dtype=np.float16), **kwargs
        )

        # With latency
        interp2 = KTIRInterpreter(latency_config=HardwareConfig())
        interp2.load(path)
        out2 = interp2.execute_function(
            func_name, x_ptr=x.copy(), y_ptr=y.copy(),
            output_ptr=np.zeros(n, dtype=np.float16), **kwargs
        )

        np.testing.assert_array_equal(out1["output_ptr"], out2["output_ptr"])

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_execute_resets_counters(self, path, func_name, entry):
        """Each execute_function call should reset latency counters."""
        meta = parse_example(path, func_name)
        n = meta.tensor_sizes["x_ptr"]["shape"][0]

        interp = KTIRInterpreter(latency_config=HardwareConfig())
        interp.load(path)

        x = np.zeros(n, dtype=np.float16)
        y = np.zeros(n, dtype=np.float16)
        out = np.zeros(n, dtype=np.float16)

        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}

        # First run
        interp.execute_function(
            func_name, x_ptr=x, y_ptr=y, output_ptr=out, **kwargs
        )
        first_cycles = interp.get_latency_report().kernel_cycles

        # Second run — counters should reflect only the second execution
        interp.execute_function(
            func_name, x_ptr=x, y_ptr=y, output_ptr=out, **kwargs
        )
        second_cycles = interp.get_latency_report().kernel_cycles

        assert second_cycles == pytest.approx(first_cycles)
        # If counters accumulated, second_cycles would be ~2× first_cycles


# ---------------------------------------------------------------------------
# Latency edge cases
# ---------------------------------------------------------------------------

class TestLatencyEdgeCases:
    """Edge-case tests for the latency model: large SIMD widths and
    zero-element tiles."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_large_simd_exceeding_tile_size(self, path, func_name, entry):
        """SIMD width larger than the per-core tile still produces valid cycles.

        When simd_elements_per_cycle > tile size, the addf should cost
        tile_size / simd < 1 cycle (fractional).  The model should not
        produce negative or NaN cycles.
        """
        # Per-core tile is 128 elements (BLOCK_SIZE=128)
        cfg = HardwareConfig(simd_elements_per_cycle=1024)
        report, _ = _run_vector_add(path, func_name, entry, cfg, trace=True)
        core0 = report.counters[0]

        # addf on 128 elements with SIMD=1024 → 128/1024 = 0.125 cycles
        addf_cycles = sum(e.cycles for e in core0.trace
                          if e.op_type == "arith.addf" and e.category.startswith("compute_"))
        assert addf_cycles == pytest.approx(128.0 / 1024)
        assert addf_cycles > 0
        assert not math.isnan(addf_cycles)

        # Total cycles should still be valid and positive
        assert core0.total_cycles > 0
        assert report.kernel_cycles > 0

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_very_large_simd_compute_near_zero(self, path, func_name, entry):
        """With extremely large SIMD, compute is negligible; memory dominates entirely."""
        cfg = HardwareConfig(simd_elements_per_cycle=1_000_000)
        report, _ = _run_vector_add(path, func_name, entry, cfg)
        core0 = report.counters[0]

        # Compute should be negligible compared to memory
        if core0.compute_cycles > 0:
            assert core0.memory_cycles / core0.compute_cycles > 100
        assert report.bottleneck == "memory"

    def test_zero_element_tile_latency(self):
        """A zero-element Tile should report zero bytes and zero FLOPs.

        The latency tracker's _data_size and _num_elements helpers should
        handle zero-size arrays gracefully.
        """
        from ktir_cpu.latency import LatencyTracker, CoreLatencyCounters
        from ktir_cpu.ir_types import Tile

        cfg = HardwareConfig()
        tracker = LatencyTracker(cfg)

        # A zero-element tile (e.g. empty slice). unique_sticks=0 honors
        # the HBM-load contract: a zero-element load spans zero sticks.
        zero_tile = Tile(
            np.array([], dtype=np.float16), "f16", (0,), unique_sticks=0,
        )
        assert zero_tile.size_bytes() == 0

        # _data_size should return 0 for a zero-element result
        nbytes = LatencyTracker._data_size(zero_tile, [])
        assert nbytes == 0

        # _num_elements should return 0 for a zero-element tile
        n_elems = LatencyTracker._num_elements(zero_tile, [])
        assert n_elems == 0

    def test_lx_index_views_excluded_from_hbm_bytes(self):
        """_data_size() ignores LX index views; _memory_space() falls back to parent.

        The result Tile is constructed with ``index_unique_sticks=0`` to
        honor the IAT-load contract — in real workflow ``indirect_load``
        produces ``0`` for an all-LX IAT (LX has no stick concept).
        """
        from ktir_cpu.latency import LatencyTracker
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef, Tile
        from ktir_cpu.parser_ast import parse_affine_set

        vss = parse_affine_set("(d0, d1) : (d0 >= 0, d1 >= 0)")
        lx_idx = MemRef(base_ptr=0, shape=(4, 4), strides=[4, 1],
                        memory_space="LX", dtype="i32")
        parent = MemRef(base_ptr=0, shape=(4, 4), strides=[4, 1],
                        memory_space="HBM", dtype="f16")
        iat = IndirectAccessTile(
            parent_ref=parent, shape=(4, 4), dim_subscripts=[],
            index_views=[lx_idx, lx_idx],
            variables_space_set=vss, variables_space_order=None,
        )
        # 4x4 f16 = 32 bytes — fits within one 128-byte stick.
        # index_unique_sticks=0 honors the IAT-load contract for an
        # all-LX IAT (LX has no stick concept).
        result = Tile(
            np.zeros((4, 4), dtype=np.float16), "f16", (4, 4),
            unique_sticks=1,
            index_unique_sticks=0,
        )

        # Data side: 1 stick * 128 bytes. Idx side: 0 (all-LX index views
        # contribute nothing). Total stays stick-granular, not data.nbytes.
        assert LatencyTracker._data_size(result, [iat]) == 1 * 128
        assert LatencyTracker._memory_space([iat]) == "HBM"

    def test_empty_counters_bottleneck(self):
        """LatencyReport with no counters reports bottleneck='none'."""
        from ktir_cpu.latency import LatencyReport
        report = LatencyReport(config=HardwareConfig(), counters={})
        assert report.bottleneck == "none"
        assert report.kernel_cycles == 0.0
        assert report.kernel_time_us == 0.0


class TestIndirectAccessLatency:
    """Verify that indirect access loads account for index tensor HBM traffic."""

    @pytest.mark.parametrize("path,func_name,_entry", get_test_params("indirect_access_copy"))
    def test_indirect_load_includes_index_tensor_bytes(self, path, func_name, _entry):
        """memory_cycles should reflect index tensor reads, not just the result tile.

        indirect-access-copy.mlir does a 2-D gather: Y[m,k] = X[IDX1[m,k], IDX2[m,k]]
        with 64x64 tiles.  Here IDX1/IDX2 are seeded with zeros (see
        ``_prepare_and_seed`` below), so every gather element reads X[0,0] —
        all 4096 reads land on a single 128-byte stick.  The single
        ``ktdp.load`` on the IndirectAccessTile therefore costs:
          result (X gather):  unique_sticks * 128 = 1 * 128 = 128 bytes
          IDX1:               64*64*4 (i32)       = 16,384 bytes
          IDX2:               64*64*4 (i32)       = 16,384 bytes
        plus the Y store of 64*64*2 = 8,192 bytes.

        The ``unique_sticks`` accounting (see ``Tile.unique_sticks``)
        replaces the previous optimistic ``result.data.nbytes`` —
        scattered gathers now charge the real per-stick HBM traffic.
        """
        cfg = HardwareConfig(num_cores=1)
        interp = KTIRInterpreter(latency_config=cfg)
        interp.load(path)

        sizes = interp.tensor_input_output_sizes(func_name)
        _dtype_map = {"f16": np.float16, "i32": np.int32, "f32": np.float32}

        # Derive addresses from parsed module so the test stays correct if
        # indirect-access-copy.mlir changes its arith.constant values.
        func = interp.module.get_function(func_name)
        constants = {
            op.result.lstrip("%"): op.attributes["value"]
            for op in func.operations
            if op.op_type == "arith.constant" and op.result
        }
        _addr_map = {name: constants[name] for name in sizes}

        from ktir_cpu.memory import HBMSimulator
        _orig = interp._prepare_execution
        def _prepare_and_seed(grid_shape):
            _orig(grid_shape)
            hbm = interp.memory.hbm
            for name, info in sizes.items():
                n_elements = int(np.prod(info["shape"]))
                hbm.write(_addr_map[name],
                          np.zeros(n_elements, dtype=_dtype_map[info["dtype"]]))
        interp._prepare_execution = _prepare_and_seed

        interp.execute_function(func_name)
        report = interp.get_latency_report()

        # With 1 core, all work is on core 0.
        counters = report.counters[0]

        # The kernel does 1 indirect load (X via IDX1+IDX2) + 1 regular store (Y).
        def _nbytes(name):
            info = sizes[name]
            return int(np.prod(info["shape"])) * np.dtype(_dtype_map[info["dtype"]]).itemsize
        # Zero-seeded indices collapse every gather read to X[0,0] → 1 unique stick.
        expected_gather_bytes = 1 * 128
        expected_load_bytes = expected_gather_bytes + _nbytes("IDX1_addr") + _nbytes("IDX2_addr")
        expected_store_bytes = _nbytes("Y_addr")
        expected_total_bytes = expected_load_bytes + expected_store_bytes
        bw = cfg.hbm_bytes_per_cycle_per_core
        expected_memory_cycles = expected_total_bytes / bw

        assert counters.total_bytes == expected_total_bytes, (
            f"total_bytes={counters.total_bytes}, expected={expected_total_bytes}"
        )
        assert counters.memory_cycles == pytest.approx(expected_memory_cycles, rel=1e-3)

    # ---------------------------------------------------------------------
    # Unit tests for the stick-counting formula used by gather latency.
    # These exercise ``MemoryOps._count_unique_sticks`` and ``_data_size``
    # directly, without standing up a full interpreter / HBM.
    # ---------------------------------------------------------------------

    def test_flat_memory_offsets_returns_n_sticks_when_fully_scattered(self):
        """_flat_memory_offsets returns n_elements sticks when every element lands on its own."""
        from ktir_cpu.ops.memory_ops import MemoryOps

        # f16 stick holds 64 elements; indices 0, 64, 128, 192 each land on
        # a different stick — no sharing.
        coords = [(i * 64,) for i in range(4)]
        _, unique_sticks = MemoryOps._flat_memory_offsets(
            base_ptr=0x10000, shape=(4096,), strides=[1], dtype="f16",
            coords=coords, stick_bytes=128
        )
        assert unique_sticks == 4

    def test_flat_memory_offsets_dedups_sticks_shared_by_multiple_reads(self):
        """_flat_memory_offsets collapses repeated coords into distinct sticks."""
        from ktir_cpu.ops.memory_ops import MemoryOps

        # Six reads alternate between element 0 and element 64 — two sticks.
        coords = [(0,), (64,), (0,), (64,), (0,), (64,)]
        _, unique_sticks = MemoryOps._flat_memory_offsets(
            base_ptr=0x10000, shape=(4096,), strides=[1], dtype="f16",
            coords=coords, stick_bytes=128
        )
        assert unique_sticks == 2

    def test_data_size_uses_unique_sticks_for_gather_result(self):
        """_data_size charges ``unique_sticks * 128`` when the field is set."""
        from ktir_cpu.ir_types import Tile
        from ktir_cpu.latency import LatencyTracker

        # 64 f16 elements = 128 bytes packed, but scattered across 64 sticks
        # (each element on its own stick): actual traffic = 64 * 128 = 8192.
        result = Tile(np.zeros(64, dtype=np.float16), "f16", (64,),
                      unique_sticks=64)

        assert LatencyTracker._data_size(result, []) == 64 * 128

    def test_coalescing_efficiency_returns_bpe_over_stick_for_worst_case(self):
        """Tile.coalescing_efficiency drops to bpe/128 when each element owns a stick."""
        from ktir_cpu.ir_types import Tile

        # 64 f16 elements scattered across 64 sticks: efficiency = 2 / 128.
        tile = Tile(np.zeros(64, dtype=np.float16), "f16", (64,), unique_sticks=64)

        assert tile.coalescing_efficiency == 2 / 128

    def test_coalescing_efficiency_is_none_for_non_gather_tile(self):
        """Tile.coalescing_efficiency is None when unique_sticks is not set."""
        from ktir_cpu.ir_types import Tile

        tile = Tile(np.zeros(64, dtype=np.float16), "f16", (64,))  # default None

        assert tile.coalescing_efficiency is None

    def test_copy_propagates_unique_sticks(self):
        """Tile.copy() preserves unique_sticks — it's a property of the data layout.

        This may change depending on the final implementation of comm_ops —
        if copies land at a different base_ptr, unique_sticks may need to be
        recomputed for the target device.
        """
        from ktir_cpu.ir_types import Tile

        original = Tile(np.zeros(64, dtype=np.float16), "f16", (64,), unique_sticks=7)

        assert original.copy().unique_sticks == 7

    def test_copy_propagates_index_unique_sticks(self):
        """Tile.copy() preserves index_unique_sticks alongside unique_sticks.

        Both fields describe HBM traffic of the load that produced the tile;
        a deep copy of the tile data does not change either.
        """
        from ktir_cpu.ir_types import Tile

        original = Tile(
            np.zeros(64, dtype=np.float16), "f16", (64,),
            unique_sticks=7, index_unique_sticks=11,
        )
        copied = original.copy()
        assert copied.unique_sticks == 7
        assert copied.index_unique_sticks == 11

    def test_data_size_charges_index_unique_sticks(self):
        """_data_size adds ``index_unique_sticks * STICK_BYTES`` for indirect_load result.

        For a gather result with both fields set:
          data side  = unique_sticks * 128
          idx side   = index_unique_sticks * 128
        ``_data_size`` returns the sum.
        """
        from ktir_cpu.ir_types import Tile
        from ktir_cpu.latency import LatencyTracker

        # 1 stick of data + 3 sticks of idx reads = (1 + 3) * 128 = 512.
        result = Tile(
            np.zeros(64, dtype=np.float16), "f16", (64,),
            unique_sticks=1, index_unique_sticks=3,
        )

        assert LatencyTracker._data_size(result, []) == (1 + 3) * 128

    @pytest.mark.parametrize("idx_sticks", [
        pytest.param(5, id="positive_idx_sticks"),
        pytest.param(0, id="zero_idx_sticks_lx_only_safe"),
    ])
    def test_data_size_iat_load_skips_operand_branch_when_result_field_set(
        self, idx_sticks,
    ):
        """When ``result.index_unique_sticks`` is set (any int, including 0),
        the IAT operand branch in :meth:`LatencyTracker._data_size` is
        skipped — load case routes through the result field, sidestepping
        the side-channel ``_idx_unique_sticks_no_reads(iat)`` charge.

        Two regression scenarios are covered:

        * positive (``idx_sticks=5``): an HBM IAT whose
          ``_idx_unique_sticks_no_reads`` would itself give a positive
          count. If the operand branch ever stops being skipped on the
          load path, the assertion catches the double-charge.
        * zero (``idx_sticks=0``): an LX-only IAT load legitimately
          stamps ``0`` on the result. The gated raise must distinguish
          ``0`` from ``None`` and let this through.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef, Tile
        from ktir_cpu.latency import LatencyTracker
        from ktir_cpu.parser_ast import parse_affine_set

        # HBM idx_view: _idx_unique_sticks_no_reads(iat) would give a
        # positive count if the operand branch fired. The result field
        # forces it skipped.
        idx_view = MemRef(
            base_ptr=0, shape=(4,), strides=[1],
            memory_space="HBM", dtype="i32",
        )
        iat = IndirectAccessTile(
            parent_ref=idx_view, shape=(4,),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            variables_space_set=parse_affine_set(
                "(d0) : (d0 >= 0, -d0 + 3 >= 0)"
            ),
            variables_space_order=None,
        )
        result = Tile(
            np.zeros(4, dtype=np.float16), "f16", (4,),
            unique_sticks=2, index_unique_sticks=idx_sticks,
        )

        # Expected: (data sticks + idx sticks from result field) * 128.
        # An operand-branch double-charge would add
        # _idx_unique_sticks_no_reads(iat) * 128 on top, breaking equality.
        assert LatencyTracker._data_size(result, [iat]) == (2 + idx_sticks) * 128

    def test_read_scattered_empty_raises(self):
        """_MemAccessor.read_scattered rejects empty address lists.

        Empty input is ambiguous — could mean "zero sticks" or "caller bug" —
        so raise rather than silently return ``([], 0)``.
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.hbm = MagicMock()
        accessor = _MemAccessor(ctx, "HBM", byte_addr=0x10000)

        with pytest.raises(ValueError, match="empty address list"):
            accessor.read_scattered([], "i32")

    # ---------------------------------------------------------------------
    # _MemAccessor.count_sticks — single source of truth for stick counting.
    # ---------------------------------------------------------------------

    @pytest.mark.parametrize("addrs,expected", [
        # All in stick 0 (bytes 0..127): 1 stick total.
        pytest.param([0, 4, 8, 12], 1, id="four_addrs_one_stick"),
        # Repeats fold into the same stick: dedup is by stick, not address.
        pytest.param([0, 0, 0, 0], 1, id="repeated_addrs_same_stick"),
        # 0 → stick 0, 128 → stick 1, 256 → stick 2.
        pytest.param([0, 128, 256], 3, id="three_distinct_sticks"),
        # 0 + 4 share stick 0; 128 lives on stick 1.
        pytest.param([0, 4, 128], 2, id="subset_two_sticks"),
        # Empty input: defined "no traffic" (kept distinct from None).
        pytest.param([], 0, id="empty_returns_zero"),
    ])
    def test_count_sticks_hbm(self, addrs, expected):
        """_MemAccessor.count_sticks counts distinct HBM sticks via set dedup.

        Stick boundaries are 128 bytes (HBMSimulator.STICK_BYTES):
        addresses 0..127 share stick 0, 128..255 share stick 1, etc.
        Counting routes through this classmethod so callers stay free
        of ``addr // STICK_BYTES`` arithmetic.
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor

        assert _MemAccessor.count_sticks("HBM", addrs) == expected

    def test_count_sticks_lx_returns_none(self):
        """LX has no stick concept — count_sticks returns None for any input.

        ``None`` is the LX answer (no stick boundaries exist there);
        ``0`` is HBM's answer for an empty address list. The two are
        kept distinct so callers can route on memory space.
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor

        assert _MemAccessor.count_sticks("LX", [0, 128, 256]) is None
        assert _MemAccessor.count_sticks("LX", []) is None

    # ---------------------------------------------------------------------
    # _MemAccessor.read_scattered — per-element reads with stick counting.
    # ---------------------------------------------------------------------

    def _seeded_hbm_accessor(self):
        """Build an HBM accessor over a freshly-allocated, pre-seeded region.

        Layout: stick 0 (bytes 0..127) holds i32 values [10, 20, 30, 40]
        at byte offsets 0, 4, 8, 12. Stick 1 (bytes 128..255) holds
        i32 value 99 at byte offset 128. All other addresses are
        unwritten (would read zero).
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor
        from ktir_cpu.memory import HBMSimulator
        from unittest.mock import MagicMock

        hbm = HBMSimulator()
        hbm.write(0, np.array([10, 20, 30, 40], dtype=np.int32))
        hbm.write(1, np.array([99], dtype=np.int32))

        ctx = MagicMock()
        ctx.hbm = hbm
        return _MemAccessor(ctx, "HBM", byte_addr=0)

    @pytest.mark.parametrize("addrs,expected_values,expected_sticks", [
        # Per-element read in input order, all sharing stick 0.
        pytest.param([0, 4, 8, 12], [10, 20, 30, 40], 1, id="dense_within_stick"),
        # Repeated address reads the same value each time (cache hit
        # internally); stick count is set-deduped to 1.
        pytest.param([0, 0, 0], [10, 10, 10], 1, id="repeated_same_addr"),
        # Subset preserves order and skips unaccessed addresses.
        pytest.param([12, 0], [40, 10], 1, id="subset_preserves_order"),
        # Two distinct sticks: one address per stick, span 128 bytes apart.
        pytest.param([0, 128], [10, 99], 2, id="two_sticks"),
    ])
    def test_read_scattered_hbm_per_element(
        self, addrs, expected_values, expected_sticks,
    ):
        """read_scattered reads one element per address, returns (values, sticks).

        Stick count is set-deduped on the address side regardless of how
        many physical reads are issued; values are returned in caller
        order. Per-stick formula: ``len({addr // 128 for addr in addrs})``.
        """
        accessor = self._seeded_hbm_accessor()

        values, sticks = accessor.read_scattered(addrs, "i32")
        assert list(values) == expected_values
        assert sticks == expected_sticks

    def test_read_scattered_lx_returns_none_sticks(self):
        """read_scattered on LX returns ``None`` for the stick count.

        Mirror of :meth:`_MemAccessor.count_sticks` semantics — LX has
        no stick concept, so the second tuple element is ``None`` and
        callers must skip stick-based latency accounting.
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor
        from ktir_cpu.memory import LXScratchpad
        from unittest.mock import MagicMock

        lx = LXScratchpad()
        lx.write(0, np.array([10, 20, 30, 40], dtype=np.int32))

        ctx = MagicMock()
        ctx.get_lx = MagicMock(return_value=lx)
        accessor = _MemAccessor(ctx, "LX", byte_addr=0, lx_core_id=0)

        values, sticks = accessor.read_scattered([0, 4, 8, 12], "i32")
        assert list(values) == [10, 20, 30, 40]
        assert sticks is None

    # ---------------------------------------------------------------------
    # _MemAccessor.read_scattered — contiguous-run batching.
    #
    # Runs are formed by sorting the unique addresses and merging any pair
    # whose diff equals ``bpe`` (one element apart in the access dtype).
    # Each run becomes one ``sim.read(start, n=run_len, dtype, intra_byte)``
    # call, which models a single DMA descriptor. Number of calls = run
    # count, which equals 1 for fully dense access and ``unique_sticks``
    # in the worst case (every address isolated by a gap).
    # ---------------------------------------------------------------------

    @pytest.mark.parametrize(
        "addrs,expected_call_args",
        [
            # Dense within a stick: all four 4-byte gaps, one run of 4.
            pytest.param(
                [0, 4, 8, 12],
                [(0, 4, "i32", 0)],
                id="dense_within_stick_one_run",
            ),
            # Mid-stick start: intra_byte propagated; one run of 3.
            pytest.param(
                [4, 8, 12],
                [(0, 3, "i32", 4)],
                id="mid_stick_run_intra_byte_propagated",
            ),
            # Two pairs of dense addresses, separated by a stick boundary:
            # ``[0, 4, 128, 132]`` → two runs of 2 (one per stick).
            pytest.param(
                [0, 4, 128, 132],
                [(0, 2, "i32", 0), (1, 2, "i32", 0)],
                id="scattered_two_runs_of_two",
            ),
            # Two isolated points (gap > bpe between them): two runs of 1.
            pytest.param(
                [0, 128],
                [(0, 1, "i32", 0), (1, 1, "i32", 0)],
                id="two_isolated_points_two_runs",
            ),
            # Repeated address: 1 unique → 1 run; values broadcast in caller order.
            pytest.param(
                [0, 0, 0],
                [(0, 1, "i32", 0)],
                id="repeated_addr_one_run",
            ),
            # Non-monotonic input: caller order ``[12, 0]`` resolves via runs
            # sorted by address, but values reassemble in input order.
            pytest.param(
                [12, 0],
                [(0, 1, "i32", 0), (0, 1, "i32", 12)],
                id="non_monotonic_input_runs_sorted_values_in_input_order",
            ),
            # Run that crosses a stick boundary: addresses ``[124, 128]``
            # differ by ``bpe`` so they merge; ``intra_byte=124``, ``n=2``
            # crosses sticks. ``HBMSimulator._read_flat`` handles this
            # as long as the underlying allocation spans both sticks.
            pytest.param(
                [124, 128],
                [(0, 2, "i32", 124)],
                id="run_crosses_stick_boundary_one_call",
            ),
        ],
    )
    def test_read_scattered_run_batching_call_count_and_args(
        self, addrs, expected_call_args,
    ):
        """read_scattered groups adjacent (diff == bpe) addresses into runs.

        Each run is a single ``sim.read(start, n=run_len, dtype, intra_byte)``
        call — the simulator-side equivalent of one DMA descriptor.
        Stubs ``ctx.hbm.read`` with a synthetic byte-addressable layout
        (``value = (byte_address // 4) + 10``) so the run-merging algorithm
        can be exercised over arbitrary stick spans without setting up a
        real HBM region. Verifies (a) actual calls match the expected
        run decomposition (sorted-address order), (b) returned values
        reassemble in caller's input order, (c) ``unique_sticks``
        matches set-dedup over ``addr // 128``.
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor
        from unittest.mock import MagicMock

        def fake_read(stick, n, dtype, *, intra_byte=0):
            base_byte = stick * 128 + intra_byte
            return np.array(
                [(base_byte // 4) + 10 + i for i in range(n)],
                dtype=np.int32,
            )

        ctx = MagicMock()
        ctx.hbm.read = MagicMock(side_effect=fake_read)
        accessor = _MemAccessor(ctx, "HBM", byte_addr=0)

        values, sticks = accessor.read_scattered(addrs, "i32")
        # Synthetic layout mirrors the stub: caller sees the byte-address
        # formula directly, so expected values derive from input addrs.
        assert list(values) == [(a // 4) + 10 for a in addrs]
        # Stick count: set-deduped over unique stick indices.
        assert sticks == len({a // 128 for a in addrs})
        # Run-batching assertion: one sim.read per run, in sorted-address order.
        actual_calls = [
            (c.args[0], c.args[1], c.args[2], c.kwargs.get("intra_byte", 0))
            for c in ctx.hbm.read.call_args_list
        ]
        assert actual_calls == expected_call_args

    def test_read_scattered_run_batching_lx_path(self):
        """LX run-batching: ``sim.read(byte_addr, n, dtype)``, no intra_byte.

        LX has no stick concept (``stick_bytes = None``), so ``run_start``
        is passed as the byte address directly. Runs still merge by
        ``diff == bpe``; the test asserts both the call shape and that
        ``unique_sticks`` is ``None`` (LX semantics from
        :meth:`_MemAccessor.count_sticks`).
        """
        from ktir_cpu.ops.memory_ops import _MemAccessor
        from unittest.mock import MagicMock

        def fake_read(byte_addr, n, dtype):
            base = byte_addr // 4
            return np.array(
                [10 * (base + i) for i in range(n)], dtype=np.int32,
            )

        lx = MagicMock()
        lx.read = MagicMock(side_effect=fake_read)
        ctx = MagicMock()
        ctx.get_lx = MagicMock(return_value=lx)
        accessor = _MemAccessor(ctx, "LX", byte_addr=0, lx_core_id=0)

        # Two runs: [0, 4, 8] (one run of 3) and [16] (one run of 1).
        values, sticks = accessor.read_scattered([0, 4, 8, 16], "i32")
        assert list(values) == [0, 10, 20, 40]
        assert sticks is None
        assert lx.read.call_args_list == [
            ((0, 3, "i32"),),
            ((16, 1, "i32"),),
        ]

    # ---------------------------------------------------------------------
    # _data_size — gated raise enforces the IAT-load contract.
    # ---------------------------------------------------------------------

    def test_data_size_raises_when_iat_load_result_missing_index_unique_sticks(self):
        """_data_size raises when an IAT-load result Tile arrives with index_unique_sticks=None.

        The contract: a handler that produces a Tile from an IAT operand
        must populate ``index_unique_sticks`` (``0`` for an all-LX IAT,
        positive for HBM). A surviving ``None`` indicates the handler
        skipped ``_resolve_idx_reads``, which would silently undercount.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef, Tile
        from ktir_cpu.latency import LatencyTracker
        from ktir_cpu.parser_ast import parse_affine_set

        idx_view = MemRef(
            base_ptr=0, shape=(4,), strides=[1],
            memory_space="HBM", dtype="i32",
        )
        iat = IndirectAccessTile(
            parent_ref=idx_view, shape=(4,),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            variables_space_set=parse_affine_set(
                "(d0) : (d0 >= 0, -d0 + 3 >= 0)"
            ),
            variables_space_order=None,
        )
        # Result Tile with index_unique_sticks left at default None —
        # mimics a buggy handler that skipped _resolve_idx_reads.
        result = Tile(
            np.zeros(4, dtype=np.float16), "f16", (4,), unique_sticks=1,
        )

        with pytest.raises(
            RuntimeError, match="must populate index_unique_sticks"
        ):
            LatencyTracker._data_size(result, [iat])

    def test_resolve_idx_reads_zero_extent_skips_view(self, monkeypatch):
        """Zero-extent IAT enumeration: ``_resolve_idx_reads`` returns
        ``({}, 0)`` rather than calling ``read_scattered([])``.

        A zero-extent dim is a legitimate degenerate case (the
        enumeration yields no points, hence no addresses to resolve).
        The view is skipped; ``_build_indirect_coords`` iterates the
        same enumeration and likewise produces no coords, so the
        missing key is never consumed.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef
        from ktir_cpu.ops.memory_ops import _resolve_idx_reads
        from ktir_cpu.parser_ast import parse_affine_set_raw

        idx_view = MemRef(
            base_ptr=0, shape=(4,), strides=[1],
            memory_space="HBM", dtype="i32",
        )
        iat = IndirectAccessTile(
            parent_ref=idx_view, shape=(4,),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            # Raw AffineSet (not a BoxSet) so enumerate goes through the
            # module-level enumerate_affine_set we stub below.
            variables_space_set=parse_affine_set_raw(
                "(d0) : (d0 >= 0, -d0 + 3 >= 0)"
            ),
            variables_space_order=None,
        )
        # Stub the enumerator to an empty list — the legitimate
        # zero-extent case. context is never accessed: ``continue`` fires
        # before any ``_MemAccessor`` is constructed.
        import ktir_cpu.parser_ast as parser_ast
        monkeypatch.setattr(
            parser_ast, "enumerate_affine_set", lambda *a, **kw: []
        )
        per_view_values, total_sticks = _resolve_idx_reads(None, iat)
        assert per_view_values == {}
        assert total_sticks == 0

    def test_data_size_raises_when_tile_result_missing_unique_sticks(self):
        """Tile result with ``unique_sticks=None`` raises.

        ``_data_size`` is reached only on the HBM path (LX short-circuits
        before it). On HBM, load handlers must populate ``unique_sticks``;
        a None here is a handler bug.
        """
        from ktir_cpu.ir_types import Tile
        from ktir_cpu.latency import LatencyTracker

        result = Tile(
            np.zeros(4, dtype=np.float16), "f16", (4,), unique_sticks=None,
        )

        with pytest.raises(
            RuntimeError, match="must populate unique_sticks"
        ):
            LatencyTracker._data_size(result, [])

    # ---------------------------------------------------------------------
    # Store sideband — _data_size charges HBM bytes from the int sideband
    # returned by MemoryOps.{store, indirect_store, distributed_store}.
    # The handler propagates that int as the op result; loads carry
    # stick counts on the result Tile (guard symmetry).
    # ---------------------------------------------------------------------

    def test_data_size_int_sideband_charges_stick_bytes(self):
        """``_data_size`` returns ``result * STICK_BYTES`` for an int result.

        The sideband int (from ``MemoryOps.indirect_store`` for IATs, or
        ``MemoryOps.store`` for direct stores) already aggregates data
        sticks (destination) and idx sticks (IAT). Operands are
        therefore ignored on the int branch.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef, Tile
        from ktir_cpu.latency import LatencyTracker
        from ktir_cpu.parser_ast import parse_affine_set

        idx_view = MemRef(
            base_ptr=0, shape=(128,), strides=[32],
            memory_space="HBM", dtype="i32",
        )
        iat = IndirectAccessTile(
            parent_ref=idx_view, shape=(4,),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            variables_space_set=parse_affine_set(
                "(d0) : (d0 >= 0, -d0 + 3 >= 0)"
            ),
            variables_space_order=None,
        )
        src = Tile(np.zeros(4, dtype=np.float16), "f16", (4,))

        # result=4 (sideband: total unique sticks for this store);
        # operands=[iat, src] are ignored on the int branch.
        assert LatencyTracker._data_size(4, [iat, src]) == 4 * 128

    def test_data_size_int_sideband_direct_store_64x64_scatter(self):
        """Direct store cost is stick-granular, not source-tile bytes.

        For a 64×64 f16 tile (8192 logical bytes) scattered to 100
        distinct sticks, HBM traffic is ``100 * 128 = 12800`` bytes —
        HBM is stick-addressed, so a partial-stick write still costs
        the full 128 bytes. The sideband int (``unique_sticks=100``)
        carries the stick count; ``_data_size`` returns ``100 * 128``,
        which differs from ``64 * 64 * 2`` by 4608 bytes.
        """
        from ktir_cpu.latency import LatencyTracker

        assert LatencyTracker._data_size(100, []) == 100 * 128
        # Stick-granular cost differs from logical-bytes by a non-trivial margin.
        assert 100 * 128 != 64 * 64 * 2

    def test_data_size_rejects_tile_operand_with_none_result(self):
        """Tile operand with None result raises.

        Store handlers must propagate ``MemoryOps.store``'s int return
        as the op result; a None result with a Tile operand violates
        the contract and raises.
        """
        from ktir_cpu.ir_types import Tile
        from ktir_cpu.latency import LatencyTracker

        src = Tile(np.zeros(4, dtype=np.float16), "f16", (4,))

        with pytest.raises(RuntimeError, match="propagate MemoryOps.store"):
            LatencyTracker._data_size(None, [src])

    def test_data_size_rejects_iat_operand_with_none_result(self):
        """IAT operand with None result raises.

        Indirect-store handlers must propagate
        ``MemoryOps.indirect_store``'s int return as the op result; a
        None result with an IAT operand violates the contract and raises.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef
        from ktir_cpu.latency import LatencyTracker
        from ktir_cpu.parser_ast import parse_affine_set

        idx_view = MemRef(
            base_ptr=0, shape=(4,), strides=[1],
            memory_space="HBM", dtype="i32",
        )
        iat = IndirectAccessTile(
            parent_ref=idx_view, shape=(4,),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            variables_space_set=parse_affine_set(
                "(d0) : (d0 >= 0, -d0 + 3 >= 0)"
            ),
            variables_space_order=None,
        )

        with pytest.raises(RuntimeError, match="without int sideband"):
            LatencyTracker._data_size(None, [iat])

    # ---------------------------------------------------------------------
    # LX stores: the sideband int is 0 (no HBM stick concept), and
    # _data_size charges 0 bytes. These guard against the rejects-tests
    # above accidentally over-firing on legitimate LX paths — store
    # handlers must always return an int, never None.
    # ---------------------------------------------------------------------

    def test_store_returns_zero_for_lx_destination(self):
        """``MemoryOps.store`` to an LX tile returns ``0``, not ``None``.

        LX has no stick concept, so HBM stick traffic is 0 by definition.
        The handler propagates that 0 as the op result; ``_data_size``'s
        int branch fires with ``0 * STICK_BYTES = 0``. Returning ``None``
        instead would trip ``test_data_size_rejects_tile_operand_with_none_result``
        on every LX store path.
        """
        from ktir_cpu.grid import CoreContext
        from ktir_cpu.ir_types import MemRef, Tile
        from ktir_cpu.memory import HBMSimulator, LXScratchpad
        from ktir_cpu.ops.memory_ops import MemoryOps

        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                          lx=LXScratchpad(core_id=0), hbm=HBMSimulator())
        tile_ref = MemRef(
            base_ptr=0, shape=(4,), strides=[1],
            memory_space="LX", dtype="f16",
        ).to_tile_ref()
        src = Tile(np.arange(4, dtype=np.float16), "f16", (4,))

        result = MemoryOps.store(ctx, src, tile_ref)
        assert result == 0
        assert isinstance(result, int)  # not None

    def test_indirect_store_returns_zero_for_all_lx(self):
        """``MemoryOps.indirect_store`` returns ``0`` when parent + every
        idx view live in LX (no HBM traffic on either side).
        """
        from ktir_cpu.grid import CoreContext
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef, Tile
        from ktir_cpu.memory import HBMSimulator, LXScratchpad
        from ktir_cpu.ops.memory_ops import MemoryOps
        from ktir_cpu.parser_ast import parse_affine_set

        lx = LXScratchpad(core_id=0)
        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=HBMSimulator())

        parent_ptr = 0
        idx_ptr = 64  # past parent's 8 f16 = 16 bytes; safe non-overlap
        lx.write(parent_ptr, np.zeros(8, dtype=np.float16))  # seed for read-modify-write
        lx.write(idx_ptr, np.arange(4, dtype=np.int32))

        parent_ref = MemRef(
            base_ptr=parent_ptr, shape=(8,), strides=[1],
            memory_space="LX", dtype="f16",
        )
        idx_view = MemRef(
            base_ptr=idx_ptr, shape=(4,), strides=[1],
            memory_space="LX", dtype="i32",
        )
        iat = IndirectAccessTile(
            parent_ref=parent_ref, shape=(4,),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            variables_space_set=parse_affine_set(
                "(d0) : (d0 >= 0, -d0 + 3 >= 0)"
            ),
            variables_space_order=None,
        )
        src = Tile(np.arange(4, dtype=np.float16), "f16", (4,))

        result = MemoryOps.indirect_store(ctx, src, iat)
        assert result == 0
        assert isinstance(result, int)

    def test_distributed_store_returns_zero_for_all_lx(self):
        """``MemoryOps.distributed_store`` returns ``0`` when every
        surviving partition lives in LX. Mirrors the all-HBM aggregation
        path but with the LX sentinel (0 sticks per survivor).
        """
        from ktir_cpu.affine import BoxSet
        from ktir_cpu.grid import CoreContext
        from ktir_cpu.ir_types import (
            DistributedTileRef, MemRef, Tile, TileRef,
        )
        from ktir_cpu.memory import HBMSimulator, LXScratchpad
        from ktir_cpu.ops.memory_ops import MemoryOps

        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                          lx=LXScratchpad(core_id=0), hbm=HBMSimulator())

        ptr = 0
        memref = MemRef(
            base_ptr=ptr, shape=(8,), strides=[1],
            memory_space="LX", dtype="f16",
        )
        survivor = TileRef(
            base_ptr=ptr, shape=(8,), strides=[1], memref=memref, dtype="f16",
            coordinate_set=BoxSet(lo=(0,), hi=(8,)),
            partition_origin=(0,),
        )
        dist_ref = DistributedTileRef(
            partitions=[survivor], shape=(8,), dtype="f16", global_base=(0,),
        )
        src = Tile(np.arange(8, dtype=np.float16), "f16", (8,))

        result = MemoryOps.distributed_store(ctx, src, dist_ref)
        assert result == 0
        assert isinstance(result, int)

    # ---------------------------------------------------------------------
    # End-to-end: MemoryOps.indirect_load / .indirect_store actually stash
    # the count on the returned Tile / produce the matching count via the
    # IAT side-channel. These pin the wiring that connects the helpers to
    # the public ops — a plain unit test on _data_size cannot catch a
    # regression where indirect_load forgets to set the field.
    # ---------------------------------------------------------------------

    @staticmethod
    def _build_simple_gather_iat(hbm):
        """Allocate X / IDX1 / IDX2 in HBM and return an IAT for the gather.

        Layout matches RFC §5 Example 1 in miniature (8×8 instead of
        64×64 to keep the formula human-readable; the same per-stick
        math applies):

        * X:    8×8 f16 (128 bytes = 1 stick)
        * IDX1: 8×8 i32 (256 bytes = 2 sticks)
        * IDX2: 8×8 i32 (256 bytes = 2 sticks)

        Both index tensors are seeded to all-zeros, so every gather
        collapses to ``X[0, 0]`` — but the IAT enumeration still reads
        every index entry (8*8 = 64 reads per view), and those 64
        addresses span the full 256 bytes of each view → 2 sticks each.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef
        from ktir_cpu.parser_ast import parse_affine_set

        x_stick = hbm.allocate(8 * 8 * 2)         # 128 bytes (1 stick)
        idx1_stick = hbm.allocate(8 * 8 * 4)      # 256 bytes (2 sticks)
        idx2_stick = hbm.allocate(8 * 8 * 4)      # 256 bytes (2 sticks)

        hbm.write(x_stick, np.zeros(64, dtype=np.float16))
        hbm.write(idx1_stick, np.zeros(64, dtype=np.int32))
        hbm.write(idx2_stick, np.zeros(64, dtype=np.int32))

        parent = MemRef(base_ptr=x_stick, shape=(8, 8), strides=[8, 1],
                        memory_space="HBM", dtype="f16")
        idx1 = MemRef(base_ptr=idx1_stick, shape=(8, 8), strides=[8, 1],
                      memory_space="HBM", dtype="i32")
        idx2 = MemRef(base_ptr=idx2_stick, shape=(8, 8), strides=[8, 1],
                      memory_space="HBM", dtype="i32")

        iat = IndirectAccessTile(
            parent_ref=parent, shape=(8, 8),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0), ("dim", 1)]},
                {"kind": "indirect", "index_view_idx": 1,
                 "idx_exprs": [("dim", 0), ("dim", 1)]},
            ],
            index_views=[idx1, idx2],
            variables_space_set=parse_affine_set(
                "(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, "
                "d1 >= 0, -d1 + 7 >= 0)"
            ),
            variables_space_order=None,
        )
        return iat

    def test_indirect_load_index_sticks_simple_gather(self):
        """End-to-end: MemoryOps.indirect_load stamps index_unique_sticks on the result Tile.

        RFC §5 Example 1 pattern (2-D gather Y[m, k] = X[IDX1[m, k], IDX2[m, k]]).
        Per-stick formula::

            addrs per view = 8 * 8 = 64 (one per enumerated (m, k) pt)
            byte span      = 64 * 4 = 256 bytes (i32, contiguous)
            sticks per view= 256 / 128 = 2
            total          = 2 + 2 = 4

        If a future refactor drops ``result.index_unique_sticks =
        idx_unique_sticks`` from :meth:`MemoryOps.indirect_load`, this
        assertion catches it; helper-level tests would not.
        """
        from ktir_cpu.grid import CoreContext
        from ktir_cpu.memory import HBMSimulator, LXScratchpad
        from ktir_cpu.ops.memory_ops import MemoryOps

        hbm = HBMSimulator()
        lx = LXScratchpad(size_mb=2, core_id=0)
        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)

        iat = self._build_simple_gather_iat(hbm)

        result = MemoryOps.indirect_load(ctx, iat)

        assert result.index_unique_sticks == 4, (
            f"expected 4 idx sticks (2 per view × 2 views = 4), "
            f"got {result.index_unique_sticks}"
        )

    def test_indirect_store_index_sticks_mirrors_load(self):
        """Guard symmetry: indirect_store's idx-side count matches indirect_load.

        Both ops share ``_resolve_idx_reads`` for the runtime read.
        ``indirect_load`` stamps idx sticks on ``Tile.index_unique_sticks``;
        ``indirect_store`` returns the same count (plus data sticks) via
        the int sideband. Pinning load's idx total against the store's
        return minus its data sticks locks the symmetry — drift on
        either path trips this test.
        """
        from ktir_cpu.grid import CoreContext
        from ktir_cpu.ir_types import Tile
        from ktir_cpu.memory import HBMSimulator, LXScratchpad
        from ktir_cpu.ops.memory_ops import MemoryOps

        hbm = HBMSimulator()
        lx = LXScratchpad(size_mb=2, core_id=0)
        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)

        iat = self._build_simple_gather_iat(hbm)

        load_result = MemoryOps.indirect_load(ctx, iat)
        load_idx_sticks = load_result.index_unique_sticks
        load_data_sticks = load_result.unique_sticks

        # Store side: returned int = data_sticks + idx_sticks. Subtract
        # data_sticks (1 stick: 8×8 f16 = 128 bytes = 1 stick) to recover
        # the idx-side count and assert symmetry against the load.
        src_tile = Tile(np.zeros((8, 8), dtype=np.float16), "f16", (8, 8))
        store_total_sticks = MemoryOps.indirect_store(ctx, src_tile, iat)
        store_idx_sticks = store_total_sticks - load_data_sticks

        assert load_idx_sticks == store_idx_sticks == 4, (
            f"load idx sticks={load_idx_sticks}, "
            f"store idx sticks={store_idx_sticks}, expected both to be 4"
        )


# ---------------------------------------------------------------------------
# Ring reduce latency — comm-dominated path
# ---------------------------------------------------------------------------
# 🧪 Experimental — tracks ktir-mlir-frontend#23.  Validates the
# inter-tile + scheduler + latency pipeline: ring bytes accumulated by
# RingReduceBackend land on Tile.comm_bytes and flow into the per-core
# LatencyReport.  Correctness counterpart:
# tests/test_examples.py::TestRingReduceExecution.

class TestRingReduceLatency:
    """Per-core latency report from the 4-core ring reduce in
    ``examples/ktir/ring_reduce.mlir``.

    Scope of validity
    -----------------
    This test pins ``comm_cycles = ring_bytes / ring_bytes_per_cycle``
    for the kernel as written.  That formula is exact **only** under two
    conditions, both of which hold here:

    1. **Single group, contiguous cores.**  The kernel uses
       ``producer_tiles_per_group = (i)[g] : (i - 4*g >= 0,
       -i + 4*g + 3 >= 0)`` with ``groups = (g) : (g == 0)``, so the
       producer set for the single group is ``{0, 1, 2, 3}`` — exactly
       the four physical cores in core-id order.  ``RingReduceBackend``
       lays the ring out in that order, so ``send_to(next_core, ...)``
       is always a 1-hop transfer to the *immediate* neighbor:
       0 → 1 → 2 → 3 → 0.

    2. **No per-hop overhead in the latency model today.**  ``LC.COMM``
       in ``latency.py`` charges ``bytes / ring_bytes_per_cycle`` and
       nothing else.  No constant per-hop latency, no multi-hop scaling.

    For any pattern where the producer set's logical order does **not**
    match physical wire order — e.g. multi-group with strided membership
    like ``{g, g+8, g+16, g+24}``, butterfly mirror, or one-to-many
    broadcast — the model would underestimate and this test's cycle
    assertion would no longer reflect the real cost.  The structural
    assertions (per-core counts, category buckets, store-only-on-core-0)
    remain valid regardless.  A multi-group latency test will need to
    pin per-hop costs once that part of the model is added.

    Ring layout — 4 cores, 3 rounds
    -------------------------------
    Each arrow ``→`` is one ``send_to`` of one ``1x128 f16`` payload
    (256 bytes) to the *immediate* core-id neighbor.  The wire wraps
    3 → 0.  Each core does ``N - 1 = 3`` rounds.  After round 3 every
    core's accumulator has folded in every starting tile exactly once,
    so all four cores hold ``t0 + t1 + t2 + t3``.

    Each cell shows ``acc | fwd``: the accumulator value on that core
    *after* the round's reduce, and the tile that core will *forward*
    in the next round (= the tile it just received this round).
    ``send`` annotations on the arrows show the round-1 payloads.

                        round 1               round 2          round 3

      core 0  start:t0  ┌── send t0 ─→┐    ┌── send t3 ─→┐    ┌── send t2 ─→┐
                        │ recv t3      │    │ recv t2      │    │ recv t1      │
                        │ acc=t0+t3    │    │ acc=t0+t3+t2 │    │ acc=Σt       │
                        │ fwd=t3       │    │ fwd=t2       │    │              │
                        └──────────────┘    └──────────────┘    └──────────────┘

      core 1  start:t1  ┌── send t1 ─→┐    ┌── send t0 ─→┐    ┌── send t3 ─→┐
                        │ recv t0      │    │ recv t3      │    │ recv t2      │
                        │ acc=t1+t0    │    │ acc=t1+t0+t3 │    │ acc=Σt       │
                        │ fwd=t0       │    │ fwd=t3       │    │              │
                        └──────────────┘    └──────────────┘    └──────────────┘

      core 2  start:t2  ┌── send t2 ─→┐    ┌── send t1 ─→┐    ┌── send t0 ─→┐
                        │ recv t1      │    │ recv t0      │    │ recv t3      │
                        │ acc=t2+t1    │    │ acc=t2+t1+t0 │    │ acc=Σt       │
                        │ fwd=t1       │    │ fwd=t0       │    │              │
                        └──────────────┘    └──────────────┘    └──────────────┘

      core 3  start:t3  ┌── send t3 ─→┐    ┌── send t2 ─→┐    ┌── send t1 ─→┐
                        │ recv t2      │    │ recv t1      │    │ recv t0      │
                        │ acc=t3+t2    │    │ acc=t3+t2+t1 │    │ acc=Σt       │
                        │ fwd=t2       │    │ fwd=t1       │    │              │
                        └──────────────┘    └──────────────┘    └──────────────┘

    Wire usage per core: 3 sends × 256 bytes = 768 bytes/core, all over
    1-hop neighbor edges.  That's the byte total ``RingReduceBackend``
    accumulates into ``self.bytes_moved`` and the consume handler stamps
    onto ``Tile.comm_bytes``.

    Per-core trace (each core, in order)
    ------------------------------------
    With ``HardwareConfig()`` defaults — ``simd_elements_per_cycle = 64``,
    ``hbm_bandwidth_tb_s = 1.0``, ``ring_bandwidth_tb_s = 4.0``,
    ``num_cores = 32`` — and a ``1 x 128 f16`` partial (256 bytes,
    occupying 2 HBM sticks)::

      arith.constant                  zero    cycles=0.0
      arith.constant                  zero    cycles=0.0
      ktdp.get_compute_tile_id        zero    cycles=0.0
      arith.muli                      compute cycles=0.0   (scalar index)
      arith.addi                      compute cycles=0.0   (scalar index)
      ktdp.construct_memory_view      zero    cycles=0.0
      ktdp.construct_access_tile      zero    cycles=0.0
      ktdp.load                       memory  cycles=8.192 (256 bytes / 31.25 B/cyc)
      ktdp.yield_partial              zero    cycles=0.0
      ktdp.inter_tile_produce         zero    cycles=0.0   (no LC.COMM)
      arith.constant                  zero    cycles=0.0
      tensor.empty                    zero    cycles=0.0
      linalg.fill                     zero    cycles=0.0
      tensor.empty                    zero    cycles=0.0
      linalg.add                      compute cycles=2.0   (128 elems / 64 simd)
      ktdp.yield_reduced              zero    cycles=0.0
      tensor.empty                    zero    cycles=0.0
      linalg.add                      compute cycles=2.0   <- combiner round 2
      ktdp.yield_reduced              zero    cycles=0.0
      tensor.empty                    zero    cycles=0.0
      linalg.add                      compute cycles=2.0   <- combiner round 3
      ktdp.yield_reduced              zero    cycles=0.0
      ktdp.inter_tile_reduce          comm    cycles=0.192 (768 bytes / 4000 B/cyc)
      arith.cmpi                      compute cycles=0.015625 (1 elem / 64 simd)
      [core 0 only:  tensor.expand_shape  zero    cycles=0.0]
      [core 0 only:  ktdp.construct_memory_view  zero    cycles=0.0]
      [core 0 only:  ktdp.construct_access_tile  zero    cycles=0.0]
      [core 0 only:  ktdp.store              memory  cycles=8.192 (256 bytes)]
      scf.if                          zero    cycles=0.0
      return                          zero    cycles=0.0

    The ring reduce contributes ``(N_ring - 1) * partial_bytes = 3 * 256
    = 768`` bytes per core.  The combiner region (``linalg.add``) runs
    ``N_ring - 1 = 3`` times on each core, once per ring round (the
    backend's ``reduce_fn`` invocation; see
    ``RingReduceBackend.run`` in ``ktir_cpu/ops/comm_ops.py``).

    Per-core totals
    ---------------
    - All cores: ``compute = 2.0 * 3 + 0.015625 = 6.015625`` cycles,
      ``comm = 0.192`` cycles.
    - Cores 1, 2, 3: ``memory = 8.192`` cycles (one ``ktdp.load``).
    - Core 0:       ``memory = 16.384`` cycles (``ktdp.load`` +
      ``ktdp.store``).
    - ``rep.kernel_cycles = max(total_cycles)`` is on core 0, since it
      carries the extra HBM store.
    """

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("ring_reduce"))
    def test_ring_reduce_per_core_breakdown(self, path, func_name, entry):
        cfg = HardwareConfig()
        rep, rows, n_cols = _run_ring_reduce(
            path, func_name, entry, cfg, trace=True
        )

        assert rep is not None, (
            "latency report should be present when latency_config is set"
        )

        N_ring = rows.shape[0]                  # 4 producers per group
        partial_bytes = n_cols * 2              # 1 x 128 f16 = 256 bytes
        per_core_ring_bytes = (N_ring - 1) * partial_bytes
        expected_comm_cycles = per_core_ring_bytes / cfg.ring_bytes_per_cycle

        # Every core should have a counters record.
        assert set(rep.counters.keys()) == set(range(N_ring))

        # ---- Per-core invariants ----
        for core_id, cc in rep.counters.items():
            ops = [t.op_type for t in cc.trace]

            # Each core runs the produce + reduce pair exactly once.
            assert ops.count("ktdp.inter_tile_produce") == 1
            assert ops.count("ktdp.inter_tile_reduce") == 1

            # Produce is a metadata-only op now (no LC.COMM): cycles=0.
            produce_entry = next(
                t for t in cc.trace if t.op_type == "ktdp.inter_tile_produce"
            )
            assert produce_entry.cycles == 0
            assert produce_entry.category == "zero"

            # Reduce is the comm-charged op.  Bytes stamped onto
            # Tile.comm_bytes by RingReduceBackend flow into the
            # tracker via _comm_size; cycles = bytes /
            # ring_bytes_per_cycle, with no log2(num_cores) multiplier
            # (dropped in the per-message accounting redesign).
            reduce_entry = next(
                t for t in cc.trace if t.op_type == "ktdp.inter_tile_reduce"
            )
            assert reduce_entry.category == "comm"
            assert reduce_entry.cycles == pytest.approx(expected_comm_cycles)

            # Total per-core comm equals exactly the reduce charge —
            # no other op in the trace contributes to the comm bucket.
            assert cc.comm_cycles == pytest.approx(expected_comm_cycles)

            # Cumulative comm bytes for this core (the only comm op is
            # the reduce, so total_bytes - HBM-side load/store bytes
            # equals the ring's per-core wire load).  We pin the comm
            # bytes via the cycle assertion above; this asserts the
            # aggregate counter agrees: ring bytes contribute exactly
            # ``per_core_ring_bytes`` to ``total_bytes``.
            # Note ``total_bytes`` also includes HBM bytes from
            # ``ktdp.load`` / ``ktdp.store``, so we can't pin it on its
            # own — we just assert it's at least the ring bytes.
            assert cc.total_bytes >= per_core_ring_bytes

            # Each core loads its own row from HBM exactly once.
            assert ops.count("ktdp.load") == 1

        # ---- Cross-core invariants ----
        # Only core 0 writes back (gated by ``scf.if %is_writer``).
        store_cores = [
            cid for cid, cc in rep.counters.items()
            if any(t.op_type == "ktdp.store" for t in cc.trace)
        ]
        assert store_cores == [0], (
            f"expected only core 0 to ktdp.store, got {store_cores}"
        )

        # Kernel wall-clock = max of per-core total cycles.  Core 0
        # carries the extra HBM store, so it must be the heaviest.
        per_core_total = {cid: cc.total_cycles for cid, cc in rep.counters.items()}
        assert rep.kernel_cycles == max(per_core_total.values())
        assert per_core_total[0] >= max(per_core_total[c] for c in (1, 2, 3))


# ---------------------------------------------------------------------------
# Multi-group ring reduce — comm-dominated, 4 groups of 4 on a 16-core grid
# ---------------------------------------------------------------------------
# 🧪 Experimental — tracks ktir-mlir-frontend#23.  Validates two
# things the single-group test cannot:
#   1. Correctness of the plan-aware fold in ``RingReduceBackend.run``
#      — every group's accumulator must contain only its own four
#      tiles, even though the ring spans the whole 16-core
#      workgroup.  Out-of-plan tiles flow through but are discarded
#      at fold time; misbehaviour here would smear values across
#      groups.
#   2. Per-core ring bytes scale with ``ctx.num_cores``, not group
#      size — the ring is over the whole workgroup, so a 16-core
#      workgroup with 4 groups of 4 sends ``num_cores - 1 = 15``
#      messages per core, not ``group_size - 1 = 3``.

class TestRingReduceMultiGroupLatency:
    """4 concurrent groups of 4 cores on a 16-core workgroup.

    Hardware-config defaults: ``ring_bandwidth_tb_s = 4.0``,
    ``ring_bytes_per_cycle = 4000``.  Workgroup geometry:

    - 16 cores, 4 groups of 4: group ``g`` is ``{4g, 4g+1, 4g+2, 4g+3}``.
    - Each core holds a ``tensor<1x128xf16>`` partial (256 bytes).
    - Per-core ring traffic = ``(num_cores - 1) * partial_bytes`` =
      ``15 * 256 = 3840`` bytes.
    - Per-core comm cycles = ``3840 / 4000 = 0.96``.

    Note the difference from the single-group test: the ring's size
    is ``num_cores - 1``, *not* ``group_size - 1``.  Multi-group does
    not shrink the per-op ring; it just narrows the fold via the
    plan.
    """

    @pytest.mark.parametrize(
        "path,func_name,entry",
        get_test_params("ring_reduce_multi_group"),
    )
    def test_multi_group_correctness_and_bytes(self, path, func_name, entry):
        cfg = HardwareConfig()
        rep, rows, n_cols, n_groups, group_size, output = (
            _run_ring_reduce_multi_group(path, func_name, entry, cfg, trace=True)
        )

        num_cores = n_groups * group_size
        partial_bytes = n_cols * 2          # 1 x 128 f16 = 256 bytes
        per_core_ring_bytes = (num_cores - 1) * partial_bytes
        expected_comm_cycles = per_core_ring_bytes / cfg.ring_bytes_per_cycle

        # ---- Correctness: each group's writer holds the in-group sum ----
        # ``output[g]`` was written by core ``4*g``; expected value is
        # the sum of rows[4g..4g+3].  If the plan-aware fold slipped
        # and folded out-of-group tiles, this assertion would catch it.
        for g in range(n_groups):
            expected = rows[g * group_size:(g + 1) * group_size].sum(axis=0)
            np.testing.assert_allclose(
                output[g], expected, rtol=1e-2,
                err_msg=f"group {g}: output does not match in-group sum",
            )

        # ---- Latency: every core's comm bucket = (num_cores-1) * partial_bytes ----
        # Includes non-writer cores (most of the 16) — they all run the
        # ring; only the comm bucket varies, not store traffic.
        assert set(rep.counters.keys()) == set(range(num_cores))
        for core_id, cc in rep.counters.items():
            assert cc.comm_cycles == pytest.approx(expected_comm_cycles), (
                f"core {core_id}: comm_cycles {cc.comm_cycles} "
                f"!= expected {expected_comm_cycles}"
            )

            ops = [t.op_type for t in cc.trace]
            assert ops.count("ktdp.inter_tile_produce") == 1
            assert ops.count("ktdp.inter_tile_reduce") == 1

            reduce_entry = next(
                t for t in cc.trace if t.op_type == "ktdp.inter_tile_reduce"
            )
            assert reduce_entry.category == "comm"
            assert reduce_entry.cycles == pytest.approx(expected_comm_cycles)

        # ---- Cross-core invariants: only group writers store ----
        store_cores = sorted(
            cid for cid, cc in rep.counters.items()
            if any(t.op_type == "ktdp.store" for t in cc.trace)
        )
        expected_writers = [g * group_size for g in range(n_groups)]
        assert store_cores == expected_writers, (
            f"expected writers {expected_writers}, got {store_cores}"
        )
