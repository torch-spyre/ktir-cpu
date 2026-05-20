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
                          if e.op_type == "arith.addf" and e.category == "compute")
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
        assert "achieved_gflops" in rf
        assert "peak_gflops" in rf
        assert "peak_bw_gb_s" in rf
        assert "ridge_point" in rf
        assert "ceiling_gflops" in rf
        assert "efficiency" in rf

        assert 0 < rf["efficiency"] <= 1.0
        assert rf["achieved_gflops"] <= rf["ceiling_gflops"]
        assert rf["ceiling_gflops"] <= rf["peak_gflops"]

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_vector_add_is_memory_bound(self, path, func_name, entry):
        """vector_add has low arithmetic intensity → memory-bound on roofline."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        rf = report.roofline()

        # AI should be well below the ridge point (memory-bound)
        assert rf["arithmetic_intensity"] < rf["ridge_point"]

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
        """When bottleneck is memory, roofline AI should be below ridge point."""
        report, _ = _run_vector_add(path, func_name, entry, HardwareConfig())
        assert report.bottleneck == "memory"
        rf = report.roofline()
        assert rf["arithmetic_intensity"] < rf["ridge_point"]

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel_small"))
    def test_compute_bound_roofline_matches_bottleneck(self, path, func_name, entry):
        """When bottleneck is compute, roofline AI should be above ridge point."""
        # Single core + high HBM BW → compute-dominated
        cfg = HardwareConfig(num_cores=1, hbm_bandwidth_tb_s=100.0)
        report = _run_softmax(path, func_name, entry, cfg)
        assert report.bottleneck == "compute"
        rf = report.roofline()
        assert rf["arithmetic_intensity"] > rf["ridge_point"]


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
            if e.category == "compute":
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
                          if e.op_type == "arith.addf" and e.category == "compute")
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

        # A zero-element tile (e.g. empty slice)
        zero_tile = Tile(np.array([], dtype=np.float16), "f16", (0,))
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
        result = Tile(
            np.zeros((4, 4), dtype=np.float16), "f16", (4, 4),
            index_unique_sticks=0,
        )

        assert LatencyTracker._data_size(result, [iat]) == result.data.nbytes
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

    def test_idx_unique_sticks_no_reads_dedups_repeated_coords(self):
        """_idx_unique_sticks_no_reads dedups across vss-enumerated points.

        Models a paged-attention-shape access pattern in miniature: a
        2-D variable space ``(d0, d1)`` where ``idx_exprs = (d0,)``
        projects out ``d1`` entirely.  Each idx coord (d0=0, d0=1) is
        looked up |range(d1)| = 4 times — but they live on the same
        stick (i32 idx, base_ptr=0, addresses 0 and 4 both fall in
        stick 0 of a 128-byte stick).

        Expected: 1 unique stick (despite 8 enumerated points).
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef
        from ktir_cpu.ops.memory_ops import _idx_unique_sticks_no_reads
        from ktir_cpu.parser_ast import parse_affine_set

        idx_view = MemRef(
            base_ptr=0, shape=(64,), strides=[1],
            memory_space="HBM", dtype="i32",  # bpe=4
        )
        iat = IndirectAccessTile(
            parent_ref=idx_view, shape=(2, 4),
            dim_subscripts=[
                {"kind": "indirect", "index_view_idx": 0,
                 "idx_exprs": [("dim", 0)]},
            ],
            index_views=[idx_view],
            variables_space_set=parse_affine_set(
                "(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)"
            ),
            variables_space_order=None,
        )

        assert _idx_unique_sticks_no_reads(iat) == 1

    def test_idx_unique_sticks_no_reads_n_distinct_sticks(self):
        """_idx_unique_sticks_no_reads counts N sticks when each idx coord
        owns its own stick.

        ``idx_view`` is i32 with one element per stick (stride forces each
        coord onto a distinct stick).  Enumerating 4 distinct d0 values
        therefore gives 4 unique sticks.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef
        from ktir_cpu.ops.memory_ops import _idx_unique_sticks_no_reads
        from ktir_cpu.parser_ast import parse_affine_set

        # i32 with stride 32 means each idx element is 32 * 4 = 128 bytes
        # apart — one stick per element.
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

        assert _idx_unique_sticks_no_reads(iat) == 4

    def test_idx_unique_sticks_no_reads_returns_zero_for_lx_only_iat(self):
        """LX has no stick concept — return 0 (defined "no HBM traffic"), not None.

        Reserving ``None`` for "not computed" lets the gated raise in
        ``_data_size`` catch handlers that forget to populate the field.
        An LX-only IAT load is a fully legitimate workflow that must
        report zero HBM idx-side traffic without tripping the guard.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef
        from ktir_cpu.ops.memory_ops import _idx_unique_sticks_no_reads
        from ktir_cpu.parser_ast import parse_affine_set

        idx_view = MemRef(
            base_ptr=0, shape=(64,), strides=[1],
            memory_space="LX", dtype="i32",
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

        assert _idx_unique_sticks_no_reads(iat) == 0

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

        Verifies the per-element semantics fabian's review required:
        stick count is set-deduped on the address side regardless of how
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

    # ---------------------------------------------------------------------
    # Guard symmetry — indirect_store charges idx sticks via _data_size's
    # IAT-operand fallback, mirroring the indirect_load load-side charge.
    # ---------------------------------------------------------------------

    def test_data_size_indirect_store_charges_idx_sticks(self):
        """Store path: _data_size recomputes idx sticks from the IAT (no result Tile).

        ``ktdp.store`` with an indirect access tile has ``result=None``,
        so the IAT-operand branch in :meth:`LatencyTracker._data_size`
        falls through to :func:`_idx_unique_sticks_no_reads` rather
        than reading ``result.index_unique_sticks`` (which doesn't
        exist). Mirrors the load-side charge for guard symmetry.

        The IAT here enumerates 4 distinct d0 values; idx_view stride
        forces each onto its own stick → 4 unique sticks → 4 * 128 bytes.
        """
        from ktir_cpu.ir_types import IndirectAccessTile, MemRef, Tile
        from ktir_cpu.latency import LatencyTracker
        from ktir_cpu.parser_ast import parse_affine_set

        # i32 stride 32 → each idx element is 32 * 4 = 128 bytes apart,
        # one stick per element.
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
        # Store source tile lands as the second branch's Tile operand;
        # we only care that the IAT operand branch contributes its idx
        # sticks. The store source contributes its data.nbytes.
        src = Tile(np.zeros(4, dtype=np.float16), "f16", (4,))

        # result=None (store), operands=[iat, src]:
        #   IAT branch (store path) → 4 sticks * 128 = 512
        #   Tile branch (src)       → src.data.nbytes = 8
        assert LatencyTracker._data_size(None, [iat, src]) == 4 * 128 + 8

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
        """Guard symmetry: indirect_store's idx-side stick count matches indirect_load.

        Both ops share ``_resolve_idx_reads`` for the runtime read and
        delegate stick counting to ``_MemAccessor.count_sticks``. The
        latency model recomputes the store-side count via
        ``_idx_unique_sticks_no_reads(iat)`` (since stores have no
        result Tile). Pinning the load value and the no-read value
        equal locks the symmetry — if either path drifts, this trips.
        """
        from ktir_cpu.grid import CoreContext
        from ktir_cpu.ir_types import Tile
        from ktir_cpu.memory import HBMSimulator, LXScratchpad
        from ktir_cpu.ops.memory_ops import MemoryOps, _idx_unique_sticks_no_reads

        hbm = HBMSimulator()
        lx = LXScratchpad(size_mb=2, core_id=0)
        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)

        iat = self._build_simple_gather_iat(hbm)

        load_result = MemoryOps.indirect_load(ctx, iat)
        load_sticks = load_result.index_unique_sticks

        # Store side: _data_size goes through _idx_unique_sticks_no_reads,
        # which is the no-IO mirror of _resolve_idx_reads' stick count.
        store_sticks = _idx_unique_sticks_no_reads(iat)

        assert load_sticks == store_sticks == 4, (
            f"load idx sticks={load_sticks}, store idx sticks={store_sticks}, "
            f"expected both to be 4"
        )

        # Smoke-test the actual store call (does it execute, does it
        # leave HBM in a consistent state). Source is i32 zeros sized
        # to match the IAT shape; we already seeded X to zeros so the
        # store is effectively a no-op write.
        src_tile = Tile(np.zeros((8, 8), dtype=np.float16), "f16", (8, 8))
        MemoryOps.indirect_store(ctx, src_tile, iat)
