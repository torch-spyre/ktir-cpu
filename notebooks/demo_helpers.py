"""Demo helper functions for the KTIR latency estimation notebook.

Print helpers wrap verbose formatted output.
MLIR generators live in demo_gen_mlir.py and are re-exported here.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ktir_cpu.interpreter import KTIRInterpreter
from ktir_cpu.latency import HardwareConfig

from demo_gen_mlir import (  # noqa: F401 — re-exported for notebook compat
    gen_matmul_mlir,
    gen_softmax_mlir,
    gen_sdpa_mlir,
    gen_paged_attention_mlir,
)


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def run_kernel(hw, mlir_text, func_name, tensor_kwargs, scalar_kwargs=None):
    """Load MLIR, execute a kernel, return the LatencyReport."""
    if scalar_kwargs is None:
        scalar_kwargs = {}
    interp = KTIRInterpreter(latency_config=hw)
    interp.load(mlir_text)
    interp.execute_function(func_name, **tensor_kwargs, **scalar_kwargs)
    return interp.get_latency_report()


# ---------------------------------------------------------------------------
# MLIR viewer
# ---------------------------------------------------------------------------

def show_mlir(mlir_text, max_lines=40):
    """Print generated MLIR text, optionally truncated."""
    lines = mlir_text.splitlines()
    if max_lines and len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(line)
        print(f"  ... ({len(lines) - max_lines} more lines, {len(lines)} total)")
    else:
        print(mlir_text)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_hw_config(hw: HardwareConfig):
    """Print HardwareConfig summary with ridge points."""
    print("HardwareConfig")
    print(f"  num_cores                : {hw.num_cores}")
    print(f"  clock                    : {hw.clock_ghz} GHz")
    print(f"  LX scratchpad            : {hw.lx_size_mb} MiB / core")
    print(f"  HBM bandwidth (chip)     : {hw.hbm_bandwidth_tb_s:.3f} TB/s = {hw.hbm_bw_chip/hw.clock_hz:.0f} B/cycle")
    print(f"  Ring bandwidth           : {hw.ring_bandwidth_tb_s:.3f} TB/s = {hw.ring_bytes_per_cycle:.0f} B/cycle")
    print(f"  Systolic array (1 core)  : {hw.systolic_rows} rows × {hw.simd_elements_per_cycle} cols × 2 (FMA) = {hw.systolic_flops_per_cycle} F/cycle")
    print(f"  SIMD (1 core)            : {hw.simd_elements_per_cycle} F/cycle")
    print(f"  Transcendental penalty   : {hw.transcendental_penalty}×")
    chip_bw_per_cycle = hw.hbm_bw_chip / hw.clock_hz
    print(f"  Per-core HBM BW          : chip_bw / cores_active  (e.g. {chip_bw_per_cycle:.0f}/4 = {chip_bw_per_cycle/4:.0f} B/cycle)")
    print(f"\nChip-wise roofline ridge (= peak_F_per_cycle / per_core_BW):")
    print(f"  {'cores_active':<14} {'systolic':>10} {'SIMD':>10}")
    for nc in [4, 32]:
        bw = chip_bw_per_cycle / nc
        print(f"  {nc:<14} {hw.systolic_flops_per_cycle / bw:>7.0f} F/B {hw.simd_elements_per_cycle / bw:>7.0f} F/B")


def print_core_roofline(report, hw: HardwareConfig):
    """Print per-core roofline detail for the critical-path core."""
    print(f"Kernel cycles : {report.kernel_cycles:,.0f}")
    print(f"Kernel time   : {report.kernel_time_us:.3f} us")
    print(f"Bottleneck    : {report.bottleneck}")

    core = report.core_roofline()
    dom = core["core_dominant_unit"]
    u = core["units"][dom]
    print(f"\nPer-core roofline  (critical-path core; dominant unit: {dom})")
    print(f"  core_AI         : {core['core_AI']:.2f} FLOP/B")
    cores_active = core.get("cores_active", len(report.per_core_summary()))
    print(f"  ridge           : {u['ridge']:.2f} FLOP/B  "
          f"(= {dom}_peak / (hbm_bw_chip / cores_active={cores_active}))")
    print(f"  core_attainment : {core['core_attainment']:.1%}   (achieved/ceiling at operating AI)")
    print(f"  peak_bw/core    : {core['peak_bw_gb_s']:.0f} GB/s  "
          f"({hw.hbm_bw_chip/1e12:.3f} TB/s / {cores_active} cores)")
    print(f"  Kernel is {'compute-bound' if core['core_AI'] >= u['ridge'] else 'memory-bound'} (per-core view)")


def print_per_core_table(report):
    """Print per-core breakdown table (compute/memory/comm/total)."""
    per_core = report.per_core_summary()
    header = f"{'core':>4}  {'compute':>12}  {'memory':>12}  {'comm':>12}  {'total':>12}"
    print(header)
    print("-" * len(header))
    for row in per_core:
        print(f"{row['core_id']:>4}  {row['compute_cycles']:>12.0f}  "
              f"{row['memory_cycles']:>12.0f}  {row['comm_cycles']:>12.0f}  "
              f"{row['total_cycles']:>12.0f}")


def print_kernel_comparison(kernels):
    """Print one-line per-core roofline summary for each kernel."""
    for name, r in kernels:
        core = r.core_roofline()
        print(f"{name:15s} cycles={r.kernel_cycles:8.0f}  bottleneck={r.bottleneck:8s}  "
              f"core_AI={core['core_AI']:6.2f} F/B  dom={core['core_dominant_unit']:8s}  "
              f"attainment={core['core_attainment']:.1%}")


def print_chip_comparison(kernels):
    """Print chip-level comparison table."""
    print("== Chip-level (whole chip as one unit) ==")
    for k in kernels:
        name, r = k[0], k[1]
        chip = r.chip_roofline()
        print(f"{name:15s} AI={chip['AI']:6.2f} F/B  compute_thru={chip['compute_throughput']:6.1%}  "
              f"dram_thru={chip['dram_throughput']:6.1%}  mean_core_active={chip['mean_core_active_frac']:5.1%}  "
              f"grid_cov={chip['grid_coverage']:5.1%}  ({chip['cores_active']}/{chip['num_cores']} cores)")


def print_scaling_table(kernels):
    """Print scaling experiment results table from all_kernels list."""
    print(f"{'Kernel':<22s} {'AI':>6s} {'comp%':>6s} {'dram%':>6s} {'att%':>6s} "
          f"{'grid':>5s} {'bottleneck':<10s} {'core_AI':>8s} {'c_att%':>6s}")
    print("-" * 82)
    for k in kernels:
        name, r = k[0], k[1]
        chip = r.chip_roofline()
        core = r.core_roofline()
        print(f"{name:<22s} {chip['AI']:>6.1f} {chip['compute_throughput']:>5.1%} "
              f"{chip['dram_throughput']:>5.1%} {chip['attainment']:>5.1%} "
              f"{chip['grid_coverage']:>5.0%} {r.bottleneck:<10s} "
              f"{core['core_AI']:>8.1f} {core['core_attainment']:>5.1%}")


def print_trace_summary(report, core_id=0):
    """Print per-op trace summary grouped by (op_type, category)."""
    core = report.counters[core_id]
    groups = defaultdict(lambda: [0.0, 0])
    for e in core.trace:
        groups[(e.op_type, e.category)][0] += e.cycles
        groups[(e.op_type, e.category)][1] += 1

    print(f"{'Op':<45}  {'Category':<25}  {'Count':>5}  {'Cycles':>10}")
    print("-" * 95)
    for (op, cat), (cyc, cnt) in sorted(groups.items(), key=lambda x: -x[1][0]):
        if cyc > 0 or cnt > 1:
            print(f"{op:<45}  {cat:<25}  {cnt:>5}  {cyc:>10.1f}")
    print("-" * 95)
    print(f"{'TOTAL':<77}  {core.total_cycles:>10.1f}")
    print(f"  compute={core.compute_cycles:.1f}  memory={core.memory_cycles:.1f}  comm={core.comm_cycles:.1f}")


# ---------------------------------------------------------------------------
# Tensor-setup helpers
# ---------------------------------------------------------------------------

def make_sdpa_tensors(seq_len, head_dim, rng=None):
    """Create Q, K, V, output tensors for SDPA."""
    if rng is None:
        rng = np.random.default_rng(0)
    return dict(
        q_ptr=rng.standard_normal((seq_len, head_dim)).astype(np.float16),
        k_ptr=rng.standard_normal((seq_len, head_dim)).astype(np.float16),
        v_ptr=rng.standard_normal((seq_len, head_dim)).astype(np.float16),
        output_ptr=np.zeros((seq_len, head_dim), dtype=np.float16),
    )


def make_pa_tensors(num_tokens, num_query_heads, num_kv_heads, head_dim,
                    block_size, context_len, rng=None):
    """Create query, KV cache, block_tables, and output tensors for paged attention."""
    if rng is None:
        rng = np.random.default_rng(0)
    num_tiles = (context_len + block_size - 1) // block_size
    return dict(
        output_ptr=np.zeros((num_tokens, num_query_heads, head_dim), dtype=np.float16),
        query_ptr=rng.standard_normal((num_tokens, num_query_heads, head_dim)).astype(np.float16),
        key_cache_ptr=rng.standard_normal((num_tiles, block_size, num_kv_heads, head_dim)).astype(np.float16),
        value_cache_ptr=rng.standard_normal((num_tiles, block_size, num_kv_heads, head_dim)).astype(np.float16),
        block_tables_ptr=rng.integers(0, num_tiles, size=(1, num_tiles), dtype=np.int32),
    )


def make_pa_scalars(context_len, block_size, head_dim):
    """Create scalar kwargs for paged attention kernel."""
    num_tiles = (context_len + block_size - 1) // block_size
    scale = 1.0 / (head_dim ** 0.5)
    return dict(cur_batch_start_index=0, block_table_offset=0,
                num_tiles=num_tiles, context_len=context_len, scale=scale)


# ---------------------------------------------------------------------------
# High-level kernel runners (gen MLIR + tensors + run in one call)
# ---------------------------------------------------------------------------

def run_kernel_matmul(hw, M, N, K, bm, bn, bk, rng=None):
    """Generate matmul MLIR, create tensors, run, return LatencyReport."""
    if rng is None:
        rng = np.random.default_rng(0)
    mlir = gen_matmul_mlir(M, N, K, bm, bn, bk)
    return run_kernel(hw, mlir, "matmul_kernel",
        dict(a_ptr=rng.standard_normal((M, K)).astype(np.float16),
             b_ptr=rng.standard_normal((K, N)).astype(np.float16),
             c_ptr=np.zeros((M, N), dtype=np.float16)),
        dict(K=K, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk))


def run_kernel_softmax(hw, n_rows, row_width, num_cores, rng=None):
    """Generate softmax MLIR, create tensors, run, return LatencyReport."""
    if rng is None:
        rng = np.random.default_rng(0)
    mlir = gen_softmax_mlir(n_rows, row_width, num_cores)
    return run_kernel(hw, mlir, "softmax_kernel",
        dict(output_ptr=np.zeros((n_rows, row_width), dtype=np.float16),
             input_ptr=rng.standard_normal((n_rows, row_width)).astype(np.float16)),
        dict(n_rows=n_rows))


def run_kernel_sdpa(hw, seq_len, head_dim, block_m, rng=None):
    """Generate SDPA MLIR, create tensors, run, return LatencyReport."""
    if rng is None:
        rng = np.random.default_rng(0)
    mlir = gen_sdpa_mlir(seq_len, head_dim, block_m)
    return run_kernel(hw, mlir, "sdpa_kernel", make_sdpa_tensors(seq_len, head_dim, rng))


def run_kernel_pa(hw, num_tokens, context_len, num_query_heads, num_kv_heads,
                  head_dim, block_size, block_q, rng=None):
    """Generate paged attention MLIR, create tensors, run, return LatencyReport."""
    if rng is None:
        rng = np.random.default_rng(0)
    mlir = gen_paged_attention_mlir(num_tokens, context_len, num_query_heads,
                                    num_kv_heads, head_dim, block_size, block_q)
    return run_kernel(hw, mlir, "paged_attention_kernel",
        make_pa_tensors(num_tokens, num_query_heads, num_kv_heads, head_dim,
                        block_size, context_len, rng),
        make_pa_scalars(context_len, block_size, head_dim))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roofline(hw: HardwareConfig, kernels: list,
                  granularity: str, title: str | None = None,
                  legend_loc: str | None = None) -> None:
    """Log-log roofline with absolute TFLOP/s on Y.

    Each entry in *kernels* is (name, report) or (name, report, marker) where
    marker is a matplotlib marker string (default 'o').
    legend_loc: matplotlib legend location string (default: 'upper left' for
    chip, 'lower right' for core)."""
    assert granularity in ("chip", "core")
    if legend_loc is None:
        legend_loc = "upper left" if granularity == "chip" else "lower right"

    entries = []
    for k in kernels:
        name, rep = k[0], k[1]
        marker = k[2] if len(k) > 2 else "o"
        rf = rep.chip_roofline() if granularity == "chip" else rep.core_roofline()
        entries.append((name, rf, marker))

    dom_key = "dominant_unit" if granularity == "chip" else "core_dominant_unit"
    ai_key = "AI" if granularity == "chip" else "core_AI"

    # Fixed color per kernel base name (text before first parenthesis).
    base_names = list(dict.fromkeys(
        n.split("(")[0].strip() for n, _, _ in entries
    ))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    kernel_color = {bn: cycle[i % len(cycle)] for i, bn in enumerate(base_names)}

    groups: dict[str, list] = defaultdict(list)
    for name, rf, marker in entries:
        groups[rf[dom_key]].append((name, rf, marker))

    n_panels = len(groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4.5 * n_panels), squeeze=False)
    fig.suptitle(title or f"Roofline -- {granularity}-level", fontsize=12)

    for row, (unit, kgroup) in enumerate(groups.items()):
        ax = axes[row][0]

        if granularity == "core":
            ridges = sorted(set(rf["units"][unit]["ridge"] for _, rf, _ in kgroup))
            peak_tflops = kgroup[0][1]["units"][unit]["peak_gflops"] / 1e3
        else:
            ridges = [kgroup[0][1]["ridge"]]
            peak_tflops = kgroup[0][1]["peak_gflops"] / 1e3

        xs = [rf[ai_key] for _, rf, _ in kgroup]
        finite = [x for x in xs if x > 0 and x != float("inf")]
        ai_min = min(ridges + finite) * 0.5 if (ridges or finite) else 0.1
        ai_max = max(ridges + finite) * 2.0

        ai_range = np.geomspace(ai_min, ai_max, 400)
        ridge_colors = ["gray", "silver", "darkgray", "lightgray"]
        for ri, ridge in enumerate(ridges):
            bw_tflops = peak_tflops / ridge
            roof = np.minimum(bw_tflops * ai_range, peak_tflops)
            rc = ridge_colors[ri % len(ridge_colors)]
            ax.plot(ai_range, roof, color=rc, linewidth=2 if ri == 0 else 1.5,
                    linestyle="-" if ri == 0 else "--",
                    label=f"peak={peak_tflops:.2f} TFLOP/s, ridge={ridge:.4g} F/B")
            mid_ai = ai_min * (ridge / ai_min) ** 0.15
            mid_tflops = bw_tflops * mid_ai
            bw_gb_s = bw_tflops * 1e3
            angle = np.degrees(np.arctan2(np.log10(peak_tflops) - np.log10(bw_tflops * ai_min),
                                          np.log10(ridge) - np.log10(ai_min)))
            ax.text(mid_ai, mid_tflops * 1.3, f"{bw_gb_s:.0f} GB/s",
                    color=rc, fontsize=7, ha="center", va="bottom",
                    rotation=angle, rotation_mode="anchor")

        for name, rf, marker in kgroup:
            c = kernel_color[name.split("(")[0].strip()]
            if granularity == "chip":
                x = rf["AI"]
                y = rf["achieved_gflops"] / 1e3
            else:
                u = rf["units"][unit]
                x = rf["core_AI"]
                y = u["achieved_gflops"] / 1e3
            if x > 0 and x != float("inf") and y > 0:
                ax.scatter(x, y, s=100, zorder=5, color=c, edgecolor="black", linewidth=0.5,
                           marker=marker, label=f"{name} (AI={x:.1f}, {y:.3f} TF/s)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(peak_tflops, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(ai_max * 0.85, peak_tflops, f"{peak_tflops:.2f} TF/s", color="gray", fontsize=7, ha="right", va="top")
        for ri, ridge in enumerate(ridges):
            rc = ridge_colors[ri % len(ridge_colors)]
            ax.axvline(ridge, color=rc, linestyle=":", linewidth=1, alpha=0.7)
            ax.text(ridge * 0.9, peak_tflops, f"ridge={ridge:.4g} F/B", color=rc, fontsize=7, ha="right", va="center")
        ax.set_xlabel("Arithmetic intensity (FLOP/B)", fontsize=10)
        ax.set_ylabel("Throughput (TFLOP/s)", fontsize=10)
        ax.set_title(f"Dominant unit: {unit}", fontsize=10)
        ax.legend(fontsize=7, loc=legend_loc)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.show()

