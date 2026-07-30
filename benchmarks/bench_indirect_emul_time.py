"""Indirect memory access emulation timing — blocked-indirect vs element-wise.

Compares the blocked-indirect fast path (3-step numpy broadcast) against the
general inspector-executor element-wise path (7-step per-point Python loop).

Usage:
    uv run python benchmarks/bench_indirect_emul_time.py
    uv run python benchmarks/bench_indirect_emul_time.py --config configs/custom.toml
"""

import time
import numpy as np

from bench_utils import (
    load_config, make_bench_context,
    build_moe_iat, build_sparse_attn_iat, build_multi_head_iat, build_paged_attn_iat,
    reset_lx, flush_cache, format_size, BenchTimer, BenchTable,
)
from ktir_cpu.ops.memory_ops import (
    MemoryOps, _MemAccessor,
    _analyze_blocked_indirect,
    _runtime_read_and_expand_sub_space, _prepare_dep_var_sub_space,
    _gen_offsets_vso_space_via_broadcast,
    _resolve_idx_reads, _build_indirect_coords,
    _enumerate_in_vso_order,
    _element_offsets,
)
from ktir_cpu.dtypes import bytes_per_elem, to_np_dtype


# ---------------------------------------------------------------------------
# IAT builder dispatch
# ---------------------------------------------------------------------------

def _build_iat(ctx, w):
    """Build an IAT from a workload entry based on its pattern field."""
    pattern = w["pattern"]
    if pattern == "moe_ffn":
        return build_moe_iat(
            ctx, w["num_experts"], w["M"], w["N"],
            w["n_selected"], w.get("dtype", "f16"),
        )
    if pattern == "paged_attn":
        return build_paged_attn_iat(
            ctx, w["n_pages"], w["n_heads"], w["block_size"], w["head_dim"],
            w["n_sel_pages"], w.get("dtype", "f16"),
        )
    if pattern == "sparse_attn":
        return build_sparse_attn_iat(
            ctx, w["n_pages"], w["n_tokens"], w["hidden_dim"],
            w["n_sel_pages"], w["n_sel_tokens"], w.get("dtype", "f16"),
        )
    if pattern == "multi_head":
        return build_multi_head_iat(
            ctx, w["n_experts"], w["n_heads"], w["M"], w["N"],
            w["n_sel_experts"], w["n_sel_heads"], w.get("dtype", "f16"),
        )
    raise ValueError(f"Unknown pattern: {pattern!r}")


def _count_points(iat):
    """Total iteration points from the VSS."""
    n = 1
    for d in range(iat.variables_space_set.n_dims):
        extent = int(iat.variables_space_set.hi[d]) - int(iat.variables_space_set.lo[d])
        if extent > 0:
            n *= extent
    return n


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _element_path_steps(ctx, iat) -> dict:
    """One iteration of element-wise 7-step path, returns per-step ms."""
    reset_lx(ctx)
    t0 = time.perf_counter()
    _enumerate_in_vso_order(iat)
    t1 = time.perf_counter()
    idx_values, _ = _resolve_idx_reads(ctx, iat)
    t2 = time.perf_counter()
    coords = _build_indirect_coords(iat, idx_values)
    t3 = time.perf_counter()
    tile_ref = iat.parent_ref.to_tile_ref()
    mgr = _MemAccessor(ctx, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
    offsets, _ = MemoryOps._flat_memory_offsets(
        tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype,
        coords, stick_bytes=mgr.stick_bytes,
    )
    t4 = time.perf_counter()
    gathered = mgr.gather(offsets, tile_ref.dtype)
    t5 = time.perf_counter()
    data = gathered.reshape(iat.shape)
    t6 = time.perf_counter()
    MemoryOps._write_to_lx(ctx, data)
    t7 = time.perf_counter()
    return {
        "1. Enumerate iteration space": (t1 - t0) * 1000,
        "2. Read index tensors": (t2 - t1) * 1000,
        "3. Build coordinate list": (t3 - t2) * 1000,
        "4. Linearize flat offsets": (t4 - t3) * 1000,
        "5. Gather from HBM": (t5 - t4) * 1000,
        "6. Reshape": (t6 - t5) * 1000,
        "7. Write to LX": (t7 - t6) * 1000,
    }


def _reference_path_steps(ctx, iat) -> dict:
    """Reference implementation via _element_offsets (Python loop), returns per-step ms."""
    reset_lx(ctx)
    t0 = time.perf_counter()
    offsets, idx_sticks = _element_offsets(ctx, iat)
    t1 = time.perf_counter()
    tile_ref = iat.parent_ref.to_tile_ref()
    mgr = _MemAccessor(ctx, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
    gathered = mgr.gather(offsets, tile_ref.dtype)
    t2 = time.perf_counter()
    data = gathered.reshape(iat.shape)
    MemoryOps._write_to_lx(ctx, data)
    t3 = time.perf_counter()
    return {
        "1. Compute offsets (Python loop)": (t1 - t0) * 1000,
        "2. Gather from HBM": (t2 - t1) * 1000,
        "3. Reshape + write LX": (t3 - t2) * 1000,
    }


def _block_path_steps(ctx, iat) -> dict:
    """One iteration of blocked-indirect 3-step path, returns per-step ms."""
    reset_lx(ctx)
    info = _analyze_blocked_indirect(iat)
    indirect_subs, dep_vars, dep_var_list, dep_extents = info

    t0 = time.perf_counter()
    points = _prepare_dep_var_sub_space(iat, dep_vars, dep_var_list)
    idx_values_map, _ = _runtime_read_and_expand_sub_space(ctx, iat, points, indirect_subs)
    t1 = time.perf_counter()
    offsets = _gen_offsets_vso_space_via_broadcast(iat, idx_values_map, indirect_subs, dep_vars, dep_var_list, dep_extents)
    t2 = time.perf_counter()
    tile_ref = iat.parent_ref.to_tile_ref()
    mgr = _MemAccessor(ctx, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
    gathered = mgr.gather(offsets, tile_ref.dtype)
    data = gathered.reshape(iat.shape)
    MemoryOps._write_to_lx(ctx, data)
    t3 = time.perf_counter()
    return {
        "1. Read K index values": (t1 - t0) * 1000,
        "2. Numpy broadcast offsets": (t2 - t1) * 1000,
        "3. Gather + reshape + LX": (t3 - t2) * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(config):
    """Block-gather fast path vs element-wise general path."""
    print(f"{config.name}: blocked-indirect fast path vs element-wise")
    print()

    table = BenchTable(
        headers=["Pattern", "Workload", "Points", "Element (ms)", "Block (ms)", "Speedup"],
    )

    for w in config.workloads:
        ctx = make_bench_context()
        iat = _build_iat(ctx, w)
        assert _analyze_blocked_indirect(iat) is not None, f"Workload {w['label']} does not qualify for fast path"

        timer = BenchTimer(
            n_warmup=w.get("warmup", config.defaults.get("warmup", 2)),
            n_rounds=w.get("n_rounds", config.defaults.get("n_rounds", 5)),
            cache_flush=True,
        )

        def run_element(ctx=ctx, iat=iat):
            reset_lx(ctx)
            idx_values, _ = _resolve_idx_reads(ctx, iat)
            coords = _build_indirect_coords(iat, idx_values)
            MemoryOps.load(ctx, iat.parent_ref.to_tile_ref(), coords=coords, result_shape=iat.shape)

        def run_block(ctx=ctx, iat=iat):
            reset_lx(ctx)
            MemoryOps.indirect_load(ctx, iat)

        elem_ms, block_ms = timer.measure_pair(run_element, run_block)
        n_points = _count_points(iat)
        speedup = elem_ms / block_ms if block_ms > 0 else float("inf")
        table.add_row([
            w["pattern"], w["label"], f"{n_points:,}",
            f"{elem_ms:.2f}", f"{block_ms:.2f}", f"{speedup:.1f}x",
        ])

    table.print()

    if config.modes.get("breakdown"):
        w = config.workloads[-1]
        print(f"Step breakdown ({w['label']} workload):")
        print("-" * 60)

        ctx = make_bench_context()
        iat = _build_iat(ctx, w)
        timer = BenchTimer(
            n_warmup=w.get("warmup", config.defaults.get("warmup", 2)),
            n_rounds=w.get("n_rounds", config.defaults.get("n_rounds", 5)),
        )

        ref = timer.measure_steps(lambda: _reference_path_steps(ctx, iat))
        elem = timer.measure_steps(lambda: _element_path_steps(ctx, iat))
        block = timer.measure_steps(lambda: _block_path_steps(ctx, iat))

        ref_total = sum(ref.values())
        elem_total = sum(elem.values())
        block_total = sum(block.values())

        print("  Reference path — Python loop via _element_offsets (3 steps):")
        for step, ms in ref.items():
            print(f"    {step}:{' ' * (35 - len(step))}{ms:>8.3f} ms  ({ms/ref_total*100:>5.1f}%)")
        print(f"    {'TOTAL':{35}}{ref_total:>8.3f} ms")
        print()

        print("  Element-wise path — vectorized numpy (7 steps):")
        for step, ms in elem.items():
            print(f"    {step}:{' ' * (35 - len(step))}{ms:>8.3f} ms  ({ms/elem_total*100:>5.1f}%)")
        print(f"    {'TOTAL':{35}}{elem_total:>8.3f} ms")
        print()

        print("  Block-gather path — numpy broadcast (3 steps):")
        for step, ms in block.items():
            print(f"    {step}:{' ' * (35 - len(step))}{ms:>8.3f} ms  ({ms/block_total*100:>5.1f}%)")
        print(f"    {'TOTAL':{35}}{block_total:>8.3f} ms")
        print()

        print(f"  Speedup: ref/block = {ref_total / block_total:.0f}x, "
              f"elem/block = {elem_total / block_total:.0f}x, "
              f"ref/elem = {ref_total / elem_total:.1f}x")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Block-gather vs element-wise path comparison",
    )
    parser.add_argument("--config", default="configs/indirect_emul.toml",
                        help="Path to TOML config (default: configs/indirect_emul.toml)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_comparison(config)


if __name__ == "__main__":
    main()
