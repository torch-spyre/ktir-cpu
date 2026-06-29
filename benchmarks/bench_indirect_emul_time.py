"""Indirect memory access emulation timing — MoE block-gather patterns.

Subcommands:
  block   — block-gather fast path vs general inspector (summary + breakdown)
  gather  — 3-memcpy (old) vs 1-memcpy (new) gather step comparison

Usage:
    uv run python benchmarks/bench_indirect_emul_time.py block
    uv run python benchmarks/bench_indirect_emul_time.py gather
    uv run python benchmarks/bench_indirect_emul_time.py block --config configs/custom.toml
"""

import time
import numpy as np

from bench_utils import (
    load_config, make_bench_context, build_moe_iat, reset_lx,
    flush_cache, format_size, BenchTimer, BenchTable,
)
from ktir_cpu.ops.memory_ops import (
    MemoryOps, _MemAccessor,
    _is_block_gather, _block_gather_load,
    _resolve_idx_reads, _build_indirect_coords,
    _enumerate_in_vso_order,
)
from ktir_cpu.memory import _read_flat
from ktir_cpu.dtypes import bytes_per_elem, to_np_dtype


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_general_path(ctx, iat):
    reset_lx(ctx)
    idx_values, _ = _resolve_idx_reads(ctx, iat)
    coords = _build_indirect_coords(iat, idx_values)
    MemoryOps.load(ctx, iat.parent_ref.to_tile_ref(), coords=coords, result_shape=iat.shape)


def _run_fast_path(ctx, iat):
    reset_lx(ctx)
    _block_gather_load(ctx, iat)


# ---------------------------------------------------------------------------
# block subcommand
# ---------------------------------------------------------------------------

def _old_7_steps(ctx, iat) -> dict:
    """One iteration of old 7-step path, returns per-step ms."""
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
    MemoryOps._place_in_lx(ctx, data)
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


def _new_3_steps(ctx, iat) -> dict:
    """One iteration of new 3-step path, returns per-step ms."""
    from ktir_cpu.ops.memory_ops import _block_gather_analyze, _block_gather_read_idx, _block_gather_offsets

    reset_lx(ctx)
    info = _block_gather_analyze(iat)
    indirect_sub, dep_vars, dep_var_list, dep_extents, dep_los = info

    t0 = time.perf_counter()
    idx_values_arr, _ = _block_gather_read_idx(ctx, iat, indirect_sub, dep_vars, dep_var_list)
    t1 = time.perf_counter()
    offsets = _block_gather_offsets(iat, idx_values_arr, dep_vars, dep_var_list, dep_extents, dep_los)
    t2 = time.perf_counter()
    tile_ref = iat.parent_ref.to_tile_ref()
    mgr = _MemAccessor(ctx, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
    gathered = mgr.gather(offsets, tile_ref.dtype)
    data = gathered.reshape(iat.shape)
    MemoryOps._place_in_lx(ctx, data)
    t3 = time.perf_counter()
    return {
        "1. Read K index values": (t1 - t0) * 1000,
        "2. Numpy broadcast offsets": (t2 - t1) * 1000,
        "3. Gather + reshape + LX": (t3 - t2) * 1000,
    }


def cmd_block(config):
    """Fast-path vs general-path summary + per-step breakdown."""
    print(f"{config.name}: block-gather fast path")
    print()

    # --- Summary table ---
    table = BenchTable(
        headers=["Workload", "Shape", "Points", "General (ms)", "Fast (ms)", "Speedup"],
    )

    for w in config.workloads:
        ctx = make_bench_context()
        iat = build_moe_iat(ctx, w["num_experts"], w["M"], w["N"], w["n_selected"], w.get("dtype", "f16"))
        assert _is_block_gather(iat), f"Workload {w['label']} does not qualify for fast path"

        timer = BenchTimer(
            n_warmup=w.get("warmup", config.defaults.get("warmup", 2)),
            n_rounds=w.get("n_rounds", config.defaults.get("n_rounds", 5)),
            cache_flush=True,
        )
        general_ms, fast_ms = timer.measure_pair(
            lambda: _run_general_path(ctx, iat),
            lambda: _run_fast_path(ctx, iat),
        )
        total_pts = w["n_selected"] * w["M"] * w["N"]
        speedup = general_ms / fast_ms if fast_ms > 0 else float("inf")
        shape_str = f"{w['num_experts']}x{w['M']}x{w['N']}"
        table.add_row([
            w["label"], shape_str, f"{total_pts:,}",
            f"{general_ms:.2f}", f"{fast_ms:.2f}", f"{speedup:.1f}x",
        ])

    table.print()

    # --- Per-step breakdown on last (largest) workload ---
    if config.modes.get("breakdown"):
        w = config.workloads[-1]
        print(f"Step breakdown ({w['label']} workload):")
        print("-" * 60)

        ctx = make_bench_context()
        iat = build_moe_iat(ctx, w["num_experts"], w["M"], w["N"], w["n_selected"], w.get("dtype", "f16"))
        timer = BenchTimer(
            n_warmup=w.get("warmup", config.defaults.get("warmup", 2)),
            n_rounds=w.get("n_rounds", config.defaults.get("n_rounds", 5)),
        )

        old = timer.measure_steps(lambda: _old_7_steps(ctx, iat))
        new = timer.measure_steps(lambda: _new_3_steps(ctx, iat))

        old_total = sum(old.values())
        new_total = sum(new.values())

        print("  Old path (7 steps):")
        for step, ms in old.items():
            print(f"    {step}:{' ' * (35 - len(step))}{ms:>8.3f} ms  ({ms/old_total*100:>5.1f}%)")
        print(f"    {'TOTAL':{35}}{old_total:>8.3f} ms")
        print()

        print("  New path (3 steps):")
        for step, ms in new.items():
            print(f"    {step}:{' ' * (35 - len(step))}{ms:>8.3f} ms  ({ms/new_total*100:>5.1f}%)")
        print(f"    {'TOTAL':{35}}{new_total:>8.3f} ms")
        print()

        print(f"  Speedup: {old_total / new_total:.0f}x")
        print()


# ---------------------------------------------------------------------------
# gather subcommand
# ---------------------------------------------------------------------------

def cmd_gather(config):
    """3-memcpy (old) vs 1-memcpy (new) gather step comparison."""
    print(f"{config.name}: gather memcpy comparison (3-copy vs 1-copy)")
    print()

    table = BenchTable(
        headers=["Workload", "Gather (elems)", "Span (elems)", "Old (ms)", "New (ms)", "Speedup"],
    )

    for w in config.workloads:
        ctx = make_bench_context()
        iat = build_moe_iat(ctx, w["num_experts"], w["M"], w["N"], w["n_selected"], w.get("dtype", "f16"))
        dtype = w.get("dtype", "f16")

        # Compute offsets once (use fast path to get them)
        from ktir_cpu.ops.memory_ops import _block_gather_analyze, _block_gather_read_idx, _block_gather_offsets

        info = _block_gather_analyze(iat)
        indirect_sub, dep_vars, dep_var_list, dep_extents, dep_los = info
        idx_values_arr, _ = _block_gather_read_idx(ctx, iat, indirect_sub, dep_vars, dep_var_list)
        offsets = _block_gather_offsets(iat, idx_values_arr, dep_vars, dep_var_list, dep_extents, dep_los)

        tile_ref = iat.parent_ref.to_tile_ref()
        mgr = _MemAccessor(ctx, tile_ref.memref.memory_space, tile_ref.base_ptr, tile_ref.memref.lx_core_id)
        span = int(offsets.max()) + 1
        np_dtype = to_np_dtype(dtype)
        elem_size = bytes_per_elem(dtype)

        def old_gather():
            reset_lx(ctx)
            flat = _read_flat(ctx.hbm.memory, tile_ref.base_ptr, span, np_dtype, elem_size)
            gathered = flat[offsets]
            ctx.lx.memory[ctx.lx.next_ptr] = gathered.flatten()

        def new_gather():
            reset_lx(ctx)
            gathered = mgr.gather(offsets, dtype)
            ctx.lx.memory[ctx.lx.next_ptr] = gathered

        timer = BenchTimer(
            n_warmup=w.get("warmup", config.defaults.get("warmup", 2)),
            n_rounds=w.get("n_rounds", config.defaults.get("n_rounds", 5)),
            cache_flush=True,
        )
        old_ms, new_ms = timer.measure_pair(old_gather, new_gather)
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")
        table.add_row([
            w["label"], f"{offsets.size:,}", f"{span:,}",
            f"{old_ms:.4f}", f"{new_ms:.4f}", f"{speedup:.1f}x",
        ])

    table.print(notes=[
        "Old: _read_flat(span) + flat[offsets] + write_flat(lx) — 3 memcpys",
        "New: mgr.gather(offsets) + place_in_lx — 1 memcpy",
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Indirect memory access emulation timing",
    )
    parser.add_argument("mode", choices=["block", "gather"],
                        help="'block': fast-path vs general (summary + breakdown). "
                             "'gather': 3-copy vs 1-copy gather step.")
    parser.add_argument("--config", default="configs/indirect_emul.toml",
                        help="Path to TOML config (default: configs/indirect_emul.toml)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode == "block":
        cmd_block(config)
    elif args.mode == "gather":
        cmd_gather(config)


if __name__ == "__main__":
    main()
