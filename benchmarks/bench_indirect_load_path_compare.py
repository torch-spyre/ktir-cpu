"""Indirect-load path timing: T_indirect_load, T_load, T_offsets.

Measures the full indirect_load and its internal MemoryOps.load call separately
via a timing wrapper. Implementation-agnostic — works on any branch that has
MemoryOps.indirect_load calling MemoryOps.load internally.

Usage:
    uv run python benchmarks/bench_indirect_load_path_compare.py
    uv run python benchmarks/bench_indirect_load_path_compare.py --config configs/load_path_compare.toml
"""

import time
import argparse
import numpy as np

from bench_utils import (
    load_config, make_bench_context, reset_lx,
    build_moe_iat, build_sparse_attn_iat, build_multi_head_iat, build_paged_attn_iat,
    BenchTable,
)
from ktir_cpu.ops.memory_ops import MemoryOps


class LoadTimer:
    """Context manager that wraps MemoryOps.load to capture its elapsed time."""

    def __init__(self):
        self._orig = MemoryOps.load
        self.elapsed = 0.0

    def __enter__(self):
        timer = self
        orig = self._orig

        def timed_load(*args, **kwargs):
            t0 = time.perf_counter()
            result = orig(*args, **kwargs)
            timer.elapsed += time.perf_counter() - t0
            return result

        MemoryOps.load = staticmethod(timed_load)
        return self

    def __exit__(self, *_):
        MemoryOps.load = staticmethod(self._orig)


def _build_iat(ctx, w):
    """Build an IAT from a workload config entry."""
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
            ctx, w.get("n_experts", w.get("num_experts")),
            w["n_heads"], w["M"], w["N"],
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


def _measure_one(ctx, iat, n_warmup, n_rounds):
    """Measure T_indirect_load and T_load for one workload.

    Returns (median_indirect_load_ms, median_load_ms).
    """
    for _ in range(n_warmup):
        reset_lx(ctx)
        MemoryOps.indirect_load(ctx, iat)

    indirect_times = []
    load_times = []
    for _ in range(n_rounds):
        reset_lx(ctx)
        with LoadTimer() as lt:
            t0 = time.perf_counter()
            MemoryOps.indirect_load(ctx, iat)
            t_indirect = (time.perf_counter() - t0) * 1000
            t_load = lt.elapsed * 1000
        indirect_times.append(t_indirect)
        load_times.append(t_load)

    return float(np.median(indirect_times)), float(np.median(load_times))


def main():
    parser = argparse.ArgumentParser(
        description="Indirect-load path timing benchmark",
    )
    parser.add_argument(
        "--config", default="configs/load_path_compare.toml",
        help="Path to TOML config (default: configs/load_path_compare.toml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    table = BenchTable(
        headers=["Workload", "Access expr", "Source shape", "Result shape", "Model",
                 "T_indirect_load", "T_load", "T_offsets"],
        align=["<", "<", "<", "<", "<", ">", ">", ">"],
    )

    for w in config.workloads:
        ctx = make_bench_context()
        iat = _build_iat(ctx, w)

        n_warmup = w.get("warmup", config.defaults.get("warmup", 2))
        n_rounds = w.get("n_rounds", config.defaults.get("n_rounds", 5))

        t_indirect, t_load = _measure_one(ctx, iat, n_warmup, n_rounds)
        t_offsets = t_indirect - t_load

        src_shape_str = str(list(iat.parent_ref.shape))
        result_shape_str = str(list(iat.shape))
        table.add_row([
            w["label"],
            w.get("access_expr", "—"),
            src_shape_str,
            result_shape_str,
            w.get("model_scale_ref", "—"),
            f"{t_indirect:.2f} ms",
            f"{t_load:.2f} ms",
            f"{t_offsets:.2f} ms",
        ])

    table.print(title=f"bench_indirect_load_path_compare — {config.name}")


if __name__ == "__main__":
    main()
