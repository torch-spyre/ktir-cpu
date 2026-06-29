"""Benchmark utilities for KTIR CPU interpreter.

Provides size parsing, context/IAT factories, timing harness,
table output, and TOML config loading.
"""

from __future__ import annotations

import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ktir_cpu.affine import BoxSet
from ktir_cpu.ir_types import MemRef, IndirectAccessTile
from ktir_cpu.grid import CoreContext
from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.dtypes import bytes_per_elem


# ---------------------------------------------------------------------------
# Size parsing / formatting
# ---------------------------------------------------------------------------

_SI_SUFFIXES = {
    "k": 1_000, "kilo": 1_000,
    "m": 1_000_000, "million": 1_000_000, "mega": 1_000_000,
    "g": 1_000_000_000, "giga": 1_000_000_000, "billion": 1_000_000_000,
    "t": 1_000_000_000_000, "tera": 1_000_000_000_000,
}

_BINARY_SUFFIXES = {
    "kb": 1024,
    "mb": 1024 ** 2,
    "gb": 1024 ** 3,
    "tb": 1024 ** 4,
}

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")


def parse_size(s: str | int | float) -> int:
    """Parse a human-readable size string to an integer.

    SI (element counts): '16M'→16e6, '4k'→4000, '1G'→1e9
    Binary (byte sizes): '512MB'→536870912, '4KB'→4096
    Rule: suffix ending in 'B' → binary bytes; otherwise → SI elements.
    """
    if isinstance(s, (int, float)):
        return int(s)
    m = _SIZE_RE.match(str(s))
    if not m:
        raise ValueError(f"Cannot parse size: {s!r}")
    num_str, suffix = m.group(1), m.group(2).lower()
    num = float(num_str)
    if not suffix:
        return int(num)
    if suffix in _BINARY_SUFFIXES:
        return int(num * _BINARY_SUFFIXES[suffix])
    if suffix in _SI_SUFFIXES:
        return int(num * _SI_SUFFIXES[suffix])
    raise ValueError(f"Unknown size suffix: {suffix!r} in {s!r}")


def format_size(n: int, mode: str = "si") -> str:
    """Pretty-print an integer as a human-readable size."""
    if mode == "bytes":
        for suffix, thresh in [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
            if n >= thresh:
                val = n / thresh
                return f"{val:.0f} {suffix}" if val == int(val) else f"{val:.1f} {suffix}"
        return f"{n} B"
    for suffix, thresh in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= thresh:
            val = n / thresh
            return f"{val:.0f}{suffix}" if val == int(val) else f"{val:.1f}{suffix}"
    return str(n)


# ---------------------------------------------------------------------------
# Context and IAT factories
# ---------------------------------------------------------------------------

def make_bench_context(lx_size_mb: int = 512) -> CoreContext:
    """Create a fresh CoreContext for benchmarking."""
    hbm = HBMSimulator()
    lx = LXScratchpad(size_mb=lx_size_mb)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)


def build_moe_iat(
    ctx: CoreContext,
    num_experts: int,
    M: int,
    N: int,
    n_selected: int,
    dtype: str = "f16",
) -> IndirectAccessTile:
    """Build an MoE-style IAT: X[IDX[e], M, N] selecting n_selected from num_experts."""
    hbm = ctx.hbm
    bpe_data = bytes_per_elem(dtype)
    bpe_idx = bytes_per_elem("i32")

    x_data = np.random.randn(num_experts * M * N).astype(np.float16)
    x_stick = hbm.allocate(x_data.nbytes)
    hbm.write(x_stick, x_data)
    x_base_ptr = (x_stick * HBMSimulator.STICK_BYTES) // bpe_data

    selected = np.sort(
        np.random.choice(num_experts, size=n_selected, replace=False)
    ).astype(np.int32)
    idx_stick = hbm.allocate(selected.nbytes)
    hbm.write(idx_stick, selected)
    idx_base_ptr = (idx_stick * HBMSimulator.STICK_BYTES) // bpe_idx

    x_memref = MemRef(
        base_ptr=x_base_ptr,
        shape=(num_experts, M, N),
        strides=[M * N, N, 1],
        memory_space="HBM", dtype=dtype,
    )
    idx_memref = MemRef(
        base_ptr=idx_base_ptr, shape=(n_selected,), strides=[1],
        memory_space="HBM", dtype="i32",
    )

    dim_subscripts = [
        {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
        {"kind": "direct", "var_index": 1},
        {"kind": "direct", "var_index": 2},
    ]
    vss = BoxSet(lo=(0, 0, 0), hi=(n_selected, M, N))
    return IndirectAccessTile(
        parent_ref=x_memref, shape=(n_selected, M, N),
        dim_subscripts=dim_subscripts, index_views=[idx_memref],
        variables_space_set=vss, variables_space_order=None,
    )


def reset_lx(ctx: CoreContext):
    """Clear LX memory and reset pointer."""
    ctx.lx.memory.clear()
    ctx.lx.next_ptr = 0


# ---------------------------------------------------------------------------
# Cache flushing
# ---------------------------------------------------------------------------

_SCRUBBER: np.ndarray | None = None


def flush_cache(scrubber_mb: int = 32):
    """Write to a large array to evict working set from CPU cache."""
    global _SCRUBBER
    needed = scrubber_mb * 1024 * 1024 // 4
    if _SCRUBBER is None or _SCRUBBER.size != needed:
        _SCRUBBER = np.empty(needed, dtype=np.float32)
    _SCRUBBER[:] = 0.0


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

@dataclass
class BenchTimer:
    """Configurable timing harness for benchmark measurements."""
    n_warmup: int = 3
    n_rounds: int = 10
    cache_flush: bool = False

    def measure(self, fn: Callable[[], Any]) -> float:
        """Time fn() over n_rounds, return median in milliseconds."""
        for _ in range(self.n_warmup):
            fn()
        times = []
        for _ in range(self.n_rounds):
            if self.cache_flush:
                flush_cache()
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    def measure_pair(
        self, fn_a: Callable[[], Any], fn_b: Callable[[], Any],
    ) -> Tuple[float, float]:
        """Interleaved A/B measurement with alternating order.

        Returns (median_a_ms, median_b_ms).
        """
        for _ in range(self.n_warmup):
            fn_a()
            fn_b()
        a_times, b_times = [], []
        for i in range(self.n_rounds):
            if i % 2 == 0:
                if self.cache_flush:
                    flush_cache()
                t0 = time.perf_counter()
                fn_a()
                a_times.append((time.perf_counter() - t0) * 1000)
                if self.cache_flush:
                    flush_cache()
                t0 = time.perf_counter()
                fn_b()
                b_times.append((time.perf_counter() - t0) * 1000)
            else:
                if self.cache_flush:
                    flush_cache()
                t0 = time.perf_counter()
                fn_b()
                b_times.append((time.perf_counter() - t0) * 1000)
                if self.cache_flush:
                    flush_cache()
                t0 = time.perf_counter()
                fn_a()
                a_times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(a_times)), float(np.median(b_times))

    def measure_steps(
        self, fn: Callable[[], Dict[str, float]],
    ) -> Dict[str, float]:
        """Per-step timing. fn() returns {step_name: elapsed_ms}.

        Returns per-key medians over n_rounds.
        """
        for _ in range(self.n_warmup):
            fn()
        all_results: Dict[str, List[float]] = {}
        for _ in range(self.n_rounds):
            result = fn()
            for k, v in result.items():
                all_results.setdefault(k, []).append(v)
        return {k: float(np.median(v)) for k, v in all_results.items()}


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

@dataclass
class BenchTable:
    """Simple formatted table printer for benchmark results."""
    headers: List[str]
    align: Optional[List[str]] = None

    def __post_init__(self):
        self._rows: List[List[str] | None] = []
        if self.align is None:
            self.align = [">"] * len(self.headers)

    def add_row(self, values: List[Any]):
        self._rows.append([str(v) for v in values])

    def add_separator(self):
        self._rows.append(None)

    def print(self, title: str = "", notes: List[str] | None = None):
        widths = [len(h) for h in self.headers]
        for row in self._rows:
            if row is None:
                continue
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(cells):
            parts = []
            for i, cell in enumerate(cells):
                a = self.align[i] if i < len(self.align) else ">"
                parts.append(f"{cell:{a}{widths[i]}}")
            return " | ".join(parts)

        sep = "-+-".join("-" * w for w in widths)

        if title:
            print(title)
            print("=" * len(sep))
        print(fmt_row(self.headers))
        print(sep)
        for row in self._rows:
            if row is None:
                print(sep)
            else:
                print(fmt_row(row))
        if notes:
            print()
            for note in notes:
                print(f"  {note}")
        print()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    """Loaded benchmark configuration."""
    name: str
    description: str
    defaults: Dict[str, Any]
    modes: Dict[str, bool]
    workloads: List[Dict[str, Any]]


def _expand_workload(entry: Dict[str, Any], defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a single [[workloads]] entry into one or more dicts."""
    mode = entry.pop("mode", None)
    list_keys = [k for k, v in entry.items() if isinstance(v, list)]

    if not list_keys:
        merged = {**defaults, **entry}
        _parse_size_fields(merged)
        return [merged]

    if mode is None:
        raise ValueError(
            f"Workload has list-valued fields {list_keys} but no 'mode' "
            f"('product' or 'zip') specified. Mode is required when lists are present."
        )

    if mode == "zip":
        lengths = [len(entry[k]) for k in list_keys]
        if len(set(lengths)) != 1:
            raise ValueError(f"zip mode requires equal-length lists, got {dict(zip(list_keys, lengths))}")
        n = lengths[0]
        results = []
        for i in range(n):
            row = {**defaults}
            for k, v in entry.items():
                row[k] = v[i] if k in list_keys else v
            _parse_size_fields(row)
            results.append(row)
        return results

    if mode == "product":
        import itertools
        list_values = [entry[k] for k in list_keys]
        scalar_keys = [k for k in entry if k not in list_keys]
        results = []
        for combo in itertools.product(*list_values):
            row = {**defaults}
            for k in scalar_keys:
                row[k] = entry[k]
            for k, v in zip(list_keys, combo):
                row[k] = v
            _parse_size_fields(row)
            results.append(row)
        return results

    raise ValueError(f"Unknown mode: {mode!r}. Must be 'product' or 'zip'.")


_SKIP_PARSE_KEYS = {"label", "dtype", "mode", "description", "name"}


def _parse_size_fields(d: Dict[str, Any]):
    """Auto-parse string values that look like sizes (e.g. '1K', '16M')."""
    for k, v in d.items():
        if k in _SKIP_PARSE_KEYS:
            continue
        if isinstance(v, str) and _SIZE_RE.match(v):
            d[k] = parse_size(v)


def load_config(toml_path: str | Path) -> BenchConfig:
    """Load a TOML benchmark config and expand the workload matrix."""
    path = Path(toml_path)
    if not path.is_absolute():
        # Resolve relative to the caller's script directory
        import inspect
        caller_file = inspect.stack()[1].filename
        path = Path(caller_file).parent / path

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    defaults = raw.get("defaults", {})
    modes = raw.get("modes", {})
    workloads_raw = raw.get("workloads", [])

    workloads = []
    for entry in workloads_raw:
        entry_copy = dict(entry)
        workloads.extend(_expand_workload(entry_copy, defaults))

    return BenchConfig(
        name=raw.get("name", "benchmark"),
        description=raw.get("description", ""),
        defaults=defaults,
        modes=modes,
        workloads=workloads,
    )
