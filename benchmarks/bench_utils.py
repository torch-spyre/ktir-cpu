"""Benchmark utilities for KTIR CPU interpreter.

Provides context/IAT factories, table output, and TOML config loading.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ktir_cpu.affine import BoxSet
from ktir_cpu.ir_types import MemRef, IndirectAccessTile
from ktir_cpu.grid import CoreContext
from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.dtypes import bytes_per_elem


# ---------------------------------------------------------------------------
# Context and IAT factories
# ---------------------------------------------------------------------------

def make_bench_context(lx_size_mb: int = 512) -> CoreContext:
    """Create a fresh CoreContext for benchmarking."""
    hbm = HBMSimulator()
    lx = LXScratchpad(size_mb=lx_size_mb)
    return CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=lx, hbm=hbm)


def _alloc_tensor(hbm: HBMSimulator, n_elems: int, dtype: str = "f16") -> int:
    """Allocate random f16 tensor in HBM, return element-addressed base_ptr."""
    bpe = bytes_per_elem(dtype)
    data = np.random.randn(n_elems).astype(np.float16)
    stick = hbm.allocate(data.nbytes)
    hbm.write(stick, data)
    return (stick * HBMSimulator.STICK_BYTES) // bpe


def _alloc_index(hbm: HBMSimulator, pool: int, n_sel: int) -> int:
    """Allocate sorted random i32 index selection in HBM, return base_ptr."""
    bpe = bytes_per_elem("i32")
    sel = np.sort(np.random.choice(pool, size=n_sel, replace=False)).astype(np.int32)
    stick = hbm.allocate(sel.nbytes)
    hbm.write(stick, sel)
    return (stick * HBMSimulator.STICK_BYTES) // bpe


def _strides_from_shape(shape: tuple) -> list:
    """Row-major strides for a given shape."""
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return list(reversed(strides))


def build_moe_iat(
    ctx: CoreContext,
    num_experts: int,
    M: int,
    N: int,
    n_selected: int,
    dtype: str = "f16",
) -> IndirectAccessTile:
    """X[IDX[e], M, N] — 1 indirect + 2 direct."""
    hbm = ctx.hbm
    shape = (num_experts, M, N)
    data_ptr = _alloc_tensor(hbm, num_experts * M * N, dtype)
    idx_ptr = _alloc_index(hbm, num_experts, n_selected)

    dim_subscripts = [
        {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
        {"kind": "direct", "var_index": 1},
        {"kind": "direct", "var_index": 2},
    ]
    vss = BoxSet(lo=(0, 0, 0), hi=(n_selected, M, N))
    return IndirectAccessTile(
        parent_ref=MemRef(base_ptr=data_ptr, shape=shape,
                          strides=_strides_from_shape(shape),
                          memory_space="HBM", dtype=dtype),
        shape=(n_selected, M, N),
        dim_subscripts=dim_subscripts,
        index_views=[MemRef(base_ptr=idx_ptr, shape=(n_selected,), strides=[1],
                            memory_space="HBM", dtype="i32")],
        variables_space_set=vss, variables_space_order=None,
    )


def build_sparse_attn_iat(
    ctx: CoreContext,
    n_pages: int,
    n_tokens: int,
    hidden_dim: int,
    n_sel_pages: int,
    n_sel_tokens: int,
    dtype: str = "f16",
) -> IndirectAccessTile:
    """cache[page_idx[b], token_idx[t], d] — 2 indirect + 1 direct."""
    hbm = ctx.hbm
    shape = (n_pages, n_tokens, hidden_dim)
    data_ptr = _alloc_tensor(hbm, n_pages * n_tokens * hidden_dim, dtype)
    page_ptr = _alloc_index(hbm, n_pages, n_sel_pages)
    token_ptr = _alloc_index(hbm, n_tokens, n_sel_tokens)

    dim_subscripts = [
        {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
        {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
        {"kind": "direct", "var_index": 2},
    ]
    vss = BoxSet(lo=(0, 0, 0), hi=(n_sel_pages, n_sel_tokens, hidden_dim))
    return IndirectAccessTile(
        parent_ref=MemRef(base_ptr=data_ptr, shape=shape,
                          strides=_strides_from_shape(shape),
                          memory_space="HBM", dtype=dtype),
        shape=(n_sel_pages, n_sel_tokens, hidden_dim),
        dim_subscripts=dim_subscripts,
        index_views=[
            MemRef(base_ptr=page_ptr, shape=(n_sel_pages,), strides=[1],
                   memory_space="HBM", dtype="i32"),
            MemRef(base_ptr=token_ptr, shape=(n_sel_tokens,), strides=[1],
                   memory_space="HBM", dtype="i32"),
        ],
        variables_space_set=vss, variables_space_order=None,
    )


def build_multi_head_iat(
    ctx: CoreContext,
    n_experts: int,
    n_heads: int,
    M: int,
    N: int,
    n_sel_experts: int,
    n_sel_heads: int,
    dtype: str = "f16",
) -> IndirectAccessTile:
    """weights[expert_idx[e], head_idx[h], m, n] — 2 indirect + 2 direct."""
    hbm = ctx.hbm
    shape = (n_experts, n_heads, M, N)
    data_ptr = _alloc_tensor(hbm, n_experts * n_heads * M * N, dtype)
    expert_ptr = _alloc_index(hbm, n_experts, n_sel_experts)
    head_ptr = _alloc_index(hbm, n_heads, n_sel_heads)

    dim_subscripts = [
        {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
        {"kind": "indirect", "index_view_idx": 1, "idx_exprs": [("dim", 1)]},
        {"kind": "direct", "var_index": 2},
        {"kind": "direct", "var_index": 3},
    ]
    vss = BoxSet(lo=(0, 0, 0, 0), hi=(n_sel_experts, n_sel_heads, M, N))
    return IndirectAccessTile(
        parent_ref=MemRef(base_ptr=data_ptr, shape=shape,
                          strides=_strides_from_shape(shape),
                          memory_space="HBM", dtype=dtype),
        shape=(n_sel_experts, n_sel_heads, M, N),
        dim_subscripts=dim_subscripts,
        index_views=[
            MemRef(base_ptr=expert_ptr, shape=(n_sel_experts,), strides=[1],
                   memory_space="HBM", dtype="i32"),
            MemRef(base_ptr=head_ptr, shape=(n_sel_heads,), strides=[1],
                   memory_space="HBM", dtype="i32"),
        ],
        variables_space_set=vss, variables_space_order=None,
    )


def build_paged_attn_iat(
    ctx: CoreContext,
    n_pages: int,
    n_heads: int,
    block_size: int,
    head_dim: int,
    n_sel_pages: int,
    dtype: str = "f16",
) -> IndirectAccessTile:
    """cache[page_idx[p], heads, block_size, head_dim] — 1 indirect + 3 direct."""
    hbm = ctx.hbm
    shape = (n_pages, n_heads, block_size, head_dim)
    data_ptr = _alloc_tensor(hbm, n_pages * n_heads * block_size * head_dim, dtype)
    page_ptr = _alloc_index(hbm, n_pages, n_sel_pages)

    dim_subscripts = [
        {"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0)]},
        {"kind": "direct", "var_index": 1},
        {"kind": "direct", "var_index": 2},
        {"kind": "direct", "var_index": 3},
    ]
    vss = BoxSet(lo=(0, 0, 0, 0), hi=(n_sel_pages, n_heads, block_size, head_dim))
    return IndirectAccessTile(
        parent_ref=MemRef(base_ptr=data_ptr, shape=shape,
                          strides=_strides_from_shape(shape),
                          memory_space="HBM", dtype=dtype),
        shape=(n_sel_pages, n_heads, block_size, head_dim),
        dim_subscripts=dim_subscripts,
        index_views=[MemRef(base_ptr=page_ptr, shape=(n_sel_pages,), strides=[1],
                            memory_space="HBM", dtype="i32")],
        variables_space_set=vss, variables_space_order=None,
    )


def reset_lx(ctx: CoreContext):
    """Clear LX memory and reset pointer."""
    ctx.lx.memory.clear()
    ctx.lx.next_ptr = 0


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
    workloads: List[Dict[str, Any]]
    raw: Dict[str, Any] = field(default_factory=dict)


def _expand_workload(entry: Dict[str, Any], defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a single [[workloads]] entry into one or more dicts."""
    mode = entry.pop("mode", None)
    list_keys = [k for k, v in entry.items() if isinstance(v, list)]

    if not list_keys:
        return [{**defaults, **entry}]

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
            results.append(row)
        return results

    raise ValueError(f"Unknown mode: {mode!r}. Must be 'product' or 'zip'.")


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
    workloads_raw = raw.get("workloads", [])

    workloads = []
    for entry in workloads_raw:
        entry_copy = dict(entry)
        workloads.extend(_expand_workload(entry_copy, defaults))

    return BenchConfig(
        name=raw.get("name", "benchmark"),
        description=raw.get("description", ""),
        defaults=defaults,
        workloads=workloads,
        raw=raw,
    )
