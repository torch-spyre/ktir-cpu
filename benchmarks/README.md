# Indirect-Load Path Benchmark

## What it measures

For each workload, the benchmark reports three timing values:

| Metric | What it captures |
|--------|-----------------|
| **T_indirect_load** | Full `MemoryOps.indirect_load(ctx, iat)` — end-to-end |
| **T_load** | Time spent inside `MemoryOps.load()` (data movement only) |
| **T_offsets** | `T_indirect_load − T_load` — the indexication overhead |

The benchmark is **implementation-agnostic**: it works on any branch that has
`indirect_load` calling `MemoryOps.load` internally. Run the same script on
different branches to compare implementations.

## Usage

```bash
cd benchmarks/
uv run python bench_indirect_load_path_compare.py
uv run python bench_indirect_load_path_compare.py --config configs/load_path_compare.toml
```

## Workload configuration

Workloads are defined in TOML files under `configs/`. Each entry specifies:

```toml
[[workloads]]
label = "moe-262K"
pattern = "moe_ffn"                # dispatches to build_moe_iat
access_expr = "X[IDX[e], m, n]"    # for display
model_scale_ref = "deepseek-v2"    # which LLM motivates this shape
num_experts = 128
M = 256
N = 128
n_selected = 8
```

Supported patterns: `moe_ffn`, `paged_attn`, `sparse_attn`, `multi_head`.

The output table includes a **Source shape** column (from `iat.parent_ref.shape`)
showing the full allocation shape that the indirect load selects from.

## Design: monkeypatch instrumentation

The benchmark needs to time `MemoryOps.load` *as called from within*
`indirect_load`, without importing any internal helper functions. This keeps
the benchmark decoupled from implementation details.

**Approach:** temporarily replace `MemoryOps.load` with a timing wrapper via
a context manager (`LoadTimer`):

```
┌─────────────────────────────────────────────────┐
│  with LoadTimer() as lt:                        │
│      indirect_load(ctx, iat)                    │
│          ↓                                      │
│      [indexication work]                        │
│          ↓                                      │
│      MemoryOps.load(...)  ← hits timed wrapper  │
│          ↓                                      │
│      wrapper records elapsed, calls original    │
│                                                 │
│  lt.elapsed = time inside load()                │
└─────────────────────────────────────────────────┘
```

The wrapper:
1. Saves the original `MemoryOps.load` reference
2. Replaces it with a function that records `time.perf_counter()` around the
   original call
3. Restores the original on context-manager exit

This means:
- **No internal imports** — only the public `MemoryOps` class is used
- **Any implementation works** — whether `indirect_load` calls
  `load(offsets=...)`, `load(coords=...)`, or any future variant
- **Accurate attribution** — T_load captures exactly what `load` costs,
  T_offsets captures everything else (index reads, broadcast, analysis)

### Limitation

If an implementation bypasses `MemoryOps.load` entirely (e.g., calls
`mgr.gather` directly), then `T_load = 0` and `T_offsets = T_indirect_load`.
This is still informative — it means the implementation owns the full pipeline.
