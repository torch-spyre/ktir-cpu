# Benchmarks: Indirect Memory Access Emulation

Measures the blocked-gather optimization against the general 7-step
inspector-executor path for indirect memory access emulation.

## Named Patterns

The fast path applies to any indirect access with at least one indirect
dimension (looked up via an index tensor) and at least one direct dimension
(the contiguous "block"), where the ratio of total iteration points to unique
index lookups exceeds 16×.

| Pattern | Access expression | Indirect | Direct | Realistic workload |
|---------|------------------|----------|--------|-------------------|
| `moe_ffn` | `W[expert_idx[e], m, n]` | 1 | 2 | MoE feed-forward: select k expert weight matrices from a pool |
| `paged_attn` | `cache[page_idx[p], heads, bs, hd]` | 1 | 3 | Paged KV-cache: gather selected pages for attention decode |
| `sparse_attn` | `cache[page_idx[b], token_idx[t], d]` | 2 | 1 | Sparse attention: select page+token pairs from a large cache |
| `multi_head` | `W[expert_idx[e], head_idx[h], m, n]` | 2 | 2 | Multi-head expert routing: select (expert, head) weight blocks |

## 4-Way Path Comparison

Four distinct implementations compared end-to-end:

| Path | Description | Data movement |
|------|-------------|---------------|
| **P1** Pre-PR-147 elem-wise | 7-step: enumerate → per-point reads → coords → span-read → slice | `mgr.read(span)` then `flat[offsets]` (intermediate alloc) |
| **P2** PR-147 block-path | 3-step: broadcast offsets → direct `mgr.gather()` | `_gather_from`: `data.ravel()[elem_offset + offsets]` |
| **P3** Current elem-wise | 7-step: enumerate → per-point reads → coords → direct gather | Same as P2 at terminal step |
| **P4** Current block-path | 3-step: broadcast offsets → `MemoryOps.load(offsets=...)` | Routes through unified load API, same terminal gather |

### Summary Table

| Workload | Points | P1 pre147 (ms) | P2 orig-blk (ms) | P3 cur-elem (ms) | P4 cur-blk (ms) | Δ P1→P3 | Δ P2→P4 |
|----------|--------|----------------|-------------------|-------------------|-----------------|---------|---------|
| `moe-262K` | 262,144 | 278 | 2.4 | 278 | 2.4 | ~0% | ~0% |
| `moe-1M` | 1,024,000 | 1,101 | 9.4 | 1,129 | 9.2 | ~0% | ~0% |
| `paged-attn-256K` | 262,144 | 290 | 2.6 | 291 | 2.7 | ~0% | ~0% |
| `paged-attn-1M` | 1,048,576 | 1,182 | 10.0 | 1,181 | 10.3 | ~0% | ~0% |
| `sparse-attn-32K` | 32,000 | 57 | 0.5 | 57 | 0.5 | ~0% | ~0% |
| `sparse-attn-128K` | 128,000 | 228 | 1.3 | 228 | 1.4 | ~0% | ~0% |
| `multi-head-512K` | 512,000 | 958 | 5.2 | 948 | 5.1 | ~0% | ~0% |
| `multi-head-4M` | 4,096,000 | 7,663 | 40.7 | 7,696 | 40.4 | ~0% | ~0% |

### Key Findings

1. **Δ P1→P3 ≈ 0%** — The element-wise path is NOT faster after the gather
   change. The terminal gather step (5ms) is <0.1% of total time. The
   bottleneck is steps 2–4: per-point index reads (74%) and coordinate
   building (17%) — pure Python loops that dominate regardless of how the
   final data movement is done.

2. **Δ P2→P4 ≈ 0%** — Routing through `MemoryOps.load(offsets=...)` adds
   zero measurable overhead versus calling `mgr.gather()` directly. The
   refactoring is free.

3. **P3→P4 = 190×** — The block-path speedup is fully preserved. The
   algorithmic difference (numpy broadcast vs per-point Python loop) is
   what matters, not which API the gather routes through.

### Conclusion

The element-wise path's slowness is **algorithmic**, not API-level: O(N)
Python-loop index reads + coordinate assembly. No amount of terminal-step
optimization (gather vs span-read) can fix it — the 190× speedup comes
entirely from collapsing those loops into numpy broadcast. The unified
`MemoryOps.load(offsets=...)` API is structurally cleaner without any
performance cost.

## Step Breakdown (multi-head-4M, 4,096,000 points)

### Path 1: Pre-PR-147 element-wise (span-read)

| Step | Time (ms) | Share |
|------|-----------|-------|
| Enumerate iteration space | 102 | 1.3% |
| Read index tensors | 5,725 | 74.3% |
| Build coordinate list | 1,315 | 17.1% |
| Linearize flat offsets | 554 | 7.2% |
| Span-read + slice | 5.0 | 0.1% |
| Reshape | 0.002 | — |
| Write to LX | 0.004 | — |
| **Total** | **7,701** | |

### Path 2: PR-147 block-path (direct mgr.gather)

| Step | Time (ms) | Share |
|------|-----------|-------|
| Read K index values | 0.10 | 0.2% |
| NumPy broadcast offsets | 10.1 | 25.0% |
| Sticks + gather + LX | 30.3 | 74.8% |
| **Total** | **40.6** | |

### Path 3: Current element-wise (direct gather)

| Step | Time (ms) | Share |
|------|-----------|-------|
| Enumerate iteration space | 100 | 1.3% |
| Read index tensors | 5,750 | 74.3% |
| Build coordinate list | 1,321 | 17.1% |
| Linearize flat offsets | 563 | 7.3% |
| Direct gather | 3.7 | 0.0% |
| Reshape | 0.003 | — |
| Write to LX | 0.004 | — |
| **Total** | **7,738** | |

### Path 4: Current block-path (MemoryOps.load(offsets=...))

| Step | Time (ms) | Share |
|------|-----------|-------|
| Read K index values | 0.10 | 0.2% |
| NumPy broadcast offsets | 10.2 | 25.3% |
| MemoryOps.load(offsets=) | 30.1 | 74.4% |
| **Total** | **40.4** | |

### Speedups

| Comparison | Factor | Interpretation |
|------------|--------|----------------|
| P1 → P3 | 1.00× | Gather change doesn't help (bottleneck elsewhere) |
| P1 → P4 | 190× | Old elem-wise vs new block-path |
| P2 → P4 | 1.00× | No routing overhead from refactoring |
| P3 → P4 | 191× | Current elem-wise vs current block-path |

## Usage

```bash
cd ktir-cpu

# 4-way path evolution comparison (summary + step breakdown)
uv run python benchmarks/bench_indirect_emul_time.py 4way

# Original 2-way: fast-path vs general (summary + breakdown)
uv run python benchmarks/bench_indirect_emul_time.py block

# Gather memcpy optimization (3-copy vs 1-copy)
uv run python benchmarks/bench_indirect_emul_time.py gather

# Custom config
uv run python benchmarks/bench_indirect_emul_time.py 4way --config configs/custom.toml
```

## Config

`configs/indirect_emul.toml` defines workloads with a `pattern` field that
selects the IAT builder. Each entry specifies pattern-specific parameters
(expert counts, page sizes, head dims, etc.). Size strings like `"2K"` are
auto-parsed. List fields require explicit `mode = "product"` or `"zip"`.

## Code Structure

```
bench_indirect_emul_time.py   — CLI entry point (block/gather/4way subcommands)
bench_utils.py                — BenchTimer, BenchTable, IAT builders, config loader
configs/indirect_emul.toml    — workload definitions (4 patterns × 2 sizes)
```
