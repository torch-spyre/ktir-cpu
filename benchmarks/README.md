# Benchmarks: Block-Gather Fast Path

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

## Summary

All patterns show 100–190× end-to-end speedup over the general path.
The blocked-gather replaces 7 sequential steps with 3 steps whose cost is
dominated by a single NumPy broadcast (offset computation) and a flat gather.

| Pattern | Points | General (ms) | Fast (ms) | Speedup |
|---------|--------|--------------|-----------|---------|
| `moe_ffn` | 1,024,000 | 1,084 | 9.4 | 115× |
| `paged_attn` | 1,048,576 | 1,176 | 10.5 | 112× |
| `sparse_attn` | 128,000 | 227 | 1.2 | 190× |
| `multi_head` | 4,096,000 | 7,745 | 40.5 | 191× |

## Step Breakdown by Pattern

### `moe_ffn` — MoE feed-forward weight selection

1M points (256 experts × 64 × 2K, selecting 8).

The general path spends 67% of its time resolving index tensor reads
element-by-element. The fast path reads all 8 index values in one call,
then computes offsets via a single broadcast multiply-add.

**General path (7 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Enumerate iteration space | 23.7 | 2.1% |
| Read index tensors | 765.5 | 67.1% |
| Build coordinate list | 230.9 | 20.2% |
| Linearize flat offsets | 120.2 | 10.5% |
| Gather from HBM | 1.2 | 0.1% |
| Reshape | 0.003 | — |
| Write to LX | 0.004 | — |
| **Total** | **1,141** | |

**Fast path (3 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Read K index values | 0.023 | 0.7% |
| NumPy broadcast offsets | 2.12 | 68.6% |
| Gather + reshape + LX | 0.95 | 30.7% |
| **Total** | **3.1** | |

**Step-level speedup: 369×**

---

### `paged_attn` — Paged KV-cache page gather

1M points (256 pages × 16 heads × 16 block_size × 128 head_dim, selecting 32 pages).

Same bottleneck structure as MoE. The 3-direct-dim block (heads × bs × hd = 32K
elements per page) gives high reuse per index lookup.

**General path (7 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Enumerate iteration space | 23.5 | 2.0% |
| Read index tensors | 758.1 | 63.5% |
| Build coordinate list | 266.6 | 22.3% |
| Linearize flat offsets | 143.8 | 12.1% |
| Gather from HBM | 1.0 | 0.1% |
| Reshape | 0.002 | — |
| Write to LX | 0.003 | — |
| **Total** | **1,193** | |

**Fast path (3 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Read K index values | 0.038 | 1.2% |
| NumPy broadcast offsets | 2.19 | 71.7% |
| Gather + reshape + LX | 0.82 | 27.0% |
| **Total** | **3.0** | |

**Step-level speedup: 391×**

---

### `sparse_attn` — Sparse attention page+token selection

128K points (128 pages × 64 tokens × 2K hidden, selecting 8 pages × 8 tokens).

Two indirect dimensions multiply the general path's per-element index resolution
cost. The fast path reads each index tensor once (8 + 8 = 16 values total), then
broadcasts both into a combined offset grid.

**General path (7 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Enumerate iteration space | 2.3 | 1.0% |
| Read index tensors | 174.7 | 76.6% |
| Build coordinate list | 36.2 | 15.9% |
| Linearize flat offsets | 14.7 | 6.4% |
| Gather from HBM | 0.08 | — |
| Reshape | 0.002 | — |
| Write to LX | 0.002 | — |
| **Total** | **228** | |

**Fast path (3 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Read K index values | 0.11 | 27.8% |
| NumPy broadcast offsets | 0.21 | 52.9% |
| Gather + reshape + LX | 0.08 | 19.3% |
| **Total** | **0.41** | |

**Step-level speedup: 563×**

---

### `multi_head` — Multi-head expert weight routing

4M points (32 experts × 8 heads × 64 × 2K, selecting 8 experts × 4 heads).

Largest workload: the general path takes nearly 8 seconds due to 5.8s spent
resolving index reads across 4M points. The fast path reads 8 + 4 = 12 index
values and computes the full 4M-element offset grid in under 5 ms.

**General path (7 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Enumerate iteration space | 95.5 | 1.2% |
| Read index tensors | 5,822 | 74.7% |
| Build coordinate list | 1,328 | 17.0% |
| Linearize flat offsets | 544.5 | 7.0% |
| Gather from HBM | 3.9 | 0.1% |
| Reshape | 0.003 | — |
| Write to LX | 0.004 | — |
| **Total** | **7,794** | |

**Fast path (3 steps):**

| Step | Time (ms) | Share |
|------|-----------|-------|
| Read K index values | 0.076 | 1.1% |
| NumPy broadcast offsets | 4.54 | 63.2% |
| Gather + reshape + LX | 2.57 | 35.8% |
| **Total** | **7.2** | |

**Step-level speedup: 1,084×**

## Usage

```bash
cd ktir-cpu

# Summary table + step breakdown (all patterns)
uv run python benchmarks/bench_indirect_emul_time.py block

# Gather memcpy optimization (3-copy vs 1-copy)
uv run python benchmarks/bench_indirect_emul_time.py gather

# Custom config
uv run python benchmarks/bench_indirect_emul_time.py block --config configs/custom.toml
```

## Config

`configs/indirect_emul.toml` defines workloads with a `pattern` field that
selects the IAT builder. Each entry specifies pattern-specific parameters
(expert counts, page sizes, head dims, etc.). Size strings like `"2K"` are
auto-parsed. List fields require explicit `mode = "product"` or `"zip"`.

## Code Structure

```
bench_indirect_emul_time.py   — CLI entry point (block/gather subcommands)
bench_utils.py                — BenchTimer, BenchTable, IAT builders, config loader
configs/indirect_emul.toml    — workload definitions (4 patterns × 2 sizes)
```
