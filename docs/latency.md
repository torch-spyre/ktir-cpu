# Latency Estimation

The interpreter can estimate cycle-approximate execution latency on Spyre hardware without access to real hardware. This is **opt-in** — pass a `HardwareConfig` to enable it. When disabled (the default), there is zero overhead and existing behavior is unchanged.

```python
interp = KTIRInterpreter(latency_config=HardwareConfig())
interp.load("examples/triton-ktir/vector_add_ktir.mlir")
interp.execute_function("add_kernel", x_ptr=x, y_ptr=y, output_ptr=out)

report = interp.get_latency_report()
print(report)
print(report.bottleneck)      # "memory", "compute", or "comm"
print(report.kernel_time_us)  # estimated wall time in microseconds
```

## How LatencyTracker integrates into KTIRInterpreter

The tracker is designed as **post-execution instrumentation** — it observes each operation after it runs, never altering execution. Here is the integration flow:

```
KTIRInterpreter.__init__(latency_config=HardwareConfig())
    │
    │  Stores config; tracker is not yet created (num_cores unknown)
    ▼
execute_function("kernel_name", **inputs)
    │
    │  Allocates HBM, then for each core calls _execute_operation() per op
    ▼
_execute_operation(op, context, env)
    │
    │  1. Resolve operand SSA names → values (cheap dict lookups on context.values)
    │     Only done when tracker is present; stored in resolved_operands list
    │
    │  2. Dispatch to handler via registry:
    │     handler = dispatch(op.op_type)  →  looks up _REGISTRY dict
    │     result = handler(op, context, env)
    │
    │  3. Store result in context (context.set_value)
    │
    │  4. Record latency:
    │     tracker.record_op(core_id, op.op_type, result, resolved_operands)
    │         │
    │         ├─ Look up category via get_latency_category(op_type)
    │         ├─ Estimate cycles from result/operand shapes + HardwareConfig
    │         └─ Accumulate into CoreLatencyCounters[core_id]
    ▼
get_latency_report() → LatencyReport
    │
    │  kernel_cycles = max(core.total_cycles for each core)
    │  bottleneck = category with most cycles on critical-path core
```

Key design points:

- **Registry-based dispatch**: Operations are dispatched through a central registry (`dialects/registry.py`). Each dialect module registers its handlers and latency categories at import time via `@register()`.
- **Post-execution recording**: The tracker runs *after* each op executes, so the result shape is known for accurate size estimation. For direct loads, `ktdp.load` cost is the result Tile's `nbytes`. For indirect (gather) loads the index tensors are also counted — see [Indirect access memory cost](#indirect-access-memory-cost).
- **Operand resolution before the try block**: Operands are resolved once into a flat list before the operation's try block. This avoids duplicating resolution logic inside every op branch while keeping the cost negligible (dict lookups on already-populated SSA maps).
- **Control flow is zero-cost**: `scf.for`, `scf.if`, and `scf.yield` are classified as zero-cost. Child operations inside loops/branches record their own costs naturally via recursive `execute_region` calls, so loop iterations accumulate correctly without the tracker needing to understand control flow.
- **Memory-space-aware**: LX (on-chip SRAM) load/store is zero-cost. Only HBM transfers incur latency. This reflects the hardware: all live Tiles reside in LX, so an LX load/store is redundant with SSA dataflow — it's just an on-chip copy.
- **Backward compatible**: `get_latency_report()` is a new method returning `Optional[LatencyReport]`. The return type of `execute_function` is unchanged. Existing callers are unaffected.

## Cycle model

Sequential within each core — no overlap between compute, memory, and communication:

```
core_total = compute_cycles + memory_cycles + comm_cycles
kernel_latency = max(core_total for all cores)
```

This is a v1 simplification. Real hardware pipelines these stages, so actual latency would be lower. The model is useful for identifying bottleneck categories and comparing kernels relatively.

## Operation cost classification

Each dialect module declares its latency category at registration time via the `@register()` decorator and the `LatencyCategory` enum defined in `latency.py`. Adding a new op only requires annotating its handler — no separate classification list to maintain.

```python
from ..latency import LatencyCategory as LC
from .registry import register

@register("arith.addf", latency_category=LC.COMPUTE_FLOAT)
def arith__addf(op, context, env):
    ...
```

| Category | `LatencyCategory` | Cost formula |
|----------|-------------------|-------------|
| **Zero-cost** | `ZERO` (default) | 0 cycles |
| **Memory (HBM)** | `MEMORY` | `bytes / hbm_bytes_per_cycle_per_core` |
| **Memory (LX)** | `MEMORY` | 0 cycles (on-chip, negligible) |
| **Compute (SIMD)** | `COMPUTE_FLOAT` | `elements / simd_elements_per_cycle` |
| **Compute (transcendental)** | `COMPUTE_TRANSCENDENTAL` | `elements / simd_elements_per_cycle × transcendental_penalty` |
| **Compute (matmul)** | `COMPUTE_MATMUL` | `2·M·N·K / systolic_flops_per_cycle` |
| **Compute (integer)** | `COMPUTE_INT` | 1 cycle (scalar) or `elements / simd_elements_per_cycle` (tile) |
| **Communication** | `COMM` | `bytes / ring_bytes_per_cycle` (`ktdp.reduce` adds `× log2(num_cores)`) |

## Indirect access memory cost

A `ktdp.load` over an `IndirectAccessTile` with N indirect dimensions performs N+1 HBM reads. The tracker counts all HBM-resident transfers; LX-resident index views are free (on-chip):

```
memory_bytes = result_tile_bytes + Σ(i=1..N) index_view_i_bytes   (HBM views only)
```

For `indirect-access-copy.mlir` (64×64 gather, all HBM):

```
X    (f16,  64×64): 64×64 × 2 =  8,192 bytes   ← data tensor
IDX1 (i32, 64×64): 64×64 × 4 = 16,384 bytes   ┐
IDX2 (i32, 64×64): 64×64 × 4 = 16,384 bytes   ┘ N=2 index tensors
                               ─────────────
Total:                          40,960 bytes
```

**Open question**: the model applies `hbm_bandwidth_tb_s` uniformly across all dtypes. Whether f16 data tensors and i32 index tensors achieve the same effective HBM bandwidth on Spyre is unconfirmed — if they differ, a separate bandwidth parameter would be needed.

## Hardware parameters

Default `HardwareConfig()` values:

| Parameter | Default | Source |
|-----------|---------|--------|
| `num_cores` | 32 | All example kernels use `grid = [32, 1]` |
| `clock_ghz` | 1.0 | Convenience (1 cycle = 1 ns). Not a real clock frequency |
| `hbm_bandwidth_tb_s` | 1.0 TB/s | **Estimated** — typical HBM2-class; actual Spyre BW unknown |
| `ring_bandwidth_tb_s` | 4.0 TB/s | `comm_ops.py` docstring |
| `simd_elements_per_cycle` | 64 (1 stick of f16) | **Estimated** — one 128-byte stick per cycle |
| `systolic_flops_per_cycle` | 524,288 (2×64³) | **Estimated** — see [Systolic array model](#systolic-array-model) below |
| `transcendental_penalty` | 4× | **Estimated** — common heuristic for exp/sqrt vs add/mul |

Derived properties:
- `hbm_bytes_per_cycle_per_core` = `hbm_bandwidth_tb_s × 1e12 / (clock_ghz × 1e9) / num_cores` → 31.25 bytes/cycle at defaults
- `ring_bytes_per_cycle` = `ring_bandwidth_tb_s × 1e12 / (clock_ghz × 1e9)` → 4000 bytes/cycle at defaults

Override any parameter:

```python
cfg = HardwareConfig(hbm_bandwidth_tb_s=2.0, num_cores=8)
interp = KTIRInterpreter(latency_config=cfg)
```

## Systolic array model

The `systolic_flops_per_cycle` parameter models the throughput of a [systolic array](https://en.wikipedia.org/wiki/Systolic_array) — a grid of processing elements (PEs) that perform matrix multiply in lock-step. Data flows through the array rhythmically: one matrix streams left-to-right, the other top-to-bottom, so each PE reuses data from its neighbors without going back to memory.

For an N×N systolic array:

```
Each PE does 1 multiply-add per cycle = 2 FLOPs
N×N PEs × 2 FLOPs/PE = 2·N² FLOPs per outer-product step
Over K steps (one per column of A): 2·N²·K total FLOPs
```

The default assumes a 64×64 array processing a 64-deep K dimension in a pipelined fashion:

```
2 × 64 × 64 = 8,192 FLOPs/cycle per step
× 64 K-steps pipelined
= 524,288 effective FLOPs/cycle
```

A `linalg.matmul` with dimensions M×N×K costs `2·M·N·K / systolic_flops_per_cycle` cycles. For the default 64×64×64 matmul: `2 × 64³ / 524,288 = 1.0` cycle.

The actual Spyre systolic array dimensions are not publicly documented. If the real array is 32×32, set `systolic_flops_per_cycle = 2 * 32 * 32 * 32` (= 65,536), which would make the same 64×64×64 matmul cost 8 cycles instead of 1.

## Example: sanity-checking vector_add

For 1 core, 1024 f16 elements (2048 bytes per tensor):

```
2 loads  × 2048 bytes / 31.25 bytes/cycle = 131 memory cycles
1 store  × 2048 bytes / 31.25 bytes/cycle =  66 memory cycles
1 addf   × 1024 elems / 64 elems/cycle    =  16 compute cycles
                                             ───
Total ≈ 213 cycles, bottleneck = memory
```

The vector_add MLIR has 8 operations per core. With `trace_latency=True`, each one is recorded:

```
Op                                       Category      Cycles
--------------------------------------------------------------
ktdp.construct_memory_view               zero             0.0
ktdp.load                                memory          65.5
ktdp.construct_memory_view               zero             0.0
ktdp.load                                memory          65.5
arith.addf                               compute         16.0
ktdp.construct_memory_view               zero             0.0
ktdp.store                               memory          65.5
return                                   zero             0.0
--------------------------------------------------------------
TOTAL                                                   212.6
  compute: 16.0   memory: 196.6   comm: 0.0
```

Step by step:

1. **`ktdp.construct_memory_view %x_ptr`** — Creates a `TileRef` descriptor (base pointer, shape `[1024]`, strides, memory space). No data moves. Zero-cost.
2. **`ktdp.load %x_view_1`** — Reads 1024 × f16 = 2048 bytes from HBM into a `Tile`. The tracker sees the result Tile has `nbytes=2048`. Cost = `2048 / 31.25 = 65.5 memory cycles`.
3. **`ktdp.construct_memory_view %y_ptr`** — Another descriptor. Zero-cost.
4. **`ktdp.load %y_view_2`** — Same as step 2: 65.5 memory cycles.
5. **`arith.addf %x_24, %y_26`** — NumPy adds two `(1024,)` f16 arrays. Cost = `1024 / 64 = 16 compute cycles`.
6. **`ktdp.construct_memory_view %output_ptr`** — Descriptor. Zero-cost.
7. **`ktdp.store %output, %output_view_3`** — Writes 2048 bytes to HBM. The tracker looks at the first Tile operand (the stored value). Cost = `2048 / 31.25 = 65.5 memory cycles`.
8. **`return`** — Zero-cost.

Memory is 92% of total → `bottleneck = "memory"`.

**Multi-core execution.** With `grid = [32, 1]` and 32×1024 total elements, each core processes its own 1024-element chunk (the MLIR's `construct_memory_view sizes: [1024]` defines the per-core tile size). Every core runs the same 8 operations independently, so all 32 cores report ~213 cycles each. `kernel_cycles = max(core.total_cycles)` stays 213 — the parallelism shows up as 32× higher throughput (32K elements in 213 cycles), not lower per-core latency. The per-core HBM bandwidth (`31.25 bytes/cycle`) reflects 32 cores sharing the 1 TB/s bus, which is a hardware property independent of how many cores a given kernel activates.

## Modeling assumption tests

`tests/test_latency_modeling.py` verifies the cycle model empirically by running small kernels with varied inputs and asserting that measured cycle counts match the analytical formulas above.

| Test | What is varied | Expected observation |
|---|---|---|
| `test_shared_bus_bandwidth_penalty` | `num_cores` (1–32), fixed 128-elem tile | `compute_cycles` constant; `memory_cycles` grows ∝ `num_cores` (shared HBM bus) |
| `test_work_splitting_elementwise` | `num_cores` (1–8), fixed total=128 elems | `compute_cycles ∝ 1/num_cores`; `memory_cycles` constant (tile↓ and bw_pc↓ cancel) |
| `test_work_splitting_matmul` | `grid_x` (2–8), `BLOCK_SIZE_M` halved each doubling | `compute_cycles ∝ 1/scale`; `memory_cycles` matches analytical A+B+C byte count |
| `test_work_splitting_transcendental` | `num_cores` (1–8), fixed total=128 elems | `compute_cycles ∝ 1/num_cores`; `memory_cycles` constant (same cancellation) |
| `test_tile_size_memory_cycles` | `tile_size` (64–512) | `memory_cycles ∝ tile_size` |
| `test_lx_ops_zero_cycles` | `memory_space` patched to LX | `memory_cycles == 0` |
| `test_lx_reuse_vs_hbm_reload` | LX vs HBM inline kernel | `memory_cycles(LX) == 0 < memory_cycles(HBM)` |
| `test_balanced_work_distribution` | 32-core add_kernel | `max(total_cycles) == min(total_cycles)` |

IR patching helpers (`_patch_grid`, `_patch_tile_size`, `_patch_tile_dim0`, `_patch_memory_space`) mutate parsed `op.attributes` and `func.grid` before execution so scaling laws can be tested without separate MLIR files.

## Roofline analysis

The report includes a roofline model that bounds achievable performance by two hardware ceilings:

```
GFLOP/s
  ^
  |         peak_gflops
  |        .-------------------  compute ceiling
  |       /
  |      /    * achieved
  |     /
  |    /
  |   /  BW ceiling = peak_bw × AI
  |  /
  | /
  +-----------------------------------> AI (FLOP/B)
             ^
        ridge_point
```

- **BW ceiling** (the slope): `peak_bw × AI`. When a kernel has low arithmetic intensity it cannot feed the compute units fast enough — performance is limited by memory bandwidth.
- **Compute ceiling** (the flat top): `peak_gflops`. Once AI is high enough, the compute units are fully utilized.
- **Ridge point**: `peak_gflops / peak_bw`. Left of it the kernel is memory-bound; right of it, compute-bound.
- **Efficiency**: `achieved / ceiling` — how close the kernel gets to the roofline at its operating point.

Access roofline metrics via `report.roofline()`:

```python
report = interp.get_latency_report()
rf = report.roofline()
print(rf["arithmetic_intensity"])  # FLOP/B
print(rf["efficiency"])            # 0..1
```

> **Note:** The roofline model only covers compute and HBM bandwidth. Communication cycles (ring allgather/reduce) are not modelled. For comm-dominated kernels, `report.bottleneck` may report `"comm"` while the roofline classifies based on the compute-vs-HBM ratio alone. For `"compute"` or `"memory"` bottlenecks, the roofline bound always agrees with `bottleneck`.

### Example: vector_add (memory-bound)

Continuing the vector_add example above (1 core, 1024 f16 elements):

```
FLOPs:  1024  (one addf per element)
Bytes:  6144  (2 loads + 1 store × 2048 bytes each)

AI = 1024 / 6144 = 0.17 FLOP/B

Peak compute = 64 elems/cycle × 1 GHz = 64 GFLOP/s
Peak BW      = 31.25 bytes/cycle × 1 GHz = 31.25 GB/s
Ridge point  = 64 / 31.25 = 2.05 FLOP/B

AI (0.17) << ridge point (2.05)  →  memory-bound

Ceiling     = peak_bw × AI = 31.25 × 0.17 = 5.21 GFLOP/s
Achieved    = 1024 FLOPs / 212.6 cycles × 1 GHz = 4.82 GFLOP/s
Efficiency  = 4.82 / 5.21 = 92.5%
```

The kernel is deep in the memory-bound region. The 92.5% efficiency means it nearly saturates the bandwidth ceiling — the small gap comes from compute cycles (16 out of 213 total) that aren't overlapped with memory.

### Example: softmax (compute-bound at 1 core)

Softmax with 1 core, 1024×1024 f16 matrix. The dominant cost is `math.exp` (transcendental penalty 4×):

```
FLOPs:  ~5.2M  (exp + sub + div + reduces over 1024 rows × 1024 cols)
Bytes:  ~4.2K  (1 load + 1 store per row, each 2048 bytes — small
                relative to compute because tiles stay in LX between ops)

AI = 5.2M / 4.2K ≈ 1250 FLOP/B

Ridge point = 2.05 FLOP/B

AI (1250) >> ridge point (2.05)  →  compute-bound

Ceiling    = peak_gflops = 64 GFLOP/s
Achieved   ≈ 38.8 GFLOP/s
Efficiency ≈ 60.6%
```

The AI is far above the ridge point because the bulk of the work (exp, subtract, divide, reduce) happens on tiles already in LX scratchpad — only the initial load and final store touch HBM. The 60% efficiency reflects that transcendental ops take 4× longer per element than the SIMD peak assumes.
