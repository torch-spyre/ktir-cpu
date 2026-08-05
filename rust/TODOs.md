# Rust port — deferred work (TODOs)

This file tracks work that is intentionally **not yet ported** from the Python
`ktir_cpu` reference into the Rust port. Each item explains what is missing, why
it is deferred, and what a port would entail, so it can be picked up later
without re-deriving the analysis.

---

## 1. Inter-tile reduce — full four-op port (DEFERRED) 🅿️

**Upstream:** `c428844` — *[Experimental] Inter tile reduce* (#72).
**Tracks:** the still-**unmerged** frontend spec `ktir-mlir-frontend#23`.

### What it is
Upstream replaced the legacy single-op `ktdp.reduce` ring all-reduce with an
experimental **four-op** inter-tile design:

- `ktdp.inter_tile_produce` — produces a per-core partial tensor for a group
  (region terminated by `ktdp.yield_partial`), yielding a `!ktdp.tile_future`.
- `ktdp.inter_tile_reduce` — consumes the future + an identity operand and runs
  the combiner (region terminated by `ktdp.yield_reduced`) across the group.

`examples/ktir/ring_reduce.mlir` was rewritten to this surface and
`examples/latency/ring_reduce_multi_group.mlir` was added (both come in via the
rebase onto `origin/main`). The Python side fully implements and tests them.

### Why deferred
The op names, attributes, and `!ktdp.tile_future` type are explicitly
experimental and **may shift** when `ktir-mlir-frontend#23` is finalized.
Porting now means chasing a moving target; the cost/benefit favors waiting for
the spec to stabilize. This is a *feature*, not a spec-compliance/correctness
bug fix, so it is not required for parity on the current corpus.

### Current Rust state (legacy path still works)
The port still implements the **legacy** `ktdp.reduce` ring all-reduce, which is
correct and tested:
- `crates/ktir-emulator/src/comm_sched.rs` — `is_comm_op` matches only
  `"ktdp.reduce"`; `RingReduce` rings over the operand `core_group` with a fixed
  `tile_add` combiner.
- `crates/ktir-emulator/src/latency.rs` — the `Comm` branch adds a
  `log2(num_cores)` rounds factor gated on `op_type == "ktdp.reduce"`.
- `crates/ktir-emulator/tests/port_grid_scheduler.rs` — exercises `ktdp.reduce`
  via inline IR (does **not** depend on the example file), so it stays green.

### How the gap is currently contained (so the branch is green & honest)
- `crates/ktir-emulator/tests/dispatch_coverage.rs` lists the three unhandled
  ops (`ktdp.inter_tile_produce`, `ktdp.inter_tile_reduce`, `ktdp.yield_reduced`)
  in `KNOWN_GAP_OPS` — the existing not-yet-ported burn-down mechanism.
- `crates/ktir-emulator/tests/port_examples.rs::ring_reduce_sum` is `#[ignore]`d.

`linalg.add` (the one non-experimental compute op c428844 touched) is **already
ported** — see `crates/ktir-emulator/src/dialects/linalg.rs`.

### What a port would entail (plan to validate against the Python reference)
Reference: `ktir_cpu/ops/comm_ops.py`, `ktir_cpu/dialects/ktdp_ops.py`,
`ktir_cpu/ir_types.py`, `ktir_cpu/grid.py`, `ktir_cpu/interpreter.py`,
`ktir_cpu/latency.py`. Do these together so the parser/scheduler/latency
contracts stay coherent:

1. **Value/type model** (`crates/ktir-core/src/ir.rs`, `tile.rs`): add a
   `TileFuture` value variant (partial tensor types, local partial, producer/
   group sets, group index) and a `comm_bytes: Option<u64>` field on `Tile`
   (not propagated by `clone`/`compute`, mirroring Python `copy()`).
2. **Context** (`crates/ktir-emulator/src/machine_state/context.rs`): expose
   `num_cores` on `CoreContext` (the scheduler already knows `grid.num_cores`).
3. **Parser** (`crates/ktir-core/src/parser.rs`): the parser already *structurally*
   parses the rewritten example (the ops surface as op-types), but verify it
   captures the `!ktdp.tile_future<...>` result type, the `produce`/`reduce`
   regions, and the identity operand correctly; add bare `key = value` attr
   extraction if needed.
4. **Comm plan** (`comm.rs` / `comm_sched.rs`): add a `CommPlan`
   (producers/consumers/deps) and rewrite `RingReduce` to take plan + identity +
   combiner instead of a raw `core_group`, seed identity for non-producers, fold
   only producer contributions, return `None` for non-consumers, accumulate
   `bytes_moved`, and stamp `result.comm_bytes`. Extend `is_comm_op` to
   `ktdp.inter_tile_reduce`.
5. **Dialect handlers** (`crates/ktir-emulator/src/dialects/ktdp.rs`):
   `inter_tile_produce` (resolve group, run producer region, build `TileFuture`)
   and `inter_tile_reduce` (build `CommPlan`, run combiner region as the reduce
   fn, drive the backend, reshape result).
6. **Latency** (`latency.rs`): read `comm_bytes` off the result tile and **drop**
   the `log2(num_cores)` rounds multiplier (Python removed it for the new path).
7. **Interpreter** (`interpreter.rs`): defer the comm latency record until the
   scheduler-driven op's final tile (with `comm_bytes`) is known.
8. **Tests**: un-ignore/rewrite `port_examples.rs::ring_reduce_sum` to *execute*
   the new example; port `TestRingReduceLatency` /
   `TestRingReduceMultiGroupLatency`; remove the three ops from `KNOWN_GAP_OPS`.

**Acceptance:** `dispatch_coverage::ring_reduce` passes with an empty (or shrunk)
`KNOWN_GAP_OPS`; `ring_reduce.mlir` and `ring_reduce_multi_group.mlir` execute
and match the Python golden; latency tests match the per-core `comm_bytes` model.

---

## 2. Per-unit roofline latency model (OPTIONAL, low priority)

**Upstream:** `7f8bc83` — *feat: per-unit roofline model + latency demo notebook* (#107).

Python's `roofline()` now reports **per-unit** ceilings (systolic vs SIMD) plus a
`dominant_unit`, replacing the old single-SIMD-ceiling flat model. The Rust port
(`crates/ktir-emulator/src/latency.rs`) keeps the **old flat model**.

This is a **reporting/observability** change only — it is **not** an RFC-0682
obligation and does **not** affect any correctness-bearing output: total cycles,
`kernel_time_us`, `bottleneck`, and `per_core_summary` are numerically identical
either way (compute-category FLOP sums are unchanged). Port only if a consumer
needs to cross-check `roofline()` efficiency/`dominant_unit` against Python.

Sketch: split `CoreLatencyCounters` compute scalars into
`flops_by_category`/`cycles_by_category` maps; thread the specific
`LatencyCategory` into `record()`; rewrite `struct Roofline` + `roofline()` to
emit per-unit `{systolic, simd}` ceilings and a `dominant_unit`.

---

## 3. Reject IR missing a required `access_tile_set` (OPTIONAL, minor)

**Upstream:** `7fa20ca` — *[Refactor] Route ktdp.load/store through `_subtile_ref`* (#79).

The refactor itself is behavior-preserving and **already matches** in Rust (the
port reaches the same contiguous fast path via the `coordinate_set = None`
sentinel). The one genuinely new behavior is that the Python parser now **raises**
when `access_tile_set` is absent (it is required per ODS). Rust currently treats
an absent `access_tile_set` as a contiguous full-tile load instead of erroring
(`crates/ktir-core/src/parser.rs`, `parse_construct_access_tile_attrs`). This is
invalid-IR hardening, not a divergence on any valid emitted IR — low value.
