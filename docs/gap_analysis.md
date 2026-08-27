# KTIR Spec RFC vs. `ktir_cpu` Implementation — Gap Analysis

**Date**: 2026-05-30
**Spec**: [RFC 0682 — KTIR Spec](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)

**Legend**: ✅ implemented — 🟡 partial — ❌ not implemented — 🧪 experimental (tracks an unmerged upstream spec PR; semantics may change) — ⚠️ spec conflict (implementation intentionally diverges from the RFC; the RFC needs updating)

---

## A. `ktdp` Dialect Operations

| # | Spec Operation | Status | Notes |
|---|---------------|--------|-------|
| 1 | `ktdp.construct_distributed_memory_view` | ✅ | Handler and parser implemented in `ktir_cpu/dialects/ktdp_ops.py`; produces `DistributedMemRef` (composition of N per-partition `MemRef`s). Per-partition routing at access time via `MemoryOps.distributed_tile_access` → `DistributedTileRef`. Tests in `tests/test_distributed_view.py`. |
| 2 | `ktdp.construct_indirect_access_tile` | ✅ | Handler and parser implemented in `ktir_cpu/dialects/ktdp_ops.py`; tests passing in `tests/test_indirect_access.py`. Both `ktdp.load` (gather, via `MemoryOps.indirect_load`) and `ktdp.store` (scatter, via `MemoryOps.indirect_store`) accept `IndirectAccessTile` (#44 closed). |
| 2a | `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` (four-op design, reduce path) | 🧪 experimental | The four-op design lives in [ktir-mlir-frontend PR #23](https://github.com/torch-spyre/ktir-mlir-frontend/pull/23). **One divergence stands:** `ktdp.inter_tile_reduce` here takes its result type as a reshape target (`_result_shape` → `attach_reshape`), while the dialect verifies `result types must match future partial types`, so a reduce that reshapes cannot be written in the dialect's own assembly. `examples/ktir/ring_reduce.mlir`, `examples/ktir/ring_reduce_inner_loop.mlir` and `examples/latency/ring_reduce_multi_group.mlir` reduce `tensor<1x128xf16>` to `tensor<128xf16>` — the only non-identity reduce reshape here — and are read by the regex parser only; the other three inter-tile kernels reduce shape-to-shape and the frontend accepts them. Implementation: handlers and parsers in `ktir_cpu/dialects/ktdp_ops.py`; `TileFuture` per-core handle binds the local partial and a not-yet-running `RingReduceBackend`; reduce delivery triggers the backend with the combiner region as `reduce_fn`. Per-core wire bytes flow to the latency tracker via `Tile.comm_bytes` (mirrors `unique_sticks`). End-to-end tests: `tests/test_examples.py::TestRingReduceExecution` (contiguous reduce groups) and `tests/test_examples.py::TestMulticoreSdpaExecution` (strided reduce groups, 32 cores). Replaces the legacy `ktdp.reduce` / `ktdp.transfer` ops. See `docs/cross_core_scheduling.md`. |
| 2b | `ktdp.inter_tile_consume` (broadcast delivery) | ❌ experimental | Same upstream PR. Not implemented. |
| 2c | `ktdp.inter_tile_reduce_scatter` | ❌ experimental | Same upstream PR. Not implemented. |
| 2d | `producer_dependency_per_consumer` (per-tile sync) | 🟡 experimental | Same upstream PR. Parsed and stored on the delivery op AST; runtime is full-barrier mode (waits for all producers). Per-tile mode unimplemented. |

## B. `ktdp` Types & Attributes

| # | Spec Item | Status | Notes |
|---|-----------|--------|-------|
| 3 | `AccessTileType` with dynamic dimensions (`?`) | ❌ | The spec allows `access_tile<? x 64 x index>` (partially/fully dynamic shapes). The parser only extracts static integer dimensions — dynamic `?` dimensions are silently dropped. |
| 4 | `MemorySpaceAttr` (generic) | ✅ | The RFC's concern here — that the attribute was Spyre-specific rather than a generic extensible wrapper — was addressed upstream by [ktir-mlir-frontend#58](https://github.com/torch-spyre/ktir-mlir-frontend/pull/58), which removed the `KtdpMemorySpaceAttr` interface and made the attribute itself device-agnostic. `ktir_cpu` parses the new spelling and maps it onto the interpreter's `HBM`/`LX` via `KTDP_MEMORY_SPACE_KINDS` in `ktir_cpu/ir_types.py`, alongside the `MemRef.memory_space` validation that defines those names, so Spyre specifics stay confined to the simulator and latency model. **The RFC text itself is now stale — see row 4a.** |
| 4a | `MemorySpaceAttr` spelling conflicts with RFC | ⚠️ **spec conflict** | **The implementation intentionally diverges from RFC 0682, and the RFC needs updating.** The RFC specifies `#ktdp.spyre_memory_space<HBM\|LX[, core = N]>` with an `unspecified` kind; [ktir-mlir-frontend#58](https://github.com/torch-spyre/ktir-mlir-frontend/pull/58) replaced this with `#ktdp.memory_space<global\|ct_local[, ct_id = N]>` and dropped `unspecified` with no replacement. This is a semantic change, not just a rename: the enum went from naming *devices* (HBM/LX) to naming *visibility* (reachable by all compute tiles vs. private to one). The RFC spelling no longer parses in the dialect at all, so the implementation follows the dialect. **Action: RFC 0682 §MemorySpaceAttr should be revised to match; until then treat the RFC as stale on this point.** |

## C. Affine/Polyhedral Attributes

| # | Spec Attribute | Status | Notes |
|---|---------------|--------|-------|
| 5 | `coordinate_set` on `construct_memory_view` | 🟡 | Parsed and stored in `TileRef`. Not yet used to enforce coordinate constraints during load/store dispatch — the spec-required disjointness/overlap semantics are not enforced. |
| 6 | `access_tile_set` on `construct_access_tile` | ✅ | Parsed, stored in `AccessTile`, and used by `ktdp.load`/`ktdp.store` to enumerate coordinate tuples via the affine evaluator. |
| 7 | `access_tile_order` on `construct_access_tile` | ✅ | Parsed and stored; `ktdp.load`/`ktdp.store` apply the traversal order when iterating over coordinates. |
| 8 | `base_map` on `construct_access_tile` | ✅ | Parsed (identity map synthesized if absent); evaluated in `MemoryOps.tile_access` via the affine expression evaluator. |

**Note**: #6–8 and #33–35 are resolved — the interpreter uses affine coordinate sets and traversal order for load/store. Remaining gap is #5: `coordinate_set` on memory views is preserved but not enforced during access, so overlapping/disjoint distribution semantics are not yet checked.

## D. SCF Control-Flow Operations

The spec lists these SCF operations as "currently contemplated":

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 9 | `scf.reduce` | ❌ | Inter-core reduction is covered by the experimental `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` pair (rows 2a–2d), which is semantically different from SCF-level reductions. |
| 10 | `scf.reduce.return` | ❌ | |
| 11 | `scf.parallel` | ❌ | |
| 12 | `scf.forall` | ❌ | |

Currently implemented: `scf.for`, `scf.if`, `scf.yield`.

## E. Standard MLIR Dialect Operations

This section is best read as a **coverage backlog**, not a list of equally
strong RFC violations. The RFC explicitly defines the `ktdp` surface and
explicitly calls out only a small subset of non-`ktdp` ops, so an absent op the
RFC names — `memref.subview` is the one still missing — is a stronger finding
than an absent op from the broader Arith/Math dialects.

### Arith dialect

The spec references the [full Arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/).

Which arith ops have a handler is no longer listed here. It is generated per op in [`docs/supported_ops.md`](supported_ops.md), read from the registry itself and held current by `tests/test_kernelentry.py` — an inventory copied into prose is what let rows 13–19 below sit at ❌ for months after they were implemented.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 13 | `arith.cmpf` | ✅ | `ktir_cpu/dialects/arith_ops.py:170`, priced `COMPUTE_FLOAT`; shares a parser with `arith.cmpi`. Used by `examples/ktir/reduce_multiop.mlir` to build a max fold. |
| 14 | `arith.negf` | ✅ | Registered in `ktir_cpu/dialects/arith_ops.py`; used by `examples/ktir/ffn_swiglu.mlir`. |
| 15 | `arith.absf` | ✅ | Registered in `ktir_cpu/dialects/arith_ops.py`. |
| 16 | `arith.minf` | ✅ | Registered alongside `maxf` / `maxnumf` / `minimumf` / `minnumf` in `ktir_cpu/dialects/arith_ops.py`. |
| 17 | `arith.minnumf` | ✅ | Same registration as row 16. |
| 18 | `arith.fptosi`, `arith.fptoui`, `arith.uitofp` | ✅ | All three registered in `ktir_cpu/dialects/arith_ops.py` alongside `sitofp`. |
| 19 | `arith.divsi`, `arith.remsi`, `arith.andi`, `arith.ori`, `arith.xori`, `arith.ceildivsi`, `arith.floordivsi` | ✅ | All registered in `ktir_cpu/dialects/arith_ops.py` alongside the unsigned variants. |

### Math dialect

The spec references the [full Math dialect](https://mlir.llvm.org/docs/Dialects/MathOps/). Currently implemented: `math.exp`, `math.sqrt`, `math.log`, `math.rsqrt`, `math.log2`, `math.log1p`, `math.tanh`, `math.sin`, `math.cos`, `math.absf`, `math.absi`, `math.ceil`, `math.floor`, `math.erf`, `math.powf`, `math.fma`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 20 | `math.log2`, `math.log1p` | ✅ | |
| 21 | `math.tanh`, `math.sin`, `math.cos` | ✅ | |
| 22 | `math.rsqrt` | ✅ | |
| 23 | `math.absf`, `math.absi`, `math.ceil`, `math.floor` | ✅ | |
| 24 | `math.erf`, `math.powf`, `math.fma` | ✅ | `math.erf` uses polynomial approximation (no scipy) |

### Linalg dialect

The spec references the [full Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/). Currently implemented: `linalg.reduce`, `linalg.matmul`, `linalg.generic`, `linalg.broadcast`, `linalg.transpose`, `linalg.add`, `linalg.fill`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 25 | `linalg.add` | ✅ | Implemented in `ktir_cpu/dialects/linalg_ops.py` as elementwise `Tile + Tile`. Used by `ktdp.inter_tile_reduce`'s combiner region in the rewritten `ring_reduce.mlir` example. |
| 26 | `linalg.generic` | ✅ | Full `bb0` block handling in `ktir_cpu/dialects/linalg_ops.py` |
| 27 | `linalg.map`, `linalg.broadcast`, `linalg.transpose` | 🟡 | `broadcast` and `transpose` implemented; `map` still missing |

### Tensor dialect

Which tensor ops have a handler is generated in [`docs/supported_ops.md`](supported_ops.md), for the reason given under Arith above.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 28 | `tensor.extract_slice` | ✅ | Handler at `ktir_cpu/dialects/tensor_ops.py:217`, frontend adapter at `ktir_cpu/mlir_frontend/parser.py:772`. No example under `examples/` uses it, so only unit tests cover it. |
| 29 | `tensor.insert_slice`, `tensor.collapse_shape` | ✅ | Both implemented. `insert_slice` handler at `ktir_cpu/dialects/tensor_ops.py:258`, regex parser at `:714`, frontend adapter at `ktir_cpu/mlir_frontend/parser.py:786`. No example under `examples/` uses it; `tests/test_dialects_exec.py` and `tests/test_dialects_parse.py` cover it. |

### MemRef dialect

The spec explicitly mentions `memref.subview` for view-based transformations. **The entire `memref` dialect is absent from the implementation.**

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 30 | `memref.subview` | ❌ | Spec explicitly calls this out |
| 31 | All other `memref` operations | ❌ | No `memref` dialect module exists |

## F. Semantic/Behavioral Gaps

| # | Gap | Status | Details |
|---|-----|--------|---------|
| 32 | `construct_memory_view` doesn't support mixed static+dynamic sizes/strides | 🟡 | The spec supports dynamic SSA operands for sizes/strides (`$sizes` as variadic index). The parser only handles static integer literals in `sizes: [96, 64]`. |
| 33 | `construct_access_tile` ignores base coordinate computation | ✅ | `base_map` is parsed and evaluated via the affine expression engine in `MemoryOps.tile_access`. |
| 34 | `ktdp.load` only implements rectangular slice semantics | ✅ | Now enumerates coordinates from `access_tile_set` and applies `access_tile_order`; supports general polyhedral regions. |
| 35 | `ktdp.store` only implements rectangular slice semantics | ✅ | Same coordinate-set enumeration as load. |
| 36 | `module { }` is tolerated, but module-level structure is not modeled | 🟡 | The parser can find `func.func` inside a `module { ... }` wrapper, but it does not model module-level attributes, declarations, or non-function top-level constructs. |
| 36a | `ktdp.region_terminator` has a frontend adapter and no execution handler | ❌ | The MLIR bindings' region walk emits an implicit terminator that never appears in text IR; `ktir_cpu/mlir_frontend/parser.py:238` installs an adapter for it so the walk survives, and `adapt_block` keeps it in the region's op list. The interpreter has no handler, so `_execute_op` would raise `Unknown operation` if a kernel ever reached one. None does: walking every op the frontend produces for the three inter-tile kernels it accepts — `examples/latency/rmsnorm_4core_2x2.mlir`, `examples/ktir/ffn_swiglu_4core.mlir` and `examples/sdsc/sdpa_pv_ksplit.mlir`, regions included — finds `ktdp.yield_partial` / `ktdp.yield_reduced` terminating them and no `ktdp.region_terminator` anywhere. So the asymmetry is real and the adapter unreached, but which construct makes the bindings emit an implicit terminator is not established. `docs/supported_ops.md` shows it as the one op with a frontend adapter and no handler. |

## G. Parser Limitations

| # | Gap | Status | Details |
|---|-----|--------|---------|
| 37 | No affine expression evaluation | ✅ | Full affine map and integer set parsing and evaluation implemented in `ktir_cpu/parser_ast.py` (`parse_affine_map`, `parse_affine_set`, `eval_affine_map`, `enumerate_affine_set`). |
| 38 | No `#alias = affine_set<...>` / `#alias = affine_map<...>` support | ✅ | Parser pre-scans module scope and populates an `aliases` dict; dialect parsers resolve aliases via `parse_ctx.aliases`. |
| 39 | `func.func` signature parsing is limited | 🟡 | The parser handles the basic typed signatures used in the shipped examples, but not the full MLIR function-signature space (richer types, attributes, or more complex declarative forms). |

## H. Priority Summary

### High Priority
Blocks running spec-compliant KTIR programs:

- **#5**: 🟡 `coordinate_set` on memory views preserved but not enforced

### Medium Priority
Limits dialect coverage for real-world kernels:

- **#9–12**: ❌ SCF parallel/reduce operations
- **#20–24**: ✅ All math ops now implemented (log2, log1p, tanh, sin, cos, rsqrt, absf, ceil, floor, erf, powf, fma)
- **#30–31**: ❌ Entire `memref` dialect
- **#36a**: ❌ `ktdp.region_terminator` has a frontend adapter and no execution handler — an asymmetry no kernel under `examples/` reaches, on a construct whose trigger is not established
- **#32**: 🟡 Dynamic sizes/strides not supported

### Lower Priority
Extensibility and completeness:

- **#3, #4**: ❌/🟡 Dynamic access tile dimensions, generic `MemorySpaceAttr`
- **#27**: 🟡 `linalg.map` still missing (`broadcast` and `transpose` implemented)
- **#36, #39**: 🟡 Module-level handling, full function signatures

### Resolved
- **#2**: ✅ `construct_indirect_access_tile`
- **#6, #7, #8**: ✅ `access_tile_set`, `access_tile_order`, `base_map`
- **#20–24**: ✅ All math ops (rsqrt, log2, log1p, tanh, sin, cos, absf, ceil, floor, erf, powf, fma)
- **#25**: ✅ `linalg.add`
- **#13–19**: ✅ The standard arith ops (cmpf, negf, absf, minf/minnumf, the float casts, the signed and bitwise int ops)
- **#26**: ✅ `linalg.generic`
- **#28, #29**: ✅ `tensor.extract_slice`, `tensor.insert_slice`
- **#33, #34, #35**: ✅ Access tile coordinate semantics
- **#37, #38**: ✅ Affine expression evaluation and alias support

### Experimental (tracks unmerged upstream spec PRs)
- **#2a–2d**: 🧪 Inter-tile communication four-op design — see [ktir-mlir-frontend#23](https://github.com/torch-spyre/ktir-mlir-frontend/pull/23). Reduce path implemented; broadcast / reduce-scatter / per-tile sync not.

## I. Status as of 2026-05-30

Significant progress since the original writeup:

- `ktdp.construct_indirect_access_tile` (#2) is fully implemented with passing tests.
- The entire affine/polyhedral foundation (#6–8, #33–35, #37, #38) is now in
  place: affine maps and integer sets are parsed, evaluated, and used by
  `ktdp.load`/`ktdp.store` to enumerate coordinate tuples.
- `linalg.generic` (#26) is implemented with full `bb0` block handling.
- `linalg.broadcast`, `linalg.transpose` (#27 partial), `tensor.collapse_shape`
  (#29 partial), `math.log`, and **`linalg.add` (#25)** are now implemented.
- 🧪 **Experimental** inter-tile reduce path (rows 2a, 2d): the four-op
  design from [ktir-mlir-frontend#23](https://github.com/torch-spyre/ktir-mlir-frontend/pull/23)
  is implemented for `inter_tile_produce` + `inter_tile_reduce` (reduce
  delivery only) ahead of upstream merge. Per-core wire bytes flow to the
  latency tracker via `Tile.comm_bytes`. End-to-end ring_reduce test
  passes. Op surface may shift to track the upstream PR.

Remaining notable gaps:

- `coordinate_set` on memory views (#5) is preserved in the IR but not used
  to enforce coordinate constraints during dispatch.
- SCF parallel/reduce ops (#9–12), `tensor.extract_slice` (#28), and the
  entire `memref` dialect (#30–31) remain unimplemented.
- 🧪 Inter-tile broadcast (`inter_tile_consume`, row 2b) and reduce-scatter
  (`inter_tile_reduce_scatter`, row 2c) not yet implemented; per-tile sync
  (`producer_dependency_per_consumer`, row 2d) is parsed but runs in
  full-barrier mode.

## J. Prioritized Conformance Roadmap

The initial phases of this roadmap are complete. The conformance target was established as "execute the RFC-defined `ktdp` subset plus the specific non-`ktdp` ops used by compiler-generated kernels." Spec gap tests were added to make missing coverage explicit. The access-tile foundation was then rebuilt: `base_map`, `access_tile_set`, `access_tile_order`, and `coordinate_set` are now parsed and preserved; a full affine/integer-set evaluator was implemented; and `ktdp.load`/`ktdp.store` iterate over affine coordinate tuples rather than rectangular subviews. `ktdp.construct_indirect_access_tile` was also completed as part of this work.

### Add The Missing KTDP Ops 🟡

Goal: cover the RFC-defined `ktdp` surface.

- ✅ Implement `ktdp.construct_indirect_access_tile`.
- ✅ Implement `ktdp.construct_distributed_memory_view`.
- Add validation rules for:
  matching dimensionalities,
  allowed direct versus indirect dimensions,
  and the RFC restriction that indirect indices are not further affine-scaled.
- Preserve dynamic `access_tile` dimensions (`?`) in the IR even if runtime
  support is initially partial.

### Close The RFC-Explicit Non-KTDP Gaps 🟡

Goal: support the rest of the ops the RFC explicitly calls out.

- ✅ Add `linalg.add` so the RFC's canonical matrix-add example can execute
  without translation.
- ❌ Add `tensor.extract_slice`.
- ❌ Add `memref.subview` and the minimal `memref` dialect support required to
  interpret it.
- ❌ Add the missing SCF ops explicitly named by the RFC:
  `scf.reduce`,
  `scf.reduce.return`,
  `scf.parallel`,
  and `scf.forall`.

### Widen Dialect Coverage Opportunistically ❌

Goal: improve practicality for real compiler output without pretending every
MLIR op is equally important for RFC conformance.

- Add only the Arith/Math/Linalg ops actually observed in upstream-generated
  KTIR or required by target workloads.
- Track these as "compiler coverage" rather than "spec blockers."
- Keep a small compatibility matrix in docs that separates:
  `RFC core`,
  `example coverage`,
  and `observed compiler output coverage`.

## K. Runtime / Simulation Correctness

These items concern the CPU simulator's fidelity rather than missing KTIR
spec surface.

### K1. Multi-round communication re-execution

**Status**: ✅ Fixed in PR-B (grid-network branch, issue #50).

`execute_with_communication` now uses a generator-based cooperative scheduler.
Each core runs as a Python generator via `CoreExecutionStack`; blocking `recv`
operations suspend the generator (`yield RecvRequest(src)`) until the expected
tile is delivered. No BSP replay — each core executes exactly once.

See `docs/cross_core_scheduling.md` for the full design.

### K2. Cyclic communication correctness

**Status**: ✅ Fixed in PR-B (grid-network branch, issue #50).

`CommOps.reduce` is now a generator that yields `RecvRequest` per ring round.
The scheduler drives it to completion via `gen.send(tile)`, consuming each
message exactly once in order. No duplicate sends, no message loss.
Bidirectional exchanges (both cores send then recv) are handled correctly
because `send_to` is fire-and-forget — the sender enqueues and continues
without blocking, so symmetric patterns never deadlock.

### K3. Comm ops restricted to top-level function body

**Status**: ✅ Fixed (issue #131).

`ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` can now appear inside
`scf.for` / `scf.if` bodies.  The fix propagates the generator protocol through
`execute_region_with_comms` → `ControlOps.for_op_with_comms` / `if_op_with_comms`
→ `scf__for` / `scf__if` → `_execute_until_block`.  Compute-only region callers
(`linalg.reduce` combiners, `tensor.generate`, ktdp combiner bodies) continue
using the synchronous `execute_region` path unchanged.

End-to-end test: `tests/test_examples.py::TestRingReduceInnerLoopExecution`
(`examples/ktir/ring_reduce_inner_loop.mlir`).
Unit tests: `tests/test_grid_scheduler.py::test_inner_loop_comm`.

### K4. Multi-cast load modeling

**Status**: ❌ Not modeled. No existing kernels require it.

There is currently no model for multi-cast loads where one ring-bus
transaction serves multiple cores simultaneously. Two variations exist:

- LX-to-LX memory transfer (unicast or multi-cast)
- HBM-to-LX multi-cast load

The kernel optimizer would need to annotate `ktdp.load` with a
participating-core group attribute so the latency calculator can account for
the shared transaction cost. This is a future design question.

### Suggested Execution Order

If we want the fastest path to meaningful conformance progress:

1. ✅ Build the first-class access-tile IR and affine evaluator.
2. ✅ Rework `ktdp.load` / `ktdp.store` around that representation.
3. ✅ Add `construct_indirect_access_tile`.
4. ✅ Add `construct_distributed_memory_view`.
5. 🟡 Add `linalg.add` (✅), `tensor.extract_slice` (❌), and `memref.subview` (❌).
6. ❌ Fill in the missing RFC-listed SCF ops.
7. ❌ Expand broader Arith/Math/Linalg coverage as compiler demand appears.
8. 🧪 Track [ktir-mlir-frontend#23](https://github.com/torch-spyre/ktir-mlir-frontend/pull/23)
   to upstream merge; reconcile any op-surface drift in the experimental
   four-op inter-tile path (rows 2a–2d).

### Definition Of "Good Enough" For A First Conformance Milestone

A strong first milestone would be:

- ✅ affine attributes are preserved and exercised in tests
- ✅ `ktdp.load` / `ktdp.store` operate over real coordinate collections
- ✅ all RFC-defined `ktdp` ops parse and execute
- 🟡 the RFC matrix-add example can run with only mechanical syntax adaptation
  (`linalg.add` ✅; pending `tensor.extract_slice` / `memref.subview` if the
  example uses them)
- ❌ the repo has explicit tests for unsupported versus supported RFC surface
