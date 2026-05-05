# KTIR Spec RFC vs. `ktir_cpu` Implementation — Gap Analysis

**Date**: 2026-04-22
**Spec**: [RFC 0682 — KTIR Spec](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)

**Legend**: ✅ implemented — 🟡 partial — ❌ not implemented

---

## A. `ktdp` Dialect Operations

| # | Spec Operation | Status | Notes |
|---|---------------|--------|-------|
| 1 | `ktdp.construct_distributed_memory_view` | ❌ | No handler, no parser. Key primitive for composing multiple per-partition memory views into a single distributed logical view. Critical for modeling distributed scratchpads. |
| 2 | `ktdp.construct_indirect_access_tile` | ✅ | Handler and parser implemented in `ktir_cpu/dialects/ktdp_ops.py`; tests passing in `tests/test_indirect_access.py`. |

## B. `ktdp` Types & Attributes

| # | Spec Item | Status | Notes |
|---|-----------|--------|-------|
| 3 | `AccessTileType` with dynamic dimensions (`?`) | ❌ | The spec allows `access_tile<? x 64 x index>` (partially/fully dynamic shapes). The parser only extracts static integer dimensions — dynamic `?` dimensions are silently dropped. |
| 4 | `MemorySpaceAttr` (generic) | 🟡 | The parser extracts `SpyreMemorySpaceAttr` (`HBM`/`LX`), but the spec describes `MemorySpaceAttr` as a generic extensible wrapper that could encapsulate other hardware backends. The implementation hardcodes Spyre-specific memory spaces only. |

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
| 9 | `scf.reduce` | ❌ | Only `ktdp.reduce` exists for inter-core comm, which is semantically different |
| 10 | `scf.reduce.return` | ❌ | |
| 11 | `scf.parallel` | ❌ | |
| 12 | `scf.forall` | ❌ | |

Currently implemented: `scf.for`, `scf.if`, `scf.yield`.

## E. Standard MLIR Dialect Operations

This section is best read as a **coverage backlog**, not a list of equally
strong RFC violations. The RFC explicitly defines the `ktdp` surface and
explicitly calls out only a small subset of non-`ktdp` ops. Missing
`linalg.add`, `tensor.extract_slice`, and `memref.subview` are more directly
grounded in the RFC text than every absent op from the broader Arith/Math
dialects.

### Arith dialect

The spec references the [full Arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/). Currently implemented: `addf`, `subf`, `mulf`, `divf`, `addi`, `subi`, `muli`, `divui`, `remui`, `constant`, `maxf`, `maxnumf`, `extf`, `truncf`, `index_cast`, `sitofp`, `cmpi`, `select`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 13 | `arith.cmpf` | ❌ | float compare — only `arith.cmpi` (int compare) exists |
| 14 | `arith.negf` | ❌ | |
| 15 | `arith.absf` | ❌ | |
| 16 | `arith.minf` | ❌ | only `maxf` / `maxnumf` exist |
| 17 | `arith.minnumf` | ❌ | |
| 18 | `arith.fptosi`, `arith.fptoui`, `arith.uitofp` | 🟡 | only `sitofp` exists |
| 19 | `arith.divsi`, `arith.remsi`, `arith.andi`, `arith.ori`, `arith.xori`, `arith.ceildivsi`, `arith.floordivsi` | ❌ | only unsigned variants `divui`/`remui` exist |

### Math dialect

The spec references the [full Math dialect](https://mlir.llvm.org/docs/Dialects/MathOps/). Currently implemented: `math.exp`, `math.sqrt`, `math.log`, `math.rsqrt`, `math.log2`, `math.log1p`, `math.tanh`, `math.sin`, `math.cos`, `math.absf`, `math.ceil`, `math.floor`, `math.erf`, `math.powf`, `math.fma`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 20 | `math.log2`, `math.log1p` | ✅ | |
| 21 | `math.tanh`, `math.sin`, `math.cos` | ✅ | |
| 22 | `math.rsqrt` | ✅ | |
| 23 | `math.absf`, `math.ceil`, `math.floor` | ✅ | |
| 24 | `math.erf`, `math.powf`, `math.fma` | ✅ | `math.erf` uses polynomial approximation (no scipy) |

### Linalg dialect

The spec references the [full Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/). Currently implemented: `linalg.reduce`, `linalg.matmul`, `linalg.generic`, `linalg.broadcast`, `linalg.transpose`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 25 | `linalg.add` | ❌ | Used in the spec's primary matrix-add example — won't execute today |
| 26 | `linalg.generic` | ✅ | Full `bb0` block handling in `ktir_cpu/dialects/linalg_ops.py` |
| 27 | `linalg.map`, `linalg.broadcast`, `linalg.transpose` | 🟡 | `broadcast` and `transpose` implemented; `map` still missing |

### Tensor dialect

Currently implemented: `tensor.splat`, `tensor.extract`, `tensor.expand_shape`, `tensor.collapse_shape`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 28 | `tensor.extract_slice` | ❌ | Spec explicitly calls this out for tensor-level slicing |
| 29 | `tensor.insert_slice`, `tensor.collapse_shape` | 🟡 | `collapse_shape` implemented; `insert_slice` still missing |

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

## G. Parser Limitations

| # | Gap | Status | Details |
|---|-----|--------|---------|
| 37 | No affine expression evaluation | ✅ | Full affine map and integer set parsing and evaluation implemented in `ktir_cpu/parser_ast.py` (`parse_affine_map`, `parse_affine_set`, `eval_affine_map`, `enumerate_affine_set`). |
| 38 | No `#alias = affine_set<...>` / `#alias = affine_map<...>` support | ✅ | Parser pre-scans module scope and populates an `aliases` dict; dialect parsers resolve aliases via `parse_ctx.aliases`. |
| 39 | `func.func` signature parsing is limited | 🟡 | The parser handles the basic typed signatures used in the shipped examples, but not the full MLIR function-signature space (richer types, attributes, or more complex declarative forms). |

## H. Priority Summary

### High Priority
Blocks running spec-compliant KTIR programs:

- **#1**: ❌ `construct_distributed_memory_view`
- **#25**: ❌ `linalg.add` — used in the spec's primary example and won't execute
- **#5**: 🟡 `coordinate_set` on memory views preserved but not enforced

### Medium Priority
Limits dialect coverage for real-world kernels:

- **#9–12**: ❌ SCF parallel/reduce operations
- **#13–19**: ❌/🟡 Many standard arith ops (cmpf, negf, absf, minf, signed int ops)
- **#20–24**: ✅ All math ops now implemented (log2, log1p, tanh, sin, cos, rsqrt, absf, ceil, floor, erf, powf, fma)
- **#28, #30–31**: ❌ `tensor.extract_slice`, entire `memref` dialect
- **#32**: 🟡 Dynamic sizes/strides not supported

### Lower Priority
Extensibility and completeness:

- **#3, #4**: ❌/🟡 Dynamic access tile dimensions, generic `MemorySpaceAttr`
- **#27, #29**: 🟡 Remaining linalg/tensor ops (`linalg.map`, `tensor.insert_slice`)
- **#36, #39**: 🟡 Module-level handling, full function signatures

### Resolved
- **#2**: ✅ `construct_indirect_access_tile`
- **#6, #7, #8**: ✅ `access_tile_set`, `access_tile_order`, `base_map`
- **#20–24**: ✅ All math ops (rsqrt, log2, log1p, tanh, sin, cos, absf, ceil, floor, erf, powf, fma)
- **#26**: ✅ `linalg.generic`
- **#33, #34, #35**: ✅ Access tile coordinate semantics
- **#37, #38**: ✅ Affine expression evaluation and alias support

## I. Status as of 2026-04-22

Significant progress since the original writeup:

- `ktdp.construct_indirect_access_tile` (#2) is fully implemented with passing tests.
- The entire affine/polyhedral foundation (#6–8, #33–35, #37, #38) is now in
  place: affine maps and integer sets are parsed, evaluated, and used by
  `ktdp.load`/`ktdp.store` to enumerate coordinate tuples.
- `linalg.generic` (#26) is implemented with full `bb0` block handling.
- `linalg.broadcast`, `linalg.transpose` (#27 partial), `tensor.collapse_shape`
  (#29 partial), and `math.log` are now implemented.

Remaining notable gaps:

- `ktdp.construct_distributed_memory_view` (#1) is still the largest missing
  ktdp op; it blocks distributed scratchpad modeling.
- `linalg.add` (#25) is still missing — the RFC's canonical matrix-add example
  cannot execute without it.
- `coordinate_set` on memory views (#5) is preserved in the IR but not used
  to enforce coordinate constraints during dispatch.
- SCF parallel/reduce ops (#9–12), `tensor.extract_slice` (#28), and the
  entire `memref` dialect (#30–31) remain unimplemented.

## J. Prioritized Conformance Roadmap

The initial phases of this roadmap are complete. The conformance target was established as "execute the RFC-defined `ktdp` subset plus the specific non-`ktdp` ops used by compiler-generated kernels." Spec gap tests were added to make missing coverage explicit. The access-tile foundation was then rebuilt: `base_map`, `access_tile_set`, `access_tile_order`, and `coordinate_set` are now parsed and preserved; a full affine/integer-set evaluator was implemented; and `ktdp.load`/`ktdp.store` iterate over affine coordinate tuples rather than rectangular subviews. `ktdp.construct_indirect_access_tile` was also completed as part of this work.

### Add The Missing KTDP Ops 🟡

Goal: cover the RFC-defined `ktdp` surface.

- ✅ Implement `ktdp.construct_indirect_access_tile`.
- ❌ Implement `ktdp.construct_distributed_memory_view`.
- Add validation rules for:
  matching dimensionalities,
  allowed direct versus indirect dimensions,
  and the RFC restriction that indirect indices are not further affine-scaled.
- Preserve dynamic `access_tile` dimensions (`?`) in the IR even if runtime
  support is initially partial.

### Close The RFC-Explicit Non-KTDP Gaps ❌

Goal: support the rest of the ops the RFC explicitly calls out.

- Add `linalg.add` so the RFC's canonical matrix-add example can execute
  without translation.
- Add `tensor.extract_slice`.
- Add `memref.subview` and the minimal `memref` dialect support required to
  interpret it.
- Add the missing SCF ops explicitly named by the RFC:
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

**Status**: ❌ Not yet fixed. Does not affect current example kernels.

`execute_with_communication` in `grid.py` re-executes the entire function body
on every core for each communication round. The latency tracker accumulates
costs from every round, so kernels using `ktdp.transfer` or `ktdp.reduce`
would have inflated latency — all operations (loads, compute, stores) counted
N times instead of once, where N is the number of communication rounds.

Current example kernels (vector_add, softmax, layernorm, matmul) are all
embarrassingly parallel — no comm ops — so `execute_with_communication`
completes in 1 round and latency is counted exactly once.

The multi-round replay exists because the simulator executes cores
sequentially. When core 0 calls `ktdp.transfer` to send data to core 5,
core 5 may have already executed in this round and missed the message.
BSP-style replay re-runs all cores so core 5 picks up the message on the
next round.

**Proposed fix**: Replace the BSP replay loop with a single-pass execution
model where each core executes at most once, scheduled in dependency order.
Cores blocked on a `recv_from` are retried after other cores complete. Cyclic
dependencies are detected and raised as errors.

### K2. Cyclic communication correctness

**Status**: ❌ Functional correctness problem.

The BSP replay model breaks for bidirectional exchanges and `ktdp.reduce`
(which does its own internal ring loop within a single call):

- `CommOps.reduce` calls `ring.recv_from()` with `pop(0)`, consuming messages.
  On the next round the consumed messages are gone, producing different
  intermediate results.
- `CommOps.transfer` calls `ring.send()` every round, so duplicate messages
  accumulate in the buffer.
- Neither `ring_network` nor `core.values` are reset between rounds.

The single-pass execution fix in K1 would also resolve this.

### K3. Multi-cast load modeling

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
4. ❌ Add `construct_distributed_memory_view`.
5. ❌ Add `linalg.add`, `tensor.extract_slice`, and `memref.subview`.
6. ❌ Fill in the missing RFC-listed SCF ops.
7. ❌ Expand broader Arith/Math/Linalg coverage as compiler demand appears.

### Definition Of "Good Enough" For A First Conformance Milestone

A strong first milestone would be:

- ✅ affine attributes are preserved and exercised in tests
- ✅ `ktdp.load` / `ktdp.store` operate over real coordinate collections
- 🟡 all RFC-defined `ktdp` ops parse and execute (`construct_distributed_memory_view` still missing)
- ❌ the RFC matrix-add example can run with only mechanical syntax adaptation
- ❌ the repo has explicit tests for unsupported versus supported RFC surface
