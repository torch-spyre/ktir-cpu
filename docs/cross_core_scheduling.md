# Cross-Core Scheduling Design

## Overview

The scheduler enables correct multi-core execution with blocking communication.
Each core runs as a Python generator; blocking `recv` operations suspend the generator
until the expected tile arrives, rather than replaying the entire kernel in BSP rounds.

---

## Architecture

```
KTIRInterpreter
  └── GridExecutor.execute_with_communication(ops, input_ptrs, execute_op)
        │
        ├── CoreExecutionStack (one per core)   ← only generator-aware layer
        │     ├── _execute_until_block(ops, execute_op)   generator
        │     │     for each op:
        │     │       result = execute_op(op, core)
        │     │       if isgenerator(result):
        │     │           result = yield from result   ← bubbles RecvRequests up
        │     │           _store(op, result, core)
        │     └── resume(send_val) / is_blocked() / waiting_on
        │
        ├── CoreContext (one per core)
        │     ├── SSA value map (_scope_stack)
        │     ├── LX scratchpad + usage tracking
        │     ├── send_to(dst, tile)   ← wired to scheduler _enqueue before run
        │     └── get_lx(core_id)     ← returns local LX or delegates to ring_backend
        │
        ├── RingBackend (injected into CoreContext at construction)
        │     ├── DirectLXBackend [current]
        │     │     get_lx(core_id) → direct dict lookup into SpyreMemoryHierarchy
        │     │     bypasses the ring — valid when LX is pre-seeded by the host
        │     │     (e.g. ktdp.construct_distributed_memory_view)
        │     └── RingTransferBackend [planned, PR-C]
        │           get_lx(core_id) issues a ring send and yields RecvRequest
        │           — same scheduler protocol as CommOps, zero new infrastructure
        │
        └── Scheduler loop
              messages: {(src,dst): deque[Tile]}
              waiting:  {core_id: src_core}
              while stacks:
                  if no _try_deliver succeeds → RuntimeError("Deadlock detected")
```

**Roles:**

| Component | Responsibility |
|---|---|
| `CoreExecutionStack` | Generator lifecycle; sole place that calls `send()` / `next()` on generators; bubbles `RecvRequest` to the scheduler via `yield from` |
| `CoreContext` | Per-core SSA state, LX, and `send_to` (fire-and-forget send wired by scheduler) |
| `CoreContext.get_lx(core_id)` | Returns local LX directly; for remote cores delegates to `ring_backend.get_lx()` |
| `GridExecutor` | Owns cores, drives the scheduler loop, detects deadlock |
| `CommOps` | Communication algorithms (ring reduce, transfer) expressed as generators; yields `RecvRequest` at each blocking point |
| `DirectLXBackend` | Remote LX via direct dict lookup — no ring hop, used for pre-seeded distributed views |
| `RingTransferBackend` (planned) | Remote LX via ring message; will yield `RecvRequest` and integrate with the existing scheduler — no protocol changes needed |

---

## Control Flow: Blocking Recv

```
dialect handler (ktdp_ops.py)
  └── CommOps.reduce(context, tile, core_group, reduce_fn)  ← generator
        for each round:
          context.send_to(next_core, result)         ← enqueues tile immediately
          received = yield RecvRequest(src=prev_core) ← suspends here
          result = reduce_fn(result, received)
        return result

CoreExecutionStack._execute_until_block
  result = execute_op(op, core)              ← returns generator
  result = yield from result                 ← forwards RecvRequest to scheduler

GridExecutor.execute_with_communication
  _advance(core_id):
    request = next(stack._gen)               ← steps until RecvRequest or done
    if isinstance(request, RecvRequest):
        waiting[core_id] = request.src        ← park core
    else:
        results[core_id] = done

  _try_deliver(core_id):
    tile = _pop(src, core_id)
    if tile: del waiting[core_id]; _advance(core_id, tile)  ← resume with tile

  while stacks:
    if not any(_try_deliver(c) for c in stacks):
        raise RuntimeError("Deadlock detected: ...")
```

**Key invariant:** `send_to` is fire-and-forget — the sender never blocks.
Only `yield RecvRequest` suspends a core. This prevents sender-side deadlock
in symmetric patterns (all-to-all, ring).

---

## Generator Design: Inversion of Control

Without generators, blocking communication would require either:
- **Callbacks + asyncio**: complex event loop, hard to integrate with sync interpreter
- **BSP replay**: re-execute the entire kernel per round, incorrect for asymmetric patterns

With generators, the scheduler drives execution:

```
Normal op   → execute_op returns a plain value  → scheduler stores it, continues
Blocking op → execute_op returns a generator   → scheduler steps it, parks on RecvRequest
                                                → resumes with tile.send(tile) when ready
```

`execute_region` (for SCF loops, if-branches) stays **fully synchronous** — it executes
a flat list of ops and returns the last result. Comm ops cannot appear inside nested
regions in the current spec; if they appear at the top level of a function body,
`CoreExecutionStack` handles the generator switching transparently via `yield from`.

---

## Deadlock Detection

After each scheduler pass, if no core made progress (no `_try_deliver` succeeded)
but some cores are still parked in `waiting`, the scheduler raises:

```
RuntimeError("Deadlock detected: core 0 waiting on recv from core 1; ...")
```

This catches both flat deadlocks (mutual recv with no sends) and loop-induced
deadlocks (one side exhausts its sends before the other finishes recving).

---

## Send/Recv API

| Symbol | Location | Purpose |
|---|---|---|
| `RecvRequest(src: int)` | `ktir_cpu/grid.py` | Typed yield token; scheduler checks `isinstance(x, RecvRequest)` |
| `CoreContext.send_to(dst, tile)` | `ktir_cpu/grid.py` | Wired to `_enqueue` by scheduler before each core runs |
| `CommOps.reduce(...)` | `ktir_cpu/ops/comm_ops.py` | N-1 ring rounds as a generator; yields `RecvRequest` per round |
| `CommOps.transfer(...)` | `ktir_cpu/ops/comm_ops.py` | Fire-and-forget; not a generator |

`RecvRequest` is defined in `grid.py` (not `comm_ops.py`) to avoid a circular import:
`comm_ops.py` imports `CoreContext` from `grid.py`, so `RecvRequest` must live there.
