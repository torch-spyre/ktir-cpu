# Cross-Core Scheduling — Roles & Boundaries

## One scheduler path

There is exactly one way cores communicate during execution:

```
client code           ctx.send_to(dst, tile)        — fire-and-forget
client code (yields)  RecvRequest(src)              — blocks until tile arrives
GridExecutor          owns the message queue and drives every core's generator
```

Everything else is either above this path (clients) or off it entirely
(remote-LX peeks).

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
        │     └── get_lx(core_id)      ← returns local LX or delegates to ring_backend
        │
        ├── RingBackend (injected into CoreContext at construction)
        │     └── DirectLXBackend [current]
        │           get_lx(core_id) → direct lookup into SpyreMemoryHierarchy
        │           used for pre-seeded distributed views
        │
        └── Scheduler loop
              messages: {(src,dst): deque[Tile]}
              waiting:  {core_id: src_core}
              while stacks:
                  if no _try_deliver succeeds → RuntimeError("Deadlock detected")
```

---

## Layers

| Layer | Responsibility | Notes |
|---|---|---|
| **`CommOps`** | High-level **per-core** comm primitives. Each call describes one core's view of the algorithm — its sends, its recvs, its result. Today: `reduce` (ring) and `transfer`. | Stable surface. Dialect handlers and tests should call into here, not into scheduler primitives directly. |
| **`CoreContext`** | Per-core SSA scope, LX scratchpad, and `send_to`. `send_to` is wired by `GridExecutor` to enqueue into the scheduler's message queue. | |
| **`CoreExecutionStack`** | Wraps one core's op list as a generator. The only place that calls `.send()` / `next()` on client generators. Bubbles `RecvRequest` to the scheduler via `yield from`. | Internal. |
| **`GridExecutor`** | Owns cores, the message queue, and the scheduler loop. Drives every core's generator to completion or raises on deadlock. | Internal. |
| **`RingBackend.get_lx`** | Remote LX peek for cross-core memory views. Off the comm path — returns a scratchpad reference, no messages, no scheduling. | Only `get_lx` is live (see "Marked for deletion" below). |

---

## Per-core semantics

Every `CommOps` primitive describes **one core's behavior**. The function
runs once per participating core; the scheduler runs N copies concurrently;
they cooperate via the message queue.

### `CommOps.reduce` — ring reduction

Implements ring reduction from the perspective of one core. Each round:
send to the next neighbor, recv from the previous neighbor, accumulate.
After N-1 rounds, each core holds the full reduction.

```
my_idx   = group.index(ctx.core_id)
result   = tile
for _ in range(len(group) - 1):
    ctx.send_to(group[(my_idx + 1) % n], result)
    received = yield RecvRequest(src=group[(my_idx - 1) % n])
    result   = fn(result, received)
return result
```

Yields **N-1 `RecvRequest`s** for the calling core (one per ring hop).
Every participating core runs its own copy. Cores not in `group` simply
don't call `CommOps.reduce` — they're not participants, no special case.

### `CommOps.transfer` — multi-destination send

Plain function, no yields. Calls `ctx.send_to(d, tile)` for each `d` in
`dst_cores`. The receiving cores must call into `CommOps` (or a future
recv primitive) to consume the tile.

---

## Generator vs plain function — the rule

A `CommOps` primitive is a generator iff it can block.

- Generator: contains `yield RecvRequest(...)`. Calling it returns a
  generator object; the body has not yet run. The scheduler drives it.
- Plain function: no `yield`. Calling it runs the body to completion.

| Primitive | Shape |
|---|---|
| `CommOps.reduce` | generator (yields N-1 times for the calling core) |
| `CommOps.transfer` | plain |

A handler that calls a generator primitive must **propagate** the
generator — don't call it and discard the result:

```python
return CommOps.reduce(ctx, tile, group, fn)              # OK — scheduler drives it
return (yield from CommOps.reduce(ctx, tile, group, fn)) # OK — handler is itself a generator
CommOps.reduce(ctx, tile, group, fn)                     # BUG — generator created, body never runs
```

**Key invariant:** `send_to` is fire-and-forget — the sender never blocks.
Only `yield RecvRequest` suspends a core. This prevents sender-side
deadlock in symmetric patterns (all-to-all, ring).

---

## Control flow: blocking recv (end-to-end)

```
dialect handler / test handler
  └── CommOps.reduce(ctx, tile, group, fn)        ← generator object
        for each round:
          ctx.send_to(next_core, result)          ← enqueues immediately
          received = yield RecvRequest(src=prev)  ← suspends here
          result   = fn(result, received)
        return result

CoreExecutionStack._execute_until_block
  result = execute_op(op, core)                   ← gets the generator
  result = yield from result                      ← forwards RecvRequest up

GridExecutor.execute_with_communication
  _advance(core_id):
    request = next(stack._gen)
    if isinstance(request, RecvRequest):
        waiting[core_id] = request.src            ← park
    else:
        results[core_id] = done

  _try_deliver(core_id):
    tile = _pop(src, core_id)
    if tile: del waiting[core_id]; _advance(core_id, tile)

  while stacks:
    if not any(_try_deliver(c) for c in stacks):
        raise RuntimeError("Deadlock detected: ...")
```

---

## Deadlock detection

After each scheduler pass, if every active core is parked in `waiting`
and no message arrived to unblock anyone:

```
RuntimeError("Deadlock detected: core 0 waiting on recv from core 1; ...")
```

This catches both flat deadlocks (mutual recv with no sends) and
loop-induced deadlocks (one side exhausts its sends before the other
finishes recving).

---

## Future direction — pluggable backends

`CommOps.reduce` today hard-codes the ring algorithm. The next
iteration makes the reduction strategy pluggable, and folds the
remote-LX peek path (`RingBackend.get_lx`) into the same abstraction.
This section describes the target shape; nothing here exists yet.

### Pluggable reduction algorithms

The motivation is hardware variety: ring is one of several plausible
implementations of "combine values across a group." Tree, recursive
halving-doubling, and an LX-scratchpad reduction (each core writes its
partial into a designated LX slot on a target core, no ring messages)
all have valid cost profiles depending on tile size, group shape, and
available bandwidth.

The design separates *what* (semantics: combine N tiles into one) from
*how* (algorithm: ring vs. tree vs. LX):

```python
class ReduceBackend(ABC):
    """Owns the full reduce protocol: messaging, compute, completion.

    run() is a generator — yields RecvRequest at each blocking point.
    The scheduler drives it identically to CommOps.reduce today.
    Returns the result tile for this core when complete.
    """

    @abstractmethod
    def run(self, ctx: CoreContext, tile: Tile,
            core_group: List[int]) -> Generator[RecvRequest, Tile, Tile]:
        ...
```

`CommOps.reduce` becomes a passthrough:

```python
@staticmethod
def reduce(ctx, tile, core_group, backend: ReduceBackend):
    return backend.run(ctx, tile, core_group)
```

Concrete backends:

- **`RingReduceBackend`** — current logic moves here verbatim. Yields
  `N-1` `RecvRequest`s per core.
- **`LXReduceBackend`** — each core writes its partial into a target
  core's LX scratchpad slot via `ctx.get_lx(target_core)`. Synchronous
  (no yields) when LX is directly addressable. Useful when partition
  count is small and ring latency dominates.

### Backend selection — open question

Two plausible injection points:

1. **From op attributes.** A `ktdp.reduce` op carries
   `#ktdp.reduce_kind = "ring" | "lx" | …`; the dialect handler picks
   the backend.
2. **From an execution environment.** `env.reduce_backend_cls` is set
   before `execute_function`; handler reads it and instantiates. Lets
   experiments swap backends without touching MLIR.

(2) is more flexible for split-K and similar exploration; (1) is
preferable once a backend choice is meant to be persisted with the IR.
Likely both end up supported, with attribute precedence over env.

### Unifying remote-LX peeks under the same abstraction

The remote-LX path (`CoreContext.get_lx(other_core)`) currently goes
through a separate `RingBackend` hierarchy whose only live method is
`get_lx`. The off-the-comm-path `send` / `recv` methods on
`RingBackend` and `DirectLXBackend` exist as vestigial parallel comm —
they buffer through `RingNetwork`, which the scheduler never reads
from. Those methods, and `RingNetwork` itself, will be deleted (see
"Marked for deletion" below).

Once they're gone, the remaining `RingBackend.get_lx` covers two
distinct cases:

- **Pre-seeded LX.** Host wrote each partition before kernel start
  (e.g. `construct_distributed_memory_view`). Synchronous lookup —
  return the scratchpad reference, no scheduling needed. This is the
  current `DirectLXBackend.get_lx`.
- **Ring-transfer LX.** Remote partition has to be fetched at runtime
  via the ring. The peek must yield `RecvRequest`s the scheduler can
  drive — the same protocol as `ReduceBackend.run`.

Both cases produce a generator-or-value contract that matches the
scheduler protocol. So `RingBackend` and `ReduceBackend` end up
sharing a contract: *"a client of the scheduler protocol, generating
`send_to` calls and `RecvRequest` yields, returning a value the
scheduler binds."*

Whether to merge them under one base class or keep them as siblings
sharing a documented contract is a judgement call:

- **Sibling abstractions, shared contract** (preferred for now). They
  have different *purposes* — one fetches memory, one runs a
  reduction algorithm — and merging them creates a god-class that
  does two unrelated things just because both speak the protocol.
  Keep them separate; document the shared contract.
- **One base class.** Worth doing once a third client of the protocol
  shows up and the duplication starts to cost something concrete.

### `CoreContext.send_to` — proper attach API

The scheduler currently sets `core._send_fn` directly (a
leading-underscore attribute). Replace with an explicit attach/detach
API on `CoreContext`:

```python
core.attach_scheduler(send_fn)   # called once before running this core
core.detach_scheduler()           # called when this core's stack is done
```

Same effect, but the leak of the underscore name into `GridExecutor`
goes away, and the lifetime of the binding is documented at the
attach/detach call sites instead of being implicit. This is a small
mechanical cleanup, not a behavior change.

---

## Marked for deletion

These exist from an earlier iteration of the comm design and are **not
on the live scheduler path**. They are obsoleted by the future
direction above; deletion is independent of and should not block that
work.

| Symbol | Location | Why dead |
|---|---|---|
| `RingBackend.send` / `RingBackend.recv` | `ktir_cpu/ops/comm_ops.py` | No caller on the scheduler path. `CommOps` and `CoreContext.send_to` do not use them. |
| `RingNetwork` | `ktir_cpu/ops/comm_ops.py` | Parallel message-buffer the scheduler doesn't read from or write to. The future ring-transfer path will use the scheduler protocol directly, not a separate buffer. |
| `DirectLXBackend.send` / `DirectLXBackend.recv` | `ktir_cpu/ops/comm_ops.py` | Pass-through to `RingNetwork`; same reason. |

`RingBackend.get_lx` and `DirectLXBackend.get_lx` are live and stay —
they support remote LX peeks for distributed memory views.

---

## Where things are defined

| Symbol | File |
|---|---|
| `RecvRequest` | `ktir_cpu/grid.py` (lives here, not `comm_ops.py`, to avoid circular import) |
| `CoreContext.send_to`, `CoreContext.get_lx` | `ktir_cpu/grid.py` |
| `CoreExecutionStack` | `ktir_cpu/grid.py` |
| `GridExecutor.execute_with_communication` | `ktir_cpu/grid.py` |
| `CommOps.reduce`, `CommOps.transfer` | `ktir_cpu/ops/comm_ops.py` |
| `RingBackend`, `DirectLXBackend` | `ktir_cpu/ops/comm_ops.py` |
