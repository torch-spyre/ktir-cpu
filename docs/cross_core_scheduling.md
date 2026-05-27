# Cross-Core Scheduling — Roles & Boundaries

## One scheduler path

There is exactly one way cores communicate during execution:

```
client code           ctx.send_to(dst, tile)        — fire-and-forget
client code (yields)  RecvRequest(src)              — blocks until tile arrives
GridExecutor          owns the message queue and drives every core's generator
```

Everything else is either above this path (clients of `CommOps`) or
off it entirely (synchronous remote-LX peeks).

---

## Architecture

```
KTIRInterpreter
  └── GridExecutor.execute_with_communication(ops, input_ptrs, execute_op,
                                              transfer_backend)
        │
        ├── For each core:
        │     CoreContext.attach_scheduler(send_fn, transfer_fn)
        │       send_fn(dst, tile)     ← enqueues onto the scheduler's queue
        │       transfer_fn(src_core)  ← curried transfer_backend.run(ctx, src)
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
        │     ├── send_to(dst, tile)         ← public; uses _send_fn
        │     └── get_lx(core_id)            ← public; local fast path,
        │                                     remote case uses _transfer_fn
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
| **`CommOps`** | High-level **per-core** comm primitives. Stable surface for dialect handlers. Today: `reduce` (passthrough to a `ReduceBackend`). | Dialect handlers call into here, not into scheduler primitives directly. |
| **`CoreContext`** | Per-core SSA scope, LX scratchpad, and the public `send_to` / `get_lx` methods. Per-run scheduler bindings (`_send_fn`, `_transfer_fn`) are installed via `attach_scheduler` and cleared via `detach_scheduler`. | |
| **`CoreExecutionStack`** | Wraps one core's op list as a generator. The only place that calls `.send()` / `next()` on client generators. Bubbles `RecvRequest` to the scheduler via `yield from`. | Internal. |
| **`GridExecutor`** | Owns cores, the message queue, and the scheduler loop. Attaches scheduler bindings on every core before stepping; drives every generator to completion or raises on deadlock. | Internal. |
| **`ReduceBackend`** / **`TransferBackend`** | Pluggable algorithms / transports. `ReduceBackend.run` is per-core, may yield `RecvRequest`. `TransferBackend.run` resolves remote LX access. Siblings under the scheduler protocol; distinct purposes. | See "Pluggable …" sections. |

---

## Per-core semantics

`CommOps` primitives describe **one core's behavior**. The function
runs once per participating core; the scheduler runs N copies
concurrently; they cooperate via the message queue.

### `CommOps.reduce` — passthrough to a `ReduceBackend`

```python
@staticmethod
def reduce(ctx, tile, core_group, backend: ReduceBackend):
    return backend.run(ctx, tile, core_group)
```

The algorithm — ring rounds, LX-scratchpad accumulation, etc. — lives
in the backend. `CommOps.reduce` exists so dialect handlers and tests
have a single stable entry point regardless of which algorithm is in
play. See `RingReduceBackend` for the canonical ring algorithm and a
worked example.

---

## Generator vs plain function — the rule

A backend method (or any client of the scheduler protocol) is a
generator iff it can block.

- Generator: contains `yield RecvRequest(...)`. Calling it returns a
  generator object; the body has not yet run. The scheduler drives
  it.
- Plain function: no `yield`. Calling it runs the body to completion.

A handler that calls a generator-shaped backend must **propagate**
the generator — don't call it and discard the result:

```python
return CommOps.reduce(ctx, tile, group, backend)              # OK — scheduler drives it
return (yield from CommOps.reduce(ctx, tile, group, backend)) # OK — handler is itself a generator
CommOps.reduce(ctx, tile, group, backend)                     # BUG — generator created, body never runs
```

**Key invariant:** `send_to` is fire-and-forget — the sender never
blocks. Only `yield RecvRequest` suspends a core. This prevents
sender-side deadlock in symmetric patterns (all-to-all, ring).

---

## Control flow: blocking recv (end-to-end)

```
dialect handler / test handler
  └── CommOps.reduce(ctx, tile, group, backend)   ← passthrough
        backend.run(ctx, tile, group)             ← generator object
          for each round:
            ctx.send_to(next_core, tile)          ← enqueues immediately
            received = yield RecvRequest(src=prev) ← suspends here
            ...
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
finishes recving). The deadlock tests in `tests/test_grid_scheduler.py`
exercise three protocol-break shapes (recv-only, wrong-destination,
extra-recv) by monkey-patching `RingReduceBackend.run` and assert the
scheduler raises this error.

---

## Pluggable reduction backends

`CommOps.reduce` is a passthrough — the algorithm lives in a
`ReduceBackend`. The motivation is hardware variety: ring is one of
several plausible implementations of "combine values across a group."
Tree, recursive halving-doubling, and an LX-scratchpad reduction
(each core writes its partial into a designated LX slot on a target
core, no ring messages) all have valid cost profiles depending on
tile size, group shape, and available bandwidth.

The design separates *what* (semantics: combine N tiles into one)
from *how* (algorithm: ring vs. tree vs. LX):

```python
class ReduceBackend(ABC):
    """Owns the full reduce protocol: messaging, compute, completion."""

    @abstractmethod
    def run(self, ctx: CoreContext, tile: Tile,
            core_group: List[int]) -> Generator[RecvRequest, Tile, Tile]:
        ...
```

Concrete backends:

- **`RingReduceBackend`** *(implemented)* — N-1 ring rounds. Yields
  `N-1` `RecvRequest`s per core. The canonical algorithm.
- **`LXReduceBackend`** *(future)* — each core writes its partial into
  a target core's LX scratchpad slot via `ctx.get_lx(target_core)`.
  Synchronous (no yields) when LX is directly addressable. Useful
  when partition count is small and ring latency dominates.

### Backend selection

The dialect handler picks the backend. Selection happens at the
handler boundary, not in `CommOps` and not via the execution
environment — backend choice belongs with the IR.

The pattern uses a small explicit registry, mirroring the existing
parser/handler registries:

```python
@register("ktdp.reduce", latency_category=LC.COMM)
@register_reduce_backend("ktdp.reduce", RingReduceBackend)
def ktdp__reduce(op, context, env):
    backend_cls = get_reduce_backend(op.op_type)
    reduce_fn = lambda t1, t2: ArithOps.addf(t1, t2)
    return CommOps.reduce(context, tile, core_group, backend_cls(reduce_fn))
```

When a second backend lands, the handler resolves it from the same
registry. Whether a single op type with a `kind` attribute or a
sibling op type (`ktdp.reduce_ring`, `ktdp.reduce_lx`) is the right
shape is a future decision — both put the choice in the IR.

---

## Pluggable transfer backends

Cross-core memory access (`ctx.get_lx(other_core)`) is served by a
`TransferBackend`. The backend is supplied per execution as a direct
parameter to `GridExecutor.execute_with_communication`, which curries
it into a per-core `transfer_fn` and binds it to each core through
`CoreContext.attach_scheduler` for the duration of the run.

`transfer_backend` deliberately does **not** live on `ExecutionEnv` —
`ExecutionEnv` is for handler-visible services, and no handler reads
the backend directly.

Concrete backends:

- **`InstantTransferBackend`** *(implemented)* — synchronous lookup,
  returns the target scratchpad handle. The default for pre-seeded
  LX (e.g. `construct_distributed_memory_view`).
- **`LXTransferBackend`** *(future)* — fetches bytes via the ring,
  yielding `RecvRequest` while data crosses the ring. Same scheduler
  protocol as `RingReduceBackend`. When this lands, callers of
  `ctx.get_lx` will need to drive the resulting generator through the
  scheduler (today the path is synchronous).

`ReduceBackend` and `TransferBackend` are siblings under the same
scheduler protocol but distinct purposes (algorithm vs. memory), so
they remain separate hierarchies rather than merging under a common
base.

---

## Per-core scheduler bindings

`CoreContext` does not own a backend — only the per-run injected
functions. The scheduler binds them at the start of a run via an
explicit lifecycle:

```python
core.attach_scheduler(send_fn=..., transfer_fn=...)
# scheduler drives each core's generator
core.detach_scheduler()
```

`send_fn(dst, tile)` enqueues a tile into the scheduler's message
buffer. `transfer_fn(src_core)` is the curried
`transfer_backend.run(ctx, src_core)` and is invoked only for
genuinely-remote cores — the local fast path short-circuits inside
`ctx.get_lx` before reaching the function.

The two bindings are installed and cleared together; this replaces
the earlier implicit `core._send_fn = ...` field mutation. A
detached `CoreContext` raises clearly when `send_to` or remote
`get_lx` is called.

---

## Where things are defined

| Symbol | File |
|---|---|
| `RecvRequest`, `CoreContext`, `CoreExecutionStack`, `GridExecutor` | `ktir_cpu/grid.py` |
| `CommOps`, `ReduceBackend`, `RingReduceBackend` | `ktir_cpu/ops/comm_ops.py` |
| `TransferBackend`, `InstantTransferBackend` | `ktir_cpu/ops/comm_ops.py` |
| `register_reduce_backend`, `get_reduce_backend` | `ktir_cpu/ops/comm_ops.py` |
| `ExecutionEnv` | `ktir_cpu/dialects/registry.py` |
