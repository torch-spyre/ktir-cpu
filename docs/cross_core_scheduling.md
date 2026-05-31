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
| **`CoreContext`** | Per-core SSA scope, LX scratchpad, and the public `send_to` / `get_lx` methods. Per-run scheduler bindings (`_send_fn`, `_transfer_fn`) are installed via `attach_scheduler` and cleared via `detach_scheduler`. | |
| **`CoreExecutionStack`** | Wraps one core's op list as a generator. The only place that calls `.send()` / `next()` on client generators. Bubbles `RecvRequest` to the scheduler via `yield from`. | Internal. |
| **`GridExecutor`** | Owns cores, the message queue, and the scheduler loop. Attaches scheduler bindings on every core before stepping; drives every generator to completion or raises on deadlock. | Internal. |
| **`ReduceBackend`** / **`TransferBackend`** | Pluggable algorithms / transports. `ReduceBackend.run` is per-core, may yield `RecvRequest`. `TransferBackend.run` resolves remote LX access. Siblings under the scheduler protocol; distinct purposes. | See "Pluggable …" sections. |

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
return backend.run(ctx, tile, group, reduce_fn)              # OK — scheduler drives it
return (yield from backend.run(ctx, tile, group, reduce_fn)) # OK — handler is itself a generator
backend.run(ctx, tile, group, reduce_fn)                     # BUG — generator created, body never runs
```

**Key invariant:** `send_to` is fire-and-forget — the sender never
blocks. Only `yield RecvRequest` suspends a core. This prevents
sender-side deadlock in symmetric patterns (all-to-all, ring).

---

## Control flow: blocking recv (end-to-end)

```
dialect handler
  └── backend.run(ctx, tile, group, reduce_fn)    ← generator object
        for each round:
          ctx.send_to(next_core, tile)            ← enqueues immediately
          received = yield RecvRequest(src=prev)  ← suspends here
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

The algorithm — ring rounds, LX-scratchpad accumulation, etc. — lives
in a `ReduceBackend`. The motivation is hardware variety: ring is one
of several plausible implementations of "combine values across a
group." Tree, recursive halving-doubling, and an LX-scratchpad
reduction (each core writes its partial into a designated LX slot on a
target core, no ring messages) all have valid cost profiles depending
on tile size, group shape, and available bandwidth.

The design separates *what* (semantics: combine N tiles into one) from
*how* (algorithm: ring vs. tree vs. LX):

```python
class ReduceBackend(ABC):
    """Owns the full reduce protocol: messaging, compute, completion."""

    @abstractmethod
    def run(self, ctx, tile, core_group, reduce_fn):
        ...
```

`reduce_fn` arrives as a parameter to `run`, not the constructor —
this lets the same backend instance back any delivery op
(`inter_tile_reduce`, `inter_tile_reduce_scatter`, …) by passing a
different fold at call time. See "Inter-tile communication" below.

Concrete backends:

- **`RingReduceBackend`** *(implemented)* — N-1 ring rounds. Yields
  `N-1` `RecvRequest`s per core. The canonical algorithm.
- **`LXReduceBackend`** *(future)* — each core writes its partial into
  a target core's LX scratchpad slot via `ctx.get_lx(target_core)`.
  Synchronous (no yields) when LX is directly addressable. Useful
  when partition count is small and ring latency dominates.

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

**Instance counts.** A `ReduceBackend` is constructed by the
`inter_tile_produce` handler — once per core per produce op. A
`TransferBackend` is constructed once per execution by the
interpreter and shared across all cores via curried per-core
`transfer_fn`s. The asymmetry reflects what each tracks: reduce
backends carry no state across calls (they could equally be free
generator functions; the class wrapper exists only because the
registry keys on a class), while a transfer backend is just a name
resolver pointing at the workgroup-shared `SpyreMemoryHierarchy`.

NOTE: for future consistency, will consider to also make
`TransferBackend` to be constructed once per transfer.

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

# Inter-tile communication: produce + reduce, end to end

This section walks the `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce`
op pair from IR through the scheduler, and explains how `%fut` is
designed as a per-core handle so no cross-core shared state is
needed at the dialect level.

The example is the 4-core all-reduce in
`examples/ktir/ring_reduce.mlir`: every core holds a
`tensor<1x128xf16>` partial, and every core ends up holding the sum.

## How an MLIR kernel reaches the scheduler

The KTIR backend simulates a multi-core accelerator. One *kernel*
(an MLIR function annotated with `grid = [N]`) runs across N cores
of the grid. **Every core runs the same kernel** — the same op list,
the same handler functions — but each core has its own
`CoreContext` (its own `core_id`, its own LX scratchpad, its own
SSA value bindings). When `ktdp.get_compute_tile_id` runs on core 3
it returns 3; on core 0 it returns 0.

This is "every core runs the same code with different identity" —
the standard accelerator execution model. There is no host
coordinator and no shared memory map at the SSA level; cores
exchange data only by going through the scheduler's mailbox.

So when an op like `ktdp.inter_tile_produce` returns a `TileFuture`,
the handler runs *N times in total* (once per core), each time
producing a separate `TileFuture` instance bound to that core's
local `%fut` SSA name. There is no single, workgroup-shared
`TileFuture` object.

### Why per-core, not workgroup-shared

The alternative is to have one workgroup-shared object that every
core's `%fut` resolves to (an earlier draft of this design used a
shared dict on `GridExecutor`).  Both shapes can satisfy the spec's
"workgroup-visible" property — the difference is where the
abstraction line falls.

The per-core design is the natural fit for two reasons:

1. **Consistent execution model across delivery patterns.**  When
   the future drives a per-tile dependency
   (`producer_dependency_per_consumer` — e.g. butterfly mirror or
   pairwise exchange), each consumer needs a distinct subset of
   producers.  In a shared-future model, the consumer would have to
   reach into someone else's contribution slot in the dict.  In the
   per-core model, the same scheduler protocol that already moves
   plain tiles (`send_to` + `RecvRequest`) moves the per-tile
   partials too — one mechanism, one ordering story.

2. **No out-of-band shared state at the dialect level.**  Cross-core
   data flows exclusively through the scheduler's mailbox, which we
   already trust as the single visibility boundary.  The dialect
   handler stays a thin translator from IR to obligation; it doesn't
   need a private side channel.

The trade-off: in the all-reduce case the per-core model does
**redundant compute** — every core runs the producer region and
materialises its own partial, even though logically one reusable
materialisation could feed all consumers.  We accept that cost (it
matches what the hardware actually does — every tile runs the
kernel) in exchange for one execution model that covers the full
inter-tile op family.

### Counting comm cost — charge per message at delivery

`send_to` is fire-and-forget on the sender; the actual ring
bandwidth gets consumed when a message is **delivered** to its
destination — i.e. inside `_try_deliver`, after the message has
been popped off `messages[(src, dst)]`.  That's the natural place
to charge the latency model.

Sends from different cores are disjoint events (each `send_to`
queues a distinct `(src, dst, payload)` triple), so there is no
double-counting in the message-count sense.  What the per-handler
`@register("ktdp.inter_tile_reduce", latency_category=LC.COMM)`
annotation gets *wrong* is something different: it charges
fixed-per-op cycles N times (once per core) and re-derives bytes
from the dialect-level operand sizes, which mismatches what the
transport actually moved when delivery patterns are asymmetric
(per-tile sync, butterfly).  Per-message charging at delivery is
exact for any topology.

**Hook into the existing latency surface.**  The latency tracker
already exposes `record_op(core_id, op_type, result, operands)`
which the interpreter calls once per dialect-handler invocation.
The scheduler can call the same method at delivery, with a
synthetic op type that the tracker recognises as a wire-transit
charge:

```python
# in GridExecutor._try_deliver, after pop succeeds:
self._latency_tracker.record_op(
    core_id=dst,                  # the receiving core absorbs the wire
    op_type="scheduler.message",  # resolves to LC.COMM in latency.py
    result=tile,                  # gives _comm_size the byte count
    operands=[],
)
```

In `latency.py`'s `_estimate`, the `LC.COMM` arm becomes:

```python
if category == LC.COMM:
    nbytes = self._data_size(result, [])
    cycles = nbytes / self.config.ring_bytes_per_cycle
    return ("comm", cycles, 0.0, nbytes)
```

— with the existing
`if op_type == "ktdp.inter_tile_reduce": cycles *= rounds`
special-case removed (the rounds are real messages now, each
charged once).  And `inter_tile_*` registrations drop
`latency_category=LC.COMM` so the per-handler call no longer
contributes a parallel charge.

**Provenance for breakdowns.**  For "how much comm time was
`ktdp.inter_tile_reduce`?", tag the message at `send_to` time with
the dialect op currently running on the sender, and pass that tag
through to `record_op` at delivery:

```python
core.send_to(dst, tile, op_type="ktdp.inter_tile_reduce")  # tagged
# scheduler later:
self._latency_tracker.record_op(dst, msg.op_type, msg.payload, [])
```

For v1 a single bucket (`"scheduler.message"`) is enough; tagging
is additive when we want per-op breakdowns.

**Status.**  Unimplemented today.  The current code carries
`LC.COMM` on `inter_tile_*` and the tracker double-charges across
cores.  See the
[GitHub issue]() (TODO once filed) for the migration plan.

## Two functions, two phases

The four-op design (see [RFC 0682](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)
§5 and
[`docs/inter-tile-communication.md`](https://github.com/torch-spyre/ktir-mlir-frontend/blob/main/docs/inter-tile-communication.md)
in the frontend repo) hands the
backend two different functions at different points:

| Function | Defined by | Bound at | Used at |
|---|---|---|---|
| **producer function** | the body of `inter_tile_produce`'s region (`yield_partial %x`) | produce-handler time | produce-handler time — execute it to materialise this core's partial |
| **reducer function** | the body of the delivery op's region (`yield_reduced %sum`) | consume-handler time | consume-handler time — passed to `backend.run` so the transport knows how to fold partials |

Concretely the two handlers do this:

```python
# inter_tile_produce handler — runs once per core
local_partial = run_producer_region()        # executes yield_partial body
fut = TileFuture(
    local_partial = local_partial,
    backend       = RingReduceBackend(),     # not-yet-running
    producer_set  = ...,                     # parsed IR sets
    groups_set    = ...,
    group_idx     = this_core_group,
)
return fut                                    # bound to %fut on THIS core only

# inter_tile_reduce handler — runs once per core
fut        = ctx.get_value(%fut)
core_group = compute_core_group(fut, op.attributes)
reduce_fn  = build_reducer_from_region(op.regions[0])
return fut.backend.run(ctx, fut.local_partial, core_group, reduce_fn)
```

The produce handler builds the future and returns immediately — no
cross-core movement, no blocking. The consume handler hands a
`reduce_fn` to the future's backend; that's where the
generator-driven blocking begins.

## Why `%fut` is per-core

The KTIR spec calls `%fut` "workgroup-visible" — every consumer
must be able to see the partials the producers contributed. We
satisfy that *logically*: when a consumer reads `%fut` on its core
and triggers `backend.run`, the data it needs arrives via the
scheduler's mailbox. Physical visibility is the transport's job.

So per-core futures are the natural fit. Each core's `%fut` SSA
binding holds a different `TileFuture` instance, carrying *that*
core's local partial. Cores never read each other's futures — they
read each other's partials through `ctx.send_to` + `RecvRequest`,
the same path every other comm op uses. This negates the need for 
a class of shared-state bookkeeping in the design. The drawback is
we need to design to avoid double counting messages that are actually
shared in an actual implementation.

The def-use edge `%fut → consume(%fut)` is a *per-core* edge that
serialises produce-then-consume on each individual core. Cross-core
ordering is provided by the transport.

### Def-use is simulated by `TileFuture`, not walked

The spec uses the SSA def-use edge for two roles: (1) pin produce
before consume in execution order, and (2) identify *which* produce
a delivery op is paired with (the spec's "single-use" invariant).
This simulator does not walk the IR's def-use graph for either
role.  Both are absorbed by the per-core `TileFuture` object:

- **Ordering.**  Each core executes its IR top-down; the consume
  handler reads `%fut` via `ctx.get_value`, which only succeeds if
  the produce op already bound it.  Standard SSA scoping is the
  ordering check.
- **Pairing.**  The `TileFuture` instance returned by produce *is*
  the edge.  When the consume handler picks up `%fut`, it gets
  exactly the future the matching produce built — there's no
  IR-level "trace from this consume back to its produce" step
  because the producer-side data already lives on the future the
  consumer holds.

Spec invariants that *do* require checking the producer-side
attributes against the consumer-side attributes — `groups` match,
`producer_dependency_per_consumer ⊆ producer_tiles_per_group`,
coverage of producers by consumers' deps — are not enforced today.
They are deferred until the upstream spec finalises (see
`ktir_cpu/dialects/ktdp_ops.py` "Verification — deferred"); when
they land they'll fit in `CommPlan.for_reduce` (subset / coverage)
and a small consume-handler-entry check (`groups` match via
bounded enumeration over `ctx.num_cores`).

## RingReduceBackend — what the generator does

Per-core (one generator instance per core, all running the same code):

```python
def run(self, context, tile, core_group, reduce_fn):
    if context.core_id not in core_group:
        return tile     # opt out — never yields

    n_cores = len(core_group)
    my_idx     = core_group.index(context.core_id)
    next_core  = core_group[(my_idx + 1) % n_cores]
    prev_core  = core_group[(my_idx - 1) % n_cores]

    result      = tile.copy()   # accumulator
    to_forward  = tile.copy()   # what to send next round (round 1: local tile)

    for _ in range(n_cores - 1):
        context.send_to(next_core, to_forward)        # → scheduler queue
        received = yield RecvRequest(src=prev_core)   # ← scheduler delivery
        result = reduce_fn(result, received)
        to_forward = received                         # forward received tile, not accumulator

    return result
```

State per generator instance:

- `result`, `to_forward` — the protocol's local working set.
- `core_group`, `n_cores`, `my_idx`, `next_core`, `prev_core` —
  derived once at entry.

The **algorithmic invariant** (forward `received`, not `result`) is
what keeps each starting tile visiting every other core exactly once
over `n_cores - 1` rounds. Sending the accumulator instead would
double-count the reduction result.

## End-to-end trace for the 4-core all-reduce

`core_group = [0, 1, 2, 3]`. Local tiles `t0`, `t1`, `t2`, `t3`.

```
Setup loop (each core runs until first yield):

core 0:  send_to(1, t0)   →  messages[(0,1)] = [t0]
         yield RecvRequest(src=3)              waiting[0] = 3
core 1:  send_to(2, t1)   →  messages[(1,2)] = [t1]
         yield RecvRequest(src=0)              waiting[1] = 0
core 2:  send_to(3, t2)   →  messages[(2,3)] = [t2]
         yield RecvRequest(src=1)              waiting[2] = 1
core 3:  send_to(0, t3)   →  messages[(3,0)] = [t3]
         yield RecvRequest(src=2)              waiting[3] = 2

State after setup:
  messages = {(0,1):[t0], (1,2):[t1], (2,3):[t2], (3,0):[t3]}
  waiting  = {0:3, 1:0, 2:1, 3:2}

Delivery loop drives each core through the remaining rounds. Per-core
view of all three rounds:

core 0:  starts with t0
         R1: send t0→1, recv t3 from 3,  result=t0+t3,        forward=t3
         R2: send t3→1, recv t2 from 3,  result=t0+t3+t2,     forward=t2
         R3: send t2→1, recv t1 from 3,  result=t0+t3+t2+t1,  done
core 1:  starts with t1
         R1: send t1→2, recv t0 from 0,  result=t1+t0,        forward=t0
         R2: send t0→2, recv t3 from 0,  result=t1+t0+t3,     forward=t3
         R3: send t3→2, recv t2 from 0,  result=t1+t0+t3+t2,  done
core 2:  starts with t2
         R1: send t2→3, recv t1 from 1,  result=t2+t1,        forward=t1
         R2: send t1→3, recv t0 from 1,  result=t2+t1+t0,     forward=t0
         R3: send t0→3, recv t3 from 1,  result=t2+t1+t0+t3,  done
core 3:  starts with t3
         R1: send t3→0, recv t2 from 2,  result=t3+t2,        forward=t2
         R2: send t2→0, recv t1 from 2,  result=t3+t2+t1,     forward=t1
         R3: send t1→0, recv t0 from 2,  result=t3+t2+t1+t0,  done
```

Each generator returns its final `result`; `_execute_until_block`
rebinds `%reduced` to the actual tile; the scheduler tears down the
stack.

## Where state lives

| State | Owner | Lifetime |
|---|---|---|
| `messages: Dict[(src,dst), deque]` | `GridExecutor.execute_with_communication` (local var) | One execution invocation |
| `waiting: Dict[core_id, src]` | same | One execution invocation |
| `stacks: Dict[core_id, CoreExecutionStack]` | same | One execution invocation |
| `result`, `to_forward` | local frame inside each suspended `RingReduceBackend.run` generator | Per-call; reclaimed when the generator returns |
| `core_group`, `reduce_fn` | passed as args to `RingReduceBackend.run` | Per-call |
| `_send_fn` on `CoreContext` | wired in setup loop via `attach_scheduler` | One execution invocation |
| `TileFuture` instance (per `%fut`, per core) | the producing core's `CoreContext` SSA scope | Until `inter_tile_reduce` consumes it |

No cross-core shared state at the dialect level. Everything cross-core
flows through the scheduler's mailbox; everything per-core lives in a
`CoreContext` SSA binding or a suspended generator frame.

---

## Where things are defined

| Symbol | File |
|---|---|
| `RecvRequest`, `CoreContext`, `CoreExecutionStack`, `GridExecutor` | `ktir_cpu/grid.py` |
| `ReduceBackend`, `RingReduceBackend` | `ktir_cpu/ops/comm_ops.py` |
| `TransferBackend`, `InstantTransferBackend` | `ktir_cpu/ops/comm_ops.py` |
| `register_reduce_backend`, `get_reduce_backend` | `ktir_cpu/ops/comm_ops.py` |
| `TileFuture` | `ktir_cpu/ir_types.py` |
| `ktdp.inter_tile_produce` / `ktdp.inter_tile_reduce` handlers | `ktir_cpu/dialects/ktdp_ops.py` |
| `ExecutionEnv` | `ktir_cpu/dialects/registry.py` |
