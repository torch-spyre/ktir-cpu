# LX Tracking

## High-level overview

Each core has a fixed-capacity LX scratchpad (`lx.capacity`). The interpreter
tracks live usage in `lx.used` and raises `MemoryError` when an allocation
would exceed capacity.

Allocation follows MLIR's SSA scoping rules: each region (function body,
`scf.for` body, `scf.if` branch) gets its own scope on a stack.
`set_value` writes into the topmost scope and is the **single charge point**
for `lx.used` — only `Tile` values (tensor data backed by NumPy arrays) occupy
LX; metadata types (`TileRef`, `AccessTile`, scalars) do not.

At scope exit (`pop_scope`), all tiles in the popped scope are freed and
`lx.next_ptr` is rewound to the watermark saved at scope entry (`push_scope`).
This keeps the bump pointer in sync with `lx.used`.


## Complication 1: aliased tiles

A single tile (one Python object, one allocation) can be bound to multiple SSA
names simultaneously. The canonical case is `linalg.reduce`, which binds its
result to both the op's SSA result name and the `outs` buffer name:

```
%result = linalg.reduce(%input, %init) outs(%accum)
# %result and %accum now refer to the same Python Tile object
```

Charging `lx.used` once per `set_value` call would double-count the same
allocation.

**Solution: `_tile_refcount` keyed by `id(Tile)`.**

```
_tile_refcount: Dict[int, int]   # id(Tile) → reference count
```

- **First binding** (`_tile_refcount.get(id(tile), 0) == 0`): charge
  `lx.used += tile.size_bytes()`, set refcount to 1.
- **Subsequent bindings** (same `id()`): increment refcount only — no
  additional charge.
- **Overwrite** (`set_value` replacing an existing name): decrement old tile's
  refcount; free if it reaches 0.
- **Scope exit** (`pop_scope`): decrement refcount for every tile in the popped
  scope; free if it reaches 0.

This ensures each physical allocation is charged exactly once regardless of how
many SSA names point to it.


## Complication 2: single-use tiles

Scope-exit deallocation is correct but conservative. Within a loop body, tiles
that are produced and immediately consumed (used exactly once) stay live in the
scope until the body scope is popped at the end of the iteration. In a
reduction loop this inflates peak LX needlessly:

```
// %accum_zero is used exactly once — to initialize the loop's iter_arg.
// With scope-exit-only deallocation it occupies LX for the entire loop.
%accum_zero = ktdp.load(%c_view, %c_tile) : tensor<1x128xf16>
%accum_itr = scf.for ... iter_args(%accum_itr = %accum_zero) {
    %accum_next = arith.addf %accum_itr, %partial : tensor<1x128xf16>
    scf.yield %accum_next
}
```

**Solution: consume-on-last-use in `get_value`.**

At parse time, a single pass over all ops and nested regions counts how many
times each SSA name appears as an operand:

```python
_use_counts: Dict[str, int]   # SSA name → total operand occurrences
```

SSA names are unique across the function, so a flat global map suffices — no
per-region threading required.

In `get_value`, when a name is fetched for its last use, it is consumed
(removed from scope and freed) before the consuming op runs:

```python
if (isinstance(value, Tile)
        and self._use_counts.get(name, 0) == 1   # last use
        and scope is self._scope_stack[-1]):       # topmost-scope guard (see below)
    del scope[name]
    self._tile_refcount[id(value)] -= 1
    if self._tile_refcount[id(value)] == 0:
        del self._tile_refcount[id(value)]
        self.lx.used -= value.size_bytes()
return value   # caller still receives the tile data
```

The result is that the new tile produced by the consuming op can occupy the
freed slot at no net increase in `lx.used`.

**Topmost-scope guard.** Without the `scope is self._scope_stack[-1]` check,
an outer-scope name with `use_count == 1` — e.g. a constant loaded before a
loop and referenced once per iteration inside it — would be consumed on the
first iteration fetch and raise `KeyError` on subsequent iterations. The guard
restricts early-freeing to names that live in the currently-active scope,
where "currently-active" means the scope that would normally own the value's
lifetime anyway.


## Limitation: simultaneous window for iter-arg tiles

The consume-on-last-use mechanism cannot eliminate the peak-LX cost of loop
carry variables. Consider:

```
// Parent scope holds %accum_itr.
// Inside the body scope, arith.addf produces %accum_next (new Tile, new id()).
// At the moment %accum_next is charged, %accum_itr is still live in the parent
// scope — both tiles occupy LX simultaneously.
%accum_next = arith.addf %accum_itr, %partial : tensor<1x128xf16>
scf.yield %accum_next
// After body scope pops, the for-op rebinds %accum_itr := %accum_next in the
// parent scope via set_value, which frees the old %accum_itr tile.
// But the peak has already been reached.
```

`%accum_itr` lives in the parent scope, not the topmost (body) scope, so the
topmost-scope guard correctly blocks early consumption — `%accum_itr` is
fetched on every iteration, not just once.

The `set_value` rebind does reclaim the old carry tile promptly after each
iteration (refcount drops to 0 when the parent-scope name is overwritten), but
it cannot act before `%accum_next` is charged. One iteration's worth of both
the old and new carry tile must be simultaneously live.

Eliminating this window would require either explicit last-use annotations in
the IR or a liveness analysis pass that is currently out of scope.


## Feature flags: LXOptions

Both mechanisms above are controlled by `LXOptions`, a dataclass passed to
`CoreContext` alongside `LXScratchpad`:

```python
@dataclass
class LXOptions:
    alias_dedup: bool = True       # Complication 1: refcount by id(Tile)
    consume_last_use: bool = True  # Complication 2: free at last fetch
```

Both default to `True` so production behavior is unchanged. Tests use explicit
presets to isolate each feature and measure its effect on `lx.used`:

```python
_LX_BASELINE = LXOptions(alias_dedup=False, consume_last_use=False)
_LX_DEDUP    = LXOptions(alias_dedup=True,  consume_last_use=False)
_LX_FULL     = LXOptions(alias_dedup=True,  consume_last_use=True)
```

`consume_last_use` requires `alias_dedup` to be correct — the refcount must
reach 0 at the right moment for the free in `get_value` to fire. Running
`consume_last_use=True` with `alias_dedup=False` is unsupported.

