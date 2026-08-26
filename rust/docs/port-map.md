# grid+context

## CoreContext

**Fields:**
- `core_id: int` — Unique linear ID for this core
- `grid_pos: Tuple[int, int, int]` — (x, y, z) position in grid; derived once from core_id, immutable
- `lx: LXScratchpad` — Per-core local scratchpad (2 MB capacity); reference to memory object owned by SpyreMemoryHierarchy
- `hbm: HBMSimulator` — Shared HBM across all cores; reference to global memory object
- `_scope_stack: List[Dict[str, Any]]` — Region-scoped SSA value map; one dict per scope. Function body is bottom scope (index 0). Each dict is name → value.
- `_lx_bytes: Dict[str, int]` — SSA name → LX byte size; single source of truth for lx.used. Grows as track_lx() called; shrinks as untrack_lx() called.
- `_lx_next_ptr_stack: List[int]` — Bump-allocator watermarks; snapshot at each push_scope(), restored at pop_scope(). Invariant: `len(_lx_next_ptr_stack) == len(_scope_stack) - 1` always.
- `_send_fn: Optional[Callable[[int, Tile], None]]` — Scheduler-managed function to enqueue a tile to dst_core. Set by attach_scheduler(), cleared by detach_scheduler(). Used by send_to().
- `_transfer_fn: Optional[Callable[[int], LXScratchpad]]` — Scheduler-managed function to fetch remote core's LX. Set by attach_scheduler(), cleared by detach_scheduler(). Used by get_lx().

**Methods:**

- `__init__(core_id: int, grid_pos: Tuple[int, int, int], lx: LXScratchpad, hbm: HBMSimulator) -> None` — Initialize core context with identity, position, and memory references.

- `attach_scheduler(send_fn: Callable[[int, Tile], None], transfer_fn: Callable[[int], LXScratchpad]) -> None` — Wire cross-core communication for one execute_with_communication session. Stores send_fn and transfer_fn for use by send_to() and get_lx(). Called once per core at start of execute_with_communication; cleared by detach_scheduler() at end.

- `detach_scheduler() -> None` — Nullify _send_fn and _transfer_fn after a run completes.

- `get_lx(core_id: Optional[int] = None) -> LXScratchpad` — Return LX for a core. If core_id is None or == self.core_id, fast path returns self.lx directly. If remote core_id, calls _transfer_fn(core_id) (raises RuntimeError if _transfer_fn is None). Owner: does not mutate the returned LX, just returns reference.

- `get_grid_id(dim: int) -> int` — Return grid_pos[dim]. Dimension 0=x, 1=y, 2=z.

- `push_scope() -> None` — Enter a region (scf.for body, scf.if branch). Appends watermark (lx.next_ptr) to _lx_next_ptr_stack, appends empty dict to _scope_stack.

- `pop_scope() -> None` — Exit current region. Pops topmost scope dict, calls untrack_lx() for all names in it (freeing bytes from lx.used and _lx_bytes), rewinds lx.next_ptr to watermark. Raises if called on function-body scope (len(_scope_stack) <= 1).

- `send_to(dst_core: int, tile: Tile) -> None` — Enqueue tile for delivery to dst_core. Calls _send_fn(dst_core, tile); raises RuntimeError if _send_fn is None.

- `set_value(name: str, value: Any) -> None` — Bind SSA value in topmost scope (_scope_stack[-1]). Multiple SSA names may alias same Python object; Python reference semantics apply.

- `get_value(name: str) -> Any` — Lookup SSA value by searching _scope_stack top-to-bottom. Raises KeyError if not found in any scope.

- `has_value(name: str) -> bool` — Return True if value exists in any scope.

- `clear_values() -> None` — Reset for next execution round: reinitialize _scope_stack to [{}], clear _lx_bytes, clear _lx_next_ptr_stack, call lx.clear().

- `track_lx(name: str, size_bytes: int) -> None` — Record SSA value occupying size_bytes in LX. Increments lx.used; stores size in _lx_bytes[name]. Raises MemoryError if lx.used + size_bytes > lx.capacity (capacity is 2 MB per core; grid.py:280).

- `untrack_lx(name: str) -> None` — Free LX for name. Decrements lx.used by the stored byte size; pops name from _lx_bytes. No-op if name not in _lx_bytes.

**Key Invariants:**
- Region scoping: len(_lx_next_ptr_stack) == len(_scope_stack) - 1 at all times. Function body is always index 0 of _scope_stack.
- SSA immutability: Values are not mutated once set, only scope-exited.
- Scope stack is searchable: get_value finds in topmost-first order, allowing inner regions to shadow/read outer values.
- LX coherence: lx.used equals sum of all values in _lx_bytes. This is the "single source of truth" (grid.py:93).
- Scheduler attachment: _send_fn and _transfer_fn are None outside execute_with_communication, preventing accidental use in single-core tests that never attach.

**Python-isms and Redesign Notes:**
- **Duck typing on values**: _scope_stack holds `Any`. Rust needs a tagged enum or trait object. Tile is special-cased in track_lx (grid.py:339-340): if result is Tile, auto-track LX. In Rust, track_lx must be called explicitly after setting a value; no implicit tracking.
- **Generator yield in CoreExecutionStack**: Comm ops return Python generators that yield RecvRequest. Rust will not have generators; instead, comm handlers return a Result or Option that the scheduler interprets.
- **Scope-lifetime deallocation**: Bump allocator with watermark rewinding. Rust can implement this identically with a Vec<usize> for watermarks and lx.next_ptr managed as a mutable reference.
- **Message queues**: messages dict uses tuple (src, dst) as key with deque as value. Rust can use a HashMap<(u32, u32), VecDeque<Tile>>.
- **Mutable shared state**: waiting and results dicts are mutated by _advance, _try_deliver closures. In Rust, these would be mutable borrows in the scheduling loop.

**LX-liveness (peak-LX accuracy — #134 / #118):**
The plain scope-lifetime model above (free a tile only when its defining scope
exits) overcharges peak LX. Two refinements tighten it to match the Python
reference:

- **Consume-single-use-at-last-use (#134).** A Tile with `use_count == 1` in the
  global use-count map is freed at its single fetch instead of at scope exit.
  Python: `LXOptions.consume_last_use` (`ktir_cpu/memory.py:103–139`), driven by
  `KTIRParser._build_use_counts` (`ktir_cpu/parser.py:276–285`) and checked in
  `CoreContext.get_value` (`ktir_cpu/grid.py:321–325`). Rust:
  `consume_if_last_use` (`ktir-emulator/src/machine_state/context.rs:514`),
  **gated to `cur_gen != 0`** (`:528`) — it only fires inside an scf.for / scf.if
  body, because at function top level the resident scheduler owns liveness via
  `dies_at`/`forget` and the Metal map-window / matmul-loop offloads read tiles
  at a deferred point after their nominal last use (eager top-level consume
  raced those reads — diverged the e2e golden ~30 logits).
- **iter_arg LX double-count removal (#118).** A single physical allocation
  aliased by several SSA ids (aliases, scf.for iter_arg rebinds) must be charged
  once. Python tracks this with a refcount so iter_arg rebinds increment rather
  than re-charge (`ktir_cpu/grid.py:257–289`). Rust:
  `tile_refcount: FxHashMap<usize, (refcount, bytes)>`
  (`ktir-emulator/src/machine_state/context.rs:117`) keyed on the physical
  allocation pointer; `track_lx_tile` (`:370`) looks the tile up by
  `Tile::data_ptr()` (`ktir-core/src/tile.rs:216`, the `Rc::as_ptr` of the
  backing storage, identical across clones) and frees the bytes only when the
  refcount drops to 0.

---

## CoreExecutionStack

**Fields:**
- `core: CoreContext` — Reference to this core's context.
- `waiting_on: Optional[int]` — If blocked, the src core_id this core is waiting on. None if running or done.
- `_gen: Generator` — Python generator wrapping _execute_until_block. Yields RecvRequest or returns final result.

**Methods:**

- `__init__(core: CoreContext, operations: List[Operation], input_ptrs: Dict[str, Any], execute_op: Callable[[Operation, CoreContext], Any]) -> None` — Create stack. Binds input_ptrs into core scopes and wraps operation sequence in generator.

- `resume(send_val: Any = None) -> Any` — Step generator. If send_val is provided, calls gen.send(send_val); else calls next(gen). Catches RecvRequest and stores src in waiting_on; catches StopIteration and returns e.value (final result). Raises TypeError if yielded value is not RecvRequest.

- `is_blocked() -> bool` — Return waiting_on is not None.

**Key Invariants:**
- Generator lifecycle: _gen is created once and stepped only via resume(). Each resume() either advances to next recv (blocked) or finishes.
- RecvRequest handling: Only valid yield value from comm op. Any other type raises TypeError.

**Python-ism: Generators**
- Python generator protocol (yield, send, StopIteration) has no direct Rust equivalent. Rust scheduler will use an explicit state machine (enum with Running / Blocked / Done states) or an async runtime.

---

## GridExecutor

**Fields:**
- `grid_shape: Tuple[int, int, int]` — (nx, ny, nz) dimensions of core grid.
- `memory: SpyreMemoryHierarchy` — Shared memory hierarchy; owns HBM and per-core LX.
- `num_cores: int` — Total cores = nx * ny * nz.
- `cores: List[CoreContext]` — One CoreContext per core ID (linear order 0..num_cores-1).

**Methods:**

- `__init__(grid_shape: Tuple[int, int, int], memory: SpyreMemoryHierarchy) -> None` — Create grid. Allocates CoreContext for each core_id in range(num_cores), assigning each (x,y,z) position via _linear_to_grid and fetching its LX from memory.

- `_linear_to_grid(core_id: int) -> Tuple[int, int, int]` — Convert linear core ID to (x, y, z). Formula: z = core_id // (nx * ny); remainder = core_id % (nx * ny); y = remainder // nx; x = remainder % nx. Deterministic bijection.

- `_grid_to_linear(x: int, y: int, z: int) -> int` — Inverse: z * (nx * ny) + y * nx + x.

- `get_core(core_id: int) -> CoreContext` — Return cores[core_id].

- `get_core_at_pos(x: int, y: int, z: int = 0) -> CoreContext` — Return core at position. Calls _grid_to_linear then get_core.

- `get_cores_in_group(grid_coords: Tuple[int, int, int]) -> List[int]` — Return list of core IDs matching a masked coordinate tuple. -1 in any dimension means "all cores in that dimension." Example: (-1, 2, 0) returns all cores with y=2, z=0 (any x).

- `execute_with_communication(operations: List[Operation], input_ptrs: Dict[str, Any], execute_op: Callable[[Operation, CoreContext], Any], transfer_backend: Optional[TransferBackend] = None) -> List[Any]` — Drive all cores to completion via event-loop scheduler.

  **Execution model:**
  - Creates one CoreExecutionStack per core.
  - Before starting, attaches scheduler state (send_fn, transfer_fn) to each core via attach_scheduler (grid.py:519–531).
  - send_fn queues tile to messages dict keyed by (src, dst).
  - transfer_fn calls transfer_backend.run(ctx, src) if backend is not None; else raises RuntimeError (grid.py:526–530).
  - Calls _advance(core_id) for each core to run ops until blocked or done.
  - Scheduler loop (grid.py:535–541): repeatedly calls _try_deliver(core_id) for blocked cores. _try_deliver pops a message from messages[(src, core_id)] and resumes the core. If no core makes progress, raises RuntimeError("Deadlock detected: ...").
  - Returns list of results indexed by core_id (grid.py:543).

  **Mutation:**
  - messages dict: tiles appended by _enqueue, popped by _pop.
  - stacks dict: cores added at init, removed when done by _advance (grid.py:506).
  - waiting dict: added/removed by _try_deliver and _advance (grid.py:503, 515).
  - results dict: populated by _advance (grid.py:505).
  - Each core's scope, LX, and SSA values mutated by execute_op via CoreContext.

  **Error handling:**
  - Per-core exceptions caught and re-raised with core_id note (grid.py:499–501).
  - Deadlock on unresolvable waits (grid.py:541).
  - RecvRequest type validation in CoreExecutionStack.resume (grid.py:348).
  - transfer_fn raises if called without backend (grid.py:527–530).

**Key Invariants:**
- Grid is immutable after construction. cores list is fixed-size.
- Linear ↔ grid conversion is deterministic and reversible.
- Scheduler is single-threaded; no true parallelism. Cores interleave via generator stepping.
- Each core's operations are the same but context (grid_pos, core_id, LX) is unique.
- Deadlock detection: if no core makes progress after one full loop of _try_deliver, execution fails.

**Python-isms and Redesign Notes:**
- **Closure over local state**: _enqueue, _pop, _advance, _try_deliver closures capture messages, stacks, waiting, results (grid.py:483–517). Rust will need explicit structs or a scheduler object with these as fields.
- **Dict-based message routing**: messages uses (src, dst) tuple as key. Rust should use a HashMap<(u32, u32), VecDeque<Tile>>.
- **Generator-based event loop**: The scheduler drives generators (CoreExecutionStack._gen) via send/next. Rust will replace with explicit state machine or async/.await.
- **Lambda captures and call conventions**: send_fn and transfer_fn are lambdas capturing core.core_id, core, and transfer_backend (grid.py:520–530). Rust will use function pointers or closures that capture via mutable borrow.
- **Return type heterogeneity**: execute_with_communication returns List[Any] (results can be anything execute_op produces). Rust will need a generic or trait object, or a tagged enum.

---

## RecvRequest

**Fields:**
- `src: int` — Linear core ID to receive from.

**Semantics:**
Frozen dataclass. Yielded by comm generator to signal scheduler that the core is blocked waiting for a tile from src. Scheduler delivers the tile via gen.send(tile) when available, or raises RuntimeError("Deadlock") if src never sends. No Python-ism specific to this type.

---

## Handler Access Patterns

**How a handler reaches scope:**
- Handler receives CoreContext as argument.
- Handler calls ctx.get_value(name) to look up SSA values (top-to-bottom scope search).
- Handler calls ctx.set_value(name, value) to bind results (always to topmost scope).

**How a handler reaches LX:**
- Handler calls ctx.get_lx() or ctx.get_lx(core_id) to obtain LXScratchpad.
- Local core: returns ctx.lx directly.
- Remote core: calls _transfer_fn(core_id), which invokes transfer_backend.run(ctx, src). Raises if no scheduler attached.
- Handler calls ctx.track_lx(name, size_bytes) to record allocations after creating Tiles.

**How a handler reaches grid coords:**
- Handler calls ctx.get_grid_id(dim) to fetch x, y, or z (grid.py:145–154).
- Alternatively, handler reads ctx.grid_pos directly for all three coords.

**How a handler reaches neighbor cores:**
- Handler calls GridExecutor.get_cores_in_group(grid_coords) with masked tuple to find neighbors.
- Example: to find all cores with same x and z, call get_cores_in_group((ctx.get_grid_id(0), -1, ctx.get_grid_id(2))).
- get_cores_in_group returns list of core_ids; handler can then call ctx.get_lx(core_id) to access neighbor's LX or send_to(neighbor_id, tile) to enqueue a message.

**Cross-core communication flow:**
1. Handler calls ctx.send_to(dst_core, tile).
2. send_to calls _send_fn(dst_core, tile) → _enqueue(self.core_id, dst_core, tile) → messages[(self.core_id, dst_core)].append(tile).
3. Scheduler's _try_deliver loops; when destination core is blocked on src recv, _pop(src, dst) retrieves tile and _advance resumes the waiting core via stack.resume(tile).

---

# memory-sim

## HBMSimulator

**File**: `/Users/moosevan/git/ktir-cpu/ktir_cpu/memory.py` lines 201–289

**Public types and fields**:
- `STICK_BYTES: int = 128` — constant; HBM interleaved every 128 bytes.
- `size_gb: int` — HBM capacity in GB (default 128).
- `size_bytes: int` — capacity in bytes (`size_gb * 1024 * 1024 * 1024`).
- `memory: Dict[int, np.ndarray]` — sparse dict-based storage mapping byte address → ndarray allocation.
- `next_ptr: int` — next unallocated byte address (stick-aligned); initialized to `0x10000`.

**Methods**:

| Signature | Semantics |
|-----------|-----------|
| `allocate(size: int) -> int` | Allocate *size* bytes, advance *next_ptr* to next stick boundary using `(x + STICK_BYTES - 1) & ~(STICK_BYTES - 1)`, return stick address (`ptr // STICK_BYTES`). Mutates `next_ptr`. Asserts `next_ptr % STICK_BYTES == 0` pre and post. No enforced capacity limit (host allocator handles HBM budget). |
| `read(stick: int, n_elements: int, dtype: str, *, intra_byte: int = 0) -> np.ndarray` | Read *n_elements* from stick address *stick* + *intra_byte* offset. Calls `_read_flat(memory, stick * STICK_BYTES + intra_byte, ...)`. Returns flat ndarray, zero-pads if read extends past allocation. |
| `write(stick: int, data: np.ndarray, *, intra_byte: int = 0)` | Write flat *data* at stick address *stick* + *intra_byte*. Calls `_write_flat(memory, stick * STICK_BYTES + intra_byte, data)`. Patches existing allocation in-place or creates new one. |
| `read_element(addr: int, dtype: str = "f16")` | Deprecated: read one element by byte address. Uses `_find_allocation`. Returns `0.0` (f16) if unmapped. |

**Invariants**:
- `size_bytes` is tracked but **not enforced** during allocation. Kernels only reference host-placed tensors; no new HBM allocation by kernel itself.
- All byte addresses must be stick-aligned for API surface; internal `_read_flat` / `_write_flat` accept arbitrary byte offsets within allocations.
- Sparse dict: only touched allocations exist in `memory`; an unallocated address raises ValueError on read.

---

## LXScratchpad

**File**: `/Users/moosevan/git/ktir-cpu/ktir_cpu/memory.py` lines 290–338

**Public types and fields**:
- `size_mb: int` — capacity in MB (default 2).
- `capacity: int` — capacity in bytes (`size_mb * 1024 * 1024`).
- `used: int` — tracked but **never enforced** (unlike HBM, allocation is implicit in SSA lifetime).
- `core_id: int` — which core owns this scratchpad.
- `memory: Dict[int, np.ndarray]` — sparse dict-based storage mapping local byte address → ndarray.
- `next_ptr: int` — next unallocated local address; initialized to `0`.

**Methods**:

| Signature | Semantics |
|-----------|-----------|
| `read(ptr: int, n_elements: int, dtype: str) -> np.ndarray` | Read *n_elements* from local byte address *ptr*. Calls `_read_flat(memory, ptr, ...)`. Returns flat ndarray, zero-pads if read extends past allocation. Raises ValueError if *ptr* unmapped. |
| `write(ptr: int, data: np.ndarray)` | Write flat *data* at local byte address *ptr*. Calls `_write_flat(memory, ptr, data)`. Patches in-place or creates new allocation. |
| `clear()` | Reset scratchpad: empty `memory` dict, reset `next_ptr` to 0, reset `used` to 0. |

**Invariants**:
- Each SSA Tile value occupies LX from creation (via `load` or compute op) until its defining scope exits.
- CoreContext uses `_scope_stack` mirroring MLIR's region structure; on `pop_scope`, all values in that scope are untracked and their LX freed.
- The 2 MB limit is the real constraint for tile coexistence in a single iteration; not enforced here—CoreContext.track_lx() does the accounting.
- No per-element dict scans: a single contiguous allocation covers each SSA value's entire lifetime.

---

## SpyreMemoryHierarchy

**File**: `/Users/moosevan/git/ktir-cpu/ktir_cpu/memory.py` lines 339–355

**Public types and fields**:
- `num_cores: int` — number of cores.
- `hbm: HBMSimulator` — shared HBM across all cores.
- `lx_scratchpads: List[LXScratchpad]` — one per core, indexed by core_id.

**Methods**:

| Signature | Semantics |
|-----------|-----------|
| `get_lx(core_id: int) -> LXScratchpad` | Route to the LX scratchpad for core *core_id*. Returns `lx_scratchpads[core_id]`. |

---

## _MemAccessor (internal utility)

**File**: `/Users/moosevan/git/ktir-cpu/ktir_cpu/ops/memory_ops.py` lines 34–168

Abstracts HBM vs. LX dispatch. Single place managing stick-byte offset logic for HBM.

**Fields**:
- `_memory_space: str` — "HBM" or "LX".
- `stick_bytes: Optional[int]` — `HBMSimulator.STICK_BYTES` (128) for HBM, None for LX.
- `_sim` — reference to HBMSimulator or LXScratchpad.
- `_args: Tuple[int, ...]` — (stick,) for HBM or (byte_addr,) for LX.
- `_kwargs: Dict` — {"intra_byte": intra} for HBM or {} for LX.

**Methods**:

| Signature | Semantics |
|-----------|-----------|
| `__init__(context, memory_space: str, byte_addr: int, lx_core_id: Optional[int])` | Route *byte_addr* to HBM (split into stick + intra_byte via divmod) or LX (direct). For LX, route via `context.get_lx(lx_core_id)` when *lx_core_id* is set; else use `context.lx` directly. |
| `count_sticks(memory_space: str, byte_addresses: Iterable[int]) -> Optional[int]` | Class method. Count distinct HBM sticks: `len({a // STICK_BYTES})`. For LX, return None. Empty input on HBM path returns 0 (distinct "no stick traffic"). |
| `read(n: int, dtype: str) -> np.ndarray` | Dispatch to `_sim.read(*_args, n, dtype, **_kwargs)`. |
| `read_scattered(byte_addresses: List[int], dtype: str) -> (np.ndarray, Optional[int])` | Batch scatter-read: deduplicate & sort addresses, merge adjacent ones (diff == bytes_per_elem), issue one `_sim.read` per contiguous run (DMA descriptor granularity). Return (values in caller's order, unique_sticks). Call raises ValueError on empty input. **Known issue** (lines 125–131): cross-allocation merging is silently wrong—no guard yet. |
| `write(data: np.ndarray) -> None` | Dispatch to `_sim.write(*_args, data, **_kwargs)`. |

**_read_flat & _write_flat (module-level, lines 97–199)**:
- `_find_allocation(memory: Dict, ptr, elem_size) -> Optional[(base_ptr, data, elem_offset)]`: Find the allocation containing byte address *ptr*. Return (base_ptr, array, flat element offset) or None. Note: ptr in memory check (line 112) is first for efficiency; then line 125 skips base_ptr == ptr case.
- `_read_flat(memory: Dict, ptr, n_elements, np_dtype, elem_size) -> np.ndarray`: Read *n_elements* from *ptr*, zero-padding if read extends past allocation end. Raises ValueError if *ptr* unmapped.
- `_write_flat(memory: Dict, ptr, data: np.ndarray)`: Write flat *data* at *ptr*. Patch in-place when ptr falls within existing allocation; create new allocation if unmapped. Handles partial-write (data extends past allocation end).

---

## MemoryOps

**File**: `/Users/moosevan/git/ktir-cpu/ktir_cpu/ops/memory_ops.py` lines 312–967

Core load/store and tile access logic. All public; static methods.

### View construction

| Signature | Semantics |
|-----------|-----------|
| `tile_view(context, ptr: int, shape: Tuple[int, ...], strides: List[int], memory_space: str, dtype: str = "f16", coordinate_set: Optional[str] = None, lx_core_id: Optional[int] = None) -> MemRef` | Build a MemRef describing a contiguous region in HBM or LX. No data movement. Wraps input in MemRef with *lx_core_id* parsed from spyre_memory_space attribute. |
| `tile_access(context, parent_ref: MemRef, indices: List[int], access_shape: Tuple[int, ...], base_map: AffineMap) -> TileRef` | Extract sub-tile: eval base_map(*indices*) → base coords, compute byte offset via dot(base_coords, strides) * bpe, return TileRef. Byte address must fall within parent's allocation (invariant). |

### Contiguity test

| Signature | Semantics |
|-----------|-----------|
| `_is_contiguous(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> bool` | Check row-major C-order: iterate dims in reverse, expect stride = product of subsequent dims. |

### Core load/store (symmetric paths)

**`load(context, tile_ref: TileRef, coords: Optional[List[Tuple[int, ...]]], result_shape: Optional[Tuple[int, ...]]) -> Tile`** (lines 439–520)

Dispatch by source memory_space:
- HBM → DMA read into LX.
- LX → logical copy within LX (no physical movement).

**Fast path** (coords=None, contiguous strides): Single `mgr.read(n, dtype)` of full tile shape, reshape. Compute unique_sticks: `(end + STICK_BYTES - 1) // STICK_BYTES - base_ptr // STICK_BYTES` (HBM only). Write to LX via `_write_to_lx`.

**Slow path** (strided or coords-set): 
1. Call `_flat_memory_offsets(base_ptr, shape, strides, dtype, coords, stick_bytes)` → (offsets, unique_sticks). Linearizes N-d coords to flat element offsets; computes stick set if stick_bytes is not None.
2. Read span = max(offsets) + 1 elements.
3. NumPy fancy-index gather: `flat[offsets]`.
4. Reshape to result_shape (or tile_ref.shape if coords=None).
5. Write to LX, return Tile.

When *coords* is given, gathers only elements at those local coordinates within tile_ref.shape. Span read ensures all data is fetched in one contiguous range, then fancy-index selects. Zero-padding by `_read_flat` covers reads beyond allocation end.

**`store(context, tile: Tile, tile_ref: TileRef, coords: Optional[List[Tuple[int, ...]]]) -> int`** (lines 523–587)

Dispatch by destination memory_space. Symmetric to load.

**Fast path** (coords=None, contiguous): Write flattened tile.data directly. Compute and return unique_sticks; 0 for LX.

**Slow path** (strided or coords-set): Read-modify-write via `_flat_memory_offsets` (same as load), then NumPy fancy-index scatter: `flat[offsets] = tile.data.flatten()`, write back.

Returns `unique_sticks` (int): distinct HBM sticks touched, or 0 for LX. Dialect handler uses this to charge HBM traffic at stick granularity rather than logical tile nbytes (accounts for scatter writes).

**Invariants**:
- Source tile data is always read in C-order (via `ndarray.flatten()`); non-contiguous source arrays are handled internally.
- Coordinate collisions in scatter are last-writer-wins (NumPy assignment semantics).
- For HBM loads, the stick count formula: `(end_byte + STICK_BYTES - 1) // STICK_BYTES - base_byte // STICK_BYTES` counts distinct boundary-crossing sticks.

### Indirect access (gather/scatter)

**`indirect_load(context, iat: IndirectAccessTile, result_shape: Optional[Tuple[int, ...]]) -> Tile`** (lines 590–630)

Gather pattern. Enumerates variable space, resolves coords (direct dims from variable point, indirect dims from index memref lookups), delegates to `load`.

1. Validate `variables_space_order` is identity or permutation.
2. Call `_resolve_idx_reads(context, iat)` → (per_view_values: Dict[iv_idx → ndarray], idx_unique_sticks: int).
3. Call `_build_indirect_coords(iat, idx_values)` → coords: List[Tuple[int, ...]].
4. Call `load(context, iat.parent_ref.to_tile_ref(), coords=coords, result_shape=...)`.
5. Stamp result.index_unique_sticks = idx_unique_sticks.

**`indirect_store(context, tile: Tile, iat: IndirectAccessTile) -> int`** (lines 915–966)

Scatter pattern. Mirror of indirect_load.

1. Validate tile.shape == iat.shape and variables_space_order.
2. Call `_resolve_idx_reads` (same as load).
3. Call `_build_indirect_coords`.
4. Call `store(context, tile, iat.parent_ref.to_tile_ref(), coords=coords)` → data_sticks.
5. Return data_sticks + idx_unique_sticks (aggregate HBM traffic).

**Helper: `_resolve_idx_reads(context, iat: IndirectAccessTile) -> (Dict[int, np.ndarray], int)`** (lines 189–261)

For each indirect dimension's index view:
1. Enumerate variable-space points in `variables_space_order` order (via `_enumerate_in_vso_order`).
2. For each point, compute byte addresses for that view's idx values via subscript expressions.
3. Per-view: one `_MemAccessor.read_scattered` call (dedupes and merges adjacent runs).
4. Return per_view_values dict and total_idx_unique_sticks (sum of HBM views; 0 for all-LX).

**Hoisting** (lines 221–231): Per-view loop-invariants (bpe, strides, byte_address) are precomputed before pt enumeration.

**Helper: `_build_indirect_coords(iat, idx_values: Dict) -> List[Tuple[int, ...]]`** (lines 264–309)

For each enumerated point, construct coordinate tuple:
- Direct dims: take directly from variable point.
- Direct_expr dims: eval subscript expression over point.
- Indirect dims: consume next value from idx_values[iv_idx] iterator (pre-resolved in pt-major, dim-minor order).

Raises IndexError if any idx value is negative (rejects NumPy's silent wrap-around).

**Helper: `_enumerate_in_vso_order(iat) -> List[Tuple[int, ...]]`** (lines 170–186)

Enumerate variable-space points. If `variables_space_order` is non-identity permutation, sort points by `vso.eval(pt)` (RFC 0682 §473). Both `_resolve_idx_reads` and `_build_indirect_coords` route through this so iteration stays in lockstep (guard symmetry).

---

### Distributed memory (RFC 0682 §3.3)

Coordinates:
- x = global_base (access tile's global origin).
- A = access_tile_set (local 0..access_shape, or None for full box).
- x+A = global footprint of access tile.
- B_i = partition i's coordinate_set (global coords).
- C_i = (x+A) ∩ B_i (global coords covered by both; per-survivor coordinate_set).
- p_i = min(B_i) = partition i's origin (global coords).

**`distributed_tile_access(dist_ref: DistributedMemRef, access_shape, base_map, indices, access_tile_set: Optional[BoxSet|AffineSet]) -> DistributedTileRef`** (lines 652–755)

Resolve partition routing once.

1. Compute global_base = base_map.eval(indices).
2. Pre-compute (x+A) as BoxSet when possible (None ⇒ implicit full box [x, x+access_shape)).
3. For each partition B_i:
   - **Fast path** (both BoxSet): compute C_i = B_i.intersect(xA_box) in O(ndim).
   - **Slow path** (AffineSet or either side): enumerate B_i, filter by membership in x+A.
   - Skip empty intersections.
4. Return DistributedTileRef with survivors (each a TileRef with coordinate_set=C_i and partition_origin=p_i).

**Raise ValueError** if no partition covers the access region.

**`_subtile_ref(survivor: TileRef, box: BoxSet) -> TileRef`** (lines 758–781)

Build a TileRef covering exactly *box* (global coords) within *survivor*. Inherit strides; shape shrinks to box extent, base_ptr shifts to box.lo local origin: `(box.lo - p_i) * stride * bpe`. Plugs into load/store; strided iteration lands each element at correct byte offset (both row-major and column-packed work uniformly).

**`distributed_load(context, dist_tile_ref: DistributedTileRef, result_shape: Optional[Tuple[int, ...]]) -> Tile`** (lines 784–850)

Gather across surviving partitions into single LX-resident Tile.

1. Pre-allocate out buffer.
2. For each survivor:
   - **Fast path** (BoxSet C_i): build sub-TileRef via `_subtile_ref`, delegate to `load`, write result to rectangular slice out[C_i - x].
   - **Slow path** (List[Tuple] C_i): per-coord scatter—translate C_i to partition-local coords, batch-read via `_MemAccessor.read_scattered`, scatter each element to access-local position out[C_i - x].
3. Aggregate unique_sticks from all survivors.
4. Write out to LX, return Tile.

**`distributed_store(context, tile: Tile, dist_tile_ref: DistributedTileRef) -> int`** (lines 853–912)

Scatter Tile to surviving partitions. Mirror of distributed_load.

1. For each survivor:
   - **Fast path** (BoxSet C_i): slice source rectangularly at C_i - x, wrap in Tile, write via sub-TileRef (np.ascontiguousarray covers non-contiguous slices).
   - **Slow path** (List[Tuple] C_i): per-coord gather/write via read-modify-write.
2. Aggregate unique_sticks from all survivors.
3. Return total (HBM stick cost for coordination).

---

## Key load-bearing semantics

### Byte-address arithmetic (constant throughout)

- **HBM**: stick-aligned, 128-byte boundaries. `addr = stick * STICK_BYTES + intra_byte`. `divmod(byte_addr, STICK_BYTES)` extracts (stick, intra_byte).
- **LX**: plain byte addresses in local address space. No stick concept.

### `base_ptr` is an element index (RFC #110)

A MemRef's `base_ptr` is the **number of elements** from the start of its
address space (matching what MLIR pointer operands carry), **not** a byte or
stick offset. The byte address is derived by multiplying by the element width:

- Python: `MemRef.byte_address = base_ptr * bytes_per_elem(dtype)` (`ktir_cpu/ir_types.py:87`; docstring at `:49–52`).
- Rust: `MemRef::byte_address()` (`ktir-core/src/memref.rs:81–82`) — `self.base_ptr * self.dtype.bytes_per_elem()`.

**Pointer-binding sites (resident path).** When the resident executor binds an
HBM-stick allocation to an MLIR pointer operand, it converts the **stick → an
element index** so `base_ptr*bpe` lands back on the stick byte
`stick*STICK_BYTES`: `elem = stick * STICK_BYTES / bytes_per_elem(dtype)`. Sites:
`interpreter.rs:726` (`marshal_inputs`), `resident.rs:744` / `:799` / `:1084`
(fused segment, native attention, node-tile dataflow). Symmetrically, the Metal
GEMM weight readers treat the resident `base_ptr` as an element index and
recover the byte address as `elem*bpe`: `metal.rs:911`
(`resolve_gemm_b_operand`) and `metal.rs:1077` (`resolve_gemm_bt_operand`,
`[n,k]` block).

### Span-read strategy (lines 512, 584, 837–838, 905)

To avoid per-element dict scans, all load/store paths read a single contiguous span from base to max(offsets), then use NumPy fancy-indexing (gather or scatter). Formula: `span = max(offsets) + 1` (in elements). If offsets is empty (zero-extent enumeration), span defaults to 1 (guard against divide-by-zero).

### Unique-stick counting

**HBM fast path** (lines 496–502): `unique_sticks = (end + STICK_BYTES - 1) // STICK_BYTES - base // STICK_BYTES`. Counts boundary-crossing sticks for a single contiguous range.

**HBM slow path** (line 435): `sticks.add((base_ptr + o * bpe) // stick_bytes)` per offset. Set dedup counts distinct sticks.

**LX**: stick_bytes=None, returns None (not counted).

### Indirect dimension iteration guard (symmetry across load/store)

Both `_resolve_idx_reads` and `_build_indirect_coords` enumerate via `_enumerate_in_vso_order`. If `variables_space_order` exists and is non-identity, iteration is sorted by `vso.eval(pt)` rather than natural order. This keeps idx reads and coord construction in lockstep (RFC 0682 §473). Non-permutation vso raises ValueError before enumeration.

### Coordinate-set semantics (load/store with coords)

When *coords* is supplied, it is a list of local coordinate tuples **within tile_ref.shape**. Each tuple indexes into the N-d shape using standard multi-dimensional indexing. Linearization via `sum(c * s for c, s in zip(coord, strides))` produces flat element offsets into the loaded/stored span. NumPy fancy-indexing (`flat[offsets]`) both gathers (load) and scatters (store).

### Distributed coordinate translation

- **Partition-local coords**: `c_local = c_global - p_i` (where p_i = min(B_i)).
- **Access-local coords**: `c_access = c_global - x` (where x = global_base of access).

For slow-path scatters, both are precomputed once per point and zipped with offsets (lines 826–831, 893–898, 906–907).

### Data layout safety

`tile.data` is always read via `ndarray.flatten()` (C-order, contiguous copy). Non-contiguous source arrays are handled; callers do not pre-contiguity-check. On load, reshape (`reshape(result_shape)`) is applied post-fancy-index. On distributed_store fast path, `np.ascontiguousarray(tile.data[slc])` covers non-contiguous slices (line 886).

### Known limitation: cross-allocation merging (lines 125–131)

`read_scattered` merges adjacent byte addresses into contiguous runs. If two addresses are physically adjacent but from different allocations (host or bug), the run merges silently, and `_read_flat` reads only from the run's start allocation (zero-filling the second). No error is raised. Hard-guarding requires simulator to expose allocation extents (tracked as follow-up).

---

## Type signatures (from ir_types.py perspective)

- **Tile**: data: np.ndarray, dtype: str, shape: Tuple[int, ...], unique_sticks: Optional[int], index_unique_sticks: int (only set by indirect_load).
- **MemRef**: base_ptr: int (**element index** — see RFC #110 note above; `byte_address = base_ptr * bytes_per_elem(dtype)`), shape: Tuple[int, ...], strides: List[int], memory_space: str, dtype: str, coordinate_set: Optional[str], lx_core_id: Optional[int].
- **TileRef**: base_ptr: int, shape: Tuple[int, ...], strides: List[int], memref: MemRef, dtype: str, coordinate_set: Optional[CoordinateSet], partition_origin: Optional[Tuple[int, ...]].
- **IndirectAccessTile**: shape, dtype, parent_ref: MemRef, index_views: List, dim_subscripts: List[Dict], variables_space_set, variables_space_order: Optional[AffineMap].
- **DistributedMemRef**: shape, dtype, partitions: List[MemRef].
- **DistributedTileRef**: partitions: List[TileRef], shape, dtype, global_base: Tuple[int, ...].

---

# latency

## LatencyCategory (StrEnum)

All members (string enum variants):
- `ZERO` = "zero"
- `MEMORY` = "memory"
- `COMPUTE_FLOAT` = "compute_float"
- `COMPUTE_TRANSCENDENTAL` = "compute_transcendental"
- `COMPUTE_INT` = "compute_int"
- `COMPUTE_MATMUL` = "compute_matmul"
- `COMM` = "comm"

## HardwareConfig (dataclass)

**Fields** (all public, mutable):
- `num_cores: int` = 32 — number of processing cores in grid
- `clock_ghz: float` = 1.0 — clock frequency; 1 cycle = 1 ns at 1.0 GHz
- `hbm_bandwidth_tb_s: float` = 1.0 — aggregate HBM bandwidth in TB/s (estimated)
- `ring_bandwidth_tb_s: float` = 4.0 — ring network bandwidth per direction in TB/s
- `simd_elements_per_cycle: int` = 64 — SIMD throughput in f16 elements/cycle (estimated)
- `systolic_flops_per_cycle: int` = `2 * 64 * 64 * 64` = 524288 — peak systolic array throughput in FLOPs/cycle (64×64 PE grid, 64 K-steps pipelined; default: each PE does 2 FLOPs/cycle = 1 fused multiply-add)
- `transcendental_penalty: int` = 4 — multiplier for latency cost of transcendental ops vs elementwise (estimated)

**Computed properties** (immutable):
- `hbm_bytes_per_cycle_per_core: float` — formula: `(hbm_bandwidth_tb_s * 1e12) / (clock_ghz * 1e9) / num_cores`
- `ring_bytes_per_cycle: float` — formula: `ring_bandwidth_tb_s * 1e12 / (clock_ghz * 1e9)`

**Invariants**: All parameters assumed positive; division by zero guarded in client code.

**Rust redesign notes**: Immutable after construction; make properties const fns or cache computed values at instantiation time.

---

## CoreLatencyCounters (dataclass)

**Fields** (all mutable):
- `compute_cycles: float` = 0.0 — accumulated compute cycles
- `memory_cycles: float` = 0.0 — accumulated memory (HBM/LX) cycles
- `comm_cycles: float` = 0.0 — accumulated ring communication cycles
- `total_flops: float` = 0.0 — sum of FLOPs executed on this core
- `total_bytes: int` = 0 — sum of bytes transferred on this core
- `trace: Optional[List[_TraceEntry]]` = None — optional operation trace (only populated if tracing enabled)

**Methods**:
- `total_cycles: float` [property] — `compute_cycles + memory_cycles + comm_cycles`
- `record(category: str, cycles: float, op_type: str = "", flops: float = 0.0, nbytes: int = 0)` — accumulates cycles into one of compute/memory/comm bucket, adds to total_flops/total_bytes, optionally appends to trace

**Invariants**: Cycles and counters monotonically increase; trace is either `None` (no tracing) or owned list (can append).

**Rust notes**: Use enum for category instead of string; consider separate structs for traced vs untraced variants to avoid Option overhead.

---

## _TraceEntry (dataclass, internal)

**Fields**:
- `op_type: str` — MLIR operation type string
- `cycles: float` — cycle cost for this operation
- `category: str` — one of "compute", "memory", "comm", "zero"

Simple record; no methods. Only populated when `LatencyTracker._trace == True`.

---

## LatencyTracker (class)

**Constructor**:
- `__init__(config: HardwareConfig, trace: bool = False)` — stores config and trace flag, initializes empty counters dict

**Public methods**:

- `reset()` — clears all accumulated counters (dict.clear())
- `record_op(core_id: int, op_type: str, result: Any, operands: List[Any])` — main entry point
  - Lazily creates `CoreLatencyCounters` for `core_id` if absent
  - Calls `_estimate()` to compute category/cycles/flops/nbytes
  - Records into core's counters via `record()`
  - **Sideband channel semantics**: Handlers return values in `result` that encode per-op metadata:
    - **Store ops** return `int` (unique_sticks count) as `result` instead of Tile; `_data_size()` converts to bytes via `int * HBMSimulator.STICK_BYTES` (file:line 306–307)
    - **Load ops** return `Tile` with `unique_sticks` and optional `index_unique_sticks` fields set (file:line 311–319)
    - **Indirect loads/stores** may populate both; logic validates presence (file:line 322–334)
- `report() -> LatencyReport` — constructs LatencyReport from current counters dict

**Private helper methods**:

- `_estimate(op_type: str, result: Any, operands: List[Any]) -> Tuple[str, float, float, int]` 
  - Returns `(category, cycles, flops, nbytes)` for one op
  - Routes on `get_latency_category(op_type)` (calls external registry; file:line 37)
  - **ZERO** (line 196–197): category == "zero" → `("zero", 0.0, 0.0, 0)`
  - **MEMORY** (line 199–209):
    - If memory space is "LX" (on-chip scratchpad), free (no DMA): `("memory", 0.0, 0.0, 0)`
    - If "HBM": `nbytes = _data_size(result, operands)` → `cycles = nbytes / hbm_bytes_per_cycle_per_core` → `("memory", cycles, 0.0, nbytes)`
    - Handles division by zero
  - **COMPUTE_MATMUL** (line 211–218):
    - Extracts `(M, N, K) = _matmul_dims(operands)`
    - `flops = 2.0 * M * N * K`
    - `cycles = flops / systolic_flops_per_cycle`
    - Returns `("compute", cycles, flops, 0)` — no HBM traffic assumed
  - **COMPUTE_TRANSCENDENTAL** (line 220–227):
    - `n_elems = _num_elements(result, operands)`
    - `cycles = (n_elems / simd_elements_per_cycle) * transcendental_penalty`
    - penalty models higher latency, **not** increased FLOP count
    - Returns `("compute", cycles, float(n_elems), 0)`
  - **COMPUTE_FLOAT** (line 229–234):
    - `n_elems = _num_elements(result, operands)`
    - `cycles = n_elems / simd_elements_per_cycle`
    - Returns `("compute", cycles, float(n_elems), 0)`
  - **COMPUTE_INT** (line 236–244):
    - `n_elems = _num_elements(result, operands)`
    - If `n_elems <= 1` (scalar index arithmetic resolved at compile time), free: `("compute", 0.0, 0.0, 0)`
    - Else: `cycles = n_elems / simd_elements_per_cycle` → `("compute", cycles, float(n_elems), 0)`
  - **COMM** (line 246–256):
    - `nbytes = _comm_size(operands)`
    - `cycles = nbytes / ring_bytes_per_cycle`
    - Special case: `op_type == "ktdp.reduce"` → multiply cycles by `ceil(log2(num_cores))` (allreduce is O(log(cores)) rounds)
    - Returns `("comm", cycles, 0.0, nbytes)`
  - Raises `NotImplementedError` on unknown category

- `_memory_space(operands: List[Any]) -> str` [static]
  - Inspects operands for `MemRef`, `TileRef`, `AccessTile`, `IndirectAccessTile`
  - Returns memory_space string ("HBM" or "LX") from nested TileRef or parent_ref
  - For `IndirectAccessTile`: returns "LX" only if **both** parent_ref and all index_views have memory_space == "LX"; else "HBM"
  - Defaults to "HBM" if no TileRef found (e.g., tt.load pointer-based access)

- `_data_size(result: Any, operands: List[Any]) -> int` [static]
  - **Load handlers** embed `unique_sticks` (and optional `index_unique_sticks`) on result Tile
  - **Store handlers** return int sideband from operation result (file:line 300–307)
  - Computation (file:line 309–347):
    - If `isinstance(result, int)`: store sideband → return `result * HBMSimulator.STICK_BYTES`
    - If `isinstance(result, Tile)`:
      - Assert `unique_sticks` is not None (runtime error file:line 312–316)
      - Accumulate `result.unique_sticks * HBMSimulator.STICK_BYTES`
      - If `index_unique_sticks` present, add `index_unique_sticks * HBMSimulator.STICK_BYTES`
    - Iterate operands: 
      - Skip `IndirectAccessTile` (already aggregated in result via index_unique_sticks)
      - If bare `Tile` in operands, error: store handlers must provide int sideband
  - Returns total bytes
  - **Invariant**: result is either int (store) or Tile (load), never both. File:line 337–341 guards this.

- `_num_elements(result: Any, operands: List[Any]) -> int` [static]
  - If result is Tile: return `prod(result.shape)` (numpy product)
  - Else, search operands for any Tile and return prod of first Tile's shape
  - Fallback: return 1 (scalar)

- `_matmul_dims(operands: List[Any]) -> Tuple[int, int, int]` [static]
  - Extracts first two Tile operands: assume shape `(M, K)` and `(K, N)`
  - Returns `(M, N, K)` with defaults `(1, 1, 1)` if fewer than 2 tiles

- `_comm_size(operands: List[Any]) -> int` [static]
  - Finds first Tile operand, returns `tile.data.nbytes` (numpy nbytes attribute)
  - Returns 0 if no Tile found

**Rust redesign notes**:
- Replace `Any` with a sealed enum variant type (e.g., `OpValue`) that holds MemRef, TileRef, AccessTile, IndirectAccessTile, Tile, int, or scalar
- Sideband channel: encode in return type (e.g., `Result<OpValue, EstimationError>` or separate struct with op result + metadata)
- Static methods → module-level functions or impl block
- Dict[int, CoreLatencyCounters] → HashMap or BTreeMap
- String enums for category → use native Rust enum; call registry only once to resolve category before branching

---

## LatencyReport (dataclass)

**Fields** (immutable after construction):
- `config: HardwareConfig` — reference to hardware configuration
- `counters: Dict[int, CoreLatencyCounters]` — per-core cycle counters

**Properties** (computed, no arguments):

- `kernel_cycles: float` — max total_cycles across all cores; 0 if counters empty
- `kernel_time_us: float` — kernel_cycles / (clock_ghz * 1e3) [cycles to microseconds]
- `bottleneck: str` — on critical-path core (max total_cycles), return name of largest category: "compute", "memory", or "comm"; "none" if empty

**Methods**:

- `per_core_summary() -> List[Dict[str, Any]]` — returns list of dicts with keys:
  - `"core_id"`, `"compute_cycles"`, `"memory_cycles"`, `"comm_cycles"`, `"total_cycles"`
  - Sorted by core_id

- `roofline() -> Dict[str, float]` — computes roofline analysis for critical-path core; returns dict:
  - `"arithmetic_intensity"`: `total_flops / total_bytes` or `inf` if `total_bytes == 0`
  - `"achieved_gflops"`: `total_flops / elapsed_s / 1e9` (where `elapsed_s = total_cycles / clock_hz`)
  - `"peak_gflops"`: `simd_elements_per_cycle * clock_hz / 1e9`
  - `"peak_bw_gb_s"`: `hbm_bytes_per_cycle_per_core * clock_hz / 1e9`
  - `"ridge_point"`: `peak_gflops / peak_bw_gb_s` (FLOP/B threshold where compute ceiling = BW ceiling)
  - `"ceiling_gflops"`: `min(peak_gflops, peak_bw * arithmetic_intensity) / 1e9` (roofline ceiling at kernel's AI)
  - `"efficiency"`: `achieved_gflops / ceiling_gflops` ∈ [0, 1]
  - Empty dict if counters empty
  - **Key formula** (file:line 499): `ridge_point = peak_flops / peak_bw` (AI where memory and compute ceilings meet)
  - **Invariant** (file:line 464–470): roofline covers only compute + HBM; ring communication cycles excluded from model. If bottleneck is "comm", roofline may still classify kernel as compute- or memory-bound on compute-vs-HBM axis alone.

- `summary_dict() -> Dict[str, Any]` — returns flat dict:
  - `"kernel_cycles"`, `"kernel_time_us"`, `"bottleneck"`, `"num_cores"`, `"per_core"` (list from per_core_summary)

- `__str__() -> str` — human-readable report with tables; includes roofline section if `critical.total_flops > 0 or critical.total_bytes > 0`

**Rust redesign notes**: All properties are derived; make them methods (consume or borrow self). Return type for roofline should be struct (not dict string keys) for type safety. The roofline computation is CPU-bound; no special parallelism needed.

---

## Cross-Module Dependencies

- **Imports from `ir_types`**: `AccessTile`, `IndirectAccessTile`, `MemRef`, `Tile`, `TileRef` — all used in type-checking within _estimate and _data_size
  - `Tile.shape`, `Tile.unique_sticks`, `Tile.index_unique_sticks`, `Tile.data` (numpy array)
  - `TileRef.memref`
  - `MemRef.memory_space`
  - `AccessTile.parent_ref`
  - `IndirectAccessTile.parent_ref`, `IndirectAccessTile.index_views`

- **Imports from `dtypes`**: `bytes_per_elem` — declared but not used in latency.py (dead import; can remove)

- **Imports from `memory`**: `HBMSimulator.STICK_BYTES` — constant multiplier for stick-granular accounting

- **Imports from `dialects.registry`**: `get_latency_category(op_type: str) -> LatencyCategory` — external dispatch to assign category to op_type string

---

## Key Invariants & Python-isms

1. **Sideband channel for HBM accounting** (file:line 289–303):
   - Store ops: handler returns int (unique_sticks), client propagates as op result
   - Load ops: handler stamps result Tile with unique_sticks/index_unique_sticks
   - No Duck typing in Rust: use sealed enum for result/operand types; validate at construction

2. **Dynamic dispatch via category string** (file:line 192):
   - `get_latency_category(op_type)` called at runtime to resolve handler's op_type to LatencyCategory
   - Rust: call registry once, match on native enum

3. **Optional tracing** (file:line 157–178):
   - Trace enabled if `trace=True`; CoreLatencyCounters.trace = [] or None
   - Rust: use Option<Vec<TraceEntry>>; no performance overhead when disabled

4. **Numpy arrays in Tile.data** (file:line 353, 378):
   - `_num_elements()` uses `np.prod(result.shape)` 
   - `_comm_size()` reads `tile.data.nbytes`
   - Rust: assume Tile wraps shape tuple + element count; compute nbytes = prod(shape) * element_size_bytes

5. **Memory space lattice** (file:line 280–282):
   - `IndirectAccessTile`: "LX" iff **all** of (parent_ref.memory_space and all index_views.memory_space) are "LX"
   - Else defaults to "HBM"
   - Rust: encode logic carefully; use conjunction

6. **Critical path max reduction** (file:line 397, 409):
   - Kernel cycles = max(all cores' total_cycles)
   - Bottleneck = category with max cycles on critical-path core
   - Rust: compute once, cache if needed

7. **Roofline ridge point** (file:line 499):
   - `ridge_point = peak_flops / peak_bw` (units: FLOP/B)
   - Kernel AI < ridge → memory-bound; AI > ridge → compute-bound
   - Divide-by-zero: peak_bw is derived from hbm_bytes_per_cycle_per_core (clock × BW_TB_s) and is only zero if clock_ghz or hbm_bandwidth_tb_s is zero (degenerate config)

8. **Reduce latency scaling** (file:line 253–255):
   - `op_type == "ktdp.reduce"` → `cycles *= ceil(log2(num_cores))`
   - Hardcoded op_type string check; no registry dispatch
   - Rust: if deploying to different comm patterns, generalize via metadata

---

## Constants

- `HBMSimulator.STICK_BYTES` — stick granularity for HBM accounting (imported from memory module; not defined here)
- `systolic_flops_per_cycle` default: `2 * 64 * 64 * 64` = 524288 (64×64 PE grid, 64 K-steps pipelined)
- `transcendental_penalty` default: 4 (unitless multiplier)

---

Perfect. Now I have enough information to write the spec. Let me create the markdown spec:

# registry+env

## Handler Registry & Dispatch

**Registry:** `_REGISTRY: Dict[str, HandlerFn]` (module-level, mutable)

**Handler Function Signature:**
```
HandlerFn = Callable[[Operation, CoreContext, ExecutionEnv], Any]
```

Handler contract:
- **Parameters:**
  - `op: Operation` — the IR operation to execute (fields: `op_type`, `operands`, `attributes`, `result`, `regions`)
  - `context: CoreContext` — per-core execution state with SSA value map, LX scratchpad, grid position
  - `env: ExecutionEnv` — core-external resources (grid executor, region execution callback)
- **Returns:** Any value to be stored as `op.result` in the context; generators allowed for comm ops (yield `RecvRequest`)
- **Semantics:** Execute the operation within the core's context, reading operands via `context.get_value()`, writing results via `context.set_value()`, managing LX memory via `context.track_lx()`, and optionally dispatching sub-regions via `env.execute_region()`

**Registration decorator:** `@register(*op_names, latency_category="zero")`
- Maps op name(s) to handler and latency category
- Infers op name from function name if no `op_names` provided (e.g. `arith__addf` → `"arith.addf"`)

**Dispatch function:** `dispatch(op_name: str) -> Optional[HandlerFn]`
- Returns handler for op_name or `None` if not registered

---

## Latency Category Registry

**Registry:** `_LATENCY_CATEGORIES: Dict[str, str]` (module-level, immutable after init)

**Query function:** `get_latency_category(op_name: str) -> str`
- Returns registered latency category (e.g. `"zero"`, `LC.COMPUTE_FLOAT`) for op_name
- Defaults to `"zero"` if not found
- Values are `LatencyCategory` enum members (StrEnum)

---

## Parser Registry & Context

**Parser Function Signature:**
```
ParserFn = Callable[[str, ParseContext], Optional[Operation]]
```

Parser contract:
- **Parameters:**
  - `op_text: str` — raw operation text (single operation, may span multiple lines)
  - `parse_ctx: ParseContext` — parse-time context with alias table
- **Returns:** `Operation` object or `None` to fall through to default parser
- **Semantics:** Parse op_text into an IR `Operation` by resolving aliases and constructing attributes; decoupled from execution concerns

**Registration decorator:** `@register_parser(*op_patterns)`
- Maps pattern(s) to parser via substring match (`if pattern in op_text`)
- Patterns matched in iteration order; first match wins

**Dispatch function:** `dispatch_parser(op_text: str) -> Optional[ParserFn]`
- Returns parser whose pattern appears in op_text, or `None`

**ParseContext dataclass:**
- `aliases: Dict[str, str]` — module-level named attribute aliases (maps `"#name"` → verbatim value string, e.g. `"#X_coord_set"` → `"affine_set<(d0, d1) : (d0 >= 0, ...)>"`)
- Populated by module-level pre-scan (parser.py line 103-106)
- Passed to dialect parsers to resolve `#name` references in op attributes without re-parsing module scope

**Construction helper:** `make_parse_context(aliases: Dict[str, str]) -> ParseContext`

---

## Execution Environment

**ExecutionEnv dataclass:**
- `grid_executor: GridExecutor` — manages all cores; enables cross-core queries (get_core, get_cores_in_group, coordinate transforms)
- `execute_region: Callable[[CoreContext, List[Operation]], Any]` — synchronous region executor for nested control flow (scf.for body, scf.if branch, etc.); invoked by handlers; does not return generators

**Ownership & Mutation:**
- `grid_executor` is **shared immutably** across all cores; handlers call read-only methods (get_core_at_pos, get_cores_in_group) or reference it for cross-core comm setup
- `execute_region` is a **bound callback** to `KTIRInterpreter.execute_region()` (interpreter.py:265-274); mutates nothing directly; calls `_execute_op` recursively per operation in the region

---

## Interpreter Execution Flow

**Class:** `KTIRInterpreter`
- Fields:
  - `module: Optional[IRModule]` — parsed IR (set by `load()`)
  - `memory: Optional[SpyreMemoryHierarchy]` — shared HBM + per-core LX scratchpads (created in `_prepare_execution()`)
  - `grid_executor: Optional[GridExecutor]` — multi-core scheduler (created in `_prepare_execution()`)
  - `ring_backend: Optional[TransferBackend]` — remote LX access for comm ops (set to `InstantTransferBackend(memory)` in `_prepare_execution()`)
  - `_env: Optional[ExecutionEnv]` — passed to handlers (created in `_prepare_execution()`)
  - `_latency_tracker: Optional[LatencyTracker]` — if latency_config provided to __init__

### `load(ktir_source: str)`
Parses MLIR text (inline or file path) via `KTIRParser.parse_module()` or `parse_file()`.
- Heuristic: if `ktir_source` contains `\n` or starts with `"module"`, treat as inline MLIR; else treat as file path
- Sets `self.module`

### `execute_function(func_name: str, **kwargs) -> Dict[str, np.ndarray]`
Executes a function with tensor + scalar arguments; coordinates grid setup, per-core execution, result collection.

**Steps:**
1. Retrieve function from module (raises if module not loaded)
2. Call `_prepare_execution(func.grid)` to:
   - Allocate `SpyreMemoryHierarchy(num_cores)` (HBM + per-core LX)
   - Set `ring_backend = InstantTransferBackend(memory)`
   - Create `GridExecutor(grid_shape, memory)` with per-core `CoreContext` instances
   - Build `ExecutionEnv(grid_executor, execute_region)` and store in `self._env`
   - Reset latency tracker if enabled
3. Normalize kwargs names if needed (positional remap when declared names don't match kwargs keys but counts match)
4. **Allocate inputs in HBM:** for each tensor argument, call `memory.hbm.allocate(tensor.nbytes)`, `memory.hbm.write(stick, tensor)`, store stick in `input_ptrs[arg_name]`; scalar arguments stored directly as values (not pointers)
5. **Execute:** call `grid_executor.execute_with_communication(func.operations, input_ptrs, self._execute_op, transfer_backend=ring_backend)` (orchestrates multi-core scheduling with generator support)
6. **Collect outputs:** for each tensor argument, read from HBM using `memory.hbm.read(stick, n_elements, dtype)` and reshape to original tensor shape; return dict of output arrays

**Latency tracking (if enabled):** _execute_op resolves operand values before dispatch for recording; latency_tracker.record_op() called after handler returns

### `_execute_op(op: Operation, context: CoreContext) -> Any`
Single-operation executor (called per-core by CoreExecutionStack).

**Steps:**
1. (Optionally) resolve operand values from context for latency tracking
2. Look up handler via `dispatch(op.op_type)`; raise if not found
3. **Invoke handler:** `result = handler(op, context, self._env)`
   - Handler may return a generator (comm ops); generator is consumed by CoreExecutionStack (not stored)
   - Or returns a concrete value (compute ops)
4. **Store result:** if `op.result` is set and result is not None:
   - If multi-result (op.result is list, result is tuple): zip and store each pair
   - Else: single-result; store result in context; if result is `Tile`, call `context.track_lx(op.result, result.size_bytes())` to bump LX usage counter
5. Record latency if tracker enabled
6. Return result (or None if generator — CoreExecutionStack will re-store after generator completes)

### `execute_region(context: CoreContext, operations: List[Operation]) -> Any`
Synchronous nested region executor (called by handlers for scf.for body, scf.if branch, etc.).

**Contract:** loops through operations, calling `_execute_op(op, context)` per operation; returns final result; no generator machinery (comm ops forbidden in nested regions per spec).

**Invariants:**
- Runs synchronously (no blocking recv)
- Scope lifetime: handlers call `context.push_scope()` before entering region, `context.pop_scope()` after exiting (to manage LX lifetime)

---

## CoreContext Scope & LX Management

**CoreContext fields (subset relevant to handlers):**
- `core_id: int` — linear core ID
- `grid_pos: Tuple[int, int, int]` — (x, y, z) grid position
- `lx: LXScratchpad` — core-local scratchpad (2 MB capacity)
- `hbm: HBMSimulator` — shared HBM reference
- `_scope_stack: List[Dict[str, Any]]` — nested scope stack (function level at bottom)
- `_lx_bytes: Dict[str, int]` — SSA name → bytes allocated (single source of truth for lx.used)

**Handler-facing methods:**
- `get_value(name: str) -> Any` — lookup SSA value, searching scopes top-to-bottom (inner sees outer)
- `set_value(name: str, value: Any)` — store SSA value in topmost scope
- `track_lx(name: str, size_bytes: int)` — increment lx.used; raises MemoryError if overflow
- `push_scope()` — enter region scope, snapshot lx.next_ptr for watermark-based deallocation
- `pop_scope()` — exit region scope, rewind lx.next_ptr and untrack all values in scope
- `send_to(dst_core: int, tile: Tile)` — enqueue tile to destination core (wired by scheduler; raises if no scheduler attached)
- `get_lx(core_id: Optional[int]) -> LXScratchpad` — return local scratchpad if core_id is None/self; else delegate to transfer_fn (fails if no scheduler)
- `get_grid_id(dim: int) -> int` — return grid coordinate for dimension (0=x, 1=y, 2=z)
- `attach_scheduler(send_fn, transfer_fn)` — wire comm functions for duration of run (called by GridExecutor before first core step)
- `detach_scheduler()` — clear comm bindings after run completes

**Scope lifetime semantics (interpreter.py:50-84, grid.py:156-205):**
- SSA values are immutable bindings scoped to their region
- Tiles (only) occupy LX; other results (TileRef, int, AccessTile) are bookkeeping (zero LX cost)
- Function body = base scope; each scf.for/if body = nested scope
- Peak LX usage in a scope = sum of all Tiles produced in that scope (no intra-scope reuse; all tiles coexist)
- Bump-allocator with scope-level watermarks: `push_scope()` snapshots lx.next_ptr; `pop_scope()` rewinds it (earliest safe deallocation without liveness analysis)
- Invariant: `len(_lx_next_ptr_stack) == len(_scope_stack) - 1`

---

## GridExecutor & Communication

**GridExecutor.execute_with_communication()** (grid.py:448-543)
Drives all cores to completion via generator scheduler.

**Flow:**
1. Build per-core `CoreExecutionStack` (wraps a generator that runs ops until blocked on recv)
2. Attach scheduler to each core: `send_fn` enqueues tile into scheduler's message buffer; `transfer_fn(src_core)` returns remote LXScratchpad
3. Advance each core's generator via `resume()` until blocked or done
4. Loop: for each blocked core, try to deliver a message from a sender via `_pop(src, dst)`; advance receiver on delivery
5. Repeat until all cores done or detect deadlock
6. Deadlock detected if no progress on any iteration (blocked cores have no waiting messages)

**CoreExecutionStack (grid.py:291-356):**
- Wraps per-core generator that yields `RecvRequest` (src core ID)
- `resume(send_val=None)` steps generator with optional data; returns final value when done
- `is_blocked()` checks if awaiting recv; `waiting_on` holds src core ID

**Generator awareness:** Comm ops (dialect-specific) return a generator that yields `RecvRequest` objects. The scheduler intercepts these and parks the core. Compute ops return concrete values or None. CoreExecutionStack transparently drives generators via `gen.send(tile)` when resuming with a delivered message.

**Key constraint:** Multi-result ops (e.g. `linalg.reduce` that yields both SSA result and updates outs buffer) can alias SSA names (both refer to same Python object); detect via `id(a) == id(b)`.

---

## Operation & Region Structure

**Operation dataclass** (ir_types.py:322-336):
- `result: Optional[str]` — result SSA name (e.g. `"%x"`) or list of names for multi-result ops; `None` if no result
- `op_type: str` — dialect.op (e.g. `"arith.addf"`, `"ktdp.get_compute_tile_id"`)
- `operands: List[str]` — SSA value names of inputs
- `attributes: Dict[str, Any]` — operation attributes (parsed; may contain affine maps, shapes, dtypes)
- `result_type: Optional[str]` — result type string (e.g. `"tensor<32x1024xf16>"`)
- `regions: List[List[Operation]]` — nested operation lists (scf.for body, scf.if branches); each region is a list of ops

**Region semantics:**
- Control-flow ops (scf.for, scf.if) have regions in `op.regions`
- Handlers access regions via `env.execute_region(context, op.regions[i])` to run nested ops in a new scope
- Regions cannot contain comm ops (per spec; execute_region runs synchronously)

---

## Python-isms That Don't Map Cleanly to Rust

1. **Generator-based comm:** Comm ops return Python generators that yield `RecvRequest` objects. CoreExecutionStack drives these via `gen.send(tile)`. Rust equivalent: use async/await or explicit state machines + message passing; a trait `CommOp` returning `enum CommResult { Sent | Blocked(SrcCore) | Done }` with explicit resume semantics.

2. **Duck typing for results:** Handlers return `Any` (compute: concrete value; comm: generator). Rust needs explicit `enum OperationResult { Value(Box<dyn Any>), Comm(Box<dyn Generator>) }` or result type trait.

3. **Scope-scoped SSA map:** `CoreContext._scope_stack` is a list of dicts that shadow outer scopes. Rust: use a proper environment/symbol table with scope markers or a flat map with scope IDs.

4. **Mutable shared state via Python references:** Multiple SSA names can alias the same object (e.g. `linalg.reduce` result and outs buffer); detected via `id(a) == id(b)`. Rust: use `Rc<RefCell<>>` or `Arc<Mutex<>>` and compare pointer equality; or redesign to avoid aliasing.

5. **Bump allocator with rewinding:** LX uses a watermark-based bump allocator that rewinds on scope exit. Rust: straightforward with a `next_ptr` field; but requires careful lifetime tracking to avoid use-after-free.

6. **Dict-based dispatch:** Handlers stored in `_REGISTRY` dict, looked up by string op_name. Rust: use a match statement on enum op types or a static registry (HashMap or match-all at compile time).

7. **Latency tracker integration:** Optional per-operation latency recording requires resolving operands and calling tracker callbacks. Rust: make latency a trait impl on handlers or wrap handler calls in a recording function.

8. **Generator scheduler with deadlock detection:** The scheduler polls cores, tries message delivery, and detects deadlock if no progress. Rust: event-driven or explicit scheduling loop; requires careful state management (blocked cores, pending messages, cycle detection).

---

## Key Implementation Notes

**Handoff between parser and execution (interpreter.py:78-91):**
- Parser (via `KTIRParser`) produces `IRModule` with parsed `Operation` objects
- Module stores function list and alias table (module-scope `#name` attributes)
- Interpreter holds module reference and reconstructs ParseContext from aliases to pass to handlers if re-parsing occurs (currently not done; aliases passed to dialect parsers at parse time)

**Handler latency tracking (interpreter.py:191-230):**
- If `_latency_tracker` is enabled, resolve operand values **before** dispatch (cheap dict lookups in `context._scope_stack`)
- Call `_latency_tracker.record_op(core_id, op_type, result, resolved_operands)` **after** handler returns (result known)
- Tracked values passed to LatencyReport for per-op aggregation

**Grid executor lifecycle (interpreter.py:92-114):**
- Called once per `execute_function()` call
- Allocates fresh `SpyreMemoryHierarchy`, `GridExecutor`, `ExecutionEnv`
- All subsequent handler calls in that execution share the same `ExecutionEnv._env` reference
- Reset latency tracker if enabled

**Handler contract compliance (interpreter.py:179-232):**
- Handlers must read operands via `context.get_value(name)` (raises KeyError if not found)
- Must write results via `context.set_value(name, value)` and (for Tiles) call `context.track_lx(name, size_bytes)`
- May call `env.execute_region(context, ops)` to run nested regions synchronously
- Must not mutate `env.grid_executor` (read-only queries only) or `context` fields directly (use methods)

---

Perfect! Now I have all the information I need. Let me compile the comprehensive markdown spec:

# control+scf

## Type: `_YieldResult` (control_ops.py:32-36)
**Fields:**
- `values: List[Any]` — values wrapped by scf.yield / linalg.yield for loop-back feeding

**Semantics:** Sentinel wrapper to distinguish yielded values from normal function returns. Unwrapped by `unwrap_yield()` before being seen by dialect handlers.

---

## Type: `RegionExecutor` (control_ops.py:28)
**Signature:** `Callable[[CoreContext, List[Operation]], Any]`

**Semantics:** Region execution callback. In Python: `execute_region(context, operations) -> Any`. Returns last operation's result (or None). **CRITICAL:** In the current spec, `execute_region` is **synchronous and cannot yield `RecvRequest`**. Nested regions (scf.for body, scf.if branch, linalg.generic combiner) do NOT contain comm ops; only top-level function bodies step the generator. This is a hard constraint — execute_region has no generator machinery (interpreter.py:195-201).

---

## Class: `ControlOps` (control_ops.py:39-219)

### `ControlOps.if_op(context, condition, then_region, else_region, region_executor) -> Any`
**Operands:**
- `context: CoreContext` — execution state
- `condition: bool` — branch selector
- `then_region: List[Operation]` — ops if condition truthy
- `else_region: List[Operation]` — ops if condition falsy
- `region_executor: RegionExecutor` — callback to step the region

**Returns:** Result from executed region (then or else), unwrapped if `_YieldResult`

**Semantics (control_ops.py:42-74):**
1. Select branch based on condition
2. Return None if branch is empty
3. **Scope isolation:** Push a new scope before executing, pop after. This isolates body-local SSA values.
4. Pop frees all LX tracked in that scope (via CoreContext.pop_scope()).
5. If branch yields Tiles, pop_scope untracts their LX; caller re-tracks via track_lx when binding result.
6. Unwrap `_YieldResult` sentinel before returning to caller (via `unwrap_yield` from _helpers.py:86-100).

**Invariant:** Branch body always executes in isolation; SSA values do not leak out except the result.

---

### `ControlOps.for_op(context, lower_bound, upper_bound, step, iter_var_name, body_region, region_executor, iter_arg_names=None, iter_init_values=None) -> Any`
**Operands:**
- `context: CoreContext` — execution state
- `lower_bound: int` — loop start (inclusive)
- `upper_bound: int` — loop end (exclusive)
- `step: int` — loop increment; clamped to `max(int(step), 1)` (control_ops.py:121)
- `iter_var_name: str` — SSA name for induction var (e.g., `"%i"`)
- `body_region: List[Operation]` — loop body ops
- `region_executor: RegionExecutor` — region callback
- `iter_arg_names: List[str]` — optional list of iter_arg SSA names (e.g., `["%m_acc", "%l_acc"]`)
- `iter_init_values: List[Any]` — initial values for iter_args (scalars or Tiles)

**Returns:** Final iter_arg values as `List[Any]` if iter_args present; None otherwise

**Semantics (control_ops.py:77-167):**

1. **Initialization (parent scope):** Bind initial iter_arg values in the *parent* scope (the one active when for_op is called). If any init value is a Tile, track its LX via `context.track_lx(name, val.size_bytes())`. These persist across iterations.

2. **Loop iterations:** For i in `range(lower_bound, upper_bound, step)`:
   - Push new scope for body-local values
   - Set iteration variable: `context.set_value(iter_var_name, i)`
   - Execute body: `result = region_executor(context, body_region)`
   - Extract yielded values if present: if `result` is `_YieldResult` and iter_arg_names is non-empty, save `result.values`
   - Pop scope: frees all body-local LX, including any Tiles that were yielded (they lived in body scope)
   - **Re-bind iter_args:** If yielded values exist, iterate `zip(iter_arg_names, yielded_values)`:
     - `context.untrack_lx(name)` — free old iter_arg's LX
     - `context.set_value(name, val)` — bind new value in parent scope
     - If new value is a Tile, `context.track_lx(name, val.size_bytes())` — track new LX
   - Update `current_values = yielded_values`

3. **Return:** List of final iter_arg values if any values carried; None otherwise.

**Invariant - iter_arg semantics (control_ops.py:144-162):**
Iter_args are loop-carried state. Example from softmax_rowchunk:
```
scf.for %col = %c0 to %c_C step %c_Bc
    iter_args(%m_acc = %m_init, %l_acc = %l_init) {
  ...
  scf.yield %m_new, %l_new   // fed back as next %m_acc, %l_acc
}
```
- `%m_init`, `%l_init` are `tensor<32x1xf16>` → `%m_acc`, `%l_acc` are Tiles occupying LX
- On each yield, old Tile LX is untracked, new Tile LX is tracked
- Yielded values live only in body scope and are freed on pop; re-binding in parent scope requires explicit track/untrack

**Critical Python-ism not mapping to Rust:** The use of `_YieldResult` as a sentinel to distinguish `scf.yield` output from normal returns. In Rust, this becomes an enum type or explicit Result wrapper.

---

### `ControlOps.yield_op(values: List[Any]) -> _YieldResult`
**Returns:** `_YieldResult(values)`

**Semantics (control_ops.py:169-179):** Wraps values in sentinel so loop driver can update iter_args. Called by scf.yield handler; never called directly by user code.

---

### `ControlOps.while_op(context, before_region, after_region, region_executor) -> None`
**Operands:**
- `context: CoreContext` — execution state
- `before_region: List[Operation]` — condition check region
- `after_region: List[Operation]` — loop body region
- `region_executor: RegionExecutor` — region callback

**Returns:** None

**Semantics (control_ops.py:182-218):**
1. Loop up to 10,000 iterations (safety limit; control_ops.py:202)
2. Push scope, execute before_region (yields condition), pop scope
3. If condition falsy, break
4. Push scope, execute after_region (body), pop scope
5. Repeat

---

## Dialect Handlers (scf_ops.py)

### `scf__if(op, context, env) -> Any`
**Call path:** `env.execute_region` → `ControlOps.if_op`

**Operands extraction (scf_ops.py:28-30):**
- `op.operands[0]` — condition SSA name, resolved via `context.get_value()`
- `op.regions[0]` — then_region (empty list if missing)
- `op.regions[1]` — else_region (empty list if missing)

**Returns:** Unwrapped result from ControlOps.if_op

---

### `scf__for(op, context, env) -> Any`
**Call path:** `env.execute_region` → `ControlOps.for_op`

**Operands extraction (scf_ops.py:35-44):**
- `op.operands[0..2]` — lower_bound, upper_bound, step (SSA names, resolved)
- `op.operands[3..]` — iter_arg initial values (SSA names, resolved)
- `op.attributes["iter_var"]` — induction var name (default `"%i"`)
- `op.attributes["iter_args"]` — list of iter_arg SSA names
- `op.regions[0]` — body_region

**Returns (scf_ops.py:53-60):**
- Single-element list → unwrapped scalar/Tile
- Multi-element list → tuple
- None if no iter_args

---

### `scf__yield(op, context, env) -> _YieldResult`
**Operands extraction (scf_ops.py:65-66):**
- `op.operands` — list of SSA names to yield, resolved to values

**Returns:** `_YieldResult(values)`

---

## Block Arguments & Region Scoping

**Parser (scf_ops.py:96-116):**
- `^bb0(%arg0: type, %arg1: type, ...)` syntax parsed into synthetic `region.bb0_args` operation
- Handler emits no-op at execution time (scf_ops.py:90-93)
- **Block arg binding:** Performed by enclosing op handler (scf.for binds iter_var; linalg.generic binds bb0 args)

**Scope stack example (grid.py:50-84):**
```
Function entry:
  _scope_stack = [{"%core_id": 0, "%c32": 32, "%input_view": MemRef(...)}]

Inside scf.for body (iteration 0):
  _scope_stack = [
    {"%core_id": 0, "%c32": 32, "%input_view": MemRef(...)},  # function
    {"%row": 0, "%tile": Tile(32x1024), "%row_max": Tile(32x1)}  # body
  ]
  get_value("%input_view") → searches top-to-bottom, finds in scope[0]

After pop_scope():
  _scope_stack = [{"%core_id": 0, "%c32": 32, "%input_view": MemRef(...)}]
  %tile, %row_max freed from LX via untrack_lx in pop_scope
```

---

## **CRITICAL: Generator Yield & Cross-Core Communication**

**Current constraint (interpreter.py:195-201):**
```python
def execute_region(self, context: CoreContext, operations: List[Operation]) -> Any:
    """Execute a nested region synchronously (scf.for body, scf.if branch, etc.).

    Comm ops cannot appear inside nested regions in the current spec, so
    this stays sync — no generator machinery needed.
    """
```

**Consequence:** Nested regions (scf.for body, scf.if branch) **cannot yield `RecvRequest`**. Only the top-level function body steps a generator; when a comm op (e.g., `ktdp.transfer`) yields `RecvRequest`, the scheduler's `yield from` machinery (grid.py, not shown here) parks the core until the tile arrives. Once resumed, execution continues in the top-level function body, not inside a nested region.

**If future specs allow comm inside nested regions, redesign needed:**
- `execute_region` must become a generator: `def execute_region(...) -> Generator[RecvRequest, Tile, Any]`
- Loop driver must handle `yield from env.execute_region(context, body)` to bubble RecvRequest up
- Iter_arg re-binding must happen *after* generator resumes (after RecvRequest is satisfied)
- State machine required: track pending iter_arg rebind across suspension boundary

**Current execution model (no nested comm):**
- Regions run **synchronously to completion**
- RegionExecutor signature is `Callable[[CoreContext, List[Operation]], Any]` (not a generator type)
- Dialect handler for scf.for owns the entire loop; no suspend/resume inside
- All cross-core comm happens at top-level function body → pushed to scheduler as `RecvRequest`

**UPDATE — comm inside nested regions now landed (#133):** The "future specs"
case above is now implemented. Python made `execute_region_with_comms`
(`ktir_cpu/ops/control_ops.py` — the `for_op_with_comms` / `if_op_with_comms`
generators) a generator that `yield from`s a comm op's generator so a recv
inside an scf.for / scf.if body bubbles up to the scheduler; the iter_arg rebind
happens after resume.

Rust has no generators, so the port is an **explicit resumable state machine**:
`RegionCommDriver` (`ktir-emulator/src/comm_sched.rs:757`), which implements
`CommOp` so the runner drives it through the same recv/resume protocol as a
top-level collective. It holds a **`frames: Vec<RegionFrame>` stack** (frame enum
at `comm_sched.rs:714`): each `For` / `If` frame stores an **index `path:
Vec<usize>`** locating its scf op in the op tree (re-navigated via `op_at`
instead of holding a borrow), a `body_cursor` (next body op), iter_arg
`current_values`, `cur_i`, and a `scope_open` flag. On a comm op it **parks**
the inner `Box<dyn CommOp>` in `inner` and returns `FrameStep::Suspend(req)`
(`~comm_sched.rs:1036`); on resume it steps `inner` with the delivered tile
(`~:919`), then re-navigates the path + cursor to continue the body exactly where
it left off. `CoreRunner::step` parks/resumes the whole driver via
`active_region` (`~comm_sched.rs:1279`), the dual of Python's implicit
call-stack preservation made explicit and heap-resident.

---

## Tile Lifetime & LX Tracking

**Scope-lifetime semantics (grid.py:157-178):**
- SSA values immutable; once bound, never reassigned in same scope
- Scope exit = earliest safe deallocation point
- Watermark bump-allocator: `push_scope()` snapshots `lx.next_ptr`; `pop_scope()` rewinds it
- Invariant: `len(_lx_next_ptr_stack) == len(_scope_stack) - 1` (watermark stack one shorter than scope stack)

**Iter_arg mutation (control_ops.py:157-163):**
- Iter_args **are reassigned** in parent scope (special case, allowed)
- Old Tile's LX untracked; new Tile's LX tracked
- Yielded Tiles live in body scope; freed on pop; new binding in parent scope is a different Tile

---

## Summary of Redesign Points for Rust

1. **`_YieldResult` sentinel:** Becomes an enum `YieldValue { Yielded(Vec<Value>), Normal(Value) }` or use `Result<Value, Vec<Value>>` pattern
2. **`RegionExecutor` callback:** Currently `Fn(CoreContext, Vec<Operation>) -> Any`. In Rust: `Fn(&mut CoreContext, &[Operation]) -> RustValue` (no generators in nested regions yet)
3. **Scope stack:** `Vec<HashMap<String, Value>>` or `Vec<BTreeMap<String, Value>>`. Push/pop on function entry/exit and region nesting.
4. **LX watermark:** `Vec<usize>` parallel to scope stack. Snapshot/restore on push/pop.
5. **Iter_arg re-binding:** Explicit untrack + set_value + track pattern, not implicit in a sentinel
6. **Block args:** No-op at runtime; binding happens in enclosing op handler (copy linalg_ops.py pattern)
7. **Generator/yield (if enabled in future):** Regions return `enum LoopResult { Yield(Vec<Value>, RecvRequest), Done(Value) }` or use Rust's async/await or explicit state enum. Loop driver must `match` on this and re-bind iter_args after yield resolves.

---

Perfect. Now I have all the necessary context. Let me create a comprehensive spec for the Rust engineer to implement the comm subsystem.

# comm

## Overview
Cross-core communication ops: send/recv/collectives. The subsystem couples generator-based suspension/resume with a scheduler that resolves blocked recvs across a grid of cores. Ring reduction is the canonical collective.

---

## Public Types

### `RecvRequest` (grid.py:33-40)
**Frozen dataclass (immutable).**
- `src: int` — core ID to receive from (>= 0, < num_cores)

**Semantics**: Yielded by a comm generator to signal the scheduler that execution is blocked waiting for a tile from the specified source core. The scheduler parks the generator and resumes it with `gen.send(tile)` once the tile arrives.

**Ownership**: RecvRequest is created by the generator and owned by the scheduler. Immutable, no mutation.

---

### `TransferBackend` (comm_ops.py:29-46)
**Abstract trait (ABC in Python).**

**Methods**:
- `run(ctx: CoreContext, core_id: int) -> LXScratchpad`
  - Return the LXScratchpad for *core_id* (remote case only).
  - Synchronous today; future variants may yield `RecvRequest`.
  - Raises `ValueError` if core_id out of range.
  - **Semantics**: Resolves remote LX access. Callers invoke this only for non-local cores.

**Invariant**: No state mutation within `run`. Pure lookup.

**Redesign note**: This is a seam between memory ops and the transport model. Future variants that yield will require driving them through the scheduler protocol (same machinery as `ReduceBackend`). In Rust, this becomes a trait object (dyn TransferBackend) or an enum dispatching on concrete backend variants.

---

### `InstantTransferBackend` (comm_ops.py:49-69)
**Concrete implementation of TransferBackend.**

**Fields**:
- `_memory: SpyreMemoryHierarchy` (private) — reference to the memory hierarchy.

**Methods**:
- `__init__(memory: SpyreMemoryHierarchy)` — Store memory reference.
- `run(ctx: CoreContext, core_id: int) -> LXScratchpad`
  - Direct lookup: `self._memory.get_lx(core_id)`.
  - Validates `0 <= core_id < num_cores`.
  - Raises `ValueError` if out of range (exact message: `"InstantTransferBackend.run: core_id={core_id} is out of range [0, {num}) for this grid"`).
  - **Semantics**: No latency model, no ring messages. Valid for distributed-view cases where LX partitions are pre-seeded by host.

**Ownership**: Holds a reference to SpyreMemoryHierarchy (immutable for the duration of a run).

---

### `ReduceBackend` (comm_ops.py:86-108)
**Abstract trait (ABC).**

**Methods**:
- `run(context: CoreContext, tile: Tile, core_group: List[int]) -> Union[Tile, Generator[RecvRequest, Tile, Tile]]`
  - **Signature notes**:
    - Generator form yields `RecvRequest` at blocking points; receives `Tile` on resume; returns final reduced `Tile`.
    - Plain function form returns `Tile` directly (synchronous).
    - **Semantics**: Caller uses `inspect.isgenerator()` to distinguish; scheduler treats both uniformly.
  - Returns the reduced tile for *this* core.
  - **Invariant**: Cores not in `core_group` return *tile* unchanged without communicating.
  - **Key semantics**: Each call runs *once* per participating core. The backend owns algorithm (ring rounds, LX-scratchpad accumulation, etc.), messaging, and completion.

**Ownership**: `context`, `tile`, and `core_group` are borrowed. The backend may call `context.send_to()` to enqueue messages.

**Redesign note**: Python generators are suspended/resumed via `gen.send(value)`. Rust has no first-class generators; replace with explicit state machine or async/await-like combinator pattern. The protocol is:
1. Yield `RecvRequest(src=X)` → suspend.
2. Scheduler delivers tile from core X → resume with `gen.send(tile)`.
3. Repeat until done, then return final result.

---

### `RingReduceBackend` (comm_ops.py:110-187)
**Concrete generator-based reduction.**

**Fields**:
- `reduce_fn: Callable[[Tile, Tile], Tile]` — Binary associative reduce operation (e.g., sum, max). Called as `reduce_fn(result, received)`.

**Methods**:
- `__init__(reduce_fn: Callable[[Tile, Tile], Tile])` — Store the reduction function.
- `run(context: CoreContext, tile: Tile, core_group: List[int]) -> Generator[RecvRequest, Tile, Tile]`
  - **Generator protocol**:
    1. If `context.core_id not in core_group`, return *tile* unchanged (non-participating).
    2. Compute `n_cores = len(core_group)`.
    3. Compute `my_idx = core_group.index(context.core_id)`.
    4. Compute ring neighbors:
       - `next_core = core_group[(my_idx + 1) % n_cores]`
       - `prev_core = core_group[(my_idx - 1) % n_cores]`
    5. Initialize state:
       - `result = tile.copy()` — accumulator.
       - `to_forward = tile.copy()` — tile to send next round.
    6. Loop *exactly* `n_cores - 1` times:
       - Call `context.send_to(next_core, to_forward)`.
       - Yield `RecvRequest(src=prev_core)`.
       - Receive tile on resume (bound to `received`).
       - `result = self.reduce_fn(result, received)`.
       - `to_forward = received` — **always forward the received tile unchanged** (not the accumulator; forwarding accumulator causes double-counting).
    7. Return `result`.

  - **Algorithm correctness** (from docstring):
    - Each starting tile travels exactly `N-1` hops around the ring, visiting every other core once.
    - Each visited core folds the tile into its accumulator.
    - After `N-1` rounds, every core's accumulator has seen all `N` starting tiles → full reduction.
    - Example (4 cores, sum, [1,2,3,4]): round 1 accumulators are [1+4=5, 2+1=3, 3+2=5, 4+3=7]; round 3 all cores hold 10.

  - **Invariants**:
    - Core 0 sends to core 1, core 1 sends to core 2, …, core N-1 sends to core 0 (cyclic).
    - The *received* tile (not the accumulator) is forwarded to the next core.
    - Exactly `N-1` yields per core.

**Ownership**: `reduce_fn` is a closure or function pointer, immutable. `context` is borrowed for `send_to()` and resume. `tile` and received values are copied (via `.copy()`) so no shared mutation.

**Key Python-ism**: Generator protocol (`yield`, `gen.send()`, implicit state machine). **Rust redesign**: Replace with explicit state enum or async task. State transitions:
  ```
  Idle → SendAndWaitRound1 → (recv) → FoldAndForward → SendAndWaitRound2 → … → Return
  ```

---

### Backend Registry (comm_ops.py:199-236)
**Module-level state** (mutable):
- `_REDUCE_BACKENDS: Dict[str, Type[ReduceBackend]]` — Global dict keyed by op_name.

**Functions**:
- `register_reduce_backend(op_name: str, backend_cls: Type[ReduceBackend]) -> Callable`
  - Decorator. Adds `op_name -> backend_cls` to `_REDUCE_BACKENDS`.
  - Re-registration silently overwrites.
  - **Exact semantics** (from docstring): "Single op_name per call — keep registrations explicit. Re-registration silently overwrites (matches the parser/handler registries)."
  - Returns the decorated function unchanged (identity decorator).

- `get_reduce_backend(op_name: str) -> Type[ReduceBackend]`
  - Look up class registered for *op_name*.
  - Raises `RuntimeError` with message: `f"No reduce backend registered for op_name {op_name!r}. Add @register_reduce_backend({op_name!r}, <BackendCls>) above the dialect handler."` if not found.

**Redesign note**: Python's decorator system and dict registry are straightforward. In Rust, implement as a `HashMap<String, Box<dyn Fn() -> Box<dyn ReduceBackend>>>` or similar factory pattern. Or use a compile-time registry macro system. The key invariant: each op_name maps to *one* backend class; lookup failures are hard errors (not fallback/default).

---

### `CommOps` (comm_ops.py:243-273)
**Stable per-core comm surface. Static methods only (stateless).**

**Methods**:
- `reduce(context: CoreContext, tile: Tile, core_group: List[int], backend: ReduceBackend) -> Generator`
  - Passthrough: `return backend.run(context, tile, core_group)`.
  - **Semantics**: Single entry point for dialect handlers and tests. The backend owns the algorithm; `CommOps.reduce` is a thin wrapper that wires `context` into the chosen backend.
  - **Return type**: Generator (when backend.run is generator-shaped) or Tile (when synchronous). The return type annotation is `Generator` but the actual runtime type depends on backend.run.

- `reduce_return(value: Tile) -> Tile`
  - Identity passthrough: `return value`.
  - **Semantics**: Used to return a value from a reduction block (probably dialect-specific context). No-op in the comm module itself.

**Ownership**: No state. All args borrowed.

---

## Scheduler: `GridExecutor.execute_with_communication` (grid.py:448-543)

**Entry point for scheduler-driven execution across multiple cores.**

### State Machine & Protocol

#### Input State
- `operations: List[Operation]` — IR ops to execute on all cores.
- `input_ptrs: Dict[str, Any]` — Function inputs (input names → values).
- `execute_op: Callable[[Operation, CoreContext], Any]` — User-supplied op executor.
- `transfer_backend: Optional[TransferBackend]` — For resolving remote `ctx.get_lx()` calls.

#### Scheduler Internals (local to `execute_with_communication`)

**Message queue**:
```
messages: Dict[Tuple[int, int], deque]  # (src, dst) -> deque[Tile]
```
Maps `(source_core, dest_core)` pairs to FIFO queues of tiles in flight.

**Execution stacks**:
```
stacks: Dict[int, CoreExecutionStack]  # core_id -> stack (active cores only)
```
Each core has a `CoreExecutionStack` that wraps the generator returned by `_execute_until_block` and tracks the generator's state.

**Wait state**:
```
waiting: Dict[int, int]  # core_id -> src_core (cores blocked on recv)
results: Dict[int, Any]  # core_id -> final result (completed cores only)
```

**Nested functions** (closure over message queue and stacks):
1. `_enqueue(src: int, dst: int, tile: Tile)` → Add tile to `messages[(src, dst)]` queue.
2. `_pop(src: int, dst: int) -> Optional[Tile]` → Remove and return oldest tile from `messages[(src, dst)]`, or None if queue empty.
3. `_advance(core_id: int, send_val: Any = None)` → Step the generator:
   - Call `stack.resume(send_val)`.
   - If generator yields `RecvRequest`, set `waiting[core_id] = request.src`.
   - If generator completes (StopIteration), set `results[core_id]` and remove from stacks.
4. `_try_deliver(core_id: int) -> bool` → Attempt to deliver a pending message:
   - If core not waiting, return False.
   - If no tile from the awaited source, return False.
   - Pop tile, delete core from waiting, call `_advance(core_id, tile)`, return True.

#### Main Loop

1. **Initialization** (lines 519–533):
   - For each core, attach scheduler functions:
     - `send_fn = lambda dst, tile: _enqueue(core.core_id, dst, tile)`
     - `transfer_fn = lambda src: transfer_backend.run(core, src)` if backend else raises error.
   - Create `CoreExecutionStack` for each core and store in stacks.
   - Call `_advance(core.core_id)` (initial step; no send_val).

2. **Scheduler loop** (lines 535–541):
   ```python
   while stacks:
       if not any(_try_deliver(c) for c in tuple(stacks)):
           # Deadlock: no core could advance
           raise RuntimeError(f"Deadlock detected: {wait_desc}")
   ```
   - Round-robin over all active cores.
   - For each core, attempt to deliver a pending message via `_try_deliver`.
   - If any core advanced, loop again.
   - If no core advanced (all waiting, but no messages can be delivered), raise deadlock error.

3. **Return** (line 543):
   - Collect results for all cores in order: `[results[i] for i in range(self.num_cores)]`.
   - Empty results default to None.

#### Key Invariants

1. **Generator/non-generator duality**: The executor distinguishes via `inspect.isgenerator(result)`.
   - If True: yield from that generator, which may yield `RecvRequest`.
   - If False: store directly and move to next op.

2. **RecvRequest type safety** (grid.py:347–349): If a generator yields a non-RecvRequest value, raise `TypeError`.

3. **Deadlock detection**: If all remaining cores are waiting but no message can be delivered, raise with diagnostic info (which cores wait on which sources).

4. **Message FIFO ordering**: Per (src, dst) pair, tiles are delivered in order (deque.popleft()).

5. **Per-core context attachment** (lines 105–126):
   - `CoreContext.attach_scheduler(send_fn, transfer_fn)` wires the scheduler functions.
   - `send_to()` and `get_lx()` call these functions; raise if not attached.
   - Detach (or re-attach) at the end or between runs.

#### `CoreExecutionStack` (grid.py:291–356)

Wraps a single core's generator and tracks wait state.

**Fields**:
- `core: CoreContext` — The core context.
- `waiting_on: Optional[int]` — Src core ID if blocked, else None.
- `_gen: Generator` — The generator from `_execute_until_block`.

**Methods**:
- `resume(send_val: Any = None) -> Any`:
  - Step the generator: `self._gen.send(send_val) if send_val else next(self._gen)`.
  - If generator yields `RecvRequest`, set `waiting_on = request.src` and return None.
  - If StopIteration, capture and return `e.value` (the final result).
  - **Semantics**: On success, generator is paused at a yield; on completion, generator is exhausted.

- `is_blocked() -> bool`:
  - Return `self.waiting_on is not None`.

**Generator shape** (`_execute_until_block`, lines 315–328):
- Bind input SSA values from `input_ptrs`.
- For each op, call `execute_op(op, core)`.
- If result is a generator, `yield from result` (drive it through the scheduler).
- If a comm op returns a generator, the `yield from` bubbles each `RecvRequest` up to the scheduler.
- When the generator resumes (via `gen.send(tile)`), the value is bound to the op's result and execution continues.
- Return the final op result (or None if no ops).

---

## State Machine & Suspension/Resume Protocol

### From a single core's perspective:

```
┌─────────────────────────────────────────┐
│ CoreExecutionStack created; _gen ready  │
└──────────────────┬──────────────────────┘
                   │ scheduler._advance(core_id)
                   ↓
     ┌─────────────────────────┐
     │ Execute ops until yield │
     └──────────┬──────────────┘
                │
        ┌───────┴───────┐
        │               │
   No yield         Yields RecvRequest
   (StopIteration)  (comm op)
        │               │
        ↓               ↓
    ┌─────────┐   ┌──────────────────┐
    │ DONE    │   │ BLOCKED on recv   │
    │ (move   │   │ waiting_on=src    │
    │ to      │   │ (park generator)  │
    │results) │   └──────────┬────────┘
    └─────────┘              │
                   Tile arrives from src
                   _try_deliver succeeds
                             │
                             ↓
             ┌──────────────────────────┐
             │ _advance(core, tile)     │
             │ gen.send(tile) resumes   │
             │ Continue execution       │
             └──────────────┬───────────┘
                            │
                    (loop back or done)
```

### From the scheduler's perspective:

```
for each core i:
  create CoreExecutionStack(i)
  _advance(i)  # Initial step
  if waiting[i] is set, mark as blocked

while stacks not empty:
  for each core i in stacks:
    if waiting[i] == src_core:
      tile = pop message from (src_core, i)
      if tile exists:
        _advance(i, tile)
        remove from waiting
        
  if no core advanced:
    raise deadlock(waiting dict)

return [results[0], ..., results[n]]
```

---

## Key Python-isms & Rust Redesign Notes

1. **Generators (suspension/resume)**:
   - Python: `yield RecvRequest(...)`, `gen.send(tile)`.
   - Rust: Replace with explicit state machine enum or async/await. Each state represents a pause point.
   - Example state enum:
     ```rust
     enum ReduceState {
       Idle,
       SendAndWait { round: usize, /* prev state */ },
       WaitingOnRecv { round: usize, pending_tile: Tile },
       Done(Tile),
     }
     ```
   - Or: Use a custom combinator (e.g., `Suspendable<T>` trait) that returns `Suspended(RecvRequest)` or `Ready(T)`.

2. **`inspect.isgenerator()` type dispatch**:
   - Python: Runtime check `if inspect.isgenerator(result)`.
   - Rust: Use enum or trait object to represent "Result or Generator".
   - Example:
     ```rust
     enum OpResult {
       Ready(Tile),
       Suspended(RecvRequest),
     }
     ```

3. **Generator protocol (`send`/`yield`)**:
   - Python: Implicit state saved by the interpreter; `gen.send(val)` resumes with val bound to yield expression.
   - Rust: Explicit state machine with `step(input: Option<Tile>) -> StepResult` method. Or use async/await if moving to an async executor.

4. **Dict-based message queue**:
   - Python: `Dict[Tuple[int, int], deque]` — flexible.
   - Rust: Use `HashMap<(u32, u32), VecDeque<Tile>>` or a more cache-friendly layout (e.g., matrix of queues for small core counts).

5. **Mutable shared state (scheduler internals)**:
   - Python: Direct mutation (send_fn closure mutates messages).
   - Rust: Encapsulate in a struct; pass `&mut` to step functions. Or use interior mutability (Mutex/Cell) if sharing across threads (not needed for single-threaded sim).

6. **Duck typing (ReduceBackend + TransferBackend)**:
   - Python: Classes inherit ABC; runtime `isinstance` checks (implicit).
   - Rust: Trait objects (`dyn ReduceBackend`, `dyn TransferBackend`) or enum dispatch.

7. **Decorator-based registry**:
   - Python: `@register_reduce_backend(op_name, BackendCls)` modifies global dict.
   - Rust: Macro-based static registry or function returning factory. Example:
     ```rust
     lazy_static! {
         static ref REDUCE_BACKENDS: Mutex<HashMap<String, Box<dyn Fn() -> Box<dyn ReduceBackend>>>> = Mutex::new(HashMap::new());
     }
     macro_rules! register_reduce_backend { ... }
     ```

---

## Constants

- `HBMSimulator.STICK_BYTES = 128` (memory.py:214).
- `LXScratchpad.capacity = 2 MB` (default; memory.py:298).
- `HBM capacity = 128 GB` (default; memory.py:216).

---

## Critical Sections & Exact Formulas

### Ring reduce loop (comm_ops.py:180–185)
```python
for _ in range(n_cores - 1):
    context.send_to(next_core, to_forward)
    received = yield RecvRequest(src=prev_core)
    result = self.reduce_fn(result, received)
    to_forward = received  # NOT result — critical!
```
**Formula**: Exactly `N-1` iterations for an N-core group. Each iteration sends, waits, folds, and prepares the next message. The invariant `to_forward = received` (not accumulated value) ensures each starting tile hops exactly `N-1` times without double-counting.

### Neighbor computation (comm_ops.py:174–175)
```python
next_core = core_group[(my_idx + 1) % n_cores]
prev_core = core_group[(my_idx - 1) % n_cores]
```
**Formula**: Cyclic indexing with modulo arithmetic. `(my_idx + 1) % n_cores` wraps to 0 after the last core; `(my_idx - 1) % n_cores` wraps to `n_cores - 1` before the first.

### Deadlock detection (grid.py:536–541)
```python
if not any(_try_deliver(c) for c in tuple(stacks)):
    wait_desc = "; ".join(f"core {c} waiting on recv from core {s}" for c, s in waiting.items())
    raise RuntimeError(f"Deadlock detected: {wait_desc}")
```
**Condition**: If no core in the current stacks dictionary can advance (all waiting, all messages empty or from non-waiting sources), deadlock. The diagnostic message lists all waiting cores and their sources.

---

## Ownership & Mutation Tracking

| Entity | Owner | Mutated By | Ownership Model |
|--------|-------|-----------|-----------------|
| `messages` | Scheduler | `_enqueue`, `_pop` | Mutable dict (internal to execute_with_communication) |
| `stacks` | Scheduler | `_advance` (remove on completion) | Mutable dict |
| `waiting` | Scheduler | `_advance`, `_try_deliver` | Mutable dict |
| `results` | Scheduler | `_advance` | Mutable dict |
| `core.lx` | Core | Ops (load, compute, store) via LX allocator | Borrowed from SpyreMemoryHierarchy |
| `core._send_fn` | Core | `attach_scheduler` | Stored function pointer |
| `core._transfer_fn` | Core | `attach_scheduler` | Stored function pointer |
| `generator` (CoreExecutionStack) | Stack | `_execute_until_block` (internal) | Owned by stack; stepped via `resume()` |
| Tile (in message queue) | Message queue | None (immutable) | Copied when sent; borrowed when received |
| `reduce_fn` (in RingReduceBackend) | Backend instance | None (immutable) | Closure or function pointer |

---

## Summary of Trickiest Redesign Areas

1. **Generator protocol → State machine**: Lines 168–185 (RingReduceBackend.run) must become explicit state with labeled pause points. The loop over `n_cores - 1` rounds and the `yield`/`send` dance is the core complexity.

2. **Scheduler main loop**: Lines 535–541 (GridExecutor) implements a work-stealing loop with deadlock detection. In Rust, avoid unbounded spinning; use a work queue or condition variable to signal message arrivals.

3. **Type-driven dispatch (generator vs non-generator)**: Lines 321–328 (CoreExecutionStack._execute_until_block) uses `inspect.isgenerator()` to branch. Rust must encode this in the type system (Result enum or trait object) and match at call sites.

4. **Closure-based send/transfer functions**: Lines 520–531 (GridExecutor) bind `_src` and `_bk` via lambdas. Rust can use Box<dyn Fn> or move closures; the key is that `send_fn` captures the core_id and `_enqueue` reference.

5. **Deadlock detection via "try all, none succeeded"**: Lines 535–541. This is a busy-wait + backoff or event-driven wakeup. For single-threaded sim, busy-wait is fine; for multi-threaded, use condition variables or channels.

---

## Inter-tile collective (`ktdp.inter_tile_produce` / `ktdp.inter_tile_reduce`)

The grouped all-reduce collective: each core in a workgroup produces a per-core
*partial* and then every core reduces every other core's partial into the same
group result. **Consumer set == producer set ⇒ in-group all-reduce** (every
producer is also a consumer). It runs on the **same ring machinery** as the
`ktdp.reduce` backend above.

**Python**:
- `ktdp__inter_tile_produce` (`ktdp_ops.py:944`) — resolve the group index from
  the affine `groups`/`producer` sets, run the producer region with the group
  index bound, capture the `yield_partial` value, and return a per-core
  `TileFuture`.
- `ktdp__yield_partial` (`ktdp_ops.py:1042`) / `ktdp__yield_reduced`
  (`ktdp_ops.py:1048`) — region terminators that park the yielded partial /
  combiner result (both just extract operands and return them).
- `ktdp__inter_tile_reduce` (`ktdp_ops.py:1096`) — validate the `TileFuture`
  operand, build a `CommPlan` from the producer/consumer sets, derive
  `reduce_fn` from the combiner region, pick `RingReduceBackend`, run it. Cores
  not in the producer set inject identity so the lock-step fold stays
  well-defined (`comm_ops.py:299–345`); non-consumers run but return `None`.
- `CommPlan.for_reduce` (`comm_ops.py:123`) — enumerate `producer_set` and
  `consumer_set` over the workgroup at `group_idx`.
- `TileFuture` (`ir_types.py:330`) — per-core handle: `partial_tensor_types`,
  `local_partial` (Option tuple of Tiles), `producer_set`, `groups_set`,
  `group_idx`. **Per-core**, not workgroup-shared.

**Rust** (port — first time this collective is captured in the map):
- `inter_tile_produce` (`ktir-emulator/src/dialects/ktdp_comm.rs:104`) mirrors
  the Python producer; `yield_partial` (`:45`) / `yield_reduced` (`:53`) park
  the value under `COMM_YIELD_KEY` via `park_yield`.
- `InterTileReduce` CommOp (`ktir-emulator/src/comm_sched.rs:448`) is the
  consume side: it carries `plan` (`CommPlan`), `local_partial`, `identity`,
  the combiner block-arg names + body, `result_shape`, `num_cores`, and a
  `RingState`. `CommPlan::for_reduce` (`comm_sched.rs:418`) enumerates the
  producer/consumer sets (the full-barrier case). The `RingState` enum
  (`comm_sched.rs:294`) — `Init` / `Running { accumulator, to_forward, rounds,
  next, prev }` — is **reused verbatim** by both `RingReduce` and
  `InterTileReduce`, so the inter-tile collective is exactly the ring loop above
  driven through the same `step()`/recv/resume protocol.
- The all-reduce invariant is stated in-code at `comm_sched.rs:399–400`
  (`// consumer_set == producer_set ⇒ // in-group all-reduce.`); the
  `ring_reduce.mlir` example pins the same set for both
  `producer_tiles_per_group` and `consumer_tiles_per_group`.
- `TileFuture` (`ktir-core/src/ir.rs:79`): `local_partial: Option<Tile>`,
  `producer_set`, `groups_set`, `group_idx: i64` — the Rust analogue of the
  Python per-core future.

---

# dialect-arith-math

## Core Data Model

**Tile** struct: `Tile { data: np.ndarray, dtype: str, shape: tuple }`
- Tile.data owns the ndarray (immutable from op perspective; new Tile returned for each result)
- dtype: KTIR dtype string (e.g., "f16", "f32", "i32", "i64", "i1", "index")
- shape: output shape tuple
- Broadcasting: scalar + Tile returns Tile (scalar broadcasts implicitly)

**Scalars**: Python int, float, or numpy scalar (np.floating, np.integer, np.generic)

**Duality**: Every op has two code paths — Tile (vectorized np.ndarray ops) and scalar (single-value Python ops). Duplication is pervasive; no shared dispatch.

---

## Arith Dialect Ops

### Float Binary Ops (latency_category=COMPUTE_FLOAT)

| Op Name | Operands | Semantics | Duality | Result Type |
|---------|----------|-----------|---------|------------|
| **arith.addf** | a: f32/f16, b: f32/f16 | Element-wise `a + b` | Tile or float | Same as inputs |
| **arith.subf** | a: f32/f16, b: f32/f16 | Element-wise `a - b` | Tile or float | Same as inputs |
| **arith.mulf** | a: f32/f16, b: f32/f16 | Element-wise `a * b` | Tile or float | Same as inputs |
| **arith.divf** | a: f32/f16, b: f32/f16 | Element-wise `a / b` (true division) | Tile or float | Same as inputs |
| **arith.remf** | a: f32/f16, b: f32/f16 | Element-wise `a % b` (fmod) | Tile or float | Same as inputs |

All use `_float_binop(op, context, operator_fn)` helper which extracts operands and applies operator.

### Float Unary Ops (latency_category=COMPUTE_FLOAT)

| Op Name | Operand | Semantics | Duality |
|---------|---------|-----------|---------|
| **arith.negf** | x: f32/f16 | Element-wise `-x` | Tile or float |
| **arith.absf** | x: f32/f16 | Element-wise `abs(x)` (np.abs or Python abs) | Tile or float |

### Float Min/Max (latency_category=COMPUTE_FLOAT)

| Op Name | Operands | Semantics | Duality | Note |
|---------|----------|-----------|---------|------|
| **arith.maxf** / **arith.maximumf** | a: Tile, b: Tile | `np.maximum(a.data, b.data)` | Tile-only | NaN propagates (e.g., NaN vs 5.0 → NaN) |
| **arith.maxnumf** | a: Tile, b: Tile | `np.fmax(a.data, b.data)` | Tile-only | NaN non-propagating (NaN vs 5.0 → 5.0) |
| **arith.minf** / **arith.minimumf** | a: Tile, b: Tile | `np.minimum(a.data, b.data)` | Tile-only | NaN propagates |
| **arith.minnumf** | a: Tile, b: Tile | `np.fmin(a.data, b.data)` | Tile-only | NaN non-propagating |

**Invariant**: All min/max ops take Tile operands directly (no type hints say Tile-only, but dialect handlers pass Tile objects).

### Float Comparison (latency_category=COMPUTE_FLOAT)

**arith.cmpf**
- Operands: a, b (Tile or scalar float)
- Attribute: `predicate` ∈ {`oeq`, `ogt`, `oge`, `olt`, `ole`, `one`, `ord`, `ueq`, `ugt`, `uge`, `ult`, `ule`, `une`, `uno`, `false`, `true`}
- Returns: bool (scalar) or Tile with dtype "i1"
- Semantics by predicate:
  - **Ordered** (`o*`): standard comparison, returns False if either operand is NaN
  - **Unordered** (`u*`): OR result with "either is NaN" condition
  - `oeq` / `ueq`: equality (ueq → `(a==b) | (isnan(a)|isnan(b))`)
  - `one`: `(a != b) & !(isnan(a)|isnan(b))`
  - `ord`: `!(isnan(a)|isnan(b))`
  - `uno`: `isnan(a)|isnan(b)`
  - `false`, `true`: constant False/True
- Broadcasting: if either operand is Tile, broadcast scalar to shape and return Tile; else return scalar bool
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/arith_ops.py:289–320

### Integer Binary Ops (latency_category=COMPUTE_INT)

| Op Name | Operands | Semantics | Duality | Note |
|---------|----------|-----------|---------|------|
| **arith.addi** | a, b: int or Tile | Element-wise `a + b` | Both | Broadcasts scalar |
| **arith.subi** | a, b: int or Tile | Element-wise `a - b` | Both | Broadcasts scalar |
| **arith.muli** | a, b: int or Tile | Element-wise `a * b` | Both | Broadcasts scalar |
| **arith.divui** | a, b: int or Tile | Unsigned floor division `a // b` | Both | Broadcasts scalar |
| **arith.divsi** | a, b: int or Tile | Signed truncating division (toward zero) | Both | Uses `np.trunc(a/b).astype(...)` for arrays; formula: `a - (a/b)*b` |
| **arith.ceildivsi** | a, b: int or Tile | Signed ceiling division | Both | `math.ceil(a/b)` for scalar; `np.ceil(a/b).astype(...)` for Tile |
| **arith.floordivsi** | a, b: int or Tile | Signed floor division `a // b` | Both | Same as Python `//` |
| **arith.remui** | a, b: int or Tile | Unsigned remainder `a % b` | Both | Broadcasts scalar |
| **arith.remsi** | a, b: int or Tile | Signed truncating remainder: `a - (a/b)*b` | Both | Uses `_truncrem` helper |
| **arith.minsi** | a, b: int or Tile | Signed integer minimum | Both | `np.minimum` or Python `min` |
| **arith.maxsi** | a, b: int or Tile | Signed integer maximum | Both | `np.maximum` or Python `max` |
| **arith.minui** | a, b: int or Tile | Unsigned integer minimum | Both | `np.minimum` or Python `min` |
| **arith.maxui** | a, b: int or Tile | Unsigned integer maximum | Both | `np.maximum` or Python `max` |
| **arith.ceildivui** | a, b: int or Tile | Unsigned ceiling division | Both | `math.ceil(int(a)/int(b))` for scalar |

**Critical invariant** (divsi/remsi): MLIR uses truncation toward zero; Python `//` floors toward -∞. File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/arith_ops.py:149–157

### Integer Bitwise Ops (latency_category=COMPUTE_INT)

| Op Name | Operands | Semantics | Duality |
|---------|----------|-----------|---------|
| **arith.andi** | a, b: int or Tile | Bitwise AND `a & b` | Both |
| **arith.ori** | a, b: int or Tile | Bitwise OR `a \| b` | Both |
| **arith.xori** | a, b: int or Tile | Bitwise XOR `a ^ b` | Both |
| **arith.shli** | a, b: int or Tile | Left shift `a << b` | Both |
| **arith.shrsi** | a, b: int or Tile | Arithmetic right shift `a >> b` (sign-extends) | Both |
| **arith.shrui** | a, b: int or Tile | Logical right shift (zero-fills) | Both | Uses `val1.data.view(np.uint32)` for Tile; `np.uint32(val)` for scalar |

### Integer Comparison (latency_category=COMPUTE_FLOAT)

**arith.cmpi**
- Operands: a, b (int or Tile)
- Attribute: `predicate` ∈ {`eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`, `ule`, `ugt`, `uge`}
- Returns: bool (scalar) or Tile with dtype "i1"
- Semantics: Unsigned and signed predicates use identical comparisons (Python ints have no fixed-width overflow, so sign-bit reinterpretation is N/A)
- Broadcasting: if either operand is Tile, broadcast scalar and return Tile with shape matching Tile operand
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/arith_ops.py:261–286

### Constants & Casts

**arith.constant** (no latency category)
- Attributes: `value`, optional `shape`, `dtype`, `is_tensor`, `dense_list`
- Returns: scalar (int/float) or Tile
- Semantics:
  - If `is_tensor=True`, creates `Tile(np.full(shape, value, dtype=np_dtype), dtype_str, shape)` unless `dense_list=True` (then `np.array(value).reshape(shape)`)
  - Otherwise returns scalar `value`
- Parser handles three forms:
  1. Braced: `{dense<val> : inner_type} : result_type`
  2. Dense: `dense<val> : tensor<NxMxdtype>`
  3. Scalar: `val : dtype`
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/arith_ops.py:327–339

**arith.extf** (no latency)
- Operand: x: Tile or float
- Returns: Tile or float
- Semantics: Widen float (f16→f32); uses `arith_cast(value, np.float32, expect_floating=True, op_name="extf")`
- Invariant: Must validate input is float type; raises TypeError if not

**arith.truncf** (no latency)
- Operand: x: Tile or float
- Returns: Tile or float
- Semantics: Narrow float (f32→f16); uses `arith_cast(value, np.float16, expect_floating=True, op_name="truncf")`

**arith.extsi** / **arith.extui**
- Operand: x: int or Tile
- Returns: int or Tile (i64 / i64)
- Semantics: Zero-extend (ui) or sign-extend (si); cast to i64

**arith.trunci**
- Operand: x: int or Tile (i64)
- Returns: int or Tile (i32)
- Semantics: Truncate to narrower type

**arith.sitofp** (no latency)
- Operand: x: int or Tile
- Attribute: implicit result_type (e.g., "f32")
- Returns: float or Tile (dtype per result_type)
- Semantics: `ArithOps.sitofp(v, dtype)` → `Tile(v.data.astype(np_dtype), dtype, v.shape)` or scalar cast

**arith.uitofp**
- Operand: x: int or Tile
- Returns: float or Tile (f32)
- Semantics: Convert unsigned to float

**arith.fptosi** (no latency)
- Operand: x: float or Tile
- Returns: int or Tile (i32)
- Semantics: Truncate toward zero; `int(value)` for scalar, `value.data.astype(np.int32)` for Tile

**arith.fptoui**
- Operand: x: float or Tile
- Returns: int or Tile (ui32)
- Semantics: Convert float to unsigned integer

**arith.index_cast** / **arith.index_castui**
- Operand: x: int or Tile
- Returns: int
- Semantics: Cast to/from `index` type; Rust must treat `index` as usize-equivalent; Python path returns `int(value)`

**arith.convertf**
- Operand: x: float or Tile
- Returns: float or Tile
- Semantics: Float-to-float conversion; infers direction from input dtype (f16↔f32)

**arith.bitcast** (no latency)
- Operand: x: int or Tile or float
- Attribute: `dst_type` ∈ {`f32`, `i32`, `si32`}
- Returns: Tile or scalar (dst_type)
- Semantics: Reinterpret bits without arithmetic conversion
  - For Tile: `tile.data.view(np.float32)` or `view(np.int32)`
  - For scalar: convert via `.to_bytes(4, "little", signed=...)` → `np.frombuffer(..., dtype=...)[0]`
  - Handles both signed and unsigned integer inputs (e.g., 0xFF800000 as signed -8388608 or unsigned both represent IEEE 754 bit pattern for -inf)
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/arith_ops.py:403–422

### Select

**arith.select** (latency_category=COMPUTE_FLOAT)
- Operands: condition (bool/Tile i1), true_val, false_val (same type)
- Returns: Same type as true_val/false_val
- Semantics: `np.where(cond, true, false)` for Tile; scalar ternary for bool
- Broadcasting:
  - If condition is Tile, extract `.data` from any Tile operands; broadcast scalars and apply `np.where`
  - Result dtype/shape: preserves true_val/false_val dtype if Tile, else infers from result
- Invariant: preserve dtypes of true/false values, not force to f16 (was a bug in old version)
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/arith_ops.py:429–448

---

## Math Dialect Ops

All transcendental and element-wise math ops split into Tile and scalar variants (both registered under same handler).

### Unary Transcendental (latency_category=COMPUTE_TRANSCENDENTAL)

| Op Name | Operand | Tile Method | Scalar Method | Formula / Reference |
|---------|---------|-------------|---------------|---------------------|
| **math.exp** | x: Tile/float | `MathOps.exp(tile)` | `MathOps.exp_scalar(val)` | `e^x` (computed f32, cast back to input dtype) |
| **math.sqrt** | x: Tile/float | `MathOps.sqrt(tile)` | `MathOps.sqrt_scalar(val)` | `√x` |
| **math.rsqrt** | x: Tile/float | `MathOps.rsqrt(tile)` | `MathOps.rsqrt_scalar(val)` | `1/√x` |
| **math.log** | x: Tile/float | `MathOps.log(tile)` | `MathOps.log_scalar(val)` | Natural log `ln(x)` |
| **math.log2** | x: Tile/float | `MathOps.log2(tile)` | `MathOps.log2_scalar(val)` | Base-2 log `log₂(x)` |
| **math.log1p** | x: Tile/float | `MathOps.log1p(tile)` | `MathOps.log1p_scalar(val)` | `log(1+x)` (numerically stable) |
| **math.tanh** | x: Tile/float | `MathOps.tanh(tile)` | `MathOps.tanh_scalar(val)` | Hyperbolic tangent |
| **math.sin** | x: Tile/float | `MathOps.sin(tile)` | `MathOps.sin_scalar(val)` | Sine (radians) |
| **math.cos** | x: Tile/float | `MathOps.cos(tile)` | `MathOps.cos_scalar(val)` | Cosine (radians) |
| **math.erf** | x: Tile/float | `MathOps.erf(tile)` | `MathOps.erf_scalar(val)` | Error function; polynomial approx via Abramowitz & Stegun 7.1.26 (max error <1.5e-7) |

**Pattern**: Tile methods cast `tile.data` to f32, apply `np.op()`, cast back to `tile.data.dtype`. Scalar methods preserve input type (e.g., np.float16 → compute as float → return np.float16).

**erf implementation** (critical): File /Users/moosevan/git/ktir-cpu/ktir_cpu/ops/math_ops.py:221–230
```python
def _erf_f32(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (
        1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return np.sign(x) * (1.0 - poly * np.exp(-a * a))
```
Avoids scipy dependency; max error <1.5e-7.

### Unary (Float/Int) (latency_category varies)

| Op Name | Operand | Tile Method | Scalar Method | Latency | Notes |
|---------|---------|-------------|---------------|---------|-------|
| **math.absf** | x: Tile/float | `MathOps.absf(tile)` | `MathOps.absf_scalar(val)` | COMPUTE_FLOAT | Float abs `np.abs(x)` |
| **math.absi** | x: Tile/int | `MathOps.absi(tile)` | `MathOps.absi_scalar(val)` | COMPUTE_FLOAT | Integer abs `np.abs(x)` |
| **math.ceil** | x: Tile/float | `MathOps.ceil(tile)` | `MathOps.ceil_scalar(val)` | COMPUTE_FLOAT | `np.ceil(x)` |
| **math.floor** | x: Tile/float | `MathOps.floor(tile)` | `MathOps.floor_scalar(val)` | COMPUTE_FLOAT | `np.floor(x)` |

### Binary Ops (latency_category=COMPUTE_TRANSCENDENTAL)

**math.powf**
- Operands: base (Tile/float), exponent (Tile/float)
- Semantics:
  - Tile path: `MathOps.powf(base: Tile, exponent: Tile)` → `np.power(base.data.astype(f32), exp.data.astype(f32)).astype(base.data.dtype)`
  - Scalar path: `MathOps.powf_scalar(base, exponent)` → `float(base) ** float(exponent)` cast back to base type
- Returns: Tile or scalar (same dtype as base)

**math.fma** (latency_category=COMPUTE_FLOAT)
- Operands: a, b, c (all Tile or all scalar)
- Semantics: Fused multiply-add `a*b + c`
  - Tile: `(a.data.astype(f32) * b.data.astype(f32) + c.data.astype(f32)).astype(a.data.dtype)`
  - Scalar: `float(a)*float(b) + float(c)` cast back to a's type
- Returns: Tile or scalar (same dtype as a)
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/ops/math_ops.py:257–268

---

## Helper Infrastructure

**_unary(op, context, tile_fn, scalar_fn=None)** (used by all dialect handlers)
- Extracts operand from context
- If Tile: calls `tile_fn(operand)` → returns Tile
- If scalar: calls `scalar_fn(operand)` if provided, else tile_fn coerced to scalar
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/dialects/_helpers.py (not shown, but called everywhere)

**_float_binop(op, context, operator_fn)**
- Extracts two operands, applies operator element-wise (via Tile or scalar op)
- Uses operator.add, .sub, .mul, .truediv, .mod

**_int_binop(op, context, operator_fn)**
- Like _float_binop but for integers

**arith_cast(value, target_np_dtype, expect_floating, op_name)**
- Type validation and narrowing/widening cast helper
- Checks input category (float vs int) matches expectation
- Returns Tile with new dtype or numpy scalar
- Raises TypeError/OverflowError if mismatched type or overflow
- File: /Users/moosevan/git/ktir-cpu/ktir_cpu/ops/arith_ops.py:27–76

---

## Critical Python-isms for Rust Port

1. **Dual scalar/Tile dispatch**: No runtime type tags; Python's `isinstance()` checks branch. Rust must use enums `Value = Scalar(f32|i32|...) | Tile(ndarray)` or trait objects.

2. **Broadcasting by dimension mismatch**: Python implicit broadcasting in operator overloads. Rust must explicit-broadcast scalars to Tile shape.

3. **NumPy dtype tracking**: Python vars are untyped; `.dtype` attribute on arrays. Rust must track dtype separately (KTIR string or enum).

4. **Truncation semantics for divsi/remsi**: MLIR truncates toward zero; Python `//` floors. Must use `trunc(a/b)` not `a//b` for signed division. Line 149–156.

5. **NaN handling in cmpf/minf/maxf**: Two distinct behaviors (`maximum` vs `fmax`, `minimum` vs `fmin`). Port must track which variant.

6. **arith.bitcast view-casting**: Reinterprets bits via NumPy `.view()` for Tiles, `.to_bytes()` + `np.frombuffer()` for scalars. Rust can use `transmute` or byte-level reinterpret (but safety concerns).

7. **Arbitrary-precision Python ints in arith.cmpi/select**: Python ints have no overflow. Fixed-width Rust ints will differ on boundary cases (e.g., i32::MIN vs i32::MAX comparisons).

8. **erf polynomial constants**: Hard-coded Abramowitz & Stegun coefficients (0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429). Port must preserve exact values.

9. **Tile shape mutation**: Python Tiles are created new on each op; no in-place mutation. Result shape determined by `np.broadcast_shapes()` or explicit operand shapes. Rust builder/struct pattern.

10. **Type preservation in select**: Old code forced f16; new code preserves true_val/false_val dtype. Ensure Rust port does NOT default to f16.

---

## Ownership & Mutation

- **Input values**: Read-only; extracted from context, never modified.
- **Output Tiles**: Created fresh for each op; owned by result.
- **Context**: Mutable store, written to via `context.set_value(result_name, value)` (not shown here, but implied by `get_value` pattern).
- **Operand broadcasting**: Scalars broadcast to Tile shapes; no mutation of originals.

---

## Latency Categories Registered

- `LC.COMPUTE_FLOAT`: arith.addf, arith.subf, arith.mulf, arith.divf, arith.remf, arith.negf, arith.absf, arith.maxf/f, arith.minf/f, arith.maxnumf, arith.minnumf, arith.cmpf, arith.select, math.absf, math.ceil, math.floor
- `LC.COMPUTE_INT`: arith.addi, arith.subi, arith.muli, arith.divui, arith.divsi, arith.ceildivsi, arith.floordivsi, arith.remui, arith.remsi, arith.minsi, arith.maxsi, arith.minui, arith.maxui, arith.ceildivui, arith.andi, arith.ori, arith.xori, arith.shli, arith.shrsi, arith.shrui, arith.cmpi
- `LC.COMPUTE_TRANSCENDENTAL`: math.exp, math.sqrt, math.rsqrt, math.log, math.log2, math.log1p, math.tanh, math.sin, math.cos, math.erf, math.powf, math.fma
- No latency: arith.constant, cast ops (extf, truncf, extsi, extui, trunci, sitofp, uitofp, fptosi, fptoui, index_cast, index_castui, convertf, bitcast)

---

## Design Decisions Requiring Redesign in Rust

1. **Enum Value type** (not Union): Python's duck typing requires runtime checks. Rust must define `enum Value { Scalar(ScalarValue), Tile(TileValue) }` or use trait dispatch.

2. **Broadcasting function**: Explicit broadcast_shapes and promote-to-Tile logic (Python does implicitly in operators).

3. **Result dtype inference**: Python infers from input tiles or uses fallback. Rust op handlers must declare expected result dtype explicitly.

4. **Error handling**: Python raises TypeError/NotImplementedError. Rust must use Result<Value, OpError> or similar; no bare panics.

5. **Arbitrary-precision integers**: Use fixed-width i32/i64 with wrapping semantics; differ from Python on overflow (deliberate choice to match MLIR semantics).

6. **erf approximation**: Port polynomial exactly; consider an inline const array for coefficients.

7. **Tile.shape as Vec<usize>**: Python uses tuple; Rust uses Vec or array. Must handle variable-rank tensors.

---

# dialect-linalg-tensor

## Linalg Dialect

### Registered Operations

#### `linalg.matmul`
**Latency:** `COMPUTE_MATMUL`

**Signature:**  
```
linalg.matmul ins(%A, %B) outs(%C) -> result
```

**Operands:**
- `operands[0]` (ins[0]): tensor A, shape MxK
- `operands[1]` (ins[1]): tensor B, shape KxN  
- `operands[2]` (outs): accumulator C, shape MxN (optional)

**Semantics:**  
Computes `result = C + (A @ B)` using `numpy.matmul` where C is the initial accumulator. When C omitted, degenerates to plain matrix multiplication. If len(operands) <= 2, return only `A @ B` without accumulation.

**Mutation:** None; returns new Tile. outs operand (operands[2]) is read-only in execution; no back-write.

---

#### `linalg.batch_matmul`
**Latency:** `COMPUTE_MATMUL`

**Signature:**  
```
linalg.batch_matmul ins(%A, %B) outs(%C) -> result
```

**Operands:**
- `operands[0]` (ins[0]): batched tensor A, shape BxMxK
- `operands[1]` (ins[1]): batched tensor B, shape BxKxN
- `operands[2]` (outs): batched accumulator C, shape BxMxN (optional)

**Semantics:**  
Computes `result = C + (A @ B)` element-wise across batch dimension. Uses `numpy.matmul` which broadcasts batch axes automatically.

**Mutation:** None; returns new Tile.

---

#### `linalg.reduce`
**Latency:** `LC.ZERO` (cost attributed to combiner region ops)

**Signature:**  
```
linalg.reduce ins(%x) outs(%y) { region } dimensions = [dim] -> result
```
Also shorthand: `linalg.reduce { arith.addf } ins(...) outs(...) dimensions = [1]`

**Operands:**
- `operands[0]`: input tensor (ins)

**Attributes:**
- `reduce_fn` (str, optional): Name of combiner op for shorthand form (e.g. "arith.addf"). When present, no region parsed; synthetic region synthesized.
- `dimensions` (int list, optional): Axes to reduce. **Multi-dim now supported (#106)** — see semantics below. (Legacy `dim` is the single-axis form.)
- `outs_var` (str, optional): SSA name of outs buffer; bound as output name for downstream references.

**Region:**  
Two forms:
1. **Explicit region:** `(%in, %out) { %s = <op> %in, %out; linalg.yield %s }`
2. **Shorthand (no region):** Named combiner only; executor synthesizes region.

Block arguments captured via `_resolve_region_body()` (priority: `region.bb0_args` synthetic op > `op.attributes["bb0_names"]` > operands of first body op).

**Semantics:**  
Folds the input tensor along the reduced axes using a **pairwise tree reduction** of the combiner region. MLIR requires the combiner to be associative, so fold order is free. Tree reduction executes `ceil(log2(N))` vectorized region calls instead of N sequential steps. Each region call combines whole sub-array slices as Tiles.

**Multi-dim `dimensions` semantics (#106):**
- `dimensions` **absent** → collapse all axes (flatten then fold to scalar, shape `()`).
- `dimensions = []` (empty) → **identity**: reduce zero axes, returns the input unchanged.
- `dimensions = [d…]` → **tree-fold each axis rightmost-first, then squeeze** the reduced axes out. (Folding axes independently rather than as one flattened group can reorder element groupings vs. MLIR's left-associative scalar loop — pinned by `test_reduce_multi_axis_treefold_bug`.)

Odd-length slices carry forward to the next round (Python: `if n % 2: combined = np.concatenate([combined, tail], axis=dim)`).

**Outs-accumulator fold (now unconditional):** `outs` is the **initial accumulator value**, so the result is `combiner(reduce(ins), outs)`. Python folds the `outs` operand **unconditionally** whenever it is a Tile of the reduced shape (no identity-only guard) — `sum([1,2,3,4])` with `outs` init `100` is `110`, not `10`.

**Execution path:**  
`_tree_fold()` (Python `linalg_ops.py`): recursively splits `acc` in half along the axis, calls `_run_combiner()` on slices as Tiles, concatenates result with unpaired tail, repeats until `n=1`. `linalg__reduce` (`ktir_cpu/dialects/linalg_ops.py:137–246`) drives the absent / `[]` / multi-dim cases and the final unconditional `_run_combiner(reduced, outs_tile)` fold; the `dimensions` attribute is parsed at `linalg_ops.py:473–478`.

`_run_combiner()`: pushes scope, binds `bb0_names[0]` ← `lhs`, `bb0_names[1]` ← `rhs`, executes region via `env.execute_region()`, pops scope, unwraps `_YieldResult`.

**Rust port:** `reduce` (`ktir-emulator/src/dialects/linalg.rs:529–689`) mirrors all three cases — absent=collapse-all (`:609`), `[]`=identity (`:615`), `[d…]`=`tree_fold` rightmost-first + squeeze (`:619`) — and applies the **unconditional** `outs_var` fold at `linalg.rs:642–680` (the comment there pins `test_reduce_folds_outs_init`). The dedicated parser branch is at `ktir-core/src/parser.rs:713` (captures the `outs(...)` buffer name; the `dimensions` IntList is parsed by the generic bare-attr path). **Resident-path correctness:** the fusion pass renames `outs_var` along with every other SSA attr — `rename_attrs` (`ktir-optimizer/src/fusion.rs:1086`, key list at `~:1106`) prefixes `outs_var` so each fused reduce gets its **own fresh** per-node identity accumulator (an unprefixed shared `%sinit` slot let node N's reduce fold node N-1's stale partial, diverging the e2e golden ~30 logits).

**Output binding:**  
Both `result` name and `outs_var` (if present) are bound to context; allows downstream ops to reference by either name.

**Python-ism:** Region execution via interpreter (line 77: `env.execute_region()`); no NumPy shortcut for arbitrary combiners. Full op latency comes from executed body ops.

**Invariant:** Combiner associativity required (MLIR legalization constraint); implementation makes no verification.

---

#### `linalg.fill`
**Latency:** Default (not specified in decorator)

**Signature:**  
```
linalg.fill ins(%scalar) outs(%init) -> result
```

**Operands:**
- `operands[0]`: scalar value (ins)
- `operands[1]`: output tensor (outs)

**Semantics:**  
Creates a new Tile with shape and dtype from outs, all elements set to scalar. `np.full(out_shape, scalar_val, dtype=out_dtype)`. Scalar coerced to float.

**Mutation:** None; returns new Tile.

---

#### `linalg.broadcast`
**Latency:** Default

**Signature:**  
```
linalg.broadcast ins(%x) outs(%y) dimensions = [d0, d1, ...] -> result
```

**Operands:**
- `operands[0]`: input tensor (ins)
- `operands[1]`: output shape template (outs)

**Attributes:**
- `dimensions` (list[int]): Axes at which to insert new dimensions before broadcast.

**Semantics:**  
Expands input by inserting new axes at sorted positions in `dimensions`, then broadcasts to outs shape. Example: `input (2,3)`, `dimensions=[0,2]`, `out_shape=(5,2,4,3)` → `expand_dims` at 0 and 2 → shape `(1,2,1,3)` → broadcast to `(5,2,4,3)`.

Line 251–254: `for d in sorted(dims): data = np.expand_dims(data, axis=d); result = np.broadcast_to(data, out_shape).copy()`

**Mutation:** None; returns new Tile (explicit `.copy()` to avoid broadcast view).

---

#### `linalg.generic`
**Latency:** `LC.COMPUTE_FLOAT`

**Signature:**  
```
linalg.generic { region } ins(%x, %y, ...) outs(%z) indexing_maps = [map0, map1, ...] -> result
```

**Operands:**  
- `operands[0:n_ins]`: input tensors
- `operands[n_ins]`: output tensor (outs)

**Attributes:**
- `n_ins` (int): Number of inputs (outs comes after).
- `indexing_maps` (list[list[int]]): For each operand, list of dimension indices it depends on. E.g. `[[0, 1], [1], [0, 1]]` means ins[0] uses (d0, d1), ins[1] uses (d1), outs uses (d0, d1).

**Region:**  
Block arguments named via `_resolve_region_body()` (same priority as reduce). One arg per input, one for outs. Body ops execute once with full arrays (vectorized).

**Semantics:**  
Broadcasts each input to outs shape per indexing_map: missing dimensions (not in imap) get `np.expand_dims`, present dimensions retained. All inputs + outs broadcast to full iteration space (outs shape). Region body executes **once** with full Tiles bound to block args (vectorized execution, not per-element).

Line 340–342: `for d in range(out_ndim): if d not in imap: data = np.expand_dims(data, axis=d)`

Output block arg (block_args[n_ins]) initialized as copy of outs.data (line 352–354: `Tile(outs_val.data.copy(), ...)`).

Region result unwrapped via `unwrap_yield()` (line 359), broadcast to outs shape if needed (line 362: `np.broadcast_to(out_data.data, out_shape).copy().astype(out_np_dtype)`), returned as Tile.

**Scope management:** `context.push_scope()` / `context.pop_scope()` isolate region bindings; `context.set_value("__linalg_shape__", out_shape)` makes shape available to `linalg.index` (line 334).

**Python-ism:** Vectorized execution via element-wise NumPy ops. No per-element loop; region body must accept Tile arrays, not scalars.

**Invariant:** All indexing_maps must be subsets of `[0, ..., out_ndim-1]`.

---

#### `linalg.index`
**Latency:** Default

**Signature:**  
```
%idx = linalg.index dim : index
```

**Attributes:**
- `dim` (int): Dimension to index.

**Semantics:**  
Returns a 1-D array `np.arange(out_shape[dim])` reshaped to broadcast alongside the output iteration space. Used inside `linalg.generic` body to obtain per-element indices.

Line 372–375: `idx = np.arange(out_shape[dim], dtype=np.int64); reshape = [1]*len(out_shape); reshape[dim] = out_shape[dim]; return Tile(idx.reshape(reshape), "index", tuple(reshape))`

**Context dependency:** Reads `"__linalg_shape__"` from context (set by `linalg.generic` line 334).

---

#### `linalg.yield`
**Latency:** Default

**Signature:**  
```
linalg.yield %val : type
```

**Operands:**  
- `operands`: SSA names to yield (typically one, may be multiple in future MLIR versions).

**Semantics:**  
Terminates region, wraps operand(s) via `ControlOps.yield_op()`. Returns `_YieldResult` object.

---

#### `linalg.transpose`
**Latency:** Default

**Signature:**  
```
linalg.transpose ins(%x) outs(%y) permutation = [d0, d1, ...]
```

**Operands:**
- `operands[0]`: input tensor (ins)
- `operands[1]`: output shape template (outs, used for shape only)

**Attributes:**
- `permutation` (list[int]): Axis permutation. E.g. `[1, 0]` swaps axes.

**Semantics:**  
Applies `np.transpose(data, axes=permutation)`, recomputes shape by permuting dims: `new_shape = tuple(inp.shape[i] for i in permutation)`.

**Mutation:** None; returns new Tile.

---

## Tensor Dialect

### Registered Operations

#### `tensor.empty`
**Latency:** Default

**Signature:**  
```
%t = tensor.empty() : tensor<shape>
```

**Attributes:**
- `shape` (tuple[int]): Output shape (from parsed type).
- `dtype` (str): NumPy dtype string (default "f16").

**Semantics:**  
Creates uninitialized tensor via `np.zeros(shape, dtype)`. Semantically uninitialized but implemented as zeros.

**Mutation:** None; returns new Tile.

---

#### `tensor.splat`
**Latency:** Default

**Signature:**  
```
%t = tensor.splat %scalar : scalar_type -> result_type
```

**Operands:**
- `operands[0]`: scalar value.

**Attributes:**
- `shape` (tuple[int]): Target shape (may be empty).
- `dtype` (str): Output dtype (default "f16").
- `_result_shape`, `_result_dtype` (optional): Fallback shape/dtype from type annotation.

**Semantics:**  
If `shape` is empty (0-tuple), attempts recovery:
1. Check `_result_shape` / `_result_dtype` attributes.
2. Call `_infer_splat_shape()` to find largest Tile in scope (heuristic).
3. Default to `(1,)`.

Operand (if Tile) flattened to scalar (line 62: `scalar.data.flat[0]`). Integer scalars default to `np.int32`; else parsed dtype. Result: `np.full(shape, scalar, dtype)`.

**Python-ism:** Shape inference heuristic (searching context scope stack for largest Tile) is a Python-specific fallback; Rust must require explicit shape.

**Mutation:** None; returns new Tile.

---

#### `tensor.extract`
**Latency:** Default

**Signature:**  
```
%scalar = tensor.extract %tensor[%i, %j, ...] : tensor<...>
```

**Operands:**
- `operands[0]`: source tensor.
- `operands[1:]`: index operands (one per dimension or empty for 0-D).

**Semantics:**  
If src is Tile:
- Empty indices → return single element (0-D tensor): `src.data.flat[0]`.
- Non-empty indices → index as tuple: `src.data[tuple(int(i) for i in indices)]`.

If src already scalar, return as-is.

**Mutation:** None; read-only indexing.

---

#### `tensor.expand_shape`
**Latency:** Default

**Signature:**  
```
%out = tensor.expand_shape %src into tensor<target_shape>
```

**Operands:**
- `operands[0]`: source tensor.

**Attributes:**
- `target_shape` (tuple[int]): New shape (must have same total element count).

**Semantics:**  
Calls `src.data.reshape(target_shape)` if src is Tile; else returns src unchanged. No-copy reshape.

**Mutation:** None; returns new Tile.

---

#### `tensor.collapse_shape`
**Latency:** Default

**Signature:**  
```
%out = tensor.collapse_shape %src into tensor<target_shape>
```

**Operands:**
- `operands[0]`: source tensor.

**Attributes:**
- `target_shape` (tuple[int]): Collapsed shape.

**Semantics:**  
Identical to `expand_shape`: `src.data.reshape(target_shape)`. Name distinction is semantic (expand vs. collapse in MLIR), implementation identical.

**Mutation:** None; returns new Tile.

---

#### `tensor.reshape`
**Latency:** Default

**Signature:**  
```
%out = tensor.reshape %src(%shape_tensor) : (...) -> tensor<target_shape>
```

**Operands:**
- `operands[0]`: source tensor.
- `operands[1]`: shape tensor (1-D, dtype=index; **not used at execution time**).

**Attributes:**
- `target_shape` (tuple[int], required): Static target shape (from result type annotation).
- `dtype` (str): Output dtype.

**Semantics:**  
Ignores runtime shape operand; reads static `target_shape` from attributes (set by parser from result type annotation). Raises `ValueError` if `target_shape` missing. Calls `src.data.reshape(target_shape)` if src is Tile.

**Design note:** Shape operand is parsed but ignored—MLIR always pins target shape statically in the result type. No dynamic reshape.

**Mutation:** None; returns new Tile.

---

#### `tensor.from_elements`
**Latency:** Default

**Signature:**  
```
%shape = tensor.from_elements %d0, %d1, ... : tensor<NxT>
```

**Operands:**
- `operands[0:]`: scalar elements (N elements for shape tensor<N>).

**Attributes:**
- `shape` (tuple[int], required): Output shape (must match operand count).
- `dtype` (str, required): Output dtype (typically "index" for shape tensors).

**Semantics:**  
Collects N operands from context. If operand is Tile, extract first element (line 173: `v.data.flat[0]`). Stack into NumPy array, reshape to `shape`, return as Tile.

Raises `ValueError` if `shape` or `dtype` missing.

**Mutation:** None; returns new Tile.

---

#### `tensor.generate`
**Latency:** Default

**Signature:**  
```
%t = tensor.generate {
  ^bb0(%i: index, %j: index, ...):
    %val = ... (compute from %i, %j)
    tensor.yield %val : dtype
} : tensor<shape>
```

**Operands:**  
None (indices generated internally).

**Attributes:**
- `shape` (tuple[int]): Output shape.
- `dtype` (str): Output dtype.

**Region:**  
Block arguments: one per dimension (all type=index). Body terminates with `tensor.yield`. Block arg names extracted via `_resolve_region_body()` (same as linalg.generic/reduce).

**Semantics:**  
Executes region **once** with **vectorized index grids** (not per-element loop):

Line 231: `grids = np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')`

For shape `(3, 3)`, grids gives:
```
%i -> [[0,0,0],    %j -> [[0,1,2],
       [1,1,1],           [0,1,2],
       [2,2,2]]           [0,1,2]]
```

Block args bound to Tiles of these grids (line 234). Region body (arith ops, comparisons, etc.) operates element-wise on Tile arrays, producing full output in one pass.

Example use: causal mask for attention (line 204): `mask[i,j] = 0.0 if i >= j else -10000.0`

Region result (wrapped by `tensor.yield`) converted to NumPy dtype and returned as Tile.

**Python-ism:** Vectorized meshgrid execution; Rust must use SIMD or parallel index iteration, not scalar per-element loop.

**Scope:** `context.push_scope()` / `context.pop_scope()` isolate block arg bindings.

---

#### `tensor.yield`
**Latency:** Default

**Signature:**  
```
tensor.yield %val : type
```

**Operands:**
- `operands[0:]`: Values to yield.

**Semantics:**  
Terminates `tensor.generate` region body. Wraps operand(s) via `ControlOps.yield_op()`, returning `_YieldResult`.

---

## Cross-Dialect Notes

### NumPy Operations Used

- `np.full()`: Fill array (linalg.fill, tensor.splat).
- `np.broadcast_to()`, `np.expand_dims()`: Shape adjustment (linalg.generic, linalg.broadcast).
- `np.matmul()` / `@`: Matrix multiplication (linalg.matmul, linalg.batch_matmul).
- `np.transpose()`: Axis permutation (linalg.transpose).
- `np.reshape()`: Shape reinterpretation (tensor.reshape, etc.).
- `np.asarray()`, dtype coercion: Uniform array conversion.
- `np.concatenate()`: Tile unpaired slices in tree reduction (linalg.reduce).
- `np.meshgrid()`: Index grid generation (tensor.generate).
- `np.arange()`: Index arrays (linalg.index, tensor.generate grids).
- `np.squeeze()`: Remove axis (linalg.reduce result post-fold).
- `.copy()`: Explicit copies to avoid broadcast views.

### Region Execution

**Shared resolution (`_resolve_region_body`):**  
Both linalg.reduce and linalg.generic (and tensor.generate) resolve block arg names uniformly:
1. `region.bb0_args` synthetic op (from parser) → extract `names` attribute.
2. `op.attributes["bb0_names"]` (mlir_frontend path).
3. First body op's operands (inline shorthand).

**Combiner execution (`_run_combiner`):**  
Isolated scope, binds args, executes region via `env.execute_region()`, unwraps `_YieldResult`, pops scope. Full latency charged to region body ops.

**Tree fold (`_tree_fold`):**  
Pairwise reduction with odd-slice carryforward. `ceil(log2(N))` passes. Each pass calls `_run_combiner` on Tile slices.

### Context Scoping

- `context.push_scope()` / `context.pop_scope()`: Isolate variable bindings for regions.
- `context.set_value()` / `context.get_value()`: Variable lookup.
- `context.get_value("__linalg_shape__")`: Special variable set by linalg.generic, read by linalg.index.

### Key Invariants

1. **linalg.reduce:** Combiner must be associative (MLIR legalization requirement).
2. **linalg.generic:** All indexing_maps dimensions must be valid (< output rank).
3. **tensor.reshape:** Runtime shape operand ignored; target_shape from result type required.
4. **tensor.generate:** Block arg count must equal output rank; all args type=index.
5. **linalg.index:** Must be called within linalg.generic region (reads `__linalg_shape__`).
6. **Broadcast semantics:** No mutable alias views; all reshapes and broadcasts call `.copy()`.

### Python-isms Requiring Redesign in Rust

1. **Region execution as interpreter:** `env.execute_region(context, body_ops)` dispatches ops dynamically. Rust must inline or compile regions; no duck-typed op dispatch.
2. **Dynamic block arg resolution:** Three-priority fallback for block arg names. Rust parser must disambiguate statically.
3. **`_YieldResult` duck-typing:** `isinstance(result, _YieldResult)` unwrapping. Rust enums with pattern matching required.
4. **Shape inference heuristic:** `_infer_splat_shape()` scans entire scope stack to find largest Tile. Rust must require explicit shape or use type inference at parse time.
5. **Vectorized indexing over scope stack:** `tensor_ops.py` line 38–43 iterates all scopes. Rust borrow checker incompatible; must use explicit scope handle or arena.
6. **Mutable shared context:** `context` object passed through all ops, mutations visible globally. Rust must use interior mutability (`RefCell`, `Mutex`) or thread-local storage; or redesign as immutable DAG.
7. **Generator-less tree fold:** NumPy array slicing and concatenation replace scalar loop. Rust integer ranges or SIMD required.
8. **numpy.matmul broadcasting:** Handles arbitrary batch dims; Rust matmul library may require explicit shape contract.

### Type Flow

- **Tile:** Owns `data` (np.ndarray), `dtype` (str), `shape` (tuple[int]). Immutable externally; new Tile created per op.
- **Scalars:** Plain Python int/float; coerced as needed.
- **Index:** Special dtype="index" for linalg.index / tensor.generate block args.
- **_YieldResult:** Wrapper from ControlOps; contains `.values` list.

---

## Parser Patterns

### Regex-based Extraction

- `ins(...)`, `outs(...)`: Matched, operands extracted via `find_ssa_names()`.
- `dimensions = [...]`, `permutation = [...]`: Split on `,`, parse ints.
- `indexing_maps = [...]`: Complex; parsed as affine_map structs, dims extracted.
- `reduce_fn` shorthand: `{ arith.addf }` matched, combiner name extracted (no %SSA inside).
- Type annotations: `tensor<shape>` parsed to shape tuple + dtype via `parse_tensor_type()`.

All parsers return `Operation` IR object or `None` if no match.

---

Perfect. Now I have all the information I need. Let me create a comprehensive markdown specification for the Rust port.

# dialect-ktdp-full

## Overview

The KTDP dialect subsystem consists of grid compute operations (`get_compute_tile_id`, `coreid`), memory view construction operations (`construct_memory_view`, `construct_distributed_memory_view`, `construct_access_tile`), indirect access tile construction (`construct_indirect_access_tile`), and load/store operations with support for coordinate-ordered and distributed memory access patterns. All operations are parsed from MLIR text and execute within a CoreContext environment managing SSA values and per-core state.

---

## Public Types

### `MemRef`
*File: `ktir_cpu/ir_types.py:46–118`*

**Fields:**
- `base_ptr: i32` — stick index (HBM) or byte address (LX)
- `shape: Vec<usize>` — logical tensor dimensions
- `strides: Vec<usize>` — element-count strides (one per axis)
- `memory_space: &str` — `"HBM"` or `"LX"` (validated in constructor)
- `dtype: &str` — element type, default `"f16"`
- `coordinate_set: Option<Union<BoxSet, AffineSet>>` — global coordinates owned by this partition; None for single-allocation views
- `lx_core_id: Option<usize>` — per-core LX SRAM routing when memory_space="LX"; None means executing core's scratchpad

**Methods:**
- `byte_address(&self) -> usize` — absolute byte position; HBM: `base_ptr * STICK_BYTES`, LX: `base_ptr`
- `to_tile_ref(&self) -> TileRef` — convert to byte-addressed TileRef
- `split_addr(byte_addr: usize) -> (usize, usize)` — split byte address into (main, intra) pair; HBM: `(stick_idx, offset)`, LX: `(byte_addr, 0)`
- `size_bytes(&self) -> usize` — total bytes; `prod(shape) * bytes_per_elem(dtype)`

**Invariants:**
- `memory_space ∈ {"HBM", "LX"}`; validated at construction
- `lx_core_id` may only be set when `memory_space == "LX"`
- When `coordinate_set` is set, it must be concrete (no symbolic bounds at MemRef construction time; symbols resolved at `construct_memory_view` time per line 92–96)

---

### `DistributedMemRef`
*File: `ktir_cpu/ir_types.py:121–167`*

**Fields:**
- `partitions: Vec<MemRef>` — N per-partition MemRefs, each carrying its own `coordinate_set`
- `shape: Vec<usize>` — global logical shape (in global coordinates)
- `dtype: &str` — all partitions must have matching dtype

**Methods:**
- `find_partition(coord: &[usize]) -> (usize, &MemRef)` — return first partition whose `coordinate_set` contains *coord*; raises `IndexError` if none found

**Invariants:**
- At least one partition required
- Every partition must have a non-None `coordinate_set` (line 144–147)
- All partitions' `dtype` must match the wrapper's `dtype`
- No allocation/data movement occurs; partition resolution is deferred to access time

---

### `TileRef`
*File: `ktir_cpu/ir_types.py:170–196`*

**Fields:**
- `base_ptr: usize` — always absolute byte address (regardless of memory space)
- `shape: Vec<usize>` — tile dimensions
- `strides: Vec<usize>` — element-count strides
- `memref: MemRef` — parent MemRef (always set; owns memory_space and address conversion)
- `dtype: &str` — default `"f16"`
- `coordinate_set: Option<Union<BoxSet, List<Tuple<usize, ...>>>>` — per-survivor metadata from `distributed_tile_access`; None for single-allocation tiles
  - `BoxSet`: axis-aligned C_i (O(ndim) ops)
  - `List`: pre-enumerated points from slow path (B_i or A is AffineSet)
- `partition_origin: Option<Tuple<usize, ...>>` — p_i = min(B_i) in global coords; set by `distributed_tile_access`

**Methods:**
- `size_bytes(&self) -> usize` — `prod(shape) * bytes_per_elem(dtype)`

---

### `DistributedTileRef`
*File: `ktir_cpu/ir_types.py:199–229`*

**Fields:**
- `partitions: Vec<TileRef>` — per-partition survivors of access intersection
- `shape: Vec<usize>` — global logical shape (inherited from DistributedMemRef)
- `dtype: &str` — all partitions must match
- `global_base: Option<Tuple<usize, ...>>` — origin of access tile in global coords; x = `base_map.eval(indices)`, set by `distributed_tile_access`

**Invariants:**
- At least one partition
- All partitions' dtype must match wrapper's dtype

---

### `Tile`
*File: `ktir_cpu/ir_types.py:232–283`*

**Fields:**
- `data: ndarray` — NumPy array holding element data
- `dtype: &str` — element type
- `shape: Vec<usize>` — tensor dimensions
- `unique_sticks: Option<usize>` — number of distinct HBM sticks touched by load; None for compute-produced tiles
- `index_unique_sticks: Option<usize>` — sticks touched by index-tensor reads in indirect load/store; None for direct loads

**Methods:**
- `copy(&self) -> Tile` — deep copy; propagates `unique_sticks` and `index_unique_sticks`
- `size_bytes(&self) -> usize` — `data.nbytes`
- `coalescing_efficiency(&self) -> Option<f64>` — `data.nbytes / (unique_sticks * STICK_BYTES)` when `unique_sticks` is set; None otherwise

**Invariants:**
- `unique_sticks` is set by load operations; None only for compute-produced tiles

---

### `AccessTile`
*File: `ktir_cpu/ir_types.py:286–303`*

**Fields:**
- `parent_ref: Union<TileRef, DistributedTileRef>` — single-allocation TileRef or distributed routed DistributedTileRef
- `shape: Vec<usize>` — access tile shape
- `base_map: AffineMap` — always present; synthesized as identity if absent in MLIR (line 138, 520–524)
- `coordinate_set: Option<Union<BoxSet, AffineSet>>` — parsed access_tile_set; None if omitted (line 301, 384–387)
- `coordinate_order: Option<AffineMap>` — parsed access_tile_order; None if omitted (line 302, 536–541)

**Special handling:**
- When `coordinate_set` is non-rectangular or non-concrete (symbolic), `construct_access_tile` raises `NotImplementedError` if parser did not surface `$symbol_operands` (line 152–158)
- When `coordinate_set` is axis-aligned and fully covers the rectangle, parse-time normalization sets it to None (line 533–534)
- When `coordinate_order` is the identity map, parse-time normalization sets it to None (line 540–541)

---

### `IndirectAccessTile`
*File: `ktir_cpu/ir_types.py:306–319`*

**Fields:**
- `parent_ref: MemRef` — primary memory view (e.g., X in gather/scatter)
- `shape: Vec<usize>` — output access tile shape
- `dim_subscripts: Vec<Dict<String, Any>>` — per-dimension descriptor; each has `kind` ∈ {`"direct"`, `"direct_expr"`, `"indirect"`}:
  - `"direct"`: `{"kind": "direct", "var_index": usize}` — reference intermediate variable by index
  - `"direct_expr"`: `{"kind": "direct_expr", "subscript": subscript_tuple}` — expression node (resolved SSA or const)
  - `"indirect"`: `{"kind": "indirect", "index_view_idx": usize, "idx_exprs": Vec<subscript_tuple>}` — gather via index tensor
- `index_views: Vec<MemRef>` — index memrefs for indirect dimensions
- `variables_space_set: AffineSet` — domain of intermediate variables; concrete (line 681)
- `variables_space_order: Option<AffineMap>` — iteration order; None = row-major default (line 709)

**Invariants:**
- All intermediate variables that resolve to outer SSA scalars must have zero-range dimensions in `variables_space_set` (line 694–701); violation raises `ValueError`
- `variables_space_set` is always concrete at construction time (no symbolic bounds)
- Subscript tuples are of form `("const", v) | ("dim", i) | ("ssa", "%name") | ("add"|"sub"|"mul"|"neg"|"floordiv"|"mod", ...)`; `"ssa"` nodes must be pre-resolved to `"const"` by `_resolve_node` before eval

---

### `AffineMap`
*File: `ktir_cpu/affine.py:85–150`*

**Fields:**
- `n_dims: usize` — number of input dimension variables (d0, d1, ...)
- `exprs: Vec<Node>` — AST nodes, one per output dimension
- `source: String` — original verbatim string (for debugging/round-trip)

**Methods:**
- `eval(dims: &[usize]) -> Vec<usize>` — evaluate output tuple for given input values
- `is_identity() -> bool` — True iff output[i] == d_i for all i (structural check on AST; used at parse time to normalize identity maps to None)
- `is_permutation() -> bool` — True iff map permutes input dims (square, bijective, each output is exactly one dim variable)

---

### `BoxSet`
*File: `ktir_cpu/affine.py`*

Specialization of `AffineSet` for axis-aligned sets with explicit (lo, hi) bounds per axis. Operations (contains, enumerate, intersect, translate, lower_bounds, is_empty, is_full) are O(ndim).

**Fields:**
- Per-axis lo/hi bounds; each bound is `Union<int, Bound>` (Bound is an AST node for symbolic expressions)
- `_all_concrete: bool` — optimization flag; True iff all bounds are integer constants

**Methods:**
- `is_concrete(&self) -> bool` — True iff all bounds are concrete (no symbolic nodes)
- `specialize(symbols: &[usize]) -> BoxSet` — resolve symbolic bounds by substituting symbol values
- Other: contains, enumerate, intersect, translate, is_empty, is_full, try_from_affine_set (parse-time lowering from AffineSet)

---

### `AffineSet`
*File: `ktir_cpu/affine.py`*

General affine integer set with constraint list (fallback for non-rectangular sets).

**Fields:**
- `n_dims: usize` — number of dimension variables
- `n_syms: usize` — number of symbol variables
- `constraints: Vec<Node>` — constraint AST nodes
- `source: String` — original string

**Methods:**
- `eval(dims: &[usize], syms: Option<&[usize]>) -> usize` — evaluate a constraint
- `enumerate(shape: &[usize], syms: Option<&[usize]>) -> Vec<Tuple<usize, ...>>` — enumerate all lattice points satisfying constraints within bounding box
- `contains(pt: &[usize], syms: Option<&[usize]>) -> bool` — check if point satisfies constraints
- `intersect(other: AffineSet, ...) -> AffineSet` — geometric intersection
- Other: is_concrete, specialize, is_empty, is_full

---

### `Operation`
*File: `ktir_cpu/ir_types.py:322–335`*

**Fields:**
- `result: Option<String>` — SSA result name (e.g., `"%x"`)
- `op_type: String` — operation type (e.g., `"ktdp.get_compute_tile_id"`)
- `operands: Vec<String>` — operand SSA names
- `attributes: Dict<String, Any>` — parsed op attributes
- `result_type: Option<String>` — result type string
- `regions: Vec<Vec<Operation>>` — control-flow regions (optional)

**Note:** Operands and attributes are populated by parsers; handlers resolve operands via `context.get_value(operand_name)`.

---

## Handler Functions

### `ktdp__get_compute_tile_id`
*File: `ktir_cpu/dialects/ktdp_ops.py:45–50`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> Union<usize, Tuple<usize, ...>>`

**Semantics:** Return grid coordinates of the current core in the given dimension(s).

**Implementation:**
```
num_dims = 1 if isinstance(op.result, str) else len(op.result)
if num_dims == 1:
    return GridOps.gridid(context, 0)
return tuple(GridOps.gridid(context, d) for d in range(num_dims))
```

**Contract:**
- When `op.result` is a single SSA name (string), return a single `usize` grid coordinate
- When `op.result` is a list of N names, return tuple of N grid coordinates (one per dimension)
- Always calls `GridOps.gridid(context, dim)` for each dimension

---

### `ktdp__coreid`
*File: `ktir_cpu/dialects/ktdp_ops.py:53–56`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> Vec<usize>`

**Semantics:** Return core IDs matching the given grid coordinates (use -1 as wildcard).

**Input:** Operands are grid coordinates [x, y, z]; -1 means "all cores in that dimension"

**Implementation:**
```
grid_coords = [context.get_value(operand) for operand in op.operands]
return GridOps.coreid(context, grid_coords, env.grid_executor)
```

**Contract:**
- Resolves operands to integer grid coordinates
- Delegates to `GridOps.coreid(context, grid_coords, grid_executor)`
- Returns list of matching core IDs

---

### `GridOps.gridid`
*File: `ktir_cpu/ops/grid_ops.py:30–40`*

**Signature:** `(context: CoreContext, dim: usize) -> usize`

**Semantics:** Return the grid coordinate of the current core in *dim*.

**Implementation:** `context.get_grid_id(dim)`

**Contract:**
- *dim* is 0=x, 1=y, 2=z
- In KTDP, always called with dim=0
- Returns index-typed value (usize in Rust; "index" in MLIR)

---

### `GridOps.coreid`
*File: `ktir_cpu/ops/grid_ops.py:43–62`*

**Signature:** `(context: CoreContext, grid_coords: Vec<usize>, grid_executor: GridExecutor) -> Vec<usize>`

**Semantics:** Return core IDs matching *grid_coords* (use -1 as wildcard).

**Implementation:**
```
Pad grid_coords to 3 dimensions if needed (append 0).
Call grid_executor.get_cores_in_group(tuple(grid_coords[:3]))
```

**Contract:**
- Pads grid_coords to [x, y, z] with trailing zeros if needed
- -1 means "all cores in that dimension"
- Delegates to `GridExecutor.get_cores_in_group((x, y, z))`

---

### `ktdp__construct_memory_view`
*File: `ktir_cpu/dialects/ktdp_ops.py:59–98`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> MemRef`

**Semantics:** Create a hardware-aware memory view (MemRef) from a pointer and shape/strides attributes.

**Attributes (required):**
- `shape: Tuple[Union[usize, str], ...]` — dimensions; strings are SSA operand names resolved at runtime
- `strides: Vec<Union<usize, str>>` — element-count strides; strings are SSA operand names
- `memory_space: str` — `"HBM"` or `"LX"`
- `dtype: str` — element type

**Attributes (optional):**
- `coordinate_set: Option<Union<BoxSet, AffineSet>>` — parsed coordinate set; resolved at runtime if symbolic
- `lx_core_id: Option<usize>` — per-core LX routing; only valid when memory_space="LX"

**Processing:**
1. Resolve pointer operand via `context.get_value(op.operands[0])`
2. Resolve shape: for each dimension, if SSA name string, call `context.get_value(name)` and cache the integer value
3. Resolve strides: same as shape
4. If `coordinate_set` is symbolic (contains bound AST nodes for symbolic dims), specialize it using shape values corresponding to dynamic `?` dims in the memref type (line 92–96)
5. Call `MemoryOps.tile_view(context, ptr, shape, strides, memory_space, dtype, coordinate_set, lx_core_id)`

**Operands:**
- `op.operands[0]` — pointer (base address as int)
- `op.operands[1..]` — SSA size operands, followed by SSA stride operands (from parser)

**Contract:**
- All dynamic dimension sizes must be provided as operands
- Symbol resolution is lazy: if coordinate_set has symbolic bounds, they are resolved before returning
- The parser pre-validates that sizes count matches memref dimension count

---

### `ktdp__construct_distributed_memory_view`
*File: `ktir_cpu/dialects/ktdp_ops.py:101–125`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> DistributedMemRef`

**Semantics:** Compose N per-partition MemRefs into a distributed view without allocating or moving data.

**Attributes (required):**
- `shape: Tuple[usize, ...]` — global logical shape
- `dtype: str` — element type

**Input:** Operands are N SSA names, each resolving to a MemRef (line 110)

**Processing:**
1. Resolve each operand to a MemRef via `context.get_value(name)` (line 110)
2. Validate each is a MemRef; raise ValueError if not (line 112–116)
3. Return `DistributedMemRef(partitions=partitions, shape=shape, dtype=dtype)`

**Contract:**
- Each partition must carry its own non-None `coordinate_set` (global coordinates of that partition)
- Dtype of all partitions must match the wrapper's dtype
- No data movement occurs; partition resolution happens at access time in `distributed_tile_access`

---

### `ktdp__construct_access_tile`
*File: `ktir_cpu/dialects/ktdp_ops.py:128–182`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> AccessTile`

**Semantics:** Create a coordinate access tile referencing a sub-region of a parent MemRef.

**Attributes (required):**
- `shape: Tuple[usize, ...]` — access tile shape
- `base_map: AffineMap` — always present; synthesized as identity if absent in MLIR

**Attributes (optional):**
- `coordinate_set: Option<Union<BoxSet, AffineSet>>` — parsed access_tile_set; normalized to None if it's the full rectangle
- `coordinate_order: Option<AffineMap>` — parsed access_tile_order; normalized to None if identity

**Input:**
- `op.operands[0]` — parent memref (MemRef or DistributedMemRef)
- `op.operands[1..]` — index operands (resolved to usize via context)

**Processing:**
1. Resolve parent_ref via `context.get_value(op.operands[0])` (line 130)
2. Resolve indices via `context.get_value(operand)` for each operand[1:] (line 131)
3. If parent_ref is DistributedMemRef (line 159):
   - Call `MemoryOps.distributed_tile_access(parent_ref, access_shape, base_map, indices, access_tile_set=coordinate_set)` to route the access (line 165–166)
   - Return AccessTile with the DistributedTileRef result
4. Otherwise, call `MemoryOps.tile_access(context, parent_ref, indices, access_shape, base_map)` (line 175)
5. Return AccessTile with TileRef and optional coordinate_set/coordinate_order

**Contract:**
- If coordinate_set is symbolic, raise NotImplementedError (parser does not surface $symbol_operands for binding symbols) (line 152–158)
- coordinate_set and coordinate_order are normalized to None at parse time if they are trivial (full rectangle / identity map)
- The base_map is always present; if absent in MLIR, synthesize identity map from the number of index operands (line 520–524)

---

### `ktdp__construct_indirect_access_tile`
*File: `ktir_cpu/dialects/ktdp_ops.py:562–710`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> IndirectAccessTile`

**Semantics:** Construct a gather/scatter access tile using intermediate variables and index tensors.

**Attributes (required):**
- `dim_subscripts: Vec<Dict[str, Any>>` — per-dimension subscript descriptor (see IndirectAccessTile type)
- `shape: Tuple[usize, ...]` — output tile shape
- `variables_space_set: AffineSet` — concrete domain of intermediate variables
- `intermediate_vars: Vec<str>` — intermediate variable names (without `%`)

**Attributes (optional):**
- `variables_space_order: Option<AffineMap>` — iteration order; None = row-major

**Input:**
- `op.operands[0]` — parent memory view
- `op.operands[1..]` — index view operands (for indirect dimensions)

**Processing:**
1. Resolve parent_ref via `context.get_value(op.operands[0])` (line 564)
2. Resolve index_views via `context.get_value(name)` for operands[1:] (line 565)
3. For each subscript in dim_subscripts, call `_resolve_node(subscript)` (line 569–636):
   - `("ssa", "%name")` leaf → resolve to `("const", value)` via `context.get_value`
   - `("dim", i)` leaf → check if intermediate_vars[i] is in context; if yes, resolve to `("const", value)` (backward-compat case (a)); otherwise leave as `("dim", i)` (pure iterator)
   - Compound nodes → recurse into children
   - **Key Python-ism:** This function mutates a shallow copy of each subscript dict and reassembles it (line 640–678)
4. Validate that intermediate variables resolving to SSA scalars have zero-range dimensions in variables_space_set (line 694–701)
5. Return IndirectAccessTile with resolved dim_subscripts

**Contract:**
- All `("ssa", ...)` nodes in dim_subscripts are resolved to `("const", ...)` before return
- Pure iteration variables remain as `("dim", i)` for eval at load time
- variables_space_set must be concrete (no symbolic bounds)
- Subscript nodes follow the grammar: `("const", v) | ("dim", i) | ("add"|"sub"|"mul"|"neg"|"floordiv"|"mod", ...)`
- **Backward-compat case (a)** (line 621–628): intermediate_vars[i] matching an outer SSA binding is resolved at construct time; questionable semantics (line 580–587); consider removing

---

### `ktdp__load`
*File: `ktir_cpu/dialects/ktdp_ops.py:185–204`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> Tile`

**Semantics:** Load a tile from memory, handling direct and indirect access patterns with optional coordinate ordering.

**Attributes (optional):**
- `_result_shape: Option<Tuple<usize, ...>>` — override access_tile.shape for result; used to reshape indirect loads

**Input:**
- `op.operands[0]` — access_tile (AccessTile or IndirectAccessTile)

**Processing:**
1. Resolve access_tile via `context.get_value(op.operands[0])` (line 187)
2. If access_tile is IndirectAccessTile (line 188):
   - Call `MemoryOps.indirect_load(context, access_tile, result_shape=result_shape)` (line 190)
3. Else if access_tile.parent_ref is DistributedTileRef (line 191):
   - Call `MemoryOps.distributed_load(context, access_tile.parent_ref, result_shape=result_shape)` (line 193–194)
4. Else (direct access, single-allocation):
   - If access_tile.coordinate_set is not None (line 198):
     - Enumerate coordinates via `coordinate_set.enumerate(access_tile.shape)` (line 199)
     - If coordinate_order is not None, apply it: `[coordinate_order.eval(pt) for pt in coords]` (line 201)
     - Call `MemoryOps.load(context, access_tile.parent_ref, coords=coords, result_shape=result_shape)` (line 203)
   - Else: call `MemoryOps.load(context, access_tile.parent_ref)` for contiguous fast path (line 204)

**Contract:**
- When coordinate_set is non-None, enumerate all points and optionally permute via coordinate_order
- When both coordinate_set and coordinate_order are None, use contiguous fast path
- Result is a Tile with numpy data and metadata (unique_sticks, index_unique_sticks)

---

### `ktdp__store`
*File: `ktir_cpu/dialects/ktdp_ops.py:207–231`*

**Signature:** `(op: Operation, context: CoreContext, env: ExecutionEnv) -> Union<None, int>`

**Semantics:** Store a tile to memory; no IR result, but handler returns HBM unique_sticks as latency sideband.

**Input:**
- `op.operands[0]` — value (Tile)
- `op.operands[1]` — access_tile (AccessTile or IndirectAccessTile)

**Processing:**
1. Resolve value via `context.get_value(op.operands[0])` and assert it's a Tile (line 216–217)
2. Resolve access_tile via `context.get_value(op.operands[1])` (line 218)
3. If access_tile is IndirectAccessTile (line 219):
   - Call `MemoryOps.indirect_store(context, value, access_tile)` (line 220)
4. Else if access_tile.parent_ref is DistributedTileRef (line 221):
   - Call `MemoryOps.distributed_store(context, value, access_tile.parent_ref)` (line 222)
5. Else (direct access):
   - If access_tile.coordinate_set is not None (line 224):
     - Enumerate coordinates via `coordinate_set.enumerate(access_tile.shape)` (line 227)
     - If coordinate_order is not None, apply it (line 228–229)
     - Call `MemoryOps.store(context, value, tile_ref, coords=coords)` (line 230)
   - Else: call `MemoryOps.store(context, value, tile_ref)` (line 231)

**Return value:** Integer (0 for LX, unique_sticks for HBM) used by latency tracker; line 209–214 explains the sideband mechanism.

**Contract:**
- Stores have no SSA result in IR, but handler returns int for latency accounting
- Mirrors load logic for coordinate enumeration and ordering
- MemoryOps.store() returns unique_sticks count (HBM) or 0 (LX)

---

## Parser Functions

### `parse_get_compute_tile_id`
*File: `ktir_cpu/dialects/ktdp_ops.py:254–275`*

**Signature:** `(op_text: str, parse_ctx: ParseContext) -> Option<Operation>`

**Pattern match:** `r"^(.*?)\s*=\s*ktdp\.get_compute_tile_id\s*:\s*([^{(]*)\s*$"`

**Processing:**
1. Extract LHS names via `parse_multi_result_lhs(m.group(1))` (handles bundled form `%g:2` → `["%g#0", "%g#1"]` and comma form `%x, %y` → `["%x", "%y"]`)
2. Parse type list from RHS (comma-separated)
3. Validate: name count == type count (line 269–273)
4. Return Operation with result as single name (if len==1) or list

**Contract:**
- Supports both bundled and comma form multi-result syntax
- Result types are not stored (type information is in op.result_type)
- Operands list is empty

---

### `parse_construct_memory_view`
*File: `ktir_cpu/dialects/ktdp_ops.py:278–405`*

**Pattern match:** `r'(%\w+)\s*=\s*ktdp\.construct_memory_view\s+(%\w+)'`

**Processing:**
1. Extract result name and pointer operand
2. Parse `sizes: [...]` (line 291–301):
   - Split on commas; try to parse as int, else store as SSA name string
   - Collect SSA names in ssa_size_operands
3. Parse `strides: [...]` (line 308–320):
   - Same as sizes; default strides=[1]
4. Parse memory space: regex match `#ktdp\.spyre_memory_space<\s*(\w+)(?:\s*,\s*core\s*=\s*(\d+))?\s*>` (line 328–335)
   - Extract memory_space (HBM or LX) and optional lx_core_id
5. Parse memref type from result: `r'(?:}\s*)?:\s*(?:index\s*->\s*)?memref<([^>]+)>'` (line 339–340)
   - Split on 'x'; last part is dtype, leading parts are dimensions
   - Parse '?' as None, integers as concrete dims
6. Validate sizes against memref dims (line 351–382):
   - Concrete memref dims must not conflict with provided sizes
   - Dynamic ('?') dims must have SSA size operands
7. Parse attribute block via `parse_attr_block(op_text, parse_ctx.aliases)` (line 384):
   - Extract coordinate_set string; parse via `parse_affine_set(str)` if present
8. Return Operation with attributes:
   - `shape: Tuple[Union[int, str], ...]` — static ints and SSA names (strings)
   - `strides: Vec<Union[int, str>]` — static ints and SSA names
   - `memory_space: str`
   - `dtype: str`
   - `coordinate_set: Option<Union<BoxSet, AffineSet>>` — parsed and lowered
   - `lx_core_id: Option<int>` — if set

**Operands:** `[ptr_operand] + ssa_size_operands + ssa_stride_operands` in that order (line 402)

**Contract:**
- Sizes and strides are lazily resolved at execution time if they are SSA names
- Coordinate set is parsed but may be symbolic; it's resolved at execution time via `specialize()`
- Parser validates shape count against memref dims but doesn't resolve SSA operands

---

### `parse_construct_distributed_memory_view`
*File: `ktir_cpu/dialects/ktdp_ops.py:408–479`*

**Pattern match:** `r'(%\w+)\s*=\s*ktdp\.construct_distributed_memory_view'`

**Processing:**
1. Extract result name
2. Find operand parenthesis; extract operand list via `_extract_bracket_content(op_text[paren_start:], '()')` (line 431–435)
3. Split operands/types section on first top-level ':' (line 437–446)
4. Extract operand names: filter for names starting with '%' via `split_top_level()` (line 448–450)
5. Parse result memref type: `r'(?:}\s*)?:\s*memref<([^>]+)>\s*$'` (line 457–471)
   - Extract shape (all dims must be concrete integers) and dtype
6. Return Operation with attributes:
   - `shape: Tuple[int, ...]`
   - `dtype: str`

**Operands:** list of memref SSA names (line 476)

**Contract:**
- All shape dimensions must be concrete integers (no '?' allowed in result type)
- Each operand must resolve at execution time to a MemRef with a coordinate_set

---

### `parse_construct_access_tile`
*File: `ktir_cpu/dialects/ktdp_ops.py:482–555`*

**Pattern match:** `r'(%\w+)\s*=\s*ktdp\.construct_access_tile\s+'`

**Processing:**
1. Extract result name
2. Extract operands via `find_ssa_names(after_eq)` after `=` (line 491)
3. Parse access tile shape from result type: `r'!ktdp\.access_tile<([^>]+)>'` (line 493–512)
   - Match `NxMx...xindex` pattern; element type must be "index"
4. Parse attribute block via `parse_attr_block(op_text, parse_ctx.aliases)` (line 516)
5. Parse base_map:
   - Extract `base_map` string from attrs; if not present, synthesize identity map from number of index operands (line 520–524)
   - Parse via `parse_affine_map(base_map_str)` (line 525)
6. Parse coordinate_set:
   - Extract `access_tile_set` string; parse via `parse_affine_set()` (line 527–528)
   - Normalize to None if set is full (covers entire rectangle) (line 533–534)
7. Parse coordinate_order:
   - Extract `access_tile_order` string; parse via `parse_affine_map()` (line 536–537)
   - Normalize to None if map is identity (line 540–541)
8. Return Operation with attributes:
   - `shape: Tuple[int, ...]`
   - `base_map: AffineMap`
   - `coordinate_set: Option<Union<BoxSet, AffineSet>>`
   - `coordinate_order: Option<AffineMap>`

**Operands:** list of SSA names (memref + indices)

**Contract:**
- If base_map is not present, synthesize identity map with n=max(1, len(operands)-1) inputs (line 520–524)
- coordinate_set and coordinate_order are normalized to None for trivial cases at parse time

---

### `parse_construct_indirect_access_tile`
*File: `ktir_cpu/dialects/ktdp_ops.py:713–823`*

**Pattern match:** `r'(%\w+)\s*=\s*ktdp\.construct_indirect_access_tile\s+'`

**Processing:**
1. Extract result name
2. Parse intermediate variables: regex match `r'intermediate_variables\s*\(([^)]+)\)'` (line 725–728)
   - Strip '%' from each variable name
3. Find primary operand: match `r'\s*(%\w+)\[` after intermediate_variables block (line 734–737)
4. Extract subscript content via `_extract_bracket_content(op_text[bracket_start:], '[]')` (line 740–743)
5. For each dimension (split on top-level commas):
   - If starts with `ind(`: indirect subscript (line 752–767)
     - Match `r'(%\w+)\[([^\]]*)\]'` inside ind(...)
     - Parse each variable reference via `parse_subscript_expr(v, intermediate_vars)` (line 760)
     - Append to operands
   - Else: direct subscript (line 768–781)
     - Strip parentheses and check if bare name is in intermediate_vars
     - If yes: `{"kind": "direct", "var_index": idx}` (line 773–776)
     - Else: parse as expression via `parse_subscript_expr(inner, intermediate_vars)` and wrap as `{"kind": "direct_expr", "subscript": expr}` (line 778–781)
6. Parse attribute block (line 784)
7. Parse variables_space_set: required; raise if not present (line 786–789)
8. Parse variables_space_order: optional; normalize to None if identity (line 791–794)
9. Parse access tile shape from result type: `r'!ktdp\.access_tile<([^>]+)>'` (line 797–806)
10. Return Operation with attributes:
    - `shape: Tuple[int, ...]`
    - `dim_subscripts: Vec<Dict[str, Any>>`
    - `intermediate_vars: Vec[str]`
    - `variables_space_set: AffineSet`
    - `variables_space_order: Option[AffineMap]`

**Operands:** `[primary_operand] + index_views` in order of appearance

**Contract:**
- intermediate_vars are bare names (without '%')
- Each indirect dimension consumes one index_view operand
- subscript_expr nodes use `("const" | "dim" | "ssa" | "add" | "sub" | "mul" | "neg" | "floordiv" | "mod", ...)` tuple representation
- variables_space_set is required and concrete
- variables_space_order is optional; normalized to None if identity

---

## Parser Utilities

### `parse_subscript_expr`
*File: `ktir_cpu/dialects/ktdp_helpers.py:106–138`*

**Signature:** `(token: str, var_names: Vec<str>) -> tuple`

**Semantics:** Parse one subscript expression into an AST tuple.

**Output:**
- `("const", int)` — integer literal
- `("dim", i)` — reference to var_names[i] (iteration variable)
- `("ssa", "%name")` — outer SSA scalar (resolved to const at construct time)
- Compound: `("add" | "sub" | "mul" | "neg" | "floordiv" | "mod", ...)`

**Special cases:**
- Legacy fast-path for `%name floordiv N` and `%name mod N` (line 129–134)
- General case: parse via affine expression parser, then classify refs (line 137–138)

**Contract:**
- var_names are bare names without '%'
- SSA refs (e.g., `%pid1`) are classified as `("ssa", "%pid1")` for later resolution
- Iteration variables are classified as `("dim", i)` to be evaluated at load time

---

### `parse_affine_map`
*File: `ktir_cpu/parser_ast.py:345–368`*

**Signature:** `(s: str) -> AffineMap`

**Semantics:** Parse `affine_map<(d0,...) -> (e0,...)>` (wrapper optional).

**Processing:**
1. Strip outer `affine_map<...>` wrapper (optional)
2. Tokenize and parse dimension names
3. Build dim_index map for ref resolution
4. Parse `->` separator
5. Parse output expression list
6. Return AffineMap with n_dims, exprs tuple, and source string

**Contract:**
- Output expressions are AST nodes with structure from parser_ast._Node type
- dim_index map is built from dimension names (includes non-canonical names like "i", "row", etc.)

---

### `parse_affine_set`
*File: `ktir_cpu/parser_ast.py:411–428`*

**Signature:** `(s: str) -> Union<BoxSet, AffineSet>`

**Semantics:** Parse `affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>` with parse-time lowering.

**Processing:**
1. Call `parse_affine_set_raw(s)` to get AffineSet
2. Try lowering via `BoxSet.try_from_affine_set(aset)` (line 427)
3. Return BoxSet if lowering succeeds, else AffineSet

**Contract:**
- Axis-aligned, fully-pinned, unit-coefficient, concrete (non-symbolic) sets lower to BoxSet (O(ndim) ops)
- Symbolic sets stay as AffineSet (even if axis-aligned; TODO to fix per line 420–421)
- Fallback AffineSet is always valid and slower but more general

---

### `parse_attr_block`
*File: `ktir_cpu/parser_utils.py`* (used by all construct_* parsers)

**Semantics:** Extract and parse an MLIR attribute block `{...}` and resolve aliases.

**Return:** Dict<str, Any> with attribute values (strings, ints, etc.); affine types are kept as strings for downstream parsing.

---

## Execution Environment

### `CoreContext`
*File: `ktir_cpu/grid.py:41–289`*

**Fields:**
- `core_id: usize` — linear core ID
- `grid_pos: (usize, usize, usize)` — (x, y, z) position
- `lx: LXScratchpad` — per-core local SRAM (2 MB capacity)
- `hbm: HBMSimulator` — shared HBM
- `_scope_stack: Vec<Dict<str, Any>>` — SSA value scopes (nested for control flow)
- `_lx_bytes: Dict<str, usize>` — SSA name → LX bytes (single source of truth)

**Methods:**
- `get_value(name: &str) -> Any` — search scope stack top-to-bottom for SSA value
- `set_value(name: &str, value: Any)` — bind SSA value in topmost scope
- `get_grid_id(dim: usize) -> usize` — return grid_pos[dim]
- `push_scope()` / `pop_scope()` — manage nested control-flow regions
- `track_lx(name: &str, size_bytes: usize)` / `untrack_lx(name: &str)` — update lx.used

---

### `GridExecutor`
*File: `ktir_cpu/grid.py:358–446`*

**Fields:**
- `grid_shape: (usize, usize, usize)` — (nx, ny, nz) grid dimensions
- `memory: SpyreMemoryHierarchy` — shared memory (HBM + per-core LX)
- `cores: Vec<CoreContext>` — per-core execution contexts (one per linear ID)

**Methods:**
- `_linear_to_grid(core_id: usize) -> (usize, usize, usize)` — convert ID to (x, y, z)
- `_grid_to_linear(x, y, z) -> usize` — convert (x, y, z) to linear ID
- `get_cores_in_group(grid_coords: (int, int, int)) -> Vec<usize>` — return core IDs matching wildcard coords (line 416–446)
  - -1 in a position means "all cores in that dimension"
  - Example: (-1, 2, 0) returns all cores at y=2, z=0

---

## Key Python-isms and Redesign Notes

### 1. **Subscript Expression AST — tuple representation**
Python uses tuples like `("const", 5)`, `("dim", 0)`, `("ssa", "%x")`, `("add", left, right)` to represent AST nodes. Rust will need an enum-based AST:
```
enum SubscriptExpr {
    Const(i32),
    Dim(usize),
    Ssa(String),  // "%name"
    Add(Box<SubscriptExpr>, Box<SubscriptExpr>),
    Sub(Box<SubscriptExpr>, Box<SubscriptExpr>),
    Mul(i32, Box<SubscriptExpr>),  // coefficient * expr
    Neg(Box<SubscriptExpr>),
    Floordiv(usize, i32),  // dim_index floordiv modulus
    Mod(usize, i32),       // dim_index mod modulus
}
```

### 2. **Mutual Resolution of SSA and Iteration Variables**
In `construct_indirect_access_tile`, the `_resolve_node` function (line 569–636) resolves SSA refs to constants but leaves iteration variables as `("dim", i)`. This is a two-phase process:
- **Phase 1 (construct):** resolve all SSA operands to concrete values; keep pure iteration variables symbolic
- **Phase 2 (load):** enumerate iteration variables across the variables_space_set and evaluate subscript expressions for each point

Rust should:
1. Resolve SSA nodes immediately in the handler to SubscriptExpr::Const
2. Keep SubscriptExpr::Dim nodes for load-time evaluation
3. Thread subscript_tuple through to memory ops without eager evaluation

### 3. **Dict[str, Any] for dim_subscripts**
Each subscript is a heterogeneous dict with `kind` discriminator and kind-specific fields:
```python
{"kind": "direct", "var_index": 0}
{"kind": "direct_expr", "subscript": ("const", 5)}
{"kind": "indirect", "index_view_idx": 0, "idx_exprs": [("dim", 0), ("add", ("const", 1), ("dim", 1))]}
```

Rust should define an enum:
```
enum DimSubscript {
    Direct { var_index: usize },
    DirectExpr { subscript: SubscriptExpr },
    Indirect { index_view_idx: usize, idx_exprs: Vec<SubscriptExpr> },
}
```

### 4. **Lazy SSA Operand Resolution**
In `construct_memory_view`, operand[1:] are SSA names that are not resolved at parse time; they are collected as strings and resolved at execution time via `context.get_value()`. This defers the binding until the core reaches the operation.

Rust should treat operands as Vec<String> and resolve on-demand in the handler; the parser can validate syntax but not value availability.

### 5. **Parse-time Normalization**
Several attributes are normalized to `None` at parse time if they represent trivial cases:
- `coordinate_set` → None if the set covers the full rectangle (line 533–534)
- `coordinate_order` → None if the map is identity (line 540–541)
- `variables_space_order` → None if identity (line 793–794)

Rust should apply the same normalization to avoid runtime checks.

### 6. **Symbolic Bounds and Runtime Specialization**
AffineSet and BoxSet can carry symbolic bounds (AST nodes for symbol-dependent expressions). At construct_memory_view time, if coordinate_set has symbolic bounds, they are resolved via `specialize(symbols)` where symbols come from shape operands corresponding to dynamic ('?') memref dims (line 92–96).

Rust should:
1. Parse symbolic bounds as AST nodes in AffineSet/BoxSet
2. Implement `specialize(&self, symbols: &[usize]) -> Self` to substitute and simplify
3. Return a concrete (non-symbolic) set after specialization

### 7. **Generator-based Comm Ops (not applicable to ktdp.*)**
The Python codebase uses Python generators (yield) for blocking recv in communication ops. KTDP ops don't use generators, so this is not a concern for this subsystem, but the broader codebase will need an async/continuation mechanism in Rust (e.g., MaybeReceive enum or explicit callback).

### 8. **Operand Order and Collection**
Operands for construct_memory_view are collected as `[ptr] + ssa_size_operands + ssa_stride_operands` (line 402). This ordering is critical because the handler reconstructs the lists by filtering for strings vs. ints. Rust should enforce this at parse time and document the contract clearly.

### 9. **Backward-compat Intermediate Variable Resolution**
Line 621–628 handles a legacy case where intermediate_vars[i] can be a bound SSA operand (resolves to a constant). This complicates the semantics: a variable that resolves to a constant doesn't truly vary. The code validates this (line 694–701) but suggests removing this case and requiring explicit offset syntax (line 699–700). Rust port should preserve the logic but flag this for future simplification.

### 10. **Multi-result Bundled Form**
The parser handles both comma form (`%x, %y = ...`) and bundled form (`%g:2 = ...` → synthesized `%g#0`, `%g#1`). Rust should use a separate enum or tuple variant to represent multi-result LHS and synthesize names at parse time.

---

## Constant Formulas

- **HBM stick size:** `STICK_BYTES = 128` (implicit in split_addr, byte_address conversions)
- **Grid linear-to-grid conversion:** `z = core_id // (nx*ny)`, `y = (core_id % (nx*ny)) // nx`, `x = core_id % nx`
- **Access tile shape validation:** `coordinate_set.is_full(shape)` checks if set covers the entire rectangular region
- **Coalescing efficiency:** `data.nbytes / (unique_sticks * STICK_BYTES)`

---

## Critical Implementation Dependencies

1. **AffineMap/AffineSet parsing and evaluation** — parser_ast provides AST evaluators; Rust must replicate `parse_affine_map`, `parse_affine_set`, `eval(dims)`, `enumerate(shape)`, `contains(pt)`, `is_identity()`, `is_permutation()`, `specialize(symbols)`, `try_from_affine_set`.

2. **BoxSet specialization** — `specialize(symbols)` for runtime binding of symbolic bounds; this is called at construct_memory_view time (line 96).

3. **GridExecutor.get_cores_in_group(grid_coords)** — wildcard matching with -1; invoked by coreid handler.

4. **MemoryOps module** — (external to ktdp_ops but heavily used)
   - `tile_view(context, ptr, shape, strides, memory_space, dtype, coordinate_set, lx_core_id) -> MemRef`
   - `tile_access(context, parent_ref, indices, access_shape, base_map) -> TileRef`
   - `distributed_tile_access(parent_ref, access_shape, base_map, indices, access_tile_set) -> DistributedTileRef`
   - `load(..., coords, result_shape) -> Tile`, `indirect_load(...)`, `distributed_load(...)`
   - `store(..., coords) -> int`, `indirect_store(...)`, `distributed_store(...)`

5. **CoreContext.get_value()** — SSA scope lookup; must search scope stack top-to-bottom.

---

## Ownership and Mutation

- **MemRef and DistributedMemRef:** immutable after construction; no mutation.
- **AccessTile:** immutable.
- **IndirectAccessTile:** immutable.
- **dim_subscripts:** each dict is shallow-copied in the handler and mutated in-place to resolve subscripts; the original is not modified (Python: `sub = dict(sub)` on line 640).
- **CoreContext._scope_stack:** mutable; handlers call `set_value()` and `get_value()` to interact with the topmost scope.
- **Tile.data:** owned by Tile; moved or copied depending on semantics (e.g., `copy()` creates a deep copy).

---

## Test Entry Points

- **ktdp.get_compute_tile_id parser:** 254–275 (single and multi-result forms)
- **construct_memory_view parser:** 278–405 (shapes, strides, memory_space, coordinate_set)
- **construct_distributed_memory_view parser:** 408–479
- **construct_access_tile parser:** 482–555 (affine attributes, normalization)
- **construct_indirect_access_tile parser and handler:** 713–823 and 562–710 (subscript resolution, validation)
- **load/store handlers:** 185–231 (coordinate enumeration, distributed/indirect dispatch)

---

Now I have all the context needed. Let me create a comprehensive markdown spec for the Rust engineer.

# parser-internals

## Overview

The parser internals implement recursive-descent parsing for MLIR affine expressions and attribute blocks. The core subsystem transforms source text into AST nodes (tagged tuples), evaluates them against concrete or symbolic values, and lowers axis-aligned integer sets to optimized `BoxSet` form. All parsing is immutable; evaluation delegates to recursive walkers.

---

## AST Node Representation

**Node format:** Plain tuple (hashable, immutable) with a tag string in position 0. Used to represent affine expressions and constraints.

- `("const", int)` — integer constant
- `("dim", int)` — dimension variable d_N; int is positional index
- `("sym", int)` — symbol variable s_N; int is positional index
- `("ref", str)` — named reference; str is raw token (e.g. "%grid0", "d0") — resolves to domain-specific semantics at call site (parser_ast.py:231-234)
- `("add", node, node)` — addition
- `("sub", node, node)` — subtraction (normalized from both `lhs >= rhs` and `lhs <= rhs`)
- `("neg", node)` — unary negation
- `("mul", int, node)` — constant-coefficient multiplication; int is coefficient
- `("max", node, node)` — pointwise max (constructed by `sym_max`, not surface parser; used in `BoxSet.lo` after intersect)
- `("min", node, node)` — pointwise min (constructed by `sym_min`, not surface parser; used in `BoxSet.hi` after intersect)
- `("eq", node, node)` — equality constraint (first-class 3-tuple for `parse_constraint_list`)

**Invariant:** `("ref", ...)` nodes pass through the parser untouched; resolution to `("dim", ...)` or `("sym", ...)` happens only after `dim_index` / `sym_index` population.

---

## Tokenisation

**Module:** parser_ast.py:93–114

`_tokenise(text: str) → List[str]`

Regex-driven tokenizer producing flat token stream. Regex (line 93–100):
```
r'(%[a-zA-Z_]\w*)'                    # %identifier (group 1)
r'|([a-zA-Z_]\w*)'                    # bare identifier (group 2)
r'|(-?\d+)'                           # integer, possibly negative (group 3)
r'|(==|>=|<=|->|[+\-*(),:[\]])'      # operators; == before >= (group 4)
```

**Tokens produced:**
- `%name` — SSA references
- `d0`, `s0`, bare identifiers — variable names
- `-123`, `456` — integer literals
- `(`, `)`, `[`, `]`, `{`, `}` — brackets
- `+`, `-`, `*` — arithmetic ops
- `==`, `>=`, `<=` — constraint operators
- `->` — affine_map result arrow
- `,`, `:` — separators

**Non-tokens:** whitespace (skipped). Handles negative integer literals directly in regex (group 3).

---

## Recursive-Descent Parser: `_Parser` Class

**Module:** parser_ast.py:129–314

Stateful parser holding token stream, position cursor, and name-to-index maps.

### Fields
- `tokens: List[str]` — flattened token stream
- `pos: int` — current cursor position
- `dim_index: dict` — maps dim name (e.g. "i", "d0", "row") → positional index; populated by caller after parsing dim list (line 361, 399)
- `sym_index: dict` — maps symbol name (e.g. "s0", "n") → positional index; populated by caller after parsing symbol list (line 400)

### Core Methods

**`peek() → Optional[str]`** (line 143)
Returns token at current position without advancing; `None` if at end.

**`consume(expected: Optional[str] = None) → str`** (line 146)
Advance to next token and return current. Raises `ValueError` if `expected` and actual differ.

**`parse_dim_list() → List[str]`** (line 155)
Parse `(d0, d1, ...)` → return name list. Does NOT set `dim_index`; caller does.

**`parse_sym_list() → List[str]`** (line 166)
Parse optional `[s0, s1, ...]` → return name list. Returns `[]` if no `[...]` prefix. Does NOT set `sym_index`.

**`parse_expr() → _Node`** (line 179)
Entry point for expression parsing. Delegates to `_additive()`.

**`_additive() → _Node`** (line 182)
Left-associative addition/subtraction. Produces `("add", ..., ...)` / `("sub", ..., ...)` chains.

**`_term() → _Node`** (line 190)
Handles unary minus and multiplication. Both `N * expr` and `expr * N` syntax.
- Unary `-expr` → `("neg", expr)`
- `N * expr` (int coeff before `*`) → `("mul", N, expr)`
- `expr * N` (int coeff after) → `("mul", N, expr)`
- Bare int → `("const", N)`

**`_atom() → _Node`** (line 218)
Base expression unit. Produces:
- `(expr)` — parenthesised sub-expr
- `%name` → `("ref", "%name")` (line 234)
- Dimension variable (in `dim_index`) → `("dim", idx)` (line 244)
- Fallback canonical `d\d+` (when `dim_index` empty) → `("dim", N)` by parsing numeric suffix (line 250)
- Symbol variable (in `sym_index`) → `("sym", idx)` (line 264)
- Fallback canonical `s\d+` (when `sym_index` empty) → `("sym", N)` by parsing suffix (line 265)
- Positive integer → `("const", N)` (line 271)

**Fallback semantics:** When called via `parse_expr()` directly (no surrounding `affine_map`/`affine_set`), `dim_index`/`sym_index` are empty, so the parser accepts canonical `d0`, `d1`, `s0`, `s1` forms by numeric suffix extraction (lines 250–267).

**`parse_expr_list() → List[_Node]`** (line 275)
Parse `(e0, e1, ...)` → return expression list.

**`parse_constraint_list() → List[_Node]`** (line 286)
Parse `(lhs >= rhs, lhs <= rhs, lhs == rhs, ...)` → return constraint nodes.

**Constraint normalization (lines 289–307):**
- `lhs >= rhs` → stored as `("sub", lhs, rhs)` (ready for `>= 0` check)
- `lhs <= rhs` → stored as `("sub", rhs, lhs)` (flip operands)
- `lhs == rhs` → stored as `("eq", lhs, rhs)` (first-class tuple)

All inequality constraints internally represent `lhs - rhs >= 0` form.

---

## High-Level Parse Functions

### `parse_affine_map(s: str) → AffineMap`
**Module:** parser_ast.py:345–368

Parse `affine_map<(d0,...) -> (e0,...)>` into frozen `AffineMap` dataclass.

**Steps:**
1. Strip outer `affine_map<...>` wrapper via `_strip_outer` (line 355) — wrapper optional
2. Tokenise inner text (line 356)
3. Parse dim list, build `dim_index` map (lines 358–361)
4. Consume `->` token (line 362)
5. Parse output expression list (line 363)
6. Return `AffineMap(n_dims, exprs, source)` (line 364)

**Return type:**
```rust
struct AffineMap {
    n_dims: usize,
    exprs: Vec<Node>,  // or Tuple<_>
    source: String,    // original text for debugging
}
```

**Errors:** `ValueError` on tokenisation, parse, or missing `->`.

### `parse_affine_set(s: str) → AffineSet | BoxSet`
**Module:** parser_ast.py:411–428

Parse `affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>` with **parse-time lowering to BoxSet** when the set is axis-aligned, fully pinned on every axis, and has no symbols.

**Steps:**
1. Call `parse_affine_set_raw` to get raw `AffineSet` (line 426)
2. Attempt lowering via `BoxSet.try_from_affine_set(aset)` (line 427)
3. Return `BoxSet` if lowering succeeds; else return `AffineSet` (line 428)

**Lowering condition (affine.py:440–462):** An `AffineSet` lowers to `BoxSet` iff:
- Every constraint has form `c * d_i + k(syms) >= 0` or `c * d_i + k(syms) == 0` with `c ∈ {+1, -1}`
- Exactly one dimension variable per constraint (unit coefficient)
- Every axis pinned on **both** sides (has explicit lo and hi)
- `k(syms)` is either `int` constant or linear combination of symbols (no dim × sym products)
- Symbolic sets (n_syms > 0) stay on `AffineSet` branch (line 421 TODO comment)

### `parse_affine_set_raw(s: str) → AffineSet`
**Module:** parser_ast.py:371–408

Parse `affine_set<...>` **without** lowering to `BoxSet`. Returns raw `AffineSet` with constraint AST intact.

**Steps:**
1. Strip outer `affine_set<...>` wrapper (line 385)
2. Split on `:` to get dim part and constraint part (lines 386–388)
3. Parse dim list from `(d0, ...)` and optional symbol list from `[s0, ...]` (lines 391–394)
4. Build `dim_index` and `sym_index` maps (lines 399–400)
5. Parse constraint list (lines 401)
6. Return `AffineSet(n_dims, n_syms, constraints, source)` (line 403)

**Colon requirement:** The `:` character is **required** to separate dim/symbol declaration from constraints (line 386 uses `index(":")`). If absent, this raises `ValueError`.

---

## Evaluation & Membership

### `_eval_node(node: _Node, dims: List[int], syms: Optional[List[int]] = None) → int`
**Module:** parser_ast.py:435–456

Recursively evaluate AST node given concrete dimension and symbol values.

**Dispatch by tag:**
- `"const"` → return const value
- `"dim"` → return `dims[index]`
- `"sym"` → return `syms[index]` if `syms` provided, else `[index]` on empty list (undefined)
- `"add"` → recurse both operands and sum
- `"sub"` → recurse and subtract
- `"neg"` → negate operand
- `"mul"` → multiply coefficient by operand
- `"max"`, `"min"` → pointwise max/min of operands
- Else → raise `ValueError("Unknown AST node tag: ...")`

**Ownership:** `dims` and `syms` are passed by value (copied to list); no mutation.

### `eval_affine_map(amap: AffineMap, dims: Sequence[int]) → Tuple[int, ...]`
**Module:** parser_ast.py:459–477

Evaluate affine map. Validates `len(dims) == amap.n_dims`, then evaluates each output expression.

**Returns:** Tuple of output integers, one per expression.

**Errors:** `ValueError` if dimension count mismatch.

### `affine_set_contains(aset: AffineSet, point: Sequence[int], symbols: Sequence[int] = ()) → bool`
**Module:** parser_ast.py:480–488

Check membership. For each constraint:
- If `("eq", lhs, rhs)` → require `eval(lhs) == eval(rhs)`
- Else → require `eval(constraint) >= 0` (constraint is normalized `lhs - rhs`)

**Returns:** `True` iff all constraints satisfied.

### `enumerate_affine_set(aset: AffineSet, shape: Tuple[int, ...], symbols: Sequence[int] = ()) → List[Tuple[int, ...]]`
**Module:** parser_ast.py:491–510

Brute-force enumeration of integer points in `[0, shape)` satisfying all constraints.

**Algorithm:**
1. Validate `len(shape) == aset.n_dims`
2. Cartesian product of ranges: `itertools.product(*ranges)` where `ranges = [range(s) for s in shape]`
3. Filter by membership test `affine_set_contains(aset, pt, symbols)` (line 510)
4. Return row-major ordered list

**Complexity:** O(∏shape × n_constraints × depth_per_constraint).

### `enumerate_membership_keys(family: AffineSet, domain: AffineSet, point: Sequence[int], bound: int) → List[int]`
**Module:** parser_ast.py:513–553

Higher-level query: return keys `k ∈ domain ∩ [0, bound)` for which `point` is in `family(k)`.

**Treats `family` as parameterised:** `family` has at least one symbol; the key is bound to symbol slot 0.

**Algorithm:**
1. Enumerate keys from `domain.enumerate((bound,))`
2. For each key, test `family.contains(point, [key])`
3. Collect keys where membership holds

**Use case:** Memory subsystem queries which partition family members contain a given access point.

---

## Symbolic Bound Helpers

**Module:** parser_ast.py:557–656

A `Bound` is either a plain `int` (concrete leaf, fast path) or an AST node tuple over symbol variables only (no `dim` nodes). Bounds in `BoxSet` use this union type. Concrete ints stay unwrapped for `isinstance(b, int)` fast paths.

**Type alias:** `Bound = Union[int, tuple]`

### `eval_bound(b, symbols: Sequence[int]) → int`
**Module:** parser_ast.py:577–587

Evaluate a `Bound` against concrete symbols.

- If `int` → return unchanged (no AST walk)
- Else → delegate to `_eval_node(b, dims=[], syms=list(symbols))`

Concrete bounds short-circuit; symbolic bounds walk the AST.

### `sym_add(a, b) → Bound`
**Module:** parser_ast.py:590–604

Build `a + b` with constant folding.

**Folds when:**
- Both operands are `int` → return sum
- `a == 0` and `int` → return `b`
- `b == 0` and `int` → return `a`

**Else:** construct `("add", a_node, b_node)` where nodes wrap plain ints in `("const", ...)`

**Invariant:** If both operands are concrete, result is concrete (no AST node).

### `sym_neg(a) → Bound`
**Module:** parser_ast.py:607–615

Build `-a` with constant folding and double-negation collapse.

- If `int` → return negated value
- If `("neg", x)` → return `x` (double-negation cancels)
- Else → return `("neg", a)`

### `sym_max(a, b) → Bound`
**Module:** parser_ast.py:618–637

Build `max(a, b)` with MVP (Minimum Viable Product) folding.

**Folds when:**
- Both operands are `int` → return max
- Both are `("sym", k)` with same index → return one copy (idempotent)

**No deep canonicalisation** (no commutativity rewriting, no nested absorption). Per-axis candidate count is ≤ 2 in practice, so deep nesting does not arise.

### `sym_min(a, b) → Bound`
**Module:** parser_ast.py:640–655

Mirror of `sym_max` with identical folding rules.

---

## Parsing Attribute Values

**Module:** parser_utils.py:167–405

Attribute parsing extracts key=value pairs from MLIR operation blocks, handling nested brackets and alias resolution.

### `parse_attr_block(op_text: str, aliases: Optional[Dict] = None, brackets: str = '{}') → Dict`
**Module:** parser_utils.py:167–225

Extract attribute block from outer-most bracketed pair (default `{...}`; pass `brackets='[]'` for `[...]`).

**Grammar handled:**
- `keyword<...>` values: e.g. `affine_map<...>`, `affine_set<...>`, `#ktdp.spyre_memory_space<HBM>` — bracket depth counted while skipping `>=` and `->` (lines 177–179 semantics)
- `#alias` references: resolved via optional `aliases` dict
- Plain tokens, integers, floats, `[...]` lists

**Algorithm:**
1. Extract bracket content via `_extract_bracket_content(op_text, brackets)` (line 192)
2. Loop through block (line 198):
   - Skip whitespace/commas (line 200)
   - Match key with regex `r'[\w.]+'` (line 206)
   - Match optional `=` (line 214)
   - Extract value via `_extract_attr_value` (line 220)
   - Coerce value via `_coerce_attr_value` (line 223)
3. Return `{key: value}` dict

**Bracket depth tracking (lines 334–341):**
When parsing `keyword<...>`:
- Increment depth at `<`
- Decrement at `>`
- Skip `>=` (single token) — not a closing bracket
- Skip `->` (single token) — not a bracket pair

**Return type:** `Dict[str, Any]` where values are coerced to Python scalars (int/float/list/str).

**Errors:** Malformed input skipped; missing block returns `{}`.

### `_extract_attr_value(text: str, aliases: Optional[Dict]) → tuple`
**Module:** parser_utils.py:308–355

Extract one attribute value from start of text. Returns `(value_str, chars_consumed)`.

**Handles:**
- `#alias` reference (lines 320–325): consume until `,` or `}`, resolve via `aliases` if provided
- `keyword<...>` values (lines 327–349): count angle-bracket depth, skip `>=` and `->` as per `parse_attr_block` logic
- Plain tokens (lines 351–355): consume until `,` or `}`

**Bracket depth state machine (lines 332–348):**
```
if ch == '>' and next == '=' → skip both (not closing bracket)
else if ch == '-' and next == '>' → skip both (not bracket pair)
else if ch == '<' → depth += 1
else if ch == '>' → depth -= 1; if depth == 0: return matched string
```

### `extract_named_attr(op_text: str, key: str, aliases: Optional[Dict] = None) → Optional[str]`
**Module:** parser_utils.py:257–305

Extract single `key = value` attribute from op text (for attributes outside `{...}` blocks).

**Returns:** Resolved value string or `None` if key not found.

**Algorithm:**
1. Search for `\bkey\s*=\s*` (word boundary, line 270)
2. Parse value from position after `=`:
   - `#alias` → resolve via `aliases` (lines 276–279)
   - `keyword<...>` → count brackets, skip `>=` and `->` (lines 282–301)
   - Plain token → consume to comma/newline/brace/colon/arrow (lines 304–305)

### `_extract_bracket_content(op_text: str, brackets: str = '{}') → Optional[str]`
**Module:** parser_utils.py:146–164

Return content inside outermost matched bracket pair. Handles nested brackets of same kind.

**Algorithm:**
1. Find first open bracket (line 153)
2. Track depth, scanning until depth returns to 0 (lines 156–163)
3. Return content or `None` if no matching close bracket

### `_coerce_attr_value(value_str: str) → Any`
**Module:** parser_utils.py:387–405

Coerce raw attribute value string to Python scalar.

**Steps:**
1. Strip MLIR type annotation suffix (e.g. `"0 : i32"` → `"0"`) via regex `r'\s*:\s*\S+$'` (line 390)
2. Try parse as `int` (line 392)
3. Try parse as `float` (line 396)
4. Try parse as list: `[e0, e1, ...]` → list of ints (lines 399–403)
5. Fallback → return as string

**Returns:** `int`, `float`, `list[int]`, or `str`.

---

## Tensor Type Parsing

### `parse_tensor_type(type_str: str) → Optional[Dict]`
**Module:** parser_utils.py:53–84

Parse tensor type string to shape and dtype.

**Grammar:**
- `tensor<DIM1 x DIM2 x ... x DTYPE>`
- Dynamic dims (`?`) silently dropped (line 75 comment)
- Dtype may contain `x` (e.g. `index`) — pattern terminates at right boundary

**Algorithm:**
1. Match outer `tensor<...>` (line 67)
2. Extract inner type string (line 70)
3. Regex match `^((?:\d+\s*x\s*|[?]\s*x\s*)+)` to find all dim tokens (line 75)
4. Extract all `\d+` values from dim prefix (line 78)
5. Consume dtype from end of dim prefix to first `,` (line 81)
6. Return `{"shape": tuple, "dtype": str}` or `None` if no dims or missing dtype

**Dynamic dims:** Matched in regex but discarded (line 75 allows `[?]\s*x\s*` but not included in extraction line 78).

**Examples:**
- `tensor<256xf16>` → `{"shape": (256,), "dtype": "f16"}`
- `tensor<32x64xf32>` → `{"shape": (32, 64), "dtype": "f32"}`
- `tensor<?x64xi32>` → `{"shape": (64,), "dtype": "i32"}` (dynamic dim dropped)
- `tensor<32xindex>` → `{"shape": (32,), "dtype": "index"}`

---

## Numeric Parsing

### `parse_numeric(s: str, dtype: Optional[str] = None) → int | float`
**Module:** parser_utils.py:86–127

Parse numeric string to Python `int` or `float`.

**Handles:**
- Decimal integers: `123`, `-45`
- Floats: `1.5`, `1e-10`
- Hex constants: `0xFF00`, `0x123ABC`
- Float dtype hex reinterpretation (lines 104–114)

**IEEE 754 Reinterpretation (lines 104–114):**
When `dtype` is a float type and literal is hex, reinterpret as bit pattern:

- **`f32`:** Mask to 32 bits, view as `np.float32` (line 106)
- **`f16`:** Mask to 16 bits, view as `np.float16` (line 108)
- **`bf16`:** Mask to 16 bits, shift left 16, view as `np.float32` (bf16 = upper 16 bits of f32; layout is 1 sign + 8 exp + 7 mantissa, unlike f16's 1 sign + 5 exp + 10 mantissa) (lines 109–114)
- **Integer/index types:** Hex stays as plain `int` (line 115)

**Widening:** All float results are `Python float` (64-bit double); all ints are `Python int` (arbitrary precision). Caller narrowing at use-site.

**Fallback:** Return `0` if no parse succeeds (line 127).

**Errors:** None — always returns a value.

---

## Dense Payload Parsing

### `parse_dense_payload(payload: str, elem_dtype: Optional[str] = None) → tuple`
**Module:** parser_utils.py:130–143

Parse content extracted from inside `dense<...>`.

**Returns:** `(value, is_list)` where:
- Scalar payload `0.0` → `(scalar, False)`
- List payload `[16, 32]` → `([16, 32], True)`

**Algorithm:**
1. Check if payload starts with `[` (line 138)
2. If list: extract content, split on `,`, parse each via `parse_numeric` (lines 140–142)
3. Else: parse scalar via `parse_numeric` (line 143)

---

## SSA Name Finding

### `find_ssa_names(text: str) → List[str]`
**Module:** parser_utils.py:28–30

Find all SSA value references in text, including multi-result `%base#N` forms.

**Regex:** `r'%\w+(?:#\d+)?'` (line 25)
- Matches `%name` or `%name#0`, `%name#1`, etc.

**Returns:** List of all matches in order of appearance.

---

## Multi-Result LHS Parsing

### `parse_multi_result_lhs(lhs_text: str) → List[str]`
**Module:** parser_utils.py:33–50

Parse MLIR multi-result assignment LHS.

**Accepts three forms:**
- Bundled: `"%g:2"` → `["%g#0", "%g#1"]` (line 44–46)
- Comma form: `"%x, %y"` → `["%x", "%y"]` (lines 47–49)
- Single: `"%x"` → `["%x"]` (implicit list of one)

**Algorithm:**
1. Try match bundled form `(%\w+):([1-9]\d*)` (line 43)
2. If match: expand to explicit `"%base#{i}"` list (lines 45–46)
3. Else: split on `,`, validate each part matches `%\w+` (lines 47–49)

**Errors:** `ValueError` on malformed input.

---

## BoxSet Lowering

### `BoxSet.try_from_affine_set(aset: AffineSet) → Optional[BoxSet]`
**Module:** affine.py:439–507

Lower axis-aligned `AffineSet` to optimized `BoxSet`.

**Lowering succeeds iff:**
1. Every constraint is separable into `dim_coeffs[i] * d_i + sym_term >= 0` or `sym_term == 0` form (via `_constraint_to_linear_syms`)
2. Exactly one dim coefficient is non-zero per constraint
3. That coefficient is `±1` (unit magnitude)
4. Every axis pinned on both sides (has `lo` and `hi` candidates after processing all constraints)

**Constraint processing (lines 469–497):**

For each constraint:
1. Check if equality or inequality (line 470)
2. Linearize via `_constraint_to_linear_syms(expr, n_dims, n_syms)` → `(dim_coeffs, sym_coeffs, const)` (line 472)
3. Find non-zero dim indices (line 476)
4. Validate exactly one dim with coefficient `±1` (lines 477–482)
5. Build symbolic term `sym_term` from `sym_coeffs` and `const` via `_build_sym_term` (line 484)

**Bound assignment (lines 485–497):**
- **Equality `d_i == pin`:** Set both `lo[i]` and `hi[i]` (hi exclusive = `pin + 1`) (lines 486–489)
- **Inequality `d_i >= -sym_term` (when coeff +1):** Update `lo[i]` (line 493)
- **Inequality `d_i <= sym_term` (when coeff -1):** Update `hi[i]` (line 497, hi = `sym_term + 1`)

Multiple constraints per axis are combined with `sym_max` (lo) and `sym_min` (hi).

**Validation (lines 499–506):**
- Every axis must have both `lo` and `hi` set (line 499)
- For concrete boxes, detect contradictions early: `lo[i] >= hi[i]` returns `None` (line 505)
- Symbolic boxes skip early contradiction check; callers detect via `is_empty(symbols=...)` after `specialize()`

**Returns:** `None` if not representable; else `BoxSet(lo, hi)`.

### `_constraint_to_linear(node: _Node, n_dims: int) → Optional[Tuple[List[int], int]]`
**Module:** affine.py:548–560

Flatten dim-only constraint AST into `(coeffs, const)`.

Wrapper over `_constraint_to_linear_syms` with `n_syms=0` — any `sym` atom trips the bounds check and returns `None`.

### `_constraint_to_linear_syms(node: _Node, n_dims: int, n_syms: int) → Optional[Tuple[List[int], List[int], int]]`
**Module:** affine.py:586–647

**Core linearisation engine.** Flatten constraint AST into `(dim_coeffs, sym_coeffs, const)` representing:
```
sum(dim_coeffs[i] * d_i) + sum(sym_coeffs[j] * s_j) + const >= 0
```

**Recursive walker (lines 606–643):**

```
walk(node, sign):
  const      → const_box[0] += sign * value
  dim        → dim_coeffs[i] += sign
  sym        → sym_coeffs[j] += sign (if j < n_syms)
  add        → walk(left, sign) and walk(right, sign)
  sub        → walk(left, sign) and walk(right, -sign)
  neg        → walk(operand, -sign)
  mul        → walk(coefficient * operand):
               - if operand is dim → dim_coeffs[i] += sign * coef
               - if operand is sym → sym_coeffs[j] += sign * coef
               - if operand is const → const += sign * coef * value
  ref, max, min, ...  → return False
```

**Returns:** `None` if any `ref` node, dim × dim product, or dim × sym product is encountered. Otherwise returns tuple.

**Invariant:** Rejects first-class `("max", ...)` / `("min", ...)` nodes (product of two operands with variables). These are constructed by `sym_max` / `sym_min` at `BoxSet.intersect` time, not the surface parser, so lowering never sees them.

### `_build_sym_term(sym_coeffs: List[int], const: int) → Bound`
**Module:** affine.py:563–583

Reassemble a `Bound` from flattened `sum(sym_coeffs[j] * s_j) + const`.

**Algorithm:**
1. Start with `const` as `Bound` (line 573)
2. For each symbol with non-zero coefficient (lines 574–582):
   - Coefficient `+1` → term = `("sym", j)`
   - Coefficient `-1` → term = `sym_neg(("sym", j))`
   - Else → term = `("mul", c, ("sym", j))`
   - Accumulate via `sym_add(expr, term)` (line 582)

**Returns:** Plain `int` when no symbol contributes; else `Bound` AST node. Concrete bounds stay unwrapped for fast-path `isinstance(b, int)` checks.

---

## Ownership & Mutation Model

**Immutability principle:** All parsed objects (`AffineMap`, `AffineSet`, `BoxSet`) are **frozen dataclasses** (immutable after construction). No mutation after parsing.

**Mutable state during parsing:**
- `_Parser.pos` — advanced by `consume()`, not shared across threads
- `_Parser.dim_index`, `_Parser.sym_index` — populated by caller after parsing; read-only afterwards

**AST nodes:** Plain tuples, hashable, immutable.

**Bound evaluation:** `eval_bound` takes `symbols` as immutable sequence; does not mutate either operand.

**Box operations** (`intersect`, `translate`, `specialize`): All return **new** `BoxSet` instances; original unchanged.

---

## Error Handling

**Parsing errors:** Raise `ValueError` with descriptive message. No custom exception types.

- **Tokenisation:** Skips unparseable chars (line 109); no error
- **Parse failures:** E.g. missing expected token, unexpected EOL, invalid identifier
- **Dimension/symbol count mismatch:** Raised at evaluation time, not parse time
- **Multi-result LHS:** `ValueError` on malformed bundled/comma syntax (line 50)
- **Attribute extraction:** Skips malformed entries; returns partial dict (lines 207–209)

---

## Key Invariants & Design Notes

1. **Affine expressions only:** No floor, ceil, mod, or non-linear operations. Surface parser rejects them.

2. **Constraint normalisation:** All inequality constraints stored as `lhs - rhs >= 0`, allowing uniform `>= 0` evaluation.

3. **Equality as first-class tuple:** `("eq", lhs, rhs)` preserved through the AST so `affine_set_contains` can dispatch correctly (line 485–487 in parser_ast.py).

4. **Bracket depth skipping:** When counting angle brackets in `keyword<...>`, both `>=` and `->` are explicitly skipped (lines 334–341 in parser_utils.py). This is critical for parsing constraint expressions like `affine_set<(d0) : (d0 >= 0)>` without prematurely closing the `<...>` span.

5. **Canonical fallback:** When `_Parser.dim_index` is empty (e.g. standalone `parse_expr()` call), the parser accepts `d0`, `d1`, `s0`, `s1` by numeric suffix extraction (lines 250–267). This enables testing without wrapping in full `affine_map`/`affine_set` syntax.

6. **Reference pass-through:** `("ref", "%name")` nodes are not resolved during parsing; domain-specific code (e.g. subscript evaluators) resolves them post-parse.

7. **Parse-time lowering:** `parse_affine_set` lowers axis-aligned, concrete, non-symbolic sets to `BoxSet` automatically. Tests needing raw `AffineSet` call `parse_affine_set_raw`.

8. **Symbolic bounds:** `BoxSet.lo` / `hi` may hold AST nodes over symbols only (no dims). `_all_concrete` flag caches whether all bounds are plain ints, enabling O(ndim) operations on concrete boxes without AST walks.

9. **Constant folding in sym_* helpers:** `sym_add`, `sym_neg`, `sym_max`, `sym_min` apply MVP (Minimum Viable Product) folding: two concrete operands → fold to int; idempotence on repeated symbols; additive identity. No deep canonicalisation (no commutativity, no nested absorption).

10. **Row-major enumeration:** `enumerate_affine_set` returns tuples in row-major order via `itertools.product`.

11. **Duck typing in Python:** The `Bound` union type (`Union[int, tuple]`) is identified by `isinstance(b, int)` checks. Rust will need explicit `enum Bound { Concrete(i32), Symbolic(Node) }`.

12. **Symbolic sets stay on AffineSet:** `BoxSet.try_from_affine_set` returns `None` for any set with symbols (n_syms > 0), keeping them on the `AffineSet` branch. See affine.py:421 TODO for potential future optimisation.

---

## Summary Table

| Function | Module | Input | Output | Key Constraint |
|----------|--------|-------|--------|-----------------|
| `_tokenise` | parser_ast | `str` | `List[str]` | Regex-driven; handles %ref, d0, -123 |
| `parse_affine_map` | parser_ast | `str` | `AffineMap` | Requires `->` separator |
| `parse_affine_set` | parser_ast | `str` | `AffineSet \| BoxSet` | Lowers concrete axis-aligned sets |
| `parse_affine_set_raw` | parser_ast | `str` | `AffineSet` | No lowering; requires `:` separator |
| `parse_expr` | parser_ast | `str` | `_Node` | Standalone expression; accepts canonical d0/s0 fallback |
| `eval_affine_map` | parser_ast | `AffineMap, dims` | `Tuple[int, ...]` | len(dims) == n_dims |
| `affine_set_contains` | parser_ast | `AffineSet, point, symbols` | `bool` | All constraints must satisfy |
| `enumerate_affine_set` | parser_ast | `AffineSet, shape, symbols` | `List[Tuple[int, ...]]` | Brute-force; row-major |
| `enumerate_membership_keys` | parser_ast | `family, domain, point, bound` | `List[int]` | Parameterised query |
| `eval_bound` | parser_ast | `Bound, symbols` | `int` | Symbolic bounds resolved |
| `sym_add`, `sym_neg`, `sym_max`, `sym_min` | parser_ast | `Bound, Bound` | `Bound` | MVP folding only |
| `parse_attr_block` | parser_utils | `str, aliases` | `Dict[str, Any]` | Handles keyword<...>, #alias |
| `_extract_attr_value` | parser_utils | `str, aliases` | `(str, int)` | Bracket depth skips >=, -> |
| `parse_tensor_type` | parser_utils | `str` | `Optional[Dict]` | Extracts shape, dtype; drops ? |
| `parse_numeric` | parser_utils | `str, dtype` | `int \| float` | IEEE 754 hex reinterpret for floats |
| `find_ssa_names` | parser_utils | `str` | `List[str]` | Regex %name, %name#N |
| `parse_multi_result_lhs` | parser_utils | `str` | `List[str]` | Bundled "%g:2" or comma form |
| `BoxSet.try_from_affine_set` | affine | `AffineSet` | `Optional[BoxSet]` | Unit-coeff, pinned every axis |
| `_constraint_to_linear_syms` | affine | `_Node, n_dims, n_syms` | `Optional[Tuple[...]]` | Rejects dim×dim, dim×sym products |

---

Perfect! Now I have the full picture. Let me create a comprehensive spec for the Rust engineer:

# affine-impl

## Overview
Complete specification for porting `ktir_cpu/affine.py` to Rust. This subsystem provides parsed value containers (`AffineMap`, `AffineSet`, `BoxSet`) and constraint-solving utilities. The existing Rust port (in `rust/src/affine.rs`) covers only `AffineMap.eval`, `BoxSet` containment/intersection, and `AffineSet.contains` — it is missing **9 critical methods** listed below.

---

## Public Types

### `AffineMap`
**Immutable value type** (frozen dataclass in Python → owned struct in Rust; thread-safe if all inner types are `Send`).

**Fields:**
- `n_dims: usize` — number of input dimension variables (d0, d1, ...)
- `exprs: Vec<_Node>` — tuple of AST nodes (one per output dimension); in Rust use `Vec<AffineExpr>` or equivalent AST representation
- `source: String` — original verbatim string for debugging / round-trip

**Methods:**

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `eval` | `fn eval(dims: &[i64]) -> Vec<i64>` | **PARTIALLY PORTED.** Evaluate each output expression against concrete dims. Returns tuple of output integers. Raises on dim count mismatch. Delegates to `_eval_node` in parser. |
| `is_identity` | `fn is_identity() -> bool` | **MISSING.** Structural check: true iff output[i] == d_i for every i. Uses `_match_pure_dim_ref` to ensure each output flattens to `1 * d_i + 0` (not fooled by probe-based checks on e.g. `d0 + d1 - 2`). Used at parse time to detect trivial maps; when true, callers drop `coordinate_order` → `None` to skip per-coord `eval()` calls. |
| `is_permutation` | `fn is_permutation() -> bool` | **MISSING.** Structural check: true iff map is square (output count == input count) AND each output is a single dim variable AND every dim index appears exactly once. Rejects shears, scalings, constant offsets, many-to-one collapses. Used by ops that iterate in a permuted order; implementation sorts enumerated points by the map's image (only well-defined for permutations). |

**Ownership/Mutation:**
- Immutable everywhere; no mutable borrowing.
- `exprs` and `source` are never modified post-construction.

**Key Invariants:**
- `len(exprs) > 0` (always has at least one output).
- `n_dims ≥ 0` (can be 0 for constant-only maps like `() -> (42)`).
- All `_Node`s in `exprs` respect the grammar: atoms are `"dim"`, `"sym"`, `"const"`, `"ref"`; operators are `"add"`, `"sub"`, `"neg"`, `"mul"` (constant coeff only), `"max"`, `"min"`.

**Python-isms:**
- None; straightforward value type.

---

### `AffineSet`
**Immutable value type** (frozen dataclass).

**Fields:**
- `n_dims: usize` — number of dimension variables
- `constraints: Vec<_Node>` — tuple of AST nodes; each is the LHS of `expr >= 0` or `expr == 0`
- `source: String` — original verbatim string
- `n_syms: usize = 0` — number of symbol variables (s0, s1, ...)

**Methods:**

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `contains` | `fn contains(point: &[i64], symbols: &[i64]) -> bool` | **PARTIALLY PORTED.** Check if *point* satisfies all constraints. For `("eq", lhs, rhs)` nodes: `lhs == rhs`; for `("sub", ...)` nodes: `expr >= 0`. Delegates to `_eval_node`. Raises on dimension/symbol count mismatch. |
| `enumerate` | `fn enumerate(shape: &[usize], symbols: &[i64]) -> Vec<Vec<i64>>` | **MISSING.** Return all integer points in `[0, shape)` satisfying all constraints. Brute-force iteration: `itertools.product(*ranges)` filtered by `contains`. Raises on shape dimension mismatch. Semantics match `parser_ast.enumerate_affine_set` (line 491–510). |
| `is_full` | `fn is_full(shape: &[usize]) -> bool` | **MISSING.** Return true iff this set covers every coordinate in *shape* (i.e. = `[0, shape)`). Uses **vertex check**: an affine set is convex, so it contains `[0, shape)` iff it contains all `2^n_dims` corners of that box. This is O(2^n_dims) constraint evaluations instead of O(∏ shape). Called at parse time to detect trivial coordinate sets; when true, callers drop `coordinate_set` → `None` to take the contiguous fast path. |

**Ownership/Mutation:**
- Immutable.

**Key Invariants:**
- All constraints are AST nodes over dims and symbols only (no `"ref"` atoms).
- Constraint format: either `("sub", lhs, rhs)` (meaning `lhs - rhs >= 0`) or `("eq", lhs, rhs)` (meaning `lhs == rhs`).
- `n_dims > 0` (always has at least one dimension).

**Python-isms:**
- None; straightforward value type.

---

### `BoxSet`
**Immutable value type** (frozen dataclass with a cached derived field).

**Fields:**
- `lo: Vec<Bound>` — inclusive lower bounds per axis (see **Bound** below)
- `hi: Vec<Bound>` — exclusive upper bounds per axis
- `_all_concrete: bool` — **cached at construction** (via `__post_init__` equivalent); true iff every entry in `lo` and `hi` is a plain `int`. Set via `object.__setattr__` in Python (frozen dataclass trick); in Rust, set in a `new()` or `__post_init__` equivalent. `init=False, compare=False, repr=False` in Python → in Rust, keep it private and use a property accessor.

**Type Alias:**
```rust
type Bound = Union<i64, AffineExpr>;  // or enum Bound { Concrete(i64), Symbolic(Box<AffineExpr>) }
```
A bound is either a concrete `i64` (fast path) or an AST node representing a linear expression over **symbol variables only** (no `"dim"` nodes). Concrete bounds stay unwrapped so `isinstance(b, int)` checks work.

**Properties / Accessors:**

| Property | Type | Semantics |
|----------|------|-----------|
| `n_dims` | `fn n_dims() -> usize` | Read-only: `len(lo)` (always == `len(hi)` by invariant). |
| `is_concrete` | `fn is_concrete() -> bool` | **MISSING public accessor.** Read-only: returns `_all_concrete`. Use instead of touching `_all_concrete` directly. |

**Methods:**

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `contains` | `fn contains(point: &[i64], symbols: &[i64]) -> bool` | **PARTIALLY PORTED.** True iff `lo[d] <= point[d] < hi[d]` for every dim. `symbols` required to resolve symbolic bounds; concrete boxes ignore it (cached flag short-circuits, no AST walk). Passing too few symbols on a symbolic box raises `IndexError` (from `eval_bound`). |
| `enumerate` | `fn enumerate(shape: Option<&[usize]>, symbols: &[i64]) -> Vec<Vec<i64>>` | **MISSING.** Return all integer points in the box in row-major order. `shape` optional for signature parity with `AffineSet.enumerate` (which needs external bounding box). `shape` is a sanity check: passed values must upper-bound `hi` componentwise, else raises. Symbolic boxes specialize first (line 310), concrete boxes skip that. Returns `itertools.product(*(range(lo[d], hi[d])))` for each dim. |
| `is_empty` | `fn is_empty(symbols: &[i64]) -> bool` | **MISSING.** True iff any axis has `hi[d] <= lo[d]`. On symbolic boxes, per-axis comparison done after resolving bounds. |
| `is_full` | `fn is_full(shape: &[usize], symbols: &[i64]) -> bool` | **MISSING.** True iff this box equals `[0, shape)` exactly. A translated box `[x, x + shape)` returns false even when per-axis extent matches (intentional — callers use `true` as licence to drop `coordinate_set` → `None`; reporting full on translated box would silently miscompile). Symbolic boxes specialized first. |
| `lower_bounds` | `fn lower_bounds(symbols: &[i64]) -> Vec<i64>` | **MISSING.** Return `lo` resolved to `i64`. Concrete boxes use cached flag, return `lo` directly (cast from `Vec<Bound>` to `Vec<i64>`). Symbolic boxes resolve each entry via `eval_bound`. Used to get the partition origin in `distributed_tile_access`. |
| `specialize` | `fn specialize(symbols: &[i64]) -> BoxSet` | **MISSING.** Return a concrete `BoxSet` with all symbolic bounds resolved. Concrete boxes return `self` unchanged (cached flag check, no copy). Used at boundary between symbolic-IR-time and runtime-resolved values. |
| `translate` | `fn translate(offset: &[Bound]) -> BoxSet` | **MISSING.** Return a new box shifted by *offset* along each axis. `offset` may carry symbolic entries; `sym_add` folds concrete-on-concrete so a static box translated by static offset stays concrete (line 410–413). Raises on offset dim mismatch. |
| `intersect` | `fn intersect(other: &BoxSet) -> BoxSet` | **PARTIALLY PORTED** (returns `Option<BoxSet>` in current Rust, but Python returns a potentially-empty `BoxSet`). Axis-wise intersection; result may be empty (check via `is_empty()`). Uses `sym_max`/`sym_min` so concrete-on-concrete folds to ints (no AST allocation). Raises `TypeError` on mixed-type (other is `AffineSet`), `ValueError` on dim mismatch. |
| `try_from_affine_set` (class method) | `fn try_from_affine_set(aset: &AffineSet) -> Option<BoxSet>` | **MISSING.** Lower an axis-aligned `AffineSet` to `BoxSet`; returns `None` if not representable. Succeeds iff every constraint has form `c * d_i + k(syms) >= 0` or `c * d_i + k(syms) == 0` with `c ∈ {+1, -1}` (single dim, unit coeff) AND every axis pinned on **both** sides. `k(syms)` may be int or linear combination of symbols. Equality constraints pin `lo[i]` and `hi[i] = pin + 1`. Inequality/equality on same axis combined with `sym_max` (lo) / `sym_min` (hi). Assumes symbols ≥ 0 (matches dim-size semantics). Constraints with non-±1 dim coefficients, dim coeff with symbol, or non-linear symbol products return `None`. See lines 440–507. |

**Ownership/Mutation:**
- Immutable after construction.
- `_all_concrete` is set once at construction via `__post_init__`-equivalent; never changes.

**Key Invariants:**
- `len(lo) == len(hi)` (enforced in constructor).
- `_all_concrete` accurately reflects whether every entry is `i64` (not AST node).
- All `Bound` entries in `lo`/`hi` are pure expressions over symbols only (no `"dim"` nodes).
- For concrete boxes, implicit: no contradictions (e.g. `lo[d] < hi[d]`; detected early in `try_from_affine_set` line 504–506).

**Python-isms:**
- Frozen dataclass with cached derived field requires post-init hook to bypass descriptor lock. In Rust, use a private field + public accessor method or initialize in `new()`.
- `cast(Tuple[int, ...], self.lo)` in `lower_bounds` (line 378) — narrows static type hint to reflect runtime guarantee. In Rust, pattern match or `unwrap()` after asserting all entries are concrete.

**Design Note:** `BoxSet` and `AffineSet` are **structural peers under a `Union`**, not parent/child classes. Fast paths via `isinstance(obj, BoxSet)` dispatch must be visible at call sites; no polymorphism. Mixed-type operations raise `TypeError`.

---

## Helper Functions (Internal)

### `_match_pure_dim_ref(node: _Node, n_dims: usize) -> Option<usize>`
(Line 510–545)

**Semantics:** Match *node* against `1 * d_i + 0` and return `i`, else `None`. Uses `_constraint_to_linear` to flatten AST. Returns `None` if constant is nonzero, coefficient is not 1, or multiple dims present. Used by `AffineMap.is_identity` and `is_permutation` for structural checks that cannot be fooled by probe-based evaluation.

**Examples:**
- `d0` → `Some(0)`, `d2` → `Some(2)`, `d1 + 0` → `Some(1)`
- `d0 + 1` → `None` (nonzero const), `2 * d0` → `None` (non-unit), `d0 + d1` → `None` (multiple dims), `-d0` → `None` (coeff -1, not 1)

---

### `_constraint_to_linear(node: _Node, n_dims: usize) -> Option<(Vec<i64>, i64)>`
(Line 548–560)

**Semantics:** Flatten a dim-only constraint AST into `(coeffs, const)` where the constraint represents `sum(coeffs[i] * d_i) + const >= 0`. Wrapper over `_constraint_to_linear_syms(node, n_dims, n_syms=0)` — any `"sym"` atom trips the guard and returns `None`, preserving "reject symbols" contract.

**Returns:** `None` if expression is not separable (e.g. `"ref"` atom, non-linear symbol product). Otherwise `(dim_coeffs, const)`.

---

### `_build_sym_term(sym_coeffs: Vec<i64>, const: i64) -> Bound`
(Line 563–583)

**Semantics:** Reassemble a `Bound` from `sum(sym_coeffs[j] * s_j) + const`. Returns a plain `i64` when no symbol contributes (every coefficient is zero) — the structural fast path on concrete bounds depends on that. Otherwise returns an AST node tuple suitable for `eval_bound`. Uses `sym_add`, `sym_neg` for constant folding.

---

### `_constraint_to_linear_syms(node: _Node, n_dims: usize, n_syms: usize) -> Option<(Vec<i64>, Vec<i64>, i64)>`
(Line 586–647)

**Semantics:** Flatten a parsed constraint AST into `(dim_coeffs, sym_coeffs, const)` representing `sum(dim_coeffs[i] * d_i) + sum(sym_coeffs[j] * s_j) + const >= 0`. **Core linear-algebra engine for `BoxSet.try_from_affine_set`.** Returns `None` if expression is not separable (e.g. `"ref"` atom, sym × dim product, non-linear term). Otherwise `(dim_coeffs, sym_coeffs, const)` with lengths `n_dims`, `n_syms`, respectively.

**Walk function:**
- Mutates three accumulators: `dim_coeffs` (length `n_dims`), `sym_coeffs` (length `n_syms`), `const_box` (single-element list for mutability in Python closure).
- Handles tags: `"const"`, `"dim"`, `"sym"`, `"add"`, `"sub"`, `"neg"`, `"mul"` (constant coeff with inner dim/sym/const).
- Sign-aware recursion: flips sign for `"sub"` and `"neg"` operands.
- Guards: `j >= n_syms` returns `False` (rejects out-of-range sym indices).
- Returns `False` for `"ref"` or any non-linear structure.

**Critical:** Used to extract `(dim_coeff, sym_term, const)` per constraint in `try_from_affine_set` (line 472). Constraints with `len(nz_dim_indices) != 1` or `abs(dim_coeff) != 1` cause lowering to fail.

---

## Parser Functions (External)

These live in `parser_ast.py` but are called by `affine.py`; the Rust port **must replicate their behavior**:

### `eval_affine_map(amap: AffineMap, dims: &[i64]) -> Vec<i64>`
(parser_ast.py:459–477)

Evaluate each expression in `amap.exprs` against dims. Raises on dim count mismatch. Already partially ported to Rust.

---

### `affine_set_contains(aset: AffineSet, point: &[i64], symbols: &[i64]) -> bool`
(parser_ast.py:480–488)

Check membership via constraint evaluation. For `("eq", lhs, rhs)`: `lhs == rhs`; for `("sub", ...)`: `expr >= 0`. Already partially ported to Rust.

---

### `enumerate_affine_set(aset: AffineSet, shape: &[usize], symbols: &[i64]) -> Vec<Vec<i64>>`
(parser_ast.py:491–510)

Brute-force enumerate all points in `[0, shape)` satisfying constraints. **MISSING from Rust port.** Required for `AffineSet.enumerate`.

---

### `eval_bound(b: Bound, symbols: &[i64]) -> i64`
(parser_ast.py:577–587)

Evaluate a `Bound` (int or AST node) against symbols. Concrete ints short-circuit without AST walk; symbolic bounds delegate to `_eval_node(b, dims=[], syms=symbols)`.

---

### `sym_add(a: Bound, b: Bound) -> Bound`
(parser_ast.py:590–604)

Build `a + b` with constant folding. Returns `i64` if both operands are concrete; absorbs additive identity. Otherwise constructs `("add", ...)` AST node. **CRITICAL for `BoxSet.translate` and `try_from_affine_set`.** In Rust, implement as generic function over `Bound` enum.

---

### `sym_neg(a: Bound) -> Bound`
(parser_ast.py:607–615)

Build `-a` with constant folding and double-negation collapse (`-(-x) → x`). Used in `try_from_affine_set`.

---

### `sym_max(a: Bound, b: Bound) -> Bound`
(parser_ast.py:618–637)

Build `max(a, b)` with MVP folding. Folds concrete-on-concrete; recognizes identical `("sym", k)` as idempotent. No deep canonicalization. **CRITICAL for `BoxSet.try_from_affine_set` and `intersect`.** Per-axis candidate count ≤ 2, so no explosion.

---

### `sym_min(a: Bound, b: Bound) -> Bound`
(parser_ast.py:640–655)

Mirror of `sym_max`; same folding rules.

---

## AST Node Type (_Node)

In Python, a plain tuple; in Rust, use an enum:

```rust
pub enum AffineExpr {
    Const(i64),
    Dim(usize),
    Sym(usize),
    Ref(String),  // named reference; domain-specific semantics
    Add(Box<AffineExpr>, Box<AffineExpr>),
    Sub(Box<AffineExpr>, Box<AffineExpr>),
    Neg(Box<AffineExpr>),
    Mul(i64, Box<AffineExpr>),  // constant coeff only
    Max(Box<AffineExpr>, Box<AffineExpr>),  // constructed by sym_max, not surface parser
    Min(Box<AffineExpr>, Box<AffineExpr>),  // constructed by sym_min, not surface parser
}
```

(Note: Current Rust port uses a simpler enum without `Sub`, `Neg`, `Ref`, `Max`, `Min`. **These must be added** for full parity.)

---

## Constraint Handling

Constraints in `AffineSet.constraints` are stored as:
- **Inequality:** `("sub", lhs, rhs)` meaning `lhs - rhs >= 0`
- **Equality:** `("eq", lhs, rhs)` meaning `lhs == rhs`

The `parse_constraint_list` normalizes inequalities to `lhs - rhs >= 0` form (flips `<=` to `>=` by swapping operands). Evaluation interprets `("eq", ...)` as equality and `("sub", ...)` as `>= 0`.

---

## Key Design Decisions for Rust Port

1. **`Bound` enum:** Define as `enum Bound { Concrete(i64), Symbolic(Box<AffineExpr>) }` or use a type alias `Union<i64, AffineExpr>` (less idiomatic). The `isinstance(b, int)` checks in Python must become enum matches or a helper method.

2. **Cached `_all_concrete` field:** In Rust, make it private and expose via `pub fn is_concrete(&self) -> bool`. Compute once in `new()` or `__post_init__`-equivalent; never recompute. This drives hot-path fast-forwarding in `contains`, `is_empty`, `is_full`, `enumerate`.

3. **No polymorphic `Union`:** Rust will use separate types, not a `Union`. Call sites that need `BoxSet | AffineSet` dispatch should use an enum:
   ```rust
   pub enum CoordinateSet {
       Box(BoxSet),
       Affine(AffineSet),
   }
   ```
   All methods raise `TypeError` on mixed-type operations.

4. **Ownership model:** All three types are immutable value types. `Vec<Bound>` and `Vec<AffineExpr>` are owned by the struct; no shared references or mutable borrows.

5. **Constraint encoding:** Store `("eq", lhs, rhs)` and `("sub", lhs, rhs)` as an enum:
   ```rust
   pub enum ConstraintNode {
       Inequality(AffineExpr),  // lhs - rhs >= 0
       Equality(AffineExpr, AffineExpr),
   }
   ```
   Or flatten to `AffineExpr` and use a parallel `Vec<ConstraintKind>`. (Current Rust port uses a `Constraint` struct with `expr` and `kind` — **extend `AffineExpr` enum to handle equality check separately**.)

6. **Symbolic bounds in `BoxSet`:** The current Rust port uses `Vec<i64>` for `lo`/`hi` (concrete only). **Must be refactored to `Vec<Bound>`** to support symbolic bounds from `try_from_affine_set`. Update `contains`, `is_empty`, `is_full`, `enumerate`, `specialize`, `translate`, `intersect` to handle symbolic operands.

---

## Missing Implementations (vs. Python)

1. **`AffineMap.is_identity`** — structural check for identity map.
2. **`AffineMap.is_permutation`** — structural check for permutation map.
3. **`AffineSet.enumerate`** — brute-force enumeration of points.
4. **`AffineSet.is_full`** — vertex check for full set.
5. **`BoxSet.n_dims` property** — accessor for `len(lo)`.
6. **`BoxSet.is_concrete` property** — public accessor for `_all_concrete`.
7. **`BoxSet.enumerate`** — enumerate all points in row-major order.
8. **`BoxSet.is_empty`** — check for empty extent.
9. **`BoxSet.is_full`** — check if box equals `[0, shape)`.
10. **`BoxSet.lower_bounds`** — resolve `lo` to concrete `Vec<i64>`.
11. **`BoxSet.specialize`** — resolve symbolic bounds against symbol values.
12. **`BoxSet.translate`** — shift box by offset (handling symbolic bounds).
13. **`BoxSet.try_from_affine_set`** — lower axis-aligned affine set to box.
14. **`_match_pure_dim_ref`** — match `1 * d_i + 0` pattern.
15. **`_constraint_to_linear`** — flatten dim-only constraint.
16. **`_build_sym_term`** — reassemble symbolic bound.
17. **`_constraint_to_linear_syms`** — flatten constraint with symbols (core lowering engine).
18. **Parser helpers:** `eval_bound`, `sym_add`, `sym_neg`, `sym_max`, `sym_min`.
19. **Enum variants in `AffineExpr`:** `Sub`, `Neg`, `Ref`, `Max`, `Min` (currently missing).

---

## Formulas & Constants

From `BoxSet.try_from_affine_set` (lines 486–497):
- **Equality pinning:** `k*d_i + k(syms) == 0` → `d_i == pin` where `pin = -k(syms) / k` (computed via `sym_neg` and `sym_term` assembly).
- **Lower bound from inequality:** `d_i + k(syms) >= 0` (k=1) → `d_i >= -k(syms)` → `lo[i] = -k(syms)`.
- **Upper bound from inequality:** `-d_i + k(syms) >= 0` (k=-1) → `d_i <= k(syms)` → `hi[i] = k(syms) + 1` (exclusive).
- **Axis combination:** `sym_max(lo[i], candidate)` and `sym_min(hi[i], candidate)`.
- **Contradiction detection (concrete):** `lo[i] >= hi[i]` detected early; symbolic boxes checked at `specialize` time via `is_empty()`.

---

## Trickiest Implementation Bits

**File: `ktir_cpu/affine.py`**

- **Line 250–259 (`BoxSet.__post_init__`):** Frozen dataclass cached-field pattern. Rust: use a `new()` function or `__post_init__`-equivalent method to compute `_all_concrete` once; make the field private.

- **Line 378 (`lower_bounds`):** Python `cast` to narrow type hint. Rust: either assert all entries are concrete, or use `match` on the `Bound` enum and `unwrap()`.

- **Line 440–507 (`BoxSet.try_from_affine_set`):** The constraint-lowering engine. See `_constraint_to_linear_syms` internals (line 586–647) for the walk function that extracts coefficients. **Core redesign in Rust:** replace tuple-based constraint representation with an enum and implement the walk as a recursive `match` on `AffineExpr`.

- **Line 472 (`_constraint_to_linear_syms` walk):** Sign-aware recursion with mutable accumulators. Rust: use immutable builders (`sym_add`, `sym_max`, `sym_min`) to construct the result, or mutable locals within a helper function.

- **Line 504–506 (contradiction detection):** Early detection of infeasible constraints in concrete boxes. Symbolic boxes may violate this at `specialize` time — callers must check `is_empty(symbols=...)` after specializing.

---

## Cross-Module Dependencies

- **From `parser_ast.py`:** `eval_bound`, `sym_add`, `sym_neg`, `sym_max`, `sym_min`, `_eval_node` (internals). These must be implemented in the Rust parser module or imported from it. Current Rust port is missing the symbolic-bound helpers (`sym_*` functions).

- **Used by:** `memory_ops.py` calls `BoxSet.intersect`, `translate`, `specialize`, `lower_bounds`, `is_empty`, `contains`, `enumerate`. The orchestrator relies on the full API contract (not just `eval` and `contains`).

---

## Summary Table: Existing vs. Missing

| Item | Existing (Rust) | Missing |
|------|-----------------|---------|
| `AffineExpr` enum | Yes (basic: Dim, Sym, Const, Add, Mul, FloorDiv, Mod) | Sub, Neg, Ref, Max, Min |
| `AffineMap.eval` | Yes | — |
| `AffineMap.is_identity` | No | Yes |
| `AffineMap.is_permutation` | No | Yes |
| `AffineSet.contains` | Yes | — |
| `AffineSet.enumerate` | No | Yes |
| `AffineSet.is_full` | No | Yes |
| `BoxSet.contains` | Yes | — |
| `BoxSet.intersect` | Yes (returns `Option`) | Update to handle symbolic bounds |
| `BoxSet.is_concrete` property | No | Yes |
| `BoxSet.n_dims` property | No | Yes |
| `BoxSet.enumerate` | No | Yes |
| `BoxSet.is_empty` | No | Yes |
| `BoxSet.is_full` | No | Yes |
| `BoxSet.lower_bounds` | No | Yes |
| `BoxSet.specialize` | No | Yes |
| `BoxSet.translate` | No | Yes |
| `BoxSet.try_from_affine_set` | No | Yes |
| `Bound` type | No | Yes (union of int / AffineExpr) |
| `eval_bound` | No | Yes |
| `sym_add`, `sym_neg`, `sym_max`, `sym_min` | No | Yes |
| `_constraint_to_linear_syms` | No | Yes |

---

Now I have enough information. Let me create the comprehensive spec:

# tests

## Overview
Test suite has **881 test functions** across **20 main test files** (~12,641 LOC), plus **~100 adapter tests** in `mlir_frontend/` for alternative parser. Organized by subsystem: dtype mapping, AST/affine logic, dialect parsing/execution, interpreter, memory, scheduling, latency modeling, and RFC spec gaps. **21 example MLIR kernels** drive end-to-end execution tests.

## dtypes.py Module (Rust Port Target)

**File**: `/Users/moosevan/git/ktir-cpu/ktir_cpu/dtypes.py` (89 LOC)

### Public Data

- **`SUPPORTED_DTYPES: dict[str, np.dtype]`** — Canonical KTIR→NumPy dtype mapping:
  - Keys: `"f16"`, `"fp16"`, `"float16"` → `np.float16` (2 bytes)
  - Keys: `"f32"`, `"float32"` → `np.float32` (4 bytes)
  - Keys: `"i1"` → `np.bool_` (1 byte)
  - Keys: `"i32"`, `"si32"`, `"index"` → `np.int32` (4 bytes)
  - Keys: `"i64"`, `"si64"` → `np.int64` (8 bytes)

- **`_PLACEHOLDER_DTYPES: frozenset[str]`** = `{"fp8", "mxfp8"}` — Placeholder types that raise `NotImplementedError` when accessed via `to_np_dtype()`. Not yet exercised by any example kernel.

- **`_REVERSE_MAP: dict[np.dtype, str]`** — Inverse: NumPy dtype→KTIR string (`np.float16`→`"f16"`, etc.). Subset of `SUPPORTED_DTYPES` (only 4 entries: f16, f32, i32, i64).

### Functions

1. **`to_np_dtype(dtype: str) -> np.dtype`** (lines 57–71)
   - Convert KTIR dtype string to NumPy dtype.
   - **Raises**: `NotImplementedError` if dtype in `_PLACEHOLDER_DTYPES` with message: `"dtype {dtype!r} is a placeholder pending hardware confirmation; update SUPPORTED_DTYPES before adding examples that use it"`.
   - **Raises**: `ValueError` if dtype not in `SUPPORTED_DTYPES` with message: `"Unsupported KTIR dtype: {dtype!r}"`.
   - **Semantics**: Direct lookup + exception gatekeeping (prevents silent failures on unimplemented hardware types).

2. **`bytes_per_elem(dtype: str) -> int`** (lines 74–76)
   - Return element size in bytes for KTIR dtype string.
   - **Delegates** to `to_np_dtype(dtype).itemsize`.
   - **Raises**: Same as `to_np_dtype()`.

3. **`to_ktir_dtype(np_dtype: np.dtype) -> str`** (lines 79–88)
   - Map NumPy dtype to canonical KTIR string.
   - **Coerces** input via `np.dtype()` (allows numpy type instances).
   - **Raises**: `ValueError` if np_dtype not in `_REVERSE_MAP` with message: `"No KTIR dtype for NumPy dtype: {np_dtype!r}"`.
   - **Semantics**: Inverse mapping with restricted codomain (only 4 KTIR types have reverse mapping; e.g., `np.bool_` has no inverse).

### Invariants
- All dtype strings are lowercase alphanumeric (no special chars).
- `SUPPORTED_DTYPES` is the single source of truth for KTIR↔NumPy bidirectional conversion.
- Placeholder dtypes (`fp8`, `mxfp8`) are gatekeepers: any production example using them fails immediately.
- Byte sizes (`itemsize`) are platform-independent (hardware standard: f16/bool 1–2 bytes, i32 4, i64 8).

### Python-isms (Rust Redesign Notes)
- **NumPy dependency**: `np.dtype` is duck-typed by `.itemsize` property; Rust should use explicit byte-size constants.
- **String keys**: Both dicts are string-keyed; no enum yet. Rust port should use `enum DType` with serde/display derives.
- **Bidirectional mapping**: `_REVERSE_MAP` is a lossy inverse (e.g., `float16` aliases to `"f16"`); Rust should define canonical forms at definition site.
- **Exception gatekeeping**: `_PLACEHOLDER_DTYPES` uses set membership test; Rust should embed in enum variants (e.g., `Placeholder(PlaceholderType)`).

---

## Test File Summary (881 tests / 20 files)

| File | Tests | Type | Key Fixtures | Markers |
|------|-------|------|--------------|---------|
| **test_dialects_exec.py** | 158 | Execution | CoreContext, HBMSimulator, Tile | parametrize (ops) |
| **test_dialects_parse.py** | 94 | Parsing | arith/linalg/tensor/ktdp/scf ops | parametrize (op_text) |
| **test_ast.py** | 81 | Unit | _tokenise, parse_expr, eval_expr | parametrize (constants/dims) |
| **test_affine.py** | 80 | Unit | AffineMap, AffineSet, BoxSet | parametrize (maps/sets) |
| **test_latency.py** | 63 | Latency | LatencyModel, HBMSimulator | parametrize (kernels) |
| **test_ops.py** | 42 | Unit | ArithOps, MathOps, GridOps | _make_ctx, _tile helpers |
| **test_parser_errors.py** | 33 | Negative | MLIR parsing errors | pytest.raises |
| **test_ktir_cpu.py** | 23 | Integration | KTIRInterpreter | load(), execute_function() |
| **test_lx_scoping.py** | 19 | Semantics | LX memory access tracking | TileOps, affine attrs |
| **test_tile.py** | 17 | Unit | Tile shape/stride/dtype | _from_memref_str |
| **test_examples.py** | 16 | E2E | MLIR examples (21 kernels) | parametrize(get_test_params) |
| **test_ktir_simple.py** | 15 | Integration | Simple KTIR constructs | load(), execute_region() |
| **test_latency_modeling.py** | 13 | Latency | LatencyModel config/scaling | parametrize (SIMD, systolic) |
| **test_indirect_access.py** | 12 | Feature | Indirect access tiles | parametrize (kernels) |
| **test_interpreter.py** | 9 | Unit | execute_region(), scalar args | multi-result unpacking |
| **test_distributed_view.py** | 8 | Feature | Distributed HBM+LX views | parametrize (cores) |
| **test_spec_gaps.py** | 6 | Negative | RFC feature gaps | xfail(strict=True) |
| **test_parser_utils.py** | 5 | Unit | Affine alias parsing | _parse_affine_aliases |
| **test_dtypes.py** | 5 | Unit | dtype conversions | parametrize (dtype strings) |
| **test_grid_scheduler.py** | 2 | Integration | GridScheduler task dispatch | parametrize (num_cores) |
| **mlir_frontend/test_examples_adapt.py** | ~57 | E2E (adapt) | MLIRFrontendParser | Inherits TestXxxExecution |
| **mlir_frontend/test_parse_adapt.py** | ~47 | Parse (adapt) | MLIRFrontendParser | Inherits TestXxxParsers |
| **mlir_frontend/test_indirect_access_adapt.py** | ~6 | Feature (adapt) | MLIRFrontendParser | Inherits indirect tests |
| **mlir_frontend/test_registry_consistency.py** | ~4 | Meta | Regex vs bindings parser consistency | N/A |

---

## Examples: 21 MLIR Kernels

**Directory layout** (`examples/`):
- **`triton-ktir/`** (10 files) — Production-scale kernels from Triton compilation path
- **`latency/`** (3 files) — Reduced-footprint kernels for latency/scaling tests
- **`ktir/`** (5 files) — Hand-written KTIR; edge-case/failure fixtures
- **`rfc/`** (3 files) — RFC spec examples (currently all xfail)

| File | Function | Coverage | Grid | Notes |
|------|----------|----------|------|-------|
| **triton-ktir/vector_add_ktir.mlir** | `add_kernel` | Vector add (basic) | [1] | 4096 elem, BLOCK_SIZE=128 |
| **triton-ktir/vector_add_dynamic_ktir.mlir** | `add_kernel_dynamic` | Vector add (symbolic size) | [1] | Dynamic memref<?xf32>; tests n_elements ∈ {256,512,1024} |
| **triton-ktir/softmax_fwd_ktir.mlir** | `softmax_kernel` | Softmax (row-wise) | [32,1] | 4096×1024, online-softmax, f16 |
| **triton-ktir/layernorm_fwd_ktir.mlir** | `_layer_norm_fwd_fused` | Layer norm (fused Y+stats) | [32,1] | 1151×8192, mean/rstd outputs |
| **triton-ktir/matmul_fwd_ktir.mlir** | `matmul_kernel` | MatMul | [2,4] | M=64, N=8192, K=2048; K=16 accum iter |
| **triton-ktir/indexed_add.mlir** | `indexed_add_kernel` | Indirect access gather | [2,8] | x[index[grid0], :], output=x_gather+y |
| **triton-ktir/sdpa_2d.mlir** | `sdpa_kernel_2d` | Scaled dot-product attention | [1] | Q,K,V,out all [32,64] f16 |
| **triton-ktir/paged_attention.mlir** | `kernel_unified_attention_spyre_2d` | Paged attention (2-D grid) | [8,32] | Tiled online-softmax w/ block_tables |
| **latency/softmax_small.mlir** | `softmax_kernel_small` | Softmax (small, 64×64) | [32,1] | Latency test fixture |
| **latency/softmax_small_explicit.mlir** | `softmax_kernel_small_explicit` | Softmax (explicit linalg.reduce region) | [32,1] | Tests generic combiner syntax |
| **latency/matmul_small.mlir** | `matmul_kernel_small` | MatMul (small, 16×64×64) | [2,2] | Latency/scaling test |
| **ktir/softmax_wide.mlir** | `softmax_kernel` | Softmax overflow test (XFAIL) | [1] | C=262144 → LX overflow (16 MB > 2 MB) |
| **ktir/reduce_generic.mlir** | `reduce_explicit_region` | linalg.reduce (explicit region) | [1] | Generic combiner w/ yield |
| **ktir/reduce_multiop.mlir** | `reduce_multiop` | linalg.reduce (multi-op combiner) | [1] | max via cmpf+select |
| **ktir/ring_reduce.mlir** | `ring_reduce` | Cross-core ring reduce (XFAIL) | [4,1,1] | Requires #ktdp.reduce_kind attributes |
| **rfc/indirect-access-copy.mlir** | `indirect_access_copy` | 2-D indirect gather (XFAIL) | [1] | Y[m,k]=X[IDX1[m,k],IDX2[m,k]]; spec-gap |
| **rfc/indirect-scatter.mlir** | `indirect_scatter` | 2-D indirect scatter (XFAIL) | [1] | Dual of indirect-access-copy |
| **rfc/paged-tensor-copy.mlir** | `paged_tensor_copy_1core` | 4-D paged indirect gather (XFAIL) | [1] | Production-size; LX overflow |
| **rfc/paged-tensor-write.mlir** | `paged_tensor_write_1core` | 4-D paged indirect scatter (XFAIL) | [1] | Scatter dual; LX overflow |
| **rfc/distributed-view-copy.mlir** | `distributed_view_copy` | Distributed HBM+LX view (XFAIL) | [1] | RFC §C.3 |
| **rfc/add-with-control-flow.mlir** | `add` | Elementwise add w/ scf.for (XFAIL) | [1] | Requires linalg.add + tensor.empty |

---

## Test Categories & Porting Strategy

### 1. **Unit Tests (Direct 1:1 mapping to Rust #[test])**
- **test_dtypes.py** (5 tests): dtype conversions
  - `test_to_np_dtype()` — parametrize over (string, expected dtype, bytes)
  - `test_unknown_dtype_raises()` — parametrize ValueError cases
  - `test_placeholder_dtype_raises()` — fp8/mxfp8 gatekeeping
  - `test_to_ktir_dtype()` — parametrize reverse mapping
  - `test_to_ktir_dtype_unknown_raises()` — float64 has no KTIR form

- **test_ast.py** (81 tests): Affine expression parsing/evaluation
  - `TestTokenise` (3) — _tokenise() → token stream
  - `TestParseExpr` (10+) — parse_expr() + eval_expr() AST nodes
  - `TestParseAffineMap` (10+) — Parse affine maps from MLIR syntax
  - `TestEvalAffineMap` (15+) — eval_affine_map() over coordinate tuples
  - `TestAffineSetContains` (5+) — Membership tests on affine sets
  - `TestEnumerateAffineSet` (10+) — enumerate_affine_set() + sort-order verification
  - **Action**: Port AST node types as Rust enums, parsing as recursive descent, evaluation as pattern-match fold. Use `thiserror` for exceptions.

- **test_affine.py** (80 tests): AffineMap/AffineSet/BoxSet value objects
  - `TestAffineMapObject` (5) — AffineMap.eval(), .source field, frozen
  - `TestAffineMapIsPermutation` (8) — is_permutation() detection
  - `TestAffineSetObject` (10+) — AffineSet API
  - `TestBoxSetBasics` (10+) — Axis-aligned lowering, symbolic bounds
  - **Action**: Implement as immutable struct wrappers around parsed AST; use `derive(Eq)` for comparison.

- **test_ops.py** (42 tests): Dialect operation execution
  - `TestArithFloat` (12) — arith.addf, subf, mulf, divf, negf
  - `TestArithInt` (8) — arith.addi, subi, muli
  - `TestMath` (8) — math.sqrt, math.exp, math.log
  - `TestLinalg` (5) — linalg.reduce, linalg.generic
  - **Action**: Implement as methods on Op trait implementors; use `.execute(ctx: &mut CoreContext)` pattern.

- **test_interpreter.py** (9 tests): Interpreter edge cases
  - `test_execute_function_scalar_arg()` — Non-ndarray args stored as values
  - `test_execute_region_*()` — Region execution in isolation
  - `test_unknown_op_raises()` — ValueError on unregistered op
  - `test_multi_result_tuple_unpacked()` — Multi-result op handling
  - **Action**: Test interpreter API directly; mock register for multi-result cases.

### 2. **Parametrized Tests (Expand to multiple #[test] or use Rust test harness generators)**
- **test_dialects_parse.py** (94 tests) & **test_dialects_exec.py** (158 tests)
  - Parametrize over operation strings + expected parse results + execution values.
  - **Example**: `arith.addf` with f16 inputs → `(+, x, y)` AST → f16 output.
  - **Action**: Generate test cases via `macro_rules!` or explicit test functions per op. Merge parse+exec assertions into single test (Python uses separate test classes; Rust can inline).

- **test_parser_errors.py** (33 tests)
  - Invalid MLIR syntax → specific error message + error kind.
  - **Action**: Test parser error recovery; use `assert_matches!` macro for error types.

### 3. **Execution Tests Over MLIR Examples (Harness-driven, not 1:1 #[test])**
- **test_examples.py** (16 tests, ~57 with adapt)
  - **Classes**:
    - `TestExampleParsing` (3 parametrized) — Parse all 21 kernels, verify structure (grid, arguments, tensors).
    - `TestVectorAddExecution` (2) — Verify add_kernel output == x + y (f16 with f32 reference).
    - `TestVectorAddDynamicExecution` (1) — Dynamic memref<?xf32> with n_elements ∈ {256,512,1024}.
    - `TestSoftmaxExecution` (2) — Online softmax with padding (f16 vs f32 reference) + LX-overflow xfail.
    - `TestLayerNormExecution` (1) — Y output + mean/rstd statistics (row-wise norm).
    - `TestReduceExplicitRegion` (2) — linalg.reduce with explicit region vs shorthand.
    - `TestMatMulExecution` (1) — C ≈ A @ B (relaxed tolerance for f16 accumulation).
    - `TestIndexedAddExecution` (1) — Indirect gather x[index[grid0], dim1_start:] + y.
    - `TestPagedAttentionExecution` (1) — 2-D paged SDPA with block_tables + online softmax + causal mask.
    - `TestSdpaExecution` (1) — SDPA on 1 core.
    - `TestRingReduceExecution` (1, xfail) — 4-core ring reduce (missing parser support).

  - **Fixture Harness** (`conftest.py`):
    - `EXAMPLE_PARAMS: dict[str, list[dict]]` — Maps function name → list of `{path, execute_kwargs, [exception_msg]}`.
    - `get_test_params(*func_names, filter=None)` — Returns `(abs_path, func_name, entry)` triples; expands list-valued kwargs (e.g., `n_elements: [256, 512, 1024]`).
    - `parse_example(path, func_name)` → `ExampleMeta` — Metadata extracted from MLIR text via regex (independent of parser under test).
    - `InterpreterTestMixin._make_interp()` — Overridable for alternate parsers (regex vs MLIRFrontendParser).
    - `_build_kwargs(entry, tensors, overrides)` — Merge scalar execute_kwargs with tensor args.

  - **Porting Strategy**: 
    - **Do NOT port full end-to-end tests to Rust** (requires full interpreter, MLIR parsing, memory simulation).
    - **Port as integration test fixtures**: Load example MLIR from string, execute on mock Spyre hardware, verify output tensors.
    - **Use Rust test harness** (e.g., `proptest` or `parameterized` crate) to iterate over examples programmatically.
    - **Expected vs actual**: Maintain reference ground-truth values (NumPy pre-computed in conftest.py; Rust can embed as constants or .mlir files).

- **test_latency.py** (63 tests) — Latency/cycle counting on small kernels
  - Parametrize over `(simd, systolic, penalty, hbm_bw)` configurations.
  - Verify cycle formula matches LatencyModel predictions.
  - **Action**: Port LatencyModel as standalone library; unit-test cycle calculations, not kernel execution.

- **test_latency_modeling.py** (13 tests) — LatencyModel scaling laws
  - Verify cycle ∝ 1/bandwidth, ∝ SIMD_width, ∝ systolic_throughput.
  - **Action**: Keep as pure computation tests; mock hardware params.

### 4. **Feature Tests (RFC Spec Gaps & Adapter Tests)**

#### **Spec Gaps** (test_spec_gaps.py, 6 tests, all xfail)
- Mark with `#[ignore]` or skip in CI; document as spec gaps.
- Examples:
  - `test_paged_tensor_indirect_access()` — LX overflow (16 MB > 2 MB) — correct parse, partial exec.
  - `test_paged_tensor_indirect_scatter()` — Scatter dual.
  - `test_linalg_add_tensor_empty()` — linalg.add not implemented.
  - `test_tensor_extract_slice()` — tensor.extract_slice not implemented.
  - `test_scf_parallel()`, `test_scf_reduce()` — Control flow not yet implemented.

#### **Adapter Tests** (mlir_frontend/)
- **Conditional compilation**: Gate on `mlir_ktdp` availability.
- **Porting strategy**: If Rust has a MLIR frontend, create analogous adapter suite. Otherwise, skip.
- Files:
  - `test_examples_adapt.py` (57 tests) — Inherits TestXxxExecution, injects MLIRFrontendParser.
  - `test_parse_adapt.py` (47 tests) — Inherits TestXxxParsers, overrides regex-only tests.
  - `test_indirect_access_adapt.py` (6 tests) — Indirect access via MLIRFrontendParser.
  - `test_registry_consistency.py` (4 tests) — Regex vs bindings parser consistency (metadata extraction).

### 5. **Latency-critical Tests** (test_latency.py, test_lx_scoping.py)
- **test_lx_scoping.py** (19 tests) — LX memory access tracking, affine attrs preserved through tiles.
- **test_distributed_view.py** (8 tests) — HBM+LX partitioning, stride calculations.
- **test_indirect_access.py** (12 tests) — Indirect access tile semantics, variable-order enumeration.
- **Action**: Inline into main test suite; not latency-specific, but rely on correct memory/affine semantics.

---

## Pytest Markers

- **`@pytest.mark.parametrize`** — Heavy use; distribute across test parameters (examples, ops, dtypes).
- **`@pytest.mark.spec_gap`** (6 tests) — Known RFC conformance gaps; skip in normal CI, include in coverage reports.
- **`@pytest.mark.regex_only`** (4 tests) — Parser-specific (regex parse syntax not valid MLIR); skip mlir_frontend/conftest.py.
- **`@pytest.mark.xfail(strict=True)`** (4+ tests) — Expected failures that should become passing (gates for future work).
- **`@pytest.mark.xfail(reason=...)`** (1+ tests) — Expected failures, non-strict (known blockers).

---

## Porting Plan: Rust Test Coverage

### Phase 1: Unit Tests (Easy, ~200 tests)
1. **dtypes.py** → `dtypes/mod.rs` + `tests/test_dtypes.rs` (5 tests)
2. **parser_ast.py** (AST + evaluation) → `parser_ast/mod.rs` + tests (81 tests)
3. **affine.py** → `affine/mod.rs` + tests (80 tests)
4. **ops/** execution handlers → Per-dialect tests (42 ops tests)
5. **interpreter.py** edge cases → `tests/test_interpreter.rs` (9 tests)

### Phase 2: Integration Tests (Moderate, ~150 tests)
1. **Dialect parsing** → Generate test cases from example ops (94 parse + 33 error cases).
2. **Dialect execution** → Generate test cases from arith/math/linalg ops (158 exec tests).
3. **MLIR fixture loading** — Implement minimal MLIR text→IR parser or use `cxxbridge` to wrap Python parser.
4. **Kernel parametrization** — Use Rust macro to emit test function for each example kernel variant.

### Phase 3: Example-Driven Tests (Hard, ~16 tests, may skip full execution)
- **Decision**: Full end-to-end tests require complete interpreter + MLIR frontend.
  - **Option A** (Full port): Implement interpreter + memory model + all dialect ops. Timeline: weeks. Coverage: definitive.
  - **Option B** (Python harness): Keep test_examples.py in Python, invoke Rust via FFI. Timeline: days. Coverage: integration boundaries only.
  - **Option C** (Snapshot tests): Pre-compute expected outputs, store as Rust constants, verify Rust interpreter matches. Timeline: moderate.
  - **Recommendation**: Option C (snapshot tests) — low risk, high confidence.

### Phase 4: Spec Gap & Adapter Tests (Conditional)
- Skip if Rust MLIR frontend unavailable.
- Document as xfail fixtures in Rust; update when features land.

### Fixtures to Port
- **conftest.py**: EXAMPLE_PARAMS registry → Rust const array of test fixtures.
- **_make_ctx()** → Rust helper function.
- **_tile()** → Rust macro or helper.
- **InterpreterTestMixin** → Rust trait or generic test harness.

---

## Load-Bearing Implementation Details

### Affine Evaluation
- **File**: `ktir_cpu/parser_ast.py`
- **Semantics**: `eval_expr(node, dims)` evaluates AST node given dimension values; `eval_affine_map(map, dims)` returns tuple of output expressions evaluated. Used in `ktdp.load`/`ktdp.store` to enumerate coordinate sets and track memory access patterns.
- **Rust**: Recursive pattern-match on AST; same semantics.

### Coordinate Set Enumeration
- **File**: `ktir_cpu/parser_ast.py:enumerate_affine_set()`
- **Semantics**: Iterate over integer points in affine set `{(d0, d1, ...) : constraints}`, sorted by `access_tile_order` map.
- **Rust**: Generate integer points via constraint solver (GCD-based or Fourier-Motzkin); sort by affine map image.

### Exception Gatekeeping (dtype, spec gaps)
- **File**: `ktir_cpu/dtypes.py` (placeholder dtypes), `test_spec_gaps.py` (feature gaps)
- **Semantics**: Raise `NotImplementedError` immediately when unimplemented type/feature accessed, forcing spec updates before example kernels can use it.
- **Rust**: Use enum variants + `match` statements; compile-time exhaustiveness checking replaces runtime exceptions.

### Memory Simulation (HBM + LX)
- **File**: `ktir_cpu/memory.py`, `test_examples.py`, `test_lx_scoping.py`
- **Semantics**: HBMSimulator (unbounded flat address space), LXScratchpad (2 MB, per-core, overflow raises `MemoryError`).
- **Rust**: Implement as `struct HBM { data: HashMap<usize, u8> }`, `struct LX { data: [u8; 2_MB], used: usize }`.

### Online Softmax (Tiled)
- **File**: `test_examples.py:TestSoftmaxExecution::test_softmax_correct()` (line 226–256)
- **Semantics**: Tile-wise online softmax using Welford variance + max tracking; reference in NumPy (lines 251–254):
  ```python
  m = np.max(inp, axis=1, keepdims=True)
  e = np.exp((inp - m).astype(np.float32))
  s = np.sum(e, axis=1, keepdims=True)
  expected = (e / s).astype(np.float16)
  ```
- **Rust**: Same; f32 for stability during exp/sum, f16 for I/O.

### Paged Attention (Causal Mask + Block Table Indirect Access)
- **File**: `test_examples.py:TestPagedAttentionExecution::test_paged_attention()` (line 444–525)
- **Semantics**: 
  - Q, K, V loaded from block-table indexed KV cache.
  - Causal mask: `scores[row, col] = -inf` if `col > context_len + query_pos[row]`.
  - Online softmax across tiles: `M, L, acc` carry state between iterations.
  - Reference: Lines 482–525 (nested loop, causal mask generation, online softmax accumulation).
- **Rust**: Same loop structure; use `f32` for stability.

### Generator Semantics (Not in Rust)
- **File**: `test_examples.py:TestExampleParsing::test_parse_module()` parametrize (line 109–115)
- **Python**: `@pytest.mark.parametrize("path,func_name,entry", get_test_params())` yields tuples on-the-fly.
- **Rust**: Pre-expand to const array of test fixtures; emit test function per fixture via macro.

---

## Error Messages & Assertions (Exact Constants)

From **dtypes.py**:
- `"dtype {dtype!r} is a placeholder pending hardware confirmation; update SUPPORTED_DTYPES before adding examples that use it"`
- `"Unsupported KTIR dtype: {dtype!r}"`
- `"No KTIR dtype for NumPy dtype: {np_dtype!r}"`

From **test_spec_gaps.py**:
- `"LX scratchpad overflow"` (exception_msg for softmax_wide.mlir)
- `"ktdp.load of 4x8x2048x128 f16 tile (16 MB) exceeds 2 MB LX scratchpad"` (xfail reason)
- `"ktdp.reduce_kind / reduce_mode / grid_axis attributes"` (parser support gap for ring_reduce)

From **test_examples.py**:
- Vector add tolerance: `rtol=1e-2, atol=1e-2` (f16 rounding)
- Softmax tolerance: `rtol=1e-2, atol=1e-2` (f16 + online softmax accumulation)
- MatMul tolerance: `rtol=2e-2, atol=2e-1` (K=2048, 16 accumulation iterations in f16)
- Paged attention tolerance: `rtol=1e-2, atol=1e-2`

---

## Known Limitations (Not Testable in Rust Yet)

1. **MLIR Frontend Parser** (`mlir_ktdp` / MLIRFrontendParser) — Python bindings; skip adapter tests in Rust unless wrapping via FFI.
2. **NumPy Broadcasting** — Not directly tested; dtypes.py only maps scalar types. Tile broadcasting (if any) tested implicitly in dialect ops.
3. **Dynamic Memref** (`memref<?xf32>`) — Parsed as Tile with `None` in shape; enumeration assumes flattened coordinate set. Tested in `test_vector_add_dynamic`.
4. **Multi-dimensional Affine Sets** — Tested in test_affine.py; coordinate enumeration O(|points|) per access, not optimized.

---

## Summary Table: Test Coverage by Subsystem

| Subsystem | Files | Tests | Unit | Integration | E2E | Fixtures | Python Idiom |
|-----------|-------|-------|------|-------------|-----|----------|--------------|
| **dtypes** | 1 | 5 | ✓ | — | — | Constant maps | Dict lookup |
| **parser_ast** | 1 | 81 | ✓ | — | — | AST nodes | Recursive tuples |
| **affine** | 1 | 80 | ✓ | — | — | AffineMap, BoxSet | Immutable freeze() |
| **ops.arith** | 1 | 12 | ✓ | ✓ | — | Tile, CoreContext | np.float16 duck-type |
| **ops.math** | 1 | 8 | ✓ | ✓ | — | Tile, CoreContext | — |
| **ops.linalg** | 1 | 5 | ✓ | ✓ | — | Tile, region ops | Region with yield |
| **ops.ktdp** | 1 | 10 | ✓ | ✓ | — | Access tiles | Memory view construct |
| **dialects.parse** | 1 | 94 | — | ✓ | — | Op text snippets | Regex parse, duck typing |
| **dialects.exec** | 1 | 158 | — | ✓ | — | Op, CoreContext | Loose coupling, op registry |
| **parser_errors** | 1 | 33 | — | ✓ | — | Invalid MLIR | Error message matching |
| **interpreter** | 1 | 9 | ✓ | — | — | MLIR text, scalars | Dict context, SSA names |
| **examples** | 1 | 16 | — | — | ✓ | 21 MLIR kernels | parametrize, xfail |
| **latency** | 1 | 63 | ✓ | ✓ | — | Small kernels | Cycle formulas |
| **latency_modeling** | 1 | 13 | ✓ | — | — | LatencyModel config | Parametrize scaling |
| **lx_scoping** | 1 | 19 | — | ✓ | — | TileOps, memory | LX overflow tracking |
| **distributed_view** | 1 | 8 | — | ✓ | — | 4-D strided access | HBM+LX partitioning |
| **indirect_access** | 1 | 12 | — | ✓ | — | Variable-order enums | Permutation maps |
| **spec_gaps** | 1 | 6 | — | — | ✓ | RFC examples | xfail(strict=True) |
| **grid_scheduler** | 1 | 2 | ✓ | — | — | Task dispatch | Core context mgmt |
| **mlir_frontend/adapt** | 3 | ~110 | — | ✓ | ✓ | Inherit main tests | Mixin override _make_interp() |

**Total**: 881 tests; ~500 unit, ~250 integration, ~130 E2E.

---

## Differential conformance harness (Python ↔ Rust)

Beyond per-subsystem tests, the port's faithfulness to the Python reference is
verified by a **direct differential harness** that runs both interpreters on the
**same seeded inputs** and diffs the outputs:

- Rust CLI: `ktir-emulator/examples/ktir_diff_run.rs` — reads a JSON batch of
  `(program, function, inputs)` cases (tensor args as raw little-endian bytes, or
  HBM/LX stick-seeded for the resident programs), runs the Rust interpreter, and
  writes each result tensor's bytes back plus a `manifest.json`.
- Python driver: `ktir-emulator/tests/equiv/diff_py_vs_rust.py` — seeds numpy
  inputs, runs the Python `KTIRInterpreter`, stages the identical bytes for the
  Rust CLI (one invocation per batch), then computes per-output `max-abs(Python −
  Rust)` against tolerances (integers/most f16 paths are bit-exact).

Across **all 19 shared example programs**: **17 bit-exact PASS + 2
matched-failure (intentional error fixtures where both sides raise the same
normalized error), 0 gaps**. The full table lives in **`rust/PERFORMANCE.md`**
under "Python ↔ Rust conformance".

That CPU/AMX path is bit-exact; the **Metal fast path** (NAX/simdgroup GEMM + fused
map) is covered separately and **tolerance-banded** (NAX uses bf16, so it cannot be
bit-exact with Python f16). `KTIR_DIFF_RESIDENT=1` drives every program through the
resident/segmented Metal executor (`resident_runner.rs` builds a per-kernel
`ProgramSpec`); `KTIR_DIFF_GPU=1` uses the per-op GPU path; both force every offload
(GEMM gate, the `scf.for`-descending map-window, fused attention — all gated behind
`KTIR_FORCE_GPU_*` so the production path is byte-identical) and assert an
`OffloadProof > 0` so a silent CPU fallback FAILS. All 9 Metal-eligible programs
conform within the band; gated test `tests/metal_conformance.rs`.