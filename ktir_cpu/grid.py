# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Grid and core execution management for KTIR CPU backend.

Simulates multi-core grid execution with per-core execution contexts.
"""

import inspect
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Set, Tuple, Any
from .ir_types import Tile
from .memory import LXScratchpad, SpyreMemoryHierarchy, HBMSimulator

if TYPE_CHECKING:
    from .ops.comm_ops import TransferBackend


@dataclass(frozen=True)
class RecvRequest:
    """Yielded by a comm generator to signal a blocking recv.

    The scheduler parks the core until a tile from *src* is available,
    then resumes the generator via ``gen.send(tile)``.
    """
    src: int  # core ID to receive from

class CoreContext:
    """Execution context for a single core.

    Maintains per-core state including grid position, LX scratchpad,
    and a region-scoped SSA value map.  Each MLIR region (function body,
    scf.for body, scf.if branch) gets its own scope on the stack.
    ``set_value`` writes to the topmost scope; ``get_value`` searches
    top-to-bottom so inner regions can read outer values.

    ``_scope_stack`` example during softmax_rowchunk execution::

        # Function entry (after constants, before loop):
        _scope_stack = [
          {"%core_id": 0, "%c32": 32, "%input_view": MemRef(...), ...}
        ]
        # One scope — the function body.

        # Inside scf.for body (after loading tile):
        _scope_stack = [
          {"%core_id": 0, "%c32": 32, "%input_view": MemRef(...)},  # function
          {"%row": 0, "%tile": Tile(32x1024), "%row_max": Tile(32x1)}  # body
        ]
        # get_value("%input_view") finds it in scope[0] — inner sees outer.

        # After pop_scope() at end of iteration:
        _scope_stack = [
          {"%core_id": 0, "%c32": 32, "%input_view": MemRef(...), ...}
        ]
        # Body scope popped. %tile, %row_max freed from LX.

        # clear_values() between function invocations:
        _scope_stack = [{}]
        # Reset to single empty scope — ready for next core.

    For ``scf.for`` with ``iter_args`` (e.g. online softmax), the
    iter_arg values live in the *parent* scope and are re-bound after
    each yield::

        _scope_stack = [
          {"%m_acc": Tile(32x1), "%l_acc": Tile(32x1), ...},  # function
          {"%col": 0, "%tile": Tile(32x256), "%m_new": Tile(32x1)}  # body
        ]
        # At yield: save %m_new, %l_new. pop_scope() decrements body Tile refcounts.
        # set_value("%m_acc", %m_new) in parent scope: refcount tracks automatically.
    """

    def __init__(self, core_id: int, grid_pos: Tuple[int, int, int], lx: LXScratchpad, hbm: HBMSimulator):
        self.core_id = core_id
        self.grid_pos = grid_pos  # (x, y, z) position in grid
        self.lx = lx              # Core-local LX scratchpad
        self.hbm = hbm            # Shared HBM
        self._scope_stack: List[Dict[str, Any]] = [{}]  # bottom = function body
        self._tile_refcount: Dict[int, int] = {}  # id(Tile) -> refcount
        self._lx_next_ptr_stack: List[int] = []  # watermarks for lx.next_ptr rewind on pop_scope
        # Scheduler-managed cross-core access — both functions are set by
        # GridExecutor.execute_with_communication via attach_scheduler() and
        # cleared via detach_scheduler() at the end of the run.
        # NOTE: these private fields back the public send_to() and get_lx()
        # methods. The naming keeps "ring_*" out of the type — see
        # docs/cross_core_scheduling.md for the planned rename of the
        # ring_backend → transfer_backend nomenclature site by site.
        self._send_fn: Optional[Callable[[int, "Tile"], None]] = None
        self._transfer_fn: Optional[Callable[[int], LXScratchpad]] = None
        self._num_cores: Optional[int] = None

    def attach_scheduler(
        self,
        send_fn: Callable[[int, "Tile"], None],
        transfer_fn: Callable[[int], LXScratchpad],
        num_cores: int,
    ) -> None:
        """Wire scheduler-managed cross-core access for the duration of a run.

        Called by ``GridExecutor.execute_with_communication`` before
        stepping this core's generator. ``send_fn`` enqueues a tile into
        the scheduler's message queue; ``transfer_fn(src_core)`` returns
        the LXScratchpad for a remote core (only invoked for genuinely
        remote cores — local fast path stays in ``get_lx``).
        ``num_cores`` is the workgroup size; backends read it via
        :attr:`num_cores` to size their protocols (e.g. ``RingReduceBackend``
        runs ``num_cores - 1`` rounds).

        The binding is valid until ``detach_scheduler`` is called.
        """
        self._send_fn = send_fn
        self._transfer_fn = transfer_fn
        self._num_cores = num_cores

    def detach_scheduler(self) -> None:
        """Clear the scheduler bindings installed by :meth:`attach_scheduler`."""
        self._send_fn = None
        self._transfer_fn = None
        self._num_cores = None

    @property
    def num_cores(self) -> int:
        """Workgroup size, available while a scheduler is attached.

        Distinct from ``HardwareConfig.num_cores`` — that one models
        the chip's per-core HBM bandwidth divisor and stays where it
        is.  This is the live workgroup size the IR's ``grid``
        attribute names.
        """
        if self._num_cores is None:
            raise RuntimeError(
                f"CoreContext(core_id={self.core_id}).num_cores accessed "
                f"without an attached scheduler"
            )
        return self._num_cores

    def get_lx(self, core_id: Optional[int] = None) -> LXScratchpad:
        """Return the LXScratchpad for *core_id*.

        When *core_id* is None or equals this core's own id, returns the
        local scratchpad directly.  For a remote core, delegates to the
        attached transfer function — raises if no scheduler is attached.
        """
        if core_id is None or core_id == self.core_id:
            return self.lx
        if self._transfer_fn is None:
            raise RuntimeError(
                f"CoreContext(core_id={self.core_id}).get_lx requested remote "
                f"core {core_id} but no scheduler is attached. "
                f"GridExecutor.execute_with_communication must run first."
            )
        return self._transfer_fn(core_id)

    def get_grid_id(self, dim: int) -> int:
        """Get grid ID for dimension.

        Args:
            dim: Dimension (0=x, 1=y, 2=z)

        Returns:
            Grid coordinate in that dimension
        """
        return self.grid_pos[dim]

    # ------------------------------------------------------------------
    # Region scoping
    # LX allocation model: scope-watermark bump allocator
    # LX address space is managed as a bump allocator with scope-level watermarks.
    # push_scope snapshots lx.next_ptr; pop_scope rewinds it. This is the
    # most aggressive deallocation strategy the IR permits without liveness analysis:

    # 1). SSA values are immutable bindings with scope-lifetime semantics. A value
    # that is "in scope" is reachable — there is no point in the IR where a value
    # is dead but still in scope. So scope-exit is the earliest safe deallocation
    # point.

    # 2). Intra-scope reuse is not possible because all tiles produced within a scope
    # coexist simultaneously in LX until the scope pops. Peak LX usage within a
    # scope equals the sum of all tiles produced in that scope, even if some are
    # logically "consumed" by a downstream op and never read again.

    # 3). A free-list or explicit dealloc op would require liveness analysis or
    # last-use annotations, neither of which exist in KTIR's current model
    # (compute uses standard MLIR dialects; SCF regions define lifetime).

    # The watermark stack stays in lockstep with _scope_stack — the invariant
    # len(_lx_next_ptr_stack) == len(_scope_stack) - 1 holds by construction.
    # ------------------------------------------------------------------

    def push_scope(self):
        """Enter a new region scope (scf.for body, scf.if branch).

        Snapshots ``lx.next_ptr`` so ``pop_scope`` can rewind it and
        keep the bump-pointer in lockstep with ``lx.used``.
        """
        self._lx_next_ptr_stack.append(self.lx.next_ptr)
        self._scope_stack.append({})
        assert len(self._lx_next_ptr_stack) == len(self._scope_stack) - 1

    def pop_scope(self):
        """Exit the current region scope.

        Discards all SSA values in the topmost scope, decrements refcounts
        for any Tiles, and rewinds ``lx.next_ptr`` to its pre-push value so
        the bump-pointer reclaims address space alongside the byte counter.
        """
        if len(self._scope_stack) <= 1:
            raise RuntimeError("Cannot pop the function-body scope")
        scope = self._scope_stack.pop()
        for value in scope.values():
            if isinstance(value, Tile):
                self._tile_refcount[id(value)] -= 1
                if self._tile_refcount[id(value)] == 0:
                    del self._tile_refcount[id(value)]
                    self.lx.used -= value.size_bytes()
        self.lx.next_ptr = self._lx_next_ptr_stack.pop()
        assert len(self._lx_next_ptr_stack) == len(self._scope_stack) - 1

    def send_to(self, dst_core: int, tile: "Tile") -> None:
        """Enqueue *tile* for delivery to *dst_core*.

        Wired by the scheduler via :meth:`attach_scheduler` before
        stepping this core's generator. Raises if called outside a
        scheduler-managed execution (e.g. single-core tests that never
        attach).
        """
        if self._send_fn is None:
            raise RuntimeError(
                f"CoreContext(core_id={self.core_id}).send_to called without "
                f"a scheduler — GridExecutor.execute_with_communication must run first."
            )
        self._send_fn(dst_core, tile)

    # ------------------------------------------------------------------
    # SSA value access
    # ------------------------------------------------------------------

    def set_value(self, name: str, value: Any):
        """Set SSA value in the topmost scope, auto-tracking Tile LX usage.

        - First binding of a Tile id: charges lx.used once.
        - Subsequent bindings (aliases, iter_arg rebinds): refcount increments, no double-charge.
        - Overwriting an existing name: decrements refcount of the old Tile.

        Multiple SSA names may legally map to the same underlying tensor object
        (e.g. when ``linalg.reduce`` binds its result to both the SSA result name
        and the ``outs`` buffer name).  This is valid because Python assignment
        creates a second reference to the same object — it does not copy the data.
        """
        old = self._scope_stack[-1].get(name)
        if isinstance(old, Tile):
            self._tile_refcount[id(old)] -= 1
            if self._tile_refcount[id(old)] == 0:
                del self._tile_refcount[id(old)]
                self.lx.used -= old.size_bytes()
        self._scope_stack[-1][name] = value
        if isinstance(value, Tile):
            if self._tile_refcount.get(id(value), 0) == 0:
                if self.lx.used + value.size_bytes() > self.lx.capacity:
                    raise MemoryError(
                        f"LX scratchpad overflow on core {self.core_id}: "
                        f"requested {value.size_bytes()} bytes, "
                        f"available {self.lx.capacity - self.lx.used} bytes "
                        f"(capacity {self.lx.capacity} bytes)"
                    )
                self.lx.used += value.size_bytes()
            self._tile_refcount[id(value)] = self._tile_refcount.get(id(value), 0) + 1

    def get_value(self, name: str) -> Any:
        """Get SSA value, searching top-to-bottom.

        Args:
            name: SSA value name (e.g., "%x_tile")

        Returns:
            The value

        Raises:
            KeyError: If value not found in any scope
        """
        for scope in reversed(self._scope_stack):
            if name in scope:
                return scope[name]
        raise KeyError(f"Value '{name}' not found in core {self.core_id}")

    def has_value(self, name: str) -> bool:
        """Check if value exists in any scope."""
        return any(name in scope for scope in self._scope_stack)

    def clear_values(self):
        """Clear all scopes and LX (for next round of execution)."""
        self._scope_stack = [{}]
        self._tile_refcount.clear()
        self._lx_next_ptr_stack.clear()
        self.lx.clear()

    # ------------------------------------------------------------------
    # LX tracking — single source of truth for lx.used
    # Tracking is now automatic via set_value / pop_scope.
    # ------------------------------------------------------------------


class CoreExecutionStack:
    """Represents one core's execution state as a resumable generator.

    ``execute_until_block`` runs ops via *execute_op* until either all ops
    complete (done) or a comm op returns a generator that yields
    ``('recv', src)`` (blocked).  Generator-awareness is contained here —
    callers and dialect handlers never see generators.

    After ``resume()`` returns, check ``is_blocked()`` to know whether the
    core is waiting on a recv or has finished.  If blocked, ``waiting_on``
    holds the src core ID to wait for.
    """

    def __init__(
        self,
        core: "CoreContext",
        operations: List["Operation"],
        input_ptrs: Dict[str, Any],
        execute_op: Callable[["Operation", "CoreContext"], Any],
    ):
        self.core = core
        self.waiting_on: Optional[int] = None
        self._gen = self._execute_until_block(operations, input_ptrs, execute_op)

    def _execute_until_block(self, operations, input_ptrs, execute_op):
        for k, v in input_ptrs.items():
            self.core.set_value("%" + k, v)
        result = None
        for op in operations:
            result = execute_op(op, self.core)
            if inspect.isgenerator(result):
                # Comm op returned a generator — drive it, yielding recv
                # requests up to the scheduler one at a time.
                result = yield from result
                # Post-resume: store the final result (execute_op couldn't,
                # it returned the generator before the value was known).
                self._store(op, result, self.core)
        return result

    def _store(self, op: "Operation", result: Any, context: "CoreContext") -> None:
        """Bind *result* into *context* for *op* (mirrors interpreter logic)."""
        if op.result and result is not None:
            if isinstance(op.result, list) and isinstance(result, tuple):
                for name, val in zip(op.result, result):
                    context.set_value(name, val)
            else:
                context.set_value(op.result, result)

    def resume(self, send_val: Any = None) -> Any:
        """Step the generator.  Returns the final value when done."""
        self.waiting_on = None
        try:
            request = self._gen.send(send_val) if send_val is not None else next(self._gen)
            if not isinstance(request, RecvRequest):
                raise TypeError(f"Comm op yielded unexpected type {type(request)!r}; expected RecvRequest")
            self.waiting_on = request.src
            return None
        except StopIteration as e:
            return e.value

    def is_blocked(self) -> bool:
        return self.waiting_on is not None


class GridExecutor:
    """Manages execution across grid of cores.

    Handles sequential simulation of cores with communication support.
    """

    def __init__(self, grid_shape: Tuple[int, int, int], memory: SpyreMemoryHierarchy):
        self.grid_shape = grid_shape
        self.memory = memory
        self.num_cores = grid_shape[0] * grid_shape[1] * grid_shape[2]

        # CoreContexts hold per-core local state. The transfer backend
        # used to satisfy remote ``get_lx`` is supplied at execution
        # time via ``execute_with_communication`` and bound onto each
        # core through ``attach_scheduler``.
        self.cores = []
        for core_id in range(self.num_cores):
            grid_pos = self._linear_to_grid(core_id)
            lx = memory.get_lx(core_id)
            self.cores.append(CoreContext(core_id, grid_pos, lx, memory.hbm))

    def _linear_to_grid(self, core_id: int) -> Tuple[int, int, int]:
        """Convert linear core ID to (x, y, z) grid position.

        Args:
            core_id: Linear core ID

        Returns:
            (x, y, z) position
        """
        nx, ny, nz = self.grid_shape
        z = core_id // (nx * ny)
        remainder = core_id % (nx * ny)
        y = remainder // nx
        x = remainder % nx
        return (x, y, z)

    def _grid_to_linear(self, x: int, y: int, z: int) -> int:
        """Convert (x, y, z) grid position to linear core ID.

        Args:
            x, y, z: Grid coordinates

        Returns:
            Linear core ID
        """
        nx, ny, nz = self.grid_shape
        return z * (nx * ny) + y * nx + x

    def get_core(self, core_id: int) -> CoreContext:
        """Get core context by ID."""
        return self.cores[core_id]

    def get_core_at_pos(self, x: int, y: int, z: int = 0) -> CoreContext:
        """Get core context at grid position."""
        core_id = self._grid_to_linear(x, y, z)
        return self.cores[core_id]

    def get_cores_in_group(self, grid_coords: Tuple[int, int, int]) -> List[int]:
        """Get list of core IDs matching grid coordinates.

        Args:
            grid_coords: (x, y, z) where -1 means "all in that dimension"

        Returns:
            List of core IDs

        Example:
            get_cores_in_group((-1, 2, 0)) returns all cores with y=2, z=0
            get_cores_in_group((5, -1, 0)) returns all cores with x=5, z=0
        """
        nx, ny, nz = self.grid_shape
        target_x, target_y, target_z = grid_coords

        core_ids = []
        for core_id in range(self.num_cores):
            x, y, z = self._linear_to_grid(core_id)

            # Check if this core matches
            if target_x != -1 and x != target_x:
                continue
            if target_y != -1 and y != target_y:
                continue
            if target_z != -1 and z != target_z:
                continue

            core_ids.append(core_id)

        return core_ids

    def execute_with_communication(
        self,
        operations: List["Operation"],
        input_ptrs: Dict[str, Any],
        execute_op: Callable[["Operation", "CoreContext"], Any],
        transfer_backend: Optional["TransferBackend"] = None,
    ) -> List[Any]:
        """Drive all cores to completion via the generator scheduler.

        Each core is represented by a ``CoreExecutionStack``.  The stack
        runs ops synchronously until a comm op returns a generator (a
        blocking recv).  The scheduler parks that core, delivers the
        message when it arrives, and resumes.  Unresolvable waits raise
        ``RuntimeError('Deadlock detected: ...')``.

        Before stepping any core, attaches per-core scheduler state via
        ``CoreContext.attach_scheduler``: a send function (queues onto
        the scheduler's message buffer) and a transfer function
        (resolves remote ``ctx.get_lx``). When *transfer_backend* is
        None, the transfer function raises if invoked — fine for
        single-core or all-local executions.
        """
        if not operations:
            return [None] * self.num_cores
        if not callable(execute_op):
            raise ValueError(
                f"execute_op must be callable, got {type(execute_op)!r}"
            )
        if input_ptrs is None:
            input_ptrs = {}
        messages: Dict[Tuple[int, int], deque] = {}
        stacks: Dict[int, "CoreExecutionStack"] = {}
        waiting: Dict[int, int] = {}   # core_id -> src_core it is waiting on
        results: Dict[int, Any] = {}

        def _enqueue(src: int, dst: int, tile: "Tile") -> None:
            messages.setdefault((src, dst), deque()).append(tile)

        def _pop(src: int, dst: int) -> Optional["Tile"]:
            q = messages.get((src, dst))
            if not q:
                return None
            tile = q.popleft()
            if not q:
                del messages[(src, dst)]
            return tile

        def _advance(core_id: int, send_val: Any = None) -> None:
            stack = stacks[core_id]
            try:
                result = stack.resume(send_val)
            except Exception as e:
                e.add_note(f"  [core {core_id}]")
                raise
            if stack.is_blocked():
                waiting[core_id] = stack.waiting_on
            else:
                results[core_id] = result
                del stacks[core_id]

        def _try_deliver(core_id: int) -> bool:
            src = waiting.get(core_id)
            if src is None:
                return False
            tile = _pop(src, core_id)
            if tile is None:
                return False
            del waiting[core_id]
            _advance(core_id, tile)
            return True

        for core in self.cores:
            send_fn = (lambda dst, tile, _src=core.core_id:
                       _enqueue(_src, dst, tile))
            if transfer_backend is not None:
                transfer_fn = (lambda src, _ctx=core, _bk=transfer_backend:
                               _bk.run(_ctx, src))
            else:
                def transfer_fn(src):  # noqa: E306 — explicit error message
                    raise RuntimeError(
                        "transfer_fn invoked but no transfer_backend was "
                        "passed to execute_with_communication."
                    )
            core.attach_scheduler(
                send_fn=send_fn,
                transfer_fn=transfer_fn,
                num_cores=self.num_cores,
            )
            stacks[core.core_id] = CoreExecutionStack(core, operations, input_ptrs, execute_op)
            _advance(core.core_id)

        while stacks:
            if not any(_try_deliver(c) for c in tuple(stacks)):
                wait_desc = "; ".join(
                    f"core {c} waiting on recv from core {s}"
                    for c, s in waiting.items()
                )
                raise RuntimeError(f"Deadlock detected: {wait_desc}")

        return [results[i] for i in range(self.num_cores)]
