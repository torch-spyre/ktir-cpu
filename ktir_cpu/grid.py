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

from typing import Dict, List, Optional, Set, Tuple, Any
from .ir_types import Tile
from .memory import LXScratchpad, SpyreMemoryHierarchy, HBMSimulator


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
        # At yield: save %m_new, %l_new. pop_scope() frees body LX.
        # Re-bind %m_acc = %m_new in parent scope, re-track LX.
    """

    def __init__(self, core_id: int, grid_pos: Tuple[int, int, int], lx: LXScratchpad, hbm: HBMSimulator):
        self.core_id = core_id
        self.grid_pos = grid_pos  # (x, y, z) position in grid
        self.lx = lx  # Core-local LX scratchpad
        self.hbm = hbm  # Shared HBM
        self._scope_stack: List[Dict[str, Any]] = [{}]  # bottom = function body
        self._lx_bytes: Dict[str, int] = {}  # SSA name -> LX bytes (single source of truth)

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
    # ------------------------------------------------------------------

    def push_scope(self):
        """Enter a new region scope (scf.for body, scf.if branch)."""
        self._scope_stack.append({})

    def pop_scope(self):
        """Exit the current region scope.

        Discards all SSA values in the topmost scope and frees their
        LX via ``untrack_lx``.
        """
        if len(self._scope_stack) <= 1:
            raise RuntimeError("Cannot pop the function-body scope")
        scope = self._scope_stack.pop()
        for name in scope:
            self.untrack_lx(name)

    # ------------------------------------------------------------------
    # SSA value access
    # ------------------------------------------------------------------

    def set_value(self, name: str, value: Any):
        """Set SSA value in the topmost scope.

        Multiple SSA names may legally map to the same underlying tensor object
        (e.g. when ``linalg.reduce`` binds its result to both the SSA result name
        and the ``outs`` buffer name).  This is valid because Python assignment
        creates a second reference to the same object — it does not copy the data.
        If you need to detect aliasing, compare ``id(a) == id(b)`` on the values.
        """
        self._scope_stack[-1][name] = value

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
        self._lx_bytes.clear()
        self.lx.clear()

    # ------------------------------------------------------------------
    # LX tracking — single source of truth for lx.used
    # ------------------------------------------------------------------

    def track_lx(self, name: str, size_bytes: int):
        """Record that SSA value *name* occupies *size_bytes* in LX.

        Increments ``lx.used``.  Raises ``MemoryError`` if the
        allocation would exceed the 2 MB LX capacity.
        """
        if self.lx.used + size_bytes > self.lx.capacity:
            raise MemoryError(
                f"LX scratchpad overflow on core {self.core_id}: "
                f"requested {size_bytes} bytes, "
                f"available {self.lx.capacity - self.lx.used} bytes "
                f"(capacity {self.lx.capacity} bytes)"
            )
        self._lx_bytes[name] = size_bytes
        self.lx.used += size_bytes

    def untrack_lx(self, name: str):
        """Free LX for SSA value *name*.  Decrements ``lx.used``."""
        if name in self._lx_bytes:
            self.lx.used -= self._lx_bytes.pop(name)


class GridExecutor:
    """Manages execution across grid of cores.

    Handles sequential simulation of cores with communication support.
    """

    def __init__(self, grid_shape: Tuple[int, int, int], memory: SpyreMemoryHierarchy):
        self.grid_shape = grid_shape
        self.memory = memory
        self.num_cores = grid_shape[0] * grid_shape[1] * grid_shape[2]

        # Create core contexts
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

    def execute_with_communication(self, execute_fn, ring_network, max_rounds=10, *args, **kwargs):
        """Execute function on all cores with communication support.

        Uses multi-pass execution to handle inter-core communication.

        Args:
            execute_fn: Function to execute
            ring_network: RingNetwork instance for communication
            max_rounds: Maximum communication rounds
            *args, **kwargs: Additional arguments

        Returns:
            List of results from each core
        """
        for round_num in range(max_rounds):
            # Execute all cores
            results = []
            for core in self.cores:
                result = execute_fn(core, ring_network, *args, **kwargs)
                results.append(result)

            # Deliver messages
            ring_network.deliver_all()

            # Check if communication is done
            if not ring_network.has_pending_messages():
                return results

        # Communication didn't converge
        raise RuntimeError(
            f"Communication didn't converge after {max_rounds} rounds. "
            f"Still have {ring_network.pending_message_count()} pending messages."
        )
