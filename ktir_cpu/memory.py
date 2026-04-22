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
Memory simulation for Spyre hardware.

Simulates the memory hierarchy:
- HBM: 128GB High Bandwidth Memory (shared across all cores)
- LX: 2MB per-core scratchpad memory

Design notes — memory-space-aware load/store
=============================================

Two memory spaces
-----------------

- **HBM** (128 GB, shared) holds host-provided input and output tensors.
  All function arguments (``%input_ptr``, ``%output_ptr``, …) are
  addresses in HBM.  Kernels never allocate new HBM themselves.

- **LX** (2 MB per core) holds all live SSA tensor values.  Every
  ``Tile`` produced by ``ktdp.load`` or by a compute operation (arith,
  math, linalg) resides in LX.

Operations (implemented in ``memory_ops.py``)
----------------------------------------------

1. **``construct_memory_view`` / ``construct_access_tile``** create
   metadata (``TileRef`` / ``AccessTile``) that carry a ``memory_space``
   ("HBM" or "LX").  No data movement occurs.

2. **``ktdp.load``** — ``MemoryOps.load`` inspects
   ``tile_ref.memory_space``:

   - Source is HBM → read from HBM, copy into LX.
   - Source is LX → read directly from LX (no DMA).

3. **``ktdp.store``** — ``MemoryOps.store`` inspects
   ``tile_ref.memory_space``:

   - Target is HBM → write from LX to HBM.
   - Target is LX → write directly to LX (no HBM involved).

LX lifetime
------------

Each SSA ``Tile`` value occupies LX from the point it is created
(via ``ktdp.load`` or a compute op) until the defining region ends.
``CoreContext`` uses a scope stack (``_scope_stack``) that mirrors
MLIR's region structure.  When a region exits (``pop_scope``), all
SSA values defined in that scope are discarded and their LX is freed
via ``untrack_lx``.  Values returned via ``scf.yield`` (iter_args)
are re-bound in the parent scope and their LX is re-tracked.

The 2 MB LX limit is the real constraint: it determines how large a
tile can be loaded, and how many tiles can coexist in LX at any one
time within a single iteration.

Freeing LX on store
~~~~~~~~~~~~~~~~~~~~

One could free a ``Tile``'s LX allocation when it is stored to HBM
(since the data now lives in HBM and the SSA value is consumed).
However this gets complicated: the same SSA value may be read again
after the store, and tracking last-use requires liveness analysis.
We choose not to do this — if an SSA value is referenced again after
a store, it is already in LX and no reallocation is needed.

Note: on real hardware, the compiler performs scratchpad planning to
optimise LX allocation (e.g. reusing buffers, pinning statistics
between kernels).  This is not modelled here — we simply track the
live set of SSA tensor values as the LX footprint.
"""

from typing import Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers shared by HBMSimulator and LXScratchpad
# ---------------------------------------------------------------------------

def _get_np_dtype(dtype: str) -> np.dtype:
    """Convert a KTIR dtype string to a NumPy dtype."""
    if dtype in ("f16", "fp16", "float16"):
        return np.float16
    if dtype in ("i32", "si32", "index"):
        return np.int32
    if dtype in ("i64", "si64"):
        return np.int64
    return np.float32  # f8 / mxfp8 / others approximated as float32


def _bytes_per_elem(dtype: str) -> int:
    """Return element size in bytes for a KTIR dtype string."""
    return int(np.dtype(_get_np_dtype(dtype)).itemsize)


def _find_allocation(
    memory: Dict[int, np.ndarray],
    ptr: int,
    bytes_per_elem: int,
) -> Optional[Tuple[int, np.ndarray, int]]:
    """Find the allocation containing byte address *ptr*.

    Returns ``(base_ptr, array, elem_offset)`` where *elem_offset* is the
    flat element index of *ptr* within *array*, or ``None`` if no allocation
    covers *ptr*.

    This is the single place where byte addresses are translated to array
    indices.  All read/write helpers call this rather than doing their own
    address arithmetic.
    """
    if ptr in memory:
        return (ptr, memory[ptr], 0)
    for base_ptr, data in memory.items():
        # Use the allocation's own itemsize to compute its byte span,
        # not the caller's bytes_per_elem (which reflects the access dtype
        # and may differ from the stored dtype).
        end_ptr = base_ptr + data.size * data.itemsize
 
        # NOTE: the ptr in memory check in the first return
        #       actually handles the ptr == base_ptr case.
        # - therefore the strict equality base_ptr < ptr
        #   is meant to emphasize that equality checks will 
        #   not come to this branch of logic.
        if base_ptr < ptr < end_ptr:
            elem_offset = (ptr - base_ptr) // bytes_per_elem
            return (base_ptr, data, elem_offset)
    return None


def _read_flat(
    memory: Dict[int, np.ndarray],
    ptr: int,
    n_elements: int,
    np_dtype: np.dtype,
    bytes_per_elem: int,
) -> np.ndarray:
    """Read *n_elements* elements starting at byte address *ptr*.

    Returns a flat array of length *n_elements*.  Elements beyond the end of
    the containing allocation are zero-padded.  Raises ``ValueError`` if *ptr*
    is unmapped.

    Example — reading 13 elements from inside a 4×4 f16 allocation::

        # 4×4 f16 tensor at ptr=0x1000, values 0..15
        memory = {0x1000: np.arange(16, dtype=np.float16)}
        # Read 13 elements starting at element 2 (byte offset 4 from base)
        flat = _read_flat(memory, ptr=0x1004, n_elements=13,
                          np_dtype=np.float16, bytes_per_elem=2)
        # flat == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    """
    alloc = _find_allocation(memory, ptr, bytes_per_elem)
    if alloc is None:
        raise ValueError(f"Read from unmapped address 0x{ptr:x} (n_elements={n_elements})")
    _, data, elem_offset = alloc
    flat = data.flatten()
    end = elem_offset + n_elements
    if end <= flat.size:
        return flat[elem_offset:end].astype(np_dtype, copy=True)
    # Partial allocation — pad remainder with zeros
    result = np.zeros(n_elements, dtype=np_dtype)
    avail = flat.size - elem_offset
    result[:avail] = flat[elem_offset:]
    return result


def _write_flat(memory: Dict[int, np.ndarray], ptr: int, data: np.ndarray):
    """Write *data* (flat ndarray) at byte address *ptr*.

    Patches an existing allocation in-place when *ptr* falls within one.
    Creates a new allocation at *ptr* if unmapped.

    Example — writing a single element into the middle of a 4×4 f16 tensor::

        # 4×4 f16 tensor at ptr=0x1000, all zeros
        memory = {0x1000: np.zeros(16, dtype=np.float16)}
        # Write value 99 at element [1,2] (flat offset 6, byte offset 12)
        _write_flat(memory, ptr=0x100C, data=np.array([99.0], dtype=np.float16))
        # memory[0x1000].reshape(4,4)[1, 2] == 99.0, all other elements unchanged
    """
    bytes_per_elem = data.itemsize
    alloc = _find_allocation(memory, ptr, bytes_per_elem)
    if alloc is not None:
        base_ptr, existing, elem_offset = alloc
        flat = existing.flatten().copy()
        src = data.flatten()
        end_elem = elem_offset + src.size
        if end_elem <= flat.size:
            flat[elem_offset:end_elem] = src
            memory[base_ptr] = flat.reshape(existing.shape)
            return
        # src extends past allocation end — write what fits
        fit = flat.size - elem_offset
        flat[elem_offset:] = src[:fit]
        memory[base_ptr] = flat.reshape(existing.shape)
        return
    memory[ptr] = data.flatten().copy()


class HBMSimulator:
    """Simulates 128GB HBM using sparse storage.

    Uses dict-based sparse storage to avoid allocating full 128GB.

    Note: the ``size_bytes`` capacity is tracked but **not enforced**
    during allocation.  In practice, kernel-level MLIR programs only
    reference tensors that the host has already placed in HBM — the
    kernel never allocates new HBM itself.  Enforcing the limit here
    would require modelling the host-side memory allocator, which is
    outside the scope of the KTIR interpreter.
    """

    def __init__(self, size_gb: int = 128):
        self.size_gb = size_gb
        self.size_bytes = size_gb * 1024 * 1024 * 1024
        self.memory: Dict[int, np.ndarray] = {}  # Sparse storage
        self.next_ptr = 0x10000  # Start allocations at 64KB

    def allocate(self, size: int) -> int:
        """Allocate memory and return pointer.

        Called by ``KTIRInterpreter.execute_function`` to place host input
        tensors in HBM before kernel execution (interpreter.py).

        Args:
            size: Size in bytes

        Returns:
            Pointer (memory address)
        """
        ptr = self.next_ptr
        self.next_ptr += size
        # Align to 128-byte stick boundary
        self.next_ptr = (self.next_ptr + 127) & ~127
        return ptr

    def read(self, ptr: int, n_elements: int, dtype: str) -> np.ndarray:
        """Read *n_elements* elements starting at byte address *ptr*.

        Returns a flat array of length *n_elements*.  Raises ValueError if
        *ptr* is unmapped.

        Callers:
        - ``MemoryOps.load`` — reads tile data from HBM (memory_ops.py)
        - ``KTIRInterpreter.execute_function`` — reads output tensors back
          after execution (interpreter.py)

        Args:
            ptr: Memory address (byte offset)
            n_elements: Number of elements to read
            dtype: Data type

        Returns:
            Flat NumPy array of length n_elements
        """
        np_dtype = _get_np_dtype(dtype)
        return _read_flat(self.memory, ptr, n_elements, np_dtype, _bytes_per_elem(dtype))

    def write(self, ptr: int, data: np.ndarray):
        """Write *data* (flat ndarray) starting at byte address *ptr*.

        Patches an existing allocation in-place when *ptr* falls within one.
        Creates a new allocation at *ptr* if unmapped.

        Callers:
        - ``KTIRInterpreter.execute_function`` — places host input tensors
          in HBM before kernel execution (interpreter.py)
        - ``MemoryOps.store`` — writes tile data to HBM (memory_ops.py)

        Args:
            ptr: Memory address (byte offset)
            data: Flat NumPy array to write
        """
        _write_flat(self.memory, ptr, data)

    def read_element(self, addr: int, dtype: str = "f16"):
        """Read a single element by byte address.

        .. deprecated::
            Use ``read(ptr, 1, dtype)[0]`` instead.  This method will be
            removed when the tt dialect is updated.
        """
        bytes_per_elem = 2 if dtype in ("f16", "fp16", "float16") else 4
        alloc = _find_allocation(self.memory, addr, bytes_per_elem)
        if alloc is None:
            return np.float16(0.0)
        _, data, elem_offset = alloc
        return data.flat[elem_offset]

    def _get_np_dtype(self, dtype: str) -> np.dtype:
        return _get_np_dtype(dtype)


class LXScratchpad:
    """Simulates 2MB per-core scratchpad memory.

    Core-local fast memory with capacity limit.
    """

    def __init__(self, size_mb: int = 2, core_id: int = 0):
        self.size_mb = size_mb
        self.capacity = size_mb * 1024 * 1024
        self.used = 0
        self.core_id = core_id
        self.memory: Dict[int, np.ndarray] = {}
        self.next_ptr = 0  # Local address space

    def read(self, ptr: int, n_elements: int, dtype: str) -> np.ndarray:
        """Read *n_elements* elements starting at byte address *ptr*.

        Returns a flat array of length *n_elements*.  Raises ValueError if
        *ptr* is unmapped.

        Args:
            ptr: Local address (byte offset)
            n_elements: Number of elements to read
            dtype: Data type

        Returns:
            Flat NumPy array of length n_elements
        """
        np_dtype = _get_np_dtype(dtype)
        return _read_flat(self.memory, ptr, n_elements, np_dtype, _bytes_per_elem(dtype))

    def write(self, ptr: int, data: np.ndarray):
        """Write *data* (flat ndarray) starting at byte address *ptr*.

        Patches an existing allocation in-place when *ptr* falls within one.
        Creates a new allocation at *ptr* if unmapped.

        Args:
            ptr: Local address (byte offset)
            data: Flat NumPy array to write
        """
        _write_flat(self.memory, ptr, data)

    def clear(self):
        """Clear scratchpad and reset allocation."""
        self.memory.clear()
        self.next_ptr = 0
        self.used = 0

    def _get_np_dtype(self, dtype: str) -> np.dtype:
        return _get_np_dtype(dtype)


class SpyreMemoryHierarchy:
    """Complete memory hierarchy for Spyre: one shared HBM + per-core LX scratchpads.

    Data movement between HBM and LX is handled by ``MemoryOps.load``
    and ``MemoryOps.store`` in ``memory_ops.py``, which inspect
    ``TileRef.memory_space`` to determine the source/destination.
    """

    def __init__(self, num_cores: int):
        self.num_cores = num_cores
        self.hbm = HBMSimulator()
        self.lx_scratchpads = [LXScratchpad(core_id=i) for i in range(num_cores)]

    def get_lx(self, core_id: int) -> LXScratchpad:
        """Get LX scratchpad for a specific core."""
        return self.lx_scratchpads[core_id]
