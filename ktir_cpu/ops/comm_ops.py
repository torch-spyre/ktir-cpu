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
Communication compute helpers.

Inter-core tile transfer and ring-based reduction primitives used by
dialect handlers in ``ktir_cpu.dialects``.
"""

from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from ..ir_types import Tile
from ..grid import CoreContext
from ..memory import LXScratchpad, SpyreMemoryHierarchy


class RingBackend:
    """Abstract backend for inter-core LX access and ring communication.

    The backend is the single seam between memory operations that need
    remote LX data and the communication model used to obtain it.

    Two implementations are planned:

    DirectLXBackend (this file, first pass):
        Returns the target scratchpad directly from SpyreMemoryHierarchy.
        No ring messages, no latency modelling.  Valid when LX partitions
        are pre-seeded by the host before kernel execution (e.g. the RFC
        distributed-view-copy test).

    RingTransferBackend (future):
        When a core requests remote LX data, it issues a send on
        RingNetwork and suspends execution at that point (coroutine model).
        The scheduler resumes the core once the sender has delivered the
        message.  This models actual ring hop cost and enables correct
        cyclic communication.
    """

    def get_lx(self, core_id: int) -> LXScratchpad:
        """Return the LXScratchpad for *core_id*."""
        raise NotImplementedError

    def send(self, src_core: int, dst_core: int, tile: Tile) -> None:
        """Send *tile* from *src_core* to *dst_core* via the ring."""
        raise NotImplementedError

    def recv(self, src_core: int, dst_core: int) -> Optional[Tile]:
        """Receive a tile sent from *src_core* to *dst_core*."""
        raise NotImplementedError


class DirectLXBackend(RingBackend):
    """First-pass backend: direct scratchpad access, no ring modelling.

    get_lx(N) returns memory.lx_scratchpads[N] directly — valid for the
    distributed-view use case where LX partitions are pre-seeded by the
    host and no cross-core data movement occurs at runtime.

    send/recv delegate to the provided RingNetwork so that ktdp.transfer
    and ktdp.reduce continue to work as before.
    """

    def __init__(self, memory: SpyreMemoryHierarchy, ring: Optional["RingNetwork"] = None):
        self._memory = memory
        self._ring = ring

    def get_lx(self, core_id: int) -> LXScratchpad:
        """Return the LXScratchpad for *core_id* via direct lookup."""
        num = self._memory.num_cores
        if core_id < 0 or core_id >= num:
            raise ValueError(
                f"get_lx: core_id={core_id} is out of range [0, {num}) for this grid"
            )
        return self._memory.get_lx(core_id)

    def send(self, src_core: int, dst_core: int, tile: Tile) -> None:
        if self._ring is None:
            raise NotImplementedError(
                "DirectLXBackend was constructed without a RingNetwork — "
                "ktdp.transfer/reduce require one"
            )
        self._ring.send(src_core, dst_core, tile)

    def recv(self, src_core: int, dst_core: int) -> Optional[Tile]:
        if self._ring is None:
            raise NotImplementedError(
                "DirectLXBackend was constructed without a RingNetwork — "
                "ktdp.transfer/reduce require one"
            )
        return self._ring.recv_from(src_core, dst_core)


class RingNetwork:
    """Simulates inter-core ring network topology.

    Spyre's 32 cores are connected via two rings:
    - One clockwise ring (4 TB/s)
    - One anti-clockwise ring (4 TB/s)

    Both rings connect all 32 cores. We simulate a single logical ring
    since directionality doesn't affect correctness in CPU simulation.
    """

    def __init__(self, num_cores: int):
        self.num_cores = num_cores
        self.message_buffers: Dict[Tuple[int, int], List[Tile]] = {}  # (src, dst) -> [tiles]
        self.pending_messages = []

    def send(self, src_core: int, dst_core: int, tile: Tile):
        """Send tile from source core to destination core.

        Messages are buffered until deliver_all() is called.

        Args:
            src_core: Source core ID
            dst_core: Destination core ID
            tile: Tile to send
        """
        key = (src_core, dst_core)
        if key not in self.message_buffers:
            self.message_buffers[key] = []
        self.message_buffers[key].append(tile.copy())
        self.pending_messages.append(key)

    def recv_from(self, src_core: int, dst_core: int) -> Optional[Tile]:
        """Receive tile from specific source.

        Args:
            src_core: Source core ID
            dst_core: Destination core ID (receiver)

        Returns:
            Tile if available, None otherwise
        """
        key = (src_core, dst_core)
        if key in self.message_buffers and self.message_buffers[key]:
            return self.message_buffers[key].pop(0)
        return None

    def deliver_all(self):
        """Deliver all pending messages.

        In sequential simulation, this is a no-op since messages
        are immediately available after send().
        """
        self.pending_messages.clear()

    def has_pending_messages(self) -> bool:
        """Check if there are pending messages."""
        return any(len(msgs) > 0 for msgs in self.message_buffers.values())

    def pending_message_count(self) -> int:
        """Count pending messages."""
        return sum(len(msgs) for msgs in self.message_buffers.values())

    def clear(self):
        """Clear all messages."""
        self.message_buffers.clear()
        self.pending_messages.clear()


class CommOps:
    """Inter-core transfer and reduction helpers."""

    @staticmethod
    def transfer(
        context: CoreContext,
        tile: Tile,
        dst_cores: List[int],
        ring_backend: RingBackend,
    ):
        """Transfer a tile to destination cores via the ring backend."""
        for dst_core in dst_cores:
            ring_backend.send(context.core_id, dst_core, tile)

    @staticmethod
    def reduce(
        context: CoreContext,
        tile: Tile,
        core_group: List[int],
        reduce_fn: Callable[[Tile, Tile], Tile],
        ring_backend: RingBackend,
    ) -> Tuple[Tile, bool]:
        """Reduce a tile across a core group using the ring network.

        Args:
            context: Core execution context
            tile: Local tile value to reduce
            core_group: List of core IDs participating in reduction
            reduce_fn: Reduction function (e.g., add, max)
            ring: Ring network

        Returns:
            (reduced_result, has_result) tuple
            - reduced_result: The reduced tile
            - has_result: True if this core has the final result

        Note:
            Only cores in core_group participate. Others return dummy result.
        """
        # Check if this core participates
        if context.core_id not in core_group:
            # Not participating, return dummy
            dummy_data = np.zeros_like(tile.data)
            return (Tile(dummy_data, tile.dtype, tile.shape), False)

        # Participating in reduction
        n_cores = len(core_group)
        my_idx = core_group.index(context.core_id)

        # Ring reduction algorithm
        # Each round, send to next in ring and receive from previous
        result = tile.copy()

        num_rounds = int(np.ceil(np.log2(n_cores))) if n_cores > 1 else 0

        for round_num in range(num_rounds):
            # Send to next core in ring
            next_idx = (my_idx + 1) % n_cores
            next_core = core_group[next_idx]
            ring_backend.send(context.core_id, next_core, result)

            # Receive from previous core in ring
            prev_idx = (my_idx - 1) % n_cores
            prev_core = core_group[prev_idx]

            # In sequential simulation, message is immediately available
            received = ring_backend.recv(prev_core, context.core_id)

            # Apply reduction
            if received is not None:
                result = reduce_fn(result, received)

        # Determine if this core has final result
        # Typically, first core in group gets the final result
        has_result = (my_idx == 0)

        return (result, has_result)

    @staticmethod
    def reduce_return(value: Tile) -> Tile:
        """Return a value from a reduction block (identity passthrough).

        Args:
            value: Value to return

        Returns:
            The value
        """
        return value
