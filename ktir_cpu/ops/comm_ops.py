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

from typing import Dict, Generator, List, Tuple, Callable, Optional
import numpy as np
from ..ir_types import Tile
from ..grid import CoreContext, RecvRequest
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
    ) -> Generator:
        """Send *tile* to each core in *dst_cores*, then return.

        Yields nothing — fire-and-forget: the sender does not block.
        The caller (scheduler) wraps sync returns in a one-shot iterator.

        Spec: the receiver blocks on its own recv; the sender is free
        to continue.
        """
        for dst_core in dst_cores:
            context.send_to(dst_core, tile)
        return tile  # result value (non-generator; scheduler wraps it)

    @staticmethod
    def reduce(
        context: CoreContext,
        tile: Tile,
        core_group: List[int],
        reduce_fn: Callable[[Tile, Tile], Tile],
    ) -> Generator:
        """Ring reduction across *core_group* — one core's view.

        Generator. Yields ``RecvRequest`` exactly ``len(core_group) - 1``
        times. After all rounds every participating core holds the full
        reduction (sum, max, …) of every starting tile in the group.

        Algorithm
        ---------
        Cores in *core_group* are arranged into a logical ring in list
        order: each core sends to ``(my_idx + 1) % N`` and receives from
        ``(my_idx - 1) % N``. The local core runs ``N-1`` rounds and
        maintains two values:

        - ``result``     — the accumulator, folded in via ``reduce_fn``.
        - ``to_forward`` — the tile to send next round.

        In round 1, ``to_forward`` is the local starting tile. In every
        subsequent round, ``to_forward`` is the tile we received the
        previous round — we pass it onward unchanged. Each starting
        tile thus travels exactly ``N-1`` hops around the ring,
        visiting every other core once, and is folded into each
        visited core's accumulator. After ``N-1`` rounds, every core's
        accumulator has seen all ``N`` starting tiles, so every core
        holds the full reduction.

        This is the standard reduce-then-forward ring algorithm — not a
        scan, and not all-to-all. Sending the *received* tile (rather
        than the accumulator) is what keeps each value visiting every
        core exactly once; sending the accumulator instead causes
        double-counting.

        Example: 4 cores, sum, starting values [1, 2, 3, 4]::

            round 1:
              core 0:  recv 4 (from 3); send 1 to 1; acc = 1+4 = 5
              core 1:  recv 1 (from 0); send 2 to 2; acc = 2+1 = 3
              core 2:  recv 2 (from 1); send 3 to 3; acc = 3+2 = 5
              core 3:  recv 3 (from 2); send 4 to 0; acc = 4+3 = 7
            round 2:
              core 0:  recv 3 (the tile that was at 3 last round);
                       forward 4 to 1; acc = 5+3 = 8
              core 1:  recv 4; forward 1 to 2; acc = 3+4 = 7
              core 2:  recv 1; forward 2 to 3; acc = 5+1 = 6
              core 3:  recv 2; forward 3 to 0; acc = 7+2 = 9
            round 3:
              core 0:  recv 2; acc = 8+2 = 10
              core 1:  recv 3; acc = 7+3 = 10
              core 2:  recv 4; acc = 6+4 = 10
              core 3:  recv 1; acc = 9+1 = 10

        Cores not in *core_group* return their input tile unchanged
        without communicating.
        """
        if context.core_id not in core_group:
            return tile

        n_cores = len(core_group)
        my_idx = core_group.index(context.core_id)
        next_core = core_group[(my_idx + 1) % n_cores]
        prev_core = core_group[(my_idx - 1) % n_cores]

        result = tile.copy()       # accumulator — folded via reduce_fn each round
        to_forward = tile.copy()   # tile to send; round 1 = local starting value

        for _ in range(n_cores - 1):
            context.send_to(next_core, to_forward)
            received = yield RecvRequest(src=prev_core)
            result = reduce_fn(result, received)
            to_forward = received  # pass the received tile onward unchanged

        return result

    @staticmethod
    def reduce_return(value: Tile) -> Tile:
        """Return a value from a reduction block (identity passthrough)."""
        return value
