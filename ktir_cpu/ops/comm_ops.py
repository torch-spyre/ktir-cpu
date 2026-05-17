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

from abc import ABC, abstractmethod
from typing import Callable, Generator, List
from ..ir_types import Tile
from ..grid import CoreContext, RecvRequest
from ..memory import LXScratchpad, SpyreMemoryHierarchy


class RingBackend:
    """Abstract backend for remote LX scratchpad access.

    The single seam between memory ops that need a remote core's LX
    data and the model used to obtain it. Cross-core comm is *not* on
    this path — it goes through the scheduler protocol
    (``CoreContext.send_to`` + ``RecvRequest``); see
    ``docs/cross_core_scheduling.md``.
    """

    def get_lx(self, core_id: int) -> LXScratchpad:
        """Return the LXScratchpad for *core_id*."""
        raise NotImplementedError


class DirectLXBackend(RingBackend):
    """Direct scratchpad access — no ring messages, no latency model.

    Returns ``memory.lx_scratchpads[N]`` by index. Valid for the
    distributed-view use case where LX partitions are pre-seeded by the
    host and no cross-core data movement occurs at runtime.
    """

    def __init__(self, memory: SpyreMemoryHierarchy):
        self._memory = memory

    def get_lx(self, core_id: int) -> LXScratchpad:
        """Return the LXScratchpad for *core_id* via direct lookup."""
        num = self._memory.num_cores
        if core_id < 0 or core_id >= num:
            raise ValueError(
                f"get_lx: core_id={core_id} is out of range [0, {num}) for this grid"
            )
        return self._memory.get_lx(core_id)


# ---------------------------------------------------------------------------
# Reduce backends
# ---------------------------------------------------------------------------
# A ReduceBackend owns the full reduce protocol — messaging, compute,
# completion. Backends are clients of the scheduler protocol: ``run``
# may yield ``RecvRequest`` and call ``ctx.send_to``; the scheduler
# drives any returned generator to completion. ``CommOps.reduce`` is
# now a passthrough — it picks no algorithm of its own.
#
# See docs/cross_core_scheduling.md (Future direction) for the design
# rationale and the open question of attribute- vs. env-based backend
# selection at the dialect-handler boundary.
# ---------------------------------------------------------------------------


class ReduceBackend(ABC):
    """Abstract reduce algorithm — owns messaging, compute, completion.

    ``run`` is called once per participating core. It may be a
    generator (yields ``RecvRequest`` at each blocking point) or a
    plain function (synchronous algorithms, e.g. LX-scratchpad
    reduce). The scheduler treats both shapes uniformly via
    ``inspect.isgenerator``.

    Returns the reduced tile for *this* core. Cores not in
    ``core_group`` should return *tile* unchanged without
    communicating.
    """

    @abstractmethod
    def run(
        self,
        context: CoreContext,
        tile: Tile,
        core_group: List[int],
    ) -> Generator[RecvRequest, Tile, Tile]:
        ...


class RingReduceBackend(ReduceBackend):
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
    tile thus travels exactly ``N-1`` hops around the ring, visiting
    every other core once, and is folded into each visited core's
    accumulator. After ``N-1`` rounds, every core's accumulator has
    seen all ``N`` starting tiles, so every core holds the full
    reduction.

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

    Cores not in *core_group* return *tile* unchanged without
    communicating.
    """

    def __init__(self, reduce_fn: Callable[[Tile, Tile], Tile]):
        self.reduce_fn = reduce_fn

    def run(self, context, tile, core_group):
        if context.core_id not in core_group:
            return tile

        n_cores = len(core_group)
        my_idx = core_group.index(context.core_id)
        next_core = core_group[(my_idx + 1) % n_cores]
        prev_core = core_group[(my_idx - 1) % n_cores]

        result = tile.copy()       # accumulator
        to_forward = tile.copy()   # tile to send next round (round 1: local)

        for _ in range(n_cores - 1):
            context.send_to(next_core, to_forward)
            received = yield RecvRequest(src=prev_core)
            result = self.reduce_fn(result, received)
            to_forward = received  # pass received tile onward unchanged

        return result


# ---------------------------------------------------------------------------
# CommOps — stable per-core comm surface
# ---------------------------------------------------------------------------


class CommOps:
    """Inter-core comm primitives. Stable surface for dialect handlers
    and tests. Algorithm choices live in pluggable backends; CommOps
    methods are passthroughs that wire ``ctx`` into the chosen backend.
    """

    @staticmethod
    def reduce(
        context: CoreContext,
        tile: Tile,
        core_group: List[int],
        backend: ReduceBackend,
    ) -> Generator:
        """Reduce *tile* across *core_group* using *backend* — one
        core's view.

        Passthrough. The backend owns the algorithm (ring rounds,
        LX-scratchpad accumulation, …); ``CommOps.reduce`` exists so
        dialect handlers and tests have a single stable entry point
        regardless of which algorithm is in play.

        See ``RingReduceBackend`` for the canonical ring algorithm and
        ``docs/cross_core_scheduling.md`` for the design rationale.
        """
        return backend.run(context, tile, core_group)

    @staticmethod
    def reduce_return(value: Tile) -> Tile:
        """Return a value from a reduction block (identity passthrough)."""
        return value
