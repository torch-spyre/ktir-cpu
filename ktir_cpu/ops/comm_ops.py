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
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Generator, List, Optional, Tuple, Type, Union
from ..affine import AffineSet
from ..ir_types import Tile
from ..grid import CoreContext, RecvRequest
from ..memory import LXScratchpad, SpyreMemoryHierarchy


class TransferBackend:
    """Abstract backend for remote LX scratchpad access.

    The single seam between memory ops that need a remote core's LX
    data and the model used to obtain it. ``run`` returns the
    LXScratchpad for the requested core; future variants (e.g. a
    ring-hop transport) may be generator-shaped and yield
    ``RecvRequest`` while data crosses the ring.

    Synchronous today (see :class:`InstantTransferBackend`). When a
    yielding variant lands, callers of ``CoreContext.get_lx`` will need
    to drive the resulting generator through the scheduler protocol —
    same machinery as :class:`ReduceBackend` already uses.
    """

    def run(self, ctx: "CoreContext", core_id: int) -> LXScratchpad:
        """Return the LXScratchpad for *core_id* (remote case only)."""
        raise NotImplementedError


class InstantTransferBackend(TransferBackend):
    """Synchronous lookup — no ring messages, no latency model.

    Returns ``memory.lx_scratchpads[N]`` by index. Valid for the
    distributed-view use case where LX partitions are pre-seeded by
    the host and no cross-core data movement occurs at runtime.
    """

    def __init__(self, memory: SpyreMemoryHierarchy):
        self._memory = memory

    def run(self, ctx: "CoreContext", core_id: int) -> LXScratchpad:
        """Return the LXScratchpad for *core_id* via direct lookup."""
        num = self._memory.num_cores
        if core_id < 0 or core_id >= num:
            raise ValueError(
                f"InstantTransferBackend.run: core_id={core_id} is out of "
                f"range [0, {num}) for this grid"
            )
        return self._memory.get_lx(core_id)


# ---------------------------------------------------------------------------
# CommPlan — logical input to a ReduceBackend
# ---------------------------------------------------------------------------
# The dialect handler builds a CommPlan from the IR's affine sets at
# consume-op entry; the backend reads it to know who participates and
# (when set) which producers each consumer depends on.  Pure logical:
# no ring order, no neighbour relation, no schedule, no topology —
# those are the backend's concern.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommPlan:
    """Logical structure of one inter-tile op: who produces, who
    consumes, who depends on whom.

    Built fresh by the consume handler at op entry from the future's
    parsed sets and the consume op's own attributes; passed to
    ``ReduceBackend.run``; discarded when ``run`` returns.

    Fields:
      ``producers``: tile ids producing partials in this group.
      ``consumers``: tile ids that should hold the result.
      ``deps``: optional ``producer_dependency_per_consumer`` resolved
          to ``{consumer_id: frozenset[producer_id]}``.  ``None`` =
          full-barrier mode (every consumer depends on every producer).
    """
    producers: Tuple[int, ...]
    consumers: Tuple[int, ...]
    deps: Optional[Dict[int, FrozenSet[int]]] = None

    def is_producer(self, core_id: int) -> bool:
        return core_id in self.producers

    def is_consumer(self, core_id: int) -> bool:
        return core_id in self.consumers

    def producers_for(self, consumer_id: int) -> FrozenSet[int]:
        """Producers this consumer depends on.

        Full-barrier (``deps`` is ``None``) → every producer.
        Per-tile sync (``deps`` set) → the declared subset for this
        consumer; raises ``KeyError`` if the consumer is missing
        from the deps map (caller bug — every consumer must appear).
        """
        if self.deps is None:
            return frozenset(self.producers)
        return self.deps[consumer_id]

    @classmethod
    def for_reduce(
        cls,
        *,
        producer_set: AffineSet,
        consumer_set: AffineSet,
        group_idx: int,
        num_cores: int,
        dep_set: Optional[AffineSet] = None,
    ) -> "CommPlan":
        """Build a CommPlan for an ``inter_tile_reduce`` op.

        Enumerates ``producer_set`` and ``consumer_set`` over the
        ``num_cores`` workgroup at the bound ``group_idx``.  When
        ``dep_set`` is given, evaluates it as ``(p)[c, g]`` (or
        ``(p)[c]`` when group-independent) for each consumer to
        populate ``deps``.
        """
        producers = tuple(
            i for i in range(num_cores)
            if producer_set.contains([i], [group_idx])
        )
        consumers = tuple(
            i for i in range(num_cores)
            if consumer_set.contains([i], [group_idx])
        )
        deps: Optional[Dict[int, FrozenSet[int]]] = None
        if dep_set is not None:
            # producer_dependency_per_consumer is parameterised either
            # ``(p)[c]`` (group-independent: one symbol) or ``(p)[c, g]``
            # (group-aware: two symbols).  Both shapes pass through
            # ``contains`` cleanly with the right symbol vector.
            deps = {}
            for c in consumers:
                syms = [c, group_idx] if dep_set.n_syms >= 2 else [c]
                deps[c] = frozenset(
                    p for p in producers if dep_set.contains([p], syms)
                )
        return cls(producers=producers, consumers=consumers, deps=deps)


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

    ``run`` is called once per core in the workgroup.  It may be a
    generator (yields ``RecvRequest`` at each blocking point) or a
    plain function (synchronous algorithms, e.g. LX-scratchpad
    reduce). The scheduler treats both shapes uniformly via
    ``inspect.isgenerator``.

    Returns the reduced tile for *this* core if it is a consumer;
    ``None`` otherwise.  ``CommPlan`` is a *mask*: non-producers
    contribute ``identity`` so the fold stays well-defined;
    non-consumers run the protocol but discard the result.

    Bandwidth accounting — the ``comm_bytes`` contract
    ----------------------------------------------------
    Subclasses MUST populate ``self.bytes_moved`` during ``run`` with
    the total bytes this core sent over the transport, then stamp it
    onto ``result.comm_bytes`` before returning (when returning a
    Tile).  The latency tracker reads ``comm_bytes`` from the result
    in ``_comm_size`` to charge ring bandwidth.  End-to-end:

        backend.run            : self.bytes_moved += tile.size_bytes()
                                 ...
                                 result.comm_bytes = self.bytes_moved
                                 return result
        latency._comm_size     : return result.comm_bytes

    The dialect handler does not touch ``bytes_moved`` or
    ``comm_bytes`` — the field is published by the backend that
    filled it.  Backend instances are constructed fresh per call
    (one per core per consume op), so the counter starts at 0 and
    accumulates across the rounds of a single ``run``.
    """

    # Default bandwidth counter — subclasses overwrite during ``run``.
    # Declared at class level so the contract is visible on the base.
    bytes_moved: int = 0

    @abstractmethod
    def run(
        self,
        context: CoreContext,
        tile: Tile,
        plan: "CommPlan",
        reduce_fn: Callable[[Tile, Tile], Tile],
        identity: Tile,
    ) -> Union[Tile, None, Generator[RecvRequest, Tile, Union[Tile, None]]]:
        ...


class RingReduceBackend(ReduceBackend):
    """Ring reduction over the whole workgroup — one core's view.

    Generator. Every core in the workgroup runs the protocol; the
    ring spans all ``ctx.num_cores`` cores in id order.  Yields
    ``RecvRequest`` exactly ``num_cores - 1`` times.

    ``CommPlan`` masks contributions and outputs:

    - **Non-producers** inject ``identity`` so the fold remains
      well-defined.  They still execute the protocol — they have to,
      because they're on the wire — but their seed is the identity
      tensor instead of a real partial.
    - **Non-consumers** run the protocol but ``return None``; the
      interpreter's SSA-bind logic skips storing ``None`` results,
      matching the spec's "results are undefined for non-participants"
      rule.

    Layout: naive ``range(n)`` ring in core-id order.  Each core's
    neighbour is ``(my_idx ± 1) % n``.  A different layout
    (e.g. coordinated with sibling groups, or a snake through a 2-D
    mesh) would be a different ``ReduceBackend`` subclass.

    Algorithm — reduce-then-forward
    -------------------------------
    Each core maintains two values:

    - ``result``     — the accumulator, folded via ``reduce_fn``.
    - ``to_forward`` — the tile to send next round.

    Round 1's ``to_forward`` is the local seed (real partial for
    producers, ``identity`` for non-producers).  Every subsequent
    round forwards the tile *received* the previous round, unchanged.
    Each starting tile thus travels exactly ``N - 1`` hops, visiting
    every other core once, folded into each visited core's
    accumulator.  Sending the accumulator instead would
    double-count.

    Example: 4 cores, sum, starting values [1, 2, 3, 4]::

        round 1:
          core 0:  recv 4 (from 3); send 1 to 1; acc = 1+4 = 5
          core 1:  recv 1 (from 0); send 2 to 2; acc = 2+1 = 3
          core 2:  recv 2 (from 1); send 3 to 3; acc = 3+2 = 5
          core 3:  recv 3 (from 2); send 4 to 0; acc = 4+3 = 7
        round 2:
          core 0:  recv 3; forward 4 to 1; acc = 5+3 = 8
          core 1:  recv 4; forward 1 to 2; acc = 3+4 = 7
          core 2:  recv 1; forward 2 to 3; acc = 5+1 = 6
          core 3:  recv 2; forward 3 to 0; acc = 7+2 = 9
        round 3:
          core 0:  recv 2; acc = 8+2 = 10
          core 1:  recv 3; acc = 7+3 = 10
          core 2:  recv 4; acc = 6+4 = 10
          core 3:  recv 1; acc = 9+1 = 10
    """

    def __init__(self):
        # Per-instance bandwidth counter.  The base class declares
        # ``bytes_moved: int = 0`` at class level, but we re-init on
        # the instance to avoid the class-level slot being shared
        # across instances if a subclass ever mutates without
        # re-assigning to self.
        self.bytes_moved = 0

    def run(self, context, tile, plan, reduce_fn, identity):
        n = context.num_cores
        next_core = (context.core_id + 1) % n
        prev_core = (context.core_id - 1) % n

        # Seed.
        # Producers start with their partial; non-producer consumers
        # start with identity so the first fold gives the right
        # answer.  ``to_forward`` is what this core injects onto the
        # wire — its content matters only when the *receiving* core
        # treats this core as an in-plan producer; otherwise the
        # receiver discards it.  We default to ``tile`` for
        # producers and ``identity`` for non-producers; the latter is
        # arbitrary (any tile of the right shape works) but keeps the
        # wire payload predictable.
        is_prod = plan.is_producer(context.core_id)
        seed = tile if is_prod else identity
        result = seed.copy()       # accumulator
        to_forward = seed.copy()   # tile to send next round (round 1: local)

        # Ring runs over the whole workgroup in lock-step: every core
        # sends and receives every round, regardless of whether it's
        # in ``plan.producers``.  The fold is plan-aware — only tiles
        # originating from in-plan producers are folded into the
        # accumulator.  Out-of-plan tiles still flow through
        # (preserving lock-step) but are discarded at fold time.
        #
        # Origin tracking: the tile this core receives in round k from
        # ``prev_core`` was originally produced by core
        # ``(my_id - k) mod n`` in the ring's id-order layout.  Each
        # subsequent round, the same tile is forwarded one hop further,
        # so we just step the origin index.
        my_id = context.core_id
        for k in range(1, n):
            origin = (my_id - k) % n
            self.bytes_moved += to_forward.size_bytes()
            context.send_to(next_core, to_forward)
            received = yield RecvRequest(src=prev_core)
            if plan.is_producer(origin):
                result = reduce_fn(result, received)
            to_forward = received  # pass received tile onward unchanged

        # Mask: non-consumers run the ring but discard the result.
        # The interpreter's SSA-bind ``if result is not None`` guard
        # then skips storing anything for this core.
        if not plan.is_consumer(context.core_id):
            return None

        # Stamp the per-core wire total onto the result Tile so the
        # latency tracker can read it via ``Tile.comm_bytes``.
        result.comm_bytes = self.bytes_moved
        return result


# ---------------------------------------------------------------------------
# Backend registry — explicit lookup keyed by op_name
# ---------------------------------------------------------------------------
# Dialect handlers declare which ReduceBackend to use via
# @register_reduce_backend(op_name, backend_cls). Lookup is explicit:
# the handler reads ``op.op_type`` and calls ``get_reduce_backend(...)``.
# Same pattern as the parser/handler registries in
# ``ktir_cpu.dialects.registry`` — explicit keys, no introspection.
# ---------------------------------------------------------------------------

_REDUCE_BACKENDS: Dict[str, Type[ReduceBackend]] = {}


def register_reduce_backend(op_name: str, backend_cls: Type[ReduceBackend]):
    """Decorator: declare *backend_cls* as the ReduceBackend for *op_name*.

    Used alongside ``@register(...)`` on a dialect handler::

        @register("ktdp.reduce", latency_category=LC.COMM)
        @register_reduce_backend("ktdp.reduce", RingReduceBackend)
        def ktdp__reduce(op, ctx, env):
            backend_cls = get_reduce_backend(op.op_type)
            return CommOps.reduce(ctx, tile, group, backend_cls(reduce_fn))

    Single op_name per call — keep registrations explicit. Re-registration
    silently overwrites (matches the parser/handler registries).
    """
    def deco(fn):
        _REDUCE_BACKENDS[op_name] = backend_cls
        return fn
    return deco


def get_reduce_backend(op_name: str) -> Type[ReduceBackend]:
    """Look up the ReduceBackend class registered for *op_name*.

    Raises ``RuntimeError`` if no backend is registered — the message
    points at the missing decorator.
    """
    backend_cls = _REDUCE_BACKENDS.get(op_name)
    if backend_cls is None:
        raise RuntimeError(
            f"No reduce backend registered for op_name {op_name!r}. "
            f"Add @register_reduce_backend({op_name!r}, <BackendCls>) "
            f"above the dialect handler."
        )
    return backend_cls


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
