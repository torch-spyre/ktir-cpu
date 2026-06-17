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

"""Scheduler tests for ``GridExecutor.execute_with_communication``.

What is under test
------------------
The scheduler primitives (message queue, generator driving, deadlock
detection) and ``CommOps`` algorithms together. ``CommOps.reduce`` is
the multi-yield generator path — it implements ring reduction and
yields ``N-1`` ``RecvRequest``s per core. Tests exercise the round
trip::

    test.reduce op
      → stub handler  (this file)
      → CommOps.reduce (yields RecvRequest(prev) N-1 times)
      → CoreExecutionStack._execute_until_block (yield from)
      → GridExecutor scheduler loop
      → message delivery
      → resume with received tile

What is *not* under test
------------------------
The dialect / interpreter dispatch path. Tests do not go through
``KTIRInterpreter`` or the global op registry. The experimental
``ktdp.transfer`` / ``ktdp.reduce`` ops are out of scope; tests speak
directly to ``CommOps``, which is the stable per-core comm surface.

Scaffold design — three layers
------------------------------
A test is a **declarative spec**: grid shape, per-core seed tiles, a
shared op list (broadcast to all cores), and per-core expected results.
The scaffold has three layers; each layer has one job and is replaceable
in isolation.

1. **Spec** (module-level dicts like ``SPEC_RING_REDUCE_FOUR``).
   Pure data. Describes *what* the test wants without saying *how* to
   run it. Adding a test = writing a spec dict and a one-line test
   function calling ``run_spec``.

2. **Harness** (``initialize_ops``, ``seed_tiles``, ``check_expectations``,
   ``build_grid``, ``run_spec``).
   Translates a spec into runtime actions: builds ``Operation`` objects
   from op-descriptor dicts, seeds tiles into core scopes, drives the
   scheduler, asserts post-conditions. Knows nothing about which ops
   exist — handlers are looked up at runtime by ``op_type``.

3. **Stub handlers** (``_h_reduce``, ``_STUB_HANDLERS``, ``execute_fn``).
   The minimal ``execute_op`` the scheduler needs. Looks up an op by
   ``op_type``, calls a handler that returns either a value (plain op)
   or a generator (blocking op). Mirrors the bind-result / track-LX
   logic of ``KTIRInterpreter.execute_op`` for plain returns; for
   generator returns hands off to the scheduler unchanged.

   Adding a new comm primitive to test = add one handler and one entry
   to ``_STUB_HANDLERS``. No registry, no fixture patching.

Why this shape
~~~~~~~~~~~~~~
- **Spec is data, not code.** Lets tests communicate intent at a glance
  and makes the failure mode (which value disagrees) localized.
- **Harness is generic.** One implementation; every test calls
  ``run_spec(SPEC, execute_fn)``.
- **Stub handlers are the smallest seam to ``CommOps``.** They route an
  ``Operation`` to a ``CommOps`` call and propagate the result. They
  are not ops themselves and have no semantics beyond "call this
  ``CommOps`` function with these arguments."

Spec format
-----------
::

    {
        "grid":       (nx, ny, nz),
        "seed":       {core_id: {ssa_name: Tile}},
        "operations": [{"op": str, "args": [str], "attrs": {...},
                        "result": str}],
        "expect":     {core_id: {ssa_name: scalar}},
    }

- ``grid``: passed to ``GridExecutor`` (number of cores = product).
- ``seed``: tiles bound into each core's scope before the scheduler runs.
- ``operations``: same list runs on every core (broadcast). Per-core
  variation comes from ``CommOps`` primitives that self-select via
  attributes (e.g. ``group``).
- ``expect``: each named tile's ``data[0]`` must match the scalar
  (within ``pytest.approx`` tolerance).

Ring reduction (recap)
----------------------
Cores in ``core_group`` form a logical ring in list order. Each core
runs ``N-1`` rounds. Per round it forwards the tile it just received
(round 1: its own starting tile) to the next neighbor and folds the
incoming tile into its local accumulator. After ``N-1`` rounds every
participating core holds the full reduction. See ``CommOps.reduce`` for
a worked example.

For a 4-core sum of ``[1, 2, 3, 4]``, every core ends with ``10``. For
``N=2``, exactly one round runs and both cores end with ``a + b``. The
two tests below cover these cases.

See also
--------
- ``docs/cross_core_scheduling.md`` — scheduler protocol, layer roles.
- ``PLAN_test_grid.md`` — design rationale for this harness.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

from ktir_cpu.grid import GridExecutor, RecvRequest
from ktir_cpu.ir_types import Operation, Tile
from ktir_cpu.memory import SpyreMemoryHierarchy
from ktir_cpu.ops.comm_ops import (
    CommOps,
    CommPlan,
    RingReduceBackend,
    get_reduce_backend,
    register_reduce_backend,
)


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------
# Tiny utilities for constructing the data values that flow through the
# scheduler. Tests only need scalar values to verify reduction
# correctness, so every tile is shape (1,) float16. Element-wise sum is
# the only reduction function exercised today; add more here when a new
# spec needs a different ``reduce_fn`` (max, mean, …).
# ---------------------------------------------------------------------------

def _tile(value: float) -> Tile:
    """Build a 1-element float16 tile holding ``value``."""
    return Tile(np.array([value], dtype=np.float16), "f16", (1,))


def _sum_tiles(a: Tile, b: Tile) -> Tile:
    """Element-wise float16 add — the reduction function used by tests."""
    return Tile(
        np.array(a.data + b.data, dtype=np.float16),
        "f16",
        a.shape,
    )


# ---------------------------------------------------------------------------
# Stub handlers — keyed by op_type
# ---------------------------------------------------------------------------
# Each handler is the minimal bridge between an ``Operation`` (data) and
# a ``CommOps`` call (behavior). It pulls operands and attributes off the
# op, calls a ``CommOps`` primitive, and returns whatever the primitive
# returned — a value (plain op) or a generator (blocking op). The
# scheduler does the rest:
#
#   - plain return → ``execute_fn`` binds it to ``op.result``.
#   - generator    → scheduler drives it via ``yield from``, parking the
#                    core on each ``RecvRequest`` until a tile arrives.
#
# Handlers do NOT register into ``ktir_cpu.dialects.registry``. The stub
# ``execute_fn`` looks them up directly in ``_STUB_HANDLERS``. This keeps
# tests isolated from the global registry and from any dialect-level
# changes.
#
# Adding a new comm primitive to test = (1) write a handler that calls
# the relevant ``CommOps`` function, (2) add an entry to
# ``_STUB_HANDLERS``. No registry, no fixture patching.
# ---------------------------------------------------------------------------

@register_reduce_backend("test.reduce", RingReduceBackend)
def _h_reduce(op: Operation, ctx) -> Any:
    """test.reduce — wraps RingReduceBackend.run with the registered backend.

    Builds a ``CommPlan`` from the ``group`` attribute (treated as both
    the producer set and the consumer set — synthetic all-reduce
    semantics) and an identity tile of zeros matching the input shape.
    Returns the generator produced by ``RingReduceBackend.run``; the
    scheduler drives the per-round ``RecvRequest`` yields and the
    final accumulator is bound to ``op.result``.
    """
    tile = ctx.get_value(op.operands[0])
    group = tuple(op.attributes["group"])
    plan = CommPlan(producers=group, consumers=group)
    identity = Tile(np.zeros_like(tile.data), tile.dtype, tile.shape)
    backend_cls = get_reduce_backend(op.op_type)
    return backend_cls().run(ctx, tile, plan, _sum_tiles, identity)


_STUB_HANDLERS: Dict[str, Callable[[Operation, Any], Any]] = {
    "test.reduce": _h_reduce,
}


# ---------------------------------------------------------------------------
# Stub execute_op — minimal replacement for KTIRInterpreter.execute_op
# ---------------------------------------------------------------------------
# ``execute_with_communication`` calls back into an ``execute_op`` for
# every op a core encounters. Production passes
# ``KTIRInterpreter.execute_op``; tests pass this stub.
#
# What the stub does
# ~~~~~~~~~~~~~~~~~~
# 1. Look up a handler in ``_STUB_HANDLERS`` keyed by ``op.op_type``.
# 2. Call the handler.
# 3. If the handler returned a generator, hand it back to the scheduler
#    unchanged — the scheduler's ``yield from`` will drive it through
#    every ``RecvRequest`` and collect the final return value.
# 4. Otherwise treat the return value as the op's result: bind it to
#    ``op.result`` in the core's scope and (for tiles) charge LX. This
#    mirrors the post-op handling in ``KTIRInterpreter.execute_op`` so a
#    handler returning a tile behaves identically in either harness.
#
# What the stub does NOT do
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Anything else ``KTIRInterpreter.execute_op`` does — region descent,
# latency tracking, dialect-specific pre-processing, error wrapping. The
# stub is deliberately the smallest piece of code that lets the
# scheduler call back into a comm-aware handler.
#
# Exposed as a fixture so individual tests can override (e.g. to inject
# tracing or fault injection) without touching the harness.
# ---------------------------------------------------------------------------

@pytest.fixture
def execute_fn() -> Callable[[Operation, Any], Any]:
    """Dispatch by op_type to a handler in ``_STUB_HANDLERS``.

    Mirrors the bind-result/track-LX logic of
    ``KTIRInterpreter.execute_op`` for plain returns; for generator
    returns, hands off to the scheduler (which drives via ``yield from``).
    """
    def _execute(op: Operation, ctx) -> Any:
        handler = _STUB_HANDLERS.get(op.op_type)
        if handler is None:
            raise KeyError(f"No stub handler for op_type {op.op_type!r}")
        result = handler(op, ctx)
        if inspect.isgenerator(result):
            return result
        if op.result and result is not None:
            ctx.set_value(op.result, result)
        return result
    return _execute


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
# Generic translation from a spec dict (data) to runtime actions
# (behavior). The harness knows nothing about which ops or handlers
# exist — handler lookup happens at runtime in ``execute_fn`` via
# ``op.op_type``. Tests interact with the harness through ``run_spec``;
# the smaller helpers are exposed for tests that want to drive parts
# of the flow individually (e.g. for custom assertions on intermediate
# state).
#
# Pipeline
# ~~~~~~~~
#   spec ──▶ build_grid       — fresh GridExecutor from spec["grid"]
#        ──▶ initialize_ops   — op-descriptor dicts → Operation objects
#        ──▶ seed_tiles       — bind tiles into each core's scope
#        ──▶ execute_with_communication(ops, {}, execute_fn)
#        ──▶ check_expectations — assert per-core post-conditions
# ---------------------------------------------------------------------------

def build_grid(shape: Tuple[int, int, int]) -> GridExecutor:
    n = shape[0] * shape[1] * shape[2]
    return GridExecutor(grid_shape=shape, memory=SpyreMemoryHierarchy(num_cores=n))


def initialize_ops(op_descs: List[dict]) -> List[Operation]:
    """Convert spec op-descriptor dicts to Operation objects."""
    return [
        Operation(
            result=desc.get("result"),
            op_type=desc["op"],
            operands=desc.get("args", []),
            attributes=desc.get("attrs", {}),
            result_type=desc.get("result_type", "tile"),
            regions=[],
        )
        for desc in op_descs
    ]


def seed_tiles(grid: GridExecutor, seed: Dict[int, Dict[str, Tile]]) -> None:
    for core_id, values in seed.items():
        for name, tile in values.items():
            grid.cores[core_id].set_value(name, tile)


def check_expectations(grid: GridExecutor,
                       expect: Dict[int, Dict[str, float]]) -> None:
    for core_id, values in expect.items():
        for name, expected in values.items():
            tile = grid.cores[core_id].get_value(name)
            actual = float(tile.data[0])
            assert actual == pytest.approx(expected), (
                f"core {core_id} {name}: expected {expected}, got {actual}"
            )


def run_spec(spec: dict, execute_fn) -> GridExecutor:
    grid = build_grid(spec["grid"])
    ops = initialize_ops(spec["operations"])
    seed_tiles(grid, spec["seed"])
    grid.execute_with_communication(ops, {}, execute_fn)
    check_expectations(grid, spec["expect"])
    return grid


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
# Module-level test specs — pure data describing what each test exercises.
# Each spec is a plain dict matching the format documented in the module
# docstring. A test reads as: pick a spec, run it, check expectations.
#
# Multi-group specs broadcast multiple ``test.reduce`` ops to every
# core. ``CommOps.reduce`` checks ``ctx.core_id in group`` and returns
# the input tile unchanged for non-participants, so a core that appears
# in only one group's reduction simply no-ops on the other ops. That is
# why every core must have ``%t`` bound — the broadcast op references it
# even when the core does not participate.
# ---------------------------------------------------------------------------

SPEC_RING_REDUCE_2X1X1 = {
    "grid": (2, 1, 1),
    "seed": {
        0: {"%t": _tile(5.0)},
        1: {"%t": _tile(7.0)},
    },
    "operations": [
        {"op": "test.reduce", "args": ["%t"],
         "attrs": {"group": [0, 1]}, "result": "%r"},
    ],
    "expect": {
        0: {"%r": 12.0},
        1: {"%r": 12.0},
    },
}


SPEC_RING_REDUCE_4X1X1 = {
    "grid": (4, 1, 1),
    "seed": {
        0: {"%t": _tile(1.0)},
        1: {"%t": _tile(2.0)},
        2: {"%t": _tile(3.0)},
        3: {"%t": _tile(4.0)},
    },
    "operations": [
        {"op": "test.reduce", "args": ["%t"],
         "attrs": {"group": [0, 1, 2, 3]}, "result": "%r"},
    ],
    "expect": {
        # After N-1=3 ring rounds, every participating core holds the full sum.
        0: {"%r": 10.0},
        1: {"%r": 10.0},
        2: {"%r": 10.0},
        3: {"%r": 10.0},
    },
}


# 4x4 grid, four concurrent ring reductions along rows.
# Linear id = y * 4 + x. Row y = [4y, 4y+1, 4y+2, 4y+3].
# Each row sums [1, 2, 3, 4] = 10.
#
# Each row gets its own result name (%r0..%r3) so that a core that
# only participates in row y has only %r{y} checked. Non-participants
# still return their input tile via ``CommOps.reduce``, but the spec
# never asserts on those values.
SPEC_RING_REDUCE_4X4X1_ROWS = {
    "grid": (4, 4, 1),
    "seed": {core_id: {"%t": _tile(float((core_id % 4) + 1))}
             for core_id in range(16)},
    "operations": [
        {"op": "test.reduce", "args": ["%t"],
         "attrs": {"group": [4 * y + x for x in range(4)]},
         "result": f"%r{y}"}
        for y in range(4)
    ],
    "expect": {core_id: {f"%r{core_id // 4}": 10.0} for core_id in range(16)},
}


# 4x4 grid, four concurrent ring reductions along columns.
# Column x = [x, x+4, x+8, x+12]. Each column sums [1, 2, 3, 4] = 10.
# Each column gets its own result name (%c0..%c3); see the row spec for
# the rationale on per-group result names.
SPEC_RING_REDUCE_4X4X1_COLS = {
    "grid": (4, 4, 1),
    "seed": {core_id: {"%t": _tile(float((core_id // 4) + 1))}
             for core_id in range(16)},
    "operations": [
        {"op": "test.reduce", "args": ["%t"],
         "attrs": {"group": [4 * y + x for y in range(4)]},
         "result": f"%c{x}"}
        for x in range(4)
    ],
    "expect": {core_id: {f"%c{core_id % 4}": 10.0} for core_id in range(16)},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# One parametrized test, one entry per spec. Adding coverage = define a
# spec above and add it to the parametrize list with a readable id.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(SPEC_RING_REDUCE_2X1X1, id="2x1x1"),
        pytest.param(SPEC_RING_REDUCE_4X1X1, id="4x1x1"),
        pytest.param(SPEC_RING_REDUCE_4X4X1_ROWS, id="4x4x1_rows"),
        pytest.param(SPEC_RING_REDUCE_4X4X1_COLS, id="4x4x1_cols"),
    ],
)
def test_ring_reduce(spec, execute_fn):
    """Ring reduction across one or more disjoint groups."""
    run_spec(spec, execute_fn)


# ---------------------------------------------------------------------------
# Deadlock detection
# ---------------------------------------------------------------------------
# These tests exist as both a check on the scheduler's deadlock detector
# *and* a demonstration that the framework cannot deadlock under normal
# usage — to get here we have to monkey-patch ``RingReduceBackend.run``
# with deliberately broken protocol.
#
# Three failure modes are covered; each is a class of bug a backend
# implementer might introduce, and each must be caught by the scheduler:
#
#   - mutual_recv  : recv-only, no sends                 (flat deadlock)
#   - wrong_dest   : send to wrong destination           (miscoordinated channels)
#   - extra_recv   : N recvs after N-1 sends (off-by-one) (deadlock after partial
#                                                          progress)
#
# All three exercise the same end-to-end harness (run_spec on the
# 4-core spec) so any change to the harness propagates to the
# deadlock cases automatically.
# ---------------------------------------------------------------------------

def _broken_recv_only(self, ctx, tile, plan, reduce_fn, identity):
    """Recv from the previous neighbor, never send. Every core in
    the workgroup blocks on round 1 → flat mutual-recv deadlock.

    Ring spans the whole workgroup; ``plan`` is informational only
    in the broken stubs (the bug is in send/recv pairing, not in
    the fold).
    """
    n = ctx.num_cores
    prev = (ctx.core_id - 1) % n
    yield RecvRequest(src=prev)
    return tile


def _broken_wrong_dest(self, ctx, tile, plan, reduce_fn, identity):
    """Send to ``(my_id + 2) % n`` while recv-ing from
    ``(my_id - 1) % n``. Every core's send lands in a queue no one
    reads from; every core's recv waits on a queue no one writes to.
    Deadlock at round 1.
    """
    n = ctx.num_cores
    wrong_dst = (ctx.core_id + 2) % n
    prev = (ctx.core_id - 1) % n
    ctx.send_to(wrong_dst, tile)
    yield RecvRequest(src=prev)
    return tile


def _broken_extra_recv(self, ctx, tile, plan, reduce_fn, identity):
    """Run N-1 ring rounds correctly, then dangle one extra recv. The
    final recv has no matching send → deadlock after the algorithm
    has otherwise made progress.
    """
    n = ctx.num_cores
    next_core = (ctx.core_id + 1) % n
    prev_core = (ctx.core_id - 1) % n
    result = tile.copy()
    to_forward = tile.copy()
    for _ in range(n - 1):
        ctx.send_to(next_core, to_forward)
        received = yield RecvRequest(src=prev_core)
        result = reduce_fn(result, received)
        to_forward = received
    # Dangling recv — no one sent for this round.
    yield RecvRequest(src=prev_core)
    return result


@pytest.mark.parametrize(
    "broken_run",
    [
        pytest.param(_broken_recv_only,  id="mutual_recv"),
        pytest.param(_broken_wrong_dest, id="wrong_dest"),
        pytest.param(_broken_extra_recv, id="extra_recv"),
    ],
)
def test_scheduler_detects_deadlock(broken_run, execute_fn, monkeypatch):
    """Scheduler raises RuntimeError('Deadlock detected: ...') when a
    backend breaks the send/recv protocol."""
    monkeypatch.setattr(RingReduceBackend, "run", broken_run)
    with pytest.raises(RuntimeError, match="Deadlock detected"):
        run_spec(SPEC_RING_REDUCE_4X1X1, execute_fn)
