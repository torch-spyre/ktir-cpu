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

"""Scheduler correctness tests for GridExecutor._run_scheduler / execute_with_communication.

Test harness
------------
Scripts describe per-core execution as a list of steps:

    ("sync",  value)          -- return *value* immediately (no blocking)
    ("send",  dst, tile)      -- enqueue *tile* to *dst*, continue
    ("recv",  src)            -- block until a tile arrives from *src*
    ("loop",  n, body_steps)  -- repeat body_steps n times (simulates scf.for)

``build_ops(scripts)`` translates a dict[core_id -> steps] into
(operations, execute_op) ready for GridExecutor.execute_with_communication.
The returned *operations* is a list of one sentinel FakeOp per step;
*execute_op* dispatches on op index and core_id.

This lets tests express complex multi-core, multi-round, nested-loop
communication patterns without touching the parser or interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pytest

from ktir_cpu.grid import GridExecutor, RecvRequest
from ktir_cpu.ir_types import Tile
from ktir_cpu.memory import SpyreMemoryHierarchy


# ---------------------------------------------------------------------------
# Harness types
# ---------------------------------------------------------------------------

Step = Tuple  # one of the tuples described above

def _tile(val: float) -> Tile:
    return Tile(np.array([val], dtype=np.float16), "f16", (1,))


def _run_script(core, steps: List[Step], log: List):
    """Generator that executes a core's step list, yielding RecvRequests."""
    for step in steps:
        kind = step[0]
        if kind == "sync":
            log.append(("sync", core.core_id, step[1]))
        elif kind == "send":
            _, dst, tile = step
            core.send_to(dst, tile)
            log.append(("send", core.core_id, dst, tile.data[0]))
        elif kind == "recv":
            _, src = step
            tile = yield RecvRequest(src=src)
            log.append(("recv", core.core_id, src, tile.data[0]))
        elif kind == "loop":
            _, n, body = step
            for _ in range(n):
                yield from _run_script(core, body, log)
        else:
            raise ValueError(f"Unknown step kind: {kind!r}")


class _FakeOp:
    """Minimal op-like object for the scheduler harness."""
    op_type = "test.script"
    result = None
    operands: List[str] = field(default_factory=list)

    def __init__(self):
        self.operands = []


def build_grid(num_cores: int) -> GridExecutor:
    mem = SpyreMemoryHierarchy(num_cores=num_cores)
    return GridExecutor(grid_shape=(num_cores, 1, 1), memory=mem)


def run_scripts(
    grid: GridExecutor,
    scripts: Dict[int, List[Step]],
) -> List:
    """Run *scripts* on *grid*, return the event log in arrival order."""
    log: List = []
    sentinel = _FakeOp()

    def execute_op(op, context):
        steps = scripts.get(context.core_id, [])
        gen = _run_script(context, steps, log)
        # Peek: if the generator has nothing to yield, exhaust it and return.
        try:
            first = next(gen)
            # It yielded — return a generator that re-yields first then continues.
            def _replay():
                tile = yield first
                try:
                    while True:
                        tile = yield gen.send(tile)
                except StopIteration:
                    pass
            return _replay()
        except StopIteration:
            return None

    grid.execute_with_communication([sentinel], {}, execute_op)
    return log


# ---------------------------------------------------------------------------
# 1. Single cross-core send/recv (no loops)
# ---------------------------------------------------------------------------

def test_single_region_send_recv():
    """Core 0 sends a tile to core 1; core 1 receives it."""
    grid = build_grid(2)
    t = _tile(42.0)
    log = run_scripts(grid, {
        0: [("send", 1, t)],
        1: [("recv", 0)],
    })
    sends  = [e for e in log if e[0] == "send"]
    recvs  = [e for e in log if e[0] == "recv"]
    assert len(sends) == 1 and sends[0][3] == 42.0
    assert len(recvs) == 1 and recvs[0][3] == 42.0


def test_bidirectional_exchange():
    """Both cores send to each other simultaneously — no deadlock."""
    grid = build_grid(2)
    log = run_scripts(grid, {
        0: [("send", 1, _tile(1.0)), ("recv", 1)],
        1: [("send", 0, _tile(2.0)), ("recv", 0)],
    })
    recvs = {e[1]: e[3] for e in log if e[0] == "recv"}
    assert recvs[0] == 2.0  # core 0 received from core 1
    assert recvs[1] == 1.0  # core 1 received from core 0


# ---------------------------------------------------------------------------
# 2. Single scf.for across cores — recv inside the loop body
# ---------------------------------------------------------------------------

def test_single_scf_for_recv_in_loop():
    """Core 0 sends 3 tiles one per iteration; core 1 receives each inside the loop."""
    N = 3
    grid = build_grid(2)
    log = run_scripts(grid, {
        0: [("loop", N, [("send", 1, _tile(float(i))) for i in range(N)])],
        1: [("loop", N, [("recv", 0)])],
    })
    recvs = [e[3] for e in log if e[0] == "recv"]
    assert recvs == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# 3. Double scf.for — recv inside the inner loop
# ---------------------------------------------------------------------------

def test_double_scf_for_recv_in_inner_loop():
    """Nested loops: core 0 sends outer*inner tiles; core 1 receives in the inner loop."""
    OUTER, INNER = 2, 3
    grid = build_grid(2)
    log = run_scripts(grid, {
        0: [("loop", OUTER, [
                ("loop", INNER, [("send", 1, _tile(1.0))])
            ])],
        1: [("loop", OUTER, [
                ("loop", INNER, [("recv", 0)])
            ])],
    })
    recvs = [e for e in log if e[0] == "recv"]
    assert len(recvs) == OUTER * INNER


# ---------------------------------------------------------------------------
# 4. Double scf.for — recv inside the outer loop only (inner is sync)
# ---------------------------------------------------------------------------

def test_double_scf_for_recv_in_outer_loop():
    """Blocking recv in outer loop; inner loop is all sync work."""
    OUTER, INNER = 3, 4
    grid = build_grid(2)
    log = run_scripts(grid, {
        0: [("loop", OUTER, [
                ("send", 1, _tile(1.0)),
                ("loop", INNER, [("sync", None)]),
            ])],
        1: [("loop", OUTER, [
                ("recv", 0),
                ("loop", INNER, [("sync", None)]),
            ])],
    })
    recvs = [e for e in log if e[0] == "recv"]
    assert len(recvs) == OUTER


# ---------------------------------------------------------------------------
# 5. Deadlock detection — flat (no loops)
# ---------------------------------------------------------------------------

def test_deadlock_flat():
    """Both cores wait on each other with no sends — deadlock detected."""
    grid = build_grid(2)
    with pytest.raises(RuntimeError, match="Deadlock detected"):
        run_scripts(grid, {
            0: [("recv", 1)],
            1: [("recv", 0)],
        })


# ---------------------------------------------------------------------------
# 6. Deadlock detection — inside a loop
# ---------------------------------------------------------------------------

def test_deadlock_inside_loop():
    """Core 0 sends once then waits; core 1 waits twice — deadlock on second recv."""
    grid = build_grid(2)
    with pytest.raises(RuntimeError, match="Deadlock detected"):
        run_scripts(grid, {
            0: [("send", 1, _tile(1.0)), ("recv", 1)],  # sends once, then waits — never gets reply
            1: [("recv", 0), ("recv", 0)],              # receives once, then waits for a second send that never comes
        })


# ---------------------------------------------------------------------------
# 7. 4-core ring reduce (N-1 rounds)
# ---------------------------------------------------------------------------

def test_ring_reduce_4cores():
    """4-core ring: each core holds a value, N-1 rounds of send+recv accumulate the sum."""
    N = 4
    grid = build_grid(N)
    initial = {i: float(i + 1) for i in range(N)}  # [1, 2, 3, 4]
    accumulated = {i: initial[i] for i in range(N)}
    recv_log: Dict[int, List[float]] = {i: [] for i in range(N)}

    def scripts():
        result = {}
        for core_id in range(N):
            steps = []
            for _ in range(N - 1):
                next_core = (core_id + 1) % N
                prev_core = (core_id - 1) % N
                steps.append(("send", next_core, _tile(initial[core_id])))
                steps.append(("recv", prev_core))
            result[core_id] = steps
        return result

    log = run_scripts(grid, scripts())
    recvs_per_core = {}
    for e in log:
        if e[0] == "recv":
            recvs_per_core.setdefault(e[1], []).append(e[3])

    for core_id in range(N):
        assert len(recvs_per_core[core_id]) == N - 1
