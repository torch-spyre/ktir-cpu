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
Execution handler tests for the KTIR dialect layer.

Each test calls dispatch(op_type)(op, context, env) directly — the same
path the interpreter uses — with minimal hand-built Operation objects.

Covers: arith, math, linalg, tensor, scf/func, ktdp.
ktdp.transfer / ktdp.reduce skipped — replay bug (see docs/gap_analysis.md section K).
"""

import numpy as np
import pytest

from ktir_cpu.ir_types import Operation, Tile
from ktir_cpu.grid import CoreContext, GridExecutor
from ktir_cpu.memory import HBMSimulator, LXScratchpad, SpyreMemoryHierarchy
from ktir_cpu.dialects.registry import dispatch, ExecutionEnv
from ktir_cpu.ops.comm_ops import RingNetwork

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _op(op_type, operands=None, attributes=None, result=None, regions=None):
    return Operation(
        op_type=op_type,
        operands=operands or [],
        attributes=attributes or {},
        result=result,
        result_type=None,
        regions=regions or [],
    )


def _make_ctx(grid_pos=(0, 0, 0), core_id=0):
    return CoreContext(
        core_id=core_id,
        grid_pos=grid_pos,
        lx=LXScratchpad(size_mb=2, core_id=core_id),
        hbm=HBMSimulator(),
    )


def _make_env(grid_shape=(1, 1, 1)):
    memory = SpyreMemoryHierarchy(num_cores=1)
    grid = GridExecutor(grid_shape=grid_shape, memory=memory)
    ring = RingNetwork(num_cores=1)

    def execute_region(context, ops):
        result = None
        for op in ops:
            result = op(context)
        return result

    return ExecutionEnv(grid_executor=grid, ring=ring, execute_region=execute_region)


def _call(op_type, context, env, **op_kwargs):
    handler = dispatch(op_type)
    assert handler is not None, f"No handler for {op_type!r}"
    return handler(_op(op_type, **op_kwargs), context, env)


def _tile(values, dtype="f16"):
    data = np.array(values, dtype=np.float16)
    return Tile(data, dtype, data.shape)


def _ctx_with(**bindings):
    ctx = _make_ctx()
    for k, v in bindings.items():
        ctx.set_value(k, v)
    return ctx

# ---------------------------------------------------------------------------
# arith dialect exec
# ---------------------------------------------------------------------------

class TestArithFloat:
    def test_addf_tiles(self):
        # tile + tile
        ctx = _ctx_with(**{"%a": _tile([1, 2]), "%b": _tile([3, 4])})
        result = _call("arith.addf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([4, 6], dtype=np.float16))

    def test_addf_scalars(self):
        # scalar + scalar
        ctx = _ctx_with(**{"%a": np.float16(2.0), "%b": np.float16(3.0)})
        result = _call("arith.addf", ctx, _make_env(), operands=["%a", "%b"])
        assert float(result) == pytest.approx(5.0, rel=1e-2)

    def test_addf_scalar_tile(self):
        # scalar broadcast into tile (scalar on left)
        ctx = _ctx_with(**{"%a": np.float16(1.0), "%b": _tile([1, 2, 3])})
        result = _call("arith.addf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([2, 3, 4], dtype=np.float16))

    def test_addf_tile_scalar(self):
        # scalar broadcast into tile (scalar on right)
        ctx = _ctx_with(**{"%a": _tile([1, 2, 3]), "%b": np.float16(1.0)})
        result = _call("arith.addf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([2, 3, 4], dtype=np.float16))

    def test_subf_scalar_tile(self):
        # scalar minus tile
        ctx = _ctx_with(**{"%a": np.float16(10.0), "%b": _tile([1, 2, 3])})
        result = _call("arith.subf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([9, 8, 7], dtype=np.float16))

    def test_mulf_tile_scalar(self):
        # tile * scalar
        ctx = _ctx_with(**{"%a": _tile([1, 2, 3]), "%b": np.float16(2.0)})
        result = _call("arith.mulf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float16))

    def test_mulf_scalar_tile(self):
        # scalar * tile
        ctx = _ctx_with(**{"%a": np.float16(3.0), "%b": _tile([1, 2, 3])})
        result = _call("arith.mulf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([3, 6, 9], dtype=np.float16))

    def test_divf_tile_scalar(self):
        # tile / scalar
        ctx = _ctx_with(**{"%a": _tile([4, 6, 8]), "%b": np.float16(2.0)})
        result = _call("arith.divf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.allclose(result.data, np.array([2, 3, 4], dtype=np.float16), rtol=1e-2)

    def test_divf_scalar_tile(self):
        # scalar / tile
        ctx = _ctx_with(**{"%a": np.float16(12.0), "%b": _tile([2, 3, 4])})
        result = _call("arith.divf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.allclose(result.data, np.array([6, 4, 3], dtype=np.float16), rtol=1e-2)

    def test_maxf(self):
        # element-wise maximum
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([4, 2, 6])})
        result = _call("arith.maxf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([4, 5, 6], dtype=np.float16))

    def test_maxnumf(self):
        # NaN-aware max
        ctx = _ctx_with(**{"%a": _tile([1, 5]), "%b": _tile([4, 2])})
        result = _call("arith.maxnumf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([4, 5], dtype=np.float16))

    def test_maximumf_tiles(self):
        # arith.maximumf is the same dispatch as arith.maxf
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([4, 2, 6])})
        result = _call("arith.maximumf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([4, 5, 6], dtype=np.float16))

    def test_minimumf(self):
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([4, 2, 6])})
        result = _call("arith.minimumf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([1, 2, 3], dtype=np.float16))

    def test_minnumf(self):
        ctx = _ctx_with(**{"%a": _tile([1, 5]), "%b": _tile([4, 2])})
        result = _call("arith.minnumf", ctx, _make_env(), operands=["%a", "%b"])
        assert np.array_equal(result.data, np.array([1, 2], dtype=np.float16))

    def test_minnumf_nan(self):
        # NaN non-propagating: fmin(NaN, 2) → 2;  fmin(3, NaN) → 3
        a = Tile(np.array([float('nan'), 3], dtype=np.float16), "f16", (2,))
        b = Tile(np.array([2, float('nan')], dtype=np.float16), "f16", (2,))
        ctx = _ctx_with(**{"%a": a, "%b": b})
        result = _call("arith.minnumf", ctx, _make_env(), operands=["%a", "%b"])
        assert result.data[0] == 2.0
        assert result.data[1] == 3.0

    def test_extf_promotes_to_f32(self):
        # extf widens f16 → f32
        t = _tile([1, 2])  # f16 tile
        ctx = _ctx_with(**{"%a": t})
        result = _call("arith.extf", ctx, _make_env(), operands=["%a"])
        assert result.dtype == "f32"
        np.testing.assert_array_equal(result.data, np.array([1, 2], dtype=np.float32))

    def test_truncf_passthrough(self):
        # truncf is a no-op in simulation
        t = _tile([1, 2])
        ctx = _ctx_with(**{"%a": t})
        assert _call("arith.truncf", ctx, _make_env(), operands=["%a"]) is t

# ---------------------------------------------------------------------------
# arith (int) dialect exec
# ---------------------------------------------------------------------------

class TestArithInt:
    def test_addi_tile_broadcast(self):
        # tile + scalar broadcast produces a tile
        ctx = _ctx_with(**{"%a": _tile([1, 2, 3]), "%b": 5})
        result = _call("arith.addi", ctx, _make_env(), operands=["%a", "%b"])
        assert isinstance(result, Tile)
        assert np.array_equal(result.data, np.array([6, 7, 8], dtype=np.float16))

    def test_addi_broadcast_tile(self):
        # scalar + tile broadcast produces a tile
        ctx = _ctx_with(**{"%a": 10, "%b": _tile([1, 2, 3])})
        assert isinstance(_call("arith.addi", ctx, _make_env(), operands=["%a", "%b"]), Tile)

    def test_muli_tile_broadcast(self):
        # tile * scalar broadcast produces a tile
        ctx = _ctx_with(**{"%a": _tile([1, 2, 3]), "%b": 3})
        result = _call("arith.muli", ctx, _make_env(), operands=["%a", "%b"])
        assert isinstance(result, Tile)
        assert np.array_equal(result.data, np.array([3, 6, 9], dtype=np.float16))

    def test_muli_broadcast_tile(self):
        # scalar * tile broadcast produces a tile
        ctx = _ctx_with(**{"%a": 2, "%b": _tile([1, 2, 3])})
        assert isinstance(_call("arith.muli", ctx, _make_env(), operands=["%a", "%b"]), Tile)

    def test_subi(self):
        # scalar integer subtraction
        ctx = _ctx_with(**{"%a": 10, "%b": 3})
        assert _call("arith.subi", ctx, _make_env(), operands=["%a", "%b"]) == 7

    def test_remui(self):
        # unsigned integer remainder
        ctx = _ctx_with(**{"%a": 10, "%b": 3})
        assert _call("arith.remui", ctx, _make_env(), operands=["%a", "%b"]) == 1

# ---------------------------------------------------------------------------
# arith (casts) dialect exec
# ---------------------------------------------------------------------------

class TestArithCastsConstants:
    def test_constant_scalar(self):
        # scalar constant is returned as-is
        result = _call("arith.constant", _make_ctx(), _make_env(), attributes={"value": 42})
        assert result == 42

    def test_constant_tensor(self):
        # tensor constant produces a zero-filled tile of the given shape
        result = _call("arith.constant", _make_ctx(), _make_env(),
                       attributes={"value": 0.0, "is_tensor": True, "shape": (4,), "dtype": "f16"})
        assert isinstance(result, Tile)
        assert result.shape == (4,)
        assert np.all(result.data == 0)

    def test_extsi(self):
        # sign-extend integer — returns Python int
        ctx = _ctx_with(**{"%a": 5})
        assert _call("arith.extsi", ctx, _make_env(), operands=["%a"]) == 5

    def test_index_cast(self):
        # cast to index type — returns Python int
        ctx = _ctx_with(**{"%a": 7})
        assert _call("arith.index_cast", ctx, _make_env(), operands=["%a"]) == 7

    def test_sitofp(self):
        # signed int to float — returns np.float16
        ctx = _ctx_with(**{"%a": 3})
        result = _call("arith.sitofp", ctx, _make_env(), operands=["%a"])
        assert isinstance(result, np.float16)
        assert float(result) == pytest.approx(3.0, rel=1e-2)


class TestArithBitcast:
    def test_bitcast_i32_to_f32_scalar(self):
        # reinterpret int bits as float
        ctx = _ctx_with(**{"%a": 1065353216})  # 0x3F800000 = 1.0f
        result = _call("arith.bitcast", ctx, _make_env(),
                       operands=["%a"], attributes={"dst_type": "f32"})
        assert abs(float(result) - 1.0) < 1e-6

    def test_bitcast_f32_to_i32_scalar(self):
        # reinterpret float bits as int
        ctx = _ctx_with(**{"%a": np.float32(1.0)})
        result = _call("arith.bitcast", ctx, _make_env(),
                       operands=["%a"], attributes={"dst_type": "i32"})
        assert result == 0x3F800000

    def test_bitcast_i32_to_f32_tile(self):
        # reinterpret bits on a tile (view, no data change)
        data = np.array([0x3F800000, 0x40000000], dtype=np.int32)  # 1.0, 2.0
        t = Tile(data, "i32", data.shape)
        ctx = _ctx_with(**{"%a": t})
        result = _call("arith.bitcast", ctx, _make_env(),
                       operands=["%a"], attributes={"dst_type": "f32"})
        assert isinstance(result, Tile)
        assert result.dtype == "f32"
        assert np.allclose(result.data, [1.0, 2.0])

# ---------------------------------------------------------------------------
# arith (cmpi) dialect exec
# ---------------------------------------------------------------------------

class TestArithCmpiSelect:
    def test_cmpi_scalar(self):
        # scalar comparison returns Python bool
        ctx = _ctx_with(**{"%a": 1, "%b": 2})
        result = _call("arith.cmpi", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "slt"})
        assert result is True

    def test_cmpi_tile(self):
        # tile comparison returns boolean tile
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([2, 4, 3])})
        result = _call("arith.cmpi", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "slt"})
        assert isinstance(result, Tile)
        assert np.array_equal(result.data, np.array([True, False, False]))

    def test_cmpi_ult(self):
        # unsigned less-than on scalars
        ctx = _ctx_with(**{"%a": 1, "%b": 2})
        result = _call("arith.cmpi", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "ult"})
        assert result is True

    def test_cmpi_uge_tile(self):
        # unsigned greater-or-equal on tiles
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([2, 4, 3])})
        result = _call("arith.cmpi", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "uge"})
        assert isinstance(result, Tile)
        assert np.array_equal(result.data, np.array([False, True, True]))

    def test_select_scalar(self):
        # scalar select returns the chosen value
        ctx = _ctx_with(**{"%cond": True, "%t": 10, "%f": 20})
        assert _call("arith.select", ctx, _make_env(), operands=["%cond", "%t", "%f"]) == 10

    def test_select_tile(self):
        # element-wise select via boolean tile
        cond = Tile(np.array([True, False, True]), "i1", (3,))
        ctx = _ctx_with(**{"%cond": cond, "%t": _tile([1, 2, 3]), "%f": _tile([4, 5, 6])})
        result = _call("arith.select", ctx, _make_env(), operands=["%cond", "%t", "%f"])
        assert np.array_equal(result.data, np.array([1, 5, 3], dtype=np.float16))

# ---------------------------------------------------------------------------
# arith.cmpf dialect exec
# ---------------------------------------------------------------------------

class TestArithCmpf:
    def test_cmpf_olt(self):
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([2, 4, 3])})
        result = _call("arith.cmpf", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "olt"})
        assert isinstance(result, Tile)
        assert np.array_equal(result.data, np.array([True, False, False]))

    def test_cmpf_oge(self):
        ctx = _ctx_with(**{"%a": _tile([1, 5, 3]), "%b": _tile([2, 4, 3])})
        result = _call("arith.cmpf", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "oge"})
        assert np.array_equal(result.data, np.array([False, True, True]))

    def test_cmpf_olt_nan(self):
        # Ordered predicates always return False when either operand is NaN.
        # olt(NaN, 2) → False;  olt(1, NaN) → False
        a = Tile(np.array([float('nan'), 1], dtype=np.float16), "f16", (2,))
        b = Tile(np.array([2, float('nan')], dtype=np.float16), "f16", (2,))
        ctx = _ctx_with(**{"%a": a, "%b": b})
        result = _call("arith.cmpf", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "olt"})
        assert np.array_equal(result.data, np.array([False, False]))

    def test_cmpf_ueq_nan(self):
        # Unordered predicates return True when either operand is NaN.
        # ueq(NaN, 2) → True (NaN present);  ueq(3, 3) → True (equal)
        a = Tile(np.array([float('nan'), 3], dtype=np.float16), "f16", (2,))
        b = Tile(np.array([2, 3], dtype=np.float16), "f16", (2,))
        ctx = _ctx_with(**{"%a": a, "%b": b})
        result = _call("arith.cmpf", ctx, _make_env(),
                       operands=["%a", "%b"], attributes={"predicate": "ueq"})
        assert np.array_equal(result.data, np.array([True, True]))

    def test_cmpf_ord_uno(self):
        # ord: True iff neither operand is NaN.  uno: True iff either is NaN.
        # ord(NaN, 2) → False;  ord(3, 4) → True
        # uno(NaN, 2) → True;   uno(3, 4) → False
        a = Tile(np.array([float('nan'), 3], dtype=np.float16), "f16", (2,))
        b = Tile(np.array([2, 4], dtype=np.float16), "f16", (2,))
        ctx = _ctx_with(**{"%a": a, "%b": b})
        result_ord = _call("arith.cmpf", ctx, _make_env(),
                           operands=["%a", "%b"], attributes={"predicate": "ord"})
        assert np.array_equal(result_ord.data, np.array([False, True]))
        result_uno = _call("arith.cmpf", ctx, _make_env(),
                           operands=["%a", "%b"], attributes={"predicate": "uno"})
        assert np.array_equal(result_uno.data, np.array([True, False]))


# ---------------------------------------------------------------------------
# math dialect exec
# ---------------------------------------------------------------------------

class TestMath:
    def test_exp_tile(self):
        # element-wise exp on a tile
        ctx = _ctx_with(**{"%x": _tile([0, 1])})
        result = _call("math.exp", ctx, _make_env(), operands=["%x"])
        assert isinstance(result, Tile)
        assert np.allclose(result.data, np.exp(np.array([0, 1], dtype=np.float32)).astype(np.float16), rtol=1e-2)

    def test_exp_scalar(self):
        # scalar exp returns scalar
        ctx = _ctx_with(**{"%x": np.float16(0.0)})
        assert abs(float(_call("math.exp", ctx, _make_env(), operands=["%x"])) - 1.0) < 0.01

    def test_sqrt_tile(self):
        # element-wise sqrt on a tile
        ctx = _ctx_with(**{"%x": _tile([4, 9, 16])})
        result = _call("math.sqrt", ctx, _make_env(), operands=["%x"])
        assert isinstance(result, Tile)
        assert np.allclose(result.data, np.array([2, 3, 4], dtype=np.float16), rtol=1e-2)

    def test_sqrt_scalar(self):
        # scalar sqrt returns scalar
        ctx = _ctx_with(**{"%x": np.float16(4.0)})
        assert abs(float(_call("math.sqrt", ctx, _make_env(), operands=["%x"])) - 2.0) < 0.01

    def test_log_tile(self):
        # element-wise log on a tile
        ctx = _ctx_with(**{"%x": _tile([1, 2, 4])})
        result = _call("math.log", ctx, _make_env(), operands=["%x"])
        assert isinstance(result, Tile)
        assert np.allclose(result.data, np.log(np.array([1, 2, 4], dtype=np.float32)).astype(np.float16), rtol=1e-2)

    def test_log_scalar(self):
        # scalar log returns scalar
        ctx = _ctx_with(**{"%x": np.float16(1.0)})
        assert abs(float(_call("math.log", ctx, _make_env(), operands=["%x"])) - 0.0) < 0.01

# ---------------------------------------------------------------------------
# linalg dialect exec
# ---------------------------------------------------------------------------

class TestLinalg:
    def test_reduce_along_dim(self):
        # reduce a 1×4 tile along dim 1 — result is a (1,) tile summing to 10
        data = np.array([[1, 2, 3, 4]], dtype=np.float16)
        t = Tile(data, "f16", data.shape)
        ctx = _ctx_with(**{"%x": t, "%init": Tile(np.zeros((1,), dtype=np.float16), "f16", (1,))})
        result = _call("linalg.reduce", ctx, _make_env(),
                       operands=["%x"],
                       attributes={"reduce_fn": "arith.addf", "dim": 1, "outs_var": "%init"})
        val = float(result.data.flat[0]) if isinstance(result, Tile) else float(result)
        assert abs(val - 10.0) < 0.1

    def test_reduce_full_collapse(self):
        # reduce all elements to a scalar
        t = Tile(np.array([1, 2, 3, 4], dtype=np.float16), "f16", (4,))
        ctx = _ctx_with(**{"%x": t})
        result = _call("linalg.reduce", ctx, _make_env(),
                       operands=["%x"], attributes={"reduce_fn": "arith.addf"})
        assert abs(float(result) - 10.0) < 0.1

    def test_reduce_scalar_input(self):
        # scalar input passes through unchanged
        ctx = _ctx_with(**{"%x": np.float16(5.0)})
        result = _call("linalg.reduce", ctx, _make_env(),
                       operands=["%x"], attributes={"reduce_fn": "arith.addf"})
        assert float(result) == pytest.approx(5.0, rel=1e-2)

    def test_fill(self):
        # fill a tile with a scalar value
        out = Tile(np.zeros((4,), dtype=np.float16), "f16", (4,))
        ctx = _ctx_with(**{"%val": np.float16(3.0), "%out": out})
        result = _call("linalg.fill", ctx, _make_env(), operands=["%val", "%out"])
        assert np.all(result.data == np.float16(3.0))
        assert result.shape == (4,)

    def test_broadcast(self):
        # broadcast 1-D tile to 2-D by repeating along dim 0
        inp = Tile(np.array([1, 2, 3, 4], dtype=np.float16), "f16", (4,))
        out = Tile(np.zeros((2, 4), dtype=np.float16), "f16", (2, 4))
        ctx = _ctx_with(**{"%inp": inp, "%out": out})
        result = _call("linalg.broadcast", ctx, _make_env(),
                       operands=["%inp", "%out"], attributes={"dimensions": [0]})
        assert result.shape == (2, 4)
        assert np.array_equal(result.data[0], inp.data)
        assert np.array_equal(result.data[1], inp.data)

    def test_matmul(self):
        # identity matrix times B equals B
        a = Tile(np.eye(2, dtype=np.float16), "f16", (2, 2))
        b = Tile(np.array([[1, 2], [3, 4]], dtype=np.float16), "f16", (2, 2))
        ctx = _ctx_with(**{"%a": a, "%b": b})
        result = _call("linalg.matmul", ctx, _make_env(), operands=["%a", "%b"])
        assert np.allclose(result.data, b.data, rtol=1e-2)

    def test_generic_reads_outs_arg(self):
        # linalg.generic where the body reads the outs bb0 arg.
        # outs is non-zero — the body adds the input to the existing outs value.
        # If the handler initialised outs to zeros instead of the real outs data,
        # the result would be wrong.
        ins_tile = Tile(np.array([10, 20], dtype=np.float16), "f16", (2,))
        outs_tile = Tile(np.array([1, 2], dtype=np.float16), "f16", (2,))

        ctx = _ctx_with(**{"%ins": ins_tile, "%outs": outs_tile})

        # Use the real dispatcher for region execution
        env = _make_env()
        def _exec_region(context, ops):
            result = None
            for region_op in ops:
                handler = dispatch(region_op.op_type)
                result = handler(region_op, context, env)
                if region_op.result and result is not None:
                    context.set_value(region_op.result, result)
            return result
        env.execute_region = _exec_region

        # Region body (as Operation objects the dispatcher can execute):
        #   %sum = arith.addf %in_arg, %out_arg
        #   linalg.yield %sum
        region_ops = [
            _op("arith.addf", operands=["%in_arg", "%out_arg"], result="%sum"),
            _op("linalg.yield", operands=["%sum"]),
        ]

        op = _op(
            "linalg.generic",
            operands=["%ins", "%outs"],
            attributes={
                "n_ins": 1,
                "indexing_maps": [[0]],
            },
            regions=[[
                Operation(op_type="region.bb0_args", operands=[], attributes={"names": ["%in_arg", "%out_arg"]}, result=None, result_type=None),
            ] + region_ops],
        )

        result = dispatch("linalg.generic")(op, ctx, env)
        # Expected: outs (1,2) + ins (10,20) = (11, 22)
        assert np.allclose(result.data, np.array([11, 22], dtype=np.float16), rtol=1e-2)

    def test_linalg_index(self):
        # linalg.index returns a broadcasting index array for a dimension
        ctx = _make_ctx()
        ctx.set_value("__linalg_shape__", (4, 3))
        result = _call("linalg.index", ctx, _make_env(), attributes={"dim": 0})
        assert isinstance(result, Tile)
        assert result.shape == (4, 1)
        assert np.array_equal(result.data.flatten(), [0, 1, 2, 3])

    def test_linalg_yield(self):
        # linalg.yield wraps values in a _YieldResult
        ctx = _ctx_with(**{"%v": 42})
        result = _call("linalg.yield", ctx, _make_env(), operands=["%v"])
        from ktir_cpu.ops.control_ops import _YieldResult
        assert isinstance(result, _YieldResult)
        assert result.values == [42]

# ---------------------------------------------------------------------------
# tensor dialect exec
# ---------------------------------------------------------------------------

class TestTensor:
    def test_empty(self):
        # creates a zero-filled tile of the requested shape
        result = _call("tensor.empty", _make_ctx(), _make_env(),
                       attributes={"shape": (2, 4), "dtype": "f16"})
        assert isinstance(result, Tile)
        assert result.shape == (2, 4)

    def test_splat(self):
        # broadcast a scalar into a tile of the given shape
        ctx = _ctx_with(**{"%val": np.float16(7.0)})
        result = _call("tensor.splat", ctx, _make_env(),
                       operands=["%val"], attributes={"shape": (4,), "dtype": "f16"})
        assert isinstance(result, Tile)
        assert np.all(result.data == np.float16(7.0))

    def test_extract(self):
        # index into a 2-D tile with two indices
        t = Tile(np.array([[1, 2], [3, 4]], dtype=np.float16), "f16", (2, 2))
        ctx = _ctx_with(**{"%t": t, "%i": 1, "%j": 0})
        result = _call("tensor.extract", ctx, _make_env(), operands=["%t", "%i", "%j"])
        assert float(result) == pytest.approx(3.0, rel=1e-2)

    def test_expand_shape(self):
        # reshape a flat tile to a 2-D shape
        t = Tile(np.array([1, 2, 3, 4], dtype=np.float16), "f16", (4,))
        ctx = _ctx_with(**{"%t": t})
        result = _call("tensor.expand_shape", ctx, _make_env(),
                       operands=["%t"], attributes={"target_shape": (1, 4)})
        assert result.shape == (1, 4)

    def test_collapse_shape(self):
        # collapse a 2-D tile back to 1-D
        t = Tile(np.array([[1, 2], [3, 4]], dtype=np.float16), "f16", (2, 2))
        ctx = _ctx_with(**{"%t": t})
        result = _call("tensor.collapse_shape", ctx, _make_env(),
                       operands=["%t"], attributes={"target_shape": (4,)})
        assert result.shape == (4,)
        assert np.array_equal(result.data, [1, 2, 3, 4])

# ---------------------------------------------------------------------------
# tensor.generate exec
# ---------------------------------------------------------------------------

class TestTensorGenerate:
    def _exec_env(self):
        # tensor.generate calls env.execute_region(context, body) for each index
        # combination.  The default _make_env().execute_region expects callables,
        # but we pass Operation objects.  Override it to dispatch each op through
        # the real handler registry (same pattern as test_generic_reads_outs_arg).
        env = _make_env()
        def _exec_region(context, ops):
            result = None
            for region_op in ops:
                handler = dispatch(region_op.op_type)
                result = handler(region_op, context, env)
                if region_op.result and result is not None:
                    context.set_value(region_op.result, result)
            return result
        env.execute_region = _exec_region
        return env

    def test_generate_1d(self):
        ctx = _make_ctx()
        ctx.set_value("%c2", 2)
        env = self._exec_env()
        region = [
            _op("region.bb0_args", operands=[], attributes={"names": ["%i"]}),
            _op("arith.muli", operands=["%i", "%c2"], result="%val"),
            _op("tensor.yield", operands=["%val"]),
        ]
        op = _op("tensor.generate", operands=[],
                 attributes={"shape": (4,), "dtype": "f16"},
                 regions=[region])
        result = dispatch("tensor.generate")(op, ctx, env)
        assert isinstance(result, Tile)
        assert result.shape == (4,)
        assert np.array_equal(result.data, np.array([0, 2, 4, 6], dtype=np.float16))

    def test_generate_2d(self):
        ctx = _make_ctx()
        env = self._exec_env()
        region = [
            _op("region.bb0_args", operands=[], attributes={"names": ["%i", "%j"]}),
            _op("arith.cmpi", operands=["%i", "%j"],
                attributes={"predicate": "sge"}, result="%cmp"),
            _op("tensor.yield", operands=["%cmp"]),
        ]
        op = _op("tensor.generate", operands=[],
                 attributes={"shape": (3, 3), "dtype": "f16"},
                 regions=[region])
        result = dispatch("tensor.generate")(op, ctx, env)
        assert result.shape == (3, 3)
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float16)
        assert np.array_equal(result.data, expected)


# ---------------------------------------------------------------------------
# scf dialect exec
# ---------------------------------------------------------------------------

class TestScfFunc:
    def test_yield(self):
        # scf.yield wraps operand values in a _YieldResult
        ctx = _ctx_with(**{"%a": 5, "%b": 6})
        result = _call("scf.yield", ctx, _make_env(), operands=["%a", "%b"])
        assert result.values == [5, 6]

    def test_return_with_value(self):
        # func.return returns the operand value
        ctx = _ctx_with(**{"%r": 42})
        assert _call("func.return", ctx, _make_env(), operands=["%r"]) == 42

    def test_return_no_value(self):
        # func.return with no operands returns None
        assert _call("func.return", _make_ctx(), _make_env(), operands=[]) is None

    def test_if_then_branch(self):
        # condition=True runs then_region
        ctx = _ctx_with(**{"%cond": True})
        ran = []
        op = Operation(op_type="scf.if", operands=["%cond"], attributes={},
                       result=None, result_type=None,
                       regions=[[lambda c: ran.append("then")], []])
        env = _make_env()
        env.execute_region = lambda ctx, ops: [f(ctx) for f in ops]
        dispatch("scf.if")(op, ctx, env)
        assert ran == ["then"]

    def test_if_else_branch(self):
        # condition=False runs else_region
        ctx = _ctx_with(**{"%cond": False})
        ran = []
        op = Operation(op_type="scf.if", operands=["%cond"], attributes={},
                       result=None, result_type=None,
                       regions=[[], [lambda c: ran.append("else")]])
        env = _make_env()
        env.execute_region = lambda ctx, ops: [f(ctx) for f in ops]
        dispatch("scf.if")(op, ctx, env)
        assert ran == ["else"]

# ---------------------------------------------------------------------------
# ktdp dialect parsers
# ---------------------------------------------------------------------------

class TestKtdp:
    def test_get_compute_tile_id_single(self):
        # single-dim returns the x coordinate as a scalar
        ctx = _make_ctx(grid_pos=(3, 0, 0), core_id=3)
        assert _call("ktdp.get_compute_tile_id", ctx, _make_env(),
                     result="%id") == 3

    def test_get_compute_tile_id_multi(self):
        # multi-dim returns a tuple of coordinates
        ctx = _make_ctx(grid_pos=(2, 1, 0), core_id=2)
        assert _call("ktdp.get_compute_tile_id", ctx, _make_env(),
                     result=["%x", "%y"]) == (2, 1)

    def test_construct_memory_view(self):
        # creates a TileRef pointing at the given pointer with the given shape
        hbm = HBMSimulator()
        ptr = hbm.allocate(256 * 2)
        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                         lx=LXScratchpad(size_mb=2, core_id=0), hbm=hbm)
        ctx.set_value("%ptr", ptr)
        result = _call("ktdp.construct_memory_view", ctx, _make_env(),
                       operands=["%ptr"],
                       attributes={"shape": (256,), "strides": [1],
                                   "memory_space": "HBM", "dtype": "f16"})
        assert result.base_ptr == ptr
        assert result.shape == (256,)

    def test_load_store_roundtrip(self):
        # load reads data from HBM; store writes it back modified
        from ktir_cpu.ir_types import AccessTile, TileRef
        from ktir_cpu.parser_ast import parse_affine_map

        hbm = HBMSimulator()
        data = np.arange(8, dtype=np.float16)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)

        ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                         lx=LXScratchpad(size_mb=2, core_id=0), hbm=hbm)
        identity_map = parse_affine_map("affine_map<(d0) -> (d0)>")
        tile_ref = TileRef(base_ptr=ptr, shape=(8,), strides=[1],
                           memory_space="HBM", dtype="f16")
        access_tile = AccessTile(parent_ref=tile_ref, shape=(8,),
                                 base_map=identity_map,
                                 coordinate_set=None,
                                 coordinate_order=None)
        ctx.set_value("%acc", access_tile)
        env = _make_env()

        loaded = _call("ktdp.load", ctx, env, operands=["%acc"])
        assert isinstance(loaded, Tile)
        assert np.array_equal(loaded.data, data)

        modified = Tile(data * 2, "f16", (8,))
        ctx.set_value("%tile", modified)
        ctx.set_value("%acc2", access_tile)
        _call("ktdp.store", ctx, env, operands=["%tile", "%acc2"])
        assert np.array_equal(hbm.read(ptr, 8, "f16"), data * 2)
