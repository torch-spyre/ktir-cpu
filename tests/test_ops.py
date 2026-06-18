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
Tests for ktir_cpu/ops/ layer.

Covers: ArithOps, MathOps, GridOps, ControlOps.
CommOps is exercised via the scheduler in tests/test_grid_scheduler.py.
"""

import numpy as np
import pytest

from ktir_cpu.ir_types import Tile
from ktir_cpu.ops.arith_ops import ArithOps
from ktir_cpu.ops.math_ops import MathOps
from ktir_cpu.ops.grid_ops import GridOps
from ktir_cpu.ops.control_ops import ControlOps
from ktir_cpu.grid import CoreContext, GridExecutor
from ktir_cpu.memory import HBMSimulator, LXScratchpad, SpyreMemoryHierarchy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ctx():
    return CoreContext(
        core_id=0,
        grid_pos=(0, 0, 0),
        lx=LXScratchpad(size_mb=2, core_id=0),
        hbm=HBMSimulator(),
    )


def _tile(values, dtype="f16"):
    data = np.array(values, dtype=np.float16)
    return Tile(data, dtype, data.shape)


def _simple_executor(context, ops):
    result = None
    for op in ops:
        result = op(context)
    return result

# ---------------------------------------------------------------------------
# ArithOps
# ---------------------------------------------------------------------------

class TestArithOpsFloat:
    def test_addf(self):
        # element-wise addition of two tiles
        t1, t2 = _tile([1, 2, 3, 4]), _tile([5, 6, 7, 8])
        assert np.array_equal(ArithOps.addf(t1, t2).data, t1.data + t2.data)

    def test_subf(self):
        # element-wise subtraction of two tiles
        t1, t2 = _tile([5, 6, 7, 8]), _tile([1, 2, 3, 4])
        assert np.array_equal(ArithOps.subf(t1, t2).data, t1.data - t2.data)

    def test_mulf(self):
        # element-wise multiplication of two tiles
        t1, t2 = _tile([1, 2, 3, 4]), _tile([5, 6, 7, 8])
        assert np.array_equal(ArithOps.mulf(t1, t2).data, t1.data * t2.data)

    def test_divf(self):
        # element-wise division of two tiles
        t1, t2 = _tile([4, 6, 8, 10]), _tile([2, 2, 2, 2])
        assert np.allclose(ArithOps.divf(t1, t2).data, t1.data / t2.data)

    def test_maxf(self):
        # element-wise maximum of two tiles
        t1, t2 = _tile([1, 5, 3, 8]), _tile([4, 2, 6, 7])
        assert np.array_equal(ArithOps.maxf(t1, t2).data, np.maximum(t1.data, t2.data))

    def test_minf(self):
        # element-wise minimum of two tiles
        t1, t2 = _tile([1, 5, 3, 8]), _tile([4, 2, 6, 7])
        assert np.array_equal(ArithOps.minf(t1, t2).data, np.minimum(t1.data, t2.data))

    def test_maxnumf(self):
        # NaN-aware max; same as maxf for non-NaN inputs
        t1, t2 = _tile([1, 5]), _tile([4, 2])
        assert np.array_equal(ArithOps.maxnumf(t1, t2).data, np.maximum(t1.data, t2.data))

    def test_maxnumf_nan(self):
        # NaN non-propagating (np.fmax): if one operand is NaN, return the other.
        # fmax(NaN, 2) → 2;  fmax(3, NaN) → 3;  fmax(NaN, NaN) → NaN
        t1 = _tile([float('nan'), 3, float('nan')])
        t2 = _tile([2, float('nan'), float('nan')])
        result = ArithOps.maxnumf(t1, t2)
        assert result.data[0] == 2.0
        assert result.data[1] == 3.0
        assert np.isnan(result.data[2])

    def test_minnumf(self):
        t1, t2 = _tile([1, 5, 3, 8]), _tile([4, 2, 6, 7])
        assert np.array_equal(ArithOps.minnumf(t1, t2).data, np.fmin(t1.data, t2.data))

    def test_minnumf_nan(self):
        # NaN non-propagating (np.fmin): if one operand is NaN, return the other.
        # fmin(NaN, 2) → 2;  fmin(3, NaN) → 3;  fmin(NaN, NaN) → NaN
        t1 = _tile([float('nan'), 3, float('nan')])
        t2 = _tile([2, float('nan'), float('nan')])
        result = ArithOps.minnumf(t1, t2)
        assert result.data[0] == 2.0
        assert result.data[1] == 3.0
        assert np.isnan(result.data[2])

    def test_addf_2d_tiles(self):
        # element-wise addf on 4x4 float16 tensors
        data1 = np.arange(16, dtype=np.float16).reshape(4, 4)
        data2 = np.ones((4, 4), dtype=np.float16)
        t1 = Tile(data1, "f16", (4, 4))
        t2 = Tile(data2, "f16", (4, 4))
        result = ArithOps.addf(t1, t2)
        assert result.data.shape == (4, 4)
        assert np.array_equal(result.data, data1 + data2)

    def test_mulf_2d_tiles(self):
        # element-wise mulf on 4x4 float16 tensors
        data1 = np.arange(16, dtype=np.float16).reshape(4, 4)
        data2 = np.full((4, 4), 2.0, dtype=np.float16)
        t1 = Tile(data1, "f16", (4, 4))
        t2 = Tile(data2, "f16", (4, 4))
        result = ArithOps.mulf(t1, t2)
        assert result.data.shape == (4, 4)
        assert np.array_equal(result.data, data1 * data2)

    def test_extf_promotes_f32(self):
        # extf widens f16 → f32
        t = _tile([1, 2, 3])  # f16
        result = ArithOps.extf(t)
        assert result.dtype == "f32"
        assert result.data.dtype == np.float32
        np.testing.assert_array_equal(result.data, np.array([1, 2, 3], dtype=np.float32))
        # scalar path
        s = np.float16(3.0)
        assert ArithOps.extf(s) == np.float32(3.0)

    def test_truncf_passthrough(self):
        # truncf is a no-op in simulation
        t = _tile([1, 2, 3])
        assert ArithOps.truncf(t) is t
        s = np.float16(3.0)
        assert ArithOps.truncf(s) is s

# ---------------------------------------------------------------------------
# ArithOps (Integer)
# ---------------------------------------------------------------------------

class TestArithOpsInt:
    def test_addi_scalars(self):
        # scalar + scalar
        assert ArithOps.addi(3, 4) == 7

    def test_addi_tile_scalar(self):
        # tile + scalar and scalar + tile broadcast
        t = _tile([1, 2, 3])
        assert np.array_equal(ArithOps.addi(t, 10).data, t.data + 10)
        assert np.array_equal(ArithOps.addi(10, t).data, 10 + t.data)

    def test_addi_tile_tile(self):
        # element-wise tile + tile
        t1, t2 = _tile([1, 2, 3]), _tile([4, 5, 6])
        assert np.array_equal(ArithOps.addi(t1, t2).data, t1.data + t2.data)

    def test_muli_scalars(self):
        # scalar * scalar
        assert ArithOps.muli(3, 4) == 12

    def test_muli_tile_scalar(self):
        # tile * scalar and scalar * tile broadcast
        t = _tile([1, 2, 3])
        assert np.array_equal(ArithOps.muli(t, 3).data, t.data * 3)
        assert np.array_equal(ArithOps.muli(3, t).data, 3 * t.data)

    def test_muli_tile_tile(self):
        # element-wise tile * tile
        t1, t2 = _tile([1, 2, 3]), _tile([4, 5, 6])
        assert np.array_equal(ArithOps.muli(t1, t2).data, t1.data * t2.data)

    def test_subi(self):
        # integer subtraction
        assert ArithOps.subi(10, 3) == 7

    def test_divui(self):
        # unsigned integer floor division
        assert ArithOps.divui(10, 3) == 3

    def test_remui(self):
        # unsigned integer remainder
        assert ArithOps.remui(10, 3) == 1

# ---------------------------------------------------------------------------
# ArithOps (Shrui)
# ---------------------------------------------------------------------------

class TestArithOpsShrui:
    def test_scalar_positive(self):
        # shrui on positive scalars matches shrsi
        assert ArithOps.shrui(8, 2) == 2

    def test_scalar_negative_zerofills(self):
        # shrui must zero-fill the high bits, not sign-extend (unlike shrsi)
        # -1 as i32 is 0xFFFFFFFF; >> 1 should give 0x7FFFFFFF, not -1
        result = ArithOps.shrui(-1, 1)
        assert result == 0x7FFFFFFF

    def test_tile_negative_zerofills(self):
        # same semantics for a tile: high bit becomes 0, not 1
        # -1 = 0xFFFFFFFF >> 1 = 0x7FFFFFFF; -2 = 0xFFFFFFFE >> 1 = 0x7FFFFFFF
        data = np.array([-1, -2], dtype=np.int32)
        t = Tile(data, "i32", data.shape)
        result = ArithOps.shrui(t, 1)
        assert isinstance(result, Tile)
        assert np.array_equal(result.data, np.array([0x7FFFFFFF, 0x7FFFFFFF], dtype=np.uint32))

    def test_tile_scalar_shift(self):
        # tile shifted by a scalar amount; use int32 to avoid overflow on assignment
        # -16777216 = 0xFF000000 as int32; >> 8 = 0x00FF0000
        data = np.array([-16777216], dtype=np.int32)
        t = Tile(data, "i32", data.shape)
        result = ArithOps.shrui(t, 8)
        assert isinstance(result, Tile)
        assert result.data[0] == 0x00FF0000


# ---------------------------------------------------------------------------
# ArithOps (Cmpi)
# ---------------------------------------------------------------------------

class TestArithOpsCmpi:
    def test_scalar_predicates(self):
        # all comparison predicates on scalars
        assert ArithOps.cmpi(1, 2, "slt") is True
        assert ArithOps.cmpi(2, 1, "slt") is False
        assert ArithOps.cmpi(1, 1, "eq") is True
        assert ArithOps.cmpi(1, 2, "ne") is True
        assert ArithOps.cmpi(2, 1, "sgt") is True
        assert ArithOps.cmpi(1, 1, "sge") is True
        assert ArithOps.cmpi(1, 2, "sle") is True
        assert ArithOps.cmpi(1, 2, "ult") is True
        assert ArithOps.cmpi(1, 2, "ule") is True
        assert ArithOps.cmpi(2, 1, "ugt") is True
        assert ArithOps.cmpi(1, 1, "uge") is True

    def test_tile_tile(self):
        # element-wise comparison returns i1 tile
        t1, t2 = _tile([1, 5, 3]), _tile([2, 4, 3])
        result = ArithOps.cmpi(t1, t2, "slt")
        assert result.dtype == "i1"
        assert np.array_equal(result.data, t1.data < t2.data)

    def test_tile_scalar(self):
        # tile compared against a scalar, and scalar against a tile
        t = _tile([1, 5, 3])
        assert np.array_equal(ArithOps.cmpi(t, 3, "slt").data, t.data < 3)
        assert np.array_equal(ArithOps.cmpi(3, t, "sgt").data, 3 > t.data)

# ---------------------------------------------------------------------------
# ArithOps (select)
# ---------------------------------------------------------------------------

class TestArithOpsSelect:
    def test_scalar(self):
        # scalar select returns true_val or false_val
        assert ArithOps.select(True, 10, 20) == 10
        assert ArithOps.select(False, 10, 20) == 20

    def test_tile(self):
        # element-wise select via boolean tile condition
        cond = Tile(np.array([True, False, True]), "i1", (3,))
        t1, t2 = _tile([1, 2, 3]), _tile([4, 5, 6])
        result = ArithOps.select(cond, t1, t2)
        assert np.array_equal(result.data, np.array([1, 5, 3], dtype=np.float16))

# ---------------------------------------------------------------------------
# ArithOps (Cmpf)
# ---------------------------------------------------------------------------

class TestArithOpsCmpf:
    def test_result_dtype_is_i1(self):
        # cmpf must return i1 regardless of input dtype — not propagate the float dtype
        t1 = _tile([1.0, 2.0, 3.0])  # f16
        t2 = _tile([3.0, 2.0, 1.0])  # f16
        result = ArithOps.cmpf(t1, t2, "olt")
        assert result.dtype == "i1"
        assert result.data.dtype == np.bool_

# ---------------------------------------------------------------------------
# MathOps
# ---------------------------------------------------------------------------

class TestMathOps:
    def test_exp_tile(self):
        # element-wise exp on a tile
        t = _tile([0, 1, 2])
        result = MathOps.exp(t)
        assert np.allclose(result.data, np.exp(t.data.astype(np.float32)).astype(np.float16), rtol=1e-2)

    def test_exp_scalar(self):
        # scalar exp preserves type
        val = np.float16(1.0)
        assert abs(float(MathOps.exp(val)) - np.e) < 0.01

    def test_sqrt_tile(self):
        # element-wise sqrt on a tile
        t = _tile([1, 4, 9, 16])
        result = MathOps.sqrt(t)
        assert np.allclose(result.data, np.array([1, 2, 3, 4], dtype=np.float16), rtol=1e-2)

    def test_sqrt_scalar(self):
        # scalar sqrt preserves type
        val = np.float16(4.0)
        assert abs(float(MathOps.sqrt(val)) - 2.0) < 0.01

# ---------------------------------------------------------------------------
# Regression: bug C — tile_unary_float scalar path must return float, not
# silently truncate via type(val)() when val is an int.
# ---------------------------------------------------------------------------

class TestTileUnaryFloatScalar:
    def test_returns_float_not_int(self):
        # Passing a Python int scalar must not truncate the result
        result = MathOps.exp(4)
        assert isinstance(result, float), f"expected float, got {type(result)}"
        assert abs(result - np.e ** 4) < 0.1, f"value wrong: {result}"

    def test_sqrt_int_scalar_not_truncated(self):
        result = MathOps.sqrt(2)
        assert isinstance(result, float)
        assert abs(result - 2 ** 0.5) < 0.01


# ---------------------------------------------------------------------------
# Regression: bug E — minsi/maxsi/minui/maxui scalar path must return
# Python int, not numpy.int64 (np.minimum/maximum return numpy scalar types).
# ---------------------------------------------------------------------------

class TestArithOpsMinMaxScalarType:
    def test_minsi_scalar_returns_int(self):
        result = ArithOps.minsi(3, 7)
        assert type(result) is int, f"expected int, got {type(result)}"
        assert result == 3

    def test_maxsi_scalar_returns_int(self):
        result = ArithOps.maxsi(3, 7)
        assert type(result) is int, f"expected int, got {type(result)}"
        assert result == 7

    def test_minui_scalar_returns_int(self):
        result = ArithOps.minui(5, 2)
        assert type(result) is int, f"expected int, got {type(result)}"
        assert result == 2

    def test_maxui_scalar_returns_int(self):
        result = ArithOps.maxui(5, 2)
        assert type(result) is int, f"expected int, got {type(result)}"
        assert result == 5

    def test_minsi_tile_still_returns_tile(self):
        t1 = _tile([1, 5, 3], "i32")
        t2 = _tile([4, 2, 3], "i32")
        result = ArithOps.minsi(t1, t2)
        assert isinstance(result, Tile)
        assert list(result.data) == [1, 2, 3]


# ---------------------------------------------------------------------------
# GridOps
# ---------------------------------------------------------------------------

class TestGridOps:
    def test_gridid(self):
        # returns the grid coordinate for each dimension
        ctx = CoreContext(core_id=5, grid_pos=(5, 0, 0),
                         lx=LXScratchpad(size_mb=2, core_id=5), hbm=HBMSimulator())
        assert GridOps.gridid(ctx, 0) == 5
        assert GridOps.gridid(ctx, 1) == 0
        assert GridOps.gridid(ctx, 2) == 0

    def test_coreid_wildcard(self):
        # -1 wildcard matches all cores; specific coord matches one
        memory = SpyreMemoryHierarchy(num_cores=8)
        grid = GridExecutor(grid_shape=(8, 1, 1), memory=memory)
        ctx = _make_ctx()
        assert len(GridOps.coreid(ctx, [-1], grid)) == 8
        assert GridOps.coreid(ctx, [3], grid) == [3]

    def test_coreid_pads_to_3d(self):
        # coords shorter than 3 are zero-padded
        memory = SpyreMemoryHierarchy(num_cores=4)
        grid = GridExecutor(grid_shape=(4, 1, 1), memory=memory)
        assert GridOps.coreid(_make_ctx(), [2], grid) == [2]


# ---------------------------------------------------------------------------
# ControlOps
# ---------------------------------------------------------------------------

class TestControlOps:
    def test_if_then_branch(self):
        # condition=True runs then_region
        ctx = _make_ctx()
        ran = []
        ControlOps.if_op(ctx, True,
                         [lambda c: ran.append("then") or None],
                         [lambda c: ran.append("else") or None],
                         _simple_executor)
        assert ran == ["then"]

    def test_if_else_branch(self):
        # condition=False runs else_region
        ctx = _make_ctx()
        ran = []
        ControlOps.if_op(ctx, False,
                         [lambda c: ran.append("then") or None],
                         [lambda c: ran.append("else") or None],
                         _simple_executor)
        assert ran == ["else"]

    def test_if_empty_region(self):
        # empty regions return None without error
        assert ControlOps.if_op(_make_ctx(), True, [], [], _simple_executor) is None

    def test_for_op(self):
        # body runs once per step with correct iteration variable
        ctx = _make_ctx()
        iterations = []
        ControlOps.for_op(ctx, 0, 5, 1, "%i", [],
                          lambda c, ops: iterations.append(c.get_value("%i")))
        assert iterations == [0, 1, 2, 3, 4]

    def test_for_op_step_2(self):
        # scf.for with step=2 should visit only even indices
        ctx = _make_ctx()
        iterations = []
        ControlOps.for_op(ctx, 0, 6, 2, "%i", [],
                          lambda c, ops: iterations.append(c.get_value("%i")))
        assert iterations == [0, 2, 4]

    def test_for_op_iter_args_running_sum(self):
        # iter_args carry a running scalar sum across iterations
        from ktir_cpu.ops.control_ops import _YieldResult

        ctx = _make_ctx()

        def body(context, ops):
            acc = context.get_value("%acc")
            i = context.get_value("%i")
            return _YieldResult([acc + i])

        result = ControlOps.for_op(
            ctx, 0, 4, 1, "%i", [],
            body,
            iter_arg_names=["%acc"],
            iter_init_values=[0],
        )
        # sum(0+1+2+3) = 6
        assert result == [6]

    def test_while_op(self):
        # while loop runs until before_region returns False
        ctx = _make_ctx()
        count = [0]

        def executor(context, ops):
            if ops == "before":
                return count[0] < 3
            count[0] += 1
            return None

        ControlOps.while_op(ctx, "before", "after", executor)
        assert count[0] == 3

