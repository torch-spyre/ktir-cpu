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

"""Shared dispatch helpers for dialect op handlers.

These helpers eliminate the repetitive Tile-vs-scalar branching that
appears across arith_ops.py, math_ops.py, and other dialect files.
See issue #23 for motivation.

Public API
----------
_is_scalar(v)                  — True if v is a numeric scalar (not Tile)
_float_binop(op, context, fn)  — fetch two operands, dispatch float binop
_int_binop(op, context, fn)    — fetch two operands, dispatch int binop
_unary(op, context, tile_fn, scalar_fn) — fetch one operand, dispatch unary
unwrap_yield(result)           — unwrap a _YieldResult sentinel; pass through anything else
"""

import numpy as np

from ..ir_types import Tile


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating))


def _float_binop(op, context, fn):
    """Fetch two operands and apply a float element-wise function.

    *fn* receives two numpy arrays (or scalars coerced to np.float16) and
    returns a numpy array.  Handles scalar/scalar, Tile/Tile, and mixed cases.
    """
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    if _is_scalar(a) and _is_scalar(b):
        result = fn(np.float16(a), np.float16(b))
        return np.float16(result) if not isinstance(result, np.float16) else result
    if isinstance(a, Tile) and isinstance(b, Tile):
        return Tile(fn(a.data, b.data), a.dtype, a.shape)
    if isinstance(a, Tile):
        return Tile(fn(a.data, np.float16(b)), a.dtype, a.shape)
    return Tile(fn(np.float16(a), b.data), b.dtype, b.shape)


def _int_binop(op, context, fn):
    """Fetch two operands and apply an integer element-wise function.

    *fn* receives two numpy arrays (or Python ints) and returns a numpy
    array or int.  Scalars are broadcast into Tiles when one operand is a Tile.
    """
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    if isinstance(a, Tile) or isinstance(b, Tile):
        if isinstance(a, Tile) and isinstance(b, Tile):
            return Tile(fn(a.data, b.data), a.dtype, a.shape)
        if isinstance(a, Tile):
            return Tile(fn(a.data, int(b)), a.dtype, a.shape)
        return Tile(fn(int(a), b.data), b.dtype, b.shape)
    return fn(a, b)


def _unary(op, context, tile_fn, scalar_fn=None):
    """Fetch one operand and apply a unary function.

    *tile_fn* is called when the operand is a Tile.
    *scalar_fn* is called for scalars; defaults to *tile_fn* if omitted.
    """
    val = context.get_value(op.operands[0])
    if isinstance(val, Tile):
        return tile_fn(val)
    return (scalar_fn or tile_fn)(val)


def unwrap_yield(result):
    """Unwrap a _YieldResult sentinel produced by scf.yield / linalg.yield.

    Dialect region drivers (scf.if, scf.for, linalg.generic) call this on
    the value returned by execute_region so they always receive plain values.
    Returns the single value for a 1-element yield, a tuple for multi-value,
    None for an empty yield, or *result* unchanged if it is not a _YieldResult.
    """
    from ..ops.control_ops import _YieldResult
    if not isinstance(result, _YieldResult):
        return result
    values = result.values
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return tuple(values)
