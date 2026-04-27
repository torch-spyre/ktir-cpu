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

"""Arith dialect handlers — arithmetic on scalars and tiles."""

import re

import numpy as np

from ..ir_types import Operation, Tile
from ..latency import LatencyCategory as LC
from ..ops.arith_ops import ArithOps
from .registry import register, register_parser


def _is_scalar(v):
    """Return True if *v* is a scalar numeric value (not a Tile)."""
    return isinstance(v, (int, float, np.integer, np.floating))


def _scalar_binop(a, b, py_op):
    """Perform *py_op* on two scalars, preserving np.float16 when possible."""
    if isinstance(a, (np.float16, np.floating)) or isinstance(b, (np.float16, np.floating)):
        return np.float16(py_op(float(a), float(b)))
    return py_op(float(a), float(b))



# ---------------------------------------------------------------------------
# Float binary ops
# ---------------------------------------------------------------------------

@register("arith.addf", latency_category=LC.COMPUTE_FLOAT)
def arith__addf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    if _is_scalar(a) and _is_scalar(b):
        return _scalar_binop(a, b, lambda x, y: x + y)
    if _is_scalar(a) and isinstance(b, Tile):
        return Tile(np.float16(a) + b.data, b.dtype, b.shape)
    if isinstance(a, Tile) and _is_scalar(b):
        return Tile(a.data + np.float16(b), a.dtype, a.shape)
    return ArithOps.addf(a, b)


@register("arith.subf", latency_category=LC.COMPUTE_FLOAT)
def arith__subf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    if _is_scalar(a) and _is_scalar(b):
        return _scalar_binop(a, b, lambda x, y: x - y)
    if isinstance(a, Tile) and isinstance(b, Tile):
        return ArithOps.subf(a, b)
    if isinstance(a, Tile):
        return Tile(a.data - np.float16(b), a.dtype, a.shape)
    return Tile(np.float16(a) - b.data, b.dtype, b.shape)


@register("arith.mulf", latency_category=LC.COMPUTE_FLOAT)
def arith__mulf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    if _is_scalar(a) and _is_scalar(b):
        return _scalar_binop(a, b, lambda x, y: x * y)
    if isinstance(a, Tile) and isinstance(b, Tile):
        return ArithOps.mulf(a, b)
    if isinstance(a, Tile):
        return Tile(a.data * np.float16(b), a.dtype, a.shape)
    return Tile(np.float16(a) * b.data, b.dtype, b.shape)


@register("arith.divf", latency_category=LC.COMPUTE_FLOAT)
def arith__divf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    if _is_scalar(a) and _is_scalar(b):
        return _scalar_binop(a, b, lambda x, y: x / y)
    if isinstance(a, Tile) and isinstance(b, Tile):
        return ArithOps.divf(a, b)
    if isinstance(a, Tile):
        return Tile(a.data / np.float16(b), a.dtype, a.shape)
    return Tile(np.float16(a) / b.data, b.dtype, b.shape)


# ---------------------------------------------------------------------------
# Integer ops
# ---------------------------------------------------------------------------

@register("arith.addi", latency_category=LC.COMPUTE_INT)
def arith__addi(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    if isinstance(val1, Tile) or isinstance(val2, Tile):
        t1 = val1 if isinstance(val1, Tile) else Tile(np.full(val2.shape, val1, dtype=val2.data.dtype), val2.dtype, val2.shape)
        t2 = val2 if isinstance(val2, Tile) else Tile(np.full(val1.shape, val2, dtype=val1.data.dtype), val1.dtype, val1.shape)
        return Tile(t1.data + t2.data, t1.dtype, t1.shape)
    return ArithOps.addi(val1, val2)


@register("arith.subi", latency_category=LC.COMPUTE_INT)
def arith__subi(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    return ArithOps.subi(val1, val2)


@register("arith.muli", latency_category=LC.COMPUTE_INT)
def arith__muli(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    if isinstance(val1, Tile) or isinstance(val2, Tile):
        t1 = val1 if isinstance(val1, Tile) else Tile(np.full(val2.shape, val1, dtype=val2.data.dtype), val2.dtype, val2.shape)
        t2 = val2 if isinstance(val2, Tile) else Tile(np.full(val1.shape, val2, dtype=val1.data.dtype), val1.dtype, val1.shape)
        return Tile(t1.data * t2.data, t1.dtype, t1.shape)
    return ArithOps.muli(val1, val2)


@register("arith.divui", latency_category=LC.COMPUTE_INT)
def arith__divui(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    return ArithOps.divui(val1, val2)


@register("arith.remui", latency_category=LC.COMPUTE_INT)
def arith__remui(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    return ArithOps.remui(val1, val2)


# ---------------------------------------------------------------------------
# Constants & casts
# ---------------------------------------------------------------------------

@register("arith.constant")
def arith__constant(op, context, env):
    value = op.attributes.get("value", 0)
    if op.attributes.get("is_tensor"):
        shape = op.attributes["shape"]
        dtype_str = op.attributes.get("dtype", "f16")
        np_dtype = np.float16 if dtype_str == "f16" else np.int32
        return Tile(np.full(shape, value, dtype=np_dtype), dtype_str, shape)
    return value


@register("arith.maxf", "arith.maximumf", latency_category=LC.COMPUTE_FLOAT)
def arith__maxf(op, context, env):
    tile1 = context.get_value(op.operands[0])
    tile2 = context.get_value(op.operands[1])
    return ArithOps.maxf(tile1, tile2)


@register("arith.maxnumf", latency_category=LC.COMPUTE_FLOAT)
def arith__maxnumf(op, context, env):
    tile1 = context.get_value(op.operands[0])
    tile2 = context.get_value(op.operands[1])
    return ArithOps.maxnumf(tile1, tile2)


@register("arith.extf")
def arith__extf(op, context, env):
    return ArithOps.extf(context.get_value(op.operands[0]))


@register("arith.truncf")
def arith__truncf(op, context, env):
    return ArithOps.truncf(context.get_value(op.operands[0]))


@register("arith.bitcast")
def arith__bitcast(op, context, env):
    """Reinterpret bits between integer and float types of the same width."""
    val = context.get_value(op.operands[0])
    dst_type = op.attributes.get("dst_type", "f32")
    if isinstance(val, Tile):
        if dst_type == "f32":
            return Tile(val.data.view(np.float32), "f32", val.shape)
        if dst_type in ("i32", "si32"):
            return Tile(val.data.view(np.int32), "i32", val.shape)
        raise NotImplementedError(f"arith.bitcast: unsupported dst_type '{dst_type}' for Tile")
    # Scalar path — convert via raw bytes to handle both signed and unsigned
    # integer inputs (e.g. 0xFF800000 from regex as unsigned, -8388608 from
    # MLIR frontend as signed — both represent the same bit pattern).
    if dst_type == "f32":
        return float(np.frombuffer(int(val).to_bytes(4, "little", signed=(val < 0)),
                                   dtype=np.float32)[0])
    if dst_type in ("i32", "si32"):
        return int(np.frombuffer(np.float32(val).tobytes(), dtype=np.int32)[0])
    raise NotImplementedError(f"arith.bitcast: unsupported dst_type '{dst_type}' for scalar")


@register("arith.extsi")
def arith__extsi(op, context, env):
    """Sign-extend integer (e.g., i32 -> i64). In Python, ints have arbitrary precision."""
    return int(context.get_value(op.operands[0]))


@register("arith.index_cast")
def arith__index_cast(op, context, env):
    return int(context.get_value(op.operands[0]))


@register("arith.sitofp")
def arith__sitofp(op, context, env):
    return np.float16(context.get_value(op.operands[0]))


@register("arith.cmpi", latency_category=LC.COMPUTE_FLOAT)
def arith__cmpi(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    predicate = op.attributes.get("predicate", "slt")
    is_tile = isinstance(a, Tile) or isinstance(b, Tile)
    if is_tile:
        lhs = a.data if isinstance(a, Tile) else np.full(b.shape, a, dtype=b.data.dtype)
        rhs = b.data if isinstance(b, Tile) else np.full(a.shape, b, dtype=a.data.dtype)
    else:
        lhs, rhs = a, b
    # Unsigned predicates use the same comparisons as signed: this interpreter uses
    # Python ints / NumPy arrays which have no fixed-width overflow, so sign-bit
    # reinterpretation never occurs.
    cmp_ops = {
        "slt": lambda: lhs < rhs,  "ult": lambda: lhs < rhs,
        "sle": lambda: lhs <= rhs, "ule": lambda: lhs <= rhs,
        "sgt": lambda: lhs > rhs,  "ugt": lambda: lhs > rhs,
        "sge": lambda: lhs >= rhs, "uge": lambda: lhs >= rhs,
        "eq":  lambda: lhs == rhs,
        "ne":  lambda: lhs != rhs,
    }
    if predicate not in cmp_ops:
        raise NotImplementedError(f"arith.cmpi: unsupported predicate '{predicate}'")
    result = cmp_ops[predicate]()
    return Tile(result, (a if isinstance(a, Tile) else b).dtype, result.shape) if is_tile else result


@register("arith.select", latency_category=LC.COMPUTE_FLOAT)
def arith__select(op, context, env):
    cond = context.get_value(op.operands[0])
    true_val = context.get_value(op.operands[1])
    false_val = context.get_value(op.operands[2])
    # Also accept bare np.ndarray conditions (e.g. from cmpi on Tiles which
    # returns a Tile whose .data is ndarray) and preserve the true/false value
    # dtype and shape rather than forcing float16 and cond.shape — the old
    # version dropped integer dtypes by casting to f16 and used the condition
    # tile's shape which could differ from the data tile's.
    if isinstance(cond, (Tile, np.ndarray)):
        c = cond.data if isinstance(cond, Tile) else cond
        t = true_val.data if isinstance(true_val, Tile) else true_val
        f = false_val.data if isinstance(false_val, Tile) else false_val
        result = np.where(c, t, f)
        ref = true_val if isinstance(true_val, Tile) else false_val
        dtype = ref.dtype if isinstance(ref, Tile) else "f16"
        shape = ref.shape if isinstance(ref, Tile) else result.shape
        return Tile(result, dtype, shape)
    return true_val if cond else false_val


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@register_parser("arith.constant")
def parse_arith_constant(op_text, parse_ctx):
    from ..parser_utils import parse_numeric, parse_tensor_type

    result_match = re.match(r'(%\w+)\s*=\s*arith\.constant\s*(.*)', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    rest = result_match.group(2).strip()

    value = None
    result_type = None
    attributes = {}

    braced_match = re.match(r'\{([^}]+)\}\s*:\s*(.+)$', rest)
    if braced_match:
        inner = braced_match.group(1).strip()
        result_type = braced_match.group(2).strip()

        dense_match = re.match(r'dense<([^>]+)>', inner)
        if dense_match:
            value = parse_numeric(dense_match.group(1))
        else:
            typed_val = re.match(r'(.+?)\s*:\s*\S+', inner)
            if typed_val:
                value = parse_numeric(typed_val.group(1).strip())
            else:
                value = parse_numeric(inner)

        # If result_type is a tensor, mark as tensor constant
        if result_type and 'tensor<' in result_type:
            type_info = parse_tensor_type(result_type)
            if type_info:
                attributes["shape"] = type_info["shape"]
                attributes["dtype"] = type_info.get("dtype", "f16")
                attributes["is_tensor"] = True
    else:
        # Match unbraced dense<value> followed by `: type`.  Covers:
        #   dense<0.0> : tensor<4xf16>       (splat tensor constant)
        #   dense<42> : tensor<1xi32>         (scalar tensor constant)
        dense_match = re.match(r'dense<([^>]+)>\s*:\s*(.+)$', rest)
        # Match a scalar value followed by `: type`.  Covers:
        #   42 : index              (decimal integer)
        #   0.0 : f32               (float)
        #   -1.5e-3 : f16           (scientific notation)
        #   0xFF800000 : i32        (hex integer, e.g. -inf bit pattern)
        simple_match = re.match(r'(-?(?:0[xX][0-9a-fA-F]+|[\d.eE+\-]+))\s*:\s*(.+)$', rest)
        if dense_match:
            value = parse_numeric(dense_match.group(1))
            result_type = dense_match.group(2).strip()
            type_info = parse_tensor_type(result_type)
            attributes["shape"] = type_info["shape"]
            attributes["dtype"] = type_info["dtype"]
            attributes["is_tensor"] = True
        elif simple_match:
            value = parse_numeric(simple_match.group(1))
            result_type = simple_match.group(2).strip()
        else:
            type_only_match = re.match(r':\s*(.+)$', rest)
            if type_only_match:
                result_type = type_only_match.group(1).strip()
                if result_type and 'tensor<' in result_type:
                    type_info = parse_tensor_type(result_type)
                    if type_info:
                        shape = type_info["shape"]
                        dtype_str = type_info.get("dtype", "f16")
                        value = 0
                        attributes["shape"] = shape
                        attributes["dtype"] = dtype_str
                        attributes["is_tensor"] = True
                else:
                    value = 0
            else:
                value = 0
                result_type = "unknown"

    if value is None:
        value = 0

    attributes["value"] = value

    return Operation(
        result=result_name,
        op_type="arith.constant",
        operands=[],
        attributes=attributes,
        result_type=result_type
    )


@register_parser("arith.cmpi")
def parse_arith_cmpi(op_text, parse_ctx):
    result_match = re.match(r'(%\w+)\s*=\s*arith\.cmpi\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    predicate = "slt"
    pred_match = re.search(r'arith\.cmpi\s+(eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)', op_text)
    if pred_match:
        predicate = pred_match.group(1)

    operands = re.findall(r'%\w+', op_text)
    operands = [o for o in operands if o != result_name]

    result_type = "unknown"
    type_match = re.search(r':\s*(.+)$', op_text)
    if type_match:
        result_type = type_match.group(1).strip()

    return Operation(
        result=result_name,
        op_type="arith.cmpi",
        operands=operands,
        attributes={"predicate": predicate},
        result_type=result_type
    )


@register_parser("arith.sitofp")
def parse_arith_sitofp(op_text, parse_ctx):
    result_match = re.match(r'(%\w+)\s*=\s*arith\.sitofp\s+(%\w+)', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    operand = result_match.group(2)

    result_type = "f16"
    to_match = re.search(r'to\s+(f\d+)', op_text)
    if to_match:
        result_type = to_match.group(1)

    return Operation(
        result=result_name,
        op_type="arith.sitofp",
        operands=[operand],
        attributes={},
        result_type=result_type
    )


@register_parser("arith.bitcast")
def parse_arith_bitcast(op_text, parse_ctx):
    # %result = arith.bitcast %src : src_type to dst_type
    result_match = re.match(r'(%\w+)\s*=\s*arith\.bitcast\s+(%\w+)', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    operand = result_match.group(2)

    dst_type = "f32"
    to_match = re.search(r'\bto\s+(\S+)\s*$', op_text)
    if to_match:
        dst_type = to_match.group(1)

    return Operation(
        result=result_name,
        op_type="arith.bitcast",
        operands=[operand],
        attributes={"dst_type": dst_type},
        result_type=dst_type
    )
