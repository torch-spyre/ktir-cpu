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

import operator
import re

import numpy as np

from ..dtypes import to_np_dtype
from ..parser_utils import find_ssa_names
from ..ir_types import Operation, Tile
from ..latency import LatencyCategory as LC
from ..ops.arith_ops import ArithOps
from ._helpers import _float_binop, _int_binop, _unary
from .registry import register, register_parser


def _bool_not(x):
    return ~x if isinstance(x, np.ndarray) else not x



# ---------------------------------------------------------------------------
# Float binary ops
# ---------------------------------------------------------------------------

@register("arith.addf", latency_category=LC.COMPUTE_FLOAT)
def arith__addf(op, context, env):
    return _float_binop(op, context, operator.add)


@register("arith.subf", latency_category=LC.COMPUTE_FLOAT)
def arith__subf(op, context, env):
    return _float_binop(op, context, operator.sub)


@register("arith.mulf", latency_category=LC.COMPUTE_FLOAT)
def arith__mulf(op, context, env):
    return _float_binop(op, context, operator.mul)


@register("arith.divf", latency_category=LC.COMPUTE_FLOAT)
def arith__divf(op, context, env):
    return _float_binop(op, context, operator.truediv)


@register("arith.remf", latency_category=LC.COMPUTE_FLOAT)
def arith__remf(op, context, env):
    return _float_binop(op, context, operator.mod)


# ---------------------------------------------------------------------------
# Float unary ops
# ---------------------------------------------------------------------------

@register("arith.negf", latency_category=LC.COMPUTE_FLOAT)
def arith__negf(op, context, env):
    return _unary(op, context, ArithOps.negf)


@register("arith.absf", latency_category=LC.COMPUTE_FLOAT)
def arith__absf(op, context, env):
    return _unary(op, context, ArithOps.absf)


# ---------------------------------------------------------------------------
# Float min/max
# ---------------------------------------------------------------------------

# TODO: consider deprecating arith.maxf / arith.minf aliases — these were
# renamed to arith.maximumf / arith.minimumf in upstream MLIR.
@register("arith.maxf", "arith.maximumf", latency_category=LC.COMPUTE_FLOAT)
def arith__maxf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.maxf(a, b)


@register("arith.maxnumf", latency_category=LC.COMPUTE_FLOAT)
def arith__maxnumf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.maxnumf(a, b)


@register("arith.minf", "arith.minimumf", latency_category=LC.COMPUTE_FLOAT)
def arith__minf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.minf(a, b)


@register("arith.minnumf", latency_category=LC.COMPUTE_FLOAT)
def arith__minnumf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.minnumf(a, b)


# ---------------------------------------------------------------------------
# Float comparison
# ---------------------------------------------------------------------------

@register("arith.cmpf", latency_category=LC.COMPUTE_FLOAT)
def arith__cmpf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    predicate = op.attributes.get("predicate", "oeq")
    return ArithOps.cmpf(a, b, predicate)


# ---------------------------------------------------------------------------
# Integer binary ops
# ---------------------------------------------------------------------------

@register("arith.addi", latency_category=LC.COMPUTE_INT)
def arith__addi(op, context, env):
    return _int_binop(op, context, operator.add)


@register("arith.subi", latency_category=LC.COMPUTE_INT)
def arith__subi(op, context, env):
    return _int_binop(op, context, operator.sub)


@register("arith.muli", latency_category=LC.COMPUTE_INT)
def arith__muli(op, context, env):
    return _int_binop(op, context, operator.mul)


@register("arith.divui", latency_category=LC.COMPUTE_INT)
def arith__divui(op, context, env):
    return _int_binop(op, context, operator.floordiv)


def _truncdiv(a, b):
    # MLIR divsi truncates toward zero; Python // floors toward -inf.
    return np.trunc(a / b).astype(np.asarray(a).dtype)


def _truncrem(a, b):
    # MLIR remsi is remainder after truncating division: a - (a/b)*b.
    return np.asarray(a) - _truncdiv(a, b) * np.asarray(b)


@register("arith.divsi", latency_category=LC.COMPUTE_INT)
def arith__divsi(op, context, env):
    return _int_binop(op, context, _truncdiv)


@register("arith.ceildivsi", latency_category=LC.COMPUTE_INT)
def arith__ceildivsi(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    return ArithOps.ceildivsi(val1, val2)


@register("arith.floordivsi", latency_category=LC.COMPUTE_INT)
def arith__floordivsi(op, context, env):
    return _int_binop(op, context, operator.floordiv)


@register("arith.remui", latency_category=LC.COMPUTE_INT)
def arith__remui(op, context, env):
    return _int_binop(op, context, operator.mod)


@register("arith.remsi", latency_category=LC.COMPUTE_INT)
def arith__remsi(op, context, env):
    return _int_binop(op, context, _truncrem)


@register("arith.minsi", latency_category=LC.COMPUTE_INT)
def arith__minsi(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.minsi(a, b)


@register("arith.maxsi", latency_category=LC.COMPUTE_INT)
def arith__maxsi(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.maxsi(a, b)


@register("arith.minui", latency_category=LC.COMPUTE_INT)
def arith__minui(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.minui(a, b)


@register("arith.maxui", latency_category=LC.COMPUTE_INT)
def arith__maxui(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.maxui(a, b)


@register("arith.ceildivui", latency_category=LC.COMPUTE_INT)
def arith__ceildivui(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    return ArithOps.ceildivui(a, b)


# ---------------------------------------------------------------------------
# Integer bitwise ops
# ---------------------------------------------------------------------------

@register("arith.andi", latency_category=LC.COMPUTE_INT)
def arith__andi(op, context, env):
    return _int_binop(op, context, operator.and_)


@register("arith.ori", latency_category=LC.COMPUTE_INT)
def arith__ori(op, context, env):
    return _int_binop(op, context, operator.or_)


@register("arith.xori", latency_category=LC.COMPUTE_INT)
def arith__xori(op, context, env):
    return _int_binop(op, context, operator.xor)


@register("arith.shli", latency_category=LC.COMPUTE_INT)
def arith__shli(op, context, env):
    return _int_binop(op, context, operator.lshift)


@register("arith.shrsi", latency_category=LC.COMPUTE_INT)
def arith__shrsi(op, context, env):
    return _int_binop(op, context, operator.rshift)


@register("arith.shrui", latency_category=LC.COMPUTE_INT)
def arith__shrui(op, context, env):
    val1 = context.get_value(op.operands[0])
    val2 = context.get_value(op.operands[1])
    return ArithOps.shrui(val1, val2)


# ---------------------------------------------------------------------------
# Integer comparison
# ---------------------------------------------------------------------------

@register("arith.cmpi", latency_category=LC.COMPUTE_FLOAT)
def arith__cmpi(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    predicate = op.attributes["predicate"]
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
    return Tile(result, "i1", result.shape) if is_tile else result


@register("arith.cmpf", latency_category=LC.COMPUTE_FLOAT)
def arith__cmpf(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    predicate = op.attributes["predicate"]
    is_tile = isinstance(a, Tile) or isinstance(b, Tile)
    if is_tile:
        lhs = a.data if isinstance(a, Tile) else np.full(b.shape, a, dtype=b.data.dtype)
        rhs = b.data if isinstance(b, Tile) else np.full(a.shape, b, dtype=a.data.dtype)
    else:
        lhs, rhs = float(a), float(b)
    # Ordered (o*): numpy default — returns False when NaN is involved.
    # Unordered (u*): same comparison, but OR with nan_either.
    cmp_ops = {
        "false": lambda: np.zeros_like(lhs, dtype=bool) if is_tile else False,
        "oeq": lambda: lhs == rhs,  "one": lambda: (lhs != rhs) & _bool_not(np.isnan(lhs) | np.isnan(rhs)),
        "olt": lambda: lhs < rhs,   "ole": lambda: lhs <= rhs,
        "ogt": lambda: lhs > rhs,   "oge": lambda: lhs >= rhs,
        "ueq": lambda: (lhs == rhs) | (np.isnan(lhs) | np.isnan(rhs)),
        "une": lambda: lhs != rhs,
        "ult": lambda: (lhs < rhs)  | (np.isnan(lhs) | np.isnan(rhs)),
        "ule": lambda: (lhs <= rhs) | (np.isnan(lhs) | np.isnan(rhs)),
        "ugt": lambda: (lhs > rhs)  | (np.isnan(lhs) | np.isnan(rhs)),
        "uge": lambda: (lhs >= rhs) | (np.isnan(lhs) | np.isnan(rhs)),
        "ord": lambda: _bool_not(np.isnan(lhs) | np.isnan(rhs)),
        "uno": lambda: np.isnan(lhs) | np.isnan(rhs),
        "true": lambda: np.ones_like(lhs, dtype=bool) if is_tile else True,
    }
    if predicate not in cmp_ops:
        raise NotImplementedError(f"arith.cmpf: unsupported predicate '{predicate}'")
    result = cmp_ops[predicate]()
    return Tile(result, "i1", result.shape) if is_tile else result


# ---------------------------------------------------------------------------
# Constants & casts
# ---------------------------------------------------------------------------

@register("arith.constant")
def arith__constant(op, context, env):
    value = op.attributes.get("value", 0)
    if op.attributes.get("is_tensor"):
        shape = op.attributes["shape"]
        dtype_str = op.attributes.get("dtype", "f16")
        np_dtype = to_np_dtype(dtype_str)
        return Tile(np.full(shape, value, dtype=np_dtype), dtype_str, shape)
    return value


@register("arith.extf")
def arith__extf(op, context, env):
    return _unary(op, context, ArithOps.extf, np.float32)


@register("arith.truncf")
def arith__truncf(op, context, env):
    return _unary(op, context, ArithOps.truncf)


@register("arith.extsi")
def arith__extsi(op, context, env):
    return _unary(op, context, lambda t: Tile(t.data.astype(np.int64), "i64", t.shape), int)


@register("arith.extui")
def arith__extui(op, context, env):
    return _unary(op, context, ArithOps.extui, int)


@register("arith.trunci")
def arith__trunci(op, context, env):
    return _unary(op, context, ArithOps.trunci, int)


@register("arith.sitofp")
def arith__sitofp(op, context, env):
    dtype = op.result_type or "f32"
    return _unary(op, context, lambda v: ArithOps.sitofp(v, dtype))


@register("arith.uitofp")
def arith__uitofp(op, context, env):
    return _unary(op, context, ArithOps.uitofp, float)


@register("arith.fptosi")
def arith__fptosi(op, context, env):
    return _unary(op, context, ArithOps.fptosi, int)


@register("arith.fptoui")
def arith__fptoui(op, context, env):
    return _unary(op, context, ArithOps.fptoui, int)


@register("arith.index_cast")
def arith__index_cast(op, context, env):
    return int(context.get_value(op.operands[0]))


@register("arith.index_castui")
def arith__index_castui(op, context, env):
    return int(context.get_value(op.operands[0]))


@register("arith.convertf")
def arith__convertf(op, context, env):
    return _unary(op, context, ArithOps.convertf)


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


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------

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

    # Three syntax forms for arith.constant:
    #
    # Form 1 (braced):   {dense<val> : inner_type} : result_type
    #                     {val : inner_type} : result_type
    # Form 2 (dense):    dense<val> : tensor<NxMxdtype>
    # Form 3 (scalar):   val : dtype
    #                     e.g. 0xFF800000 : f32, 42 : index, 0.0 : f16
    #
    # All forms pass dtype to parse_numeric so hex literals are correctly
    # interpreted as IEEE 754 bit patterns for float types.

    braced_match = re.match(r'\{([^}]+)\}\s*:\s*(.+)$', rest)
    if braced_match:
        # Form 1: {inner} : result_type
        inner = braced_match.group(1).strip()
        result_type = braced_match.group(2).strip()

        # Resolve element dtype from result_type (could be "f32" or "tensor<4xf32>")
        _type_info = parse_tensor_type(result_type)
        elem_dtype = _type_info.get("dtype") if _type_info else result_type

        dense_match = re.match(r'dense<([^>]+)>', inner)
        if dense_match:
            value = parse_numeric(dense_match.group(1), dtype=elem_dtype)
        else:
            typed_val = re.match(r'(.+?)\s*:\s*\S+', inner)
            if typed_val:
                value = parse_numeric(typed_val.group(1).strip(), dtype=elem_dtype)
            else:
                value = parse_numeric(inner, dtype=elem_dtype)

        if _type_info and 'tensor<' in result_type:
            attributes["shape"] = _type_info["shape"]
            attributes["dtype"] = _type_info.get("dtype", "f16")
            attributes["is_tensor"] = True
    else:
        # Form 2: dense<value> : type.  Covers:
        #   dense<0.0> : tensor<4xf16>       (splat tensor constant)
        #   dense<42> : tensor<1xi32>         (scalar tensor constant)
        dense_match = re.match(r'dense<([^>]+)>\s*:\s*(.+)$', rest)
        # Form 3: scalar value : type.  Covers:
        #   42 : index              (decimal integer)
        #   0.0 : f32               (float)
        #   -1.5e-3 : f16           (scientific notation)
        #   0xFF800000 : f32        (hex float — IEEE 754 bit pattern for -inf)
        #   0xFF800000 : i32        (hex integer — kept as plain int)
        simple_match = re.match(r'(-?(?:0[xX][0-9a-fA-F]+|[\d.eE+\-]+))\s*:\s*(.+)$', rest)
        if dense_match:
            result_type = dense_match.group(2).strip()
            type_info = parse_tensor_type(result_type)
            elem_dtype = type_info.get("dtype") if type_info else None
            value = parse_numeric(dense_match.group(1), dtype=elem_dtype)
            attributes["shape"] = type_info["shape"]
            attributes["dtype"] = type_info["dtype"]
            attributes["is_tensor"] = True
        elif simple_match:
            result_type = simple_match.group(2).strip()
            value = parse_numeric(simple_match.group(1), dtype=result_type)
        else:
            # Defensive fallback: type-only with no parseable value — defaults to 0.
            # No known MLIR examples hit this path; kept for robustness.
            #   : tensor<1x64xf16>     (zero-initialized tensor)
            #   : index                (zero scalar)

            type_only_match = re.match(r':\s*(.+)$', rest)
            if type_only_match:
                result_type = type_only_match.group(1).strip()
                if result_type and 'tensor<' in result_type:
                    type_info = parse_tensor_type(result_type)
                    if type_info:
                        attributes["shape"] = type_info["shape"]
                        attributes["dtype"] = type_info.get("dtype", "f16")
                        attributes["is_tensor"] = True

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

    pred_match = re.search(r'arith\.cmpi\s+(eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)', op_text)
    if not pred_match:
        raise ValueError(f"arith.cmpi: no valid predicate found in: {op_text!r}")
    predicate = pred_match.group(1)

    operands = find_ssa_names(op_text)
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


@register_parser("arith.cmpf")
def parse_arith_cmpf(op_text, parse_ctx):
    result_match = re.match(r'(%\w+)\s*=\s*arith\.cmpf\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    pred_match = re.search(
        r'arith\.cmpf\s+(true|false|oeq|ogt|oge|olt|ole|one|ord|ueq|ugt|uge|ult|ule|une|uno)', op_text)
    if not pred_match:
        raise ValueError(f"arith.cmpf: no valid predicate found in: {op_text!r}")
    predicate = pred_match.group(1)

    operands = find_ssa_names(op_text)
    operands = [o for o in operands if o != result_name]

    result_type = "i1"
    type_match = re.search(r':\s*(.+)$', op_text)
    if type_match:
        result_type = type_match.group(1).strip()

    return Operation(
        result=result_name,
        op_type="arith.cmpf",
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
