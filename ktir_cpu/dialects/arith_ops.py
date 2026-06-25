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
from ._helpers import _float_binop, _int_binop, _unary, _binop_via_op
from .registry import register, register_parser


def _bool_not(x):
    return ~x if isinstance(x, np.ndarray) else not x


def _truncdiv(a, b):
    # MLIR divsi truncates toward zero; Python // floors toward -inf.
    return np.trunc(a / b).astype(np.asarray(a).dtype)


def _truncrem(a, b):
    # MLIR remsi is remainder after truncating division: a - (a/b)*b.
    return np.asarray(a) - _truncdiv(a, b) * np.asarray(b)


# ---------------------------------------------------------------------------
# Float binary ops  (Pattern A.1)
# ---------------------------------------------------------------------------

_FLOAT_BINOPS = {
    "arith.addf": operator.add,
    "arith.subf": operator.sub,
    "arith.mulf": operator.mul,
    "arith.divf": operator.truediv,
    "arith.remf": operator.mod,
}
for _name, _fn in _FLOAT_BINOPS.items():
    @register(_name, latency_category=LC.COMPUTE_FLOAT)
    def _(op, context, env, _fn=_fn):
        return _float_binop(op, context, _fn)


# ---------------------------------------------------------------------------
# Float unary ops  (Pattern A.4 — COMPUTE_FLOAT cluster)
# ---------------------------------------------------------------------------

_FLOAT_UNARY_OPS = {
    "arith.negf": (ArithOps.negf, None),
    "arith.absf": (ArithOps.absf, None),
}
for _name, (_fn, _sfn) in _FLOAT_UNARY_OPS.items():
    @register(_name, latency_category=LC.COMPUTE_FLOAT)
    def _(op, context, env, _fn=_fn, _sfn=_sfn):
        return _unary(op, context, _fn, _sfn)


# ---------------------------------------------------------------------------
# Float min/max  (Pattern A.3 — via _binop_via_op)
# ---------------------------------------------------------------------------

# TODO: consider deprecating arith.maxf / arith.minf aliases — these were
# renamed to arith.maximumf / arith.minimumf in upstream MLIR.
_FLOAT_MINMAX_OPS = {
    ("arith.maxf", "arith.maximumf"): ArithOps.maxf,
    ("arith.maxnumf",):               ArithOps.maxnumf,
    ("arith.minf", "arith.minimumf"): ArithOps.minf,
    ("arith.minnumf",):               ArithOps.minnumf,
}
for _names, _fn in _FLOAT_MINMAX_OPS.items():
    @register(*_names, latency_category=LC.COMPUTE_FLOAT)
    def _(op, context, env, _fn=_fn):
        return _binop_via_op(op, context, _fn)


# ---------------------------------------------------------------------------
# Integer binary ops  (Pattern A.2 — _int_binop table)
# ---------------------------------------------------------------------------

_INT_BINOPS = {
    "arith.addi":       operator.add,
    "arith.subi":       operator.sub,
    "arith.muli":       operator.mul,
    "arith.divui":      operator.floordiv,
    "arith.divsi":      _truncdiv,
    "arith.floordivsi": operator.floordiv,
    "arith.remui":      operator.mod,
    "arith.remsi":      _truncrem,
    "arith.andi":       operator.and_,
    "arith.ori":        operator.or_,
    "arith.xori":       operator.xor,
    "arith.shli":       operator.lshift,
    "arith.shrsi":      operator.rshift,
}
for _name, _fn in _INT_BINOPS.items():
    @register(_name, latency_category=LC.COMPUTE_INT)
    def _(op, context, env, _fn=_fn):
        return _int_binop(op, context, _fn)


# ---------------------------------------------------------------------------
# Integer binary ops via ops-layer  (Pattern A.3 — _binop_via_op table)
# ---------------------------------------------------------------------------

_INT_BINOPS_VIA_OP = {
    "arith.minsi":     ArithOps.minsi,
    "arith.maxsi":     ArithOps.maxsi,
    "arith.minui":     ArithOps.minui,
    "arith.maxui":     ArithOps.maxui,
    "arith.ceildivsi": ArithOps.ceildivsi,
    "arith.ceildivui": ArithOps.ceildivui,
    "arith.shrui":     ArithOps.shrui,
}
for _name, _fn in _INT_BINOPS_VIA_OP.items():
    @register(_name, latency_category=LC.COMPUTE_INT)
    def _(op, context, env, _fn=_fn):
        return _binop_via_op(op, context, _fn)


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

@register("arith.constant", no_lx_charge=True)
def arith__constant(op, context, env):
    value = op.attributes.get("value", 0)
    if op.attributes.get("is_tensor"):
        shape = op.attributes["shape"]
        dtype_str = op.attributes.get("dtype", "f16")
        np_dtype = to_np_dtype(dtype_str)
        # dense<[v0, v1, ...]> list form: each element is distinct, so
        # build the array element-by-element rather than splatting one value.
        if op.attributes.get("dense_list"):
            return Tile(np.array(value, dtype=np_dtype).reshape(shape), dtype_str, shape)
        return Tile(np.full(shape, value, dtype=np_dtype), dtype_str, shape)
    return value


# Cast ops — no latency category  (Pattern A.4 — cast cluster)
# extsi stays bespoke: uses a lambda that can't be expressed as an ArithOps method.

_CAST_UNARY_OPS = {
    "arith.extf":     (ArithOps.extf,     np.float32),
    "arith.truncf":   (ArithOps.truncf,   None),
    "arith.extui":    (ArithOps.extui,    int),
    "arith.trunci":   (ArithOps.trunci,   int),
    "arith.fptosi":   (ArithOps.fptosi,   int),
    "arith.fptoui":   (ArithOps.fptoui,   int),
    "arith.uitofp":   (ArithOps.uitofp,   float),
    "arith.convertf": (ArithOps.convertf, None),
}
for _name, (_fn, _sfn) in _CAST_UNARY_OPS.items():
    @register(_name)
    def _(op, context, env, _fn=_fn, _sfn=_sfn):
        return _unary(op, context, _fn, _sfn)


@register("arith.extsi")
def arith__extsi(op, context, env):
    return _unary(op, context, lambda t: Tile(t.data.astype(np.int64), "i64", t.shape), int)


@register("arith.sitofp")
def arith__sitofp(op, context, env):
    dtype = op.result_type or "f32"
    return _unary(op, context, lambda v: ArithOps.sitofp(v, dtype))


@register("arith.index_cast")
def arith__index_cast(op, context, env):
    return int(context.get_value(op.operands[0]))


@register("arith.index_castui")
def arith__index_castui(op, context, env):
    return int(context.get_value(op.operands[0]))


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
    from ..parser_utils import parse_numeric, parse_tensor_type, parse_dense_payload

    result_match = re.match(r'arith\.constant\s*(.*)', op_text)
    if not result_match:
        return None

    rest = result_match.group(1).strip()

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

    def _set_dense(payload, elem_dtype):
        # Handles both forms inside dense<...>: splat scalar and [v0, v1, ...] list.
        value, is_list = parse_dense_payload(payload, elem_dtype)
        attributes["value"] = value
        if is_list:
            attributes["dense_list"] = True

    def _set_tensor_attrs(type_info, type_str):
        # Used by Form 1 (braced) and Form 2 (dense) only — scalar Form 3
        # has no tensor result type so these attributes are not set there.
        if type_info and "tensor<" in type_str:
            attributes["shape"] = type_info["shape"]
            attributes["dtype"] = type_info.get("dtype", "f16")
            attributes["is_tensor"] = True

    braced_match = re.match(r'\{([^}]+)\}\s*:\s*(.+)$', rest)
    if braced_match:
        # Form 1: {inner} : result_type
        inner = braced_match.group(1).strip()
        result_type = braced_match.group(2).strip()

        _type_info = parse_tensor_type(result_type)
        elem_dtype = _type_info.get("dtype") if _type_info else result_type

        dense_match = re.match(r'dense<([^>]+)>', inner)
        if dense_match:
            _set_dense(dense_match.group(1), elem_dtype)
        else:
            typed_val = re.match(r'(.+?)\s*:\s*\S+', inner)
            raw = typed_val.group(1).strip() if typed_val else inner
            attributes["value"] = parse_numeric(raw, dtype=elem_dtype)

        _set_tensor_attrs(_type_info, result_type)
    else:
        # Form 2: dense<value> : type.  Covers:
        #   dense<0.0> : tensor<4xf16>       (splat tensor constant)
        #   dense<42> : tensor<1xi32>         (scalar tensor constant)
        #   dense<[16, 32]> : tensor<2xindex> (list — one value per element)
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
            _set_dense(dense_match.group(1), elem_dtype)
            _set_tensor_attrs(type_info, result_type)
        elif simple_match:
            result_type = simple_match.group(2).strip()
            attributes["value"] = parse_numeric(simple_match.group(1), dtype=result_type)
        else:
            # Defensive fallback: type-only with no parseable value — defaults to 0.
            # No known MLIR examples hit this path; kept for robustness.
            #   : tensor<1x64xf16>     (zero-initialized tensor)
            #   : index                (zero scalar)
            type_only_match = re.match(r':\s*(.+)$', rest)
            if type_only_match:
                result_type = type_only_match.group(1).strip()
                type_info = parse_tensor_type(result_type) if result_type and "tensor<" in result_type else None
                _set_tensor_attrs(type_info, result_type or "")
            attributes.setdefault("value", 0)

    return Operation(
        result=None,
        op_type="arith.constant",
        operands=[],
        attributes=attributes,
        result_type=result_type
    )


# ---------------------------------------------------------------------------
# Comparison parsers  (Pattern B — single parser for cmpi + cmpf)
# ---------------------------------------------------------------------------

_CMP_PREDICATES = {
    "arith.cmpi": (r"eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge", "unknown"),
    "arith.cmpf": (r"true|false|oeq|ogt|oge|olt|ole|one|ord|ueq|ugt|uge|ult|ule|une|uno", "i1"),
}


@register_parser("arith.cmpi", "arith.cmpf")
def parse_arith_cmp(op_text, parse_ctx):
    m = re.match(r'(arith\.cmp[if])\s+', op_text)
    if not m:
        return None
    op_type = m.group(1)
    pred_pattern, default_type = _CMP_PREDICATES[op_type]
    pred_match = re.search(rf'{re.escape(op_type)}\s+({pred_pattern})', op_text)
    if not pred_match:
        raise ValueError(f"{op_type}: no valid predicate found in: {op_text!r}")
    predicate = pred_match.group(1)
    operands = find_ssa_names(op_text)
    result_type = default_type
    type_match = re.search(r':\s*(.+)$', op_text)
    if type_match:
        result_type = type_match.group(1).strip()
    return Operation(
        result=None,
        op_type=op_type,
        operands=operands,
        attributes={"predicate": predicate},
        result_type=result_type,
    )


@register_parser("arith.sitofp")
def parse_arith_sitofp(op_text, parse_ctx):
    result_match = re.match(r'arith\.sitofp\s+(%\w+)', op_text)
    if not result_match:
        return None

    operand = result_match.group(1)

    result_type = "f16"
    to_match = re.search(r'to\s+(f\d+)', op_text)
    if to_match:
        result_type = to_match.group(1)

    return Operation(
        result=None,
        op_type="arith.sitofp",
        operands=[operand],
        attributes={},
        result_type=result_type
    )


@register_parser("arith.bitcast")
def parse_arith_bitcast(op_text, parse_ctx):
    result_match = re.match(r'arith\.bitcast\s+(%\w+)', op_text)
    if not result_match:
        return None

    operand = result_match.group(1)

    dst_type = "f32"
    to_match = re.search(r'\bto\s+(\S+)\s*$', op_text)
    if to_match:
        dst_type = to_match.group(1)

    return Operation(
        result=None,
        op_type="arith.bitcast",
        operands=[operand],
        attributes={"dst_type": dst_type},
        result_type=dst_type
    )
