# Copyright 2026 The Torch-Spyre Authors.
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

"""Spyreop dialect handlers — Spyre hardware intrinsics.

Executes the ``spyreop.*`` ops that Triton's ``tl.spyre_op``/
``tts.spyre_op`` lowers directly into a ``linalg.generic`` body (see
torch-spyre/triton's ``spyreop-frontend-api.md``). This repo's parser
never sees the custom ``!spyreop.df16``/fused types or the
reduction/select/compare/slice_reduction ops the upstream SpyreOp dialect
also defines — only the 12 scalar-typed ops Triton actually emits, each
with plain ``f16``/``f32``/``i32``/``i64`` operands, same as every other
``linalg.generic`` body this interpreter already runs.

Most of these ops parse through the regex path's general-purpose
fallback with no dedicated ``register_parser`` at all — the same reason
``math.sqrt``/``math.exp`` need none (see ``math_ops.py``): their
assembly format is `` %operand(s)[ : type]`` and the fallback's SSA-name
scan already finds every ``%``-operand in call order regardless of
interleaved bare keywords like ``squares``/``scale``/``weight``/``bias``
(``layernormscale``, ``layernormnorm``) or ``base``/``stride``
(``idx32toaddr``) sitting between them. Only ``softplus`` needs one,
because its ``beta``/``threshold`` are non-SSA numeric literals the
fallback has no attribute syntax for (see ``register_parser("spyreop.softplus")``
below).
"""

from ..ir_types import Operation, Tile
from ..latency import LatencyCategory as LC
from ..ops.spyreop_ops import SpyreOps
from ._helpers import _unary
from .registry import register, register_parser


# ---------------------------------------------------------------------------
# Pure unary activations — `` %operand : type``, no attributes.
# Parses via the general-purpose fallback; no register_parser needed.
# ---------------------------------------------------------------------------

_UNARY = {
    "spyreop.gelu": SpyreOps.gelu,
    "spyreop.silu": SpyreOps.silu,
    "spyreop.reciprocal": SpyreOps.reciprocal,
}
for _name, _fn in _UNARY.items():
    @register(_name, latency_category=LC.COMPUTE_TRANSCENDENTAL)
    def _(op, context, env, _fn=_fn):
        return _unary(op, context, _fn)


# ---------------------------------------------------------------------------
# Integer address arithmetic — `` %lhs, %rhs``, no result-type suffix.
# Parses via the fallback (SSA-name scan finds both operands in order).
#
# addi32toi32/addi64toi64/muli32toi32 are COMPUTE_TRANSCENDENTAL, not
# COMPUTE_INT: Spyre has no native 32/64-bit integer ALU, so these decompose
# into floating-point operations and reconstruct the integer result — the
# same complex-instruction cost class as gelu/softplus/etc. idx32toaddr
# below stays COMPUTE_INT (plain index/offset arithmetic).
# ---------------------------------------------------------------------------

@register("spyreop.addi32toi32", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def spyreop__addi32toi32(op, context, env):
    lhs = context.get_value(op.operands[0])
    rhs = context.get_value(op.operands[1])
    return SpyreOps.addi32toi32(lhs, rhs)


@register("spyreop.addi64toi64", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def spyreop__addi64toi64(op, context, env):
    lhs = context.get_value(op.operands[0])
    rhs = context.get_value(op.operands[1])
    return SpyreOps.addi64toi64(lhs, rhs)


@register("spyreop.muli32toi32", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def spyreop__muli32toi32(op, context, env):
    lhs = context.get_value(op.operands[0])
    rhs = context.get_value(op.operands[1])
    return SpyreOps.muli32toi32(lhs, rhs)


@register("spyreop.idx32toaddr", latency_category=LC.COMPUTE_INT)
def spyreop__idx32toaddr(op, context, env):
    """``%index base %base stride %stride``.

    ``base``/``stride`` are bare keywords in the assembly format, not
    attributes — the fallback parser's SSA-name scan still finds all
    three ``%``-operands in call order (``index``, ``base``, ``stride``),
    matching ``SpyreOp.td``'s own argument order.
    """
    index = context.get_value(op.operands[0])
    base = context.get_value(op.operands[1])
    stride = context.get_value(op.operands[2])
    return SpyreOps.idx32toaddr(index, base, stride)


# ---------------------------------------------------------------------------
# Softplus — the one op whose `` beta B threshold T`` keyword/literal
# pairs the fallback parser cannot capture, so it needs its own parser.
# ---------------------------------------------------------------------------

@register("spyreop.softplus", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def spyreop__softplus(op, context, env):
    val = context.get_value(op.operands[0])
    beta = op.attributes["beta"]
    threshold = op.attributes["threshold"]
    return SpyreOps.softplus(val, beta, threshold)


@register_parser("spyreop.softplus")
def parse_spyreop_softplus(op_text, parse_ctx):
    """``spyreop.softplus %x beta B threshold T [attr-dict] : type``.

    ``beta``/``threshold`` are plain float literals in the assembly
    format (``SpyreOp.td``: ``$operand `beta` $beta `threshold` $threshold
    attr-dict `:` type($result)``), not SSA operands — the only reason
    this op needs a dedicated parser among the 12 in scope.
    """
    import re
    from ..parser_utils import find_ssa_names, parse_numeric, parse_tensor_or_memref_type

    if not op_text.startswith("spyreop.softplus"):
        return None

    operands = find_ssa_names(op_text[len("spyreop.softplus"):].split("beta")[0])
    beta_match = re.search(r'\bbeta\s+([^\s]+)', op_text)
    threshold_match = re.search(r'\bthreshold\s+([^\s]+)', op_text)
    if not operands or not beta_match or not threshold_match:
        raise ValueError(f"spyreop.softplus: malformed operation text: {op_text!r}")

    result_type = None
    type_match = re.search(r':\s*(\S+)\s*$', op_text)
    if type_match:
        result_type = type_match.group(1).strip()

    attributes = {
        "beta": parse_numeric(beta_match.group(1), dtype="f32"),
        "threshold": parse_numeric(threshold_match.group(1), dtype="f32"),
    }
    if result_type:
        type_info = parse_tensor_or_memref_type(result_type)
        if type_info:
            attributes["_result_shape"] = type_info["shape"]
            attributes["_result_dtype"] = type_info.get("dtype", "f16")

    return Operation(
        result=None,
        op_type="spyreop.softplus",
        operands=operands,
        attributes=attributes,
        result_type=result_type,
    )


# ---------------------------------------------------------------------------
# LayerNorm's three stages. layernormscale/layernormnorm's keyword-
# interleaved operands (``squares``/``scale``/``weight``/``bias``) parse
# via the fallback like idx32toaddr's do; exx2 needs its axis, which the
# surrounding linalg.generic (Group D2's reduction-shaped body) carries
# in context, not in the op's own text.
# ---------------------------------------------------------------------------

@register("spyreop.layernormscale", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def spyreop__layernormscale(op, context, env):
    mean = context.get_value(op.operands[0])
    msq = context.get_value(op.operands[1])
    return SpyreOps.layernormscale(mean, msq)


@register("spyreop.layernormnorm", latency_category=LC.COMPUTE_FLOAT)
def spyreop__layernormnorm(op, context, env):
    x = context.get_value(op.operands[0])
    mean = context.get_value(op.operands[1])
    scale = context.get_value(op.operands[2])
    weight = context.get_value(op.operands[3])
    bias = context.get_value(op.operands[4])
    return SpyreOps.layernormnorm(x, mean, scale, weight, bias)


@register("spyreop.exx2", latency_category=LC.COMPUTE_FLOAT)
def spyreop__exx2(op, context, env):
    """``%mean, %squares = spyreop.exx2 %in : f16``.

    Reduction-shaped: called by ``linalg.generic``'s whole-body reduction
    path (``_whole_body_reduce`` in ``dialects/linalg_ops.py``) over the
    full, un-reduced ``ins`` operand, with the axis to reduce carried in
    ``context["__linalg_reduction_dims__"]`` — the same convention
    ``linalg.index`` uses for ``__linalg_shape__``, since this op's own
    text names no axis at all (Triton bakes it into the surrounding
    generic's ``iterator_types``, not into ``spyreop.exx2`` itself).
    """
    val = context.get_value(op.operands[0])
    reduction_dims = context.get_value("__linalg_reduction_dims__")
    if len(reduction_dims) != 1:
        raise ValueError(
            f"spyreop.exx2: expected exactly one reduction dim, got {reduction_dims}"
        )
    return SpyreOps.exx2(val, reduction_dims[0])
