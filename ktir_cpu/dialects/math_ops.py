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

"""Math dialect handlers — transcendental and element-wise math ops."""

from ..ir_types import Tile
from ..latency import LatencyCategory as LC
from ..ops.math_ops import MathOps
from ._helpers import _unary
from .registry import register


# ---------------------------------------------------------------------------
# Unary transcendental ops  (Pattern A.5)
# ---------------------------------------------------------------------------

_TRANSCENDENTAL_UNARY = {
    "math.exp":   MathOps.exp,
    "math.sqrt":  MathOps.sqrt,
    "math.rsqrt": MathOps.rsqrt,
    "math.log":   MathOps.log,
    "math.log2":  MathOps.log2,
    "math.log1p": MathOps.log1p,
    "math.tanh":  MathOps.tanh,
    "math.sin":   MathOps.sin,
    "math.cos":   MathOps.cos,
    "math.erf":   MathOps.erf,
}
for _name, _fn in _TRANSCENDENTAL_UNARY.items():
    @register(_name, latency_category=LC.COMPUTE_TRANSCENDENTAL)
    def _(op, context, env, _fn=_fn):
        return _unary(op, context, _fn)


# ---------------------------------------------------------------------------
# Unary float ops  (Pattern A.5)
# ---------------------------------------------------------------------------

_FLOAT_UNARY = {
    "math.absf":  MathOps.absf,
    "math.absi":  MathOps.absi,
    "math.ceil":  MathOps.ceil,
    "math.floor": MathOps.floor,
}
for _name, _fn in _FLOAT_UNARY.items():
    @register(_name, latency_category=LC.COMPUTE_FLOAT)
    def _(op, context, env, _fn=_fn):
        return _unary(op, context, _fn)


# ---------------------------------------------------------------------------
# Multi-operand ops (bespoke — don't fit _unary)
# ---------------------------------------------------------------------------

@register("math.powf", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__powf(op, context, env):
    base = context.get_value(op.operands[0])
    exponent = context.get_value(op.operands[1])
    if isinstance(base, Tile):
        return MathOps.powf(base, exponent)
    return MathOps.powf_scalar(base, exponent)


@register("math.fma", latency_category=LC.COMPUTE_FLOAT)
def math__fma(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    c = context.get_value(op.operands[2])
    if isinstance(a, Tile):
        return MathOps.fma(a, b, c)
    return MathOps.fma_scalar(a, b, c)
