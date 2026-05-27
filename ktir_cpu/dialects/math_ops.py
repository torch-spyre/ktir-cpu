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


@register("math.exp", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__exp(op, context, env):
    return _unary(op, context, MathOps.exp, MathOps.exp_scalar)


@register("math.sqrt", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__sqrt(op, context, env):
    return _unary(op, context, MathOps.sqrt, MathOps.sqrt_scalar)


@register("math.rsqrt", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__rsqrt(op, context, env):
    return _unary(op, context, MathOps.rsqrt, MathOps.rsqrt_scalar)


@register("math.log", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log(op, context, env):
    return _unary(op, context, MathOps.log, MathOps.log_scalar)


@register("math.log2", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log2(op, context, env):
    return _unary(op, context, MathOps.log2, MathOps.log2_scalar)


@register("math.log1p", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log1p(op, context, env):
    return _unary(op, context, MathOps.log1p, MathOps.log1p_scalar)


@register("math.tanh", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__tanh(op, context, env):
    return _unary(op, context, MathOps.tanh, MathOps.tanh_scalar)


@register("math.sin", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__sin(op, context, env):
    return _unary(op, context, MathOps.sin, MathOps.sin_scalar)


@register("math.cos", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__cos(op, context, env):
    return _unary(op, context, MathOps.cos, MathOps.cos_scalar)


@register("math.absf", latency_category=LC.COMPUTE_FLOAT)
def math__absf(op, context, env):
    return _unary(op, context, MathOps.absf, MathOps.absf_scalar)


@register("math.absi", latency_category=LC.COMPUTE_FLOAT)
def math__absi(op, context, env):
    return _unary(op, context, MathOps.absi, MathOps.absi_scalar)


@register("math.ceil", latency_category=LC.COMPUTE_FLOAT)
def math__ceil(op, context, env):
    return _unary(op, context, MathOps.ceil, MathOps.ceil_scalar)


@register("math.floor", latency_category=LC.COMPUTE_FLOAT)
def math__floor(op, context, env):
    return _unary(op, context, MathOps.floor, MathOps.floor_scalar)


@register("math.erf", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__erf(op, context, env):
    return _unary(op, context, MathOps.erf, MathOps.erf_scalar)


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
