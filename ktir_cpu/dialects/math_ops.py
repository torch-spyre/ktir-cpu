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
from .registry import register


@register("math.exp", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__exp(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.exp(operand)
    return MathOps.exp_scalar(operand)


@register("math.sqrt", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__sqrt(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.sqrt(operand)
    return MathOps.sqrt_scalar(operand)


@register("math.rsqrt", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__rsqrt(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.rsqrt(operand)
    return MathOps.rsqrt_scalar(operand)


@register("math.log", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.log(operand)
    return MathOps.log_scalar(operand)


@register("math.log2", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log2(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.log2(operand)
    return MathOps.log2_scalar(operand)


@register("math.log1p", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log1p(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.log1p(operand)
    return MathOps.log1p_scalar(operand)


@register("math.tanh", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__tanh(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.tanh(operand)
    return MathOps.tanh_scalar(operand)


@register("math.sin", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__sin(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.sin(operand)
    return MathOps.sin_scalar(operand)


@register("math.cos", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__cos(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.cos(operand)
    return MathOps.cos_scalar(operand)


@register("math.absf", latency_category=LC.COMPUTE_FLOAT)
def math__absf(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.abs(operand)
    return MathOps.abs_scalar(operand)


@register("math.ceil", latency_category=LC.COMPUTE_FLOAT)
def math__ceil(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.ceil(operand)
    return MathOps.ceil_scalar(operand)


@register("math.floor", latency_category=LC.COMPUTE_FLOAT)
def math__floor(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.floor(operand)
    return MathOps.floor_scalar(operand)


@register("math.erf", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__erf(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.erf(operand)
    return MathOps.erf_scalar(operand)


@register("math.powf", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__powf(op, context, env):
    base = context.get_value(op.operands[0])
    exponent = context.get_value(op.operands[1])
    return MathOps.powf(base, exponent)


@register("math.fma", latency_category=LC.COMPUTE_FLOAT)
def math__fma(op, context, env):
    a = context.get_value(op.operands[0])
    b = context.get_value(op.operands[1])
    c = context.get_value(op.operands[2])
    return MathOps.fma(a, b, c)
