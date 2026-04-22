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

"""Math dialect handlers — exp, sqrt, log."""

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


@register("math.log", latency_category=LC.COMPUTE_TRANSCENDENTAL)
def math__log(op, context, env):
    operand = context.get_value(op.operands[0])
    if isinstance(operand, Tile):
        return MathOps.log(operand)
    return MathOps.log_scalar(operand)
