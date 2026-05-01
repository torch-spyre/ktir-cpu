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

"""SCF / func dialect handlers — if, for, yield, return."""

import re

from ..ir_types import Operation
from ..latency import LatencyCategory as LC
from ..ops.comm_ops import CommOps
from ..ops.arith_ops import ArithOps
from ..ops.control_ops import ControlOps
from .registry import register, register_parser


@register("scf.if")
def scf__if(op, context, env):
    condition = context.get_value(op.operands[0])
    then_region = op.regions[0] if len(op.regions) > 0 else []
    else_region = op.regions[1] if len(op.regions) > 1 else []
    return ControlOps.if_op(context, condition, then_region, else_region, env.execute_region)


@register("scf.for")
def scf__for(op, context, env):
    lb = context.get_value(op.operands[0])
    ub = context.get_value(op.operands[1])
    step = context.get_value(op.operands[2])
    iter_var = op.attributes.get("iter_var", "%i")
    body_region = op.regions[0] if op.regions else []

    iter_arg_names = op.attributes.get("iter_args", [])
    iter_init_operands = op.operands[3:]
    iter_init_values = [context.get_value(n) for n in iter_init_operands]

    result = ControlOps.for_op(
        context, lb, ub, step, iter_var, body_region, env.execute_region,
        iter_arg_names=iter_arg_names,
        iter_init_values=iter_init_values,
    )
    # for_op returns a list of iter_arg final values; unwrap when there is
    # exactly one (the common case for a single result var).
    if isinstance(result, list):
        if len(result) == 1:
            result = result[0]
        else:
            assert len(result) == len(iter_arg_names), (
                f"scf.for: expected {len(iter_arg_names)} results, got {len(result)}"
            )
            result = tuple(result)
    return result


@register("scf.yield")
def scf__yield(op, context, env):
    values = [context.get_value(name) for name in op.operands]
    return ControlOps.yield_op(values)


@register("func.return")
def func__return(op, context, env):
    if op.operands:
        return context.get_value(op.operands[0])
    return None


# Also register the bare "return" alias
@register("return")
def _return_alias(op, context, env):
    return func__return(op, context, env)


# Communication operations (bundled here since they're few)
@register("ktdp.transfer", latency_category=LC.COMM)
def ktdp__transfer(op, context, env):
    tile = context.get_value(op.operands[0])
    dst_cores = context.get_value(op.operands[1])
    CommOps.transfer(context, tile, dst_cores, env.ring)
    return None


@register("ktdp.reduce", latency_category=LC.COMM)
def ktdp__reduce(op, context, env):
    tile = context.get_value(op.operands[0])
    core_group = context.get_value(op.operands[1])
    reduce_fn = lambda t1, t2: ArithOps.addf(t1, t2)
    return CommOps.reduce(context, tile, core_group, reduce_fn, env.ring)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@register("region.bb0_args")
def region__bb0_args(op, context, env):
    """No-op at execution time — bb0 args are bound by the enclosing op handler."""
    return None


@register_parser("^bb0")
def parse_bb0_block_args(op_text, parse_ctx):
    """Parse a ^bb0 block-argument label inside any region body.

    Syntax:
        ^bb0(%arg0: type, %arg1: type, ...):

    Emitted as a synthetic region.bb0_args op so that handlers
    (linalg.generic, tensor.generate, etc.) can bind block-argument
    names to their values.
    """
    paren_match = re.search(r'\(([^)]*)\)', op_text)
    arg_text = paren_match.group(1) if paren_match else op_text
    arg_names = re.findall(r'%\w+', arg_text)
    return Operation(
        result=None,
        op_type="region.bb0_args",
        operands=[],
        attributes={"names": arg_names},
        result_type=None,
    )


@register_parser("scf.for ")
def parse_scf_for(op_text, parse_ctx):
    # Detect optional outer result variable(s): %a, %b, %c = scf.for ...
    outer_result = None
    outer_match = re.match(r'((?:%\w+\s*,\s*)*%\w+)\s*=\s*scf\.for\s+', op_text)
    if outer_match:
        names = [n.strip() for n in outer_match.group(1).split(',')]
        outer_result = names if len(names) > 1 else names[0]

    scf_match = re.match(
        r'(?:(?:%\w+\s*,\s*)*%\w+\s*=\s*)?scf\.for\s+(%\w+)\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)',
        op_text
    )
    if not scf_match:
        return None

    iter_var = scf_match.group(1)
    lb = scf_match.group(2)
    ub = scf_match.group(3)
    step = scf_match.group(4)

    iter_args = []
    iter_inits = []
    ia_match = re.search(r'iter_args\(([^)]+)\)', op_text)
    if ia_match:
        for pair in re.finditer(r'(%\w+)\s*=\s*(%\w+)', ia_match.group(1)):
            iter_args.append(pair.group(1))
            iter_inits.append(pair.group(2))

    attributes = {"iter_var": iter_var}
    if iter_args:
        attributes["iter_args"] = iter_args

    # Use the outer result variable if present (e.g. %c = scf.for %off_k = ...);
    # otherwise fall back to the iter_var for loops without a result.
    result_name = outer_result if outer_result else iter_var

    return Operation(
        result=result_name,
        op_type="scf.for",
        operands=[lb, ub, step] + iter_inits,
        attributes=attributes,
        result_type="index"
    )


@register_parser("scf.yield", "= scf.yield")
def parse_scf_yield(op_text, parse_ctx):
    rest = op_text
    yield_match = re.match(r'scf\.yield\s*(.*)', op_text)
    if yield_match:
        rest = yield_match.group(1)
    operand_text = rest.split(':')[0] if ':' in rest else rest
    operands = re.findall(r'%\w+', operand_text)
    return Operation(
        result=None,
        op_type="scf.yield",
        operands=operands,
        attributes={},
        result_type=None,
    )
