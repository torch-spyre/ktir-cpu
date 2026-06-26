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

"""func dialect handlers — func.return, func.call."""

import re

from ..ir_types import Operation
from .registry import register, register_parser


@register("func.return")
def func__return(op, context, env):
    if op.operands:
        return context.get_value(op.operands[0])
    return None


# Bare "return" is a common alias used in many KTIR examples.
@register("return")
def _return_alias(op, context, env):
    return func__return(op, context, env)


@register("func.call")
def func__call(op, context, env):
    """Execute a func.call by inlining the callee in a fresh scope.

    Pushes a new scope for the callee so callee-local SSA values don't
    collide with caller values.  LX state (ktdp.store writes) persists
    across func.call boundaries — only the per-stage temp watermark is
    rewound on scope pop, not the persistent LX region above it.

    Constraint: func.call must appear at the top level of a function
    body, not inside an scf.for or scf.if region.
    """
    callee_name = op.attributes["callee"]
    callee = env.module.get_function(callee_name)

    # Resolve caller operands before scope / use_counts swap so peeking uses
    # the caller's use_counts (no premature consumption of caller values).
    arg_values = [context.get_value(operand, peek=True) for operand in op.operands]

    # Push callee scope; swap use_counts so last-use tracking is correct inside.
    context.push_scope()
    caller_use_counts = context._use_counts
    context._use_counts = callee.use_counts

    # Bind callee args (charge=False — these are passed-in scalars, no LX).
    for param, val in zip(callee.arg_names, arg_values):
        context.set_value("%" + param, val, charge=False)

    # Execute callee body; propagate comm yields so the scheduler sees them.
    result = yield from env.execute_region_with_comms(context, callee.operations)

    # Pop scope (frees callee Tiles, rewinds lx.next_ptr) and restore use_counts.
    context.pop_scope()
    context._use_counts = caller_use_counts

    return result


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@register_parser("func.call")
def parse_func_call(op_text, parse_ctx):
    """Parse func.call — void and single-result forms.

    Syntax (void):   func.call @name(%a, %b) : (index, index) -> ()
    Syntax (value):  func.call @name(%a) : (index) -> index
    """
    m = re.match(r'func\.call\s+@(\w+)\s*\(([^)]*)\)', op_text)
    if not m:
        return None
    callee = m.group(1)
    args_text = m.group(2).strip()
    operands = re.findall(r'%\w+', args_text) if args_text else []
    return Operation(
        result=None,
        op_type="func.call",
        operands=operands,
        attributes={"callee": callee},
        result_type=None,
    )
