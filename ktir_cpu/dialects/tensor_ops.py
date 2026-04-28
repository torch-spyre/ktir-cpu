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

"""Tensor dialect handlers — splat, expand_shape."""

import re
from typing import Optional, Tuple

import numpy as np

from ..grid import CoreContext
from ..dtypes import to_np_dtype
from ..ir_types import Operation, Tile
from .registry import register, register_parser


def _infer_splat_shape(context: CoreContext) -> Optional[Tuple[int, ...]]:
    """Find the shape of the largest Tile in the context.

    Heuristic for tensor.splat when the parser couldn't determine the
    target shape from the result type.
    """
    best_shape = None
    best_size = 0
    for scope in context._scope_stack:
        for name, val in scope.items():
            if isinstance(val, Tile):
                size = val.data.size
                if size > best_size:
                    best_size = size
                    best_shape = val.shape
    return best_shape


@register("tensor.empty")
def tensor__empty(op, context, env):
    """Create an uninitialized tensor of the given shape."""
    shape = op.attributes.get("shape", (1,))
    dtype_str = op.attributes.get("dtype", "f16")
    dtype = to_np_dtype(dtype_str)
    data = np.zeros(shape, dtype=dtype)
    return Tile(data, dtype_str, shape)


@register("tensor.splat")
def tensor__splat(op, context, env):
    scalar = context.get_value(op.operands[0])

    if isinstance(scalar, Tile):
        scalar = scalar.data.flat[0]

    shape = tuple(op.attributes.get("shape", ()))
    dtype_str = op.attributes.get("dtype", "f16")

    if not shape:
        rt = op.attributes.get("_result_shape")
        if rt:
            shape = rt
            dtype_str = op.attributes.get("_result_dtype", "f16")

    if not shape:
        shape = _infer_splat_shape(context)
        if not shape:
            shape = (1,)

    if isinstance(scalar, (np.integer, int)):
        np_dtype = np.int32
        dtype_str = "i32"
    else:
        np_dtype = to_np_dtype(dtype_str)

    data = np.full(shape, scalar, dtype=np_dtype)
    return Tile(data, dtype_str, shape)


@register("tensor.extract")
def tensor__extract(op, context, env):
    src = context.get_value(op.operands[0])
    indices = [context.get_value(o) for o in op.operands[1:]]

    if isinstance(src, Tile):
        if not indices:
            # 0D tensor — return the single element
            return src.data.flat[0]
        idx = tuple(int(i) for i in indices)
        return src.data[idx]

    # Already a scalar
    return src


@register("tensor.expand_shape")
def tensor__expand_shape(op, context, env):
    src = context.get_value(op.operands[0])
    target_shape = op.attributes.get("target_shape")
    if isinstance(src, Tile) and target_shape:
        reshaped = src.data.reshape(target_shape)
        return Tile(reshaped, src.dtype, target_shape)
    return src


@register("tensor.collapse_shape")
def tensor__collapse_shape(op, context, env):
    src = context.get_value(op.operands[0])
    target_shape = op.attributes.get("target_shape")
    if isinstance(src, Tile) and target_shape:
        reshaped = src.data.reshape(target_shape)
        return Tile(reshaped, src.dtype, target_shape)
    return src


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@register_parser("tensor.empty")
def parse_tensor_empty(op_text, parse_ctx):
    """Parse tensor.empty() : tensor<1x1024xf16>"""
    from ..parser_utils import parse_tensor_type
    result_match = re.match(r'(%\w+)\s*=\s*tensor\.empty\s*\(\s*\)\s*:\s*(.+)', op_text)
    if not result_match:
        return None
    result_name = result_match.group(1)
    type_str = result_match.group(2).strip()
    type_info = parse_tensor_type(type_str)
    attributes = {}
    if type_info:
        attributes["shape"] = type_info["shape"]
        attributes["dtype"] = type_info.get("dtype", "f16")
    return Operation(
        result=result_name,
        op_type="tensor.empty",
        operands=[],
        attributes=attributes,
        result_type=type_str,
    )


@register_parser("tensor.splat")
def parse_tensor_splat(op_text, parse_ctx):
    from ..parser_utils import parse_tensor_type
    result_match = re.match(r'(%\w+)\s*=\s*tensor\.splat\s+(%\w+)\s*(?::\s*(.+))?', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    scalar_operand = result_match.group(2)
    type_str = result_match.group(3).strip() if result_match.group(3) else "unknown"

    # When the syntax is `src_type -> dst_type`, parse the destination (result) type.
    # e.g. tensor<1x1xf16> -> tensor<1x1024xf16>  — we want the 1x1024 shape.
    if "->" in type_str:
        result_type = type_str.split("->", 1)[1].strip()
    else:
        result_type = type_str

    attributes = {}

    type_info = parse_tensor_type(result_type)
    if type_info:
        attributes["shape"] = type_info["shape"]
        attributes["dtype"] = type_info.get("dtype", "f16")

    return Operation(
        result=result_name,
        op_type="tensor.splat",
        operands=[scalar_operand],
        attributes=attributes,
        result_type=result_type
    )


@register_parser("tensor.extract")
def parse_tensor_extract(op_text, parse_ctx):
    # %scalar = tensor.extract %tensor[%i0, %i1] : tensor<...>
    result_match = re.match(r'(%\w+)\s*=\s*tensor\.extract\s+(%\w+)', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    src_operand = result_match.group(2)

    # Extract index operands from brackets: [%c0] or [%i, %j] or []
    indices = []
    bracket_match = re.search(r'\[([^\]]*)\]', op_text)
    if bracket_match:
        bracket_content = bracket_match.group(1).strip()
        if bracket_content:
            indices = re.findall(r'%\w+', bracket_content)

    return Operation(
        result=result_name,
        op_type="tensor.extract",
        operands=[src_operand] + indices,
        attributes={},
        result_type="scalar"
    )


def _parse_reshape_op(op_text, op_name):
    """Shared parser for tensor.expand_shape and tensor.collapse_shape."""
    result_match = re.match(
        r'(%\w+)\s*=\s*tensor\.' + op_name + r'\s+(%\w+)', op_text
    )
    if not result_match:
        return None

    result_name = result_match.group(1)
    operand = result_match.group(2)

    target_shape = None
    target_dtype = "f16"
    into_match = re.search(r'into\s+(?:tile|tensor)<([^>]+)>', op_text)
    if into_match:
        inner = into_match.group(1)
        parts = inner.split('x')
        shape_parts = []
        for p in parts[:-1]:
            try:
                shape_parts.append(int(p))
            except ValueError:
                pass
        if shape_parts:
            target_shape = tuple(shape_parts)
            target_dtype = parts[-1]

    attributes = {}
    if target_shape:
        attributes["target_shape"] = target_shape
        attributes["dtype"] = target_dtype

    return Operation(
        result=result_name,
        op_type=f"tensor.{op_name}",
        operands=[operand],
        attributes=attributes,
        result_type=f"tensor<{'x'.join(str(s) for s in target_shape)}x{target_dtype}>" if target_shape else "unknown"
    )


@register_parser("tensor.expand_shape")
def parse_tensor_expand_shape(op_text, parse_ctx):
    return _parse_reshape_op(op_text, "expand_shape")


@register_parser("tensor.collapse_shape")
def parse_tensor_collapse_shape(op_text, parse_ctx):
    return _parse_reshape_op(op_text, "collapse_shape")
