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
from ..parser_utils import find_ssa_names, parse_tensor_or_memref_type
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


@register("tensor.reshape")
def tensor__reshape(op, context, env):
    """Reinterpret a tensor with the same total element count under a new shape.

    The MLIR op carries two operands: the source tensor and a 1-D shape tensor
    (typically ``tensor<Nxindex>``) holding the target dimensions. The result
    type annotation always pins the target shape statically, so the executor
    reads ``target_shape`` from attributes (synthesized by the parser from the
    result type) rather than reading the runtime shape operand — matching how
    ``tensor.expand_shape`` / ``tensor.collapse_shape`` already behave.
    """
    src = context.get_value(op.operands[0])
    target_shape = op.attributes.get("target_shape")
    if target_shape is None:
        raise ValueError(
            f"tensor.reshape: missing 'target_shape' attribute on op {op}"
        )
    if not isinstance(src, Tile):
        return src
    reshaped = src.data.reshape(target_shape)
    return Tile(reshaped, src.dtype, target_shape)


@register("tensor.from_elements")
def tensor__from_elements(op, context, env):
    """Build a 1-D tensor from N scalar SSA operands.

    Used as the shape-operand producer for tensor.reshape. Each operand is
    fetched from the context, coerced to a Python scalar, and stacked into a
    NumPy array of the result type. Scalars from arith.constant arrive as
    plain ints/floats; those from another tensor (rare) arrive as Tiles —
    flatten the first element in that case.
    """
    shape_attr = op.attributes.get("shape")
    if shape_attr is None:
        raise ValueError(
            f"tensor.from_elements: missing 'shape' attribute on op {op}"
        )
    shape = tuple(shape_attr)
    dtype_str = op.attributes.get("dtype")
    if dtype_str is None:
        raise ValueError(
            f"tensor.from_elements: missing 'dtype' attribute on op {op}"
        )
    np_dtype = to_np_dtype(dtype_str)
    values = []
    for name in op.operands:
        v = context.get_value(name)
        if isinstance(v, Tile):
            v = v.data.flat[0]
        values.append(v)
    data = np.array(values, dtype=np_dtype).reshape(shape)
    return Tile(data, dtype_str, shape)


@register("tensor.yield")
def tensor__yield(op, context, env):
    """Yield a value from a tensor.generate body — same semantics as scf.yield."""
    from ..ops.control_ops import ControlOps
    values = [context.get_value(name) for name in op.operands]
    return ControlOps.yield_op(values)


_KDYNAMIC = -(1 << 63)  # ShapedType::kDynamic sentinel


def _resolve_dynamic(static_list, dynamic_operands, context):
    """Substitute kDynamic sentinels with values from dynamic_operands.

    static_list    — list of ints, some may be _KDYNAMIC
    dynamic_operands — SSA names for the dynamic positions, in order
    Returns a list of resolved ints.
    """
    dyn_iter = iter(dynamic_operands)
    result = []
    for v in static_list:
        if v == _KDYNAMIC:
            name = next(dyn_iter)
            result.append(int(context.get_value(name)))
        else:
            result.append(int(v))
    return result


@register("tensor.extract_slice")
def tensor__extract_slice(op, context, env):
    """Slice a sub-tensor and reshape to the result type's shape.

    Supports both fully-static and mixed static/dynamic offsets, sizes, and
    strides.  Dynamic positions carry the sentinel _KDYNAMIC in the static
    array; their real values are the trailing SSA operands after the source
    (offsets first, then sizes, then strides).
    """
    src = context.get_value(op.operands[0])
    if not isinstance(src, Tile):
        raise ValueError(f"tensor.extract_slice: expected Tile source, got {type(src)}")

    static_offsets = op.attributes["static_offsets"]
    static_sizes = op.attributes["static_sizes"]
    static_strides = op.attributes["static_strides"]
    result_shape = tuple(op.attributes["result_shape"])
    dtype_str = op.attributes["dtype"]

    # Dynamic operands follow the source operand, ordered: offsets, sizes, strides.
    n_dyn_off = sum(1 for v in static_offsets if v == _KDYNAMIC)
    n_dyn_sz  = sum(1 for v in static_sizes   if v == _KDYNAMIC)
    n_dyn_st  = sum(1 for v in static_strides  if v == _KDYNAMIC)
    dyn_ops = op.operands[1:]
    dyn_off_ops = dyn_ops[:n_dyn_off]
    dyn_sz_ops  = dyn_ops[n_dyn_off:n_dyn_off + n_dyn_sz]
    dyn_st_ops  = dyn_ops[n_dyn_off + n_dyn_sz:n_dyn_off + n_dyn_sz + n_dyn_st]

    offsets = _resolve_dynamic(static_offsets, dyn_off_ops, context)
    sizes   = _resolve_dynamic(static_sizes,   dyn_sz_ops,  context)
    strides = _resolve_dynamic(static_strides, dyn_st_ops,  context)

    idx = tuple(
        slice(off, off + sz * st, st)
        for off, sz, st in zip(offsets, sizes, strides)
    )
    sliced = src.data[idx]
    reshaped = sliced.reshape(result_shape)
    return Tile(reshaped, dtype_str, result_shape)


@register("tensor.insert_slice")
def tensor__insert_slice(op, context, env):
    """Insert a sub-tensor into a destination tensor and return the result.

    Supports both fully-static and mixed static/dynamic offsets, sizes, and
    strides.  Dynamic positions carry the sentinel _KDYNAMIC in the static
    array; their real values are the trailing SSA operands after source and
    dest (offsets first, then sizes, then strides).
    """
    src = context.get_value(op.operands[0])
    dst = context.get_value(op.operands[1])

    if not isinstance(src, Tile):
        raise ValueError(f"tensor.insert_slice: expected Tile source, got {type(src)}")
    if not isinstance(dst, Tile):
        raise ValueError(f"tensor.insert_slice: expected Tile dest, got {type(dst)}")

    static_offsets = op.attributes["static_offsets"]
    static_sizes = op.attributes["static_sizes"]
    static_strides = op.attributes["static_strides"]
    result_shape = tuple(op.attributes["result_shape"])
    dtype_str = op.attributes["dtype"]

    n_dyn_off = sum(1 for v in static_offsets if v == _KDYNAMIC)
    n_dyn_sz  = sum(1 for v in static_sizes   if v == _KDYNAMIC)
    n_dyn_st  = sum(1 for v in static_strides  if v == _KDYNAMIC)
    dyn_ops = op.operands[2:]
    dyn_off_ops = dyn_ops[:n_dyn_off]
    dyn_sz_ops  = dyn_ops[n_dyn_off:n_dyn_off + n_dyn_sz]
    dyn_st_ops  = dyn_ops[n_dyn_off + n_dyn_sz:n_dyn_off + n_dyn_sz + n_dyn_st]

    offsets = _resolve_dynamic(static_offsets, dyn_off_ops, context)
    sizes   = _resolve_dynamic(static_sizes,   dyn_sz_ops,  context)
    strides = _resolve_dynamic(static_strides, dyn_st_ops,  context)

    idx = tuple(
        slice(off, off + sz * st, st)
        for off, sz, st in zip(offsets, sizes, strides)
    )
    result_data = dst.data.copy()
    result_data[idx] = src.data.reshape([int(sz) for sz in sizes])
    return Tile(result_data, dtype_str, result_shape)


@register("tensor.generate")
def tensor__generate(op, context, env):
    """Generate a tensor by evaluating a region body at each index.

    MLIR syntax:
        %mask = tensor.generate {
        ^bb0(%i: index, %j: index):
          %cmp = arith.cmpi sge, %i, %j : index
          %val = arith.select %cmp, %zero, %neg_inf : f16
          tensor.yield %val : f16
        } : tensor<16x16xf16>

    The body receives one block argument per dimension (the indices),
    computes a scalar, and yields it via tensor.yield.  This handler
    iterates over all index combinations, executes the body each time,
    and assembles the results into a Tile.

    Used by prefill_attention to build the causal mask on-chip:
        mask[i, j] = 0.0 if i >= j else -10000.0
    """
    shape = tuple(op.attributes.get("shape", ()))
    dtype_str = op.attributes.get("dtype", "f16")
    np_dtype = to_np_dtype(dtype_str)
    region = op.regions[0] if op.regions else []

    # Extract block arg names from the ^bb0 label (parsed as region.bb0_args)
    bb0_op = next((o for o in region if o.op_type == "region.bb0_args"), None)
    block_args = bb0_op.attributes["names"] if bb0_op else []
    body = [o for o in region if o.op_type != "region.bb0_args"]

    # Vectorized execution: instead of looping per-element, we pass full
    # index grids as Tiles and execute the body once.  This works because:
    #   - block_args (e.g. ["%i", "%j"]) are always index-typed (MLIR spec)
    #   - arith/compute ops already handle Tile inputs element-wise
    # The body therefore computes the entire output tensor in one pass.
    #
    # Example for shape=(3,3): grids gives two 3x3 arrays:
    #   %i -> [[0,0,0],    %j -> [[0,1,2],
    #          [1,1,1],           [0,1,2],
    #          [2,2,2]]           [0,1,2]]
    # Then `arith.cmpi sge, %i, %j` compares element-wise → lower-triangular:
    #   [[1,0,0],
    #    [1,1,0],
    #    [1,1,1]]
    grids = np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')
    context.push_scope()
    for arg_name, grid in zip(block_args, grids):
        context.set_value(arg_name, Tile(grid, "index", shape))
    result = env.execute_region(context, body)
    if hasattr(result, 'values'):
        result = result.values[0]
    context.pop_scope()

    if isinstance(result, Tile):
        data = result.data.astype(np_dtype)
    else:
        data = np.full(shape, result, dtype=np_dtype)
    return Tile(data, dtype_str, shape)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@register_parser("tensor.empty")
def parse_tensor_empty(op_text, parse_ctx):
    """Parse tensor.empty() : tensor<1x1024xf16>"""
    from ..parser_utils import parse_tensor_or_memref_type
    result_match = re.match(r'tensor\.empty\s*\(\s*\)\s*:\s*(.+)', op_text)
    if not result_match:
        return None
    type_str = result_match.group(1).strip()
    type_info = parse_tensor_or_memref_type(type_str)
    attributes = {}
    if type_info:
        attributes["shape"] = type_info["shape"]
        attributes["dtype"] = type_info.get("dtype", "f16")
    return Operation(
        result=None,
        op_type="tensor.empty",
        operands=[],
        attributes=attributes,
        result_type=type_str,
    )


@register_parser("tensor.splat")
def parse_tensor_splat(op_text, parse_ctx):
    from ..parser_utils import parse_tensor_or_memref_type
    result_match = re.match(r'tensor\.splat\s+(%\w+)\s*(?::\s*(.+))?', op_text)
    if not result_match:
        return None

    scalar_operand = result_match.group(1)
    type_str = result_match.group(2).strip() if result_match.group(2) else "unknown"

    # When the syntax is `src_type -> dst_type`, parse the destination (result) type.
    # e.g. tensor<1x1xf16> -> tensor<1x1024xf16>  — we want the 1x1024 shape.
    if "->" in type_str:
        result_type = type_str.split("->", 1)[1].strip()
    else:
        result_type = type_str

    attributes = {}

    type_info = parse_tensor_or_memref_type(result_type)
    if type_info:
        attributes["shape"] = type_info["shape"]
        attributes["dtype"] = type_info.get("dtype", "f16")

    return Operation(
        result=None,
        op_type="tensor.splat",
        operands=[scalar_operand],
        attributes=attributes,
        result_type=result_type
    )


@register_parser("tensor.extract ")
def parse_tensor_extract(op_text, parse_ctx):
    # %scalar = tensor.extract %tensor[%i0, %i1] : tensor<...>
    result_match = re.match(r'tensor\.extract\s+(%\w+)', op_text)
    if not result_match:
        return None

    src_operand = result_match.group(1)

    # Extract index operands from brackets: [%c0] or [%i, %j] or []
    indices = []
    bracket_match = re.search(r'\[([^\]]*)\]', op_text)
    if bracket_match:
        bracket_content = bracket_match.group(1).strip()
        if bracket_content:
            indices = find_ssa_names(bracket_content)

    return Operation(
        result=None,
        op_type="tensor.extract",
        operands=[src_operand] + indices,
        attributes={},
        result_type="scalar"
    )


def _parse_reshape_op(op_text, op_name):
    """Shared parser for tensor.expand_shape and tensor.collapse_shape."""
    result_match = re.match(
        r'tensor\.' + op_name + r'\s+(%\w+)', op_text
    )
    if not result_match:
        return None

    operand = result_match.group(1)

    target_shape = None
    target_dtype = "f16"
    into_match = re.search(r'into\s+(?:tile|tensor)<([^>]+)>', op_text)
    if into_match:
        info = parse_tensor_or_memref_type(into_match.group(1))
        if info:
            target_shape = tuple(d for d in info["shape"] if d is not None)
            target_dtype = info["dtype"]

    attributes = {}
    if target_shape:
        attributes["target_shape"] = target_shape
        attributes["dtype"] = target_dtype

    return Operation(
        result=None,
        op_type=f"tensor.{op_name}",
        operands=[operand],
        attributes=attributes,
        result_type=f"tensor<{'x'.join(str(s) for s in target_shape)}x{target_dtype}>" if target_shape else "unknown"
    )


@register_parser("tensor.yield")
def parse_tensor_yield(op_text, parse_ctx):
    """Parse tensor.yield — terminates a tensor.generate body.

    Syntax:
        tensor.yield %val : f16

    Extracts the yielded operand (%val) from the text before the `:` type
    annotation.  Mirrors the scf.yield parser structure.
    """
    # Strip the op name prefix to get "%val : f16"
    rest = op_text
    yield_match = re.match(r'tensor\.yield\s*(.*)', op_text)
    if yield_match:
        rest = yield_match.group(1)
    # Operands are before the `:` type annotation
    operand_text = rest.split(':')[0] if ':' in rest else rest
    operands = find_ssa_names(operand_text)
    return Operation(
        result=None,
        op_type="tensor.yield",
        operands=operands,
        attributes={},
        result_type=None,
    )


@register_parser("tensor.generate")
def parse_tensor_generate(op_text, parse_ctx):
    """Parse tensor.generate.

    Full syntax:
        %mask = tensor.generate {
        ^bb0(%i: index, %j: index):
          %cmp = arith.cmpi sge, %i, %j : index
          %val = arith.select %cmp, %zero, %neg_inf : f16
          tensor.yield %val : f16
        } : tensor<16x16xf16>

    The parser only handles the outer op line (after region extraction):
        %mask = tensor.generate : tensor<16x16xf16>

    The region body (^bb0 + ops + tensor.yield) is automatically extracted
    by _tokenize_operations and attached as op.regions[0].  The ^bb0 line
    becomes a synthetic region.bb0_args op containing the block arg names.

    We extract:
      - result_name: %mask
      - shape/dtype from the trailing `: tensor<16x16xf16>` type annotation
    """
    from ..parser_utils import parse_tensor_or_memref_type

    # Match: tensor.generate ...
    result_match = re.match(r'tensor\.generate', op_text)
    if not result_match:
        return None

    attributes = {}

    # Extract shape and dtype from trailing `: tensor<16x16xf16>`
    # (required — without it we don't know what size tensor to generate)
    type_match = re.search(r':\s*(tensor<[^>]+>)\s*$', op_text)
    if not type_match:
        raise ValueError(f"tensor.generate: missing result type in '{op_text}'")
    type_info = parse_tensor_or_memref_type(type_match.group(1))
    if type_info:
        attributes["shape"] = type_info["shape"]
        attributes["dtype"] = type_info.get("dtype", "f16")

    return Operation(
        result=None,
        op_type="tensor.generate",
        operands=[],
        attributes=attributes,
        result_type=type_match.group(1),
    )


@register_parser("tensor.expand_shape")
def parse_tensor_expand_shape(op_text, parse_ctx):
    return _parse_reshape_op(op_text, "expand_shape")


@register_parser("tensor.collapse_shape")
def parse_tensor_collapse_shape(op_text, parse_ctx):
    return _parse_reshape_op(op_text, "collapse_shape")


@register_parser("tensor.reshape")
def parse_tensor_reshape(op_text, parse_ctx):
    """Parse `%out = tensor.reshape %src(%shape) : (...) -> tensor<...>`.

    Two SSA operands: source tensor and shape tensor. Target shape is read
    from the trailing result-type annotation (after `->`), since that is
    always statically pinned, while the shape operand may be runtime.
    """
    from ..parser_utils import parse_tensor_or_memref_type
    m = re.match(
        r'tensor\.reshape\s+(%\w+)\s*\(\s*(%\w+)\s*\)', op_text
    )
    if not m:
        return None
    src, shape_operand = m.group(1), m.group(2)

    arrow = re.search(r'->\s*(tensor<[^>]+>)', op_text)
    if not arrow:
        raise ValueError(f"tensor.reshape: missing result type in '{op_text}'")
    result_type = arrow.group(1)
    info = parse_tensor_or_memref_type(result_type)
    if info is None:
        raise ValueError(
            f"tensor.reshape: cannot parse result type {result_type!r} in '{op_text}'"
        )

    return Operation(
        result=None,
        op_type="tensor.reshape",
        operands=[src, shape_operand],
        attributes={
            "target_shape": info["shape"],
            "dtype": info["dtype"],
        },
        result_type=result_type,
    )


@register_parser("tensor.from_elements")
def parse_tensor_from_elements(op_text, parse_ctx):
    """Parse `%shape = tensor.from_elements %d0, %d1, ... : tensor<NxT>`."""
    from ..parser_utils import parse_tensor_or_memref_type
    m = re.match(
        r'tensor\.from_elements\s+(.*?)\s*:\s*(tensor<[^>]+>)', op_text
    )
    if not m:
        return None
    operand_text = m.group(1)
    type_str = m.group(2)
    operands = find_ssa_names(operand_text)

    info = parse_tensor_or_memref_type(type_str)
    if info is None:
        raise ValueError(
            f"tensor.from_elements: cannot parse result type {type_str!r} in '{op_text}'"
        )

    return Operation(
        result=None,
        op_type="tensor.from_elements",
        operands=operands,
        attributes={
            "shape": info["shape"],
            "dtype": info["dtype"],
        },
        result_type=type_str,
    )


def _parse_index_list(content):
    """Parse a bracket group's content into (static_list, dynamic_names).

    Each comma-separated entry is either an integer (static) or an SSA name
    like ``%off`` (dynamic).  Dynamic entries are stored as _KDYNAMIC in the
    static list; their SSA names are collected in order into dynamic_names.
    """
    static = []
    dynamic_names = []
    for token in content.split(','):
        token = token.strip()
        if not token:
            continue
        if token.startswith('%'):
            static.append(_KDYNAMIC)
            dynamic_names.append(token)
        else:
            static.append(int(token))
    return static, dynamic_names


def _parse_index_bracket_groups(op_text):
    """Return three (static_list, dynamic_names) tuples for [offsets][sizes][strides]."""
    groups = []
    for m in re.finditer(r'\[([^\]]*)\]', op_text):
        groups.append(_parse_index_list(m.group(1)))
    if len(groups) < 3:
        raise ValueError(f"expected [offsets][sizes][strides] in {op_text!r}")
    return groups[0], groups[1], groups[2]


@register_parser("tensor.extract_slice")
def parse_tensor_extract_slice(op_text, parse_ctx):
    """Parse `%r = tensor.extract_slice %src[offsets][sizes][strides] : T to T`.

    Supports mixed static/dynamic entries: ``%off`` tokens become _KDYNAMIC
    in the static array and are appended to operands after the source.
    """
    from ..parser_utils import parse_tensor_or_memref_type
    m = re.match(r'tensor\.extract_slice\s+(%\w+)', op_text)
    if not m:
        return None
    src_operand = m.group(1)

    (off_s, off_d), (sz_s, sz_d), (st_s, st_d) = _parse_index_bracket_groups(op_text)

    result_type_m = re.search(r'\bto\s+(tensor<[^>]+>)\s*$', op_text)
    if not result_type_m:
        raise ValueError(f"tensor.extract_slice: missing result type in {op_text!r}")
    result_type = result_type_m.group(1)
    info = parse_tensor_or_memref_type(result_type)
    if info is None:
        raise ValueError(f"tensor.extract_slice: cannot parse result type {result_type!r}")

    return Operation(
        result=None,
        op_type="tensor.extract_slice",
        operands=[src_operand] + off_d + sz_d + st_d,
        attributes={
            "static_offsets": off_s,
            "static_sizes": sz_s,
            "static_strides": st_s,
            "result_shape": info["shape"],
            "dtype": info["dtype"],
        },
        result_type=result_type,
    )


@register_parser("tensor.insert_slice")
def parse_tensor_insert_slice(op_text, parse_ctx):
    """Parse `%r = tensor.insert_slice %src into %dst[offsets][sizes][strides] : T into T`.

    Supports mixed static/dynamic entries: ``%off`` tokens become _KDYNAMIC
    in the static array and are appended to operands after source and dest.
    """
    from ..parser_utils import parse_tensor_or_memref_type
    m = re.match(
        r'tensor\.insert_slice\s+(%\w+)\s+into\s+(%\w+)', op_text
    )
    if not m:
        return None
    src_operand = m.group(1)
    dst_operand = m.group(2)

    (off_s, off_d), (sz_s, sz_d), (st_s, st_d) = _parse_index_bracket_groups(op_text)

    result_type_m = re.search(r'\binto\s+(tensor<[^>]+>)\s*$', op_text)
    if not result_type_m:
        raise ValueError(f"tensor.insert_slice: missing result type in {op_text!r}")
    result_type = result_type_m.group(1)
    info = parse_tensor_or_memref_type(result_type)
    if info is None:
        raise ValueError(f"tensor.insert_slice: cannot parse result type {result_type!r}")

    return Operation(
        result=None,
        op_type="tensor.insert_slice",
        operands=[src_operand, dst_operand] + off_d + sz_d + st_d,
        attributes={
            "static_offsets": off_s,
            "static_sizes": sz_s,
            "static_strides": st_s,
            "result_shape": info["shape"],
            "dtype": info["dtype"],
        },
        result_type=result_type,
    )
