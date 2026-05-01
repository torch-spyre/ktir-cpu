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

"""Linalg dialect handlers — reduce, matmul, generic."""

import re

import numpy as np

from ..ir_types import Operation, Tile
from ..latency import LatencyCategory as LC
from ..ops.arith_ops import ArithOps
from ..parser_ast import parse_affine_map
from ..parser_utils import parse_attr_list
from .registry import register, register_parser


@register("linalg.reduce", latency_category=LC.COMPUTE_FLOAT)
def linalg__reduce(op, context, env):
    """Standard linalg.reduce — removes the reduced dimension."""
    # operands[0] is the ins tensor; the outs tensor is handled separately via outs_var
    tile = context.get_value(op.operands[0])
    # Resolve the combiner op name.  The shorthand form stores it in
    # attributes; the explicit-region form has it in op.regions.
    reduce_fn = op.attributes.get("reduce_fn")
    if reduce_fn is None and op.regions:
        for region_op in op.regions[0]:
            if region_op.op_type != "linalg.yield":
                reduce_fn = region_op.op_type
                break
    if reduce_fn is None:
        reduce_fn = "arith.addf"
    # axis to reduce along; None means reduce all elements to a scalar
    dim = op.attributes.get("dim")

    # Map MLIR combiner names to NumPy reduction functions
    np_reduce = {
        "arith.addf": np.sum,
        "arith.maxf": np.max,
        "arith.maxnumf": np.max,
        "arith.maximumf": np.maximum.reduce,
        "arith.minf": np.min,
        "arith.minimumf": np.minimum.reduce,
        "arith.minnumf": np.min,
        "arith.mulf": np.prod,
    }.get(reduce_fn)

    if np_reduce is None:
        raise ValueError(f"Unknown linalg.reduce combiner: {reduce_fn}")

    if isinstance(tile, Tile):
        if dim is not None:
            # Reduce along the specified axis; promote to f32 to avoid overflow
            reduced = np_reduce(tile.data.astype(np.float32), axis=dim, keepdims=False)
            # Cast back to the original element dtype
            reduced = reduced.astype(tile.data.dtype)
            if reduced.ndim == 0:
                # Fully reduced to a Python scalar
                result = reduced.item()
            else:
                # Partial reduction — wrap remaining dimensions back into a Tile
                result = Tile(reduced, tile.dtype, reduced.shape)
        else:
            # No dim specified — collapse everything to a scalar
            result = np_reduce(tile.data).astype(tile.data.dtype)
    else:
        # Already a scalar, nothing to reduce
        result = tile

    # In MLIR linalg semantics the result is written back into the outs buffer,
    # so downstream ops may reference it by the outs SSA name rather than the
    # result name. Bind both so either reference resolves correctly.
    # Note: this is a pure context-dict write — no latency is charged here;
    # the cost was already recorded by the dispatcher when the handler ran.
    outs_var = op.attributes.get("outs_var")
    if outs_var and result is not None:
        context.set_value(outs_var, result)

    return result


@register("linalg.fill")
def linalg__fill(op, context, env):
    """Fill a tensor with a scalar value."""
    scalar = context.get_value(op.operands[0])
    out_tile = context.get_value(op.operands[1])

    if not isinstance(out_tile, Tile):
        raise TypeError(f"linalg.fill: expected Tile for outs operand, got {type(out_tile)}")

    scalar_val = float(scalar)
    filled = np.full(out_tile.shape, scalar_val, dtype=out_tile.data.dtype)
    return Tile(filled, out_tile.dtype, out_tile.shape)


@register("linalg.broadcast")
def linalg__broadcast(op, context, env):
    """Broadcast a tensor along specified dimensions."""
    inp = context.get_value(op.operands[0])
    out_tile = context.get_value(op.operands[1])
    dims = op.attributes.get("dimensions", [])

    if not isinstance(inp, Tile):
        raise TypeError(f"linalg.broadcast: expected Tile for ins operand, got {type(inp)}")
    if not isinstance(out_tile, Tile):
        raise TypeError(f"linalg.broadcast: expected Tile for outs operand, got {type(out_tile)}")

    data = inp.data
    out_shape = out_tile.shape
    # Expand dims that are being broadcast
    for d in sorted(dims):
        data = np.expand_dims(data, axis=d)
    result = np.broadcast_to(data, out_shape).copy()
    return Tile(result, inp.dtype, out_shape)


@register("linalg.matmul", latency_category=LC.COMPUTE_MATMUL)
def linalg__matmul(op, context, env):
    """Execute linalg.matmul: result = outs + ins[0] @ ins[1].

    MLIR linalg.matmul syntax:
        %result = linalg.matmul
                    ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
                    outs(%C    : tensor<MxNxf32>) -> tensor<MxNxf32>

    The fallback parser (_parse_general_operation) extracts all %name
    references in order, giving operands = [%A, %B, %C].

    In MLIR semantics, outs provides the initial accumulator so the
    result is C + A @ B (not just A @ B).  When C is all zeros this
    degenerates to a plain matmul.
    """
    tile_a = context.get_value(op.operands[0])  # ins[0] = A
    tile_b = context.get_value(op.operands[1])  # ins[1] = B
    result = ArithOps.matmul(tile_a, tile_b)    # A @ B
    # Accumulate into outs (operands[2] = C) when present.
    if len(op.operands) > 2:
        acc = context.get_value(op.operands[2])
        if isinstance(acc, Tile):
            result = Tile(acc.data + result.data, acc.dtype, acc.shape)
    return result


@register("linalg.generic", latency_category=LC.COMPUTE_FLOAT)
def linalg__generic(op, context, env):
    """Vectorised linalg.generic executor.

    Broadcasts each input to the output shape per its indexing_map
    (np.expand_dims for missing dims), binds bb0 block-arg names, then
    executes the region body once with full arrays.
    """
    from ..ops.control_ops import _YieldResult

    n_ins = op.attributes.get("n_ins", 0)
    indexing_maps = op.attributes.get("indexing_maps", [])

    ins_vals = [context.get_value(op.operands[i]) for i in range(n_ins)]
    outs_val = context.get_value(op.operands[n_ins])

    region = op.regions[0] if op.regions else []

    # Resolve bb0 block-argument names.
    # Path 1: synthetic region.bb0_args op prepended to the region (from ^bb0 parser).
    # Path 2: names stored directly in op.attributes["bb0_names"].
    bb0_op = next((o for o in region if o.op_type == "region.bb0_args"), None)
    body_ops = [o for o in region if o.op_type != "region.bb0_args"]
    if bb0_op is not None:
        bb0_names = bb0_op.attributes.get("names", [])
    elif "bb0_names" in op.attributes:
        bb0_names = op.attributes["bb0_names"]
    else:
        raise ValueError("linalg.generic: cannot determine bb0 argument names")

    if not isinstance(outs_val, Tile):
        raise TypeError(f"linalg.generic: outs must be a Tile, got {type(outs_val)}")
    out_shape = outs_val.shape
    out_ndim = len(out_shape)
    out_np_dtype = outs_val.data.dtype

    context.push_scope()

    # Store output shape so linalg.index can build index arrays.
    context.set_value("__linalg_shape__", out_shape)

    # Broadcast each input to the iteration space and bind to its bb0 arg.
    for i, (val, imap) in enumerate(zip(ins_vals, indexing_maps[:n_ins])):
        if isinstance(val, Tile):
            data = val.data
            for d in range(out_ndim):
                if d not in imap:
                    data = np.expand_dims(data, axis=d)
            arg_val = Tile(np.broadcast_to(data, out_shape).copy(), val.dtype, out_shape)
        else:
            arg_val = val
        if i < len(bb0_names):
            context.set_value(bb0_names[i], arg_val)

    # Bind the outs bb0 arg — in MLIR semantics the outs buffer is the
    # initial value of the output block argument.
    if n_ins < len(bb0_names):
        context.set_value(bb0_names[n_ins], Tile(
            outs_val.data.copy(), outs_val.dtype, out_shape
        ))

    result = env.execute_region(context, body_ops)
    context.pop_scope()

    if isinstance(result, _YieldResult):
        out_data = result.values[0]
    else:
        out_data = result

    if isinstance(out_data, Tile):
        data = np.broadcast_to(out_data.data, out_shape).copy().astype(out_np_dtype)
        return Tile(data, outs_val.dtype, out_shape)
    return Tile(np.full(out_shape, out_data, dtype=out_np_dtype), outs_val.dtype, out_shape)


@register("linalg.index")
def linalg__index(op, context, env):
    """Return a broadcasting index array for the given iteration dimension."""
    dim = op.attributes.get("dim", 0)
    out_shape = context.get_value("__linalg_shape__")
    idx = np.arange(out_shape[dim], dtype=np.int64)
    reshape = [1] * len(out_shape)
    reshape[dim] = out_shape[dim]
    return Tile(idx.reshape(reshape), "index", tuple(reshape))


@register("linalg.yield")
def linalg__yield(op, context, env):
    from ..ops.control_ops import _YieldResult
    values = [context.get_value(n) for n in op.operands]
    return _YieldResult(values)


@register("linalg.transpose")
def linalg__transpose(op, context, env):
    inp = context.get_value(op.operands[0])
    permutation = op.attributes.get("permutation")
    if permutation is None:
        raise ValueError("linalg.transpose: missing permutation attribute")
    transposed = np.transpose(inp.data, axes=permutation)
    new_shape = tuple(inp.shape[i] for i in permutation)
    return Tile(transposed.copy(), inp.dtype, new_shape)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# linalg.reduce has two forms:
#   Shorthand:  linalg.reduce { arith.addf } ins(...) outs(...) dimensions = [1]
#     The { arith.addf } has no %SSA references, so the tokenizer keeps it
#     as inline op text.  The parser extracts the combiner name via regex.
#   Explicit region:  linalg.reduce ins(...) outs(...) dimensions = [1]
#                       (%a : f32, %b : f32) { %s = arith.addf ... }
#     The { } block contains %SSA references, so the tokenizer extracts it
#     as a region.  The parser sets reduce_fn=None and the executor resolves
#     the combiner from op.regions.
#
# In both cases the executor maps the combiner name to a NumPy reduction
# (e.g. arith.addf → np.sum) rather than executing the region body
# element-by-element.  A Python-level fold over every element would be
# prohibitively slow for the tile sizes seen in practice.
@register_parser("linalg.reduce")
def parse_linalg_reduce(op_text, parse_ctx):
    """Parse linalg.reduce — shorthand or explicit-region form."""
    result_match = re.match(r'(%\w+)\s*=\s*linalg\.reduce\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    # Shorthand combiner: { arith.addf } in the op text (no %SSA inside)
    reduce_fn = None
    combiner_match = re.search(r'\{\s*(\w+\.\w+)\s*\}', op_text)
    if combiner_match:
        reduce_fn = combiner_match.group(1)

    # dimensions = [1]
    dim = None
    dims_match = re.search(r'dimensions\s*=\s*\[(\d+(?:\s*,\s*\d+)*)\]', op_text)
    if dims_match:
        dims = [int(d.strip()) for d in dims_match.group(1).split(',')]
        dim = dims[0]

    # ins(%x : type) — first operand is the input
    operands = []
    ins_match = re.search(r'ins\((%\w+)', op_text)
    if ins_match:
        operands = [ins_match.group(1)]

    # Extract the outs variable — downstream ops may reference it by this name
    outs_var = None
    outs_match = re.search(r'outs\((%\w+)', op_text)
    if outs_match:
        outs_var = outs_match.group(1)

    attributes = {"reduce_fn": reduce_fn}
    if dim is not None:
        attributes["dim"] = dim
    if outs_var is not None:
        attributes["outs_var"] = outs_var

    return Operation(
        result=result_name,
        op_type="linalg.reduce",
        operands=operands,
        attributes=attributes,
        result_type="unknown"
    )


@register_parser("linalg.fill")
def parse_linalg_fill(op_text, parse_ctx):
    """Parse linalg.fill ins(%scalar : f16) outs(%init : tensor<1xf16>) -> tensor<1xf16>"""
    result_match = re.match(r'(%\w+)\s*=\s*linalg\.fill\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    # Extract ins and outs operands
    ins_match = re.search(r'ins\(([^)]+)\)', op_text)
    outs_match = re.search(r'outs\(([^)]+)\)', op_text)

    operands = []
    if ins_match:
        operands.extend(re.findall(r'%\w+', ins_match.group(1)))
    if outs_match:
        operands.extend(re.findall(r'%\w+', outs_match.group(1)))

    return Operation(
        result=result_name,
        op_type="linalg.fill",
        operands=operands,
        attributes={},
        result_type="unknown"
    )


@register_parser("linalg.transpose")
def parse_linalg_transpose(op_text, parse_ctx):
    """Parse linalg.transpose ins(%x : type) outs(%y : type) permutation = [d0, d1, ...]"""
    result_match = re.match(r'(%\w+)\s*=\s*linalg\.transpose', op_text)
    if not result_match:
        return None
    result_name = result_match.group(1)

    ins_match = re.search(r'ins\s*\(\s*(%\w+)\s*:', op_text)
    outs_match = re.search(r'outs\s*\(\s*(%\w+)\s*:', op_text)
    perm_match = re.search(r'permutation\s*=\s*\[([^\]]+)\]', op_text)

    if not ins_match or not outs_match or not perm_match:
        return None

    permutation = [int(d.strip()) for d in perm_match.group(1).split(',')]
    return Operation(
        result=result_name,
        op_type="linalg.transpose",
        operands=[ins_match.group(1), outs_match.group(1)],
        attributes={"permutation": permutation},
        result_type="unknown"
    )


@register_parser("linalg.generic")
def parse_linalg_generic(op_text, parse_ctx):
    """Parse linalg.generic header."""
    result_match = re.match(r'(%\w+)\s*=\s*linalg\.generic\s+', op_text)
    if not result_match:
        return None
    result_name = result_match.group(1)

    # indexing_maps = [affine_map<(d0, d1) -> (d0)>, ...]
    maps = []
    maps_match = re.search(r'indexing_maps\s*=\s*', op_text)
    if maps_match:
        raw_maps = parse_attr_list(op_text[maps_match.end() - 1:])
        for raw in raw_maps:
            amap = parse_affine_map(raw)
            dims = [e[1] for e in amap.exprs if e[0] == 'dim']
            maps.append(dims)

    ins_operands = []
    ins_match = re.search(r'\bins\s*\(([^)]+)\)', op_text)
    if ins_match:
        ins_operands = re.findall(r'%\w+', ins_match.group(1).split(':')[0])

    outs_operands = []
    outs_match = re.search(r'\bouts\s*\(([^)]+)\)', op_text)
    if outs_match:
        outs_operands = re.findall(r'%\w+', outs_match.group(1).split(':')[0])

    return Operation(
        result=result_name,
        op_type="linalg.generic",
        operands=ins_operands + outs_operands,
        attributes={"indexing_maps": maps, "n_ins": len(ins_operands)},
        result_type="unknown",
    )


@register_parser("linalg.index")
def parse_linalg_index(op_text, parse_ctx):
    """Parse %row = linalg.index 0 : index"""
    m = re.match(r'(%\w+)\s*=\s*linalg\.index\s+(\d+)', op_text)
    if not m:
        return None
    return Operation(
        result=m.group(1),
        op_type="linalg.index",
        operands=[],
        attributes={"dim": int(m.group(2))},
        result_type="index",
    )


@register_parser("linalg.yield")
def parse_linalg_yield(op_text, parse_ctx):
    """Parse linalg.yield %val : type"""
    m = re.match(r'linalg\.yield\s+(.*)', op_text)
    if not m:
        return None
    operands = re.findall(r'%\w+', m.group(1).split(':')[0])
    return Operation(
        result=None,
        op_type="linalg.yield",
        operands=operands,
        attributes={},
        result_type=None,
    )


@register_parser("linalg.broadcast")
def parse_linalg_broadcast(op_text, parse_ctx):
    """Parse linalg.broadcast ins(%x : tensor<1xf16>) outs(%y : tensor<1x1024xf16>) dimensions = [1]"""
    result_match = re.match(r'(%\w+)\s*=\s*linalg\.broadcast\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    ins_match = re.search(r'ins\(([^)]+)\)', op_text)
    outs_match = re.search(r'outs\(([^)]+)\)', op_text)

    operands = []
    if ins_match:
        operands.extend(re.findall(r'%\w+', ins_match.group(1)))
    if outs_match:
        operands.extend(re.findall(r'%\w+', outs_match.group(1)))

    dims = []
    dims_match = re.search(r'dimensions\s*=\s*\[([^\]]*)\]', op_text)
    if dims_match and dims_match.group(1).strip():
        dims = [int(d.strip()) for d in dims_match.group(1).split(',')]

    return Operation(
        result=result_name,
        op_type="linalg.broadcast",
        operands=operands,
        attributes={"dimensions": dims},
        result_type="unknown"
    )
