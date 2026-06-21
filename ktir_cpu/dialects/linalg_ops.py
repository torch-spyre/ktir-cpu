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
from ..parser_utils import find_ssa_names, parse_attr_list
from ._helpers import unwrap_yield
from .registry import register, register_parser


def _resolve_region_body(op):
    """Resolve a linalg op's region into ``(bb0_names, body_ops)``.

    Shared by ``linalg.generic`` and ``linalg.reduce`` so both capture block
    arguments uniformly, whether the source wrote an explicit ``^bb0`` label
    or used inline block args / shorthand.  Block-argument names are found in
    priority order:

      1. a synthetic ``region.bb0_args`` op (from the ``^bb0(...)`` parser);
      2. an ``op.attributes["bb0_names"]`` list (mlir_frontend path);
      3. the operand names of the region's first non-yield op — this is the
         ``linalg.reduce`` explicit form ``(%in, %out) { %s = addf %in, %out }``,
         where the block args appear only as the combiner's operands.

    Returns ``([], [])`` when the op has no region (e.g. reduce shorthand);
    callers synthesize a region in that case.
    """
    region = op.regions[0] if op.regions else []
    body_ops = [o for o in region if o.op_type != "region.bb0_args"]

    bb0_op = next((o for o in region if o.op_type == "region.bb0_args"), None)
    if bb0_op is not None:
        return bb0_op.attributes.get("names", []), body_ops
    if "bb0_names" in op.attributes:
        return op.attributes["bb0_names"], body_ops
    if body_ops:
        # Inline-block form: block args are the first body op's operands.
        return list(body_ops[0].operands), body_ops
    return [], []


def _run_combiner(bb0_names, body_ops, lhs, rhs, context, env):
    """Run the combiner region once on two equal-shaped operands.

    Binds the block args to ``lhs``/``rhs`` (Tiles or scalars) in an isolated
    scope and dispatches the region via ``execute_region`` — so each combiner
    op fires through ``record_op`` and is charged latency under its own
    category.  Returns the yielded value.
    """
    from ..ops.control_ops import _YieldResult

    context.push_scope()
    try:
        if bb0_names:
            context.set_value(bb0_names[0], lhs)
        if len(bb0_names) > 1:
            context.set_value(bb0_names[1], rhs)
        result = env.execute_region(context, body_ops)
    finally:
        context.pop_scope()

    if isinstance(result, _YieldResult):
        return result.values[0]
    return result


def _tree_fold(tile, dim, bb0_names, body_ops, context, env):
    """Reduce ``tile`` along ``dim`` by folding the combiner region pairwise.

    MLIR legalizes ``linalg.reduce`` only for an **associative** combiner, so
    the fold order is free to choose.  We fold pairwise (tree reduction): split
    the reduced axis in half, combine the two halves with one *vectorised*
    region call, and repeat.  This needs ``ceil(log2(N))`` region executions
    instead of ``N`` sequential steps — each call combines whole sub-arrays in
    a single op, so a 1×262144 reduce is ~18 vectorised passes, not 262k scalar
    folds.  Odd lengths carry the unpaired slice forward to the next round.

    The value *and* the latency both come from these region calls — there is no
    NumPy reduction shortcut and no mapping from combiner name to a NumPy op;
    whatever ops the region contains are executed and charged as-is.
    """
    data = tile.data

    # Reduce to scalar (no dim) → flatten everything onto one axis first.
    if dim is None:
        data = data.reshape(-1)
        dim = 0
    n = data.shape[dim]

    def slice_along(arr, lo, hi):
        idx = [slice(None)] * arr.ndim
        idx[dim] = slice(lo, hi)
        return arr[tuple(idx)]

    # Current accumulator is the full tile; collapse `dim` from n → 1 pairwise.
    acc = data
    while n > 1:
        half = n // 2
        left = slice_along(acc, 0, half)
        right = slice_along(acc, half, 2 * half)
        combined = _run_combiner(
            bb0_names, body_ops,
            Tile(left.copy(), tile.dtype, left.shape),
            Tile(right.copy(), tile.dtype, right.shape),
            context, env,
        )
        combined = combined.data if isinstance(combined, Tile) else np.asarray(combined)
        if n % 2:  # odd: carry the leftover slice into the next round
            tail = slice_along(acc, 2 * half, n)
            combined = np.concatenate([combined, tail], axis=dim)
        acc = combined
        n = acc.shape[dim]

    return acc  # extent 1 along `dim`


@register("linalg.reduce", latency_category=LC.ZERO)
def linalg__reduce(op, context, env):
    """Standard linalg.reduce — removes the reduced dimension.

    Like ``linalg.generic``, ``linalg.reduce`` is a zero-cost orchestrator: the
    cost belongs to the ops in its combiner *region*, not to the reduce op
    itself.  Rather than mapping the combiner to a hardcoded NumPy reduction
    (which only covers the handful of combiners we anticipated), we **execute
    the region** to compute the result — so the data value and the latency both
    come from the real combiner ops, whatever they are.

    Both surface forms feed the same region path:

      * explicit form — use the region as parsed, capturing bb0 args via
        :func:`_resolve_region_body`;
      * shorthand form (``linalg.reduce { arith.addf }``) — *build* a one-op
        region from the named combiner so it goes through the identical path.
        (We read the combiner *name* to construct the op — that is the
        shorthand syntax itself, not a semantic interpretation of it.)

    The reduction is computed by a pairwise tree fold of the combiner region
    (:func:`_tree_fold`); this relies on the combiner being associative, which
    MLIR's ``linalg.reduce`` legalization already requires.
    """
    # operands[0] is the ins tensor; the outs tensor is handled separately via outs_var
    tile = context.get_value(op.operands[0])

    # --- Resolve the combiner region (capturing bb0 args) ------------------
    bb0_names, body_ops = _resolve_region_body(op)

    # Combiner op name: explicit form has it as the region's first op;
    # shorthand form stores it in the reduce_fn attribute.
    reduce_fn = op.attributes.get("reduce_fn")
    if reduce_fn is None and body_ops:
        reduce_fn = next(
            (o.op_type for o in body_ops if o.op_type != "linalg.yield"), None
        )
    if reduce_fn is None:
        reduce_fn = "arith.addf"

    # Shorthand has no region — build one so it goes through the same fold.
    # The synthesized block mirrors the explicit form:
    #   (%in, %out) { %s = <reduce_fn> %in, %out; linalg.yield %s }
    if not body_ops:
        bb0_names = ["__reduce_in__", "__reduce_acc__"]
        body_ops = [
            Operation(
                result="__reduce_combined__",
                op_type=reduce_fn,
                operands=bb0_names,
                attributes={},
                result_type="unknown",
            ),
            Operation(
                result=None, op_type="linalg.yield",
                operands=["__reduce_combined__"], attributes={},
                result_type=None,
            ),
        ]

    dims = op.attributes.get("dims")
    if dims is None and op.attributes.get("dim") is not None:
        dims = [op.attributes["dim"]]

    outs_var = op.attributes.get("outs_var")
    outs_tile = context.get_value(outs_var) if outs_var else None

    if isinstance(tile, Tile):
        if dims is None:
            # Collapse all axes: flatten then fold.
            folded = _tree_fold(tile, None, bb0_names, body_ops, context, env)
        else:
            # Fold each axis, fastest-moving (rightmost) first.
            folded = tile.data
	    # Tree-folding each axis independently reorders element groupings vs.
	    # MLIR's left-associative scalar loop — see test_reduce_multi_axis_treefold_bug.
            for d in sorted(dims, reverse=True):
                folded = _tree_fold(
                    Tile(folded, tile.dtype, folded.shape),
                    d, bb0_names, body_ops, context, env,
                )

        # Squeeze all reduced axes.
        if dims is None:
            reduced = folded.reshape(()).astype(tile.data.dtype)
        else:
            reduced = folded
            for d in sorted(dims, reverse=True):
                reduced = np.squeeze(reduced, axis=d)
            reduced = reduced.astype(tile.data.dtype)

        # Combine with the outs initial accumulator.
        if isinstance(outs_tile, Tile):
            reduced_tile = Tile(reduced.copy(), tile.dtype, reduced.shape)
            combined = _run_combiner(
                bb0_names, body_ops, reduced_tile, outs_tile, context, env,
            )
            reduced = combined.data if isinstance(combined, Tile) else np.asarray(combined)

        if reduced.ndim == 0:
            result = reduced.item()
        else:
            result = Tile(reduced, tile.dtype, reduced.shape)
    else:
        # Already a scalar, nothing to reduce.
        result = tile

    if outs_var and result is not None:
        context.set_value(outs_var, result)

    return result


@register("linalg.add", latency_category=LC.COMPUTE_FLOAT)
def linalg__add(op, context, env):
    """Elementwise tensor add — ``%c = linalg.add ins(%a, %b) outs(%init)``.

    Standard MLIR named op: result = a + b (the outs buffer provides the
    destination shape only; its values are not accumulated into in v1).
    """
    tile_a = context.get_value(op.operands[0])
    tile_b = context.get_value(op.operands[1])
    if not isinstance(tile_a, Tile) or not isinstance(tile_b, Tile):
        raise TypeError(
            f"linalg.add: ins must be Tiles, got {type(tile_a)} and {type(tile_b)}"
        )
    return Tile(tile_a.data + tile_b.data, tile_a.dtype, tile_a.shape)


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


@register("linalg.batch_matmul", latency_category=LC.COMPUTE_MATMUL)
def linalg__batch_matmul(op, context, env):
    """Execute linalg.batch_matmul: result = outs + ins[0] @ ins[1] (batched).

    MLIR linalg.batch_matmul syntax:
        %result = linalg.batch_matmul
                    ins(%A, %B : tensor<BxMxKxf32>, tensor<BxKxNxf32>)
                    outs(%C    : tensor<BxMxNxf32>) -> tensor<BxMxNxf32>

    Operands = [%A, %B, %C] (fallback parser order).
    """
    tile_a = context.get_value(op.operands[0])  # ins[0] = A  (B×M×K)
    tile_b = context.get_value(op.operands[1])  # ins[1] = B  (B×K×N)
    product = tile_a.data @ tile_b.data
    result = Tile(product, tile_a.dtype, product.shape)
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
    n_ins = op.attributes.get("n_ins", 0)
    indexing_maps = op.attributes.get("indexing_maps", [])

    ins_vals = [context.get_value(op.operands[i]) for i in range(n_ins)]
    outs_val = context.get_value(op.operands[n_ins])

    # Resolve bb0 block-argument names and body via the shared helper
    # (also used by linalg.reduce).
    bb0_names, body_ops = _resolve_region_body(op)
    if not bb0_names:
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

    out_data = unwrap_yield(result)

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
    from ..ops.control_ops import ControlOps
    values = [context.get_value(n) for n in op.operands]
    return ControlOps.yield_op(values)


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

    # dimensions = [1] or dimensions = [0, 1] or dimensions = []
    dims = None
    dims_match = re.search(r'dimensions\s*=\s*\[([^\]]*)\]', op_text)
    if dims_match:
        content = dims_match.group(1).strip()
        dims = [int(d.strip()) for d in content.split(',') if d.strip()] if content else []

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
    if dims is not None:
        attributes["dims"] = dims
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
        operands.extend(find_ssa_names(ins_match.group(1)))
    if outs_match:
        operands.extend(find_ssa_names(outs_match.group(1)))

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
        ins_operands = find_ssa_names(ins_match.group(1).split(':')[0])

    outs_operands = []
    outs_match = re.search(r'\bouts\s*\(([^)]+)\)', op_text)
    if outs_match:
        outs_operands = find_ssa_names(outs_match.group(1).split(':')[0])

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
    operands = find_ssa_names(m.group(1).split(':')[0])
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
        operands.extend(find_ssa_names(ins_match.group(1)))
    if outs_match:
        operands.extend(find_ssa_names(outs_match.group(1)))

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
