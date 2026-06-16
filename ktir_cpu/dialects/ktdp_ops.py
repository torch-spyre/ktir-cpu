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

"""KTDP dialect handlers — grid, memory view, access tile, load/store."""

import re

from .ktdp_helpers import attach_reshape, parse_subscript_expr
from ..affine import BoxSet
from ..ir_types import (
    AccessTile,
    DistributedMemRef,
    DistributedTileRef,
    IndirectAccessTile,
    MemRef,
    Operation,
    Tile,
    TileFuture,
)
from ..latency import LatencyCategory as LC
from ..ops.arith_ops import ArithOps
from ..ops.comm_ops import CommPlan, RingReduceBackend
from ..ops.grid_ops import GridOps
from ..ops.memory_ops import MemoryOps
from ..parser_ast import enumerate_membership_keys, parse_affine_map, parse_affine_set
from ..parser_utils import _extract_bracket_content, extract_named_attr, find_ssa_names, parse_attr_block, parse_multi_result_lhs, parse_tensor_type, split_top_level
from .registry import ParseContext, register, register_parser


@register("ktdp.get_compute_tile_id")
def ktdp__get_compute_tile_id(op, context, env):
    num_dims = 1 if isinstance(op.result, str) else len(op.result)
    if num_dims == 1:
        return GridOps.gridid(context, 0)
    return tuple(GridOps.gridid(context, d) for d in range(num_dims))


@register("ktdp.coreid")
def ktdp__coreid(op, context, env):
    grid_coords = [context.get_value(operand) for operand in op.operands]
    return GridOps.coreid(context, grid_coords, env.grid_executor)


@register("ktdp.construct_memory_view")
def ktdp__construct_memory_view(op, context, env):
    if not op.operands:
        raise ValueError("construct_memory_view: missing pointer operand")
    ptr = context.get_value(op.operands[0])
    for attr in ("shape", "strides", "memory_space", "dtype"):
        if attr not in op.attributes:
            raise ValueError(f"construct_memory_view: missing required attribute '{attr}'")
    # SSA size names are stored as strings by the parser; resolve them at runtime.
    shape_attr = op.attributes["shape"]
    shape = tuple(
        context.get_value(s) if isinstance(s, str) else s
        for s in shape_attr
    )
    # SSA stride names are stored as strings by the parser; resolve them at runtime.
    strides = [context.get_value(s) if isinstance(s, str) else s for s in op.attributes["strides"]]
    memory_space = op.attributes["memory_space"]
    dtype = op.attributes["dtype"]
    coordinate_set = op.attributes.get("coordinate_set")
    # Eagerly resolve any symbolic bounds in the coordinate set, per the
    # ODS contract on ``ktdp.construct_memory_view``: "When dimension
    # sizes are symbolic, the symbols in the ``coordinate_set`` integer
    # set are bound to variables in the ``sizes`` operand, from
    # left-to-right in a one-to-one fashion."  Variables here means the
    # SSA operands listed in ``sizes:`` (the dynamic dims), not the
    # static-int entries — for ``memref<64x?xf16>`` the static leading
    # ``64`` must not steal symbol index 0 from the dynamic tail dim.
    # Pre-resolution, dynamic entries appear as SSA-name strings in
    # ``shape_attr``; we filter on that to recover the dynamic operands
    # in declaration order, then resolve each through ``context``.
    # Substituting here keeps downstream consumers (find_partition /
    # distributed_tile_access / load) on the concrete fast path — they
    # never see a symbolic set.
    if isinstance(coordinate_set, BoxSet) and not coordinate_set.is_concrete:
        symbols = tuple(
            context.get_value(s) for s in shape_attr if isinstance(s, str)
        )
        coordinate_set = coordinate_set.specialize(symbols)
    lx_core_id = op.attributes.get("lx_core_id") if memory_space == "LX" else None
    return MemoryOps.tile_view(context, ptr, shape, strides, memory_space, dtype, coordinate_set, lx_core_id)


@register("ktdp.construct_distributed_memory_view")
def ktdp__construct_distributed_memory_view(op, context, env):
    """Compose N per-partition memory views into one distributed view.

    Each operand must resolve to a :class:`MemRef` carrying its own
    ``coordinate_set`` (= B_i in global coords).  The op does not allocate
    or move data — partition resolution at access time is performed by
    ``MemoryOps.distributed_tile_access``.
    """
    partitions = [context.get_value(name) for name in op.operands]
    for i, p in enumerate(partitions):
        if not isinstance(p, MemRef):
            raise ValueError(
                f"construct_distributed_memory_view: operand {i} "
                f"is {type(p).__name__}, expected MemRef"
            )
    if "shape" not in op.attributes or "dtype" not in op.attributes:
        raise ValueError(
            "construct_distributed_memory_view: missing required attributes 'shape'/'dtype'"
        )
    return DistributedMemRef(
        partitions=partitions,
        shape=tuple(op.attributes["shape"]),
        dtype=op.attributes["dtype"],
    )


@register("ktdp.construct_access_tile")
def ktdp__construct_access_tile(op, context, env):
    parent_ref = context.get_value(op.operands[0])
    indices = [context.get_value(operand) for operand in op.operands[1:]]
    if "shape" not in op.attributes:
        raise ValueError("construct_access_tile: missing required attribute 'shape'")
    access_shape = tuple(op.attributes["shape"])
    # base_map is an AffineMap object (always present; synthesized as identity if absent in MLIR).
    # Pass it to tile_access so the offset is computed via affine evaluation rather than
    # the old rectangular sum(idx * stride) shortcut.
    base_map = op.attributes["base_map"]
    coordinate_set = op.attributes.get("coordinate_set")
    # Symbolic ``access_tile_set`` is not yet supported.  Per the ODS
    # spec ("when symbols are present, the ``symbol_operands`` list
    # provides SSA values that are bound to these symbols in
    # left-to-right order"), the binding source is the op's
    # ``$symbol_operands`` argument list — independent from
    # ``$indices`` and from the access tile shape.  The ktir-cpu Python
    # parser does not yet surface that operand list on the
    # ``Operation`` object; wiring it through and threading the
    # resolved symbols here is tracked as a follow-up.
    # Fail fast at this boundary rather than leaking a symbolic set
    # into ``distributed_tile_access`` where it would surface as an
    # opaque ``IndexError`` from ``eval_bound``.
    if isinstance(coordinate_set, BoxSet) and not coordinate_set.is_concrete:
        raise NotImplementedError(
            "construct_access_tile: symbolic access_tile_set is not yet "
            "supported — the parser does not surface the op's "
            "$symbol_operands operand list (per ODS) needed to resolve "
            "the symbols.  Tracked as a follow-up."
        )
    if isinstance(parent_ref, DistributedMemRef):
        # Distributed parent: resolve partition routing now.  Each
        # partition's coordinate_set was already specialised at
        # construct_memory_view time, and ``access_tile_set`` is concrete
        # by current scope — so distributed_tile_access runs entirely on
        # concrete bounds.
        dist_tile_ref = MemoryOps.distributed_tile_access(
            parent_ref, access_shape, base_map, indices, access_tile_set=coordinate_set
        )
        return AccessTile(
            parent_ref=dist_tile_ref,
            shape=access_shape,
            base_map=base_map,
            coordinate_set=coordinate_set,
            coordinate_order=op.attributes.get("coordinate_order"),
        )
    tile_ref = MemoryOps.tile_access(context, parent_ref, indices, access_shape, base_map)
    return AccessTile(
        parent_ref=tile_ref,
        shape=access_shape,
        base_map=base_map,
        coordinate_set=coordinate_set,
        coordinate_order=op.attributes.get("coordinate_order"),
    )


@register("ktdp.load", latency_category=LC.MEMORY)
def ktdp__load(op, context, env):
    access_tile = context.get_value(op.operands[0])
    if isinstance(access_tile, IndirectAccessTile):
        result_shape = op.attributes.get("_result_shape", access_tile.shape)
        return MemoryOps.indirect_load(context, access_tile, result_shape=result_shape)
    if isinstance(access_tile.parent_ref, DistributedTileRef):
        result_shape = op.attributes.get("_result_shape", access_tile.shape)
        return MemoryOps.distributed_load(
            context, access_tile.parent_ref, result_shape=result_shape
        )
    css = access_tile.coordinate_set    # BoxSet | AffineSet (always present post-parse)
    cso = access_tile.coordinate_order  # AffineMap | None
    # BoxSet fast path: rectangular access with identity coordinate order →
    # build a sub-TileRef on the box and let MemoryOps.load take its own
    # contiguous/strided fast path on the sub-ref.  Mirrors the BoxSet branch
    # in MemoryOps.distributed_load — same _subtile_ref + MemoryOps.load
    # composition, keeping a single strategy across distributed and single-
    # allocation paths.  Non-rectangular AffineSet still routes through the
    # coord-list path below where ``_result_shape`` reshapes the gather.
    if isinstance(css, BoxSet) and (cso is None or cso.is_identity()):
        sub_ref = MemoryOps._subtile_ref(access_tile.parent_ref, css)
        return MemoryOps.load(context, sub_ref)
    coords = css.enumerate(access_tile.shape)
    if cso is not None:
        coords = [cso.eval(pt) for pt in coords]
    result_shape = op.attributes.get("_result_shape", access_tile.shape)
    return MemoryOps.load(context, access_tile.parent_ref, coords=coords, result_shape=result_shape)


@register("ktdp.store", latency_category=LC.MEMORY)
def ktdp__store(op, context, env):
    """Stores have no IR result, but the handler returns the HBM
    ``unique_sticks`` (``int``, ``0`` for LX) from ``MemoryOps.store`` /
    ``indirect_store`` / ``distributed_store`` as a latency sideband.
    :meth:`LatencyTracker._data_size` reads it via ``isinstance(result,
    int)`` and charges HBM traffic at stick granularity, matching the
    load-side carrier on ``Tile.unique_sticks``.
    """
    value = context.get_value(op.operands[0])
    assert isinstance(value, Tile), f"ktdp.store expects a Tile, got {type(value)}"
    access_tile = context.get_value(op.operands[1])
    if isinstance(access_tile, IndirectAccessTile):
        return MemoryOps.indirect_store(context, value, access_tile)
    if isinstance(access_tile.parent_ref, DistributedTileRef):
        return MemoryOps.distributed_store(context, value, access_tile.parent_ref)
    tile_ref = access_tile.parent_ref
    css = access_tile.coordinate_set
    cso = access_tile.coordinate_order
    # BoxSet fast path: symmetric to ktdp.load.  Build a sub-TileRef on the
    # box and store through it.  Avoids the read-modify-write scatter that
    # the coord-list path needs for general AffineSet stores.  Same shape as
    # the BoxSet branch in MemoryOps.distributed_store — single strategy
    # across distributed and single-allocation paths.
    if isinstance(css, BoxSet) and (cso is None or cso.is_identity()):
        sub_ref = MemoryOps._subtile_ref(tile_ref, css)
        return MemoryOps.store(context, value, sub_ref)
    coords = css.enumerate(access_tile.shape)
    if cso is not None:
        coords = [cso.eval(pt) for pt in coords]
    return MemoryOps.store(context, value, tile_ref, coords=coords)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
# A multi-result op has two MLIR surface forms:
#   Comma   : "%x, %y = op : index, index"     — referenced as %x, %y
#   Bundled : "%g:2  = op : index, index"      — referenced as %g#0, %g#1
#
# Comma form keeps result names verbatim ("%x", "%y").
# Bundled form synthesizes "%g#0", "%g#1" so downstream operand lookup
# finds distinct keys.
def _make_compute_tile_id_op(result: str | list[str]) -> Operation:
    return Operation(
        result=result,
        op_type="ktdp.get_compute_tile_id",
        operands=[],
        attributes={},
        result_type="index",
    )


@register_parser("ktdp.get_compute_tile_id")
def parse_get_compute_tile_id(op_text, parse_ctx: ParseContext):
    m = re.match(
        r"^(.*?)\s*=\s*ktdp\.get_compute_tile_id\s*:\s*([^{(]*)\s*$", op_text
    )
    if not m:
        return None
    names = parse_multi_result_lhs(m.group(1))
    types_text = m.group(2).strip()
    if not types_text:
        raise ValueError("no result types specified")
    type_list = [t.strip() for t in types_text.split(",")]
    if any(not t for t in type_list):
        raise ValueError(f"empty type in list: {types_text!r}")
    type_count = len(type_list)
    if len(names) != type_count:
        raise ValueError(
            f"ktdp.get_compute_tile_id: {len(names)} result name(s) but "
            f"{type_count} result type(s) in: {op_text!r}"
        )
    result = names[0] if len(names) == 1 else names
    return _make_compute_tile_id_op(result)


@register_parser("ktdp.construct_memory_view")
def parse_construct_memory_view(op_text, parse_ctx: ParseContext):
    result_match = re.match(r'(%\w+)\s*=\s*ktdp\.construct_memory_view\s+(%\w+)', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    ptr_operand = result_match.group(2)

    # Parse sizes — int values validated against memref dims; SSA names stored as strings
    # and collected in ssa_size_operands so the executor can resolve them at runtime.
    sizes = None
    ssa_size_operands = []
    sizes_match = re.search(r'sizes\s*:\s*\[([^\]]+)\]', op_text)
    if sizes_match:
        sizes = []
        for token in sizes_match.group(1).split(','):
            token = token.strip()
            try:
                sizes.append(int(token))
            except ValueError:
                sizes.append(token)  # SSA name — resolved at runtime
                if token not in ssa_size_operands:
                    ssa_size_operands.append(token)

    # SSA stride operands: when a stride is a runtime SSA value (not a literal),
    # the parser stores the name as a string and appends it to ssa_stride_operands
    # so it appears in the Operation's operand list for context.get_value() resolution
    # at execution time (see ktdp__construct_memory_view above).
    strides = [1]
    ssa_stride_operands = []
    strides_match = re.search(r'strides\s*:\s*\[([^\]]+)\]', op_text)
    if strides_match:
        parsed = []
        for token in strides_match.group(1).split(','):
            token = token.strip()
            try:
                parsed.append(int(token))
            except ValueError:
                parsed.append(token)  # SSA name — resolved at runtime
                if token not in ssa_stride_operands:
                    ssa_stride_operands.append(token)
        strides = parsed

    memory_space = "HBM"
    lx_core_id = None
    # Accept both `<HBM>`/`<LX>` and the RFC's per-core LX form
    # `<LX, core = N>`.  On real hardware each compute core has its own
    # private LX SRAM, so a partition tagged `core = N` lives in core N's
    # scratchpad — captured into lx_core_id and used at load/store time.
    mem_match = re.search(
        r'#ktdp\.spyre_memory_space<\s*(\w+)(?:\s*,\s*core\s*=\s*(\d+))?\s*>',
        op_text,
    )
    if mem_match:
        memory_space = mem_match.group(1)
        if mem_match.group(2) is not None:
            lx_core_id = int(mem_match.group(2))

    # dtype and shape are parsed from the memref result type.
    # Validate sizes against memref dimensions when both are concrete.
    memref_match = re.search(r'(?:}\s*)?:\s*(?:index\s*->\s*)?memref<([^>]+)>', op_text)
    if not memref_match:
        raise ValueError("construct_memory_view: could not parse dtype from memref<> type")
    parts = memref_match.group(1).split('x')
    dtype = parts[-1]
    if len(parts) <= 1:
        raise ValueError(
            f"construct_memory_view: memref<{memref_match.group(1)}> has no dimensions"
        )
    # '?' in the memref type means the dimension is dynamic (value only known at
    # runtime).  We keep it as None until we can substitute the SSA size below.
    memref_dims = [None if p == "?" else int(p) for p in parts[:-1]]
    if sizes is not None:
        if len(sizes) != len(memref_dims):
            raise ValueError(
                f"construct_memory_view: sizes count {len(sizes)} does not match "
                f"memref dimension count {len(memref_dims)}"
            )
        resolved = []
        for i, (s, mem_d) in enumerate(zip(sizes, memref_dims)):
            if mem_d is not None:
                # concrete dim — s must equal it
                if not isinstance(s, str) and s != mem_d:
                    raise ValueError(
                        f"construct_memory_view: sizes[{i}]={s} does not match "
                        f"memref dimension {mem_d}"
                    )
                resolved.append(s)
            else:
                # dynamic dim ('?') — s must be an SSA name
                if not isinstance(s, str):
                    raise ValueError(
                        f"construct_memory_view: sizes[{i}]={s} given for a '?' dim; "
                        f"dynamic dim requires an SSA name, not a literal"
                    )
                resolved.append(s)
        shape = tuple(resolved)
    else:
        if any(d is None for d in memref_dims):
            raise ValueError(
                "construct_memory_view: memref has dynamic '?' dim(s) but no sizes: "
                "attribute was provided; dynamic dims require SSA sizes"
            )
        shape = tuple(memref_dims)

    attrs = parse_attr_block(op_text, parse_ctx.aliases)
    coordinate_set_str = attrs.get('coordinate_set')
    coordinate_set = parse_affine_set(coordinate_set_str) if isinstance(coordinate_set_str, str) else None

    attributes = {
        "shape": shape,
        "strides": strides,
        "memory_space": memory_space,
        "dtype": dtype,
    }
    if lx_core_id is not None:
        attributes["lx_core_id"] = lx_core_id
    if coordinate_set is not None:
        attributes["coordinate_set"] = coordinate_set

    return Operation(
        result=result_name,
        op_type="ktdp.construct_memory_view",
        operands=[ptr_operand] + ssa_size_operands + ssa_stride_operands,
        attributes=attributes,
        result_type=f"memref<{'x'.join(str(s) if isinstance(s, int) else '?' for s in shape)}x{dtype}>"
    )


@register_parser("ktdp.construct_distributed_memory_view")
def parse_construct_distributed_memory_view(op_text, parse_ctx: ParseContext):
    """Parse ``%R = ktdp.construct_distributed_memory_view (%a, %b, ... : types) : memref<...>``.

    Variadic memref operands, no required attributes — each input carries
    its own ``coordinate_set`` (on the input's ``construct_memory_view``).
    Result type encodes the global logical shape and dtype.
    """
    result_match = re.match(
        r'(%\w+)\s*=\s*ktdp\.construct_distributed_memory_view', op_text
    )
    if not result_match:
        return None

    result_name = result_match.group(1)

    # Extract operand list from the first parenthesized block.  The block's
    # content is "<%ops> : <types>"; we only need the %ops portion.
    paren_start = op_text.find('(', result_match.end())
    if paren_start == -1:
        raise ValueError(
            "construct_distributed_memory_view: missing operand parenthesis"
        )
    inner = _extract_bracket_content(op_text[paren_start:], '()')
    if inner is None:
        raise ValueError(
            "construct_distributed_memory_view: unbalanced operand parenthesis"
        )
    # Split on the first top-level ':' to separate operands from types.
    colon_idx = None
    depth = 0
    for i, ch in enumerate(inner):
        if ch in '([<':
            depth += 1
        elif ch in ')]>':
            depth -= 1
        elif ch == ':' and depth == 0:
            colon_idx = i
            break
    ops_section = inner if colon_idx is None else inner[:colon_idx]
    operands = [
        t.strip() for t in split_top_level(ops_section) if t.strip().startswith('%')
    ]
    if not operands:
        raise ValueError(
            "construct_distributed_memory_view: no memref operands found"
        )

    # Parse the result memref type: same pattern as construct_memory_view.
    memref_match = re.search(
        r'(?:}\s*)?:\s*memref<([^>]+)>\s*$', op_text.rstrip()
    )
    if not memref_match:
        raise ValueError(
            "construct_distributed_memory_view: could not parse result memref<> type"
        )
    parts = memref_match.group(1).split('x')
    if len(parts) <= 1:
        raise ValueError(
            f"construct_distributed_memory_view: memref<{memref_match.group(1)}> "
            "has no dimensions"
        )
    dtype = parts[-1]
    shape = tuple(int(p) for p in parts[:-1])

    return Operation(
        result=result_name,
        op_type="ktdp.construct_distributed_memory_view",
        operands=operands,
        attributes={"shape": shape, "dtype": dtype},
        result_type=f"memref<{'x'.join(str(s) for s in shape)}x{dtype}>"
    )


@register_parser("ktdp.construct_access_tile")
def parse_construct_access_tile(op_text, parse_ctx: ParseContext):
    result_match = re.match(r'(%\w+)\s*=\s*ktdp\.construct_access_tile\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    after_eq = op_text[op_text.index('=') + 1:]
    operands = find_ssa_names(after_eq)

    tile_match = re.search(r'!ktdp\.access_tile<([^>]+)>', op_text)
    if not tile_match:
        raise ValueError("construct_access_tile: missing !ktdp.access_tile<> result type")
    inner = tile_match.group(1)
    # Split on the boundary between the numeric dims and the element-type
    # identifier.  A naive split('x') would shred "index" into ['inde', '']
    # because 'index' contains 'x'.  The regex anchors group(1) to the
    # leading "NxMx..." portion and group(2) to the trailing type name.
    type_match = re.match(r'^(\d+(?:x\d+)*)x([a-zA-Z_]\w*)$', inner)
    if not type_match:
        raise ValueError(
            f"Malformed access_tile type {inner!r}: expected '<dims>xindex>'"
        )
    elem_type = type_match.group(2)
    if elem_type != "index":
        raise ValueError(
            f"AccessTileType element type must be 'index', got {elem_type!r}"
        )
    shape_parts = [int(d) for d in type_match.group(1).strip('x').split('x') if d]
    access_shape = tuple(shape_parts)

    # Extract and resolve affine attributes; parse into AffineMap/AffineSet objects.
    # Aliases are resolved immediately so Operation.attributes holds concrete objects.
    attrs = parse_attr_block(op_text, parse_ctx.aliases)

    base_map_str = attrs.get('base_map')
    if not isinstance(base_map_str, str):
        # Synthesize an identity map from the number of index operands
        # (all operands except the first, which is the memref).
        n = max(1, len(operands) - 1)
        dims = ', '.join(f'd{i}' for i in range(n))
        base_map_str = f'affine_map<({dims}) -> ({dims})>'
    base_map = parse_affine_map(base_map_str)

    # ``access_tile_set`` is required per ODS
    # (Builtin_IntegerSetAttr:$access_tile_set, no OptionalAttr); absence
    # is invalid IR.  ``parse_affine_set`` lowers axis-aligned sets to
    # ``BoxSet`` automatically, so the full-rectangle case naturally
    # comes out as ``BoxSet([0, shape))`` and routes through the BoxSet
    # fast path in the dialect handler — no normalisation needed.
    coord_set_str = attrs.get('access_tile_set')
    if not isinstance(coord_set_str, str):
        raise ValueError(
            "construct_access_tile: access_tile_set is required (per ODS)"
        )
    coordinate_set = parse_affine_set(coord_set_str)
    # ``access_shape`` is structurally ``int`` (parsed from
    # ``<NxMxindex>``); ``raise`` (not ``assert``) so the guard survives
    # ``python -O``.
    if not all(isinstance(s, int) for s in access_shape):
        raise TypeError(
            f"construct_access_tile: access_shape must be concrete ints, "
            f"got {access_shape!r}"
        )

    coord_order_str = attrs.get('access_tile_order')
    coordinate_order = parse_affine_map(coord_order_str) if isinstance(coord_order_str, str) else None
    # Normalise to None when the order map is the identity — it does not
    # permute element positions, so applying it would be a no-op.
    if coordinate_order is not None and coordinate_order.is_identity():
        coordinate_order = None

    attributes = {"shape": access_shape, "base_map": base_map}
    if coordinate_set is not None:
        attributes["coordinate_set"] = coordinate_set
    if coordinate_order is not None:
        attributes["coordinate_order"] = coordinate_order

    return Operation(
        result=result_name,
        op_type="ktdp.construct_access_tile",
        operands=operands,
        attributes=attributes,
        result_type=f"!ktdp.access_tile<{'x'.join(str(s) for s in access_shape)}xindex>"
    )


# ---------------------------------------------------------------------------
# ktdp.construct_indirect_access_tile
# ---------------------------------------------------------------------------

@register("ktdp.construct_indirect_access_tile")
def ktdp__construct_indirect_access_tile(op, context, env):
    parent_ref = context.get_value(op.operands[0])
    index_views = [context.get_value(name) for name in op.operands[1:]]

    intermediate_vars = op.attributes.get("intermediate_vars", [])

    def _resolve_node(e):
        """Resolve a subscript node to a concrete ("const", value) if possible.

        Leaf nodes:
          ("ssa",  "%name") – outer SSA value; always resolved to ("const", v).
          ("dim",  i)       – iteration variable i; two sub-cases:
                              (a) Backward-compat: intermediate variable name
                                  matches an outer SSA binding (old syntax where
                                  intermediate_variables listed outer SSA names).
                                  Resolved to ("const", v).

                                  NOTE: this case is questionable. An intermediate
                                  variable should vary across the iteration space;
                                  if it is bound to a constant at runtime, it is not
                                  truly a variable. This also complicates coordinate
                                  sets: a runtime-resolved SSA value cannot satisfy
                                  static polyhedral constraints. Consider removing
                                  this case and requiring such values to be passed as
                                  explicit SSA operands instead.
                              (b) Pure iteration variable (new syntax: %d0..%dN
                                  not bound in context).  Left as ("dim", i) for
                                  eval_subscript_expr at load time.
          ("const", value)  – already resolved; returned unchanged.

        Compound nodes ("add", "sub", "mul", "neg") are recursed into so that
        mixed expressions like ("add", ("ssa", "%grid0"), ("dim", 0)) become
        ("add", ("const", grid0_val), ("dim", 0)) after resolution.

        Example — paged attention K-tile (current form, %bt_idx + %d0 is evaluated as an "affine" expr):
            intermediate_variables(%d0, %d1, %d2, %d3)
            %K_cache_view[ind(%block_tables_view[%c0, %bt_idx + %d0]), (%d1), (%pid1 + %d2), (%d3)]
            %bt_idx → ("ssa", "%bt_idx") → ("const", bt_idx_value)  [explicit SSA ref]
            %d0     → ("dim", 0)                                   [pure iterator]

        Example — illustrative form that exercises var case (a):
            intermediate_variables(%bt_idx, %m, %pid1, %k)
            %K_cache_view[ind(%block_tables_view[%c0, %bt_idx]), (%m), (%pid1), (%k)]
            %c0     → ("ssa", "%c0")  → ("const", 0)          [outer SSA]
            %bt_idx → ("dim", 0); "%bt_idx" is in context      → ("const", bt_idx_value)  [dim case (a)]
            %m      → ("dim", 1); "%m" is NOT in context       → ("dim", 1)  [iterator]

        Example — RFC 2-D gather:
            intermediate_variables(%m, %k)
            ind(%IDX1[%m, %k])

            %m → ("dim", 0); "%m" is NOT in context → ("dim", 0)  [iterator]
            %k → ("dim", 1); "%k" is NOT in context → ("dim", 1)  [iterator]
        """
        tag = e[0]
        if tag == "ssa":
            return ("const", int(context.get_value(e[1])))
        if tag == "dim":
            # [var case (a) — remove this block if eliminating]
            ssa_name = "%" + intermediate_vars[e[1]] if e[1] < len(intermediate_vars) else None
            if ssa_name is not None:
                try:
                    val = context.get_value(ssa_name)
                    return ("const", int(val))
                except (KeyError, TypeError):
                    pass  # pure iteration variable — keep as ("dim", i)
            return e
        if tag == "const":
            return e
        # Compound node (add, sub, mul, neg, ...): recurse into tuple children.
        # Non-tuple children (e.g. integer coefficients in "mul") pass through.
        return (tag,) + tuple(
            _resolve_node(c) if isinstance(c, tuple) else c for c in e[1:]
        )

    dim_subscripts = []
    for sub in op.attributes["dim_subscripts"]:
        sub = dict(sub)  # shallow copy

        if "idx_exprs" in sub:
            # Indirect dim: resolve each index expression used to look up the
            # index tensor (e.g. block_tables[%c0, %bt_idx]).
            sub["idx_exprs"] = [_resolve_node(e) for e in sub["idx_exprs"]]

        elif sub.get("kind") == "direct_expr":
            # Direct dim with an expression node (e.g. a quasi-affine offset).
            sub["subscript"] = _resolve_node(sub["subscript"])

        elif sub.get("kind") == "direct":
            # Direct dim referencing an intermediate variable by index.
            # If that variable is actually an outer SSA scalar (e.g. %pid1),
            # convert it to a direct_expr constant so indirect_load uses the
            # real value instead of the variable-space position (which would
            # always be 0 for a scalar dimension constrained to a single point).
            #
            # [var case (a) — remove the promotion below if eliminating;
            #  the entire elif body would then be a no-op and can be removed]
            #
            # Example — paged attention:
            #   (%pid1) → sub["var_index"] = 2; "%pid1" is in context with value 3
            #   → _resolve_node(("dim", 2)) returns ("const", 3)
            #   → becomes {"kind": "direct_expr", "subscript": ("const", 3)}
            #
            # Example — 2-D gather:
            #   (%m) → sub["var_index"] = 0; "%m" is NOT in context
            #   → left unchanged as {"kind": "direct", "var_index": 0}
            var_idx = sub["var_index"]
            ssa_name = "%" + intermediate_vars[var_idx] if var_idx < len(intermediate_vars) else None
            if ssa_name is not None:
                try:
                    val = context.get_value(ssa_name)
                    sub = {"kind": "direct_expr", "subscript": ("const", int(val))}
                except (KeyError, TypeError):
                    pass  # pure iteration variable — leave unchanged

        dim_subscripts.append(sub)

    shape = tuple(op.attributes["shape"])
    vss = op.attributes["variables_space_set"]

    # Validate: an intermediate variable that resolves to an outer SSA scalar
    # must have a zero-range dimension in variables_space_set (i.e. pt[i] == 0
    # for all enumerated points).  Using a non-zero range with an SSA variable
    # silently produces wrong results because the iterator position, not the
    # SSA value, would drive that coordinate.  Use the new explicit-offset
    # syntax (%base + %di) instead.
    for i, var_name in enumerate(intermediate_vars):
        try:
            context.get_value("%" + var_name)  # bound → outer SSA scalar
        except (KeyError, TypeError):
            continue  # pure iteration variable — no constraint needed
        if any(pt[i] != 0 for pt in vss.enumerate(shape)):
            raise ValueError(
                f"construct_indirect_access_tile: intermediate variable "
                f"%{var_name} (index {i}) is an outer SSA value but its "
                f"dimension in variables_space_set has a non-zero range. "
                f"Use the explicit-offset syntax (%{var_name} + %d{i}) and "
                f"list pure iteration variables in intermediate_variables."
            )

    return IndirectAccessTile(
        parent_ref=parent_ref,
        shape=shape,
        dim_subscripts=dim_subscripts,
        index_views=index_views,
        variables_space_set=vss,
        variables_space_order=op.attributes.get("variables_space_order"),
    )


@register_parser("ktdp.construct_indirect_access_tile")
def parse_construct_indirect_access_tile(op_text, parse_ctx: ParseContext):
    # Match: %result = ktdp.construct_indirect_access_tile
    result_match = re.match(
        r'(%\w+)\s*=\s*ktdp\.construct_indirect_access_tile\s+', op_text
    )
    if not result_match:
        return None

    result_name = result_match.group(1)

    # Extract intermediate variable names from: intermediate_variables(%m, %k)
    iv_match = re.search(r'intermediate_variables\s*\(([^)]+)\)', op_text)
    if not iv_match:
        return None
    intermediate_vars = [v.strip().lstrip('%') for v in iv_match.group(1).split(',')]

    # Extract the primary memref operand and subscript block: %X[...]
    # Find the first %name[ after intermediate_variables(...)
    iv_end = iv_match.end()
    rest = op_text[iv_end:]
    primary_match = re.match(r'\s*(%\w+)\[', rest)
    if not primary_match:
        return None
    primary_operand = primary_match.group(1)

    # Extract the full subscript content between the outer [ ... ]
    bracket_start = iv_end + primary_match.end() - 1  # index of '['
    subscript_text = _extract_bracket_content(op_text[bracket_start:], '[]')
    if subscript_text is None:
        return None

    # Parse each dimension subscript, splitting on top-level commas
    dim_subscripts = []
    operands = [primary_operand]
    index_view_idx = 0

    for dim_text in split_top_level(subscript_text):
        dim_text = dim_text.strip()
        if dim_text.startswith('ind('):
            # Indirect: ind(%IDX1[%m, %k])
            inner = dim_text[4:-1]  # strip ind( ... )
            view_match = re.match(r'(%\w+)\[([^\]]*)\]', inner)
            if not view_match:
                return None
            view_name = view_match.group(1)
            var_refs = [v.strip() for v in view_match.group(2).split(',')]
            idx_exprs = [parse_subscript_expr(v, intermediate_vars) for v in var_refs]
            dim_subscripts.append({
                "kind": "indirect",
                "index_view_idx": index_view_idx,
                "idx_exprs": idx_exprs,
            })
            operands.append(view_name)
            index_view_idx += 1
        else:
            # Direct subscript: bare variable (%h), parenthesised expression
            # ((%tkv mod 64)), or compound expression ((%a + %b * 64) floordiv 64).
            # Parenthesised forms like (expr) are valid affine syntax — _atom
            # handles the outer parens, so pass dim_text directly rather than
            # stripping, which would corrupt compound-LHS expressions.
            dim_subscripts.append({
                "kind": "direct_expr",
                "subscript": parse_subscript_expr(dim_text, intermediate_vars),
            })

    # Parse attribute block
    attrs = parse_attr_block(op_text, parse_ctx.aliases)

    vs_set_str = attrs.get('variables_space_set')
    if not isinstance(vs_set_str, str):
        return None  # variables_space_set is required
    variables_space_set = parse_affine_set(vs_set_str)

    vs_order_str = attrs.get('variables_space_order')
    variables_space_order = parse_affine_map(vs_order_str) if isinstance(vs_order_str, str) else None
    if variables_space_order is not None and variables_space_order.is_identity():
        variables_space_order = None

    # Parse access tile shape from result type
    tile_match = re.search(r'!ktdp\.access_tile<([^>]+)>', op_text)
    if not tile_match:
        raise ValueError("construct_indirect_access_tile: missing !ktdp.access_tile<> result type")
    inner = tile_match.group(1)
    type_match = re.match(r'^(\d+(?:x\d+)*)x([a-zA-Z_]\w*)$', inner)
    if not type_match:
        raise ValueError(
            f"construct_indirect_access_tile: malformed access_tile type {inner!r}"
        )
    access_shape = tuple(int(d) for d in type_match.group(1).split('x'))

    attributes = {
        "shape": access_shape,
        "dim_subscripts": dim_subscripts,
        "intermediate_vars": intermediate_vars,
        "variables_space_set": variables_space_set,
    }
    if variables_space_order is not None:
        attributes["variables_space_order"] = variables_space_order

    return Operation(
        result=result_name,
        op_type="ktdp.construct_indirect_access_tile",
        operands=operands,
        attributes=attributes,
        result_type=f"!ktdp.access_tile<{'x'.join(str(s) for s in access_shape)}xindex>"
    )




# ---------------------------------------------------------------------------
# Inter-tile communication — `ktdp.inter_tile_produce` / `inter_tile_reduce`
# ---------------------------------------------------------------------------
# 🧪 EXPERIMENTAL — TRACKS UNMERGED UPSTREAM SPEC PR
#
# The four-op inter-tile design (produce + consume / reduce / reduce_scatter)
# lives in ktir-mlir-frontend PR #23:
#   https://github.com/torch-spyre/ktir-mlir-frontend/pull/23
#
# The spec PR is not yet merged.  Op names, attribute keys (e.g.
# ``producer_tiles_per_group``, ``consumer_tiles_per_group``,
# ``producer_dependency_per_consumer``), and the ``!ktdp.tile_future<...>``
# type may shift to track the upstream PR before it lands on main.  Avoid
# baking these names into stable APIs until the spec is final.
#
# Implemented here: the reduce path only — `ktdp.inter_tile_produce` +
# `ktdp.inter_tile_reduce`, with `yield_partial` / `yield_reduced` region
# terminators.  Production returns a `!ktdp.tile_future<T_p>`; the delivery
# op consumes it.  v1 supports a single partial role (N=1) and the
# full-barrier sync model (no `producer_dependency_per_consumer` execution
# semantics — parsed but not honoured at runtime).
#
# Not implemented: `inter_tile_consume` (broadcast),
# `inter_tile_reduce_scatter`, per-tile sync runtime.
#
# Def-use edge — how it's simulated.  The spec uses the SSA def-use
# edge ``%fut → consume(%fut)`` to pin produce-then-consume ordering
# and identify which produce a delivery op is paired with.  In this
# simulator we don't walk the def-use graph: each core's
# ``inter_tile_produce`` returns a per-core ``TileFuture`` instance
# bound to that core's local ``%fut``; the consume handler reads
# ``%fut`` via ``ctx.get_value`` and dispatches.  The TileFuture
# *is* the def-use edge for our purposes — its identity per core
# substitutes for tracing the IR-level chain.
#
# Verification — deferred until the upstream spec is final.  Once
# ktir-mlir-frontend#23 lands, add:
#
#   A2. ``groups`` match between produce and consume.
#       Bounded-enumeration equality over ``ctx.num_cores`` —
#       compare ``{g : groups_a.contains([g])}`` and
#       ``{g : groups_b.contains([g])}`` once at consume-handler
#       entry.  No polyhedral set difference needed.
#
#   B1. Subset.  ``producer_dependency_per_consumer ⊆
#       producer_tiles_per_group`` for every (p, c) the deps name.
#       Trivial post-check on ``CommPlan.for_reduce`` — every
#       producer-id mentioned in ``deps`` must be in ``producers``.
#
#   B2. Coverage.  Every producer in ``producer_tiles_per_group(g)``
#       must be the dependency of at least one consumer in
#       ``consumer_tiles_per_group(g)``.  Same place as B1, post
#       ``CommPlan.for_reduce``.
#
# These are spec invariants the runtime currently trusts.  They
# should be enforced when the upstream surface stabilises so we
# don't lock in error messages keyed on names that may yet change.
#
# See `docs/cross_core_scheduling.md` for the simulator design and
# `docs/gap_analysis.md` rows 2a–2d for status.
# ---------------------------------------------------------------------------


def _find_tile_group(tile_id, producer_set, groups_set, num_cores):
    """Return the unique group index ``g`` whose membership set
    contains ``tile_id``.

    Thin wrapper over :func:`enumerate_membership_keys`: ``producer_set``
    is the family ``(i)[g]``, ``groups_set`` is the key domain, and
    we want the keys ``g`` for which ``tile_id`` is in the family.
    Group count is upper-bounded by ``num_cores`` (every group must
    contain at least one core).  Raises if zero or more than one
    match — the spec's disjointness invariant says exactly one.
    """
    matches = enumerate_membership_keys(
        family=producer_set,
        domain=groups_set,
        point=(tile_id,),
        bound=num_cores,
    )
    if not matches:
        raise RuntimeError(
            f"tile {tile_id} is not contained in any producer group "
            f"(producer_set={producer_set.source!r}, groups_set={groups_set.source!r})"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"tile {tile_id} matched multiple groups {matches} — "
            f"violates the disjointness invariant"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# ktdp.inter_tile_produce
# ---------------------------------------------------------------------------

@register("ktdp.inter_tile_produce")
def ktdp__inter_tile_produce(op, context, env):
    """Materialise this core's partial and stash it on a per-core
    TileFuture.

    Every core in the workgroup runs this handler once with its own
    ``CoreContext``; each call returns a separate ``TileFuture``
    bound to that core's local ``%fut`` SSA value.  No cross-core
    shared state — partials reach consumers via the scheduler's
    mailbox once the matching delivery op runs.  See
    ``docs/cross_core_scheduling.md``, "Inter-tile communication:
    produce + reduce, end to end".

    Backend selection happens at consume time, not here — the future
    just carries the producer/groups affine sets and the bound
    group index alongside the local partial.

    Steps:
      1. Resolve this core's group index ``gid`` from the IR sets.
      2. Execute the producer region with ``%gid`` bound; capture
         the ``yield_partial`` value(s) as the local partial.
      3. Wrap everything in a ``TileFuture`` and return.
    """
    producer_set = op.attributes["producer_tiles_per_group"]
    groups_set = op.attributes["groups"]
    partial_types = op.attributes["partial_tensor_types"]

    tile_id = context.core_id
    gid = _find_tile_group(tile_id, producer_set, groups_set, context.num_cores)

    region = op.regions[0] if op.regions else []
    bb0_op = next((o for o in region if o.op_type == "region.bb0_args"), None)
    body = [o for o in region if o.op_type != "region.bb0_args"]
    if bb0_op is not None:
        names = bb0_op.attributes.get("names", [])
        if names:
            context.set_value(names[0], gid)

    yielded = env.execute_region(context, body)
    if isinstance(yielded, Tile):
        yielded = (yielded,)
    local_partial = tuple(yielded) if yielded is not None else None

    return TileFuture(
        partial_tensor_types=partial_types,
        local_partial=local_partial,
        producer_set=producer_set,
        groups_set=groups_set,
        group_idx=gid,
    )


@register_parser("ktdp.inter_tile_produce")
def parse_inter_tile_produce(op_text, parse_ctx: ParseContext):
    m = re.match(r'(%\w+)\s*=\s*ktdp\.inter_tile_produce\b', op_text)
    if not m:
        return None
    result_name = m.group(1)

    producer_set_str = extract_named_attr(
        op_text, "producer_tiles_per_group", parse_ctx.aliases
    )
    if producer_set_str is None:
        raise ValueError("ktdp.inter_tile_produce: missing producer_tiles_per_group")
    groups_str = extract_named_attr(op_text, "groups", parse_ctx.aliases)
    if groups_str is None:
        raise ValueError("ktdp.inter_tile_produce: missing groups")

    producer_set = parse_affine_set(producer_set_str)
    groups_set = parse_affine_set(groups_str)

    # Result type:  !ktdp.tile_future<T_p_1, ..., T_p_N>
    fut_match = re.search(r'!ktdp\.tile_future<(.+)>', op_text)
    if not fut_match:
        raise ValueError(
            "ktdp.inter_tile_produce: missing !ktdp.tile_future<...> result type"
        )
    inner = fut_match.group(1).strip()
    # split_top_level handles commas inside nested tensor<...> brackets.
    partial_types = tuple(p.strip() for p in split_top_level(inner))

    return Operation(
        result=result_name,
        op_type="ktdp.inter_tile_produce",
        operands=[],
        attributes={
            "producer_tiles_per_group": producer_set,
            "groups": groups_set,
            "partial_tensor_types": partial_types,
        },
        result_type=f"!ktdp.tile_future<{inner}>",
    )


# ---------------------------------------------------------------------------
# ktdp.yield_partial / ktdp.yield_reduced — region terminators
# ---------------------------------------------------------------------------

@register("ktdp.yield_partial")
def ktdp__yield_partial(op, context, env):
    values = [context.get_value(name) for name in op.operands]
    return values[0] if len(values) == 1 else tuple(values)


@register("ktdp.yield_reduced")
def ktdp__yield_reduced(op, context, env):
    values = [context.get_value(name) for name in op.operands]
    return values[0] if len(values) == 1 else tuple(values)


def _parse_yield(op_text, op_name):
    m = re.match(rf'{re.escape(op_name)}\s+(.*)', op_text)
    if not m:
        return None
    rest = m.group(1)
    # Strip trailing type annotation
    if ':' in rest:
        rest = rest[:rest.rindex(':')]
    operands = find_ssa_names(rest)
    return Operation(
        result=None,
        op_type=op_name,
        operands=operands,
        attributes={},
        result_type=None,
    )


@register_parser("ktdp.yield_partial")
def parse_yield_partial(op_text, parse_ctx: ParseContext):
    return _parse_yield(op_text, "ktdp.yield_partial")


@register_parser("ktdp.yield_reduced")
def parse_yield_reduced(op_text, parse_ctx: ParseContext):
    return _parse_yield(op_text, "ktdp.yield_reduced")


# ---------------------------------------------------------------------------
# ktdp.inter_tile_reduce
# ---------------------------------------------------------------------------

def _select_reduce_backend(plan: CommPlan, op_attrs):
    """Pick a ``ReduceBackend`` for an ``inter_tile_reduce`` op.

    Today: a single fixed strategy (``RingReduceBackend``).  When
    other strategies (tree, point-to-point for sparse deps, …) land,
    this dispatcher pattern-matches on ``plan`` shape and selects.
    """
    return RingReduceBackend()


@register("ktdp.inter_tile_reduce", latency_category=LC.COMM)
def ktdp__inter_tile_reduce(op, context, env):
    """Build a ``CommPlan``, pick a backend, run it.

    Every core in the workgroup runs this handler, even cores not in
    ``consumer_tiles_per_group`` — the backend's ring spans the whole
    workgroup; ``CommPlan`` masks contributions and outputs.  See
    ``docs/cross_core_scheduling.md`` §"Inter-tile communication".

    Steps:
      1. Validate the ``%fut`` operand and resolve the ``identity``
         operand to a Tile.
      2. Build a ``CommPlan`` from the future's producer set (plus
         this op's consumer set + optional dep set) at
         ``ctx.num_cores``.
      3. Build a ``reduce_fn`` from the combiner region
         (``^bb0(%lhs, %rhs) → yield_reduced``).
      4. Pick a backend and run it; ``attach_reshape`` collapses
         ``T_p → T_r`` on the result tile.
    """
    fut = context.get_value(op.operands[0])
    if not isinstance(fut, TileFuture):
        raise TypeError(
            f"ktdp.inter_tile_reduce: operand 0 must be a TileFuture, got {type(fut)}"
        )

    # Producers see local_partial; non-producers seed the ring with
    # the identity tensor instead.  ``RingReduceBackend.run`` does the
    # mask check via ``plan.is_producer``.
    local_tile = fut.local_partial[0] if fut.local_partial else None
    identity = context.get_value(op.operands[1])

    # Build the logical plan.
    plan = CommPlan.for_reduce(
        producer_set=fut.producer_set,
        consumer_set=op.attributes["consumer_tiles_per_group"],
        group_idx=fut.group_idx,
        num_cores=context.num_cores,
        dep_set=op.attributes.get("producer_dependency_per_consumer"),
    )

    # Build reduce_fn from the combiner region.
    region = op.regions[0] if op.regions else []
    bb0_op = next((o for o in region if o.op_type == "region.bb0_args"), None)
    body = [o for o in region if o.op_type != "region.bb0_args"]
    if bb0_op is None:
        raise ValueError("ktdp.inter_tile_reduce: combiner region missing ^bb0 args")
    bb0_names = bb0_op.attributes.get("names", [])
    if len(bb0_names) < 2:
        raise ValueError(
            f"ktdp.inter_tile_reduce: combiner region needs >=2 block args "
            f"(lhs, rhs), got {bb0_names}"
        )
    lhs_name, rhs_name = bb0_names[0], bb0_names[1]

    def reduce_fn(t1: Tile, t2: Tile) -> Tile:
        context.push_scope()
        context.set_value(lhs_name, t1)
        context.set_value(rhs_name, t2)
        result = env.execute_region(context, body)
        context.pop_scope()
        if not isinstance(result, Tile):
            raise TypeError(
                f"ktdp.inter_tile_reduce: combiner did not yield a Tile, got {type(result)}"
            )
        return result

    backend = _select_reduce_backend(plan, op.attributes)
    target_shape = op.attributes.get("_result_shape")
    raw = backend.run(context, local_tile, plan, reduce_fn, identity)
    return attach_reshape(raw, target_shape)


@register_parser("ktdp.inter_tile_reduce")
def parse_inter_tile_reduce(op_text, parse_ctx: ParseContext):
    m = re.match(
        r'(%\w+)\s*=\s*ktdp\.inter_tile_reduce\s*\(\s*(%\w+)\s*\)',
        op_text,
    )
    if not m:
        return None
    result_name = m.group(1)
    fut_operand = m.group(2)

    consumer_set_str = extract_named_attr(
        op_text, "consumer_tiles_per_group", parse_ctx.aliases
    )
    if consumer_set_str is None:
        raise ValueError("ktdp.inter_tile_reduce: missing consumer_tiles_per_group")
    consumer_set = parse_affine_set(consumer_set_str)

    groups_str = extract_named_attr(op_text, "groups", parse_ctx.aliases)
    if groups_str is None:
        raise ValueError("ktdp.inter_tile_reduce: missing groups")
    groups_set = parse_affine_set(groups_str)

    pdpc_str = extract_named_attr(
        op_text, "producer_dependency_per_consumer", parse_ctx.aliases
    )
    pdpc = parse_affine_set(pdpc_str) if pdpc_str is not None else None

    # identity(%add_id : T_p_1, ...): SSA operands hoisted before the op.
    id_match = re.search(r'identity\s*\(([^)]*)\)', op_text)
    identity_operands = []
    if id_match:
        for part in split_top_level(id_match.group(1)):
            for name in find_ssa_names(part):
                identity_operands.append(name)

    # Result type: T_r (after collapsing within-group tile axes).
    arrow_match = re.search(r'->\s*(.+?)\s*\{?\s*$', op_text)
    result_type = arrow_match.group(1).strip() if arrow_match else None
    result_shape = None
    if result_type is not None:
        parsed = parse_tensor_type(result_type)
        if parsed is not None:
            result_shape = parsed["shape"]

    attributes = {
        "consumer_tiles_per_group": consumer_set,
        "groups": groups_set,
    }
    if pdpc is not None:
        attributes["producer_dependency_per_consumer"] = pdpc
    if result_shape is not None:
        attributes["_result_shape"] = result_shape

    operands = [fut_operand] + identity_operands

    return Operation(
        result=result_name,
        op_type="ktdp.inter_tile_reduce",
        operands=operands,
        attributes=attributes,
        result_type=result_type,
    )
