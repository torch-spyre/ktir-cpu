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

from .ktdp_helpers import parse_subscript_expr
from ..ir_types import AccessTile, IndirectAccessTile, Operation, Tile
from ..latency import LatencyCategory as LC
from ..ops.grid_ops import GridOps
from ..ops.memory_ops import MemoryOps
from ..parser_ast import parse_affine_map, parse_affine_set
from ..parser_utils import _extract_bracket_content, parse_attr_block, split_top_level
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
    shape = tuple(op.attributes["shape"])
    # SSA stride names are stored as strings by the parser; resolve them at runtime.
    strides = [context.get_value(s) if isinstance(s, str) else s for s in op.attributes["strides"]]
    memory_space = op.attributes["memory_space"]
    dtype = op.attributes["dtype"]
    coordinate_set = op.attributes.get("coordinate_set")
    return MemoryOps.tile_view(context, ptr, shape, strides, memory_space, dtype, coordinate_set)


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
    tile_ref = MemoryOps.tile_access(context, parent_ref, indices, access_shape, base_map)
    return AccessTile(
        parent_ref=tile_ref,
        shape=access_shape,
        base_map=base_map,
        coordinate_set=op.attributes.get("coordinate_set"),
        coordinate_order=op.attributes.get("coordinate_order"),
    )


@register("ktdp.load", latency_category=LC.MEMORY)
def ktdp__load(op, context, env):
    access_tile = context.get_value(op.operands[0])
    if isinstance(access_tile, IndirectAccessTile):
        result_shape = op.attributes.get("_result_shape", access_tile.shape)
        return MemoryOps.indirect_load(context, access_tile, result_shape=result_shape)
    css = access_tile.coordinate_set    # AffineSet | None
    cso = access_tile.coordinate_order  # AffineMap | None
    if css is not None:
        coords = css.enumerate(access_tile.shape)
        if cso is not None:
            coords = [cso.eval(pt) for pt in coords]
        result_shape = op.attributes.get("_result_shape", access_tile.shape)
        return MemoryOps.load(context, access_tile.parent_ref, coords=coords, result_shape=result_shape)
    return MemoryOps.load(context, access_tile.parent_ref)


@register("ktdp.store", latency_category=LC.MEMORY)
def ktdp__store(op, context, env):
    value = context.get_value(op.operands[0])
    assert isinstance(value, Tile), f"ktdp.store expects a Tile, got {type(value)}"
    access_tile = context.get_value(op.operands[1])
    if isinstance(access_tile, IndirectAccessTile):
        raise NotImplementedError("ktdp.store with IndirectAccessTile is not yet supported")
    tile_ref = access_tile.parent_ref
    css = access_tile.coordinate_set
    cso = access_tile.coordinate_order
    if css is not None:
        coords = css.enumerate(access_tile.shape)
        if cso is not None:
            coords = [cso.eval(pt) for pt in coords]
        MemoryOps.store(context, value, tile_ref, coords=coords)
    else:
        MemoryOps.store(context, value, tile_ref)
    return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@register_parser("ktdp.get_compute_tile_id")
def parse_get_compute_tile_id(op_text, parse_ctx: ParseContext):
    multi_match = re.match(
        r'((?:%\w+,\s*)*%\w+)\s*=\s*ktdp\.get_compute_tile_id\s*:\s*(.*)', op_text
    )
    if not multi_match:
        return None

    result_names = [r.strip() for r in multi_match.group(1).split(',')]
    result = result_names[0] if len(result_names) == 1 else result_names
    return Operation(
        result=result,
        op_type="ktdp.get_compute_tile_id",
        operands=[],
        attributes={},
        result_type="index"
    )


@register_parser("ktdp.construct_memory_view")
def parse_construct_memory_view(op_text, parse_ctx: ParseContext):
    result_match = re.match(r'(%\w+)\s*=\s*ktdp\.construct_memory_view\s+(%\w+)', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)
    ptr_operand = result_match.group(2)

    # Parse sizes — int values validated against memref dims; SSA names stored as None.
    sizes = None
    sizes_match = re.search(r'sizes\s*:\s*\[([^\]]+)\]', op_text)
    if sizes_match:
        sizes = []
        for token in sizes_match.group(1).split(','):
            token = token.strip()
            try:
                sizes.append(int(token))
            except ValueError:
                sizes.append(None)  # SSA name

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
    mem_match = re.search(r'#ktdp\.spyre_memory_space<(\w+)>', op_text)
    if mem_match:
        memory_space = mem_match.group(1)

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
    memref_dims = [int(p) for p in parts[:-1]]
    shape = tuple(memref_dims)
    # If sizes is None, no checking is performed.
    if sizes is not None:
        for i, s in enumerate(sizes):
            if s is not None and s != memref_dims[i]:
                raise ValueError(
                    f"construct_memory_view: sizes[{i}]={s} does not match "
                    f"memref dimension {memref_dims[i]}"
                )

    attrs = parse_attr_block(op_text, parse_ctx.aliases)
    coordinate_set_str = attrs.get('coordinate_set')
    coordinate_set = parse_affine_set(coordinate_set_str) if isinstance(coordinate_set_str, str) else None

    attributes = {
        "shape": shape,
        "strides": strides,
        "memory_space": memory_space,
        "dtype": dtype,
    }
    if coordinate_set is not None:
        attributes["coordinate_set"] = coordinate_set

    return Operation(
        result=result_name,
        op_type="ktdp.construct_memory_view",
        operands=[ptr_operand] + ssa_stride_operands,
        attributes=attributes,
        result_type=f"memref<{'x'.join(str(s) for s in shape)}x{dtype}>"
    )


@register_parser("ktdp.construct_access_tile")
def parse_construct_access_tile(op_text, parse_ctx: ParseContext):
    result_match = re.match(r'(%\w+)\s*=\s*ktdp\.construct_access_tile\s+', op_text)
    if not result_match:
        return None

    result_name = result_match.group(1)

    after_eq = op_text[op_text.index('=') + 1:]
    operands = re.findall(r'%\w+', after_eq)

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

    coord_set_str = attrs.get('access_tile_set')
    coordinate_set = parse_affine_set(coord_set_str) if isinstance(coord_set_str, str) else None
    # Normalise to None when the set covers the full rectangular tile in
    # row-major order — it carries no information beyond "load/store everything".
    # This lets ktdp.load/store take the contiguous fast path instead of
    # enumerating all coords on every execution.
    if coordinate_set is not None and coordinate_set.is_full(access_shape):
        coordinate_set = None

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
            # Direct: (%h) or (%tkv mod 64) etc.
            inner = dim_text.strip('()')
            var_ref = inner.strip().lstrip('%')
            if var_ref in intermediate_vars:
                dim_subscripts.append({
                    "kind": "direct",
                    "var_index": intermediate_vars.index(var_ref),
                })
            else:
                dim_subscripts.append({
                    "kind": "direct_expr",
                    "subscript": parse_subscript_expr(inner, intermediate_vars),
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


