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

"""
KTDP dialect helpers.

Functions
---------
Subscript-expression helpers — parse and evaluate subscript
descriptors used by ``construct_indirect_access_tile`` and
``load``/``store``.  They live in this module (rather than in
``ktdp_ops`` or ``affine``) to avoid circular imports: ``ktdp_ops``
imports ``MemoryOps`` from ``ops/memory_ops``, which in turn needs
``eval_subscript_expr``.

  parse_subscript_expr  — parse one subscript token into an AST tuple
  eval_subscript_expr   — evaluate a subscript tuple against a variable point
  _classify_refs        — post-process ``("ref", ...)`` nodes (text parser path)
  _reclassify_dims      — post-process ``("dim", N)`` nodes (frontend path)

Region- and post-processing helpers — small utilities used by
region-bearing inter-tile op handlers.

  reshape_tile_to_target  — collapse a tile to its declared ``T_r`` shape
  attach_reshape          — drive a backend generator and reshape its return
"""

from __future__ import annotations

import re as _re


def _classify_refs(node, var_names: list):
    """Post-process ("ref", tok) nodes from the affine parser into ktdp nodes.

    ("ref", tok) nodes are produced for any %name or unrecognised bare name.
    This function maps them to:
      - ("dim", i)        if bare name is in var_names  (iteration variable)
      - ("ssa", "%name")  otherwise                     (outer SSA scalar,
                          resolved to ("const", v) at construction time)

    Recurses into ("add", ...), ("sub", ...), ("mul", ...), ("neg", ...).
    ("const", ...) and ("dim", ...) pass through unchanged.

    Without this step, ("ref", ...) nodes would reach eval_subscript_expr
    and _eval_node, which have no "ref" case and would raise ValueError.
    More subtly, outer SSA scalars (e.g. %grid0) must become ("ssa", ...)
    so that _resolve_node in ktdp_ops can resolve them to ("const", value)
    at construct-op execution time — before load-time iteration begins.
    """
    tag = node[0]
    if tag == "ref":
        bare = node[1].lstrip("%")
        if bare in var_names:
            return ("dim", var_names.index(bare))
        return ("ssa", node[1] if node[1].startswith("%") else "%" + node[1])
    if tag in ("add", "sub"):
        return (tag, _classify_refs(node[1], var_names), _classify_refs(node[2], var_names))
    if tag == "mul":
        return (tag, node[1], _classify_refs(node[2], var_names))
    if tag == "neg":
        return (tag, _classify_refs(node[1], var_names))
    if tag in ("floordiv", "mod"):
        return (tag, _classify_refs(node[1], var_names), node[2])
    return node  # "const", "dim" — pass through unchanged


def _reclassify_dims(node, ssa_operand_names: list):
    """Post-process ("dim", N) nodes from an affine map into ktdp nodes.

    Counterpart to _classify_refs, but for the MLIR frontend parser path.

    _classify_refs handles text-parsed IR: _Parser emits ("ref", "%name") for
    every variable reference and _classify_refs decides whether the name is an
    iteration variable or an outer SSA scalar.

    _reclassify_dims handles affine-map-parsed IR: the MLIR frontend exposes subscript
    expressions as AffineMap objects whose domain dims encode both SSA operands
    (d0..d(n_ssa-1)) and iteration variables (d(n_ssa)..).  _Parser already
    emits ("dim", N) for dN notation, so no "ref" nodes exist; instead we
    reclassify by index:

      ("dim", N) where N < n_ssa  → ("ssa", ssa_operand_names[N])
      ("dim", N) where N >= n_ssa → ("dim", N - n_ssa)   (iteration variable)

    After reclassification the AST is identical to what _classify_refs produces,
    and the executor handles it without any further changes.
    """
    n_ssa = len(ssa_operand_names)
    tag = node[0]
    if tag == "dim":
        j = node[1]
        return ("ssa", ssa_operand_names[j]) if j < n_ssa else ("dim", j - n_ssa)
    if tag in ("add", "sub"):
        return (tag, _reclassify_dims(node[1], ssa_operand_names),
                _reclassify_dims(node[2], ssa_operand_names))
    if tag == "mul":
        return (tag, node[1], _reclassify_dims(node[2], ssa_operand_names))
    if tag == "neg":
        return (tag, _reclassify_dims(node[1], ssa_operand_names))
    if tag in ("floordiv", "mod"):
        return (tag, _reclassify_dims(node[1], ssa_operand_names), node[2])
    return node  # "const" — pass through unchanged


def parse_subscript_expr(token: str, var_names: list):
    """Parse one subscript token from an ``ind(...)`` or direct dim expression.

    Uses the affine expression parser for full binary-op support (+, -, *),
    then classifies ("ref", ...) nodes via :func:`_classify_refs`:

      - ``%name`` where bare name is in *var_names*  → ``("dim", i)``
      - ``%name`` not in *var_names*                 → ``("ssa", "%name")``
      - integer literal                              → ``("const", value)``
      - ``lhs OP rhs``  (OP = +, -, *)              → ``("add/sub/mul", ...)``

    ``var_names`` is the list of intermediate variable names (without ``%``).
    """
    from ..parser_ast import _Parser, _tokenise

    t = token.strip()
    node = _Parser(_tokenise(t)).parse_expr()
    return _classify_refs(node, var_names)


def eval_subscript_expr(expr: tuple, pt: tuple) -> int:
    """Evaluate a subscript descriptor against an intermediate-variable point.

    *expr* must be a tuple produced by :func:`parse_subscript_expr`.
    *pt* is the variable vector (one int per intermediate variable).

    Fully delegates to the affine AST evaluator, which handles all node
    types including floordiv and mod.  ("ssa", ...) nodes must be
    pre-resolved to ("const", v) by _resolve_node before eval is called.
    """
    from ..parser_ast import _eval_node
    return _eval_node(expr, list(pt))


# ---------------------------------------------------------------------------
# Region / post-processing helpers
# ---------------------------------------------------------------------------


def reshape_tile_to_target(reduced, target_shape):
    """Reshape a ``Tile`` to ``target_shape``, validating element count.

    Returns ``reduced`` unchanged when ``target_shape`` is ``None`` or
    already matches.  Used by inter-tile reduce to collapse the
    within-group tile axis (``T_p → T_r``) on the post-ring tile.

    Raises:
        TypeError: if ``reduced`` is not a ``Tile`` (the backend's
            return contract is violated).
        ValueError: if reshape is requested but element counts differ.
    """
    from ..ir_types import Tile
    if not isinstance(reduced, Tile):
        raise TypeError(
            f"reshape_tile_to_target: expected Tile, got {type(reduced).__name__}"
        )
    if target_shape is None or reduced.shape == tuple(target_shape):
        return reduced
    from numpy import prod
    if int(prod(reduced.shape)) != int(prod(target_shape)):
        raise ValueError(
            f"reshape_tile_to_target: result shape {reduced.shape} and "
            f"declared shape {target_shape} have different element counts"
        )
    new_tile = Tile(
        reduced.data.reshape(target_shape),
        reduced.dtype,
        tuple(target_shape),
        unique_sticks=reduced.unique_sticks,
        index_unique_sticks=reduced.index_unique_sticks,
    )
    # Tile() doesn't construct comm_bytes (mirrors how copy() leaves
    # it None for produced-from-data copies).  Reshape is a structural
    # rewrite of an already-comm-stamped tile, so propagate explicitly.
    new_tile.comm_bytes = reduced.comm_bytes
    return new_tile


def attach_reshape(gen, target_shape):
    """Drive ``gen`` to completion and reshape its return value.

    Returns a generator that proxies every ``yield`` from ``gen`` and
    post-processes the return through :func:`reshape_tile_to_target`.
    The handler that wants its backend's output reshaped writes::

        return attach_reshape(backend.run(...), target_shape)

    instead of an inline ``def _gen(): reduced = yield from ...``
    closure.
    """
    reduced = yield from gen
    if reduced is None:
        return None
    return reshape_tile_to_target(reduced, target_shape)
