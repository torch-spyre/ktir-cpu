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
KTDP subscript-expression helpers.

These functions parse and evaluate subscript descriptors used by
``construct_indirect_access_tile`` and ``load``/``store``.  They live in
this separate module (rather than in ``ktdp_ops`` or ``affine``) to avoid
circular imports: ``ktdp_ops`` imports ``MemoryOps`` from
``ops/memory_ops``, which in turn needs ``eval_subscript_expr``.

Functions
---------
parse_subscript_expr  — parse one subscript token into an AST tuple
eval_subscript_expr   — evaluate a subscript tuple against a variable point
_classify_refs        — post-process ("ref", ...) nodes from the text-parser path
_reclassify_dims      — post-process ("dim", N) nodes from the MLIR-frontend-parser path
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
    return node  # "const" — pass through unchanged


def parse_subscript_expr(token: str, var_names: list):
    """Parse one subscript token from an ``ind(...)`` or direct dim expression.

    Uses the affine expression parser for full binary-op support (+, -, *),
    then classifies ("ref", ...) nodes via :func:`_classify_refs`:

      - ``%name`` where bare name is in *var_names*  → ``("dim", i)``
      - ``%name`` not in *var_names*                 → ``("ssa", "%name")``
      - integer literal                              → ``("const", value)``
      - ``lhs OP rhs``  (OP = +, -, *)              → ``("add/sub/mul", ...)``

    Legacy keyword forms handled via regex fast-path:
      - ``%name floordiv N``  → ``("floordiv", i, N)``
      - ``%name mod N``       → ``("mod", i, N)``

    ``var_names`` is the list of intermediate variable names (without ``%``).
    """
    from ..parser_ast import _Parser, _tokenise

    t = token.strip()
    bare = t.lstrip("%")

    # Legacy fast-path: keyword forms not in the expression grammar.
    m = _re.match(r'^(\w+)\s+floordiv\s+(\d+)$', bare)
    if m:
        return ("floordiv", var_names.index(m.group(1)), int(m.group(2)))
    m = _re.match(r'^(\w+)\s+mod\s+(\d+)$', bare)
    if m:
        return ("mod", var_names.index(m.group(1)), int(m.group(2)))

    # General case: recursive-descent parse, then classify named refs.
    node = _Parser(_tokenise(t)).parse_expr()
    return _classify_refs(node, var_names)


def eval_subscript_expr(expr: tuple, pt: tuple) -> int:
    """Evaluate a subscript descriptor against an intermediate-variable point.

    *expr* must be a tuple produced by :func:`parse_subscript_expr`.
    *pt* is the variable vector (one int per intermediate variable).

    Delegates to the affine AST evaluator for generic node types
    (const, dim, add, sub, mul, neg).  Intervenes only for the legacy
    ("floordiv", i, N) / ("mod", i, N) forms not in the affine grammar.
    ("ssa", ...) nodes must be pre-resolved to ("const", v) by
    _resolve_node before eval is called.
    """
    from ..parser_ast import _eval_node
    if expr[0] == "floordiv":
        return pt[expr[1]] // expr[2]
    if expr[0] == "mod":
        return pt[expr[1]] % expr[2]
    return _eval_node(expr, list(pt))
