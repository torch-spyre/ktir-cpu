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
Affine expression parser and evaluator.

This module owns all AST-level concerns for affine maps and sets:
  - Tokenisation
  - Recursive-descent expression parser → AST nodes
  - AST evaluation / enumeration

Public API
----------
parse_affine_map(s)          -> AffineMap
parse_affine_set(s)          -> AffineSet
eval_affine_map(amap, dims)  -> tuple[int, ...]
affine_set_contains(aset, point) -> bool
enumerate_affine_set(aset, shape) -> list[tuple[int, ...]]

The first two are called by ``parser.py`` (MLIR structure parser) when it
encounters an ``affine_map<...>`` or ``affine_set<...>`` string.
The remaining three are called by the convenience methods on ``AffineMap``
and ``AffineSet`` in ``affine.py``, and directly by ``memory_ops.py``.

Supported scope
---------------
Linear affine expressions only:
  - Dimension variables:  d0, d1, ...
  - Integer constants
  - Addition (+), subtraction (-), negation (unary -)
  - Multiplication by a constant coefficient (N * dI)

No symbolic identifiers (s0, s1, ...), floordiv, ceildiv, or mod.

Assumption for affine_set enumeration
-------------------------------------
Constraints are interpreted in *local* (0-based) coordinates within the
access tile.  The caller passes the tile shape as a bounding box.
"""

from __future__ import annotations

import itertools
import re
from typing import List, Optional, Sequence, Tuple

from .affine import AffineMap, AffineSet


# ---------------------------------------------------------------------------
# AST node representation
#
# A node is a plain tuple so it is hashable and works inside frozen dataclasses.
#
#   ("const", int)            — integer constant
#   ("dim",   int)            — dimension variable dN → index N
#   ("ref",   str)            — named reference; str is the raw token (e.g.
#                               "d0" or "%grid0"). Callers that need domain-
#                               specific semantics post-process these nodes.
#   ("add",   node, node)
#   ("sub",   node, node)
#   ("neg",   node)           — unary negation
#   ("mul",   int,  node)     — constant-coefficient multiplication
# ---------------------------------------------------------------------------

_Node = tuple


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(%[a-zA-Z_]\w*)'             # named reference %name (group 1)
    r'|([a-zA-Z_]\w*)'             # bare identifier  (group 2)
    r'|(-?\d+)'                    # integer literal, possibly negative (group 3)
    r'|(>=|<=|->|[+\-*(),:])'      # operator / punctuation  (group 4)
    r')'
)


def _tokenise(text: str) -> List[str]:
    tokens: List[str] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or not m.group(0).strip():
            pos += 1
            continue
        tok = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        tokens.append(tok)
        pos = m.end()
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent expression parser
# https://en.wikipedia.org/wiki/Recursive_descent_parser
# - For affine expressions, it is more robust to use AST-parsing as opposed to 
#   regexes.
# - MLIR is a context-free grammar in EBNF, see
#  https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
# - Current implementation mainly uses regex to parse the MLIR (see parser.py), but
#   AST-parsing is more robust, so can consider to use AST-parsing to handle 
#   generic MLIR (custom dialect parsing could still be handled using regex).
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens
        self.pos = 0
        # Maps dim name (e.g. "i", "d0", "row") to its positional index.
        # Set by callers after parsing the dim list so that _atom can resolve
        # arbitrary identifiers used as dimension variables.
        self.dim_index: dict = {}

    # --- helpers ---

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.tokens[self.pos]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r} (pos {self.pos})")
        self.pos += 1
        return tok

    # --- grammar ---

    def parse_dim_list(self) -> List[str]:
        """Parse ``(d0, d1, ...)`` and return the name list."""
        self.consume("(")
        names: List[str] = []
        while self.peek() != ")":
            names.append(self.consume())
            if self.peek() == ",":
                self.consume(",")
        self.consume(")")
        return names

    def parse_expr(self) -> _Node:
        return self._additive()

    def _additive(self) -> _Node:
        left = self._term()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self._term()
            left = ("add", left, right) if op == "+" else ("sub", left, right)
        return left

    def _term(self) -> _Node:
        tok = self.peek()

        # Unary minus
        if tok == "-":
            self.consume("-")
            operand = self._atom()
            return ("neg", operand)

        # Integer that may be a coefficient: N * expr
        if tok is not None and re.fullmatch(r'-?\d+', tok):
            num = int(self.consume())
            if self.peek() == "*":
                self.consume("*")
                operand = self._atom()
                return ("mul", num, operand)
            return ("const", num)

        # Atom that may be followed by a coefficient: expr * N
        node = self._atom()
        if self.peek() == "*":
            self.consume("*")
            num_tok = self.peek()
            if num_tok is None or not re.fullmatch(r'-?\d+', num_tok):
                raise ValueError(f"Expected integer coefficient after '*', got {num_tok!r}")
            return ("mul", int(self.consume()), node)
        return node

    def _atom(self) -> _Node:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        # Parenthesised sub-expression
        if tok == "(":
            self.consume("(")
            node = self.parse_expr()
            self.consume(")")
            return node

        # Named reference: %identifier — raw token passed through for callers
        # to resolve into domain-specific node types (e.g. iteration variable
        # or outer SSA value in ktdp subscript expressions).
        if tok.startswith("%"):
            return ("ref", self.consume())

        # Dimension variable — any identifier that appears in the dim list.
        # Canonical names are dN (e.g. d0, d1), but arbitrary names like "i"
        # or "row" are valid MLIR affine syntax.  We look the name up in
        # dim_index (populated from the parsed dim list) so we can resolve any
        # name to its positional index.
        # Fallback: if dim_index is empty (e.g. standalone parse_expr call),
        # accept dN directly by parsing the numeric suffix.
        if tok in self.dim_index:
            self.consume()
            return ("dim", self.dim_index[tok])
        # When called via parse_expr() directly (e.g. in tests) there is no
        # surrounding affine_map/set to build dim_index from.  In that context
        # we fall back to treating canonical dN identifiers as positional dims
        # by parsing the numeric suffix directly (d0→0, d1→1, ...).
        if re.fullmatch(r'd\d+', tok) and not self.dim_index:
            self.consume()
            return ("dim", int(tok[1:]))

        # Positive integer constant (negative handled in _term)
        if re.fullmatch(r'\d+', tok):
            return ("const", int(self.consume()))

        raise ValueError(f"Unexpected token: {tok!r}")

    def parse_expr_list(self) -> List[_Node]:
        """Parse ``(e0, e1, ...)`` and return the expression list."""
        self.consume("(")
        exprs: List[_Node] = []
        while self.peek() != ")":
            exprs.append(self.parse_expr())
            if self.peek() == ",":
                self.consume(",")
        self.consume(")")
        return exprs

    def parse_constraint_list(self) -> List[_Node]:
        """Parse ``(lhs >= rhs, lhs <= rhs, ...)`` and return normalised nodes.

        Each constraint is normalised to ``lhs - rhs >= 0``:
          - ``lhs >= rhs``  → stored as ``("sub", lhs, rhs)``
          - ``lhs <= rhs``  → stored as ``("sub", rhs, lhs)``

        The special case ``expr >= 0`` / ``expr <= 0`` is handled naturally
        since ``("sub", expr, 0_const)`` evaluates identically to ``expr``.
        """
        self.consume("(")
        constraints: List[_Node] = []
        while self.peek() != ")":
            lhs = self.parse_expr()
            op = self.consume()  # ">=" or "<="
            rhs = self.parse_expr()
            if op == ">=":
                # lhs >= rhs  →  lhs - rhs >= 0
                node = ("sub", lhs, rhs)
            elif op == "<=":
                # lhs <= rhs  →  rhs - lhs >= 0
                node = ("sub", rhs, lhs)
            else:
                raise ValueError(f"Unsupported constraint operator: {op!r}")
            constraints.append(node)
            if self.peek() == ",":
                self.consume(",")
        self.consume(")")
        return constraints


# ---------------------------------------------------------------------------
# Outer wrapper stripping
# ---------------------------------------------------------------------------

def _strip_outer(s: str, keyword: str) -> str:
    """Strip the ``affine_map<...>`` or ``affine_set<...>`` wrapper.

    If the string does not start with *keyword*, it is returned unchanged
    (the caller passed inner text directly, which is also accepted).
    """
    s = s.strip()
    # Match: keyword < inner_content >
    # - re.escape(keyword) matches the literal keyword (e.g. "affine_map")
    # - <(.+)> captures everything between the outermost < and the final >
    # - re.DOTALL so '.' matches newlines in multi-line attributes
    # - fullmatch ensures the entire string is consumed (no trailing junk)
    m = re.fullmatch(re.escape(keyword) + r'<(.+)>', s, re.DOTALL)
    if m:
        return m.group(1)
    if s.startswith(keyword):
        raise ValueError(f"Malformed {keyword} expression: {s!r}")
    return s


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------

def parse_affine_map(s: str) -> AffineMap:
    """Parse ``affine_map<(d0,...) -> (e0,...)>`` into an :class:`AffineMap`.

    The ``affine_map<...>`` wrapper is optional — the inner text is also
    accepted.

    Raises:
        ValueError: on any parse error.
    """
    source = s.strip()
    inner = _strip_outer(source, "affine_map")
    tokens = _tokenise(inner)
    p = _Parser(tokens)
    dim_names = p.parse_dim_list()
    # Build name→index map so _atom can resolve any identifier (e.g. "i", "row")
    # to its positional index, not just canonical dN names.
    p.dim_index = {name: idx for idx, name in enumerate(dim_names)}
    p.consume("->")
    out_exprs = p.parse_expr_list()
    return AffineMap(
        n_dims=len(dim_names),
        exprs=tuple(out_exprs),
        source=source,
    )


def parse_affine_set(s: str) -> AffineSet:
    """Parse ``affine_set<(d0,...) : (c0 >= 0, ...)>`` into an :class:`AffineSet`.

    The ``affine_set<...>`` wrapper is optional.

    Raises:
        ValueError: on any parse error.
    """
    source = s.strip()
    inner = _strip_outer(source, "affine_set")
    colon = inner.index(":")
    dim_part = inner[:colon].strip()
    con_part = inner[colon + 1:].strip()

    dim_tokens = _tokenise(dim_part)
    p1 = _Parser(dim_tokens)
    dim_names = p1.parse_dim_list()

    con_tokens = _tokenise(con_part)
    p2 = _Parser(con_tokens)
    # Share the same name→index map so constraint expressions can reference
    # the same dim names as the dim list.
    p2.dim_index = {name: idx for idx, name in enumerate(dim_names)}
    constraints = p2.parse_constraint_list()

    return AffineSet(
        n_dims=len(dim_names),
        constraints=tuple(constraints),
        source=source,
    )


# ---------------------------------------------------------------------------
# Evaluation functions (called by AffineMap / AffineSet convenience methods)
# ---------------------------------------------------------------------------

def _eval_node(node: _Node, dims: List[int]) -> int:
    """Recursively evaluate an AST node given concrete dimension values."""
    tag = node[0]
    if tag == "const":
        return node[1]
    if tag == "dim":
        return dims[node[1]]
    if tag == "add":
        return _eval_node(node[1], dims) + _eval_node(node[2], dims)
    if tag == "sub":
        return _eval_node(node[1], dims) - _eval_node(node[2], dims)
    if tag == "neg":
        return -_eval_node(node[1], dims)
    if tag == "mul":
        return node[1] * _eval_node(node[2], dims)
    raise ValueError(f"Unknown AST node tag: {tag!r}")  # pragma: no cover


def eval_affine_map(amap: AffineMap, dims: Sequence[int]) -> Tuple[int, ...]:
    """Evaluate *amap* given concrete dimension values.

    Args:
        amap: Parsed AffineMap.
        dims: Concrete values for d0, d1, ...

    Returns:
        Tuple of output integers, one per output expression.

    Raises:
        ValueError: if ``len(dims) != amap.n_dims``.
    """
    if len(dims) != amap.n_dims:
        raise ValueError(
            f"AffineMap expects {amap.n_dims} dim(s), got {len(dims)}: {amap.source!r}"
        )
    env = list(dims)
    return tuple(_eval_node(e, env) for e in amap.exprs)


def affine_set_contains(aset: AffineSet, point: Sequence[int]) -> bool:
    """Return True if *point* satisfies all constraints in *aset*."""
    env = list(point)
    return all(_eval_node(c, env) >= 0 for c in aset.constraints)


def enumerate_affine_set(aset: AffineSet, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Return all integer points in ``[0, shape)`` satisfying *aset*.

    Args:
        aset:  Parsed AffineSet.
        shape: Bounding box — one upper bound (exclusive) per dimension.

    Returns:
        List of coordinate tuples in row-major order.

    Raises:
        ValueError: if ``len(shape) != aset.n_dims``.
    """
    if len(shape) != aset.n_dims:
        raise ValueError(
            f"AffineSet has {aset.n_dims} dim(s), got shape with {len(shape)}: {aset.source!r}"
        )
    ranges = [range(s) for s in shape]
    return [pt for pt in itertools.product(*ranges) if affine_set_contains(aset, pt)]


# ---------------------------------------------------------------------------
# Low-level helpers exposed for testing
# ---------------------------------------------------------------------------

def parse_expr(s: str) -> _Node:
    """Parse a single affine expression string into an AST node.

    Useful for unit-testing the expression parser in isolation without
    wrapping the expression in a full ``affine_map<...>`` or
    ``affine_set<...>`` string.

    Args:
        s: Expression text, e.g. ``"-d0 + 2 * d1 + 3"``.

    Returns:
        AST node tuple.
    """
    tokens = _tokenise(s.strip())
    return _Parser(tokens).parse_expr()


def eval_expr(node: _Node, dims: Sequence[int]) -> int:
    """Evaluate a single AST node given concrete dimension values.

    Companion to :func:`parse_expr` for testing.
    """
    return _eval_node(node, list(dims))
