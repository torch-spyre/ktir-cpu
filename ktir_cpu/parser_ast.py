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
parse_affine_map(s)                        -> AffineMap
parse_affine_set(s)                        -> AffineSet
eval_affine_map(amap, dims)                -> tuple[int, ...]
affine_set_contains(aset, point, symbols)  -> bool
enumerate_affine_set(aset, shape, symbols) -> list[tuple[int, ...]]

``parse_affine_map`` / ``parse_affine_set`` are called by ``parser.py``
when it encounters an ``affine_map<...>`` or ``affine_set<...>`` string.
The evaluation functions are called by the convenience methods on
``AffineMap`` and ``AffineSet`` in ``affine.py``, and directly by
``memory_ops.py``.

Supported scope
---------------
Linear affine expressions:
  - Dimension variables: d0, d1, ... (or arbitrary names via the dim list)
  - Symbol variables:    s0, s1, ... (or arbitrary names via the symbol list)
  - Integer constants
  - Addition (+), subtraction (-), negation (unary -)
  - Multiplication by a constant coefficient (N * dI or dI * N)

Not supported: floordiv, ceildiv, mod.

Affine set enumeration
----------------------
Constraints are interpreted in local (0-based) coordinates within the
access tile.  The caller passes the tile shape as a bounding box and,
for sets with symbol variables, the concrete symbol values.
"""

from __future__ import annotations

import itertools
import re
from typing import List, Optional, Sequence, Tuple, Union

from .affine import AffineMap, AffineSet, BoxSet


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
#   ("max",   node, node)     — pointwise maximum (constructed by sym_max,
#                               not the surface parser; used to express
#                               symbolic ``BoxSet.lo`` after intersect)
#   ("min",   node, node)     — pointwise minimum (constructed by sym_min,
#                               not the surface parser; used to express
#                               symbolic ``BoxSet.hi`` after intersect)
# ---------------------------------------------------------------------------

_Node = tuple


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(%[a-zA-Z_]\w*)'               # named reference %name (group 1)
    r'|([a-zA-Z_]\w*)'               # bare identifier  (group 2)
    r'|(-?\d+)'                      # integer literal, possibly negative (group 3)
    r'|(==|>=|<=|->|[+\-*(),:[\]])'  # operator / punctuation; == before >= (group 4)
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
        # Maps symbol name (e.g. "s0", "n") to its positional index.
        # Set by callers after parsing the optional symbol list in affine sets.
        self.sym_index: dict = {}

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

    def parse_sym_list(self) -> List[str]:
        """Parse optional ``[s0, s1, ...]`` symbol list and return names."""
        if self.peek() != "[":
            return []
        self.consume("[")
        names: List[str] = []
        while self.peek() != "]":
            names.append(self.consume())
            if self.peek() == ",":
                self.consume(",")
        self.consume("]")
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

        # Symbol variable — any identifier that appears in the symbol list,
        # e.g. affine_set<(d0)[s0, n] : (d0 >= 0, -d0 + n - 1 >= 0)>.
        # Symbols are runtime-known constants (unlike dims, which index the
        # iteration space).  sym_index is populated from the parsed [s0, ...]
        # list in parse_affine_set; callers pass concrete values via the
        # ``symbols`` argument to affine_set_contains / enumerate_affine_set.
        # Fallback: if sym_index is empty (standalone parse_expr call), accept
        # canonical sN names by parsing the numeric suffix directly (s0→0, ...).
        if tok in self.sym_index:
            self.consume()
            return ("sym", self.sym_index[tok])
        if re.fullmatch(r's\d+', tok) and not self.sym_index:
            self.consume()
            return ("sym", int(tok[1:]))

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
        """Parse ``(lhs >= rhs, lhs <= rhs, lhs == rhs, ...)`` and return nodes.

        Inequality constraints are normalised to ``lhs - rhs >= 0``:
          - ``lhs >= rhs``  → stored as ``("sub", lhs, rhs)``
          - ``lhs <= rhs``  → stored as ``("sub", rhs, lhs)``

        Equality constraints are stored as a first-class 3-tuple:
          - ``lhs == rhs``  → stored as ``("eq", lhs, rhs)``
        """
        self.consume("(")
        constraints: List[_Node] = []
        while self.peek() != ")":
            lhs = self.parse_expr()
            op = self.consume()  # ">=", "<=", or "=="
            rhs = self.parse_expr()
            if op == ">=":
                node = ("sub", lhs, rhs)
            elif op == "<=":
                node = ("sub", rhs, lhs)
            elif op == "==":
                node = ("eq", lhs, rhs)
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


def parse_affine_set_raw(s: str) -> AffineSet:
    """Parse ``affine_set<...>`` into an :class:`AffineSet` without lowering.

    Unlike :func:`parse_affine_set`, this always returns an ``AffineSet`` —
    no parse-time lowering to :class:`BoxSet`.  Used by tests that
    validate the constraint AST structure, and as a building block for
    :func:`parse_affine_set`.

    The ``affine_set<...>`` wrapper and the ``[s0, ...]`` symbol list are optional.

    Raises:
        ValueError: on any parse error.
    """
    source = s.strip()
    inner = _strip_outer(source, "affine_set")
    colon = inner.index(":")
    dim_part = inner[:colon].strip()
    con_part = inner[colon + 1:].strip()

    # Parse dim list and optional symbol list from dim_part, e.g. "(d0)[s0]"
    dim_tokens = _tokenise(dim_part)
    p1 = _Parser(dim_tokens)
    dim_names = p1.parse_dim_list()
    sym_names = p1.parse_sym_list()

    con_tokens = _tokenise(con_part)
    p2 = _Parser(con_tokens)
    # Share name→index maps so constraint expressions can reference dims and syms.
    p2.dim_index = {name: idx for idx, name in enumerate(dim_names)}
    p2.sym_index = {name: idx for idx, name in enumerate(sym_names)}
    constraints = p2.parse_constraint_list()

    return AffineSet(
        n_dims=len(dim_names),
        n_syms=len(sym_names),
        constraints=tuple(constraints),
        source=source,
    )


def parse_affine_set(s: str):
    """Parse ``affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>``.

    Returns a :class:`BoxSet` when the set is axis-aligned and both lo/hi
    are pinned on every axis; otherwise returns the :class:`AffineSet`
    fallback.  The ``affine_set<...>`` wrapper is optional.  For tests
    that need the raw ``AffineSet`` regardless of lowerability, call
    :func:`parse_affine_set_raw`.

    Symbolic sets (``n_syms > 0``) currently stay on the ``AffineSet``
    branch — see ``BoxSet.try_from_affine_set``'s TODO.

    Raises:
        ValueError: on any parse error.
    """
    aset = parse_affine_set_raw(s)
    box = BoxSet.try_from_affine_set(aset)
    return box if box is not None else aset


# ---------------------------------------------------------------------------
# Evaluation functions (called by AffineMap / AffineSet convenience methods)
# ---------------------------------------------------------------------------

def _eval_node(node: _Node, dims: List[int], syms: Optional[List[int]] = None) -> int:
    """Recursively evaluate an AST node given concrete dimension and symbol values."""
    tag = node[0]
    if tag == "const":
        return node[1]
    if tag == "dim":
        return dims[node[1]]
    if tag == "sym":
        return (syms or [])[node[1]]
    if tag == "add":
        return _eval_node(node[1], dims, syms) + _eval_node(node[2], dims, syms)
    if tag == "sub":
        return _eval_node(node[1], dims, syms) - _eval_node(node[2], dims, syms)
    if tag == "neg":
        return -_eval_node(node[1], dims, syms)
    if tag == "mul":
        return node[1] * _eval_node(node[2], dims, syms)
    if tag == "max":
        return max(_eval_node(node[1], dims, syms), _eval_node(node[2], dims, syms))
    if tag == "min":
        return min(_eval_node(node[1], dims, syms), _eval_node(node[2], dims, syms))
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


def affine_set_contains(aset: AffineSet, point: Sequence[int], symbols: Sequence[int] = ()) -> bool:
    """Return True if *point* satisfies all constraints in *aset*."""
    env = list(point)
    syms = list(symbols)
    return all(
        _eval_node(c[1], env, syms) == _eval_node(c[2], env, syms) if c[0] == "eq"
        else _eval_node(c, env, syms) >= 0
        for c in aset.constraints
    )


def enumerate_affine_set(aset: AffineSet, shape: Tuple[int, ...], symbols: Sequence[int] = ()) -> List[Tuple[int, ...]]:
    """Return all integer points in ``[0, shape)`` satisfying *aset*.

    Args:
        aset:    Parsed AffineSet.
        shape:   Bounding box — one upper bound (exclusive) per dimension.
        symbols: Concrete values for symbol variables s0, s1, ...

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
    return [pt for pt in itertools.product(*ranges) if affine_set_contains(aset, pt, symbols)]


# ---------------------------------------------------------------------------
# Symbolic bound helpers
#
# A ``Bound`` is either a Python ``int`` (concrete leaf) or an AST node
# tuple representing an expression over symbol variables only (no ``dim``
# nodes — bounds in :class:`BoxSet` are pure expressions of ``symbols``).
# Concrete ints stay unwrapped so structural fast paths can identify them
# with ``isinstance(b, int)`` rather than walking the AST.
#
# ``sym_*`` constructors apply the minimum constant folding needed to keep
# ``intersect`` / ``translate`` from accumulating trivially redundant AST
# nodes per partition (concrete fold of two ints; idempotence on the same
# ``("sym", k)`` reference; additive identity).  Deeper canonicalisation
# (commutativity, nested absorption) is intentionally out of scope — it
# would need a structural-equality engine and the realistic candidate
# count per axis is ≤ 2, so deep nesting does not arise.
# ---------------------------------------------------------------------------

Bound = Union[int, tuple]


def eval_bound(b, symbols: Sequence[int]) -> int:
    """Evaluate a :data:`Bound` against concrete *symbols*.

    Concrete ``int`` bounds short-circuit without touching the AST.
    Symbolic bounds delegate to :func:`_eval_node` with an empty ``dims``
    environment — :class:`BoxSet` bounds never reference dimension
    variables by construction.
    """
    if isinstance(b, int):
        return b
    return _eval_node(b, dims=[], syms=list(symbols))


def sym_add(a, b):
    """Build ``a + b`` over :data:`Bound` operands with constant folding.

    Folds when both operands are concrete; absorbs additive identity
    (``a + 0 → a``).  Otherwise constructs an ``("add", ...)`` AST node.
    """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    a_node = ("const", a) if isinstance(a, int) else a
    b_node = ("const", b) if isinstance(b, int) else b
    return ("add", a_node, b_node)


def sym_neg(a):
    """Build ``-a`` over a :data:`Bound` operand with constant folding."""
    if isinstance(a, int):
        return -a
    # Double-negation collapses: -(-x) → x.  Cheap and avoids gratuitous
    # nesting from ``-k`` on already-negative symbolic constants.
    if a[0] == "neg":
        return a[1]
    return ("neg", a)


def sym_max(a, b):
    """Build ``max(a, b)`` over :data:`Bound` operands with MVP folding.

    Folds when both operands are concrete.  Recognises identical
    ``("sym", k)`` references as idempotent (``max(s_k, s_k) → s_k``).
    No commutativity / nested-absorption rewriting — those would require
    structural equality with canonicalisation (out of scope; per-axis
    candidate count is ≤ 2 so deep nesting does not arise in practice).
    """
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    if (
        not isinstance(a, int)
        and not isinstance(b, int)
        and a[0] == "sym" and b[0] == "sym" and a[1] == b[1]
    ):
        return a
    a_node = ("const", a) if isinstance(a, int) else a
    b_node = ("const", b) if isinstance(b, int) else b
    return ("max", a_node, b_node)


def sym_min(a, b):
    """Build ``min(a, b)`` over :data:`Bound` operands with MVP folding.

    Mirror of :func:`sym_max`; see that function for the folding rules.
    """
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    if (
        not isinstance(a, int)
        and not isinstance(b, int)
        and a[0] == "sym" and b[0] == "sym" and a[1] == b[1]
    ):
        return a
    a_node = ("const", a) if isinstance(a, int) else a
    b_node = ("const", b) if isinstance(b, int) else b
    return ("min", a_node, b_node)


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
