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
Affine map and integer-set value objects.

These are plain data containers.  All parsing and heavy-lifting evaluation
logic lives in ``parser_ast.py``; the convenience methods below simply
delegate there.

Types
-----
AffineMap   — represents affine_map<(d0,...) -> (e0,...)>
AffineSet   — represents affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>
BoxSet      — axis-aligned specialisation of AffineSet; O(ndim) ops.

Relationship between AffineSet and BoxSet
-----------------------------------------
``BoxSet`` is the axis-aligned specialisation of ``AffineSet``.  Every
``BoxSet`` could equivalently be expressed as an ``AffineSet`` with
per-axis inequalities, but the explicit ``(lo, hi)`` form makes
``contains`` / ``enumerate`` / ``intersect`` / ``translate`` /
``lower_bounds`` / ``is_empty`` / ``is_full`` all O(ndim) with no
constraint-AST walk.

They are peer dataclasses under a ``Union``, NOT a class hierarchy:
structural fast paths must be visible at each call site via
``isinstance`` dispatch.  Mixed-type operations (e.g.
``BoxSet.intersect(AffineSet)``) raise ``TypeError`` — there is no
auto-promotion.

Parse-time lowering: ``parse_affine_set`` (in ``parser_ast.py``) lowers
axis-aligned, fully-pinned, unit-coefficient, non-symbolic sets to
``BoxSet`` via :meth:`BoxSet.try_from_affine_set`.  Other sets stay as
``AffineSet``.  ``parse_affine_set_raw`` skips the lowering for tests
that need to inspect the AST directly.

TODO: BoxSet does not yet compose with AffineSet's ``n_syms`` /
``symbols`` parameters (added in PR #42 for dynamic-shape symbolic
bounds).  ``try_from_affine_set`` currently rejects symbolic sets
(``n_syms > 0``) so they stay on the AffineSet branch.  When symbolic
boxes are needed, ``BoxSet.lo``/``hi`` will need to accept symbolic
expressions and ``contains``/``enumerate`` will need to take a
``symbols`` argument.

TODO: ``parse_affine_set_raw`` (in ``parser_ast.py``) exists only because
``parse_affine_set`` now lowers axis-aligned inputs to ``BoxSet`` by
default, and ``test_ast.py`` needs raw ``AffineSet`` access to inspect
``.constraints`` / ``.source`` / ``.n_syms``.  A cleaner shape is to
refactor ``BoxSet.try_from_affine_set`` to parse directly from the
source string (skipping the AST round-trip and ``_constraint_to_linear``
entirely), at which point ``parse_affine_set_raw`` collapses into the
underlying AST parser and can be retired from the public API.  Defer
until after C4 to keep this commit faithful to backup.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    # Avoid circular import at runtime; parser_ast imports nothing from here.
    from .parser_ast import _Node


@dataclass(frozen=True)
class AffineMap:
    """Parsed affine_map<(d0,...) -> (e0,...)>.

    Attributes:
        n_dims:  number of input dimension variables (d0, d1, ...)
        exprs:   tuple of AST nodes, one per output dimension
        source:  original verbatim string (for debugging / round-trip)
    """
    n_dims: int
    exprs: Tuple["_Node", ...]
    source: str

    def eval(self, dims: Sequence[int]) -> Tuple[int, ...]:
        """Return the output tuple for the given dimension values.

        Delegates to ``parser_ast.eval_affine_map``.
        """
        from .parser_ast import eval_affine_map
        return eval_affine_map(self, dims)

    def is_identity(self) -> bool:
        """Return True if this map is the identity: output[i] == d_i for every i.

        Used at parse time to detect trivial coordinate-order maps.  When
        ``access_tile_order`` is an identity map it has no effect on which
        memory element lands at each output position, so we set
        ``coordinate_order`` to ``None``.  This allows load/store to skip
        the per-coord ``cso.eval()`` calls and, combined with a full
        ``coordinate_set``, enables the contiguous fast path entirely.

        Implemented structurally: each output expression must flatten to
        ``1 * d_i + 0`` with output position ``i`` matching the dim index.
        A probe-based ``eval(probe) == probe`` check would accept maps
        like ``(d0, d1) -> (d1 - 1, d0 + 1)`` (probe ``[1,2]`` → ``(1,2)``),
        which are not identity.
        """
        if len(self.exprs) != self.n_dims:
            return False
        for i, expr in enumerate(self.exprs):
            idx = _expr_as_single_dim(expr, self.n_dims)
            if idx != i:
                return False
        return True

    def is_permutation(self) -> bool:
        """Return True if this map permutes its input dimensions.

        A permutation map is square (output count equals input count) and
        each output expression is exactly one dim variable, with every dim
        index appearing exactly once.  Accepts coordinate permutations
        like ``(d0, d1, d2) -> (d2, d0, d1)``; rejects shears, scalings,
        constant offsets, and many-to-one collapses.

        Used by ops whose semantics require iteration over the input space
        in a permuted order (e.g. ``variables_space_order`` on indirect
        access tiles): the implementation sorts enumerated points by the
        map's image, which is well-defined only when the image is a
        permutation of the original points.

        Implemented structurally on the parsed AST.  A probe-based
        ``sorted(eval(probe)) == probe`` check would accept linear
        combinations such as ``(d0, d1) -> (d0 + d1 - 2, d0 + d1 - 1)``
        (probe ``[1,2]`` → ``(1,2)``), which are not coordinate
        permutations.
        """
        if len(self.exprs) != self.n_dims:
            return False
        seen = set()
        for expr in self.exprs:
            idx = _expr_as_single_dim(expr, self.n_dims)
            if idx is None or idx in seen:
                return False
            seen.add(idx)
        return True


@dataclass(frozen=True)
class AffineSet:
    """Parsed affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>.

    Attributes:
        n_dims:       number of dimension variables
        n_syms:       number of symbol variables (s0, s1, ...)
        constraints:  tuple of AST nodes; each node is the LHS of ``expr >= 0``
        source:       original verbatim string (for debugging / round-trip)
    """
    n_dims: int
    constraints: Tuple["_Node", ...]
    source: str
    n_syms: int = 0

    def contains(self, point: Sequence[int], symbols: Sequence[int] = ()) -> bool:
        """Return True if *point* satisfies all constraints.

        Delegates to ``parser_ast.affine_set_contains``.
        """
        from .parser_ast import affine_set_contains
        return affine_set_contains(self, point, symbols)

    def enumerate(self, shape: Tuple[int, ...], symbols: Sequence[int] = ()) -> List[Tuple[int, ...]]:
        """Return all integer points in ``[0, shape)`` satisfying all constraints.

        Delegates to ``parser_ast.enumerate_affine_set``.
        """
        from .parser_ast import enumerate_affine_set
        return enumerate_affine_set(self, shape, symbols)

    def is_full(self, shape: Tuple[int, ...]) -> bool:
        """Return True if this set covers every coordinate in *shape*.

        Called once at parse time to detect trivial coordinate sets — i.e.
        those that enumerate the full rectangular tile in row-major order.
        When a set is full, ``coordinate_set`` is set to ``None`` so
        that load/store can take the contiguous fast path instead of building
        and iterating a coordinate list on every execution.  Without this,
        even plain rectangular tiles pay the cost of enumerating all coords
        on every load/store (e.g. 46k times for a 32-core layernorm).

        Uses a vertex check: an affine set is convex, so it covers [0, shape)
        iff it contains all 2^n_dims corners of that box.  This is O(2^n_dims)
        constraint evaluations instead of O(∏ shape).
        """
        if len(shape) != self.n_dims:
            return False

        import itertools as _it
        corners = _it.product(*((0, n - 1) for n in shape))
        return all(self.contains(pt) for pt in corners)


@dataclass(frozen=True)
class BoxSet:
    """Axis-aligned integer hyperrectangle: ``{p : lo[d] <= p[d] < hi[d]}``.

    The axis-aligned specialisation of :class:`AffineSet`: every ``BoxSet``
    could equivalently be written as an ``AffineSet`` with per-axis
    inequalities, but carrying the ``(lo, hi)`` structure explicitly makes
    every operation (``contains``, ``enumerate``, ``is_empty``, ``is_full``,
    ``lower_bounds``, ``translate``, ``intersect``) O(ndim) with no
    constraint-AST walk.  Used for partition extents (``B_i``), access tile
    sets (``A``), and their intersections (``C_i``) in
    ``distributed_tile_access``; the parser lowers axis-aligned affine sets
    to this form at parse time (see ``try_from_affine_set``).

    ``BoxSet`` and ``AffineSet`` are peer dataclasses under a ``Union``
    rather than parent/child classes — structural fast paths must be
    visible at each call site via ``isinstance`` dispatch, not hidden
    behind polymorphism.  Mixed-type operations — e.g.
    ``BoxSet.intersect(aset: AffineSet)`` — raise ``TypeError``.
    """
    lo: Tuple[int, ...]   # inclusive
    hi: Tuple[int, ...]   # exclusive

    def __post_init__(self) -> None:
        if len(self.lo) != len(self.hi):
            raise ValueError(
                f"BoxSet: lo/hi length mismatch: lo={self.lo} hi={self.hi}"
            )

    @property
    def n_dims(self) -> int:
        return len(self.lo)

    def contains(self, point: Sequence[int]) -> bool:
        """True iff ``lo[d] <= point[d] < hi[d]`` for every dim."""
        if len(point) != self.n_dims:
            return False
        return all(self.lo[d] <= point[d] < self.hi[d] for d in range(self.n_dims))

    def enumerate(self, shape: Optional[Tuple[int, ...]] = None) -> List[Tuple[int, ...]]:
        """Return all integer points in the box in row-major order.

        ``shape`` is accepted for signature parity with
        :meth:`AffineSet.enumerate` (which needs an external bounding box
        for its brute-force iteration).  A ``BoxSet`` is self-bounded,
        so ``shape`` only serves as a sanity check: passed values must
        upper-bound ``hi`` componentwise.
        """
        if shape is not None:
            if len(shape) != self.n_dims:
                raise ValueError(
                    f"BoxSet.enumerate: shape ndim {len(shape)} does not "
                    f"match box ndim {self.n_dims}"
                )
            for d in range(self.n_dims):
                if self.hi[d] > shape[d]:
                    raise ValueError(
                        f"BoxSet.enumerate: hi[{d}]={self.hi[d]} exceeds "
                        f"shape[{d}]={shape[d]} — box is not contained in "
                        f"the nominal bounding box."
                    )
        return list(itertools.product(*(range(self.lo[d], self.hi[d]) for d in range(self.n_dims))))

    def is_empty(self) -> bool:
        """True iff any axis has ``hi[d] <= lo[d]`` (i.e. empty extent)."""
        return any(self.hi[d] <= self.lo[d] for d in range(self.n_dims))

    def is_full(self, shape: Tuple[int, ...]) -> bool:
        """True iff this box equals ``[0, shape)``."""
        if len(shape) != self.n_dims:
            return False
        return self.lo == (0,) * self.n_dims and tuple(self.hi) == tuple(shape)

    def lower_bounds(self) -> Tuple[int, ...]:
        """Return ``lo`` — the per-axis minimum coordinate, O(1)."""
        return self.lo

    def translate(self, offset: Sequence[int]) -> "BoxSet":
        """Return a new box shifted by *offset* along each axis."""
        if len(offset) != self.n_dims:
            raise ValueError(
                f"BoxSet.translate: offset dim mismatch: "
                f"offset={tuple(offset)} n_dims={self.n_dims}"
            )
        return BoxSet(
            lo=tuple(self.lo[d] + offset[d] for d in range(self.n_dims)),
            hi=tuple(self.hi[d] + offset[d] for d in range(self.n_dims)),
        )

    def intersect(self, other: "BoxSet") -> "BoxSet":
        """Axis-wise intersection; result may be empty (``is_empty()``)."""
        if not isinstance(other, BoxSet):
            raise TypeError(
                f"BoxSet.intersect: mixed-type intersection not supported "
                f"(other is {type(other).__name__}).  Box and AffineSet are "
                f"structural peers, not interchangeable."
            )
        if other.n_dims != self.n_dims:
            raise ValueError(
                f"BoxSet.intersect: n_dims mismatch {self.n_dims} vs {other.n_dims}"
            )
        return BoxSet(
            lo=tuple(max(self.lo[d], other.lo[d]) for d in range(self.n_dims)),
            hi=tuple(min(self.hi[d], other.hi[d]) for d in range(self.n_dims)),
        )

    @classmethod
    def try_from_affine_set(cls, aset: "AffineSet") -> Optional["BoxSet"]:
        """Lower an axis-aligned :class:`AffineSet` to a ``BoxSet``.

        Returns ``None`` when the set is not representable as an integer
        box.  Lowering succeeds iff every constraint has the form
        ``c * d_i + k >= 0`` with ``c ∈ {+1, -1}`` (single dim, unit coeff)
        and every axis is pinned on **both** sides (at least one ``+d_i``
        and one ``-d_i`` constraint).

        TODO: symbolic sets (``aset.n_syms > 0``, added by PR #42 for
        dynamic shapes) are rejected here — they stay on the AffineSet
        branch.  Lifting this would require BoxSet to carry symbolic
        bounds and a ``symbols`` argument on contains/enumerate to
        compose with AffineSet's signature.  ``getattr`` keeps this
        correct on older AffineSet objects without ``n_syms``.
        """
        if getattr(aset, "n_syms", 0) != 0:
            return None
        n = aset.n_dims
        los: List[Optional[int]] = [None] * n
        his: List[Optional[int]] = [None] * n
        for c in aset.constraints:
            lin = _constraint_to_linear(c, n)
            if lin is None:
                return None
            coeffs, const = lin
            nz = [i for i, k in enumerate(coeffs) if k != 0]
            if len(nz) != 1:
                return None
            i = nz[0]
            k = coeffs[i]
            if k == 1:
                # d_i + const >= 0  →  d_i >= -const
                candidate = -const
                los[i] = candidate if los[i] is None else max(los[i], candidate)
            elif k == -1:
                # -d_i + const >= 0  →  d_i <= const  →  hi = const + 1
                candidate = const + 1
                his[i] = candidate if his[i] is None else min(his[i], candidate)
            else:
                return None
        if any(v is None for v in los) or any(v is None for v in his):
            return None
        return cls(lo=tuple(los), hi=tuple(his))  # type: ignore[arg-type]


def _expr_as_single_dim(node: "_Node", n_dims: int) -> Optional[int]:
    """Return ``i`` if *node* is structurally equivalent to ``1 * d_i + 0``.

    Returns ``None`` for any expression that flattens to a linear form
    with a non-zero constant, a non-unit coefficient, or coefficients on
    more than one dim variable.  Used by :meth:`AffineMap.is_identity`
    and :meth:`AffineMap.is_permutation` for structural checks that
    cannot be fooled by linear combinations whose evaluation on a
    specific probe happens to coincide with the probe (e.g.
    ``d0 + d1 - 2`` evaluates to ``1`` on probe ``[1, 2]``).
    """
    lin = _constraint_to_linear(node, n_dims)
    if lin is None:
        return None
    coeffs, const = lin
    if const != 0:
        return None
    nz = [i for i, k in enumerate(coeffs) if k != 0]
    if len(nz) != 1 or coeffs[nz[0]] != 1:
        return None
    return nz[0]


def _constraint_to_linear(node: "_Node", n_dims: int) -> Optional[Tuple[List[int], int]]:
    """Flatten a parsed constraint AST into ``(coeffs, const)``.

    Returns ``None`` if the expression isn't a pure linear combination of
    dimension variables and integer constants (e.g. contains ``sym`` or
    ``ref`` atoms).  The constraint represents
    ``sum(coeffs[i] * d_i) + const >= 0``.
    """
    coeffs = [0] * n_dims
    const_box = [0]

    def walk(n: "_Node", sign: int) -> bool:
        tag = n[0]
        if tag == "const":
            const_box[0] += sign * n[1]
            return True
        if tag == "dim":
            coeffs[n[1]] += sign
            return True
        if tag == "add":
            return walk(n[1], sign) and walk(n[2], sign)
        if tag == "sub":
            return walk(n[1], sign) and walk(n[2], -sign)
        if tag == "neg":
            return walk(n[1], -sign)
        if tag == "mul":
            coef = n[1]
            inner = n[2]
            if inner[0] == "dim":
                coeffs[inner[1]] += sign * coef
                return True
            if inner[0] == "const":
                const_box[0] += sign * coef * inner[1]
                return True
            return False
        # 'sym', 'ref', or anything else — not a linear combination of dims.
        return False

    if not walk(node, 1):
        return None
    return coeffs, const_box[0]
