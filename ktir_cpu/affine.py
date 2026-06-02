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

Symbolic bounds: ``BoxSet.lo`` / ``hi`` may carry an integer constant
(concrete leaf, fast path) or an AST node (a :data:`Bound` over symbol
variables).  Per-axis operations that depend on a concrete value
(``contains``, ``is_empty``, ``is_full``, ``enumerate``) take a
``symbols`` argument that resolves the symbolic bounds; pure-int boxes
short-circuit on the ``_all_concrete`` flag and ignore ``symbols``.
``try_from_affine_set`` lowers symbolic axis-aligned sets — see its
docstring for the accepted constraint shape.  Geometric operations
(``intersect``, ``translate``) use :func:`parser_ast.sym_max` /
``sym_min`` / ``sym_add`` so concrete-on-concrete cases stay O(ndim)
without constructing AST nodes.

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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, cast

if TYPE_CHECKING:
    # Avoid circular import at runtime; parser_ast imports affine types.
    # ``Bound`` (axis bound type for :class:`BoxSet`) lives in parser_ast
    # next to the AST helpers (``eval_bound``, ``sym_add``, ``sym_max``).
    from .parser_ast import _Node, Bound  # noqa: F401


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
            idx = _match_pure_dim_ref(expr, self.n_dims)
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
            idx = _match_pure_dim_ref(expr, self.n_dims)
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

    Symbolic bounds: ``lo[d]`` / ``hi[d]`` may be an ``int`` (concrete)
    or an AST node tuple over symbol variables (see :data:`Bound` and
    :func:`parser_ast.eval_bound`).  Whether the box is fully concrete
    is cached in :attr:`_all_concrete` at construction so the hot path
    (per-axis ``contains`` / ``is_empty``) skips ``isinstance`` checks
    on every element.
    """
    lo: Tuple["Bound", ...]   # inclusive
    hi: Tuple["Bound", ...]   # exclusive
    # Cached at __post_init__: True iff every entry in lo/hi is a plain int.
    # ``compare=False`` keeps equality / hashing keyed off the bounds alone
    # (the flag is a derived property).  ``init=False`` so callers don't pass it.
    _all_concrete: bool = field(default=False, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.lo) != len(self.hi):
            raise ValueError(
                f"BoxSet: lo/hi length mismatch: lo={self.lo} hi={self.hi}"
            )
        all_concrete = all(isinstance(b, int) for b in self.lo) and all(
            isinstance(b, int) for b in self.hi
        )
        # Frozen dataclass: bypass the descriptor to set the cached flag once.
        object.__setattr__(self, "_all_concrete", all_concrete)

    @property
    def n_dims(self) -> int:
        return len(self.lo)

    @property
    def is_concrete(self) -> bool:
        """True iff every ``lo`` / ``hi`` entry is a Python ``int``.

        Public read accessor for the cached structural fast-path flag.
        Use this from outside the class instead of touching
        ``_all_concrete`` directly.
        """
        return self._all_concrete

    def contains(self, point: Sequence[int], symbols: Sequence[int] = ()) -> bool:
        """True iff ``lo[d] <= point[d] < hi[d]`` for every dim.

        ``symbols`` is required to resolve symbolic bounds; concrete
        boxes ignore it (the ``_all_concrete`` flag short-circuits the
        AST walk).  Passing too few symbols on a symbolic box raises
        ``IndexError`` (from :func:`parser_ast.eval_bound`) — the
        contract matches :meth:`AffineSet.contains`.
        """
        if len(point) != self.n_dims:
            return False
        if self._all_concrete:
            return all(self.lo[d] <= point[d] < self.hi[d] for d in range(self.n_dims))
        from .parser_ast import eval_bound
        return all(
            eval_bound(self.lo[d], symbols) <= point[d] < eval_bound(self.hi[d], symbols)
            for d in range(self.n_dims)
        )

    def enumerate(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        symbols: Sequence[int] = (),
    ) -> List[Tuple[int, ...]]:
        """Return all integer points in the box in row-major order.

        ``shape`` is accepted for signature parity with
        :meth:`AffineSet.enumerate` (which needs an external bounding box
        for its brute-force iteration).  A ``BoxSet`` is self-bounded,
        so ``shape`` only serves as a sanity check: passed values must
        upper-bound ``hi`` componentwise.

        Symbolic boxes are resolved against ``symbols`` by specialising
        once before enumerating; concrete boxes skip that step.
        """
        box = self if self._all_concrete else self.specialize(symbols)
        if shape is not None:
            if len(shape) != box.n_dims:
                raise ValueError(
                    f"BoxSet.enumerate: shape ndim {len(shape)} does not "
                    f"match box ndim {box.n_dims}"
                )
            for d in range(box.n_dims):
                if box.hi[d] > shape[d]:
                    raise ValueError(
                        f"BoxSet.enumerate: hi[{d}]={box.hi[d]} exceeds "
                        f"shape[{d}]={shape[d]} — box is not contained in "
                        f"the nominal bounding box."
                    )
        return list(itertools.product(*(range(box.lo[d], box.hi[d]) for d in range(box.n_dims))))

    def is_empty(self, symbols: Sequence[int] = ()) -> bool:
        """True iff any axis has ``hi[d] <= lo[d]`` (i.e. empty extent).

        On symbolic boxes the per-axis comparison is done after resolving
        the bounds against ``symbols``.
        """
        if self._all_concrete:
            return any(self.hi[d] <= self.lo[d] for d in range(self.n_dims))
        from .parser_ast import eval_bound
        return any(
            eval_bound(self.hi[d], symbols) <= eval_bound(self.lo[d], symbols)
            for d in range(self.n_dims)
        )

    def is_full(
        self, shape: Tuple[int, ...], symbols: Sequence[int] = ()
    ) -> bool:
        """True iff this box equals ``[0, shape)``.

        Matches :meth:`AffineSet.is_full` semantics: a translated box
        ``[x, x + shape)`` returns ``False`` even when its per-axis
        extent matches.  The asymmetry is intentional — callers
        (e.g. ``ktdp.load`` / ``ktdp.store`` normalisation in
        ``ktdp_ops.parse_construct_access_tile``) use a ``True`` here
        as licence to drop ``coordinate_set`` to ``None`` and take the
        contiguous fast path that assumes a zero origin; reporting full
        on a translated box would silently miscompile.
        Symbolic boxes are resolved against ``symbols`` first, since
        AST nodes cannot be compared structurally for runtime equality.
        """
        if len(shape) != self.n_dims:
            return False
        spec = self if self._all_concrete else self.specialize(symbols)
        return all(
            spec.lo[d] == 0 and spec.hi[d] == shape[d]
            for d in range(self.n_dims)
        )

    def lower_bounds(self, symbols: Sequence[int] = ()) -> Tuple[int, ...]:
        """Return ``lo`` — the per-axis minimum coordinate, resolved to ``int``.

        ``symbols`` is accepted for signature parity with :meth:`contains`
        / :meth:`is_empty` / :meth:`enumerate` so call sites can thread a
        single ``symbols`` tuple uniformly through the API.  On a concrete
        box the cached ``_all_concrete`` flag short-circuits and returns
        ``self.lo`` directly (``cast`` narrows the static
        ``Tuple[Bound, ...]`` field type to the dynamic guarantee that
        every entry is ``int``).  On a symbolic box each entry is
        resolved against ``symbols`` so the return type is always a
        tuple of ``int``.
        """
        if self._all_concrete:
            return cast(Tuple[int, ...], self.lo)
        from .parser_ast import eval_bound
        return tuple(eval_bound(b, symbols) for b in self.lo)

    def specialize(self, symbols: Sequence[int]) -> "BoxSet":
        """Return a concrete ``BoxSet`` with all symbolic bounds resolved.

        Concrete boxes return ``self`` unchanged (cached flag check, no
        copy).  Used at the boundary between symbolic-IR-time and
        runtime-resolved values — see ``MemoryOps.distributed_tile_access``.
        """
        if self._all_concrete:
            return self
        from .parser_ast import eval_bound
        return BoxSet(
            lo=tuple(eval_bound(b, symbols) for b in self.lo),
            hi=tuple(eval_bound(b, symbols) for b in self.hi),
        )

    def translate(self, offset: Sequence["Bound"]) -> "BoxSet":
        """Return a new box shifted by *offset* along each axis.

        ``offset`` may carry symbolic entries; ``sym_add`` folds the
        concrete-on-concrete case so a static box translated by a static
        offset stays fully concrete.
        """
        if len(offset) != self.n_dims:
            raise ValueError(
                f"BoxSet.translate: offset dim mismatch: "
                f"offset={tuple(offset)} n_dims={self.n_dims}"
            )
        from .parser_ast import sym_add
        return BoxSet(
            lo=tuple(sym_add(self.lo[d], offset[d]) for d in range(self.n_dims)),
            hi=tuple(sym_add(self.hi[d], offset[d]) for d in range(self.n_dims)),
        )

    def intersect(self, other: "BoxSet") -> "BoxSet":
        """Axis-wise intersection; result may be empty (``is_empty()``).

        Uses ``sym_max`` / ``sym_min`` so concrete-on-concrete intersects
        fold to ints (no AST allocation), while symbolic operands keep
        their symbols in the result for evaluation against ``symbols``
        later.
        """
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
        from .parser_ast import sym_max, sym_min
        return BoxSet(
            lo=tuple(sym_max(self.lo[d], other.lo[d]) for d in range(self.n_dims)),
            hi=tuple(sym_min(self.hi[d], other.hi[d]) for d in range(self.n_dims)),
        )

    @classmethod
    def try_from_affine_set(cls, aset: "AffineSet") -> Optional["BoxSet"]:
        """Lower an axis-aligned :class:`AffineSet` to a ``BoxSet``.

        Returns ``None`` when the set is not representable as an integer
        box.  Lowering succeeds iff every constraint has the form
        ``c * d_i + k(syms) >= 0`` or ``c * d_i + k(syms) == 0`` with
        ``c ∈ {+1, -1}`` (single dim, unit coeff) and every axis is
        pinned on **both** sides.  ``k(syms)`` may be an integer constant
        or a linear combination of symbol variables — in the symbolic case
        the resulting ``lo`` / ``hi`` carry an AST node (see :data:`Bound`)
        instead of a plain ``int``.

        Equality constraints pin both ``lo[i]`` and ``hi[i]`` to the
        solved value (``hi`` is exclusive, so ``hi = pin + 1``).
        Inequality / equality bounds on the same axis are combined with
        ``sym_max`` (lo) / ``sym_min`` (hi).

        Assumes all symbols ``s_i >= 0`` (matches dim-size semantics).
        Constraints with non-``±1`` dim coefficients, dim coefficients
        that themselves carry a symbol, or non-linear symbol products
        cause this function to return ``None`` so the set stays on the
        AffineSet slow path.
        """
        from .parser_ast import sym_add, sym_max, sym_min, sym_neg

        n = aset.n_dims
        n_syms = aset.n_syms
        los: List[Optional["Bound"]] = [None] * n
        his: List[Optional["Bound"]] = [None] * n
        for c in aset.constraints:
            is_eq = (c[0] == "eq")
            expr = ("sub", c[1], c[2]) if is_eq else c
            lin = _constraint_to_linear_syms(expr, n, n_syms)
            if lin is None:
                return None
            dim_coeffs, sym_coeffs, const = lin
            nz = [i for i, k in enumerate(dim_coeffs) if k != 0]
            if len(nz) != 1:
                return None
            i = nz[0]
            k = dim_coeffs[i]
            if abs(k) != 1:
                return None
            # Build k(syms): int constant + sum(sym_coeffs[j] * s_j).
            sym_term = _build_sym_term(sym_coeffs, const)
            if is_eq:
                # k*d_i + k(syms) == 0  →  d_i == pin
                pin = sym_neg(sym_term) if k == 1 else sym_term
                los[i] = pin             if los[i] is None else sym_max(los[i], pin)
                his[i] = sym_add(pin, 1) if his[i] is None else sym_min(his[i], sym_add(pin, 1))
            elif k == 1:
                # d_i + k(syms) >= 0  →  d_i >= -k(syms)
                candidate = sym_neg(sym_term)
                los[i] = candidate if los[i] is None else sym_max(los[i], candidate)
            else:
                # -d_i + k(syms) >= 0  →  d_i <= k(syms)  →  hi exclusive = k(syms) + 1
                candidate = sym_add(sym_term, 1)
                his[i] = candidate if his[i] is None else sym_min(his[i], candidate)

        if any(v is None for v in los) or any(v is None for v in his):
            return None
        # For concrete boxes, detect contradictions early (e.g. d0 >= 5, d0 <= 3).
        # Symbolic boxes may resolve to contradictions at specialize time; callers
        # can detect that via is_empty(symbols=...) after specialising.
        if all(isinstance(lo, int) and isinstance(hi, int) for lo, hi in zip(los, his)):
            if any(los[i] >= his[i] for i in range(n)):  # type: ignore[operator]
                return None
        return cls(lo=tuple(los), hi=tuple(his))  # type: ignore[arg-type]


def _match_pure_dim_ref(node: "_Node", n_dims: int) -> Optional[int]:
    """Match *node* against ``1 * d_i + 0`` and return ``i``, else ``None``.

    A "pure dim ref" is an expression that flattens (via
    :func:`_constraint_to_linear`) to exactly one dim variable with unit
    coefficient and zero constant.  Used by :meth:`AffineMap.is_identity`
    and :meth:`AffineMap.is_permutation` for structural checks that cannot
    be fooled by linear combinations whose evaluation on a specific probe
    happens to coincide with the probe (e.g. ``d0 + d1 - 2`` evaluates to
    ``1`` on probe ``[1, 2]``).

    Because matching goes through ``_constraint_to_linear``,
    algebraically-equivalent forms collapse to the same flattened
    representation: ``d0 + 0`` and ``d0 + d1 - d1`` both flatten to
    ``1 * d0 + 0`` and match identically to bare ``d0``.

    Examples (n_dims=3)::

        d0           -> 0
        d2           -> 2
        d1 + 0       -> 1     # zero constant ok
        d0 + 1       -> None  # non-zero constant
        2 * d0       -> None  # non-unit coefficient
        d0 + d1      -> None  # more than one dim
        -d0          -> None  # coefficient -1, not 1
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
    """Flatten a dim-only constraint AST into ``(coeffs, const)``.

    Wrapper over :func:`_constraint_to_linear_syms` with ``n_syms=0`` —
    any ``sym`` atom in the AST trips the ``j >= n_syms`` guard and
    returns ``None``, preserving the original "reject symbols" contract.
    The constraint represents ``sum(coeffs[i] * d_i) + const >= 0``.
    """
    result = _constraint_to_linear_syms(node, n_dims, n_syms=0)
    if result is None:
        return None
    dim_coeffs, _sym_coeffs, const = result
    return dim_coeffs, const


def _build_sym_term(sym_coeffs: List[int], const: int) -> "Bound":
    """Reassemble a ``Bound`` from ``sum(sym_coeffs[j] * s_j) + const``.

    Returns a plain ``int`` when no symbol contributes (every coefficient
    is zero) — the structural fast path on concrete bounds depends on
    that.  Otherwise returns an AST node tuple suitable for evaluation
    by :func:`parser_ast.eval_bound`.
    """
    from .parser_ast import sym_add, sym_neg

    expr: "Bound" = const
    for j, c in enumerate(sym_coeffs):
        if c == 0:
            continue
        term: "Bound" = ("sym", j)
        if c == -1:
            term = sym_neg(term)
        elif c != 1:
            term = ("mul", c, ("sym", j))
        expr = sym_add(expr, term)
    return expr


def _constraint_to_linear_syms(
    node: "_Node", n_dims: int, n_syms: int
) -> Optional[Tuple[List[int], List[int], int]]:
    """Flatten a parsed constraint AST into ``(dim_coeffs, sym_coeffs, const)``.

    Canonical flattener used by :meth:`BoxSet.try_from_affine_set`; the
    dim-only :func:`_constraint_to_linear` is a thin wrapper over this
    function.  The constraint represents
    ``sum(dim_coeffs[i] * d_i) + sum(sym_coeffs[j] * s_j) + const >= 0``.

    Returns ``None`` if the expression isn't separable into that form
    — e.g. a ``ref`` atom appears, or a sym × dim product is constructed
    at the AST layer (the surface parser cannot produce
    ``("mul", sym, dim)`` since the first ``mul`` operand must be an
    int literal, but the walker still rejects defensively).
    """
    dim_coeffs = [0] * n_dims
    sym_coeffs = [0] * n_syms
    const_box = [0]

    def walk(n: "_Node", sign: int) -> bool:
        tag = n[0]
        if tag == "const":
            const_box[0] += sign * n[1]
            return True
        if tag == "dim":
            dim_coeffs[n[1]] += sign
            return True
        if tag == "sym":
            j = n[1]
            if j >= n_syms:
                return False
            sym_coeffs[j] += sign
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
                dim_coeffs[inner[1]] += sign * coef
                return True
            if inner[0] == "sym":
                j = inner[1]
                if j >= n_syms:
                    return False
                sym_coeffs[j] += sign * coef
                return True
            if inner[0] == "const":
                const_box[0] += sign * coef * inner[1]
                return True
            return False
        # 'ref' or anything else — not a separable linear combination.
        return False

    if not walk(node, 1):
        return None
    return dim_coeffs, sym_coeffs, const_box[0]
