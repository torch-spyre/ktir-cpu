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

"""Tests for affine.py — AffineMap, AffineSet, and BoxSet value objects.

These tests verify that the convenience methods on AffineMap and AffineSet
correctly delegate to parser_ast.py.  They are intentionally thin — the heavy
evaluation logic is tested in test_ast.py.

Note on parse-time lowering: axis-aligned sets (e.g. box constraints) are
lowered to BoxSet at parse time.  Tests that specifically exercise the
AffineSet branch use non-axis-aligned sets like ``d1 - d0 >= 0``.
"""

import pytest

from ktir_cpu.affine import AffineSet, BoxSet
from ktir_cpu.parser_ast import parse_affine_map, parse_affine_set


class TestAffineMapObject:

    def test_eval_delegates(self):
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        assert m.eval([7]) == (7,)

    def test_eval_non_identity(self):
        m = parse_affine_map("affine_map<(i) -> (i, 0)>")
        assert m.eval([3]) == (3, 0)

    def test_eval_wrong_dims_raises(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        with pytest.raises(ValueError):
            m.eval([1])

    def test_source_field(self):
        s = "affine_map<(d0) -> (d0)>"
        m = parse_affine_map(s)
        assert m.source == s

    def test_frozen(self):
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        with pytest.raises((AttributeError, TypeError)):
            m.n_dims = 99  # type: ignore[misc]


class TestAffineMapIsPermutation:
    """is_permutation() detects coordinate-permutation affine maps.

    Used as a precondition guard for ops whose semantics require sorting
    enumerated points by the map's image (e.g. ``variables_space_order``
    on indirect access tiles).  Non-permutation maps would produce
    duplicate sort keys with undefined relative output positions.
    """

    def test_1d_identity(self):
        """1-D edge: ``(d0) -> (d0)`` is the trivial permutation."""
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        assert m.is_permutation()

    def test_identity(self):
        m = parse_affine_map("affine_map<(d0, d1, d2) -> (d0, d1, d2)>")
        assert m.is_permutation()

    def test_2d_swap(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert m.is_permutation()

    def test_3d_cycle(self):
        m = parse_affine_map("affine_map<(d0, d1, d2) -> (d2, d0, d1)>")
        assert m.is_permutation()

    def test_shear_rejected(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0 + d1, d1)>")
        assert not m.is_permutation()

    def test_many_to_one_rejected(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d0)>")
        assert not m.is_permutation()

    def test_non_square_rejected(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0)>")
        assert not m.is_permutation()

    def test_linear_combination_rejected(self):
        """Regression: probe-based ``sorted(eval(probe)) == probe`` accepts
        ``(d0+d1-2, d0+d1-1)`` because probe ``[1,2]`` happens to give
        ``(1,2)``.  Structural check rejects it: neither output is a
        single dim variable.  pt=(3,1) → (2,3), confirming non-permutation
        behaviour at runtime.
        """
        m = parse_affine_map("affine_map<(d0, d1) -> (d0 + d1 - 2, d0 + d1 - 1)>")
        assert not m.is_permutation()
        assert m.eval([3, 1]) == (2, 3)  # would shift the box if accepted

    def test_constant_offset_rejected(self):
        """``(d1-1, d0+1)`` shifts both coordinates; not a coordinate
        permutation.  Probe ``[1,2]`` evaluates to ``(1,2)`` and would slip
        past a probe-based check.
        """
        m = parse_affine_map("affine_map<(d0, d1) -> (d1 - 1, d0 + 1)>")
        assert not m.is_permutation()

    def test_trivial_wrappers_accepted(self):
        """Permutations expressed with redundant ``+0`` / ``1*`` wrappers
        still accepted — the structural check flattens to linear form
        before inspecting coefficients.
        """
        m = parse_affine_map("affine_map<(d0, d1) -> (1 * d1 + 0, d0)>")
        assert m.is_permutation()


class TestAffineMapIsIdentity:
    """is_identity() detects the strict coordinate-identity map.

    Regression for a probe-based check that accepts linear combinations
    whose evaluation on ``[1, 2, ..., n]`` happens to coincide with the
    probe (e.g. ``(d1-1, d0+1)`` on probe ``[1,2]`` returns ``(1,2)``).
    The structural check requires each output ``i`` to be exactly
    ``d_i``.
    """

    def test_identity_accepted(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        assert m.is_identity()

    def test_swap_rejected(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert not m.is_identity()

    def test_constant_offset_rejected(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d1 - 1, d0 + 1)>")
        assert not m.is_identity()

    def test_trivial_wrappers_accepted(self):
        """``d0 + 0`` and ``1 * d0`` flatten to ``d0`` and remain identity."""
        m = parse_affine_map("affine_map<(d0, d1) -> (d0 + 0, 1 * d1)>")
        assert m.is_identity()

    def test_identity_through_cancellation(self):
        """Subtractive cancellation must collapse through flatten.

        ``d0 + d1 - d1`` flattens to ``1 * d0 + 0`` and ``d1 + d0 - d0``
        flattens to ``1 * d1 + 0``, so the map is identity.  Pins the
        flatten-form contract: a future syntactic-only matcher would
        reject these forms and silently regress the structural check.
        """
        m = parse_affine_map("affine_map<(d0, d1) -> (d0 + d1 - d1, d1 + d0 - d0)>")
        assert m.is_identity()


class TestAffineSetObject:
    """AffineSet behaviour on sets that are *not* lowerable to BoxSet."""

    def test_contains_delegates(self):
        # Non-axis-aligned: d1 >= d0 and the box bounds.  Parse-time lowering
        # rejects this and keeps it as AffineSet.
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0, d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
        assert isinstance(s, AffineSet)
        assert s.contains([1, 2])
        assert not s.contains([2, 1])

    def test_enumerate_delegates(self):
        # Upper-triangular 2x2: points satisfying d1 >= d0 in [0,2)^2.
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert isinstance(s, AffineSet)
        assert s.enumerate((2, 2)) == [(0, 0), (0, 1), (1, 1)]

    def test_enumerate_wrong_shape_raises(self):
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        with pytest.raises(ValueError):
            s.enumerate((4,))

    def test_source_field(self):
        src = "affine_set<(d0, d1) : (d1 - d0 >= 0)>"
        s = parse_affine_set(src)
        assert isinstance(s, AffineSet)
        assert s.source == src

    def test_frozen(self):
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        with pytest.raises((AttributeError, TypeError)):
            s.n_dims = 99  # type: ignore[misc]

    def test_is_full_false(self):
        """Upper-triangular set (d1 >= d0) is not full — corner (3,0) is excluded."""
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert isinstance(s, AffineSet)
        assert not s.is_full((4, 4))

    def test_is_full_wrong_ndim(self):
        """Shape ndim != set n_dims always returns False."""
        # Use a non-lowerable set to stay on the AffineSet branch.
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert isinstance(s, AffineSet)
        assert not s.is_full((2,))


class TestBoxSetBasics:
    """Direct construction and core operations on BoxSet."""

    def test_contains(self):
        b = BoxSet(lo=(0, 0), hi=(2, 3))
        assert b.contains((0, 0))
        assert b.contains((1, 2))
        assert not b.contains((2, 0))  # hi is exclusive
        assert not b.contains((0, 3))
        assert not b.contains((-1, 0))

    def test_contains_wrong_ndim(self):
        b = BoxSet(lo=(0,), hi=(3,))
        assert not b.contains((0, 0))

    def test_enumerate_no_shape(self):
        b = BoxSet(lo=(1, 2), hi=(3, 4))
        assert b.enumerate() == [(1, 2), (1, 3), (2, 2), (2, 3)]

    def test_enumerate_shape_matches_hi(self):
        """Passing shape == hi is allowed (signature parity with AffineSet)."""
        b = BoxSet(lo=(0, 0), hi=(2, 2))
        assert b.enumerate((2, 2)) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_enumerate_shape_upper_bounds_hi(self):
        """Shape may be a strict upper bound — box stays self-bounded."""
        b = BoxSet(lo=(0, 0), hi=(2, 2))
        # 4×4 nominal bounding box; box is a 2×2 sub-region.  The call site
        # treats shape as the enclosing tile; the box only enumerates itself.
        assert b.enumerate((4, 4)) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_enumerate_shape_below_hi_raises(self):
        """If the box extends past shape, the call site has an invariant bug."""
        b = BoxSet(lo=(0, 0), hi=(3, 3))
        with pytest.raises(ValueError):
            b.enumerate((2, 4))

    def test_enumerate_shape_ndim_mismatch_raises(self):
        b = BoxSet(lo=(0, 0), hi=(2, 2))
        with pytest.raises(ValueError):
            b.enumerate((2,))

    def test_is_empty(self):
        assert not BoxSet(lo=(0, 0), hi=(2, 2)).is_empty()
        assert BoxSet(lo=(2, 0), hi=(2, 2)).is_empty()   # zero-width axis
        assert BoxSet(lo=(3, 0), hi=(2, 2)).is_empty()   # hi < lo

    def test_is_full(self):
        assert BoxSet(lo=(0, 0), hi=(2, 3)).is_full((2, 3))
        assert not BoxSet(lo=(0, 0), hi=(2, 3)).is_full((2, 4))
        assert not BoxSet(lo=(1, 0), hi=(2, 3)).is_full((2, 3))

    def test_is_full_wrong_ndim(self):
        assert not BoxSet(lo=(0,), hi=(3,)).is_full((3, 3))

    def test_lower_bounds(self):
        assert BoxSet(lo=(2, 5), hi=(4, 7)).lower_bounds() == (2, 5)

    def test_translate(self):
        b = BoxSet(lo=(0, 0), hi=(2, 2))
        t = b.translate((10, 20))
        assert t == BoxSet(lo=(10, 20), hi=(12, 22))

    def test_translate_wrong_ndim(self):
        with pytest.raises(ValueError):
            BoxSet(lo=(0, 0), hi=(2, 2)).translate((1,))

    def test_intersect_disjoint_is_empty(self):
        a = BoxSet(lo=(0, 0), hi=(2, 2))
        b = BoxSet(lo=(2, 0), hi=(4, 2))
        c = a.intersect(b)
        assert c.is_empty()

    def test_intersect_overlap(self):
        a = BoxSet(lo=(0, 0), hi=(3, 3))
        b = BoxSet(lo=(1, 1), hi=(5, 5))
        c = a.intersect(b)
        assert c == BoxSet(lo=(1, 1), hi=(3, 3))

    def test_intersect_ndim_mismatch(self):
        with pytest.raises(ValueError):
            BoxSet(lo=(0,), hi=(2,)).intersect(BoxSet(lo=(0, 0), hi=(2, 2)))

    def test_intersect_mixed_type_raises(self):
        """BoxSet.intersect(AffineSet) is rejected — no auto-promotion."""
        box = BoxSet(lo=(0, 0), hi=(2, 2))
        aset = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert isinstance(aset, AffineSet)
        with pytest.raises(TypeError):
            box.intersect(aset)  # type: ignore[arg-type]

    def test_frozen(self):
        b = BoxSet(lo=(0, 0), hi=(2, 2))
        with pytest.raises((AttributeError, TypeError)):
            b.lo = (1, 1)  # type: ignore[misc]

    def test_construction_ndim_mismatch(self):
        with pytest.raises(ValueError):
            BoxSet(lo=(0, 0), hi=(2,))


class TestTryFromAffineSet:
    """Parse-time lowering from AffineSet to BoxSet."""

    def _parse_aset(self, src: str) -> AffineSet:
        """Build an AffineSet bypassing parse_affine_set's own lowering hook."""
        # We call the parser but it may return BoxSet; for reject tests we
        # want to drive try_from_affine_set directly with an AffineSet AST.
        # Constructing via parser internals is fine because these tests sit
        # next to the parser.
        from ktir_cpu.parser_ast import (
            _Parser, _strip_outer, _tokenise,
        )
        source = src.strip()
        inner = _strip_outer(source, "affine_set")
        colon = inner.index(":")
        dim_part = inner[:colon].strip()
        con_part = inner[colon + 1:].strip()
        p1 = _Parser(_tokenise(dim_part))
        dim_names = p1.parse_dim_list()
        p2 = _Parser(_tokenise(con_part))
        p2.dim_index = {name: idx for idx, name in enumerate(dim_names)}
        constraints = p2.parse_constraint_list()
        return AffineSet(n_dims=len(dim_names), constraints=tuple(constraints), source=source)

    def test_accept_1d_range(self):
        aset = self._parse_aset("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        box = BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(0,), hi=(4,))

    def test_accept_2d_box(self):
        aset = self._parse_aset(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>"
        )
        box = BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(0, 0), hi=(2, 4))

    def test_accept_nonzero_origin(self):
        # d0 >= 2, d0 <= 5  →  lo=2, hi=6
        aset = self._parse_aset("affine_set<(d0) : (d0 - 2 >= 0, -d0 + 5 >= 0)>")
        box = BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(2,), hi=(6,))

    def test_accept_tightest_bounds(self):
        # Two lo constraints (d0 >= 0, d0 >= 2) and two hi (d0 <= 5, d0 <= 3):
        # lo = max(0, 2) = 2, hi = min(6, 4) = 4.
        aset = self._parse_aset(
            "affine_set<(d0) : (d0 >= 0, d0 - 2 >= 0, -d0 + 5 >= 0, -d0 + 3 >= 0)>"
        )
        box = BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(2,), hi=(4,))

    def test_reject_not_axis_aligned(self):
        """Upper-triangular d1 >= d0: two dims in one constraint."""
        aset = self._parse_aset("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_missing_upper_bound(self):
        """d0 >= 0 alone pins lo but not hi — reject."""
        aset = self._parse_aset("affine_set<(d0) : (d0 >= 0)>")
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_missing_lower_bound(self):
        aset = self._parse_aset("affine_set<(d0) : (-d0 + 3 >= 0)>")
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_nonunit_coefficient(self):
        """2 * d0 >= 0 — unit coefficients only."""
        aset = self._parse_aset("affine_set<(d0) : (2 * d0 >= 0, -d0 + 3 >= 0)>")
        assert BoxSet.try_from_affine_set(aset) is None

    def test_accept_eq_and_range(self):
        """d0 == 2, 1 <= d1 <= 3  →  BoxSet(lo=(2, 1), hi=(3, 4))."""
        aset = self._parse_aset(
            "affine_set<(d0, d1) : (d0 - 2 == 0, d1 - 1 >= 0, -d1 + 3 >= 0)>"
        )
        box = BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(2, 1), hi=(3, 4))

    def test_reject_one_axis_unpinned(self):
        """2D set where d1 has no upper bound."""
        aset = self._parse_aset(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>"
        )
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_axis_with_no_constraints(self):
        """2D set where d1 has no constraints at all — both lo and hi missing."""
        aset = self._parse_aset(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0)>"
        )
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_eq_pins_one_axis_other_unconstrained(self):
        """2D set where d0 is pinned by eq but d1 has no constraints."""
        aset = self._parse_aset(
            "affine_set<(d0, d1) : (d0 == 0)>"
        )
        assert BoxSet.try_from_affine_set(aset) is None


class TestParseAffineSetLowering:
    """End-to-end: parse_affine_set returns BoxSet for axis-aligned sets."""

    def test_axis_aligned_becomes_box(self):
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>"
        )
        assert isinstance(s, BoxSet)
        assert s == BoxSet(lo=(0, 0), hi=(4, 4))

    def test_non_box_stays_affine_set(self):
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert isinstance(s, AffineSet)


# ===========================================================================
# Symbolic BoxSet — lowering, ops, and specialize round-trip
# ===========================================================================

class TestSymbolicBoxSet:
    """Symbolic ``BoxSet`` lifted from ``n_syms > 0`` AffineSets.

    Validates the full chain: parse-time lowering preserves the symbolic
    bound, per-axis ops accept ``symbols``, ``specialize`` yields a
    fully-concrete box, and the symbol-resolved set agrees pointwise
    with the AffineSet slow path on the same input.
    """

    def test_lowering_preserves_bounds_via_parse(self):
        # ``-d0 + s0 - 1 >= 0``  →  d0 < s0  →  hi[0] is symbolic in s0.
        s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>")
        assert isinstance(s, BoxSet)
        # Concrete lo on the same axis stays a plain int (no AST allocation).
        assert s.lo == (0,)
        assert not s._all_concrete

    def test_specialize_resolves_to_concrete_box(self):
        s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>")
        for n in (1, 16, 1024):
            spec = s.specialize([n])
            assert spec._all_concrete
            assert spec == BoxSet(lo=(0,), hi=(n,))

    def test_query_methods_accept_symbols(self):
        # Cross-check BoxSet symbolic answers against the equivalent
        # slow-path AffineSet so the expectation comes from the IR
        # rather than a hand-coded constant.
        from ktir_cpu.parser_ast import parse_affine_set_raw
        src = "affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>"
        box = parse_affine_set(src)
        aset = parse_affine_set_raw(src)
        assert isinstance(box, BoxSet)
        for n in (1, 8):
            for pt in [(0,), (n - 1,), (n,), (-1,)]:
                assert box.contains(pt, symbols=[n]) == aset.contains(pt, symbols=[n])
            assert box.is_empty(symbols=[n]) is False
            assert box.is_full((n,), symbols=[n]) is True
            assert box.enumerate((n,), symbols=[n]) == [(i,) for i in range(n)]
        # Empty extent: s0 = 0 collapses hi to lo.
        assert box.is_empty(symbols=[0]) is True

    def test_intersect_specialized_then_concrete(self):
        # Symbolic lo on d0, concrete elsewhere.  After specialize, the
        # axis-wise intersect should fall to plain ints.
        sym = parse_affine_set(
            "affine_set<(d0)[s0] : (d0 - s0 >= 0, -d0 + 1023 >= 0)>"
        )
        concrete = BoxSet(lo=(0,), hi=(8,))
        spec = sym.specialize([3])  # lo=(3,), hi=(1024,)
        out = spec.intersect(concrete)
        assert out == BoxSet(lo=(3,), hi=(8,))
        assert out._all_concrete

    def test_translate_concrete_offset_preserves_symbols(self):
        # Translating a symbolic box by a concrete offset retains the
        # AST shape on the symbolic side; concrete sides fold.
        s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>")
        shifted = s.translate([5])
        # Concrete side folds to a plain int; symbolic side stays as an
        # AST tuple (not ``int``) and still resolves against ``symbols``
        # after translation.
        assert eval_bound_eq(shifted.lo[0], (), 5)
        assert isinstance(shifted.hi[0], tuple), (
            f"hi[0] should remain an AST tuple after translate, "
            f"got {type(shifted.hi[0]).__name__}: {shifted.hi[0]!r}"
        )
        for n in (8, 64):
            # Per-symbol resolution: hi specialises to 5 + n.
            assert eval_bound_eq(shifted.hi[0], (n,), 5 + n)
            assert shifted.specialize([n]) == BoxSet(lo=(5,), hi=(5 + n,))

    def test_reject_multi_dim_with_symbol(self):
        # Symbol mixed with two dims in one constraint — not separable
        # into a single-axis bound, even though every term is linear.
        from ktir_cpu.parser_ast import parse_affine_set_raw
        aset = parse_affine_set_raw(
            "affine_set<(d0, d1)[s0] : (d0 + d1 - s0 >= 0, -d0 + 3 >= 0, -d1 + 3 >= 0)>"
        )
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_nonunit_coefficient_with_symbol(self):
        # ``2 * d0 + s0 >= 0`` — non-±1 dim coefficient on the symbolic
        # path mirrors the all-concrete reject (guard symmetry).
        from ktir_cpu.parser_ast import parse_affine_set_raw
        aset = parse_affine_set_raw(
            "affine_set<(d0)[s0] : (2 * d0 + s0 >= 0, -d0 + 3 >= 0)>"
        )
        assert BoxSet.try_from_affine_set(aset) is None

    def test_negative_symbol_coefficient_in_bound(self):
        # ``d0 - s0 >= 0``  →  d0 >= s0  →  lo[0] depends on +s0; the
        # symmetric upper-bound case ``-d0 + s0 - 1 >= 0`` already
        # exercises the +s0 path.  This case forces ``_build_sym_term``
        # to emit a negated symbol term in lo[0] (`("neg", ("sym", 0))`
        # via ``sym_neg``) instead of a bare ``("sym", 0)`` reference.
        s = parse_affine_set(
            "affine_set<(d0)[s0] : (d0 - s0 >= 0, -d0 + 1023 >= 0)>"
        )
        assert isinstance(s, BoxSet)
        assert s.hi == (1024,)                     # concrete fold
        assert isinstance(s.lo[0], tuple)          # symbolic AST retained
        # Cross-check against the AffineSet slow path on equivalent input.
        from ktir_cpu.parser_ast import parse_affine_set_raw
        aset = parse_affine_set_raw(
            "affine_set<(d0)[s0] : (d0 - s0 >= 0, -d0 + 1023 >= 0)>"
        )
        for n in (0, 3, 1023):
            spec = s.specialize([n])
            assert spec == BoxSet(lo=(n,), hi=(1024,))
            for pt in [(n - 1,), (n,), (n + 1,), (1022,), (1023,)]:
                assert s.contains(pt, symbols=[n]) == aset.contains(pt, symbols=[n])


def eval_bound_eq(b, symbols, expected):
    """Helper: evaluate a Bound and assert it matches *expected*."""
    from ktir_cpu.parser_ast import eval_bound
    return eval_bound(b, symbols) == expected


class TestEqualityBoxSetLowering:
    """BoxSet.try_from_affine_set handles ("eq", lhs, rhs) directly."""

    def test_eq_pins_single_dim(self):
        """affine_set<(g) : (g == 0)> lowers to BoxSet(lo=(0,), hi=(1,))."""
        s = parse_affine_set("affine_set<(g) : (g == 0)>")
        assert isinstance(s, BoxSet)
        assert s == BoxSet(lo=(0,), hi=(1,))

    def test_eq_i_equals_zero(self):
        """Spec example: affine_set<(i) : (i == 0)>."""
        s = parse_affine_set("affine_set<(i) : (i == 0)>")
        assert isinstance(s, BoxSet)
        assert s == BoxSet(lo=(0,), hi=(1,))

    def test_eq_nonzero_pin(self):
        """g == 3 lowers to BoxSet(lo=(3,), hi=(4,))."""
        s = parse_affine_set("affine_set<(g) : (g - 3 == 0)>")
        assert isinstance(s, BoxSet)
        assert s == BoxSet(lo=(3,), hi=(4,))

    def test_eq_pin_and_ineq_intersection(self):
        """g == 0 combined with inequality g <= 5 — pin wins: lo=0, hi=1."""
        from ktir_cpu.parser_ast import parse_affine_set_raw
        from ktir_cpu.affine import BoxSet as _BoxSet
        aset = parse_affine_set_raw("affine_set<(g) : (g == 0, -g + 5 >= 0)>")
        box = _BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(0,), hi=(1,))

    def test_eq_negative_coeff_pin(self):
        """-g + 3 == 0 means g == 3 → BoxSet(lo=(3,), hi=(4,))."""
        from ktir_cpu.parser_ast import parse_affine_set_raw
        from ktir_cpu.affine import BoxSet as _BoxSet
        aset = parse_affine_set_raw("affine_set<(g) : (-g + 3 == 0)>")
        box = _BoxSet.try_from_affine_set(aset)
        assert box == BoxSet(lo=(3,), hi=(4,))

    def test_eq_multi_dim_rejected(self):
        """p - c == 0 involves two dims — cannot lower to BoxSet.

        TODO: remove or update when PR #69 lands (symbolic/multi-dim eq support).
        """
        from ktir_cpu.parser_ast import parse_affine_set_raw
        from ktir_cpu.affine import BoxSet as _BoxSet
        aset = parse_affine_set_raw("affine_set<(p, c) : (p - c == 0)>")
        assert _BoxSet.try_from_affine_set(aset) is None

    def test_spec_example_g_eq_0(self):
        """Full spec example: affine_set<(g) : (g == 0)>."""
        s = parse_affine_set("affine_set<(g) : (g == 0)>")
        assert isinstance(s, BoxSet)
        assert s.lo == (0,)
        assert s.hi == (1,)

    def test_symbolic_eq_pin(self):
        """d0 == s0 lowers to a symbolic BoxSet that specialises to a point."""
        from ktir_cpu.parser_ast import parse_affine_set_raw
        from ktir_cpu.affine import BoxSet as _BoxSet
        aset = parse_affine_set_raw("affine_set<(d0)[s0] : (d0 - s0 == 0)>")
        box = _BoxSet.try_from_affine_set(aset)
        assert box is not None
        assert not box.is_concrete
        assert box.specialize([3]) == BoxSet(lo=(3,), hi=(4,))
        assert box.specialize([7]) == BoxSet(lo=(7,), hi=(8,))

    def test_symbolic_eq_with_offset(self):
        """p - c + 2 == 0  →  p == c - 2; specialise([5]) → BoxSet(lo=(3,), hi=(4,))."""
        from ktir_cpu.parser_ast import parse_affine_set_raw
        from ktir_cpu.affine import BoxSet as _BoxSet
        aset = parse_affine_set_raw("affine_set<(p)[c] : (p - c + 2 == 0)>")
        box = _BoxSet.try_from_affine_set(aset)
        assert box is not None
        assert not box.is_concrete
        assert box.specialize([5]) == BoxSet(lo=(3,), hi=(4,))

    def test_reject_conflicting_eq_constraints(self):
        from ktir_cpu.parser_ast import parse_affine_set_raw
        aset = parse_affine_set_raw("affine_set<(d0) : (d0 == 2, d0 == 3)>")
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_eq_ineq_conflict(self):
        from ktir_cpu.parser_ast import parse_affine_set_raw
        aset = parse_affine_set_raw("affine_set<(d0) : (d0 == 2, d0 >= 5)>")
        assert BoxSet.try_from_affine_set(aset) is None

    def test_reject_conflicting_inequalities(self):
        from ktir_cpu.parser_ast import parse_affine_set_raw
        aset = parse_affine_set_raw("affine_set<(d0) : (d0 >= 5, -d0 + 3 >= 0)>")
        assert BoxSet.try_from_affine_set(aset) is None
