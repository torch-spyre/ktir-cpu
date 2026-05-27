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

    def test_reject_one_axis_unpinned(self):
        """2D set where d1 has no upper bound."""
        aset = self._parse_aset(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>"
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
