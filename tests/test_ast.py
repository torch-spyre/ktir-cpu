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

"""Tests for parser_ast.py — AST parsing and evaluation mechanics.

Focuses on tokenisation, expression structure, and evaluation, independent
of any MLIR interpreter context.
"""

import pytest

from ktir_cpu.parser_ast import (
    _tokenise,
    parse_expr,
    eval_expr,
    parse_affine_map,
    parse_affine_set,
    eval_affine_map,
    affine_set_contains,
    enumerate_affine_set,
)


# ===========================================================================
# Tokeniser
# ===========================================================================

class TestTokenise:
    def test_simple_map_inner(self):
        tokens = _tokenise("(d0, d1) -> (d0, d1)")
        assert tokens == ["(", "d0", ",", "d1", ")", "->", "(", "d0", ",", "d1", ")"]

    def test_constraint_tokens(self):
        tokens = _tokenise("(d0 >= 0, -d0 + 63 >= 0)")
        assert ">=" in tokens
        assert "0" in tokens

    def test_arrow_token(self):
        tokens = _tokenise("(d0) -> (d0)")
        assert "->" in tokens

    def test_whitespace_ignored(self):
        t1 = _tokenise("(d0)->(d0)")
        t2 = _tokenise("( d0 ) -> ( d0 )")
        assert t1 == t2


# ===========================================================================
# Expression parsing — generic (non-affine-map/set context)
# ===========================================================================

class TestParseExpr:
    """Test the expression parser in isolation via parse_expr / eval_expr."""

    def test_constant(self):
        node = parse_expr("42")
        assert node == ("const", 42)
        assert eval_expr(node, []) == 42

    def test_dim_variable(self):
        node = parse_expr("d0")
        assert node == ("dim", 0)
        assert eval_expr(node, [7]) == 7

    def test_dim_variable_index(self):
        node = parse_expr("d2")
        assert node == ("dim", 2)
        assert eval_expr(node, [0, 0, 99]) == 99

    def test_addition(self):
        node = parse_expr("d0 + d1")
        assert node == ("add", ("dim", 0), ("dim", 1))
        assert eval_expr(node, [3, 4]) == 7

    def test_subtraction(self):
        node = parse_expr("d0 - d1")
        assert node == ("sub", ("dim", 0), ("dim", 1))
        assert eval_expr(node, [10, 3]) == 7

    def test_unary_negation(self):
        node = parse_expr("-d0")
        assert node == ("neg", ("dim", 0))
        assert eval_expr(node, [5]) == -5

    def test_constant_coefficient(self):
        node = parse_expr("2 * d0")
        assert node == ("mul", 2, ("dim", 0))
        assert eval_expr(node, [4]) == 8

    def test_negative_coefficient_expr(self):
        # -d0 + 63  (common RFC constraint pattern)
        node = parse_expr("-d0 + 63")
        assert node == ("add", ("neg", ("dim", 0)), ("const", 63))
        assert eval_expr(node, [0]) == 63
        assert eval_expr(node, [63]) == 0
        assert eval_expr(node, [64]) == -1

    def test_compound_expr(self):
        # d0 + 2 * d1 + 3
        node = parse_expr("d0 + 2 * d1 + 3")
        assert eval_expr(node, [1, 2]) == 1 + 2 * 2 + 3  # == 8

    def test_left_associativity(self):
        # a - b + c  should be (a - b) + c, not a - (b + c)
        node = parse_expr("d0 - d1 + d2")
        assert eval_expr(node, [10, 3, 1]) == 8   # (10-3)+1

    def test_parenthesised(self):
        node = parse_expr("2 * (d0 + 1)")
        assert eval_expr(node, [4]) == 10

    def test_zero_constant(self):
        node = parse_expr("0")
        assert node == ("const", 0)
        assert eval_expr(node, []) == 0


# ===========================================================================
# parse_affine_map — AST structure
# ===========================================================================

class TestParseAffineMap:

    def test_identity_1d(self):
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        assert m.n_dims == 1
        assert len(m.exprs) == 1
        assert m.exprs[0] == ("dim", 0)

    def test_identity_2d(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        assert m.n_dims == 2
        assert m.exprs == (("dim", 0), ("dim", 1))

    def test_non_identity_row_select(self):
        # (i) -> (i, 0)  — softmax_wide.mlir pattern
        m = parse_affine_map("affine_map<(i) -> (i, 0)>")
        assert m.n_dims == 1
        assert len(m.exprs) == 2
        assert m.exprs[0] == ("dim", 0)
        assert m.exprs[1] == ("const", 0)

    def test_transposed(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert m.exprs == (("dim", 1), ("dim", 0))

    def test_constant_offset(self):
        m = parse_affine_map("affine_map<(d0) -> (d0 + 1)>")
        assert m.exprs[0] == ("add", ("dim", 0), ("const", 1))

    def test_scaled(self):
        m = parse_affine_map("affine_map<(d0) -> (2 * d0)>")
        assert m.exprs[0] == ("mul", 2, ("dim", 0))

    def test_complex_expr(self):
        # (d0 + 2 * d1)
        m = parse_affine_map("affine_map<(d0, d1) -> (d0 + 2 * d1)>")
        assert m.exprs[0] == ("add", ("dim", 0), ("mul", 2, ("dim", 1)))

    def test_negative_expr(self):
        # -d0 + 63
        m = parse_affine_map("affine_map<(d0) -> (-d0 + 63)>")
        assert m.exprs[0] == ("add", ("neg", ("dim", 0)), ("const", 63))

    def test_inner_text_without_wrapper(self):
        m = parse_affine_map("(d0) -> (d0)")
        assert m.n_dims == 1

    def test_source_preserved(self):
        s = "affine_map<(d0) -> (d0)>"
        m = parse_affine_map(s)
        assert m.source == s

    def test_zero_dims(self):
        m = parse_affine_map("affine_map<() -> (0)>")
        assert m.n_dims == 0
        assert m.exprs == (("const", 0),)


# ===========================================================================
# parse_affine_set — AST structure
# ===========================================================================

class TestParseAffineSet:

    def test_1d_range(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>")
        assert s.n_dims == 1
        assert len(s.constraints) == 2
        # Constraints normalised to (lhs - rhs >= 0)
        assert s.constraints[0] == ("sub", ("dim", 0), ("const", 0))
        assert s.constraints[1] == ("sub", ("add", ("neg", ("dim", 0)), ("const", 31)), ("const", 0))

    def test_2d_rect(self):
        src = "affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>"
        s = parse_affine_set(src)
        assert s.n_dims == 2
        assert len(s.constraints) == 4

    def test_source_preserved(self):
        src = "affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>"
        s = parse_affine_set(src)
        assert s.source == src

    def test_inner_text_without_wrapper(self):
        s = parse_affine_set("(d0) : (d0 >= 0, -d0 + 3 >= 0)")
        assert s.n_dims == 1
        assert len(s.constraints) == 2

    def test_leq_normalised(self):
        # d0 <= 0  normalised to  0 - d0 >= 0
        s = parse_affine_set("affine_set<(d0) : (d0 <= 0)>")
        assert s.constraints[0] == ("sub", ("const", 0), ("dim", 0))

    def test_general_rhs(self):
        # d0 >= d1  →  d0 - d1 >= 0
        # d0 <= 63  →  63 - d0 >= 0
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= d1, d0 <= 63)>")
        assert s.n_dims == 2
        assert s.constraints[0] == ("sub", ("dim", 0), ("dim", 1))
        assert s.constraints[1] == ("sub", ("const", 63), ("dim", 0))


# ===========================================================================
# eval_affine_map
# ===========================================================================

class TestEvalAffineMap:

    def test_identity_1d(self):
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        assert eval_affine_map(m, [5]) == (5,)

    def test_identity_2d(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        assert eval_affine_map(m, [3, 7]) == (3, 7)

    def test_row_select(self):
        m = parse_affine_map("affine_map<(i) -> (i, 0)>")
        assert eval_affine_map(m, [2]) == (2, 0)
        assert eval_affine_map(m, [0]) == (0, 0)

    def test_transposed(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>")
        assert eval_affine_map(m, [3, 7]) == (7, 3)

    def test_constant_offset(self):
        m = parse_affine_map("affine_map<(d0) -> (d0 + 1)>")
        assert eval_affine_map(m, [4]) == (5,)

    def test_scaled(self):
        m = parse_affine_map("affine_map<(d0) -> (2 * d0)>")
        assert eval_affine_map(m, [3]) == (6,)

    def test_wrong_dim_count_raises(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        with pytest.raises(ValueError, match="expects 2"):
            eval_affine_map(m, [1])

    def test_zero_dims(self):
        m = parse_affine_map("affine_map<() -> (0)>")
        assert eval_affine_map(m, []) == (0,)


# ===========================================================================
# affine_set_contains
# ===========================================================================

class TestAffineSetContains:

    def test_inside_1d(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        for i in range(4):
            assert affine_set_contains(s, [i])

    def test_outside_1d(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        assert not affine_set_contains(s, [-1])
        assert not affine_set_contains(s, [4])

    def test_2d_boundary(self):
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0)>"
        )
        assert affine_set_contains(s, [0, 0])
        assert affine_set_contains(s, [1, 1])
        assert not affine_set_contains(s, [2, 0])

    def test_general_rhs_contains(self):
        # d0 >= d1 and d0 <= 63
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= d1, d0 <= 63)>")
        assert affine_set_contains(s, [5, 3])    # 5 >= 3, 5 <= 63
        assert affine_set_contains(s, [63, 63])  # 63 >= 63, 63 <= 63
        assert not affine_set_contains(s, [2, 5])  # 2 < 5
        assert not affine_set_contains(s, [64, 0]) # 64 > 63


# ===========================================================================
# enumerate_affine_set
# ===========================================================================

class TestEnumerateAffineSet:

    def test_1d_range(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>")
        pts = enumerate_affine_set(s, (32,))
        assert len(pts) == 32
        assert pts[0] == (0,)
        assert pts[-1] == (31,)

    def test_2d_rect_64x64(self):
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>"
        )
        pts = enumerate_affine_set(s, (64, 64))
        assert len(pts) == 64 * 64
        assert pts[0] == (0, 0)
        assert pts[-1] == (63, 63)

    def test_shape_larger_than_set(self):
        # set says d0 in [0,3], shape is (8,) — only 4 points back
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        pts = enumerate_affine_set(s, (8,))
        assert len(pts) == 4
        assert all(0 <= p[0] <= 3 for p in pts)

    def test_empty_set(self):
        # infeasible: d0 >= 5 and d0 <= 3
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0, d0 + -5 >= 0)>")
        pts = enumerate_affine_set(s, (10,))
        assert pts == []

    def test_shape_dim_mismatch_raises(self):
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>")
        with pytest.raises(ValueError, match="2 dim"):
            enumerate_affine_set(s, (4,))


# ===========================================================================
# Edge-case tests — non-rectangular sets, conflicting constraints, zero-dim maps
# ===========================================================================

class TestAffineEdgeCases:
    """Edge cases for affine maps and sets: triangular sets, empty sets
    from conflicting constraints, and zero-dimensional maps."""

    def test_triangular_affine_set(self):
        """Non-rectangular affine set: lower-triangular (d0 >= d1).

        For a 4x4 bounding box, d0 >= d1 yields the lower triangle:
          (0,0), (1,0),(1,1), (2,0),(2,1),(2,2), (3,0),(3,1),(3,2),(3,3)
          = 1 + 2 + 3 + 4 = 10 points
        """
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d0 - d1 >= 0)>"
        )
        pts = enumerate_affine_set(s, (4, 4))
        assert len(pts) == 10
        # Every point satisfies d0 >= d1
        for r, c in pts:
            assert r >= c, f"({r},{c}) violates d0 >= d1"

    def test_triangular_affine_set_sum_constraint(self):
        """Non-rectangular affine set with d0 + d1 <= N constraint.

        For a 4x4 bounding box with d0 + d1 <= 3, valid points form a triangle:
          All (d0, d1) where d0 >= 0, d1 >= 0, d0 + d1 <= 3
          = 10 points (same count as lower-triangular 4x4).
        """
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, -d0 + 3 >= 0, -d1 + 3 >= 0, -d0 - d1 + 3 >= 0)>"
        )
        # -d0 - d1 + 3 >= 0 is equivalent to d0 + d1 <= 3
        pts = enumerate_affine_set(s, (4, 4))
        assert len(pts) == 10
        for d0, d1 in pts:
            assert d0 + d1 <= 3, f"({d0},{d1}) violates d0 + d1 <= 3"

    def test_triangular_contains(self):
        """affine_set_contains correctly evaluates a triangular constraint."""
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, d0 - d1 >= 0)>"
        )
        assert affine_set_contains(s, [3, 1])    # 3 >= 1
        assert affine_set_contains(s, [2, 2])    # 2 >= 2
        assert not affine_set_contains(s, [1, 3])  # 1 < 3

    def test_conflicting_constraints_empty_set(self):
        """Conflicting constraints produce an empty enumeration.

        d0 >= 5 AND d0 <= 2 is infeasible within any bounding box.
        """
        s = parse_affine_set(
            "affine_set<(d0) : (d0 - 5 >= 0, -d0 + 2 >= 0)>"
        )
        pts = enumerate_affine_set(s, (10,))
        assert pts == []

    def test_conflicting_constraints_2d_empty(self):
        """2D conflicting constraints: d0 > d1 AND d1 > d0 is unsatisfiable.

        d0 - d1 - 1 >= 0 means d0 > d1; d1 - d0 - 1 >= 0 means d1 > d0.
        Together they are contradictory.
        """
        s = parse_affine_set(
            "affine_set<(d0, d1) : (d0 - d1 - 1 >= 0, d1 - d0 - 1 >= 0)>"
        )
        pts = enumerate_affine_set(s, (4, 4))
        assert pts == []

    def test_zero_dim_affine_map_parse(self):
        """Zero-dimensional affine map: () -> (0) has n_dims=0 and one constant output."""
        m = parse_affine_map("affine_map<() -> (0)>")
        assert m.n_dims == 0
        assert len(m.exprs) == 1
        assert m.exprs[0] == ("const", 0)

    def test_zero_dim_affine_map_eval(self):
        """Zero-dimensional affine map evaluates correctly with empty dim list."""
        m = parse_affine_map("affine_map<() -> (42)>")
        assert eval_affine_map(m, []) == (42,)

    def test_zero_dim_affine_map_multi_output(self):
        """Zero-dimensional affine map with multiple constant outputs."""
        m = parse_affine_map("affine_map<() -> (1, 2, 3)>")
        assert m.n_dims == 0
        assert len(m.exprs) == 3
        assert eval_affine_map(m, []) == (1, 2, 3)
