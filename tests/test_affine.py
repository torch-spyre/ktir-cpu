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

"""Tests for affine.py — AffineMap and AffineSet value objects.

These tests verify that the convenience methods (eval, contains, enumerate)
on AffineMap and AffineSet correctly delegate to parser_ast.py.  They are
intentionally thin — the heavy evaluation logic is tested in test_ast.py.
"""

import pytest

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

    def test_contains_delegates(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        assert s.contains([2])
        assert not s.contains([5])

    def test_enumerate_delegates(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        assert s.enumerate((4,)) == [(0,), (1,), (2,), (3,)]

    def test_enumerate_wrong_shape_raises(self):
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>")
        with pytest.raises(ValueError):
            s.enumerate((4,))

    def test_source_field(self):
        src = "affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>"
        s = parse_affine_set(src)
        assert s.source == src

    def test_frozen(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0)>")
        with pytest.raises((AttributeError, TypeError)):
            s.n_dims = 99  # type: ignore[misc]
