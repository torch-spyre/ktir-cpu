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
AffineSet   — represents affine_set<(d0,...) : (c0 >= 0, ...)>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

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
        """Return True if this map is the identity: output == input for all inputs.

        Used at parse time to detect trivial coordinate-order maps.  When
        ``access_tile_order`` is an identity map it has no effect on which
        memory element lands at each output position, so we set
        ``coordinate_order`` to ``None``.  This allows load/store to skip
        the per-coord ``cso.eval()`` calls and, combined with a full
        ``coordinate_set``, enables the contiguous fast path entirely.
        """
        # Quick structural check: same number of inputs and outputs, and each
        # output expression is just the corresponding input dimension variable.
        if len(self.exprs) != self.n_dims:
            return False
        # Verify by evaluation: identity map satisfies eval(dims) == dims for
        # a non-trivial probe vector.  Using [1, 2, ..., n_dims] avoids false
        # positives from constant-zero expressions.
        probe = list(range(1, self.n_dims + 1))
        return list(self.eval(probe)) == probe


@dataclass(frozen=True)
class AffineSet:
    """Parsed affine_set<(d0,...) : (c0 >= 0, ...)>.

    Attributes:
        n_dims:       number of dimension variables
        constraints:  tuple of AST nodes; each node is the LHS of ``expr >= 0``
        source:       original verbatim string (for debugging / round-trip)
    """
    n_dims: int
    constraints: Tuple["_Node", ...]
    source: str

    def contains(self, point: Sequence[int]) -> bool:
        """Return True if *point* satisfies all constraints.

        Delegates to ``parser_ast.affine_set_contains``.
        """
        from .parser_ast import affine_set_contains
        return affine_set_contains(self, point)

    def enumerate(self, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Return all integer points in ``[0, shape)`` satisfying all constraints.

        Delegates to ``parser_ast.enumerate_affine_set``.
        """
        from .parser_ast import enumerate_affine_set
        return enumerate_affine_set(self, shape)

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
