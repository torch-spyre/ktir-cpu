# Copyright 2026 The Torch-Spyre Authors.
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

"""Shared compute helpers for ops/ layer.

Pure helpers (numpy in, result out) with no IR knowledge.
"""

from functools import partial

import numpy as np

from ..ir_types import Tile


def _tile_binop(fn, val1, val2, *, scalar_cast):
    """Generic binary dispatch across Tile/scalar combinations.

    Dispatches fn across all four cases: Tile×Tile, Tile×scalar,
    scalar×Tile, scalar×scalar. scalar_cast converts scalars before
    passing to fn.
    """
    if isinstance(val1, Tile) and isinstance(val2, Tile):
        r = fn(val1.data, val2.data)
        return Tile(r, val1.dtype, r.shape)
    if isinstance(val1, Tile):
        r = fn(val1.data, scalar_cast(val2))
        return Tile(r, val1.dtype, r.shape)
    if isinstance(val2, Tile):
        r = fn(scalar_cast(val1), val2.data)
        return Tile(r, val2.dtype, r.shape)
    return fn(scalar_cast(val1), scalar_cast(val2))


tile_binop_int = partial(_tile_binop, scalar_cast=int)
tile_binop_float = partial(_tile_binop, scalar_cast=float)


def _tile_unary(fn, val, *, scalar_cast):
    """Generic unary dispatch for Tile or scalar.

    For Tiles: casts to float32, applies fn, casts back to the tile's
    original dtype, and wraps in a new Tile. For scalars: casts via
    scalar_cast, applies fn, and returns a value of the same type as
    the input.
    """
    if isinstance(val, Tile):
        data = fn(val.data.astype(np.float32)).astype(val.data.dtype)
        return Tile(data, val.dtype, data.shape)
    return type(val)(fn(scalar_cast(val)))


tile_unary_float = partial(_tile_unary, scalar_cast=float)
tile_unary_int = partial(_tile_unary, scalar_cast=int)
