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
Arithmetic compute helpers.

Pure element-wise and scalar arithmetic on Tiles and ints/floats,
used by dialect handlers in ``ktir_cpu.dialects``.
"""

import numpy as np
from ..ir_types import Tile


class ArithOps:
    """Element-wise arithmetic on tiles and scalars."""

    @staticmethod
    def addf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise floating-point addition.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = tile1.data + tile2.data
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def subf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise floating-point subtraction.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = tile1.data - tile2.data
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def mulf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise floating-point multiplication.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = tile1.data * tile2.data
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def divf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise floating-point division.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = tile1.data / tile2.data
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def addi(val1, val2):
        """Integer addition. Works on both scalars and Tiles.

        When both are Tiles, performs element-wise integer addition.
        When one is a scalar and the other a Tile, broadcasts the scalar.

        Args:
            val1, val2: Input integers or Tiles

        Returns:
            int or Tile result
        """
        if isinstance(val1, Tile) and isinstance(val2, Tile):
            result_data = val1.data + val2.data
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val1, Tile):
            result_data = val1.data + int(val2)
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val2, Tile):
            result_data = int(val1) + val2.data
            return Tile(result_data, val2.dtype, result_data.shape)
        else:
            return int(val1) + int(val2)

    @staticmethod
    def subi(val1, val2):
        """Integer subtraction. Works on both scalars and Tiles."""
        if isinstance(val1, Tile) and isinstance(val2, Tile):
            result_data = val1.data - val2.data
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val1, Tile):
            result_data = val1.data - int(val2)
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val2, Tile):
            result_data = int(val1) - val2.data
            return Tile(result_data, val2.dtype, result_data.shape)
        else:
            return val1 - val2

    @staticmethod
    def muli(val1, val2):
        """Integer multiplication. Works on both scalars and Tiles.

        When both are Tiles, performs element-wise integer multiplication.
        When one is a scalar and the other a Tile, broadcasts the scalar.

        Args:
            val1, val2: Input integers or Tiles

        Returns:
            int or Tile result
        """
        if isinstance(val1, Tile) and isinstance(val2, Tile):
            result_data = val1.data * val2.data
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val1, Tile):
            result_data = val1.data * int(val2)
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val2, Tile):
            result_data = int(val1) * val2.data
            return Tile(result_data, val2.dtype, result_data.shape)
        else:
            return int(val1) * int(val2)

    @staticmethod
    def maxnumf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise max, NaN-aware (NaN is treated as missing).

        For CPU simulation this is identical to maxf since we use
        np.maximum which already handles NaN correctly.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = np.maximum(tile1.data, tile2.data)
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def cmpi(val1, val2, predicate: str = "slt"):
        """Integer comparison. Works on both scalars and Tiles.

        Returns a boolean (scalar) or a Tile of booleans.

        Predicates: eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge.

        Args:
            val1, val2: Input integers or Tiles
            predicate: Comparison predicate string

        Returns:
            bool or Tile result
        """
        # Map predicate string to a comparison function
        cmp_ops = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "slt": lambda a, b: a < b,
            "sle": lambda a, b: a <= b,
            "sgt": lambda a, b: a > b,
            "sge": lambda a, b: a >= b,
            "ult": lambda a, b: a < b,   # Unsigned — same for positive values
            "ule": lambda a, b: a <= b,
            "ugt": lambda a, b: a > b,
            "uge": lambda a, b: a >= b,
        }
        cmp_fn = cmp_ops.get(predicate, cmp_ops["slt"])

        if isinstance(val1, Tile) and isinstance(val2, Tile):
            result_data = cmp_fn(val1.data, val2.data)
            return Tile(result_data, "i1", result_data.shape)
        elif isinstance(val1, Tile):
            scalar = int(val2) if not isinstance(val2, (np.generic, float)) else val2
            result_data = cmp_fn(val1.data, scalar)
            return Tile(result_data, "i1", result_data.shape)
        elif isinstance(val2, Tile):
            scalar = int(val1) if not isinstance(val1, (np.generic, float)) else val1
            result_data = cmp_fn(scalar, val2.data)
            return Tile(result_data, "i1", result_data.shape)
        else:
            return bool(cmp_fn(val1, val2))

    @staticmethod
    def select(condition, true_val, false_val):
        """Conditional selection. Works on both scalars and Tiles.

        For scalars: returns true_val if condition else false_val.
        For Tiles: element-wise np.where.

        Args:
            condition: bool or Tile of booleans
            true_val: Value to use where condition is true
            false_val: Value to use where condition is false

        Returns:
            Selected value (same type as true_val/false_val)
        """
        if isinstance(condition, Tile):
            # Get the data arrays, broadcasting as needed
            t_data = true_val.data if isinstance(true_val, Tile) else true_val
            f_data = false_val.data if isinstance(false_val, Tile) else false_val
            result_data = np.where(condition.data, t_data, f_data)
            # Determine output dtype and shape from the result
            out_dtype = true_val.dtype if isinstance(true_val, Tile) else (false_val.dtype if isinstance(false_val, Tile) else "f16")
            return Tile(result_data, out_dtype, result_data.shape)
        else:
            return true_val if condition else false_val

    @staticmethod
    def extf(value):
        """Widen float (e.g. f16 to f32).

        Args:
            value: Tile or scalar

        Returns:
            Value promoted to f32.
        """
        if isinstance(value, Tile):
            return Tile(value.data.astype(np.float32), "f32", value.shape)
        return np.float32(value)

    @staticmethod
    def truncf(value):
        """Narrow float (e.g. f32 to f16).

        Args:
            value: Tile or scalar

        Returns:
            Value converted to f16, or passthrough if already f16
        """
        if isinstance(value, Tile):
            if value.data.dtype == np.float16:
                return value
            return Tile(value.data.astype(np.float16), "f16", value.shape)
        if isinstance(value, np.float16):
            return value
        return np.float16(value)

    @staticmethod
    def maxf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise maximum.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = np.maximum(tile1.data, tile2.data)
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def minf(tile1: Tile, tile2: Tile) -> Tile:
        """Element-wise minimum.

        Args:
            tile1, tile2: Input tiles

        Returns:
            Result tile
        """
        result_data = np.minimum(tile1.data, tile2.data)
        return Tile(result_data, tile1.dtype, tile1.shape)

    @staticmethod
    def divui(val1, val2):
        """Unsigned integer floor division. Works on both scalars and Tiles."""
        if isinstance(val1, Tile) and isinstance(val2, Tile):
            result_data = val1.data // val2.data
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val1, Tile):
            result_data = val1.data // int(val2)
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val2, Tile):
            result_data = int(val1) // val2.data
            return Tile(result_data, val2.dtype, result_data.shape)
        else:
            return val1 // val2

    @staticmethod
    def remui(val1, val2):
        """Unsigned integer remainder. Works on both scalars and Tiles."""
        if isinstance(val1, Tile) and isinstance(val2, Tile):
            result_data = val1.data % val2.data
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val1, Tile):
            result_data = val1.data % int(val2)
            return Tile(result_data, val1.dtype, result_data.shape)
        elif isinstance(val2, Tile):
            result_data = int(val1) % val2.data
            return Tile(result_data, val2.dtype, result_data.shape)
        else:
            return val1 % val2

    @staticmethod
    def matmul(tile_a: Tile, tile_b: Tile) -> Tile:
        """Matrix multiplication of two 2D tiles.

        Computes tile_a @ tile_b using NumPy's matmul in f16.
        This maps to Spyre's hardware tensor compute units which
        natively support matrix multiply on 2D tiles.

        Args:
            tile_a: Left input tile, shape [M, K], dtype f16.
            tile_b: Right input tile, shape [K, N], dtype f16.

        Returns:
            Result tile, shape [M, N], dtype f16.
        """
        result_data = np.matmul(tile_a.data, tile_b.data).astype(tile_a.data.dtype)
        result_shape = (tile_a.shape[0], tile_b.shape[1])
        return Tile(result_data, tile_a.dtype, result_shape)
