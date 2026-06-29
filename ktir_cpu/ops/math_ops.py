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
Transcendental math compute helpers.

Element-wise exp, sqrt, and log on Tiles and scalars, used by dialect
handlers in ``ktir_cpu.dialects``.
"""

import numpy as np
from ..ir_types import Tile
from ._helpers import tile_unary_float, tile_unary_int


class MathOps:
    """Transcendental math on tiles and scalars."""

    @staticmethod
    def exp(val):
        """Element-wise exponential (e^x)."""
        return tile_unary_float(np.exp, val)

    @staticmethod
    def sqrt(val):
        """Element-wise square root."""
        return tile_unary_float(np.sqrt, val)

    @staticmethod
    def rsqrt(val):
        """Element-wise reciprocal square root (1/sqrt(x))."""
        return tile_unary_float(lambda x: 1.0 / np.sqrt(x), val)

    @staticmethod
    def log(val):
        """Element-wise natural logarithm."""
        return tile_unary_float(np.log, val)

    @staticmethod
    def log2(val):
        """Element-wise base-2 logarithm."""
        return tile_unary_float(np.log2, val)

    @staticmethod
    def log1p(val):
        """Element-wise log(1 + x)."""
        return tile_unary_float(np.log1p, val)

    @staticmethod
    def tanh(val):
        """Element-wise hyperbolic tangent."""
        return tile_unary_float(np.tanh, val)

    @staticmethod
    def sin(val):
        """Element-wise sine."""
        return tile_unary_float(np.sin, val)

    @staticmethod
    def cos(val):
        """Element-wise cosine."""
        return tile_unary_float(np.cos, val)

    @staticmethod
    def ceil(val):
        """Element-wise ceiling."""
        return tile_unary_float(np.ceil, val)

    @staticmethod
    def floor(val):
        """Element-wise floor."""
        return tile_unary_float(np.floor, val)

    @staticmethod
    def absf(val):
        """Element-wise absolute value (float)."""
        # Intentionally not using tile_unary_float: np.abs works on any dtype
        # directly, so the float32 round-trip would be lossy without benefit.
        if isinstance(val, Tile):
            return Tile(np.abs(val.data), val.dtype, val.shape)
        return type(val)(abs(float(val)))

    @staticmethod
    def absi(val):
        """Element-wise absolute value (integer)."""
        return tile_unary_int(np.abs, val)

    @staticmethod
    def _erf_f32(x: np.ndarray) -> np.ndarray:
        """Vectorized erf via Abramowitz & Stegun 7.1.26 (max error < 1.5e-7)."""
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (
            1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return np.sign(x) * (1.0 - poly * np.exp(-a * a))

    @staticmethod
    def erf(val):
        """Element-wise error function."""
        return tile_unary_float(MathOps._erf_f32, val)

    @staticmethod
    def powf(tile: Tile, exponent: Tile) -> Tile:
        """Element-wise power."""
        result_data = np.power(
            tile.data.astype(np.float32), exponent.data.astype(np.float32)
        ).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def powf_scalar(base: np.floating, exponent: np.floating) -> np.floating:
        """Scalar power."""
        return type(base)(float(base) ** float(exponent))

    @staticmethod
    def fma(a: Tile, b: Tile, c: Tile) -> Tile:
        """Fused multiply-add: a * b + c."""
        result_data = (
            a.data.astype(np.float32) * b.data.astype(np.float32)
            + c.data.astype(np.float32)
        ).astype(a.data.dtype)
        return Tile(result_data, a.dtype, a.shape)

    @staticmethod
    def fma_scalar(a: np.floating, b: np.floating, c: np.floating) -> np.floating:
        """Scalar fused multiply-add: a * b + c."""
        return type(a)(float(a) * float(b) + float(c))
