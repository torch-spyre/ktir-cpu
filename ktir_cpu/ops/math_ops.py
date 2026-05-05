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


class MathOps:
    """Transcendental math on tiles and scalars."""

    @staticmethod
    def exp(tile: Tile) -> Tile:
        """Element-wise exponential (e^x).

        Computes e raised to the power of each element in the input tile.

        Args:
            tile: Input tile with f16 data

        Returns:
            Result tile where each element is exp(input_element)
        """
        result_data = np.exp(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def exp_scalar(value: np.floating) -> np.floating:
        """Scalar exponential.

        Args:
            value: Input scalar

        Returns:
            exp(value) in the same type
        """
        return type(value)(np.exp(float(value)))

    @staticmethod
    def sqrt(tile: Tile) -> Tile:
        """Element-wise square root.

        Computes the square root of each element in the input tile.

        Args:
            tile: Input tile with f16 data

        Returns:
            Result tile where each element is sqrt(input_element)
        """
        result_data = np.sqrt(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def sqrt_scalar(value: np.floating) -> np.floating:
        """Scalar square root.

        Args:
            value: Input scalar

        Returns:
            sqrt(value) in the same type
        """
        return type(value)(np.sqrt(float(value)))

    @staticmethod
    def rsqrt(tile: Tile) -> Tile:
        """Element-wise reciprocal square root (1/sqrt(x))."""
        result_data = (1.0 / np.sqrt(tile.data.astype(np.float32))).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def rsqrt_scalar(value: np.floating) -> np.floating:
        """Scalar reciprocal square root."""
        return type(value)(1.0 / np.sqrt(float(value)))

    @staticmethod
    def log(tile: Tile) -> Tile:
        """Element-wise logarithm.

        Computes the log of each element in the input tile.

        Args:
            tile: Input tile with f16 data

        Returns:
            Result tile where each element is log(input_element)
        """
        result_data = np.log(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def log_scalar(value: np.floating) -> np.floating:
        """Scalar logarithm.

        Args:
            value: Input scalar

        Returns:
            log(value) in the same type
        """
        return type(value)(np.log(float(value)))

    @staticmethod
    def log2(tile: Tile) -> Tile:
        """Element-wise base-2 logarithm."""
        result_data = np.log2(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def log2_scalar(value: np.floating) -> np.floating:
        """Scalar base-2 logarithm."""
        return type(value)(np.log2(float(value)))

    @staticmethod
    def log1p(tile: Tile) -> Tile:
        """Element-wise log(1 + x)."""
        result_data = np.log1p(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def log1p_scalar(value: np.floating) -> np.floating:
        """Scalar log(1 + x)."""
        return type(value)(np.log1p(float(value)))

    @staticmethod
    def tanh(tile: Tile) -> Tile:
        """Element-wise hyperbolic tangent."""
        result_data = np.tanh(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def tanh_scalar(value: np.floating) -> np.floating:
        """Scalar hyperbolic tangent."""
        return type(value)(np.tanh(float(value)))

    @staticmethod
    def sin(tile: Tile) -> Tile:
        """Element-wise sine."""
        result_data = np.sin(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def sin_scalar(value: np.floating) -> np.floating:
        """Scalar sine."""
        return type(value)(np.sin(float(value)))

    @staticmethod
    def cos(tile: Tile) -> Tile:
        """Element-wise cosine."""
        result_data = np.cos(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def cos_scalar(value: np.floating) -> np.floating:
        """Scalar cosine."""
        return type(value)(np.cos(float(value)))

    @staticmethod
    def abs(tile: Tile) -> Tile:
        """Element-wise absolute value."""
        result_data = np.abs(tile.data)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def abs_scalar(value: np.floating) -> np.floating:
        """Scalar absolute value."""
        return type(value)(abs(float(value)))

    @staticmethod
    def ceil(tile: Tile) -> Tile:
        """Element-wise ceiling."""
        result_data = np.ceil(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def ceil_scalar(value: np.floating) -> np.floating:
        """Scalar ceiling."""
        return type(value)(np.ceil(float(value)))

    @staticmethod
    def floor(tile: Tile) -> Tile:
        """Element-wise floor."""
        result_data = np.floor(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def floor_scalar(value: np.floating) -> np.floating:
        """Scalar floor."""
        return type(value)(np.floor(float(value)))

    @staticmethod
    def _erf_f32(x: np.ndarray) -> np.ndarray:
        """Vectorized erf via Abramowitz & Stegun 7.1.26 (max error < 1.5e-7).

        Uses a polynomial approximation to avoid a scipy dependency.
        """
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (
            1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return np.sign(x) * (1.0 - poly * np.exp(-a * a))

    @staticmethod
    def erf(tile: Tile) -> Tile:
        """Element-wise error function."""
        result_data = MathOps._erf_f32(tile.data.astype(np.float32)).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def erf_scalar(value: np.floating) -> np.floating:
        """Scalar error function."""
        return type(value)(MathOps._erf_f32(np.array([float(value)]))[0])

    @staticmethod
    def powf(tile: Tile, exponent: Tile) -> Tile:
        """Element-wise power."""
        result_data = np.power(
            tile.data.astype(np.float32), exponent.data.astype(np.float32)
        ).astype(tile.data.dtype)
        return Tile(result_data, tile.dtype, tile.shape)

    @staticmethod
    def fma(a: Tile, b: Tile, c: Tile) -> Tile:
        """Fused multiply-add: a * b + c."""
        result_data = (
            a.data.astype(np.float32) * b.data.astype(np.float32)
            + c.data.astype(np.float32)
        ).astype(a.data.dtype)
        return Tile(result_data, a.dtype, a.shape)
