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
