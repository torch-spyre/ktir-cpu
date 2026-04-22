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
Grid compute helpers.

Core identity and grid coordinate queries used by dialect handlers
in ``ktir_cpu.dialects``.
"""

from typing import List
from ..grid import CoreContext, GridExecutor


class GridOps:
    """Core identity and grid coordinate helpers."""

    @staticmethod
    def gridid(context: CoreContext, dim: int) -> int:
        """Return the grid coordinate for the current core in *dim*.

        Args:
            context: Core execution context
            dim: Dimension (0=x, 1=y, 2=z). In ktdp, always 0.

        Returns:
            Grid coordinate in that dimension (index-typed value)
        """
        return context.get_grid_id(dim)

    @staticmethod
    def coreid(context: CoreContext, grid_coords: List[int], grid_executor: GridExecutor) -> List[int]:
        """Return core IDs matching *grid_coords* (use -1 as wildcard).

        Args:
            context: Core execution context
            grid_coords: List of [x, y, z] coordinates (-1 for wildcard)
            grid_executor: GridExecutor to query cores

        Returns:
            List of matching core IDs

        Example:
            coreid(ctx, [-1, 2, 0], grid) returns all cores at y=2, z=0
        """
        # Pad grid_coords to 3 dimensions if needed
        while len(grid_coords) < 3:
            grid_coords.append(0)

        # Get matching cores
        return grid_executor.get_cores_in_group(tuple(grid_coords[:3]))
