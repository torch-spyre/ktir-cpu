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
Control flow compute helpers.

Conditional, loop, and yield primitives used by dialect handlers
in ``ktir_cpu.dialects``.
"""

from typing import List, Any, Callable

from ..grid import CoreContext
from ..ir_types import Operation, Tile

# (context, operations) -> Any
RegionExecutor = Callable[["CoreContext", List[Operation]], Any]


# Sentinel used by scf.yield to propagate values up to the for-loop driver.
class _YieldResult:
    __slots__ = ("values",)

    def __init__(self, values: List[Any]):
        self.values = values


class ControlOps:
    """Conditional, loop, and yield helpers."""

    @staticmethod
    def if_op(
        context: CoreContext,
        condition: bool,
        then_region: List[Operation],
        else_region: List[Operation],
        region_executor: RegionExecutor,
    ) -> Any:
        """Execute *then_region* or *else_region* based on *condition*.

        Args:
            context: Core execution context
            condition: Boolean condition
            then_region: Operations to execute if true
            else_region: Operations to execute if false
            region_executor: Callable to execute a region

        Returns:
            Result from executed region
        """
        region = then_region if condition else else_region
        if not region:
            return None

        # Branch body gets its own scope; body-local LX is freed on pop.
        # If the branch yields Tile values (via scf.yield), pop_scope frees
        # their LX.  The caller (dialect handler) will re-track them when it
        # binds the scf.if result via _execute_operation -> track_lx.
        context.push_scope()
        result = region_executor(context, region)
        context.pop_scope()
        return result

    @staticmethod
    def for_op(
        context: CoreContext,
        lower_bound: int,
        upper_bound: int,
        step: int,
        iter_var_name: str,
        body_region: List[Operation],
        region_executor: RegionExecutor,
        iter_arg_names: List[str] = None,
        iter_init_values: List[Any] = None,
    ) -> Any:
        """Execute a counted loop with optional carried state (iter_args).

        Runs *body_region* for each step from *lower_bound* to
        *upper_bound*.  When *iter_arg_names* is provided, yielded
        values are fed back as the next iteration's iter_arg bindings.

        Args:
            context: Core execution context
            lower_bound: Loop start (inclusive)
            upper_bound: Loop end (exclusive)
            step: Loop increment
            iter_var_name: Name of iteration variable (e.g., "%i")
            body_region: Loop body operations
            region_executor: Callable to execute a region
            iter_arg_names: Optional list of iter_arg SSA names
            iter_init_values: Optional list of initial values for iter_args

        Returns:
            Final iter_arg values (list) or None if no iter_args.
        """
        if not iter_arg_names:
            iter_arg_names = []
        if not iter_init_values:
            iter_init_values = []

        # Bind initial iter_arg values in the *parent* scope.
        # These persist across iterations; body-local values do not.
        current_values = list(iter_init_values)
        for name, val in zip(iter_arg_names, current_values):
            context.set_value(name, val)
            if isinstance(val, Tile):
                context.track_lx(name, val.size_bytes())

        for i in range(int(lower_bound), int(upper_bound), max(int(step), 1)):
            # New scope for this iteration's body-local values.
            # pop_scope() at the end frees their LX automatically.
            context.push_scope()

            # Set iteration variable
            context.set_value(iter_var_name, i)

            # Execute body
            result = region_executor(context, body_region)

            # Save yielded values before pop_scope() discards them.
            yielded_values = None
            if isinstance(result, _YieldResult) and iter_arg_names:
                yielded_values = result.values

            # Pop body scope — frees LX for all body-local Tiles,
            # including any Tiles that were yielded (they lived in this scope).
            context.pop_scope()

            # Re-bind yielded values as iter_args in the parent scope.
            #
            # iter_args are loop-carried state in scf.for.  They can be
            # scalars (e.g. an index accumulator) or Tiles (e.g. running
            # statistics in online softmax):
            #
            #   scf.for %col = %c0 to %c_C step %c_Bc
            #       iter_args(%m_acc = %m_init, %l_acc = %l_init) {
            #     ...
            #     scf.yield %m_new, %l_new   // Tiles fed back as next %m_acc, %l_acc
            #   }
            #
            # Here %m_init and %l_init are tensor<32x1xf16> — so the
            # iter_args %m_acc and %l_acc are Tiles that occupy LX.
            # When we re-bind them after yield, we must untrack the old
            # Tile's LX and track the new one.
            if yielded_values is not None:
                for name, val in zip(iter_arg_names, yielded_values):
                    context.untrack_lx(name)
                    context.set_value(name, val)
                    if isinstance(val, Tile):
                        context.track_lx(name, val.size_bytes())
                current_values = yielded_values

        if current_values:
            return current_values
        return None

    @staticmethod
    def yield_op(values: List[Any]) -> _YieldResult:
        """Wrap *values* so the loop driver can update iter_args.

        Args:
            values: Values to yield

        Returns:
            _YieldResult wrapping the values
        """
        return _YieldResult(values)

    @staticmethod
    def while_op(
        context: CoreContext,
        before_region: List[Operation],
        after_region: List[Operation],
        region_executor: RegionExecutor,
    ) -> Any:
        """Execute a condition-checked loop.

        Runs *before_region* to obtain a condition; if truthy, executes
        *after_region* and repeats.

        Args:
            context: Core execution context
            before_region: Condition check region
            after_region: Loop body region
            region_executor: Callable to execute a region

        Returns:
            None
        """
        max_iterations = 10000  # Safety limit

        for _ in range(max_iterations):
            # Execute before region (condition check)
            context.push_scope()
            condition = region_executor(context, before_region)
            context.pop_scope()

            if not condition:
                break

            # Execute after region (loop body)
            context.push_scope()
            region_executor(context, after_region)
            context.pop_scope()

        return None
