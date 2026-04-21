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
Main KTIR interpreter engine.

Orchestrates parsing, execution, and validation of KTIR code.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import math

from .ir_types import Operation, IRModule, Tile
from .parser import KTIRParser, KTIRParserBase
from .memory import SpyreMemoryHierarchy
from .grid import GridExecutor, CoreContext
from .ops.comm_ops import RingNetwork
from .latency import HardwareConfig, LatencyTracker, LatencyReport
from .dialects import dispatch, ExecutionEnv


def _ktir_dtype(np_dtype: np.dtype) -> str:
    """Map a numpy dtype to a KTIR dtype string."""
    if np_dtype == np.float16:
        return "f16"
    elif np_dtype == np.float32:
        return "f32"
    elif np_dtype == np.int32:
        return "i32"
    else:
        return "f16"


class KTIRInterpreter:
    """Main KTIR interpreter.

    Loads KTIR MLIR code, sets up execution environment, and executes functions.
    """

    def __init__(
        self,
        latency_config: Optional[HardwareConfig] = None,
        trace_latency: bool = False,
        parser: Optional[KTIRParserBase] = None,
    ):
        """Create a KTIR interpreter.

        Args:
            latency_config: Hardware parameters for latency estimation.
                When provided, each operation records its estimated cycle
                cost as it executes.  When ``None`` (default), latency
                tracking is disabled and there is zero overhead.
            trace_latency: If ``True`` (and *latency_config* is set),
                record a per-operation trace in addition to aggregate
                counters.  Useful for inspecting the cost of individual
                operations.
            parser: Parser to use in ``load()``.  Defaults to ``KTIRParser``
                when ``None``.  Any object satisfying ``KTIRParserBase``
                (i.e. has ``parse_module(str) -> IRModule``) is accepted.
        """
        self.module: Optional[IRModule] = None
        self.memory: Optional[SpyreMemoryHierarchy] = None
        self.grid_executor: Optional[GridExecutor] = None
        self.ring_network: Optional[RingNetwork] = None
        self._env: Optional[ExecutionEnv] = None
        self._parser: Optional[KTIRParserBase] = parser
        self._latency_tracker: Optional[LatencyTracker] = (
            LatencyTracker(latency_config, trace=trace_latency)
            if latency_config is not None else None
        )

    def load(self, ktir_source: str):
        """Load and parse KTIR from a file path or MLIR text.

        Args:
            ktir_source: Path to a KTIR file, or MLIR text directly.
                File paths are dispatched to ``parse_file``; inline text
                is dispatched to ``parse_module``.
        """
        parser = self._parser if self._parser is not None else KTIRParser()
        if '\n' in ktir_source or ktir_source.lstrip().startswith('module'):
            self.module = parser.parse_module(ktir_source)
        else:
            self.module = parser.parse_file(ktir_source)

    def _prepare_execution(self, grid_shape: Tuple[int, int, int]):
        """Set up execution environment.

        Args:
            grid_shape: (x, y, z) grid dimensions
        """
        num_cores = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.memory = SpyreMemoryHierarchy(num_cores)
        self.grid_executor = GridExecutor(grid_shape, self.memory)
        self.ring_network = RingNetwork(num_cores)
        self._env = ExecutionEnv(
            grid_executor=self.grid_executor,
            ring=self.ring_network,
            execute_region=self.execute_region,
        )
        if self._latency_tracker is not None:
            self._latency_tracker.reset()

    def execute_function(self, func_name: str, **kwargs) -> Dict[str, np.ndarray]:
        """Execute KTIR function with given arguments.

        Args:
            func_name: Function name to execute
            **kwargs: Input tensors (NumPy arrays)

        Returns:
            Dict of output tensors

        Example:
            outputs = interp.execute_function(
                "add_kernel",
                x=np.array([1, 2, 3], dtype=np.float16),
                y=np.array([4, 5, 6], dtype=np.float16),
                z=np.zeros(3, dtype=np.float16),
                n=3
            )
        """
        if not self.module:
            raise RuntimeError("No module loaded. Call load() first.")

        func = self.module.get_function(func_name)
        self._prepare_execution(func.grid)

        # Allocate input tensors in HBM
        input_ptrs = {}
        input_dtypes = {}
        for arg_name, tensor in kwargs.items():
            if isinstance(tensor, np.ndarray):
                ptr = self.memory.hbm.allocate(tensor.nbytes)
                self.memory.hbm.write(ptr, tensor)
                input_ptrs[arg_name] = ptr
                input_dtypes[arg_name] = _ktir_dtype(tensor.dtype)
            else:
                # Scalar argument (like n)
                input_ptrs[arg_name] = tensor

        # Execute on all cores
        def execute_on_core(core: CoreContext, ring: RingNetwork):
            # Initialize context with function arguments
            for arg_name, arg_val in input_ptrs.items():
                core.set_value("%" + arg_name, arg_val)

            # Execute function body
            for op in func.operations:
                self._execute_operation(op, core, ring)

        # Execute with communication support
        self.grid_executor.execute_with_communication(
            execute_on_core,
            self.ring_network,
            max_rounds=10
        )

        # Read output tensors from HBM
        outputs = {}
        for arg_name, tensor in kwargs.items():
            if isinstance(tensor, np.ndarray):
                ptr = input_ptrs[arg_name]
                n_elements = math.prod(tensor.shape)
                output_data = self.memory.hbm.read(ptr, n_elements, input_dtypes[arg_name]).reshape(tensor.shape)
                outputs[arg_name] = output_data

        return outputs

    def _execute_operation(self, op: Operation, context: CoreContext, ring: RingNetwork) -> Any:
        """Execute a single operation.

        Args:
            op: Operation to execute
            context: Core execution context
            ring: Ring network

        Returns:
            Operation result
        """
        result = None

        # Resolve operand values for latency tracking (cheap dict lookups)
        resolved_operands = None
        if self._latency_tracker is not None:
            resolved_operands = []
            for name in op.operands:
                try:
                    resolved_operands.append(context.get_value(name))
                except KeyError:
                    resolved_operands.append(None)

        try:
            handler = dispatch(op.op_type)
            if handler:
                result = handler(op, context, self._env)
            else:
                raise ValueError(f"Unknown operation: {op.op_type}")

        except Exception as e:
            print(f"Error executing {op.op_type} on core {context.core_id}: {e}")
            raise

        # Store result in context.
        # Only Tile values (tensor data backed by NumPy arrays) occupy LX.
        # Other result types — TileRef (metadata), AccessTile (coordinates),
        # int/index (scalars) — are bookkeeping and use no scratchpad memory.
        if op.result and result is not None:
            # Multi-result ops return a tuple and have a list of result names.
            if isinstance(op.result, list) and isinstance(result, tuple):
                for name, val in zip(op.result, result):
                    context.set_value(name, val)
            else:
                context.set_value(op.result, result)
                if isinstance(result, Tile):
                    context.track_lx(op.result, result.size_bytes())

        # Record latency
        if self._latency_tracker is not None:
            self._latency_tracker.record_op(
                context.core_id, op.op_type, result, resolved_operands
            )

        return result

    def tensor_input_output_sizes(self, func_name: str) -> Dict[str, Dict[str, Any]]:
        """Return shape and dtype for each tensor argument of a function.

        Args:
            func_name: Name of the function to inspect.

        Returns:
            Dict mapping argument name (without ``%``) to
            ``{"shape": tuple[int, ...], "dtype": str}``.

        Example::

            sizes = interp.tensor_input_output_sizes("add_kernel")
            # {"x_ptr": {"shape": (1024,), "dtype": "f16"}, ...}
        """
        if not self.module:
            raise RuntimeError("No module loaded. Call load() first.")
        return self.module.get_function(func_name).tensor_sizes

    def get_latency_report(self) -> Optional[LatencyReport]:
        """Return latency report, or None if latency tracking is disabled."""
        if self._latency_tracker is None:
            return None
        return self._latency_tracker.report()

    def execute_region(self, context: CoreContext, operations: List[Operation]) -> Any:
        """Execute a region (list of operations).

        Args:
            context: Core execution context
            operations: List of operations to execute

        Returns:
            Result from last operation
        """
        result = None
        for op in operations:
            result = self._execute_operation(op, context, self.ring_network)
        return result
