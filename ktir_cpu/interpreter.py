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

import inspect
from typing import Dict, Generator, Tuple, Optional, Any, List
import numpy as np
import math

from .ir_types import Operation, IRModule, Tile
from .parser import KTIRParser, KTIRParserBase
from .memory import SpyreMemoryHierarchy
from .grid import GridExecutor, CoreContext
from .ops.comm_ops import InstantTransferBackend, TransferBackend
from .latency import HardwareConfig, LatencyTracker, LatencyReport
from .dialects import dispatch, ExecutionEnv


from .dtypes import to_ktir_dtype as _ktir_dtype, stick_to_elem_idx as _stick_to_elem_idx
from .memory import HBMSimulator
from .ops.memory_ops import hbm_read, hbm_write


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
        # TODO: rename ring_backend → transfer_backend across the codebase.
        # Keeping the old name for now; the type is TransferBackend.
        self.ring_backend: Optional[TransferBackend] = None
        self._env: Optional[ExecutionEnv] = None
        self._parser: Optional[KTIRParserBase] = parser
        self._latency_config: Optional[HardwareConfig] = latency_config
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
        lx_size_mb = self._latency_config.lx_size_mb if self._latency_config else 2
        self.memory = SpyreMemoryHierarchy(num_cores, lx_size_mb=lx_size_mb)
        # ring_backend (a TransferBackend) serves CoreContext.get_lx() —
        # remote LX peeks for distributed memory views. Passed directly
        # to GridExecutor.execute_with_communication; the scheduler
        # curries it into a per-core transfer_fn via attach_scheduler.
        # Cross-core comm goes through the scheduler protocol
        # (CoreContext.send_to + RecvRequest), not through a backend.
        # See docs/cross_core_scheduling.md.
        self.ring_backend = InstantTransferBackend(self.memory)
        self.grid_executor = GridExecutor(grid_shape, self.memory)
        self._env = ExecutionEnv(
            grid_executor=self.grid_executor,
            execute_region=self.execute_region,
            execute_region_with_comms=self.execute_region_with_comms,
        )
        if self._latency_tracker is not None:
            self._latency_tracker.reset()
            self._latency_tracker.set_cores_active(num_cores)

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

        # If the parser normalised argument names (e.g. mlir-frontend uses
        # %arg0, %arg1, ...) the caller's kwargs may use the original source
        # names.  Remap positionally when none of the supplied keys appear in
        # the declared arg list but the counts match.
        declared = func.arg_names  # list without leading %
        if kwargs and declared and not any(k in declared for k in kwargs):
            if len(kwargs) == len(declared):
                kwargs = dict(zip(declared, kwargs.values()))

        # Allocate input tensors in HBM.
        # input_byte_ptrs: byte address for read-back (use case B).
        # input_ptrs: element index for the kernel SSA env (MemRef.base_ptr contract).
        input_byte_ptrs = {}
        input_ptrs = {}
        for arg_name, tensor in kwargs.items():
            if isinstance(tensor, np.ndarray):
                dtype = _ktir_dtype(tensor.dtype)
                stick = self.memory.hbm.allocate(tensor.nbytes)
                byte_addr = stick * HBMSimulator.STICK_BYTES
                hbm_write(self.memory.hbm, byte_addr, tensor.flatten())
                input_byte_ptrs[arg_name] = byte_addr
                # MLIR pointer operands are element indices.
                input_ptrs[arg_name] = _stick_to_elem_idx(stick, dtype)
            else:
                # Scalar argument (like n)
                input_ptrs[arg_name] = tensor

        for core in self.grid_executor.cores:
            core._use_counts = func.use_counts

        self.grid_executor.execute_with_communication(
            func.operations, input_ptrs, self._execute_op,
            transfer_backend=self.ring_backend,
        )

        # Read output tensors from HBM via byte address (use case B).
        outputs = {}
        for arg_name, tensor in kwargs.items():
            if isinstance(tensor, np.ndarray):
                byte_addr = input_byte_ptrs[arg_name]
                n_elements = math.prod(tensor.shape)
                dtype = _ktir_dtype(tensor.dtype)
                output_data = hbm_read(
                    self.memory.hbm, byte_addr, n_elements, dtype
                ).reshape(tensor.shape)
                outputs[arg_name] = output_data

        return outputs

    def _execute_op(self, op: Operation, context: CoreContext) -> Any:
        """Execute a single operation.

        Args:
            op: Operation to execute
            context: Core execution context

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
                    resolved_operands.append(context.get_value(name, peek=True))
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

        # Structural invariant: accumulating ops must return the same outs object.
        # Only applies to ops where result = f(ins) + outs (in-place accumulation).
        # Not enforced when outs is an uncharged (0-LX literal) tile.
        if (op.outs_operands and not op.regions
                and result is not None and not inspect.isgenerator(result)):
            for outs_name in op.outs_operands:
                try:
                    outs_tile = context.get_value(outs_name, peek=True)
                except KeyError:
                    continue
                if (isinstance(outs_tile, Tile) and isinstance(result, Tile)
                        and id(outs_tile) not in context._uncharged_tiles):
                    assert id(result) == id(outs_tile), (
                        f"{op.op_type}: handler returned new Tile instead of "
                        f"mutating outs {outs_name!r}"
                    )

        # Store result in context.
        # Only Tile values (tensor data backed by NumPy arrays) occupy LX.
        # Other result types — TileRef (metadata), AccessTile (coordinates),
        # int/index (scalars) — are bookkeeping and use no scratchpad memory.
        # Generator results (comm ops, scf.for/if with comm bodies) are stored
        # after the scheduler drives them to completion — skip here.
        if op.result and result is not None and not inspect.isgenerator(result):
            names = op.result if isinstance(op.result, list) else [op.result]
            values = result if isinstance(result, tuple) else [result]
            if len(names) != len(values):
                raise RuntimeError(
                    f"{op.op_type}: expected {len(names)} result(s), got {len(values)}"
                )
            from .dialects.registry import is_no_lx_charge
            charge = not is_no_lx_charge(op.op_type)
            for name, val in zip(names, values):
                context.set_value(name, val, charge=charge)

        # Record latency.
        # Sync handlers: charge now, with the final value.
        # Generator handlers (comm ops): the handler returned a generator
        # whose body has not yet run.  The "real" result is only known
        # after the scheduler drives ``yield from`` to completion.  Wrap
        # the generator so the record fires once it returns its final
        # value.  Skip wrapping entirely when the tracker is off, so the
        # cost-free path stays cost-free.
        if self._latency_tracker is not None:
            if inspect.isgenerator(result):
                result = self._wrap_latency_counter(
                    result, op, context, resolved_operands)
            else:
                self._latency_tracker.record_op(
                    context.core_id, op.op_type, result, resolved_operands
                )

        return result

    def _wrap_latency_counter(
        self,
        gen: Generator,
        op: Operation,
        context: CoreContext,
        resolved_operands: List[Any],
    ) -> Generator:
        """Wrap a handler-returned generator so ``record_op`` fires after
        the generator yields its final value.

        Used for comm ops (`ktdp.inter_tile_reduce`, etc.) whose handler
        returns a generator whose body runs only after the scheduler
        drives ``yield from``.  The wrapper preserves the original
        generator's ``yield`` protocol (``RecvRequest``s flow through
        unchanged via ``yield from``) and emits the latency record when
        the inner generator returns, with the *real* final result rather
        than the placeholder generator object.
        """
        tracker = self._latency_tracker
        core_id = context.core_id
        op_type = op.op_type

        def _wrapped():
            final = yield from gen
            tracker.record_op(core_id, op_type, final, resolved_operands)
            return final

        return _wrapped()

    def arg_names(self, func_name: str) -> List[str]:
        """Return argument names for a function in declaration order (without %)."""
        if not self.module:
            raise RuntimeError("No module loaded. Call load() first.")
        return self.module.get_function(func_name).arg_names

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
        """Execute a nested region synchronously (scf.for body, scf.if branch, etc.).

        Comm ops cannot appear inside compute-only regions (linalg combiners,
        tensor.generate bodies, ktdp combiner bodies).  For scf regions that
        may contain comm ops use ``execute_region_with_comms`` instead.
        """
        result = None
        for op in operations:
            result = self._execute_op(op, context)
        return result

    def execute_region_with_comms(self, context: CoreContext, operations: List[Operation]):
        """Execute a nested region that may contain comm ops (scf.for / scf.if bodies).

        Generator-aware: if an op returns a generator (comm op), propagates it
        via ``yield from`` so the scheduler can drive it.  The resolved result
        overwrites the placeholder generator binding that ``_execute_op`` stored
        before returning.  When no comm op fires this runs synchronously —
        ``yield from`` on a generator that never yields returns immediately.
        """
        result = None
        for op in operations:
            result = self._execute_op(op, context)
            if inspect.isgenerator(result):
                result = yield from result
                # _execute_op stored the raw generator under op.result before
                # returning it.  Overwrite with the resolved tile now that the
                # scheduler has driven the generator to completion.
                if op.result is not None and result is not None:
                    from .dialects.registry import is_no_lx_charge
                    charge = not is_no_lx_charge(op.op_type)
                    names = op.result if isinstance(op.result, list) else [op.result]
                    values = result if isinstance(result, tuple) else [result]
                    for name, val in zip(names, values):
                        context.set_value(name, val, charge=charge)
        return result
