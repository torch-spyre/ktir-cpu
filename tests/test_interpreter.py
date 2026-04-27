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
Tests for interpreter edge cases and previously uncovered paths.

Covers:
- Scalar (non-NumPy) arguments to execute_function
- execute_region in isolation
- Unknown op dispatch (warning, no crash)
- Multi-result operation unpacking
"""

import numpy as np
import pytest

from ktir_cpu.interpreter import KTIRInterpreter
from ktir_cpu.ir_types import Operation
from ktir_cpu.grid import CoreContext
from ktir_cpu.memory import SpyreMemoryHierarchy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interpreter_with_module(ktir_text: str) -> KTIRInterpreter:
    interp = KTIRInterpreter()
    interp.load(ktir_text)
    return interp


def _minimal_core(interp: KTIRInterpreter) -> CoreContext:
    """Set up execution environment and return a CoreContext for core 0."""
    interp._prepare_execution((1, 1, 1))
    return interp.grid_executor.cores[0]


# ---------------------------------------------------------------------------
# 1. Scalar (non-NumPy) argument handling
# ---------------------------------------------------------------------------

_SCALAR_KTIR = """
module {
    func.func @scalar_fn(%n: index) -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"""


def test_execute_function_scalar_arg():
    """Non-ndarray kwargs are stored as plain values (not allocated in HBM)."""
    interp = _make_interpreter_with_module(_SCALAR_KTIR)
    # Should not raise; scalar 'n' takes the else-branch in execute_function.
    outputs = interp.execute_function("scalar_fn", n=42)
    # Scalar args are not echoed in outputs (only ndarray args are).
    assert "n" not in outputs


def test_execute_function_scalar_and_array_args():
    """Mixed scalar + ndarray args: array is in outputs, scalar is not."""
    ktir = """
module {
    func.func @mixed(%buf: memref<4xf16, "HBM">, %n: index) -> ()
            attributes { grid = [1, 1, 1] } {
        return
    }
}
"""
    interp = _make_interpreter_with_module(ktir)
    buf = np.zeros(4, dtype=np.float16)
    outputs = interp.execute_function("mixed", buf=buf, n=7)
    assert "buf" in outputs
    assert "n" not in outputs


def test_execute_function_scalar_int_and_float():
    """Both int and float scalars are accepted without error."""
    interp = _make_interpreter_with_module(_SCALAR_KTIR)
    interp.execute_function("scalar_fn", n=0)
    interp.execute_function("scalar_fn", n=3.14)


# ---------------------------------------------------------------------------
# 2. execute_region in isolation
# ---------------------------------------------------------------------------

def test_execute_region_empty():
    """execute_region with an empty operation list returns None."""
    interp = _make_interpreter_with_module(_SCALAR_KTIR)
    core = _minimal_core(interp)
    result = interp.execute_region(core, [])
    assert result is None


def test_execute_region_single_op():
    """execute_region executes ops and returns the last result."""
    ktir = """
module {
    func.func @dummy() -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"""
    interp = _make_interpreter_with_module(ktir)
    core = _minimal_core(interp)

    # Build a minimal arith.constant op manually.
    op = Operation(
        result="%c",
        op_type="arith.constant",
        operands=[],
        attributes={"value": 99},
        result_type="index",
        regions=[],
    )
    result = interp.execute_region(core, [op])
    # The constant handler stores and returns 99.
    assert result == 99
    assert core.get_value("%c") == 99


def test_execute_region_multiple_ops_returns_last():
    """execute_region returns the result of the final op."""
    ktir = """
module {
    func.func @dummy() -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"""
    interp = _make_interpreter_with_module(ktir)
    core = _minimal_core(interp)

    ops = [
        Operation(result="%a", op_type="arith.constant", operands=[], attributes={"value": 1}, result_type="index", regions=[]),
        Operation(result="%b", op_type="arith.constant", operands=[], attributes={"value": 2}, result_type="index", regions=[]),
    ]
    result = interp.execute_region(core, ops)
    assert result == 2


# ---------------------------------------------------------------------------
# 3. Unknown op dispatch — warning, no exception
# ---------------------------------------------------------------------------

def test_unknown_op_raises():
    """An unregistered op_type raises ValueError."""
    ktir = """
module {
    func.func @dummy() -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"""
    interp = _make_interpreter_with_module(ktir)
    core = _minimal_core(interp)

    unknown_op = Operation(
        result=None,
        op_type="totally.unknown_op",
        operands=[],
        attributes={},
        result_type=None,
        regions=[],
    )
    with pytest.raises(ValueError, match="totally.unknown_op"):
        interp._execute_operation(unknown_op, core, interp.ring_network)


# ---------------------------------------------------------------------------
# 4. Multi-result operation unpacking
# ---------------------------------------------------------------------------

def test_multi_result_tuple_unpacked():
    """When op.result is a list and handler returns tuple, each name is set."""
    from unittest.mock import patch
    from ktir_cpu.dialects import registry

    ktir = """
module {
    func.func @dummy() -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"""
    interp = _make_interpreter_with_module(ktir)
    core = _minimal_core(interp)

    # Register a temporary multi-result handler.
    FAKE_OP = "test.multi_result"

    def fake_handler(op, ctx, env):
        return (10, 20)

    with patch.dict(registry._REGISTRY, {FAKE_OP: fake_handler}):
        op = Operation(
            result=["%x", "%y"],
            op_type=FAKE_OP,
            operands=[],
            attributes={},
            result_type=None,
            regions=[],
        )
        interp._execute_operation(op, core, interp.ring_network)

    assert core.get_value("%x") == 10
    assert core.get_value("%y") == 20


def test_multi_result_single_value_raises_error():
    """When op.result is a list but handler returns a non-tuple, a TypeError is raised.

    This documents the current (unguarded) behavior: the interpreter tries to
    use the list as a dict key in set_value, which is unhashable.  This is a
    known edge-case that callers must avoid by always returning a tuple from
    multi-result handlers.
    """
    from unittest.mock import patch
    from ktir_cpu.dialects import registry

    ktir = """
module {
    func.func @dummy() -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"""
    interp = _make_interpreter_with_module(ktir)
    core = _minimal_core(interp)

    FAKE_OP = "test.single_not_tuple"

    def fake_handler(op, ctx, env):
        return 42  # not a tuple — mismatched with list result

    with patch.dict(registry._REGISTRY, {FAKE_OP: fake_handler}):
        op = Operation(
            result=["%a", "%b"],
            op_type=FAKE_OP,
            operands=[],
            attributes={},
            result_type=None,
            regions=[],
        )
        with pytest.raises(TypeError):
            interp._execute_operation(op, core, interp.ring_network)
