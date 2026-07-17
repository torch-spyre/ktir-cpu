"""Frontend adapter test for ktdp.construct_distributed_memory_view.

Mirrors tests/test_distributed_view.py::test_distributed_view_copy_rfc but
drives parsing through MLIRFrontendParser instead of the regex parser.
"""

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_distributed_view import _write_strided
from conftest import get_test_params


@pytest.mark.parametrize("path,func_name,entry", get_test_params("distributed_view_copy"))
def test_distributed_view_copy_rfc_frontend(path, func_name, entry):
    """construct_distributed_memory_view via MLIRFrontendParser - RFC §C.3."""
    interp = KTIRInterpreter(parser=MLIRFrontendParser())
    interp.load(path)
    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        lx0 = interp.memory.get_lx(0)
        lx1 = interp.memory.get_lx(1)
        full = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
        hbm.write(0, full[0:96, :].flatten())
        _write_strided(lx0, 12288, full[96:128, :].copy(), strides=[1, 64])
        lx1.write(16384 * 2, full[128:192, :].flatten())
        hbm.write(24576 * 2 // hbm.STICK_BYTES, np.zeros(192 * 64, dtype=np.float16))
        lx0.next_ptr = 16384 * 2 + 8128
        lx1.next_ptr = 16384 * 2 + 8192

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)

    expected = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
    b = interp.memory.hbm.read(
        24576 * 2 // interp.memory.hbm.STICK_BYTES, 192 * 64, "f16"
    ).reshape(192, 64)
    np.testing.assert_array_equal(b, expected)
