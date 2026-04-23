"""
MLIRFrontendParser adapter tests for indirect access patterns.

Covers MLIR-frontend-specific concerns for construct_indirect_access_tile
that are not exercised by the generic test_examples_adapt suite (which
validates numerical correctness against NumPy references).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ktir_cpu import KTIRInterpreter
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser
from conftest import get_test_params


# ---------------------------------------------------------------------------
# intermediate_vars naming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,func_name,entry", get_test_params("indexed_add_kernel"))
def test_intermediate_vars_names_irrelevant(path, func_name, entry):
    """Scrambling intermediate_vars names does not change execution results.

    The MLIR bindings don't preserve user-given variable names (block args
    are renumbered to %argN), so the adapter generates canonical ['d0', ...]
    names.  The executor uses these names only for SSA-scalar lookups, which
    always fail for iteration variables — they are loop coordinates enumerated
    by variables_space_set, not SSA-defined values.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 64, 8, 128)).astype(np.float16)
    y = rng.standard_normal((2, 32, 8, 128)).astype(np.float16)
    index = np.array([3, 7], dtype=np.int64)

    def run_with_patched_names(var_names):
        interp = KTIRInterpreter(parser=MLIRFrontendParser())
        interp.load(path)
        func = interp.module.get_function(func_name)
        for op in func.operations:
            if "indirect" in op.op_type:
                op.attributes["intermediate_vars"] = var_names
        args = interp.arg_names(func_name)
        output = np.zeros((2, 32, 8, 128), dtype=np.float16)
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        outputs = interp.execute_function(
            func_name,
            **{args[0]: x, args[1]: y, args[2]: index,
               args[3]: output, args[4]: kwargs.get("dim1_start", 0)},
        )
        return outputs[args[3]]

    canonical = run_with_patched_names(["d0", "d1", "d2", "d3"])
    scrambled = run_with_patched_names(["zz0", "zz1", "zz2", "zz3"])
    reversed_ = run_with_patched_names(["d3", "d2", "d1", "d0"])

    np.testing.assert_array_equal(canonical, scrambled)
    np.testing.assert_array_equal(canonical, reversed_)
