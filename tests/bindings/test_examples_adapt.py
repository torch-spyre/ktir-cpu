"""
Adapter tests: same assertions as test_examples.py, but driven through
MLIRBindingsParser instead of the regex parser.

Each TestXxxAdapt class inherits the corresponding TestXxxExecution base.
BindingsInterpMixin overrides _make_interp() to inject MLIRBindingsParser.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ktir_cpu import KTIRInterpreter
from ktir_cpu.bindings.parser import MLIRBindingsParser

from test_examples import (  # noqa: E402
    TestVectorAddExecution as _TestVectorAddExecution,
    TestSoftmaxExecution as _TestSoftmaxExecution,
    TestLayerNormExecution as _TestLayerNormExecution,
    TestReduceExplicitRegion as _TestReduceExplicitRegion,
    TestMatMulExecution as _TestMatMulExecution,
    TestIndexedAddExecution as _TestIndexedAddExecution,
    TestSdpaExecution as _TestSdpaExecution,
    TestPagedAttentionExecution as _TestPagedAttentionExecution,
)


class BindingsInterpMixin:
    """Override _make_interp to inject MLIRBindingsParser."""

    def _make_interp(self):
        return KTIRInterpreter(parser=MLIRBindingsParser())


class TestVectorAddAdapt(BindingsInterpMixin, _TestVectorAddExecution):
    """Vector add tests via MLIRBindingsParser."""


class TestSoftmaxAdapt(BindingsInterpMixin, _TestSoftmaxExecution):
    """Softmax tests via MLIRBindingsParser."""


class TestLayerNormAdapt(BindingsInterpMixin, _TestLayerNormExecution):
    """Layer norm tests via MLIRBindingsParser."""


class TestReduceExplicitRegionAdapt(BindingsInterpMixin, _TestReduceExplicitRegion):
    """Reduce explicit region tests via MLIRBindingsParser."""


class TestMatMulAdapt(BindingsInterpMixin, _TestMatMulExecution):
    """MatMul tests via MLIRBindingsParser."""


class TestIndexedAddAdapt(BindingsInterpMixin, _TestIndexedAddExecution):
    """Indexed add tests via MLIRBindingsParser."""


class TestSdpaAdapt(BindingsInterpMixin, _TestSdpaExecution):
    """SDPA tests via MLIRBindingsParser."""


class TestPagedAttentionAdapt(BindingsInterpMixin, _TestPagedAttentionExecution):
    """Paged attention tests via MLIRBindingsParser."""
