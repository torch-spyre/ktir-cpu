"""
Adapter tests: same assertions as test_examples.py, but driven through
MLIRFrontendParser instead of the regex parser.

Each TestXxxAdapt class inherits the corresponding TestXxxExecution base.
MLIRFrontendInterpMixin overrides _make_interp() to inject MLIRFrontendParser.
"""

from ktir_cpu import KTIRInterpreter
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

from test_examples import (
    TestVectorAddExecution as _TestVectorAddExecution,
    TestSoftmaxExecution as _TestSoftmaxExecution,
    TestLayerNormExecution as _TestLayerNormExecution,
    TestReduceExplicitRegion as _TestReduceExplicitRegion,
    TestMatMulExecution as _TestMatMulExecution,
    TestIndexedAddExecution as _TestIndexedAddExecution,
    TestSdpaExecution as _TestSdpaExecution,
    TestPagedAttentionExecution as _TestPagedAttentionExecution,
)


class MLIRFrontendInterpMixin:
    """Override _make_interp to inject MLIRFrontendParser."""

    def _make_interp(self):
        return KTIRInterpreter(parser=MLIRFrontendParser())


class TestVectorAddAdapt(MLIRFrontendInterpMixin, _TestVectorAddExecution):
    """Vector add tests via MLIRFrontendParser."""


class TestSoftmaxAdapt(MLIRFrontendInterpMixin, _TestSoftmaxExecution):
    """Softmax tests via MLIRFrontendParser."""


class TestLayerNormAdapt(MLIRFrontendInterpMixin, _TestLayerNormExecution):
    """Layer norm tests via MLIRFrontendParser."""


class TestReduceExplicitRegionAdapt(MLIRFrontendInterpMixin, _TestReduceExplicitRegion):
    """Reduce explicit region tests via MLIRFrontendParser."""


class TestMatMulAdapt(MLIRFrontendInterpMixin, _TestMatMulExecution):
    """MatMul tests via MLIRFrontendParser."""


class TestIndexedAddAdapt(MLIRFrontendInterpMixin, _TestIndexedAddExecution):
    """Indexed add tests via MLIRFrontendParser."""


class TestSdpaAdapt(MLIRFrontendInterpMixin, _TestSdpaExecution):
    """SDPA tests via MLIRFrontendParser."""


class TestPagedAttentionAdapt(MLIRFrontendInterpMixin, _TestPagedAttentionExecution):
    """Paged attention tests via MLIRFrontendParser."""
