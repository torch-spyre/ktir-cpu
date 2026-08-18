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
    TestVectorAddDynamicExecution as _TestVectorAddDynamicExecution,
    TestSoftmaxExecution as _TestSoftmaxExecution,
    TestLayerNormExecution as _TestLayerNormExecution,
    TestReduceExplicitRegion as _TestReduceExplicitRegion,
    TestMatMulExecution as _TestMatMulExecution,
    TestIndexedAddExecution as _TestIndexedAddExecution,
    TestSdpaExecution as _TestSdpaExecution,
    TestPagedAttentionExecution as _TestPagedAttentionExecution,
    TestScalarBroadcastExecution as _TestScalarBroadcastExecution,
    TestRoPEExecution as _TestRoPEExecution,
    TestRMSNormExecution as _TestRMSNormExecution,
    TestRMSNorm2x2Execution as _TestRMSNorm2x2Execution,
)


class MLIRFrontendInterpMixin:
    """Override _make_interp to inject MLIRFrontendParser."""

    def _make_interp(self):
        return KTIRInterpreter(parser=MLIRFrontendParser())


class TestVectorAddAdapt(MLIRFrontendInterpMixin, _TestVectorAddExecution):
    """Vector add tests via MLIRFrontendParser."""


class TestVectorAddDynamicAdapt(MLIRFrontendInterpMixin, _TestVectorAddDynamicExecution):
    """Dynamic vector add (memref<?>, symbolic coordinate set) via MLIRFrontendParser."""


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


class TestScalarBroadcastAdapt(MLIRFrontendInterpMixin, _TestScalarBroadcastExecution):
    """Scalar broadcast (rank-0 collapse) via MLIRFrontendParser."""


class TestRoPEAdapt(MLIRFrontendInterpMixin, _TestRoPEExecution):
    """RoPE tests via MLIRFrontendParser."""
class TestRMSNormAdapt(MLIRFrontendInterpMixin, _TestRMSNormExecution):
    """RMSNorm 4x1 tests via MLIRFrontendParser."""


class TestRMSNorm2x2Adapt(MLIRFrontendInterpMixin, _TestRMSNorm2x2Execution):
    """RMSNorm 2x2 tests via MLIRFrontendParser — inherits correctness test from base."""

    def _make_interp(self):
        from ktir_cpu.latency import HardwareConfig
        return KTIRInterpreter(parser=MLIRFrontendParser(), latency_config=HardwareConfig())
