"""
Adapter tests: the notebook generator's correctness assertions, driven through
MLIRFrontendParser instead of the regex parser.

Only the execution class is adapted.  The roofline, cross-core-reduce and
precondition classes assert costs and generator behaviour, neither of which
depends on which parser read the module, so running them twice would buy nothing.
"""

from ktir_cpu import KTIRInterpreter
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

from test_notebook_generators import (
    HW,
    TestSdpaDecodePvExecution as _TestSdpaDecodePvExecution,
)


class NotebookMLIRFrontendMixin:
    """Override _make_interp to inject MLIRFrontendParser.

    Deliberately not test_examples_adapt.MLIRFrontendInterpMixin: this one also
    carries the notebook's latency config, so the adapted class runs the same code
    path as the regex-parser original, including the fold's cost accounting.
    """

    def _make_interp(self):
        return KTIRInterpreter(latency_config=HW, parser=MLIRFrontendParser())


class TestSdpaDecodePvAdapt(NotebookMLIRFrontendMixin, _TestSdpaDecodePvExecution):
    """Decode P@V correctness via MLIRFrontendParser."""
