import pytest

from conftest import get_test_params  # noqa: F401 — re-exported for subpackage tests

try:
    import mlir_ktdp  # noqa: F401
except ImportError:
    pytest.skip("mlir_ktdp not installed", allow_module_level=True)
