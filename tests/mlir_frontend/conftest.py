import pytest

from conftest import get_test_params  # noqa: F401 — re-exported for subpackage tests

try:
    import mlir_ktdp  # noqa: F401
except ImportError:
    pytest.skip("mlir_ktdp not installed", allow_module_level=True)


@pytest.fixture(autouse=True)
def _skip_regex_only(request):
    if request.node.get_closest_marker("regex_only"):
        pytest.skip("regex-parser only")
