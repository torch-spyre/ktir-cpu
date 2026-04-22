import importlib.util as _ilu
import os
import pytest

_parent = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conftest.py")
_spec = _ilu.spec_from_file_location("_parent_conftest", _parent)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_test_params = _mod.get_test_params

try:
    import mlir_ktdp  # noqa: F401
except ImportError:
    pytest.skip("mlir_ktdp not installed", allow_module_level=True)
