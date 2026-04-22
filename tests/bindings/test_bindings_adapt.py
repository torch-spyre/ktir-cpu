"""
Adapter tests: same assertions as test_dialects_parse.py, but driven through
MLIRBindingsParser instead of the regex parser.

Each ``TestXxxAdapt`` class inherits the corresponding ``TestXxxParsers`` base
and overrides only tests that rely on regex-parser-specific syntax not accepted
by MLIR (overridden to ``pytest.skip``).  Attribute normalisation (e.g.
arith.cmpi integer predicate → string) is handled by MLIRTypeAdapter handlers,
so the inherited assertions pass unchanged.
"""

import re
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_dialects_parse import (  # noqa: E402
    TestArithParsers as _TestArithParsers,
    TestLinalgParsers as _TestLinalgParsers,
    TestTensorParsers as _TestTensorParsers,
    TestKtdpParsers as _TestKtdpParsers,
    TestScfParsers as _TestScfParsers,
)

from ktir_cpu.bindings.parser import MLIRBindingsParser  # noqa: E402

# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class BindingsParseTestMixin:
    """Override _parse to drive tests through MLIRBindingsParser."""

    def assert_operand_names(self, op, *names):
        pass  # bindings parser uses positional %argN names — not portable

    def assert_attribute(self, op, key, value, transform=None):
        if key in ("iter_var", "iter_args"):
            # Bindings parser assigns positional names; e.g. for:
            #   func.func @_test(%lb: index, %ub: index, %step: index) {
            #     scf.for %i = %lb to %ub step %step { ... }
            # key="iter_var", op.attributes={"iter_var": "%i"}        (regex)
            # key="iter_var", op.attributes={"iter_var": "%arg3"}     (bindings, %arg0-2 are func args)
            assert key in op.attributes
        else:
            super().assert_attribute(op, key, value, transform=transform)

    def _parse(self, op_text, parse_ctx=None, args=None):
        args = self._resolve_args(op_text, args)
        sig = ", ".join(f"{n}: {t}" for n, t in args.items())
        module_text = f"""\
module {{
  func.func @_test({sig}) attributes {{ grid = [1] }} {{
    {op_text}
    return
  }}
}}
"""
        ir_module = MLIRBindingsParser().parse_module(module_text)
        for op in ir_module.get_function("_test").operations:
            if op.op_type not in ("func.return", "return"):
                return op
        raise RuntimeError(f"No target op found in:\n{module_text}")


# ---------------------------------------------------------------------------
# Arith
# ---------------------------------------------------------------------------

class TestArithAdapt(BindingsParseTestMixin, _TestArithParsers):
    """Arith tests via MLIRBindingsParser."""


# ---------------------------------------------------------------------------
# Linalg
# ---------------------------------------------------------------------------

class TestLinalgAdapt(BindingsParseTestMixin, _TestLinalgParsers):
    """Linalg tests via MLIRBindingsParser."""


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class TestTensorAdapt(BindingsParseTestMixin, _TestTensorParsers):
    """Tensor tests via MLIRBindingsParser."""


# ---------------------------------------------------------------------------
# Ktdp
# ---------------------------------------------------------------------------

class TestKtdpAdapt(BindingsParseTestMixin, _TestKtdpParsers):
    """Ktdp tests via MLIRBindingsParser."""


    # test_construct_access_tile: inherited

    # test_construct_access_tile_non_index_elem_type_rejected: inherited
    # test_construct_access_tile_malformed_type_rejected: inherited


# ---------------------------------------------------------------------------
# Scf
# ---------------------------------------------------------------------------


class TestScfAdapt(BindingsParseTestMixin, _TestScfParsers):
    """Scf tests via MLIRBindingsParser."""
