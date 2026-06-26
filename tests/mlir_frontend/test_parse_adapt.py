"""
Adapter tests: same assertions as test_dialects_parse.py, but driven through
MLIRFrontendParser instead of the regex parser.

Each ``TestXxxAdapt`` class inherits the corresponding ``TestXxxParsers`` base
and overrides only tests that rely on regex-parser-specific syntax not accepted
by MLIR (overridden to ``pytest.skip``).  Attribute normalisation (e.g.
arith.cmpi integer predicate → string) is handled by MLIRTypeAdapter handlers,
so the inherited assertions pass unchanged.
"""

import pytest

from test_dialects_parse import (
    TestArithParsers as _TestArithParsers,
    TestFuncParsers as _TestFuncParsers,
    TestLinalgParsers as _TestLinalgParsers,
    TestTensorParsers as _TestTensorParsers,
    TestKtdpParsers as _TestKtdpParsers,
    TestScfParsers as _TestScfParsers,
    TestMathParsers as _TestMathParsers,
)

from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser  # noqa: E402

# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class MLIRFrontendParseTestMixin:
    """Override _parse to drive tests through MLIRFrontendParser."""

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
        ir_module = MLIRFrontendParser().parse_module(module_text)
        for op in ir_module.get_function("_test").operations:
            if op.op_type not in ("func.return", "return"):
                return op
        raise RuntimeError(f"No target op found in:\n{module_text}")


# ---------------------------------------------------------------------------
# Func
# ---------------------------------------------------------------------------

class TestFuncAdapt(MLIRFrontendParseTestMixin, _TestFuncParsers):
    """Func tests via MLIRFrontendParser.

    func.call references a callee that must exist in the module for MLIR
    verification to pass, so the single-op tests are overridden to embed a
    callee declaration alongside @_test.  The multi-call test uses
    KTIRInterpreter directly and is inherited unchanged.
    """

    def test_void_call_no_args(self):
        module_text = """\
module {
  func.func @callee() attributes {grid = [1]} { return }
  func.func @_test() attributes {grid = [1]} {
    func.call @callee() : () -> ()
    return
  }
}
"""
        ir_module = MLIRFrontendParser().parse_module(module_text)
        calls = [o for o in ir_module.get_function("_test").operations
                 if o.op_type == "func.call"]
        op = calls[0]
        self.assert_op_type(op, "func.call")
        assert op.result is None
        assert op.operands == []
        self.assert_attribute(op, "callee", "callee")

    def test_call_with_args(self):
        module_text = """\
module {
  func.func @add(%x: index, %y: index) attributes {grid = [1]} { return }
  func.func @_test(%x: index, %y: index) attributes {grid = [1]} {
    func.call @add(%x, %y) : (index, index) -> ()
    return
  }
}
"""
        ir_module = MLIRFrontendParser().parse_module(module_text)
        calls = [o for o in ir_module.get_function("_test").operations
                 if o.op_type == "func.call"]
        op = calls[0]
        self.assert_op_type(op, "func.call")
        self.assert_attribute(op, "callee", "add")
        self.assert_num_operands(op, 2)


# ---------------------------------------------------------------------------
# Arith
# ---------------------------------------------------------------------------

class TestArithAdapt(MLIRFrontendParseTestMixin, _TestArithParsers):
    """Arith tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Linalg
# ---------------------------------------------------------------------------

class TestLinalgAdapt(MLIRFrontendParseTestMixin, _TestLinalgParsers):
    """Linalg tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class TestTensorAdapt(MLIRFrontendParseTestMixin, _TestTensorParsers):
    """Tensor tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Ktdp
# ---------------------------------------------------------------------------

class TestKtdpAdapt(MLIRFrontendParseTestMixin, _TestKtdpParsers):
    """Ktdp tests via MLIRFrontendParser."""

    # test_construct_access_tile: inherited
    # test_construct_access_tile_non_index_elem_type_rejected: inherited
    # test_construct_access_tile_malformed_type_rejected: inherited

    # test_affine_set_with_symbolic_dim: inherited
    # test_construct_memory_view_dynamic_memref_type: inherited
    # test_construct_memory_view_ssa_size_as_operand: inherited

    # test_construct_memory_view_multi_dim_mixed_static_dynamic: inherited


# ---------------------------------------------------------------------------
# Scf
# ---------------------------------------------------------------------------


class TestScfAdapt(MLIRFrontendParseTestMixin, _TestScfParsers):
    """Scf tests via MLIRFrontendParser."""

    def test_if_then_else(self):
        # scf.if is supported by the MLIR frontend (the regex parser does not
        # parse it, so this test is frontend-only rather than in the shared
        # base class). operand[0] is the condition; then/else are regions
        # [0]/[1]; no execution-relevant attributes.
        op = self._parse(
            "%r = scf.if %c -> (i32) {\n"
            "      %a = arith.constant 1 : i32\n"
            "      scf.yield %a : i32\n"
            "    } else {\n"
            "      %b = arith.constant 2 : i32\n"
            "      scf.yield %b : i32\n"
            "    }",
            args={"%c": "i1"},
        )
        self.assert_op_type(op, "scf.if")
        self.assert_num_operands(op, 1)
        assert len(op.regions) == 2


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

class TestMathAdapt(MLIRFrontendParseTestMixin, _TestMathParsers):
    """Math tests via MLIRFrontendParser."""
