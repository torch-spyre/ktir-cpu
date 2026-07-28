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
# Arith
# ---------------------------------------------------------------------------

class TestArithAdapt(MLIRFrontendParseTestMixin, _TestArithParsers):
    """Arith tests via MLIRFrontendParser."""


# ---------------------------------------------------------------------------
# Linalg
# ---------------------------------------------------------------------------

class TestLinalgAdapt(MLIRFrontendParseTestMixin, _TestLinalgParsers):
    """Linalg tests via MLIRFrontendParser."""

    def test_generic_indexing_maps_and_iterator_types(self):
        """Verify _adapt_linalg_generic parses indexing_maps and iterator_types."""
        op = self._parse(
            "%r = linalg.generic {\n"
            "    indexing_maps = [\n"
            "      affine_map<(d0, d1, d2) -> (d0, d2)>,\n"
            "      affine_map<(d0, d1, d2) -> (d2, d1)>,\n"
            "      affine_map<(d0, d1, d2) -> (d0, d1)>\n"
            "    ],\n"
            '    iterator_types = ["parallel", "parallel", "reduction"]\n'
            "  } ins(%a, %b : tensor<3x5xf32>, tensor<5x4xf32>)\n"
            "    outs(%c : tensor<3x4xf32>) {\n"
            "  ^bb0(%aa: f32, %bb: f32, %cc: f32):\n"
            "    %mul = arith.mulf %aa, %bb : f32\n"
            "    %add = arith.addf %mul, %cc : f32\n"
            "    linalg.yield %add : f32\n"
            "  } -> tensor<3x4xf32>",
            args={"%a": "tensor<3x5xf32>", "%b": "tensor<5x4xf32>",
                  "%c": "tensor<3x4xf32>"},
        )
        assert op.op_type == "linalg.generic"
        assert op.attributes["n_ins"] == 2
        # indexing_maps should be AffineMap objects
        maps = op.attributes["indexing_maps"]
        assert len(maps) == 3
        assert maps[0].n_dims == 3
        # iterator_types should be parsed
        assert op.attributes["iterator_types"] == ["parallel", "parallel", "reduction"]


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
