# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parser tests for the KTIR dialect layer and module-level parser.

Covers:
- Module/function-level parser (moved from test_ktir_cpu.py)
- arith dialect parsers: arith.constant, arith.cmpi, arith.sitofp
- linalg dialect parsers: linalg.reduce, linalg.fill, linalg.broadcast
- tensor dialect parsers: tensor.empty, tensor.splat, tensor.extract, tensor.expand_shape
- ktdp dialect parsers: ktdp.get_compute_tile_id, ktdp.construct_memory_view,
                         ktdp.construct_access_tile
- scf dialect parsers: scf.for, scf.yield

Design rule
-----------
The goal of the regex parser is to parse valid MLIR.  All op text examples
in this file must therefore be valid MLIR.  Non-MLIR syntax forces the
corresponding test_parse_adapt.py test to be skipped, which defeats the
purpose of sharing tests between the two parser backends.

All op text examples have been verified to parse with both backends.
"""

import numpy as np
import pytest

from ktir_cpu.parser import KTIRParser
from ktir_cpu.dialects.registry import dispatch_parser, make_parse_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _parse_ctx():
    return make_parse_context(aliases={})


class ParseTestMixin:
    """Mixin that provides _parse and a portable assertion API.

    Subclasses can override _parse to inject a different parser backend
    (e.g. MLIRBindingsParser).

    Assertion API
    -------------
    Use these methods for all op-level checks in base test classes.
    They define the boundary between parser-agnostic checks (safe to
    inherit) and parser-specific checks (must be overridden or skipped).

    Parser-agnostic — always use these in base tests:
        assert_op_type, assert_num_operands, assert_result_type,
        assert_attribute

    Parser-specific — override in subclasses or avoid in base tests:
        assert_operand_names: checks exact SSA names; only valid for the
        regex parser. BindingsParseTestMixin overrides this to a no-op since
        the MLIR frontend parser uses positional %argN names.
    """

    def _parse(self, op_text, parse_ctx=None, args=None):
        """Parse a single op and return the resulting Operation.

        ``args`` is an optional ``{name: mlir_type}`` mapping that declares
        **all** external SSA operands the op depends on — values defined
        outside the op text that the op references.  Names defined *within*
        the op (results on the LHS of ``=``, region block args like ``%i``
        in ``scf.for %i = ...``, and iter-args) should **not** be included.

        This is a convenience for single-op testing: it avoids writing a
        full MLIR module while still providing explicit types.  Example::

            self._parse(
                "%r = linalg.reduce { arith.maxnumf }"
                " ins(%x : tensor<1x1024xf16>)"
                " outs(%init : tensor<1xf16>)"
                " dimensions = [1]",
                args={"%x": "tensor<1x1024xf16>", "%init": "tensor<1xf16>"},
            )

        The regex parser ignores ``args`` and operates on the op text
        directly.  The MLIR frontend adapter uses it to build a typed
        ``func.func`` wrapper, so **missing operands will cause MLIR
        verification errors** when the adapter tests run.

        If ``args`` is provided, every declared name must appear in
        ``op_text``.
        """
        args = self._resolve_args(op_text, args)
        ctx = parse_ctx or _parse_ctx()
        ops = KTIRParser()._parse_operations(op_text, ctx)
        assert ops, f"No op parsed from: {op_text!r}"
        return ops[0]

    def _resolve_args(self, op_text, args):
        """Normalise and validate the args schema against op_text.

        Returns ``args`` as a ``dict``, defaulting to ``{}`` if not provided.
        Asserts that every declared name appears in ``op_text``.
        """
        args = args or {}
        for name in args:
            assert name in op_text, f"arg {name!r} not found in op_text"
        return args

    def assert_op_type(self, op, expected):
        assert op.op_type == expected

    def assert_num_operands(self, op, n):
        assert len(op.operands) == n

    def assert_result_type(self, op, expected):
        assert op.result_type == expected

    def assert_attribute(self, op, key, value, transform=None):
        actual = op.attributes[key]
        if transform is not None:
            actual = transform(actual)
        assert actual == value

    def assert_operand_names(self, op, *names):
        """Regex-parser-specific: checks exact SSA operand names.
        Override to no-op in MLIR frontend subclasses."""
        for name in names:
            assert name in op.operands


# ---------------------------------------------------------------------------
# module-level parser
# ---------------------------------------------------------------------------

class TestModuleParser:
    def test_parser_basic(self):
        # minimal module with grid attribute parses correctly
        parser = KTIRParser()
        module = parser.parse_module("""
        module {
            func.func @test_func() -> index attributes { grid = [32, 1, 1] } {
                %c0 = arith.constant 0 : index
                %grid0 = ktdp.get_compute_tile_id : index
                return %c0 : index
            }
        }
        """)
        assert "test_func" in module.functions
        func = module.get_function("test_func")
        assert func.grid == (32, 1, 1)
        assert len(func.operations) >= 2

    def test_parser_attributes_body(self):
        # function with arguments, 2-d grid, and ktdp ops all parsed
        parser = KTIRParser()
        module = parser.parse_module("""
        module {
          func.func @add(%a: index, %b: index, %c: index) -> index attributes { grid = [4, 4] } {
            %c0 = arith.constant 0 : index
            %grid0 = ktdp.get_compute_tile_id : index
            %acc = ktdp.construct_access_tile %ref[%c0, %c0] : memref<128x256xf16> -> !ktdp.access_tile<128x256xindex>
            %tile = ktdp.load %acc : !ktdp.access_tile<128x256xindex> -> tensor<128x256xf16>
            %out_acc = ktdp.construct_access_tile %out_ref[%c0, %c0] : memref<128x256xf16> -> !ktdp.access_tile<128x256xindex>
            ktdp.store %tile, %out_acc : tensor<128x256xf16>, !ktdp.access_tile<128x256xindex>
            return %c0 : index
          }
        }
        """)
        assert "add" in module.functions
        func = module.get_function("add")
        assert func.grid == (4, 4, 1)
        assert len(func.arguments) == 3
        op_types = [op.op_type for op in func.operations]
        for expected in ["arith.constant", "ktdp.get_compute_tile_id",
                         "ktdp.construct_access_tile", "ktdp.load", "ktdp.store"]:
            assert expected in op_types

    def test_parser_no_attributes(self):
        # function without attributes defaults to grid (1,1,1)
        parser = KTIRParser()
        module = parser.parse_module("""
        module {
          func.func @simple() -> index {
            %c0 = arith.constant 0 : index
            return %c0 : index
          }
        }
        """)
        func = module.get_function("simple")
        assert func.grid == (1, 1, 1)
        assert len(func.operations) >= 2

    def test_parser_multiple_functions(self):
        # module with two functions each gets its own grid shape
        parser = KTIRParser()
        module = parser.parse_module("""
        module {
          func.func @first() attributes { grid = [8, 4] } {
            %c0 = arith.constant 0 : index
            return
          }
          func.func @second(%x: index) -> index attributes { grid = [16, 2, 1] } {
            %c1 = arith.constant 1 : index
            return %c1 : index
          }
        }
        """)
        assert module.functions["first"].grid == (8, 4, 1)
        assert module.functions["second"].grid == (16, 2, 1)


# ---------------------------------------------------------------------------
# arith dialect parsers
# ---------------------------------------------------------------------------

class TestArithParsers(ParseTestMixin):
    def test_constant_scalar(self):
        # scalar integer constant parsed with correct value
        op = self._parse("%c0 = arith.constant 42 : index")
        assert op.op_type == "arith.constant"
        assert op.attributes["value"] == 42

    def test_constant_hex_integer(self):
        # hex integer constant (e.g. 0xFF800000 for -inf bitcast)
        op = self._parse("%x = arith.constant 0xFF800000 : i32")
        self.assert_op_type(op, "arith.constant")
        # 0xFF800000: regex returns 4286578688 (unsigned), MLIR returns
        # -8388608 (signed i32).  Both are the same bit pattern — accept either.
        val = op.attributes["value"]
        assert val == 0xFF800000 or val == -8388608

    def test_constant_float(self):
        # float constant
        op = self._parse("%x = arith.constant 0.0 : f32")
        self.assert_op_type(op, "arith.constant")
        assert op.attributes["value"] == 0.0

    def test_constant_dense_tensor(self):
        # dense<0.0> tensor constant sets is_tensor, shape, and dtype
        op = self._parse("%t = arith.constant dense<0.0> : tensor<4xf16>")
        self.assert_op_type(op, "arith.constant")
        self.assert_attribute(op, "is_tensor", True)
        self.assert_attribute(op, "shape", (4,))
        self.assert_attribute(op, "dtype", "f16")

    def test_cmpi_basic(self):
        # cmpi records predicate and both operands
        op = self._parse(
            "%b = arith.cmpi slt, %a, %c0 : index",
            args={"%a": "index", "%c0": "index"},
        )
        self.assert_op_type(op, "arith.cmpi")
        self.assert_attribute(op, "predicate", "slt")
        self.assert_num_operands(op, 2)
        self.assert_operand_names(op, "%a", "%c0")

    def test_cmpi_all_predicates(self):
        # all six comparison predicates are recognised
        for pred in ("eq", "ne", "slt", "sle", "sgt", "sge"):
            op = self._parse(
                f"%b = arith.cmpi {pred}, %x, %y : i32",
                args={"%x": "i32", "%y": "i32"},
            )
            self.assert_attribute(op, "predicate", pred)

    def test_sitofp(self):
        # sitofp records operand and target float type
        op = self._parse(
            "%f = arith.sitofp %i : i32 to f16",
            args={"%i": "i32"},
        )
        self.assert_op_type(op, "arith.sitofp")
        self.assert_num_operands(op, 1)
        self.assert_operand_names(op, "%i")
        self.assert_result_type(op, "f16")


# ---------------------------------------------------------------------------
# linalg dialect parsers
# ---------------------------------------------------------------------------

class TestLinalgParsers(ParseTestMixin):
    def test_reduce(self):
        # reduce records reduce_fn, dim, outs_var, and ins operand
        op = self._parse(
            "%r = linalg.reduce { arith.maxnumf }"
            " ins(%x : tensor<1x1024xf16>)"
            " outs(%init : tensor<1xf16>)"
            " dimensions = [1]",
            args={"%x": "tensor<1x1024xf16>", "%init": "tensor<1xf16>"},
        )
        self.assert_op_type(op, "linalg.reduce")
        self.assert_attribute(op, "reduce_fn", "arith.maxnumf")
        self.assert_attribute(op, "dim", 1)
        self.assert_num_operands(op, 1)
        self.assert_operand_names(op, "%x")

    def test_fill(self):
        # fill records both ins and outs operands
        op = self._parse(
            "%out = linalg.fill ins(%val : f16) outs(%buf : tensor<4xf16>) -> tensor<4xf16>",
            args={"%val": "f16", "%buf": "tensor<4xf16>"},
        )
        self.assert_op_type(op, "linalg.fill")
        self.assert_num_operands(op, 2)
        self.assert_operand_names(op, "%val", "%buf")

    def test_broadcast(self):
        # broadcast records dimensions and both ins/outs operands
        op = self._parse(
            "%out = linalg.broadcast ins(%x : tensor<4xf16>) outs(%buf : tensor<4x8xf16>) dimensions = [1]",
            args={"%x": "tensor<4xf16>", "%buf": "tensor<4x8xf16>"},
        )
        self.assert_op_type(op, "linalg.broadcast")
        self.assert_attribute(op, "dimensions", [1])
        self.assert_num_operands(op, 2)
        self.assert_operand_names(op, "%x", "%buf")


# ---------------------------------------------------------------------------
# tensor dialect parsers
# ---------------------------------------------------------------------------

class TestTensorParsers(ParseTestMixin):
    def test_empty(self):
        # tensor.empty records shape and dtype from type annotation
        op = self._parse("%t = tensor.empty() : tensor<1x1024xf16>")
        self.assert_op_type(op, "tensor.empty")
        self.assert_attribute(op, "shape", (1, 1024))
        self.assert_attribute(op, "dtype", "f16")

    def test_splat(self):
        # tensor.splat records scalar operand and target shape
        op = self._parse(
            "%t = tensor.splat %val : tensor<4xf16>",
            args={"%val": "f16"},
        )
        self.assert_op_type(op, "tensor.splat")
        self.assert_num_operands(op, 1)
        self.assert_operand_names(op, "%val")
        self.assert_attribute(op, "shape", (4,))

    def test_extract(self):
        # tensor.extract records tensor operand and index operands
        op = self._parse(
            "%s = tensor.extract %t[%i, %j] : tensor<4x4xf16>",
            args={"%t": "tensor<4x4xf16>", "%i": "index", "%j": "index"},
        )
        self.assert_op_type(op, "tensor.extract")
        self.assert_num_operands(op, 3)
        self.assert_operand_names(op, "%t", "%i", "%j")

    def test_expand_shape(self):
        # tensor.expand_shape records source operand and target shape
        op = self._parse(
            "%out = tensor.expand_shape %in [[0, 1]] output_shape [1, 1024]"
            " : tensor<1024xf16> into tensor<1x1024xf16>",
            args={"%in": "tensor<1024xf16>"},
        )
        self.assert_op_type(op, "tensor.expand_shape")
        self.assert_num_operands(op, 1)
        self.assert_operand_names(op, "%in")
        self.assert_attribute(op, "target_shape", (1, 1024))


# ---------------------------------------------------------------------------
# ktdp dialect parsers
# ---------------------------------------------------------------------------

class TestKtdpParsers(ParseTestMixin):
    def test_get_compute_tile_id_single(self):
        op = self._parse("%id = ktdp.get_compute_tile_id : index")
        self.assert_op_type(op, "ktdp.get_compute_tile_id")
        assert isinstance(op.result, str)

    def test_get_compute_tile_id_multi(self):
        op = self._parse("%x, %y = ktdp.get_compute_tile_id : index, index")
        self.assert_op_type(op, "ktdp.get_compute_tile_id")
        assert isinstance(op.result, list)
        assert len(op.result) == 2

    def test_construct_memory_view(self):
        # construct_memory_view records shape, strides, dtype, memory_space, and pointer operand
        op = self._parse(
            "%view = ktdp.construct_memory_view %ptr, sizes: [1024], strides: [1]"
            " { coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>,"
            " memory_space = #ktdp.spyre_memory_space<HBM> } : memref<1024xf16>",
            args={"%ptr": "index"},
        )
        self.assert_op_type(op, "ktdp.construct_memory_view")
        self.assert_attribute(op, "shape", (1024,))
        self.assert_attribute(op, "strides", [1])
        self.assert_attribute(op, "dtype", "f16")
        self.assert_attribute(op, "memory_space", "HBM")
        self.assert_num_operands(op, 1)
        self.assert_operand_names(op, "%ptr")

    def test_construct_access_tile(self):
        # construct_access_tile records tile shape and all operands
        op = self._parse(
            "%acc = ktdp.construct_access_tile %view[%c0]"
            " { access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,"
            " access_tile_order = affine_map<(d0) -> (d0)> }"
            " : memref<1024xf16> -> !ktdp.access_tile<128xindex>",
            args={"%view": "memref<1024xf16>", "%c0": "index"},
        )
        self.assert_op_type(op, "ktdp.construct_access_tile")
        self.assert_attribute(op, "shape", (128,))
        self.assert_attribute(op, "base_map", "affine_map<(d0) -> (d0)>", transform=lambda x: x.source)
        self.assert_num_operands(op, 2)
        self.assert_operand_names(op, "%view", "%c0")

    def test_construct_access_tile_non_index_elem_type_rejected(self):
        # Per spec, AccessTileType element type must be 'index'; any other type
        # is a spec violation and must be rejected at parse time.
        # The result type below is !ktdp.access_tile<128xf16> — 'f16' must be 'index'.
        # regex raises: ValueError "AccessTileType element type must be 'index', got 'f16'"
        # mlir  raises: MLIRError  "tile element type must be 'index', but got: 'f16'"
        with pytest.raises(Exception, match=r"element type must be 'index'.*f16"):
            self._parse(
                "%acc = ktdp.construct_access_tile %view[%c0]"
                " { access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,"
                " access_tile_order = affine_map<(d0) -> (d0)> }"
                " : memref<1024xf16> -> !ktdp.access_tile<128xf16>",
                args={"%view": "memref<1024xf16>", "%c0": "index"},
            )

    def test_construct_access_tile_malformed_type_rejected(self):
        # The result type !ktdp.access_tile<128> has no element type (must be <128xindex>).
        # Both parsers reject the missing 'x<type>' suffix, with different messages:
        # regex raises: ValueError "Malformed access_tile type '128': expected '<dims>xindex>'"
        # mlir  raises: MLIRError  "expected 'x' in dimension list"
        with pytest.raises(Exception, match=r"Malformed access_tile|expected 'x' in dimension"):
            self._parse(
                "%acc = ktdp.construct_access_tile %view[%c0]"
                " { access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,"
                " access_tile_order = affine_map<(d0) -> (d0)> }"
                " : memref<1024xf16> -> !ktdp.access_tile<128>",
                args={"%view": "memref<1024xf16>", "%c0": "index"},
            )


# ---------------------------------------------------------------------------
# scf dialect parsers
# ---------------------------------------------------------------------------

class TestScfParsers(ParseTestMixin):
    def test_for_basic(self):
        # scf.for records iter_var and lb/ub/step operands in order
        op = self._parse(
            "scf.for %i = %lb to %ub step %step {\n      scf.yield\n    }",
            args={"%lb": "index", "%ub": "index", "%step": "index"},
        )
        self.assert_op_type(op, "scf.for")
        self.assert_attribute(op, "iter_var", "%i")
        self.assert_num_operands(op, 3)
        self.assert_operand_names(op, "%lb", "%ub", "%step")

    def test_for_with_result(self):
        # optional result prefix on scf.for is captured
        op = self._parse(
            "%res = scf.for %i = %lb to %ub step %step iter_args(%acc = %lb) -> (index) {\n"
            "      scf.yield %acc : index\n    }",
            args={"%lb": "index", "%ub": "index", "%step": "index"},
        )
        assert isinstance(op.result, str)
        self.assert_attribute(op, "iter_var", "%i")

    def test_for_iter_args(self):
        # iter_args clause records carried variable and init operand
        op = self._parse(
            "%res = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f16) {\n"
            "      scf.yield %acc : f16\n    }",
            args={"%lb": "index", "%ub": "index", "%step": "index", "%init": "f16"},
        )
        self.assert_attribute(op, "iter_args", ["%acc"])
        self.assert_operand_names(op, "%init")

    def test_for_multi_result(self):
        # multi-result scf.for records a list of result names
        op = self._parse(
            "%M, %L, %acc = scf.for %j = %c0 to %n step %c1"
            " iter_args(%m = %M0, %l = %L0, %a = %A0) -> (f16, f16, f16) {\n"
            "      scf.yield %m, %l, %a : f16, f16, f16\n    }",
            args={"%c0": "index", "%n": "index", "%c1": "index",
                  "%M0": "f16", "%L0": "f16", "%A0": "f16"},
        )
        self.assert_op_type(op, "scf.for")
        assert isinstance(op.result, list)
        assert len(op.result) == 3
        self.assert_attribute(op, "iter_var", "%j")
        self.assert_attribute(op, "iter_args", ["%m", "%l", "%a"])
        # operands = [lb, ub, step, init_m, init_l, init_a]
        self.assert_num_operands(op, 6)
        self.assert_operand_names(op, "%c0", "%n", "%c1", "%M0", "%L0", "%A0")

    def test_for_multi_result_two(self):
        # two-result variant also produces a list
        op = self._parse(
            "%x, %y = scf.for %i = %lo to %hi step %s"
            " iter_args(%a = %lo, %b = %hi) -> (index, index) {\n"
            "      scf.yield %a, %b : index, index\n    }",
            args={"%lo": "index", "%hi": "index", "%s": "index"},
        )
        assert isinstance(op.result, list)
        assert len(op.result) == 2

    def test_yield_single(self):
        # scf.yield with one value records the operand
        for_op = self._parse(
            "%res = scf.for %i = %lb to %ub step %step iter_args(%acc = %val) -> (f16) {\n"
            "      scf.yield %val : f16\n    }",
            args={"%lb": "index", "%ub": "index", "%step": "index", "%val": "f16"},
        )
        op = for_op.regions[0][0]
        self.assert_op_type(op, "scf.yield")
        self.assert_num_operands(op, 1)
        self.assert_operand_names(op, "%val")

    def test_yield_multi(self):
        # scf.yield with two values records both operands
        for_op = self._parse(
            "%r, %s = scf.for %i = %lb to %ub step %step"
            " iter_args(%acc = %a, %acc2 = %b) -> (f16, f16) {\n"
            "      scf.yield %a, %b : f16, f16\n    }",
            args={"%lb": "index", "%ub": "index", "%step": "index",
                  "%a": "f16", "%b": "f16"},
        )
        op = for_op.regions[0][0]
        self.assert_num_operands(op, 2)
        self.assert_operand_names(op, "%a", "%b")


# ---------------------------------------------------------------------------
# parser infrastructure (tokenizer, region detection, line joining)
# ---------------------------------------------------------------------------

class TestParserInfrastructure:
    def test_multi_result_continuation_joined(self):
        # multi-result split across two lines is joined into one operation
        parser = KTIRParser()
        module = parser.parse_module("""
        module {
          func.func @test() attributes { grid = [1, 1, 1] } {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %n = arith.constant 4 : index
            %init = arith.constant 0 : index
            %res = scf.for %i = %c0 to %n step %c1
              iter_args(%acc = %init) -> (index) {
              scf.yield %acc : index
            }
            return
          }
        }
        """)
        func = module.get_function("test")
        op_types = [op.op_type for op in func.operations]
        assert "scf.for" in op_types

    def test_linalg_generic_region_detected(self):
        # linalg.generic with outs(...) { has its region extracted
        # TODO: test without result name (%r = ...) once parser supports it
        parser = KTIRParser()
        module = parser.parse_module("""
        module {
          func.func @test() attributes { grid = [1, 1, 1] } {
            %c0 = arith.constant 0 : index
            %r = linalg.generic
              ins(%c0 : index)
              outs(%c0 : index) {
            ^bb0(%in: index, %out: index):
              linalg.yield %in : index
            }
            return
          }
        }
        """)
        func = module.get_function("test")
        generic_ops = [op for op in func.operations if op.op_type == "linalg.generic"]
        assert len(generic_ops) == 1
        assert len(generic_ops[0].regions) == 1


# ---------------------------------------------------------------------------
# parser_utils: _extract_bracket_content, parse_attr_list
# ---------------------------------------------------------------------------

from ktir_cpu.parser_utils import _extract_bracket_content, parse_attr_list


class TestExtractBracketContent:
    def test_curly_braces(self):
        assert _extract_bracket_content("op { key = val }") == " key = val "

    def test_square_brackets(self):
        assert _extract_bracket_content("[a, b, c]", brackets="[]") == "a, b, c"

    def test_nested_braces(self):
        assert _extract_bracket_content("{ outer { inner } }") == " outer { inner } "

    def test_no_match_returns_none(self):
        assert _extract_bracket_content("no brackets here") is None

    def test_unmatched_open_returns_none(self):
        assert _extract_bracket_content("{ unclosed") is None


class TestParseAttrList:
    def test_affine_maps(self):
        text = (
            "indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,"
            " affine_map<(d0, d1) -> (d0)>]"
        )
        result = parse_attr_list(text)
        assert len(result) == 2
        assert "affine_map<(d0, d1) -> (d0, d1)>" in result[0]
        assert "affine_map<(d0, d1) -> (d0)>" in result[1]

    def test_empty_brackets(self):
        assert parse_attr_list("[]") == []

    def test_no_brackets(self):
        assert parse_attr_list("no list here") == []

    def test_single_element(self):
        result = parse_attr_list("[affine_map<(d0) -> (d0)>]")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# registry: variadic register()
# ---------------------------------------------------------------------------

from unittest.mock import patch
from ktir_cpu.dialects.registry import register, dispatch, _REGISTRY


class TestVariadicRegister:
    def test_multiple_op_names(self):
        # register with two names maps both to the same handler
        with patch.dict(_REGISTRY, clear=False):
            @register("test.op_a", "test.op_b")
            def handler(op, context, env):
                return "ok"

            assert dispatch("test.op_a") is handler
            assert dispatch("test.op_b") is handler

    def test_inferred_name(self):
        # register() with no args infers name from function name
        with patch.dict(_REGISTRY, clear=False):
            @register()
            def test__inferred(op, context, env):
                return "ok"

            assert dispatch("test.inferred") is test__inferred
