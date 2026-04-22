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


def _parse(op_text, parse_ctx=None):
    """Dispatch-parse a single op line and return the Operation."""
    ctx = parse_ctx or _parse_ctx()
    parser_fn = dispatch_parser(op_text)
    assert parser_fn is not None, f"No parser found for: {op_text!r}"
    return parser_fn(op_text, ctx)


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

class TestArithParsers:
    def test_constant_scalar(self):
        # scalar integer constant parsed with correct value
        op = _parse("%c0 = arith.constant 42 : index")
        assert op.op_type == "arith.constant"
        assert op.attributes["value"] == 42

    def test_constant_dense_tensor(self):
        # dense<0.0> tensor constant sets is_tensor, shape, and dtype
        op = _parse("%t = arith.constant {dense<0.0>} : tensor<4xf16>")
        assert op.attributes["is_tensor"] is True
        assert op.attributes["shape"] == (4,)
        assert op.attributes["dtype"] == "f16"

    def test_cmpi_basic(self):
        # cmpi records predicate and both operands
        op = _parse("%b = arith.cmpi slt, %a, %c0 : index")
        assert op.op_type == "arith.cmpi"
        assert op.attributes["predicate"] == "slt"
        assert "%a" in op.operands
        assert "%c0" in op.operands

    def test_cmpi_all_predicates(self):
        # all six comparison predicates are recognised
        for pred in ("eq", "ne", "slt", "sle", "sgt", "sge"):
            op = _parse(f"%b = arith.cmpi {pred}, %x, %y : i32")
            assert op.attributes["predicate"] == pred

    def test_sitofp(self):
        # sitofp records operand and target float type
        op = _parse("%f = arith.sitofp %i : i32 to f16")
        assert op.op_type == "arith.sitofp"
        assert op.operands == ["%i"]
        assert op.result_type == "f16"


# ---------------------------------------------------------------------------
# linalg dialect parsers
# ---------------------------------------------------------------------------

class TestLinalgParsers:
    def test_reduce(self):
        # reduce records reduce_fn, dim, outs_var, and ins operand
        op = _parse(
            "%r = linalg.reduce { arith.maxnumf }"
            " ins(%x : tensor<1x1024xf16>)"
            " outs(%init : tensor<1xf16>)"
            " dimensions = [1]"
        )
        assert op.op_type == "linalg.reduce"
        assert op.attributes["reduce_fn"] == "arith.maxnumf"
        assert op.attributes["dim"] == 1
        assert op.attributes["outs_var"] == "%init"
        assert "%x" in op.operands

    def test_fill(self):
        # fill records both ins and outs operands
        op = _parse(
            "%out = linalg.fill ins(%val : f16) outs(%buf : tensor<4xf16>) -> tensor<4xf16>"
        )
        assert op.op_type == "linalg.fill"
        assert "%val" in op.operands
        assert "%buf" in op.operands

    def test_broadcast(self):
        # broadcast records dimensions and both ins/outs operands
        op = _parse(
            "%out = linalg.broadcast ins(%x : tensor<4xf16>) outs(%buf : tensor<4x8xf16>) dimensions = [1]"
        )
        assert op.op_type == "linalg.broadcast"
        assert op.attributes["dimensions"] == [1]
        assert "%x" in op.operands
        assert "%buf" in op.operands


# ---------------------------------------------------------------------------
# tensor dialect parsers
# ---------------------------------------------------------------------------

class TestTensorParsers:
    def test_empty(self):
        # tensor.empty records shape and dtype from type annotation
        op = _parse("%t = tensor.empty() : tensor<1x1024xf16>")
        assert op.op_type == "tensor.empty"
        assert op.attributes["shape"] == (1, 1024)
        assert op.attributes["dtype"] == "f16"

    def test_splat(self):
        # tensor.splat records scalar operand and target shape
        op = _parse("%t = tensor.splat %val : tensor<4xf16>")
        assert op.op_type == "tensor.splat"
        assert op.operands == ["%val"]
        assert op.attributes["shape"] == (4,)

    def test_extract(self):
        # tensor.extract records tensor operand and index operands
        op = _parse("%s = tensor.extract %t[%i, %j] : tensor<4x4xf16>")
        assert op.op_type == "tensor.extract"
        assert op.operands[0] == "%t"
        assert "%i" in op.operands
        assert "%j" in op.operands

    def test_expand_shape(self):
        # tensor.expand_shape records source operand and target shape
        op = _parse("%out = tensor.expand_shape %in into tile<1x1024xf16>")
        assert op.op_type == "tensor.expand_shape"
        assert op.operands == ["%in"]
        assert op.attributes["target_shape"] == (1, 1024)


# ---------------------------------------------------------------------------
# ktdp dialect parsers
# ---------------------------------------------------------------------------

class TestKtdpParsers:
    def test_get_compute_tile_id_single(self):
        # single-result form records num_dims=1 and result name
        op = _parse("%id = ktdp.get_compute_tile_id : index")
        assert op.op_type == "ktdp.get_compute_tile_id"
        assert op.attributes["num_dims"] == 1
        assert op.result == "%id"

    def test_get_compute_tile_id_multi(self):
        # multi-result form records num_dims=2 and a list result
        op = _parse("%x, %y = ktdp.get_compute_tile_id : index, index")
        assert op.op_type == "ktdp.get_compute_tile_id"
        assert op.attributes["num_dims"] == 2
        assert isinstance(op.result, list)

    def test_construct_memory_view(self):
        # construct_memory_view records shape, dtype, and pointer operand
        op = _parse(
            "%view = ktdp.construct_memory_view %ptr,"
            " sizes: [1024], strides: [1] : index -> memref<1024xf16>"
        )
        assert op.op_type == "ktdp.construct_memory_view"
        assert op.attributes["shape"] == (1024,)
        assert op.attributes["dtype"] == "f16"
        assert op.operands == ["%ptr"]

    def test_construct_access_tile(self):
        # construct_access_tile records tile shape and base view operand
        op = _parse(
            "%acc = ktdp.construct_access_tile %view[%c0]"
            " : memref<1024xf16> -> !ktdp.access_tile<128xindex>"
        )
        assert op.op_type == "ktdp.construct_access_tile"
        assert op.attributes["shape"] == (128,)
        assert "%view" in op.operands

    def test_construct_access_tile_non_index_elem_type_rejected(self):
        # Per spec, AccessTileType element type must be 'index'; any other type
        # is a spec violation and must be rejected at parse time.
        with pytest.raises(ValueError, match="element type must be 'index'"):
            _parse(
                "%acc = ktdp.construct_access_tile %view[%c0]"
                " : memref<1024xf16> -> !ktdp.access_tile<128xf16>"
            )

    def test_construct_access_tile_malformed_type_rejected(self):
        # A type string with no alphabetic element-type suffix is malformed.
        with pytest.raises(ValueError, match="Malformed access_tile type"):
            _parse(
                "%acc = ktdp.construct_access_tile %view[%c0]"
                " : memref<1024xf16> -> !ktdp.access_tile<128>"
            )


# ---------------------------------------------------------------------------
# scf dialect parsers
# ---------------------------------------------------------------------------

class TestScfParsers:
    def test_for_basic(self):
        # scf.for records iter_var and lb/ub/step operands in order
        op = _parse("scf.for %i = %lb to %ub step %step {")
        assert op.op_type == "scf.for"
        assert op.attributes["iter_var"] == "%i"
        assert op.operands[:3] == ["%lb", "%ub", "%step"]

    def test_for_with_result(self):
        # optional result prefix on scf.for is captured
        op = _parse("%res = scf.for %i = %lb to %ub step %step {")
        assert op.result == "%res"
        assert op.attributes["iter_var"] == "%i"

    def test_for_iter_args(self):
        # iter_args clause records carried variable and init operand
        op = _parse(
            "scf.for %i = %lb to %ub step %step iter_args(%acc = %init) {"
        )
        assert op.attributes["iter_args"] == ["%acc"]
        assert "%init" in op.operands

    def test_for_multi_result(self):
        # multi-result scf.for records a list of result names
        op = _parse(
            "%M, %L, %acc = scf.for %j = %c0 to %n step %c1"
            " iter_args(%m = %M0, %l = %L0, %a = %A0) {"
        )
        assert op.op_type == "scf.for"
        assert isinstance(op.result, list)
        assert op.result == ["%M", "%L", "%acc"]
        assert op.attributes["iter_var"] == "%j"
        assert op.attributes["iter_args"] == ["%m", "%l", "%a"]
        assert op.operands[:3] == ["%c0", "%n", "%c1"]

    def test_for_multi_result_two(self):
        # two-result variant also produces a list
        op = _parse("%x, %y = scf.for %i = %lo to %hi step %s {")
        assert isinstance(op.result, list)
        assert op.result == ["%x", "%y"]

    def test_yield_single(self):
        # scf.yield with one value records the operand
        op = _parse("scf.yield %val : f16")
        assert op.op_type == "scf.yield"
        assert "%val" in op.operands

    def test_yield_multi(self):
        # scf.yield with two values records both operands
        op = _parse("scf.yield %a, %b : f16, f16")
        assert "%a" in op.operands
        assert "%b" in op.operands


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
