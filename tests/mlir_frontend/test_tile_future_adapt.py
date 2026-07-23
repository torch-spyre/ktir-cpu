"""Adapter tests for the ``!ktdp.tile_future`` type parser.

The tile-future type has an optional ``groups`` clause:

    !ktdp.tile_future< partial_types [, groups = <affine-set>] >

The ``_adapt_inter_tile_produce`` / ``_adapt_inter_tile_reduce`` handlers
extract ``groups`` from the type when present, and otherwise from the
op's own ``groups`` attribute.

The tests split along two layers:

- :class:`TestTileFutureType` parses a full MLIR module through
  :class:`MLIRFrontendParser` and inspects the adapted
  :class:`Operation` for ``groups`` and ``partial_tensor_types``.
- :class:`TestParseTileFutureType` calls ``_parse_tile_future_type``
  directly to cover grammar variants without an MLIR round trip.
"""

import textwrap

import pytest

from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser
from ktir_cpu.mlir_frontend.parser import _parse_tile_future_type
from ktir_cpu.mlir_frontend.parser import parse_affine_set


def _first_op(module, func_name, op_type):
    for op in module.get_function(func_name).operations:
        if op.op_type == op_type:
            return op
    raise AssertionError(
        f"No {op_type!r} op found in function {func_name!r}"
    )


def _assert_same_set(actual, expected_text):
    """Assert *actual* (a parsed affine set / BoxSet) equals what
    ``parse_affine_set`` produces from *expected_text*.

    Using ``parse_affine_set`` on both sides normalises across the two
    kinds of value the parser can return (BoxSet for rectangular sets,
    AffineSet otherwise) — comparing raw ``str()`` output would depend on
    that internal choice."""
    expected = parse_affine_set(expected_text)
    assert actual == expected, (
        f"parsed groups mismatch: got {actual!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# End-to-end: parse a module through MLIRFrontendParser
# ---------------------------------------------------------------------------


class TestTileFutureType:
    """``groups`` embedded in the ``!ktdp.tile_future`` type parameter."""

    @pytest.fixture(scope="class")
    def parsed_module(self):
        module_text = textwrap.dedent("""\
            #ptpg      = affine_set<(i)[g] : (i - g == 0, i >= 0, -i + 3 >= 0)>
            #one_group = affine_set<(g) : (g == 0)>
            module {
              func.func @_k(%p: tensor<1x64xf16>) -> tensor<1x64xf16>
                  attributes { grid = [4] }
              {
                %f = ktdp.inter_tile_produce
                    producer_tiles_per_group = #ptpg
                    -> <(tensor<1x64xf16>), groups = #one_group>
                {
                  ^bb0(%gid: index):
                    ktdp.yield_partial %p : tensor<1x64xf16>
                }
                %id = tensor.empty() : tensor<1x64xf16>
                %r = ktdp.inter_tile_reduce(%f)
                    consumer_tiles_per_group = #ptpg,
                    identity(%id : tensor<1x64xf16>)
                    : <(tensor<1x64xf16>), groups = #one_group> -> tensor<1x64xf16>
                {
                  ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
                    %sum = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                                      outs(%lhs : tensor<1x64xf16>)
                                      -> tensor<1x64xf16>
                    ktdp.yield_reduced %sum : tensor<1x64xf16>
                }
                return %r : tensor<1x64xf16>
              }
            }
            """)
        return MLIRFrontendParser().parse_module(module_text)

    def test_produce_extracts_groups_and_partials(self, parsed_module):
        """The produce op's result tile_future type contributes both the
        ``groups`` set and the single-role ``partial_tensor_types``.

        Asserts on parsed values (not just key presence): a regression that
        stored the wrong affine set — for example a copy of
        ``producer_tiles_per_group`` — would change the extracted text and
        fail here.
        """
        produce = _first_op(parsed_module, "_k", "ktdp.inter_tile_produce")
        assert produce.attributes["partial_tensor_types"] == (
            "tensor<1x64xf16>",
        )
        _assert_same_set(
            produce.attributes["groups"], "affine_set<(g) : (g == 0)>"
        )

    def test_reduce_extracts_groups_from_operand_type(self, parsed_module):
        """The reduce op's ``%future`` operand's tile_future type carries
        ``groups``; the adapter extracts the same set the producer emitted.
        """
        reduce = _first_op(parsed_module, "_k", "ktdp.inter_tile_reduce")
        _assert_same_set(
            reduce.attributes["groups"], "affine_set<(g) : (g == 0)>"
        )
        # consumer_tiles_per_group still comes from the op attribute; it must
        # equal the source `#ptpg` set. This pins that the type-parameter
        # extraction did not accidentally overwrite it.
        _assert_same_set(
            reduce.attributes["consumer_tiles_per_group"],
            "affine_set<(i)[g] : (i - g == 0, i >= 0, -i + 3 >= 0)>",
        )


# ---------------------------------------------------------------------------
# Unit: _parse_tile_future_type grammar variants
# ---------------------------------------------------------------------------


class TestParseTileFutureType:
    """Directly exercise the tile_future grammar variants.

    The full grammar is ``!ktdp.tile_future< partial_types [, groups = <affine-set>] >``
    where ``partial_types`` may be a bare list or a parenthesised tuple.
    These tests cover each combination without an MLIR round trip.
    """

    def test_parenthesised_partials_with_groups(self):
        """Parenthesised single-role partials with an embedded groups clause."""
        partials, groups = _parse_tile_future_type(
            "!ktdp.tile_future<(tensor<1x64xf16>), groups = affine_set<(g) : (g == 0)>>",
            context="test",
        )
        assert partials == ("tensor<1x64xf16>",)
        assert groups == "affine_set<(g) : (g == 0)>"

    def test_parenthesised_multi_role_partials(self):
        """Argmax-style variadic future: N > 1 roles inside the parens."""
        partials, groups = _parse_tile_future_type(
            "!ktdp.tile_future<(tensor<128xf32>, tensor<128xi32>), "
            "groups = affine_set<(g) : (g == 0)>>",
            context="test",
        )
        assert partials == ("tensor<128xf32>", "tensor<128xi32>")
        assert groups == "affine_set<(g) : (g == 0)>"

    def test_bare_single_partial_no_groups(self):
        """Legacy form: bare partials list, no groups clause. This exercises
        the ``not startswith('(')`` branch and the ``groups_str is None``
        return contract that the adapter's fallback depends on."""
        partials, groups = _parse_tile_future_type(
            "!ktdp.tile_future<tensor<1x128xf16>>",
            context="test",
        )
        assert partials == ("tensor<1x128xf16>",)
        assert groups is None

    def test_bare_multi_partial_no_groups(self):
        """Legacy variadic form: several partials, no groups clause."""
        partials, groups = _parse_tile_future_type(
            "!ktdp.tile_future<tensor<128xf32>, tensor<128xi32>>",
            context="test",
        )
        assert partials == ("tensor<128xf32>", "tensor<128xi32>")
        assert groups is None

    def test_malformed_type_error_names_context(self):
        """The ValueError names the caller-supplied context so log triage
        can distinguish produce/result from reduce/operand-0 sites."""
        with pytest.raises(ValueError, match="ktdp.inter_tile_reduce operand 0"):
            _parse_tile_future_type(
                "!not.a.tile_future<tensor<f32>>",
                context="ktdp.inter_tile_reduce operand 0",
            )
