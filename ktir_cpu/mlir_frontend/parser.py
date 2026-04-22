"""
MLIR frontend parser.

Produces the same ir_types.IRModule / IRFunction / Operation objects as
KTIRParser, but drives the mlir.ir structural walk instead of regex-based
text parsing.

Design: adapter pattern
-----------------------
MLIRTypeAdapter      — converts mlir.ir typed attribute objects to ktir_cpu
                       abstraction types (AffineMap, AffineSet, Python scalars).
                       Subclass to customise individual conversions.

MLIRFrontendParser   — drives the mlir.ir parse/verify/walk pipeline and
                       delegates all type conversions to an MLIRTypeAdapter.

Handler design rules
--------------------
- No fallbacks: every handler must raise on unexpected input rather than
  silently returning a default.  Fallbacks hide bugs and make test failures
  harder to diagnose.
- No handler registered → NotImplementedError from adapt_op.
- Missing required attribute → KeyError / ValueError from the handler.
- Unrecognised attribute type → TypeError from the handler.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Callable, Dict, List, Optional, Tuple

try:
    from mlir_ktdp.ir import (
        AffineMapAttr,
        DenseElementsAttr,
        DenseI32ArrayAttr,
        DenseI64ArrayAttr,
        FloatAttr,
        IntegerAttr,
        IntegerSetAttr,
        Module,
    )
    from mlir_ktdp.passmanager import PassManager
    from tools_ktdp.ir_utils import ktdp_context, walk_module
    _HAS_MLIR = True
except ImportError:
    _HAS_MLIR = False

from ..affine import AffineMap, AffineSet
from ..ir_types import IRFunction, IRModule, Operation
from ..parser_ast import parse_affine_map, parse_affine_set
from ..parser import KTIRParserBase


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MLIRTypeAdapter:
    """Adapts mlir.ir typed attribute objects to ktir_cpu abstraction types.

    Op-specific adaptation
    ----------------------
    Handlers are registered via the @MLIRTypeAdapter.install(*opnames)
    decorator.  adapt_op() starts with an empty attributes dict and calls
    the handler (if any) to populate it.  Each handler reads raw attributes
    from mlir_op and owns all conversions for the attributes it sets.

    Handler signature:
        fn(mlir_op, attributes: dict, result_type: str | None, operands: list) -> None

    Utility converters (shared by multiple handlers)
    -------------------------------------------------
    adapt_affine_map(attr) — AffineMapAttr → AffineMap  (parses str repr)
    adapt_affine_set(attr) — IntegerSetAttr → AffineSet (parses str repr)
    """

    _adapt_handlers: ClassVar[Dict[str, Callable]] = {}

    @staticmethod
    def install(*opnames: str):
        """Decorator: register an adapt handler for one or more op names."""
        def decorator(fn: Callable) -> Callable:
            for name in opnames:
                MLIRTypeAdapter._adapt_handlers[name] = fn
            return fn
        return decorator

    # ------------------------------------------------------------------
    # Shared attribute converters (used by handlers)
    # ------------------------------------------------------------------

    def adapt_affine_map(self, attr) -> AffineMap:
        canonical = str(AffineMapAttr(attr).value)
        return parse_affine_map(f"affine_map<{canonical}>")

    def adapt_affine_set(self, attr) -> AffineSet:
        return parse_affine_set(str(attr))

    # ------------------------------------------------------------------
    # Op adapter
    # ------------------------------------------------------------------

    def adapt_op(self, mlir_op) -> Operation:
        """mlir.ir.Operation → ir_types.Operation."""
        n_results = len(mlir_op.results)
        if n_results == 0:
            result = None
        elif n_results == 1:
            result = mlir_op.results[0].get_name()
        else:
            result = [r.get_name() for r in mlir_op.results]

        operands = [v.get_name() for v in mlir_op.operands]
        attributes: Dict[str, Any] = {}

        result_type: Optional[str] = (
            str(mlir_op.results[0].type) if n_results >= 1 else None
        )

        regions: List[List[Operation]] = [
            self.adapt_block(blk)
            for region in mlir_op.regions
            for blk in region.blocks
        ]

        handler = self._adapt_handlers.get(mlir_op.name)
        if handler is None:
            raise NotImplementedError(
                f"No MLIRTypeAdapter handler registered for op '{mlir_op.name}'. "
                "Add a @MLIRTypeAdapter.install(...) handler."
            )
        handler(mlir_op, attributes, result_type, operands)

        return Operation(
            result=result,
            op_type=mlir_op.name,
            operands=operands,
            attributes=attributes,
            result_type=result_type,
            regions=regions,
        )

    def adapt_block(self, block) -> List[Operation]:
        return [self.adapt_op(op) for op in block.operations]


# ---------------------------------------------------------------------------
# Op-specific handlers
# ---------------------------------------------------------------------------

def _no_attrs(mlir_op, attributes, result_type, operands):
    """Handler for ops that carry no execution-relevant attributes."""


MLIRTypeAdapter.install(
    "func.return",
    "linalg.fill",
    "linalg.matmul",
    "linalg.yield",
    "scf.yield",
    "arith.addf", "arith.addi",
    "arith.subf", "arith.subi",
    "arith.mulf", "arith.muli",
    "arith.divf", "arith.divsi", "arith.divui",
    "arith.sitofp",
    "arith.index_cast",
    "arith.maxnumf",
    "arith.maximumf",
    "arith.extf",
    "arith.truncf",
    "arith.bitcast",
    "arith.select",
    "math.exp",
    "math.sqrt",
    "tensor.extract",
    "ktdp.load",
    "ktdp.store",
    # emitted by the bindings walk but not present in text IR
    "ktdp.region_terminator",
)(_no_attrs)


@MLIRTypeAdapter.install("scf.for")
def _adapt_scf_for(mlir_op, attributes, result_type, operands):
    """Synthesize iter_var / iter_args from block arguments."""
    body_args = list(mlir_op.regions[0].blocks[0].arguments)
    attributes["iter_var"] = body_args[0].get_name()
    if len(body_args) > 1:
        attributes["iter_args"] = [a.get_name() for a in body_args[1:]]


@MLIRTypeAdapter.install("ktdp.get_compute_tile_id")
def _adapt_get_compute_tile_id(mlir_op, attributes, result_type, operands):
    """Synthesize num_dims from result count; extract dim if present."""
    attributes["num_dims"] = len(mlir_op.results)
    if "dim" in mlir_op.attributes:
        attributes["dim"] = mlir_op.attributes["dim"]
        if isinstance(attributes["dim"], IntegerAttr):
            attributes["dim"] = attributes["dim"].value


@MLIRTypeAdapter.install("ktdp.construct_access_tile")
def _adapt_construct_access_tile(mlir_op, attributes, result_type, operands):
    """Extract shape, base_map, coordinate_set, coordinate_order from op."""
    m = re.match(r'!ktdp\.access_tile<(\d+(?:x\d+)*)xindex>', result_type)
    if not m:
        raise ValueError(f"ktdp.construct_access_tile: cannot parse result type {result_type!r}")
    shape = tuple(int(d) for d in m.group(1).split('x'))
    attributes["shape"] = shape

    # str(AffineMapAttr(...).value) → "(d0) -> (d0)"
    # base_map is always present in verified MLIR IR
    canonical = str(AffineMapAttr(mlir_op.attributes["base_map"]).value)
    attributes["base_map"] = parse_affine_map(f"affine_map<{canonical}>")

    # access_tile_set → coordinate_set; normalize full sets to None
    if "access_tile_set" in mlir_op.attributes:
        # str(IntegerSetAttr) → "affine_set<(d0) : ...>"
        cs = parse_affine_set(str(mlir_op.attributes["access_tile_set"]))
        if not cs.is_full(shape):
            attributes["coordinate_set"] = cs

    # access_tile_order → coordinate_order; normalize identity maps to None
    if "access_tile_order" in mlir_op.attributes:
        canonical_ord = str(AffineMapAttr(mlir_op.attributes["access_tile_order"]).value)
        co = parse_affine_map(f"affine_map<{canonical_ord}>")
        if not co.is_identity():
            attributes["coordinate_order"] = co


@MLIRTypeAdapter.install("ktdp.construct_memory_view")
def _adapt_construct_memory_view(mlir_op, attributes, result_type, operands):
    """Map static_sizes→shape, static_strides→strides; extract dtype, memory_space from result/attrs."""
    attributes["shape"] = tuple(DenseI64ArrayAttr(mlir_op.attributes["static_sizes"]))
    attributes["strides"] = list(DenseI64ArrayAttr(mlir_op.attributes["static_strides"]))
    m = re.search(r'x([a-zA-Z]\w*)(?:[,>])', result_type)
    if not m:
        raise ValueError(f"ktdp.construct_memory_view: cannot parse dtype from {result_type!r}")
    attributes["dtype"] = m.group(1)
    # str(memory_space attr) → "#ktdp.spyre_memory_space<HBM>"
    ms = re.search(r'#ktdp\.spyre_memory_space<(\w+)>',
                   str(mlir_op.attributes["memory_space"]))
    if not ms:
        raise ValueError(
            f"ktdp.construct_memory_view: cannot parse memory_space from "
            f"{mlir_op.attributes['memory_space']!r}"
        )
    attributes["memory_space"] = ms.group(1)
    # str(coordinate_set attr) → "affine_set<(d0) : ...>"
    if "coordinate_set" in mlir_op.attributes:
        attributes["coordinate_set"] = parse_affine_set(
            str(mlir_op.attributes["coordinate_set"])
        )


@MLIRTypeAdapter.install("ktdp.construct_indirect_access_tile")
def _adapt_construct_indirect_access_tile(mlir_op, attributes, result_type, operands):
    """Reconstruct dim_subscripts from structured bindings attributes.

    operandSegmentSizes encodes the operand layout as four fixed segments
    defined by the op's ODS spec:
      [n_primary_views, n_index_views, n_ssa_vars, n_init_args]
    where n_init_args is always 0 for this op.

    per_dim_subscript_kinds: boolean array — true = indirect (ind(...)), false = direct.
    per_dim_subscript_maps:  one affine map per output dimension giving the subscript
        expression.  The map domain is (ssa_var0, ..., iter_var0, ...) — first n_ssa
        dims are outer SSA scalars, remaining dims are iteration variables.
        _reclassify_dims converts ("dim", N) nodes from this domain to ("ssa", name)
        or ("dim", iter_idx) as the executor expects.
    """
    from ..parser_ast import _Parser, _tokenise
    from ..dialects.ktdp_helpers import _reclassify_dims

    seg = list(DenseI32ArrayAttr(mlir_op.attributes["operandSegmentSizes"]))
    if len(seg) != 4:
        raise ValueError(
            f"ktdp.construct_indirect_access_tile: expected 4 operand segments, got {len(seg)}"
        )
    n_primary, n_index, n_ssa = seg[0], seg[1], seg[2]

    # Slice operand names by segment; executor expects [primary, *index_views]
    primary_name = operands[0]
    index_view_names = operands[n_primary:n_primary + n_index]
    ssa_operand_names = operands[n_primary + n_index:n_primary + n_index + n_ssa]
    operands[:] = [primary_name] + index_view_names

    kinds_attr = mlir_op.attributes["per_dim_subscript_kinds"]
    maps_attr = mlir_op.attributes["per_dim_subscript_maps"]

    def _parse_map_results(map_val_str):
        """Parse the result expressions of one per_dim affine map into executor AST nodes.

        map_val_str is the canonical string of an AffineMap value, e.g.
        "(d0, d1, d2, d3) -> (d0 + d2)".  We extract the result tuple,
        parse each expression with _Parser (which emits ("dim", N) for dN),
        then reclassify: dims in the SSA range become ("ssa", name), dims
        in the iteration range become ("dim", iter_idx).
        """
        results_str = re.search(r"->\s*(\(.*\))\s*$", map_val_str).group(1)
        nodes = _Parser(_tokenise(results_str)).parse_expr_list()
        return [_reclassify_dims(n, ssa_operand_names) for n in nodes]

    dim_subscripts = []
    index_view_idx = 0
    for i in range(len(kinds_attr)):
        is_indirect = str(kinds_attr[i]) == "true"
        exprs = _parse_map_results(str(AffineMapAttr(maps_attr[i]).value))
        if is_indirect:
            # idx_exprs: one expr per dimension of the index view
            dim_subscripts.append({
                "kind": "indirect",
                "index_view_idx": index_view_idx,
                "idx_exprs": exprs,
            })
            index_view_idx += 1
        else:
            expr = exprs[0]
            # Pure iteration variable → compact "direct" form; otherwise "direct_expr"
            if expr[0] == "dim":
                dim_subscripts.append({"kind": "direct", "var_index": expr[1]})
            else:
                dim_subscripts.append({"kind": "direct_expr", "subscript": expr})

    m = re.match(r"!ktdp\.access_tile<(\d+(?:x\d+)*)xindex>", result_type)
    if not m:
        raise ValueError(
            f"ktdp.construct_indirect_access_tile: cannot parse result type {result_type!r}"
        )
    attributes["shape"] = tuple(int(d) for d in m.group(1).split("x"))
    attributes["dim_subscripts"] = dim_subscripts
    attributes["intermediate_vars"] = []
    attributes["variables_space_set"] = parse_affine_set(
        str(mlir_op.attributes["variables_space_set"])
    )
    if "variables_space_order" in mlir_op.attributes:
        canonical_ord = str(AffineMapAttr(mlir_op.attributes["variables_space_order"]).value)
        vso = parse_affine_map(f"affine_map<{canonical_ord}>")
        if not vso.is_identity():
            attributes["variables_space_order"] = vso


@MLIRTypeAdapter.install("linalg.reduce")
def _adapt_linalg_reduce(mlir_op, attributes, result_type, operands):
    """Extract scalar dim; synthesize outs_var and reduce_fn; drop outs from operands."""
    attributes["dim"] = list(DenseI64ArrayAttr(mlir_op.attributes["dimensions"]))[0]
    n_ins = len(operands) // 2
    attributes["outs_var"] = operands[n_ins]
    del operands[n_ins:]  # drop outs — executor only uses ins operands
    body_ops = list(mlir_op.regions[0].blocks[0].operations)
    attributes["reduce_fn"] = body_ops[0].name


@MLIRTypeAdapter.install("tensor.empty")
def _adapt_tensor_empty(mlir_op, attributes, result_type, operands):
    """Synthesize shape/dtype from result type."""
    from ..parser_utils import parse_tensor_type
    info = parse_tensor_type(result_type)
    if info is None:
        raise ValueError(f"tensor.empty: cannot parse result type {result_type!r}")
    attributes["shape"] = info["shape"]
    attributes["dtype"] = info["dtype"]


@MLIRTypeAdapter.install("tensor.expand_shape")
def _adapt_tensor_expand_shape(mlir_op, attributes, result_type, operands):
    """Synthesize target_shape/dtype from result type."""
    from ..parser_utils import parse_tensor_type
    info = parse_tensor_type(result_type)
    if info is None:
        raise ValueError(f"tensor.expand_shape: cannot parse result type {result_type!r}")
    attributes["target_shape"] = info["shape"]
    attributes["dtype"] = info["dtype"]


@MLIRTypeAdapter.install("tensor.collapse_shape")
def _adapt_tensor_collapse_shape(mlir_op, attributes, result_type, operands):
    """Synthesize target_shape/dtype from result type."""
    from ..parser_utils import parse_tensor_type
    info = parse_tensor_type(result_type)
    if info is None:
        raise ValueError(f"tensor.collapse_shape: cannot parse result type {result_type!r}")
    attributes["target_shape"] = info["shape"]
    attributes["dtype"] = info["dtype"]


@MLIRTypeAdapter.install("tensor.splat")
def _adapt_tensor_splat(mlir_op, attributes, result_type, operands):
    """Synthesize shape/dtype from result type."""
    from ..parser_utils import parse_tensor_type
    info = parse_tensor_type(result_type)
    if info is None:
        raise ValueError(f"tensor.splat: cannot parse result type {result_type!r}")
    attributes["shape"] = info["shape"]
    attributes["dtype"] = info["dtype"]


@MLIRTypeAdapter.install("arith.constant")
def _adapt_arith_constant(mlir_op, attributes, result_type, operands):
    """Extract value: unwrap splat tensors, convert int/float attrs."""
    val_attr = mlir_op.attributes["value"]
    if isinstance(val_attr, DenseElementsAttr) and val_attr.is_splat:
        val_attr = val_attr.get_splat_value()
    if not isinstance(val_attr, (IntegerAttr, FloatAttr)):
        raise TypeError(f"arith.constant: unhandled value attr type {type(val_attr)}")
    attributes["value"] = val_attr.value
    if result_type and "tensor<" in result_type:
        from ..parser_utils import parse_tensor_type
        info = parse_tensor_type(result_type)
        attributes["shape"] = info["shape"]
        attributes["dtype"] = info["dtype"]
        attributes["is_tensor"] = True


@MLIRTypeAdapter.install("linalg.broadcast")
def _adapt_linalg_broadcast(mlir_op, attributes, result_type, operands):
    """Extract dimensions list from DenseI64ArrayAttr."""
    attributes["dimensions"] = list(DenseI64ArrayAttr(mlir_op.attributes["dimensions"]))


@MLIRTypeAdapter.install("linalg.transpose")
def _adapt_linalg_transpose(mlir_op, attributes, result_type, operands):
    """Extract permutation from DenseI64ArrayAttr."""
    attributes["permutation"] = list(DenseI64ArrayAttr(mlir_op.attributes["permutation"]))


@MLIRTypeAdapter.install("linalg.index")
def _adapt_linalg_index(mlir_op, attributes, result_type, operands):
    """Extract dim from IntegerAttr."""
    attributes["dim"] = mlir_op.attributes["dim"]
    if isinstance(attributes["dim"], IntegerAttr):
        attributes["dim"] = attributes["dim"].value


@MLIRTypeAdapter.install("linalg.generic")
def _adapt_linalg_generic(mlir_op, attributes, result_type, operands):
    """Extract n_ins, indexing_maps, and bb0_names from bindings attrs.

    operandSegmentSizes gives [n_ins, n_outs]; executor expects operands as
    [*ins, *outs].  indexing_maps is an ArrayAttr of AffineMapAttr; we extract
    each map's result-dim indices (e.g. (d0,d1)->(d0) → [0]) for broadcasting.
    bb0_names come from the body block's arguments; the executor picks them up
    via attributes["bb0_names"] (see linalg__generic executor fallback).
    """
    from ..parser_ast import _Parser, _tokenise

    seg = list(DenseI32ArrayAttr(mlir_op.attributes["operandSegmentSizes"]))
    n_ins = seg[0]
    attributes["n_ins"] = n_ins

    # Extract dimension index list per operand from AffineMapAttr array
    maps = []
    for map_attr in mlir_op.attributes["indexing_maps"]:
        map_val_str = str(AffineMapAttr(map_attr).value)
        results_str = re.search(r"->\s*(\(.*\))\s*$", map_val_str).group(1)
        nodes = _Parser(_tokenise(results_str)).parse_expr_list()
        maps.append([n[1] for n in nodes if n[0] == "dim"])
    attributes["indexing_maps"] = maps

    # Block arguments are the bb0 names; stored in attributes for the executor fallback
    block = mlir_op.regions[0].blocks[0]
    attributes["bb0_names"] = [arg.get_name() for arg in block.arguments]


@MLIRTypeAdapter.install("arith.cmpi")
def _adapt_arith_cmpi(mlir_op, attributes, result_type, operands):
    """Normalize MLIR integer predicate encoding → string the executor expects."""
    # MLIR CmpIPredicateAttr integer encoding (matches mlir::arith::CmpIPredicate)
    _pred_map = {
        0: "eq",  1: "ne",
        2: "slt", 3: "sle", 4: "sgt", 5: "sge",
        6: "ult", 7: "ule", 8: "ugt", 9: "uge",
    }
    pred_int = IntegerAttr(mlir_op.attributes["predicate"]).value
    if pred_int not in _pred_map:
        raise ValueError(f"arith.cmpi: unknown predicate integer {pred_int}")
    attributes["predicate"] = _pred_map[pred_int]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class MLIRFrontendParser(KTIRParserBase):
    """Parse MLIR text into IRModule objects using the MLIR Python bindings.

    Uses MLIRTypeAdapter (or a subclass) for all mlir.ir → ktir_cpu type
    conversions.

    Usage::

        parser = MLIRFrontendParser()
        module = parser.parse_module(mlir_text)

    Raises ImportError at construction time if mlir_ktdp is not installed.
    """

    _NORMALIZE_PIPELINE = "builtin.module(canonicalize,cse)"

    def __init__(self, adapter: Optional[MLIRTypeAdapter] = None):
        if not _HAS_MLIR:
            raise ImportError(
                "mlir_ktdp / tools_ktdp not installed; "
                "MLIRFrontendParser is unavailable."
            )
        self._adapter = adapter or MLIRTypeAdapter()

    def parse_module(self, mlir_text: str, normalize: bool = False) -> IRModule:
        """Parse MLIR text into an IRModule.

        Args:
            mlir_text:  MLIR source text.
            normalize:  If True, run canonicalize + CSE before walking.

        Raises:
            mlir_ktdp.ir.MLIRError: on parse failure.
            RuntimeError: if post-parse verify() fails.
        """
        with ktdp_context() as ctx:
            m = Module.parse(mlir_text, ctx)
            if not m.operation.verify():
                raise RuntimeError("MLIR module failed verification after parse")
            if normalize:
                PassManager.parse(self._NORMALIZE_PIPELINE).run(m.operation)
                if not m.operation.verify():
                    raise RuntimeError(
                        "MLIR module failed verification after canonicalize"
                    )
                mlir_text = str(m)

        operations = walk_module(mlir_text)
        return self._build_ir_module(operations)

    @staticmethod
    def lint(mlir_text: str) -> Tuple[bool, str]:
        """Validate MLIR text without producing an IRModule.

        Returns:
            (True, "")           — no issues found
            (False, diagnostics) — parse or verify error message
        """
        try:
            with ktdp_context() as ctx:
                m = Module.parse(mlir_text, ctx)
                if not m.operation.verify():
                    return False, "module failed verify()"
                return True, ""
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_ir_module(self, operations) -> IRModule:
        module = IRModule()
        for op, depth in operations:
            if depth == 1 and op.name == "func.func":
                module.add_function(self._build_ir_function(op))
        return module

    def _build_ir_function(self, func_op) -> IRFunction:
        sym_name = func_op.attributes["sym_name"].value
        block = func_op.regions[0].blocks[0]

        arguments: List[Tuple[str, str]] = [
            (arg.get_name(), str(arg.type))
            for arg in block.arguments
        ]

        grid = (1, 1, 1)
        if "grid" in func_op.attributes:
            dims = [IntegerAttr(e).value for e in func_op.attributes["grid"]]
            dims += [1] * (3 - len(dims))
            grid = tuple(dims)

        return IRFunction(
            name=sym_name,
            arguments=arguments,
            operations=self._adapter.adapt_block(block),
            grid=grid,
        )
