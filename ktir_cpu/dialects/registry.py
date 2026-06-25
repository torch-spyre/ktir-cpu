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
Dialect handler registry for KTIR CPU backend.

Provides a decorator-based registration system so each dialect module
can declare its operation handlers at import time, plus a parallel
registry for dialect-specific parsers.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..grid import CoreContext, GridExecutor
from ..ir_types import Operation

# ---------------------------------------------------------------------------
# Execution handler registry
# ---------------------------------------------------------------------------

# Handler signature: (op, context, env) -> Any
HandlerFn = Callable[[Operation, CoreContext, "ExecutionEnv"], Any]

_REGISTRY: Dict[str, HandlerFn] = {}

# ---------------------------------------------------------------------------
# Latency category registry
# ---------------------------------------------------------------------------

_LATENCY_CATEGORIES: Dict[str, str] = {}

# Ops registered with inplace_outs=True — their outs buffer is an in-place
# accumulator and the handler must return the same Tile object.
_INPLACE_OPS_SET: set = set()

# Ops registered with no_lx_charge=True — their result is a compile-time
# literal (e.g. arith.constant) that does not occupy the LX scratchpad. LX is
# charged only when a consuming op materializes the literal into a real working
# tile (see CoreContext.set_value / scf.for iter_arg binding).
_NO_LX_CHARGE_OPS_SET: set = set()


def get_latency_category(op_name: str) -> str:
    """Return the latency category for *op_name*, defaulting to ``"zero"``.

    Values are :class:`~ktir_cpu.latency.LatencyCategory` members (``StrEnum``).
    """
    return _LATENCY_CATEGORIES.get(op_name, "zero")

# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

# Parser signature: (op_text: str, parse_ctx: ParseContext) -> Optional[Operation]
# op_text is LHS-free (body only); caller assigns op.result after return.
ParserFn = Callable[..., Optional[Operation]]

_PARSER_REGISTRY: Dict[str, ParserFn] = {}


def register_parser(*op_patterns: str):
    """Register a parse function for one or more op name patterns.

    Patterns are matched with ``in`` against the op text.
    """
    def decorator(fn: ParserFn) -> ParserFn:
        for p in op_patterns:
            _PARSER_REGISTRY[p] = fn
        return fn
    return decorator


def dispatch_parser(op_text: str) -> Optional[ParserFn]:
    """Return the first registered parser whose pattern appears in *op_text*."""
    for pattern, fn in _PARSER_REGISTRY.items():
        if pattern in op_text:
            return fn
    return None


def make_parse_context(aliases: Dict[str, str]) -> "ParseContext":
    """Convenience constructor for ParseContext."""
    return ParseContext(aliases=aliases)


def register(*op_names: str, latency_category: str = "zero", inplace_outs: bool = False,
             no_lx_charge: bool = False):
    """Decorator that registers a dialect operation handler.

    Usage::

        @register("arith.addf", latency_category=LC.COMPUTE_FLOAT)
        def arith_addf(op, context, env):
            ...

        @register("arith.maxf", "arith.maximumf", latency_category=LC.COMPUTE_FLOAT)
        def arith_maxf(op, context, env):
            ...

        @register()  # infers "arith.addf" from function name "arith__addf"
        def arith__addf(op, context, env):
            ...

    Args:
        inplace_outs: When True, the op's outs buffer is an in-place accumulator
            and parsers will populate ``Operation.outs_operands``.  The structural
            assertion in ``_execute_op`` then verifies the handler returns the
            same Tile object.
        no_lx_charge: When True, the op's result Tile is a compile-time literal
            that does not occupy LX (e.g. ``arith.constant``).  ``_execute_op``
            binds it without charging the scratchpad; a consuming op pays for
            the real buffer when it materializes the literal.
    """
    def decorator(fn: HandlerFn) -> HandlerFn:
        names = op_names or (fn.__name__.replace("__", "."),)
        for name in names:
            _REGISTRY[name] = fn
            _LATENCY_CATEGORIES[name] = latency_category
            if inplace_outs:
                _INPLACE_OPS_SET.add(name)
            if no_lx_charge:
                _NO_LX_CHARGE_OPS_SET.add(name)
        return fn
    return decorator


def is_inplace_outs(op_name: str) -> bool:
    """Return whether *op_name* was registered with ``inplace_outs=True``."""
    return op_name in _INPLACE_OPS_SET


def is_no_lx_charge(op_name: str) -> bool:
    """Return whether *op_name* was registered with ``no_lx_charge=True``."""
    return op_name in _NO_LX_CHARGE_OPS_SET


def dispatch(op_name: str) -> Optional[HandlerFn]:
    """Look up the handler for *op_name*, or return ``None``."""
    return _REGISTRY.get(op_name)


def temp_registry():
    """Context manager that restores _REGISTRY, _LATENCY_CATEGORIES, and _INPLACE_OPS_SET on exit."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        saved_registry = _REGISTRY.copy()
        saved_latency = _LATENCY_CATEGORIES.copy()
        saved_inplace = _INPLACE_OPS_SET.copy()
        try:
            yield
        finally:
            _REGISTRY.clear()
            _REGISTRY.update(saved_registry)
            _LATENCY_CATEGORIES.clear()
            _LATENCY_CATEGORIES.update(saved_latency)
            _INPLACE_OPS_SET.clear()
            _INPLACE_OPS_SET.update(saved_inplace)

    return _ctx()


@dataclass
class ParseContext:
    """Parse-time context passed to dialect parsers alongside the op text.

    Carries everything a parser needs that is not present in the op text
    itself.  Keeping this separate from ExecutionEnv ensures that parsing
    concerns (alias resolution, IR construction) stay decoupled from
    execution concerns (grid, memory, latency).
    """
    # Named attribute aliases collected by the module-level pre-scan.
    # Maps "#name" -> verbatim value string, e.g.
    #   "#X_coord_set" -> "affine_set<(d0, d1) : (d0 >= 0, ...)>"
    aliases: Dict[str, str]


@dataclass
class ExecutionEnv:
    """Lightweight bag of core-external resources passed to handlers."""

    grid_executor: GridExecutor
    execute_region: Callable[[CoreContext, List[Operation]], Any]
    # Generator variant used by scf handlers (scf.for, scf.if) whose bodies
    # may contain comm ops.  When None, scf handlers fall back to a thin
    # generator wrapper around execute_region (no comm ops will be scheduled).
    execute_region_with_comms: Callable[[CoreContext, List[Operation]], Any] = None
