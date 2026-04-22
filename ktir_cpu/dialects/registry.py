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
from ..ops.comm_ops import RingNetwork

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


def get_latency_category(op_name: str) -> str:
    """Return the latency category for *op_name*, defaulting to ``"zero"``.

    Values are :class:`~ktir_cpu.latency.LatencyCategory` members (``StrEnum``).
    """
    return _LATENCY_CATEGORIES.get(op_name, "zero")

# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

# Parser signature: (op_text: str, parse_ctx: ParseContext) -> Optional[Operation]
# ParseContext carries everything dialect parsers need beyond the op text itself
# (currently just the alias table).  Using a typed object keeps the signature
# stable when new parse-time context is added later.
ParserFn = Callable[[str, "ParseContext"], Optional[Operation]]

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


def register(*op_names: str, latency_category: str = "zero"):
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
    """
    def decorator(fn: HandlerFn) -> HandlerFn:
        names = op_names or (fn.__name__.replace("__", "."),)
        for name in names:
            _REGISTRY[name] = fn
            _LATENCY_CATEGORIES[name] = latency_category
        return fn
    return decorator


def dispatch(op_name: str) -> Optional[HandlerFn]:
    """Look up the handler for *op_name*, or return ``None``."""
    return _REGISTRY.get(op_name)


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
    ring: RingNetwork
    execute_region: Callable[[CoreContext, List[Operation]], Any]
