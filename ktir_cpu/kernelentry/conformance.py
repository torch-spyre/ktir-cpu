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

"""Kernels under ``examples/`` that the MLIR frontend does not accept.

This is a record, not a verdict. ``parse.frontend`` can only be *decided* where the
MLIR bindings are installed, so on most machines it reads ``skip`` — and a committed
support report that printed the outcome it saw would be a document about the machine
that generated it. Writing the outcome down here instead makes the report a function
of the repository, and ``tests/mlir_frontend/test_kernelentry_adapt.py`` holds this
mapping to the truth in the one environment that can check it. Same shape and same
reason as ``FRONTEND_UNSUPPORTED`` in
``tests/mlir_frontend/test_registry_consistency.py``: a committed mapping of known
gaps, each with a reason, verified rather than trusted.

What this mapping deliberately does not say is which side is wrong. Unlike
``FRONTEND_UNSUPPORTED``, which covers ops the dialect does not define at all, these
are defined ops, and the disagreement is a typing one rather than a spelling one.
Gap row 2a in ``docs/gap_analysis.md`` carries it; ``inter_tile_produce`` /
``inter_tile_reduce`` are not in RFC 0682, so the specification does not adjudicate
it either.
"""

from __future__ import annotations

from typing import Dict

# One gap, reached through three kernels, so the reason is named once rather than
# repeated per path: all three write the inter-tile ops in the form only the regex
# parser reads
#     : T -> !ktdp.tile_future<T, groups = S>
# rather than the dialect's own
#     -> <(T), groups = S>            (produce)
#     : <(T), groups = S> -> R        (reduce)
# so the frontend stops at the first inter-tile op with `expected '->'`. Rewriting
# only that spelling moves the error rather than removing it: all three reduce
# `tensor<1x128xf16>` to `tensor<128xf16>`, and the dialect verifies that a reduce
# result matches the future's partial type, so it then fails with `result types must
# match future partial types`. Closing that means deciding whether the reduce should
# reshape at all, which is gap row 2a's question, not this mapping's.
_RESHAPING_REDUCE = (
    "inter-tile ops in the form only the regex parser reads, and a reduce that "
    "reshapes its result, which the dialect's type relation does not express "
    "(gap row 2a)"
)

#: Repository-relative kernel path -> why the frontend rejects it.
#:
#: A kernel belongs here only if it has no declaration. A declared kernel records
#: the same fact as a ``deferred`` or ``waived`` ``parse.frontend`` claim, and two
#: places recording one fact is how they come to disagree.
FRONTEND_REJECTS: Dict[str, str] = {
    "examples/ktir/ring_reduce.mlir": _RESHAPING_REDUCE,
    "examples/ktir/ring_reduce_inner_loop.mlir": _RESHAPING_REDUCE,
    "examples/latency/ring_reduce_multi_group.mlir": _RESHAPING_REDUCE,

    # A different defect class, and it does not overlap with the one above: this
    # kernel has no `coordinate_set` on `construct_memory_view` at all, which the
    # dialect requires and the regex parser does not.
    "examples/ktir/nested_yield.ktir":
        "construct_memory_view has no coordinate_set, which the dialect requires",
}
