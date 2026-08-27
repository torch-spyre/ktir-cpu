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

"""Splitting "this op is free" from "nobody priced this op".

``@register()`` defaults ``latency_category`` to ``"zero"``, so the registry holds
one state where there are two facts: an op the hardware really does no measurable
work for, and an op whose cost nobody has decided yet.  A kernel leaning on the
second reports a lower cost than the hardware would, and every figure the cost leg
prints is read out of that registry — which makes a *clean* cost report
untrustworthy rather than merely incomplete.

This module is the committed record that splits them.  Every op priced ``zero``
must appear in exactly one of two mappings:

``ZERO_COST_OPS``
    Free by decision, with the reason the hardware does no measurable work.

``UNJUDGED_ZERO_OPS``
    Not judged yet, with what is unresolved about it and the issue tracking the
    decision.  :func:`audit` accepts these — an open question that is written
    down is not the failure mode being guarded against.

:func:`audit` fails on an op in **neither**, which is what stops a newly
registered op from being silently free.  It also fails on the three ways the
record can rot: an op in both mappings, an entry whose op has since been priced,
and an entry naming an op that is no longer registered.  All four are one-line
edits to this file; none of them touches a cost formula.

**Repository-wide, not per kernel** — measured, not assumed.  A per-kernel
prototype of this check over six real artifacts produced 74 false positives, 66 of
them here: about fifteen of the ops it flagged are the same structural ones in
every kernel (``arith.constant``, ``ktdp.construct_memory_view``, ``scf.for``),
genuinely free.  Per kernel it is a chore each contributor waives their way
through, and a waiver mapping that is mostly noise stops carrying information.
So it is asked once of the registry, and no kernel declaration mentions it.

**What this check cannot see: a wrong category.**  It compares against ``zero``,
so an op priced in the wrong non-zero class passes it — an integer compare billed
to the float pipe is invisible here.  Only a reader who knows the op's semantics
catches that, which is why the reasons below are written for a reader rather than
being generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import ktir_cpu.dialects  # noqa: F401 — import triggers @register side effects
from ktir_cpu.dialects import registry

# ---------------------------------------------------------------------------
# Free by decision
# ---------------------------------------------------------------------------

#: Ops that cost nothing because of what they *are*, with the reason each one
#: does no measurable work.  Four kinds appear here and it is worth reading them
#: as four: a value that exists at compile time, a terminator that only names
#: values, addressing metadata that computes where data is without moving it, and
#: an orchestrator whose body is charged op by op.
ZERO_COST_OPS: Dict[str, str] = {
    # -- compile-time values ------------------------------------------------
    "arith.constant":
        "a compile-time literal, and registered `no_lx_charge=True` for the same "
        "reason: the scratchpad is charged when a consumer materializes the "
        "value into a working tile, not here",

    # -- terminators: they name values leaving a region, and issue nothing ---
    "func.return": "a terminator; it names the values leaving the function",
    "return": "the regex parser's spelling of `func.return`; same reason",
    "linalg.yield": "a region terminator; it names the combiner's result",
    "scf.yield": "a region terminator",
    "tensor.yield": "a region terminator, for a `tensor.generate` body",
    "ktdp.yield_partial": "a region terminator; it names this core's partial",
    "ktdp.yield_reduced": "a region terminator; it names the reduced value",
    "region.bb0_args":
        "not an op at all — the parser's record of a region's block arguments, "
        "which the enclosing op's handler binds",

    # -- addressing: where the data is, not the data -------------------------
    "ktdp.construct_memory_view":
        "computes an address, moves nothing; the `ktdp.load` or `ktdp.store` "
        "that reads through the view is what carries the bytes",
    "ktdp.construct_distributed_memory_view":
        "the same address computation, per partition; the traffic is charged on "
        "the loads and stores that address it",
    "ktdp.construct_access_tile":
        "narrows an existing view to a tile's worth of it — index arithmetic on "
        "the view, with no access performed",
    "ktdp.construct_indirect_access_tile":
        "the same narrowing with a gathered index set; the gather itself is "
        "charged on the indirect load, whose figure covers both the data and "
        "the index lookups",
    "ktdp.get_compute_tile_id":
        "reads the core's own coordinate in the grid, which is available to it "
        "without a memory access",

    # -- the produce half of a cross-core reduce -----------------------------
    "ktdp.inter_tile_produce":
        "publishes this core's partial to the scheduler's mailbox; the wire time "
        "for the whole exchange is charged once, as `comm`, on the matching "
        "`ktdp.inter_tile_reduce`. Pricing both would count one transfer twice",

    # -- orchestrators: the body is charged, op by op ------------------------
    "linalg.reduce":
        "executes its combiner region rather than mapping to a fixed reduction, "
        "so the arithmetic inside it is charged individually: one core's trace "
        "for the RMSNorm generator carries 256 `arith.addf` entries totalling "
        "1,280 cycles from inside a reduce. The orchestrator is free; the "
        "arithmetic is not",
    "tensor.generate":
        "evaluates its region body per index, so the ops in the body are charged "
        "individually — the same split as `linalg.reduce`",
    "linalg.index":
        "produces the iteration index inside a region body; the arithmetic that "
        "consumes it is charged, again the `linalg.reduce` split",
    "scf.for":
        "loop control; the body's ops are charged once per iteration, so a cost "
        "for the loop itself would be on top of the work it drives",
    "scf.if":
        "branch control; the ops of the taken branch are charged",

    # -- allocation without initialization -----------------------------------
    "tensor.empty":
        "names an uninitialized buffer. No data moves, and a consumer that "
        "writes into it pays for the write",
}

# ---------------------------------------------------------------------------
# Not judged yet
# ---------------------------------------------------------------------------

#: Ops priced ``zero`` where zero has not been decided, with what is unresolved.
#: Every entry names the issue that would settle it.  ``audit`` accepts these:
#: the state being guarded against is an op nobody looked at, not a question
#: somebody wrote down.
#:
#: Most of these share one shape.  ``_unary`` in ``ktir_cpu/dialects/_helpers.py``
#: applies its function to a whole ``Tile`` when given one, so a cast that reads
#: like a scalar conversion in the IR is an elementwise pass over a tile at run
#: time — and an elementwise pass is what every op in ``compute_float`` is priced
#: for.  Whether Spyre does measurable work for them cannot be settled from
#: RFC 0682: a conversion folded into the consumer's read is free, and a
#: materialized one is not.  That is a hardware question, so these are decisions
#: to be made rather than a fix to be applied.
UNJUDGED_ZERO_OPS: Dict[str, str] = {
    # -- tile-wide work, and it shows in kernels already on main (#211) ------
    "linalg.fill":
        "writes a scalar across the whole `outs` tile. Eight kernels under "
        "`examples/` use it. #211",
    "linalg.broadcast":
        "expands a tile along new dimensions — free if the consumer reads it "
        "strided, not free if it is materialized. #211",
    "arith.sitofp":
        "integer to float across the whole tile, via `_unary`. #211",

    # -- the rest of the cast cluster: same question, cheaper to be wrong ----
    "arith.extf": "widens every element of a tile, via `_unary`. #211",
    "arith.truncf": "narrows every element of a tile, via `_unary`. #211",
    "arith.convertf": "converts every element of a tile, via `_unary`. #211",
    "arith.extsi": "sign-extends every element of a tile. #211",
    "arith.extui": "zero-extends every element of a tile, via `_unary`. #211",
    "arith.trunci": "truncates every element of a tile, via `_unary`. #211",
    "arith.fptosi": "float to signed integer across the tile, via `_unary`. #211",
    "arith.fptoui": "float to unsigned integer across the tile, via `_unary`. #211",
    "arith.uitofp": "unsigned integer to float across the tile, via `_unary`. #211",
    "arith.bitcast":
        "reinterprets a tile's bits under another type. Free if it is a type "
        "relabel and not free if the data is copied; the handler uses "
        "`ndarray.view`, which is the free reading. #211",
    "arith.index_cast":
        "zero is defensible — the handler returns a Python `int`, so this is one "
        "scalar conversion rather than a tile's worth — but it has not been "
        "decided. #211",
    "arith.index_castui":
        "the same one-scalar conversion as `arith.index_cast`, undecided for the "
        "same reason. #211",

    # -- shape metadata, or a copy? ------------------------------------------
    "linalg.transpose":
        "free if it is a stride permutation, not free if the data moves; the "
        "handler calls `np.transpose(...).copy()`, which is the second reading. "
        "#211",
    "tensor.reshape":
        "reinterprets the same elements under a new shape. Free as metadata, not "
        "free if the layout is rebuilt. #211",
    "tensor.expand_shape": "the same question as `tensor.reshape`. #211",
    "tensor.collapse_shape": "the same question as `tensor.reshape`. #211",
    "tensor.extract_slice":
        "reads a strided sub-tensor. Free if the consumer reads the parent "
        "strided, not free if the slice is materialized. #211",
    "tensor.insert_slice":
        "writes a sub-tensor into a destination, which is a copy of the slice's "
        "worth of elements unless it folds into the producer. #211",
    "tensor.splat":
        "broadcasts one scalar across a whole tile — the same question as "
        "`linalg.fill`, and priced the same way. #211",
    "tensor.from_elements":
        "builds a small tensor from N scalar operands, so its cost is N element "
        "writes rather than a tile's worth. #211",
    "tensor.extract":
        "reads one element out of a tile. One scalar read, so zero is "
        "defensible, but undecided. #211",

    # -- not a pricing question at all ---------------------------------------
    "ktdp.coreid":
        "not an op to price: it is not in the authoritative `ktdp` dialect and "
        "survives only on the regex path, so the resolution is to reconcile or "
        "remove it rather than to give it a category. #88",
}


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

#: An op priced ``zero`` and listed in neither mapping.  The state AC7 exists to
#: reject: registering an op without a category makes it free, and nothing said so.
UNLISTED = "unlisted"
#: An op in both mappings — the record contradicts itself about whether the
#: question is settled.
BOTH = "both"
#: An op with a real category that is still listed here.  Someone priced it and
#: the entry outlived the decision; the fix is to delete the line.
PRICED = "priced"
#: An entry naming an op no longer in the registry, i.e. a rename or a removal
#: that left the record behind.
UNREGISTERED = "unregistered"


@dataclass(frozen=True)
class Finding:
    """One way the pricing record and the registry disagree.

    ``fix`` names the edit rather than describing the problem, for the same
    reason a ``Claim`` carries a ``closer``: the difference between a report that
    says something is wrong and a report that says what to do about it.
    """

    op: str
    kind: str
    detail: str
    fix: str

    def __str__(self) -> str:
        return f"{self.op}: {self.detail}\n    {self.fix}"


def zero_priced_ops() -> List[str]:
    """Every registered op whose ``latency_category`` is ``zero``."""
    return sorted(op for op in registry._REGISTRY
                  if registry.get_latency_category(op) == "zero")


def audit() -> List[Finding]:
    """Compare the record above against the registry.  Empty means agreement.

    Both directions are checked, because only one of them is the interesting
    one and the other is the one that rots.  A zero-priced op missing from the
    record is the defect this exists for; an entry that outlived its op, or its
    ``zero``, is how the record stops meaning anything.
    """
    findings: List[Finding] = []
    zero = set(zero_priced_ops())

    for op in sorted(zero - set(ZERO_COST_OPS) - set(UNJUDGED_ZERO_OPS)):
        findings.append(Finding(
            op, UNLISTED,
            "priced `zero` and listed in neither mapping, so its cost is the "
            "`@register()` default rather than a decision",
            "add it to ZERO_COST_OPS with the reason the hardware does no "
            "measurable work for it, or to UNJUDGED_ZERO_OPS with what is "
            "unresolved and the issue tracking it — both in "
            "ktir_cpu/kernelentry/pricing.py",
        ))

    for op in sorted(set(ZERO_COST_OPS) & set(UNJUDGED_ZERO_OPS)):
        findings.append(Finding(
            op, BOTH,
            "listed as free by decision and as not yet judged at the same time",
            "keep one of the two entries in ktir_cpu/kernelentry/pricing.py",
        ))

    for mapping, which in ((ZERO_COST_OPS, "ZERO_COST_OPS"),
                           (UNJUDGED_ZERO_OPS, "UNJUDGED_ZERO_OPS")):
        for op in sorted(mapping):
            if op not in registry._REGISTRY:
                findings.append(Finding(
                    op, UNREGISTERED,
                    f"listed in {which} but no longer has a `@register` handler",
                    f"remove it from {which} in ktir_cpu/kernelentry/pricing.py, "
                    "or restore the handler",
                ))
            elif op not in zero:
                category = registry.get_latency_category(op)
                findings.append(Finding(
                    op, PRICED,
                    f"listed in {which} but priced `{category}`, so the entry "
                    "outlived the decision it records",
                    f"remove it from {which} in ktir_cpu/kernelentry/pricing.py",
                ))

    return findings
