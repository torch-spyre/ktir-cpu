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

"""``docs/supported_ops.md`` — the same repository seen per op, not per kernel.

``docs/kernel_support.md`` answers five questions about each *kernel*, and the
last three of them need a declaration, so it is mostly a report about how far
that ledger reaches.  This document answers a narrower question that needs no
declaration at all — *does this repository have a handler for this op, and can
the MLIR frontend reach it* — and it answers it for every op in the registry,
including the ones no example exercises.

Every column is read from something the repository already maintains:

============================  ========================================
column                        source
============================  ========================================
executor handler              ``registry._REGISTRY``
MLIR frontend                 ``MLIRTypeAdapter._adapt_handlers``
reason, where not reachable   ``FRONTEND_UNSUPPORTED`` in
                              ``tests/mlir_frontend/test_registry_consistency.py``
kernels                       the walk in ``cli.survey``, inverted
pricing                       ``ZERO_COST_OPS`` / ``UNJUDGED_ZERO_OPS`` in
                              ``ktir_cpu/kernelentry/pricing.py``
============================  ========================================

The first two are the same two sets ``test_registry_consistency.py`` already
asserts about; this module does not invent a second criterion, it publishes the
one that is enforced.  The last comes from the ledger's own walk rather than a
second traversal of ``examples/``, so "which ops does this file use" has one
answer here and in ``docs/kernel_support.md``.

**The pricing column does not print the registry's answer.**  ``@register()``
defaults ``latency_category`` to ``"zero"``, so "this op is free" and "nobody
priced this op" are the same state there, and a column that printed it would
report a default as a decision.  What it prints instead is
``ktir_cpu/kernelentry/pricing.py``, which splits that state in two and is
audited against the registry by ``tests/test_kernelentry.py``.  The audit is
repository-wide rather than per kernel because the ops it would flag are the same
structural ones in every kernel — measured: 66 of a per-kernel prototype's 74
false positives came from asking it once per kernel.

One thing it still cannot see: a **wrong** category.  It compares against
``zero``, so an op billed to the wrong non-zero class passes it.  That needs a
reader who knows the op's semantics, which is why the reasons in ``pricing.py``
are written rather than generated.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List

import ktir_cpu.dialects  # noqa: F401 — import triggers @register side effects
from ktir_cpu.dialects import registry

from . import REPO_ROOT
from .pricing import UNJUDGED_ZERO_OPS, ZERO_COST_OPS

CONSISTENCY_TEST = (REPO_ROOT / "tests" / "mlir_frontend"
                    / "test_registry_consistency.py")

# Above this many files, "Where each op appears" summarises instead of naming
# every path: an op that every example uses is one fact, not one fact per file.
_MAX_NAMED_FILES = 3

# What an op's appearance in one example directory is worth as evidence.  The
# distinction matters for the ``kernels`` column: ``rfc/`` examples are
# specification cases the suite expects to fail execution, so an op seen only
# there has been parsed, not run.
CATEGORY_NOTE = {
    "ktir": "hand-written dialect cases; executed by the test suite",
    "latency": "small kernels for the latency tests; executed by the test suite",
    "rfc": "RFC 0682 specification examples; **expected to fail execution** "
           "(they carry absolute addresses rather than arguments), so an op "
           "seen only here is parsed, not run",
    "sdsc": "kernels from the SuperDSC lowering path; executed by the test suite",
    "triton-ktir": "kernels as the Triton → KTIR path emits them, i.e. captured "
                   "compiler output; executed by the test suite",
}


def _frontend_allowlist() -> Dict[str, str]:
    """Read ``FRONTEND_UNSUPPORTED`` out of the test that enforces it.

    It lives in the test rather than the package because it is a statement about
    a known divergence, not a fact the library needs at run time.  That makes it
    un-importable as a module path, so it is loaded by location — the same way
    this package already reads ``examples/`` by path.  Reading a checkout from
    library code is sound only because this package is a development tool and is
    not packaged: ``pyproject.toml`` excludes ``ktir_cpu.kernelentry`` from the
    wheel, so there is no install in which these paths are absent.
    """
    spec = importlib.util.spec_from_file_location(
        "_registry_consistency", CONSISTENCY_TEST)
    if spec is None or spec.loader is None:  # pragma: no cover — path is fixed
        raise ValueError(f"cannot load {CONSISTENCY_TEST}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return dict(module.FRONTEND_UNSUPPORTED)
    except AttributeError:  # pragma: no cover — guarded so a rename is loud
        raise ValueError(
            f"{CONSISTENCY_TEST.relative_to(REPO_ROOT)} no longer defines "
            "FRONTEND_UNSUPPORTED; update ktir_cpu/kernelentry/ops.py to read "
            "whatever replaced it.")


def _frontend_handlers() -> set:
    """Ops the MLIR frontend can adapt.

    Host-independent, which this document needs and the kernel report has to work
    for: ``_adapt_handlers`` is filled by ``@MLIRTypeAdapter.install`` decorators
    in this package, not by anything the MLIR bindings provide, and the module
    guards its own ``mlir_ktdp`` import.  So this table reads the same on a
    machine without the bindings, and a committed document generated there does
    not contradict one generated in CI.  Imported plainly rather than guarded for
    the same reason: if that ever stops being true, it should fail loudly instead
    of publishing "no op is reachable".
    """
    from ktir_cpu.mlir_frontend.parser import MLIRTypeAdapter

    return set(MLIRTypeAdapter._adapt_handlers)


def _pricing_cell(op: str) -> str:
    """What the cost model charges for *op*, with ``zero`` split into two answers.

    A real category is printed as itself.  ``zero`` is printed as the decision
    behind it, never as the registry's word for it, because the registry cannot
    tell a decision from its own default.  ``**unlisted**`` should not occur:
    ``tests/test_kernelentry.py`` fails on it, the same way a bare ``**no**`` in
    the frontend column is a test failure rather than a row.
    """
    if op not in registry._REGISTRY:
        return "—"
    category = registry.get_latency_category(op)
    if category != "zero":
        return f"`{category}`"
    if op in ZERO_COST_OPS:
        return "free, by decision"
    if op in UNJUDGED_ZERO_OPS:
        return "**not judged**"
    return "**unlisted**"


def render_ops_report(kernels_by_op: Dict[str, List[str]]) -> str:
    """The committed op-level report.  A pure function of the registries.

    ``kernels_by_op`` maps an op name to the kernels under ``examples/`` that use
    it, as ``cli.survey`` observed them.  It is passed in rather than recomputed
    so that both generated documents agree about what an op is: the parser's
    answer, read back off the ledger's claims.
    """
    executor = dict(registry._REGISTRY)
    frontend = _frontend_handlers()
    allowlist = _frontend_allowlist()

    all_ops = sorted(set(executor) | frontend | set(kernels_by_op))
    unhandled = [op for op in sorted(kernels_by_op) if op not in executor]
    unexercised = sorted(set(executor) - set(kernels_by_op))
    zero = [op for op in sorted(executor)
            if registry.get_latency_category(op) == "zero"]
    free = [op for op in zero if op in ZERO_COST_OPS]
    unjudged = [op for op in zero if op in UNJUDGED_ZERO_OPS]

    out = [
        "<!-- generated by "
        "`python -m ktir_cpu.kernelentry probe --all --write-report` -->",
        "",
        "# Supported operations",
        "",
        "Op-level truth about this interpreter, read from the registries "
        "themselves. `tests/test_kernelentry.py` fails while this file and the "
        "registries disagree, so it cannot go stale silently.",
        "",
        f"{len(executor)} ops have an execution handler. "
        f"{len(set(executor) & frontend)} of them are also reachable through "
        f"the MLIR frontend; {len(allowlist)} are deliberately not "
        "([why](#ops-not-reachable-through-the-mlir-frontend)). "
        f"{len(kernels_by_op)} are used by a kernel under `examples/`, leaving "
        f"{len(unexercised)} that no example exercises "
        "([which](#ops-no-example-exercises)).",
        "",
        f"{len(zero)} of them cost nothing. That is two facts, not one: "
        f"{len(free)} are [free by decision](#ops-that-are-free-by-decision) and "
        f"{len(unjudged)} are [priced zero with nobody having decided]"
        "(#ops-priced-zero-without-a-decision). `@register()` defaults "
        "`latency_category` to `zero`, so the registry itself cannot tell those "
        "apart — `ktir_cpu/kernelentry/pricing.py` is where they are split, and "
        "an op priced zero that appears in neither list fails "
        "`tests/test_kernelentry.py`.",
        "",
        "**Division of labour.** `docs/kernel_support.md` asks five questions "
        "about each *kernel*, and three of them need a declared entry — "
        "so it reports how far that ledger reaches. This file asks one question "
        "about each *op*, needs no declaration, and therefore covers the whole "
        "registry. `docs/gap_analysis.md` is the third: conformance against "
        "RFC 0682, judged by a reader rather than generated.",
        "",
        "## Matrix",
        "",
        "| op | executor handler | MLIR frontend | cost | kernels |",
        "|---|---|---|---|---|",
    ]
    for op in all_ops:
        has_exec = "yes" if op in executor else "**no**"
        if op in frontend:
            fe = "yes"
        elif op in allowlist:
            fe = "no, by design"
        else:
            fe = "**no**"
        n = len(kernels_by_op.get(op, ()))
        out.append(f"| `{op}` | {has_exec} | {fe} | {_pricing_cell(op)} | "
                   f"{n or 'none'} |")

    out += [
        "",
        "### How to read the columns",
        "",
        "- **executor handler** — a `@register` handler exists, so the "
        "interpreter can execute the op. `no` means it cannot; such an op is "
        "listed here only because the MLIR frontend or an example mentions it.",
        "- **MLIR frontend** — an `@MLIRTypeAdapter.install` handler exists, so "
        "the op survives the real MLIR parser and its `verify()`. `no, by "
        "design` is an entry in the allowlist below. A bare **no** should not "
        "occur: `tests/mlir_frontend/test_registry_consistency.py` fails on it.",
        "- **cost** — what the latency model charges. A named category is the "
        "one `@register` gave it. `free, by decision` and `**not judged**` are "
        "both `zero` in the registry, split apart by "
        "`ktir_cpu/kernelentry/pricing.py`: the first has a written reason why "
        "the hardware does no measurable work, the second has a written "
        "statement of what is unresolved and the issue tracking it. A cost "
        "column can only catch a *missing* price — an op billed to the wrong "
        "non-zero unit passes it, and needs a reader.",
        "- **kernels** — how many files under `examples/` use the op; "
        "[the files themselves](#where-each-op-appears) are listed below, with "
        "what each directory is worth as evidence. `none` means the op is "
        "registered but no example exercises it, so only unit tests, if any, "
        "cover it.",
        "",
        "What this file does not say: whether a handler is **correct**, and "
        "whether a particular attribute or type of the op is supported. The unit "
        "here is the op name — `ktdp.construct_access_tile` having a handler "
        "says nothing about a particular `base_map` or `coordinate_set` reaching "
        "the conclusion the specification does. `docs/gap_analysis.md` tracks "
        "that.",
        "",
        "## Ops not reachable through the MLIR frontend",
        "",
        "Executor ops with no MLIR frontend handler, and the reason each is "
        "allowed to stay that way. This is the allowlist "
        "`tests/mlir_frontend/test_registry_consistency.py` enforces — an op "
        "missing from both the frontend and this list fails that test.",
        "",
        "| op | reason |",
        "|---|---|",
    ]
    for op, reason in sorted(allowlist.items()):
        out.append(f"| `{op}` | {' '.join(reason.split())} |")

    if unhandled:
        out += [
            "",
            "## Ops used by an example with no execution handler",
            "",
            "These appear in a file under `examples/` that the interpreter "
            "cannot execute as written:",
            "",
        ]
        out += [f"- `{op}` — {', '.join(kernels_by_op[op])}" for op in unhandled]

    out += [
        "",
        "## Ops that are free by decision",
        "",
        f"{len(free)} ops the cost model charges nothing for, and the reason each "
        "one does no measurable work. Four kinds: a value that exists at compile "
        "time, a terminator that only names values, addressing metadata that "
        "computes where data is without moving it, and an orchestrator whose "
        "region is charged op by op.",
        "",
        "| op | why it is free |",
        "|---|---|",
    ]
    out += [f"| `{op}` | {' '.join(ZERO_COST_OPS[op].split())} |" for op in free]

    out += [
        "",
        "## Ops priced zero without a decision",
        "",
        f"{len(unjudged)} ops that cost nothing today because "
        "`latency_category` defaults to `zero`, not because anyone decided they "
        "are free. A kernel using one of these reports a lower cost than the "
        "hardware would. They are listed rather than fixed because each is a "
        "hardware question RFC 0682 does not settle — a conversion folded into "
        "the consumer's read is free and a materialized one is not — and "
        "repricing one changes the committed derivations of every kernel using "
        "it.",
        "",
        "| op | what is unresolved | kernels |",
        "|---|---|---|",
    ]
    out += [f"| `{op}` | {' '.join(UNJUDGED_ZERO_OPS[op].split())} | "
            f"{len(kernels_by_op.get(op, ())) or 'none'} |" for op in unjudged]

    out += [
        "",
        "## Ops no example exercises",
        "",
        f"{len(unexercised)} registered ops that no file under `examples/` uses. "
        "Not a defect on its own — an op can be covered by a unit test — but it "
        "is where a handler nothing has ever run would hide:",
        "",
    ]
    out += [f"- `{op}`" for op in unexercised] or ["None."]

    out += [
        "",
        "## Where each op appears",
        "",
        f"Up to {_MAX_NAMED_FILES} files are named; above that the directories "
        "and a count, "
        "because the identity of the file is what matters when there are few and "
        "the coverage class is what matters when there are many. Paths are "
        "relative to `examples/`, and what each directory is worth as evidence:",
        "",
    ]
    # Keys arrive repo-relative (``examples/ktir/x.mlir``); the category is the
    # segment after ``examples/``, so strip that first rather than splitting on
    # the leading slash and getting ``examples`` for every one of them.
    def _category(rel: str) -> str:
        return rel.split("/")[1] if rel.startswith("examples/") else rel

    seen_dirs = sorted({_category(k) for ks in kernels_by_op.values() for k in ks})
    for name in seen_dirs:
        note = CATEGORY_NOTE.get(
            name, "**no note** — add one to `CATEGORY_NOTE` in "
                  "`ktir_cpu/kernelentry/ops.py`")
        out.append(f"- `{name}/` — {note}")
    out.append("")
    for op in sorted(kernels_by_op):
        files = sorted(kernels_by_op[op])
        if len(files) <= _MAX_NAMED_FILES:
            where = ", ".join(k.replace("examples/", "") for k in files)
        else:
            dirs = ", ".join(f"{d}/" for d in sorted({_category(k)
                                                       for k in files}))
            where = f"{len(files)} files in {dirs}"
        out.append(f"- `{op}` — {where}")
    out.append("")
    return "\n".join(out)
