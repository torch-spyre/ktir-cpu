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

"""``probe`` / ``adopt`` / ``verify``.

    python -m ktir_cpu.kernelentry probe  examples/latency/matmul_small.mlir
    python -m ktir_cpu.kernelentry probe  matmul_small
    python -m ktir_cpu.kernelentry probe  --all
    python -m ktir_cpu.kernelentry adopt  matmul_small
    python -m ktir_cpu.kernelentry verify --all

A kernel is addressed either way, and the difference is the point of the first
line: a ``.mlir`` path is read cold, with no declaration consulted even if one
exists, which is what makes the first question answerable before any paperwork.  A
bare name is the declared entry in ``examples/entries.py``, with all five questions
in reach.

``probe`` writes nothing.  ``adopt`` writes only what it owns — one kernel's section
of the committed cost document — and never edits the interpreter: when a kernel
needs a new handler or an op repriced, it prints the edit for a person to make.  A
tool that silently changes what the simulator charges would be changing the answer
to the question it is being asked.

``verify`` shares :func:`ktir_cpu.kernelentry.ledger.probe` with
``tests/test_kernelentry.py``, so the local loop and the gate cannot disagree.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import (
    BLOCKING, CLOSED, COST, DEFERRED, EXAMPLES_DIR, FUNCTION, OPEN, REPO_ROOT,
    SKIP, UNDETERMINED, WAIVED, KernelEntry, registered,
)
from .conformance import FRONTEND_REJECTS
from .ledger import Ledger, probe
from .pricing import audit, zero_priced_ops

_STATE_LABEL = {
    CLOSED: "closed", OPEN: "OPEN", UNDETERMINED: "UNDETERMINED",
    DEFERRED: "deferred", WAIVED: "waived", SKIP: "skip",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

#: Declarations already executed, so that loading is idempotent.  These modules are
#: not put in ``sys.modules`` — they are loaded by file location, not by name — so
#: nothing else would stop a second call from executing the same file again.
_LOADED: set = set()


def load_declaration(path: Path) -> None:
    """Execute one declaration file, registering whatever it declares.

    ``examples/`` is not an importable package — ``examples/triton-ktir`` is not
    even a legal identifier for an ``import`` statement — so the file is loaded by
    location.  That constraint is also why the declarations are one table rather
    than a module per kernel: a declaration under ``examples/`` cannot import a
    sibling, so anything shared between two kernels has to live with them.

    Loading the same file twice is a no-op rather than an error.  Two test modules
    call :func:`discover_all` at import time, and ``pytest tests/`` collects both
    into one session — without this, the second one re-executes every declaration
    and ``register_entry`` rejects the duplicate name, taking down collection for
    the whole run.  The duplicate-name guard stays: two *different* files claiming
    one name is the hazard it is there for.
    """
    resolved = path.resolve()
    if resolved in _LOADED:
        return
    spec = importlib.util.spec_from_file_location(f"_kernelentry_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    _LOADED.add(resolved)
    spec.loader.exec_module(module)


#: Declarations are found by this name, at any depth under ``examples/``.
DECLARATION_FILE = "entries.py"


def discover_all() -> None:
    """Register every declared kernel — today, the one ``examples/entries.py``.

    A glob for one fixed name, which is neither of the two extremes.  A single
    hard-coded path would make a second table (a vendored kernel set, a fork's
    own) silently do nothing, and silence is the failure mode this whole module
    exists to remove — ``undeclared_kernels`` already walks the same tree for the
    same reason.  Importing *any* ``.py`` under ``examples/`` is the other
    extreme, and too much: it makes every helper file a declaration table, so a
    module put there for one kernel's reference implementation gets executed as
    one.  Naming the file is the whole convention, and it is what
    ``docs/kernelentry.md`` tells a contributor to open.
    """
    for path in sorted(EXAMPLES_DIR.rglob(DECLARATION_FILE)):
        load_declaration(path)


def undeclared_kernels() -> List[str]:
    """Kernels under ``examples/`` that no entry declares.

    ``probe --all`` can only report on entries that exist, so without this the
    report would describe a handful of kernels and read as though it described
    the repository.  Coverage is only honest when what is *not* covered is on the
    page too.
    """
    declared = {entry.mlir_path.resolve() for entry in registered().values()}
    found = []
    for path in sorted(EXAMPLES_DIR.rglob("*")):
        if path.suffix not in (".mlir", ".ktir") or path.resolve() in declared:
            continue
        found.append(str(path.relative_to(REPO_ROOT)))
    return found


def function_names(path: Path) -> Optional[List[str]]:
    """The functions the regex parser reads out of *path*, ``None`` if it rejects it.

    Separated from ``entry_for_bare_kernel`` because two callers want different
    things from the same read.  Somebody probing one file wants the parser's own
    exception, which says where in the file it gave up; the repository report wants
    that rejection as a cell, because a walk that stops at the first unreadable file
    reports nothing about the other thirty.
    """
    from ktir_cpu import KTIRInterpreter

    interp = KTIRInterpreter()
    try:
        interp.load(path.read_text())
    except Exception:  # noqa: BLE001 — any rejection is the answer, not an error
        return None
    return sorted(interp.module.functions)


def entry_for_bare_kernel(path: Path, func: Optional[str] = None) -> KernelEntry:
    """A throwaway entry for a ``.mlir`` that has no declaration yet.

    The first question a kernel raises — which of its ops the simulator has no
    handler for, and whether it survives both parse paths — needs no declaration to
    answer. Requiring one first would mean writing the paperwork before finding out
    whether the kernel can run at all, which is the opposite of the order that
    helps. The claims that do need a declaration report that as their reason.
    """
    from ktir_cpu import KTIRInterpreter

    if func is None:
        # Read through the interpreter directly rather than through
        # ``function_names``, so a file this parser rejects raises the parser's own
        # error here.  Somebody probing a kernel they have just written needs to
        # know where the parse gave up, which a swallowed exception cannot say.
        interp = KTIRInterpreter()
        interp.load(path.read_text())
        names = sorted(interp.module.functions)
        if not names:
            # Distinct from the several-functions case below: telling somebody to
            # name a function in a file that declares none sends them to look for
            # something that is not there.
            raise SystemExit(
                f"{path.name} is read by the parser but declares no function, so "
                "there is no kernel in it to probe"
            )
        if len(names) > 1:
            raise SystemExit(
                f"{path.name} declares {len(names)} functions ({', '.join(names)}); "
                "name one with --func"
            )
        func = names[0]
    # Relative under ``examples/``, so a cold probe and a declared entry name the
    # same kernel the same way; absolute for a kernel versioned outside this
    # repository, which a cold read takes because it writes nothing.
    resolved = path.resolve()
    where = (resolved.relative_to(EXAMPLES_DIR)
             if resolved.is_relative_to(EXAMPLES_DIR) else resolved)
    return KernelEntry(name=f"{path.stem} (no declaration)", func=func,
                       path=str(where))


def _resolve(target: Optional[str], use_all: bool,
             func: Optional[str] = None) -> List[KernelEntry]:
    """The entries one invocation is about.

    Two ways to name one kernel, and the difference between them is deliberate.  A
    ``.mlir`` path is read cold — no declaration is consulted even if one exists,
    which is what makes the first question answerable before any paperwork does.  A
    bare name is the declared entry, with all five questions in reach.
    """
    if use_all:
        discover_all()
        return [registered()[name] for name in sorted(registered())]
    if not target:
        raise SystemExit("name a declared kernel, or give a .mlir path, or --all")
    path = Path(target)
    if not path.exists():
        path = REPO_ROOT / target
    if path.suffix in (".mlir", ".ktir"):
        return [entry_for_bare_kernel(path, func)]
    discover_all()
    entries = registered()
    if target not in entries:
        raise SystemExit(
            f"no declared kernel named {target!r}. Declared: "
            f"{', '.join(sorted(entries))}. An undeclared kernel is addressed by "
            f"its path instead."
        )
    return [entries[target]]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _tally_text(ledger: Ledger, leg: str) -> str:
    """Counts per state, for the console.

    The committed report does not use this: it is compared verbatim, and
    ``parse.frontend`` reads ``closed`` where the MLIR bindings are installed and
    ``skip`` where they are not, so any tally of it would depend on the machine that
    generated the document. The report answers that claim from a committed record
    instead — see ``_support_rows``.
    """
    tally = ledger.tally(leg)
    total = sum(tally.values())
    if not total:
        return "no claims"
    parts = [f"{tally.get(CLOSED, 0)}/{total} closed"]
    for state in (OPEN, UNDETERMINED, DEFERRED, WAIVED, SKIP):
        if tally.get(state):
            parts.append(f"{tally[state]} {_STATE_LABEL[state]}")
    return ", ".join(parts)


def render_ledger(ledger: Ledger, *, verbose: bool) -> str:
    entry = ledger.entry
    lines = [
        f"{entry.name}  function {entry.func}",
        f"  function leg   {_tally_text(ledger, FUNCTION)}",
        f"  cost leg       {_tally_text(ledger, COST)}",
    ]
    shown = ledger.claims if verbose else [
        c for c in ledger.claims if c.state in BLOCKING or c.state == DEFERRED]
    for claim in sorted(shown, key=lambda c: c.sort_key()):
        lines.append(f"  {_STATE_LABEL[claim.state]:<13} {claim.id}")
        if claim.detail:
            lines.append(f"                {claim.detail}")
        if claim.closer and claim.state in BLOCKING:
            lines.append(f"                -> {claim.closer}")
    return "\n".join(lines)


#: How many rows of "Read, not declared" have an empty ``execute_kwargs``.  The
#: figure belongs to ``tests/conftest.py::EXAMPLE_PARAMS``, and a document
#: renderer that imports the test tree to compute one integer would point the
#: dependency the wrong way round.  Instead
#: ``tests/test_kernelentry.py::test_support_report_covers_conftest_examples``
#: pins it from the side that imports both, so it cannot drift unnoticed.
EMPTY_EXECUTE_KWARGS_ROWS = 8

_COLUMNS = ("read: regex", "read: frontend", "runs", "output",
            "cost: derivation")


def _issue(detail: str) -> str:
    """The issue an excuse names, for a cell too narrow to carry the whole reason.

    ``KernelEntry`` will not accept a deferral that names no issue, so there is no
    fallback here on purpose: a ``?`` in this column would put the report's own
    inability to read a reason where a reader expects a fact about the kernel.
    """
    match = re.search(r"#\d+", detail or "")
    if match is None:
        raise ValueError(
            f"deferral reason names no issue as #N: {detail!r}. This is "
            "supposed to be unreachable — KernelEntry.__post_init__ rejects it."
        )
    return match.group(0)


def _cell(claims: Sequence) -> str:
    """One column of one kernel, from the claims that column covers.

    Deferred outranks open in the same cell: a gap somebody wrote down and tracked
    is a different fact from one nobody has looked at, and the sections below carry
    the reason either way. A waiver is reported rather than folded into ``yes``,
    because "this check does not apply here" is a decision a reader may want to
    disagree with.
    """
    if not claims:
        return "—"
    deferred = [c for c in claims if c.state == DEFERRED]
    if deferred:
        return f"deferred {_issue(deferred[0].detail)}"
    if any(c.state in BLOCKING for c in claims):
        return "**no**"
    waived = sum(1 for c in claims if c.state == WAIVED)
    if waived == len(claims):
        return "waived"
    return "yes" if not waived else f"yes ({waived} waived)"


def _frontend_cell(rel: str, ledger=None) -> str:
    """Whether the MLIR frontend accepts this file, from a committed record.

    Never from the generating run. The check needs the optional MLIR bindings, so a
    machine without them would print ``skip`` for all thirty kernels and the document
    would describe the machine. ``conformance.py`` carries the rejections and a
    declaration carries its own excuse; both are in the repository, and
    ``tests/mlir_frontend/test_kernelentry_adapt.py`` holds the first to the real
    frontend in both directions.
    """
    if ledger is not None:
        excused = [c for c in ledger.claims
                   if c.id == "parse.frontend" and c.state in (DEFERRED, WAIVED)]
        if excused:
            return _cell(excused)
    return "**no**" if rel in FRONTEND_REJECTS else "yes"


def _row_without_one_kernel(rel: str) -> Optional[Tuple[str, List[str]]]:
    """The undeclared row for a file no one kernel can be read out of.

    ``None`` when one can, and the caller probes it as usual.  Two ways this
    happens and they are different answers, which is why the branch is not one: the
    regex parser rejects the file, and the first column *is* that answer; or it
    reads the file and finds no single function to probe, and both parse answers
    stand while the op count does not.  Folding the two together and printing
    ``yes`` puts a file nothing could read in the column that says it was read,
    which is the one outcome this ledger exists to make impossible.
    """
    names = function_names(REPO_ROOT / rel)
    if names is None:
        return ("?", ["**no**", _frontend_cell(rel)])
    if len(names) != 1:
        return ("?", ["yes", _frontend_cell(rel)])
    return None


class Survey(NamedTuple):
    """Everything one walk over ``examples/`` knows, for both generated documents."""

    declared: List[Tuple[str, str, List[str]]]
    undeclared: List[Tuple[str, str, List[str]]]
    kernels_by_op: Dict[str, List[str]]
    handlers: int
    unhandled: int


def survey(ledgers: Sequence[Ledger]) -> Survey:
    """One row per kernel under ``examples/``, declared or not, plus the op inversion.

    A row is ``(path, ops, cells)``.  Declared and undeclared kernels are separate
    lists because the five questions are asked *in order*: the last three need input
    that only a declaration supplies, so an undeclared kernel answers a **prefix** of
    the five rather than a row with holes in it.  An earlier version printed one
    table and filled the unasked cells with em-dashes; at 29 of 31 rows that was most
    of the document by area, and it read as a repository that supports almost nothing
    — the opposite of what those cells actually say.  Two tables, each with only the
    columns whose question was asked, cannot be misread that way.

    ``kernels_by_op`` is the same walk inverted, and it is what
    ``render_ops_report`` publishes.  Deriving it here rather than traversing
    ``examples/`` a second time keeps one answer to "which ops does this file use" —
    the parser's, read back off the claims — instead of two that can disagree.

    ``handlers`` and ``unhandled`` are repo-wide and come free with the walk: the
    ``ops`` column is how many distinct ops a kernel uses, and asking whether the
    interpreter has a handler for each is the same question.
    """
    by_path = {}
    for ledger in ledgers:
        rel = str(ledger.entry.mlir_path.resolve().relative_to(REPO_ROOT.resolve()))
        by_path[rel] = ledger

    declared: List[Tuple[str, str, List[str]]] = []
    undeclared: List[Tuple[str, str, List[str]]] = []
    kernels_by_op: Dict[str, List[str]] = {}
    handlers = unhandled = 0

    def _ops(claims) -> List[str]:
        """The op names behind this kernel's handler claims."""
        return [c.id[len("op."):-len(".handler")]
                for c in claims if c.id.endswith(".handler")]

    for rel in sorted(set(by_path) | set(undeclared_kernels())):
        ledger = by_path.get(rel)
        if ledger is None:
            row = _row_without_one_kernel(rel)
            if row is not None:
                undeclared.append((rel, row[0], row[1]))
                continue
            ledger = probe(entry_for_bare_kernel(REPO_ROOT / rel))

        ops = _ops(ledger.claims)
        handlers += len(ops)
        unhandled += sum(1 for c in ledger.claims
                         if c.id.endswith(".handler") and c.state != CLOSED)
        for op in ops:
            kernels_by_op.setdefault(op, []).append(rel)

        pick = lambda pred: [c for c in ledger.claims if pred(c.id)]
        regex = _cell(pick(lambda i: i == "parse.regex"))
        if rel in by_path:
            declared.append((rel, str(len(ops)), [
                regex,
                _frontend_cell(rel, ledger),
                _cell(pick(lambda i: i == "exec.runs")),
                _cell(pick(lambda i: i.startswith("out."))),
                _cell(pick(lambda i: i == "cost.derivation")),
            ]))
        else:
            undeclared.append((rel, str(len(ops)),
                               [regex, _frontend_cell(rel)]))

    return Survey(declared, undeclared, kernels_by_op, handlers, unhandled)


def _count(n: int, total: int) -> str:
    """``All 31 are`` or ``29 of 31 are`` — never ``All 29`` out of thirty-one.

    The first phrasing is only available when the count is the whole set; used
    otherwise it reads as though nothing were missing, in the one sentence a reader
    takes the repository's state from.
    """
    return f"All {total} are" if n == total else f"{n} of {total} are"


def _status_line(sv: Survey, green, blocking) -> str:
    """Where the repository stands, in one sentence, before any vocabulary.

    The mass goes first — kernels, ops behind them, how many run — because a reader
    who meets the narrow tables first counts rows rather than reading them.
    """
    rows = sv.declared + sv.undeclared
    claims = "claim is" if len(blocking) == 1 else "claims are"
    # ``handlers`` sums the per-kernel op counts, so it is what the ``ops`` columns
    # add up to rather than a count of distinct ops in the registry.
    ops = ("has a handler for every one" if not sv.unhandled
           else f"has a handler for {sv.handlers - sv.unhandled} of them")
    return (
        f"{len(rows)} kernels under `examples/`, using {sv.handlers} ops between "
        f"them; the interpreter {ops}. "
        f"{_count(sum(1 for r in rows if r[2][0] == 'yes'), len(rows))} read by "
        f"the regex parser and "
        f"{sum(1 for r in rows if r[2][1] == 'yes')} are also accepted by the MLIR "
        f"frontend and MLIR's own verifier. "
        f"{len(sv.declared)} are declared to this ledger, "
        f"{'all' if len(green) == len(sv.declared) else len(green)} of which "
        f"answer all five questions below; {len(blocking)} {claims} open across "
        f"the repository."
    )


def render_report(ledgers: Sequence[Ledger],
                  sv: Optional[Survey] = None) -> str:
    """The committed support report: how far this repository supports what.

    Written to ``docs/kernel_support.md`` and compared there on lock-file
    discipline, so "what is supported" has a history rather than being re-derived.
    ``sv`` is accepted so ``generated_docs`` can walk ``examples/`` once for both
    documents; passing nothing walks it here.
    """
    sv = sv if sv is not None else survey(ledgers)
    blocking = [(led, c) for led in ledgers for c in led.blocking
                if not c.env_dependent]
    green = [r for r in sv.declared if not any(c == "**no**" for c in r[2])]

    lines = [
        "# Kernel support",
        "",
        "Generated by `python -m ktir_cpu.kernelentry probe --all --write-report`;",
        "`tests/test_kernelentry.py` fails while it is stale.",
        "",
        # The state of the repository, before the vocabulary for describing it.
        _status_line(sv, green, blocking),
        "",
        "Support is not one property. It is five questions asked in order, and the "
        "columns are them: "
        "**read: regex** — `KTIRInterpreter.load` accepts the file; "
        "**read: frontend** — the MLIR frontend accepts it and MLIR's own "
        "verifier passes; **runs** — it executes on the declared grid without "
        "overflowing LX; **output** — every tensor it writes matches a "
        "reference computed in f32, and none is silently all-zero; "
        "**cost: derivation** — the committed per-tensor cost breakdown still "
        "matches what the model reports.",
        "",
        "A cell reads `yes` when the check ran and passed, **`no`** when it ran "
        "and failed or could not be evaluated, `deferred #N` against a tracked "
        "issue, and `waived` where the check does not apply to that kernel. A "
        "kernel is fully supported when no cell in its row is **`no`**.",
        "",
        "`docs/kernelentry.md` is how to declare a kernel and what a declaration "
        "has to supply. `docs/supported_ops.md` is the same repository seen per "
        "op rather than per kernel.",
        "",
        f"## Declared to this ledger ({len(sv.declared)})",
        "",
        "| kernel | ops | " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * (len(_COLUMNS) + 2),
    ]
    for rel, ops, cells in sv.declared:
        lines.append(f"| `{rel}` | {ops} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Read, not declared ({len(sv.undeclared)})",
        "",
        "Both reading questions are answered for these, and every op they use has "
        "an execution handler. The other three have not been *asked*, which is not "
        "the same as answered no: they need what only a declaration supplies — "
        "tensors to drive the kernel with, and a reference independent of the "
        "simulator — so their columns are absent here rather than empty.",
        "",
        "`tests/` already drives every file below, from "
        "`tests/conftest.py::EXAMPLE_PARAMS`, but what that listing supplies is not "
        "a declaration waiting to be copied: `execute_kwargs` is empty for "
        f"{EMPTY_EXECUTE_KWARGS_ROWS} of them and carries raw HBM element indices, or a "
        "scalar size, for the rest. `docs/kernelentry.md` groups these files by "
        "which reason applies.",
        "",
        "| kernel | ops | read: regex | read: frontend |",
        "|---|---|---|---|",
    ]
    for rel, ops, cells in sv.undeclared:
        lines.append(f"| `{rel}` | {ops} | " + " | ".join(cells) + " |")

    if any(ops == "?" for _, ops, _ in sv.undeclared):
        lines += ["",
                  "`?` — no one kernel to count ops for, either because the file "
                  "holds several functions or because the regex parser does not "
                  "read it at all. The **read: regex** cell says which."]

    def _listed(state: str):
        """Detail rows for one excuse state.

        An applied excuse clears ``env_dependent`` — it overrides any state but
        ``closed`` — so this is the same filter the blocking list uses, not a
        special case.
        """
        return [(led, c) for led in ledgers for c in led.claims
                if c.state == state and not c.env_dependent]

    deferred = _listed(DEFERRED)
    waived = _listed(WAIVED)

    lines += ["", f"## Open and undetermined ({len(blocking)})", ""]
    if not blocking:
        lines.append("None.")
    for led, claim in sorted(blocking, key=lambda kv: kv[1].sort_key()):
        lines.append(f"- `{led.entry.name}` — **{claim.id}** "
                     f"({_STATE_LABEL[claim.state]}): {claim.detail}")

    lines += ["", f"## Deferred ({len(deferred)})", ""]
    if not deferred:
        lines.append("None.")
    for led, claim in deferred:
        lines.append(f"- `{led.entry.name}` — **{claim.id}**: {claim.detail}")

    # Grouped by reason: the same waiver on several kernels is one decision, and
    # listing it once per kernel pads the document without adding anything.
    by_reason: Dict[str, List[str]] = {}
    for led, claim in waived:
        by_reason.setdefault(claim.detail, []).append(f"{led.entry.name}:{claim.id}")
    lines += ["", f"## Waived ({len(waived)})", ""]
    if not waived:
        lines.append("None.")
    for reason in sorted(by_reason):
        where = ", ".join(f"`{w}`" for w in sorted(by_reason[reason]))
        lines.append(f"- {where} — {reason}")

    rejected = sorted(rel for rel, _, cells in sv.declared + sv.undeclared
                      if cells[1] == "**no**" and rel in FRONTEND_REJECTS)
    # Split by state: a declared kernel's rejection is a deferred claim and is
    # listed above, which a reader counting kernels cannot tell unless it is said.
    also_deferred = sorted(rel for rel, _, cells in sv.declared
                           if cells[1].startswith("deferred"))
    lines += ["", f"## Not accepted by the MLIR frontend ({len(rejected)})", ""]
    lines.append("A record, not a verdict: it does not say whether the file or "
                 "the parser reading it should change.")
    if also_deferred:
        lines.append("")
        lines.append("No kernel is listed twice: a declared kernel's rejection is a "
                     "deferred `parse.frontend` claim, listed under **Deferred** "
                     f"above rather than here ({len(also_deferred)} of them).")
    lines.append("")
    if not rejected:
        lines.append("None.")
    # Grouped by reason, as the waived list is: one gap reached through several
    # kernels is one gap, and repeating its sentence per kernel says nothing more.
    rejects_by_reason: Dict[str, List[str]] = {}
    for rel in rejected:
        rejects_by_reason.setdefault(FRONTEND_REJECTS[rel], []).append(rel)
    for reason in sorted(rejects_by_reason, key=lambda r: rejects_by_reason[r][0]):
        where = ", ".join(f"`{w}`" for w in rejects_by_reason[reason])
        lines.append(f"- {where} — {reason}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------

def generated_docs(ledgers: Sequence[Ledger]) -> Dict[Path, str]:
    """Every document this tool owns, as ``path -> contents``.

    Both are derived from one walk over ``examples/``: they are two views of the
    same repository — per kernel and per op — and a second walk could disagree
    with the first about which ops a kernel uses.

    Returning the pair rather than writing it is what lets ``verify`` and
    ``tests/test_kernelentry.py`` check staleness with the code that writes them,
    so a document cannot be checked against a different renderer than the one
    that generated it.
    """
    from .ops import render_ops_report

    sv = survey(ledgers)
    docs = REPO_ROOT / "docs"
    return {
        docs / "kernel_support.md": render_report(ledgers, sv),
        docs / "supported_ops.md": render_ops_report(sv.kernels_by_op),
    }


def cmd_probe(args: argparse.Namespace) -> int:
    entries = _resolve(args.target, args.all, args.func)
    ledgers = [probe(entry) for entry in entries]
    for ledger in ledgers:
        print(render_ledger(ledger, verbose=args.verbose))
        print()

    if args.write_report:
        if not args.all:
            raise SystemExit("--write-report needs --all: a partial report would "
                             "claim the kernels it omits are absent")
        for path, text in generated_docs(ledgers).items():
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    blocking = sum(len(led.blocking) for led in ledgers)
    print(f"{len(ledgers)} kernel(s), {blocking} open or undetermined claim(s)"
          f"{_skip_note(ledgers)}")
    if args.all:
        # Repository-wide, so it is asked once and only when the walk was whole.
        findings = audit()
        print(f"{len(findings)} pricing finding(s) against "
              f"{len(zero_priced_ops())} zero-priced op(s)")
        for finding in findings:
            print(f"  {finding}")
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    """Write the files this tool owns, and print the edits it will not make."""
    from .derivation import DOCUMENT, render_derivation, update_document

    entries = _resolve(args.target, args.all, args.func)
    path = REPO_ROOT / DOCUMENT
    for entry in entries:
        print(f"{entry.name}:")

        ledger = probe(entry)
        if ledger.report is None:
            print("  the kernel did not run, so no derivation was written:")
            for claim in ledger.blocking:
                print(f"    {claim.id}: {claim.detail}")
        else:
            body = render_derivation(entry, ledger, dict(entry.gate_params))
            if update_document(path, entry.name, body):
                print(f"  wrote the {entry.name} section of {DOCUMENT} — read it "
                      "before committing; it is the attribution a reviewer confirms")
            else:
                print(f"  the {entry.name} section of {DOCUMENT} is unchanged")

        _print_manual_edits(ledger)
    return 0


def _print_manual_edits(ledger: Ledger) -> None:
    """Say what a human has to change, with the file, and change nothing."""
    remaining = [c for c in ledger.blocking
                 if not c.id.startswith("cost.derivation")]
    if not remaining:
        return
    print("  edits for you to make (this tool does not touch the interpreter):")
    for claim in sorted(remaining, key=lambda c: c.sort_key()):
        print(f"    {claim.id}")
        print(f"      {claim.detail}")
        if claim.closer:
            print(f"      in: {claim.closer}")


def _skip_note(ledgers: Sequence[Ledger]) -> str:
    """What a clean run did not actually check, for the line that says it is clean.

    ``skip`` is not blocking — the claim needs an optional dependency, and CI has
    it — but a summary that reports only the kernels leaves "fully supported"
    standing on checks this machine never ran.  Naming them is the difference
    between a gate that passed and a gate that was not fully evaluated here.
    """
    skipped = [c for led in ledgers for c in led.claims if c.state == SKIP]
    if not skipped:
        return ""
    ids = sorted({c.id for c in skipped})
    return (f" — {len(skipped)} claim(s) skipped on this machine "
            f"({', '.join(ids)}); CI is where that layer is checked")


def _repo_wide_problems(ledgers: Sequence[Ledger]) -> List[str]:
    """Checks that belong to the repository rather than to any one kernel.

    Both are here for the same reason: they are true or false once, not once per
    kernel.  The pricing audit was measured to be noise per kernel — 66 of a
    prototype's 74 false positives — and a stale generated document is a fact
    about the document.
    """
    problems: List[str] = []
    for path, expected in generated_docs(ledgers).items():
        rel = path.relative_to(REPO_ROOT)
        if not path.exists():
            problems.append(f"{rel} is missing — regenerate with "
                            "`probe --all --write-report`")
        elif path.read_text() != expected:
            problems.append(f"{rel} is stale — regenerate with "
                            "`probe --all --write-report`")
    problems += [str(finding) for finding in audit()]
    return problems


def cmd_verify(args: argparse.Namespace) -> int:
    entries = _resolve(args.target, args.all, args.func)
    ledgers = [probe(entry) for entry in entries]
    bad_kernels = 0
    for ledger in ledgers:
        if ledger.clean:
            print(f"ok    {ledger.entry.name}")
            continue
        bad_kernels += 1
        print(render_ledger(ledger, verbose=False))

    problems = _repo_wide_problems(ledgers) if args.all else []
    for problem in problems:
        print(problem)

    if bad_kernels or problems:
        # Counted separately because they are different failures with different
        # fixes: a kernel with an open claim needs work on the kernel or the
        # interpreter, and a repository-wide problem is one edit in one file.
        parts = []
        if bad_kernels:
            parts.append(f"{bad_kernels} kernel(s) with open claims")
        if problems:
            parts.append(f"{len(problems)} repository-wide problem(s)")
        print(f"\n{', '.join(parts)}")
        return 1
    print(f"\n{len(ledgers)} kernel(s) fully supported{_skip_note(ledgers)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ktir_cpu.kernelentry",
        description="What it takes for this simulator to fully support a kernel.")
    sub = parser.add_subparsers(dest="verb", required=True)

    for name, handler, helptext in (
        ("probe", cmd_probe, "compute the ledger; writes nothing"),
        ("adopt", cmd_adopt, "write the files this tool owns; print the rest"),
        ("verify", cmd_verify, "gate: fail while any claim is open"),
    ):
        child = sub.add_parser(name, help=helptext)
        child.add_argument("target", nargs="?",
                           help="a declared kernel's name, or the path of any "
                                ".mlir to read cold, inside this repository "
                                "or not")
        child.add_argument("--func", default=None,
                           help="function to probe, when a bare .mlir declares "
                                "more than one")
        child.add_argument("--all", action="store_true",
                           help="every kernel declared in examples/entries.py")
        child.add_argument("-v", "--verbose", action="store_true",
                           help="show closed claims too")
        child.add_argument("--write-report", action="store_true",
                           help="write docs/kernel_support.md (probe --all only)")
        child.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    return args.handler(args)
