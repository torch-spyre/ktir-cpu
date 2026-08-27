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

"""The gate: one test per claim, over every kernel that declares an entry.

This module contains no checks of its own.  It calls
``ktir_cpu.kernelentry.ledger.probe`` — the same function the CLI calls — and
turns each claim into a test result.  A check implemented here as well as there
would be a second opinion, and the two would eventually disagree about whether a
kernel is supported.

A ``deferred`` claim becomes ``xfail(strict=True)``, following
``tests/test_spec_gaps.py``, so a deferral reads as a known gap rather than a
failure.  What makes the deferral *expire* is not that marker, though: the marker
is applied from the claim's current state, so once the gap closes the claim is no
longer deferred, no marker is applied, and this test simply passes.  The engine
raises ``deferred.unnecessary.<id>`` for that case, which is what actually fails
the build until the declaration is updated.

Cost: each entry is probed once, at ``gate_params``, and the result is cached for
every claim derived from it — without that cache the full-size kernels would be
re-executed once per claim, and ``layernorm_fwd_ktir`` alone raises 30.  Gate cost
is driven by shape rather than by how many kernels are declared, which is the
trade ``gate_params`` makes; ``docs/kernelentry.md`` carries the measurement.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pytest

from ktir_cpu.dialects import registry
from ktir_cpu.kernelentry import (
    BLOCKING, CLOSED, DEFERRED, REPO_ROOT, KernelEntry, registered,
)
from ktir_cpu.kernelentry.cli import discover_all, render_report
from ktir_cpu.kernelentry.ledger import Ledger, probe
from ktir_cpu.kernelentry.pricing import (
    BOTH, PRICED, UNJUDGED_ZERO_OPS, UNLISTED, UNREGISTERED, ZERO_COST_OPS, audit,
)

discover_all()
ENTRIES: Dict[str, KernelEntry] = registered()

_CACHE: Dict[str, Ledger] = {}


def ledger_for(name: str) -> Ledger:
    if name not in _CACHE:
        _CACHE[name] = probe(ENTRIES[name])
    return _CACHE[name]


def _claim_params() -> List[pytest.param]:
    """One parameter per (kernel, claim), with deferred claims marked xfail.

    The ledger has to be computed to know the claim set, which is the point — the
    set is derived from each kernel rather than listed here, so a kernel with more
    ops contributes more tests without this file changing.
    """
    params = []
    for name in sorted(ENTRIES):
        for claim in ledger_for(name).claims:
            marks = []
            if claim.state == DEFERRED:
                marks.append(pytest.mark.xfail(
                    strict=True, reason=claim.detail or "deferred"))
            params.append(pytest.param(name, claim.id,
                                       marks=marks, id=f"{name}-{claim.id}"))
    return params


def test_at_least_one_entry_is_declared():
    """A gate over an empty set passes, and would do so silently forever."""
    assert ENTRIES, (
        "no kernel entries found under examples/ — the gate would pass "
        "vacuously. See docs/kernelentry.md."
    )


@pytest.mark.parametrize("name,claim_id", _claim_params())
def test_claim(name: str, claim_id: str):
    """One claim about one kernel.

    ``undetermined`` fails alongside ``open`` on purpose: a check the engine could
    not evaluate must not read as one that passed, which is exactly what happens
    when such a claim is quietly omitted instead.
    """
    claim = next(c for c in ledger_for(name).claims if c.id == claim_id)
    assert claim.state not in BLOCKING, (
        f"{claim.state}: {claim.id}\n  {claim.detail}\n"
        f"  fix in: {claim.closer or 'unknown'}"
    )
    if claim.state == DEFERRED:
        # Reached only while the deferral holds; the xfail mark above turns a pass
        # here into the failure that forces the declaration to be updated.
        pytest.fail(f"deferred: {claim.detail}")


class TestClaimsDetectViolations:
    """A gate nobody has watched fail is a gate nobody knows the state of.

    ``test_claim`` above asserts that every claim is closed, which passes just as
    well for an engine that closes everything. These are the other direction: each
    one breaks something a claim is supposed to notice, and asserts it opens.

    Each case mutates a *copy* of the declaration, so the entry the rest of the
    suite reads is untouched.
    """

    @staticmethod
    def _copy(name: str, **overrides) -> KernelEntry:
        import dataclasses

        return dataclasses.replace(ENTRIES[name], **overrides)

    @staticmethod
    def _state(entry: KernelEntry, claim_id: str) -> str:
        return next(c.state for c in probe(entry).claims if c.id == claim_id)

    def test_a_missing_reference_opens_the_reference_claim(self):
        """Absent, not skipped: a kernel writing plausible nonsense reports cleanly."""
        entry = self._copy("matmul_small", reference=None)
        assert self._state(entry, "out.c_ptr.reference") == "open"

    def test_a_narrow_reference_opens_the_reference_claim(self):
        """A reference at the kernel's own precision reproduces its own overflow."""
        import numpy as np

        def narrow(params, tensors):
            a = np.asarray(tensors["a_ptr"], dtype=np.float32)
            b = np.asarray(tensors["b_ptr"], dtype=np.float32)
            return {"c_ptr": (a @ b).astype(np.float16)}

        entry = self._copy("matmul_small", reference=narrow)
        assert self._state(entry, "out.c_ptr.reference") == "open"

    def test_a_reference_for_an_unwritten_tensor_is_undetermined(self):
        """The claim set is derived from the store trace, not from the reference.

        So a reference entry the trace never names is the direction that raises no
        claim at all rather than a failing one, and the support report carries no
        per-kernel claim count in which a vanished claim would show up. The defect
        it hides is the one this ledger exists to catch: a kernel that stops writing
        a declared output loses that output's comparison instead of failing it.
        """
        import numpy as np

        def with_ghost(params, tensors):
            got = ENTRIES["matmul_small"].reference(params=params, tensors=tensors)
            return {**got, "ghost_ptr": np.zeros(4, dtype=np.float32)}

        entry = self._copy("matmul_small", reference=with_ghost)
        assert self._state(entry, "out.ghost_ptr.reference") == "undetermined"
        assert not probe(entry).clean

    def test_dropping_a_declared_tolerance_opens_the_reference_claim(self):
        """The one widened tolerance in the repository has to be load-bearing.

        A tolerance is the one input to this claim that can be adjusted until it
        passes, so the entry that declares its own has to be the entry that needs
        it. If this test starts failing, the override is decoration and the row
        should lose it rather than keep it.
        """
        entry = self._copy("ffn_swiglu", tolerance={})
        assert self._state(entry, "out.out_ptr.reference") == "open"

    def test_a_declared_tolerance_is_stated_by_the_claim_that_passes(self):
        """A green run has to say which pair it was green against."""
        claim = next(c for c in probe(ENTRIES["ffn_swiglu"]).claims
                     if c.id == "out.out_ptr.reference")
        assert claim.state == "closed"
        assert "declared rtol=" in claim.detail

    def test_a_waiver_for_a_claim_the_kernel_never_raises_is_reported(self):
        """A stale waiver reads as a decision about a check that no longer runs."""
        entry = self._copy(
            "matmul_small",
            waived={"op.linalg.batch_matmul.handler": "this kernel has no batched matmul"},
        )
        states = {c.id: c.state for c in probe(entry).claims}
        assert "waived.stale.op.linalg.batch_matmul.handler" in states
        assert states["waived.stale.op.linalg.batch_matmul.handler"] == "open"

    def test_an_excuse_with_no_reason_is_rejected_at_declaration(self):
        with pytest.raises(ValueError, match="needs a reason"):
            self._copy("matmul_small", waived={"parse.regex": "   "})

    def test_a_deferral_naming_no_issue_is_rejected_at_declaration(self):
        """The issue is the whole of what a deferral promises.

        Without it the report cannot group the gap, nobody comes back for it, and
        what was declared as temporary is a waiver written in the column that is
        supposed to expire.
        """
        with pytest.raises(ValueError, match="does not name an issue"):
            self._copy("matmul_small",
                       deferred={"parse.regex": "will look at this later"})

    def test_a_tolerance_for_an_argument_the_kernel_does_not_have_is_rejected(self):
        """The ledger reads this mapping with ``.get()``, so a typo is invisible.

        An entry that is never read looks exactly like a widening that took
        effect, and the claim it was meant to loosen passes or fails for reasons
        the reviewer is no longer looking at.
        """
        with pytest.raises(ValueError, match="names no declared tensor"):
            self._copy("ffn_swiglu", tolerance={"out_ptrr": (2e-2, 2.0)})

    def test_a_malformed_tolerance_is_rejected_at_declaration(self):
        """Not an (rtol, atol) pair, caught where the row is read rather than deep
        in a comparison whose message would be about the kernel."""
        with pytest.raises(ValueError, match="pair of numbers"):
            self._copy("ffn_swiglu", tolerance={"out_ptr": 2e-2})
        with pytest.raises(ValueError, match="is negative"):
            self._copy("ffn_swiglu", tolerance={"out_ptr": (-1e-2, 2.0)})

    def test_a_waiver_for_a_claim_that_now_passes_is_reported(self):
        """An excuse outliving what made it necessary is misinformation.

        The stale-waiver check above only catches a waiver naming a check the
        kernel never raises. This is the other case: the check runs and passes,
        and the waiver would otherwise keep reporting it as excused.
        """
        entry = self._copy(
            "matmul_small",
            waived={"parse.regex": "this kernel is known not to parse"},
        )
        states = {c.id: c.state for c in probe(entry).claims}
        assert states["parse.regex"] == "closed", (
            "a passing check must report as closed, not as waived"
        )
        assert states["waived.unnecessary.parse.regex"] == "open"

    def test_a_deferral_for_a_claim_that_now_passes_is_reported(self):
        """A deferral has to expire on its own, and xfail alone does not do it.

        The xfail(strict=True) marker in the gate is applied from the claim's
        *current* state, so once the gap closes the claim is no longer deferred, no
        marker is applied, and the test simply passes — leaving the deferral in the
        declaration with nothing pointing at it. The expiry has to be a claim.
        """
        entry = self._copy(
            "matmul_small",
            deferred={"parse.regex": "#1 — assumed not to be readable yet"},
        )
        states = {c.id: c.state for c in probe(entry).claims}
        assert states["parse.regex"] == "closed", (
            "a passing check must report as closed, not as deferred"
        )
        assert states["deferred.unnecessary.parse.regex"] == "open"


class TestFrontendClaimWithoutBindings:
    """``parse.frontend``'s three outcomes, checked without the MLIR bindings.

    The claim itself can only be *decided* where ``mlir_ktdp`` is installed, which
    on most machines is CI only. Its three-way branch is this repository's code
    though, and getting it wrong is expensive in a specific way: treating a missing
    local dependency as a rejection would open the claim on every kernel at once,
    and treating a rejection as a skip would hide the one thing this claim exists
    to catch. So the branch is exercised here against stand-ins.
    """

    ENTRY = "matmul_small"

    def _state_with(self, monkeypatch, stub) -> str:
        import ktir_cpu.mlir_frontend.parser as frontend

        monkeypatch.setattr(frontend, "MLIRFrontendParser", stub)
        ledger = probe(ENTRIES[self.ENTRY])
        return next(c.state for c in ledger.claims if c.id == "parse.frontend")

    def test_a_parser_that_accepts_closes_the_claim(self, monkeypatch):
        class Accepts:
            def parse_module(self, text):
                return object()

        assert self._state_with(monkeypatch, Accepts) == "closed"

    def test_a_parser_that_rejects_opens_the_claim(self, monkeypatch):
        class Rejects:
            def parse_module(self, text):
                raise ValueError("op 'ktdp.load' verification failed")

        assert self._state_with(monkeypatch, Rejects) == "open"

    def test_absent_bindings_skip_rather_than_open_the_claim(self, monkeypatch):
        """The constructor raises ImportError, not the import — hence the guard."""
        class NoBindings:
            def __init__(self):
                raise ImportError(
                    "mlir_ktdp not installed; "
                    "MLIRFrontendParser is unavailable."
                )

        assert self._state_with(monkeypatch, NoBindings) == "skip"


class TestProbingAKernelWithNoDeclaration:
    """``probe`` on a bare ``.mlir``, which is the first thing anyone does.

    Requiring a declaration before the tool will say anything inverts the useful
    order: the questions worth asking first — which ops have no handler, does it
    survive both parse paths — do not depend on one.
    """

    KERNEL = "triton-ktir/layernorm_fwd_ktir.mlir"

    def _ledger(self):
        from ktir_cpu.kernelentry import EXAMPLES_DIR
        from ktir_cpu.kernelentry.cli import entry_for_bare_kernel

        return probe(entry_for_bare_kernel(EXAMPLES_DIR / self.KERNEL))

    def test_the_op_and_parse_claims_are_answered(self):
        states = {c.id: c.state for c in self._ledger().claims}
        handlers = [v for k, v in states.items() if k.endswith(".handler")]
        assert handlers and all(v == CLOSED for v in handlers)
        assert states["parse.regex"] == CLOSED

    def test_claims_needing_a_declaration_say_so(self):
        claims = {c.id: c for c in self._ledger().claims}
        assert claims["exec.runs"].state == "open"
        assert "tensors=" in claims["exec.runs"].detail
        assert claims["cost.derivation"].state == "undetermined"

    def test_a_kernel_outside_this_repository_is_probed_where_it_lies(self, tmp_path):
        """The case the tool is for: a kernel somebody brings, versioned elsewhere.

        The entry keeps the absolute path, there being no relative form for it to
        have, and every claim that needs no declaration is answered anyway.
        """
        from ktir_cpu.kernelentry import EXAMPLES_DIR
        from ktir_cpu.kernelentry.cli import entry_for_bare_kernel

        outside = tmp_path / "brought_from_elsewhere.mlir"
        outside.write_text((EXAMPLES_DIR / self.KERNEL).read_text())

        entry = entry_for_bare_kernel(outside)
        assert entry.mlir_path.resolve() == outside.resolve()

        states = {c.id: c.state for c in probe(entry).claims}
        handlers = [v for k, v in states.items() if k.endswith(".handler")]
        assert handlers and all(v == CLOSED for v in handlers)
        assert states["parse.regex"] == CLOSED

    def test_a_multi_function_module_asks_which_one(self, tmp_path):
        """Guessing would probe an arbitrary function and report on the wrong kernel.

        Written on a file this test makes rather than one under ``examples/``: every
        kernel there holds exactly one function today, so a test pointed at one of
        them would pass without the branch ever running.
        """
        from ktir_cpu.kernelentry.cli import entry_for_bare_kernel

        multi = tmp_path / "two_functions.mlir"
        multi.write_text("module {\n"
                         "  func.func @first() { return }\n"
                         "  func.func @second() { return }\n"
                         "}\n")
        with pytest.raises(SystemExit, match="--func"):
            entry_for_bare_kernel(multi)

    def test_a_module_with_no_function_says_that_instead(self, tmp_path):
        """Not the same message: there is nothing to name with ``--func``.

        The regex parser has a catch-all fallback and validates against no dialect,
        so text that is not KTIR at all reaches this point *accepted*, with no
        function in it. Sending somebody to name one sends them looking for
        something that is not there.
        """
        from ktir_cpu.kernelentry.cli import entry_for_bare_kernel

        empty = tmp_path / "not_ktir.mlir"
        empty.write_text("module {\n  this is not KTIR at all\n}\n")
        with pytest.raises(SystemExit, match="declares no function"):
            entry_for_bare_kernel(empty)

    def test_a_file_the_parser_rejects_is_a_cell_not_a_crash(self, tmp_path,
                                                             monkeypatch):
        """One unreadable file must not take the report for the other thirty with it.

        And the cell it gets is the answer, not ``yes``: an earlier version caught
        every failure to read out a single kernel as "several functions" and printed
        ``yes`` in the column that says the parser read the file — a claim nothing
        had checked, in the one document whose job is to not do that.
        """
        from ktir_cpu.kernelentry import cli

        unreadable = tmp_path / "undecodable.mlir"
        unreadable.write_bytes(b"module {\n  \xff\xfe not text\n}\n")
        monkeypatch.setattr(cli, "undeclared_kernels",
                            lambda: [str(unreadable)])

        assert cli.function_names(unreadable) is None
        sv = cli.survey([])
        assert sv.undeclared == [(str(unreadable), "?", ["**no**", "yes"])]
        # And the headline counts it out rather than saying "All 1": the sentence a
        # reader takes the repository's state from cannot read as complete while a
        # row below it says otherwise.
        assert "0 of 1 are read by the regex parser" in cli.render_report([], sv)


def test_the_generated_documents_do_not_depend_on_this_machine():
    """Both committed documents must be functions of the repository, not the host.

    It is compared verbatim, and `parse.frontend` reads `closed` where the MLIR
    bindings are installed and `skip` where they are not — so a report that showed
    what the machine writing it happened to see would differ between CI and a
    contributor's laptop, and each would call the other's stale, forever. Two things keep it
    host-independent: the frontend column is read from the committed record in
    `conformance.py` and from a declaration's own excuse, and claims still marked
    environment-dependent are left out of the lists below the table.

    `docs/supported_ops.md` is host-independent for a different reason — both its
    registries are decorator tables in this package rather than anything the
    bindings provide — and it is checked here so that a later column reading a
    real parser fails rather than committing what one machine could see.
    """
    import ktir_cpu.mlir_frontend.parser as frontend

    from ktir_cpu.kernelentry.cli import generated_docs

    def rendered(ledgers):
        return {p.name: t for p, t in generated_docs(ledgers).items()}

    # The first pass reuses the session cache; the second cannot, because the
    # stub has to be in place while the kernels are probed. So this test costs one
    # extra full probe of every declared kernel — most of this module's runtime,
    # and the reason it is one test rather than one per document.
    as_is = rendered([ledger_for(n) for n in sorted(ENTRIES)])

    class NoBindings:
        def __init__(self):
            raise ImportError(
                "mlir_ktdp not installed; "
                "MLIRFrontendParser is unavailable."
            )

    original = frontend.MLIRFrontendParser
    try:
        frontend.MLIRFrontendParser = NoBindings
        without = rendered([probe(ENTRIES[n]) for n in sorted(ENTRIES)])
    finally:
        frontend.MLIRFrontendParser = original

    differ = sorted(n for n in as_is if as_is[n] != without[n])
    assert not differ, (
        f"{', '.join(differ)} differs between a machine with the MLIR bindings "
        "and one without: it is rendering a state it observed rather than one "
        "that is written down"
    )


@pytest.mark.parametrize("name", ["kernel_support.md", "supported_ops.md"])
def test_generated_document_is_current(name: str):
    """Each generated document must match what the engine reports now.

    Same discipline as a lock file: the documents are generated, and this is what
    keeps generation from being optional. Regenerate with::

        python -m ktir_cpu.kernelentry probe --all --write-report

    Parametrized over the names rather than over ``generated_docs`` itself so that
    a document dropped from the writer fails here instead of quietly reducing this
    test to the ones that remain.
    """
    from ktir_cpu.kernelentry.cli import generated_docs

    ledgers = [ledger_for(name_) for name_ in sorted(ENTRIES)]
    docs = {path.name: text for path, text in generated_docs(ledgers).items()}
    assert name in docs, f"{name} is no longer generated by generated_docs()"

    path = REPO_ROOT / "docs" / name
    assert path.exists(), (
        f"{name} is missing — run "
        "`python -m ktir_cpu.kernelentry probe --all --write-report`"
    )
    assert path.read_text() == docs[name], (
        f"{name} is stale — regenerate with "
        "`python -m ktir_cpu.kernelentry probe --all --write-report`"
    )


def test_support_report_covers_conftest_examples():
    """Every kernel in the report is already driven from ``EXAMPLE_PARAMS``.

    The report says so in prose, and the sentence is load-bearing: it is what
    separates "this ledger has not asked" from "the simulator cannot". A reader
    who does not believe it has to go count 31 files by hand, and the claim rots
    silently the moment somebody adds an example without arguments — so assert it
    here instead. Failing means the report's opening paragraph has become wrong
    and the new file needs either an ``EXAMPLE_PARAMS`` entry or a different
    sentence.

    The report cannot check this itself, and neither can it count how many of
    those listings are empty: ``ktir_cpu`` ships as a package and
    ``tests/conftest.py`` is not part of it. A test is the only place both sides
    are importable, so the figure the report prints is pinned here too.
    """
    from conftest import EXAMPLE_PARAMS

    from ktir_cpu.kernelentry.cli import (EMPTY_EXECUTE_KWARGS_ROWS,
                                          undeclared_kernels)

    kwargs_by_path: Dict[str, List[dict]] = {}
    for entries in EXAMPLE_PARAMS.values():
        for entry in entries:
            kwargs_by_path.setdefault(
                f"examples/{entry['path']}", []).append(entry["execute_kwargs"])
    undeclared = set(undeclared_kernels())
    rows = undeclared | {
        str(entry.mlir_path.resolve().relative_to(REPO_ROOT.resolve()))
        for entry in registered().values()
    }
    missing = sorted(rows - set(kwargs_by_path))
    assert not missing, (
        "docs/kernel_support.md says tests/ already drives every file in its "
        "table, from tests/conftest.py::EXAMPLE_PARAMS, and these are not: "
        f"{missing}"
    )

    # A path listed twice with different arguments would make "empty" ambiguous,
    # so require it of every listing for that path rather than of one of them.
    empty = sorted(p for p in undeclared
                   if all(not kw for kw in kwargs_by_path[p]))
    assert len(empty) == EMPTY_EXECUTE_KWARGS_ROWS, (
        "docs/kernel_support.md says execute_kwargs is empty for "
        f"{EMPTY_EXECUTE_KWARGS_ROWS} of the files it lists as read-not-declared, "
        f"but {len(empty)} are: {empty}. Update "
        "ktir_cpu.kernelentry.cli.EMPTY_EXECUTE_KWARGS_ROWS."
    )

    # The same sentence's other half — what the non-empty listings carry — would
    # otherwise be prose nothing checks.  An index or a size is a scalar; a listing
    # supplying an array would be the declaration the sentence says it is not.
    nonscalar = sorted(
        (path, key) for path in undeclared for kw in kwargs_by_path[path]
        for key, value in kw.items() if not isinstance(value, (int, float))
    )
    assert not nonscalar, (
        "docs/kernel_support.md says the non-empty execute_kwargs listings carry raw "
        f"HBM element indices or a scalar size, and these carry neither: {nonscalar}"
    )


class TestAnExcusedEnvironmentDependentClaim:
    """A declared excuse on ``parse.frontend`` must be reported, not filtered out.

    A claim whose outcome depends on the machine is left out of the report's lists,
    because a document that named it would say something different depending on who
    generated it. An excuse in the declaration removes that dependence: it applies
    to every state but ``closed``, and a missing dependency is one of those, so both
    machines report the same thing. Keeping such a claim out of the report anyway
    would hide a known, written-down gap behind "your machine could not tell" —
    the one outcome this report is supposed to make impossible.

    Both cases stub the parser rather than reading whichever answer this machine
    happens to give, so they assert the same thing in CI and on a laptop.
    """

    @staticmethod
    def _probe(monkeypatch, stub, **overrides):
        import dataclasses

        import ktir_cpu.mlir_frontend.parser as frontend

        monkeypatch.setattr(frontend, "MLIRFrontendParser", stub)
        return probe(dataclasses.replace(ENTRIES["matmul_small"], **overrides))

    class _NoBindings:
        def __init__(self):
            raise ImportError(
                "mlir_ktdp not installed; "
                "MLIRFrontendParser is unavailable."
            )

    class _Accepts:
        def parse_module(self, text):
            return object()

    def test_the_deferral_survives_into_the_committed_report(self, monkeypatch):
        """Without the bindings the claim is ``skip``, which the deferral covers."""
        ledger = self._probe(
            monkeypatch, self._NoBindings,
            deferred={"parse.frontend": "#1 — known not to parse there"})
        claim = next(c for c in ledger.claims if c.id == "parse.frontend")
        assert claim.state == DEFERRED
        assert not claim.env_dependent, (
            "an applied excuse pins the outcome on every machine, so the claim is "
            "no longer environment-dependent"
        )
        assert "#1 — known not to parse there" in render_report([ledger])

    def test_an_unnecessary_excuse_for_it_stays_out_of_the_report(self, monkeypatch):
        """The other direction: only a machine with the bindings sees it is stale.

        Where the bindings are absent the claim is ``skip``, the excuse applies, and
        nothing is unnecessary. Where they are present the check may pass and the
        excuse become stale — so the expiry claim's own outcome depends on the
        machine, and leaving it out of the document is what keeps the document
        host-independent. The gate reads raw states, so CI still fails on it.
        """
        ledger = self._probe(
            monkeypatch, self._Accepts,
            waived={"parse.frontend": "the MLIR frontend cannot see this kernel"})
        claims = {c.id: c for c in ledger.claims}
        assert claims["parse.frontend"].state == CLOSED
        expiry = claims["waived.unnecessary.parse.frontend"]
        assert expiry.state == "open"
        assert expiry.env_dependent
        assert "waived.unnecessary" not in render_report([ledger])


class TestArgumentSpecs:
    """The properties the rest of the ledger reads the declared arguments through.

    ``tensors`` is a table rather than a callable, and two claims depend on what
    that table guarantees: the reference is handed a *rebuild* rather than what the
    run was given, and the committed derivation rebuilds again to recover shapes.
    Neither is sound unless a rebuild is the same tensors.
    """

    def test_rebuilding_an_entry_s_arguments_gives_the_same_values(self):
        """Why the reference can be handed a pristine copy at all.

        A kernel whose output argument aliases its input has overwritten that array
        by the time the reference runs; handing the reference a rebuild is what
        keeps it from comparing the result against itself. That substitution is only
        valid if the rebuild is bit-identical to what the kernel was given.
        """
        import numpy as np

        for name, entry in sorted(ENTRIES.items()):
            first, second = entry.build_tensors(), entry.build_tensors()
            assert sorted(first) == sorted(second), name
            for arg in first:
                np.testing.assert_array_equal(
                    np.asarray(first[arg]), np.asarray(second[arg]),
                    err_msg=f"{name}.{arg} is not reproducible")

    def test_two_arguments_of_one_kernel_are_drawn_independently(self):
        """Seeded per argument, not per kernel.

        One generator drawn twice gives the second tensor the first one's values
        wherever their shapes overlap — and a kernel that swapped its two operands
        would then still agree with its reference. This is the property that makes
        the swap observable.
        """
        import numpy as np

        tensors = ENTRIES["matmul_small"].build_tensors()
        a = np.asarray(tensors["a_ptr"], dtype=np.float32).ravel()
        b = np.asarray(tensors["b_ptr"], dtype=np.float32).ravel()
        overlap = min(a.size, b.size)
        assert not np.array_equal(a[:overlap], b[:overlap])

    def test_a_spec_naming_a_parameter_that_does_not_exist_is_rejected(self):
        """At declaration, not on the one kernel at the moment it ran.

        A misspelled parameter name would otherwise surface as a ``KeyError`` from
        inside the engine, which reads as a fault in the tool rather than a typo in
        the row.
        """
        from ktir_cpu.kernelentry.tensorspec import param, zeros

        with pytest.raises(ValueError, match="not in gate_params"):
            KernelEntry(name="_typo", func="f", path="ktir/ffn_swiglu.mlir",
                        gate_params={"rows": 1},
                        tensors={"out": zeros("rows"), "n": "colums"})
        with pytest.raises(ValueError, match="not in gate_params"):
            KernelEntry(name="_typo", func="f", path="ktir/ffn_swiglu.mlir",
                        gate_params={"rows": 1},
                        tensors={"n": param("colums", "i32")})


def test_discovering_declarations_twice_is_a_no_op():
    """Two test modules discover at import time, and one session collects both.

    Declarations are loaded by file location rather than by module name, so they
    never enter ``sys.modules`` and nothing stops a second call from executing the
    same file again — at which point ``register_entry`` rejects the duplicate name
    and takes down collection for the entire run, not just this module. That failure
    only appears when both modules are collected together, which is what
    ``uv run pytest tests/`` does and what running either file alone does not.
    """
    before = dict(registered())
    discover_all()
    discover_all()
    assert dict(registered()) == before


class TestPricingAudit:
    """`zero` in the registry is two facts, and this is the test that splits them.

    ``@register()`` defaults ``latency_category`` to ``"zero"``, so registering an
    op without naming a category makes it free and says nothing.  The record in
    ``ktir_cpu/kernelentry/pricing.py`` is what makes that a decision; these tests
    hold the record to the registry in both directions, because the direction that
    rots is not the one the check exists for.

    Repository-wide rather than per kernel, and that came from measurement: asked
    once per kernel, this check produced 66 of a prototype's 74 false positives,
    since the ops it flags are the same structural ones in every kernel.
    """

    def test_every_zero_priced_op_is_free_by_decision_or_recorded_as_unjudged(self):
        findings = audit()
        assert not findings, (
            "the pricing record and the op registry disagree:\n"
            + "\n".join(f"  {finding}" for finding in findings)
        )

    def test_every_reason_is_written(self):
        """An entry with no reason is a waiver, not a decision.

        The same rule ``KernelEntry`` applies to ``waived`` and ``deferred``: an
        excuse with no reason is indistinguishable from an oversight.
        """
        for mapping, which in ((ZERO_COST_OPS, "ZERO_COST_OPS"),
                               (UNJUDGED_ZERO_OPS, "UNJUDGED_ZERO_OPS")):
            for op, reason in mapping.items():
                assert reason.strip(), f"{which}[{op!r}] needs a reason"

    def test_every_unjudged_op_names_the_issue_that_would_settle_it(self):
        """The half of the record that expires has to say what would expire it.

        ``ZERO_COST_OPS`` is a decision and stands on its reason alone.  An
        unjudged op is an open question, and an open question with no issue behind
        it is how the list becomes permanent.
        """
        for op, reason in UNJUDGED_ZERO_OPS.items():
            assert re.search(r"#\d+", reason), (
                f"UNJUDGED_ZERO_OPS[{op!r}] does not name an issue. Without one "
                "the entry is a decision to leave it unpriced, spelled as a "
                "question."
            )

    def test_a_newly_registered_op_is_not_silently_free(self):
        """The state AC7 of #209 exists to reject."""
        with registry.temp_registry():
            registry._REGISTRY["ktdp.invented_op"] = lambda op, ctx, env: None
            registry._LATENCY_CATEGORIES["ktdp.invented_op"] = "zero"
            findings = audit()
        assert [f.op for f in findings] == ["ktdp.invented_op"]
        assert findings[0].kind == UNLISTED
        assert "pricing.py" in findings[0].fix

    def test_an_entry_that_outlived_its_zero_is_reported(self):
        """Deciding an op's price has to take its entry with it.

        This is the direction that rots: #211 will price three of these, and a
        record still calling them unjudged afterwards would describe a repository
        that no longer exists.
        """
        with registry.temp_registry():
            registry._LATENCY_CATEGORIES["linalg.fill"] = "compute_float"
            findings = audit()
        assert [(f.op, f.kind) for f in findings] == [("linalg.fill", PRICED)]

    def test_an_entry_whose_op_is_gone_is_reported(self):
        """A rename or a removal must not leave the record behind."""
        with registry.temp_registry():
            del registry._REGISTRY["ktdp.coreid"]
            findings = audit()
        assert [(f.op, f.kind) for f in findings] == [("ktdp.coreid", UNREGISTERED)]

    def test_claiming_both_answers_for_one_op_is_reported(self, monkeypatch):
        """Free by decision and not yet judged are mutually exclusive."""
        monkeypatch.setitem(UNJUDGED_ZERO_OPS, "scf.for", "undecided #1")
        findings = audit()
        assert [(f.op, f.kind) for f in findings] == [("scf.for", BOTH)]

    def test_the_audit_cannot_see_a_wrong_category(self):
        """Stated as a test so the limitation is not mistaken for coverage.

        The check compares against ``zero``, so an op billed to the wrong non-zero
        unit passes it untouched — an integer compare charged to the float pipe
        was a real defect in this repository, and this audit would not have found
        it.  Only a reader who knows the op's semantics does.
        """
        with registry.temp_registry():
            registry._LATENCY_CATEGORIES["arith.addi"] = "compute_matmul"
            assert audit() == []
