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

"""The one engine that decides a kernel's claims.

``probe``, ``verify`` and ``tests/test_kernelentry.py`` all call :func:`probe`;
none of them re-implements a check.  A second implementation would be a second
opinion, and the two would eventually disagree about whether a kernel is
supported.

Every check here delegates to machinery that already exists — the regex parser,
the MLIR frontend's own ``verify()``, the dialect handler registry, the frontend
adapter table, ``LatencyReport`` — so a claim closing means the library accepted
the kernel, not that this module was satisfied.

The hazard this module is built against is **false positives**.  A ledger that
flags correct kernels gets waived into silence, and then the waiver mapping
carries no information.  A probe-only prototype run over six real artifacts --
the example files and notebook generators of the three kernels most recently
added to this repository -- produced 74 unexpected open claims, of which 66 came
from a single check that fired on every structural op in every kernel; the
remaining rows were the informative ones.  That check is not here: whether an op is priced is a question
about the one cost model every kernel shares, and asking it per kernel made every
kernel answer it.  When adding a claim, the question to answer first is not
"what does this catch" but "what correct kernel does this flag".
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import ktir_cpu.dialects  # noqa: F401 — import triggers @register side effects
from ktir_cpu import KTIRInterpreter
from ktir_cpu.dialects import registry
from ktir_cpu.ir_types import _iter_ops
from ktir_cpu.latency import HardwareConfig

from .derivation import ArgExtents, DOCUMENT, read_sections, render_derivation

from . import (
    CLOSED, COST, DEFERRED, FUNCTION, OPEN, REPO_ROOT, SKIP, UNDETERMINED, WAIVED,
    Claim, KernelEntry,
)

#: Default tolerances for `out.<arg>.reference`.  Wide because every value in a
#: KTIR kernel is f16 and cross-core folds round their running sum at each step;
#: `tests/test_examples.py` uses the same pair for the split-K matmul and the
#: decode SDPA fold.  An entry may widen it for one output through
#: ``KernelEntry.tolerance``, which the claim then reports so the looser pair is
#: not invisible in a green run.
REF_RTOL, REF_ATOL = 2e-2, 2e-1


@dataclass
class Ledger:
    """Every claim about one kernel, plus what the engine learned while probing."""

    entry: KernelEntry
    claims: List[Claim] = field(default_factory=list)
    #: Populated when the kernel executed, for callers that want the figures
    #: rather than the verdicts (``adopt`` writes the derivation from these).
    report: Any = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    output_args: Tuple[str, ...] = ()
    #: arg name -> pointer value from the run, needed to turn the attribution's
    #: element-index keys back into argument names.
    arg_ptrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> List[Claim]:
        return [c for c in self.claims if c.blocking]

    @property
    def clean(self) -> bool:
        return not self.blocking

    def by_leg(self, leg: str) -> List[Claim]:
        return [c for c in self.claims if c.leg == leg]

    def tally(self, leg: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for claim in self.by_leg(leg):
            out[claim.state] = out.get(claim.state, 0) + 1
        return out


class _Builder:
    """Accumulates claims, applying the entry's waivers and deferrals uniformly.

    Every claim goes through :meth:`add`, so ``waived``/``deferred`` cannot be
    honoured for some checks and forgotten for others — and a waiver naming a
    claim the kernel never raises is reported instead of being ignored, since a
    stale waiver reads as a considered decision about a check that no longer runs.
    """

    def __init__(self, entry: KernelEntry) -> None:
        self.entry = entry
        self.claims: List[Claim] = []
        self._seen: set[str] = set()
        # Excuses whose check now passes on its own, waived and deferred alike.
        self._unnecessary: set[str] = set()
        #: Of those, the ones whose check needs an optional dependency, so that
        #: "this excuse is no longer needed" is itself only decidable where that
        #: dependency is installed — which keeps it out of the committed report.
        self._unnecessary_env: set[str] = set()

    def add(self, claim_id: str, leg: str, state: str, detail: str = "",
            closer: str = "", env_dependent: bool = False) -> None:
        self._seen.add(claim_id)
        # One loop over the two excuse mappings, in the order that decides which
        # one wins, so a guard cannot be written for one and forgotten for the
        # other. ``finish()`` walks the same pair.
        for mapping, excused_state in ((self.entry.waived, WAIVED),
                                       (self.entry.deferred, DEFERRED)):
            if claim_id not in mapping:
                continue
            if state == CLOSED:
                # Excusing a check that passes is misinformation, not caution:
                # the excuse outlives whatever made it necessary and nothing
                # says so. ``finish()`` turns this into a claim of its own.
                self._unnecessary.add(claim_id)
                if env_dependent:
                    self._unnecessary_env.add(claim_id)
                break
            # The author's reason replaces the engine's: an excuse says why this
            # is knowingly not met and, for a deferral, where it is tracked —
            # strictly more informative than repeating what the check looked for.
            state, detail = excused_state, mapping[claim_id]
            # The declaration decided this one, and it decided it the same way on
            # every machine: the excuse applies to any state but CLOSED, and a
            # missing optional dependency is one of those. So the outcome stops
            # being a fact about the host, and leaving it out of the committed
            # report would hide a failure the author already wrote down.
            env_dependent = False
            break
        self.claims.append(
            Claim(claim_id, leg, state, detail, closer, env_dependent))

    def finish(self) -> List[Claim]:
        for mapping, what in ((self.entry.waived, "waived"),
                              (self.entry.deferred, "deferred")):
            for claim_id in mapping:
                if claim_id not in self._seen:
                    self.claims.append(Claim(
                        f"{what}.stale.{claim_id}", FUNCTION, OPEN,
                        f"{what} names {claim_id!r}, which this kernel does not "
                        "raise — remove it or fix the id",
                        closer=f"the entry's {what}= mapping",
                    ))
        for claim_id in sorted(self._unnecessary):
            # A deferral is meant to expire on its own once the gap closes. It
            # cannot do that through xfail(strict=True) alone: the marker is
            # applied from the claim's *current* state, so a closed claim gets no
            # marker and the test simply passes. The expiry has to be a claim.
            what = "waived" if claim_id in self.entry.waived else "deferred"
            self.claims.append(Claim(
                f"{what}.unnecessary.{claim_id}", FUNCTION, OPEN,
                f"{what} names {claim_id!r}, which now passes on its own — "
                f"remove it from {what}=",
                closer=f"the entry's {what}= mapping",
                env_dependent=claim_id in self._unnecessary_env,
            ))
        return self.claims


# ---------------------------------------------------------------------------
# Derivations from the IR and from a run
# ---------------------------------------------------------------------------

def distinct_ops(func) -> List[str]:
    """Every op type appearing in *func*, regions included, sorted."""
    return sorted({op.op_type for op in _iter_ops(func.operations)})


def store_targets(report) -> Tuple[Dict[Any, int], int]:
    """Bytes written by store ops, keyed by the memory they addressed.

    Returns ``(targets, unattributed)``.  Reading the trace rather than walking
    SSA names is what makes this work for a distributed memory view: the walk
    from a store back through ``construct_access_tile`` to a pointer argument
    does not survive ``construct_distributed_memory_view``, and when it fails it
    fails silently — the prototype produced a kernel with no ``out.*`` claims at
    all and a summary line reading ``60/61 closed``.  The trace carries the
    resolved memory instead, so the failure mode becomes a nonzero
    *unattributed* count that the caller must report.

    "Wrote" means wrote to HBM, so the filter names the ``memory`` category
    explicitly rather than inheriting it.  A store whose view is entirely in LX
    is already excluded — ``_estimate`` charges it zero bytes because the tile
    never leaves the chip — but that is a coincidence between two files, and this
    one is about what the kernel produced.  An op priced under any other category
    is not output traffic no matter what it is called.
    """
    targets: Dict[Any, int] = {}
    unattributed = 0
    for counters in report.counters.values():
        for tentry in counters.trace or ():
            # Substring, because "store" is how the one storing op is spelled
            # today (``ktdp.store``) and a dialect may add a qualified spelling.
            if tentry.category != "memory" or "store" not in tentry.op_type:
                continue
            if not tentry.nbytes:
                continue
            if tentry.target is None:
                unattributed += tentry.nbytes
            else:
                targets[tentry.target] = targets.get(tentry.target, 0) + tentry.nbytes
    return targets, unattributed


def fold_to_args(targets: Iterable[Any], arg_ptrs: Dict[str, Any],
                 tensors: Dict[str, Any]) -> Tuple[set, set]:
    """Attribute view origins to the arguments whose element extent contains them.

    The mapping itself is ``derivation.ArgExtents``, shared with the cost
    derivation: both answer "which argument is this origin" and two copies of
    that answer would eventually disagree about what a kernel wrote to.  Returns
    ``(arg_names, unresolved_targets)`` — the second is never dropped, because a
    target that matched no argument means the ledger does not know what the
    kernel wrote to.
    """
    extents = ArgExtents(arg_ptrs, tensors)
    args, unresolved = set(), set()
    for target in targets:
        name = extents.of(target)
        if name is None:
            unresolved.add(target)
        else:
            args.add(name)
    return args, unresolved


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def probe(entry: KernelEntry, *, params: Optional[Dict[str, Any]] = None,
          hardware: Optional[HardwareConfig] = None) -> Ledger:
    """Compute *entry*'s ledger against the simulator as it stands right now.

    Read-only: nothing is written, and the kernel is executed only in the
    simulator's own memory model.  *params* defaults to the entry's
    ``gate_params``.
    """
    params = dict(entry.gate_params if params is None else params)
    hardware = hardware or HardwareConfig()
    build = _Builder(entry)
    ledger = Ledger(entry=entry)

    text = _mlir_text(build, entry)
    if text is None:
        ledger.claims = build.finish()
        return ledger

    interp = _claim_parse(build, entry, text, hardware)
    if interp is None:
        ledger.claims = build.finish()
        return ledger

    func = interp.module.get_function(entry.func)
    _claim_ops(build, func)
    _claim_frontend_parse(build, text)
    _claim_execution(build, ledger, entry, interp, params)
    _claim_cost(build, ledger, entry, params)

    ledger.claims = build.finish()
    return ledger


def _mlir_text(build: _Builder, entry: KernelEntry) -> Optional[str]:
    try:
        return entry.mlir_text()
    except Exception as exc:
        build.add("parse.regex", FUNCTION, OPEN,
                  f"could not obtain MLIR: {type(exc).__name__}: {exc}",
                  closer=str(entry.mlir_path))
        return None


def _claim_parse(build: _Builder, entry: KernelEntry, text: str,
                 hardware: HardwareConfig) -> Optional[KTIRInterpreter]:
    interp = KTIRInterpreter(latency_config=hardware, trace_latency=True)
    try:
        interp.load(text)
        interp.module.get_function(entry.func)
    except Exception as exc:
        build.add("parse.regex", FUNCTION, OPEN,
                  f"{type(exc).__name__}: {exc}",
                  closer="the kernel, or ktir_cpu/parser.py")
        return None
    build.add("parse.regex", FUNCTION, CLOSED)
    return interp


def _claim_ops(build: _Builder, func) -> None:
    """One handler claim per distinct op.

    Deliberately *not* also a per-op frontend-reachability claim.
    ``tests/mlir_frontend/test_registry_consistency.py`` already asserts that
    every executor op is either frontend-installed or in that file's
    ``FRONTEND_UNSUPPORTED`` allow-list — and an op appearing in a kernel is
    necessarily registered, or the claim below would open.  So an unreachable op
    already breaks the build repository-wide, and ``parse.frontend`` catches it
    for this kernel specifically.  A third check would only restate them, at the
    cost of reading a list that lives in a test module.
    """
    for op in distinct_ops(func):
        handled = registry.dispatch(op) is not None
        build.add(f"op.{op}.handler", FUNCTION, CLOSED if handled else OPEN,
                  "" if handled else "no execution handler is registered",
                  closer="ktir_cpu/dialects/ — add a @register handler")


def _claim_frontend_parse(build: _Builder, text: str) -> None:
    """Does the real MLIR frontend accept the kernel, and does verify() pass?

    Absent ``mlir_ktdp`` this is ``skip``, never ``closed``: the frontend has no
    catch-all where the regex parser has one, so a kernel that only works on the
    tolerant path is exactly what this claim exists to catch, and reporting it as
    closed locally would hide that until CI.
    """
    try:
        # MLIRFrontendParser raises ImportError from its *constructor*, not on
        # import, so the guard has to wrap construction. Getting this wrong makes
        # a missing local dependency look like a kernel the frontend rejected —
        # the loudest possible false positive, on every kernel at once.
        from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser
        parser = MLIRFrontendParser()
    except ImportError as exc:
        build.add("parse.frontend", FUNCTION, SKIP,
                  f"{exc} — this layer is verified by CI only",
                  closer="CI", env_dependent=True)
        return
    try:
        parser.parse_module(text)
    except Exception as exc:
        build.add("parse.frontend", FUNCTION, OPEN,
                  f"{type(exc).__name__}: {str(exc)[:200]}",
                  closer="the kernel, or ktir_cpu/mlir_frontend/parser.py",
                  env_dependent=True)
        return
    build.add("parse.frontend", FUNCTION, CLOSED, env_dependent=True)


def _claim_execution(build: _Builder, ledger: Ledger, entry: KernelEntry,
                     interp: KTIRInterpreter, params: Dict[str, Any]) -> None:
    """exec.runs, then reference / nontrivial for each output it identifies."""
    if not entry.tensors:
        detail = ("no tensors= on the entry, so the kernel cannot be driven")
        build.add("exec.runs", FUNCTION, OPEN, detail,
                  closer="the entry's tensors= mapping")
        return

    try:
        tensors = entry.build_tensors(params)
    except Exception as exc:
        build.add("exec.runs", FUNCTION, OPEN,
                  f"tensors= raised {type(exc).__name__}: {exc}",
                  closer="the entry's tensors= mapping")
        return

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = interp.execute_function(entry.func, **tensors)
    except Exception as exc:
        build.add("exec.runs", FUNCTION, OPEN,
                  f"{type(exc).__name__}: {str(exc)[:200]}",
                  closer="the kernel, or the interpreter")
        return
    build.add("exec.runs", FUNCTION, CLOSED)
    ledger.outputs = outputs or {}
    ledger.report = interp.get_latency_report()
    ledger.arg_ptrs = dict(interp.arg_ptrs)

    targets, unattributed = store_targets(ledger.report)
    out_args, unresolved = fold_to_args(targets, interp.arg_ptrs, tensors)
    ledger.output_args = tuple(sorted(out_args))

    identified = not (unattributed or unresolved) and bool(out_args)
    if unattributed or unresolved:
        build.add("out.identified", FUNCTION, UNDETERMINED,
                  f"{unattributed} bytes of store traffic had no resolvable "
                  f"origin and {len(unresolved)} origin(s) matched no argument, "
                  "so the set of output tensors is not fully known",
                  closer="ktir_cpu/latency.py::LatencyReport.traffic_by_target")
    elif not out_args:
        build.add("out.identified", FUNCTION, UNDETERMINED,
                  "no store traffic was recorded, so no output tensor could be "
                  "identified — a kernel that writes nothing is either wrong or "
                  "not driven by the arguments tensors= supplies",
                  closer="the kernel, or the entry's tensors= mapping")
    else:
        build.add("out.identified", FUNCTION, CLOSED,
                  f"outputs: {', '.join(ledger.output_args)}")

    expected: Optional[Dict[str, Any]] = None
    reference_error: Optional[str] = None
    if entry.reference is not None:
        try:
            # A pristine rebuild, not the mapping the run was given: a kernel
            # whose output argument aliases its input has overwritten that array
            # by now, and a reference reading it would compare the result with
            # itself and agree.  The specs are deterministic, so the two builds
            # are the same arrays until the kernel touches one of them.
            expected = entry.reference(params=params,
                                       tensors=entry.build_tensors(params))
        except Exception as exc:
            # Reported through each output's own claim rather than as a claim of
            # its own. A claim that exists only when something fails is absent
            # from the ledger the rest of the time, so it never appears in the
            # support report and a waiver naming it reads as stale.
            reference_error = f"reference= raised {type(exc).__name__}: {exc}"

    for name in ledger.output_args:
        _claim_nontrivial(build, name, outputs.get(name))
        _claim_reference(build, name, outputs.get(name), expected, reference_error)
    if identified and expected is not None:
        _claim_unwritten_references(build, ledger.output_args, expected)


def _claim_nontrivial(build: _Builder, name: str, value: Any) -> None:
    """An all-zero or all-NaN output, which no cost report would ever flag.

    The decode-SDPA notebook helper already computes an fp32 reference and counts
    zero rows on every run, and its docstring gives the reason: every value in
    that kernel is f16 including the cross-core fold's running sum, so an input
    scale large enough to overflow shows up here and nowhere else in the report.
    This generalises that check.  A legitimately all-zero output (a fully masked
    kernel) waives the claim with that as the reason.
    """
    claim_id = f"out.{name}.nontrivial"
    if value is None:
        # The store trace named this argument as written, and the run did not
        # return it. Reporting the dtype of a None instead — which is what
        # asarray produces — would name the symptom and hide the cause.
        build.add(claim_id, FUNCTION, UNDETERMINED,
                  f"the kernel wrote to {name!r} but the run returned no such "
                  "output, so there is nothing to inspect",
                  closer="the entry's tensors= mapping")
        return
    got = np.asarray(value)
    if got.dtype.kind not in "fc":
        build.add(claim_id, FUNCTION, WAIVED,
                  f"dtype {got.dtype} is not floating point")
        return
    as_f32 = got.astype(np.float32)
    if not np.any(got):
        build.add(claim_id, FUNCTION, OPEN, "output is entirely zero",
                  closer="the kernel, or a waiver if zero is correct here")
        return
    if np.any(np.isnan(as_f32)):
        build.add(claim_id, FUNCTION, OPEN, "output contains NaN",
                  closer="the kernel")
        return
    flat = as_f32.reshape(-1, as_f32.shape[-1]) if as_f32.ndim else as_f32
    zero_rows = int(np.sum(~np.any(flat, axis=1))) if as_f32.ndim else 0
    if zero_rows:
        build.add(claim_id, FUNCTION, OPEN,
                  f"{zero_rows}/{flat.shape[0]} rows are entirely zero",
                  closer="the kernel, or a waiver if that is correct here")
        return
    build.add(claim_id, FUNCTION, CLOSED)


def _claim_reference(build: _Builder, name: str, value: Any,
                     expected: Optional[Dict[str, Any]],
                     reference_error: Optional[str] = None) -> None:
    claim_id = f"out.{name}.reference"
    if reference_error is not None:
        build.add(claim_id, FUNCTION, OPEN, reference_error,
                  closer="the entry's reference= callable")
        return
    if value is None:
        build.add(claim_id, FUNCTION, UNDETERMINED,
                  f"the run returned no output named {name!r}",
                  closer="the entry's tensors= mapping")
        return
    got = np.asarray(value)
    if expected is None:
        build.add(claim_id, FUNCTION, OPEN,
                  "no reference declared, so correctness is unknown — this is "
                  "open rather than skipped because a kernel that writes "
                  "plausible nonsense produces a perfectly consistent report",
                  closer="the entry's reference= callable")
        return
    if name not in expected:
        build.add(claim_id, FUNCTION, UNDETERMINED,
                  f"reference= returned no entry for {name!r}",
                  closer="the entry's reference= callable")
        return
    want = np.asarray(expected[name])
    if want.dtype.kind == "f" and want.dtype.itemsize < 4:
        build.add(claim_id, FUNCTION, OPEN,
                  f"reference for {name!r} is {want.dtype}: a reference computed "
                  "at the kernel's own precision reproduces the kernel's own "
                  "overflow and then agrees with it — compute in f32 or wider",
                  closer="the entry's reference= callable")
        return
    if want.shape != got.shape:
        build.add(claim_id, FUNCTION, OPEN,
                  f"shape {got.shape} != reference {want.shape}",
                  closer="the kernel, or the reference")
        return
    rtol, atol = build.entry.tolerance.get(name, (REF_RTOL, REF_ATOL))
    close = np.allclose(got.astype(np.float32), want.astype(np.float32),
                        rtol=rtol, atol=atol)
    diff = np.abs(got.astype(np.float32) - want.astype(np.float32))
    if close:
        # A widened pair is stated even when the claim closes.  A tolerance is the
        # one part of this claim that can be adjusted until it passes, so the run
        # that passes has to say which pair it passed against.
        build.add(claim_id, FUNCTION, CLOSED,
                  f"max abs diff {float(diff.max()):.4g} against a declared "
                  f"rtol={rtol}, atol={atol}"
                  if (rtol, atol) != (REF_RTOL, REF_ATOL) else "")
        return
    build.add(claim_id, FUNCTION, OPEN,
              f"max abs diff {float(diff.max()):.4g} at output magnitude "
              f"{float(np.abs(want).max()):.4g} "
              f"(rtol={rtol}, atol={atol})",
              closer="the kernel, or the reference")


def _claim_unwritten_references(build: _Builder, output_args: Tuple[str, ...],
                                expected: Dict[str, Any]) -> None:
    """A reference entry for a tensor no store in the trace wrote.

    The mirror of the ``name not in expected`` branch above, and both directions are
    needed because the claim set is derived from the store trace rather than from
    the declaration: a reference key the trace never names raises no claim at all,
    so an output that stops being written loses its comparison instead of failing
    it — which is one of the defects this ledger exists to catch.  Reported at the
    id the comparison would have had, so the vanished claim reappears under its own
    name rather than as a remark on a different one.

    Asked only where ``out.identified`` closed.  Where it did not, which tensors the
    kernel wrote is exactly what is in doubt, "no store wrote it" is not established
    for any key, and that claim already blocks.
    """
    for name in sorted(set(expected) - set(output_args)):
        build.add(f"out.{name}.reference", FUNCTION, UNDETERMINED,
                  f"reference= returned an entry for {name!r} and no store in the "
                  "trace wrote it, so nothing was compared against it",
                  closer="the entry's reference= callable, or the kernel")


def _claim_cost(build: _Builder, ledger: Ledger, entry: KernelEntry,
                params: Dict[str, Any]) -> None:
    """cost.derivation — the committed attribution, compared verbatim.

    The derivation is generated, committed, and compared verbatim, so a change in
    what the kernel costs arrives as a reviewable diff naming the term that moved.
    That is what a sentence in prose disagreeing with the model does not survive.

    What it structurally cannot do is catch the cost model itself being wrong: the
    derivation is produced by the same code as the measurement, so a mis-charged op
    moves the total and its breakdown together and the page stays self-consistent.
    That question is asked of the model rather than of any one kernel, and lives in
    ``tests/test_latency.py``, which holds hand-counted bytes, FLOPs and cycles
    against every latency category and every hardware parameter that scales them.
    Re-asking it here, once per declaration, would make each kernel re-answer one
    question about the single cost model they all share.
    """
    if ledger.report is None:
        build.add("cost.derivation", COST, UNDETERMINED,
                  "the kernel did not run, so there is nothing to derive",
                  closer="exec.runs")
        return

    path = REPO_ROOT / DOCUMENT
    try:
        rendered = render_derivation(entry, ledger, params)
    except Exception as exc:
        build.add("cost.derivation", COST, UNDETERMINED,
                  f"could not render: {type(exc).__name__}: {exc}",
                  closer="ktir_cpu/kernelentry/derivation.py")
        return

    committed = read_sections(path.read_text()).get(entry.name) if path.exists() else None
    if committed is None:
        build.add("cost.derivation", COST, OPEN,
                  f"{DOCUMENT} has no section for {entry.name} — run `adopt` to "
                  "write it, then read it: it is the attribution a reviewer confirms",
                  closer=DOCUMENT)
    elif committed == rendered:
        build.add("cost.derivation", COST, CLOSED)
    else:
        build.add("cost.derivation", COST, OPEN,
                  f"the {entry.name} section of {DOCUMENT} is stale — regenerate "
                  "with `adopt` and read the diff, which names the term that moved",
                  closer=DOCUMENT)
