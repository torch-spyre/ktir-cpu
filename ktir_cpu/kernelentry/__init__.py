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

"""What it takes for this simulator to fully support one kernel, as a computed set.

"Fully supported" is otherwise an adjective, and an adjective cannot be checked.
Here it is a **ledger**: a set of claims derived from the kernel itself, each one
either closed, or open with the file that would close it named.  The set is not a
fixed checklist — every distinct op in the kernel contributes claims, every output
tensor contributes claims — so it grows with the kernel rather than with this
module.

A contributor's loop is::

    python -m ktir_cpu.kernelentry probe   examples/latency/my_kernel.py
    python -m ktir_cpu.kernelentry adopt   examples/latency/my_kernel.py
    python -m ktir_cpu.kernelentry verify --all

``probe`` is read-only and answers "what does the simulator not support about my
kernel" before any work starts.  ``adopt`` writes only the files it owns and never
edits the interpreter: when a new handler or a repriced op is needed it prints the
edit for a human to apply.  ``verify`` is the gate, shared with
``tests/test_kernelentry.py`` so CI enforces the same engine.

Two idioms here are the repository's own: a waiver mapping whose every entry
carries a reason (``tests/mlir_frontend/test_registry_consistency.py``), and
``xfail(strict=True)`` so a gap that closes fails the build until the excuse for it
is removed (``tests/test_spec_gaps.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .tensorspec import build_tensors, validate_specs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# ---------------------------------------------------------------------------
# Claim states
# ---------------------------------------------------------------------------

CLOSED = "closed"
OPEN = "open"
UNDETERMINED = "undetermined"
DEFERRED = "deferred"
WAIVED = "waived"
SKIP = "skip"

#: States that keep the ledger from being clean, i.e. that fail ``verify``.
BLOCKING = (OPEN, UNDETERMINED)

# ``undetermined`` exists because its absence is invisible.  A claim the engine
# cannot evaluate — an output tensor it failed to identify, a figure it could not
# resolve — must not be omitted, because an omitted claim reads exactly like a
# closed one in any summary.  It is distinct from ``waived`` (never applies to
# this kernel) and from ``skip`` (this machine lacks a dependency; CI decides).

_STATE_RANK = {CLOSED: 0, WAIVED: 1, SKIP: 2, DEFERRED: 3, UNDETERMINED: 4, OPEN: 5}

FUNCTION, COST = "function", "cost"


@dataclass(frozen=True)
class Claim:
    """One checkable assertion about one kernel.

    ``closer`` names what would move the claim to ``closed`` — a file, a registry,
    a missing declaration field.  It is the difference between a report that says
    a kernel is unsupported and a report that says what to do about it.
    """

    id: str
    leg: str
    state: str
    detail: str = ""
    closer: str = ""
    #: True when the check needs an optional dependency, so its state is a fact
    #: about the machine as much as about the kernel. The committed support report
    #: renders these to a fixed value: a document compared verbatim cannot depend
    #: on which environment generated it, or CI and a laptop disagree forever.
    env_dependent: bool = False

    @property
    def blocking(self) -> bool:
        return self.state in BLOCKING

    def sort_key(self) -> tuple:
        return (-_STATE_RANK[self.state], self.id)


# ---------------------------------------------------------------------------
# Entry declaration
# ---------------------------------------------------------------------------

@dataclass
class KernelEntry:
    """The single declaration a kernel needs in order to be gated.

    *path* is the kernel's ``.mlir``, relative to ``examples/``, and it is the
    kernel's source — hand-written IR, or captured compiler output such as
    ``examples/triton-ktir/``, which is "kernels as the Triton -> KTIR path emits
    them".  A declaration's path is relative because the report names it that way,
    and a row pointing at somebody's own disk is a row nobody else can regenerate.
    An absolute path is what a kernel read cold from outside this repository carries
    instead: it is answering the questions that need no declaration.

    *gate_params* is deliberately a reduced shape: gate cost is driven by shape and
    not by the number of entries, which is why ``examples/latency/`` holds reduced
    sizes in the first place.  A ``cost.*`` claim evaluated there checks the
    *composition* of a kernel's cost — which tensor dominates — and not the
    absolute figure at full size.

    ``waived`` and ``deferred`` both excuse a claim; the difference is whether
    anything will ever come back for it.  A waived claim never applies here and its
    value is the reason.  A deferred claim does apply and is not met yet: its value
    is an issue reference, the claim runs as ``xfail(strict=True)``, and closing the
    gap fails the build until the deferral is removed.  That is what lets the second
    of a kernel's two PRs carry the cost leg without the first one having to lie.
    """

    name: str
    func: str
    path: str

    gate_params: Dict[str, Any] = field(default_factory=dict)

    #: ``{arg_name: spec}`` — the arguments to call the kernel with, as data.
    #: See ``ktir_cpu/kernelentry/tensorspec.py`` for the vocabulary and for the
    #: per-argument escape hatch.  Empty means the kernel cannot be driven, which
    #: ``exec.runs`` reports rather than skipping.
    tensors: Dict[str, Any] = field(default_factory=dict)
    #: ``reference(params, tensors) -> {arg_name: ndarray}``.  Must compute in
    #: f32 or wider: a reference evaluated in f16 reproduces the very overflow it
    #: is supposed to catch, and then agrees with the kernel about a wrong answer.
    #: *tensors* is a pristine rebuild, not what the run left behind.
    reference: Optional[Callable[..., Dict[str, Any]]] = None

    waived: Dict[str, str] = field(default_factory=dict)
    deferred: Dict[str, str] = field(default_factory=dict)
    #: ``{arg_name: (rtol, atol)}`` — a wider pair than the ledger's default for one
    #: output, because f16 error is set by the magnitude of a dot product's *terms*
    #: and not of its result.  Loosening a tolerance weakens the claim, so it belongs
    #: in the row where a reviewer reads the reason beside it, not in the engine.
    tolerance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError(
                f"{self.name}: path= is the kernel's .mlir, relative to "
                "examples/, and there is nothing to check without it"
            )
        for mapping, what in ((self.waived, "waived"), (self.deferred, "deferred")):
            for claim_id, reason in mapping.items():
                if not str(reason).strip():
                    raise ValueError(
                        f"{self.name}: {what}[{claim_id!r}] needs a reason. An "
                        "excuse with no reason is indistinguishable from an "
                        "oversight."
                    )
        for claim_id, reason in self.deferred.items():
            # The issue is the whole of a deferral's promise: the report groups
            # deferrals by issue, so a closed issue with an open gap stays visible.
            if not re.search(r"#\d+", str(reason)):
                raise ValueError(
                    f"{self.name}: deferred[{claim_id!r}] does not name an issue "
                    f"as #N: {reason!r}. Use waived= for a check that will never "
                    "apply here; a deferral has to say where it is tracked."
                )
        for name, pair in self.tolerance.items():
            # The ledger reads this mapping with .get(), so a misspelled argument
            # name is indistinguishable from a considered widening that took effect.
            if name not in self.tensors:
                raise ValueError(
                    f"{self.name}: tolerance[{name!r}] names no declared tensor "
                    f"(have {sorted(self.tensors)}). A tolerance for an argument "
                    "that does not exist is silently never applied."
                )
            try:
                rtol, atol = (float(v) for v in pair)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{self.name}: tolerance[{name!r}] must be an (rtol, atol) "
                    f"pair of numbers, not {pair!r}"
                ) from None
            if rtol < 0 or atol < 0:
                raise ValueError(
                    f"{self.name}: tolerance[{name!r}] = {pair!r} is negative, "
                    "which no comparison can satisfy"
                )
        validate_specs(self.tensors, self.gate_params, self.name)

    def build_tensors(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The keyword arguments for one run, rebuilt from the declared specs."""
        return build_tensors(self.tensors,
                             self.gate_params if params is None else params)

    @property
    def mlir_path(self) -> Path:
        """Absolute path of this kernel's ``.mlir``."""
        return EXAMPLES_DIR / self.path

    def mlir_text(self) -> str:
        """The kernel's MLIR, as committed."""
        return self.mlir_path.read_text()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ENTRIES: Dict[str, KernelEntry] = {}


def register_entry(entry: KernelEntry) -> KernelEntry:
    """Add *entry* to the registry that ``probe --all`` and the gate iterate."""
    if entry.name in _ENTRIES:
        raise ValueError(f"duplicate kernelentry name {entry.name!r}")
    _ENTRIES[entry.name] = entry
    return entry


def registered() -> Dict[str, KernelEntry]:
    """Every declaration discovered so far, by name."""
    return dict(_ENTRIES)


__all__ = [
    "BLOCKING", "CLOSED", "COST", "Claim", "DEFERRED", "EXAMPLES_DIR", "FUNCTION",
    "KernelEntry", "OPEN", "REPO_ROOT", "SKIP", "UNDETERMINED", "WAIVED",
    "register_entry", "registered",
]
