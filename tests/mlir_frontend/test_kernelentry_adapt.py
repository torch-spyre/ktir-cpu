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

"""The frontend leg of the gate: ``parse.frontend`` must actually be decided.

Without ``mlir_ktdp`` the ledger reports ``parse.frontend`` as ``skip``, which is
correct — but a claim that is only ever skipped is a claim that never runs. This
module is where it runs. It lives under ``tests/mlir_frontend/`` so it inherits
that package's module-level skip, and therefore only executes where the bindings
exist: in CI.

Why the claim matters more than the other parse claim: the regex parser accepts
unregistered ops through a catch-all, and the frontend has none — it runs MLIR's
own verifier. A kernel that works only on the tolerant path is exactly what a
green local run hides.
"""

from __future__ import annotations

import pytest

from ktir_cpu.kernelentry import BLOCKING, DEFERRED, SKIP, registered
from ktir_cpu.kernelentry.cli import discover_all
from ktir_cpu.kernelentry.ledger import probe

discover_all()
ENTRIES = registered()


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_kernel_parses_on_the_mlir_frontend(name: str):
    """Every declared kernel is accepted by the real frontend, verifier included.

    A deferral passes here. The declaration having recorded a known frontend gap
    against an issue is a different situation from nobody having looked, and it is
    the gate in ``tests/test_kernelentry.py`` that holds the deferral to account —
    duplicating that here would report one gap as two failures.
    """
    ledger = probe(ENTRIES[name])
    claim = next(c for c in ledger.claims if c.id == "parse.frontend")
    assert claim.state != SKIP, (
        "parse.frontend was skipped in an environment that has mlir_ktdp — the "
        "ledger's import guard is too broad, and the claim will never be decided"
    )
    assert claim.state not in BLOCKING, f"{claim.state}: {claim.detail}"
    if claim.state == DEFERRED:
        pytest.skip(f"deferred: {claim.detail}")


class TestTheCommittedRecordOfFrontendRejections:
    """``conformance.FRONTEND_REJECTS`` must say what the frontend actually says.

    ``docs/kernel_support.md`` prints that mapping instead of what the generating
    machine saw, because the check needs the optional MLIR bindings and the report
    is compared verbatim. A committed record only earns that role if something
    checks it, and this is the one environment that can: here the frontend is real.

    Both directions, because each failure mode is its own kind of wrong. A missing
    entry means the report calls a rejected kernel accepted — the exact outcome
    listing undeclared kernels at all is meant to prevent. A stale entry means it
    reports a gap that somebody has since closed, which is how a record stops being
    read.
    """

    @staticmethod
    def _rejected() -> dict:
        """Every kernel under ``examples/`` the frontend will not accept."""
        from ktir_cpu.kernelentry import EXAMPLES_DIR, REPO_ROOT
        from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

        out = {}
        for path in sorted(EXAMPLES_DIR.rglob("*")):
            if path.suffix not in (".mlir", ".ktir"):
                continue
            rel = str(path.relative_to(REPO_ROOT))
            try:
                MLIRFrontendParser().parse_module(path.read_text())
            except Exception as exc:  # noqa: BLE001 — any rejection counts
                out[rel] = str(exc)
        return out

    def test_every_rejected_kernel_is_recorded_or_declared(self):
        """Recorded in the mapping, or declared with the claim excused. Not neither."""
        from ktir_cpu.kernelentry import EXAMPLES_DIR, REPO_ROOT
        from ktir_cpu.kernelentry.conformance import FRONTEND_REJECTS

        excused = set()
        for entry in ENTRIES.values():
            if "parse.frontend" in entry.waived or "parse.frontend" in entry.deferred:
                excused.add(str(entry.mlir_path.resolve()
                                .relative_to(REPO_ROOT.resolve())))

        unrecorded = {
            rel: err for rel, err in self._rejected().items()
            if rel not in FRONTEND_REJECTS and rel not in excused
        }
        assert not unrecorded, (
            "the MLIR frontend rejects these kernels and nothing in the repository "
            "says so, so docs/kernel_support.md reports them as accepted. Add each "
            "to FRONTEND_REJECTS in ktir_cpu/kernelentry/conformance.py with a "
            "reason, or excuse parse.frontend in its declaration:\n  "
            + "\n  ".join(f"{rel}: {err.splitlines()[0]}"
                          for rel, err in sorted(unrecorded.items()))
        )

    def test_the_record_has_no_stale_entries(self):
        """A recorded gap that has been closed must come out of the record."""
        from ktir_cpu.kernelentry import REPO_ROOT
        from ktir_cpu.kernelentry.conformance import FRONTEND_REJECTS

        rejected = self._rejected()
        missing = [rel for rel in FRONTEND_REJECTS if not (REPO_ROOT / rel).exists()]
        assert not missing, (
            "FRONTEND_REJECTS names kernels that no longer exist (remove them): "
            f"{sorted(missing)}"
        )
        now_accepted = [rel for rel in FRONTEND_REJECTS if rel not in rejected]
        assert not now_accepted, (
            "the MLIR frontend now accepts these, so FRONTEND_REJECTS is reporting "
            "a gap that is closed — remove the entries and regenerate "
            f"docs/kernel_support.md: {sorted(now_accepted)}"
        )
