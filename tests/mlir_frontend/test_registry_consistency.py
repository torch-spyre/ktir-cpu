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

"""Guard test: every executor op must be reachable through the MLIR frontend.

The regex parser tolerates new ops via a catch-all fallback
(``_parse_general_operation``), so a freshly ``@register``'d executor op
"just works" on that path with no parser change. The MLIR frontend has no
fallback — ``MLIRTypeAdapter.adapt_op`` raises ``NotImplementedError`` for any
op lacking an explicit ``@MLIRTypeAdapter.install`` handler.

That asymmetry means an op can be added to the executor (with only a direct
handler-level test) and silently break the frontend, undetected, because the
frontend's coverage is derived from whichever regex test classes happen to be
subclassed into the adapter suites. This has bitten us repeatedly
(dynamic shapes, distributed views, ``linalg.batch_matmul``).

This test makes the divergence fail loudly: any executor op that is neither
frontend-installed nor explicitly listed below (with a reason) breaks the
build the moment it is registered.
"""

import pytest

import ktir_cpu.dialects  # noqa: F401 — triggers @register side effects
from ktir_cpu.dialects import registry
from ktir_cpu.mlir_frontend.parser import MLIRTypeAdapter


# Executor ops intentionally NOT reachable through the MLIR frontend.
# Every entry must carry a reason; remove an entry once its handler lands.
FRONTEND_UNSUPPORTED = {
    # Synthetic / non-IR ops — never appear in parsed MLIR text.
    "region.bb0_args": "regex-only synthetic op; the frontend carries bb0 "
                       "names on linalg.generic via the bb0_names attribute.",
    "return": "alias of func.return for the bare-`return` regex form; the "
              "bindings always emit func.return.",

    # NON-SPEC ops: invented op names that only survive on the regex path
    # (which validates against no dialect). They are NOT in the authoritative
    # ktdp dialect (ktir-mlir-frontend KtdpOps.td) / upstream arith, so the
    # MLIR frontend rightly rejects them. These are ktir-cpu conformance bugs
    # to reconcile to the real op form or remove — NOT frontend gaps to fill.
    "arith.convertf": "not a real upstream arith op — unknown to the MLIR "
                      "parser; reconcile to a real cast op or remove.",
    "ktdp.reduce": "non-spec op (not in the ktdp dialect). Reconcile to the "
                   "real cross-core reduce form or remove. See issue #88.",
    "ktdp.coreid": "non-spec op (not in the ktdp dialect). Reconcile to the "
                   "real core-identity form or remove. See issue #88.",

    # Real spec op missing only a frontend adapter handler (distinct from the
    # non-spec ops above).
    "ktdp.construct_distributed_memory_view": "real ktdp dialect op; no frontend "
                   "adapter handler yet. Tracked in issue #87.",
}


def test_every_executor_op_is_frontend_reachable():
    exec_ops = set(registry._REGISTRY)
    frontend_ops = set(MLIRTypeAdapter._adapt_handlers)

    missing = exec_ops - frontend_ops - set(FRONTEND_UNSUPPORTED)
    assert not missing, (
        "Executor ops with no MLIRTypeAdapter handler and not in "
        "FRONTEND_UNSUPPORTED:\n  "
        + "\n  ".join(sorted(missing))
        + "\n\nEither add a @MLIRTypeAdapter.install handler in "
        "ktir_cpu/mlir_frontend/parser.py, or add the op to "
        "FRONTEND_UNSUPPORTED with a reason."
    )


def test_unsupported_allowlist_has_no_stale_entries():
    # An op listed as unsupported must still be a real executor op, and must
    # NOT have quietly gained a frontend handler (which would make the entry
    # stale and misleading).
    exec_ops = set(registry._REGISTRY)
    frontend_ops = set(MLIRTypeAdapter._adapt_handlers)

    not_registered = set(FRONTEND_UNSUPPORTED) - exec_ops
    assert not not_registered, (
        "FRONTEND_UNSUPPORTED lists ops that are not executor-registered "
        f"(remove them): {sorted(not_registered)}"
    )

    now_supported = set(FRONTEND_UNSUPPORTED) & frontend_ops
    assert not now_supported, (
        "FRONTEND_UNSUPPORTED lists ops that now HAVE a frontend handler "
        f"(remove them from the allowlist): {sorted(now_supported)}"
    )
