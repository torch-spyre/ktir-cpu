# ktir_cpu review checks

The closed list of traps specific to this repository. Every item was raised in an actual review here, or is enforced by the implementation today. Sections 1–9 cover the code and are ordered by how well attested each category is, strongest first. The last two sit at the end by kind rather than by weakness: §10 is about the text around the change — the PR description, commit trailers, and what belongs in a code comment versus the PR thread — and §11 is an open spec question rather than a rule.

`SKILL.md` in this directory owns the review *procedure* and the open-ended judgment passes. This file is the backstop it runs last — **not** a substitute for reading the change. A finding that fits none of these categories is still a finding.

Severity is attached per item: **BLOCKER** (fix before merge), **SUGGESTION** (improvement), **QUESTION** (needs clarification).

## What other repo docs already own

Not repeated here:

- `CLAUDE.md` — the per-op KTIR spec conformance list, code-generation rules, `docs/gap_analysis.md` upkeep, and PR guidelines (CC'ing maintainers and issue authors)
- `CONTRIBUTING.md` — setup, `uv run pytest -v`, one logical change per **commit**, Apache 2.0 headers
- `docs/latency.md` — the cost model, roofline definitions, and metric conventions
- `.claude/skills/ktir-dialect.md` — a cached normative subset of RFC 0682, for orientation

General engineering style — naming, import placement, API layering — is out of scope. This file is for what is particular to KTIR and to this interpreter.

## What CI does and does not cover

The `test (3.12)` workflow runs **two pytest invocations and nothing else**: `pytest -v --ignore=tests/mlir_frontend` and `pytest tests/mlir_frontend/ -v`. There is no ruff/black configuration, no pre-commit hooks and no coverage gate — formatting and comment hygiene are manual review items here, so don't assume a linter caught anything.

There is a second required check, **`DCO`**, and it is a GitHub App rather than a workflow — so it is invisible in `.github/workflows/`. Read `gh pr checks` for the real gate list; the repo tree does not carry it.

CI does more than test, though: it builds the MLIR frontend from a commit of the upstream `ktir-mlir-frontend` repository, pinned in `pyproject.toml`. It fetches `scripts/setup_mlir.py` and `cmake/llvm-hash.txt` from that commit — both live upstream, not in this repository, so don't look for them in the tree.

- **BLOCKER** — Bumping that pin is a cross-repo change. The new commit must exist upstream, and the `llvm-hash.txt` at that commit is what selects the LLVM artifact. Both suites need re-running, because the frontend's op coverage moves with the pin.

## 1. Boundary validation and guards

The most broadly-attested category in this repo's review history — raised by five different reviewers across seven PRs.

- **BLOCKER** — **Never silently accept malformed input.** At every boundary where a wrong-but-non-crashing value can pass through, reject it rather than coercing. The recurring forms:
  - a lookup or dispatch table with a silent fallback instead of a raising else-branch
  - an **arity mismatch** — a multi-result handler returning a scalar or a short tuple while the caller expects N names
  - **empty-list-as-absent** — `dims=[]` parsed as unset, then read downstream as "all axes", so a shape-preserving reduce silently becomes a full reduction
  - an **unvalidated rank or shape assumption** — a handler that flattens assuming 1-D
  - a helper that can return `None`, unpacked without a guard
- **BLOCKER** — An `assert` is not a guard for input-driven invariants: it compiles out under `python -O`, so malformed input passes silently. Use `ValueError` when the invariant can be violated by input rather than only by internal logic. Keep `assert` (with a message) for internal invariants.
- **BLOCKER** — **Guards are symmetric across paired operations.** When a check lands on one side of a pair (load↔store, push↔pop, alloc↔dealloc), the dual needs it too. Also check *paths*, not just ops: a fast-path early return that skips validation the slow path performs is the same defect. Hoist the check above the branch.
- **BLOCKER** — Every `raise` message names the actual root cause, in terms of the concept that triggered it. A message describing a different limitation than the one that fired sends the next implementer to fix the wrong thing.
- **SUGGESTION** — Each new `raise` has a test that triggers it. Writing the `raise` asserts the case is reachable; the test is what shows it is.
- **SUGGESTION** — Paired operations, paired-state structs, and fields with a range constraint assert their invariant where the object is constructed or at the method head, rather than relying on convention. Docstrings state preconditions explicitly.

## 2. MLIR semantic fidelity

- **BLOCKER** — A cast handler must re-encode bits whenever itemsize differs; identity-passthrough corrupts downstream loads and stores. Comparison ops return `i1`, not the input dtype.
- **BLOCKER** — Match the MLIR spec, not the NumPy function that looks similar. Four traps to check on any one-line handler:
  - NaN propagation — `maxnumf`/`np.fmax` (non-propagating) vs `maximumf`/`np.maximum` (propagating)
  - rounding — `divsi` truncates toward zero, Python `//` floors; they differ only on negative operands, so positive-input tests pass either way
  - signed vs unsigned
  - comparison predicates — ordered (`olt`/`ogt`) vs unordered (`ult`/`ugt`) differ on NaN
- **BLOCKER** — Casting integer tile data through `float32` loses precision for values inside `i32` range but past the f32 mantissa (~2^24). Compute in the widest exact dtype.
- **SUGGESTION** — A handler written for arrays breaks when one operand arrives as a Python scalar. Branch on `np.isscalar` where operands may be mixed.

## 3. Spec conformance and the authoritative op set

Work through the per-op list in `CLAUDE.md`. Two points that recur beyond it:

- **BLOCKER** — What a parser tolerates is not what the spec supports. Result count is determined by `type(results)`; a non-standard syntactic shortcut is not supported merely because a parser accepts it. If an implementation depends on a form the spec doesn't define, the spec change comes first.
- **BLOCKER** — The authority is layered, and reaching for the merged RFC by reflex will sometimes find nothing. Merged semantics live in RFC 0682. The `ktdp` ops **as actually built** are defined by the pinned `ktir-mlir-frontend`'s `KtdpOps.td` — `tests/mlir_frontend/test_registry_consistency.py` calls it "the authoritative ktdp dialect". Ops still under design track an upstream revision, and `docs/gap_analysis.md` records which. Cite the layer that actually settles the question.

## 4. Parser paths — reachability and blast radius

Two parse paths reach the same executor: the regex parser (`KTIRParser` in `ktir_cpu/parser.py`) and the MLIR frontend (`ktir_cpu/mlir_frontend/parser.py`, via `@MLIRTypeAdapter.install` handlers). CI runs them as two separate suites.

The boundary is **not** a directory boundary, and assuming it is will mis-scope a review. Only `ktir_cpu/mlir_frontend/parser.py` is frontend-only, and only `KTIRParser` itself is regex-only. `KTIRParserBase`, `parser_utils.py`, `parser_ast.py`, `affine.py`, `ir_types.py`, all of `dialects/` and `ops/`, and the entire execution and cost-model side are **shared**. A change to `parser_utils.py` reads like a regex-parser change and lands on both paths.

- **BLOCKER** — The regex parser validates against no dialect and has a catch-all fallback (`_parse_general_operation`), so a freshly `@register`'d executor op "just works" there. The frontend has no fallback: `MLIRTypeAdapter.adapt_op` raises `NotImplementedError` for any op without an explicit handler. A PR adding an op either installs a frontend handler or adds itself to `FRONTEND_UNSUPPORTED` in `tests/mlir_frontend/test_registry_consistency.py` **with a reason**. That guard test exists because the divergence "has bitten us repeatedly".
- **BLOCKER** — An op that parses only on the regex path *because that path validates nothing* is a conformance bug in this repo, not a frontend gap. The allow-list says so in its own comment. Reconcile the op to its real form or remove it; do not close the gap by teaching the frontend a non-spec op name.
- **SUGGESTION** — Tests land in the suite matching the path changed. A change to a shared module needs coverage on both.

## 5. Addressing and cost accounting

- **BLOCKER** — HBM addresses are **stick addresses**, not byte addresses. `HBMSimulator.alloc` returns `byte_addr // STICK_BYTES` and asserts the allocation pointer stays stick-aligned. A new allocation or addressing path preserves both properties; a byte address in a stick-address position is silent corruption, not a type error.
- **BLOCKER** — HBM traffic is charged at stick granularity — `unique_sticks * STICK_BYTES` — never as the packed byte size of the result. A load handler stamps `unique_sticks` on the result Tile (plus `index_unique_sticks` when an indirect access tile is involved); a store handler returns the count as the op result, since a store has no result Tile to carry it. `_data_size` in `ktir_cpu/latency.py` raises if either is missing — read its docstring for the two carriers before adding a memory op.
- **SUGGESTION** — The worked example in `docs/latency.md` §"Indirect access memory cost" uses `result_tile_bytes + Σ index_view_bytes`, which agrees with stick-granular accounting only because that fixture is dense and stick-aligned (64×64 f16 = 8192 B = exactly 64 sticks). Don't generalise the packed-bytes form to a non-contiguous access pattern.
- **SUGGESTION** — A test asserting a cost-model change pins the per-unit formula in a comment or docstring, not just the aggregate total. An aggregate assertion passes for the wrong reason as soon as two terms move in opposite directions. Update `docs/latency.md` in the same PR when the model changes, and don't restate its formulas elsewhere — a second copy drifts.

## 6. In-place `outs` safety

- **BLOCKER** — An op registered with `inplace_outs=True` routes accumulation through `_accumulate_inplace` in `ktir_cpu/dialects/linalg_ops.py`; it does not mutate its `outs` buffer directly. MLIR value semantics let the frontend feed a **shared** value into `outs` — a hoisted `arith.constant dense<0.0>`, an `scf.for` iter_arg init, one `%cst` consumed twice — and in-place mutation then corrupts it for every later use. `_accumulate_inplace` copies an uncharged buffer into a fresh chargeable one first, restoring the precondition without disturbing the cost model. Note the precondition is *uniqueness per use*, not object identity: an assertion comparing object identity passes the aliased-constant case it is meant to catch.

## 7. Dynamic shape

- **BLOCKER** — Dynamic shape must propagate end-to-end. `memref<?>`, SSA sizes and symbolic affine dims need to flow through parsing → load/store → execution. **Parser support alone is not "support"** — drive at least one `ktdp.load` / `ktdp.store` through the same SSA dim before the feature is considered done.

## 8. MLIR fixtures

- **BLOCKER** — For each `.mlir` fixture touched: compute each tensor's byte range from its shape and dtype, and verify no two seed addresses produce overlapping ranges.
- **BLOCKER** — `conftest.py` entries match the actual kernel signatures. Stale argument lists outlive the MLIR they described.
- **SUGGESTION** — Magic numbers in fixtures carry a one-line comment giving the unit and the equivalent in the canonical unit, e.g. `// stick 64 = byte 8192`. A bare `arith.constant 8192 : index` is opaque.

## 9. Tests

- **BLOCKER** — Every `pytest.raises(...)` specifies `match=` on a non-trivial substring of the message. Without it the assertion passes on *any* exception, so an unrelated bug masquerades as the expected failure. `pytest.raises(Exception, match=...)` is the established form here — for parser paths that wrap or re-raise, the concrete class is an implementation detail and pinning it makes the test brittle. Name a narrower class where the raise site guarantees one.
- **BLOCKER** — A numeric value appearing in both a fixture and a test assertion is parsed out of the IR, not hardcoded in both places. Duplicated constants desynchronise silently.
- **SUGGESTION** — Edge cases adjacent to the change are enumerated explicitly: `N=1` as well as N=many, divisibility boundaries (`n % cores == 0` and not), signed-negative inputs, NaN/inf paths, and any dynamic-shape path driven through to load/store rather than only parsed.
- **BLOCKER** — A new spec-gap test carries `xfail(strict=True)`, not a bare `xfail`. `tests/test_spec_gaps.py` is the gap ledger — its own docstring says "one `xfail(strict=True)` per known RFC conformance gap" — and `strict=True` is what makes a *closed* gap fail the build, so `docs/gap_analysis.md` cannot go stale unnoticed. A non-strict `xfail` reports XPASS, CI stays green, and the gap closes in silence. (`xfail_strict` is not set in `pyproject.toml`, so the marker is the only thing carrying this.)
- **SUGGESTION** — A known limitation deliberately left out of scope has a test asserting the *current* behaviour, with the tracking issue named in the test docstring. A future fix flips the assertion, and the limitation stays regression-tested meanwhile.
- **SUGGESTION** — A fix comes with a focused unit test, not only end-to-end coverage: the end-to-end test catches the bug, the unit test pins the contract.

## 10. PR description and commits

- **SUGGESTION** — The PR description works as the spec of the change: summary, changes by file or area, **what is *not* changing**, test plan, suggested follow-ups, open questions. A reviewer should not have to infer intent from the diff. The "what is not changing" section prevents the most wasted review — it stops a reviewer assuming a neighbouring boundary was also touched. Quantitative changes (latency, byte counts) come with a before/after table rather than prose.
- **BLOCKER** — Every commit carries a `Signed-off-by:` trailer. The `DCO` check gates this per commit, so a missing trailer is a red build rather than a review nit.
- **SUGGESTION** — Inline comments are self-contained and role-neutral: no `#NNN` issue references (they rot once the issue closes and the code stays) and no contributor names or first-person pronouns. Issue links and discussion belong in the commit body or PR thread, which records authorship already.

## 11. Out-of-bounds semantics — an open question, not a rule

- **QUESTION** — RFC 0682 defines the *valid* points of an access tile via `access_tile_set`, but does not state what happens outside them. There is no specified zero-padding, masking or drop behaviour. A change that depends on out-of-bounds behaviour should raise the spec question rather than encode an assumption — and should say in the PR which behaviour the implementation assumes.
