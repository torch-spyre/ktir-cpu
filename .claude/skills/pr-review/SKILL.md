---
name: pr-review
description: "Review a pull request on ktir_cpu — correctness, design, spec grounding, test coverage and this repo's known traps. Use when asked to review a PR, re-review one after new commits, or check whether a change is ready to merge."
allowed-tools: Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh pr checks:*), Bash(gh issue view:*), Bash(gh api:*), Bash(git:*), Bash(uv:*), Bash(grep:*), Bash(rg:*), Bash(cd:*), Read, Grep, Glob
---

# PR Review for ktir_cpu

This file is the review *procedure*. `ktir-checks.md` in this directory is the closed list of traps specific to this repository, and it runs **last** — see step 8.

The order matters. Form your own account of the change first; use the list to catch what you missed. Running the list first narrows what you are able to see.

Everything needed is in this repository plus an authenticated `gh` (`gh auth status`), run from inside a checkout. Nothing is read from outside it, and no review state is stored anywhere local — the PR thread is the record, so a reviewer picking a PR up cold has the same ledger as everyone else.

## Principles

The steps below implement these. Every list in this file — the correctness lens, the five design questions, the coverage archetypes, the checks — is a prompt, not a boundary. When something comes up that no list anticipates, follow these principles and report what you actually found.

1. **Distrust every claim, including your own from last round.** Trust the output of a command you just ran. Not a PR description, not a coverage claim, not a prior review's status, not "the diff looks like it fixes it".
2. **A review is an append-only ledger, not a snapshot.** Every finding gets a label and, in every later round, an explicit disposition. Nothing is dropped silently and nothing is renumbered. This is what makes round three cheap without making it shallower.
3. **Correctness, design and spec-safety are independent.** A change can be correct, badly placed and spec-safe all at once. Judge and report them separately; a single blended impression loses two of the three.
4. **A negative result is a deliverable.** "This touches no spec-defined semantics, because it is confined to interpreter-internal bookkeeping" is a finding. So is "the design holds up." Say them.
5. **The review's own premise can expire.** A PR's scope gets renegotiated in its comments. When it shifts, some earlier findings become *superseded*, not unresolved.
6. **Converge, and lead with it.** One sentence up front: the verdict, and the single thing standing between this PR and merge. Everything after it is supporting evidence, and is left out where there is none.

**Reference convention.** Bare `#N` is only ever a GitHub issue or PR. Findings are `F1`, `F2`…, test gaps `G1`, `G2`…, design findings `M1`, `M2`…. Spell out review-round ordinals ("second round"), so they can't be read as GitHub references.

The label sequence belongs to the PR, not to you. If the thread already contains a `G1` — from any reviewer, in any earlier round — your first gap is `G2`. Read before numbering; a duplicate label makes two findings indistinguishable in the thread.

## Step 1 — Orient

```bash
gh pr view <N> --json title,state,mergeable,mergeStateStatus,reviewDecision,reviews,comments,statusCheckRollup,body,commits,files,additions,deletions,closingIssuesReferences,url
gh pr diff <N>
gh pr checks <N>
```

Note the review decision, CI conclusion, merge readiness and head commit SHA. **Translate status enums into plain language** — write "the branch is behind main", not `BEHIND`. Read the existing comments for the narrative: who found what, what was fixed in response, who approved and on which commit.

**Then read the issue behind the PR.** It is the independent statement of what was needed; the PR body is only the author's claim about what the change does, and principle #1 applies to it too. `closingIssuesReferences` carries the formally linked ones, but it does not catch a PR that names its issue in prose only, which is common here — so also scan the title, body and comments for a bare `#N`, then `gh issue view` it. Two things come out of it: the acceptance criteria step 4 measures against, and the case where `Closes #N` sits on a change implementing only part of the issue, which merges by silently closing a live requirement.

Then branch: is this a first review, or a re-review?

## Step 2 — Re-review path

Take this path when the PR already carries review comments — yours from an earlier round, or another reviewer's thread you are joining. Step 1's `--json reviews,comments` already carries the review bodies and the conversation thread; add the inline comments, which are a separate endpoint:

```bash
gh api 'repos/{owner}/{repo}/pulls/<N>/comments' --paginate
```

All three together are the ledger and none is optional — a PR here can carry a full changes-requested-then-approved cycle with **zero** inline comments, so treating the inline endpoint alone as the ledger would read that PR as never reviewed. Work the combined ledger in this order:

1. **Baseline.** The head SHA you last reviewed, and every prior finding label.
2. **Scope drift, before anything else.** Re-read the current title and body against what your last round assumed the PR was doing, and skim new comments for renegotiation — "let's split this", "also handling X now", a requested cut. Scope drift decides which checks still apply, so establishing it first prevents re-checking things that no longer matter.
3. **Disposition every prior finding**: resolved (cite the commit or line), not resolved, partially resolved (say what is left), or superseded (scope moved). All of them, explicitly, including ones another reviewer raised — an open thread you can see is answered stays open until someone says so. A claimed fix is resolved when you can point at the commit that does it and CI is green on that head — not because the diff looks right.
4. **Then the lenses below, on genuinely new hunks only.** Continue the label sequence; never reuse or renumber.

If there are new commits but no prior record of yours, say so and treat this as a first review rather than implying the earlier rounds were covered.

## Step 3 — Use CI as the test evidence

**`gh pr checks` is the test result.** CI runs both suites on every PR, and `xfail(strict=True)` turns a silently-closed spec gap into a red build, so a green CI is real evidence. Don't rebuild the project to re-derive what CI already told you.

One thing CI cannot tell you: **whether a new test actually fails without the fix.** A regression test that passes either way is decorative, and nothing in the pipeline detects it. Ask the author — "does this test fail on `main`?" — rather than building an environment to find out yourself. It is cheaper, and it asks the person whose environment is already set up.

Run something locally only to **substantiate a finding of your own**: you suspect a defect no test covers and want a reproduction before asserting it. Then work outside your own checkout (a worktree is convenient) and rule out your own setup before writing it up — a failing command is not evidence against the author until you know it fails for the PR's reason.

Never build the MLIR frontend to review a change. It needs the pinned LLVM artifact, `ninja`, `nanobind`, `--no-build-isolation` and `MLIR_DIR`; a failure there is far more likely to be your toolchain than the PR. The divergence that suite guards is checkable *statically* anyway, in the `FRONTEND_UNSUPPORTED` allow-list of `tests/mlir_frontend/test_registry_consistency.py`.

## Step 4 — Correctness

- **Does it do what the PR claims, and what the issue asked for?** Trace the affected paths; don't infer from either summary. Where the two disagree, the issue is the requirement and the difference is a finding — including a partial implementation that still closes the issue.
- **Scope completeness.** Grep for every op or caller matching the same pattern and confirm none was silently missed. Note what holds only by symmetry.
- **Side effects.** Accounting and bookkeeping, aliasing, lifetime, and any invariant the change *relaxes* — confirm a loosened assertion doesn't open a new hole.

## Step 5 — Design

Independent of steps 4 and 7 (principle #3). Applies to **all** code the PR touches, tests included — a new test file that forks the existing suite structure is as much a finding as duplicated production logic. Five questions:

- **Structural divergence.** A new abstraction, mechanism, file or location where an existing one already covers this need. Would a maintainer expect this shape, in this place, given the conventions here?
- **Duplication.** Could this extend or call the existing implementation instead of standing up a second one?
- **Complexity.** Does the added branching and interdependency match the problem's actual difficulty?
- **Cohesion.** Does this make one file or component do two jobs the codebase otherwise keeps apart?
- **Proportionality.** Is the change's size commensurate with what it fixes? This catches both a small fix dragging in a disproportionate mechanism and a broad problem patched with a one-off.

Label these `M1`, `M2`…. **A PR can have zero `M` findings** — don't force one if the design holds up against all five.

## Step 6 — Test coverage gaps

Verify coverage against the **actual test files on the branch**. Do not infer it from the PR description. For each gap, name the uncovered path and why it matters:

- the dtype the bug was reported in, versus the dtypes actually tested;
- every branch the PR added — a helper applied to two ops but exercised through only one;
- an accounting or invariant assertion that exists on a sibling path and is missing on the new one;
- end-to-end coverage (parse → execute real IR) versus hand-built unit objects;
- multiplicity and aliasing cases.

Label `G1`, `G2`… and split them: cheap **and** covering code this PR introduced (worth closing before merge) versus quality improvements (follow-ups).

## Step 7 — Ground it against the spec

Don't reach for the merged RFC by reflex; the authority is layered (see `ktir-checks.md` §3). State plainly whether the change touches spec-defined semantics or is confined to interpreter-internal simulation bookkeeping — **and if it is spec-safe, say so and say why** (principle #4). Check it against `CLAUDE.md`'s conformance rules: don't invent ops or attributes, don't change the semantics of existing ones, preserve the separation of concerns.

One more question on any change to the cost model or the hardware constants: **the coarse constants are load-bearing.** This repo's hardware constants are deliberate approximations, and its own `README.md`, `docs/` and simulator code are the source of truth for them. A PR that changes one because some other source says otherwise needs a maintainer's decision, not a silent edit. Specifics from internal sources belong in neither the repo nor the PR text.

## Step 8 — Run `ktir-checks.md`

Now work the closed list, as a backstop for what steps 4–7 missed.

Three things keep it from becoming a ceiling:

- **It is a floor.** The list is what has bitten this repo before, not the set of things that can go wrong. A finding that fits no category is still a finding. Read the unchanged code around the diff, not only the diff.
- **Suspect the rule.** If a check flags code a maintainer wrote, re-read the check before writing the finding. A rule generalised past its evidence is likelier than a maintainer breaking their own convention. Cite what the check rests on rather than asserting it, and prefer QUESTION.
- **Rule out your own environment** before writing up anything that rests on a command failing.

## Output

One review comment, drafted for a human to read and post. **Never post automatically.**

Three rules about the form, before the shape:

- **Emit it as ordinary markdown in the reply — not inside a code fence.** The fence below only shows the layout. A fenced block does not soft-wrap, so it reads as one long line per paragraph in a terminal, and it is the wrong thing to paste into GitHub. Let prose wrap naturally — one line per paragraph, no hard line breaks mid-sentence, which is what the markdown in this repo does.
- **Length is set by the findings, not by the template.** A section with nothing in it is **left out**, not filled with reassurance that there was nothing. Only the verdict line and `Summary` are always present — a clean small PR is four lines. The exception is a deliberate negative result (principle #4): "spec-safe, and here is why", "the design holds up against all five" — that is a finding and it gets said, as one clause rather than a paragraph. Empty is not the same as checked-and-clean.
- **Don't restate the change.** The author already described it, and the diff is one click away. One sentence per finding and one for the fix; no walking the reader through code they can read.

No first person, no raw GitHub status enums.

```markdown
## PR #<number>: <title>

**REQUEST CHANGES** — <the single thing standing between this PR and merge; or, on APPROVE, that nothing does>

### Summary
<one or two sentences: what changed, files touched with +/- counts, the issue it closes, plain-language merge readiness>

### Findings
- F1 — `path/to/file.py:42` — <defect> → <fix>   [blocker | suggestion | question]

### Design
- M1 — <finding>

### Test gaps
- G1 — <uncovered path, why it matters>   [close before merge | follow-up]

### Spec grounding
<which layer settles it; whether the change is spec-safe, and why>

### Verification performed
<CI conclusion in plain language; scope-completeness greps done; anything reproduced locally and how>
```

Where the issue result goes: when the change matches what the issue asked for, one clause in `Summary` carries it — "closes #168, whose acceptance criteria it meets". When it doesn't, step 4 has already made that difference a finding, so it lands in `Findings` with the rest.

A finding's tag is one of the same three severities `ktir-checks.md` uses — `blocker` (fix before merge), `suggestion` (improvement), `question` (needs clarification) — so a check that fires there carries its severity straight into the report without a second vocabulary.

The verdict is one of `APPROVE`, `COMMENT`, `REQUEST CHANGES`. `COMMENT` is the normal in-progress state while threads remain open — not a rejection. Approval follows once they close.

`Spec grounding` and `Verification performed` sit last because they are the audit trail, not the message. Keep them to a line each unless something in them is itself a finding.

On a re-review, replace **Findings** with a *Status of previous findings* block giving every prior label its disposition, then a *New findings* block continuing the sequence.
