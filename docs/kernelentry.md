# Adding a kernel

You have a KTIR kernel and you want `ktir_cpu` to support it — to run it, agree
that it computes the right thing, and report a cost you can trust. "Fully
supported" is otherwise an adjective, and an adjective cannot be checked. Here it
is a **ledger**: claims derived from your kernel, each closed or open with the
file that would close it named.

Sections 1 to 5 are five commands, one each. What follows them is reference.

```bash
uv run python -m ktir_cpu.kernelentry probe examples/latency/my_kernel.mlir   # 1
$EDITOR examples/entries.py                                                  # 2
uv run python -m ktir_cpu.kernelentry adopt my_kernel                        # 3
uv run python -m ktir_cpu.kernelentry probe --all --write-report             # 4
uv run python -m ktir_cpu.kernelentry verify --all                           # 5
```

## 1. Probe, before you write anything

```bash
uv run python -m ktir_cpu.kernelentry probe examples/latency/my_kernel.mlir
```

Point it at the `.mlir` itself; add `--func` if the file declares more than one
function. A path is read cold — no declaration is consulted even if one exists — so
this step is available before step 2 rather than after it. Once the kernel is
declared, name it instead: `probe my_kernel`.

The file does not have to be in this repository. `examples/` holds fixtures, while
the kernels this interpreter is asked to support are versioned elsewhere, so a cold
probe takes any path — a checkout beside this one, a scratch file you are still
editing. Only the steps that write anything need the kernel to live here.

It answers the questions that need no declaration: which of your kernel's ops have
no execution handler, and whether it survives both parse paths. Anything that needs
to *run* the kernel reports that there is no declaration yet. `probe` writes
nothing.

## 2. Add a row

Declaring a kernel is one entry in the `ENTRIES` table at the bottom of
`examples/entries.py`. There is no new file to create, nothing to register and no
test to write — the file is found by name at any depth under `examples/`, so a
vendored kernel set carries its own `entries.py`:

```python
KernelEntry(
    name="my_kernel",
    func="my_kernel",                        # the func.func name, not the file name
    path="latency/my_kernel.mlir",           # relative to examples/
    gate_params={"M": 16, "N": 64, "K": 64},
    tensors={
        "a_ptr": normal(("M", "K")),         # the arguments, as data
        "b_ptr": normal(("K", "N")),
        "c_ptr": zeros(("M", "N")),
        "K": "K",                            # a bare string forwards a parameter
    },
    reference=matmul_reference,               # (params, tensors) -> {arg: ndarray}
),
```

`matmul_small` in that table is this row filled in. Four of the fields have a rule
behind them.

**`tensors` is a mapping, not a function.** The specs — `normal`, `zeros`, `full`,
`tile`, `arange`, `integers`, `asarray`, `param` — are declared in
`ktir_cpu/kernelentry/tensorspec.py` and resolved against `gate_params`: an `int`
dimension is itself, a `str` names a parameter, and a callable is evaluated on the
parameters for a dimension that is an expression of them. A bare string as a *value*
forwards that parameter to the kernel unchanged. Where no spec fits, a callable
`(params, rng) -> value` is the escape hatch, per argument rather than per kernel:
RoPE declares its cos/sin tables the long way and leaves its other two arguments as
rows. Every draw is seeded from the argument's own name, so two arguments are
independent — which is what makes a swapped operand observable — and a rebuild is
bit-identical, so the reference is handed a pristine rebuild rather than the arrays
the kernel has by then written over.

**`path=` is the kernel, not a copy of it.** The `.mlir` under `examples/` *is* the
source — hand-written IR whose point is its exact shape (`examples/ktir/`,
`examples/rfc/`) or captured compiler output (`examples/triton-ktir/`).

**`gate_params` small.** CI runs it on every push, so a `cost.*` claim is evaluated
at reduced size: it checks the *composition* of your kernel's cost, not the absolute
figure at full size.

**`reference` computed in f32 or wider,** and written against the short form of the
answer rather than the kernel's own decomposition. Every value in a KTIR kernel is
f16, so an f16 reference reproduces the kernel's own rounding and then agrees with it
about a wrong answer; the ledger rejects one rather than comparing against it. A
reference that sums shards the way the kernel sums them checks the arithmetic while
assuming the decomposition, which is usually the part under test. No reference at all
leaves `out.<arg>.reference` **open**, not skipped. `out.<arg>.nontrivial` covers the
same f16 range from the other side: an input scale large enough to overflow can zero
an output outright, and no cost report would flag it.

Two fields are for the cases the rules above do not fit:

- `waived=` / `deferred=` — usually omitted, and no row declares either today. See
  *When a claim will not close*.
- `tolerance={"c_ptr": (rtol, atol)}` — a wider pair than the ledger's default, for
  one output. `ffn_swiglu` is the only row that declares one and its comment says
  why. This is the one input to a claim that can be adjusted until it passes, so it
  belongs in the row where the reason is read beside it, and the claim states the
  pair even when it closes.

## 3. Adopt

```bash
uv run python -m ktir_cpu.kernelentry adopt my_kernel
```

It writes your kernel's section of `docs/kernel_cost.md` and leaves the others as
committed, so adopting one kernel does not depend on having run the rest. You commit
the result. It never edits the interpreter: a new op handler or a repriced op is
printed for you to apply, because a tool that quietly changed what the simulator
charges would be changing the answer you asked it for.

You are not asked to write that section, only to read it — and so does the reviewer,
in the diff. Read it there rather than here: a copy of a figure in this document is a
figure nothing regenerates. Four things to look at:

- **The composition**, not the totals: which tensor dominates, what the
  `traffic_ratio` is, which unit the cycles land on. "The weight operand's traffic is
  negligible" does not survive next to a row attributing most of the traffic to it.
- **What it cannot catch is the cost model being wrong.** The same code produces the
  breakdown and the figure it is checked against, so a mis-charged op moves both.
- **`<unattributed>`** is traffic whose origin could not be resolved — printed rather
  than dropped, so the rows always sum to the total. An access landing in more than
  one partition of a distributed view produces it.
- **`traffic_ratio` below 1 is not an error.** The denominator is the whole footprint
  of the tensors you declared, so a kernel that indexes instead of sweeping moves
  less: `examples/triton-ktir/indexed_add.mlir` reaches two of 128 slices, and its
  ratio says so.

## 4. Regenerate the reports

```bash
uv run python -m ktir_cpu.kernelentry probe --all --write-report
```

Writes `docs/kernel_support.md` (per kernel) and `docs/supported_ops.md` (per op),
both committed on the same discipline as a lock file. Not `adopt`'s job: `adopt` acts
on one kernel, these are whole-repository views, and a partial one would claim the
kernels it omits are absent — so `--write-report` requires `--all`.

Adding any kernel under `examples/` makes them stale whether or not you declared it,
so the omission appears in your own diff instead of being an absence. `verify` names
the command when it finds a document stale.

## 5. The gate

```bash
uv run python -m ktir_cpu.kernelentry verify --all
uv run pytest -q --ignore=tests/mlir_frontend
uv run pytest tests/mlir_frontend/ -q     # skips without mlir_ktdp; CI runs it
```

`verify` shares one engine with `tests/test_kernelentry.py`, so the local loop and CI
cannot disagree. Two of the things it checks belong to the repository rather than to
any kernel — that the two generated reports are current, and that the op registry's
zero prices are all accounted for — so `verify --all` can fail with every declared
kernel clean. It says which of the two it is.

The third command is the one a green local run hides. The regex parser validates
against no dialect, so a form only it accepts is one the dialect does not define, and
only the MLIR frontend notices. Without `mlir_ktdp` installed that suite skips at
module level and `parse.frontend` reads `skip`, so CI decides it (see README for
installing the bindings); `verify` counts those claims in its summary line instead of
calling the kernels fully supported.

What it catches, it does not adjudicate: it reports that a form is one the dialect
does not define, not which side should change. Four kernels under `examples/` are in
that position, all undeclared and recorded with their reason in
`ktir_cpu/kernelentry/conformance.py`. If yours lands there, what to do splits on the
kind of disagreement. Where only the spelling differs, rewriting it is a rewrite and
nothing else — the two cross-core kernels declared here were two lines each, both
parsers accepting the result and the cost figures identical to the digit. Where the
dialect disagrees about *types*, closing the gap means deciding something about the
op, which `docs/gap_analysis.md` row 2a carries for the three `ring_reduce*` kernels;
for a declared kernel there, `deferred` against an issue is the honest state.

And per `CLAUDE.md`: CC the maintainers in a comment on the pull request, plus the
issue author if it is linked to an issue.

---

## What is checked, and what you supply

Grouped by the question each claim answers, in the order the tool asks them. Those
questions are the columns of `docs/kernel_support.md`; reading splits in two there,
because the two parsers fail for different reasons and are fixed by different people.
Everything you write is in the last column.

| question | claim | what it asserts | you supply |
|---|---|---|---|
| **is it read** | `parse.regex` | `KTIRInterpreter.load` accepts the kernel | — |
| | `parse.frontend` | the MLIR frontend accepts it and MLIR's own verifier passes | — |
| **does it run** | `op.<name>.handler` | every distinct op has an execution handler | — |
| | `exec.runs` | it executes on the declared grid without overflowing LX | `gate_params`, `tensors` |
| | `out.identified` | the engine could work out which tensors the kernel writes | — |
| **is the output right** | `out.<arg>.reference` | each output matches an independent reference | `reference` |
| | `out.<arg>.nontrivial` | no output is entirely zero or NaN | — |
| **is the cost trustworthy** | `cost.derivation` | the kernel's section of `docs/kernel_cost.md` matches what the model reports now | `adopt` writes it; you read it |

## When a claim will not close

| state | meaning | when the gap closes |
|---|---|---|
| `closed` | checked, passed | — |
| `open` | should hold, does not | — |
| `undetermined` | applies, but the engine could not evaluate it | — |
| `deferred` | known gap, tracked against an issue | reported as unnecessary, which **fails the build** |
| `waived` | never applies to this kernel | reported as unnecessary |
| `skip` | this environment lacks a dependency; CI decides | — |

`open` and `undetermined` both block the gate. They are separate because a check that
*could not run* must not read as one that passed — with no such state an unevaluable
claim gets quietly omitted, which looks identical to a clean result.

`deferred` and `waived` both excuse a claim; the difference is whether anything ever
comes back for it. Deferral is what lets a kernel arrive over two pull requests
without the first one having to pretend the cost leg is done.

```python
deferred={"cost.derivation": "#<issue> — the cost leg lands in the follow-up"},
waived={"out.y.nontrivial": "fully masked at this shape, so all-zero is correct"},
```

- A deferral needs an issue reference, `#` followed by digits; only the format is
  checked, because verifying more would make a local gate depend on the network. So a
  deferral against a closed issue passes, and that is how this state rots — a gap that
  outlives the issue tracking it is caught by a reader of `docs/kernel_support.md`,
  which prints every deferral with its issue, and by nothing else. A waiver needs a
  reason, and an empty one is rejected.
- Either excuse overrides any state but `closed`, a missing optional dependency
  included, so an excused claim reads the same on every machine. Both are printed in
  `docs/kernel_support.md`, so an accumulation is visible in the diff rather than only
  in the code.
- **The claim no longer exists** → `waived.stale.<id>`; usually a typo in the id, or a
  leftover from a kernel that has since changed.
- **The claim now passes on its own** → `waived.unnecessary.<id>` /
  `deferred.unnecessary.<id>`, and the check itself reports `closed`. An excuse that
  outlives what made it necessary is misinformation. This is what makes a deferral
  expire — not the `xfail(strict=True)` marker, which is applied from the claim's
  current state and so is simply absent once the gap closes.

## If your kernel needs a new op

Price it, and pin the price in `tests/test_latency.py` rather than on your
declaration: it holds hand-counted bytes, FLOPs and cycles per latency category and
per hardware parameter that scales them, which is one question about the one cost
model every kernel here shares.

`@register()` defaults `latency_category` to `"zero"`, so "this op is free" and
"nobody priced this op" are the same state in the registry, and a kernel leaning on
an unpriced op reports a lower cost than the hardware would with nothing saying so.
`ktir_cpu/kernelentry/pricing.py` splits the two apart. Every op the registry prices
`zero` has to appear in exactly one of two mappings, each entry carrying a written
reason:

- **`ZERO_COST_OPS`** — free by decision. A terminator, a compile-time constant, an
  address computation, an orchestrator whose body is priced op by op.
- **`UNJUDGED_ZERO_OPS`** — priced zero with nobody having decided that. Each reason
  names the issue that would settle it, because an open question with no issue behind
  it is how the list becomes permanent.

An op in neither, or in both, is a finding: `verify` fails and `probe --all` prints it
with the fix. So registering an op without naming a category no longer passes
silently — but it does not become priced either, and that is the point. The gate asks
you to decide, and recording "not yet judged" against an issue is a legitimate answer.

The check compares against `zero`, so it **cannot see a category that is simply
wrong** — an integer compare billed to the float pipe passed it for as long as it
existed, and only a reader who knows the op's semantics catches that.
`docs/supported_ops.md` carries the result as a **cost** column, printing the decision
rather than the registry's default.

## What no claim can tell you

The ledger decides whether the simulator supports your kernel, not whether your
kernel is the kernel you meant: whether the IR expresses the algorithm, whether the
grid is the one you want to model, whether `access_tile_order` is lexicographic with
the rightmost dimension innermost, whether overlapping coordinate sets in a
distributed view are constrained enough to have defined behaviour. Those are read by a
person, against RFC 0682 and the checklist in `CLAUDE.md`.

## Where this repository stands

18 of the 33 kernels under `examples/` are declared, every claim they raise is closed,
and none of the eighteen defers or waives one. `docs/kernel_support.md` is the current
state per kernel, and its second table is the other 15: what *has* been asked of them,
with the columns nobody asked absent rather than empty.

None of those 15 is waiting on somebody to transcribe a reference. Thirteen take no
tensor arguments the `tensors=` mapping can express — the eight `examples/rfc/*` files
address memrefs at absolute HBM bases, and five more (the three ring-reduce kernels
and the two `rmsnorm_4core_*`) take raw HBM element indices with no shape attached,
which their tests supply by replacing `KTIRInterpreter._prepare_execution`. That is a
property of the argument convention rather than of those files, and it is the live one:
the two rmsnorm kernels are the most recent to arrive. The last two are deliberate:
`softmax_wide.mlir` overflows LX on purpose and its test asserts the exception, so
`exec.runs` failing is the kernel behaving; `nested_yield.ktir` is a reproducer written
down to the one op under test, so being minimal IR rather than dialect-valid IR is what
it is for.

What the 18 do not buy is cost coverage at real shapes for free. Probing all of them
takes about 21 s and five kernels are 97% of it — `layernorm_fwd_ktir` alone is 9.4 s,
then `paged_attention`, `softmax_fwd_ktir`, `matmul_fwd_ktir` and `rope_fwd_4x2` —
while the remaining 13 come to well under a second together. Shape drives that, which
is the trade `gate_params` exists to make; for these five there is nothing to trade,
because their shapes are baked into the IR. They are here anyway, on the grounds that a
cost model checked only at reduced size is checked where the padding, the tail core and
the page table are not real yet.
