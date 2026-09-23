# Real-model e2e fixtures

These drive `tests/e2e_real_forward.rs`, which runs a **real forward of a real
model** end-to-end (prefill + decode, smollm2-135m and llama-3.2-1b) and asserts
the production path (`segmented::execute_segmented`) reproduces the
**transformers** logits. Everything here is vendored so the tests run in default
`cargo test` with **no dependency on the scratchy `~/.cache/cudaforge` bundle**
(weights are fetched from public HuggingFace at test time and content-addressed
cached, which is a different cache).

## Layout (one dir per model × config)

`smollm2-135m/`, `smollm2-135m-prefill/`, `llama-3.2-1b/`, `llama-3.2-1b-prefill/`:

| file | what | source |
|---|---|---|
| `manifest.json` | program description: `tensors` (id/rows/cols/is_source), `sources` (id/**role**/**disk**), `nodes` (fn/mlir/args), `result`, `attn_mask`, `decode_position` | scratchy dump |
| `node*.mlir` | the KTIR program — one `func.func` per node, **weights-free** | scratchy dump |
| `t<id>.f16.gz` | runtime input activations (see roles below), little-endian **f16**, **gzip -9** | `gen_golden.py` |
| `golden.f16.gz` | reference logits `[m, vocab]`, f16, gzip -9 | `gen_golden.py` |

`gen_golden.py` is the (re)generator, committed alongside.

Decode dirs are `m=1`; `-prefill` dirs are `m=32`. The program differs by `m`; the
weights are identical (fetched from the same HF repo).

### Vendoring format — one `tar.gz` per fixture

Each of the four dirs above is committed as a single archive — `smollm2-135m.tar.gz`,
`smollm2-135m-prefill.tar.gz`, `llama-3.2-1b.tar.gz`, `llama-3.2-1b-prefill.tar.gz` —
**not** as the ~1.6k loose files (which turned the PR into a 480k-line diff).
`e2e_real_forward::fixture_dir` unpacks the archive on demand into the cargo target
tmp dir, so the tests are unchanged. An already-unpacked `tests/fixtures/<name>/` is
preferred when present (so you can `tar xzf <name>.tar.gz` and edit in place); such
dirs are **gitignored**, so re-archive before committing:

```bash
# from this directory (tests/fixtures/), after editing/regenerating a fixture dir:
for d in smollm2-135m smollm2-135m-prefill llama-3.2-1b llama-3.2-1b-prefill; do
  COPYFILE_DISABLE=1 tar -czf "$d.tar.gz" -C "$d" .
done
```

### Source roles (manifest `sources[].role`)
- `weight` — an HF weight. `disk` is the HF tensor name (`<disk>.weight`); a tied
  `lm_head` falls back to `model.embed_tokens.weight`. Bound **verbatim `[out,in]`**.
- `embed` — the input activation `[m, hidden]` (= `embed_tokens(input_ids)`, NOT the
  embedding table). `cos`/`sin` — RoPE tables `[m, head_dim]`. `prefix_k`/`prefix_v`
  — the per-layer KV cache (empty/zero for a fresh forward).
- `attn_mask` — additive mask over the (empty) prefix KV cache; all `-65504` (f16
  min) so attention uses only the fresh token(s).

## Origin story — where the `.mlir` programs come from

The programs are **emitted by scratchy** (the cudaforge KTIR emitter) via its
`SCRATCHY_KTIR_DUMP` path, which writes one dump per model/config into
`~/.cache/cudaforge/ktir/<model>{,-prefill}/`. We then **vendor** the program (the
`manifest.json` + the `node*.mlir` it references) into this repo so CI never touches
that cache.

Key properties the emit must have (these were the bugs we hit and fixed — see
`~/.claude/.../memory/hermetic-e2e-tests.md`):
- **Weight matmuls use `linalg.matmul_transpose_b`**, which reads a PyTorch `Linear`
  weight `[out,in]` *verbatim* (contracting the last axis = `xWᵀ`). So we bind HF
  weights **zero-copy with NO transpose** — neither at load nor per-call. (An older
  scratchy emitted plain `linalg.matmul` expecting weights pre-transposed to
  `[in,out]`; that is the *wrong* dump — re-emit, do not transpose on our side.)
- **Wide-output GEMMs are column-tiled** so each tile's `[m, tile_n]` accumulator
  fits the **2 MB per-core LX** (`m · tile_n · 2 ≤ 2 MB`). E.g. the `lm_head` at
  `m=32`: llama (vocab 128256) → 8 tiles ≤16384; smollm2 (vocab 49152) → 3×16384.
  An un-tiled wide GEMM overflows LX at prefill and cannot run on real Spyre either.

### Re-vendoring after scratchy re-emits a dump
Manifest-driven copy (only the `.mlir` the manifest references — the live dir can
hold stale leftovers, and scratchy may still be mid-write, so check the file age):

```bash
# from this directory (tests/fixtures/)
uv run --no-project --with numpy python - <<'PY'
import json, os, shutil, time
for m in ['smollm2-135m','smollm2-135m-prefill','llama-3.2-1b','llama-3.2-1b-prefill']:
    src = os.path.expanduser(f'~/.cache/cudaforge/ktir/{m}')
    man = json.load(open(f'{src}/manifest.json'))
    refs = sorted(set(n['mlir'] for n in man['nodes']))
    age = time.time() - max(os.path.getmtime(os.path.join(src,f)) for f in os.listdir(src))
    assert age > 30, f'{m}: dump modified {age:.0f}s ago — may be mid-write, wait'
    assert all(os.path.exists(os.path.join(src,r)) for r in refs), f'{m}: missing referenced mlir'
    for f in os.listdir(m):                      # wipe old program (keep nothing stale)
        if f.startswith('node') and f.endswith('.mlir'): os.remove(os.path.join(m,f))
    shutil.copy(f'{src}/manifest.json', f'{m}/manifest.json')
    for r in refs: shutil.copy(os.path.join(src,r), os.path.join(m,r))
    print(f'{m}: vendored {len(refs)} nodes')
PY
```
Then regenerate the goldens (next section) — the manifest may have new tensor ids.

## How to (re)generate the goldens + runtime inputs

`gen_golden.py` runs a real `transformers` forward (public repos, **no `HF_TOKEN`**)
and writes `golden.f16.gz` + the `t<id>.f16.gz` inputs into each fixture dir, reading
each dir's `manifest.json` so it stays correct by construction. It runs in an
**ephemeral, isolated env** — it does NOT touch the global pip, `pyproject.toml`,
`uv.lock`, or the `ktir_emulator` package:

```bash
# from the repo root — regenerate ALL fixtures:
uv run --no-project \
  --with "transformers>=4.45" --with torch --with numpy --with safetensors \
  python rust/crates/ktir-emulator/tests/fixtures/gen_golden.py

# ...or only specific fixtures (e.g. after re-vendoring just smollm2):
uv run --no-project --with "transformers>=4.45" --with torch --with numpy --with safetensors \
  python rust/crates/ktir-emulator/tests/fixtures/gen_golden.py smollm2-135m smollm2-135m-prefill
```

Deterministic: fixed in-range token ids, f32 forward, results narrowed to f16.

## Format & why

- **f16 + gzip -9.** Spyre runs f16, so f32 goldens are needless precision (and the
  test's `max_abs < 1.0` band is far looser than f16). The constant tensors (zero
  KV-prefix, identity cos/sin, all-masked mask) compress to ~nothing. The Rust side
  reads these via `read_f16_gz` (flate2 + the f16 codec). dev-deps: `flate2`,
  `hf-hub`, `safetensors` (all already in the workspace lock).
- **Weights are NOT vendored** — fetched from public HF (`HuggingFaceTB/SmolLM2-135M`,
  `unsloth/Llama-3.2-1B-Instruct`; bf16 on disk → `codec::bf16_to_f16`), bound
  verbatim `[out,in]`.

## Running

```bash
cargo test -p ktir-emulator --test e2e_real_forward -- --test-threads=1
```
Prefill (m>1) gates on `cfg(metal)` (the fused [1,1] segments need the GPU/AMX
offload for full-M reconstruction). `cfg(metal)` is auto-on on macOS, so plain
`cargo test` runs everything there; on non-mac add `--features metal`.
