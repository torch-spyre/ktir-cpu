#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors. Apache-2.0.
#
# Repeatable generator for the REAL-MODEL e2e goldens (tests/e2e_real_forward.rs).
#
# Runs a real HuggingFace model forward (public repos, NO HF_TOKEN) and writes,
# into each vendored fixture dir, the GOLDEN logits + the runtime-input tensors the
# KTIR program needs to reproduce that forward (input activation, RoPE cos/sin,
# zero KV-prefix, padding mask). Weights are NOT written — the Rust test fetches
# them from HF too. Everything is little-endian f16, row-major, headerless, then
# gzip -9 (t<id>.f16.gz / golden.f16.gz) — Spyre is f16, so f32 is needless and the
# constant tensors compress to ~nothing. The Rust side reads them via read_f16_gz.
#
# RUN IT (ephemeral, isolated env — does NOT touch global pip / pyproject / uv.lock
# / the ktir_emulator package):
#
#   uv run --no-project \
#     --with "transformers>=4.45" --with torch --with numpy --with safetensors \
#     python rust/crates/ktir-emulator/tests/fixtures/gen_golden.py
#
# Deterministic: fixed token ids, f32 forward. Re-run after the vendored program
# changes (it reads each fixture's manifest, so it stays correct by construction).

import gzip
import json
import os
import struct
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))

# fixture dir -> public HF repo (no token). decode + prefill share a repo; the
# program differs (m), the weights are identical.
FIXTURES = [
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M"),
    ("smollm2-135m-prefill", "HuggingFaceTB/SmolLM2-135M"),
    ("llama-3.2-1b", "unsloth/Llama-3.2-1B-Instruct"),
    ("llama-3.2-1b-prefill", "unsloth/Llama-3.2-1B-Instruct"),
]

# Fixed, in-range, deterministic token ids. decode uses [:1], prefill uses [:m];
# decode's token == prefill's first token (so decode logits == prefill row 0).
def token_ids(vocab: int, m: int) -> list[int]:
    return [(1000 + 137 * i) % (vocab - 1) + 1 for i in range(m)]


# Goldens + runtime inputs are stored as f16 (Spyre's dtype — the program runs f16,
# so f32 is needless precision and the Rust `max_abs` band is far looser than f16),
# gzip -9 compressed (the constant tensors — zero KV-prefix, identity cos/sin, the
# all-masked attn mask — shrink to ~nothing; logits compress modestly). The Rust
# side reads these via `read_f16_gz`.
def write_f16_gz(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype="<f2")
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(arr.tobytes())


def load_manifest(fdir: str) -> dict:
    with open(os.path.join(fdir, "manifest.json")) as f:
        return json.load(f)


def gen_one(fixture: str, repo: str, model, tok_cache: dict):
    fdir = os.path.join(HERE, fixture)
    man = load_manifest(fdir)
    tn = {t["id"]: t for t in man["tensors"]}
    srcs = {s["id"]: s for s in man["sources"]}

    embed_id = next(i for i, s in srcs.items() if s.get("role") == "embed")
    m, d = tn[embed_id]["rows"], tn[embed_id]["cols"]
    cfg = model.config
    vocab = cfg.vocab_size
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    ids = token_ids(vocab, m)
    input_ids = torch.tensor([ids], dtype=torch.long)
    position_ids = torch.arange(m, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)            # [1, m, d]
        cos, sin = model.model.rotary_emb(hidden, position_ids)  # [1, m, head_dim]
        logits = model(input_ids).logits                         # [1, m, vocab]

    embed = hidden[0].float().numpy()                # [m, d]
    cos_np = cos[0].float().numpy()                  # [m, head_dim]
    sin_np = sin[0].float().numpy()
    golden = logits[0].float().numpy()               # [m, vocab]

    assert embed.shape == (m, d), (embed.shape, (m, d))
    assert cos_np.shape == (m, head_dim), (cos_np.shape, (m, head_dim))
    assert golden.shape == (m, vocab), (golden.shape, (m, vocab))

    # --- write the runtime inputs, keyed by KTIR tensor id (f16 + gzip) ---
    write_f16_gz(os.path.join(fdir, f"t{embed_id}.f16.gz"), embed)
    for i, s in srcs.items():
        role = s.get("role")
        rows, cols = tn[i]["rows"], tn[i]["cols"]
        if role == "cos":
            assert (rows, cols) == (m, head_dim), (i, rows, cols)
            write_f16_gz(os.path.join(fdir, f"t{i}.f16.gz"), cos_np)
        elif role == "sin":
            write_f16_gz(os.path.join(fdir, f"t{i}.f16.gz"), sin_np)
        elif role in ("prefix_k", "prefix_v"):
            # fresh forward: empty KV cache. HF golden is past_key_values=None to match.
            write_f16_gz(os.path.join(fdir, f"t{i}.f16.gz"), np.zeros((rows, cols), "<f4"))

    # The additive mask applies to the 64-position prefix KV CACHE, which is empty
    # for a fresh forward (current tokens enter via a separate fresh K/V path). Mask
    # ALL prefix positions (f16-min) so attention uses only the fresh tokens.
    mask_id = man["attn_mask"]
    mrows, mcols = tn[mask_id]["rows"], tn[mask_id]["cols"]
    mask = np.full((mrows, mcols), -65504.0, dtype="<f4")
    write_f16_gz(os.path.join(fdir, f"t{mask_id}.f16.gz"), mask)

    write_f16_gz(os.path.join(fdir, "golden.f16.gz"), golden)

    print(
        f"  {fixture}: m={m} d={d} head_dim={head_dim} vocab={vocab} "
        f"rope_theta={getattr(cfg,'rope_theta',None)} -> golden[{m},{vocab}], "
        f"inputs t{embed_id}/t cos/sin/prefix*/mask written"
    )


def main():
    # Optional CLI filter: `python gen_golden.py llama-3.2-1b llama-3.2-1b-prefill`
    # regenerates only those fixtures (default: all).
    only = set(sys.argv[1:])
    fixtures = [(fx, repo) for fx, repo in FIXTURES if not only or fx in only]
    by_repo: dict[str, list[str]] = {}
    for fx, repo in fixtures:
        by_repo.setdefault(repo, []).append(fx)
    for repo, fxs in by_repo.items():
        if not all(os.path.isdir(os.path.join(HERE, fx)) for fx in fxs):
            print(f"skip {repo}: fixture dir(s) absent")
            continue
        print(f"loading {repo} (f32, eager) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float32, attn_implementation="eager"
        )
        model.eval()
        for fx in fxs:
            gen_one(fx, repo, model, {})
        del model


if __name__ == "__main__":
    sys.exit(main())
