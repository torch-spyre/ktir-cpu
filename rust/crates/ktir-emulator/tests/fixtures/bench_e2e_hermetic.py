#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors. Apache-2.0.
#
"""HERMETIC end-to-end Python timing for the Python-vs-Rust ktir_cpu comparison.

Runs the WHOLE-MODEL Python interpreter (`ktir_cpu.KTIRInterpreter`, node-by-node)
on the SAME hermetic fixtures the Rust e2e uses — the program + runtime inputs are
vendored in `tests/fixtures/<fixture>.tar.gz`, and the real weights are fetched
from public HuggingFace (NO token) and bound verbatim `[out,in]`, EXACTLY as the
Rust harness does (`e2e_real_forward.rs` `build_args` / `Weights`). There is NO
dependency on the `~/.cache/cudaforge/ktir/<bundle>/` scratchy bundle.

This is the Python side of PERFORMANCE.md's "E2E whole-model" table; the Rust side
is `e2e_real_forward` `time_resident_*` (see PERFORMANCE.md → How to regenerate).

Run from the repo ROOT:

  uv run --with huggingface_hub python \
    rust/crates/ktir-emulator/tests/fixtures/bench_e2e_hermetic.py

Env: ITERS (timed passes, default 5; one warm-up excluded unless SKIP_WARMUP=1).
Args: fixture names to run (default: all four). E.g. `... bench_e2e_hermetic.py smollm2-135m`.
"""
import gzip
import json
import os
import struct
import sys
import tarfile
import tempfile
import time

import numpy as np

from ktir_cpu import KTIRInterpreter

# --- transpose-B layout shim -------------------------------------------------
# The Rust/NAX path stores every GEMM weight transpose-B ([n, k], contraction on
# B's LAST axis) and dispatches it via indexing_maps — the layout that keeps the
# weight contiguous for the tensor engine. ktir_cpu's `linalg.matmul` handler
# computes a PLAIN `A @ B` and ignores indexing_maps, so on these fixtures it
# shape-mismatches ([m,k] @ [n,k]). We re-register a layout-aware handler HERE (in
# the bench only — ktir_cpu itself is unmodified) that transposes B when its
# contraction axis is last. Detection is shape-driven (B's inner dim == A's k),
# which is exact for these programs (every GEMM is non-square: k != n). Same
# arithmetic as the Rust transpose-B GEMM, so the per-node timing is faithful.
from ktir_cpu.dialects.registry import register  # noqa: E402
from ktir_cpu.ir_types import Tile  # noqa: E402
from ktir_cpu.latency import LatencyCategory as LC  # noqa: E402


@register("linalg.matmul", latency_category=LC.COMPUTE_MATMUL)
def _linalg_matmul_layout(op, context, env):
    a = context.get_value(op.operands[0]).data  # A [m, k]
    tile_b = context.get_value(op.operands[1])
    b = tile_b.data
    k = a.shape[-1]
    # transpose-B: weight stored [n, k] (contraction on the last axis). Plain B is
    # [k, n] (b.shape[-2] == k); only flip when the contraction is B's last axis.
    if b.shape[-2] != k and b.shape[-1] == k:
        b = np.swapaxes(b, -1, -2)
    product = a @ b
    result = Tile(product, tile_b.dtype, product.shape)
    if len(op.operands) > 2:  # outs accumulator: result = C + A·B
        acc = context.get_value(op.operands[2])
        if isinstance(acc, Tile):
            result = Tile(acc.data + result.data, acc.dtype, acc.shape)
    return result
# -----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

# fixture -> public HF repo (no token). decode + prefill share a repo (same
# weights, different m); identical to gen_golden.py / e2e_real_forward.rs.
FIXTURES = [
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M"),
    ("smollm2-135m-prefill", "HuggingFaceTB/SmolLM2-135M"),
    ("llama-3.2-1b", "unsloth/Llama-3.2-1B-Instruct"),
    ("llama-3.2-1b-prefill", "unsloth/Llama-3.2-1B-Instruct"),
]


def read_f16_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype="<f2").copy()


def bf16_bytes_to_f16(raw: bytes) -> np.ndarray:
    """bf16 little-endian bytes -> f16 numpy. bf16 is the top 16 bits of f32, so
    widen to f32 (<<16) then narrow to f16 — matching `codec::bf16_to_f16`."""
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    f32 = (u16 << 16).view(np.float32)
    return f32.astype(np.float16)


def load_hf_weights(repo: str) -> dict:
    """Download a repo's safetensors (cached) and return {name: f16 flat numpy},
    converting bf16->f16 verbatim. Mirrors `Weights::fetch`: index.json if sharded,
    else a single `model.safetensors`. Parses the safetensors container by hand
    (8-byte header len + JSON header + raw bytes) so only `huggingface_hub` + numpy
    are needed (no torch / safetensors pkg, and numpy has no bf16 dtype)."""
    from huggingface_hub import hf_hub_download

    try:
        idx_path = hf_hub_download(repo, "model.safetensors.index.json")
        weight_map = json.load(open(idx_path))["weight_map"]
        shards = sorted(set(weight_map.values()))
    except Exception:
        shards = ["model.safetensors"]

    out = {}
    for shard in shards:
        path = hf_hub_download(repo, shard)
        with open(path, "rb") as f:
            blob = f.read()
        (hlen,) = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8 : 8 + hlen])
        base = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            raw = blob[base + start : base + end]
            dt = meta["dtype"]
            if dt == "BF16":
                out[name] = bf16_bytes_to_f16(raw)
            elif dt == "F16":
                out[name] = np.frombuffer(raw, dtype="<f2").copy()
            elif dt == "F32":
                out[name] = np.frombuffer(raw, dtype="<f4").astype(np.float16)
            else:
                raise ValueError(f"{name}: unsupported dtype {dt}")
    return out


def hf_weight(weights: dict, disk: str) -> np.ndarray:
    """Verbatim f16 for `<disk>.weight`; tied lm_head -> embed_tokens. Mirrors
    `Weights::weight`."""
    v = weights.get(f"{disk}.weight")
    if v is not None:
        return v
    if disk == "lm_head":
        return weights["model.embed_tokens.weight"]
    raise KeyError(f"HF weight {disk!r} not found in repo")


def fixture_dir(fixture: str) -> str:
    """Prefer an already-unpacked `<fixture>/` dir (gitignored, editable in place);
    else unpack `<fixture>.tar.gz` into a temp dir. Mirrors `fixture_dir` in Rust."""
    unpacked = os.path.join(HERE, fixture)
    if os.path.isfile(os.path.join(unpacked, "manifest.json")):
        return unpacked
    archive = os.path.join(HERE, f"{fixture}.tar.gz")
    if not os.path.isfile(archive):
        return ""
    dest = os.path.join(tempfile.gettempdir(), f"ktir-fixture-{fixture}")
    os.makedirs(dest, exist_ok=True)
    if not os.path.isfile(os.path.join(dest, "manifest.json")):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest)
    return dest


def synth(role: str, n: int) -> np.ndarray:
    if role == "cos":
        return np.ones(n, dtype=np.float16)
    if role == "sin":
        return np.zeros(n, dtype=np.float16)
    return (((np.arange(n) % 17) - 8) * 0.01).astype(np.float16)


def build_sources(fdir: str, man: dict, weights: dict) -> dict:
    """tensor id -> flat f16 source buffer, bound exactly like Rust `build_args`:
    weight sources from HF (verbatim [out,in]); runtime activations from the vendored
    t<id>.f16.gz (else a deterministic fallback)."""
    tn = {t["id"]: t for t in man["tensors"]}
    srcmeta = {s["id"]: s for s in man["sources"]}
    ids = [t["id"] for t in man["tensors"] if t.get("is_source")]
    mask_id = man.get("attn_mask")
    if mask_id is not None and mask_id not in ids:
        ids.append(mask_id)

    sources = {}
    for tid in ids:
        rows, cols = tn[tid]["rows"], tn[tid]["cols"]
        n = rows * cols
        disk = srcmeta.get(tid, {}).get("disk")
        if disk:  # weight source -> HF, verbatim
            w = hf_weight(weights, disk)
            assert w.size == n, f"weight {disk} (t{tid}) numel {w.size} != {n}"
            sources[tid] = w
            continue
        vendored = os.path.join(fdir, f"t{tid}.f16.gz")
        if os.path.isfile(vendored):
            v = read_f16_gz(vendored)
            assert v.size == n, f"vendored t{tid} len {v.size} != {n}"
            sources[tid] = v
        elif tid == mask_id:
            sources[tid] = np.full(n, -65504.0, dtype=np.float16)
        else:
            role = srcmeta.get(tid, {}).get("role", "weight")
            if role in ("prefix_k", "prefix_v"):
                sources[tid] = np.zeros(n, dtype=np.float16)
            else:
                sources[tid] = synth(role, n)
    return sources


def bench_one(fixture: str, repo: str, weights_cache: dict) -> None:
    fdir = fixture_dir(fixture)
    if not fdir:
        print(f"{fixture}: fixture archive absent — skipping")
        return
    man = json.load(open(os.path.join(fdir, "manifest.json")))
    tn = {t["id"]: t for t in man["tensors"]}

    if repo not in weights_cache:
        print(f"  loading HF weights {repo} (bf16->f16) ...", flush=True)
        weights_cache[repo] = load_hf_weights(repo)
    sources = build_sources(fdir, man, weights_cache[repo])

    nodes = man["nodes"]
    interps = {}
    for node in nodes:
        name = node["mlir"]
        if name not in interps:
            interp = KTIRInterpreter()
            interp.load(open(os.path.join(fdir, name)).read())
            interps[name] = interp

    def one_pass():
        buf = {tid: v.copy() for tid, v in sources.items()}
        for node in nodes:
            interp = interps[node["mlir"]]
            kwargs = {}
            outs = []
            for a in node["args"]:
                tid = a["tensor"]
                r, c, _ = tn[tid]["rows"], tn[tid]["cols"], None
                if a.get("is_output"):
                    data = np.zeros(r * c, dtype=np.float16)
                    outs.append((a["name"], tid))
                else:
                    data = buf[tid].astype(np.float16)
                kwargs[a["name"]] = data.reshape(r, c)
            result = interp.execute_function(node["fn"], **kwargs)
            for nm, tid in outs:
                buf[tid] = np.asarray(result[nm], dtype="<f2").reshape(-1)
        return buf

    iters = int(os.environ.get("ITERS", "5"))
    if not os.environ.get("SKIP_WARMUP"):
        one_pass()  # warm-up — excluded
    t0 = time.perf_counter()
    for _ in range(iters):
        one_pass()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"{fixture} e2e (Python per-node): {ms:.1f} ms/pass  ({len(nodes)} nodes, {iters} passes)")


def main():
    only = set(sys.argv[1:])
    fixtures = [(fx, repo) for fx, repo in FIXTURES if not only or fx in only]
    weights_cache = {}
    for fx, repo in fixtures:
        bench_one(fx, repo, weights_cache)


if __name__ == "__main__":
    sys.exit(main())
