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

"""Every kernel under ``examples/`` that this repository's simulator gates on.

Declaring a kernel is adding a row to ``ENTRIES`` at the bottom of this file: the
kernel's ``.mlir``, the function in it, the parameters to gate at, the arguments to
call it with, and a reference to compare its output against.  Nothing else — no new
file, no registration call, no test.

Two things are worth knowing before adding a row.

**The arguments are data.**  ``tensors`` is a mapping from the kernel's argument
names to specs — ``normal``, ``zeros``, ``full``, ``tile``, ``arange``,
``integers``, ``asarray``, ``param`` — resolved against ``gate_params``; a bare
string forwards a parameter unchanged, and a callable ``(params, rng)`` is the
escape hatch for input no spec expresses.  ``ktir_cpu/kernelentry/tensorspec.py``
holds the vocabulary and the reason each draw is seeded the way it is.

**A reference computes in f32 or wider, and does not fold the way the kernel
folds.**  Both halves matter.  Every kernel here holds its intermediates in f16,
so a reference evaluated in f16 reproduces the same overflow and then agrees with
the kernel about a wrong answer — which is what the end-to-end tests in
``tests/test_examples.py`` do when they cast a result back to f16 before comparing.
And a reference written in the kernel's own decomposition — summing shards the way
the shards are summed, carrying a running softmax denominator tile by tile —
checks the arithmetic while assuming the decomposition, which is usually the part
under test.  Write the short form of the answer.

``docs/kernelentry.md`` is the contributor's entry point; this file is where the
kernels are.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ktir_cpu.kernelentry import KernelEntry, register_entry
from ktir_cpu.kernelentry.tensorspec import (
    FAN_IN, arange, asarray, full, integers, normal, param, tile, zeros,
)

# ---------------------------------------------------------------------------
# Argument specs no vocabulary covers
# ---------------------------------------------------------------------------

def rope_cos(params: Dict[str, Any], rng) -> np.ndarray:
    """The RoPE cosine table, built in f64 before rounding to the kernel's f16.

    The angles are input, not part of what is checked: computed at f16 precision
    the reference and the kernel would rotate by slightly different amounts, and
    the difference would be charged to the kernel.
    """
    return np.cos(_rope_angles(params)).astype(np.float16)


def rope_sin(params: Dict[str, Any], rng) -> np.ndarray:
    """The RoPE sine table; see :func:`rope_cos`."""
    return np.sin(_rope_angles(params)).astype(np.float16)


def _rope_angles(params: Dict[str, Any]) -> np.ndarray:
    half = params["D"] // 2
    freqs = 10000.0 ** (-np.arange(half, dtype=np.float64) * 2.0 / params["D"])
    return np.outer(np.arange(params["S"], dtype=np.float64), freqs)


def ridge_row(params: Dict[str, Any], rng) -> np.ndarray:
    """A row whose maximum is in the middle, repeated down the rows.

    Ordered this way on purpose: on a monotonic row, a fold that returned its
    final input rather than the running maximum would agree with the reference.
    """
    cols = params["cols"]
    row = np.linspace(-2.0, 2.0, cols, dtype=np.float16)
    row[cols // 2:] = row[cols // 2:][::-1]
    return np.broadcast_to(row, (params["rows"], cols)).copy()


def padded_rows(params: Dict[str, Any], rng) -> np.ndarray:
    """Real values up to ``n_real_cols``, then ``-inf`` to the block width.

    The padding is the point of the full-size softmax entry: the lowering pads a
    row to the block width, and a kernel that included the padding in its
    denominator would still produce a plausible distribution.
    """
    rows, cols = params["n_rows"], params["n_cols"]
    real = params["n_real_cols"]
    out = np.full((rows, cols), -np.inf, dtype=np.float16)
    out[:, :real] = rng.standard_normal((rows, real)).astype(np.float16)
    return out


#: ``indexed_add``'s views, which are baked into its MLIR rather than derived
#: from its parameters.
INDEXED_ADD_X = (128, 64, 8, 128)
INDEXED_ADD_Y = (2, 32, 8, 128)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def matmul_reference(params: Dict[str, Any],
                     tensors: Dict[str, Any]) -> Dict[str, Any]:
    """``C = A @ B``, accumulated in f32 across all of K at once.

    The kernel folds K in f16 — sixteen accumulation steps at
    ``matmul_fwd_ktir``'s shape — and a reference folding the same way would
    reproduce the same rounding and then agree about a wrong answer.
    """
    a = np.asarray(tensors["a_ptr"], dtype=np.float32)
    b = np.asarray(tensors["b_ptr"], dtype=np.float32)
    return {"c_ptr": a @ b}


def softmax_reference(params: Dict[str, Any],
                      tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Row-wise softmax with the shift, the exponential and the sum all in f32.

    The kernel subtracts the row maximum before exponentiating, so what needs
    widening is not the exponential but the denominator: an f16 sum of values near
    1 loses the low bits of every term after the first few.  Any ``-inf`` padding
    is left in rather than sliced off — ``exp`` of it is exactly zero, so both
    sides are asked the same question about the padded columns.
    """
    x = np.asarray(tensors["input_ptr"], dtype=np.float32)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return {"output_ptr": e / e.sum(axis=1, keepdims=True)}


def vector_add_reference(params: Dict[str, Any],
                         tensors: Dict[str, Any]) -> Dict[str, Any]:
    x = np.asarray(tensors["x_ptr"], dtype=np.float64)
    y = np.asarray(tensors["y_ptr"], dtype=np.float64)
    return {"output_ptr": x + y}


def swiglu_reference(params: Dict[str, Any],
                     tensors: Dict[str, Any]) -> Dict[str, Any]:
    """``x + (silu(x @ W_gate) * (x @ W_up)) @ W_down``, the whole block in f32.

    Written against the unsharded expression, which is what makes it a check of
    ``ffn_swiglu_4core``'s sharding: a reference summing four 256-wide partials
    the way that kernel does would agree with it about a wrong split.
    """
    x = np.asarray(tensors["x_ptr"], dtype=np.float32)
    w_gate = np.asarray(tensors["w_gate_ptr"], dtype=np.float32)
    w_up = np.asarray(tensors["w_up_ptr"], dtype=np.float32)
    w_down = np.asarray(tensors["w_down_ptr"], dtype=np.float32)
    gate = x @ w_gate
    silu = gate / (1.0 + np.exp(-gate))
    return {"out_ptr": x + (silu * (x @ w_up)) @ w_down}


def row_sum_reference(params: Dict[str, Any],
                      tensors: Dict[str, Any]) -> Dict[str, Any]:
    """The row sum, broadcast back across the row, accumulated in f32."""
    data = np.asarray(tensors["arg0"], dtype=np.float32)
    return {"arg0": np.broadcast_to(
        data.sum(axis=1, keepdims=True), data.shape).copy()}


def row_max_reference(params: Dict[str, Any],
                      tensors: Dict[str, Any]) -> Dict[str, Any]:
    """The row maximum, broadcast back across the row."""
    data = np.asarray(tensors["arg0"], dtype=np.float32)
    return {"arg0": np.broadcast_to(
        data.max(axis=1, keepdims=True), data.shape).copy()}


def scalar_broadcast_reference(params: Dict[str, Any],
                               tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Exact rather than approximate: broadcasting a value is the one output
    where any difference at all is a defect."""
    return {"out_ptr": np.full((params["out_rows"], params["out_cols"]),
                               params["value"], dtype=np.float32)}


def rope_reference(params: Dict[str, Any],
                   tensors: Dict[str, Any]) -> Dict[str, Any]:
    """The half-layout rotation in f32, over the same tables the kernel was given.

        y[:, :D/2] = x[:, :D/2] * cos - x[:, D/2:] * sin
        y[:, D/2:] = x[:, :D/2] * sin + x[:, D/2:] * cos
    """
    H, S, D = params["H"], params["S"], params["D"]
    half = D // 2
    x = np.asarray(tensors["x_ptr"], dtype=np.float32).reshape(H, S, D)
    cos = np.asarray(tensors["cos_ptr"], dtype=np.float32)[np.newaxis]
    sin = np.asarray(tensors["sin_ptr"], dtype=np.float32)[np.newaxis]
    first, second = x[:, :, :half], x[:, :, half:]
    y = np.concatenate([first * cos - second * sin,
                        first * sin + second * cos], axis=-1)
    return {"out_ptr": y.reshape(H * S, D)}


def layernorm_reference(params: Dict[str, Any],
                        tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Mean, rstd and the normalised rows, all in f32.

    Widening matters more here than in a kernel with one output: the kernel folds
    both the sum and the sum of squares over 8192 f16 values, and a reference
    folding them the same way would agree with whatever those two reductions
    drifted to.  ``eps`` is read from the parameters rather than repeated, so
    changing it in the row changes both sides.
    """
    x = np.asarray(tensors["X"], dtype=np.float32)
    w = np.asarray(tensors["W"], dtype=np.float32)
    b = np.asarray(tensors["B"], dtype=np.float32)
    mean = x.mean(axis=1)
    rstd = 1.0 / np.sqrt(x.var(axis=1) + params["eps"])
    return {"Y": (x - mean[:, None]) * rstd[:, None] * w + b,
            "Mean": mean, "Rstd": rstd}


def indexed_add_reference(params: Dict[str, Any],
                          tensors: Dict[str, Any]) -> Dict[str, Any]:
    """The gather done in numpy, with the add in f32.

    The gather itself is exact — it moves f16 values without arithmetic — so the
    widening is only for the add.  Getting the *indices* wrong is the failure this
    reference is really for, and no tolerance hides it: two different rows of
    ``x`` are uncorrelated.
    """
    x = np.asarray(tensors["x_ptr"], dtype=np.float32)
    y = np.asarray(tensors["y_ptr"], dtype=np.float32)
    index = np.asarray(tensors["index_ptr"], dtype=np.intp)
    start = params["dim1_start"]
    rows = x[index][:, start:start + INDEXED_ADD_Y[1], :, :]
    return {"output_ptr": rows + y}


def sdpa_2d_reference(params: Dict[str, Any],
                      tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Attention with every intermediate in f32.

    Widening all of them rather than only the matmuls is deliberate: the softmax
    denominator sums 32 terms, and an f32 reference that folded that sum in f16
    would reproduce the kernel's rounding in the one place the two are most
    likely to differ.
    """
    q = np.asarray(tensors["q_ptr"], dtype=np.float32)
    k = np.asarray(tensors["k_ptr"], dtype=np.float32)
    v = np.asarray(tensors["v_ptr"], dtype=np.float32)
    scores = (q @ k.T) * np.float32(params["scale"])
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return {"output_ptr": (e / e.sum(axis=1, keepdims=True)) @ v}


def paged_attention_reference(params: Dict[str, Any],
                              tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Causal attention at every grid position, in f32 and untiled.

    Two differences from the reference in ``tests/test_examples.py``, both load
    bearing.  It covers all 32 grid positions rather than the first, because a
    mask indexed by query position is exactly what can be right at ``pid0 = 0``
    and wrong everywhere else.  And it reduces over the whole masked row at once
    instead of carrying a running maximum and denominator tile by tile: the tiled
    form is the kernel's own algorithm, so a reference written that way checks the
    arithmetic and assumes the decomposition.
    """
    q = np.asarray(tensors["query_ptr"], dtype=np.float32)
    k_cache = np.asarray(tensors["key_cache_ptr"], dtype=np.float32)
    v_cache = np.asarray(tensors["value_cache_ptr"], dtype=np.float32)
    table = np.asarray(tensors["block_tables_ptr"])
    scale = float(tensors["scale"])
    context_len = int(tensors["context_len"])
    num_tiles = int(tensors["num_tiles"])
    block_q = params["block_q"]
    per_kv = params["num_query_heads"] // params["num_kv_heads"]

    out = np.zeros_like(q)
    pages = table[0, :num_tiles]
    for pid1 in range(params["num_kv_heads"]):
        # (num_tiles * blk_size, head_size): the pages this KV head reads, in the
        # order block_tables gives them.
        keys = np.concatenate([k_cache[p, :, pid1, :] for p in pages])
        values = np.concatenate([v_cache[p, :, pid1, :] for p in pages])
        for pid0 in range(params["num_tokens"] // block_q):
            rows = slice(pid0 * block_q, (pid0 + 1) * block_q)
            heads = slice(pid1 * per_kv, (pid1 + 1) * per_kv)
            tile_q = q[rows, heads, :].reshape(block_q * per_kv, -1)
            scores = tile_q @ keys.T * scale
            # Query row r sits at absolute position context_len + pid0*block_q
            # + r // per_kv, and may not see past it.
            positions = context_len + pid0 * block_q + np.arange(
                block_q * per_kv) // per_kv
            scores[np.arange(keys.shape[0])[None, :] > positions[:, None]] = -np.inf
            p = np.exp(scores - scores.max(axis=1, keepdims=True))
            attended = (p @ values) / p.sum(axis=1, keepdims=True)
            out[rows, heads, :] = attended.reshape(block_q, per_kv, -1)
    return {"output_ptr": out}


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------

ENTRIES = [
    # --- examples/ktir: hand-written IR -----------------------------------
    KernelEntry(
        # Three matmuls and a sigmoid in one kernel: the longest chain of f16
        # intermediates here at a gate-sized shape. The weights are unit variance
        # and unscaled, which saturates the sigmoid on purpose — `gate` spans
        # [-24.3, +21.2], and 14 of its 128 f16 values leave f16 range in the
        # exponential: 11 overflow to inf, where 1 / inf makes sigmoid 0, and 3
        # underflow to exactly 0, where sigmoid is exactly 1. Both are the value
        # the f32 reference converges to there, so the output still agrees with
        # it. It is the only entry that exercises saturation, and
        # `ktir_cpu/ops/_helpers.py` reports the overflowing 11 as a
        # RuntimeWarning while it runs.
        name="ffn_swiglu", func="ffn_swiglu", path="ktir/ffn_swiglu.mlir",
        gate_params={"seq": 1, "d_model": 64, "d_ffn": 128},
        tensors={
            "x_ptr": normal(("seq", "d_model")),
            "w_gate_ptr": normal(("d_model", "d_ffn")),
            "w_up_ptr": normal(("d_model", "d_ffn")),
            "w_down_ptr": normal(("d_ffn", "d_model")),
            "out_ptr": zeros(("seq", "d_model")),
        },
        reference=swiglu_reference,
        # The one entry that declares its own tolerance, and the unscaled weights
        # are why. `silu * up` reaches 331 and the down projection sums 128 of
        # those, so every output element accumulates through a peak far above the
        # value it ends on — 183 at the smallest of the 64, 2095 at the largest.
        # A relative tolerance is charged against the value, not against the peak
        # the sum passed through, so an element that cancels escapes it: element
        # 30 comes down from a peak of 338 to -1.85, and two f16 rounding steps at
        # 338 put it 0.34 away, 19% off. atol has to cover that at whichever
        # element cancels, so it is set from the largest peak in the tensor rather
        # than fitted to the one that cancels here: 2 * 2095 * 2^-11 = 2.05.
        # Only 2 of the 64 elements need it at all and the worst needs 0.31, so
        # the declared pair sits ~6x above what this seed asks for — deliberately,
        # because a shape change re-rolls the tensors and moves which element
        # cancels. It is a correctness-leg number only: the cost leg reads the
        # same run, and cycle counts there are structural rather than
        # data-dependent, so no tolerance of any width moves them.
        tolerance={"out_ptr": (2e-2, 2.0)},
    ),
    KernelEntry(
        # The same block with the hidden dimension sharded over 4 cores: `x` is
        # replicated, each core owns a 256-wide slice of W_gate / W_up and the
        # matching 256-row slice of W_down, and the four [4, 256] partials fold
        # through inter_tile_produce / inter_tile_reduce before the residual.
        # Weights scaled by fan-in, unlike the single-core entry: `gate` widens
        # with sqrt(d_model), so unscaled at 256 it would have std 16.8 against
        # the single-core entry's 8.2 and 2069 of its 4096 values would saturate
        # instead of 22 of 128. A comparison mostly between saturated values
        # would stop being about whether the shards folded.
        name="ffn_swiglu_4core", func="ffn_swiglu_4core",
        path="ktir/ffn_swiglu_4core.mlir",
        gate_params={"seq": 4, "d_model": 256, "d_ffn": 1024},
        tensors={
            "x_ptr": normal(("seq", "d_model")),
            "w_gate_ptr": normal(("d_model", "d_ffn"), scale=FAN_IN),
            "w_up_ptr": normal(("d_model", "d_ffn"), scale=FAN_IN),
            "w_down_ptr": normal(("d_ffn", "d_model"), scale=FAN_IN),
            "out_ptr": zeros(("seq", "d_model")),
        },
        reference=swiglu_reference,
    ),
    KernelEntry(
        # A linalg.reduce combiner written as an explicit (%in, %out) region
        # rather than the { arith.addf } shorthand, which is the form the Triton
        # Spyre ConvertTTReduce pass emits. One argument, input and output both,
        # so the sum is written back over the data it came from — which is why the
        # reference is handed a pristine rebuild rather than what the run left.
        name="reduce_generic", func="reduce_explicit_region",
        path="ktir/reduce_generic.mlir",
        gate_params={"rows": 1, "cols": 4},
        tensors={"arg0": arange(("rows", "cols"), start=1)},
        reference=row_sum_reference,
    ),
    KernelEntry(
        # `max` written as arith.cmpf ogt + arith.select rather than a single
        # arith.maximumf, so the result cannot come from recognising a combiner op
        # by name: every op in the region has to run.
        name="reduce_multiop", func="reduce_multiop",
        path="ktir/reduce_multiop.mlir",
        gate_params={"rows": 1, "cols": 8},
        tensors={"arg0": ridge_row},
        reference=row_max_reference,
    ),
    KernelEntry(
        # Collapses a tensor<1x1xf16> to a scalar tensor<f16> and broadcasts it,
        # which is the shape the Triton -> KTIR lowering emits for a 1x1
        # broadcast.
        name="scalar_broadcast", func="scalar_broadcast",
        path="ktir/scalar_broadcast.mlir",
        gate_params={"value": 2.5, "out_rows": 4, "out_cols": 64},
        tensors={
            "in_ptr": full((1, 1), "value"),
            "out_ptr": zeros(("out_rows", "out_cols")),
        },
        reference=scalar_broadcast_reference,
    ),

    # --- examples/latency: reduced shapes, which is why they are the gate ---
    KernelEntry(
        # A tiled matmul on a [2, 2] grid. Every parameter here is baked into the
        # MLIR — view sizes, grid, tile constants — so they are parameters of this
        # row rather than of the kernel: changing one does not change the kernel,
        # it makes this row disagree with it, which out.c_ptr.reference reports.
        name="matmul_small", func="matmul_kernel_small",
        path="latency/matmul_small.mlir",
        gate_params={"M": 16, "N": 64, "K": 64,
                     "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
        tensors={
            "a_ptr": normal(("M", "K")),
            "b_ptr": normal(("K", "N")),
            "c_ptr": zeros(("M", "N")),
            "K": "K",
            "BLOCK_SIZE_M": "BLOCK_SIZE_M",
            "BLOCK_SIZE_N": "BLOCK_SIZE_N",
            "BLOCK_SIZE_K": "BLOCK_SIZE_K",
        },
        reference=matmul_reference,
    ),
    KernelEntry(
        # Half-layout RoPE on a [4, 2] grid at LLaMA-8B / Granite-8B shapes: 40
        # heads, 4096 positions, D=128, flattened to [H*S, D]. The cos/sin tables
        # are inputs rather than constants, so the kernel is checked against the
        # rotation it was asked for and not against whatever tables it read.
        name="rope_fwd_4x2", func="rope_fwd_kernel",
        path="latency/rope_fwd_4x2.mlir",
        gate_params={"H": 40, "S": 4096, "D": 128},
        tensors={
            "x_ptr": normal((lambda p: p["H"] * p["S"], "D")),
            "cos_ptr": rope_cos,
            "sin_ptr": rope_sin,
            "out_ptr": zeros((lambda p: p["H"] * p["S"], "D")),
        },
        reference=rope_reference,
    ),
    KernelEntry(
        # softmax_fwd_ktir.mlir at reduced size, which is what makes it a gate
        # kernel: the same row-wise max / exp / sum / divide chain over 64 rows
        # instead of 4096. tests/test_latency.py already drives it for its cost
        # breakdown; this row is what says those figures came from a kernel
        # computing the right thing.
        name="softmax_small", func="softmax_kernel_small",
        path="latency/softmax_small.mlir",
        gate_params={"n_rows": 64, "n_cols": 64},
        tensors={
            "output_ptr": zeros(("n_rows", "n_cols")),
            "input_ptr": normal(("n_rows", "n_cols")),
            "n_rows": "n_rows",
        },
        reference=softmax_reference,
    ),
    KernelEntry(
        # The same kernel with each linalg.reduce combiner written as an explicit
        # region instead of the shorthand. Same inputs and same reference as the
        # row above, which is the whole point of the second file: two spellings
        # fed different inputs could be reported as agreeing without ever
        # computing the same thing.
        name="softmax_small_explicit", func="softmax_kernel_small_explicit",
        path="latency/softmax_small_explicit.mlir",
        gate_params={"n_rows": 64, "n_cols": 64},
        tensors={
            "output_ptr": zeros(("n_rows", "n_cols")),
            "input_ptr": normal(("n_rows", "n_cols")),
            "n_rows": "n_rows",
        },
        reference=softmax_reference,
    ),

    # --- examples/sdsc: decode attention, split across cores ---------------
    KernelEntry(
        # C = A @ B for (1 x 8192) @ (8192 x 128): the output split x2 and the KV
        # contraction split x16, so partial sums fold across cores in two strided
        # reduce groups. The entry that exercises what a single-core kernel cannot
        # — comm bytes in the derivation, and a reference that has to tolerate an
        # f16 fold, since nothing in the kernel is wider than f16 and each of the
        # 16 fold steps rounds its running sum back.
        name="sdpa_pv_ksplit", func="sdpa_pv_ksplit",
        path="sdsc/sdpa_pv_ksplit.mlir",
        gate_params={"M": 1, "N": 128, "K": 8192},
        tensors={
            "a_ptr": normal(("M", "K")),
            "b_ptr": normal(("K", "N")),
            "c_ptr": zeros(("M", "N")),
        },
        reference=matmul_reference,
    ),

    # --- examples/triton-ktir: captured Triton -> KTIR output --------------
    KernelEntry(
        # An indirect gather on the leading axis: the row of `x` each core reads
        # comes out of an i64 index tensor at run time, so this is the one
        # declared kernel whose addresses are not a function of its grid
        # position. It reaches ktdp.construct_indirect_access_tile.
        name="indexed_add", func="indexed_add_kernel",
        path="triton-ktir/indexed_add.mlir",
        gate_params={"dim1_start": 0, "index": (3, 7)},
        tensors={
            "x_ptr": normal(INDEXED_ADD_X),
            "y_ptr": normal(INDEXED_ADD_Y),
            "index_ptr": asarray("index", "i64"),
            "output_ptr": zeros(INDEXED_ADD_Y),
            "dim1_start": "dim1_start",
        },
        reference=indexed_add_reference,
    ),
    KernelEntry(
        # Layer norm at full size, and the one kernel here with three outputs:
        # the normalised rows, plus the per-row mean and reciprocal standard
        # deviation the backward pass would consume. All three are referenced,
        # which tests/test_examples.py does not do for Rstd — and Rstd is the
        # interesting one, because it is where a reduction over 8192 f16 elements
        # either survives a division and a square root or does not.
        #
        # W and B are a weight and bias vector, but the MLIR views them at
        # [n_rows, n_cols] — the same row read once per row of X rather than
        # broadcast — so they are declared at that shape, built by tiling one row.
        # The row count is deliberately not a multiple of the 32-core grid:
        # 1151 = 35*32 + 31, so the last core takes a short tail.
        name="layernorm_fwd_ktir", func="_layer_norm_fwd_fused",
        path="triton-ktir/layernorm_fwd_ktir.mlir",
        gate_params={"n_rows": 1151, "n_cols": 8192, "eps": 1e-5,
                     "BLOCK_SIZE": 1024},
        tensors={
            "X": normal(("n_rows", "n_cols")),
            "Y": zeros(("n_rows", "n_cols")),
            "W": tile(normal("n_cols"), ("n_rows", 1)),
            "B": tile(normal("n_cols"), ("n_rows", 1)),
            "Mean": zeros("n_rows"),
            "Rstd": zeros("n_rows"),
            "N": "n_cols",
            "eps": "eps",
            "BLOCK_SIZE": "BLOCK_SIZE",
        },
        reference=layernorm_reference,
    ),
    KernelEntry(
        # Split-K matmul at full size. matmul_small is the same kernel shape at
        # [16, 64, 64]; what only appears here is the depth of the accumulation —
        # K=2048 over BLOCK_SIZE_K=128 is 16 f16 accumulation steps, and the
        # rounding that produces is proportional to the output magnitude rather
        # than bounded by it.
        name="matmul_fwd_ktir", func="matmul_kernel",
        path="triton-ktir/matmul_fwd_ktir.mlir",
        gate_params={"M": 64, "N": 8192, "K": 2048,
                     "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 512,
                     "BLOCK_SIZE_K": 128},
        tensors={
            "a_ptr": normal(("M", "K")),
            "b_ptr": normal(("K", "N")),
            "c_ptr": zeros(("M", "N")),
            "K": "K",
            "BLOCK_SIZE_M": "BLOCK_SIZE_M",
            "BLOCK_SIZE_N": "BLOCK_SIZE_N",
            "BLOCK_SIZE_K": "BLOCK_SIZE_K",
        },
        reference=matmul_reference,
    ),
    KernelEntry(
        # Unified attention over a paged KV cache: block_tables names which of the
        # 64 cache pages each step reads, so the kernel's traffic is decided at
        # run time by data rather than by its loop bounds. The page table is drawn
        # over the whole cache rather than set to identity, so the pages a step
        # reads are not the pages a direct view would have reached — which is what
        # makes the indirect path observable in the output at all.
        name="paged_attention", func="kernel_unified_attention_spyre_2d",
        path="triton-ktir/paged_attention.mlir",
        gate_params={
            "num_tokens": 8, "num_query_heads": 32, "num_kv_heads": 8,
            "head_size": 128, "num_blks": 64, "blk_size": 16,
            "max_num_blocks_per_seq": 16, "block_q": 2, "num_tiles": 8,
            "context_len": 120,
        },
        tensors={
            "output_ptr": zeros(("num_tokens", "num_query_heads", "head_size")),
            "query_ptr": normal(("num_tokens", "num_query_heads", "head_size")),
            "key_cache_ptr": normal(
                ("num_blks", "blk_size", "num_kv_heads", "head_size")),
            "value_cache_ptr": normal(
                ("num_blks", "blk_size", "num_kv_heads", "head_size")),
            "block_tables_ptr": integers((1, "max_num_blocks_per_seq"),
                                         high="num_blks"),
            "cur_batch_start_index": 0,
            "block_table_offset": 0,
            "num_tiles": "num_tiles",
            "context_len": "context_len",
            "scale": lambda p, rng: 1.0 / math.sqrt(p["head_size"]),
        },
        reference=paged_attention_reference,
    ),
    KernelEntry(
        # softmax(Q @ K^T * scale) @ V on one core, so the whole attention chain —
        # two matmuls with a row-wise max, exp and sum between them — runs with no
        # cross-core fold. The scale is a constant in the MLIR, which is why it is
        # a parameter of this row: changing it here makes the row disagree with
        # the kernel.
        name="sdpa_2d", func="sdpa_kernel_2d",
        path="triton-ktir/sdpa_2d.mlir",
        gate_params={"n_rows": 32, "head_dim": 64, "scale": 0.125},
        tensors={
            "q_ptr": normal(("n_rows", "head_dim")),
            "k_ptr": normal(("n_rows", "head_dim")),
            "v_ptr": normal(("n_rows", "head_dim")),
            "output_ptr": zeros(("n_rows", "head_dim")),
        },
        reference=sdpa_2d_reference,
    ),
    KernelEntry(
        # Row-wise softmax at full size. softmax_small is the same chain at
        # [64, 64] and is the one the gate leans on; this row exists because the
        # padding is only real here — see padded_rows.
        name="softmax_fwd_ktir", func="softmax_kernel",
        path="triton-ktir/softmax_fwd_ktir.mlir",
        gate_params={"n_rows": 4096, "n_cols": 1024, "n_real_cols": 778},
        tensors={
            "output_ptr": zeros(("n_rows", "n_cols")),
            "input_ptr": padded_rows,
            "n_rows": "n_rows",
        },
        reference=softmax_reference,
    ),
    KernelEntry(
        # A symbolic extent: the views are memref<?xf32> and the access tile's
        # coordinate set covers d0 in [0, 1023], so n_elements masks the tail at
        # run time rather than being baked in. Gated at 1024, the value at which
        # the mask is a no-op — masking is what the smaller extents in
        # tests/test_examples.py exercise, and this ledger's question is whether
        # the kernel computes the right thing, not how many extents it does so at.
        name="vector_add_dynamic_ktir", func="add_kernel_dynamic",
        path="triton-ktir/vector_add_dynamic_ktir.mlir",
        gate_params={"n_elements": 1024},
        tensors={
            "x_ptr": normal("n_elements", "f32"),
            "y_ptr": normal("n_elements", "f32"),
            "output_ptr": zeros("n_elements", "f32"),
            # Read as an i32 function argument, not as an index constant.
            "n_elements": param("n_elements", "i32"),
        },
        reference=vector_add_reference,
    ),
    KernelEntry(
        # The smallest kernel on this path: one load per input, one linalg.add,
        # one store. Its value here is as the floor — anything this ledger reports
        # about a larger kernel it also reports about this one.
        name="vector_add_ktir", func="add_kernel",
        path="triton-ktir/vector_add_ktir.mlir",
        gate_params={"n_elements": 4096, "BLOCK_SIZE": 128},
        tensors={
            "x_ptr": normal("n_elements"),
            "y_ptr": normal("n_elements"),
            "output_ptr": zeros("n_elements"),
            "BLOCK_SIZE": "BLOCK_SIZE",
        },
        reference=vector_add_reference,
    ),
]

for _entry in ENTRIES:
    register_entry(_entry)
