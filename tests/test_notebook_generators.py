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

#!/usr/bin/env python3
"""Tests for the parameterised MLIR generators used by the latency demo notebook.

The generators in ``notebooks/demo_gen_mlir.py`` had no test coverage, so a
mis-shaped grid or an f16 overflow could only be caught by reading the notebook
output.  This module covers ``gen_sdpa_decode_pv_mlir`` along the four axes the
kernel is used for: correctness against numpy, the roofline coordinates the
notebook plots, the cross-core reduce, and the generator's own preconditions.

The latency assertions are exact rather than bounds.  Every per-core cycle count
in this kernel closes by hand:

    compute = 2*m*n*k / systolic  +  (fan_in - 1) * m*n / simd
    memory  = (m*k + k*n) * 2 / (chip_bw / cores)      (+ the writer's store)
    comm    = (cores - 1) * m*n*2 / ring_bw

with ``m = q_per_kv``, ``n = head_dim / out_split``, ``k = kv_len / k_split``.
Asserting the closed form rather than a range means a change in how the model
prices the fold shows up here instead of silently moving the notebook's plot.
"""

import os
import sys

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from ktir_cpu.latency import HardwareConfig

# The notebook helpers are not an installed package.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "notebooks")
)

from demo_gen_mlir import gen_sdpa_decode_pv_mlir  # noqa: E402

FUNC = "sdpa_decode_pv"

# The notebook's own hardware configuration, so these numbers are the ones the
# notebook plots.
HW = HardwareConfig(lx_size_mb=2, hbm_bandwidth_tb_s=1.024)
SIMD = HW.simd_elements_per_cycle
SYSTOLIC = HW.systolic_flops_per_cycle
CHIP_B_PER_CYCLE = HW.hbm_bw_chip / HW.clock_hz
RING_B_PER_CYCLE = HW.ring_bytes_per_cycle

# Granite-8B decode: 32 query heads over 8 key/value heads at head_dim 128, so
# one key/value head is shared by 4 query heads.
Q_PER_KV = 4
HEAD_DIM = 128
KV_LEN = 8192

# Any fixed value: these tests need reproducible inputs, not particular ones.
SEED = 42

# The three configurations the notebook plots: baseline, strong scaling (same
# problem on 8x the cores) and weak scaling (8x the problem on 8x the cores, so
# the per-core contraction slice is the same 4096 tokens as the baseline).
CONFIGS = [
    pytest.param(KV_LEN, 2, 2, id="4-core"),
    pytest.param(KV_LEN, 2, 16, id="strong-32"),
    pytest.param(KV_LEN * 8, 2, 16, id="weak-32"),
]


def slices(kv_len, head_dim, q_per_kv, out_split, k_split):
    """(m, n, k) per-core tile extents."""
    return q_per_kv, head_dim // out_split, kv_len // k_split


def chip_dram_bytes(report):
    """Total HBM bytes charged across every core."""
    return sum(counter.bytes_by_category.get("memory", 0)
               for counter in report.counters.values())


class NotebookInterpMixin:
    """Provides _make_interp() so a subclass can inject another parser backend."""

    def _make_interp(self):
        return KTIRInterpreter(latency_config=HW)


class _SdpaDecodePvBase(NotebookInterpMixin):
    """Shared runner.  Argument names come from the parser, never hardcoded."""

    def run(self, kv_len=KV_LEN, head_dim=HEAD_DIM, q_per_kv=Q_PER_KV,
            out_split=2, k_split=16, scale=1.0):
        interp = self._make_interp()
        interp.load(gen_sdpa_decode_pv_mlir(kv_len, head_dim, q_per_kv,
                                            out_split, k_split))
        a_name, b_name, c_name = interp.arg_names(FUNC)
        rng = np.random.default_rng(SEED)
        a = (rng.standard_normal((q_per_kv, kv_len)) * scale).astype(np.float16)
        b = (rng.standard_normal((kv_len, head_dim)) * scale).astype(np.float16)
        c = np.zeros((q_per_kv, head_dim), dtype=np.float16)
        outputs = interp.execute_function(
            FUNC, **{a_name: a, b_name: b, c_name: c}
        )
        return a, b, np.asarray(outputs[c_name]), interp.get_latency_report()


class TestSdpaDecodePvExecution(_SdpaDecodePvBase):
    """Correctness of the decode P@V kernel, including the cross-core fold."""

    @pytest.mark.parametrize("kv_len,out_split,k_split", CONFIGS)
    def test_matches_numpy_reference(self, kv_len, out_split, k_split):
        """C == A @ B, within the tolerance the f16 fold allows.

        Nothing in the kernel accumulates in f32 -- not the matmul ``outs`` and
        not the reduce's combiner region -- so the running sum rounds at each of
        the ``k_split`` fold steps.  The error is proportional to the output
        magnitude, so rtol carries it and atol only covers near-zero outputs.
        """
        a, b, got, _ = self.run(kv_len=kv_len, out_split=out_split,
                                k_split=k_split)
        want = a.astype(np.float32) @ b.astype(np.float32)
        np.testing.assert_allclose(got.astype(np.float32), want,
                                   rtol=3e-2, atol=1.5)

    def test_zero_input_gives_zero_output(self):
        """An all-zero V must produce an all-zero result, fold included.

        Catches an identity leaking into the sum: the reduce op requires an
        identity operand, but the producer set covers every core of the group, so
        no core should ever seed from it.
        """
        interp = self._make_interp()
        interp.load(gen_sdpa_decode_pv_mlir(KV_LEN, HEAD_DIM, Q_PER_KV, 2, 16))
        a_name, b_name, c_name = interp.arg_names(FUNC)
        rng = np.random.default_rng(0)
        outputs = interp.execute_function(FUNC, **{
            a_name: rng.standard_normal((Q_PER_KV, KV_LEN)).astype(np.float16),
            b_name: np.zeros((KV_LEN, HEAD_DIM), dtype=np.float16),
            c_name: np.zeros((Q_PER_KV, HEAD_DIM), dtype=np.float16),
        })
        assert not np.any(np.asarray(outputs[c_name]))

    def test_split_contraction_matches_unsplit(self):
        """Folding 16 partial sums agrees with computing the whole product locally.

        ``k_split = 1`` takes the no-fold path, so this compares the two code
        paths the generator can emit on identical inputs and is the only test
        that isolates the fold from the matmul.
        """
        _, _, folded, _ = self.run(out_split=2, k_split=16)
        _, _, whole, _ = self.run(out_split=2, k_split=1)
        np.testing.assert_allclose(folded.astype(np.float32),
                                   whole.astype(np.float32),
                                   rtol=3e-2, atol=1.5)

    def test_large_inputs_overflow_f16(self):
        """The generator's documented input-magnitude limit actually bites.

        Nothing accumulates in f32, so the running sum over ``kv_len`` terms
        grows like ``std**2 * sqrt(kv_len)`` and leaves f16 range once the inputs
        are scaled up -- 8192 terms need only ``std`` around 27.  Pinned because
        the limit is a precondition the notebook relies on when it picks its input
        scale, and an overflow surfaces as ``inf``, not as an error.
        """
        _, _, unit, _ = self.run(scale=1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            _, _, large, _ = self.run(scale=32.0)
        assert np.all(np.isfinite(unit.astype(np.float32)))
        assert not np.all(np.isfinite(large.astype(np.float32)))

class TestSdpaDecodePvRoofline(_SdpaDecodePvBase):
    """The roofline coordinates the notebook plots, asserted against closed forms."""

    @pytest.mark.parametrize("kv_len,out_split,k_split", CONFIGS)
    def test_compute_cycles_match_closed_form(self, kv_len, out_split, k_split):
        """matmul on the systolic array plus the fold's combiner on SIMD.

        There is no term for the fold's writer guard: the model resolves scalar
        integer work at compile time and charges it nothing, so the guard is free
        however many cores the fold spans.
        """
        m, n, k = slices(kv_len, HEAD_DIM, Q_PER_KV, out_split, k_split)
        expected = 2 * m * n * k / SYSTOLIC + (k_split - 1) * m * n / SIMD
        per_core = self.run(kv_len=kv_len, out_split=out_split,
                            k_split=k_split)[3].per_core_summary()
        assert {row["compute_cycles"] for row in per_core} == {expected}

    @pytest.mark.parametrize("kv_len,out_split,k_split", CONFIGS)
    def test_memory_cycles_match_closed_form(self, kv_len, out_split, k_split):
        """Per-core bytes over the per-core share of chip bandwidth.

        Bandwidth is split evenly across active cores, so the cores that also
        store their group's output slice pay exactly that store on top.
        """
        m, n, k = slices(kv_len, HEAD_DIM, Q_PER_KV, out_split, k_split)
        bw = CHIP_B_PER_CYCLE / (out_split * k_split)
        loads = (m * k + k * n) * 2
        store = m * n * 2
        per_core = [row["memory_cycles"]
                    for row in self.run(kv_len=kv_len, out_split=out_split,
                                        k_split=k_split)[3].per_core_summary()]
        assert min(per_core) == loads / bw
        assert max(per_core) == (loads + store) / bw

    @pytest.mark.parametrize("kv_len,out_split,k_split", CONFIGS)
    def test_memory_bound(self, kv_len, out_split, k_split):
        """Decode P@V is memory-bound at every configuration the notebook plots.

        Unqualified, unlike a claim about a single core count: the kernel's
        intensity is far below the systolic ridge at both 4 and 32 cores.
        """
        report = self.run(kv_len=kv_len, out_split=out_split,
                          k_split=k_split)[3]
        core = report.core_roofline()
        assert report.bottleneck == "memory"
        assert core["core_dominant_unit"] == "systolic"
        assert core["core_AI"] < core["units"]["systolic"]["ridge"]

    @pytest.mark.parametrize("kv_len,out_split,k_split", CONFIGS)
    def test_arithmetic_intensity_near_harmonic_mean(self, kv_len, out_split,
                                                    k_split):
        """AI ~= harmonic mean of the two tile widths.

        ``2*m*n*k / ((m*k + k*n) * 2) = m*n / (m + n)``: the contraction extent
        cancels, which is why splitting the contraction harder does not change
        the intensity.  Approximate, not exact -- the fold's own flops and the
        writer's store are not in the closed form.
        """
        m, n, _ = slices(kv_len, HEAD_DIM, Q_PER_KV, out_split, k_split)
        harmonic = m * n / (m + n)
        measured = self.run(kv_len=kv_len, out_split=out_split,
                            k_split=k_split)[3].core_roofline()["core_AI"]
        assert measured == pytest.approx(harmonic, rel=0.05)

    def test_intensity_barely_moves_with_contraction_length(self):
        """8x the KV length leaves the roofline x-coordinate where it was.

        This is why the three configurations stack in one column of the plot:
        the intensity is a property of the tile widths, not of the problem size
        or the core count.  The residual drift is the fold's fixed flops being
        amortised over a longer matmul.
        """
        short = self.run(kv_len=KV_LEN, k_split=16)[3].core_roofline()["core_AI"]
        long_ = self.run(kv_len=KV_LEN * 8, k_split=16)[3].core_roofline()["core_AI"]
        assert long_ == pytest.approx(short, rel=0.01)

    def test_strong_scaling_moves_no_fewer_bytes(self):
        """8x the cores on the same problem reads exactly the same chip traffic.

        Each core takes a disjoint slice of the contraction, so chip bytes are
        invariant while chip bandwidth is fixed -- the memory term is a floor
        that parallelism cannot lower, and only the compute term shrinks.  This
        is the whole reason strong scaling on this kernel is nearly flat.
        """
        four = self.run(out_split=2, k_split=2)[3]
        thirty_two = self.run(out_split=2, k_split=16)[3]
        assert chip_dram_bytes(four) == chip_dram_bytes(thirty_two)
        assert thirty_two.kernel_cycles < four.kernel_cycles


class TestSdpaDecodePvCrossCoreReduce(_SdpaDecodePvBase):
    """The cross-core fold -- the part a single-configuration test cannot reach."""

    @pytest.mark.parametrize("kv_len,out_split,k_split", CONFIGS)
    def test_comm_cycles_match_closed_form(self, kv_len, out_split, k_split):
        """Partial-sum payload over ring bandwidth, priced across the whole chip."""
        m, n, _ = slices(kv_len, HEAD_DIM, Q_PER_KV, out_split, k_split)
        cores = out_split * k_split
        expected = (cores - 1) * (m * n * 2) / RING_B_PER_CYCLE
        per_core = self.run(kv_len=kv_len, out_split=out_split,
                            k_split=k_split)[3].per_core_summary()
        assert {row["comm_cycles"] for row in per_core} == {expected}

    def test_unsplit_contraction_has_no_comm(self):
        """``k_split = 1`` emits no fold at all, so the comm term is exactly zero.

        Guards the boundary between the generator's two code paths: the notebook
        needs the fold, and this is the configuration that proves the comm cycles
        come from the fold rather than from having a two-dimensional grid.
        """
        per_core = self.run(out_split=2, k_split=1)[3].per_core_summary()
        assert all(row["comm_cycles"] == 0 for row in per_core)

    def test_comm_tracks_core_count_not_fan_in(self):
        """Two groups of 16 and four groups of 8 are charged the same.

        The model prices the ring across every active core rather than across the
        reduce group, so halving the fan-in at a fixed core count and payload
        does not reduce the charge.  Asserted so that a ring model which does
        become group-aware fails here loudly instead of quietly redrawing the
        notebook's plot.
        """
        # Both sides must carry the same payload, so pick out_split to leave
        # n_local = 64 either way; only the fan-in differs (16 vs 8).
        wide = self.run(head_dim=128, out_split=2, k_split=16)[3]
        narrow = self.run(head_dim=256, out_split=4, k_split=8)[3]
        assert (max(r["comm_cycles"] for r in wide.per_core_summary())
                == max(r["comm_cycles"] for r in narrow.per_core_summary()))


class TestSdpaDecodePvPreconditions:
    """The generator rejects shapes whose reported cost would not be its own."""

    @pytest.mark.parametrize("kwargs,expected", [
        (dict(out_split=3), "divisible by out_split"),
        (dict(kv_len=1000, k_split=3), "divisible by k_split"),
        # 128/2 = 64 f16 = 128 B is exactly one stick; 64/2 = 32 f16 is half of
        # one, and HBM charges the whole stick either way.
        (dict(head_dim=64, out_split=2), "not a whole 128 B stick"),
        # 8192/256 = 32 tokens = 64 B, the same failure on the contraction axis.
        (dict(k_split=256), "not a whole 128 B stick"),
    ])
    def test_rejects_bad_shapes(self, kwargs, expected):
        params = dict(kv_len=KV_LEN, head_dim=HEAD_DIM, q_per_kv=Q_PER_KV,
                      out_split=2, k_split=16)
        params.update(kwargs)
        with pytest.raises(ValueError, match=expected):
            gen_sdpa_decode_pv_mlir(**params)

