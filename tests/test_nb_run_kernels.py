"""Tests for notebook kernel generators (demo_gen_mlir / demo_helpers).

Covers the per-unit watermark checks and input guards that the notebook
itself does not exercise programmatically.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from demo_gen_mlir import gen_rope_mlir  # noqa: E402
from demo_helpers import run_kernel_rope  # noqa: E402
from ktir_cpu.latency import HardwareConfig  # noqa: E402


@pytest.fixture
def hw():
    return HardwareConfig()


class TestRoPEAIWatermark:
    """Verify AI = 3h/(4h+2) for various head/grid configs."""

    @pytest.mark.parametrize("num_heads,grid_h,expected_ai", [
        (4, 2, 3 * 2 / (4 * 2 + 2)),       # h=2, AI=0.6
        (40, 2, 3 * 20 / (4 * 20 + 2)),     # h=20, AI=0.7317
        (40, 4, 3 * 10 / (4 * 10 + 2)),     # h=10, AI=0.7143
    ])
    def test_ai_matches_formula(self, hw, num_heads, grid_h, expected_ai):
        report = run_kernel_rope(
            hw, num_heads=num_heads, seq_len=512, head_dim=128,
            grid_s=2, grid_h=grid_h, tile_seq=256)
        rf = report.core_roofline()
        assert rf["core_AI"] == pytest.approx(expected_ai, rel=1e-3)


class TestRoPEPerCoreSummary:
    """Verify per-core cycle structure for embarrassingly-parallel kernel."""

    def test_memory_bound(self, hw):
        report = run_kernel_rope(
            hw, num_heads=4, seq_len=512, head_dim=128,
            grid_s=2, grid_h=2, tile_seq=256)
        summary = report.per_core_summary()
        for core in summary:
            assert core["memory_cycles"] > core["compute_cycles"]

    def test_all_cores_equal(self, hw):
        report = run_kernel_rope(
            hw, num_heads=4, seq_len=512, head_dim=128,
            grid_s=2, grid_h=2, tile_seq=256)
        summary = report.per_core_summary()
        cycles = [c["total_cycles"] for c in summary]
        assert all(c == cycles[0] for c in cycles)

    def test_no_communication(self, hw):
        report = run_kernel_rope(
            hw, num_heads=4, seq_len=512, head_dim=128,
            grid_s=2, grid_h=2, tile_seq=256)
        summary = report.per_core_summary()
        for core in summary:
            assert core["comm_cycles"] == 0.0


class TestRoPEDivisibilityGuard:
    """Verify ValueError on non-divisible inputs."""

    def test_heads_not_divisible(self):
        with pytest.raises(ValueError, match="num_heads=40 not divisible by grid_h=3"):
            gen_rope_mlir(num_heads=40, seq_len=1024, head_dim=128,
                          grid_s=2, grid_h=3, tile_seq=256)

    def test_seq_not_divisible(self):
        with pytest.raises(ValueError, match="seq_len=100 not divisible"):
            gen_rope_mlir(num_heads=4, seq_len=100, head_dim=128,
                          grid_s=2, grid_h=2, tile_seq=256)

    def test_seq_tiles_not_divisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            gen_rope_mlir(num_heads=4, seq_len=1024, head_dim=128,
                          grid_s=4, grid_h=2, tile_seq=512)
