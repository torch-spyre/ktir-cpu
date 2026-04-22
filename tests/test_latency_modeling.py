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
"""Tests for latency modeling assumptions.

Verifies the cycle model described in LATENCY_NOTES.md:
  - compute_cycles ∝ 1/num_cores  (work-splitting speedup)
  - memory_cycles  ≈ constant      (shared-bus HBM, no work-splitting benefit)
  - LX ops cost zero memory cycles
  - memory_cycles ∝ tile_size
  - max(total_cycles) == min(total_cycles)  (balanced load)
"""

import textwrap
import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter, HardwareConfig
from ktir_cpu.ir_types import _iter_ops

from ktir_cpu.parser_ast import parse_affine_set

from conftest import get_test_params


# ---------------------------------------------------------------------------
# Generic execution helper
# ---------------------------------------------------------------------------

def build_inputs(interp, func_name, entry=None):
    """Return a kwargs dict ready for execute_function.

    Tensor args are filled with random f16 arrays sized from
    tensor_input_output_sizes().  Scalar args come from entry["execute_kwargs"];
    None values are skipped (caller is expected to fill them if needed).
    entry may be omitted for inline kernels that have no scalar args.
    """
    sizes = interp.tensor_input_output_sizes(func_name)
    rng = np.random.default_rng(42)
    kwargs = {}
    for name, info in sizes.items():
        kwargs[name] = rng.standard_normal(info["shape"]).astype(np.float16)
    if entry is not None:
        for k, v in entry["execute_kwargs"].items():
            if v is not None:
                kwargs[k] = v
    return kwargs


# ---------------------------------------------------------------------------
# Inline MLIR kernels used by modeling-assumption tests
# ---------------------------------------------------------------------------

# Single-pass exp kernel: each core loads an HBM tile, applies math.exp
# element-wise, and stores the result back to HBM.
# Grid, total array size, and per-core tile size are parameterized so the
# same template serves both the shared-bus penalty test (fixed tile=128,
# growing total) and the work-splitting test (fixed total=128, shrinking tile).
_EXP_MLIR = textwrap.dedent("""\
    module {{
      func.func @exp_kernel(
          %x_ptr: index,
          %out_ptr: index
      ) attributes {{grid = [{num_cores}, 1]}} {{
        %core_id = ktdp.get_compute_tile_id : index
        %ctile = arith.constant {{{tile} : index}} : index
        %offset = arith.muli %core_id, %ctile : index
        %x_view = ktdp.construct_memory_view %x_ptr,
            sizes: [{total}], strides: [1]
            {{ coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + {total_m1} >= 0)>,
               memory_space = #ktdp.spyre_memory_space<HBM> }}
            : index -> memref<{total}xf16>
        %x_acc = ktdp.construct_access_tile %x_view[%offset] {{
            base_map = affine_map<(i) -> (i)>,
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + {tile_m1} >= 0)>
        }} : memref<{total}xf16> -> !ktdp.access_tile<{tile}xindex>
        %x_tile = ktdp.load %x_acc : !ktdp.access_tile<{tile}xindex> -> tensor<{tile}xf16>
        %y_tile = math.exp %x_tile : tensor<{tile}xf16>
        %out_view = ktdp.construct_memory_view %out_ptr,
            sizes: [{total}], strides: [1]
            {{ coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + {total_m1} >= 0)>,
               memory_space = #ktdp.spyre_memory_space<HBM> }}
            : index -> memref<{total}xf16>
        %out_acc = ktdp.construct_access_tile %out_view[%offset] {{
            base_map = affine_map<(i) -> (i)>,
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + {tile_m1} >= 0)>
        }} : memref<{total}xf16> -> !ktdp.access_tile<{tile}xindex>
        ktdp.store %y_tile, %out_acc : tensor<{tile}xf16>, !ktdp.access_tile<{tile}xindex>
        return
      }}
    }}
""")

# Copy kernel template: loads 128 f16 elements and stores them back.
# memory_space controls whether memory_cycles are charged (LX → 0, HBM → positive).
# Instantiated as _LX_MLIR and _HBM_MLIR for test_lx_reuse_vs_hbm_reload.
def _copy_mlir(func_name: str, memory_space: str) -> str:
    return textwrap.dedent(f"""\
        module {{
          func.func @{func_name}(
              %x_ptr: index,
              %out_ptr: index
          ) attributes {{grid = [1, 1]}} {{
            %x_view = ktdp.construct_memory_view %x_ptr,
                sizes: [128], strides: [1]
                {{ coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,
                  memory_space = #ktdp.spyre_memory_space<{memory_space}> }}
                : index -> memref<128xf16>
            %c0 = arith.constant {{0 : index}} : index
            %x_acc = ktdp.construct_access_tile %x_view[%c0] {{
                base_map = affine_map<(i) -> (i)>,
                access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
            }} : memref<128xf16> -> !ktdp.access_tile<128xindex>
            %x_tile = ktdp.load %x_acc : !ktdp.access_tile<128xindex> -> tensor<128xf16>
            %out_view = ktdp.construct_memory_view %out_ptr,
                sizes: [128], strides: [1]
                {{ coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,
                  memory_space = #ktdp.spyre_memory_space<{memory_space}> }}
                : index -> memref<128xf16>
            %c0b = arith.constant {{0 : index}} : index
            %out_acc = ktdp.construct_access_tile %out_view[%c0b] {{
                base_map = affine_map<(i) -> (i)>,
                access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
            }} : memref<128xf16> -> !ktdp.access_tile<128xindex>
            ktdp.store %x_tile, %out_acc : tensor<128xf16>, !ktdp.access_tile<128xindex>
            return
          }}
        }}
    """)


def _run_inline(mlir_text, func_name, cfg=None, seed_lx=False):
    """Load inline MLIR and execute *func_name*, returning the latency report.

    seed_lx: when True, mirror every HBM write into each core's LX at the
    same address. Used when memory views are patched to LX so that ktdp.load
    does not hit unmapped addresses. The data content is irrelevant for
    latency-only tests.
    """
    if cfg is None:
        cfg = HardwareConfig()
    interp = KTIRInterpreter(latency_config=cfg)
    interp.load(mlir_text)
    if seed_lx:
        _patch_seed_lx(interp)
    interp.execute_function(func_name, **build_inputs(interp, func_name))
    return interp.get_latency_report()


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patch_seed_lx(interp):
    """Mirror every HBM write into each core's LX at the same address.

    Wraps _prepare_execution so the mirror hook is installed after each fresh
    memory setup. Used when memory views are patched to LX but inputs are
    allocated in HBM by execute_function.
    """
    orig_prepare = interp._prepare_execution
    def _prepare_and_seed(grid_shape):
        orig_prepare(grid_shape)
        orig_write = interp.memory.hbm.write
        def _write_and_mirror(ptr, data):
            orig_write(ptr, data)
            for core in interp.grid_executor.cores:
                core.lx.write(ptr, data)
        interp.memory.hbm.write = _write_and_mirror
    interp._prepare_execution = _prepare_and_seed


def _patch_grid(interp, func_name, new_grid: tuple):
    """Mutate func.grid in the parsed IR (e.g. to change core count)."""
    interp.module.get_function(func_name).grid = new_grid


def _patch_tile_size(interp, func_name, new_size: int):
    """Override all 1-D construct_memory_view sizes and access_tile shapes.

    Only touches 1-D ops so 2-D kernels (softmax, matmul) are unaffected.
    Also updates coordinate_set / coordinate_space_set bounds to match the
    new size, so future bound-checking passes the correct upper limit.
    """
    func = interp.module.get_function(func_name)
    for op in _iter_ops(func.operations):
        if op.op_type in ("ktdp.construct_memory_view", "ktdp.construct_access_tile"):
            if len(op.attributes.get("shape", ())) == 1:
                op.attributes["shape"] = (new_size,)
                new_set = parse_affine_set(
                    f"affine_set<(d0) : (d0 >= 0, -d0 + {new_size - 1} >= 0)>"
                )
                if op.op_type == "ktdp.construct_memory_view":
                    if op.attributes.get("coordinate_set") is not None:
                        op.attributes["coordinate_set"] = new_set
                else:
                    if op.attributes.get("coordinate_space_set") is not None:
                        op.attributes["coordinate_space_set"] = new_set

def _patch_tile_dim0(interp, func_name, old_dim0: int, new_dim0: int):
    """Scale dim-0 of all 2-D IR shapes whose current dim-0 equals *old_dim0*.

    Covers:
      - ktdp.construct_memory_view / construct_access_tile  (op.attributes["shape"])
      - arith.constant with is_tensor=True  (the SSA accumulator, e.g. accum_zero)

    Also updates coordinate_space_set on construct_access_tile to reflect the
    new dim-0 bound, so future bound-checking uses the correct upper limit.
    The coordinate_set on construct_memory_view is left unchanged because it
    describes the total memref extent, which does not shrink with the tile.

    This lets a test halve BLOCK_SIZE_M (e.g. 8→4) while leaving tiles whose
    dim-0 is BLOCK_SIZE_K (e.g. 32) untouched.
    """
    func = interp.module.get_function(func_name)
    for op in _iter_ops(func.operations):
        if op.op_type in ("ktdp.construct_memory_view", "ktdp.construct_access_tile"):
            shape = op.attributes.get("shape", ())
            if len(shape) == 2 and shape[0] == old_dim0:
                op.attributes["shape"] = (new_dim0, shape[1])
                if op.op_type == "ktdp.construct_access_tile":
                    if op.attributes.get("coordinate_space_set") is not None:
                        dim1 = shape[1]
                        op.attributes["coordinate_space_set"] = parse_affine_set(
                            f"affine_set<(d0, d1) : (d0 >= 0, -d0 + {new_dim0 - 1} >= 0, "
                            f"d1 >= 0, -d1 + {dim1 - 1} >= 0)>"
                        )
        elif op.op_type in ("arith.constant", "tensor.empty") and (
            op.op_type != "arith.constant" or op.attributes.get("is_tensor")
        ):
            shape = op.attributes.get("shape", ())
            if len(shape) == 2 and shape[0] == old_dim0:
                op.attributes["shape"] = (new_dim0, shape[1])

def _patch_memory_space(interp, func_name, space: str):
    """Set memory_space on every construct_memory_view op in *func_name*."""
    func = interp.module.get_function(func_name)
    for op in _iter_ops(func.operations):
        if op.op_type == "ktdp.construct_memory_view":
            op.attributes["memory_space"] = space


# ---------------------------------------------------------------------------
# TestHardwareConfig  (moved from test_latency.py)
# ---------------------------------------------------------------------------

class TestHardwareConfig:
    def test_default_values(self):
        cfg = HardwareConfig()
        assert cfg.num_cores == 32
        assert cfg.clock_ghz == 1.0
        assert cfg.hbm_bandwidth_tb_s == 1.0
        assert cfg.ring_bandwidth_tb_s == 4.0
        assert cfg.simd_elements_per_cycle == 64
        assert cfg.systolic_flops_per_cycle == 2 * 64 * 64 * 64
        assert cfg.transcendental_penalty == 4

    def test_custom_config(self):
        cfg = HardwareConfig(num_cores=8, hbm_bandwidth_tb_s=2.0)
        assert cfg.num_cores == 8
        assert cfg.hbm_bandwidth_tb_s == 2.0

    def test_hbm_bytes_per_cycle_per_core(self):
        cfg = HardwareConfig()
        # 1 TB/s at 1 GHz = 1e12 / 1e9 = 1000 bytes/cycle total
        # Per core: 1000 / 32 = 31.25
        assert cfg.hbm_bytes_per_cycle_per_core == pytest.approx(31.25)

    def test_ring_bytes_per_cycle(self):
        cfg = HardwareConfig()
        # 4 TB/s at 1 GHz = 4e12 / 1e9 = 4000 bytes/cycle
        assert cfg.ring_bytes_per_cycle == pytest.approx(4000.0)

    def test_derived_scales_with_clock(self):
        cfg = HardwareConfig(clock_ghz=2.0)
        # 1 TB/s at 2 GHz = 1e12 / 2e9 = 500 bytes/cycle total
        # Per core: 500 / 32 = 15.625
        assert cfg.hbm_bytes_per_cycle_per_core == pytest.approx(15.625)


# ---------------------------------------------------------------------------
# TestModelingAssumptions
# ---------------------------------------------------------------------------

class TestModelingAssumptions:

    # --- Test 1a: shared-bus bandwidth penalty (elementwise) ---

    @pytest.mark.parametrize("num_cores", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_shared_bus_bandwidth_penalty(self, path, func_name, entry, num_cores):
        """Per-core memory_cycles grows with num_cores due to the shared HBM bus.

        Each core processes a fixed 128-element tile regardless of num_cores
        (the total array grows with num_cores — no work splitting here).
        hbm_bytes_per_cycle_per_core = total_bw / num_cores shrinks as num_cores
        grows, so memory_cycles = bytes / (total_bw / num_cores) grows proportionally.
        compute_cycles is constant (fixed tile, independent of num_cores).
        Both quantities are verified analytically.

        num_cores=1             num_cores=2             num_cores=4
        ┌────────────────┐      ┌────────┐┌────────┐    ┌────┐┌────┐┌────┐┌────┐
        │   core 0       │      │ core 0 ││ core 1 │    │ c0 ││ c1 ││ c2 ││ c3 │
        │  128 elements  │      │  128el ││  128el │    │128 ││128 ││128 ││128 │
        └────────────────┘      └────────┘└────────┘    └────┘└────┘└────┘└────┘
        HBM bus: 1000 B/cy      HBM bus: 1000 B/cy      HBM bus: 1000 B/cy
        per-core BW: 1000       per-core BW: 500         per-core BW: 250

        compute_cycles =  2.0   compute_cycles =  2.0    compute_cycles =  2.0
        memory_cycles  = 16.4   memory_cycles  = 32.8    memory_cycles  = 65.5
                                                            (memory_cycles ∝ num_cores)
        """
        cfg = HardwareConfig(num_cores=num_cores)
        interp = KTIRInterpreter(latency_config=cfg)
        interp.load(path)
        _patch_grid(interp, func_name, (num_cores, 1, 1))
        interp.execute_function(func_name, **build_inputs(interp, func_name, entry))
        report = interp.get_latency_report()

        assert len(report.counters) == num_cores

        # All cores do equal work — counters must be identical
        compute_vals = [c.compute_cycles for c in report.counters.values()]
        memory_vals = [c.memory_cycles for c in report.counters.values()]
        assert all(v == pytest.approx(compute_vals[0], rel=1e-6) for v in compute_vals), \
            "Cores have unequal compute_cycles"
        assert all(v == pytest.approx(memory_vals[0], rel=1e-6) for v in memory_vals), \
            "Cores have unequal memory_cycles"

        # Analytic checks
        tile_elements = 128  # access tile: !ktdp.access_tile<128xindex>
        expected_compute = tile_elements / cfg.simd_elements_per_cycle  # 1 addf per element
        assert compute_vals[0] == pytest.approx(expected_compute, rel=1e-6)
        bytes_per_core = tile_elements * 2 * 3  # 2 loads + 1 store, f16
        expected_mem = bytes_per_core / cfg.hbm_bytes_per_cycle_per_core
        assert memory_vals[0] == pytest.approx(expected_mem, rel=1e-6)

    # --- Test 1b: work-splitting elementwise ---

    @pytest.mark.parametrize("num_cores", [1, 2, 4, 8])
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_work_splitting_elementwise(self, path, func_name, entry, num_cores):
        """compute_cycles ∝ 1/num_cores when a fixed 128-element array is split across cores.

        Total work is fixed at 128 elements.  Each core gets tile = 128 // num_cores.
        compute_cycles halves each time num_cores doubles (fewer elements per core).

        memory_cycles stays constant despite the tile shrinking: fewer bytes per core,
        but hbm_bytes_per_cycle_per_core also shrinks (shared bus), so they cancel:
          memory_cycles = (tile * f16 * 3) / (total_bw / num_cores)
                        = (128/nc * 2 * 3) / (total_bw / nc)
                        = 128 * 2 * 3 / total_bw   (nc cancels)
        This shows that the shared HBM bus prevents memory from benefiting from
        work splitting — only compute scales down.

        test_shared_bus_bandwidth_penalty       test_work_splitting_elementwise
        (fixed tile, growing total)             (fixed total=128, shrinking tile)

        num_cores=1      num_cores=2            num_cores=1      num_cores=2
        ┌──────────┐     ┌─────┐┌─────┐        ┌──────────┐     ┌─────┐┌─────┐
        │  core 0  │     │ c0  ││ c1  │        │  core 0  │     │ c0  ││ c1  │
        │ 128 elem │     │128el││128el│        │ 128 elem │     │ 64el││ 64el│
        └──────────┘     └─────┘└─────┘        └──────────┘     └─────┘└─────┘
        bw_pc: 1000      bw_pc: 500            bw_pc: 1000      bw_pc: 500

        compute =  2.0   compute =  2.0        compute =  2.0   compute =  1.0  (÷2)
        memory  = 16.4   memory  = 32.8  (×2)  memory  = 16.4   memory  = 16.4  (const)
        """
        total = 128
        tile = total // num_cores

        cfg = HardwareConfig(num_cores=num_cores)
        interp = KTIRInterpreter(latency_config=cfg)
        interp.load(path)
        _patch_grid(interp, func_name, (num_cores, 1, 1))
        _patch_tile_size(interp, func_name, tile)
        interp.execute_function(func_name, **build_inputs(interp, func_name, entry))
        report = interp.get_latency_report()

        assert len(report.counters) == num_cores

        # compute_cycles ∝ 1/num_cores
        base_cfg = HardwareConfig(num_cores=1)
        expected_compute = (tile / base_cfg.simd_elements_per_cycle)
        assert report.counters[0].compute_cycles == pytest.approx(expected_compute, rel=1e-6)

        # memory_cycles constant (tile halves, bw_pc halves — cancel)
        total_bw_bytes_per_cycle = cfg.hbm_bytes_per_cycle_per_core * num_cores
        expected_mem = (total * 2 * 3) / total_bw_bytes_per_cycle
        assert report.counters[0].memory_cycles == pytest.approx(expected_mem, rel=1e-6)

    # --- Test 2: work-splitting matmul ---

    @pytest.mark.parametrize("grid_x", [2, 4, 8])
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("matmul_kernel_small"))
    def test_work_splitting_matmul(self, path, func_name, entry, grid_x):
        """compute_cycles ∝ 1/grid_x when per-core M tile is halved each time grid_x doubles.

        matmul_small baseline: grid [2, 2], A tile 8×32 (BLOCK_SIZE_M=8).
        We simulate splitting by doubling grid_x and halving the M dimension
        of the A and C tiles (8×32 → 4×32 → 2×32) via _patch_tile_dim0.
        The B tile (32×32) is left unchanged because its dim-0 is BLOCK_SIZE_K=32,
        not BLOCK_SIZE_M.

        Expected: compute_cycles(grid_x) = compute_cycles(baseline) / (grid_x / 2)
        since FLOPs = 2 * M * N * K and M halves with each doubling of grid_x.

        memory_cycles are derived analytically from the tile shapes and the
        shared-bus bandwidth formula, accounting for the fact that only the A
        and C tiles shrink with M while the B tile is unchanged.

        test_work_splitting_matmul
        (fixed total M=16, halve per-core tile as grid_x doubles)

        Baseline: grid_x=2, BLOCK_SIZE_M=8        Scaled: grid_x=4, BLOCK_SIZE_M=4

        Matrix A (16×64)   Matrix B (64×64)        Matrix A (16×64)   Matrix B (64×64)
        ┌──────────────┐   ┌──────────────┐        ┌──────────────┐   ┌──────────────┐
        │▓▓▓▓▓▓▓▓ c0   │   │              │        │▓▓▓▓ c0       │   │              │
        │▓▓▓▓▓▓▓▓ c0   │ × │  B (shared)  │   →    │▓▓▓▓ c1       │ × │  B (shared)  │
        │▒▒▒▒▒▒▒▒ c1   │   │              │        │▒▒▒▒ c2       │   │              │
        │▒▒▒▒▒▒▒▒ c1   │   │              │        │▒▒▒▒ c3       │   │              │
        │  ...   ...   │   └──────────────┘        │  ...   ...   │   └──────────────┘
        └──────────────┘                           └──────────────┘
        A tile: 8×32 per core                     A tile: 4×32 per core
        FLOPs per core: 2×8×32×64 = 32768         FLOPs per core: 2×4×32×64 = 16384

        compute_cycles  = 32768 / systolic        compute_cycles  = 16384 / systolic  (÷2)
        memory_cycles   = (A + B + C) / bw_pc     memory_cycles   = (A' + B + C') / bw_pc'
                        = (8×32×2×2 + 32×32×2×2   (B unchanged; A,C halve; bw_pc halves)
                            + 8×32×2) / 31.25
        """
        base_block_m = entry["execute_kwargs"]["BLOCK_SIZE_M"]  # 8
        base_grid_x = 2  # baseline from the MLIR
        base_grid_y = 2

        def _run(gx):
            block_m = base_block_m * base_grid_x // gx  # 8 → 4 → 2
            cfg = HardwareConfig(num_cores=gx * base_grid_y)
            interp = KTIRInterpreter(latency_config=cfg)
            interp.load(path)
            _patch_grid(interp, func_name, (gx, 2, 1))
            _patch_tile_dim0(interp, func_name, old_dim0=base_block_m, new_dim0=block_m)
            kwargs = build_inputs(interp, func_name, entry)
            kwargs["BLOCK_SIZE_M"] = block_m
            interp.execute_function(func_name, **kwargs)
            return interp.get_latency_report(), cfg, block_m

        base_report, base_cfg, _ = _run(base_grid_x)
        scaled_report, scaled_cfg, scaled_block_m = _run(grid_x)

        assert len(scaled_report.counters) == grid_x * base_grid_y

        bk = entry["execute_kwargs"]["BLOCK_SIZE_K"]
        bn = entry["execute_kwargs"]["BLOCK_SIZE_N"]
        K  = entry["execute_kwargs"]["K"]
        n_iters = K // bk
        f16 = 2  # bytes per element

        scale = grid_x / base_grid_x  # 1, 2, or 4
        base_compute = base_report.counters[0].compute_cycles

        # FLOPs ∝ block_m → compute_cycles scale as 1/scale
        assert scaled_report.counters[0].compute_cycles == pytest.approx(base_compute / scale, rel=1e-6)

        # Analytical memory_cycles for the scaled run:
        #   A tiles: block_m * bk * f16 bytes, loaded n_iters times
        #   B tiles: bk * bn * f16 bytes, loaded n_iters times (unchanged)
        #   C tile:  block_m * bn * f16 bytes, stored once
        a_bytes = scaled_block_m * bk * f16 * n_iters
        b_bytes = bk * bn * f16 * n_iters
        c_bytes = scaled_block_m * bn * f16
        expected_mem = (a_bytes + b_bytes + c_bytes) / scaled_cfg.hbm_bytes_per_cycle_per_core
        assert scaled_report.counters[0].memory_cycles == pytest.approx(expected_mem, rel=1e-6)

    # --- Test 3: work-splitting transcendental ---

    @pytest.mark.parametrize("num_cores", [1, 2, 4, 8])
    def test_work_splitting_transcendental(self, num_cores):
        """compute_cycles ∝ 1/num_cores when a fixed 128-element array is split across cores.

        Total work is fixed at 128 elements; each core processes tile = 128 // num_cores.
        compute_cycles halves with each doubling of num_cores (fewer exp ops per core).

        Same shared-bus cancellation as test_work_splitting_elementwise: tile halves
        but hbm_bytes_per_cycle_per_core also halves, so memory_cycles stays constant:
          memory_cycles = (tile * f16 * 2) / (total_bw / num_cores)
                        = (128/nc * 2 * 2) / (total_bw / nc)
                        = 128 * 2 * 2 / total_bw   (nc cancels)
                    
        similar to test_work_splitting_elementwise
        """
        total = 128
        tile = total // num_cores
        mlir = _EXP_MLIR.format(
            num_cores=num_cores,
            total=total,
            total_m1=total - 1,
            tile=tile,
            tile_m1=tile - 1,
        )
        cfg = HardwareConfig(num_cores=num_cores)
        report = _run_inline(mlir, "exp_kernel", cfg=cfg)

        assert len(report.counters) == num_cores

        # All cores have equal compute and memory cycles
        compute_vals = [c.compute_cycles for c in report.counters.values()]
        memory_vals = [c.memory_cycles for c in report.counters.values()]
        assert all(v == pytest.approx(compute_vals[0], rel=1e-6) for v in compute_vals)
        assert all(v == pytest.approx(memory_vals[0], rel=1e-6) for v in memory_vals)

        # compute_cycles = tile / simd * penalty  (∝ 1/num_cores)
        expected_compute = (tile / cfg.simd_elements_per_cycle) * cfg.transcendental_penalty
        assert compute_vals[0] == pytest.approx(expected_compute, rel=1e-6)

        # memory_cycles constant: tile halves, bw_pc halves — nc cancels
        total_bw_bytes_per_cycle = cfg.hbm_bytes_per_cycle_per_core * num_cores
        expected_memory = (total * 2 * 2) / total_bw_bytes_per_cycle  # 1 load + 1 store
        assert memory_vals[0] == pytest.approx(expected_memory, rel=1e-6)

    # --- Test 4: tile size → memory cycles proportional ---

    @pytest.mark.parametrize("tile_size", [64, 128, 256, 512])
    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_tile_size_memory_cycles(self, path, func_name, entry, tile_size):
        """memory_cycles ∝ tile_size for a single-core HBM load/store kernel.

        We patch all 1-D construct_memory_view sizes on add_kernel (single core)
        and verify that memory_cycles scale linearly with tile_size.

        tile_size=64        tile_size=128       tile_size=256
        ┌──────────┐        ┌──────────────┐    ┌─────────────────────┐
        │ HBM load │ 128B   │ HBM load     │    │ HBM load            │
        │ HBM load │ 128B   │ HBM load     │    │ HBM load            │
        │ HBM store│ 128B   │ HBM store    │    │ HBM store           │
        └──────────┘        └──────────────┘    └─────────────────────┘
        bytes = 64×2×3      bytes = 128×2×3     bytes = 256×2×3
              = 384               = 768               = 1536
        memory_cycles ∝ tile_size  (fixed 1-core bw_pc = 1000 B/cy)
        """
        cfg = HardwareConfig(num_cores=1)
        interp = KTIRInterpreter(latency_config=cfg)
        interp.load(path)
        _patch_grid(interp, func_name, (1, 1, 1))
        _patch_tile_size(interp, func_name, tile_size)
        interp.execute_function(func_name, **build_inputs(interp, func_name, entry))
        report = interp.get_latency_report()

        core0 = report.counters[0]
        # 3 HBM ops (2 loads + 1 store), each transferring tile_size * 2 bytes (f16)
        expected_bytes = tile_size * 2 * 3
        expected_mem = expected_bytes / cfg.hbm_bytes_per_cycle_per_core
        assert core0.memory_cycles == pytest.approx(expected_mem, rel=1e-6)

    # --- Test 5: LX ops cost zero memory cycles ---

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_lx_ops_zero_cycles(self, path, func_name, entry):
        """Patching all memory views to LX → memory_cycles == 0 on every core."""
        cfg = HardwareConfig(num_cores=1)
        interp = KTIRInterpreter(latency_config=cfg)
        interp.load(path)
        _patch_grid(interp, func_name, (1, 1, 1))
        _patch_memory_space(interp, func_name, "LX")
        _patch_seed_lx(interp)
        interp.execute_function(func_name, **build_inputs(interp, func_name, entry))
        report = interp.get_latency_report()

        for core_id, counters in report.counters.items():
            assert counters.memory_cycles == 0.0, \
                f"Core {core_id}: expected 0 memory_cycles for LX ops, got {counters.memory_cycles}"

    # --- Test 6: LX reuse vs HBM reload ---

    def test_lx_reuse_vs_hbm_reload(self):
        """LX variant has strictly lower memory_cycles than HBM variant.

        Both kernels copy 128 f16 elements (load + store).  The only
        difference is memory_space on the construct_memory_view ops.

        HBM kernel                      LX kernel
        ┌────────────────────┐          ┌────────────────────┐
        │ ktdp.load  (HBM)   │──DMA──▶  │ ktdp.load  (LX)    │──▶ (on-chip, free)
        │ memory_cycles > 0  │          │ memory_cycles = 0  │
        │ ktdp.store (HBM)   │◀──DMA──  │ ktdp.store (LX)    │◀── (on-chip, free)
        │ memory_cycles > 0  │          │ memory_cycles = 0  │
        └────────────────────┘          └────────────────────┘
        """
        cfg = HardwareConfig()

        lx_report = _run_inline(_copy_mlir("lx_kernel", "LX"), "lx_kernel", cfg=cfg, seed_lx=True)
        hbm_report = _run_inline(_copy_mlir("hbm_kernel", "HBM"), "hbm_kernel", cfg=cfg)

        lx_mem = lx_report.counters[0].memory_cycles
        hbm_mem = hbm_report.counters[0].memory_cycles

        assert lx_mem < hbm_mem, (
            f"Expected LX memory_cycles ({lx_mem}) < HBM memory_cycles ({hbm_mem})"
        )
        assert lx_mem == 0.0, f"LX ops should cost 0 memory cycles, got {lx_mem}"

    # --- Test 7: balanced work distribution ---

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_balanced_work_distribution(self, path, func_name, entry):
        """With 32 cores, max(total_cycles) == min(total_cycles).

        The model assigns equal tiles to every core, so there is no
        load imbalance.  Kernel cycles = max = min.
        """
        cfg = HardwareConfig(num_cores=32)
        interp = KTIRInterpreter(latency_config=cfg)
        interp.load(path)
        interp.execute_function(func_name, **build_inputs(interp, func_name, entry))
        report = interp.get_latency_report()

        assert len(report.counters) == 32

        totals = [c.total_cycles for c in report.counters.values()]
        assert max(totals) == pytest.approx(min(totals), rel=1e-9), (
            f"Load imbalance: max={max(totals):.2f} min={min(totals):.2f}"
        )
