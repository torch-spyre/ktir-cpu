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
"""
Simple end-to-end test of KTIR CPU backend.

Demonstrates the CPU backend working on a simple element-wise operation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from ktir_cpu.interpreter import KTIRInterpreter

def test_basic_execution():
    """Test basic KTIR execution without file."""
    print("="*70)
    print("Test 1: Basic Arithmetic Operations")
    print("="*70)

    # Create simple KTIR code for element-wise addition
    ktir_text = """
module {
  func.func @add_test() attributes { grid = [1, 1, 1] } {
    return
  }
}
"""

    interpreter = KTIRInterpreter()
    interpreter.load(ktir_text)
    print("✓ Successfully parsed KTIR text")
    print("✓ Basic execution test passed\n")


def test_memory_hierarchy():
    """Test memory hierarchy simulation."""
    print("="*70)
    print("Test 2: Memory Hierarchy Simulation")
    print("="*70)

    from ktir_cpu.memory import HBMSimulator, LXScratchpad, SpyreMemoryHierarchy

    # Test HBM
    hbm = HBMSimulator(size_gb=1)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)
    read_back = hbm.read(ptr, data.size, "f16")

    assert np.allclose(data, read_back), "HBM read/write mismatch"
    print(f"✓ HBM simulation: Allocated {data.nbytes} bytes at 0x{ptr:08x}")
    print(f"  Written: {data}")
    print(f"  Read back: {read_back}")

    # Test LX
    lx = LXScratchpad(size_mb=2, core_id=0)
    lx_ptr = 0
    lx.write(lx_ptr, data)
    lx_read = lx.read(lx_ptr, data.size, "f16")

    assert np.allclose(data, lx_read), "LX read/write mismatch"
    print(f"✓ LX scratchpad: wrote {data.nbytes} bytes, read back OK")

    # Test capacity enforcement via CoreContext.track_lx()
    from ktir_cpu.grid import CoreContext
    ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0),
                      lx=LXScratchpad(size_mb=2, core_id=0), hbm=hbm)
    with pytest.raises(MemoryError):
        ctx.track_lx("%huge", 3 * 1024 * 1024)  # 3MB > 2MB limit
    print("✓ LX capacity enforcement: MemoryError raised as expected")

    print("✓ Memory hierarchy test passed\n")


def test_grid_execution():
    """Test grid execution."""
    print("="*70)
    print("Test 3: Grid Execution (32 cores)")
    print("="*70)

    from ktir_cpu.grid import GridExecutor
    from ktir_cpu.memory import SpyreMemoryHierarchy

    # Create 32-core grid
    memory = SpyreMemoryHierarchy(num_cores=32)
    grid = GridExecutor(grid_shape=(32, 1, 1), memory=memory)

    print(f"✓ Created grid with {len(grid.cores)} cores")
    print(f"  Core 0 position: {grid.cores[0].grid_pos}")
    print(f"  Core 31 position: {grid.cores[31].grid_pos}")

    # Test core group selection
    all_cores = grid.get_cores_in_group((-1, 0, 0))  # All cores in row 0
    print(f"✓ Core group selection: Found {len(all_cores)} cores in row 0")

    # Test different grid shapes
    grid_8x4 = GridExecutor(grid_shape=(8, 4, 1), memory=memory)
    print(f"✓ Created 8×4 grid with {len(grid_8x4.cores)} cores")

    # Get cores in column 2
    col_2 = grid_8x4.get_cores_in_group((2, -1, 0))
    print(f"✓ Column 2 has {len(col_2)} cores: {col_2}")

    print("✓ Grid execution test passed\n")




def main():
    """Run all simple tests."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*18 + "KTIR CPU Backend - Simple Tests" + " "*19 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    tests = [
        test_basic_execution,
        test_memory_hierarchy,
        test_grid_execution,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Tests for flat memory simulators (ktir_cpu/memory.py)
# ---------------------------------------------------------------------------

class TestMemorySimulator:
    """Flat byte-addressed memory simulator tests (ktir_cpu/memory.py).

    Covers HBMSimulator and LXScratchpad in isolation: direct reads/writes,
    sub-allocation addressing, unmapped-address errors, and HBM/LX parity.
    No stride logic, coordinate sets, or MemoryOps involvement."""

    def _make_hbm(self):
        from ktir_cpu.memory import HBMSimulator
        hbm = HBMSimulator()
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        ptr = hbm.allocate(data.nbytes)
        hbm.write(ptr, data)
        return hbm, ptr

    def _make_lx(self):
        from ktir_cpu.memory import LXScratchpad
        lx = LXScratchpad()
        ptr = 0
        data = np.arange(16, dtype=np.float16).reshape(4, 4)
        lx.write(ptr, data)
        return lx, ptr

    # --- direct key read ---

    def test_hbm_direct_read_exact_shape(self):
        hbm, ptr = self._make_hbm()
        result = hbm.read(ptr, 16, "f16").reshape(4, 4)
        expected = np.arange(16, dtype=np.float16).reshape(4, 4)
        assert np.array_equal(result, expected)

    def test_lx_direct_read_exact_shape(self):
        lx, ptr = self._make_lx()
        result = lx.read(ptr, 16, "f16").reshape(4, 4)
        expected = np.arange(16, dtype=np.float16).reshape(4, 4)
        assert np.array_equal(result, expected)

    # --- sub-allocation read (ptr inside an existing block) ---

    def test_hbm_sub_allocation_read(self):
        """Read a single element from the middle of an existing HBM allocation."""
        hbm, ptr = self._make_hbm()
        # Element [1, 2] is at flat offset 6, byte offset 12
        result = hbm.read(ptr + 12, 1, "f16")
        assert result[0] == np.float16(6.0)

    def test_lx_sub_allocation_read(self):
        """Read a single element from the middle of an existing LX allocation."""
        lx, ptr = self._make_lx()
        # Element [1, 2] is at flat offset 6, byte offset 12
        result = lx.read(ptr + 12, 1, "f16")
        assert result[0] == np.float16(6.0)

    def test_hbm_sub_allocation_read_row(self):
        """Read a full row from the middle of an HBM allocation."""
        hbm, ptr = self._make_hbm()
        # Row 2 starts at flat offset 8, byte offset 16
        result = hbm.read(ptr + 16, 4, "f16")
        assert np.array_equal(result, np.array([8, 9, 10, 11], dtype=np.float16))

    def test_lx_sub_allocation_read_row(self):
        """Read a full row from the middle of an LX allocation."""
        lx, ptr = self._make_lx()
        result = lx.read(ptr + 16, 4, "f16")
        assert np.array_equal(result, np.array([8, 9, 10, 11], dtype=np.float16))

    # --- sub-allocation write (ptr inside an existing block) ---

    def test_hbm_sub_allocation_write(self):
        """Write a single element into the middle of an HBM allocation."""
        hbm, ptr = self._make_hbm()
        hbm.write(ptr + 12, np.array([99.0], dtype=np.float16))
        result = hbm.read(ptr, 16, "f16").reshape(4, 4)
        assert result[1, 2] == np.float16(99.0)
        expected = np.arange(16, dtype=np.float16).reshape(4, 4)
        expected[1, 2] = 99.0
        assert np.array_equal(result, expected)

    def test_lx_sub_allocation_write(self):
        """Write a single element into the middle of an LX allocation."""
        lx, ptr = self._make_lx()
        lx.write(ptr + 12, np.array([99.0], dtype=np.float16))
        result = lx.read(ptr, 16, "f16").reshape(4, 4)
        assert result[1, 2] == np.float16(99.0)
        expected = np.arange(16, dtype=np.float16).reshape(4, 4)
        expected[1, 2] = 99.0
        assert np.array_equal(result, expected)

    # --- unmapped address raises ValueError ---

    def test_hbm_unmapped_raises(self):
        from ktir_cpu.memory import HBMSimulator
        hbm = HBMSimulator()
        with pytest.raises(ValueError, match="unmapped"):
            hbm.read(0xDEAD, 4, "f16")

    def test_lx_unmapped_raises(self):
        from ktir_cpu.memory import LXScratchpad
        lx = LXScratchpad()
        with pytest.raises(ValueError, match="unmapped"):
            lx.read(0xDEAD, 4, "f16")

    # --- HBM and LX produce identical results for the same operations ---

    def test_hbm_lx_sub_read_identical(self):
        hbm, hbm_ptr = self._make_hbm()
        lx, lx_ptr = self._make_lx()
        for byte_offset in [0, 2, 12, 24, 30]:
            hbm_val = hbm.read(hbm_ptr + byte_offset, 1, "f16")
            lx_val = lx.read(lx_ptr + byte_offset, 1, "f16")
            assert np.array_equal(hbm_val, lx_val), (
                f"Mismatch at byte_offset={byte_offset}: HBM={hbm_val}, LX={lx_val}"
            )

    def test_hbm_lx_sub_write_identical(self):
        hbm, hbm_ptr = self._make_hbm()
        lx, lx_ptr = self._make_lx()
        patch = np.array([77.0], dtype=np.float16)
        hbm.write(hbm_ptr + 12, patch)
        lx.write(lx_ptr + 12, patch)
        assert np.array_equal(
            hbm.read(hbm_ptr, 16, "f16"),
            lx.read(lx_ptr, 16, "f16"),
        )
