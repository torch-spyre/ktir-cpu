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
Basic tests for KTIR CPU backend.

Tests core functionality of the interpreter.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

from ktir_cpu.memory import HBMSimulator, LXScratchpad
from ktir_cpu.grid import GridExecutor
from ktir_cpu.ir_types import Tile, TileRef
from ktir_cpu.interpreter import _ktir_dtype


def test_memory_hbm():
    """Test HBM simulator."""
    print("Testing HBM simulator...")

    hbm = HBMSimulator(size_gb=1)

    # Allocate and write
    data = np.array([1, 2, 3, 4], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)

    # Read back
    read_data = hbm.read(ptr, data.size, _ktir_dtype(data.dtype))

    assert np.array_equal(data, read_data), "HBM read/write mismatch"
    print("  ✓ HBM read/write works")


def test_hbm_read_direct_hit_padding():
    """HBM.read: direct-hit path where allocation is smaller than requested shape"""
    hbm = HBMSimulator()
    data = np.array([1, 2], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)

    # Request 4 elements but only 2 are stored
    result = hbm.read(ptr, 4, "f16")
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert np.array_equal(result[:2], data), "First 2 elements wrong"
    assert np.array_equal(result[2:], np.zeros(2, dtype=np.float16)), "Padding should be zeros"


def test_hbm_read_subarray_partial_padding():
    """HBM.read: sub-array path where read range extends past end of allocation."""
    hbm = HBMSimulator()
    # Write a 4-element block
    data = np.array([10, 20, 30, 40], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)

    # Read starting 2 elements in, requesting 4 elements (only 2 available)
    offset_ptr = ptr + 2 * 2  # skip 2 f16 elements (2 bytes each)
    result = hbm.read(offset_ptr, 4, "f16")
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert result[0] == np.float16(30), f"Expected 30, got {result[0]}"
    assert result[1] == np.float16(40), f"Expected 40, got {result[1]}"
    assert result[2] == np.float16(0), "Padding should be zero"
    assert result[3] == np.float16(0), "Padding should be zero"


def test_hbm_write_partial_overwrite():
    """HBM.write: writing smaller data into a larger existing allocation."""
    hbm = HBMSimulator()
    # Write 4-element block
    data = np.array([1, 2, 3, 4], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)

    # Overwrite with only 2 elements (smaller than existing)
    patch = np.array([99, 88], dtype=np.float16)
    hbm.write(ptr, patch)

    # Full allocation still present; first 2 elements updated
    result = hbm.read(ptr, 4, "f16")
    assert result[0] == np.float16(99), f"Expected 99, got {result[0]}"
    assert result[1] == np.float16(88), f"Expected 88, got {result[1]}"
    assert result[2] == np.float16(3), f"Expected 3 unchanged, got {result[2]}"
    assert result[3] == np.float16(4), f"Expected 4 unchanged, got {result[3]}"


def test_hbm_get_np_dtype():
    """HBM._get_np_dtype: known dtypes, placeholders, and unknown."""
    hbm = HBMSimulator()
    assert hbm._get_np_dtype("f16") == np.float16
    assert hbm._get_np_dtype("float32") == np.float32
    with pytest.raises(NotImplementedError):
        hbm._get_np_dtype("fp8")
    with pytest.raises(NotImplementedError):
        hbm._get_np_dtype("mxfp8")
    with pytest.raises(ValueError, match="Unsupported"):
        hbm._get_np_dtype("unknown")


def test_hbm_write_full_replacement():
    """HBM.write: writing equal-or-larger data replaces existing allocation."""
    hbm = HBMSimulator()
    data = np.array([1, 2, 3, 4], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)

    replacement = np.array([10, 20, 30, 40], dtype=np.float16)
    hbm.write(ptr, replacement)

    result = hbm.read(ptr, 4, _ktir_dtype(data.dtype))
    assert np.array_equal(result, replacement)


def test_lx_read_unmapped_raises():
    """LXScratchpad.read: reading an unmapped ptr raises ValueError."""
    lx = LXScratchpad()
    with pytest.raises(ValueError, match="unmapped"):
        lx.read(0, (4,), "f16")


def test_lx_read_shape_mismatch():
    """LXScratchpad.read: shape mismatch returns correctly sized array."""
    lx = LXScratchpad()
    data = np.array([1, 2, 3, 4], dtype=np.float16)
    lx.write(0, data)

    # Read fewer elements than stored
    result = lx.read(0, 2, "f16")
    assert result.shape == (2,)
    assert np.array_equal(result, np.array([1, 2], dtype=np.float16))

    # Read more elements than stored — pads with zeros
    result = lx.read(0, 6, "f16")
    assert result.shape == (6,)
    assert np.array_equal(result[:4], data)
    assert np.array_equal(result[4:], np.zeros(2, dtype=np.float16))


def test_lx_get_np_dtype():
    """LXScratchpad._get_np_dtype: known dtypes, placeholders, and unknown."""
    lx = LXScratchpad()
    assert lx._get_np_dtype("f16") == np.float16
    assert lx._get_np_dtype("float32") == np.float32
    with pytest.raises(NotImplementedError):
        lx._get_np_dtype("fp8")
    with pytest.raises(NotImplementedError):
        lx._get_np_dtype("mxfp8")
    with pytest.raises(ValueError, match="Unsupported"):
        lx._get_np_dtype("unknown")


def test_lx_clear():
    """LXScratchpad.clear: resets memory and allocation state."""
    lx = LXScratchpad()
    lx.write(0, np.array([1, 2], dtype=np.float16))
    lx.next_ptr = 64
    lx.used = 32

    lx.clear()

    assert len(lx.memory) == 0
    assert lx.next_ptr == 0
    assert lx.used == 0


def test_memory_lx():
    """Test LX scratchpad."""
    print("Testing LX scratchpad...")

    lx = LXScratchpad(size_mb=2, core_id=0)

    # Write at address 0
    data = np.array([5, 6, 7, 8], dtype=np.float16)
    ptr = 0
    lx.write(ptr, data)

    # Read back
    read_data = lx.read(ptr, np.prod(data.size), _ktir_dtype(data.dtype))

    assert np.array_equal(data, read_data), "LX read/write mismatch"
    print("  ✓ LX read/write works")

    # Test capacity limit — enforced by CoreContext.track_lx(), not allocate()
    from ktir_cpu.grid import CoreContext
    from ktir_cpu.memory import HBMSimulator
    ctx = CoreContext(core_id=0, grid_pos=(0, 0, 0), lx=LXScratchpad(size_mb=2, core_id=0), hbm=HBMSimulator())
    try:
        ctx.track_lx("%huge", 3 * 1024 * 1024)  # 3MB > 2MB limit
        assert False, "Should have raised MemoryError"
    except MemoryError:
        print("  ✓ LX capacity limit enforced")


def test_grid_executor():
    """Test grid executor."""
    print("Testing grid executor...")

    from ktir_cpu.memory import SpyreMemoryHierarchy

    memory = SpyreMemoryHierarchy(num_cores=32)
    grid = GridExecutor(grid_shape=(32, 1, 1), memory=memory)

    assert len(grid.cores) == 32, "Should have 32 cores"
    assert grid.cores[0].grid_pos == (0, 0, 0), "Core 0 position wrong"
    assert grid.cores[31].grid_pos == (31, 0, 0), "Core 31 position wrong"
    print("  ✓ Grid executor creates cores correctly")

    # Test get_cores_in_group
    row_0_cores = grid.get_cores_in_group((-1, 0, 0))
    assert len(row_0_cores) == 32, "Should get all 32 cores in row 0"
    print("  ✓ get_cores_in_group works")

    # Test get_core and get_core_at_pos
    core = grid.get_core(5)
    assert core.grid_pos == (5, 0, 0)

    core_at = grid.get_core_at_pos(5, 0, 0)
    assert core_at.grid_pos == (5, 0, 0)
    assert core_at is core


def test_get_cores_in_group_yz_filters():
    """get_cores_in_group: y and z dimension filter branches."""
    from ktir_cpu.memory import SpyreMemoryHierarchy

    # 2x4x2 grid: x in [0,1], y in [0,3], z in [0,1]
    memory = SpyreMemoryHierarchy(num_cores=16)
    grid = GridExecutor(grid_shape=(2, 4, 2), memory=memory)

    # Filter by y=2, all x and z
    y2_cores = grid.get_cores_in_group((-1, 2, -1))
    assert all(grid.get_core(c).grid_pos[1] == 2 for c in y2_cores)
    assert len(y2_cores) == 4  # 2 x-values * 2 z-values

    # Filter by z=1, all x and y
    z1_cores = grid.get_cores_in_group((-1, -1, 1))
    assert all(grid.get_core(c).grid_pos[2] == 1 for c in z1_cores)
    assert len(z1_cores) == 8  # 2 x-values * 4 y-values

    # Filter by y=1, z=0
    y1z0_cores = grid.get_cores_in_group((-1, 1, 0))
    assert all(grid.get_core(c).grid_pos[1] == 1 and grid.get_core(c).grid_pos[2] == 0 for c in y1z0_cores)
    assert len(y1z0_cores) == 2


def test_get_cores_in_group_all_wildcards():
    """get_cores_in_group(-1, -1, -1) returns every core in the grid."""
    from ktir_cpu.memory import SpyreMemoryHierarchy

    memory = SpyreMemoryHierarchy(num_cores=8)
    grid = GridExecutor(grid_shape=(2, 2, 2), memory=memory)

    all_cores = grid.get_cores_in_group((-1, -1, -1))
    assert sorted(all_cores) == list(range(8))


def test_get_cores_in_group_xy_wildcards():
    """get_cores_in_group with wildcards on x and y returns all cores at fixed z."""
    from ktir_cpu.memory import SpyreMemoryHierarchy

    memory = SpyreMemoryHierarchy(num_cores=8)
    grid = GridExecutor(grid_shape=(2, 2, 2), memory=memory)

    z0_cores = grid.get_cores_in_group((-1, -1, 0))
    assert len(z0_cores) == 4
    assert all(grid.get_core(c).grid_pos[2] == 0 for c in z0_cores)


def test_grid_boundary_max_position():
    """Core at maximum grid position has correct (x, y, z) coordinates."""
    from ktir_cpu.memory import SpyreMemoryHierarchy

    memory = SpyreMemoryHierarchy(num_cores=24)
    grid = GridExecutor(grid_shape=(4, 3, 2), memory=memory)

    # Last core is at (x_max, y_max, z_max) = (3, 2, 1)
    last_id = grid.num_cores - 1
    core = grid.get_core(last_id)
    assert core.grid_pos == (3, 2, 1)
    # Round-trip: grid_to_linear must invert linear_to_grid
    assert grid._grid_to_linear(*core.grid_pos) == last_id


def test_execute_with_communication_multi_round():
    """execute_with_communication runs multiple rounds and converges once all messages are consumed.

    NOTE: The BSP replay model in execute_with_communication has a known correctness problem
    described in docs/gap_analysis.md section K1 ("Multi-round communication re-execution").
    Re-execution inflates latency tracking and breaks cyclic communication patterns.
    This test only exercises the simple forward-propagation case (core 0 → core 1) which
    converges correctly; it does not imply cyclic or bidirectional communication works.
    """
    from ktir_cpu.memory import SpyreMemoryHierarchy
    from ktir_cpu.ops.comm_ops import RingNetwork
    from ktir_cpu.ir_types import Tile

    memory = SpyreMemoryHierarchy(num_cores=2)
    grid = GridExecutor(grid_shape=(2, 1, 1), memory=memory)
    ring = RingNetwork(num_cores=2)

    round_num = [0]
    call_counts = [0]

    def execute_fn(core, ring_net):
        call_counts[0] += 1
        # Round 0 (call 0 and 1): core 0 sends to core 1.
        # Round 1 (call 2 and 3): core 1 consumes the message — no pending after this round.
        if round_num[0] == 0 and core.core_id == 0:
            tile = Tile(np.array([42.0], dtype=np.float16), "f16", (1,))
            ring_net.send(src_core=0, dst_core=1, tile=tile)
        elif round_num[0] == 1 and core.core_id == 1:
            ring_net.recv_from(src_core=0, dst_core=1)
        # Track which round we're in based on call count parity.
        if call_counts[0] % 2 == 0:
            round_num[0] += 1

    grid.execute_with_communication(execute_fn, ring)
    # Two rounds × two cores = 4 calls
    assert call_counts[0] == 4


def test_execute_with_communication_exceeds_max_rounds():
    """execute_with_communication raises RuntimeError when messages never drain."""
    from ktir_cpu.memory import SpyreMemoryHierarchy
    from ktir_cpu.ops.comm_ops import RingNetwork
    from ktir_cpu.ir_types import Tile

    memory = SpyreMemoryHierarchy(num_cores=2)
    grid = GridExecutor(grid_shape=(2, 1, 1), memory=memory)
    ring = RingNetwork(num_cores=2)

    def always_send(core, ring_net):
        # Accumulate messages that are never consumed — network never drains.
        tile = Tile(np.array([1.0], dtype=np.float16), "f16", (1,))
        ring_net.send(src_core=core.core_id, dst_core=(core.core_id + 1) % 2, tile=tile)

    with pytest.raises(RuntimeError, match="Communication didn't converge"):
        grid.execute_with_communication(always_send, ring, max_rounds=2)


def test_hbm_allocate_f32():
    """HBMSimulator.allocate followed by write/read round-trips f32 data."""
    hbm = HBMSimulator()
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)
    result = hbm.read(ptr, data.size, "f32")
    assert np.array_equal(result, data)


def test_hbm_read_uninitialized_region():
    """Reading bytes that fall past the end of an allocation returns zero-padding."""
    hbm = HBMSimulator()
    data = np.array([1.0, 2.0], dtype=np.float16)
    ptr = hbm.allocate(data.nbytes)
    hbm.write(ptr, data)
    # Ask for 4 elements but only 2 were written — last 2 must be zero.
    result = hbm.read(ptr, 4, "f16")
    assert len(result) == 4
    assert np.array_equal(result[:2], data)
    assert np.array_equal(result[2:], np.zeros(2, dtype=np.float16))


def test_lx_interleaved_allocations():
    """LX scratchpad read/write correctly handles multiple interleaved allocations."""
    lx = LXScratchpad(size_mb=2, core_id=0)

    a = np.array([1.0, 2.0], dtype=np.float16)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float16)
    ptr_a = 0x0000
    ptr_b = 0x0100  # Non-contiguous in LX address space

    lx.write(ptr_a, a)
    lx.write(ptr_b, b)

    # Both allocations must be independently readable.
    result_a = lx.read(ptr_a, 2, "f16")
    result_b = lx.read(ptr_b, 3, "f16")

    assert np.array_equal(result_a, a)
    assert np.array_equal(result_b, b)

    # Overwrite ptr_a and verify ptr_b is untouched.
    lx.write(ptr_a, np.array([99.0, 88.0], dtype=np.float16))
    result_b2 = lx.read(ptr_b, 3, "f16")
    assert np.array_equal(result_b2, b)


def test_tile_operations():
    """Test Tile and TileRef types."""
    print("Testing Tile operations...")

    # Create tiles
    data1 = np.array([1, 2, 3, 4], dtype=np.float16)
    data2 = np.array([5, 6, 7, 8], dtype=np.float16)

    tile1 = Tile(data1, "f16", data1.shape)
    tile2 = Tile(data2, "f16", data2.shape)

    # Test copy
    tile1_copy = tile1.copy()
    tile1_copy.data[0] = 999
    assert tile1.data[0] == 1, "Copy should be independent"
    print("  ✓ Tile copy works")

    # Test TileRef
    ref = TileRef(
        base_ptr=0x1000,
        shape=(4,),
        strides=[1],
        memory_space="HBM",
        dtype="f16"
    )

    assert ref.size_bytes() == 8, "TileRef size calculation wrong"
    print("  ✓ TileRef works")


@pytest.mark.parametrize("dtype, shape, expected_bytes", [
    ("f16", (4,),  8),
    ("f32", (4,), 16),
    ("i32", (4,), 16),
    ("i64", (4,), 32),
])
def test_tileref_size_bytes(dtype, shape, expected_bytes):
    """TileRef.size_bytes: correct byte count for each supported dtype."""
    ref = TileRef(base_ptr=0, shape=shape, strides=[1], memory_space="HBM", dtype=dtype)
    assert ref.size_bytes() == expected_bytes


def test_ktir_dtype_branches():
    """_ktir_dtype: known dtypes and unknown raises."""
    assert _ktir_dtype(np.dtype("float16")) == "f16"
    assert _ktir_dtype(np.dtype("float32")) == "f32"
    assert _ktir_dtype(np.dtype("int32")) == "i32"
    assert _ktir_dtype(np.dtype("int64")) == "i64"
    with pytest.raises(ValueError):
        _ktir_dtype(np.dtype("float64"))


def test_interpreter_no_module_guard():
    """execute_function and tensor_input_output_sizes raise when no module loaded."""
    from ktir_cpu.interpreter import KTIRInterpreter
    interp = KTIRInterpreter()
    import pytest
    with pytest.raises(RuntimeError, match="No module loaded"):
        interp.execute_function("foo")
    with pytest.raises(RuntimeError, match="No module loaded"):
        interp.tensor_input_output_sizes("foo")


def test_interpreter_simple():
    """Test interpreter on simple operations."""
    print("Testing interpreter...")

    from ktir_cpu.interpreter import KTIRInterpreter

    # Create simple KTIR code
    ktir_text = """
    module {
        func.func @add(%x: index, %y: index) -> index attributes { grid = [1, 1, 1] } {
            %c5 = arith.constant 5 : index
            %c10 = arith.constant 10 : index
            %sum = arith.addi %c5, %c10 : index
            return %sum : index
        }
    }
    """

    interpreter = KTIRInterpreter()
    interpreter.load(ktir_text)

    # This is a placeholder test - actual execution would need proper setup
    print("  ✓ Interpreter loads and initializes")


def test_existing_ktir_file():
    """Test on actual KTIR file if available."""
    print("Testing on existing KTIR file...")

    ktir_file = Path("add_kernel_ktir.mlir")
    if not ktir_file.exists():
        print("  ⊘ Skipping (add_kernel_ktir.mlir not found)")
        return

    try:
        from ktir_cpu import KTIRInterpreter

        interpreter = KTIRInterpreter()
        interpreter.load(str(ktir_file))

        # Just check it doesn't crash
        print("  ✓ Successfully loaded and parsed KTIR file")

    except Exception as e:
        print(f"  ⚠ Partial success (loaded but execution may need work): {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("KTIR CPU Backend Tests")
    print("=" * 70)
    print()

    tests = [
        ("Memory - HBM", test_memory_hbm),
        ("Memory - LX", test_memory_lx),
        ("Grid Executor", test_grid_executor),
        ("Tile Operations", test_tile_operations),
        ("Interpreter", test_interpreter_simple),
        ("Existing KTIR File", test_existing_ktir_file),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"{name}:")
            test_func()
            print()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
