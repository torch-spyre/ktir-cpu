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

"""Shared test parameters and parsed metadata for MLIR example files.

Examples directory layout (examples/)
--------------------------------------
triton-ktir/
    Full-size kernels produced by the Triton -> KTIR compilation path.
    Used as the primary functional correctness fixtures (test_examples.py).
    These match real workload shapes and should not be resized.

latency/
    Small-footprint kernels used exclusively by latency/scaling tests
    (test_latency.py, test_latency_modeling.py).  Sizes are reduced to keep
    test runtime low while preserving the structural properties needed to
    verify cycle scaling laws (e.g. memory_cycles ∝ 1/bandwidth,
    matmul_cycles ∝ 1/systolic_throughput).  Patch helpers in
    test_latency_modeling.py mutate op attributes after parsing, so tile
    sizes baked into the MLIR are overridden at test time.

ktir/
    Kernels written directly in the KTDP/KTIR dialect, not generated from a
    frontend like Triton.  Currently holds edge-case kernels (e.g.
    softmax_wide.mlir which intentionally overflows LX scratchpad) used by
    dedicated failure tests.

rfc/
    Kernels transcribed from the KTIR RFC spec examples.  They exercise
    features that are not yet implemented (indirect access tiles, distributed
    memory views, etc.) and are therefore all xfail.  See
    tests/test_spec_gaps.py.

EXAMPLE_PARAMS maps each function name to a list of entry dicts.  Each entry
has a ``path`` (relative to examples/) and ``execute_kwargs``.  Entries may
also have an optional ``exception_msg`` field — these are *failure examples*
designed to raise a specific exception when executed.

Failure examples are skipped by ``get_test_params`` unless a ``filter``
string explicitly selects them, so they never appear in normal parametrize
lists by accident.  Dedicated failure tests select them via filter and read
``entry["exception_msg"]`` to assert the expected error message.

``get_test_params`` returns ``(abs_path, func_name, entry)`` triples.  Tests
receive the full entry dict — including ``execute_kwargs`` and, for failure
examples, ``exception_msg`` — so no separate helper is needed.

``parse_example`` extracts metadata directly from the MLIR text with
regex (independent of the KTIRInterpreter parser under test).
"""

import re
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# ---------------------------------------------------------------------------
# Example registry — func_name -> list[dict] with path + execute_kwargs
# ---------------------------------------------------------------------------

EXAMPLE_PARAMS: dict[str, list[dict]] = {
    "add_kernel": [
        {
            "path": "triton-ktir/vector_add_ktir.mlir",
            "execute_kwargs": {},
        },
    ],
    "softmax_kernel_small": [
        {
            "path": "latency/softmax_small.mlir",
            "execute_kwargs": {},
        },
    ],
    "softmax_kernel": [
        {
            "path": "triton-ktir/softmax_fwd_ktir.mlir",
            "execute_kwargs": {},
        },
        {
            "path": "ktir/softmax_wide.mlir",
            # C=262144 intentionally exceeds the 2MB LX scratchpad.
            # exception_msg marks this as a failure example: get_test_params
            # skips it by default; tests that explicitly filter for it assert
            # the raised exception using entry["exception_msg"].
            "execute_kwargs": {},
            "exception_msg": "LX scratchpad overflow",
        },
    ],
    "_layer_norm_fwd_fused": [
        {
            "path": "triton-ktir/layernorm_fwd_ktir.mlir",
            # sizes match construct_memory_view: [1152, 8192]
            # TODO: 1151→1152 change papers over non-divisible grid partition;
            # see torch-spyre/ktir-cpu#15 review point 2.
            "execute_kwargs": {"eps": 1e-5},
        },
    ],
    "matmul_kernel": [
        {
            "path": "triton-ktir/matmul_fwd_ktir.mlir",
            # M, N, K, BLOCK_SIZE_* are not function args in this MLIR (baked as
            # constants), but _run_matmul passes them as dead kwargs because it
            # also serves matmul_kernel_small which takes them as function args.
            # test_matmul_flops and test_work_splitting_matmul read these values.
            "execute_kwargs": {
                "M": 64, "N": 8192, "K": 2048,
                "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 128,
            },
        },
    ],
    "matmul_kernel_small": [
        {
            "path": "latency/matmul_small.mlir",
            # M=16, N=64, K=64 match construct_memory_view sizes
            # grid = [2, 2], BLOCK_SIZE_* match the MLIR function args
            "execute_kwargs": {
                "M": 16, "N": 64, "K": 64,
                "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
            },
        },
    ],
    "reduce_explicit_region": [
        {
            "path": "ktir/reduce_generic.mlir",
            # Tests linalg.reduce in the generic MLIR format (explicit combiner
            # region body) as emitted by the Triton Spyre C++ ConvertTTReduce pass.
            # %arg0 is used as both input and output pointer (same buffer).
            "execute_kwargs": {},
        },
    ],
    "sdpa_kernel_2d": [
        {
            "path": "triton-ktir/sdpa_2d.mlir",
            # Q, K, V, output all [32, 64] f16 — grid [1], no extra scalar kwargs
            "execute_kwargs": {},
        },
    ],
    "indexed_add_kernel": [
        {
            "path": "triton-ktir/indexed_add.mlir",
            "execute_kwargs": {"dim1_start": 0},
        },
    ],
    "kernel_unified_attention_spyre_2d": [
        {
            "path": "triton-ktir/paged_attention.mlir",
            # Concrete parameters from the MLIR header comments:
            #   num_tokens=8, seq_len_total=128, context_len=120, TILE_SIZE=16
            #   num_tiles = seq_len_total / TILE_SIZE = 128 / 16 = 8
            #   scale = 1 / sqrt(128)
            # Requires ktdp.construct_indirect_access_tile.
            "execute_kwargs": {
                "cur_batch_start_index": 0,
                "block_table_offset": 0,
                "num_tiles": 8,
                "context_len": 120,
                "scale": 0.08838834764831843,  # 1/sqrt(128)
            },
        },
    ],
    # ---------------------------------------------------------------------------
    # RFC example files (examples/rfc/) — spec-gap fixtures
    # These use hardcoded addresses and take no kernel arguments.
    # All are currently xfail; see tests/test_spec_gaps.py.
    # ---------------------------------------------------------------------------
    "indirect_access_copy": [
        {
            "path": "rfc/indirect-access-copy.mlir",
            # RFC §C.5 Example 1: Y[m,k] = X[IDX1[m,k], IDX2[m,k]]
            # Requires ktdp.construct_indirect_access_tile.
            "execute_kwargs": {},
        },
    ],
    "paged_tensor_copy_1core": [
        {
            "path": "rfc/paged-tensor-copy.mlir",
            # RFC §C.5 Example 2: paged-attention 4-D indirect gather
            # Requires ktdp.construct_indirect_access_tile.
            "execute_kwargs": {},
        },
    ],
    "distributed_view_copy": [
        {
            "path": "rfc/distributed-view-copy.mlir",
            # RFC §C.3: copy from distributed HBM+LX view into contiguous HBM tensor
            # Requires ktdp.construct_distributed_memory_view.
            "execute_kwargs": {},
        },
    ],
    "add": [
        {
            "path": "rfc/add-with-control-flow.mlir",
            # RFC §B: elementwise add with scf.for, linalg.add, tensor.empty
            # Requires scf.for + linalg.add + tensor.empty.
            "execute_kwargs": {},
        },
    ],
}


def get_test_params(
    *func_names: str,
    filter: str | None = None,
) -> list[tuple[str, str, dict]]:
    """Return ``(abs_path, func_name, entry)`` triples for the given function names.

    If no names are given, returns triples for all registered functions.
    If *filter* is set, only entries where the substring appears in either
    the relative path or the function name are included.

    Entries with an ``exception_msg`` field are *failure examples* and are
    skipped unless a ``filter`` string explicitly selects them.  This prevents
    failure examples from appearing in normal parametrize lists while still
    allowing dedicated failure tests to select them by path substring.
    """
    names = func_names or tuple(EXAMPLE_PARAMS.keys())
    result = []
    for fn in names:
        for entry in EXAMPLE_PARAMS[fn]:
            rel_path = entry["path"]
            is_failure = "exception_msg" in entry
            if filter is not None:
                if filter not in rel_path and filter not in fn:
                    continue
                # Filter explicitly matched — include even failure examples.
            elif is_failure:
                # No filter: skip failure examples in normal parametrize lists.
                continue
            result.append((str(EXAMPLES_DIR / rel_path), fn, entry))
    return result


# ---------------------------------------------------------------------------
# Parsed example metadata — extracted from MLIR text via regex
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExampleMeta:
    """Metadata extracted from a parsed MLIR example function."""
    grid: tuple[int, ...]
    arg_names: tuple[str, ...]
    num_args: int
    tensor_sizes: dict[str, dict]  # {arg_name: {"shape": tuple, "dtype": str}}


def _parse_mlir_meta(text: str, func_name: str) -> ExampleMeta:
    """Extract function metadata from raw MLIR text using regex.

    This is intentionally independent of the KTIRInterpreter parser so
    that tests can compare the interpreter's output against this ground
    truth.
    """
    # --- grid ---
    # e.g. ) attributes {grid = [32, 1]} {
    # if no grid attribute; default to (1, 1, 1).
    grid_match = re.search(
        r'@' + re.escape(func_name) + r'\b.*?'
        r'attributes\s*\{[^}]*grid\s*=\s*\[([^\]]+)\]',
        text, re.DOTALL,
    )
    if grid_match:
        grid = tuple(int(x) for x in grid_match.group(1).split(","))
        # Normalize to 3-tuple (x, y, z) with defaults
        assert len(grid) >= 1 and len(grid) <= 3, \
            "len(grid) should be at least 1 and at most 3"
        grid = grid + ((1, ) * (3 - len(grid)))
    else:
        grid = (1, 1, 1)  # default: single-core when no grid attribute is present

    # --- arguments ---
    # Find the function signature block: @func_name(\n  %arg: type, ...)
    # functions may have no arguments and no attributes clause.
    sig_match = re.search(
        r'@' + re.escape(func_name) + r'\s*\((.*?)\)\s*(?:->.*?)?\s*(?:attributes\b|\{)',
        text, re.DOTALL,
    )
    assert sig_match, f"No function signature found for @{func_name}"
    sig_body = sig_match.group(1)
    args = re.findall(r'(%\w+)\s*:\s*(\S+)', sig_body)
    arg_names = tuple(name for name, _ in args)

    # --- tensor_sizes from construct_memory_view ---
    # Pattern: construct_memory_view %arg_ref,\n  sizes: [d1, d2, ...], ...
    #          ... : index -> memref<SHAPExDTYPE>
    tensor_sizes: dict[str, dict] = {}
    cmv_pattern = re.compile(
        r'construct_memory_view\s+(%\w+)\s*,'
        r'\s*sizes\s*:\s*\[([^\]]+)\]'
        r'.*?'
        r'memref<([^>]+)>',
        re.DOTALL,
    )
    for m in cmv_pattern.finditer(text):
        ref_name = m.group(1)  # e.g. %x_ptr
        sizes_str = m.group(2)  # e.g. "1024" or "1823, 1024"
        memref_str = m.group(3)  # e.g. "1024xf16" or "1823x1024xf16"

        # Strip leading % for the dict key
        arg_key = ref_name.lstrip("%")
        if arg_key in tensor_sizes:
            continue  # first occurrence wins

        tokens = [t.strip() for t in sizes_str.split(",")]
        if any(not t.lstrip("-").isdigit() for t in tokens):
            # SSA names in sizes: — fall back to the concrete memref<NxMxdtype>
            memref_parts = memref_str.split("x")
            dtype = memref_parts[-1]
            shape = tuple(int(p) for p in memref_parts[:-1])
        else:
            shape = tuple(int(t) for t in tokens)
            dtype = memref_str.rsplit("x", 1)[-1]
        tensor_sizes[arg_key] = {"shape": shape, "dtype": dtype}

    return ExampleMeta(
        grid=grid,
        arg_names=arg_names,
        num_args=len(args),
        tensor_sizes=tensor_sizes,
    )


@lru_cache(maxsize=None)
def parse_example(path: str, func_name: str) -> ExampleMeta:
    """Load an MLIR file and return cached metadata for *func_name*.

    *path* is absolute.  Parses the raw MLIR text with regex — does NOT
    use KTIRInterpreter.
    """
    mlir_text = Path(path).read_text()
    return _parse_mlir_meta(mlir_text, func_name)
