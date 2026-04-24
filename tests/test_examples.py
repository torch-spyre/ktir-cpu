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
"""Tests that run the example KTIR MLIR files through the interpreter.

Tests parsing and execution of the compiler-generated KTIR examples in
examples/triton-ktir/ and examples/latency/.
"""

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter

from conftest import get_test_params, parse_example


class InterpreterTestMixin:
    """Mixin that provides _make_interp() for injecting alternative parser backends.

    Usage
    -----
    Execution test classes inherit this mixin and call ``self._make_interp()``
    instead of constructing ``KTIRInterpreter()`` directly.  To run the same
    tests against a different parser backend, subclass the execution test class
    and override ``_make_interp()``::

        class MyAdaptTests(MyBackendMixin, TestVectorAddExecution):
            pass

    where ``MyBackendMixin`` overrides ``_make_interp`` to inject the
    alternative parser.

    Parser-agnostic test hygiene
    ----------------------------
    Tests must not hardcode argument names from the MLIR source (e.g.
    ``"x_ptr"``) because different parser backends may normalise them
    differently (e.g. the MLIR bindings parser normalises to positional
    ``arg0``, ``arg1``, ...).  Instead, use ``interp.arg_names(func_name)``
    to get the names in declaration order and unpack them positionally::

        x_ptr, y_ptr, output_ptr, BLOCK_SIZE = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        (n,) = sizes[x_ptr]["shape"]
        outputs = interp.execute_function(
            func_name, **{x_ptr: x, y_ptr: y, output_ptr: out, BLOCK_SIZE: bs}
        )
        result = outputs[output_ptr]
    """

    def _make_interp(self):
        return KTIRInterpreter()


# ---------------------------------------------------------------------------
# Parsing tests — all example files must parse into valid modules
# ---------------------------------------------------------------------------

class TestExampleParsing:
    """Test that all example MLIR files parse correctly."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params())
    def test_parse_module(self, path, func_name, entry):
        """Each MLIR file should parse into a module with one function."""
        interp = KTIRInterpreter()
        interp.load(path)
        assert interp.module is not None
        assert len(interp.module.functions) >= 1

    @pytest.mark.parametrize("path,func_name,entry", get_test_params())
    def test_structure(self, path, func_name, entry):
        """Parsed function should match metadata extracted from the MLIR."""
        meta = parse_example(path, func_name)
        interp = KTIRInterpreter()
        interp.load(path)
        func = interp.module.functions[func_name]
        assert func.grid == meta.grid
        arg_names = [a[0] for a in func.arguments]
        for name in meta.arg_names:
            assert name in arg_names
        assert len(func.arguments) == meta.num_args

    @pytest.mark.parametrize("path,func_name,entry", get_test_params())
    def test_construct_memory_view_attributes(self, path, func_name, entry):
        """construct_memory_view ops should have shape, strides, memory_space, dtype."""
        meta = parse_example(path, func_name)
        assert len(meta.tensor_sizes) > 0
        for info in meta.tensor_sizes.values():
            assert "shape" in info
            assert len(info["shape"]) > 0
            assert "dtype" in info


# ---------------------------------------------------------------------------
# Execution tests — run KTIR through the interpreter and check results
# ---------------------------------------------------------------------------

class TestVectorAddExecution(InterpreterTestMixin):
    """End-to-end execution of vector_add MLIR."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_single_core(self, path, func_name, entry):
        """Run vector add on K cores, verify output = x + y."""
        interp = self._make_interp()
        interp.load(path)

        x_ptr, y_ptr, output_ptr, BLOCK_SIZE = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        (n,) = sizes[x_ptr]["shape"]
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n).astype(np.float16)
        y = rng.standard_normal(n).astype(np.float16)
        output = np.zeros(n, dtype=np.float16)

        outputs = interp.execute_function(
            func_name,
            **{x_ptr: x, y_ptr: y, output_ptr: output,
               BLOCK_SIZE: entry["execute_kwargs"]["BLOCK_SIZE"]},
        )

        result = outputs[output_ptr]
        expected = (x + y).astype(np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("add_kernel"))
    def test_various_values(self, path, func_name, entry):
        """Vector add with zeros, negatives, and large values."""
        interp = self._make_interp()
        interp.load(path)

        x_ptr, y_ptr, output_ptr, BLOCK_SIZE = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        (n,) = sizes[x_ptr]["shape"]
        x = np.zeros(n, dtype=np.float16)
        y = np.linspace(-10, 10, n, dtype=np.float16)
        output = np.zeros(n, dtype=np.float16)

        outputs = interp.execute_function(
            func_name,
            **{x_ptr: x, y_ptr: y, output_ptr: output,
               BLOCK_SIZE: entry["execute_kwargs"]["BLOCK_SIZE"]},
        )

        result = outputs[output_ptr]
        expected = (x + y).astype(np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


class TestSoftmaxExecution(InterpreterTestMixin):
    """End-to-end execution of softmax MLIR."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("softmax_kernel", filter="triton-ktir"))
    def test_softmax_correct(self, path, func_name, entry):
        """Run softmax on 32 cores, verify against NumPy ground truth."""
        interp = self._make_interp()
        interp.load(path)

        output_ptr, input_ptr, n_rows = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        n_rows_val, n_padded_cols = sizes[input_ptr]["shape"]
        rng = np.random.default_rng(42)
        n_real_cols = int(n_padded_cols * 0.76)  # test with padding: ~76% real data

        # Padded input: first n_real_cols are real data, rest is -inf
        inp = np.full((n_rows_val, n_padded_cols), float('-inf'), dtype=np.float16)
        inp[:, :n_real_cols] = rng.standard_normal(
            (n_rows_val, n_real_cols)
        ).astype(np.float16)
        out = np.zeros((n_rows_val, n_padded_cols), dtype=np.float16)

        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        kwargs["n_cols"] = n_real_cols  # fill dynamic kwarg

        outputs = interp.execute_function(
            func_name, **{**kwargs, output_ptr: out, input_ptr: inp},
        )
        result = outputs[output_ptr]

        # NumPy reference: softmax per row
        m = np.max(inp, axis=1, keepdims=True)
        e = np.exp((inp - m).astype(np.float32))
        s = np.sum(e, axis=1, keepdims=True)
        expected = (e / s).astype(np.float16)

        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params(
        "softmax_kernel", filter="ktir/softmax_wide"))
    def test_softmax_lx_overflow(self, path, func_name, entry):
        """Softmax with a row too wide for LX should raise MemoryError.

        A naive rowwise softmax loads the full row (1×C) and produces several
        intermediate Tiles of the same shape (splat, subf, exp, divf).
        With C=262144 the peak live set is ~3MB, exceeding the 2MB LX.
        This is exactly the scenario that online_rowchunk solves by
        chunking the column dimension.
        """
        interp = self._make_interp()
        interp.load(path)

        output_ptr, input_ptr, *_ = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        n_rows_val, n_cols = sizes[input_ptr]["shape"]
        rng = np.random.default_rng(42)
        inp = rng.standard_normal((n_rows_val, n_cols)).astype(np.float16)
        out = np.zeros((n_rows_val, n_cols), dtype=np.float16)

        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        with pytest.raises(MemoryError, match=entry["exception_msg"]):
            interp.execute_function(
                func_name,
                **{**kwargs, output_ptr: out, input_ptr: inp},
            )


class TestLayerNormExecution(InterpreterTestMixin):
    """End-to-end execution of layernorm_fwd_ktir.mlir."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("_layer_norm_fwd_fused"))
    def test_layernorm_32_cores(self, path, func_name, entry):
        """Run layer norm on 32 cores, verify Y and Mean."""
        interp = self._make_interp()
        interp.load(path)

        X, Y, W, B, Mean, Rstd, N, eps, BLOCK_SIZE = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        n_rows, n_cols = sizes[X]["shape"]
        rng = np.random.default_rng(42)
        X_data = rng.standard_normal((n_rows, n_cols)).astype(np.float16)
        # W and B are 1D weight/bias vectors, but the MLIR construct_memory_view
        # declares them as 2D (n_rows × n_cols) — same row tiled across all rows.
        W_1d = rng.standard_normal(n_cols).astype(np.float16)
        B_1d = rng.standard_normal(n_cols).astype(np.float16)
        W_data = np.tile(W_1d, (n_rows, 1))
        B_data = np.tile(B_1d, (n_rows, 1))
        Y_data = np.zeros((n_rows, n_cols), dtype=np.float16)
        Mean_data = np.zeros(n_rows, dtype=np.float16)
        Rstd_data = np.zeros(n_rows, dtype=np.float16)

        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        outputs = interp.execute_function(
            func_name,
            **{X: X_data, Y: Y_data, W: W_data, B: B_data,
               Mean: Mean_data, Rstd: Rstd_data,
               N: kwargs.pop("N"), eps: kwargs.pop("eps"),
               BLOCK_SIZE: kwargs.pop("BLOCK_SIZE"),
               **kwargs},  # remaining: stride (body-level)
        )
        result_Y = outputs[Y]
        result_Mean = outputs[Mean]

        # NumPy reference: standard layer norm
        X_f32 = X_data.astype(np.float32)
        W_f32 = W_1d.astype(np.float32)
        B_f32 = B_1d.astype(np.float32)
        mean_ref = np.mean(X_f32, axis=1)
        var_ref = np.var(X_f32, axis=1)
        rstd_ref = 1.0 / np.sqrt(var_ref + 1e-5)
        Y_ref = (
            (X_f32 - mean_ref[:, None]) * rstd_ref[:, None] * W_f32 + B_f32
        ).astype(np.float16)

        np.testing.assert_allclose(result_Y, Y_ref, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(
            result_Mean.astype(np.float32),
            mean_ref.astype(np.float16).astype(np.float32),
            rtol=1e-2, atol=1e-2,
        )


class TestReduceExplicitRegion(InterpreterTestMixin):
    """End-to-end execution of reduce_generic.mlir.

    Verifies that linalg.reduce in the generic MLIR format (explicit combiner
    region with block args and linalg.yield) produces the same result as the
    shorthand { arith.addf } form.
    """

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("reduce_explicit_region"))
    def test_reduce_sum(self, path, func_name, entry):
        """Reduce [1, 2, 3, 4] along dim 1 — result broadcast to [10, 10, 10, 10]."""
        interp = self._make_interp()
        interp.load(path)

        (arg0,) = interp.arg_names(func_name)
        data = np.array([[1, 2, 3, 4]], dtype=np.float16)
        outputs = interp.execute_function(func_name, **{arg0: data})
        result = outputs[arg0]

        expected = np.full((1, 4), 10.0, dtype=np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("reduce_explicit_region"))
    def test_reduce_zeros(self, path, func_name, entry):
        """Reduce all-zeros — result should be all zeros."""
        interp = self._make_interp()
        interp.load(path)

        (arg0,) = interp.arg_names(func_name)
        data = np.zeros((1, 4), dtype=np.float16)
        outputs = interp.execute_function(func_name, **{arg0: data})
        result = outputs[arg0]

        np.testing.assert_allclose(result, np.zeros((1, 4), dtype=np.float16), atol=1e-3)


class TestMatMulExecution(InterpreterTestMixin):
    """End-to-end execution of matmul_fwd_ktir.mlir."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("matmul_kernel"))
    def test_matmul(self, path, func_name, entry):
        """Run matmul on the full grid, verify C ≈ A @ B."""
        interp = self._make_interp()
        interp.load(path)

        a_ptr, b_ptr, c_ptr, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = interp.arg_names(func_name)
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        M, N, K_val = kwargs["M"], kwargs["N"], kwargs["K"]
        rng = np.random.default_rng(42)
        A = rng.standard_normal((M, K_val)).astype(np.float16)
        B = rng.standard_normal((K_val, N)).astype(np.float16)
        C = np.zeros((M, N), dtype=np.float16)

        outputs = interp.execute_function(
            func_name,
            **{a_ptr: A, b_ptr: B, c_ptr: C,
               K: kwargs.pop("K"),
               BLOCK_SIZE_M: kwargs.pop("BLOCK_SIZE_M"),
               BLOCK_SIZE_N: kwargs.pop("BLOCK_SIZE_N"),
               BLOCK_SIZE_K: kwargs.pop("BLOCK_SIZE_K"),
               **kwargs},  # remaining: M, N, stride_* (body-level)
        )
        result_C = outputs[c_ptr]

        expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        # K=2048 with BLOCK_SIZE_K=128 → 16 accumulation iterations in f16;
        # rounding error grows with K, so tolerance is relaxed accordingly.
        np.testing.assert_allclose(result_C, expected, rtol=2e-2, atol=2e-1)


class TestIndexedAddExecution(InterpreterTestMixin):
    """End-to-end execution of indexed_add.mlir."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("indexed_add_kernel"))
    def test_indexed_add(self, path, func_name, entry):
        """Run indexed_add with indirect x access, verify output = x[index[grid0]] + y."""
        interp = self._make_interp()
        interp.load(path)

        x_ptr, y_ptr, index_ptr, output_ptr, dim1_start = interp.arg_names(func_name)
        rng = np.random.default_rng(0)
        # x: [128, 64, 8, 128], y: [2, 32, 8, 128], index: [2] (i64), output: [2, 32, 8, 128]
        x = rng.standard_normal((128, 64, 8, 128)).astype(np.float16)
        y = rng.standard_normal((2, 32, 8, 128)).astype(np.float16)
        index = np.array([3, 7], dtype=np.int64)
        output = np.zeros((2, 32, 8, 128), dtype=np.float16)
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        dim1_start_val = kwargs.get("dim1_start", 0)
        outputs = interp.execute_function(
            func_name,
            **{x_ptr: x, y_ptr: y, index_ptr: index,
               output_ptr: output, dim1_start: dim1_start_val},
        )
        result = outputs[output_ptr]
        assert result.shape == (2, 32, 8, 128)
        assert not np.any(np.isnan(result)), "output contains NaN"
        assert not np.any(np.isinf(result)), "output contains Inf"

        # NumPy reference:
        #   For each grid0 ∈ {0,1}, grid1 ∈ {0..7}:
        #     x_row = index[grid0]
        #     output[grid0, :, grid1, :] = x[x_row, 0:32, grid1, :] + y[grid0, :, grid1, :]
        x_rows = x[index.astype(np.intp)]  # (2, 64, 8, 128)
        expected = (x_rows[:, dim1_start_val:dim1_start_val + 32, :, :] + y).astype(np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


class TestPagedAttentionExecution(InterpreterTestMixin):
    """End-to-end execution of paged_attention.mlir."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("kernel_unified_attention_spyre_2d"))
    def test_paged_attention(self, path, func_name, entry):
        """Run paged attention with indirect access via block_tables.

        Verifies shape, finiteness, and checks the first query block
        (pid0=0, pid1=0) against a NumPy reference.
        """
        interp = self._make_interp()
        interp.load(path)
        (output_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
         block_tables_ptr, cur_batch_start_index, block_table_offset,
         num_tiles, context_len, scale) = interp.arg_names(func_name)
        rng = np.random.default_rng(42)
        query = rng.standard_normal((8, 32, 128)).astype(np.float16)
        key_cache = rng.standard_normal((64, 16, 8, 128)).astype(np.float16)
        value_cache = rng.standard_normal((64, 16, 8, 128)).astype(np.float16)
        block_tables = rng.integers(0, 64, size=(1, 16), dtype=np.int32)
        output = np.zeros((8, 32, 128), dtype=np.float16)
        kwargs = {k: v for k, v in entry["execute_kwargs"].items() if v is not None}
        scale_val = kwargs.pop("scale")
        num_tiles_val = kwargs.pop("num_tiles")
        context_len_val = kwargs.pop("context_len")
        cur_batch_start_index_val = kwargs.pop("cur_batch_start_index", 0)
        block_table_offset_val = kwargs.pop("block_table_offset", 0)
        outputs = interp.execute_function(
            func_name,
            **{output_ptr: output, query_ptr: query,
               key_cache_ptr: key_cache, value_cache_ptr: value_cache,
               block_tables_ptr: block_tables,
               cur_batch_start_index: cur_batch_start_index_val,
               block_table_offset: block_table_offset_val,
               scale: scale_val, num_tiles: num_tiles_val, context_len: context_len_val,
               **kwargs},
        )
        result = outputs[output_ptr]
        assert result.shape == (8, 32, 128)
        assert not np.any(np.isnan(result)), "output contains NaN"
        assert not np.any(np.isinf(result)), "output contains Inf"
        assert not np.all(result == 0), "output is all zeros"

        # NumPy reference for first block (pid0=0, pid1=0):
        #   Q = query[0:2, 0:4, :] collapsed to (8, 128)
        #   For each tile j: K = key_cache[block_tables[0,j], :, 0, :] → (16, 128)
        #                    V = value_cache[block_tables[0,j], :, 0, :] → (16, 128)
        #   Causal mask: seq_offset(j,col) = j*16+col; mask where > context_len + query_pos
        #   output = softmax(Q @ K^T * scale, masked) @ V, then reshape to (2, 4, 128)
        pid0 = 0
        Q = query[0:2, 0:4, :].reshape(8, 128).astype(np.float32)
        K_full = np.concatenate([
            key_cache[block_tables[0, j], :, 0, :] for j in range(num_tiles_val)
        ], axis=0).astype(np.float32)  # (num_tiles*16, 128)
        V_full = np.concatenate([
            value_cache[block_tables[0, j], :, 0, :] for j in range(num_tiles_val)
        ], axis=0).astype(np.float32)

        scores = Q @ K_full.T * scale_val  # (8, num_tiles*16)

        # Causal mask: query_pos[row] = pid0*2 + row//4
        #              query_abs_pos = context_len + query_pos
        #              seq_offset[col] = col  (flattened across tiles)
        for row in range(8):
            query_abs_pos = context_len_val + pid0 * 2 + row // 4
            for col in range(num_tiles_val * 16):
                if col > query_abs_pos:
                    scores[row, col] = -np.inf

        # Tiled online-softmax reference (mirrors the kernel's loop exactly).
        M_ref = np.full(8, -np.inf, dtype=np.float32)
        L_ref = np.ones(8, dtype=np.float32)
        acc_ref = np.zeros((8, 128), dtype=np.float32)
        for j in range(num_tiles_val):
            S_j = scores[:, j*16:(j+1)*16]
            m_j = np.maximum(M_ref, S_j.max(axis=1))
            P_j = np.exp(S_j - m_j[:, None])
            l_j = P_j.sum(axis=1)
            alpha = np.exp(M_ref - m_j)
            V_j = V_full[j*16:(j+1)*16]
            acc_ref = acc_ref * alpha[:, None] + P_j @ V_j
            L_ref = L_ref * alpha + l_j
            M_ref = m_j
        expected = (acc_ref / L_ref[:, None]).reshape(2, 4, 128).astype(np.float16)

        actual = result[0:2, 0:4, :]
        np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)


class TestSdpaExecution(InterpreterTestMixin):
    """End-to-end execution of sdpa_2d.mlir."""

    @pytest.mark.parametrize("path,func_name,entry", get_test_params("sdpa_kernel_2d"))
    def test_sdpa(self, path, func_name, entry):
        """Run SDPA on 1 core, verify output ≈ softmax(Q @ K^T * scale) @ V."""
        interp = self._make_interp()
        interp.load(path)

        q_ptr, k_ptr, v_ptr, output_ptr = interp.arg_names(func_name)
        sizes = interp.tensor_input_output_sizes(func_name)
        n_rows, head_dim = sizes[q_ptr]["shape"]
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((n_rows, head_dim)).astype(np.float16)
        K = rng.standard_normal((n_rows, head_dim)).astype(np.float16)
        V = rng.standard_normal((n_rows, head_dim)).astype(np.float16)
        output = np.zeros((n_rows, head_dim), dtype=np.float16)

        outputs = interp.execute_function(
            func_name, **{q_ptr: Q, k_ptr: K, v_ptr: V, output_ptr: output},
        )
        result = outputs[output_ptr]

        # NumPy reference: scaled dot-product attention (f32 for stability)
        scale = np.float32(0.125)  # 1/sqrt(64), matches MLIR constant
        QK = Q.astype(np.float32) @ K.astype(np.float32).T  # [n_rows, n_rows]
        QK_scaled = QK * scale
        m = np.max(QK_scaled, axis=1, keepdims=True)
        P = np.exp(QK_scaled - m)
        P_norm = (P / np.sum(P, axis=1, keepdims=True)).astype(np.float16)
        expected = (P_norm.astype(np.float32) @ V.astype(np.float32)).astype(np.float16)

        # Two f16 matmuls accumulate rounding error; tolerance matched to matmul test.
        np.testing.assert_allclose(result, expected, rtol=2e-2, atol=2e-1)
