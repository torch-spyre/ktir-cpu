# Copyright 2026 The Torch-Spyre Authors.
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

"""
Spyreop compute helpers — the numerical reference for Spyre hardware
intrinsics reachable from Triton's ``tl.spyre_op``/``tts.spyre_op``
(see torch-spyre/triton's ``spyreop-frontend-api.md``).

Pure element-wise math on Tiles and scalars, used by the ``spyreop.*``
dialect handlers in ``ktir_cpu.dialects.spyreop_ops``. Every op here is
the scalar form the real ``spyreop.*`` MLIR op declares (one call per
element/stick); the dialect layer is what runs them vectorised over a
whole Tile, same as ``math_ops.py``'s transcendentals.

Numerics are the ordinary closed-form definitions, not a hardware
microcode transliteration: ``gelu`` is the tanh-based approximation —
matching both the DDL templates and ``gelu.smc`` in Deeptools, which use
the same tanh approximation as PyTorch's ``F.gelu(x, approximate="tanh")``
— ``softplus`` matches PyTorch's ``F.softplus(x, beta, threshold)``, and
the LayerNorm triple (``exx2``/``layernormscale``/``layernormnorm``)
matches the standard E[x]/E[x^2] formulation. All four do their
intermediate math in float32 regardless of the Tile's own dtype, then
cast back — the same convention ``MathOps.erf``/``ArithOps``'s FMA-style
helpers already use. This is a choice of this reference implementation,
not a hardware constraint: the real ``.smc`` templates select FP16 or
FP32 arithmetic externally, and compute in FP16 when given FP16 inputs.
"""

import numpy as np

from ..ir_types import Tile
from ._helpers import tile_unary_float


class SpyreOps:
    """Spyre hardware intrinsics: scalar/Tile element-wise math."""

    # ------------------------------------------------------------------
    # Unary activations (gelu, silu, reciprocal, idx32toaddr's building
    # blocks are elsewhere; these are the pure-math unary intrinsics)
    # ------------------------------------------------------------------

    @staticmethod
    def gelu(val):
        """Tanh-approximate GeLU, matching PyTorch's
        ``F.gelu(x, approximate="tanh")``:
        ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

        Spyre's ``gelu.smc`` template (and the corresponding DDL templates
        in Deeptools) use this same tanh approximation, not the exact
        erf-based formula.

        F16/DF16 only per ``SpyreOp.td`` (no ``spyreop.gelu`` for F32);
        this helper itself is dtype-agnostic — the caller's Tile dtype
        decides what gets cast back.
        """
        def _gelu(x):
            xf = x.astype(np.float32)
            inner = np.float32(0.7978845608028654) * (
                xf + np.float32(0.044715) * xf * xf * xf)
            return np.float32(0.5) * xf * (np.float32(1.0) + np.tanh(inner))

        return tile_unary_float(_gelu, val)

    @staticmethod
    def silu(val):
        """SiLU / Swish: ``x * sigmoid(x)``."""
        def _silu(x):
            xf = x.astype(np.float32)
            sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-xf))
            return xf * sig

        return tile_unary_float(_silu, val)

    @staticmethod
    def reciprocal(val):
        """Multiplicative inverse: ``1 / x``.

        The real op trades exactness for speed on hardware; this
        reference computes the exact value, which is what every other
        ``spyreop.*`` numerical reference in this module does too (the
        interpreter models the intrinsic's *result*, not its
        approximation error).
        """
        return tile_unary_float(lambda x: np.float32(1.0) / x.astype(np.float32), val)

    # ------------------------------------------------------------------
    # Softplus (one operand, two compile-time F32 attributes)
    # ------------------------------------------------------------------

    @staticmethod
    def softplus(val, beta, threshold):
        """``F.softplus(x, beta, threshold)``:

        ``(1/beta) * log(1 + exp(beta*x))``, reverting to the identity
        ``x`` wherever ``beta*x > threshold`` for numerical stability
        (matches PyTorch's own definition, which this op's hardware
        template implements per ``spyreop-fused-ops-table.md``).
        """
        beta = np.float32(beta)
        threshold = np.float32(threshold)

        def _softplus(x):
            xf = x.astype(np.float32)
            bx = beta * xf
            above = bx > threshold
            # Clip before exp: a lane already routed to the identity branch
            # by np.where below would otherwise still overflow computing the
            # unused soft branch for it (bx here can be far above threshold).
            safe_bx = np.where(above, np.float32(0.0), bx)
            soft = (np.float32(1.0) / beta) * np.log1p(np.exp(safe_bx))
            return np.where(above, xf, soft)

        return tile_unary_float(_softplus, val)

    # ------------------------------------------------------------------
    # Address arithmetic (plain int32/int64 add/multiply, and the
    # idx32toaddr address-generation intrinsic)
    # ------------------------------------------------------------------

    @staticmethod
    def addi32toi32(lhs, rhs):
        """32-bit integer add, address-arithmetic entry point.

        Ordinary ``int32`` wraparound addition — the real op's hardware
        template splits the operand into bytes/halfwords and recombines
        them, which is exactly 32-bit modular addition (confirmed against
        ``addi32toi32.smc``'s halfword-carry construction).
        """
        return SpyreOps._int_add(lhs, rhs, np.int32)

    @staticmethod
    def addi64toi64(lhs, rhs):
        """64-bit integer add, address-arithmetic entry point."""
        return SpyreOps._int_add(lhs, rhs, np.int64)

    @staticmethod
    def muli32toi32(lhs, rhs):
        """32-bit integer multiply, address-arithmetic entry point."""
        return SpyreOps._int_mul(lhs, rhs, np.int32)

    @staticmethod
    def idx32toaddr(index, base, stride):
        """Address generation: ``base + stride * index`` (32-bit, wrapping).

        Per ``SpyreOp.td``'s own summary ("Generates an address via
        `b + (s*i)`") and confirmed against ``idx32toaddr.smc``'s
        byte-split multiply-add construction.
        """
        def _compute(i, b, s):
            return (b.astype(np.int64) + s.astype(np.int64) * i.astype(np.int64)).astype(np.int32)

        if isinstance(index, Tile) or isinstance(base, Tile) or isinstance(stride, Tile):
            shape = next(v.shape for v in (index, base, stride) if isinstance(v, Tile))
            dtype = next(v.dtype for v in (index, base, stride) if isinstance(v, Tile))
            i_data = index.data if isinstance(index, Tile) else np.int32(index)
            b_data = base.data if isinstance(base, Tile) else np.int32(base)
            s_data = stride.data if isinstance(stride, Tile) else np.int32(stride)
            return Tile(_compute(i_data, b_data, s_data), dtype, shape)
        return int(_compute(np.int32(index), np.int32(base), np.int32(stride)))

    @staticmethod
    def _int_add(lhs, rhs, np_dtype):
        if isinstance(lhs, Tile) or isinstance(rhs, Tile):
            a = lhs.data if isinstance(lhs, Tile) else np_dtype(lhs)
            b = rhs.data if isinstance(rhs, Tile) else np_dtype(rhs)
            dtype = lhs.dtype if isinstance(lhs, Tile) else rhs.dtype
            shape = lhs.shape if isinstance(lhs, Tile) else rhs.shape
            return Tile((a.astype(np_dtype) + b.astype(np_dtype)).astype(np_dtype), dtype, shape)
        return int(np_dtype(np_dtype(lhs) + np_dtype(rhs)))

    @staticmethod
    def _int_mul(lhs, rhs, np_dtype):
        if isinstance(lhs, Tile) or isinstance(rhs, Tile):
            a = lhs.data if isinstance(lhs, Tile) else np_dtype(lhs)
            b = rhs.data if isinstance(rhs, Tile) else np_dtype(rhs)
            dtype = lhs.dtype if isinstance(lhs, Tile) else rhs.dtype
            shape = lhs.shape if isinstance(lhs, Tile) else rhs.shape
            return Tile((a.astype(np_dtype) * b.astype(np_dtype)).astype(np_dtype), dtype, shape)
        return int(np_dtype(np_dtype(lhs) * np_dtype(rhs)))

    # ------------------------------------------------------------------
    # LayerNorm's three stages (exx2, layernormscale, layernormnorm) --
    # see spyreop-frontend-api.md's "LayerNorm: a three-stage pattern".
    # ------------------------------------------------------------------

    @staticmethod
    def exx2(val, axis):
        """LayerNorm stage 0: ``(mean, mean-of-squares)`` of ``val`` along ``axis``.

        Reduction-shaped: the ``linalg.generic`` body ``LowerComputeOps``'s
        Group D2 builds gathers the un-reduced input to the whole
        iteration shape first (see ``ktir_cpu.dialects.spyreop_ops``'s
        ``spyreop__exx2``, which resolves ``axis`` from the generic's own
        ``iterator_types`` the same way ``linalg.index`` resolves its
        shape), so this helper reduces exactly the one axis the caller
        names, not an assumed one.
        """
        data = val.data.astype(np.float32) if isinstance(val, Tile) else np.float32(val)
        mean = data.mean(axis=axis)
        msq = (data * data).mean(axis=axis)
        if isinstance(val, Tile):
            out_np_dtype = val.data.dtype
            return (
                Tile(mean.astype(out_np_dtype), val.dtype, mean.shape),
                Tile(msq.astype(out_np_dtype), val.dtype, msq.shape),
            )
        return float(mean), float(msq)

    @staticmethod
    def layernormscale(mean, msq):
        """LayerNorm stage 1: ``1 / sqrt(E[x^2] - E[x]^2 + eps)``.

        Takes exactly the ``(mean, msq)`` pair ``exx2`` produces -- no
        attributes (confirmed against ``SpyreOp.td``: ``layernormscale``
        declares none) -- so the epsilon is a small fixed constant baked
        into the reference the same way it is baked into the real
        hardware template (``layernormscale.smc``'s own constant
        registers), not a frontend-supplied value.
        """
        eps = np.float32(1e-5)

        def _scale(m, sq):
            mf = m.astype(np.float32) if isinstance(m, np.ndarray) else np.float32(m)
            sqf = sq.astype(np.float32) if isinstance(sq, np.ndarray) else np.float32(sq)
            var = sqf - mf * mf
            return np.float32(1.0) / np.sqrt(var + eps)

        if isinstance(mean, Tile) or isinstance(msq, Tile):
            m = mean.data if isinstance(mean, Tile) else np.float32(mean)
            sq = msq.data if isinstance(msq, Tile) else np.float32(msq)
            dtype = mean.dtype if isinstance(mean, Tile) else msq.dtype
            shape = mean.shape if isinstance(mean, Tile) else msq.shape
            result = _scale(m, sq)
            out_np_dtype = mean.data.dtype if isinstance(mean, Tile) else msq.data.dtype
            return Tile(result.astype(out_np_dtype), dtype, shape)
        return float(_scale(mean, msq))

    @staticmethod
    def layernormnorm(x, mean, scale, weight, bias):
        """LayerNorm stage 2: ``(x - mean) * scale * weight + bias``.

        ``scale`` is ``layernormscale``'s ``1/sqrt(var+eps)`` output, not
        a raw standard deviation -- this stage only multiplies by it, it
        never derives it.
        """
        def _norm(xv, m, s, w, b):
            return ((xv.astype(np.float32) - m.astype(np.float32))
                    * s.astype(np.float32) * w.astype(np.float32)
                    + b.astype(np.float32))

        operands = [x, mean, scale, weight, bias]
        tiles = [o for o in operands if isinstance(o, Tile)]
        if tiles:
            dtype = tiles[0].dtype
            shape = tiles[0].shape
            out_np_dtype = tiles[0].data.dtype
            arrs = [o.data if isinstance(o, Tile) else np.float32(o) for o in operands]
            result = _norm(*arrs)
            return Tile(result.astype(out_np_dtype), dtype, shape)
        result = _norm(*(np.float32(o) for o in operands))
        return float(result)
