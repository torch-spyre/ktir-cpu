"""MLIR text generators for KTIR demo kernels.

Each gen_* function returns a complete MLIR module string ready for
KTIRInterpreter.load(). Private helpers (_mem_view, _access_tile,
_indirect_kv_tile) factor out repeated boilerplate.
"""

from __future__ import annotations

from ktir_cpu.memory import HBMSimulator

# HBM traffic is charged per unique stick, not per packed byte, so a tile row
# narrower than one stick is billed as a full one.  Taken from the simulator so
# the two cannot drift apart.
_STICK_BYTES = HBMSimulator.STICK_BYTES


# ---------------------------------------------------------------------------
# Private MLIR-string helpers
# ---------------------------------------------------------------------------

def _require_stick_aligned(what: str, n_elems: int) -> None:
    """Reject a tile row that is not a whole number of HBM sticks.

    Every generator in this module emits f16, so the element size is fixed at 2
    bytes rather than taken as a parameter.

    Raised rather than asserted: these are caller-supplied shapes, and ``python
    -O`` strips asserts, which would let a mis-shaped tile through silently and
    report an arithmetic intensity that belongs to the padding rather than to
    the kernel.
    """
    row_bytes = n_elems * 2
    if row_bytes % _STICK_BYTES:
        raise ValueError(
            f"{what} is {n_elems} elements = {row_bytes} B, not a whole "
            f"{_STICK_BYTES} B stick; HBM traffic would be charged for the "
            f"padding and the reported arithmetic intensity would not be the "
            f"kernel's own"
        )


def _mem_view(name: str, ptr: str, shape: list[int], strides: list[int],
              dtype: str = "f16") -> str:
    """Generate MLIR for ktdp.construct_memory_view."""
    dims = [f"d{i}" for i in range(len(shape))]
    bounds = ", ".join(
        f"{d} >= 0, -{d} + {s-1} >= 0" for d, s in zip(dims, shape)
    )
    dim_decl = ", ".join(dims)
    sizes_str = ", ".join(str(s) for s in shape)
    strides_str = ", ".join(str(s) for s in strides)
    shape_str = "x".join(str(s) for s in shape)
    return (
        f"    %{name} = ktdp.construct_memory_view %{ptr}, "
        f"sizes: [{sizes_str}], strides: [{strides_str}] {{\n"
        f"      coordinate_set = affine_set<({dim_decl}) : ({bounds})>,\n"
        f"      memory_space = #ktdp.spyre_memory_space<HBM>\n"
        f"    }} : memref<{shape_str}x{dtype}>"
    )


def _access_tile(name: str, view: str, offsets: list[str],
                 tile_shape: list[int], view_shape: list[int],
                 dtype: str = "f16") -> str:
    """Generate MLIR for ktdp.construct_access_tile."""
    dims = [f"d{i}" for i in range(len(tile_shape))]
    bounds = ", ".join(
        f"{d} >= 0, -{d} + {s-1} >= 0" for d, s in zip(dims, tile_shape)
    )
    dim_decl = ", ".join(dims)
    offsets_str = ", ".join(offsets)
    tile_str = "x".join(str(s) for s in tile_shape)
    view_str = "x".join(str(s) for s in view_shape)
    order_map = f"({dim_decl}) -> ({dim_decl})"
    return (
        f"    %{name} = ktdp.construct_access_tile %{view}[{offsets_str}] {{\n"
        f"      access_tile_set = affine_set<({dim_decl}) : ({bounds})>,\n"
        f"      access_tile_order = affine_map<{order_map}>\n"
        f"    }} : memref<{view_str}x{dtype}> -> !ktdp.access_tile<{tile_str}xindex>"
    )


def _indirect_kv_tile(name: str, cache_view: str, bt_view: str,
                      bt_idx: str, pid_kv: str,
                      nb: int, bs: int, num_kv_heads: int, hd: int,
                      mbps: int) -> str:
    """Generate MLIR for ktdp.construct_indirect_access_tile (KV cache page lookup)."""
    return (
        f"      %{name} = ktdp.construct_indirect_access_tile\n"
        f"          intermediate_variables(%d0, %d1, %d2, %d3)\n"
        f"          %{cache_view}[ind(%{bt_view}[%c0, {bt_idx} + %d0]),\n"
        f"                        (%d1), ({pid_kv} + %d2), (%d3)] {{\n"
        f"        variables_space_set   = affine_set<(d0, d1, d2, d3) : (\n"
        f"            d0 >= 0, -d0 + 0 >= 0,\n"
        f"            d1 >= 0, -d1 + {bs-1} >= 0,\n"
        f"            d2 >= 0, -d2 + 0 >= 0,\n"
        f"            d3 >= 0, -d3 + {hd-1} >= 0)>,\n"
        f"        variables_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>\n"
        f"      }} : memref<{nb}x{bs}x{num_kv_heads}x{hd}xf16>, memref<1x{mbps}xi32>\n"
        f"          -> !ktdp.access_tile<1x{bs}x1x{hd}xindex>"
    )


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def gen_matmul_mlir(M, N, K, bm, bn, bk):
    """Generate a tiled matmul MLIR kernel with grid = [M/bm, N/bn]."""
    gm, gn = M // bm, N // bn
    a_view = _mem_view("a_view", "a_ptr", [M, K], [K, 1])
    b_view = _mem_view("b_view", "b_ptr", [K, N], [N, 1])
    c_view = _mem_view("c_view", "c_ptr", [M, N], [N, 1])
    a_acc = _access_tile("a_acc", "a_view", ["%offs_am", "%off_k"],
                         [bm, bk], [M, K])
    b_acc = _access_tile("b_acc", "b_view", ["%off_k", "%offs_bn"],
                         [bk, bn], [K, N])
    c_acc = _access_tile("c_acc", "c_view", ["%offs_am", "%offs_bn"],
                         [bm, bn], [M, N])
    return f"""module {{
  func.func @matmul_kernel(
      %a_ptr: index, %b_ptr: index, %c_ptr: index,
      %K: index, %BLOCK_SIZE_M: index, %BLOCK_SIZE_N: index, %BLOCK_SIZE_K: index
  ) attributes {{grid = [{gm}, {gn}]}} {{
    %pid_m, %pid_n = ktdp.get_compute_tile_id : index, index

{a_view}

{b_view}

{c_view}

    %offs_am = arith.muli %pid_m, %BLOCK_SIZE_M : index
    %offs_bn = arith.muli %pid_n, %BLOCK_SIZE_N : index
    %accum_zero = arith.constant dense<0.0> : tensor<{bm}x{bn}xf16>

    %c0 = arith.constant 0 : index
    %c = scf.for %off_k = %c0 to %K step %BLOCK_SIZE_K iter_args(%acc = %accum_zero) -> (tensor<{bm}x{bn}xf16>) {{
  {a_acc}

  {b_acc}

      %a = ktdp.load %a_acc : !ktdp.access_tile<{bm}x{bk}xindex> -> tensor<{bm}x{bk}xf16>
      %b = ktdp.load %b_acc : !ktdp.access_tile<{bk}x{bn}xindex> -> tensor<{bk}x{bn}xf16>

      %c_init = tensor.empty() : tensor<{bm}x{bn}xf16>
      %ab = linalg.matmul ins(%a, %b : tensor<{bm}x{bk}xf16>, tensor<{bk}x{bn}xf16>)
                          outs(%c_init : tensor<{bm}x{bn}xf16>) -> tensor<{bm}x{bn}xf16>
      %next = arith.addf %acc, %ab : tensor<{bm}x{bn}xf16>
      scf.yield %next : tensor<{bm}x{bn}xf16>
    }}

{c_acc}
    ktdp.store %c, %c_acc : tensor<{bm}x{bn}xf16>, !ktdp.access_tile<{bm}x{bn}xindex>
    return
  }}
}}"""


def gen_softmax_mlir(n_rows, row_width, num_cores=32):
    """Generate a row-wise softmax kernel. Grid = [num_cores, 1]."""
    rw = row_width
    input_view = _mem_view("input_view", "input_ptr", [n_rows, rw], [rw, 1])
    output_view = _mem_view("output_view", "output_ptr", [n_rows, rw], [rw, 1])
    input_acc = _access_tile("input_acc", "input_view", ["%row", "%c0"],
                             [1, rw], [n_rows, rw])
    output_acc = _access_tile("output_acc", "output_view", ["%row", "%c0"],
                              [1, rw], [n_rows, rw])
    return f"""module {{
  func.func @softmax_kernel(
      %output_ptr: index, %input_ptr: index, %n_rows: index
  ) attributes {{grid = [{num_cores}, 1]}} {{
    %core_id = ktdp.get_compute_tile_id : index
    %step = arith.constant {num_cores} : index
    %c0 = arith.constant 0 : index

{input_view}

{output_view}

    scf.for %row = %core_id to %n_rows step %step : index {{
  {input_acc}

      %input_row = ktdp.load %input_acc : !ktdp.access_tile<1x{rw}xindex> -> tensor<1x{rw}xf16>

      %neg_inf = arith.constant 0xFC00 : f16
      %max_init = tensor.splat %neg_inf : tensor<1xf16>
      %reduce_max = linalg.reduce {{ arith.maximumf }}
        ins(%input_row : tensor<1x{rw}xf16>)
        outs(%max_init : tensor<1xf16>)
        dimensions = [1]

      %c0_ext = arith.constant 0 : index
      %max_scalar = tensor.extract %reduce_max[%c0_ext] : tensor<1xf16>
      %max_row = tensor.splat %max_scalar : tensor<1x{rw}xf16>
      %input_minus_max = arith.subf %input_row, %max_row : tensor<1x{rw}xf16>
      %numerator = math.exp %input_minus_max : tensor<1x{rw}xf16>

      %zero = arith.constant 0.0 : f16
      %sum_init = tensor.splat %zero : tensor<1xf16>
      %reduce_add = linalg.reduce {{ arith.addf }}
        ins(%numerator : tensor<1x{rw}xf16>)
        outs(%sum_init : tensor<1xf16>)
        dimensions = [1]

      %denom_scalar = tensor.extract %reduce_add[%c0_ext] : tensor<1xf16>
      %denominator_row = tensor.splat %denom_scalar : tensor<1x{rw}xf16>
      %softmax_output = arith.divf %numerator, %denominator_row : tensor<1x{rw}xf16>

  {output_acc}

      ktdp.store %softmax_output, %output_acc : tensor<1x{rw}xf16>, !ktdp.access_tile<1x{rw}xindex>
      scf.yield
    }}
    return
  }}
}}"""


def gen_sdpa_mlir(seq_len, head_dim, block_m):
    """Generate a naive fused SDPA kernel (single-pass, no K-tiling).

    Algorithm: Output = softmax(Q @ K^T / sqrt(head_dim)) @ V
    Grid = [seq_len // block_m] -- one core per query block.
    """
    grid = seq_len // block_m
    scale = 1.0 / (head_dim ** 0.5)
    sl = seq_len
    hd = head_dim
    bm = block_m

    q_view = _mem_view("q_view", "q_ptr", [sl, hd], [hd, 1])
    k_view = _mem_view("k_view", "k_ptr", [sl, hd], [hd, 1])
    v_view = _mem_view("v_view", "v_ptr", [sl, hd], [hd, 1])
    output_view = _mem_view("output_view", "output_ptr", [sl, hd], [hd, 1])

    q_tile = _access_tile("q_tile", "q_view", ["%pid_m", "%c0"],
                          [bm, hd], [sl, hd])
    k_tile = _access_tile("k_tile", "k_view", ["%c0", "%c0"],
                          [sl, hd], [sl, hd])
    v_tile = _access_tile("v_tile", "v_view", ["%c0", "%c0"],
                          [sl, hd], [sl, hd])
    output_tile = _access_tile("output_tile", "output_view", ["%pid_m", "%c0"],
                               [bm, hd], [sl, hd])

    return f"""module {{
  func.func @sdpa_kernel(
    %q_ptr: index, %k_ptr: index, %v_ptr: index, %output_ptr: index
  ) attributes {{grid = [{grid}]}} {{
    %pid_m = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %zero_f16 = arith.constant 0.0 : f16
    %scale = arith.constant {scale:.10e} : f16
    %neg_inf = arith.constant 0xFC00 : f16

{q_view}
{q_tile}
    %q = ktdp.load %q_tile : !ktdp.access_tile<{bm}x{hd}xindex> -> tensor<{bm}x{hd}xf16>

{k_view}
{k_tile}
    %k = ktdp.load %k_tile : !ktdp.access_tile<{sl}x{hd}xindex> -> tensor<{sl}x{hd}xf16>

    %k_t_init = tensor.empty() : tensor<{hd}x{sl}xf16>
    %k_t = linalg.transpose ins(%k : tensor<{sl}x{hd}xf16>)
                            outs(%k_t_init : tensor<{hd}x{sl}xf16>)
                            permutation = [1, 0]
    %qk_init = tensor.empty() : tensor<{bm}x{sl}xf16>
    %qk = linalg.matmul ins(%q, %k_t : tensor<{bm}x{hd}xf16>, tensor<{hd}x{sl}xf16>)
                        outs(%qk_init : tensor<{bm}x{sl}xf16>) -> tensor<{bm}x{sl}xf16>

    %scale_splat = tensor.splat %scale : tensor<{bm}x{sl}xf16>
    %qk_scaled = arith.mulf %qk, %scale_splat : tensor<{bm}x{sl}xf16>

    %m_i_init = tensor.empty() : tensor<{bm}xf16>
    %m_i_neginf = linalg.fill ins(%neg_inf : f16) outs(%m_i_init : tensor<{bm}xf16>) -> tensor<{bm}xf16>
    %m_i = linalg.reduce {{ arith.maximumf }}
             ins(%qk_scaled : tensor<{bm}x{sl}xf16>)
             outs(%m_i_neginf : tensor<{bm}xf16>)
             dimensions = [1]

    %m_i_bcast_init = tensor.empty() : tensor<{bm}x{sl}xf16>
    %m_i_bcast = linalg.broadcast ins(%m_i : tensor<{bm}xf16>)
                                   outs(%m_i_bcast_init : tensor<{bm}x{sl}xf16>)
                                   dimensions = [1]
    %qk_shifted = arith.subf %qk_scaled, %m_i_bcast : tensor<{bm}x{sl}xf16>
    %p = math.exp %qk_shifted : tensor<{bm}x{sl}xf16>

    %l_i_init = tensor.empty() : tensor<{bm}xf16>
    %l_i_zeros = linalg.fill ins(%zero_f16 : f16) outs(%l_i_init : tensor<{bm}xf16>) -> tensor<{bm}xf16>
    %l_i = linalg.reduce {{ arith.addf }}
             ins(%p : tensor<{bm}x{sl}xf16>)
             outs(%l_i_zeros : tensor<{bm}xf16>)
             dimensions = [1]

    %l_i_bcast_init = tensor.empty() : tensor<{bm}x{sl}xf16>
    %l_i_bcast = linalg.broadcast ins(%l_i : tensor<{bm}xf16>)
                                   outs(%l_i_bcast_init : tensor<{bm}x{sl}xf16>)
                                   dimensions = [1]
    %p_norm = arith.divf %p, %l_i_bcast : tensor<{bm}x{sl}xf16>

{v_view}
{v_tile}
    %v = ktdp.load %v_tile : !ktdp.access_tile<{sl}x{hd}xindex> -> tensor<{sl}x{hd}xf16>

    %acc_init = tensor.empty() : tensor<{bm}x{hd}xf16>
    %acc_zeros = linalg.fill ins(%zero_f16 : f16) outs(%acc_init : tensor<{bm}x{hd}xf16>) -> tensor<{bm}x{hd}xf16>
    %acc = linalg.matmul ins(%p_norm, %v : tensor<{bm}x{sl}xf16>, tensor<{sl}x{hd}xf16>)
                         outs(%acc_zeros : tensor<{bm}x{hd}xf16>) -> tensor<{bm}x{hd}xf16>

{output_view}
{output_tile}
    ktdp.store %acc, %output_tile : tensor<{bm}x{hd}xf16>, !ktdp.access_tile<{bm}x{hd}xindex>
    return
  }}
}}"""


def gen_sdpa_decode_pv_mlir(kv_len=8192, head_dim=128, q_per_kv=4,
                            out_split=2, k_split=16):
    """Generate the decode-attention P@V kernel, split across cores on both axes.

    Single-stream decode over one KV head's group: the ``q_per_kv`` query heads
    that share a key/value head each contribute one row of attention weights over
    the same ``kv_len`` cached tokens, so the P@V stage is

        ``C[q_per_kv, head_dim] = A[q_per_kv, kv_len] @ B[kv_len, head_dim]``

    with ``A`` the post-softmax weights and ``B`` the shared V.  ``q_per_kv`` is
    the grouped-query ratio: 4 for Granite-8B, which has 32 query heads over 8
    key/value heads at ``head_dim = 128`` (32 x 128 = 4096 hidden).  This is the
    P@V stage only -- no QK^T and no softmax; ``gen_sdpa_mlir`` covers the fused
    prefill shape.  Decode is the shape that has to cross cores: a prefill shape's
    parallel dimensions fill the grid on their own, while a single decode step
    leaves the output axis only ``head_dim / 64`` sticks of parallel work, so the
    remaining cores come from splitting the KV contraction.

    Grid is ``[out_split, k_split]``, dimension 0 varying fastest, so the flat core
    id is ``pid_in * out_split + pid_out``: ``out_split`` cores divide the output
    (``head_dim``) and ``k_split`` cores divide the KV contraction.  Because the
    contraction is split, each group of ``k_split`` cores holds a partial sum that
    is folded with ``ktdp.inter_tile_produce`` / ``ktdp.inter_tile_reduce`` over
    *strided* groups -- group ``g`` is cores ``{g, g + out_split, ...}`` -- and the
    ``pid_in == 0`` core of each group is the single writer for its output slice.
    ``k_split = 1`` degenerates to no fold at all and stores the local product.

    Preconditions, raised rather than asserted (``python -O`` strips asserts):

    * ``out_split`` divides ``head_dim`` and ``k_split`` divides ``kv_len``
    * both tile row widths -- the output slice ``head_dim / out_split`` and the
      contraction slice ``kv_len / k_split`` -- must be whole 128-byte sticks,
      i.e. multiples of 64 f16 elements
    * every value is f16, including the fold's combiner region, so the running sum
      rounds at each of the ``k_split`` steps and the *inputs* must leave headroom:
      ``|A@B|`` grows like ``std**2 * sqrt(kv_len)``, which overflows f16 above
      roughly ``std = 27`` at ``kv_len = 8192``.  The caller owns the input scale;
      overflow shows up as ``inf`` in the output rather than silently.

    Arithmetic intensity is close to the harmonic mean of the two tile widths,

        ``AI ~= (q_per_kv * n_local) / (q_per_kv + n_local)``,   ``n_local = head_dim / out_split``

    about 3.8 FLOP/B at the Granite ratio, and independent of ``kv_len``, of
    ``k_split`` and of the core count -- each core reads a disjoint slice of the
    contraction, so splitting harder moves no fewer bytes.  Memory-bound at 4 and
    at 32 cores.
    """
    if head_dim % out_split:
        raise ValueError(
            f"head_dim {head_dim} must be divisible by out_split {out_split}"
        )
    if kv_len % k_split:
        raise ValueError(
            f"kv_len {kv_len} must be divisible by k_split {k_split}"
        )
    n_local = head_dim // out_split
    k_local = kv_len // k_split
    _require_stick_aligned(
        f"output slice head_dim/out_split = {head_dim}/{out_split}", n_local)
    _require_stick_aligned(
        f"contraction slice kv_len/k_split = {kv_len}/{k_split}", k_local)
    hd, kl, nl, m = head_dim, k_local, n_local, q_per_kv

    a_view = _mem_view("a_view", "a_ptr", [m, kv_len], [kv_len, 1])
    b_view = _mem_view("b_view", "b_ptr", [kv_len, hd], [hd, 1])
    c_view = _mem_view("c_view", "c_ptr", [m, hd], [hd, 1])
    a_acc = _access_tile("a_acc", "a_view", ["%c0", "%off_in"],
                         [m, kl], [m, kv_len])
    b_acc = _access_tile("b_acc", "b_view", ["%off_in", "%off_out"],
                         [kl, nl], [kv_len, hd])
    c_acc = _access_tile("c_acc", "c_view", ["%c0", "%off_out"],
                         [m, nl], [m, hd])

    # Strided reduce groups: group g owns cores {g, g + out_split, ...}, which is
    # the set of cores sharing an output slice.  Spelled as a modulo constraint
    # because the grid varies dimension 0 fastest.
    red_tiles = (
        f"#red_tiles = affine_set<(i)[g] : ((i - g) mod {out_split} == 0, "
        f"i - g >= 0, -i + g + {out_split * (k_split - 1)} >= 0)>\n"
    )
    groups = f"affine_set<(g) : (g >= 0, -g + {out_split - 1} >= 0)>"

    if k_split > 1:
        fold = f"""
    %fut = ktdp.inter_tile_produce
        producer_tiles_per_group = #red_tiles
        : tensor<{m}x{nl}xf16> -> !ktdp.tile_future<tensor<{m}x{nl}xf16>, groups = {groups}>
    {{
      ^bb0(%gid: index):
        ktdp.yield_partial %prod : tensor<{m}x{nl}xf16>
    }}
    %zero = arith.constant 0.0 : f16
    %id_empty = tensor.empty() : tensor<{m}x{nl}xf16>
    %id_init = linalg.fill ins(%zero : f16) outs(%id_empty : tensor<{m}x{nl}xf16>) -> tensor<{m}x{nl}xf16>
    %reduced = ktdp.inter_tile_reduce(%fut)
        consumer_tiles_per_group = #red_tiles,
        identity(%id_init : tensor<{m}x{nl}xf16>)
        : !ktdp.tile_future<tensor<{m}x{nl}xf16>, groups = {groups}> -> tensor<{m}x{nl}xf16>
    {{
      ^bb0(%lhs: tensor<{m}x{nl}xf16>, %rhs: tensor<{m}x{nl}xf16>):
        %r_empty = tensor.empty() : tensor<{m}x{nl}xf16>
        %r_sum = linalg.add ins(%lhs, %rhs : tensor<{m}x{nl}xf16>, tensor<{m}x{nl}xf16>) outs(%r_empty : tensor<{m}x{nl}xf16>) -> tensor<{m}x{nl}xf16>
        ktdp.yield_reduced %r_sum : tensor<{m}x{nl}xf16>
    }}
"""
        # One writer per output slice: every core of a group holds the same
        # reduced tile, so guard the store on the group's first core.
        store = f"""    %is_writer = arith.cmpi eq, %pid_in, %c0 : index
    scf.if %is_writer {{
      ktdp.store %reduced, %c_acc : tensor<{m}x{nl}xf16>, !ktdp.access_tile<{m}x{nl}xindex>
    }}"""
    else:
        fold = ""
        store = (f"    ktdp.store %prod, %c_acc : tensor<{m}x{nl}xf16>, "
                 f"!ktdp.access_tile<{m}x{nl}xindex>")

    prologue = red_tiles if k_split > 1 else ""
    return f"""{prologue}module {{
  func.func @sdpa_decode_pv(
      %a_ptr: index, %b_ptr: index, %c_ptr: index
  ) attributes {{grid = [{out_split}, {k_split}]}} {{
    %pid_out, %pid_in = ktdp.get_compute_tile_id : index, index
    %c0 = arith.constant 0 : index
    %blk_in = arith.constant {kl} : index
    %off_in = arith.muli %pid_in, %blk_in : index
    %blk_out = arith.constant {nl} : index
    %off_out = arith.muli %pid_out, %blk_out : index

{a_view}

{b_view}

{c_view}

{a_acc}

{b_acc}

    %a = ktdp.load %a_acc : !ktdp.access_tile<{m}x{kl}xindex> -> tensor<{m}x{kl}xf16>
    %b = ktdp.load %b_acc : !ktdp.access_tile<{kl}x{nl}xindex> -> tensor<{kl}x{nl}xf16>

    %init = arith.constant dense<0.0> : tensor<{m}x{nl}xf16>
    %prod = linalg.matmul ins(%a, %b : tensor<{m}x{kl}xf16>, tensor<{kl}x{nl}xf16>) outs(%init : tensor<{m}x{nl}xf16>) -> tensor<{m}x{nl}xf16>
{fold}
{c_acc}

{store}
    return
  }}
}}"""


def gen_paged_attention_mlir(num_tokens=64, context_len=512,
                             num_query_heads=4, num_kv_heads=1, head_dim=128,
                             block_size=64, block_q=16):
    """Generate a paged attention kernel with online softmax and indirect KV access.

    Grid = [num_tokens // block_q, num_kv_heads].
    Each core handles block_m = (num_query_heads // num_kv_heads) * block_q query rows.
    """
    nqpkv = num_query_heads // num_kv_heads
    block_m = nqpkv * block_q
    grid_0 = num_tokens // block_q
    grid_1 = num_kv_heads
    num_tiles = (context_len + block_size - 1) // block_size
    max_blocks_per_seq = num_tiles
    num_blks = max_blocks_per_seq
    scale = 1.0 / (head_dim ** 0.5)

    q_stride0 = num_query_heads * head_dim
    q_stride1 = head_dim
    kv_stride0 = block_size * num_kv_heads * head_dim
    kv_stride1 = num_kv_heads * head_dim
    kv_stride2 = head_dim

    nt = num_tokens
    hd = head_dim
    bs = block_size
    bm = block_m
    bq = block_q
    nb = num_blks
    mbps = max_blocks_per_seq

    query_view = _mem_view("query_view", "query_ptr",
                           [nt, num_query_heads, hd],
                           [q_stride0, q_stride1, 1])
    key_cache_view = _mem_view("key_cache_view", "key_cache_ptr",
                               [nb, bs, num_kv_heads, hd],
                               [kv_stride0, kv_stride1, kv_stride2, 1])
    value_cache_view = _mem_view("value_cache_view", "value_cache_ptr",
                                 [nb, bs, num_kv_heads, hd],
                                 [kv_stride0, kv_stride1, kv_stride2, 1])
    block_tables_view = _mem_view("block_tables_view", "block_tables_ptr",
                                  [1, mbps], [mbps, 1], dtype="i32")
    output_view = _mem_view("output_view", "output_ptr",
                            [nt, num_query_heads, hd],
                            [q_stride0, q_stride1, 1])

    q_access_tile = _access_tile("q_access_tile", "query_view",
                                 ["%q_row_start", "%pid1_x_nqpkv", "%c0"],
                                 [bq, nqpkv, hd],
                                 [nt, num_query_heads, hd])
    output_access_tile = _access_tile("output_access_tile", "output_view",
                                      ["%q_row_start", "%pid1_x_nqpkv", "%c0"],
                                      [bq, nqpkv, hd],
                                      [nt, num_query_heads, hd])

    k_tile = _indirect_kv_tile("K_tile", "key_cache_view", "block_tables_view",
                               "%bt_idx", "%pid1", nb, bs, num_kv_heads, hd, mbps)
    v_tile = _indirect_kv_tile("V_tile", "value_cache_view", "block_tables_view",
                               "%bt_idx", "%pid1", nb, bs, num_kv_heads, hd, mbps)

    return f"""module {{
  func.func @paged_attention_kernel(
    %output_ptr            : index,
    %query_ptr             : index,
    %key_cache_ptr         : index,
    %value_cache_ptr       : index,
    %block_tables_ptr      : index,
    %cur_batch_start_index : index,
    %block_table_offset    : index,
    %num_tiles             : index,
    %context_len           : index,
    %scale                 : f32
  ) attributes {{grid = [{grid_0}, {grid_1}]}} {{

    %pid0, %pid1 = ktdp.get_compute_tile_id : index, index
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %c_bq = arith.constant {bq} : index
    %c_nqpkv = arith.constant {nqpkv} : index
    %c_bs = arith.constant {bs} : index

    %pid0_x_bq   = arith.muli %pid0, %c_bq : index
    %q_row_start = arith.addi %cur_batch_start_index, %pid0_x_bq : index
    %pid1_x_nqpkv = arith.muli %pid1, %c_nqpkv : index

{query_view}

{key_cache_view}

{value_cache_view}

{block_tables_view}

{output_view}

{q_access_tile}

    %Q_3d = ktdp.load %q_access_tile : !ktdp.access_tile<{bq}x{nqpkv}x{hd}xindex> -> tensor<{bq}x{nqpkv}x{hd}xf16>
    %Q_f16 = tensor.collapse_shape %Q_3d [[0, 1], [2]]
        : tensor<{bq}x{nqpkv}x{hd}xf16> into tensor<{bm}x{hd}xf16>
    %Q = arith.extf %Q_f16 : tensor<{bm}x{hd}xf16> to tensor<{bm}x{hd}xf32>

    %neg_inf_i32 = arith.constant 0xFF800000 : i32
    %neg_inf_f32 = arith.bitcast %neg_inf_i32 : i32 to f32
    %zero_f32    = arith.constant 0.0 : f32
    %one_f32     = arith.constant 1.0 : f32

    %M_init   = tensor.splat %neg_inf_f32 : tensor<{bm}xf32>
    %L_init   = tensor.splat %one_f32     : tensor<{bm}xf32>
    %acc_init = tensor.splat %zero_f32    : tensor<{bm}x{hd}xf32>

    %M_final, %L_final, %acc_final =
      scf.for %j = %c0 to %num_tiles step %c1
        iter_args(%M = %M_init, %L = %L_init, %acc = %acc_init)
        -> (tensor<{bm}xf32>, tensor<{bm}xf32>, tensor<{bm}x{hd}xf32>) {{

      %bt_idx = arith.addi %block_table_offset, %j : index

{k_tile}

{v_tile}

      %K_f16_4d = ktdp.load %K_tile : !ktdp.access_tile<1x{bs}x1x{hd}xindex> -> tensor<1x{bs}x1x{hd}xf16>
      %V_f16_4d = ktdp.load %V_tile : !ktdp.access_tile<1x{bs}x1x{hd}xindex> -> tensor<1x{bs}x1x{hd}xf16>

      %K_f16_2d = tensor.collapse_shape %K_f16_4d [[0, 1], [2, 3]]
          : tensor<1x{bs}x1x{hd}xf16> into tensor<{bs}x{hd}xf16>
      %K_f16_2d_empty = tensor.empty() : tensor<{hd}x{bs}xf16>
      %K_f16 = linalg.transpose
          ins(%K_f16_2d : tensor<{bs}x{hd}xf16>)
          outs(%K_f16_2d_empty : tensor<{hd}x{bs}xf16>)
          permutation = [1, 0]

      %V_f16 = tensor.collapse_shape %V_f16_4d [[0, 1], [2, 3]]
          : tensor<1x{bs}x1x{hd}xf16> into tensor<{bs}x{hd}xf16>

      %K = arith.extf %K_f16 : tensor<{hd}x{bs}xf16> to tensor<{hd}x{bs}xf32>
      %V = arith.extf %V_f16 : tensor<{bs}x{hd}xf16>  to tensor<{bs}x{hd}xf32>

      %S_zeros = tensor.splat %zero_f32 : tensor<{bm}x{bs}xf32>
      %QK = linalg.matmul
            ins(%Q, %K  : tensor<{bm}x{hd}xf32>, tensor<{hd}x{bs}xf32>)
            outs(%S_zeros : tensor<{bm}x{bs}xf32>) -> tensor<{bm}x{bs}xf32>

      %scale_tile = tensor.splat %scale : tensor<{bm}x{bs}xf32>
      %S = arith.mulf %scale_tile, %QK : tensor<{bm}x{bs}xf32>

      %S_masked_empty = tensor.empty() : tensor<{bm}x{bs}xf32>
      %S_masked = linalg.generic {{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%S : tensor<{bm}x{bs}xf32>)
          outs(%S_masked_empty : tensor<{bm}x{bs}xf32>) {{
        ^bb0(%s_val : f32, %out : f32):
          %row       = linalg.index 0 : index
          %col       = linalg.index 1 : index
          %row_div_nqpkv = arith.divui %row, %c_nqpkv : index
          %qpos_row  = arith.addi %pid0_x_bq, %row_div_nqpkv : index
          %qabs_row  = arith.addi %context_len, %qpos_row : index
          %j_x_bs    = arith.muli %j, %c_bs : index
          %seq_off   = arith.addi %j_x_bs, %col : index
          %in_causal = arith.cmpi ule, %seq_off, %qabs_row : index
          %result    = arith.select %in_causal, %s_val, %neg_inf_f32 : f32
          linalg.yield %result : f32
      }} -> tensor<{bm}x{bs}xf32>

      %row_max_init = tensor.splat %neg_inf_f32 : tensor<{bm}xf32>
      %S_row_max = linalg.reduce
          ins(%S_masked : tensor<{bm}x{bs}xf32>)
          outs(%row_max_init : tensor<{bm}xf32>)
          dimensions = [1]
          (%a : f32, %b : f32) {{
        %max_val = arith.maximumf %a, %b : f32
        linalg.yield %max_val : f32
      }}

      %m_j_empty = tensor.empty() : tensor<{bm}xf32>
      %m_j = linalg.generic {{
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}}
          ins(%M, %S_row_max : tensor<{bm}xf32>, tensor<{bm}xf32>)
          outs(%m_j_empty : tensor<{bm}xf32>) {{
        ^bb0(%m_old : f32, %m_new : f32, %out : f32):
          %max_val = arith.maximumf %m_old, %m_new : f32
          linalg.yield %max_val : f32
      }} -> tensor<{bm}xf32>

      %P_empty = tensor.empty() : tensor<{bm}x{bs}xf32>
      %P = linalg.generic {{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%S_masked, %m_j : tensor<{bm}x{bs}xf32>, tensor<{bm}xf32>)
          outs(%P_empty : tensor<{bm}x{bs}xf32>) {{
        ^bb0(%s : f32, %m : f32, %out : f32):
          %sub = arith.subf %s, %m : f32
          %exp = math.exp %sub : f32
          linalg.yield %exp : f32
      }} -> tensor<{bm}x{bs}xf32>

      %zero_vec = tensor.splat %zero_f32 : tensor<{bm}xf32>
      %l_j = linalg.reduce
          ins(%P : tensor<{bm}x{bs}xf32>)
          outs(%zero_vec : tensor<{bm}xf32>)
          dimensions = [1]
          (%a : f32, %b : f32) {{
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
      }}

      %alpha_empty = tensor.empty() : tensor<{bm}xf32>
      %alpha = linalg.generic {{
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}}
          ins(%M, %m_j : tensor<{bm}xf32>, tensor<{bm}xf32>)
          outs(%alpha_empty : tensor<{bm}xf32>) {{
        ^bb0(%m_old : f32, %m_new : f32, %out : f32):
          %diff = arith.subf %m_old, %m_new : f32
          %exp  = math.exp %diff : f32
          linalg.yield %exp : f32
      }} -> tensor<{bm}xf32>

      %acc_scaled_empty = tensor.empty() : tensor<{bm}x{hd}xf32>
      %acc_scaled = linalg.generic {{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%acc, %alpha : tensor<{bm}x{hd}xf32>, tensor<{bm}xf32>)
          outs(%acc_scaled_empty : tensor<{bm}x{hd}xf32>) {{
        ^bb0(%a : f32, %al : f32, %out : f32):
          %scaled = arith.mulf %a, %al : f32
          linalg.yield %scaled : f32
      }} -> tensor<{bm}x{hd}xf32>

      %acc_new = linalg.matmul
          ins(%P, %V   : tensor<{bm}x{bs}xf32>, tensor<{bs}x{hd}xf32>)
          outs(%acc_scaled : tensor<{bm}x{hd}xf32>) -> tensor<{bm}x{hd}xf32>

      %L_new_empty = tensor.empty() : tensor<{bm}xf32>
      %L_new = linalg.generic {{
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}}
          ins(%L, %alpha, %l_j : tensor<{bm}xf32>, tensor<{bm}xf32>, tensor<{bm}xf32>)
          outs(%L_new_empty : tensor<{bm}xf32>) {{
        ^bb0(%l : f32, %al : f32, %lj : f32, %out : f32):
          %la  = arith.mulf %l,  %al : f32
          %new = arith.addf %la, %lj : f32
          linalg.yield %new : f32
      }} -> tensor<{bm}xf32>

      scf.yield %m_j, %L_new, %acc_new
          : tensor<{bm}xf32>, tensor<{bm}xf32>, tensor<{bm}x{hd}xf32>
    }}

    %out_f32_empty = tensor.empty() : tensor<{bm}x{hd}xf32>
    %out_f32 = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}}
        ins(%acc_final, %L_final : tensor<{bm}x{hd}xf32>, tensor<{bm}xf32>)
        outs(%out_f32_empty : tensor<{bm}x{hd}xf32>) {{
      ^bb0(%a : f32, %l : f32, %out : f32):
        %div = arith.divf %a, %l : f32
        linalg.yield %div : f32
    }} -> tensor<{bm}x{hd}xf32>

    %out_f16 = arith.truncf %out_f32 : tensor<{bm}x{hd}xf32> to tensor<{bm}x{hd}xf16>
    %out_3d = tensor.expand_shape %out_f16 [[0, 1], [2]] output_shape [{bq}, {nqpkv}, {hd}]
        : tensor<{bm}x{hd}xf16> into tensor<{bq}x{nqpkv}x{hd}xf16>

{output_access_tile}

    ktdp.store %out_3d, %output_access_tile
        : tensor<{bq}x{nqpkv}x{hd}xf16>, !ktdp.access_tile<{bq}x{nqpkv}x{hd}xindex>

    return
  }}
}}"""
