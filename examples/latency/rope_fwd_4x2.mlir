// RoPE Forward — Standalone Rotary Position Embedding
//
// Formulation (half-layout, LLaMA convention):
//   y[:, 0:D/2]  = x[:, 0:D/2] * cos - x[:, D/2:D] * sin
//   y[:, D/2:D]  = x[:, 0:D/2] * sin + x[:, D/2:D] * cos
//
// Reference model: LLaMA-3-8B / Granite-8B (H_q=32, H_kv=8, D=128, S=4096)
// Total head-lanes: 40 (32 Q + 8 K, both rotated)
//
// Grid (4, 2): dim0=seq partitions (4), dim1=head groups (2)
//   - x/out: PARTITIONED along both dims (block-contiguous)
//   - cos/sin: PARTITIONED along dim0 (seq), REPLICATED along dim1 (heads)
//   - cos/sin depend only on seq position → shared across all heads
//   - No inter-core communication (embarrassingly parallel)
//
// Per-core: 20 heads (= 40 total / 2 grid_dim1) × 1024 positions × 128 dim
// Loop structure: head loop (20 iter) → inner seq-tile loop (4 iter, TILE_SEQ=256)
// cos/sin [256, 64] loaded per tile (no tensor-slice op → can't hoist [1024,64] across tiles)
//
// Arithmetic intensity: ~0.63 FLOPs/byte (memory-bound, vector-unit workload)
// Peak working set per inner iteration: 192 KB (6 × [256,64] × 2B)

module {
  func.func @rope_fwd_kernel(
      %x_ptr: index,     // input  [H=40, S=4096, D=128] flattened to [163840, 128] f16
      %cos_ptr: index,   // precomputed cos [S=4096, D/2=64] f16
      %sin_ptr: index,   // precomputed sin [S=4096, D/2=64] f16
      %out_ptr: index    // output [H=40, S=4096, D=128] flattened to [163840, 128] f16
  ) attributes {grid = [4, 2]} {

    // --- Tile IDs and offsets ---
    %pid_s, %pid_h = ktdp.get_compute_tile_id : index, index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c20 = arith.constant 20 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index

    %seq_offset = arith.muli %pid_s, %c1024 : index
    %head_offset = arith.muli %pid_h, %c20 : index

    // --- Construct memory views ---
    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [163840, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 163839 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<163840x128xf16>

    %cos_view = ktdp.construct_memory_view %cos_ptr, sizes: [4096, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096x64xf16>

    %sin_view = ktdp.construct_memory_view %sin_ptr, sizes: [4096, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096x64xf16>

    %out_view = ktdp.construct_memory_view %out_ptr, sizes: [163840, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 163839 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<163840x128xf16>

    // --- OUTER LOOP: head loop (20 iterations) ---
    scf.for %h = %c0 to %c20 step %c1 {

      // Row base in flattened [H*S, D]: (head_offset + h) * S
      %h_abs = arith.addi %head_offset, %h : index
      %row_base = arith.muli %h_abs, %c4096 : index

      // --- INNER LOOP: seq-tile loop (4 iterations, TILE_SEQ=256) ---
      scf.for %t = %c0 to %c4 step %c1 {

        %tile_offset = arith.muli %t, %c256 : index
        %row = arith.addi %row_base, %seq_offset : index
        %row_t = arith.addi %row, %tile_offset : index
        // cos/sin index: position within the [4096, 64] table
        %cos_row = arith.addi %seq_offset, %tile_offset : index

        // Load cos [256, 64] for this seq tile
        %cos_acc = ktdp.construct_access_tile %cos_view[%cos_row, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<4096x64xf16> -> !ktdp.access_tile<256x64xindex>

        %cos_tile = ktdp.load %cos_acc : !ktdp.access_tile<256x64xindex> -> tensor<256x64xf16>

        // Load sin [256, 64] for this seq tile
        %sin_acc = ktdp.construct_access_tile %sin_view[%cos_row, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<4096x64xf16> -> !ktdp.access_tile<256x64xindex>

        %sin_tile = ktdp.load %sin_acc : !ktdp.access_tile<256x64xindex> -> tensor<256x64xf16>

        // Load x_first [256, 64] — first half of head_dim
        %x_first_acc = ktdp.construct_access_tile %x_view[%row_t, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<163840x128xf16> -> !ktdp.access_tile<256x64xindex>

        %x_first = ktdp.load %x_first_acc : !ktdp.access_tile<256x64xindex> -> tensor<256x64xf16>

        // Load x_second [256, 64] — second half of head_dim
        %x_second_acc = ktdp.construct_access_tile %x_view[%row_t, %c64] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<163840x128xf16> -> !ktdp.access_tile<256x64xindex>

        %x_second = ktdp.load %x_second_acc : !ktdp.access_tile<256x64xindex> -> tensor<256x64xf16>

        // Compute y_first = x_first * cos - x_second * sin
        %tmp1 = arith.mulf %x_first, %cos_tile : tensor<256x64xf16>
        %tmp2 = arith.mulf %x_second, %sin_tile : tensor<256x64xf16>
        %y_first = arith.subf %tmp1, %tmp2 : tensor<256x64xf16>

        // Store y_first [256, 64]
        %y_first_acc = ktdp.construct_access_tile %out_view[%row_t, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<163840x128xf16> -> !ktdp.access_tile<256x64xindex>

        ktdp.store %y_first, %y_first_acc : tensor<256x64xf16>, !ktdp.access_tile<256x64xindex>

        // Compute y_second = x_first * sin + x_second * cos
        %tmp3 = arith.mulf %x_first, %sin_tile : tensor<256x64xf16>
        %tmp4 = arith.mulf %x_second, %cos_tile : tensor<256x64xf16>
        %y_second = arith.addf %tmp3, %tmp4 : tensor<256x64xf16>

        // Store y_second [256, 64]
        %y_second_acc = ktdp.construct_access_tile %out_view[%row_t, %c64] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<163840x128xf16> -> !ktdp.access_tile<256x64xindex>

        ktdp.store %y_second, %y_second_acc : tensor<256x64xf16>, !ktdp.access_tile<256x64xindex>

        scf.yield
      }
      scf.yield
    }
    return
  }
}
