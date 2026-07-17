// FFN-SwiGLU: Feedforward network with SwiGLU activation
// 
// Algorithm:
//   gate = x @ W_gate
//   up = x @ W_up
//   silu = gate * sigmoid(gate)  where sigmoid(x) = 1 / (1 + exp(-x))
//   fused = silu * up
//   out = fused @ W_down
//   result = x + out  (residual connection)
//
// Dimensions (Phase 1 - minimal):
//   seq = 1, d_model = 64, d_ffn = 128
//   All operations single-core, no tiling

module {
  func.func @ffn_swiglu(
      %x_ptr: index,      // input [1, 64]
      %w_gate_ptr: index, // gate weights [64, 128]
      %w_up_ptr: index,   // up weights [64, 128]
      %w_down_ptr: index, // down weights [128, 64]
      %out_ptr: index     // output [1, 64]
  ) attributes {grid = [1, 1]} {
    
    %c0 = arith.constant 0 : index
    
    // Create memory views
    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [1, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x64xf16>
    
    %w_gate_view = ktdp.construct_memory_view %w_gate_ptr, sizes: [64, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x128xf16>
    
    %w_up_view = ktdp.construct_memory_view %w_up_ptr, sizes: [64, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x128xf16>
    
    %w_down_view = ktdp.construct_memory_view %w_down_ptr, sizes: [128, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x64xf16>
    
    %out_view = ktdp.construct_memory_view %out_ptr, sizes: [1, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x64xf16>
    
    // Load input x
    %x_acc = ktdp.construct_access_tile %x_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    %x = ktdp.load %x_acc : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    
    // Gate projection: gate = x @ W_gate  [1,64] @ [64,128] -> [1,128]
    %w_gate_acc = ktdp.construct_access_tile %w_gate_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
    %w_gate = ktdp.load %w_gate_acc : !ktdp.access_tile<64x128xindex> -> tensor<64x128xf16>
    
    %gate_init = tensor.empty() : tensor<1x128xf16>
    %gate = linalg.matmul ins(%x, %w_gate : tensor<1x64xf16>, tensor<64x128xf16>)
                          outs(%gate_init : tensor<1x128xf16>) -> tensor<1x128xf16>
    
    // Up projection: up = x @ W_up  [1,64] @ [64,128] -> [1,128]
    %w_up_acc = ktdp.construct_access_tile %w_up_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
    %w_up = ktdp.load %w_up_acc : !ktdp.access_tile<64x128xindex> -> tensor<64x128xf16>
    
    %up_init = tensor.empty() : tensor<1x128xf16>
    %up = linalg.matmul ins(%x, %w_up : tensor<1x64xf16>, tensor<64x128xf16>)
                        outs(%up_init : tensor<1x128xf16>) -> tensor<1x128xf16>
    
    // SiLU activation: silu(gate) = gate * sigmoid(gate)
    // sigmoid(x) = 1 / (1 + exp(-x))
    %neg_gate = arith.negf %gate : tensor<1x128xf16>
    %exp_neg_gate = math.exp %neg_gate : tensor<1x128xf16>
    %one = arith.constant dense<1.0> : tensor<1x128xf16>
    %one_plus_exp = arith.addf %one, %exp_neg_gate : tensor<1x128xf16>
    %sigmoid = arith.divf %one, %one_plus_exp : tensor<1x128xf16>
    %silu = arith.mulf %gate, %sigmoid : tensor<1x128xf16>
    
    // Gated fusion: fused = silu * up
    %fused = arith.mulf %silu, %up : tensor<1x128xf16>
    
    // Down projection: out_partial = fused @ W_down  [1,128] @ [128,64] -> [1,64]
    %w_down_acc = ktdp.construct_access_tile %w_down_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
    %w_down = ktdp.load %w_down_acc : !ktdp.access_tile<128x64xindex> -> tensor<128x64xf16>
    
    %out_partial_init = tensor.empty() : tensor<1x64xf16>
    %out_partial = linalg.matmul ins(%fused, %w_down : tensor<1x128xf16>, tensor<128x64xf16>)
                                 outs(%out_partial_init : tensor<1x64xf16>) -> tensor<1x64xf16>
    
    // Residual connection: result = x + out_partial
    %result = arith.addf %x, %out_partial : tensor<1x64xf16>
    
    // Store result
    %out_acc = ktdp.construct_access_tile %out_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    ktdp.store %result, %out_acc : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    
    return
  }
}