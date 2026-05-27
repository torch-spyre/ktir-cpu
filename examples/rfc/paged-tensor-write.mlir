// Scatter dual of paged-tensor-copy.mlir.
// Same semantics, X and Y roles swapped:
//   paged-tensor-copy:  Y[b][tkv][h][dkv] = X[Idx[b][tkv/Ptkv]][h][tkv%Ptkv][dkv]
//   paged-tensor-write: Y[Idx[b][tkv/Ptkv]][h][tkv%Ptkv][dkv] = X[b][tkv][h][dkv]
//
// X input  = contiguous 4D tensor, shape {Nb, Ntkv, Nh, Ndkv} = {4, 2048, 8, 128} f16
// Idx input = 2D tensor, shape {Nb, Ntkv/Ptkv} = {4, 32} i32
// Y output = paged 4D tensor, shape {Npages, Nh, Ptkv, Ndkv} = {10000, 8, 64, 128} f16
//
// The `var_space` declarations are kept verbatim from paged-tensor-copy.mlir;
// the `var_space` qualifier refers to the iteration variable layout, not to
// which tensor the IAT acts on. So `#X_var_space_*` continues to describe the
// indirect-access (b, h, tkv, dkv) iteration even though X is now the
// (contiguous) source.

#paged_coord_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 9999 >= 0,
                                              d1 >= 0, -d1 + 7 >=0,
                                              d2 >= 0, -d2 + 63 >= 0,
                                              d3 >= 0, -d3 + 127>= 0)>
#contiguous_coord_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0,
                                              d1 >= 0, -d1 + 2047 >=0,
                                              d2 >= 0, -d2 + 7 >= 0,
                                              d3 >= 0, -d3 + 127>= 0)>
#X_var_space_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0,
                                                  d1 >= 0, -d1 + 7 >= 0,
                                                  d2 >= 0, -d2 + 2047>= 0,
                                                  d3 >= 0, -d3 + 127 >= 0)>
#Y_var_space_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0,
                                                  d1 >= 0, -d1 + 2047>= 0,
                                                  d2 >= 0, -d2 + 7 >= 0,
                                                  d3 >= 0, -d3 + 127 >= 0)>
#X_var_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#Y_var_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

module {
  func.func @paged_tensor_write_1core() {
        %Nb = arith.constant 4 : index
        %Ntkv = arith.constant 2048 : index
        %Ptkv = arith.constant 64 : index
        %Ndkv = arith.constant 128 : index
        %Nhkv = arith.constant 8 : index
        %Npages = arith.constant 10000 : index
        %Ntkv_Ptkv = arith.divui %Ntkv, %Ptkv : index

        // Memory layout: addresses are spaced so all three tensors are byte-disjoint.
        //   X  spans [10M, 26.78M)  — 4*2048*8*128 f16 = 16.78 MB (contiguous source)
        //   Idx spans [30M, 30M+512] — 4*32 i32 = 512 B (page table)
        //   Y  starts at 40M         — 10000*8*64*128 f16 = 1.31 GB (paged dest)
        // (paged-tensor-copy.mlir uses overlapping addresses; we don't replicate
        // that here — see test_spec_gaps.py for the matching seed values.)
        %X_start_address = arith.constant 10000000 : index   // X (contiguous source)
        %Idx_start_address = arith.constant 30000000 : index // Idx (page table)
        %Y_start_address = arith.constant 40000000 : index   // Y (paged destination)

        // (1) Construct memory view for Index tensor
        %Idx_mem_view = ktdp.construct_memory_view %Idx_start_address,
                        sizes: [%Nb, %Ntkv_Ptkv], strides: [%Ntkv_Ptkv, 1] {
                        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                        memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<4x32xi32>

        // (1) Construct memory view for input X (contiguous)
        // dim order (outermost to innermost) {Nb, Ntkv, Nh, Ndkv}
        %X_str_Ntkv = arith.muli %Ndkv, %Nhkv : index
        %X_str_Nb = arith.muli %X_str_Ntkv, %Ntkv : index
        %X_mem_view = ktdp.construct_memory_view %X_start_address,
                        sizes: [%Nb, %Ntkv, %Nhkv, %Ndkv], strides: [%X_str_Nb, %X_str_Ntkv, %Ndkv, 1] {
                        coordinate_set = #contiguous_coord_set,
                        memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<4x2048x8x128xf16>

        // (1) Construct memory view for output Y (paged)
        // dim order (outermost to innermost) {Npages, Nh, Ptkv, Ndkv}
        %Y_mem_view = ktdp.construct_memory_view %Y_start_address,
                        sizes: [%Npages, %Nhkv, %Ptkv, %Ndkv], strides: [65536, 8192, %Ndkv, 1] {
                        coordinate_set = #paged_coord_set,
                        memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<10000x8x64x128xf16>

        // (2) Direct access tile for X[b, tkv, h, dkv] — load source contiguously
        // access_tile_set/order use the (b, tkv, h, dkv) iteration so flatten()
        // of the loaded tile aligns with the (b, h, tkv, dkv) IAT enumeration.
        %c0 = arith.constant 0 : index
        %X_access_tile = ktdp.construct_access_tile %X_mem_view[%c0, %c0, %c0, %c0] {
            access_tile_set   = #Y_var_space_set,
            access_tile_order = #Y_var_space_order
        } : memref<4x2048x8x128xf16> -> !ktdp.access_tile<4x8x2048x128xindex>

        // (3) Indirect access tile for Y[Idx[b, tkv/Ptkv], h, tkv%Ptkv, dkv]
        %Y_access_tile = ktdp.construct_indirect_access_tile
            intermediate_variables(%b, %h, %tkv, %dkv)
            %Y_mem_view[ind(%Idx_mem_view[%b, %tkv floordiv 64]), (%h), (%tkv mod 64), (%dkv)] {
                variables_space_set   = #X_var_space_set,
                variables_space_order = #X_var_space_order
            } : memref<10000x8x64x128xf16>, memref<4x32xi32> -> !ktdp.access_tile<4x8x2048x128xindex>

        // (4) Load X[b, tkv, h, dkv]
        %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<4x8x2048x128xindex> -> tensor<4x8x2048x128xf16>

        // (5) Scatter: Y[Idx[b, tkv/Ptkv], h, tkv%Ptkv, dkv] = X[b, tkv, h, dkv]
        ktdp.store %X_data_tile, %Y_access_tile : tensor<4x8x2048x128xf16>, !ktdp.access_tile<4x8x2048x128xindex>

        return
  }
}
