// RFC 0682 Section C.5 — construct_indirect_access_tile (scatter)
// Symmetric counterpart to indirect-access-copy.mlir.
// indirect-access-copy:  Y[m, k] = X[ IDX1[m, k], IDX2[m, k] ]   (gather via load)
// indirect-scatter:      Y[ IDX1[m, k], IDX2[m, k] ] = X[m, k]   (scatter via store)

// X input  = 2D tensor, shape {64, 64}                          (source, direct access)
// IDX1     = 2D tensor, shape {64, 64}, provides row indices into Y
// IDX2     = 2D tensor, shape {64, 64}, provides column indices into Y
// Y output = 2D tensor, shape {64, 64}                          (destination, indirect access)

#X_coord_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#IDX_coord_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#Y_coord_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_space_order = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @indirect_scatter() {
        // Stick-indexed addresses (HBM); same layout convention as indirect-access-copy.mlir.
        %X_addr    = arith.constant 0   : index   // stick 0   = byte 0
        %IDX1_addr = arith.constant 64  : index   // stick 64  = byte 8192
        %IDX2_addr = arith.constant 128 : index   // stick 128 = byte 16384
        %Y_addr    = arith.constant 192 : index   // stick 192 = byte 24576

        // (1) Construct memory view for X (source)
        %X_view = ktdp.construct_memory_view %X_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #X_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xf16>

        // (1) Construct memory view for IDX1 (row indices into Y)
        %IDX1 = ktdp.construct_memory_view %IDX1_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #IDX_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xi32>

        // (1) Construct memory view for IDX2 (column indices into Y)
        %IDX2 = ktdp.construct_memory_view %IDX2_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #IDX_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xi32>

        // (1) Construct memory view for Y (destination)
        %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #Y_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xf16>

        // (2) Direct access tile for X[m, k] — load source contiguously
        %c0 = arith.constant 0 : index
        %X_access_tile = ktdp.construct_access_tile %X_view[%c0, %c0] {
            access_tile_set   = #X_coord_set,
            access_tile_order = #var_space_order
        } : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>

        // (2) Indirect access tile for Y[IDX1[m,k], IDX2[m,k]] — scatter destination
        %Y_access_tile = ktdp.construct_indirect_access_tile
            intermediate_variables(%m, %k)
            %Y_view[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
                variables_space_set = #var_space_set,
                variables_space_order = #var_space_order
            } : memref<64x64xf16>, memref<64x64xi32>, memref<64x64xi32> -> !ktdp.access_tile<64x64xindex>

        // (3) Load source tile X[m, k]
        %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<64x64xindex> -> tensor<64x64xf16>

        // (4) Scatter: Y[IDX1[m, k], IDX2[m, k]] = X[m, k]
        ktdp.store %X_data_tile, %Y_access_tile : tensor<64x64xf16>, !ktdp.access_tile<64x64xindex>

        return
  }
}
