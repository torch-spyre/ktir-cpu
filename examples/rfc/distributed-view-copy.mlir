// RFC 0682 Section C.3 — construct_distributed_memory_view
// This is an example program to illustrate the use of ktdp.construct_distributed_memory_view.
// The program copies a distributed tensor A (partitioned across HBM and two LX scratchpads)
// into a contiguous output tensor B on HBM.
// NOTE: Copied from KTIR-V1.ppt slide 9 - construct distributed memory view
// NOTE: Minor correction #ktpd.spyre_memory_space<LX0> -> LX

// A is a 192x64 logical tensor partitioned across three memory spaces:
//   A_HBM  = rows   0..95,  stored on HBM,  row-major    (96x64)
//   A_LX0  = rows  96..127, stored on LX,   column-major (32x64)
//   A_LX1  = rows 128..191, stored on LX,   row-major    (64x64)
// B output = 2D tensor, shape {192, 64}, stored on HBM

#A_HBM_coord_set = affine_set<(d0, d1) : (d0 >= 0,   -d0 + 95  >= 0, d1 >= 0, -d1 + 63 >= 0)>
#A_LX0_coord_set = affine_set<(d0, d1) : (d0 >= 96,  -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#A_LX1_coord_set = affine_set<(d0, d1) : (d0 >= 128, -d0 + 191 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#B_coord_set     = affine_set<(d0, d1) : (d0 >= 0,   -d0 + 191 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_space_set   = affine_set<(d0, d1) : (d0 >= 0,   -d0 + 191 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_space_order = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @distributed_view_copy() {

        %c0 = arith.constant 0 : index

        // In this example, A is distributed across HBM and two LX scratchpads; B is on HBM
        %A_HBM_addr = arith.constant 0       : index
        %A_LX0_addr = arith.constant 12288   : index
        %A_LX1_addr = arith.constant 16384   : index
        %B_addr     = arith.constant 24576   : index

        // Accessing a tensor in KTIR follows a 3 step process:
        // Note1: Accesses are single-ended i.e.,
        //        load = read from memory to produce a data-tile
        //        store = write a data-tile to memory
        // Note2: To accomplish a data-transfer (or copy) from source to destination,
        //        we need a separate load from source and a store to destination
        // (1) Create memory view: Informs how the tensor is present in memory
        //    Note: construct_distributed_memory_view composes per-partition views into a single logical view
        //    Note: the coordinate_set on each partition view defines which global coordinates it covers
        // (2) Create access tile: Informs the logical coordinates of the tensor that must be accessed
        // (3) Create data tile: Extract a sub-portion of the tensor corresponding to the coordinates present in the access tile

        // (1) Construct per-partition memory views for A
        // Note: number of entries in sizes, strides, dims in coordinate_set, shape of memref must be identical
        %A_HBM_view = ktdp.construct_memory_view %A_HBM_addr, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = #A_HBM_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>

        // Note: column-major layout expressed via strides [1, 64]
        %A_LX0_view = ktdp.construct_memory_view %A_LX0_addr, sizes: [32, 64], strides: [1, 64] {
            coordinate_set = #A_LX0_coord_set,
            memory_space = #ktdp.spyre_memory_space<LX>
        } : memref<32x64xf16>

        %A_LX1_view = ktdp.construct_memory_view %A_LX1_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #A_LX1_coord_set,
            memory_space = #ktdp.spyre_memory_space<LX>
        } : memref<64x64xf16>

        // (1) Compose the three partition views into a single logical distributed view of shape 192x64
        // Note: the global coordinate domain is the union of the coordinate_sets of the input views
        // Note: this operation does not allocate or move data
        %A_global_view = ktdp.construct_distributed_memory_view
            (%A_HBM_view, %A_LX0_view, %A_LX1_view : memref<96x64xf16>, memref<32x64xf16>, memref<64x64xf16>)
            : memref<192x64xf16>

        // (1) Construct memory view for output B
        %B_view = ktdp.construct_memory_view %B_addr, sizes: [192, 64], strides: [64, 1] {
            coordinate_set = #B_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<192x64xf16>

        // (2) Construct direct access tile for A over the full 192x64 global coordinate space
        %A_access_tile = ktdp.construct_access_tile %A_global_view[%c0, %c0] {
            access_tile_set = #var_space_set,
            access_tile_order = #var_space_order
        } : memref<192x64xf16> -> !ktdp.access_tile<192x64xindex>

        // (2) Construct direct access tile for B
        %B_access_tile = ktdp.construct_access_tile %B_view[%c0, %c0] {
            access_tile_set = #var_space_set,
            access_tile_order = #var_space_order
        } : memref<192x64xf16> -> !ktdp.access_tile<192x64xindex>

        // (3) Load data tile from A using the distributed access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<192x64xindex> -> tensor<192x64xf16>

        // (4) Store B = A_data_tile
        ktdp.store %A_data_tile, %B_access_tile : tensor<192x64xf16>, !ktdp.access_tile<192x64xindex>

        return
  }
}
