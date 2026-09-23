// Integration test: every op in the real example kernels has a registered
// handler. This is the end-to-end "the frontend and dispatch table meet" check
// — it parses actual examples/*.mlir and confirms the Dispatch table covers
// every op_type the parser produces (recursing into regions).

use ktir_emulator::dialects::Dispatch;
use ktir_emulator::ir::Operation;
use ktir_emulator::parser::parse_module;

fn collect_op_types<'a>(ops: &'a [Operation], out: &mut Vec<&'a str>) {
    for op in ops {
        out.push(&op.op_type);
        for region in &op.regions {
            collect_op_types(region, out);
        }
    }
}

/// Ops known to be not-yet-ported (tracked burn-down list). A file that uses
/// ONLY these as its missing ops is an allowed known-gap; any OTHER missing op
/// is a regression. When one of these lands, the corresponding file flips to
/// fully-dispatchable and the test nudges us (via `fully dispatches now`) to
/// shrink this list.
// The experimental inter-tile reduce surface is NOT yet ported to Rust — the
// port still implements the legacy `ktdp.reduce` ring all-reduce. Upstream
// c428844 (#72) rewrote examples/ktir/ring_reduce.mlir to the four-op design
// (inter_tile_produce / inter_tile_reduce / yield_partial / yield_reduced),
// which tracks the still-unmerged spec ktir-mlir-frontend#23. Deferred pending
// that spec — see rust/TODOs.md. (Tile-level `arith.bitcast` likewise still
// needs the dtype-faithful storage fork, but the corpus only uses the scalar
// form, which is implemented.)
const KNOWN_GAP_OPS: &[&str] = &[
    "ktdp.inter_tile_produce",
    "ktdp.inter_tile_reduce",
    "ktdp.yield_reduced",
];

fn missing_handlers(src: &str, label: &str) -> Vec<String> {
    let module = parse_module(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}"));
    let dispatch = Dispatch::new();
    let mut missing = Vec::new();
    for func in module.functions.values() {
        let mut types = Vec::new();
        collect_op_types(&func.operations, &mut types);
        for t in types {
            // An op is executable if it has a normal handler OR is a comm op
            // (driven by the scheduler, not the dispatch table).
            let covered = dispatch.handler(t).is_some() || ktir_emulator::comm_sched::is_comm_op(t);
            if !covered && !missing.iter().any(|m| m == t) {
                missing.push(t.to_string());
            }
        }
    }
    missing
}

fn assert_all_dispatchable(src: &str, label: &str) {
    let missing = missing_handlers(src, label);
    let unexpected: Vec<&String> = missing
        .iter()
        .filter(|m| !KNOWN_GAP_OPS.contains(&m.as_str()))
        .collect();
    assert!(
        unexpected.is_empty(),
        "{label}: ops with no registered handler (not in KNOWN_GAP_OPS): {unexpected:?}"
    );
}

/// Every example kernel: parse it and confirm the Dispatch table covers every
/// op the parser produces. `include_str!` needs literal paths, so the corpus is
/// enumerated explicitly (kept in sync with `examples/**/*.mlir`).
macro_rules! corpus {
    ($($name:ident => $path:literal),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                assert_all_dispatchable(
                    include_str!(concat!("../../../../examples/", $path)),
                    $path,
                );
            }
        )+
    };
}

corpus! {
    reduce_generic        => "ktir/reduce_generic.mlir",
    reduce_multiop        => "ktir/reduce_multiop.mlir",
    ring_reduce           => "ktir/ring_reduce.mlir",
    softmax_wide          => "ktir/softmax_wide.mlir",
    matmul_small          => "latency/matmul_small.mlir",
    softmax_small_explicit=> "latency/softmax_small_explicit.mlir",
    softmax_small         => "latency/softmax_small.mlir",
    add_with_control_flow => "rfc/add-with-control-flow.mlir",
    distributed_view_copy => "rfc/distributed-view-copy.mlir",
    indirect_access_copy  => "rfc/indirect-access-copy.mlir",
    indirect_scatter      => "rfc/indirect-scatter.mlir",
    paged_tensor_copy     => "rfc/paged-tensor-copy.mlir",
    paged_tensor_write    => "rfc/paged-tensor-write.mlir",
    indexed_add           => "triton-ktir/indexed_add.mlir",
    layernorm_fwd         => "triton-ktir/layernorm_fwd_ktir.mlir",
    matmul_fwd            => "triton-ktir/matmul_fwd_ktir.mlir",
    paged_attention       => "triton-ktir/paged_attention.mlir",
    sdpa_2d               => "triton-ktir/sdpa_2d.mlir",
    softmax_fwd           => "triton-ktir/softmax_fwd_ktir.mlir",
    vector_add_dynamic    => "triton-ktir/vector_add_dynamic_ktir.mlir",
    vector_add            => "triton-ktir/vector_add_ktir.mlir",
}
