// Enables the Metal/NAX backend automatically on macOS — no feature flag — so
// the emulator dispatches large matmuls to the M5 tensor engine by default,
// exactly as Apple Accelerate is linked by default on macOS. The `objc2-*`
// crates are macOS-only, so on macOS they are always present (target deps);
// elsewhere the optional `metal` feature can still force it on for cross builds.
//
// Emits `cfg(metal)`; all backend code gates on `cfg(metal)`.
//
// AOT: on macOS (with a working offline `xcrun metal` toolchain that honors
// `-mmacosx-version-min=26.2`) we ALSO precompile every GEMM kernel variant to a
// `.metallib` at build time and embed them, so the runtime loads compiled
// libraries instead of JIT-compiling MSL on first use. This emits `cfg(metal_aot)`.
// The `-mmacosx-version-min=26.2` flag is MANDATORY for the NAX (`matmul2d`)
// kernels: on SDK 26.5 the offline toolchain otherwise miscompiles MPP `matmul2d`
// to reduce only HALF its K (see scratchy AI-native PR #56 / MLX #3622). If the
// toolchain or that flag is unavailable we DO NOT emit `cfg(metal_aot)` and the
// runtime falls back to the (always-correct) JIT `newLibraryWithSource` path.
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(metal)");
    println!("cargo::rustc-check-cfg=cfg(metal_aot)");
    let is_macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let feature_on = std::env::var("CARGO_FEATURE_METAL").is_ok();
    if is_macos || feature_on {
        println!("cargo::rustc-cfg=metal");
    }

    // AOT precompilation only makes sense where the Metal runtime is present
    // (the metallibs are loaded by the `cfg(metal)` backend). Mirror the cfg(metal)
    // gate, but additionally require a functioning offline toolchain.
    if is_macos || feature_on {
        try_build_aot();
    }
}

/// The shader sources (relative to the crate root). `include_str!` in metal.rs
/// ships these for the JIT fallback; here we feed the SAME files to the offline
/// compiler so the AOT metallibs are byte-equivalent to the JIT build.
const SHADER_NAX: &str = "shaders/nax_matmul.metal";
const SHADER_SIMD: &str = "shaders/simd_matmul.metal";
const SHADER_GEMV: &str = "shaders/nax_gemv.metal";

/// One AOT variant: the output metallib stem, the shader file, and the `#define`s
/// prepended to the source before compiling. The stem MUST match the name metal.rs
/// `include_bytes!`s. (kernel name is implied by the shader and checked at load.)
struct Variant {
    stem: &'static str,
    shader: &'static str,
    defines: &'static [(&'static str, u32)],
}

/// The FULL variant matrix the runtime `NaxGemm::compile` builds — kept in lockstep
/// with metal.rs. NAX matmul: 8 variants {transpose_b 0|1} x {full | small-M} x
/// {f32 | f16-B}. nax_gemv: 2 {transpose_b 0|1}. simdgroup matmul: 4 {transpose_b
/// 0|1} x {f32 | f16-B}.
fn variants() -> Vec<Variant> {
    use Variant as V;
    let mut v = Vec::new();
    // NAX matmul — 8 variants.
    for &tb in &[0u32, 1] {
        for &sm in &[0u32, 1] {
            for &f16b in &[0u32, 1] {
                // Build a stable leaked &'static [(&str,u32)] for the defines.
                let mut defs: Vec<(&'static str, u32)> = vec![("KTIR_TRANSPOSE_B", tb)];
                if sm == 1 {
                    defs.push(("KTIR_SGS_M", 1));
                }
                if f16b == 1 {
                    defs.push(("KTIR_B_F16", 1));
                }
                let stem: &'static str =
                    Box::leak(format!("nax_matmul__tb{tb}_sm{sm}_f16b{f16b}").into_boxed_str());
                v.push(V {
                    stem,
                    shader: SHADER_NAX,
                    defines: Box::leak(defs.into_boxed_slice()),
                });
            }
        }
    }
    // nax_gemv — 2 variants.
    for &tb in &[0u32, 1] {
        let stem: &'static str = Box::leak(format!("nax_gemv__tb{tb}").into_boxed_str());
        v.push(V {
            stem,
            shader: SHADER_GEMV,
            defines: Box::leak(vec![("KTIR_TRANSPOSE_B", tb)].into_boxed_slice()),
        });
    }
    // simdgroup matmul — 4 variants {transpose_b 0|1} x {f32 | f16-B} (no small-M).
    for &tb in &[0u32, 1] {
        for &f16b in &[0u32, 1] {
            let mut defs: Vec<(&'static str, u32)> = vec![("KTIR_TRANSPOSE_B", tb)];
            if f16b == 1 {
                defs.push(("KTIR_B_F16", 1));
            }
            let stem: &'static str =
                Box::leak(format!("simd_matmul__tb{tb}_f16b{f16b}").into_boxed_str());
            v.push(V {
                stem,
                shader: SHADER_SIMD,
                defines: Box::leak(defs.into_boxed_slice()),
            });
        }
    }
    v
}

fn try_build_aot() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Rerun whenever any shader (or this build script) changes.
    println!("cargo::rerun-if-changed=build.rs");
    for s in [SHADER_NAX, SHADER_SIMD, SHADER_GEMV] {
        println!("cargo::rerun-if-changed={s}");
    }

    // 1) Locate the offline metal compiler.
    let metal_bin = match find_metal() {
        Some(p) => p,
        None => {
            println!(
                "cargo::warning=ktir-emulator: `xcrun --find metal` failed — \
                 AOT GEMM metallibs disabled, falling back to runtime JIT."
            );
            return;
        }
    };

    // 2) Probe that the toolchain honors -mmacosx-version-min=26.2 (the MANDATORY
    //    flag that selects the MLX #3622-fixed MPP `matmul2d` headers; without it
    //    the offline toolchain miscompiles to a half-K reduction). We compile a
    //    tiny MPP probe; if it fails, do NOT emit metal_aot.
    if !probe_toolchain(&metal_bin, &out_dir) {
        println!(
            "cargo::warning=ktir-emulator: offline metal toolchain does not support \
             the MPP matmul2d AOT path (-mmacosx-version-min=26.2 probe failed) — \
             AOT disabled, falling back to runtime JIT."
        );
        return;
    }

    // 3) Compile every variant.
    for var in variants() {
        let shader_path = Path::new(&manifest_dir).join(var.shader);
        let src = match std::fs::read_to_string(&shader_path) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "cargo::warning=ktir-emulator: cannot read {} ({e}) — AOT disabled.",
                    shader_path.display()
                );
                return;
            }
        };
        let mut full = String::new();
        for (k, v) in var.defines {
            full.push_str(&format!("#define {k} {v}\n"));
        }
        full.push_str(&src);

        let tmp_metal = out_dir.join(format!("{}.gen.metal", var.stem));
        let metallib = out_dir.join(format!("{}.metallib", var.stem));
        if let Err(e) = std::fs::write(&tmp_metal, full.as_bytes()) {
            println!(
                "cargo::warning=ktir-emulator: write {tmp_metal:?} failed ({e}) — AOT disabled."
            );
            return;
        }
        if !compile_metallib(&metal_bin, &tmp_metal, &metallib) {
            println!(
                "cargo::warning=ktir-emulator: offline compile of variant {} failed — \
                 AOT disabled, falling back to runtime JIT.",
                var.stem
            );
            return;
        }
    }

    // All variants compiled — turn on the AOT load path.
    println!("cargo::rustc-cfg=metal_aot");
}

fn find_metal() -> Option<PathBuf> {
    let out = Command::new("xcrun")
        .args(["--find", "metal"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if p.is_empty() {
        return None;
    }
    Some(PathBuf::from(p))
}

/// Compile a `.metal` to a `.metallib` with the MANDATORY 26.2 flag set.
/// `xcrun metal ... in.metal -o out.metallib` emits a metallib directly.
fn compile_metallib(metal_bin: &Path, src: &Path, out: &Path) -> bool {
    let status = Command::new(metal_bin)
        .args([
            "-std=metal4.0",
            "-fno-fast-math",
            "-mmacosx-version-min=26.2",
            "-c",
        ])
        .arg(src)
        .arg("-o")
        .arg(out.with_extension("air"))
        .status();
    let air_ok = matches!(status, Ok(s) if s.success());
    if !air_ok {
        return false;
    }
    // air -> metallib via `xcrun metallib`.
    let status = Command::new("xcrun")
        .arg("metallib")
        .arg(out.with_extension("air"))
        .arg("-o")
        .arg(out)
        .status();
    matches!(status, Ok(s) if s.success()) && out.exists()
}

/// Compile a minimal MPP `matmul2d` kernel with the 26.2 flag to confirm the
/// offline toolchain can produce the (correct-K) NAX path at all.
fn probe_toolchain(metal_bin: &Path, out_dir: &Path) -> bool {
    let probe = out_dir.join("aot_probe.metal");
    let src = "#include <metal_stdlib>\n\
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
        using namespace metal;\n\
        [[kernel]] void probe(device float* o [[buffer(0)]]) {\n\
        constexpr auto d = mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,true,false,\n\
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n\
        mpp::tensor_ops::matmul2d<d, metal::execution_simdgroup> g; (void)g; o[0]=1.0f; }\n";
    if std::fs::write(&probe, src).is_err() {
        return false;
    }
    compile_metallib(metal_bin, &probe, &out_dir.join("aot_probe.metallib"))
}
