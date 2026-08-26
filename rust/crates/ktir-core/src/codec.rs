// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! dtype <-> byte codecs for the HBM/LX boundary. Tile data is flat `Vec<f32>`
//! (see `tile.rs`); these encode/decode it to the raw bytes the memory
//! simulator stores, per the source dtype. `f16` uses an inline IEEE
//! half-precision round-trip (round-to-nearest-even) — no `half` dependency.
//!
//! (Note: `arith.rs` and `ops_memory.rs` currently carry their own private
//! copies of the f16 round-trip; consolidating them onto this module is a
//! follow-up cleanup.)

use crate::dtypes::DType;

/// Decode IEEE-754 half-precision bits to f32.
/// Convert IEEE-754 half-precision bits to f32. f16 has only 65536 possible bit
/// patterns, so this is a single load from a lazily-built 256 KB lookup table —
/// the hot path of every `ktdp.load` / `read_back` / `round_to_dtype` (the
/// dominant cost in a real-model run profile). The exact arithmetic lives in
/// [`f16_bits_to_f32_compute`], which fills the table.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| (0..=u16::MAX).map(f16_bits_to_f32_compute).collect())[h as usize]
}

/// Reference f16→f32 arithmetic (round-trip exact); used to build the table.
fn f16_bits_to_f32_compute(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val: f32 = if exp == 0 {
        // subnormal / zero
        (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

/// Encode f32 to IEEE-754 half-precision bits (round-to-nearest-even).
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if f.is_nan() {
        return sign | 0x7e00;
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        // subnormal / underflow
        if exp < -10 {
            return sign;
        }
        let mant_full = mant | 0x800000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant_full >> shift;
        // round-to-nearest-even
        let rem = mant_full & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half_mant & 1) == 1) {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }
    let mut half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1) {
        half_mant += 1;
    }
    // ADD (not OR) the mantissa so a rounding carry (half_mant -> 0x400) spills
    // into the exponent: e.g. 32767.994 rounds up to 32768 (exp+1, mant 0), and
    // a carry that reaches exp 0x1f naturally yields inf. `| half_mant` would
    // collide with the exponent's low bit when exp is odd. Matches hardware RNE.
    sign | (((exp as u16) << 10) + half_mant)
}

/// Round each value in place to `dtype`'s representable set — NumPy assignment
/// semantics for a typed array. `f16` rounds to nearest-even half precision;
/// integer dtypes truncate toward zero; `bool` maps nonzero -> 1. `f32` is a
/// no-op. Used by `Tile::compute` so op results round per-op like NumPy.
pub fn round_to_dtype(data: &mut [f32], dtype: DType) {
    match dtype {
        DType::F32 => {}
        DType::F16 => round_f16_in_place(data),
        DType::I32 => {
            for x in data.iter_mut() {
                *x = (*x as i32) as f32;
            }
        }
        DType::I64 => {
            for x in data.iter_mut() {
                *x = (*x as i64) as f32;
            }
        }
        DType::Bool => {
            for x in data.iter_mut() {
                *x = if *x != 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
}

/// Encode a flat f32 tile into raw bytes for memory, per `dtype`.
pub fn encode(data: &[f32], dtype: DType) -> Vec<u8> {
    // f16 is the hot dtype in a real-model run — convert 4 lanes/instruction
    // with hardware `vcvt_f16_f32` (round-to-nearest-even, matching the scalar
    // `f32_to_f16_bits`) instead of the per-element bit twiddle.
    if dtype == DType::F16 {
        return encode_f16(data);
    }
    let mut out = Vec::with_capacity(data.len() * dtype.bytes_per_elem());
    for &v in data {
        match dtype {
            DType::F16 => out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes()),
            DType::F32 => out.extend_from_slice(&v.to_le_bytes()),
            DType::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
            DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
            DType::Bool => out.push((v != 0.0) as u8),
        }
    }
    out
}

/// Decode `n` elements of `dtype` from raw bytes into a flat f32 tile.
/// Zero-pads if `bytes` is short (matches the memory sim's zero-fill).
pub fn decode(bytes: &[u8], n: usize, dtype: DType) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    decode_into(bytes, &mut out, dtype);
    out
}

/// Decode `out.len()` elements of `dtype` from `bytes` directly INTO `out` — the
/// no-allocation analogue of [`decode`], for hot paths that decode many runs into
/// one preallocated buffer (e.g. the row-contiguous load reading each strided
/// run). Zero-pads when `bytes` is short, matching [`decode`].
pub fn decode_into(bytes: &[u8], out: &mut [f32], dtype: DType) {
    let n = out.len();
    // f16 fast path: the bytes are fully present (the common case — only the
    // zero-pad-short fallback below needs per-element bounds checks). f16→f32 is
    // exact, so hardware `vcvt_f32_f16` over 4 lanes matches the table exactly.
    if dtype == DType::F16 && bytes.len() >= n * 2 {
        decode_f16_into(bytes, out);
        return;
    }
    let bpe = dtype.bytes_per_elem();
    for (i, o) in out.iter_mut().enumerate() {
        let off = i * bpe;
        let chunk = bytes.get(off..off + bpe);
        *o = match (dtype, chunk) {
            (_, None) => 0.0,
            (DType::F16, Some(c)) => f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])),
            (DType::F32, Some(c)) => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            (DType::I32, Some(c)) => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32,
            (DType::I64, Some(c)) => {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            }
            (DType::Bool, Some(c)) => (c[0] != 0) as i32 as f32,
        };
    }
}

/// bfloat16 bytes (little-endian) -> f32 tile. bf16 is NOT a Spyre/KTIR HBM dtype
/// (the hardware is f16) — this is a host-side convenience for ingesting bf16
/// model weights (the stock Llama/SmolLM2 checkpoint format), which the caller
/// then narrows to the f16 stick layout. bf16→f32 is EXACT and zero-arithmetic:
/// a bf16 value is simply the high 16 bits of the f32 with the same sign,
/// exponent, and 7 mantissa bits, so widening is `(bits as u32) << 16`. A short
/// tail zero-pads (matches [`decode`]). The full-length fast path is SIMD; the
/// short-tail / partial-byte case falls back to scalar.
pub fn bf16_to_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    if bytes.len() >= n * 2 {
        bf16_to_f32_into(bytes, &mut out);
    } else {
        // Short input: per-element with the zero-pad fallback.
        for (i, o) in out.iter_mut().enumerate() {
            if let Some(c) = bytes.get(i * 2..i * 2 + 2) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
    }
    out
}

/// bf16 bytes (fully present) -> f32, batched. A bf16→f32 widen is a pure 16-bit
/// left-shift into the f32 high half — no `vcvt` needed; NEON's widening shift
/// `vshll_n_u16` lifts 4 u16 lanes to 4 u32 lanes in one instruction.
#[cfg(target_arch = "aarch64")]
fn bf16_to_f32_into(bytes: &[u8], out: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = out.len();
    debug_assert!(bytes.len() >= n * 2);
    unsafe {
        let (src, dst) = (bytes.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i + 8 <= n {
            // Unaligned 8×u16 load -> two 4-lane widening shifts (<<16) -> 8×f32.
            let h = vld1q_u16(src.add(i * 2).cast::<u16>());
            let lo = vshll_n_u16::<16>(vget_low_u16(h));
            let hi = vshll_n_u16::<16>(vget_high_u16(h));
            vst1q_f32(dst.add(i), vreinterpretq_f32_u32(lo));
            vst1q_f32(dst.add(i + 4), vreinterpretq_f32_u32(hi));
            i += 8;
        }
        while i < n {
            let (lo, hi) = (*src.add(i * 2), *src.add(i * 2 + 1));
            *dst.add(i) = f32::from_bits((u16::from_le_bytes([lo, hi]) as u32) << 16);
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn bf16_to_f32_into(bytes: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16);
    }
}

/// bfloat16 bytes (little-endian) -> f16 bytes (little-endian) — the HBM stick
/// layout. This is the path a bf16 weight takes into Spyre's f16 HBM: it does the
/// REAL format conversion (bf16's 8e/7m → f16's 5e/10m, RNE, overflow→inf), but
/// FUSED — it never materializes an intermediate `f32` tile. Conceptually it's
/// bf16→f32 (exact `<<16`) → f32→f16 (round-to-nearest-even), done per element in
/// registers, so the output is bit-identical to `encode(bf16_to_f32(..), F16)`
/// without the extra full-length f32 buffer + second pass. A short input
/// zero-pads (a zero bf16 narrows to a zero f16). Use this for ingest into an
/// f16 stick; use [`bf16_to_f32`] only for the f32 oracle path.
pub fn bf16_to_f16(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * 2];
    if bytes.len() >= n * 2 {
        bf16_to_f16_into(bytes, &mut out);
    } else {
        for i in 0..n {
            let v = match bytes.get(i * 2..i * 2 + 2) {
                Some(c) => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
                None => 0.0,
            };
            let b = f32_to_f16_bits(v).to_le_bytes();
            out[i * 2] = b[0];
            out[i * 2 + 1] = b[1];
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn bf16_to_f16_into(bytes: &[u8], out: &mut [u8]) {
    use core::arch::aarch64::*;
    let n = out.len() / 2;
    debug_assert!(bytes.len() >= n * 2);
    unsafe {
        let (src, dst) = (bytes.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i + 8 <= n {
            // 8×bf16 -> two 4-lane (widen <<16 -> reinterpret f32 -> narrow f16).
            let h = vld1q_u16(src.add(i * 2).cast::<u16>());
            let lo = vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h))));
            let hi = vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(h))));
            vst1_u16(dst.add(i * 2).cast::<u16>(), vreinterpret_u16_f16(lo));
            vst1_u16(dst.add((i + 4) * 2).cast::<u16>(), vreinterpret_u16_f16(hi));
            i += 8;
        }
        while i < n {
            let v = f32::from_bits(
                (u16::from_le_bytes([*src.add(i * 2), *src.add(i * 2 + 1)]) as u32) << 16,
            );
            let b = f32_to_f16_bits(v).to_le_bytes();
            *dst.add(i * 2) = b[0];
            *dst.add(i * 2 + 1) = b[1];
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn bf16_to_f16_into(bytes: &[u8], out: &mut [u8]) {
    for i in 0..(out.len() / 2) {
        let v = f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16);
        let b = f32_to_f16_bits(v).to_le_bytes();
        out[i * 2] = b[0];
        out[i * 2 + 1] = b[1];
    }
}

// ===========================================================================
// f16 batch conversion — SIMD on aarch64 (hardware FP16), scalar elsewhere.
//
// f16 dominates real-model self-time (every ktdp.load decodes, every store and
// round_to_dtype encodes). Apple Silicon has native half<->single conversion:
// `vcvt_f32_f16` / `vcvt_f16_f32` do 4 lanes per instruction with no memory
// traffic, versus the 256 KB f16->f32 lookup table thrashing L1. The hardware
// converters use round-to-nearest-even — bit-identical to the scalar helpers
// for finite values (verified exhaustively in the tests below).
// ===========================================================================

/// Decode a slice of f16 bit patterns (one `u16` per element) to f32. Exact —
/// the native-storage analogue of [`decode`] for `Tile`'s `F16` arm, routed
/// through the same `decode`/SIMD path so it is bit-identical. f16 → f32 is
/// lossless.
pub fn f16_units_to_f32(units: &[u16]) -> Vec<f32> {
    let mut bytes = Vec::with_capacity(units.len() * 2);
    for &u in units {
        bytes.extend_from_slice(&u.to_le_bytes());
    }
    decode(&bytes, units.len(), DType::F16)
}

/// Encode an f32 slice to f16 bit patterns (one `u16` per element),
/// round-to-nearest-even — the native-storage analogue of [`encode`] for
/// `Tile`'s `F16` arm, routed through the same `encode`/SIMD path so it is
/// bit-identical.
pub fn f32_to_f16_units(data: &[f32]) -> Vec<u16> {
    let bytes = encode(data, DType::F16);
    (0..data.len())
        .map(|i| u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]))
        .collect()
}

/// f32 tile -> f16 bytes (little-endian), round-to-nearest-even.
fn encode_f16(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() * 2];
    encode_f16_into(data, &mut out);
    out
}

/// Quantize each f32 to its nearest f16 value, in place (f32->f16->f32).
fn round_f16_in_place(data: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let (p, n) = (data.as_mut_ptr(), data.len());
        let mut i = 0;
        while i + 4 <= n {
            let f = vld1q_f32(p.add(i));
            vst1q_f32(p.add(i), vcvt_f32_f16(vcvt_f16_f32(f)));
            i += 4;
        }
        while i < n {
            *p.add(i) = f16_bits_to_f32(f32_to_f16_bits(*p.add(i)));
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in data.iter_mut() {
        *x = f16_bits_to_f32(f32_to_f16_bits(*x));
    }
}

#[cfg(target_arch = "aarch64")]
fn decode_f16_into(bytes: &[u8], out: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = out.len();
    debug_assert!(bytes.len() >= n * 2);
    unsafe {
        let (src, dst) = (bytes.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i + 4 <= n {
            // Unaligned 4×u16 load -> reinterpret as f16 -> widen to 4×f32.
            let h = vld1_u16(src.add(i * 2).cast::<u16>());
            vst1q_f32(dst.add(i), vcvt_f32_f16(vreinterpret_f16_u16(h)));
            i += 4;
        }
        while i < n {
            let (lo, hi) = (*src.add(i * 2), *src.add(i * 2 + 1));
            *dst.add(i) = f16_bits_to_f32(u16::from_le_bytes([lo, hi]));
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn decode_f16_into(bytes: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = f16_bits_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
    }
}

#[cfg(target_arch = "aarch64")]
fn encode_f16_into(data: &[f32], out: &mut [u8]) {
    use core::arch::aarch64::*;
    let n = data.len();
    debug_assert!(out.len() >= n * 2);
    unsafe {
        let (src, dst) = (data.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i + 4 <= n {
            // 4×f32 -> narrow to 4×f16 (RNE) -> reinterpret u16 -> unaligned store.
            let f = vld1q_f32(src.add(i));
            let h = vreinterpret_u16_f16(vcvt_f16_f32(f));
            vst1_u16(dst.add(i * 2).cast::<u16>(), h);
            i += 4;
        }
        while i < n {
            let b = f32_to_f16_bits(*src.add(i)).to_le_bytes();
            *dst.add(i * 2) = b[0];
            *dst.add(i * 2 + 1) = b[1];
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn encode_f16_into(data: &[f32], out: &mut [u8]) {
    for (i, &v) in data.iter().enumerate() {
        let b = f32_to_f16_bits(v).to_le_bytes();
        out[i * 2] = b[0];
        out[i * 2 + 1] = b[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_for_small_ints() {
        for v in [0.0f32, 1.0, -1.0, 2.5, 128.0, -0.5, 1024.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert_eq!(back, v, "f16 round-trip for {v}");
        }
    }

    // The SIMD f16 batch path must be bit-identical to the scalar helpers for
    // every finite f16 value — otherwise it would silently shift golden output.
    // f16 has only 65536 patterns, so we can check all of them exhaustively.
    #[test]
    fn simd_f16_decode_matches_scalar_for_all_patterns() {
        let bytes: Vec<u8> = (0..=u16::MAX).flat_map(|h| h.to_le_bytes()).collect();
        let n = 1 << 16;
        let simd = decode(&bytes, n, DType::F16); // hits decode_f16 fast path
        for h in 0..=u16::MAX {
            let want = f16_bits_to_f32(h);
            let got = simd[h as usize];
            if want.is_nan() {
                assert!(got.is_nan(), "pattern {h:#06x}: want NaN, got {got}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "decode pattern {h:#06x}");
            }
        }
    }

    #[test]
    fn simd_f16_encode_matches_scalar_for_all_representable() {
        // Widen every f16 value to f32, then re-encode: the SIMD narrow must
        // produce the original bit pattern (round-trip is exact for these),
        // matching the scalar `f32_to_f16_bits`.
        let vals: Vec<f32> = (0..=u16::MAX).map(f16_bits_to_f32).collect();
        let simd = encode(&vals, DType::F16); // hits encode_f16 fast path
        for (i, &v) in vals.iter().enumerate() {
            if v.is_nan() {
                // NaN payloads are don't-care; just require an f16 NaN out.
                let got = u16::from_le_bytes([simd[i * 2], simd[i * 2 + 1]]);
                assert_eq!(got & 0x7c00, 0x7c00, "encode NaN slot {i}");
                assert_ne!(got & 0x03ff, 0, "encode NaN must keep mantissa");
                continue;
            }
            let got = u16::from_le_bytes([simd[i * 2], simd[i * 2 + 1]]);
            assert_eq!(got, f32_to_f16_bits(v), "encode value {v} (slot {i})");
        }
    }

    #[test]
    fn simd_f16_encode_matches_scalar_for_unrepresentable() {
        // Values needing real rounding (not f16-exact): SIMD RNE must equal the
        // scalar RNE bit-for-bit. Sweep a dense range across magnitudes.
        let mut vals = Vec::new();
        let mut x = -70000.0f32;
        while x < 70000.0 {
            vals.push(x);
            x += 0.013;
        }
        let simd = encode(&vals, DType::F16);
        for (i, &v) in vals.iter().enumerate() {
            let got = u16::from_le_bytes([simd[i * 2], simd[i * 2 + 1]]);
            assert_eq!(
                got,
                f32_to_f16_bits(v),
                "encode rounding for {v} (slot {i})"
            );
        }
    }

    #[test]
    fn simd_round_f16_matches_scalar() {
        // Odd length to exercise the scalar tail after the 4-lane body.
        let mut a: Vec<f32> = (0..103).map(|i| (i as f32) * 0.37 - 12.5).collect();
        let mut b = a.clone();
        round_to_dtype(&mut a, DType::F16); // SIMD round_f16_in_place
        for x in b.iter_mut() {
            *x = f16_bits_to_f32(f32_to_f16_bits(*x));
        }
        assert_eq!(a, b);
    }

    #[test]
    fn f16_inf_encoding() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xfc00), f32::NEG_INFINITY);
    }

    // bf16 = the high 16 bits of an f32, so widening is exact and lossless.
    #[test]
    fn bf16_widens_known_values() {
        // (bf16 bits, exact f32)
        for (bits, want) in [
            (0x0000u16, 0.0f32),
            (0x8000, -0.0),
            (0x3f80, 1.0),
            (0xbf80, -1.0),
            (0x4000, 2.0),
            (0x4049, 3.140625), // bf16(pi)
            (0x7f80, f32::INFINITY),
            (0xff80, f32::NEG_INFINITY),
        ] {
            let got = bf16_to_f32(&bits.to_le_bytes(), 1)[0];
            assert_eq!(got.to_bits(), want.to_bits(), "bf16 {bits:#06x}");
        }
    }

    // The SIMD bf16 path must equal the spec definition `(bits as u32) << 16` for
    // every one of the 65536 bf16 patterns. Odd `n` exercises the scalar tail
    // after the 8-lane SIMD body.
    #[test]
    fn simd_bf16_matches_scalar_for_all_patterns() {
        let bytes: Vec<u8> = (0..=u16::MAX).flat_map(|h| h.to_le_bytes()).collect();
        let n = 1 << 16;
        let got = bf16_to_f32(&bytes, n); // full-length SIMD path
        for h in 0..=u16::MAX {
            let want = f32::from_bits((h as u32) << 16);
            if want.is_nan() {
                assert!(got[h as usize].is_nan(), "bf16 {h:#06x}: want NaN");
            } else {
                assert_eq!(got[h as usize].to_bits(), want.to_bits(), "bf16 {h:#06x}");
            }
        }
        // Odd-length slice: same values, exercises the tail.
        let m = 65533;
        let tail = bf16_to_f32(&bytes[..m * 2], m);
        for (h, &got) in tail.iter().enumerate() {
            let want = f32::from_bits(((h as u16) as u32) << 16);
            if !want.is_nan() {
                assert_eq!(got.to_bits(), want.to_bits(), "bf16 tail slot {h}");
            }
        }
    }

    // A short input zero-pads past its end (matches `decode`).
    #[test]
    fn bf16_zero_pads_short_input() {
        let got = bf16_to_f32(&0x3f80u16.to_le_bytes(), 4); // one value, ask for 4
        assert_eq!(got, vec![1.0, 0.0, 0.0, 0.0]);
    }

    // The FUSED bf16->f16 path (no f32 intermediate) must be bit-identical to the
    // two-step `encode(bf16_to_f32(..), F16)` for every bf16 pattern — that's the
    // whole point of fusing it. Odd length exercises the scalar tail.
    #[test]
    fn fused_bf16_to_f16_matches_two_step_for_all_patterns() {
        let bytes: Vec<u8> = (0..=u16::MAX).flat_map(|h| h.to_le_bytes()).collect();
        let n = 1 << 16;
        let fused = bf16_to_f16(&bytes, n);
        let two_step = encode(&bf16_to_f32(&bytes, n), DType::F16);
        assert_eq!(fused, two_step, "fused bf16->f16 must equal bf16->f32->f16");

        // And the produced f16 bytes decode back to the f16-rounded bf16 value.
        let m = 4095; // odd-ish, hits the tail
        let f16_bytes = bf16_to_f16(&bytes[..m * 2], m);
        let back = decode(&f16_bytes, m, DType::F16);
        for (h, &got) in back.iter().enumerate() {
            let want = f16_bits_to_f32(f32_to_f16_bits(f32::from_bits(((h as u16) as u32) << 16)));
            if !want.is_nan() {
                assert_eq!(got.to_bits(), want.to_bits(), "fused bf16->f16 slot {h}");
            }
        }
    }

    // bf16 values outside f16's range must saturate to f16 infinity (not wrap):
    // bf16 carries f32's full exponent, so e.g. 1e30 is representable in bf16 but
    // overflows f16.
    #[test]
    fn bf16_to_f16_overflows_to_inf() {
        let big = f32::from_bits(0x7149_0000); // bf16 ~ 1e30
        let bf16_bits = (big.to_bits() >> 16) as u16;
        let f16 = bf16_to_f16(&bf16_bits.to_le_bytes(), 1);
        let v = decode(&f16, 1, DType::F16)[0];
        assert_eq!(v, f32::INFINITY, "huge bf16 must saturate to f16 +inf");
    }

    #[test]
    fn encode_decode_roundtrips_each_dtype() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        for dt in [DType::F16, DType::F32, DType::I32, DType::I64] {
            let bytes = encode(&data, dt);
            assert_eq!(bytes.len(), 4 * dt.bytes_per_elem());
            assert_eq!(decode(&bytes, 4, dt), data, "round-trip {dt}");
        }
    }

    #[test]
    fn decode_zero_pads_short_input() {
        assert_eq!(decode(&[], 3, DType::F32), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn round_to_dtype_matches_numpy_assignment() {
        // f16 rounds to nearest-even half precision (0.1 is not f16-exact).
        let mut f = vec![0.1f32];
        round_to_dtype(&mut f, DType::F16);
        assert_eq!(f[0], f16_bits_to_f32(f32_to_f16_bits(0.1)));
        assert_ne!(f[0], 0.1, "0.1 must round under f16");
        // integer dtypes truncate toward zero; bool maps nonzero -> 1.
        let mut i = vec![2.9f32, -2.9];
        round_to_dtype(&mut i, DType::I32);
        assert_eq!(i, vec![2.0, -2.0]);
        let mut b = vec![0.0f32, 5.0];
        round_to_dtype(&mut b, DType::Bool);
        assert_eq!(b, vec![0.0, 1.0]);
        // f32 is exact (no-op) — 0.15625 = 5/32 is f32-exact.
        let mut g = vec![0.15625f32];
        round_to_dtype(&mut g, DType::F32);
        assert_eq!(g[0], 0.15625);
    }
}

#[cfg(test)]
mod tile_round_tests {
    use crate::dtypes::DType;
    use crate::tile::Tile;

    #[test]
    fn tile_compute_rounds_f16_per_op() {
        // A chain of f16 ops rounds each step (NumPy float16 semantics): building
        // an f16 tile stores the f16-rounded value, not the exact f32 input.
        let t = Tile::compute(vec![0.1, 0.2, 0.3], DType::F16, vec![3]);
        for (&got, &raw) in t.as_f32().iter().zip(&[0.1f32, 0.2, 0.3]) {
            assert_eq!(
                got,
                crate::codec::f16_bits_to_f32(crate::codec::f32_to_f16_bits(raw))
            );
        }
        // f32 tiles keep exact values.
        let g = Tile::compute(vec![0.1, 0.2], DType::F32, vec![2]);
        assert_eq!(g.as_f32().to_vec(), vec![0.1, 0.2]);
    }
}
