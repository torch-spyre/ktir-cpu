// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! A tiny, dependency-free FxHash — the fast non-cryptographic hash rustc uses
//! internally. The default `HashMap` hasher (SipHash) is DoS-resistant but slow
//! for the short string keys this interpreter hammers (SSA names, op types). The
//! scope, dispatch, and memory maps don't need DoS resistance, so they use this.
//!
//! `FxHashMap<K, V>` is a drop-in `HashMap` alias with this hasher.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// FxHasher — the rotate-multiply-xor hash from rustc's `rustc_hash`.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[..8]);
            self.add(u64::from_le_bytes(b));
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut b = [0u8; 8];
            b[..bytes.len()].copy_from_slice(bytes);
            self.add(u64::from_le_bytes(b));
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// `HashMap` keyed with [`FxHasher`].
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
