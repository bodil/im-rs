// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Bitmap sizes for array mapped tries.

use std::hash::{BuildHasher, Hash, Hasher};

pub type Bitmap = u64; // a uint of HASH_SIZE bits
pub const HASH_BITS: usize = 6;
pub const HASH_SIZE: usize = 1 << HASH_BITS;
pub const HASH_MASK: Bitmap = (HASH_SIZE - 1) as Bitmap;
//pub const HASH_COERCE: u64 = ((1 << HASH_SIZE as u64) - 1);

#[inline]
pub fn mask(hash: Bitmap, shift: usize) -> Bitmap {
    hash >> shift & HASH_MASK
}

#[inline]
pub fn bitpos(hash: Bitmap, shift: usize) -> Bitmap {
    1 << mask(hash, shift)
}

#[inline]
pub fn index(bitmap: Bitmap, bit: Bitmap) -> usize {
    (bitmap & (bit - 1)).count_ones() as usize
}

pub fn hash_key<K: Hash + ?Sized, S: BuildHasher>(bh: &S, key: &K) -> Bitmap {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    // (hasher.finish() & HASH_COERCE) as Bitmap
    hasher.finish() as Bitmap
}
