// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::hash::{BuildHasher, Hash, Hasher};

pub const HASH_SHIFT: usize = 5;
pub const HASH_SIZE: usize = 1 << HASH_SHIFT;

pub type HashBits = u32; // a uint of HASH_SIZE bits
pub const HASH_MASK: HashBits = (HASH_SIZE - 1) as HashBits;
//pub const HASH_COERCE: u64 = ((1 << HASH_SIZE as u64) - 1);

#[inline]
pub fn mask(hash: HashBits, shift: usize) -> HashBits {
    hash >> shift & HASH_MASK
}

pub fn hash_key<K: Hash + ?Sized, S: BuildHasher>(bh: &S, key: &K) -> HashBits {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    // (hasher.finish() & HASH_COERCE) as HashBits
    hasher.finish() as HashBits
}

#[derive(PartialEq, Eq, Clone, Copy, Default)]
pub struct Bitmap {
    data: HashBits,
}

impl Bitmap {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn get(self, index: usize) -> bool {
        self.data & (1 << index) != 0
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) -> bool {
        let mask = 1 << index;
        let prev = self.data & mask;
        if value {
            self.data |= mask;
        } else {
            self.data &= !mask;
        }
        prev != 0
    }

    #[inline]
    pub fn len(self) -> usize {
        self.data.count_ones() as usize
    }
}

impl IntoIterator for Bitmap {
    type Item = usize;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            index: 0,
            data: self.data,
        }
    }
}

pub struct Iter {
    index: usize,
    data: HashBits,
}

impl Iterator for Iter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= HASH_SIZE {
            return None;
        }
        if self.data & 1 != 0 {
            self.data >>= 1;
            self.index += 1;
            Some(self.index - 1)
        } else {
            self.data >>= 1;
            self.index += 1;
            self.next()
        }
    }
}
