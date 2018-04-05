// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::hash::Hasher;

pub fn is_sorted<A, I>(l: I) -> bool
where
    I: IntoIterator<Item = A>,
    A: Ord,
{
    let mut it = l.into_iter().peekable();
    loop {
        match (it.next(), it.peek()) {
            (_, None) => return true,
            (Some(ref a), Some(b)) if a > b => return false,
            _ => (),
        }
    }
}

pub struct LolHasher {
    state: u64,
    shift: usize,
}

impl LolHasher {
    fn feed_me(&mut self, byte: u8) {
        self.state ^= u64::from(byte) << self.shift;
        self.shift += 8;
        if self.shift >= 64 {
            self.shift = 0;
        }
    }
}

impl Hasher for LolHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.feed_me(*byte)
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for LolHasher {
    fn default() -> Self {
        LolHasher { state: 0, shift: 0 }
    }
}
