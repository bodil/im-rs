use std::hash::{BuildHasher, Hash, Hasher};

use super::bits::{Bitmap, HASH_COERCE};

pub fn hash_key<K: Hash, S: BuildHasher>(bh: &S, key: &K) -> Bitmap {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    (hasher.finish() & HASH_COERCE) as Bitmap
}
