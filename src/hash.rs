//! Tools for hashing.

use std::sync::Arc;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

use bits::{Bitmap, HASH_COERCE};

lazy_static! {
    static ref DEFAULT_HASHER: Arc<RandomState> = Arc::new(RandomState::new());
}

/// A trait for acquiring a global shared [`BuildHasher`](std::hash::BuildHasher) instance.
///
/// This is desirable because, unlike mutable data structures, persistent
/// data structures can share data between them. Shared subtrees of hash maps
/// must be using the same hasher to be compatible with each other, and so
/// this trait is provided to make that easier to accomplish without increased
/// boilerplate and one more thing to keep track of.
///
/// [std::hash::BuildHasher]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
pub trait SharedHasher: BuildHasher {
    /// Get a reference to a global instance of [`BuildHasher`](std::hash::BuildHasher) for
    /// the hashing algorithm we're implementing.
    ///
    /// [std::hash::BuildHasher]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    fn shared_hasher() -> Arc<Self>;
}

impl SharedHasher for RandomState {
    #[inline]
    fn shared_hasher() -> Arc<Self> {
        DEFAULT_HASHER.clone()
    }
}

pub fn hash_key<K: Hash, S: BuildHasher>(bh: &S, key: &K) -> Bitmap {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    (hasher.finish() & HASH_COERCE) as Bitmap
}
