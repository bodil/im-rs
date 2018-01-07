use std::sync::Arc;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

lazy_static! {
    static ref DEFAULT_HASHER: Arc<RandomState> = Arc::new(RandomState::new());
}

pub trait SharedHasher: BuildHasher {
    fn shared_hasher() -> Arc<Self>;
}

impl SharedHasher for RandomState {
    #[inline]
    fn shared_hasher() -> Arc<Self> {
        DEFAULT_HASHER.clone()
    }
}
