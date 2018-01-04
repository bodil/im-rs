// This module works around the fact that you can't rely on
// `Arc<T>: From<U>` to only have one possible implementation anymore
// (that is, where `U = T`).

use std::sync::Arc;

pub trait Shared<T> {
    fn shared(self) -> Arc<T>;
}

impl<T> Shared<T> for T {
    fn shared(self) -> Arc<T> {
        Arc::from(self)
    }
}

impl<T> Shared<T> for Arc<T> {
    fn shared(self) -> Arc<T> {
        self
    }
}
