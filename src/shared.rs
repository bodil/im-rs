//! Automatic `Arc` wrapping.

use std::sync::Arc;

/// # Automatic `Arc` wrapping
///
/// The `Shared` trait provides automatic wrapping for things
/// which take `Arc`s, meaning that anything which takes an argument
/// of type `Shared<A>` will accept either an `A` or an `Arc<A>`.
///
/// Because everything stored in `im`'s persistent data structures
/// is wrapped in `Arc`s, `Shared` makes you have to worry less about
/// whether what you've got is an `A` or an `Arc<A>` - the compiler
/// will just figure it out for you, which is as it should be.
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
