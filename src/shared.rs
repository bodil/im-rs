// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Automatic `Arc` wrapping.

use std::sync::Arc;

/// # Automatic `Arc` wrapping
///
/// The `Shared` trait provides automatic wrapping for things which
/// take [`Arc`][std::sync::Arc]s, meaning that anything which takes
/// an argument of type `Shared<A>` will accept either an `A` or an
/// `Arc<A>`.
///
/// Because everything stored in `im`'s persistent data structures is
/// wrapped in [`Arc`][std::sync::Arc]s, `Shared` makes you have to
/// worry less about whether what you've got is an `A` or an `Arc<A>`
/// or a reference to such - the compiler will just figure it out for
/// you, which is as it should be.
///
/// [std::sync::Arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
pub trait Shared<A> {
    fn shared(self) -> Arc<A>;
}

impl<A> Shared<A> for A {
    fn shared(self) -> Arc<A> {
        Arc::new(self)
    }
}

impl<'a, A> Shared<A> for &'a A
where
    A: Clone,
{
    fn shared(self) -> Arc<A> {
        Arc::new(self.clone())
    }
}

impl<A> Shared<A> for Arc<A> {
    fn shared(self) -> Arc<A> {
        self
    }
}

impl<'a, A> Shared<A> for &'a Arc<A> {
    fn shared(self) -> Arc<A> {
        self.clone()
    }
}
