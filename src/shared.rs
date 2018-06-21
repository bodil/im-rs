// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Automatic `Ref` wrapping.

use util::Ref;

/// # Automatic `Ref` wrapping
///
/// The `Shared` trait provides automatic wrapping for things which
/// take `Ref`s, meaning that anything which takes
/// an argument of type `Shared<A>` will accept either an `A` or a
/// `Ref<A>`.
pub trait Shared<A> {
    fn shared(self) -> Ref<A>;
}

impl<A> Shared<A> for A {
    fn shared(self) -> Ref<A> {
        Ref::new(self)
    }
}

impl<'a, A> Shared<A> for &'a A
where
    A: Clone,
{
    fn shared(self) -> Ref<A> {
        Ref::new(self.clone())
    }
}

impl<A> Shared<A> for Ref<A> {
    fn shared(self) -> Ref<A> {
        self
    }
}

impl<'a, A> Shared<A> for &'a Ref<A> {
    fn shared(self) -> Ref<A> {
        self.clone()
    }
}
