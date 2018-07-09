// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Every codebase needs a `util` module.

use std::ops::IndexMut;
use std::ptr;

// The `Ref` type is an alias for either `Rc` or `Arc`, user's choice.
#[cfg(feature = "arc")]
use std::sync::Arc;
#[cfg(feature = "arc")]
pub type Ref<A> = Arc<A>;
#[cfg(not(feature = "arc"))]
use std::rc::Rc;
#[cfg(not(feature = "arc"))]
pub type Ref<A> = Rc<A>;

pub fn clone_ref<A>(r: Ref<A>) -> A
where
    A: Clone,
{
    Ref::try_unwrap(r).unwrap_or_else(|r| (*r).clone())
}

/// Swap two values of anything implementing `IndexMut`.
///
/// Like `slice::swap`, but more generic.
pub fn swap_indices<V>(vector: &mut V, a: usize, b: usize)
where
    V: IndexMut<usize>,
    V::Output: Sized,
{
    if a == b {
        return;
    }
    // so sorry, but there's no implementation for this in std that's
    // sufficiently generic
    unsafe {
        let pa: *mut V::Output = &mut vector[a];
        let pb: *mut V::Output = &mut vector[b];
        ptr::swap(pa, pb);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}
