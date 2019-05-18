// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A persistent vector.
//!
//! This is a sequence of elements in insertion order - if you need a
//! list of things, any kind of list of things, this is what you're
//! looking for.
//!
//! It's implemented as an [RRB vector][rrbpaper] with [smart
//! head/tail chunking][chunkedseq]. In performance terms, this means
//! that practically every operation is O(log n), except push/pop on
//! both sides, which will be O(1) amortised, and O(log n) in the
//! worst case. In practice, the push/pop operations will be
//! blindingly fast, nearly on par with the native
//! [`VecDeque`][VecDeque], and other operations will have decent, if
//! not high, performance, but they all have more or less the same
//! O(log n) complexity, so you don't need to keep their performance
//! characteristics in mind - everything, even splitting and merging,
//! is safe to use and never too slow.
//!
//! ## Performance Notes
//!
//! Because of the head/tail chunking technique, until you push a
//! number of items above double the tree's branching factor (that's
//! `self.len()` = 2 × *k* (where *k* = 64) = 128) on either side, the
//! data structure is still just a handful of arrays, not yet an RRB
//! tree, so you'll see performance and memory characteristics fairly
//! close to [`Vec`][Vec] or [`VecDeque`][VecDeque].
//!
//! This means that the structure always preallocates four chunks of
//! size *k* (*k* being the tree's branching factor), equivalent to a
//! [`Vec`][Vec] with an initial capacity of 256. Beyond that, it will
//! allocate tree nodes of capacity *k* as needed.
//!
//! In addition, vectors start out as single chunks, and only expand into the
//! full data structure once you go past the chunk size. This makes them
//! perform identically to [`Vec`][Vec] at small sizes.
//!
//! [rrbpaper]: https://infoscience.epfl.ch/record/213452/files/rrbvector.pdf
//! [chunkedseq]: http://deepsea.inria.fr/pasl/chunkedseq.pdf
//! [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//! [VecDeque]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::iter::{Chain, FromIterator, FusedIterator};
use std::mem::{replace, swap};
use std::ops::{Add, Index, IndexMut, RangeBounds};

use sized_chunks::{inline_array::Iter as InlineIter, InlineArray};

use crate::nodes::chunk::{Chunk, Iter as ChunkIter, CHUNK_SIZE};
use crate::nodes::rrb::{
    ConsumingIter as ConsumingNodeIter, Node, PopResult, PushResult, SplitResult,
};
use crate::sort;
use crate::util::{clone_ref, swap_indices, to_range, Ref, Side};

use self::Vector::{Full, Inline, Single};

mod focus;

pub use self::focus::{Focus, FocusMut};

/// Construct a vector from a sequence of elements.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::vector::Vector;
/// # fn main() {
/// assert_eq!(
///   vector![1, 2, 3],
///   Vector::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! vector {
    () => { $crate::vector::Vector::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::vector::Vector::new();
        $(
            l.push_back($x);
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::vector::Vector::new();
        $(
            l.push_back($x);
        )*
            l
    }};
}

/// A persistent vector.
///
/// This is a sequence of elements in insertion order - if you need a list of
/// things, any kind of list of things, this is what you're looking for.
///
/// It's implemented as an [RRB vector][rrbpaper] with [smart head/tail
/// chunking][chunkedseq]. In performance terms, this means that practically
/// every operation is O(log n), except push/pop on both sides, which will be
/// O(1) amortised, and O(log n) in the worst case. In practice, the push/pop
/// operations will be blindingly fast, nearly on par with the native
/// [`VecDeque`][VecDeque], and other operations will have decent, if not high,
/// performance, but they all have more or less the same O(log n) complexity, so
/// you don't need to keep their performance characteristics in mind -
/// everything, even splitting and merging, is safe to use and never too slow.
///
/// ## Performance Notes
///
/// Because of the head/tail chunking technique, until you push a number of
/// items above double the tree's branching factor (that's `self.len()` = 2 ×
/// *k* (where *k* = 64) = 128) on either side, the data structure is still just
/// a handful of arrays, not yet an RRB tree, so you'll see performance and
/// memory characteristics similar to [`Vec`][Vec] or [`VecDeque`][VecDeque].
///
/// This means that the structure always preallocates four chunks of size *k*
/// (*k* being the tree's branching factor), equivalent to a [`Vec`][Vec] with
/// an initial capacity of 256. Beyond that, it will allocate tree nodes of
/// capacity *k* as needed.
///
/// In addition, vectors start out as single chunks, and only expand into the
/// full data structure once you go past the chunk size. This makes them
/// perform identically to [`Vec`][Vec] at small sizes.
///
/// [rrbpaper]: https://infoscience.epfl.ch/record/213452/files/rrbvector.pdf
/// [chunkedseq]: http://deepsea.inria.fr/pasl/chunkedseq.pdf
/// [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
/// [VecDeque]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html
pub enum Vector<A> {
    #[doc(hidden)]
    Inline(InlineArray<A, RRB<A>>),
    #[doc(hidden)]
    Single(Ref<Chunk<A>>),
    #[doc(hidden)]
    Full(RRB<A>),
}

#[doc(hidden)]
pub struct RRB<A> {
    length: usize,
    middle_level: usize,
    outer_f: Ref<Chunk<A>>,
    inner_f: Ref<Chunk<A>>,
    middle: Ref<Node<A>>,
    inner_b: Ref<Chunk<A>>,
    outer_b: Ref<Chunk<A>>,
}

impl<A> Clone for RRB<A> {
    fn clone(&self) -> Self {
        RRB {
            length: self.length,
            middle_level: self.middle_level,
            outer_f: self.outer_f.clone(),
            inner_f: self.inner_f.clone(),
            middle: self.middle.clone(),
            inner_b: self.inner_b.clone(),
            outer_b: self.outer_b.clone(),
        }
    }
}

impl<A: Clone> Vector<A> {
    /// True if a vector is a full inline or single chunk, ie. must be promoted
    /// to grow further.
    fn needs_promotion(&self) -> bool {
        match self {
            Inline(chunk) if chunk.is_full() => true,
            Single(chunk) if chunk.is_full() => true,
            _ => false,
        }
    }

    /// Promote an inline to a single.
    fn promote_inline(&mut self) {
        if let Inline(chunk) = self {
            *self = Single(Ref::new(chunk.into()))
        }
    }

    /// Promote a single to a full, with the single chunk becoming inner_f, or
    /// promote an inline to a single.
    fn promote_front(&mut self) {
        *self = match self {
            Inline(chunk) => Single(Ref::new(chunk.into())),
            Single(chunk) => {
                let chunk = chunk.clone();
                Full(RRB {
                    length: chunk.len(),
                    middle_level: 0,
                    outer_f: Ref::new(Chunk::new()),
                    inner_f: chunk,
                    middle: Ref::new(Node::new()),
                    inner_b: Ref::new(Chunk::new()),
                    outer_b: Ref::new(Chunk::new()),
                })
            }
            Full(_) => return,
        }
    }

    /// Promote a single to a full, with the single chunk becoming inner_b, or
    /// promote an inline to a single.
    fn promote_back(&mut self) {
        *self = match self {
            Inline(chunk) => Single(Ref::new(chunk.into())),
            Single(chunk) => {
                let chunk = chunk.clone();
                Full(RRB {
                    length: chunk.len(),
                    middle_level: 0,
                    outer_f: Ref::new(Chunk::new()),
                    inner_f: Ref::new(Chunk::new()),
                    middle: Ref::new(Node::new()),
                    inner_b: chunk,
                    outer_b: Ref::new(Chunk::new()),
                })
            }
            Full(_) => return,
        }
    }

    /// Construct an empty vector.
    #[must_use]
    pub fn new() -> Self {
        Inline(InlineArray::new())
    }

    /// Get the length of a vector.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(5, vector![1, 2, 3, 4, 5].len());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Inline(chunk) => chunk.len(),
            Single(chunk) => chunk.len(),
            Full(tree) => tree.length,
        }
    }

    /// Test whether a vector is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let vec = vector!["Joe", "Mike", "Robert"];
    /// assert_eq!(false, vec.is_empty());
    /// assert_eq!(true, Vector::<i32>::new().is_empty());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an iterator over a vector.
    ///
    /// Time: O(1)
    #[inline]
    #[must_use]
    pub fn iter(&self) -> Iter<A> {
        Iter::new(self)
    }

    /// Get a mutable iterator over a vector.
    ///
    /// Time: O(1)
    #[inline]
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<A> {
        IterMut::new(self)
    }

    /// Get an iterator over the leaf nodes of a vector.
    ///
    /// This returns an iterator over the [`Chunk`s][Chunk] at the leaves of the
    /// RRB tree. These are useful for efficient parallelisation of work on
    /// the vector, but should not be used for basic iteration.
    ///
    /// Time: O(1)
    ///
    /// [Chunk]: ../chunk/struct.Chunk.html
    #[inline]
    #[must_use]
    pub fn leaves(&self) -> Chunks<'_, A> {
        Chunks::new(self)
    }

    /// Get a mutable iterator over the leaf nodes of a vector.
    //
    /// This returns an iterator over the [`Chunk`s][Chunk] at the leaves of the
    /// RRB tree. These are useful for efficient parallelisation of work on
    /// the vector, but should not be used for basic iteration.
    ///
    /// Time: O(1)
    ///
    /// [Chunk]: ../chunk/struct.Chunk.html
    #[inline]
    #[must_use]
    pub fn leaves_mut(&mut self) -> ChunksMut<'_, A> {
        ChunksMut::new(self)
    }

    /// Construct a [`Focus`][Focus] for a vector.
    ///
    /// Time: O(1)
    ///
    /// [Focus]: enum.Focus.html
    #[inline]
    #[must_use]
    pub fn focus(&self) -> Focus<'_, A> {
        Focus::new(self)
    }

    /// Construct a [`FocusMut`][FocusMut] for a vector.
    ///
    /// Time: O(1)
    ///
    /// [FocusMut]: enum.FocusMut.html
    #[inline]
    #[must_use]
    pub fn focus_mut(&mut self) -> FocusMut<'_, A> {
        FocusMut::new(self)
    }

    /// Get a reference to the value at index `index` in a vector.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let vec = vector!["Joe", "Mike", "Robert"];
    /// assert_eq!(Some(&"Robert"), vec.get(2));
    /// assert_eq!(None, vec.get(5));
    /// # }
    /// ```
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&A> {
        if index >= self.len() {
            return None;
        }

        match self {
            Inline(chunk) => chunk.get(index),
            Single(chunk) => chunk.get(index),
            Full(tree) => {
                let mut local_index = index;

                if local_index < tree.outer_f.len() {
                    return Some(&tree.outer_f[local_index]);
                }
                local_index -= tree.outer_f.len();

                if local_index < tree.inner_f.len() {
                    return Some(&tree.inner_f[local_index]);
                }
                local_index -= tree.inner_f.len();

                if local_index < tree.middle.len() {
                    return Some(tree.middle.index(tree.middle_level, local_index));
                }
                local_index -= tree.middle.len();

                if local_index < tree.inner_b.len() {
                    return Some(&tree.inner_b[local_index]);
                }
                local_index -= tree.inner_b.len();

                Some(&tree.outer_b[local_index])
            }
        }
    }

    /// Get a mutable reference to the value at index `index` in a
    /// vector.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector!["Joe", "Mike", "Robert"];
    /// {
    ///     let robert = vec.get_mut(2).unwrap();
    ///     assert_eq!(&mut "Robert", robert);
    ///     *robert = "Bjarne";
    /// }
    /// assert_eq!(vector!["Joe", "Mike", "Bjarne"], vec);
    /// # }
    /// ```
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut A> {
        if index >= self.len() {
            return None;
        }

        match self {
            Inline(chunk) => chunk.get_mut(index),
            Single(chunk) => Ref::make_mut(chunk).get_mut(index),
            Full(tree) => {
                let mut local_index = index;

                if local_index < tree.outer_f.len() {
                    let outer_f = Ref::make_mut(&mut tree.outer_f);
                    return Some(&mut outer_f[local_index]);
                }
                local_index -= tree.outer_f.len();

                if local_index < tree.inner_f.len() {
                    let inner_f = Ref::make_mut(&mut tree.inner_f);
                    return Some(&mut inner_f[local_index]);
                }
                local_index -= tree.inner_f.len();

                if local_index < tree.middle.len() {
                    let middle = Ref::make_mut(&mut tree.middle);
                    return Some(middle.index_mut(tree.middle_level, local_index));
                }
                local_index -= tree.middle.len();

                if local_index < tree.inner_b.len() {
                    let inner_b = Ref::make_mut(&mut tree.inner_b);
                    return Some(&mut inner_b[local_index]);
                }
                local_index -= tree.inner_b.len();

                let outer_b = Ref::make_mut(&mut tree.outer_b);
                Some(&mut outer_b[local_index])
            }
        }
    }

    /// Get the first element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    #[inline]
    #[must_use]
    pub fn front(&self) -> Option<&A> {
        self.get(0)
    }

    /// Get a mutable reference to the first element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    #[inline]
    #[must_use]
    pub fn front_mut(&mut self) -> Option<&mut A> {
        self.get_mut(0)
    }

    /// Get the first element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// This is an alias for the [`front`][front] method.
    ///
    /// Time: O(log n)
    ///
    /// [front]: #method.front
    #[inline]
    #[must_use]
    pub fn head(&self) -> Option<&A> {
        self.get(0)
    }

    /// Get the last element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn back(&self) -> Option<&A> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Get a mutable reference to the last element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn back_mut(&mut self) -> Option<&mut A> {
        if self.is_empty() {
            None
        } else {
            let len = self.len();
            self.get_mut(len - 1)
        }
    }

    /// Get the last element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// This is an alias for the [`back`][back] method.
    ///
    /// Time: O(log n)
    ///
    /// [back]: #method.back
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&A> {
        self.back()
    }

    /// Get the index of a given element in the vector.
    ///
    /// Searches the vector for the first occurrence of a given value,
    /// and returns the index of the value if it's there. Otherwise,
    /// it returns `None`.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3, 4, 5];
    /// assert_eq!(Some(2), vec.index_of(&3));
    /// assert_eq!(None, vec.index_of(&31337));
    /// # }
    /// ```
    #[must_use]
    pub fn index_of(&self, value: &A) -> Option<usize>
    where
        A: PartialEq,
    {
        for (index, item) in self.iter().enumerate() {
            if value == item {
                return Some(index);
            }
        }
        None
    }

    /// Test if a given element is in the vector.
    ///
    /// Searches the vector for the first occurrence of a given value,
    /// and returns `true if it's there. If it's nowhere to be found
    /// in the vector, it returns `false`.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3, 4, 5];
    /// assert_eq!(true, vec.contains(&3));
    /// assert_eq!(false, vec.contains(&31337));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn contains(&self, value: &A) -> bool
    where
        A: PartialEq,
    {
        self.index_of(value).is_some()
    }

    /// Discard all elements from the vector.
    ///
    /// This leaves you with an empty vector, and all elements that
    /// were previously inside it are dropped.
    ///
    /// Time: O(n)
    pub fn clear(&mut self) {
        if !self.is_empty() {
            *self = Single(Ref::new(Chunk::new()));
        }
    }

    /// Binary search a sorted vector for a given element using a comparator
    /// function.
    ///
    /// Assumes the vector has already been sorted using the same comparator
    /// function, eg. by using [`sort_by`][sort_by].
    ///
    /// If the value is found, it returns `Ok(index)` where `index` is the index
    /// of the element. If the value isn't found, it returns `Err(index)` where
    /// `index` is the index at which the element would need to be inserted to
    /// maintain sorted order.
    ///
    /// Time: O(log n)
    ///
    /// [sort_by]: #method.sort_by
    #[must_use]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&A) -> Ordering,
    {
        let mut size = self.len();
        if size == 0 {
            return Err(0);
        }
        let mut base = 0;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            base = match f(&self[mid]) {
                Ordering::Greater => base,
                _ => mid,
            };
            size -= half;
        }
        match f(&self[base]) {
            Ordering::Equal => Ok(base),
            Ordering::Greater => Err(base),
            Ordering::Less => Err(base + 1),
        }
    }

    /// Binary search a sorted vector for a given element.
    ///
    /// If the value is found, it returns `Ok(index)` where `index` is the index
    /// of the element. If the value isn't found, it returns `Err(index)` where
    /// `index` is the index at which the element would need to be inserted to
    /// maintain sorted order.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn binary_search(&self, value: &A) -> Result<usize, usize>
    where
        A: Ord,
    {
        self.binary_search_by(|e| e.cmp(value))
    }

    /// Binary search a sorted vector for a given element with a key extract
    /// function.
    ///
    /// Assumes the vector has already been sorted using the same key extract
    /// function, eg. by using [`sort_by_key`][sort_by_key].
    ///
    /// If the value is found, it returns `Ok(index)` where `index` is the index
    /// of the element. If the value isn't found, it returns `Err(index)` where
    /// `index` is the index at which the element would need to be inserted to
    /// maintain sorted order.
    ///
    /// Time: O(log n)
    ///
    /// [sort_by_key]: #method.sort_by_key
    #[must_use]
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&A) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

impl<A: Clone> Vector<A> {
    /// Construct a vector with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let vec = Vector::unit(1337);
    /// assert_eq!(1, vec.len());
    /// assert_eq!(
    ///   vec.get(0),
    ///   Some(&1337)
    /// );
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn unit(a: A) -> Self {
        if InlineArray::<A, RRB<A>>::CAPACITY > 0 {
            let mut array = InlineArray::new();
            array.push(a);
            Inline(array)
        } else {
            Single(Ref::new(Chunk::unit(a)))
        }
    }

    /// Create a new vector with the value at index `index` updated.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3];
    /// assert_eq!(vector![1, 5, 3], vec.update(1, 5));
    /// # }
    /// ```
    #[must_use]
    pub fn update(&self, index: usize, value: A) -> Self {
        let mut out = self.clone();
        out[index] = value;
        out
    }

    /// Update the value at index `index` in a vector.
    ///
    /// Returns the previous value at the index.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn set(&mut self, index: usize, value: A) -> A {
        replace(&mut self[index], value)
    }

    /// Swap the elements at indices `i` and `j`.
    ///
    /// Time: O(log n)
    pub fn swap(&mut self, i: usize, j: usize) {
        swap_indices(self, i, j)
    }

    /// Push a value to the front of a vector.
    ///
    /// Time: O(1)*
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![5, 6, 7];
    /// vec.push_front(4);
    /// assert_eq!(vector![4, 5, 6, 7], vec);
    /// # }
    /// ```
    pub fn push_front(&mut self, value: A) {
        if self.needs_promotion() {
            self.promote_back();
        }
        match self {
            Inline(chunk) => {
                chunk.insert(0, value);
            }
            Single(chunk) => Ref::make_mut(chunk).push_front(value),
            Full(tree) => tree.push_front(value),
        }
    }

    /// Push a value to the back of a vector.
    ///
    /// Time: O(1)*
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3];
    /// vec.push_back(4);
    /// assert_eq!(vector![1, 2, 3, 4], vec);
    /// # }
    /// ```
    pub fn push_back(&mut self, value: A) {
        if self.needs_promotion() {
            self.promote_front();
        }
        match self {
            Inline(chunk) => {
                chunk.push(value);
            }
            Single(chunk) => Ref::make_mut(chunk).push_back(value),
            Full(tree) => tree.push_back(value),
        }
    }

    /// Remove the first element from a vector and return it.
    ///
    /// Time: O(1)*
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3];
    /// assert_eq!(Some(1), vec.pop_front());
    /// assert_eq!(vector![2, 3], vec);
    /// # }
    /// ```
    pub fn pop_front(&mut self) -> Option<A> {
        if self.is_empty() {
            None
        } else {
            match self {
                Inline(chunk) => chunk.remove(0),
                Single(chunk) => Some(Ref::make_mut(chunk).pop_front()),
                Full(tree) => tree.pop_front(),
            }
        }
    }

    /// Remove the last element from a vector and return it.
    ///
    /// Time: O(1)*
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3];
    /// assert_eq!(Some(3), vec.pop_back());
    /// assert_eq!(vector![1, 2], vec);
    /// # }
    /// ```
    pub fn pop_back(&mut self) -> Option<A> {
        if self.is_empty() {
            None
        } else {
            match self {
                Inline(chunk) => chunk.pop(),
                Single(chunk) => Some(Ref::make_mut(chunk).pop_back()),
                Full(tree) => tree.pop_back(),
            }
        }
    }

    /// Append the vector `other` to the end of the current vector.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3];
    /// vec.append(vector![7, 8, 9]);
    /// assert_eq!(vector![1, 2, 3, 7, 8, 9], vec);
    /// # }
    /// ```
    pub fn append(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            replace(self, other);
            return;
        }

        self.promote_inline();
        other.promote_inline();

        let total_length = self
            .len()
            .checked_add(other.len())
            .expect("Vector length overflow");

        match self {
            Inline(_) => unreachable!("inline vecs should have been promoted"),
            Single(left) => {
                match other {
                    Inline(_) => unreachable!("inline vecs should have been promoted"),
                    // If both are single chunks and left has room for right: directly
                    // memcpy right into left
                    Single(ref mut right) if total_length <= CHUNK_SIZE => {
                        Ref::make_mut(left).append(Ref::make_mut(right));
                        return;
                    }
                    // If only left is a single chunk and has room for right: push
                    // right's elements into left
                    ref mut right if total_length <= CHUNK_SIZE => {
                        while let Some(value) = right.pop_front() {
                            Ref::make_mut(left).push_back(value);
                        }
                        return;
                    }
                    _ => {}
                }
            }
            Full(left) => {
                if let Full(mut right) = other {
                    // If left and right are trees with empty middles, left has no back
                    // buffers, and right has no front buffers: copy right's back
                    // buffers over to left
                    if left.middle.is_empty()
                        && right.middle.is_empty()
                        && left.outer_b.is_empty()
                        && left.inner_b.is_empty()
                        && right.outer_f.is_empty()
                        && right.inner_f.is_empty()
                    {
                        left.inner_b = right.inner_b;
                        left.outer_b = right.outer_b;
                        left.length = total_length;
                        return;
                    }
                    // If left and right are trees with empty middles and left's buffers
                    // can fit right's buffers: push right's elements onto left
                    if left.middle.is_empty()
                        && right.middle.is_empty()
                        && total_length <= CHUNK_SIZE * 4
                    {
                        while let Some(value) = right.pop_front() {
                            left.push_back(value);
                        }
                        return;
                    }
                    // Both are full and big: do the full RRB join
                    let inner_b1 = left.inner_b.clone();
                    left.push_middle(Side::Right, inner_b1);
                    let outer_b1 = left.outer_b.clone();
                    left.push_middle(Side::Right, outer_b1);
                    let inner_f2 = right.inner_f.clone();
                    right.push_middle(Side::Left, inner_f2);
                    let outer_f2 = right.outer_f.clone();
                    right.push_middle(Side::Left, outer_f2);

                    let mut middle1 = clone_ref(replace(&mut left.middle, Ref::from(Node::new())));
                    let mut middle2 = clone_ref(right.middle);
                    let normalised_middle = if left.middle_level > right.middle_level {
                        middle2 = middle2.elevate(left.middle_level - right.middle_level);
                        left.middle_level
                    } else if left.middle_level < right.middle_level {
                        middle1 = middle1.elevate(right.middle_level - left.middle_level);
                        right.middle_level
                    } else {
                        left.middle_level
                    };
                    left.middle = Ref::new(Node::merge(
                        Ref::from(middle1),
                        Ref::from(middle2),
                        normalised_middle,
                    ));
                    left.middle_level = normalised_middle + 1;

                    left.inner_b = right.inner_b;
                    left.outer_b = right.outer_b;
                    left.length = total_length;
                    left.prune();
                    return;
                }
            }
        }
        // No optimisations available, and either left, right or both are
        // single: promote both to full and retry
        self.promote_front();
        other.promote_back();
        self.append(other)
    }

    /// Retain only the elements specified by the predicate.
    ///
    /// Remove all elements for which the provided function `f`
    /// returns false from the vector.
    ///
    /// Time: O(n)
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&A) -> bool,
    {
        let len = self.len();
        let mut del = 0;
        {
            let mut focus = self.focus_mut();
            for i in 0..len {
                if !f(focus.index(i)) {
                    del += 1;
                } else if del > 0 {
                    focus.swap(i - del, i);
                }
            }
        }
        if del > 0 {
            self.split_off(len - del);
        }
    }

    /// Split a vector at a given index.
    ///
    /// Split a vector at a given index, consuming the vector and
    /// returning a pair of the left hand side and the right hand side
    /// of the split.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3, 7, 8, 9];
    /// let (left, right) = vec.split_at(3);
    /// assert_eq!(vector![1, 2, 3], left);
    /// assert_eq!(vector![7, 8, 9], right);
    /// # }
    /// ```
    pub fn split_at(mut self, index: usize) -> (Self, Self) {
        let right = self.split_off(index);
        (self, right)
    }

    /// Split a vector at a given index.
    ///
    /// Split a vector at a given index, leaving the left hand side in
    /// the current vector and returning a new vector containing the
    /// right hand side.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut left = vector![1, 2, 3, 7, 8, 9];
    /// let right = left.split_off(3);
    /// assert_eq!(vector![1, 2, 3], left);
    /// assert_eq!(vector![7, 8, 9], right);
    /// # }
    /// ```
    pub fn split_off(&mut self, index: usize) -> Self {
        assert!(index <= self.len());

        match self {
            Inline(chunk) => Inline(chunk.split_off(index)),
            Single(chunk) => Single(Ref::new(Ref::make_mut(chunk).split_off(index))),
            Full(tree) => {
                let mut local_index = index;

                if local_index < tree.outer_f.len() {
                    let of2 = Ref::make_mut(&mut tree.outer_f).split_off(local_index);
                    let right = RRB {
                        length: tree.length - index,
                        middle_level: tree.middle_level,
                        outer_f: Ref::new(of2),
                        inner_f: replace_def(&mut tree.inner_f),
                        middle: replace_def(&mut tree.middle),
                        inner_b: replace_def(&mut tree.inner_b),
                        outer_b: replace_def(&mut tree.outer_b),
                    };
                    tree.length = index;
                    tree.middle_level = 0;
                    return Full(right);
                }

                local_index -= tree.outer_f.len();

                if local_index < tree.inner_f.len() {
                    let if2 = Ref::make_mut(&mut tree.inner_f).split_off(local_index);
                    let right = RRB {
                        length: tree.length - index,
                        middle_level: tree.middle_level,
                        outer_f: Ref::new(if2),
                        inner_f: Ref::<Chunk<A>>::default(),
                        middle: replace_def(&mut tree.middle),
                        inner_b: replace_def(&mut tree.inner_b),
                        outer_b: replace_def(&mut tree.outer_b),
                    };
                    tree.length = index;
                    tree.middle_level = 0;
                    swap(&mut tree.outer_b, &mut tree.inner_f);
                    return Full(right);
                }

                local_index -= tree.inner_f.len();

                if local_index < tree.middle.len() {
                    let mut right_middle = tree.middle.clone();
                    let (c1, c2) = {
                        let m1 = Ref::make_mut(&mut tree.middle);
                        let m2 = Ref::make_mut(&mut right_middle);
                        match m1.split(tree.middle_level, Side::Right, local_index) {
                            SplitResult::Dropped(_) => (),
                            SplitResult::OutOfBounds => unreachable!(),
                        };
                        match m2.split(tree.middle_level, Side::Left, local_index) {
                            SplitResult::Dropped(_) => (),
                            SplitResult::OutOfBounds => unreachable!(),
                        };
                        let c1 = match m1.pop_chunk(tree.middle_level, Side::Right) {
                            PopResult::Empty => Ref::<Chunk<A>>::default(),
                            PopResult::Done(chunk) => chunk,
                            PopResult::Drained(chunk) => {
                                m1.clear_node();
                                chunk
                            }
                        };
                        let c2 = match m2.pop_chunk(tree.middle_level, Side::Left) {
                            PopResult::Empty => Ref::<Chunk<A>>::default(),
                            PopResult::Done(chunk) => chunk,
                            PopResult::Drained(chunk) => {
                                m2.clear_node();
                                chunk
                            }
                        };
                        (c1, c2)
                    };
                    let mut right = RRB {
                        length: tree.length - index,
                        middle_level: tree.middle_level,
                        outer_f: c2,
                        inner_f: Ref::<Chunk<A>>::default(),
                        middle: right_middle,
                        inner_b: replace_def(&mut tree.inner_b),
                        outer_b: replace(&mut tree.outer_b, c1),
                    };
                    tree.length = index;
                    tree.prune();
                    right.prune();
                    return Full(right);
                }

                local_index -= tree.middle.len();

                if local_index < tree.inner_b.len() {
                    let ib2 = Ref::make_mut(&mut tree.inner_b).split_off(local_index);
                    let right = RRB {
                        length: tree.length - index,
                        outer_b: replace_def(&mut tree.outer_b),
                        outer_f: Ref::new(ib2),
                        ..RRB::new()
                    };
                    tree.length = index;
                    swap(&mut tree.outer_b, &mut tree.inner_b);
                    return Full(right);
                }

                local_index -= tree.inner_b.len();

                let ob2 = Ref::make_mut(&mut tree.outer_b).split_off(local_index);
                tree.length = index;
                Single(Ref::new(ob2))
            }
        }
    }

    /// Construct a vector with `count` elements removed from the
    /// start of the current vector.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn skip(&self, count: usize) -> Self {
        // FIXME can be made more efficient by dropping the unwanted side without constructing it
        self.clone().split_off(count)
    }

    /// Construct a vector of the first `count` elements from the
    /// current vector.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn take(&self, count: usize) -> Self {
        // FIXME can be made more efficient by dropping the unwanted side without constructing it
        let mut left = self.clone();
        left.split_off(count);
        left
    }

    /// Truncate a vector to the given size.
    ///
    /// Discards all elements in the vector beyond the given length.
    ///
    /// Panics if the new length is greater than the current length.
    ///
    /// Time: O(log n)
    pub fn truncate(&mut self, len: usize) {
        // FIXME can be made more efficient by dropping the unwanted side without constructing it
        self.split_off(len);
    }

    /// Extract a slice from a vector.
    ///
    /// Remove the elements from `start_index` until `end_index` in
    /// the current vector and return the removed slice as a new
    /// vector.
    ///
    /// Time: O(log n)
    pub fn slice<R>(&mut self, range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let r = to_range(&range, self.len());
        if r.start >= r.end || r.start >= self.len() {
            return Vector::new();
        }
        let mut middle = self.split_off(r.start);
        let right = middle.split_off(r.end - r.start);
        self.append(right);
        middle
    }

    /// Insert an element into a vector.
    ///
    /// Insert an element at position `index`, shifting all elements
    /// after it to the right.
    ///
    /// ## Performance Note
    ///
    /// While `push_front` and `push_back` are heavily optimised
    /// operations, `insert` in the middle of a vector requires a
    /// split, a push, and an append. Thus, if you want to insert
    /// many elements at the same location, instead of `insert`ing
    /// them one by one, you should rather create a new vector
    /// containing the elements to insert, split the vector at the
    /// insertion point, and append the left hand, the new vector and
    /// the right hand in order.
    ///
    /// Time: O(log n)
    pub fn insert(&mut self, index: usize, value: A) {
        if index == 0 {
            return self.push_front(value);
        }
        if index == self.len() {
            return self.push_back(value);
        }
        assert!(index < self.len());
        if if let Inline(chunk) = self {
            chunk.is_full()
        } else {
            false
        } {
            self.promote_inline();
        }
        match self {
            Inline(chunk) => {
                chunk.insert(index, value);
            }
            Single(chunk) if chunk.len() < CHUNK_SIZE => Ref::make_mut(chunk).insert(index, value),
            // TODO a lot of optimisations still possible here
            _ => {
                let right = self.split_off(index);
                self.push_back(value);
                self.append(right);
            }
        }
    }

    /// Remove an element from a vector.
    ///
    /// Remove the element from position 'index', shifting all
    /// elements after it to the left, and return the removec element.
    ///
    /// ## Performance Note
    ///
    /// While `pop_front` and `pop_back` are heavily optimised
    /// operations, `remove` in the middle of a vector requires a
    /// split, a pop, and an append. Thus, if you want to remove many
    /// elements from the same location, instead of `remove`ing them
    /// one by one, it is much better to use [`slice`][slice].
    ///
    /// Time: O(log n)
    ///
    /// [slice]: #method.slice
    pub fn remove(&mut self, index: usize) -> A {
        assert!(index < self.len());
        match self {
            Inline(chunk) => chunk.remove(index).unwrap(),
            Single(chunk) => Ref::make_mut(chunk).remove(index),
            _ => {
                if index == 0 {
                    return self.pop_front().unwrap();
                }
                if index == self.len() - 1 {
                    return self.pop_back().unwrap();
                }
                // TODO a lot of optimisations still possible here
                let mut right = self.split_off(index);
                let value = right.pop_front().unwrap();
                self.append(right);
                value
            }
        }
    }

    /// Insert an element into a sorted vector.
    ///
    /// Insert an element into a vector in sorted order, assuming the vector is
    /// already in sorted order.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3, 7, 8, 9];
    /// vec.insert_ord(5);
    /// assert_eq!(vector![1, 2, 3, 5, 7, 8, 9], vec);
    /// # }
    /// ```
    pub fn insert_ord(&mut self, item: A)
    where
        A: Ord,
    {
        match self.binary_search(&item) {
            Ok(index) => self.insert(index, item),
            Err(index) => self.insert(index, item),
        }
    }

    /// Sort a vector.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut vec = vector![3, 2, 5, 4, 1];
    /// vec.sort();
    /// assert_eq!(vector![1, 2, 3, 4, 5], vec);
    /// # }
    /// ```
    pub fn sort(&mut self)
    where
        A: Ord,
    {
        self.sort_by(Ord::cmp)
    }

    /// Sort a vector using a comparator function.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut vec = vector![3, 2, 5, 4, 1];
    /// vec.sort_by(|left, right| left.cmp(right));
    /// assert_eq!(vector![1, 2, 3, 4, 5], vec);
    /// # }
    /// ```
    pub fn sort_by<F>(&mut self, cmp: F)
    where
        F: Fn(&A, &A) -> Ordering,
    {
        let len = self.len();
        if len > 1 {
            sort::quicksort(&mut self.focus_mut(), 0, len - 1, &cmp);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn assert_invariants(&self) {
        if let Vector::Full(ref tree) = self {
            tree.middle.assert_invariants();
        }
    }
}

// Implementation details

impl<A: Clone> RRB<A> {
    fn into_iter(
        self,
    ) -> Chain<
        Chain<Chain<Chain<ChunkIter<A>, ChunkIter<A>>, ConsumingNodeIter<A>>, ChunkIter<A>>,
        ChunkIter<A>,
    > {
        let outer_f = clone_ref(self.outer_f).into_iter();
        let inner_f = clone_ref(self.inner_f).into_iter();
        let middle = ConsumingNodeIter::new(clone_ref(self.middle), self.middle_level);
        let inner_b = clone_ref(self.inner_b).into_iter();
        let outer_b = clone_ref(self.outer_b).into_iter();
        outer_f
            .chain(inner_f)
            .chain(middle)
            .chain(inner_b)
            .chain(outer_b)
    }

    fn new() -> Self {
        RRB {
            length: 0,
            middle_level: 0,
            outer_f: Ref::new(Chunk::new()),
            inner_f: Ref::new(Chunk::new()),
            middle: Ref::new(Node::new()),
            inner_b: Ref::new(Chunk::new()),
            outer_b: Ref::new(Chunk::new()),
        }
    }

    fn prune(&mut self) {
        if self.middle.is_empty() {
            self.middle = Ref::new(Node::new());
            self.middle_level = 0;
        } else {
            while self.middle_level > 0 && self.middle.is_single() {
                self.middle = self.middle.first_child().clone();
                self.middle_level -= 1;
            }
        }
    }

    fn pop_front(&mut self) -> Option<A> {
        if self.length == 0 {
            return None;
        }
        if self.outer_f.is_empty() {
            if self.inner_f.is_empty() {
                if self.middle.is_empty() {
                    if self.inner_b.is_empty() {
                        swap(&mut self.outer_f, &mut self.outer_b);
                    } else {
                        swap(&mut self.outer_f, &mut self.inner_b);
                    }
                } else {
                    self.outer_f = self.pop_middle(Side::Left).unwrap();
                }
            } else {
                swap(&mut self.outer_f, &mut self.inner_f);
            }
        }
        self.length -= 1;
        let outer_f = Ref::make_mut(&mut self.outer_f);
        Some(outer_f.pop_front())
    }

    fn pop_back(&mut self) -> Option<A> {
        if self.length == 0 {
            return None;
        }
        if self.outer_b.is_empty() {
            if self.inner_b.is_empty() {
                if self.middle.is_empty() {
                    if self.inner_f.is_empty() {
                        swap(&mut self.outer_b, &mut self.outer_f);
                    } else {
                        swap(&mut self.outer_b, &mut self.inner_f);
                    }
                } else {
                    self.outer_b = self.pop_middle(Side::Right).unwrap();
                }
            } else {
                swap(&mut self.outer_b, &mut self.inner_b);
            }
        }
        self.length -= 1;
        let outer_b = Ref::make_mut(&mut self.outer_b);
        Some(outer_b.pop_back())
    }

    fn push_front(&mut self, value: A) {
        if self.outer_f.is_full() {
            swap(&mut self.outer_f, &mut self.inner_f);
            if !self.outer_f.is_empty() {
                let mut chunk = Ref::new(Chunk::new());
                swap(&mut chunk, &mut self.outer_f);
                self.push_middle(Side::Left, chunk);
            }
        }
        self.length = self.length.checked_add(1).expect("Vector length overflow");
        let outer_f = Ref::make_mut(&mut self.outer_f);
        outer_f.push_front(value)
    }

    fn push_back(&mut self, value: A) {
        if self.outer_b.is_full() {
            swap(&mut self.outer_b, &mut self.inner_b);
            if !self.outer_b.is_empty() {
                let mut chunk = Ref::new(Chunk::new());
                swap(&mut chunk, &mut self.outer_b);
                self.push_middle(Side::Right, chunk);
            }
        }
        self.length = self.length.checked_add(1).expect("Vector length overflow");
        let outer_b = Ref::make_mut(&mut self.outer_b);
        outer_b.push_back(value)
    }

    fn push_middle(&mut self, side: Side, chunk: Ref<Chunk<A>>) {
        if chunk.is_empty() {
            return;
        }
        let new_middle = {
            let middle = Ref::make_mut(&mut self.middle);
            match middle.push_chunk(self.middle_level, side, chunk) {
                PushResult::Done => return,
                PushResult::Full(chunk, _num_drained) => Ref::from({
                    match side {
                        Side::Left => Node::from_chunk(self.middle_level, chunk)
                            .join_branches(middle.clone(), self.middle_level),
                        Side::Right => middle.clone().join_branches(
                            Node::from_chunk(self.middle_level, chunk),
                            self.middle_level,
                        ),
                    }
                }),
            }
        };
        self.middle_level += 1;
        self.middle = new_middle;
    }

    fn pop_middle(&mut self, side: Side) -> Option<Ref<Chunk<A>>> {
        let chunk = {
            let middle = Ref::make_mut(&mut self.middle);
            match middle.pop_chunk(self.middle_level, side) {
                PopResult::Empty => return None,
                PopResult::Done(chunk) => chunk,
                PopResult::Drained(chunk) => {
                    middle.clear_node();
                    self.middle_level = 0;
                    chunk
                }
            }
        };
        Some(chunk)
    }
}

#[inline]
fn replace_def<A: Default>(dest: &mut A) -> A {
    replace(dest, Default::default())
}

// Core traits

impl<A: Clone> Default for Vector<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Clone> Clone for Vector<A> {
    fn clone(&self) -> Self {
        match self {
            Inline(chunk) => Inline(chunk.clone()),
            Single(chunk) => Single(chunk.clone()),
            Full(tree) => Full(tree.clone()),
        }
    }
}

impl<A: Clone + Debug> Debug for Vector<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_list().entries(self.iter()).finish()
        // match self {
        //     Full(rrb) => {
        //         writeln!(f, "Head: {:?} {:?}", rrb.outer_f, rrb.inner_f)?;
        //         rrb.middle.print(f, 0, rrb.middle_level)?;
        //         writeln!(f, "Tail: {:?} {:?}", rrb.inner_b, rrb.outer_b)
        //     }
        //     Single(_) => write!(f, "nowt"),
        // }
    }
}

#[cfg(not(has_specialisation))]
impl<A: Clone + PartialEq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Clone + PartialEq> PartialEq for Vector<A> {
    default fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Clone + Eq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Full(left), Full(right)) => {
                if left.length != right.length {
                    return false;
                }

                fn cmp_chunk<A>(left: &Ref<Chunk<A>>, right: &Ref<Chunk<A>>) -> bool {
                    (left.is_empty() && right.is_empty()) || Ref::ptr_eq(left, right)
                }

                if cmp_chunk(&left.outer_f, &right.outer_f)
                    && cmp_chunk(&left.inner_f, &right.inner_f)
                    && cmp_chunk(&left.inner_b, &right.inner_b)
                    && cmp_chunk(&left.outer_b, &right.outer_b)
                    && (left.middle.is_empty() && right.middle.is_empty())
                    || Ref::ptr_eq(&left.middle, &right.middle)
                {
                    return true;
                }
                self.iter().eq(other.iter())
            }
            (left, right) => left.len() == right.len() && left.iter().eq(right.iter()),
        }
    }
}

impl<A: Clone + Eq> Eq for Vector<A> {}

impl<A: Clone + PartialOrd> PartialOrd for Vector<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Clone + Ord> Ord for Vector<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Clone + Hash> Hash for Vector<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self {
            i.hash(state)
        }
    }
}

impl<A: Clone> Sum for Vector<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A: Clone> Add for Vector<A> {
    type Output = Vector<A>;

    /// Concatenate two vectors.
    ///
    /// Time: O(log n)
    fn add(mut self, other: Self) -> Self::Output {
        self.append(other);
        self
    }
}

impl<'a, A: Clone> Add for &'a Vector<A> {
    type Output = Vector<A>;

    /// Concatenate two vectors.
    ///
    /// Time: O(log n)
    fn add(self, other: Self) -> Self::Output {
        let mut out = self.clone();
        out.append(other.clone());
        out
    }
}

impl<A: Clone> Extend<A> for Vector<A> {
    /// Add values to the end of a vector by consuming an iterator.
    ///
    /// Time: O(n)
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = A>,
    {
        for item in iter {
            self.push_back(item)
        }
    }
}

impl<A: Clone> Index<usize> for Vector<A> {
    type Output = A;
    /// Get a reference to the value at index `index` in the vector.
    ///
    /// Time: O(log n)
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(value) => value,
            None => panic!(
                "Vector::index: index out of bounds: {} < {}",
                index,
                self.len()
            ),
        }
    }
}

impl<A: Clone> IndexMut<usize> for Vector<A> {
    /// Get a mutable reference to the value at index `index` in the
    /// vector.
    ///
    /// Time: O(log n)
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.get_mut(index) {
            Some(value) => value,
            None => panic!("Vector::index_mut: index out of bounds"),
        }
    }
}

// Conversions

impl<'a, A: Clone> IntoIterator for &'a Vector<A> {
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A: Clone> IntoIterator for Vector<A> {
    type Item = A;
    type IntoIter = ConsumingIter<A>;
    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter::new(self)
    }
}

impl<A: Clone> FromIterator<A> for Vector<A> {
    /// Create a vector from an iterator.
    ///
    /// Time: O(n)
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        let mut seq = Self::new();
        for item in iter {
            seq.push_back(item)
        }
        seq
    }
}

impl<'s, 'a, A, OA> From<&'s Vector<&'a A>> for Vector<OA>
where
    A: ToOwned<Owned = OA>,
    OA: Borrow<A> + Clone,
{
    fn from(vec: &Vector<&A>) -> Self {
        vec.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<'a, A: Clone> From<&'a [A]> for Vector<A> {
    fn from(slice: &[A]) -> Self {
        slice.iter().cloned().collect()
    }
}

impl<A: Clone> From<Vec<A>> for Vector<A> {
    /// Create a vector from a [`std::vec::Vec`][vec].
    ///
    /// Time: O(n)
    ///
    /// [vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Clone> From<&'a Vec<A>> for Vector<A> {
    /// Create a vector from a [`std::vec::Vec`][vec].
    ///
    /// Time: O(n)
    ///
    /// [vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    fn from(vec: &Vec<A>) -> Self {
        vec.iter().cloned().collect()
    }
}

// Iterators

/// An iterator over vectors with values of type `A`.
///
/// To obtain one, use [`Vector::iter()`][iter].
///
/// [iter]: enum.Vector.html#method.iter
pub struct Iter<'a, A: 'a> {
    focus: Focus<'a, A>,
    front_index: usize,
    back_index: usize,
}

impl<'a, A: Clone> Iter<'a, A> {
    fn new(seq: &'a Vector<A>) -> Self {
        Iter {
            focus: seq.focus(),
            front_index: 0,
            back_index: seq.len(),
        }
    }

    fn from_focus(focus: Focus<'a, A>) -> Self {
        Iter {
            front_index: 0,
            back_index: focus.len(),
            focus,
        }
    }
}

impl<'a, A: Clone> Iterator for Iter<'a, A> {
    type Item = &'a A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        #[allow(unsafe_code)]
        let focus: &'a mut Focus<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        let value = focus.get(self.front_index);
        self.front_index += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back_index - self.front_index;
        (remaining, Some(remaining))
    }
}

impl<'a, A: Clone> DoubleEndedIterator for Iter<'a, A> {
    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        self.back_index -= 1;
        #[allow(unsafe_code)]
        let focus: &'a mut Focus<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        focus.get(self.back_index)
    }
}

impl<'a, A: Clone> ExactSizeIterator for Iter<'a, A> {}

impl<'a, A: Clone> FusedIterator for Iter<'a, A> {}

/// A mutable iterator over vectors with values of type `A`.
///
/// To obtain one, use [`Vector::iter_mut()`][iter_mut].
///
/// [iter_mut]: enum.Vector.html#method.iter_mut
pub struct IterMut<'a, A>
where
    A: 'a,
{
    focus: FocusMut<'a, A>,
    front_index: usize,
    back_index: usize,
}

impl<'a, A> IterMut<'a, A>
where
    A: 'a + Clone,
{
    fn new(seq: &'a mut Vector<A>) -> Self {
        let focus = seq.focus_mut();
        let len = focus.len();
        IterMut {
            focus,
            front_index: 0,
            back_index: len,
        }
    }

    fn from_focus(focus: FocusMut<'a, A>) -> Self {
        IterMut {
            front_index: 0,
            back_index: focus.len(),
            focus,
        }
    }
}

impl<'a, A> Iterator for IterMut<'a, A>
where
    A: 'a + Clone,
{
    type Item = &'a mut A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        #[allow(unsafe_code)]
        let focus: &'a mut FocusMut<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        let value = focus.get_mut(self.front_index);
        self.front_index += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back_index - self.front_index;
        (remaining, Some(remaining))
    }
}

impl<'a, A> DoubleEndedIterator for IterMut<'a, A>
where
    A: 'a + Clone,
{
    /// Remove and return an element from the back of the iterator.
    ///
    /// Time: O(1)*
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        self.back_index -= 1;
        #[allow(unsafe_code)]
        let focus: &'a mut FocusMut<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        focus.get_mut(self.back_index)
    }
}

impl<'a, A: Clone> ExactSizeIterator for IterMut<'a, A> {}

impl<'a, A: Clone> FusedIterator for IterMut<'a, A> {}

/// A consuming iterator over vectors with values of type `A`.
pub enum ConsumingIter<A> {
    Inline(InlineIter<A, RRB<A>>),
    Single(ChunkIter<A>),
    Full(
        Chain<
            Chain<Chain<Chain<ChunkIter<A>, ChunkIter<A>>, ConsumingNodeIter<A>>, ChunkIter<A>>,
            ChunkIter<A>,
        >,
    ),
}

impl<A: Clone> ConsumingIter<A> {
    fn new(seq: Vector<A>) -> Self {
        match seq {
            Inline(chunk) => ConsumingIter::Inline(chunk.into_iter()),
            Single(chunk) => ConsumingIter::Single(clone_ref(chunk).into_iter()),
            Full(tree) => ConsumingIter::Full(tree.into_iter()),
        }
    }
}

impl<A: Clone> Iterator for ConsumingIter<A> {
    type Item = A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ConsumingIter::Inline(iter) => iter.next(),
            ConsumingIter::Single(iter) => iter.next(),
            ConsumingIter::Full(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            ConsumingIter::Inline(iter) => iter.size_hint(),
            ConsumingIter::Single(iter) => iter.size_hint(),
            ConsumingIter::Full(iter) => iter.size_hint(),
        }
    }
}

impl<A: Clone> DoubleEndedIterator for ConsumingIter<A> {
    /// Remove and return an element from the back of the iterator.
    ///
    /// Time: O(1)*
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            ConsumingIter::Inline(iter) => iter.next_back(),
            ConsumingIter::Single(iter) => iter.next_back(),
            ConsumingIter::Full(iter) => iter.next_back(),
        }
    }
}

impl<A: Clone> ExactSizeIterator for ConsumingIter<A> {}

impl<A: Clone> FusedIterator for ConsumingIter<A> {}

/// An iterator over the leaf nodes of a vector.
///
/// To obtain one, use [`Vector::chunks()`][chunks].
///
/// [chunks]: enum.Vector.html#method.chunks
pub struct Chunks<'a, A: 'a> {
    focus: Focus<'a, A>,
    front_index: usize,
    back_index: usize,
}

impl<'a, A: Clone> Chunks<'a, A> {
    fn new(seq: &'a Vector<A>) -> Self {
        Chunks {
            focus: seq.focus(),
            front_index: 0,
            back_index: seq.len(),
        }
    }
}

impl<'a, A: Clone> Iterator for Chunks<'a, A> {
    type Item = &'a [A];

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        #[allow(unsafe_code)]
        let focus: &'a mut Focus<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        let (range, value) = focus.chunk_at(self.front_index);
        self.front_index = range.end;
        Some(value)
    }
}

impl<'a, A: Clone> DoubleEndedIterator for Chunks<'a, A> {
    /// Remove and return an element from the back of the iterator.
    ///
    /// Time: O(1)*
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        self.back_index -= 1;
        #[allow(unsafe_code)]
        let focus: &'a mut Focus<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        let (range, value) = focus.chunk_at(self.back_index);
        self.back_index = range.start;
        Some(value)
    }
}

impl<'a, A: Clone> FusedIterator for Chunks<'a, A> {}

/// A mutable iterator over the leaf nodes of a vector.
///
/// To obtain one, use [`Vector::chunks_mut()`][chunks_mut].
///
/// [chunks_mut]: enum.Vector.html#method.chunks_mut
pub struct ChunksMut<'a, A: 'a> {
    focus: FocusMut<'a, A>,
    front_index: usize,
    back_index: usize,
}

impl<'a, A: Clone> ChunksMut<'a, A> {
    fn new(seq: &'a mut Vector<A>) -> Self {
        let len = seq.len();
        ChunksMut {
            focus: seq.focus_mut(),
            front_index: 0,
            back_index: len,
        }
    }
}

impl<'a, A: Clone> Iterator for ChunksMut<'a, A> {
    type Item = &'a mut [A];

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        #[allow(unsafe_code)]
        let focus: &'a mut FocusMut<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        let (range, value) = focus.chunk_at(self.front_index);
        self.front_index = range.end;
        Some(value)
    }
}

impl<'a, A: Clone> DoubleEndedIterator for ChunksMut<'a, A> {
    /// Remove and return an element from the back of the iterator.
    ///
    /// Time: O(1)*
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        self.back_index -= 1;
        #[allow(unsafe_code)]
        let focus: &'a mut FocusMut<'a, A> = unsafe { &mut *(&mut self.focus as *mut _) };
        let (range, value) = focus.chunk_at(self.back_index);
        self.back_index = range.start;
        Some(value)
    }
}

impl<'a, A: Clone> FusedIterator for ChunksMut<'a, A> {}

// Rayon

#[cfg(all(threadsafe, any(test, feature = "rayon")))]
pub mod rayon {
    use super::*;

    use ::rayon::iter::plumbing::{
        bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer,
    };
    use ::rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    };

    impl<'a, A> IntoParallelRefIterator<'a> for Vector<A>
    where
        A: Clone + Send + Sync + 'a,
    {
        type Item = &'a A;
        type Iter = ParIter<'a, A>;

        fn par_iter(&'a self) -> Self::Iter {
            ParIter {
                focus: self.focus(),
            }
        }
    }

    impl<'a, A> IntoParallelRefMutIterator<'a> for Vector<A>
    where
        A: Clone + Send + Sync + 'a,
    {
        type Item = &'a mut A;
        type Iter = ParIterMut<'a, A>;

        fn par_iter_mut(&'a mut self) -> Self::Iter {
            ParIterMut {
                focus: self.focus_mut(),
            }
        }
    }

    pub struct ParIter<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        focus: Focus<'a, A>,
    }

    impl<'a, A> ParallelIterator for ParIter<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        type Item = &'a A;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }
    }

    impl<'a, A> IndexedParallelIterator for ParIter<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize {
            self.focus.len()
        }

        fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: ProducerCallback<Self::Item>,
        {
            callback.callback(VectorProducer { focus: self.focus })
        }
    }

    pub struct ParIterMut<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        focus: FocusMut<'a, A>,
    }

    impl<'a, A> ParallelIterator for ParIterMut<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        type Item = &'a mut A;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }
    }

    impl<'a, A> IndexedParallelIterator for ParIterMut<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize {
            self.focus.len()
        }

        fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: ProducerCallback<Self::Item>,
        {
            callback.callback(VectorMutProducer { focus: self.focus })
        }
    }

    struct VectorProducer<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        focus: Focus<'a, A>,
    }

    impl<'a, A> Producer for VectorProducer<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        type Item = &'a A;
        type IntoIter = Iter<'a, A>;

        fn into_iter(self) -> Self::IntoIter {
            self.focus.into_iter()
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let (left, right) = self.focus.split_at(index);
            (
                VectorProducer { focus: left },
                VectorProducer { focus: right },
            )
        }
    }

    struct VectorMutProducer<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        focus: FocusMut<'a, A>,
    }

    impl<'a, A> Producer for VectorMutProducer<'a, A>
    where
        A: Clone + Send + Sync + 'a,
    {
        type Item = &'a mut A;
        type IntoIter = IterMut<'a, A>;

        fn into_iter(self) -> Self::IntoIter {
            self.focus.into_iter()
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let (left, right) = self.focus.split_at(index);
            (
                VectorMutProducer { focus: left },
                VectorMutProducer { focus: right },
            )
        }
    }

    #[cfg(test)]
    mod test {
        use super::super::*;
        use super::proptest::vector;
        use ::proptest::num::i32;
        use ::proptest::proptest;
        use ::rayon::iter::{
            IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
        };

        proptest! {
            #[test]
            fn par_iter(ref mut input in vector(i32::ANY, 0..10000)) {
                assert_eq!(input.iter().max(), input.par_iter().max())
            }

            #[test]
            fn par_mut_iter(ref mut input in vector(i32::ANY, 0..10000)) {
                let mut vec = input.clone();
                vec.par_iter_mut().for_each(|i| *i = i.overflowing_add(1).0);
                let expected: Vector<i32> = input.clone().into_iter().map(|i| i.overflowing_add(1).0).collect();
                assert_eq!(expected, vec);
            }
        }
    }
}

// QuickCheck
#[cfg(all(threadsafe, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(threadsafe, feature = "quickcheck"))]
impl<A: Arbitrary + Sync + Clone> Arbitrary for Vector<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Vector::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use ::proptest::collection::vec;
    use ::proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for generating a vector of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_vector(ref l in vector(".*", 10..100)) {
    ///         assert!(l.len() < 100);
    ///         assert!(l.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn vector<A: Strategy + 'static>(
        element: A,
        size: Range<usize>,
    ) -> BoxedStrategy<Vector<<A::Tree as ValueTree>::Value>>
    where
        <A::Tree as ValueTree>::Value: Clone,
    {
        vec(element, size).prop_map(Vector::from_iter).boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::proptest::vector;
    use super::*;
    use ::proptest::collection::vec;
    use ::proptest::num::{i32, usize};
    use ::proptest::proptest;

    #[test]
    fn macro_allows_trailing_comma() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![1, 2, 3,];
        assert_eq!(vec1, vec2);
    }

    #[test]
    fn indexing() {
        let vec1 = vector![0, 1, 2, 3, 4, 5];
        let mut vec2 = vec1.clone();
        vec2.push_front(0);
        assert_eq!(0, *vec2.get(0).unwrap());
        assert_eq!(0, vec2[0]);
    }

    #[test]
    fn large_vector_focus() {
        let input = Vector::from_iter(0..100_000);
        let vec = input.clone();
        let mut sum: i64 = 0;
        let mut focus = vec.focus();
        for i in 0..input.len() {
            sum += *focus.index(i);
        }
        let expected: i64 = (0..100_000).sum();
        assert_eq!(expected, sum);
    }

    #[test]
    fn large_vector_focus_mut() {
        let input = Vector::from_iter(0..100_000);
        let mut vec = input.clone();
        {
            let mut focus = vec.focus_mut();
            for i in 0..input.len() {
                let p = focus.index_mut(i);
                *p += 1;
            }
        }
        let expected: Vector<i32> = input.clone().into_iter().map(|i| i + 1).collect();
        assert_eq!(expected, vec);
    }

    #[test]
    fn issue_55_fwd() {
        let mut l = Vector::new();
        for i in 0..4098 {
            l.append(Vector::unit(i));
        }
        l.append(Vector::unit(4098));
        assert_eq!(Some(&4097), l.get(4097));
        assert_eq!(Some(&4096), l.get(4096));
    }

    #[test]
    fn issue_55_back() {
        let mut l = Vector::unit(0);
        for i in 0..4099 {
            let mut tmp = Vector::unit(i + 1);
            tmp.append(l);
            l = tmp;
        }
        assert_eq!(Some(&4098), l.get(1));
        assert_eq!(Some(&4097), l.get(2));
        let len = l.len();
        l.slice(2..len);
    }

    #[test]
    fn issue_55_append() {
        let mut vec1 = Vector::from_iter(0..92);
        let vec2 = Vector::from_iter(0..165);
        vec1.append(vec2);
    }

    #[test]
    fn issue_70() {
        let mut x = Vector::new();
        for _ in 0..262 {
            x.push_back(0);
        }
        for _ in 0..97 {
            x.pop_front();
        }
        for &offset in &[160, 163, 160] {
            x.remove(offset);
        }
        for _ in 0..64 {
            x.push_back(0);
        }
        // At this point middle contains three chunks of size 64, 64 and 1
        // respectively. Previously the next `push_back()` would append another
        // zero-sized chunk to middle even though there is enough space left.
        match x {
            Vector::Full(ref tree) => {
                assert_eq!(129, tree.middle.len());
                assert_eq!(3, tree.middle.number_of_children());
            }
            _ => unreachable!(),
        }
        x.push_back(0);
        match x {
            Vector::Full(ref tree) => {
                assert_eq!(131, tree.middle.len());
                assert_eq!(3, tree.middle.number_of_children())
            }
            _ => unreachable!(),
        }
        for _ in 0..64 {
            x.push_back(0);
        }
        for _ in x.iter() {}
    }

    #[test]
    fn issue_67() {
        let mut l = Vector::unit(4100);
        for i in (0..4099).rev() {
            let mut tmp = Vector::unit(i);
            tmp.append(l);
            l = tmp;
        }
        assert_eq!(4100, l.len());
        let len = l.len();
        let tail = l.slice(1..len);
        assert_eq!(1, l.len());
        assert_eq!(4099, tail.len());
        assert_eq!(Some(&0), l.get(0));
        assert_eq!(Some(&1), tail.get(0));
    }

    #[test]
    fn issue_74_simple_size() {
        use crate::nodes::rrb::NODE_SIZE;
        let mut x = Vector::new();
        for _ in 0..(CHUNK_SIZE
            * (
                1 // inner_f
                + (2 * NODE_SIZE) // middle: two full Entry::Nodes (4096 elements each)
                + 1 // inner_b
                + 1
                // outer_b
            ))
        {
            x.push_back(0u32);
        }
        let middle_first_node_start = CHUNK_SIZE;
        let middle_second_node_start = middle_first_node_start + NODE_SIZE * CHUNK_SIZE;
        // This reduces the size of the second node to 4095.
        x.remove(middle_second_node_start);
        // As outer_b is full, this will cause inner_b (length 64) to be pushed
        // to middle. The first element will be merged into the second node, the
        // remaining 63 elements will end up in a new node.
        x.push_back(0u32);
        match x {
            Vector::Full(tree) => {
                assert_eq!(3, tree.middle.number_of_children());
                assert_eq!(
                    2 * NODE_SIZE * CHUNK_SIZE + CHUNK_SIZE - 1,
                    tree.middle.len()
                );
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn issue_77() {
        let mut x = Vector::new();
        for _ in 0..44 {
            x.push_back(0);
        }
        for _ in 0..20 {
            x.insert(0, 0);
        }
        x.insert(1, 0);
        for _ in 0..441 {
            x.push_back(0);
        }
        for _ in 0..58 {
            x.insert(0, 0);
        }
        x.insert(514, 0);
        for _ in 0..73 {
            x.push_back(0);
        }
        for _ in 0..10 {
            x.insert(0, 0);
        }
        x.insert(514, 0);
    }

    proptest! {
        #[test]
        fn iter(ref vec in vec(i32::ANY, 0..1000)) {
            let seq: Vector<i32> = Vector::from_iter(vec.iter().cloned());
            for (index, item) in seq.iter().enumerate() {
                assert_eq!(&vec[index], item);
            }
            assert_eq!(vec.len(), seq.len());
        }

        #[test]
        fn push_front_mut(ref input in vec(i32::ANY, 0..1000)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector.push_front(value);
                assert_eq!(count + 1, vector.len());
            }
            let input2 = Vec::from_iter(input.iter().rev().cloned());
            assert_eq!(input2, Vec::from_iter(vector.iter().cloned()));
        }

        #[test]
        fn push_back_mut(ref input in vec(i32::ANY, 0..1000)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector.push_back(value);
                assert_eq!(count + 1, vector.len());
            }
            assert_eq!(input, &Vec::from_iter(vector.iter().cloned()));
        }

        #[test]
        fn pop_back_mut(ref input in vec(i32::ANY, 0..1000)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop_back() {
                    None => panic!("vector emptied unexpectedly"),
                    Some(item) => {
                        assert_eq!(index, vector.len());
                        assert_eq!(value, item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn pop_front_mut(ref input in vec(i32::ANY, 0..1000)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().rev().enumerate().rev() {
                match vector.pop_front() {
                    None => panic!("vector emptied unexpectedly"),
                    Some(item) => {
                        assert_eq!(index, vector.len());
                        assert_eq!(value, item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        // #[test]
        // fn push_and_pop(ref input in vec(i32::ANY, 0..1000)) {
        //     let mut vector = Vector::new();
        //     for (count, value) in input.iter().cloned().enumerate() {
        //         assert_eq!(count, vector.len());
        //         vector.push_back(value);
        //         assert_eq!(count + 1, vector.len());
        //     }
        //     for (index, value) in input.iter().cloned().rev().enumerate().rev() {
        //         match vector.pop_front() {
        //             None => panic!("vector emptied unexpectedly"),
        //             Some(item) => {
        //                 assert_eq!(index, vector.len());
        //                 assert_eq!(value, item);
        //             }
        //         }
        //     }
        //     assert_eq!(true, vector.is_empty());
        // }

        #[test]
        fn split(ref vec in vec(i32::ANY, 1..2000), split_pos in usize::ANY) {
            let split_index = split_pos % (vec.len() + 1);
            let mut left = Vector::from_iter(vec.iter().cloned());
            let right = left.split_off(split_index);
            assert_eq!(left.len(), split_index);
            assert_eq!(right.len(), vec.len() - split_index);
            for (index, item) in left.iter().enumerate() {
                assert_eq!(& vec[index], item);
            }
            for (index, item) in right.iter().enumerate() {
                assert_eq!(&vec[split_index + index], item);
            }
        }

        #[test]
        fn append(ref vec1 in vec(i32::ANY, 0..1000), ref vec2 in vec(i32::ANY, 0..1000)) {
            let mut seq1 = Vector::from_iter(vec1.iter().cloned());
            let seq2 = Vector::from_iter(vec2.iter().cloned());
            assert_eq!(seq1.len(), vec1.len());
            assert_eq!(seq2.len(), vec2.len());
            seq1.append(seq2);
            let mut vec = vec1.clone();
            vec.extend(vec2);
            assert_eq!(seq1.len(), vec.len());
            for (index, item) in seq1.into_iter().enumerate() {
                assert_eq!(vec[index], item);
            }
        }

        #[test]
        fn iter_mut(ref input in vector(i32::ANY, 0..10000)) {
            let mut vec = input.clone();
            {
                for p in vec.iter_mut() {
                    *p = p.overflowing_add(1).0;
                }
            }
            let expected: Vector<i32> = input.clone().into_iter().map(|i| i.overflowing_add(1).0).collect();
            assert_eq!(expected, vec);
        }

        #[test]
        fn focus(ref input in vector(i32::ANY, 0..10000)) {
            let mut vec = input.clone();
            {
                let mut focus = vec.focus_mut();
                for i in 0..input.len() {
                    let p = focus.index_mut(i);
                    *p = p.overflowing_add(1).0;
                }
            }
            let expected: Vector<i32> = input.clone().into_iter().map(|i| i.overflowing_add(1).0).collect();
            assert_eq!(expected, vec);
        }

        #[test]
        fn focus_mut_split(ref input in vector(i32::ANY, 0..10000)) {
            let mut vec = input.clone();

            fn split_down(focus: FocusMut<'_, i32>) {
                let len = focus.len();
                if len < 8 {
                    for p in focus {
                        *p = p.overflowing_add(1).0;
                    }
                } else {
                    let (left, right) = focus.split_at(len / 2);
                    split_down(left);
                    split_down(right);
                }
            }

            split_down(vec.focus_mut());

            let expected: Vector<i32> = input.clone().into_iter().map(|i| i.overflowing_add(1).0).collect();
            assert_eq!(expected, vec);
        }

        #[test]
        fn chunks(ref input in vector(i32::ANY, 0..10000)) {
            let output: Vector<_> = input.leaves().flat_map(|a|a).cloned().collect();
            assert_eq!(input, &output);
            let rev_in: Vector<_> = input.iter().rev().cloned().collect();
            let rev_out: Vector<_> = input.leaves().rev().map(|c| c.iter().rev()).flat_map(|a|a).cloned().collect();
            assert_eq!(rev_in, rev_out);
        }

        #[test]
        fn chunks_mut(ref mut input_src in vector(i32::ANY, 0..10000)) {
            let mut input = input_src.clone();
            #[allow(clippy::map_clone)]
            let output: Vector<_> = input.leaves_mut().flat_map(|a| a).map(|v| *v).collect();
            assert_eq!(input, output);
            let rev_in: Vector<_> = input.iter().rev().cloned().collect();
            let rev_out: Vector<_> = input.leaves_mut().rev().map(|c| c.iter().rev()).flat_map(|a|a).cloned().collect();
            assert_eq!(rev_in, rev_out);
        }

        // The following two tests are very slow and there are unit tests above
        // which test for regression of issue #55.  It would still be good to
        // run them occasionally.

        // #[test]
        // fn issue55_back(count in 0..10000, slice_at in usize::ANY) {
        //     let count = count as usize;
        //     let slice_at = slice_at % count;
        //     let mut l = Vector::unit(0);
        //     for _ in 0..count {
        //         let mut tmp = Vector::unit(0);
        //         tmp.append(l);
        //         l = tmp;
        //     }
        //     let len = l.len();
        //     l.slice(slice_at..len);
        // }

        // #[test]
        // fn issue55_fwd(count in 0..10000, slice_at in usize::ANY) {
        //     let count = count as usize;
        //     let slice_at = slice_at % count;
        //     let mut l = Vector::new();
        //     for i in 0..count {
        //         l.append(Vector::unit(i));
        //     }
        //     assert_eq!(Some(&slice_at), l.get(slice_at));
        // }
    }
}
