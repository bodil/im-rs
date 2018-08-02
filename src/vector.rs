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
//! tree, so you'll see performance and memory characteristics similar
//! to [`Vec`][Vec] or [`VecDeque`][VecDeque].
//!
//! This means that the structure always preallocates four chunks of
//! size *k* (*k* being the tree's branching factor), equivalent to a
//! [`Vec`][Vec] with an initial capacity of 256. Beyond that, it will
//! allocate tree nodes of capacity *k* as needed.
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
use std::ops::{Add, Index, IndexMut};

use std::ops::{Bound, RangeBounds};

use nodes::chunk::{
    Chunk, ConsumingIter as ConsumingChunkIter, Iter as ChunkIter, IterMut as ChunkIterMut,
    CHUNK_SIZE,
};
use nodes::rrb::{
    ConsumingIter as ConsumingNodeIter, Iter as NodeIter, IterMut as NodeIterMut, Node, PopResult,
    PushResult, SplitResult,
};
use sort;
use util::{clone_ref, swap_indices, Ref, Side};

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
/// This is a sequence of elements in insertion order - if you need a
/// list of things, any kind of list of things, this is what you're
/// looking for.
///
/// It's implemented as an [RRB vector][rrbpaper] with [smart
/// head/tail chunking][chunkedseq]. In performance terms, this means
/// that practically every operation is O(log n), except push/pop on
/// both sides, which will be O(1) amortised, and O(log n) in the
/// worst case. In practice, the push/pop operations will be
/// blindingly fast, nearly on par with the native
/// [`VecDeque`][VecDeque], and other operations will have decent, if
/// not high, performance, but they all have more or less the same
/// O(log n) complexity, so you don't need to keep their performance
/// characteristics in mind - everything, even splitting and merging,
/// is safe to use and never too slow.
///
/// ## Performance Notes
///
/// Because of the head/tail chunking technique, until you push a
/// number of items above double the tree's branching factor (that's
/// `self.len()` = 2 × *k* (where *k* = 64) = 128) on either side, the
/// data structure is still just a handful of arrays, not yet an RRB
/// tree, so you'll see performance and memory characteristics similar
/// to [`Vec`][Vec] or [`VecDeque`][VecDeque].
///
/// This means that the structure always preallocates four chunks of
/// size *k* (*k* being the tree's branching factor), equivalent to a
/// [`Vec`][Vec] with an initial capacity of 256. Beyond that, it will
/// allocate tree nodes of capacity *k* as needed.
///
/// [rrbpaper]: https://infoscience.epfl.ch/record/213452/files/rrbvector.pdf
/// [chunkedseq]: http://deepsea.inria.fr/pasl/chunkedseq.pdf
/// [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
/// [VecDeque]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html
pub struct Vector<A> {
    length: usize,
    middle_level: usize,
    outer_f: Ref<Chunk<A>>,
    inner_f: Ref<Chunk<A>>,
    middle: Ref<Node<A>>,
    inner_b: Ref<Chunk<A>>,
    outer_b: Ref<Chunk<A>>,
}

impl<A: Clone> Vector<A> {
    /// Construct an empty vector.
    #[must_use]
    pub fn new() -> Self {
        Vector {
            length: 0,
            middle_level: 0,
            outer_f: Ref::new(Chunk::new()),
            inner_f: Ref::new(Chunk::new()),
            middle: Ref::new(Node::new()),
            inner_b: Ref::new(Chunk::new()),
            outer_b: Ref::new(Chunk::new()),
        }
    }

    /// Construct a vector with a single value.
    #[must_use]
    pub fn singleton(a: A) -> Self {
        let mut out = Self::new();
        out.push_back(a);
        out
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
        self.length
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

        let mut local_index = index;

        if local_index < self.outer_f.len() {
            return Some(&self.outer_f[local_index]);
        }
        local_index -= self.outer_f.len();

        if local_index < self.inner_f.len() {
            return Some(&self.inner_f[local_index]);
        }
        local_index -= self.inner_f.len();

        if local_index < self.middle.len() {
            return Some(self.middle.index(self.middle_level, local_index));
        }
        local_index -= self.middle.len();

        if local_index < self.inner_b.len() {
            return Some(&self.inner_b[local_index]);
        }
        local_index -= self.inner_b.len();

        Some(&self.outer_b[local_index])
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

        let mut local_index = index;

        if local_index < self.outer_f.len() {
            let outer_f = Ref::make_mut(&mut self.outer_f);
            return Some(&mut outer_f[local_index]);
        }
        local_index -= self.outer_f.len();

        if local_index < self.inner_f.len() {
            let inner_f = Ref::make_mut(&mut self.inner_f);
            return Some(&mut inner_f[local_index]);
        }
        local_index -= self.inner_f.len();

        if local_index < self.middle.len() {
            let middle = Ref::make_mut(&mut self.middle);
            return Some(middle.index_mut(self.middle_level, local_index));
        }
        local_index -= self.middle.len();

        if local_index < self.inner_b.len() {
            let inner_b = Ref::make_mut(&mut self.inner_b);
            return Some(&mut inner_b[local_index]);
        }
        local_index -= self.inner_b.len();

        let outer_b = Ref::make_mut(&mut self.outer_b);
        Some(&mut outer_b[local_index])
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
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn set(&mut self, index: usize, value: A) {
        self[index] = value;
    }

    /// Swaps the elements at indices `i` and `j`.
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
        if self.outer_f.is_full() {
            swap(&mut self.outer_f, &mut self.inner_f);
            if !self.outer_f.is_empty() {
                let mut chunk = Ref::new(Chunk::new());
                swap(&mut chunk, &mut self.outer_f);
                self.push_middle(Side::Left, chunk);
            }
        }
        self.length += 1;
        let outer_f = Ref::make_mut(&mut self.outer_f);
        outer_f.push_front(value)
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
        if self.outer_b.is_full() {
            swap(&mut self.outer_b, &mut self.inner_b);
            if !self.outer_b.is_empty() {
                let mut chunk = Ref::new(Chunk::new());
                swap(&mut chunk, &mut self.outer_b);
                self.push_middle(Side::Right, chunk);
            }
        }
        self.length += 1;
        let outer_b = Ref::make_mut(&mut self.outer_b);
        outer_b.push_back(value)
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

        if self.middle.is_empty() && other.middle.is_empty() {
            if self.inner_b.is_empty()
                && self.outer_b.is_empty()
                && other.inner_f.is_empty()
                && other.outer_f.is_empty()
            {
                self.inner_b = other.inner_b;
                self.outer_b = other.outer_b;
                self.length += other.length;
                return;
            }
            if self.len() + other.len() <= CHUNK_SIZE * 4 {
                while let Some(value) = other.pop_front() {
                    self.push_back(value);
                }
                return;
            }
        }

        let inner_b1 = self.inner_b.clone();
        self.push_middle(Side::Right, inner_b1);
        let outer_b1 = self.outer_b.clone();
        self.push_middle(Side::Right, outer_b1);
        let inner_f2 = other.inner_f.clone();
        other.push_middle(Side::Left, inner_f2);
        let outer_f2 = other.outer_f.clone();
        other.push_middle(Side::Left, outer_f2);

        let mut middle1 = clone_ref(replace(&mut self.middle, Ref::from(Node::new())));
        let mut middle2 = clone_ref(other.middle);
        self.middle_level = if self.middle_level > other.middle_level {
            middle2 = middle2.elevate(self.middle_level - other.middle_level);
            self.middle_level
        } else if self.middle_level < other.middle_level {
            middle1 = middle1.elevate(other.middle_level - self.middle_level);
            other.middle_level
        } else {
            self.middle_level
        } + 1;
        self.middle = Ref::new(Node::merge(
            Ref::from(middle1),
            Ref::from(middle2),
            self.middle_level - 1,
        ));

        self.inner_b = other.inner_b;
        self.outer_b = other.outer_b;
        self.length += other.length;
        self.prune();
    }

    /// Retain only the elements specified by the predicate.
    ///
    /// Remove all elements for which the provided function `f`
    /// returns false from the vector.
    ///
    /// Time: O(n)
    // FIXME will actually be O(n log n) without the focus optimisation
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&A) -> bool,
    {
        let len = self.len();
        let mut del = 0;
        for i in 0..len {
            if !f(&self[i]) {
                del += 1;
            } else if del > 0 {
                self.swap(i - del, i);
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

        let mut local_index = index;

        if local_index < self.outer_f.len() {
            let of2 = Ref::make_mut(&mut self.outer_f).split(local_index);
            let right = Vector {
                length: self.length - index,
                middle_level: self.middle_level,
                outer_f: Ref::new(of2),
                inner_f: replace_def(&mut self.inner_f),
                middle: replace_def(&mut self.middle),
                inner_b: replace_def(&mut self.inner_b),
                outer_b: replace_def(&mut self.outer_b),
            };
            self.length = index;
            self.middle_level = 0;
            return right;
        }

        local_index -= self.outer_f.len();

        if local_index < self.inner_f.len() {
            let if2 = Ref::make_mut(&mut self.inner_f).split(local_index);
            let right = Vector {
                length: self.length - index,
                middle_level: self.middle_level,
                outer_f: Ref::new(if2),
                inner_f: Ref::<Chunk<A>>::default(),
                middle: replace_def(&mut self.middle),
                inner_b: replace_def(&mut self.inner_b),
                outer_b: replace_def(&mut self.outer_b),
            };
            self.length = index;
            self.middle_level = 0;
            swap(&mut self.outer_b, &mut self.inner_f);
            return right;
        }

        local_index -= self.inner_f.len();

        if local_index < self.middle.len() {
            let mut right_middle = self.middle.clone();
            let (c1, c2) = {
                let m1 = Ref::make_mut(&mut self.middle);
                let m2 = Ref::make_mut(&mut right_middle);
                match m1.split(self.middle_level, Side::Right, local_index) {
                    SplitResult::Dropped(_) => (),
                    SplitResult::OutOfBounds => unreachable!(),
                };
                match m2.split(self.middle_level, Side::Left, local_index) {
                    SplitResult::Dropped(_) => (),
                    SplitResult::OutOfBounds => unreachable!(),
                };
                let c1 = match m1.pop_chunk(self.middle_level, Side::Right) {
                    PopResult::Empty => Ref::<Chunk<A>>::default(),
                    PopResult::Done(chunk) => chunk,
                    PopResult::Drained(chunk) => {
                        m1.clear_node();
                        chunk
                    }
                };
                let c2 = match m2.pop_chunk(self.middle_level, Side::Left) {
                    PopResult::Empty => Ref::<Chunk<A>>::default(),
                    PopResult::Done(chunk) => chunk,
                    PopResult::Drained(chunk) => {
                        m2.clear_node();
                        chunk
                    }
                };
                (c1, c2)
            };
            let mut right = Vector {
                length: self.length - index,
                middle_level: self.middle_level,
                outer_f: c2,
                inner_f: Ref::<Chunk<A>>::default(),
                middle: right_middle,
                inner_b: replace_def(&mut self.inner_b),
                outer_b: replace(&mut self.outer_b, c1),
            };
            self.length = index;
            self.prune();
            right.prune();
            return right;
        }

        local_index -= self.middle.len();

        if local_index < self.inner_b.len() {
            let ib2 = Ref::make_mut(&mut self.inner_b).split(local_index);
            let right = Vector {
                length: self.length - index,
                outer_b: replace_def(&mut self.outer_b),
                outer_f: Ref::new(ib2),
                ..Vector::new()
            };
            self.length = index;
            swap(&mut self.outer_b, &mut self.inner_b);
            return right;
        }

        local_index -= self.inner_b.len();

        let ob2 = Ref::make_mut(&mut self.outer_b).split(local_index);
        let right = Vector {
            length: self.length - index,
            outer_b: Ref::new(ob2),
            ..Vector::new()
        };
        self.length = index;
        right
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
        let start_index = match range.start_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
            Bound::Unbounded => 0,
        };
        let end_index = match range.end_bound() {
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
            Bound::Unbounded => self.len(),
        };
        if start_index >= end_index || start_index >= self.len() {
            return Vector::new();
        }
        let mut middle = self.split_off(start_index);
        let right = middle.split_off(end_index - start_index);
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
        // TODO a lot of optimisations still possible here
        assert!(index < self.len());
        let right = self.split_off(index);
        self.push_back(value);
        self.append(right);
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

    /// Discard all elements from the vector.
    ///
    /// This leaves you with an empty vector, and all elements that
    /// were previously inside it are dropped.
    ///
    /// Time: O(n)
    pub fn clear(&mut self) {
        if !self.is_empty() {
            self.length = 0;
            self.middle_level = 0;
            if !self.outer_f.is_empty() {
                self.outer_f = Ref::<Chunk<A>>::default();
            }
            if !self.inner_f.is_empty() {
                self.inner_f = Ref::<Chunk<A>>::default();
            }
            if !self.middle.is_empty() {
                self.middle = Ref::<Node<A>>::default();
            }
            if !self.inner_b.is_empty() {
                self.inner_b = Ref::<Chunk<A>>::default();
            }
            if !self.outer_b.is_empty() {
                self.outer_b = Ref::<Chunk<A>>::default();
            }
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
            sort::quicksort(self, 0, len - 1, &cmp);
        }
    }

    // Implementation details

    fn prune(&mut self) {
        while self.middle_level > 0 && self.middle.is_single() {
            self.middle = self.middle.first_child().clone();
            self.middle_level -= 1;
        }
    }

    fn push_middle(&mut self, side: Side, chunk: Ref<Chunk<A>>) {
        let new_middle = {
            let middle = Ref::make_mut(&mut self.middle);
            match middle.push_chunk(self.middle_level, side, chunk) {
                PushResult::Done => return,
                PushResult::Full(chunk) => Ref::from({
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

impl<A> Clone for Vector<A> {
    fn clone(&self) -> Self {
        Vector {
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

impl<A: Clone + Debug> Debug for Vector<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(not(has_specialisation))]
impl<A: Clone + PartialEq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Clone + PartialEq> PartialEq for Vector<A> {
    default fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Clone + Eq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() || self.middle_level != other.middle_level {
            return false;
        }

        fn cmp_chunk<A>(left: &Ref<Chunk<A>>, right: &Ref<Chunk<A>>) -> bool {
            (left.is_empty() && right.is_empty()) || Ref::ptr_eq(left, right)
        }

        if cmp_chunk(&self.outer_f, &other.outer_f)
            && cmp_chunk(&self.inner_f, &other.inner_f)
            && ((self.middle.is_empty() && other.middle.is_empty())
                || Ref::ptr_eq(&self.middle, &other.middle))
            && cmp_chunk(&self.inner_b, &other.inner_b)
            && cmp_chunk(&self.outer_b, &other.outer_b)
        {
            return true;
        }
        self.iter().eq(other.iter())
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
        slice.into_iter().cloned().collect()
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
        vec.into_iter().cloned().collect()
    }
}

// Iterators

/// An iterator over vectors with values of type `A`.
///
/// To obtain one, use [`Vector::iter()`][iter].
///
/// [iter]: struct.Vector.html#method.iter
pub struct Iter<'a, A: 'a> {
    iter: Chain<
        Chain<Chain<Chain<ChunkIter<'a, A>, ChunkIter<'a, A>>, NodeIter<'a, A>>, ChunkIter<'a, A>>,
        ChunkIter<'a, A>,
    >,
}

impl<'a, A: Clone> Iter<'a, A> {
    fn new(seq: &'a Vector<A>) -> Self {
        let outer_f = seq.outer_f.iter();
        let inner_f = seq.inner_f.iter();
        let middle = NodeIter::new(&seq.middle);
        let inner_b = seq.inner_b.iter();
        let outer_b = seq.outer_b.iter();
        Iter {
            iter: outer_f
                .chain(inner_f)
                .chain(middle)
                .chain(inner_b)
                .chain(outer_b),
        }
    }
}

impl<'a, A: Clone> Iterator for Iter<'a, A> {
    type Item = &'a A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A: Clone> DoubleEndedIterator for Iter<'a, A> {
    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, A: Clone> ExactSizeIterator for Iter<'a, A> {}

impl<'a, A: Clone> FusedIterator for Iter<'a, A> {}

/// A mutable iterator over vectors with values of type `A`.
///
/// To obtain one, use [`Vector::iter_mut()`][iter_mut].
///
/// [iter_mut]: struct.Vector.html#method.iter_mut
pub struct IterMut<'a, A: 'a> {
    iter: Chain<
        Chain<
            Chain<Chain<ChunkIterMut<'a, A>, ChunkIterMut<'a, A>>, NodeIterMut<'a, A>>,
            ChunkIterMut<'a, A>,
        >,
        ChunkIterMut<'a, A>,
    >,
}

impl<'a, A: Clone> IterMut<'a, A> {
    fn new(seq: &'a mut Vector<A>) -> Self {
        let outer_f = Ref::make_mut(&mut seq.outer_f).iter_mut();
        let inner_f = Ref::make_mut(&mut seq.inner_f).iter_mut();
        let middle = NodeIterMut::new(Ref::make_mut(&mut seq.middle), seq.middle_level);
        let inner_b = Ref::make_mut(&mut seq.inner_b).iter_mut();
        let outer_b = Ref::make_mut(&mut seq.outer_b).iter_mut();
        IterMut {
            iter: outer_f
                .chain(inner_f)
                .chain(middle)
                .chain(inner_b)
                .chain(outer_b),
        }
    }
}

impl<'a, A: Clone> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(log n)
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A: Clone> DoubleEndedIterator for IterMut<'a, A> {
    /// Remove and return an element from the back of the iterator.
    ///
    /// Time: O(log n)
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, A: Clone> ExactSizeIterator for IterMut<'a, A> {}

impl<'a, A: Clone> FusedIterator for IterMut<'a, A> {}

/// A consuming iterator over vectors with values of type `A`.
pub struct ConsumingIter<A> {
    iter: Chain<
        Chain<
            Chain<Chain<ConsumingChunkIter<A>, ConsumingChunkIter<A>>, ConsumingNodeIter<A>>,
            ConsumingChunkIter<A>,
        >,
        ConsumingChunkIter<A>,
    >,
}

impl<A: Clone> ConsumingIter<A> {
    fn new(seq: Vector<A>) -> Self {
        let outer_f = clone_ref(seq.outer_f).into_iter();
        let inner_f = clone_ref(seq.inner_f).into_iter();
        let middle = ConsumingNodeIter::new(clone_ref(seq.middle), seq.middle_level);
        let inner_b = clone_ref(seq.inner_b).into_iter();
        let outer_b = clone_ref(seq.outer_b).into_iter();
        ConsumingIter {
            iter: outer_f
                .chain(inner_f)
                .chain(middle)
                .chain(inner_b)
                .chain(outer_b),
        }
    }
}

impl<A: Clone> Iterator for ConsumingIter<A> {
    type Item = A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<A: Clone> DoubleEndedIterator for ConsumingIter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<A: Clone> ExactSizeIterator for ConsumingIter<A> {}

impl<A: Clone> FusedIterator for ConsumingIter<A> {}

// QuickCheck

#[cfg(all(feature = "arc", any(test, feature = "quickcheck")))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(feature = "arc", any(test, feature = "quickcheck")))]
impl<A: Arbitrary + Sync + Clone> Arbitrary for Vector<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Vector::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::collection::vec;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
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
    use super::*;
    use proptest::collection::vec;
    use proptest::num::{i32, usize};

    // #[test]
    // fn push_and_pop_things() {
    //     let mut seq = Vector::new();
    //     for i in 0..1000 {
    //         seq.push_back(i);
    //     }
    //     for i in 0..1000 {
    //         assert_eq!(Some(i), seq.pop_front());
    //     }
    //     assert!(seq.is_empty());
    //     for i in 0..1000 {
    //         seq.push_front(i);
    //     }
    //     for i in 0..1000 {
    //         assert_eq!(Some(i), seq.pop_back());
    //     }
    //     assert!(seq.is_empty());
    // }

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
    }
}
