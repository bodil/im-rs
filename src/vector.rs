// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A vector.
//!
//! This is an implementation of [bitmapped vector tries][bmvt], which
//! offers highly efficient (amortised linear time) index lookups as
//! well as appending elements to, or popping elements off, either
//! side of the vector.
//!
//! This is generally the best data structure if you're looking for
//! something list like. If you don't need lookups or updates by
//! index, but do need fast concatenation of whole lists, you should
//! use the [`CatList`][CatList] instead.
//!
//! If you're familiar with the Clojure variant, this improves on it
//! by being efficiently extensible at the front as well as the back.
//! If you're familiar with [Immutable.js][immutablejs], this is
//! essentially the same, but with easier mutability because Rust has
//! the advantage of direct access to the garbage collector (which in
//! our case is just [`Arc`][Arc]).
//!
//! [bmvt]: https://hypirion.com/musings/understanding-persistent-vector-pt-1
//! [immutablejs]: https://facebook.github.io/immutable-js/
//! [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//! [Arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
//! [CatList]: ../catlist/struct.CatList.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, Sum};
use std::ops::{Add, Index, IndexMut};
use std::sync::Arc;

use bits::{HASH_BITS, HASH_MASK, HASH_SIZE};
use shared::Shared;

use nodes::vector::{Entry, Node};

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
            l.push_back_mut($x);
        )*
            l
    }};
}

#[derive(Clone, Copy)]
struct Meta {
    origin: usize,
    capacity: usize,
    level: usize,
    reverse: bool,
}

impl Default for Meta {
    fn default() -> Self {
        Meta {
            origin: 0,
            capacity: 0,
            level: HASH_BITS,
            reverse: false,
        }
    }
}

/// A persistent vector of elements of type `A`.
///
/// This is an implementation of [bitmapped vector tries][bmvt], which
/// offers highly efficient index lookups as well as appending
/// elements to, or popping elements off, either side of the vector.
///
/// This is generally the best data structure if you're looking for
/// something list like. If you don't need lookups or updates by
/// index, but do need fast concatenation of whole lists, you should
/// use the [`CatList`][CatList] instead.
///
/// If you're familiar with the Clojure variant, this improves on it
/// by being efficiently extensible at the front as well as the back.
/// If you're familiar with [Immutable.js][immutablejs], this is
/// essentially the same, but with easier mutability because Rust has
/// the advantage of direct access to the garbage collector (which in
/// our case is just [`Arc`][Arc]).
///
/// [bmvt]: https://hypirion.com/musings/understanding-persistent-vector-pt-1
/// [immutablejs]: https://facebook.github.io/immutable-js/
/// [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
/// [Arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
/// [CatList]: ../catlist/struct.CatList.html
pub struct Vector<A> {
    meta: Meta,
    root: Arc<Node<A>>,
    tail: Arc<Node<A>>,
}

impl<A> Vector<A> {
    /// Construct an empty vector.
    pub fn new() -> Self {
        Vector {
            meta: Default::default(),
            root: Default::default(),
            tail: Default::default(),
        }
    }

    /// Construct a vector with a single value.
    pub fn singleton<R>(a: R) -> Self
    where
        R: Shared<A>,
    {
        let mut tail = Node::new();
        tail.push(Entry::Value(a.shared()));
        let mut meta = Meta::default();
        meta.capacity = 1;
        Vector {
            meta,
            root: Default::default(),
            tail: Arc::new(tail),
        }
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
    pub fn len(&self) -> usize {
        self.meta.capacity - self.meta.origin
    }

    /// Test whether a list is empty.
    ///
    /// Time: O(1)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an iterator over a vector.
    ///
    /// Time: O(log n) per [`next()`][next] call
    ///
    /// [next]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
    #[inline]
    pub fn iter(&self) -> Iter<A> {
        Iter::new(self.clone())
    }

    /// Get the first element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn head(&self) -> Option<Arc<A>> {
        self.get(0)
    }

    /// Get the vector without the first element.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    pub fn tail(&self) -> Option<Vector<A>> {
        if self.is_empty() {
            None
        } else {
            let mut v = self.clone();
            v.resize(1, self.len() as isize);
            Some(v)
        }
    }

    /// Get the last element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    pub fn last(&self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Get the vector without the last element.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(log n)
    pub fn init(&self) -> Option<Vector<A>> {
        if self.is_empty() {
            None
        } else {
            let mut v = self.clone();
            v.resize(0, (self.len() - 1) as isize);
            Some(v)
        }
    }

    /// Get the value at index `index` in a vector.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// Time: O(log n)
    pub fn get(&self, index: usize) -> Option<Arc<A>> {
        let i = match self.map_index(index) {
            None => return None,
            Some(i) => i,
        };

        let node = self.node_for(i);
        match node.get(i & HASH_MASK as usize) {
            Some(&Entry::Value(ref value)) => Some(value.clone()),
            Some(&Entry::Node(_)) => panic!("Vector::get: encountered node, expected value"),
            Some(&Entry::Empty) => panic!("Vector::get: encountered null, expected value"),
            None => panic!("Vector::get: unhandled index out of bounds situation!"),
        }
    }

    /// Get the value at index `index` in a vector, directly.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(log n)
    pub fn get_unwrapped(&self, index: usize) -> Arc<A> {
        self.get(index).expect("get_unwrapped index out of bounds")
    }

    /// Create a new vector with the value at index `index` updated.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(log n)
    pub fn set<RA>(&self, index: usize, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        let i = match self.map_index(index) {
            None => panic!("index out of bounds: {} < {}", index, self.len()),
            Some(i) => i,
        };
        if i >= tail_offset(self.meta.capacity) {
            let mut tail = (*self.tail).clone();
            tail.set(i & HASH_MASK as usize, Entry::Value(value.shared()));
            self.update_tail(tail)
        } else {
            self.update_root(
                self.root
                    .set_in(self.meta.level, i, Entry::Value(value.shared())),
            )
        }
    }

    /// Update the value at index `index` in a vector.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// vector's structure which are shared with other vectors will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn set_mut<RA>(&mut self, index: usize, value: RA)
    where
        RA: Shared<A>,
    {
        let i = match self.map_index(index) {
            None => panic!("index out of bounds: {} < {}", index, self.len()),
            Some(i) => i,
        };
        if i >= tail_offset(self.meta.capacity) {
            let tail = Arc::make_mut(&mut self.tail);
            tail.set(i & HASH_MASK as usize, Entry::Value(value.shared()));
        } else {
            let root = Arc::make_mut(&mut self.root);
            root.set_in_mut(self.meta.level, 0, i, Entry::Value(value.shared()))
        }
    }

    /// Construct a vector with a new value prepended to the end of
    /// the current vector.
    ///
    /// Time: O(log n)
    pub fn push_back<RA>(&self, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        let len = self.len();
        let mut v = self.clone();
        v.resize(0, (len + 1) as isize);
        v.set_mut(len, value.shared());
        v
    }

    /// Construct a vector with a new value prepended to the end of
    /// the current vector.
    ///
    /// `snoc`, for the curious, is [`cons`][cons] spelled backwards,
    /// to denote that it works on the back of the list rather than
    /// the front. If you don't find that as clever as whoever coined
    /// the term no doubt did, this method is also available as
    /// [`push_back()`][push_back].
    ///
    /// Time: O(log n)
    ///
    /// [push_back]: #method.push_back
    /// [cons]: #method.cons
    #[inline]
    pub fn snoc<RA>(&self, a: RA) -> Self
    where
        RA: Shared<A>,
    {
        self.push_back(a)
    }

    /// Update a vector in place with a new value prepended to the end
    /// of it.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// vector's structure which are shared with other vectors will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn push_back_mut<RA>(&mut self, value: RA)
    where
        RA: Shared<A>,
    {
        let len = self.len();
        self.resize(0, (len + 1) as isize);
        self.set_mut(len, value.shared());
    }

    /// Construct a vector with a new value prepended to the front of
    /// the current vector.
    ///
    /// Time: O(log n)
    pub fn push_front<RA>(&self, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        let mut v = self.clone();
        v.resize(-1, self.len() as isize);
        v.set_mut(0, value.shared());
        v
    }

    /// Construct a vector with a new value prepended to the front of
    /// the current vector.
    ///
    /// This is an alias for [push_front], for the Lispers in the
    /// house.
    ///
    /// Time: O(log n)
    ///
    /// [push_front]: #method.push_front
    #[inline]
    pub fn cons<RA>(&self, a: RA) -> Self
    where
        RA: Shared<A>,
    {
        self.push_front(a)
    }

    /// Update a vector in place with a new value prepended to the
    /// front of it.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// vector's structure which are shared with other vectors will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn push_front_mut<RA>(&mut self, value: RA)
    where
        RA: Shared<A>,
    {
        let len = self.len();
        self.resize(-1, len as isize);
        self.set_mut(0, value.shared());
    }

    /// Get the last element of a vector, as well as the vector with
    /// the last element removed.
    ///
    /// If the vector is empty, [`None`][None] is returned.
    ///
    /// Time: O(log n)
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn pop_back(&self) -> Option<(Arc<A>, Self)> {
        if self.is_empty() {
            return None;
        }
        let val = self.get(self.len() - 1).unwrap();
        let mut v = self.clone();
        v.resize(0, -1 as isize);
        Some((val, v))
    }

    /// Remove the last element of a vector in place and return it.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// vector's structure which are shared with other vectors will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn pop_back_mut(&mut self) -> Option<Arc<A>> {
        if self.is_empty() {
            return None;
        }
        let val = self.get(self.len() - 1).unwrap();
        self.resize(0, -1);
        Some(val)
    }

    /// Get the first element of a vector, as well as the vector with
    /// the first element removed.
    ///
    /// If the vector is empty, [`None`][None] is returned.
    ///
    /// Time: O(log n)
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn pop_front(&self) -> Option<(Arc<A>, Self)> {
        if self.is_empty() {
            return None;
        }
        let val = self.get(0).unwrap();
        let mut v = self.clone();
        v.resize(1, self.len() as isize);
        Some((val, v))
    }

    /// Get the head and the tail of a vector.
    ///
    /// If the vector is empty, [`None`][None] is returned.
    ///
    /// This is an alias for [`pop_front`][pop_front].
    ///
    /// Time: O(log n)
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    /// [pop_front]: #method.pop_front
    #[inline]
    pub fn uncons(&self) -> Option<(Arc<A>, Self)> {
        self.pop_front()
    }

    /// Get the last element of a vector, as well as the vector with the
    /// last element removed.
    ///
    /// If the vector is empty, [`None`][None] is returned.
    ///
    /// This is an alias for [`pop_back`][pop_back].
    ///
    /// Time: O(1)*
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    /// [pop_back]: #method.pop_back
    #[inline]
    pub fn unsnoc(&self) -> Option<(Arc<A>, Vector<A>)> {
        self.pop_back()
    }

    /// Remove the first element of a vector in place and return it.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// vector's structure which are shared with other vectors will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn pop_front_mut(&mut self) -> Option<Arc<A>> {
        if self.is_empty() {
            return None;
        }
        let len = self.len();
        let val = self.get(0).unwrap();
        self.resize(1, len as isize);
        Some(val)
    }

    /// Split a vector at a given index, returning a vector containing
    /// every element before of the index and a vector containing
    /// every element from the index onward.
    ///
    /// Time: O(log n)
    pub fn split_at(&self, index: usize) -> (Self, Self) {
        if index >= self.len() {
            return (self.clone(), Vector::new());
        }
        let mut left = self.clone();
        left.resize(0, index as isize);
        let mut right = self.clone();
        right.resize(index as isize, self.len() as isize);
        (left, right)
    }

    /// Construct a vector with `count` elements removed from the
    /// start of the current vector.
    ///
    /// Time: O(log n)
    pub fn skip(&self, count: usize) -> Self {
        let mut v = self.clone();
        v.resize(count as isize, self.len() as isize);
        v
    }

    /// Construct a vector of the first `count` elements from the
    /// current vector.
    ///
    /// Time: O(log n)
    pub fn take(&self, count: usize) -> Self {
        let mut v = self.clone();
        v.resize(0, count as isize);
        v
    }

    /// Construct a vector with the elements from `start_index`
    /// until `end_index` in the current vector.
    ///
    /// Time: O(log n)
    pub fn slice(&self, start_index: usize, end_index: usize) -> Self {
        if start_index >= end_index || start_index >= self.len() {
            return Vector::new();
        }
        let mut v = self.clone();
        v.resize(start_index as isize, end_index as isize);
        v
    }

    /// Append the vector `other` to the end of the current vector.
    ///
    /// Time: O(n) where n = the length of `other`
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![1, 2, 3].append(vector![7, 8, 9]),
    ///   vector![1, 2, 3, 7, 8, 9]
    /// );
    /// # }
    /// ```
    pub fn append<R>(&self, other: R) -> Self
    where
        R: Borrow<Self>,
    {
        let o = other.borrow();
        let mut v = self.clone();
        v.resize(0, (self.len() + o.len()) as isize);
        v.write(self.len(), o);
        v
    }

    /// Write from an iterator into a vector, starting at the given
    /// index.
    ///
    /// This will overwrite elements in the vector until the iterator
    /// ends or the end of the vector is reached.
    ///
    /// Time: O(n) where n = the length of the iterator
    pub fn write<I: IntoIterator<Item = R>, R: Shared<A>>(&mut self, index: usize, iter: I) {
        if let Some(raw_index) = self.map_index(index) {
            let cap = self.meta.capacity;
            let tail_offset = tail_offset(cap);
            let mut it = iter.into_iter().map(|i| i.shared());
            if raw_index >= tail_offset {
                let mut tail = Arc::make_mut(&mut self.tail);
                let mut i = raw_index - tail_offset;
                loop {
                    match it.next() {
                        None => return,
                        Some(value) => {
                            tail.set(i, Entry::Value(value));
                            i += 1;
                            if tail_offset + i >= cap {
                                return;
                            }
                        }
                    }
                }
            } else {
                let root = Arc::make_mut(&mut self.root);
                let mut max_in_root = tail_offset - raw_index;
                root.write(self.meta.level, raw_index, &mut it, &mut max_in_root);
                let mut tail = Arc::make_mut(&mut self.tail);
                let mut i = 0;
                loop {
                    match it.next() {
                        None => return,
                        Some(value) => {
                            tail.set(i, Entry::Value(value));
                            i += 1;
                            if tail_offset + i >= cap {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Construct a vector which is the reverse of the current vector.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![1, 2, 3, 4, 5].reverse(),
    ///   vector![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    ///
    /// [rev]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.rev
    pub fn reverse(&self) -> Self {
        let mut v = self.clone();
        v.meta.reverse = !v.meta.reverse;
        v
    }

    /// Reverse a vector in place.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// let mut v = vector![1, 2, 3, 4, 5];
    /// v.reverse_mut();
    ///
    /// assert_eq!(
    ///   v,
    ///   vector![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    pub fn reverse_mut(&mut self) {
        self.meta.reverse = !self.meta.reverse;
    }

    /// Sort a vector of ordered elements.
    ///
    /// Time: O(n log n) worst case
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   vector![1, 2, 3, 4, 5, 6, 7, 8]
    /// );
    /// # }
    /// ```
    pub fn sort(&self) -> Self
    where
        A: Ord,
    {
        self.sort_by(Ord::cmp)
    }

    /// Sort a vector using a comparator function.
    ///
    /// Time: O(n log n) roughly
    pub fn sort_by<F>(&self, cmp: F) -> Self
    where
        F: Fn(&A, &A) -> Ordering,
    {
        // FIXME: This is a simple in-place quicksort. There are
        // probably faster algorithms.
        fn swap<A>(vector: &mut Vector<A>, a: usize, b: usize) {
            let aval = vector.get(a).unwrap();
            let bval = vector.get(b).unwrap();
            vector.set_mut(a, bval);
            vector.set_mut(b, aval);
        }

        // Ported from the Java version at
        // http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
        #[cfg_attr(feature = "clippy", allow(many_single_char_names))]
        fn quicksort<A, F>(vector: &mut Vector<A>, l: usize, r: usize, cmp: &F)
        where
            F: Fn(&A, &A) -> Ordering,
        {
            if r <= l {
                return;
            }

            let mut i = l;
            let mut j = r;
            let mut p = i;
            let mut q = j;
            let v = vector.get(r).unwrap();
            loop {
                while cmp(&vector.get_unwrapped(i), &v) == Ordering::Less {
                    i += 1
                }
                j -= 1;
                while cmp(&v, &vector.get_unwrapped(j)) == Ordering::Less {
                    if j == l {
                        break;
                    }
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                swap(vector, i, j);
                if cmp(&vector.get_unwrapped(i), &v) == Ordering::Equal {
                    p += 1;
                    swap(vector, p, i);
                }
                if cmp(&v, &vector.get_unwrapped(j)) == Ordering::Equal {
                    q -= 1;
                    swap(vector, j, q);
                }
                i += 1;
            }
            swap(vector, i, r);

            let mut jp: isize = i as isize - 1;
            let mut k = l;
            i += 1;
            while k < p {
                swap(vector, k, jp as usize);
                jp -= 1;
                k += 1;
            }
            k = r - 1;
            while k > q {
                swap(vector, i, k);
                k -= 1;
                i += 1;
            }

            if jp >= 0 {
                quicksort(vector, l, jp as usize, cmp);
            }
            quicksort(vector, i, r, cmp);
        }

        if self.len() < 2 {
            self.clone()
        } else {
            let mut out = self.clone();
            quicksort(&mut out, 0, self.len() - 1, &cmp);
            out
        }
    }

    /// Insert an item into a sorted vector.
    ///
    /// Constructs a new vector with the new item inserted before the
    /// first item in the vector which is larger than the new item, as
    /// determined by the `Ord` trait.
    ///
    /// Please note that this is a very inefficient operation; if you
    /// want a sorted list, consider if [`OrdSet`][ordset::OrdSet]
    /// might be a better choice for you.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![2, 4, 5].insert(1).insert(3).insert(6),
    ///   vector![1, 2, 3, 4, 5, 6]
    /// );
    /// # }
    /// ```
    ///
    /// [ordset::OrdSet]: ../ordset/struct.OrdSet.html
    pub fn insert<RA>(&self, item: RA) -> Self
    where
        A: Ord,
        RA: Shared<A>,
    {
        let value = item.shared();
        let mut out = Vector::new();
        let mut inserted = false;
        for next in self {
            if next < value {
                out.push_back_mut(next);
                continue;
            }
            if !inserted {
                out.push_back_mut(value.clone());
                inserted = true;
            }
            out.push_back_mut(next);
        }
        if !inserted {
            out.push_back_mut(value);
        }
        out
    }

    // Implementation details

    fn map_index(&self, index: usize) -> Option<usize> {
        let len = self.len();
        if index >= len {
            return None;
        }
        Some(
            self.meta.origin + if self.meta.reverse {
                (len - 1) - index
            } else {
                index
            },
        )
    }

    fn node_for(&self, index: usize) -> &Arc<Node<A>> {
        if index >= tail_offset(self.meta.capacity) {
            &self.tail
        } else {
            let mut node = &self.root;
            let mut level = self.meta.level;
            while level > 0 {
                node = if let Some(&Entry::Node(ref child_node)) =
                    node.children.get((index >> level) & HASH_MASK as usize)
                {
                    level -= HASH_BITS;
                    child_node
                } else {
                    panic!("Vector::node_for: encountered value or null where node was expected")
                };
            }
            node
        }
    }

    fn clear(&mut self) {
        self.meta = Default::default();
        self.root = Default::default();
        self.tail = Default::default();
    }

    fn resize(&mut self, mut start: isize, mut end: isize) {
        if self.meta.reverse {
            let len = self.len() as isize;
            let swap = start;
            start = if end < 0 { 0 } else { len } - end;
            end = len - swap;
        }

        let mut o0 = self.meta.origin;
        let mut c0 = self.meta.capacity;
        let mut o = o0 as isize + start;
        let mut c = if end < 0 {
            c0 as isize + end
        } else {
            o0 as isize + end
        } as usize;
        if o == o0 as isize && c == c0 {
            return;
        }

        if o >= c as isize {
            self.clear();
            return;
        }

        let mut level = self.meta.level;

        // Create higher level roots until origin is positive
        let mut origin_shift = 0;
        while o + origin_shift < 0 {
            if self.root.is_empty() {
                self.root = Default::default()
            } else {
                self.root = Arc::new(Node::from_vec(
                    Some(1),
                    vec![Entry::Empty, Entry::Node(self.root.clone())],
                ));
            }
            level += HASH_BITS;
            origin_shift += 1 << level;
        }
        if origin_shift > 0 {
            o0 += origin_shift as usize;
            c0 += origin_shift as usize;
            o += origin_shift;
            c += origin_shift as usize;
        }

        // Create higher level roots until size fits
        let tail_offset0 = tail_offset(c0);
        let tail_offset = tail_offset(c as usize);
        while tail_offset >= 1 << (level + HASH_BITS) {
            if self.root.is_empty() {
                self.root = Default::default()
            } else {
                self.root = Arc::new(Node::from_vec(
                    Some(0),
                    vec![Entry::Node(self.root.clone())],
                ));
            }
            level += HASH_BITS;
        }

        // Merge old tail into tree
        if tail_offset > tail_offset0 && (o as usize) < c0 && !self.tail.is_empty() {
            let root = Arc::make_mut(&mut self.root);
            root.set_in_mut(
                level,
                HASH_BITS,
                tail_offset0,
                Entry::Node(self.tail.clone()),
            );
        }

        // Find the new tail
        if tail_offset < tail_offset0 {
            self.tail = self.node_for(c - 1).clone()
        } else if tail_offset > tail_offset0 {
            self.tail = Default::default()
        }

        // Trim new tail if needed
        if c < c0 {
            let t = Arc::make_mut(&mut self.tail);
            t.remove_after(0, c);
        }

        // If new origin is inside tail, drop the root
        if o as usize >= tail_offset {
            o -= tail_offset as isize;
            c -= tail_offset;
            level = HASH_BITS;
            self.root = Default::default();
            let tail = Arc::make_mut(&mut self.tail);
            tail.remove_before(0, o as usize);
        } else if o as usize > o0 || tail_offset < tail_offset0 {
            // If root has been trimmed, clean it up.
            let mut shift = 0;
            // Find the top root node inside the old tree.
            loop {
                let start_index = (o as usize >> level) & HASH_MASK as usize;
                if start_index != (tail_offset >> level) & HASH_MASK as usize {
                    break;
                }
                if start_index != 0 {
                    shift += (1 << level) * start_index;
                }
                level -= HASH_BITS;
                self.root = self.root.get(start_index).unwrap().unwrap_node();
            }
            // Trim the edges of the root.
            if o as usize > o0 {
                let root = Arc::make_mut(&mut self.root);
                root.remove_before(level, o as usize - shift);
            }
            if tail_offset < tail_offset0 {
                let root = Arc::make_mut(&mut self.root);
                root.remove_after(level, tail_offset - shift);
            }
            if shift > 0 {
                o -= shift as isize;
                c -= shift;
            }
        }

        self.meta.level = level;
        self.meta.origin = o as usize;
        self.meta.capacity = c;
    }

    fn update_root<Root: Shared<Node<A>>>(&self, root: Root) -> Self {
        Vector {
            meta: self.meta,
            root: root.shared(),
            tail: self.tail.clone(),
        }
    }

    fn update_tail<Root: Shared<Node<A>>>(&self, tail: Root) -> Self {
        Vector {
            meta: self.meta,
            root: self.root.clone(),
            tail: tail.shared(),
        }
    }
}

fn tail_offset(size: usize) -> usize {
    if size < HASH_SIZE {
        0
    } else {
        ((size - 1) >> HASH_BITS) << HASH_BITS
    }
}

// Core traits

impl<A> Clone for Vector<A> {
    fn clone(&self) -> Self {
        Vector {
            meta: self.meta,
            root: self.root.clone(),
            tail: self.tail.clone(),
        }
    }
}

impl<A> Default for Vector<A> {
    fn default() -> Self {
        Vector::new()
    }
}

impl<A: Debug> Debug for Vector<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(not(has_specialisation))]
impl<A: PartialEq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: PartialEq> PartialEq for Vector<A> {
    default fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Eq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if Arc::ptr_eq(&self.root, &other.root) && Arc::ptr_eq(&self.tail, &other.tail) {
            return true;
        }
        self.iter().eq(other.iter())
    }
}

impl<A: Eq> Eq for Vector<A> {}

impl<A: PartialOrd> PartialOrd for Vector<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord> Ord for Vector<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A> Add for Vector<A> {
    type Output = Vector<A>;

    fn add(mut self, other: Self) -> Self::Output {
        self.extend(other);
        self
    }
}

impl<'a, A> Add for &'a Vector<A> {
    type Output = Vector<A>;

    fn add(self, other: Self) -> Self::Output {
        let mut out = self.clone();
        out.extend(other);
        out
    }
}

impl<A: Hash> Hash for Vector<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self {
            i.hash(state)
        }
    }
}

impl<A> Sum for Vector<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, R: Shared<A>> Extend<R> for Vector<A> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = R>,
    {
        let len = self.len();
        let it = iter.into_iter();
        let (lower, upper) = it.size_hint();
        if Some(lower) == upper {
            self.resize(0, (len + lower) as isize);
            self.write(len, it);
        } else {
            for item in it {
                self.push_back_mut(item)
            }
        }
    }
}

impl<A> Index<usize> for Vector<A> {
    type Output = A;

    fn index(&self, index: usize) -> &Self::Output {
        let i = match self.map_index(index) {
            None => panic!("index out of bounds: {} < {}", index, self.len()),
            Some(i) => i,
        };

        let node = self.node_for(i);

        match node.children[index] {
            Entry::Value(ref value) => value,
            _ => panic!("Vector::index: vector structure inconsistent"),
        }
    }
}

impl<A> IndexMut<usize> for Vector<A>
where
    A: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i = match self.map_index(index) {
            None => panic!("index out of bounds: {} < {}", index, self.len()),
            Some(i) => i,
        };
        let entry = if i >= tail_offset(self.meta.capacity) {
            let tail = Arc::make_mut(&mut self.tail);
            &mut tail.children[i & HASH_MASK as usize]
        } else {
            let root = Arc::make_mut(&mut self.root);
            root.ref_mut(self.meta.level, 0, i)
        };
        match *entry {
            Entry::Value(ref mut value) => Arc::make_mut(value),
            _ => panic!("Vector::index_mut: vector structure inconsistent"),
        }
    }
}

// Conversions

impl<A, RA: Shared<A>> FromIterator<RA> for Vector<A> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        let mut v = Vector::new();
        v.extend(iter);
        v
    }
}

impl<A> IntoIterator for Vector<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A> IntoIterator for &'a Vector<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> From<Vec<A>> for Vector<A> {
    fn from(v: Vec<A>) -> Self {
        v.into_iter().collect()
    }
}

// Iterators

/// An iterator over vectors with values of type `A`.
pub struct Iter<A> {
    vector: Vector<A>,
    start_node: Arc<Node<A>>,
    start_index: usize,
    start_offset: usize,
    end_node: Arc<Node<A>>,
    end_index: usize,
    end_offset: usize,
}

impl<A> Iter<A> {
    fn new(vector: Vector<A>) -> Self {
        let start = vector.meta.origin;
        let start_index = start & !(HASH_MASK as usize);
        let end = vector.meta.capacity;
        let end_index = end & !(HASH_MASK as usize);
        Iter {
            start_node: vector.node_for(start_index).clone(),
            end_node: vector.node_for(end_index).clone(),
            start_index,
            start_offset: start - start_index,
            end_index,
            end_offset: end - end_index,
            vector,
        }
    }

    fn get_next(&mut self) -> Option<Arc<A>> {
        if self.start_index + self.start_offset == self.end_index + self.end_offset {
            return None;
        }
        if self.start_offset < HASH_SIZE {
            let item = self.start_node.get(self.start_offset).unwrap().unwrap_val();
            self.start_offset += 1;
            return Some(item);
        }
        self.start_offset = 0;
        self.start_index += HASH_SIZE;
        self.start_node = self.vector.node_for(self.start_index).clone();
        self.get_next()
    }

    fn get_next_back(&mut self) -> Option<Arc<A>> {
        if self.start_index + self.start_offset == self.end_index + self.end_offset {
            return None;
        }
        if self.end_offset > 0 {
            self.end_offset -= 1;
            let item = self.end_node.get(self.end_offset).unwrap().unwrap_val();
            return Some(item);
        }
        self.end_offset = HASH_SIZE;
        self.end_index -= HASH_SIZE;
        self.end_node = self.vector.node_for(self.end_index).clone();
        self.get_next_back()
    }
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.vector.meta.reverse {
            self.get_next_back()
        } else {
            self.get_next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.end_index + self.end_offset) - (self.start_index + self.start_offset);
        (size, Some(size))
    }
}

impl<A> DoubleEndedIterator for Iter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.vector.meta.reverse {
            self.get_next()
        } else {
            self.get_next_back()
        }
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for Vector<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Vector::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
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
    pub fn vector<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<Vector<<T::Value as ValueTree>::Value>> {
        ::proptest::collection::vec(element, size)
            .prop_map(Vector::from_iter)
            .boxed()
    }
}

// Tests

// impl<A: Debug> Debug for Vector<A> {
//     fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
//         write!(
//             f,
//             "Vector o={} c={} l={} n={} root={:?} tail={:?}",
//             self.meta.origin,
//             self.meta.capacity,
//             self.meta.level,
//             self.len(),
//             self.root,
//             self.tail
//         )
//     }
// }

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;
    use proptest::collection;
    use proptest::num::i32;
    use std::iter;

    #[test]
    fn wat() {
        let v1 = Vec::from_iter((0..1000).into_iter().map(Arc::new));
        let v2 = Vector::from_iter(0..1000);
        for (i, item) in v1.into_iter().enumerate() {
            assert_eq!(Some(item), v2.get(i));
        }
    }

    #[test]
    fn double_ended_iterator() {
        let vector = Vector::<i32>::from_iter(1..6);
        let mut it = vector.iter();
        assert_eq!(Some(Arc::new(1)), it.next());
        assert_eq!(Some(Arc::new(5)), it.next_back());
        assert_eq!(Some(Arc::new(2)), it.next());
        assert_eq!(Some(Arc::new(4)), it.next_back());
        assert_eq!(Some(Arc::new(3)), it.next());
        assert_eq!(None, it.next_back());
        assert_eq!(None, it.next());
    }

    #[test]
    fn safe_mutation() {
        let v1 = Vector::from_iter(0..131072);
        let mut v2 = v1.clone();
        v2.set_mut(131000, 23);
        assert_eq!(Some(Arc::new(23)), v2.get(131000));
        assert_eq!(Some(Arc::new(131000)), v1.get(131000));
    }

    #[test]
    fn index_operator() {
        let mut vec = vector![1, 2, 3, 4, 5];
        assert_eq!(4, vec[3]);
        vec[3] = 9;
        assert_eq!(vector![1, 2, 3, 9, 5], vec);
    }

    #[test]
    fn add_operator() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![4, 5, 6];
        assert_eq!(vector![1, 2, 3, 4, 5, 6], vec1 + vec2);
    }

    #[test]
    fn vector_singleton() {
        assert_eq!(Vector::singleton(5).len(), 1);
    }

    proptest! {
        #[test]
        fn push_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector = vector.push_back(value);
                assert_eq!(count + 1, vector.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn push_back_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector.push_back_mut(value);
                assert_eq!(count + 1, vector.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn push_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector = vector.push_front(value);
                assert_eq!(count + 1, vector.len());
            }
            assert_eq!(input, &Vec::from_iter(vector.iter().rev().map(|a| *a)));
        }

        #[test]
        fn push_front_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector.push_front_mut(value);
                assert_eq!(count + 1, vector.len());
            }
            assert_eq!(input, &Vec::from_iter(vector.iter().rev().map(|a| *a)));
        }

        #[test]
        fn from_iter(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn set(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(iter::repeat(0).take(input.len()));
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                vector = vector.set(index, value);
            }
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn set_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(iter::repeat(0).take(input.len()));
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                vector.set_mut(index, value);
            }
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn pop_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop_back() {
                    None => panic!("vector emptied unexpectedly"),
                    Some((item, next)) => {
                        vector = next;
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn pop_back_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop_back_mut() {
                    None => panic!("vector emptied unexpectedly"),
                    Some(item) => {
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn pop_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().rev().enumerate().rev() {
                match vector.pop_front() {
                    None => panic!("vector emptied unexpectedly"),
                    Some((item, next)) => {
                        vector = next;
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn pop_front_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().rev().enumerate().rev() {
                match vector.pop_front_mut() {
                    None => panic!("vector emptied unexpectedly"),
                    Some(item) => {
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            let mut it1 = input.iter().cloned();
            let mut it2 = vector.iter();
            loop {
                match (it1.next(), it2.next()) {
                    (None, None) => break,
                    (Some(i1), Some(i2)) => assert_eq!(i1, *i2),
                    (Some(i1), None) => panic!(format!("expected {:?} but got EOF", i1)),
                    (None, Some(i2)) => panic!(format!("expected EOF but got {:?}", *i2)),
                }
            }
        }

        #[test]
        fn reverse_iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            let mut it1 = input.iter().cloned().rev();
            let mut it2 = vector.iter().rev();
            loop {
                match (it1.next(), it2.next()) {
                    (None, None) => break,
                    (Some(i1), Some(i2)) => assert_eq!(i1, *i2),
                    (Some(i1), None) => panic!(format!("expected {:?} but got EOF", i1)),
                    (None, Some(i2)) => panic!(format!("expected EOF but got {:?}", *i2)),
                }
            }
        }

        #[test]
        fn exact_size_iterator(ref vector in vector(i32::ANY, 0..100)) {
            let mut should_be = vector.len();
            let mut it = vector.iter();
            loop {
                assert_eq!(should_be, it.len());
                match it.next() {
                    None => break,
                    Some(_) => should_be -= 1,
                }
            }
            assert_eq!(0, it.len());
        }

        #[test]
        fn sort(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            let mut input_sorted = input.clone();
            input_sorted.sort();
            let sorted = Vector::from_iter(input_sorted.iter().cloned());
            assert_eq!(sorted, vector.sort());
        }

        #[test]
        fn reverse(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned()).reverse();
            let mut reversed = input.clone();
            reversed.reverse();
            for (index, value) in reversed.into_iter().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
            vector.reverse_mut();
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn reversed_push_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new().reverse();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector = vector.push_front(value);
                assert_eq!(count + 1, vector.len());
            }
            vector = vector.reverse();
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn reversed_push_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new().reverse();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector = vector.push_back(value);
                assert_eq!(count + 1, vector.len());
            }
            vector = vector.reverse();
            assert_eq!(input, &Vec::from_iter(vector.iter().rev().map(|a| *a)));
        }

        #[test]
        fn reversed_pop_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned()).reverse();
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop_front() {
                    None => panic!("vector emptied unexpectedly"),
                    Some((item, next)) => {
                        vector = next;
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn reversed_pop_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned()).reverse();
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().rev().enumerate().rev() {
                match vector.pop_back() {
                    None => panic!("vector emptied unexpectedly"),
                    Some((item, next)) => {
                        vector = next;
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }
    }
}
