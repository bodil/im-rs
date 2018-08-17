// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An unordered set.
//!
//! An immutable hash set using [hash array mapped tries] [1].
//!
//! Most operations on this set are O(log<sub>x</sub> n) for a
//! suitably high *x* that it should be nearly O(1) for most sets.
//! Because of this, it's a great choice for a generic set as long as
//! you don't mind that values will need to implement
//! [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
//!
//! Values will have a predictable order based on the hasher being
//! used. Unless otherwise specified, all sets will use the default
//! [`RandomState`][std::collections::hash_map::RandomState] hasher,
//! which will produce consistent hashes for the duration of its
//! lifetime, but not between restarts of your program.
//!
//! [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
//! [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
//! [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
//! [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.h

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::RandomState;
use std::collections::{self, BTreeSet};
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FusedIterator;
use std::iter::{FromIterator, IntoIterator, Sum};
use std::ops::{Add, Deref, Mul};

use nodes::bitmap::hash_key;
use nodes::hamt::{Drain as NodeDrain, HashValue, Iter as NodeIter, IterMut as NodeIterMut, Node};
use ordset::OrdSet;
use util::Ref;

/// Construct a set from a sequence of values.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::hashset::HashSet;
/// # fn main() {
/// assert_eq!(
///   hashset![1, 2, 3],
///   HashSet::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! hashset {
    () => { $crate::hashset::HashSet::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::hashset::HashSet::new();
        $(
            l.insert($x);
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::hashset::HashSet::new();
        $(
            l.insert($x);
        )*
            l
    }};
}

/// An unordered set.
///
/// An immutable hash set using [hash array mapped tries] [1].
///
/// Most operations on this set are O(log<sub>x</sub> n) for a
/// suitably high *x* that it should be nearly O(1) for most sets.
/// Because of this, it's a great choice for a generic set as long as
/// you don't mind that values will need to implement
/// [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
///
/// Values will have a predictable order based on the hasher being
/// used. Unless otherwise specified, all sets will use the default
/// [`RandomState`][std::collections::hash_map::RandomState] hasher,
/// which will produce consistent hashes for the duration of its
/// lifetime, but not between restarts of your program.
///
/// [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
/// [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
/// [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
/// [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.h
pub struct HashSet<A, S = RandomState> {
    hasher: Ref<S>,
    root: Ref<Node<Value<A>>>,
    size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Value<A>(A);

impl<A> Deref for Value<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// FIXME lacking specialisation, we can't simply implement `HashValue`
// for `A`, we have to use the `Value<A>` indirection.
impl<A> HashValue for Value<A>
where
    A: Hash + Eq + Clone,
{
    type Key = A;

    fn extract_key(&self) -> &Self::Key {
        &self.0
    }

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<A> HashSet<A, RandomState>
where
    A: Hash + Eq + Clone,
{
    /// Construct an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a set with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let set = HashSet::singleton(123);
    /// assert!(set.contains(&123));
    /// # }
    /// ```
    #[must_use]
    pub fn singleton(a: A) -> Self {
        HashSet::new().update(a)
    }
}

impl<A, S> HashSet<A, S> {
    /// Test whether a set is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # fn main() {
    /// assert!(
    ///   !hashset![1, 2, 3].is_empty()
    /// );
    /// assert!(
    ///   HashSet::<i32>::new().is_empty()
    /// );
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size of a set.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # fn main() {
    /// assert_eq!(3, hashset![1, 2, 3].len());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }
}

impl<A, S> HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    fn test_eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let mut seen = collections::HashSet::new();
        for value in self.iter() {
            if !other.contains(&value) {
                return false;
            }
            seen.insert(value);
        }
        for value in other.iter() {
            if !seen.contains(&value) {
                return false;
            }
        }
        true
    }

    /// Construct an empty hash set using the provided hasher.
    #[inline]
    #[must_use]
    pub fn with_hasher<RS>(hasher: RS) -> Self
    where
        Ref<S>: From<RS>,
    {
        HashSet {
            size: 0,
            root: Ref::new(Node::new()),
            hasher: From::from(hasher),
        }
    }

    /// Get a reference to the set's [`BuildHasher`][BuildHasher].
    ///
    /// [BuildHasher]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    #[must_use]
    pub fn hasher(&self) -> &Ref<S> {
        &self.hasher
    }

    /// Construct an empty hash set using the same hasher as the current hash set.
    #[inline]
    #[must_use]
    pub fn new_from<A1>(&self) -> HashSet<A1, S>
    where
        A1: Hash + Eq + Clone,
    {
        HashSet {
            size: 0,
            root: Ref::new(Node::new()),
            hasher: self.hasher.clone(),
        }
    }

    /// Get an iterator over the values in a hash set.
    ///
    /// Please note that the order is consistent between sets using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same set.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, A> {
        Iter {
            it: NodeIter::new(&self.root, self.size),
        }
    }

    /// Get a mutable iterator over the values in a hash set.
    ///
    /// Please note that the order is consistent between sets using the same
    /// hasher, but no other ordering guarantee is offered.  Items will not come
    /// out in insertion order or sort order.  They will, however, come out in
    /// the same order every time for the same set.
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, A> {
        let root = Ref::make_mut(&mut self.root);
        IterMut {
            it: NodeIterMut::new(root, self.size),
        }
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn contains<BA>(&self, a: &BA) -> bool
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        self.root.get(hash_key(&*self.hasher, a), 0, a).is_some()
    }

    /// Insert a value into a set.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn insert(&mut self, a: A) -> Option<A> {
        let hash = hash_key(&*self.hasher, &a);
        let root = Ref::make_mut(&mut self.root);
        match root.insert(hash, 0, Value(a)) {
            None => {
                self.size += 1;
                None
            }
            Some(Value(old_value)) => Some(old_value),
        }
    }

    /// Remove a value from a set if it exists.
    ///
    /// Time: O(log n)
    pub fn remove<BA>(&mut self, a: &BA) -> Option<A>
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        let root = Ref::make_mut(&mut self.root);
        let result = root.remove(hash_key(&*self.hasher, a), 0, a);
        if result.is_some() {
            self.size -= 1;
        }
        result.map(|v| v.0)
    }

    /// Construct a new set from the current set with the given value
    /// added.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let set = hashset![123];
    /// assert_eq!(
    ///   set.update(456),
    ///   hashset![123, 456]
    /// );
    /// # }
    /// ```
    #[must_use]
    pub fn update(&self, a: A) -> Self {
        let mut out = self.clone();
        out.insert(a);
        out
    }

    /// Construct a new set with the given value removed if it's in
    /// the set.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without<BA>(&self, a: &BA) -> Self
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        let mut out = self.clone();
        out.remove(a);
        out
    }

    /// Filter out values from a set which don't satisfy a predicate.
    ///
    /// This is slightly more efficient than filtering using an
    /// iterator, in that it doesn't need to rehash the retained
    /// values, but it still needs to reconstruct the entire tree
    /// structure of the set.
    ///
    /// Time: O(n log n)
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&A) -> bool,
    {
        let old_root = self.root.clone();
        let root = Ref::make_mut(&mut self.root);
        for (value, hash) in NodeIter::new(&old_root, self.size) {
            if !f(value) && root.remove(hash, 0, value).is_some() {
                self.size -= 1;
            }
        }
    }

    /// Construct the union of two sets.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # fn main() {
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{1, 2, 3};
    /// assert_eq!(expected, set1.union(set2));
    /// # }
    /// ```
    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        for value in other {
            self.insert(value);
        }
        self
    }

    /// Construct the union of multiple sets.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
        S: Default,
    {
        i.into_iter().fold(Self::default(), |a, b| a.union(b))
    }

    /// Construct the difference between two sets.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # fn main() {
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{1, 3};
    /// assert_eq!(expected, set1.difference(set2));
    /// # }
    /// ```
    #[must_use]
    pub fn difference(mut self, other: Self) -> Self {
        for value in other {
            if self.remove(&value).is_none() {
                self.insert(value);
            }
        }
        self
    }

    /// Construct the intersection of two sets.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashset::HashSet;
    /// # fn main() {
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{2};
    /// assert_eq!(expected, set1.intersection(set2));
    /// # }
    /// ```
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        let mut out = self.new_from();
        for value in other {
            if self.contains(&value) {
                out.insert(value);
            }
        }
        out
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        let o = other.borrow();
        self.iter().all(|a| o.contains(&a))
    }

    /// Test whether a set is a proper subset of another set, meaning
    /// that all values in our set must also be in the other set. A
    /// proper subset must also be smaller than the other set.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        self.len() != other.borrow().len() && self.is_subset(other)
    }
}

// Core traits

impl<A, S> Clone for HashSet<A, S>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        HashSet {
            hasher: self.hasher.clone(),
            root: self.root.clone(),
            size: self.size,
        }
    }
}

impl<A, S> PartialEq for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn eq(&self, other: &Self) -> bool {
        self.test_eq(other)
    }
}

impl<A, S> Eq for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
}

impl<A, S> PartialOrd for HashSet<A, S>
where
    A: Hash + Eq + Clone + PartialOrd,
    S: BuildHasher + Default,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Ref::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().partial_cmp(other.iter());
        }
        let m1: ::std::collections::HashSet<A> = self.iter().cloned().collect();
        let m2: ::std::collections::HashSet<A> = other.iter().cloned().collect();
        m1.iter().partial_cmp(m2.iter())
    }
}

impl<A, S> Ord for HashSet<A, S>
where
    A: Hash + Eq + Clone + Ord,
    S: BuildHasher + Default,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if Ref::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().cmp(other.iter());
        }
        let m1: ::std::collections::HashSet<A> = self.iter().cloned().collect();
        let m2: ::std::collections::HashSet<A> = other.iter().cloned().collect();
        m1.iter().cmp(m2.iter())
    }
}

impl<A, S> Hash for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A, S> Default for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        HashSet {
            hasher: Ref::<S>::default(),
            root: Ref::new(Node::new()),
            size: 0,
        }
    }
}

impl<A, S> Add for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Output = HashSet<A, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<A, S> Mul for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Output = HashSet<A, S>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<'a, A, S> Add for &'a HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Output = HashSet<A, S>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<'a, A, S> Mul for &'a HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Output = HashSet<A, S>;

    fn mul(self, other: Self) -> Self::Output {
        self.clone().intersection(other.clone())
    }
}

impl<A, S> Sum for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::default(), |a, b| a + b)
    }
}

impl<A, S, R> Extend<R> for HashSet<A, S>
where
    A: Hash + Eq + Clone + From<R>,
    S: BuildHasher,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = R>,
    {
        for value in iter {
            self.insert(From::from(value));
        }
    }
}

#[cfg(not(has_specialisation))]
impl<A, S> Debug for HashSet<A, S>
where
    A: Hash + Eq + Clone + Debug,
    S: BuildHasher + Default,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(has_specialisation)]
impl<A, S> Debug for HashSet<A, S>
where
    A: Hash + Eq + Clone + Debug,
    S: BuildHasher + Default,
{
    default fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(has_specialisation)]
impl<A, S> Debug for HashSet<A, S>
where
    A: Hash + Eq + Clone + Debug + Ord,
    S: BuildHasher + Default,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

// Iterators

// An iterator over the elements of a set.
pub struct Iter<'a, A>
where
    A: 'a,
{
    it: NodeIter<'a, Value<A>>,
}

impl<'a, A> Iterator for Iter<'a, A>
where
    A: 'a + Clone,
{
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(v, _)| &v.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, A> ExactSizeIterator for Iter<'a, A> where A: Clone {}

impl<'a, A> FusedIterator for Iter<'a, A> where A: Clone {}

// A mutable iterator over the elements of a set.
pub struct IterMut<'a, A>
where
    A: 'a,
{
    it: NodeIterMut<'a, Value<A>>,
}

impl<'a, A> Iterator for IterMut<'a, A>
where
    A: 'a + Clone,
{
    type Item = &'a mut A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(v, _)| &mut v.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, A> ExactSizeIterator for IterMut<'a, A> where A: Clone {}

impl<'a, A> FusedIterator for IterMut<'a, A> where A: Clone {}

// A consuming iterator over the elements of a set.
pub struct ConsumingIter<A>
where
    A: Hash + Eq + Clone,
{
    it: NodeDrain<Value<A>>,
}

impl<A> Iterator for ConsumingIter<A>
where
    A: Hash + Eq + Clone,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(v, _)| v.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<A> ExactSizeIterator for ConsumingIter<A> where A: Hash + Eq + Clone {}

impl<A> FusedIterator for ConsumingIter<A> where A: Hash + Eq + Clone {}

// Iterator conversions

impl<A, RA, S> FromIterator<RA> for HashSet<A, S>
where
    A: Hash + Eq + Clone + From<RA>,
    S: BuildHasher + Default,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        let mut set = Self::default();
        for value in i {
            set.insert(From::from(value));
        }
        set
    }
}

impl<'a, A, S> IntoIterator for &'a HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A, S> IntoIterator for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Item = A;
    type IntoIter = ConsumingIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: NodeDrain::new(self.root, self.size),
        }
    }
}

// Conversions

impl<'s, 'a, A, OA, SA, SB> From<&'s HashSet<&'a A, SA>> for HashSet<OA, SB>
where
    A: ToOwned<Owned = OA> + Hash + Eq + ?Sized,
    OA: Borrow<A> + Hash + Eq + Clone,
    SA: BuildHasher,
    SB: BuildHasher + Default,
{
    fn from(set: &HashSet<&A, SA>) -> Self {
        set.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<'a, A, S> From<&'a [A]> for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from(slice: &'a [A]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<A, S> From<Vec<A>> for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A, S> From<&'a Vec<A>> for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A, S> From<collections::HashSet<A>> for HashSet<A, S>
where
    A: Eq + Hash + Clone,
    S: BuildHasher + Default,
{
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A, S> From<&'a collections::HashSet<A>> for HashSet<A, S>
where
    A: Eq + Hash + Clone,
    S: BuildHasher + Default,
{
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<'a, A, S> From<&'a BTreeSet<A>> for HashSet<A, S>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from(btree_set: &BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<A, S> From<OrdSet<A>> for HashSet<A, S>
where
    A: Ord + Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from(ordset: OrdSet<A>) -> Self {
        ordset.into_iter().collect()
    }
}

impl<'a, A, S> From<&'a OrdSet<A>> for HashSet<A, S>
where
    A: Ord + Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from(ordset: &OrdSet<A>) -> Self {
        ordset.into_iter().cloned().collect()
    }
}

// QuickCheck

#[cfg(all(threadsafe, any(test, feature = "quickcheck")))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(threadsafe, any(test, feature = "quickcheck")))]
impl<A, S> Arbitrary for HashSet<A, S>
where
    A: Hash + Eq + Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        HashSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for a hash set of a given size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_set(ref s in hashset(".*", 10..100)) {
    ///         assert!(s.len() < 100);
    ///         assert!(s.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn hash_set<A: Strategy + 'static>(
        element: A,
        size: Range<usize>,
    ) -> BoxedStrategy<HashSet<<A::Tree as ValueTree>::Value>>
    where
        <A::Tree as ValueTree>::Value: Hash + Eq + Clone,
    {
        ::proptest::collection::vec(element, size.clone())
            .prop_map(HashSet::from)
            .prop_filter("HashSet minimum size".to_owned(), move |s| {
                s.len() >= size.start
            }).boxed()
    }
}

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;
    use proptest::num::i16;
    use std::hash::BuildHasherDefault;
    use test::LolHasher;

    #[test]
    fn insert_failing() {
        let mut set: HashSet<i16, BuildHasherDefault<LolHasher>> = Default::default();
        set.insert(14658);
        assert_eq!(1, set.len());
        set.insert(-19198);
        assert_eq!(2, set.len());
    }

    #[test]
    fn match_strings_with_string_slices() {
        let mut set: HashSet<String> = From::from(&hashset!["foo", "bar"]);
        set = set.without("bar");
        assert!(!set.contains("bar"));
        set.remove("foo");
        assert!(!set.contains("foo"));
    }

    #[test]
    fn macro_allows_trailing_comma() {
        let set1 = hashset!{"foo", "bar"};
        let set2 = hashset!{
            "foo",
            "bar",
        };
        assert_eq!(set1, set2);
    }

    proptest! {
        #[test]
        fn proptest_a_set(ref s in hash_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
