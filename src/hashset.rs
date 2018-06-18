// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A hash set.
//!
//! An immutable hash set.
//!
//! This is implemented as a [`HashMap`][hashmap::HashMap] with no
//! values, so it shares the exact performance characteristics of
//! [`HashMap`][hashmap::HashMap].
//!
//! [hashmap::HashMap]: ../hashmap/struct.HashMap.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::RandomState;
use std::collections::{self, BTreeSet};
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, IntoIterator, Sum};
use std::ops::{Add, Mul};
use std::sync::Arc;

use bits::hash_key;
use nodes::hamt::{HashValue, Iter, Node};
use ordset::OrdSet;
use shared::Shared;

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
            l.insert_mut($x);
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::hashset::HashSet::new();
        $(
            l.insert_mut($x);
        )*
            l
    }};
}

/// A hash set.
///
/// An immutable hash set.
///
/// This is implemented as a [`HashMap`][hashmap::HashMap] with no
/// values, so it shares the exact performance characteristics of
/// [`HashMap`][hashmap::HashMap].
///
/// [hashmap::HashMap]: ../hashmap/struct.HashMap.html
pub struct HashSet<A, S = RandomState> {
    hasher: Arc<S>,
    root: Arc<Node<Arc<A>>>,
    size: usize,
}

impl<A: Hash + Eq> HashValue for Arc<A> {
    type Key = A;

    fn extract_key(&self) -> &Self::Key {
        &*self
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other)
    }
}

impl<A> HashSet<A, RandomState>
where
    A: Hash + Eq,
{
    /// Construct an empty set.
    pub fn new() -> Self {
        Default::default()
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
    pub fn singleton<R>(a: R) -> Self
    where
        R: Shared<A>,
    {
        HashSet::new().insert(a)
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
    pub fn len(&self) -> usize {
        self.size
    }
}

impl<A, S> HashSet<A, S>
where
    A: Hash + Eq,
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
    pub fn with_hasher<RS>(hasher: RS) -> Self
    where
        RS: Shared<S>,
    {
        HashSet {
            size: 0,
            root: Arc::new(Node::new()),
            hasher: hasher.shared(),
        }
    }

    /// Construct an empty hash set using the same hasher as the current hash set.
    #[inline]
    pub fn new_from<A1>(&self) -> HashSet<A1, S>
    where
        A1: Hash + Eq,
    {
        HashSet {
            size: 0,
            root: Arc::new(Node::new()),
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
    pub fn iter(&self) -> Iter<Arc<A>> {
        Node::iter(self.root.clone(), self.size)
    }

    /// Insert a value into a set.
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
    ///   set.insert(456),
    ///   hashset![123, 456]
    /// );
    /// # }
    /// ```
    pub fn insert<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        self.insert_ref(a.shared())
    }

    fn insert_ref(&self, a: Arc<A>) -> Self {
        let (added, new_node) = self.root.insert(hash_key(&*self.hasher, &a), 0, a);
        HashSet {
            root: Arc::new(new_node),
            size: if added {
                self.size + 1
            } else {
                self.size
            },
            hasher: self.hasher.clone(),
        }
    }

    /// Insert a value into a set.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn insert_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        self.insert_mut_ref(a.shared())
    }

    fn insert_mut_ref(&mut self, a: Arc<A>) {
        let hash = hash_key(&*self.hasher, &a);
        let root = Arc::make_mut(&mut self.root);
        let added = root.insert_mut(hash, 0, a);
        if added {
            self.size += 1
        }
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    pub fn contains<BA>(&self, a: &BA) -> bool
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        self.root.get(hash_key(&*self.hasher, a), 0, a).is_some()
    }

    /// Remove a value from a set if it exists.
    ///
    /// Time: O(log n)
    pub fn remove<BA>(&self, a: &BA) -> Self
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        self.root
            .remove(hash_key(&*self.hasher, a), 0, a)
            .map(|(_, node)| HashSet {
                hasher: self.hasher.clone(),
                size: self.size - 1,
                root: Arc::new(node),
            })
            .unwrap_or_else(|| self.clone())
    }

    /// Remove a value from a set if it exists.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn remove_mut<BA>(&mut self, a: &BA)
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        let root = Arc::make_mut(&mut self.root);
        let result = root.remove_mut(hash_key(&*self.hasher, a), 0, a);
        if result.is_some() {
            self.size -= 1;
        }
    }

    /// Construct the union of two sets.
    pub fn union<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other
            .borrow()
            .iter()
            .fold(self.clone(), |set, a| set.insert(a.clone()))
    }

    /// Construct the union of multiple sets.
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
        S: Default,
    {
        i.into_iter().fold(Default::default(), |a, b| a.union(&b))
    }

    /// Construct the difference between two sets.
    pub fn difference<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other
            .borrow()
            .iter()
            .fold(self.clone(), |set, a| set.remove(&a))
    }

    /// Construct the intersection of two sets.
    pub fn intersection<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other.borrow().iter().fold(self.new_from(), |set, a| {
            if self.contains(&a) {
                set.insert(a)
            } else {
                set
            }
        })
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
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
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        self.len() != other.borrow().len() && self.is_subset(other)
    }
}

// Core traits

impl<A, S> Clone for HashSet<A, S> {
    fn clone(&self) -> Self {
        HashSet {
            hasher: self.hasher.clone(),
            root: self.root.clone(),
            size: self.size,
        }
    }
}

impl<A: Hash + Eq, S: BuildHasher + Default> PartialEq for HashSet<A, S> {
    fn eq(&self, other: &Self) -> bool {
        self.test_eq(other)
    }
}

impl<A: Hash + Eq, S: BuildHasher + Default> Eq for HashSet<A, S> {}

impl<A: Hash + Eq + PartialOrd, S: BuildHasher + Default> PartialOrd for HashSet<A, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().partial_cmp(other.iter());
        }
        let m1: ::std::collections::HashSet<Arc<A>> = self.iter().collect();
        let m2: ::std::collections::HashSet<Arc<A>> = other.iter().collect();
        m1.iter().partial_cmp(m2.iter())
    }
}

impl<A: Hash + Eq + Ord, S: BuildHasher + Default> Ord for HashSet<A, S> {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().cmp(other.iter());
        }
        let m1: ::std::collections::HashSet<Arc<A>> = self.iter().collect();
        let m2: ::std::collections::HashSet<Arc<A>> = other.iter().collect();
        m1.iter().cmp(m2.iter())
    }
}

impl<A: Hash + Eq, S: BuildHasher + Default> Hash for HashSet<A, S> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A: Hash + Eq, S: BuildHasher + Default> Default for HashSet<A, S> {
    fn default() -> Self {
        HashSet {
            hasher: Default::default(),
            root: Arc::new(Node::new()),
            size: 0,
        }
    }
}

impl<A: Hash + Eq, S: BuildHasher> Add for HashSet<A, S> {
    type Output = HashSet<A, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(&other)
    }
}

impl<A: Hash + Eq, S: BuildHasher> Mul for HashSet<A, S> {
    type Output = HashSet<A, S>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(&other)
    }
}

impl<'a, A: Hash + Eq, S: BuildHasher> Add for &'a HashSet<A, S> {
    type Output = HashSet<A, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, A: Hash + Eq, S: BuildHasher> Mul for &'a HashSet<A, S> {
    type Output = HashSet<A, S>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A, S> Sum for HashSet<A, S>
where
    A: Hash + Eq,
    S: BuildHasher + Default,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Default::default(), |a, b| a + b)
    }
}

impl<A, S, R> Extend<R> for HashSet<A, S>
where
    A: Hash + Eq,
    S: BuildHasher,
    R: Shared<A>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = R>,
    {
        for value in iter {
            self.insert_mut(value);
        }
    }
}

#[cfg(not(has_specialisation))]
impl<A: Hash + Eq + Debug, S: BuildHasher + Default> Debug for HashSet<A, S> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(has_specialisation)]
impl<A: Hash + Eq + Debug, S: BuildHasher + Default> Debug for HashSet<A, S> {
    default fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(has_specialisation)]
impl<A: Hash + Eq + Debug + Ord, S: BuildHasher + Default> Debug for HashSet<A, S> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

// Iterators

impl<A: Hash + Eq, RA, S> FromIterator<RA> for HashSet<A, S>
where
    RA: Shared<A>,
    S: BuildHasher + Default,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        let mut set: Self = Default::default();
        for value in i {
            set.insert_mut(value)
        }
        set
    }
}

impl<'a, A: Hash + Eq, S: BuildHasher> IntoIterator for &'a HashSet<A, S> {
    type Item = Arc<A>;
    type IntoIter = Iter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A: Hash + Eq, S: BuildHasher> IntoIterator for HashSet<A, S> {
    type Item = Arc<A>;
    type IntoIter = Iter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<'s, 'a, A, OA, SA, SB> From<&'s HashSet<&'a A, SA>> for HashSet<OA, SB>
where
    A: ToOwned<Owned = OA> + Hash + Eq + ?Sized,
    OA: Borrow<A> + Hash + Eq,
    SA: BuildHasher,
    SB: BuildHasher + Default,
{
    fn from(set: &HashSet<&A, SA>) -> Self {
        set.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<'a, A: Hash + Eq + Clone, S: BuildHasher + Default> From<&'a [A]> for HashSet<A, S> {
    fn from(slice: &'a [A]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq, S: BuildHasher + Default> From<&'a [Arc<A>]> for HashSet<A, S> {
    fn from(slice: &'a [Arc<A>]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq, S: BuildHasher + Default> From<Vec<A>> for HashSet<A, S> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Clone, S: BuildHasher + Default> From<&'a Vec<A>> for HashSet<A, S> {
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq, S: BuildHasher + Default> From<&'a Vec<Arc<A>>> for HashSet<A, S> {
    fn from(vec: &Vec<Arc<A>>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A: Eq + Hash, S: BuildHasher + Default> From<collections::HashSet<A>> for HashSet<A, S> {
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Clone, S: BuildHasher + Default> From<&'a collections::HashSet<A>>
    for HashSet<A, S>
{
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Eq + Hash, S: BuildHasher + Default> From<&'a collections::HashSet<Arc<A>>>
    for HashSet<A, S>
{
    fn from(hash_set: &collections::HashSet<Arc<A>>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq, S: BuildHasher + Default> From<BTreeSet<A>> for HashSet<A, S> {
    fn from(btree_set: BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Clone, S: BuildHasher + Default> From<&'a BTreeSet<A>> for HashSet<A, S> {
    fn from(btree_set: &BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq, S: BuildHasher + Default> From<&'a BTreeSet<Arc<A>>> for HashSet<A, S> {
    fn from(btree_set: &BTreeSet<Arc<A>>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<A: Ord + Hash + Eq, S: BuildHasher + Default> From<OrdSet<A>> for HashSet<A, S> {
    fn from(ordset: OrdSet<A>) -> Self {
        ordset.into_iter().collect()
    }
}

impl<'a, A: Ord + Hash + Eq, S: BuildHasher + Default> From<&'a OrdSet<A>> for HashSet<A, S> {
    fn from(ordset: &OrdSet<A>) -> Self {
        ordset.into_iter().collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Hash + Eq + Arbitrary + Sync> Arbitrary for HashSet<A> {
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
    ) -> BoxedStrategy<HashSet<<A::Value as ValueTree>::Value>>
    where
        <A::Value as ValueTree>::Value: Hash + Eq,
    {
        ::proptest::collection::vec(element, size.clone())
            .prop_map(HashSet::from)
            .prop_filter("HashSet minimum size".to_owned(), move |s| {
                s.len() >= size.start
            })
            .boxed()
    }
}

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;

    #[test]
    fn match_strings_with_string_slices() {
        let mut set: HashSet<String> = From::from(&hashset!["foo", "bar"]);
        set = set.remove("bar");
        assert!(!set.contains("bar"));
        set.remove_mut("foo");
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
