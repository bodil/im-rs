// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An ordered set.
//!
//! This is implemented as an [`OrdMap`][ordmap::OrdMap] with no
//! values, so it shares the exact performance characteristics of
//! [`OrdMap`][ordmap::OrdMap].
//!
//! [ordmap::OrdMap]: ../ordmap/struct.OrdMap.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, IntoIterator, Sum};
use std::ops::{Add, Mul};
use std::sync::Arc;

use hashset::HashSet;
use nodes::btree::{BTreeValue, DiffIter, Insert, Iter, Node, Remove};
use shared::Shared;

/// Construct a set from a sequence of values.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::ordset::OrdSet;
/// # fn main() {
/// assert_eq!(
///   ordset![1, 2, 3],
///   OrdSet::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! ordset {
    () => { $crate::ordset::OrdSet::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::ordset::OrdSet::new();
        $(
            l = l.insert($x);
        )*
            l
    }};
}

impl<A: Ord> BTreeValue for Arc<A> {
    type Key = A;

    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other)
    }

    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        slice.binary_search_by(|value| Self::Key::borrow(value).cmp(key))
    }

    fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        slice.binary_search_by(|value| value.cmp(key))
    }

    fn cmp_keys(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

/// # Ordered Set
///
/// This is implemented as an [`OrdMap`][ordmap::OrdMap] with no
/// values, so it shares the exact performance characteristics of
/// [`OrdMap`][ordmap::OrdMap].
///
/// [ordmap::OrdMap]: ../ordmap/struct.OrdMap.html
pub struct OrdSet<A> {
    root: Node<Arc<A>>,
}

impl<A> OrdSet<A> {
    /// Construct an empty set.
    pub fn new() -> Self {
        OrdSet { root: Node::new() }
    }

    /// Construct a set with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordset::OrdSet;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let set = OrdSet::singleton(123);
    /// assert!(set.contains(&123));
    /// # }
    /// ```
    pub fn singleton<R>(a: R) -> Self
    where
        R: Shared<A>,
    {
        OrdSet {
            root: Node::singleton(a.shared()),
        }
    }

    /// Test whether a set is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// assert!(
    ///   !ordset![1, 2, 3].is_empty()
    /// );
    /// assert!(
    ///   OrdSet::<i32>::new().is_empty()
    /// );
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.root.len() == 0
    }

    /// Get the size of a set.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// assert_eq!(3, ordset![1, 2, 3].len());
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Get the smallest value in a set.
    ///
    /// If the set is empty, returns `None`.
    pub fn get_min(&self) -> Option<Arc<A>> {
        self.root.min().cloned()
    }

    /// Get the largest value in a set.
    ///
    /// If the set is empty, returns `None`.
    pub fn get_max(&self) -> Option<Arc<A>> {
        self.root.max().cloned()
    }
}

impl<A: Ord> OrdSet<A> {
    // Create an iterator over the contents of the set.
    pub fn iter(&self) -> Iter<Arc<A>> {
        Iter::new(&self.root)
    }

    /// Get an iterator over the differences between this set and
    /// another, i.e. the set of entries to add or remove to this set
    /// in order to make it equal to the other set.
    ///
    /// This function will avoid visiting nodes which are shared
    /// between the two sets, meaning that even very large sets can be
    /// compared quickly if most of their structure is shared.
    ///
    /// Time: O(n) (where n is the number of unique elements across
    /// the two sets, minus the number of elements belonging to nodes
    /// shared between them)
    pub fn diff<RS: Borrow<Self>>(&self, other: RS) -> DiffIter<Arc<A>> {
        DiffIter::new(&self.root, &other.borrow().root)
    }

    /// Insert a value into a set.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordset::OrdSet;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let set = ordset![456];
    /// assert_eq!(
    ///   set.insert(123),
    ///   ordset![123, 456]
    /// );
    /// # }
    /// ```
    pub fn insert<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        match self.root.insert(a.shared()) {
            Insert::NoChange => self.clone(),
            Insert::JustInc => unreachable!(),
            Insert::Update(root) => OrdSet { root },
            Insert::Split(left, median, right) => OrdSet {
                root: Node::from_split(left, median, right),
            },
        }
    }

    /// Insert a value into a set.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordset::OrdSet;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let mut set = ordset!{};
    /// set.insert_mut(123);
    /// set.insert_mut(456);
    /// assert_eq!(
    ///   set,
    ///   ordset![123, 456]
    /// );
    /// # }
    /// ```
    ///
    /// [insert]: #method.insert
    #[inline]
    pub fn insert_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        match self.root.insert_mut(a.shared()) {
            Insert::NoChange | Insert::JustInc => {}
            Insert::Update(root) => self.root = root,
            Insert::Split(left, median, right) => self.root = Node::from_split(left, median, right),
        }
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    pub fn contains<BA>(&self, a: &BA) -> bool
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.root.lookup(a).is_some()
    }

    /// Remove a value from a set.
    ///
    /// Time: O(log n)
    pub fn remove<BA>(&self, a: &BA) -> Self
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        match self.root.remove(a) {
            Remove::NoChange => self.clone(),
            Remove::Removed(_) => unreachable!(),
            Remove::Update(_, root) => OrdSet { root },
        }
    }

    /// Remove a value from a set.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn remove_mut<BA>(&mut self, a: &BA)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        if let Remove::Update(_, root) = self.root.remove_mut(a) {
            self.root = root;
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
            .fold(self.clone(), |set, item| set.insert(item))
    }

    /// Construct the union of multiple sets.
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(OrdSet::new(), |a, b| a.union(&b))
    }

    /// Construct the difference between two sets.
    pub fn difference<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other
            .borrow()
            .iter()
            .fold(self.clone(), |set, item| set.remove(&item))
    }

    /// Construct the intersection of two sets.
    pub fn intersection<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other.borrow().iter().fold(OrdSet::new(), |set, item| {
            if self.contains(&item) {
                set.insert(item)
            } else {
                set
            }
        })
    }

    /// Split a set into two, with the left hand set containing values
    /// which are smaller than `split`, and the right hand set
    /// containing values which are larger than `split`.
    ///
    /// The `split` value itself is discarded.
    pub fn split<BA>(&self, split: &BA) -> (Self, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.iter().fold(
            (OrdSet::new(), OrdSet::new()),
            |(less, greater), item| match (*item).borrow().cmp(split) {
                Ordering::Less => (less.insert(item), greater),
                Ordering::Equal => (less, greater),
                Ordering::Greater => (less, greater.insert(item)),
            },
        )
    }

    /// Split a set into two, with the left hand set containing values
    /// which are smaller than `split`, and the right hand set
    /// containing values which are larger than `split`.
    ///
    /// Returns a tuple of the two sets and a boolean which is true if
    /// the `split` value existed in the original set, and false
    /// otherwise.
    pub fn split_member<BA>(&self, split: &BA) -> (Self, bool, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.iter().fold(
            (OrdSet::new(), false, OrdSet::new()),
            |(less, present, greater), item| match (*item).borrow().cmp(split) {
                Ordering::Less => (less.insert(item), present, greater),
                Ordering::Equal => (less, true, greater),
                Ordering::Greater => (less, present, greater.insert(item)),
            },
        )
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
    pub fn is_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        let o = other.borrow();
        self.iter().all(|v| o.contains(&v))
    }

    /// Test whether a set is a proper subset of another set, meaning
    /// that all values in our set must also be in the other set. A
    /// proper subset must also be smaller than the other set.
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        let o = other.borrow();
        self.len() < o.len() && self.is_subset(o)
    }

    /// Construct a set with only the `n` smallest values from a given
    /// set.
    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).collect()
    }

    /// Construct a set with the `n` smallest values removed from a
    /// given set.
    pub fn skip(&self, n: usize) -> Self {
        self.iter().skip(n).collect()
    }

    /// Remove the smallest value from a set, and return that value as
    /// well as the updated set.
    pub fn pop_min(&self) -> (Option<Arc<A>>, Self) {
        match self.get_min() {
            Some(v) => (Some(v.clone()), self.remove(&v)),
            None => (None, self.clone()),
        }
    }

    /// Remove the largest value from a set, and return that value as
    /// well as the updated set.
    pub fn pop_max(&self) -> (Option<Arc<A>>, Self) {
        match self.get_max() {
            Some(v) => (Some(v.clone()), self.remove(&v)),
            None => (None, self.clone()),
        }
    }

    /// Discard the smallest value from a set, returning the updated
    /// set.
    pub fn remove_min(&self) -> Self {
        self.pop_min().1
    }

    /// Discard the largest value from a set, returning the updated
    /// set.
    pub fn remove_max(&self) -> Self {
        self.pop_max().1
    }
}

// Core traits

impl<A> Clone for OrdSet<A> {
    fn clone(&self) -> Self {
        OrdSet {
            root: self.root.clone(),
        }
    }
}

impl<A: Ord> PartialEq for OrdSet<A> {
    fn eq(&self, other: &Self) -> bool {
        self.root.ptr_eq(&other.root)
            || (self.len() == other.len() && self.diff(other).next().is_none())
    }
}

impl<A: Ord + Eq> Eq for OrdSet<A> {}

impl<A: Ord> PartialOrd for OrdSet<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord> Ord for OrdSet<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Ord + Hash> Hash for OrdSet<A> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A> Default for OrdSet<A> {
    fn default() -> Self {
        OrdSet::new()
    }
}

impl<A: Ord> Add for OrdSet<A> {
    type Output = OrdSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(&other)
    }
}

impl<'a, A: Ord> Add for &'a OrdSet<A> {
    type Output = OrdSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<A: Ord> Mul for OrdSet<A> {
    type Output = OrdSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(&other)
    }
}

impl<'a, A: Ord> Mul for &'a OrdSet<A> {
    type Output = OrdSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A: Ord> Sum for OrdSet<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, R> Extend<R> for OrdSet<A>
where
    A: Ord,
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

impl<A: Ord + Debug> Debug for OrdSet<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

// Iterators

impl<A: Ord, RA> FromIterator<RA> for OrdSet<A>
where
    RA: Shared<A>,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        i.into_iter().fold(ordset![], |s, a| s.insert(a))
    }
}

impl<'a, A> IntoIterator for &'a OrdSet<A>
where
    A: Ord,
{
    type Item = Arc<A>;
    type IntoIter = Iter<Arc<A>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> IntoIterator for OrdSet<A>
where
    A: Ord,
{
    type Item = Arc<A>;
    type IntoIter = Iter<Arc<A>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<'s, 'a, A, OA> From<&'s OrdSet<&'a A>> for OrdSet<OA>
where
    A: ToOwned<Owned = OA> + Ord + ?Sized,
    OA: Borrow<A> + Ord,
{
    fn from(set: &OrdSet<&A>) -> Self {
        set.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a [A]> for OrdSet<A> {
    fn from(slice: &'a [A]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<'a, A: Ord> From<&'a [Arc<A>]> for OrdSet<A> {
    fn from(slice: &'a [Arc<A>]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<A: Ord> From<Vec<A>> for OrdSet<A> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a Vec<A>> for OrdSet<A> {
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<'a, A: Ord> From<&'a Vec<Arc<A>>> for OrdSet<A> {
    fn from(vec: &Vec<Arc<A>>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A: Eq + Hash + Ord> From<collections::HashSet<A>> for OrdSet<A> {
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Ord + Clone> From<&'a collections::HashSet<A>> for OrdSet<A> {
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Eq + Hash + Ord> From<&'a collections::HashSet<Arc<A>>> for OrdSet<A> {
    fn from(hash_set: &collections::HashSet<Arc<A>>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<A: Ord> From<collections::BTreeSet<A>> for OrdSet<A> {
    fn from(btree_set: collections::BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a collections::BTreeSet<A>> for OrdSet<A> {
    fn from(btree_set: &collections::BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Ord> From<&'a collections::BTreeSet<Arc<A>>> for OrdSet<A> {
    fn from(btree_set: &collections::BTreeSet<Arc<A>>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq + Ord, S: BuildHasher> From<HashSet<A, S>> for OrdSet<A> {
    fn from(hashset: HashSet<A, S>) -> Self {
        hashset.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Ord, S: BuildHasher> From<&'a HashSet<A, S>> for OrdSet<A> {
    fn from(hashset: &HashSet<A, S>) -> Self {
        hashset.into_iter().collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Ord + Arbitrary + Sync> Arbitrary for OrdSet<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        OrdSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for a set of a given size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_set(ref s in set(".*", 10..100)) {
    ///         assert!(s.len() < 100);
    ///         assert!(s.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn ord_set<A: Strategy + 'static>(
        element: A,
        size: Range<usize>,
    ) -> BoxedStrategy<OrdSet<<A::Value as ValueTree>::Value>>
    where
        <A::Value as ValueTree>::Value: Ord,
    {
        ::proptest::collection::vec(element, size.clone())
            .prop_map(OrdSet::from)
            .prop_filter("OrdSet minimum size".to_owned(), move |s| {
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
        let set: OrdSet<String> = From::from(&ordset!["foo"]);
        assert!(set.contains("foo"));
    }

    proptest! {
        #[test]
        fn proptest_a_set(ref s in ord_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
