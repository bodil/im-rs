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
use std::ops::{Add, Deref, Mul};

use hashset::HashSet;
use nodes::btree::{
    BTreeValue, ConsumingIter as ConsumingNodeIter, DiffItem as NodeDiffItem,
    DiffIter as NodeDiffIter, Insert, Iter as NodeIter, Node, Remove,
};

pub type DiffItem<'a, A> = NodeDiffItem<'a, A>;

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
            l.insert($x);
        )*
            l
    }};
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Value<A>(A);

impl<A> Deref for Value<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// FIXME lacking specialisation, we can't simply implement `BTreeValue`
// for `A`, we have to use the `Value<A>` indirection.
impl<A: Ord + Clone> BTreeValue for Value<A> {
    type Key = A;

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
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
    root: Node<Value<A>>,
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
    /// # fn main() {
    /// let set = OrdSet::singleton(123);
    /// assert!(set.contains(&123));
    /// # }
    /// ```
    pub fn singleton(a: A) -> Self {
        OrdSet {
            root: Node::singleton(Value(a)),
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
    pub fn get_min(&self) -> Option<&A> {
        self.root.min().map(Deref::deref)
    }

    /// Get the largest value in a set.
    ///
    /// If the set is empty, returns `None`.
    pub fn get_max(&self) -> Option<&A> {
        self.root.max().map(Deref::deref)
    }
}

impl<A: Ord + Clone> OrdSet<A> {
    // Create an iterator over the contents of the set.
    pub fn iter(&self) -> Iter<A> {
        Iter {
            it: NodeIter::new(&self.root),
        }
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
    pub fn diff<'a>(&'a self, other: &'a Self) -> DiffIter<A> {
        DiffIter {
            it: NodeDiffIter::new(&self.root, &other.root),
        }
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
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// let set = ordset![456];
    /// assert_eq!(
    ///   set.update(123),
    ///   ordset![123, 456]
    /// );
    /// # }
    /// ```
    pub fn update(&self, a: A) -> Self {
        match self.root.insert(Value(a)) {
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
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// let mut set = ordset!{};
    /// set.insert(123);
    /// set.insert(456);
    /// assert_eq!(
    ///   set,
    ///   ordset![123, 456]
    /// );
    /// # }
    /// ```
    ///
    /// [insert]: #method.insert
    #[inline]
    pub fn insert(&mut self, a: A) {
        match self.root.insert_mut(Value(a)) {
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

    /// Construct a new set with the given value removed if it's in
    /// the set.
    ///
    /// Time: O(log n)
    pub fn without<BA>(&self, a: &BA) -> Self
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
    /// Time: O(log n)
    #[inline]
    pub fn remove<BA>(&mut self, a: &BA) -> Option<A>
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        match self.root.remove_mut(a) {
            Remove::Update(value, root) => {
                self.root = root;
                Some(value.0)
            }
            Remove::Removed(value) => Some(value.0),
            Remove::NoChange => None,
        }
    }

    /// Construct the union of two sets.
    ///
    /// Time: O(n)
    pub fn union<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other
            .borrow()
            .iter()
            .fold(self.clone(), |set, item| set.update(item.clone()))
    }

    /// Construct the union of multiple sets.
    ///
    /// Time: O(n)
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(OrdSet::new(), |a, b| a.union(&b))
    }

    /// Construct the difference between two sets.
    ///
    /// Time: O(n)
    pub fn difference<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other
            .borrow()
            .iter()
            .fold(self.clone(), |set, item| set.without(&item))
    }

    /// Construct the intersection of two sets.
    ///
    /// Time: O(n)
    pub fn intersection<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        other.borrow().iter().fold(OrdSet::new(), |set, item| {
            if self.contains(&item) {
                set.update(item.clone())
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
    ///
    /// Time: O(n)
    pub fn split<BA>(&self, split: &BA) -> (Self, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.iter().fold(
            (OrdSet::new(), OrdSet::new()),
            |(less, greater), item| match (*item).borrow().cmp(split) {
                Ordering::Less => (less.update(item.clone()), greater),
                Ordering::Equal => (less, greater),
                Ordering::Greater => (less, greater.update(item.clone())),
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
    ///
    /// Time: O(n)
    pub fn split_member<BA>(&self, split: &BA) -> (Self, bool, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.iter().fold(
            (OrdSet::new(), false, OrdSet::new()),
            |(less, present, greater), item| match (*item).borrow().cmp(split) {
                Ordering::Less => (less.update(item.clone()), present, greater),
                Ordering::Equal => (less, true, greater),
                Ordering::Greater => (less, present, greater.update(item.clone())),
            },
        )
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
    ///
    /// Time: O(n)
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
    ///
    /// Time: O(n)
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        let o = other.borrow();
        self.len() < o.len() && self.is_subset(o)
    }

    /// Construct a set with only the `n` smallest values from a given
    /// set.
    ///
    /// Time: O(n)
    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).cloned().collect()
    }

    /// Construct a set with the `n` smallest values removed from a
    /// given set.
    ///
    /// Time: O(n)
    pub fn skip(&self, n: usize) -> Self {
        self.iter().skip(n).cloned().collect()
    }

    /// Remove the smallest value from a set, and return that value as
    /// well as the updated set.
    ///
    /// Time: O(log n)
    pub fn without_min(&self) -> (Option<A>, Self) {
        match self.get_min() {
            Some(v) => (Some(v.clone()), self.without(&v)),
            None => (None, self.clone()),
        }
    }

    /// Remove the largest value from a set, and return that value as
    /// well as the updated set.
    ///
    /// Time: O(log n)
    pub fn without_max(&self) -> (Option<A>, Self) {
        match self.get_max() {
            Some(v) => (Some(v.clone()), self.without(&v)),
            None => (None, self.clone()),
        }
    }

    /// Remove the smallest value from a set.
    ///
    /// Time: O(log n)
    pub fn remove_min(&mut self) -> Option<A> {
        // FIXME implement this at the node level for better efficiency
        let key = match self.get_min() {
            None => return None,
            Some(v) => v,
        }.clone();
        self.remove(&key)
    }

    /// Remove the largest value from a set.
    ///
    /// Time: O(log n)
    pub fn remove_max(&mut self) -> Option<A> {
        // FIXME implement this at the node level for better efficiency
        let key = match self.get_max() {
            None => return None,
            Some(v) => v,
        }.clone();
        self.remove(&key)
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

impl<A: Ord + Clone> PartialEq for OrdSet<A> {
    fn eq(&self, other: &Self) -> bool {
        self.root.ptr_eq(&other.root)
            || (self.len() == other.len() && self.diff(other).next().is_none())
    }
}

impl<A: Ord + Eq + Clone> Eq for OrdSet<A> {}

impl<A: Ord + Clone> PartialOrd for OrdSet<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord + Clone> Ord for OrdSet<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Ord + Clone + Hash> Hash for OrdSet<A> {
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

impl<A: Ord + Clone> Add for OrdSet<A> {
    type Output = OrdSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(&other)
    }
}

impl<'a, A: Ord + Clone> Add for &'a OrdSet<A> {
    type Output = OrdSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<A: Ord + Clone> Mul for OrdSet<A> {
    type Output = OrdSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(&other)
    }
}

impl<'a, A: Ord + Clone> Mul for &'a OrdSet<A> {
    type Output = OrdSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A: Ord + Clone> Sum for OrdSet<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, R> Extend<R> for OrdSet<A>
where
    A: Ord + Clone + From<R>,
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

impl<A: Ord + Clone + Debug> Debug for OrdSet<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

// Iterators

pub struct Iter<'a, A>
where
    A: 'a,
{
    it: NodeIter<'a, Value<A>>,
}

impl<'a, A> Iterator for Iter<'a, A>
where
    A: 'a + Ord + Clone,
{
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(Deref::deref)
    }
}

pub struct ConsumingIter<A> {
    it: ConsumingNodeIter<Value<A>>,
}

impl<A> Iterator for ConsumingIter<A>
where
    A: Ord + Clone,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|v| v.0)
    }
}

pub struct DiffIter<'a, A: 'a> {
    it: NodeDiffIter<'a, Value<A>>,
}

impl<'a, A> Iterator for DiffIter<'a, A>
where
    A: 'a + Ord + Clone + PartialEq,
{
    type Item = DiffItem<'a, A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|item| match item {
            NodeDiffItem::Add(v) => NodeDiffItem::Add(v.deref()),
            NodeDiffItem::Update { old, new } => NodeDiffItem::Update {
                old: old.deref(),
                new: new.deref(),
            },
            NodeDiffItem::Remove(v) => NodeDiffItem::Remove(v.deref()),
        })
    }
}

impl<A, R> FromIterator<R> for OrdSet<A>
where
    A: Ord + Clone + From<R>,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = R>,
    {
        let mut out = Self::new();
        for item in i {
            out.insert(From::from(item));
        }
        out
    }
}

impl<'a, A> IntoIterator for &'a OrdSet<A>
where
    A: 'a + Ord + Clone,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> IntoIterator for OrdSet<A>
where
    A: Ord + Clone,
{
    type Item = A;
    type IntoIter = ConsumingIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: ConsumingNodeIter::new(&self.root),
        }
    }
}

// Conversions

impl<'s, 'a, A, OA> From<&'s OrdSet<&'a A>> for OrdSet<OA>
where
    A: ToOwned<Owned = OA> + Ord + Clone + ?Sized,
    OA: Borrow<A> + Ord + Clone,
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

impl<A: Ord + Clone> From<Vec<A>> for OrdSet<A> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a Vec<A>> for OrdSet<A> {
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A: Eq + Hash + Ord + Clone> From<collections::HashSet<A>> for OrdSet<A> {
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Ord + Clone> From<&'a collections::HashSet<A>> for OrdSet<A> {
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<A: Ord + Clone> From<collections::BTreeSet<A>> for OrdSet<A> {
    fn from(btree_set: collections::BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a collections::BTreeSet<A>> for OrdSet<A> {
    fn from(btree_set: &collections::BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq + Ord + Clone, S: BuildHasher> From<HashSet<A, S>> for OrdSet<A> {
    fn from(hashset: HashSet<A, S>) -> Self {
        hashset.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Ord + Clone, S: BuildHasher> From<&'a HashSet<A, S>> for OrdSet<A> {
    fn from(hashset: &HashSet<A, S>) -> Self {
        hashset.into_iter().cloned().collect()
    }
}

// QuickCheck

#[cfg(all(not(feature = "no_arc"), any(test, feature = "quickcheck")))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(not(feature = "no_arc"), any(test, feature = "quickcheck")))]
impl<A: Ord + Clone + Arbitrary + Sync> Arbitrary for OrdSet<A> {
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
    ) -> BoxedStrategy<OrdSet<<A::Tree as ValueTree>::Value>>
    where
        <A::Tree as ValueTree>::Value: Ord + Clone,
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
    // use super::*;

    // #[test]
    // fn match_strings_with_string_slices() {
    //     let set: OrdSet<String> = From::from(&ordset!["foo"]);
    //     assert!(set.contains("foo"));
    // }

    proptest! {
        #[test]
        fn proptest_a_set(ref s in ord_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
