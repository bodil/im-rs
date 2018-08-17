// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An ordered set.
//!
//! An immutable ordered set implemented as a [B-tree] [1].
//!
//! Most operations on this type of set are O(log n). A
//! [`HashSet`][hashset::HashSet] is usually a better choice for
//! performance, but the `OrdSet` has the advantage of only requiring
//! an [`Ord`][std::cmp::Ord] constraint on its values, and of being
//! ordered, so values always come out from lowest to highest, where a
//! [`HashSet`][hashset::HashSet] has no guaranteed ordering.
//!
//! [1]: https://en.wikipedia.org/wiki/B-tree
//! [hashset::HashSet]: ../hashset/struct.HashSet.html
//! [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html

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
#[cfg(has_specialisation)]
use util::linear_search_by;
use util::Ref;

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
#[cfg(not(has_specialisation))]
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

    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        Self::Key::borrow(self).cmp(other)
    }

    fn cmp_values(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

#[cfg(has_specialisation)]
impl<A: Ord + Clone> BTreeValue for Value<A> {
    type Key = A;

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }

    default fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        slice.binary_search_by(|value| Self::Key::borrow(value).cmp(key))
    }

    default fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        slice.binary_search_by(|value| value.cmp(key))
    }

    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        Self::Key::borrow(self).cmp(other)
    }

    fn cmp_values(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

#[cfg(has_specialisation)]
impl<A: Ord + Clone + Copy> BTreeValue for Value<A> {
    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        linear_search_by(slice, |value| Self::Key::borrow(value).cmp(key))
    }

    fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        linear_search_by(slice, |value| value.cmp(key))
    }
}

/// An ordered set.
///
/// An immutable ordered set implemented as a [B-tree] [1].
///
/// Most operations on this type of set are O(log n). A
/// [`HashSet`][hashset::HashSet] is usually a better choice for
/// performance, but the `OrdSet` has the advantage of only requiring
/// an [`Ord`][std::cmp::Ord] constraint on its values, and of being
/// ordered, so values always come out from lowest to highest, where a
/// [`HashSet`][hashset::HashSet] has no guaranteed ordering.
///
/// [1]: https://en.wikipedia.org/wiki/B-tree
/// [hashset::HashSet]: ../hashset/struct.HashSet.html
/// [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html
pub struct OrdSet<A> {
    root: Ref<Node<Value<A>>>,
}

impl<A> OrdSet<A>
where
    A: Ord + Clone,
{
    /// Construct an empty set.
    #[must_use]
    pub fn new() -> Self {
        OrdSet {
            root: Ref::from(Node::new()),
        }
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
    #[must_use]
    pub fn singleton(a: A) -> Self {
        OrdSet {
            root: Ref::from(Node::unit(Value(a))),
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
    #[inline]
    #[must_use]
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
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Get the smallest value in a set.
    ///
    /// If the set is empty, returns `None`.
    #[must_use]
    pub fn get_min(&self) -> Option<&A> {
        self.root.min().map(Deref::deref)
    }

    /// Get the largest value in a set.
    ///
    /// If the set is empty, returns `None`.
    #[must_use]
    pub fn get_max(&self) -> Option<&A> {
        self.root.max().map(Deref::deref)
    }

    // Create an iterator over the contents of the set.
    #[must_use]
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
    #[must_use]
    pub fn diff<'a>(&'a self, other: &'a Self) -> DiffIter<A> {
        DiffIter {
            it: NodeDiffIter::new(&self.root, &other.root),
        }
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    #[inline]
    #[must_use]
    pub fn contains<BA>(&self, a: &BA) -> bool
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.root.lookup(a).is_some()
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
    pub fn insert(&mut self, a: A) -> Option<A> {
        let new_root = {
            let root = Ref::make_mut(&mut self.root);
            match root.insert(Value(a)) {
                Insert::Replaced(Value(old_value)) => return Some(old_value),
                Insert::Added => return None,
                Insert::Update(root) => Ref::from(root),
                Insert::Split(left, median, right) => {
                    Ref::from(Node::from_split(left, median, right))
                }
            }
        };
        self.root = new_root;
        None
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
        let (new_root, removed_value) = {
            let root = Ref::make_mut(&mut self.root);
            match root.remove(a) {
                Remove::Update(value, root) => (Ref::from(root), Some(value.0)),
                Remove::Removed(value) => return Some(value.0),
                Remove::NoChange => return None,
            }
        };
        self.root = new_root;
        removed_value
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
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let mut out = self.clone();
        out.remove(a);
        out
    }

    /// Remove the smallest value from a set, and return that value as
    /// well as the updated set.
    ///
    /// Time: O(log n)
    #[must_use]
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
    #[must_use]
    pub fn without_max(&self) -> (Option<A>, Self) {
        match self.get_max() {
            Some(v) => (Some(v.clone()), self.without(&v)),
            None => (None, self.clone()),
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
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{1, 2, 3};
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
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{1, 3};
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
    /// # use im::ordset::OrdSet;
    /// # fn main() {
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{2};
    /// assert_eq!(expected, set1.intersection(set2));
    /// # }
    /// ```
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        let mut out = Self::default();
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

    /// Split a set into two, with the left hand set containing values
    /// which are smaller than `split`, and the right hand set
    /// containing values which are larger than `split`.
    ///
    /// The `split` value itself is discarded.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn split<BA>(self, split: &BA) -> (Self, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let (left, _, right) = self.split_member(split);
        (left, right)
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
    #[must_use]
    pub fn split_member<BA>(self, split: &BA) -> (Self, bool, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let mut left = Self::default();
        let mut right = Self::default();
        let mut present = false;
        for value in self {
            match value.borrow().cmp(split) {
                Ordering::Less => {
                    left.insert(value);
                }
                Ordering::Equal => {
                    present = true;
                }
                Ordering::Greater => {
                    right.insert(value);
                }
            }
        }
        (left, present, right)
    }

    /// Construct a set with only the `n` smallest values from a given
    /// set.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).cloned().collect()
    }

    /// Construct a set with the `n` smallest values removed from a
    /// given set.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn skip(&self, n: usize) -> Self {
        self.iter().skip(n).cloned().collect()
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
        Ref::ptr_eq(&self.root, &other.root)
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

impl<A> Default for OrdSet<A>
where
    A: Ord + Clone,
{
    fn default() -> Self {
        OrdSet::new()
    }
}

impl<A: Ord + Clone> Add for OrdSet<A> {
    type Output = OrdSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, A: Ord + Clone> Add for &'a OrdSet<A> {
    type Output = OrdSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<A: Ord + Clone> Mul for OrdSet<A> {
    type Output = OrdSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<'a, A: Ord + Clone> Mul for &'a OrdSet<A> {
    type Output = OrdSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.clone().intersection(other.clone())
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

// An iterator over the elements of a set.
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

// A consuming iterator over the elements of a set.
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

// An iterator over the difference between two sets.
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
    A: ToOwned<Owned = OA> + Ord + ?Sized,
    OA: Borrow<A> + Ord + Clone,
{
    fn from(set: &OrdSet<&A>) -> Self {
        set.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<'a, A> From<&'a [A]> for OrdSet<A>
where
    A: Ord + Clone,
{
    fn from(slice: &'a [A]) -> Self {
        slice.iter().cloned().collect()
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

#[cfg(all(threadsafe, any(test, feature = "quickcheck")))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(threadsafe, any(test, feature = "quickcheck")))]
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
            }).boxed()
    }
}

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;

    #[test]
    fn match_strings_with_string_slices() {
        let mut set: OrdSet<String> = From::from(&ordset!["foo", "bar"]);
        set = set.without("bar");
        assert!(!set.contains("bar"));
        set.remove("foo");
        assert!(!set.contains("foo"));
    }

    proptest! {
        #[test]
        fn proptest_a_set(ref s in ord_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
