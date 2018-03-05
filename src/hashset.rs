//! A hash set.
//!
//! An immutable hash set backed by a [`HashMap`][hashmap::HashMap].
//!
//! This is implemented as a [`HashMap`][hashmap::HashMap] with no values, so it shares
//! the exact performance characteristics of [`HashMap`][hashmap::HashMap].
//!
//! [hashmap::HashMap]: ../hashmap/struct.HashMap.html

use std::sync::Arc;
use std::iter::{FromIterator, IntoIterator};
use std::fmt::{Debug, Error, Formatter};
use std::collections::{self, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::RandomState;

use hashmap::{self, HashMap};
use shared::Shared;
use hash::SharedHasher;
use ordset::OrdSet;

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
}

/// A hash set.
///
/// An immutable hash set backed by a [`HashMap`][hashmap::HashMap].
///
/// This is implemented as a [`HashMap`][hashmap::HashMap] with no values, so it shares
/// the exact performance characteristics of [`HashMap`][hashmap::HashMap].
///
/// [hashmap::HashMap]: ../hashmap/struct.HashMap.html
pub struct HashSet<A, S = RandomState>(HashMap<A, (), S>);

impl<A> HashSet<A, RandomState>
where
    A: Hash + Eq,
{
    /// Construct an empty set.
    pub fn new() -> Self {
        HashSet(HashMap::new())
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
        HashSet(HashMap::<A, ()>::singleton(a, ()))
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
        self.0.is_empty()
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
        self.0.len()
    }

    pub fn iter(&self) -> Iter<A> {
        Iter { it: self.0.iter() }
    }
}

impl<A, S> HashSet<A, S>
where
    A: Hash + Eq,
    S: SharedHasher,
{
    /// Construct an empty hash set using the provided hasher.
    #[inline]
    pub fn with_hasher(hasher: &Arc<S>) -> Self {
        HashSet(HashMap::with_hasher(hasher))
    }

    /// Construct an empty hash set using the same hasher as the current hash set.
    #[inline]
    pub fn new_from<A1>(&self) -> HashSet<A1, S>
    where
        A1: Hash + Eq,
    {
        HashSet(self.0.new_from())
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
        HashSet(self.0.insert(a, ()))
    }

    /// Insert a value into a set, mutating it in place when it is
    /// safe to do so.
    ///
    /// If you are the sole owner of the set, it is safe to mutate it without
    /// losing immutability guarantees, gaining us a considerable performance
    /// advantage. If the set is in use elsewhere, this operation will safely
    /// clone the map before mutating it, acting just like the immutable `insert`
    /// operation.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn insert_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        self.0.insert_mut(a, ())
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    pub fn contains(&self, a: &A) -> bool {
        self.0.contains_key(a)
    }

    /// Remove a value from a set.
    pub fn remove(&self, a: &A) -> Self {
        HashSet(self.0.remove(a))
    }

    /// Remove a value from a set if it exists, mutating it in place
    /// when it is safe to do so.
    ///
    /// If you are the sole owner of the set, it is safe to mutate it without
    /// losing immutability guarantees, gaining us a considerable performance
    /// advantage. If the set is in use elsewhere, this operation will safely
    /// clone the map before mutating it, acting just like the immutable `insert`
    /// operation.
    ///
    /// Time: O(log n)
    pub fn remove_mut(&mut self, a: &A) {
        self.0.remove_mut(a)
    }

    /// Construct the union of two sets.
    pub fn union(&self, other: &Self) -> Self {
        HashSet(self.0.union(&other.0))
    }

    /// Construct the union of multiple sets.
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(Default::default(), |a, b| a.union(&b))
    }

    /// Construct the difference between two sets.
    pub fn difference<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        HashSet(self.0.difference(&other.borrow().0))
    }

    /// Construct the intersection of two sets.
    pub fn intersection<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        HashSet(self.0.intersection(&other.borrow().0))
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
    pub fn is_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        self.0.is_submap(&other.borrow().0)
    }

    /// Test whether a set is a proper subset of another set, meaning that
    /// all values in our set must also be in the other set.
    /// A proper subset must also be smaller than the other set.
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        self.0.is_proper_submap(&other.borrow().0)
    }
}

// Core traits

impl<A, S> Clone for HashSet<A, S> {
    fn clone(&self) -> Self {
        HashSet(self.0.clone())
    }
}

impl<A: Hash + Eq, S: SharedHasher> PartialEq for HashSet<A, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<A: Hash + Eq, S: SharedHasher> Eq for HashSet<A, S> {}

impl<A: Hash + Eq + PartialOrd, S: SharedHasher> PartialOrd for HashSet<A, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<A: Hash + Eq + Ord, S: SharedHasher> Ord for HashSet<A, S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<A: Hash + Eq, S: SharedHasher> Hash for HashSet<A, S> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A: Hash + Eq, S: SharedHasher> Default for HashSet<A, S> {
    fn default() -> Self {
        HashSet(Default::default())
    }
}

impl<'a, A: Hash + Eq, S: SharedHasher> Add for &'a HashSet<A, S> {
    type Output = HashSet<A, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, A: Hash + Eq, S: SharedHasher> Mul for &'a HashSet<A, S> {
    type Output = HashSet<A, S>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A: Hash + Eq + Debug, S: SharedHasher> Debug for HashSet<A, S> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{{ ")?;
        let mut it = self.iter().peekable();
        loop {
            match it.next() {
                None => break,
                Some(a) => {
                    write!(f, "{:?}", a)?;
                    match it.peek() {
                        None => write!(f, " }}")?,
                        Some(_) => write!(f, ", ")?,
                    }
                }
            }
        }
        Ok(())
    }
}

// Iterators

pub struct Iter<A> {
    it: hashmap::Iter<A, ()>,
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(a, _)| a)
    }
}

impl<A: Hash + Eq, RA, S> FromIterator<RA> for HashSet<A, S>
where
    RA: Shared<A>,
    S: SharedHasher,
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

impl<'a, A, S> IntoIterator for &'a HashSet<A, S> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A, S> IntoIterator for HashSet<A, S> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<'a, A: Hash + Eq + Clone, S: SharedHasher> From<&'a [A]> for HashSet<A, S> {
    fn from(slice: &'a [A]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq, S: SharedHasher> From<&'a [Arc<A>]> for HashSet<A, S> {
    fn from(slice: &'a [Arc<A>]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq, S: SharedHasher> From<Vec<A>> for HashSet<A, S> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Clone, S: SharedHasher> From<&'a Vec<A>> for HashSet<A, S> {
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq, S: SharedHasher> From<&'a Vec<Arc<A>>> for HashSet<A, S> {
    fn from(vec: &Vec<Arc<A>>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A: Eq + Hash, S: SharedHasher> From<collections::HashSet<A>> for HashSet<A, S> {
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Clone, S: SharedHasher> From<&'a collections::HashSet<A>>
    for HashSet<A, S> {
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Eq + Hash, S: SharedHasher> From<&'a collections::HashSet<Arc<A>>> for HashSet<A, S> {
    fn from(hash_set: &collections::HashSet<Arc<A>>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq, S: SharedHasher> From<BTreeSet<A>> for HashSet<A, S> {
    fn from(btree_set: BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Clone, S: SharedHasher> From<&'a BTreeSet<A>> for HashSet<A, S> {
    fn from(btree_set: &BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq, S: SharedHasher> From<&'a BTreeSet<Arc<A>>> for HashSet<A, S> {
    fn from(btree_set: &BTreeSet<Arc<A>>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<A: Ord + Hash + Eq, S: SharedHasher> From<OrdSet<A>> for HashSet<A, S> {
    fn from(ordset: OrdSet<A>) -> Self {
        ordset.into_iter().collect()
    }
}

impl<'a, A: Ord + Hash + Eq, S: SharedHasher> From<&'a OrdSet<A>> for HashSet<A, S> {
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
            .prop_map(|v| HashSet::from(v))
            .prop_filter("HashSet minimum size".to_owned(), move |s| {
                s.len() >= size.start
            })
            .boxed()
    }
}

#[cfg(test)]
mod test {
    use super::proptest::*;

    proptest! {
        #[test]
        fn proptest_a_set(ref s in hash_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
