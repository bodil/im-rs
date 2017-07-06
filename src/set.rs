//! An ordered set.
//!
//! An immutable ordered set implemented as a balanced 2-3 tree.
//!
//! This is implemented as a `Map` with no values, so it shares
//! the exact performance characteristics of `Map`.

use std::sync::Arc;
use std::iter::{IntoIterator, FromIterator};
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Error};
use std::collections::{HashSet, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul};
use std::borrow::Borrow;
use map::{self, Map};

/// Construct a set from a sequence of values.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::set::Set;
/// # fn main() {
/// assert_eq!(
///   set![1, 2, 3],
///   Set::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! set {
    () => { $crate::set::Set::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::set::Set::new();
        $(
            l = l.insert($x);
        )*
            l
    }};
}

/// # Ordered Set
///
/// An immutable ordered set implemented as a balanced 2-3 tree.
///
/// This is implemented as a `Map` with no values, so it shares
/// the exact performance characteristics of `Map`.
pub struct Set<A>(Map<A, ()>);

impl<A> Set<A> {
    /// Construct an empty set.
    pub fn new() -> Self {
        Set(Map::new())
    }

    /// Construct a set with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::set::Set;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let set = Set::singleton(123);
    /// assert!(set.contains(&123));
    /// # }
    /// ```
    pub fn singleton<R>(a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        Set(Map::<A, ()>::singleton(a, ()))
    }

    pub fn iter(&self) -> Iter<A> {
        Iter { it: self.0.iter() }
    }

    /// Test whether a set is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::set::Set;
    /// # fn main() {
    /// assert!(
    ///   !set![1, 2, 3].is_empty()
    /// );
    /// assert!(
    ///   Set::<i32>::new().is_empty()
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
    /// # use im::set::Set;
    /// # fn main() {
    /// assert_eq!(3, set![1, 2, 3].len());
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Get the smallest value in a set.
    ///
    /// If the set is empty, returns `None`.
    pub fn get_min(&self) -> Option<Arc<A>> {
        self.0.get_min().map(|(a, _)| a)
    }

    /// Get the largest value in a set.
    ///
    /// If the set is empty, returns `None`.
    pub fn get_max(&self) -> Option<Arc<A>> {
        self.0.get_max().map(|(a, _)| a)
    }
}

impl<A: Ord> Set<A> {
    /// Insert a value into a set.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::set::Set;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let set = set![456];
    /// assert_eq!(
    ///   set.insert(123),
    ///   set![123, 456]
    /// );
    /// # }
    /// ```
    pub fn insert<R>(&self, a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        Set(self.0.insert(a, ()))
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    pub fn contains(&self, a: &A) -> bool {
        self.0.contains_key(a)
    }

    /// Remove a value from a set.
    pub fn remove(&self, a: &A) -> Self {
        Set(self.0.remove(a))
    }

    /// Construct the union of two sets.
    pub fn union(&self, other: &Self) -> Self {
        Set(self.0.union(&other.0))
    }

    /// Construct the union of multiple sets.
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(set![], |a, b| a.union(&b))
    }

    /// Construct the difference between two sets.
    pub fn difference<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        Set(self.0.difference(&other.borrow().0))
    }

    /// Construct the intersection of two sets.
    pub fn intersection<RS>(&self, other: RS) -> Self
    where
        RS: Borrow<Self>,
    {
        Set(self.0.intersection(&other.borrow().0))
    }

    /// Split a set into two, with the left hand set containing values which are smaller
    /// than `split`, and the right hand set containing values which are larger than `split`.
    ///
    /// The `split` value itself is discarded.
    pub fn split(&self, split: &A) -> (Self, Self) {
        let (l, r) = self.0.split(split);
        (Set(l), Set(r))
    }

    /// Split a set into two, with the left hand set containing values which are smaller
    /// than `split`, and the right hand set containing values which are larger than `split`.
    ///
    /// Returns a tuple of the two maps and a boolean which is true if the `split` value
    /// existed in the original set, and false otherwise.
    pub fn split_member(&self, split: &A) -> (Self, bool, Self) {
        let (l, m, r) = self.0.split_lookup(split);
        (Set(l), m.is_some(), Set(r))
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

    /// Construct a set with only the `n` smallest values from a given set.
    pub fn take(&self, n: usize) -> Self {
        Set(self.0.take(n))
    }

    /// Construct a set with the `n` smallest values removed from a given set.
    pub fn drop(&self, n: usize) -> Self {
        Set(self.0.drop(n))
    }

    /// Remove the smallest value from a set, and return that value as well as the updated set.
    pub fn pop_min(&self) -> (Option<Arc<A>>, Self) {
        let (pair, set) = self.0.pop_min_with_key();
        (pair.map(|(a, _)| a), Set(set))
    }

    /// Remove the largest value from a set, and return that value as well as the updated set.
    pub fn pop_max(&self) -> (Option<Arc<A>>, Self) {
        let (pair, set) = self.0.pop_max_with_key();
        (pair.map(|(a, _)| a), Set(set))
    }

    /// Discard the smallest value from a set, returning the updated set.
    pub fn remove_min(&self) -> Self {
        self.pop_min().1
    }

    /// Discard the largest value from a set, returning the updated set.
    pub fn remove_max(&self) -> Self {
        self.pop_max().1
    }
}

// Core traits

impl<A> Clone for Set<A> {
    fn clone(&self) -> Self {
        Set(self.0.clone())
    }
}

impl<A: PartialEq> PartialEq for Set<A> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<A: Eq> Eq for Set<A> {}

impl<A: PartialOrd> PartialOrd for Set<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<A: Ord> Ord for Set<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<A: Hash> Hash for Set<A> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A> Default for Set<A> {
    fn default() -> Self {
        set![]
    }
}

impl<'a, A: Ord> Add for &'a Set<A> {
    type Output = Set<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, A: Ord> Mul for &'a Set<A> {
    type Output = Set<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A: Debug> Debug for Set<A> {
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
    it: map::Iter<A, ()>,
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(a, _)| a)
    }
}

impl<A: Ord, RA> FromIterator<RA> for Set<A>
where
    Arc<A>: From<RA>,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        i.into_iter().fold(set![], |s, a| s.insert(a))
    }
}

impl<'a, A> IntoIterator for &'a Set<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> IntoIterator for Set<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<'a, A: Ord + Clone> From<&'a [A]> for Set<A> {
    fn from(slice: &'a [A]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<'a, A: Ord> From<&'a [Arc<A>]> for Set<A> {
    fn from(slice: &'a [Arc<A>]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<A: Ord> From<Vec<A>> for Set<A> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a Vec<A>> for Set<A> {
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<'a, A: Ord> From<&'a Vec<Arc<A>>> for Set<A> {
    fn from(vec: &Vec<Arc<A>>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A: Eq + Hash + Ord> From<HashSet<A>> for Set<A> {
    fn from(hash_set: HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Ord + Clone> From<&'a HashSet<A>> for Set<A> {
    fn from(hash_set: &HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Eq + Hash + Ord> From<&'a HashSet<Arc<A>>> for Set<A> {
    fn from(hash_set: &HashSet<Arc<A>>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<A: Ord> From<BTreeSet<A>> for Set<A> {
    fn from(btree_set: BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone> From<&'a BTreeSet<A>> for Set<A> {
    fn from(btree_set: &BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Ord> From<&'a BTreeSet<Arc<A>>> for Set<A> {
    fn from(btree_set: &BTreeSet<Arc<A>>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Ord + Arbitrary + Sync> Arbitrary for Set<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Set::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{Strategy, BoxedStrategy, ValueTree};
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
    pub fn set<A: Strategy + 'static>(
        element: A,
        size: Range<usize>,
    ) -> BoxedStrategy<Set<<A::Value as ValueTree>::Value>>
    where
        <A::Value as ValueTree>::Value: Ord,
    {
        ::proptest::collection::vec(element, size.clone())
            .prop_map(|v| Set::from(v))
            .prop_filter(
                "Set minimum size".to_owned(),
                move |s| s.len() >= size.start,
            )
            .boxed()
    }
}

#[cfg(test)]
mod test {
    use super::proptest::*;

    proptest! {
        #[test]
        fn proptest_a_set(ref s in set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
