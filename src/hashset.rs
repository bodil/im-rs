//! A hash set.
//!
//! An immutable has set backed by a `HashMap`.
//!
//! This is implemented as a `HashMap` with no values, so it shares
//! the exact performance characteristics of `HashMap`.

use std::sync::Arc;
use std::iter::{FromIterator, IntoIterator};
use std::fmt::{Debug, Error, Formatter};
use std::collections::{self, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul};
use std::borrow::Borrow;
use hashmap::{self, HashMap};
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
            l = l.insert($x);
        )*
            l
    }};
}

/// A hash set.
///
/// An immutable has set backed by a `HashMap`.
///
/// This is implemented as a `HashMap` with no values, so it shares
/// the exact performance characteristics of `HashMap`.
pub struct HashSet<A>(HashMap<A, ()>);

impl<A> HashSet<A>
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
    /// let set = hashset![456];
    /// assert_eq!(
    ///   set.insert(123),
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

    /// Construct the union of two sets.
    pub fn union(&self, other: &Self) -> Self {
        HashSet(self.0.union(&other.0))
    }

    /// Construct the union of multiple sets.
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(hashset![], |a, b| a.union(&b))
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

impl<A> Clone for HashSet<A> {
    fn clone(&self) -> Self {
        HashSet(self.0.clone())
    }
}

impl<A: Hash + Eq> PartialEq for HashSet<A> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<A: Hash + Eq> Eq for HashSet<A> {}

impl<A: Hash + Eq> Hash for HashSet<A> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A: Hash + Eq> Default for HashSet<A> {
    fn default() -> Self {
        hashset![]
    }
}

impl<'a, A: Hash + Eq> Add for &'a HashSet<A> {
    type Output = HashSet<A>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, A: Hash + Eq> Mul for &'a HashSet<A> {
    type Output = HashSet<A>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A: Hash + Eq + Debug> Debug for HashSet<A> {
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

impl<A: Hash + Eq> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(a, _)| a)
    }
}

impl<A: Hash + Eq, RA> FromIterator<RA> for HashSet<A>
where
    RA: Shared<A>,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        i.into_iter().fold(hashset![], |s, a| s.insert(a))
    }
}

impl<'a, A: Hash + Eq> IntoIterator for &'a HashSet<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A: Hash + Eq> IntoIterator for HashSet<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<'a, A: Hash + Eq + Clone> From<&'a [A]> for HashSet<A> {
    fn from(slice: &'a [A]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq> From<&'a [Arc<A>]> for HashSet<A> {
    fn from(slice: &'a [Arc<A>]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq> From<Vec<A>> for HashSet<A> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Clone> From<&'a Vec<A>> for HashSet<A> {
    fn from(vec: &Vec<A>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq> From<&'a Vec<Arc<A>>> for HashSet<A> {
    fn from(vec: &Vec<Arc<A>>) -> Self {
        vec.into_iter().cloned().collect()
    }
}

impl<A: Eq + Hash> From<collections::HashSet<A>> for HashSet<A> {
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Clone> From<&'a collections::HashSet<A>> for HashSet<A> {
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Eq + Hash> From<&'a collections::HashSet<Arc<A>>> for HashSet<A> {
    fn from(hash_set: &collections::HashSet<Arc<A>>) -> Self {
        hash_set.into_iter().cloned().collect()
    }
}

impl<A: Hash + Eq> From<BTreeSet<A>> for HashSet<A> {
    fn from(btree_set: BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Hash + Eq + Clone> From<&'a BTreeSet<A>> for HashSet<A> {
    fn from(btree_set: &BTreeSet<A>) -> Self {
        btree_set.into_iter().cloned().collect()
    }
}

impl<'a, A: Hash + Eq> From<&'a BTreeSet<Arc<A>>> for HashSet<A> {
    fn from(btree_set: &BTreeSet<Arc<A>>) -> Self {
        btree_set.into_iter().cloned().collect()
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
    pub fn hashset<A: Strategy + 'static>(
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
        fn proptest_a_set(ref s in hashset(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
