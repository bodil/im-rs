// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A cons list.
//!
//! The cons list is perhaps the most basic immutable data structure:
//! a singly linked list built out of 'cons cells,' which are cells
//! containing two values, the left hand value being the head of the
//! list and the right hand value being a reference to the rest of the
//! list, or a `Nil` value denoting the end of the list.
//!
//! Structure can be shared between lists (and is reference counted),
//! and append to the front of a list is O(1). Cons cells keep track
//! of the length of the list at the current position, as an extra
//! optimisation, so getting the length of a list is also O(1).
//! Otherwise, operations are generally O(n).
//!
//! Unless you know you want a `ConsList`, you're probably better off
//! using a [`Vector`][vector::Vector], which has more efficient
//! performance characteristics in almost all cases. The `ConsList` is
//! particularly useful as an immutable stack where you only push and
//! pop items from the front of the list. Beware that it has no
//! mutable operations.
//!
//! [vector::Vector]: ../vector/struct.Vector.html

use shared::Shared;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, Iterator, Sum};
use std::ops::{Add, Deref};
use std::sync::Arc;

use self::ConsListNode::{Cons, Nil};

/// Construct a list from a sequence of elements.
///
/// # Examples
///
/// Here are some different ways of constructing a list of
/// three numbers 1, 2 and 3:
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::conslist::{ConsList, cons};
/// # fn main() {
/// assert_eq!(
///   conslist![1, 2, 3],
///   ConsList::from(vec![1, 2, 3])
/// );
///
/// assert_eq!(
///   conslist![1, 2, 3],
///   cons(1, cons(2, cons(3, ConsList::new())))
/// );
/// # }
/// ```
#[macro_export]
macro_rules! conslist {
    () => { $crate::conslist::ConsList::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::conslist::ConsList::new();
        $(
            l = l.cons($x);
        )*
            l.reverse()
    }};
}

/// Prepend a value to a list.
///
/// Constructs a list with the value `car` prepended to the front of
/// the list `cdr`.
///
/// This is just a shorthand for `list.cons(item)`, but I find it much
/// easier to read `cons(1, cons(2, ConsList::new()))` than
/// `ConsList::new().cons(2).cons(1)`, given that the resulting list
/// will be `[1, 2]`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::conslist::{ConsList, cons};
/// # fn main() {
/// assert_eq!(
///   cons(1, cons(2, cons(3, ConsList::new()))),
///   conslist![1, 2, 3]
/// );
/// # }
/// ```
///
/// # Historical Anecdote
///
/// The words `car` and `cdr` come from Lisp, and were the original
/// names of the functions to get the left and the right hands of a
/// cons cell, respectively. Cons cells in Lisp were simply containers
/// for two values: the car and the cdr (pronounced 'cudder'), and,
/// Lisp being an untyped language, had no restrictions on cons cells
/// forming proper lists, but this is how they were most commonly
/// used: forming singly linked lists by having the left hand side
/// contain a value, and the right hand side a pointer to the rest of
/// the list.
///
/// `cons` is short for 'construct', which is the easy one. `car`
/// means 'contents of address register' and `cdr` means 'contents of
/// decrement register.' These were the registers on the CPU of the
/// IBM 704 computer (on which Lisp was originally implemented) used
/// to hold the respective values.
///
/// Lisp also commonly provided pre-composed sequences of the `car`
/// and `cdr` functions, such as `cadr`, the `car` of the `cdr`, ie.
/// the second element of a list, and `cddr`, the list with the two
/// first elements dropped. Pronunciation goes like this: `cadr` is,
/// obviously, 'cadder', while `cddr` is 'cududder', and `caddr` (the
/// `car` of the `cdr` of the `cdr`) is 'cadudder'. It can get a
/// little subtle for the untrained ear.
#[inline]
pub fn cons<A, RA, RD>(car: RA, cdr: RD) -> ConsList<A>
where
    RA: Shared<A>,
    RD: Borrow<ConsList<A>>,
{
    cdr.borrow().cons(car)
}

/// An immutable proper cons lists.
///
/// The cons list is perhaps the most basic immutable data structure:
/// a singly linked list built out of 'cons cells,' which are cells
/// containing two values, the left hand value being the head of the
/// list and the right hand value being a reference to the rest of the
/// list, or a `Nil` value denoting the end of the list.
///
/// Structure can be shared between lists (and is reference counted),
/// and append to the front of a list is O(1). Cons cells keep track
/// of the length of the list at the current position, as an extra
/// optimisation, so getting the length of a list is also O(1).
/// Otherwise, operations are generally O(n).
///
/// Unless you know you want a `ConsList`, you're probably better off
/// using a [`Vector`][vector::Vector], which has more efficient
/// performance characteristics in almost all cases. The `ConsList` is
/// particularly useful as an immutable stack where you only push and
/// pop items from the front of the list. Beware that it has no
/// mutable operations.
///
/// [vector::Vector]: ../vector/struct.Vector.html
pub struct ConsList<A>(Arc<ConsListNode<A>>);

#[doc(hidden)]
pub enum ConsListNode<A> {
    Cons(usize, Arc<A>, ConsList<A>),
    Nil,
}

impl<A> ConsList<A> {
    /// Construct an empty list.
    pub fn new() -> ConsList<A> {
        ConsList(Arc::new(Nil))
    }

    /// Construct a list with a single element.
    pub fn singleton<R>(v: R) -> ConsList<A>
    where
        R: Shared<A>,
    {
        ConsList(Arc::new(Cons(1, v.shared(), conslist![])))
    }

    /// Test whether a list is empty.
    ///
    /// Time: O(1)
    pub fn is_empty(&self) -> bool {
        match *self.0 {
            Nil => true,
            _ => false,
        }
    }

    /// Construct a list with a new value prepended to the front of
    /// the current list.
    ///
    /// Time: O(1)
    pub fn cons<R>(&self, car: R) -> ConsList<A>
    where
        R: Shared<A>,
    {
        ConsList(Arc::new(Cons(self.len() + 1, car.shared(), self.clone())))
    }

    /// Get the first element of a list.
    ///
    /// If the list is empty, `None` is returned.
    ///
    /// Time: O(1)
    pub fn head(&self) -> Option<Arc<A>> {
        match *self.0 {
            Cons(_, ref a, _) => Some(a.clone()),
            _ => None,
        }
    }

    /// Get the tail of a list.
    ///
    /// The tail means all elements in the list after the first item
    /// (the head). If the list only has one element, the result is an
    /// empty list. If the list is empty, the result is `None`.
    ///
    /// Time: O(1)
    pub fn tail(&self) -> Option<ConsList<A>> {
        match *self.0 {
            Cons(_, _, ref d) => Some(d.clone()),
            Nil => None,
        }
    }

    /// Get the head and the tail of a list.
    ///
    /// This function performs both the [`head`][head] function and
    /// the [`tail`][tail] function in one go, returning a tuple of
    /// the head and the tail, or [`None`][None] if the list is empty.
    ///
    /// # Examples
    ///
    /// This can be useful when pattern matching your way through a
    /// list:
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::conslist::{ConsList, cons};
    /// # use std::fmt::Debug;
    /// fn walk_through_list<A>(list: &ConsList<A>) where A: Debug {
    ///     match list.uncons() {
    ///         None => (),
    ///         Some((ref head, ref tail)) => {
    ///             print!("{:?}", head);
    ///             walk_through_list(tail)
    ///         }
    ///     }
    /// }
    /// # fn main() {
    /// # }
    /// ```
    ///
    /// Time: O(1)
    ///
    /// [head]: #method.head
    /// [tail]: #method.tail
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn uncons(&self) -> Option<(Arc<A>, ConsList<A>)> {
        match *self.0 {
            Nil => None,
            Cons(_, ref a, ref d) => Some((a.clone(), d.clone())),
        }
    }

    pub fn uncons2(&self) -> Option<(Arc<A>, Arc<A>, ConsList<A>)> {
        self.uncons()
            .and_then(|(a1, d)| d.uncons().map(|(a2, d)| (a1, a2, d)))
    }

    /// Get the length of a list.
    ///
    /// This operation is instant, because cons cells store the length
    /// of the list they're the head of.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(5, conslist![1, 2, 3, 4, 5].len());
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        match *self.0 {
            Nil => 0,
            Cons(l, _, _) => l,
        }
    }

    /// Append the list `right` to the end of the current list.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::conslist::ConsList;
    /// # fn main() {
    /// assert_eq!(
    ///   conslist![1, 2, 3].append(conslist![7, 8, 9]),
    ///   conslist![1, 2, 3, 7, 8, 9]
    /// );
    /// # }
    /// ```
    pub fn append<R>(&self, right: R) -> Self
    where
        R: Borrow<Self>,
    {
        match *self.0 {
            Nil => right.borrow().clone(),
            Cons(_, ref a, ref d) => cons(a.clone(), &d.append(right)),
        }
    }

    /// Construct a list which is the reverse of the current list.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::conslist::ConsList;
    /// # fn main() {
    /// assert_eq!(
    ///   conslist![1, 2, 3, 4, 5].reverse(),
    ///   conslist![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    pub fn reverse(&self) -> ConsList<A> {
        let mut out = ConsList::new();
        for i in self.iter() {
            out = out.cons(i);
        }
        out
    }

    /// Get an iterator over a list.
    pub fn iter(&self) -> Iter<A> {
        Iter {
            current: self.clone(),
        }
    }

    /// Sort a list using a comparator function.
    ///
    /// Time: O(n log n)
    pub fn sort_by<F>(&self, cmp: F) -> ConsList<A>
    where
        F: Fn(Arc<A>, Arc<A>) -> Ordering,
    {
        fn merge<A>(
            la: &ConsList<A>,
            lb: &ConsList<A>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> ConsList<A> {
            match (la.uncons(), lb.uncons()) {
                (Some((ref a, _)), Some((ref b, ref lb1)))
                    if cmp(a.clone(), b.clone()) == Ordering::Greater =>
                {
                    cons(b.clone(), &merge(la, lb1, cmp))
                }
                (Some((a, la1)), Some((_, _))) => cons(a.clone(), &merge(&la1, lb, cmp)),
                (None, _) => lb.clone(),
                (_, None) => la.clone(),
            }
        }

        fn merge_pairs<A>(
            l: &ConsList<ConsList<A>>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> ConsList<ConsList<A>> {
            match l.uncons2() {
                Some((a, b, rest)) => cons(merge(&a, &b, cmp), &merge_pairs(&rest, cmp)),
                _ => l.clone(),
            }
        }

        fn merge_all<A>(
            l: &ConsList<ConsList<A>>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> ConsList<A> {
            match l.uncons() {
                None => conslist![],
                Some((ref a, ref d)) if d.is_empty() => a.deref().clone(),
                _ => merge_all(&merge_pairs(l, cmp), cmp),
            }
        }

        fn ascending<A>(
            a: &Arc<A>,
            f: &Fn(ConsList<A>) -> ConsList<A>,
            l: &ConsList<A>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> ConsList<ConsList<A>> {
            match l.uncons() {
                Some((ref b, ref lb)) if cmp(a.clone(), b.clone()) != Ordering::Greater => {
                    ascending(&b.clone(), &|ys| f(cons(a.clone(), &ys)), lb, cmp)
                }
                _ => cons(f(ConsList::singleton(a.clone())), &sequences(l, cmp)),
            }
        }

        fn descending<A>(
            a: &Arc<A>,
            la: &ConsList<A>,
            lb: &ConsList<A>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> ConsList<ConsList<A>> {
            match lb.uncons() {
                Some((ref b, ref bs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(&b.clone(), &cons(a.clone(), la), bs, cmp)
                }
                _ => cons(cons(a.clone(), la), &sequences(lb, cmp)),
            }
        }

        fn sequences<A>(
            l: &ConsList<A>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> ConsList<ConsList<A>> {
            match l.uncons2() {
                Some((ref a, ref b, ref xs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(&b.clone(), &ConsList::singleton(a.clone()), xs, cmp)
                }
                Some((ref a, ref b, ref xs)) => {
                    ascending(&b.clone(), &|l| cons(a.clone(), l), xs, cmp)
                }
                None => conslist![l.clone()],
            }
        }

        merge_all(&sequences(self, &cmp), &cmp)
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Insert an item into a sorted list.
    ///
    /// Constructs a new list with the new item inserted before the
    /// first item in the list which is larger than the new item,
    /// as determined by the `Ord` trait.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(
    ///   conslist![2, 4, 6].insert(5).insert(1).insert(3),
    ///   conslist![1, 2, 3, 4, 5, 6]
    /// );
    /// # }
    /// ```
    pub fn insert<T>(&self, item: T) -> ConsList<A>
    where
        A: Ord,
        T: Shared<A>,
    {
        self.insert_ref(item.shared())
    }

    fn insert_ref(&self, item: Arc<A>) -> ConsList<A>
    where
        A: Ord,
    {
        match *self.0 {
            Nil => ConsList(Arc::new(Cons(0, item, ConsList::new()))),
            Cons(_, ref a, ref d) => {
                if a.deref() > item.deref() {
                    self.cons(item)
                } else {
                    d.insert_ref(item).cons(a.clone())
                }
            }
        }
    }

    /// Sort a list.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::conslist::ConsList;
    /// # use std::iter::FromIterator;
    /// # fn main() {
    /// assert_eq!(
    ///   conslist![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   ConsList::from_iter(1..9)
    /// );
    /// # }
    /// ```
    pub fn sort(&self) -> ConsList<A>
    where
        A: Ord,
    {
        self.sort_by(|a: Arc<A>, b: Arc<A>| a.cmp(&b))
    }
}

impl<A> ConsList<A>
where
    A: Ord,
{
}

// Core traits

impl<A> Clone for ConsList<A> {
    /// Clone a list.
    ///
    /// Cons cells use `Arc` behind the scenes, so this is no more
    /// expensive than cloning an `Arc` reference.
    ///
    /// Time: O(1)
    fn clone(&self) -> Self {
        match *self {
            ConsList(ref node) => ConsList(node.clone()),
        }
    }
}

impl<A> Default for ConsList<A> {
    /// `Default` for lists is the empty list.
    fn default() -> Self {
        ConsList::new()
    }
}

#[cfg(not(has_specialisation))]
impl<A> PartialEq for ConsList<A>
where
    A: PartialEq,
{
    /// Test if two lists are equal.
    ///
    /// This could potentially be an expensive operation, as we need to walk
    /// both lists to test for equality. We can very quickly determine equality
    /// if the lists have different lengths (can't be equal). Otherwise, we walk the
    /// lists to compare values.
    ///
    /// Time: O(n)
    fn eq(&self, other: &ConsList<A>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A> PartialEq for ConsList<A>
where
    A: PartialEq,
{
    /// Test if two lists are equal.
    ///
    /// This could potentially be an expensive operation, as we need to walk
    /// both lists to test for equality. We can very quickly determine equality
    /// if the lists have different lengths (can't be equal). Otherwise, we walk the
    /// lists to compare values.
    ///
    /// If `A` implements `Eq`, we have an additional shortcut available to us: if
    /// both lists refer to the same cons cell, as determined by `Arc::ptr_eq`, they
    /// have to be equal.
    ///
    /// Time: O(n)
    default fn eq(&self, other: &ConsList<A>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A> PartialEq for ConsList<A>
where
    A: Eq,
{
    /// Test if two lists are equal.
    ///
    /// This could potentially be an expensive operation, as we need to walk
    /// both lists to test for equality. We can very quickly determine equality
    /// if the lists have different lengths (can't be equal). Otherwise, we walk the
    /// lists to compare values.
    ///
    /// If `A` implements `Eq`, we have an additional shortcut available to us: if
    /// both lists refer to the same cons cell, as determined by `Arc::ptr_eq`, they
    /// have to be equal.
    ///
    /// Time: O(n)
    fn eq(&self, other: &ConsList<A>) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.len() == other.len() && self.iter().eq(other.iter()))
    }
}

impl<A> Eq for ConsList<A>
where
    A: Eq,
{
}

impl<A> Hash for ConsList<A>
where
    A: Hash,
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

impl<'a, A> Add for &'a ConsList<A> {
    type Output = ConsList<A>;

    fn add(self, other: Self) -> Self::Output {
        self.append(other)
    }
}

impl<A> Add for ConsList<A> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.append(&other)
    }
}

impl<A> Debug for ConsList<A>
where
    A: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        fn items<A>(l: &ConsList<A>, f: &mut Formatter) -> Result<(), Error>
        where
            A: Debug,
        {
            match *l.0 {
                Nil => Ok(()),
                Cons(_, ref a, ref d) => {
                    write!(f, ", {:?}", a)?;
                    items(d, f)
                }
            }
        }
        write!(f, "[")?;
        match *self.0 {
            Nil => Ok(()),
            Cons(_, ref a, ref d) => {
                write!(f, "{:?}", a)?;
                items(d, f)
            }
        }?;
        write!(f, "]")
    }
}

// Iterators

pub struct Iter<A> {
    #[doc(hidden)]
    current: ConsList<A>,
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current.uncons() {
            None => None,
            Some((ref a, ref d)) => {
                self.current = d.clone();
                Some(a.clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.current.len();
        (l, Some(l))
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

impl<A> IntoIterator for ConsList<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Iter<A> {
        self.iter()
    }
}

impl<A> Sum for ConsList<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a.append(b))
    }
}

impl<A, T> FromIterator<T> for ConsList<A>
where
    T: Shared<A>,
{
    fn from_iter<I>(source: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        source
            .into_iter()
            .fold(conslist![], |l, v| l.cons(v))
            .reverse()
    }
}

// Conversions

impl<'a, A, R> From<&'a [R]> for ConsList<A>
where
    &'a R: Shared<A>,
{
    fn from(slice: &'a [R]) -> Self {
        slice.into_iter().map(|a| a.shared()).collect()
    }
}

impl<A, R> From<Vec<R>> for ConsList<A>
where
    R: Shared<A>,
{
    fn from(vec: Vec<R>) -> Self {
        vec.into_iter().map(|a| a.shared()).collect()
    }
}

impl<'a, A, R> From<&'a Vec<R>> for ConsList<A>
where
    &'a R: Shared<A>,
{
    fn from(vec: &'a Vec<R>) -> Self {
        vec.into_iter().map(|a| a.shared()).collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for ConsList<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        ConsList::from(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for a cons list of a given size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_conslist(ref l in conslist(".*", 10..100)) {
    ///         assert!(l.len() < 100);
    ///         assert!(l.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn conslist<A: Strategy + 'static>(
        element: A,
        size: Range<usize>,
    ) -> BoxedStrategy<ConsList<<A::Value as ValueTree>::Value>> {
        ::proptest::collection::vec(element, size.clone())
            .prop_map(ConsList::from)
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;
    use test::is_sorted;

    #[test]
    fn exact_size_iterator() {
        assert_eq!(10, ConsList::from_iter(1..11).iter().len());
    }

    #[test]
    fn collect_from_iterator() {
        let o: ConsList<i32> = vec![5, 6, 7].iter().cloned().collect();
        assert_eq!(o, conslist![5, 6, 7]);
    }

    #[test]
    fn disequality() {
        let l = ConsList::from_iter(1..6);
        assert_ne!(l, cons(0, &l));
        assert_ne!(l, conslist![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn equality_of_empty_lists() {
        let l1 = ConsList::<String>::new();
        let l2 = ConsList::<String>::new();
        assert_eq!(l1, l2);
    }

    quickcheck! {
        fn length(vec: Vec<i32>) -> bool {
            let list = ConsList::from(vec.clone());
            vec.len() == list.len()
        }

        fn equality(vec: Vec<i32>) -> bool {
            let list1 = ConsList::from(vec.clone());
            let list2 = ConsList::from(vec.clone());
            list1 == list2
        }

        fn order(vec: Vec<i32>) -> bool {
            let list = ConsList::from(vec.clone());
            list.iter().map(|a| *a).eq(vec.into_iter())
        }

        fn reverse_a_list(l: ConsList<i32>) -> bool {
            let vec: Vec<i32> = l.iter().map(|v| *v).collect();
            let rev = ConsList::from_iter(vec.into_iter().rev());
            l.reverse() == rev
        }

        fn append_two_lists(xs: ConsList<i32>, ys: ConsList<i32>) -> bool {
            let extended = ConsList::from_iter(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            xs.append(&ys) == extended
        }

        fn sort_a_list(l: ConsList<i32>) -> bool {
            let sorted = l.sort();
            l.len() == sorted.len() && is_sorted(sorted)
        }
    }

    proptest! {
        #[test]
        fn proptest_a_conslist(ref l in conslist(".*", 10..100)) {
            assert!(l.len() < 100);
            assert!(l.len() >= 10);
        }
    }
}
