//! A catenable list.
//!
//! A data structure like the simple `ConsList` but with
//! efficient (generally O(1) in the worst case) add
//! and remove operations on both ends, implemented as a `Queue`
//! of `ConsList`s.
//!
//! If you need a list but haven't thought hard about your
//! performance requirements, this is most likely the list you
//! want. If you're mostly going to be consing and unconsing, and
//! you have a lot of data, or a lot of lists, you might want the
//! `ConsList` instead. If you really just need a queue, you might
//! be looking for the `Queue`. When in doubt, choose the `List`.

use std::sync::Arc;
use std::iter::{Sum, FromIterator};
use std::ops::{Add, Deref};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Formatter, Error};
use std::borrow::Borrow;
use queue::Queue;

use self::ListNode::{Nil, Cons};

/// Construct a list from a sequence of elements.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::list::{List, cons};
/// # fn main() {
/// assert_eq!(
///   list![1, 2, 3],
///   List::from(vec![1, 2, 3])
/// );
///
/// assert_eq!(
///   list![1, 2, 3],
///   cons(1, cons(2, cons(3, List::new())))
/// );
/// # }
/// ```
#[macro_export]
macro_rules! list {
    () => { $crate::list::List::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::list::List::new();
        $(
            l = l.push_back($x);
        )*
            l
    }};
}

/// Prepend a value to a list.
///
/// Constructs a list with the value `car` prepended to the
/// front of the list `cdr`.
///
/// This is just a shorthand for `list.cons(item)`, but I find
/// it much easier to read `cons(1, cons(2, List::new()))`
/// than `List::new().cons(2).cons(1)`, given that the resulting
/// list will be `[1, 2]`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::list::{List, cons};
/// # fn main() {
/// assert_eq!(
///   cons(1, cons(2, cons(3, List::new()))),
///   list![1, 2, 3]
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
/// forming proper lists, but this is how they were most commonly used:
/// forming singly linked lists by having the left hand side contain a
/// value, and the right hand side a pointer to the rest of the list.
///
/// `cons` is short for 'construct', which is the easy one. `car` means
/// 'contents of address register' and `cdr` means 'contents of decrement
/// register.' These were the registers on the CPU of the IBM 704 computer
/// (on which Lisp was originally implemented) used to hold the respective
/// values.
///
/// Lisp also commonly provided pre-composed sequences of the `car` and
/// `cdr` functions, such as `cadr`, the `car` of the `cdr`, ie. the
/// second element of a list, and `cddr`, the list with the two first
/// elements dropped. Pronunciation goes like this: `cadr` is, obviously,
/// 'cadder', while `cddr` is 'cududder', and `caddr` (the `car` of the
/// `cdr` of the `cdr`) is 'cadudder'. It can get a little subtle for the
/// untrained ear.
pub fn cons<A, RA, RD>(car: RA, cdr: RD) -> List<A>
where
    Arc<A>: From<RA>,
    RD: Borrow<List<A>>,
{
    cdr.borrow().cons(car)
}

/// A catenable list of values of type `A`.
///
/// A data structure like the simple `ConsList` but with
/// efficient (generally O(1) in the worst case) add
/// and remove operations on both ends.
///
/// If you need a list but haven't thought hard about your
/// performance requirements, this is most likely the list you
/// want. If you're mostly going to be consing and unconsing, and
/// you have a lot of data, or a lot of lists, you might want the
/// `ConsList` instead. When in doubt, choose the `List`.
pub struct List<A>(Arc<ListNode<A>>);

#[doc(hidden)]
pub enum ListNode<A> {
    Nil,
    Cons(usize, Arc<A>, Queue<List<A>>),
}

impl<A> List<A> {
    /// Construct an empty list.
    pub fn new() -> Self {
        List(Arc::new(Nil))
    }

    /// Construct a list with a single value.
    pub fn singleton<R>(a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        List::new().push_front(a)
    }

    /// Construct a list by consuming an `IntoIterator`.
    ///
    /// Allows you to construct a list out of anything that implements
    /// the `IntoIterator` trait.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::list::List;
    /// # fn main() {
    /// assert_eq!(
    ///   List::from(vec![1, 2, 3, 4, 5]),
    ///   list![1, 2, 3, 4, 5]
    /// );
    /// # }
    /// ```
    pub fn from<R, I>(it: I) -> List<A>
    where
        I: IntoIterator<Item = R>,
        Arc<A>: From<R>,
    {
        it.into_iter().map(|a| Arc::from(a)).collect()
    }

    /// Test whether a list is empty.
    pub fn is_empty(&self) -> bool {
        match *self.0 {
            Nil => true,
            _ => false,
        }
    }

    /// Get the length of a list.
    ///
    /// This operation is instant, because cons cells store the
    /// length of the list they're the head of.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(5, list![1, 2, 3, 4, 5].len());
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        match *self.0 {
            Nil => 0,
            Cons(l, _, _) => l,
        }
    }

    /// Get the first element of a list.
    ///
    /// If the list is empty, `None` is returned.
    pub fn head(&self) -> Option<Arc<A>> {
        match *self.0 {
            Nil => None,
            Cons(_, ref a, _) => Some(a.clone()),
        }
    }

    /// Get the tail of a list.
    ///
    /// The tail means all elements in the list after the
    /// first item (the head). If the list only has one
    /// element, the result is an empty list. If the list is
    /// empty, the result is `None`.
    pub fn tail(&self) -> Option<Self> {
        match *self.0 {
            Nil => None,
            Cons(_, _, ref q) if q.is_empty() => Some(List::new()),
            Cons(_, _, ref q) => Some(fold_queue(|a, b| a.link(&b), List::new(), q)),
        }
    }

    /// Append the list `right` to the end of the current list.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::list::List;
    /// # fn main() {
    /// assert_eq!(
    ///   list![1, 2, 3].append(list![7, 8, 9]),
    ///   list![1, 2, 3, 7, 8, 9]
    /// );
    /// # }
    /// ```
    pub fn append<R>(&self, other: R) -> Self
    where
        R: Borrow<Self>,
    {
        match (self, other.borrow()) {
            (l, r) if l.is_empty() => r.clone(),
            (l, r) if r.is_empty() => l.clone(),
            (l, r) => l.link(r),
        }
    }

    pub fn link<R>(&self, other: R) -> Self
    where
        R: Borrow<Self>,
    {
        match *self.0 {
            Nil => other.borrow().clone(),
            Cons(l, ref a, ref q) => {
                List(Arc::new(Cons(
                    l + other.borrow().len(),
                    a.clone(),
                    q.push(other.borrow().clone()),
                )))
            }
        }
    }

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    pub fn cons<R>(&self, a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        List(Arc::new(Cons(1, Arc::from(a), Queue::new()))).append(self)
    }

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    pub fn push_front<R>(&self, a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        self.cons(a)
    }

    /// Construct a list with a new value appended to the back of the
    /// current list.
    ///
    /// `snoc`, for the curious, is `cons` spelled backwards, to denote
    /// that it works on the back of the list rather than the front.
    /// If you don't find that as clever as whoever coined the term no
    /// doubt did, this method is also available as `List::push_back()`.
    pub fn snoc<R>(&self, a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        self.append(&List(Arc::new(Cons(1, Arc::from(a), Queue::new()))))
    }

    /// Construct a list with a new value appended to the back of the
    /// current list.
    pub fn push_back<R>(&self, a: R) -> Self
    where
        Arc<A>: From<R>,
    {
        self.snoc(a)
    }

    /// Get the head and the tail of a list.
    ///
    /// This function performs both the `head` function and
    /// the `tail` function in one go, returning a tuple
    /// of the head and the tail, or `None` if the list is
    /// empty.
    ///
    /// # Examples
    ///
    /// This can be useful when pattern matching your way through
    /// a list:
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::list::{List, cons};
    /// # use std::fmt::Debug;
    /// fn walk_through_list<A>(list: &List<A>) where A: Debug {
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
    pub fn uncons(&self) -> Option<(Arc<A>, List<A>)> {
        self.head().and_then(|h| self.tail().map(|t| (h, t)))
    }

    pub fn uncons2(&self) -> Option<(Arc<A>, Arc<A>, List<A>)> {
        self.uncons().and_then(
            |(a1, d)| d.uncons().map(|(a2, d)| (a1, a2, d)),
        )
    }

    /// Get an iterator over a list.
    pub fn iter(&self) -> Iter<A> {
        Iter { current: self.clone() }
    }

    /// Construct a list which is the reverse of the current list.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::list::List;
    /// # fn main() {
    /// assert_eq!(
    ///   list![1, 2, 3, 4, 5].reverse(),
    ///   list![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    pub fn reverse(&self) -> Self {
        let mut out = List::new();
        for i in self.iter() {
            out = out.cons(i)
        }
        out
    }

    /// Sort a list using a comparator function.
    ///
    /// Time: O(n log n)
    pub fn sort_by<F>(&self, cmp: F) -> Self
    where
        F: Fn(Arc<A>, Arc<A>) -> Ordering,
    {
        fn merge<A>(la: &List<A>, lb: &List<A>, cmp: &Fn(Arc<A>, Arc<A>) -> Ordering) -> List<A> {
            match (la.uncons(), lb.uncons()) {
                (Some((ref a, _)), Some((ref b, ref lb1)))
                    if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    cons(b.clone(), &merge(la, &lb1, cmp))
                }
                (Some((a, la1)), Some((_, _))) => cons(a.clone(), &merge(&la1, lb, cmp)),
                (None, _) => lb.clone(),
                (_, None) => la.clone(),
            }
        }

        fn merge_pairs<A>(
            l: &List<List<A>>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> List<List<A>> {
            match l.uncons2() {
                Some((a, b, rest)) => cons(merge(&a, &b, cmp), &merge_pairs(&rest, cmp)),
                _ => l.clone(),
            }
        }

        fn merge_all<A>(l: &List<List<A>>, cmp: &Fn(Arc<A>, Arc<A>) -> Ordering) -> List<A> {
            match l.uncons() {
                None => list![],
                Some((ref a, ref d)) if d.is_empty() => a.deref().clone(),
                _ => merge_all(&merge_pairs(l, cmp), cmp),
            }
        }

        fn ascending<A>(
            a: Arc<A>,
            f: &Fn(List<A>) -> List<A>,
            l: &List<A>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> List<List<A>> {
            match l.uncons() {
                Some((ref b, ref lb)) if cmp(a.clone(), b.clone()) != Ordering::Greater => {
                    ascending(b.clone(), &|ys| f(cons(a.clone(), &ys)), &lb, cmp)
                }
                _ => cons(f(List::singleton(a.clone())), &sequences(l, cmp)),
            }
        }

        fn descending<A>(
            a: Arc<A>,
            la: &List<A>,
            lb: &List<A>,
            cmp: &Fn(Arc<A>, Arc<A>) -> Ordering,
        ) -> List<List<A>> {
            match lb.uncons() {
                Some((ref b, ref bs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(b.clone(), &cons(a.clone(), la), bs, cmp)
                }
                _ => cons(cons(a.clone(), la), &sequences(&lb, cmp)),
            }
        }

        fn sequences<A>(l: &List<A>, cmp: &Fn(Arc<A>, Arc<A>) -> Ordering) -> List<List<A>> {
            match l.uncons2() {
                Some((ref a, ref b, ref xs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(b.clone(), &List::singleton(a.clone()), xs, cmp)
                }
                Some((ref a, ref b, ref xs)) => {
                    ascending(b.clone(), &|l| cons(a.clone(), l), &xs, cmp)
                }
                None => list![l.clone()],
            }
        }

        merge_all(&sequences(self, &cmp), &cmp)
    }
}

impl List<i32> {
    /// Construct a list of numbers between `from` and `to` inclusive.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::list::{List, cons};
    /// # fn main() {
    /// assert_eq!(
    ///   List::range(1, 5),
    ///   list![1, 2, 3, 4, 5]
    /// );
    /// # }
    /// ```
    pub fn range(from: i32, to: i32) -> List<i32> {
        let mut list = List::new();
        let mut c = to;
        while c >= from {
            list = cons(c, &list);
            c -= 1;
        }
        list
    }
}

impl<A: Ord> List<A> {
    /// Sort a list of ordered elements.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::list::List;
    /// # fn main() {
    /// assert_eq!(
    ///   list![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   List::range(1, 8)
    /// );
    /// # }
    /// ```
    pub fn sort(&self) -> Self {
        self.sort_by(|a, b| a.as_ref().cmp(b.as_ref()))
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
    ///   list![2, 4, 6].insert(5).insert(1).insert(3),
    ///   list![1, 2, 3, 4, 5, 6]
    /// );
    /// # }
    /// ```
    pub fn insert<T>(&self, item: T) -> Self
    where
        Arc<A>: From<T>,
    {
        self.insert_ref(Arc::from(item))
    }

    fn insert_ref(&self, item: Arc<A>) -> Self {
        match self.uncons() {
            None => List::singleton(item),
            Some((a, d)) => {
                if a.deref() > item.deref() {
                    self.cons(item)
                } else {
                    d.insert_ref(item).cons(a.clone())
                }
            }
        }
    }
}

fn fold_queue<A, F>(f: F, seed: List<A>, queue: &Queue<List<A>>) -> List<A>
where
    F: Fn(List<A>, List<A>) -> List<A>,
{
    let mut out = seed;
    let mut q = Vec::new();
    for v in queue {
        q.push(v)
    }
    for a in q.iter().rev() {
        out = f(a.as_ref().clone(), out)
    }
    out
}

// Core traits

impl<A> Clone for List<A> {
    fn clone(&self) -> Self {
        List(self.0.clone())
    }
}

impl<A> Default for List<A> {
    fn default() -> Self {
        List::new()
    }
}

impl<A> Add for List<A> {
    type Output = List<A>;

    fn add(self, other: Self) -> Self::Output {
        self.append(&other)
    }
}

impl<A: PartialEq> PartialEq for List<A> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<A: Eq> Eq for List<A> {}

impl<A: PartialOrd> PartialOrd for List<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord> Ord for List<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Hash> Hash for List<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self.iter() {
            i.hash(state)
        }
    }
}

impl<A: Debug> Debug for List<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "[")?;
        let mut it = self.iter().peekable();
        loop {
            match it.next() {
                None => break,
                Some(a) => {
                    write!(f, "{:?}", a)?;
                    match it.peek() {
                        None => write!(f, "]")?,
                        Some(_) => write!(f, ", ")?,
                    }
                }
            }
        }
        Ok(())
    }
}

// Iterators

/// An iterator over lists with values of type `A`.
pub struct Iter<A> {
    current: List<A>,
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current.uncons() {
            None => None,
            Some((a, d)) => {
                self.current = d;
                Some(a)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.current.len();
        (l, Some(l))
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

impl<A> IntoIterator for List<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        Iter { current: self }
    }
}

impl<'a, A> IntoIterator for &'a List<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> Sum for List<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, T> FromIterator<T> for List<A>
where
    Arc<A>: From<T>,
{
    fn from_iter<I>(source: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        source.into_iter().map(List::singleton).sum()
    }
}

// Conversions

impl<'a, A, T> From<&'a [T]> for List<A>
where
    Arc<A>: From<&'a T>,
{
    fn from(slice: &'a [T]) -> List<A> {
        slice.into_iter().collect()
    }
}

impl<A, T> From<Vec<T>> for List<A>
where
    Arc<A>: From<T>,
{
    fn from(vec: Vec<T>) -> List<A> {
        vec.into_iter().collect()
    }
}

impl<'a, A, T> From<&'a Vec<T>> for List<A>
where
    Arc<A>: From<&'a T>,
{
    fn from(vec: &'a Vec<T>) -> List<A> {
        vec.into_iter().collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for List<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        List::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{Strategy, BoxedStrategy, ValueTree};
    use std::ops::Range;

    /// A strategy for generating a list of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_list(ref l in list(".*", 10..100)) {
    ///         assert!(l.len() < 100);
    ///         assert!(l.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn list<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<List<<T::Value as ValueTree>::Value>> {
        ::proptest::collection::vec(element, size)
            .prop_map(|v| List::from(v))
            .boxed()
    }

    /// A strategy for an ordered list of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_ordered_list(ref l in ordered_list(".*", 10..100)) {
    ///         assert_eq!(l, l.sort());
    ///     }
    /// }
    /// ```
    pub fn ordered_list<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<List<<T::Value as ValueTree>::Value>>
    where
        <T::Value as ValueTree>::Value: Ord,
    {
        ::proptest::collection::vec(element, size)
            .prop_map(|v| List::from(v).sort())
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use super::proptest::*;
    use test::is_sorted;

    quickcheck! {
        fn length(vec: Vec<i32>) -> bool {
            let list = List::from_iter(vec.clone());
            vec.len() == list.len()
        }

        fn order(vec: Vec<i32>) -> bool {
            let list = List::from_iter(vec.clone());
            list.iter().map(|a| *a).eq(vec.into_iter())
        }

        fn equality(vec: Vec<i32>) -> bool {
            let left = List::from_iter(vec.clone());
            let right = List::from_iter(vec);
            left == right
        }

        fn reverse_a_list(l: List<i32>) -> bool {
            let vec: Vec<i32> = l.iter().map(|v| *v).collect();
            let rev = List::from_iter(vec.into_iter().rev());
            l.reverse() == rev
        }

        fn append_two_lists(xs: List<i32>, ys: List<i32>) -> bool {
            let extended = List::from_iter(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            xs.append(&ys) == extended
        }

        fn length_of_append(xs: List<i32>, ys: List<i32>) -> bool {
            let extended = List::from_iter(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            xs.append(&ys).len() == extended.len()
        }

        fn sort_a_list(l: List<i32>) -> bool {
            let sorted = l.sort();
            l.len() == sorted.len() && is_sorted(&sorted)
        }
    }

    proptest! {
        #[test]
        fn proptest_a_list(ref l in list(".*", 10..100)) {
            assert!(l.len() < 100);
            assert!(l.len() >= 10);
        }

        #[test]
        fn proptest_ordered_list(ref l in ordered_list(".*", 10..100)) {
            assert_eq!(l, &l.sort());
        }
    }
}
