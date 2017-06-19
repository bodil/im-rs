//! # Cons List
//!
//! An implementation of immutable proper cons lists.
//!
//! Structure can be shared between lists (and is reference
//! counted), and append to the front of a list is O(1).
//! Cons cells keep track of the length of the list at the
//! current position, as an extra optimisation, so getting
//! the length of a list is also O(1). Otherwise, operations
//! are generally O(n).
//!
//! Items in the list are stored in `Arc`s, and insertion
//! operations accept any value for which there's a `From`
//! implementation into `Arc<A>`. Iterators and lookup
//! operations, conversely, produce `Arc<A>`.

use std::sync::Arc;
use std::iter::{Iterator, FromIterator};
use std::fmt::{Debug, Formatter, Error};
use std::ops::Deref;
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;

use self::ListNode::{Cons, Nil};

/// Construct a list from a sequence of elements.
///
/// # Examples
///
/// Here are some different ways of constructing a list of
/// three numbers 1, 2 and 3:
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
///   cons(1, &cons(2, &cons(3, &List::new())))
/// );
/// # }
/// ```
#[macro_export]
macro_rules! list {
    () => { $crate::list::List::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::list::List::new();
        $(
            l = l.cons($x);
        )*
            l.reverse()
    }};
}

/// Prepend a value to a list.
///
/// Constructs a list with the value `car` prepended to the
/// front of the list `cdr`.
///
/// This is just a shorthand for `list.cons(item)`, but I find
/// it much easier to read `cons(1, &cons(2, &List::empty()))`
/// than `List::empty().cons(2).cons(1)`, given that the resulting
/// list will be `[1, 2]`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::list::{List, cons};
/// # fn main() {
/// assert_eq!(
///   cons(1, &cons(2, &cons(3, &List::new()))),
///   list![1, 2, 3]
/// );
/// # }
/// ```
pub fn cons<A, R>(car: R, cdr: &List<A>) -> List<A>
    where Arc<A>: From<R>
{
    cdr.cons(car)
}

/// A list of elements of type A.
pub struct List<A>(Arc<ListNode<A>>);

#[doc(hidden)]
pub enum ListNode<A> {
    Cons(usize, Arc<A>, List<A>),
    Nil,
}

impl<A> List<A> {
    /// Construct an empty list.
    pub fn new() -> List<A> {
        List(Arc::new(Nil))
    }

    /// Construct a list with a single element.
    pub fn singleton<R>(v: R) -> List<A>
        where Arc<A>: From<R>
    {
        List(Arc::new(Cons(1, Arc::from(v), list![])))
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
        where I: IntoIterator<Item = R>,
              Arc<A>: From<R>
    {
        it.into_iter().map(|a| Arc::from(a)).collect()
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

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    ///
    /// Time: O(1)
    pub fn cons<R>(&self, car: R) -> List<A>
        where Arc<A>: From<R>
    {
        List(Arc::new(Cons(self.len() + 1, Arc::from(car), self.clone())))
    }

    /// Get the first element of a list.
    ///
    /// If the list is empty, `None` is returned.
    ///
    /// Time: O(1)
    pub fn head<T>(&self) -> Option<T>
        where T: From<Arc<A>>
    {
        match *self.0 {
            Cons(_, ref a, _) => Some(T::from(a.clone())),
            _ => None,
        }
    }

    /// Get the tail of a list.
    ///
    /// The tail means all elements in the list after the
    /// first item (the head). If the list only has one
    /// element, the result is an empty list. If the list is
    /// empty, the result is `None`.
    ///
    /// Time: O(1)
    pub fn tail(&self) -> Option<List<A>> {
        match *self.0 {
            Cons(_, _, ref d) => Some(d.clone()),
            Nil => None,
        }
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
    ///
    /// Time: O(1)
    pub fn uncons(&self) -> Option<(Arc<A>, List<A>)> {
        match *self.0 {
            Nil => None,
            Cons(_, ref a, ref d) => Some((a.clone(), d.clone())),
        }
    }

    pub fn uncons2(&self) -> Option<(Arc<A>, Arc<A>, List<A>)> {
        self.uncons()
            .and_then(|(a1, d)| d.uncons().map(|(a2, d)| (a1, a2, d)))
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

    /// Append the list `right` to the end of the current list.
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
    ///   list![1, 2, 3].append(&list![7, 8, 9]),
    ///   list![1, 2, 3, 7, 8, 9]
    /// );
    /// # }
    /// ```
    pub fn append(&self, right: &List<A>) -> List<A> {
        match *self.0 {
            Nil => right.clone(),
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
    /// # use im::list::List;
    /// # fn main() {
    /// assert_eq!(
    ///   list![1, 2, 3, 4, 5].reverse(),
    ///   list![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    pub fn reverse(&self) -> List<A> {
        let mut out = List::new();
        for i in self.iter() {
            out = out.cons(i);
        }
        out
    }

    /// Get an iterator over a list.
    pub fn iter(&self) -> ListIter<A> {
        ListIter { current: self.clone() }
    }

    /// Sort a list using a comparator function.
    ///
    /// Time: O(n log n)
    pub fn sort_by<F>(&self, cmp: F) -> List<A>
        where F: Fn(Arc<A>, Arc<A>) -> Ordering
    {
        fn merge<A>(la: &List<A>, lb: &List<A>, cmp: &Fn(Arc<A>, Arc<A>) -> Ordering) -> List<A> {
            match (la.uncons(), lb.uncons()) {
                (Some((ref a, _)), Some((ref b, ref lb1))) if cmp(a.clone(), b.clone()) ==
                                                              Ordering::Greater => {
                    cons(b.clone(), &merge(la, &lb1, cmp))
                }
                (Some((a, la1)), Some((_, _))) => cons(a.clone(), &merge(&la1, lb, cmp)),
                (None, _) => lb.clone(),
                (_, None) => la.clone(),
            }
        }

        fn merge_pairs<A>(l: &List<List<A>>,
                          cmp: &Fn(Arc<A>, Arc<A>) -> Ordering)
                          -> List<List<A>> {
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

        fn ascending<A>(a: Arc<A>,
                        f: &Fn(List<A>) -> List<A>,
                        l: &List<A>,
                        cmp: &Fn(Arc<A>, Arc<A>) -> Ordering)
                        -> List<List<A>> {
            match l.uncons() {
                Some((ref b, ref lb)) if cmp(a.clone(), b.clone()) != Ordering::Greater => {
                    ascending(b.clone(), &|ys| f(cons(a.clone(), &ys)), &lb, cmp)
                }
                _ => cons(f(List::singleton(a.clone())), &sequences(l, cmp)),
            }
        }

        fn descending<A>(a: Arc<A>,
                         la: &List<A>,
                         lb: &List<A>,
                         cmp: &Fn(Arc<A>, Arc<A>) -> Ordering)
                         -> List<List<A>> {
            match lb.uncons() {
                Some((ref b, ref bs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(b.clone(), &cons(a.clone(), &la), bs, cmp)
                }
                _ => cons(cons(a.clone(), &la), &sequences(&lb, cmp)),
            }
        }

        fn sequences<A>(l: &List<A>, cmp: &Fn(Arc<A>, Arc<A>) -> Ordering) -> List<List<A>> {
            match l.uncons2() {
                Some((ref a, ref b, ref xs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(b.clone(), &List::singleton(a.clone()), xs, cmp)
                }
                Some((ref a, ref b, ref xs)) => {
                    ascending(b.clone(), &|l| cons(a.clone(), &l), &xs, cmp)
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

impl<A> List<A>
    where A: Ord
{
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
    pub fn insert<T>(&self, item: T) -> List<A>
        where Arc<A>: From<T>
    {
        self.insert_ref(Arc::from(item))
    }

    fn insert_ref(&self, item: Arc<A>) -> List<A> {
        match *self.0 {
            Nil => List(Arc::new(Cons(0, item, List::new()))),
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
    /// # use im::list::List;
    /// # fn main() {
    /// assert_eq!(
    ///   list![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   List::range(1, 8)
    /// );
    /// # }
    /// ```
    pub fn sort(&self) -> List<A> {
        self.sort_by(|a: Arc<A>, b: Arc<A>| a.cmp(&b))
    }
}

// Core traits

impl<A> Clone for List<A> {
    /// Clone a list.
    ///
    /// Cons cells use `Arc` behind the scenes, so this is no more
    /// expensive than cloning an `Arc` reference.
    ///
    /// Time: O(1)
    fn clone(&self) -> Self {
        match self {
            &List(ref node) => List(node.clone()),
        }
    }
}

impl<A> Default for List<A> {
    /// `Default` for lists is the empty list.
    fn default() -> Self {
        List::new()
    }
}

impl<A> PartialEq for List<A>
    where A: PartialEq
{
    /// Test if two lists are equal.
    ///
    /// This could potentially be an expensive operation, as we need to walk
    /// both lists to test for equality. We can very quickly determine equality
    /// if both lists are references to the same cons cell (they're equal) or if
    /// the lists have different lengths (can't be equal). Otherwise, we walk the
    /// lists to compare values.
    ///
    /// Time: O(n)
    fn eq(&self, other: &List<A>) -> bool {
        Arc::ptr_eq(&self.0, &other.0) ||
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }

    fn ne(&self, other: &List<A>) -> bool {
        !Arc::ptr_eq(&self.0, &other.0) &&
        (self.len() != other.len() || self.iter().zip(other.iter()).all(|(a, b)| a != b))
    }
}

impl<A> Eq for List<A> where A: Eq {}

impl<A> Hash for List<A>
    where A: Hash
{
    fn hash<H>(&self, state: &mut H)
        where H: Hasher
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A> Debug for List<A>
    where A: Debug
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        fn items<A>(l: &List<A>, f: &mut Formatter) -> Result<(), Error>
            where A: Debug
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

pub struct ListIter<A> {
    #[doc(hidden)]
    current: List<A>,
}

impl<A> Iterator for ListIter<A> {
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

impl<A> ExactSizeIterator for ListIter<A> {}

impl<A> IntoIterator for List<A> {
    type Item = Arc<A>;
    type IntoIter = ListIter<A>;

    fn into_iter(self) -> ListIter<A> {
        self.iter()
    }
}

impl<A, T> FromIterator<T> for List<A>
    where Arc<A>: From<T>
{
    fn from_iter<I>(source: I) -> Self
        where I: IntoIterator<Item = T>
    {
        source.into_iter().fold(list![], |l, v| l.cons(v)).reverse()
    }
}

// Conversions

impl<'a, A, R> From<&'a [R]> for List<A>
    where Arc<A>: From<&'a R>
{
    fn from(slice: &'a [R]) -> Self {
        slice.into_iter().map(|a| Arc::from(a)).collect()
    }
}

impl<A, R> From<Vec<R>> for List<A>
    where Arc<A>: From<R>
{
    fn from(vec: Vec<R>) -> Self {
        vec.into_iter().map(|a| Arc::from(a)).collect()
    }
}

impl<'a, A, R> From<&'a Vec<R>> for List<A>
    where Arc<A>: From<&'a R>
{
    fn from(vec: &'a Vec<R>) -> Self {
        vec.into_iter().map(|a| Arc::from(a)).collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for List<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        List::from(Vec::<A>::arbitrary(g))
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use test::is_sorted;

    #[test]
    fn exact_size_iterator() {
        assert_eq!(10, List::range(1, 10).iter().len());
    }

    #[test]
    fn collect_from_iterator() {
        let o: List<i32> = vec![5, 6, 7].iter().cloned().collect();
        assert_eq!(o, list![5, 6, 7]);
    }

    #[test]
    fn equality() {
        let l = List::range(1, 5);
        assert_eq!(l, l);
        assert_eq!(l, list![1, 2, 3, 4, 5]);
    }

    #[test]
    fn disequality() {
        let l = List::range(1, 5);
        assert_ne!(l, cons(0, &l));
        assert_ne!(l, list![1, 2, 3, 4, 5, 6]);
    }

    quickcheck! {
        fn length(vec: Vec<i32>) -> bool {
            let list = List::from(vec.clone());
            vec.len() == list.len()
        }

        fn order(vec: Vec<i32>) -> bool {
            let list = List::from(vec.clone());
            list.iter().map(|a| *a).eq(vec.into_iter())
        }

        fn reverse_a_list(l: List<i32>) -> bool {
            let vec: Vec<i32> = l.iter().map(|v| *v).collect();
            let rev = List::from(vec.into_iter().rev());
            l.reverse() == rev
        }

        fn append_two_lists(xs: List<i32>, ys: List<i32>) -> bool {
            let extended = List::from(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            xs.append(&ys) == extended
        }

        fn sort_a_list(l: List<i32>) -> bool {
            let sorted = l.sort();
            l.len() == sorted.len() && is_sorted(sorted)
        }
    }
}
