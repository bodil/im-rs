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
//! Items in the list generally need to implement `Clone`,
//! because the shared structure makes it impossible to
//! determine the lifetime of a reference inside the list.
//! When cloning values would be too expensive,
//! use `List<Rc<T>>` or `List<Arc<T>>`.

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
///   cons(1, &cons(2, &cons(3, &List::empty())))
/// );
/// # }
/// ```
#[macro_export]
macro_rules! list {
    () => { $crate::list::List::empty() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::list::List::empty();
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
///   cons(1, &cons(2, &cons(3, &List::empty()))),
///   list![1, 2, 3]
/// );
/// # }
/// ```
pub fn cons<A>(car: A, cdr: &List<A>) -> List<A> {
    cdr.cons(car)
}

pub fn cons_ref<A>(car: Arc<A>, cdr: &List<A>) -> List<A> {
    cdr.cons_ref(car)
}

/// A list of elements of type A.
pub struct List<A>(Arc<ListNode<A>>);

#[doc(hidden)]
#[derive(Clone)]
pub enum ListNode<A> {
    Cons(usize, Arc<A>, List<A>),
    Nil,
}

impl<A> List<A> {
    /// Construct an empty list.
    pub fn empty() -> List<A> {
        List(Arc::new(Nil))
    }

    /// Construct a list with a single element.
    pub fn singleton(v: A) -> List<A> {
        List(Arc::new(Cons(1, Arc::new(v), list![])))
    }

    pub fn singleton_ref(v: Arc<A>) -> List<A> {
        List(Arc::new(Cons(1, v, list![])))
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
    pub fn from<I: IntoIterator<Item = A>>(it: I) -> List<A> {
        it.into_iter().collect()
    }

    /// Test whether a list is empty.
    ///
    /// Time: O(1)
    pub fn null(&self) -> bool {
        match *self.0 {
            Nil => true,
            _ => false,
        }
    }

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    ///
    /// Time: O(1)
    pub fn cons(&self, car: A) -> List<A> {
        List(Arc::new(Cons(self.length() + 1, Arc::new(car), self.clone())))
    }

    pub fn cons_ref(&self, car: Arc<A>) -> List<A> {
        List(Arc::new(Cons(self.length() + 1, car, self.clone())))
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
    /// assert_eq!(5, list![1, 2, 3, 4, 5].length());
    /// # }
    /// ```
    pub fn length(&self) -> usize {
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
            Nil => right.as_ref().clone(),
            Cons(_, ref a, ref d) => cons_ref(a.clone(), &d.append(right.as_ref())),
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
        let mut out = List::empty();
        for i in self.iter() {
            out = out.cons_ref(i);
        }
        out
    }

    /// Get an iterator over a list.
    pub fn iter(&self) -> ListIter<A> {
        ListIter { current: self.clone() }
    }

    pub fn clone_iter(&self) -> ListCloneIter<A> {
        ListCloneIter { it: self.iter() }
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
                    cons_ref(b.clone(), &merge(la, &lb1, cmp))
                }
                (Some((a, la1)), Some((_, _))) => cons_ref(a.clone(), &merge(&la1, lb, cmp)),
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
                Some((ref a, ref d)) if d.null() => a.deref().clone(),
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
                    ascending(b.clone(), &|ys| f(cons_ref(a.clone(), &ys)), &lb, cmp)
                }
                _ => cons(f(List::singleton_ref(a.clone())), &sequences(l, cmp)),
            }
        }

        fn descending<A>(a: Arc<A>,
                         la: &List<A>,
                         lb: &List<A>,
                         cmp: &Fn(Arc<A>, Arc<A>) -> Ordering)
                         -> List<List<A>> {
            match lb.uncons() {
                Some((ref b, ref bs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(b.clone(), &cons_ref(a.clone(), &la), bs, cmp)
                }
                _ => cons(cons_ref(a.clone(), &la), &sequences(&lb, cmp)),
            }
        }

        fn sequences<A>(l: &List<A>, cmp: &Fn(Arc<A>, Arc<A>) -> Ordering) -> List<List<A>> {
            match l.uncons2() {
                Some((ref a, ref b, ref xs)) if cmp(a.clone(), b.clone()) == Ordering::Greater => {
                    descending(b.clone(), &List::singleton_ref(a.clone()), xs, cmp)
                }
                Some((ref a, ref b, ref xs)) => {
                    ascending(b.clone(), &|l| cons_ref(a.clone(), &l), &xs, cmp)
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
        let mut list = List::empty();
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
    pub fn insert(&self, item: A) -> List<A> {
        match *self.0 {
            Nil => List::singleton(item),
            Cons(_, ref a, ref d) => {
                if a.as_ref() > &item {
                    cons(item, self)
                } else {
                    cons_ref(a.clone(), &d.insert(item))
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
        List::empty()
    }
}

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
        match *self.current.0 {
            Nil => (0, Some(0)),
            Cons(l, _, _) => (l, Some(l)),
        }
    }
}

impl<A> ExactSizeIterator for ListIter<A> where A: Clone {}

pub struct ListCloneIter<A> {
    #[doc(hidden)]
    it: ListIter<A>,
}

impl<A: Clone> Iterator for ListCloneIter<A> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|a| a.deref().clone())
    }
}

impl<A> IntoIterator for List<A> {
    type Item = Arc<A>;
    type IntoIter = ListIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> FromIterator<A> for List<A> {
    fn from_iter<I>(source: I) -> Self
        where I: IntoIterator<Item = A>
    {
        let mut input: Vec<A> = source.into_iter().collect();
        input.reverse();
        let mut l = List::empty();
        for i in input.into_iter() {
            l = cons(i, &l)
        }
        l
    }
}

impl<'a, A> FromIterator<&'a A> for List<A>
    where A: 'a + Clone
{
    fn from_iter<I>(source: I) -> Self
        where I: IntoIterator<Item = &'a A>
    {
        source.into_iter().cloned().collect()
    }
}

impl<'a, A> From<&'a [A]> for List<A>
    where A: Clone
{
    fn from(slice: &'a [A]) -> List<A> {
        slice.iter().cloned().collect()
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
        self.length() == other.length() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }

    fn ne(&self, other: &List<A>) -> bool {
        !Arc::ptr_eq(&self.0, &other.0) &&
        (self.length() != other.length() || self.iter().zip(other.iter()).all(|(a, b)| a != b))
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

impl<A> AsRef<List<A>> for List<A> {
    fn as_ref(&self) -> &List<A> {
        self
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

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for List<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        List::from(Vec::arbitrary(g))
    }
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_coercion() {
        assert_eq!(list![1, 2, 3], From::from(&[1, 2, 3][..]));
    }

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

    fn is_sorted<A: Ord + Clone>(l: &List<A>) -> bool {
        if l.length() == 0 {
            true
        } else {
            let mut it = l.iter();
            let mut prev = it.next().unwrap();
            loop {
                match it.next() {
                    None => return true,
                    Some(ref i) if i < &prev => return false,
                    Some(ref i) => prev = i.clone(),
                }
            }
        }
    }

    quickcheck! {
        fn reverse_a_list(l: List<i32>) -> bool {
            let vec: Vec<i32> = l.clone_iter().collect();
            let rev = List::from(vec.into_iter().rev());
            l.reverse() == rev
        }

        fn append_two_lists(xs: List<i32>, ys: List<i32>) -> bool {
            let extended = List::from(xs.clone_iter().chain(ys.clone_iter()));
            xs.append(&ys) == extended
        }

        fn sort_a_list(l: List<i32>) -> bool {
            let sorted = l.sort();
            l.length() == sorted.length() && is_sorted(&sorted)
        }
    }
}
