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

use std::iter::{Iterator, FromIterator};
use std::fmt::{Debug, Formatter, Error};
use std::ops::Deref;
use std::sync::Arc;
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
///   List::from_slice(&[1, 2, 3])
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

/// A list of elements of type A.
pub enum List<A> {
    #[doc(hidden)]
    List(Arc<ListNode<A>>),
}

#[doc(hidden)]
pub enum ListNode<A> {
    Cons(usize, A, List<A>),
    Nil,
}

impl<A> List<A> {
    /// Construct an empty list.
    pub fn empty() -> List<A> {
        List::List(Arc::new(Nil))
    }

    /// Construct a list with a single element.
    pub fn singleton(v: A) -> List<A> {
        List::List(Arc::new(Cons(1, v, list![])))
    }

    fn as_arc<'a>(&'a self) -> &'a Arc<ListNode<A>> {
        match self {
            &List::List(ref arc) => arc,
        }
    }

    /// Test whether a list is empty.
    ///
    /// Time: O(1)
    pub fn null(&self) -> bool {
        match self.as_arc().deref() {
            &Nil => true,
            _ => false,
        }
    }

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    ///
    /// Time: O(1)
    pub fn cons(&self, car: A) -> List<A> {
        match self.as_arc().deref() {
            &Nil => List::singleton(car),
            &Cons(l, _, _) => List::List(Arc::new(Cons(l + 1, car, self.clone()))),
        }
    }

    /// Get the first element of a list.
    ///
    /// If the list is empty, `None` is returned.
    ///
    /// Time: O(1)
    pub fn head<'a>(&'a self) -> Option<&'a A> {
        match self.as_arc().deref() {
            &Cons(_, ref a, _) => Some(a),
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
        match self.as_arc().deref() {
            &Cons(_, _, ref d) => Some(d.clone()),
            &Nil => None,
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
    pub fn uncons<'a>(&'a self) -> Option<(&'a A, List<A>)> {
        match self.as_arc().deref() {
            &Nil => None,
            &Cons(_, ref a, ref d) => Some((a, d.clone())),
        }
    }

    pub fn uncons2<'a>(&'a self) -> Option<(&'a A, &'a A, List<A>)> {
        match self.as_arc().deref() {
            &Nil => None,
            &Cons(_, ref a, ref d) => {
                match d.as_arc().deref() {
                    &Nil => None,
                    &Cons(_, ref ad, ref dd) => Some((a, ad, dd.clone())),
                }
            }
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
    /// assert_eq!(5, list![1, 2, 3, 4, 5].length());
    /// # }
    /// ```
    pub fn length(&self) -> usize {
        match self.as_arc().deref() {
            &Nil => 0,
            &Cons(l, _, _) => l,
        }
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
    where A: Clone + Ord
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
    ///   list![2, 4, 6].insert(&5).insert(&1).insert(&3),
    ///   list![1, 2, 3, 4, 5, 6]
    /// );
    /// # }
    /// ```
    pub fn insert(&self, item: &A) -> List<A> {
        match self.as_arc().deref() {
            &Nil => List::singleton(item.clone()),
            &Cons(_, ref a, ref d) => {
                if a > item {
                    cons(item.clone(), self)
                } else {
                    cons(a.clone(), &d.insert(item))
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
        self.sort_by(&|a, b| a.cmp(b))
    }
}

impl<A> List<A>
    where A: Clone
{
    /// Construct a list from a slice of items.
    ///
    /// Time: O(n)
    pub fn from_slice(slice: &[A]) -> List<A> {
        From::from(slice)
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
        match self.as_arc().deref() {
            &Nil => right.as_ref().clone(),
            &Cons(_, ref a, ref d) => cons(a.clone(), &d.append(right.as_ref())),
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
            out = out.cons(i);
        }
        out
    }

    /// Get an iterator over a list.
    pub fn iter(&self) -> ListIter<A> {
        ListIter { current: self.as_arc().clone() }
    }

    /// Sort a list using a comparator function.
    ///
    /// Time: O(n log n)
    pub fn sort_by(&self, cmp: &Fn(&A, &A) -> Ordering) -> List<A> {
        fn merge<A>(la: &List<A>, lb: &List<A>, cmp: &Fn(&A, &A) -> Ordering) -> List<A>
            where A: Clone
        {
            match (la.uncons(), lb.uncons()) {
                (Some((a, _)), Some((b, ref lb1))) if cmp(a, b) == Ordering::Greater => {
                    cons(b.clone(), &merge(la, &lb1, cmp))
                }
                (Some((a, la1)), Some((_, _))) => cons(a.clone(), &merge(&la1, lb, cmp)),
                (None, _) => lb.clone(),
                (_, None) => la.clone(),
            }
        }

        fn merge_pairs<A>(l: &List<List<A>>, cmp: &Fn(&A, &A) -> Ordering) -> List<List<A>>
            where A: Clone
        {
            match l.uncons2() {
                Some((a, b, rest)) => cons(merge(a, b, cmp), &merge_pairs(&rest, cmp)),
                _ => l.clone(),
            }
        }

        fn merge_all<A>(l: &List<List<A>>, cmp: &Fn(&A, &A) -> Ordering) -> List<A>
            where A: Clone
        {
            match l.uncons() {
                None => list![],
                Some((a, ref d)) if d.null() => a.clone(),
                _ => merge_all(&merge_pairs(l, cmp), cmp),
            }
        }

        fn ascending<A>(a: &A,
                        f: &Fn(List<A>) -> List<A>,
                        l: &List<A>,
                        cmp: &Fn(&A, &A) -> Ordering)
                        -> List<List<A>>
            where A: Clone
        {
            match l.uncons() {
                Some((b, ref lb)) if cmp(a, b) != Ordering::Greater => {
                    ascending(b, &|ys| f(cons(a.clone(), &ys)), &lb, cmp)
                }
                _ => cons(f(list![a.clone()]), &sequences(l, cmp)),
            }
        }

        fn descending<A>(a: &A,
                         la: &List<A>,
                         lb: &List<A>,
                         cmp: &Fn(&A, &A) -> Ordering)
                         -> List<List<A>>
            where A: Clone
        {
            match lb.uncons() {
                Some((b, ref bs)) if cmp(a, b) == Ordering::Greater => {
                    descending(b, &cons(a.clone(), &la), bs, cmp)
                }
                _ => cons(cons(a.clone(), &la), &sequences(&lb, cmp)),
            }
        }

        fn sequences<A>(l: &List<A>, cmp: &Fn(&A, &A) -> Ordering) -> List<List<A>>
            where A: Clone
        {
            match l.uncons2() {
                Some((a, b, ref xs)) if cmp(a, b) == Ordering::Greater => {
                    descending(b, &list![a.clone()], xs, cmp)
                }
                Some((a, b, ref xs)) => ascending(b, &|l| cons(a.clone(), &l), &xs, cmp),
                None => list![l.clone()],
            }
        }

        merge_all(&sequences(self, cmp), cmp)
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
            &List::List(ref arc) => List::List(arc.clone()),
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
    current: Arc<ListNode<A>>,
}

impl<A> Iterator for ListIter<A>
    where A: Clone
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current.clone().deref() {
            &Nil => None,
            &Cons(_, ref a, ref d) => {
                self.current = d.as_arc().clone();
                Some(a.clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.current.deref() {
            &Nil => (0, Some(0)),
            &Cons(l, _, _) => (l, Some(l)),
        }
    }
}

impl<A> ExactSizeIterator for ListIter<A> where A: Clone {}

impl<A> IntoIterator for List<A>
    where A: Clone
{
    type Item = A;
    type IntoIter = ListIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> FromIterator<A> for List<A>
    where A: Clone
{
    fn from_iter<T>(iter: T) -> Self
        where T: IntoIterator<Item = A>
    {
        let mut l = List::empty();
        for i in iter {
            l = cons(i, &l)
        }
        l.reverse()
    }
}

impl<'a, A> FromIterator<&'a A> for List<A>
    where A: 'a + Clone
{
    fn from_iter<T>(iter: T) -> Self
        where T: IntoIterator<Item = &'a A>
    {
        let mut l = List::empty();
        for i in iter {
            l = cons(i.clone(), &l)
        }
        l.reverse()
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
    where A: PartialEq + Clone
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
    #[cfg(not(feature = "ptr_eq"))]
    fn eq(&self, other: &List<A>) -> bool {
        self.length() == other.length() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }

    #[cfg(feature = "ptr_eq")]
    fn eq(&self, other: &List<A>) -> bool {
        Arc::ptr_eq(self.as_arc(), other.as_arc()) ||
        self.length() == other.length() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }

    #[cfg(not(feature = "ptr_eq"))]
    fn ne(&self, other: &List<A>) -> bool {
        self.length() != other.length() || self.iter().zip(other.iter()).all(|(a, b)| a != b)
    }

    #[cfg(feature = "ptr_eq")]
    fn ne(&self, other: &List<A>) -> bool {
        !Arc::ptr_eq(self.as_arc(), other.as_arc()) &&
        (self.length() != other.length() || self.iter().zip(other.iter()).all(|(a, b)| a != b))
    }
}

impl<A> Eq for List<A> where A: Eq + Clone {}

impl<A> Hash for List<A>
    where A: Clone + Hash
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
            match l.as_arc().deref() {
                &Nil => Ok(()),
                &Cons(_, ref a, ref d) => {
                    write!(f, ", {:?}", a)?;
                    items(d, f)
                }
            }
        }
        write!(f, "[")?;
        match self.as_arc().deref() {
            &Nil => Ok(()),
            &Cons(_, ref a, ref d) => {
                write!(f, "{:?}", a)?;
                items(d, f)
            }
        }?;
        write!(f, "]")
    }
}



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
