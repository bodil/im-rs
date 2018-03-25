//! A catenable list.
//!
//! A list data structure with O(1)* push and pop operations on both ends and
//! O(1) concatenation of lists.
//!
//! You usually want the [`Vector`][vector::Vector] instead, which performs better
//! on all operations except concatenation. If fast concatenation is what you need,
//! the `CatList` is the cat for you.
//!
//! [queue::Queue]: ../queue/struct.Queue.html
//! [vector::Vector]: ../vector/struct.Vector.html
//! [conslist::ConsList]: ../conslist/struct.ConsList.html

use std::sync::Arc;
use std::iter::{FromIterator, Sum};
use std::ops::{Add, Deref};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Error, Formatter};
use std::borrow::Borrow;
use vector::Vector;
use shared::Shared;
use bits::HASH_SIZE;

/// Construct a list from a sequence of elements.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::catlist::{CatList, cons};
/// # fn main() {
/// assert_eq!(
///   catlist![1, 2, 3],
///   CatList::from(vec![1, 2, 3])
/// );
///
/// assert_eq!(
///   catlist![1, 2, 3],
///   cons(1, cons(2, cons(3, CatList::new())))
/// );
/// # }
/// ```
#[macro_export]
macro_rules! catlist {
    () => { $crate::catlist::CatList::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::catlist::CatList::new();
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
/// it much easier to read `cons(1, cons(2, CatList::new()))`
/// than `CatList::new().cons(2).cons(1)`, given that the resulting
/// list will be `[1, 2]`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::catlist::{CatList, cons};
/// # fn main() {
/// assert_eq!(
///   cons(1, cons(2, cons(3, CatList::new()))),
///   catlist![1, 2, 3]
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
#[inline]
pub fn cons<A, RA, RD>(car: RA, cdr: RD) -> CatList<A>
where
    RA: Shared<A>,
    RD: Borrow<CatList<A>>,
{
    cdr.borrow().cons(car)
}

/// A catenable list of values of type `A`.
///
/// A list data structure with O(1)* push and pop operations on both ends and
/// O(1) concatenation of lists.
///
/// You usually want the [`Vector`][vector::Vector] instead, which performs better
/// on all operations except concatenation. If fast concatenation is what you need,
/// the `CatList` is the cat for you.
///
/// [queue::Queue]: ../queue/struct.Queue.html
/// [vector::Vector]: ../vector/struct.Vector.html
/// [conslist::ConsList]: ../conslist/struct.ConsList.html
pub struct CatList<A>(Arc<ListNode<A>>);

#[doc(hidden)]
pub struct ListNode<A> {
    size: usize,
    head: Arc<Vec<Arc<A>>>,
    tail: Vector<CatList<A>>,
}

impl<A> ListNode<A> {
    fn new() -> Self {
        ListNode {
            size: 0,
            head: Arc::new(Vec::new()),
            tail: Vector::new(),
        }
    }
}

impl<A> Clone for ListNode<A> {
    fn clone(&self) -> Self {
        ListNode {
            size: self.size,
            head: self.head.clone(),
            tail: self.tail.clone(),
        }
    }
}

impl<A> CatList<A> {
    /// Construct an empty list.
    pub fn new() -> Self {
        CatList(Arc::new(ListNode::new()))
    }

    /// Construct a list with a single value.
    pub fn singleton<R>(a: R) -> Self
    where
        R: Shared<A>,
    {
        CatList::from_head(vec![a.shared()])
    }

    fn from_head(head: Vec<Arc<A>>) -> Self {
        CatList(Arc::new(ListNode {
            size: head.len(),
            head: Arc::new(head),
            tail: Vector::new(),
        }))
    }

    fn make<VA: Shared<Vec<Arc<A>>>>(size: usize, head: VA, tail: Vector<CatList<A>>) -> Self {
        CatList(Arc::new(ListNode {
            size,
            head: head.shared(),
            tail,
        }))
    }

    /// Construct a list by consuming an [`IntoIterator`][std::iter::IntoIterator].
    ///
    /// Allows you to construct a list out of anything that implements
    /// the [`IntoIterator`][std::iter::IntoIterator] trait.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::CatList;
    /// # fn main() {
    /// assert_eq!(
    ///   CatList::from(vec![1, 2, 3, 4, 5]),
    ///   catlist![1, 2, 3, 4, 5]
    /// );
    /// # }
    /// ```
    ///
    /// [std::iter::IntoIterator]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
    pub fn from<R, I>(it: I) -> CatList<A>
    where
        I: IntoIterator<Item = R>,
        R: Shared<A>,
    {
        it.into_iter().collect()
    }

    /// Test whether a list is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the length of a list.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(5, catlist![1, 2, 3, 4, 5].len());
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.0.size
    }

    /// Get the first element of a list.
    ///
    /// If the list is empty, `None` is returned.
    pub fn head(&self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else {
            self.0.head.last().cloned()
        }
    }

    /// Get the last element of a list, as well as the list with the last
    /// element removed.
    ///
    /// If the list is empty, [`None`][None] is returned.
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn pop_back(&self) -> Option<(Arc<A>, CatList<A>)> {
        if self.is_empty() {
            None
        } else if self.0.tail.is_empty() {
            Some((
                self.0.head.first().unwrap().clone(),
                CatList::from_head(self.0.head.iter().skip(1).cloned().collect()),
            ))
        } else {
            match self.0.tail.pop_back() {
                None => unreachable!(),
                Some((last_list, queue_without_last_list)) => match last_list.pop_back() {
                    None => unreachable!(),
                    Some((last_item, list_without_last_item)) => {
                        let new_node = ListNode {
                            size: self.0.size - last_list.len(),
                            head: self.0.head.clone(),
                            tail: queue_without_last_list,
                        };
                        Some((
                            last_item,
                            CatList(Arc::new(new_node)).append(list_without_last_item),
                        ))
                    }
                },
            }
        }
    }

    /// Get the last element of a list.
    ///
    /// If the list is empty, `None` is returned.
    pub fn last(&self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else if self.0.tail.is_empty() {
            self.0.head.first().cloned()
        } else {
            self.0.tail.last().unwrap().last()
        }
    }

    /// Get the list without the last element.
    ///
    /// If the list is empty, `None` is returned.
    pub fn init(&self) -> Option<CatList<A>> {
        self.pop_back().map(|a| a.1)
    }

    /// Get the tail of a list.
    ///
    /// The tail means all elements in the list after the
    /// first item (the head). If the list only has one
    /// element, the result is an empty list. If the list is
    /// empty, the result is `None`.
    pub fn tail(&self) -> Option<Self> {
        if self.is_empty() {
            None
        } else if self.len() == 1 {
            Some(CatList::new())
        } else if self.0.tail.is_empty() {
            Some(CatList::from_head(
                self.0
                    .head
                    .iter()
                    .take(self.0.head.len() - 1)
                    .cloned()
                    .collect(),
            ))
        } else if self.0.head.len() > 1 {
            Some(CatList::make(
                self.len() - 1,
                Arc::new(
                    self.0
                        .head
                        .iter()
                        .take(self.0.head.len() - 1)
                        .cloned()
                        .collect(),
                ),
                self.0.tail.clone(),
            ))
        } else {
            Some(self.0.tail.iter().fold(CatList::new(), |a, b| a.append(b)))
        }
    }

    /// Append the list `other` to the end of the current list.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::CatList;
    /// # fn main() {
    /// assert_eq!(
    ///   catlist![1, 2, 3].append(catlist![7, 8, 9]),
    ///   catlist![1, 2, 3, 7, 8, 9]
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
            (l, r) => {
                if l.0.tail.is_empty() && l.0.head.len() + r.0.head.len() <= HASH_SIZE {
                    let mut new_head = (*r.0.head).clone();
                    new_head.extend(l.0.head.iter().cloned());
                    CatList::make(l.len() + r.len(), new_head, r.0.tail.clone())
                } else if !l.0.tail.is_empty() && l.0.tail.last().unwrap().0.tail.is_empty()
                    && l.0.tail.last().unwrap().0.head.len() + r.0.head.len() <= HASH_SIZE
                {
                    let (last, tail_but_last) = l.0.tail.pop_back().unwrap();
                    let mut new_head = (*r.0.head).clone();
                    new_head.extend(last.0.head.iter().cloned());
                    let last_plus_right =
                        CatList::make(last.len() + r.len(), new_head, r.0.tail.clone());
                    CatList::make(
                        l.len() + r.len(),
                        l.0.head.clone(),
                        tail_but_last.push_back(last_plus_right),
                    )
                } else {
                    CatList::make(
                        l.len() + r.len(),
                        self.0.head.clone(),
                        self.0.tail.push_back(r.clone()),
                    )
                }
            }
        }
    }

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    pub fn cons<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        CatList::singleton(a).append(self)
    }

    /// Construct a list with a new value prepended to the front of the
    /// current list.
    #[inline]
    pub fn push_front<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        self.cons(a)
    }

    pub fn append_mut<R>(&mut self, other_ref: R)
    where
        R: Borrow<Self>,
    {
        let other = other_ref.borrow();
        if other.is_empty() {
            return;
        } else if self.is_empty() {
            self.0 = other.0.clone();
        } else if self.0.tail.is_empty() && self.0.head.len() + other.0.head.len() <= HASH_SIZE {
            let node = Arc::make_mut(&mut self.0);
            node.head = Arc::new(
                other
                    .0
                    .head
                    .iter()
                    .chain(node.head.iter())
                    .cloned()
                    .collect(),
            );
            node.tail = Vector::singleton(other.clone());
            node.size += other.len();
        } else {
            let node = Arc::make_mut(&mut self.0);
            node.tail.push_back_mut(other.borrow().clone());
            node.size += other.len();
        }
    }

    pub fn push_front_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        if self.0.head.len() >= HASH_SIZE {
            let next = CatList(self.0.clone());
            self.0 = Arc::new(ListNode {
                size: 1,
                head: Arc::new(vec![a.shared()]),
                tail: Vector::new(),
            });
            self.append_mut(next);
        } else {
            let node = Arc::make_mut(&mut self.0);
            let head = Arc::make_mut(&mut node.head);
            head.push(a.shared());
            node.size += 1;
        }
    }

    pub fn push_back_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        if self.0.tail.is_empty() && self.0.head.len() < HASH_SIZE {
            let node = Arc::make_mut(&mut self.0);
            let head = Arc::make_mut(&mut node.head);
            head.insert(0, a.shared());
            node.size += 1;
        } else {
            self.append_mut(CatList::singleton(a))
        }
    }

    pub fn pop_front_mut(&mut self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else if self.0.head.len() > 1 {
            let node = Arc::make_mut(&mut self.0);
            let head = Arc::make_mut(&mut node.head);
            let item = head.pop();
            node.size -= 1;
            item
        } else {
            let item = self.0.head.last().cloned();
            let tail = self.0.tail.clone();
            self.0 = Arc::new(ListNode {
                size: 0,
                head: Arc::new(Vec::new()),
                tail: Vector::new(),
            });
            for list in tail {
                self.append_mut(list);
            }
            item
        }
    }

    pub fn pop_back_mut(&mut self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else if self.0.tail.is_empty() {
            let node = Arc::make_mut(&mut self.0);
            node.size -= 1;
            let head = Arc::make_mut(&mut node.head);
            Some(head.remove(0))
        } else {
            let node = Arc::make_mut(&mut self.0);
            node.size -= 1;
            let mut last = node.tail.pop_back_mut().unwrap();
            let last_node = Arc::make_mut(&mut last);
            let item = last_node.pop_back_mut();
            if !last_node.is_empty() {
                node.tail.push_back_mut(last_node.clone());
            }
            item
        }
    }

    /// Construct a list with a new value appended to the back of the
    /// current list.
    ///
    /// `snoc`, for the curious, is [`cons`][cons] spelled backwards, to denote
    /// that it works on the back of the list rather than the front.
    /// If you don't find that as clever as whoever coined the term no
    /// doubt did, this method is also available as [`push_back()`][push_back].
    ///
    /// [push_back]: #method.push_back
    /// [cons]: #method.cons
    pub fn snoc<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        self.append(CatList::singleton(a))
    }

    /// Construct a list with a new value appended to the back of the
    /// current list.
    #[inline]
    pub fn push_back<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        self.snoc(a)
    }

    /// Get the head and the tail of a list.
    ///
    /// This function performs both the [`head`][head] function and
    /// the [`tail`][tail] function in one go, returning a tuple
    /// of the head and the tail, or [`None`][None] if the list is
    /// empty.
    ///
    /// # Examples
    ///
    /// This can be useful when pattern matching your way through
    /// a list:
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::{CatList, cons};
    /// # use std::fmt::Debug;
    /// fn walk_through_list<A>(list: &CatList<A>) where A: Debug {
    ///     match list.pop_front() {
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
    /// [head]: #method.head
    /// [tail]: #method.tail
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn pop_front(&self) -> Option<(Arc<A>, CatList<A>)> {
        self.head().and_then(|h| self.tail().map(|t| (h, t)))
    }

    /// Get the head and the tail of a list.
    ///
    /// This is an alias for [`pop_front`][pop_front].
    ///
    /// [pop_front]: #method.pop_front
    #[inline]
    pub fn uncons(&self) -> Option<(Arc<A>, CatList<A>)> {
        self.pop_front()
    }

    pub fn uncons2(&self) -> Option<(Arc<A>, Arc<A>, CatList<A>)> {
        self.uncons()
            .and_then(|(a1, d)| d.uncons().map(|(a2, d)| (a1, a2, d)))
    }

    /// Get an iterator over a list.
    #[inline]
    pub fn iter(&self) -> Iter<A> {
        Iter {
            current: self.clone(),
        }
    }

    /// Construct a list which is the reverse of the current list.
    ///
    /// Please note that if all you want is to iterate over the list from back to front,
    /// it is much more efficient to use a [reversed iterator][rev] rather than doing
    /// the work of reversing the list first.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::CatList;
    /// # fn main() {
    /// assert_eq!(
    ///   catlist![1, 2, 3, 4, 5].reverse(),
    ///   catlist![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    ///
    /// [rev]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.rev
    pub fn reverse(&self) -> Self {
        let mut out = CatList::new();
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
        F: Fn(&A, &A) -> Ordering,
    {
        fn merge<A>(la: &CatList<A>, lb: &CatList<A>, cmp: &Fn(&A, &A) -> Ordering) -> CatList<A> {
            match (la.uncons(), lb.uncons()) {
                (Some((ref a, _)), Some((ref b, ref lb1))) if cmp(a, b) == Ordering::Greater => {
                    cons(b.clone(), &merge(la, lb1, cmp))
                }
                (Some((a, la1)), Some((_, _))) => cons(a.clone(), &merge(&la1, lb, cmp)),
                (None, _) => lb.clone(),
                (_, None) => la.clone(),
            }
        }

        fn merge_pairs<A>(
            l: &CatList<CatList<A>>,
            cmp: &Fn(&A, &A) -> Ordering,
        ) -> CatList<CatList<A>> {
            match l.uncons2() {
                Some((a, b, rest)) => cons(merge(&a, &b, cmp), &merge_pairs(&rest, cmp)),
                _ => l.clone(),
            }
        }

        fn merge_all<A>(l: &CatList<CatList<A>>, cmp: &Fn(&A, &A) -> Ordering) -> CatList<A> {
            match l.uncons() {
                None => catlist![],
                Some((ref a, ref d)) if d.is_empty() => a.deref().clone(),
                _ => merge_all(&merge_pairs(l, cmp), cmp),
            }
        }

        fn ascending<A>(
            a: &Arc<A>,
            f: &Fn(CatList<A>) -> CatList<A>,
            l: &CatList<A>,
            cmp: &Fn(&A, &A) -> Ordering,
        ) -> CatList<CatList<A>> {
            match l.uncons() {
                Some((ref b, ref lb)) if cmp(a, b) != Ordering::Greater => {
                    ascending(&b.clone(), &|ys| f(cons(a.clone(), &ys)), lb, cmp)
                }
                _ => cons(f(CatList::singleton(a.clone())), &sequences(l, cmp)),
            }
        }

        fn descending<A>(
            a: &Arc<A>,
            la: &CatList<A>,
            lb: &CatList<A>,
            cmp: &Fn(&A, &A) -> Ordering,
        ) -> CatList<CatList<A>> {
            match lb.uncons() {
                Some((ref b, ref bs)) if cmp(a, b) == Ordering::Greater => {
                    descending(&b.clone(), &cons(a.clone(), la), bs, cmp)
                }
                _ => cons(cons(a.clone(), la), &sequences(lb, cmp)),
            }
        }

        fn sequences<A>(l: &CatList<A>, cmp: &Fn(&A, &A) -> Ordering) -> CatList<CatList<A>> {
            match l.uncons2() {
                Some((ref a, ref b, ref xs)) if cmp(a, b) == Ordering::Greater => {
                    descending(&b.clone(), &CatList::singleton(a.clone()), xs, cmp)
                }
                Some((ref a, ref b, ref xs)) => {
                    ascending(&b.clone(), &|l| cons(a.clone(), l), xs, cmp)
                }
                None => catlist![l.clone()],
            }
        }

        merge_all(&sequences(self, &cmp), &cmp)
    }

    /// Sort a list of ordered elements.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::CatList;
    /// # fn main() {
    /// assert_eq!(
    ///   catlist![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   CatList::range(1, 8)
    /// );
    /// # }
    /// ```
    pub fn sort(&self) -> Self
    where
        A: Ord,
    {
        self.sort_by(|a, b| a.cmp(b))
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
    ///   catlist![2, 4, 6].insert(5).insert(1).insert(3),
    ///   catlist![1, 2, 3, 4, 5, 6]
    /// );
    /// # }
    /// ```
    pub fn insert<T>(&self, item: T) -> Self
    where
        A: Ord,
        T: Shared<A>,
    {
        self.insert_ref(item.shared())
    }

    fn insert_ref(&self, item: Arc<A>) -> Self
    where
        A: Ord,
    {
        match self.uncons() {
            None => CatList::singleton(item),
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

impl CatList<i32> {
    /// Construct a list of numbers between `from` and `to` inclusive.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::{CatList, cons};
    /// # fn main() {
    /// assert_eq!(
    ///   CatList::range(1, 5),
    ///   catlist![1, 2, 3, 4, 5]
    /// );
    /// # }
    /// ```
    pub fn range(from: i32, to: i32) -> CatList<i32> {
        let mut list = CatList::new();
        let mut c = to;
        while c >= from {
            list = cons(c, &list);
            c -= 1;
        }
        list
    }
}

// Core traits

impl<A> Clone for CatList<A> {
    fn clone(&self) -> Self {
        CatList(self.0.clone())
    }
}

impl<A> Default for CatList<A> {
    fn default() -> Self {
        CatList::new()
    }
}

impl<A> Add for CatList<A> {
    type Output = CatList<A>;

    fn add(self, other: Self) -> Self::Output {
        self.append(&other)
    }
}

#[cfg(not(has_specialisation))]
impl<A: PartialEq> PartialEq for CatList<A> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: PartialEq> PartialEq for CatList<A> {
    default fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Eq> PartialEq for CatList<A> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<A: Eq> Eq for CatList<A> {}

impl<A: PartialOrd> PartialOrd for CatList<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord> Ord for CatList<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Hash> Hash for CatList<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self {
            i.hash(state)
        }
    }
}

impl<A: Debug> Debug for CatList<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        // write!(
        //     f,
        //     "[ Len {} Head {}:{:?} Tail {:?} ]",
        //     self.0.size,
        //     self.0.head.len(),
        //     self.0.head,
        //     self.0.tail
        // )
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
    current: CatList<A>,
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        // FIXME immutable ops are slower than necessary here,
        // how about a good old fashioned incrementing pointer?
        match self.current.pop_front() {
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

impl<A> DoubleEndedIterator for Iter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.current.pop_back() {
            None => None,
            Some((a, q)) => {
                self.current = q;
                Some(a)
            }
        }
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

impl<A> IntoIterator for CatList<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        Iter { current: self }
    }
}

impl<'a, A> IntoIterator for &'a CatList<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> Sum for CatList<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, T> FromIterator<T> for CatList<A>
where
    T: Shared<A>,
{
    fn from_iter<I>(source: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        source.into_iter().fold(Self::new(), |l, i| l.push_back(i))
    }
}

// Conversions

impl<'a, A, T> From<&'a [T]> for CatList<A>
where
    &'a T: Shared<A>,
{
    fn from(slice: &'a [T]) -> Self {
        slice.into_iter().collect()
    }
}

impl<'a, A, T> From<&'a Vec<T>> for CatList<A>
where
    &'a T: Shared<A>,
{
    fn from(vec: &'a Vec<T>) -> Self {
        vec.into_iter().collect()
    }
}

impl<A> From<Vec<A>> for CatList<A> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<A> From<Vec<Arc<A>>> for CatList<A> {
    fn from(vec: Vec<Arc<A>>) -> Self {
        if vec.len() <= HASH_SIZE {
            Self::from_head(vec)
        } else {
            vec.into_iter().collect()
        }
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for CatList<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        CatList::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for generating a list of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_catlist(ref l in catlist(".*", 10..100)) {
    ///         assert!(l.len() < 100);
    ///         assert!(l.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn catlist<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<CatList<<T::Value as ValueTree>::Value>> {
        ::proptest::collection::vec(element, size)
            .prop_map(CatList::from)
            .boxed()
    }

    /// A strategy for an ordered list of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_ordered_catlist(ref l in ordered_catlist(".*", 10..100)) {
    ///         assert_eq!(l, l.sort());
    ///     }
    /// }
    /// ```
    pub fn ordered_catlist<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<CatList<<T::Value as ValueTree>::Value>>
    where
        <T::Value as ValueTree>::Value: Ord,
    {
        ::proptest::collection::vec(element, size)
            .prop_map(|v| CatList::from(v).sort())
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use super::proptest::*;
    use test::is_sorted;
    use proptest::num::i32;
    use proptest::collection;

    #[test]
    fn basic_consistency() {
        let vec: Vec<i32> = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, -1,
        ];
        let mut list = CatList::from_iter(vec.clone());
        assert_eq!(Some(Arc::new(-1)), list.last());
        let mut index = 0;
        loop {
            assert_eq!(vec.len() - index, list.len());
            assert_eq!(
                list.0.size,
                list.0
                    .tail
                    .iter()
                    .fold(list.0.head.len(), |a, n| a + n.len())
            );
            match list.pop_front() {
                None => {
                    assert_eq!(vec.len(), index);
                    break;
                }
                Some((head, tail)) => {
                    assert_eq!(vec[index], *head);
                    index += 1;
                    list = tail;
                }
            }
        }
    }

    proptest! {
        #[test]
        fn length(ref v in collection::vec(i32::ANY, 0..100)) {
            let list = CatList::from_iter(v.clone());
            v.len() == list.len()
        }

        #[test]
        fn order(ref vec in collection::vec(i32::ANY, 0..100)) {
            let list = CatList::from_iter(vec.clone());
            assert_eq!(vec, &Vec::from_iter(list.iter().map(|a| *a)));
        }

        #[test]
        fn reverse_order(ref vec in collection::vec(i32::ANY, 0..100)) {
            let list = CatList::from_iter(vec.iter().rev().cloned());
            assert_eq!(vec, &Vec::from_iter(list.iter().rev().map(|a| *a)));
        }

        #[test]
        fn equality(ref vec in collection::vec(i32::ANY, 0..100)) {
            let left = CatList::from_iter(vec.clone());
            let right = CatList::from_iter(vec.clone());
            assert_eq!(left, right);
        }

        #[test]
        fn proptest_a_list(ref l in catlist(i32::ANY, 10..100)) {
            assert!(l.len() < 100);
            assert!(l.len() >= 10);
        }

        #[test]
        fn proptest_ordered_list(ref l in ordered_catlist(i32::ANY, 10..100)) {
            assert_eq!(l, &l.sort());
        }

        #[test]
        fn push_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, list.len());
                list = list.push_back(value);
                assert_eq!(count + 1, list.len());
            }
            assert_eq!(input, &Vec::from_iter(list.iter().map(|a| *a)));
        }

        #[test]
        fn push_back_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, list.len());
                list.push_back_mut(value);
                assert_eq!(count + 1, list.len());
            }
            assert_eq!(input, &Vec::from_iter(list.iter().map(|a| *a)));
        }

        #[test]
        fn push_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, list.len());
                list = list.push_front(value);
                assert_eq!(count + 1, list.len());
            }
            assert_eq!(input, &Vec::from_iter(list.iter().rev().map(|a| *a)));
        }

        #[test]
        fn push_front_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, list.len());
                list.push_front_mut(value);
                assert_eq!(count + 1, list.len());
            }
            assert_eq!(input, &Vec::from_iter(list.iter().rev().map(|a| *a)));
        }

        #[test]
        fn pop_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::from_iter(input.iter().cloned());
            for value in input.iter().rev().cloned() {
                if let Some((popped, new_list)) = list.pop_back() {
                    assert_eq!(Arc::new(value), popped);
                    list = new_list;
                } else {
                    panic!("pop_back ended prematurely");
                }
            }
            assert_eq!(None, list.pop_back());
        }

        #[test]
        fn pop_back_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::from_iter(input.iter().cloned());
            for value in input.iter().rev().cloned() {
                assert_eq!(Some(Arc::new(value)), list.pop_back_mut());
            }
            assert_eq!(None, list.pop_back_mut());
        }

        #[test]
        fn pop_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::from_iter(input.iter().cloned());
            for value in input.iter().cloned() {
                if let Some((popped, new_list)) = list.pop_front() {
                    assert_eq!(Arc::new(value), popped);
                    list = new_list;
                } else {
                    panic!("pop_front ended prematurely");
                }
            }
            assert_eq!(None, list.pop_front());
        }

        #[test]
        fn pop_front_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut list = CatList::from_iter(input.iter().cloned());
            for value in input.iter().cloned() {
                assert_eq!(Some(Arc::new(value)), list.pop_front_mut());
            }
            assert_eq!(None, list.pop_front_mut());
        }

        #[test]
        fn reverse_a_list(ref l in catlist(i32::ANY, 0..100)) {
            let vec: Vec<i32> = l.iter().map(|v| *v).collect();
            let rev = CatList::from_iter(vec.into_iter().rev());
            assert_eq!(l.reverse(), rev);
        }

        #[test]
        fn append_two_lists(ref xs in catlist(i32::ANY, 0..100), ref ys in catlist(i32::ANY, 0..100)) {
            let extended = CatList::from_iter(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            assert_eq!(xs.append(ys), extended);
            assert_eq!(xs.append(ys).len(), extended.len());
        }

        #[test]
        fn sort_a_list(ref l in catlist(i32::ANY, 0..100)) {
            let sorted = l.sort();
            assert_eq!(l.len(), sorted.len());
            assert!(is_sorted(&sorted));
        }
    }
}
