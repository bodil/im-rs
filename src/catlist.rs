// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A catenable list.
//!
//! A list data structure with O(1)* push and pop operations on both
//! ends and O(1) concatenation of lists.
//!
//! You usually want the [`Vector`][vector::Vector] instead, which
//! performs better on all operations except concatenation. If instant
//! concatenation is what you need, the `CatList` is the cat for you.
//!
//! [queue::Queue]: ../queue/struct.Queue.html
//! [vector::Vector]: ../vector/struct.Vector.html
//! [conslist::ConsList]: ../conslist/struct.ConsList.html

use bits::HASH_SIZE;
use shared::Shared;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, Sum};
use std::ops::{Add, Deref};
use std::sync::Arc;
use vector::Vector;

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
/// Constructs a list with the value `car` prepended to the front of
/// the list `cdr`.
///
/// This is just a shorthand for `list.cons(item)`, but I find it much
/// easier to read `cons(1, cons(2, CatList::new()))` than
/// `CatList::new().cons(2).cons(1)`, given that the resulting list
/// will be `[1, 2]`.
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
/// A list data structure with O(1)* push and pop operations on both
/// ends and O(1) concatenation of lists.
///
/// You usually want the [`Vector`][vector::Vector] instead, which
/// performs better on all operations except concatenation. If instant
/// concatenation is what you need, the `CatList` is the cat for you.
///
/// [queue::Queue]: ../queue/struct.Queue.html
/// [vector::Vector]: ../vector/struct.Vector.html
/// [conslist::ConsList]: ../conslist/struct.ConsList.html
pub struct CatList<A> {
    size: usize,
    head: Arc<Vec<Arc<A>>>,
    tail: Vector<CatList<A>>,
}

impl<A> CatList<A> {
    /// Construct an empty list.
    pub fn new() -> Self {
        CatList {
            size: 0,
            head: Arc::new(Vec::new()),
            tail: Vector::new(),
        }
    }

    /// Construct a list with a single value.
    pub fn singleton<R>(a: R) -> Self
    where
        R: Shared<A>,
    {
        CatList::from_head(vec![a.shared()])
    }

    fn from_head(head: Vec<Arc<A>>) -> Self {
        CatList {
            size: head.len(),
            head: Arc::new(head),
            tail: Vector::new(),
        }
    }

    fn make<VA: Shared<Vec<Arc<A>>>>(size: usize, head: VA, tail: Vector<CatList<A>>) -> Self {
        CatList {
            size,
            head: head.shared(),
            tail,
        }
    }

    /// Test whether a list is empty.
    #[inline]
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
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Get the first element of a list.
    ///
    /// If the list is empty, `None` is returned.
    pub fn head(&self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else {
            self.head.last().cloned()
        }
    }

    /// Get the last element of a list, as well as the list with the
    /// last element removed.
    ///
    /// If the list is empty, [`None`][None] is returned.
    ///
    /// Time: O(1)*
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn pop_back(&self) -> Option<(Arc<A>, CatList<A>)> {
        if self.is_empty() {
            None
        } else if self.tail.is_empty() {
            Some((
                self.head.first().unwrap().clone(),
                CatList::from_head(self.head.iter().skip(1).cloned().collect()),
            ))
        } else {
            match self.tail.pop_back() {
                None => unreachable!(),
                Some((last_list, queue_without_last_list)) => match last_list.pop_back() {
                    None => unreachable!(),
                    Some((last_item, list_without_last_item)) => {
                        let new_node = CatList {
                            size: self.size - last_list.len(),
                            head: self.head.clone(),
                            tail: queue_without_last_list,
                        };
                        Some((last_item, new_node.append(list_without_last_item)))
                    }
                },
            }
        }
    }

    /// Get the last element of a list.
    ///
    /// Time: O(1)*
    ///
    /// If the list is empty, `None` is returned.
    pub fn last(&self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else if self.tail.is_empty() {
            self.head.first().cloned()
        } else {
            self.tail.last().unwrap().last()
        }
    }

    /// Get the list without the last element.
    ///
    /// Time: O(1)*
    ///
    /// If the list is empty, `None` is returned.
    pub fn init(&self) -> Option<CatList<A>> {
        self.pop_back().map(|a| a.1)
    }

    /// Get the tail of a list.
    ///
    /// Time: O(1)
    ///
    /// The tail means all elements in the list after the first item
    /// (the head). If the list only has one element, the result is an
    /// empty list. If the list is empty, the result is `None`.
    pub fn tail(&self) -> Option<Self> {
        if self.is_empty() {
            None
        } else if self.len() == 1 {
            Some(CatList::new())
        } else if self.tail.is_empty() {
            Some(CatList::from_head(
                self.head
                    .iter()
                    .take(self.head.len() - 1)
                    .cloned()
                    .collect(),
            ))
        } else if self.head.len() > 1 {
            Some(CatList::make(
                self.len() - 1,
                Arc::new(
                    self.head
                        .iter()
                        .take(self.head.len() - 1)
                        .cloned()
                        .collect(),
                ),
                self.tail.clone(),
            ))
        } else {
            Some(self.tail.iter().fold(CatList::new(), |a, b| a.append(b)))
        }
    }

    /// Append the list `other` to the end of the current list.
    ///
    /// Time: O(1)
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
                if l.tail.is_empty() && l.head.len() + r.head.len() <= HASH_SIZE {
                    let mut new_head = (*r.head).clone();
                    new_head.extend(l.head.iter().cloned());
                    CatList::make(l.len() + r.len(), new_head, r.tail.clone())
                } else if !l.tail.is_empty() && l.tail.last().unwrap().tail.is_empty()
                    && l.tail.last().unwrap().head.len() + r.head.len() <= HASH_SIZE
                {
                    let (last, tail_but_last) = l.tail.pop_back().unwrap();
                    let mut new_head = (*r.head).clone();
                    new_head.extend(last.head.iter().cloned());
                    let last_plus_right =
                        CatList::make(last.len() + r.len(), new_head, r.tail.clone());
                    CatList::make(
                        l.len() + r.len(),
                        l.head.clone(),
                        tail_but_last.push_back(last_plus_right),
                    )
                } else {
                    CatList::make(
                        l.len() + r.len(),
                        self.head.clone(),
                        self.tail.push_back(r.clone()),
                    )
                }
            }
        }
    }

    /// Construct a list with a new value prepended to the front of
    /// the current list.
    ///
    /// Time: O(1)
    #[inline]
    pub fn push_front<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        CatList::singleton(a).append(self)
    }

    /// Append the list `other` to the end of the current list, in
    /// place.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::catlist::CatList;
    /// # fn main() {
    /// let mut l = catlist![1, 2, 3];
    /// l.append_mut(catlist![7, 8, 9]);
    ///
    /// assert_eq!(l, catlist![1, 2, 3, 7, 8, 9]);
    /// # }
    /// ```
    pub fn append_mut<R>(&mut self, other_ref: R)
    where
        R: Borrow<Self>,
    {
        let other = other_ref.borrow();
        if other.is_empty() {
            return;
        } else if self.is_empty() {
            self.size = other.size;
            self.head = other.head.clone();
            self.tail = other.tail.clone();
        } else if self.tail.is_empty() && self.head.len() + other.head.len() <= HASH_SIZE {
            self.head = Arc::new(other.head.iter().chain(self.head.iter()).cloned().collect());
            self.tail = Vector::singleton(other.clone());
            self.size += other.len();
        } else {
            self.tail.push_back_mut(other.borrow().clone());
            self.size += other.len();
        }
    }

    /// Push a value onto the front of a list, updating in
    /// place.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(1)
    pub fn push_front_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        if self.head.len() >= HASH_SIZE {
            let next = self.clone();
            self.size = 1;
            self.head = Arc::new(vec![a.shared()]);
            self.tail = Vector::new();
            self.append_mut(next);
        } else {
            let head = Arc::make_mut(&mut self.head);
            head.push(a.shared());
            self.size += 1;
        }
    }

    /// Push a value onto the back of a list, updating in
    /// place.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(1)
    pub fn push_back_mut<R>(&mut self, a: R)
    where
        R: Shared<A>,
    {
        if self.tail.is_empty() && self.head.len() < HASH_SIZE {
            let head = Arc::make_mut(&mut self.head);
            head.insert(0, a.shared());
            self.size += 1;
        } else {
            self.append_mut(CatList::singleton(a))
        }
    }

    /// Remove a value from the front of a list, updating in place.
    /// Returns the removed value.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(1)*
    pub fn pop_front_mut(&mut self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else if self.head.len() > 1 {
            let head = Arc::make_mut(&mut self.head);
            let item = head.pop();
            self.size -= 1;
            item
        } else {
            let item = self.head.last().cloned();
            let tail = self.tail.clone();
            self.size = 0;
            self.head = Default::default();
            self.tail = Default::default();
            for list in tail {
                self.append_mut(list);
            }
            item
        }
    }

    /// Remove a value from the back of a list, updating in place.
    /// Returns the removed value.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(1)*
    pub fn pop_back_mut(&mut self) -> Option<Arc<A>> {
        if self.is_empty() {
            None
        } else if self.tail.is_empty() {
            self.size -= 1;
            let head = Arc::make_mut(&mut self.head);
            Some(head.remove(0))
        } else {
            self.size -= 1;
            let mut last = self.tail.pop_back_mut().unwrap();
            let last_node = Arc::make_mut(&mut last);
            let item = last_node.pop_back_mut();
            if !last_node.is_empty() {
                self.tail.push_back_mut(last_node.clone());
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
    /// Time: O(1)
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
    ///
    /// Time: O(1)
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
    /// Time: O(1)*
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

    /// Construct a list with a new value prepended to the front of
    /// the current list.
    ///
    /// This is an alias for [push_front], for the Lispers in the
    /// house.
    ///
    /// Time: O(1)
    ///
    /// [push_front]: #method.push_front
    pub fn cons<R>(&self, a: R) -> Self
    where
        R: Shared<A>,
    {
        self.push_front(a)
    }

    /// Get the head and the tail of a list.
    ///
    /// This is an alias for [`pop_front`][pop_front].
    ///
    /// Time: O(1)*
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
        Iter::new(self)
    }

    /// Construct a list which is the reverse of the current list.
    ///
    /// Please note that if all you want is to iterate over the list
    /// from back to front, it is much more efficient to use a
    /// [reversed iterator][rev] rather than doing the work of
    /// reversing the list first.
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
    /// # use std::iter::FromIterator;
    /// # fn main() {
    /// assert_eq!(
    ///   catlist![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   CatList::from_iter(1..9)
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn sort(&self) -> Self
    where
        A: Ord,
    {
        self.sort_by(|a, b| a.cmp(b))
    }

    /// Insert an item into a sorted list.
    ///
    /// Constructs a new list with the new item inserted before the
    /// first item in the list which is larger than the new item, as
    /// determined by the `Ord` trait.
    ///
    /// Please note that this is a very inefficient operation; if you
    /// want a sorted list, consider if [`OrdSet`][ordset::OrdSet]
    /// might be a better choice for you.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(
    ///   catlist![2, 4, 5].insert(1).insert(3).insert(6),
    ///   catlist![1, 2, 3, 4, 5, 6]
    /// );
    /// # }
    /// ```
    ///
    /// [ordset::OrdSet]: ../ordset/struct.OrdSet.html
    #[inline]
    pub fn insert<T>(&self, item: T) -> Self
    where
        A: Ord,
        T: Shared<A>,
    {
        let value = item.shared();
        let mut out = CatList::new();
        let mut inserted = false;
        for next in self {
            if next < value {
                out.push_back_mut(next);
                continue;
            }
            if !inserted {
                out.push_back_mut(value.clone());
                inserted = true;
            }
            out.push_back_mut(next);
        }
        if !inserted {
            out.push_back_mut(value);
        }
        out
    }
}

// Core traits

impl<A> Clone for CatList<A> {
    fn clone(&self) -> Self {
        CatList {
            size: self.size,
            head: self.head.clone(),
            tail: self.tail.clone(),
        }
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

impl<'a, A> Add for &'a CatList<A> {
    type Output = CatList<A>;

    fn add(self, other: Self) -> Self::Output {
        self.append(other)
    }
}

impl<A, R> Extend<R> for CatList<A>
where
    A: Ord,
    R: Shared<A>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = R>,
    {
        for value in iter {
            self.push_back_mut(value);
        }
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
        self.len() == other.len()
            && ((Arc::ptr_eq(&self.head, &other.head) && self.tail == other.tail)
                || self.iter().eq(other.iter()))
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
    fwd_stack: Vec<(Arc<CatList<A>>, usize)>,
    fwd_current: Arc<CatList<A>>,
    fwd_head_index: usize,
    fwd_tail_index: usize,
    rev_stack: Vec<(Arc<CatList<A>>, usize)>,
    rev_current: Arc<CatList<A>>,
    rev_head_index: usize,
    rev_tail_index: usize,
    remaining: usize,
}

impl<A> Iter<A> {
    fn new<RL>(list: RL) -> Self
    where
        RL: Shared<CatList<A>>,
    {
        let l = list.shared();
        let mut stack = Vec::new();
        let mut item = l.clone();
        while let Some(last) = item.tail.last() {
            stack.push((item, 1));
            item = last;
        }
        assert!(item.tail.is_empty());
        Iter {
            remaining: l.len(),
            fwd_current: l.clone(),
            fwd_stack: Vec::new(),
            fwd_head_index: 0,
            fwd_tail_index: 0,
            rev_current: item,
            rev_stack: stack,
            rev_head_index: 0,
            rev_tail_index: 0,
        }
    }
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else if self.fwd_current.head.len() > self.fwd_head_index {
            let item =
                &self.fwd_current.head[(self.fwd_current.head.len() - 1) - self.fwd_head_index];
            self.fwd_head_index += 1;
            self.remaining -= 1;
            Some(item.clone())
        } else if let Some(list) = self.fwd_current.tail.get(self.fwd_tail_index) {
            self.fwd_stack
                .push((self.fwd_current.clone(), self.fwd_tail_index + 1));
            self.fwd_current = list;
            self.fwd_head_index = 0;
            self.fwd_tail_index = 0;
            self.next()
        } else if let Some((list, index)) = self.fwd_stack.pop() {
            self.fwd_head_index = list.head.len();
            self.fwd_tail_index = index;
            self.fwd_current = list;
            self.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<A> DoubleEndedIterator for Iter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else if self.rev_current.tail.len() > self.rev_tail_index {
            println!("pushing from tail");
            let list = self.rev_current
                .tail
                .get((self.rev_current.tail.len() - 1) - self.rev_tail_index)
                .unwrap();
            self.rev_stack
                .push((self.rev_current.clone(), self.rev_tail_index + 1));
            self.rev_current = list;
            self.rev_head_index = 0;
            self.rev_tail_index = 0;
            self.next_back()
        } else if self.rev_current.head.len() > self.rev_head_index {
            println!("yielding from head");
            let item = &self.rev_current.head[self.rev_head_index];
            self.rev_head_index += 1;
            self.remaining -= 1;
            Some(item.clone())
        } else if let Some((list, index)) = self.rev_stack.pop() {
            println!("popping from stack");
            self.rev_head_index = 0;
            self.rev_tail_index = index;
            self.rev_current = list;
            self.next_back()
        } else {
            None
        }
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

impl<A> IntoIterator for CatList<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
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
    use super::proptest::*;
    use super::*;
    use proptest::collection;
    use proptest::num::i32;
    use test::is_sorted;

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
                list.size,
                list.tail.iter().fold(list.head.len(), |a, n| a + n.len())
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
        fn append_two_lists(
            ref xs in catlist(i32::ANY, 0..100),
            ref ys in catlist(i32::ANY, 0..100)
        ) {
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
