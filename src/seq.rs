use std::sync::Arc;
use std::iter::{Sum, FromIterator};
use std::ops::{Add, Deref};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Formatter, Error};
use queue::Queue;
use list::List;

use self::SeqNode::{Nil, Cons};

#[macro_export]
macro_rules! seq {
    () => { $crate::seq::Seq::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::seq::Seq::new();
        $(
            l = l.push_back($x);
        )*
            l
    }};
}

pub struct Seq<A>(Arc<SeqNode<A>>);

pub enum SeqNode<A> {
    Nil,
    Cons(usize, Arc<A>, Queue<Seq<A>>),
}

impl<A> Seq<A> {
    pub fn new() -> Self {
        Seq(Arc::new(Nil))
    }

    pub fn singleton<R>(a: R) -> Self
        where Arc<A>: From<R>
    {
        Seq::new().push_front(a)
    }

    pub fn is_empty(&self) -> bool {
        match *self.0 {
            Nil => true,
            _ => false,
        }
    }

    pub fn len(&self) -> usize {
        match *self.0 {
            Nil => 0,
            Cons(l, _, _) => l,
        }
    }

    pub fn head(&self) -> Option<Arc<A>> {
        match *self.0 {
            Nil => None,
            Cons(_, ref a, _) => Some(a.clone()),
        }
    }

    pub fn tail(&self) -> Option<Self> {
        match *self.0 {
            Nil => None,
            Cons(_, _, ref q) if q.is_empty() => Some(Seq::new()),
            Cons(_, _, ref q) => Some(fold_queue(|a, b| a.link(&b), Seq::new(), q))
        }
    }

    pub fn append(&self, other: &Self) -> Self {
        match (self, other) {
            (l, r) if l.is_empty() => r.clone(),
            (l, r) if r.is_empty() => l.clone(),
            _ => self.link(other),
        }
    }

    pub fn link(&self, other: &Self) -> Self {
        match *self.0 {
            Nil => other.clone(),
            Cons(l, ref a, ref q) => {
                Seq(Arc::new(Cons(l + other.len(), a.clone(), q.push(other.clone()))))
            }
        }
    }

    pub fn cons<R>(&self, a: R) -> Self
        where Arc<A>: From<R>
    {
        Seq(Arc::new(Cons(1, Arc::from(a), Queue::new()))).append(self)
    }

    pub fn push_front<R>(&self, a: R) -> Self
        where Arc<A>: From<R>
    {
        self.cons(a)
    }

    pub fn snoc<R>(&self, a: R) -> Self
        where Arc<A>: From<R>
    {
        self.append(&Seq(Arc::new(Cons(1, Arc::from(a), Queue::new()))))
    }

    pub fn push_back<R>(&self, a: R) -> Self
        where Arc<A>: From<R>
    {
        self.snoc(a)
    }

    pub fn uncons(&self) -> Option<(Arc<A>, Seq<A>)> {
        self.head().and_then(|h| self.tail().map(|t| (h, t)))
    }

    pub fn iter(&self) -> SeqIter<A> {
        SeqIter { current: self.clone() }
    }

    pub fn reverse(&self) -> Self {
        let mut out = Seq::new();
        for i in self.iter() {
            out = out.cons(i)
        }
        out
    }

    pub fn sort_by<F>(&self, cmp: F) -> Self
        where F: Fn(Arc<A>, Arc<A>) -> Ordering
    {
        // TODO should perhaps implement this directly for Seq
        Seq::from_iter(List::from_iter(self).sort_by(cmp))
    }
}

impl<A: Ord> Seq<A> {
    pub fn sort(&self) -> Self {
        self.sort_by(|a, b| a.as_ref().cmp(b.as_ref()))
    }

    pub fn insert<T>(&self, item: T) -> Self
        where Arc<A>: From<T>
    {
        self.insert_ref(Arc::from(item))
    }

    fn insert_ref(&self, item: Arc<A>) -> Self {
        let (l, r) = self.iter()
            .fold((seq![], seq![]), |(l, r), a| if a.deref() > item.deref() {
                (l, r.cons(a))
            } else {
                (l.cons(a), r)
            });
        l + seq![item] + r
    }
}

fn fold_queue<A, F>(f: F, seed: Seq<A>, queue: &Queue<Seq<A>>) -> Seq<A>
    where F: Fn(Seq<A>, Seq<A>) -> Seq<A>
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

impl<A> Clone for Seq<A> {
    fn clone(&self) -> Self {
        Seq(self.0.clone())
    }
}

impl<A> Default for Seq<A> {
    fn default() -> Self {
        Seq::new()
    }
}

impl<A> Add for Seq<A> {
    type Output = Seq<A>;

    fn add(self, other: Self) -> Self::Output {
        self.append(&other)
    }
}

impl<A: PartialEq> PartialEq for Seq<A> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<A: Eq> Eq for Seq<A> {}

impl<A: PartialOrd> PartialOrd for Seq<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord> Ord for Seq<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Hash> Hash for Seq<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self.iter() {
            i.hash(state)
        }
    }
}

impl<A: Debug> Debug for Seq<A> {
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

pub struct SeqIter<A> {
    current: Seq<A>,
}

impl<A> Iterator for SeqIter<A> {
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

impl<A> ExactSizeIterator for SeqIter<A> {}

impl<A> IntoIterator for Seq<A> {
    type Item = Arc<A>;
    type IntoIter = SeqIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        SeqIter { current: self }
    }
}

impl<'a, A> IntoIterator for &'a Seq<A> {
    type Item = Arc<A>;
    type IntoIter = SeqIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> Sum for Seq<A> {
    fn sum<I>(it: I) -> Self
        where I: Iterator<Item = Self>
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, T> FromIterator<T> for Seq<A>
    where Arc<A>: From<T>
{
    fn from_iter<I>(source: I) -> Self
        where I: IntoIterator<Item = T>
    {
        source.into_iter().map(Seq::singleton).sum()
    }
}

// Conversions

impl<'a, A, T> From<&'a [T]> for Seq<A>
    where Arc<A>: From<&'a T>
{
    fn from(slice: &'a [T]) -> Seq<A> {
        slice.into_iter().collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for Seq<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Seq::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use test::is_sorted;

    quickcheck! {
        fn length(vec: Vec<i32>) -> bool {
            let seq = Seq::from_iter(vec.clone());
            vec.len() == seq.len()
        }

        fn order(vec: Vec<i32>) -> bool {
            let seq = Seq::from_iter(vec.clone());
            seq.iter().map(|a| *a).eq(vec.into_iter())
        }

        fn equality(vec: Vec<i32>) -> bool {
            let left = Seq::from_iter(vec.clone());
            let right = Seq::from_iter(vec);
            left == right
        }

        fn reverse_a_seq(l: Seq<i32>) -> bool {
            let vec: Vec<i32> = l.iter().map(|v| *v).collect();
            let rev = Seq::from_iter(vec.into_iter().rev());
            l.reverse() == rev
        }

        fn append_two_seqs(xs: Seq<i32>, ys: Seq<i32>) -> bool {
            let extended = Seq::from_iter(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            xs.append(&ys) == extended
        }

        fn length_of_append(xs: Seq<i32>, ys: Seq<i32>) -> bool {
            let extended = Seq::from_iter(xs.iter().map(|v| *v).chain(ys.iter().map(|v| *v)));
            xs.append(&ys).len() == extended.len()
        }

        fn sort_a_seq(l: Seq<i32>) -> bool {
            let sorted = l.sort();
            l.len() == sorted.len() && is_sorted(&sorted)
        }
    }
}
