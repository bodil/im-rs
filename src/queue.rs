//! A strict queue.

use std::sync::Arc;
use std::iter::FromIterator;
use std::fmt;
use shared::Shared;
use vector::Vector;

/// A strict queue backed by a pair of vectors.
///
/// All operations run in O(1) amortised time, but the `pop`
/// operation may run in O(n) time in the worst case.
pub struct Queue<A>(Vector<A>, Vector<A>);

impl<A> Queue<A> {
    /// Construct an empty queue.
    #[inline]
    pub fn new() -> Self {
        Queue(Vector::new(), Vector::new())
    }

    /// Construct a queue by consuming an [`IntoIterator`][std::iter::IntoIterator].
    ///
    /// Allows you to construct a queue out of anything that implements
    /// the [`IntoIterator`][std::iter::IntoIterator] trait.
    ///
    /// Time: O(n)
    ///
    /// [std::iter::IntoIterator]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
    pub fn from<R, I>(it: I) -> Queue<A>
    where
        I: IntoIterator<Item = R>,
        R: Shared<A>,
    {
        it.into_iter().collect()
    }

    /// Test whether a queue is empty.
    ///
    /// Time: O(1)
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() && self.1.is_empty()
    }

    /// Get the length of a queue.
    ///
    /// Time: O(1)
    pub fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }

    /// Construct a new queue by appending an element to the end
    /// of the current queue.
    ///
    /// Time: O(1)
    pub fn push_back<R>(&self, v: R) -> Self
    where
        R: Shared<A>,
    {
        Queue(self.0.clone(), self.1.push(v))
    }

    /// Append an element to the end of the current queue.
    ///
    /// This is a copy-on-write operation, so that the parts of the queue's
    /// structure which are shared with other queues will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)
    pub fn push_back_mut<R>(&mut self, v: R)
    where
        R: Shared<A>,
    {
        self.1.push_mut(v)
    }

    /// Construct a new queue by appending an element to the end
    /// of the current queue.
    ///
    /// Time: O(1)
    //
    /// This is an alias for [`push_back`][push_back].
    ///
    /// [push_back]: #method.push_back
    #[inline]
    pub fn push<R>(&self, v: R) -> Self
    where
        R: Shared<A>,
    {
        self.push_back(v)
    }

    /// Construct a new queue by appending an element to the front
    /// of the current queue.
    ///
    /// Time: O(1)
    pub fn push_front<R>(&self, v: R) -> Self
    where
        R: Shared<A>,
    {
        Queue(self.0.push(v), self.1.clone())
    }

    /// Append an element to the front of the current queue.
    ///
    /// This is a copy-on-write operation, so that the parts of the queue's
    /// structure which are shared with other queues will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)
    pub fn push_front_mut<R>(&mut self, v: R)
    where
        R: Shared<A>,
    {
        self.0.push_mut(v);
    }

    /// Get the first element out of a queue, as well as the remainder
    /// of the queue.
    ///
    /// Returns `None` if the queue is empty. Otherwise, you get a tuple
    /// of the first element and the remainder of the queue.
    pub fn pop_front(&self) -> Option<(Arc<A>, Queue<A>)> {
        match *self {
            Queue(ref l, ref r) if l.is_empty() && r.is_empty() => None,
            Queue(ref l, ref r) => match l.pop() {
                None => Queue(r.reverse(), Vector::new()).pop_front(),
                Some((a, d)) => Some((a, Queue(d, r.clone()))),
            },
        }
    }

    /// Pop an element off the front of the queue.
    ///
    /// Returns `None` if the queue is empty.
    ///
    /// This is a copy-on-write operation, so that the parts of the queue's
    /// structure which are shared with other queues will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)*
    pub fn pop_front_mut(&mut self) -> Option<Arc<A>> {
        if self.0.is_empty() && self.1.is_empty() {
            return None;
        }
        match self.0.pop_mut() {
            None => {
                self.0 = self.1.reverse();
                self.1 = Vector::new();
                self.pop_front_mut()
            }
            value => value,
        }
    }

    /// Get the first element out of a queue, as well as the remainder
    /// of the queue.
    ///
    /// Returns `None` if the queue is empty. Otherwise, you get a tuple
    /// of the first element and the remainder of the queue.
    ///
    /// This is an alias for [`pop_front`][pop_front].
    ///
    /// [pop_front]: #method.pop_front
    #[inline]
    pub fn pop(&self) -> Option<(Arc<A>, Queue<A>)> {
        self.pop_front()
    }

    /// Get the last element out of a queue, as well as the remainder
    /// of the queue.
    ///
    /// Returns `None` if the queue is empty. Otherwise, you get a tuple
    /// of the last element and the remainder of the queue.
    pub fn pop_back(&self) -> Option<(Arc<A>, Queue<A>)> {
        match *self {
            Queue(ref l, ref r) if l.is_empty() && r.is_empty() => None,
            Queue(ref l, ref r) => match r.pop() {
                None => Queue(Vector::new(), l.reverse()).pop_back(),
                Some((a, d)) => Some((a, Queue(l.clone(), d))),
            },
        }
    }

    /// Pop an element off the back of the queue.
    ///
    /// Returns `None` if the queue is empty.
    ///
    /// This is a copy-on-write operation, so that the parts of the queue's
    /// structure which are shared with other queues will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)*
    pub fn pop_back_mut(&mut self) -> Option<Arc<A>> {
        if self.0.is_empty() && self.1.is_empty() {
            return None;
        }
        match self.1.pop_mut() {
            None => {
                self.1 = self.0.reverse();
                self.0 = Vector::new();
                self.pop_back_mut()
            }
            value => value,
        }
    }

    /// Get the element at index `index` in the queue.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// Time: O(1)*
    pub fn get(&self, index: usize) -> Option<Arc<A>> {
        if index < self.0.len() {
            self.0.get((self.0.len() - 1) - index)
        } else {
            self.1.get(index - self.0.len())
        }
    }

    /// Get an iterator over a queue.
    pub fn iter(&self) -> Iter<A> {
        Iter {
            queue: self.clone(),
            left: 0,
            right: self.len(),
        }
    }
}

// Core traits

impl<A> Default for Queue<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Clone for Queue<A> {
    fn clone(&self) -> Self {
        Queue(self.0.clone(), self.1.clone())
    }
}

impl<A> fmt::Debug for Queue<A>
where
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        Vector::<A>::from_iter(self.iter()).fmt(f)
    }
}

impl<A: PartialEq> PartialEq for Queue<A> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<A: Eq> Eq for Queue<A> {}

// Iterators

/// An iterator over a queue of elements of type `A`.
pub struct Iter<A> {
    queue: Queue<A>,
    left: usize,
    right: usize,
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left >= self.right {
            None
        } else {
            let item = self.queue.get(self.left);
            self.left += 1;
            item
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.right - self.left;
        (l, Some(l))
    }
}

impl<A> DoubleEndedIterator for Iter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.left >= self.right {
            None
        } else {
            self.right -= 1;
            self.queue.get(self.right)
        }
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

impl<A> IntoIterator for Queue<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A> IntoIterator for &'a Queue<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A, T> FromIterator<T> for Queue<A>
where
    T: Shared<A>,
{
    fn from_iter<I>(source: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut out = Queue::new();
        for item in source {
            out.push_back_mut(item);
        }
        out
    }
}

// Conversions

impl<'a, A, T> From<&'a [T]> for Queue<A>
where
    &'a T: Shared<A>,
{
    fn from(slice: &'a [T]) -> Self {
        slice.into_iter().collect()
    }
}

impl<A, T> From<Vec<T>> for Queue<A>
where
    T: Shared<A>,
{
    fn from(vec: Vec<T>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A, T> From<&'a Vec<T>> for Queue<A>
where
    &'a T: Shared<A>,
{
    fn from(vec: &'a Vec<T>) -> Self {
        vec.into_iter().collect()
    }
}

impl<A> From<Vector<A>> for Queue<A> {
    fn from(vector: Vector<A>) -> Self {
        Queue(Vector::new(), vector)
    }
}

impl<'a, A> From<&'a Vector<A>> for Queue<A> {
    fn from(vector: &'a Vector<A>) -> Self {
        Queue(Vector::new(), vector.clone())
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for Queue<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Queue::from(Vector::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for generating a queue of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_queue(ref q in queue(".*", 10..100)) {
    ///         assert!(q.len() < 100);
    ///         assert!(q.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn queue<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<Queue<<T::Value as ValueTree>::Value>> {
        ::vector::proptest::vector(element, size)
            .prop_map(Queue::from)
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use std::iter::FromIterator;
    use proptest::num::i32;
    use proptest::collection;

    #[test]
    fn general_consistency() {
        let q = Queue::new().push(1).push(2).push(3).push(4).push(5).push(6);
        assert_eq!(6, q.len());
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(vec, Vec::from_iter(q.iter().map(|a| *a)))
    }

    proptest! {
        #[test]
        fn push_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, queue.len());
                queue = queue.push_back(value);
                assert_eq!(count + 1, queue.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), queue.get(index));
            }
        }

        #[test]
        fn push_back_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, queue.len());
                queue.push_back_mut(value);
                assert_eq!(count + 1, queue.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), queue.get(index));
            }
        }

        #[test]
        fn push_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, queue.len());
                queue = queue.push_front(value);
                assert_eq!(count + 1, queue.len());
            }
            for (index, value) in input.iter().rev().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), queue.get(index));
            }
        }

        #[test]
        fn push_front_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, queue.len());
                queue.push_front_mut(value);
                assert_eq!(count + 1, queue.len());
            }
            for (index, value) in input.iter().rev().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), queue.get(index));
            }
        }

        #[test]
        fn pop_back(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::from_iter(input.iter().cloned());
            for value in input.iter().rev().cloned() {
                if let Some((popped, new_queue)) = queue.pop_back() {
                    assert_eq!(Arc::new(value), popped);
                    queue = new_queue;
                } else {
                    panic!("pop_back ended prematurely");
                }
            }
            assert_eq!(None, queue.pop_back());
        }

        #[test]
        fn pop_back_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::from_iter(input.iter().cloned());
            for value in input.iter().rev().cloned() {
                assert_eq!(Some(Arc::new(value)), queue.pop_back_mut());
            }
            assert_eq!(None, queue.pop_back_mut());
        }

        #[test]
        fn pop_front(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::from_iter(input.iter().cloned());
            for value in input.iter().cloned() {
                if let Some((popped, new_queue)) = queue.pop_front() {
                    assert_eq!(Arc::new(value), popped);
                    queue = new_queue;
                } else {
                    panic!("pop_front ended prematurely");
                }
            }
            assert_eq!(None, queue.pop_front());
        }

        #[test]
        fn pop_front_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut queue = Queue::from_iter(input.iter().cloned());
            for value in input.iter().cloned() {
                assert_eq!(Some(Arc::new(value)), queue.pop_front_mut());
            }
            assert_eq!(None, queue.pop_front_mut());
        }

        #[test]
        fn iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let queue = Queue::from_iter(input.iter().cloned());
            assert!(input.iter().cloned().map(Arc::new).eq(queue.iter()));
        }

        #[test]
        fn reverse_iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let queue = Queue::from_iter(input.iter().cloned());
            assert!(input.iter().rev().cloned().map(Arc::new).eq(queue.iter().rev()));
        }
    }
}
