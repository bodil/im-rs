use std::sync::Arc;
use list::List;

pub struct Queue<A>(List<A>, List<A>);

impl<A> Queue<A> {
    pub fn new() -> Self {
        Queue(list![], list![])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty() && self.1.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }

    pub fn push<R>(&self, v: R) -> Self where Arc<A>: From<R> {
        Queue(self.0.clone(), self.1.cons(v))
    }

    pub fn pop(&self) -> Option<(Arc<A>, Queue<A>)> {
        match self {
            &Queue(ref l, ref r) if l.is_empty() && r.is_empty() => None,
            &Queue(ref l, ref r) => match l.uncons() {
                None => Queue(r.reverse(), list![]).pop(),
                Some((a, d)) => Some((a, Queue(d, r.clone())))
            }
        }
    }

    pub fn iter(&self) -> QueueIter<A> {
        QueueIter { current: self.clone() }
    }
}

impl<A> Clone for Queue<A> {
    fn clone(&self) -> Self {
        Queue(self.0.clone(), self.1.clone())
    }
}

pub struct QueueIter<A> {
    current: Queue<A>
}

impl<A> Iterator for QueueIter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current.pop() {
            None => None,
            Some((a, q)) => {
                self.current = q;
                Some(a)
            }
        }
    }
}

impl<A> IntoIterator for Queue<A> {
    type Item = Arc<A>;
    type IntoIter = QueueIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        QueueIter { current: self }
    }
}

impl<'a, A> IntoIterator for &'a Queue<A> {
    type Item = Arc<A>;
    type IntoIter = QueueIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn general_consistency() {
        let q = Queue::new().push(1).push(2).push(3).push(4).push(5).push(6);
        assert_eq!(6, q.len());
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(vec, Vec::from_iter(q.iter().map(|a| *a)))
    }

    quickcheck! {
        fn length(v: Vec<i32>) -> bool {
            let mut q = Queue::new();
            for i in v.iter() {
                q = q.push(i)
            }
            v.len() == q.len()
        }

        fn order(v: Vec<i32>) -> bool {
            let mut q = Queue::new();
            for i in v.iter() {
                q = q.push(i)
            }
            v == Vec::from_iter(q.iter().map(|a| **a))
        }
    }
}
