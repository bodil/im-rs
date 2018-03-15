//! A vector.
//!
//! This is an implementation of Rich Hickey's [bitmapped vector tries][bmvt],
//! which offers highly efficient (amortised linear time) index lookups as well
//! as appending elements to, or popping elements off, the end of the vector.
//!
//! Use this if you need a [`Vec`][Vec]-like structure with fast index lookups and
//! similar performance characteristics otherwise.
//! If you don't need lookups or updates by index, you might be better off using
//! a [`List`][List], which has better performance characteristics for other
//! structural operations.
//!
//! [bmvt]: https://hypirion.com/musings/understanding-persistent-vector-pt-1
//! [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//! [List]: ../list/struct.List.html

use std::sync::Arc;
use std::mem;
use std::iter::{Extend, FromIterator, Sum};
use std::fmt::{Debug, Error, Formatter};
use std::cmp::Ordering;
use std::ops::Add;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::borrow::Borrow;

use shared::Shared;
use lens::PartialLens;
use bits::{HASH_BITS, HASH_MASK, HASH_SIZE};

mod nodes;

use self::nodes::{Entry, Node};

/// Construct a vector from a sequence of elements.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::vector::Vector;
/// # fn main() {
/// assert_eq!(
///   vector![1, 2, 3],
///   Vector::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! vector {
    () => { $crate::vector::Vector::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::vector::Vector::new();
        $(
            l.push_mut($x);
        )*
            l
    }};
}

/// A persistent vector of elements of type `A`.
///
/// This is an implementation of Rich Hickey's [bitmapped vector tries][bmvt],
/// which offers highly efficient (amortised linear time) index lookups as well
/// as appending elements to, or popping elements off, the end of the vector.
///
/// Use this if you need a [`Vec`][Vec]-like structure with fast index lookups and
/// similar performance characteristics otherwise.
/// If you don't need lookups or updates by index, you might be better off using
/// a [`List`][List], which has better performance characteristics for other
/// structural operations.
///
/// [bmvt]: https://hypirion.com/musings/understanding-persistent-vector-pt-1
/// [Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
/// [List]: ../list/struct.List.html
pub struct Vector<A> {
    count: usize,
    shift: usize,
    root: Arc<Node<A>>,
    tail: Arc<Vec<Entry<A>>>,
}

impl<A> Vector<A> {
    /// Construct an empty vector.
    pub fn new() -> Self {
        Vector {
            count: 0,
            shift: HASH_BITS,
            root: Arc::new(Node::new()),
            tail: Arc::new(Vec::with_capacity(HASH_SIZE)),
        }
    }

    /// Construct a vector with a single value.
    pub fn singleton<R>(a: R) -> Self
    where
        R: Shared<A>,
    {
        let mut tail = Vec::with_capacity(HASH_SIZE);
        tail.push(Entry::Value(a.shared()));
        Vector {
            count: 0,
            shift: HASH_BITS,
            root: Arc::new(Node::new()),
            tail: Arc::new(tail),
        }
    }

    /// Get the length of a vector.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # fn main() {
    /// assert_eq!(5, vector![1, 2, 3, 4, 5].len());
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Test whether a list is empty.
    ///
    /// Time: O(1)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an iterator over a vector.
    ///
    /// Time: O(1)* per [`next()`][next] call
    ///
    /// [next]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
    #[inline]
    pub fn iter(&self) -> Iter<A> {
        Iter::new(self)
    }

    /// Get the first element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(1)*
    #[inline]
    pub fn head(&self) -> Option<Arc<A>> {
        self.get(0)
    }

    /// Get the last element of a vector.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(1)*
    pub fn last(&self) -> Option<Arc<A>> {
        self.get(self.len() - 1)
    }

    /// Get the vector without the last element.
    ///
    /// If the vector is empty, `None` is returned.
    ///
    /// Time: O(1)*
    pub fn init(&self) -> Option<Vector<A>> {
        self.pop().map(|(_, v)| v)
    }

    /// Get the value at index `index` in a vector.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// Time: O(1)*
    pub fn get(&self, index: usize) -> Option<Arc<A>> {
        if index >= self.len() {
            return None;
        }
        let entry = match self.node_for(index) {
            None => self.tail.get(index & HASH_MASK as usize).cloned(),
            Some(node) => node.children.get(index & HASH_MASK as usize).cloned(),
        };
        match entry {
            Some(Entry::Value(ref value)) => Some(value.clone()),
            Some(Entry::Node(_)) => {
                panic!("Vector::get: encountered node where value was expected")
            }
            Some(Entry::Empty) => panic!("Vector::get: encountered null, expected value"),
            None => panic!("Vector::get: unhandled index out of bounds situation!"),
        }
    }

    /// Get the value at index `index` in a vector, directly.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(1)*
    pub fn get_unwrapped(&self, index: usize) -> Arc<A> {
        self.get(index).expect("get_unwrapped index out of bounds")
    }

    /// Create a new vector with the value at index `index` updated.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(1)*
    pub fn set<RA>(&self, index: usize, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        assert!(
            index < self.count,
            "index out of bounds: index {} < {}",
            index,
            self.count
        );
        if index >= self.tailoff() {
            let mut tail = (*self.tail).clone();
            tail[index & HASH_MASK as usize] = Entry::Value(value.shared());
            self.update_tail(tail)
        } else {
            self.update_root(nodes::set_in(self.shift, &self.root, index, value.shared()))
        }
    }

    /// Update the value at index `index` in a vector.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// This is a copy-on-write operation, so that the parts of the vector's
    /// structure which are shared with other vectors will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)*
    pub fn set_mut<RA>(&mut self, index: usize, value: RA)
    where
        RA: Shared<A>,
    {
        assert!(
            index < self.count,
            "index out of bounds: index {} < {}",
            index,
            self.count
        );
        if index >= self.tailoff() {
            let tail = Arc::make_mut(&mut self.tail);
            tail[index & HASH_MASK as usize] = Entry::Value(value.shared());
        } else {
            let root = Arc::make_mut(&mut self.root);
            nodes::set_in_mut(self.shift, root, index, value.shared())
        }
    }

    /// Construct a vector with a new value prepended to the end of the
    /// current vector.
    ///
    /// Time: O(1)*
    pub fn push<RA>(&self, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        if self.len() - self.tailoff() < HASH_SIZE {
            let mut tail = (*self.tail).clone();
            tail.push(Entry::Value(value.shared()));
            Vector {
                count: self.count + 1,
                shift: self.shift,
                root: self.root.clone(),
                tail: Arc::new(tail),
            }
        } else {
            let mut shift = self.shift;
            let tail_node = Node::from_vec((*self.tail).clone());
            let new_root = if (self.len() >> HASH_BITS) > (1 << self.shift) {
                let mut node = Node::new();
                node.children[0] = Entry::Node(self.root.clone());
                node.children[1] = Entry::Node(Arc::new(nodes::new_path(self.shift, tail_node)));
                shift += HASH_BITS;
                node
            } else {
                nodes::push_tail(self.count, self.shift, &self.root, tail_node)
            };
            Vector {
                count: self.count + 1,
                shift,
                root: Arc::new(new_root),
                tail: Arc::new(vec![Entry::Value(value.shared())]),
            }
        }
    }

    /// Update a vector in place with a new value prepended to the end of it.
    ///
    /// This is a copy-on-write operation, so that the parts of the vector's
    /// structure which are shared with other vectors will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)*
    pub fn push_mut<RA>(&mut self, value: RA)
    where
        RA: Shared<A>,
    {
        if self.count - self.tailoff() < HASH_SIZE {
            let tail = Arc::make_mut(&mut self.tail);
            tail.push(Entry::Value(value.shared()));
        } else {
            let tail_node = {
                let tail = Arc::make_mut(&mut self.tail);
                Node::from_vec(mem::replace(tail, Vec::with_capacity(HASH_SIZE)))
            };
            if (self.count >> HASH_BITS) > (1 << self.shift) {
                let mut node = Node::new();
                node.children[0] = Entry::Node(self.root.clone());
                node.children[1] = Entry::Node(Arc::new(nodes::new_path(self.shift, tail_node)));
                self.shift += HASH_BITS;
                self.root = Arc::new(node);
            } else {
                let root = Arc::make_mut(&mut self.root);
                nodes::push_tail_mut(self.count, self.shift, root, tail_node)
            }
            let tail = Arc::make_mut(&mut self.tail);
            tail.push(Entry::Value(value.shared()));
        }
        self.count += 1;
    }

    /// Get the last element of a vector, as well as the vector with the last
    /// element removed.
    ///
    /// If the vector is empty, [`None`][None] is returned.
    ///
    /// Time: O(1)*
    ///
    /// [None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    pub fn pop(&self) -> Option<(Arc<A>, Self)> {
        if self.count == 0 {
            return None;
        }
        if self.count == 1 {
            return Some((self.tail[0].unwrap_val(), Vector::new()));
        }
        if self.tail.len() > 1 {
            let mut tail = (*self.tail).clone();
            let value = tail.pop().unwrap().unwrap_val();
            Some((
                value,
                Vector {
                    count: self.count - 1,
                    shift: self.shift,
                    root: self.root.clone(),
                    tail: Arc::new(tail),
                },
            ))
        } else {
            match self.node_for(self.count - 2) {
                None => panic!("Vector::pop: unexpected non-node from node_for"),
                Some(ref node) => {
                    let tail = node.children.clone();
                    let mut shift = self.shift;
                    let mut root = Arc::new(
                        nodes::pop_tail(self.count, self.shift, &self.root).unwrap_or_default(),
                    );
                    if self.shift > HASH_BITS && root.children[1].is_empty() {
                        root = root.children[0].unwrap_node();
                        shift -= HASH_BITS;
                    }
                    Some((
                        self.get(self.count - 1).unwrap(),
                        Vector {
                            count: self.count - 1,
                            shift,
                            root,
                            tail: Arc::new(tail),
                        },
                    ))
                }
            }
        }
    }

    /// Remove the last element of a vector in place and return it.
    ///
    /// This is a copy-on-write operation, so that the parts of the vector's
    /// structure which are shared with other vectors will be safely copied
    /// before mutating.
    ///
    /// Time: O(1)*
    pub fn pop_mut(&mut self) -> Option<Arc<A>> {
        if self.count == 0 {
            return None;
        }
        if self.count == 1 || (self.count - 1) & HASH_MASK as usize > 0 {
            self.count -= 1;
            let tail = Arc::make_mut(&mut self.tail);
            return tail.pop().map(Entry::into_val);
        }
        let value = self.get(self.count - 1);
        match self.node_for(self.count - 2) {
            None => panic!("Vector::pop_mut: unexpected non-node from node_for"),
            Some(tail_node) => {
                let tail = tail_node.children.clone();
                let mut set_root = None;
                {
                    let mut root = Arc::make_mut(&mut self.root);
                    if nodes::pop_tail_mut(self.count, self.shift, root) {
                        if self.shift > HASH_BITS && root.children[1].is_empty() {
                            set_root = Some(root.children[0].unwrap_node());
                            self.shift -= HASH_BITS;
                        }
                    } else {
                        set_root = Some(Arc::new(Node::new()));
                    }
                }
                if let Some(root) = set_root {
                    self.root = root;
                }
                self.tail = Arc::new(tail);
            }
        }
        self.count -= 1;
        value
    }

    /// Append the vector `other` to the end of the current vector.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![1, 2, 3].append(vector![7, 8, 9]),
    ///   vector![1, 2, 3, 7, 8, 9]
    /// );
    /// # }
    /// ```
    pub fn append<R>(&self, other: R) -> Self
    where
        R: Borrow<Self>,
    {
        let mut out = self.clone();
        out.extend(other.borrow());
        out
    }

    /// Construct a vector which is the reverse of the current vector.
    ///
    /// Please note that if all you want is to iterate over the vector from back to front,
    /// it is much more efficient to use a [reversed iterator][rev] rather than doing
    /// the work of reversing the vector first.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![1, 2, 3, 4, 5].reverse(),
    ///   vector![5, 4, 3, 2, 1]
    /// );
    /// # }
    /// ```
    ///
    /// [rev]: https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.rev
    pub fn reverse(&self) -> Self {
        self.iter().rev().collect()
    }

    /// Sort a vector of ordered elements.
    ///
    /// Time: O(n log n) roughly
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # fn main() {
    /// assert_eq!(
    ///   vector![2, 8, 1, 6, 3, 7, 5, 4].sort(),
    ///   vector![1, 2, 3, 4, 5, 6, 7, 8]
    /// );
    /// # }
    /// ```
    pub fn sort(&self) -> Self
    where
        A: Ord,
    {
        self.sort_by(|a, b| a.cmp(b))
    }

    /// Sort a vector using a comparator function.
    ///
    /// Time: O(n log n) roughly
    pub fn sort_by<F>(&self, cmp: F) -> Self
    where
        F: Fn(&A, &A) -> Ordering,
    {
        // FIXME: This is a simple in-place quicksort. There are faster algorithms.
        fn swap<A>(vector: &mut Vector<A>, a: usize, b: usize) {
            let aval = vector.get(a).unwrap();
            let bval = vector.get(b).unwrap();
            vector.set_mut(a, bval);
            vector.set_mut(b, aval);
        }

        // Ported from the Java version at
        // http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
        #[cfg_attr(feature = "clippy", allow(many_single_char_names))]
        fn quicksort<A, F>(vector: &mut Vector<A>, l: usize, r: usize, cmp: &F)
        where
            F: Fn(&A, &A) -> Ordering,
        {
            if r <= l {
                return;
            }

            let mut i = l;
            let mut j = r;
            let mut p = i;
            let mut q = j;
            let v = vector.get(r).unwrap();
            loop {
                while cmp(&vector.get_unwrapped(i), &v) == Ordering::Less {
                    i += 1
                }
                j -= 1;
                while cmp(&v, &vector.get_unwrapped(j)) == Ordering::Less {
                    if j == l {
                        break;
                    }
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                swap(vector, i, j);
                if cmp(&vector.get_unwrapped(i), &v) == Ordering::Equal {
                    p += 1;
                    swap(vector, p, i);
                }
                if cmp(&v, &vector.get_unwrapped(j)) == Ordering::Equal {
                    q -= 1;
                    swap(vector, j, q);
                }
                i += 1;
            }
            swap(vector, i, r);
            // FIXME: This is annoying: i - 1 will be negative occasionally,
            // can't see an immediate workaround other than allowing it.
            let mut jp: isize = i as isize - 1;
            let mut k = l;
            i += 1;
            while k < p {
                swap(vector, k, jp as usize);
                jp -= 1;
                k += 1;
            }
            k = r - 1;
            while k > q {
                swap(vector, i, k);
                k -= 1;
                i += 1;
            }

            if jp >= 0 {
                quicksort(vector, l, jp as usize, cmp);
            }
            quicksort(vector, i, r, cmp);
        }

        if self.len() < 2 {
            self.clone()
        } else {
            let mut out = self.clone();
            quicksort(&mut out, 0, self.len() - 1, &cmp);
            out
        }
    }

    /// Make a `PartialLens` from the vector to the value at the
    /// given `index`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # use std::sync::Arc;
    /// # use im::lens::{self, PartialLens};
    /// # fn main() {
    /// let vector = vector!["foo"];
    /// let lens = Vector::lens(0);
    /// assert_eq!(lens.try_get(&vector), Some(Arc::new("foo")));
    /// # }
    /// ```
    pub fn lens(index: usize) -> VectorLens<A> {
        VectorLens {
            index,
            value: PhantomData,
        }
    }

    // Implementation details

    fn adopt(v: Vec<Entry<A>>) -> Self {
        Vector {
            count: v.len(),
            shift: HASH_BITS,
            root: Arc::new(Node::new()),
            tail: Arc::new(v),
        }
    }

    fn tailoff(&self) -> usize {
        if self.count < HASH_SIZE {
            0
        } else {
            ((self.count - 1) >> HASH_BITS) << HASH_BITS
        }
    }

    fn node_for(&self, index: usize) -> Option<Arc<Node<A>>> {
        if index >= self.tailoff() {
            None
        } else {
            let mut node = self.root.clone();
            let mut level = self.shift;
            while level > 0 {
                node = if let Some(&Entry::Node(ref child_node)) =
                    node.children.get((index >> level) & HASH_MASK as usize)
                {
                    level -= HASH_BITS;
                    child_node.clone()
                } else {
                    panic!("Vector::node_for: encountered value or null where node was expected")
                };
            }
            Some(node)
        }
    }

    fn update_root(&self, root: Node<A>) -> Self {
        Vector {
            count: self.count,
            shift: self.shift,
            root: Arc::new(root),
            tail: self.tail.clone(),
        }
    }

    fn update_tail(&self, tail: Vec<Entry<A>>) -> Self {
        Vector {
            count: self.count,
            shift: self.shift,
            root: self.root.clone(),
            tail: Arc::new(tail),
        }
    }
}

// Core traits

impl<A> Clone for Vector<A> {
    fn clone(&self) -> Self {
        Vector {
            count: self.count,
            shift: self.shift,
            root: self.root.clone(),
            tail: self.tail.clone(),
        }
    }
}

impl<A> Default for Vector<A> {
    fn default() -> Self {
        Vector::new()
    }
}

impl<A: Debug> Debug for Vector<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "[")?;
        let mut it = self.iter().peekable();
        while let Some(item) = it.next() {
            write!(f, "{:?}", item)?;
            if it.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(not(has_specialisation))]
impl<A: PartialEq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.count != other.count {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: PartialEq> PartialEq for Vector<A> {
    default fn eq(&self, other: &Self) -> bool {
        if self.count != other.count {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<A: Eq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.count != other.count {
            return false;
        }
        if Arc::ptr_eq(&self.root, &other.root) && Arc::ptr_eq(&self.tail, &other.tail) {
            return true;
        }
        self.iter().eq(other.iter())
    }
}

impl<A: Eq> Eq for Vector<A> {}

impl<A: PartialOrd> PartialOrd for Vector<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A: Ord> Ord for Vector<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A> Add for Vector<A> {
    type Output = Vector<A>;

    fn add(mut self, other: Self) -> Self::Output {
        self.extend(other);
        self
    }
}

impl<'a, A> Add for &'a Vector<A> {
    type Output = Vector<A>;

    fn add(self, other: Self) -> Self::Output {
        let mut out = self.clone();
        out.extend(other);
        out
    }
}

impl<A: Hash> Hash for Vector<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self {
            i.hash(state)
        }
    }
}

impl<A> Sum for Vector<A> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, R: Shared<A>> Extend<R> for Vector<A> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = R>,
    {
        for item in iter {
            self.push_mut(item)
        }
    }
}

// Conversions

pub struct VectorLens<A> {
    index: usize,
    value: PhantomData<A>,
}

impl<A> Clone for VectorLens<A> {
    fn clone(&self) -> Self {
        VectorLens {
            index: self.index,
            value: PhantomData,
        }
    }
}

impl<A> PartialLens for VectorLens<A> {
    type From = Vector<A>;
    type To = A;

    fn try_get(&self, s: &Self::From) -> Option<Arc<Self::To>> {
        s.get(self.index)
    }

    fn try_put<Convert>(&self, cv: Option<Convert>, s: &Self::From) -> Option<Self::From>
    where
        Convert: Shared<Self::To>,
    {
        match cv.map(Shared::shared) {
            None => panic!("can't remove from a vector through a lens"),
            Some(v) => if self.index < s.len() {
                Some(s.set(self.index, v))
            } else {
                None
            },
        }
    }
}

impl<A, RA: Shared<A>> FromIterator<RA> for Vector<A> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        let mut v = Vec::with_capacity(HASH_SIZE);
        let mut it = iter.into_iter().peekable();
        while let Some(item) = it.next() {
            v.push(Entry::Value(item.shared()));
            if v.len() == HASH_SIZE {
                break;
            }
        }
        if v.len() == HASH_SIZE && it.peek().is_some() {
            let mut out = Vector::adopt(v);
            for item in it {
                out.push_mut(item);
            }
            out
        } else {
            Vector::adopt(v)
        }
    }
}

impl<A> IntoIterator for Vector<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A> IntoIterator for &'a Vector<A> {
    type Item = Arc<A>;
    type IntoIter = Iter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A> From<Vec<A>> for Vector<A> {
    fn from(v: Vec<A>) -> Self {
        v.into_iter().collect()
    }
}

// Iterators

/// An iterator over vectors with values of type `A`.
pub struct Iter<A> {
    vector: Vector<A>,
    start_node: Option<Arc<Node<A>>>,
    start_index: usize,
    start_offset: usize,
    end_node: Option<Arc<Node<A>>>,
    end_index: usize,
    end_offset: usize,
}

impl<A> Iter<A> {
    fn new(vector: &Vector<A>) -> Self {
        let end_index = vector.len() & !(HASH_MASK as usize);
        Iter {
            vector: vector.clone(),
            start_node: vector.node_for(0),
            end_node: vector.node_for(end_index),
            start_index: 0,
            start_offset: 0,
            end_index,
            end_offset: vector.len() - end_index,
        }
    }
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_index + self.start_offset == self.end_index + self.end_offset {
            return None;
        }
        match self.start_node {
            None => {
                let item = self.vector.tail[self.start_offset].unwrap_val();
                self.start_offset += 1;
                return Some(item);
            }
            Some(ref node) => {
                if self.start_offset < HASH_SIZE {
                    let item = node.children[self.start_offset].unwrap_val();
                    self.start_offset += 1;
                    return Some(item);
                }
            }
        }
        self.start_offset = 0;
        self.start_index += HASH_SIZE;
        self.start_node = self.vector.node_for(self.start_index);
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.end_index + self.end_offset) - (self.start_index + self.start_offset);
        (size, Some(size))
    }
}

impl<A> DoubleEndedIterator for Iter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start_index + self.start_offset == self.end_index + self.end_offset {
            return None;
        }
        match self.end_node {
            None => {
                if self.end_offset > 0 {
                    self.end_offset -= 1;
                    let item = self.vector.tail[self.end_offset].unwrap_val();
                    return Some(item);
                }
            }
            Some(ref node) => {
                if self.end_offset > 0 {
                    self.end_offset -= 1;
                    let item = node.children[self.end_offset].unwrap_val();
                    return Some(item);
                }
            }
        }
        self.end_offset = HASH_SIZE;
        self.end_index -= HASH_SIZE;
        self.end_node = self.vector.node_for(self.end_index);
        self.next_back()
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<A: Arbitrary + Sync> Arbitrary for Vector<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Vector::from_iter(Vec::<A>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for generating a vector of a certain size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_a_vector(ref l in vector(".*", 10..100)) {
    ///         assert!(l.len() < 100);
    ///         assert!(l.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn vector<T: Strategy + 'static>(
        element: T,
        size: Range<usize>,
    ) -> BoxedStrategy<Vector<<T::Value as ValueTree>::Value>> {
        ::proptest::collection::vec(element, size)
            .prop_map(Vector::from_iter)
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use super::proptest::*;
    use std::iter;
    use proptest::num::i32;
    use proptest::collection;

    #[test]
    fn double_ended_iterator() {
        let vector = Vector::<i32>::from_iter(1..6);
        let mut it = vector.iter();
        assert_eq!(Some(Arc::new(1)), it.next());
        assert_eq!(Some(Arc::new(5)), it.next_back());
        assert_eq!(Some(Arc::new(2)), it.next());
        assert_eq!(Some(Arc::new(4)), it.next_back());
        assert_eq!(Some(Arc::new(3)), it.next());
        assert_eq!(None, it.next_back());
        assert_eq!(None, it.next());
    }

    proptest! {
        #[test]
        fn push(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector = vector.push(value);
                assert_eq!(count + 1, vector.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn push_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector.push_mut(value);
                assert_eq!(count + 1, vector.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn from_iter(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn set(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(iter::repeat(0).take(input.len()));
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                vector = vector.set(index, value);
            }
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn set_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(iter::repeat(0).take(input.len()));
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                vector.set_mut(index, value);
            }
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn pop(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop() {
                    None => panic!("vector emptied unexpectedly"),
                    Some((item, next)) => {
                        vector = next;
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn pop_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop_mut() {
                    None => panic!("vector emptied unexpectedly"),
                    Some(item) => {
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            let mut it1 = input.iter().cloned();
            let mut it2 = vector.iter();
            loop {
                match (it1.next(), it2.next()) {
                    (None, None) => break,
                    (Some(i1), Some(i2)) => assert_eq!(i1, *i2),
                    (Some(i1), None) => panic!(format!("expected {:?} but got EOF", i1)),
                    (None, Some(i2)) => panic!(format!("expected EOF but got {:?}", *i2)),
                }
            }
        }

        #[test]
        fn reverse_iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            let mut it1 = input.iter().cloned().rev();
            let mut it2 = vector.iter().rev();
            loop {
                match (it1.next(), it2.next()) {
                    (None, None) => break,
                    (Some(i1), Some(i2)) => assert_eq!(i1, *i2),
                    (Some(i1), None) => panic!(format!("expected {:?} but got EOF", i1)),
                    (None, Some(i2)) => panic!(format!("expected EOF but got {:?}", *i2)),
                }
            }
        }

        #[test]
        fn exact_size_iterator(ref vector in vector(i32::ANY, 0..100)) {
            let mut should_be = vector.len();
            let mut it = vector.iter();
            loop {
                assert_eq!(should_be, it.len());
                match it.next() {
                    None => break,
                    Some(_) => should_be -= 1,
                }
            }
            assert_eq!(0, it.len());
        }

        #[test]
        fn sort(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            let mut input_sorted = input.clone();
            input_sorted.sort();
            let sorted = Vector::from_iter(input_sorted.iter().cloned());
            assert_eq!(sorted, vector.sort());
        }
    }
}
