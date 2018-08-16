use std::mem::{replace, swap};
use std::ops::{Range, RangeBounds};
use std::ptr::{null, null_mut};

use nodes::chunk::Chunk;
use nodes::rrb::Node;
use util::{to_range, Ref};
use vector::{Iter, Vector, RRB};

/// Focused indexing over a `Vector`.
///
/// By remembering the last tree node accessed through an index lookup and the
/// path we took to get there, we can speed up lookups for adjacent indices
/// tremendously. Lookups on indices in the same node are instantaneous, and
/// lookups on sibling nodes are also very fast.
///
/// A `Focus` can also be used as a restricted view into a vector, using the
/// `narrow` and `split_at` methods.
///
/// # When should I use a `Focus` for better performance?
///
/// `Focus` is useful when you need to perform a large number of index lookups
/// that are more likely than not to be close to each other. It's usually worth
/// using a `Focus` in any situation where you're batching a lot of index
/// lookups together, even if they're not obviously adjacent - there's likely
/// to be some performance gain for even completely random access.
///
/// If you're just iterating forwards or backwards over the `Vector` in order,
/// you're better off with a regular `Iterator`.
///
/// If you're just doing a very small number of index lookups, the setup cost
/// for the `Focus` is probably not worth it.
///
/// A `Focus` is never faster than an index lookup on a small `Vector` with a
/// length below the internal RRB tree's branching factor of 64.
///
/// # Examples
///
/// This example is contrived, as the better way to iterate forwards or
/// backwards over a vector is with an actual iterator. Even so, the version
/// using a `Focus` should run nearly an order of magnitude faster than the
/// version using index lookups at a length of 1000. It should also be noted
/// that `vector::Iter` is actually implemented using a `Focus` behind the
/// scenes, so the performance of the two should be identical.
///
/// ```rust
/// # #[macro_use] extern crate im;
/// # use im::vector::Vector;
/// # use std::iter::FromIterator;
/// # fn main() {
/// let mut vec = Vector::from_iter(0..1000);
///
/// // Summing a vector, the slow way:
/// let mut sum = 0;
/// for i in 0..1000 {
///     sum += *vec.get(i).unwrap();
/// }
/// assert_eq!(499500, sum);
///
/// // Summing a vector faster using a Focus:
/// let mut sum = 0;
/// let mut focus = vec.focus();
/// for i in 0..1000 {
///     sum += *focus.get(i).unwrap();
/// }
/// assert_eq!(499500, sum);
/// # }
/// ```
pub enum Focus<'a, A>
where
    A: 'a,
{
    #[doc(hidden)]
    Single(&'a [A]),
    #[doc(hidden)]
    Full(TreeFocus<A>),
}

impl<'a, A> Focus<'a, A>
where
    A: Clone + 'a,
{
    /// Construct a `Focus` for a `Vector`.
    pub fn new(vector: &'a Vector<A>) -> Self {
        match vector {
            Vector::Single(chunk) => Focus::Single(chunk.as_slice()),
            Vector::Full(tree) => Focus::Full(TreeFocus::new(tree)),
        }
    }

    /// Get the length of the focused `Vector`.
    pub fn len(&self) -> usize {
        match self {
            Focus::Single(chunk) => chunk.len(),
            Focus::Full(tree) => tree.len(),
        }
    }

    /// Test if the focused `Vector` is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the value at a given index.
    pub fn get(&mut self, index: usize) -> Option<&A> {
        match self {
            Focus::Single(chunk) => chunk.get(index),
            Focus::Full(tree) => tree.get(index),
        }
    }

    /// Get a reference to the value at a given index.
    ///
    /// Panics if the index is out of bounds.
    pub fn index(&mut self, index: usize) -> &A {
        self.get(index).expect("index out of bounds")
    }

    /// Get the chunk for the given index.
    ///
    /// This gives you a reference to the leaf node that contains the index,
    /// along with its start and end indices.
    pub fn chunk_at(&mut self, index: usize) -> (Range<usize>, &[A]) {
        let len = self.len();
        if index >= len {
            panic!("vector::Focus::chunk_at: index out of bounds");
        }
        match self {
            Focus::Single(chunk) => (0..len, chunk),
            Focus::Full(tree) => tree.get_chunk(index),
        }
    }

    /// Narrow the focus onto a subslice of the vector.
    ///
    /// `Focus::narrow(range)` has the same effect as `&slice[range]`, without
    /// actually modifying the underlying vector.
    ///
    /// Panics if the range isn't fully inside the current focus.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # use std::iter::FromIterator;
    /// # fn main() {
    /// let vec = Vector::from_iter(0..1000);
    /// let narrowed = vec.focus().narrow(100..200);
    /// let narrowed_vec = narrowed.into_iter().cloned().collect();
    /// assert_eq!(Vector::from_iter(100..200), narrowed_vec);
    /// # }
    /// ```
    ///
    /// [slice::split_at]: https://doc.rust-lang.org/std/primitive.slice.html#method.split_at
    /// [Vector::split_at]: enum.Vector.html#method.split_at
    pub fn narrow<R>(self, range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let r = to_range(&range, self.len());
        if r.start >= r.end || r.start >= self.len() {
            panic!("vector::Focus::narrow: range out of bounds");
        }
        match self {
            Focus::Single(chunk) => Focus::Single(&chunk[r]),
            Focus::Full(tree) => Focus::Full(tree.narrow(r)),
        }
    }
    /// Split the focus into two.
    ///
    /// Given an index `index`, consume the focus and produce two new foci, the
    /// left onto indices `0..index`, and the right onto indices `index..N`
    /// where `N` is the length of the current focus.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// This is the moral equivalent of [`slice::split_at`][slice::split_at], in
    /// that it leaves the underlying data structure unchanged, unlike
    /// [`Vector::split_at`][Vector::split_at].
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # use std::iter::FromIterator;
    /// # fn main() {
    /// let vec = Vector::from_iter(0..1000);
    /// let (left, right) = vec.focus().split_at(500);
    /// let left_vec = left.into_iter().cloned().collect();
    /// let right_vec = right.into_iter().cloned().collect();
    /// assert_eq!(Vector::from_iter(0..500), left_vec);
    /// assert_eq!(Vector::from_iter(500..1000), right_vec);
    /// # }
    /// ```
    ///
    /// [slice::split_at]: https://doc.rust-lang.org/std/primitive.slice.html#method.split_at
    /// [Vector::split_at]: enum.Vector.html#method.split_at
    pub fn split_at(self, index: usize) -> (Self, Self) {
        if index >= self.len() {
            panic!("vector::Focus::split_at: index out of bounds");
        }
        match self {
            Focus::Single(chunk) => {
                let (left, right) = chunk.split_at(index);
                (Focus::Single(left), Focus::Single(right))
            }
            Focus::Full(tree) => {
                let (left, right) = tree.split_at(index);
                (Focus::Full(left), Focus::Full(right))
            }
        }
    }
}

impl<'a, A> IntoIterator for Focus<'a, A>
where
    A: Clone + 'a,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::from_focus(self)
    }
}

pub struct TreeFocus<A> {
    tree: RRB<A>,
    view: Range<usize>,
    middle_range: Range<usize>,
    root: *const Node<A>,
    target_range: Range<usize>,
    target_ptr: *const Chunk<A>,
}

impl<A> Clone for TreeFocus<A> {
    fn clone(&self) -> Self {
        let tree = self.tree.clone();
        TreeFocus {
            view: self.view.clone(),
            middle_range: self.middle_range.clone(),
            root: &*tree.middle,
            target_range: 0..0,
            target_ptr: null(),
            tree,
        }
    }
}

#[allow(unsafe_code)]
#[cfg(ref_arc)]
unsafe impl<A> Send for TreeFocus<A> {}
#[allow(unsafe_code)]
#[cfg(ref_arc)]
unsafe impl<A> Sync for TreeFocus<A> {}

#[inline]
fn contains<A: Ord>(range: &Range<A>, index: &A) -> bool {
    *index >= range.start && *index < range.end
}

impl<A> TreeFocus<A>
where
    A: Clone,
{
    fn new(tree: &RRB<A>) -> Self {
        let middle_start = tree.outer_f.len() + tree.inner_f.len();
        let middle_end = middle_start + tree.middle.len();
        TreeFocus {
            tree: tree.clone(),
            view: 0..tree.length,
            middle_range: middle_start..middle_end,
            root: &*tree.middle,
            target_range: 0..0,
            target_ptr: null(),
        }
    }

    fn len(&self) -> usize {
        self.view.end - self.view.start
    }

    fn narrow(self, mut view: Range<usize>) -> Self {
        view.start += self.view.start;
        view.end += self.view.start;
        TreeFocus {
            view,
            middle_range: self.middle_range.clone(),
            root: &*self.tree.middle,
            target_range: 0..0,
            target_ptr: null(),
            tree: self.tree,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let len = self.len();
        let left = self.clone().narrow(0..index);
        let right = self.narrow(index..len);
        (left, right)
    }

    fn physical_index(&self, index: usize) -> usize {
        debug_assert!(index < self.view.end);
        self.view.start + index
    }

    fn logical_range(&self, range: &Range<usize>) -> Range<usize> {
        (range.start - self.view.start)..(range.end - self.view.start)
    }

    #[allow(unsafe_code)]
    fn set_focus(&mut self, index: usize) {
        // FIXME use path to shorten lookup if available
        if index < self.middle_range.start {
            let outer_len = self.tree.outer_f.len();
            if index < outer_len {
                self.target_range = 0..outer_len;
                self.target_ptr = &*self.tree.outer_f;
            } else {
                self.target_range = outer_len..self.middle_range.start;
                self.target_ptr = &*self.tree.inner_f;
            }
        } else if index >= self.middle_range.end {
            let outer_start = self.middle_range.end + self.tree.inner_b.len();
            if index < outer_start {
                self.target_range = self.middle_range.end..outer_start;
                self.target_ptr = &*self.tree.inner_b;
            } else {
                self.target_range = outer_start..self.tree.length;
                self.target_ptr = &*self.tree.outer_b;
            }
        } else {
            let tree_index = index - self.middle_range.start;
            let (range, ptr) =
                unsafe { &*self.root }.lookup_chunk(self.tree.middle_level, 0, tree_index);
            self.target_range =
                (range.start + self.middle_range.start)..(range.end + self.middle_range.start);
            self.target_ptr = ptr;
        }
    }

    #[allow(unsafe_code)]
    fn get_focus(&self) -> &Chunk<A> {
        unsafe { &*self.target_ptr }
    }

    pub fn get(&mut self, index: usize) -> Option<&A> {
        if index >= self.len() {
            return None;
        }
        let phys_index = self.physical_index(index);
        if !contains(&self.target_range, &phys_index) {
            self.set_focus(phys_index);
        }
        let target_phys_index = phys_index - self.target_range.start;
        Some(&self.get_focus()[target_phys_index])
    }

    pub fn get_chunk(&mut self, index: usize) -> (Range<usize>, &[A]) {
        let phys_index = self.physical_index(index);
        if !contains(&self.target_range, &phys_index) {
            self.set_focus(phys_index);
        }
        let mut slice = self.get_focus().as_slice();
        let mut left = 0;
        let mut right = 0;
        if self.target_range.start < self.view.start {
            left = self.view.start - self.target_range.start;
        }
        if self.target_range.end > self.view.end {
            right = self.target_range.end - self.view.end;
        }
        slice = &slice[left..(slice.len() - right)];
        let phys_range = (self.target_range.start + left)..(self.target_range.end - right);
        (self.logical_range(&phys_range), slice)
    }
}

/// A mutable version of [`Focus`][Focus].
///
/// See [`Focus`][Focus] for more details.
///
/// [Focus]: enum.Focus.html
pub enum FocusMut<'a, A>
where
    A: 'a,
{
    #[doc(hidden)]
    Single(&'a mut Chunk<A>),
    #[doc(hidden)]
    Full(TreeFocusMut<'a, A>),
}

impl<'a, A> FocusMut<'a, A>
where
    A: Clone + 'a,
{
    /// Construct a `FocusMut` for a `Vector`.
    pub fn new(vector: &'a mut Vector<A>) -> Self {
        match vector {
            Vector::Single(chunk) => FocusMut::Single(chunk),
            Vector::Full(tree) => FocusMut::Full(TreeFocusMut::new(tree)),
        }
    }

    /// Get the length of the focused `Vector`.
    pub fn len(&self) -> usize {
        match self {
            FocusMut::Single(chunk) => chunk.len(),
            FocusMut::Full(tree) => tree.tree.length,
        }
    }

    /// Test if the focused `Vector` is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the value at a given index.
    pub fn get(&mut self, index: usize) -> Option<&A> {
        self.get_mut(index).map(|r| &*r)
    }

    /// Get a mutable reference to the value at a given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut A> {
        match self {
            FocusMut::Single(chunk) => chunk.get_mut(index),
            FocusMut::Full(tree) => tree.get(index),
        }
    }

    /// Get a reference to the value at a given index.
    ///
    /// Panics if the index is out of bounds.
    pub fn index(&mut self, index: usize) -> &A {
        &*self.index_mut(index)
    }

    /// Get a mutable reference to the value at a given index.
    ///
    /// Panics if the index is out of bounds.
    #[allow(unknown_lints)]
    #[allow(should_implement_trait)] // would if I could
    pub fn index_mut(&mut self, index: usize) -> &mut A {
        match self {
            FocusMut::Single(chunk) => chunk.get_mut(index).expect("index out of bounds"),
            FocusMut::Full(tree) => tree.get(index).expect("index out of bounds"),
        }
    }

    /// Update the value at a given index, returning the replaced value.
    pub fn set(&mut self, index: usize, value: A) -> Option<A> {
        match self.get_mut(index) {
            Some(ref mut pos) => Some(replace(pos, value)),
            None => None,
        }
    }

    /// Swap the values at two given indices.
    ///
    /// Panics if either index is out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.pair(a, b, |left, right| swap(left, right));
    }

    /// Lookup two indices simultaneously and run a function over them.
    ///
    /// Useful because the borrow checker won't let you have more than one
    /// mutable reference into the same data structure at any given time.
    ///
    /// Panics if either index is out of bounds, or if they are the same index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # use std::iter::FromIterator;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3, 4, 5];
    /// vec.focus_mut().pair(1, 3, |a, b| *a += *b);
    /// assert_eq!(vector![1, 6, 3, 4, 5], vec);
    /// # }
    /// ```
    #[allow(unsafe_code)]
    pub fn pair<F, B>(&mut self, a: usize, b: usize, mut f: F) -> B
    where
        F: FnMut(&mut A, &mut A) -> B,
    {
        if a == b {
            panic!("vector::FocusMut::pair: indices cannot be equal!");
        }
        let pa: *mut A = self.index_mut(a);
        let pb: *mut A = self.index_mut(b);
        unsafe { f(&mut *pa, &mut *pb) }
    }

    /// Lookup three indices simultaneously and run a function over them.
    ///
    /// Useful because the borrow checker won't let you have more than one
    /// mutable reference into the same data structure at any given time.
    ///
    /// Panics if any index is out of bounds, or if any indices are equal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate im;
    /// # use im::vector::Vector;
    /// # use std::iter::FromIterator;
    /// # fn main() {
    /// let mut vec = vector![1, 2, 3, 4, 5];
    /// vec.focus_mut().triplet(0, 2, 4, |a, b, c| *a += *b + *c);
    /// assert_eq!(vector![9, 2, 3, 4, 5], vec);
    /// # }
    /// ```
    #[allow(unsafe_code)]
    pub fn triplet<F, B>(&mut self, a: usize, b: usize, c: usize, mut f: F) -> B
    where
        F: FnMut(&mut A, &mut A, &mut A) -> B,
    {
        if a == b || b == c || a == c {
            panic!("vector::FocusMut::triplet: indices cannot be equal!");
        }
        let pa: *mut A = self.index_mut(a);
        let pb: *mut A = self.index_mut(b);
        let pc: *mut A = self.index_mut(c);
        unsafe { f(&mut *pa, &mut *pb, &mut *pc) }
    }

    /// Get the chunk for the given index.
    ///
    /// This gives you a reference to the leaf node that contains the index,
    /// along with its start and end indices.
    pub fn chunk_at(&mut self, index: usize) -> (Range<usize>, &mut [A]) {
        let len = self.len();
        if index >= len {
            panic!("vector::FocusMut::chunk_at: index out of bounds");
        }
        match self {
            FocusMut::Single(chunk) => (0..len, chunk.as_mut_slice()),
            FocusMut::Full(tree) => {
                let (range, chunk) = tree.get_chunk(index);
                (range, chunk.as_mut_slice())
            }
        }
    }
}

pub struct TreeFocusMut<'a, A>
where
    A: 'a,
{
    tree: &'a mut RRB<A>,
    middle_range: Range<usize>,
    root: *mut Node<A>,
    target_range: Range<usize>,
    target_ptr: *mut Chunk<A>,
}

#[allow(unsafe_code)]
#[cfg(ref_arc)]
unsafe impl<'a, A> Send for TreeFocusMut<'a, A> {}
#[allow(unsafe_code)]
#[cfg(ref_arc)]
unsafe impl<'a, A> Sync for TreeFocusMut<'a, A> {}

impl<'a, A> TreeFocusMut<'a, A>
where
    A: Clone + 'a,
{
    fn new(tree: &'a mut RRB<A>) -> Self {
        let middle_start = tree.outer_f.len() + tree.inner_f.len();
        let middle_end = middle_start + tree.middle.len();
        let root: *mut Node<A> = Ref::make_mut(&mut tree.middle);
        TreeFocusMut {
            tree,
            root,
            middle_range: middle_start..middle_end,
            target_range: 0..0,
            target_ptr: null_mut(),
        }
    }

    #[allow(unsafe_code)]
    fn set_focus(&mut self, index: usize) {
        // FIXME use path to shorten lookup if available
        if index < self.middle_range.start {
            let outer_len = self.tree.outer_f.len();
            if index < outer_len {
                self.target_range = 0..outer_len;
                self.target_ptr = Ref::make_mut(&mut self.tree.outer_f);
            } else {
                self.target_range = outer_len..self.middle_range.start;
                self.target_ptr = Ref::make_mut(&mut self.tree.inner_f);
            }
        } else if index >= self.middle_range.end {
            let outer_start = self.middle_range.end + self.tree.inner_b.len();
            if index < outer_start {
                self.target_range = self.middle_range.end..outer_start;
                self.target_ptr = Ref::make_mut(&mut self.tree.inner_b);
            } else {
                self.target_range = outer_start..self.tree.length;
                self.target_ptr = Ref::make_mut(&mut self.tree.outer_b);
            }
        } else {
            let tree_index = index - self.middle_range.start;
            let (range, ptr) =
                unsafe { &mut *self.root }.lookup_chunk_mut(self.tree.middle_level, 0, tree_index);
            self.target_range =
                (range.start + self.middle_range.start)..(range.end + self.middle_range.start);
            self.target_ptr = ptr;
        }
    }

    #[allow(unsafe_code)]
    fn get_focus(&mut self) -> &mut Chunk<A> {
        unsafe { &mut *self.target_ptr }
    }

    pub fn get(&mut self, index: usize) -> Option<&mut A> {
        if index >= self.tree.length {
            return None;
        }
        if !contains(&self.target_range, &index) {
            self.set_focus(index);
        }
        let target_index = index - self.target_range.start;
        Some(&mut self.get_focus()[target_index])
    }

    pub fn get_chunk(&mut self, index: usize) -> (Range<usize>, &mut Chunk<A>) {
        if !contains(&self.target_range, &index) {
            self.set_focus(index);
        }
        (self.target_range.clone(), self.get_focus())
    }
}
