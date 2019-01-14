// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A fixed capacity smart array.
//!
//! See [`Chunk`](struct.Chunk.html)

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::io;
use std::iter::{FromIterator, FusedIterator};
use std::mem::{self, replace, ManuallyDrop};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr;
use std::slice::{
    from_raw_parts, from_raw_parts_mut, Iter as SliceIter, IterMut as SliceIterMut, SliceIndex,
};

use typenum::U64;

use nodes::types::ChunkLength;

/// A fixed capacity smart array.
///
/// An inline array of items with a variable length but a fixed, preallocated
/// capacity given by the `N` type, which must be an [`Unsigned`][Unsigned] type
/// level numeral.
///
/// It's 'smart' because it's able to reorganise its contents based on expected
/// behaviour. If you construct one using `push_back`, it will be laid out like
/// a `Vec` with space at the end. If you `push_front` it will start filling in
/// values from the back instead of the front, so that you still get linear time
/// push as long as you don't reverse direction. If you do, and there's no room
/// at the end you're pushing to, it'll shift its contents over to the other
/// side, creating more space to push into. This technique is tuned for
/// `Chunk`'s expected use case: usually, chunks always see either `push_front`
/// or `push_back`, but not both unless they move around inside the tree, in
/// which case they're able to reorganise themselves with reasonable efficiency
/// to suit their new usage patterns.
///
/// It maintains a `left` index and a `right` index instead of a simple length
/// counter in order to accomplish this, much like a ring buffer would, except
/// that the `Chunk` keeps all its items sequentially in memory so that you can
/// always get a `&[A]` slice for them, at the price of the occasional
/// reordering operation.
///
/// This technique also lets us choose to shift the shortest side to account for
/// the inserted or removed element when performing insert and remove
/// operations, unlike `Vec` where you always need to shift the right hand side.
///
/// Unlike a `Vec`, the `Chunk` has a fixed capacity and cannot grow beyond it.
/// Being intended for low level use, it expects you to know or test whether
/// you're pushing to a full array, and has an API more geared towards panics
/// than returning `Option`s, on the assumption that you know what you're doing.
///
/// # Examples
///
/// ```rust
/// # #[macro_use] extern crate im;
/// # extern crate typenum;
/// # use im::chunk::Chunk;
/// # use typenum::U64;
/// # fn main() {
/// // Construct a chunk with a 64 item capacity
/// let mut chunk = Chunk::<i32, U64>::new();
/// // Fill it with descending numbers
/// chunk.extend((0..64).rev());
/// // It derefs to a slice so we can use standard slice methods
/// chunk.sort();
/// // It's got all the amenities like `FromIterator` and `Eq`
/// let expected: Chunk<i32, U64> = (0..64).collect();
/// assert_eq!(expected, chunk);
/// # }
/// ```
///
/// [Unsigned]: https://docs.rs/typenum/1.10.0/typenum/marker_traits/trait.Unsigned.html
pub struct Chunk<A, N = U64>
where
    N: ChunkLength<A>,
{
    left: usize,
    right: usize,
    data: ManuallyDrop<N::SizedType>,
}

impl<A, N> Drop for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn drop(&mut self) {
        if mem::needs_drop::<A>() {
            for i in self.left..self.right {
                unsafe { Chunk::force_drop(i, self) }
            }
        }
    }
}

impl<A, N> Clone for Chunk<A, N>
where
    A: Clone,
    N: ChunkLength<A>,
{
    fn clone(&self) -> Self {
        let mut out = Self::new();
        out.left = self.left;
        out.right = self.right;
        for index in self.left..self.right {
            unsafe { Chunk::force_write(index, self.values()[index].clone(), &mut out) }
        }
        out
    }
}

impl<A, N> Chunk<A, N>
where
    N: ChunkLength<A>,
{
    /// Construct a new empty chunk.
    pub fn new() -> Self {
        let mut chunk: Self;
        unsafe {
            chunk = mem::uninitialized();
            ptr::write(&mut chunk.left, 0);
            ptr::write(&mut chunk.right, 0);
        }
        chunk
    }

    /// Construct a new chunk with one item.
    pub fn unit(value: A) -> Self {
        let mut chunk: Self;
        unsafe {
            chunk = mem::uninitialized();
            ptr::write(&mut chunk.left, 0);
            ptr::write(&mut chunk.right, 1);
            Chunk::force_write(0, value, &mut chunk);
        }
        chunk
    }

    /// Construct a new chunk with two items.
    pub fn pair(left: A, right: A) -> Self {
        let mut chunk: Self;
        unsafe {
            chunk = mem::uninitialized();
            ptr::write(&mut chunk.left, 0);
            ptr::write(&mut chunk.right, 2);
            Chunk::force_write(0, left, &mut chunk);
            Chunk::force_write(1, right, &mut chunk);
        }
        chunk
    }

    /// Construct a new chunk and move every item from `other` into the new
    /// chunk.
    ///
    /// Time: O(n)
    pub fn drain_from(other: &mut Self) -> Self {
        let other_len = other.len();
        Self::from_front(other, other_len)
    }

    /// Construct a new chunk and populate it by taking `count` items from the
    /// iterator `iter`.
    ///
    /// Panics if the iterator contains less than `count` items.
    ///
    /// Time: O(n)
    pub fn collect_from<I>(iter: &mut I, mut count: usize) -> Self
    where
        I: Iterator<Item = A>,
    {
        let mut chunk = Self::new();
        while count > 0 {
            count -= 1;
            chunk.push_back(
                iter.next()
                    .expect("Chunk::collect_from: underfull iterator"),
            );
        }
        chunk
    }

    /// Construct a new chunk and populate it by taking `count` items from the
    /// front of `other`.
    ///
    /// Time: O(n) for the number of items moved
    pub fn from_front(other: &mut Self, count: usize) -> Self {
        let other_len = other.len();
        debug_assert!(count <= other_len);
        let mut chunk = Self::new();
        unsafe { Chunk::force_copy_to(other.left, 0, count, other, &mut chunk) };
        chunk.right = count;
        other.left += count;
        chunk
    }

    /// Construct a new chunk and populate it by taking `count` items from the
    /// back of `other`.
    ///
    /// Time: O(n) for the number of items moved
    pub fn from_back(other: &mut Self, count: usize) -> Self {
        let other_len = other.len();
        debug_assert!(count <= other_len);
        let mut chunk = Self::new();
        unsafe { Chunk::force_copy_to(other.right - count, 0, count, other, &mut chunk) };
        chunk.right = count;
        other.right -= count;
        chunk
    }

    /// Get the length of the chunk.
    #[inline]
    pub fn len(&self) -> usize {
        self.right - self.left
    }

    /// Test if the chunk is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.left == self.right
    }

    /// Test if the chunk is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.left == 0 && self.right == N::USIZE
    }

    #[inline]
    fn values(&self) -> &[A] {
        unsafe {
            from_raw_parts(
                &self.data as *const ManuallyDrop<N::SizedType> as *const A,
                N::USIZE,
            )
        }
    }

    #[inline]
    fn values_mut(&mut self) -> &mut [A] {
        unsafe {
            from_raw_parts_mut(
                &mut self.data as *mut ManuallyDrop<N::SizedType> as *mut A,
                N::USIZE,
            )
        }
    }

    /// Copy the value at an index, discarding ownership of the copied value
    #[inline]
    unsafe fn force_read(index: usize, chunk: &mut Self) -> A {
        ptr::read(&chunk.values()[index])
    }

    /// Write a value at an index without trying to drop what's already there
    #[inline]
    unsafe fn force_write(index: usize, value: A, chunk: &mut Self) {
        ptr::write(&mut chunk.values_mut()[index], value)
    }

    /// Drop the value at an index
    #[inline]
    unsafe fn force_drop(index: usize, chunk: &mut Self) {
        ptr::drop_in_place(&mut chunk.values_mut()[index])
    }

    /// Copy a range within a chunk
    #[inline]
    unsafe fn force_copy(from: usize, to: usize, count: usize, chunk: &mut Self) {
        if count > 0 {
            ptr::copy(&chunk.values()[from], &mut chunk.values_mut()[to], count)
        }
    }

    /// Copy a range between chunks
    #[inline]
    unsafe fn force_copy_to(
        from: usize,
        to: usize,
        count: usize,
        chunk: &mut Self,
        other: &mut Self,
    ) {
        if count > 0 {
            ptr::copy_nonoverlapping(&chunk.values()[from], &mut other.values_mut()[to], count)
        }
    }

    /// Push an item to the front of the chunk.
    ///
    /// Panics if the capacity of the chunk is exceeded.
    ///
    /// Time: O(1) if there's room at the front, O(n) otherwise
    pub fn push_front(&mut self, value: A) {
        if self.is_full() {
            panic!("Chunk::push_front: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = N::USIZE;
            self.right = N::USIZE;
        } else if self.left == 0 {
            self.left = N::USIZE - self.right;
            unsafe { Chunk::force_copy(0, self.left, self.right, self) };
            self.right = N::USIZE;
        }
        self.left -= 1;
        unsafe { Chunk::force_write(self.left, value, self) }
    }

    /// Push an item to the back of the chunk.
    ///
    /// Panics if the capacity of the chunk is exceeded.
    ///
    /// Time: O(1) if there's room at the back, O(n) otherwise
    pub fn push_back(&mut self, value: A) {
        if self.is_full() {
            panic!("Chunk::push_back: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = 0;
            self.right = 0;
        } else if self.right == N::USIZE {
            unsafe { Chunk::force_copy(self.left, 0, self.len(), self) };
            self.right = N::USIZE - self.left;
            self.left = 0;
        }
        unsafe { Chunk::force_write(self.right, value, self) }
        self.right += 1;
    }

    /// Pop an item off the front of the chunk.
    ///
    /// Panics if the chunk is empty.
    ///
    /// Time: O(1)
    pub fn pop_front(&mut self) -> A {
        if self.is_empty() {
            panic!("Chunk::pop_front: can't pop from empty chunk");
        } else {
            let value = unsafe { Chunk::force_read(self.left, self) };
            self.left += 1;
            value
        }
    }

    /// Pop an item off the back of the chunk.
    ///
    /// Panics if the chunk is empty.
    ///
    /// Time: O(1)
    pub fn pop_back(&mut self) -> A {
        if self.is_empty() {
            panic!("Chunk::pop_back: can't pop from empty chunk");
        } else {
            self.right -= 1;
            unsafe { Chunk::force_read(self.right, self) }
        }
    }

    /// Discard all items up to but not including `index`.
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// Time: O(n) for the number of items dropped
    pub fn drop_left(&mut self, index: usize) {
        if index > 0 {
            if index > self.len() {
                panic!("Chunk::drop_left: index out of bounds");
            }
            let start = self.left;
            for i in start..(start + index) {
                unsafe { Chunk::force_drop(i, self) }
            }
            self.left += index;
        }
    }

    /// Discard all items from `index` onward.
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// Time: O(n) for the number of items dropped
    pub fn drop_right(&mut self, index: usize) {
        if index > self.len() {
            panic!("Chunk::drop_right: index out of bounds");
        }
        if index == self.len() {
            return;
        }
        let start = self.left + index;
        for i in start..self.right {
            unsafe { Chunk::force_drop(i, self) }
        }
        self.right = start;
    }

    /// Split a chunk into two, the original chunk containing
    /// everything up to `index` and the returned chunk containing
    /// everything from `index` onwards.
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// Time: O(n) for the number of items in the new chunk
    pub fn split_off(&mut self, index: usize) -> Self {
        if index > self.len() {
            panic!("Chunk::split: index out of bounds");
        }
        if index == self.len() {
            return Self::new();
        }
        let mut right_chunk = Self::new();
        let start = self.left + index;
        let len = self.right - start;
        unsafe { Chunk::force_copy_to(start, 0, len, self, &mut right_chunk) };
        right_chunk.right = len;
        self.right = start;
        right_chunk
    }

    /// Remove all items from `other` and append them to the back of `self`.
    ///
    /// Panics if the capacity of the chunk is exceeded.
    ///
    /// Time: O(n) for the number of items moved
    pub fn append(&mut self, other: &mut Self) {
        let self_len = self.len();
        let other_len = other.len();
        if self_len + other_len > N::USIZE {
            panic!("Chunk::append: chunk size overflow");
        }
        if self.right + other_len > N::USIZE {
            unsafe { Chunk::force_copy(self.left, 0, self_len, self) };
            self.right -= self.left;
            self.left = 0;
        }
        unsafe { Chunk::force_copy_to(other.left, self.right, other_len, other, self) };
        self.right += other_len;
        other.left = 0;
        other.right = 0;
    }

    /// Remove `count` items from the front of `other` and append them to the
    /// back of `self`.
    ///
    /// Panics if `self` doesn't have `count` items left, or if `other` has
    /// fewer than `count` items.
    ///
    /// Time: O(n) for the number of items moved
    pub fn drain_from_front(&mut self, other: &mut Self, count: usize) {
        let self_len = self.len();
        let other_len = other.len();
        debug_assert!(self_len + count <= N::USIZE);
        debug_assert!(other_len >= count);
        if self.right + count > N::USIZE {
            unsafe { Chunk::force_copy(self.left, 0, self_len, self) };
            self.right -= self.left;
            self.left = 0;
        }
        unsafe { Chunk::force_copy_to(other.left, self.right, count, other, self) };
        self.right += count;
        other.left += count;
    }

    /// Remove `count` items from the back of `other` and append them to the
    /// front of `self`.
    ///
    /// Panics if `self` doesn't have `count` items left, or if `other` has
    /// fewer than `count` items.
    ///
    /// Time: O(n) for the number of items moved
    pub fn drain_from_back(&mut self, other: &mut Self, count: usize) {
        let self_len = self.len();
        let other_len = other.len();
        debug_assert!(self_len + count <= N::USIZE);
        debug_assert!(other_len >= count);
        if self.left < count {
            self.left = N::USIZE - self.right;
            unsafe { Chunk::force_copy(0, self.left, self.right, self) };
            self.right = N::USIZE;
        }
        unsafe { Chunk::force_copy_to(other.right - count, self.left - count, count, other, self) };
        self.left -= count;
        other.right -= count;
    }

    /// Update the value at index `index`, returning the old value.
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// Time: O(1)
    pub fn set(&mut self, index: usize, value: A) -> A {
        replace(&mut self[index], value)
    }

    /// Insert a new value at index `index`, shifting all the following values
    /// to the right.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(n) for the number of items shifted
    pub fn insert(&mut self, index: usize, value: A) {
        if self.is_full() {
            panic!("Chunk::insert: chunk is full");
        }
        if index > self.len() {
            panic!("Chunk::insert: index out of bounds");
        }
        let real_index = index + self.left;
        let left_size = index;
        let right_size = self.right - real_index;
        if self.right == N::USIZE || (self.left > 0 && left_size < right_size) {
            unsafe {
                Chunk::force_copy(self.left, self.left - 1, left_size, self);
                Chunk::force_write(real_index - 1, value, self);
            }
            self.left -= 1;
        } else {
            unsafe {
                Chunk::force_copy(real_index, real_index + 1, right_size, self);
                Chunk::force_write(real_index, value, self);
            }
            self.right += 1;
        }
    }

    /// Remove the value at index `index`, shifting all the following values to
    /// the left.
    ///
    /// Returns the removed value.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Time: O(n) for the number of items shifted
    pub fn remove(&mut self, index: usize) -> A {
        if index >= self.len() {
            panic!("Chunk::remove: index out of bounds");
        }
        let real_index = index + self.left;
        let value = unsafe { Chunk::force_read(real_index, self) };
        let left_size = index;
        let right_size = self.right - real_index - 1;
        if left_size < right_size {
            unsafe { Chunk::force_copy(self.left, self.left + 1, left_size, self) };
            self.left += 1;
        } else {
            unsafe { Chunk::force_copy(real_index + 1, real_index, right_size, self) };
            self.right -= 1;
        }
        value
    }

    /// Construct an iterator that drains values from the front of the chunk.
    pub fn drain(&mut self) -> Drain<'_, A, N> {
        Drain { chunk: self }
    }

    /// Discard the contents of the chunk.
    ///
    /// Time: O(n)
    pub fn clear(&mut self) {
        self.drop_right(0);
        self.left = 0;
        self.right = 0;
    }

    /// Get a reference to the contents of the chunk as a slice.
    pub fn as_slice(&self) -> &[A] {
        unsafe {
            from_raw_parts(
                (&self.data as *const ManuallyDrop<N::SizedType> as *const A).add(self.left),
                self.len(),
            )
        }
    }

    /// Get a reference to the contents of the chunk as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [A] {
        unsafe {
            from_raw_parts_mut(
                (&mut self.data as *mut ManuallyDrop<N::SizedType> as *mut A).add(self.left),
                self.len(),
            )
        }
    }
}

impl<A, N> Default for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, N, I> Index<I> for Chunk<A, N>
where
    I: SliceIndex<[A]>,
    N: ChunkLength<A>,
{
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<A, N, I> IndexMut<I> for Chunk<A, N>
where
    I: SliceIndex<[A]>,
    N: ChunkLength<A>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<A, N> Debug for Chunk<A, N>
where
    A: Debug,
    N: ChunkLength<A>,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str("Chunk")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<A, N> Hash for Chunk<A, N>
where
    A: Hash,
    N: ChunkLength<A>,
{
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        for item in self {
            item.hash(hasher)
        }
    }
}

impl<A, N> PartialEq for Chunk<A, N>
where
    A: PartialEq,
    N: ChunkLength<A>,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<A, N> Eq for Chunk<A, N>
where
    A: Eq,
    N: ChunkLength<A>,
{
}

impl<A, N> PartialOrd for Chunk<A, N>
where
    A: PartialOrd,
    N: ChunkLength<A>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A, N> Ord for Chunk<A, N>
where
    A: Ord,
    N: ChunkLength<A>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<N> io::Write for Chunk<u8, N>
where
    N: ChunkLength<u8>,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let old_len = self.len();
        self.extend(buf.iter().cloned().take(N::USIZE - old_len));
        Ok(self.len() - old_len)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<A, N> Borrow<[A]> for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn borrow(&self) -> &[A] {
        self.as_slice()
    }
}

impl<A, N> BorrowMut<[A]> for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn borrow_mut(&mut self) -> &mut [A] {
        self.as_mut_slice()
    }
}

impl<A, N> AsRef<[A]> for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn as_ref(&self) -> &[A] {
        self.as_slice()
    }
}

impl<A, N> AsMut<[A]> for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn as_mut(&mut self) -> &mut [A] {
        self.as_mut_slice()
    }
}

impl<A, N> Deref for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    type Target = [A];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<A, N> DerefMut for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<A, N> FromIterator<A> for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    fn from_iter<I>(it: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        let mut chunk = Self::new();
        for item in it {
            chunk.push_back(item);
        }
        chunk
    }
}

impl<'a, A, N> IntoIterator for &'a Chunk<A, N>
where
    N: ChunkLength<A>,
{
    type Item = &'a A;
    type IntoIter = SliceIter<'a, A>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A, N> IntoIterator for &'a mut Chunk<A, N>
where
    N: ChunkLength<A>,
{
    type Item = &'a mut A;
    type IntoIter = SliceIterMut<'a, A>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<A, N> Extend<A> for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    /// Append the contents of the iterator to the back of the chunk.
    ///
    /// Panics if the chunk exceeds its capacity.
    ///
    /// Time: O(n) for the length of the iterator
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = A>,
    {
        for item in it {
            self.push_back(item);
        }
    }
}

impl<'a, A, N> Extend<&'a A> for Chunk<A, N>
where
    A: 'a + Copy,
    N: ChunkLength<A>,
{
    /// Append the contents of the iterator to the back of the chunk.
    ///
    /// Panics if the chunk exceeds its capacity.
    ///
    /// Time: O(n) for the length of the iterator
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = &'a A>,
    {
        for item in it {
            self.push_back(*item);
        }
    }
}

pub struct Iter<A, N>
where
    N: ChunkLength<A>,
{
    chunk: Chunk<A, N>,
}

impl<A, N> Iterator for Iter<A, N>
where
    N: ChunkLength<A>,
{
    type Item = A;
    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk.is_empty() {
            None
        } else {
            Some(self.chunk.pop_front())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunk.len(), Some(self.chunk.len()))
    }
}

impl<A, N> DoubleEndedIterator for Iter<A, N>
where
    N: ChunkLength<A>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.chunk.is_empty() {
            None
        } else {
            Some(self.chunk.pop_back())
        }
    }
}

impl<A, N> ExactSizeIterator for Iter<A, N> where N: ChunkLength<A> {}

impl<A, N> FusedIterator for Iter<A, N> where N: ChunkLength<A> {}

impl<A, N> IntoIterator for Chunk<A, N>
where
    N: ChunkLength<A>,
{
    type Item = A;
    type IntoIter = Iter<A, N>;

    fn into_iter(self) -> Self::IntoIter {
        Iter { chunk: self }
    }
}

pub struct Drain<'a, A, N>
where
    A: 'a,
    N: ChunkLength<A> + 'a,
{
    chunk: &'a mut Chunk<A, N>,
}

impl<'a, A, N> Iterator for Drain<'a, A, N>
where
    A: 'a,
    N: ChunkLength<A> + 'a,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk.is_empty() {
            None
        } else {
            Some(self.chunk.pop_front())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunk.len(), Some(self.chunk.len()))
    }
}

impl<'a, A, N> ExactSizeIterator for Drain<'a, A, N>
where
    A: 'a,
    N: ChunkLength<A> + 'a,
{
}

impl<'a, A, N> FusedIterator for Drain<'a, A, N>
where
    A: 'a,
    N: ChunkLength<A> + 'a,
{
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn is_full() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..64 {
            assert_eq!(false, chunk.is_full());
            chunk.push_back(i);
        }
        assert_eq!(true, chunk.is_full());
    }

    #[test]
    fn push_back_front() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 12..20 {
            chunk.push_back(i);
        }
        assert_eq!(8, chunk.len());
        for i in (0..12).rev() {
            chunk.push_front(i);
        }
        assert_eq!(20, chunk.len());
        for i in 20..32 {
            chunk.push_back(i);
        }
        assert_eq!(32, chunk.len());
        let right: Vec<i32> = chunk.into_iter().collect();
        let left: Vec<i32> = (0..32).collect();
        assert_eq!(left, right);
    }

    #[test]
    fn push_and_pop() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        for i in 0..64 {
            assert_eq!(i, chunk.pop_front());
        }
        for i in 0..64 {
            chunk.push_front(i);
        }
        for i in 0..64 {
            assert_eq!(i, chunk.pop_back());
        }
    }

    #[test]
    fn drop_left() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..6 {
            chunk.push_back(i);
        }
        chunk.drop_left(3);
        let vec: Vec<i32> = chunk.into_iter().collect();
        assert_eq!(vec![3, 4, 5], vec);
    }

    #[test]
    fn drop_right() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..6 {
            chunk.push_back(i);
        }
        chunk.drop_right(3);
        let vec: Vec<i32> = chunk.into_iter().collect();
        assert_eq!(vec![0, 1, 2], vec);
    }

    #[test]
    fn split_off() {
        let mut left = Chunk::<_, U64>::new();
        for i in 0..6 {
            left.push_back(i);
        }
        let right = left.split_off(3);
        let left_vec: Vec<i32> = left.into_iter().collect();
        let right_vec: Vec<i32> = right.into_iter().collect();
        assert_eq!(vec![0, 1, 2], left_vec);
        assert_eq!(vec![3, 4, 5], right_vec);
    }

    #[test]
    fn append() {
        let mut left = Chunk::<_, U64>::new();
        for i in 0..32 {
            left.push_back(i);
        }
        let mut right = Chunk::<_, U64>::new();
        for i in (32..64).rev() {
            right.push_front(i);
        }
        left.append(&mut right);
        let out_vec: Vec<i32> = left.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn ref_iter() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<&i32> = chunk.iter().collect();
        let should_vec_p: Vec<i32> = (0..64).collect();
        let should_vec: Vec<&i32> = should_vec_p.iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn mut_ref_iter() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<&mut i32> = chunk.iter_mut().collect();
        let mut should_vec_p: Vec<i32> = (0..64).collect();
        let should_vec: Vec<&mut i32> = should_vec_p.iter_mut().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn consuming_iter() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_middle() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..32 {
            chunk.push_back(i);
        }
        for i in 33..64 {
            chunk.push_back(i);
        }
        chunk.insert(32, 32);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_back() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..63 {
            chunk.push_back(i);
        }
        chunk.insert(63, 63);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_front() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 1..64 {
            chunk.push_front(64 - i);
        }
        chunk.insert(0, 0);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn remove_value() {
        let mut chunk = Chunk::<_, U64>::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        chunk.remove(32);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..32).chain(33..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DropTest<'a> {
        counter: &'a AtomicUsize,
    }

    impl<'a> DropTest<'a> {
        fn new(counter: &'a AtomicUsize) -> Self {
            counter.fetch_add(1, Ordering::Relaxed);
            DropTest { counter }
        }
    }

    impl<'a> Drop for DropTest<'a> {
        fn drop(&mut self) {
            self.counter.fetch_sub(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn dropping() {
        let counter = AtomicUsize::new(0);
        {
            let mut chunk: Chunk<DropTest> = Chunk::new();
            for _i in 0..20 {
                chunk.push_back(DropTest::new(&counter))
            }
            for _i in 0..20 {
                chunk.push_front(DropTest::new(&counter))
            }
            assert_eq!(40, counter.load(Ordering::Relaxed));
            for _i in 0..10 {
                chunk.pop_back();
            }
            assert_eq!(30, counter.load(Ordering::Relaxed));
        }
        assert_eq!(0, counter.load(Ordering::Relaxed));
    }
}
