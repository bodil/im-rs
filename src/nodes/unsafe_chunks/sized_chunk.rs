// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Error, Formatter};
use std::iter::FromIterator;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::{self, replace, ManuallyDrop};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr;
use std::slice::{from_raw_parts, from_raw_parts_mut};

use typenum::*;

pub trait ChunkLength<A>: Unsigned {
    type SizedType;
}

impl<A> ChunkLength<A> for UTerm {
    type SizedType = ();
}

#[allow(dead_code)]
pub struct SizeEven<A, B> {
    parent1: B,
    parent2: B,
    _marker: PhantomData<A>,
}

#[allow(dead_code)]
pub struct SizeOdd<A, B> {
    parent1: B,
    parent2: B,
    data: A,
}

impl<A, N> ChunkLength<A> for UInt<N, B0>
where
    N: ChunkLength<A>,
{
    type SizedType = SizeEven<A, N::SizedType>;
}

impl<A, N> ChunkLength<A> for UInt<N, B1>
where
    N: ChunkLength<A>,
{
    type SizedType = SizeOdd<A, N::SizedType>;
}

pub struct SizedChunk<A, N = U64>
where
    N: ChunkLength<A>,
{
    left: usize,
    right: usize,
    data: ManuallyDrop<N::SizedType>,
}

impl<A, N> Drop for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn drop(&mut self) {
        if mem::needs_drop::<A>() {
            for i in self.left..self.right {
                unsafe { SizedChunk::force_drop(i, self) }
            }
        }
    }
}

impl<A, N> Clone for SizedChunk<A, N>
where
    A: Clone,
    N: ChunkLength<A>,
{
    fn clone(&self) -> Self {
        let mut out = Self::new();
        out.left = self.left;
        out.right = self.right;
        for index in self.left..self.right {
            unsafe { SizedChunk::force_write(index, self.values()[index].clone(), &mut out) }
        }
        out
    }
}

impl<A, N> SizedChunk<A, N>
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
            SizedChunk::force_write(0, value, &mut chunk);
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
            SizedChunk::force_write(0, left, &mut chunk);
            SizedChunk::force_write(1, right, &mut chunk);
        }
        chunk
    }

    /// Construct a new chunk and move every item from `other` into the new
    /// chunk.
    pub fn drain_from(other: &mut Self) -> Self {
        let other_len = other.len();
        Self::from_front(other, other_len)
    }

    /// Construct a new chunk and populate it by taking `count` item from the
    /// iterator `iter`.
    ///
    /// Will panic if the iterator contains less than `count` items.
    pub fn collect_from<I>(iter: &mut I, mut count: usize) -> Self
    where
        I: Iterator<Item = A>,
    {
        let mut chunk = Self::new();
        while count > 0 {
            count -= 1;
            chunk.push_back(
                iter.next()
                    .expect("SizedChunk::collect_from: underfull iterator"),
            );
        }
        chunk
    }

    /// Construct a new chunk and populate it by taking `count` items from the
    /// front of `other`.
    pub fn from_front(other: &mut Self, count: usize) -> Self {
        let other_len = other.len();
        debug_assert!(count <= other_len);
        let mut chunk = Self::new();
        unsafe { SizedChunk::force_copy_to(other.left, 0, count, other, &mut chunk) };
        chunk.right = count;
        other.left += count;
        chunk
    }

    /// Construct a new chunk and populate it by taking `count` items from the
    /// back of `other`.
    pub fn from_back(other: &mut Self, count: usize) -> Self {
        let other_len = other.len();
        debug_assert!(count <= other_len);
        let mut chunk = Self::new();
        unsafe { SizedChunk::force_copy_to(other.right - count, 0, count, other, &mut chunk) };
        chunk.right = count;
        other.right -= count;
        chunk
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.right - self.left
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.left == self.right
    }

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

    pub fn push_front(&mut self, value: A) {
        if self.is_full() {
            panic!("SizedChunk::push_front: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = N::USIZE;
            self.right = N::USIZE;
        } else if self.left == 0 {
            self.left = N::USIZE - self.right;
            unsafe { SizedChunk::force_copy(0, self.left, self.right, self) };
            self.right = N::USIZE;
        }
        self.left -= 1;
        unsafe { SizedChunk::force_write(self.left, value, self) }
    }

    pub fn push_back(&mut self, value: A) {
        if self.is_full() {
            panic!("SizedChunk::push_back: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = 0;
            self.right = 0;
        } else if self.right == N::USIZE {
            unsafe { SizedChunk::force_copy(self.left, 0, self.len(), self) };
            self.right = N::USIZE - self.left;
            self.left = 0;
        }
        unsafe { SizedChunk::force_write(self.right, value, self) }
        self.right += 1;
    }

    pub fn pop_front(&mut self) -> A {
        if self.is_empty() {
            panic!("SizedChunk::pop_front: can't pop from empty chunk");
        } else {
            let value = unsafe { SizedChunk::force_read(self.left, self) };
            self.left += 1;
            value
        }
    }

    pub fn pop_back(&mut self) -> A {
        if self.is_empty() {
            panic!("SizedChunk::pop_back: can't pop from empty chunk");
        } else {
            self.right -= 1;
            unsafe { SizedChunk::force_read(self.right, self) }
        }
    }

    /// Remove all elements up to but not including `index`.
    pub fn drop_left(&mut self, index: usize) {
        if index > 0 {
            if index > self.len() {
                panic!("SizedChunk::drop_left: index out of bounds");
            }
            let start = self.left;
            #[allow(unknown_lints)]
            #[allow(redundant_field_names)] // FIXME clippy is currently broken
            for i in start..(start + index) {
                unsafe { SizedChunk::force_drop(i, self) }
            }
            self.left += index;
        }
    }

    /// Remove all elements from `index` onward.
    pub fn drop_right(&mut self, index: usize) {
        if index > self.len() {
            panic!("SizedChunk::drop_right: index out of bounds");
        }
        if index == self.len() {
            return;
        }
        let start = self.left + index;
        #[allow(unknown_lints)]
        #[allow(redundant_field_names)] // FIXME clippy is currently broken
        for i in start..self.right {
            unsafe { SizedChunk::force_drop(i, self) }
        }
        self.right = start;
    }

    /// Split a chunk into two, the original chunk containing
    /// everything up to `index` and the returned chunk containing
    /// everything from `index` onwards.
    pub fn split(&mut self, index: usize) -> Self {
        if index > self.len() {
            panic!("SizedChunk::split: index out of bounds");
        }
        if index == self.len() {
            return Self::new();
        }
        let mut right_chunk = Self::new();
        let start = self.left + index;
        let len = self.right - start;
        unsafe { SizedChunk::force_copy_to(start, 0, len, self, &mut right_chunk) };
        right_chunk.right = len;
        self.right = start;
        right_chunk
    }

    pub fn extend(&mut self, other: &mut Self) {
        let self_len = self.len();
        let other_len = other.len();
        if self_len + other_len > N::USIZE {
            panic!("SizedChunk::extend: chunk size overflow");
        }
        if self.left > 0 && self.right + other_len > N::USIZE {
            unsafe { SizedChunk::force_copy(self.left, 0, self_len, self) };
            self.right -= self.left;
            self.left = 0;
        }
        unsafe { SizedChunk::force_copy_to(other.left, self.right, other_len, other, self) };
        self.right += other_len;
        other.left = 0;
        other.right = 0;
    }

    pub fn extend_from_front(&mut self, other: &mut Self, count: usize) {
        let self_len = self.len();
        let other_len = other.len();
        debug_assert!(self_len + count <= other_len);
        if self.left > 0 && self.right + count > N::USIZE {
            unsafe { SizedChunk::force_copy(self.left, 0, self_len, self) };
            self.right -= self.left;
            self.left = 0;
        }
        unsafe { SizedChunk::force_copy_to(other.left, self.right, count, other, self) };
        self.right += count;
        other.left += count;
    }

    pub fn extend_from_back(&mut self, other: &mut Self, count: usize) {
        let self_len = self.len();
        let other_len = other.len();
        debug_assert!(self_len + count <= other_len);
        if self.left > 0 && self.right + count > N::USIZE {
            unsafe { SizedChunk::force_copy(self.left, 0, self_len, self) };
            self.right -= self.left;
            self.left = 0;
        }
        unsafe { SizedChunk::force_copy_to(other.right - count, self.right, count, other, self) };
        self.right += count;
        other.right -= count;
    }

    pub fn set(&mut self, index: usize, value: A) -> A {
        replace(&mut self[index], value)
    }

    pub fn insert(&mut self, index: usize, value: A) {
        if self.is_full() {
            panic!("SizedChunk::insert: chunk is full");
        }
        if index > self.len() {
            panic!("SizedChunk::insert: index out of bounds");
        }
        let real_index = index + self.left;
        let left_size = index;
        let right_size = self.right - real_index;
        if self.right == N::USIZE || (self.left > 0 && left_size < right_size) {
            unsafe {
                SizedChunk::force_copy(self.left, self.left - 1, left_size, self);
                SizedChunk::force_write(real_index - 1, value, self);
            }
            self.left -= 1;
        } else {
            unsafe {
                SizedChunk::force_copy(real_index, real_index + 1, right_size, self);
                SizedChunk::force_write(real_index, value, self);
            }
            self.right += 1;
        }
    }

    pub fn remove(&mut self, index: usize) -> A {
        if index >= self.len() {
            panic!("SizedChunk::remove: index out of bounds");
        }
        let real_index = index + self.left;
        let value = unsafe { SizedChunk::force_read(real_index, self) };
        let left_size = index;
        let right_size = self.right - real_index - 1;
        if left_size < right_size {
            unsafe { SizedChunk::force_copy(self.left, self.left + 1, left_size, self) };
            self.left += 1;
        } else {
            unsafe { SizedChunk::force_copy(real_index + 1, real_index, right_size, self) };
            self.right -= 1;
        }
        value
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&A> {
        if index < self.len() {
            Some(&self.values()[self.left + index])
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut A> {
        if index < self.len() {
            let left = self.left;
            Some(&mut self.values_mut()[left + index])
        } else {
            None
        }
    }

    #[inline]
    pub fn first(&self) -> Option<&A> {
        self.get(0)
    }

    #[inline]
    pub fn last(&self) -> Option<&A> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    pub fn iter(&self) -> Iter<'_, A, N> {
        Iter {
            chunk: self,
            left_index: 0,
            right_index: self.len(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, A, N> {
        IterMut {
            left_index: 0,
            right_index: self.len(),
            chunk: self,
        }
    }

    pub fn drain(&mut self) -> Drain<'_, A, N> {
        Drain { chunk: self }
    }

    pub fn as_slice(&self) -> &[A] {
        &self.values()[self.left..self.right]
    }

    pub fn as_mut_slice(&mut self) -> &mut [A] {
        let subslice = self.left..self.right;
        &mut self.values_mut()[subslice]
    }
}

impl<A, N> Default for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, N> Index<usize> for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    type Output = A;
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            None => panic!(
                "SizedChunk::index: index out of bounds: {} >= {}",
                index,
                self.len()
            ),
            Some(value) => value,
        }
    }
}

impl<A, N> IndexMut<usize> for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.get_mut(index) {
            None => panic!("SizedChunk::index_mut: index out of bounds"),
            Some(value) => value,
        }
    }
}

impl<A, N> Debug for SizedChunk<A, N>
where
    A: Debug,
    N: ChunkLength<A>,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str("Chunk")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<A, N> Borrow<[A]> for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn borrow(&self) -> &[A] {
        self.as_slice()
    }
}

impl<A, N> BorrowMut<[A]> for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn borrow_mut(&mut self) -> &mut [A] {
        self.as_mut_slice()
    }
}

impl<A, N> AsRef<[A]> for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn as_ref(&self) -> &[A] {
        self.as_slice()
    }
}

impl<A, N> AsMut<[A]> for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn as_mut(&mut self) -> &mut [A] {
        self.as_mut_slice()
    }
}

impl<A, N> Deref for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    type Target = [A];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<A, N> DerefMut for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<A, N> FromIterator<A> for SizedChunk<A, N>
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

pub struct Iter<'a, A: 'a, N>
where
    N: ChunkLength<A> + 'a,
{
    chunk: &'a SizedChunk<A, N>,
    left_index: usize,
    right_index: usize,
}

impl<'a, A, N> Iterator for Iter<'a, A, N>
where
    N: ChunkLength<A> + 'a,
{
    type Item = &'a A;
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_index >= self.right_index {
            None
        } else {
            let value = self.chunk.get(self.left_index);
            self.left_index += 1;
            value
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.right_index - self.left_index;
        (remaining, Some(remaining))
    }
}

impl<'a, A, N> DoubleEndedIterator for Iter<'a, A, N>
where
    N: ChunkLength<A> + 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.left_index >= self.right_index {
            None
        } else {
            self.right_index -= 1;
            self.chunk.get(self.right_index)
        }
    }
}

impl<'a, A, N> ExactSizeIterator for Iter<'a, A, N> where N: ChunkLength<A> + 'a {}

impl<'a, A, N> FusedIterator for Iter<'a, A, N> where N: ChunkLength<A> + 'a {}

impl<'a, A, N> IntoIterator for &'a SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct IterMut<'a, A: 'a, N>
where
    N: ChunkLength<A> + 'a,
{
    chunk: &'a mut SizedChunk<A, N>,
    left_index: usize,
    right_index: usize,
}

impl<'a, A, N> Iterator for IterMut<'a, A, N>
where
    N: ChunkLength<A> + 'a,
{
    type Item = &'a mut A;
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_index >= self.right_index {
            None
        } else {
            let value = self.chunk.get_mut(self.left_index);
            self.left_index += 1;
            // this is annoying: the trait won't allow `fn next(&'a mut self)`,
            // so we have to turn to the Dark Side to get the right lifetime
            unsafe { value.map(|p| &mut *(p as *mut _)) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.right_index - self.left_index;
        (remaining, Some(remaining))
    }
}

impl<'a, A, N> DoubleEndedIterator for IterMut<'a, A, N>
where
    N: ChunkLength<A> + 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.left_index >= self.right_index {
            None
        } else {
            self.right_index -= 1;
            let value = self.chunk.get_mut(self.right_index);
            unsafe { value.map(|p| &mut *(p as *mut _)) }
        }
    }
}

impl<'a, A, N> ExactSizeIterator for IterMut<'a, A, N> where N: ChunkLength<A> + 'a {}

impl<'a, A, N> FusedIterator for IterMut<'a, A, N> where N: ChunkLength<A> + 'a {}

pub struct ConsumingIter<A, N>
where
    N: ChunkLength<A>,
{
    chunk: SizedChunk<A, N>,
}

impl<A, N> Iterator for ConsumingIter<A, N>
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

impl<A, N> DoubleEndedIterator for ConsumingIter<A, N>
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

impl<A, N> ExactSizeIterator for ConsumingIter<A, N> where N: ChunkLength<A> {}

impl<A, N> FusedIterator for ConsumingIter<A, N> where N: ChunkLength<A> {}

impl<A, N> IntoIterator for SizedChunk<A, N>
where
    N: ChunkLength<A>,
{
    type Item = A;
    type IntoIter = ConsumingIter<A, N>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter { chunk: self }
    }
}

pub struct Drain<'a, A, N>
where
    A: 'a,
    N: ChunkLength<A> + 'a,
{
    chunk: &'a mut SizedChunk<A, N>,
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
    fn push_back_front() {
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
        for i in 0..6 {
            chunk.push_back(i);
        }
        chunk.drop_left(3);
        let vec: Vec<i32> = chunk.into_iter().collect();
        assert_eq!(vec![3, 4, 5], vec);
    }

    #[test]
    fn drop_right() {
        let mut chunk = SizedChunk::<_, U64>::new();
        for i in 0..6 {
            chunk.push_back(i);
        }
        chunk.drop_right(3);
        let vec: Vec<i32> = chunk.into_iter().collect();
        assert_eq!(vec![0, 1, 2], vec);
    }

    #[test]
    fn split() {
        let mut left = SizedChunk::<_, U64>::new();
        for i in 0..6 {
            left.push_back(i);
        }
        let right = left.split(3);
        let left_vec: Vec<i32> = left.into_iter().collect();
        let right_vec: Vec<i32> = right.into_iter().collect();
        assert_eq!(vec![0, 1, 2], left_vec);
        assert_eq!(vec![3, 4, 5], right_vec);
    }

    #[test]
    fn extend() {
        let mut left = SizedChunk::<_, U64>::new();
        for i in 0..32 {
            left.push_back(i);
        }
        let mut right = SizedChunk::<_, U64>::new();
        for i in (32..64).rev() {
            right.push_front(i);
        }
        left.extend(&mut right);
        let out_vec: Vec<i32> = left.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn ref_iter() {
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_middle() {
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
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
        let mut chunk = SizedChunk::<_, U64>::new();
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
            let mut chunk: SizedChunk<DropTest> = SizedChunk::new();
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
