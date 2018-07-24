// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::fmt::{Debug, Error, Formatter};
use std::iter::FusedIterator;
use std::mem::{self, ManuallyDrop};
use std::ops::{Index, IndexMut};
use std::ptr;

pub const CHUNK_SIZE: usize = 64;

pub struct Chunk<A> {
    left: usize,
    right: usize,
    values: ManuallyDrop<[A; CHUNK_SIZE]>,
}

impl<A> Drop for Chunk<A> {
    fn drop(&mut self) {
        if mem::needs_drop::<A>() {
            for i in self.left..self.right {
                unsafe { Chunk::force_drop(i, self) }
            }
        }
    }
}

impl<A: Clone> Clone for Chunk<A> {
    fn clone(&self) -> Self {
        let mut out = Self::new();
        out.left = self.left;
        out.right = self.right;
        for index in self.left..self.right {
            unsafe { Chunk::force_write(index, self.values[index].clone(), &mut out) }
        }
        out
    }
}

impl<A> Chunk<A> {
    pub fn new() -> Self {
        let mut chunk: Self;
        unsafe {
            chunk = mem::uninitialized();
            ptr::write(&mut chunk.left, 0);
            ptr::write(&mut chunk.right, 0);
        }
        chunk
    }

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
        self.left == 0 && self.right == CHUNK_SIZE
    }

    /// Copy the value at an index, discarding ownership of the copied value
    #[inline]
    unsafe fn force_read(index: usize, chunk: &Self) -> A {
        ptr::read(&chunk.values[index])
    }

    /// Write a value at an index without trying to drop what's already there
    #[inline]
    unsafe fn force_write(index: usize, value: A, chunk: &mut Self) {
        ptr::write(&mut chunk.values[index], value)
    }

    /// Drop the value at an index
    #[inline]
    unsafe fn force_drop(index: usize, chunk: &mut Self) {
        ptr::drop_in_place(&mut chunk.values[index])
    }

    /// Copy a range within a chunk
    #[inline]
    unsafe fn force_copy(from: usize, to: usize, count: usize, chunk: &mut Self) {
        if count > 0 {
            ptr::copy(&chunk.values[from], &mut chunk.values[to], count)
        }
    }

    /// Copy a range between chunks
    #[inline]
    unsafe fn force_copy_to(from: usize, to: usize, count: usize, chunk: &Self, other: &mut Self) {
        if count > 0 {
            ptr::copy_nonoverlapping(&chunk.values[from], &mut other.values[to], count)
        }
    }

    pub fn push_front(&mut self, value: A) {
        if self.is_full() {
            panic!("Chunk::push_front: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = CHUNK_SIZE;
            self.right = CHUNK_SIZE;
        } else if self.left == 0 {
            self.left = CHUNK_SIZE - self.right;
            unsafe { Chunk::force_copy(0, self.left, self.right, self) };
            self.right = CHUNK_SIZE;
        }
        self.left -= 1;
        unsafe { Chunk::force_write(self.left, value, self) }
    }

    pub fn push_back(&mut self, value: A) {
        if self.is_full() {
            panic!("Chunk::push_back: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = 0;
            self.right = 0;
        } else if self.right == CHUNK_SIZE {
            unsafe { Chunk::force_copy(self.left, 0, self.len(), self) };
            self.right = CHUNK_SIZE - self.left;
            self.left = 0;
        }
        unsafe { Chunk::force_write(self.right, value, self) }
        self.right += 1;
    }

    pub fn pop_front(&mut self) -> A {
        if self.is_empty() {
            panic!("Chunk::pop_front: can't pop from empty chunk");
        } else {
            let value = unsafe { Chunk::force_read(self.left, self) };
            self.left += 1;
            value
        }
    }

    pub fn pop_back(&mut self) -> A {
        if self.is_empty() {
            panic!("Chunk::pop_back: can't pop from empty chunk");
        } else {
            self.right -= 1;
            unsafe { Chunk::force_read(self.right, self) }
        }
    }

    /// Remove all elements up to but not including `index`.
    pub fn drop_left(&mut self, index: usize) {
        if index > 0 {
            if index >= self.len() {
                panic!("Chunk::drop_left: index out of bounds");
            }
            let start = self.left;
            for i in start..(start + index) {
                unsafe { Chunk::force_drop(i, self) }
            }
            self.left += index;
        }
    }

    /// Remove all elements from `index` onward.
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
    pub fn split(&mut self, index: usize) -> Chunk<A> {
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

    pub fn extend(&mut self, other: &mut Self) {
        let self_len = self.len();
        let other_len = other.len();
        if self_len + other_len > CHUNK_SIZE {
            panic!("Chunk::extend: chunk size overflow");
        }
        if self.left > 0 && self.right + other_len > CHUNK_SIZE {
            unsafe { Chunk::force_copy(self.left, 0, self_len, self) };
            self.right -= self.left;
            self.left = 0;
        }
        unsafe { Chunk::force_copy_to(other.left, self.right, other_len, other, self) };
        self.right += other_len;
        other.left = 0;
        other.right = 0;
    }

    pub fn insert(&mut self, index: usize, value: A) {
        if self.is_full() {
            panic!("Chunk::insert: chunk is full");
        }
        let real_index = index + self.left;
        if real_index < self.left || real_index > self.right {
            panic!("Chunk::insert: index out of bounds");
        }
        let left_size = index;
        let right_size = self.right - real_index;
        if self.right == CHUNK_SIZE || (self.left > 0 && left_size < right_size) {
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

    pub fn remove(&mut self, index: usize) {
        let real_index = index + self.left;
        if real_index < self.left || real_index >= self.right {
            panic!("Chunk::remove: index out of bounds");
        }
        unsafe { Chunk::force_drop(real_index, self) };
        let left_size = index;
        let right_size = self.right - real_index - 1;
        if left_size < right_size {
            unsafe { Chunk::force_copy(self.left, self.left + 1, left_size, self) };
            self.left += 1;
        } else {
            unsafe { Chunk::force_copy(real_index + 1, real_index, right_size, self) };
            self.right -= 1;
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&A> {
        let real_index = self.left + index;
        if real_index >= self.right {
            None
        } else {
            Some(&self.values[real_index])
        }
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut A> {
        let real_index = self.left + index;
        if real_index >= self.right {
            None
        } else {
            Some(&mut self.values[real_index])
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

    pub fn iter(&self) -> Iter<'_, A> {
        Iter {
            chunk: self,
            left_index: 0,
            right_index: self.len(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, A> {
        IterMut {
            left_index: 0,
            right_index: self.len(),
            chunk: self,
        }
    }
}

impl<A> Default for Chunk<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Index<usize> for Chunk<A> {
    type Output = A;
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            None => panic!(
                "Chunk::index: index out of bounds: {} >= {}",
                index,
                self.len()
            ),
            Some(value) => value,
        }
    }
}

impl<A> IndexMut<usize> for Chunk<A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.get_mut(index) {
            None => panic!("Chunk::index_mut: index out of bounds"),
            Some(value) => value,
        }
    }
}

impl<A: Debug> Debug for Chunk<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str("Chunk")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

pub struct Iter<'a, A: 'a> {
    chunk: &'a Chunk<A>,
    left_index: usize,
    right_index: usize,
}

impl<'a, A> Iterator for Iter<'a, A> {
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

impl<'a, A> DoubleEndedIterator for Iter<'a, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.left_index >= self.right_index {
            None
        } else {
            self.right_index -= 1;
            self.chunk.get(self.right_index)
        }
    }
}

impl<'a, A> ExactSizeIterator for Iter<'a, A> {}

impl<'a, A> FusedIterator for Iter<'a, A> {}

impl<'a, A> IntoIterator for &'a Chunk<A> {
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct IterMut<'a, A: 'a> {
    chunk: &'a mut Chunk<A>,
    left_index: usize,
    right_index: usize,
}

impl<'a, A> Iterator for IterMut<'a, A> {
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

impl<'a, A> DoubleEndedIterator for IterMut<'a, A> {
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

impl<'a, A> ExactSizeIterator for IterMut<'a, A> {}

impl<'a, A> FusedIterator for IterMut<'a, A> {}

pub struct ConsumingIter<A> {
    chunk: Chunk<A>,
}

impl<A> Iterator for ConsumingIter<A> {
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

impl<A> DoubleEndedIterator for ConsumingIter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.chunk.is_empty() {
            None
        } else {
            Some(self.chunk.pop_back())
        }
    }
}

impl<A> ExactSizeIterator for ConsumingIter<A> {}

impl<A> FusedIterator for ConsumingIter<A> {}

impl<A> IntoIterator for Chunk<A> {
    type Item = A;
    type IntoIter = ConsumingIter<A>;
    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter { chunk: self }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn push_back_front() {
        let mut chunk = Chunk::new();
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
        let mut chunk = Chunk::new();
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
        let mut chunk = Chunk::new();
        for i in 0..6 {
            chunk.push_back(i);
        }
        chunk.drop_left(3);
        let vec: Vec<i32> = chunk.into_iter().collect();
        assert_eq!(vec![3, 4, 5], vec);
    }

    #[test]
    fn drop_right() {
        let mut chunk = Chunk::new();
        for i in 0..6 {
            chunk.push_back(i);
        }
        chunk.drop_right(3);
        let vec: Vec<i32> = chunk.into_iter().collect();
        assert_eq!(vec![0, 1, 2], vec);
    }

    #[test]
    fn split() {
        let mut left = Chunk::new();
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
        let mut left = Chunk::new();
        for i in 0..32 {
            left.push_back(i);
        }
        let mut right = Chunk::new();
        for i in (32..64).into_iter().rev() {
            right.push_front(i);
        }
        left.extend(&mut right);
        let out_vec: Vec<i32> = left.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).into_iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn ref_iter() {
        let mut chunk = Chunk::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<&i32> = chunk.iter().collect();
        let should_vec_p: Vec<i32> = (0..64).into_iter().collect();
        let should_vec: Vec<&i32> = should_vec_p.iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn mut_ref_iter() {
        let mut chunk = Chunk::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<&mut i32> = chunk.iter_mut().collect();
        let mut should_vec_p: Vec<i32> = (0..64).into_iter().collect();
        let should_vec: Vec<&mut i32> = should_vec_p.iter_mut().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn consuming_iter() {
        let mut chunk = Chunk::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).into_iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_middle() {
        let mut chunk = Chunk::new();
        for i in 0..32 {
            chunk.push_back(i);
        }
        for i in 33..64 {
            chunk.push_back(i);
        }
        chunk.insert(32, 32);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).into_iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_back() {
        let mut chunk = Chunk::new();
        for i in 0..63 {
            chunk.push_back(i);
        }
        chunk.insert(63, 63);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).into_iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn insert_front() {
        let mut chunk = Chunk::new();
        for i in 1..64 {
            chunk.push_front(64 - i);
        }
        chunk.insert(0, 0);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = (0..64).into_iter().collect();
        assert_eq!(should_vec, out_vec);
    }

    #[test]
    fn remove_value() {
        let mut chunk = Chunk::new();
        for i in 0..64 {
            chunk.push_back(i);
        }
        chunk.remove(32);
        let out_vec: Vec<i32> = chunk.into_iter().collect();
        let should_vec: Vec<i32> = ((0..32).into_iter()).chain((33..64).into_iter()).collect();
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
