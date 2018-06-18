// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::fmt::{Debug, Error, Formatter};
use std::iter::FusedIterator;
use std::mem::replace;
use std::ops::{Index, IndexMut};

type ChunkIndex = u8;
pub const CHUNK_SIZE: usize = 64;
const CHUNK_ISIZE: ChunkIndex = CHUNK_SIZE as ChunkIndex;

#[derive(Clone, Copy)]
pub struct Chunk<A> {
    left: ChunkIndex,
    right: ChunkIndex,
    values: [Option<A>; CHUNK_SIZE],
}

impl<A> Chunk<A> {
    pub fn new() -> Self {
        Chunk {
            left: 0,
            right: 0,
            values: [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None,
            ],
        }
    }

    pub fn len(&self) -> usize {
        (self.right - self.left) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.left == self.right
    }

    pub fn is_full(&self) -> bool {
        self.left == 0 && self.right == CHUNK_ISIZE
    }

    pub fn push_front(&mut self, value: A) {
        if self.is_full() {
            panic!("Chunk::push_front: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = CHUNK_ISIZE - 1;
            self.right = CHUNK_ISIZE;
            self.values[CHUNK_SIZE - 1] = Some(value);
            return;
        }
        if self.left == 0 {
            let shift = (CHUNK_ISIZE - self.right) as usize;
            for i in (0..self.right as usize).rev() {
                self.values.swap(i, i + shift);
            }
            self.right = CHUNK_ISIZE;
            self.left += shift as ChunkIndex;
        }
        self.left -= 1;
        self.values[self.left as usize] = Some(value);
    }

    pub fn push_back(&mut self, value: A) {
        if self.is_full() {
            panic!("Chunk::push_back: can't push to full chunk");
        }
        if self.is_empty() {
            self.left = 0;
            self.right = 1;
            self.values[0] = Some(value);
            return;
        }
        if self.right == CHUNK_ISIZE {
            for i in self.left as usize..CHUNK_SIZE {
                self.values.swap(i, i - self.left as usize);
            }
            self.right = CHUNK_ISIZE - self.left;
            self.left = 0;
        }
        self.values[self.right as usize] = Some(value);
        self.right += 1;
    }

    pub fn pop_front(&mut self) -> A {
        if self.is_empty() {
            panic!("Chunk::pop_front: can't pop from empty chunk");
        } else {
            let value = replace(&mut self.values[self.left as usize], None);
            self.left += 1;
            value.expect("Chunk::pop_front: expected value but found None")
        }
    }

    pub fn pop_back(&mut self) -> A {
        if self.is_empty() {
            panic!("Chunk::pop_back: can't pop from empty chunk");
        } else {
            self.right -= 1;
            let value = replace(&mut self.values[self.right as usize], None);
            value.expect("Chunk::pop_back: expected value but found None")
        }
    }

    /// Remove all elements up to but not including `index`.
    pub fn drop_left(&mut self, index: usize) {
        if index > 0 {
            if index >= self.len() {
                panic!("Chunk::drop_left: index out of bounds");
            }
            let start = self.left as usize;
            for i in start..(start + index - 1) {
                self.values[i] = None;
            }
            self.left += index as ChunkIndex;
        }
    }

    /// Remove all elements from `index` onward.
    pub fn drop_right(&mut self, index: usize) {
        if index > self.len() {
            panic!("Chunk::drop_right: index out of bounds");
        }
        let start = (self.left as usize) + index;
        for i in start..(self.right as usize) {
            self.values[i] = None;
        }
        self.right = start as ChunkIndex;
    }

    /// Split a chunk into two, the original chunk containing
    /// everything up to `index` and the returned chunk containing
    /// everything from `index` onwards.
    pub fn split(&mut self, index: usize) -> Chunk<A> {
        if index > self.len() {
            panic!("Chunk::split: index out of bounds");
        }
        let mut right_chunk = Self::new();
        let start = (self.left as usize) + index;
        for i in start..(self.right as usize) {
            right_chunk.push_back(
                replace(&mut self.values[i], None)
                    .expect("Chunk::split: expected value but found None"),
            );
        }
        self.right = start as ChunkIndex;
        right_chunk
    }

    pub fn extend(&mut self, other: &mut Self) {
        while self.len() < CHUNK_SIZE && !other.is_empty() {
            self.push_back(other.pop_front());
        }
    }

    pub fn get(&self, index: usize) -> Option<&A> {
        let real_index = self.left as usize + index;
        if real_index >= CHUNK_SIZE {
            None
        } else {
            self.values[real_index].as_ref()
        }
    }

    pub fn first(&self) -> Option<&A> {
        self.values[self.left as usize].as_ref()
    }

    pub fn last(&self) -> Option<&A> {
        if self.is_empty() {
            None
        } else {
            self.values[self.right as usize - 1].as_ref()
        }
    }

    pub fn iter(&self) -> Iter<'_, A> {
        Iter {
            chunk: self,
            left_index: 0,
            right_index: self.len(),
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut A> {
        let len = self.len();
        self.values
            .iter_mut()
            .skip(self.left as usize)
            .take(len)
            .map(|v| match v {
                None => panic!("Chunk::iter_mut: encountered None while iterating"),
                Some(ref mut value) => value,
            })
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
        match self.values[self.left as usize + index] {
            None => panic!(
                "Chunk::index: index out of bounds: {} >= {}",
                index,
                self.len()
            ),
            Some(ref value) => value,
        }
    }
}

impl<A> IndexMut<usize> for Chunk<A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.values[self.left as usize + index] {
            None => panic!(
                "Chunk::index: index out of bounds: {} >= {}",
                index,
                self.len()
            ),
            Some(ref mut value) => value,
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
            let value = Some(&self.chunk[self.left_index]);
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
            Some(&self.chunk[self.right_index])
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
}
