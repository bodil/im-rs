// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::mem::{self, ManuallyDrop};
use std::ops::Index;
use std::ops::IndexMut;
use std::ptr;

use nodes::bitmap::{Bitmap, Iter as BitmapIter, HASH_SIZE};

const CHUNK_SIZE: usize = HASH_SIZE;

pub struct SparseChunk<A> {
    map: Bitmap,
    values: ManuallyDrop<[A; CHUNK_SIZE]>,
}

impl<A> Drop for SparseChunk<A> {
    fn drop(&mut self) {
        if mem::needs_drop::<A>() {
            for index in self.map {
                unsafe { SparseChunk::force_drop(index, self) }
            }
        }
    }
}

impl<A: Clone> Clone for SparseChunk<A> {
    fn clone(&self) -> Self {
        let mut out: Self;
        unsafe {
            out = mem::uninitialized();
            ptr::write(&mut out.map, self.map);
            for index in self.map {
                SparseChunk::force_write(index, self.values[index].clone(), &mut out);
            }
        }
        out
    }
}

impl<A> SparseChunk<A> {
    /// Copy the value at an index, discarding ownership of the copied value
    #[inline]
    unsafe fn force_read(index: usize, chunk: &Self) -> A {
        ptr::read(&chunk.values[index as usize])
    }

    /// Write a value at an index without trying to drop what's already there
    #[inline]
    unsafe fn force_write(index: usize, value: A, chunk: &mut Self) {
        ptr::write(&mut chunk.values[index as usize], value)
    }

    /// Drop the value at an index
    #[inline]
    unsafe fn force_drop(index: usize, chunk: &mut Self) {
        ptr::drop_in_place(&mut chunk.values[index])
    }

    pub fn new() -> Self {
        let mut chunk: Self;
        unsafe {
            chunk = mem::uninitialized();
            ptr::write(&mut chunk.map, Bitmap::new());
        }
        chunk
    }

    pub fn unit(index: usize, value: A) -> Self {
        let mut chunk = Self::new();
        chunk.insert(index, value);
        chunk
    }

    pub fn pair(index1: usize, value1: A, index2: usize, value2: A) -> Self {
        let mut chunk = Self::new();
        chunk.insert(index1, value1);
        chunk.insert(index2, value2);
        chunk
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn insert(&mut self, index: usize, value: A) -> Option<A> {
        if index >= CHUNK_SIZE {
            panic!("SparseChunk::insert: index out of bounds");
        }
        let prev = if self.map.set(index, true) {
            Some(unsafe { SparseChunk::force_read(index, self) })
        } else {
            None
        };
        unsafe { SparseChunk::force_write(index, value, self) };
        prev
    }

    pub fn remove(&mut self, index: usize) -> Option<A> {
        if index >= CHUNK_SIZE {
            panic!("SparseChunk::remove: index out of bounds");
        }
        if self.map.set(index, false) {
            Some(unsafe { SparseChunk::force_read(index, self) })
        } else {
            None
        }
    }

    pub fn pop(&mut self) -> Option<A> {
        self.map
            .into_iter()
            .next()
            .and_then(|index| self.remove(index))
    }

    pub fn get(&self, index: usize) -> Option<&A> {
        if self.map.get(index) {
            Some(&self.values[index])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut A> {
        if self.map.get(index) {
            Some(&mut self.values[index])
        } else {
            None
        }
    }

    pub fn indices(&self) -> BitmapIter {
        self.map.into_iter()
    }

    pub fn first_index(&self) -> Option<usize> {
        self.indices().next()
    }

    pub fn iter(&self) -> Iter<'_, A> {
        Iter {
            indices: self.indices(),
            chunk: self,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, A> {
        IterMut {
            indices: self.indices(),
            chunk: self,
        }
    }

    pub fn drain(self) -> Drain<A> {
        Drain {
            indices: self.indices(),
            chunk: self,
        }
    }
}

impl<A> Index<usize> for SparseChunk<A> {
    type Output = A;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<A> IndexMut<usize> for SparseChunk<A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<A> IntoIterator for SparseChunk<A> {
    type Item = A;
    type IntoIter = Drain<A>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.drain()
    }
}

pub struct Iter<'a, A: 'a> {
    indices: BitmapIter,
    chunk: &'a SparseChunk<A>,
}

impl<'a, A> Iterator for Iter<'a, A> {
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|index| &self.chunk.values[index])
    }
}

pub struct IterMut<'a, A: 'a> {
    indices: BitmapIter,
    chunk: &'a mut SparseChunk<A>,
}

impl<'a, A> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.indices.next() {
            unsafe {
                let p: *mut A = &mut (&mut *self.chunk.values)[index];
                Some(&mut *p)
            }
        } else {
            None
        }
    }
}

pub struct Drain<A> {
    indices: BitmapIter,
    chunk: SparseChunk<A>,
}

impl<'a, A> Iterator for Drain<A> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .next()
            .map(|index| self.chunk.remove(index).unwrap())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn insert_remove_iterate() {
        let mut chunk = SparseChunk::new();
        assert_eq!(None, chunk.insert(5, 5));
        assert_eq!(None, chunk.insert(1, 1));
        assert_eq!(None, chunk.insert(24, 42));
        assert_eq!(None, chunk.insert(22, 22));
        assert_eq!(Some(42), chunk.insert(24, 24));
        assert_eq!(None, chunk.insert(31, 31));
        assert_eq!(Some(24), chunk.remove(24));
        assert_eq!(4, chunk.len());
        let indices: Vec<_> = chunk.indices().collect();
        assert_eq!(vec![1, 5, 22, 31], indices);
        let values: Vec<_> = chunk.into_iter().collect();
        assert_eq!(vec![1, 5, 22, 31], values);
    }

    #[test]
    fn clone_chunk() {
        let mut chunk = SparseChunk::new();
        assert_eq!(None, chunk.insert(5, 5));
        assert_eq!(None, chunk.insert(1, 1));
        assert_eq!(None, chunk.insert(24, 42));
        assert_eq!(None, chunk.insert(22, 22));
        let cloned = chunk.clone();
        let right_indices: Vec<_> = chunk.indices().collect();
        let left_indices: Vec<_> = cloned.indices().collect();
        let right: Vec<_> = chunk.into_iter().collect();
        let left: Vec<_> = cloned.into_iter().collect();
        assert_eq!(left, right);
        assert_eq!(left_indices, right_indices);
        assert_eq!(vec![1, 5, 22, 24], left_indices);
        assert_eq!(vec![1, 5, 22, 24], right_indices);
    }
}
