// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use nodes::types::Bits;

pub struct Bitmap<Size: Bits> {
    data: Size::Store,
}

impl<Size: Bits> Clone for Bitmap<Size> {
    fn clone(&self) -> Self {
        Bitmap { data: self.data }
    }
}

impl<Size: Bits> Copy for Bitmap<Size> {}

impl<Size: Bits> Default for Bitmap<Size> {
    fn default() -> Self {
        Bitmap {
            data: Size::Store::default(),
        }
    }
}

impl<Size: Bits> PartialEq for Bitmap<Size> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

use std::fmt;

impl<Size: Bits> fmt::Debug for Bitmap<Size> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.data.fmt(f)
    }
}

impl<Size: Bits> Bitmap<Size> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn get(self, index: usize) -> bool {
        Size::get(&self.data, index)
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) -> bool {
        Size::set(&mut self.data, index, value)
    }

    #[inline]
    pub fn len(self) -> usize {
        Size::len(&self.data)
    }

    #[inline]
    pub fn first_index(self) -> Option<usize> {
        Size::first_index(&self.data)
    }
}

impl<Size: Bits> IntoIterator for Bitmap<Size> {
    type Item = usize;
    type IntoIter = Iter<Size>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            index: 0,
            data: self.data,
        }
    }
}

pub struct Iter<Size: Bits> {
    index: usize,
    data: Size::Store,
}

impl<Size: Bits> Iterator for Iter<Size> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= Size::USIZE {
            return None;
        }
        if Size::get(&self.data, self.index) {
            self.index += 1;
            Some(self.index - 1)
        } else {
            self.index += 1;
            self.next()
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::btree_set;
    use typenum::U64;

    proptest! {
        #[test]
        fn get_set_and_iter(bits in btree_set(0..64usize, 0..64)) {
            let mut bitmap = Bitmap::<U64>::new();
            for i in &bits {
                bitmap.set(*i, true);
            }
            for i in 0..64 {
                assert_eq!(bitmap.get(i), bits.contains(&i));
            }
            assert!(bitmap.into_iter().eq(bits.into_iter()));
        }
    }
}
