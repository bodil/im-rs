// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use nodes::types::Bits;

#[derive(PartialEq, Eq)]
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
