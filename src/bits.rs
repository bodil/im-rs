pub type Bitmap = u16; // a uint of HASH_SIZE bits
pub const HASH_BITS: usize = 4;
pub const HASH_SIZE: usize = 1 << HASH_BITS;
pub const HASH_MASK: Bitmap = (HASH_SIZE - 1) as Bitmap;
pub const HASH_COERCE: u64 = ((2 ^ HASH_SIZE) - 1) as u64;

#[inline]
pub fn mask(hash: Bitmap, shift: usize) -> Bitmap {
    hash >> shift & HASH_MASK
}

#[inline]
pub fn bitpos(hash: Bitmap, shift: usize) -> Bitmap {
    1 << mask(hash, shift)
}

#[inline]
pub fn bit_index(bitmap: Bitmap, bit: Bitmap) -> usize {
    (bitmap & (bit - 1)).count_ones() as usize
}
