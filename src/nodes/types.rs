// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::fmt::Debug;
use std::marker::PhantomData;

use typenum::*;

// Chunk sizes

pub trait ChunkLength<A>: Unsigned {
    type SizedType;
}

impl<A> ChunkLength<A> for UTerm {
    type SizedType = ();
}

#[doc(hidden)]
#[allow(dead_code)]
pub struct SizeEven<A, B> {
    parent1: B,
    parent2: B,
    _marker: PhantomData<A>,
}

#[doc(hidden)]
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

// Bit field sizes

pub trait Bits: Unsigned {
    type Store: Default + Copy + PartialEq + Debug;

    fn get(bits: &Self::Store, index: usize) -> bool;
    fn set(bits: &mut Self::Store, index: usize, value: bool) -> bool;
    fn len(bits: &Self::Store) -> usize;
    fn first_index(bits: &Self::Store) -> Option<usize>;
}

macro_rules! bits_for {
    ($num:ty, $result:ty) => {
        impl Bits for $num {
            type Store = $result;

            fn get(bits: &$result, index: usize) -> bool {
                debug_assert!(index < Self::USIZE);
                bits & (1 << index) != 0
            }

            fn set(bits: &mut $result, index: usize, value: bool) -> bool {
                debug_assert!(index < Self::USIZE);
                let mask = 1 << index;
                let prev = *bits & mask;
                if value {
                    *bits |= mask;
                } else {
                    *bits &= !mask;
                }
                prev != 0
            }

            fn len(bits: &$result) -> usize {
                bits.count_ones() as usize
            }

            fn first_index(bits: &$result) -> Option<usize> {
                if *bits == 0 {
                    None
                } else {
                    Some(bits.trailing_zeros() as usize)
                }
            }
        }
    };
}

bits_for!(U1, u8);
bits_for!(U2, u8);
bits_for!(U3, u8);
bits_for!(U4, u8);
bits_for!(U5, u8);
bits_for!(U6, u8);
bits_for!(U7, u8);
bits_for!(U8, u8);
bits_for!(U9, u16);
bits_for!(U10, u16);
bits_for!(U11, u16);
bits_for!(U12, u16);
bits_for!(U13, u16);
bits_for!(U14, u16);
bits_for!(U15, u16);
bits_for!(U16, u16);
bits_for!(U17, u32);
bits_for!(U18, u32);
bits_for!(U19, u32);
bits_for!(U20, u32);
bits_for!(U21, u32);
bits_for!(U22, u32);
bits_for!(U23, u32);
bits_for!(U24, u32);
bits_for!(U25, u32);
bits_for!(U26, u32);
bits_for!(U27, u32);
bits_for!(U28, u32);
bits_for!(U29, u32);
bits_for!(U30, u32);
bits_for!(U31, u32);
bits_for!(U32, u32);
bits_for!(U33, u64);
bits_for!(U34, u64);
bits_for!(U35, u64);
bits_for!(U36, u64);
bits_for!(U37, u64);
bits_for!(U38, u64);
bits_for!(U39, u64);
bits_for!(U40, u64);
bits_for!(U41, u64);
bits_for!(U42, u64);
bits_for!(U43, u64);
bits_for!(U44, u64);
bits_for!(U45, u64);
bits_for!(U46, u64);
bits_for!(U47, u64);
bits_for!(U48, u64);
bits_for!(U49, u64);
bits_for!(U50, u64);
bits_for!(U51, u64);
bits_for!(U52, u64);
bits_for!(U53, u64);
bits_for!(U54, u64);
bits_for!(U55, u64);
bits_for!(U56, u64);
bits_for!(U57, u64);
bits_for!(U58, u64);
bits_for!(U59, u64);
bits_for!(U60, u64);
bits_for!(U61, u64);
bits_for!(U62, u64);
bits_for!(U63, u64);
bits_for!(U64, u64);
// No u128 because stdlib hashes are only u64.
