// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub mod btree;
pub mod hamt;
pub mod rrb;

pub use sized_chunks::*;

pub mod chunk {
    use crate::config::VectorChunkSize;
    use sized_chunks as sc;
    use typenum::Unsigned;

    pub type Chunk<A> = sc::ring_buffer::RingBuffer<A, VectorChunkSize>;
    pub type Slice<'a, A> = sc::ring_buffer::Slice<'a, A, VectorChunkSize>;
    pub type SliceMut<'a, A> = sc::ring_buffer::SliceMut<'a, A, VectorChunkSize>;
    pub type Iter<A> = sc::ring_buffer::OwnedIter<A, VectorChunkSize>;
    pub const CHUNK_SIZE: usize = VectorChunkSize::USIZE;
}
