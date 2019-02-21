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

    pub type Chunk<A> = sc::sized_chunk::Chunk<A, VectorChunkSize>;
    pub type Iter<A> = sc::sized_chunk::Iter<A, VectorChunkSize>;
    pub const CHUNK_SIZE: usize = VectorChunkSize::USIZE;
}
