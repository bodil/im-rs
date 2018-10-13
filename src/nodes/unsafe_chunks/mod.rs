// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![allow(unsafe_code)]

pub mod sized_chunk;
pub mod sparse_chunk;

pub mod chunk {
    use super::sized_chunk as sc;
    use config::VectorChunkSize;
    use typenum::Unsigned;

    pub type Chunk<A> = sc::Chunk<A, VectorChunkSize>;
    pub type Iter<A> = sc::Iter<A, VectorChunkSize>;
    pub const CHUNK_SIZE: usize = VectorChunkSize::USIZE;
}
