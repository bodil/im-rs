// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub(crate) mod btree;
pub(crate) mod hamt;
pub(crate) mod rrb;

pub(crate) mod chunk {
    pub(crate) use crate::config::VECTOR_CHUNK_SIZE as CHUNK_SIZE;
    pub(crate) type Chunk<A> = sized_chunks::sized_chunk::Chunk<A, CHUNK_SIZE>;
}
