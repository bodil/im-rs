#![allow(unsafe_code)]

pub mod sized_chunk;
pub mod sparse_chunk;

pub mod chunk {
    use super::sized_chunk as sc;
    use typenum::*;

    pub type ChunkSize = U64;
    pub type Chunk<A> = sc::SizedChunk<A, ChunkSize>;
    // pub type Iter<'a, A> = sc::Iter<'a, A, ChunkSize>;
    // pub type IterMut<'a, A> = sc::IterMut<'a, A, ChunkSize>;
    pub type ConsumingIter<A> = sc::ConsumingIter<A, ChunkSize>;
    pub const CHUNK_SIZE: usize = ChunkSize::USIZE;
}
