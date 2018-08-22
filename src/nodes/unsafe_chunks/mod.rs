#![allow(unsafe_code)]

pub mod sized_chunk;
pub mod sparse_chunk;

pub mod chunk {
    use super::sized_chunk as sc;
    use typenum::*;

    pub type ChunkSize = U64;
    pub type Chunk<A> = sc::Chunk<A, ChunkSize>;
    pub type Iter<A> = sc::Iter<A, ChunkSize>;
    pub const CHUNK_SIZE: usize = ChunkSize::USIZE;
}
