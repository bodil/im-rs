use typenum::U64;

/// The branching factor of RRB-trees
pub type VectorChunkSize = U64;

/// The branching factor of B-trees
pub type OrdChunkSize = U64; // Must be an even number!

/// The branching factor of HAMTs
pub const HASH_SHIFT: usize = 5;
pub const HASH_SIZE: usize = 1 << HASH_SHIFT;
pub type HashBits = u32; // a uint of HASH_SIZE bits
