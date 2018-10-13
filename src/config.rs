// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use typenum::*;

/// The branching factor of RRB-trees
pub type VectorChunkSize = U64;

/// The branching factor of B-trees
pub type OrdChunkSize = U64; // Must be an even number!

/// The level size of HAMTs, in bits
/// Branching factor is 2 ^ HashLevelSize.
pub type HashLevelSize = U5;
