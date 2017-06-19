#[cfg(any(test, feature = "quickcheck"))]
#[macro_use]
extern crate quickcheck;

#[cfg(feature = "quickcheck")]
quickcheck!{}

#[macro_use]
pub mod list;
pub mod map;
pub mod set;
pub mod queue;
pub mod seq;

pub use list::List;
pub use map::Map;
pub use set::Set;
pub use queue::Queue;
pub use seq::Seq;

#[cfg(test)]
pub mod test;
