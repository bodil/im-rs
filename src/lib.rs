#[cfg(any(test, feature = "quickcheck"))]
#[macro_use]
extern crate quickcheck;

#[cfg(feature = "quickcheck")]
quickcheck!{}

#[macro_use]
pub mod list;

pub use list::List;
