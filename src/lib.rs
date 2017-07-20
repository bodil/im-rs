//! # Immutable Data Structures for Rust
//!
//! This library implements several of the more commonly useful
//! immutable data structures for Rust. They rely on structural
//! sharing to keep most operations fast without needing to mutate
//! the underlying data store, leading to more predictable code
//! without necessarily sacrificing performance.
//!
//! Because Rust is not a garbage collected language, and
//! immutable data structures generally rely on some sort of
//! garbage collection, values inside these data structures
//! are kept inside `Arc`s. Methods will generally accept either
//! owned values or `Arc`s and perform conversion as needed, but
//! you'll have to expect to receive `Arc`s when iterating or
//! performing lookup operations. All caveats about using
//! reference counted values apply in general (eg. reference
//! counting is simplistic and doesn't detect loops).
//!
//! A design goal of this library is to make using immutable
//! data structures as easy as it is in higher level
//! languages, but obviously there's only so much you can do.
//! Methods will generally attempt to coerce argument values
//! where they can: where an `Arc` is called for, it will be
//! able to utilise `From` implementations to coerce values
//! into `Arc`s on the fly.
//!
//! It's also been a design goal to provide as complete an
//! API as possible, which in a practical sense has meant
//! going over the equivalent implementations for Haskell to
//! ensure the API covers the same set of use cases. This
//! obviously doesn't include things like `Foldable` and
//! `Functor` which aren't yet expressible in Rust, but in
//! these cases we've tried to make sure Rust iterators are
//! able to perform the same tasks.
//!
//! Care has been taken to use method names similar to those
//! in Rust over those used in the source material (largely
//! Haskell) where possible (eg. `List::new()` rather than
//! `List::empty()`, `Map::get()` rather than `Map::lookup()`).
//! Where Rust equivalents don't exist, terminology tends to
//! follow Haskell where the Haskell isn't too confusing,
//! or, when it is, we provide more readily understandable
//! aliases (because we wouldn't want to deprive the user
//! of their enjoyment of the word 'snoc,' even though it's
//! reportedly not a readily intuitive term).

#![cfg_attr(has_specialisation, feature(specialization))]

#[cfg(any(test, feature = "quickcheck"))]
#[macro_use]
extern crate quickcheck;

#[cfg(feature = "quickcheck")]
quickcheck!{}

#[cfg(any(test, feature = "proptest"))]
#[macro_use]
extern crate proptest;

#[cfg(feature = "proptest")]
proptest!{}

#[cfg(any(test, feature = "serde"))]
extern crate serde;
#[cfg(test)]
extern crate serde_json;

#[macro_use]
pub mod conslist;
#[macro_use]
pub mod map;
#[macro_use]
pub mod set;
#[macro_use]
pub mod list;
pub mod queue;
pub mod iter;
pub mod lens;

pub use conslist::ConsList;
pub use map::Map;
pub use set::Set;
pub use queue::Queue;
pub use list::List;
pub use iter::unfold;

#[cfg(test)]
pub mod test;

#[cfg(any(test, feature = "serde"))]
pub mod ser;
