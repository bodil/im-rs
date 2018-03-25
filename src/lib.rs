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
//! are kept inside [`Arc`][std::sync::Arc]s. Methods will generally accept either
//! owned values or [`Arc`][std::sync::Arc]s and perform conversion as needed, but
//! you'll have to expect to receive [`Arc`][std::sync::Arc]s when iterating or
//! performing lookup operations. All caveats about using
//! reference counted values apply in general (eg. reference
//! counting is simplistic and doesn't detect loops).
//!
//! A design goal of this library is to make using immutable
//! data structures as easy as it is in higher level
//! languages, but obviously there's only so much you can do.
//! Methods will generally attempt to coerce argument values
//! where they can: where an [`Arc`][std::sync::Arc] is called for, it will be
//! able to figure out how to convert whatever is provided
//! into an [`Arc`][std::sync::Arc] if it isn't already.
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
//! Haskell) where possible (eg. `Vector::new()` rather than
//! `Vector::empty()`, `HashMap::get()` rather than `HashMap::lookup()`).
//! Where Rust equivalents don't exist, terminology tends to
//! follow Haskell where the Haskell isn't too confusing,
//! or, when it is, we provide more readily understandable
//! aliases (because we wouldn't want to deprive the user
//! of their enjoyment of the word '[`snoc`][conslist::ConsList::snoc],'
//! even though it's reportedly not an obviously intuitive term).
//!
//! ## Why Immutable Data Structures
//!
//! Programming with immutable values, meaning that references to values
//! can be relied on to always point to the same value, means that you can
//! stop worrying about other parts of your code tampering with a value
//! you're working on unexpectedly, or from unexpected parts of your code,
//! making it a lot easier to keep track of what your code is actually doing.
//!
//! Mutable values are, generally, a huge source of unexpected behaviour that a lot
//! of languages, like Haskell, Elm and Clojure, have been designed to avoid
//! altogether. Rust, being what it is, does a good job of discouraging this kind
//! of behaviour, and keeping it strictly controlled when it's necessary, but the
//! standard library doesn't provide collection data structures which are optimised
//! for immutable operations. This means, for instance, that if you want to add
//! an item to a [`Vec`][std::vec::Vec] without modifying it in place, you first need
//! to [`clone`][std::clone::Clone::clone] the whole thing before making your change.
//!
//! Data structures exist which are designed to be able to make these copies much
//! cheaper, usually by sharing structure between them, which, because this structure
//! is also immutable, is both cheap and safe. The most basic example of this kind
//! of data structure is the [`ConsList`][conslist::ConsList], where, if you have a
//! list *L* and you want to push an item *I* to the front of it, you'll get back a new
//! list which literally contains the data *'item I followed by list L.'* This operation
//! is extremely inexpensive, but of course this also means that certain other operations
//! which would be inexpensive for a [`Vec`][std::vec::Vec] are much more costly for a
//! [`ConsList`][conslist::ConsList]—index lookup is an example of this, where for a
//! [`Vec`][std::vec::Vec] it's just a matter of going to memory location
//! *index times item size* inside the [`Vec`][std::vec::Vec]'s memory buffer, but for
//! a [`ConsList`][conslist::ConsList] you'd have to walk through the entire list from
//! the start, following references through to other list nodes, until you get to the
//! right item.
//!
//! While all immutable data structures tend to be less efficient than their mutable
//! counterparts, when chosen carefully they can perform just as well for the operations
//! you need, and there are some, like [`Vector`][vector::Vector] and [`HashMap`][hashmap::HashMap],
//! which have performance characteristics good enough for most operations that you can
//! safely choose them without worrying too much about whether they're going to be the
//! right choice for any given use case. Better yet, most of them can even be safely
//! mutated in place when they aren't sharing any structure with other instances,
//! making them nearly as performant as their mutable counterparts.
//!
//! ## Data Structures
//!
//! We'll attempt to provide a comprehensive guide to the
//! available data structures below.
//!
//! ### Performance Notes
//!
//! If you're not familiar with big O notation, here's a quick cheat sheet:
//!
//! *O(1)* means an operation runs in constant time: it will take the same time to
//! complete regardless of the size of the data structure.
//!
//! *O(n)* means an operation runs in linear time: if you double the size of your
//! data structure, the operation will take twice as long to complete; if you
//! quadruple the size, it will take four times as long.
//!
//! *O(log n)* means an operation runs in logarithmic time: if you double the size
//! of your data structure, the operation will take roughly one step longer to
//! complete; if you quadruple the size, it will need two steps more; and so on.
//!
//! *O(1)** means 'amortised O(1),' which means that an operation usually runs in
//! constant time but will sometimes be more expensive, as much as O(n) for some
//! data structures.
//!
//! ### Lists
//!
//! Lists are ordered sequences of single elements, usually with cheap push/pop
//! operations, and index lookup tends to be O(n). Lists are for collections of
//! items where you expect to iterate rather than lookup.
//!
//! | Type | Constraints | Order | Push Front | Pop Front | Push Back | Pop Back | Append | Lookup |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | [`Vector<A>`][vector::Vector] | | insertion | O(1)* | O(1)* | O(1)* | O(1)* | O(n) | O(1)* |
//! | [`CatList<A>`][catlist::CatList] | | insertion | O(1)* | O(1)* | O(1)* | O(1)* | O(1) | O(n) |
//! | [`ConsList<A>`][conslist::ConsList] | | insertion | O(1) | O(1) | O(n) | O(n) | O(n) | O(n) |
//!
//! ### Maps
//!
//! Maps are mappings of keys to values, where the most common read operation is
//! to find the value associated with a given key. Maps may or may not have a defined
//! order. Any given key can only occur once inside a map, and setting a key to a
//! different value will overwrite the previous value.
//!
//! | Type | Key Constraints | Order | Insert | Remove | Lookup |
//! | --- | --- | --- | --- | --- | --- |
//! | [`HashMap<K, V>`][hashmap::HashMap] | [`Hash`][std::hash::Hash] + [`Eq`][std::cmp::Eq] | undefined | O(1)* | O(1)* | O(1)* |
//! | [`OrdMap<K, V>`][ordmap::OrdMap] | [`Ord`][std::cmp::Ord] | sorted | O(log n) | O(log n) | O(log n) |
//!
//! ### Sets
//!
//! Sets are collections of unique values, and may or may not have a defined order.
//! Their crucial property is that any given value can only exist once in a given set.
//!
//! Values in sets, in practice, behave exactly like keys in maps, and are therefore
//! normally implemented as maps from type `A` to type `()`.
//!
//! | Type | Constraints | Order | Insert | Remove | Lookup |
//! | --- | --- | --- | --- | --- | --- |
//! | [`HashSet<A>`][hashset::HashSet] | [`Hash`][std::hash::Hash] + [`Eq`][std::cmp::Eq] | undefined | O(1)* | O(1)* | O(1)* |
//! | [`OrdSet<A>`][ordset::OrdSet] | [`Ord`][std::cmp::Ord] | sorted | O(log n) | O(log n) | O(log n) |
//!
//! ## In-place Mutation
//!
//! Some data structures (currently only [`HashMap`][hashmap::HashMap]), support
//! in-place copy-on-write mutation, which means that if you're the sole
//! user of a data structure, you can update it in place with a huge performance
//! benefit (about an order of magnitude faster than immutable operations, almost
//! as fast as [`std::collections`][std::collections]'s mutable data structures).
//!
//! Thanks to [`Arc`][std::sync::Arc]'s reference counting, we are able to determine
//! whether a node in a data structure is being shared with other data structures,
//! or whether it's safe to mutate it in place. When it's shared, we'll automatically
//! make a copy of the node before modifying it, thus preserving the usual guarantees
//! you get from using an immutable data structure.
//!
//! [std::collections]: https://doc.rust-lang.org/std/collections/index.html
//! [std::vec::Vec]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//! [std::sync::Arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
//! [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
//! [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html
//! [std::clone::Clone]: https://doc.rust-lang.org/std/clone/trait.Clone.html
//! [std::clone::Clone::clone]: https://doc.rust-lang.org/std/clone/trait.Clone.html#tymethod.clone
//! [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
//! [hashmap::HashMap]: ./hashmap/struct.HashMap.html
//! [hashset::HashSet]: ./hashset/struct.HashSet.html
//! [ordmap::OrdMap]: ./ordmap/struct.OrdMap.html
//! [ordset::OrdSet]: ./ordset/struct.OrdSet.html
//! [conslist::ConsList]: ./conslist/struct.ConsList.html
//! [catlist::CatList]: ./catlist/struct.CatList.html
//! [vector::Vector]: ./vector/struct.Vector.html
//! [conslist::ConsList::snoc]: ./conlist/struct.ConsList.html#method.snoc

// Get some clippy feedback: `cargo +nightly build --features "clippy"`
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![cfg_attr(feature = "clippy", allow(type_complexity))]
#![cfg_attr(feature = "clippy", allow(unreadable_literal))]
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
extern crate lazy_static;

mod bits;
pub mod hash;
#[macro_use]
pub mod conslist;
#[macro_use]
pub mod ordmap;
#[macro_use]
pub mod hashmap;
#[macro_use]
pub mod ordset;
#[macro_use]
pub mod hashset;
#[macro_use]
pub mod catlist;
#[macro_use]
pub mod vector;

pub mod iter;
pub mod lens;
pub mod shared;

#[cfg(test)]
pub mod test;

#[cfg(any(test, feature = "serde"))]
pub mod ser;

pub use ordmap::OrdMap;
pub use hashmap::HashMap;
pub use ordset::OrdSet;
pub use hashset::HashSet;
pub use catlist::CatList;
pub use conslist::ConsList;
pub use vector::Vector;
pub use iter::unfold;

pub type List<A> = vector::Vector<A>;
pub type Set<A> = hashset::HashSet<A>;
pub type Map<K, V> = hashmap::HashMap<K, V>;
