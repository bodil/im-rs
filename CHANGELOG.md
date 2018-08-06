# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `Vector` now has `Focus` and `FocusMut` APIs for caching index lookups,
  yielding huge performance gains when performing multiple adjacent index
  lookups.  `Vector::iter` has been reimplemented using this API, and is now
  much simpler and about twice as fast as a result, and `Vector::iter_mut` now
  runs nearly an order of magnitude faster. Likewise, `Vector::sort` and
  `Vector::retain` are now using `FocusMut` and run considerably faster as a
  result.
- `Vector::set` now returns the replaced value.

### Changed
- As `std::ops::RangeBounds` is now stabilised in Rust 1.28, the `Vector::slice`
  method is now unconditionally available on the stable channel.
- Union/difference/intersection/is_submap methods on `HashMap` and `OrdMap` that
  take functions now take `FnMut` instead of `Fn`. This should not affect any
  existing code. (#34)
- `Vector::split_off` can now take an index equal to the length of the vector,
  yielding an empty vector as the split result. (#33)

### Fixed
- `Vector` is now represented as a single inline chunk until it grows larger
  than the chunk size, making it even faster than `Vec` at small sizes, though
  `clone` could now be slower if the clone is expensive (it's still absurdly
  fast for `A: Copy`).

## [11.0.1] - 2018-07-23

### Fixed
- Various performance improvements, amounting to a 5-10% speedup for both kinds
  of map/set.
- Fixed an edge case bug in `sort::quicksort`.

## [11.0.0] - 2018-07-10

### Changed

This is a major release with many breaking changes, and is intended to stabilise
the API more than to denote that the rewrite is now production ready. You should
expect future releases with significant performance improvements as well as
additional APIs, but there should be no further major release with breaking
changes in the immediate future, barring very serious unforeseen issues.

Specifically, you should expect imminent minor releases with performance
improvements for `Vector` and `OrdMap`, for which I have a number of known
optimisations that remain unimplemented.

#### No More `Arc`

All data structures have been reworked to take values of `A: Clone` instead of
`Arc<A>`, meaning that there's less performance overhead (as well as mental
overhead) when using values that clone cheaply. The performance gain when values
are `A: Copy` is a factor of two or more. It's expected that users should wrap
values in `Arc` themselves when using values which are expensive to clone.

Data structures still use reference counters internally to reference nodes, but
values are stored directly in the nodes with no further indirection. This is
also good for cache locality.

Data structures now use `Rc` instead of `Arc` by default to do reference
counting. If you need a thread safe version that implements `Send` and `Sync`,
you can enable the `arc` feature on the package to compile with `Arc` instead.

#### `std::collections` Compatible API

The API has been reworked to align more closely with `std::collections`,
favouring mutable operations by default, so that operations that were previously
suffixed with `_mut` are now the standard operations, and immutable operations
which return a modified copy have been given different names altogether. In
short, all your code using previous versions of this library will no longer
work, and if it was relying heavily on immutable operations, it's recommended
that you rewrite it to be mutable by preference, but you should generally be
able to make it work again by using the new method names for the immutable
operations.

Here is a list of the most notable changed method names for maps and sets:

| Previous immutable | Current immutable | Previous mutable | Current mutable |
| ------------------ | ----------------- | ---------------- | --------------- |
| `insert`           | `update`          | `insert_mut`     | `insert`        |
| `remove`           | `without`         | `remove_mut`     | `remove`        |
| `pop`              | `extract`         | `pop_mut`        | `remove`        |

You should expect to be able to rewrite code using `std::collections::HashMap`
and `std::collections::BTreeMap` with minimal or no changes using `im::HashMap`
and `im::OrdMap` respectively.

`Vector` has been completely rewritten and has an API that aligns closely with
`std::collections::VecDeque`, with very few immutable equivalents. It's expected
that you should use `Vector::clone()` to take a snapshot when you need it rather
than cause an implicit clone for each operation. (It's still O(1) and
practically instant.)

I'm considering adding back some of the immutable operations if I can come up
with good names for them, but for now, just `clone` it if you need it.

#### RRB Vector

`Vector` is now implemented as an [RRB
tree](https://infoscience.epfl.ch/record/213452/files/rrbvector.pdf) with [smart
head/tail chunking](http://gallium.inria.fr/~rainey/chunked_seq.pdf), obsoleting
the previous [Hickey
trie](https://hypirion.com/musings/understanding-persistent-vector-pt-1)
implementation.

RRB trees have generally similar performance characteristics to the Hickey trie,
with the added benefit of having O(log n) splitting and concatenation.

| Operation       | RRB tree | Hickey trie | Vec    | VecDeque |
| --------------- | -------- | ----------- | ------ | -------- |
| Push front      | O(1)\*   | O(log n)    | O(n)   | O(1)\*   |
| Push back       | O(1)\*   | O(log n)    | O(1)\* | O(1)\*   |
| Pop front       | O(1)\*   | O(log n)    | O(n)   | O(1)\*   |
| Pop back        | O(1)\*   | O(log n)    | O(1)   | O(1)\*   |
| Lookup by index | O(log n) | O(log n)    | O(1)   | O(1)     |
| Split           | O(log n) | O(log n)    | O(n)   | O(n)     |
| Join            | O(log n) | O(n)        | O(n)   | O(n)     |

(Please note that the timings above are for the `im` version of the Hickey trie,
based on the [Immutable.js](https://facebook.github.io/immutable-js/)
implementation, which performs better than the original Clojure version on
splits and push/pop front, but worse on push/pop back).

The RRB tree is the most generally efficient list like data structure currently
known, to my knowledge, but obviously it does not and cannot perform as well as
a simple `Vec` on certain operations. It makes up for that by having no
operations you need to worry about the performance complexity of: nothing you
can do to an RRB tree is going to be more expensive than just iterating over it.
For larger data sets, being able to concatenate (and, by extension, insert and
remove at arbitrary locations) several orders of magnitude faster than `Vec`
could also be considered a selling point.

#### No More `CatList` And `ConsList`

`CatList` has been superseded by `Vector`, and `ConsList` was generally not very
useful except in the more peculiar edge cases where memory consumption matters
more than performance, and keeping it in line with current API changes wasn't
practical.

#### No More Funny Words

Though it breaks my heart, words like `cons`, `snoc`, `car`, `cdr` and `uncons`
are no longer used in the `im` API, to facilitiate closer alignment with
`std::collections`. Even the `head`/`tail` pair is gone, though `head` and
`last` remain as aliases for `front` and `back`.

## [10.2.0] - 2018-04-15

### Added

-   Map/set methods which accept references to keys will now also take any value
    that's borrowable to the key's type, ie. it will take a reference to a type
    `Borrowable` where the key implements `Borrow<Borrowable>`. This is
    particularly handy for types such as `String` because you can now pass `&str`
    to key lookups instead of `&String`. So, instead of the incredibly cumbersome
    `map.get(&"foo".to_string())` you can just do `map.get("foo")` when looking up
    a mapping for a string literal.

## [10.1.0] - 2018-04-12

### Added

-   `Vector`, `OrdMap` and `HashMap` now implement `Index` and `IndexMut`,
    allowing for syntax like `map[key] = value`.
-   Added `cons`, `snoc`, `uncons` and `unsnoc` aliases where they were missing.
-   Everything now implements `Sum` and `Extend` where possible.

### Changed

-   Generalised `OrdMap`/`OrdSet`'s internal nodes so `OrdSet` now only needs to
    store pointers to its values, not pairs of pointers to value and `Unit`. This
    has caused `OrdMap/Set`'s type constraints to tighten somewhat - in
    particular, iteration over maps/sets whose keys don't implement `Ord` is no
    longer possible, but as you would only have been able to create empty
    instances of these, no sensible code should break because of this.
-   `HashMap`/`HashSet` now also cannot be iterated over unless they implement
    `Hash + Eq`, with the same note as above.
-   Constraints on single operations that take closures on `HashMap` and `OrdMap`
    have been relaxed from `Fn` to `FnOnce`. (Fixes #7.)

### Fixed

-   Hashes are now stored in `HashMap`s along with their associated values,
    removing the need to recompute the hash when a value is reordered inside the
    tree.

## [10.0.0] - 2018-03-25

### Added

This is the first release to be considered reasonably stable. No changelog has
been kept until now.
