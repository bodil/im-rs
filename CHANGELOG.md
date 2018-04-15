# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

* Map/set methods which accept references to keys will now also take
  any value that's borrowable to the key's type, ie. it will take a
  reference to a type `Borrowable` where the key implements
  `Borrow<Borrowable>`. This is particularly handy for types such as
  `String` because you can now pass `&str` to key lookups instead of
  `&String`. So, instead of the incredibly cumbersome
  `map.get(&"foo".to_string())` you can just do `map.get("foo")` when
  looking up a mapping for a string literal.

## [10.1.0] - 2018-04-12
### Added

* `Vector`, `OrdMap` and `HashMap` now implement `Index` and
  `IndexMut`, allowing for syntax like `map[key] = value`.
* Added `cons`, `snoc`, `uncons` and `unsnoc` aliases where they were missing.
* Everything now implements `Sum` and `Extend` where possible.

### Changed

* Generalised `OrdMap`/`OrdSet`'s internal nodes so `OrdSet` now only
  needs to store pointers to its values, not pairs of pointers to
  value and `Unit`. This has caused `OrdMap/Set`'s type constraints to
  tighten somewhat - in particular, iteration over maps/sets whose
  keys don't implement `Ord` is no longer possible, but as you would
  only have been able to create empty instances of these, no sensible
  code should break because of this.
* `HashMap`/`HashSet` now also cannot be iterated over unless they
  implement `Hash + Eq`, with the same note as above.
* Constraints on single operations that take closures on `HashMap` and
  `OrdMap` have been relaxed from `Fn` to `FnOnce`. (Fixes #7.)

### Fixed

* Hashes are now stored in `HashMap`s along with their associated
  values, removing the need to recompute the hash when a value is
  reordered inside the tree.

## [10.0.0] - 2018-03-25
### Added

This is the first release to be considered reasonably stable. No
changelog has been kept until now.
