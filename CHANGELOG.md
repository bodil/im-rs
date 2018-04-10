# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

* Added `cons`, `snoc`, `uncons` and `unsnoc` aliases where they were missing.
* Everything now implements `Sum` and `Extend` where possible.

### Changed

* Generalised `OrdMap`/`OrdSet`'s internal nodes so `OrdSet` now only
  needs to store pointers to its values, not pairs of pointers to
  value and `Unit`. This has caused `OrdMap/Set`'s type constraints to
  tighten somewhat - in particular, iteration over maps/sets whose
  keys don't implement `Ord` is no longer possible, but as you would
  only have been able to create empty instances of these, no real code
  should break because of this.
* `HashMap`/`HashSet` now also cannot be iterated over unless they
  implement `Hash + Eq`, with the same note as above.

### Fixed

* Hashes are now stored in `HashMap`s along with their associated
  values, removing the need to recompute the hash when a value is
  reordered inside the tree.

## [10.0.0] - 2018-03-25
### Added

This is the first release to be considered reasonably stable. No
changelog has been kept until now.
