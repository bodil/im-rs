// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An ordered map.
//!
//! An immutable ordered map implemented as a [B-tree] [1].
//!
//! Most operations on this type of map are O(log n). A
//! [`HashMap`][hashmap::HashMap] is usually a better choice for
//! performance, but the `OrdMap` has the advantage of only requiring
//! an [`Ord`][std::cmp::Ord] constraint on the key, and of being
//! ordered, so that keys always come out from lowest to highest,
//! where a [`HashMap`][hashmap::HashMap] has no guaranteed ordering.
//!
//! [1]: https://en.wikipedia.org/wiki/B-tree
//! [hashmap::HashMap]: ../hashmap/struct.HashMap.html
//! [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, Iterator, Sum};
use std::mem;
use std::ops::{Add, Index, IndexMut};

use hashmap::HashMap;
use nodes::btree::{BTreeValue, Insert, Node, Remove};
#[cfg(has_specialisation)]
use util::linear_search_by;
use util::Ref;

pub use nodes::btree::{ConsumingIter, DiffItem, DiffIter, Iter};

/// Construct a map from a sequence of key/value pairs.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::ordmap::OrdMap;
/// # fn main() {
/// assert_eq!(
///   ordmap!{
///     1 => 11,
///     2 => 22,
///     3 => 33
///   },
///   OrdMap::from(vec![(1, 11), (2, 22), (3, 33)])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! ordmap {
    () => { $crate::ordmap::OrdMap::new() };

    ( $( $key:expr => $value:expr ),* ) => {{
        let mut map = $crate::ordmap::OrdMap::new();
        $({
            map.insert($key, $value);
        })*;
        map
    }};
}

#[cfg(not(has_specialisation))]
impl<K: Ord + Clone, V: Clone> BTreeValue for (K, V) {
    type Key = K;

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }

    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        slice.binary_search_by(|value| Self::Key::borrow(&value.0).cmp(key))
    }

    fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        slice.binary_search_by(|value| value.0.cmp(&key.0))
    }

    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        Self::Key::borrow(&self.0).cmp(other)
    }

    fn cmp_values(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

#[cfg(has_specialisation)]
impl<K: Ord + Clone, V: Clone> BTreeValue for (K, V) {
    type Key = K;

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }

    default fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        slice.binary_search_by(|value| Self::Key::borrow(&value.0).cmp(key))
    }

    default fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        slice.binary_search_by(|value| value.0.cmp(&key.0))
    }

    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        Self::Key::borrow(&self.0).cmp(other)
    }

    fn cmp_values(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

#[cfg(has_specialisation)]
impl<K: Ord + Clone + Copy, V: Clone> BTreeValue for (K, V) {
    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        linear_search_by(slice, |value| Self::Key::borrow(&value.0).cmp(key))
    }

    fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        linear_search_by(slice, |value| value.0.cmp(&key.0))
    }
}

/// An ordered map.
///
/// An immutable ordered map implemented as a B-tree.
///
/// Most operations on this type of map are O(log n). A
/// [`HashMap`][hashmap::HashMap] is usually a better choice for
/// performance, but the `OrdMap` has the advantage of only requiring
/// an [`Ord`][std::cmp::Ord] constraint on the key, and of being
/// ordered, so that keys always come out from lowest to highest,
/// where a [`HashMap`][hashmap::HashMap] has no guaranteed ordering.
///
/// [hashmap::HashMap]: ../hashmap/struct.HashMap.html
/// [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html
pub struct OrdMap<K, V> {
    root: Ref<Node<(K, V)>>,
}

impl<K, V> OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Construct an empty map.
    #[must_use]
    pub fn new() -> Self {
        OrdMap {
            root: Ref::from(Node::new()),
        }
    }

    /// Construct a map with a single mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map = OrdMap::singleton(123, "onetwothree");
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"onetwothree")
    /// );
    /// # }
    /// ```
    #[must_use]
    pub fn singleton(key: K, value: V) -> Self {
        OrdMap {
            root: Ref::from(Node::unit((key, value))),
        }
    }

    /// Test whether a map is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// assert!(
    ///   !ordmap!{1 => 2}.is_empty()
    /// );
    /// assert!(
    ///   OrdMap::<i32, i32>::new().is_empty()
    /// );
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size of a map.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// assert_eq!(3, ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.len());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Get the largest key in a map, along with its value. If the map
    /// is empty, return `None`.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// assert_eq!(Some(&(3, 33)), ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.get_max());
    /// # }
    /// ```
    #[must_use]
    pub fn get_max(&self) -> Option<&(K, V)> {
        self.root.max()
    }

    /// Get the smallest key in a map, along with its value. If the
    /// map is empty, return `None`.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// assert_eq!(Some(&(1, 11)), ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.get_min());
    /// # }
    /// ```
    #[must_use]
    pub fn get_min(&self) -> Option<&(K, V)> {
        self.root.min()
    }

    /// Get an iterator over the key/value pairs of a map.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, (K, V)> {
        Iter::new(&self.root)
    }

    /// Get an iterator over a map's keys.
    #[must_use]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { it: self.iter() }
    }

    /// Get an iterator over a map's values.
    #[must_use]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { it: self.iter() }
    }

    /// Get an iterator over the differences between this map and
    /// another, i.e. the set of entries to add, update, or remove to
    /// this map in order to make it equal to the other map.
    ///
    /// This function will avoid visiting nodes which are shared
    /// between the two maps, meaning that even very large maps can be
    /// compared quickly if most of their structure is shared.
    ///
    /// Time: O(n) (where n is the number of unique elements across
    /// the two maps, minus the number of elements belonging to nodes
    /// shared between them)
    #[must_use]
    pub fn diff<'a>(&'a self, other: &'a Self) -> DiffIter<'a, (K, V)> {
        DiffIter::new(&self.root, &other.root)
    }

    /// Get the value for a key from a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map = ordmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"lol")
    /// );
    /// # }
    /// ```
    #[must_use]
    pub fn get<BK>(&self, key: &BK) -> Option<&V>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.root.lookup(key).map(|(_, v)| v)
    }

    #[must_use]
    fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut V>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let root = Ref::make_mut(&mut self.root);
        root.lookup_mut(key).map(|(_, v)| v)
    }

    /// Test for the presence of a key in a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map = ordmap!{123 => "lol"};
    /// assert!(
    ///   map.contains_key(&123)
    /// );
    /// assert!(
    ///   !map.contains_key(&321)
    /// );
    /// # }
    /// ```
    #[must_use]
    pub fn contains_key<BK>(&self, k: &BK) -> bool
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.get(k).is_some()
    }

    /// Insert a key/value mapping into a map.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// map's structure which are shared with other maps will be
    /// safely copied before mutating.
    ///
    /// If the map already has a mapping for the given key, the
    /// previous value is overwritten.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let mut map = ordmap!{};
    /// map.insert(123, "123");
    /// map.insert(456, "456");
    /// assert_eq!(
    ///   map,
    ///   ordmap!{123 => "123", 456 => "456"}
    /// );
    /// # }
    /// ```
    ///
    /// [insert]: #method.insert
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let new_root = {
            let root = Ref::make_mut(&mut self.root);
            match root.insert((key, value)) {
                Insert::Replaced((_, old_value)) => return Some(old_value),
                Insert::Added => return None,
                Insert::Update(root) => Ref::from(root),
                Insert::Split(left, median, right) => {
                    Ref::from(Node::from_split(left, median, right))
                }
            }
        };
        self.root = new_root;
        None
    }

    /// Remove a key/value mapping from a map if it exists.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let mut map = ordmap!{123 => "123", 456 => "456"};
    /// map.remove(&123);
    /// map.remove(&456);
    /// assert!(map.is_empty());
    /// # }
    /// ```
    ///
    /// [remove]: #method.remove
    #[inline]
    pub fn remove<BK>(&mut self, k: &BK) -> Option<V>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.remove_with_key(k).map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value.
    ///
    /// Time: O(log n)
    pub fn remove_with_key<BK>(&mut self, k: &BK) -> Option<(K, V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let (new_root, removed_value) = {
            let root = Ref::make_mut(&mut self.root);
            match root.remove(k) {
                Remove::NoChange => return None,
                Remove::Removed(pair) => return Some(pair),
                Remove::Update(pair, root) => (Ref::from(root), Some(pair)),
            }
        };
        self.root = new_root;
        removed_value
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, the
    /// previous value is overwritten.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map = ordmap!{};
    /// assert_eq!(
    ///   map.update(123, "123"),
    ///   ordmap!{123 => "123"}
    /// );
    /// # }
    /// ```
    #[must_use]
    pub fn update(&self, key: K, value: V) -> Self {
        let mut out = self.clone();
        out.insert(key, value);
        out
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the old value and the new value,
    /// and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_with<F>(self, k: K, v: V, f: F) -> Self
    where
        F: FnOnce(V, V) -> V,
    {
        self.update_with_key(k, v, |_, v1, v2| f(v1, v2))
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the key, the old value and the new
    /// value, and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_with_key<F>(self, k: K, v: V, f: F) -> Self
    where
        F: FnOnce(&K, V, V) -> V,
    {
        match self.extract_with_key(&k) {
            None => self.update(k, v),
            Some((_, v2, m)) => {
                let out_v = f(&k, v2, v);
                m.update(k, out_v)
            }
        }
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map, returning the old value for the key as well as the new
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the key, the old value and the new
    /// value, and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_lookup_with_key<F>(self, k: K, v: V, f: F) -> (Option<V>, Self)
    where
        F: FnOnce(&K, &V, V) -> V,
    {
        match self.extract_with_key(&k) {
            None => (None, self.update(k, v)),
            Some((_, v2, m)) => {
                let out_v = f(&k, &v2, v);
                (Some(v2), m.update(k, out_v))
            }
        }
    }

    /// Update the value for a given key by calling a function with
    /// the current value and overwriting it with the function's
    /// return value.
    ///
    /// The function gets an [`Option<V>`][std::option::Option] and
    /// returns the same, so that it can decide to delete a mapping
    /// instead of updating the value, and decide what to do if the
    /// key isn't in the map.
    ///
    /// Time: O(log n)
    ///
    /// [std::option::Option]: https://doc.rust-lang.org/std/option/enum.Option.html
    #[must_use]
    pub fn alter<F>(&self, f: F, k: K) -> Self
    where
        F: FnOnce(Option<V>) -> Option<V>,
    {
        let pop = self.extract_with_key(&k);
        match (f(pop.as_ref().map(|&(_, ref v, _)| v.clone())), pop) {
            (None, None) => self.clone(),
            (Some(v), None) => self.update(k, v),
            (None, Some((_, _, m))) => m,
            (Some(v), Some((_, _, m))) => m.update(k, v),
        }
    }

    /// Remove a key/value pair from a map, if it exists.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without<BK>(&self, k: &BK) -> Self
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.extract(k)
            .map(|(_, m)| m)
            .unwrap_or_else(|| self.clone())
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value as well as the updated list.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn extract<BK>(&self, k: &BK) -> Option<(V, Self)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.extract_with_key(k).map(|(_, v, m)| (v, m))
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value as well as the updated list.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn extract_with_key<BK>(&self, k: &BK) -> Option<(K, V, Self)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let mut out = self.clone();
        let result = out.remove_with_key(k);
        result.map(|(k, v)| (k, v, out))
    }

    /// Construct the union of two maps, keeping the values in the
    /// current map when keys exist in both maps.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 3 => 3};
    /// let map2 = ordmap!{2 => 2, 3 => 4};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, map1.union(map2));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        for (k, v) in other {
            self.entry(k).or_insert(v);
        }
        self
    }

    /// Construct the union of two maps, using a function to decide
    /// what to do with the value when a key is in both maps.
    ///
    /// The function is called when a value exists in both maps, and
    /// receives the value from the current map as its first argument,
    /// and the value from the other map as the second. It should
    /// return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    #[inline]
    #[must_use]
    pub fn union_with<F>(self, other: Self, mut f: F) -> Self
    where
        F: FnMut(V, V) -> V,
    {
        self.union_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the union of two maps, using a function to decide
    /// what to do with the value when a key is in both maps.
    ///
    /// The function is called when a value exists in both maps, and
    /// receives a reference to the key as its first argument, the
    /// value from the current map as the second argument, and the
    /// value from the other map as the third argument. It should
    /// return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.union_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// # }
    /// ```
    #[must_use]
    pub fn union_with_key<F>(mut self, other: Self, mut f: F) -> Self
    where
        F: FnMut(&K, V, V) -> V,
    {
        for (key, right_value) in other {
            match self.remove(&key) {
                None => {
                    self.insert(key, right_value);
                }
                Some(left_value) => {
                    let final_value = f(&key, left_value, right_value);
                    self.insert(key, final_value);
                }
            }
        }
        self
    }

    /// Construct the union of a sequence of maps, selecting the value
    /// of the leftmost when a key appears in more than one map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 3 => 3};
    /// let map2 = ordmap!{2 => 2};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, OrdMap::unions(vec![map1, map2]));
    /// # }
    /// ```
    #[must_use]
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(Self::default(), |a, b| a.union(b))
    }

    /// Construct the union of a sequence of maps, using a function to
    /// decide what to do with the value when a key is in more than
    /// one map.
    ///
    /// The function is called when a value exists in multiple maps,
    /// and receives the value from the current map as its first
    /// argument, and the value from the next map as the second. It
    /// should return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn unions_with<I, F>(i: I, f: F) -> Self
    where
        I: IntoIterator<Item = Self>,
        F: Fn(V, V) -> V,
    {
        i.into_iter()
            .fold(Self::default(), |a, b| a.union_with(b, &f))
    }

    /// Construct the union of a sequence of maps, using a function to
    /// decide what to do with the value when a key is in more than
    /// one map.
    ///
    /// The function is called when a value exists in multiple maps,
    /// and receives a reference to the key as its first argument, the
    /// value from the current map as the second argument, and the
    /// value from the next map as the third argument. It should
    /// return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn unions_with_key<I, F>(i: I, f: F) -> Self
    where
        I: IntoIterator<Item = Self>,
        F: Fn(&K, V, V) -> V,
    {
        i.into_iter()
            .fold(Self::default(), |a, b| a.union_with_key(b, &f))
    }

    /// Construct the difference between two maps by discarding keys
    /// which occur in both maps.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2};
    /// assert_eq!(expected, map1.difference(map2));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn difference(self, other: Self) -> Self {
        self.difference_with_key(other, |_, _, _| None)
    }

    /// Construct the difference between two maps by using a function
    /// to decide what to do if a key occurs in both.
    ///
    /// Time: O(n log n)
    #[inline]
    #[must_use]
    pub fn difference_with<F>(self, other: Self, mut f: F) -> Self
    where
        F: FnMut(V, V) -> Option<V>,
    {
        self.difference_with_key(other, |_, a, b| f(a, b))
    }

    /// Construct the difference between two maps by using a function
    /// to decide what to do if a key occurs in both. The function
    /// receives the key as well as both values.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.difference_with_key(
    ///     map2,
    ///     |key, left, right| Some(left + right)
    /// ));
    /// # }
    /// ```
    #[must_use]
    pub fn difference_with_key<F>(mut self, other: Self, mut f: F) -> Self
    where
        F: FnMut(&K, V, V) -> Option<V>,
    {
        let mut out = Self::default();
        for (key, right_value) in other {
            match self.remove(&key) {
                None => {
                    out.insert(key, right_value);
                }
                Some(left_value) => if let Some(final_value) = f(&key, left_value, right_value) {
                    out.insert(key, final_value);
                },
            }
        }
        out.union(self)
    }

    /// Construct the intersection of two maps, keeping the values
    /// from the current map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{2 => 3, 3 => 4};
    /// let expected = ordmap!{2 => 2};
    /// assert_eq!(expected, map1.intersection(map2));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        self.intersection_with_key(other, |_, v, _| v)
    }

    /// Construct the intersection of two maps, calling a function
    /// with both values for each key and using the result as the
    /// value for the key.
    ///
    /// Time: O(n log n)
    #[inline]
    #[must_use]
    pub fn intersection_with<B, C, F>(self, other: OrdMap<K, B>, mut f: F) -> OrdMap<K, C>
    where
        B: Clone,
        C: Clone,
        F: FnMut(V, B) -> C,
    {
        self.intersection_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the intersection of two maps, calling a function
    /// with the key and both values for each key and using the result
    /// as the value for the key.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{2 => 3, 3 => 4};
    /// let expected = ordmap!{2 => 5};
    /// assert_eq!(expected, map1.intersection_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// # }
    /// ```
    #[must_use]
    pub fn intersection_with_key<B, C, F>(mut self, other: OrdMap<K, B>, mut f: F) -> OrdMap<K, C>
    where
        B: Clone,
        C: Clone,
        F: FnMut(&K, V, B) -> C,
    {
        let mut out = OrdMap::<K, C>::default();
        for (key, right_value) in other {
            match self.remove(&key) {
                None => (),
                Some(left_value) => {
                    let result = f(&key, left_value, right_value);
                    out.insert(key, result);
                }
            }
        }
        out
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the
    /// same values.
    ///
    /// Use the provided function to decide whether values are equal.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_submap_by<B, RM, F>(&self, other: RM, mut cmp: F) -> bool
    where
        B: Clone,
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<OrdMap<K, B>>,
    {
        self.iter()
            .all(|(k, v)| other.borrow().get(k).map(|ov| cmp(v, ov)).unwrap_or(false))
    }

    /// Test whether a map is a proper submap of another map, meaning
    /// that all keys in our map must also be in the other map, with
    /// the same values. To be a proper submap, ours must also contain
    /// fewer keys than the other map.
    ///
    /// Use the provided function to decide whether values are equal.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_proper_submap_by<B, RM, F>(&self, other: RM, cmp: F) -> bool
    where
        B: Clone,
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<OrdMap<K, B>>,
    {
        self.len() != other.borrow().len() && self.is_submap_by(other, cmp)
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the
    /// same values.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_submap(map2));
    /// # }
    /// ```
    #[must_use]
    pub fn is_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_submap_by(other.borrow(), PartialEq::eq)
    }

    /// Test whether a map is a proper submap of another map, meaning
    /// that all keys in our map must also be in the other map, with
    /// the same values. To be a proper submap, ours must also contain
    /// fewer keys than the other map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # fn main() {
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_proper_submap(map2));
    ///
    /// let map3 = ordmap!{1 => 1, 2 => 2};
    /// let map4 = ordmap!{1 => 1, 2 => 2};
    /// assert!(!map3.is_proper_submap(map4));
    /// # }
    /// ```
    #[must_use]
    pub fn is_proper_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_proper_submap_by(other.borrow(), PartialEq::eq)
    }

    /// Split a map into two, with the left hand map containing keys
    /// which are smaller than `split`, and the right hand map
    /// containing keys which are larger than `split`.
    ///
    /// The `split` mapping is discarded.
    #[must_use]
    pub fn split<BK>(&self, split: &BK) -> (Self, Self)
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let (l, _, r) = self.split_lookup(split);
        (l, r)
    }

    /// Split a map into two, with the left hand map containing keys
    /// which are smaller than `split`, and the right hand map
    /// containing keys which are larger than `split`.
    ///
    /// Returns both the two maps and the value of `split`.
    #[must_use]
    pub fn split_lookup<BK>(&self, split: &BK) -> (Self, Option<V>, Self)
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        // TODO this is atrociously slow, got to be a better way
        self.iter()
            .fold((ordmap![], None, ordmap![]), |(l, m, r), (k, v)| {
                match k.borrow().cmp(split) {
                    Ordering::Less => (l.update(k.clone(), v.clone()), m, r),
                    Ordering::Equal => (l, Some(v.clone()), r),
                    Ordering::Greater => (l, m, r.update(k.clone(), v.clone())),
                }
            })
    }

    /// Construct a map with only the `n` smallest keys from a given
    /// map.
    #[must_use]
    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).cloned().collect()
    }

    /// Construct a map with the `n` smallest keys removed from a
    /// given map.
    #[must_use]
    pub fn skip(&self, n: usize) -> Self {
        self.iter().skip(n).cloned().collect()
    }

    /// Remove the smallest key from a map, and return its value as
    /// well as the updated map.
    #[must_use]
    pub fn without_min(&self) -> (Option<V>, Self) {
        let (pop, next) = self.without_min_with_key();
        (pop.map(|(_, v)| v), next)
    }

    /// Remove the smallest key from a map, and return that key, its
    /// value as well as the updated map.
    #[must_use]
    pub fn without_min_with_key(&self) -> (Option<(K, V)>, Self) {
        match self.get_min() {
            None => (None, self.clone()),
            Some((k, _)) => {
                let (key, value, next) = self.extract_with_key(k).unwrap();
                (Some((key, value)), next)
            }
        }
    }

    /// Remove the largest key from a map, and return its value as
    /// well as the updated map.
    #[must_use]
    pub fn without_max(&self) -> (Option<V>, Self) {
        let (pop, next) = self.without_max_with_key();
        (pop.map(|(_, v)| v), next)
    }

    /// Remove the largest key from a map, and return that key, its
    /// value as well as the updated map.
    #[must_use]
    pub fn without_max_with_key(&self) -> (Option<(K, V)>, Self) {
        match self.get_max() {
            None => (None, self.clone()),
            Some((k, _)) => {
                let (key, value, next) = self.extract_with_key(k).unwrap();
                (Some((key, value)), next)
            }
        }
    }

    #[must_use]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        if self.contains_key(&key) {
            Entry::Occupied(OccupiedEntry { map: self, key })
        } else {
            Entry::Vacant(VacantEntry { map: self, key })
        }
    }
}

// Entries

pub enum Entry<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(|| default)
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(Default::default)
    }

    #[must_use]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match &mut self {
            Entry::Occupied(ref mut entry) => f(entry.get_mut()),
            Entry::Vacant(_) => (),
        }
        self
    }
}

pub struct OccupiedEntry<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    map: &'a mut OrdMap<K, V>,
    key: K,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn remove_entry(self) -> (K, V) {
        self.map
            .remove_with_key(&self.key)
            .expect("ordmap::OccupiedEntry::remove_entry: key has vanished!")
    }

    #[must_use]
    pub fn get(&self) -> &V {
        self.map.get(&self.key).unwrap()
    }

    #[must_use]
    pub fn get_mut(&mut self) -> &mut V {
        self.map.get_mut(&self.key).unwrap()
    }

    #[must_use]
    pub fn into_mut(self) -> &'a mut V {
        self.map.get_mut(&self.key).unwrap()
    }

    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

pub struct VacantEntry<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    map: &'a mut OrdMap<K, V>,
    key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[must_use]
    pub fn into_key(self) -> K {
        self.key
    }

    pub fn insert(self, value: V) -> &'a mut V {
        self.map.insert(self.key.clone(), value);
        // TODO insert_mut ought to return this reference
        self.map.get_mut(&self.key).unwrap()
    }
}

// Core traits

impl<K, V> Clone for OrdMap<K, V> {
    fn clone(&self) -> Self {
        OrdMap {
            root: self.root.clone(),
        }
    }
}

#[cfg(not(has_specialisation))]
impl<K, V> PartialEq for OrdMap<K, V>
where
    K: Ord + PartialEq + Clone,
    V: PartialEq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.diff(other).next().is_none()
    }
}

#[cfg(has_specialisation)]
impl<K, V> PartialEq for OrdMap<K, V>
where
    K: Ord + Clone + PartialEq,
    V: Clone + PartialEq,
{
    default fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.diff(other).next().is_none()
    }
}

#[cfg(has_specialisation)]
impl<K, V> PartialEq for OrdMap<K, V>
where
    K: Ord + Eq + Clone,
    V: Eq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        Ref::ptr_eq(&self.root, &other.root)
            || (self.len() == other.len() && self.diff(other).next().is_none())
    }
}

impl<K: Ord + Clone + Eq, V: Clone + Eq> Eq for OrdMap<K, V> {}

impl<K, V> PartialOrd for OrdMap<K, V>
where
    K: Ord + Clone,
    V: PartialOrd + Clone,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, V> Ord for OrdMap<K, V>
where
    K: Ord + Clone,
    V: Ord + Clone,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, V> Hash for OrdMap<K, V>
where
    K: Ord + Clone + Hash,
    V: Clone + Hash,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<K, V> Default for OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, K, V> Add for &'a OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Output = OrdMap<K, V>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<K, V> Add for OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Output = OrdMap<K, V>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<K, V> Sum for OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::default(), |a, b| a + b)
    }
}

impl<K, V, RK, RV> Extend<(RK, RV)> for OrdMap<K, V>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (RK, RV)>,
    {
        for (key, value) in iter {
            self.insert(From::from(key), From::from(value));
        }
    }
}

impl<'a, BK, K, V> Index<&'a BK> for OrdMap<K, V>
where
    BK: Ord + ?Sized,
    K: Ord + Clone + Borrow<BK>,
    V: Clone,
{
    type Output = V;

    fn index(&self, key: &BK) -> &Self::Output {
        match self.root.lookup(key) {
            None => panic!("OrdMap::index: invalid key"),
            Some(&(_, ref value)) => value,
        }
    }
}

impl<'a, BK, K, V> IndexMut<&'a BK> for OrdMap<K, V>
where
    BK: Ord + ?Sized,
    K: Ord + Clone + Borrow<BK>,
    V: Clone,
{
    fn index_mut(&mut self, key: &BK) -> &mut Self::Output {
        let root = Ref::make_mut(&mut self.root);
        match root.lookup_mut(key) {
            None => panic!("OrdMap::index: invalid key"),
            Some(&mut (_, ref mut value)) => value,
        }
    }
}

impl<K, V> Debug for OrdMap<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone + Debug,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        let mut d = f.debug_map();
        for (k, v) in self {
            d.entry(k, v);
        }
        d.finish()
    }
}

// Iterators

pub struct Keys<'a, K: 'a, V: 'a> {
    it: Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        match self.it.next() {
            None => None,
            Some((k, _)) => Some(k),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            None => None,
            Some((k, _)) => Some(k),
        }
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{}

pub struct Values<'a, K: 'a, V: 'a> {
    it: Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Values<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        match self.it.next() {
            None => None,
            Some((_, v)) => Some(v),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            None => None,
            Some((_, v)) => Some(v),
        }
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
{}

impl<K, V, RK, RV> FromIterator<(RK, RV)> for OrdMap<K, V>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = (RK, RV)>,
    {
        let mut m = OrdMap::default();
        for (k, v) in i {
            m.insert(From::from(k), From::from(v));
        }
        m
    }
}

impl<'a, K, V> IntoIterator for &'a OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Item = &'a (K, V);
    type IntoIter = Iter<'a, (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> IntoIterator for OrdMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Item = (K, V);
    type IntoIter = ConsumingIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter::new(&self.root)
    }
}

// Conversions

impl<K, V> AsRef<OrdMap<K, V>> for OrdMap<K, V> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<'m, 'k, 'v, K, V, OK, OV> From<&'m OrdMap<&'k K, &'v V>> for OrdMap<OK, OV>
where
    K: Ord + ToOwned<Owned = OK> + ?Sized,
    V: ToOwned<Owned = OV> + ?Sized,
    OK: Ord + Clone + Borrow<K>,
    OV: Clone + Borrow<V>,
{
    fn from(m: &OrdMap<&K, &V>) -> Self {
        m.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }
}

impl<'a, K, V, RK, RV, OK, OV> From<&'a [(RK, RV)]> for OrdMap<K, V>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
{
    fn from(m: &'a [(RK, RV)]) -> OrdMap<K, V> {
        m.into_iter()
            .map(|&(ref k, ref v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K, V, RK, RV> From<Vec<(RK, RV)>> for OrdMap<K, V>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
{
    fn from(m: Vec<(RK, RV)>) -> OrdMap<K, V> {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord, V, RK, RV, OK, OV> From<&'a Vec<(RK, RV)>> for OrdMap<K, V>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
{
    fn from(m: &'a Vec<(RK, RV)>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|&(ref k, ref v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K: Ord, V, RK: Eq + Hash, RV> From<collections::HashMap<RK, RV>> for OrdMap<K, V>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
{
    fn from(m: collections::HashMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter().collect()
    }
}

impl<'a, K, V, OK, OV, RK, RV> From<&'a collections::HashMap<RK, RV>> for OrdMap<K, V>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: Hash + Eq + ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
{
    fn from(m: &'a collections::HashMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K: Ord, V, RK, RV> From<collections::BTreeMap<RK, RV>> for OrdMap<K, V>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
{
    fn from(m: collections::BTreeMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord, V, RK, RV, OK, OV> From<&'a collections::BTreeMap<RK, RV>> for OrdMap<K, V>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: Ord + ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
{
    fn from(m: &'a collections::BTreeMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K: Ord + Hash + Eq + Clone, V: Clone, S: BuildHasher> From<HashMap<K, V, S>> for OrdMap<K, V> {
    fn from(m: HashMap<K, V, S>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord + Hash + Eq + Clone, V: Clone, S: BuildHasher> From<&'a HashMap<K, V, S>>
    for OrdMap<K, V>
{
    fn from(m: &'a HashMap<K, V, S>) -> Self {
        m.into_iter().cloned().collect()
    }
}

// QuickCheck

#[cfg(all(threadsafe, any(test, feature = "quickcheck")))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(threadsafe, any(test, feature = "quickcheck")))]
impl<K: Ord + Clone + Arbitrary + Sync, V: Clone + Arbitrary + Sync> Arbitrary for OrdMap<K, V> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        OrdMap::from_iter(Vec::<(K, V)>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for a map of a given size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_works(ref m in map(0..9999, ".*", 10..100)) {
    ///         assert!(m.len() < 100);
    ///         assert!(m.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn ord_map<K: Strategy + 'static, V: Strategy + 'static>(
        key: K,
        value: V,
        size: Range<usize>,
    ) -> BoxedStrategy<OrdMap<<K::Tree as ValueTree>::Value, <V::Tree as ValueTree>::Value>>
    where
        <K::Tree as ValueTree>::Value: Ord + Clone,
        <V::Tree as ValueTree>::Value: Clone,
    {
        ::proptest::collection::vec((key, value), size.clone())
            .prop_map(OrdMap::from)
            .prop_filter("OrdMap minimum size".to_owned(), move |m| {
                m.len() >= size.start
            }).boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;
    use nodes::btree::DiffItem;
    use proptest::collection;
    use proptest::num::{i16, usize};
    use test::is_sorted;

    #[test]
    fn iterates_in_order() {
        let map = ordmap!{
            2 => 22,
            1 => 11,
            3 => 33,
            8 => 88,
            9 => 99,
            4 => 44,
            5 => 55,
            7 => 77,
            6 => 66
        };
        let mut it = map.iter();
        assert_eq!(it.next(), Some(&(1, 11)));
        assert_eq!(it.next(), Some(&(2, 22)));
        assert_eq!(it.next(), Some(&(3, 33)));
        assert_eq!(it.next(), Some(&(4, 44)));
        assert_eq!(it.next(), Some(&(5, 55)));
        assert_eq!(it.next(), Some(&(6, 66)));
        assert_eq!(it.next(), Some(&(7, 77)));
        assert_eq!(it.next(), Some(&(8, 88)));
        assert_eq!(it.next(), Some(&(9, 99)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_iter() {
        let map = ordmap!{
            2 => 22,
            1 => 11,
            3 => 33,
            8 => 88,
            9 => 99,
            4 => 44,
            5 => 55,
            7 => 77,
            6 => 66
        };
        let mut vec = vec![];
        for (k, v) in map {
            assert_eq!(k * 11, v);
            vec.push(k)
        }
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn deletes_correctly() {
        let map = ordmap!{
            2 => 22,
            1 => 11,
            3 => 33,
            8 => 88,
            9 => 99,
            4 => 44,
            5 => 55,
            7 => 77,
            6 => 66
        };
        assert_eq!(map.extract(&11), None);
        let (popped, less) = map.extract(&5).unwrap();
        assert_eq!(popped, 55);
        let mut it = less.iter();
        assert_eq!(it.next(), Some(&(1, 11)));
        assert_eq!(it.next(), Some(&(2, 22)));
        assert_eq!(it.next(), Some(&(3, 33)));
        assert_eq!(it.next(), Some(&(4, 44)));
        assert_eq!(it.next(), Some(&(6, 66)));
        assert_eq!(it.next(), Some(&(7, 77)));
        assert_eq!(it.next(), Some(&(8, 88)));
        assert_eq!(it.next(), Some(&(9, 99)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn debug_output() {
        assert_eq!(
            format!("{:?}", ordmap!{ 3 => 4, 5 => 6, 1 => 2 }),
            "{1: 2, 3: 4, 5: 6}"
        );
    }

    #[test]
    fn equality2() {
        let v1 = "1".to_string();
        let v2 = "1".to_string();
        assert_eq!(v1, v2);
        let p1 = Vec::<String>::new();
        let p2 = Vec::<String>::new();
        assert_eq!(p1, p2);
        let c1 = OrdMap::singleton(v1, p1);
        let c2 = OrdMap::singleton(v2, p2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn insert_remove_single_mut() {
        let mut m = OrdMap::new();
        m.insert(0, 0);
        assert_eq!(OrdMap::singleton(0, 0), m);
        m.remove(&0);
        assert_eq!(OrdMap::new(), m);
    }

    #[test]
    fn double_ended_iterator_1() {
        let m = ordmap!{1 => 1, 2 => 2, 3 => 3, 4 => 4};
        let mut it = m.iter();
        assert_eq!(Some(&(1, 1)), it.next());
        assert_eq!(Some(&(4, 4)), it.next_back());
        assert_eq!(Some(&(2, 2)), it.next());
        assert_eq!(Some(&(3, 3)), it.next_back());
        assert_eq!(None, it.next());
    }

    #[test]
    fn double_ended_iterator_2() {
        let m = ordmap!{1 => 1, 2 => 2, 3 => 3, 4 => 4};
        let mut it = m.iter();
        assert_eq!(Some(&(1, 1)), it.next());
        assert_eq!(Some(&(4, 4)), it.next_back());
        assert_eq!(Some(&(2, 2)), it.next());
        assert_eq!(Some(&(3, 3)), it.next_back());
        assert_eq!(None, it.next_back());
    }

    #[test]
    fn safe_mutation() {
        let v1 = OrdMap::from_iter((0..131072).map(|i| (i, i)));
        let mut v2 = v1.clone();
        v2.insert(131000, 23);
        assert_eq!(Some(&23), v2.get(&131000));
        assert_eq!(Some(&131000), v1.get(&131000));
    }

    #[test]
    fn index_operator() {
        let mut map = ordmap!{1 => 2, 3 => 4, 5 => 6};
        assert_eq!(4, map[&3]);
        map[&3] = 8;
        assert_eq!(ordmap!{1 => 2, 3 => 8, 5 => 6}, map);
    }

    #[test]
    fn entry_api() {
        let mut map = ordmap!{"bar" => 5};
        map.entry(&"foo").and_modify(|v| *v += 5).or_insert(1);
        assert_eq!(1, map[&"foo"]);
        map.entry(&"foo").and_modify(|v| *v += 5).or_insert(1);
        assert_eq!(6, map[&"foo"]);
        map.entry(&"bar").and_modify(|v| *v += 5).or_insert(1);
        assert_eq!(10, map[&"bar"]);
        assert_eq!(
            10,
            match map.entry(&"bar") {
                Entry::Occupied(entry) => entry.remove(),
                _ => panic!(),
            }
        );
        assert!(!map.contains_key(&"bar"));
    }

    #[test]
    fn match_string_keys_with_string_slices() {
        let mut map: OrdMap<String, i32> =
            From::from(&ordmap!{ "foo" => &1, "bar" => &2, "baz" => &3 });
        assert_eq!(Some(&1), map.get("foo"));
        map = map.without("foo");
        assert_eq!(Some(3), map.remove("baz"));
        map["bar"] = 8;
        assert_eq!(8, map["bar"]);
    }

    quickcheck! {
        fn length(input: Vec<i32>) -> bool {
            let mut vec = input;
            vec.sort();
            vec.dedup();
            let map: OrdMap<i32, i32> = OrdMap::from_iter(vec.iter().cloned().map(|i| (i, i)));
            vec.len() == map.len()
        }

        fn order(vec: Vec<(i32, i32)>) -> bool {
            let map: OrdMap<i32, i32> = OrdMap::from_iter(vec.into_iter());
            is_sorted(map.keys())
        }

        fn overwrite_values(vec: Vec<(usize, usize)>, index_rand: usize, new_val: usize) -> bool {
            if vec.is_empty() {
                return true
            }
            let index = vec[index_rand % vec.len()].0;
            let map1 = OrdMap::from_iter(vec.clone());
            let map2 = map1.update(index, new_val);
            map2.iter().all(|(k, v)| if *k == index {
                *v == new_val
            } else {
                match map1.get(&k) {
                    None => false,
                    Some(other_v) => v == other_v
                }
            })
        }

        fn delete_values(vec: Vec<(usize, usize)>, index_rand: usize) -> bool {
            if vec.is_empty() {
                return true
            }
            let index = vec[index_rand % vec.len()].0;
            let map1: OrdMap<usize, usize> = OrdMap::from_iter(vec.clone());
            let map2 = map1.without(&index);
            map2.keys().all(|k| *k != index) && map1.len() == map2.len() + 1
        }

        fn insert_and_delete_values(
            input_unbounded: Vec<(usize, usize)>, ops: Vec<(bool, usize, usize)>
        ) -> bool {
            let input: Vec<(usize, usize)> =
                input_unbounded.into_iter().map(|(k, v)| (k % 64, v % 64)).collect();
            let mut map = OrdMap::from(input.clone());
            let mut tree: collections::BTreeMap<usize, usize> =
                input.into_iter().collect();
            for (ins, key, val) in ops {
                if ins {
                    tree.insert(key, val);
                    map = map.update(key, val)
                } else {
                    tree.remove(&key);
                    map = map.without(&key)
                }
            }
            map.iter().map(|(k, v)| (*k, *v)).eq(tree.iter().map(|(k, v)| (*k, *v)))
        }

        fn diff_added_values(a: Vec<(usize, usize)>, b: Vec<(usize, usize)>) -> bool {
            let a: OrdMap<usize, usize> = OrdMap::from(a);
            let b: OrdMap<usize, usize> = OrdMap::from(b);
            let ab = a.clone().union(b.clone());
            a.diff(&ab).eq(b.iter().filter(|&(ref k, _)| !a.contains_key(k)).map(DiffItem::Add))
        }

        // fn diff_updated_values(a: Vec<(usize, usize)>, b: Vec<(usize, usize)>) -> bool {
        //     let a: OrdMap<usize, usize> = OrdMap::from(a);
        //     let b: OrdMap<usize, usize> = OrdMap::from(b);
        //     let ab: OrdMap<usize, usize> = a.union(&b);
        //     let ba: OrdMap<usize, usize> = ab.union_with(&b, |_, b| *b);
        //     ab.diff(&ba).eq(b.iter().filter(|&(ref k, ref v)| ab.get(k) != Some(&v))
        //                    .map(|(k, v)| DiffItem::Update {
        //                        old: &(*k, *(ab.get(&k).unwrap())),
        //                        new: &(*k, *v)
        //                    }))
        // }

        fn diff_removed_values(a: Vec<(usize, usize)>, b: Vec<(usize, usize)>) -> bool {
            let a: OrdMap<usize, usize> = OrdMap::from(a);
            let b: OrdMap<usize, usize> = OrdMap::from(b);
            let ab = a.clone().union(b.clone());
            ab.diff(&a).eq(b.iter().filter(|&(ref k, _)| !a.contains_key(k)).map(DiffItem::Remove))
        }

        // fn diff_all_values(a: Vec<(usize, usize)>, b: Vec<(usize, usize)>) -> bool {
        //     let a: OrdMap<usize, usize> = OrdMap::from(a);
        //     let b: OrdMap<usize, usize> = OrdMap::from(b);
        //     a.diff(&b).eq(b.union(&a).iter().filter_map(|(k, v)| {
        //         if a.contains_key(&k) {
        //             if b.contains_key(&k) {
        //                 let old = a.get(&k).unwrap();
        //                 if old != v	{
        //                     Some(DiffItem::Update {
        //                         old: &(*k, *old),
        //                         new: &(*k, *v),
        //                     })
        //                 } else {
        //                     None
        //                 }
        //             } else {
        //                 Some(DiffItem::Remove(&(*k, *v)))
        //             }
        //         } else {
        //             Some(DiffItem::Add(&(*k, *v)))
        //         }
        //     }))
        // }
    }

    proptest! {
        #[test]
        fn proptest_works(ref m in ord_map(0..9999, ".*", 10..100)) {
            assert!(m.len() < 100);
            assert!(m.len() >= 10);
        }

        #[test]
        fn insert_and_length(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map: OrdMap<i16, i16> = OrdMap::new();
            for (k, v) in m.iter() {
                map = map.update(*k, *v)
            }
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn from_iterator(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn iterate_over(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.iter().count());
        }

        #[test]
        fn equality(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map1: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let map2: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(map1, map2);
        }

        #[test]
        fn lookup(ref m in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for (k, v) in m {
                assert_eq!(Some(*v), map.get(k).cloned());
            }
        }

        #[test]
        fn remove(ref m in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).cloned());
                map = map.without(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn insert_mut(ref m in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut mut_map = OrdMap::new();
            let mut map = OrdMap::new();
            for (k, v) in m.iter() {
                map = map.update(*k, *v);
                mut_map.insert(*k, *v);
            }
            assert_eq!(map, mut_map);
        }

        #[test]
        fn remove_mut(ref orig in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map = orig.clone();
            for key in orig.keys() {
                let len = map.len();
                assert_eq!(orig.get(key), map.get(key));
                assert_eq!(orig.get(key).cloned(), map.remove(key));
                assert_eq!(None, map.get(key));
                assert_eq!(len - 1, map.len());
            }
        }

        #[test]
        fn remove_alien(ref orig in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map = OrdMap::<i16, i16>::from(orig.clone());
            for key in orig.keys() {
                let len = map.len();
                assert_eq!(orig.get(key), map.get(key));
                assert_eq!(orig.get(key).cloned(), map.remove(key));
                assert_eq!(None, map.get(key));
                assert_eq!(len - 1, map.len());
            }
        }

        #[test]
        fn delete_and_reinsert(
            ref input in collection::hash_map(i16::ANY, i16::ANY, 1..1000),
            index_rand in usize::ANY
        ) {
            let index = *input.keys().nth(index_rand % input.len()).unwrap();
            let map1 = OrdMap::from_iter(input.clone());
            let (val, map2): (i16, _) = map1.extract(&index).unwrap();
            let map3 = map2.update(index, val);
            for key in map2.keys() {
                assert!(*key != index);
            }
            assert_eq!(map1.len(), map2.len() + 1);
            assert_eq!(map1, map3);
        }

        #[test]
        fn exact_size_iterator(ref m in ord_map(i16::ANY, i16::ANY, 1..1000)) {
            let mut should_be = m.len();
            let mut it = m.iter();
            loop {
                assert_eq!(should_be, it.len());
                match it.next() {
                    None => break,
                    Some(_) => should_be -= 1,
                }
            }
            assert_eq!(0, it.len());
        }
    }
}
