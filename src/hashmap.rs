// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An unordered map.
//!
//! An immutable hash map using [hash array mapped tries] [1].
//!
//! Most operations on this map are O(log<sub>x</sub> n) for a
//! suitably high *x* that it should be nearly O(1) for most maps.
//! Because of this, it's a great choice for a generic map as long as
//! you don't mind that keys will need to implement
//! [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
//!
//! Map entries will have a predictable order based on the hasher
//! being used. Unless otherwise specified, all maps will use the
//! default [`RandomState`][std::collections::hash_map::RandomState]
//! hasher, which will produce consistent hashes for the duration of
//! its lifetime, but not between restarts of your program.
//!
//! [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
//! [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
//! [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
//! [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, FusedIterator, Sum};
use std::mem;
use std::ops::{Add, Index, IndexMut};

use nodes::bitmap::{hash_key, HashBits};
use nodes::hamt::{Drain as NodeDrain, HashValue, Iter as NodeIter, IterMut as NodeIterMut, Node};
use util::Ref;

/// Construct a hash map from a sequence of key/value pairs.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::hashmap::HashMap;
/// # fn main() {
/// assert_eq!(
///   hashmap!{
///     1 => 11,
///     2 => 22,
///     3 => 33
///   },
///   HashMap::from(vec![(1, 11), (2, 22), (3, 33)])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! hashmap {
    () => { $crate::hashmap::HashMap::new() };

    ( $( $key:expr => $value:expr ),* ) => {{
        let mut map = $crate::hashmap::HashMap::new();
        $({
            map.insert($key, $value);
        })*;
        map
    }};

    ( $( $key:expr => $value:expr ,)* ) => {{
        let mut map = $crate::hashmap::HashMap::new();
        $({
            map.insert($key, $value);
        })*;
        map
    }};
}

/// An unordered map.
///
/// An immutable hash map using [hash array mapped tries] [1].
///
/// Most operations on this map are O(log<sub>x</sub> n) for a
/// suitably high *x* that it should be nearly O(1) for most maps.
/// Because of this, it's a great choice for a generic map as long as
/// you don't mind that keys will need to implement
/// [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
///
/// Map entries will have a predictable order based on the hasher
/// being used. Unless otherwise specified, all maps will share an
/// instance of the default
/// [`RandomState`][std::collections::hash_map::RandomState] hasher,
/// which will produce consistent hashes for the duration of its
/// lifetime, but not between restarts of your program.
///
/// [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
/// [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
/// [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
/// [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html

pub struct HashMap<K, V, S = RandomState>
where
    K: Clone,
    V: Clone,
{
    size: usize,
    root: Ref<Node<(K, V)>>,
    hasher: Ref<S>,
}

impl<K, V> HashValue for (K, V)
where
    K: Eq + Clone,
    V: Clone,
{
    type Key = K;

    fn extract_key(&self) -> &Self::Key {
        &self.0
    }

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<K, V> HashMap<K, V, RandomState>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Construct an empty hash map.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a hash map with a single mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map = HashMap::singleton(123, "onetwothree");
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"onetwothree")
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn singleton(k: K, v: V) -> HashMap<K, V> {
        HashMap::new().update(k, v)
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Clone,
    V: Clone,
{
    /// Test whether a hash map is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// assert!(
    ///   !hashmap!{1 => 2}.is_empty()
    /// );
    /// assert!(
    ///   HashMap::<i32, i32>::new().is_empty()
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size of a hash map.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// assert_eq!(3, hashmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.len());
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    fn test_eq(&self, other: &Self) -> bool
    where
        V: PartialEq,
    {
        if self.len() != other.len() {
            return false;
        }
        let mut seen = collections::HashSet::new();
        for (key, value) in self.iter() {
            if Some(value) != other.get(&key) {
                return false;
            }
            seen.insert(key);
        }
        for key in other.keys() {
            if !seen.contains(&key) {
                return false;
            }
        }
        true
    }

    /// Construct an empty hash map using the provided hasher.
    #[inline]
    pub fn with_hasher<RS>(hasher: RS) -> Self
    where
        Ref<S>: From<RS>,
    {
        HashMap {
            size: 0,
            root: Ref::new(Node::new()),
            hasher: From::from(hasher),
        }
    }

    /// Get a reference to the map's [`BuildHasher`][BuildHasher].
    ///
    /// [BuildHasher]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    pub fn hasher(&self) -> &Ref<S> {
        &self.hasher
    }

    /// Construct an empty hash map using the same hasher as the
    /// current hash map.
    #[inline]
    pub fn new_from<K1, V1>(&self) -> HashMap<K1, V1, S>
    where
        K1: Hash + Eq + Clone,
        V1: Clone,
    {
        HashMap {
            size: 0,
            root: Ref::new(Node::new()),
            hasher: self.hasher.clone(),
        }
    }

    /// Get an iterator over the key/value pairs of a hash map.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            it: NodeIter::new(&self.root, self.size),
        }
    }

    /// Get a mutable iterator over the values of a hash map.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let root = Ref::make_mut(&mut self.root);
        IterMut {
            it: NodeIterMut::new(root, self.size),
        }
    }

    /// Get an iterator over a hash map's keys.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            it: NodeIter::new(&self.root, self.size),
        }
    }

    /// Get an iterator over a hash map's values.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    pub fn values(&self) -> Values<K, V> {
        Values {
            it: NodeIter::new(&self.root, self.size),
        }
    }

    /// Get the value for a key from a hash map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"lol")
    /// );
    /// # }
    /// ```
    pub fn get<BK>(&self, key: &BK) -> Option<&V>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        self.root
            .get(hash_key(&*self.hasher, key), 0, key)
            .map(|&(_, ref v)| v)
    }

    /// Get a mutable reference to the value for a key from a hash
    /// map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"lol")
    /// );
    /// # }
    /// ```
    pub fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut V>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        let root = Ref::make_mut(&mut self.root);
        match root.get_mut(hash_key(&*self.hasher, key), 0, key) {
            None => None,
            Some(&mut (_, ref mut value)) => Some(value),
        }
    }

    /// Test for the presence of a key in a hash map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map = hashmap!{123 => "lol"};
    /// assert!(
    ///   map.contains_key(&123)
    /// );
    /// assert!(
    ///   !map.contains_key(&321)
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn contains_key<BK>(&self, k: &BK) -> bool
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        self.get(k).is_some()
    }

    /// Insert a key/value mapping into a map.
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let mut map = hashmap!{};
    /// map.insert(123, "123");
    /// map.insert(456, "456");
    /// assert_eq!(
    ///   map,
    ///   hashmap!{123 => "123", 456 => "456"}
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let hash = hash_key(&*self.hasher, &k);
        let root = Ref::make_mut(&mut self.root);
        let result = root.insert(hash, 0, (k, v));
        if result.is_none() {
            self.size += 1;
        }
        result.map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let mut map = hashmap!{123 => "123", 456 => "456"};
    /// assert_eq!(Some("123"), map.remove(&123));
    /// assert_eq!(Some("456"), map.remove(&456));
    /// assert_eq!(None, map.remove(&789));
    /// assert!(map.is_empty());
    /// # }
    /// ```
    pub fn remove<BK>(&mut self, k: &BK) -> Option<V>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        self.remove_with_key(k).map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let mut map = hashmap!{123 => "123", 456 => "456"};
    /// assert_eq!(Some((123, "123")), map.remove_with_key(&123));
    /// assert_eq!(Some((456, "456")), map.remove_with_key(&456));
    /// assert_eq!(None, map.remove_with_key(&789));
    /// assert!(map.is_empty());
    /// # }
    /// ```
    pub fn remove_with_key<BK>(&mut self, k: &BK) -> Option<(K, V)>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        let root = Ref::make_mut(&mut self.root);
        let result = root.remove(hash_key(&*self.hasher, k), 0, k);
        if result.is_some() {
            self.size -= 1;
        }
        result
    }

    /// Get the [`Entry`][Entry] for a key in the map for in-place manipulation.
    ///
    /// Time: O(log n)
    ///
    /// [Entry]: enum.Entry.html
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        let hash = hash_key(&*self.hasher, &key);
        if self.root.get(hash, 0, &key).is_some() {
            Entry::Occupied(OccupiedEntry {
                map: self,
                hash,
                key,
            })
        } else {
            Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
            })
        }
    }

    /// Construct a new hash map by inserting a key/value mapping into a map.
    ///
    /// If the map already has a mapping for the given key, the previous value
    /// is overwritten.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map = hashmap!{};
    /// assert_eq!(
    ///   map.update(123, "123"),
    ///   hashmap!{123 => "123"}
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn update(&self, k: K, v: V) -> Self {
        let mut out = self.clone();
        out.insert(k, v);
        out
    }

    /// Construct a new hash map by inserting a key/value mapping into
    /// a map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the old value and the new value,
    /// and insert the result as the new value.
    ///
    /// Time: O(log n)
    pub fn update_with<F>(&self, k: K, v: V, f: F) -> Self
    where
        F: FnOnce(V, V) -> V,
    {
        match self.extract_with_key(&k) {
            None => self.update(k, v),
            Some((_, v2, m)) => m.update(k, f(v2, v)),
        }
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the key, the old value and the new
    /// value, and insert the result as the new value.
    ///
    /// Time: O(log n)
    pub fn update_with_key<F>(&self, k: K, v: V, f: F) -> Self
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
    pub fn update_lookup_with_key<F>(&self, k: K, v: V, f: F) -> (Option<V>, Self)
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

    /// Construct a new map without the given key.
    ///
    /// Construct a map that's a copy of the current map, absent the
    /// mapping for `key` if it's present.
    ///
    /// Time: O(log n)
    pub fn without<BK>(&self, k: &BK) -> Self
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        match self.extract_with_key(k) {
            None => self.clone(),
            Some((_, _, map)) => map,
        }
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value as well as the updated map.
    ///
    /// Time: O(log n)
    pub fn extract<BK>(&self, k: &BK) -> Option<(V, Self)>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        self.extract_with_key(k).map(|(_, v, m)| (v, m))
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value as well as the updated list.
    ///
    /// Time: O(log n)
    pub fn extract_with_key<BK>(&self, k: &BK) -> Option<(K, V, Self)>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        let mut out = self.clone();
        out.remove_with_key(k).map(|(k, v)| (k, v, out))
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 3 => 3};
    /// let map2 = hashmap!{2 => 2, 3 => 4};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, map1.union(map2));
    /// # }
    /// ```
    #[inline]
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.union_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// # }
    /// ```
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 3 => 3};
    /// let map2 = hashmap!{2 => 2};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, HashMap::unions(vec![map1, map2]));
    /// # }
    /// ```
    pub fn unions<I>(i: I) -> Self
    where
        S: Default,
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
    pub fn unions_with<I, F>(i: I, f: F) -> Self
    where
        S: Default,
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
    pub fn unions_with_key<I, F>(i: I, f: F) -> Self
    where
        S: Default,
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2};
    /// assert_eq!(expected, map1.difference(map2));
    /// # }
    /// ```
    #[inline]
    pub fn difference(self, other: Self) -> Self {
        self.difference_with_key(other, |_, _, _| None)
    }

    /// Construct the difference between two maps by using a function
    /// to decide what to do if a key occurs in both.
    ///
    /// Time: O(n log n)
    #[inline]
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.difference_with_key(
    ///     map2,
    ///     |key, left, right| Some(left + right)
    /// ));
    /// # }
    /// ```
    pub fn difference_with_key<F>(mut self, other: Self, mut f: F) -> Self
    where
        F: FnMut(&K, V, V) -> Option<V>,
    {
        let mut out = self.new_from();
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{2 => 3, 3 => 4};
    /// let expected = hashmap!{2 => 2};
    /// assert_eq!(expected, map1.intersection(map2));
    /// # }
    /// ```
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        self.intersection_with_key(other, |_, v, _| v)
    }

    /// Construct the intersection of two maps, calling a function
    /// with both values for each key and using the result as the
    /// value for the key.
    ///
    /// Time: O(n log n)
    #[inline]
    pub fn intersection_with<B, C, F>(self, other: HashMap<K, B, S>, mut f: F) -> HashMap<K, C, S>
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{2 => 3, 3 => 4};
    /// let expected = hashmap!{2 => 5};
    /// assert_eq!(expected, map1.intersection_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// # }
    /// ```
    pub fn intersection_with_key<B, C, F>(
        mut self,
        other: HashMap<K, B, S>,
        mut f: F,
    ) -> HashMap<K, C, S>
    where
        B: Clone,
        C: Clone,
        F: FnMut(&K, V, B) -> C,
    {
        let mut out = self.new_from();
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
    pub fn is_submap_by<B, RM, F>(&self, other: RM, mut cmp: F) -> bool
    where
        B: Clone,
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<HashMap<K, B, S>>,
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
    pub fn is_proper_submap_by<B, RM, F>(&self, other: RM, cmp: F) -> bool
    where
        B: Clone,
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<HashMap<K, B, S>>,
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_submap(map2));
    /// # }
    /// ```
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
    /// # use im::hashmap::HashMap;
    /// # fn main() {
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_proper_submap(map2));
    ///
    /// let map3 = hashmap!{1 => 1, 2 => 2};
    /// let map4 = hashmap!{1 => 1, 2 => 2};
    /// assert!(!map3.is_proper_submap(map4));
    /// # }
    /// ```
    pub fn is_proper_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_proper_submap_by(other.borrow(), PartialEq::eq)
    }
}

// Entries

/// A handle for a key and its associated value.
///
/// ## Performance Note
///
/// When using an `Entry`, the key is only ever hashed once, when you
/// create the `Entry`. Operations on an `Entry` will never trigger a
/// rehash, where eg. a `contains_key(key)` followed by an
/// `insert(key, default_value)` (the equivalent of
/// `Entry::or_insert()`) would need to hash the key once for the
/// `contains_key` and again for the `insert`. The operations
/// generally perform similarly otherwise.
pub enum Entry<'a, K, V, S>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher,
{
    Occupied(OccupiedEntry<'a, K, V, S>),
    Vacant(VacantEntry<'a, K, V, S>),
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher,
{
    /// Insert the default value provided if there was no value
    /// already, and return a mutable reference to the value.
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(|| default)
    }

    /// Insert the default value from the provided function if there
    /// was no value already, and return a mutable reference to the
    /// value.
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Insert a default value if there was no value already, and
    /// return a mutable reference to the value.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(Default::default)
    }

    /// Get the key for this entry.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// Call the provided function to modify the value if the value
    /// exists.
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

/// An entry for a mapping that already exists in the map.
pub struct OccupiedEntry<'a, K, V, S>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher,
{
    map: &'a mut HashMap<K, V, S>,
    hash: HashBits,
    key: K,
}

impl<'a, K, V, S> OccupiedEntry<'a, K, V, S>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher,
{
    /// Get the key for this entry.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Remove this entry from the map and return the removed mapping.
    pub fn remove_entry(self) -> (K, V) {
        let root = Ref::make_mut(&mut self.map.root);
        let result = root.remove(self.hash, 0, &self.key);
        self.map.size -= 1;
        result.unwrap()
    }

    /// Get the current value.
    pub fn get(&self) -> &V {
        &self.map.root.get(self.hash, 0, &self.key).unwrap().1
    }

    /// Get a mutable reference to the current value.
    pub fn get_mut(&mut self) -> &mut V {
        let root = Ref::make_mut(&mut self.map.root);
        &mut root.get_mut(self.hash, 0, &self.key).unwrap().1
    }

    /// Convert this entry into a mutable reference.
    pub fn into_mut(self) -> &'a mut V {
        let root = Ref::make_mut(&mut self.map.root);
        &mut root.get_mut(self.hash, 0, &self.key).unwrap().1
    }

    /// Overwrite the current value.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Remove this entry from the map and return the removed value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

/// An entry for a mapping that does not already exist in the map.
pub struct VacantEntry<'a, K, V, S>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher,
{
    map: &'a mut HashMap<K, V, S>,
    hash: HashBits,
    key: K,
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher,
{
    /// Get the key for this entry.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Convert this entry into its key.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Insert a value into this entry.
    pub fn insert(self, value: V) -> &'a mut V {
        let root = Ref::make_mut(&mut self.map.root);
        if root
            .insert(self.hash, 0, (self.key.clone(), value))
            .is_none()
        {
            self.map.size += 1;
        }
        // TODO it's unfortunate that we need to look up the key again
        // here to get the mut ref.
        &mut root.get_mut(self.hash, 0, &self.key).unwrap().1
    }
}

// Core traits

impl<K, V, S> Clone for HashMap<K, V, S>
where
    K: Clone,
    V: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        HashMap {
            root: self.root.clone(),
            size: self.size,
            hasher: self.hasher.clone(),
        }
    }
}

#[cfg(not(has_specialisation))]
impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: PartialEq + Clone,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.test_eq(other)
    }
}

#[cfg(has_specialisation)]
impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: PartialEq + Clone,
    S: BuildHasher,
{
    default fn eq(&self, other: &Self) -> bool {
        self.test_eq(other)
    }
}

#[cfg(has_specialisation)]
impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Eq + Clone,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if Ref::ptr_eq(&self.root, &other.root) {
            return true;
        }
        self.test_eq(other)
    }
}

impl<K, V, S> Eq for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Eq + Clone,
    S: BuildHasher,
{}

impl<K, V, S> PartialOrd for HashMap<K, V, S>
where
    K: Hash + Eq + Clone + PartialOrd,
    V: PartialOrd + Clone,
    S: BuildHasher,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Ref::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().partial_cmp(other.iter());
        }
        let m1: ::std::collections::HashMap<K, V> = self.iter().cloned().collect();
        let m2: ::std::collections::HashMap<K, V> = other.iter().cloned().collect();
        m1.iter().partial_cmp(m2.iter())
    }
}

impl<K, V, S> Ord for HashMap<K, V, S>
where
    K: Hash + Eq + Ord + Clone,
    V: Ord + Clone,
    S: BuildHasher,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if Ref::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().cmp(other.iter());
        }
        let m1: ::std::collections::HashMap<K, V> = self.iter().cloned().collect();
        let m2: ::std::collections::HashMap<K, V> = other.iter().cloned().collect();
        m1.iter().cmp(m2.iter())
    }
}

impl<K, V, S> Hash for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Hash + Clone,
    S: BuildHasher,
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

impl<K, V, S> Default for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    #[inline]
    fn default() -> Self {
        HashMap {
            size: 0,
            root: Ref::new(Node::new()),
            hasher: Ref::<S>::default(),
        }
    }
}

impl<K, V, S> Add for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    type Output = HashMap<K, V, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, K, V, S> Add for &'a HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    type Output = HashMap<K, V, S>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<K, V, S> Sum for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::default(), |a, b| a + b)
    }
}

impl<K, V, S, RK, RV> Extend<(RK, RV)> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone + From<RK>,
    V: Clone + From<RV>,
    S: BuildHasher,
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

impl<'a, BK, K, V, S> Index<&'a BK> for HashMap<K, V, S>
where
    BK: Hash + Eq + ?Sized,
    K: Hash + Eq + Clone + Borrow<BK>,
    V: Clone,
    S: BuildHasher,
{
    type Output = V;

    fn index(&self, key: &BK) -> &Self::Output {
        match self.root.get(hash_key(&*self.hasher, key), 0, key) {
            None => panic!("HashMap::index: invalid key"),
            Some(&(_, ref value)) => value,
        }
    }
}

impl<'a, BK, K, V, S> IndexMut<&'a BK> for HashMap<K, V, S>
where
    BK: Hash + Eq + ?Sized,
    K: Hash + Eq + Clone + Borrow<BK>,
    V: Clone,
    S: BuildHasher,
{
    fn index_mut(&mut self, key: &BK) -> &mut Self::Output {
        let root = Ref::make_mut(&mut self.root);
        match root.get_mut(hash_key(&*self.hasher, key), 0, key) {
            None => panic!("HashMap::index_mut: invalid key"),
            Some(&mut (_, ref mut value)) => value,
        }
    }
}

#[cfg(not(has_specialisation))]
impl<K, V, S> Debug for HashMap<K, V, S>
where
    K: Hash + Eq + Clone + Debug,
    V: Debug + Clone,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        let mut d = f.debug_map();
        for (k, v) in self {
            d.entry(k, v);
        }
        d.finish()
    }
}

#[cfg(has_specialisation)]
impl<K, V, S> Debug for HashMap<K, V, S>
where
    K: Hash + Eq + Clone + Debug,
    V: Debug + Clone,
    S: BuildHasher,
{
    default fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        let mut d = f.debug_map();
        for (k, v) in self {
            d.entry(k, v);
        }
        d.finish()
    }
}

#[cfg(has_specialisation)]
impl<K, V, S> Debug for HashMap<K, V, S>
where
    K: Hash + Eq + Clone + Ord + Debug,
    V: Debug + Clone,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        let mut keys = collections::BTreeSet::new();
        keys.extend(self.keys());
        let mut d = f.debug_map();
        for key in keys {
            d.entry(key, &self[key]);
        }
        d.finish()
    }
}

// // Iterators

// An iterator over the elements of a map.
pub struct Iter<'a, K: 'a, V: 'a> {
    it: NodeIter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = &'a (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(p, _)| p)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

// A mutable iterator over the values of a map.
pub struct IterMut<'a, K: 'a, V: 'a>
where
    K: Clone,
    V: Clone,
{
    it: NodeIterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(entry, _)| &mut entry.1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V>
where
    K: Clone,
    V: Clone,
{
}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V>
where
    K: Clone,
    V: Clone,
{
}

// A consuming iterator over the elements of a map.
pub struct ConsumingIter<A: HashValue> {
    it: NodeDrain<A>,
}

impl<A> Iterator for ConsumingIter<A>
where
    A: HashValue,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(p, _)| p)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<A> ExactSizeIterator for ConsumingIter<A> where A: HashValue {}

impl<A> FusedIterator for ConsumingIter<A> where A: HashValue {}

// An iterator over the keys of a map.
pub struct Keys<'a, K: 'a, V: 'a> {
    it: NodeIter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|((k, _), _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}

impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}

// An iterator over the values of a map.
pub struct Values<'a, K: 'a, V: 'a> {
    it: NodeIter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|((_, v), _)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

impl<'a, K, V> FusedIterator for Values<'a, K, V> {}

impl<'a, K, V, S> IntoIterator for &'a HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    type Item = &'a (K, V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V, S> IntoIterator for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    type Item = (K, V);
    type IntoIter = ConsumingIter<(K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: NodeDrain::new(self.root, self.size),
        }
    }
}

// // Conversions

impl<K, V, S> FromIterator<(K, V)> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut map = Self::default();
        for (k, v) in i {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, S> AsRef<HashMap<K, V, S>> for HashMap<K, V, S>
where
    K: Clone,
    V: Clone,
{
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<'m, 'k, 'v, K, V, OK, OV, SA, SB> From<&'m HashMap<&'k K, &'v V, SA>> for HashMap<OK, OV, SB>
where
    K: Hash + Eq + ToOwned<Owned = OK> + ?Sized,
    V: ToOwned<Owned = OV> + ?Sized,
    OK: Hash + Eq + Clone + Borrow<K>,
    OV: Borrow<V> + Clone,
    SA: BuildHasher,
    SB: BuildHasher + Default,
{
    fn from(m: &HashMap<&K, &V, SA>) -> Self {
        m.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }
}

impl<'a, K, V, S> From<&'a [(K, V)]> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: &'a [(K, V)]) -> Self {
        m.into_iter().cloned().collect()
    }
}

impl<K, V, S> From<Vec<(K, V)>> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: Vec<(K, V)>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K, V, S> From<&'a Vec<(K, V)>> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: &'a Vec<(K, V)>) -> Self {
        m.into_iter().cloned().collect()
    }
}

impl<K, V, S> From<collections::HashMap<K, V>> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: collections::HashMap<K, V>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K, V, S> From<&'a collections::HashMap<K, V>> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: &'a collections::HashMap<K, V>) -> Self {
        m.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

impl<K, V, S> From<collections::BTreeMap<K, V>> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: collections::BTreeMap<K, V>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K, V, S> From<&'a collections::BTreeMap<K, V>> for HashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn from(m: &'a collections::BTreeMap<K, V>) -> Self {
        m.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

// impl<K: Ord + Hash + Eq, V, S> From<OrdMap<K, V>> for HashMap<K, V, S>
// where
//     S: BuildHasher + Default,
// {
//     fn from(m: OrdMap<K, V>) -> Self {
//         m.into_iter().collect()
//     }
// }

// impl<'a, K: Ord + Hash + Eq, V, S> From<&'a OrdMap<K, V>> for HashMap<K, V, S>
// where
//     S: BuildHasher + Default,
// {
//     fn from(m: &'a OrdMap<K, V>) -> Self {
//         m.into_iter().collect()
//     }
// }

// QuickCheck

#[cfg(all(feature = "arc", any(test, feature = "quickcheck")))]
use quickcheck::{Arbitrary, Gen};

#[cfg(all(feature = "arc", any(test, feature = "quickcheck")))]
impl<K: Hash + Eq + Arbitrary + Sync, V: Arbitrary + Sync> Arbitrary for HashMap<K, V> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        HashMap::from(Vec::<(K, V)>::arbitrary(g))
    }
}

// Proptest

#[cfg(any(test, feature = "proptest"))]
pub mod proptest {
    use super::*;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use std::ops::Range;

    /// A strategy for a hash map of a given size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// proptest! {
    ///     #[test]
    ///     fn proptest_works(ref m in hash_map(0..9999, ".*", 10..100)) {
    ///         assert!(m.len() < 100);
    ///         assert!(m.len() >= 10);
    ///     }
    /// }
    /// ```
    pub fn hash_map<K: Strategy + 'static, V: Strategy + 'static>(
        key: K,
        value: V,
        size: Range<usize>,
    ) -> BoxedStrategy<HashMap<<K::Tree as ValueTree>::Value, <V::Tree as ValueTree>::Value>>
    where
        <K::Tree as ValueTree>::Value: Hash + Eq + Clone,
        <V::Tree as ValueTree>::Value: Clone,
    {
        ::proptest::collection::vec((key, value), size.clone())
            .prop_map(HashMap::from)
            .prop_filter("Map minimum size".to_owned(), move |m| {
                m.len() >= size.start
            })
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection;
    use proptest::num::{i16, usize};
    use std::hash::BuildHasherDefault;
    use test::LolHasher;

    #[test]
    fn safe_mutation() {
        let v1: HashMap<usize, usize> = HashMap::from_iter((0..131072).into_iter().map(|i| (i, i)));
        let mut v2 = v1.clone();
        v2.insert(131000, 23);
        assert_eq!(Some(&23), v2.get(&131000));
        assert_eq!(Some(&131000), v1.get(&131000));
    }

    #[test]
    fn index_operator() {
        let mut map = hashmap![1 => 2, 3 => 4, 5 => 6];
        assert_eq!(4, map[&3]);
        map[&3] = 8;
        assert_eq!(hashmap![1 => 2, 3 => 8, 5 => 6], map);
    }

    #[test]
    fn proper_formatting() {
        let map = hashmap![1 => 2];
        assert_eq!("{1: 2}", format!("{:?}", map));

        assert_eq!("{}", format!("{:?}", HashMap::<(), ()>::new()));
    }

    #[test]
    fn remove_failing() {
        let pairs = [(1469, 0), (-67, 0)];
        let hasher: BuildHasherDefault<LolHasher> = Default::default();
        let mut m: collections::HashMap<i16, i16, _> =
            collections::HashMap::with_hasher(hasher.clone());
        for &(ref k, ref v) in &pairs {
            m.insert(*k, *v);
        }
        let mut map: HashMap<i16, i16, _> = HashMap::with_hasher(hasher);
        for (k, v) in &m {
            map = map.update(*k, *v);
        }
        for k in m.keys() {
            let l = map.len();
            assert_eq!(m.get(k).cloned(), map.get(k).cloned());
            map = map.without(k);
            assert_eq!(None, map.get(k));
            assert_eq!(l - 1, map.len());
        }
    }

    #[test]
    fn match_string_keys_with_string_slices() {
        let mut map: HashMap<String, i32> =
            From::from(&hashmap!{ "foo" => &1, "bar" => &2, "baz" => &3 });
        assert_eq!(Some(&1), map.get("foo"));
        map = map.without("foo");
        assert_eq!(Some(3), map.remove("baz"));
        map["bar"] = 8;
        assert_eq!(8, map["bar"]);
    }

    #[test]
    fn macro_allows_trailing_comma() {
        let map1 = hashmap!{"x" => 1, "y" => 2};
        let map2 = hashmap!{
            "x" => 1,
            "y" => 2,
        };
        assert_eq!(map1, map2);
    }

    #[test]
    fn remove_top_level_collisions() {
        let pairs = vec![9, 2569, 27145];
        let mut map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
        for k in pairs.clone() {
            map.insert(k, k);
        }
        assert_eq!(pairs.len(), map.len());
        let keys: Vec<_> = map.keys().cloned().collect();
        for k in keys {
            let l = map.len();
            assert_eq!(Some(&k), map.get(&k));
            map.remove(&k);
            assert_eq!(None, map.get(&k));
            assert_eq!(l - 1, map.len());
        }
    }

    #[test]
    fn entry_api() {
        let mut map = hashmap!{"bar" => 5};
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
    fn large_map() {
        let mut map = HashMap::new();
        let size = 32769;
        for i in 0..size {
            map.insert(i, i);
        }
        assert_eq!(size, map.len());
        for i in 0..size {
            assert_eq!(Some(&i), map.get(&i));
        }
    }

    proptest! {
        #[test]
        fn update_and_length(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let mut map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
            for (index, (k, v)) in m.iter().enumerate() {
                map = map.update(*k, *v);
                assert_eq!(Some(v), map.get(k));
                assert_eq!(index + 1, map.len());
            }
        }

        #[test]
        fn from_iterator(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let map: HashMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn iterate_over(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let map: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.iter().count());
        }

        #[test]
        fn equality(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let map1: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let map2: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(map1, map2);
        }

        #[test]
        fn lookup(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let map: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for (k, v) in m {
                assert_eq!(Some(*v), map.get(k).cloned());
            }
        }

        #[test]
        fn without(ref pairs in collection::vec((i16::ANY, i16::ANY), 0..100)) {
            let hasher: BuildHasherDefault<LolHasher> = Default::default();
            let mut m: collections::HashMap<i16, i16, _> =
                collections::HashMap::with_hasher(hasher.clone());
            for &(ref k, ref v) in pairs {
                m.insert(*k, *v);
            }
            let mut map: HashMap<i16, i16, _> = HashMap::with_hasher(hasher);
            for (k, v) in &m {
                map = map.update(*k, *v);
            }
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).cloned());
                map = map.without(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn insert(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let mut mut_map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
            let mut map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
            for (count, (k, v)) in m.iter().enumerate() {
                map = map.update(*k, *v);
                mut_map.insert(*k, *v);
                assert_eq!(count + 1, map.len());
                assert_eq!(count + 1, mut_map.len());
            }
            assert_eq!(map, mut_map);
        }

        #[test]
        fn remove(ref pairs in collection::vec((i16::ANY, i16::ANY), 0..100)) {
            let hasher: BuildHasherDefault<LolHasher> = Default::default();
            let mut m: collections::HashMap<i16, i16, _> =
                collections::HashMap::with_hasher(hasher.clone());
            for &(ref k, ref v) in pairs {
                m.insert(*k, *v);
            }
            let mut map: HashMap<i16, i16, _> = HashMap::with_hasher(hasher);
            for (k, v) in &m {
                map.insert(*k, *v);
            }
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).cloned());
                map.remove(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn delete_and_reinsert(
            ref input in collection::hash_map(i16::ANY, i16::ANY, 1..100),
            index_rand in usize::ANY
        ) {
            let index = *input.keys().nth(index_rand % input.len()).unwrap();
            let map1: HashMap<_, _> = HashMap::from_iter(input.clone());
            let (val, map2) = map1.extract(&index).unwrap();
            let map3 = map2.update(index, val);
            for key in map2.keys() {
                assert!(*key != index);
            }
            assert_eq!(map1.len(), map2.len() + 1);
            assert_eq!(map1, map3);
        }

        #[test]
        fn proptest_works(ref m in proptest::hash_map(0..9999, ".*", 10..100)) {
            assert!(m.len() < 100);
            assert!(m.len() >= 10);
        }

        #[test]
        fn exact_size_iterator(ref m in proptest::hash_map(i16::ANY, i16::ANY, 0..100)) {
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
