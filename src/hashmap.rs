// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A hash map.
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
//! being used. Unless otherwise specified, all maps will share an
//! instance of the default
//! [`RandomState`][std::collections::hash_map::RandomState] hasher,
//! which will produce consistent hashes for the duration of its
//! lifetime, but not between restarts of your program.
//!
//! [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
//! [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
//! [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
//! [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html

#![cfg_attr(feature = "clippy", allow(implicit_hasher))]

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, Sum};
use std::ops::Add;
use std::sync::Arc;

use bits::hash_key;
use shared::Shared;

use nodes::hamt::{HashValue, Iter, Node};

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
            map.insert_mut($key, $value);
        })*;
        map
    }};
}

/// A hash map.
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

pub struct HashMap<K, V, S = RandomState> {
    size: usize,
    root: Arc<Node<(Arc<K>, Arc<V>)>>,
    hasher: Arc<S>,
}

impl<K, V> HashValue for (Arc<K>, Arc<V>)
where
    K: Eq,
{
    type Key = K;

    fn extract_key(&self) -> &Self::Key {
        &*self.0
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.1, &other.1) && Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<K, V> HashMap<K, V, RandomState>
where
    K: Hash + Eq,
{
    /// Construct an empty hash map.
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    /// Construct a hash map with a single mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = HashMap::singleton(123, "onetwothree");
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(Arc::new("onetwothree"))
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn singleton<RK, RV>(k: RK, v: RV) -> HashMap<K, V>
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        HashMap::new().insert(k, v)
    }
}

impl<K, V, S> HashMap<K, V, S> {
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
    K: Hash + Eq,
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
        RS: Shared<S>,
    {
        HashMap {
            size: 0,
            root: Arc::new(Node::new()),
            hasher: hasher.shared(),
        }
    }

    /// Construct an empty hash map using the same hasher as the
    /// current hash map.
    #[inline]
    pub fn new_from<K1, V1>(&self) -> HashMap<K1, V1, S>
    where
        K1: Hash + Eq,
    {
        HashMap {
            size: 0,
            root: Arc::new(Node::new()),
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
    pub fn iter(&self) -> Iter<(Arc<K>, Arc<V>)> {
        Node::iter(self.root.clone(), self.size)
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
        Keys { it: self.iter() }
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
        Values { it: self.iter() }
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
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(Arc::new("lol"))
    /// );
    /// # }
    /// ```
    pub fn get(&self, k: &K) -> Option<Arc<V>> {
        self.root
            .get(hash_key(&*self.hasher, k), 0, k)
            .map(|&(_, ref v)| v.clone())
    }

    /// Get the value for a key from a hash map, or a default value if
    /// the key isn't in the map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::hashmap::HashMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get_or(&123, "hi"),
    ///   Arc::new("lol")
    /// );
    /// assert_eq!(
    ///   map.get_or(&321, "hi"),
    ///   Arc::new("hi")
    /// );
    /// # }
    /// ```
    pub fn get_or<RV>(&self, k: &K, default: RV) -> Arc<V>
    where
        RV: Shared<V>,
    {
        self.get(k).unwrap_or_else(|| default.shared())
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
    /// # use std::sync::Arc;
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
    pub fn contains_key(&self, k: &K) -> bool {
        self.get(k).is_some()
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
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = hashmap!{};
    /// assert_eq!(
    ///   map.insert(123, "123"),
    ///   hashmap!{123 => "123"}
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn insert<RK, RV>(&self, k: RK, v: RV) -> Self
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        self.insert_ref(k.shared(), v.shared())
    }

    fn insert_ref(&self, k: Arc<K>, v: Arc<V>) -> Self {
        let (added, new_node) = self.root.insert(hash_key(&*self.hasher, &k), 0, (k, v));
        HashMap {
            root: Arc::new(new_node),
            size: if added {
                self.size + 1
            } else {
                self.size
            },
            hasher: self.hasher.clone(),
        }
    }

    /// Insert a key/value mapping into a map.
    ///
    /// If the map already has a mapping for the given key, the
    /// previous value is overwritten.
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
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let mut map = hashmap!{};
    /// map.insert_mut(123, "123");
    /// map.insert_mut(456, "456");
    /// assert_eq!(
    ///   map,
    ///   hashmap!{123 => "123", 456 => "456"}
    /// );
    /// # }
    /// ```
    ///
    /// [insert]: #method.insert
    #[inline]
    pub fn insert_mut<RK, RV>(&mut self, k: RK, v: RV)
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        self.insert_mut_ref(k.shared(), v.shared())
    }

    fn insert_mut_ref(&mut self, k: Arc<K>, v: Arc<V>) {
        let hash = hash_key(&*self.hasher, &k);
        let root = Arc::make_mut(&mut self.root);
        let added = root.insert_mut(hash, 0, (k, v));
        if added {
            self.size += 1
        }
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// This is an alias for [`insert`][insert].
    ///
    /// Time: O(log n)
    ///
    /// [insert]: #method.insert
    #[inline]
    pub fn set<RK, RV>(&self, k: RK, v: RV) -> Self
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        self.insert(k, v)
    }

    /// Insert a key/value mapping into a map, mutating it in place
    /// when it is safe to do so.
    ///
    /// This is an alias for [`insert_mut`][insert_mut].
    ///
    /// Time: O(log n)
    ///
    /// [insert_mut]: #method.insert_mut
    #[inline]
    pub fn set_mut<RK, RV>(&mut self, k: RK, v: RV)
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        self.insert_mut(k, v)
    }

    /// Construct a new hash map by inserting a key/value mapping into
    /// a map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the old value and the new value,
    /// and insert the result as the new value.
    ///
    /// Time: O(log n)
    pub fn insert_with<RK, RV, F>(self, k: RK, v: RV, f: F) -> Self
    where
        RK: Shared<K>,
        RV: Shared<V>,
        F: FnOnce(Arc<V>, Arc<V>) -> Arc<V>,
    {
        let ak = k.shared();
        let av = v.shared();
        match self.pop_with_key(&ak) {
            None => self.insert_ref(ak, av),
            Some((_, v2, m)) => m.insert_ref(ak, f(v2, av)),
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
    pub fn insert_with_key<RK, RV, F>(self, k: RK, v: RV, f: F) -> Self
    where
        F: FnOnce(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
        RK: Shared<K>,
        RV: Shared<V>,
    {
        let ak = k.shared();
        let av = v.shared();
        match self.pop_with_key(&ak) {
            None => self.insert_ref(ak, av),
            Some((_, v2, m)) => m.insert_ref(ak.clone(), f(ak, v2, av)),
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
    pub fn insert_lookup_with_key<RK, RV, F>(self, k: RK, v: RV, f: F) -> (Option<Arc<V>>, Self)
    where
        F: FnOnce(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
        RK: Shared<K>,
        RV: Shared<V>,
    {
        let ak = k.shared();
        let av = v.shared();
        match self.pop_with_key(&ak) {
            None => (None, self.insert_ref(ak, av)),
            Some((_, v2, m)) => (Some(v2.clone()), m.insert_ref(ak.clone(), f(ak, v2, av))),
        }
    }

    /// Update the value for a given key by calling a function with
    /// the current value and overwriting it with the function's
    /// return value.
    ///
    /// Time: O(log n)
    pub fn update<F>(&self, k: &K, f: F) -> Self
    where
        F: FnOnce(Arc<V>) -> Option<Arc<V>>,
    {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((k, v, m)) => match f(v) {
                None => m,
                Some(v) => m.insert(k, v),
            },
        }
    }

    /// Update the value for a given key by calling a function with
    /// the key and the current value and overwriting it with the
    /// function's return value.
    ///
    /// Time: O(log n)
    pub fn update_with_key<F>(&self, k: &K, f: F) -> Self
    where
        F: FnOnce(Arc<K>, Arc<V>) -> Option<Arc<V>>,
    {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((k, v, m)) => match f(k.clone(), v) {
                None => m,
                Some(v) => m.insert(k, v),
            },
        }
    }

    /// Update the value for a given key by calling a function with
    /// the key and the current value and overwriting it with the
    /// function's return value.
    ///
    /// If the key was not in the map, the function is never called
    /// and the map is left unchanged.
    ///
    /// Return a tuple of the old value, if there was one, and the new
    /// map.
    ///
    /// Time: O(log n)
    pub fn update_lookup_with_key<F>(&self, k: &K, f: F) -> (Option<Arc<V>>, Self)
    where
        F: FnOnce(Arc<K>, Arc<V>) -> Option<Arc<V>>,
    {
        match self.pop_with_key(k) {
            None => (None, self.clone()),
            Some((k, v, m)) => match f(k.clone(), v.clone()) {
                None => (Some(v), m),
                Some(v) => (Some(v.clone()), m.insert(k, v)),
            },
        }
    }

    /// Update the value for a given key by calling a function with
    /// the current value and overwriting it with the function's
    /// return value.
    ///
    /// This is like the [`update`][update] method, except with more
    /// control: the function gets an
    /// [`Option<V>`][std::option::Option] and returns the same, so
    /// that it can decide to delete a mapping instead of updating the
    /// value, and decide what to do if the key isn't in the map.
    ///
    /// Time: O(log n)
    ///
    /// [update]: #method.update
    /// [std::option::Option]: https://doc.rust-lang.org/std/option/enum.Option.html
    pub fn alter<RK, F>(&self, f: F, k: RK) -> Self
    where
        F: FnOnce(Option<Arc<V>>) -> Option<Arc<V>>,
        RK: Shared<K>,
    {
        let ak = k.shared();
        let pop = self.pop_with_key(&*ak);
        match (f(pop.as_ref().map(|&(_, ref v, _)| v.clone())), pop) {
            (None, None) => self.clone(),
            (Some(v), None) => self.insert_ref(ak, v),
            (None, Some((_, _, m))) => m,
            (Some(v), Some((_, _, m))) => m.insert_ref(ak, v),
        }
    }

    /// Remove a key/value pair from a hash map, if it exists.
    ///
    /// Time: O(log n)
    pub fn remove(&self, k: &K) -> Self {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((_, _, map)) => map,
        }
    }

    /// Remove a key/value mapping from a map if it exists.
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
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let mut map = hashmap!{123 => "123", 456 => "456"};
    /// map.remove_mut(&123);
    /// map.remove_mut(&456);
    /// assert!(map.is_empty());
    /// # }
    /// ```
    ///
    /// [remove]: #method.remove
    #[inline]
    pub fn remove_mut(&mut self, k: &K) {
        self.pop_with_key_mut(k);
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value as well as the updated list.
    ///
    /// Time: O(log n)
    pub fn pop(&self, k: &K) -> Option<(Arc<V>, Self)> {
        self.pop_with_key(k).map(|(_, v, m)| (v, m))
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn pop_mut(&mut self, k: &K) -> Option<Arc<V>> {
        self.pop_with_key_mut(k).map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value as well as the updated list.
    ///
    /// Time: O(log n)
    pub fn pop_with_key(&self, k: &K) -> Option<(Arc<K>, Arc<V>, Self)> {
        self.root
            .remove(hash_key(&*self.hasher, k), 0, k)
            .map(|((k, v), node)| {
                (
                    k,
                    v,
                    HashMap {
                        hasher: self.hasher.clone(),
                        size: self.size - 1,
                        root: Arc::new(node),
                    },
                )
            })
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    pub fn pop_with_key_mut(&mut self, k: &K) -> Option<(Arc<K>, Arc<V>)> {
        let root = Arc::make_mut(&mut self.root);
        let result = root.remove_mut(hash_key(&*self.hasher, k), 0, k);
        if result.is_some() {
            self.size -= 1;
        }
        result
    }

    /// Construct the union of two maps, keeping the values in the
    /// current map when keys exist in both maps.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        self.union_with_key(other, |_, v, _| v)
    }

    /// Construct the union of two maps, using a function to decide
    /// what to do with the value when a key is in both maps.
    #[inline]
    pub fn union_with<F, RM>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<V>, Arc<V>) -> Arc<V>,
        RM: Borrow<Self>,
    {
        self.union_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the union of two maps, using a function to decide
    /// what to do with the value when a key is in both maps. The
    /// function receives the key as well as both values.
    pub fn union_with_key<F, RM>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
        RM: Borrow<Self>,
    {
        other.borrow().iter().fold(self.clone(), |m, (k, v)| {
            m.insert(
                k.clone(),
                self.get(&*k).map(|v1| f(k, v1, v.clone())).unwrap_or(v),
            )
        })
    }

    /// Construct the union of a sequence of maps, selecting the value
    /// of the leftmost when a key appears in more than one map.
    pub fn unions<I>(i: I) -> Self
    where
        S: Default,
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(Default::default(), |a, b| a.union(&b))
    }

    /// Construct the union of a sequence of maps, using a function to
    /// decide what to do with the value when a key is in more than
    /// one map.
    pub fn unions_with<I, F>(i: I, f: F) -> Self
    where
        S: Default,
        I: IntoIterator<Item = Self>,
        F: Fn(Arc<V>, Arc<V>) -> Arc<V>,
    {
        i.into_iter()
            .fold(Default::default(), |a, b| a.union_with(&b, &f))
    }

    /// Construct the union of a sequence of maps, using a function to
    /// decide what to do with the value when a key is in more than
    /// one map. The function receives the key as well as both values.
    pub fn unions_with_key<I, F>(i: I, f: F) -> Self
    where
        S: Default,
        I: IntoIterator<Item = Self>,
        F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
    {
        i.into_iter()
            .fold(Default::default(), |a, b| a.union_with_key(&b, &f))
    }

    /// Construct the difference between two maps by discarding keys
    /// which occur in both maps.
    #[inline]
    pub fn difference<B, RM>(&self, other: RM) -> Self
    where
        RM: Borrow<HashMap<K, B, S>>,
    {
        self.difference_with_key(other, |_, _, _| None)
    }

    /// Construct the difference between two maps by using a function
    /// to decide what to do if a key occurs in both.
    #[inline]
    pub fn difference_with<B, RM, F>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<V>, Arc<B>) -> Option<Arc<V>>,
        RM: Borrow<HashMap<K, B, S>>,
    {
        self.difference_with_key(other, |_, a, b| f(a, b))
    }

    /// Construct the difference between two maps by using a function
    /// to decide what to do if a key occurs in both. The function
    /// receives the key as well as both values.
    pub fn difference_with_key<B, RM, F>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<K>, Arc<V>, Arc<B>) -> Option<Arc<V>>,
        RM: Borrow<HashMap<K, B, S>>,
    {
        other
            .borrow()
            .iter()
            .fold(self.clone(), |m, (k, v2)| match m.pop(&*k) {
                None => m,
                Some((v1, m)) => match f(k.clone(), v1, v2) {
                    None => m,
                    Some(v) => m.insert(k, v),
                },
            })
    }

    /// Construct the intersection of two maps, keeping the values
    /// from the current map.
    #[inline]
    pub fn intersection<B, RM>(&self, other: RM) -> Self
    where
        RM: Borrow<HashMap<K, B, S>>,
    {
        self.intersection_with_key(other, |_, v, _| v)
    }

    /// Construct the intersection of two maps, calling a function
    /// with both values for each key and using the result as the
    /// value for the key.
    #[inline]
    pub fn intersection_with<B, C, RM, F>(&self, other: RM, f: F) -> HashMap<K, C, S>
    where
        F: Fn(Arc<V>, Arc<B>) -> Arc<C>,
        RM: Borrow<HashMap<K, B, S>>,
    {
        self.intersection_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the intersection of two maps, calling a function
    /// with the key and both values for each key and using the result
    /// as the value for the key.
    pub fn intersection_with_key<B, C, RM, F>(&self, other: RM, f: F) -> HashMap<K, C, S>
    where
        F: Fn(Arc<K>, Arc<V>, Arc<B>) -> Arc<C>,
        RM: Borrow<HashMap<K, B, S>>,
    {
        other.borrow().iter().fold(self.new_from(), |m, (k, v2)| {
            self.get(&*k)
                .map(|v1| m.insert(k.clone(), f(k, v1, v2)))
                .unwrap_or(m)
        })
    }

    /// Merge two maps.
    ///
    /// First, we call the `combine` function for each key/value pair
    /// which exists in both maps, updating the value or discarding it
    /// according to the function's return value.
    ///
    /// The `only1` and `only2` functions are called with the
    /// key/value pairs which are only in the first and the second
    /// list respectively. The results of these are then merged with
    /// the result of the first operation.
    pub fn merge_with_key<B, C, RM, FC, F1, F2>(
        &self,
        other: RM,
        combine: FC,
        only1: F1,
        only2: F2,
    ) -> HashMap<K, C, S>
    where
        RM: Borrow<HashMap<K, B, S>>,
        FC: Fn(Arc<K>, Arc<V>, Arc<B>) -> Option<Arc<C>>,
        F1: FnOnce(Self) -> HashMap<K, C, S>,
        F2: FnOnce(HashMap<K, B, S>) -> HashMap<K, C, S>,
    {
        let (left, right, both) = other.borrow().iter().fold(
            (self.clone(), other.borrow().clone(), self.new_from()),
            |(l, r, m), (k, vr)| match l.pop(&*k) {
                None => (l, r, m),
                Some((vl, ml)) => (
                    ml,
                    r.remove(&*k),
                    combine(k.clone(), vl, vr)
                        .map(|v| m.insert(k, v))
                        .unwrap_or(m),
                ),
            },
        );
        both.union(&only1(left)).union(&only2(right))
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the
    /// same values.
    ///
    /// Use the provided function to decide whether values are equal.
    pub fn is_submap_by<B, RM, F>(&self, other: RM, cmp: F) -> bool
    where
        F: Fn(Arc<V>, Arc<B>) -> bool,
        RM: Borrow<HashMap<K, B, S>>,
    {
        self.iter().all(|(k, v)| {
            other
                .borrow()
                .get(&*k)
                .map(|ov| cmp(v, ov))
                .unwrap_or(false)
        })
    }

    /// Test whether a map is a proper submap of another map, meaning
    /// that all keys in our map must also be in the other map, with
    /// the same values. To be a proper submap, ours must also contain
    /// fewer keys than the other map.
    ///
    /// Use the provided function to decide whether values are equal.
    pub fn is_proper_submap_by<B, RM, F>(&self, other: RM, cmp: F) -> bool
    where
        F: Fn(Arc<V>, Arc<B>) -> bool,
        RM: Borrow<HashMap<K, B, S>>,
    {
        self.len() != other.borrow().len() && self.is_submap_by(other, cmp)
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the
    /// same values.
    pub fn is_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_submap_by(other.borrow(), |a, b| a.as_ref().eq(b.as_ref()))
    }

    /// Test whether a map is a proper submap of another map, meaning
    /// that all keys in our map must also be in the other map, with
    /// the same values. To be a proper submap, ours must also contain
    /// fewer keys than the other map.
    pub fn is_proper_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_proper_submap_by(other.borrow(), |a, b| a.as_ref().eq(b.as_ref()))
    }
}

// Core traits

impl<K, V, S> Clone for HashMap<K, V, S> {
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
    K: Hash + Eq,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.test_eq(other)
    }
}

#[cfg(has_specialisation)]
impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Hash + Eq,
    V: PartialEq,
    S: BuildHasher,
{
    default fn eq(&self, other: &Self) -> bool {
        self.test_eq(other)
    }
}

#[cfg(has_specialisation)]
impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Hash + Eq,
    V: Eq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.root, &other.root) {
            return true;
        }
        self.test_eq(other)
    }
}

impl<K: Hash + Eq, V: Eq, S: BuildHasher> Eq for HashMap<K, V, S> {}

impl<K, V, S> PartialOrd for HashMap<K, V, S>
where
    K: Hash + Eq + PartialOrd,
    V: PartialOrd,
    S: BuildHasher,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().partial_cmp(other.iter());
        }
        let m1: ::std::collections::HashMap<Arc<K>, Arc<V>> = self.iter().collect();
        let m2: ::std::collections::HashMap<Arc<K>, Arc<V>> = other.iter().collect();
        m1.iter().partial_cmp(m2.iter())
    }
}

impl<K, V, S> Ord for HashMap<K, V, S>
where
    K: Hash + Eq + Ord,
    V: Ord,
    S: BuildHasher,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.hasher, &other.hasher) {
            return self.iter().cmp(other.iter());
        }
        let m1: ::std::collections::HashMap<Arc<K>, Arc<V>> = self.iter().collect();
        let m2: ::std::collections::HashMap<Arc<K>, Arc<V>> = other.iter().collect();
        m1.iter().cmp(m2.iter())
    }
}

impl<K, V, S> Hash for HashMap<K, V, S>
where
    K: Hash + Eq,
    V: Hash,
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
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    #[inline]
    fn default() -> Self {
        HashMap {
            size: 0,
            root: Arc::new(Node::new()),
            hasher: Default::default(),
        }
    }
}

impl<K, V, S> Add for HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Output = HashMap<K, V, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(&other)
    }
}

impl<'a, K, V, S> Add for &'a HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Output = HashMap<K, V, S>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<K, V, S> Sum for HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Default::default(), |a, b| a + b)
    }
}

impl<K, V, S, RK, RV> Extend<(RK, RV)> for HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
    RK: Shared<K>,
    RV: Shared<V>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (RK, RV)>,
    {
        for (key, value) in iter {
            self.insert_mut(key, value);
        }
    }
}

impl<K, V, S> Debug for HashMap<K, V, S>
where
    K: Hash + Eq + Debug,
    V: Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{{ ")?;
        let mut it = self.iter().peekable();
        loop {
            match it.next() {
                None => break,
                Some((k, v)) => {
                    write!(f, "{:?} => {:?}", k, v)?;
                    match it.peek() {
                        None => write!(f, " }}")?,
                        Some(_) => write!(f, ", ")?,
                    }
                }
            }
        }
        Ok(())
    }
}

// // Iterators

pub struct Keys<K, V> {
    it: Iter<(Arc<K>, Arc<V>)>,
}

impl<K, V> Iterator for Keys<K, V> {
    type Item = Arc<K>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(k, _)| k)
    }
}

pub struct Values<K, V> {
    it: Iter<(Arc<K>, Arc<V>)>,
}

impl<K, V> Iterator for Values<K, V> {
    type Item = Arc<V>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(_, v)| v)
    }
}

impl<'a, K, V, S> IntoIterator for &'a HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<(Arc<K>, Arc<V>)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V, S> IntoIterator for HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<(Arc<K>, Arc<V>)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// // Conversions

impl<K, V, RK, RV, S> FromIterator<(RK, RV)> for HashMap<K, V, S>
where
    K: Hash + Eq,
    RK: Shared<K>,
    RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = (RK, RV)>,
    {
        let mut map: Self = Default::default();
        for (k, v) in i {
            map.insert_mut(k, v);
        }
        map
    }
}

impl<K, V, S> AsRef<HashMap<K, V, S>> for HashMap<K, V, S> {
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<'a, K: Hash + Eq, V: Clone, RK, RV, S> From<&'a [(RK, RV)]> for HashMap<K, V, S>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: &'a [(RK, RV)]) -> Self {
        m.into_iter()
            .map(|&(ref k, ref v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Hash + Eq, V, RK, RV, S> From<Vec<(RK, RV)>> for HashMap<K, V, S>
where
    RK: Shared<K>,
    RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: Vec<(RK, RV)>) -> Self {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<'a, K: Hash + Eq, V, RK, RV, S> From<&'a Vec<(RK, RV)>> for HashMap<K, V, S>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: &'a Vec<(RK, RV)>) -> Self {
        m.into_iter()
            .map(|&(ref k, ref v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Hash + Eq, V, RK: Hash + Eq, RV, S> From<collections::HashMap<RK, RV>> for HashMap<K, V, S>
where
    RK: Shared<K>,
    RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: collections::HashMap<RK, RV>) -> Self {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<'a, K: Hash + Eq, V, RK: Hash + Eq, RV, S> From<&'a collections::HashMap<RK, RV>>
    for HashMap<K, V, S>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: &'a collections::HashMap<RK, RV>) -> Self {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Hash + Eq, V, RK, RV, S> From<collections::BTreeMap<RK, RV>> for HashMap<K, V, S>
where
    RK: Shared<K>,
    RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: collections::BTreeMap<RK, RV>) -> Self {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<'a, K: Hash + Eq, V, RK, RV, S> From<&'a collections::BTreeMap<RK, RV>> for HashMap<K, V, S>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
    S: BuildHasher + Default,
{
    fn from(m: &'a collections::BTreeMap<RK, RV>) -> Self {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
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

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
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
    ) -> BoxedStrategy<HashMap<<K::Value as ValueTree>::Value, <V::Value as ValueTree>::Value>>
    where
        <K::Value as ValueTree>::Value: Hash + Eq,
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
        v2.set_mut(131000, 23);
        assert_eq!(Some(Arc::new(23)), v2.get(&131000));
        assert_eq!(Some(Arc::new(131000)), v1.get(&131000));
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
            map = map.insert(*k, *v);
        }
        for k in m.keys() {
            let l = map.len();
            assert_eq!(m.get(k).cloned(), map.get(k).map(|v| *v));
            map = map.remove(k);
            assert_eq!(None, map.get(k));
            assert_eq!(l - 1, map.len());
        }
    }

    proptest! {
        #[test]
        fn insert_and_length(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let mut map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
            for (index, (k, v)) in m.iter().enumerate() {
                map = map.insert(*k, *v);
                assert_eq!(Some(Arc::new(*v)), map.get(k));
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
                assert_eq!(Some(*v), map.get(k).map(|v| *v));
            }
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
                map = map.insert(*k, *v);
            }
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).map(|v| *v));
                map = map.remove(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn insert_mut(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let mut mut_map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
            let mut map: HashMap<i16, i16, BuildHasherDefault<LolHasher>> = Default::default();
            for (count, (k, v)) in m.iter().enumerate() {
                map = map.insert(*k, *v);
                mut_map.insert_mut(*k, *v);
                assert_eq!(count + 1, map.len());
                assert_eq!(count + 1, mut_map.len());
            }
            assert_eq!(map, mut_map);
        }

        #[test]
        fn remove_mut(ref pairs in collection::vec((i16::ANY, i16::ANY), 0..100)) {
            let hasher: BuildHasherDefault<LolHasher> = Default::default();
            let mut m: collections::HashMap<i16, i16, _> =
                collections::HashMap::with_hasher(hasher.clone());
            for &(ref k, ref v) in pairs {
                m.insert(*k, *v);
            }
            let mut map: HashMap<i16, i16, _> = HashMap::with_hasher(hasher);
            for (k, v) in &m {
                map.insert_mut(*k, *v);
            }
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).map(|v| *v));
                map.remove_mut(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn delete_and_reinsert(ref input in collection::hash_map(i16::ANY, i16::ANY, 1..100),
                               index_rand in usize::ANY) {
            let index = *input.keys().nth(index_rand % input.len()).unwrap();
            let map1: HashMap<_, _> = HashMap::from_iter(input.clone());
            let (val, map2) = map1.pop(&index).unwrap();
            let map3 = map2.insert(index, val);
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
