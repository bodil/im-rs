//! An ordered map.
//!
//! An immutable ordered map implemented as a balanced 2-3 tree.
//!
//! Most operations on this type of map are O(log n).
//! A [`HashMap`][hashmap::HashMap] is usually a better choice for
//! performance, but the `OrdMap` has the advantage of only
//! requiring an [`Ord`][std::cmp::Ord] constraint on the key, and
//! of being ordered, so that keys always come out from lowest to
//! highest, where a [`HashMap`][hashmap::HashMap] has no guaranteed
//! ordering.
//!
//! [hashmap::HashMap]: ../hashmap/struct.HashMap.html
//! [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html

use std::sync::Arc;
use std::iter::{FromIterator, Iterator};
use std::collections;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Error, Formatter};
use std::ops::Add;
use std::borrow::Borrow;
use std::marker::PhantomData;

use shared::Shared;
use lens::PartialLens;
use hashmap::HashMap;

mod nodes;
use self::nodes::{Insert, Node, Remove};
pub use self::nodes::Iter;

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
            map.insert_mut($key, $value);
        })*;
        map
    }};
}

/// # Ordered Map
///
/// An immutable ordered map implemented as a balanced 2-3 tree.
///
/// Most operations on this type of map are O(log n).
/// A [`HashMap`][hashmap::HashMap] is usually a better choice for
/// performance, but the `OrdMap` has the advantage of only
/// requiring an [`Ord`][std::cmp::Ord] constraint on the key, and
/// of being ordered, so that keys always come out from lowest to
/// highest, where a [`HashMap`][hashmap::HashMap] has no guaranteed
/// ordering.
///
/// [hashmap::HashMap]: ../hashmap/struct.HashMap.html
/// [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html
pub struct OrdMap<K, V>(Node<K, V>);

impl<K, V> OrdMap<K, V> {
    /// Construct an empty map.
    pub fn new() -> Self {
        OrdMap(Node::new())
    }

    /// Construct a map with a single mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = OrdMap::singleton(123, "onetwothree");
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(Arc::new("onetwothree"))
    /// );
    /// # }
    /// ```
    pub fn singleton<RK, RV>(key: RK, value: RV) -> Self
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        OrdMap(Node::singleton(key.shared(), value.shared()))
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
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an iterator over the key/value pairs of a map.
    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(&self.0)
    }

    /// Get an iterator over a map's keys.
    pub fn keys(&self) -> Keys<K, V> {
        Keys { it: self.iter() }
    }

    /// Get an iterator over a map's values.
    pub fn values(&self) -> Values<K, V> {
        Values { it: self.iter() }
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
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Get the largest key in a map, along with its value.
    /// If the map is empty, return `None`.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// assert_eq!(Some((Arc::new(3), Arc::new(33))), ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.get_max());
    /// # }
    /// ```
    pub fn get_max(&self) -> Option<(Arc<K>, Arc<V>)> {
        self.0.max().map(|p| (p.key.clone(), p.value.clone()))
    }

    /// Get the smallest key in a map, along with its value.
    /// If the map is empty, return `None`.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// assert_eq!(Some((Arc::new(1), Arc::new(11))), ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.get_min());
    /// # }
    /// ```
    pub fn get_min(&self) -> Option<(Arc<K>, Arc<V>)> {
        self.0.min().map(|p| (p.key.clone(), p.value.clone()))
    }
}

impl<K: Ord, V> OrdMap<K, V> {
    /// Make a `PartialLens` from the map to the value described by the
    /// given `key`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # use im::lens::{self, PartialLens};
    /// # fn main() {
    /// let map =
    ///   ordmap!{
    ///     "foo" => "bar"
    /// };
    /// let lens = OrdMap::lens("foo");
    /// assert_eq!(lens.try_get(&map), Some(Arc::new("bar")));
    /// # }
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use im::lens::{self, PartialLens};
    /// # use std::sync::Arc;
    /// # fn main() {
    /// // Make a lens into a map of maps
    /// let map =
    ///   ordmap!{
    ///     "foo" => ordmap!{
    ///       "bar" => "gazonk"
    ///     }
    /// };
    /// let lens1 = OrdMap::lens("foo");
    /// let lens2 = OrdMap::lens("bar");
    /// let lens = lens::compose(&lens1, &lens2);
    /// assert_eq!(lens.try_get(&map), Some(Arc::new("gazonk")));
    /// # }
    /// ```
    pub fn lens<RK>(key: RK) -> OrdMapLens<K, V>
    where
        RK: Shared<K>,
    {
        OrdMapLens {
            key: key.shared(),
            value: PhantomData,
        }
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
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = ordmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(Arc::new("lol"))
    /// );
    /// # }
    /// ```
    pub fn get(&self, k: &K) -> Option<Arc<V>> {
        self.0.lookup(k)
    }

    /// Get the value for a key from a map, or a default value
    /// if the key isn't in the map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = ordmap!{123 => "lol"};
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

    /// Test for the presence of a key in a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
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
    pub fn contains_key(&self, k: &K) -> bool {
        self.get(k).is_some()
    }

    /// Construct a new map by inserting a key/value mapping into a map.
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
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let map = ordmap!{};
    /// assert_eq!(
    ///   map.insert(123, "123"),
    ///   ordmap!{123 => "123"}
    /// );
    /// # }
    /// ```
    pub fn insert<RK, RV>(&self, k: RK, v: RV) -> Self
    where
        RK: Shared<K>,
        RV: Shared<V>,
    {
        self.insert_ref(k.shared(), v.shared())
    }

    fn insert_ref(&self, k: Arc<K>, v: Arc<V>) -> Self {
        match self.0.insert(k, v) {
            Insert::NoChange => self.clone(),
            Insert::JustInc => unreachable!(),
            Insert::Update(root) => OrdMap(root),
            Insert::Split(left, median, right) => OrdMap(Node::from_split(left, median, right)),
        }
    }

    /// Insert a key/value mapping into a map, mutating it in place when it is
    /// safe to do so.
    ///
    /// If you are the sole owner of the map, it is safe to mutate it without
    /// losing immutability guarantees, gaining us a considerable performance
    /// advantage. If the map is in use elsewhere, this operation will safely
    /// clone the map before mutating it, acting just like the immutable
    /// [`insert`][insert] operation.
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
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let mut map = ordmap!{};
    /// map.insert_mut(123, "123");
    /// map.insert_mut(456, "456");
    /// assert_eq!(
    ///   map,
    ///   ordmap!{123 => "123", 456 => "456"}
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
        match self.0.insert_mut(k, v) {
            Insert::NoChange | Insert::JustInc => {}
            Insert::Update(root) => self.0 = root,
            Insert::Split(left, median, right) => self.0 = Node::from_split(left, median, right),
        }
    }

    /// Construct a new map by inserting a key/value mapping into a map.
    ///
    /// If the map already has a mapping for the given key, we call the provided
    /// function with the old value and the new value, and insert the result as
    /// the new value.
    ///
    /// Time: O(log n)
    pub fn insert_with<RK, RV, F>(self, k: RK, v: RV, f: F) -> Self
    where
        RK: Shared<K>,
        RV: Shared<V>,
        F: Fn(Arc<V>, Arc<V>) -> Arc<V>,
    {
        let ak = k.shared();
        let av = v.shared();
        match self.pop_with_key(&ak) {
            None => self.insert_ref(ak, av),
            Some((_, v2, m)) => m.insert_ref(ak, f(v2, av)),
        }
    }

    /// Construct a new map by inserting a key/value mapping into a map.
    ///
    /// If the map already has a mapping for the given key, we call the provided
    /// function with the key, the old value and the new value, and insert the result as
    /// the new value.
    ///
    /// Time: O(log n)
    pub fn insert_with_key<RK, RV, F>(self, k: RK, v: RV, f: F) -> Self
    where
        F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
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

    /// Construct a new map by inserting a key/value mapping into a map, returning
    /// the old value for the key as well as the new map.
    ///
    /// If the map already has a mapping for the given key, we call the provided
    /// function with the key, the old value and the new value, and insert the result as
    /// the new value.
    ///
    /// Time: O(log n)
    pub fn insert_lookup_with_key<RK, RV, F>(self, k: RK, v: RV, f: F) -> (Option<Arc<V>>, Self)
    where
        F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
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

    /// Update the value for a given key by calling a function with the current value
    /// and overwriting it with the function's return value.
    ///
    /// Time: O(log n)
    pub fn update<F>(&self, k: &K, f: F) -> Self
    where
        F: Fn(Arc<V>) -> Option<Arc<V>>,
    {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((k, v, m)) => match f(v) {
                None => m,
                Some(v) => m.insert(k, v),
            },
        }
    }

    /// Update the value for a given key by calling a function with the key and the current value
    /// and overwriting it with the function's return value.
    ///
    /// Time: O(log n)
    pub fn update_with_key<F>(&self, k: &K, f: F) -> Self
    where
        F: Fn(Arc<K>, Arc<V>) -> Option<Arc<V>>,
    {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((k, v, m)) => match f(k.clone(), v) {
                None => m,
                Some(v) => m.insert(k, v),
            },
        }
    }

    /// Update the value for a given key by calling a function with the key and the current value
    /// and overwriting it with the function's return value.
    ///
    /// If the key was not in the map, the function is never called and the map is left unchanged.
    ///
    /// Return a tuple of the old value, if there was one, and the new map.
    ///
    /// Time: O(log n)
    pub fn update_lookup_with_key<F>(&self, k: &K, f: F) -> (Option<Arc<V>>, Self)
    where
        F: Fn(Arc<K>, Arc<V>) -> Option<Arc<V>>,
    {
        match self.pop_with_key(k) {
            None => (None, self.clone()),
            Some((k, v, m)) => match f(k.clone(), v.clone()) {
                None => (Some(v), m),
                Some(v) => (Some(v.clone()), m.insert(k, v)),
            },
        }
    }

    /// Update the value for a given key by calling a function with the current value
    /// and overwriting it with the function's return value.
    ///
    /// This is like the [`update`][update] method, except with more control: the function gets
    /// an [`Option<V>`][std::option::Option] and returns the same, so that it can decide to delete a mapping
    /// instead of updating the value, and decide what to do if the key isn't in the map.
    ///
    /// Time: O(log n)
    ///
    /// [update]: #method.update
    /// [std::option::Option]: https://doc.rust-lang.org/std/option/enum.Option.html
    pub fn alter<RK, F>(&self, f: F, k: RK) -> Self
    where
        F: Fn(Option<Arc<V>>) -> Option<Arc<V>>,
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

    /// Remove a key/value pair from a map, if it exists.
    ///
    /// Time: O(log n)
    pub fn remove(&self, k: &K) -> Self {
        self.pop(k).map(|(_, m)| m).unwrap_or_else(|| self.clone())
    }

    /// Remove a key/value mapping from a map if it exists, mutating it in place
    /// when it is safe to do so.
    ///
    /// If you are the sole owner of the map, it is safe to mutate it without
    /// losing immutability guarantees, gaining us a considerable performance
    /// advantage. If the map is in use elsewhere, this operation will safely
    /// clone the map before mutating it, acting just like the immutable
    /// [`remove`][remove] operation.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate im;
    /// # use im::ordmap::OrdMap;
    /// # use std::sync::Arc;
    /// # fn main() {
    /// let mut map = ordmap!{123 => "123", 456 => "456"};
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

    /// Remove a key/value pair from a map, if it exists, and return the removed value
    /// as well as the updated list.
    ///
    /// Time: O(log n)
    pub fn pop(&self, k: &K) -> Option<(Arc<V>, Self)> {
        self.pop_with_key(k).map(|(_, v, m)| (v, m))
    }

    pub fn pop_mut(&mut self, k: &K) -> Option<Arc<V>> {
        self.pop_with_key_mut(k).map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return the removed key and value
    /// as well as the updated list.
    ///
    /// Time: O(log n)
    pub fn pop_with_key(&self, k: &K) -> Option<(Arc<K>, Arc<V>, Self)> {
        match self.0.remove(k) {
            Remove::NoChange => None,
            Remove::Removed(_) => unreachable!(),
            Remove::Update(pair, root) => Some((pair.key, pair.value, OrdMap(root))),
        }
    }

    pub fn pop_with_key_mut(&mut self, k: &K) -> Option<(Arc<K>, Arc<V>)> {
        match self.0.remove_mut(k) {
            Remove::NoChange => None,
            Remove::Removed(pair) => Some((pair.key, pair.value)),
            Remove::Update(pair, root) => {
                self.0 = root;
                Some((pair.key, pair.value))
            }
        }
    }

    /// Construct the union of two maps, keeping the values in the current map
    /// when keys exist in both maps.
    pub fn union(&self, other: &Self) -> Self {
        self.union_with_key(other, |_, v, _| v)
    }

    /// Construct the union of two maps, using a function to decide what to do
    /// with the value when a key is in both maps.
    pub fn union_with<F, RM>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<V>, Arc<V>) -> Arc<V>,
        RM: Borrow<Self>,
    {
        self.union_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the union of two maps, using a function to decide what to do
    /// with the value when a key is in both maps. The function receives the key
    /// as well as both values.
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

    /// Construct the union of a sequence of maps, selecting the value of the
    /// leftmost when a key appears in more than one map.
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(ordmap![], |a, b| a.union(&b))
    }

    /// Construct the union of a sequence of maps, using a function to decide what to do
    /// with the value when a key is in more than one map.
    pub fn unions_with<I, F>(i: I, f: F) -> Self
    where
        I: IntoIterator<Item = Self>,
        F: Fn(Arc<V>, Arc<V>) -> Arc<V>,
    {
        i.into_iter().fold(ordmap![], |a, b| a.union_with(&b, &f))
    }

    /// Construct the union of a sequence of maps, using a function to decide what to do
    /// with the value when a key is in more than one map. The function receives the key
    /// as well as both values.
    pub fn unions_with_key<I, F>(i: I, f: F) -> Self
    where
        I: IntoIterator<Item = Self>,
        F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>,
    {
        i.into_iter()
            .fold(ordmap![], |a, b| a.union_with_key(&b, &f))
    }

    /// Construct the difference between two maps by discarding keys which occur in both maps.
    pub fn difference<B, RM>(&self, other: RM) -> Self
    where
        RM: Borrow<OrdMap<K, B>>,
    {
        self.difference_with_key(other, |_, _, _| None)
    }

    /// Construct the difference between two maps by using a function to decide
    /// what to do if a key occurs in both.
    pub fn difference_with<B, RM, F>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<V>, Arc<B>) -> Option<Arc<V>>,
        RM: Borrow<OrdMap<K, B>>,
    {
        self.difference_with_key(other, |_, a, b| f(a, b))
    }

    /// Construct the difference between two maps by using a function to decide
    /// what to do if a key occurs in both. The function receives the key
    /// as well as both values.
    pub fn difference_with_key<B, RM, F>(&self, other: RM, f: F) -> Self
    where
        F: Fn(Arc<K>, Arc<V>, Arc<B>) -> Option<Arc<V>>,
        RM: Borrow<OrdMap<K, B>>,
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

    /// Construct the intersection of two maps, keeping the values from the current map.
    pub fn intersection<B, RM>(&self, other: RM) -> Self
    where
        RM: Borrow<OrdMap<K, B>>,
    {
        self.intersection_with_key(other, |_, v, _| v)
    }

    /// Construct the intersection of two maps, calling a function with both values for each
    /// key and using the result as the value for the key.
    pub fn intersection_with<B, C, RM, F>(&self, other: RM, f: F) -> OrdMap<K, C>
    where
        F: Fn(Arc<V>, Arc<B>) -> Arc<C>,
        RM: Borrow<OrdMap<K, B>>,
    {
        self.intersection_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the intersection of two maps, calling a function
    /// with the key and both values for each
    /// key and using the result as the value for the key.
    pub fn intersection_with_key<B, C, RM, F>(&self, other: RM, f: F) -> OrdMap<K, C>
    where
        F: Fn(Arc<K>, Arc<V>, Arc<B>) -> Arc<C>,
        RM: Borrow<OrdMap<K, B>>,
    {
        other.borrow().iter().fold(ordmap![], |m, (k, v2)| {
            self.get(&*k)
                .map(|v1| m.insert(k.clone(), f(k, v1, v2)))
                .unwrap_or(m)
        })
    }

    /// Merge two maps.
    ///
    /// First, we call the `combine` function for each key/value pair which exists in both maps,
    /// updating the value or discarding it according to the function's return value.
    ///
    /// The `only1` and `only2` functions are called with the key/value pairs which are only in
    /// the first and the second list respectively. The results of these are then merged with
    /// the result of the first operation.
    pub fn merge_with_key<B, C, RM, FC, F1, F2>(
        &self,
        other: RM,
        combine: FC,
        only1: F1,
        only2: F2,
    ) -> OrdMap<K, C>
    where
        RM: Borrow<OrdMap<K, B>>,
        FC: Fn(Arc<K>, Arc<V>, Arc<B>) -> Option<Arc<C>>,
        F1: Fn(Self) -> OrdMap<K, C>,
        F2: Fn(OrdMap<K, B>) -> OrdMap<K, C>,
    {
        let (left, right, both) = other.borrow().iter().fold(
            (self.clone(), other.borrow().clone(), ordmap![]),
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

    /// Split a map into two, with the left hand map containing keys which are smaller
    /// than `split`, and the right hand map containing keys which are larger than `split`.
    ///
    /// The `split` mapping is discarded.
    pub fn split(&self, split: &K) -> (Self, Self) {
        let (l, _, r) = self.split_lookup(split);
        (l, r)
    }

    /// Split a map into two, with the left hand map containing keys which are smaller
    /// than `split`, and the right hand map containing keys which are larger than `split`.
    ///
    /// Returns both the two maps and the value of `split`.
    pub fn split_lookup(&self, split: &K) -> (Self, Option<Arc<V>>, Self) {
        self.iter().fold(
            (ordmap![], None, ordmap![]),
            |(l, m, r), (k, v)| match k.as_ref().cmp(split) {
                Ordering::Less => (l.insert(k, v), m, r),
                Ordering::Equal => (l, Some(v), r),
                Ordering::Greater => (l, m, r.insert(k, v)),
            },
        )
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the same values.
    ///
    /// Use the provided function to decide whether values are equal.
    pub fn is_submap_by<B, RM, F>(&self, other: RM, cmp: F) -> bool
    where
        F: Fn(Arc<V>, Arc<B>) -> bool,
        RM: Borrow<OrdMap<K, B>>,
    {
        self.iter().all(|(k, v)| {
            other
                .borrow()
                .get(&*k)
                .map(|ov| cmp(v, ov))
                .unwrap_or(false)
        })
    }

    /// Test whether a map is a proper submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the same values.
    /// To be a proper submap, ours must also contain fewer keys than the other map.
    ///
    /// Use the provided function to decide whether values are equal.
    pub fn is_proper_submap_by<B, RM, F>(&self, other: RM, cmp: F) -> bool
    where
        F: Fn(Arc<V>, Arc<B>) -> bool,
        RM: Borrow<OrdMap<K, B>>,
    {
        self.len() != other.borrow().len() && self.is_submap_by(other, cmp)
    }

    /// Construct a map with only the `n` smallest keys from a given map.
    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).collect()
    }

    /// Construct a map with the `n` smallest keys removed from a given map.
    pub fn drop(&self, n: usize) -> Self {
        self.iter().skip(n).collect()
    }

    /// Remove the smallest key from a map, and return its value as well as the updated map.
    pub fn pop_min(&self) -> (Option<Arc<V>>, Self) {
        let (pop, next) = self.pop_min_with_key();
        (pop.map(|(_, v)| v), next)
    }

    /// Remove the smallest key from a map, and return that key, its value
    /// as well as the updated map.
    pub fn pop_min_with_key(&self) -> (Option<(Arc<K>, Arc<V>)>, Self) {
        match self.get_min() {
            None => (None, self.clone()),
            Some((k, v)) => (Some((k.clone(), v)), self.remove(&*k)),
        }
    }

    /// Remove the largest key from a map, and return its value as well as the updated map.
    pub fn pop_max(&self) -> (Option<Arc<V>>, Self) {
        let (pop, next) = self.pop_max_with_key();
        (pop.map(|(_, v)| v), next)
    }

    /// Remove the largest key from a map, and return that key, its value
    /// as well as the updated map.
    pub fn pop_max_with_key(&self) -> (Option<(Arc<K>, Arc<V>)>, Self) {
        match self.get_max() {
            None => (None, self.clone()),
            Some((k, v)) => (Some((k.clone(), v)), self.remove(&*k)),
        }
    }

    /// Discard the smallest key from a map, returning the updated map.
    pub fn delete_min(&self) -> Self {
        self.pop_min().1
    }

    /// Discard the largest key from a map, returning the updated map.
    pub fn delete_max(&self) -> Self {
        self.pop_max().1
    }
}

impl<K: Ord, V: PartialEq> OrdMap<K, V> {
    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the same values.
    pub fn is_submap<RM>(&self, other: RM) -> bool
    where
        RM: Borrow<Self>,
    {
        self.is_submap_by(other.borrow(), |a, b| a.as_ref().eq(b.as_ref()))
    }

    /// Test whether a map is a proper submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the same values.
    /// To be a proper submap, ours must also contain fewer keys than the other map.
    pub fn is_proper_submap<RM>(&self, other: RM) -> bool
    where
        RM: Borrow<Self>,
    {
        self.is_proper_submap_by(other.borrow(), |a, b| a.as_ref().eq(b.as_ref()))
    }
}

// Core traits

impl<K, V> Clone for OrdMap<K, V> {
    fn clone(&self) -> Self {
        OrdMap(self.0.clone())
    }
}

#[cfg(not(has_specialisation))]
impl<K: Ord + PartialEq, V: PartialEq> PartialEq for OrdMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<K: PartialEq, V: PartialEq> PartialEq for OrdMap<K, V>
where
    K: Ord,
{
    default fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

#[cfg(has_specialisation)]
impl<K: Ord + Eq, V: Eq> PartialEq for OrdMap<K, V>
where
    K: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0) || (self.len() == other.len() && self.iter().eq(other.iter()))
    }
}

impl<K: Ord + Eq, V: Eq> Eq for OrdMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for OrdMap<K, V>
where
    K: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for OrdMap<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: Ord + Hash, V: Hash> Hash for OrdMap<K, V> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<K, V> Default for OrdMap<K, V> {
    fn default() -> Self {
        ordmap![]
    }
}

impl<'a, K: Ord, V> Add for &'a OrdMap<K, V> {
    type Output = OrdMap<K, V>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<K: Debug, V: Debug> Debug for OrdMap<K, V>
where
    K: Ord,
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

// Iterators

pub struct Keys<K, V> {
    it: Iter<K, V>,
}

impl<K, V> Iterator for Keys<K, V>
where
    K: Ord,
{
    type Item = Arc<K>;

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

impl<K, V> DoubleEndedIterator for Keys<K, V>
where
    K: Ord,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            None => None,
            Some((k, _)) => Some(k),
        }
    }
}

impl<K: Ord, V> ExactSizeIterator for Keys<K, V> {}

pub struct Values<K, V> {
    it: Iter<K, V>,
}

impl<K, V> Iterator for Values<K, V>
where
    K: Ord,
{
    type Item = Arc<V>;

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

impl<K, V> DoubleEndedIterator for Values<K, V>
where
    K: Ord,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            None => None,
            Some((_, v)) => Some(v),
        }
    }
}

impl<K: Ord, V> ExactSizeIterator for Values<K, V> {}

impl<K: Ord, V, RK, RV> FromIterator<(RK, RV)> for OrdMap<K, V>
where
    RK: Shared<K>,
    RV: Shared<V>,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = (RK, RV)>,
    {
        let mut m = OrdMap::default();
        for (k, v) in i {
            m.insert_mut(k, v);
        }
        m
    }
}

impl<'a, K, V> IntoIterator for &'a OrdMap<K, V>
where
    K: Ord,
{
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> IntoIterator for OrdMap<K, V>
where
    K: Ord,
{
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

pub struct OrdMapLens<K, V> {
    key: Arc<K>,
    value: PhantomData<V>,
}

impl<K, V> Clone for OrdMapLens<K, V> {
    fn clone(&self) -> Self {
        OrdMapLens {
            key: self.key.clone(),
            value: PhantomData,
        }
    }
}

impl<K, V> PartialLens for OrdMapLens<K, V>
where
    K: Ord,
{
    type From = OrdMap<K, V>;
    type To = V;

    fn try_get(&self, s: &Self::From) -> Option<Arc<Self::To>> {
        s.get(&self.key)
    }

    fn try_put<Convert>(&self, cv: Option<Convert>, s: &Self::From) -> Option<Self::From>
    where
        Convert: Shared<Self::To>,
    {
        Some(match cv.map(Shared::shared) {
            None => s.remove(&self.key),
            Some(v) => s.insert(self.key.clone(), v),
        })
    }
}

impl<K, V> AsRef<OrdMap<K, V>> for OrdMap<K, V> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<'a, K: Ord, V: Clone, RK, RV> From<&'a [(RK, RV)]> for OrdMap<K, V>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
{
    fn from(m: &'a [(RK, RV)]) -> OrdMap<K, V> {
        m.into_iter()
            .map(|&(ref k, ref v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Ord, V, RK, RV> From<Vec<(RK, RV)>> for OrdMap<K, V>
where
    RK: Shared<K>,
    RV: Shared<V>,
{
    fn from(m: Vec<(RK, RV)>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<'a, K: Ord, V, RK, RV> From<&'a Vec<(RK, RV)>> for OrdMap<K, V>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
{
    fn from(m: &'a Vec<(RK, RV)>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|&(ref k, ref v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Ord, V, RK: Eq + Hash, RV> From<collections::HashMap<RK, RV>> for OrdMap<K, V>
where
    RK: Shared<K>,
    RV: Shared<V>,
{
    fn from(m: collections::HashMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<'a, K: Ord, V, RK: Eq + Hash, RV> From<&'a collections::HashMap<RK, RV>> for OrdMap<K, V>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
{
    fn from(m: &'a collections::HashMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Ord, V, RK, RV> From<collections::BTreeMap<RK, RV>> for OrdMap<K, V>
where
    RK: Shared<K>,
    RV: Shared<V>,
{
    fn from(m: collections::BTreeMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<'a, K: Ord, V, RK, RV> From<&'a collections::BTreeMap<RK, RV>> for OrdMap<K, V>
where
    &'a RK: Shared<K>,
    &'a RV: Shared<V>,
{
    fn from(m: &'a collections::BTreeMap<RK, RV>) -> OrdMap<K, V> {
        m.into_iter()
            .map(|(k, v)| (k.shared(), v.shared()))
            .collect()
    }
}

impl<K: Ord, V, S> From<HashMap<K, V, S>> for OrdMap<K, V> {
    fn from(m: HashMap<K, V, S>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord, V, S> From<&'a HashMap<K, V, S>> for OrdMap<K, V> {
    fn from(m: &'a HashMap<K, V, S>) -> Self {
        m.into_iter().collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<K: Ord + Arbitrary + Sync, V: Arbitrary + Sync> Arbitrary for OrdMap<K, V> {
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
    ) -> BoxedStrategy<OrdMap<<K::Value as ValueTree>::Value, <V::Value as ValueTree>::Value>>
    where
        <K::Value as ValueTree>::Value: Ord,
    {
        ::proptest::collection::vec((key, value), size.clone())
            .prop_map(|v| OrdMap::from(v))
            .prop_filter("OrdMap minimum size".to_owned(), move |m| {
                m.len() >= size.start
            })
            .boxed()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use super::proptest::*;
    use test::is_sorted;
    use conslist::ConsList;
    use proptest::collection;
    use proptest::num::{usize, i16};

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
        assert_eq!(it.next(), Some((Arc::new(1), Arc::new(11))));
        assert_eq!(it.next(), Some((Arc::new(2), Arc::new(22))));
        assert_eq!(it.next(), Some((Arc::new(3), Arc::new(33))));
        assert_eq!(it.next(), Some((Arc::new(4), Arc::new(44))));
        assert_eq!(it.next(), Some((Arc::new(5), Arc::new(55))));
        assert_eq!(it.next(), Some((Arc::new(6), Arc::new(66))));
        assert_eq!(it.next(), Some((Arc::new(7), Arc::new(77))));
        assert_eq!(it.next(), Some((Arc::new(8), Arc::new(88))));
        assert_eq!(it.next(), Some((Arc::new(9), Arc::new(99))));
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
            assert_eq!(*k * 11, *v);
            vec.push(*k)
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
        assert_eq!(map.pop(&11), None);
        let (popped, less) = map.pop(&5).unwrap();
        assert_eq!(popped, Arc::new(55));
        let mut it = less.iter();
        assert_eq!(it.next(), Some((Arc::new(1), Arc::new(11))));
        assert_eq!(it.next(), Some((Arc::new(2), Arc::new(22))));
        assert_eq!(it.next(), Some((Arc::new(3), Arc::new(33))));
        assert_eq!(it.next(), Some((Arc::new(4), Arc::new(44))));
        assert_eq!(it.next(), Some((Arc::new(6), Arc::new(66))));
        assert_eq!(it.next(), Some((Arc::new(7), Arc::new(77))));
        assert_eq!(it.next(), Some((Arc::new(8), Arc::new(88))));
        assert_eq!(it.next(), Some((Arc::new(9), Arc::new(99))));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn debug_output() {
        assert_eq!(
            format!("{:?}", ordmap!{ 3 => 4, 5 => 6, 1 => 2 }),
            "{ 1 => 2, 3 => 4, 5 => 6 }"
        );
    }

    #[test]
    fn equality2() {
        let v1 = "1".to_string();
        let v2 = "1".to_string();
        assert_eq!(v1, v2);
        let p1 = ConsList::<String>::new();
        let p2 = ConsList::<String>::new();
        assert_eq!(p1, p2);
        let c1 = OrdMap::singleton(v1, p1);
        let c2 = OrdMap::singleton(v2, p2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn insert_remove_single_mut() {
        let mut m = OrdMap::new();
        m.insert_mut(0, 0);
        assert_eq!(OrdMap::singleton(0, 0), m);
        m.remove_mut(&0);
        assert_eq!(OrdMap::new(), m);
    }

    #[test]
    fn double_ended_iterator_1() {
        let m = ordmap![1 => 1, 2 => 2, 3 => 3, 4 => 4];
        let mut it = m.iter();
        assert_eq!(Some((Arc::new(1), Arc::new(1))), it.next());
        assert_eq!(Some((Arc::new(4), Arc::new(4))), it.next_back());
        assert_eq!(Some((Arc::new(2), Arc::new(2))), it.next());
        assert_eq!(Some((Arc::new(3), Arc::new(3))), it.next_back());
        assert_eq!(None, it.next());
    }

    #[test]
    fn double_ended_iterator_2() {
        let m = ordmap![1 => 1, 2 => 2, 3 => 3, 4 => 4];
        let mut it = m.iter();
        assert_eq!(Some((Arc::new(1), Arc::new(1))), it.next());
        assert_eq!(Some((Arc::new(4), Arc::new(4))), it.next_back());
        assert_eq!(Some((Arc::new(2), Arc::new(2))), it.next());
        assert_eq!(Some((Arc::new(3), Arc::new(3))), it.next_back());
        assert_eq!(None, it.next_back());
    }

    quickcheck! {
        fn length(input: Vec<i32>) -> bool {
            let mut vec = input;
            vec.sort();
            vec.dedup();
            let map = OrdMap::from_iter(vec.iter().map(|i| (i, i)));
            vec.len() == map.len()
        }

        fn order(vec: Vec<(i32, i32)>) -> bool {
            let map = OrdMap::from_iter(vec.into_iter());
            is_sorted(map.keys().map(|k| *k))
        }

        fn overwrite_values(vec: Vec<(usize, usize)>, index_rand: usize, new_val: usize) -> bool {
            if vec.len() == 0 {
                return true
            }
            let index = vec[index_rand % vec.len()].0;
            let map1 = OrdMap::from_iter(vec.clone());
            let map2 = map1.insert(index, new_val);
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
            if vec.len() == 0 {
                return true
            }
            let index = vec[index_rand % vec.len()].0;
            let map1 = OrdMap::from_iter(vec.clone());
            let map2 = map1.remove(&index);
            map2.keys().all(|k| *k != index) && map1.len() == map2.len() + 1
        }

        fn insert_and_delete_values(
            input_unbounded: Vec<(usize, usize)>, ops: Vec<(bool, usize, usize)>
        ) -> bool {
            let input: Vec<(usize, usize)> =
                input_unbounded.into_iter().map(|(k, v)| (k % 64, v % 64)).collect();
            let mut map = OrdMap::from(input.clone());
            let mut tree: collections::BTreeMap<usize, usize> = input.into_iter().collect();
            for (ins, key, val) in ops {
                if ins {
                    tree.insert(key, val);
                    map = map.insert(key, val)
                } else {
                    tree.remove(&key);
                    map = map.remove(&key)
                }
            }
            map.iter().map(|(k, v)| (*k, *v)).eq(tree.iter().map(|(k, v)| (*k, *v)))
        }
    }

    proptest! {
        #[test]
        fn proptest_works(ref m in ord_map(0..9999, ".*", 10..100)) {
            assert!(m.len() < 100);
            assert!(m.len() >= 10);
        }

        #[test]
        fn insert_and_length(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let mut map: OrdMap<i16, i16> = OrdMap::new();
            for (k, v) in m.iter() {
                map = map.insert(*k, *v)
            }
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn from_iterator(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn iterate_over(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let map: OrdMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.iter().count());
        }

        #[test]
        fn equality(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..128)) {
            let map1: OrdMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let map2: OrdMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(map1, map2);
        }

        #[test]
        fn lookup(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let map: OrdMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for (k, v) in m.into_iter() {
                assert_eq!(Some(*v), map.get(k).map(|v| *v));
            }
        }

        #[test]
        fn remove(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let mut map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).map(|v| *v), map.get(k).map(|v| *v));
                map = map.remove(&k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn insert_mut(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let mut mut_map = OrdMap::new();
            let mut map = OrdMap::new();
            for (k, v) in m.iter() {
                map = map.insert(*k, *v);
                mut_map.insert_mut(*k, *v);
            }
            assert_eq!(map, mut_map);
        }

        #[test]
        fn remove_mut(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..64)) {
            let mut map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).map(|v| *v), map.get(k).map(|v| *v));
                map.remove_mut(&k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        // Failing: ([(7, 0), (3, 0), (8, 0), (9, 0), (0, 0), (4, 0), (0, 0), (1, 0),
        //  (2, 0), (5, 0), (0, 0), (10, 0), (0, 0), (6, 0), (11, 0), (0, 0), (12, 0),
        //  (13, 0), (14, 0), (15, 0)], 48)
        #[test]
        fn delete_and_reinsert(ref input in collection::hash_map(i16::ANY, i16::ANY, 1..100),
                               index_rand in usize::ANY) {
            let index = input.keys().skip(index_rand % input.len()).next().unwrap().clone();
            let map1 = OrdMap::from_iter(input.clone());
            let (val, map2) = map1.pop(&index).unwrap();
            let map3 = map2.insert(index, val);
            for key in map2.keys() {
                assert!(*key != index);
            }
            assert_eq!(map1.len(), map2.len() + 1);
            assert_eq!(map1, map3);
        }

        #[test]
        fn exact_size_iterator(ref m in ord_map(i16::ANY, i16::ANY, 1..100)) {
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
