//! # Ordered Map
//!
//! An immutable ordered map implemented as a balanced 2-3 tree.
//!
//! Most operations on this type of map are O(log n). It's a decent
//! choice for a generic map datatype, but if you're using it for
//! large datasets, you should consider whether you need an ordered
//! map, or whether a hash map would suit you better.

mod walk;

use std::sync::Arc;
use std::iter::{Iterator, FromIterator};
use std::collections::{BTreeMap, HashMap};
use std::cmp::Ordering;
use std::hash::Hash;
use std::fmt::{Debug, Formatter, Error};
use std::ops::Add;

use self::MapNode::{Leaf, Two, Three};



#[macro_export]
macro_rules! map {
    () => { $crate::map::Map::empty() };

    ( $( $key:expr => $value:expr ),* ) => {{
        let mut map = $crate::map::Map::empty();
        $({
            map = map.insert($key, $value);
        })*;
        map
    }};
}

pub struct Map<K, V>(Arc<MapNode<K, V>>);

pub enum MapNode<K, V> {
    Leaf,
    Two(Map<K, V>, Arc<K>, Arc<V>, Map<K, V>),
    Three(Map<K, V>, Arc<K>, Arc<V>, Map<K, V>, Arc<K>, Arc<V>, Map<K, V>),
}

impl<K, V> Map<K, V> {
    pub fn empty() -> Map<K, V> {
        Map(Arc::new(Leaf))
    }

    pub fn singleton(k: K, v: V) -> Map<K, V> {
        Map::two(map![], Arc::new(k), Arc::new(v), map![])
    }

    pub fn null(&self) -> bool {
        match *self.0 {
            Leaf => true,
            _ => false,
        }
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self)
    }

    pub fn keys(&self) -> Keys<K, V> {
        Keys { it: self.iter() }
    }

    pub fn values(&self) -> Values<K, V> {
        Values { it: self.iter() }
    }

    pub fn size(&self) -> usize {
        self.iter().count()
    }

    pub fn lookup_max(&self) -> Option<(Arc<K>, Arc<V>)> {
        match *self.0 {
            Leaf => None,
            Two(_, ref k1, ref v1, ref right) => {
                Some(right.lookup_max().unwrap_or((k1.clone(), v1.clone())))
            }
            Three(_, _, _, _, ref k2, ref v2, ref right) => {
                Some(right.lookup_max().unwrap_or((k2.clone(), v2.clone())))
            }
        }
    }

    pub fn lookup_min(&self) -> Option<(Arc<K>, Arc<V>)> {
        match *self.0 {
            Leaf => None,
            Two(ref left, ref k1, ref v1, _) => {
                Some(left.lookup_min().unwrap_or((k1.clone(), v1.clone())))
            }
            Three(ref left, ref k1, ref v1, _, _, _, _) => {
                Some(left.lookup_min().unwrap_or((k1.clone(), v1.clone())))
            }
        }
    }

    fn two(left: Map<K, V>, k: Arc<K>, v: Arc<V>, right: Map<K, V>) -> Map<K, V> {
        Map(Arc::new(Two(left, k, v, right)))
    }

    fn three(left: Map<K, V>,
             k1: Arc<K>,
             v1: Arc<V>,
             mid: Map<K, V>,
             k2: Arc<K>,
             v2: Arc<V>,
             right: Map<K, V>)
             -> Map<K, V> {
        Map(Arc::new(Three(left, k1, v1, mid, k2, v2, right)))
    }

    fn all_heights(&self) -> Vec<usize> {
        match *self.0 {
            Leaf => vec![0],
            Two(ref left, _, _, ref right) => {
                left.all_heights()
                    .iter()
                    .chain(right.all_heights().iter())
                    .map(|i| i + 1)
                    .collect()
            }
            Three(ref left, _, _, ref mid, _, _, ref right) => {
                left.all_heights()
                    .iter()
                    .chain(mid.all_heights().iter())
                    .chain(right.all_heights().iter())
                    .map(|i| i + 1)
                    .collect()
            }
        }
    }

    pub fn valid(&self) -> bool {
        all_eq(self.all_heights())
    }
}

fn all_eq<A, I>(i: I) -> bool
    where I: IntoIterator<Item = A>,
          A: PartialEq
{
    let mut it = i.into_iter();
    match it.next() {
        None => true,
        Some(ref a) => it.all(|ref b| a == b),
    }
}

impl<K: Clone, V: Clone> Map<K, V> {
    pub fn clone_iter(&self) -> Cloned<K, V> {
        Cloned { it: self.iter() }
    }
}

impl<K: Ord, V> Map<K, V> {
    pub fn lookup(&self, k: &K) -> Option<Arc<V>> {
        walk::lookup(self, k)
    }

    pub fn lookup_or(&self, k: &K, default: V) -> Arc<V> {
        self.lookup(k).unwrap_or(Arc::new(default))
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.lookup(k).is_some()
    }

    pub fn insert(&self, k: K, v: V) -> Self {
        walk::ins_down(list![], Arc::new(k), Arc::new(v), self.clone())
    }

    pub fn insert_ref(&self, k: Arc<K>, v: Arc<V>) -> Self {
        walk::ins_down(list![], k, v, self.clone())
    }

    pub fn insert_with<F>(self, k: K, v: V, f: F) -> Self
        where F: Fn(Arc<V>, Arc<V>) -> Arc<V>
    {
        match self.pop_with_key(&k) {
            None => self.insert(k, v),
            Some((k, v2, m)) => m.insert_ref(k, f(Arc::new(v), v2)),
        }
    }

    pub fn insert_with_key<F>(self, k: K, v: V, f: F) -> Self
        where F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>
    {
        match self.pop_with_key(&k) {
            None => self.insert(k, v),
            Some((k, v2, m)) => m.insert_ref(k.clone(), f(k, Arc::new(v), v2)),
        }
    }

    pub fn insert_lookup_with_key<F>(self, k: K, v: V, f: F) -> (Option<Arc<V>>, Self)
        where F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>
    {
        match self.pop_with_key(&k) {
            None => (None, self.insert(k, v)),
            Some((k, v2, m)) => (Some(v2.clone()), m.insert_ref(k.clone(), f(k, Arc::new(v), v2))),
        }
    }

    pub fn update<F>(&self, k: &K, f: F) -> Self
        where F: Fn(Arc<V>) -> Option<Arc<V>>
    {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((k, v, m)) => {
                match f(v) {
                    None => m,
                    Some(v) => m.insert_ref(k, v),
                }
            }
        }
    }

    pub fn update_with_key<F>(&self, k: &K, f: F) -> Self
        where F: Fn(Arc<K>, Arc<V>) -> Option<Arc<V>>
    {
        match self.pop_with_key(k) {
            None => self.clone(),
            Some((k, v, m)) => {
                match f(k.clone(), v) {
                    None => m,
                    Some(v) => m.insert_ref(k, v),
                }
            }
        }
    }

    pub fn update_lookup_with_key<F>(&self, k: &K, f: F) -> (Option<Arc<V>>, Self)
        where F: Fn(Arc<K>, Arc<V>) -> Option<Arc<V>>
    {
        match self.pop_with_key(k) {
            None => (None, self.clone()),
            Some((k, v, m)) => {
                match f(k.clone(), v.clone()) {
                    None => (Some(v), m),
                    Some(v) => (Some(v.clone()), m.insert_ref(k, v)),
                }
            }
        }
    }

    pub fn alter<F>(&self, f: F, k: Arc<K>) -> Self
        where F: Fn(Option<Arc<V>>) -> Option<Arc<V>>
    {
        let pop = self.pop_with_key(&*k);
        match (f(pop.as_ref().map(|&(_, ref v, _)| v.clone())), pop) {
            (None, None) => self.clone(),
            (Some(v), None) => self.insert_ref(k, v),
            (None, Some((_, _, m))) => m,
            (Some(v), Some((_, _, m))) => m.insert_ref(k, v),
        }
    }

    pub fn delete(&self, k: &K) -> Self {
        self.pop(k).map(|(_, m)| m).unwrap_or_else(|| self.clone())
    }

    pub fn pop(&self, k: &K) -> Option<(Arc<V>, Self)> {
        self.pop_with_key(k).map(|(_, v, m)| (v, m))
    }

    pub fn pop_with_key(&self, k: &K) -> Option<(Arc<K>, Arc<V>, Self)> {
        walk::pop_down(list![], k, self.clone())
    }

    pub fn union(&self, other: &Self) -> Self {
        self.union_with_key(other, |_, v, _| v)
    }

    pub fn union_with<F>(&self, other: &Self, f: F) -> Self
        where F: Fn(Arc<V>, Arc<V>) -> Arc<V>
    {
        self.union_with_key(other, |_, v1, v2| f(v1, v2))
    }

    pub fn union_with_key<F>(&self, other: &Self, f: F) -> Self
        where F: Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>
    {
        other
            .iter()
            .fold(self.clone(), |m, (k, v)| {
                m.insert_ref(k.clone(),
                             self.lookup(&*k).map(|v1| f(k, v1, v.clone())).unwrap_or(v))
            })
    }

    pub fn unions<I>(i: I) -> Self
        where I: IntoIterator<Item = Self>
    {
        i.into_iter().fold(map![], |a, b| a.union(&b))
    }

    pub fn unions_with<I, F>(i: I, f: &F) -> Self
        where I: IntoIterator<Item = Self>,
              F: Fn(Arc<V>, Arc<V>) -> Arc<V>
    {
        i.into_iter().fold(map![], |a, b| a.union_with(&b, f))
    }

    pub fn unions_with_key<I>(i: I, f: &Fn(Arc<K>, Arc<V>, Arc<V>) -> Arc<V>) -> Self
        where I: IntoIterator<Item = Self>
    {
        i.into_iter()
            .fold(map![], |a, b| a.union_with_key(&b, f))
    }

    pub fn difference<B>(&self, other: &Map<K, B>) -> Self {
        self.difference_with_key(other, |_, _, _| None)
    }

    pub fn difference_with<B, F>(&self, other: &Map<K, B>, f: F) -> Self
        where F: Fn(Arc<V>, Arc<B>) -> Option<Arc<V>>
    {
        self.difference_with_key(other, |_, a, b| f(a, b))
    }

    pub fn difference_with_key<B, F>(&self, other: &Map<K, B>, f: F) -> Self
        where F: Fn(Arc<K>, Arc<V>, Arc<B>) -> Option<Arc<V>>
    {
        other
            .iter()
            .fold(self.clone(), |m, (k, v2)| match m.pop(&*k) {
                None => m,
                Some((v1, m)) => {
                    match f(k.clone(), v1, v2) {
                        None => m,
                        Some(v) => m.insert_ref(k, v),
                    }
                }
            })
    }

    pub fn intersection<B>(&self, other: &Map<K, B>) -> Self {
        self.intersection_with_key(other, |_, v, _| v)
    }

    pub fn intersection_with<B, C, F>(&self, other: &Map<K, B>, f: F) -> Map<K, C>
        where F: Fn(Arc<V>, Arc<B>) -> Arc<C>
    {
        self.intersection_with_key(other, |_, v1, v2| f(v1, v2))
    }

    pub fn intersection_with_key<B, C, F>(&self, other: &Map<K, B>, f: F) -> Map<K, C>
        where F: Fn(Arc<K>, Arc<V>, Arc<B>) -> Arc<C>
    {
        other
            .iter()
            .fold(map![], |m, (k, v2)| {
                self.lookup(&*k)
                    .map(|v1| m.insert_ref(k.clone(), f(k, v1, v2)))
                    .unwrap_or(m)
            })
    }

    pub fn merge_with_key<B, C, FC, F1, F2>(&self,
                                            other: &Map<K, B>,
                                            combine: FC,
                                            only1: F1,
                                            only2: F2)
                                            -> Map<K, C>
        where FC: Fn(Arc<K>, Arc<V>, Arc<B>) -> Option<Arc<C>>,
              F1: Fn(Self) -> Map<K, C>,
              F2: Fn(Map<K, B>) -> Map<K, C>
    {
        let (left, right, both) = other
            .iter()
            .fold((self.clone(), other.clone(), map![]),
                  |(l, r, m), (k, vr)| match l.pop(&*k) {
                      None => (l, r, m),
                      Some((vl, ml)) => {
                          (ml,
                           r.delete(&*k),
                           combine(k.clone(), vl, vr)
                               .map(|v| m.insert_ref(k, v))
                               .unwrap_or(m))
                      }
                  });
        both.union(&only1(left)).union(&only2(right))
    }

    pub fn split(&self, split: &K) -> (Self, Self) {
        let (l, _, r) = self.split_lookup(split);
        (l, r)
    }

    pub fn split_lookup(&self, split: &K) -> (Self, Option<Arc<V>>, Self) {
        self.iter()
            .fold((map![], None, map![]),
                  |(l, m, r), (k, v)| match k.as_ref().cmp(split) {
                      Ordering::Less => (l.insert_ref(k, v), m, r),
                      Ordering::Equal => (l, Some(v), r),
                      Ordering::Greater => (l, m, r.insert_ref(k, v)),
                  })
    }

    pub fn is_submap_by<B, F>(&self, other: &Map<K, B>, cmp: F) -> bool
        where F: Fn(Arc<V>, Arc<B>) -> bool
    {
        self.iter()
            .all(|(k, v)| other.lookup(&*k).map(|ov| cmp(v, ov)).unwrap_or(false))
    }

    pub fn is_proper_submap_by<B, F>(&self, other: &Map<K, B>, cmp: F) -> bool
        where F: Fn(Arc<V>, Arc<B>) -> bool
    {
        self.size() != other.size() && self.is_submap_by(other, cmp)
    }

    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).collect()
    }

    pub fn drop(&self, n: usize) -> Self {
        self.iter().skip(n).collect()
    }

    pub fn pop_min(&self) -> (Option<Arc<V>>, Self) {
        let (pop, next) = self.pop_min_with_key();
        (pop.map(|(_, v)| v), next)
    }

    pub fn pop_min_with_key(&self) -> (Option<(Arc<K>, Arc<V>)>, Self) {
        match self.lookup_min() {
            None => (None, self.clone()),
            Some((k, v)) => (Some((k.clone(), v)), self.delete(&*k)),
        }
    }

    pub fn pop_max(&self) -> (Option<Arc<V>>, Self) {
        let (pop, next) = self.pop_max_with_key();
        (pop.map(|(_, v)| v), next)
    }

    pub fn pop_max_with_key(&self) -> (Option<(Arc<K>, Arc<V>)>, Self) {
        match self.lookup_max() {
            None => (None, self.clone()),
            Some((k, v)) => (Some((k.clone(), v)), self.delete(&*k)),
        }
    }

    pub fn delete_min(&self) -> Self {
        self.pop_min().1
    }

    pub fn delete_max(&self) -> Self {
        self.pop_max().1
    }
}

impl<K: Ord, V: PartialEq> Map<K, V> {
    pub fn is_submap(&self, other: &Self) -> bool {
        self.is_submap_by(other, |a, b| a.as_ref().eq(b.as_ref()))
    }

    pub fn is_proper_submap(&self, other: &Self) -> bool {
        self.is_proper_submap_by(other, |a, b| a.as_ref().eq(b.as_ref()))
    }
}

// Core traits

impl<K, V> Clone for Map<K, V> {
    fn clone(&self) -> Self {
        Map(self.0.clone())
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<K: Eq, V: Eq> Eq for Map<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for Map<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for Map<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, V> Default for Map<K, V> {
    fn default() -> Self {
        map![]
    }
}

impl<'a, K: Ord, V> Add for &'a Map<K, V> {
    type Output = Map<K, V>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<K: Debug, V: Debug> Debug for Map<K, V> {
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

pub enum IterItem<K, V> {
    Consider(Map<K, V>),
    Yield(Arc<K>, Arc<V>),
}

pub enum IterResult<K, V> {
    Next(Arc<K>, Arc<V>),
    Walk,
    Done,
}

pub struct Iter<K, V> {
    stack: Vec<IterItem<K, V>>,
}

impl<K, V> Iter<K, V> {
    fn new(m: &Map<K, V>) -> Iter<K, V> {
        Iter { stack: vec![IterItem::Consider(m.clone())] }
    }

    fn step(&mut self) -> IterResult<K, V> {
        match self.stack.pop() {
            None => IterResult::Done,
            Some(IterItem::Consider(m)) => {
                match *m.0 {
                    Leaf => return IterResult::Walk,
                    Two(ref left, ref k, ref v, ref right) => {
                        self.stack.push(IterItem::Consider(right.clone()));
                        self.stack.push(IterItem::Yield(k.clone(), v.clone()));
                        self.stack.push(IterItem::Consider(left.clone()));
                        IterResult::Walk
                    }
                    Three(ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => {
                        self.stack.push(IterItem::Consider(right.clone()));
                        self.stack.push(IterItem::Yield(k2.clone(), v2.clone()));
                        self.stack.push(IterItem::Consider(mid.clone()));
                        self.stack.push(IterItem::Yield(k1.clone(), v1.clone()));
                        self.stack.push(IterItem::Consider(left.clone()));
                        IterResult::Walk
                    }
                }
            }
            Some(IterItem::Yield(k, v)) => IterResult::Next(k, v),
        }
    }
}

impl<K, V> Iterator for Iter<K, V> {
    type Item = (Arc<K>, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut action = IterResult::Walk;
        loop {
            match action {
                IterResult::Walk => action = self.step(),
                IterResult::Done => return None,
                IterResult::Next(k, v) => return Some((k, v)),
            }
        }
    }
}

pub struct Cloned<K, V> {
    it: Iter<K, V>,
}

impl<K: Clone, V: Clone> Iterator for Cloned<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.it.next() {
            None => None,
            Some((k, v)) => Some(((*k).clone(), (*v).clone())),
        }
    }
}

pub struct Keys<K, V> {
    it: Iter<K, V>,
}

impl<K, V> Iterator for Keys<K, V> {
    type Item = Arc<K>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.it.next() {
            None => None,
            Some((k, _)) => Some(k.clone()),
        }
    }
}

pub struct Values<K, V> {
    it: Iter<K, V>,
}

impl<K, V> Iterator for Values<K, V> {
    type Item = Arc<V>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.it.next() {
            None => None,
            Some((_, v)) => Some(v.clone()),
        }
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<T>(i: T) -> Self
        where T: IntoIterator<Item = (K, V)>
    {
        i.into_iter().fold(map![], |m, (k, v)| m.insert(k, v))
    }
}

impl<K: Ord, V> FromIterator<(Arc<K>, Arc<V>)> for Map<K, V> {
    fn from_iter<T>(i: T) -> Self
        where T: IntoIterator<Item = (Arc<K>, Arc<V>)>
    {
        i.into_iter()
            .fold(map![], |m, (k, v)| m.insert_ref(k, v))
    }
}

impl<'a, K, V> IntoIterator for &'a Map<K, V> {
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> IntoIterator for Map<K, V> {
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<'a, K: Ord + Clone, V: Clone> From<&'a [(K, V)]> for Map<K, V> {
    fn from(m: &'a [(K, V)]) -> Map<K, V> {
        m.into_iter().map(|&(ref k, ref v)| (k.clone(), v.clone())).collect()
    }
}

impl<K: Eq + Hash + Ord, V> From<HashMap<K, V>> for Map<K, V> {
    fn from(m: HashMap<K, V>) -> Map<K, V> {
        m.into_iter().collect()
    }
}

impl<'a, K: Eq + Hash + Ord + Clone, V: Clone> From<&'a HashMap<K, V>> for Map<K, V> {
    fn from(m: &'a HashMap<K, V>) -> Map<K, V> {
        m.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

impl<'a, K: Eq + Hash + Ord, V> From<&'a HashMap<Arc<K>, Arc<V>>> for Map<K, V> {
    fn from(m: &'a HashMap<Arc<K>, Arc<V>>) -> Map<K, V> {
        m.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

impl<K: Ord, V> From<BTreeMap<K, V>> for Map<K, V> {
    fn from(m: BTreeMap<K, V>) -> Map<K, V> {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord + Clone, V: Clone> From<&'a BTreeMap<K, V>> for Map<K, V> {
    fn from(m: &'a BTreeMap<K, V>) -> Map<K, V> {
        m.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

impl<'a, K: Ord, V> From<&'a BTreeMap<Arc<K>, Arc<V>>> for Map<K, V> {
    fn from(m: &'a BTreeMap<Arc<K>, Arc<V>>) -> Map<K, V> {
        m.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<K: Ord + Arbitrary + Sync, V: Arbitrary + Sync> Arbitrary for Map<K, V> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Map::from_iter(Vec::<(K, V)>::arbitrary(g))
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn iterates_in_order() {
        let map = map!{
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
        let map = map!{
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
        let map = map!{
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
        assert_eq!(format!("{:?}", map!{ 3 => 4, 5 => 6, 1 => 2 }),
                   "{ 1 => 2, 3 => 4, 5 => 6 }");
    }
}
