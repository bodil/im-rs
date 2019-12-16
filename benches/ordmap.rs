// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]

extern crate im;
extern crate rand;
extern crate test;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::BTreeSet;
use std::iter::FromIterator;
use test::Bencher;

use im::ordmap::OrdMap;

fn random_keys(size: usize) -> Vec<i64> {
    let mut gen = SmallRng::from_entropy();
    let mut set = BTreeSet::new();
    while set.len() < size {
        let next = gen.gen::<i64>() % 10000;
        set.insert(next);
    }
    set.into_iter().collect()
}

fn reorder<A: Copy>(vec: &[A]) -> Vec<A> {
    let mut gen = SmallRng::from_entropy();
    let mut set: Vec<A> = vec.to_owned();
    let mut out = Vec::new();
    while !set.is_empty() {
        let i = gen.gen::<usize>() % set.len();
        let v = set.remove(i);
        out.push(v)
    }
    out
}

fn ordmap_lookup_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let order = reorder(&keys);
    let m: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, 1)));
    b.iter(|| {
        for i in &order {
            let _ = m.get(i);
        }
    })
}

#[bench]
fn ordmap_lookup_10(b: &mut Bencher) {
    ordmap_lookup_n(10, b)
}

#[bench]
fn ordmap_lookup_100(b: &mut Bencher) {
    ordmap_lookup_n(100, b)
}

#[bench]
fn ordmap_lookup_1000(b: &mut Bencher) {
    ordmap_lookup_n(1000, b)
}

fn ordmap_insert_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = OrdMap::new();
        for i in keys.clone() {
            m = m.update(i, i)
        }
    })
}

#[bench]
fn ordmap_insert_10(b: &mut Bencher) {
    ordmap_insert_n(10, b)
}

#[bench]
fn ordmap_insert_100(b: &mut Bencher) {
    ordmap_insert_n(100, b)
}

#[bench]
fn ordmap_insert_1000(b: &mut Bencher) {
    ordmap_insert_n(1000, b)
}

fn ordmap_insert_mut_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = OrdMap::new();
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[bench]
fn ordmap_insert_mut_10(b: &mut Bencher) {
    ordmap_insert_mut_n(10, b)
}

#[bench]
fn ordmap_insert_mut_100(b: &mut Bencher) {
    ordmap_insert_mut_n(100, b)
}

#[bench]
fn ordmap_insert_mut_1000(b: &mut Bencher) {
    ordmap_insert_mut_n(1000, b)
}

#[bench]
fn ordmap_insert_mut_10000(b: &mut Bencher) {
    ordmap_insert_mut_n(10000, b)
}

#[cfg(feature = "pool")]
fn ordmap_insert_mut_pooled_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let pool = im::ordmap::OrdMapPool::new(16384);
    pool.fill();
    b.iter(|| {
        let mut m = OrdMap::with_pool(&pool);
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[cfg(feature = "pool")]
#[bench]
fn ordmap_insert_mut_pooled_10(b: &mut Bencher) {
    ordmap_insert_mut_pooled_n(10, b)
}

#[cfg(feature = "pool")]
#[bench]
fn ordmap_insert_mut_pooled_100(b: &mut Bencher) {
    ordmap_insert_mut_pooled_n(100, b)
}

#[cfg(feature = "pool")]
#[bench]
fn ordmap_insert_mut_pooled_1000(b: &mut Bencher) {
    ordmap_insert_mut_pooled_n(1000, b)
}

#[cfg(feature = "pool")]
#[bench]
fn ordmap_insert_mut_pooled_10000(b: &mut Bencher) {
    ordmap_insert_mut_pooled_n(10000, b)
}

fn ordmap_remove_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let order = reorder(&keys);
    let map: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| {
        let mut m = map.clone();
        for i in &order {
            m = m.without(i);
        }
    })
}

#[bench]
fn ordmap_remove_10(b: &mut Bencher) {
    ordmap_remove_n(10, b)
}

#[bench]
fn ordmap_remove_100(b: &mut Bencher) {
    ordmap_remove_n(100, b)
}

#[bench]
fn ordmap_remove_1000(b: &mut Bencher) {
    ordmap_remove_n(1000, b)
}

fn ordmap_remove_mut_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let order = reorder(&keys);
    let map: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| {
        let mut m = map.clone();
        for i in &order {
            m.remove(i);
        }
    })
}

#[bench]
fn ordmap_remove_mut_10(b: &mut Bencher) {
    ordmap_remove_mut_n(10, b)
}

#[bench]
fn ordmap_remove_mut_100(b: &mut Bencher) {
    ordmap_remove_mut_n(100, b)
}

#[bench]
fn ordmap_remove_mut_1000(b: &mut Bencher) {
    ordmap_remove_mut_n(1000, b)
}

#[bench]
fn ordmap_remove_min_1000(b: &mut Bencher) {
    let map: OrdMap<i64, i64> = OrdMap::from_iter((0..1000).map(|i| (i, i)));
    b.iter(|| {
        let mut m = map.clone();
        assert!(!m.is_empty());
        for _ in 0..1000 {
            m = m.without_min().1;
        }
        assert!(m.is_empty())
    })
}

#[bench]
fn ordmap_remove_max_1000(b: &mut Bencher) {
    let map: OrdMap<i64, i64> = OrdMap::from_iter((0..1000).map(|i| (i, i)));
    b.iter(|| {
        let mut m = map.clone();
        assert!(!m.is_empty());
        for _ in 0..1000 {
            m = m.without_max().1;
        }
        assert!(m.is_empty())
    })
}

fn ordmap_insert_once_n(size: usize, b: &mut Bencher) {
    let mut keys = random_keys(size + 1);
    let key = keys.pop().unwrap();
    let map: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| map.update(key, key))
}

#[bench]
fn ordmap_insert_once_10(b: &mut Bencher) {
    ordmap_insert_once_n(10, b)
}

#[bench]
fn ordmap_insert_once_100(b: &mut Bencher) {
    ordmap_insert_once_n(100, b)
}

#[bench]
fn ordmap_insert_once_1000(b: &mut Bencher) {
    ordmap_insert_once_n(1000, b)
}

#[bench]
fn ordmap_insert_once_10000(b: &mut Bencher) {
    ordmap_insert_once_n(10000, b)
}

fn ordmap_remove_once_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size + 1);
    let key = keys[0];
    let map: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| map.without(&key))
}

#[bench]
fn ordmap_remove_once_10(b: &mut Bencher) {
    ordmap_remove_once_n(10, b)
}

#[bench]
fn ordmap_remove_once_100(b: &mut Bencher) {
    ordmap_remove_once_n(100, b)
}

#[bench]
fn ordmap_remove_once_1000(b: &mut Bencher) {
    ordmap_remove_once_n(1000, b)
}

#[bench]
fn ordmap_remove_once_10000(b: &mut Bencher) {
    ordmap_remove_once_n(10000, b)
}

fn ordmap_lookup_once_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size + 1);
    let key = keys[0];
    let map: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| map.get(&key))
}

#[bench]
fn ordmap_lookup_once_10(b: &mut Bencher) {
    ordmap_lookup_once_n(10, b)
}

#[bench]
fn ordmap_lookup_once_100(b: &mut Bencher) {
    ordmap_lookup_once_n(100, b)
}

#[bench]
fn ordmap_lookup_once_1000(b: &mut Bencher) {
    ordmap_lookup_once_n(1000, b)
}

#[bench]
fn ordmap_lookup_once_10000(b: &mut Bencher) {
    ordmap_lookup_once_n(10000, b)
}

fn ordmap_iter(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let m: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, 1)));
    b.iter(|| for _ in m.iter() {})
}

#[bench]
fn ordmap_iter_10(b: &mut Bencher) {
    ordmap_iter(10, b)
}

#[bench]
fn ordmap_iter_100(b: &mut Bencher) {
    ordmap_iter(100, b)
}

#[bench]
fn ordmap_iter_1000(b: &mut Bencher) {
    ordmap_iter(1000, b)
}

#[bench]
fn ordmap_iter_10000(b: &mut Bencher) {
    ordmap_iter(10000, b)
}

fn ordmap_range_iter(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let m: OrdMap<i64, i64> = OrdMap::from_iter(keys.into_iter().map(|i| (i, 1)));
    b.iter(|| for _ in m.range(..) {})
}

#[bench]
fn ordmap_range_iter_10(b: &mut Bencher) {
    ordmap_range_iter(10, b)
}

#[bench]
fn ordmap_range_iter_100(b: &mut Bencher) {
    ordmap_range_iter(100, b)
}

#[bench]
fn ordmap_range_iter_1000(b: &mut Bencher) {
    ordmap_range_iter(1000, b)
}

#[bench]
fn ordmap_range_iter_10000(b: &mut Bencher) {
    ordmap_range_iter(10000, b)
}
