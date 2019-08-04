// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]

extern crate im;
extern crate rand;
extern crate test;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::iter::FromIterator;
use test::Bencher;

use im::hashmap::HashMap;

fn random_keys(size: usize) -> Vec<i64> {
    let mut gen = SmallRng::from_entropy();
    let mut set = Vec::new();
    while set.len() < size {
        let next = gen.gen::<i64>() % 10000;
        if !set.contains(&next) {
            set.push(next);
        }
    }
    set
}

fn reorder<A: Copy>(vec: &[A]) -> Vec<A> {
    let mut gen = SmallRng::from_entropy();
    let mut set = vec.to_owned();
    let mut out = Vec::new();
    while !set.is_empty() {
        let i = gen.gen::<usize>() % set.len();
        let v = set.remove(i);
        out.push(v)
    }
    out
}

fn hashmap_lookup_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let order = reorder(&keys);
    let m: HashMap<i64, i64> = HashMap::from_iter(keys.into_iter().map(|i| (i, 1)));
    b.iter(|| {
        for i in &order {
            let _ = m.get(i);
        }
    })
}

#[bench]
fn hashmap_lookup_10(b: &mut Bencher) {
    hashmap_lookup_n(10, b)
}

#[bench]
fn hashmap_lookup_100(b: &mut Bencher) {
    hashmap_lookup_n(100, b)
}

#[bench]
fn hashmap_lookup_1000(b: &mut Bencher) {
    hashmap_lookup_n(1000, b)
}

fn hashmap_insert_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = HashMap::new();
        for i in keys.clone() {
            m = m.update(i, i)
        }
    })
}

#[bench]
fn hashmap_insert_10(b: &mut Bencher) {
    hashmap_insert_n(10, b)
}

#[bench]
fn hashmap_insert_100(b: &mut Bencher) {
    hashmap_insert_n(100, b)
}

#[bench]
fn hashmap_insert_1000(b: &mut Bencher) {
    hashmap_insert_n(1000, b)
}

fn hashmap_insert_mut_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = HashMap::new();
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[bench]
fn hashmap_insert_mut_10(b: &mut Bencher) {
    hashmap_insert_mut_n(10, b)
}

#[bench]
fn hashmap_insert_mut_100(b: &mut Bencher) {
    hashmap_insert_mut_n(100, b)
}

#[bench]
fn hashmap_insert_mut_1000(b: &mut Bencher) {
    hashmap_insert_mut_n(1000, b)
}

#[bench]
fn hashmap_insert_mut_10000(b: &mut Bencher) {
    hashmap_insert_mut_n(10000, b)
}

fn hashmap_remove_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let order = reorder(&keys);
    let map: HashMap<i64, i64> = HashMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| {
        let mut m = map.clone();
        for i in &order {
            m = m.without(i)
        }
    })
}

#[bench]
fn hashmap_remove_10(b: &mut Bencher) {
    hashmap_remove_n(10, b)
}

#[bench]
fn hashmap_remove_100(b: &mut Bencher) {
    hashmap_remove_n(100, b)
}

#[bench]
fn hashmap_remove_1000(b: &mut Bencher) {
    hashmap_remove_n(1000, b)
}

fn hashmap_remove_mut_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    let order = reorder(&keys);
    let map: HashMap<i64, i64> = HashMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| {
        let mut m = map.clone();
        for i in &order {
            m.remove(i);
        }
    })
}

#[bench]
fn hashmap_remove_mut_10(b: &mut Bencher) {
    hashmap_remove_mut_n(10, b)
}

#[bench]
fn hashmap_remove_mut_100(b: &mut Bencher) {
    hashmap_remove_mut_n(100, b)
}

#[bench]
fn hashmap_remove_mut_1000(b: &mut Bencher) {
    hashmap_remove_mut_n(1000, b)
}

fn hashmap_insert_once_n(size: usize, b: &mut Bencher) {
    let mut keys = random_keys(size + 1);
    let key = keys.pop().unwrap();
    let map: HashMap<i64, i64> = HashMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| map.update(key, key))
}

#[bench]
fn hashmap_insert_once_10(b: &mut Bencher) {
    hashmap_insert_once_n(10, b)
}

#[bench]
fn hashmap_insert_once_100(b: &mut Bencher) {
    hashmap_insert_once_n(100, b)
}

#[bench]
fn hashmap_insert_once_1000(b: &mut Bencher) {
    hashmap_insert_once_n(1000, b)
}

#[bench]
fn hashmap_insert_once_10000(b: &mut Bencher) {
    hashmap_insert_once_n(10000, b)
}

fn hashmap_remove_once_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size + 1);
    let key = keys[0];
    let map: HashMap<i64, i64> = HashMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| map.without(&key))
}

#[bench]
fn hashmap_remove_once_10(b: &mut Bencher) {
    hashmap_remove_once_n(10, b)
}

#[bench]
fn hashmap_remove_once_100(b: &mut Bencher) {
    hashmap_remove_once_n(100, b)
}

#[bench]
fn hashmap_remove_once_1000(b: &mut Bencher) {
    hashmap_remove_once_n(1000, b)
}

#[bench]
fn hashmap_remove_once_10000(b: &mut Bencher) {
    hashmap_remove_once_n(10000, b)
}

fn hashmap_lookup_once_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size + 1);
    let key = keys[0];
    let map: HashMap<i64, i64> = HashMap::from_iter(keys.into_iter().map(|i| (i, i)));
    b.iter(|| map.get(&key))
}

#[bench]
fn hashmap_lookup_once_10(b: &mut Bencher) {
    hashmap_lookup_once_n(10, b)
}

#[bench]
fn hashmap_lookup_once_100(b: &mut Bencher) {
    hashmap_lookup_once_n(100, b)
}

#[bench]
fn hashmap_lookup_once_1000(b: &mut Bencher) {
    hashmap_lookup_once_n(1000, b)
}

#[bench]
fn hashmap_lookup_once_10000(b: &mut Bencher) {
    hashmap_lookup_once_n(10000, b)
}
