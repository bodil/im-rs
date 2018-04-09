// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]
#![cfg_attr(feature="cargo-clippy", allow(unreadable_literal))]

extern crate im;
extern crate rand;
extern crate test;

use std::iter::FromIterator;
use std::collections::{BTreeMap, HashMap};
use test::Bencher;
use rand::{weak_rng, Rng};

fn random_keys(size: usize) -> Vec<i64> {
    let mut gen = weak_rng();
    let mut set = Vec::new();
    while set.len() < size {
        let next = gen.gen::<i64>() % 10000;
        if !set.contains(&next) {
            set.push(next);
        }
    }
    set
}

fn native_hashmap_insert_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = HashMap::new();
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[bench]
fn native_hashmap_insert_10(b: &mut Bencher) {
    native_hashmap_insert_n(10, b)
}

#[bench]
fn native_hashmap_insert_100(b: &mut Bencher) {
    native_hashmap_insert_n(100, b)
}

#[bench]
fn native_hashmap_insert_1000(b: &mut Bencher) {
    native_hashmap_insert_n(1000, b)
}

fn native_btreemap_insert_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = BTreeMap::new();
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[bench]
fn native_btreemap_insert_10(b: &mut Bencher) {
    native_btreemap_insert_n(10, b)
}

#[bench]
fn native_btreemap_insert_100(b: &mut Bencher) {
    native_btreemap_insert_n(100, b)
}

#[bench]
fn native_btreemap_insert_1000(b: &mut Bencher) {
    native_btreemap_insert_n(1000, b)
}

fn native_btreemap_insert_clone_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = BTreeMap::new();
        for i in keys.clone() {
            m = m.clone();
            m.insert(i, i);
        }
    })
}

#[bench]
fn native_btreemap_insert_clone_10(b: &mut Bencher) {
    native_btreemap_insert_clone_n(10, b)
}

#[bench]
fn native_btreemap_insert_clone_100(b: &mut Bencher) {
    native_btreemap_insert_clone_n(100, b)
}

#[bench]
fn native_btreemap_insert_clone_1000(b: &mut Bencher) {
    native_btreemap_insert_clone_n(1000, b)
}

#[bench]
fn native_vec_push_back_clone_1000(b: &mut Bencher) {
    b.iter(|| {
        let mut l = Vec::new();
        for i in 0..1000 {
            l = l.clone();
            l.push(i);
        }
    })
}

#[bench]
fn native_vec_pop_back(b: &mut Bencher) {
    let l = Vec::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop().unwrap();
        }
    })
}
