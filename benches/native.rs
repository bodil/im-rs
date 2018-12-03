// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]

extern crate im;
extern crate rand;
extern crate test;

use rand::{rngs::SmallRng, FromEntropy, Rng};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::iter::FromIterator;
use test::Bencher;

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

fn std_hashmap_insert_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = HashMap::new();
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[bench]
fn std_hashmap_insert_mut_10(b: &mut Bencher) {
    std_hashmap_insert_n(10, b)
}

#[bench]
fn std_hashmap_insert_mut_100(b: &mut Bencher) {
    std_hashmap_insert_n(100, b)
}

#[bench]
fn std_hashmap_insert_mut_1000(b: &mut Bencher) {
    std_hashmap_insert_n(1000, b)
}

#[bench]
fn std_hashmap_insert_mut_10000(b: &mut Bencher) {
    std_hashmap_insert_n(10000, b)
}

fn std_btreemap_insert_n(size: usize, b: &mut Bencher) {
    let keys = random_keys(size);
    b.iter(|| {
        let mut m = BTreeMap::new();
        for i in keys.clone() {
            m.insert(i, i);
        }
    })
}

#[bench]
fn std_btreemap_insert_10(b: &mut Bencher) {
    std_btreemap_insert_n(10, b)
}

#[bench]
fn std_btreemap_insert_100(b: &mut Bencher) {
    std_btreemap_insert_n(100, b)
}

#[bench]
fn std_btreemap_insert_1000(b: &mut Bencher) {
    std_btreemap_insert_n(1000, b)
}

fn std_btreemap_insert_clone_n(size: usize, b: &mut Bencher) {
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
fn std_btreemap_insert_clone_10(b: &mut Bencher) {
    std_btreemap_insert_clone_n(10, b)
}

#[bench]
fn std_btreemap_insert_clone_100(b: &mut Bencher) {
    std_btreemap_insert_clone_n(100, b)
}

#[bench]
fn std_btreemap_insert_clone_1000(b: &mut Bencher) {
    std_btreemap_insert_clone_n(1000, b)
}

fn vecdeque_push_front_mut(b: &mut Bencher, count: usize) {
    b.iter(|| {
        let mut l = VecDeque::new();
        for i in 0..count {
            l.push_front(i);
        }
    })
}

#[bench]
fn vecdeque_push_front_mut_10(b: &mut Bencher) {
    vecdeque_push_front_mut(b, 10)
}

#[bench]
fn vecdeque_push_front_mut_100(b: &mut Bencher) {
    vecdeque_push_front_mut(b, 100)
}

#[bench]
fn vecdeque_push_front_mut_1000(b: &mut Bencher) {
    vecdeque_push_front_mut(b, 1000)
}

#[bench]
fn vecdeque_push_front_mut_100000(b: &mut Bencher) {
    vecdeque_push_front_mut(b, 100_000)
}

fn vecdeque_push_back_mut(b: &mut Bencher, count: usize) {
    b.iter(|| {
        let mut l = VecDeque::new();
        for i in 0..count {
            l.push_back(i);
        }
    })
}

#[bench]
fn vecdeque_push_back_mut_10(b: &mut Bencher) {
    vecdeque_push_back_mut(b, 10)
}

#[bench]
fn vecdeque_push_back_mut_100(b: &mut Bencher) {
    vecdeque_push_back_mut(b, 100)
}

#[bench]
fn vecdeque_push_back_mut_1000(b: &mut Bencher) {
    vecdeque_push_back_mut(b, 1000)
}

#[bench]
fn vecdeque_push_back_mut_100000(b: &mut Bencher) {
    vecdeque_push_back_mut(b, 100_000)
}

fn vecdeque_pop_front_mut(b: &mut Bencher, count: usize) {
    let l = VecDeque::from_iter(0..count);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..count {
            p.pop_front();
        }
    })
}

#[bench]
fn vecdeque_pop_front_mut_10(b: &mut Bencher) {
    vecdeque_pop_front_mut(b, 10)
}

#[bench]
fn vecdeque_pop_front_mut_100(b: &mut Bencher) {
    vecdeque_pop_front_mut(b, 100)
}

#[bench]
fn vecdeque_pop_front_mut_1000(b: &mut Bencher) {
    vecdeque_pop_front_mut(b, 1000)
}

#[bench]
fn vecdeque_pop_front_mut_100000(b: &mut Bencher) {
    vecdeque_pop_front_mut(b, 100_000)
}

fn vecdeque_pop_back_mut(b: &mut Bencher, count: usize) {
    let l = VecDeque::from_iter(0..count);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..count {
            p.pop_back();
        }
    })
}

#[bench]
fn vecdeque_pop_back_mut_10(b: &mut Bencher) {
    vecdeque_pop_back_mut(b, 10)
}

#[bench]
fn vecdeque_pop_back_mut_100(b: &mut Bencher) {
    vecdeque_pop_back_mut(b, 100)
}

#[bench]
fn vecdeque_pop_back_mut_1000(b: &mut Bencher) {
    vecdeque_pop_back_mut(b, 1000)
}

#[bench]
fn vecdeque_pop_back_mut_100000(b: &mut Bencher) {
    vecdeque_pop_back_mut(b, 100_000)
}

fn vecdeque_split(b: &mut Bencher, count: usize) {
    let vec = VecDeque::from_iter(0..count);
    b.iter(|| {
        let mut left = vec.clone();
        let _right = left.split_off(count / 2);
    })
}

#[bench]
fn vecdeque_split_10(b: &mut Bencher) {
    vecdeque_split(b, 10)
}

#[bench]
fn vecdeque_split_100(b: &mut Bencher) {
    vecdeque_split(b, 100)
}

#[bench]
fn vecdeque_split_1000(b: &mut Bencher) {
    vecdeque_split(b, 1000)
}

#[bench]
fn vecdeque_split_100000(b: &mut Bencher) {
    vecdeque_split(b, 100_000)
}

fn vecdeque_append(b: &mut Bencher, count: usize) {
    let vec1 = VecDeque::from_iter(0..count / 2);
    let vec2 = VecDeque::from_iter(count / 2..count);
    b.iter(|| {
        let mut vec = vec1.clone();
        vec.append(&mut vec2.clone());
    })
}

#[bench]
fn vecdeque_append_10(b: &mut Bencher) {
    vecdeque_append(b, 10)
}

#[bench]
fn vecdeque_append_100(b: &mut Bencher) {
    vecdeque_append(b, 100)
}

#[bench]
fn vecdeque_append_1000(b: &mut Bencher) {
    vecdeque_append(b, 1000)
}

#[bench]
fn vecdeque_append_100000(b: &mut Bencher) {
    vecdeque_append(b, 100_000)
}

fn vec_split(b: &mut Bencher, count: usize) {
    let vec = Vec::from_iter(0..count);
    b.iter(|| {
        let _left = vec[..count / 2].to_owned();
        let _right = vec[count / 2..].to_owned();
    })
}

#[bench]
fn vec_split_10(b: &mut Bencher) {
    vec_split(b, 10)
}

#[bench]
fn vec_split_100(b: &mut Bencher) {
    vec_split(b, 100)
}

#[bench]
fn vec_split_1000(b: &mut Bencher) {
    vec_split(b, 1000)
}

#[bench]
fn vec_split_100000(b: &mut Bencher) {
    vec_split(b, 100_000)
}

fn vec_append(b: &mut Bencher, count: usize) {
    let vec1 = Vec::from_iter(0..count / 2);
    let vec2 = Vec::from_iter(count / 2..count);
    b.iter(|| {
        let mut vec = vec1.clone();
        vec.append(&mut vec2.clone());
    })
}

#[bench]
fn vec_append_10(b: &mut Bencher) {
    vec_append(b, 10)
}

#[bench]
fn vec_append_100(b: &mut Bencher) {
    vec_append(b, 100)
}

#[bench]
fn vec_append_1000(b: &mut Bencher) {
    vec_append(b, 1000)
}

#[bench]
fn vec_append_100000(b: &mut Bencher) {
    vec_append(b, 100_000)
}
