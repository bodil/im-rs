// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(test)]

extern crate im;
extern crate test;

use std::iter::FromIterator;
use test::Bencher;

use im::conslist::ConsList;

fn conslist_cons(b: &mut Bencher, count: usize) {
    b.iter(|| {
        let mut l = ConsList::new();
        for i in 0..count {
            l = l.cons(i);
        }
    })
}

#[bench]
fn conslist_cons_10(b: &mut Bencher) {
    conslist_cons(b, 10)
}

#[bench]
fn conslist_cons_100(b: &mut Bencher) {
    conslist_cons(b, 100)
}

#[bench]
fn conslist_cons_1000(b: &mut Bencher) {
    conslist_cons(b, 1000)
}

fn conslist_uncons(b: &mut Bencher, count: usize) {
    let l = ConsList::from_iter(0..(count + 1));
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..count {
            p = p.tail().unwrap();
        }
    })
}

#[bench]
fn conslist_uncons_10(b: &mut Bencher) {
    conslist_uncons(b, 10)
}

#[bench]
fn conslist_uncons_100(b: &mut Bencher) {
    conslist_uncons(b, 100)
}

#[bench]
fn conslist_uncons_1000(b: &mut Bencher) {
    conslist_uncons(b, 1000)
}
