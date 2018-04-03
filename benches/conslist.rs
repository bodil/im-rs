#![feature(test)]

extern crate test;
extern crate im;

use std::iter::FromIterator;
use test::Bencher;

use im::conslist::ConsList;

#[bench]
fn conslist_cons(b: &mut Bencher) {
    b.iter(|| {
        let mut l = ConsList::new();
        for i in 0..1000 {
            l = l.cons(i);
        }
    })
}

#[bench]
fn conslist_uncons(b: &mut Bencher) {
    let l = ConsList::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.tail().unwrap();
        }
    })
}

#[bench]
fn conslist_append(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| ConsList::from_iter(0..i)));
    b.iter(|| {
        for item in &size {
            item.append(item);
        }
    })
}
