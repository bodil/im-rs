#![feature(test)]

extern crate test;
extern crate im;

use std::iter::FromIterator;
use test::Bencher;

use im::list::List;
use im::conslist::ConsList;

#[bench]
fn list_cons(b: &mut Bencher) {
    b.iter(|| {
        let mut l = List::new();
        for i in 0..1000 {
            l = l.cons(i)
        }
    })
}

#[bench]
fn conslist_cons(b: &mut Bencher) {
    b.iter(|| {
        let mut l = ConsList::new();
        for i in 0..1000 {
            l = l.cons(i)
        }
    })
}

#[bench]
fn vec_snoc(b: &mut Bencher) {
    b.iter(|| {
        let mut l = Vec::new();
        for i in 0..1000 {
            l = l.clone();
            l.push(i);
        }
    })
}

#[bench]
fn list_snoc(b: &mut Bencher) {
    b.iter(|| {
        let mut l = List::new();
        for i in 0..1000 {
            l = l.snoc(i)
        }
    })
}

#[bench]
fn list_uncons(b: &mut Bencher) {
    let l = List::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.tail().unwrap()
        }
    })
}

#[bench]
fn conslist_uncons(b: &mut Bencher) {
    let l = Vec::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.clone();
            p.pop();
        }
    })
}

#[bench]
fn vec_uncons(b: &mut Bencher) {
    let l = List::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.tail().unwrap()
        }
    })
}

#[bench]
fn list_append(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| List::from(0..i)));
    b.iter(|| {
        for i in 0..1000 {
            size[i].append(&size[i]);
        }
    })
}

#[bench]
fn conslist_append(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| ConsList::from(0..i)));
    b.iter(|| {
        for i in 0..1000 {
            size[i].append(&size[i]);
        }
    })
}
