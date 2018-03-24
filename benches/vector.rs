#![feature(test)]

extern crate test;
extern crate im;

use std::iter::FromIterator;
use test::Bencher;

use im::vector::Vector;

#[bench]
fn vector_push_front(b: &mut Bencher) {
    b.iter(|| {
        let mut l = Vector::new();
        for i in 0..1000 {
            l = l.push_front(i)
        }
    })
}

#[bench]
fn vector_push_back(b: &mut Bencher) {
    b.iter(|| {
        let mut l = Vector::new();
        for i in 0..1000 {
            l = l.push_back(i)
        }
    })
}

#[bench]
fn vector_pop_front(b: &mut Bencher) {
    let l = Vector::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.tail().unwrap()
        }
    })
}

#[bench]
fn vector_pop_back(b: &mut Bencher) {
    let l = Vector::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.init().unwrap()
        }
    })
}

#[bench]
fn vector_append(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| Vector::from_iter(0..i)));
    b.iter(|| {
        for item in &size {
            item.append(item.clone());
        }
    })
}

#[bench]
fn vector_push_front_mut(b: &mut Bencher) {
    b.iter(|| {
        let mut l = Vector::new();
        for i in 0..1000 {
            l.push_front_mut(i);
        }
    })
}

#[bench]
fn vector_push_back_mut(b: &mut Bencher) {
    b.iter(|| {
        let mut l = Vector::new();
        for i in 0..1000 {
            l.push_back_mut(i);
        }
    })
}

#[bench]
fn vector_pop_front_mut(b: &mut Bencher) {
    let l = Vector::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop_front_mut();
        }
    })
}

#[bench]
fn vector_pop_back_mut(b: &mut Bencher) {
    let l = Vector::from_iter(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop_back_mut();
        }
    })
}
