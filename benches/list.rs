#![feature(test)]

extern crate test;
extern crate im;

use std::iter::FromIterator;
use test::Bencher;

use im::list::List;

#[bench]
fn list_push_front(b: &mut Bencher) {
    b.iter(|| {
        let mut l = List::new();
        for i in 0..1000 {
            l = l.push_front(i)
        }
    })
}

#[bench]
fn list_push_back(b: &mut Bencher) {
    b.iter(|| {
        let mut l = List::new();
        for i in 0..1000 {
            l = l.push_back(i)
        }
    })
}

#[bench]
fn list_pop_front(b: &mut Bencher) {
    let l = List::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.tail().unwrap()
        }
    })
}

#[bench]
fn list_pop_back(b: &mut Bencher) {
    let l = List::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.init().unwrap()
        }
    })
}

#[bench]
fn list_append(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| List::from(0..i)));
    b.iter(|| {
        for item in &size {
            item.append(item.clone());
        }
    })
}

#[bench]
fn list_push_front_mut(b: &mut Bencher) {
    b.iter(|| {
        let mut l = List::new();
        for i in 0..1000 {
            l.push_front_mut(i);
        }
    })
}

#[bench]
fn list_push_back_mut(b: &mut Bencher) {
    b.iter(|| {
        let mut l = List::new();
        for i in 0..1000 {
            l.push_back_mut(i);
        }
    })
}

#[bench]
fn list_pop_front_mut(b: &mut Bencher) {
    let l = List::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop_front_mut();
        }
    })
}

#[bench]
fn list_pop_back_mut(b: &mut Bencher) {
    let l = List::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop_back_mut();
        }
    })
}

#[bench]
fn list_append_mut(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| List::from(0..i)));
    b.iter(|| {
        for item in &size {
            let mut l = item.clone();
            l.append_mut(item.clone());
        }
    })
}
