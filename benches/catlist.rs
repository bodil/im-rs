#![feature(test)]

extern crate im;
extern crate test;

use std::iter::FromIterator;
use test::Bencher;

use im::catlist::CatList;

#[bench]
fn catlist_push_front(b: &mut Bencher) {
    b.iter(|| {
        let mut l = CatList::new();
        for i in 0..1000 {
            l = l.push_front(i)
        }
    })
}

#[bench]
fn catlist_push_back(b: &mut Bencher) {
    b.iter(|| {
        let mut l = CatList::new();
        for i in 0..1000 {
            l = l.push_back(i)
        }
    })
}

#[bench]
fn catlist_pop_front(b: &mut Bencher) {
    let l = CatList::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.tail().unwrap()
        }
    })
}

#[bench]
fn catlist_pop_back(b: &mut Bencher) {
    let l = CatList::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p = p.init().unwrap()
        }
    })
}

#[bench]
fn catlist_append(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| CatList::from(0..i)));
    b.iter(|| {
        for item in &size {
            item.append(item.clone());
        }
    })
}

#[bench]
fn catlist_push_front_mut(b: &mut Bencher) {
    b.iter(|| {
        let mut l = CatList::new();
        for i in 0..1000 {
            l.push_front_mut(i);
        }
    })
}

#[bench]
fn catlist_push_back_mut(b: &mut Bencher) {
    b.iter(|| {
        let mut l = CatList::new();
        for i in 0..1000 {
            l.push_back_mut(i);
        }
    })
}

#[bench]
fn catlist_pop_front_mut(b: &mut Bencher) {
    let l = CatList::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop_front_mut();
        }
    })
}

#[bench]
fn catlist_pop_back_mut(b: &mut Bencher) {
    let l = CatList::from(0..1001);
    b.iter(|| {
        let mut p = l.clone();
        for _ in 0..1000 {
            p.pop_back_mut();
        }
    })
}

#[bench]
fn catlist_append_mut(b: &mut Bencher) {
    let size = Vec::from_iter((0..1000).into_iter().map(|i| CatList::from(0..i)));
    b.iter(|| {
        for item in &size {
            let mut l = item.clone();
            l.append_mut(item.clone());
        }
    })
}
