#![feature(test)]

extern crate test;

use im::Vector;
use rand::seq::SliceRandom;
use test::Bencher;

// sorted
#[bench]
<<<<<<< HEAD
fn quicksort_sorted_list_0500(bench: &mut Bencher) {
=======
fn sorted_list_0500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..500).collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_sorted_list_1000(bench: &mut Bencher) {
=======
fn sorted_list_1000(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..1000).collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_sorted_list_1500(bench: &mut Bencher) {
=======
fn sorted_list_1500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..1500).collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_sorted_list_2000(bench: &mut Bencher) {
=======
fn sorted_list_2000(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..2000).collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_sorted_list_2500(bench: &mut Bencher) {
=======
fn sorted_list_2500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..2500).rev().collect();
        v1.sort()
    });
}

// reverse sorted
#[bench]
<<<<<<< HEAD
fn quicksort_reverse_sorted_list_0500(bench: &mut Bencher) {
=======
fn reverse_sorted_list_0500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..500).rev().collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_reverse_sorted_list_1000(bench: &mut Bencher) {
=======
fn reverse_sorted_list_1000(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..1000).rev().collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_reverse_sorted_list_1500(bench: &mut Bencher) {
=======
fn reverse_sorted_list_1500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..1500).rev().collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_reverse_sorted_list_2000(bench: &mut Bencher) {
=======
fn reverse_sorted_list_2000(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..2000).rev().collect();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_reverse_sorted_list_2500(bench: &mut Bencher) {
=======
fn reverse_sorted_list_2500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    bench.iter(|| {
        let mut v1: Vector<_> = (0..2500).rev().collect();
        v1.sort()
    });
}

// shuffled
#[bench]
<<<<<<< HEAD
fn quicksort_shuffled_list_0500(bench: &mut Bencher) {
=======
fn shuffled_list_0500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let mut v1: Vec<_> = (0..500).collect();
        v1.shuffle(&mut rng);
        let mut v1: Vector<_> = v1.into();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_shuffled_list_1000(bench: &mut Bencher) {
=======
fn shuffled_list_1000(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let mut v1: Vec<_> = (0..1000).collect();
        v1.shuffle(&mut rng);
        let mut v1: Vector<_> = v1.into();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_shuffled_list_1500(bench: &mut Bencher) {
=======
fn shuffled_list_1500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let mut v1: Vec<_> = (0..1500).collect();
        v1.shuffle(&mut rng);
        let mut v1: Vector<_> = v1.into();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_shuffled_list_2000(bench: &mut Bencher) {
=======
fn shuffled_list_2000(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let mut v1: Vec<_> = (0..2000).collect();
        v1.shuffle(&mut rng);
        let mut v1: Vector<_> = v1.into();
        v1.sort()
    });
}

#[bench]
<<<<<<< HEAD
fn quicksort_shuffled_list_2500(bench: &mut Bencher) {
=======
fn shuffled_list_2500(bench: &mut Bencher) {
>>>>>>> Benchmarks for quicksort.
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let mut v1: Vec<_> = (0..2500).collect();
        v1.shuffle(&mut rng);
        let mut v1: Vector<_> = v1.into();
        v1.sort()
    });
}
