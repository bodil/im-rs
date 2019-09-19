// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::vector::FocusMut;
use rand_core::{RngCore, SeedableRng};
use std::cmp::Ordering;

fn gen_range<R: RngCore>(rng: &mut R, min: usize, max: usize) -> usize {
    let range = max - min;
    min + (rng.next_u64() as usize % range)
}

// Ported from the Java version at:
//    http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
// Should be O(n) to O(n log n)
pub fn do_quicksort<A, F, R>(
    vector: &mut FocusMut<A>,
    left: usize,
    right: usize,
    cmp: &F,
    rng: &mut R,
) where
    A: Clone,
    F: Fn(&A, &A) -> Ordering,
    R: RngCore,
{
    if right <= left {
        return;
    }

    let l = left as isize;
    let r = right as isize;
    let p = gen_range(rng, left, right + 1) as isize;
    let mut l1 = l;
    let mut r1 = r;
    let mut l2 = l - 1;
    let mut r2 = r;

    vector.swap(r as usize, p as usize);
    loop {
        while l1 != r && vector.pair(l1 as usize, r as usize, |a, b| cmp(a, b)) == Ordering::Less {
            l1 += 1;
        }
        r1 -= 1;
        while r1 != r && vector.pair(r as usize, r1 as usize, |a, b| cmp(a, b)) == Ordering::Less {
            if r1 == l {
                break;
            }
            r1 -= 1;
        }
        if l1 >= r1 {
            break;
        }
        vector.swap(l1 as usize, r1 as usize);
        if l1 != r && vector.pair(l1 as usize, r as usize, |a, b| cmp(a, b)) == Ordering::Equal {
            l2 += 1;
            vector.swap(l2 as usize, l1 as usize);
        }
        if r1 != r && vector.pair(r as usize, r1 as usize, |a, b| cmp(a, b)) == Ordering::Equal {
            r2 -= 1;
            vector.swap(r1 as usize, r2 as usize);
        }
    }
    vector.swap(l1 as usize, r as usize);

    r1 = l1 - 1;
    l1 += 1;
    let mut k = l;
    while k < l2 {
        vector.swap(k as usize, r1 as usize);
        r1 -= 1;
        k += 1;
    }
    k = r - 1;
    while k > r2 {
        vector.swap(l1 as usize, k as usize);
        k -= 1;
        l1 += 1;
    }

    if r1 >= 0 {
        do_quicksort(vector, left, r1 as usize, cmp, rng);
    }
    do_quicksort(vector, l1 as usize, right, cmp, rng);
}

pub fn quicksort<A, F>(vector: &mut FocusMut<A>, left: usize, right: usize, cmp: &F)
where
    A: Clone,
    F: Fn(&A, &A) -> Ordering,
{
    let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
    do_quicksort(vector, left, right, cmp, &mut rng);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::is_sorted;
    use crate::vector::proptest::vector;
    use ::proptest::num::i32;
    use ::proptest::proptest;

    proptest! {
        #[test]
        fn test_quicksort(ref input in vector(i32::ANY, 0..1000)) {
            let mut vec = input.clone();
            let len = vec.len();
            if len > 1 {
                quicksort(&mut vec.focus_mut(), 0, len - 1, &Ord::cmp);
            }
            assert!(is_sorted(vec));
        }
    }
}
