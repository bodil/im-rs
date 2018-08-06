// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::Ordering;
use vector::FocusMut;

// Ported from the Java version at:
//    http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
// Should be O(n) to O(n log n)
#[cfg_attr(feature = "cargo-clippy", allow(many_single_char_names))]
pub fn quicksort<A, F>(vector: &mut FocusMut<A>, left: usize, right: usize, cmp: &F)
where
    A: Clone,
    F: Fn(&A, &A) -> Ordering,
{
    if right <= left {
        return;
    }

    let l = left as isize;
    let r = right as isize;
    let mut i = l;
    let mut j = r;
    let mut p = l - 1;
    let mut q = r;
    loop {
        while i != r && vector.pair(i as usize, r as usize, |a, b| cmp(a, b)) == Ordering::Less {
            i += 1;
        }
        j -= 1;
        while j != r && vector.pair(r as usize, j as usize, |a, b| cmp(a, b)) == Ordering::Less {
            if j == l {
                break;
            }
            j -= 1;
        }
        if i >= j {
            break;
        }
        vector.swap(i as usize, j as usize);
        if i != r && vector.pair(i as usize, r as usize, |a, b| cmp(a, b)) == Ordering::Equal {
            p += 1;
            vector.swap(p as usize, i as usize);
        }
        if j != r && vector.pair(r as usize, j as usize, |a, b| cmp(a, b)) == Ordering::Equal {
            q -= 1;
            vector.swap(j as usize, q as usize);
        }
    }
    vector.swap(i as usize, r as usize);

    j = i - 1;
    i += 1;
    let mut k = l;
    while k < p {
        vector.swap(k as usize, j as usize);
        j -= 1;
        k += 1;
    }
    k = r - 1;
    while k > q {
        vector.swap(i as usize, k as usize);
        k -= 1;
        i += 1;
    }

    if j >= 0 {
        quicksort(vector, left, j as usize, cmp);
    }
    quicksort(vector, i as usize, right, cmp);
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::num::i32;
    use test::is_sorted;
    use vector::proptest::vector;

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
