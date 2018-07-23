// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::Ordering;
use std::ops::IndexMut;

use util::swap_indices as swap;

// Ported from the Java version at:
//    http://www.cs.princeton.edu/~rs/talks/QuicksortIsOptimal.pdf
// Should be O(n) to O(n log n)
#[cfg_attr(feature = "cargo-clippy", allow(many_single_char_names))]
pub fn quicksort<V, F>(vector: &mut V, left: usize, right: usize, cmp: &F)
where
    V: IndexMut<usize>,
    V::Output: Sized,
    F: Fn(&V::Output, &V::Output) -> Ordering,
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
        while cmp(&vector[i as usize], &vector[r as usize]) == Ordering::Less {
            i += 1;
        }
        j -= 1;
        while cmp(&vector[r as usize], &vector[j as usize]) == Ordering::Less {
            if j == l {
                break;
            }
            j -= 1;
        }
        if i >= j {
            break;
        }
        swap(vector, i as usize, j as usize);
        if cmp(&vector[i as usize], &vector[r as usize]) == Ordering::Equal {
            p += 1;
            swap(vector, p as usize, i as usize);
        }
        if cmp(&vector[r as usize], &vector[j as usize]) == Ordering::Equal {
            q -= 1;
            swap(vector, j as usize, q as usize);
        }
    }
    swap(vector, i as usize, r as usize);

    j = i - 1;
    i += 1;
    let mut k = l;
    while k < p {
        swap(vector, k as usize, j as usize);
        j -= 1;
        k += 1;
    }
    k = r - 1;
    while k > q {
        swap(vector, i as usize, k as usize);
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
    use proptest::collection::vec;
    use proptest::num::i32;
    use test::is_sorted;

    proptest! {
        #[test]
        fn test_quicksort(ref input in vec(i32::ANY, 0..1000)) {
            let mut vec = input.clone();
            let len = vec.len();
            if len > 1 {
                quicksort(&mut vec, 0, len - 1, &Ord::cmp);
            }
            assert!(is_sorted(vec));
        }
    }
}
