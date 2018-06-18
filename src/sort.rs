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
pub fn quicksort<V, F>(vector: &mut V, l: usize, r: usize, cmp: &F)
where
    V: IndexMut<usize>,
    V::Output: Sized,
    F: Fn(&V::Output, &V::Output) -> Ordering,
{
    if r <= l {
        return;
    }

    let mut i = l;
    let mut j = r;
    let mut p = i;
    let mut q = j;
    loop {
        while cmp(&vector[i], &vector[r]) == Ordering::Less {
            i += 1
        }
        j -= 1;
        while cmp(&vector[r], &vector[j]) == Ordering::Less {
            if j == l {
                break;
            }
            j -= 1;
        }
        if i >= j {
            break;
        }
        swap(vector, i, j);
        if cmp(&vector[i], &vector[r]) == Ordering::Equal {
            p += 1;
            swap(vector, p, i);
        }
        if cmp(&vector[r], &vector[j]) == Ordering::Equal {
            q -= 1;
            swap(vector, j, q);
        }
        i += 1;
    }
    swap(vector, i, r);

    let mut jp: isize = i as isize - 1;
    let mut k = l;
    i += 1;
    while k < p {
        swap(vector, k, jp as usize);
        jp -= 1;
        k += 1;
    }
    k = r - 1;
    while k > q {
        swap(vector, i, k);
        k -= 1;
        i += 1;
    }

    if jp >= 0 {
        quicksort(vector, l, jp as usize, cmp);
    }
    quicksort(vector, i, r, cmp);
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
