// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::hash::{BuildHasher, Hash};
use std::iter;

use ::arbitrary::{size_hint, Arbitrary, Result, Unstructured};

use crate::{HashMap, HashSet, OrdMap, OrdSet, Vector};

fn empty<T: 'static>() -> Box<dyn Iterator<Item = T>> {
    Box::new(iter::empty())
}

fn shrink_collection<T: Clone, A: Clone + Arbitrary>(
    entries: impl Iterator<Item = T>,
    f: impl Fn(&T) -> Box<dyn Iterator<Item = A>>,
) -> Box<dyn Iterator<Item = Vec<A>>> {
    let entries: Vec<_> = entries.collect();
    if entries.is_empty() {
        return empty();
    }

    let mut shrinkers: Vec<Vec<_>> = vec![];
    let mut i = entries.len();
    loop {
        shrinkers.push(entries.iter().take(i).map(&f).collect());
        i /= 2;
        if i == 0 {
            break;
        }
    }
    Box::new(iter::once(Vec::new()).chain(iter::from_fn(move || loop {
        let mut shrinker = shrinkers.pop()?;
        let x: Option<Vec<A>> = shrinker.iter_mut().map(|s| s.next()).collect();
        if x.is_none() {
            continue;
        }
        shrinkers.push(shrinker);
        return x;
    })))
}

impl<A: Arbitrary + Clone> Arbitrary for Vector<A> {
    fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'_>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let collections = shrink_collection(self.iter(), |x| x.shrink());
        Box::new(collections.map(|entries| entries.into_iter().collect()))
    }
}

impl<K: Arbitrary + Ord + Clone, V: Arbitrary + Clone> Arbitrary for OrdMap<K, V> {
    fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'_>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let collections =
            shrink_collection(self.iter(), |(k, v)| Box::new(k.shrink().zip(v.shrink())));
        Box::new(collections.map(|entries| entries.into_iter().collect()))
    }
}

impl<A: Arbitrary + Ord + Clone> Arbitrary for OrdSet<A> {
    fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'_>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let collections = shrink_collection(self.iter(), |v| v.shrink());
        Box::new(collections.map(|entries| entries.into_iter().collect()))
    }
}

impl<K, V, S> Arbitrary for HashMap<K, V, S>
where
    K: Arbitrary + Hash + Eq + Clone,
    V: Arbitrary + Clone,
    S: BuildHasher + Default + 'static,
{
    fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'_>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let collections =
            shrink_collection(self.iter(), |(k, v)| Box::new(k.shrink().zip(v.shrink())));
        Box::new(collections.map(|entries| entries.into_iter().collect()))
    }
}

impl<A, S> Arbitrary for HashSet<A, S>
where
    A: Arbitrary + Hash + Eq + Clone,
    S: BuildHasher + Default + 'static,
{
    fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'_>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let collections = shrink_collection(self.iter(), |v| v.shrink());
        Box::new(collections.map(|entries| entries.into_iter().collect()))
    }
}
