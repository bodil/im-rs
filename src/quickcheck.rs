use crate::{hashmap::HashMap, hashset::HashSet, OrdMap, OrdSet, Vector};
use ::quickcheck::{Arbitrary, Gen};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;

impl<A: Arbitrary + Sync + Clone> Arbitrary for Vector<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Vector::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<K: Ord + Clone + Arbitrary + Sync, V: Clone + Arbitrary + Sync> Arbitrary for OrdMap<K, V> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        OrdMap::from_iter(Vec::<(K, V)>::arbitrary(g))
    }
}

impl<A: Ord + Clone + Arbitrary + Sync> Arbitrary for OrdSet<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        OrdSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<A, S> Arbitrary for HashSet<A, S>
where
    A: Hash + Eq + Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        HashSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<K, V, S> Arbitrary for HashMap<K, V, S>
where
    K: Hash + Eq + Arbitrary + Sync,
    V: Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        HashMap::from(Vec::<(K, V)>::arbitrary(g))
    }
}
