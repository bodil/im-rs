use std::ops::Deref;
use std::marker::PhantomData;
use std::fmt;
use std::hash::Hash;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use hash::SharedHasher;
use list::List;
use conslist::ConsList;
use ordset::OrdSet;
use queue::Queue;
use ordmap::OrdMap;
use hashmap::HashMap;
use hashset::HashSet;

struct SeqVisitor<'de, S, A>
where
    S: From<Vec<A>>,
    A: Deserialize<'de>,
{
    phantom_s: PhantomData<S>,
    phantom_a: PhantomData<A>,
    phantom_lifetime: PhantomData<&'de ()>,
}

impl<'de, S, A> SeqVisitor<'de, S, A>
where
    S: From<Vec<A>>,
    A: Deserialize<'de>,
{
    pub fn new() -> SeqVisitor<'de, S, A> {
        SeqVisitor {
            phantom_s: PhantomData,
            phantom_a: PhantomData,
            phantom_lifetime: PhantomData,
        }
    }
}

impl<'de, S, A> Visitor<'de> for SeqVisitor<'de, S, A>
where
    S: From<Vec<A>>,
    A: Deserialize<'de>,
{
    type Value = S;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<Access>(self, mut access: Access) -> Result<Self::Value, Access::Error>
    where
        Access: SeqAccess<'de>,
    {
        let mut v: Vec<A> = match access.size_hint() {
            None => Vec::new(),
            Some(l) => Vec::with_capacity(l),
        };
        while let Some(i) = access.next_element()? {
            v.push(i)
        }
        Ok(From::from(v))
    }
}

struct MapVisitor<'de, S, K, V>
where
    S: From<Vec<(K, V)>>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    phantom_s: PhantomData<S>,
    phantom_k: PhantomData<K>,
    phantom_v: PhantomData<V>,
    phantom_lifetime: PhantomData<&'de ()>,
}

impl<'de, S, K, V> MapVisitor<'de, S, K, V>
where
    S: From<Vec<(K, V)>>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    pub fn new() -> MapVisitor<'de, S, K, V> {
        MapVisitor {
            phantom_s: PhantomData,
            phantom_k: PhantomData,
            phantom_v: PhantomData,
            phantom_lifetime: PhantomData,
        }
    }
}

impl<'de, S, K, V> Visitor<'de> for MapVisitor<'de, S, K, V>
where
    S: From<Vec<(K, V)>>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = S;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_map<Access>(self, mut access: Access) -> Result<Self::Value, Access::Error>
    where
        Access: MapAccess<'de>,
    {
        let mut v: Vec<(K, V)> = match access.size_hint() {
            None => Vec::new(),
            Some(l) => Vec::with_capacity(l),
        };
        while let Some(i) = access.next_entry()? {
            v.push(i)
        }
        Ok(From::from(v))
    }
}

// List

impl<'de, A: Deserialize<'de>> Deserialize<'de> for List<A> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::<'de, List<A>, A>::new())
    }
}

impl<A: Serialize> Serialize for List<A> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i.deref())?;
        }
        s.end()
    }
}

// ConsList

impl<'de, A: Deserialize<'de>> Deserialize<'de> for ConsList<A> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::<'de, ConsList<A>, A>::new())
    }
}

impl<A: Serialize> Serialize for ConsList<A> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i.deref())?;
        }
        s.end()
    }
}

// Set

impl<'de, A: Deserialize<'de> + Ord> Deserialize<'de> for OrdSet<A> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::new())
    }
}

impl<A: Ord + Serialize> Serialize for OrdSet<A> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i.deref())?;
        }
        s.end()
    }
}

// Queue

impl<'de, A: Deserialize<'de> + Ord> Deserialize<'de> for Queue<A> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::<'de, Queue<A>, A>::new())
    }
}

impl<A: Serialize> Serialize for Queue<A> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i.deref())?;
        }
        s.end()
    }
}

// Map

impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Deserialize<'de> for OrdMap<K, V> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_map(MapVisitor::<'de, OrdMap<K, V>, K, V>::new())
    }
}

impl<K: Serialize + Ord, V: Serialize> Serialize for OrdMap<K, V> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_map(Some(self.len()))?;
        for (k, v) in self.iter() {
            s.serialize_entry(k.deref(), v.deref())?;
        }
        s.end()
    }
}

// HashMap

impl<'de, K: Deserialize<'de> + Hash + Eq, V: Deserialize<'de>, S: SharedHasher> Deserialize<'de>
    for HashMap<K, V, S>
{
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_map(MapVisitor::<'de, HashMap<K, V, S>, K, V>::new())
    }
}

impl<K: Serialize + Hash + Eq, V: Serialize, S: SharedHasher> Serialize for HashMap<K, V, S> {
    fn serialize<Ser>(&self, ser: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut s = ser.serialize_map(Some(self.len()))?;
        for (k, v) in self.iter() {
            s.serialize_entry(k.deref(), v.deref())?;
        }
        s.end()
    }
}

// HashSet

impl<'de, A: Deserialize<'de> + Hash + Eq, S: SharedHasher> Deserialize<'de> for HashSet<A, S> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::new())
    }
}

impl<A: Serialize + Hash + Eq, S: SharedHasher> Serialize for HashSet<A, S> {
    fn serialize<Ser>(&self, ser: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i.deref())?;
        }
        s.end()
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::{from_str, to_string};
    use list::proptest::list;
    use proptest::num::i32;
    use conslist::proptest::conslist;
    use hashmap::proptest::hash_map;
    use hashset::proptest::hash_set;
    use ordmap::proptest::ord_map;
    use ordset::proptest::ord_set;
    use queue::proptest::queue;

    proptest! {
        #[test]
        fn ser_list(ref v in list(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<List<i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[test]
        fn ser_conslist(ref v in conslist(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<ConsList<i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[test]
        fn ser_ordset(ref v in ord_set(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<OrdSet<i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[test]
        fn ser_queue(ref v in queue(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<Queue<i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[test]
        fn ser_ordmap(ref v in ord_map(i32::ANY, i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<OrdMap<i32, i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[test]
        fn ser_hashmap(ref v in hash_map(i32::ANY, i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<HashMap<i32, i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[test]
        fn ser_hashset(ref v in hash_set(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<HashSet<i32>>(&to_string(&v).unwrap()).unwrap());
        }

    }
}
