use std::ops::Deref;
use std::marker::PhantomData;
use std::fmt;
use serde::ser::{Serialize, Serializer, SerializeSeq, SerializeMap};
use serde::de::{Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};
use list::List;
use conslist::ConsList;
use set::Set;
use queue::Queue;
use map::Map;

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
        loop {
            match access.next_element()? {
                Some(i) => v.push(i),
                None => break,
            }
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
        loop {
            match access.next_entry()? {
                Some(i) => v.push(i),
                None => break,
            }
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

impl<'de, A: Deserialize<'de> + Ord> Deserialize<'de> for Set<A> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::new())
    }
}

impl<A: Serialize> Serialize for Set<A> {
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

impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Deserialize<'de> for Map<K, V> {
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_map(MapVisitor::<'de, Map<K, V>, K, V>::new())
    }
}

impl<K: Serialize + Ord, V: Serialize> Serialize for Map<K, V> {
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

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::{to_string, from_str};

    quickcheck! {
        fn list(list: List<i32>) -> bool {
            from_str::<List<i32>>(&to_string(&list).unwrap()).unwrap() == list
        }

        fn conslist(list: ConsList<i32>) -> bool {
            from_str::<ConsList<i32>>(&to_string(&list).unwrap()).unwrap() == list
        }

        fn set(list: Set<i32>) -> bool {
            from_str::<Set<i32>>(&to_string(&list).unwrap()).unwrap() == list
        }

        fn queue(list: Queue<i32>) -> bool {
            from_str::<Queue<i32>>(&to_string(&list).unwrap()).unwrap() == list
        }

        fn map(list: Map<i32, i32>) -> bool {
            from_str::<Map<i32, i32>>(&to_string(&list).unwrap()).unwrap() == list
        }
    }
}
