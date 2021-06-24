#![allow(clippy::unit_arg)]

#[cfg(feature = "no_std")]
use hashbrown::HashSet as NatSet;
#[cfg(not(feature = "no_std"))]
use std::collections::HashSet as NatSet;
use core::fmt::{Debug, Error, Formatter, Write};
use core::hash::Hash;
use alloc::vec::Vec;
use alloc::string::String;

use crate::HashSet;

use proptest::proptest;
#[cfg(not(feature = "no_std"))]
use proptest_derive::Arbitrary;

#[cfg(not(feature = "no_std"))]
#[derive(Arbitrary, Debug)]
enum Action<A> {
    Insert(A),
    Remove(A),
}
#[cfg(feature = "no_std")]
#[derive(Debug)]
enum Action<A> {
    Insert(A),
    Remove(A),
}

#[cfg(not(feature = "no_std"))]
#[derive(Arbitrary)]
struct Actions<A>(Vec<Action<A>>)
where
    A: Hash + Eq + Clone;
#[cfg(feature = "no_std")]
struct Actions<A>(Vec<Action<A>>)
    where
        A: Hash + Eq + Clone;

impl<A> Debug for Actions<A>
where
    A: Hash + Eq + Debug + Clone,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let mut out = String::new();
        let mut expected = NatSet::new();
        writeln!(out, "let mut set = HashSet::new();")?;
        for action in &self.0 {
            match action {
                Action::Insert(ref value) => {
                    expected.insert(value.clone());
                    writeln!(out, "set.insert({:?});", value)?;
                }
                Action::Remove(ref value) => {
                    expected.remove(value);
                    writeln!(out, "set.remove({:?});", value)?;
                }
            }
        }
        writeln!(
            out,
            "let expected = vec!{:?};",
            expected.into_iter().collect::<Vec<_>>()
        )?;
        writeln!(out, "assert_eq!(HashSet::from(expected), set);")?;
        write!(f, "{}", super::code_fmt(&out))
    }
}

proptest! {
    #[test]
    fn comprehensive(actions: Actions<u8>) {
        let mut set = HashSet::new();
        let mut nat = NatSet::new();
        for action in actions.0 {
            match action {
                Action::Insert(value) => {
                    let len = nat.len() + if nat.contains(&value) {
                        0
                    } else {
                        1
                    };
                    nat.insert(value);
                    set.insert(value);
                    assert_eq!(len, set.len());
                }
                Action::Remove(value) => {
                    let len = nat.len() - if nat.contains(&value) {
                        1
                    } else {
                        0
                    };
                    nat.remove(&value);
                    set.remove(&value);
                    assert_eq!(len, set.len());
                }
            }
            assert_eq!(nat.len(), set.len());
            assert_eq!(HashSet::from(nat.clone()), set);
        }
    }
}
