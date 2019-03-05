#![allow(clippy::unit_arg)]

use std::collections::BTreeSet;
use std::fmt::{Debug, Error, Formatter};

use crate::OrdSet;

use proptest::proptest;
use proptest_derive::Arbitrary;

#[derive(Arbitrary, Debug)]
enum Action<A> {
    Insert(A),
    Remove(A),
}

#[derive(Arbitrary)]
struct Actions<A>(Vec<Action<A>>)
where
    A: Ord + Clone;

impl<A> Debug for Actions<A>
where
    A: Ord + Debug + Clone,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        let mut expected = BTreeSet::new();
        writeln!(f, "let mut set = OrdSet::new();")?;
        for action in &self.0 {
            match action {
                Action::Insert(ref value) => {
                    expected.insert(value.clone());
                    writeln!(f, "set.insert({:?});", value)?;
                }
                Action::Remove(ref value) => {
                    expected.remove(value);
                    writeln!(f, "set.remove({:?});", value)?;
                }
            }
        }
        writeln!(
            f,
            "let expected = vec!{:?};",
            expected.into_iter().collect::<Vec<_>>()
        )?;
        writeln!(f, "assert_eq!(OrdSet::from(expected), set);")
    }
}

proptest! {
    #[test]
    fn comprehensive(actions: Actions<u8>) {
        let mut set = OrdSet::new();
        let mut nat = BTreeSet::new();
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
            assert_eq!(OrdSet::from(nat.clone()), set);
            assert!(nat.iter().eq(set.iter()));
        }
    }
}
