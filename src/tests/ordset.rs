#![allow(clippy::unit_arg)]

use std::collections::BTreeSet;
use std::fmt::{Debug, Error, Formatter, Write};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let mut out = String::new();
        let mut expected = BTreeSet::new();
        writeln!(out, "let mut set = OrdSet::new();")?;
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
        writeln!(out, "assert_eq!(OrdSet::from(expected), set);")?;
        write!(f, "{}", super::code_fmt(&out))
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

#[test]
fn regression_ordset_removal_panic() { // issue 124
    let mut set = OrdSet::new();

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("non-proptest-regressions/tests/ordset_removal_panic.txt");
    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    for line in contents.split('\n') {
        if line.starts_with("insert ") {
            set.insert(line[7..].parse::<u32>().unwrap());
        } else if line.starts_with("remove ") {
            set.remove(&line[7..].parse::<u32>().unwrap());
        }
    }
}
