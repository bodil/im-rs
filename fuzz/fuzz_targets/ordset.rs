#![no_main]

use std::collections::HashSet as NatSet;
use std::fmt::Debug;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use im::OrdSet;

#[derive(Arbitrary, Debug)]
enum Action<A> {
    Insert(A),
    Remove(A),
}

#[derive(Arbitrary)]
struct Actions<A>(Vec<Action<A>>);

fuzz_target!(|actions: Vec<Action<u64>>| {
    let mut set = OrdSet::new();
    let mut nat = NatSet::new();
    for action in actions {
        match action {
            Action::Insert(value) => {
                let len = nat.len() + if nat.contains(&value) { 0 } else { 1 };
                nat.insert(value);
                set.insert(value);
                assert_eq!(len, set.len());
            }
            Action::Remove(value) => {
                let len = nat.len() - if nat.contains(&value) { 1 } else { 0 };
                nat.remove(&value);
                set.remove(&value);
                assert_eq!(len, set.len());
            }
        }
        assert_eq!(nat.len(), set.len());
        assert_eq!(OrdSet::from(nat.clone()), set);
    }
});
