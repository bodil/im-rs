#![no_main]

use std::fmt::Debug;
use std::iter::FromIterator;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use im::Vector;

#[derive(Arbitrary, Debug)]
enum Action<A> {
    PushFront(A),
    PushBack(A),
    PopFront,
    PopBack,
    Insert(usize, A),
    Remove(usize),
    JoinLeft(Vec<A>),
    JoinRight(Vec<A>),
    SplitLeft(usize),
    SplitRight(usize),
}

fn cap_index(len: usize, index: usize) -> usize {
    if len == 0 {
        0
    } else {
        index % len
    }
}

fuzz_target!(|actions: Vec<Action<u64>>| {
    let mut vec = Vector::new();
    let mut nat = Vec::new();
    vec.assert_invariants();
    for action in actions {
        match action {
            Action::PushFront(value) => {
                let len = vec.len();
                nat.insert(0, value);
                vec.push_front(value);
                assert_eq!(len + 1, vec.len());
            }
            Action::PushBack(value) => {
                let len = vec.len();
                nat.push(value);
                vec.push_back(value);
                assert_eq!(len + 1, vec.len());
            }
            Action::PopFront => {
                if vec.is_empty() {
                    assert_eq!(None, vec.pop_front());
                } else {
                    let len = vec.len();
                    assert_eq!(nat.remove(0), vec.pop_front().unwrap());
                    assert_eq!(len - 1, vec.len());
                }
            }
            Action::PopBack => {
                if vec.is_empty() {
                    assert_eq!(None, vec.pop_back());
                } else {
                    let len = vec.len();
                    assert_eq!(nat.pop(), vec.pop_back());
                    assert_eq!(len - 1, vec.len());
                }
            }
            Action::Insert(index, value) => {
                let index = cap_index(vec.len(), index);
                let len = vec.len();
                nat.insert(index, value);
                vec.insert(index, value);
                assert_eq!(len + 1, vec.len());
            }
            Action::Remove(index) => {
                if vec.is_empty() {
                    continue;
                }
                let index = cap_index(vec.len(), index);
                let len = vec.len();
                assert_eq!(nat.remove(index), vec.remove(index));
                assert_eq!(len - 1, vec.len());
            }
            Action::JoinLeft(mut new_nat) => {
                let mut new_vec = Vector::from_iter(new_nat.iter().cloned());
                let add_len = new_nat.len();
                let len = vec.len();
                new_vec.append(vec);
                vec = new_vec;
                new_nat.append(&mut nat);
                nat = new_nat;
                assert_eq!(len + add_len, vec.len());
            }
            Action::JoinRight(mut new_nat) => {
                let new_vec = Vector::from_iter(new_nat.iter().cloned());
                let add_len = new_nat.len();
                let len = vec.len();
                vec.append(new_vec);
                nat.append(&mut new_nat);
                assert_eq!(len + add_len, vec.len());
            }
            Action::SplitLeft(index) => {
                let index = cap_index(vec.len(), index);
                let len = vec.len();
                let vec_right = vec.split_off(index);
                let nat_right = nat.split_off(index);
                assert_eq!(index, vec.len());
                assert_eq!(len - index, vec_right.len());
                assert_eq!(Vector::from_iter(nat_right.iter().cloned()), vec_right);
            }
            Action::SplitRight(index) => {
                let index = cap_index(vec.len(), index);
                let len = vec.len();
                let vec_right = vec.split_off(index);
                let nat_right = nat.split_off(index);
                assert_eq!(index, vec.len());
                assert_eq!(len - index, vec_right.len());
                assert_eq!(Vector::from_iter(nat.iter().cloned()), vec);
                vec = vec_right;
                nat = nat_right;
            }
        }
        vec.assert_invariants();
        assert_eq!(nat.len(), vec.len());
        assert_eq!(Vector::from_iter(nat.iter().cloned()), vec);
    }
});
