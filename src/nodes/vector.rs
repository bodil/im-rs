// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::fmt::{Debug, Error, Formatter};
use std::sync::Arc;

use bits::{HASH_BITS, HASH_MASK, HASH_SIZE};

pub enum Entry<A> {
    Node(Arc<Node<A>>),
    Value(Arc<A>),
    Empty,
}

impl<A> Entry<A> {
    pub fn unwrap_val(&self) -> Arc<A> {
        match *self {
            Entry::Value(ref v) => v.clone(),
            _ => panic!("Entry::unwrap_val: tried to unwrap_val a non-value"),
        }
    }

    pub fn unwrap_node(&self) -> Arc<Node<A>> {
        match *self {
            Entry::Node(ref n) => n.clone(),
            _ => panic!("Entry::unwrap_node: tried to unwrap_node a non-node"),
        }
    }
}

impl<A> Clone for Entry<A> {
    fn clone(&self) -> Self {
        match *self {
            Entry::Node(ref node) => Entry::Node(node.clone()),
            Entry::Value(ref value) => Entry::Value(value.clone()),
            Entry::Empty => Entry::Empty,
        }
    }
}

impl<A: Debug> Debug for Entry<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match *self {
            Entry::Node(ref node) => write!(f, "Node{:?}", node),
            Entry::Value(ref value) => write!(f, "Value[ {:?} ]", value),
            Entry::Empty => write!(f, "Empty"),
        }
    }
}

pub struct Node<A> {
    first: Option<usize>,
    pub children: Vec<Entry<A>>,
}

impl<A> Node<A> {
    pub fn new() -> Self {
        Node {
            first: None,
            children: Vec::with_capacity(HASH_SIZE),
        }
    }

    pub fn from_vec(first: Option<usize>, mut children: Vec<Entry<A>>) -> Self {
        let len = children.len();
        if len < HASH_SIZE {
            children.reserve_exact(HASH_SIZE - len);
        }
        Node { children, first }
    }

    pub fn clear(&mut self) {
        self.children.clear();
        self.first = None;
    }

    pub fn len(&self) -> usize {
        self.children.len()
    }

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    pub fn push(&mut self, value: Entry<A>) {
        self.children.push(value)
    }

    pub fn get(&self, index: usize) -> Option<&Entry<A>> {
        self.children.get(index)
    }

    pub fn set(&mut self, index: usize, value: Entry<A>) {
        while self.len() < index {
            self.push(Entry::Empty);
        }
        if self.first.is_none() || Some(index) < self.first {
            self.first = Some(index);
        }
        if self.len() == index {
            self.push(value)
        } else {
            self.children[index] = value
        }
    }

    pub fn remove_before(&mut self, level: usize, index: usize) {
        let shifted = match level {
            0 => 0,
            l => 1 << l,
        };
        if index == shifted || self.is_empty() {
            return;
        }
        let origin_index = (index >> level) & HASH_MASK as usize;
        if origin_index >= self.len() {
            self.clear();
            return;
        }
        let removing_first = origin_index == 0;
        if level > 0 {
            if let Entry::Node(ref mut child) = self.children[origin_index] {
                let node = Arc::make_mut(child);
                node.remove_before(level - HASH_BITS, index);
            }
        }
        if !removing_first {
            for i in self.first.unwrap_or(0)..origin_index {
                self.children[i] = Entry::Empty;
            }
            self.first = Some(origin_index);
        }
    }

    pub fn remove_after(&mut self, level: usize, index: usize) {
        let shifted = match level {
            0 => 0,
            l => 1 << l,
        };
        if index == shifted || self.is_empty() {
            return;
        }
        let size_index = ((index - 1) >> level) & HASH_MASK as usize;
        if size_index >= self.len() {
            return;
        }
        self.children.truncate(size_index + 1);
        if let Entry::Node(ref mut n) = self.children[size_index] {
            let node = Arc::make_mut(n);
            node.remove_after(level - HASH_BITS, index);
        }
    }

    pub fn set_in(&self, level: usize, index: usize, value: Entry<A>) -> Node<A> {
        let mut out: Node<A> = self.clone();
        if level == 0 {
            out.set(index & HASH_MASK as usize, value);
        } else {
            let sub_index = (index >> level) & HASH_MASK as usize;
            if let Entry::Node(ref sub_node) = self.children[sub_index] {
                out.set(
                    sub_index,
                    Entry::Node(Arc::new(sub_node.set_in(level - HASH_BITS, index, value))),
                );
            } else {
                panic!("Vector::set_in: found non-node where node was expected");
            }
        }
        out
    }

    pub fn set_in_mut(&mut self, level: usize, end: usize, index: usize, value: Entry<A>) {
        if level == end {
            self.set((index >> end) & HASH_MASK as usize, value);
        } else {
            let sub_index = (index >> level) & HASH_MASK as usize;
            if let Some(&mut Entry::Node(ref mut sub_node)) = self.children.get_mut(sub_index) {
                let mut node = Arc::make_mut(sub_node);
                node.set_in_mut(level - HASH_BITS, end, index, value);
                return;
            }
            self.set(sub_index, Entry::Node(Arc::new(Node::new())));
            self.set_in_mut(level, end, index, value);
        }
    }

    pub fn ref_mut(&mut self, level: usize, end: usize, index: usize) -> &mut Entry<A> {
        let sub_index = (index >> level) & HASH_MASK as usize;
        let child_pos = &mut self.children[sub_index];
        if level == end {
            child_pos
        } else if let Entry::Node(ref mut sub_node) = *child_pos {
            let mut node = Arc::make_mut(sub_node);
            return node.ref_mut(level - HASH_BITS, end, index);
        } else {
            panic!("Vector::Node::ref_mut: inconsistent tree structure")
        }
    }

    pub fn write<I: Iterator<Item = Arc<A>>>(
        &mut self,
        level: usize,
        index: usize,
        it: &mut I,
        max: &mut usize,
    ) -> bool {
        let mut i = (index >> level) & HASH_MASK as usize;
        if level > 0 {
            let mut next = index;
            loop {
                let mut put = None;
                let carry_on;
                if let Some(&mut Entry::Node(ref mut sub_node)) = self.children.get_mut(i) {
                    let mut node = Arc::make_mut(sub_node);
                    carry_on = node.write(level - HASH_BITS, next, it, max);
                } else {
                    let mut node = Node::new();
                    carry_on = node.write(level - HASH_BITS, next, it, max);
                    put = Some(Entry::Node(Arc::new(node)));
                }
                if let Some(entry) = put {
                    self.set(i, entry);
                }
                if !carry_on {
                    return false;
                }
                i += 1;
                if i >= HASH_SIZE {
                    return true;
                }
                next = 0;
            }
        } else {
            loop {
                match it.next() {
                    None => {
                        return false;
                    }
                    Some(value) => {
                        self.set(i, Entry::Value(value));
                        *max -= 1;
                        if *max == 0 {
                            return false;
                        }
                        i += 1;
                        if i >= HASH_SIZE {
                            return true;
                        }
                    }
                }
            }
        }
    }
}

impl<A> Clone for Node<A> {
    fn clone(&self) -> Self {
        Node {
            first: self.first,
            children: self.children.clone(),
        }
    }
}

impl<A> Default for Node<A> {
    fn default() -> Self {
        Node::new()
    }
}

impl<A: Debug> Debug for Node<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        self.children.fmt(f)
    }
}
