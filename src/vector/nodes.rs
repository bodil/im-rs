use std::sync::Arc;
use std::iter;

use bits::{HASH_BITS, HASH_MASK, HASH_SIZE};

pub enum Entry<A> {
    Node(Arc<Node<A>>),
    Value(Arc<A>),
    Empty,
}

impl<A> Entry<A> {
    pub fn into_val(self) -> Arc<A> {
        match self {
            Entry::Value(v) => v,
            _ => panic!("Entry::into_val: tried to into_val a non-value"),
        }
    }

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

    pub fn is_empty(&self) -> bool {
        match *self {
            Entry::Empty => true,
            _ => false,
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

pub struct Node<A> {
    pub children: Vec<Entry<A>>,
}

fn make_empty<A>() -> Vec<Entry<A>> {
    iter::repeat(())
        .map(|_| Entry::Empty)
        .take(HASH_SIZE)
        .collect()
}

impl<A> Node<A> {
    pub fn new() -> Self {
        Node {
            children: make_empty(),
        }
    }

    pub fn from_vec(mut children: Vec<Entry<A>>) -> Self {
        while children.len() < HASH_SIZE {
            children.push(Entry::Empty);
        }
        Node { children }
    }
}

impl<A> Clone for Node<A> {
    fn clone(&self) -> Self {
        Node {
            children: self.children.clone(),
        }
    }
}

impl<A> Default for Node<A> {
    fn default() -> Self {
        Node::new()
    }
}

pub fn push_tail<A>(count: usize, level: usize, parent: &Node<A>, tail_node: Node<A>) -> Node<A> {
    let sub_index = ((count - 1) >> level) & HASH_MASK as usize;
    let mut out = parent.clone();
    out.children[sub_index] = Entry::Node(Arc::new(if level == HASH_BITS {
        tail_node
    } else {
        match parent.children[sub_index] {
            Entry::Node(ref child) => push_tail(count, level - HASH_BITS, child, tail_node),
            Entry::Empty => new_path(level - HASH_BITS, tail_node),
            Entry::Value(_) => {
                panic!("Vector::push_tail: encountered value where node was expected")
            }
        }
    }));
    out
}

pub fn push_tail_mut<A>(count: usize, level: usize, parent: &mut Node<A>, tail_node: Node<A>) {
    let sub_index = ((count - 1) >> level) & HASH_MASK as usize;
    parent.children[sub_index] = Entry::Node(Arc::new(if level == HASH_BITS {
        tail_node
    } else {
        match parent.children[sub_index] {
            Entry::Node(ref mut child_ref) => {
                let child = Arc::make_mut(child_ref);
                push_tail_mut(count, level - HASH_BITS, child, tail_node);
                return;
            }
            Entry::Empty => new_path(level - HASH_BITS, tail_node),
            Entry::Value(_) => {
                panic!("Vector::push_tail: encountered value where node was expected")
            }
        }
    }));
}

pub fn pop_tail<A>(count: usize, level: usize, node: &Node<A>) -> Option<Node<A>> {
    let sub_index = ((count - 2) >> level) & HASH_MASK as usize;
    if level > HASH_BITS {
        match pop_tail(
            count,
            level - HASH_BITS,
            &node.children[sub_index].unwrap_node(),
        ) {
            None if sub_index == 0 => None,
            None => {
                let mut out = node.clone();
                out.children[sub_index] = Entry::Empty;
                Some(out)
            }
            Some(new_child) => {
                let mut out = node.clone();
                out.children[sub_index] = Entry::Node(Arc::new(new_child));
                Some(out)
            }
        }
    } else if sub_index == 0 {
        None
    } else {
        let mut out = node.clone();
        out.children[sub_index] = Entry::Empty;
        Some(out)
    }
}

pub fn pop_tail_mut<A>(count: usize, level: usize, node: &mut Node<A>) -> bool {
    let sub_index = ((count - 2) >> level) & HASH_MASK as usize;
    if level > HASH_BITS {
        let mut child_ref = node.children[sub_index].unwrap_node();
        let child = Arc::make_mut(&mut child_ref);
        if !pop_tail_mut(count, level - HASH_BITS, child) {
            if sub_index == 0 {
                false
            } else {
                node.children[sub_index] = Entry::Empty;
                true
            }
        } else {
            true
        }
    } else if sub_index == 0 {
        false
    } else {
        node.children[sub_index] = Entry::Empty;
        true
    }
}

pub fn new_path<A>(level: usize, node: Node<A>) -> Node<A> {
    if level == 0 {
        node
    } else {
        let mut out = Node::new();
        out.children[0] = Entry::Node(Arc::new(new_path(level - HASH_BITS, node)));
        out
    }
}

pub fn set_in<A>(level: usize, node: &Node<A>, index: usize, value: Arc<A>) -> Node<A> {
    let mut out: Node<A> = node.clone();
    if level == 0 {
        out.children[index & HASH_MASK as usize] = Entry::Value(value);
    } else {
        let sub_index = (index >> level) & HASH_MASK as usize;
        if let Entry::Node(ref sub_node) = node.children[sub_index] {
            out.children[sub_index] =
                Entry::Node(Arc::new(set_in(level - HASH_BITS, sub_node, index, value)));
        } else {
            panic!("Vector::set_in: found non-node where node was expected");
        }
    }
    out
}

pub fn set_in_mut<A>(level: usize, node: &mut Node<A>, index: usize, value: Arc<A>) {
    if level == 0 {
        node.children[index & HASH_MASK as usize] = Entry::Value(value);
    } else {
        let sub_index = (index >> level) & HASH_MASK as usize;
        if let Entry::Node(ref mut sub_node_ref) = node.children[sub_index] {
            let mut sub_node = Arc::make_mut(sub_node_ref);
            set_in_mut(level - HASH_BITS, sub_node, index, value);
        } else {
            panic!("Vector::set_in_mut: found non-node where node was expected");
        }
    }
}
