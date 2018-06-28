// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::iter::FusedIterator;
use std::mem;

use bits::{bitpos, index, Bitmap, HASH_BITS, HASH_SIZE};
use util::Ref;

pub trait HashValue: Clone {
    type Key: Eq;

    fn extract_key(&self) -> &Self::Key;
    fn ptr_eq(&self, other: &Self) -> bool;
}

#[derive(PartialEq, Eq, Clone)]
pub struct Node<A> {
    datamap: Bitmap,
    nodemap: Bitmap,
    data: Vec<Entry<A>>,
    nodes: Vec<Ref<Node<A>>>,
}

#[derive(PartialEq, Eq, Clone)]
pub struct CollisionNode<A> {
    hash: Bitmap,
    data: Vec<A>,
}

#[derive(PartialEq, Eq, Clone)]
pub enum Entry<A> {
    Value(A, Bitmap),
    Collision(Ref<CollisionNode<A>>),
}

impl<A> Entry<A> {
    fn unwrap_value(self) -> A {
        match self {
            Entry::Value(a, _) => a,
            _ => panic!("nodes::hamt::Entry::unwrap_value: unwrapped a non-value"),
        }
    }
}

enum SizePredicate {
    Empty,
    One,
    Many,
}

impl<A: HashValue> Default for Node<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: HashValue> Node<A> {
    #[inline]
    pub fn iter<'a>(root: &'a Self, size: usize) -> Iter<'a, A>
    where
        A: 'a,
    {
        Iter::new(root, size)
    }

    #[inline]
    pub fn new() -> Self {
        Node {
            datamap: 0,
            nodemap: 0,
            data: Vec::new(),
            nodes: Vec::new(),
        }
    }

    #[inline]
    pub fn singleton(bitpos: Bitmap, value: Entry<A>) -> Self {
        Node {
            datamap: bitpos,
            data: vec![value],
            nodemap: 0,
            nodes: Vec::new(),
        }
    }

    #[inline]
    pub fn pair(bitmap: Bitmap, value1: Entry<A>, value2: Entry<A>) -> Self {
        Node {
            datamap: bitmap,
            data: vec![value1, value2],
            nodemap: 0,
            nodes: Vec::new(),
        }
    }

    #[inline]
    pub fn single_child(bitpos: Bitmap, node: Self) -> Self {
        Node {
            datamap: 0,
            data: Vec::new(),
            nodemap: bitpos,
            nodes: vec![Ref::new(node)],
        }
    }

    #[inline]
    fn data_arity(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn node_arity(&self) -> usize {
        self.nodes.len()
    }

    fn size_predicate(&self) -> SizePredicate {
        if self.node_arity() == 0 {
            match self.data_arity() {
                0 => SizePredicate::Empty,
                1 => SizePredicate::One,
                _ => SizePredicate::Many,
            }
        } else {
            SizePredicate::Many
        }
    }

    #[inline]
    fn data_index(&self, bitpos: Bitmap) -> usize {
        index(self.datamap, bitpos)
    }

    #[inline]
    fn node_index(&self, bitpos: Bitmap) -> usize {
        index(self.nodemap, bitpos)
    }

    fn update_value(&mut self, bitpos: Bitmap, value: Entry<A>) -> Entry<A> {
        let index = self.data_index(bitpos);
        mem::replace(&mut self.data[index], value)
    }

    fn insert_value(&mut self, bitpos: Bitmap, value: Entry<A>) {
        let index = self.data_index(bitpos);
        self.data.insert(index, value);
        self.datamap |= bitpos;
    }

    fn remove_value(&mut self, bitpos: Bitmap) {
        let index = self.data_index(bitpos);
        self.data.remove(index);
        self.datamap ^= bitpos;
    }

    fn value_to_node<RN>(&mut self, bitpos: Bitmap, node: RN)
    where
        Ref<Node<A>>: From<RN>,
    {
        let old_index = self.data_index(bitpos);
        let new_index = self.node_index(bitpos);
        self.data.remove(old_index);
        self.datamap ^= bitpos;
        self.nodes.insert(new_index, From::from(node));
        self.nodemap |= bitpos;
    }

    fn node_to_value(&mut self, bitpos: Bitmap, value: Entry<A>) {
        let old_index = self.node_index(bitpos);
        let new_index = self.data_index(bitpos);
        self.nodes.remove(old_index);
        self.nodemap ^= bitpos;
        self.data.insert(new_index, value);
        self.datamap |= bitpos;
    }

    fn merge_values(
        value1: Entry<A>,
        value2: Entry<A>,
        hash1: Bitmap,
        hash2: Bitmap,
        shift: usize,
    ) -> Self {
        let bitpos1 = bitpos(hash1, shift);
        let bitpos2 = bitpos(hash2, shift);
        if bitpos1 != bitpos2 {
            // Both values fit on the same level.
            let datamap = bitpos1 | bitpos2;
            if bitpos1 < bitpos2 {
                Node::pair(datamap, value1, value2)
            } else {
                Node::pair(datamap, value2, value1)
            }
        } else {
            // If we're at the bottom, we've got a collision.
            if shift + HASH_BITS >= HASH_SIZE {
                return Node::singleton(
                    bitpos(hash1, shift),
                    Entry::Collision(Ref::new(CollisionNode::new(hash1, value1, value2))),
                );
            }
            // Pass the values down a level.
            let node = Node::merge_values(value1, value2, hash1, hash2, shift + HASH_BITS);
            Node::single_child(bitpos1, node)
        }
    }

    pub fn get<BK>(&self, hash: Bitmap, shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let bitpos = bitpos(hash, shift);
        if self.datamap & bitpos != 0 {
            match self.data[self.data_index(bitpos)] {
                Entry::Value(ref value, _) => if key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                },
                Entry::Collision(ref coll) => coll.get(key),
            }
        } else if self.nodemap & bitpos != 0 {
            self.nodes[self.node_index(bitpos)].get(hash, shift + HASH_BITS, key)
        } else {
            None
        }
    }

    pub fn get_mut<BK>(&mut self, hash: Bitmap, shift: usize, key: &BK) -> Option<&mut A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let bitpos = bitpos(hash, shift);
        let data_index = self.data_index(bitpos);
        let node_index = self.node_index(bitpos);
        if self.datamap & bitpos != 0 {
            match self.data[data_index] {
                Entry::Value(ref mut value, _) => if key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                },
                Entry::Collision(ref mut coll_ref) => {
                    let coll = Ref::make_mut(coll_ref);
                    coll.get_mut(key)
                }
            }
        } else if self.nodemap & bitpos != 0 {
            let child = Ref::make_mut(&mut self.nodes[node_index]);
            child.get_mut(hash, shift + HASH_BITS, key)
        } else {
            None
        }
    }

    pub fn insert(&mut self, hash: Bitmap, shift: usize, value: A) -> Option<A> {
        let bitpos = bitpos(hash, shift);
        if self.datamap & bitpos != 0 {
            // Value is here
            let index = self.data_index(bitpos);
            let mut insert = false;
            let mut merge = None;
            match self.data[index] {
                // Update value or create a subtree
                Entry::Value(ref current, hash2) => {
                    if value.extract_key() == current.extract_key() {
                        insert = true;
                    } else {
                        merge = Some((current.clone(), hash2));
                    }
                }
                // There's already a collision here.
                Entry::Collision(ref mut collision) => {
                    let coll = Ref::make_mut(collision);
                    return coll.insert(value);
                }
            }
            if insert {
                return Some(
                    self.update_value(bitpos, Entry::Value(value, hash))
                        .unwrap_value(),
                );
            }
            if let Some((value2, hash2)) = merge {
                if shift + HASH_BITS >= HASH_SIZE {
                    // Need to set up a collision
                    let coll = CollisionNode::new(
                        hash,
                        Entry::Value(value, hash),
                        Entry::Value(value2, hash2),
                    );
                    self.update_value(bitpos, Entry::Collision(Ref::new(coll)));
                    return None;
                } else {
                    let node = Node::merge_values(
                        Entry::Value(value, hash),
                        Entry::Value(value2, hash2),
                        hash,
                        hash2,
                        shift + HASH_BITS,
                    );
                    self.value_to_node(bitpos, Ref::new(node));
                    return None;
                }
            }
            unreachable!()
        } else if self.nodemap & bitpos != 0 {
            // Child node
            let index = self.node_index(bitpos);
            let child = Ref::make_mut(&mut self.nodes[index]);
            child.insert(hash, shift + HASH_BITS, value)
        } else {
            // New value
            self.insert_value(bitpos, Entry::Value(value, hash));
            None
        }
    }

    pub fn remove<BK>(&mut self, hash: Bitmap, shift: usize, key: &BK) -> Option<A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let pos = bitpos(hash, shift);
        if self.datamap & pos != 0 {
            // Key is (or would be) in this node.
            let index = self.data_index(pos);
            let removed;
            match self.data[index] {
                Entry::Value(ref value, _) => if key == value.extract_key().borrow() {
                    removed = value.clone();
                } else {
                    return None;
                },
                Entry::Collision(ref mut collisions) => {
                    let mut coll = Ref::make_mut(collisions);
                    return coll.remove(key);
                }
            }
            self.remove_value(pos);
            Some(removed)
        } else if self.nodemap & pos != 0 {
            // Key is in a subnode.
            let index = self.node_index(pos);
            let removed;
            let remaining;
            {
                let child = Ref::make_mut(&mut self.nodes[index]);
                match child.remove(hash, shift + HASH_BITS, key) {
                    None => return None,
                    Some(value) => match child.size_predicate() {
                        SizePredicate::Empty => panic!(
                            "HashMap::remove: encountered unexpectedly empty subnode after removal"
                        ),
                        SizePredicate::One => {
                            removed = value;
                            remaining = child.data[0].clone();
                        }
                        SizePredicate::Many => return Some(value),
                    },
                }
            }
            // Subnode has single value if we get here, merge it into self.
            self.node_to_value(pos, remaining);
            Some(removed)
        } else {
            // Key isn't here.
            None
        }
    }
}

impl<A: HashValue> CollisionNode<A> {
    fn new(hash: Bitmap, value1: Entry<A>, value2: Entry<A>) -> Self {
        let mut data = Vec::new();
        match value1 {
            Entry::Value(value, _) => data.push(value),
            Entry::Collision(ref node) => data.extend(node.data.iter().cloned()),
        }
        match value2 {
            Entry::Value(value, _) => data.push(value),
            Entry::Collision(ref node) => data.extend(node.data.iter().cloned()),
        }
        CollisionNode { hash, data }
    }

    fn get<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        for entry in &self.data {
            if key == entry.extract_key().borrow() {
                return Some(entry);
            }
        }
        None
    }

    fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        for entry in &mut self.data {
            if key == entry.extract_key().borrow() {
                return Some(entry);
            }
        }
        None
    }

    fn insert(&mut self, value: A) -> Option<A> {
        for item in &mut self.data {
            if value.extract_key() == item.extract_key() {
                return Some(mem::replace(item, value));
            }
        }
        self.data.push(value);
        None
    }

    fn remove<BK>(&mut self, key: &BK) -> Option<A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let mut loc = None;
        for (index, item) in self.data.iter().enumerate() {
            if key == item.extract_key().borrow() {
                loc = Some(index);
            }
        }
        if let Some(index) = loc {
            Some(self.data.remove(index))
        } else {
            None
        }
    }
}

// Ref iterator

pub struct Iter<'a, A>
where
    A: 'a,
{
    count: usize,
    stack: Vec<(&'a Node<A>, usize)>,
    node: &'a Node<A>,
    index: usize,
    nodes: bool,
    collision: Option<&'a CollisionNode<A>>,
    coll_index: usize,
}

impl<'a, A> Iter<'a, A>
where
    A: 'a,
{
    pub fn new(root: &'a Node<A>, size: usize) -> Self {
        Iter {
            count: size,
            stack: Vec::with_capacity((HASH_SIZE / HASH_BITS) + 1),
            node: root,
            index: 0,
            nodes: false,
            collision: None,
            coll_index: 0,
        }
    }
}

impl<'a, A> Iterator for Iter<'a, A>
where
    A: 'a,
{
    type Item = (&'a A, Bitmap);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref coll) = self.collision {
            if coll.data.len() > self.coll_index {
                let value = &coll.data[self.coll_index];
                self.coll_index += 1;
                self.count -= 1;
                return Some((value, coll.hash));
            }
        }
        self.collision = None;
        if !self.nodes {
            if self.node.data.len() > self.index {
                match self.node.data[self.index] {
                    Entry::Value(ref value, hash) => {
                        self.index += 1;
                        self.count -= 1;
                        return Some((value, hash));
                    }
                    Entry::Collision(ref coll) => {
                        self.index += 1;
                        self.collision = Some(coll);
                        self.coll_index = 0;
                    }
                }
                return self.next();
            }
            self.index = 0;
            self.nodes = true;
        }
        if self.node.nodes.len() > self.index {
            self.stack.push((self.node, self.index + 1));
            self.node = &self.node.nodes[self.index];
            self.index = 0;
            self.nodes = false;
            return self.next();
        }
        match self.stack.pop() {
            None => None,
            Some((node, index)) => {
                self.node = node;
                self.index = index;
                self.nodes = true;
                self.next()
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A> ExactSizeIterator for Iter<'a, A> where A: 'a {}

impl<'a, A> FusedIterator for Iter<'a, A> where A: 'a {}

// Consuming iterator

pub struct ConsumingIter<A> {
    count: usize,
    stack: Vec<(Ref<Node<A>>, usize)>,
    node: Ref<Node<A>>,
    index: usize,
    nodes: bool,
    collision: Option<Ref<CollisionNode<A>>>,
    coll_index: usize,
}

impl<A> ConsumingIter<A> {
    pub fn new(root: Ref<Node<A>>, size: usize) -> Self {
        ConsumingIter {
            count: size,
            stack: Vec::with_capacity((HASH_SIZE / HASH_BITS) + 1),
            node: root,
            index: 0,
            nodes: false,
            collision: None,
            coll_index: 0,
        }
    }
}

impl<A: Clone> Iterator for ConsumingIter<A> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref coll) = self.collision {
            if coll.data.len() > self.coll_index {
                let value = coll.data[self.coll_index].clone();
                self.coll_index += 1;
                self.count -= 1;
                return Some(value);
            }
        }
        self.collision = None;
        if !self.nodes {
            if self.node.data.len() > self.index {
                match self.node.data[self.index] {
                    Entry::Value(ref value, _) => {
                        self.index += 1;
                        self.count -= 1;
                        return Some(value.clone());
                    }
                    Entry::Collision(ref coll) => {
                        self.index += 1;
                        self.collision = Some(coll.clone());
                        self.coll_index = 0;
                    }
                }
                return self.next();
            }
            self.index = 0;
            self.nodes = true;
        }
        if self.node.nodes.len() > self.index {
            self.stack.push((self.node.clone(), self.index + 1));
            self.node = self.node.nodes[self.index].clone();
            self.index = 0;
            self.nodes = false;
            return self.next();
        }
        match self.stack.pop() {
            None => None,
            Some((node, index)) => {
                self.node = node;
                self.index = index;
                self.nodes = true;
                self.next()
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<A: Clone> ExactSizeIterator for ConsumingIter<A> {}

impl<A: Clone> FusedIterator for ConsumingIter<A> {}
