// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::sync::Arc;

use bits::{bitpos, index, Bitmap, HASH_BITS, HASH_SIZE};
use shared::Shared;

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
    nodes: Vec<Arc<Node<A>>>,
}

#[derive(PartialEq, Eq, Clone)]
pub struct CollisionNode<A> {
    hash: Bitmap,
    data: Vec<A>,
}

#[derive(PartialEq, Eq, Clone)]
pub enum Entry<A> {
    Value(A, Bitmap),
    Collision(Arc<CollisionNode<A>>),
}

enum SizePredicate {
    Empty,
    One,
    Many,
}

impl<A: HashValue> Node<A> {
    #[inline]
    pub fn iter(root: Arc<Self>, size: usize) -> Iter<A> {
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
            nodes: vec![Arc::new(node)],
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

    fn update_node<RN>(&self, bitpos: Bitmap, node: RN) -> Self
    where
        RN: Shared<Node<A>>,
    {
        let index = self.node_index(bitpos);
        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        new_nodes.extend(self.nodes.iter().cloned().take(index));
        new_nodes.push(node.shared());
        new_nodes.extend(self.nodes.iter().cloned().skip(index + 1));
        Node {
            nodemap: self.nodemap,
            nodes: new_nodes,
            datamap: self.datamap,
            data: self.data.clone(),
        }
    }

    fn update_value(&self, bitpos: Bitmap, value: Entry<A>) -> Self {
        let index = self.data_index(bitpos);
        let mut new_data = Vec::with_capacity(self.data.len());
        new_data.extend(self.data.iter().cloned().take(index));
        new_data.push(value);
        new_data.extend(self.data.iter().cloned().skip(index + 1));
        Node {
            nodemap: self.nodemap,
            nodes: self.nodes.clone(),
            datamap: self.datamap,
            data: new_data,
        }
    }

    fn update_value_mut(&mut self, bitpos: Bitmap, value: Entry<A>) {
        let index = self.data_index(bitpos);
        self.data[index] = value;
    }

    fn insert_value(&self, bitpos: Bitmap, value: Entry<A>) -> Self {
        let index = self.data_index(bitpos);
        let mut new_data = Vec::with_capacity(self.data.len() + 1);
        new_data.extend(self.data.iter().cloned().take(index));
        new_data.push(value);
        new_data.extend(self.data.iter().cloned().skip(index));
        Node {
            nodemap: self.nodemap,
            nodes: self.nodes.clone(),
            datamap: self.datamap | bitpos,
            data: new_data,
        }
    }

    fn insert_value_mut(&mut self, bitpos: Bitmap, value: Entry<A>) {
        let index = self.data_index(bitpos);
        self.data.insert(index, value);
        self.datamap |= bitpos;
    }

    fn remove_value(&self, bitpos: Bitmap) -> Self {
        let index = self.data_index(bitpos);
        let mut new_data = Vec::with_capacity(self.data.len() - 1);
        new_data.extend(self.data.iter().cloned().take(index));
        new_data.extend(self.data.iter().cloned().skip(index + 1));
        Node {
            nodemap: self.nodemap,
            nodes: self.nodes.clone(),
            datamap: self.datamap ^ bitpos,
            data: new_data,
        }
    }

    fn remove_value_mut(&mut self, bitpos: Bitmap) {
        let index = self.data_index(bitpos);
        self.data.remove(index);
        self.datamap ^= bitpos;
    }

    fn value_to_node<RN>(&self, bitpos: Bitmap, node: RN) -> Self
    where
        RN: Shared<Node<A>>,
    {
        let old_index = self.data_index(bitpos);
        let new_index = self.node_index(bitpos);

        let mut new_data = Vec::with_capacity(self.data.len() - 1);
        new_data.extend(self.data.iter().cloned().take(old_index));
        new_data.extend(self.data.iter().cloned().skip(old_index + 1));

        let mut new_nodes = Vec::with_capacity(self.nodes.len() + 1);
        new_nodes.extend(self.nodes.iter().cloned().take(new_index));
        new_nodes.push(node.shared());
        new_nodes.extend(self.nodes.iter().cloned().skip(new_index));

        Node {
            nodemap: self.nodemap | bitpos,
            nodes: new_nodes,
            datamap: self.datamap ^ bitpos,
            data: new_data,
        }
    }

    fn value_to_node_mut<RN>(&mut self, bitpos: Bitmap, node: RN)
    where
        RN: Shared<Node<A>>,
    {
        let old_index = self.data_index(bitpos);
        let new_index = self.node_index(bitpos);
        self.data.remove(old_index);
        self.datamap ^= bitpos;
        self.nodes.insert(new_index, node.shared());
        self.nodemap |= bitpos;
    }

    fn node_to_value(&self, bitpos: Bitmap, node: &Node<A>) -> Self {
        let old_index = self.node_index(bitpos);
        let new_index = self.data_index(bitpos);

        let mut new_data = Vec::with_capacity(self.data.len() + 1);
        new_data.extend(self.data.iter().cloned().take(new_index));
        new_data.push(node.data[0].clone());
        new_data.extend(self.data.iter().cloned().skip(new_index));

        let mut new_nodes = Vec::with_capacity(self.nodes.len() - 1);
        new_nodes.extend(self.nodes.iter().cloned().take(old_index));
        new_nodes.extend(self.nodes.iter().cloned().skip(old_index + 1));

        Node {
            nodemap: self.nodemap ^ bitpos,
            nodes: new_nodes,
            datamap: self.datamap | bitpos,
            data: new_data,
        }
    }

    fn node_to_value_mut(&mut self, bitpos: Bitmap, value: Entry<A>) {
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
                    Entry::Collision(Arc::new(CollisionNode::new(hash1, value1, value2))),
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
                    let coll = Arc::make_mut(coll_ref);
                    coll.get_mut(key)
                }
            }
        } else if self.nodemap & bitpos != 0 {
            let child = Arc::make_mut(&mut self.nodes[node_index]);
            child.get_mut(hash, shift + HASH_BITS, key)
        } else {
            None
        }
    }

    pub fn insert(&self, hash: Bitmap, shift: usize, value: A) -> (bool, Self) {
        let bitpos = bitpos(hash, shift);
        if self.datamap & bitpos != 0 {
            // Value is here
            let index = self.data_index(bitpos);
            match self.data[index] {
                // Update value or create a subtree
                Entry::Value(ref current, hash2) => {
                    if value.extract_key() == current.extract_key() {
                        (false, self.update_value(bitpos, Entry::Value(value, hash)))
                    } else {
                        let value2 = current.clone();
                        if shift + HASH_BITS >= HASH_SIZE {
                            // Need to set up a collision
                            let coll = CollisionNode::new(
                                hash,
                                Entry::Value(value, hash),
                                Entry::Value(value2, hash),
                            );
                            (
                                true,
                                self.update_value(bitpos, Entry::Collision(Arc::new(coll))),
                            )
                        } else {
                            let node = Node::merge_values(
                                Entry::Value(value, hash),
                                Entry::Value(value2, hash2),
                                hash,
                                hash2,
                                shift + HASH_BITS,
                            );
                            (true, self.value_to_node(bitpos, Arc::new(node)))
                        }
                    }
                }
                // There's already a collision here.
                Entry::Collision(ref coll) => {
                    let (added, new_coll) = coll.insert(value);
                    (
                        added,
                        self.update_value(bitpos, Entry::Collision(Arc::new(new_coll))),
                    )
                }
            }
        } else if self.nodemap & bitpos != 0 {
            // Child node
            let index = self.node_index(bitpos);
            let child = &self.nodes[index];
            let (added, new_child) = child.insert(hash, shift + HASH_BITS, value);
            (added, self.update_node(bitpos, Arc::new(new_child)))
        } else {
            // New value
            (true, self.insert_value(bitpos, Entry::Value(value, hash)))
        }
    }

    pub fn insert_mut(&mut self, hash: Bitmap, shift: usize, value: A) -> bool {
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
                    let coll = Arc::make_mut(collision);
                    return coll.insert_mut(value);
                }
            }
            if insert {
                self.update_value_mut(bitpos, Entry::Value(value, hash));
                return false;
            }
            if let Some((value2, hash2)) = merge {
                if shift + HASH_BITS >= HASH_SIZE {
                    // Need to set up a collision
                    let coll = CollisionNode::new(
                        hash,
                        Entry::Value(value, hash),
                        Entry::Value(value2, hash2),
                    );
                    self.update_value_mut(bitpos, Entry::Collision(Arc::new(coll)));
                    return true;
                } else {
                    let node = Node::merge_values(
                        Entry::Value(value, hash),
                        Entry::Value(value2, hash2),
                        hash,
                        hash2,
                        shift + HASH_BITS,
                    );
                    self.value_to_node_mut(bitpos, Arc::new(node));
                    return true;
                }
            }
            unreachable!()
        } else if self.nodemap & bitpos != 0 {
            // Child node
            let index = self.node_index(bitpos);
            let child = Arc::make_mut(&mut self.nodes[index]);
            child.insert_mut(hash, shift + HASH_BITS, value)
        } else {
            // New value
            self.insert_value_mut(bitpos, Entry::Value(value, hash));
            true
        }
    }

    pub fn remove<BK>(&self, hash: Bitmap, shift: usize, key: &BK) -> Option<(A, Self)>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let pos = bitpos(hash, shift);
        if self.datamap & pos != 0 {
            // Key is (or would be) in this node.
            let index = self.data_index(pos);
            match self.data[index] {
                Entry::Value(ref value, _) => if key == value.extract_key().borrow() {
                    Some((value.clone(), self.remove_value(pos)))
                // }
                } else {
                    None
                },
                Entry::Collision(ref coll) => match coll.remove(key) {
                    None => None,
                    Some((value, next_coll)) => Some((
                        value,
                        self.update_value(pos, Entry::Collision(Arc::new(next_coll))),
                    )),
                },
            }
        } else if self.nodemap & pos != 0 {
            // Key is in a subnode.
            let remove_result =
                self.nodes[self.node_index(pos)].remove(hash, shift + HASH_BITS, key);
            match remove_result {
                None => None,
                Some((value, next_node)) => {
                    match next_node.size_predicate() {
                        SizePredicate::Empty => panic!(
                            "HashMap::remove: encountered unexpectedly empty subnode after removal"
                        ),
                        SizePredicate::One => {
                            // Subnode has single value, merge it into self.
                            Some((value, self.node_to_value(pos, &next_node)))
                        }
                        SizePredicate::Many => Some((value, self.update_node(pos, next_node))),
                    }
                }
            }
        } else {
            // Key isn't here.
            None
        }
    }

    pub fn remove_mut<BK>(&mut self, hash: Bitmap, shift: usize, key: &BK) -> Option<A>
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
                    let mut coll = Arc::make_mut(collisions);
                    return coll.remove_mut(key);
                }
            }
            self.remove_value_mut(pos);
            Some(removed)
        } else if self.nodemap & pos != 0 {
            // Key is in a subnode.
            let index = self.node_index(pos);
            let removed;
            let remaining;
            {
                let child = Arc::make_mut(&mut self.nodes[index]);
                match child.remove_mut(hash, shift + HASH_BITS, key) {
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
            self.node_to_value_mut(pos, remaining);
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

    fn insert(&self, value: A) -> (bool, Self) {
        let mut data = Vec::with_capacity(self.data.len() + 1);
        let mut add = true;
        for item in &self.data {
            if value.extract_key() == item.extract_key() {
                data.push(value.clone());
                add = false;
            } else {
                data.push(item.clone());
            }
        }
        if add {
            data.push(value);
        }
        (
            add,
            CollisionNode {
                hash: self.hash,
                data,
            },
        )
    }

    fn insert_mut(&mut self, value: A) -> bool {
        for item in &mut self.data {
            if value.extract_key() == item.extract_key() {
                *item = value;
                return false;
            }
        }
        self.data.push(value);
        true
    }

    fn remove<BK>(&self, key: &BK) -> Option<(A, Self)>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        for (index, item) in self.data.iter().enumerate() {
            if key == item.extract_key().borrow() {
                let mut data = self.data.clone();
                let value = data.remove(index);
                return Some((
                    value,
                    CollisionNode {
                        hash: self.hash,
                        data,
                    },
                ));
            }
        }
        None
    }

    fn remove_mut<BK>(&mut self, key: &BK) -> Option<A>
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

// Iterators

pub struct Iter<A> {
    count: usize,
    stack: Vec<(Arc<Node<A>>, usize)>,
    node: Arc<Node<A>>,
    index: usize,
    nodes: bool,
    collision: Option<Arc<CollisionNode<A>>>,
    coll_index: usize,
}

impl<A> Iter<A> {
    fn new(root: Arc<Node<A>>, size: usize) -> Self {
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

impl<A: Clone> Iterator for Iter<A> {
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

impl<A: Clone> ExactSizeIterator for Iter<A> {}
