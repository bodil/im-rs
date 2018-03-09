#![cfg_attr(feature = "clippy", allow(new_ret_no_self))]

use std::sync::Arc;
use std::hash::{BuildHasher, Hash};
use std::fmt::{Debug, Error, Formatter};
use std::ptr;

use bits::{bit_index, bitpos, mask, Bitmap, HASH_BITS, HASH_SIZE};
use hash::hash_key;

pub enum Node<K, V> {
    ArrayNode(Arc<ArrayNode<K, V>>),
    BitmapNode(Arc<BitmapNode<K, V>>),
    CollisionNode(Arc<CollisionNode<K, V>>),
}

impl<K, V> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        match *self {
            Node::ArrayNode(ref node) => Node::ArrayNode(node.clone()),
            Node::BitmapNode(ref node) => Node::BitmapNode(node.clone()),
            Node::CollisionNode(ref node) => Node::CollisionNode(node.clone()),
        }
    }
}

// ArrayNode

pub struct ArrayNode<K, V> {
    size: usize,
    content: Vec<Option<Node<K, V>>>,
}

impl<K, V> Clone for ArrayNode<K, V> {
    fn clone(&self) -> Self {
        ArrayNode {
            size: self.size,
            content: self.content.clone(),
        }
    }
}

impl<K, V> ArrayNode<K, V>
where
    K: Hash + Eq,
{
    fn new(size: usize, content: Vec<Option<Node<K, V>>>) -> Node<K, V> {
        Node::ArrayNode(Arc::new(ArrayNode { size, content }))
    }

    fn pack(&self, remove_idx: usize) -> Node<K, V> {
        let (new_bitmap, new_content) = self.content.iter().cloned().enumerate().fold(
            (0, Vec::new()),
            |(bitmap, mut content), (idx, maybe_node)| match maybe_node {
                _ if idx == remove_idx => (bitmap, content),
                None => (bitmap, content),
                Some(node) => {
                    content.push(Entry::Node(node));
                    (bitmap | 1 << idx, content)
                }
            },
        );
        BitmapNode::new(new_bitmap, new_content)
    }
}

// BitmapNode

pub struct BitmapNode<K, V> {
    bitmap: Bitmap,
    content: Vec<Entry<K, V>>,
}

impl<K, V> Clone for BitmapNode<K, V> {
    fn clone(&self) -> Self {
        BitmapNode {
            bitmap: self.bitmap,
            content: self.content.clone(),
        }
    }
}

enum Entry<K, V> {
    Pair(Arc<K>, Arc<V>),
    Node(Node<K, V>),
}

impl<K, V> Clone for Entry<K, V> {
    fn clone(&self) -> Self {
        match *self {
            Entry::Pair(ref k, ref v) => Entry::Pair(k.clone(), v.clone()),
            Entry::Node(ref node) => Entry::Node(node.clone()),
        }
    }
}

impl<K, V> BitmapNode<K, V>
where
    K: Hash + Eq,
{
    fn new(bitmap: Bitmap, content: Vec<Entry<K, V>>) -> Node<K, V> {
        Node::BitmapNode(Arc::new(BitmapNode { bitmap, content }))
    }
}

// CollisionNode

pub struct CollisionNode<K, V> {
    hash: Bitmap,
    size: usize,
    content: Vec<(Arc<K>, Arc<V>)>,
}

impl<K, V> Clone for CollisionNode<K, V> {
    fn clone(&self) -> Self {
        CollisionNode {
            hash: self.hash,
            size: self.size,
            content: self.content.clone(),
        }
    }
}

impl<K, V> CollisionNode<K, V>
where
    K: Hash + Eq,
{
    fn new(hash: Bitmap, size: usize, content: Vec<(Arc<K>, Arc<V>)>) -> Node<K, V> {
        Node::CollisionNode(Arc::new(CollisionNode {
            hash,
            size,
            content,
        }))
    }

    fn find_index(&self, key: &K) -> Option<usize> {
        self.content.iter().position(|&(ref k, _)| &**k == key)
    }
}

// Top level Node

pub enum RemoveResult<Node> {
    Unchanged,
    Removed,
    NewNode(Node),
}

impl<K, V> Node<K, V>
where
    K: Hash + Eq,
{
    pub fn single(bitmap: Bitmap, node: &Self) -> Self {
        BitmapNode::new(bitmap, vec![Entry::Node(node.clone())])
    }

    pub fn empty() -> Self {
        BitmapNode::new(0, Vec::new())
    }

    pub fn insert<S>(
        &self,
        hasher: &S,
        shift: usize,
        hash: Bitmap,
        key: &Arc<K>,
        value: &Arc<V>,
    ) -> (bool, Self)
    where
        S: BuildHasher,
    {
        match *self {
            // Insert for ArrayNode
            Node::ArrayNode(ref node) => {
                let idx = mask(hash, shift) as usize;
                match node.content[idx] {
                    None => {
                        let nil = Node::empty();
                        let (added, new_node) =
                            nil.insert(hasher, shift + HASH_BITS, hash, key, value);
                        (
                            added,
                            ArrayNode::new(
                                node.size + 1,
                                update(&node.content, idx, Some(new_node)),
                            ),
                        )
                    }
                    Some(ref idx_node) => {
                        let (added, new_node) =
                            idx_node.insert(hasher, shift + HASH_BITS, hash, key, value);
                        if idx_node.ptr_eq(&new_node) {
                            (added, self.clone())
                        } else {
                            (
                                added,
                                ArrayNode::new(
                                    node.size,
                                    update(&node.content, idx, Some(new_node)),
                                ),
                            )
                        }
                    }
                }
            }

            // Insert for BitmapNode
            Node::BitmapNode(ref node) => {
                let bit = bitpos(hash, shift);
                let idx = bit_index(node.bitmap, bit);

                // If flag set in bitmap
                if node.bitmap & bit != 0 {
                    match node.content[idx] {
                        // Bitmap position has a child node
                        Entry::Node(ref idx_node) => {
                            let (added, new_node) =
                                idx_node.insert(hasher, shift + HASH_BITS, hash, key, value);
                            if idx_node.ptr_eq(&new_node) {
                                (added, self.clone())
                            } else {
                                (
                                    added,
                                    BitmapNode::new(
                                        node.bitmap,
                                        update(&node.content, idx, Entry::Node(new_node)),
                                    ),
                                )
                            }
                        }

                        // Bitmap position contains the current key
                        Entry::Pair(ref old_key, ref old_value) if old_key == key => {
                            if Arc::ptr_eq(value, old_value) {
                                (false, self.clone())
                            } else {
                                (
                                    false,
                                    BitmapNode::new(
                                        node.bitmap,
                                        update(
                                            &node.content,
                                            idx,
                                            Entry::Pair(key.clone(), value.clone()),
                                        ),
                                    ),
                                )
                            }
                        }

                        // Bitmap position contains a different key
                        Entry::Pair(ref old_key, ref old_value) => {
                            let new_node = create_node(
                                hasher,
                                shift + HASH_BITS,
                                old_key,
                                old_value,
                                hash,
                                key,
                                value,
                            );
                            (
                                true,
                                BitmapNode::new(
                                    node.bitmap,
                                    update(&node.content, idx, Entry::Node(new_node)),
                                ),
                            )
                        }
                    }
                } else {
                    // No flag set in bitmap, we're going to set it and create an entry
                    let n = node.bitmap.count_ones() as usize;
                    if n >= HASH_SIZE / 2 {
                        let jdx = mask(hash, shift) as usize;
                        let mut j = 0;
                        let mut added = false;
                        let nodes = (0..HASH_SIZE)
                            .into_iter()
                            .map(|i| {
                                if i == jdx {
                                    let (added_here, new_node) = Node::empty().insert(
                                        hasher,
                                        shift + HASH_BITS,
                                        hash,
                                        key,
                                        value,
                                    );
                                    if added_here {
                                        added = true;
                                    }
                                    return Some(new_node);
                                }
                                if (node.bitmap >> i) & 1 != 0 {
                                    j += 1;
                                    match node.content[j - 1] {
                                        Entry::Node(ref c_node) => Some(c_node.clone()),
                                        Entry::Pair(ref k, ref v) => {
                                            let (added_here, new_node) = Node::empty().insert(
                                                hasher,
                                                shift + HASH_BITS,
                                                hash_key(hasher, k),
                                                k,
                                                v,
                                            );
                                            if added_here {
                                                added = true;
                                            }
                                            Some(new_node)
                                        }
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        (added, ArrayNode::new(n + 1, nodes))
                    } else {
                        let mut new_content = Vec::with_capacity((n + 1) as usize);
                        new_content.extend(node.content.iter().take(idx).cloned());
                        new_content.push(Entry::Pair(key.clone(), value.clone()));
                        new_content.extend(node.content.iter().skip(idx).cloned());
                        (true, BitmapNode::new(node.bitmap | bit, new_content))
                    }
                }
            }

            // Insert for CollisionNode
            Node::CollisionNode(ref node) => {
                if hash == node.hash {
                    match node.find_index(key) {
                        Some(idx) => {
                            if Arc::ptr_eq(value, &node.content[idx].1) {
                                (false, self.clone())
                            } else {
                                (
                                    false,
                                    CollisionNode::new(
                                        node.hash,
                                        node.size,
                                        update(&node.content, idx, (key.clone(), value.clone())),
                                    ),
                                )
                            }
                        }
                        None => {
                            let mut new_content = node.content.clone();
                            new_content.push((key.clone(), value.clone()));
                            (
                                true,
                                CollisionNode::new(node.hash, node.size + 1, new_content),
                            )
                        }
                    }
                } else {
                    let new_node = Node::single(bitpos(node.hash, shift), self);
                    new_node.insert(hasher, shift, hash, key, value)
                }
            }
        }
    }

    pub fn remove(
        &self,
        shift: usize,
        hash: Bitmap,
        key: &K,
    ) -> (Option<(Arc<K>, Arc<V>)>, Option<Self>) {
        match *self {
            // Remove for ArrayNode
            Node::ArrayNode(ref node) => {
                let idx = mask(hash, shift) as usize;
                match node.content[idx] {
                    None => (None, Some(self.clone())),
                    Some(ref child) => match child.remove(shift + HASH_BITS, hash, key) {
                        (ref p, Some(ref result)) if result.ptr_eq(self) => {
                            (p.clone(), Some(self.clone()))
                        }
                        (p, None) => {
                            if node.size <= HASH_SIZE / 4 {
                                (p, Some(node.pack(idx)))
                            } else {
                                (
                                    p,
                                    Some(ArrayNode::new(
                                        node.size - 1,
                                        update(&node.content, idx, None),
                                    )),
                                )
                            }
                        }
                        (p, Some(result)) => (
                            p,
                            Some(ArrayNode::new(
                                node.size,
                                update(&node.content, idx, Some(result)),
                            )),
                        ),
                    },
                }
            }

            // Remove for BitmapNode
            Node::BitmapNode(ref node) => {
                let bit = bitpos(hash, shift);
                if node.bitmap & bit == 0 {
                    return (None, Some(self.clone()));
                }
                let idx = bit_index(node.bitmap, bit);
                match node.content[idx] {
                    Entry::Node(ref child) => match child.remove(shift + HASH_BITS, hash, key) {
                        (ref p, Some(ref result)) if result.ptr_eq(child) => {
                            (p.clone(), Some(self.clone()))
                        }
                        (p, Some(result)) => (
                            p,
                            Some(BitmapNode::new(
                                node.bitmap,
                                update(&node.content, idx, Entry::Node(result)),
                            )),
                        ),
                        (ref p, None) if node.bitmap == bit => (p.clone(), None),
                        (p, None) => (
                            p,
                            Some(BitmapNode::new(
                                node.bitmap ^ bit,
                                remove(&node.content, idx),
                            )),
                        ),
                    },
                    Entry::Pair(ref k, ref v) if key == &**k => (
                        Some((k.clone(), v.clone())),
                        Some(BitmapNode::new(
                            node.bitmap ^ bit,
                            remove(&node.content, idx),
                        )),
                    ),
                    Entry::Pair(_, _) => (None, Some(self.clone())),
                }
            }

            // Remove for CollisionNode
            Node::CollisionNode(ref node) => match node.content
                .iter()
                .enumerate()
                .find(|&(_, &(ref k, _))| &**k == key)
            {
                None => (None, Some(self.clone())),
                Some((i, &(ref k, ref v))) => {
                    if node.size == 1 {
                        (Some((k.clone(), v.clone())), None)
                    } else {
                        let mut new_content = node.content.clone();
                        new_content.remove(i);
                        (
                            Some((k.clone(), v.clone())),
                            Some(CollisionNode::new(node.hash, node.size - 1, new_content)),
                        )
                    }
                }
            },
        }
    }

    pub fn lookup(&self, shift: usize, hash: Bitmap, key: &K) -> Option<Arc<V>> {
        match *self {
            Node::ArrayNode(ref node) => {
                let idx = mask(hash, shift) as usize;
                match node.content[idx] {
                    None => None,
                    Some(ref idx_node) => idx_node.lookup(shift + HASH_BITS, hash, key),
                }
            }
            Node::BitmapNode(ref node) => {
                let bit = bitpos(hash, shift);
                if node.bitmap & bit == 0 {
                    None
                } else {
                    match node.content[bit_index(node.bitmap, bit)] {
                        Entry::Node(ref idx_node) => idx_node.lookup(shift + HASH_BITS, hash, key),
                        Entry::Pair(ref k, ref v) if &**k == key => Some(v.clone()),
                        _ => None,
                    }
                }
            }
            Node::CollisionNode(ref node) => node.content
                .iter()
                .find(|&&(ref k, _)| &**k == key)
                .map(|&(_, ref v)| v.clone()),
        }
    }

    pub fn insert_mut<S>(
        &mut self,
        hasher: &S,
        shift: usize,
        hash: Bitmap,
        key: &Arc<K>,
        value: &Arc<V>,
    ) -> (bool, Option<Self>)
    where
        S: BuildHasher,
    {
        match *self {
            Node::ArrayNode(ref mut node) => {
                let idx = mask(hash, shift) as usize;
                let mut this = Arc::make_mut(node);
                let (update, added, inc) = match this.content[idx] {
                    None => {
                        let (_, new_node) =
                            Node::empty().insert(hasher, shift + HASH_BITS, hash, key, value);
                        (Some(Some(new_node)), true, 1)
                    }
                    Some(ref mut child) => {
                        match child.insert_mut(hasher, shift + HASH_BITS, hash, key, value) {
                            (added_leaf, Some(new_child)) => (Some(Some(new_child)), added_leaf, 0),
                            (added_leaf, None) => (None, added_leaf, 0),
                        }
                    }
                };
                if let Some(new_node) = update {
                    this.content[idx] = new_node;
                }
                if inc != 0 {
                    this.size += inc;
                }
                (added, None)
            }
            Node::BitmapNode(ref mut node) => {
                let bit = bitpos(hash, shift);
                let idx = bit_index(node.bitmap, bit);
                let mut this = Arc::make_mut(node);
                let (added, update, insert, newmap) = if this.bitmap & bit != 0 {
                    match this.content[idx] {
                        Entry::Node(ref mut idx_node) => {
                            match idx_node.insert_mut(hasher, shift + HASH_BITS, hash, key, value) {
                                (added_leaf, Some(new_child)) => {
                                    (added_leaf, Some(Entry::Node(new_child)), None, None)
                                }
                                (added_leaf, None) => (added_leaf, None, None, None),
                            }
                        }
                        Entry::Pair(ref k, ref v) if k == key => {
                            if Arc::ptr_eq(value, v) {
                                (false, None, None, None)
                            } else {
                                (
                                    false,
                                    Some(Entry::Pair(k.clone(), value.clone())),
                                    None,
                                    None,
                                )
                            }
                        }
                        Entry::Pair(ref k, ref v) => {
                            let new_child =
                                create_node(hasher, shift + HASH_BITS, k, v, hash, key, value);
                            (true, Some(Entry::Node(new_child)), None, None)
                        }
                    }
                } else {
                    let n = this.bitmap.count_ones() as usize;
                    if n < this.content.len() {
                        (
                            true,
                            Some(Entry::Pair(key.clone(), value.clone())),
                            None,
                            Some(this.bitmap | bit),
                        )
                    } else if n >= HASH_SIZE / 2 {
                        let jdx = mask(hash, shift) as usize;
                        let mut j = 0;
                        let mut added = false;
                        let nodes = (0..HASH_SIZE)
                            .into_iter()
                            .map(|i| {
                                if i == jdx {
                                    let (added_here, new_node) = Node::empty().insert(
                                        hasher,
                                        shift + HASH_BITS,
                                        hash,
                                        key,
                                        value,
                                    );
                                    if added_here {
                                        added = true;
                                    }
                                    return Some(new_node);
                                }
                                if (this.bitmap >> i) & 1 != 0 {
                                    j += 1;
                                    match this.content[j - 1] {
                                        Entry::Node(ref c_node) => Some(c_node.clone()),
                                        Entry::Pair(ref k, ref v) => {
                                            let (added_here, new_node) = Node::empty().insert(
                                                hasher,
                                                shift + HASH_BITS,
                                                hash_key(hasher, k),
                                                k,
                                                v,
                                            );
                                            if added_here {
                                                added = true;
                                            }
                                            Some(new_node)
                                        }
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        return (added, Some(ArrayNode::new(n + 1, nodes)));
                    } else {
                        (
                            true,
                            None,
                            Some(Entry::Pair(key.clone(), value.clone())),
                            Some(this.bitmap | bit),
                        )
                    }
                };
                if let Some(entry) = update {
                    this.content[idx] = entry;
                }
                if let Some(entry) = insert {
                    this.content.insert(idx, entry);
                }
                if let Some(bm) = newmap {
                    this.bitmap = bm;
                }
                (added, None)
            }
            Node::CollisionNode(ref mut node) => {
                if hash == node.hash {
                    match node.find_index(key) {
                        Some(idx) => {
                            if Arc::ptr_eq(value, &node.content[idx].1) {
                                (false, None)
                            } else {
                                let mut this = Arc::make_mut(node);
                                this.content[idx] = (key.clone(), value.clone());
                                (false, None)
                            }
                        }
                        None => {
                            let mut this = Arc::make_mut(node);
                            this.content.push((key.clone(), value.clone()));
                            this.size += 1;
                            (true, None)
                        }
                    }
                } else {
                    let mut new_node =
                        Node::single(bitpos(node.hash, shift), &Node::CollisionNode(node.clone()));
                    let (_, new_root) = new_node.insert_mut(hasher, shift, hash, key, value);
                    (true, Some(new_root.unwrap_or(new_node)))
                }
            }
        }
    }

    pub fn remove_mut(
        &mut self,
        shift: usize,
        hash: Bitmap,
        key: &K,
    ) -> (Option<(Arc<K>, Arc<V>)>, RemoveResult<Self>) {
        match *self {
            Node::ArrayNode(ref mut node) => {
                let idx = mask(hash, shift) as usize;
                let mut this = Arc::make_mut(node);
                let (removed, updated) = match this.content[idx] {
                    None => return (None, RemoveResult::Unchanged),
                    Some(ref mut child) => child.remove_mut(shift + HASH_BITS, hash, key),
                };
                match updated {
                    RemoveResult::NewNode(node) => this.content[idx] = Some(node),
                    RemoveResult::Removed => {
                        if this.size <= HASH_SIZE / 4 {
                            return (removed, RemoveResult::NewNode(this.pack(idx)));
                        }
                        this.content[idx] = None;
                        this.size -= 1;
                    }
                    RemoveResult::Unchanged => (),
                }
                (removed, RemoveResult::Unchanged)
            }
            Node::BitmapNode(ref mut node) => {
                let mut bit = bitpos(hash, shift);
                if node.bitmap & bit == 0 {
                    return (None, RemoveResult::Unchanged);
                }
                let idx = bit_index(node.bitmap, bit);
                let mut this = Arc::make_mut(node);
                let (removed, updated) = match this.content[idx] {
                    Entry::Node(ref mut child) => child.remove_mut(shift + HASH_BITS, hash, key),
                    Entry::Pair(ref k, ref v) if &**k == key => {
                        (Some((k.clone(), v.clone())), RemoveResult::Removed)
                    }
                    Entry::Pair(_, _) => return (None, RemoveResult::Unchanged),
                };
                match updated {
                    RemoveResult::NewNode(node) => this.content[idx] = Entry::Node(node),
                    RemoveResult::Removed => {
                        this.bitmap ^= bit;
                        this.content.remove(idx);
                    }
                    RemoveResult::Unchanged => (),
                }
                (
                    removed,
                    if this.bitmap == 0 {
                        RemoveResult::Removed
                    } else {
                        RemoveResult::Unchanged
                    },
                )
            }
            Node::CollisionNode(ref mut node) => {
                if node.hash != hash {
                    return (None, RemoveResult::Unchanged);
                }
                match node.find_index(key) {
                    Some(idx) => {
                        let mut this = Arc::make_mut(node);
                        this.size -= 1;
                        let (k, v) = this.content.remove(idx);
                        (
                            Some((k, v)),
                            if this.size == 0 {
                                RemoveResult::Removed
                            } else {
                                RemoveResult::Unchanged
                            },
                        )
                    }
                    None => (None, RemoveResult::Unchanged),
                }
            }
        }
    }
}

impl<K, V> Node<K, V> {
    pub fn iter(&self, length: usize) -> Iter<K, V> {
        Iter {
            queue: Vec::new(),
            current: self.clone(),
            index: 0,
            remaining: length,
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::eq(self, other) || match (self, other) {
            (&Node::ArrayNode(ref l), &Node::ArrayNode(ref r)) => Arc::ptr_eq(l, r),
            (&Node::BitmapNode(ref l), &Node::BitmapNode(ref r)) => Arc::ptr_eq(l, r),
            (&Node::CollisionNode(ref l), &Node::CollisionNode(ref r)) => Arc::ptr_eq(l, r),
            _ => false,
        }
    }
}

// Utilities

fn create_node<K: Hash + Eq, V, S: BuildHasher>(
    hasher: &S,
    shift: usize,
    key1: &Arc<K>,
    value1: &Arc<V>,
    hash2: Bitmap,
    key2: &Arc<K>,
    value2: &Arc<V>,
) -> Node<K, V> {
    let hash1 = hash_key(hasher, key1);
    if hash1 == hash2 {
        CollisionNode::new(
            hash1,
            2,
            vec![
                (key1.clone(), value1.clone()),
                (key2.clone(), value2.clone()),
            ],
        )
    } else {
        let n0 = Node::empty();
        let (_, n1) = n0.insert(hasher, shift, hash1, key1, value1);
        let (_, n2) = n1.insert(hasher, shift, hash2, key2, value2);
        n2
    }
}

fn update<A: Clone>(input: &[A], index: usize, value: A) -> Vec<A> {
    let mut output = input.to_owned();
    output[index] = value;
    output
}

fn remove<A: Clone>(input: &[A], index: usize) -> Vec<A> {
    input
        .iter()
        .take(index)
        .chain(input.iter().skip(index + 1))
        .cloned()
        .collect()
}

// Iterator

pub struct Iter<K, V> {
    queue: Vec<(Node<K, V>, usize)>,
    current: Node<K, V>,
    index: usize,
    remaining: usize,
}

impl<K, V> Iter<K, V> {
    fn push(&mut self, next: &Node<K, V>) {
        self.queue.push((self.current.clone(), self.index));
        self.current = next.clone();
        self.index = 0;
    }

    fn pop(&mut self) -> bool {
        if let Some((node, index)) = self.queue.pop() {
            self.current = node;
            self.index = index;
            true
        } else {
            false
        }
    }

    fn advance(&mut self) -> Option<Entry<K, V>> {
        match self.current {
            Node::ArrayNode(ref node) => loop {
                if self.index >= node.content.len() {
                    return None;
                }
                let item = &node.content[self.index];
                self.index += 1;
                match *item {
                    Some(ref node) => return Some(Entry::Node(node.clone())),
                    None => continue,
                }
            },
            Node::BitmapNode(ref node) => {
                let item = node.content.get(self.index).cloned();
                self.index += 1;
                item
            }
            Node::CollisionNode(ref node) => {
                let item = node.content
                    .get(self.index)
                    .map(|&(ref k, ref v)| Entry::Pair(k.clone(), v.clone()));
                self.index += 1;
                item
            }
        }
    }
}

impl<K, V> Iterator for Iter<K, V> {
    type Item = (Arc<K>, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.advance() {
                None => {
                    if self.pop() {
                        continue;
                    } else {
                        return None;
                    }
                }
                Some(Entry::Pair(ref k, ref v)) => {
                    self.remaining -= 1;
                    return Some((k.clone(), v.clone()));
                }
                Some(Entry::Node(ref node)) => {
                    self.push(node);
                    continue;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<K, V> {}

// Debug

impl<K, V> Debug for Node<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match *self {
            Node::ArrayNode(ref node) => node.fmt_in(f, 0),
            Node::BitmapNode(ref node) => node.fmt_in(f, 0),
            Node::CollisionNode(ref node) => node.fmt_in(f, 0),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt_in(&self, f: &mut Formatter, indent: usize) -> Result<(), Error> {
        match *self {
            Node::ArrayNode(ref node) => node.fmt_in(f, indent),
            Node::BitmapNode(ref node) => node.fmt_in(f, indent),
            Node::CollisionNode(ref node) => node.fmt_in(f, indent),
        }
    }
}

impl<K, V> ArrayNode<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt_in(&self, f: &mut Formatter, indent: usize) -> Result<(), Error> {
        write!(
            f,
            "{0:1$}ArrayNode size: {2} content: [\n",
            "", indent, self.size
        )?;
        for node in &self.content {
            match *node {
                None => write!(f, "{0:1$}empty\n", "", indent + 2)?,
                Some(ref node) => node.fmt_in(f, indent + 2)?,
            }
        }
        write!(f, "{0:1$}]\n", "", indent)
    }
}

impl<K, V> BitmapNode<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt_in(&self, f: &mut Formatter, indent: usize) -> Result<(), Error> {
        write!(
            f,
            "{0:1$}BitmapNode bitmap: {2:b} content: [\n",
            "", indent, self.bitmap
        )?;
        for entry in &self.content {
            match *entry {
                Entry::Node(ref node) => node.fmt_in(f, indent + 2)?,
                Entry::Pair(ref k, ref v) => {
                    write!(f, "{0:1$}{{ {2:?} => {3:?} }}\n", "", indent + 2, k, v)?
                }
            }
        }
        write!(f, "{0:1$}]\n", "", indent)
    }
}

impl<K, V> CollisionNode<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt_in(&self, f: &mut Formatter, indent: usize) -> Result<(), Error> {
        write!(
            f,
            "{0:1$}CollisionNode size: {2} hash: {3} content: [\n",
            "", indent, self.size, self.hash
        )?;
        for &(ref k, ref v) in &self.content {
            write!(f, "{0:1$}{{ {2:?} => {3:?} }}\n", "", indent + 2, k, v)?
        }
        write!(f, "{0:1$}]\n", "", indent)
    }
}
