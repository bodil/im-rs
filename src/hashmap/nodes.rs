use std::sync::Arc;
use std::hash::{BuildHasher, Hash};
use std::fmt::{Debug, Error, Formatter};

use super::bits::{bit_index, bitpos, mask, Bitmap, HASH_BITS, HASH_SIZE};
use super::hash::hash_key;

enum Entry<K, V> {
    Pair(Arc<K>, Arc<V>),
    Node(Node<K, V>),
}

impl<K, V> Clone for Entry<K, V> {
    fn clone(&self) -> Self {
        match self {
            &Entry::Pair(ref k, ref v) => Entry::Pair(k.clone(), v.clone()),
            &Entry::Node(ref node) => Entry::Node(node.clone()),
        }
    }
}

pub enum Node<K, V> {
    ArrayNode(Arc<ArrayNode<K, V>>),
    BitmapNode(Arc<BitmapNode<K, V>>),
    CollisionNode(Arc<CollisionNode<K, V>>),
}

impl<K, V> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        match self {
            &Node::ArrayNode(ref node) => Node::ArrayNode(node.clone()),
            &Node::BitmapNode(ref node) => Node::BitmapNode(node.clone()),
            &Node::CollisionNode(ref node) => Node::CollisionNode(node.clone()),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Hash + Eq,
{
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
        match self {
            &Node::ArrayNode(ref node) => node.insert(self, hasher, shift, hash, key, value),
            &Node::BitmapNode(ref node) => node.insert(self, hasher, shift, hash, key, value),
            &Node::CollisionNode(ref node) => node.insert(self, hasher, shift, hash, key, value),
        }
    }

    pub fn remove(
        &self,
        shift: usize,
        hash: Bitmap,
        key: &K,
    ) -> (Option<(Arc<K>, Arc<V>)>, Option<Self>) {
        match self {
            &Node::ArrayNode(ref node) => node.remove(self, shift, hash, key),
            &Node::BitmapNode(ref node) => node.remove(self, shift, hash, key),
            &Node::CollisionNode(ref node) => node.remove(self, key),
        }
    }

    pub fn lookup(&self, shift: usize, hash: Bitmap, key: &K) -> Option<Arc<V>> {
        match self {
            &Node::ArrayNode(ref node) => node.lookup(shift, hash, key),
            &Node::BitmapNode(ref node) => node.lookup(shift, hash, key),
            &Node::CollisionNode(ref node) => node.lookup(key),
        }
    }
}

impl<K, V> Node<K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            queue: Vec::new(),
            current: self.clone(),
            index: 0,
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&Node::ArrayNode(ref l), &Node::ArrayNode(ref r)) => Arc::ptr_eq(l, r),
            (&Node::BitmapNode(ref l), &Node::BitmapNode(ref r)) => Arc::ptr_eq(l, r),
            (&Node::CollisionNode(ref l), &Node::CollisionNode(ref r)) => Arc::ptr_eq(l, r),
            _ => false,
        }
    }
}

pub struct ArrayNode<K, V> {
    size: usize,
    content: Vec<Option<Node<K, V>>>,
}

impl<K, V> ArrayNode<K, V>
where
    K: Hash + Eq,
{
    fn new(size: usize, content: Vec<Option<Node<K, V>>>) -> Node<K, V> {
        Node::ArrayNode(Arc::new(ArrayNode { size, content }))
    }

    fn insert<S>(
        &self,
        this: &Node<K, V>,
        hasher: &S,
        shift: usize,
        hash: Bitmap,
        key: &Arc<K>,
        value: &Arc<V>,
    ) -> (bool, Node<K, V>)
    where
        S: BuildHasher,
    {
        let idx = mask(hash, shift) as usize;
        match self.content[idx] {
            None => {
                let nil = empty();
                let (added, new_node) = nil.insert(hasher, shift + HASH_BITS, hash, key, value);
                (
                    added,
                    ArrayNode::new(self.size + 1, update(&self.content, idx, Some(new_node))),
                )
            }
            Some(ref node) => {
                let (added, new_node) = node.insert(hasher, shift + HASH_BITS, hash, key, value);
                if node.ptr_eq(&new_node) {
                    (added, this.clone())
                } else {
                    (
                        added,
                        ArrayNode::new(self.size, update(&self.content, idx, Some(new_node))),
                    )
                }
            }
        }
    }

    pub fn remove(
        &self,
        node: &Node<K, V>,
        shift: usize,
        hash: Bitmap,
        key: &K,
    ) -> (Option<(Arc<K>, Arc<V>)>, Option<Node<K, V>>) {
        let idx = mask(hash, shift) as usize;
        match self.content[idx] {
            None => (None, Some(node.clone())),
            Some(ref child) => match child.remove(shift + HASH_BITS, hash, key) {
                (ref p, Some(ref result)) if result.ptr_eq(node) => (p.clone(), Some(node.clone())),
                (p, None) => {
                    if self.size <= HASH_SIZE / 4 {
                        (p, Some(self.pack(idx)))
                    } else {
                        (
                            p,
                            Some(ArrayNode::new(
                                self.size - 1,
                                update(&self.content, idx, None),
                            )),
                        )
                    }
                }
                (p, Some(result)) => (
                    p,
                    Some(ArrayNode::new(
                        self.size,
                        update(&self.content, idx, Some(result)),
                    )),
                ),
            },
        }
    }

    pub fn lookup(&self, shift: usize, hash: Bitmap, key: &K) -> Option<Arc<V>> {
        let idx = mask(hash, shift) as usize;
        match self.content[idx] {
            None => None,
            Some(ref node) => node.lookup(shift + HASH_BITS, hash, key),
        }
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

pub struct BitmapNode<K, V> {
    bitmap: Bitmap,
    content: Vec<Entry<K, V>>,
}

impl<K, V> BitmapNode<K, V>
where
    K: Hash + Eq,
{
    fn new(bitmap: Bitmap, content: Vec<Entry<K, V>>) -> Node<K, V> {
        Node::BitmapNode(Arc::new(BitmapNode { bitmap, content }))
    }

    fn insert<S>(
        &self,
        this: &Node<K, V>,
        hasher: &S,
        shift: usize,
        hash: Bitmap,
        key: &Arc<K>,
        value: &Arc<V>,
    ) -> (bool, Node<K, V>)
    where
        S: BuildHasher,
    {
        let bit = bitpos(hash, shift);
        let idx = bit_index(self.bitmap, bit);
        if self.bitmap & bit != 0 {
            match self.content[idx] {
                Entry::Node(ref node) => {
                    let (added, new_node) =
                        node.insert(hasher, shift + HASH_BITS, hash, key, value);
                    if node.ptr_eq(&new_node) {
                        (added, this.clone())
                    } else {
                        (
                            added,
                            BitmapNode::new(
                                self.bitmap,
                                update(&self.content, idx, Entry::Node(new_node)),
                            ),
                        )
                    }
                }
                Entry::Pair(ref old_key, ref old_value) if old_key == key => {
                    if Arc::ptr_eq(value, old_value) {
                        (false, this.clone())
                    } else {
                        (
                            false,
                            BitmapNode::new(
                                self.bitmap,
                                update(&self.content, idx, Entry::Pair(key.clone(), value.clone())),
                            ),
                        )
                    }
                }
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
                            self.bitmap,
                            update(&self.content, idx, Entry::Node(new_node)),
                        ),
                    )
                }
            }
        } else {
            let n = self.bitmap.count_ones() as usize;
            if n >= HASH_SIZE / 2 {
                let jdx = mask(hash, shift) as usize;
                let mut j = 0;
                let mut added = false;
                let nodes = (0..HASH_SIZE)
                    .into_iter()
                    .map(|i| {
                        if i == jdx {
                            let (added_here, new_node) =
                                empty().insert(hasher, shift + HASH_BITS, hash, key, value);
                            if added_here {
                                added = true;
                            }
                            return Some(new_node);
                        }
                        if (self.bitmap >> i) & 1 != 0 {
                            j += 1;
                            match self.content[j - 1] {
                                Entry::Node(ref node) => Some(node.clone()),
                                Entry::Pair(ref k, ref v) => {
                                    let (added_here, new_node) = empty().insert(
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
                new_content.extend(self.content.iter().take(idx).cloned());
                new_content.push(Entry::Pair(key.clone(), value.clone()));
                new_content.extend(self.content.iter().skip(idx).cloned());
                (true, BitmapNode::new(self.bitmap | bit, new_content))
            }
        }
    }

    pub fn remove(
        &self,
        node: &Node<K, V>,
        shift: usize,
        hash: Bitmap,
        key: &K,
    ) -> (Option<(Arc<K>, Arc<V>)>, Option<Node<K, V>>) {
        let bit = bitpos(hash, shift);
        if self.bitmap & bit == 0 {
            return (None, Some(node.clone()));
        }
        let idx = bit_index(self.bitmap, bit);
        match self.content[idx] {
            Entry::Node(ref child) => match child.remove(shift + 5, hash, key) {
                (ref p, Some(ref result)) if result.ptr_eq(child) => {
                    (p.clone(), Some(node.clone()))
                }
                (p, Some(result)) => (
                    p,
                    Some(BitmapNode::new(
                        self.bitmap,
                        update(&self.content, idx, Entry::Node(result)),
                    )),
                ),
                (ref p, None) if self.bitmap == bit => (p.clone(), None),
                (p, None) => (
                    p,
                    Some(BitmapNode::new(
                        self.bitmap ^ bit,
                        remove(&self.content, idx),
                    )),
                ),
            },
            Entry::Pair(ref k, ref v) if key == &**k => (
                Some((k.clone(), v.clone())),
                Some(BitmapNode::new(
                    self.bitmap ^ bit,
                    remove(&self.content, idx),
                )),
            ),
            Entry::Pair(_, _) => (None, Some(node.clone())),
        }
    }

    pub fn lookup(&self, shift: usize, hash: Bitmap, key: &K) -> Option<Arc<V>> {
        let bit = bitpos(hash, shift);
        if self.bitmap & bit == 0 {
            None
        } else {
            match self.content[bit_index(self.bitmap, bit)] {
                Entry::Node(ref node) => node.lookup(shift + HASH_BITS, hash, key),
                Entry::Pair(ref k, ref v) if &**k == key => Some(v.clone()),
                _ => None,
            }
        }
    }
}

pub struct CollisionNode<K, V> {
    hash: Bitmap,
    size: usize,
    content: Vec<(Arc<K>, Arc<V>)>,
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

    fn insert<S>(
        &self,
        this: &Node<K, V>,
        hasher: &S,
        shift: usize,
        hash: Bitmap,
        key: &Arc<K>,
        value: &Arc<V>,
    ) -> (bool, Node<K, V>)
    where
        S: BuildHasher,
    {
        if hash == self.hash {
            match self.find_index(key) {
                Some(idx) => {
                    if Arc::ptr_eq(value, &self.content[idx].1) {
                        (false, this.clone())
                    } else {
                        (
                            false,
                            CollisionNode::new(
                                self.hash,
                                self.size,
                                update(&self.content, idx, (key.clone(), value.clone())),
                            ),
                        )
                    }
                }
                None => {
                    let mut new_content = self.content.clone();
                    new_content.push((key.clone(), value.clone()));
                    (
                        true,
                        CollisionNode::new(self.hash, self.size + 1, new_content),
                    )
                }
            }
        } else {
            let new_node = single(bitpos(self.hash, shift), this);
            new_node.insert(hasher, shift, hash, key, value)
        }
    }

    pub fn remove(
        &self,
        node: &Node<K, V>,
        key: &K,
    ) -> (Option<(Arc<K>, Arc<V>)>, Option<Node<K, V>>) {
        match self.content
            .iter()
            .enumerate()
            .find(|&(_, &(ref k, _))| &**k == key)
        {
            None => (None, Some(node.clone())),
            Some((i, &(ref k, ref v))) => {
                if self.size == 1 {
                    (Some((k.clone(), v.clone())), None)
                } else {
                    let mut new_content = self.content.clone();
                    new_content.remove(i);
                    (
                        Some((k.clone(), v.clone())),
                        Some(CollisionNode::new(self.hash, self.size - 1, new_content)),
                    )
                }
            }
        }
    }

    pub fn lookup(&self, key: &K) -> Option<Arc<V>> {
        self.content
            .iter()
            .find(|&&(ref k, _)| &**k == key)
            .map(|&(_, ref v)| v.clone())
    }

    fn find_index(&self, key: &K) -> Option<usize> {
        self.content.iter().position(|&(ref k, _)| &**k == key)
    }
}

pub fn single<K: Hash + Eq, V>(bitmap: Bitmap, node: &Node<K, V>) -> Node<K, V> {
    BitmapNode::new(bitmap, vec![Entry::Node(node.clone())])
}

pub fn empty<K: Hash + Eq, V>() -> Node<K, V> {
    BitmapNode::new(0, Vec::new())
}

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
        let n0 = empty();
        let (_, n1) = n0.insert(hasher, shift, hash1, key1, value1);
        let (_, n2) = n1.insert(hasher, shift, hash2, key2, value2);
        n2
    }
}

fn update<A: Clone>(input: &Vec<A>, index: usize, value: A) -> Vec<A> {
    let mut output = input.clone();
    output[index] = value;
    output
}

fn remove<A: Clone>(input: &Vec<A>, index: usize) -> Vec<A> {
    input
        .iter()
        .take(index)
        .chain(input.iter().skip(index + 1))
        .cloned()
        .collect()
}

pub struct Iter<K, V> {
    queue: Vec<(Node<K, V>, usize)>,
    current: Node<K, V>,
    index: usize,
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
                match item {
                    &Some(ref node) => return Some(Entry::Node(node.clone())),
                    &None => continue,
                }
            },
            Node::BitmapNode(ref node) => {
                let item = node.content.get(self.index).map(Clone::clone);
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
                Some(Entry::Pair(ref k, ref v)) => return Some((k.clone(), v.clone())),
                Some(Entry::Node(ref node)) => {
                    self.push(node);
                    continue;
                }
            }
        }
    }
}

// Debug

impl<K, V> Debug for Node<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            &Node::ArrayNode(ref node) => node.fmt_in(f, 0),
            &Node::BitmapNode(ref node) => node.fmt_in(f, 0),
            &Node::CollisionNode(ref node) => node.fmt_in(f, 0),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Hash + Eq + Debug,
    V: Debug,
{
    fn fmt_in(&self, f: &mut Formatter, indent: usize) -> Result<(), Error> {
        match self {
            &Node::ArrayNode(ref node) => node.fmt_in(f, indent),
            &Node::BitmapNode(ref node) => node.fmt_in(f, indent),
            &Node::CollisionNode(ref node) => node.fmt_in(f, indent),
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
        for node in self.content.iter() {
            match node {
                &None => write!(f, "{0:1$}empty\n", "", indent + 2)?,
                &Some(ref node) => node.fmt_in(f, indent + 2)?,
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
        for entry in self.content.iter() {
            match entry {
                &Entry::Node(ref node) => node.fmt_in(f, indent + 2)?,
                &Entry::Pair(ref k, ref v) => {
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
        for &(ref k, ref v) in self.content.iter() {
            write!(f, "{0:1$}{{ {2:?} => {3:?} }}\n", "", indent + 2, k, v)?
        }
        write!(f, "{0:1$}]\n", "", indent)
    }
}
