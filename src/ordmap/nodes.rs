use std::sync::Arc;
use std::fmt::{self, Debug};
use std::ops::IndexMut;

use self::Insert::*;
use self::InsertAction::*;

const NODE_SIZE: usize = 16; // Must be an even number!
const MEDIAN: usize = (NODE_SIZE + 1) >> 1;

pub struct Node<K, V>(Arc<NodeData<K, V>>);

#[derive(PartialEq, Eq)]
pub struct Pair<K, V> {
    pub key: Arc<K>,
    pub value: Arc<V>,
}

struct NodeData<K, V> {
    count: usize,
    keys: Vec<Pair<K, V>>,
    children: Vec<Option<Node<K, V>>>,
}

pub enum Insert<K, V> {
    NoChange,
    JustInc,
    Update(Node<K, V>),
    Split(Node<K, V>, Pair<K, V>, Node<K, V>),
}

enum InsertAction<K, V> {
    NoAction,
    IncAction,
    InsertAt,
    InsertSplit(Node<K, V>, Pair<K, V>, Node<K, V>),
}

pub enum Remove<K, V> {
    NoChange,
    Removed(Pair<K, V>),
    Update(Pair<K, V>, Node<K, V>),
}

enum RemoveAction<K> {
    DeleteAt(usize),
    PullUp(Arc<K>, usize, usize),
    Merge(usize),
    StealFromLeft(usize),
    StealFromRight(usize),
    MergeFirst(usize),
    ContinueDown(usize),
}

impl<K, V> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        Node(self.0.clone())
    }
}

impl<K, V> Clone for Pair<K, V> {
    fn clone(&self) -> Self {
        Pair {
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
}

impl<K, V> Clone for NodeData<K, V> {
    fn clone(&self) -> Self {
        NodeData {
            count: self.count,
            keys: self.keys.clone(),
            children: self.children.clone(),
        }
    }
}

impl<K, V> NodeData<K, V> {
    #[inline]
    fn has_room(&self) -> bool {
        self.keys.len() < NODE_SIZE
    }

    #[inline]
    fn too_small(&self) -> bool {
        self.keys.len() < MEDIAN
    }

    fn sum_up_children(&self) -> usize {
        let mut c = self.count;
        for n in &self.children {
            match *n {
                None => continue,
                Some(ref node) => c += node.len(),
            }
        }
        c
    }
}

impl<K, V> Node<K, V> {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.count
    }

    #[inline]
    fn maybe_len(node_or: &Option<Node<K, V>>) -> usize {
        match *node_or {
            None => 0,
            Some(ref node) => node.len(),
        }
    }

    #[inline]
    fn has_room(&self) -> bool {
        self.0.has_room()
    }

    #[inline]
    fn too_small(&self) -> bool {
        self.0.too_small()
    }

    #[inline]
    pub fn new() -> Self {
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(None);
        Node(Arc::new(NodeData {
            count: 0,
            keys: Vec::with_capacity(NODE_SIZE),
            children,
        }))
    }

    #[inline]
    pub fn singleton(key: Arc<K>, value: Arc<V>) -> Self {
        let mut keys = Vec::with_capacity(NODE_SIZE);
        keys.push(Pair { key, value });
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(None);
        children.push(None);
        Node(Arc::new(NodeData {
            count: 1,
            keys,
            children,
        }))
    }

    #[inline]
    pub fn from_split(left: Node<K, V>, median: Pair<K, V>, right: Node<K, V>) -> Self {
        let count = left.len() + right.len() + 1;
        let mut keys = Vec::with_capacity(NODE_SIZE);
        keys.push(median);
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(Some(left));
        children.push(Some(right));
        Node(Arc::new(NodeData {
            count,
            keys,
            children,
        }))
    }

    #[inline]
    fn wrap(data: NodeData<K, V>) -> Self {
        Node(Arc::new(data))
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn min(&self) -> Option<&Pair<K, V>> {
        match *self.0.children.first().unwrap() {
            None => self.0.keys.first(),
            Some(ref child) => child.min(),
        }
    }

    pub fn max(&self) -> Option<&Pair<K, V>> {
        match *self.0.children.last().unwrap() {
            None => self.0.keys.last(),
            Some(ref child) => child.max(),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Ord,
{
    pub fn lookup(&self, key: &K) -> Option<Arc<V>> {
        if self.0.keys.is_empty() {
            return None;
        }
        // Start by checking if the key is greater than the node's max,
        // and search the rightmost child if so.
        if key > &*self.0.keys[self.0.keys.len() - 1].key {
            match self.0.children[self.0.keys.len()] {
                None => return None,
                Some(ref node) => return node.lookup(key),
            }
        }
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match self.0.keys.binary_search_by_key(&key, |pair| &*pair.key) {
            Ok(index) => Some(self.0.keys[index].value.clone()),
            Err(index) => match self.0.children[index] {
                None => None,
                Some(ref node) => node.lookup(key),
            },
        }
    }

    fn split(
        &self,
        pair: Pair<K, V>,
        ins_left: Option<Node<K, V>>,
        ins_right: Option<Node<K, V>>,
    ) -> Insert<K, V> {
        let mut new_keys = self.0.keys.clone();
        let mut new_children = self.0.children.clone();
        match new_keys.binary_search_by_key(&&*pair.key, |pair| &*pair.key) {
            Ok(_) => unreachable!(),
            Err(index) => {
                new_children[index] = ins_left;
                new_keys.insert(index, pair);
                new_children.insert(index + 1, ins_right);
            }
        }
        let mut left = NodeData {
            count: MEDIAN,
            keys: new_keys.drain(0..MEDIAN).collect(),
            children: new_children.drain(0..MEDIAN + 1).collect(),
        };
        let mut right = NodeData {
            count: MEDIAN,
            keys: new_keys.drain(1..).collect(),
            children: new_children,
        };
        left.count = left.sum_up_children();
        right.count = right.sum_up_children();
        Split(Node::wrap(left), new_keys.pop().unwrap(), Node::wrap(right))
    }

    pub fn insert(&self, key: Arc<K>, value: Arc<V>) -> Insert<K, V> {
        if self.0.keys.is_empty() {
            return Insert::Update(Node::singleton(key, value));
        }
        match self.0.keys.binary_search_by_key(&&*key, |pair| &*pair.key) {
            // Key exists in node
            Ok(index) => {
                if Arc::ptr_eq(&value, &self.0.keys[index].value) {
                    Insert::NoChange
                } else {
                    let mut new_data = (&*self.0).clone();
                    new_data.keys[index] = Pair { key, value };
                    Insert::Update(Node::wrap(new_data))
                }
            }
            // Key is adjacent to some key in node
            Err(index) => match self.0.children[index] {
                // No child at location, this is the target node.
                None => {
                    if self.has_room() {
                        let mut new_data = (&*self.0).clone();
                        new_data.keys.insert(index, Pair { key, value });
                        new_data.children.insert(index + 1, None);
                        new_data.count += 1;
                        Insert::Update(Node::wrap(new_data))
                    } else {
                        self.split(Pair { key, value }, None, None)
                    }
                }
                // Child at location, pass it on.
                Some(ref node) => match node.insert(key, value) {
                    Insert::NoChange => Insert::NoChange,
                    Insert::JustInc => unreachable!(),
                    Insert::Update(new_node) => {
                        // We have an updated child; record it.
                        let mut new_data = (&*self.0).clone();
                        new_data.children[index] = Some(new_node);
                        new_data.count += 1;
                        Insert::Update(Node::wrap(new_data))
                    }
                    Insert::Split(left, median, right) => {
                        // Child node split; insert it.
                        if self.has_room() {
                            let mut new_data = (&*self.0).clone();
                            new_data.children[index] = Some(left);
                            new_data.keys.insert(index, median);
                            new_data.children.insert(index + 1, Some(right));
                            new_data.count += 1;
                            Insert::Update(Node::wrap(new_data))
                        } else {
                            self.split(median, Some(left), Some(right))
                        }
                    }
                },
            },
        }
    }

    fn merge(pair: Pair<K, V>, left: &Node<K, V>, right: &Node<K, V>) -> Node<K, V> {
        let mut keys = Vec::with_capacity(NODE_SIZE);
        keys.extend(left.0.keys.iter().cloned());
        keys.push(pair);
        keys.extend(right.0.keys.iter().cloned());
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.extend(left.0.children.iter().cloned());
        children.extend(right.0.children.iter().cloned());
        Node::wrap(NodeData {
            count: left.len() + right.len() + 1,
            keys,
            children,
        })
    }

    fn pop_min(&self) -> (Node<K, V>, Pair<K, V>, Option<Node<K, V>>) {
        let mut new_data = (&*self.0).clone();
        let pair = new_data.keys.remove(0);
        let child = new_data.children.remove(0);
        new_data.count -= 1 + Node::maybe_len(&child);
        (Node::wrap(new_data), pair, child)
    }

    fn pop_min_mut(&mut self) -> (Pair<K, V>, Option<Node<K, V>>) {
        let node = Arc::make_mut(&mut self.0);
        let pair = node.keys.remove(0);
        let child = node.children.remove(0);
        node.count -= 1 + Node::maybe_len(&child);
        (pair, child)
    }

    fn pop_max(&self) -> (Node<K, V>, Pair<K, V>, Option<Node<K, V>>) {
        let mut new_data = (&*self.0).clone();
        let pair = new_data.keys.pop().unwrap();
        let child = new_data.children.pop().unwrap();
        new_data.count -= 1 + Node::maybe_len(&child);
        (Node::wrap(new_data), pair, child)
    }

    fn pop_max_mut(&mut self) -> (Pair<K, V>, Option<Node<K, V>>) {
        let node = Arc::make_mut(&mut self.0);
        let pair = node.keys.pop().unwrap();
        let child = node.children.pop().unwrap();
        node.count -= 1 + Node::maybe_len(&child);
        (pair, child)
    }

    fn push_min(&self, child: Option<Node<K, V>>, pair: Pair<K, V>) -> Node<K, V> {
        let mut new_data = (&*self.0).clone();
        new_data.count += 1 + Node::maybe_len(&child);
        new_data.keys.insert(0, pair);
        new_data.children.insert(0, child);
        Node::wrap(new_data)
    }

    fn push_min_mut(&mut self, child: Option<Node<K, V>>, pair: Pair<K, V>) {
        let node = Arc::make_mut(&mut self.0);
        node.count += 1 + Node::maybe_len(&child);
        node.keys.insert(0, pair);
        node.children.insert(0, child);
    }

    fn push_max(&self, child: Option<Node<K, V>>, pair: Pair<K, V>) -> Node<K, V> {
        let mut new_data = (&*self.0).clone();
        new_data.count += 1 + Node::maybe_len(&child);
        new_data.keys.push(pair);
        new_data.children.push(child);
        Node::wrap(new_data)
    }

    fn push_max_mut(&mut self, child: Option<Node<K, V>>, pair: Pair<K, V>) {
        let node = Arc::make_mut(&mut self.0);
        node.count += 1 + Node::maybe_len(&child);
        node.keys.push(pair);
        node.children.push(child);
    }

    fn pull_up(
        &self,
        key: &K,
        from_child: &Node<K, V>,
        pull_to: usize,
        child_index: usize,
    ) -> Remove<K, V> {
        match from_child.remove(key) {
            Remove::NoChange => unreachable!(),
            Remove::Removed(_) => unreachable!(),
            Remove::Update(pulled_pair, new_child) => {
                let mut new_data = (&*self.0).clone();
                new_data.keys.push(pulled_pair);
                let pair = new_data.keys.swap_remove(pull_to);
                new_data.children[child_index] = Some(new_child);
                new_data.count -= 1;
                Remove::Update(pair, Node::wrap(new_data))
            }
        }
    }

    pub fn remove(&self, key: &K) -> Remove<K, V> {
        match self.0.keys.binary_search_by_key(&&*key, |pair| &*pair.key) {
            // Key exists in node, remove it.
            Ok(index) => {
                match (&self.0.children[index], &self.0.children[index + 1]) {
                    // If we're a leaf, just delete the entry.
                    (&None, &None) => {
                        let mut new_data = (&*self.0).clone();
                        let pair = new_data.keys.remove(index);
                        new_data.children.remove(index);
                        new_data.count -= 1;
                        Remove::Update(pair, Node::wrap(new_data))
                    }
                    // If the left hand child has capacity, pull the predecessor up.
                    (&Some(ref left), _) if !left.too_small() => {
                        self.pull_up(&left.max().unwrap().key, left, index, index)
                    }
                    // If the right hand child has capacity, pull the successor up.
                    (_, &Some(ref right)) if !right.too_small() => {
                        self.pull_up(&right.min().unwrap().key, right, index, index + 1)
                    }
                    // If neither child has capacity, we'll have to merge them.
                    (&Some(ref left), &Some(ref right)) => {
                        let mut new_data = (&*self.0).clone();
                        let pair = new_data.keys.remove(index);
                        let merged_child = Node::merge(pair.clone(), left, right);
                        let new_child = match merged_child.remove(key) {
                            Remove::NoChange => merged_child,
                            Remove::Removed(_) => unreachable!(),
                            Remove::Update(_, updated_child) => updated_child,
                        };
                        if new_data.keys.is_empty() {
                            // If we've depleted the root node, the merged child becomes the root.
                            Remove::Update(pair, new_child)
                        } else {
                            new_data.count -= 1;
                            new_data.children.remove(index + 1);
                            new_data.children[index] = Some(new_child);
                            Remove::Update(pair, Node::wrap(new_data))
                        }
                    }
                    // If one child exists and the other doesn't, we're in a bad state.
                    _ => unreachable!(),
                }
            }
            // Key is adjacent to some key in node
            Err(index) => match self.0.children[index] {
                // No child at location means key isn't in map.
                None => Remove::NoChange,
                // Child at location, but it's at minimum capacity.
                Some(ref child) if child.too_small() => {
                    let has_left = index > 0;
                    let has_right = index < self.0.children.len() - 1;
                    // If it has a left sibling with capacity, steal a key from it.
                    if has_left {
                        match self.0.children[index - 1] {
                            Some(ref old_left) if !old_left.too_small() => {
                                // Prepare the rebalanced node.
                                let right = child.push_min(
                                    old_left.0.children.last().unwrap().clone(),
                                    self.0.keys[index - 1].clone(),
                                );
                                match right.remove(key) {
                                    Remove::NoChange => return Remove::NoChange,
                                    Remove::Removed(_) => unreachable!(),
                                    Remove::Update(pair, new_child) => {
                                        // If we did remove something, we complete the rebalancing.
                                        let mut new_data = (&*self.0).clone();
                                        let (left, left_pair, _) = old_left.pop_max();
                                        new_data.keys[index - 1] = left_pair;
                                        new_data.children[index - 1] = Some(left);
                                        new_data.children[index] = Some(new_child);
                                        new_data.count -= 1;
                                        return Remove::Update(pair, Node::wrap(new_data));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    // If it has a right sibling with capacity, same as above.
                    if has_right {
                        match self.0.children[index + 1] {
                            Some(ref old_right) if !old_right.too_small() => {
                                // Prepare the rebalanced node.
                                let left = child.push_max(
                                    old_right.0.children[0].clone(),
                                    self.0.keys[index].clone(),
                                );
                                match left.remove(key) {
                                    Remove::NoChange => return Remove::NoChange,
                                    Remove::Removed(_) => unreachable!(),
                                    Remove::Update(pair, new_child) => {
                                        // If we did remove something, we complete the rebalancing.
                                        let mut new_data = (&*self.0).clone();
                                        let (right, right_pair, _) = old_right.pop_min();
                                        new_data.keys[index] = right_pair;
                                        new_data.children[index] = Some(new_child);
                                        new_data.children[index + 1] = Some(right);
                                        new_data.count -= 1;
                                        return Remove::Update(pair, Node::wrap(new_data));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    // If it has neither, we'll have to merge it with a sibling.
                    // If we have a right sibling, we'll merge with that.
                    if has_right {
                        if let Some(ref right) = self.0.children[index + 1] {
                            let merged = Node::merge(self.0.keys[index].clone(), child, right);
                            match merged.remove(key) {
                                Remove::NoChange => return Remove::NoChange,
                                Remove::Removed(_) => unreachable!(),
                                Remove::Update(pair, new_child) => {
                                    if self.0.keys.len() == 1 {
                                        return Remove::Update(pair, new_child);
                                    }
                                    let mut new_data = (&*self.0).clone();
                                    new_data.count -= 1;
                                    new_data.keys.remove(index);
                                    new_data.children.remove(index);
                                    new_data.children[index] = Some(new_child);
                                    return Remove::Update(pair, Node::wrap(new_data));
                                }
                            }
                        }
                    }
                    // If we have a left sibling, we'll merge with that.
                    if has_left {
                        if let Some(ref left) = self.0.children[index - 1] {
                            let merged = Node::merge(self.0.keys[index - 1].clone(), left, child);
                            match merged.remove(key) {
                                Remove::NoChange => return Remove::NoChange,
                                Remove::Removed(_) => unreachable!(),
                                Remove::Update(pair, new_child) => {
                                    if self.0.keys.len() == 1 {
                                        return Remove::Update(pair, new_child);
                                    }
                                    let mut new_data = (&*self.0).clone();
                                    new_data.count -= 1;
                                    new_data.keys.remove(index - 1);
                                    new_data.children.remove(index - 1);
                                    new_data.children[index - 1] = Some(new_child);
                                    return Remove::Update(pair, Node::wrap(new_data));
                                }
                            }
                        }
                    }
                    // If none of the above, we're in a bad state.
                    unreachable!()
                }
                // Child at location, and it's big enough, we can recurse down.
                Some(ref child) => match child.remove(key) {
                    Remove::NoChange => Remove::NoChange,
                    Remove::Removed(_) => unreachable!(),
                    Remove::Update(pair, new_child) => {
                        let mut new_data = (&*self.0).clone();
                        new_data.children[index] = Some(new_child);
                        new_data.count -= 1;
                        Remove::Update(pair, Node::wrap(new_data))
                    }
                },
            },
        }
    }

    pub fn insert_mut(&mut self, key: Arc<K>, value: Arc<V>) -> Insert<K, V> {
        if self.0.keys.is_empty() {
            let node = Arc::make_mut(&mut self.0);
            node.keys.push(Pair { key, value });
            node.children.push(None);
            node.count += 1;
            return Insert::JustInc;
        }
        let (median, left, right) =
            match self.0.keys.binary_search_by_key(&&*key, |pair| &*pair.key) {
                // Key exists in node
                Ok(index) => {
                    if Arc::ptr_eq(&value, &self.0.keys[index].value) {
                        return Insert::NoChange;
                    } else {
                        let mut node = Arc::make_mut(&mut self.0);
                        node.keys[index] = Pair { key, value };
                        return Insert::JustInc;
                    }
                }
                // Key is adjacent to some key in node
                Err(index) => {
                    let mut has_room = self.has_room();
                    let mut node = Arc::make_mut(&mut self.0);
                    let action = match node.children[index] {
                        // No child at location, this is the target node.
                        None => InsertAt,
                        // Child at location, pass it on.
                        Some(ref mut child) => match child.insert_mut(key.clone(), value.clone()) {
                            Insert::NoChange => NoAction,
                            Insert::JustInc => IncAction,
                            Insert::Update(_) => unreachable!(),
                            Insert::Split(left, median, right) => InsertSplit(left, median, right),
                        },
                    };
                    match action {
                        NoAction => return Insert::NoChange,
                        IncAction => {
                            node.count += 1;
                            return Insert::JustInc;
                        }
                        InsertAt => {
                            if has_room {
                                node.keys.insert(index, Pair { key, value });
                                node.children.insert(index + 1, None);
                                node.count += 1;
                                return Insert::JustInc;
                            } else {
                                (Pair { key, value }, None, None)
                            }
                        }
                        InsertSplit(left, median, right) => {
                            if has_room {
                                node.children[index] = Some(left);
                                node.keys.insert(index, median);
                                node.children.insert(index + 1, Some(right));
                                node.count += 1;
                                return Insert::JustInc;
                            } else {
                                (median, Some(left), Some(right))
                            }
                        }
                    }
                }
            };
        self.split(median, left, right)
    }

    pub fn remove_mut(&mut self, key: &K) -> Remove<K, V> {
        let action = match self.0.keys.binary_search_by_key(&&*key, |pair| &*pair.key) {
            // Key exists in node, remove it.
            Ok(index) => {
                match (&self.0.children[index], &self.0.children[index + 1]) {
                    // If we're a leaf, just delete the entry.
                    (&None, &None) => RemoveAction::DeleteAt(index),
                    // If the left hand child has capacity, pull the predecessor up.
                    (&Some(ref left), _) if !left.too_small() => {
                        RemoveAction::PullUp(left.max().unwrap().key.clone(), index, index)
                    }
                    // If the right hand child has capacity, pull the successor up.
                    (_, &Some(ref right)) if !right.too_small() => {
                        RemoveAction::PullUp(right.min().unwrap().key.clone(), index, index + 1)
                    }
                    // If neither child has capacity, we'll have to merge them.
                    (&Some(_), &Some(_)) => RemoveAction::Merge(index),
                    // If one child exists and the other doesn't, we're in a bad state.
                    _ => unreachable!(),
                }
            }
            // Key is adjacent to some key in node
            Err(index) => match self.0.children[index] {
                // No child at location means key isn't in map.
                None => return Remove::NoChange,
                // Child at location, but it's at minimum capacity.
                Some(ref child) if child.too_small() => {
                    let left = if index > 0 {
                        self.0.children.get(index - 1)
                    } else {
                        None
                    }; // index is usize and can't be negative, best make sure it never is.
                    match (left, self.0.children.get(index + 1)) {
                        // If it has a left sibling with capacity, steal a key from it.
                        (Some(&Some(ref old_left)), _) if !old_left.too_small() => {
                            RemoveAction::StealFromLeft(index)
                        }
                        // If it has a right sibling with capacity, same as above.
                        (_, Some(&Some(ref old_right))) if !old_right.too_small() => {
                            RemoveAction::StealFromRight(index)
                        }
                        // If it has neither, we'll have to merge it with a sibling.
                        // If we have a right sibling, we'll merge with that.
                        (_, Some(&Some(_))) => RemoveAction::MergeFirst(index),
                        // If we have a left sibling, we'll merge with that.
                        (Some(&Some(_)), _) => RemoveAction::MergeFirst(index - 1),
                        // If none of the above, we're in a bad state.
                        _ => unreachable!(),
                    }
                }
                // Child at location, and it's big enough, we can recurse down.
                Some(_) => RemoveAction::ContinueDown(index),
            },
        };
        match action {
            RemoveAction::DeleteAt(index) => {
                let mut node = Arc::make_mut(&mut self.0);
                let pair = node.keys.remove(index);
                node.children.remove(index);
                node.count -= 1;
                Remove::Removed(pair)
            }
            RemoveAction::PullUp(key, pull_to, child_index) => {
                let mut node = Arc::make_mut(&mut self.0);
                let mut children = &mut node.children;
                let mut update = None;
                let mut pair;
                if let Some(&mut Some(ref mut child)) = children.get_mut(child_index) {
                    match child.remove_mut(&key) {
                        Remove::NoChange => unreachable!(),
                        Remove::Removed(pulled_pair) => {
                            node.keys.push(pulled_pair);
                            pair = node.keys.swap_remove(pull_to);
                            node.count -= 1;
                        }
                        Remove::Update(pulled_pair, new_child) => {
                            node.keys.push(pulled_pair);
                            pair = node.keys.swap_remove(pull_to);
                            node.count -= 1;
                            update = Some(new_child);
                        }
                    }
                } else {
                    unreachable!()
                }
                if let Some(new_child) = update {
                    children[child_index] = Some(new_child);
                }
                Remove::Removed(pair)
            }
            RemoveAction::Merge(index) => {
                let mut merged_child = if let (Some(&Some(ref left)), Some(&Some(ref right))) =
                    (self.0.children.get(index), self.0.children.get(index + 1))
                {
                    Node::merge(self.0.keys[index].clone(), left, right)
                } else {
                    unreachable!()
                };
                let mut node = Arc::make_mut(&mut self.0);
                let pair = node.keys.remove(index);
                let new_child = match merged_child.remove_mut(key) {
                    Remove::NoChange | Remove::Removed(_)=> merged_child,
                    Remove::Update(_, updated_child) => updated_child,
                };
                if node.keys.is_empty() {
                    // If we've depleted the root node, the merged child becomes the root.
                    Remove::Update(pair, new_child)
                } else {
                    node.count -= 1;
                    node.children.remove(index + 1);
                    node.children[index] = Some(new_child);
                    Remove::Removed(pair)
                }
            }
            RemoveAction::StealFromLeft(index) => {
                let mut node = Arc::make_mut(&mut self.0);
                let mut update = None;
                let mut out_pair;
                {
                    let mut children = node.children
                        .index_mut(index - 1..index + 1)
                        .iter_mut()
                        .map(|n| {
                            if let Some(ref mut o) = *n {
                                o
                            } else {
                                unreachable!()
                            }
                        });
                    let mut left = children.next().unwrap();
                    let mut child = children.next().unwrap();
                    // Prepare the rebalanced node.
                    child.push_min_mut(
                        left.0.children.last().unwrap().clone(),
                        node.keys[index - 1].clone(),
                    );
                    match child.remove_mut(key) {
                        Remove::NoChange => {
                            // Key wasn't there, we need to revert the steal.
                            child.pop_min_mut();
                            return Remove::NoChange;
                        }
                        Remove::Removed(pair) => {
                            // If we did remove something, we complete the rebalancing.
                            let (left_pair, _) = left.pop_max_mut();
                            node.keys[index - 1] = left_pair;
                            node.count -= 1;
                            out_pair = pair;
                        }
                        Remove::Update(pair, new_child) => {
                            // If we did remove something, we complete the rebalancing.
                            let (left_pair, _) = left.pop_max_mut();
                            node.keys[index - 1] = left_pair;
                            update = Some(new_child);
                            node.count -= 1;
                            out_pair = pair;
                        }
                    }
                }
                if let Some(new_child) = update {
                    node.children[index] = Some(new_child);
                }
                Remove::Removed(out_pair)
            }
            RemoveAction::StealFromRight(index) => {
                let mut node = Arc::make_mut(&mut self.0);
                let mut update = None;
                let mut out_pair;
                {
                    let mut children = node.children.index_mut(index..index + 2).iter_mut().map(
                        |n| {
                            if let Some(ref mut o) = *n {
                                o
                            } else {
                                unreachable!()
                            }
                        },
                    );
                    let mut child = children.next().unwrap();
                    let mut right = children.next().unwrap();
                    // Prepare the rebalanced node.
                    child.push_max_mut(right.0.children[0].clone(), node.keys[index].clone());
                    match child.remove_mut(key) {
                        Remove::NoChange => {
                            // Key wasn't there, we need to revert the steal.
                            child.pop_max_mut();
                            return Remove::NoChange;
                        }
                        Remove::Removed(pair) => {
                            // If we did remove something, we complete the rebalancing.
                            let (right_pair, _) = right.pop_min_mut();
                            node.keys[index] = right_pair;
                            node.count -= 1;
                            out_pair = pair;
                        }
                        Remove::Update(pair, new_child) => {
                            // If we did remove something, we complete the rebalancing.
                            let (right_pair, _) = right.pop_min_mut();
                            node.keys[index] = right_pair;
                            update = Some(new_child);
                            node.count -= 1;
                            out_pair = pair;
                        }
                    }
                }
                if let Some(new_child) = update {
                    node.children[index] = Some(new_child);
                }
                Remove::Removed(out_pair)
            }
            RemoveAction::MergeFirst(index) => {
                let mut merged = if let (Some(&Some(ref left)), Some(&Some(ref right))) =
                    (self.0.children.get(index), self.0.children.get(index + 1))
                {
                    Node::merge(self.0.keys[index].clone(), left, right)
                } else {
                    unreachable!()
                };
                let mut node = Arc::make_mut(&mut self.0);
                let mut update;
                let mut out_pair;
                {
                    match merged.remove_mut(key) {
                        Remove::NoChange => return Remove::NoChange,
                        Remove::Removed(pair) => {
                            if node.keys.len() == 1 {
                                return Remove::Update(pair, merged);
                            }
                            node.count -= 1;
                            node.keys.remove(index);
                            node.children.remove(index);
                            update = Some(merged);
                            out_pair = pair;
                        }
                        Remove::Update(pair, new_child) => {
                            if node.keys.len() == 1 {
                                return Remove::Update(pair, new_child);
                            }
                            node.count -= 1;
                            node.keys.remove(index);
                            node.children.remove(index);
                            update = Some(new_child);
                            out_pair = pair;
                        }
                    }
                }
                if let Some(new_child) = update {
                    node.children[index] = Some(new_child);
                }
                Remove::Removed(out_pair)
            }
            RemoveAction::ContinueDown(index) => {
                let mut node = Arc::make_mut(&mut self.0);
                let mut update = None;
                let mut out_pair;
                if let Some(&mut Some(ref mut child)) = node.children.get_mut(index) {
                    match child.remove_mut(key) {
                        Remove::NoChange => return Remove::NoChange,
                        Remove::Removed(pair) => {
                            node.count -= 1;
                            out_pair = pair;
                        }
                        Remove::Update(pair, new_child) => {
                            update = Some(new_child);
                            node.count -= 1;
                            out_pair = pair;
                        }
                    }
                } else {
                    unreachable!()
                }
                if let Some(new_child) = update {
                    node.children[index] = Some(new_child);
                }
                Remove::Removed(out_pair)
            }
        }
    }
}

impl<K, V> Debug for Node<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Node[{:?}]{{ {:?} }}", self.len(), self.0)
    }
}

impl<K, V> Debug for Pair<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "[{:?} => {:?}]", self.key, self.value)
    }
}

impl<K, V> Debug for NodeData<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for i in 0..self.keys.len() {
            write!(f, "{:?}, {:?}, ", self.children[i], self.keys[i])?;
        }
        write!(f, "{:?}", self.children[self.keys.len()])
    }
}

// Iterator

enum IterItem<K, V> {
    Consider(Node<K, V>),
    Yield(Pair<K, V>),
}

pub struct Iter<K, V> {
    stack: Vec<IterItem<K, V>>,
    remaining: usize,
}

impl<K, V> Iter<K, V> {
    pub fn new(root: &Node<K, V>) -> Self {
        Iter {
            stack: vec![IterItem::Consider(root.clone())],
            remaining: root.len(),
        }
    }

    fn push_node(&mut self, maybe_node: &Option<Node<K, V>>) {
        if let Some(ref node) = *maybe_node {
            self.stack.push(IterItem::Consider(node.clone()))
        }
    }

    fn push(&mut self, node: &Node<K, V>) {
        for n in 0..node.0.keys.len() {
            let i = node.0.keys.len() - n;
            self.push_node(&node.0.children[i]);
            self.stack.push(IterItem::Yield(node.0.keys[i - 1].clone()));
        }
        self.push_node(&node.0.children[0]);
    }
}

impl<K, V> Iterator for Iter<K, V> {
    type Item = (Arc<K>, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                None => return None,
                Some(IterItem::Consider(node)) => self.push(&node),
                Some(IterItem::Yield(pair)) => {
                    self.remaining -= 1;
                    return Some((pair.key, pair.value));
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
