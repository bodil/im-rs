// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::ops::IndexMut;
use std::sync::Arc;

use self::Insert::*;
use self::InsertAction::*;

const NODE_SIZE: usize = 16; // Must be an even number!
const MEDIAN: usize = (NODE_SIZE + 1) >> 1;

pub trait BTreeValue: Clone {
    type Key;
    fn ptr_eq(&self, other: &Self) -> bool;
    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>;
    fn search_value(slice: &[Self], value: &Self) -> Result<usize, usize>;
    fn cmp_keys(&self, other: &Self) -> Ordering;
}

pub struct Node<A>(Arc<NodeData<A>>);

struct NodeData<A> {
    count: usize,
    keys: Vec<A>,
    children: Vec<Option<Node<A>>>,
}

pub enum Insert<A> {
    NoChange,
    JustInc,
    Update(Node<A>),
    Split(Node<A>, A, Node<A>),
}

enum InsertAction<A> {
    NoAction,
    IncAction,
    InsertAt,
    InsertSplit(Node<A>, A, Node<A>),
}

pub enum Remove<A> {
    NoChange,
    Removed(A),
    Update(A, Node<A>),
}

enum RemoveAction {
    DeleteAt(usize),
    PullUp(usize, usize, usize),
    Merge(usize),
    StealFromLeft(usize),
    StealFromRight(usize),
    MergeFirst(usize),
    ContinueDown(usize),
}

impl<A> Clone for Node<A> {
    fn clone(&self) -> Self {
        Node(self.0.clone())
    }
}

impl<A: Clone> Clone for NodeData<A> {
    fn clone(&self) -> Self {
        NodeData {
            count: self.count,
            keys: self.keys.clone(),
            children: self.children.clone(),
        }
    }
}

impl<A> NodeData<A> {
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

impl<A> Default for Node<A> {
    fn default() -> Self {
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(None);
        Node(Arc::new(NodeData {
            count: 0,
            keys: Vec::with_capacity(NODE_SIZE),
            children,
        }))
    }
}

impl<A> Node<A> {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.count
    }

    #[inline]
    fn maybe_len(node_or: &Option<Node<A>>) -> usize {
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
        Default::default()
    }

    #[inline]
    pub fn singleton(value: A) -> Self {
        let mut keys = Vec::with_capacity(NODE_SIZE);
        keys.push(value);
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
    pub fn from_split(left: Node<A>, median: A, right: Node<A>) -> Self {
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
    fn wrap(data: NodeData<A>) -> Self {
        Node(Arc::new(data))
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn min(&self) -> Option<&A> {
        match *self.0.children.first().unwrap() {
            None => self.0.keys.first(),
            Some(ref child) => child.min(),
        }
    }

    pub fn max(&self) -> Option<&A> {
        match *self.0.children.last().unwrap() {
            None => self.0.keys.last(),
            Some(ref child) => child.max(),
        }
    }
}

impl<A: BTreeValue> Node<A> {
    pub fn lookup<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.0.keys.is_empty() {
            return None;
        }
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match A::search_key(&self.0.keys, key) {
            Ok(index) => Some(&self.0.keys[index]),
            Err(index) => match self.0.children[index] {
                None => None,
                Some(ref node) => node.lookup(key),
            },
        }
    }

    pub fn lookup_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.0.keys.is_empty() {
            return None;
        }
        let node = Arc::make_mut(&mut self.0);
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match A::search_key(&node.keys, key) {
            Ok(index) => Some(&mut node.keys[index]),
            Err(index) => match node.children[index] {
                None => None,
                Some(ref mut child) => child.lookup_mut(key),
            },
        }
    }

    fn split(&self, value: A, ins_left: Option<Node<A>>, ins_right: Option<Node<A>>) -> Insert<A> {
        let mut new_keys = self.0.keys.clone();
        let mut new_children = self.0.children.clone();

        match A::search_value(&new_keys, &value) {
            Ok(_) => unreachable!(),
            Err(index) => {
                new_children[index] = ins_left;
                new_keys.insert(index, value);
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

    pub fn insert(&self, value: A) -> Insert<A> {
        if self.0.keys.is_empty() {
            return Insert::Update(Node::singleton(value));
        }
        match A::search_value(&self.0.keys, &value) {
            // Key exists in node
            Ok(index) => {
                if value.ptr_eq(&self.0.keys[index]) {
                    Insert::NoChange
                } else {
                    let mut new_data = (&*self.0).clone();
                    new_data.keys[index] = value;
                    Insert::Update(Node::wrap(new_data))
                }
            }
            // Key is adjacent to some key in node
            Err(index) => match self.0.children[index] {
                // No child at location, this is the target node.
                None => {
                    if self.has_room() {
                        let mut new_data = (&*self.0).clone();
                        new_data.keys.insert(index, value);
                        new_data.children.insert(index + 1, None);
                        new_data.count += 1;
                        Insert::Update(Node::wrap(new_data))
                    } else {
                        self.split(value, None, None)
                    }
                }
                // Child at location, pass it on.
                Some(ref node) => match node.insert(value) {
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

    fn merge(pair: A, left: &Node<A>, right: &Node<A>) -> Node<A> {
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

    fn pop_min(&self) -> (Node<A>, A, Option<Node<A>>) {
        let mut new_data = (&*self.0).clone();
        let pair = new_data.keys.remove(0);
        let child = new_data.children.remove(0);
        new_data.count -= 1 + Node::maybe_len(&child);
        (Node::wrap(new_data), pair, child)
    }

    fn pop_min_mut(&mut self) -> (A, Option<Node<A>>) {
        let node = Arc::make_mut(&mut self.0);
        let pair = node.keys.remove(0);
        let child = node.children.remove(0);
        node.count -= 1 + Node::maybe_len(&child);
        (pair, child)
    }

    fn pop_max(&self) -> (Node<A>, A, Option<Node<A>>) {
        let mut new_data = (&*self.0).clone();
        let pair = new_data.keys.pop().unwrap();
        let child = new_data.children.pop().unwrap();
        new_data.count -= 1 + Node::maybe_len(&child);
        (Node::wrap(new_data), pair, child)
    }

    fn pop_max_mut(&mut self) -> (A, Option<Node<A>>) {
        let node = Arc::make_mut(&mut self.0);
        let pair = node.keys.pop().unwrap();
        let child = node.children.pop().unwrap();
        node.count -= 1 + Node::maybe_len(&child);
        (pair, child)
    }

    fn push_min(&self, child: Option<Node<A>>, pair: A) -> Node<A> {
        let mut new_data = (&*self.0).clone();
        new_data.count += 1 + Node::maybe_len(&child);
        new_data.keys.insert(0, pair);
        new_data.children.insert(0, child);
        Node::wrap(new_data)
    }

    fn push_min_mut(&mut self, child: Option<Node<A>>, pair: A) {
        let node = Arc::make_mut(&mut self.0);
        node.count += 1 + Node::maybe_len(&child);
        node.keys.insert(0, pair);
        node.children.insert(0, child);
    }

    fn push_max(&self, child: Option<Node<A>>, pair: A) -> Node<A> {
        let mut new_data = (&*self.0).clone();
        new_data.count += 1 + Node::maybe_len(&child);
        new_data.keys.push(pair);
        new_data.children.push(child);
        Node::wrap(new_data)
    }

    fn push_max_mut(&mut self, child: Option<Node<A>>, pair: A) {
        let node = Arc::make_mut(&mut self.0);
        node.count += 1 + Node::maybe_len(&child);
        node.keys.push(pair);
        node.children.push(child);
    }

    fn pull_up<BK>(
        &self,
        key: &BK,
        from_index: usize,
        from_child: &Node<A>,
        pull_to: usize,
        child_index: usize,
    ) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        match from_child.remove_index(Ok(from_index), key) {
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

    pub fn remove<BK>(&self, key: &BK) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.remove_index(A::search_key(&self.0.keys, key), key)
    }

    pub fn remove_index<BK>(&self, index: Result<usize, usize>, key: &BK) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        match index {
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
                        self.pull_up(key, left.0.keys.len() - 1, left, index, index)
                    }
                    // If the right hand child has capacity, pull the successor up.
                    (_, &Some(ref right)) if !right.too_small() => {
                        self.pull_up(key, 0, right, index, index + 1)
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

    pub fn insert_mut(&mut self, value: A) -> Insert<A> {
        if self.0.keys.is_empty() {
            let node = Arc::make_mut(&mut self.0);
            node.keys.push(value);
            node.children.push(None);
            node.count += 1;
            return Insert::JustInc;
        }
        let (median, left, right) = match A::search_value(&self.0.keys, &value) {
            // Key exists in node
            Ok(index) => {
                if value.ptr_eq(&self.0.keys[index]) {
                    return Insert::NoChange;
                } else {
                    let mut node = Arc::make_mut(&mut self.0);
                    node.keys[index] = value;
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
                    Some(ref mut child) => match child.insert_mut(value.clone()) {
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
                            node.keys.insert(index, value);
                            node.children.insert(index + 1, None);
                            node.count += 1;
                            return Insert::JustInc;
                        } else {
                            (value, None, None)
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

    pub fn remove_mut<BK>(&mut self, key: &BK) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = A::search_key(&self.0.keys, key);
        self.remove_mut_index(index, key)
    }

    fn remove_mut_index<BK>(&mut self, index: Result<usize, usize>, key: &BK) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let action = match index {
            // Key exists in node, remove it.
            Ok(index) => {
                match (&self.0.children[index], &self.0.children[index + 1]) {
                    // If we're a leaf, just delete the entry.
                    (&None, &None) => RemoveAction::DeleteAt(index),
                    // If the left hand child has capacity, pull the predecessor up.
                    (&Some(ref left), _) if !left.too_small() => {
                        RemoveAction::PullUp(left.0.keys.len() - 1, index, index)
                    }
                    // If the right hand child has capacity, pull the successor up.
                    (_, &Some(ref right)) if !right.too_small() => {
                        RemoveAction::PullUp(0, index, index + 1)
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
            RemoveAction::PullUp(target_index, pull_to, child_index) => {
                let mut node = Arc::make_mut(&mut self.0);
                let mut children = &mut node.children;
                let mut update = None;
                let mut pair;
                if let Some(&mut Some(ref mut child)) = children.get_mut(child_index) {
                    match child.remove_mut_index(Ok(target_index), key) {
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
                    Remove::NoChange | Remove::Removed(_) => merged_child,
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

// Iterator

enum IterItem<A> {
    Consider(Node<A>),
    Yield(A),
}

pub struct Iter<A> {
    fwd_last: Option<A>,
    fwd_stack: Vec<IterItem<A>>,
    back_last: Option<A>,
    back_stack: Vec<IterItem<A>>,
    remaining: usize,
}

fn push_node<A: Clone>(stack: &mut Vec<IterItem<A>>, maybe_node: &Option<Node<A>>) {
    if let Some(ref node) = *maybe_node {
        stack.push(IterItem::Consider(node.clone()))
    }
}

fn push<A: Clone>(stack: &mut Vec<IterItem<A>>, node: &Node<A>) {
    for n in 0..node.0.keys.len() {
        let i = node.0.keys.len() - n;
        push_node(stack, &node.0.children[i]);
        stack.push(IterItem::Yield(node.0.keys[i - 1].clone()));
    }
    push_node(stack, &node.0.children[0]);
}

impl<A: Clone> Iter<A> {
    pub fn new(root: &Node<A>) -> Self {
        Iter {
            fwd_last: None,
            fwd_stack: vec![IterItem::Consider(root.clone())],
            back_last: None,
            back_stack: vec![IterItem::Consider(root.clone())],
            remaining: root.len(),
        }
    }

    fn push_fwd(&mut self, node: &Node<A>) {
        push(&mut self.fwd_stack, node)
    }

    fn push_node_back(&mut self, maybe_node: &Option<Node<A>>) {
        if let Some(ref node) = *maybe_node {
            self.back_stack.push(IterItem::Consider(node.clone()))
        }
    }

    fn push_back(&mut self, node: &Node<A>) {
        for i in 0..node.0.keys.len() {
            self.push_node_back(&node.0.children[i]);
            self.back_stack
                .push(IterItem::Yield(node.0.keys[i].clone()));
        }
        self.push_node_back(&node.0.children[node.0.keys.len()]);
    }
}

impl<A> Iterator for Iter<A>
where
    A: BTreeValue,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.fwd_stack.pop() {
                None => {
                    self.remaining = 0;
                    return None;
                }
                Some(IterItem::Consider(node)) => self.push_fwd(&node),
                Some(IterItem::Yield(value)) => {
                    if let Some(ref last) = self.back_last {
                        if value.cmp_keys(last) != Ordering::Less {
                            self.fwd_stack.clear();
                            self.back_stack.clear();
                            self.remaining = 0;
                            return None;
                        }
                    }
                    self.remaining -= 1;
                    self.fwd_last = Some(value.clone());
                    return Some(value);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<A> DoubleEndedIterator for Iter<A>
where
    A: BTreeValue,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            match self.back_stack.pop() {
                None => {
                    self.remaining = 0;
                    return None;
                }
                Some(IterItem::Consider(node)) => self.push_back(&node),
                Some(IterItem::Yield(value)) => {
                    if let Some(ref last) = self.fwd_last {
                        if value.cmp_keys(last) != Ordering::Greater {
                            self.fwd_stack.clear();
                            self.back_stack.clear();
                            self.remaining = 0;
                            return None;
                        }
                    }
                    self.remaining -= 1;
                    self.back_last = Some(value.clone());
                    return Some(value);
                }
            }
        }
    }
}

impl<A: BTreeValue> ExactSizeIterator for Iter<A> {}

// DiffIter

pub struct DiffIter<A> {
    old_stack: Vec<IterItem<A>>,
    new_stack: Vec<IterItem<A>>,
}

#[derive(PartialEq, Eq)]
pub enum DiffItem<A> {
    Add(A),
    Update { old: A, new: A },
    Remove(A),
}

impl<A> DiffIter<A> {
    pub fn new(old: &Node<A>, new: &Node<A>) -> Self {
        DiffIter {
            old_stack: if old.0.keys.is_empty() {
                Vec::new()
            } else {
                vec![IterItem::Consider(old.clone())]
            },
            new_stack: if new.0.keys.is_empty() {
                Vec::new()
            } else {
                vec![IterItem::Consider(new.clone())]
            },
        }
    }
}

impl<A: PartialEq> Iterator for DiffIter<A>
where
    A: BTreeValue,
{
    type Item = DiffItem<A>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.old_stack.pop(), self.new_stack.pop()) {
                (None, None) => return None,
                (None, Some(new)) => match new {
                    IterItem::Consider(new) => push(&mut self.new_stack, &new),
                    IterItem::Yield(new) => return Some(DiffItem::Add(new)),
                },
                (Some(old), None) => match old {
                    IterItem::Consider(old) => push(&mut self.old_stack, &old),
                    IterItem::Yield(old) => return Some(DiffItem::Remove(old)),
                },
                (Some(old), Some(new)) => match (old, new) {
                    (IterItem::Consider(old), IterItem::Consider(new)) => {
                        if !Arc::ptr_eq(&old.0, &new.0) {
                            match old.0.keys[0].cmp_keys(&new.0.keys[0]) {
                                Ordering::Less => {
                                    push(&mut self.old_stack, &old);
                                    self.new_stack.push(IterItem::Consider(new));
                                }
                                Ordering::Greater => {
                                    self.old_stack.push(IterItem::Consider(old));
                                    push(&mut self.new_stack, &new);
                                }
                                Ordering::Equal => {
                                    push(&mut self.old_stack, &old);
                                    push(&mut self.new_stack, &new);
                                }
                            }
                        }
                    }
                    (IterItem::Consider(old), IterItem::Yield(new)) => {
                        push(&mut self.old_stack, &old);
                        self.new_stack.push(IterItem::Yield(new));
                    }
                    (IterItem::Yield(old), IterItem::Consider(new)) => {
                        self.old_stack.push(IterItem::Yield(old));
                        push(&mut self.new_stack, &new);
                    }
                    (IterItem::Yield(old), IterItem::Yield(new)) => match old.cmp_keys(&new) {
                        Ordering::Less => {
                            self.new_stack.push(IterItem::Yield(new));
                            return Some(DiffItem::Remove(old));
                        }
                        Ordering::Equal => if old != new {
                            return Some(DiffItem::Update { old, new });
                        },
                        Ordering::Greater => {
                            self.old_stack.push(IterItem::Yield(old));
                            return Some(DiffItem::Add(new));
                        }
                    },
                },
            }
        }
    }
}
