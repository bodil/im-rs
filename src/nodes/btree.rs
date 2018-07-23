// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::mem;
use std::ops::IndexMut;
use util::clone_ref;

use util::Ref;

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
    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>;
    fn cmp_values(&self, other: &Self) -> Ordering;
}

pub struct Node<A> {
    count: usize,
    keys: Vec<A>,
    children: Vec<Option<Ref<Node<A>>>>,
}

pub enum Insert<A> {
    Added,
    Replaced(A),
    Update(Node<A>),
    Split(Node<A>, A, Node<A>),
}

enum InsertAction<A> {
    AddedAction,
    ReplacedAction(A),
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

impl<A> Clone for Node<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        Node {
            count: self.count,
            keys: self.keys.clone(),
            children: self.children.clone(),
        }
    }
}

impl<A> Default for Node<A> {
    fn default() -> Self {
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(None);
        Node {
            count: 0,
            keys: Vec::with_capacity(NODE_SIZE),
            children,
        }
    }
}

impl<A> Node<A>
where
    A: Clone,
{
    #[inline]
    fn has_room(&self) -> bool {
        self.keys.len() < NODE_SIZE
    }

    #[inline]
    fn too_small(&self) -> bool {
        self.keys.len() < MEDIAN
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.children[0].is_none()
    }

    fn sum_up_children(&self) -> usize {
        let mut c = self.count;
        for child in &self.children {
            match child {
                None => continue,
                Some(ref node) => c += node.len(),
            }
        }
        c
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    fn maybe_len(node_or: &Option<Ref<Node<A>>>) -> usize {
        match node_or {
            None => 0,
            Some(ref node) => node.len(),
        }
    }

    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn unit(value: A) -> Self {
        let mut keys = Vec::with_capacity(NODE_SIZE);
        keys.push(value);
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(None);
        children.push(None);
        Node {
            count: 1,
            keys,
            children,
        }
    }

    #[inline]
    pub fn from_split(left: Node<A>, median: A, right: Node<A>) -> Self {
        let count = left.len() + right.len() + 1;
        let mut keys = Vec::with_capacity(NODE_SIZE);
        keys.push(median);
        let mut children = Vec::with_capacity(NODE_SIZE + 1);
        children.push(Some(Ref::from(left)));
        children.push(Some(Ref::from(right)));
        Node {
            count,
            keys,
            children,
        }
    }

    pub fn min(&self) -> Option<&A> {
        match self.children.first().unwrap() {
            None => self.keys.first(),
            Some(ref child) => child.min(),
        }
    }

    pub fn max(&self) -> Option<&A> {
        match self.children.last().unwrap() {
            None => self.keys.last(),
            Some(ref child) => child.max(),
        }
    }
}

impl<A: BTreeValue> Node<A> {
    fn child_contains<BK>(&self, index: usize, key: &BK) -> bool
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if let Some(Some(ref child)) = self.children.get(index) {
            child.lookup(key).is_some()
        } else {
            false
        }
    }

    pub fn lookup<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match A::search_key(&self.keys, key) {
            Ok(index) => Some(&self.keys[index]),
            Err(index) => match self.children[index] {
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
        if self.keys.is_empty() {
            return None;
        }
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match A::search_key(&self.keys, key) {
            Ok(index) => Some(&mut self.keys[index]),
            Err(index) => match self.children[index] {
                None => None,
                Some(ref mut child_ref) => {
                    let child = Ref::make_mut(child_ref);
                    child.lookup_mut(key)
                }
            },
        }
    }

    fn split(
        &mut self,
        value: A,
        ins_left: Option<Node<A>>,
        ins_right: Option<Node<A>>,
    ) -> Insert<A> {
        let index = A::search_value(&self.keys, &value).unwrap_err();
        self.children[index] = ins_left.map(Ref::from);
        self.keys.insert(index, value);
        self.children.insert(index + 1, ins_right.map(Ref::from));

        let mut left = Node {
            count: MEDIAN,
            keys: self.keys.drain(0..MEDIAN).collect(),
            children: self.children.drain(0..MEDIAN + 1).collect(),
        };
        let mut right = Node {
            count: MEDIAN,
            keys: self.keys.drain(1..).collect(),
            children: self.children.drain(..).collect(),
        };
        left.count = left.sum_up_children();
        right.count = right.sum_up_children();
        Split(left, self.keys.pop().unwrap(), right)
    }

    fn merge(middle: A, left: Node<A>, right: Node<A>) -> Node<A> {
        let count = left.len() + right.len() + 1;
        let mut keys = left.keys;
        keys.push(middle);
        keys.extend(right.keys);
        let mut children = left.children;
        children.extend(right.children);
        Node {
            count,
            keys,
            children,
        }
    }

    fn pop_min(&mut self) -> (A, Option<Ref<Node<A>>>) {
        let value = self.keys.remove(0);
        let child = self.children.remove(0);
        self.count -= 1 + Node::maybe_len(&child);
        (value, child)
    }

    fn pop_max(&mut self) -> (A, Option<Ref<Node<A>>>) {
        let value = self.keys.pop().unwrap();
        let child = self.children.pop().unwrap();
        self.count -= 1 + Node::maybe_len(&child);
        (value, child)
    }

    fn push_min(&mut self, child: Option<Ref<Node<A>>>, value: A) {
        self.count += 1 + Node::maybe_len(&child);
        self.keys.insert(0, value);
        self.children.insert(0, child);
    }

    fn push_max(&mut self, child: Option<Ref<Node<A>>>, value: A) {
        self.count += 1 + Node::maybe_len(&child);
        self.keys.push(value);
        self.children.push(child);
    }

    pub fn insert(&mut self, value: A) -> Insert<A> {
        if self.keys.is_empty() {
            self.keys.push(value);
            self.children.push(None);
            self.count += 1;
            return Insert::Added;
        }
        let (median, left, right) = match A::search_value(&self.keys, &value) {
            // Key exists in node
            Ok(index) => {
                return Insert::Replaced(mem::replace(&mut self.keys[index], value));
            }
            // Key is adjacent to some key in node
            Err(index) => {
                let mut has_room = self.has_room();
                let action = match self.children[index] {
                    // No child at location, this is the target node.
                    None => InsertAt,
                    // Child at location, pass it on.
                    Some(ref mut child_ref) => {
                        let child = Ref::make_mut(child_ref);
                        match child.insert(value.clone()) {
                            Insert::Added => AddedAction,
                            Insert::Replaced(value) => ReplacedAction(value),
                            Insert::Update(_) => unreachable!(),
                            Insert::Split(left, median, right) => InsertSplit(left, median, right),
                        }
                    }
                };
                match action {
                    ReplacedAction(value) => return Insert::Replaced(value),
                    AddedAction => {
                        self.count += 1;
                        return Insert::Added;
                    }
                    InsertAt => {
                        if has_room {
                            self.keys.insert(index, value);
                            self.children.insert(index + 1, None);
                            self.count += 1;
                            return Insert::Added;
                        } else {
                            (value, None, None)
                        }
                    }
                    InsertSplit(left, median, right) => {
                        if has_room {
                            self.children[index] = Some(Ref::from(left));
                            self.keys.insert(index, median);
                            self.children.insert(index + 1, Some(Ref::from(right)));
                            self.count += 1;
                            return Insert::Added;
                        } else {
                            (median, Some(left), Some(right))
                        }
                    }
                }
            }
        };
        self.split(median, left, right)
    }

    pub fn remove<BK>(&mut self, key: &BK) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = A::search_key(&self.keys, key);
        self.remove_index(index, key)
    }

    fn remove_index<BK>(&mut self, index: Result<usize, usize>, key: &BK) -> Remove<A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let action = match index {
            // Key exists in node, remove it.
            Ok(index) => {
                match (&self.children[index], &self.children[index + 1]) {
                    // If we're a leaf, just delete the entry.
                    (&None, &None) => RemoveAction::DeleteAt(index),
                    // If the left hand child has capacity, pull the predecessor up.
                    (&Some(ref left), _) if !left.too_small() => {
                        if left.is_leaf() {
                            RemoveAction::PullUp(left.keys.len() - 1, index, index)
                        } else {
                            RemoveAction::StealFromLeft(index + 1)
                        }
                    }
                    // If the right hand child has capacity, pull the successor up.
                    (_, &Some(ref right)) if !right.too_small() => {
                        if right.is_leaf() {
                            RemoveAction::PullUp(0, index, index + 1)
                        } else {
                            RemoveAction::StealFromRight(index)
                        }
                    }
                    // If neither child has capacity, we'll have to merge them.
                    (&Some(_), &Some(_)) => RemoveAction::Merge(index),
                    // If one child exists and the other doesn't, we're in a bad state.
                    _ => unreachable!(),
                }
            }
            // Key is adjacent to some key in node
            Err(index) => match self.children[index] {
                // No child at location means key isn't in map.
                None => return Remove::NoChange,
                // Child at location, but it's at minimum capacity.
                Some(ref child) if child.too_small() => {
                    let left = if index > 0 {
                        self.children.get(index - 1)
                    } else {
                        None
                    }; // index is usize and can't be negative, best make sure it never is.
                    match (left, self.children.get(index + 1)) {
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
                let pair = self.keys.remove(index);
                self.children.remove(index);
                self.count -= 1;
                Remove::Removed(pair)
            }
            RemoveAction::PullUp(target_index, pull_to, child_index) => {
                let mut children = &mut self.children;
                let mut update = None;
                let mut value;
                if let Some(&mut Some(ref mut child_ref)) = children.get_mut(child_index) {
                    let child = Ref::make_mut(child_ref);
                    match child.remove_index(Ok(target_index), key) {
                        Remove::NoChange => unreachable!(),
                        Remove::Removed(pulled_value) => {
                            self.keys.push(pulled_value);
                            value = self.keys.swap_remove(pull_to);
                            self.count -= 1;
                        }
                        Remove::Update(pulled_value, new_child) => {
                            self.keys.push(pulled_value);
                            value = self.keys.swap_remove(pull_to);
                            self.count -= 1;
                            update = Some(new_child);
                        }
                    }
                } else {
                    unreachable!()
                }
                if let Some(new_child) = update {
                    children[child_index] = Some(Ref::from(new_child));
                }
                Remove::Removed(value)
            }
            RemoveAction::Merge(index) => {
                let left = self.children.remove(index).unwrap();
                let right = mem::replace(&mut self.children[index], None).unwrap();
                let value = self.keys.remove(index);
                let mut merged_child = Node::merge(value, clone_ref(left), clone_ref(right));
                let (removed, new_child) = match merged_child.remove(key) {
                    Remove::NoChange => unreachable!(),
                    Remove::Removed(removed) => (removed, merged_child),
                    Remove::Update(removed, updated_child) => (removed, updated_child),
                };
                if self.keys.is_empty() {
                    // If we've depleted the root node, the merged child becomes the root.
                    Remove::Update(removed, new_child)
                } else {
                    self.count -= 1;
                    self.children[index] = Some(Ref::from(new_child));
                    Remove::Removed(removed)
                }
            }
            RemoveAction::StealFromLeft(index) => {
                let mut update = None;
                let mut out_value;
                {
                    let mut children = self
                        .children
                        .index_mut(index - 1..index + 1)
                        .iter_mut()
                        .map(|n| {
                            if let Some(ref mut o) = *n {
                                o
                            } else {
                                unreachable!()
                            }
                        });
                    let mut left = Ref::make_mut(children.next().unwrap());
                    let mut child = Ref::make_mut(children.next().unwrap());
                    // Prepare the rebalanced node.
                    child.push_min(
                        left.children.last().unwrap().clone(),
                        self.keys[index - 1].clone(),
                    );
                    match child.remove(key) {
                        Remove::NoChange => {
                            // Key wasn't there, we need to revert the steal.
                            child.pop_min();
                            return Remove::NoChange;
                        }
                        Remove::Removed(value) => {
                            // If we did remove something, we complete the rebalancing.
                            let (left_value, _) = left.pop_max();
                            self.keys[index - 1] = left_value;
                            self.count -= 1;
                            out_value = value;
                        }
                        Remove::Update(value, new_child) => {
                            // If we did remove something, we complete the rebalancing.
                            let (left_value, _) = left.pop_max();
                            self.keys[index - 1] = left_value;
                            update = Some(new_child);
                            self.count -= 1;
                            out_value = value;
                        }
                    }
                }
                if let Some(new_child) = update {
                    self.children[index] = Some(Ref::from(new_child));
                }
                Remove::Removed(out_value)
            }
            RemoveAction::StealFromRight(index) => {
                let mut update = None;
                let mut out_value;
                {
                    let mut children = self.children.index_mut(index..index + 2).iter_mut().map(
                        |n| {
                            if let Some(ref mut o) = *n {
                                o
                            } else {
                                unreachable!()
                            }
                        },
                    );
                    let mut child = Ref::make_mut(children.next().unwrap());
                    let mut right = Ref::make_mut(children.next().unwrap());
                    // Prepare the rebalanced node.
                    child.push_max(right.children[0].clone(), self.keys[index].clone());
                    match child.remove(key) {
                        Remove::NoChange => {
                            // Key wasn't there, we need to revert the steal.
                            child.pop_max();
                            return Remove::NoChange;
                        }
                        Remove::Removed(value) => {
                            // If we did remove something, we complete the rebalancing.
                            let (right_value, _) = right.pop_min();
                            self.keys[index] = right_value;
                            self.count -= 1;
                            out_value = value;
                        }
                        Remove::Update(value, new_child) => {
                            // If we did remove something, we complete the rebalancing.
                            let (right_value, _) = right.pop_min();
                            self.keys[index] = right_value;
                            update = Some(new_child);
                            self.count -= 1;
                            out_value = value;
                        }
                    }
                }
                if let Some(new_child) = update {
                    self.children[index] = Some(Ref::from(new_child));
                }
                Remove::Removed(out_value)
            }
            RemoveAction::MergeFirst(index) => {
                if self.keys[index].cmp_keys(key) != Ordering::Equal
                    && !self.child_contains(index, key)
                    && !self.child_contains(index + 1, key)
                {
                    return Remove::NoChange;
                }
                let left = self.children.remove(index).unwrap();
                let right = mem::replace(&mut self.children[index], None).unwrap();
                let middle = self.keys.remove(index);
                let mut merged = Node::merge(middle, clone_ref(left), clone_ref(right));
                let mut update;
                let mut out_value;
                match merged.remove(key) {
                    Remove::NoChange => {
                        panic!("nodes::btree::Node::remove: caught an absent key too late while merging");
                    }
                    Remove::Removed(value) => {
                        if self.keys.is_empty() {
                            return Remove::Update(value, merged);
                        }
                        self.count -= 1;
                        update = merged;
                        out_value = value;
                    }
                    Remove::Update(value, new_child) => {
                        if self.keys.is_empty() {
                            return Remove::Update(value, new_child);
                        }
                        self.count -= 1;
                        update = new_child;
                        out_value = value;
                    }
                }
                self.children[index] = Some(Ref::from(update));
                Remove::Removed(out_value)
            }
            RemoveAction::ContinueDown(index) => {
                let mut update = None;
                let mut out_value;
                if let Some(&mut Some(ref mut child_ref)) = self.children.get_mut(index) {
                    let child = Ref::make_mut(child_ref);
                    match child.remove(key) {
                        Remove::NoChange => return Remove::NoChange,
                        Remove::Removed(value) => {
                            self.count -= 1;
                            out_value = value;
                        }
                        Remove::Update(value, new_child) => {
                            update = Some(new_child);
                            self.count -= 1;
                            out_value = value;
                        }
                    }
                } else {
                    unreachable!()
                }
                if let Some(new_child) = update {
                    self.children[index] = Some(Ref::from(new_child));
                }
                Remove::Removed(out_value)
            }
        }
    }
}

// Iterator

enum IterItem<'a, A: 'a> {
    Consider(&'a Node<A>),
    Yield(&'a A),
}

pub struct Iter<'a, A: 'a> {
    fwd_last: Option<&'a A>,
    fwd_stack: Vec<IterItem<'a, A>>,
    back_last: Option<&'a A>,
    back_stack: Vec<IterItem<'a, A>>,
    remaining: usize,
}

impl<'a, A: 'a + Clone> Iter<'a, A> {
    pub fn new(root: &'a Node<A>) -> Self {
        Iter {
            fwd_last: None,
            fwd_stack: vec![IterItem::Consider(root)],
            back_last: None,
            back_stack: vec![IterItem::Consider(root)],
            remaining: root.len(),
        }
    }

    fn push_node(stack: &mut Vec<IterItem<'a, A>>, maybe_node: &'a Option<Ref<Node<A>>>) {
        if let Some(ref node) = *maybe_node {
            stack.push(IterItem::Consider(&node))
        }
    }

    fn push(stack: &mut Vec<IterItem<'a, A>>, node: &'a Node<A>) {
        for n in 0..node.keys.len() {
            let i = node.keys.len() - n;
            Iter::push_node(stack, &node.children[i]);
            stack.push(IterItem::Yield(&node.keys[i - 1]));
        }
        Iter::push_node(stack, &node.children[0]);
    }

    fn push_fwd(&mut self, node: &'a Node<A>) {
        Iter::push(&mut self.fwd_stack, node)
    }

    fn push_node_back(&mut self, maybe_node: &'a Option<Ref<Node<A>>>) {
        if let Some(ref node) = *maybe_node {
            self.back_stack.push(IterItem::Consider(&node))
        }
    }

    fn push_back(&mut self, node: &'a Node<A>) {
        for i in 0..node.keys.len() {
            self.push_node_back(&node.children[i]);
            self.back_stack.push(IterItem::Yield(&node.keys[i]));
        }
        self.push_node_back(&node.children[node.keys.len()]);
    }
}

impl<'a, A> Iterator for Iter<'a, A>
where
    A: 'a + BTreeValue,
{
    type Item = &'a A;

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
                        if value.cmp_values(last) != Ordering::Less {
                            self.fwd_stack.clear();
                            self.back_stack.clear();
                            self.remaining = 0;
                            return None;
                        }
                    }
                    self.remaining -= 1;
                    self.fwd_last = Some(value);
                    return Some(value);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, A> DoubleEndedIterator for Iter<'a, A>
where
    A: 'a + BTreeValue,
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
                        if value.cmp_values(last) != Ordering::Greater {
                            self.fwd_stack.clear();
                            self.back_stack.clear();
                            self.remaining = 0;
                            return None;
                        }
                    }
                    self.remaining -= 1;
                    self.back_last = Some(value);
                    return Some(value);
                }
            }
        }
    }
}

impl<'a, A: 'a + BTreeValue> ExactSizeIterator for Iter<'a, A> {}

// Consuming iterator

enum ConsumingIterItem<A> {
    Consider(Node<A>),
    Yield(A),
}

pub struct ConsumingIter<A> {
    fwd_last: Option<A>,
    fwd_stack: Vec<ConsumingIterItem<A>>,
    back_last: Option<A>,
    back_stack: Vec<ConsumingIterItem<A>>,
    remaining: usize,
}

impl<A: Clone> ConsumingIter<A> {
    pub fn new(root: &Node<A>) -> Self {
        ConsumingIter {
            fwd_last: None,
            fwd_stack: vec![ConsumingIterItem::Consider(root.clone())],
            back_last: None,
            back_stack: vec![ConsumingIterItem::Consider(root.clone())],
            remaining: root.len(),
        }
    }

    fn push_node(stack: &mut Vec<ConsumingIterItem<A>>, maybe_node: Option<Ref<Node<A>>>) {
        if let Some(node) = maybe_node {
            stack.push(ConsumingIterItem::Consider(clone_ref(node)))
        }
    }

    fn push(stack: &mut Vec<ConsumingIterItem<A>>, mut node: Node<A>) {
        for _n in 0..node.keys.len() {
            ConsumingIter::push_node(stack, node.children.pop().unwrap());
            stack.push(ConsumingIterItem::Yield(node.keys.pop().unwrap()));
        }
        ConsumingIter::push_node(stack, node.children.pop().unwrap());
    }

    fn push_fwd(&mut self, node: Node<A>) {
        ConsumingIter::push(&mut self.fwd_stack, node)
    }

    fn push_node_back(&mut self, maybe_node: Option<Ref<Node<A>>>) {
        if let Some(node) = maybe_node {
            self.back_stack
                .push(ConsumingIterItem::Consider(clone_ref(node)))
        }
    }

    fn push_back(&mut self, mut node: Node<A>) {
        for _i in 0..node.keys.len() {
            self.push_node_back(node.children.remove(0));
            self.back_stack
                .push(ConsumingIterItem::Yield(node.keys.remove(0)));
        }
        self.push_node_back(node.children.pop().unwrap());
    }
}

impl<A> Iterator for ConsumingIter<A>
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
                Some(ConsumingIterItem::Consider(node)) => self.push_fwd(node),
                Some(ConsumingIterItem::Yield(value)) => {
                    if let Some(ref last) = self.back_last {
                        if value.cmp_values(last) != Ordering::Less {
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

impl<A> DoubleEndedIterator for ConsumingIter<A>
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
                Some(ConsumingIterItem::Consider(node)) => self.push_back(node),
                Some(ConsumingIterItem::Yield(value)) => {
                    if let Some(ref last) = self.fwd_last {
                        if value.cmp_values(last) != Ordering::Greater {
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

impl<A: BTreeValue> ExactSizeIterator for ConsumingIter<A> {}

// DiffIter

pub struct DiffIter<'a, A: 'a> {
    old_stack: Vec<IterItem<'a, A>>,
    new_stack: Vec<IterItem<'a, A>>,
}

#[derive(PartialEq, Eq)]
pub enum DiffItem<'a, A: 'a> {
    Add(&'a A),
    Update { old: &'a A, new: &'a A },
    Remove(&'a A),
}

impl<'a, A: 'a> DiffIter<'a, A> {
    pub fn new(old: &'a Node<A>, new: &'a Node<A>) -> Self {
        DiffIter {
            old_stack: if old.keys.is_empty() {
                Vec::new()
            } else {
                vec![IterItem::Consider(old)]
            },
            new_stack: if new.keys.is_empty() {
                Vec::new()
            } else {
                vec![IterItem::Consider(new)]
            },
        }
    }
}

impl<'a, A> Iterator for DiffIter<'a, A>
where
    A: 'a + BTreeValue + PartialEq,
{
    type Item = DiffItem<'a, A>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.old_stack.pop(), self.new_stack.pop()) {
                (None, None) => return None,
                (None, Some(new)) => match new {
                    IterItem::Consider(new) => Iter::push(&mut self.new_stack, &new),
                    IterItem::Yield(new) => return Some(DiffItem::Add(new)),
                },
                (Some(old), None) => match old {
                    IterItem::Consider(old) => Iter::push(&mut self.old_stack, &old),
                    IterItem::Yield(old) => return Some(DiffItem::Remove(old)),
                },
                (Some(old), Some(new)) => match (old, new) {
                    (IterItem::Consider(old), IterItem::Consider(new)) => {
                        match old.keys[0].cmp_values(&new.keys[0]) {
                            Ordering::Less => {
                                Iter::push(&mut self.old_stack, &old);
                                self.new_stack.push(IterItem::Consider(new));
                            }
                            Ordering::Greater => {
                                self.old_stack.push(IterItem::Consider(old));
                                Iter::push(&mut self.new_stack, &new);
                            }
                            Ordering::Equal => {
                                Iter::push(&mut self.old_stack, &old);
                                Iter::push(&mut self.new_stack, &new);
                            }
                        }
                    }
                    (IterItem::Consider(old), IterItem::Yield(new)) => {
                        Iter::push(&mut self.old_stack, &old);
                        self.new_stack.push(IterItem::Yield(new));
                    }
                    (IterItem::Yield(old), IterItem::Consider(new)) => {
                        self.old_stack.push(IterItem::Yield(old));
                        Iter::push(&mut self.new_stack, &new);
                    }
                    (IterItem::Yield(old), IterItem::Yield(new)) => match old.cmp_values(&new) {
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
