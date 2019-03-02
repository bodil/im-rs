// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::iter::FusedIterator;
use std::mem::replace;
use std::ops::Range;

use crate::nodes::chunk::{Chunk, CHUNK_SIZE};
use crate::util::{
    clone_ref, Ref,
    Side::{self, Left, Right},
};

use self::Entry::*;

pub const NODE_SIZE: usize = CHUNK_SIZE;

#[derive(Debug)]
enum Size {
    Size(usize),
    Table(Ref<Chunk<usize>>),
}

impl Clone for Size {
    fn clone(&self) -> Self {
        match *self {
            Size::Size(size) => Size::Size(size),
            Size::Table(ref table) => Size::Table(table.clone()),
        }
    }
}

impl Size {
    fn size(&self) -> usize {
        match self {
            Size::Size(s) => *s,
            Size::Table(sizes) => sizes.iter().sum(),
        }
    }

    fn is_size(&self) -> bool {
        match self {
            Size::Size(_) => true,
            Size::Table(_) => false,
        }
    }

    fn table_from_size(level: usize, size: usize) -> Self {
        let mut chunk = Chunk::new();
        let mut remaining = size;
        let child_size = NODE_SIZE.pow(level as u32);
        while remaining > child_size {
            let next_value = chunk.last().unwrap_or(&0) + child_size;
            chunk.push_back(next_value);
            remaining -= child_size;
        }
        if remaining > 0 {
            let next_value = chunk.last().unwrap_or(&0) + remaining;
            chunk.push_back(next_value);
        }
        Size::Table(Ref::new(chunk))
    }

    fn push(&mut self, side: Side, value: usize) {
        match self {
            Size::Size(ref mut size) => *size += value,
            Size::Table(ref mut size_ref) => {
                let size_table = Ref::make_mut(size_ref);
                debug_assert!(size_table.len() < NODE_SIZE);
                match side {
                    Left => {
                        for entry in size_table.iter_mut() {
                            *entry += value;
                        }
                        size_table.push_front(value);
                    }
                    Right => {
                        let prev = *(size_table.last().unwrap_or(&0));
                        size_table.push_back(value + prev);
                    }
                }
            }
        }
    }

    fn pop(&mut self, side: Side, value: usize) {
        match self {
            Size::Size(ref mut size) => *size -= value,
            Size::Table(ref mut size_ref) => {
                let size_table = Ref::make_mut(size_ref);
                match side {
                    Left => {
                        debug_assert_eq!(value, size_table.pop_front());
                        for entry in size_table.iter_mut() {
                            *entry -= value;
                        }
                    }
                    Right => {
                        let pop = size_table.pop_back();
                        let last = size_table.last().unwrap_or(&0);
                        debug_assert_eq!(value, pop - last);
                    }
                }
            }
        }
    }

    fn update(&mut self, index: usize, value: isize) {
        match self {
            Size::Size(ref mut size) => *size = (*size as isize + value) as usize,
            Size::Table(ref mut size_ref) => {
                let size_table = Ref::make_mut(size_ref);
                for entry in size_table.iter_mut().skip(index) {
                    *entry = (*entry as isize + value) as usize;
                }
            }
        }
    }
}

pub enum PushResult<A> {
    Full(A),
    Done,
}

pub enum PopResult<A> {
    Done(A),
    Drained(A),
    Empty,
}

pub enum SplitResult {
    Dropped(usize),
    OutOfBounds,
}

// Invariants: Nodes only at level > 0, Values/Empty only at level = 0
enum Entry<A> {
    Nodes(Size, Ref<Chunk<Ref<Node<A>>>>),
    Values(Ref<Chunk<A>>),
    Empty,
}

impl<A: Clone> Clone for Entry<A> {
    fn clone(&self) -> Self {
        match *self {
            Nodes(ref size, ref nodes) => Nodes(size.clone(), nodes.clone()),
            Values(ref values) => Values(values.clone()),
            Empty => Empty,
        }
    }
}

impl<A: Clone> Entry<A> {
    fn len(&self) -> usize {
        match self {
            Nodes(_, ref nodes) => nodes.len(),
            Values(ref values) => values.len(),
            Empty => 0,
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Nodes(_, ref nodes) => nodes.is_empty(),
            Values(ref values) => values.is_empty(),
            Empty => true,
        }
    }

    fn is_full(&self) -> bool {
        match self {
            Nodes(_, ref nodes) => nodes.is_full(),
            Values(ref values) => values.is_full(),
            Empty => false,
        }
    }

    fn unwrap_values(&self) -> &Chunk<A> {
        match self {
            Values(ref values) => values,
            _ => panic!("rrb::Entry::unwrap_values: expected values, found nodes"),
        }
    }

    fn unwrap_nodes(&self) -> &Chunk<Ref<Node<A>>> {
        match self {
            Nodes(_, ref nodes) => nodes,
            _ => panic!("rrb::Entry::unwrap_nodes: expected nodes, found values"),
        }
    }

    fn unwrap_values_mut(&mut self) -> &mut Chunk<A> {
        match self {
            Values(ref mut values) => Ref::make_mut(values),
            _ => panic!("rrb::Entry::unwrap_values_mut: expected values, found nodes"),
        }
    }

    fn unwrap_nodes_mut(&mut self) -> &mut Chunk<Ref<Node<A>>> {
        match self {
            Nodes(_, ref mut nodes) => Ref::make_mut(nodes),
            _ => panic!("rrb::Entry::unwrap_nodes_mut: expected nodes, found values"),
        }
    }

    fn values(self) -> Chunk<A> {
        match self {
            Values(values) => clone_ref(values),
            _ => panic!("rrb::Entry::values: expected values, found nodes"),
        }
    }

    fn nodes(self) -> Chunk<Ref<Node<A>>> {
        match self {
            Nodes(_, nodes) => clone_ref(nodes),
            _ => panic!("rrb::Entry::nodes: expected nodes, found values"),
        }
    }

    fn is_empty_node(&self) -> bool {
        match self {
            Empty => true,
            _ => false,
        }
    }
}

// Node

pub struct Node<A> {
    children: Entry<A>,
}

impl<A: Clone> Clone for Node<A> {
    fn clone(&self) -> Self {
        Node {
            children: self.children.clone(),
        }
    }
}

impl<A: Clone> Default for Node<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Clone> Node<A> {
    pub fn new() -> Self {
        Node { children: Empty }
    }

    pub fn parent(level: usize, children: Chunk<Ref<Self>>) -> Self {
        let size = {
            let mut size = Size::Size(0);
            let mut it = children.iter().peekable();
            loop {
                match it.next() {
                    None => break,
                    Some(child) => {
                        if size.is_size()
                            && !child.is_completely_dense(level - 1)
                            && it.peek().is_some()
                        {
                            size = Size::table_from_size(level, size.size());
                        }
                        size.push(Right, child.len())
                    }
                }
            }
            size
        };
        Node {
            children: Nodes(size, Ref::from(children)),
        }
    }

    pub fn clear_node(&mut self) {
        self.children = Empty;
    }

    pub fn from_chunk(level: usize, chunk: Ref<Chunk<A>>) -> Self {
        let node = Node {
            children: Values(chunk),
        };
        node.elevate(level)
    }

    pub fn single_parent(node: Ref<Self>) -> Self {
        let size = if node.is_dense() {
            Size::Size(node.len())
        } else {
            let size_table = Chunk::unit(node.len());
            Size::Table(Ref::from(size_table))
        };
        let children = Chunk::unit(node);
        Node {
            children: Nodes(size, Ref::from(children)),
        }
    }

    pub fn join_dense(left: Ref<Self>, right: Ref<Self>) -> Self {
        let left_len = left.len();
        let right_len = right.len();
        Node {
            children: {
                let children = Chunk::pair(left, right);
                Nodes(Size::Size(left_len + right_len), Ref::from(children))
            },
        }
    }

    pub fn elevate(self, level_increment: usize) -> Self {
        if level_increment > 0 {
            Self::single_parent(Ref::from(self.elevate(level_increment - 1)))
        } else {
            self
        }
    }

    pub fn join_branches(self, right: Self, level: usize) -> Self {
        let left_len = self.len();
        let right_len = right.len();
        let size = if self.is_completely_dense(level) && right.is_dense() {
            Size::Size(left_len + right_len)
        } else {
            let size_table = Chunk::pair(left_len, left_len + right_len);
            Size::Table(Ref::from(size_table))
        };
        Node {
            children: {
                let children = Chunk::pair(Ref::from(self), Ref::from(right));
                Nodes(size, Ref::from(children))
            },
        }
    }

    pub fn len(&self) -> usize {
        match self.children {
            Entry::Nodes(Size::Size(size), _) => size,
            Entry::Nodes(Size::Table(ref size_table), _) => *(size_table.last().unwrap_or(&0)),
            Entry::Values(ref values) => values.len(),
            Entry::Empty => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    pub fn is_single(&self) -> bool {
        self.children.len() == 1
    }

    pub fn is_full(&self) -> bool {
        self.children.is_full()
    }

    pub fn first_child(&self) -> &Ref<Self> {
        self.children.unwrap_nodes().first().unwrap()
    }

    /// True if the node is dense and so doesn't have a size table
    fn is_dense(&self) -> bool {
        match self.children {
            Entry::Nodes(Size::Table(_), _) => false,
            _ => true,
        }
    }

    /// True if the node and its children are dense and at capacity
    // TODO can use this technique to quickly test if a Size::Table
    // should be converted back to a Size::Size
    fn is_completely_dense(&self, level: usize) -> bool {
        // Size of a full node is NODE_SIZE at level 0, NODE_SIZE² at
        // level 1, etc.
        self.size() == NODE_SIZE.pow(level as u32 + 1)
    }

    #[inline]
    fn size(&self) -> usize {
        match self.children {
            Entry::Nodes(ref size, _) => size.size(),
            Entry::Values(ref values) => values.len(),
            Entry::Empty => 0,
        }
    }

    #[inline]
    fn push_size(&mut self, side: Side, value: usize) {
        if let Entry::Nodes(ref mut size, _) = self.children {
            size.push(side, value)
        }
    }

    #[inline]
    fn pop_size(&mut self, side: Side, value: usize) {
        if let Entry::Nodes(ref mut size, _) = self.children {
            size.pop(side, value)
        }
    }

    #[inline]
    fn update_size(&mut self, index: usize, value: isize) {
        if let Entry::Nodes(ref mut size, _) = self.children {
            size.update(index, value)
        }
    }

    fn size_up_to(&self, level: usize, index: usize) -> usize {
        if let Entry::Nodes(ref size, _) = self.children {
            if index == 0 {
                0
            } else {
                match size {
                    Size::Table(ref size_table) => size_table[index - 1],
                    Size::Size(_) => index * NODE_SIZE.pow(level as u32),
                }
            }
        } else {
            index
        }
    }

    fn index_in(&self, level: usize, index: usize) -> Option<usize> {
        let mut target_idx = index / NODE_SIZE.pow(level as u32);
        if target_idx >= self.children.len() {
            return None;
        }
        if let Entry::Nodes(Size::Table(ref size_table), _) = self.children {
            if size_table[target_idx] <= index {
                target_idx += 1;
                if target_idx >= size_table.len() {
                    return None;
                }
            }
        }
        Some(target_idx)
    }

    pub fn index(&self, level: usize, index: usize) -> &A {
        if level == 0 {
            &self.children.unwrap_values()[index]
        } else {
            let target_idx = self.index_in(level, index).unwrap();
            self.children.unwrap_nodes()[target_idx]
                .index(level - 1, index - self.size_up_to(level, target_idx))
        }
    }

    pub fn index_mut(&mut self, level: usize, index: usize) -> &mut A {
        if level == 0 {
            &mut self.children.unwrap_values_mut()[index]
        } else {
            let target_idx = self.index_in(level, index).unwrap();
            let offset = index - self.size_up_to(level, target_idx);
            let child = Ref::make_mut(&mut self.children.unwrap_nodes_mut()[target_idx]);
            child.index_mut(level - 1, offset)
        }
    }

    pub fn lookup_chunk(
        &self,
        level: usize,
        base: usize,
        index: usize,
    ) -> (Range<usize>, *const Chunk<A>) {
        if level == 0 {
            (
                base..(base + self.children.len()),
                self.children.unwrap_values() as *const Chunk<A>,
            )
        } else {
            let target_idx = self.index_in(level, index).unwrap();
            let offset = self.size_up_to(level, target_idx);
            let child_base = base + offset;
            let children = self.children.unwrap_nodes();
            let child = &*children[target_idx];
            child.lookup_chunk(level - 1, child_base, index - offset)
        }
    }

    pub fn lookup_chunk_mut(
        &mut self,
        level: usize,
        base: usize,
        index: usize,
    ) -> (Range<usize>, *mut Chunk<A>) {
        if level == 0 {
            (
                base..(base + self.children.len()),
                self.children.unwrap_values_mut() as *mut Chunk<A>,
            )
        } else {
            let target_idx = self.index_in(level, index).unwrap();
            let offset = self.size_up_to(level, target_idx);
            let child_base = base + offset;
            let children = self.children.unwrap_nodes_mut();
            let child = Ref::make_mut(&mut children[target_idx]);
            child.lookup_chunk_mut(level - 1, child_base, index - offset)
        }
    }

    fn push_child_node(&mut self, side: Side, child: Ref<Node<A>>) {
        let children = self.children.unwrap_nodes_mut();
        match side {
            Left => children.push_front(child),
            Right => children.push_back(child),
        }
    }

    fn pop_child_node(&mut self, side: Side) -> Ref<Node<A>> {
        let children = self.children.unwrap_nodes_mut();
        match side {
            Left => children.pop_front(),
            Right => children.pop_back(),
        }
    }

    pub fn push_chunk(
        &mut self,
        level: usize,
        side: Side,
        mut chunk: Ref<Chunk<A>>,
    ) -> PushResult<Ref<Chunk<A>>> {
        if chunk.is_empty() {
            return PushResult::Done;
        }
        let is_full = self.is_full();
        if level == 0 {
            if self.children.is_empty_node() {
                self.push_size(side, chunk.len());
                self.children = Values(chunk);
                PushResult::Done
            } else {
                let values = self.children.unwrap_values_mut();
                if values.len() + chunk.len() <= NODE_SIZE {
                    let chunk = Ref::make_mut(&mut chunk);
                    match side {
                        Side::Left => {
                            chunk.append(values);
                            values.append(chunk);
                        }
                        Side::Right => values.append(chunk),
                    }
                    PushResult::Done
                } else {
                    PushResult::Full(chunk)
                }
            }
        } else if level == 1 {
            // If rightmost existing node has any room, merge as much as
            // possible over from the new node.
            match side {
                Side::Right => {
                    if let Entry::Nodes(ref mut size, ref mut children) = self.children {
                        let rightmost = Ref::make_mut(Ref::make_mut(children).last_mut().unwrap());
                        let old_size = rightmost.len();
                        let chunk = Ref::make_mut(&mut chunk);
                        let values = rightmost.children.unwrap_values_mut();
                        let to_drain = chunk.len().min(NODE_SIZE - values.len());
                        values.drain_from_front(chunk, to_drain);
                        size.pop(Side::Right, old_size);
                        size.push(Side::Right, values.len());
                    }
                }
                Side::Left => {
                    if let Entry::Nodes(ref mut size, ref mut children) = self.children {
                        let leftmost = Ref::make_mut(Ref::make_mut(children).first_mut().unwrap());
                        let old_size = leftmost.len();
                        let chunk = Ref::make_mut(&mut chunk);
                        let values = leftmost.children.unwrap_values_mut();
                        let to_drain = chunk.len().min(NODE_SIZE - values.len());
                        values.drain_from_back(chunk, to_drain);
                        size.pop(Side::Left, old_size);
                        size.push(Side::Left, values.len());
                    }
                }
            }
            if is_full {
                PushResult::Full(chunk)
            } else {
                self.push_size(side, chunk.len());
                self.push_child_node(side, Ref::new(Node::from_chunk(0, chunk)));
                PushResult::Done
            }
        } else {
            let chunk_size = chunk.len();
            let index = match side {
                Right => self.children.len() - 1,
                Left => 0,
            };
            let new_child = {
                let children = self.children.unwrap_nodes_mut();
                let child = Ref::make_mut(&mut children[index]);
                match child.push_chunk(level - 1, side, chunk) {
                    PushResult::Done => None,
                    PushResult::Full(chunk) => {
                        if is_full {
                            return PushResult::Full(chunk);
                        } else {
                            Some(Node::from_chunk(level - 1, chunk))
                        }
                    }
                }
            };
            match new_child {
                None => {
                    self.update_size(index, chunk_size as isize);
                    PushResult::Done
                }
                Some(child) => {
                    if side == Left && chunk_size < NODE_SIZE {
                        if let Entry::Nodes(ref mut size, _) = self.children {
                            if let Size::Size(value) = *size {
                                *size = Size::table_from_size(level, value);
                            }
                        }
                    }
                    self.push_size(side, child.len());
                    self.push_child_node(side, Ref::from(child));
                    PushResult::Done
                }
            }
        }
    }

    pub fn pop_chunk(&mut self, level: usize, side: Side) -> PopResult<Ref<Chunk<A>>> {
        if self.is_empty() {
            return PopResult::Empty;
        }
        if level == 0 {
            // should only get here if the tree is just one leaf node
            match replace(&mut self.children, Empty) {
                Values(chunk) => PopResult::Drained(chunk),
                Empty => panic!("rrb::Node::pop_chunk: non-empty tree with Empty leaf"),
                Nodes(_, _) => panic!("rrb::Node::pop_chunk: branch node at leaf"),
            }
        } else if level == 1 {
            let child_node = self.pop_child_node(side);
            self.pop_size(side, child_node.len());
            let chunk = match child_node.children {
                Values(ref chunk) => chunk.clone(),
                Empty => panic!("rrb::Node::pop_chunk: non-empty tree with Empty leaf"),
                Nodes(_, _) => panic!("rrb::Node::pop_chunk: branch node at leaf"),
            };
            if self.is_empty() {
                PopResult::Drained(chunk)
            } else {
                PopResult::Done(chunk)
            }
        } else {
            let index = match side {
                Right => self.children.len() - 1,
                Left => 0,
            };
            let mut drained = false;
            let chunk = {
                let children = self.children.unwrap_nodes_mut();
                let child = Ref::make_mut(&mut children[index]);
                match child.pop_chunk(level - 1, side) {
                    PopResult::Empty => return PopResult::Empty,
                    PopResult::Done(chunk) => chunk,
                    PopResult::Drained(chunk) => {
                        drained = true;
                        chunk
                    }
                }
            };
            if drained {
                self.pop_size(side, chunk.len());
                self.pop_child_node(side);
                if self.is_empty() {
                    PopResult::Drained(chunk)
                } else {
                    PopResult::Done(chunk)
                }
            } else {
                self.update_size(index, -(chunk.len() as isize));
                PopResult::Done(chunk)
            }
        }
    }

    pub fn split(&mut self, level: usize, drop_side: Side, index: usize) -> SplitResult {
        if index == 0 && drop_side == Side::Left {
            return SplitResult::Dropped(0);
        }
        let mut dropped;
        if level == 0 {
            let len = self.children.len();
            if index >= len {
                return SplitResult::OutOfBounds;
            }
            let children = self.children.unwrap_values_mut();
            match drop_side {
                Side::Left => children.drop_left(index),
                Side::Right => children.drop_right(index),
            }
            SplitResult::Dropped(match drop_side {
                Left => index,
                Right => len - index,
            })
        } else if let Some(target_idx) = self.index_in(level, index) {
            let size_up_to = self.size_up_to(level, target_idx);
            let (size, children) =
                if let Entry::Nodes(ref mut size, ref mut children) = self.children {
                    (size, Ref::make_mut(children))
                } else {
                    unreachable!()
                };
            let child_gone = 0 == {
                let child_node = Ref::make_mut(&mut children[target_idx]);
                match child_node.split(level - 1, drop_side, index - size_up_to) {
                    SplitResult::OutOfBounds => return SplitResult::OutOfBounds,
                    SplitResult::Dropped(amount) => dropped = amount,
                }
                child_node.len()
            };
            match drop_side {
                Left => {
                    let mut drop_from = target_idx;
                    if child_gone {
                        drop_from += 1;
                    }
                    children.drop_left(drop_from);
                    if let Size::Size(value) = *size {
                        *size = Size::table_from_size(level, value);
                    }
                    let size_table = if let Size::Table(ref mut size_ref) = size {
                        Ref::make_mut(size_ref)
                    } else {
                        unreachable!()
                    };
                    let dropped_size = if target_idx > 0 {
                        size_table[target_idx - 1]
                    } else {
                        0
                    };
                    dropped += dropped_size;
                    size_table.drop_left(drop_from);
                    for i in size_table.iter_mut() {
                        *i -= dropped;
                    }
                }
                Right => {
                    let at_last = target_idx == children.len() - 1;
                    let mut drop_from = target_idx + 1;
                    if child_gone {
                        drop_from -= 1;
                    }
                    if drop_from < children.len() {
                        children.drop_right(drop_from);
                    }
                    match size {
                        Size::Size(ref mut size) if at_last => {
                            *size -= dropped;
                        }
                        Size::Size(ref mut size) => {
                            let size_per_child = NODE_SIZE.pow(level as u32);
                            let remainder = (target_idx + 1) * size_per_child;
                            let new_size = remainder - dropped;
                            dropped = *size - new_size;
                            *size = new_size;
                        }
                        Size::Table(ref mut size_ref) => {
                            let size_table = Ref::make_mut(size_ref);
                            let dropped_size =
                                size_table[size_table.len() - 1] - size_table[target_idx];
                            if drop_from < size_table.len() {
                                size_table.drop_right(drop_from);
                            }
                            if !child_gone {
                                size_table[target_idx] -= dropped;
                            }
                            dropped += dropped_size;
                        }
                    }
                }
            }
            SplitResult::Dropped(dropped)
        } else {
            SplitResult::OutOfBounds
        }
    }

    fn merge_leaves(mut left: Ref<Self>, mut right: Ref<Self>) -> Self {
        if left.children.is_empty_node() {
            // Left is empty, just use right
            Self::single_parent(right)
        } else if right.children.is_empty_node() {
            // Right is empty, just use left
            Self::single_parent(left)
        } else {
            {
                let left_node = Ref::make_mut(&mut left);
                let right_node = Ref::make_mut(&mut right);
                let left_vals = left_node.children.unwrap_values_mut();
                let left_len = left_vals.len();
                let right_vals = right_node.children.unwrap_values_mut();
                let right_len = right_vals.len();
                if left_len + right_len <= NODE_SIZE {
                    left_vals.append(right_vals);
                } else {
                    let count = right_len.min(NODE_SIZE - left_len);
                    left_vals.drain_from_front(right_vals, count);
                }
            }
            if right.is_empty() {
                Self::single_parent(left)
            } else {
                Self::join_dense(left, right)
            }
        }
    }

    fn merge_rebalance(level: usize, left: Ref<Self>, middle: Self, right: Ref<Self>) -> Self {
        let left_nodes = clone_ref(left).children.nodes().into_iter();
        let middle_nodes = middle.children.nodes().into_iter();
        let right_nodes = clone_ref(right).children.nodes().into_iter();
        let mut subtree_still_balanced = true;
        let mut next_leaf = Chunk::new();
        let mut next_node = Chunk::new();
        let mut next_subtree = Chunk::new();
        let mut root = Chunk::new();

        for subtree in left_nodes.chain(middle_nodes).chain(right_nodes) {
            if subtree.is_empty() {
                continue;
            }
            if subtree.is_completely_dense(level) && subtree_still_balanced {
                root.push_back(subtree);
                continue;
            }
            subtree_still_balanced = false;

            let child = clone_ref(subtree);
            if level == 1 {
                for value in child.children.values() {
                    next_leaf.push_back(value);
                    if next_leaf.is_full() {
                        let new_node = Node::from_chunk(0, Ref::from(next_leaf));
                        next_subtree.push_back(Ref::from(new_node));
                        next_leaf = Chunk::new();
                        if next_subtree.is_full() {
                            let new_subtree = Node::parent(level, next_subtree);
                            root.push_back(Ref::from(new_subtree));
                            next_subtree = Chunk::new();
                        }
                    }
                }
            } else {
                for node in child.children.nodes() {
                    next_node.push_back(node);
                    if next_node.is_full() {
                        let new_node = Node::parent(level - 1, next_node);
                        next_subtree.push_back(Ref::from(new_node));
                        next_node = Chunk::new();
                        if next_subtree.is_full() {
                            let new_subtree = Node::parent(level, next_subtree);
                            root.push_back(Ref::from(new_subtree));
                            next_subtree = Chunk::new();
                        }
                    }
                }
            }
        }
        if !next_leaf.is_empty() {
            let new_node = Node::from_chunk(0, Ref::from(next_leaf));
            next_subtree.push_back(Ref::from(new_node));
        }
        if !next_node.is_empty() {
            let new_node = Node::parent(level - 1, next_node);
            next_subtree.push_back(Ref::from(new_node));
        }
        if !next_subtree.is_empty() {
            let new_subtree = Node::parent(level, next_subtree);
            root.push_back(Ref::from(new_subtree));
        }
        Node::parent(level + 1, root)
    }

    pub fn merge(mut left: Ref<Self>, mut right: Ref<Self>, level: usize) -> Self {
        if level == 0 {
            Self::merge_leaves(left, right)
        } else {
            let merged = {
                if level == 1 {
                    // We're going to rebalance all the leaves anyway, there's
                    // no need for a middle at level 1
                    Node::parent(0, Chunk::new())
                } else {
                    let left_node = Ref::make_mut(&mut left);
                    let right_node = Ref::make_mut(&mut right);
                    let left_last =
                        if let Entry::Nodes(ref mut size, ref mut children) = left_node.children {
                            let node = Ref::make_mut(children).pop_back();
                            size.pop(Side::Right, node.len());
                            node
                        } else {
                            panic!("expected nodes, found entries or empty");
                        };
                    let right_first =
                        if let Entry::Nodes(ref mut size, ref mut children) = right_node.children {
                            let node = Ref::make_mut(children).pop_front();
                            size.pop(Side::Left, node.len());
                            node
                        } else {
                            panic!("expected nodes, found entries or empty");
                        };
                    Self::merge(left_last, right_first, level - 1)
                }
            };
            Self::merge_rebalance(level, left, merged, right)
        }
    }

    // pub fn print<W>(&self, f: &mut W, indent: usize, level: usize) -> Result<(), fmt::Error>
    // where
    //     W: fmt::Write,
    //     A: fmt::Debug,
    // {
    //     print_indent(f, indent)?;
    //     if level == 0 {
    //         if self.children.is_empty_node() {
    //             writeln!(f, "Leaf: EMPTY")
    //         } else {
    //             writeln!(f, "Leaf: {:?}", self.children.unwrap_values())
    //         }
    //     } else {
    //         match &self.children {
    //             Entry::Nodes(size, children) => {
    //                 writeln!(f, "Node level {} size_table {:?}", level, size)?;
    //                 for child in children.iter() {
    //                     child.print(f, indent + 4, level - 1)?;
    //                 }
    //                 Ok(())
    //             }
    //             _ => unreachable!(),
    //         }
    //     }
    // }
}

// fn print_indent<W>(f: &mut W, indent: usize) -> Result<(), fmt::Error>
// where
//     W: fmt::Write,
// {
//     for _i in 0..indent {
//         write!(f, " ")?;
//     }
//     Ok(())
// }

// Consuming iterator

pub struct ConsumingIter<A> {
    root: Node<A>,
    level: usize,
    front_chunk: Option<Chunk<A>>,
    back_chunk: Option<Chunk<A>>,
    remaining: usize,
}

impl<A: Clone> ConsumingIter<A> {
    pub fn new(root: Node<A>, level: usize) -> Self {
        ConsumingIter {
            remaining: root.len(),
            root,
            level,
            front_chunk: None,
            back_chunk: None,
        }
    }
}

impl<A: Clone> Iterator for ConsumingIter<A> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        if let Some(ref mut chunk) = self.front_chunk {
            if !chunk.is_empty() {
                self.remaining -= 1;
                return Some(chunk.pop_front());
            }
        }
        match self.root.pop_chunk(self.level, Side::Left) {
            PopResult::Done(chunk) => self.front_chunk = Some(clone_ref(chunk)),
            PopResult::Drained(chunk) => self.front_chunk = Some(clone_ref(chunk)),
            PopResult::Empty => {
                if let Some(ref mut chunk) = self.back_chunk {
                    if !chunk.is_empty() {
                        self.remaining -= 1;
                        return Some(chunk.pop_front());
                    } else {
                        return None;
                    }
                }
            }
        }
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<A: Clone> DoubleEndedIterator for ConsumingIter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        if let Some(ref mut chunk) = self.back_chunk {
            if !chunk.is_empty() {
                self.remaining -= 1;
                return Some(chunk.pop_back());
            }
        }
        match self.root.pop_chunk(self.level, Side::Left) {
            PopResult::Done(chunk) => self.front_chunk = Some(clone_ref(chunk)),
            PopResult::Drained(chunk) => self.front_chunk = Some(clone_ref(chunk)),
            PopResult::Empty => {
                if let Some(ref mut chunk) = self.front_chunk {
                    if !chunk.is_empty() {
                        self.remaining -= 1;
                        return Some(chunk.pop_back());
                    } else {
                        return None;
                    }
                }
            }
        }
        self.next()
    }
}

impl<A: Clone> ExactSizeIterator for ConsumingIter<A> {}

impl<A: Clone> FusedIterator for ConsumingIter<A> {}
