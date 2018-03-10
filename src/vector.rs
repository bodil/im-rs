use std::sync::Arc;
use std::iter::{self, FromIterator};
use std::mem;
use std::fmt::{Debug, Error, Formatter};

use shared::Shared;
use bits::{HASH_BITS, HASH_MASK, HASH_SIZE};

enum Entry<A> {
    Node(Arc<Node<A>>),
    Value(Arc<A>),
    Empty,
}

impl<A> Entry<A> {
    fn into_val(self) -> Arc<A> {
        match self {
            Entry::Value(v) => v,
            _ => panic!("Entry::into_val: tried to into_val a non-value"),
        }
    }

    fn unwrap_val(&self) -> Arc<A> {
        match *self {
            Entry::Value(ref v) => v.clone(),
            _ => panic!("Entry::unwrap_val: tried to unwrap_val a non-value"),
        }
    }

    fn unwrap_node(&self) -> Arc<Node<A>> {
        match *self {
            Entry::Node(ref n) => n.clone(),
            _ => panic!("Entry::unwrap_node: tried to unwrap_node a non-node"),
        }
    }

    fn is_empty(&self) -> bool {
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

struct Node<A> {
    children: Vec<Entry<A>>,
}

fn make_empty<A>() -> Vec<Entry<A>> {
    iter::repeat(())
        .map(|_| Entry::Empty)
        .take(HASH_SIZE)
        .collect()
}

impl<A> Node<A> {
    fn new() -> Self {
        Node {
            children: make_empty(),
        }
    }

    fn from_vec(mut children: Vec<Entry<A>>) -> Self {
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

pub struct Vector<A> {
    count: usize,
    shift: usize,
    root: Arc<Node<A>>,
    tail: Arc<Vec<Entry<A>>>,
}

impl<A> Vector<A> {
    pub fn new() -> Self {
        Vector {
            count: 0,
            shift: HASH_BITS,
            root: Arc::new(Node::new()),
            tail: Arc::new(Vec::with_capacity(HASH_SIZE)),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn iter(&self) -> Iter<A> {
        Iter::new(self)
    }

    pub fn get(&self, index: usize) -> Option<Arc<A>> {
        if index >= self.len() {
            return None;
        }
        let entry = match self.node_for(index) {
            None => self.tail.get(index & HASH_MASK as usize).cloned(),
            Some(node) => node.children.get(index & HASH_MASK as usize).cloned(),
        };
        match entry {
            Some(Entry::Value(ref value)) => Some(value.clone()),
            Some(Entry::Node(_)) => {
                panic!("Vector::get: encountered node where value was expected")
            }
            Some(Entry::Empty) => panic!("Vector::get: encountered null, expected value"),
            None => panic!("Vector::get: unhandled index out of bounds situation!"),
        }
    }

    pub fn set<RA>(&self, index: usize, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        assert!(index < self.count);
        if index >= self.tailoff() {
            let mut tail = (*self.tail).clone();
            tail[index & HASH_MASK as usize] = Entry::Value(value.shared());
            self.update_tail(tail)
        } else {
            self.update_root(set_in(self.shift, &self.root, index, value.shared()))
        }
    }

    pub fn set_mut<RA>(&mut self, index: usize, value: RA)
    where
        RA: Shared<A>,
    {
        assert!(index < self.count);
        if index >= self.tailoff() {
            let tail = Arc::make_mut(&mut self.tail);
            tail[index & HASH_MASK as usize] = Entry::Value(value.shared());
        } else {
            let root = Arc::make_mut(&mut self.root);
            set_in_mut(self.shift, root, index, value.shared())
        }
    }

    pub fn push<RA>(&self, value: RA) -> Self
    where
        RA: Shared<A>,
    {
        if self.len() - self.tailoff() < HASH_SIZE {
            let mut tail = (*self.tail).clone();
            tail.push(Entry::Value(value.shared()));
            Vector {
                count: self.count + 1,
                shift: self.shift,
                root: self.root.clone(),
                tail: Arc::new(tail),
            }
        } else {
            let mut shift = self.shift;
            let tail_node = Node::from_vec((*self.tail).clone());
            let new_root = if (self.len() >> HASH_BITS) > (1 << self.shift) {
                let mut node = Node::new();
                node.children[0] = Entry::Node(self.root.clone());
                node.children[1] = Entry::Node(Arc::new(new_path(self.shift, tail_node)));
                shift += HASH_BITS;
                node
            } else {
                push_tail(self.count, self.shift, &self.root, tail_node)
            };
            Vector {
                count: self.count + 1,
                shift,
                root: Arc::new(new_root),
                tail: Arc::new(vec![Entry::Value(value.shared())]),
            }
        }
    }

    pub fn push_mut<RA>(&mut self, value: RA)
    where
        RA: Shared<A>,
    {
        if self.count - self.tailoff() < HASH_SIZE {
            let tail = Arc::make_mut(&mut self.tail);
            tail.push(Entry::Value(value.shared()));
        } else {
            let tail_node = {
                let tail = Arc::make_mut(&mut self.tail);
                Node::from_vec(mem::replace(tail, Vec::with_capacity(HASH_SIZE)))
            };
            if (self.count >> HASH_BITS) > (1 << self.shift) {
                let mut node = Node::new();
                node.children[0] = Entry::Node(self.root.clone());
                node.children[1] = Entry::Node(Arc::new(new_path(self.shift, tail_node)));
                self.shift += HASH_BITS;
                self.root = Arc::new(node);
            } else {
                let root = Arc::make_mut(&mut self.root);
                push_tail_mut(self.count, self.shift, root, tail_node)
            }
            let tail = Arc::make_mut(&mut self.tail);
            tail.push(Entry::Value(value.shared()));
        }
        self.count += 1;
    }

    pub fn pop(&self) -> Option<(Arc<A>, Self)> {
        if self.count == 0 {
            return None;
        }
        if self.count == 1 {
            return Some((self.tail[0].unwrap_val(), Vector::new()));
        }
        if self.tail.len() > 1 {
            let mut tail = (*self.tail).clone();
            let value = tail.pop().unwrap().unwrap_val();
            Some((
                value,
                Vector {
                    count: self.count - 1,
                    shift: self.shift,
                    root: self.root.clone(),
                    tail: Arc::new(tail),
                },
            ))
        } else {
            match self.node_for(self.count - 2) {
                None => panic!("Vector::pop: unexpected non-node from node_for"),
                Some(ref node) => {
                    let tail = node.children.clone();
                    let mut shift = self.shift;
                    let mut root =
                        Arc::new(pop_tail(self.count, self.shift, &self.root).unwrap_or_default());
                    if self.shift > HASH_BITS && root.children[1].is_empty() {
                        root = root.children[0].unwrap_node();
                        shift -= HASH_BITS;
                    }
                    Some((
                        self.get(self.count - 1).unwrap(),
                        Vector {
                            count: self.count - 1,
                            shift,
                            root,
                            tail: Arc::new(tail),
                        },
                    ))
                }
            }
        }
    }

    pub fn pop_mut(&mut self) -> Option<Arc<A>> {
        if self.count == 0 {
            return None;
        }
        if self.count == 1 || (self.count - 1) & HASH_MASK as usize > 0 {
            self.count -= 1;
            let tail = Arc::make_mut(&mut self.tail);
            return tail.pop().map(Entry::into_val);
        }
        let value = self.get(self.count - 1);
        match self.node_for(self.count - 2) {
            None => panic!("Vector::pop_mut: unexpected non-node from node_for"),
            Some(tail_node) => {
                let tail = tail_node.children.clone();
                let mut set_root = None;
                {
                    let mut root = Arc::make_mut(&mut self.root);
                    if pop_tail_mut(self.count, self.shift, root) {
                        if self.shift > HASH_BITS && root.children[1].is_empty() {
                            set_root = Some(root.children[0].unwrap_node());
                            self.shift -= HASH_BITS;
                        }
                    } else {
                        set_root = Some(Arc::new(Node::new()));
                    }
                }
                if let Some(root) = set_root {
                    self.root = root;
                }
                self.tail = Arc::new(tail);
            }
        }
        self.count -= 1;
        value
    }

    fn adopt(v: Vec<Entry<A>>) -> Self {
        Vector {
            count: v.len(),
            shift: HASH_BITS,
            root: Arc::new(Node::new()),
            tail: Arc::new(v),
        }
    }

    fn tailoff(&self) -> usize {
        if self.count < HASH_SIZE {
            0
        } else {
            ((self.count - 1) >> HASH_BITS) << HASH_BITS
        }
    }

    fn node_for(&self, index: usize) -> Option<Arc<Node<A>>> {
        if index >= self.tailoff() {
            None
        } else {
            let mut node = self.root.clone();
            let mut level = self.shift;
            while level > 0 {
                node = if let Some(&Entry::Node(ref child_node)) =
                    node.children.get((index >> level) & HASH_MASK as usize)
                {
                    level -= HASH_BITS;
                    child_node.clone()
                } else {
                    panic!("Vector::node_for: encountered value or null where node was expected")
                };
            }
            Some(node)
        }
    }

    fn update_root(&self, root: Node<A>) -> Self {
        Vector {
            count: self.count,
            shift: self.shift,
            root: Arc::new(root),
            tail: self.tail.clone(),
        }
    }

    fn update_tail(&self, tail: Vec<Entry<A>>) -> Self {
        Vector {
            count: self.count,
            shift: self.shift,
            root: self.root.clone(),
            tail: Arc::new(tail),
        }
    }
}

fn push_tail<A>(count: usize, level: usize, parent: &Node<A>, tail_node: Node<A>) -> Node<A> {
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

fn push_tail_mut<A>(count: usize, level: usize, parent: &mut Node<A>, tail_node: Node<A>) {
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

fn pop_tail<A>(count: usize, level: usize, node: &Node<A>) -> Option<Node<A>> {
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

fn pop_tail_mut<A>(count: usize, level: usize, node: &mut Node<A>) -> bool {
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

fn new_path<A>(level: usize, node: Node<A>) -> Node<A> {
    if level == 0 {
        node
    } else {
        let mut out = Node::new();
        out.children[0] = Entry::Node(Arc::new(new_path(level - HASH_BITS, node)));
        out
    }
}

fn set_in<A>(level: usize, node: &Node<A>, index: usize, value: Arc<A>) -> Node<A> {
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

fn set_in_mut<A>(level: usize, node: &mut Node<A>, index: usize, value: Arc<A>) {
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

// Core traits

impl<A> Clone for Vector<A> {
    fn clone(&self) -> Self {
        Vector {
            count: self.count,
            shift: self.shift,
            root: self.root.clone(),
            tail: self.tail.clone(),
        }
    }
}

impl<A> Default for Vector<A> {
    fn default() -> Self {
        Vector::new()
    }
}

impl<A: Debug> Debug for Vector<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "[")?;
        let mut it = self.iter().peekable();
        while let Some(item) = it.next() {
            write!(f, "{:?}", item)?;
            if it.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

// Conversions

impl<A, RA: Shared<A>> FromIterator<RA> for Vector<A> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        let mut v = Vec::with_capacity(HASH_SIZE);
        let mut it = iter.into_iter().peekable();
        while let Some(item) = it.next() {
            v.push(Entry::Value(item.shared()));
            if v.len() == HASH_SIZE {
                break;
            }
        }
        if v.len() == HASH_SIZE && it.peek().is_some() {
            let mut out = Vector::adopt(v);
            for item in it {
                out.push_mut(item);
            }
            out
        } else {
            Vector::adopt(v)
        }
    }
}

// Iterators

pub struct Iter<A> {
    vector: Vector<A>,
    start_node: Option<Arc<Node<A>>>,
    start_index: usize,
    start_offset: usize,
    end_node: Option<Arc<Node<A>>>,
    end_index: usize,
    end_offset: usize,
}

impl<A> Iter<A> {
    pub fn new(vector: &Vector<A>) -> Self {
        let end_index = vector.len() & !(HASH_MASK as usize);
        Iter {
            vector: vector.clone(),
            start_node: vector.node_for(0),
            end_node: vector.node_for(end_index),
            start_index: 0,
            start_offset: 0,
            end_index,
            end_offset: vector.len() - end_index,
        }
    }
}

impl<A> Iterator for Iter<A> {
    type Item = Arc<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_index + self.start_offset == self.end_index + self.end_offset {
            return None;
        }
        match self.start_node {
            None => {
                let item = self.vector.tail[self.start_offset].unwrap_val();
                self.start_offset += 1;
                return Some(item);
            }
            Some(ref node) => {
                if self.start_offset < HASH_SIZE {
                    let item = node.children[self.start_offset].unwrap_val();
                    self.start_offset += 1;
                    return Some(item);
                }
            }
        }
        self.start_offset = 0;
        self.start_index += HASH_SIZE;
        self.start_node = self.vector.node_for(self.start_index);
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.end_index + self.end_offset) - (self.start_index + self.start_offset);
        (size, Some(size))
    }
}

impl<A> DoubleEndedIterator for Iter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start_index + self.start_offset == self.end_index + self.end_offset {
            return None;
        }
        match self.end_node {
            None => {
                if self.end_offset > 0 {
                    self.end_offset -= 1;
                    let item = self.vector.tail[self.end_offset].unwrap_val();
                    return Some(item);
                }
            }
            Some(ref node) => {
                if self.end_offset > 0 {
                    self.end_offset -= 1;
                    let item = node.children[self.end_offset].unwrap_val();
                    return Some(item);
                }
            }
        }
        self.end_offset = HASH_SIZE;
        self.end_index -= HASH_SIZE;
        self.end_node = self.vector.node_for(self.end_index);
        self.next_back()
    }
}

impl<A> ExactSizeIterator for Iter<A> {}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use proptest::num::i32;
    use proptest::collection;

    #[test]
    fn double_ended_iterator() {
        let vector = Vector::<i32>::from_iter(1..6);
        let mut it = vector.iter();
        assert_eq!(Some(Arc::new(1)), it.next());
        assert_eq!(Some(Arc::new(5)), it.next_back());
        assert_eq!(Some(Arc::new(2)), it.next());
        assert_eq!(Some(Arc::new(4)), it.next_back());
        assert_eq!(Some(Arc::new(3)), it.next());
        assert_eq!(None, it.next_back());
        assert_eq!(None, it.next());
    }

    proptest! {
        #[test]
        fn push(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector = vector.push(value);
                assert_eq!(count + 1, vector.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn push_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::new();
            for (count, value) in input.iter().cloned().enumerate() {
                assert_eq!(count, vector.len());
                vector.push_mut(value);
                assert_eq!(count + 1, vector.len());
            }
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn from_iter(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn set(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(iter::repeat(0).take(input.len()));
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                vector = vector.set(index, value);
            }
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn set_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(iter::repeat(0).take(input.len()));
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                vector.set_mut(index, value);
            }
            assert_eq!(vector.len(), input.len());
            for (index, value) in input.iter().cloned().enumerate() {
                assert_eq!(Some(Arc::new(value)), vector.get(index));
            }
        }

        #[test]
        fn pop(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop() {
                    None => panic!("vector emptied unexpectedly"),
                    Some((item, next)) => {
                        vector = next;
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn pop_mut(ref input in collection::vec(i32::ANY, 0..100)) {
            let mut vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            for (index, value) in input.iter().cloned().enumerate().rev() {
                match vector.pop_mut() {
                    None => panic!("vector emptied unexpectedly"),
                    Some(item) => {
                        assert_eq!(index, vector.len());
                        assert_eq!(Arc::new(value), item);
                    }
                }
            }
            assert_eq!(0, vector.len());
        }

        #[test]
        fn iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            let mut it1 = input.iter().cloned();
            let mut it2 = vector.iter();
            loop {
                match (it1.next(), it2.next()) {
                    (None, None) => break,
                    (Some(i1), Some(i2)) => assert_eq!(i1, *i2),
                    (Some(i1), None) => panic!(format!("expected {:?} but got EOF", i1)),
                    (None, Some(i2)) => panic!(format!("expected EOF but got {:?}", *i2)),
                }
            }
        }

        #[test]
        fn reverse_iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input.iter().cloned());
            assert_eq!(input.len(), vector.len());
            let mut it1 = input.iter().cloned().rev();
            let mut it2 = vector.iter().rev();
            loop {
                match (it1.next(), it2.next()) {
                    (None, None) => break,
                    (Some(i1), Some(i2)) => assert_eq!(i1, *i2),
                    (Some(i1), None) => panic!(format!("expected {:?} but got EOF", i1)),
                    (None, Some(i2)) => panic!(format!("expected EOF but got {:?}", *i2)),
                }
            }
        }

        #[test]
        fn exact_size_iterator(ref input in collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from_iter(input);
            let mut should_be = vector.len();
            let mut it = vector.iter();
            loop {
                assert_eq!(should_be, it.len());
                match it.next() {
                    None => break,
                    Some(_) => should_be -= 1,
                }
            }
            assert_eq!(0, it.len());
        }
    }
}
