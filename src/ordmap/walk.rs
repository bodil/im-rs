// These are the nasty bits. Don't look too closely.

use std::sync::Arc;
use std::cmp::Ordering;
use std::ops::Deref;
use conslist::ConsList;

use super::OrdMap;
use super::MapNode::{Leaf, Three, Two};
use self::TreeContext::{ThreeLeft, ThreeMiddle, ThreeRight, TwoLeft, TwoRight};

// Lookup

pub fn lookup<K: Ord, V>(m: &OrdMap<K, V>, k: &K) -> Option<Arc<V>> {
    match *m.0 {
        Leaf => None,
        Two(_, ref left, ref k1, ref v, ref right) => match k.cmp(k1) {
            Ordering::Equal => Some(v.clone()),
            Ordering::Less => left.get(k),
            _ => right.get(k),
        },
        Three(_, ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => match k.cmp(k1) {
            Ordering::Equal => Some(v1.clone()),
            c1 => match (c1, k.cmp(k2)) {
                (_, Ordering::Equal) => Some(v2.clone()),
                (Ordering::Less, _) => left.get(k),
                (_, Ordering::Greater) => right.get(k),
                _ => mid.get(k),
            },
        },
    }
}

// Insertion

pub enum TreeContext<K, V> {
    TwoLeft(Arc<K>, Arc<V>, OrdMap<K, V>),
    TwoRight(OrdMap<K, V>, Arc<K>, Arc<V>),
    ThreeLeft(Arc<K>, Arc<V>, OrdMap<K, V>, Arc<K>, Arc<V>, OrdMap<K, V>),
    ThreeMiddle(OrdMap<K, V>, Arc<K>, Arc<V>, Arc<K>, Arc<V>, OrdMap<K, V>),
    ThreeRight(OrdMap<K, V>, Arc<K>, Arc<V>, OrdMap<K, V>, Arc<K>, Arc<V>),
}

// Delightfully, #[derive(Clone)] doesn't seem to be able to
// produce a working implementation of this.
impl<K, V> Clone for TreeContext<K, V> {
    fn clone(&self) -> TreeContext<K, V> {
        match self {
            &TwoLeft(ref k, ref v, ref right) => TwoLeft(k.clone(), v.clone(), right.clone()),
            &TwoRight(ref left, ref k, ref v) => TwoRight(left.clone(), k.clone(), v.clone()),
            &ThreeLeft(ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => ThreeLeft(
                k1.clone(),
                v1.clone(),
                mid.clone(),
                k2.clone(),
                v2.clone(),
                right.clone(),
            ),
            &ThreeMiddle(ref left, ref k1, ref v1, ref k2, ref v2, ref right) => ThreeMiddle(
                left.clone(),
                k1.clone(),
                v1.clone(),
                k2.clone(),
                v2.clone(),
                right.clone(),
            ),
            &ThreeRight(ref left, ref k1, ref v1, ref mid, ref k2, ref v2) => ThreeRight(
                left.clone(),
                k1.clone(),
                v1.clone(),
                mid.clone(),
                k2.clone(),
                v2.clone(),
            ),
        }
    }
}

#[derive(Clone)]
struct KickUp<K, V>(OrdMap<K, V>, Arc<K>, Arc<V>, OrdMap<K, V>);

fn from_zipper<K, V>(ctx: ConsList<TreeContext<K, V>>, tree: OrdMap<K, V>) -> OrdMap<K, V> {
    match ctx.uncons() {
        None => tree,
        Some((x, xs)) => match x.deref().clone() {
            TwoLeft(k1, v1, right) => from_zipper(xs, OrdMap::two(tree, k1, v1, right)),
            TwoRight(left, k1, v1) => from_zipper(xs, OrdMap::two(left, k1, v1, tree)),
            ThreeLeft(k1, v1, mid, k2, v2, right) => {
                from_zipper(xs, OrdMap::three(tree, k1, v1, mid, k2, v2, right))
            }
            ThreeMiddle(left, k1, v1, k2, v2, right) => {
                from_zipper(xs, OrdMap::three(left, k1, v1, tree, k2, v2, right))
            }
            ThreeRight(left, k1, v1, mid, k2, v2) => {
                from_zipper(xs, OrdMap::three(left, k1, v1, mid, k2, v2, tree))
            }
        },
    }
}

pub fn ins_down<K: Ord, V>(
    ctx: ConsList<TreeContext<K, V>>,
    k: Arc<K>,
    v: Arc<V>,
    m: OrdMap<K, V>,
) -> OrdMap<K, V> {
    match *m.0 {
        Leaf => ins_up(ctx, KickUp(OrdMap::new(), k.clone(), v.clone(), OrdMap::new())),
        Two(_, ref left, ref k1, ref v1, ref right) => match k.cmp(k1) {
            Ordering::Equal => from_zipper(
                ctx,
                OrdMap::two(left.clone(), k.clone(), v.clone(), right.clone()),
            ),
            Ordering::Less => ins_down(
                ctx.cons(TwoLeft(k1.clone(), v1.clone(), right.clone())),
                k,
                v,
                left.clone(),
            ),
            _ => ins_down(
                ctx.cons(TwoRight(left.clone(), k1.clone(), v1.clone())),
                k,
                v,
                right.clone(),
            ),
        },
        Three(_, ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => match k.cmp(k1) {
            Ordering::Equal => from_zipper(
                ctx,
                OrdMap::three(
                    left.clone(),
                    k,
                    v,
                    mid.clone(),
                    k2.clone(),
                    v2.clone(),
                    right.clone(),
                ),
            ),
            c1 => match (c1, k.cmp(k2)) {
                (_, Ordering::Equal) => from_zipper(
                    ctx,
                    OrdMap::three(
                        left.clone(),
                        k1.clone(),
                        v1.clone(),
                        mid.clone(),
                        k,
                        v,
                        right.clone(),
                    ),
                ),
                (Ordering::Less, _) => ins_down(
                    ctx.cons(ThreeLeft(
                        k1.clone(),
                        v1.clone(),
                        mid.clone(),
                        k2.clone(),
                        v2.clone(),
                        right.clone(),
                    )),
                    k,
                    v,
                    left.clone(),
                ),
                (Ordering::Greater, Ordering::Less) => ins_down(
                    ctx.cons(ThreeMiddle(
                        left.clone(),
                        k1.clone(),
                        v1.clone(),
                        k2.clone(),
                        v2.clone(),
                        right.clone(),
                    )),
                    k,
                    v,
                    mid.clone(),
                ),
                _ => ins_down(
                    ctx.cons(ThreeRight(
                        left.clone(),
                        k1.clone(),
                        v1.clone(),
                        mid.clone(),
                        k2.clone(),
                        v2.clone(),
                    )),
                    k,
                    v,
                    right.clone(),
                ),
            },
        },
    }
}

fn ins_up<K, V>(ctx: ConsList<TreeContext<K, V>>, kickup: KickUp<K, V>) -> OrdMap<K, V> {
    match ctx.uncons() {
        None => match kickup {
            KickUp(left, k, v, right) => OrdMap::two(left, k, v, right),
        },
        Some((x, xs)) => match (x.deref().clone(), kickup) {
            (TwoLeft(ref k1, ref v1, ref right), KickUp(ref left, ref k, ref v, ref mid)) => {
                from_zipper(
                    xs,
                    OrdMap::three(
                        left.clone(),
                        k.clone(),
                        v.clone(),
                        mid.clone(),
                        k1.clone(),
                        v1.clone(),
                        right.clone(),
                    ),
                )
            }
            (TwoRight(ref left, ref k1, ref v1), KickUp(ref mid, ref k, ref v, ref right)) => {
                from_zipper(
                    xs,
                    OrdMap::three(
                        left.clone(),
                        k1.clone(),
                        v1.clone(),
                        mid.clone(),
                        k.clone(),
                        v.clone(),
                        right.clone(),
                    ),
                )
            }
            (
                ThreeLeft(ref k1, ref v1, ref c, ref k2, ref v2, ref d),
                KickUp(ref a, ref k, ref v, ref b),
            ) => ins_up(
                xs,
                KickUp(
                    OrdMap::two(a.clone(), k.clone(), v.clone(), b.clone()),
                    k1.clone(),
                    v1.clone(),
                    OrdMap::two(c.clone(), k2.clone(), v2.clone(), d.clone()),
                ),
            ),
            (
                ThreeMiddle(ref a, ref k1, ref v1, ref k2, ref v2, ref d),
                KickUp(ref b, ref k, ref v, ref c),
            ) => ins_up(
                xs,
                KickUp(
                    OrdMap::two(a.clone(), k1.clone(), v1.clone(), b.clone()),
                    k.clone(),
                    v.clone(),
                    OrdMap::two(c.clone(), k2.clone(), v2.clone(), d.clone()),
                ),
            ),
            (
                ThreeRight(ref a, ref k1, ref v1, ref b, ref k2, ref v2),
                KickUp(ref c, ref k, ref v, ref d),
            ) => ins_up(
                xs,
                KickUp(
                    OrdMap::two(a.clone(), k1.clone(), v1.clone(), b.clone()),
                    k2.clone(),
                    v2.clone(),
                    OrdMap::two(c.clone(), k.clone(), v.clone(), d.clone()),
                ),
            ),
        },
    }
}

// Deletion

pub fn pop_down<K: Ord, V>(
    ctx: ConsList<TreeContext<K, V>>,
    k: &K,
    m: OrdMap<K, V>,
) -> Option<(Arc<K>, Arc<V>, OrdMap<K, V>)> {
    match *m.0 {
        Leaf => None,
        Two(_, ref left, ref k1, ref v1, ref right) => match (&*right.0, k.cmp(k1)) {
            (&Leaf, Ordering::Equal) => Some((k1.clone(), v1.clone(), pop_up(ctx, OrdMap::new()))),
            (_, Ordering::Equal) => {
                let (max_key, max_val) = max_node(left.clone());
                Some((
                    k1.clone(),
                    v1.clone(),
                    remove_max_node(
                        ctx.cons(TwoLeft(max_key, max_val, right.clone())),
                        left.clone(),
                    ),
                ))
            }
            (_, Ordering::Less) => pop_down(
                ctx.cons(TwoLeft(k1.clone(), v1.clone(), right.clone())),
                k,
                left.clone(),
            ),
            (_, _) => pop_down(
                ctx.cons(TwoRight(left.clone(), k1.clone(), v1.clone())),
                k,
                right.clone(),
            ),
        },
        Three(_, ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => {
            let leaves = match (&*left.0, &*mid.0, &*right.0) {
                (&Leaf, &Leaf, &Leaf) => true,
                _ => false,
            };
            match (leaves, k.cmp(k1), k.cmp(k2)) {
                (true, Ordering::Equal, _) => Some((
                    k1.clone(),
                    v1.clone(),
                    from_zipper(
                        ctx,
                        OrdMap::two(OrdMap::new(), k2.clone(), v2.clone(), OrdMap::new()),
                    ),
                )),
                (true, _, Ordering::Equal) => Some((
                    k2.clone(),
                    v2.clone(),
                    from_zipper(
                        ctx,
                        OrdMap::two(OrdMap::new(), k1.clone(), v1.clone(), OrdMap::new()),
                    ),
                )),
                (_, Ordering::Equal, _) => {
                    let (max_key, max_val) = max_node(left.clone());
                    Some((
                        k1.clone(),
                        v1.clone(),
                        remove_max_node(
                            ctx.cons(ThreeLeft(
                                max_key,
                                max_val,
                                mid.clone(),
                                k2.clone(),
                                v2.clone(),
                                right.clone(),
                            )),
                            left.clone(),
                        ),
                    ))
                }
                (_, _, Ordering::Equal) => {
                    let (max_key, max_val) = max_node(mid.clone());
                    Some((
                        k2.clone(),
                        v2.clone(),
                        remove_max_node(
                            ctx.cons(ThreeMiddle(
                                left.clone(),
                                k1.clone(),
                                v1.clone(),
                                max_key,
                                max_val,
                                right.clone(),
                            )),
                            mid.clone(),
                        ),
                    ))
                }
                (_, Ordering::Less, _) => pop_down(
                    ctx.cons(ThreeLeft(
                        k1.clone(),
                        v1.clone(),
                        mid.clone(),
                        k2.clone(),
                        v2.clone(),
                        right.clone(),
                    )),
                    k,
                    left.clone(),
                ),
                (_, Ordering::Greater, Ordering::Less) => pop_down(
                    ctx.cons(ThreeMiddle(
                        left.clone(),
                        k1.clone(),
                        v1.clone(),
                        k2.clone(),
                        v2.clone(),
                        right.clone(),
                    )),
                    k,
                    mid.clone(),
                ),
                _ => pop_down(
                    ctx.cons(ThreeRight(
                        left.clone(),
                        k1.clone(),
                        v1.clone(),
                        mid.clone(),
                        k2.clone(),
                        v2.clone(),
                    )),
                    k,
                    right.clone(),
                ),
            }
        }
    }
}

fn pop_up<K, V>(xs: ConsList<TreeContext<K, V>>, tree: OrdMap<K, V>) -> OrdMap<K, V> {
    match xs.uncons() {
        None => tree,
        Some((x, ctx)) => match (x.deref().clone(), tree.is_empty()) {
            (TwoLeft(ref k1, ref v1, ref right), true) if right.is_empty() => from_zipper(
                ctx,
                OrdMap::two(OrdMap::new(), k1.clone(), v1.clone(), OrdMap::new()),
            ),
            (TwoRight(ref left, ref k1, ref v1), true) if left.is_empty() => from_zipper(
                ctx,
                OrdMap::two(OrdMap::new(), k1.clone(), v1.clone(), OrdMap::new()),
            ),

            (TwoLeft(ref k1, ref v1, ref right), _) => match *right.0 {
                Leaf => unreachable!(),
                Two(_, ref m, ref k2, ref v2, ref r) => pop_up(
                    ctx,
                    OrdMap::three(
                        tree,
                        k1.clone(),
                        v1.clone(),
                        m.clone(),
                        k2.clone(),
                        v2.clone(),
                        r.clone(),
                    ),
                ),
                Three(_, ref b, ref k2, ref v2, ref c, ref k3, ref v3, ref d) => from_zipper(
                    ctx,
                    OrdMap::two(
                        OrdMap::two(tree, k1.clone(), v1.clone(), b.clone()),
                        k2.clone(),
                        v2.clone(),
                        OrdMap::two(c.clone(), k3.clone(), v3.clone(), d.clone()),
                    ),
                ),
            },
            (TwoRight(ref left, ref k3, ref v3), _) => match *left.0 {
                Leaf => unreachable!(),
                Two(_, ref l, ref k1, ref v1, ref m) => pop_up(
                    ctx,
                    OrdMap::three(
                        l.clone(),
                        k1.clone(),
                        v1.clone(),
                        m.clone(),
                        k3.clone(),
                        v3.clone(),
                        tree,
                    ),
                ),
                Three(_, ref a, ref k1, ref v1, ref b, ref k2, ref v2, ref c) => from_zipper(
                    ctx,
                    OrdMap::two(
                        OrdMap::two(a.clone(), k1.clone(), v1.clone(), b.clone()),
                        k2.clone(),
                        v2.clone(),
                        OrdMap::two(c.clone(), k3.clone(), v3.clone(), tree),
                    ),
                ),
            },
            (ThreeLeft(ref k1, ref v1, ref m, ref k2, ref v2, ref r), true)
                if m.is_empty() && r.is_empty() =>
            {
                from_zipper(
                    ctx,
                    OrdMap::three(
                        OrdMap::new(),
                        k1.clone(),
                        v1.clone(),
                        OrdMap::new(),
                        k2.clone(),
                        v2.clone(),
                        OrdMap::new(),
                    ),
                )
            }
            (ThreeMiddle(ref l, ref k1, ref v1, ref k2, ref v2, ref r), true)
                if l.is_empty() && r.is_empty() =>
            {
                from_zipper(
                    ctx,
                    OrdMap::three(
                        OrdMap::new(),
                        k1.clone(),
                        v1.clone(),
                        OrdMap::new(),
                        k2.clone(),
                        v2.clone(),
                        OrdMap::new(),
                    ),
                )
            }
            (ThreeRight(ref l, ref k1, ref v1, ref m, ref k2, ref v2), true)
                if l.is_empty() && m.is_empty() =>
            {
                from_zipper(
                    ctx,
                    OrdMap::three(
                        OrdMap::new(),
                        k1.clone(),
                        v1.clone(),
                        OrdMap::new(),
                        k2.clone(),
                        v2.clone(),
                        OrdMap::new(),
                    ),
                )
            }
            (ThreeLeft(ref k1, ref v1, ref mid, ref k4, ref v4, ref e), _) => match *mid.0 {
                Leaf => unreachable!(),
                Two(_, ref b, ref k2, ref v2, ref c) => from_zipper(
                    ctx,
                    OrdMap::two(
                        OrdMap::three(
                            tree,
                            k1.clone(),
                            v1.clone(),
                            b.clone(),
                            k2.clone(),
                            v2.clone(),
                            c.clone(),
                        ),
                        k4.clone(),
                        v4.clone(),
                        e.clone(),
                    ),
                ),
                Three(_, ref b, ref k2, ref v2, ref c, ref k3, ref v3, ref d) => from_zipper(
                    ctx,
                    OrdMap::three(
                        OrdMap::two(tree, k1.clone(), v1.clone(), b.clone()),
                        k2.clone(),
                        v2.clone(),
                        OrdMap::two(c.clone(), k3.clone(), v3.clone(), d.clone()),
                        k4.clone(),
                        v4.clone(),
                        e.clone(),
                    ),
                ),
            },
            (ThreeMiddle(ref left, ref kl, ref vl, ref kr, ref vr, ref right), _) => {
                match (&*left.0, &*right.0) {
                    (&Two(_, ref a, ref k1, ref v1, ref b), _) => from_zipper(
                        ctx,
                        OrdMap::two(
                            OrdMap::three(
                                a.clone(),
                                k1.clone(),
                                v1.clone(),
                                b.clone(),
                                kl.clone(),
                                vl.clone(),
                                tree,
                            ),
                            kr.clone(),
                            vr.clone(),
                            right.clone(),
                        ),
                    ),
                    (_, &Two(_, ref c, ref k3, ref v3, ref d)) => from_zipper(
                        ctx,
                        OrdMap::two(
                            left.clone(),
                            kl.clone(),
                            vl.clone(),
                            OrdMap::three(
                                tree,
                                kr.clone(),
                                vr.clone(),
                                c.clone(),
                                k3.clone(),
                                v3.clone(),
                                d.clone(),
                            ),
                        ),
                    ),
                    (&Three(_, ref a, ref k1, ref v1, ref b, ref k2, ref v2, ref c), _) => {
                        from_zipper(
                            ctx,
                            OrdMap::three(
                                OrdMap::two(a.clone(), k1.clone(), v1.clone(), b.clone()),
                                k2.clone(),
                                v2.clone(),
                                OrdMap::two(c.clone(), kl.clone(), vl.clone(), tree),
                                kr.clone(),
                                vr.clone(),
                                right.clone(),
                            ),
                        )
                    }
                    (_, &Three(_, ref c, ref k3, ref v3, ref d, ref k4, ref v4, ref e)) => {
                        from_zipper(
                            ctx,
                            OrdMap::three(
                                left.clone(),
                                kl.clone(),
                                vl.clone(),
                                OrdMap::two(tree, kr.clone(), vr.clone(), c.clone()),
                                k3.clone(),
                                v3.clone(),
                                OrdMap::two(d.clone(), k4.clone(), v4.clone(), e.clone()),
                            ),
                        )
                    }
                    _ => unreachable!(),
                }
            }
            (ThreeRight(ref a, ref k1, ref v1, ref mid, ref k4, ref v4), _) => match *mid.0 {
                Leaf => unreachable!(),
                Two(_, ref b, ref k2, ref v2, ref c) => from_zipper(
                    ctx,
                    OrdMap::two(
                        a.clone(),
                        k1.clone(),
                        v1.clone(),
                        OrdMap::three(
                            b.clone(),
                            k2.clone(),
                            v2.clone(),
                            c.clone(),
                            k4.clone(),
                            v4.clone(),
                            tree,
                        ),
                    ),
                ),
                Three(_, ref b, ref k2, ref v2, ref c, ref k3, ref v3, ref d) => from_zipper(
                    ctx,
                    OrdMap::three(
                        a.clone(),
                        k1.clone(),
                        v1.clone(),
                        OrdMap::two(b.clone(), k2.clone(), v2.clone(), c.clone()),
                        k3.clone(),
                        v3.clone(),
                        OrdMap::two(d.clone(), k4.clone(), v4.clone(), tree),
                    ),
                ),
            },
        },
    }
}

fn max_node<K, V>(m: OrdMap<K, V>) -> (Arc<K>, Arc<V>) {
    match *m.0 {
        Leaf => unreachable!(),
        Two(_, _, ref k, ref v, ref right) if right.is_empty() => (k.clone(), v.clone()),
        Two(_, _, _, _, ref right) => max_node(right.clone()),
        Three(_, _, _, _, _, ref k, ref v, ref right) if right.is_empty() => (k.clone(), v.clone()),
        Three(_, _, _, _, _, _, _, ref right) => max_node(right.clone()),
    }
}

fn remove_max_node<K, V>(ctx: ConsList<TreeContext<K, V>>, m: OrdMap<K, V>) -> OrdMap<K, V> {
    match *m.0 {
        Leaf => unreachable!(),
        Two(_, ref left, _, _, ref right) if left.is_empty() && right.is_empty() => {
            pop_up(ctx, OrdMap::new())
        }
        Two(_, ref left, ref k, ref v, ref right) => remove_max_node(
            ctx.cons(TwoRight(left.clone(), k.clone(), v.clone())),
            right.clone(),
        ),
        Three(_, ref left, ref k1, ref v1, ref mid, _, _, ref right)
            if left.is_empty() && mid.is_empty() && right.is_empty() =>
        {
            pop_up(
                ctx.cons(TwoRight(OrdMap::new(), k1.clone(), v1.clone())),
                OrdMap::new(),
            )
        }
        Three(_, ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => remove_max_node(
            ctx.cons(ThreeRight(
                left.clone(),
                k1.clone(),
                v1.clone(),
                mid.clone(),
                k2.clone(),
                v2.clone(),
            )),
            right.clone(),
        ),
    }
}
