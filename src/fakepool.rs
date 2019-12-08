// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc as RRc;
use std::sync::Arc as RArc;

#[cfg(feature = "pool")]
use refpool::{PoolClone, PoolDefault};

#[cfg(not(feature = "pool"))]
pub trait PoolDefault: Default {}

#[cfg(not(feature = "pool"))]
pub trait PoolClone: Clone {}

#[derive(Clone)]
pub struct Pool<A>(PhantomData<A>);

impl<A> Pool<A> {
    fn new(_size: usize) -> Self {
        Pool(PhantomData)
    }
}

// Rc

#[derive(Default)]
pub struct Rc<A>(RRc<A>);

impl<A> Rc<A> {
    #[inline(always)]
    pub fn default(pool: &Pool<A>) -> Self
    where
        A: PoolDefault,
    {
        Self(Default::default())
    }

    #[inline(always)]
    pub fn new(pool: &Pool<A>, value: A) -> Self {
        Rc(RRc::new(value))
    }

    #[inline(always)]
    pub fn clone_from(pool: &Pool<A>, value: &A) -> Self
    where
        A: PoolClone,
    {
        Rc(RRc::new(value.clone()))
    }

    #[inline(always)]
    pub fn make_mut<'a>(pool: &Pool<A>, this: &'a mut Self) -> &'a mut A
    where
        A: PoolClone,
    {
        RRc::make_mut(&mut this.0)
    }

    #[inline(always)]
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        RRc::ptr_eq(&left.0, &right.0)
    }
}

impl<A> Clone for Rc<A> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Rc(self.0.clone())
    }
}

impl<A> Deref for Rc<A> {
    type Target = A;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<A> PartialEq for Rc<A>
where
    A: PartialEq,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<A> Eq for Rc<A> where A: Eq {}

// Arc

#[derive(Default)]
pub struct Arc<A>(RArc<A>);

impl<A> Arc<A> {
    #[inline(always)]
    pub fn default(pool: &Pool<A>) -> Self
    where
        A: PoolDefault,
    {
        Self(Default::default())
    }

    #[inline(always)]
    pub fn new(pool: &Pool<A>, value: A) -> Self {
        Self(RArc::new(value))
    }

    #[inline(always)]
    pub fn clone_from(pool: &Pool<A>, value: &A) -> Self
    where
        A: PoolClone,
    {
        Self(RArc::new(value.clone()))
    }

    #[inline(always)]
    pub fn make_mut<'a>(pool: &Pool<A>, this: &'a mut Self) -> &'a mut A
    where
        A: PoolClone,
    {
        RArc::make_mut(&mut this.0)
    }

    #[inline(always)]
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        RArc::ptr_eq(&left.0, &right.0)
    }
}

impl<A> Clone for Arc<A> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<A> Deref for Arc<A> {
    type Target = A;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<A> PartialEq for Arc<A>
where
    A: PartialEq,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<A> Eq for Arc<A> where A: Eq {}
