//! A view into a data structure.
//!
//! A lens describes a view into a data structure, like an index or key lookup,
//! with the ability to modify what it sees. It provides a get operation (get
//! the thing we're a lens for) and a set operation (modify the data structure
//! we're a lens into with the a value for the thing).
//!
//! Lenses are composable, so that if you have a lens from A to a thing B inside
//! A, and you have a lens from B to another thing C inside B, you can compose
//! them to make a lens from A directly into C.
//!
//! There are two kinds of lenses defined here: `PartialLens`, which is a lens
//! into something which doesn't necessarily exist (such as a map key), and
//! `Lens`, which must always be able to succeed. All `Lens`es are also
//! `PartialLens`es, but the opposite is not true.

use std::marker::PhantomData;
use std::sync::Arc;

/// A lens from `From` to `To` where the focus of the lens isn't guaranteed to
/// exist inside `From`. Operations on these lenses therefore return `Option`s.
pub trait PartialLens: Clone {
    type From;
    type To;

    /// Get the focus of the lens, if available.
    fn try_get<R>(&self, s: R) -> Option<Arc<Self::To>>
    where
        R: AsRef<Self::From>;

    /// Put a value into the lens, returning the updated `From` value is the
    /// operation succeeded.
    fn try_put<Convert, R>(&self, v: Option<Convert>, s: R) -> Option<Self::From>
    where
        R: AsRef<Self::From>,
        Arc<Self::To>: From<Convert>;

    /// Compose this lens with a lens from `To` to a new type `Next`, yielding
    /// a lens from `From` to `Next`.
    fn try_chain<L, Next>(&self, next: &L) -> Compose<Self::From, Self::To, Next, Self, L>
    where
        L: PartialLens<From = Self::To, To = Next>,
    {
        compose(self, next)
    }
}

/// A lens from `From` to `To` where `From` is guaranteed to contain the focus
/// of the lens (ie. get and put operations cannot fail).
///
/// These must also implement `PartialLens`, so a default implementation is
/// provided which will just unwrap the results of `try_get` and `try_put`.
pub trait Lens: PartialLens {
    /// Get the focus of the lens.
    fn get<R>(&self, s: R) -> Arc<Self::To>
    where
        R: AsRef<Self::From>,
    {
        self.try_get(s).unwrap()
    }

    /// Put a value into the lens, returning the updated `From` value.
    fn put<Convert, R>(&self, v: Convert, s: R) -> Self::From
    where
        R: AsRef<Self::From>,
        Arc<Self::To>: From<Convert>,
    {
        self.try_put(Some(v), s).unwrap()
    }

    /// Compose this lens with a lens from `To` to a new type `Next`, yielding
    /// a lens from `From` to `Next`.
    fn chain<L, Next>(&self, next: &L) -> Compose<Self::From, Self::To, Next, Self, L>
    where
        L: Lens<From = Self::To, To = Next>,
    {
        compose(self, next)
    }
}

pub struct Compose<A, B, C, L, R>
where
    L: PartialLens<From = A, To = B>,
    R: PartialLens<From = B, To = C>,
{
    left: Arc<L>,
    right: Arc<R>,
    phantom_a: PhantomData<A>,
    phantom_b: PhantomData<B>,
    phantom_c: PhantomData<C>,
}

impl<A, B, C, L, R> Clone for Compose<A, B, C, L, R>
where
    L: PartialLens<From = A, To = B>,
    R: PartialLens<From = B, To = C>,
{
    fn clone(&self) -> Self {
        Compose {
            left: self.left.clone(),
            right: self.right.clone(),
            phantom_a: PhantomData,
            phantom_b: PhantomData,
            phantom_c: PhantomData,
        }
    }
}

impl<A, B, C, L, R> PartialLens for Compose<A, B, C, L, R>
where
    L: PartialLens<From = A, To = B>,
    R: PartialLens<From = B, To = C>,
{
    type From = A;
    type To = C;

    fn try_get<Re>(&self, s: Re) -> Option<Arc<C>>
    where
        Re: AsRef<A>,
    {
        match self.left.try_get(s) {
            None => None,
            Some(s2) => self.right.try_get(s2),
        }
    }

    fn try_put<FromC, Re>(&self, v: Option<FromC>, s: Re) -> Option<A>
    where
        Re: AsRef<A>,
        Arc<C>: From<FromC>,
    {
        self.left
            .try_get(&s)
            .and_then(|s2| self.right.try_put(v, s2))
            .and_then(|s3| self.left.try_put(Some(s3), &s))
    }
}

impl<A, B, C, L, R> Lens for Compose<A, B, C, L, R>
where
    L: Lens<From = A, To = B>,
    R: Lens<From = B, To = C>,
{
}

/// Compose a lens from `A` to `B` with a lens from `B` to `C`, yielding
/// a lens from `A` to `C`.
pub fn compose<A, B, C, L, R>(left: &L, right: &R) -> Compose<A, B, C, L, R>
where
    L: PartialLens<From = A, To = B>,
    R: PartialLens<From = B, To = C>,
{
    Compose {
        left: Arc::new(left.clone()),
        right: Arc::new(right.clone()),
        phantom_a: PhantomData,
        phantom_b: PhantomData,
        phantom_c: PhantomData,
    }
}
