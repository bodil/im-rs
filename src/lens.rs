//! A view into a data structure.
//!
//! A lens describes a view into a data structure, like an index or key lookup,
//! with the ability to modify what it sees. It provides a get operation (get
//! the thing we're a lens for) and a set operation (update the data structure
//! we're a lens into with the new value for the thing).
//!
//! Lenses are composable, so that if you have a lens from A to a thing B inside
//! A, and you have a lens from B to another thing C inside B, you can compose
//! them to make a lens from A directly into C.
//!
//! There are two kinds of lenses defined here: `PartialLens`, which is a lens
//! into something which doesn't necessarily exist (such as a map key), and
//! `Lens`, which must always be able to succeed (like a lens into a struct,
/// whose contents are known and guaranteed). All `Lens`es are also
/// `PartialLens`es, but the opposite is not true.

use std::marker::PhantomData;
use std::sync::Arc;

/// A lens from `From` to `To` where the focus of the lens isn't guaranteed to
/// exist inside `From`. Operations on these lenses therefore return `Option`s.
pub trait PartialLens: Clone {
    type From;
    type To;

    /// Get the focus of the lens, if available.
    fn try_get(&self, s: &Self::From) -> Option<Arc<Self::To>>;

    /// Put a value into the lens, returning the updated `From` value if the
    /// operation succeeded.
    fn try_put<Convert>(&self, v: Option<Convert>, s: &Self::From) -> Option<Self::From>
        where
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
    fn get(&self, s: &Self::From) -> Arc<Self::To> {
        self.try_get(s).unwrap()
    }

    /// Put a value into the lens, returning the updated `From` value.
    fn put<Convert>(&self, v: Convert, s: &Self::From) -> Self::From
        where
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

    fn try_get(&self, s: &A) -> Option<Arc<C>> {
        match self.left.try_get(s) {
            None => None,
            Some(s2) => self.right.try_get(&s2),
        }
    }

    fn try_put<FromC>(&self, v: Option<FromC>, s: &A) -> Option<A>
        where
        Arc<C>: From<FromC>,
    {
        self.left
            .try_get(&s)
            .and_then(|s2| self.right.try_put(v, &s2))
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

/// An arbitrary non-partial lens defined by a pair of get and put functions.
pub struct GeneralLens<From, To> {
    get: Arc<Fn(&From) -> Arc<To>>,
    put: Arc<Fn(&From, Arc<To>) -> From>,
}

impl<From, To> GeneralLens<From, To> {
    /// Construct a lens from `From` to `To` from a pair of get and put functions.
    pub fn new(
        get: Arc<Fn(&From) -> Arc<To>>,
        put: Arc<Fn(&From, Arc<To>) -> From>,
    ) -> GeneralLens<From, To> {
        GeneralLens { get, put }
    }
}

impl<From, To> Clone for GeneralLens<From, To> {
    fn clone(&self) -> Self {
        GeneralLens {
            get: self.get.clone(),
            put: self.put.clone(),
        }
    }
}

impl<A, B> PartialLens for GeneralLens<A, B> {
    type From = A;
    type To = B;

    fn try_get(&self, s: &A) -> Option<Arc<B>> {
        Some((self.get)(s))
    }

    fn try_put<Convert>(&self, v: Option<Convert>, s: &A) -> Option<A>
        where
        Arc<B>: From<Convert>,
    {
        Some((self.put)(s, From::from(v.unwrap())))
    }
}

impl<A, B> Lens for GeneralLens<A, B> {
    fn get(&self, s: &A) -> Arc<B> {
        (self.get)(s)
    }

    fn put<Convert>(&self, v: Convert, s: &A) -> A
        where
        Arc<B>: From<Convert>,
    {
        (self.put)(s, From::from(v))
    }
}

/// Construct a lens into a struct, or a series of structs.
///
/// You'll need to specify the type of the source struct, the
/// name of the field you want a lens into, and the type of that
/// field, separated by colons, eg.
/// `lens!(MyStruct: string_field: String)`.
/// You can keep repeating name/type pairs for fields inside structs
/// inside structs.
///
/// **Please note:** this only works on fields which are wrapped in an
/// `Arc` (so the type of the `string_field` field in the example in
/// the previous paragraph must be `Arc<String`>), and the source
/// struct must implement `Clone`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate im;
/// # use im::lens::Lens;
/// # use std::sync::Arc;
/// #[derive(Clone)]
/// struct Name {
///     first: Arc<String>,
///     last: Arc<String>
/// }
///
/// #[derive(Clone)]
/// struct Person {
///     name: Arc<Name>
/// }
///
/// # fn main() {
/// let person_last_name = lens!(Person: name: Name: last: String);
///
/// let the_admiral = Person {
///     name: Arc::new(Name {
///         first: Arc::new("Grace".to_string()),
///         last: Arc::new("Hopper".to_string())
///     })
/// };
///
/// assert_eq!(
///     Arc::new("Hopper".to_string()),
///     person_last_name.get(&the_admiral)
/// );
/// # }
/// ```
#[macro_export]
macro_rules! lens {
    ( $from:ident : $headpath:ident : $headto:ident : $($tail:tt):* ) => {
        $crate::lens::compose(&lens!($from : $headpath : $headto), &lens!($headto : $($tail):*))
    };

    ( $from:ident : $path:ident : $to:ident ) => {
        $crate::lens::GeneralLens::<$from, $to>::new(
            ::std::sync::Arc::new(|st| st.$path.clone()),
            ::std::sync::Arc::new(|st, v| {
                $from {
                    $path: v,
                    ..st.clone()
                }
            })
        )
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone)]
    struct Inner {
        omg: Arc<String>,
        wtf: Arc<String>
    }

    #[derive(Clone)]
    struct Outer {
        inner: Arc<Inner>,
    }

    #[test]
    fn struct_lens() {
        let l = lens!(Outer: inner: Inner: omg: String);
        let omglol = "omg lol".to_string();
        let inner = Inner { omg: Arc::new(omglol.clone()), wtf: Arc::new("nope".to_string()) };
        let outer = Outer { inner: Arc::new(inner) };
        assert_eq!(Arc::new(omglol), l.get(&outer))
    }
}
