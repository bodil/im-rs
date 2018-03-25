//! Iterators over immutable data.

pub struct Unfold<F, S> {
    f: F,
    value: S,
}

impl<F, S, A> Iterator for Unfold<F, S>
where
    F: Fn(&S) -> Option<(A, S)>,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.f)(&self.value) {
            None => None,
            Some((next, value)) => {
                self.value = value;
                Some(next)
            }
        }
    }
}

pub struct UnfoldMut<F, S> {
    f: F,
    value: S,
}

impl<F, S, A> Iterator for UnfoldMut<F, S>
where
    F: Fn(&mut S) -> Option<A>,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        (self.f)(&mut self.value)
    }
}

/// Create an iterator of values using a function to update
/// a state value.
///
/// The function is called with the current state as its
/// argument, and should return an [`Option`][std::option::Option] of a tuple of the
/// next value to yield from the iterator and the updated state.
/// If the function returns [`None`][std::option::Option::None], the iterator ends.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate im;
/// # use im::iter::unfold;
/// # use im::catlist::CatList;
/// # fn main() {
/// // Create an infinite stream of numbers, starting at 0.
/// let mut it = unfold(0, |i| Some((*i, *i + 1)));
///
/// // Make a list out of its first five elements.
/// let numbers = CatList::from(it.take(5));
/// assert_eq!(numbers, catlist![0, 1, 2, 3, 4]);
/// # }
/// ```
///
/// [std::option::Option]: https://doc.rust-lang.org/std/option/enum.Option.html
/// [std::option::Option::None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
pub fn unfold<F, S, A>(value: S, f: F) -> Unfold<F, S>
where
    F: Fn(&S) -> Option<(A, S)>,
{
    Unfold { f, value }
}

/// Create an iterator of values using a function to mutate
/// a state value.
///
/// The function is called with a mutable reference to the current state as its
/// argument, and should return an [`Option`][std::option::Option] of the
/// next value to yield from the iterator, updating the state as necessary.
/// If the function returns [`None`][std::option::Option::None], the iterator ends.
///
/// This differs from [`unfold`] in that your update functions will probably be
/// less elegant, but it's easier to deal with state that isn't efficiently cloneable.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate im;
/// # use im::iter::unfold_mut;
/// # use im::catlist::CatList;
/// # fn main() {
/// // Create an infinite stream of numbers, starting at 0.
/// let mut it = unfold_mut(0, |i| {
///   let next = *i;
///   *i += 1;
///   Some(next)
/// });
///
/// // Make a list out of its first five elements.
/// let numbers = CatList::from(it.take(5));
/// assert_eq!(numbers, catlist![0, 1, 2, 3, 4]);
/// # }
/// ```
///
/// [std::option::Option]: https://doc.rust-lang.org/std/option/enum.Option.html
/// [std::option::Option::None]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
pub fn unfold_mut<F, S, A>(value: S, f: F) -> UnfoldMut<F, S>
where
    F: Fn(&mut S) -> Option<A>,
{
    UnfoldMut { f, value }
}
