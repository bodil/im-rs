pub fn is_sorted<A, I>(l: I) -> bool
    where I: IntoIterator<Item = A>,
          A: Ord
{
    let mut it = l.into_iter().peekable();
    loop {
        match (it.next(), it.peek()) {
            (_, None) => return true,
            (Some(ref a), Some(ref b)) if a > b => return false,
            _ => ()
        }
    }
}
