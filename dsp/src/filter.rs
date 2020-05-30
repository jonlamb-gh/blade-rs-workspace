// Copied from https://github.com/signalo/signalo/blob/master/traits/src/filter.rs

use crate::{median::Filter as MedianFilter, num_traits::Num};

pub trait Filter<Input> {
    type Output;

    /// Processes the input value, returning a corresponding output.
    fn filter(&mut self, input: Input) -> Self::Output;
}

impl<T> Filter<T> for MedianFilter<T>
where
    T: Clone + Num + PartialOrd,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.consume(input)
    }
}
