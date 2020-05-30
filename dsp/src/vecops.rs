// Mostly a copy of https://github.com/razorheadfx/aether_primitives/blob/master/src/vecops.rs

use crate::{
    num_complex::Complex,
    num_traits::{Num, Zero},
};

pub trait VecOps<T = f64> {
    fn vec_scale(&mut self, scale: T) -> &mut Self;

    fn vec_zero(&mut self) -> &mut Self;

    fn vec_mutate(&mut self, f: impl FnMut(&mut Complex<T>)) -> &mut Self;

    fn vec_mirror(&mut self) -> &mut Self;
}

macro_rules! impl_vec_ops {
    ($type:ty) => {
        impl<'a, T: Clone + Num> VecOps<T> for $type {
            fn vec_scale(&mut self, scale: T) -> &mut Self {
                self.iter_mut().for_each(|c| *c = c.scale(scale.clone()));
                self
            }

            fn vec_zero(&mut self) -> &mut Self {
                self.iter_mut().for_each(|c| *c = Complex::zero());
                self
            }

            fn vec_mutate(&mut self, f: impl FnMut(&mut Complex<T>)) -> &mut Self {
                self.iter_mut().for_each(f);
                self
            }

            fn vec_mirror(&mut self) -> &mut Self {
                let mid = self.len() / 2;
                (0usize..mid).for_each(|x| self.swap(x, x + mid));
                self
            }
        }
    };
}

impl_vec_ops!(&'a mut Vec<Complex<T>>);
impl_vec_ops!(Vec<Complex<T>>);
impl_vec_ops!(&'a mut [Complex<T>]);
impl_vec_ops!([Complex<T>]);
