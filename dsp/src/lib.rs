// TODO - replace custom stuff with things from
// https://github.com/signalo/signalo

pub use crate::base_opts::BaseOpts;
pub use crate::binned_frequency_range::BinnedFrequencyRange;
pub use crate::complex_storage::ComplexStorage;
pub use crate::device_limits::{DeviceLimits, LimitsError};
pub use crate::device_reader::DeviceReader;
pub use crate::filter::Filter;
pub use crate::vecops::VecOps;
pub use hertz;
pub use median;
pub use rustfft::num_traits::{clamp, clamp_max, clamp_min};
pub use rustfft::{self, num_complex, num_traits};

use crate::num_complex::Complex;
use crate::num_traits::{real::Real, Num};

mod base_opts;
mod binned_frequency_range;
mod complex_storage;
mod device_limits;
mod device_reader;
mod filter;
mod vecops;

/// Converts i16, in the range [-2048, 2048) to [-1.0, 1.0).
/// Note that the lower bound here is inclusive, and the upper bound is exclusive.
/// Samples should always be within [-2048, 2047].
pub fn normalize_sc16_q11(s: i16) -> f64 {
    debug_assert!(s >= -2048);
    debug_assert!(s < 2048);
    f64::from(s) / 2048.0
}

pub fn downsample<T>(src: &[T], dst: &mut [T])
where
    T: Copy,
{
    debug_assert_eq!(
        src.len() % dst.len(),
        0,
        "Only even decimations are supported {} {}",
        src.len(),
        dst.len()
    );

    let dec = src.len() / dst.len();
    dst.iter_mut()
        .enumerate()
        .for_each(|(i, c)| *c = src[i * dec]);
}

// dB = 10.0 * power.log(10.0)
// amplitude = power.sqrt()
pub fn sample_to_power<T: Clone + Num>(s: &Complex<T>) -> T {
    s.norm_sqr()
}

pub fn sample_to_amplitude<T: Clone + Num + Real>(s: &Complex<T>) -> T {
    s.norm_sqr().sqrt()
}
