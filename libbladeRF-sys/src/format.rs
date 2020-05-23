use crate::ffi;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum Format {
    Sc16Q11,
    Sc16Q11Meta,
}

impl Format {
    pub(crate) fn into_ffi(self) -> ffi::bladerf_format {
        use ffi::bladerf_format::*;
        use Format::*;
        match self {
            Sc16Q11 => BLADERF_FORMAT_SC16_Q11,
            Sc16Q11Meta => BLADERF_FORMAT_SC16_Q11_META,
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Format::*;
        match self {
            Sc16Q11 => write!(f, "Signed, Complex 16-bit Q11"),
            Sc16Q11Meta => write!(f, "Signed, Complex 16-bit Q11, with Metadata"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_ffi() {
        use ffi::bladerf_format::*;
        use Format::*;
        assert_eq!(Sc16Q11.into_ffi(), BLADERF_FORMAT_SC16_Q11);
        assert_eq!(Sc16Q11Meta.into_ffi(), BLADERF_FORMAT_SC16_Q11_META);
    }
}
