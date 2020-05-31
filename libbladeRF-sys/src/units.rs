use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::ParseIntError;
use std::str::FromStr;

/// Samples per second
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Sps(pub u64);

/// Hertz
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Hertz(pub u64);

/// KiloHertz
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct KiloHertz(pub u64);

/// MegaHertz
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct MegaHertz(pub u64);

/// MilliSeconds
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct MilliSeconds(pub u64);

impl Sps {
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl Hertz {
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl KiloHertz {
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl MegaHertz {
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl MilliSeconds {
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

pub trait UnitExt {
    /// Wrap in `Sps`
    fn sps(self) -> Sps;

    /// Wrap in `Hertz`
    fn hz(self) -> Hertz;

    /// Wrap in `KiloHertz`
    fn khz(self) -> KiloHertz;

    /// Wrap in `MegaHertz`
    fn mhz(self) -> MegaHertz;

    /// Wrap in `MilliSeconds`
    fn ms(self) -> MilliSeconds;
}

impl UnitExt for u32 {
    fn sps(self) -> Sps {
        Sps(self.into())
    }

    fn hz(self) -> Hertz {
        Hertz(self.into())
    }

    fn khz(self) -> KiloHertz {
        KiloHertz(self.into())
    }

    fn mhz(self) -> MegaHertz {
        MegaHertz(self.into())
    }

    fn ms(self) -> MilliSeconds {
        MilliSeconds(self.into())
    }
}

impl UnitExt for u64 {
    fn sps(self) -> Sps {
        Sps(self)
    }

    fn hz(self) -> Hertz {
        Hertz(self)
    }

    fn khz(self) -> KiloHertz {
        KiloHertz(self)
    }

    fn mhz(self) -> MegaHertz {
        MegaHertz(self)
    }

    fn ms(self) -> MilliSeconds {
        MilliSeconds(self)
    }
}

impl From<u64> for Hertz {
    fn from(val: u64) -> Self {
        Hertz(val)
    }
}

impl From<u64> for KiloHertz {
    fn from(val: u64) -> Self {
        KiloHertz(val)
    }
}

impl From<u64> for MegaHertz {
    fn from(val: u64) -> Self {
        MegaHertz(val)
    }
}

impl Into<Hertz> for KiloHertz {
    fn into(self) -> Hertz {
        Hertz(self.0 * 1_000)
    }
}

impl Into<Hertz> for MegaHertz {
    fn into(self) -> Hertz {
        Hertz(self.0 * 1_000_000)
    }
}

impl Into<KiloHertz> for MegaHertz {
    fn into(self) -> KiloHertz {
        KiloHertz(self.0 * 1_000)
    }
}

impl PartialEq<KiloHertz> for Hertz {
    fn eq(&self, other: &KiloHertz) -> bool {
        *self == Into::<Hertz>::into(*other)
    }
}

impl PartialEq<Hertz> for KiloHertz {
    fn eq(&self, other: &Hertz) -> bool {
        Into::<Hertz>::into(*self) == *other
    }
}

impl PartialEq<MegaHertz> for Hertz {
    fn eq(&self, other: &MegaHertz) -> bool {
        *self == Into::<Hertz>::into(*other)
    }
}

impl PartialEq<Hertz> for MegaHertz {
    fn eq(&self, other: &Hertz) -> bool {
        Into::<Hertz>::into(*self) == *other
    }
}

impl PartialEq<MegaHertz> for KiloHertz {
    fn eq(&self, other: &MegaHertz) -> bool {
        Into::<Hertz>::into(*self) == Into::<Hertz>::into(*other)
    }
}

impl PartialEq<KiloHertz> for MegaHertz {
    fn eq(&self, other: &KiloHertz) -> bool {
        Into::<Hertz>::into(*self) == Into::<Hertz>::into(*other)
    }
}

impl fmt::Display for Sps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Sps", self.0)
    }
}

impl fmt::Display for Hertz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pretty_fmt(*self, f)
    }
}

impl fmt::Display for KiloHertz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} kHz", self.0)
    }
}

impl fmt::Display for MegaHertz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} MHz", self.0)
    }
}

pub const ONE_KHZ: Hertz = Hertz(1_000);
pub const ONE_MHZ: Hertz = Hertz(1_000_000);
pub const ONE_GHZ: Hertz = Hertz(1_000_000_000);

fn pretty_fmt(h: Hertz, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if h.0 >= ONE_GHZ.0 {
        write!(f, "{:.04} GHz", h.as_f64() / ONE_GHZ.as_f64())
    } else if h.0 >= ONE_MHZ.0 {
        write!(f, "{:.04} MHz", h.as_f64() / ONE_MHZ.as_f64())
    } else if h.0 >= ONE_KHZ.0 {
        write!(f, "{:.04} KHz", h.as_f64() / ONE_KHZ.as_f64())
    } else {
        write!(f, "{} Hz", h.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseHertzError {
    Empty,
    UnitError,
    IntError(ParseIntError),
}

impl From<ParseIntError> for ParseHertzError {
    fn from(e: ParseIntError) -> Self {
        ParseHertzError::IntError(e)
    }
}

impl fmt::Display for ParseHertzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for Hertz {
    type Err = ParseHertzError;

    // TODO - add floats, 1.4K -> 1_400 Hz, GHz
    // Values can be specified as an integer (89100000),
    // a float (89.1e6) or as a metric suffix (89.1M)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Err(ParseHertzError::Empty)
        } else {
            // Last char can be a unit: h | H, k | K, m | M
            let last_index = s.len() - 1;
            let last_char = s.chars().last().ok_or(ParseHertzError::Empty)?;
            let f = match last_char {
                'h' | 'H' => s
                    .get(..last_index)
                    .ok_or(ParseHertzError::Empty)?
                    .parse::<u64>()?
                    .hz(),
                'k' | 'K' => s
                    .get(..last_index)
                    .ok_or(ParseHertzError::Empty)?
                    .parse::<u64>()?
                    .khz()
                    .into(),
                'm' | 'M' => s
                    .get(..last_index)
                    .ok_or(ParseHertzError::Empty)?
                    .parse::<u64>()?
                    .mhz()
                    .into(),
                _ => s.parse::<u64>()?.hz(),
            };
            Ok(f)
        }
    }
}

impl FromStr for Sps {
    type Err = ParseIntError;

    // TODO - add floats, 1.4K -> 1_400 Sps
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.parse::<u64>()?.sps())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_hertz() {
        let a = 1_u32.khz();
        let b = 1000_u64.hz();
        assert_eq!(a, b);
        assert_eq!(b, a);
        let a = 1_u32.mhz();
        let b = 1_000_000_u64.hz();
        assert_eq!(a, b);
        assert_eq!(b, a);
        let a = 1_u32.mhz();
        let b = 1_000_u64.khz();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn hertz_from_str() {
        assert_eq!(Hertz::from_str("100"), Ok(Hertz(100)));
        assert_eq!(Hertz::from_str("1H"), Ok(Hertz(1)));
        assert_eq!(Hertz::from_str("123h"), Ok(Hertz(123)));

        assert_eq!(Hertz::from_str("13k"), Ok(13_u64.khz().into()));
        assert_eq!(Hertz::from_str("5K"), Ok(5_u64.khz().into()));

        assert_eq!(Hertz::from_str("3M"), Ok(3_u64.mhz().into()));
        assert_eq!(Hertz::from_str("12m"), Ok(12_u64.mhz().into()));
    }

    #[test]
    fn sps_from_str() {
        assert_eq!(Sps::from_str("1"), Ok(Sps(1)));
        assert_eq!(Sps::from_str("100"), Ok(Sps(100)));
    }
}
