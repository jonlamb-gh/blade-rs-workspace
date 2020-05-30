use libbladerf_sys::{units::ParseHertzError, Hertz};
use std::convert::TryFrom;
use std::fmt;
use std::ops::RangeInclusive;
use std::str::FromStr;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Error {
    NotDivisible,
}

// TODO
// Values can be specified as an integer (89100000),
// a float (89.1e6) or as a metric suffix (89.1M)
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinnedFrequencyRange {
    lower: Hertz,
    upper: Hertz,
    bin_width: Hertz,
    bins: usize,
}

impl BinnedFrequencyRange {
    pub fn new(range: RangeInclusive<Hertz>, bin_width: Hertz) -> Result<Self, Error> {
        // TODO - more checks, usize conversion, etc
        if (range.end().0 - range.start().0) % bin_width.0 != 0 {
            Err(Error::NotDivisible)
        } else {
            Ok(Self::new_unchecked(range, bin_width))
        }
    }

    pub fn new_unchecked(range: RangeInclusive<Hertz>, bin_width: Hertz) -> Self {
        let width = range.end().0 - range.start().0;
        debug_assert_eq!(width % bin_width.0, 0);
        let bins = width / bin_width.0;
        BinnedFrequencyRange {
            lower: *range.start(),
            upper: *range.end(),
            bin_width,
            bins: usize::try_from(bins).unwrap(),
        }
    }

    pub fn range(&self) -> RangeInclusive<Hertz> {
        self.lower..=self.upper
    }

    pub fn lower(&self) -> Hertz {
        self.lower
    }

    pub fn upper(&self) -> Hertz {
        self.upper
    }

    pub fn bin_width(&self) -> Hertz {
        self.bin_width
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    pub fn center_frequency(&self, bin_index: usize) -> Option<Hertz> {
        if bin_index >= self.bins() {
            None
        } else {
            let lower: u64 =
                self.lower().0 + u64::try_from(bin_index).unwrap() * self.bin_width().0;
            Some((lower + (self.bin_width().0 / 2)).into())
        }
    }
}

impl IntoIterator for BinnedFrequencyRange {
    type Item = Hertz;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let bw2 = self.bin_width.0 / 2;
        let bw = usize::try_from(self.bin_width.0).unwrap();
        let center_freqs: Vec<Hertz> = (self.lower.0 + bw2..=self.upper.0 - bw2)
            .step_by(bw)
            .map(|v| v.into())
            .collect();
        center_freqs.into_iter()
    }
}

impl fmt::Display for BinnedFrequencyRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} : {} : {}",
            self.lower(),
            self.upper(),
            self.bin_width()
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    Syntax,
    Creation(Error),
    ParseHertz(ParseHertzError),
}

impl From<Error> for ParseError {
    fn from(e: Error) -> Self {
        ParseError::Creation(e)
    }
}

impl From<ParseHertzError> for ParseError {
    fn from(e: ParseHertzError) -> Self {
        ParseError::ParseHertz(e)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for BinnedFrequencyRange {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let split: Vec<&str> = s.split(':').collect();
        if split.len() != 3 {
            return Err(ParseError::Syntax);
        }
        let lower = Hertz::from_str(split[0])?;
        let upper = Hertz::from_str(split[1])?;
        let bin_width = Hertz::from_str(split[2])?;
        Ok(BinnedFrequencyRange::new(lower..=upper, bin_width)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libbladerf_sys::UnitExt;

    #[test]
    fn from_str() {
        assert_eq!(
            BinnedFrequencyRange::from_str("60M:160M:10M"),
            Ok(BinnedFrequencyRange {
                lower: 60_u32.mhz().into(),
                upper: 160_u32.mhz().into(),
                bin_width: 10_u32.mhz().into(),
                bins: 10,
            })
        );
        assert_eq!(
            BinnedFrequencyRange::from_str("100K:200K:100K"),
            Ok(BinnedFrequencyRange {
                lower: 100_u32.khz().into(),
                upper: 200_u32.khz().into(),
                bin_width: 100_u32.khz().into(),
                bins: 1,
            })
        );
    }

    #[test]
    fn center_freq_iter() {
        let bfr = BinnedFrequencyRange::from_str("0M:100M:50M").unwrap();
        assert_eq!(bfr.bins(), 2);
        assert_eq!(bfr.bin_width(), 50_u32.mhz());
        let iter = bfr.into_iter();
        let center_freqs: Vec<Hertz> = iter.collect();
        let v: Vec<Hertz> = vec![25_u32.mhz().into(), 75_u32.mhz().into()];
        assert_eq!(center_freqs, v);
        assert_eq!(bfr.center_frequency(0), Some(25_u32.mhz().into()));
        assert_eq!(bfr.center_frequency(1), Some(75_u32.mhz().into()));
        assert_eq!(bfr.center_frequency(2), None);
        let bfr = BinnedFrequencyRange::from_str("60M:160M:10M").unwrap();
        assert_eq!(bfr.bins(), 10);
        assert_eq!(bfr.bin_width(), 10_u32.mhz());
        let iter = bfr.into_iter();
        let center_freqs: Vec<Hertz> = iter.collect();
        let v: Vec<Hertz> = vec![
            65_u32.mhz().into(),
            75_u32.mhz().into(),
            85_u32.mhz().into(),
            95_u32.mhz().into(),
            105_u32.mhz().into(),
            115_u32.mhz().into(),
            125_u32.mhz().into(),
            135_u32.mhz().into(),
            145_u32.mhz().into(),
            155_u32.mhz().into(),
        ];
        assert_eq!(center_freqs, v);
        assert_eq!(bfr.center_frequency(0), Some(65_u32.mhz().into()));
        assert_eq!(bfr.center_frequency(9), Some(155_u32.mhz().into()));
        assert_eq!(bfr.center_frequency(10), None);
    }
}
