use crate::ffi;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum Channel {
    Rx0,
    Rx1,
    Tx0,
    Tx1,
}

impl Channel {
    pub(crate) fn into_ffi(self) -> ffi::bladerf_channel {
        use Channel::*;
        match self {
            Rx0 => 0b00,
            Rx1 => 0b10,
            Tx0 => 0b01,
            Tx1 => 0b11,
        }
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Channel::*;
        match self {
            Rx0 => write!(f, "Rx0"),
            Rx1 => write!(f, "Rx1"),
            Tx0 => write!(f, "Tx0"),
            Tx1 => write!(f, "Tx1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_ffi() {
        assert_eq!(Channel::Rx0.into_ffi(), 0b00);
        assert_eq!(Channel::Rx1.into_ffi(), 0b10);

        assert_eq!(Channel::Tx0.into_ffi(), 0b01);
        assert_eq!(Channel::Tx1.into_ffi(), 0b11);
    }
}
