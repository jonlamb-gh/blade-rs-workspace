use crate::ffi;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ChannelLayout {
    RxX1,
    TxX1,
    RxX2,
    TxX2,
}

impl ChannelLayout {
    pub(crate) fn into_ffi(self) -> ffi::bladerf_channel_layout {
        use ffi::bladerf_channel_layout::*;
        use ChannelLayout::*;
        match self {
            RxX1 => BLADERF_RX_X1,
            TxX1 => BLADERF_TX_X1,
            RxX2 => BLADERF_RX_X2,
            TxX2 => BLADERF_TX_X2,
        }
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ChannelLayout::*;
        match self {
            RxX1 => write!(f, "x1 RX (SISO)"),
            TxX1 => write!(f, "x1 TX (SISO)"),
            RxX2 => write!(f, "x2 RX (MIMO)"),
            TxX2 => write!(f, "x2 TX (MIMO)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_layout_ffi() {
        use ffi::bladerf_channel_layout::*;
        use ChannelLayout::*;
        assert_eq!(RxX1.into_ffi(), BLADERF_RX_X1);
        assert_eq!(TxX1.into_ffi(), BLADERF_TX_X1);
        assert_eq!(RxX2.into_ffi(), BLADERF_RX_X2);
        assert_eq!(TxX2.into_ffi(), BLADERF_TX_X2);
    }
}
