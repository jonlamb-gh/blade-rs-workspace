use crate::ffi::bladerf_metadata;
use bitfield::bitfield;
use serde::{Deserialize, Serialize};
use std::fmt;

bitfield! {
    #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
    pub struct MetaFlags(u32);
    u32;
    pub tx_burst_start, set_tx_burst_start : 0;
    pub tx_burst_end, set_tx_burst_end : 1;
    pub tx_now, set_tx_now : 2;
    pub tx_update_timestamp, set_tx_update_timestamp : 3;

    pub rx_hw_underflow, _ : 0;
    pub rx_hw_miniexp1, _ : 16;
    pub rx_hw_miniexp2, _ : 17;
    pub rx_now, set_rx_now : 31;
}

impl Default for MetaFlags {
    fn default() -> MetaFlags {
        MetaFlags(0)
    }
}

impl From<u32> for MetaFlags {
    fn from(val: u32) -> Self {
        MetaFlags(val)
    }
}

impl MetaFlags {
    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

impl fmt::Display for MetaFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

bitfield! {
    #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
    pub struct MetaStatus(u32);
    u32;
    pub overrun, _ : 0;
    pub underrun, _ : 1;
}

impl Default for MetaStatus {
    fn default() -> MetaStatus {
        MetaStatus(0)
    }
}

impl MetaStatus {
    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

impl From<u32> for MetaStatus {
    fn from(val: u32) -> Self {
        MetaStatus(val)
    }
}

impl fmt::Display for MetaStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Metadata {
    pub(crate) inner: bladerf_metadata,
}

impl Default for Metadata {
    fn default() -> Metadata {
        Metadata::new()
    }
}

impl Metadata {
    pub fn new() -> Self {
        Metadata {
            inner: bladerf_metadata {
                timestamp: 0,
                flags: 0,
                status: 0,
                actual_count: 0,
                reserved: [0; 32],
            },
        }
    }

    pub fn new_rx_now() -> Self {
        let mut flags = MetaFlags::default();
        flags.set_rx_now(true);
        Metadata {
            inner: bladerf_metadata {
                timestamp: 0,
                flags: flags.0,
                status: 0,
                actual_count: 0,
                reserved: [0; 32],
            },
        }
    }

    pub fn clear(&mut self) {
        self.inner.timestamp = 0;
        self.inner.flags = 0;
        self.inner.status = 0;
        self.inner.actual_count = 0;
        self.inner.reserved.iter_mut().for_each(|b| *b = 0);
    }

    pub fn timestamp(&self) -> u64 {
        self.inner.timestamp
    }

    pub fn flags(&self) -> MetaFlags {
        MetaFlags(self.inner.flags)
    }

    pub fn set_flags(&mut self, flags: MetaFlags) {
        self.inner.flags = flags.0;
    }

    pub fn status(&self) -> MetaStatus {
        MetaStatus(self.inner.flags)
    }

    pub fn actual_count(&self) -> u32 {
        self.inner.actual_count
    }
}

impl From<bladerf_metadata> for Metadata {
    fn from(t: bladerf_metadata) -> Metadata {
        Metadata { inner: t }
    }
}

impl fmt::Display for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "t={}, actual-count={}, flags={}, status={}",
            self.timestamp(),
            self.actual_count(),
            self.flags(),
            self.status(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi;

    #[test]
    fn meta_flags() {
        let mut flags = MetaFlags::default();
        assert_eq!(flags.0, 0);

        flags.set_tx_burst_start(true);
        assert_eq!(flags.0, ffi::BLADERF_META_FLAG_TX_BURST_START);
        flags.clear();
        flags.set_tx_burst_end(true);
        assert_eq!(flags.0, ffi::BLADERF_META_FLAG_TX_BURST_END);
        flags.clear();
        flags.set_tx_now(true);
        assert_eq!(flags.0, ffi::BLADERF_META_FLAG_TX_NOW);
        flags.clear();
        flags.set_tx_update_timestamp(true);
        assert_eq!(flags.0, ffi::BLADERF_META_FLAG_TX_UPDATE_TIMESTAMP);

        flags.clear();
        flags.set_rx_now(true);
        assert_eq!(flags.0, ffi::BLADERF_META_FLAG_RX_NOW);

        let flags = MetaFlags(ffi::BLADERF_META_FLAG_RX_HW_UNDERFLOW);
        assert!(flags.rx_hw_underflow());
        let flags = MetaFlags(ffi::BLADERF_META_FLAG_RX_HW_MINIEXP1);
        assert!(flags.rx_hw_miniexp1());
        let flags = MetaFlags(ffi::BLADERF_META_FLAG_RX_HW_MINIEXP2);
        assert!(flags.rx_hw_miniexp2());
    }

    #[test]
    fn meta_status() {
        let status = MetaStatus::default();
        assert_eq!(status.0, 0);
        let status = MetaStatus(ffi::BLADERF_META_STATUS_OVERRUN);
        assert!(status.overrun());
        let status = MetaStatus(ffi::BLADERF_META_STATUS_UNDERRUN);
        assert!(status.underrun());
    }
}
