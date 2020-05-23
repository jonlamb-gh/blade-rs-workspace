use chrono::prelude::*;
use libbladerf_sys::{Channel, ChannelLayout, Format, Hertz, MetaFlags, MetaStatus, Sps};
use serde::{Deserialize, Serialize};

pub const HEADER_PREAMBLE: u32 = 0x0D15_EA5E;

pub const VERSION: u8 = 1;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Header {
    pub preamble: u32,
    pub version: u8,
    pub frequency: Hertz,
    pub sample_rate: Sps,
    pub bandwidth: Hertz,
    pub channel: Channel,
    pub layout: ChannelLayout,
    pub format: Format,
    pub system_time: DateTime<Utc>,
}

impl Header {
    pub fn check_preamble(&self) -> Result<(), ()> {
        if self.preamble != HEADER_PREAMBLE {
            Err(())
        } else {
            Ok(())
        }
    }

    pub fn check_version(&self) -> Result<(), ()> {
        if self.version != VERSION {
            Err(())
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Packet {
    pub timestamp: u64,
    pub flags: MetaFlags,
    pub status: MetaStatus,
    pub samples: Vec<i16>,
}
