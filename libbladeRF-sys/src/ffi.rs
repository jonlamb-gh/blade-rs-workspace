#![allow(clippy::all)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::fmt;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl fmt::Display for bladerf_dev_speed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use bladerf_dev_speed::*;
        match self {
            BLADERF_DEVICE_SPEED_UNKNOWN => write!(f, "Unknown"),
            BLADERF_DEVICE_SPEED_HIGH => write!(f, "High"),
            BLADERF_DEVICE_SPEED_SUPER => write!(f, "Super"),
        }
    }
}
