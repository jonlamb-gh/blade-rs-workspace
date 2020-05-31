//! Generated Rust bindings for libbladeRF
//!
//! [libbladeRF](https://github.com/Nuand/bladeRF/tree/master/host/libraries/libbladeRF)
//!
//! [C API Docs](https://www.nuand.com/libbladeRF-doc/)

// TODO
// - separate out this crate into two crates, sys crate is just the C stuff
// - need to double check the bindgen enum use
// - check/fix the CString stuff
// - use utils in https://crates.io/crates/hertz

use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;

mod channel;
mod channel_layout;
mod device_info;
mod error;
pub mod ffi;
mod format;
mod metadata;
pub mod units;

pub use channel::Channel;
pub use channel_layout::ChannelLayout;
pub use device_info::DeviceInfo;
pub use error::Error;
pub use format::Format;
pub use metadata::{MetaFlags, MetaStatus, Metadata};
pub use units::{Hertz, KiloHertz, MegaHertz, MilliSeconds, Sps, UnitExt};

pub type Frequency = Hertz;
pub type Bandwidth = Hertz;
pub type SampleRate = Sps;

// BladeRF 2.0 micro xA4 limits
pub mod device_limits {
    use super::{Hertz, Sps};
    pub const FREQUENCY_MIN: Hertz = Hertz(70_000_000);
    pub const FREQUENCY_MAX: Hertz = Hertz(6_000_000_000);
    pub const BANDWIDTH_MIN: Hertz = Hertz(200_000);
    pub const BANDWIDTH_MAX: Hertz = Hertz(56_000_000);
    pub const SAMPLE_RATE_MIN: Sps = Sps(1);
    pub const SAMPLE_RATE_MAX: Sps = Sps(61_440_000);
}

pub const SAMPLES_PER_BUFFER: usize = 1024;
pub const I16_PER_SAMPLE: usize = 2;

#[derive(Debug)]
pub struct Device {
    dev: *mut ffi::bladerf,
}

impl Device {
    pub fn set_usb_reset_on_open(enabled: bool) {
        unsafe { ffi::bladerf_set_usb_reset_on_open(enabled) };
    }

    pub fn open(device_id: &str) -> Result<Self, Error> {
        let dev_id_cstr = CString::new(device_id).map_err(|_| Error::CString)?;
        let mut dev = MaybeUninit::<*mut ffi::bladerf>::uninit();
        let err = unsafe { ffi::bladerf_open(dev.as_mut_ptr(), dev_id_cstr.as_c_str().as_ptr()) };
        if err != 0 {
            return Err(Error::from(err));
        }
        let dev = unsafe { dev.assume_init() };
        if dev.is_null() {
            return Err(Error::Invalid);
        }
        Ok(Device { dev })
    }

    pub fn close(mut self) {
        unsafe { ffi::bladerf_close(self.dev) };
        self.dev = std::ptr::null_mut();
    }

    pub fn device_reset(&mut self) -> Result<(), Error> {
        let err = unsafe { ffi::bladerf_device_reset(self.dev) };
        if err != 0 {
            return Err(Error::from(err));
        }
        Ok(())
    }

    pub fn device_info(&mut self) -> Result<DeviceInfo, Error> {
        let mut info = MaybeUninit::<ffi::bladerf_devinfo>::uninit();
        let err = unsafe { ffi::bladerf_get_devinfo(self.dev, info.as_mut_ptr()) };
        if err != 0 {
            return Err(Error::from(err));
        }
        let info = unsafe { info.assume_init() };
        Ok(DeviceInfo::from(info))
    }

    pub fn device_speed(&mut self) -> Result<ffi::bladerf_dev_speed, Error> {
        let speed = unsafe { ffi::bladerf_device_speed(self.dev) };
        Ok(speed)
    }

    pub fn board_name(&mut self) -> Result<&str, Error> {
        let board_name = unsafe { ffi::bladerf_get_board_name(self.dev) };
        let slice = unsafe { CStr::from_ptr(board_name) };
        slice.to_str().map_err(|_| Error::CString)
    }

    pub fn set_sample_rate<T: Into<Sps>>(
        &mut self,
        ch: Channel,
        sample_rate: T,
    ) -> Result<SampleRate, Error> {
        let sps: Sps = sample_rate.into();
        let mut actual = MaybeUninit::<ffi::bladerf_sample_rate>::uninit();
        let err = unsafe {
            ffi::bladerf_set_sample_rate(
                self.dev,
                ch.into_ffi(),
                sps.0.try_into().map_err(|_| Error::Range)?,
                actual.as_mut_ptr(),
            )
        };
        if err != 0 {
            return Err(Error::from(err));
        }
        let actual = unsafe { actual.assume_init() };
        Ok(actual.sps())
    }

    pub fn set_bandwidth<T: Into<Hertz>>(
        &mut self,
        ch: Channel,
        bandwidth: T,
    ) -> Result<Bandwidth, Error> {
        let hertz: Hertz = bandwidth.into();
        let mut actual = MaybeUninit::<ffi::bladerf_bandwidth>::uninit();
        let err = unsafe {
            ffi::bladerf_set_bandwidth(
                self.dev,
                ch.into_ffi(),
                hertz.0.try_into().map_err(|_| Error::Range)?,
                actual.as_mut_ptr(),
            )
        };
        if err != 0 {
            return Err(Error::from(err));
        }
        let actual = unsafe { actual.assume_init() };
        Ok(actual.hz())
    }

    pub fn set_frequency<T: Into<Hertz>>(
        &mut self,
        ch: Channel,
        frequency: T,
    ) -> Result<(), Error> {
        let hertz: Hertz = frequency.into();
        let err = unsafe { ffi::bladerf_set_frequency(self.dev, ch.into_ffi(), hertz.0) };
        if err != 0 {
            return Err(Error::from(err));
        }
        Ok(())
    }

    pub fn sync_config(
        &mut self,
        layout: ChannelLayout,
        format: Format,
        num_buffers: usize,
        samples_per_buffer: usize,
        num_transfers: usize,
        stream_timeout: MilliSeconds,
    ) -> Result<(), Error> {
        if samples_per_buffer % SAMPLES_PER_BUFFER != 0 {
            return Err(Error::SamplesPerBuffer);
        }
        let err = unsafe {
            ffi::bladerf_sync_config(
                self.dev,
                layout.into_ffi(),
                format.into_ffi(),
                num_buffers.try_into().map_err(|_| Error::Range)?,
                samples_per_buffer.try_into().map_err(|_| Error::Range)?,
                num_transfers.try_into().map_err(|_| Error::Range)?,
                stream_timeout.0.try_into().map_err(|_| Error::Range)?,
            )
        };
        if err != 0 {
            return Err(Error::from(err));
        }
        Ok(())
    }

    pub fn enable_module(&mut self, ch: Channel, enable: bool) -> Result<(), Error> {
        let err = unsafe { ffi::bladerf_enable_module(self.dev, ch.into_ffi(), enable) };
        if err != 0 {
            return Err(Error::from(err));
        }
        Ok(())
    }

    pub fn sync_rx(
        &mut self,
        samples: &mut [i16],
        metadata: Option<&mut Metadata>,
        timeout: MilliSeconds,
    ) -> Result<(), Error> {
        if samples.len() % I16_PER_SAMPLE != 0 {
            return Err(Error::SamplesLen);
        }
        let num_samples = samples.len() / 2;
        let md_ptr = if let Some(md_ref) = metadata {
            &mut md_ref.inner as *mut _
        } else {
            std::ptr::null_mut()
        };
        let err = unsafe {
            ffi::bladerf_sync_rx(
                self.dev,
                samples.as_mut_ptr() as *mut _,
                num_samples.try_into().map_err(|_| Error::Range)?,
                md_ptr,
                timeout.0.try_into().map_err(|_| Error::Range)?,
            )
        };
        if err != 0 {
            return Err(Error::from(err));
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !self.dev.is_null() {
            unsafe { ffi::bladerf_close(self.dev) };
        }
    }
}
