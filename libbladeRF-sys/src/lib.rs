//! Generated Rust bindings for libbladeRF
//!
//! https://github.com/Nuand/bladeRF/tree/master/host/libraries/libbladeRF

// https://www.nuand.com/bladeRF-doc/libbladeRF/v2.2.1/index.html
// https://github.com/Nuand/bladeRF/blob/master/host/libraries/libbladeRF_test/test_open/src/main.c
// https://github.com/Nuand/bladeRF/blob/master/host/libraries/libbladeRF_test/test_rx_discont/src/main.c
// https://doc.rust-lang.org/std/mem/union.MaybeUninit.html
// https://doc.rust-lang.org/std/ffi/struct.CStr.html

// TODO
// - need to double check the bindgen enum use
// - opaque struct bladerf_stream
// - de-dupe the ffi error checking with helper macro
// - newtype the aliases

use std::ffi::CStr;
use std::ffi::CString;
use std::mem::MaybeUninit;

mod device_info;
pub mod ffi;
mod metadata;

pub use device_info::DeviceInfo;
pub use metadata::{MetaStatus, Metadata};

// TODO - move to file
// use the bindgen BLADERF_ERR_UNEXPECTED... consts
// Error enum
// impl From<::std::os::raw::c_int> for Error
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    CString,
    Unexpected,
}

// BLADERF_SAMPLERATE_MIN
// BLADERF_SAMPLERATE_REC_MAX
pub type SampleRate = ffi::bladerf_sample_rate;

// BLADERF_BANDWIDTH_MIN
// BLADERF_BANDWIDTH_MAX
pub type Bandwidth = ffi::bladerf_bandwidth;

// BLADERF_FREQUENCY_MIN
// BLADERF_FREQUENCY_MAX
pub type Frequency = ffi::bladerf_frequency;

// TODO - ffi tests
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Channel {
    Rx0,
    Rx1,
    Tx0,
    Tx1,
}

impl Channel {
    fn into_ffi(self) -> ffi::bladerf_channel {
        use Channel::*;
        match self {
            Rx0 => (0 << 1) | 0x0,
            Rx1 => (1 << 1) | 0x0,
            Tx0 => (0 << 1) | 0x1,
            Tx1 => (1 << 1) | 0x1,
        }
    }
}

// TODO - ffi tests
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    RxX1,
    TxX1,
    RxX2,
    TxX2,
}

impl ChannelLayout {
    fn into_ffi(self) -> ffi::bladerf_channel_layout {
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

// TODO - ffi tests
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Format {
    Sc16Q11,
    Sc16Q11Meta,
}

impl Format {
    fn into_ffi(self) -> ffi::bladerf_format {
        use ffi::bladerf_format::*;
        use Format::*;
        match self {
            Sc16Q11 => BLADERF_FORMAT_SC16_Q11,
            Sc16Q11Meta => BLADERF_FORMAT_SC16_Q11_META,
        }
    }
}

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
        match err {
            0 => (),
            _ => todo!(),
        }
        let dev = unsafe { dev.assume_init() };
        if dev.is_null() {
            todo!();
        }
        Ok(Device { dev })
    }

    pub fn close(mut self) {
        unsafe { ffi::bladerf_close(self.dev) };
        self.dev = std::ptr::null_mut();
    }

    pub fn device_reset(&mut self) -> Result<(), Error> {
        let err = unsafe { ffi::bladerf_device_reset(self.dev) };
        match err {
            0 => (),
            _ => todo!(),
        }
        Ok(())
    }

    pub fn device_info(&mut self) -> Result<DeviceInfo, Error> {
        let mut info = MaybeUninit::<ffi::bladerf_devinfo>::uninit();
        let err = unsafe { ffi::bladerf_get_devinfo(self.dev, info.as_mut_ptr()) };
        match err {
            0 => (),
            _ => todo!(),
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

    pub fn set_sample_rate(&mut self, ch: Channel, rate: SampleRate) -> Result<SampleRate, Error> {
        let mut actual = MaybeUninit::<ffi::bladerf_sample_rate>::uninit();
        let err = unsafe {
            ffi::bladerf_set_sample_rate(self.dev, ch.into_ffi(), rate, actual.as_mut_ptr())
        };
        match err {
            0 => (),
            _ => todo!(),
        }
        let actual = unsafe { actual.assume_init() };
        Ok(actual)
    }

    pub fn set_bandwidth(&mut self, ch: Channel, bandwidth: Bandwidth) -> Result<Bandwidth, Error> {
        let mut actual = MaybeUninit::<ffi::bladerf_bandwidth>::uninit();
        let err = unsafe {
            ffi::bladerf_set_bandwidth(self.dev, ch.into_ffi(), bandwidth, actual.as_mut_ptr())
        };
        match err {
            0 => (),
            _ => todo!(),
        }
        let actual = unsafe { actual.assume_init() };
        Ok(actual)
    }

    pub fn set_frequency(&mut self, ch: Channel, frequency: Frequency) -> Result<(), Error> {
        let err = unsafe { ffi::bladerf_set_frequency(self.dev, ch.into_ffi(), frequency) };
        match err {
            0 => (),
            _ => todo!(),
        }
        Ok(())
    }

    // cli uses these defaults
    // https://github.com/Nuand/bladeRF/blob/master/host/utilities/bladeRF-cli/src/cmd/rxtx.c#L394
    pub fn sync_config(
        &mut self,
        layout: ChannelLayout,
        format: Format,
        num_buffers: u32,
        buffer_size: u32,
        num_transfers: u32,
        stream_timeout_ms: u32,
    ) -> Result<(), Error> {
        let err = unsafe {
            ffi::bladerf_sync_config(
                self.dev,
                layout.into_ffi(),
                format.into_ffi(),
                num_buffers,
                buffer_size,
                num_transfers,
                stream_timeout_ms,
            )
        };
        match err {
            0 => (),
            _ => todo!(),
        }
        Ok(())
    }

    pub fn enable_module(&mut self, ch: Channel, enable: bool) -> Result<(), Error> {
        let err = unsafe { ffi::bladerf_enable_module(self.dev, ch.into_ffi(), enable) };
        match err {
            0 => (),
            _ => todo!(),
        }
        Ok(())
    }

    pub fn sync_rx(
        &mut self,
        samples: &mut [i16],
        num_samples: u32,
        metadata: &mut Metadata,
        timeout_ms: u32,
    ) -> Result<(), Error> {
        let err = unsafe {
            ffi::bladerf_sync_rx(
                self.dev,
                samples.as_mut_ptr() as *mut _,
                num_samples,
                &mut metadata.inner as *mut _,
                timeout_ms,
            )
        };
        match err {
            0 => (),
            _ => todo!(),
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
