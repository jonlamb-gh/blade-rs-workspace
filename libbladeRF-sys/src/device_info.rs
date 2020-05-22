use crate::ffi::{bladerf_backend, bladerf_devinfo};
use crate::Error;
use std::{ffi::CStr, fmt, str};

#[derive(Copy, Clone)]
pub struct DeviceInfo {
    pub(crate) inner: bladerf_devinfo,
}

impl DeviceInfo {
    pub fn backend(&self) -> bladerf_backend {
        self.inner.backend
    }

    pub fn serial(&self) -> Result<&str, Error> {
        let slice = unsafe { CStr::from_ptr(self.inner.serial.as_ptr()) };
        slice.to_str().map_err(|_| Error::CString)
    }

    pub fn usb_bus(&self) -> u8 {
        self.inner.usb_bus
    }

    pub fn usb_addr(&self) -> u8 {
        self.inner.usb_addr
    }

    pub fn instance(&self) -> u32 {
        self.inner.instance
    }

    pub fn manufacturer(&self) -> Result<&str, Error> {
        let slice = unsafe { CStr::from_ptr(self.inner.manufacturer.as_ptr()) };
        slice.to_str().map_err(|_| Error::CString)
    }

    pub fn product(&self) -> Result<&str, Error> {
        let slice = unsafe { CStr::from_ptr(self.inner.product.as_ptr()) };
        slice.to_str().map_err(|_| Error::CString)
    }
}

impl From<bladerf_devinfo> for DeviceInfo {
    fn from(t: bladerf_devinfo) -> DeviceInfo {
        DeviceInfo { inner: t }
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}, USB {}:{}, serial={}",
            self.product().unwrap_or("NA"),
            self.usb_bus(),
            self.usb_addr(),
            self.serial().unwrap_or("NA")
        )
    }
}
