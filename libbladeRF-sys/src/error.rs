use std::fmt;

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Error {
    CString,
    SamplesPerBuffer,
    SamplesLen,
    Unexpected,
    Range,
    Invalid,
    Memory,
    Io,
    Timeout,
    NoDevice,
    Unsupported,
    QueueFull,
    WouldBlock,
    DeviceNotInit,
    Other(i32),
}

impl From<::std::os::raw::c_int> for Error {
    fn from(c_err: ::std::os::raw::c_int) -> Self {
        use crate::ffi;
        use Error::*;
        match c_err {
            ffi::BLADERF_ERR_UNEXPECTED => Unexpected,
            ffi::BLADERF_ERR_RANGE => Range,
            ffi::BLADERF_ERR_INVAL => Invalid,
            ffi::BLADERF_ERR_MEM => Memory,
            ffi::BLADERF_ERR_IO => Io,
            ffi::BLADERF_ERR_TIMEOUT => Timeout,
            ffi::BLADERF_ERR_NODEV => NoDevice,
            ffi::BLADERF_ERR_UNSUPPORTED => Unsupported,
            ffi::BLADERF_ERR_QUEUE_FULL => QueueFull,
            ffi::BLADERF_ERR_WOULD_BLOCK => WouldBlock,
            ffi::BLADERF_ERR_NOT_INIT => DeviceNotInit,
            _ => Other(c_err),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
