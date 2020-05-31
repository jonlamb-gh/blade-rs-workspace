use libbladerf_sys::{device_limits, Bandwidth, Frequency, SampleRate};
use std::fmt;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum LimitsError {
    Frequency,
    Bandwidth,
    SampleRate,
}

pub struct DeviceLimits {}

impl DeviceLimits {
    pub fn check(
        frequency: Frequency,
        bandwidth: Bandwidth,
        sample_rate: SampleRate,
    ) -> Result<(), LimitsError> {
        if (frequency < device_limits::FREQUENCY_MIN) || (frequency > device_limits::FREQUENCY_MAX)
        {
            Err(LimitsError::Frequency)
        } else if (bandwidth < device_limits::BANDWIDTH_MIN)
            || (bandwidth > device_limits::BANDWIDTH_MAX)
        {
            Err(LimitsError::Bandwidth)
        } else if (sample_rate < device_limits::SAMPLE_RATE_MIN)
            || (sample_rate > device_limits::SAMPLE_RATE_MAX)
        {
            Err(LimitsError::SampleRate)
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for LimitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
