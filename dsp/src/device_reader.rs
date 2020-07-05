use libbladerf_sys::{Device, Error, Metadata, UnitExt};
use std::convert::TryFrom;

// TODO - timeout config
pub struct DeviceReader {
    dev: Device,
    sample_buffer: Vec<i16>,
}

impl DeviceReader {
    pub fn new(dev: Device, sample_buffer_size: usize) -> Self {
        assert!(sample_buffer_size % 2 == 0);
        assert!(sample_buffer_size > 0);
        DeviceReader {
            dev,
            sample_buffer: vec![0; sample_buffer_size],
        }
    }

    pub fn into_inner(self) -> Device {
        self.dev
    }

    pub fn device_mut(&mut self) -> &mut Device {
        &mut self.dev
    }

    pub fn read(&mut self) -> Option<&[i16]> {
        let timeout_ms = 10_u32.ms();
        let mut metadata = Metadata::new_rx_now();

        log::trace!("SyncRx capacity {}", self.sample_buffer.capacity());

        let err = self
            .dev
            .sync_rx(&mut self.sample_buffer, Some(&mut metadata), timeout_ms);
        match err {
            Err(Error::Timeout) => return None,
            Err(_) => {
                log::error!("Device::sync_rx returned {:?}", err);
                return None;
            }
            Ok(_) => (),
        }

        log::trace!("SyncRx-> {}", metadata);

        let num_samples = if metadata.actual_count() == 0 {
            log::warn!("Actual count is zero - logging an empty packet");
            0
        } else if metadata.status().underrun() {
            log::warn!("Underrun detected");
            usize::try_from(metadata.actual_count()).expect("usize::From<u32>")
        } else if metadata.status().overrun() {
            log::warn!("Overrun detected - logging an emtpy packet");
            0
        } else {
            usize::try_from(metadata.actual_count()).expect("usize::From<u32>")
        };

        debug_assert!(num_samples % 2 == 0);

        Some(&self.sample_buffer[..num_samples])
    }
}
