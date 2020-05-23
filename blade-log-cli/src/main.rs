use bincode::serialize_into;
use blade_logfile::{Header, Packet, HEADER_PREAMBLE, VERSION};
use libbladerf_sys::*;
use std::convert::TryFrom;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

// TODO
// - error/log pattern
// - more opts, channel, timeouts, io-params, memory, etc
//   - opt for buffered/non-buffered IO, ie if writing to /dev/shm/... no need for it
// - fix the option docs

// conversions if needed
// https://github.com/Nuand/bladeRF/blob/master/host/utilities/bladeRF-cli/src/cmd/rx.c#L49
// https://github.com/Nuand/bladeRF/blob/master/host/utilities/bladeRF-cli/src/cmd/rxtx.c#L394

// https://github.com/servo/bincode

#[derive(Debug, StructOpt)]
#[structopt(name = "blade-log", about = "BladeRF logging")]
pub struct Opts {
    /// BladeRF device ID.
    ///
    /// Format: <backend>:[device=<bus>:<addr>] [instance=<n>] [serial=<serial>]
    ///
    /// Example: "*:serial=f12ce1037830a1b27f3ceeba1f521413"
    #[structopt(short = "d", long, env = "BLADELOG_DEVICE_ID")]
    device_id: String,

    /// Output file
    #[structopt(short = "o", long, parse(from_os_str), env = "BLADELOG_OUTPUT_PATH")]
    output_path: PathBuf,

    /// Frequency (Hertz)
    ///
    /// Accepts:
    ///
    /// * Hertz: <num>H | <num>h
    ///
    /// * KiloHertz: <num>K | <num>k
    ///
    /// * MegaHertz: <num>M | <num>m
    #[structopt(short = "f", long, parse(try_from_str = Hertz::from_str), env = "BLADELOG_FREQUENCY")]
    frequency: Hertz,

    /// Sample rate (samples per second)
    #[structopt(short = "s", long, parse(try_from_str = Sps::from_str), env = "BLADELOG_SAMPLE_RATE")]
    sample_rate: Sps,

    /// Bandwidth (Hertz)
    ///
    /// Accepts:
    ///
    /// * Hertz: <num>H | <num>h
    ///
    /// * KiloHertz: <num>K | <num>k
    ///
    /// * MegaHertz: <num>M | <num>m
    #[structopt(short = "b", long, parse(try_from_str = Hertz::from_str), env = "BLADELOG_BANDWIDTH")]
    bandwidth: Hertz,
}

fn main() -> Result<(), bincode::Error> {
    env_logger::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let opts = Opts::from_args();
    let running = Arc::new(AtomicUsize::new(0));
    let r = running.clone();
    ctrlc::set_handler(move || {
        let prev = r.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            log::info!("Shutting down");
        } else {
            log::warn!("Forcing exit");
            process::exit(0);
        }
    })
    .expect("Error setting Ctrl-C handler");

    log::info!("Creating '{}'", opts.output_path.display());
    let log_file = File::create(opts.output_path)?;
    let mut out_stream = BufWriter::new(log_file);

    log::info!("Opening device ID '{}'", opts.device_id);
    let mut dev = Device::open(&opts.device_id)
        .map_err(|e| log::error!("Device::open returned {:?}", e))
        .unwrap();

    let info = dev
        .device_info()
        .map_err(|e| log::error!("Device::device_info returned {:?}", e))
        .unwrap();
    log::info!("Device info: {}", info);

    let speed = dev
        .device_speed()
        .map_err(|e| log::error!("Device::device_speed returned {:?}", e))
        .unwrap();
    log::info!("Device speed: {}", speed);

    let channel = Channel::Rx0;

    log::info!("Channel: {}", channel);

    log::info!("Frequency: {}", opts.frequency);
    dev.set_frequency(channel, opts.frequency)
        .map_err(|e| log::error!("Device::set_frequency returned {:?}", e))
        .unwrap();

    log::info!("Sample rate: {}", opts.sample_rate);
    let actual_sample_rate = dev
        .set_sample_rate(channel, opts.sample_rate)
        .map_err(|e| log::error!("Device::set_sample_rate returned {:?}", e))
        .unwrap();
    if opts.sample_rate != actual_sample_rate {
        log::warn!("Actual sample rate: {}", actual_sample_rate);
    }

    log::info!("Bandwidth: {}", opts.bandwidth);
    let actual_bandwidth = dev
        .set_bandwidth(channel, opts.bandwidth)
        .map_err(|e| log::error!("Device::set_bandwidth returned {:?}", e))
        .unwrap();
    if opts.bandwidth != actual_bandwidth {
        log::warn!("Actual bandwidth: {}", actual_bandwidth);
    }

    let channel_layout = ChannelLayout::RxX1;
    let format = Format::Sc16Q11Meta;
    let num_buffers = 32;
    let samples_per_buffer = 32 * 1024;
    let num_transfers = 16;
    let timeout_ms = 1000_u32.ms();

    log::info!("Channel layout: {}", channel_layout);
    log::info!("Format: {}", format);
    dev.sync_config(
        channel_layout,
        format,
        num_buffers,
        samples_per_buffer,
        num_transfers,
        timeout_ms,
    )
    .map_err(|e| log::error!("Device::sync_config returned {:?}", e))
    .unwrap();

    dev.enable_module(channel, true)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();
    log::info!("Channel {} is active", channel);

    let header = Header {
        preamble: HEADER_PREAMBLE,
        version: VERSION,
        frequency: opts.frequency,
        sample_rate: opts.sample_rate,
        bandwidth: opts.bandwidth,
        channel,
        layout: channel_layout,
        format,
    };
    serialize_into(&mut out_stream, &header)?;

    // x2 since each sample is a IQ pair (i16, i16)
    //let mut samples: Vec<i16> = vec![0; 32 * 1024 * 2];

    let timeout_ms = 5000_u32.ms();
    let mut metadata = Metadata::new();
    let mut flags = MetaFlags::default();
    flags.set_rx_now(true);

    // TODO - log::info metrics periodically
    let mut total_packets: u128 = 0;
    let mut total_samples: u128 = 0;

    while running.load(Ordering::SeqCst) == 0 {
        // TODO - fix this, don't allocate on each iter
        let mut samples: Vec<i16> = vec![0; 32 * 1024 * 2];

        metadata.clear();
        metadata.set_flags(flags);

        log::trace!("SyncRx capacity {}", samples.len());
        dev.sync_rx(&mut samples, Some(&mut metadata), timeout_ms)
            .map_err(|e| log::error!("Device::sync_rx returned {:?}", e))
            .unwrap();

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

        samples.truncate(num_samples);

        let packet = Packet {
            timestamp: metadata.timestamp(),
            flags: metadata.flags(),
            status: metadata.status(),
            samples,
        };
        serialize_into(&mut out_stream, &packet)?;

        total_packets = total_packets.wrapping_add(1);
        total_samples = total_samples.wrapping_add(num_samples as _);
    }

    out_stream.flush()?;

    log::info!("Total packets: {}", total_packets);
    log::info!("Total samples: {}", total_samples);

    log::info!("Closing device");

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();

    dev.close();

    Ok(())
}
