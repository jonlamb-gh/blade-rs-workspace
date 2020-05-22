use libbladerf_sys::*;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

// TODO
// - error/log pattern
// - more opts, channel, timeouts, io-params, memory, etc
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

fn main() {
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

    // x2 since each sample is a IQ pair (i16, i16)
    let mut samples: Vec<i16> = vec![0; 32 * 1024 * 2];

    while running.load(Ordering::SeqCst) == 0 {
        let mut metadata = Metadata::new();
        let mut flags = MetaFlags::default();
        flags.set_rx_now(true);
        metadata.set_flags(flags);
        let timeout_ms = 0_u32.ms();
        println!("Requested num_samples: {}", samples.len());
        dev.sync_rx(&mut samples, Some(&mut metadata), timeout_ms)
            .map_err(|e| log::error!("Device::sync_rx returned {:?}", e))
            .unwrap();

        println!("{}", metadata);
        // TODO - check flags/status, warns/etc
    }

    log::info!("Closing device");

    dev.close();
}
