use chrono::prelude::*;
use libbladerf_sys::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
use std::convert::TryFrom;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "blade-power-log", about = "BladeRF power logging")]
pub struct Opts {
    /// BladeRF device ID.
    ///
    /// Format: <backend>:[device=<bus>:<addr>] [instance=<n>] [serial=<serial>]
    ///
    /// Example: "*:serial=f12ce1037830a1b27f3ceeba1f521413"
    #[structopt(short = "d", long, env = "BLADELOG_DEVICE_ID")]
    device_id: String,

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

    let system_time = Utc::now();
    log::info!("Header system time: {}", system_time);

    // x2 since each sample is a IQ pair (i16, i16)
    let mut samples: Vec<i16> = vec![0; 32 * 1024 * 2];

    let timeout_ms = 5000_u32.ms();
    let mut metadata = Metadata::new();
    let mut flags = MetaFlags::default();
    flags.set_rx_now(true);

    // TODO - log::info metrics periodically
    let mut total_packets: u128 = 0;
    let mut total_samples: u128 = 0;

    while running.load(Ordering::SeqCst) == 0 {
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

        assert!(num_samples % 2 == 0);
        let num_iq_pairs = num_samples / 2;
        //samples.truncate(num_samples);

        //let mut input: Vec<Complex<f32>> = vec![Complex::zero(); 1234];
        //let mut output: Vec<Complex<f32>> = vec![Complex::zero(); 1234];
        let mut input: Vec<Complex<f32>> = samples[..num_samples]
            .chunks(2)
            .map(|pair| {
                let (i, q) = (normalize_sample(pair[0]), normalize_sample(pair[1]));
                Complex::new(i, q)
            })
            .collect();
        let mut output: Vec<Complex<f32>> = vec![Complex::zero(); num_iq_pairs];
        assert_eq!(input.len(), output.len());

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(num_iq_pairs);

        // Ordered by ascending frequency,
        // with the first element corresponding to frequency 0
        fft.process(&mut input, &mut output);

        let scale = (1_f32 / num_iq_pairs as f32).sqrt();
        for iq in output.iter() {
            let norm = iq.scale(scale);
            let power = norm.norm_sqr();
            let amplitude = power.sqrt();
            println!("{} | pwr {} | amp {}", iq, power, amplitude);
        }

        total_packets = total_packets.wrapping_add(1);
        total_samples = total_samples.wrapping_add(num_samples as _);
    }

    log::info!("Total packets: {}", total_packets);
    log::info!("Total samples: {}", total_samples);

    log::info!("Closing device");

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();

    dev.close();

    Ok(())
}

// Converts i16, in the range [-2048, 2048) to [-1.0, 1.0).
// Note that the lower bound here is inclusive, and the upper bound is exclusive.
// Samples should always be within [-2048, 2047].
fn normalize_sample(s: i16) -> f32 {
    assert!(s >= -2048);
    assert!(s < 2048);
    f32::from(s) / 2048.0
}
