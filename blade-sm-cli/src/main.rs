use dsp::{sample_to_amplitude, BaseOpts, ComplexStorage, DeviceLimits, DeviceReader};
use libbladerf_sys::*;
use std::io;
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "blade-log", about = "BladeRF logging")]
pub struct Opts {
    #[structopt(flatten)]
    base_opts: BaseOpts,
    // Output file
    //#[structopt(short = "o", long, parse(from_os_str), env = "BLADELOG_OUTPUT_PATH")]
    //output_path: PathBuf,
}

fn main() -> Result<(), io::Error> {
    env_logger::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let opts = Opts::from_args();
    let base_opts = opts.base_opts.clone();
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

    let channel = Channel::Rx0;
    let channel_layout = ChannelLayout::RxX1;
    let format = Format::Sc16Q11Meta;
    let num_buffers = 32;
    let samples_per_buffer = 32 * 1024;
    let iq_pairs_per_buffer = samples_per_buffer / 2;
    let num_transfers = 16;
    let timeout_ms = 1000_u32.ms();

    log::info!("Channel: {}", channel);
    log::info!("Frequency: {}", base_opts.frequency);
    log::info!("Sample rate: {}", base_opts.sample_rate);
    log::info!("Bandwidth: {}", base_opts.bandwidth);
    log::info!("Channel layout: {}", channel_layout);
    log::info!("Format: {}", format);

    DeviceLimits::check(
        base_opts.frequency,
        base_opts.bandwidth,
        base_opts.sample_rate,
    )
    .map_err(|e| log::error!("DeviceLimits::check returned {:?}", e))
    .unwrap();

    //log::info!("Creating '{}'", opts.output_path.display());
    //let log_file = File::create(opts.output_path)?;
    //let mut log_writer = BufWriter::new(log_file);

    log::info!("Opening device ID '{}'", base_opts.device_id);
    let mut dev = Device::open(&base_opts.device_id)
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

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();

    dev.set_frequency(channel, base_opts.frequency)
        .map_err(|e| log::error!("Device::set_frequency returned {:?}", e))
        .unwrap();

    let actual_sample_rate = dev
        .set_sample_rate(channel, base_opts.sample_rate)
        .map_err(|e| log::error!("Device::set_sample_rate returned {:?}", e))
        .unwrap();
    if base_opts.sample_rate != actual_sample_rate {
        log::warn!("Actual sample rate: {}", actual_sample_rate);
    }

    let actual_bandwidth = dev
        .set_bandwidth(channel, base_opts.bandwidth)
        .map_err(|e| log::error!("Device::set_bandwidth returned {:?}", e))
        .unwrap();
    if base_opts.bandwidth != actual_bandwidth {
        log::warn!("Actual bandwidth: {}", actual_bandwidth);
    }

    // x2 since each sample is a IQ pair (i16, i16)
    //let mut samples: Vec<i16> = vec![0; 32 * 1024 * 2];

    // TODO - fix this, don't allocate on each iter
    //let mut samples: Vec<i16> = vec![0; 32 * 1024 * 2];

    let mut amplitudes: Vec<f64> = Vec::with_capacity(iq_pairs_per_buffer);
    let mut csum: Vec<f64> = Vec::with_capacity(iq_pairs_per_buffer + 1);

    let mut dev = DeviceReader::new(dev, samples_per_buffer);

    // TODO - chunks for double-buffering
    let mut complex_storage = ComplexStorage::new(iq_pairs_per_buffer);

    dev.device_mut()
        .sync_config(
            channel_layout,
            format,
            num_buffers,
            samples_per_buffer,
            num_transfers,
            timeout_ms,
        )
        .map_err(|e| log::error!("Device::sync_config returned {:?}", e))
        .unwrap();

    dev.device_mut()
        .enable_module(channel, true)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();
    log::info!("Channel {} is active", channel);

    // proto: scm+
    // center_freq: 912600155
    // sample_rate: 2359296
    // data_rate: 32768
    // chip_len: 72
    // preamble_symbols: 16
    // preamble_len: 2304
    // packet_symbols: 128
    // packet_len: 18432
    // bandwidth: ?

    while running.load(Ordering::SeqCst) == 0 {
        // TODO - probably can just double-buffer the chunks of samples instead
        if let Some(samples) = dev.read() {
            complex_storage.clear();
            complex_storage.push_normalize_sc16_q11(samples);
        }

        // Convert complex IQ into a vector of amplitudes
        amplitudes.resize(complex_storage.len(), 0.0);
        amplitudes
            .iter_mut()
            .zip(complex_storage.buffer().iter())
            .for_each(|(a, s)| *a = sample_to_amplitude(s));

        // Matched filter for Manchester coded signal
        csum.resize(complex_storage.len() + 1, 0.0);
        let mut sum = 0.0;
        for (idx, a) in amplitudes.iter().enumerate() {
            sum += a;
            csum[idx + 1] = sum;
        }
    }

    log::info!("Closing device");

    let mut dev = dev.into_inner();

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();

    dev.close();

    Ok(())
}
