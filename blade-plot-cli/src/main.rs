use crate::complex_storage::ComplexStorage;
use crate::device_reader::DeviceReader;
use dsp::{
    median::Filter as MedianFilter, num_complex::Complex, num_traits::Zero, rustfft::FFTplanner,
    Filter, VecOps,
};
use libbladerf_sys::*;
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::prelude::*;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

mod complex_storage;
mod device_reader;

#[derive(Debug, StructOpt)]
#[structopt(name = "blade-plot", about = "BladeRF plotting")]
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
    env_logger::from_env(
        env_logger::Env::default().default_filter_or("info,gfx_device_gl=warn,winit=warn"),
    )
    .init();
    let opts = Opts::from_args();
    let running = Arc::new(AtomicUsize::new(0));
    let r = running;
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

    let channel_layout = ChannelLayout::RxX1;
    let format = Format::Sc16Q11Meta;
    let num_buffers = 32;
    let samples_per_buffer = 32 * 1024;
    let iq_pairs_per_buffer = samples_per_buffer / 2;
    let num_transfers = 16;
    let timeout_ms = 1000_u32.ms();

    let fft_num_bins = 4096 / 2;
    assert!(samples_per_buffer % fft_num_bins == 0);
    assert!(fft_num_bins % 2 == 0);

    let mut complex_storage = ComplexStorage::new(iq_pairs_per_buffer);

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(fft_num_bins);

    let mut complex_samples: Vec<Complex<f64>> = vec![Complex::zero(); fft_num_bins];
    let mut plot_data = vec![0_f64; fft_num_bins];

    let filter_width = 10;
    let mut filters: Vec<MedianFilter<f64>> = vec![MedianFilter::new(filter_width); fft_num_bins];

    let bandwidth = opts.bandwidth.as_f64();
    let bandwidth_mhz = bandwidth / units::ONE_MHZ.as_f64();
    let center_freq_mhz = opts.frequency.as_f64() / units::ONE_MHZ.as_f64();
    let freq_start = center_freq_mhz - (bandwidth_mhz / 2.0);
    let freq_step = bandwidth_mhz / (fft_num_bins as f64);
    log::info!(
        "FFT bin size {:.02} Hz ({} bins)",
        bandwidth / (fft_num_bins as f64),
        fft_num_bins
    );

    // TODO - use BinnedFrequencyRange, show the center freq of the bin
    //let freq_range = BinnedFrequencyRange::new(r, opts.bandwidth).unwrap();
    let freq_bins: Vec<f64> = (0..fft_num_bins)
        .map(|index| freq_start + (index as f64 * freq_step))
        .collect();

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

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();

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

    let mut window: PistonWindow = WindowSettings::new("BladeRF Plot", [640, 480])
        .samples(4) // Anti-aliasing
        .build()
        .unwrap();
    window.set_max_fps(60);

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

    log::info!(
        "FFT bin size {:.02} Hz ({} bins)",
        bandwidth / (fft_num_bins as f64),
        fft_num_bins
    );

    dev.enable_module(channel, true)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();
    log::info!("Channel {} is active", channel);

    let mut device = DeviceReader::new(dev, samples_per_buffer);

    while let Some(_) = draw_piston_window(&mut window, |b| {
        if let Some(samples) = device.read() {
            complex_storage.push_normalize_sc16_q11(samples);
        }

        if complex_storage.len() < fft_num_bins {
            log::warn!(
                "Not enough samples to process - continue {} {}",
                complex_storage.len(),
                fft_num_bins
            );
            return Ok(());
        }

        while complex_storage.len() > fft_num_bins {
            debug_assert_eq!(complex_samples.len(), fft_num_bins);

            // Ordered by ascending frequency, with the first
            // element corresponding to frequency 0
            fft.process(
                &mut complex_storage.buffer_mut()[..fft_num_bins],
                &mut complex_samples,
            );
            complex_storage.drain(fft_num_bins);

            // TODO - scale and offset tweaks
            let scale = 1_f64 / fft_num_bins as f64;
            //let scale = (1_f64 / num_iq_pairs as f64).sqrt();

            complex_samples.vec_scale(scale);

            complex_samples.vec_mirror();

            complex_samples
                .iter()
                .zip(plot_data.iter_mut())
                .for_each(|(s, d)| *d = sample_to_db(s));

            apply_filter(&mut filters, &mut plot_data);
        }

        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let mut cc = ChartBuilder::on(&root)
            .margin(10)
            .caption("BladeRF Plot", ("sans-serif", 30).into_font())
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_ranged(0..fft_num_bins, -100.1_f64..0.0_f64)?;

        cc.configure_mesh()
            .x_label_formatter(&|x| format!("{:.02} MHz", freq_bins[*x]))
            //.y_label_formatter(&|y| format!("{}%", (*y * 100.0) as u32))
            .x_labels(16)
            .y_labels(20)
            .x_desc("Frequency")
            .y_desc("dB")
            .axis_desc_style(("sans-serif", 15).into_font())
            .draw()?;

        cc.draw_series(LineSeries::new(
            (0..).zip(plot_data.iter()).map(|(a, b)| (a, *b)),
            &Palette99::pick(0),
        ))?;
        //.label(format!("FFT,avg={}", filter_width))
        //.legend(move |(x, y)| {
        //    Rectangle::new([(x - 5, y - 5), (x + 5, y + 5)], &Palette99::pick(0))
        //});

        // Legend/labels
        //cc.configure_series_labels()
        //    .background_style(&WHITE.mix(0.8))
        //    .border_style(&BLACK)
        //    .draw()?;

        Ok(())
    }) {}

    let mut dev = device.into_inner();

    log::info!("Closing device");

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();
    dev.close();

    Ok(())
}

fn apply_filter<F>(filters: &mut [F], data: &mut [f64])
where
    F: Filter<f64, Output = f64>,
{
    debug_assert_eq!(filters.len(), data.len());
    filters
        .iter_mut()
        .zip(data.iter_mut())
        .for_each(|(f, d)| *d = f.filter(*d));
}

fn sample_to_db(s: &Complex<f64>) -> f64 {
    let power = s.norm_sqr();
    10.0 * power.log(10.0)
}

//fn sample_to_amplitude(s: &Complex<f64>) -> f64 {
//    let power = s.norm_sqr();
//    power.sqrt()
//}
