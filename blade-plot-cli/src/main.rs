use byteorder::{LittleEndian, WriteBytesExt};
use chrono::prelude::*;
use libbladerf_sys::*;
use median::Filter;
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
use std::convert::TryFrom;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

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

    /// Mirror the raw I/Q data to another file
    ///
    /// Testing with baudline:
    ///
    /// mkfifo /tmp/rx_samples.bin
    ///
    /// baudline -reset -format le16 -channels 2 -quadrature
    ///   -samplerate 2000000 -stdin < /tmp/rx_samples.bin
    ///
    #[structopt(short = "m", long, parse(from_os_str))]
    mirror: Option<PathBuf>,
}

fn main() -> Result<(), bincode::Error> {
    env_logger::from_env(
        env_logger::Env::default().default_filter_or("info,gfx_device_gl=warn,winit=warn"),
    )
    .init();
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

    let mirror_file = if let Some(mirror) = opts.mirror {
        log::info!("Opening mirror file '{}'", mirror.display());
        let f = OpenOptions::new()
            .read(false)
            .write(true)
            .create(false)
            .open(mirror)?;
        Some(f)
    } else {
        None
    };

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
    log::info!("System time: {}", system_time);

    let mut device = DeviceReader::new(dev, samples_per_buffer);

    let fft_num_bins = 4096 / 2;
    assert!(samples_per_buffer % fft_num_bins == 0);
    assert!(fft_num_bins % 2 == 0);

    let mut complex_storage = ComplexStorage::new(samples_per_buffer / 2, fft_num_bins);

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(fft_num_bins);

    // plot data
    let mut amplitudes = vec![0_f64; fft_num_bins];

    let filter_width = 10;
    let mut filters: Vec<Filter<f64>> = vec![Filter::new(filter_width); fft_num_bins];

    // TODO - display utils in lib, f64 MHz
    let bandwidth = opts.bandwidth.0 as f64;
    let bandwidth_mhz = bandwidth / 1_000_000_f64;
    let center_freq_mhz = opts.frequency.0 as f64 / 1_000_000_f64;
    let freq_start = center_freq_mhz - (bandwidth_mhz / 2.0);
    let freq_step = bandwidth_mhz / (fft_num_bins as f64);
    log::info!("FFT bin size {:.02} Hz", bandwidth / (fft_num_bins as f64));
    // indexed on bin index
    let freq_bins: Vec<f64> = (0..fft_num_bins)
        .map(|index| freq_start + (index as f64 * freq_step))
        .collect();
    //let freq_step

    while let Some(_) = draw_piston_window(&mut window, |b| {
        if let Some(samples) = device.read() {
            // Mirror raw I/Q data
            if let Some(mut f) = mirror_file.as_ref() {
                for s in samples.iter() {
                    f.write_i16::<LittleEndian>(*s)?;
                }
            }

            complex_storage.normalize_append_iq(samples);
        }

        if complex_storage.input_len() < fft_num_bins {
            log::warn!(
                "Not enough samples to process - continue {} {}",
                complex_storage.input.len(),
                fft_num_bins
            );
            return Ok(());
        }

        while complex_storage.input_len() > fft_num_bins {
            assert_eq!(complex_storage.output.len(), fft_num_bins);
            complex_storage.compute_fft(&*fft);

            // TODO - scale and offset tweaks
            let scale = 1_f64 / fft_num_bins as f64;
            //let scale = (1_f64 / num_iq_pairs as f64).sqrt();
            //let scale = 1.0;

            process_iq_data(scale, complex_storage.output(), &mut amplitudes);

            apply_filter(&mut filters, &mut amplitudes);
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
            (0..).zip(amplitudes.iter()).map(|(a, b)| (a, *b)),
            &Palette99::pick(0),
        ))?
        .label(format!("FFT,avg={}", filter_width))
        .legend(move |(x, y)| {
            Rectangle::new([(x - 5, y - 5), (x + 5, y + 5)], &Palette99::pick(0))
        });

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

struct DeviceReader {
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

        assert!(num_samples % 2 == 0);

        Some(&self.sample_buffer[..num_samples])
    }
}

struct ComplexStorage {
    input: Vec<Complex<f64>>,
    output: Vec<Complex<f64>>,
}

impl ComplexStorage {
    pub fn new(initial_capacity: usize, chunk_size: usize) -> Self {
        assert!(initial_capacity >= chunk_size);
        ComplexStorage {
            output: vec![Complex::zero(); chunk_size],
            input: Vec::with_capacity(initial_capacity),
        }
    }

    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    pub fn normalize_append_iq(&mut self, samples: &[i16]) {
        assert!(samples.len() % 2 == 0);
        samples.chunks(2).for_each(|pair| {
            let (i, q) = (normalize_sample(pair[0]), normalize_sample(pair[1]));
            self.input.push(Complex::new(i, q));
        });
    }

    // Ordered by ascending frequency, with the first
    // element corresponding to frequency 0
    pub fn compute_fft(&mut self, fft: &dyn rustfft::FFT<f64>) {
        let num_bins = self.output.capacity();
        fft.process(&mut self.input[..num_bins], &mut self.output);
        self.input.drain(..num_bins);
    }

    pub fn output(&self) -> &[Complex<f64>] {
        &self.output
    }
}

// Converts i16, in the range [-2048, 2048) to [-1.0, 1.0).
// Note that the lower bound here is inclusive, and the upper bound is exclusive.
// Samples should always be within [-2048, 2047].
fn normalize_sample(s: i16) -> f64 {
    assert!(s >= -2048);
    assert!(s < 2048);
    f64::from(s) / 2048.0
}

// TODO - db/amp in opts
fn complex_to_amplitude(c: &Complex<f64>, scale: f64) -> f64 {
    let norm = c.scale(scale);
    let power = norm.norm_sqr();
    //let amplitude = power.sqrt();
    //amplitude
    let db = 10.0 * power.log(10.0);
    db
}

// Length == num fft bins
fn process_iq_data(scale: f64, iq_data: &[Complex<f64>], amplitudes: &mut [f64]) {
    assert_eq!(iq_data.len(), amplitudes.len());
    let half_size = iq_data.len() / 2;
    for i in 0..half_size {
        let src_index = i;
        let dst_index = i + half_size;
        amplitudes[dst_index] = complex_to_amplitude(&iq_data[src_index], scale);

        let src_index = i + half_size;
        let dst_index = i;
        amplitudes[dst_index] = complex_to_amplitude(&iq_data[src_index], scale);
    }
}

fn apply_filter(filters: &mut [Filter<f64>], amplitudes: &mut [f64]) {
    assert_eq!(filters.len(), amplitudes.len());
    filters
        .iter_mut()
        .zip(amplitudes.iter_mut())
        .for_each(|(f, a)| *a = f.consume(*a));
}
