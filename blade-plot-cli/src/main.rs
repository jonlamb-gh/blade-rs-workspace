use dsp::{
    hertz,
    median::Filter as MedianFilter,
    num_complex::Complex,
    num_traits::Zero,
    rustfft::{FFTplanner, FFT},
    sample_to_power, ComplexStorage, DeviceLimits, DeviceReader, Filter, VecOps,
};
use libbladerf_sys::*;
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::prelude::*;
use std::collections::vec_deque::VecDeque;
use std::io;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use structopt::StructOpt;

// TODO
// - reference level (dB), -10 default
// - range (dB), 90 default
// - plot mode, rms power, dB, amplitude

#[derive(Debug, Clone, StructOpt)]
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

    /// Print info and exit
    #[structopt(long)]
    dry_run: bool,

    /// Plot mode
    #[structopt(short = "p", long, parse(try_from_str = PlotMode::from_str), default_value = "fft")]
    plot_mode: PlotMode,

    /// Number of bins in the FFT
    #[structopt(long, default_value = "2048")]
    fft_bins: usize,

    /// Size (width) of the median average filter
    #[structopt(long, default_value = "10")]
    avg_window_width: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PlotMode {
    /// Run I/Q data through an FFT, plot in dB
    Fft,
    /// Run I/Q data through a median average filter, plot RMS power
    RmsPower,
}

#[derive(Debug)]
pub struct UnsupportedPlotMode(String);

impl ToString for UnsupportedPlotMode {
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

impl FromStr for PlotMode {
    type Err = UnsupportedPlotMode;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        match s.trim() {
            "fft" => Ok(PlotMode::Fft),
            "rms-power" => Ok(PlotMode::RmsPower),
            _ => Err(UnsupportedPlotMode(s)),
        }
    }
}

pub struct FftPlot {
    opts: Opts,
    device: DeviceReader,
    freq_bins: Vec<f64>,
    complex_storage: ComplexStorage,
    complex_samples: Vec<Complex<f64>>,
    fft: Arc<dyn FFT<f64>>,
    filters: Vec<MedianFilter<f64>>,
    plot_data: Vec<f64>,
}

impl FftPlot {
    pub fn new(opts: Opts, device: DeviceReader, initial_capacity: usize) -> Self {
        let fft_bins = opts.fft_bins;
        let avg_window_width = opts.avg_window_width;

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(fft_bins);

        let bandwidth = opts.bandwidth.as_f64();
        let bandwidth_mhz = bandwidth / units::ONE_MHZ.as_f64();
        let center_freq_mhz = opts.frequency.as_f64() / units::ONE_MHZ.as_f64();
        let freq_start = center_freq_mhz - (bandwidth_mhz / 2.0);
        let freq_step = bandwidth_mhz / (opts.fft_bins as f64);

        // TODO - use BinnedFrequencyRange, show the center freq of the bin
        let freq_bins: Vec<f64> = (0..opts.fft_bins)
            .map(|index| freq_start + (index as f64 * freq_step))
            .collect();

        FftPlot {
            opts,
            device,
            freq_bins,
            complex_storage: ComplexStorage::new(initial_capacity),
            complex_samples: vec![Complex::zero(); fft_bins],
            fft,
            filters: vec![MedianFilter::new(avg_window_width); fft_bins],
            plot_data: vec![0_f64; fft_bins],
        }
    }

    pub fn into_inner(self) -> Device {
        self.device.into_inner()
    }

    pub fn device_mut(&mut self) -> &mut Device {
        self.device.device_mut()
    }

    // Returns true when plot data is updated and should be redrawn
    pub fn process(&mut self) -> Result<bool, io::Error> {
        debug_assert_eq!(self.plot_data.len(), self.complex_samples.len());

        if let Some(samples) = self.device.read() {
            self.complex_storage.push_normalize_sc16_q11(samples);
        }

        if self.complex_storage.len() < self.opts.fft_bins {
            log::warn!(
                "Not enough samples to process - continue {} {}",
                self.complex_storage.len(),
                self.opts.fft_bins
            );
            return Ok(false);
        }

        let mut proc_counter: u64 = 0;
        while self.complex_storage.len() > self.opts.fft_bins {
            debug_assert_eq!(self.complex_samples.len(), self.opts.fft_bins);

            // Ordered by ascending frequency, with the first
            // element corresponding to frequency 0
            self.fft.process(
                &mut self.complex_storage.buffer_mut()[..self.opts.fft_bins],
                &mut self.complex_samples,
            );
            self.complex_storage.drain(self.opts.fft_bins);

            // TODO - scale and offset tweaks
            let scale = 1_f64 / self.opts.fft_bins as f64;
            //let scale = (1_f64 / num_iq_pairs as f64).sqrt();

            self.complex_samples.vec_scale(scale);

            self.complex_samples.vec_mirror();

            self.complex_samples
                .iter()
                .zip(self.plot_data.iter_mut())
                .for_each(|(s, d)| *d = sample_to_db(s));

            apply_filter(&mut self.filters, &mut self.plot_data);

            proc_counter = proc_counter.wrapping_add(1);
        }

        if proc_counter > 4 {
            log::warn!("Processed {} FFT windows before rendering", proc_counter);
        }

        Ok(true)
    }

    pub fn draw(
        &mut self,
        b: PistonBackend,
    ) -> Result<(), DrawingAreaErrorKind<DummyBackendError>> {
        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let mut cc = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                format!(
                    "FFT,avg={},num_bins={},center={}",
                    self.opts.avg_window_width, self.opts.fft_bins, self.opts.frequency
                ),
                ("sans-serif", 30).into_font(),
            )
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_ranged(0..self.opts.fft_bins, -100.1_f64..0.0_f64)?;

        cc.configure_mesh()
            .x_label_formatter(&|x| format!("{:.02} MHz", self.freq_bins[*x]))
            //.y_label_formatter(&|y| format!("{}%", (*y * 100.0) as u32))
            .x_labels(16)
            .y_labels(20)
            .x_desc("Frequency")
            .y_desc("dB")
            .axis_desc_style(("sans-serif", 15).into_font())
            .draw()?;

        cc.draw_series(LineSeries::new(
            (0..).zip(self.plot_data.iter()).map(|(a, b)| (a, *b)),
            &Palette99::pick(0),
        ))?;

        Ok(())
    }
}

// TODO - downsample
pub struct RmsPowerPlot {
    opts: Opts,
    device: DeviceReader,
    plot_width: usize,
    filter_width_sec: f64,
    complex_storage: ComplexStorage,
    filter: MedianFilter<f64>,
    plot_data: VecDeque<f64>,
}

impl RmsPowerPlot {
    pub fn new(opts: Opts, device: DeviceReader, initial_capacity: usize) -> Self {
        let avg_window_width = opts.avg_window_width;
        let plot_width = 4096 * 2; // TODO - config

        // TODO - this isn't correct
        let filter_width_sec = avg_window_width as f64 / (opts.sample_rate.as_f64() / 8.0);
        log::info!(
            "RMS power filter width: {:.03} ms, total time {:.03} s",
            filter_width_sec * 1000.0,
            plot_width as f64 * filter_width_sec
        );

        RmsPowerPlot {
            opts,
            device,
            plot_width,
            filter_width_sec,
            complex_storage: ComplexStorage::new(initial_capacity),
            filter: MedianFilter::new(avg_window_width),
            plot_data: VecDeque::from(vec![0_f64; plot_width]),
        }
    }

    pub fn into_inner(self) -> Device {
        self.device.into_inner()
    }

    pub fn device_mut(&mut self) -> &mut Device {
        self.device.device_mut()
    }

    // Returns true when plot data is updated and should be redrawn
    pub fn process(&mut self) -> Result<bool, io::Error> {
        if let Some(samples) = self.device.read() {
            // TODO - downsample
            self.complex_storage.push_normalize_sc16_q11(samples);
        }

        if self.complex_storage.len() < self.filter.len() {
            log::warn!(
                "Not enough samples to process - continue {} {}",
                self.complex_storage.len(),
                self.filter.len()
            );
            return Ok(false);
        }

        // TODO - fix this filtering
        let mut proc_counter: u64 = 0;
        while self.complex_storage.len() > self.filter.len() {
            for s in self.complex_storage.buffer()[..self.filter.len()].iter() {
                self.filter.consume(sample_to_power(s));
            }
            self.complex_storage.drain(self.filter.len());

            if self.plot_data.len() == self.plot_width + 1 {
                self.plot_data.pop_front();
            }
            self.plot_data.push_back(self.filter.median());

            proc_counter = proc_counter.wrapping_add(1);

            if proc_counter > (self.plot_width as u64 / 16) {
                // TODO
                log::warn!("break: remain {}", self.complex_storage.len());
                break;
            }
        }

        Ok(true)
    }

    pub fn draw(
        &mut self,
        b: PistonBackend,
    ) -> Result<(), DrawingAreaErrorKind<DummyBackendError>> {
        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let mut cc = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                format!(
                    "width={},center={},t_n={:.03} ms, t_total={:.03} s",
                    self.plot_width,
                    self.opts.frequency,
                    self.filter_width_sec * 1000.0,
                    self.plot_width as f64 * self.filter_width_sec
                ),
                ("sans-serif", 30).into_font(),
            )
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_ranged(0..self.plot_width, 0_f64..1.0_f64)?;

        cc.configure_mesh()
            .x_label_formatter(&|x| {
                format!(
                    "{:.03}",
                    -(self.plot_width as f64 * self.filter_width_sec)
                        + (*x as f64 * self.filter_width_sec)
                )
            })
            //.x_label_formatter(&|x| format!("{}", *x))
            //.y_label_formatter(&|y| format!("{}%", (*y * 100.0) as u32))
            .x_labels(16)
            .y_labels(20)
            .x_desc("Time (seconds)")
            .y_desc("Power")
            .axis_desc_style(("sans-serif", 15).into_font())
            .draw()?;

        cc.draw_series(LineSeries::new(
            (0..).zip(self.plot_data.iter()).map(|(a, b)| (a, *b)),
            &Palette99::pick(0),
        ))?;

        Ok(())
    }
}

fn main() -> Result<(), io::Error> {
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

    let channel = Channel::Rx0;
    let channel_layout = ChannelLayout::RxX1;
    let format = Format::Sc16Q11Meta;
    let num_buffers = 32;
    let samples_per_buffer = 32 * 1024;
    let iq_pairs_per_buffer = samples_per_buffer / 2;
    let num_transfers = 16;
    let timeout_ms = 1000_u32.ms();

    assert!(samples_per_buffer % opts.fft_bins == 0);
    assert!(opts.fft_bins % 2 == 0);

    log::info!("Channel: {}", channel);
    log::info!("Frequency: {}", opts.frequency);
    log::info!("Sample rate: {}", opts.sample_rate);
    log::info!("Bandwidth: {}", opts.bandwidth);
    log::info!("Channel layout: {}", channel_layout);
    log::info!("Format: {}", format);
    log::info!(
        "FFT bin size {:.02} Hz ({} bins)",
        opts.bandwidth.as_f64() / (opts.fft_bins as f64),
        opts.fft_bins
    );

    let rayleigh = hertz::rayleigh(opts.sample_rate.as_f64(), opts.fft_bins as f64);
    log::info!(
        "Rayleigh, min frequency (window size = {}): {:.01} Hz",
        opts.fft_bins,
        rayleigh
    );

    let nyquist = hertz::nyquist(opts.sample_rate.as_f64());
    log::info!(
        "Nyquist, max frequency: {:.03} MHz",
        nyquist / units::ONE_MHZ.as_f64()
    );

    DeviceLimits::check(opts.frequency, opts.bandwidth, opts.sample_rate)
        .map_err(|e| log::error!("DeviceLimits::check returned {:?}", e))
        .unwrap();

    if opts.dry_run {
        return Ok(());
    }

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

    dev.enable_module(channel, false)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();

    dev.set_frequency(channel, opts.frequency)
        .map_err(|e| log::error!("Device::set_frequency returned {:?}", e))
        .unwrap();

    let actual_sample_rate = dev
        .set_sample_rate(channel, opts.sample_rate)
        .map_err(|e| log::error!("Device::set_sample_rate returned {:?}", e))
        .unwrap();
    if opts.sample_rate != actual_sample_rate {
        log::warn!("Actual sample rate: {}", actual_sample_rate);
    }

    let actual_bandwidth = dev
        .set_bandwidth(channel, opts.bandwidth)
        .map_err(|e| log::error!("Device::set_bandwidth returned {:?}", e))
        .unwrap();
    if opts.bandwidth != actual_bandwidth {
        log::warn!("Actual bandwidth: {}", actual_bandwidth);
    }

    //let mut plot = FftPlot::new(
    let mut plot = RmsPowerPlot::new(
        opts.clone(),
        DeviceReader::new(dev, samples_per_buffer),
        iq_pairs_per_buffer,
    );

    let mut window: PistonWindow = WindowSettings::new("BladeRF Plot", [640, 480])
        .samples(4) // Anti-aliasing
        .build()
        .unwrap();
    window.set_max_fps(60);

    plot.device_mut()
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

    plot.device_mut()
        .enable_module(channel, true)
        .map_err(|e| log::error!("Device::enable_module returned {:?}", e))
        .unwrap();
    log::info!("Channel {} is active", channel);

    while let Some(_) = draw_piston_window(&mut window, |b| {
        let redraw = plot.process()?;
        if redraw {
            plot.draw(b)?;
        }
        Ok(())
    }) {
        if running.load(Ordering::SeqCst) != 0 {
            break;
        }
    }

    let mut dev = plot.into_inner();

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
    let power = sample_to_power(s);
    10.0 * power.log(10.0)
}
