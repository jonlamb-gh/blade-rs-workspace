use bincode::deserialize_from;
use blade_logfile::{Header, Packet};
use std::fs::File;
use std::path::PathBuf;
use structopt::StructOpt;
//use libbladerf_sys::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "blade-logfile", about = "BladeRF logfile utilities")]
pub struct Opts {
    /// Input file
    #[structopt(short = "i", long, parse(from_os_str))]
    input_path: PathBuf,

    /// Print the samples in each packet
    #[structopt(long)]
    show_samples: bool,
}

fn main() -> Result<(), bincode::Error> {
    let opts = Opts::from_args();

    println!("Opening '{}'", opts.input_path.display());
    let log_file = File::open(opts.input_path)?;

    let header: Header = deserialize_from(&log_file)?;
    println!("Header");
    println!("  preamble: {}", header.preamble);
    println!("  version: {}", header.version);
    println!("  frequency: {}", header.frequency);
    println!("  sample_rate: {}", header.sample_rate);
    println!("  bandwidth: {}", header.bandwidth);
    println!("  channel: {}", header.channel);
    println!("  layout: {}", header.layout);
    println!("  format: {}", header.format);
    println!("  system_time: {}", header.system_time);
    header.check_preamble().expect("Bad preamble");
    header.check_version().expect("Bad version");

    let mut total_packets: u128 = 0;
    let mut total_samples: u128 = 0;

    loop {
        let packet_result: bincode::Result<Packet> = deserialize_from(&log_file);
        if is_eof(&packet_result) {
            break;
        }

        let packet = packet_result?;

        total_packets = total_packets.wrapping_add(1);
        total_samples = total_samples.wrapping_add(packet.samples.len() as _);

        println!(
            "Packet #{}, timestamp: {}, num_samples: {}",
            total_packets,
            packet.timestamp,
            packet.samples.len()
        );

        if opts.show_samples {
            for pair in packet.samples.chunks(2) {
                let i = pair[0];
                let q = pair[1];
                println!("  {}, {}", i, q);
            }
        }
    }

    println!("Total packets: {}", total_packets);
    println!("Total samples: {}", total_samples);

    Ok(())
}

fn is_eof(r: &bincode::Result<Packet>) -> bool {
    match r {
        Ok(_) => false,
        Err(e) => match **e {
            bincode::ErrorKind::Io(ref io_err) => match io_err.kind() {
                std::io::ErrorKind::UnexpectedEof => true,
                _ => false,
            },
            _ => false,
        },
    }
}
