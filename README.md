# BladeRF Rust Workspace

* [BladeRF-xA4](https://www.nuand.com/product/bladeRF-xA4/)

Check out these:
* http://kmkeen.com/rtl-power/
* https://crates.io/crates/sdr-heatmap
* https://crates.io/crates/sonogram
* https://crates.io/crates/rustfft
* https://crates.io/crates/dft
* https://crates.io/crates/hertz
* https://crates.io/crates/snap
* https://crates.io/crates/lttb
* https://crates.io/crates/running-average
* https://crates.io/crates/ta
* https://crates.io/crates/median
* https://crates.io/crates/signalo
* https://crates.io/crates/aether_primitives

## blade-log

...

## blade-logfile

...

## blade-plot

...

## blade-power-log

something like rtl-power, renders with sdr-heatmap

https://github.com/steve-m/librtlsdr/blob/master/src/rtl_power.c
https://osmocom.org/projects/rtl-sdr/repository/revisions/master/entry/src/rtl_power.c

https://dsp.stackexchange.com/questions/19615/converting-raw-i-q-to-db

https://www.tek.com/blog/calculating-rf-power-iq-samples

http://whiteboard.ping.se/SDR/IQ

https://github.com/f4exb/sdrangel/blob/da06bc30cc4c5e95cf3adccc9e2ee7b2a5f60ad4/sdrbase/dsp/spectrumvis.cpp

https://github.com/stephenong/ViewRF
https://github.com/stephenong/ViewRF/blob/master/spectrumplot.cpp
https://github.com/stephenong/ViewRF/blob/master/dialog.cpp

TODO
DC correction
scale/offset
display utils, format as f64 in best-fit units
plot modes, db, amp, power, rms-power, etc
utils lib, things like FreqRange/Hops, downsample, etc
