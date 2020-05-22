use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=dylib=bladeRF");
    println!("cargo:rerun-if-changed=bindgen/wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("bindgen/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .whitelist_function("bladerf_.*")
        .whitelist_type("BLADERF_.*")
        .whitelist_var("BLADERF_.*")
        .no_copy("bladerf")
        .no_copy("bladerf_stream")
        .rustified_non_exhaustive_enum("bladerf_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
