extern crate rustc_version;

use rustc_version::{version_meta, Channel};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_meta().unwrap().channel {
        Channel::Nightly => println!("cargo:rustc-cfg=has_specialisation"),
        _ => (),
    }
}
