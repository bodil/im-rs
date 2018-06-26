// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate rustc_version;

use rustc_version::{version, version_meta, Channel, Version};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match version_meta().unwrap().channel {
        Channel::Nightly => {
            println!("cargo:rustc-cfg=has_specialisation");
            println!("cargo:rustc-cfg=has_range_bound");
        }
        _ => (),
    }
    if version().unwrap() >= Version::parse("1.28.0").unwrap() {
        println!("cargo:rustc-cfg=has_range_bound");
    }
}
