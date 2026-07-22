//! Wave 3 — vlen STRING ATTRIBUTES surfaced through the netCDF layer.
//!
//! netCDF-C writes string attributes in two on-disk forms:
//!
//! * A *default* Python string attribute (`var.units = 'kelvin'`) becomes a
//!   **fixed-length** `NC_CHAR` array.
//! * `var.setncattr_string(name, value)` (and the `NC_STRING` type generally)
//!   becomes a **variable-length** string stored via the global heap.
//!
//! [`NcAttribute::as_text`] must return the string for both forms.  The vlen
//! form is eagerly decoded at resolve time (`new_with_view` → `AttrView::as_strings`);
//! a multi-byte value additionally exercises the multi-object global-heap
//! collection path (netCDF-C packs several string attributes into one GCOL,
//! placed at a file offset that is not 8-byte aligned).
//!
//! Skips gracefully when `python3` / `netCDF4` / `numpy` is absent (mirrors
//! `crates/oxinetcdf/tests/fix_ncr_netcdf4.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use oxinetcdf::NcFile;

enum Authored {
    Ok,
    Skip(String),
}

fn author_netcdf4(script: &str) -> Authored {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output();
    match output {
        Ok(out) if out.status.success() => Authored::Ok,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                Authored::Skip(format!("netCDF4/numpy not available: {}", stderr.trim()))
            } else {
                panic!(
                    "netCDF4 authoring script FAILED:\nstdout: {}\nstderr: {stderr}",
                    String::from_utf8_lossy(&out.stdout)
                );
            }
        }
        Err(_) => Authored::Skip("python3 not found".to_string()),
    }
}

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "oxinetcdf_wave3_vlenattr_{}_{name}.nc",
        std::process::id()
    ))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Default (fixed NC_CHAR) and NC_STRING (vlen) attributes on a coordinate
// variable, plus a global NC_STRING attribute.  The multi-byte value forces a
// multi-object global-heap collection at an unaligned base.
// ---------------------------------------------------------------------------

#[test]
fn w3_netcdf4_vlen_and_fixed_string_attrs() {
    let path = tmp_path("attrs");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('t', 3)\n\
         tv = d.createVariable('t', 'f8', ('t',)); tv[:] = [0., 1., 2.]\n\
         tv.units = 'seconds'\n\
         tv.setncattr_string('long_name', 'time-axis')\n\
         tv.setncattr_string('note', 'café温度')\n\
         tv.setncattr_string('empty', '')\n\
         d.setncattr_string('title', 'global-vlen')\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping w3_netcdf4_vlen_and_fixed_string_attrs: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let nc = NcFile::open(&path).expect("open netCDF4 file");
    let root = nc.root_group().expect("root_group");
    cleanup(&path);

    // Global NC_STRING attribute.
    assert_eq!(
        root.attr("title").unwrap().as_text().unwrap(),
        "global-vlen",
        "global NC_STRING attribute"
    );

    let t = root
        .variable("t")
        .unwrap_or_else(|| panic!("variable t missing; vars={:?}", root.variables));

    // Default str attr -> fixed-length NC_CHAR.
    assert_eq!(
        t.attr("units").unwrap().as_text().unwrap(),
        "seconds",
        "default (fixed NC_CHAR) string attribute"
    );
    // setncattr_string -> vlen NC_STRING.
    assert_eq!(
        t.attr("long_name").unwrap().as_text().unwrap(),
        "time-axis",
        "vlen NC_STRING attribute"
    );
    // Multi-byte vlen NC_STRING: object index >= 2 in a GCOL placed at a
    // non-8-aligned file offset — the global-heap alignment regression.
    assert_eq!(
        t.attr("note").unwrap().as_text().unwrap(),
        "café温度",
        "multi-byte vlen NC_STRING attribute (multi-object heap)"
    );
    // Empty vlen NC_STRING.
    assert_eq!(
        t.attr("empty").unwrap().as_text().unwrap(),
        "",
        "empty vlen NC_STRING attribute"
    );
}
