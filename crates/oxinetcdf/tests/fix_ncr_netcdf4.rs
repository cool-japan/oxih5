//! Lane NCR — the oxinetcdf reader must handle genuine netCDF-4 files.
//!
//! B012: files written by netCDF-C round-trip empty / with phantom dims through
//!       oxinetcdf's reader.  These tests author real netCDF-4 files with
//!       `python3` + `netCDF4` at test time and assert the reader reports the
//!       exact dimensions, variables, attributes, and values — no phantoms,
//!       nothing empty.
//! B014: `NcAttribute::as_i64` / `as_f64` must not append a phantom trailing
//!       element for scalar integer/float attributes.  Verified here against a
//!       libhdf5-written scalar `_FillValue` and `scale_factor`.
//!
//! Every netCDF4-facing test skips gracefully when `python3` or the `netCDF4`
//! module is absent (mirrors the h5py graceful-skip guard in
//! `crates/oxih5/tests/write_tests.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use oxinetcdf::NcFile;

/// Outcome of trying to author a file with `python3 -c <netCDF4 script>`.
enum Authored {
    /// The file was written; proceed with the Rust-side assertions.
    Ok,
    /// `python3` or the `netCDF4`/`numpy` module is unavailable — skip the test.
    Skip(String),
}

/// Run a netCDF4-python authoring script.  Returns [`Authored::Skip`] when the
/// toolchain is absent, and panics only when python3 *is* present with netCDF4
/// but the script itself failed (a real, non-environmental error).
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
    std::env::temp_dir().join(format!("oxinetcdf_fix_ncr_{name}.nc"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// A dimension named `name` with `len`/`unlimited` must exist exactly once.
fn assert_dim(root: &oxinetcdf::NcGroup, name: &str, len: u64, unlimited: bool) {
    let matches: Vec<_> = root.dimensions.iter().filter(|d| d.name == name).collect();
    assert_eq!(
        matches.len(),
        1,
        "dimension {name:?} should appear exactly once, dims={:?}",
        root.dimensions
    );
    let d = matches[0];
    assert_eq!(d.len, len, "dimension {name:?} length");
    assert_eq!(
        d.is_unlimited, unlimited,
        "dimension {name:?} unlimited flag"
    );
}

fn var<'a>(root: &'a oxinetcdf::NcGroup, name: &str) -> &'a oxinetcdf::NcVariable {
    root.variable(name)
        .unwrap_or_else(|| panic!("variable {name:?} missing; vars={:?}", root.variables))
}

// ---------------------------------------------------------------------------
// B012 — full acceptance case: 2 named dims (one unlimited) + 1-D and 2-D
// variables + global and per-variable attributes.
// ---------------------------------------------------------------------------

#[test]
fn b012_netcdf4_full_roundtrip() {
    let path = tmp_path("acceptance");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('time', None)\n\
         d.createDimension('x', 5)\n\
         xv = d.createVariable('x', 'f8', ('x',)); xv[:] = [0.,1.,2.,3.,4.]; xv.long_name = 'x coordinate'\n\
         s = d.createVariable('series', 'i4', ('time',)); s[0:3] = [7,8,9]; s.units = 'counts'\n\
         f = d.createVariable('field', 'f4', ('time','x')); f[0:3,:] = np.arange(15,dtype='f4').reshape(3,5); f.units = 'm'; f.scale_factor = np.float32(2.0)\n\
         d.institution = 'COOLJAPAN'; d.answer = np.int32(42)\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping b012_netcdf4_full_roundtrip: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let nc = NcFile::open(&path).expect("open netCDF4 file");
    let root = nc.root_group().expect("root_group");
    cleanup(&path);

    // Dimensions: EXACTLY time(unlim,3) and x(5) — no phantom axes.
    assert_eq!(
        root.dimensions.len(),
        2,
        "expected exactly 2 dimensions, got {:?}",
        root.dimensions
    );
    assert!(
        !root.dimensions.iter().any(|d| d.name.starts_with("phony")),
        "phantom dimension leaked: {:?}",
        root.dimensions
    );
    assert_dim(&root, "time", 3, true);
    assert_dim(&root, "x", 5, false);

    // Variables: EXACTLY x, series, field (the pure/unlimited placeholders must
    // NOT be surfaced as variables).
    assert_eq!(
        root.variables.len(),
        3,
        "expected exactly 3 variables, got {:?}",
        root.variables.iter().map(|v| &v.name).collect::<Vec<_>>()
    );

    let x = var(&root, "x");
    assert!(x.is_coordinate, "x should be a coordinate variable");
    assert_eq!(x.shape, vec![5]);
    assert_eq!(x.dim_names(), vec!["x"]);
    assert_eq!(
        x.attr("long_name").unwrap().as_text().unwrap(),
        "x coordinate"
    );
    assert_eq!(x.read_f64(&nc).unwrap(), vec![0., 1., 2., 3., 4.]);

    let series = var(&root, "series");
    assert!(!series.is_coordinate);
    assert_eq!(series.shape, vec![3]);
    assert_eq!(series.dim_names(), vec!["time"]);
    assert_eq!(series.nc_type(), oxinetcdf::NcType::Int32);
    assert_eq!(series.attr("units").unwrap().as_text().unwrap(), "counts");

    let field = var(&root, "field");
    assert_eq!(field.shape, vec![3, 5]);
    assert_eq!(field.dim_names(), vec!["time", "x"]);
    assert_eq!(field.nc_type(), oxinetcdf::NcType::Float32);
    assert_eq!(field.attr("units").unwrap().as_text().unwrap(), "m");
    // B014: a scalar f32 attribute decodes to exactly one element.
    assert_eq!(
        field.attr("scale_factor").unwrap().as_f64().unwrap(),
        vec![2.0],
        "scalar f32 scale_factor must not gain a phantom trailing 0.0"
    );

    // Global attributes present and correctly typed.
    assert_eq!(
        root.attr("institution").unwrap().as_text().unwrap(),
        "COOLJAPAN"
    );
    // B014: a scalar int32 global attribute decodes to exactly one element.
    assert_eq!(
        root.attr("answer").unwrap().as_i64().unwrap(),
        vec![42],
        "scalar int32 attr must not gain a phantom trailing 0"
    );
}

// ---------------------------------------------------------------------------
// B012 — pure dimension (no coordinate variable) + coordinate variables +
// unlimited dimension: the pure dim must be a dimension, never a variable, and
// a 2-D data variable must bind to [coord-dim, pure-dim].
// ---------------------------------------------------------------------------

#[test]
fn b012_netcdf4_pure_dim_and_coords() {
    let path = tmp_path("purecoord");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('time', None)\n\
         d.createDimension('lat', 4)\n\
         d.createDimension('lon', 8)\n\
         lat = d.createVariable('lat', 'f8', ('lat',)); lat[:] = [10.,20.,30.,40.]; lat.units = 'degrees_north'\n\
         temp = d.createVariable('temp', 'f4', ('lat','lon')); temp[:] = np.arange(32,dtype='f4').reshape(4,8); temp.units='K'\n\
         t = d.createVariable('time', 'f8', ('time',)); t[0:3] = [1.,2.,3.]\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping b012_netcdf4_pure_dim_and_coords: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let nc = NcFile::open(&path).expect("open netCDF4 file");
    let root = nc.root_group().expect("root_group");
    cleanup(&path);

    // Three dimensions, no phantoms.  'lon' is a *pure* dimension whose
    // placeholder dataset has an undefined data address; it must still be sized
    // (from the NAME sentinel) and reported.
    assert_eq!(
        root.dimensions.len(),
        3,
        "expected exactly 3 dimensions, got {:?}",
        root.dimensions
    );
    assert!(!root.dimensions.iter().any(|d| d.name.starts_with("phony")));
    assert_dim(&root, "time", 3, true);
    assert_dim(&root, "lat", 4, false);
    assert_dim(&root, "lon", 8, false);

    // Variables: lat, temp, time — 'lon' is pure and must NOT be a variable.
    assert!(
        root.variable("lon").is_none(),
        "pure dimension 'lon' must not be surfaced as a variable"
    );
    assert_eq!(
        root.variables.len(),
        3,
        "expected exactly 3 variables, got {:?}",
        root.variables.iter().map(|v| &v.name).collect::<Vec<_>>()
    );

    let temp = var(&root, "temp");
    assert_eq!(temp.shape, vec![4, 8]);
    assert_eq!(
        temp.dim_names(),
        vec!["lat", "lon"],
        "temp must bind to [lat, lon], not phantom dims"
    );

    let lat = var(&root, "lat");
    assert!(lat.is_coordinate);
    assert_eq!(lat.dim_names(), vec!["lat"]);
    assert_eq!(lat.read_f64(&nc).unwrap(), vec![10., 20., 30., 40.]);

    let time = var(&root, "time");
    assert!(time.is_coordinate);
    assert_eq!(time.dim_names(), vec!["time"]);
    assert_eq!(time.read_f64(&nc).unwrap(), vec![1., 2., 3.]);
}

// ---------------------------------------------------------------------------
// B014 — a libhdf5-written scalar integer _FillValue must decode to exactly one
// element through NcAttribute::as_i64 (no phantom trailing 0).
// ---------------------------------------------------------------------------

#[test]
fn b014_scalar_int_fillvalue_no_phantom() {
    let path = tmp_path("fillvalue");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('n', 3)\n\
         v = d.createVariable('q', 'i4', ('n',), fill_value=np.int32(-9999)); v[:] = [1,2,3]\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping b014_scalar_int_fillvalue_no_phantom: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let nc = NcFile::open(&path).expect("open netCDF4 file");
    let root = nc.root_group().expect("root_group");
    cleanup(&path);

    let q = var(&root, "q");
    let fill = q
        .fill_value()
        .expect("q should carry a _FillValue attribute");
    assert_eq!(
        fill.as_i64().unwrap(),
        vec![-9999],
        "scalar int32 _FillValue must decode to exactly [-9999], not [-9999, 0]"
    );
}
