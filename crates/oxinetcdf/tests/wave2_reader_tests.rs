//! Wave-2 LANE READER — netCDF-4 reader completeness (netCDF4-python authored).
//!
//! * **Item 1 (root-link enumeration on netCDF-C files).** netCDF-C writes
//!   version-2 object headers that reserve space with a NIL message and then
//!   append Link / Attribute messages after it. A reader that stops at the first
//!   NIL drops every object past the gap, so a file with more than two root
//!   objects loses variables/dimensions. These tests author genuine netCDF-4
//!   files (`>2` root objects, and the pure-dimension `temp(lat,lon)` case with
//!   no coordinate variables) and assert every variable and dimension resolves.
//! * **Item 3 (narrow-width variable reads).** netCDF-4 stores `i4`/`f4`/`i2`/
//!   `u1` variables at their native width; `read_f64`/`read_i64` must widen them
//!   and the new `read_f32`/`read_i32` must read them at native width, all with
//!   exact values.
//!
//! Every test skips gracefully when `python3`/`netCDF4` is unavailable.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use oxinetcdf::NcFile;

enum Authored {
    Ok,
    Skip(String),
}

fn author_netcdf4(script: &str) -> Authored {
    match std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
    {
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
    std::env::temp_dir().join(format!("oxinetcdf_wave2_{name}.nc"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn var<'a>(root: &'a oxinetcdf::NcGroup, name: &str) -> &'a oxinetcdf::NcVariable {
    root.variable(name)
        .unwrap_or_else(|| panic!("variable {name:?} missing; vars={:?}", root.variables))
}

/// Item 1: pure-dimension file `temp(lat,lon)` with NO coordinate variables
/// (the `ref2.nc` case). The root object header stores `lon`/`temp` links after
/// a NIL gap; the reader must enumerate all of them so `temp` resolves with its
/// real `lat=4`/`lon=8` axes instead of vanishing or gaining phantom dims.
#[test]
fn item1_puredim_netcdf_c_file_resolves_all_axes() {
    let path = tmp_path("puredim");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('lat', 4)\n\
         d.createDimension('lon', 8)\n\
         t = d.createVariable('temp', 'f4', ('lat','lon'))\n\
         t[:] = np.zeros((4,8), dtype='f4')\n\
         d.title = 'ref2 pure-dim file'\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping item1_puredim: {why}");
            return;
        }
        Authored::Ok => {}
    }

    // The underlying HDF5 root group must enumerate all three datasets, not just
    // the first one before the NIL gap.
    let h5 = oxih5::File::open(&path).unwrap();
    let mut names = h5.group("/").unwrap().datasets().unwrap();
    names.sort();
    assert_eq!(
        names,
        vec!["lat".to_string(), "lon".to_string(), "temp".to_string()],
        "root-link enumeration dropped datasets after the NIL gap"
    );

    // ...and the netCDF layer resolves temp's real axes.
    let nc = NcFile::open(&path).unwrap();
    let root = nc.root_group().unwrap();
    cleanup(&path);

    let t = var(&root, "temp");
    assert_eq!(t.shape, vec![4, 8]);
    assert_eq!(t.dims.len(), 2);
    assert_eq!(t.dims[0].name, "lat");
    assert_eq!(t.dims[0].len, 4);
    assert_eq!(t.dims[1].name, "lon");
    assert_eq!(t.dims[1].len, 8);
    assert!(
        !root.dimensions.iter().any(|d| d.name.starts_with("phony")),
        "phantom dimension leaked: {:?}",
        root.dimensions
    );
}

/// Item 1: a file with several coordinate-free data variables over one shared
/// dimension — more than two root objects — must surface every variable.
#[test]
fn item1_many_root_variables_all_enumerated() {
    let path = tmp_path("manyvars");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('n', 3)\n\
         for nm in ['alpha','beta','gamma','delta','epsilon']:\n\
         \x20   v = d.createVariable(nm, 'f8', ('n',)); v[:] = [1.0,2.0,3.0]\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping item1_many_root_variables: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let nc = NcFile::open(&path).unwrap();
    let root = nc.root_group().unwrap();
    cleanup(&path);

    for nm in ["alpha", "beta", "gamma", "delta", "epsilon"] {
        let v = var(&root, nm);
        assert_eq!(v.shape, vec![3], "variable {nm} shape");
    }
}

/// Item 3: `i4`/`f4`/`i2`/`u1` variables round-trip their exact values through
/// the widening `read_f64`/`read_i64` and the native-width `read_f32`/`read_i32`.
#[test]
fn item3_narrow_dtype_variables_round_trip_exactly() {
    let path = tmp_path("narrowtypes");
    let script = format!(
        "import numpy as np, netCDF4 as nc\n\
         d = nc.Dataset({p:?}, 'w', format='NETCDF4')\n\
         d.createDimension('x', 5)\n\
         d.createVariable('vi32','i4',('x',))[:] = [10,-20,30,-40,50]\n\
         d.createVariable('vf32','f4',('x',))[:] = [1.5,2.5,-3.5,4.25,5.125]\n\
         d.createVariable('vi16','i2',('x',))[:] = [1,2,3,4,5]\n\
         d.createVariable('vu8','u1',('x',))[:] = [200,100,50,25,12]\n\
         d.createVariable('vf64','f8',('x',))[:] = [1.0,2.0,3.0,4.0,5.0]\n\
         d.createVariable('vi64','i8',('x',))[:] = [100,200,300,400,500]\n\
         d.close()\n",
        p = path.display()
    );
    match author_netcdf4(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping item3_narrow_dtype: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let nc = NcFile::open(&path).unwrap();
    let root = nc.root_group().unwrap();

    // read_f64 widens every fixed-width numeric type exactly.
    assert_eq!(
        var(&root, "vi32").read_f64(&nc).unwrap(),
        vec![10.0, -20.0, 30.0, -40.0, 50.0]
    );
    assert_eq!(
        var(&root, "vf32").read_f64(&nc).unwrap(),
        vec![1.5, 2.5, -3.5, 4.25, 5.125]
    );
    assert_eq!(
        var(&root, "vu8").read_f64(&nc).unwrap(),
        vec![200.0, 100.0, 50.0, 25.0, 12.0]
    );

    // read_i64 widens narrower integers exactly.
    assert_eq!(
        var(&root, "vi32").read_i64(&nc).unwrap(),
        vec![10, -20, 30, -40, 50]
    );
    assert_eq!(
        var(&root, "vi16").read_i64(&nc).unwrap(),
        vec![1, 2, 3, 4, 5]
    );
    assert_eq!(
        var(&root, "vu8").read_i64(&nc).unwrap(),
        vec![200, 100, 50, 25, 12]
    );

    // Native-width readers.
    assert_eq!(
        var(&root, "vf32").read_f32(&nc).unwrap(),
        vec![1.5f32, 2.5, -3.5, 4.25, 5.125]
    );
    assert_eq!(
        var(&root, "vi32").read_i32(&nc).unwrap(),
        vec![10i32, -20, 30, -40, 50]
    );
    assert_eq!(
        var(&root, "vi16").read_i32(&nc).unwrap(),
        vec![1i32, 2, 3, 4, 5]
    );

    // The wide f64/i64 variables still read correctly.
    assert_eq!(
        var(&root, "vf64").read_f64(&nc).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(
        var(&root, "vi64").read_i64(&nc).unwrap(),
        vec![100, 200, 300, 400, 500]
    );

    cleanup(&path);
}
