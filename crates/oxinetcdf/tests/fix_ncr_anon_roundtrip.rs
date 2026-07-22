//! Lane NCR — anonymous dimension scales and an oxinetcdf write→read round-trip.
//!
//! * `b012_anonymous_dimension_scale` (h5py): a `DIMENSION_SCALE` dataset with an
//!   empty `NAME` (an *anonymous* scale, as `h5py.Dataset.make_scale()` writes it)
//!   must be reported as a dimension — never as a variable — and a data variable
//!   attached to it via the `H5T_VLEN{H5T_REFERENCE}` `DIMENSION_LIST` must bind
//!   to that dimension.  This is the case that exercises the vlen-of-reference
//!   `DIMENSION_LIST` decode path (genuine netCDF-4 files instead carry
//!   `_Netcdf4Coordinates`).
//! * `ncr_write_read_roundtrip`: locks the writer↔reader pairing.  Ignored while
//!   the NCW lane is still editing the pure-dimension writer semantics
//!   (B010/B011); the convergence pass un-ignores it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxinetcdf::{NcFile, NcFileWriter, NcType, VarOrGroup};

/// Author `path` with an h5py script; skip gracefully when python3/h5py/numpy
/// is unavailable, panic only on a real (non-environmental) script error.
fn author_h5py(script: &str) -> Option<()> {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output();
    match output {
        Ok(out) if out.status.success() => Some(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                eprintln!("h5py/numpy not available — skipping: {}", stderr.trim());
                None
            } else {
                panic!(
                    "h5py authoring FAILED:\nstdout: {}\nstderr: {stderr}",
                    String::from_utf8_lossy(&out.stdout)
                );
            }
        }
        Err(_) => {
            eprintln!("python3 not found — skipping");
            None
        }
    }
}

#[test]
fn b012_anonymous_dimension_scale() {
    let path = std::env::temp_dir().join("oxinetcdf_fix_ncr_anon.h5");
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File({p:?}, 'w')\n\
         s = f.create_dataset('anon', data=np.arange(4, dtype='i4'))\n\
         s.make_scale()\n\
         v = f.create_dataset('v', data=np.zeros((4,), dtype='f8'))\n\
         v.dims[0].attach_scale(s)\n\
         f.close()\n",
        p = path.display()
    );
    if author_h5py(&script).is_none() {
        return;
    }

    let nc = NcFile::open(&path).expect("open h5py file");
    let root = nc.root_group().expect("root_group");
    let _ = std::fs::remove_file(&path);

    // The anonymous scale is a dimension, sized from its dataset, and is NOT a
    // variable.
    assert_eq!(
        root.dimensions.len(),
        1,
        "expected exactly one dimension, got {:?}",
        root.dimensions
    );
    let anon = &root.dimensions[0];
    assert_eq!(anon.name, "anon");
    assert_eq!(anon.len, 4);
    assert!(
        root.variable("anon").is_none(),
        "an anonymous dimension scale must not be surfaced as a variable"
    );

    // The data variable binds to the anonymous dimension via the vlen-of-ref
    // DIMENSION_LIST — not a phantom axis.
    let v = root.variable("v").expect("variable 'v' missing");
    assert_eq!(v.shape, vec![4]);
    assert_eq!(
        v.dim_names(),
        vec!["anon"],
        "variable must bind to the anonymous scale, got {:?}",
        v.dim_names()
    );
}

#[test]
fn ncr_write_read_roundtrip() {
    let path = std::env::temp_dir().join("oxinetcdf_fix_ncr_roundtrip.nc");

    let mut nc = NcFileWriter::new();
    let lat = nc.def_dim("lat", 4).expect("def_dim lat");
    let lon = nc.def_dim("lon", 8).expect("def_dim lon");
    let temp = nc
        .def_var("temp", &[lat, lon], NcType::Float64)
        .expect("def_var temp");
    let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
    nc.put_var_f64(temp, &data).expect("put_var_f64");
    nc.put_att_str(VarOrGroup::Root, "title", "roundtrip")
        .expect("global attr");
    nc.put_att_str(VarOrGroup::Var(temp), "units", "K")
        .expect("var attr");
    nc.close(&path).expect("close");

    let read = NcFile::open(&path).expect("open");
    let root = read.root_group().expect("root_group");
    let _ = std::fs::remove_file(&path);

    // Dimensions survive with the right lengths.
    assert!(root
        .dimensions
        .iter()
        .any(|d| d.name == "lat" && d.len == 4));
    assert!(root
        .dimensions
        .iter()
        .any(|d| d.name == "lon" && d.len == 8));

    // The data variable round-trips with its axes bound (not phantom) and its
    // attributes intact.
    let temp_var = root.variable("temp").expect("temp variable missing");
    assert_eq!(temp_var.shape, vec![4, 8]);
    assert_eq!(temp_var.dim_names(), vec!["lat", "lon"]);
    assert_eq!(temp_var.attr("units").unwrap().as_text().unwrap(), "K");
    assert_eq!(root.attr("title").unwrap().as_text().unwrap(), "roundtrip");
}
