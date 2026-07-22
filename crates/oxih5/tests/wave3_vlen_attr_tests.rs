//! Wave 3 — variable-length (vlen) STRING ATTRIBUTE reading.
//!
//! h5py writes a scalar string attribute (`dset.attrs['units'] = 'm'`) as an
//! HDF5 *variable-length* string (datatype class 9), **not** a fixed-length
//! one.  The reader must surface that string:
//!
//! * [`AttrView::as_str`] returns the single string for a scalar vlen (or
//!   fixed) string attribute — the accessor `as_str_fixed` returns `None` for
//!   the vlen form, which is the gap this wave closes (reverse differential
//!   fuzzing: `rev_e_groups` / `rev_l_groups` reported `@units str None != "m"`).
//! * [`AttrView::as_strings`] returns every element for 1-D arrays of vlen
//!   strings, resolving each `(global-heap addr, index)` reference — including
//!   empty strings, multi-byte UTF-8, and multi-object heap collections.
//!
//! oxih5's writer only emits *fixed-length* string attributes
//! (`FileWriter::write_string_attr`), so these are h5py-authored / oxih5-read
//! interop tests.  Each skips gracefully when `python3` or `h5py`/`numpy` is
//! absent (mirrors the guard in `crates/oxih5/tests/write_tests.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use oxih5::File;

/// Outcome of trying to author a file with `python3 -c <h5py script>`.
enum Authored {
    /// The file was written; proceed with the Rust-side assertions.
    Ok,
    /// `python3` or the `h5py`/`numpy` module is unavailable — skip the test.
    Skip(String),
}

/// Run an h5py authoring script.  Returns [`Authored::Skip`] when the toolchain
/// is absent, and panics only when `python3` *is* present with `h5py` but the
/// script itself failed (a real, non-environmental error).  A silent skip on a
/// genuine failure would make the test worthless, so only the "not installed"
/// shapes are tolerated.
fn author_h5py(script: &str) -> Authored {
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
                Authored::Skip(format!("h5py/numpy not available: {}", stderr.trim()))
            } else {
                panic!(
                    "h5py authoring script FAILED:\nstdout: {}\nstderr: {stderr}",
                    String::from_utf8_lossy(&out.stdout)
                );
            }
        }
        Err(_) => Authored::Skip("python3 not found".to_string()),
    }
}

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "oxih5_wave3_vlenattr_{}_{name}.h5",
        std::process::id()
    ))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Fetch a single named attribute view for `object_path`, panicking with a
/// helpful message if it is missing.
fn attr<'a>(f: &'a File, object_path: &str, name: &str) -> oxih5::AttrView<'a> {
    let views = f
        .attr_views(object_path)
        .unwrap_or_else(|e| panic!("attr_views({object_path}): {e}"));
    views
        .into_iter()
        .find(|v| v.name() == name)
        .unwrap_or_else(|| panic!("attribute {object_path}@{name} missing"))
}

// ---------------------------------------------------------------------------
// Scalar vlen string attribute — the h5py default for `dset.attrs['x'] = 'm'`.
// This is the exact reverse-diffuzz repro (`rev_*_groups @units str None`).
// ---------------------------------------------------------------------------

#[test]
fn w3_scalar_vlen_string_attr_reads() {
    for (tag, libver) in [("earliest", "earliest"), ("latest", "latest")] {
        let path = tmp_path(&format!("scalar_{tag}"));
        let script = format!(
            "import numpy as np, h5py\n\
             f = h5py.File({p:?}, 'w', libver={lv:?})\n\
             d = f.create_dataset('x', data=np.arange(4, dtype='<i4'))\n\
             d.attrs['units'] = 'm'\n\
             f.close()\n",
            p = path.display(),
            lv = libver
        );
        match author_h5py(&script) {
            Authored::Skip(why) => {
                eprintln!("skipping w3_scalar_vlen_string_attr_reads[{tag}]: {why}");
                return;
            }
            Authored::Ok => {}
        }

        let f = File::open(&path).expect("open h5py file");
        let units = attr(&f, "/x", "units");

        // The pre-fix gap: `as_str_fixed` cannot decode a vlen string, so the
        // reverse harness saw `None` here.
        assert_eq!(
            units.as_str_fixed(),
            None,
            "[{tag}] a vlen string is not a fixed-length string"
        );
        // The fix: `as_str` decodes the scalar vlen string end-to-end.
        assert_eq!(
            units.as_str(),
            Some("m".to_string()),
            "[{tag}] as_str must resolve the scalar vlen string"
        );
        assert_eq!(units.as_strings().unwrap(), vec!["m".to_string()]);
        cleanup(&path);
    }
}

// ---------------------------------------------------------------------------
// Empty + multi-byte UTF-8 scalar vlen string attributes.
// ---------------------------------------------------------------------------

#[test]
fn w3_scalar_vlen_string_attr_empty_and_utf8() {
    let path = tmp_path("empty_utf8");
    let script = format!(
        "import numpy as np, h5py\n\
         f = h5py.File({p:?}, 'w')\n\
         d = f.create_dataset('x', data=np.arange(2, dtype='<i4'))\n\
         d.attrs['empty'] = ''\n\
         d.attrs['utf8'] = 'café温度'\n\
         f.close()\n",
        p = path.display()
    );
    match author_h5py(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping w3_scalar_vlen_string_attr_empty_and_utf8: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let f = File::open(&path).expect("open h5py file");
    assert_eq!(attr(&f, "/x", "empty").as_str(), Some(String::new()));
    assert_eq!(
        attr(&f, "/x", "utf8").as_str(),
        Some("café温度".to_string())
    );
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// 1-D array of vlen strings (multi-object global-heap collection).
// ---------------------------------------------------------------------------

#[test]
fn w3_array_vlen_string_attr_reads() {
    let path = tmp_path("array");
    // "z"*40 forces a longer heap object; mixing lengths + an empty element +
    // multi-byte UTF-8 exercises a multi-object collection.
    let script = format!(
        "import numpy as np, h5py\n\
         f = h5py.File({p:?}, 'w')\n\
         d = f.create_dataset('y', data=np.arange(2, dtype='<i4'))\n\
         dt = h5py.string_dtype(encoding='utf-8')\n\
         vals = np.array(['alpha', '', '温度', 'z'*40], dtype=object)\n\
         d.attrs.create('names', vals, dtype=dt)\n\
         f.close()\n",
        p = path.display()
    );
    match author_h5py(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping w3_array_vlen_string_attr_reads: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let f = File::open(&path).expect("open h5py file");
    let names = attr(&f, "/y", "names");
    assert_eq!(
        names.as_strings().unwrap(),
        vec![
            "alpha".to_string(),
            String::new(),
            "温度".to_string(),
            "z".repeat(40),
        ]
    );
    // A 4-element array is not a scalar: as_str returns None.
    assert_eq!(names.as_str(), None, "array attr is not a scalar string");
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Nested-group repro of the reverse-diffuzz `rev_*_groups` case: a dataset deep
// in a group tree carrying a vlen string attribute alongside numeric ones.
// ---------------------------------------------------------------------------

#[test]
fn w3_nested_group_vlen_string_attr_reads() {
    let path = tmp_path("groups");
    let script = format!(
        "import numpy as np, h5py\n\
         f = h5py.File({p:?}, 'w')\n\
         f.create_group('a/b/c')\n\
         d = f.create_dataset('a/b/c/x', data=np.arange(4, dtype='<i4'))\n\
         d.attrs['units'] = 'm'\n\
         d.attrs['scale'] = np.float64(2.5)\n\
         d.attrs['count'] = np.int64(-7)\n\
         f.close()\n",
        p = path.display()
    );
    match author_h5py(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping w3_nested_group_vlen_string_attr_reads: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let f = File::open(&path).expect("open h5py file");
    // The string attribute (the previously-dropped one) …
    assert_eq!(
        attr(&f, "/a/b/c/x", "units").as_str(),
        Some("m".to_string())
    );
    // … alongside the numeric attributes that already worked.
    assert_eq!(attr(&f, "/a/b/c/x", "scale").as_f64(), Some(2.5));
    assert_eq!(attr(&f, "/a/b/c/x", "count").as_i64(), Some(-7));
    cleanup(&path);
}
