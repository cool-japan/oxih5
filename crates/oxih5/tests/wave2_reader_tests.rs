//! Wave-2 LANE READER — reader-completeness regressions (h5py-authored).
//!
//! Covers, against files authored by `h5py` at test time (graceful-skip when the
//! toolchain is absent):
//!
//! * **Item 1 (old-style multi-SNOD enumeration).** An old-style (`libver=
//!   'earliest'`) root group with many members spans several SNOD symbol-table
//!   nodes across a version-1 B-tree; every member must be enumerated.
//! * **Item 2 (multi-object / multi-collection global heaps).** Variable-length
//!   string and sequence datasets whose heap objects fill one collection with
//!   many objects — and spill across multiple GCOL collections — must resolve
//!   every element, not fail with `GlobalHeap object N not found`.
//!
//! The version-2 object-header NIL-skip fix (item 1, netCDF-C files) is covered
//! directly and toolchain-free in `oxih5-format/tests/wave2_ohdr_nil.rs`; the
//! genuine netCDF-C enumeration + narrow-dtype reads live in
//! `oxinetcdf/tests/wave2_reader_tests.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use oxih5::File as H5File;
use oxih5_format::values::Value;

/// Outcome of trying to author an HDF5 file with `python3` + `h5py`.
enum Authored {
    Ok,
    Skip(String),
}

/// Author a file by running an h5py script. Skips gracefully when `python3` or
/// `h5py`/`numpy` is unavailable; panics only when the toolchain *is* present but
/// the script itself failed (a real error, not an environmental one).
fn author_h5py(script: &str) -> Authored {
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
    std::env::temp_dir().join(format!("oxih5_wave2_{name}.h5"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Item 1: an old-style root group with 80 datasets + 1 sub-group spans several
/// SNOD nodes; every member must be enumerated (no member dropped mid-B-tree).
#[test]
fn oldstyle_multi_snod_enumerates_every_member() {
    let path = tmp_path("multi_snod");
    let script = format!(
        r#"
import h5py, numpy as np
with h5py.File(r"{p}", "w", libver="earliest") as f:
    for i in range(80):
        f.create_dataset(f"ds_{{i:03d}}", data=np.arange(3, dtype="i4") + i)
    g = f.create_group("sub")
    g.create_dataset("inner", data=np.array([1.0, 2.0]))
print("OK")
"#,
        p = path.display()
    );
    match author_h5py(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping oldstyle_multi_snod: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let f = H5File::open(&path).unwrap();
    let root = f.group("/").unwrap();
    let datasets = root.datasets().unwrap();
    let groups = root.groups().unwrap();
    cleanup(&path);

    assert_eq!(
        datasets.len(),
        80,
        "expected all 80 old-style datasets, got {}",
        datasets.len()
    );
    for i in 0..80 {
        let name = format!("ds_{i:03}");
        assert!(datasets.contains(&name), "missing {name}");
    }
    assert_eq!(groups, vec!["sub".to_string()]);
}

/// Item 2: a vlen-int dataset large enough to spill across multiple 4 KiB global
/// heap collections must resolve every element to its exact sequence.
#[test]
fn multicollection_vlen_sequences_resolve_every_element() {
    let path = tmp_path("vlen_multicol");
    let script = format!(
        r#"
import h5py, numpy as np
vt = h5py.special_dtype(vlen=np.int32)
with h5py.File(r"{p}", "w", libver="earliest") as f:
    d = f.create_dataset("seqs", (120,), dtype=vt)
    for i in range(120):
        d[i] = np.arange(i % 30 + 1, dtype="i4")
print("OK")
"#,
        p = path.display()
    );
    match author_h5py(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping multicollection_vlen_sequences: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let f = H5File::open(&path).unwrap();
    let seqs = f.dataset_vlen_sequences("seqs").unwrap();
    cleanup(&path);

    assert_eq!(seqs.len(), 120);
    for (i, s) in seqs.iter().enumerate() {
        let Value::Sequence(elems) = s else {
            panic!("element {i} is not a sequence: {s:?}");
        };
        assert_eq!(
            elems.len(),
            i % 30 + 1,
            "element {i} has wrong length (multi-collection heap element lost?)"
        );
        for (j, v) in elems.iter().enumerate() {
            // vlen int32 elements decode as signed ints widened to i64.
            assert_eq!(v, &Value::Int(j as i64), "element {i}[{j}] wrong value");
        }
    }
}

/// Item 2: a vlen-string dataset with many objects packed into one collection
/// (and spilling into further collections) must resolve every string.
#[test]
fn multiobject_vlen_strings_resolve_every_element() {
    let path = tmp_path("vlen_multiobj");
    let script = format!(
        r#"
import h5py, numpy as np
st = h5py.special_dtype(vlen=str)
with h5py.File(r"{p}", "w", libver="earliest") as f:
    d = f.create_dataset("strs", (300,), dtype=st)
    for i in range(300):
        d[i] = ("x" * (i % 40)) + f"_{{i}}"
print("OK")
"#,
        p = path.display()
    );
    match author_h5py(&script) {
        Authored::Skip(why) => {
            eprintln!("skipping multiobject_vlen_strings: {why}");
            return;
        }
        Authored::Ok => {}
    }

    let f = H5File::open(&path).unwrap();
    let strs = f.dataset_strings("strs").unwrap();
    cleanup(&path);

    assert_eq!(strs.len(), 300);
    for (i, s) in strs.iter().enumerate() {
        let expect = format!("{}_{i}", "x".repeat(i % 40));
        assert_eq!(s, &expect, "string element {i} mismatch");
    }
}
