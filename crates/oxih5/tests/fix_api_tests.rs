//! Writer-surface input-validation and API-contract regression tests.
//!
//! Covers the api-lane findings of the 0.2.2 write-path hardening campaign:
//!
//! * **B015** — a link/dataset/group name with an embedded NUL is rejected
//!   instead of silently truncating in the local heap into a duplicate link.
//! * **B021** — writing the same attribute name twice overwrites (h5py
//!   `attrs[k]=v` semantics) instead of emitting two spec-invalid messages
//!   whose second value no reader can reach.
//! * **R001 / R002** — `create_dataset` / `create_dataset_unlimited` size their
//!   buffers with overflow-checked arithmetic (typed error, not a debug panic
//!   or a release wraparound).
//! * **R008** — `set_deflate` on a scalar dataset fails fast at the call site
//!   rather than deferring a confusing error to `build()`.
//!
//! Each h5py-facing test shells out to `python3` and skips gracefully when
//! `python3`/`h5py` is absent, mirroring `write_tests.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxih5::{ByteOrder, Dtype, File, FileWriter};

// ---------------------------------------------------------------------------
// Shared helpers (h5py graceful-skip oracle — same three-way guard as
// write_tests.rs: python3 missing → skip, h5py/numpy missing → skip, any other
// non-zero exit → fail).
// ---------------------------------------------------------------------------

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("oxih5_fix_api_{name}.h5"))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

fn h5py_check(path: &std::path::Path, script: &str, what: &str) {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output();

    cleanup(path);

    match output {
        Ok(out) if out.status.success() => {
            eprintln!("{what}: {}", String::from_utf8_lossy(&out.stdout).trim());
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                eprintln!("h5py not available — skipping {what}");
            } else {
                panic!("{what} FAILED:\nstdout: {stdout}\nstderr: {stderr}");
            }
        }
        Err(_) => {
            eprintln!("python3 not found — skipping {what}");
        }
    }
}

fn le_f64() -> Dtype {
    Dtype::Float {
        size: 8,
        order: ByteOrder::Little,
    }
}

// ---------------------------------------------------------------------------
// B015 — embedded NUL in a name is rejected at every entry point
// ---------------------------------------------------------------------------

#[test]
fn b015_nul_in_name_is_rejected_at_every_entry_point() {
    let mut w = FileWriter::new();

    // Typed dataset path. `write_dataset_*` returns `&mut Self` on success,
    // which is not `Debug`, so inspect the error through `.err()`.
    let msg = w
        .write_dataset_f64("a\u{0}b", &[1.0], &[1])
        .err()
        .map(|e| e.to_string());
    assert!(
        msg.as_deref().is_some_and(|m| m.contains("NUL")),
        "NUL dataset name must be rejected: {msg:?}"
    );

    // The exact repro collision: two distinct names that both truncate to "a".
    assert!(w.write_dataset_f64("a\u{0}c", &[2.0], &[1]).is_err());

    // Dtype-driven path, unlimited path, vlen-string path, group path, and a
    // NUL buried at depth — every seam funnels through `split_path`.
    assert!(w.create_dataset("g\u{0}x", &[1], &le_f64()).is_err());
    assert!(w
        .create_dataset_unlimited("u\u{0}v", &[1], &[1], &le_f64(), &0.0f64.to_le_bytes())
        .is_err());
    assert!(w.create_vlen_string_dataset("s\u{0}t", &["x"]).is_err());
    assert!(w.create_group("grp\u{0}").is_err());
    assert!(w.write_dataset_f64("ok/a\u{0}b", &[1.0], &[1]).is_err());

    // A rejected write must leave no trace: the plain, legitimate names still
    // succeed and build cleanly.
    w.write_dataset_f64("a", &[5.0], &[1]).expect("plain a");
    w.write_dataset_f64("c", &[6.0], &[1]).expect("plain c");
    // A name that merely *contains* dots or spaces (which libhdf5 accepts) is
    // not over-rejected.
    w.write_dataset_f64("a.b c", &[7.0], &[1]).expect("dotted");
    let bytes = w.build_to_vec().expect("build after rejects");
    assert!(!bytes.is_empty());
}

#[test]
fn b015_namespace_stays_clean_h5py() {
    // Before the fix, `a\0zzz` was stored as a second dataset whose on-disk heap
    // name truncated to "a", so h5py saw duplicate keys ['a','a']. The write is
    // now rejected, leaving exactly one unambiguous link 'a'.
    let path = tmp("b015_clean");
    let mut w = FileWriter::new();
    w.write_dataset_f64("a", &[5.0], &[1]).expect("real a");
    assert!(
        w.write_dataset_f64("a\u{0}zzz", &[9.0], &[1]).is_err(),
        "NUL-truncating name must be rejected, not stored as a duplicate 'a'"
    );
    w.build(&path).expect("build");

    let script = format!(
        "import h5py\n\
         f = h5py.File({path:?}, 'r')\n\
         ks = sorted(f.keys())\n\
         assert ks == ['a'], f'expected a single clean link, got {{ks}}'\n\
         v = list(f['a'][:])\n\
         assert v == [5.0], f'wrong value {{v}}'\n\
         print('one link a =', v)",
        path = path.display()
    );
    h5py_check(&path, &script, "B015 clean namespace");
}

// ---------------------------------------------------------------------------
// B021 — a duplicate attribute name overwrites (h5py semantics)
// ---------------------------------------------------------------------------

#[test]
fn b021_duplicate_attr_overwrites_reader() {
    let path = tmp("b021_reader");
    let mut w = FileWriter::new();
    w.write_dataset_f64("d", &[1.0], &[1]).expect("d");
    w.write_string_attr("d", "units", "km").expect("km");
    w.write_string_attr("d", "units", "miles").expect("miles");
    w.build(&path).expect("build");

    let f = File::open(&path).expect("open");
    cleanup(&path);
    let attrs = f.attr_views("d").expect("attr_views");
    let units: Vec<_> = attrs.iter().filter(|a| a.name() == "units").collect();
    assert_eq!(
        units.len(),
        1,
        "duplicate name must collapse to one attribute"
    );
    assert_eq!(
        units[0].as_str_fixed().expect("fixed str"),
        "miles",
        "the second write must win (overwrite), not the first"
    );
}

#[test]
fn b021_duplicate_attr_overwrites_h5py() {
    let path = tmp("b021_h5py");
    let mut w = FileWriter::new();
    w.write_dataset_f64("d", &[1.0], &[1]).expect("d");
    w.write_string_attr("d", "units", "km").expect("km");
    w.write_string_attr("d", "units", "miles").expect("miles");
    w.build(&path).expect("build");

    let script = format!(
        "import h5py\n\
         f = h5py.File({path:?}, 'r')\n\
         d = f['d']\n\
         ks = list(d.attrs.keys())\n\
         assert ks == ['units'], f'expected one unique attr, got {{ks}}'\n\
         v = d.attrs['units']\n\
         sv = v.decode() if isinstance(v, bytes) else str(v)\n\
         assert sv == 'miles', f'expected overwrite to miles, got {{v!r}}'\n\
         print('units =', sv)",
        path = path.display()
    );
    h5py_check(&path, &script, "B021 duplicate-attr overwrite");
}

#[test]
fn b021_overwrite_can_change_value_type_and_survives_on_a_group() {
    // Overwrite is by name regardless of value kind, and it works on a group
    // target as well as a dataset.
    let path = tmp("b021_group");
    let mut w = FileWriter::new();
    w.create_group("g").expect("g");
    w.write_string_attr("g", "x", "first").expect("first");
    w.write_i64_attr("g", "x", 42).expect("overwrite with int");
    w.build(&path).expect("build");

    let f = File::open(&path).expect("open");
    cleanup(&path);
    let attrs = f.attr_views("g").expect("attr_views");
    let xs: Vec<_> = attrs.iter().filter(|a| a.name() == "x").collect();
    assert_eq!(xs.len(), 1, "one attribute named x after overwrite");
}

// ---------------------------------------------------------------------------
// R001 — create_dataset overflow is a typed error, not a debug panic
// ---------------------------------------------------------------------------

#[test]
fn r001_create_dataset_shape_overflow_is_typed_error() {
    let mut w = FileWriter::new();
    // Shape product 2^80 overflows usize. Before the fix this panicked in a
    // debug/overflow-checked build (this test) and wrapped in release.
    let err = w
        .create_dataset("x", &[1usize << 40, 1usize << 40], &le_f64())
        .expect_err("shape-product overflow must be a typed error");
    assert!(format!("{err}").contains("overflows usize"), "{err}");

    // Byte-size overflow: 2^61 elements × 8 bytes overflows usize even though
    // the element count itself fits.
    let err = w
        .create_dataset("y", &[1usize << 61], &le_f64())
        .expect_err("byte-size overflow must be a typed error");
    assert!(format!("{err}").contains("overflows usize"), "{err}");
}

// ---------------------------------------------------------------------------
// R002 — create_dataset_unlimited overflow + real length validation
// ---------------------------------------------------------------------------

#[test]
fn r002_create_dataset_unlimited_overflow_is_typed_error() {
    let mut w = FileWriter::new();
    // Before the fix: debug panic; release wrapped the product to 0, so the
    // empty data slice matched `0 * 8` and a 2^80-shape / 0-byte dataset was
    // *accepted*. The checked form now rejects the overflow in either profile.
    let err = w
        .create_dataset_unlimited("x", &[1usize << 40, 1usize << 40], &[1], &le_f64(), &[])
        .expect_err("overflow must be rejected, never silently accepted");
    assert!(format!("{err}").contains("overflows usize"), "{err}");
}

#[test]
fn r002_create_dataset_unlimited_rejects_length_mismatch() {
    let mut w = FileWriter::new();
    // Shape [2,3] f64 wants 48 bytes; hand it 8.
    let err = w
        .create_dataset_unlimited("x", &[2, 3], &[2, 3], &le_f64(), &0.0f64.to_le_bytes())
        .expect_err("data length must match the declared shape");
    let msg = format!("{err}");
    assert!(msg.contains("data length"), "{msg}");
    // A correct length still succeeds and builds.
    let raw: Vec<u8> = (0..6i32).flat_map(|i| f64::from(i).to_le_bytes()).collect();
    w.create_dataset_unlimited("ok", &[2, 3], &[2, 3], &le_f64(), &raw)
        .expect("consistent length accepted");
    assert!(!w.build_to_vec().expect("build").is_empty());
}

// ---------------------------------------------------------------------------
// R008 — set_deflate on a scalar dataset fails fast
// ---------------------------------------------------------------------------

#[test]
fn r008_set_deflate_on_scalar_fails_fast() {
    let mut w = FileWriter::new();
    w.create_dataset("s", &[], &le_f64()).expect("scalar");
    let err = w
        .set_deflate("s", 6)
        .expect_err("a scalar dataset cannot be compressed");
    assert!(format!("{err}").contains("scalar"), "{err}");
    // The rejection leaves the dataset contiguous, so it still builds — the
    // failure is no longer deferred to build().
    assert!(
        w.build_to_vec().is_ok(),
        "a rejected set_deflate must not poison the scalar dataset"
    );
}

#[test]
fn r008_scalar_dataset_still_writes_and_reads_h5py() {
    let path = tmp("r008_scalar");
    let mut w = FileWriter::new();
    w.create_dataset("s", &[], &le_f64()).expect("scalar");
    assert!(w.set_deflate("s", 6).is_err(), "scalar deflate rejected");
    w.build(&path).expect("build scalar");

    let script = format!(
        "import h5py\n\
         f = h5py.File({path:?}, 'r')\n\
         d = f['s']\n\
         assert d.shape == (), f'expected scalar, got {{d.shape}}'\n\
         assert float(d[()]) == 0.0, f'wrong value {{d[()]}}'\n\
         print('scalar shape', d.shape, 'value', float(d[()]))",
        path = path.display()
    );
    h5py_check(&path, &script, "R008 scalar round-trip");
}
