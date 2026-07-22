//! Fix-lane ELEM regression tests — datatype-encoding hardening in
//! `write/elem.rs`.
//!
//! Covers:
//! * **B003** — an empty scalar fixed-string attribute must not emit a size-0
//!   string datatype (which makes libhdf5 unable to iterate ANY attribute on the
//!   object, poisoning its siblings).
//! * **B017** — variable-length string datatypes must declare the UTF-8 charset.
//! * **B022** — fixed-length string datatypes must use NULLPAD, as libhdf5 does.
//! * **R004** — an empty attribute name must be rejected, not written to a file
//!   libhdf5 cannot iterate.
//!
//! Every h5py assertion uses the same three-way graceful-skip guard as
//! `write_tests.rs`: python3 missing → skip, h5py/numpy missing → skip, any
//! other non-zero exit → fail.  The Rust-only tests never skip.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxih5::FileWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static FE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    let n = FE_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_fixelem_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Run an h5py check over `path`, then delete it.
///
/// Mirrors `write_tests.rs::h5py_check`: only the two "not installed" shapes are
/// tolerated, so a silent skip on a real failure cannot happen.
fn h5py_check(path: &Path, script: &str, what: &str) {
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

// ---------------------------------------------------------------------------
// B003 — empty scalar fixed-string attribute
// ---------------------------------------------------------------------------

/// oxih5's own reader must round-trip an empty scalar string attribute, and it
/// must not disturb its non-empty siblings — on a dataset, a group, and root.
#[test]
fn empty_string_attr_round_trips_and_keeps_siblings() {
    let path = tmp_path("b003_roundtrip");
    let mut w = FileWriter::new();
    w.write_dataset_i32("data", &[1, 2, 3], &[3]).unwrap();
    w.write_i32_attr("data", "good_i32", 42).unwrap();
    w.write_string_attr("data", "empty", "").unwrap();
    w.write_string_attr("data", "good_str", "hello").unwrap();
    w.create_group("grp").unwrap();
    w.write_string_attr("grp", "g_empty", "").unwrap();
    w.write_string_attr("grp", "g_str", "world").unwrap();
    w.write_root_str_attr("r_empty", "");
    w.write_root_str_attr("r_str", "title");
    w.build(&path).unwrap();

    let file = oxih5::open(&path).unwrap();

    let names: Vec<String> = file
        .attr_views("data")
        .unwrap()
        .iter()
        .map(|v| v.name().to_string())
        .collect();
    assert!(names.contains(&"good_i32".to_string()), "sibling int lost");
    assert!(names.contains(&"good_str".to_string()), "sibling str lost");
    assert!(names.contains(&"empty".to_string()), "empty attr lost");

    for v in file.attr_views("data").unwrap() {
        match v.name() {
            "empty" => assert_eq!(v.as_str_fixed().as_deref(), Some("")),
            "good_str" => assert_eq!(v.as_str_fixed().as_deref(), Some("hello")),
            _ => {}
        }
    }

    for v in file.attr_views("grp").unwrap() {
        match v.name() {
            "g_empty" => assert_eq!(v.as_str_fixed().as_deref(), Some("")),
            "g_str" => assert_eq!(v.as_str_fixed().as_deref(), Some("world")),
            _ => {}
        }
    }

    cleanup(&path);
}

/// libhdf5 (via h5py) must be able to enumerate and read EVERY attribute on an
/// object that carries an empty scalar string — the exact operation that raised
/// `RuntimeError: invalid datatype size` before the size-1 clamp.
#[test]
fn empty_string_attr_lets_h5py_enumerate_every_sibling() {
    let path = tmp_path("b003_h5py");
    let mut w = FileWriter::new();
    w.write_dataset_i32("data", &[1, 2, 3], &[3]).unwrap();
    w.write_i32_attr("data", "good_i32", 42).unwrap();
    w.write_string_attr("data", "empty", "").unwrap();
    w.write_string_attr("data", "good_str", "hello").unwrap();
    w.create_group("grp").unwrap();
    w.write_string_attr("grp", "g_empty", "").unwrap();
    w.write_string_attr("grp", "g_str", "world").unwrap();
    w.write_root_str_attr("r_empty", "");
    w.write_root_str_attr("r_str", "title");
    w.build(&path).unwrap();

    let script = format!(
        "import h5py\n\
         def s(x): return x.decode() if isinstance(x,bytes) else x\n\
         f=h5py.File({path:?},'r')\n\
         d=dict(f['data'].attrs)\n\
         assert sorted(d.keys())==['empty','good_i32','good_str'], sorted(d.keys())\n\
         assert s(d['empty'])=='', repr(d['empty'])\n\
         assert s(d['good_str'])=='hello', repr(d['good_str'])\n\
         assert int(d['good_i32'])==42, repr(d['good_i32'])\n\
         g=dict(f['grp'].attrs)\n\
         assert sorted(g.keys())==['g_empty','g_str'], sorted(g.keys())\n\
         assert s(g['g_empty'])=='' and s(g['g_str'])=='world'\n\
         r=dict(f.attrs)\n\
         assert sorted(r.keys())==['r_empty','r_str'], sorted(r.keys())\n\
         assert s(r['r_empty'])=='' and s(r['r_str'])=='title'\n\
         print('h5py enumerated', len(d)+len(g)+len(r), 'attrs across dataset/group/root')\n",
        path = path.display()
    );
    h5py_check(&path, &script, "B003 empty-string enumeration");
}

// ---------------------------------------------------------------------------
// B022 — fixed-length string uses NULLPAD
// ---------------------------------------------------------------------------

/// A fixed-length string attribute must report `strpad == 1` (NULLPAD) to
/// libhdf5, matching what h5py itself emits for a same-length string.
#[test]
fn fixed_string_attr_reports_nullpad_to_h5py() {
    let path = tmp_path("b022_h5py");
    let mut w = FileWriter::new();
    w.write_dataset_i32("ds", &[1], &[1]).unwrap();
    w.write_string_attr("ds", "units", "meters").unwrap();
    w.build(&path).unwrap();

    let script = format!(
        "import h5py\n\
         f=h5py.File({path:?},'r')\n\
         t=f['ds'].attrs.get_id('units').get_type()\n\
         assert t.get_class()==3, t.get_class()\n\
         assert t.get_strpad()==1, ('strpad', t.get_strpad())\n\
         assert t.get_size()==6, ('size', t.get_size())\n\
         v=f['ds'].attrs['units']\n\
         assert (v.decode() if isinstance(v,bytes) else v)=='meters', repr(v)\n\
         print('h5py: fixed string strpad=NULLPAD size=6 value=meters')\n",
        path = path.display()
    );
    h5py_check(&path, &script, "B022 fixed-string NULLPAD");
}

// ---------------------------------------------------------------------------
// B017 — vlen string declares UTF-8
// ---------------------------------------------------------------------------

/// A variable-length string dataset must declare the UTF-8 charset, so h5py
/// reports `encoding == 'utf-8'` rather than 'ascii' for a UTF-8 payload.
#[test]
fn vlen_string_dataset_declares_utf8_to_h5py() {
    let path = tmp_path("b017_h5py");
    let mut w = FileWriter::new();
    w.create_vlen_string_dataset("vl", &["café", "naïve"])
        .unwrap();
    w.build(&path).unwrap();

    // The dtype charset is inspectable without reading the heap data, so this
    // check is independent of the vlen-data lanes.
    let script = format!(
        "import h5py\n\
         f=h5py.File({path:?},'r')\n\
         info=h5py.check_string_dtype(f['vl'].dtype)\n\
         assert info is not None, 'not a string dtype'\n\
         assert info.length is None, ('not vlen', info.length)\n\
         assert info.encoding=='utf-8', ('encoding', info.encoding)\n\
         print('h5py: vlen string dtype encoding=utf-8')\n",
        path = path.display()
    );
    h5py_check(&path, &script, "B017 vlen UTF-8 charset");
}

// ---------------------------------------------------------------------------
// R004 — empty attribute name is rejected
// ---------------------------------------------------------------------------

/// `build()` must fail for an object that carries a zero-length attribute name,
/// and no file may be produced.
#[test]
fn empty_attribute_name_fails_the_build() {
    let path = tmp_path("r004_reject");
    let mut w = FileWriter::new();
    w.write_dataset_i32("ds", &[1], &[1]).unwrap();
    w.write_i32_attr("ds", "", 7).unwrap(); // accepted at attach time
    let err = w
        .build(&path)
        .expect_err("empty attr name must fail the build");
    assert!(
        err.to_string().contains("empty"),
        "error should explain the empty name, got: {err}"
    );
    assert!(
        !path.exists(),
        "no file may be produced for a rejected build"
    );
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Backward compatibility — the fixed reader still reads pre-fix (NULLTERM) bytes
// ---------------------------------------------------------------------------

/// A file written by the pre-fix writer (fixed strings as NULLTERM, byte1=0x10)
/// must still read correctly.  The reader is strpad-agnostic — it trims at the
/// first NUL — so the NULLTERM→NULLPAD writer change cannot regress old files.
#[test]
fn reader_still_reads_prefix_nullterm_string_attrs() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/prefix_string_attrs_nullterm.h5");
    let file = oxih5::open(&fixture).expect("open pre-fix fixture");

    let mut good_str = None;
    let mut notempty = None;
    for v in file.attr_views("data").expect("attr_views") {
        match v.name() {
            "good_str" => good_str = v.as_str_fixed(),
            "notempty" => notempty = v.as_str_fixed(),
            _ => {}
        }
    }
    assert_eq!(
        good_str.as_deref(),
        Some("hello"),
        "old NULLTERM string lost"
    );
    assert_eq!(
        notempty.as_deref(),
        Some("x"),
        "old size-1 NULLTERM string lost"
    );
}
