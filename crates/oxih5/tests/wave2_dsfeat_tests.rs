// Wave 2 DSFEAT integration tests.
//
// Dataset-level writer features that live entirely in the dataset/object-header
// path:
//   * G003 — fixed-length string datasets (numpy `S<width>`)
//   * G009 — boolean datasets (`H5T_ENUM{ FALSE=0, TRUE=1 }` over i8)
//   * G014 — custom dataset fill values
//
// Every feature writes a file, reads it back through oxih5's own reader, and —
// where python3 + h5py are available — verifies the exact on-disk shape against
// libhdf5 with the same three-way graceful-skip guard the other write tests use
// (python3 missing → skip, h5py/numpy missing → skip, any other non-zero exit →
// fail).  G014 additionally reads an h5py-*written* sparse file back through
// oxih5, proving the pair honours a custom fill in a chunked dataset's holes.

use oxih5::{Dtype, FileWriter};
use std::sync::atomic::{AtomicU64, Ordering};

static W2_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = W2_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_w2ds_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

/// Run an h5py check over an oxih5-written `path`, then delete it.
///
/// python3 missing → skip; h5py/numpy missing → skip; any other non-zero exit →
/// fail.  A silent skip on a *real* failure would make the test worthless, so
/// only the two "not installed" shapes are tolerated.
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

/// Create a file *with* h5py (the reverse of `h5py_check`): run `script`, and
/// return whether it produced the file.  Same graceful-skip contract — a missing
/// interpreter or module returns `false` (the caller skips its assertions), any
/// other failure panics.
fn python_make(script: &str, what: &str) -> bool {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output();
    match output {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                eprintln!("h5py not available — skipping {what}");
                false
            } else {
                panic!(
                    "{what} setup FAILED:\nstdout: {}\nstderr: {stderr}",
                    String::from_utf8_lossy(&out.stdout)
                );
            }
        }
        Err(_) => {
            eprintln!("python3 not found — skipping {what}");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// G003 — fixed-length string datasets
// ---------------------------------------------------------------------------

#[test]
fn fixed_string_dataset_round_trips_and_h5py_reads_s_width() {
    let path = tmp_path("g003");
    let mut w = FileWriter::new();
    // Width 8: "hello" fits, "" is all-NUL, "café" is 5 UTF-8 bytes fitting 8.
    w.create_fixed_string_dataset("labels", 8, &["abc", "hello", "de", "", "café"])
        .expect("labels");
    // A sibling numeric dataset so a size mistake in the string data area would
    // corrupt something detectable rather than just ending the file early.
    w.write_dataset_i32("after", &[7, 8], &[2]).expect("after");
    w.build(&path).expect("build");

    // oxih5's own reader: the fixed strings decode back, NUL padding trimmed,
    // multi-byte UTF-8 intact.
    {
        let f = oxih5::open(&path).expect("open");
        let ds = f.dataset("labels").expect("labels");
        assert_eq!(ds.shape, vec![5usize]);
        assert!(
            matches!(
                &ds.dtype,
                Dtype::String {
                    fixed_len: Some(8),
                    ..
                }
            ),
            "fixed-length S8, got {:?}",
            ds.dtype
        );
        assert_eq!(
            f.dataset_strings("labels").expect("strings"),
            vec!["abc", "hello", "de", "", "café"]
        );
        assert_eq!(
            f.dataset("after").expect("after").as_i32().expect("i32"),
            vec![7, 8],
            "the sibling dataset is intact"
        );
    }

    // h5py: exactly numpy S8 (raw bytes), every element byte-for-byte.
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','r'); d=f['labels']\n\
         assert d.dtype==np.dtype('S8'), d.dtype\n\
         got=list(d[()])\n\
         want=[b'abc',b'hello',b'de',b'',b'caf\\xc3\\xa9']\n\
         assert got==want, (got, want)\n\
         assert list(f['after'][()])==[7,8]\n\
         print('g003 S8 bytes OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "G003 fixed-length strings");
}

#[test]
fn fixed_string_truncation_and_zero_width_are_typed_errors() {
    let mut w = FileWriter::new();
    // A string longer than the width is refused, never silently truncated.
    let err = w
        .create_fixed_string_dataset("s", 3, &["ok", "toolong"])
        .expect_err("truncation must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("over the fixed width") && msg.contains("truncation is refused"),
        "error must explain: {msg}"
    );
    // A zero width is not a representable string datatype.
    let err = w
        .create_fixed_string_dataset("z", 0, &["a"])
        .expect_err("zero width must be refused");
    assert!(format!("{err}").contains("at least 1"), "{err}");
    // A refused dataset leaves no trace: the names are still free.
    w.create_fixed_string_dataset("s", 8, &["ok"])
        .expect("s now fits");
    w.create_fixed_string_dataset("z", 8, &["a"])
        .expect("z now fits");
}

// ---------------------------------------------------------------------------
// G009 — boolean datasets
// ---------------------------------------------------------------------------

#[test]
fn bool_dataset_round_trips_and_h5py_reads_a_bool_array() {
    let path = tmp_path("g009");
    let mut w = FileWriter::new();
    let flags = [true, false, true, true, false];
    w.write_dataset_bool("mask", &flags).expect("mask");
    w.write_dataset_f64("after", &[1.5], &[1]).expect("after");
    w.build(&path).expect("build");

    // oxih5's own reader: the enum decodes, and the raw bytes are 0/1 per flag.
    {
        let f = oxih5::open(&path).expect("open");
        let ds = f.dataset("mask").expect("mask");
        assert_eq!(ds.shape, vec![5usize]);
        assert!(
            matches!(&ds.dtype, Dtype::Enum { .. }),
            "boolean dataset is an enum, got {:?}",
            ds.dtype
        );
        assert_eq!(
            ds.data,
            flags.iter().map(|&b| u8::from(b)).collect::<Vec<u8>>(),
            "one 0/1 byte per flag"
        );
        assert_eq!(
            f.dataset("after").expect("after").as_f64().expect("f64"),
            vec![1.5]
        );
    }

    // h5py: a numpy bool array with the same values.
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','r'); d=f['mask']\n\
         assert d.dtype==np.dtype('bool'), d.dtype\n\
         assert list(d[()])==[True,False,True,True,False], list(d[()])\n\
         print('g009 bool OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "G009 boolean dataset");
}

// ---------------------------------------------------------------------------
// G014 — custom fill values
// ---------------------------------------------------------------------------

#[test]
fn custom_fill_value_round_trips_and_h5py_reads_fillvalue() {
    let path = tmp_path("g014");
    let dtype_i32 = Dtype::Int {
        size: 4,
        signed: true,
        order: oxih5::ByteOrder::Little,
    };
    let mut w = FileWriter::new();
    // Contiguous f64 with a custom fill and full data: the data is unaffected,
    // the fill message just declares the sentinel.
    w.write_dataset_f64("temp", &[1.0, 2.0, 3.0], &[3])
        .expect("temp");
    w.set_fill_value_f64("temp", -999.0).expect("temp fill");
    // Chunked i32 with a custom fill (space allocation time differs: 3, not 2).
    let raw: Vec<u8> = [10i32, 20, 30, 40, 50, 60]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    w.create_dataset_unlimited("counts", &[6], &[3], &dtype_i32, &raw)
        .expect("counts");
    w.set_fill_value_i32("counts", -7).expect("counts fill");
    // A dataset with no custom fill keeps the zero default.
    w.write_dataset_f64("plain", &[5.0], &[1]).expect("plain");
    w.build(&path).expect("build");

    // oxih5's own reader: every dataset's declared data reads back exactly.
    {
        let f = oxih5::open(&path).expect("open");
        assert_eq!(
            f.dataset("temp").expect("temp").as_f64().expect("f64"),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            f.dataset("counts").expect("counts").as_i32().expect("i32"),
            vec![10, 20, 30, 40, 50, 60]
        );
        assert_eq!(
            f.dataset("plain").expect("plain").as_f64().expect("f64"),
            vec![5.0]
        );
    }

    // h5py: the declared fill value and the data both read back.
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','r')\n\
         t=f['temp']; assert t.fillvalue==-999.0, t.fillvalue\n\
         assert np.array_equal(t[()], [1.0,2.0,3.0])\n\
         c=f['counts']; assert c.fillvalue==-7, c.fillvalue\n\
         assert np.array_equal(c[()], [10,20,30,40,50,60])\n\
         p=f['plain']; assert p.fillvalue==0.0, p.fillvalue\n\
         print('g014 fillvalue OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "G014 custom fill value");
}

#[test]
fn set_fill_value_rejects_type_mismatch_and_replaces() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("counts", &[1, 2], &[2])
        .expect("counts");

    // An f64 fill on an i32 dataset would put eight bytes where four belong.
    // (`FileWriter` is not `Debug`, so match rather than `expect_err`.)
    let err = match w.set_fill_value_f64("counts", -1.0) {
        Err(e) => e,
        Ok(_) => panic!("type mismatch must be refused"),
    };
    assert!(
        format!("{err}").contains("does not match the dataset's element type"),
        "{err}"
    );
    // A group / the root / a missing dataset are all refused too.
    w.create_group("grp").expect("grp");
    assert!(
        w.set_fill_value_i32("grp", 0).is_err(),
        "a group is not a dataset"
    );
    assert!(w.set_fill_value_i32("/", 0).is_err(), "nor the root");
    assert!(
        w.set_fill_value_i32("nope", 0).is_err(),
        "nor a missing dataset"
    );

    // Setting twice keeps only the last value — the file still builds and reads.
    w.set_fill_value_i32("counts", 1).expect("first");
    w.set_fill_value_i32("counts", -42)
        .expect("second replaces");
    let path = tmp_path("g014replace");
    w.build(&path).expect("build");
    {
        let f = oxih5::open(&path).expect("open");
        assert_eq!(
            f.dataset("counts").expect("counts").as_i32().expect("i32"),
            vec![1, 2]
        );
    }
    let script = format!(
        "import h5py\n\
         f=h5py.File(r'{p}','r'); c=f['counts']\n\
         assert c.fillvalue==-42, c.fillvalue\n\
         print('g014 replace OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "G014 fill replace");
}

// ---------------------------------------------------------------------------
// G017 — compact layout
// ---------------------------------------------------------------------------

#[test]
fn compact_dataset_round_trips_and_h5py_reads_it() {
    let path = tmp_path("g017");
    let mut w = FileWriter::new();
    // A small dataset stored inline in its object header.
    w.write_dataset_i32("small", &[10, 20, 30, 40, 50, 60], &[6])
        .expect("small");
    w.set_compact("small").expect("compact");
    // A contiguous dataset after it, so a mis-sized compact header (which no
    // longer reserves a data area) would corrupt something detectable.
    w.write_dataset_f64("after", &[1.5, 2.5], &[2])
        .expect("after");
    w.build(&path).expect("build");

    // oxih5's own reader: the inline data reads back, and the sibling is intact.
    {
        let f = oxih5::open(&path).expect("open");
        let ds = f.dataset("small").expect("small");
        assert_eq!(ds.shape, vec![6usize]);
        assert_eq!(ds.as_i32().expect("i32"), vec![10, 20, 30, 40, 50, 60]);
        assert_eq!(
            f.dataset("after").expect("after").as_f64().expect("f64"),
            vec![1.5, 2.5]
        );
    }

    // h5py: the layout really is compact, and the values read back.
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','r'); d=f['small']\n\
         assert d.id.get_create_plist().get_layout()==h5py.h5d.COMPACT, 'not compact'\n\
         assert np.array_equal(d[()], [10,20,30,40,50,60]), list(d[()])\n\
         assert np.array_equal(f['after'][()], [1.5,2.5])\n\
         print('g017 compact OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "G017 compact layout");
}

#[test]
fn set_compact_rejects_chunked_vlen_and_oversized_and_is_idempotent() {
    let dtype_i32 = Dtype::Int {
        size: 4,
        signed: true,
        order: oxih5::ByteOrder::Little,
    };
    let mut w = FileWriter::new();

    // A chunked dataset has no single inline byte run.
    let raw: Vec<u8> = [1i32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    w.create_dataset_unlimited("ch", &[4], &[2], &dtype_i32, &raw)
        .expect("ch");
    assert!(w.set_compact("ch").is_err(), "chunked cannot be compact");

    // Nor a vlen-string dataset.
    w.create_vlen_string_dataset("vl", &["a", "b"]).expect("vl");
    assert!(w.set_compact("vl").is_err(), "vlen cannot be compact");

    // Nor a group / the root / a missing dataset.
    w.create_group("grp").expect("grp");
    assert!(w.set_compact("grp").is_err(), "a group is not a dataset");
    assert!(w.set_compact("/").is_err(), "nor the root");
    assert!(w.set_compact("nope").is_err(), "nor a missing dataset");

    // Over the 64 KiB cap: 20000 i32 = 80000 bytes.
    let big: Vec<i32> = (0..20_000).collect();
    w.write_dataset_i32("big", &big, &[20_000]).expect("big");
    let err = match w.set_compact("big") {
        Err(e) => e,
        Ok(_) => panic!("oversized compact must be refused"),
    };
    assert!(format!("{err}").contains("64 KiB"), "{err}");

    // Idempotent: marking a dataset compact twice leaves one valid, readable copy.
    w.write_dataset_i32("ok", &[7, 8, 9], &[3]).expect("ok");
    w.set_compact("ok").expect("first");
    w.set_compact("ok").expect("second is a no-op");
    let path = tmp_path("g017idem");
    w.build(&path).expect("build");
    {
        let f = oxih5::open(&path).expect("open");
        assert_eq!(
            f.dataset("ok").expect("ok").as_i32().expect("i32"),
            vec![7, 8, 9]
        );
    }
    cleanup(&path);
}

/// The pair, end to end: h5py *writes* a sparse chunked dataset with a custom
/// fill (only the first chunk populated), and oxih5 reads the unwritten holes
/// back as that fill — the same version-2 fill message oxih5 itself emits.
#[test]
fn custom_fill_reads_in_holes_across_h5py_and_oxih5() {
    let path = tmp_path("g014holes");
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','w',libver='earliest')\n\
         d=f.create_dataset('d', shape=(10,), chunks=(4,), dtype='f8', fillvalue=-999.0)\n\
         d[0:4]=[1.0,2.0,3.0,4.0]\n\
         f.close()\n\
         print('sparse written')",
        p = path.display()
    );
    if !python_make(&script, "G014 sparse-fill setup") {
        return;
    }

    // oxih5 reads the populated chunk verbatim and every hole as the fill.
    let f = oxih5::open(&path).expect("open");
    let got = f.dataset("d").expect("d").as_f64().expect("f64");
    cleanup(&path);
    assert_eq!(
        got,
        vec![1.0, 2.0, 3.0, 4.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0],
        "the six unwritten elements read back as the custom fill"
    );
}
