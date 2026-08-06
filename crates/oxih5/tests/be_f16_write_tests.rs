//! Writer coverage for **G008 (big-endian datasets and attributes)** and
//! **G012 (half-precision floats)**.
//!
//! The acceptance oracle for a *writer* is a third-party reader, not this
//! crate's own: a datatype message whose byte-order bit disagrees with the
//! payload round-trips perfectly through oxih5 (which would swap both halves
//! consistently) while libhdf5 reads every value byte-swapped.  So the
//! load-bearing test here is `h5py_reads_big_endian_and_half_precision`, which
//! asserts on numpy's dtype string (`>f4`, `>i8`, `float16`) and on the values.
//! The self-round-trip tests below are the cheap always-on guard that the
//! writer and reader stayed in step; they are not evidence of conformance.

use oxih5::{ByteOrder, Dtype, FileWriter, NumericValues};

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("oxih5_be_f16_{name}"))
}

/// Write the fixture both tests below read.
fn build_fixture(path: &std::path::Path) {
    let mut w = FileWriter::new();
    let be = ByteOrder::Big;
    let le = ByteOrder::Little;
    w.write_dataset_numeric("be_f32", NumericValues::F32(&[1.5, -2.25, 3.0]), be, &[3])
        .expect("be f32");
    w.write_dataset_numeric("be_f64", NumericValues::F64(&[1.5, -2.25]), be, &[2])
        .expect("be f64");
    w.write_dataset_numeric("be_i16", NumericValues::I16(&[1, -2, 258]), be, &[3])
        .expect("be i16");
    w.write_dataset_numeric(
        "be_i32",
        NumericValues::I32(&[1, -2, 0x0001_0203]),
        be,
        &[3],
    )
    .expect("be i32");
    w.write_dataset_numeric("be_i64", NumericValues::I64(&[-1, 2]), be, &[2])
        .expect("be i64");
    w.write_dataset_numeric("be_u16", NumericValues::U16(&[1, 258, 65535]), be, &[3])
        .expect("be u16");
    w.write_dataset_numeric("be_u32", NumericValues::U32(&[1, 0x0102_0304]), be, &[2])
        .expect("be u32");
    w.write_dataset_numeric("be_u64", NumericValues::U64(&[1, u64::MAX]), be, &[2])
        .expect("be u64");
    // ≥ 2-D, the other half of the G008 gap.
    w.write_dataset_numeric(
        "be_2d",
        NumericValues::F64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        be,
        &[2, 3],
    )
    .expect("be 2-D");
    w.write_dataset_numeric("le_f32", NumericValues::F32(&[1.5, -2.25, 3.0]), le, &[3])
        .expect("le f32");
    // Half precision: exact halves, the largest finite binary16, and the
    // smallest normal — the values whose encoding the rounding path decides.
    w.write_dataset_numeric(
        "f16",
        NumericValues::F16(&[0.5, -1.5, 1.0, 65504.0, 6.103_515_6e-5]),
        le,
        &[5],
    )
    .expect("f16");
    w.write_dataset_numeric("be_f16", NumericValues::F16(&[0.5, -1.5]), be, &[2])
        .expect("be f16");

    w.write_numeric_attr("be_f32", "be_array_attr", NumericValues::I32(&[7, 8]), be)
        .expect("be array attr");
    w.write_numeric_scalar_attr("be_f32", "be_scalar_attr", NumericValues::F64(&[2.5]), be)
        .expect("be scalar attr");
    w.write_numeric_scalar_attr("be_f32", "half_attr", NumericValues::F16(&[0.25]), le)
        .expect("f16 attr");

    w.build(path).expect("build");
}

/// **The conformance test.** h5py/libhdf5 must see the declared byte order and
/// the declared width, and read the values back exactly.
#[test]
fn h5py_reads_big_endian_and_half_precision() {
    let path = tmp_path("h5py.h5");
    let _ = std::fs::remove_file(&path);
    build_fixture(&path);

    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{p}', 'r')\n\
         def chk(name, dt, want):\n\
         \x20   d = f[name]\n\
         \x20   assert d.dtype.str == dt, (name, d.dtype.str, dt)\n\
         \x20   got = np.asarray(d[...]).ravel().tolist()\n\
         \x20   assert got == want, (name, got, want)\n\
         chk('be_f32', '>f4', [1.5, -2.25, 3.0])\n\
         chk('be_f64', '>f8', [1.5, -2.25])\n\
         chk('be_i16', '>i2', [1, -2, 258])\n\
         chk('be_i32', '>i4', [1, -2, 66051])\n\
         chk('be_i64', '>i8', [-1, 2])\n\
         chk('be_u16', '>u2', [1, 258, 65535])\n\
         chk('be_u32', '>u4', [1, 16909060])\n\
         chk('be_u64', '>u8', [1, 18446744073709551615])\n\
         chk('be_2d', '>f8', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])\n\
         assert f['be_2d'].shape == (2, 3), f['be_2d'].shape\n\
         chk('le_f32', '<f4', [1.5, -2.25, 3.0])\n\
         chk('f16', '<f2', [0.5, -1.5, 1.0, 65504.0, np.float16(6.1035156e-05)])\n\
         chk('be_f16', '>f2', [0.5, -1.5])\n\
         a = f['be_f32'].attrs\n\
         assert a['be_array_attr'].dtype.str == '>i4', a['be_array_attr'].dtype.str\n\
         assert a['be_array_attr'].tolist() == [7, 8]\n\
         assert np.asarray(a['be_scalar_attr']).shape == ()\n\
         assert float(a['be_scalar_attr']) == 2.5\n\
         assert np.asarray(a['half_attr']).dtype == np.float16\n\
         assert float(a['half_attr']) == 0.25\n\
         f.close()\n\
         print('OK')",
        p = path.display()
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output();
    let _ = std::fs::remove_file(&path);
    match out {
        Ok(o) if o.status.success() => {
            assert!(
                String::from_utf8_lossy(&o.stdout).contains("OK"),
                "h5py check produced no OK: {}",
                String::from_utf8_lossy(&o.stdout)
            );
        }
        // A python3 that exists but *fails* is a real signal, not an
        // environment gap: distinguish "not installed" from "assertion failed".
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            if stderr.contains("ModuleNotFoundError") || stderr.contains("No module named") {
                eprintln!("h5py unavailable — skipping the big-endian/float16 conformance check");
            } else {
                panic!("h5py rejected the writer's output:\n{stderr}");
            }
        }
        Err(_) => eprintln!("python3 unavailable — skipping the big-endian/float16 check"),
    }
}

/// The reader must report the declared byte order and give the values back —
/// the always-on guard that writer and reader agree.
#[test]
fn oxih5_reads_back_big_endian_datasets() {
    let path = tmp_path("selfread.h5");
    let _ = std::fs::remove_file(&path);
    build_fixture(&path);
    let f = oxih5::open(&path).expect("open");

    let be = ByteOrder::Big;
    let ds = f.dataset("be_f32").expect("be_f32");
    assert_eq!(ds.dtype, Dtype::Float { size: 4, order: be });
    assert_eq!(ds.as_f32().expect("f32"), vec![1.5, -2.25, 3.0]);

    let ds = f.dataset("be_i64").expect("be_i64");
    assert_eq!(
        ds.dtype,
        Dtype::Int {
            size: 8,
            signed: true,
            order: be
        }
    );
    assert_eq!(ds.as_i64().expect("i64"), vec![-1, 2]);

    let ds = f.dataset("be_u64").expect("be_u64");
    assert_eq!(ds.as_u64().expect("u64"), vec![1, u64::MAX]);

    let ds = f.dataset("be_2d").expect("be_2d");
    assert_eq!(ds.shape, vec![2, 3]);
    assert_eq!(
        ds.as_f64().expect("f64"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    // The little-endian twin must still be little-endian.
    assert_eq!(
        f.dataset("le_f32").expect("le_f32").dtype,
        Dtype::Float {
            size: 4,
            order: ByteOrder::Little
        }
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn oxih5_reads_back_half_precision() {
    let path = tmp_path("f16.h5");
    let _ = std::fs::remove_file(&path);
    build_fixture(&path);
    let f = oxih5::open(&path).expect("open");

    let ds = f.dataset("f16").expect("f16");
    assert_eq!(
        ds.dtype,
        Dtype::Float {
            size: 2,
            order: ByteOrder::Little
        }
    );
    // `as_f32` is for binary32; binary16 has its own accessor.
    let values = ds.as_f16().expect("f16");
    assert_eq!(values[..4], [0.5, -1.5, 1.0, 65504.0]);
    // 2^-14, the smallest normal binary16, survives exactly.
    assert!((values[4] - 6.103_515_6e-5).abs() < f32::EPSILON);

    let be = f.dataset("be_f16").expect("be_f16");
    assert_eq!(
        be.dtype,
        Dtype::Float {
            size: 2,
            order: ByteOrder::Big
        }
    );
    assert_eq!(be.as_f16().expect("f16"), vec![0.5, -1.5]);
    let _ = std::fs::remove_file(&path);
}

/// A big-endian dataset's bytes must actually differ from its little-endian
/// twin's — the failure this whole feature exists to prevent is a file that
/// *claims* big-endian while holding little-endian bytes, which self-round-trips
/// perfectly and reads back swapped everywhere else.
#[test]
fn big_endian_payload_is_actually_byte_swapped() {
    let mut w = FileWriter::new();
    w.write_dataset_numeric(
        "be",
        NumericValues::U32(&[0x0102_0304]),
        ByteOrder::Big,
        &[1],
    )
    .expect("be");
    w.write_dataset_numeric(
        "le",
        NumericValues::U32(&[0x0102_0304]),
        ByteOrder::Little,
        &[1],
    )
    .expect("le");
    let bytes = w.build_to_vec().expect("build");
    assert!(
        bytes.windows(4).any(|c| c == [0x01, 0x02, 0x03, 0x04]),
        "the big-endian payload 01 02 03 04 must be present verbatim"
    );
    assert!(
        bytes.windows(4).any(|c| c == [0x04, 0x03, 0x02, 0x01]),
        "the little-endian payload 04 03 02 01 must be present verbatim"
    );
}

/// The value-count guards on the two attribute entry points.
#[test]
fn numeric_attr_arity_is_checked() {
    let mut w = FileWriter::new();
    w.write_dataset_f32("d", &[1.0], &[1]).expect("dataset");
    let err = w
        .write_numeric_attr("d", "empty", NumericValues::I32(&[]), ByteOrder::Little)
        .expect_err("an empty array attribute must be refused");
    assert!(err.to_string().contains("at least one value"), "{err}");
    let err = w
        .write_numeric_scalar_attr("d", "two", NumericValues::I32(&[1, 2]), ByteOrder::Little)
        .expect_err("a two-value scalar attribute must be refused");
    assert!(err.to_string().contains("exactly one value"), "{err}");
}
