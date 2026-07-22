// LANE OHRD regression tests — object-header emission + reader correctness.
//
// Covers the four confirmed findings owned by this lane:
//   B004  zero-length contiguous dataset must write the UNDEFINED data address
//         (all ones) with size 0, not an aliasing defined address libhdf5 reads
//         as corruption; oxih5's own reader must tolerate both.
//   B007  a chunked dataset's sparse/unallocated chunks must read back the
//         declared fill value, not hard-coded zeros.
//   B013  a v1 attribute value must be trimmed to `n_elems * elem_size`; the
//         object-header 8-byte alignment padding must not fold into the value.
//   B023  a chunked dataset's fill-value message must record space-allocation
//         time = Incremental(3), matching libhdf5 (contiguous stays Late(2)).
//
// Pre-fix fixtures under tests/fixtures/ohrd_* were generated with the current
// (pre-fix) oxih5 writer and with h5py/libhdf5, so the reader-side changes are
// proven backward-compatible against real old bytes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxih5::{ByteOrder, Dtype, File, FileWriter};
use std::sync::atomic::{AtomicU64, Ordering};

static OHRD_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = OHRD_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_ohrd_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

/// Does `haystack` contain the byte sequence `needle`?
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Run a Python script (which reads/writes `path`) and either succeed, skip
/// (python3 or h5py absent), or fail loudly on a genuine assertion error — the
/// same three-way guard the crate's `h5py_check` uses, so a real interop
/// failure can never masquerade as a skip.
fn h5py_run(path: &std::path::Path, script: &str, what: &str) {
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

fn dt_i32() -> Dtype {
    Dtype::Int {
        size: 4,
        signed: true,
        order: ByteOrder::Little,
    }
}

fn dt_f64() -> Dtype {
    Dtype::Float {
        size: 8,
        order: ByteOrder::Little,
    }
}

// ===========================================================================
// B004 — zero-length contiguous dataset: undefined address, size 0
// ===========================================================================

/// The empty contiguous layout message must carry the UNDEFINED address
/// (`0xFFFF_FFFF_FFFF_FFFF`) with size 0.  Pre-fix the writer left the write
/// cursor (a defined, aliasing address) here, so this byte pattern is absent.
#[test]
fn b004_empty_contiguous_layout_uses_undefined_address() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("data", &[], &[0]).unwrap();
    let bytes = w.build_to_vec().unwrap();

    // v3 contiguous layout body: version(03) class(01) addr(8) size(8).
    let mut needle = vec![0x03u8, 0x01];
    needle.extend_from_slice(&[0xFFu8; 8]); // H5_ADDR_UNDEF
    needle.extend_from_slice(&[0x00u8; 8]); // size == 0
    assert!(
        contains(&bytes, &needle),
        "empty contiguous layout must use the undefined address with size 0"
    );
}

/// oxih5's own reader round-trips a freshly written empty dataset.
#[test]
fn b004_roundtrip_empty_dataset() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("data", &[], &[0]).unwrap();
    let bytes = w.build_to_vec().unwrap();

    let file = File::open_from_bytes(&bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape, vec![0]);
    assert!(ds.as_i32().unwrap().is_empty());
}

/// Backward compatibility: a pre-fix oxih5 file (defined, aliasing address with
/// size 0) must still read as an empty dataset through the fixed reader.
#[test]
fn b004_reader_reads_prefix_oxih5_empty() {
    const PREFIX: &[u8] = include_bytes!("fixtures/ohrd_b004_empty_oxih5_prefix.h5");
    let file = File::open_from_bytes(PREFIX).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape, vec![0]);
    assert!(ds.as_i32().unwrap().is_empty());
}

/// The reader must handle the UNDEFINED address as written by real libhdf5
/// (h5py `libver='earliest'` empty dataset), returning an empty buffer rather
/// than dereferencing `u64::MAX`.
#[test]
fn b004_reader_reads_h5py_empty() {
    const GOLD: &[u8] = include_bytes!("fixtures/ohrd_b004_empty_h5py.h5");
    let file = File::open_from_bytes(GOLD).unwrap();
    let ds = file.dataset("z").unwrap();
    assert_eq!(ds.shape, vec![0]);
    assert!(ds.as_f64().unwrap().is_empty());
}

/// libhdf5 (via h5py) must accept an oxih5-written empty dataset instead of
/// rejecting it as "invalid dataset size, likely file corruption".
#[test]
fn b004_h5py_accepts_empty_dataset() {
    let path = tmp_path("b004_empty");
    let mut w = FileWriter::new();
    w.write_dataset_i32("data", &[], &[0]).unwrap();
    w.build(&path).unwrap();

    let script = format!(
        "import h5py\n\
         f = h5py.File({path:?}, 'r')\n\
         d = f['data']\n\
         assert d.shape == (0,), repr(d.shape)\n\
         assert d[...].shape == (0,), repr(d[...].shape)\n\
         print('h5py opened empty dataset, shape', d.shape)\n",
        path = path.display()
    );
    h5py_run(&path, &script, "B004 h5py empty-dataset accept");
}

// ===========================================================================
// B023 — chunked fill-value message uses Incremental(3); contiguous stays Late(2)
// ===========================================================================

/// Full on-disk fill-value message (8-byte header + 8-byte body) for a given
/// space-allocation-time byte.
fn fill_value_message(alloc_time: u8) -> [u8; 16] {
    [
        0x05, 0x00, 0x08, 0x00, 0x01, 0x00, 0x00,
        0x00, // header: type 0x0005, size 8, flag 0x01
        0x02, alloc_time, 0x02, 0x01, 0x00, 0x00, 0x00,
        0x00, // body: v2, alloc, never, defined
    ]
}

#[test]
fn b023_chunked_fill_value_is_incremental() {
    let mut w = FileWriter::new();
    let raw: Vec<u8> = (0..3i32).flat_map(i32::to_le_bytes).collect();
    w.create_dataset_unlimited("c", &[3], &[3], &dt_i32(), &raw)
        .unwrap();
    let bytes = w.build_to_vec().unwrap();

    assert!(
        contains(&bytes, &fill_value_message(0x03)),
        "chunked dataset fill-value must record Incremental(3) allocation time"
    );
    assert!(
        !contains(&bytes, &fill_value_message(0x02)),
        "chunked dataset must not keep the Late(2) allocation time"
    );
}

#[test]
fn b023_contiguous_fill_value_stays_late() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("d", &[1, 2, 3], &[3]).unwrap();
    let bytes = w.build_to_vec().unwrap();

    assert!(
        contains(&bytes, &fill_value_message(0x02)),
        "contiguous dataset fill-value must stay Late(2)"
    );
    assert!(
        !contains(&bytes, &fill_value_message(0x03)),
        "contiguous dataset must not use Incremental(3)"
    );
}

/// h5py must still read an oxih5 chunked dataset correctly after the
/// allocation-time change — the deviation is tolerated, not corrupting.
#[test]
fn b023_h5py_reads_chunked_after_alloc_time_change() {
    let path = tmp_path("b023_chunked");
    let raw: Vec<u8> = (0..4).flat_map(|i| f64::from(i).to_le_bytes()).collect();
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("t", &[4], &[2], &dt_f64(), &raw)
        .unwrap();
    w.build(&path).unwrap();

    let script = format!(
        "import h5py\n\
         f = h5py.File({path:?}, 'r')\n\
         d = f['t']\n\
         assert d.shape == (4,), repr(d.shape)\n\
         assert d.chunks == (2,), repr(d.chunks)\n\
         assert list(d[:]) == [0.0, 1.0, 2.0, 3.0], repr(list(d[:]))\n\
         assert d.fillvalue == 0.0, repr(d.fillvalue)\n\
         print('h5py read chunked dataset, alloctime', d.id.get_create_plist().get_alloc_time())\n",
        path = path.display()
    );
    h5py_run(&path, &script, "B023 h5py chunked read");
}

// ===========================================================================
// B007 — chunked sparse chunks read the declared fill value, not zeros
// ===========================================================================

/// i32, chunks (3,3), gzip, fillvalue=-7, only chunk (0,0) written (h5py).
/// Pre-fix the primary read returned 0 in every unwritten position.
#[test]
fn b007_sparse_i32_reads_fill_value() {
    const F: &[u8] = include_bytes!("fixtures/ohrd_b007_sparse_i32_fill_neg7.h5");
    let file = File::open_from_bytes(F).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.shape, vec![10, 10]);
    let v = ds.as_i32().unwrap();
    assert_eq!(v[0], 5, "written chunk value");
    assert_eq!(
        v[5 * 10 + 5],
        -7,
        "unwritten position must read the fill value"
    );
    let mut uniq = v.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq, vec![-7, 5]);
}

/// f64, chunks (2,2), fillvalue=-999.5, only chunk (0,0) written.
#[test]
fn b007_sparse_f64_reads_fill_value() {
    const F: &[u8] = include_bytes!("fixtures/ohrd_b007_sparse_f64_fill_neg999.h5");
    let file = File::open_from_bytes(F).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.shape, vec![6, 6]);
    let v = ds.as_f64().unwrap();
    assert_eq!(v[0], 1.5, "written chunk value");
    assert_eq!(
        v[5 * 6 + 5],
        -999.5,
        "unwritten position must read the fill value"
    );
}

/// Default fill (h5py fillvalue 0): sparse gaps must read as 0 — the fix must
/// not regress the zero-fill case.
#[test]
fn b007_sparse_default_fill_reads_zero() {
    const F: &[u8] = include_bytes!("fixtures/ohrd_b007_sparse_i32_default0.h5");
    let file = File::open_from_bytes(F).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.shape, vec![6, 6]);
    let v = ds.as_i32().unwrap();
    assert_eq!(v[0], 7, "written chunk value");
    assert_eq!(v[5 * 6 + 5], 0, "unwritten position defaults to 0");
}

/// The primary full read and the hyperslab (slice) read must now agree on the
/// fill value — the disagreement was the whole bug.
#[test]
fn b007_primary_and_slice_agree_on_fill() {
    const F: &[u8] = include_bytes!("fixtures/ohrd_b007_sparse_i32_fill_neg7.h5");
    let file = File::open_from_bytes(F).unwrap();
    let primary = file.dataset("d").unwrap().as_i32().unwrap();
    let slice = file
        .dataset_slice("d", &[0..10, 0..10])
        .unwrap()
        .as_i32()
        .unwrap();
    assert_eq!(primary, slice);
    assert_eq!(slice[5 * 10 + 5], -7);
}

// ===========================================================================
// B013 — attribute value trimmed to element count, no folded padding
// ===========================================================================

/// A freshly written i32 attribute value must come back as exactly 4 bytes
/// (not 8 with alignment padding), and a fixed string as its exact length.
#[test]
fn b013_oxih5_roundtrip_attr_data_is_exact() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("ds", &[0i32], &[1]).unwrap();
    w.write_i32_attr("ds", "_FillValue", -9999).unwrap();
    w.write_string_attr("ds", "units", "meters").unwrap();
    w.write_i64_attr("ds", "count", 5).unwrap();
    w.write_f64_attr("ds", "scale", 0.5).unwrap();
    let bytes = w.build_to_vec().unwrap();

    let file = File::open_from_bytes(&bytes).unwrap();
    let ds = file.dataset("ds").unwrap();
    let find = |name: &str| ds.attributes.iter().find(|a| a.name == name).unwrap();

    let fv = find("_FillValue");
    assert_eq!(fv.data, (-9999i32).to_le_bytes().to_vec());
    assert_eq!(fv.data.len(), 4, "i32 attr payload must be exactly 4 bytes");
    assert_eq!(fv.as_i64(), Some(-9999));

    let units = find("units");
    assert_eq!(units.data, b"meters".to_vec());
    assert_eq!(units.data.len(), 6);

    // 8-byte payloads are naturally aligned and must remain 8 bytes.
    assert_eq!(find("count").data.len(), 8);
    assert_eq!(find("scale").data.len(), 8);
}

/// Backward compatibility: a pre-fix oxih5 file (whose attribute messages carry
/// the padded declared size, byte-identical to 0.2.0/0.2.1) must read the
/// values back at their exact lengths through the fixed reader.
#[test]
fn b013_reads_prefix_oxih5_padded_attrs() {
    const PREFIX: &[u8] = include_bytes!("fixtures/ohrd_b013_attrs_oxih5_prefix.h5");
    let file = File::open_from_bytes(PREFIX).unwrap();
    let ds = file.dataset("ds").unwrap();
    let find = |name: &str| ds.attributes.iter().find(|a| a.name == name).unwrap();

    let fv = find("_FillValue");
    assert_eq!(fv.data.len(), 4);
    assert_eq!(fv.as_i64(), Some(-9999));

    let units = find("units");
    assert_eq!(units.data, b"meters".to_vec());
}

/// The same must hold for attributes authored by real libhdf5 (h5py
/// `libver='earliest'`), which the pre-fix reader also over-read.
#[test]
fn b013_reads_h5py_authored_attrs() {
    const H5PY: &[u8] = include_bytes!("fixtures/ohrd_b013_attrs_h5py.h5");
    let file = File::open_from_bytes(H5PY).unwrap();
    let ds = file.dataset("ds").unwrap();
    let find = |name: &str| ds.attributes.iter().find(|a| a.name == name).unwrap();

    let fv = find("_FillValue");
    assert_eq!(
        fv.data.len(),
        4,
        "libhdf5 i32 attr must read back as 4 bytes"
    );
    assert_eq!(fv.as_i64(), Some(-9999));

    let units = find("units");
    assert_eq!(units.data, b"meters".to_vec());
}
