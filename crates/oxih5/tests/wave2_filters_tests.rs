//! Wave 2 — writable filter pipeline (G001 shuffle, G007 fletcher32) and
//! fixed-maxshape tiling (G010).
//!
//! Every filter feature is validated against the reference reader (h5py /
//! libhdf5) in **both** directions — oxih5 writes, h5py reads exact values and
//! reports the filters; h5py writes, oxih5 reads them back — plus a pure oxih5
//! write→read round-trip that needs no Python.  The h5py checks use the
//! campaign's three-way graceful skip: python3 missing → skip, h5py/numpy
//! missing → skip, any *other* non-zero exit → fail, so a silent skip never
//! hides a real regression.  fletcher32 is additionally checked with a negative
//! case: a corrupted chunk must make h5py error, which proves the checksum is
//! the byte-exact one libhdf5 verifies.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxih5::FileWriter;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_w2_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

/// Run an h5py check over `path`, then delete it.
///
/// python3 missing → skip, h5py/numpy missing → skip, any other non-zero exit
/// → fail.  Mirrors `write_tests::h5py_check`.
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
        Err(_) => eprintln!("python3 not found — skipping {what}"),
    }
}

/// Have h5py *write* a file from `script` (the script embeds its own path).
/// Returns `true` if the file was created, `false` if the check was skipped
/// (python3 / h5py absent); any other failure panics.  The caller uses the
/// boolean to short-circuit an interop test when the toolchain is missing.
#[must_use]
fn h5py_write(script: &str, what: &str) -> bool {
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
                    "{what} (h5py generator) FAILED:\nstderr: {stderr}\nstdout: {}",
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

// ===========================================================================
// oxih5 writes, h5py reads (exact values + reported filters)
// ===========================================================================

/// shuffle+deflate over a fixed-maxshape 2-D dataset with ragged edge chunks:
/// h5py must read every value, and report shuffle + gzip + the exact level and
/// geometry — this is the standard NetCDF-4 integer compression combo.
#[test]
fn shuffle_deflate_2d_read_by_h5py() {
    for level in [1u8, 6, 9] {
        let path = tmp_path(&format!("shufdef_2d_l{level}"));
        // 6×8, chunks 2×3 → 3×3 grid, the last column of chunks hangs over 8/3.
        let values: Vec<i32> = (0..48).collect();
        let mut w = FileWriter::new();
        w.write_dataset_i32("d", &values, &[6, 8]).unwrap();
        w.set_chunking("d", &[2, 3]).unwrap();
        w.set_shuffle("d").unwrap();
        w.set_deflate("d", level).unwrap();
        w.build(&path).unwrap();

        let want: String = (0..48).map(|i| format!("{i},")).collect();
        let script = format!(
            "import h5py, numpy as np\n\
             f=h5py.File({p:?},'r'); d=f['d']\n\
             assert d.shape==(6,8), d.shape\n\
             assert d.maxshape==(6,8), ('maxshape',d.maxshape)\n\
             assert d.chunks==(2,3), ('chunks',d.chunks)\n\
             assert d.shuffle==True, 'shuffle flag not set'\n\
             assert d.compression=='gzip', ('compression',d.compression)\n\
             assert d.compression_opts=={level}, ('opts',d.compression_opts)\n\
             got=list(d[:].reshape(-1)); want=[{want}]\n\
             assert got==want, ('values',got[:8])\n\
             print('shuffle+deflate 2d level {level} OK')\n",
            p = path.display(),
        );
        h5py_check(&path, &script, "shuffle+deflate 2d");
    }
}

/// All three filters together over a tiled 1-D dataset: h5py reads the exact
/// values and its create-plist lists the filters in the canonical order
/// shuffle(2), deflate(1), fletcher32(3).
#[test]
fn all_three_filters_read_by_h5py() {
    let path = tmp_path("all3_1d");
    let values: Vec<f64> = (0..200).map(|i| f64::from(i % 7) * 0.25).collect();
    let mut w = FileWriter::new();
    w.write_dataset_f64("d", &values, &[200]).unwrap();
    w.set_chunking("d", &[64]).unwrap(); // 4 chunks, last ragged (200/64)
    w.set_fletcher32("d").unwrap(); // set out of canonical order on purpose
    w.set_shuffle("d").unwrap();
    w.set_deflate("d", 6).unwrap();
    w.build(&path).unwrap();

    let want: String = values.iter().map(|v| format!("{v},")).collect();
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File({p:?},'r'); d=f['d']\n\
         assert d.shuffle==True and d.fletcher32==True and d.compression=='gzip'\n\
         ids=[d.id.get_create_plist().get_filter(i)[0] \
              for i in range(d.id.get_create_plist().get_nfilters())]\n\
         assert ids==[2,1,3], ('filter order',ids)\n\
         got=list(d[:]); want=[{want}]\n\
         assert all(abs(a-b)<1e-12 for a,b in zip(got,want)) and len(got)==len(want), 'values'\n\
         print('all-three 1d OK, filter ids',ids)\n",
        p = path.display(),
    );
    h5py_check(&path, &script, "all-three filters");
}

/// fletcher32 on its own: h5py verifies the checksum natively when it reads the
/// chunk, so a clean read proves the checksum is byte-exact with libhdf5.
#[test]
fn fletcher32_read_by_h5py() {
    let path = tmp_path("fl_only");
    let values: Vec<i32> = (0..100).map(|i| i * 3 - 50).collect();
    let mut w = FileWriter::new();
    w.write_dataset_i32("d", &values, &[100]).unwrap();
    w.set_chunking("d", &[32]).unwrap(); // multi-chunk, ragged
    w.set_fletcher32("d").unwrap();
    w.build(&path).unwrap();

    let want: String = values.iter().map(|v| format!("{v},")).collect();
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File({p:?},'r'); d=f['d']\n\
         assert d.fletcher32==True, 'fletcher32 flag not set'\n\
         got=list(d[:]); want=[{want}]\n\
         assert got==want, 'values'\n\
         print('fletcher32 verified by h5py OK')\n",
        p = path.display(),
    );
    h5py_check(&path, &script, "fletcher32 clean read");
}

/// A corrupted fletcher32 chunk must make h5py **error**.  The dataset is
/// uncompressed (fletcher32 only, one chunk), so the raw little-endian payload
/// is on disk verbatim; flipping one of its bytes leaves the stored checksum
/// disagreeing with the data, which libhdf5 rejects on read.
#[test]
fn corrupt_fletcher32_makes_h5py_error() {
    let path = tmp_path("fl_corrupt");
    // Distinctive values so the payload is easy to find in the file.
    let values: Vec<i32> = vec![
        0x1122_3344,
        0x5566_7788,
        0x0a0b_0c0d,
        0x1314_1516,
        0x2122_2324,
    ];
    let mut w = FileWriter::new();
    w.write_dataset_i32("d", &values, &[5]).unwrap();
    w.set_fletcher32("d").unwrap(); // single whole-dataset chunk, no compression
    w.build(&path).unwrap();

    // Corrupt one payload byte in place.
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut bytes = std::fs::read(&path).unwrap();
    let at = bytes
        .windows(payload.len())
        .position(|w| w == payload.as_slice())
        .expect("raw payload must appear verbatim in an uncompressed chunk");
    bytes[at] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    // h5py must raise on the read; the script exits 0 only if it did.
    let script = format!(
        "import h5py\n\
         f=h5py.File({p:?},'r'); d=f['d']\n\
         try:\n\
         \x20   _=d[:]\n\
         \x20   raise SystemExit('BUG: corrupt fletcher32 chunk read without error')\n\
         except OSError:\n\
         \x20   print('h5py correctly rejected the corrupt checksum')\n",
        p = path.display(),
    );
    h5py_check(&path, &script, "fletcher32 corruption rejected");
}

// ===========================================================================
// h5py writes, oxih5 reads (the other direction)
// ===========================================================================

/// oxih5 reads an h5py-written shuffle+gzip dataset back to exact values, over a
/// multi-chunk 2-D grid with ragged edges.
#[test]
fn oxih5_reads_h5py_shuffle_deflate() {
    let path = tmp_path("h5py_shufdef");
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File({p:?},'w',libver='earliest')\n\
         a=np.arange(48,dtype='int32').reshape(6,8)\n\
         f.create_dataset('d',data=a,chunks=(2,3),shuffle=True,compression='gzip',compression_opts=6)\n\
         f.close()\n",
        p = path.display(),
    );
    if !h5py_write(&script, "gen h5py shuffle+deflate") {
        return;
    }
    let file = oxih5::open(&path).unwrap();
    cleanup(&path);
    let d = file.dataset("d").unwrap();
    assert_eq!(d.shape, vec![6, 8]);
    assert_eq!(d.as_i32().unwrap(), (0..48).collect::<Vec<i32>>());
}

/// oxih5 reads an h5py-written all-three (shuffle+gzip+fletcher32) dataset,
/// including verifying and stripping the fletcher32 checksum on every chunk.
#[test]
fn oxih5_reads_h5py_all_three() {
    let path = tmp_path("h5py_all3");
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File({p:?},'w',libver='earliest')\n\
         a=np.arange(300,dtype='float64')*0.5\n\
         f.create_dataset('d',data=a,chunks=(64,),shuffle=True,\
             compression='gzip',compression_opts=9,fletcher32=True)\n\
         f.close()\n",
        p = path.display(),
    );
    if !h5py_write(&script, "gen h5py all-three") {
        return;
    }
    let file = oxih5::open(&path).unwrap();
    cleanup(&path);
    let d = file.dataset("d").unwrap();
    assert_eq!(d.shape, vec![300]);
    let got = d.as_f64().unwrap();
    let want: Vec<f64> = (0..300).map(|i| f64::from(i) * 0.5).collect();
    assert_eq!(got, want);
}

// ===========================================================================
// Pure oxih5 write → read round-trip (no Python), every filter combination
// ===========================================================================

/// Build a dataset with the given filters and geometry, read it back through
/// oxih5's own reader, and assert bit-exact recovery.
fn oxih5_roundtrip_i32(
    tag: &str,
    shape: &[usize],
    chunk: &[usize],
    filters: (bool, Option<u8>, bool),
) {
    let (shuffle, deflate, fletcher32) = filters;
    let n: usize = shape.iter().product();
    let values: Vec<i32> = (0..n as i32).map(|i| i * 7 - 11).collect();
    let path = tmp_path(tag);
    let mut w = FileWriter::new();
    w.write_dataset_i32("d", &values, shape).unwrap();
    w.set_chunking("d", chunk).unwrap();
    if shuffle {
        w.set_shuffle("d").unwrap();
    }
    if let Some(level) = deflate {
        w.set_deflate("d", level).unwrap();
    }
    if fletcher32 {
        w.set_fletcher32("d").unwrap();
    }
    w.build(&path).unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);
    let d = file.dataset("d").unwrap();
    assert_eq!(d.shape, shape.to_vec(), "{tag} shape");
    assert_eq!(d.as_i32().unwrap(), values, "{tag} values");
}

#[test]
fn oxih5_roundtrips_every_filter_combination() {
    // 1-D ragged (7/3) and 2-D ragged in both axes (5×3 by 2×2).
    let geometries: &[(&str, &[usize], &[usize])] = &[("1d", &[7], &[3]), ("2d", &[5, 3], &[2, 2])];
    let combos: &[(bool, Option<u8>, bool)] = &[
        (true, None, false),
        (false, Some(1), false),
        (false, Some(6), false),
        (false, Some(9), false),
        (false, None, true),
        (true, Some(6), false),
        (true, None, true),
        (false, Some(9), true),
        (true, Some(6), true),
    ];
    for (gtag, shape, chunk) in geometries {
        for (i, &combo) in combos.iter().enumerate() {
            oxih5_roundtrip_i32(&format!("rt_{gtag}_{i}"), shape, chunk, combo);
        }
    }
}

// ===========================================================================
// G010 — fixed-maxshape tiling
// ===========================================================================

/// A tiled, compressed, *fixed* dataset must keep `maxshape == shape` — h5py
/// reports a bounded shape, not the `(None, …)` an unlimited dimension carries.
#[test]
fn fixed_maxshape_tiled_reports_bounded_shape_to_h5py() {
    let path = tmp_path("g010_fixed");
    let values: Vec<i32> = (0..64).collect();
    let mut w = FileWriter::new();
    w.write_dataset_i32("raster", &values, &[8, 8]).unwrap();
    w.set_chunking("raster", &[4, 4]).unwrap();
    w.set_deflate("raster", 6).unwrap();
    w.build(&path).unwrap();

    let want: String = (0..64).map(|i| format!("{i},")).collect();
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File({p:?},'r'); d=f['raster']\n\
         assert d.shape==(8,8), d.shape\n\
         assert d.maxshape==(8,8), ('maxshape must be bounded, not None',d.maxshape)\n\
         assert d.chunks==(4,4), ('chunks',d.chunks)\n\
         assert d.compression=='gzip'\n\
         got=list(d[:].reshape(-1)); want=[{want}]\n\
         assert got==want, 'values'\n\
         print('fixed-maxshape tiled OK, maxshape',d.maxshape)\n",
        p = path.display(),
    );
    h5py_check(&path, &script, "G010 fixed maxshape");
}

/// The unlimited tiled path still marks dimension 0 as unlimited — so the two
/// APIs are genuinely different, and G010 did not turn every tiled dataset
/// fixed.
#[test]
fn unlimited_tiled_reports_none_maxshape_to_h5py() {
    let path = tmp_path("g010_unlimited");
    let dtype = oxih5_core::Dtype::Int {
        size: 4,
        signed: true,
        order: oxih5_core::ByteOrder::Little,
    };
    let raw: Vec<u8> = (0..16i32).flat_map(i32::to_le_bytes).collect();
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("t", &[16], &[4], &dtype, &raw)
        .unwrap();
    w.set_deflate("t", 6).unwrap();
    w.build(&path).unwrap();

    let script = format!(
        "import h5py\n\
         f=h5py.File({p:?},'r'); d=f['t']\n\
         assert d.shape==(16,), d.shape\n\
         assert d.maxshape==(None,), ('unlimited dim0 expected',d.maxshape)\n\
         print('unlimited tiled OK, maxshape',d.maxshape)\n",
        p = path.display(),
    );
    h5py_check(&path, &script, "unlimited maxshape contrast");
}

/// `set_chunking` enforces the geometry rules at call time and leaves the
/// dataset untouched when it refuses — a rejected call must be a no-op.
#[test]
fn set_chunking_validates_and_leaves_no_trace() {
    // A chunk extent wider than a fixed dimension is refused (B024 coherence),
    // and the refusal must not have converted the storage: a following valid
    // set_chunking + build still succeeds.
    let mut w = FileWriter::new();
    w.write_dataset_i32("d", &(0..8).collect::<Vec<_>>(), &[8])
        .unwrap();
    let err = w
        .set_chunking("d", &[16])
        .expect_err("chunk wider than fixed dim");
    assert!(format!("{err}").contains("fixed dimension"), "{err}");
    // A zero chunk extent, more chunk dims than the dataspace, and non-dataset
    // targets are all refused too.
    assert!(w.set_chunking("d", &[0]).is_err(), "zero extent");
    assert!(w.set_chunking("d", &[2, 2]).is_err(), "too many chunk dims");
    assert!(w.set_chunking("missing", &[2]).is_err(), "no such dataset");

    // The dataset survived every refusal intact and still writes and reads back.
    let path = tmp_path("g010_noleak");
    w.set_chunking("d", &[3]).unwrap();
    w.build(&path).unwrap();
    let file = oxih5::open(&path).unwrap();
    cleanup(&path);
    assert_eq!(
        file.dataset("d").unwrap().as_i32().unwrap(),
        (0..8).collect::<Vec<i32>>()
    );
}

/// Filters refuse a scalar and a vlen-string dataset, with reasons — the two
/// shapes that have no chunked form.
#[test]
fn filters_refuse_scalar_and_vlen() {
    let mut w = FileWriter::new();
    w.write_dataset_f64("scalar", &[1.0], &[]).unwrap();
    w.create_vlen_string_dataset("labels", &["a", "b"]).unwrap();

    for (path, is_vlen) in [("scalar", false), ("labels", true)] {
        let needle = if is_vlen {
            "global-heap references"
        } else {
            "scalar"
        };
        for setter in ["shuffle", "fletcher32", "chunking", "deflate"] {
            let err = match setter {
                "shuffle" => w.set_shuffle(path).unwrap_err(),
                "fletcher32" => w.set_fletcher32(path).unwrap_err(),
                "chunking" => w.set_chunking(path, &[1]).unwrap_err(),
                _ => w.set_deflate(path, 6).unwrap_err(),
            };
            assert!(
                format!("{err}").contains(needle),
                "{setter} on {path}: {err}"
            );
        }
    }
}
