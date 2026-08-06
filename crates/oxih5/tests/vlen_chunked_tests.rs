//! Integration tests for variable-length (datatype class 9) elements stored in
//! chunked datasets, read both in full and via a hyperslab selection.
//!
//! On disk each vlen element is a fixed-size 16-byte global-heap reference, so a
//! chunked vlen dataset can be assembled chunk-by-chunk exactly like a
//! fixed-width one; the references are then dereferenced through the global heap.
//! The `vlen_str_chunked.h5` fixture is written by h5py with `chunks=(3,)` over a
//! length-10 dataset, so several elements straddle chunk boundaries.

use std::path::Path;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn expected() -> Vec<String> {
    (0..10).map(|i| format!("item-{i:02}-value")).collect()
}

#[test]
fn chunked_vlen_strings_full() {
    let path = fixture("vlen_str_chunked.h5");
    // This fixture is committed to git (`tests/fixtures/vlen_str_chunked.h5`);
    // a missing file means a broken checkout, not an environment where the
    // test should be silently skipped — fail loudly instead of vacuously
    // passing (see tests/gen_fixtures.py for how it's regenerated).
    assert!(
        path.exists(),
        "fixture vlen_str_chunked.h5 missing at {} — it is tracked in git",
        path.display()
    );
    let file = oxih5::open(&path).expect("open chunked vlen fixture");
    let strings = file
        .dataset_strings("words")
        .expect("decode chunked vlen strings");
    assert_eq!(strings, expected());
}

#[test]
fn chunked_vlen_strings_hyperslab() {
    let path = fixture("vlen_str_chunked.h5");
    // See `chunked_vlen_strings_full` above: this fixture is tracked in git,
    // so a missing file must fail the test, not silently skip it.
    assert!(
        path.exists(),
        "fixture vlen_str_chunked.h5 missing at {} — it is tracked in git",
        path.display()
    );
    let file = oxih5::open(&path).expect("open chunked vlen fixture");
    // Select elements 2..7, which spans chunk boundaries (chunks of 3).
    let ranges: Vec<std::ops::Range<usize>> = std::iter::once(2..7).collect();
    let slice = file
        .dataset_slice("words", &ranges)
        .expect("hyperslab slice of chunked vlen");
    let raw = std::fs::read(&path).expect("read file bytes");
    let strings = oxih5_format::values::decode_vlen_strings(&raw, &slice.data, slice.len())
        .expect("decode sliced vlen references");
    assert_eq!(strings, &expected()[2..7]);
}
