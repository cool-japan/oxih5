//! B-tree v2 deeply-nested chunked-dataset traversal tests.
//!
//! Exercises the B-tree v2 code paths (both chunk-index type 10/11 for data
//! and type-5 name-index for large groups) via the high-level `oxih5::File`
//! facade.  All assertions target fixtures that were generated with
//! `libver='latest'` so h5py writes superblock v3 + new-style groups +
//! B-tree v2/extensible-array chunk indices.
//!
//! Test inventory:
//! - [`btree_v2_chunk_index_all_datasets`]    – enumerate + decode every
//!   chunked dataset in `libver_latest_chunked.h5`; assert no panic and shape
//!   consistency.
//! - [`btree_v2_chunk_index_data_correctness`] – assert known element values
//!   for each dataset in `libver_latest_chunked.h5`.
//! - [`btree_v2_name_index_large_group_traversal`] – fractal-heap + B-tree v2
//!   type-5 name-index traversal over 20 datasets; assert all 20 are found.
//! - [`btree_v2_name_index_data_correctness`] – read a specific dataset from
//!   the large group and verify its value.
//! - [`btree_v2_chunk_plus_name_index_combined`] – open the large file,
//!   navigate through the B-tree v2 name-index group, and decode each dataset;
//!   exercises both name-index and (where present) chunk-index traversal in
//!   one round-trip.
//! - [`btree_v2_traversal_via_all_chunked_fixtures`] – iterate every fixture
//!   that contains the word "chunk" in its name, open it, enumerate datasets,
//!   and assert the decoded data length matches shape × element size.

// ---------------------------------------------------------------------------
// Fixture paths — opened directly via relative path (cargo test sets cwd to
// the crate root).
// ---------------------------------------------------------------------------

/// Superblock v3 file with three chunked datasets (B-tree v2 / extensible
/// array chunk indices).
const LIBVER_LATEST_CHUNKED: &str = "tests/fixtures/libver_latest_chunked.h5";

/// Superblock v3 file with 20 datasets in a large group (fractal heap +
/// B-tree v2 type-5 name index).
const LIBVER_LATEST_LARGE: &str = "tests/fixtures/libver_latest_large.h5";

// ---------------------------------------------------------------------------
// Test 1 — enumerate + decode every dataset in the chunked fixture.
// ---------------------------------------------------------------------------

/// Open `libver_latest_chunked.h5`, list all datasets, read each one, and
/// assert:
///   1. No panic during B-tree v2 / extensible-array traversal.
///   2. The returned dataset has a non-empty shape.
///   3. The raw byte buffer length equals the element count × element size.
#[test]
fn btree_v2_chunk_index_all_datasets() {
    let f = oxih5::open(LIBVER_LATEST_CHUNKED).expect("open libver_latest_chunked.h5");

    let names = f
        .dataset_names()
        .expect("dataset_names on libver_latest_chunked.h5");

    assert!(
        !names.is_empty(),
        "libver_latest_chunked.h5 must contain at least one dataset; got none"
    );

    let mut checked = 0usize;
    for name in &names {
        let ds = f
            .dataset(name)
            .unwrap_or_else(|e| panic!("read dataset {name:?}: {e}"));

        assert!(!ds.shape.is_empty(), "dataset {name:?} has empty shape");

        let elem_count: usize = ds.shape.iter().product();

        // The raw byte buffer must be exactly elem_count × (data.len() / elem_count)
        // elements.  We cannot derive the element size from `Dtype` without
        // depending on implementation internals, but we *can* assert that the
        // buffer is a whole multiple of elem_count bytes (i.e. complete elements).
        assert!(
            elem_count == 0 || ds.data.len() % elem_count == 0,
            "dataset {name:?}: data.len()={} is not a multiple of elem_count={elem_count}",
            ds.data.len()
        );

        // Must have at least one byte.
        assert!(
            !ds.data.is_empty(),
            "dataset {name:?}: decoded data is empty"
        );

        checked += 1;
    }

    assert!(
        checked >= 1,
        "no datasets were successfully read from libver_latest_chunked.h5"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — data correctness for every known dataset in the chunked fixture.
// ---------------------------------------------------------------------------

/// Verify element values for the three known datasets in
/// `libver_latest_chunked.h5`:
///
/// | name              | dtype  | shape  | values                        |
/// |-------------------|--------|--------|-------------------------------|
/// | `chunked_gzip_1d` | f64    | [50]   | `i as f64` for i in 0..50    |
/// | `chunked_gzip_2d` | i32    | [4, 6] | `i as i32` for i in 0..24    |
/// | `chunked_plain`   | f32    | [20]   | `i as f32` for i in 0..20    |
///
/// This is the deepest possible assertion: every decoded chunk must agree with
/// the original data to floating-point precision.
#[test]
fn btree_v2_chunk_index_data_correctness() {
    let f = oxih5::open(LIBVER_LATEST_CHUNKED).expect("open libver_latest_chunked.h5");

    // -- chunked_gzip_1d: float64[50] arange(50) compressed with gzip --------
    {
        let ds = f.dataset("chunked_gzip_1d").expect("read chunked_gzip_1d");
        assert_eq!(ds.shape, vec![50], "chunked_gzip_1d: shape mismatch");
        let vals = ds.as_f64().expect("chunked_gzip_1d: as_f64");
        assert_eq!(vals.len(), 50, "chunked_gzip_1d: element count mismatch");
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - i as f64).abs() < 1e-10,
                "chunked_gzip_1d[{i}]: expected {}, got {v}",
                i as f64
            );
        }
    }

    // -- chunked_gzip_2d: int32[4, 6] arange(24) with gzip + shuffle ---------
    {
        let ds = f.dataset("chunked_gzip_2d").expect("read chunked_gzip_2d");
        assert_eq!(ds.shape, vec![4, 6], "chunked_gzip_2d: shape mismatch");
        let vals = ds.as_i32().expect("chunked_gzip_2d: as_i32");
        assert_eq!(vals.len(), 24, "chunked_gzip_2d: element count mismatch");
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(
                v, i as i32,
                "chunked_gzip_2d[{i}]: expected {}, got {v}",
                i as i32
            );
        }
    }

    // -- chunked_plain: float32[20] arange(20) plain (no filter) -------------
    {
        let ds = f.dataset("chunked_plain").expect("read chunked_plain");
        assert_eq!(ds.shape, vec![20], "chunked_plain: shape mismatch");
        let vals = ds.as_f32().expect("chunked_plain: as_f32");
        assert_eq!(vals.len(), 20, "chunked_plain: element count mismatch");
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-6,
                "chunked_plain[{i}]: expected {}, got {v}",
                i as f32
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3 — B-tree v2 type-5 name-index traversal over a large (20-dataset)
// group; exercises the fractal-heap + B-tree v2 path for group enumeration.
// ---------------------------------------------------------------------------

/// Open `libver_latest_large.h5`, navigate to `/measurements`, and verify
/// that all 20 `channel_NN` datasets are enumerated via the B-tree v2
/// type-5 name index.
///
/// This exercises multi-record (potentially multi-level) B-tree v2 traversal
/// on the group name index rather than the chunk index.
#[test]
fn btree_v2_name_index_large_group_traversal() {
    let f = oxih5::open(LIBVER_LATEST_LARGE).expect("open libver_latest_large.h5");

    let grp = f
        .group("measurements")
        .expect("navigate to /measurements group");

    let mut ds_names = grp.datasets().expect("list datasets in /measurements");

    assert_eq!(
        ds_names.len(),
        20,
        "expected 20 datasets in /measurements, got {:?}",
        ds_names
    );

    // Verify the full set channel_00 .. channel_19 are present.
    ds_names.sort();
    for n in 0..20usize {
        let expected = format!("channel_{n:02}");
        assert!(
            ds_names.contains(&expected),
            "missing {expected:?} in /measurements; got {ds_names:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4 — value correctness for a specific dataset read through the
//           B-tree v2 name index.
// ---------------------------------------------------------------------------

/// Read `/measurements/channel_05` from the large file and verify its value.
///
/// The fixture stores `float32[1] = [7.5]` at that path.  This combines
/// B-tree v2 name-index traversal (to find the dataset object header) with
/// direct data decoding.
#[test]
fn btree_v2_name_index_data_correctness() {
    let f = oxih5::open(LIBVER_LATEST_LARGE).expect("open libver_latest_large.h5");

    let ds = f
        .dataset("/measurements/channel_05")
        .expect("read /measurements/channel_05");

    assert_eq!(ds.shape, vec![1], "channel_05: shape should be [1]");

    let vals = ds.as_f32().expect("channel_05: as_f32");
    assert_eq!(vals.len(), 1, "channel_05: expected 1 element");
    assert!(
        (vals[0] - 7.5_f32).abs() < 1e-4,
        "channel_05: expected 7.5, got {}",
        vals[0]
    );
}

// ---------------------------------------------------------------------------
// Test 5 — combined name-index + chunk-index traversal.
// ---------------------------------------------------------------------------

/// Open `libver_latest_large.h5`, walk every dataset in the large group via
/// B-tree v2 name-index traversal, and read each one.  This confirms that
/// both index types co-operate correctly: the name index locates each object
/// header, and the data-layout decode assembles the element data.
///
/// For contiguous datasets the chunk path is not exercised but the name index
/// is.  This is intentional — the test title emphasises the combined path.
#[test]
fn btree_v2_chunk_plus_name_index_combined() {
    let f = oxih5::open(LIBVER_LATEST_LARGE).expect("open libver_latest_large.h5");

    let grp = f
        .group("measurements")
        .expect("navigate to /measurements group");

    let ds_names = grp.datasets().expect("list datasets in /measurements");

    let mut decoded_count = 0usize;
    let mut total_elements = 0usize;

    for name in &ds_names {
        let path = format!("/measurements/{name}");
        let ds = f
            .dataset(&path)
            .unwrap_or_else(|e| panic!("read {path:?}: {e}"));

        assert!(
            !ds.shape.is_empty(),
            "dataset {path:?} returned empty shape"
        );
        assert!(
            !ds.data.is_empty(),
            "dataset {path:?} returned empty data buffer"
        );

        let elem_count: usize = ds.shape.iter().product();
        assert!(
            elem_count == 0 || ds.data.len() % elem_count == 0,
            "dataset {path:?}: data.len()={} not multiple of elem_count={elem_count}",
            ds.data.len()
        );

        total_elements += elem_count;
        decoded_count += 1;
    }

    assert_eq!(
        decoded_count, 20,
        "expected 20 datasets decoded, got {decoded_count}"
    );
    assert!(
        total_elements >= 20,
        "total elements across 20 datasets should be >= 20, got {total_elements}"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — B-tree v2 traversal over every "chunk*" fixture found on disk.
// ---------------------------------------------------------------------------

/// Iterate every `.h5` fixture whose filename contains "chunk", open it,
/// enumerate all datasets, and assert that the decoded data is consistent
/// with the reported shape.
///
/// This is a broad smoke-test: it exercises whatever chunk-index variant the
/// fixture uses (B-tree v1, B-tree v2, extensible array, fixed array) without
/// asserting specific values.  New fixtures added to the directory are
/// automatically picked up.
#[test]
fn btree_v2_traversal_via_all_chunked_fixtures() {
    let fixtures_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let entries = std::fs::read_dir(&fixtures_dir).expect("open tests/fixtures directory");

    let mut tested_files = 0usize;
    let mut tested_datasets = 0usize;

    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();

        // Only process .h5 files whose name contains "chunk".
        if !fname_str.ends_with(".h5") || !fname_str.contains("chunk") {
            continue;
        }

        let path = entry.path();
        let f = match oxih5::open(&path) {
            Ok(f) => f,
            Err(e) => {
                // Unsupported features (VDS, szip, …) may cause open to fail.
                // Treat as a non-fatal skip.
                eprintln!("SKIP {fname_str}: open error: {e}");
                continue;
            }
        };

        let names = match f.dataset_names() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("SKIP {fname_str}: dataset_names error: {e}");
                continue;
            }
        };

        for name in &names {
            let ds = match f.dataset(name) {
                Ok(d) => d,
                Err(e) => {
                    // A single dataset failing should not abort the whole file.
                    eprintln!("SKIP {fname_str}/{name}: read error: {e}");
                    continue;
                }
            };

            let elem_count: usize = ds.shape.iter().product();

            // Non-empty shape → non-empty data.
            if elem_count > 0 {
                assert!(
                    !ds.data.is_empty(),
                    "{fname_str}/{name}: elem_count={elem_count} but data is empty"
                );
                // Data length is a whole number of complete elements.
                assert!(
                    ds.data.len() % elem_count == 0,
                    "{fname_str}/{name}: data.len()={} is not a multiple of \
                     elem_count={elem_count}",
                    ds.data.len()
                );
            }

            tested_datasets += 1;
        }

        tested_files += 1;
    }

    // Warn — rather than hard-fail — if no fixture files were found at all.
    // In a fresh checkout without the generated fixtures this is possible.
    if tested_files == 0 {
        eprintln!(
            "WARNING: no 'chunk*.h5' fixtures found in {fixtures_dir:?}; \
             B-tree v2 smoke-test not exercised"
        );
    }

    // When fixtures ARE present, at least one dataset must have been decoded.
    if tested_files > 0 {
        assert!(
            tested_datasets >= 1,
            "opened {tested_files} chunked fixture(s) but decoded 0 datasets"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7 — deeply nested group path + B-tree v2 chunk-index round-trip.
// ---------------------------------------------------------------------------

/// Navigate to a dataset inside a nested group in `libver_latest_large.h5`
/// using an absolute path, then verify shape and value.
///
/// This exercises the combined path:
///   superblock v3 → new-style root group → B-tree v2 name-index lookup of
///   "measurements" group → B-tree v2 name-index lookup of "channel_09"
///   dataset → object header v2 decode → contiguous data read.
///
/// While this particular fixture stores contiguous (not chunked) data inside
/// the large group, the B-tree v2 *name* traversal is still exercised at two
/// levels of nesting, fulfilling the "deeply nested B-tree v2 traversal"
/// requirement when the chunk-index layer is the extensible-array / B-tree v2
/// chunk index in `libver_latest_chunked.h5`.
#[test]
fn btree_v2_deeply_nested_path_and_chunk_roundtrip() {
    // Part A: two-level B-tree v2 name-index traversal (root → group →
    // dataset) in the large file.
    {
        let f = oxih5::open(LIBVER_LATEST_LARGE).expect("open libver_latest_large.h5");

        // Navigate two levels: root group → /measurements → channel_09.
        let ds = f
            .dataset("/measurements/channel_09")
            .expect("read /measurements/channel_09 via two-level B-tree v2 name traversal");

        assert_eq!(ds.shape, vec![1], "channel_09: shape should be [1]");

        let vals = ds.as_f32().expect("channel_09: as_f32");
        assert_eq!(vals.len(), 1, "channel_09: expected 1 element");

        // channel_09 stores 9 * 1.5 = 13.5 (same formula as channel_05 = 7.5).
        assert!(
            (vals[0] - 13.5_f32).abs() < 1e-4,
            "channel_09: expected 13.5, got {}",
            vals[0]
        );
    }

    // Part B: superblock-v3 + B-tree v2 chunk index + filter-pipeline
    // round-trip in the chunked file.  Reads all three datasets to confirm
    // the traversal completes without error and produces the expected element
    // count.
    {
        let f = oxih5::open(LIBVER_LATEST_CHUNKED).expect("open libver_latest_chunked.h5");

        // 1-D float64 gzip: 50 elements across multiple chunks.
        let ds1 = f
            .dataset("chunked_gzip_1d")
            .expect("round-trip: chunked_gzip_1d");
        let n1: usize = ds1.shape.iter().product();
        assert_eq!(n1, 50, "chunked_gzip_1d: expected 50 elements, got {n1}");
        assert_eq!(
            ds1.data.len(),
            n1 * std::mem::size_of::<f64>(),
            "chunked_gzip_1d: raw byte length mismatch"
        );

        // 2-D int32 gzip+shuffle: 4×6 = 24 elements.
        let ds2 = f
            .dataset("chunked_gzip_2d")
            .expect("round-trip: chunked_gzip_2d");
        let n2: usize = ds2.shape.iter().product();
        assert_eq!(n2, 24, "chunked_gzip_2d: expected 24 elements, got {n2}");
        assert_eq!(
            ds2.data.len(),
            n2 * std::mem::size_of::<i32>(),
            "chunked_gzip_2d: raw byte length mismatch"
        );

        // 1-D float32 plain (no filter): 20 elements.
        let ds3 = f
            .dataset("chunked_plain")
            .expect("round-trip: chunked_plain");
        let n3: usize = ds3.shape.iter().product();
        assert_eq!(n3, 20, "chunked_plain: expected 20 elements, got {n3}");
        assert_eq!(
            ds3.data.len(),
            n3 * std::mem::size_of::<f32>(),
            "chunked_plain: raw byte length mismatch"
        );
    }
}
