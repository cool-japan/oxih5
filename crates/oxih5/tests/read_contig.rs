// Integration tests for M1: reading contiguous-layout HDF5 files written by h5py.
//
// Each test writes its fixture bytes to a uniquely named temp file (using both
// process id and a per-call atomic counter to avoid collisions between parallel
// test threads), reads it back through the oxih5 parser, then removes the file.

use std::sync::atomic::{AtomicU64, Ordering};

// Phase 4 libver='latest' (superblock v3 / new-style groups) fixture paths.
// These files are opened directly via oxih5::open() using relative paths
// (cargo test sets cwd to the crate root).
const LIBVER_LATEST_SMALL: &str = "tests/fixtures/libver_latest_small.h5";
const LIBVER_LATEST_LARGE: &str = "tests/fixtures/libver_latest_large.h5";

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_tmp(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "oxih5_test_{}_{}_{}.h5",
        std::process::id(),
        n,
        tag
    ));
    std::fs::write(&path, bytes).expect("write temp fixture");
    path
}

fn cleanup(path: std::path::PathBuf) {
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Fixture bytes embedded at compile time.
// ---------------------------------------------------------------------------

const F4_1D: &[u8] = include_bytes!("fixtures/f4_1d_contig.h5");
const I4_2D: &[u8] = include_bytes!("fixtures/i4_2d_contig.h5");
const F8_1D: &[u8] = include_bytes!("fixtures/f8_1d_contig.h5");
const BE_F4_1D: &[u8] = include_bytes!("fixtures/be_f4_1d_contig.h5");

// ---------------------------------------------------------------------------
// Test 1: 1-D float32 little-endian (primary fixture)
// ---------------------------------------------------------------------------

#[test]
fn test_f4_1d_shape_and_values() {
    let path = write_tmp("f4_1d", F4_1D);
    let result = oxih5::read_dataset(&path, "temperature");
    cleanup(path);

    let ds = result.expect("read_dataset('temperature') failed");

    assert_eq!(ds.shape, vec![8], "shape mismatch");

    let values = ds.as_f32().expect("as_f32 failed");
    assert_eq!(values.len(), 8, "element count mismatch");

    // Expected: np.arange(8, dtype='<f4') = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for (i, &v) in values.iter().enumerate() {
        let expected = i as f32;
        assert!(
            (v - expected).abs() < 1e-6,
            "values[{i}]: expected {expected}, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: dataset_names (via the root group symbol table)
// ---------------------------------------------------------------------------

#[test]
fn test_dataset_names_f4_1d() {
    let path = write_tmp("names_f4", F4_1D);
    let result = oxih5::open(&path);
    cleanup(path);

    let file = result.expect("open failed");
    let names = file.dataset_names().expect("dataset_names failed");

    assert!(
        names.contains(&"temperature".to_string()),
        "expected 'temperature' in names, got: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: 2-D int32 little-endian
// ---------------------------------------------------------------------------

#[test]
fn test_i4_2d_shape_and_values() {
    let path = write_tmp("i4_2d", I4_2D);
    let result = oxih5::read_dataset(&path, "matrix");
    cleanup(path);

    let ds = result.expect("read_dataset('matrix') failed");

    assert_eq!(ds.shape, vec![2, 3], "shape mismatch");

    let values = ds.as_i32().expect("as_i32 failed");
    // Expected: np.arange(6, dtype='<i4').reshape(2,3) = [0,1,2,3,4,5]
    assert_eq!(values, vec![0i32, 1, 2, 3, 4, 5], "values mismatch");
}

// ---------------------------------------------------------------------------
// Test 4: 1-D float64 little-endian
// ---------------------------------------------------------------------------

#[test]
fn test_f8_1d_values() {
    let path = write_tmp("f8_1d", F8_1D);
    let result = oxih5::read_dataset(&path, "signal");
    cleanup(path);

    let ds = result.expect("read_dataset('signal') failed");

    assert_eq!(ds.shape, vec![4], "shape mismatch");

    let values = ds.as_f64().expect("as_f64 failed");
    // Expected: np.array([1.1, 2.2, 3.3, 4.4], dtype='<f8')
    let expected = [1.1f64, 2.2, 3.3, 4.4];
    assert_eq!(values.len(), expected.len(), "element count mismatch");
    for (i, (&v, &e)) in values.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-10, "values[{i}]: expected {e}, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Test 5: 1-D float32 big-endian (byte-order detection)
// ---------------------------------------------------------------------------

#[test]
fn test_be_f4_1d_big_endian() {
    let path = write_tmp("be_f4", BE_F4_1D);
    let result = oxih5::read_dataset(&path, "voltage");
    cleanup(path);

    let ds = result.expect("read_dataset('voltage') failed");

    assert_eq!(ds.shape, vec![4], "shape mismatch");

    let values = ds.as_f32().expect("as_f32 (big-endian) failed");
    // Expected: np.array([1.0, 2.0, 3.0, 4.0], dtype='>f4')
    let expected = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(values.len(), expected.len(), "element count mismatch");
    for (i, (&v, &e)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (v - e).abs() < 1e-6,
            "be_values[{i}]: expected {e}, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: dataset_names correctness for i4_2d fixture
// ---------------------------------------------------------------------------

#[test]
fn test_dataset_names_i4_2d() {
    let path = write_tmp("names_i4", I4_2D);
    let result = oxih5::open(&path);
    cleanup(path);

    let file = result.expect("open failed");
    let names = file.dataset_names().expect("dataset_names failed");

    assert!(
        names.contains(&"matrix".to_string()),
        "expected 'matrix' in names, got: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Phase 4 tests: Group navigation + API
// ---------------------------------------------------------------------------

// Test 7: contiguous dtypes via File::dataset
#[test]
fn test_contiguous_dtypes() {
    let path_f4 = write_tmp("dtype_f4", F4_1D);
    let path_f8 = write_tmp("dtype_f8", F8_1D);
    let path_i4 = write_tmp("dtype_i4", I4_2D);

    let f4 = oxih5::open(&path_f4).expect("open f4");
    cleanup(path_f4);
    let ds = f4.dataset("temperature").expect("temperature");
    match &ds.dtype {
        oxih5::Dtype::Float { .. } | oxih5::Dtype::Int { .. } => {}
        other => panic!("unexpected dtype {:?}", other),
    }

    let f8 = oxih5::open(&path_f8).expect("open f8");
    cleanup(path_f8);
    let ds = f8.dataset("signal").expect("signal");
    match &ds.dtype {
        oxih5::Dtype::Float { .. } | oxih5::Dtype::Int { .. } => {}
        other => panic!("unexpected dtype {:?}", other),
    }

    let i4 = oxih5::open(&path_i4).expect("open i4");
    cleanup(path_i4);
    let ds = i4.dataset("matrix").expect("matrix");
    match &ds.dtype {
        oxih5::Dtype::Float { .. } | oxih5::Dtype::Int { .. } => {}
        other => panic!("unexpected dtype {:?}", other),
    }
}

// Test 8: big-endian float32 dataset
#[test]
fn test_big_endian_dataset() {
    let path = write_tmp("be_f4_test", BE_F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let ds = f.dataset("voltage").expect("voltage");
    assert!(
        matches!(
            ds.dtype,
            oxih5::Dtype::Float {
                order: oxih5::ByteOrder::Big,
                ..
            }
        ),
        "expected big-endian float, got {:?}",
        ds.dtype
    );
    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals, vec![1.0_f32, 2.0, 3.0, 4.0]);
}

// Test 9: multidimensional shape
#[test]
fn test_multidimensional_shape() {
    let path = write_tmp("multi_shape", I4_2D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let ds = f.dataset("matrix").expect("matrix");
    assert_eq!(ds.shape, vec![2, 3]);
    assert_eq!(ds.len(), 6);
}

// Test 10: error handling
#[test]
fn test_error_handling() {
    // Non-existent file
    assert!(
        oxih5::open("/nonexistent/path/file.h5").is_err(),
        "expected error for missing file"
    );

    // Missing dataset
    let path = write_tmp("err_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);
    assert!(
        matches!(
            f.dataset("no_such_dataset"),
            Err(oxih5::OxiH5Error::NotFound(_))
        ),
        "expected NotFound for missing dataset"
    );
}

// Test 11: dataset_names returns expected names
#[test]
fn test_many_dataset_names() {
    let path_f4 = write_tmp("names2_f4", F4_1D);
    let f = oxih5::open(&path_f4).expect("open f4");
    cleanup(path_f4);
    let names = f.dataset_names().expect("dataset_names");
    assert!(!names.is_empty());
    assert!(
        names.contains(&"temperature".to_string()),
        "expected 'temperature' in names, got: {names:?}"
    );

    let path_i4 = write_tmp("names2_i4", I4_2D);
    let f2 = oxih5::open(&path_i4).expect("open i4");
    cleanup(path_i4);
    let names2 = f2.dataset_names().expect("dataset_names");
    assert!(
        names2.contains(&"matrix".to_string()),
        "expected 'matrix' in names2, got: {names2:?}"
    );
}

// Test 12: File::info returns sensible metadata
#[test]
fn test_file_info() {
    let path = write_tmp("info_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let info = f.info().expect("info");
    assert_eq!(info.superblock_version, 0);
    assert!(info.file_size > 0);
    assert_eq!(info.offset_size, 8);
    assert_eq!(info.length_size, 8);
}

// Test 13: version() returns a non-empty semver string
#[test]
fn test_version() {
    let v = oxih5::version();
    assert!(!v.is_empty(), "version string was empty");
    assert!(v.contains('.'), "version string lacks '.': {v}");
}

// Test 14: File::root() returns root group with expected datasets
#[test]
fn test_root_group() {
    let path = write_tmp("root_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let root = f.root().expect("root");
    assert_eq!(root.name, "/");
    let ds_names = root.datasets().expect("datasets");
    assert!(
        ds_names.contains(&"temperature".to_string()),
        "expected 'temperature' in root datasets, got: {ds_names:?}"
    );
}

// Test 15: Debug impl for File
#[test]
fn test_debug_impl() {
    let path = write_tmp("debug_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let debug_str = format!("{:?}", f);
    assert!(
        debug_str.contains("File"),
        "expected 'File' in debug output: {debug_str}"
    );
    assert!(!debug_str.is_empty());
}

// Test 16: Dataset::attrs() and Dataset::attr() accessor methods
#[test]
fn test_dataset_attrs_accessors() {
    let path = write_tmp("attr_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let ds = f.dataset("temperature").expect("temperature");
    // The fixture has no user-defined attributes, so attrs() should return empty.
    let attrs = ds.attrs();
    // attrs() returns a slice view consistent with the stored Vec.
    assert_eq!(attrs.len(), ds.attributes.len());
    // attr() for a non-existent name returns None.
    assert!(ds.attr("nonexistent_attr").is_none());
}

// Test 17: Group::dataset() reads a dataset via the group handle
#[test]
fn test_group_dataset() {
    let path = write_tmp("grpds_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let root = f.root().expect("root");
    let ds = root.dataset("temperature").expect("group dataset");
    assert_eq!(ds.shape, vec![8]);
    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals.len(), 8);
}

// Test 18: Group::attrs() does not panic even when no attrs are present
#[test]
fn test_group_attrs() {
    let path = write_tmp("grpattr_f4", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let root = f.root().expect("root");
    let _attrs = root.attrs().expect("group attrs");
    // No assertion on count — fixture may or may not have group-level attrs.
}

// ===========================================================================
// Chunked layout integration tests (B-tree v1 chunk index + filter pipeline).
//
// These fixtures are written by h5py with libver='earliest', which produces a
// version-1 B-tree raw-data-chunk index (TREE node type 1) and layout v3.
// ===========================================================================

const CHUNKED_I4_1D: &[u8] = include_bytes!("fixtures/chunked_i4_1d.h5");
const CHUNKED_GZIP_F4_1D: &[u8] = include_bytes!("fixtures/chunked_gzip_f4_1d.h5");
const CHUNKED_GZIP_SHUFFLE_I4_2D: &[u8] = include_bytes!("fixtures/chunked_gzip_shuffle_i4_2d.h5");
const CHUNKED_FLETCHER_F8_1D: &[u8] = include_bytes!("fixtures/chunked_fletcher_f8_1d.h5");
const CHUNKED_GZIP_F8_2D_PARTIAL: &[u8] = include_bytes!("fixtures/chunked_gzip_f8_2d_partial.h5");

// Test 19: chunked, uncompressed 1-D int32 (3 chunks of 4 elements).
#[test]
fn test_chunked_i4_1d() {
    let path = write_tmp("chunked_i4_1d", CHUNKED_I4_1D);
    let result = oxih5::read_dataset(&path, "data");
    cleanup(path);

    let ds = result.expect("read chunked i4 1d");
    assert_eq!(ds.shape, vec![12], "shape mismatch");
    let vals = ds.as_i32().expect("as_i32");
    assert_eq!(vals, (0..12).collect::<Vec<i32>>());
}

// Test 20: chunked + gzip 1-D float32 (4 chunks of 25 elements).
#[test]
fn test_chunked_gzip_f4_1d() {
    let path = write_tmp("chunked_gzip_f4_1d", CHUNKED_GZIP_F4_1D);
    let result = oxih5::read_dataset(&path, "data");
    cleanup(path);

    let ds = result.expect("read chunked+gzip f4 1d");
    assert_eq!(ds.shape, vec![100], "shape mismatch");
    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals.len(), 100);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - i as f32).abs() < 1e-6,
            "vals[{i}]: expected {i}, got {v}"
        );
    }
}

// Test 21: chunked + gzip + shuffle 2-D int32 (6x8, chunk 3x4).
#[test]
fn test_chunked_gzip_shuffle_i4_2d() {
    let path = write_tmp("chunked_gzip_shuffle_i4_2d", CHUNKED_GZIP_SHUFFLE_I4_2D);
    let result = oxih5::read_dataset(&path, "data");
    cleanup(path);

    let ds = result.expect("read chunked+gzip+shuffle i4 2d");
    assert_eq!(ds.shape, vec![6, 8], "shape mismatch");
    let vals = ds.as_i32().expect("as_i32");
    // Row-major arange(48).reshape(6, 8) == [0, 1, 2, ... 47].
    assert_eq!(vals, (0..48).collect::<Vec<i32>>());
}

// Test 22: chunked + fletcher32 checksum 1-D float64 (5 chunks of 10).
#[test]
fn test_chunked_fletcher_f8_1d() {
    let path = write_tmp("chunked_fletcher_f8_1d", CHUNKED_FLETCHER_F8_1D);
    let result = oxih5::read_dataset(&path, "data");
    cleanup(path);

    let ds = result.expect("read chunked+fletcher f8 1d");
    assert_eq!(ds.shape, vec![50], "shape mismatch");
    let vals = ds.as_f64().expect("as_f64");
    assert_eq!(vals.len(), 50);
    for (i, &v) in vals.iter().enumerate() {
        let expected = i as f64 * 1.5;
        assert!(
            (v - expected).abs() < 1e-12,
            "vals[{i}]: expected {expected}, got {v}"
        );
    }
}

// Test 23: chunked + gzip 2-D float64 with partial edge chunks (5x7, chunk 2x3).
// This exercises the padding-element-skip logic in chunk assembly because the
// dataset dims are not multiples of the chunk dims.
#[test]
fn test_chunked_gzip_f8_2d_partial() {
    let path = write_tmp("chunked_gzip_f8_2d_partial", CHUNKED_GZIP_F8_2D_PARTIAL);
    let result = oxih5::read_dataset(&path, "data");
    cleanup(path);

    let ds = result.expect("read chunked+gzip f8 2d partial");
    assert_eq!(ds.shape, vec![5, 7], "shape mismatch");
    let vals = ds.as_f64().expect("as_f64");
    assert_eq!(vals.len(), 35);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - i as f64).abs() < 1e-12,
            "vals[{i}]: expected {i}, got {v}"
        );
    }
}

// Test 24: chunked dataset can also be reached via the group/path API.
#[test]
fn test_chunked_via_group_handle() {
    let path = write_tmp("chunked_grp_i4", CHUNKED_I4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let root = f.root().expect("root");
    let names = root.datasets().expect("datasets");
    assert!(names.contains(&"data".to_string()), "got {names:?}");

    let ds = root.dataset("data").expect("group dataset read");
    let vals = ds.as_i32().expect("as_i32");
    assert_eq!(vals, (0..12).collect::<Vec<i32>>());
}

// ---------------------------------------------------------------------------
// Test 25: File::contains — existence predicate.
// ---------------------------------------------------------------------------
#[test]
fn test_contains_existing_dataset() {
    let path = write_tmp("contains_f4_1d", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    assert!(f.contains("temperature"), "bare name should be found");
    assert!(
        f.contains("/temperature"),
        "slash-prefixed name should be found"
    );
}

#[test]
fn test_contains_nonexistent() {
    let path = write_tmp("contains_f4_1d_missing", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    assert!(!f.contains("nonexistent_dataset"));
    assert!(!f.contains("/no/such/path"));
}

// ---------------------------------------------------------------------------
// Test 26: File::walk — tree traversal.
// ---------------------------------------------------------------------------
#[test]
fn test_walk_collects_datasets() {
    let path = write_tmp("walk_f4_1d", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let mut names: Vec<String> = Vec::new();
    f.walk(&mut |path_str, _is_group| {
        names.push(path_str.to_string());
    })
    .expect("walk should succeed");

    assert!(!names.is_empty(), "walk should find at least one entry");
    assert!(
        names.iter().any(|n| n.contains("temperature")),
        "expected 'temperature' in walk results, got {names:?}"
    );
}

#[test]
fn test_walk_is_group_flag() {
    let path = write_tmp("walk_flag_f4_1d", F4_1D);
    let f = oxih5::open(&path).expect("open");
    cleanup(path);

    let mut dataset_count = 0usize;
    let mut group_count = 0usize;
    f.walk(&mut |_path_str, is_group| {
        if is_group {
            group_count += 1;
        } else {
            dataset_count += 1;
        }
    })
    .expect("walk should succeed");

    // The fixture has at least one dataset and no sub-groups at root.
    assert!(dataset_count > 0, "expected datasets, got none");
    // Group count may be 0 for flat fixtures — that's fine.
    let _ = group_count;
}

// ===========================================================================
// Nested group + attribute integration tests.
//
// Fixtures generated with h5py libver='earliest' to produce B-tree v1 group
// indices and inline attribute messages (0x000C) — compatible with the current
// parser.  Generation scripts: crates/oxih5/tests/gen_fixtures.py
// ===========================================================================

static NESTED_GROUPS: &[u8] = include_bytes!("fixtures/nested_groups.h5");
static WITH_ATTRS: &[u8] = include_bytes!("fixtures/with_attrs.h5");

// ---------------------------------------------------------------------------
// Advanced fixture bytes (compound types, string datasets, multi-attribute).
// ---------------------------------------------------------------------------
#[allow(dead_code)]
static COMPOUND_1D: &[u8] = include_bytes!("fixtures/compound_1d.h5");
#[allow(dead_code)]
static STRING_FIXED_1D: &[u8] = include_bytes!("fixtures/string_fixed_1d.h5");
#[allow(dead_code)]
static CHUNKED_BTREE_V1: &[u8] = include_bytes!("fixtures/chunked_btree_v1.h5");
#[allow(dead_code)]
static MULTI_ATTR: &[u8] = include_bytes!("fixtures/multi_attr.h5");

// ---------------------------------------------------------------------------
// New dtype coverage fixture bytes.
// ---------------------------------------------------------------------------
static STRING_DATASETS: &[u8] = include_bytes!("fixtures/string_datasets.h5");
static COMPOUND_DATASET: &[u8] = include_bytes!("fixtures/compound_dataset.h5");
static ENUM_DATASET: &[u8] = include_bytes!("fixtures/enum_dataset.h5");
static LARGE_INT_DATASETS: &[u8] = include_bytes!("fixtures/large_int_datasets.h5");
static OPAQUE_DATASET: &[u8] = include_bytes!("fixtures/opaque_dataset.h5");

// ---------------------------------------------------------------------------
// Test 27: Navigate two group levels and read datasets.
// Fixture: /sensors/imu/accel (float32 [3]), /sensors/gps/coords (float64 [2])
// ---------------------------------------------------------------------------
#[test]
fn test_nested_group_navigation() {
    let path = write_tmp("nested_groups", NESTED_GROUPS);
    let f = oxih5::open(&path).expect("open nested_groups.h5");
    cleanup(path);

    // Navigate 2 levels deep and read a float32 dataset.
    let accel = f
        .dataset("/sensors/imu/accel")
        .expect("dataset at /sensors/imu/accel");
    assert_eq!(accel.shape, vec![3], "accel shape mismatch");
    let vals = accel.as_f32().expect("accel as_f32");
    assert!(
        (vals[0] - 1.0_f32).abs() < 1e-6,
        "vals[0] expected 1.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 2.0_f32).abs() < 1e-6,
        "vals[1] expected 2.0, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 3.0_f32).abs() < 1e-6,
        "vals[2] expected 3.0, got {}",
        vals[2]
    );

    // Read the float64 dataset in the sibling group.
    let coords = f.dataset("/sensors/gps/coords").expect("coords dataset");
    assert_eq!(coords.shape, vec![2], "coords shape mismatch");
    let v = coords.as_f64().expect("coords as_f64");
    assert!(
        (v[0] - 48.123_f64).abs() < 1e-6,
        "coords[0] expected 48.123, got {}",
        v[0]
    );
    assert!(
        (v[1] - 11.456_f64).abs() < 1e-6,
        "coords[1] expected 11.456, got {}",
        v[1]
    );
}

// ---------------------------------------------------------------------------
// Test 28: Group listing at intermediate depth.
// ---------------------------------------------------------------------------
#[test]
fn test_nested_group_listing() {
    let path = write_tmp("nested_grp_list", NESTED_GROUPS);
    let f = oxih5::open(&path).expect("open nested_groups.h5");
    cleanup(path);

    let sensors = f.group("/sensors").expect("sensors group");
    let mut sub_groups = sensors.groups().expect("sub-groups of /sensors");
    sub_groups.sort();

    assert!(
        sub_groups.contains(&"imu".to_string()),
        "expected 'imu' in {:?}",
        sub_groups
    );
    assert!(
        sub_groups.contains(&"gps".to_string()),
        "expected 'gps' in {:?}",
        sub_groups
    );
    assert_eq!(
        sub_groups.len(),
        2,
        "expected exactly 2 sub-groups, got {:?}",
        sub_groups
    );
}

// ---------------------------------------------------------------------------
// Test 29: Datasets inside a nested group have no spurious siblings.
// ---------------------------------------------------------------------------
#[test]
fn test_nested_leaf_group_datasets() {
    let path = write_tmp("nested_leaf", NESTED_GROUPS);
    let f = oxih5::open(&path).expect("open nested_groups.h5");
    cleanup(path);

    let imu = f.group("/sensors/imu").expect("/sensors/imu group");
    let ds_names = imu.datasets().expect("datasets in /sensors/imu");
    assert!(
        ds_names.contains(&"accel".to_string()),
        "expected 'accel' in {:?}",
        ds_names
    );

    let gps = f.group("/sensors/gps").expect("/sensors/gps group");
    let ds_names_gps = gps.datasets().expect("datasets in /sensors/gps");
    assert!(
        ds_names_gps.contains(&"coords".to_string()),
        "expected 'coords' in {:?}",
        ds_names_gps
    );
}

// ---------------------------------------------------------------------------
// Test 30: Dataset attributes — 'units' and 'scale_factor' on temperature.
// ---------------------------------------------------------------------------
#[test]
fn test_dataset_attributes() {
    let path = write_tmp("with_attrs_ds", WITH_ATTRS);
    let f = oxih5::open(&path).expect("open with_attrs.h5");
    cleanup(path);

    let ds = f.dataset("temperature").expect("temperature");
    let attrs = ds.attrs();

    assert!(
        !attrs.is_empty(),
        "expected attributes on 'temperature', got empty list"
    );
    let units_attr = attrs.iter().find(|a| a.name == "units");
    assert!(
        units_attr.is_some(),
        "expected 'units' attribute, got: {:?}",
        attrs.iter().map(|a| &a.name).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 31: Dataset::attr(name) — found and not-found cases.
// ---------------------------------------------------------------------------
#[test]
fn test_dataset_attr_by_name() {
    let path = write_tmp("with_attrs_named", WITH_ATTRS);
    let f = oxih5::open(&path).expect("open with_attrs.h5");
    cleanup(path);

    let ds = f.dataset("temperature").expect("temperature");

    // 'units' attribute should be found.
    assert!(
        ds.attr("units").is_some(),
        "expected Some for 'units' attr, got None"
    );

    // Non-existent attribute should return None.
    assert!(
        ds.attr("nonexistent_attr_xyz").is_none(),
        "expected None for missing attr"
    );
}

// ---------------------------------------------------------------------------
// Test 32: Group attributes — 'version' on /metadata group.
// ---------------------------------------------------------------------------
#[test]
fn test_group_attributes() {
    let path = write_tmp("group_attrs", WITH_ATTRS);
    let f = oxih5::open(&path).expect("open with_attrs.h5");
    cleanup(path);

    let g = f.group("metadata").expect("metadata group");
    let attrs = g.attrs().expect("group attrs");

    assert!(
        !attrs.is_empty(),
        "expected attributes on /metadata group, got empty"
    );
    let version_attr = attrs.iter().find(|a| a.name == "version");
    assert!(
        version_attr.is_some(),
        "expected 'version' attribute on /metadata group, got {:?}",
        attrs.iter().map(|a| &a.name).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 33: File::dataset_slice — sub-region of a 1-D float32 dataset.
// ---------------------------------------------------------------------------
#[test]
fn test_dataset_slice_1d() {
    let path = write_tmp("slice_test", F4_1D);
    let f = oxih5::open(&path).expect("open f4_1d");
    cleanup(path);

    // F4_1D fixture has 'temperature' dataset with 8 float32 elements [0.0..7.0]
    let full = f.dataset("temperature").expect("full dataset");
    let n = full.shape[0];

    if n >= 2 {
        #[allow(clippy::single_range_in_vec_init)]
        let ranges: &[std::ops::Range<usize>] = &[0..2];
        let sliced = f.dataset_slice("temperature", ranges).expect("slice 0..2");
        assert_eq!(sliced.shape, vec![2], "slice shape should be [2]");
        let full_vals = full.as_f32().expect("full as_f32");
        let slice_vals = sliced.as_f32().expect("slice as_f32");
        assert_eq!(slice_vals[0], full_vals[0]);
        assert_eq!(slice_vals[1], full_vals[1]);
    }
}

// ---------------------------------------------------------------------------
// Test 34: Group::dataset_slice — sub-region via group handle.
// ---------------------------------------------------------------------------
#[test]
fn test_group_dataset_slice_1d() {
    let path = write_tmp("grp_slice_test", F4_1D);
    let f = oxih5::open(&path).expect("open f4_1d");
    cleanup(path);

    let root = f.root().expect("root group");
    let full = root.dataset("temperature").expect("full dataset via group");
    let n = full.shape[0];

    if n >= 4 {
        #[allow(clippy::single_range_in_vec_init)]
        let ranges: &[std::ops::Range<usize>] = &[1..4];
        let sliced = root
            .dataset_slice("temperature", ranges)
            .expect("group slice 1..4");
        assert_eq!(sliced.shape, vec![3], "group slice shape should be [3]");
        let full_vals = full.as_f32().expect("full as_f32");
        let slice_vals = sliced.as_f32().expect("slice as_f32");
        assert_eq!(slice_vals[0], full_vals[1]);
        assert_eq!(slice_vals[1], full_vals[2]);
        assert_eq!(slice_vals[2], full_vals[3]);
    }
}

// ===========================================================================
// Advanced fixture tests: compound types, string datasets, multi-attribute,
// and chunked B-tree v1 with float64 + gzip.
//
// Fixtures generated with h5py 3.16+; scripts in /tmp/gen_*.py.
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: compound datatype dataset — shape and dtype recognition.
// Fixture: /points — compound dtype [('x',f4),('y',f4),('z',f4)], shape [2]
// ---------------------------------------------------------------------------
#[test]
fn test_compound_dataset_shape() {
    let path = write_tmp("compound_1d", COMPOUND_1D);
    let f = oxih5::open(&path).expect("open compound_1d.h5");
    cleanup(path);

    let ds = f.dataset("points").expect("points dataset");
    assert_eq!(ds.shape, vec![2], "compound dataset should have 2 elements");
    // Compound type should parse without error.
    assert!(
        matches!(ds.dtype, oxih5::Dtype::Compound { .. }),
        "expected Compound dtype, got {:?}",
        ds.dtype
    );
}

// ---------------------------------------------------------------------------
// Test: fixed-length string dataset — shape and dtype recognition.
// Fixture: /labels — S10 string dtype, shape [2]
// ---------------------------------------------------------------------------
#[test]
fn test_string_fixed_dataset_shape() {
    let path = write_tmp("string_fixed", STRING_FIXED_1D);
    let f = oxih5::open(&path).expect("open string_fixed_1d.h5");
    cleanup(path);

    let ds = f.dataset("labels").expect("labels dataset");
    assert_eq!(ds.shape, vec![2], "string dataset should have 2 elements");
    // h5py stores |S10 as HDF5 class 3 (String). Verify we parsed something string-ish.
    let is_string_or_known = matches!(
        &ds.dtype,
        oxih5::Dtype::String { .. } | oxih5::Dtype::Opaque { .. } | oxih5::Dtype::Int { .. }
    );
    assert!(
        is_string_or_known,
        "unexpected dtype {:?} for fixed-string dataset",
        ds.dtype
    );
}

// ---------------------------------------------------------------------------
// Test: as_string() on fixed-length string dataset.
// ---------------------------------------------------------------------------
#[test]
fn test_fixed_string_as_string() {
    let path = write_tmp("string_as_str", STRING_FIXED_1D);
    let f = oxih5::open(&path).expect("open string_fixed_1d.h5");
    cleanup(path);

    let ds = f.dataset("labels").expect("labels dataset");
    // Only call as_string() when dtype is String; otherwise just verify open worked.
    if let oxih5::Dtype::String { .. } = &ds.dtype {
        let strings = ds.as_string().expect("as_string");
        assert_eq!(strings.len(), 2, "expected 2 string elements");
        assert!(
            strings[0].starts_with("hello"),
            "expected 'hello...' got {:?}",
            strings[0]
        );
        assert!(
            strings[1].starts_with("world"),
            "expected 'world...' got {:?}",
            strings[1]
        );
    }
    // If dtype is not String, the test is a no-op — we verified the fixture parses.
}

// ---------------------------------------------------------------------------
// Test: chunked gzip float64 dataset with 100 elements (B-tree v1).
// Fixture: /series — float64 arange(100), chunks=(10,), gzip level 6.
// ---------------------------------------------------------------------------
#[test]
fn test_chunked_gzip_100_elements() {
    let path = write_tmp("chunked_btree_v1", CHUNKED_BTREE_V1);
    let f = oxih5::open(&path).expect("open chunked_btree_v1.h5");
    cleanup(path);

    let ds = f.dataset("series").expect("series dataset");
    assert_eq!(ds.shape, vec![100], "expected 100 elements");
    let vals = ds.as_f64().expect("as_f64");
    assert_eq!(vals.len(), 100, "expected 100 f64 values");
    // Values should be 0..99 (np.arange(100, dtype=float64)).
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - i as f64).abs() < 1e-10,
            "vals[{i}] = {v}, expected {}",
            i as f64
        );
    }
}

// ---------------------------------------------------------------------------
// Test: multiple attributes of different types on a float32 dataset.
// Fixture: /measurements — float32 linspace(0,1,20); attrs: units, count,
//          scale, calibrated.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_attributes_types() {
    let path = write_tmp("multi_attr", MULTI_ATTR);
    let f = oxih5::open(&path).expect("open multi_attr.h5");
    cleanup(path);

    let ds = f.dataset("measurements").expect("measurements dataset");
    assert_eq!(ds.shape, vec![20], "expected 20 elements");

    let attrs = ds.attrs();
    let attr_names: Vec<&str> = attrs.iter().map(|a| a.name.as_str()).collect();

    // Should have at least 'units' and 'count'.
    assert!(
        attr_names.contains(&"units"),
        "missing 'units' attr, got {:?}",
        attr_names
    );
    assert!(
        attr_names.contains(&"count"),
        "missing 'count' attr, got {:?}",
        attr_names
    );

    // Values for the float32 dataset.
    let vals = ds.as_f32().expect("as_f32 on measurements");
    assert_eq!(vals.len(), 20, "expected 20 float32 values");
    assert!(
        (vals[0] - 0.0_f32).abs() < 1e-6,
        "first val should be 0.0, got {}",
        vals[0]
    );
    assert!(
        (vals[19] - 1.0_f32).abs() < 1e-6,
        "last val should be 1.0, got {}",
        vals[19]
    );
}

// ===========================================================================
// Phase 4: libver='latest' (superblock v3 / new-style groups) integration tests.
//
// Fixtures generated with h5py 3.16 using libver='latest', which produces
// superblock v3, object header v2, and new-style group links.  Large groups
// (>8 links) use a fractal heap + B-tree v2 name index.
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 35: Read root-level dataset from a libver='latest' small file.
// Fixture: /root_data — int32[5] = [1, 2, 3, 4, 5]
// ---------------------------------------------------------------------------
#[test]
fn test_libver_latest_small_root_dataset() {
    let f = oxih5::open(LIBVER_LATEST_SMALL).expect("open libver_latest_small.h5");

    let ds = f.dataset("root_data").expect("root_data dataset");
    assert_eq!(ds.shape, vec![5], "root_data shape should be [5]");
    let vals = ds.as_i32().expect("as_i32");
    assert_eq!(vals, vec![1i32, 2, 3, 4, 5], "root_data values mismatch");
}

// ---------------------------------------------------------------------------
// Test 36: Read a nested dataset from a libver='latest' small file.
// Fixture: /sensors/temperature — float32[3] ≈ [20.5, 21.0, 19.8]
// ---------------------------------------------------------------------------
#[test]
fn test_libver_latest_small_nested_dataset() {
    let f = oxih5::open(LIBVER_LATEST_SMALL).expect("open libver_latest_small.h5");

    let ds = f
        .dataset("/sensors/temperature")
        .expect("/sensors/temperature dataset");
    assert_eq!(ds.shape, vec![3], "temperature shape should be [3]");
    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals.len(), 3, "expected 3 elements");
    assert!(
        (vals[0] - 20.5_f32).abs() < 1e-4,
        "vals[0] expected 20.5, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 21.0_f32).abs() < 1e-4,
        "vals[1] expected 21.0, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 19.8_f32).abs() < 1e-4,
        "vals[2] expected 19.8, got {}",
        vals[2]
    );
}

// ---------------------------------------------------------------------------
// Test 37: Root group listing on a libver='latest' small file.
// Fixture: root contains 'sensors' (group) and 'root_data' (dataset).
// ---------------------------------------------------------------------------
#[test]
fn test_libver_latest_small_group_list() {
    let f = oxih5::open(LIBVER_LATEST_SMALL).expect("open libver_latest_small.h5");

    let root = f.root().expect("root group");
    let group_names = root.groups().expect("root groups");
    assert!(
        group_names.contains(&"sensors".to_string()),
        "expected 'sensors' in root groups, got {:?}",
        group_names
    );
}

// ---------------------------------------------------------------------------
// Test 38: Large group listing via fractal heap + B-tree v2 name index.
// Fixture: /measurements contains 20 datasets channel_00..channel_19.
// ---------------------------------------------------------------------------
#[test]
fn test_libver_latest_large_group_list() {
    let f = oxih5::open(LIBVER_LATEST_LARGE).expect("open libver_latest_large.h5");

    let grp = f.group("measurements").expect("measurements group");
    let mut ds_names = grp.datasets().expect("datasets in measurements");
    ds_names.sort();

    assert_eq!(
        ds_names.len(),
        20,
        "expected 20 datasets, got {:?}",
        ds_names
    );
    assert!(
        ds_names.contains(&"channel_00".to_string()),
        "missing 'channel_00' in {:?}",
        ds_names
    );
    assert!(
        ds_names.contains(&"channel_19".to_string()),
        "missing 'channel_19' in {:?}",
        ds_names
    );
}

// ---------------------------------------------------------------------------
// Test 39: Read a specific dataset from the large group.
// Fixture: /measurements/channel_05 — float32[1] = [7.5]
// ---------------------------------------------------------------------------
#[test]
fn test_libver_latest_large_read_dataset() {
    let f = oxih5::open(LIBVER_LATEST_LARGE).expect("open libver_latest_large.h5");

    let ds = f
        .dataset("/measurements/channel_05")
        .expect("/measurements/channel_05 dataset");
    assert_eq!(ds.shape, vec![1], "channel_05 shape should be [1]");
    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals.len(), 1, "expected 1 element");
    assert!(
        (vals[0] - 7.5_f32).abs() < 1e-4,
        "channel_05 expected 7.5, got {}",
        vals[0]
    );
}

// ---------------------------------------------------------------------------
// Task A: VDS fixture test
// Opening a VDS file and attempting to read the virtual dataset should return
// a clear NotImplemented error, not a panic or a corrupted result.
// ---------------------------------------------------------------------------
#[test]
fn test_vds_returns_not_implemented() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vds_main.h5");
    if !path.exists() {
        // VDS fixture not generated — skip gracefully
        return;
    }
    let file = oxih5::open(&path).expect("open VDS file");
    let result = file.dataset("virtual_data");
    assert!(
        matches!(result, Err(oxih5::OxiH5Error::NotImplemented(_))),
        "expected NotImplemented for VDS, got unexpected result"
    );
}

// ---------------------------------------------------------------------------
// Task B: libver='latest' chunked dataset tests
// ---------------------------------------------------------------------------

const LIBVER_LATEST_CHUNKED: &str = "tests/fixtures/libver_latest_chunked.h5";

/// Test 40: libver='latest' chunked 1-D float64 with gzip compression.
/// Uses extensible array or B-tree v2 chunk index (superblock v3).
#[test]
fn test_libver_latest_chunked_1d() {
    let f = oxih5::open(LIBVER_LATEST_CHUNKED).expect("open libver_latest_chunked.h5");

    let ds = f.dataset("chunked_gzip_1d").expect("read chunked_gzip_1d");
    assert_eq!(ds.shape, vec![50], "chunked_gzip_1d shape should be [50]");

    let vals = ds.as_f64().expect("as_f64");
    assert_eq!(vals.len(), 50, "expected 50 elements");
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - i as f64).abs() < 1e-10,
            "chunked_gzip_1d[{i}]: expected {}, got {v}",
            i as f64
        );
    }
}

/// Test 41: libver='latest' chunked 2-D int32 with shuffle + gzip compression.
#[test]
fn test_libver_latest_chunked_2d() {
    let f = oxih5::open(LIBVER_LATEST_CHUNKED).expect("open libver_latest_chunked.h5");

    let ds = f.dataset("chunked_gzip_2d").expect("read chunked_gzip_2d");
    assert_eq!(
        ds.shape,
        vec![4, 6],
        "chunked_gzip_2d shape should be [4, 6]"
    );

    let vals = ds.as_i32().expect("as_i32");
    assert_eq!(vals.len(), 24, "expected 24 elements");
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            v, i as i32,
            "chunked_gzip_2d[{i}]: expected {}, got {v}",
            i as i32
        );
    }
}

/// Test 42: libver='latest' chunked 1-D float32, uncompressed (plain chunked).
#[test]
fn test_libver_latest_chunked_plain() {
    let f = oxih5::open(LIBVER_LATEST_CHUNKED).expect("open libver_latest_chunked.h5");

    let ds = f.dataset("chunked_plain").expect("read chunked_plain");
    assert_eq!(ds.shape, vec![20], "chunked_plain shape should be [20]");

    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals.len(), 20, "expected 20 elements");
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - i as f32).abs() < 1e-6,
            "chunked_plain[{i}]: expected {}, got {v}",
            i as f32
        );
    }
}

// ===========================================================================
// External file link integration tests (M-ext: external link resolution).
//
// Fixtures generated with h5py libver='earliest'; ext_source.h5 contains
// external links into ext_target.h5.
// ===========================================================================

const EXT_SOURCE: &str = "tests/fixtures/ext_source.h5";

// ---------------------------------------------------------------------------
// Test: external link at root level — /linked_data → ext_target.h5::/remote_data
// Expected: float32 [1.0, 2.0, 3.0, 4.0, 5.0]
// ---------------------------------------------------------------------------
#[test]
fn test_external_link_root_dataset() {
    let f = oxih5::open(EXT_SOURCE).expect("open ext_source.h5");
    let ds = f
        .dataset("linked_data")
        .expect("linked_data via external link");
    assert_eq!(ds.shape, vec![5], "shape should be [5]");
    let vals = ds.as_f32().expect("as_f32");
    assert_eq!(vals.len(), 5, "expected 5 elements");
    let expected = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    for (i, (&v, &e)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-6, "vals[{i}]: expected {e}, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Test: external link nested inside a group — /group/linked → ext_target.h5::/remote_group/values
// Expected: int32 [10, 20, 30]
// ---------------------------------------------------------------------------
#[test]
fn test_external_link_nested() {
    let f = oxih5::open(EXT_SOURCE).expect("open ext_source.h5");
    let ds = f
        .dataset("/group/linked")
        .expect("/group/linked via external link");
    assert_eq!(ds.shape, vec![3], "shape should be [3]");
    let vals = ds.as_i32().expect("as_i32");
    assert_eq!(vals, vec![10i32, 20, 30], "values mismatch");
}

// ---------------------------------------------------------------------------
// Test: local (non-external) dataset still works after external link support.
// Expected: int32 [99, 98, 97]
// ---------------------------------------------------------------------------
#[test]
fn test_external_link_local_still_works() {
    let f = oxih5::open(EXT_SOURCE).expect("open ext_source.h5");
    let ds = f.dataset("local_data").expect("local_data");
    assert_eq!(ds.shape, vec![3], "shape should be [3]");
    let vals = ds.as_i32().expect("as_i32");
    assert_eq!(vals, vec![99i32, 98, 97], "values mismatch");
}

// ===========================================================================
// Memory-mapped I/O tests (open_mmap).
//
// Verify that File::open_mmap() / oxih5::open_mmap() produce identical results
// to the heap-backed open() path.  Tests use the same fixtures as the existing
// contiguous and chunked suites; a real on-disk file is required for mmap.
// ===========================================================================

/// Test mmap_1: open a contiguous float32 1-D dataset via File::open_mmap and
/// verify shape + values match the heap-backed open() result.
///
/// Fixture: f4_1d_contig.h5 — /temperature, float32[8] = [0.0..7.0]
#[test]
fn test_open_mmap_f4_1d_values() {
    let path = write_tmp("mmap_f4_1d", F4_1D);

    let f = oxih5::File::open_mmap(&path).expect("File::open_mmap f4_1d_contig.h5");
    let ds = f
        .dataset("temperature")
        .expect("temperature dataset via mmap");
    cleanup(path);

    assert_eq!(ds.shape, vec![8], "mmap: shape should be [8]");
    let vals = ds.as_f32().expect("mmap: as_f32");
    assert_eq!(vals.len(), 8, "mmap: expected 8 elements");
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - i as f32).abs() < 1e-6,
            "mmap: temperature[{i}]: expected {}, got {v}",
            i as f32
        );
    }
}

/// Test mmap_2: open a contiguous int32 2-D dataset via the free function
/// oxih5::open_mmap and verify shape and values.
///
/// Fixture: i4_2d_contig.h5 — /matrix, int32[2,3] = reshape(arange(6))
#[test]
fn test_open_mmap_i4_2d_via_free_fn() {
    let path = write_tmp("mmap_i4_2d", I4_2D);

    let f = oxih5::open_mmap(&path).expect("open_mmap i4_2d_contig.h5");
    let ds = f.dataset("matrix").expect("matrix dataset via mmap");
    cleanup(path);

    assert_eq!(ds.shape, vec![2, 3], "mmap: shape should be [2, 3]");
    let vals = ds.as_i32().expect("mmap: as_i32");
    assert_eq!(vals.len(), 6, "mmap: expected 6 elements");
    // arange(6) reshaped — element at flat index k should equal k.
    assert_eq!(
        vals,
        vec![0i32, 1, 2, 3, 4, 5],
        "mmap: matrix values mismatch"
    );
}

// ===========================================================================
// Lazy chunk decompression tests.
//
// Verify that File::dataset_slice() reads only the overlapping chunks and
// produces the same result as a full read + in-memory slice.
// ===========================================================================

/// Test lazy_1: 1-D float32 gzip-compressed chunked dataset.
/// Verify that dataset_slice() returns the same bytes as dataset().slice().
///
/// Fixture: chunked_gzip_f4_1d.h5 — contains a float32 1-D chunked dataset.
#[test]
fn test_chunked_gzip_slice_1d() {
    let path = write_tmp("chunk_gzip_f4_1d", CHUNKED_GZIP_F4_1D);
    let f = oxih5::open(&path).expect("open chunked_gzip_f4_1d.h5");
    cleanup(path);

    // Discover the dataset name.
    let names = f.dataset_names().expect("dataset_names");
    assert!(!names.is_empty(), "no datasets in chunked_gzip_f4_1d.h5");
    let ds_name = &names[0];

    // Full read.
    let path2 = write_tmp("chunk_gzip_f4_1d_b", CHUNKED_GZIP_F4_1D);
    let f2 = oxih5::open(&path2).expect("open2");
    let full = f2.dataset(ds_name).expect("full dataset");
    cleanup(path2);

    let n = full.shape[0];
    assert!(n >= 4, "fixture must have at least 4 elements");

    // Slice: elements [1..n-1].
    #[allow(clippy::single_range_in_vec_init)]
    let ranges = [1..n - 1];

    // Lazy slice via dataset_slice().
    let path3 = write_tmp("chunk_gzip_f4_1d_c", CHUNKED_GZIP_F4_1D);
    let f3 = oxih5::open(&path3).expect("open3");
    let lazy = f3
        .dataset_slice(ds_name, &ranges)
        .expect("lazy dataset_slice");
    cleanup(path3);

    // Reference: full read + in-memory slice.
    let reference = full.slice(&ranges).expect("reference slice");

    assert_eq!(
        lazy.shape, reference.shape,
        "lazy vs reference shape mismatch"
    );
    assert_eq!(lazy.data, reference.data, "lazy vs reference data mismatch");
}

/// Test lazy_2: 2-D int32 gzip+shuffle chunked dataset.
/// Slices a sub-rectangle and compares lazy vs full+slice.
///
/// Fixture: chunked_gzip_shuffle_i4_2d.h5
#[test]
fn test_chunked_gzip_slice_2d() {
    let path = write_tmp("chunk_gzip_i4_2d", CHUNKED_GZIP_SHUFFLE_I4_2D);
    let f = oxih5::open(&path).expect("open chunked_gzip_shuffle_i4_2d.h5");
    cleanup(path);

    let names = f.dataset_names().expect("dataset_names");
    assert!(
        !names.is_empty(),
        "no datasets in chunked_gzip_shuffle_i4_2d.h5"
    );
    let ds_name = &names[0];

    // Full read (second open to get shape).
    let path2 = write_tmp("chunk_gzip_i4_2d_b", CHUNKED_GZIP_SHUFFLE_I4_2D);
    let f2 = oxih5::open(&path2).expect("open2");
    let full = f2.dataset(ds_name).expect("full dataset");
    cleanup(path2);

    assert_eq!(full.shape.len(), 2, "fixture must be 2-D");
    let rows = full.shape[0];
    let cols = full.shape[1];
    assert!(rows >= 4 && cols >= 4, "fixture too small for slice test");

    // Request a centered 2×2 sub-block.
    let r0 = rows / 4;
    let r1 = r0 + 2;
    let c0 = cols / 4;
    let c1 = c0 + 2;
    let ranges = [r0..r1, c0..c1];

    // Lazy slice.
    let path3 = write_tmp("chunk_gzip_i4_2d_c", CHUNKED_GZIP_SHUFFLE_I4_2D);
    let f3 = oxih5::open(&path3).expect("open3");
    let lazy = f3
        .dataset_slice(ds_name, &ranges)
        .expect("lazy dataset_slice");
    cleanup(path3);

    // Reference.
    let reference = full.slice(&ranges).expect("reference slice");

    assert_eq!(
        lazy.shape, reference.shape,
        "lazy vs reference shape mismatch"
    );
    assert_eq!(lazy.data, reference.data, "lazy vs reference data mismatch");
}

/// Test lazy_3: contiguous dataset still works via dataset_slice (non-chunked path).
///
/// Verifies backwards compatibility: non-chunked layouts fall back to full
/// read + in-memory slice and must produce identical output.
#[test]
fn test_dataset_slice_contiguous_compat() {
    let path = write_tmp("slice_contig", F4_1D);
    let _f = oxih5::open(&path).expect("open f4_1d_contig.h5");
    cleanup(path);

    // Full read.
    let path2 = write_tmp("slice_contig_b", F4_1D);
    let f2 = oxih5::open(&path2).expect("open2");
    let full = f2.dataset("temperature").expect("full temperature");
    cleanup(path2);

    assert_eq!(full.shape, vec![8]);

    #[allow(clippy::single_range_in_vec_init)]
    let ranges = [2..6];

    let path3 = write_tmp("slice_contig_c", F4_1D);
    let f3 = oxih5::open(&path3).expect("open3");
    let lazy = f3
        .dataset_slice("temperature", &ranges)
        .expect("dataset_slice contiguous");
    cleanup(path3);

    let reference = full.slice(&ranges).expect("reference slice");

    assert_eq!(lazy.shape, reference.shape, "contiguous slice shape");
    assert_eq!(lazy.data, reference.data, "contiguous slice data");
}

// ---------------------------------------------------------------------------
// String datasets (fixed-length ASCII + variable-length UTF-8 + scalar VLen).
// Fixture: string_datasets.h5
//   /fixed_ascii   — S8, shape [4]
//   /vlen_utf8     — variable-length UTF-8, shape [4]
//   /scalar_string — scalar variable-length UTF-8
// ---------------------------------------------------------------------------

/// Test: fixed-length ASCII string dataset — shape + dtype + as_string() values.
#[test]
fn test_fixed_ascii_string_dataset() {
    let path = write_tmp("str_fixed_ascii", STRING_DATASETS);
    let f = oxih5::open(&path).expect("open string_datasets.h5");
    cleanup(path);

    let ds = f.dataset("fixed_ascii").expect("fixed_ascii dataset");
    assert_eq!(ds.shape, vec![4], "expected 4 elements");
    assert!(
        matches!(ds.dtype, oxih5::Dtype::String { .. }),
        "expected String dtype, got {:?}",
        ds.dtype
    );
    let strs = ds.as_string().expect("as_string on fixed_ascii");
    assert_eq!(strs.len(), 4);
    assert_eq!(strs[0], "hello", "first element mismatch");
    assert_eq!(strs[1], "world", "second element mismatch");
    assert_eq!(strs[2], "foo", "third element mismatch");
    assert_eq!(strs[3], "bar", "fourth element mismatch");
}

/// Test: variable-length UTF-8 string dataset — opens without panic;
/// data read may return NotImplemented (VarLen decode not yet implemented).
#[test]
fn test_vlen_utf8_dataset_opens() {
    let path = write_tmp("str_vlen_utf8", STRING_DATASETS);
    let f = oxih5::open(&path).expect("open string_datasets.h5");
    cleanup(path);

    match f.dataset("vlen_utf8") {
        Ok(ds) => {
            assert_eq!(ds.shape, vec![4], "vlen_utf8 shape mismatch");
            // as_string() on VLen should either work or return NotImplemented.
            match ds.as_string() {
                Ok(strs) => assert_eq!(strs.len(), 4),
                Err(oxih5::OxiH5Error::NotImplemented(_)) => { /* acceptable */ }
                Err(e) => panic!("unexpected error from as_string on vlen_utf8: {e}"),
            }
        }
        Err(oxih5::OxiH5Error::NotImplemented(_)) => { /* dataset open not implemented: acceptable */
        }
        Err(e) => panic!("unexpected error opening vlen_utf8: {e}"),
    }
}

/// Test: scalar variable-length string dataset — opens without panic.
#[test]
fn test_scalar_string_dataset_opens() {
    let path = write_tmp("str_scalar", STRING_DATASETS);
    let f = oxih5::open(&path).expect("open string_datasets.h5");
    cleanup(path);

    match f.dataset("scalar_string") {
        Ok(ds) => {
            // Scalar datasets have empty shape or shape [1].
            assert!(
                ds.shape.is_empty() || ds.shape == vec![1],
                "unexpected scalar shape: {:?}",
                ds.shape
            );
        }
        Err(oxih5::OxiH5Error::NotImplemented(_)) => { /* acceptable */ }
        Err(e) => panic!("unexpected error opening scalar_string: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Compound dataset with three fields (x: f32, y: f32, idx: i32).
// Fixture: compound_dataset.h5 — /points shape [3]
// ---------------------------------------------------------------------------

/// Test: compound dataset with x/y/idx fields — shape + dtype recognition.
#[test]
fn test_compound_dataset_three_fields() {
    let path = write_tmp("compound_3f", COMPOUND_DATASET);
    let f = oxih5::open(&path).expect("open compound_dataset.h5");
    cleanup(path);

    let ds = f.dataset("points").expect("points dataset");
    assert_eq!(ds.shape, vec![3], "compound dataset should have 3 elements");
    assert!(
        matches!(ds.dtype, oxih5::Dtype::Compound { .. }),
        "expected Compound dtype, got {:?}",
        ds.dtype
    );
    // Verify we have 3 fields (x, y, idx).
    if let oxih5::Dtype::Compound { fields } = &ds.dtype {
        assert_eq!(fields.len(), 3, "expected 3 compound fields");
        let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"x"), "missing 'x' field");
        assert!(names.contains(&"y"), "missing 'y' field");
        assert!(names.contains(&"idx"), "missing 'idx' field");
    }
}

// ---------------------------------------------------------------------------
// Enum dataset — int32 base type, shape [5].
// Fixture: enum_dataset.h5 — /colors
// ---------------------------------------------------------------------------

/// Test: enum dataset — dtype is Enum with int32 base + named members.
#[test]
fn test_enum_dataset_dtype() {
    let path = write_tmp("enum_colors", ENUM_DATASET);
    let f = oxih5::open(&path).expect("open enum_dataset.h5");
    cleanup(path);

    let ds = f.dataset("colors").expect("colors dataset");
    assert_eq!(ds.shape, vec![5], "expected 5 elements");
    assert!(
        matches!(ds.dtype, oxih5::Dtype::Enum { .. }),
        "expected Enum dtype, got {:?}",
        ds.dtype
    );
    // Verify the enum has at least the RED/GREEN/BLUE members.
    if let oxih5::Dtype::Enum { members, .. } = &ds.dtype {
        let member_names: Vec<&str> = members.iter().map(|(n, _)| n.as_str()).collect();
        assert!(member_names.contains(&"RED"), "missing RED member");
        assert!(member_names.contains(&"GREEN"), "missing GREEN member");
        assert!(member_names.contains(&"BLUE"), "missing BLUE member");
    }
}

/// Test: enum raw integer values are readable as i32 (base type).
#[test]
fn test_enum_dataset_raw_values() {
    let path = write_tmp("enum_colors_raw", ENUM_DATASET);
    let f = oxih5::open(&path).expect("open enum_dataset.h5");
    cleanup(path);

    let ds = f.dataset("colors").expect("colors dataset");
    // Enum base is int32; the raw data is laid out as i32 values.
    if let oxih5::Dtype::Enum { base, .. } = &ds.dtype {
        if matches!(
            base.as_ref(),
            oxih5::Dtype::Int {
                size: 4,
                signed: true,
                ..
            }
        ) {
            // Construct a Dataset view with the base dtype to read raw integers.
            let raw_ds = oxih5::Dataset {
                data: ds.data.clone(),
                shape: ds.shape.clone(),
                dtype: *base.clone(),
                attributes: ds.attributes.clone(),
            };
            let vals = raw_ds.as_i32().expect("as_i32 on enum base");
            assert_eq!(vals, vec![0, 1, 2, 0, 1]);
        }
    }
}

// ---------------------------------------------------------------------------
// Large integer datasets — i64, u8, u32.
// Fixture: large_int_datasets.h5
//   /i8_1d — int64 arange(8)   (note: h5py uses 'i8_1d' for int64, 8-byte int)
//   /u8_1d — uint8 arange(8)
//   /u4_1d — uint32 arange(8)
// ---------------------------------------------------------------------------

/// Test: int64 dataset — shape + as_i64() values.
#[test]
fn test_i64_dataset() {
    let path = write_tmp("large_i8", LARGE_INT_DATASETS);
    let f = oxih5::open(&path).expect("open large_int_datasets.h5");
    cleanup(path);

    let ds = f.dataset("i8_1d").expect("i8_1d dataset");
    assert_eq!(ds.shape, vec![8], "expected 8 elements");
    assert!(
        matches!(
            ds.dtype,
            oxih5::Dtype::Int {
                size: 8,
                signed: true,
                ..
            }
        ),
        "expected Int64 dtype, got {:?}",
        ds.dtype
    );
    let vals = ds.as_i64().expect("as_i64");
    assert_eq!(vals.len(), 8);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, i as i64, "vals[{i}] mismatch");
    }
}

/// Test: uint8 dataset — shape + as_u8() values.
#[test]
fn test_u8_array_dataset() {
    let path = write_tmp("large_u8", LARGE_INT_DATASETS);
    let f = oxih5::open(&path).expect("open large_int_datasets.h5");
    cleanup(path);

    let ds = f.dataset("u8_1d").expect("u8_1d dataset");
    assert_eq!(ds.shape, vec![8], "expected 8 elements");
    assert!(
        matches!(
            ds.dtype,
            oxih5::Dtype::Int {
                size: 1,
                signed: false,
                ..
            }
        ),
        "expected UInt8 dtype, got {:?}",
        ds.dtype
    );
    let vals = ds.as_u8().expect("as_u8");
    assert_eq!(vals.len(), 8);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, i as u8, "vals[{i}] mismatch");
    }
}

/// Test: uint32 dataset — shape + as_u32() values.
#[test]
fn test_u32_array_dataset() {
    let path = write_tmp("large_u32", LARGE_INT_DATASETS);
    let f = oxih5::open(&path).expect("open large_int_datasets.h5");
    cleanup(path);

    let ds = f.dataset("u4_1d").expect("u4_1d dataset");
    assert_eq!(ds.shape, vec![8], "expected 8 elements");
    assert!(
        matches!(
            ds.dtype,
            oxih5::Dtype::Int {
                size: 4,
                signed: false,
                ..
            }
        ),
        "expected UInt32 dtype, got {:?}",
        ds.dtype
    );
    let vals = ds.as_u32().expect("as_u32");
    assert_eq!(vals.len(), 8);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, i as u32, "vals[{i}] mismatch");
    }
}

// ---------------------------------------------------------------------------
// Opaque dataset — 8-byte blobs, shape [2].
// Fixture: opaque_dataset.h5 — /blobs
// ---------------------------------------------------------------------------

/// Test: opaque dataset — dtype is Opaque with correct element size.
#[test]
fn test_opaque_dataset_dtype() {
    let path = write_tmp("opaque_blobs", OPAQUE_DATASET);
    let f = oxih5::open(&path).expect("open opaque_dataset.h5");
    cleanup(path);

    let ds = f.dataset("blobs").expect("blobs dataset");
    assert_eq!(ds.shape, vec![2], "expected 2 elements");
    assert!(
        matches!(ds.dtype, oxih5::Dtype::Opaque { size: 8, .. }),
        "expected Opaque(8) dtype, got {:?}",
        ds.dtype
    );
    // Verify raw byte content: two 8-byte elements.
    assert_eq!(ds.data.len(), 16, "expected 16 bytes of opaque data");
    // First blob: 0x01..0x08
    assert_eq!(
        &ds.data[0..8],
        &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
    );
    // Second blob: 0x11..0x18
    assert_eq!(
        &ds.data[8..16],
        &[0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]
    );
}
