// Integration tests for Item C: strided hyperslab selection public API.
//
// Tests exercise File::dataset_hyperslab, Group::dataset_hyperslab, and
// the free function read_dataset_hyperslab against the fixtures already used
// by the rest of the test suite.

use oxih5::DimSelection;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_tmp(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("oxih5_hs_{}_{}_{}.h5", std::process::id(), n, tag));
    std::fs::write(&path, bytes).expect("write temp fixture");
    path
}

fn cleanup(path: std::path::PathBuf) {
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Fixture bytes (same as read_contig.rs — re-embedded here for independence).
// ---------------------------------------------------------------------------

const F4_1D: &[u8] = include_bytes!("fixtures/f4_1d_contig.h5");
const CHUNKED_GZIP_F4_1D: &[u8] = include_bytes!("fixtures/chunked_gzip_f4_1d.h5");
const CHUNKED_GZIP_SHUFFLE_I4_2D: &[u8] = include_bytes!("fixtures/chunked_gzip_shuffle_i4_2d.h5");

// ---------------------------------------------------------------------------
// Test hs_1: contiguous DimSelection == dataset_slice for chunked fixture.
//
// A contiguous hyperslab (stride=1, block=1) must produce exactly the same
// bytes as the range-based dataset_slice API.
// ---------------------------------------------------------------------------
#[test]
fn test_hyperslab_contiguous_eq_slice_chunked_1d_gzip() {
    // Full read via standard API.
    let path_full = write_tmp("hs_eq_full", CHUNKED_GZIP_F4_1D);
    let f_full = oxih5::open(&path_full).expect("open full");
    let full = f_full.dataset("data").expect("full dataset");
    cleanup(path_full);

    let n = full.shape[0];
    assert!(n >= 4, "fixture must have >= 4 elements");

    // Contiguous hyperslab selecting [2..n-1].
    let sel = [DimSelection::contiguous(2..n as u64 - 1)];

    let path_hs = write_tmp("hs_eq_hs", CHUNKED_GZIP_F4_1D);
    let f_hs = oxih5::open(&path_hs).expect("open hs");
    let hs = f_hs
        .dataset_hyperslab("data", &sel)
        .expect("dataset_hyperslab");
    cleanup(path_hs);

    // Range-based slice for the same range.
    let path_sl = write_tmp("hs_eq_sl", CHUNKED_GZIP_F4_1D);
    let f_sl = oxih5::open(&path_sl).expect("open sl");
    #[allow(clippy::single_range_in_vec_init)]
    let sl = f_sl
        .dataset_slice("data", &[2..n - 1])
        .expect("dataset_slice");
    cleanup(path_sl);

    assert_eq!(hs.shape, sl.shape, "hyperslab shape != slice shape");
    assert_eq!(hs.data, sl.data, "hyperslab data != slice data");
}

// ---------------------------------------------------------------------------
// Test hs_2: stride=2 in 1-D → every other element.
//
// Selects indices {0, 2, 4, ...} and compares against manually extracted
// even-index elements from the full dataset.
// ---------------------------------------------------------------------------
#[test]
fn test_hyperslab_strided_1d() {
    // Full read.
    let path = write_tmp("hs_stride_full", CHUNKED_GZIP_F4_1D);
    let f = oxih5::open(&path).expect("open");
    let full = f.dataset("data").expect("full dataset");
    cleanup(path);

    let n = full.shape[0] as u64;
    let es = full.data.len() / full.shape[0].max(1); // element size in bytes
                                                     // count = ceil(n / 2): how many even-indexed elements.
    let count = n.div_ceil(2);

    // stride=2, block=1 → selects indices {0, 2, 4, ..., 2*(count-1)}
    let sel = [DimSelection {
        start: 0,
        stride: 2,
        count,
        block: 1,
    }];

    let path2 = write_tmp("hs_stride_hs", CHUNKED_GZIP_F4_1D);
    let f2 = oxih5::open(&path2).expect("open2");
    let hs = f2
        .dataset_hyperslab("data", &sel)
        .expect("strided hyperslab");
    cleanup(path2);

    assert_eq!(
        hs.shape,
        vec![count as usize],
        "strided output shape mismatch"
    );

    // Build reference by manually picking every second element.
    let mut reference = Vec::with_capacity(count as usize * es);
    for i in 0..count as usize {
        let src = i * 2 * es;
        reference.extend_from_slice(&full.data[src..src + es]);
    }

    assert_eq!(hs.data, reference, "strided data mismatch");
}

// ---------------------------------------------------------------------------
// Test hs_3: block=2 in the row dimension of a 2-D dataset.
//
// Uses a 2-D chunked fixture (i4, rows=6, cols=8).
// Row selection: start=0, stride=2, count=2, block=2 → rows {0,1,2,3}.
// Col selection: contiguous 0..cols.
// Result must match the first 4 rows of the full dataset.
// ---------------------------------------------------------------------------
#[test]
fn test_hyperslab_block2_2d() {
    let path = write_tmp("hs_block2_full", CHUNKED_GZIP_SHUFFLE_I4_2D);
    let f = oxih5::open(&path).expect("open");
    let full = f.dataset("data").expect("full 2d dataset");
    cleanup(path);

    let rows = full.shape[0] as u64;
    let cols = full.shape[1] as u64;
    assert!(rows >= 4, "fixture must have >= 4 rows");

    // Row dim: blocks of 2 with stride 2 → picks rows {0,1,2,3}.
    // Equivalent to contiguous 0..4 since stride==block.
    let sel = [
        DimSelection {
            start: 0,
            stride: 2,
            count: 2,
            block: 2,
        },
        DimSelection::contiguous(0..cols),
    ];

    let path2 = write_tmp("hs_block2_hs", CHUNKED_GZIP_SHUFFLE_I4_2D);
    let f2 = oxih5::open(&path2).expect("open2");
    let hs = f2
        .dataset_hyperslab("data", &sel)
        .expect("block2 hyperslab");
    cleanup(path2);

    // Expected: first 4 rows of the full dataset.
    let es = 4usize; // i32 elem size
    let row_bytes = cols as usize * es;
    let expected_data: Vec<u8> = full.data[..4 * row_bytes].to_vec();

    assert_eq!(hs.shape, vec![4, cols as usize], "block2 output shape");
    assert_eq!(hs.data, expected_data, "block2 data mismatch");
}

// ---------------------------------------------------------------------------
// Test hs_4: contiguous layout falls back to gather path.
//
// The contiguous float32 fixture has 8 elements [0.0..7.0].
// Select first 4 elements: should equal the first 4 floats.
// ---------------------------------------------------------------------------
#[test]
fn test_hyperslab_contiguous_layout_fallback() {
    let path = write_tmp("hs_contig_full", F4_1D);
    let f = oxih5::open(&path).expect("open contig fixture");
    let full = f.dataset("temperature").expect("full dataset");
    cleanup(path);

    let k: u64 = 4;
    let sel = [DimSelection::contiguous(0..k)];

    let path2 = write_tmp("hs_contig_hs", F4_1D);
    let f2 = oxih5::open(&path2).expect("open2");
    let hs = f2
        .dataset_hyperslab("temperature", &sel)
        .expect("contiguous fallback hyperslab");
    cleanup(path2);

    let es = 4usize; // f32
    assert_eq!(hs.shape, vec![k as usize], "contiguous fallback shape");
    assert_eq!(
        hs.data,
        full.data[..k as usize * es],
        "contiguous fallback data"
    );
}

// ---------------------------------------------------------------------------
// Test hs_5: empty selection returns empty data vector.
// ---------------------------------------------------------------------------
#[test]
fn test_hyperslab_empty_selection() {
    let sel = [DimSelection {
        start: 0,
        stride: 1,
        count: 0,
        block: 1,
    }];

    let path = write_tmp("hs_empty", CHUNKED_GZIP_F4_1D);
    let f = oxih5::open(&path).expect("open");
    let hs = f.dataset_hyperslab("data", &sel).expect("empty hyperslab");
    cleanup(path);

    assert!(hs.data.is_empty(), "empty selection should return no data");
    assert!(
        hs.shape.iter().product::<usize>() == 0,
        "empty selection shape product should be 0"
    );
}

// ---------------------------------------------------------------------------
// Test hs_6: free function read_dataset_hyperslab.
// ---------------------------------------------------------------------------
#[test]
fn test_read_dataset_hyperslab_free_fn() {
    let path = write_tmp("hs_free_fn", CHUNKED_GZIP_F4_1D);

    // Full read for reference.
    let f_ref = oxih5::open(&path).expect("open ref");
    let full = f_ref.dataset("data").expect("full dataset");
    let n = full.shape[0];

    let sel = [DimSelection::contiguous(0..n as u64)];
    let hs =
        oxih5::read_dataset_hyperslab(&path, "data", &sel).expect("read_dataset_hyperslab free fn");
    cleanup(path);

    assert_eq!(hs.shape, full.shape, "free fn shape mismatch");
    assert_eq!(hs.data, full.data, "free fn data mismatch");
}

// ---------------------------------------------------------------------------
// Test hs_7: Group::dataset_hyperslab method.
// ---------------------------------------------------------------------------
#[test]
fn test_group_dataset_hyperslab() {
    let path = write_tmp("hs_group", CHUNKED_GZIP_F4_1D);
    let f = oxih5::open(&path).expect("open");
    let root = f.root().expect("root group");

    let full = root.dataset("data").expect("full dataset via group");
    let n = full.shape[0];
    let es = full.data.len() / n.max(1);

    let sel = [DimSelection::contiguous(1..n as u64)];
    let hs = root
        .dataset_hyperslab("data", &sel)
        .expect("group dataset_hyperslab");
    cleanup(path);

    assert_eq!(hs.shape, vec![n - 1], "group hs shape mismatch");
    assert_eq!(hs.data, full.data[es..], "group hs data mismatch");
}
