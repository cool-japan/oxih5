//! Integration tests for the **extensible array** chunk index (index type 4).
//!
//! `tests/fixtures/chunked_ea.h5` is written by `tests/gen_fixtures.py` with
//! `libver='latest'`, which is what makes libhdf5 choose an extensible array for
//! every chunked dataset that has exactly one unlimited dimension — the
//! ordinary `create_dataset(..., chunks=..., maxshape=(None, ...))` case.
//!
//! The fixture deliberately reaches every level of the on-disk structure: the
//! index block's inline elements, data blocks addressed from the index block,
//! data blocks reached through a secondary block, the filtered element format,
//! unallocated elements, and the coordinate *rotation* that decides element
//! order once the unlimited dimension is not dimension 0.

use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn open_ea() -> oxih5::File {
    let path = fixture("chunked_ea.h5");
    // Tracked in git — a missing file means a broken checkout and must fail
    // loudly rather than turn every test in this file into a silent pass.
    assert!(
        path.exists(),
        "fixture chunked_ea.h5 missing at {} — it is tracked in git",
        path.display()
    );
    oxih5::open(&path).expect("open chunked_ea.h5")
}

// ---------------------------------------------------------------------------
// 1-D: the shape libhdf5 uses an extensible array for most often.
// ---------------------------------------------------------------------------

/// Four chunks fit in the index block's inline element area, so no data block
/// is ever allocated.
#[test]
fn ea_inline_elements_only() {
    let file = open_ea();
    let ds = file.dataset("ea_1d").expect("read ea_1d");
    assert_eq!(ds.shape, vec![10]);
    assert_eq!(ds.as_i32().expect("i32"), (0..10).collect::<Vec<i32>>());
}

/// 125 chunks spill into five data blocks of 16/32/32/32/64 elements, all
/// addressed straight from the index block.
#[test]
fn ea_index_block_data_blocks() {
    let file = open_ea();
    let ds = file.dataset("ea_1d_blocks").expect("read ea_1d_blocks");
    assert_eq!(ds.shape, vec![500]);
    assert_eq!(ds.as_i32().expect("i32"), (0..500).collect::<Vec<i32>>());
}

/// 400 chunks exhaust the index block's six data-block slots and continue
/// through a secondary block ("EASB").
#[test]
fn ea_secondary_block_traversal() {
    let file = open_ea();
    let ds = file
        .dataset("ea_1d_secondary")
        .expect("read ea_1d_secondary");
    assert_eq!(ds.shape, vec![400]);
    assert_eq!(ds.as_i32().expect("i32"), (0..400).collect::<Vec<i32>>());
}

/// A filtered array stores address + a variable-width stored size + the filter
/// mask per element; the stored size is what the deflate decoder is handed.
#[test]
fn ea_filtered_chunks() {
    let file = open_ea();
    let ds = file.dataset("ea_1d_gzip").expect("read ea_1d_gzip");
    assert_eq!(ds.shape, vec![200]);
    assert_eq!(ds.as_i32().expect("i32"), (0..200).collect::<Vec<i32>>());
}

/// Chunks that were never written have no element in the array, and must read
/// back as the dataset's fill value rather than as zeros or as an error.
#[test]
fn ea_unallocated_chunks_read_as_fill_value() {
    let file = open_ea();
    let ds = file.dataset("ea_1d_sparse").expect("read ea_1d_sparse");
    assert_eq!(ds.shape, vec![40]);
    let values = ds.as_i32().expect("i32");
    let mut expected = vec![-7i32; 40];
    expected[0..4].copy_from_slice(&[100, 101, 102, 103]);
    expected[20..24].copy_from_slice(&[200, 201, 202, 203]);
    assert_eq!(values, expected);
}

// ---------------------------------------------------------------------------
// Rank ≥ 2: element order depends on where the unlimited dimension sits.
// ---------------------------------------------------------------------------

/// Unlimited dimension first: element order is plain row-major over the chunk
/// grid.
#[test]
fn ea_2d_unlimited_first_dim() {
    let file = open_ea();
    let ds = file.dataset("ea_2d_unlim0").expect("read ea_2d_unlim0");
    assert_eq!(ds.shape, vec![6, 4]);
    assert_eq!(ds.as_i32().expect("i32"), (0..24).collect::<Vec<i32>>());
}

/// Unlimited dimension last: libhdf5 rotates it to the front before computing
/// the element index, so the chunks appear in a different order on disk.
#[test]
fn ea_2d_unlimited_last_dim() {
    let file = open_ea();
    let ds = file.dataset("ea_2d_unlim1").expect("read ea_2d_unlim1");
    assert_eq!(ds.shape, vec![4, 6]);
    assert_eq!(ds.as_i32().expect("i32"), (0..24).collect::<Vec<i32>>());
}

/// The chunk grid comes from the *maximum* dimensions: with `maxshape=(None, 8)`
/// each row of the grid holds four chunks even though only two are in use, so
/// element indices skip.  Using the current shape instead would shift every
/// chunk after the first row.
#[test]
fn ea_2d_grid_uses_max_dims() {
    let file = open_ea();
    let ds = file.dataset("ea_2d_wide_max").expect("read ea_2d_wide_max");
    assert_eq!(ds.shape, vec![6, 4]);
    assert_eq!(ds.as_i32().expect("i32"), (0..24).collect::<Vec<i32>>());
}

/// Rank 3 with the unlimited dimension last.  `H5VM_swizzle_coords` rotates
/// (0, 1, 2) to (2, 0, 1); swapping positions 0 and 2 would give (2, 1, 0) and
/// misplace eight of the twelve chunks.
#[test]
fn ea_3d_unlimited_last_dim_rotates() {
    let file = open_ea();
    let ds = file.dataset("ea_3d_unlim2").expect("read ea_3d_unlim2");
    assert_eq!(ds.shape, vec![4, 6, 4]);
    assert_eq!(ds.as_i32().expect("i32"), (0..96).collect::<Vec<i32>>());
}

// ---------------------------------------------------------------------------
// The other two readers share the index but not the scatter path.
// ---------------------------------------------------------------------------

// `dataset_slice` takes one range per dimension; clippy reads `&[8..20]` as a
// mistyped `&[8; 20]`, but a one-element array of ranges is exactly right here.
#[allow(clippy::single_range_in_vec_init)]
#[test]
fn ea_slice_spans_several_data_blocks() {
    let file = open_ea();
    // 8..200 crosses the inline elements, the 16-element data block and into
    // the 32-element one.
    let ds = file
        .dataset_slice("ea_1d_blocks", &[8..200])
        .expect("slice ea_1d_blocks");
    assert_eq!(ds.shape, vec![192]);
    assert_eq!(ds.as_i32().expect("i32"), (8..200).collect::<Vec<i32>>());

    let sparse = file
        .dataset_slice("ea_1d_sparse", &[2..24])
        .expect("slice ea_1d_sparse");
    let mut expected = vec![-7i32; 22];
    expected[0..2].copy_from_slice(&[102, 103]);
    expected[18..22].copy_from_slice(&[200, 201, 202, 203]);
    assert_eq!(sparse.as_i32().expect("i32"), expected);
}

#[test]
fn ea_hyperslab_over_rotated_grid() {
    let file = open_ea();
    let sel = [
        oxih5::DimSelection {
            start: 0,
            stride: 2,
            count: 2,
            block: 1,
        },
        oxih5::DimSelection {
            start: 1,
            stride: 2,
            count: 3,
            block: 1,
        },
    ];
    let ds = file
        .dataset_hyperslab("ea_2d_unlim1", &sel)
        .expect("hyperslab over ea_2d_unlim1");
    assert_eq!(ds.shape, vec![2, 3]);
    // Rows 0 and 2, columns 1, 3 and 5 of 0..24 reshaped (4, 6).
    assert_eq!(ds.as_i32().expect("i32"), vec![1, 3, 5, 13, 15, 17]);
}

/// Paged data blocks, at libhdf5's own creation parameters.
///
/// With the parameters libhdf5 hard-codes (`max_nelmts_bits=32`,
/// `data_blk_min_elmts=16`, `sup_blk_min_data_ptrs=4`,
/// `max_dblk_page_nelmts_bits=10`) a data block only exceeds one 1024-element
/// page from super block 13 onwards, whose first element is number 131 056.  A
/// fixture reaching that is ~1.2 MB — too large to track next to 20 KB
/// fixtures — so it is generated here instead.  The unit tests in
/// `oxih5-format/src/ea_index/tests.rs` cover the same code paths at synthetic
/// parameters; this pins the field *widths* (a 4-byte block offset and a
/// 64-byte page-init bitmask) that only the real parameters produce.
///
/// Skipped, loudly, when python3/h5py is not installed.
#[test]
fn ea_paged_data_blocks_at_libhdf5_parameters() {
    let path = std::env::temp_dir().join("oxih5_ea_paged_test.h5");
    let _ = std::fs::remove_file(&path);
    let script = format!(
        "import h5py, numpy as np\n\
         with h5py.File(r'{p}', 'w', libver='latest') as f:\n\
         \x20   f.create_dataset('paged', data=np.arange(131200, dtype='int8'), \
         chunks=(1,), maxshape=(None,))\n\
         print('written')",
        p = path.display()
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output();
    match out {
        Ok(o) if o.status.success() && path.exists() => {
            let file = oxih5::open(&path).expect("open generated paged fixture");
            let ds = file.dataset("paged").expect("read paged dataset");
            assert_eq!(ds.shape, vec![131200]);
            let values = ds.as_i8().expect("i8");
            let expected: Vec<i8> = (0..131200u32).map(|i| i as u8 as i8).collect();
            assert_eq!(values, expected);
            let _ = std::fs::remove_file(&path);
        }
        _ => {
            let _ = std::fs::remove_file(&path);
            eprintln!(
                "python3/h5py unavailable — skipping the paged extensible-array \
                 check (synthetic coverage in oxih5-format/src/ea_index/tests.rs)"
            );
        }
    }
}

#[test]
fn ea_all_datasets_are_readable() {
    let file = open_ea();
    let names = file.dataset_names().expect("list root datasets");
    for expected in [
        "ea_1d",
        "ea_1d_blocks",
        "ea_1d_secondary",
        "ea_1d_gzip",
        "ea_1d_sparse",
        "ea_2d_unlim0",
        "ea_2d_unlim1",
        "ea_2d_wide_max",
        "ea_3d_unlim2",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "expected '{expected}' in {names:?}"
        );
        file.dataset(expected)
            .unwrap_or_else(|e| panic!("dataset '{expected}' must be readable: {e}"));
    }
}
