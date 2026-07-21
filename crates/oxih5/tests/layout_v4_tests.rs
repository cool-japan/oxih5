//! Integration tests for **data-layout message version 4**.
//!
//! `tests/fixtures/layout_v4.h5` is produced by `tests/gen_fixtures.py` with
//! `libver='latest'`, which makes libhdf5 emit version-4 layout messages for
//! every dataset.  Version 4 keeps the version-3 encoding for classes 0
//! (compact) and 1 (contiguous), but re-encodes class 2 (chunked): the chunk
//! dimensions become variable-width integers preceded by a "dimension size
//! encoded length" byte, and the version-1 B-tree is replaced by an explicit
//! chunk-indexing-type byte plus that index's creation parameters.
//!
//! The fixture carries one dataset per chunk-index type so that every branch of
//! the v4 chunked decoder is exercised against bytes libhdf5 actually wrote.

use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn open_fixture() -> Option<oxih5::File> {
    let path = fixture("layout_v4.h5");
    if !path.exists() {
        return None;
    }
    Some(oxih5::open(&path).expect("open layout_v4.h5"))
}

// ---------------------------------------------------------------------------
// Class 1 — contiguous.  This is the confirmed regression: `libver='latest'`
// files could not have their contiguous datasets read at all.
// ---------------------------------------------------------------------------

#[test]
fn layout_v4_contiguous_1d() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("contig").expect("read v4 contiguous dataset");
    assert_eq!(ds.shape, vec![5]);
    assert_eq!(ds.as_f64().expect("f64"), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn layout_v4_contiguous_2d() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("contig2d").expect("read v4 contiguous 2-D");
    assert_eq!(ds.shape, vec![3, 4]);
    assert_eq!(ds.as_i32().expect("i32"), (0..12).collect::<Vec<i32>>());
}

#[test]
fn layout_v4_contiguous_scalar() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("scalar").expect("read v4 contiguous scalar");
    assert!(
        ds.shape.is_empty(),
        "scalar shape should be []: {:?}",
        ds.shape
    );
    assert_eq!(ds.as_f64().expect("f64"), vec![3.5]);
}

/// Attributes on a version-4 layout dataset still decode.  h5py writes a
/// Python `str` attribute as a *variable-length* string, so its bytes are a
/// global-heap reference rather than inline characters — reading it needs the
/// file, hence `attr_views` rather than `Dataset::attributes`.
#[test]
fn layout_v4_contiguous_carries_attributes() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("contig").expect("read v4 contiguous dataset");
    assert!(
        ds.attributes.iter().any(|a| a.name == "units"),
        "the 'units' attribute message must be found: {:?}",
        ds.attributes.iter().map(|a| &a.name).collect::<Vec<_>>()
    );

    let views = file.attr_views("contig").expect("attr views");
    let units = views
        .iter()
        .find(|a| a.name() == "units")
        .expect("'units' attribute");
    assert_eq!(units.as_strings().expect("decode vlen string"), ["meters"]);
}

// ---------------------------------------------------------------------------
// Class 0 — compact (raw data inline in the layout message).
// ---------------------------------------------------------------------------

#[test]
fn layout_v4_compact() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("compact").expect("read v4 compact dataset");
    assert_eq!(ds.shape, vec![4]);
    assert_eq!(ds.as_i32().expect("i32"), vec![5, 6, 7, 8]);
}

// ---------------------------------------------------------------------------
// Class 2 — chunked, one test per chunk-indexing type.
// ---------------------------------------------------------------------------

/// Index type 3 — fixed array (fixed dimensions, several chunks).
#[test]
fn layout_v4_chunked_fixed_array() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("chunk_fa").expect("read fixed-array chunked");
    assert_eq!(ds.shape, vec![20]);
    assert_eq!(ds.as_i32().expect("i32"), (0..20).collect::<Vec<i32>>());
}

/// Index type 4 — extensible array (exactly one unlimited dimension).
///
/// The *layout message* now decodes correctly: the extensible-array index
/// carries five creation-parameter bytes between the index-type byte and the
/// index address, and the extensible-array header's index-block address sits
/// after six 8-byte counters.  Both of those were wrong before and produced a
/// nonsense address.
///
/// What is still missing is the element decoding: a real extensible array
/// stores 8-byte bare chunk addresses (or, when filtered, address + a
/// variable-width stored size + filter mask), and derives each chunk's position
/// from its linear index — whereas this crate's parser expects each element to
/// be a self-describing record.  That is reported as a typed `NotImplemented`
/// rather than decoded speculatively; see the note in `ea_index.rs`.
#[test]
fn layout_v4_chunked_extensible_array_is_reported_unsupported() {
    let Some(file) = open_fixture() else { return };
    match file.dataset("chunk_ea") {
        Err(oxih5::OxiH5Error::NotImplemented(msg)) => {
            assert!(
                msg.contains("extensible array"),
                "the error must name the index type: {msg}"
            );
        }
        Err(other) => panic!("expected a typed NotImplemented, got: {other:?}"),
        Ok(_) => panic!(
            "extensible-array element decoding is not implemented; if this now succeeds, \
             replace this test with a real value assertion"
        ),
    }
}

/// Index type 5 — version-2 B-tree (two or more unlimited dimensions).
///
/// Six creation-parameter bytes precede the index address here.
#[test]
fn layout_v4_chunked_btree_v2() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("chunk_bt2").expect("read B-tree-v2 chunked");
    assert_eq!(ds.shape, vec![3, 4]);
    assert_eq!(ds.as_i32().expect("i32"), (0..12).collect::<Vec<i32>>());
}

/// Index type 1 — single chunk, no filter pipeline.
#[test]
fn layout_v4_chunked_single_chunk() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("chunk_single").expect("read single-chunk");
    assert_eq!(ds.shape, vec![6]);
    assert_eq!(ds.as_i32().expect("i32"), (0..6).collect::<Vec<i32>>());
}

/// Index type 1 with a filter pipeline: the layout message itself carries the
/// stored (deflated) chunk size and the filter mask, because there is no index
/// structure on disk to hold them.
#[test]
fn layout_v4_chunked_single_chunk_filtered() {
    let Some(file) = open_fixture() else { return };
    let ds = file
        .dataset("chunk_single_gzip")
        .expect("read filtered single-chunk");
    assert_eq!(ds.shape, vec![64]);
    assert_eq!(ds.as_i32().expect("i32"), (0..64).collect::<Vec<i32>>());
}

/// Index type 2 — implicit: no index at all, chunks laid out contiguously in
/// row-major order starting at the layout message's address.
#[test]
fn layout_v4_chunked_implicit() {
    let Some(file) = open_fixture() else { return };
    let ds = file
        .dataset("chunk_implicit")
        .expect("read implicit-index chunked");
    assert_eq!(ds.shape, vec![8]);
    assert_eq!(ds.as_i32().expect("i32"), (0..8).collect::<Vec<i32>>());
}

/// A chunk dimension larger than 255 forces libhdf5 to widen the per-dimension
/// encoding to two bytes, which shifts every field after the dimension list.
#[test]
fn layout_v4_chunked_wide_dimension_encoding() {
    let Some(file) = open_fixture() else { return };
    let ds = file
        .dataset("chunk_wide")
        .expect("read wide-encoding chunked");
    assert_eq!(ds.shape, vec![2000]);
    let values = ds.as_i32().expect("i32");
    assert_eq!(values.len(), 2000);
    assert_eq!(values[0], 0);
    assert_eq!(values[999], 999);
    assert_eq!(values[1999], 1999);
}

// ---------------------------------------------------------------------------
// Cross-cutting: navigation and sub-region reads over v4 layouts.
// ---------------------------------------------------------------------------

#[test]
fn layout_v4_all_datasets_listed_and_readable() {
    let Some(file) = open_fixture() else { return };
    let names = file.dataset_names().expect("list root datasets");
    // `chunk_ea` is listed but not yet decodable — see
    // `layout_v4_chunked_extensible_array_is_reported_unsupported`.
    for expected in [
        "contig",
        "contig2d",
        "scalar",
        "compact",
        "chunk_fa",
        "chunk_ea",
        "chunk_bt2",
        "chunk_single",
        "chunk_single_gzip",
        "chunk_implicit",
        "chunk_wide",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "expected '{expected}' in {names:?}"
        );
    }
    for expected in [
        "contig",
        "contig2d",
        "scalar",
        "compact",
        "chunk_fa",
        "chunk_bt2",
        "chunk_single",
        "chunk_single_gzip",
        "chunk_implicit",
        "chunk_wide",
    ] {
        file.dataset(expected)
            .unwrap_or_else(|e| panic!("dataset '{expected}' must be readable: {e}"));
    }
}

// `dataset_slice` takes one range per dimension, so a 1-D dataset is sliced
// with a one-element array of ranges.  Clippy reads `&[1..4]` as a mistyped
// `&[1; 4]`; here it is exactly what is meant.
#[allow(clippy::single_range_in_vec_init)]
#[test]
fn layout_v4_slice_contiguous_and_chunked() {
    let Some(file) = open_fixture() else { return };

    let contig = file
        .dataset_slice("contig", &[1..4])
        .expect("slice v4 contiguous");
    assert_eq!(contig.as_f64().expect("f64"), vec![1.0, 2.0, 3.0]);

    let fa = file
        .dataset_slice("chunk_fa", &[6..11])
        .expect("slice fixed-array chunked");
    assert_eq!(fa.as_i32().expect("i32"), vec![6, 7, 8, 9, 10]);

    let single = file
        .dataset_slice("chunk_single", &[2..5])
        .expect("slice single-chunk");
    assert_eq!(single.as_i32().expect("i32"), vec![2, 3, 4]);

    let implicit = file
        .dataset_slice("chunk_implicit", &[3..7])
        .expect("slice implicit-index chunked");
    assert_eq!(implicit.as_i32().expect("i32"), vec![3, 4, 5, 6]);
}

/// A strided hyperslab that spans both columns of the chunk grid, so the
/// selection is only satisfiable if every v2 B-tree record resolved.
///
/// NOTE: expressed with `count: 2, block: 1` rather than the equivalent
/// `count: 1, block: 2`.  Blocks wider than one element are currently dropped
/// by the chunked hyperslab reader for *every* chunk index type — including
/// layout-v3 version-1 B-trees — so using `block > 1` here would test that
/// unrelated pre-existing defect instead of the chunk index.
#[test]
fn layout_v4_hyperslab_over_btree_v2() {
    let Some(file) = open_fixture() else { return };
    let sel = [
        oxih5::DimSelection {
            start: 0,
            stride: 2,
            count: 2,
            block: 1,
        },
        oxih5::DimSelection {
            start: 1,
            stride: 1,
            count: 2,
            block: 1,
        },
    ];
    let ds = file
        .dataset_hyperslab("chunk_bt2", &sel)
        .expect("hyperslab over B-tree-v2 chunked");
    assert_eq!(ds.shape, vec![2, 2]);
    // rows 0 and 2, columns 1 and 2 of 0..12 reshaped (3, 4)
    assert_eq!(ds.as_i32().expect("i32"), vec![1, 2, 9, 10]);
}
