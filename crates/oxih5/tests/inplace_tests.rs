//! Integration tests for in-place dataset overwrite.
//!
//! These exercise the public API (`oxih5::write_dataset_in_place` and friends)
//! against files authored by **libhdf5 itself** — the fixtures under
//! `tests/fixtures/` were produced by h5py (see `tests/gen_fixtures.py`).  That
//! matters: the whole point of the feature is to modify files oxih5 did not
//! write, so the interesting cases are real libhdf5 object headers, real
//! symbol-table groups, and real filter pipelines.
//!
//! Every test copies its fixture into a temporary file first; the fixtures
//! themselves are never opened for writing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static IP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    let n = IP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_inplace_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Copy a fixture to a fresh temp path, or `None` when it was never generated.
fn fixture_copy(name: &str, tag: &str) -> Option<PathBuf> {
    let src = fixture(name);
    if !src.exists() {
        return None;
    }
    let dst = tmp_path(tag);
    std::fs::copy(&src, &dst).expect("copy fixture to temp");
    Some(dst)
}

// ---------------------------------------------------------------------------
// Round-trip against libhdf5-authored files
// ---------------------------------------------------------------------------

/// Overwrite a contiguous float64 dataset in a file libhdf5 wrote.
#[test]
fn in_place_overwrite_of_libhdf5_contiguous_dataset() {
    let Some(path) = fixture_copy("f8_1d_contig.h5", "contig") else {
        return;
    };

    let result = oxih5::write_dataset_in_place_f64(&path, "signal", &[9.5, 8.5, 7.5, 6.5]);
    let values = oxih5::open(&path)
        .expect("reopen")
        .dataset("signal")
        .expect("signal")
        .as_f64()
        .expect("as_f64");
    cleanup(&path);

    result.expect("overwrite must succeed");
    assert_eq!(values, vec![9.5, 8.5, 7.5, 6.5]);
}

/// Overwriting a dataset inside a nested group must leave the *other* group's
/// dataset — and the group structure itself — completely intact.
#[test]
fn in_place_overwrite_inside_sub_group_preserves_siblings() {
    let Some(path) = fixture_copy("nested_groups.h5", "nested") else {
        return;
    };

    let result =
        oxih5::write_dataset_in_place_f32(&path, "/sensors/imu/accel", &[-1.5, -2.5, -3.5]);

    let file = oxih5::open(&path).expect("reopen");
    let accel = file
        .dataset("/sensors/imu/accel")
        .expect("accel")
        .as_f32()
        .expect("as_f32");
    let coords = file
        .dataset("/sensors/gps/coords")
        .expect("coords")
        .as_f64()
        .expect("as_f64");
    let groups = file.group("/sensors").expect("sensors").groups();
    drop(file);
    cleanup(&path);

    result.expect("overwrite must succeed");
    assert_eq!(accel, vec![-1.5, -2.5, -3.5], "target dataset updated");
    assert_eq!(coords.len(), 2, "sibling group's dataset still readable");
    let mut names = groups.expect("list sub-groups");
    names.sort();
    assert_eq!(names, vec!["gps", "imu"], "group structure intact");
}

/// A big-endian dataset must be written back big-endian.  The typed wrapper
/// reads the dataset's own byte order and encodes to match, so the caller never
/// has to know — a raw-bytes caller would silently corrupt this file.
#[test]
fn in_place_typed_write_respects_file_byte_order() {
    let Some(path) = fixture_copy("be_f4_1d_contig.h5", "bigendian") else {
        return;
    };

    let extent = oxih5::dataset_data_extent(&path, "voltage").expect("extent");
    let result = oxih5::write_dataset_in_place_f32(&path, "voltage", &[1.0, 2.0, 4.0, 8.0]);

    let values = oxih5::open(&path)
        .expect("reopen")
        .dataset("voltage")
        .expect("voltage")
        .as_f32()
        .expect("as_f32");
    let raw = std::fs::read(&path).expect("read raw");
    cleanup(&path);

    result.expect("overwrite must succeed");
    assert_eq!(
        values,
        vec![1.0, 2.0, 4.0, 8.0],
        "decoded through the dtype"
    );
    // 1.0f32 big-endian is 3F 80 00 00; little-endian would be 00 00 80 3F.
    let first = &raw[extent.address as usize..extent.address as usize + 4];
    assert_eq!(
        first,
        &1.0f32.to_be_bytes(),
        "bytes must be stored big-endian, got {first:?}"
    );
}

/// The contract, stated at the byte level, on a libhdf5-authored file: the file
/// before and after differs *only* inside the target dataset's data range.
#[test]
fn in_place_overwrite_disturbs_only_the_target_data_range() {
    let Some(path) = fixture_copy("i4_2d_contig.h5", "byteproof") else {
        return;
    };

    let extent = oxih5::dataset_data_extent(&path, "matrix").expect("extent");
    let before = std::fs::read(&path).expect("read before");
    let result = oxih5::write_dataset_in_place_i32(&path, "matrix", &[-1, -2, -3, -4, -5, -6]);
    let after = std::fs::read(&path).expect("read after");
    cleanup(&path);

    result.expect("overwrite must succeed");
    assert_eq!(before.len(), after.len(), "file size must not change");
    assert_eq!(extent.size, 24, "2×3 int32");

    let lo = extent.address as usize;
    let hi = lo + extent.size as usize;
    let outside: Vec<usize> = (0..before.len())
        .filter(|&i| before[i] != after[i])
        .filter(|&i| i < lo || i >= hi)
        .collect();
    assert!(
        outside.is_empty(),
        "bytes outside [{lo}, {hi}) changed at offsets {outside:?}"
    );
    assert_ne!(
        &before[lo..hi],
        &after[lo..hi],
        "the data range itself must have changed"
    );
}

// ---------------------------------------------------------------------------
// Rejections that need a real libhdf5 file to construct
// ---------------------------------------------------------------------------

/// A gzip-compressed dataset cannot be overwritten with raw bytes: the on-disk
/// bytes are a compressed encoding whose length is not a function of the value
/// count.  The file must come back untouched.
#[test]
fn in_place_rejects_filtered_dataset() {
    let Some(path) = fixture_copy("chunked_gzip_f4_1d.h5", "gzip") else {
        return;
    };

    let before = std::fs::read(&path).expect("read before");
    let err = oxih5::write_dataset_in_place(&path, "data", &[0u8; 400]).expect_err("must reject");
    let after = std::fs::read(&path).expect("read after");
    cleanup(&path);

    match err {
        oxih5::OxiH5Error::NotImplemented(msg) => {
            assert!(
                msg.contains("filter pipeline"),
                "must name the obstacle: {msg}"
            );
            assert!(
                msg.contains("filter ids 1"),
                "must name gzip's filter id 1: {msg}"
            );
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    assert_eq!(before, after, "a rejected write must not touch the file");
}

/// A virtual dataset has no data of its own — its bytes live in other files.
#[test]
fn in_place_rejects_virtual_dataset() {
    let Some(path) = fixture_copy("vds_simple.h5", "vds") else {
        return;
    };

    let before = std::fs::read(&path).expect("read before");
    let err = oxih5::write_dataset_in_place(&path, "virt", &[0u8; 48]).expect_err("must reject");
    let after = std::fs::read(&path).expect("read after");
    cleanup(&path);

    match err {
        oxih5::OxiH5Error::NotImplemented(msg) => assert!(
            msg.contains("virtual"),
            "must name the virtual layout: {msg}"
        ),
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    assert_eq!(before, after, "a rejected write must not touch the file");
}

/// A chunked (but unfiltered) dataset is still not a flat run of bytes.
#[test]
fn in_place_rejects_chunked_dataset() {
    let Some(path) = fixture_copy("chunked_i4_1d.h5", "chunked") else {
        return;
    };

    let before = std::fs::read(&path).expect("read before");
    let err = oxih5::write_dataset_in_place_i32(&path, "data", &[0i32; 12]).expect_err("reject");
    let extent_err = oxih5::dataset_data_extent(&path, "data").expect_err("extent must refuse too");
    let after = std::fs::read(&path).expect("read after");
    cleanup(&path);

    for e in [err, extent_err] {
        match e {
            oxih5::OxiH5Error::NotImplemented(msg) => {
                assert!(msg.contains("chunked"), "must name the layout: {msg}");
                assert!(msg.contains("class 2"), "must name the layout class: {msg}");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }
    assert_eq!(before, after, "a rejected write must not touch the file");
}

/// Supplying the wrong number of bytes is refused, and the message names both
/// lengths so the caller can see the discrepancy.
#[test]
fn in_place_rejects_length_mismatch() {
    let Some(path) = fixture_copy("f8_1d_contig.h5", "lenmismatch") else {
        return;
    };

    let before = std::fs::read(&path).expect("read before");
    // 'signal' is 4 × f64 = 32 bytes.
    let err = oxih5::write_dataset_in_place(&path, "signal", &[0u8; 40]).expect_err("must reject");
    let after = std::fs::read(&path).expect("read after");
    cleanup(&path);

    match err {
        oxih5::OxiH5Error::Format(msg) => {
            assert!(msg.contains("32"), "must name the allocated size: {msg}");
            assert!(msg.contains("40"), "must name the supplied size: {msg}");
        }
        other => panic!("expected Format, got {other:?}"),
    }
    assert_eq!(before, after, "a rejected write must not touch the file");
}

/// A path that resolves to a group has no data area to overwrite.
#[test]
fn in_place_rejects_group_path() {
    let Some(path) = fixture_copy("nested_groups.h5", "grouppath") else {
        return;
    };

    let err = oxih5::write_dataset_in_place(&path, "/sensors/imu", &[0u8; 12]).expect_err("reject");
    cleanup(&path);

    assert!(
        matches!(err, oxih5::OxiH5Error::TypeMismatch),
        "expected TypeMismatch for a group path, got {err:?}"
    );
}

/// A fixed-length string dataset *is* a flat run of bytes and may be
/// overwritten; a vlen-string dataset is not.  This pins the former, which is
/// the case MATLAB `char` arrays hit.
#[test]
fn in_place_overwrites_fixed_length_string_dataset() {
    let Some(path) = fixture_copy("string_fixed_1d.h5", "fixedstr") else {
        return;
    };

    let extent = oxih5::dataset_data_extent(&path, "labels").expect("extent");
    // 2 elements × 10 bytes: pad each label to the fixed width with NULs.
    let mut bytes = vec![0u8; extent.size as usize];
    bytes[..5].copy_from_slice(b"alpha");
    bytes[10..14].copy_from_slice(b"beta");
    let result = oxih5::write_dataset_in_place(&path, "labels", &bytes);

    let strings = oxih5::open(&path)
        .expect("reopen")
        .dataset("labels")
        .expect("labels")
        .as_string();
    cleanup(&path);

    result.expect("overwrite must succeed");
    assert_eq!(extent.size, 20, "2 × 10-byte fixed strings");
    let strings = strings.expect("decode strings");
    assert_eq!(strings, vec!["alpha".to_string(), "beta".to_string()]);
}

// ---------------------------------------------------------------------------
// Link resolution
// ---------------------------------------------------------------------------

/// Resolving an overwrite target follows soft links, and reports an external
/// link for what it is.
///
/// `soft_ext_main.h5` chains `soft` → `ext` → a dataset in `soft_ext_target.h5`.
/// The soft hop is followed (otherwise the error would be about the *link type*
/// rather than about the external target it leads to), and the external hop is
/// refused because a dataset in another file has no address in this one — which
/// is exactly what an in-place overwrite needs.
#[test]
fn in_place_follows_soft_links_and_names_external_links() {
    let Some(path) = fixture_copy("soft_ext_main.h5", "links") else {
        return;
    };

    let via_soft = oxih5::dataset_data_extent(&path, "soft");
    let via_ext = oxih5::dataset_data_extent(&path, "ext");
    cleanup(&path);

    match via_soft {
        Err(oxih5::OxiH5Error::NotImplemented(msg)) => assert!(
            msg.contains("external"),
            "the soft hop must be followed through to its external target, got: {msg}"
        ),
        other => panic!("expected NotImplemented naming the external target, got {other:?}"),
    }
    match via_ext {
        Err(oxih5::OxiH5Error::NotImplemented(msg)) => {
            assert!(
                msg.contains("external link"),
                "must name the link kind: {msg}"
            );
            assert!(
                msg.contains("soft_ext_target.h5"),
                "must name the target file: {msg}"
            );
        }
        other => panic!("expected NotImplemented for an external link, got {other:?}"),
    }
}
