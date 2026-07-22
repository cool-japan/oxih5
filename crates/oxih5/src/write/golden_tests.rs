//! Byte-exact and structural regression guards for the writer.
//!
//! The W1x golden-bytes test pins the entire output of a feature-rich build
//! through [`W1X_GOLDEN_HASH`]; a pure refactor must leave that constant
//! untouched.  The remaining tests are the W0/W1 round-trip and alignment
//! guards that used to live inline in [`super`].

use super::*;
use crate::File;
use oxih5_core::Dtype;

#[test]
fn write_string_attr_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_write_str_attr.h5");
    let mut w = FileWriter::new();
    w.write_dataset_f64("x", &[1.0, 2.0, 3.0, 4.0], &[4])
        .expect("write");
    w.write_string_attr("x", "units", "km").expect("attr");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let attrs = f.attr_views("x").expect("attr_views");
    let a = attrs.iter().find(|a| a.name() == "units").expect("units");
    assert_eq!(a.as_str_fixed().expect("as_str_fixed"), "km");
}

#[test]
fn write_f64_attr_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_write_f64_attr.h5");
    let mut w = FileWriter::new();
    w.write_dataset_f32("data", &[0.0f32; 4], &[4])
        .expect("write");
    w.write_f64_attr("data", "scale_factor", 2.5).expect("attr");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let attrs = f.attr_views("data").expect("attr_views");
    let sf = attrs
        .iter()
        .find(|a| a.name() == "scale_factor")
        .expect("sf");
    let v = sf.as_f64().expect("as_f64");
    assert!((v - 2.5).abs() < 1e-15, "expected 2.5 got {v}");
}

#[test]
fn write_i32_attr_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_write_i32_attr.h5");
    let mut w = FileWriter::new();
    w.write_dataset_i32("ds", &[1i32, 2, 3], &[3])
        .expect("write");
    w.write_i32_attr("ds", "_Netcdf4Dimid", 7).expect("attr");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let attrs = f.attr_views("ds").expect("attr_views");
    let a = attrs
        .iter()
        .find(|a| a.name() == "_Netcdf4Dimid")
        .expect("dimid");
    assert_eq!(a.as_i64(), Some(7));
}

#[test]
fn write_i64_attr_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_write_i64_attr.h5");
    let mut w = FileWriter::new();
    w.write_dataset_f64("big", &[0.0; 2], &[2]).expect("write");
    w.write_i64_attr("big", "count", i64::MAX).expect("attr");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let attrs = f.attr_views("big").expect("attr_views");
    let a = attrs.iter().find(|a| a.name() == "count").expect("count");
    assert_eq!(a.as_i64(), Some(i64::MAX));
}

#[test]
fn write_multiple_attrs_on_same_dataset() {
    let tmp = std::env::temp_dir().join("oxih5_test_multi_attr.h5");
    let mut w = FileWriter::new();
    w.write_dataset_f64("temp", &[20.0, 21.0], &[2])
        .expect("write");
    w.write_string_attr("temp", "units", "degC").expect("units");
    w.write_string_attr("temp", "long_name", "Surface Temperature")
        .expect("long_name");
    w.write_f64_attr("temp", "_FillValue", 9.969_209_968_386_869e36)
        .expect("fillval");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let attrs = f.attr_views("temp").expect("attr_views");
    assert!(attrs.iter().any(|a| a.name() == "units"));
    assert!(attrs.iter().any(|a| a.name() == "long_name"));
    assert!(attrs.iter().any(|a| a.name() == "_FillValue"));
}

#[test]
fn write_obj_ref_list_attr_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_objref_attr.h5");
    let mut w = FileWriter::new();
    w.write_dataset_i32("lat", &[0i32, 1, 2], &[3])
        .expect("lat");
    w.write_string_attr("lat", "CLASS", "DIMENSION_SCALE")
        .expect("CLASS");
    w.write_dataset_f64("temp", &[0.0; 3], &[3]).expect("temp");
    w.write_obj_ref_list_attr("temp", "DIMENSION_LIST", &["lat"])
        .expect("dim_list");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let lat_addr = f.header_addr_of("lat").expect("lat addr");
    let attrs = f.attr_views("temp").expect("attr_views temp");
    let dl = attrs
        .iter()
        .find(|a| a.name() == "DIMENSION_LIST")
        .expect("dl");
    let refs = dl.as_object_refs().expect("refs");
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0], lat_addr);
}

#[test]
fn attr_on_unknown_dataset_returns_error() {
    let mut w = FileWriter::new();
    let result = w.write_string_attr("nonexistent", "key", "val");
    assert!(result.is_err());
}

#[test]
fn nc_file_writer_simulation() {
    let tmp = std::env::temp_dir().join("oxih5_test_ncfw_sim.h5");
    let mut w = FileWriter::new();
    w.write_dataset_i32("lat", &[0i32, 1, 2, 3], &[4]).unwrap();
    w.write_string_attr("lat", "CLASS", "DIMENSION_SCALE")
        .unwrap();
    w.write_string_attr("lat", "NAME", "lat").unwrap();
    w.write_i32_attr("lat", "_Netcdf4Dimid", 0).unwrap();
    w.write_dataset_i32("lon", &[0i32, 1, 2, 3, 4, 5, 6, 7], &[8])
        .unwrap();
    w.write_string_attr("lon", "CLASS", "DIMENSION_SCALE")
        .unwrap();
    w.write_string_attr("lon", "NAME", "lon").unwrap();
    w.write_i32_attr("lon", "_Netcdf4Dimid", 1).unwrap();
    w.write_dataset_f64("temp", &[0.0f64; 32], &[4, 8]).unwrap();
    w.write_obj_ref_list_attr("temp", "DIMENSION_LIST", &["lat", "lon"])
        .unwrap();
    w.build(&tmp).unwrap();
    let f = File::open(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    let lat_addr = f.header_addr_of("lat").unwrap();
    let lon_addr = f.header_addr_of("lon").unwrap();
    let lat_attrs = f.attr_views("lat").unwrap();
    let class_attr = lat_attrs.iter().find(|a| a.name() == "CLASS");
    assert!(class_attr.is_some());
    assert_eq!(
        class_attr
            .unwrap()
            .as_str_fixed()
            .unwrap_or_default()
            .trim(),
        "DIMENSION_SCALE"
    );
    let temp_attrs = f.attr_views("temp").unwrap();
    let dl = temp_attrs.iter().find(|a| a.name() == "DIMENSION_LIST");
    assert!(dl.is_some());
    let refs = dl.unwrap().as_object_refs().expect("refs");
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0], lat_addr);
    assert_eq!(refs[1], lon_addr);
}

/// There is no root-item cap any more.
///
/// This test used to be named for a 64-item ceiling that the writer
/// enforced because it emitted exactly one symbol table node per group.
/// The B-tree now grows instead — SNODs chain, then levels stack — so the
/// 65th, 100th and 300th item are all ordinary.
#[test]
fn root_items_are_not_capped_at_a_single_snod() {
    let tmp = std::env::temp_dir().join("oxih5_test_no_root_cap.h5");
    let mut w = FileWriter::new();
    for i in 0..100usize {
        w.write_dataset_f64(&format!("ds{i:03}"), &[i as f64], &[1])
            .expect("write");
    }
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    let names = f.dataset_names().expect("names");
    assert_eq!(names.len(), 100);
}

// -----------------------------------------------------------------------
// C11 infrastructure: root group string attributes
// -----------------------------------------------------------------------

#[test]
fn root_str_attr_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_root_str_attr.h5");
    let mut w = FileWriter::new();
    w.write_root_str_attr("_nc3_strict", "");
    w.write_dataset_f64("data", &[1.0, 2.0], &[2]).unwrap();
    w.build(&tmp).unwrap();

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    // Root group attrs via root().attr_views()
    let root = f.root().expect("root");
    let attrs = root.attr_views().expect("root_attr_views");
    let nc3 = attrs.iter().find(|a| a.name() == "_nc3_strict");
    assert!(nc3.is_some(), "_nc3_strict not found on root group");
}

#[test]
fn root_str_attr_does_not_break_dataset_reads() {
    let tmp = std::env::temp_dir().join("oxih5_test_root_attr_ds.h5");
    let mut w = FileWriter::new();
    w.write_root_str_attr("convention", "CF-1.8");
    w.write_dataset_f32("pressure", &[101.3f32, 99.8], &[2])
        .unwrap();
    w.write_string_attr("pressure", "units", "hPa").unwrap();
    w.build(&tmp).unwrap();

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let ds = f.dataset("pressure").expect("pressure");
    assert_eq!(ds.shape, vec![2usize]);
    let attrs = f.attr_views("pressure").expect("pressure attrs");
    assert!(attrs.iter().any(|a| a.name() == "units"));
}

// -----------------------------------------------------------------------
// W0b: Sub-group round-trip
// -----------------------------------------------------------------------

#[test]
fn w0b_create_group_and_dataset_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0b_group.h5");
    let mut w = FileWriter::new();

    // Root dataset
    w.write_dataset_f64("root_ds", &[1.0, 2.0], &[2]).unwrap();

    // Sub-group with one dataset
    w.create_group("sensors").unwrap();
    w.write_group_dataset_f64("sensors", "temperature", &[22.5, 23.0, 21.8], &[3])
        .unwrap();
    w.write_group_string_attr("sensors", "temperature", "units", "degC")
        .unwrap();

    w.build(&tmp).unwrap();

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    // Root dataset should still be readable
    let root_ds = f.dataset("root_ds").expect("root_ds");
    assert_eq!(root_ds.shape, vec![2usize]);

    // Navigate to sub-group
    let grp = f.group("sensors").expect("sensors group");
    let names = grp.datasets().expect("group datasets");
    assert!(
        names.iter().any(|n| n == "temperature"),
        "temperature not in sensors: {names:?}"
    );

    // Read dataset via path
    let temp = f
        .dataset("sensors/temperature")
        .expect("sensors/temperature");
    assert_eq!(temp.shape, vec![3usize]);
}

#[test]
fn w0b_group_groups_listing() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0b_groups.h5");
    let mut w = FileWriter::new();
    w.create_group("grp1").unwrap();
    w.create_group("grp2").unwrap();
    w.write_group_dataset_i32("grp1", "x", &[1i32, 2], &[2])
        .unwrap();
    w.build(&tmp).unwrap();

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let root = f.root().expect("root");
    let group_names = root.groups().expect("root groups");
    assert!(
        group_names.iter().any(|n| n == "grp1"),
        "grp1 missing: {group_names:?}"
    );
    assert!(
        group_names.iter().any(|n| n == "grp2"),
        "grp2 missing: {group_names:?}"
    );
}

// -----------------------------------------------------------------------
// W0c: Unlimited / chunked dataset round-trip
// -----------------------------------------------------------------------

#[test]
fn w0c_unlimited_dataset_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0c_unlimited.h5");

    let data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dtype = Dtype::Float {
        size: 8,
        order: oxih5_core::ByteOrder::Little,
    };
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("time_series", &[10], &[10], &dtype, &raw)
        .unwrap();
    w.build(&tmp).unwrap();

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let ds = f.dataset("time_series").expect("time_series");
    assert_eq!(ds.shape, vec![10usize]);
    assert!(ds.is_unlimited(), "expected unlimited dim 0");

    let vals = ds.as_f64().expect("as_f64");
    assert_eq!(vals.len(), 10);
    for (i, &v) in vals.iter().enumerate() {
        assert!((v - i as f64 * 0.5).abs() < 1e-15, "mismatch at {i}: {v}");
    }
}

#[test]
fn w0c_2d_unlimited_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0c_2d_unlimited.h5");

    // Shape: [3, 4] — unlimited on dim 0
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dtype = Dtype::Float {
        size: 4,
        order: oxih5_core::ByteOrder::Little,
    };
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("grid", &[3, 4], &[3, 4], &dtype, &raw)
        .unwrap();
    w.build(&tmp).unwrap();

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let ds = f.dataset("grid").expect("grid");
    assert_eq!(ds.shape, vec![3usize, 4]);
    assert!(ds.is_unlimited(), "expected unlimited");
}

// -----------------------------------------------------------------------
// W0d: GlobalHeap / vlen-string dataset round-trip
// -----------------------------------------------------------------------

#[test]
fn w0d_gcol_round_trip() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0d_gcol.h5");
    let mut w = FileWriter::new();
    w.create_vlen_string_dataset("strs", &["hello", "world", "foo"])
        .expect("create_vlen_string_dataset");
    w.build(&tmp).expect("build");

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let result = f.dataset_strings("strs").expect("dataset_strings");
    assert_eq!(result, vec!["hello", "world", "foo"]);
}

#[test]
fn w0d_gcol_empty_string() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0d_empty_str.h5");
    let mut w = FileWriter::new();
    w.create_vlen_string_dataset("s", &["", "non-empty", ""])
        .expect("create_vlen_string_dataset");
    w.build(&tmp).expect("build");

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let result = f.dataset_strings("s").expect("dataset_strings");
    assert_eq!(result, vec!["", "non-empty", ""]);
}

#[test]
fn w0d_gcol_with_coexisting_numeric_dataset() {
    let tmp = std::env::temp_dir().join("oxih5_test_w0d_mixed.h5");
    let mut w = FileWriter::new();
    w.write_dataset_f64("nums", &[1.0, 2.0, 3.0], &[3])
        .expect("nums");
    w.create_vlen_string_dataset("labels", &["alpha", "beta", "gamma"])
        .expect("labels");
    w.build(&tmp).expect("build");

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    // Numeric dataset still readable.
    let nums = f.dataset("nums").expect("nums");
    let vals = nums.as_f64().expect("as_f64");
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);

    // Vlen string dataset readable.
    let labels = f.dataset_strings("labels").expect("labels");
    assert_eq!(labels, vec!["alpha", "beta", "gamma"]);
}

// -----------------------------------------------------------------------
// W1x: golden-bytes guard for the writer refactor
// -----------------------------------------------------------------------

/// FNV-1a fingerprint of [`w1x_fixture_bytes`].
///
/// This is a behaviour oracle, not a style preference: it must only ever be
/// updated together with a deliberate, documented change to the bytes the
/// writer emits.  A pure refactor must leave it untouched.
///
/// History:
/// * `0x0f0c_074d_1aec_f822` — baseline, and unchanged across the
///   elem/oh/size-contract refactor, which is what proved that refactor
///   byte-for-byte behaviour-preserving.
/// * `0xda25_bb31_0978_19c7` — object-header messages now declare their
///   *padded* body size (see [`super::oh::write_oh`]).  libhdf5 rejects a
///   whole object-header chunk with "message not aligned" otherwise, which
///   silently lost every dataset carrying an i32 or odd-length string
///   attribute and made a file with an odd-length root string attribute
///   completely unopenable.  The file *length* is unchanged: the padding was
///   always reserved, only the declared size was wrong.
/// * `0x3a9a_d435_7143_2277` — symbol-table conformance (see
///   [`super::btree_v1`]).  Three things moved at once, all of them
///   deliberate.  Links are now **sorted by name**, so the local heap, the
///   symbol table entries and every address that follows them are laid out
///   in a different order; without this, `H5G__node_found`'s binary search
///   silently lost datasets that were declared out of order.  Symbol table
///   nodes are now **chunked at eight entries** and every node — root and
///   sub-group alike — is 328 bytes, the one width the superblock's
///   `leaf_node_K = 4` describes; the fixture's ten root links therefore
///   span two SNODs where they used to sit in one oversized node.  And the
///   group B-tree is now a full-width 544-byte node instead of a 48-byte
///   single-child leaf.  The file is *longer* as a result.
/// * `0x199c_ba54_0314_ee2a` — **the fixture grew; the layout did not.**
///   Nested groups, group attributes and array-valued attributes landed
///   together with the collapse of `GroupDesc` and the root's ad-hoc
///   `root_str_attrs` into one recursive [`tree::GroupNode`], planned by one
///   recursive [`plan::plan_group`].  That restructure was verified
///   byte-for-byte neutral first: the *unchanged* fixture still hashed
///   `0x3a9a_d435_7143_2277` afterwards, because a root group planned "like
///   any other group" happens to lay out in exactly the order the
///   hand-written root path used.  The constant then moved only because the
///   fixture itself was extended, to keep the new capabilities inside the
///   oracle: a three-level nested group, attributes on two different
///   sub-groups, non-string attributes on the root group, and one attribute
///   of each array kind.
/// * `0xa273_56da_84d2_67f6` — **chunk B-tree conformance** (see
///   [`super::chunked`]).  The fixture is untouched; two of its datasets are
///   chunked, and each one's index node grew from a node sized for its
///   contents to the fixed width libhdf5 reads: 80 → 2096 bytes for `time`
///   (1-D) and 96 → 2616 for `counts` (2-D), so the file went from 6968 to
///   11504 bytes.  Nothing else moved — the layout message, the data area
///   and every address before the first chunked dataset are byte-identical.
///   Before this, libhdf5 refused the file outright with `addr overflow,
///   addr = 3000, size = 2096, eoa = 3112`: it asks the file for a node
///   image sized from a *compile-time* `K` of 32 and ran off the end of a
///   node we had sized at 80 bytes.  The terminal key also stopped being
///   all-zero, which is what `H5D__btree_cmp3` needs to find a chunk at all.
/// * `0x8a1e_d680_49db_52d2` — **write-path hardening: B001-B024/R001-R010
///   byte-level fixes.**  A single convergence over seven fix lanes moved
///   the bytes at many object headers at once, all deliberate and each
///   proven against h5py 3.16 / libhdf5 2.0.0: global-heap collections are
///   padded to `H5HG_MINSIZE` (4096) with a real free-space object instead
///   of a size-0 NIL terminator (B001/B002/B008/B009); fixed-length strings
///   now emit `strpad = NULLPAD` and vlen strings carry the UTF-8 charset
///   bit, and an empty scalar string no longer emits a size-0 datatype
///   (B017/B022/B003); chunked datasets declare an alloc-time *incremental*
///   fill-value message where contiguous ones stay *late* (B023); the
///   terminal chunk B-tree key matches libhdf5's insertion replay (B016).
///   The fixture is untouched and its length is unchanged at 15504 bytes —
///   only the values inside the moved messages differ.  (The empty-dataset
///   undefined-address fix, B004, does not touch this fixture, whose
///   datasets are all non-empty.)
/// * `0x7ee9_7cf0_fce7_ebc1` — **Wave 2: the writer's new features folded
///   into the fixture.**  The fixture grew a `w2` sub-group (length now
///   24240 bytes) that exercises every Wave-2 writer path end to end, each
///   verified against h5py 3.16 / netCDF4 1.7.4: a netCDF-4 dimension-scale
///   trio wired with a NULLTERM `CLASS` string attribute (`H5T_STR_NULLTERM`,
///   the padding libnetcdf's `H5DSis_scale` requires), a compound
///   `REFERENCE_LIST` `{ dataset, dimension }`, a vlen-of-object-reference
///   `DIMENSION_LIST` (heap payloads placed by the build-path hook), and an
///   **int32** `_Netcdf4Coordinates` (netCDF reads it with `H5T_NATIVE_INT`,
///   so an int64 `[0, 1]` would decode as `[0, 0]` and collapse the axes);
///   the widened numeric attribute family (`f32`/`i8`/`i16`/`u8`/`u16`/`u32`/
///   `u64`, scalar and array); a boolean enum dataset, a fixed-length string
///   dataset, a custom fill value, and the compact (inline) layout; and the
///   two pure-oxih5 filters SHUFFLE and FLETCHER32.  Every address before the
///   `w2` group is byte-identical to the previous fixture — the earlier
///   objects were not touched.
///
/// Deliberately **not** in the fixture: a DEFLATE-compressed dataset.  The
/// compressed bytes come out of `oxiarc-deflate`, so folding them in here
/// would make this oracle fire on an upstream encoder improvement — a
/// false alarm that teaches the reader to update the constant without
/// reading it.  Compression is pinned instead where it is actually ours:
/// the pipeline message body in [`super::pipeline`], the B-tree node in
/// [`super::chunked`], and the observable behaviour in the h5py interop
/// tests.
const W1X_GOLDEN_HASH: u64 = 0x7ee9_7cf0_fce7_ebc1;

/// 64-bit FNV-1a hash — a dependency-free byte-stream fingerprint.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Build the canonical feature-rich fixture shared by the W1x guard tests.
///
/// Exercises, in one file: all five element types, the `Dtype`-driven
/// zero-filled path, a vlen-string dataset (global heap collection), 1-D and
/// 2-D chunked/unlimited datasets, every supported attribute kind including
/// the three array kinds, a three-level nested group hierarchy, attributes
/// on sub-groups, and string *and* non-string root-group attributes.
///
/// The `w2` sub-group added in Wave 2 further exercises the reference-bearing
/// attributes (NULLTERM `CLASS`, compound `REFERENCE_LIST`, vlen-objref
/// `DIMENSION_LIST`, int32 `_Netcdf4Coordinates`), the widened numeric
/// attribute family, boolean/fixed-string datasets, a custom fill value, the
/// compact layout, and the SHUFFLE and FLETCHER32 filters — see the
/// [`W1X_GOLDEN_HASH`] history.
fn w1x_fixture_bytes() -> Vec<u8> {
    let le = oxih5_core::ByteOrder::Little;
    let dt_f64 = Dtype::Float { size: 8, order: le };
    let dt_i32 = Dtype::Int {
        size: 4,
        signed: true,
        order: le,
    };

    let mut w = FileWriter::new();

    // Root group attributes: strings, non-strings, and an array.
    w.write_root_str_attr("Conventions", "CF-1.8");
    w.write_root_str_attr("_nc3_strict", "");
    w.write_i64_attr("/", "total_records", 9_000_000_000)
        .expect("total_records");
    w.write_f64_attr("/", "resolution", 0.125)
        .expect("resolution");
    w.write_i32_attr("/", "revision", -7).expect("revision");
    w.write_string_array_attr("/", "sources", &["buoy", "satellite"])
        .expect("sources");

    // One root dataset per supported element type.
    w.write_dataset_f32("f32_ds", &[1.5f32, -2.25, 3.0, 4.75], &[4])
        .expect("f32_ds");
    w.write_dataset_f64("f64_ds", &[0.5f64, 1.5, 2.5, 3.5, 4.5, 5.5], &[2, 3])
        .expect("f64_ds");
    w.write_dataset_i32("i32_ds", &[-1i32, 0, 1], &[3])
        .expect("i32_ds");
    w.write_dataset_i64("i64_ds", &[i64::MIN, 0, i64::MAX], &[3])
        .expect("i64_ds");
    w.write_dataset_u8("u8_ds", &[0u8, 127, 255], &[3])
        .expect("u8_ds");

    // Zero-filled dataset via the `Dtype` path.
    w.create_dataset("zeros", &[2, 2], &Dtype::Float { size: 4, order: le })
        .expect("zeros");

    // Vlen-string dataset (global heap collection), including an empty entry.
    w.create_vlen_string_dataset("labels", &["alpha", "", "gamma"])
        .expect("labels");

    // Chunked / unlimited datasets, 1-D and 2-D.
    let raw_time: Vec<u8> = (0..4).flat_map(|i| f64::from(i).to_le_bytes()).collect();
    w.create_dataset_unlimited("time", &[4], &[4], &dt_f64, &raw_time)
        .expect("time");
    let raw_counts: Vec<u8> = (0..6i32).flat_map(i32::to_le_bytes).collect();
    w.create_dataset_unlimited("counts", &[2, 3], &[2, 3], &dt_i32, &raw_counts)
        .expect("counts");

    // Every supported attribute kind.
    w.write_string_attr("f32_ds", "units", "m s-1")
        .expect("units");
    w.write_f64_attr("f32_ds", "scale_factor", 0.125)
        .expect("scale_factor");
    w.write_i64_attr("f32_ds", "valid_max", 1_000_000)
        .expect("valid_max");
    w.write_i32_attr("f32_ds", "_Netcdf4Dimid", 3)
        .expect("_Netcdf4Dimid");
    w.write_string_attr("i32_ds", "CLASS", "DIMENSION_SCALE")
        .expect("CLASS");
    w.write_obj_ref_list_attr("f32_ds", "DIMENSION_LIST", &["i32_ds", "i64_ds"])
        .expect("DIMENSION_LIST");

    // Array-valued attributes, one of each kind.
    w.write_i64_array_attr("i64_ds", "valid_range", &[i64::MIN, i64::MAX])
        .expect("valid_range");
    w.write_f64_array_attr("f64_ds", "bounds", &[-1.5, 0.0, 2.25])
        .expect("bounds");
    w.write_string_array_attr("u8_ds", "flag_meanings", &["low", "high"])
        .expect("flag_meanings");

    // Sub-group with datasets and a dataset attribute.
    w.create_group("grp").expect("grp");
    w.write_group_dataset_f64("grp", "gf64", &[10.0, 20.0], &[2])
        .expect("gf64");
    w.write_group_dataset_i32("grp", "gi32", &[7i32, 8, 9, 10], &[2, 2])
        .expect("gi32");
    w.write_group_string_attr("grp", "gf64", "units", "K")
        .expect("group units");

    // Attributes on the sub-group itself, and a three-level hierarchy whose
    // intermediate group is created implicitly by the dataset path.
    w.write_string_attr("grp", "title", "instrument group")
        .expect("group title");
    w.write_i32_attr("grp", "_Netcdf4Dimid", 11)
        .expect("group dimid");
    w.write_dataset_f64("grp/sub/deep", &[3.5, 4.5], &[2])
        .expect("deep");
    w.write_f64_attr("grp/sub", "scale", 0.5)
        .expect("sub scale");
    w.write_obj_ref_list_attr("grp/sub/deep", "SOURCE", &["/grp/gf64"])
        .expect("cross-group object reference");

    // -- Wave 2 feature coverage -------------------------------------------
    // Everything Wave 2 added to the writer, folded in under one sub-group so
    // the fixture fingerprints those bytes too.  DEFLATE is deliberately still
    // absent (its bytes come from `oxiarc-deflate`, upstream — see the
    // `W1X_GOLDEN_HASH` note); the two filters here are the ones that are
    // ours end to end: SHUFFLE (a byte permutation) and FLETCHER32 (our
    // checksum).
    w.create_group("w2").expect("w2");

    // A netCDF-4-style dimension-scale trio wired with the reference-bearing
    // attributes: NULLTERM `CLASS`, compound `REFERENCE_LIST`, vlen-objref
    // `DIMENSION_LIST`, and an int32 `_Netcdf4Coordinates`.
    w.write_dataset_i32("w2/x", &[0, 1, 2], &[3]).expect("w2/x");
    w.write_dataset_i32("w2/y", &[0, 1, 2, 3], &[4])
        .expect("w2/y");
    w.write_dataset_f64("w2/grid", &[0.0; 12], &[3, 4])
        .expect("w2/grid");
    w.write_string_attr_nullterm("w2/x", "CLASS", "DIMENSION_SCALE")
        .expect("x CLASS");
    w.write_string_attr_nullterm("w2/y", "CLASS", "DIMENSION_SCALE")
        .expect("y CLASS");
    w.write_ref_index_list_attr("w2/x", "REFERENCE_LIST", &[("w2/grid".to_string(), 0)])
        .expect("x REFERENCE_LIST");
    w.write_ref_index_list_attr("w2/y", "REFERENCE_LIST", &[("w2/grid".to_string(), 1)])
        .expect("y REFERENCE_LIST");
    w.write_vlen_obj_ref_attr(
        "w2/grid",
        "DIMENSION_LIST",
        &[vec!["w2/x".to_string()], vec!["w2/y".to_string()]],
    )
    .expect("grid DIMENSION_LIST");
    w.write_i32_array_attr("w2/grid", "_Netcdf4Coordinates", &[0, 1])
        .expect("grid _Netcdf4Coordinates");

    // The widened numeric attribute family (scalars and arrays).
    w.write_f32_attr("w2/x", "f32s", 1.5).expect("f32s");
    w.write_i8_attr("w2/x", "i8s", -8).expect("i8s");
    w.write_i16_attr("w2/x", "i16s", -1600).expect("i16s");
    w.write_u8_attr("w2/x", "u8s", 250).expect("u8s");
    w.write_u16_attr("w2/x", "u16s", 60_000).expect("u16s");
    w.write_u32_attr("w2/x", "u32s", 4_000_000_000)
        .expect("u32s");
    w.write_u64_attr("w2/x", "u64s", 18_000_000_000_000_000_000)
        .expect("u64s");
    w.write_f32_array_attr("w2/y", "f32a", &[1.0, 2.0])
        .expect("f32a");
    w.write_u8_array_attr("w2/y", "u8a", &[1, 2, 3])
        .expect("u8a");

    // Dataset-feature coverage: bool enum, fixed-length string, custom fill
    // value (contiguous), and the compact (inline) layout.
    w.write_dataset_bool("w2/flags", &[true, false, true, true])
        .expect("w2/flags");
    w.create_fixed_string_dataset("w2/names", 8, &["north", "south"])
        .expect("w2/names");
    w.write_dataset_f64("w2/filled", &[1.0, 2.0, 3.0, 4.0], &[4])
        .expect("w2/filled");
    w.set_fill_value_f64("w2/filled", -999.0)
        .expect("w2/filled fill value");
    w.write_dataset_i32("w2/inline", &[10, 20, 30], &[3])
        .expect("w2/inline");
    w.set_compact("w2/inline").expect("w2/inline compact");

    // Filter coverage (pure-oxih5 only): a shuffled dataset and a
    // fletcher32-checksummed one.
    w.write_dataset_i32("w2/shuffled", &(0..64).collect::<Vec<_>>(), &[64])
        .expect("w2/shuffled");
    w.set_shuffle("w2/shuffled").expect("set_shuffle");
    w.write_dataset_f64("w2/checked", &[0.5; 32], &[32])
        .expect("w2/checked");
    w.set_fletcher32("w2/checked").expect("set_fletcher32");

    w.build_to_vec().expect("build_to_vec")
}

/// Byte-exact regression guard for the writer.
///
/// The writer computes every absolute address in a first pass and writes at
/// those addresses in a second pass, so a one-byte disagreement between an
/// "allocate" and a "write" formula silently corrupts the file.  This hash
/// pins the entire output of a feature-rich build; any change to it means a
/// behavioural difference, not just a code reshuffle.
#[test]
fn w1x_golden_bytes_multi_feature() {
    let bytes = w1x_fixture_bytes();
    assert_eq!(
        fnv1a(&bytes),
        W1X_GOLDEN_HASH,
        "writer byte output changed (len = {})",
        bytes.len()
    );

    // A stable hash over a corrupt file would be a useless oracle, so also
    // assert the fixture still parses and reads back correctly.
    let tmp = std::env::temp_dir().join("oxih5_test_w1x_golden.h5");
    std::fs::write(&tmp, &bytes).expect("write fixture");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(
        f.dataset("f64_ds").expect("f64_ds").shape,
        vec![2usize, 3],
        "f64_ds shape"
    );
    assert!(
        f.dataset("time").expect("time").is_unlimited(),
        "time must be unlimited"
    );
    assert_eq!(
        f.dataset_strings("labels").expect("labels"),
        vec!["alpha", "", "gamma"]
    );
    let attrs = f.attr_views("f32_ds").expect("f32_ds attrs");
    assert_eq!(attrs.len(), 5, "f32_ds attribute count");
    assert_eq!(
        f.dataset("grp/gi32").expect("grp/gi32").shape,
        vec![2usize, 2],
        "grp/gi32 shape"
    );

    // The three-level hierarchy, and the attributes on the groups in it.
    assert_eq!(
        f.dataset("grp/sub/deep")
            .expect("grp/sub/deep")
            .as_f64()
            .expect("as_f64"),
        vec![3.5, 4.5]
    );
    // `AttrView` borrows its `Group`, so each handle has to outlive its
    // views.
    let grp = f.group("grp").expect("grp");
    let grp_attrs = grp.attr_views().expect("grp attrs");
    assert_eq!(grp_attrs.len(), 2, "grp attribute count");
    let sub = f.group("grp/sub").expect("grp/sub");
    let sub_attrs = sub.attr_views().expect("sub attrs");
    assert_eq!(sub_attrs.len(), 1, "grp/sub attribute count");
    assert_eq!(sub_attrs[0].as_f64(), Some(0.5));

    // Non-string and array attributes on the root group.
    let root = f.root().expect("root");
    let root_attrs = root.attr_views().expect("root attrs");
    assert_eq!(root_attrs.len(), 6, "root attribute count");
    let sources = root_attrs
        .iter()
        .find(|a| a.name() == "sources")
        .expect("sources");
    assert_eq!(
        sources.as_strings().expect("as_strings"),
        vec!["buoy".to_string(), "satellite".to_string()]
    );

    // The cross-group object reference resolves to the real header.
    let deep_attrs = f.attr_views("grp/sub/deep").expect("deep attrs");
    let source = deep_attrs
        .iter()
        .find(|a| a.name() == "SOURCE")
        .expect("SOURCE");
    assert_eq!(
        source.as_object_refs().expect("refs"),
        vec![f.header_addr_of("grp/gf64").expect("gf64 addr")]
    );

    // Wave-2 objects parse back: the gridded variable's shape, its vlen-objref
    // DIMENSION_LIST (two axes), the fixed-string dataset, the bool dataset, and
    // the compact/filtered datasets all read.
    assert_eq!(
        f.dataset("w2/grid").expect("w2/grid").shape,
        vec![3usize, 4],
        "w2/grid shape"
    );
    let grid_attrs = f.attr_views("w2/grid").expect("w2/grid attrs");
    let dim_list = grid_attrs
        .iter()
        .find(|a| a.name() == "DIMENSION_LIST")
        .expect("DIMENSION_LIST present");
    assert_eq!(
        dim_list.shape(),
        vec![2u64],
        "DIMENSION_LIST has one entry per axis"
    );
    assert_eq!(
        f.dataset("w2/names").expect("w2/names").shape,
        vec![2usize],
        "w2/names shape"
    );
    assert_eq!(
        f.dataset("w2/inline")
            .expect("w2/inline")
            .as_i32()
            .expect("inline i32"),
        vec![10, 20, 30],
        "compact dataset reads back inline"
    );
    assert_eq!(
        f.dataset("w2/shuffled")
            .expect("w2/shuffled")
            .as_i32()
            .expect("shuffled i32"),
        (0..64).collect::<Vec<i32>>(),
        "shuffle round-trips through the reader"
    );
    assert_eq!(
        f.dataset("w2/checked")
            .expect("w2/checked")
            .as_f64()
            .expect("checked f64"),
        vec![0.5; 32],
        "fletcher32 dataset reads back"
    );
}
/// Walk every object header in `bytes`, checking message alignment.
///
/// Returns the addresses visited, so the caller can assert it actually
/// inspected something.
fn walk_object_headers(bytes: &[u8]) -> Vec<usize> {
    let u16_at = |off: usize| u16::from_le_bytes([bytes[off], bytes[off + 1]]);
    let u32_at = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let u64_at = |off: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[off..off + 8]);
        u64::from_le_bytes(b)
    };

    /// Check one object header; panics with a precise message on violation.
    fn check_oh(
        bytes: &[u8],
        addr: usize,
        u16_at: &dyn Fn(usize) -> u16,
        u32_at: &dyn Fn(usize) -> u32,
    ) {
        assert_eq!(bytes[addr], 0x01, "OH at {addr}: expected version 1");
        let num_messages = u16_at(addr + 2) as usize;
        let header_data_size = u32_at(addr + 8) as usize;

        let mut pos = addr + 16;
        let end = pos + header_data_size;
        let mut seen = 0usize;
        while pos < end {
            let msg_type = u16_at(pos);
            let msg_size = u16_at(pos + 2) as usize;
            assert_eq!(
                msg_size % 8,
                0,
                "OH at {addr}: message {seen} (type 0x{msg_type:04x}) declares \
                     size {msg_size}, which is not 8-byte aligned — libhdf5 rejects \
                     the whole header chunk with \"message not aligned\""
            );
            pos += 8 + msg_size;
            seen += 1;
        }
        assert_eq!(
            pos, end,
            "OH at {addr}: messages overran header_data_size {header_data_size}"
        );
        assert_eq!(
            seen, num_messages,
            "OH at {addr}: walked {seen} messages but num_messages says {num_messages}"
        );
    }

    let mut visited = Vec::new();

    // The root group header sits at a fixed address.
    let root_oh = u64_at(64) as usize;
    check_oh(bytes, root_oh, &u16_at, &u32_at);
    visited.push(root_oh);

    // Walk root B-tree -> every SNOD -> every entry, into sub-groups too.
    let mut pending = vec![u64_at(80) as usize];
    while let Some(btree) = pending.pop() {
        for snod in btree_v1::probe::collect_snods(bytes, btree) {
            assert_eq!(&bytes[snod..snod + 4], b"SNOD", "expected a SNOD at {snod}");
            let n = u16_at(snod + 6) as usize;
            for i in 0..n {
                let ste = snod + 8 + i * 40;
                let oh_addr = u64_at(ste + 8) as usize;
                check_oh(bytes, oh_addr, &u16_at, &u32_at);
                visited.push(oh_addr);
                if u32_at(ste + 16) == 1 {
                    // cache_type 1: a group, with its B-tree cached.
                    pending.push(u64_at(ste + 24) as usize);
                }
            }
        }
    }
    visited
}

/// Every object-header message must declare an 8-byte-aligned size.
///
/// libhdf5's `H5O__chunk_deserialize` walks a v1 header by advancing
/// `8 + declared_size` and aborts the entire chunk with "message not
/// aligned" the moment that lands off an 8-byte boundary.  Our own reader
/// re-aligns regardless, so nothing but this test notices.
///
/// The fixture deliberately contains the cases that used to be misdeclared:
/// an i32 attribute (4 data bytes), odd-length string attributes on both a
/// dataset and the root group, and a chunked dataset (whose layout body is
/// `11 + (ndims+1)*4`).
#[test]
fn w1x_message_sizes_are_8_byte_aligned() {
    let le = oxih5_core::ByteOrder::Little;
    let dt_f64 = Dtype::Float { size: 8, order: le };

    let mut w = FileWriter::new();
    w.write_root_str_attr("title", "hello"); // 5 bytes — odd
    w.write_root_str_attr("aligned", "12345678"); // 8 bytes — already aligned

    w.write_dataset_f32("alpha", &[1.0f32, 2.0, 3.0], &[3])
        .expect("alpha");
    w.write_i32_attr("alpha", "dimid", 7).expect("i32 attr"); // 4 data bytes
    w.write_string_attr("alpha", "units", "meters")
        .expect("str attr"); // 6 bytes
    w.write_string_attr("alpha", "even", "12345678")
        .expect("even attr");
    w.write_i64_attr("alpha", "big", 1).expect("i64 attr");
    w.write_f64_attr("alpha", "sf", 0.5).expect("f64 attr");

    w.write_dataset_i32("beta", &[0i32, 1], &[2]).expect("beta");
    w.write_obj_ref_list_attr("alpha", "DIMENSION_LIST", &["beta"])
        .expect("objrefs");

    // Chunked layout bodies are 11 + (ndims+1)*4 — never a multiple of 8
    // for ndims 1 or 2.
    let raw1: Vec<u8> = (0..4).flat_map(|i| f64::from(i).to_le_bytes()).collect();
    w.create_dataset_unlimited("chunk1d", &[4], &[4], &dt_f64, &raw1)
        .expect("chunk1d");
    let raw2: Vec<u8> = (0..6).flat_map(|i| f64::from(i).to_le_bytes()).collect();
    w.create_dataset_unlimited("chunk2d", &[2, 3], &[2, 3], &dt_f64, &raw2)
        .expect("chunk2d");

    w.create_vlen_string_dataset("labels", &["a", "bc"])
        .expect("labels");

    // Sub-groups, so the walker recurses through further symbol tables —
    // and attributes on the groups themselves, which is where a group
    // header stops being the fixed 40 bytes it used to be.  Every one of
    // these has an odd body size.
    w.create_group("grp").expect("grp");
    w.write_group_dataset_f64("grp", "inner", &[1.0], &[1])
        .expect("inner");
    w.write_group_string_attr("grp", "inner", "units", "K")
        .expect("group attr"); // 1 byte — odd
    w.write_string_attr("grp", "title", "odd")
        .expect("group title"); // 3 bytes — odd
    w.write_i32_attr("grp", "dimid", 2).expect("group dimid"); // 4 bytes
    w.write_dataset_f64("grp/deeper/leaf", &[2.0], &[1])
        .expect("leaf");
    w.write_string_array_attr("grp/deeper", "names", &["a", "bcd"])
        .expect("deep names"); // 2 × 3 = 6 bytes — odd

    // Push past one SNOD's worth of root links, so the walker has to follow
    // more than one child out of the B-tree to reach every header.
    for i in 0..5usize {
        w.write_dataset_u8(&format!("pad{i}"), &[i as u8], &[1])
            .expect("pad");
    }

    let bytes = w.build_to_vec().expect("build_to_vec");
    let visited = walk_object_headers(&bytes);

    // root + 11 root-level objects + grp's two links + grp/deeper's one.
    assert_eq!(visited.len(), 15, "visited object headers: {visited:?}");
}

/// Every emitter must write exactly the number of bytes reserved for it.
///
/// `check_size` is active in release builds, so a formula desync surfaces
/// here as an `OxiH5Error::Format` instead of a silently corrupt file.
#[test]
fn w1x_size_contract_holds_for_every_structure() {
    let le = oxih5_core::ByteOrder::Little;
    let dt_f64 = Dtype::Float { size: 8, order: le };

    // The shared fixture, plus a few extra shapes that stress the size
    // formulas from different directions.
    let _ = w1x_fixture_bytes();

    let mut w = FileWriter::new();
    w.write_root_str_attr("a", "");
    // Scalar-ish, 1-D, and 3-D shapes.
    w.write_dataset_u8("d0", &[1u8], &[1]).expect("d0");
    w.write_dataset_f32("d1", &[0.0f32; 24], &[2, 3, 4])
        .expect("d1");
    // Long names and long values exercise the padding arithmetic.
    w.write_string_attr(
        "d1",
        "a_rather_long_attribute_name_here",
        "0123456789abcdef",
    )
    .expect("long attr");
    w.write_string_attr("d1", "x", "y").expect("short attr");
    w.write_obj_ref_list_attr("d1", "refs", &["d0"])
        .expect("refs");
    // 3-D unlimited dataset.
    let raw: Vec<u8> = (0..8).flat_map(|i| f64::from(i).to_le_bytes()).collect();
    w.create_dataset_unlimited("cube", &[2, 2, 2], &[2, 2, 2], &dt_f64, &raw)
        .expect("cube");
    w.create_vlen_string_dataset("s", &["", "", "x"])
        .expect("s");
    // Empty group and a populated group.
    w.create_group("empty").expect("empty");
    w.create_group("full").expect("full");
    for i in 0..4usize {
        w.write_group_dataset_i32("full", &format!("g{i}"), &[i as i32], &[1])
            .expect("group ds");
        w.write_group_string_attr("full", &format!("g{i}"), "units", "1")
            .expect("group attr");
    }
    let bytes = w.build_to_vec().expect("size contract must hold");

    let tmp = std::env::temp_dir().join("oxih5_test_w1x_size_contract.h5");
    std::fs::write(&tmp, &bytes).expect("write");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(f.dataset("d1").expect("d1").shape, vec![2usize, 3, 4]);
    assert_eq!(f.dataset("cube").expect("cube").shape, vec![2usize, 2, 2]);
}
// -----------------------------------------------------------------------
// W1d: the remaining fixed-width integer element types
// -----------------------------------------------------------------------

/// Every newly writable integer width must survive a real file round trip.
///
/// The values are deliberately the extremes of each range.  A wrong sign
/// flag or element size in the datatype message is invisible for small
/// positive numbers and shows up immediately at the boundaries, where it
/// either flips the sign or truncates the high bytes.
#[test]
fn w1d_u16_u32_u64_i8_i16_roundtrip() {
    let tmp = std::env::temp_dir().join("oxih5_test_w1d_int_widths.h5");

    let i8_vals = [i8::MIN, -1, 0, 1, i8::MAX];
    let i16_vals = [i16::MIN, -1, 0, 1, i16::MAX];
    let u16_vals = [0u16, 1, 32_768, u16::MAX];
    let u32_vals = [0u32, 1, 2_147_483_648, u32::MAX];
    let u64_vals = [0u64, 1, 9_223_372_036_854_775_808, u64::MAX];

    // Ascending names, five datasets: comfortably inside every symbol-table
    // limit, so a failure here is about element types and nothing else.
    let mut w = FileWriter::new();
    w.write_dataset_i8("a_i8", &i8_vals, &[5]).expect("i8");
    w.write_dataset_i16("b_i16", &i16_vals, &[5]).expect("i16");
    w.write_dataset_u16("c_u16", &u16_vals, &[4]).expect("u16");
    w.write_dataset_u32("d_u32", &u32_vals, &[4]).expect("u32");
    w.write_dataset_u64("e_u64", &u64_vals, &[4]).expect("u64");
    w.build(&tmp).expect("build");

    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);

    let i8_ds = f.dataset("a_i8").expect("a_i8");
    assert_eq!(i8_ds.shape, vec![5usize]);
    assert_eq!(i8_ds.as_i8().expect("as_i8"), i8_vals);

    let i16_ds = f.dataset("b_i16").expect("b_i16");
    assert_eq!(i16_ds.shape, vec![5usize]);
    assert_eq!(i16_ds.as_i16().expect("as_i16"), i16_vals);

    let u16_ds = f.dataset("c_u16").expect("c_u16");
    assert_eq!(u16_ds.shape, vec![4usize]);
    assert_eq!(u16_ds.as_u16().expect("as_u16"), u16_vals);

    assert_eq!(
        f.dataset("d_u32").expect("d_u32").as_u32().expect("as_u32"),
        u32_vals
    );
    assert_eq!(
        f.dataset("e_u64").expect("e_u64").as_u64().expect("as_u64"),
        u64_vals
    );

    // Signedness really is on disk, not inferred: the accessors that
    // disagree with the written sign flag must refuse the data rather than
    // reinterpret it.
    assert!(
        f.dataset("c_u16").expect("c_u16").as_i16().is_err(),
        "a u16 dataset must not decode as i16"
    );
    assert!(
        f.dataset("b_i16").expect("b_i16").as_u16().is_err(),
        "an i16 dataset must not decode as u16"
    );
}

/// A big-endian `Dtype` must be refused rather than written little-endian.
///
/// `create_dataset` and `create_dataset_unlimited` are the two entry points
/// that take a caller-supplied `Dtype`, so they are the only places a
/// big-endian request can arrive.  Before this guard the `order` field was
/// dropped on the floor: the file claimed big-endian and contained
/// little-endian bytes, and every value read back byte-swapped with no
/// error reported anywhere.
#[test]
fn w1d_big_endian_dtype_rejected() {
    let be = oxih5_core::ByteOrder::Big;
    let le = oxih5_core::ByteOrder::Little;
    let mut w = FileWriter::new();

    let be_f64 = Dtype::Float { size: 8, order: be };
    assert!(
        w.create_dataset("be_f64", &[2], &be_f64).is_err(),
        "big-endian float must be rejected"
    );
    let be_i32 = Dtype::Int {
        size: 4,
        signed: true,
        order: be,
    };
    assert!(
        w.create_dataset("be_i32", &[2], &be_i32).is_err(),
        "big-endian int must be rejected"
    );
    let raw = vec![0u8; 16];
    assert!(
        w.create_dataset_unlimited("be_chunk", &[2], &[2], &be_f64, &raw)
            .is_err(),
        "big-endian unlimited dataset must be rejected"
    );

    // The rejection must leave no trace: a dtype the writer refused must not
    // have been half-registered before the check ran.
    w.create_dataset("le_f64", &[2], &Dtype::Float { size: 8, order: le })
        .expect("little-endian must still be accepted");

    let tmp = std::env::temp_dir().join("oxih5_test_w1d_big_endian.h5");
    w.build(&tmp).expect("build");
    let f = File::open(&tmp).expect("open");
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(
        f.dataset_names().expect("names"),
        vec!["le_f64".to_string()],
        "rejected datasets must not reach the file"
    );
}
