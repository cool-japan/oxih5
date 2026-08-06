//! Integration tests for Virtual Dataset (layout class 3) resolution.
//!
//! The fixtures under `tests/fixtures/vds_*.h5` are produced by h5py (see
//! `tests/gen_fixtures.py`) with `libver='latest'` so the virtual dataset uses
//! the version-4 "virtual" data-layout message and a global-heap mapping block.
//! Each virtual dataset maps onto one or more source datasets stored in
//! `vds_source.h5` in the same directory; resolving the VDS requires opening the
//! source file, reading the mapped regions and scattering them into the virtual
//! dataspace.

use std::path::Path;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A full "all → all" mapping: the whole source dataset fills the whole virtual
/// dataset.
#[test]
fn vds_simple_full_mapping() {
    let path = fixture("vds_simple.h5");
    // Tracked in git (`tests/fixtures/vds_simple.h5`) — a missing file means a
    // broken checkout and must fail the test loudly, not skip it silently.
    assert!(
        path.exists(),
        "fixture vds_simple.h5 missing at {} — it is tracked in git",
        path.display()
    );
    let file = oxih5::open(&path).expect("open vds_simple.h5");
    let ds = file.dataset("virt").expect("resolve virtual dataset");
    assert_eq!(ds.shape, vec![6]);
    let values = ds.as_f64().expect("decode f64");
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Two source datasets concatenated into one virtual dataset via hyperslab
/// virtual selections `[0:4]` and `[4:8]`.
#[test]
fn vds_concat_two_sources() {
    let path = fixture("vds_concat.h5");
    // Tracked in git — see `vds_simple_full_mapping` above.
    assert!(
        path.exists(),
        "fixture vds_concat.h5 missing at {} — it is tracked in git",
        path.display()
    );
    let file = oxih5::open(&path).expect("open vds_concat.h5");
    let ds = file.dataset("cat").expect("resolve virtual dataset");
    assert_eq!(ds.shape, vec![8]);
    let values = ds.as_i32().expect("decode i32");
    assert_eq!(values, vec![10, 11, 12, 13, 20, 21, 22, 23]);
}

/// The pre-existing multi-source fixture must resolve identically.
#[test]
fn vds_main_fixture() {
    let path = fixture("vds_main.h5");
    // Tracked in git — see `vds_simple_full_mapping` above.
    assert!(
        path.exists(),
        "fixture vds_main.h5 missing at {} — it is tracked in git",
        path.display()
    );
    let file = oxih5::open(&path).expect("open vds_main.h5");
    let ds = file
        .dataset("virtual_data")
        .expect("resolve virtual dataset");
    let values = ds.as_f32().expect("decode f32");
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ---------------------------------------------------------------------------
// H4(c): soft link that points through to an external link.
// `soft_ext_main.h5` has `/ext` (external link -> soft_ext_target.h5:/payload)
// and `/soft` (soft link -> /ext).  Reading `/soft` must resolve through both.
// ---------------------------------------------------------------------------
#[test]
fn soft_link_through_external_link() {
    let path = fixture("soft_ext_main.h5");
    // Tracked in git — see `vds_simple_full_mapping` above.
    assert!(
        path.exists(),
        "fixture soft_ext_main.h5 missing at {} — it is tracked in git",
        path.display()
    );
    let file = oxih5::open(&path).expect("open soft_ext_main.h5");
    let via_ext = file.dataset("ext").expect("resolve external link");
    assert_eq!(via_ext.as_i32().expect("i32"), vec![7, 8, 9, 10]);
    let via_soft = file
        .dataset("soft")
        .expect("resolve soft link through external link");
    assert_eq!(via_soft.as_i32().expect("i32"), vec![7, 8, 9, 10]);
}
