//! Integration tests for soft links stored in **old-style** (symbol-table) groups.
//!
//! `tests/fixtures/soft_links_old.h5` is produced by `tests/gen_fixtures.py`
//! using h5py's *default* `libver`, which is what ordinary h5py code writes:
//! superblock v0 with the root group indexed by a version-1 B-tree over SNOD
//! symbol-table nodes plus a local heap.
//!
//! In such a group a soft link is a symbol-table entry whose *cache type* is 2:
//! the entry's object-header address field holds the HDF5 "undefined address"
//! sentinel (`u64::MAX`) and the link value is a NUL-terminated path stored in
//! the group's local heap at the offset held in the first four bytes of the
//! entry's scratch-pad area.  Treating that sentinel as a real address is what
//! used to produce `Corrupted("object header offset 18446744073709551615 too
//! large")`.
//!
//! Fixture contents:
//!
//! ```text
//! /target_ds        int32[4] = [11, 22, 33, 44]
//! /grp/inner        int32[3] = [1, 2, 3]
//! /grp/rel_alias -> "inner"        (relative, resolved from /grp)
//! /alias         -> "/target_ds"
//! /galias        -> "/grp"
//! /deep_alias    -> "/grp/inner"
//! /chain         -> "/alias"       (soft -> soft -> dataset)
//! /dangling      -> "/no_such"
//! /cycle_a       -> "/cycle_b", /cycle_b -> "/cycle_a"
//! ```

use oxih5::OxiH5Error;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn open_fixture() -> Option<oxih5::File> {
    let path = fixture("soft_links_old.h5");
    // Tracked in git (`tests/fixtures/soft_links_old.h5`) — a missing file
    // means a broken checkout and must fail every caller loudly, not
    // silently skip.
    assert!(
        path.exists(),
        "fixture soft_links_old.h5 missing at {} — it is tracked in git",
        path.display()
    );
    Some(oxih5::open(&path).expect("open soft_links_old.h5"))
}

/// The headline regression: a soft link to a dataset in an old-style group.
#[test]
fn soft_link_to_dataset_resolves() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("alias").expect("resolve /alias soft link");
    assert_eq!(ds.shape, vec![4]);
    assert_eq!(ds.as_i32().expect("i32"), vec![11, 22, 33, 44]);
}

/// A soft link and its hard-linked target must resolve to the same object
/// header, not merely to equal data.
#[test]
fn soft_link_shares_target_header_address() {
    let Some(file) = open_fixture() else { return };
    let via_alias = file
        .header_addr_of("alias")
        .expect("header address via soft link");
    let direct = file
        .header_addr_of("target_ds")
        .expect("header address of hard link");
    assert_eq!(via_alias, direct);
    assert_ne!(via_alias, u64::MAX);
}

/// A soft link whose target is a *group*: navigable, and its children readable.
#[test]
fn soft_link_to_group_resolves() {
    let Some(file) = open_fixture() else { return };

    let g = file.group("galias").expect("navigate soft link to group");
    let inner = g
        .dataset("inner")
        .expect("read dataset inside linked group");
    assert_eq!(inner.as_i32().expect("i32"), vec![1, 2, 3]);

    // The same target reached as a path segment.
    let via_path = file
        .dataset("galias/inner")
        .expect("read through soft-linked group segment");
    assert_eq!(via_path.as_i32().expect("i32"), vec![1, 2, 3]);
}

/// A soft link pointing at a dataset nested one level down.
#[test]
fn soft_link_to_nested_dataset_resolves() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("deep_alias").expect("resolve /deep_alias");
    assert_eq!(ds.as_i32().expect("i32"), vec![1, 2, 3]);
}

/// soft -> soft -> dataset must follow the whole chain.
#[test]
fn soft_link_chain_resolves() {
    let Some(file) = open_fixture() else { return };
    let ds = file.dataset("chain").expect("resolve /chain -> /alias");
    assert_eq!(ds.as_i32().expect("i32"), vec![11, 22, 33, 44]);
}

/// A soft link value that is *not* absolute resolves against the group holding
/// the link, exactly as libhdf5 does.
#[test]
fn relative_soft_link_resolves_against_containing_group() {
    let Some(file) = open_fixture() else { return };

    let via_path = file
        .dataset("grp/rel_alias")
        .expect("resolve relative soft link by path");
    assert_eq!(via_path.as_i32().expect("i32"), vec![1, 2, 3]);

    let g = file.group("grp").expect("open /grp");
    let via_group = g
        .dataset("rel_alias")
        .expect("resolve relative soft link from group handle");
    assert_eq!(via_group.as_i32().expect("i32"), vec![1, 2, 3]);
}

/// A dangling soft link is a clean typed error — never a panic and never a
/// bogus `u64::MAX` address error.
#[test]
fn dangling_soft_link_is_a_typed_not_found() {
    let Some(file) = open_fixture() else { return };
    match file.dataset("dangling") {
        Err(OxiH5Error::NotFound(msg)) => {
            assert!(
                msg.contains("no_such"),
                "error should name the missing target, got: {msg}"
            );
        }
        Err(other) => panic!("expected NotFound for a dangling soft link, got: {other:?}"),
        Ok(_) => panic!("dangling soft link must not resolve"),
    }
    assert!(!file.contains("dangling"));
}

/// Mutually recursive soft links terminate with an error rather than looping.
#[test]
fn cyclic_soft_links_are_rejected() {
    let Some(file) = open_fixture() else { return };
    let msg = match file.dataset("cycle_a") {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("a soft-link cycle must not resolve"),
    };
    assert!(
        msg.contains("cycle"),
        "error should mention the cycle, got: {msg}"
    );
}

/// Listing an old-style group classifies soft links by what they point at,
/// matching the behaviour already implemented for new-style groups.
#[test]
fn soft_links_are_listed_and_classified() {
    let Some(file) = open_fixture() else { return };
    let root = file.root().expect("root group");

    let datasets = root.datasets().expect("list datasets");
    for name in ["target_ds", "alias", "chain", "deep_alias"] {
        assert!(
            datasets.iter().any(|n| n == name),
            "expected dataset '{name}' in {datasets:?}"
        );
    }

    let groups = root.groups().expect("list groups");
    for name in ["grp", "galias"] {
        assert!(
            groups.iter().any(|n| n == name),
            "expected group '{name}' in {groups:?}"
        );
    }

    // Unresolvable links are reported in neither list.
    for name in ["dangling", "cycle_a", "cycle_b"] {
        assert!(
            !datasets.iter().any(|n| n == name) && !groups.iter().any(|n| n == name),
            "'{name}' is unresolvable and must not be listed"
        );
    }
}

/// Sub-region readers resolve soft links too (they take their own lookup path).
// `dataset_slice` takes one range per dimension, so a 1-D dataset is sliced with
// a one-element array of ranges; clippy reads `&[1..3]` as a mistyped `&[1; 3]`.
#[allow(clippy::single_range_in_vec_init)]
#[test]
fn soft_link_supports_slice_and_hyperslab() {
    let Some(file) = open_fixture() else { return };

    let slice = file
        .dataset_slice("alias", &[1..3])
        .expect("slice through a soft link");
    assert_eq!(slice.as_i32().expect("i32"), vec![22, 33]);

    let hs = file
        .dataset_hyperslab(
            "alias",
            &[oxih5::DimSelection {
                start: 0,
                stride: 2,
                count: 2,
                block: 1,
            }],
        )
        .expect("hyperslab through a soft link");
    assert_eq!(hs.as_i32().expect("i32"), vec![11, 33]);
}

/// `File::walk` reaches objects that are only named by a soft link.
#[test]
fn walk_visits_soft_linked_objects() {
    let Some(file) = open_fixture() else { return };
    let mut seen: Vec<(String, bool)> = Vec::new();
    file.walk(&mut |path, is_group| seen.push((path.to_string(), is_group)))
        .expect("walk");
    assert!(
        seen.iter().any(|(p, g)| p == "/alias" && !*g),
        "walk should visit /alias as a dataset: {seen:?}"
    );
    assert!(
        seen.iter().any(|(p, g)| p == "/galias" && *g),
        "walk should visit /galias as a group: {seen:?}"
    );
}
