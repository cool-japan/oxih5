//! G011 / G013 — link creation and new-style (link-message) group storage.
//!
//! Every test writes a file with `FileWriter` and reads it back with this
//! crate's own reader, asserting **both** the decoded meaning and the bytes the
//! writer put on disk.  The byte assertions are the point: the reader is
//! forgiving in places (it never verifies a version-2 checksum, and it re-aligns
//! object-header messages either way), so a structure that is merely
//! *self-consistent* would round-trip while remaining unreadable to libhdf5.
//!
//! The layouts asserted here were captured from files written by **h5py 3.16 /
//! libhdf5 2.0.0**, and every file these tests produce has been read back by
//! that same libhdf5.

use oxih5::FileWriter;
use oxih5_format::context::ParseContext;
use oxih5_format::{group, header, superblock};

/// Write `writer` to a scratch file and return both the path and its bytes.
fn build(writer: &mut FileWriter, name: &str) -> (std::path::PathBuf, Vec<u8>) {
    let path = std::env::temp_dir().join(format!("oxih5_wave5_{name}.h5"));
    writer.build(&path).expect("build");
    let bytes = std::fs::read(&path).expect("read back");
    (path, bytes)
}

/// The message types present in the object header at `addr`, in order.
fn msg_types(bytes: &[u8], addr: u64) -> Vec<u16> {
    header::parse_messages(bytes, addr)
        .expect("parse messages")
        .iter()
        .map(|m| m.msg_type)
        .collect()
}

/// The root group's object header address.
fn root_addr(bytes: &[u8]) -> u64 {
    superblock::parse(bytes)
        .expect("superblock")
        .root_object_header_address
}

/// The object header reference count a v1 header declares.
fn refcount(bytes: &[u8], addr: u64) -> u32 {
    let at = addr as usize + 4;
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

/// Resolve `name` in the root group, whichever storage style it uses.
fn root_link(bytes: &[u8], name: &str) -> oxih5_core::Link {
    let ctx = ParseContext::default_v0();
    let root = root_addr(bytes);
    if let Some((btree, heap)) = header::parse_messages(bytes, root)
        .expect("messages")
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .map(|m| {
            (
                u64::from_le_bytes(m.data[0..8].try_into().expect("8 bytes")),
                u64::from_le_bytes(m.data[8..16].try_into().expect("8 bytes")),
            )
        })
    {
        return match group::find_entry(bytes, btree, heap, name).expect("find entry") {
            group::SymTabLink::Hard(address) => oxih5_core::Link::Hard { address },
            group::SymTabLink::Soft(path) => oxih5_core::Link::Soft { path },
        };
    }
    group::list_new_style_links(bytes, root, &ctx)
        .expect("new-style links")
        .into_iter()
        .find(|pl| pl.name == name)
        .unwrap_or_else(|| panic!("no link named '{name}'"))
        .link
}

// ---------------------------------------------------------------------------
// G011 — old-style groups: soft links and hard aliases
// ---------------------------------------------------------------------------

/// A soft link and a hard alias both live in an ordinary symbol table, and the
/// group stays old-style — no Link Info message anywhere.
#[test]
fn soft_link_and_hard_alias_stay_in_the_symbol_table() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1, 2, 3], &[3]).expect("d");
    w.create_soft_link("/soft", "/d").expect("soft");
    w.create_hard_link("/hard", "/d").expect("hard");
    let (path, bytes) = build(&mut w, "old_links");

    // The root group is still a symbol-table group.
    let types = msg_types(&bytes, root_addr(&bytes));
    assert!(types.contains(&0x0011), "symbol table message: {types:?}");
    assert!(!types.contains(&0x0002), "no link info: {types:?}");
    assert!(!types.contains(&0x0006), "no link messages: {types:?}");

    // The soft link decodes to its stored path, and the alias to the very same
    // object header address as the dataset it aliases.
    assert_eq!(
        root_link(&bytes, "soft"),
        oxih5_core::Link::Soft {
            path: "/d".to_string()
        }
    );
    let (oxih5_core::Link::Hard { address: aliased }, oxih5_core::Link::Hard { address: direct }) =
        (root_link(&bytes, "hard"), root_link(&bytes, "d"))
    else {
        panic!("both must be hard links");
    };
    assert_eq!(aliased, direct, "an alias names the same object header");

    // And both names read the same data through the public API.
    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("d").expect("d").as_i32().expect("i32"),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("hard").expect("hard").as_i32().expect("i32"),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("soft").expect("soft").as_i32().expect("i32"),
        vec![1, 2, 3]
    );
    let _ = std::fs::remove_file(&path);
}

/// A hard alias raises the target's object header reference count.
///
/// libhdf5 writes 2 for a dataset with one alias; leaving it at 1 would make
/// `H5Ldelete` on either name free storage the other name still reaches.
#[test]
fn a_hard_alias_raises_the_object_header_reference_count() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1], &[1]).expect("d");
    w.write_dataset_i32("/lonely", &[1], &[1]).expect("lonely");
    w.create_hard_link("/a", "/d").expect("a");
    w.create_hard_link("/b", "/d").expect("b");
    let (path, bytes) = build(&mut w, "refcount");

    let oxih5_core::Link::Hard { address: d } = root_link(&bytes, "d") else {
        panic!("d is a hard link");
    };
    let oxih5_core::Link::Hard { address: lonely } = root_link(&bytes, "lonely") else {
        panic!("lonely is a hard link");
    };
    assert_eq!(refcount(&bytes, d), 3, "one name plus two aliases");
    assert_eq!(refcount(&bytes, lonely), 1, "an un-aliased object stays 1");
    // The root group is reached by the superblock, not by a link.
    assert_eq!(refcount(&bytes, root_addr(&bytes)), 1);
    let _ = std::fs::remove_file(&path);
}

/// A hard alias may be created *before* its target: it is resolved against the
/// finished layout, not against whatever existed at the call.
#[test]
fn a_hard_alias_may_name_a_target_created_later() {
    let mut w = FileWriter::new();
    w.create_hard_link("/alias", "/g/deep").expect("alias");
    w.write_dataset_f64("/g/deep", &[2.5], &[1]).expect("deep");
    let (path, bytes) = build(&mut w, "forward_alias");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("alias").expect("alias").as_f64().expect("f64"),
        vec![2.5]
    );
    let oxih5_core::Link::Hard { address } = root_link(&bytes, "alias") else {
        panic!("alias is a hard link");
    };
    assert_eq!(refcount(&bytes, address), 2);
    let _ = std::fs::remove_file(&path);
}

/// A hard alias whose target is not in the finished file is a typed error at
/// build time — the one place where it can be known.
#[test]
fn a_hard_alias_to_nothing_is_rejected_at_build() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1], &[1]).expect("d");
    w.create_hard_link("/dangling", "/nowhere")
        .expect("accepted");
    let Err(err) = w.build_to_vec() else {
        panic!("a dangling hard link must not build");
    };
    assert!(
        format!("{err}").contains("not an object in this file"),
        "{err}"
    );
}

/// A soft link's target need not exist — that is the whole difference between a
/// soft link and a hard one — and the writer must not invent one.
#[test]
fn a_soft_link_may_dangle() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1], &[1]).expect("d");
    w.create_soft_link("/later", "/not/yet").expect("soft");
    let (path, bytes) = build(&mut w, "dangling_soft");

    assert_eq!(
        root_link(&bytes, "later"),
        oxih5_core::Link::Soft {
            path: "/not/yet".to_string()
        }
    );
    let f = oxih5::open(&path).expect("open");
    assert!(f.dataset("later").is_err(), "the target really is missing");
    let _ = std::fs::remove_file(&path);
}

/// A relative soft link resolves against the group that holds it, not the root.
#[test]
fn a_relative_soft_link_resolves_within_its_own_group() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/g/inner", &[7], &[1]).expect("inner");
    w.write_dataset_i32("/inner", &[9], &[1]).expect("outer");
    w.create_soft_link("/g/near", "inner").expect("near");
    let (path, _) = build(&mut w, "relative_soft");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("g/near").expect("near").as_i32().expect("i32"),
        vec![7],
        "a relative target is the sibling, not the root object of the same name"
    );
    let _ = std::fs::remove_file(&path);
}

/// A link owns its name like any other member: nothing may reuse it, and the
/// three kinds are checked against each other.
#[test]
fn link_names_collide_with_every_other_member_kind() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1], &[1]).expect("d");
    w.create_soft_link("/s", "/d").expect("s");
    w.create_group("/g").expect("g");

    assert!(w.create_soft_link("/d", "/d").is_err(), "dataset name");
    assert!(w.create_soft_link("/g", "/d").is_err(), "group name");
    assert!(w.create_soft_link("/s", "/d").is_err(), "link name");
    assert!(
        w.write_dataset_i32("/s", &[1], &[1]).is_err(),
        "reuse by dataset"
    );
    assert!(w.create_group("/s").is_err(), "reuse by group");
    assert!(
        w.write_dataset_i32("/s/below", &[1], &[1]).is_err(),
        "under a link"
    );
    // Malformed values are refused too.
    assert!(w.create_soft_link("/e", "").is_err(), "empty target");
    assert!(w.create_soft_link("/e", "a\0b").is_err(), "NUL in target");
    assert!(
        w.create_external_link("/e", "", "/x").is_err(),
        "empty file"
    );
}

// ---------------------------------------------------------------------------
// G013 — new-style groups: compact link messages
// ---------------------------------------------------------------------------

/// An external link converts its group to link-message storage, exactly as
/// libhdf5 does, and every other member comes along as a Link message.
#[test]
fn an_external_link_converts_its_group_to_link_messages() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1, 2, 3], &[3]).expect("d");
    w.create_soft_link("/soft", "/d").expect("soft");
    w.create_hard_link("/hard", "/d").expect("hard");
    w.create_external_link("/ext", "peer.h5", "/ds")
        .expect("ext");
    let (path, bytes) = build(&mut w, "compact_links");

    // Link Info + Group Info + one Link per member; and *no* symbol table
    // message, or the reader would classify the group as old-style and see none
    // of the links.
    let types = msg_types(&bytes, root_addr(&bytes));
    assert_eq!(
        types,
        vec![0x0002, 0x000A, 0x0006, 0x0006, 0x0006, 0x0006],
        "link info, group info, four links"
    );

    // The Link Info message names no fractal heap and no index: this group is
    // compact, so both addresses are the undefined sentinel.
    let msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let info = msgs
        .iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    assert_eq!(info.data[0], 0, "version 0");
    assert_eq!(info.data[1], 0, "creation order not tracked");
    assert!(
        info.data[2..18].iter().all(|&b| b == 0xFF),
        "compact: both addresses undefined"
    );

    // Every link kind survives.
    assert_eq!(
        root_link(&bytes, "ext"),
        oxih5_core::Link::External {
            file: "peer.h5".to_string(),
            path: "/ds".to_string()
        }
    );
    assert_eq!(
        root_link(&bytes, "soft"),
        oxih5_core::Link::Soft {
            path: "/d".to_string()
        }
    );
    let (oxih5_core::Link::Hard { address: aliased }, oxih5_core::Link::Hard { address: direct }) =
        (root_link(&bytes, "hard"), root_link(&bytes, "d"))
    else {
        panic!("both must be hard links");
    };
    assert_eq!(aliased, direct);
    assert_eq!(refcount(&bytes, direct), 2, "the alias is counted");

    // The data still reads through every name.
    let f = oxih5::open(&path).expect("open");
    for name in ["d", "hard", "soft"] {
        assert_eq!(
            f.dataset(name).expect(name).as_i32().expect("i32"),
            vec![1, 2, 3],
            "{name}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// A converted group keeps its symbol table *structures* — an empty B-tree,
/// local heap and SNOD — because the parent's cached entry still points at them.
///
/// That is what libhdf5 leaves behind when it converts a group, and it is why
/// the writer never has to special-case a new-style group's parent entry.
#[test]
fn a_converted_group_keeps_real_but_empty_symbol_table_structures() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/g/d", &[1], &[1]).expect("d");
    w.create_external_link("/g/ext", "peer.h5", "/ds")
        .expect("ext");
    let (path, bytes) = build(&mut w, "vestigial_stab");

    // The *root* is still old-style, and its entry for `g` is an ordinary
    // cached-group entry (cache type 1).
    let root_msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = root_msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let entries = group::list_entries(&bytes, btree, heap).expect("entries");
    let (name, link) = entries.first().expect("one entry");
    assert_eq!(name, "g");
    let group::SymTabLink::Hard(g_addr) = link else {
        panic!("g is a hard link");
    };

    // `g` itself is new-style…
    let g_types = msg_types(&bytes, *g_addr);
    assert!(!g_types.contains(&0x0011), "no symbol table message");
    assert!(g_types.contains(&0x0002), "link info present");
    // …and the reader finds both of its members through the link messages.
    let ctx = ParseContext::default_v0();
    let mut names: Vec<String> = group::list_new_style_links(&bytes, *g_addr, &ctx)
        .expect("links")
        .into_iter()
        .map(|pl| pl.name)
        .collect();
    names.sort();
    assert_eq!(names, vec!["d".to_string(), "ext".to_string()]);

    let f = oxih5::open(&path).expect("open");
    assert_eq!(f.dataset("g/d").expect("d").as_i32().expect("i32"), vec![1]);
    let _ = std::fs::remove_file(&path);
}

/// `set_track_order` records each member's creation order and preserves it in
/// the *stored* order of the link messages, which is what a reader walks.
#[test]
fn track_order_preserves_creation_order_in_the_stored_links() {
    let mut w = FileWriter::new();
    w.create_group("/t").expect("t");
    w.set_track_order("/t").expect("track");
    for name in ["zeta", "alpha", "mid"] {
        w.write_dataset_i32(&format!("/t/{name}"), &[1], &[1])
            .expect("ds");
    }
    let (path, bytes) = build(&mut w, "track_order");

    let root_msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = root_msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let group::SymTabLink::Hard(t_addr) = group::find_entry(&bytes, btree, heap, "t").expect("t")
    else {
        panic!("t is a hard link");
    };

    // The Link Info message declares that order is tracked, and records one
    // past the highest value handed out.
    let msgs = header::parse_messages(&bytes, t_addr).expect("messages");
    let info = msgs
        .iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    assert_eq!(info.data[1] & 0x01, 0x01, "creation order tracked");
    assert_eq!(
        u64::from_le_bytes(info.data[2..10].try_into().expect("8 bytes")),
        3,
        "three members were created"
    );

    // Stored order is creation order, not name order.
    let ctx = ParseContext::default_v0();
    let names: Vec<String> = group::list_new_style_links(&bytes, t_addr, &ctx)
        .expect("links")
        .into_iter()
        .map(|pl| pl.name)
        .collect();
    assert_eq!(names, vec!["zeta", "alpha", "mid"]);

    // Each Link message carries its own creation-order field, in order.
    let orders: Vec<u64> = msgs
        .iter()
        .filter(|m| m.msg_type == 0x0006)
        .map(|m| {
            assert_eq!(m.data[1] & 0x04, 0x04, "creation order field present");
            u64::from_le_bytes(m.data[2..10].try_into().expect("8 bytes"))
        })
        .collect();
    assert_eq!(orders, vec![0, 1, 2]);

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("t/alpha").expect("alpha").as_i32().expect("i32"),
        vec![1]
    );
    let _ = std::fs::remove_file(&path);
}

/// Without `track_order`, a link-message group keeps name ordering — turning a
/// group new-style must not silently reshuffle it.
#[test]
fn without_track_order_link_messages_stay_in_name_order() {
    let mut w = FileWriter::new();
    w.create_group("/g").expect("g");
    w.set_link_storage("/g").expect("link storage");
    for name in ["zeta", "alpha", "mid"] {
        w.write_dataset_i32(&format!("/g/{name}"), &[1], &[1])
            .expect("ds");
    }
    let (path, bytes) = build(&mut w, "no_track_order");

    let f = oxih5::open(&path).expect("open");
    let g = f.group("g").expect("g");
    let mut names = g.datasets().expect("names");
    names.sort();
    assert_eq!(names, vec!["alpha", "mid", "zeta"]);

    let root_msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = root_msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let group::SymTabLink::Hard(g_addr) = group::find_entry(&bytes, btree, heap, "g").expect("g")
    else {
        panic!("g is a hard link");
    };
    let ctx = ParseContext::default_v0();
    let stored: Vec<String> = group::list_new_style_links(&bytes, g_addr, &ctx)
        .expect("links")
        .into_iter()
        .map(|pl| pl.name)
        .collect();
    assert_eq!(stored, vec!["alpha", "mid", "zeta"]);
    // And no creation-order field is written when it is not tracked.
    let msgs = header::parse_messages(&bytes, g_addr).expect("messages");
    for msg in msgs.iter().filter(|m| m.msg_type == 0x0006) {
        assert_eq!(msg.data[1] & 0x04, 0, "no creation order field");
    }
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// G013 — new-style groups: dense (fractal heap + version-2 B-tree)
// ---------------------------------------------------------------------------

/// Past libhdf5's `max_compact` of 8, a group's links move into a fractal heap
/// indexed by a version-2 B-tree — and every one of them still resolves.
#[test]
fn a_group_past_max_compact_moves_its_links_into_a_fractal_heap() {
    let mut w = FileWriter::new();
    w.create_group("/g").expect("g");
    w.set_link_storage("/g").expect("link storage");
    for i in 0..12 {
        w.write_dataset_i32(&format!("/g/ds{i:02}"), &[i], &[1])
            .expect("ds");
    }
    w.create_external_link("/g/ext", "peer.h5", "/target")
        .expect("ext");
    let (path, bytes) = build(&mut w, "dense_links");

    let root_msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = root_msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let group::SymTabLink::Hard(g_addr) = group::find_entry(&bytes, btree, heap, "g").expect("g")
    else {
        panic!("g is a hard link");
    };

    // Link Info and Group Info only: the links are *not* header messages.
    assert_eq!(msg_types(&bytes, g_addr), vec![0x0002, 0x000A]);

    // The Link Info message names a real fractal heap and a real name index,
    // and both signatures are where it says they are.
    let msgs = header::parse_messages(&bytes, g_addr).expect("messages");
    let info = msgs
        .iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    let fh = u64::from_le_bytes(info.data[2..10].try_into().expect("8 bytes"));
    let ni = u64::from_le_bytes(info.data[10..18].try_into().expect("8 bytes"));
    assert_ne!(fh, u64::MAX, "a dense group has a fractal heap");
    assert_ne!(ni, u64::MAX, "a dense group has a name index");
    assert_eq!(&bytes[fh as usize..fh as usize + 4], b"FRHP");
    assert_eq!(&bytes[ni as usize..ni as usize + 4], b"BTHD");
    assert_eq!(bytes[ni as usize + 5], 5, "type 5 = link name index");

    // Its root direct block is where the heap header says, and the heap's own
    // object count matches the link count.
    let heap_root = u64::from_le_bytes(
        bytes[fh as usize + 132..fh as usize + 140]
            .try_into()
            .expect("8 bytes"),
    );
    assert_eq!(&bytes[heap_root as usize..heap_root as usize + 4], b"FHDB");
    assert_eq!(
        u64::from_le_bytes(
            bytes[fh as usize + 70..fh as usize + 78]
                .try_into()
                .expect("8 bytes")
        ),
        13,
        "twelve datasets and one external link"
    );

    // Every link comes back out through the heap and the index.
    let ctx = ParseContext::default_v0();
    let links = group::list_new_style_links(&bytes, g_addr, &ctx).expect("links");
    let mut names: Vec<&str> = links.iter().map(|pl| pl.name.as_str()).collect();
    names.sort_unstable();
    let mut want: Vec<String> = (0..12).map(|i| format!("ds{i:02}")).collect();
    want.push("ext".to_string());
    want.sort();
    assert_eq!(names, want.iter().map(String::as_str).collect::<Vec<_>>());

    assert_eq!(
        links.iter().find(|pl| pl.name == "ext").expect("ext").link,
        oxih5_core::Link::External {
            file: "peer.h5".to_string(),
            path: "/target".to_string()
        }
    );

    // And the data reads through every one of them.
    let f = oxih5::open(&path).expect("open");
    for i in 0..12 {
        let name = format!("g/ds{i:02}");
        assert_eq!(
            f.dataset(&name).expect(&name).as_i32().expect("i32"),
            vec![i],
            "{name}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// The name index's records must ascend by the Jenkins lookup3 hash of the link
/// name, because libhdf5 binary-searches them.  Out-of-order records do not
/// fail loudly — libhdf5 simply cannot find some links.
#[test]
fn the_name_index_records_ascend_by_link_name_hash() {
    let mut w = FileWriter::new();
    w.create_group("/g").expect("g");
    w.set_link_storage("/g").expect("link storage");
    for name in [
        "zeta", "alpha", "mid", "omega", "beta", "gamma", "delta", "epsilon", "eta", "theta",
    ] {
        w.write_dataset_i32(&format!("/g/{name}"), &[1], &[1])
            .expect("ds");
    }
    let (path, bytes) = build(&mut w, "hash_order");

    let root_msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = root_msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let group::SymTabLink::Hard(g_addr) = group::find_entry(&bytes, btree, heap, "g").expect("g")
    else {
        panic!("g is a hard link");
    };
    let msgs = header::parse_messages(&bytes, g_addr).expect("messages");
    let info = msgs
        .iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    let ni = u64::from_le_bytes(info.data[10..18].try_into().expect("8 bytes")) as usize;

    let record_size = u16::from_le_bytes([bytes[ni + 10], bytes[ni + 11]]) as usize;
    let depth = u16::from_le_bytes([bytes[ni + 12], bytes[ni + 13]]);
    let root = u64::from_le_bytes(bytes[ni + 16..ni + 24].try_into().expect("8 bytes")) as usize;
    let nrec = u16::from_le_bytes([bytes[ni + 24], bytes[ni + 25]]) as usize;
    assert_eq!(depth, 0, "a single leaf");
    assert_eq!(nrec, 10);
    assert_eq!(record_size, 4 + 7, "hash plus a 7-byte heap ID");
    assert_eq!(&bytes[root..root + 4], b"BTLF");

    let hashes: Vec<u32> = (0..nrec)
        .map(|i| {
            let at = root + 6 + i * record_size;
            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
        })
        .collect();
    let mut sorted = hashes.clone();
    sorted.sort_unstable();
    assert_eq!(hashes, sorted, "records must ascend by name hash");

    let _ = std::fs::remove_file(&path);
}

/// A dense group's heap and index carry the metadata checksums libhdf5
/// computes; the reader verifies neither, so this test has to.
#[test]
fn dense_link_structures_carry_correct_metadata_checksums() {
    /// Jenkins lookup3 (`H5_checksum_metadata`), independently re-implemented
    /// here so a test failure means the writer changed, not that both sides
    /// share one mistake.
    fn lookup3(key: &[u8]) -> u32 {
        fn word(b: &[u8]) -> u32 {
            b.iter()
                .take(4)
                .enumerate()
                .fold(0u32, |acc, (i, &v)| acc | (u32::from(v) << (8 * i)))
        }
        let seed = 0xdead_beefu32.wrapping_add(key.len() as u32);
        let (mut a, mut b, mut c) = (seed, seed, seed);
        let mut rest = key;
        while rest.len() > 12 {
            a = a.wrapping_add(word(&rest[0..4]));
            b = b.wrapping_add(word(&rest[4..8]));
            c = c.wrapping_add(word(&rest[8..12]));
            a = a.wrapping_sub(c);
            a ^= c.rotate_left(4);
            c = c.wrapping_add(b);
            b = b.wrapping_sub(a);
            b ^= a.rotate_left(6);
            a = a.wrapping_add(c);
            c = c.wrapping_sub(b);
            c ^= b.rotate_left(8);
            b = b.wrapping_add(a);
            a = a.wrapping_sub(c);
            a ^= c.rotate_left(16);
            c = c.wrapping_add(b);
            b = b.wrapping_sub(a);
            b ^= a.rotate_left(19);
            a = a.wrapping_add(c);
            c = c.wrapping_sub(b);
            c ^= b.rotate_left(4);
            b = b.wrapping_add(a);
            rest = &rest[12..];
        }
        if rest.is_empty() {
            return c;
        }
        if rest.len() > 8 {
            c = c.wrapping_add(word(&rest[8..]));
        }
        if rest.len() > 4 {
            b = b.wrapping_add(word(&rest[4..]));
        }
        a = a.wrapping_add(word(rest));
        c ^= b;
        c = c.wrapping_sub(b.rotate_left(14));
        a ^= c;
        a = a.wrapping_sub(c.rotate_left(11));
        b ^= a;
        b = b.wrapping_sub(a.rotate_left(25));
        c ^= b;
        c = c.wrapping_sub(b.rotate_left(16));
        a ^= c;
        a = a.wrapping_sub(c.rotate_left(4));
        b ^= a;
        b = b.wrapping_sub(a.rotate_left(14));
        c ^= b;
        c.wrapping_sub(b.rotate_left(24))
    }

    let mut w = FileWriter::new();
    w.set_link_storage("/").expect("link storage");
    for i in 0..10 {
        w.write_dataset_i32(&format!("/ds{i:02}"), &[i], &[1])
            .expect("ds");
    }
    let (path, bytes) = build(&mut w, "checksums");

    let msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let info = msgs
        .iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    let fh = u64::from_le_bytes(info.data[2..10].try_into().expect("8 bytes")) as usize;
    let ni = u64::from_le_bytes(info.data[10..18].try_into().expect("8 bytes")) as usize;

    // Heap header: checksum over its first 142 bytes.
    assert_eq!(
        u32::from_le_bytes(bytes[fh + 142..fh + 146].try_into().expect("4 bytes")),
        lookup3(&bytes[fh..fh + 142]),
        "fractal heap header checksum"
    );

    // Root direct block: checksum over the *whole* block with the field zeroed.
    let block = u64::from_le_bytes(bytes[fh + 132..fh + 140].try_into().expect("8 bytes")) as usize;
    let block_size =
        u64::from_le_bytes(bytes[fh + 112..fh + 120].try_into().expect("8 bytes")) as usize;
    let stored = u32::from_le_bytes(bytes[block + 17..block + 21].try_into().expect("4 bytes"));
    let mut image = bytes[block..block + block_size].to_vec();
    image[17..21].fill(0);
    assert_eq!(
        stored,
        lookup3(&image),
        "fractal heap direct block checksum"
    );

    // B-tree header: checksum over its first 34 bytes.
    assert_eq!(
        u32::from_le_bytes(bytes[ni + 34..ni + 38].try_into().expect("4 bytes")),
        lookup3(&bytes[ni..ni + 34]),
        "name index header checksum"
    );

    // Leaf node: checksum over the prefix plus the records only, not the whole
    // padded node.
    let record_size = u16::from_le_bytes([bytes[ni + 10], bytes[ni + 11]]) as usize;
    let root = u64::from_le_bytes(bytes[ni + 16..ni + 24].try_into().expect("8 bytes")) as usize;
    let nrec = u16::from_le_bytes([bytes[ni + 24], bytes[ni + 25]]) as usize;
    let live = 6 + nrec * record_size;
    assert_eq!(
        u32::from_le_bytes(
            bytes[root + live..root + live + 4]
                .try_into()
                .expect("4 bytes")
        ),
        lookup3(&bytes[root..root + live]),
        "name index leaf checksum"
    );

    let _ = std::fs::remove_file(&path);
}

/// A dense group nests, aliases and soft-links like any other, and the whole
/// file still walks from the root.
#[test]
fn a_dense_group_supports_every_link_kind_at_once() {
    let mut w = FileWriter::new();
    w.create_group("/big").expect("big");
    w.set_track_order("/big").expect("track");
    for i in 0..9 {
        w.write_dataset_f64(&format!("/big/v{i}"), &[f64::from(i)], &[1])
            .expect("ds");
    }
    w.create_group("/big/sub").expect("sub");
    w.write_dataset_i32("/big/sub/leaf", &[42], &[1])
        .expect("leaf");
    w.create_soft_link("/big/first", "v0").expect("soft");
    w.create_hard_link("/big/alias", "/big/v3").expect("alias");
    w.create_external_link("/big/away", "peer.h5", "/x")
        .expect("ext");
    let (path, bytes) = build(&mut w, "dense_mixed");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("big/first")
            .expect("first")
            .as_f64()
            .expect("f64"),
        vec![0.0]
    );
    assert_eq!(
        f.dataset("big/alias")
            .expect("alias")
            .as_f64()
            .expect("f64"),
        vec![3.0]
    );
    assert_eq!(
        f.dataset("big/sub/leaf")
            .expect("leaf")
            .as_i32()
            .expect("i32"),
        vec![42]
    );

    // Creation order is still recorded on every link in the heap.
    let root_msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = root_msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let group::SymTabLink::Hard(big) = group::find_entry(&bytes, btree, heap, "big").expect("big")
    else {
        panic!("big is a hard link");
    };
    let msgs = header::parse_messages(&bytes, big).expect("messages");
    let info = msgs
        .iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    assert_eq!(info.data[1] & 0x01, 0x01, "creation order tracked");
    assert_eq!(
        u64::from_le_bytes(info.data[2..10].try_into().expect("8 bytes")),
        13,
        "nine values, a sub-group, and three links"
    );
    let ctx = ParseContext::default_v0();
    assert_eq!(
        group::list_new_style_links(&bytes, big, &ctx)
            .expect("links")
            .len(),
        13
    );
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Cross-feature: links and group styles against everything else the writer does
// ---------------------------------------------------------------------------

/// A hard alias and a soft link may name a **group**, not just a dataset — and
/// in both storage styles.
///
/// In a symbol table this is the case where the entry's *cache type* could have
/// been wrong: a sub-group's own entry caches its B-tree and local heap
/// (cache type 1), but an alias caches nothing (cache type 0), which is legal
/// precisely because the field is a cache and libhdf5 falls back to the group's
/// own Symbol Table message.
#[test]
fn a_link_may_name_a_group_in_either_storage_style() {
    for (label, link_style) in [("old", false), ("new", true)] {
        let mut w = FileWriter::new();
        w.write_dataset_i32("/g/inner", &[5], &[1]).expect("inner");
        w.create_hard_link("/galias", "/g").expect("alias");
        w.create_soft_link("/gsoft", "/g").expect("soft");
        if link_style {
            w.set_link_storage("/").expect("link storage");
        }
        let (path, bytes) = build(&mut w, &format!("group_link_{label}"));

        let f = oxih5::open(&path).expect("open");
        for name in ["g", "galias", "gsoft"] {
            let grp = f
                .group(name)
                .unwrap_or_else(|e| panic!("{label}/{name}: {e}"));
            assert_eq!(
                grp.datasets().expect("datasets"),
                vec!["inner".to_string()],
                "{label}/{name}"
            );
        }
        assert_eq!(
            f.dataset("galias/inner")
                .expect("through the alias")
                .as_i32()
                .expect("i32"),
            vec![5],
            "{label}"
        );

        // The alias counts against the group's own object header.
        let oxih5_core::Link::Hard { address } = root_link(&bytes, "g") else {
            panic!("g is a hard link");
        };
        assert_eq!(
            refcount(&bytes, address),
            2,
            "{label}: one name plus one alias"
        );
        let _ = std::fs::remove_file(&path);
    }
}

/// Group styles mix freely: an old-style group holding a new-style one holding
/// an old-style one, each with its own attributes, all reachable from the root.
#[test]
fn old_and_new_style_groups_nest_inside_each_other() {
    let mut w = FileWriter::new();
    w.write_dataset_i32("/plain/a", &[1], &[1]).expect("a");
    w.write_dataset_i32("/plain/linky/b", &[2], &[1])
        .expect("b");
    w.write_dataset_i32("/plain/linky/deeper/c", &[3], &[1])
        .expect("c");
    w.set_link_storage("/plain/linky").expect("link storage");
    w.create_external_link("/plain/linky/away", "peer.h5", "/x")
        .expect("ext");
    w.write_string_attr("/plain", "style", "symbol table")
        .expect("attr");
    w.write_string_attr("/plain/linky", "style", "link messages")
        .expect("attr");
    w.write_f64_attr("/plain/linky", "version", 2.0)
        .expect("attr");
    let (path, bytes) = build(&mut w, "mixed_styles");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("plain/a").expect("a").as_i32().expect("i32"),
        vec![1]
    );
    assert_eq!(
        f.dataset("plain/linky/b")
            .expect("b")
            .as_i32()
            .expect("i32"),
        vec![2]
    );
    assert_eq!(
        f.dataset("plain/linky/deeper/c")
            .expect("c")
            .as_i32()
            .expect("i32"),
        vec![3]
    );

    // The attributes ride the new-style group's object header beside its Link
    // messages, in the order the message list declares.
    let linky = f.header_addr_of("plain/linky").expect("linky address");
    let types = msg_types(&bytes, linky);
    assert_eq!(
        types,
        vec![0x0002, 0x000A, 0x0006, 0x0006, 0x0006, 0x000C, 0x000C],
        "link info, group info, three links, two attributes"
    );
    let attrs = f
        .group("plain/linky")
        .expect("linky")
        .attrs()
        .expect("attrs");
    let mut names: Vec<&str> = attrs.iter().map(|a| a.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["style", "version"]);
    let _ = std::fs::remove_file(&path);
}

/// Every dataset feature still works inside a **dense** group: chunking, a
/// filter pipeline, vlen strings, a custom fill value and a compact layout.
///
/// This is the test that would catch a link-storage change that quietly moved
/// a dataset's data area, since a dense group reserves file space between its
/// object header and its datasets.
#[test]
fn a_dense_group_holds_every_dataset_feature() {
    let mut w = FileWriter::new();
    w.create_group("/d").expect("d");
    w.set_link_storage("/d").expect("link storage");
    // Enough members to force dense storage, then the interesting ones.
    for i in 0..8i32 {
        w.write_dataset_i32(&format!("/d/pad{i}"), &[i], &[1])
            .expect("pad");
    }
    let big: Vec<f64> = (0..256).map(f64::from).collect();
    w.write_dataset_f64("/d/packed", &big, &[256])
        .expect("packed");
    w.set_shuffle("d/packed").expect("shuffle");
    w.set_deflate("d/packed", 6).expect("deflate");
    w.set_fletcher32("d/packed").expect("fletcher32");
    w.create_vlen_string_dataset("/d/names", &["alpha", "beta"])
        .expect("vlen strings");
    w.write_dataset_i32("/d/inline", &[7, 8, 9], &[3])
        .expect("inline");
    w.set_compact("d/inline").expect("compact");
    w.write_dataset_f64("/d/filled", &[1.0, 2.0], &[2])
        .expect("filled");
    w.set_fill_value_f64("d/filled", -999.0).expect("fill");
    w.write_string_attr("/d/packed", "units", "km")
        .expect("attr");
    let (path, bytes) = build(&mut w, "dense_features");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("d/packed")
            .expect("packed")
            .as_f64()
            .expect("f64"),
        big
    );
    assert_eq!(
        f.dataset_strings("d/names").expect("names"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
    assert_eq!(
        f.dataset("d/inline")
            .expect("inline")
            .as_i32()
            .expect("i32"),
        vec![7, 8, 9]
    );
    assert_eq!(
        f.dataset("d/filled")
            .expect("filled")
            .as_f64()
            .expect("f64"),
        vec![1.0, 2.0]
    );

    // The group really is dense, and the file ends where the superblock says.
    let d_addr = f.header_addr_of("d/pad0").expect("pad0");
    assert!(d_addr > 0);
    let msgs = header::parse_messages(&bytes, root_addr(&bytes)).expect("messages");
    let stab = msgs
        .iter()
        .find(|m| m.msg_type == 0x0011)
        .expect("root symbol table");
    let btree = u64::from_le_bytes(stab.data[0..8].try_into().expect("8 bytes"));
    let heap = u64::from_le_bytes(stab.data[8..16].try_into().expect("8 bytes"));
    let group::SymTabLink::Hard(group_addr) =
        group::find_entry(&bytes, btree, heap, "d").expect("d")
    else {
        panic!("d is a hard link");
    };
    let info = header::parse_messages(&bytes, group_addr)
        .expect("messages")
        .into_iter()
        .find(|m| m.msg_type == 0x0002)
        .expect("link info");
    assert_ne!(
        u64::from_le_bytes(info.data[2..10].try_into().expect("8 bytes")),
        u64::MAX,
        "twelve members means dense storage"
    );
    assert_eq!(
        u64::from_le_bytes(bytes[40..48].try_into().expect("8 bytes")),
        bytes.len() as u64,
        "the end-of-file address is where the file actually ends"
    );
    let _ = std::fs::remove_file(&path);
}
