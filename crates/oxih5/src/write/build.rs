//! Pass two: writing the plan out.
//!
//! [`super::plan`] worked out every absolute address; this module allocates one
//! buffer of the final size and writes each structure where it was told to.
//! Nothing is ever back-patched, which is fast but unforgiving: a size formula
//! that disagrees with its emitter by one byte makes two regions overlap.
//!
//! Two properties keep that safe:
//!
//! * Both passes size a structure through the *same* function — object headers
//!   through [`oh::oh_size`] over an [`oh::OhMsg`] list, chunk indices through
//!   [`chunked::chunk_btree_size`], symbol tables through
//!   [`btree_v1::SymTable`] — over the *same* resolved attributes, which live
//!   in the plan rather than beside it.
//! * Every emitter returns the number of bytes it wrote and every caller feeds
//!   that to [`check_size`], in release builds as well as debug.
//!
//! Groups are emitted recursively, exactly as they were planned, so the root
//! group takes the same path as every other group.  A sub-group's symbol table
//! entry in its parent carries the child's B-tree and local heap addresses in
//! its scratch pad; those are read straight out of the finished child plan, so
//! not even that back-reference needs a second visit to the buffer.

use std::collections::HashMap;

use oxih5_core::OxiH5Error;

use super::elem::VLEN_REF_SIZE;
use super::format::{self, SnodEntry};
use super::oh::{self, OhAddrs};
use super::payload::Payload;
use super::plan::{self, DatasetPlan, GroupPlan, Link, LinkTarget, SymPlan};
use super::tree::DatasetDesc;
use super::{check_size, chunked, narrow, FileWriter};

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

/// Copy `src` into `buf` at `addr`; returns bytes written.
fn copy_into(buf: &mut [u8], addr: usize, src: &[u8]) -> usize {
    buf[addr..addr + src.len()].copy_from_slice(src);
    src.len()
}

/// Addresses and per-object placement of the file's global-heap collections.
///
/// A vlen-string dataset's data area holds one 16-byte reference per string;
/// each reference must name the collection *address* and 1-based *object index*
/// of the collection that actually stores that string.  Because the writer
/// splits objects across [`oxih5_format::H5HG_MAXSIZE`]-capped collections,
/// that mapping is per-object rather than a single shared address.
#[derive(Clone, Copy)]
struct GheapLayout<'a> {
    /// Absolute file address of each collection, in ordinal order.
    collection_addrs: &'a [u64],
    /// Where each heap object landed, indexed by 0-based global ordinal.
    locations: &'a [oxih5_format::HeapObjectLocation],
}

/// Write a 16-byte HDF5 on-disk vlen reference at `offset`.
///
/// Matches `H5T__vlen_disk_write` with `size_of_offsets = 8`:
///
/// ```text
/// [0..4]   sequence length (u32 LE) = strlen (no NUL terminator)
/// [4..12]  global-heap collection address (u64 LE)
/// [12..16] global-heap object index (u32 LE)
/// ```
fn write_vlen_ref(buf: &mut [u8], offset: usize, seq_len: u32, obj_idx: u32, heap_addr: u64) {
    buf[offset..offset + 4].copy_from_slice(&seq_len.to_le_bytes());
    buf[offset + 4..offset + 12].copy_from_slice(&heap_addr.to_le_bytes());
    buf[offset + 12..offset + 16].copy_from_slice(&obj_idx.to_le_bytes());
}

/// Fill a vlen-string dataset's data area with global-heap references; returns
/// bytes written.
///
/// `obj_ordinals` holds the 1-based *global* heap ordinal of each string (as
/// handed back by [`oxih5_format::GlobalHeapWriter::write_string`]); the layout
/// resolves each ordinal to the collection address and local index that
/// actually store it.  The sequence length is `strlen` — no NUL terminator —
/// matching libhdf5 (`H5T__vlen_disk_write`).
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if an ordinal falls outside the heap layout,
/// which would mean pass one and pass two disagree about the global heap.
fn write_vlen_refs(
    buf: &mut [u8],
    base: usize,
    strings: &[String],
    obj_ordinals: &[u32],
    gheap: GheapLayout<'_>,
) -> Result<usize, OxiH5Error> {
    let mut wrote = 0usize;
    for (i, (s, &ordinal)) in strings.iter().zip(obj_ordinals.iter()).enumerate() {
        let loc = ordinal
            .checked_sub(1)
            .and_then(|z| gheap.locations.get(z as usize))
            .ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "internal writer error: vlen string ordinal {ordinal} has no global-heap location"
                ))
            })?;
        let heap_addr = gheap
            .collection_addrs
            .get(loc.collection as usize)
            .copied()
            .ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "internal writer error: global-heap collection {} was never placed",
                    loc.collection
                ))
            })?;
        write_vlen_ref(
            buf,
            base + i * VLEN_REF_SIZE,
            s.len() as u32,
            loc.index,
            heap_addr,
        );
        wrote += VLEN_REF_SIZE;
    }
    Ok(wrote)
}

/// Emit one dataset's data area, and the chunk index that addresses it.
///
/// Every branch reports what it wrote against what pass one reserved for it,
/// which for a chunked dataset means each chunk individually: a filter that
/// produced a different number of bytes here than it did during planning would
/// otherwise overwrite the chunk after it.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if any structure writes a different number of
/// bytes than pass one reserved for it, if the plan and the payload disagree
/// about how many chunks there are, or if a chunk is too long for the 32-bit
/// length field in its B-tree key.
fn emit_payload(
    buf: &mut [u8],
    plan: &DatasetPlan<'_>,
    gheap: GheapLayout<'_>,
) -> Result<(), OxiH5Error> {
    let ds: &DatasetDesc = plan.desc;
    match &plan.payload {
        Payload::Raw(bytes) => check_size(
            "dataset data area",
            copy_into(buf, plan.data_addr, bytes),
            ds.data_len(),
        ),

        Payload::VlenRefs { .. } => {
            let strings = ds.vlen_strings.as_deref().unwrap_or(&[]);
            check_size(
                "vlen reference area",
                write_vlen_refs(buf, plan.data_addr, strings, &plan.vlen_obj_idx, gheap)?,
                ds.data_len(),
            )
        }

        Payload::Chunked(images) => {
            if images.len() != plan.chunk_addrs.len() {
                return Err(OxiH5Error::Format(format!(
                    "internal writer error: dataset '{}' planned {} chunk addresses for {} chunks",
                    ds.name,
                    plan.chunk_addrs.len(),
                    images.len()
                )));
            }

            let mut entries = Vec::with_capacity(images.len());
            for (image, &addr) in images.iter().zip(&plan.chunk_addrs) {
                check_size(
                    "dataset chunk",
                    copy_into(buf, addr, &image.bytes),
                    image.bytes.len(),
                )?;
                entries.push(chunked::ChunkEntry {
                    offsets: image.offsets.clone(),
                    addr: addr as u64,
                    nbytes: narrow("chunk byte size", image.bytes.len())?,
                    // Every chunk went through the whole pipeline; no filter
                    // was skipped, so no bit of the mask is set.
                    filter_mask: 0,
                });
            }

            let tree = plan.chunk_tree.as_ref().ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "internal writer error: dataset '{}' has chunks but no planned index",
                    ds.name
                ))
            })?;
            check_size(
                "chunk index B-tree",
                tree.write(
                    buf,
                    &entries,
                    &chunked::end_offsets(&ds.shape, &plan.chunk_shape),
                )?,
                tree.bytes(),
            )
        }
    }
}

/// Emit one dataset: object header, then its data and chunk index.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if any structure writes a different number of
/// bytes than pass one reserved for it.
fn emit_dataset(
    buf: &mut [u8],
    plan: &DatasetPlan<'_>,
    gheap: GheapLayout<'_>,
) -> Result<(), OxiH5Error> {
    let msgs = oh::dataset_oh_msgs(plan.desc, &plan.chunk_dims, &plan.attrs);
    let addrs = OhAddrs {
        btree: plan.btree_addr as u64,
        heap: 0,
        data: plan.data_addr as u64,
    };
    check_size(
        "dataset object header",
        oh::write_oh(buf, plan.oh_addr, &msgs, &addrs)?,
        plan.oh_size,
    )?;

    emit_payload(buf, plan, gheap)
}

/// Resolve a group's sorted links into symbol table entries.
///
/// A sub-group link carries `cache_type = 1` and the child's own B-tree root
/// and local heap in the entry's scratch pad.  Those addresses are read out of
/// the child's finished plan, never patched into the buffer after the fact.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if a link points at an object that was never
/// planned, which would otherwise emit a symbol table entry addressing byte 0.
fn snod_entries(
    links: &[Link<'_>],
    name_offsets: &[u64],
    datasets: &[DatasetPlan<'_>],
    groups: &[GroupPlan<'_>],
) -> Result<Vec<SnodEntry>, OxiH5Error> {
    let missing = |kind: &str, name: &str, index: usize| {
        OxiH5Error::Format(format!(
            "internal writer error: link '{name}' points at {kind} {index}, which was not planned"
        ))
    };

    links
        .iter()
        .zip(name_offsets)
        .map(|(link, &name_offset)| {
            let (oh_addr, group_cache) = match link.target {
                LinkTarget::Dataset(index) => {
                    let plan = datasets
                        .get(index)
                        .ok_or_else(|| missing("dataset", link.name, index))?;
                    (plan.oh_addr as u64, None)
                }
                LinkTarget::Group(index) => {
                    let plan = groups
                        .get(index)
                        .ok_or_else(|| missing("group", link.name, index))?;
                    (
                        plan.oh_addr as u64,
                        Some((
                            plan.sym.table.root_addr() as u64,
                            plan.sym.heap_hdr_addr as u64,
                        )),
                    )
                }
            };
            Ok(SnodEntry {
                name_offset,
                oh_addr,
                group_cache,
            })
        })
        .collect()
}

/// Emit one group's local heap, symbol table nodes, and symbol table B-tree.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if any structure writes a different number of
/// bytes than pass one reserved for it.
fn emit_sym_table(
    buf: &mut [u8],
    what: &str,
    sym: &SymPlan,
    entries: &[SnodEntry],
) -> Result<(), OxiH5Error> {
    check_size(
        &format!("{what} local heap header"),
        format::write_local_heap(
            buf,
            sym.heap_hdr_addr,
            sym.heap_data_addr,
            sym.heap.data.len(),
            sym.heap.used,
        ),
        format::HEAP_HEADER_SIZE,
    )?;
    check_size(
        &format!("{what} local heap data"),
        copy_into(buf, sym.heap_data_addr, &sym.heap.data),
        sym.heap.data.len(),
    )?;
    check_size(
        &format!("{what} symbol table"),
        sym.table.write(buf, entries)?,
        sym.table.btree_bytes() + sym.table.snod_bytes(),
    )
}

/// Emit one group: object header, symbol table, datasets, then sub-groups.
///
/// The root group goes through here too; it differs only in that its address is
/// fixed and mirrored into the superblock by [`build_bytes`].
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if any structure writes a different number of
/// bytes than pass one reserved for it.
fn emit_group(
    buf: &mut [u8],
    plan: &GroupPlan<'_>,
    gheap: GheapLayout<'_>,
) -> Result<(), OxiH5Error> {
    let what = if plan.node.name.is_empty() {
        "root group"
    } else {
        "sub-group"
    };

    let msgs = oh::group_oh_msgs(&plan.attrs);
    let addrs = OhAddrs {
        btree: plan.sym.table.root_addr() as u64,
        heap: plan.sym.heap_hdr_addr as u64,
        data: 0,
    };
    check_size(
        &format!("{what} object header"),
        oh::write_oh(buf, plan.oh_addr, &msgs, &addrs)?,
        plan.oh_size,
    )?;

    let entries = snod_entries(
        &plan.links,
        &plan.sym.heap.name_offsets,
        &plan.datasets,
        &plan.groups,
    )?;
    emit_sym_table(buf, what, &plan.sym, &entries)?;

    for ds in &plan.datasets {
        emit_dataset(buf, ds, gheap)?;
    }
    for grp in &plan.groups {
        emit_group(buf, grp, gheap)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Serialize `writer` into a complete HDF5 file image.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if a value overflows an on-disk field, if an
/// object reference names nothing in the file, or if any structure writes a
/// different number of bytes than was reserved for it — see [`check_size`].
pub(super) fn build_bytes(writer: &FileWriter) -> Result<Vec<u8>, OxiH5Error> {
    // -- Pass one: lay the file out, root group included. --------------------
    let mut current = format::ROOT_OH_ADDR;
    let mut root_plan = plan::plan_group(writer.root_node(), &mut current)?;

    // -- Now that object-header addresses exist, fill in object references. ---
    // This runs BEFORE the global-heap block on purpose: a vlen-object-reference
    // attribute's heap payload *is* its resolved target addresses, so those must
    // be settled before `register_vlen_objref_attrs` serializes them.
    let mut path_to_addr: HashMap<String, u64> = HashMap::new();
    root_plan.collect_addresses("", &mut path_to_addr);
    root_plan.fill_obj_refs(&path_to_addr)?;

    // -- Global heap collections, shared by every vlen-string dataset. --------
    // The writer splits objects into libhdf5-conformant collections (each padded
    // to at least 4096 bytes and capped at 65536); each vlen reference names the
    // collection that actually holds its string, so the collections are laid out
    // at consecutive addresses and their placement recorded per object.
    let mut gcol = oxih5_format::GlobalHeapWriter::new();
    root_plan.register_vlen_strings(&mut gcol);
    // vlen-object-reference attributes (e.g. netCDF-4 DIMENSION_LIST) store their
    // resolved target addresses as one heap object per sequence, in the same
    // shared collection set.
    root_plan.register_vlen_objref_attrs(&mut gcol);
    let (gcol_collections, gcol_locations) = if gcol.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        gcol.build_collections()
    };
    let mut collection_addrs: Vec<u64> = Vec::with_capacity(gcol_collections.len());
    let mut gcol_cursor = current;
    for collection in &gcol_collections {
        collection_addrs.push(gcol_cursor as u64);
        gcol_cursor += collection.len();
    }
    let eof_addr = gcol_cursor;
    // Resolve each vlen-objref sequence's heap ordinal into a concrete
    // (collection address, local index) now that the collections are laid out.
    root_plan.fill_vlen_objref_locs(&collection_addrs, &gcol_locations);
    let gheap = GheapLayout {
        collection_addrs: &collection_addrs,
        locations: &gcol_locations,
    };

    // -- Pass two: emit. -----------------------------------------------------
    let mut buf = vec![0u8; eof_addr];

    // The root group's symbol table is mirrored into the superblock's root
    // entry; every other group's lives only in its parent's scratch pad.
    let prefix = format::write_signature(&mut buf)
        + format::write_superblock(
            &mut buf,
            root_plan.sym.table.root_addr(),
            root_plan.sym.heap_hdr_addr,
            eof_addr as u64,
        );
    check_size("file prefix", prefix, format::ROOT_OH_ADDR)?;

    emit_group(&mut buf, &root_plan, gheap)?;

    for (collection, &addr) in gcol_collections.iter().zip(&collection_addrs) {
        check_size(
            "global heap collection",
            copy_into(&mut buf, addr as usize, collection),
            collection.len(),
        )?;
    }

    check_size("file image", eof_addr, buf.len())?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::btree_v1::probe;
    use crate::File;

    #[test]
    fn check_size_rejects_a_mismatch() {
        assert!(check_size("thing", 8, 8).is_ok());
        let err = check_size("thing", 9, 8).expect_err("must reject");
        assert!(
            format!("{err}").contains("9 bytes into 8 reserved"),
            "{err}"
        );
    }

    // -----------------------------------------------------------------------
    // W1b: nested groups
    // -----------------------------------------------------------------------

    /// A group three levels down must be reachable both ways.
    ///
    /// `File::dataset` walks the whole path in one go; `File::group` descends
    /// one symbol table at a time, re-reading each child's cached B-tree and
    /// heap out of the parent's scratch pad.  A sub-group whose scratch pad is
    /// wrong still resolves through the first and not the second.
    #[test]
    fn w1b_three_level_nesting_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1b_three_level.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("/a/b/c/ds", &[1.0, 2.0, 3.0], &[3])
            .expect("nested dataset");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let by_path = f.dataset("a/b/c/ds").expect("dataset by path");
        assert_eq!(by_path.as_f64().expect("as_f64"), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            f.dataset("/a/b/c/ds").expect("absolute path").shape,
            vec![3usize]
        );

        // Descend one group at a time.
        let a = f.group("a").expect("group a");
        assert_eq!(a.groups().expect("a groups"), vec!["b".to_string()]);
        let b = a.group("b").expect("group b");
        let c = b.group("c").expect("group c");
        assert_eq!(c.datasets().expect("c datasets"), vec!["ds".to_string()]);
        assert_eq!(c.dataset("ds").expect("c/ds").shape, vec![3usize]);

        // And by path in one hop.
        let c_direct = f.group("/a/b/c").expect("group by path");
        assert_eq!(
            c_direct.dataset("ds").expect("ds").as_f64().expect("f64"),
            vec![1.0, 2.0, 3.0]
        );
    }

    /// Writing to a path creates the groups above it, like h5py's
    /// `create_intermediate_group=True`.
    #[test]
    fn w1b_intermediate_groups_auto_created() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1b_intermediate.h5");
        let mut w = FileWriter::new();
        // Never mentions "obs" or "obs/2024" as groups.
        w.write_dataset_i32("obs/2024/counts", &[1i32, 2], &[2])
            .expect("counts");
        w.write_dataset_i32("obs/2023/counts", &[3i32, 4], &[2])
            .expect("older counts");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let root = f.root().expect("root");
        assert_eq!(root.groups().expect("root groups"), vec!["obs".to_string()]);
        let obs = f.group("obs").expect("obs");
        let mut years = obs.groups().expect("obs groups");
        years.sort();
        assert_eq!(years, vec!["2023".to_string(), "2024".to_string()]);
        assert_eq!(
            f.dataset("obs/2024/counts")
                .expect("counts")
                .as_i32()
                .expect("as_i32"),
            vec![1, 2]
        );
    }

    /// A dataset may not take a name a group already holds.
    ///
    /// Both become links in one symbol table, where a duplicate name is not
    /// representable.  `create_group` checked both directions; the three
    /// dataset entry points checked only against other datasets, so a dataset
    /// could shadow a group.
    #[test]
    fn w1b_dataset_cannot_shadow_group_name() {
        let mut w = FileWriter::new();
        w.create_group("shared").expect("group");
        assert!(
            w.write_dataset_f64("shared", &[1.0], &[1]).is_err(),
            "a dataset must not take a group's name"
        );
        assert!(
            w.create_vlen_string_dataset("shared", &["x"]).is_err(),
            "a vlen-string dataset must not take a group's name"
        );
        let dtype = oxih5_core::Dtype::Float {
            size: 8,
            order: oxih5_core::ByteOrder::Little,
        };
        assert!(
            w.create_dataset("shared", &[1], &dtype).is_err(),
            "create_dataset must not take a group's name"
        );
        assert!(
            w.create_dataset_unlimited("shared", &[1], &[1], &dtype, &[0u8; 8])
                .is_err(),
            "an unlimited dataset must not take a group's name"
        );

        // The same protection inside a sub-group, not only at the root.
        w.create_group("outer/inner").expect("nested group");
        assert!(
            w.write_dataset_i32("outer/inner", &[1], &[1]).is_err(),
            "a nested dataset must not take a nested group's name"
        );
    }

    /// And the reverse: a group may not take a dataset's name.
    #[test]
    fn w1b_group_cannot_shadow_dataset_name() {
        let mut w = FileWriter::new();
        w.write_dataset_f64("taken", &[1.0], &[1]).expect("dataset");
        assert!(w.create_group("taken").is_err(), "group over dataset");
        // Nor may a path walk *through* a dataset.
        assert!(
            w.write_dataset_f64("taken/deeper", &[1.0], &[1]).is_err(),
            "a dataset cannot also be a group"
        );
        assert!(w.create_group("taken/deeper").is_err(), "same for groups");
    }

    /// Object references resolve across groups, by full path.
    #[test]
    fn w1b_objref_across_groups_resolves() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1b_objref.h5");
        let mut w = FileWriter::new();
        w.write_dataset_i32("dims/lat", &[0i32, 1, 2], &[3])
            .expect("lat");
        w.write_dataset_f64("root_scale", &[1.0], &[1])
            .expect("root_scale");
        w.write_dataset_f64("data/temp", &[0.0; 3], &[3])
            .expect("temp");
        // One nested target by path, one root target by bare name.
        w.write_obj_ref_list_attr("data/temp", "DIMENSION_LIST", &["/dims/lat", "root_scale"])
            .expect("DIMENSION_LIST");
        // A reference to a *group* is legal too.
        w.write_obj_ref_list_attr("data/temp", "GROUP_REF", &["dims"])
            .expect("GROUP_REF");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let lat_addr = f.header_addr_of("dims/lat").expect("lat addr");
        let scale_addr = f.header_addr_of("root_scale").expect("scale addr");
        let attrs = f.attr_views("data/temp").expect("temp attrs");
        let dl = attrs
            .iter()
            .find(|a| a.name() == "DIMENSION_LIST")
            .expect("DIMENSION_LIST");
        assert_eq!(
            dl.as_object_refs().expect("refs"),
            vec![lat_addr, scale_addr]
        );

        let gr = attrs
            .iter()
            .find(|a| a.name() == "GROUP_REF")
            .expect("GROUP_REF");
        let group_refs = gr.as_object_refs().expect("group refs");
        assert_eq!(group_refs.len(), 1);
        assert_ne!(group_refs[0], u64::MAX, "group ref must be resolved");
    }

    /// A target that names nothing is an error at build time.
    ///
    /// It used to become `u64::MAX`, producing a file that opened cleanly and
    /// carried a dangling reference — the mistake surfaced, if ever, at
    /// dereference time and named nothing useful.
    #[test]
    fn w1b_unresolvable_objref_is_an_error() {
        let mut w = FileWriter::new();
        w.write_dataset_f64("data/temp", &[0.0], &[1])
            .expect("temp");
        w.write_obj_ref_list_attr("data/temp", "DIMENSION_LIST", &["no_such_thing"])
            .expect("attribute is accepted");
        let err = w.build_to_vec().expect_err("build must refuse");
        let msg = format!("{err}");
        assert!(msg.contains("no_such_thing"), "must name the target: {msg}");

        // A dataset that exists but only in another group is equally unresolved
        // when referred to by bare name.
        let mut w = FileWriter::new();
        w.write_dataset_f64("grp/lat", &[0.0], &[1]).expect("lat");
        w.write_dataset_f64("temp", &[0.0], &[1]).expect("temp");
        assert!(
            w.write_obj_ref_list_attr("temp", "DIMENSION_LIST", &["lat"])
                .is_ok(),
            "the attribute is accepted; resolution happens at build time"
        );
        assert!(
            w.build_to_vec().is_err(),
            "a nested target must be named by its path"
        );
    }

    /// Enough sub-groups at one level to force multiple SNODs *inside* a
    /// sub-group, plus enough depth to stack them.
    ///
    /// The symbol-table work chunked links into 328-byte nodes and grew a
    /// B-tree over them; nesting has to compose with that at every level, not
    /// only at the root, and a sub-group's cached B-tree address in its
    /// parent's scratch pad has to point at the *root* of that tree rather than
    /// at its first node.
    #[test]
    fn w1b_deep_nesting_many_groups() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1b_deep_nesting.h5");
        let mut w = FileWriter::new();
        // 20 sub-groups of "outer" — three SNODs' worth — each with a dataset,
        // and one of them nested three deeper again.
        for i in 0..20usize {
            w.write_dataset_i32(&format!("outer/g{i:02}/value"), &[i as i32], &[1])
                .expect("nested value");
        }
        w.write_dataset_i32("outer/g07/deeper/still/here", &[99i32], &[1])
            .expect("deep");
        w.build(&tmp).expect("build");

        let bytes = std::fs::read(&tmp).expect("read back");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let outer = f.group("outer").expect("outer");
        let names = outer.groups().expect("outer groups");
        assert_eq!(names.len(), 20, "{names:?}");

        for i in 0..20usize {
            let path = format!("outer/g{i:02}/value");
            let ds = f.dataset(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert_eq!(ds.as_i32().expect("as_i32"), vec![i as i32]);
        }
        assert_eq!(
            f.dataset("outer/g07/deeper/still/here")
                .expect("deep")
                .as_i32()
                .expect("as_i32"),
            vec![99]
        );

        // "outer" really does span more than one symbol table node: 20 links at
        // eight per node is three.
        let (root_btree, root_heap) = probe::root_symbol_table(&bytes);
        let mut outer_btree = None;
        for snod in probe::collect_snods(&bytes, root_btree) {
            for i in 0..probe::le_u16(&bytes, snod + 6) as usize {
                let ste = snod + 8 + i * 40;
                if probe::heap_name(&bytes, root_heap, probe::le_u64(&bytes, ste)) == "outer" {
                    outer_btree = Some(probe::le_u64(&bytes, ste + 24) as usize);
                }
            }
        }
        let outer_btree = outer_btree.expect("outer must be a cache_type 1 entry");
        assert_eq!(
            probe::collect_snods(&bytes, outer_btree).len(),
            3,
            "20 links must span three symbol table nodes"
        );
    }

    // -----------------------------------------------------------------------
    // W1c: attributes on groups, on the root group, and array-valued
    // -----------------------------------------------------------------------

    /// Decode a 1-D little-endian `f64` attribute payload.
    fn f64_values(view: &crate::AttrView<'_>) -> Vec<f64> {
        view.attr
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()
    }

    /// Decode a 1-D little-endian `i64` attribute payload.
    fn i64_values(view: &crate::AttrView<'_>) -> Vec<i64> {
        view.attr
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()
    }

    /// A sub-group at any depth carries attributes of every scalar type.
    ///
    /// A sub-group's object header was a hard-coded 40 bytes — a 16-byte prefix
    /// and one symbol table message — so there was nowhere for an attribute to
    /// go, and only the root group could carry any.
    #[test]
    fn w1c_subgroup_attrs_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1c_subgroup_attrs.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("a/b/inner", &[1.0], &[1])
            .expect("inner");
        w.write_string_attr("a/b", "title", "nested group")
            .expect("title");
        w.write_f64_attr("a/b", "scale", 0.25).expect("scale");
        w.write_i64_attr("a/b", "count", i64::MAX).expect("count");
        w.write_i32_attr("a/b", "_Netcdf4Dimid", -3).expect("dimid");
        // The group one level up gets one of its own, so a mix-up between the
        // two would show.
        w.write_string_attr("/a", "title", "outer group")
            .expect("outer title");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let b = f.group("a/b").expect("group a/b");
        let attrs = b.attr_views().expect("attr_views");
        assert_eq!(attrs.len(), 4, "a/b attribute count");
        let by_name = |name: &str| {
            attrs
                .iter()
                .find(|a| a.name() == name)
                .unwrap_or_else(|| panic!("{name} missing from a/b"))
        };
        assert_eq!(
            by_name("title").as_str_fixed().expect("title"),
            "nested group"
        );
        assert_eq!(by_name("scale").as_f64(), Some(0.25));
        assert_eq!(by_name("count").as_i64(), Some(i64::MAX));
        assert_eq!(by_name("_Netcdf4Dimid").as_i64(), Some(-3));

        // The parent group's attribute is its own, and the group still works as
        // a group: its symbol table message survives the extra messages.
        let a = f.group("a").expect("group a");
        let a_attrs = a.attr_views().expect("a attr_views");
        assert_eq!(a_attrs.len(), 1);
        assert_eq!(
            a_attrs[0].as_str_fixed().expect("outer title"),
            "outer group"
        );
        assert_eq!(a.groups().expect("a groups"), vec!["b".to_string()]);
        assert_eq!(
            f.dataset("a/b/inner").expect("inner").shape,
            vec![1usize],
            "the group's datasets are still reachable"
        );
    }

    /// The root group takes non-string attributes.
    ///
    /// `write_root_str_attr` was the only way to attach anything to the root,
    /// and it took a `&str`; an integer or float global attribute could not be
    /// written at all.
    #[test]
    fn w1c_root_group_non_string_attrs() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1c_root_attrs.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("data", &[1.0], &[1]).expect("data");
        w.write_root_str_attr("Conventions", "CF-1.8");
        w.write_i64_attr("/", "total_records", 9_000_000_000)
            .expect("total_records");
        w.write_f64_attr("/", "resolution", 0.125)
            .expect("resolution");
        w.write_i32_attr("/", "_Netcdf4Dimid", 4).expect("dimid");
        // The empty path names the root group too.
        w.write_string_attr("", "history", "created by oxih5")
            .expect("history");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let root = f.root().expect("root");
        let attrs = root.attr_views().expect("root attr_views");
        assert_eq!(attrs.len(), 5, "root attribute count");
        let by_name = |name: &str| {
            attrs
                .iter()
                .find(|a| a.name() == name)
                .unwrap_or_else(|| panic!("{name} missing from the root group"))
        };
        assert_eq!(
            by_name("Conventions").as_str_fixed().expect("Conventions"),
            "CF-1.8"
        );
        assert_eq!(by_name("total_records").as_i64(), Some(9_000_000_000));
        assert_eq!(by_name("resolution").as_f64(), Some(0.125));
        assert_eq!(by_name("_Netcdf4Dimid").as_i64(), Some(4));
        assert_eq!(
            by_name("history").as_str_fixed().expect("history"),
            "created by oxih5"
        );

        // Root attributes must not disturb the root's own symbol table.
        assert_eq!(f.dataset("data").expect("data").shape, vec![1usize]);
    }

    /// `write_root_str_attr` is now literally `write_string_attr("/", …)`.
    #[test]
    fn w1c_root_str_attr_is_a_root_group_attr() {
        let mut a = FileWriter::new();
        a.write_dataset_u8("d", &[1u8], &[1]).expect("d");
        a.write_root_str_attr("k", "v");

        let mut b = FileWriter::new();
        b.write_dataset_u8("d", &[1u8], &[1]).expect("d");
        b.write_string_attr("/", "k", "v").expect("k");

        assert_eq!(
            a.build_to_vec().expect("a"),
            b.build_to_vec().expect("b"),
            "the two spellings must produce the same file"
        );
    }

    /// Array-valued attributes: `i64[]`, `f64[]` and `str[]`.
    ///
    /// Only scalars were supported, so a `valid_range` or a `flag_meanings`
    /// list — both routine in CF-conventions data — had no representation.
    #[test]
    fn w1c_array_valued_attrs() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1c_array_attrs.h5");
        let ints = [i64::MIN, 0, i64::MAX];
        let floats = [-1.5f64, 0.0, 2.25];
        let labels = ["low", "medium", "high-and-longest"];

        let mut w = FileWriter::new();
        w.write_dataset_f64("ds", &[1.0], &[1]).expect("ds");
        w.write_i64_array_attr("ds", "valid_range", &ints)
            .expect("valid_range");
        w.write_f64_array_attr("ds", "bounds", &floats)
            .expect("bounds");
        w.write_string_array_attr("ds", "flag_meanings", &labels)
            .expect("flag_meanings");
        // The same three kinds on a nested group and on the root group.
        w.create_group("grp/inner").expect("grp/inner");
        w.write_f64_array_attr("grp/inner", "corners", &floats)
            .expect("corners");
        w.write_string_array_attr("/", "sources", &["a", "bb"])
            .expect("sources");
        // Degenerate lengths must not reserve a byte they do not write.
        w.write_i64_array_attr("ds", "empty_ints", &[])
            .expect("empty_ints");
        w.write_string_array_attr("ds", "empty_strs", &[])
            .expect("empty_strs");
        w.write_string_array_attr("ds", "all_empty", &["", ""])
            .expect("all_empty");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let attrs = f.attr_views("ds").expect("ds attrs");
        let by_name = |name: &str| {
            attrs
                .iter()
                .find(|a| a.name() == name)
                .unwrap_or_else(|| panic!("{name} missing from ds"))
        };

        let range = by_name("valid_range");
        assert_eq!(range.shape(), vec![3u64], "valid_range is 1-D of 3");
        assert!(!range.is_scalar());
        assert_eq!(i64_values(range), ints);

        let bounds = by_name("bounds");
        assert_eq!(bounds.shape(), vec![3u64]);
        assert_eq!(f64_values(bounds), floats);

        let meanings = by_name("flag_meanings");
        assert_eq!(meanings.shape(), vec![3u64]);
        assert_eq!(
            meanings.as_strings().expect("as_strings"),
            labels.iter().map(|s| (*s).to_string()).collect::<Vec<_>>()
        );

        assert_eq!(by_name("empty_ints").shape(), vec![0u64]);
        assert!(i64_values(by_name("empty_ints")).is_empty());
        assert_eq!(by_name("empty_strs").shape(), vec![0u64]);
        assert_eq!(
            by_name("all_empty").as_strings().expect("all_empty"),
            vec![String::new(), String::new()]
        );

        // …on a nested group…
        let inner = f.group("grp/inner").expect("grp/inner");
        let inner_attrs = inner.attr_views().expect("inner attrs");
        assert_eq!(inner_attrs.len(), 1);
        assert_eq!(f64_values(&inner_attrs[0]), floats);

        // …and on the root group.
        let root = f.root().expect("root");
        let root_attrs = root.attr_views().expect("root attrs");
        assert_eq!(
            root_attrs[0].as_strings().expect("sources"),
            vec!["a".to_string(), "bb".to_string()]
        );
    }
}
