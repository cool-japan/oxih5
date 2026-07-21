//! Pass one: where everything goes.
//!
//! [`plan_group`] walks the group tree and works out every absolute address in
//! the file before a single byte is written; [`super::build`] then writes at
//! those addresses.  Nothing is back-patched, which is what makes the two
//! passes cheap and what makes a one-byte disagreement between them fatal — see
//! [`super::check_size`].
//!
//! The plan mirrors the tree it was built from, node for node, and each node
//! holds a borrow of the descriptor it came from together with the resolved
//! attributes that were sized into its object header.  That is deliberate:
//! sizing and emission read the *same* `attrs` field, so an attribute that
//! reserved space and was then not emitted — or vice versa — is not
//! representable.
//!
//! Because a sub-group's plan is complete before its parent is emitted, the
//! parent's symbol table entries can simply *read* the child's B-tree and local
//! heap addresses out of the plan when they are needed.  There is no back-fill
//! into the output buffer anywhere in the writer.

use std::collections::HashMap;

use oxih5_core::OxiH5Error;

use super::btree_v1;
use super::elem::{self, ResolvedAttr};
use super::format;
use super::oh;
use super::payload::{self, Payload};
use super::tree::{DatasetDesc, GroupNode};
use super::{chunked, pad8};

/// Smallest local heap data segment libhdf5 is happy to see.
const MIN_HEAP_DATA_SIZE: usize = 88;

// ---------------------------------------------------------------------------
// Plan structures
// ---------------------------------------------------------------------------

/// A local heap data segment together with the name offsets into it.
pub(super) struct LocalHeap {
    /// Offset of each name within the segment, in insertion order.
    pub(super) name_offsets: Vec<u64>,
    /// The segment itself, already padded out to its allocated size.
    pub(super) data: Vec<u8>,
    /// Bytes of the segment in use; the free list starts here.
    pub(super) used: usize,
}

/// Where one dataset's structures live and how much room they were given.
pub(super) struct DatasetPlan<'a> {
    /// The dataset this plan was built from.
    pub(super) desc: &'a DatasetDesc,
    /// The dataset's attributes, resolved before its header was sized.
    pub(super) attrs: Vec<ResolvedAttr<'a>>,
    /// The dataset's data area, filtered if it carries a filter.
    ///
    /// Built *before* anything is sized, because a compressed dataset's length
    /// is not knowable any other way — see [`super::payload`].
    pub(super) payload: Payload<'a>,
    /// Address of the dataset's object header.
    pub(super) oh_addr: usize,
    /// Bytes reserved for that object header.
    pub(super) oh_size: usize,
    /// Address of the chunk index B-tree's **root** node, or 0 for a
    /// contiguous dataset.  For a multi-level tree that is the last level, not
    /// the first — the layout message must not point at a leaf.
    pub(super) btree_addr: usize,
    /// The planned chunk index; `None` for a contiguous dataset.
    pub(super) chunk_tree: Option<chunked::ChunkTree>,
    /// Address of the data area — the first chunk, for a chunked dataset.
    pub(super) data_addr: usize,
    /// Address of each chunk image; empty unless the dataset is chunked.
    pub(super) chunk_addrs: Vec<usize>,
    /// Chunk shape in elements; empty for a contiguous dataset.
    pub(super) chunk_shape: Vec<usize>,
    /// Chunk dimension vector as the layout message stores it (chunk shape,
    /// then the element size); empty for a contiguous dataset.
    pub(super) chunk_dims: Vec<u32>,
    /// Global-heap object index of each string, for a vlen-string dataset.
    pub(super) vlen_obj_idx: Vec<u32>,
}

/// What one symbol table link points at.
pub(super) enum LinkTarget {
    /// Index into the enclosing group's dataset plans.
    Dataset(usize),
    /// Index into the enclosing group's sub-group plans.
    Group(usize),
}

/// One link of a group's symbol table.
pub(super) struct Link<'a> {
    /// The link name — also the sort key, compared as raw bytes.
    pub(super) name: &'a str,
    /// Where the link points.
    pub(super) target: LinkTarget,
}

/// Where one group's local heap and symbol table live.
pub(super) struct SymPlan {
    /// Address of the local heap header.
    pub(super) heap_hdr_addr: usize,
    /// Address of the local heap data segment.
    pub(super) heap_data_addr: usize,
    /// The heap contents; one name offset per link, in sorted order.
    pub(super) heap: LocalHeap,
    /// The planned B-tree + symbol table node block.
    pub(super) table: btree_v1::SymTable,
}

/// Where one group's structures live, and the plans of everything beneath it.
pub(super) struct GroupPlan<'a> {
    /// The group this plan was built from.
    pub(super) node: &'a GroupNode,
    /// The group's attributes, resolved before its header was sized.
    pub(super) attrs: Vec<ResolvedAttr<'a>>,
    /// Address of the group's object header.
    pub(super) oh_addr: usize,
    /// Bytes reserved for that object header.
    pub(super) oh_size: usize,
    /// The group's local heap and symbol table.
    pub(super) sym: SymPlan,
    /// The group's links, in sorted order.
    pub(super) links: Vec<Link<'a>>,
    /// Plans for the group's datasets, in declaration order.
    pub(super) datasets: Vec<DatasetPlan<'a>>,
    /// Plans for the group's sub-groups, in creation order.
    pub(super) groups: Vec<GroupPlan<'a>>,
}

// ---------------------------------------------------------------------------
// Links and the local heap
// ---------------------------------------------------------------------------

/// Order a group's links the way libhdf5 orders them: ascending by raw name
/// bytes.
///
/// This is not cosmetic.  `H5G__node_found` binary-searches a symbol table node
/// with the names resolved through the local heap, so an unsorted node does not
/// fail loudly — it silently hides links, while `list(f.keys())` (a linear
/// walk) still reports them.  Every downstream offset depends on this order, so
/// it has to be settled before the local heap is built.
///
/// Datasets and sub-groups interleave here: a group's links are ordered by
/// name, never by kind.
fn sorted_links<'a>(datasets: &'a [DatasetDesc], groups: &'a [GroupNode]) -> Vec<Link<'a>> {
    let mut links: Vec<Link<'a>> = datasets
        .iter()
        .enumerate()
        .map(|(index, ds)| Link {
            name: ds.name.as_str(),
            target: LinkTarget::Dataset(index),
        })
        .chain(groups.iter().enumerate().map(|(index, grp)| Link {
            name: grp.name.as_str(),
            target: LinkTarget::Group(index),
        }))
        .collect();
    // Names are unique within a group, so the sort is total and an unstable
    // sort is deterministic.
    links.sort_unstable_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    links
}

/// Build a local heap data segment holding `names`.
///
/// Offset 0 of the segment is reserved for the free-block link, every name is
/// NUL-terminated and 8-byte aligned, and the tail carries a single free-list
/// entry covering whatever slack the allocation rounded up to.
fn build_local_heap<'a>(names: impl Iterator<Item = &'a str>) -> LocalHeap {
    let mut bytes: Vec<u8> = vec![0u8; 8];
    let mut name_offsets = Vec::new();
    for name in names {
        name_offsets.push(bytes.len() as u64);
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        let end = bytes.len();
        bytes.resize(pad8(end), 0);
    }

    let used = bytes.len();
    let size = pad8(used + 16).max(MIN_HEAP_DATA_SIZE);
    let mut data = vec![0u8; size];
    data[..used].copy_from_slice(&bytes);
    // Free list: link = 1 ("no next block"), then the free block's length.
    data[used..used + 8].copy_from_slice(&1u64.to_le_bytes());
    data[used + 8..used + 16].copy_from_slice(&((size - used) as u64).to_le_bytes());

    LocalHeap {
        name_offsets,
        data,
        used,
    }
}

/// Reserve the local heap and symbol table of a group holding `links`.
///
/// The order here is the whole point: the links are already sorted, so the heap
/// is built in sorted order, which makes the name offsets — and therefore the
/// B-tree keys derived from them — ascend with the names.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the group holds more links than the writer
/// will lay out.
fn plan_sym_table(links: &[Link<'_>], current: &mut usize) -> Result<SymPlan, OxiH5Error> {
    let heap = build_local_heap(links.iter().map(|link| link.name));
    let mut table = btree_v1::SymTable::plan(links.len())?;

    let btree_addr = *current;
    *current += table.btree_bytes();

    let heap_hdr_addr = *current;
    *current += format::HEAP_HEADER_SIZE;

    let heap_data_addr = *current;
    *current += heap.data.len();

    let snod_addr = *current;
    *current += table.snod_bytes();

    table.assign(btree_addr, snod_addr);
    Ok(SymPlan {
        heap_hdr_addr,
        heap_data_addr,
        heap,
        table,
    })
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Reserve space for one dataset, advancing `current` past everything it owns.
///
/// The order matters twice over.  The **payload is built first**: a compressed
/// dataset's on-disk length is not derivable from anything the descriptor
/// holds, so nothing can be sized until the filter has run.  And the **chunk
/// index precedes the data**, so that the B-tree's chunk addresses are already
/// settled when its keys are written — the writer back-patches nothing.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the chunk geometry is degenerate or beyond
/// what the writer emits, if a chunk extent does not fit its on-disk field, or
/// if the filter rejects its own parameters.
fn plan_dataset<'a>(
    ds: &'a DatasetDesc,
    current: &mut usize,
) -> Result<DatasetPlan<'a>, OxiH5Error> {
    // Attributes are resolved before anything is sized.  Object references are
    // the only value that depends on addresses, and only their *count* — fixed
    // here — affects size, so `fill_obj_refs` can patch the addresses in
    // afterwards without moving a single byte.
    let attrs = elem::resolve_attrs(&ds.attrs);
    let chunk_shape = chunked::chunk_shape_of(ds)?;
    let chunk_dims = if chunk_shape.is_empty() {
        Vec::new()
    } else {
        chunked::layout_chunk_dims(&chunk_shape, ds.elem_type.byte_size())?
    };
    let payload = payload::build(ds, &chunk_shape)?;

    let oh_addr = *current;
    let oh_size = oh::oh_size(&oh::dataset_oh_msgs(ds, &chunk_dims, &attrs));
    *current += oh_size;

    let chunk_tree = match &payload {
        Payload::Chunked(images) => {
            let mut tree = chunked::ChunkTree::plan(ds.shape.len(), images.len())?;
            tree.assign(*current);
            *current += tree.bytes();
            Some(tree)
        }
        _ => None,
    };
    let btree_addr = chunk_tree.as_ref().map_or(0, chunked::ChunkTree::root_addr);

    let data_addr = *current;
    let chunk_addrs = payload::reserve(&payload, current);

    Ok(DatasetPlan {
        desc: ds,
        attrs,
        payload,
        oh_addr,
        oh_size,
        btree_addr,
        chunk_tree,
        data_addr,
        chunk_addrs,
        chunk_shape,
        chunk_dims,
        vlen_obj_idx: Vec::new(),
    })
}

/// Reserve space for one group and everything beneath it.
///
/// The root group is planned by this very function, with `current` starting at
/// [`format::ROOT_OH_ADDR`]; nothing else distinguishes it.
///
/// The order is load-bearing and identical at every level:
///
/// 1. The group's own object header, sized from its resolved attributes.
/// 2. **Sort the links by name** — the sort key for the symbol table, settled
///    before anything derived from it exists.
/// 3. **Build the local heap** in that order, so name offsets ascend with names;
///    then **chunk into SNODs**, **plan the B-tree levels** and **assign
///    addresses**, each of which depends on the one before — all inside
///    [`plan_sym_table`].
/// 4. The group's datasets, then its sub-groups, recursively.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the group holds more links than the writer
/// will lay out, or if one of its datasets or sub-groups cannot be planned.
pub(super) fn plan_group<'a>(
    node: &'a GroupNode,
    current: &mut usize,
) -> Result<GroupPlan<'a>, OxiH5Error> {
    let attrs = elem::resolve_attrs(&node.attrs);

    let oh_addr = *current;
    let oh_size = oh::oh_size(&oh::group_oh_msgs(&attrs));
    *current += oh_size;

    let links = sorted_links(&node.datasets, &node.groups);
    let sym = plan_sym_table(&links, current)?;

    let mut datasets = Vec::with_capacity(node.datasets.len());
    for ds in &node.datasets {
        datasets.push(plan_dataset(ds, current)?);
    }

    // Recursion depth is bounded by `tree::MAX_PATH_DEPTH`: a group at depth n
    // can only have been created by a path carrying n components.
    let mut groups = Vec::with_capacity(node.groups.len());
    for child in &node.groups {
        groups.push(plan_group(child, current)?);
    }

    Ok(GroupPlan {
        node,
        attrs,
        oh_addr,
        oh_size,
        sym,
        links,
        datasets,
        groups,
    })
}

// ---------------------------------------------------------------------------
// Walks over the finished plan
// ---------------------------------------------------------------------------

impl<'a> GroupPlan<'a> {
    /// Record every object in this subtree under its path from the root.
    ///
    /// `prefix` is this group's own path, empty for the root — which therefore
    /// registers itself under `""`, the key a reference written as `"/"`
    /// normalises to.  Root-level objects get a bare name because their path
    /// *is* their name, so the two spellings an object reference may use are one
    /// key rather than two.
    pub(super) fn collect_addresses(&self, prefix: &str, map: &mut HashMap<String, u64>) {
        let child_path = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            }
        };

        map.insert(prefix.to_string(), self.oh_addr as u64);
        for ds in &self.datasets {
            map.insert(child_path(&ds.desc.name), ds.oh_addr as u64);
        }
        for grp in &self.groups {
            grp.collect_addresses(&child_path(&grp.node.name), map);
        }
    }

    /// Resolve every object reference in this subtree.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` naming the attribute and the target if a
    /// reference cannot be resolved.
    pub(super) fn fill_obj_refs(
        &mut self,
        path_to_addr: &HashMap<String, u64>,
    ) -> Result<(), OxiH5Error> {
        elem::fill_obj_refs(&mut self.attrs, path_to_addr)?;
        for ds in &mut self.datasets {
            elem::fill_obj_refs(&mut ds.attrs, path_to_addr)?;
        }
        for grp in &mut self.groups {
            grp.fill_obj_refs(path_to_addr)?;
        }
        Ok(())
    }

    /// Register every vlen-string dataset in this subtree with the file's one
    /// shared global heap collection, recording the object indices it hands back.
    pub(super) fn register_vlen_strings(&mut self, gcol: &mut oxih5_format::GlobalHeapWriter) {
        for ds in &mut self.datasets {
            ds.vlen_obj_idx = match &ds.desc.vlen_strings {
                Some(strings) => strings.iter().map(|s| gcol.write_string(s)).collect(),
                None => Vec::new(),
            };
        }
        for grp in &mut self.groups {
            grp.register_vlen_strings(gcol);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::elem::ElemType;
    use crate::write::tree::{Filter, Storage};

    fn dataset(name: &str) -> DatasetDesc {
        DatasetDesc {
            name: name.to_string(),
            raw: vec![0u8; 8],
            shape: vec![1],
            elem_type: ElemType::F64,
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: None,
        }
    }

    /// A compressible dataset of `bytes` bytes, chunked and deflated.
    fn deflated(name: &str, bytes: usize) -> DatasetDesc {
        DatasetDesc {
            name: name.to_string(),
            raw: vec![0x5Au8; bytes],
            shape: vec![bytes],
            elem_type: ElemType::U8,
            attrs: Vec::new(),
            storage: Storage::Chunked {
                chunk_shape: Vec::new(),
                unlimited_dim0: false,
            },
            filter: Some(Filter::Deflate { level: 6 }),
            vlen_strings: None,
        }
    }

    #[test]
    fn empty_local_heap_is_padded_to_the_minimum() {
        let heap = build_local_heap(std::iter::empty());
        assert!(heap.name_offsets.is_empty());
        assert_eq!(heap.used, 8);
        assert_eq!(heap.data.len(), MIN_HEAP_DATA_SIZE);
        // Free list starts at `used`: link = 1, then the free block length.
        assert_eq!(
            u64::from_le_bytes(heap.data[8..16].try_into().expect("8 bytes")),
            1
        );
        assert_eq!(
            u64::from_le_bytes(heap.data[16..24].try_into().expect("8 bytes")),
            (MIN_HEAP_DATA_SIZE - 8) as u64
        );
    }

    #[test]
    fn local_heap_names_are_nul_terminated_and_aligned() {
        let heap = build_local_heap(["lat", "longitude"].into_iter());
        assert_eq!(heap.name_offsets, vec![8, 16]);
        assert_eq!(&heap.data[8..12], b"lat\0");
        assert_eq!(&heap.data[16..26], b"longitude\0");
        // "longitude\0" is 10 bytes, padded up to 16.
        assert_eq!(heap.used, 32);
        assert_eq!(heap.used % 8, 0);
    }

    #[test]
    fn local_heap_grows_past_the_minimum_when_needed() {
        let names: Vec<String> = (0..16).map(|i| format!("dataset_number_{i:03}")).collect();
        let heap = build_local_heap(names.iter().map(String::as_str));
        assert!(heap.data.len() > MIN_HEAP_DATA_SIZE);
        assert!(heap.data.len() >= heap.used + 16);
        assert_eq!(heap.data.len() % 8, 0);
    }

    /// Datasets and sub-groups share one name ordering.
    #[test]
    fn links_interleave_datasets_and_groups_by_name() {
        let datasets = vec![dataset("zebra"), dataset("apple")];
        let groups = vec![GroupNode::new("mango"), GroupNode::new("banana")];
        let links = sorted_links(&datasets, &groups);
        let names: Vec<&str> = links.iter().map(|link| link.name).collect();
        assert_eq!(names, vec!["apple", "banana", "mango", "zebra"]);
        // The targets still index the *declaration* order they came from.
        assert!(matches!(links[0].target, LinkTarget::Dataset(1)));
        assert!(matches!(links[1].target, LinkTarget::Group(1)));
        assert!(matches!(links[2].target, LinkTarget::Group(0)));
        assert!(matches!(links[3].target, LinkTarget::Dataset(0)));
    }

    /// A compressed dataset reserves its **compressed** length, and reserves it
    /// before anything is written.
    ///
    /// This is the property that makes the two-pass writer work at all for a
    /// filtered dataset: the cursor advances past a number that cannot be
    /// derived from the descriptor, only from running the filter.  If pass one
    /// reserved `raw.len()` instead, every address after the dataset would be
    /// wrong by the compression ratio.
    #[test]
    fn a_compressed_dataset_reserves_its_compressed_length() {
        let mut root = GroupNode::new("");
        root.datasets.push(deflated("packed", 4096));
        root.datasets.push(dataset("after"));

        let mut current = format::ROOT_OH_ADDR;
        let plan = plan_group(&root, &mut current).expect("plan");
        let packed = &plan.datasets[0];
        let after = &plan.datasets[1];

        let Payload::Chunked(images) = &packed.payload else {
            panic!("a filtered dataset must be chunked");
        };
        assert_eq!(images.len(), 1, "one chunk covers the whole dataset");
        assert!(
            images[0].bytes.len() < 4096,
            "4 KiB of one repeated byte must compress: {} bytes",
            images[0].bytes.len()
        );
        assert_eq!(packed.chunk_addrs, vec![packed.data_addr]);
        assert_eq!(
            packed.payload.data_size(),
            crate::write::pad8(images[0].bytes.len()),
            "the reservation is the compressed length, not the raw one"
        );

        // The chunk index is reserved *between* the header and the data, so the
        // keys can name chunk addresses that are already final.
        assert!(packed.btree_addr > packed.oh_addr);
        assert!(packed.data_addr > packed.btree_addr);
        assert_eq!(
            packed.data_addr - packed.btree_addr,
            chunked::chunk_node_size(1),
            "the node occupies the full width libhdf5 reads"
        );

        // And the dataset after it starts past everything the first one owns —
        // which it would not if the reservation had used the raw length.
        assert!(after.oh_addr >= packed.data_addr + packed.payload.data_size());
        assert!(
            after.oh_addr < packed.data_addr + 4096,
            "reserving the *raw* 4096 bytes would push the next dataset out here"
        );
    }

    /// Nested groups are laid out depth-first, and every reserved region is
    /// disjoint and ascending.
    #[test]
    fn nested_groups_are_planned_without_overlap() {
        let mut root = GroupNode::new("");
        root.datasets.push(dataset("top"));
        let mut a = GroupNode::new("a");
        let mut b = GroupNode::new("b");
        b.datasets.push(dataset("deep"));
        a.groups.push(b);
        root.groups.push(a);

        let mut current = format::ROOT_OH_ADDR;
        let plan = plan_group(&root, &mut current).expect("plan");

        assert_eq!(plan.oh_addr, format::ROOT_OH_ADDR);
        let a_plan = &plan.groups[0];
        let b_plan = &a_plan.groups[0];
        assert!(a_plan.oh_addr > plan.oh_addr);
        assert!(b_plan.oh_addr > a_plan.oh_addr);
        assert!(b_plan.datasets[0].oh_addr > b_plan.oh_addr);
        assert!(current > b_plan.datasets[0].data_addr);

        // Each group owns its own symbol table, at its own address.
        assert_ne!(plan.sym.heap_hdr_addr, a_plan.sym.heap_hdr_addr);
        assert_ne!(a_plan.sym.heap_hdr_addr, b_plan.sym.heap_hdr_addr);
        assert_ne!(plan.sym.table.root_addr(), a_plan.sym.table.root_addr());
    }

    /// The address map keys every object by its path, root items by bare name,
    /// and the root group itself by the empty path.
    #[test]
    fn addresses_are_collected_by_full_path() {
        let mut root = GroupNode::new("");
        root.datasets.push(dataset("top"));
        let mut a = GroupNode::new("a");
        let mut b = GroupNode::new("b");
        b.datasets.push(dataset("deep"));
        a.groups.push(b);
        root.groups.push(a);

        let mut current = format::ROOT_OH_ADDR;
        let plan = plan_group(&root, &mut current).expect("plan");
        let mut map = HashMap::new();
        plan.collect_addresses("", &mut map);

        let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["", "a", "a/b", "a/b/deep", "top"]);
        assert_eq!(map.get(""), Some(&(format::ROOT_OH_ADDR as u64)));
        assert_eq!(
            map.get("a/b/deep").copied(),
            Some(plan.groups[0].groups[0].datasets[0].oh_addr as u64)
        );
    }
}
