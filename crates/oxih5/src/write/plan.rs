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
use super::btree_v2::NameIndexWriter;
use super::elem::{self, ResolvedAttr};
use super::format;
use super::fractal_heap::FractalHeapWriter;
use super::link::{LinkInfo, LinkMsg, LinkValue};
use super::oh;
use super::payload::{self, Payload};
use super::tree::{DatasetDesc, GroupNode, LinkKind};
use super::{chunked, pad8};

/// Smallest local heap data segment libhdf5 is happy to see.
const MIN_HEAP_DATA_SIZE: usize = 88;

/// Most links a new-style group keeps in its own object header before moving
/// them into a fractal heap.
///
/// libhdf5's default `max_compact` for a group creation property list.  Below
/// it the links are object-header messages a reader finds by walking the
/// header; at it and above, a fractal heap plus a version-2 name index.
const MAX_COMPACT_LINKS: usize = 8;

/// The "undefined address" sentinel a compact group's Link Info message carries
/// in place of a fractal heap and a name index.
const UNDEFINED_ADDRESS: u64 = u64::MAX;

// ---------------------------------------------------------------------------
// Plan structures
// ---------------------------------------------------------------------------

/// A local heap data segment together with the offsets into it.
pub(super) struct LocalHeap {
    /// Offset of each link's name within the segment, in sorted-link order.
    pub(super) name_offsets: Vec<u64>,
    /// Offset of each **soft link's value** within the segment, in the same
    /// order; `None` for every link that is not a soft link.
    ///
    /// An old-style symbol table has nowhere else to put a soft link's target:
    /// the entry itself carries only a 4-byte offset into this very heap, so
    /// the path is interned here beside the names.
    pub(super) value_offsets: Vec<Option<u64>>,
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
    /// Global-heap object index of each string, for a vlen-string dataset, or
    /// of each sequence, for a vlen-sequence dataset.
    pub(super) vlen_obj_idx: Vec<u32>,
    /// Object header reference count: one, plus one per hard alias that names
    /// this dataset.  Filled by [`GroupPlan::apply_refcounts`].
    pub(super) refcount: u32,
}

/// What one of a group's links points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LinkTarget {
    /// Index into the enclosing group's dataset plans.
    Dataset(usize),
    /// Index into the enclosing group's sub-group plans.
    Group(usize),
    /// Index into the enclosing group node's [`super::tree::LinkDesc`] list —
    /// a link that owns no object: a hard alias, a soft link or an external
    /// link.
    Link(usize),
}

/// One member of a group, in whichever encoding the group uses.
pub(super) struct Link<'a> {
    /// The link name — also the sort key, compared as raw bytes.
    pub(super) name: &'a str,
    /// Where the link points.
    pub(super) target: LinkTarget,
    /// Position in the group's creation order.
    pub(super) creation_order: u64,
}

/// One link of a **new-style** group, with the message that encodes it.
///
/// The message is built during the layout pass, when its *length* is already
/// final but a hard link's target address is not; [`GroupPlan::patch_links`]
/// fills the addresses in afterwards, which cannot change any length because
/// every hard-link value is exactly eight bytes.
pub(super) struct PlannedLink {
    /// The encoded link, address placeholder included.
    pub(super) msg: LinkMsg,
    /// Where the link points, for the address-patching pass.
    pub(super) target: LinkTarget,
}

/// A new-style group's link storage, compact or dense.
pub(super) struct LinkStorage {
    /// Every member as a link message, in emission order.
    pub(super) links: Vec<PlannedLink>,
    /// The Link Info message's own fields.
    pub(super) info: LinkInfo,
    /// Set when the links live in a fractal heap rather than in the header.
    pub(super) dense: Option<DenseLinks>,
}

impl LinkStorage {
    /// The links this group carries as object-header messages: all of them for
    /// a compact group, none for a dense one.
    pub(super) fn header_links(&self) -> &[PlannedLink] {
        match self.dense {
            Some(_) => &[],
            None => &self.links,
        }
    }

    /// Address of the fractal heap, or the undefined sentinel when compact.
    pub(super) fn fractal_heap_addr(&self) -> u64 {
        self.dense
            .as_ref()
            .map_or(UNDEFINED_ADDRESS, |dense| dense.heap.header_addr())
    }

    /// Address of the name index, or the undefined sentinel when compact.
    pub(super) fn name_index_addr(&self) -> u64 {
        self.dense
            .as_ref()
            .map_or(UNDEFINED_ADDRESS, |dense| dense.index.header_addr())
    }
}

/// The two structures a dense group's links live in.
pub(super) struct DenseLinks {
    /// Holds the link message bytes as heap objects.
    pub(super) heap: FractalHeapWriter,
    /// Maps each link name's hash to the heap ID that retrieves it.
    pub(super) index: NameIndexWriter,
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
    ///
    /// Present for **every** group, new-style ones included: libhdf5 leaves the
    /// symbol table structures in place when it converts a group to link
    /// messages, and the parent's symbol table entry keeps caching their
    /// addresses.  Reproducing that means a new-style group's entry stays a
    /// perfectly ordinary `cache_type = 1` entry pointing at real (empty)
    /// structures, rather than at addresses of things that do not exist.
    pub(super) sym: SymPlan,
    /// The group's links, in sorted order.
    pub(super) links: Vec<Link<'a>>,
    /// New-style link storage; `None` for an old-style symbol-table group.
    pub(super) link_storage: Option<LinkStorage>,
    /// Resolved object-header address of each hard alias, indexed like
    /// `node.links`; `None` for a soft or external link, and until resolution.
    pub(super) link_addrs: Vec<Option<u64>>,
    /// Object header reference count: one, plus one per hard alias naming this
    /// group.  Filled by [`GroupPlan::apply_refcounts`].
    pub(super) refcount: u32,
    /// Plans for the group's datasets, in declaration order.
    pub(super) datasets: Vec<DatasetPlan<'a>>,
    /// Plans for the group's sub-groups, in creation order.
    pub(super) groups: Vec<GroupPlan<'a>>,
}

// ---------------------------------------------------------------------------
// Links and the local heap
// ---------------------------------------------------------------------------

/// Every member of a group, in creation order, whatever kind it is.
///
/// Datasets, sub-groups and bare links are all *links* of the group they sit
/// in; a group's member list is one list, never three, which is what makes the
/// duplicate-name rule and the creation order well defined across all of them.
fn collect_links(node: &GroupNode) -> Vec<Link<'_>> {
    let mut links: Vec<Link<'_>> = node
        .datasets
        .iter()
        .enumerate()
        .map(|(index, ds)| Link {
            name: ds.name.as_str(),
            target: LinkTarget::Dataset(index),
            creation_order: ds.creation_order,
        })
        .chain(node.groups.iter().enumerate().map(|(index, grp)| Link {
            name: grp.name.as_str(),
            target: LinkTarget::Group(index),
            creation_order: grp.creation_order,
        }))
        .chain(node.links.iter().enumerate().map(|(index, link)| Link {
            name: link.name.as_str(),
            target: LinkTarget::Link(index),
            creation_order: link.creation_order,
        }))
        .collect();
    // Creation order is unique within a group, so this sort is total.
    links.sort_unstable_by_key(|link| link.creation_order);
    links
}

/// Order a group's links the way libhdf5 orders them: ascending by raw name
/// bytes.
///
/// This is not cosmetic.  `H5G__node_found` binary-searches a symbol table node
/// with the names resolved through the local heap, so an unsorted node does not
/// fail loudly — it silently hides links, while `list(f.keys())` (a linear
/// walk) still reports them.  Every downstream offset depends on this order, so
/// it has to be settled before the local heap is built.
///
/// Datasets, sub-groups and bare links interleave here: a group's links are
/// ordered by name, never by kind.
fn sorted_links(node: &GroupNode) -> Vec<Link<'_>> {
    let mut links = collect_links(node);
    // Names are unique within a group, so the sort is total and an unstable
    // sort is deterministic.
    links.sort_unstable_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    links
}

/// The value a link interns in its group's local heap, if any.
///
/// Only a soft link has one: a symbol table entry stores a 4-byte offset into
/// the group's local heap instead of an address, and the target path lives
/// there.  A hard alias resolves to an address, and an external link cannot
/// appear in a symbol table at all.
fn heap_value(node: &GroupNode, target: LinkTarget) -> Option<&str> {
    let LinkTarget::Link(index) = target else {
        return None;
    };
    match &node.links.get(index)?.kind {
        LinkKind::Soft { path } => Some(path.as_str()),
        LinkKind::HardAlias { .. } | LinkKind::External { .. } => None,
    }
}

/// Build a local heap data segment holding every link name, and every soft
/// link's target path beside it.
///
/// Offset 0 of the segment is reserved for the free-block link, every string is
/// NUL-terminated and 8-byte aligned, and the tail carries a single free-list
/// entry covering whatever slack the allocation rounded up to.
fn build_local_heap(node: &GroupNode, links: &[Link<'_>]) -> LocalHeap {
    let mut bytes: Vec<u8> = vec![0u8; 8];
    let intern = |bytes: &mut Vec<u8>, text: &str| -> u64 {
        let offset = bytes.len() as u64;
        bytes.extend_from_slice(text.as_bytes());
        bytes.push(0);
        let end = bytes.len();
        bytes.resize(pad8(end), 0);
        offset
    };

    let mut name_offsets = Vec::with_capacity(links.len());
    let mut value_offsets = Vec::with_capacity(links.len());
    for link in links {
        name_offsets.push(intern(&mut bytes, link.name));
        value_offsets.push(heap_value(node, link.target).map(|value| intern(&mut bytes, value)));
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
        value_offsets,
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
fn plan_sym_table(
    node: &GroupNode,
    links: &[Link<'_>],
    current: &mut usize,
) -> Result<SymPlan, OxiH5Error> {
    let heap = build_local_heap(node, links);
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
        chunked::layout_chunk_dims(&chunk_shape, ds.elem_size())?
    };
    // Refuse an over-cap chunk count from the geometry alone, before
    // `payload::build` cuts and compresses a single tile.  The chunk index caps
    // the count too, but only after every chunk is already materialised — so a
    // pathological request would spend gigabytes reaching a guard that fires in
    // microseconds here.
    if !chunk_shape.is_empty() {
        chunked::chunk_count(&ds.shape, &chunk_shape)?;
    }
    let payload = payload::build(ds, &chunk_shape)?;

    let oh_addr = *current;
    let oh_size = oh::oh_size(&oh::dataset_oh_msgs(ds, &chunk_dims, &attrs))?;
    *current += oh_size;

    let chunk_tree = match &payload {
        Payload::Chunked(images) => {
            let mut tree = chunked::ChunkTree::plan(ds.shape.len(), images.len(), ds.elem_size())?;
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
        // One name reaches every object until a hard alias adds another; see
        // [`GroupPlan::apply_refcounts`].
        refcount: 1,
    })
}

/// Encode one member of a new-style group as a link message.
///
/// Hard links — to a dataset, a sub-group, or an aliased object elsewhere in
/// the file — are encoded with a placeholder address, because at this point in
/// the layout pass the target's object header has not been placed yet.  The
/// *length* is already final (a hard link's value is always eight bytes), which
/// is what lets the group's object header be sized here and patched later by
/// [`GroupPlan::patch_links`].
fn planned_link(node: &GroupNode, link: &Link<'_>, track_order: bool) -> PlannedLink {
    let value = match link.target {
        LinkTarget::Dataset(_) | LinkTarget::Group(_) => LinkValue::Hard(0),
        LinkTarget::Link(index) => match node.links.get(index).map(|desc| &desc.kind) {
            Some(LinkKind::Soft { path }) => LinkValue::Soft(path.clone()),
            Some(LinkKind::External { file, path }) => LinkValue::External {
                file: file.clone(),
                path: path.clone(),
            },
            // A hard alias, or an index that cannot occur because `collect_links`
            // built it from this very vector; either way the value is an address.
            Some(LinkKind::HardAlias { .. }) | None => LinkValue::Hard(0),
        },
    };
    PlannedLink {
        msg: LinkMsg {
            name: link.name.to_string(),
            value,
            creation_order: track_order.then_some(link.creation_order),
        },
        target: link.target,
    }
}

/// Decide how a new-style group stores its links, and encode them.
///
/// Nothing here depends on where anything lands, which is the point: the
/// group's object header can be sized from this *before* the cursor moves, and
/// the addresses are filled in afterwards by [`LinkStorage::assign`].
///
/// A group at or below [`MAX_COMPACT_LINKS`] members keeps its links in its own
/// object header; a larger one gets a fractal heap and a version-2 name index,
/// both sized from the link messages' *lengths*, which are final even though
/// their hard-link addresses are not.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if a link message cannot be encoded, or if the
/// links together do not fit the one direct block the heap writer emits.
fn build_link_storage(node: &GroupNode) -> Result<LinkStorage, OxiH5Error> {
    // Emission order is creation order when the group tracks it, and name order
    // otherwise — the same order an old-style group would have used, so turning
    // a group new-style does not silently reshuffle it.
    let ordered = if node.track_order {
        collect_links(node)
    } else {
        sorted_links(node)
    };
    let links: Vec<PlannedLink> = ordered
        .iter()
        .map(|link| planned_link(node, link, node.track_order))
        .collect();

    let info = LinkInfo {
        creation_order_tracked: node.track_order,
        max_creation_order: node.next_creation_order,
    };

    if links.len() <= MAX_COMPACT_LINKS {
        return Ok(LinkStorage {
            links,
            info,
            dense: None,
        });
    }

    let mut sizes = Vec::with_capacity(links.len());
    for link in &links {
        sizes.push(link.msg.body_size()?);
    }
    let heap = FractalHeapWriter::plan(&sizes)?;

    let mut entries = Vec::with_capacity(links.len());
    for (index, link) in links.iter().enumerate() {
        entries.push((link.msg.name.clone(), heap.heap_id(index)?));
    }
    let index = NameIndexWriter::plan(entries)?;

    Ok(LinkStorage {
        links,
        info,
        dense: Some(DenseLinks { heap, index }),
    })
}

impl LinkStorage {
    /// Place this group's dense structures, advancing `current` past them.
    ///
    /// A compact group has nothing to place and leaves the cursor alone.
    fn assign(&mut self, current: &mut usize) {
        if let Some(dense) = &mut self.dense {
            dense.heap.assign(*current);
            *current += dense.heap.bytes();
            dense.index.assign(*current);
            *current += dense.index.bytes();
        }
    }
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
    let links = sorted_links(node);

    // A new-style group's links are messages in its own object header, so the
    // link storage has to be planned before the header can be sized.  It also
    // *reserves* file space (a dense group's heap and index), which is why the
    // cursor is walked past the header first: the header sits at the group's
    // own address, exactly as it does in the old-style case.
    let oh_addr = *current;
    let (link_storage, oh_size) = if node.is_link_style() {
        let mut storage = build_link_storage(node)?;
        let size = oh::oh_size(&oh::link_group_oh_msgs(&storage, &attrs)?)?;
        *current += size;
        storage.assign(current);
        (Some(storage), size)
    } else {
        let size = oh::oh_size(&oh::group_oh_msgs(&attrs))?;
        *current += size;
        (None, size)
    };

    // Every group owns a symbol table, new-style ones included: libhdf5 leaves
    // the structures behind when it converts a group, and the parent's cached
    // entry keeps pointing at them.  A new-style group's table is empty — its
    // members are links, not symbol table entries.
    let sym_links: &[Link<'_>] = if node.is_link_style() { &[] } else { &links };
    let sym = plan_sym_table(node, sym_links, current)?;

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
        link_storage,
        link_addrs: vec![None; node.links.len()],
        refcount: 1,
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

    /// Resolve every hard alias in this subtree to the address it names, and
    /// tally how many aliases reach each object.
    ///
    /// A hard alias is the one link kind whose value is an address the writer
    /// has to look up: a soft link's value is a path the *reader* resolves, and
    /// an external link's target is not in this file at all.  Resolving them
    /// here — after the layout pass, against the same path map object-reference
    /// attributes use — is what lets `create_hard_link` name its target the way
    /// a user thinks of it.
    ///
    /// The tally is not bookkeeping: an object header's reference count is the
    /// number of hard links that reach it, and libhdf5 writes 2 for a dataset
    /// with one alias.  Leaving it at 1 makes `H5Ldelete` on either name free
    /// storage the other name still points at.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` naming the link and its target if the
    /// target is not an object in this file.
    pub(super) fn resolve_link_targets(
        &mut self,
        path_to_addr: &HashMap<String, u64>,
        counts: &mut HashMap<u64, u32>,
    ) -> Result<(), OxiH5Error> {
        for (index, link) in self.node.links.iter().enumerate() {
            let LinkKind::HardAlias { target } = &link.kind else {
                continue;
            };
            // "/a/b" and "a/b" are the same object; the map is keyed by the
            // latter, which is also what `collect_addresses` records.
            let key = target.trim_start_matches('/');
            let addr = path_to_addr.get(key).copied().ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "hard link '{}' names '{target}', which is not an object in this file",
                    link.name
                ))
            })?;
            if let Some(slot) = self.link_addrs.get_mut(index) {
                *slot = Some(addr);
            }
            *counts.entry(addr).or_insert(0) += 1;
        }
        for grp in &mut self.groups {
            grp.resolve_link_targets(path_to_addr, counts)?;
        }
        Ok(())
    }

    /// Set every object's header reference count from the alias tally.
    pub(super) fn apply_refcounts(&mut self, counts: &HashMap<u64, u32>) {
        let extra = |addr: usize| 1 + counts.get(&(addr as u64)).copied().unwrap_or(0);
        self.refcount = extra(self.oh_addr);
        for ds in &mut self.datasets {
            ds.refcount = extra(ds.oh_addr);
        }
        for grp in &mut self.groups {
            grp.apply_refcounts(counts);
        }
    }

    /// Fill each new-style link message's hard-link address in.
    ///
    /// Every address a link message can carry is now known: a dataset's and a
    /// sub-group's from their own plans, an alias's from
    /// [`Self::resolve_link_targets`].  Because a hard link's value is a
    /// fixed-width address, nothing here can change a message's length — which
    /// is checked, not assumed.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a link points at something that was
    /// never planned, or if patching an address changed a message's length.
    pub(super) fn patch_links(&mut self) -> Result<(), OxiH5Error> {
        if let Some(storage) = &mut self.link_storage {
            for link in &mut storage.links {
                let LinkValue::Hard(_) = link.msg.value else {
                    continue;
                };
                let before = link.msg.body_size()?;
                let addr = match link.target {
                    LinkTarget::Dataset(index) => {
                        self.datasets.get(index).map(|plan| plan.oh_addr as u64)
                    }
                    LinkTarget::Group(index) => {
                        self.groups.get(index).map(|plan| plan.oh_addr as u64)
                    }
                    LinkTarget::Link(index) => self.link_addrs.get(index).copied().flatten(),
                };
                let addr = addr.ok_or_else(|| {
                    OxiH5Error::Format(format!(
                        "internal writer error: link '{}' has no resolved target address",
                        link.msg.name
                    ))
                })?;
                link.msg.value = LinkValue::Hard(addr);
                super::check_size("patched link message", link.msg.body_size()?, before)?;
            }
        }
        for grp in &mut self.groups {
            grp.patch_links()?;
        }
        Ok(())
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

    /// Register every variable-length dataset in this subtree with the file's
    /// one shared global heap collection, recording the object indices it hands
    /// back.
    ///
    /// Strings and sequences go into the very same collection set: on disk they
    /// are the same thing, a heap object addressed by a 16-byte reference, and
    /// only the datatype message distinguishes them.  An **empty** sequence
    /// registers nothing — its reference is the all-zero null reference, which
    /// is what libhdf5 writes and what `decode_vlen_sequences` reads back as an
    /// empty sequence — so the index vector carries a 0 in that slot.
    pub(super) fn register_vlen_strings(&mut self, gcol: &mut oxih5_format::GlobalHeapWriter) {
        for ds in &mut self.datasets {
            ds.vlen_obj_idx = match (&ds.desc.vlen_strings, &ds.desc.vlen_seqs) {
                (Some(strings), _) => strings.iter().map(|s| gcol.write_string(s)).collect(),
                (None, Some(seqs)) => seqs
                    .iter()
                    .map(|bytes| {
                        if bytes.is_empty() {
                            0
                        } else {
                            gcol.write_bytes(bytes)
                        }
                    })
                    .collect(),
                (None, None) => Vec::new(),
            };
        }
        for grp in &mut self.groups {
            grp.register_vlen_strings(gcol);
        }
    }

    /// Register the global-heap payload of every vlen-object-reference attribute
    /// in this subtree with the shared collection writer `gcol`.
    ///
    /// Mirrors [`Self::fill_obj_refs`]'s recursion (group attrs, each dataset's
    /// attrs, then child groups) and must run **after** `fill_obj_refs` (so the
    /// resolved addresses that form each payload exist) and **before**
    /// [`oxih5_format::GlobalHeapWriter::build_collections`].
    pub(super) fn register_vlen_objref_attrs(&mut self, gcol: &mut oxih5_format::GlobalHeapWriter) {
        elem::register_vlen_objref_heap(&mut self.attrs, gcol);
        for ds in &mut self.datasets {
            elem::register_vlen_objref_heap(&mut ds.attrs, gcol);
        }
        for grp in &mut self.groups {
            grp.register_vlen_objref_attrs(gcol);
        }
    }

    /// Resolve the heap ordinals recorded by [`Self::register_vlen_objref_attrs`]
    /// into the absolute collection address and 1-based local index each sequence
    /// landed at, across this subtree.
    ///
    /// Must run **after** the collections are laid out (so `collection_addrs` and
    /// `locations` are known).
    pub(super) fn fill_vlen_objref_locs(
        &mut self,
        collection_addrs: &[u64],
        locations: &[oxih5_format::HeapObjectLocation],
    ) {
        elem::fill_vlen_objref_locs(&mut self.attrs, collection_addrs, locations);
        for ds in &mut self.datasets {
            elem::fill_vlen_objref_locs(&mut ds.attrs, collection_addrs, locations);
        }
        for grp in &mut self.groups {
            grp.fill_vlen_objref_locs(collection_addrs, locations);
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
            dtype: None,
            vlen_strings: None,
            vlen_seqs: None,
            creation_order: 0,
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
            filter: Some(Filter {
                deflate: Some(6),
                ..Filter::default()
            }),
            dtype: None,
            vlen_strings: None,
            vlen_seqs: None,
            creation_order: 0,
        }
    }

    #[test]
    fn empty_local_heap_is_padded_to_the_minimum() {
        let heap = build_local_heap(&GroupNode::new(""), &[]);
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
        let mut node = GroupNode::new("");
        node.push_dataset(dataset("lat"));
        node.push_dataset(dataset("longitude"));
        let links = sorted_links(&node);
        let heap = build_local_heap(&node, &links);
        assert_eq!(heap.name_offsets, vec![8, 16]);
        assert_eq!(&heap.data[8..12], b"lat\0");
        assert_eq!(&heap.data[16..26], b"longitude\0");
        // "longitude\0" is 10 bytes, padded up to 16.
        assert_eq!(heap.used, 32);
        assert_eq!(heap.used % 8, 0);
    }

    #[test]
    fn local_heap_grows_past_the_minimum_when_needed() {
        let mut node = GroupNode::new("");
        for i in 0..16 {
            node.push_dataset(dataset(&format!("dataset_number_{i:03}")));
        }
        let links = sorted_links(&node);
        let heap = build_local_heap(&node, &links);
        assert!(heap.data.len() > MIN_HEAP_DATA_SIZE);
        assert!(heap.data.len() >= heap.used + 16);
        assert_eq!(heap.data.len() % 8, 0);
    }

    /// Datasets and sub-groups share one name ordering.
    #[test]
    fn links_interleave_datasets_and_groups_by_name() {
        let mut node = GroupNode::new("");
        node.push_dataset(dataset("zebra"));
        node.push_dataset(dataset("apple"));
        node.push_group(GroupNode::new("mango"));
        node.push_group(GroupNode::new("banana"));
        let links = sorted_links(&node);
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
        root.push_dataset(deflated("packed", 4096));
        root.push_dataset(dataset("after"));

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
        root.push_dataset(dataset("top"));
        let mut a = GroupNode::new("a");
        let mut b = GroupNode::new("b");
        b.push_dataset(dataset("deep"));
        a.push_group(b);
        root.push_group(a);

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
        root.push_dataset(dataset("top"));
        let mut a = GroupNode::new("a");
        let mut b = GroupNode::new("b");
        b.push_dataset(dataset("deep"));
        a.push_group(b);
        root.push_group(a);

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
