//! Old-style group symbol tables: geometry, planning, and emission.
//!
//! An old-style HDF5 group is a **B-tree v1 of node type 0** whose leaves are
//! **symbol table nodes (SNODs)**, plus a local heap holding the link names.
//! This module owns the shape of that structure — how wide a node is, how many
//! links fit in a SNOD, how the levels stack up — and emits it.
//!
//! Three properties are load-bearing.  Each was verified against
//! libhdf5 1.14.6 (via h5py 3.15.1), and getting any of them wrong yields a file
//! that libhdf5 either rejects outright or, worse, reads *incorrectly*:
//!
//! # 1. A node's on-disk image is a fixed size, not `entries_used` wide
//!
//! `H5G__cache_node_deserialize` sizes the SNOD image it reads from the
//! superblock's `leaf_node_K` — as `8 + 2*K*40` — and only *then* looks at the
//! node's own `nsyms`.  Writing a ninth entry into a node declared with K = 4
//! therefore does not overflow our writer, it overflows libhdf5's reader:
//!
//! ```text
//! #012: H5Gcache.c line 188 in H5G__cache_node_deserialize(): unable to decode symbol table entries
//! #013: H5Gent.c line 86 in H5G__ent_decode_vec(): ran off the end of the image buffer
//! ```
//!
//! That is why [`SNOD_SIZE`] is derived from [`SYM_LEAF_K`] rather than from the
//! number of links, why every node is written at its full width with the slots
//! past `entries_used` zero-filled, and why [`super::format::write_superblock`]
//! reads `leaf_node_K` from here instead of repeating a literal.
//!
//! # 2. Entries must be sorted by name
//!
//! `H5G__node_found` **binary-searches** a SNOD, resolving each entry's name
//! through the local heap.  An unsorted node does not fail loudly; it silently
//! hides links.  Two files differing only in declaration order:
//!
//! ```text
//! ["aaa","bbb"] -> opens fine
//! ["bbb","aaa"] -> KeyError: object 'bbb' doesn't exist
//! ```
//!
//! `list(f.keys())` still returns everything, because iteration is linear, so
//! the damage reads as a missing dataset rather than a malformed file.  Sorting
//! is required on its own account: a control file whose B-tree key range
//! provably covered every name still lost an entry, so this is **not** a
//! symptom of a bad key range.
//!
//! # 3. Keys bound children on the right, exclusively on the left
//!
//! `H5G__node_cmp3(lt_key, name, rt_key)` steers left when `name <= lt_key` and
//! right when `name > rt_key`, so child `i` owns the half-open name interval
//! `(key[i], key[i+1]]`.  Hence `key[i+1]` is the *greatest* name under child
//! `i`, and `key[0]` is the local heap's reserved offset 0, which holds the
//! empty string — smaller than every legal link name, since the writer rejects
//! an empty name.

use oxih5_core::OxiH5Error;

use super::format::{self, fill_zero, write_u16_le, write_u64_le, SnodEntry};
use super::narrow;

// ---------------------------------------------------------------------------
// Geometry — the single source of truth for both the superblock and the nodes
// ---------------------------------------------------------------------------

/// Superblock `leaf_node_K`: half the capacity of a symbol table node.
///
/// 4 is libhdf5's own default, confirmed against a file libhdf5 wrote itself
/// with `libver='earliest'`.
pub(super) const SYM_LEAF_K: usize = 4;

/// Most symbol table entries one SNOD may hold.
pub(super) const SNOD_MAX_ENTRIES: usize = 2 * SYM_LEAF_K;

/// Size of one symbol table entry, for `size_of_offsets = 8`.
///
/// `link_name_offset(8) + object_header_address(8) + cache_type(4) +
/// reserved(4) + scratch_pad(16)`.
pub(super) const STE_SIZE: usize = 40;

/// Size of the SNOD prefix that precedes the entries.
pub(super) const SNOD_PREFIX: usize = 8;

/// Fixed on-disk size of every symbol table node in the file.
///
/// libhdf5 computes this same value from `leaf_node_K` alone, so it must not
/// vary between the root group and a sub-group — `leaf_node_K` is one
/// per-file field.
pub(super) const SNOD_SIZE: usize = SNOD_PREFIX + SNOD_MAX_ENTRIES * STE_SIZE;

/// Superblock `internal_node_K`: half the child capacity of a B-tree node.
///
/// 16 is libhdf5's own default, from the same reference file.
pub(super) const GROUP_INTERNAL_K: usize = 16;

/// Most children one group B-tree node may hold.
pub(super) const GROUP_NODE_MAX_CHILDREN: usize = 2 * GROUP_INTERNAL_K;

/// Size of a B-tree v1 node prefix: signature, type, level, `entries_used`,
/// and the two sibling addresses.
const NODE_PREFIX: usize = 24;

/// Size of one B-tree v1 group key: an offset into the group's local heap.
const KEY_SIZE: usize = 8;

/// Size of one B-tree v1 child pointer.
const CHILD_SIZE: usize = 8;

/// Fixed on-disk size of every group B-tree node in the file.
///
/// A node stores `n` children interleaved with `n + 1` keys, so a node written
/// at full width is `prefix + key[0] + 2K * (child + key)`.
pub(super) const GROUP_NODE_SIZE: usize =
    NODE_PREFIX + KEY_SIZE + GROUP_NODE_MAX_CHILDREN * (CHILD_SIZE + KEY_SIZE);

/// HDF5's "this address is undefined" sentinel.
const UNDEFINED_ADDR: u64 = u64::MAX;

/// Deepest symbol table B-tree the writer will build.
///
/// Purely structural insurance: [`MAX_GROUP_LINKS`] caps a group well before
/// this can bite.  oxih5's own reader gives up at depth 64, and libhdf5 has no
/// fixed limit, so 8 is a conservative ceiling that keeps
/// [`SymTable::write`]'s bottom-up loop finite even if the recurrence below
/// were ever changed.
const MAX_LEVELS: usize = 8;

/// Largest number of links the writer will place in a single group.
///
/// This is a resource guard, not a format limit: `1 << 24` links already imply
/// two million symbol table nodes and a multi-gigabyte image, so refusing up
/// front turns an out-of-memory abort into a typed error.  Structurally the
/// tree reaches much further — level `L` addresses
/// `SNOD_MAX_ENTRIES * GROUP_NODE_MAX_CHILDREN^L` links, i.e. 256, then 8 192,
/// then 262 144, and so on.
pub(super) const MAX_GROUP_LINKS: usize = 1 << 24;

// ---------------------------------------------------------------------------
// Level planning
// ---------------------------------------------------------------------------

/// Number of B-tree nodes on each level, level 0 (the SNODs' parents) first.
///
/// The last entry is always 1: that node is the tree's root, the address the
/// group's symbol table message points at.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the tree would need more than
/// [`MAX_LEVELS`] levels.
pub(super) fn plan_levels(n_snods: usize) -> Result<Vec<usize>, OxiH5Error> {
    // A group with no links still gets one (empty) SNOD, so there is always at
    // least one level-0 node to point at it.
    let mut nodes = n_snods.max(1).div_ceil(GROUP_NODE_MAX_CHILDREN);
    let mut levels = vec![nodes];
    while nodes > 1 {
        if levels.len() >= MAX_LEVELS {
            return Err(OxiH5Error::Format(format!(
                "symbol table B-tree for {n_snods} symbol table nodes would exceed \
                 {MAX_LEVELS} levels"
            )));
        }
        nodes = nodes.div_ceil(GROUP_NODE_MAX_CHILDREN);
        levels.push(nodes);
    }
    Ok(levels)
}

// ---------------------------------------------------------------------------
// SymTable — one group's B-tree + SNOD block
// ---------------------------------------------------------------------------

/// The planned symbol table of one group: a block of B-tree nodes followed by a
/// block of SNODs, both laid out at fixed stride.
///
/// Built by [`SymTable::plan`] from the link count alone, then given its
/// addresses by [`SymTable::assign`], then emitted by [`SymTable::write`].
#[derive(Debug)]
pub(super) struct SymTable {
    /// Number of links, i.e. of symbol table entries.
    n_links: usize,
    /// Number of SNODs; at least one, even for an empty group.
    n_snods: usize,
    /// Node count of each level, level 0 first; the last is always 1.
    levels: Vec<usize>,
    /// Address of the first node of each level, parallel to `levels`.
    level_addr: Vec<usize>,
    /// Address of the tree's root node.
    root_addr: usize,
    /// Address of the first SNOD; the rest follow at [`SNOD_SIZE`] stride.
    snod_base: usize,
}

impl SymTable {
    /// Plan the symbol table of a group holding `n_links` links.
    ///
    /// Addresses are left at zero until [`SymTable::assign`] is called; the
    /// sizes ([`SymTable::btree_bytes`], [`SymTable::snod_bytes`]) are final
    /// immediately, which is what lets the caller reserve space in the same
    /// pass that computes them.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `n_links` exceeds [`MAX_GROUP_LINKS`],
    /// or if the resulting tree would be deeper than [`MAX_LEVELS`].
    pub(super) fn plan(n_links: usize) -> Result<Self, OxiH5Error> {
        if n_links > MAX_GROUP_LINKS {
            return Err(OxiH5Error::Format(format!(
                "group holds {n_links} links, over the writer's limit of {MAX_GROUP_LINKS}"
            )));
        }
        let n_snods = n_links.div_ceil(SNOD_MAX_ENTRIES).max(1);
        let levels = plan_levels(n_snods)?;
        Ok(Self {
            n_links,
            n_snods,
            level_addr: vec![0; levels.len()],
            levels,
            root_addr: 0,
            snod_base: 0,
        })
    }

    /// Total bytes the group's B-tree nodes occupy.
    pub(super) fn btree_bytes(&self) -> usize {
        self.levels.iter().sum::<usize>() * GROUP_NODE_SIZE
    }

    /// Total bytes the group's symbol table nodes occupy.
    pub(super) fn snod_bytes(&self) -> usize {
        self.n_snods * SNOD_SIZE
    }

    /// Place the planned nodes: B-tree levels from `btree_base`, level 0 first,
    /// and the SNODs from `snod_base`.
    pub(super) fn assign(&mut self, btree_base: usize, snod_base: usize) {
        let mut addr = btree_base;
        for (slot, &count) in self.level_addr.iter_mut().zip(self.levels.iter()) {
            *slot = addr;
            addr += count * GROUP_NODE_SIZE;
        }
        // The deepest level is the single root node; `plan_levels` guarantees
        // at least one level, so the last address is always the root's.
        self.root_addr = self.level_addr.last().copied().unwrap_or(btree_base);
        self.snod_base = snod_base;
    }

    /// Address of the tree's root node — what a symbol table message, the
    /// superblock, and a parent entry's scratch pad all point at.
    pub(super) fn root_addr(&self) -> usize {
        self.root_addr
    }

    /// Emit every SNOD and every B-tree node; returns bytes written, always
    /// `btree_bytes() + snod_bytes()`.
    ///
    /// `entries` must already be sorted by name — see the module docs; nothing
    /// downstream can detect that it is not.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `entries` does not match the planned
    /// link count, or if a node field overflows its on-disk width.
    pub(super) fn write(&self, buf: &mut [u8], entries: &[SnodEntry]) -> Result<usize, OxiH5Error> {
        if entries.len() != self.n_links {
            return Err(OxiH5Error::Format(format!(
                "internal writer error: symbol table planned for {} links but got {}",
                self.n_links,
                entries.len()
            )));
        }

        // An empty group still owns one SNOD, so that its B-tree has a child.
        let mut chunks: Vec<&[SnodEntry]> = entries.chunks(SNOD_MAX_ENTRIES).collect();
        if chunks.is_empty() {
            chunks.push(&[]);
        }

        let mut wrote = 0usize;
        let mut child_addrs: Vec<u64> = Vec::with_capacity(self.n_snods);
        let mut child_keys: Vec<u64> = Vec::with_capacity(self.n_snods);
        for (i, chunk) in chunks.iter().enumerate() {
            let addr = self.snod_base + i * SNOD_SIZE;
            wrote += format::write_snod(buf, addr, chunk)?;
            child_addrs.push(addr as u64);
            // The key that closes a child is the greatest name under it, and
            // the entries are sorted, so that is the last one.  An empty SNOD
            // closes at heap offset 0, the empty string.
            child_keys.push(chunk.last().map_or(0, |e| e.name_offset));
        }

        wrote += self.write_levels(buf, child_addrs, child_keys)?;
        Ok(wrote)
    }

    /// Emit the B-tree levels bottom-up, folding each level's nodes into the
    /// child list of the level above; returns bytes written.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a level or entry count overflows its
    /// on-disk width.
    fn write_levels(
        &self,
        buf: &mut [u8],
        mut child_addrs: Vec<u64>,
        mut child_keys: Vec<u64>,
    ) -> Result<usize, OxiH5Error> {
        let mut wrote = 0usize;
        for (level, &base) in self.level_addr.iter().enumerate() {
            let n_nodes = child_addrs.len().div_ceil(GROUP_NODE_MAX_CHILDREN);
            let mut next_addrs = Vec::with_capacity(n_nodes);
            let mut next_keys = Vec::with_capacity(n_nodes);

            let addr_chunks = child_addrs.chunks(GROUP_NODE_MAX_CHILDREN);
            let key_chunks = child_keys.chunks(GROUP_NODE_MAX_CHILDREN);
            for (i, (addrs, keys)) in addr_chunks.zip(key_chunks).enumerate() {
                let addr = base + i * GROUP_NODE_SIZE;
                // Siblings are linked within a level only; the ends are
                // undefined.  oxih5's reader ignores both, but libhdf5 walks
                // them for range operations.
                let left = if i == 0 {
                    UNDEFINED_ADDR
                } else {
                    (addr - GROUP_NODE_SIZE) as u64
                };
                let right = if i + 1 == n_nodes {
                    UNDEFINED_ADDR
                } else {
                    (addr + GROUP_NODE_SIZE) as u64
                };
                wrote += write_node(buf, addr, level, left, right, addrs, keys)?;
                next_addrs.push(addr as u64);
                // A node closes where its rightmost child closes.
                next_keys.push(keys.last().copied().unwrap_or(0));
            }

            child_addrs = next_addrs;
            child_keys = next_keys;
        }
        Ok(wrote)
    }
}

/// Write one group B-tree node at `addr`; returns bytes written, always
/// [`GROUP_NODE_SIZE`].
///
/// The node is emitted at its full width with the slots past `entries_used`
/// zero-filled, because libhdf5 reads a fixed-size image — see the module docs.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `children` overruns
/// [`GROUP_NODE_MAX_CHILDREN`], or if the level or child count does not fit its
/// on-disk field.
fn write_node(
    buf: &mut [u8],
    addr: usize,
    level: usize,
    left_sibling: u64,
    right_sibling: u64,
    children: &[u64],
    keys: &[u64],
) -> Result<usize, OxiH5Error> {
    if children.len() > GROUP_NODE_MAX_CHILDREN || children.len() != keys.len() {
        return Err(OxiH5Error::Format(format!(
            "internal writer error: B-tree node with {} children and {} keys, \
             capacity {GROUP_NODE_MAX_CHILDREN}",
            children.len(),
            keys.len()
        )));
    }

    fill_zero(buf, addr, GROUP_NODE_SIZE);
    buf[addr..addr + 4].copy_from_slice(b"TREE");
    buf[addr + 4] = 0x00; // node type = 0 (group)
    buf[addr + 5] = narrow::<u8>("B-tree node level", level)?;
    write_u16_le(
        buf,
        addr + 6,
        narrow::<u16>("B-tree entries used", children.len())?,
    );
    write_u64_le(buf, addr + 8, left_sibling);
    write_u64_le(buf, addr + 16, right_sibling);

    // key[0] is the *exclusive* lower bound of child[0].  Heap offset 0 holds
    // the empty string, which is below every legal link name, so it never
    // rejects an entry.
    write_u64_le(buf, addr + NODE_PREFIX, 0);
    for (i, (&child, &key)) in children.iter().zip(keys).enumerate() {
        let slot = addr + NODE_PREFIX + KEY_SIZE + i * (CHILD_SIZE + KEY_SIZE);
        write_u64_le(buf, slot, child);
        write_u64_le(buf, slot + CHILD_SIZE, key);
    }

    Ok(GROUP_NODE_SIZE)
}

/// Byte-level inspection of a built file image.
///
/// Lives here rather than in a test module because both this module's tests
/// and [`super`]'s object-header walker read the same structures: the root
/// symbol table, the B-tree levels beneath it, and the local heap the entry
/// names resolve through.
#[cfg(test)]
pub(in crate::write) mod probe {
    pub(in crate::write) fn le_u16(bytes: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([bytes[off], bytes[off + 1]])
    }

    /// Read a little-endian `u64` out of a built file image.
    pub(in crate::write) fn le_u64(bytes: &[u8], off: usize) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[off..off + 8]);
        u64::from_le_bytes(b)
    }

    /// Collect every SNOD address under a group B-tree, descending through
    /// however many levels the tree has.
    pub(in crate::write) fn collect_snods(bytes: &[u8], btree: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![btree];
        while let Some(node) = stack.pop() {
            assert_eq!(&bytes[node..node + 4], b"TREE", "expected a TREE at {node}");
            assert_eq!(bytes[node + 4], 0, "expected a group B-tree at {node}");
            let level = bytes[node + 5];
            let entries = le_u16(bytes, node + 6) as usize;
            for i in 0..entries {
                // key[0] first, then child/key pairs: child[i] is at 24+8+i*16.
                let child = le_u64(bytes, node + 32 + i * 16) as usize;
                if level == 0 {
                    out.push(child);
                } else {
                    stack.push(child);
                }
            }
        }
        out
    }

    /// The root group's B-tree root address and local heap *data* address.
    pub(in crate::write) fn root_symbol_table(bytes: &[u8]) -> (usize, usize) {
        let btree = le_u64(bytes, 80) as usize;
        let heap_hdr = le_u64(bytes, 88) as usize;
        // Local heap header: signature(4) version(1) reserved(3) size(8)
        // free-list head(8) data segment address(8).
        let heap_data = le_u64(bytes, heap_hdr + 24) as usize;
        (btree, heap_data)
    }

    /// Resolve one symbol table entry's link name through the local heap.
    pub(in crate::write) fn heap_name(bytes: &[u8], heap_data: usize, name_offset: u64) -> &str {
        let start = heap_data + name_offset as usize;
        let len = bytes[start..]
            .iter()
            .position(|&b| b == 0)
            .expect("link name must be NUL-terminated");
        std::str::from_utf8(&bytes[start..start + len]).expect("link name must be UTF-8")
    }

    /// Every root link, in the order the symbol table actually stores it.
    pub(in crate::write) fn root_link_names(bytes: &[u8]) -> Vec<&str> {
        let (btree, heap_data) = root_symbol_table(bytes);
        let mut names = Vec::new();
        for snod in collect_snods(bytes, btree) {
            assert_eq!(&bytes[snod..snod + 4], b"SNOD", "expected a SNOD at {snod}");
            for i in 0..le_u16(bytes, snod + 6) as usize {
                let ste = snod + 8 + i * 40;
                names.push(heap_name(bytes, heap_data, le_u64(bytes, ste)));
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::FileWriter;
    use crate::File;
    use probe::*;

    /// The geometry libhdf5 1.14.6 itself writes, and the arithmetic
    /// `H5G__cache_node_deserialize` performs on it.
    #[test]
    fn geometry_matches_libhdf5_defaults() {
        assert_eq!(SYM_LEAF_K, 4);
        assert_eq!(GROUP_INTERNAL_K, 16);
        assert_eq!(SNOD_MAX_ENTRIES, 8);
        assert_eq!(SNOD_SIZE, 328);
        assert_eq!(SNOD_SIZE, 8 + 2 * SYM_LEAF_K * 40);
        assert_eq!(GROUP_NODE_MAX_CHILDREN, 32);
        assert_eq!(GROUP_NODE_SIZE, 544);
        assert_eq!(GROUP_NODE_SIZE, 24 + 8 + 2 * GROUP_INTERNAL_K * 16);
    }

    #[test]
    fn levels_stay_flat_until_the_root_would_overflow() {
        // One level while the SNODs fit one node's worth of children.
        assert_eq!(plan_levels(0).expect("0"), vec![1]);
        assert_eq!(plan_levels(1).expect("1"), vec![1]);
        assert_eq!(plan_levels(32).expect("32"), vec![1]);
        // The 33rd SNOD forces a parent.
        assert_eq!(plan_levels(33).expect("33"), vec![2, 1]);
        assert_eq!(plan_levels(1024).expect("1024"), vec![32, 1]);
        assert_eq!(plan_levels(1025).expect("1025"), vec![33, 2, 1]);
    }

    /// Level `L` addresses `8 * 32^L` links: 256, 8 192, 262 144.
    #[test]
    fn level_capacity_follows_the_documented_progression() {
        for (links, want_levels) in [(256usize, 1usize), (8192, 2), (262_144, 3)] {
            let table = SymTable::plan(links).expect("plan");
            assert_eq!(
                table.levels.len(),
                want_levels,
                "{links} links should need {want_levels} level(s)"
            );
            // One more link tips it into the next level.
            let table = SymTable::plan(links + 1).expect("plan");
            assert_eq!(table.levels.len(), want_levels + 1, "{links} + 1 links");
        }
    }

    #[test]
    fn too_many_links_is_a_typed_error_not_an_allocation() {
        let err = SymTable::plan(MAX_GROUP_LINKS + 1).expect_err("must refuse");
        assert!(
            format!("{err}").contains("over the writer's limit"),
            "{err}"
        );
        assert!(SymTable::plan(MAX_GROUP_LINKS).is_ok());
    }

    #[test]
    fn absurd_node_counts_hit_the_depth_cap() {
        let err = plan_levels(usize::MAX).expect_err("must refuse");
        assert!(format!("{err}").contains("exceed 8 levels"), "{err}");
    }

    /// Sizes are final before addresses are known — that is what lets pass one
    /// reserve space in the same sweep that plans the tree.
    #[test]
    fn sizes_are_known_before_addresses_are_assigned() {
        let mut table = SymTable::plan(9).expect("plan");
        assert_eq!(table.snod_bytes(), 2 * SNOD_SIZE);
        assert_eq!(table.btree_bytes(), GROUP_NODE_SIZE);
        table.assign(1000, 2000);
        assert_eq!(table.root_addr(), 1000);
        assert_eq!(table.snod_bytes(), 2 * SNOD_SIZE);
        assert_eq!(table.btree_bytes(), GROUP_NODE_SIZE);
    }

    #[test]
    fn levels_are_placed_bottom_up_with_the_root_last() {
        let mut table = SymTable::plan(8 * 33).expect("plan");
        assert_eq!(table.levels, vec![2, 1]);
        table.assign(1000, 9000);
        assert_eq!(table.level_addr, vec![1000, 1000 + 2 * GROUP_NODE_SIZE]);
        assert_eq!(table.root_addr(), 1000 + 2 * GROUP_NODE_SIZE);
        assert_eq!(table.btree_bytes(), 3 * GROUP_NODE_SIZE);
    }

    /// Build a `SymTable` over `n` synthetic links and emit it into a buffer
    /// large enough to hold the whole thing, returning the buffer and the
    /// table.  Heap offsets are `8 * (i + 1)`, i.e. already ascending.
    fn emit(n: usize) -> (Vec<u8>, SymTable) {
        let mut table = SymTable::plan(n).expect("plan");
        table.assign(0, table.btree_bytes());
        let entries: Vec<SnodEntry> = (0..n)
            .map(|i| SnodEntry {
                name_offset: 8 * (i as u64 + 1),
                oh_addr: 0x1_0000 + i as u64,
                group_cache: None,
            })
            .collect();
        let total = table.btree_bytes() + table.snod_bytes();
        let mut buf = vec![0u8; total];
        assert_eq!(table.write(&mut buf, &entries).expect("write"), total);
        (buf, table)
    }

    fn u16_at(buf: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([buf[off], buf[off + 1]])
    }

    fn u64_at(buf: &[u8], off: usize) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[off..off + 8]);
        u64::from_le_bytes(b)
    }

    #[test]
    fn nine_links_fill_one_snod_and_spill_into_a_second() {
        let (buf, table) = emit(9);
        assert_eq!(table.n_snods, 2);
        let first = table.snod_base;
        let second = first + SNOD_SIZE;
        assert_eq!(&buf[first..first + 4], b"SNOD");
        assert_eq!(&buf[second..second + 4], b"SNOD");
        assert_eq!(u16_at(&buf, first + 6), 8);
        assert_eq!(u16_at(&buf, second + 6), 1);
    }

    #[test]
    fn keys_close_each_child_on_its_greatest_name() {
        let (buf, table) = emit(9);
        // One level-0 node with two children.
        assert_eq!(u16_at(&buf, 6), 2);
        assert_eq!(buf[5], 0, "level");
        assert_eq!(u64_at(&buf, 24), 0, "key[0] is the empty-string offset");
        assert_eq!(u64_at(&buf, 32), table.snod_base as u64, "child[0]");
        assert_eq!(u64_at(&buf, 40), 8 * 8, "key[1] closes the 8th entry");
        assert_eq!(u64_at(&buf, 48), (table.snod_base + SNOD_SIZE) as u64);
        assert_eq!(u64_at(&buf, 56), 8 * 9, "key[2] closes the 9th entry");
    }

    #[test]
    fn nodes_are_written_at_full_width_with_the_tail_zeroed() {
        let (buf, _) = emit(9);
        // Two children occupy key[0..3] and child[0..2]; everything from the
        // third key/child pair to the end of the node must be zero.
        let used = NODE_PREFIX + KEY_SIZE + 2 * (CHILD_SIZE + KEY_SIZE);
        assert!(
            buf[used..GROUP_NODE_SIZE].iter().all(|&b| b == 0),
            "slots past entries_used must be zero"
        );
    }

    #[test]
    fn a_two_level_tree_links_its_siblings_and_stacks_its_keys() {
        // 33 SNODs -> two level-0 nodes (32 + 1) under one root.
        let n = 33 * SNOD_MAX_ENTRIES;
        let (buf, table) = emit(n);
        assert_eq!(table.levels, vec![2, 1]);

        let left = table.level_addr[0];
        let right = left + GROUP_NODE_SIZE;
        let root = table.root_addr();

        assert_eq!(u16_at(&buf, left + 6), 32);
        assert_eq!(u16_at(&buf, right + 6), 1);
        assert_eq!(buf[left + 5], 0);
        assert_eq!(buf[root + 5], 1, "root sits one level up");
        assert_eq!(u16_at(&buf, root + 6), 2);

        // Siblings are linked within the level, undefined at the ends.
        assert_eq!(u64_at(&buf, left + 8), UNDEFINED_ADDR);
        assert_eq!(u64_at(&buf, left + 16), right as u64);
        assert_eq!(u64_at(&buf, right + 8), left as u64);
        assert_eq!(u64_at(&buf, right + 16), UNDEFINED_ADDR);
        assert_eq!(u64_at(&buf, root + 8), UNDEFINED_ADDR);
        assert_eq!(u64_at(&buf, root + 16), UNDEFINED_ADDR);

        // The root's keys are its children's closing keys.
        assert_eq!(u64_at(&buf, root + 24), 0);
        assert_eq!(u64_at(&buf, root + 32), left as u64);
        assert_eq!(u64_at(&buf, root + 40), u64_at(&buf, left + 24 + 32 * 16));
        assert_eq!(u64_at(&buf, root + 48), right as u64);
        assert_eq!(
            u64_at(&buf, root + 56),
            8 * n as u64,
            "greatest name overall"
        );
    }

    #[test]
    fn an_empty_group_still_gets_one_snod_under_one_node() {
        let (buf, table) = emit(0);
        assert_eq!(table.n_snods, 1);
        assert_eq!(table.btree_bytes(), GROUP_NODE_SIZE);
        assert_eq!(u16_at(&buf, 6), 1, "one child");
        assert_eq!(u64_at(&buf, 24), 0, "key[0]");
        assert_eq!(u64_at(&buf, 40), 0, "key[1]");
        assert_eq!(u16_at(&buf, table.snod_base + 6), 0, "no symbols");
    }

    #[test]
    fn write_rejects_an_entry_count_it_was_not_planned_for() {
        let mut table = SymTable::plan(2).expect("plan");
        table.assign(0, GROUP_NODE_SIZE);
        let mut buf = vec![0u8; GROUP_NODE_SIZE + SNOD_SIZE];
        let err = table.write(&mut buf, &[]).expect_err("must reject");
        assert!(format!("{err}").contains("planned for 2 links"), "{err}");
    }

    // -----------------------------------------------------------------------
    // W1a: symbol-table conformance — node width, sorting, and B-tree levels
    // -----------------------------------------------------------------------

    /// Build `n` root datasets named `ds000`, `ds001`, … and return the image.
    fn w1a_numbered_datasets(n: usize) -> Vec<u8> {
        let mut w = FileWriter::new();
        for i in 0..n {
            w.write_dataset_i32(&format!("ds{i:03}"), &[i as i32], &[1])
                .expect("write");
        }
        w.build_to_vec().expect("build_to_vec")
    }

    /// The ninth link must start a second symbol table node.
    ///
    /// libhdf5 sizes a SNOD image from the superblock's `leaf_node_K` alone —
    /// `8 + 2*K*40` = 328 bytes for K = 4 — and then reads `nsyms` out of it.
    /// A ninth entry packed into that node makes `H5G__ent_decode_vec` run off
    /// the end of the image buffer, so files with nine or more root items were
    /// simply unopenable.
    #[test]
    fn w1a_nine_datasets_span_two_snods() {
        let bytes = w1a_numbered_datasets(9);
        let (btree, _) = root_symbol_table(&bytes);
        let snods = collect_snods(&bytes, btree);
        assert_eq!(snods.len(), 2, "nine links need two symbol table nodes");

        let counts: Vec<u16> = snods
            .iter()
            .map(|&snod| {
                assert_eq!(&bytes[snod..snod + 4], b"SNOD", "expected a SNOD at {snod}");
                le_u16(&bytes, snod + 6)
            })
            .collect();
        assert_eq!(counts, vec![8, 1]);

        // Each node is the one fixed width the superblock advertises.
        assert_eq!(snods[1] - snods[0], SNOD_SIZE);
    }

    /// Every symbol table node must hold its entries in ascending name order.
    ///
    /// `H5G__node_found` binary-searches the node with the names resolved
    /// through the local heap.  An unsorted node does not fail loudly, it
    /// silently hides links: `["bbb","aaa"]` opened fine but `f['bbb']` raised
    /// `KeyError`, while `list(f.keys())` — a linear walk — still listed both.
    #[test]
    fn w1a_snod_entries_are_name_sorted() {
        let mut w = FileWriter::new();
        // Declared in an order that is neither sorted nor reverse-sorted, and
        // that mixes datasets with sub-groups.
        for name in ["values", "lat", "zulu", "alpha", "mike", "lon"] {
            w.write_dataset_f64(name, &[1.0], &[1]).expect("dataset");
        }
        w.create_group("nested").expect("nested");
        w.create_group("aardvark").expect("aardvark");
        for name in ["kilo", "bravo", "yankee"] {
            w.write_dataset_i32(name, &[1], &[1]).expect("dataset");
        }

        let bytes = w.build_to_vec().expect("build_to_vec");
        let names = root_link_names(&bytes);
        assert_eq!(names.len(), 11, "every link must appear exactly once");

        let mut sorted = names.clone();
        sorted.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        assert_eq!(names, sorted, "symbol table entries are out of order");

        // Sorting the links also makes the heap offsets ascend with them, which
        // is what keeps the B-tree keys monotonic.
        let (btree, heap_data) = root_symbol_table(&bytes);
        let mut previous = 0u64;
        for snod in collect_snods(&bytes, btree) {
            for i in 0..le_u16(&bytes, snod + 6) as usize {
                let offset = le_u64(&bytes, snod + 8 + i * 40);
                assert!(offset > previous, "heap offsets must ascend: {offset}");
                previous = offset;
            }
        }
        assert!(heap_data > 0);
    }

    /// A B-tree node occupies its full `2K`-child width, tail included.
    ///
    /// The same fixed-image rule that governs SNODs governs the B-tree: the
    /// node is written at [`GROUP_NODE_SIZE`] with every slot past
    /// `entries_used` zero-filled.
    #[test]
    fn w1a_btree_node_is_full_width() {
        let bytes = w1a_numbered_datasets(9);
        let btree = le_u64(&bytes, 80) as usize;
        let heap_hdr = le_u64(&bytes, 88) as usize;

        // One level-0 node, and the local heap header starts immediately after
        // it — so the node occupies exactly GROUP_NODE_SIZE bytes.
        assert_eq!(bytes[btree + 5], 0, "level");
        assert_eq!(
            heap_hdr - btree,
            GROUP_NODE_SIZE,
            "B-tree node must be written at full width"
        );

        let entries = le_u16(&bytes, btree + 6) as usize;
        assert_eq!(entries, 2, "two SNOD children");
        // 24-byte prefix, key[0], then one child/key pair per entry.
        let used = 24 + 8 + entries * 16;
        assert!(
            bytes[btree + used..btree + GROUP_NODE_SIZE]
                .iter()
                .all(|&b| b == 0),
            "slots past entries_used must be zero-filled"
        );
    }

    /// More than 256 links no longer fit one node's worth of SNODs, so the
    /// B-tree grows a level — and everything must still be readable.
    #[test]
    fn w1a_three_hundred_datasets_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1a_300.h5");
        let bytes = w1a_numbered_datasets(300);
        std::fs::write(&tmp, &bytes).expect("write");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        // 300 links -> 38 SNODs -> two level-0 nodes under one level-1 root.
        let btree = le_u64(&bytes, 80) as usize;
        assert_eq!(bytes[btree + 5], 1, "root must sit one level up");
        assert_eq!(le_u16(&bytes, btree + 6), 2, "two level-0 children");
        assert_eq!(collect_snods(&bytes, btree).len(), 38);

        let names = f.dataset_names().expect("names");
        assert_eq!(names.len(), 300);
        for i in 0..300usize {
            let name = format!("ds{i:03}");
            let ds = f.dataset(&name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(ds.as_i32().expect("as_i32"), vec![i as i32]);
        }
    }

    /// The superblock's `leaf_node_K` and the SNOD writer must agree.
    ///
    /// They are one number in two places: libhdf5 sizes the node image from the
    /// field, oxih5 sizes it from the constant.  A file once carried three
    /// different K's at once — superblock 4, root SNOD 32, group SNODs 16 —
    /// which is unrepresentable, because `leaf_node_K` is per file.
    #[test]
    fn w1a_superblock_k_matches_snod_capacity() {
        let bytes = w1a_numbered_datasets(3);
        let leaf_k = le_u16(&bytes, 16) as usize;
        assert_eq!(leaf_k, SYM_LEAF_K);
        assert_eq!(bytes[16], 4, "libhdf5's own default leaf_node_K");
        assert_eq!(le_u16(&bytes, 18) as usize, GROUP_INTERNAL_K);
        assert_eq!(8 + 2 * leaf_k * 40, SNOD_SIZE);

        // Same K for the root group and for a sub-group: it is a per-file field.
        let mut w = FileWriter::new();
        w.write_dataset_f64("root_ds", &[1.0], &[1]).expect("ds");
        w.create_group("grp").expect("grp");
        w.write_group_dataset_i32("grp", "inner", &[1], &[1])
            .expect("inner");
        let bytes = w.build_to_vec().expect("build_to_vec");

        let (root_btree, _) = root_symbol_table(&bytes);
        let mut all_snods = collect_snods(&bytes, root_btree);
        for snod in all_snods.clone() {
            for i in 0..le_u16(&bytes, snod + 6) as usize {
                let ste = snod + 8 + i * 40;
                if le_u64(&bytes, ste + 16) & 0xffff_ffff == 1 {
                    // cache_type 1: a sub-group, B-tree cached in the scratch pad.
                    all_snods.extend(collect_snods(&bytes, le_u64(&bytes, ste + 24) as usize));
                }
            }
        }
        assert_eq!(all_snods.len(), 2, "root SNOD + sub-group SNOD");
        for snod in all_snods {
            assert_eq!(&bytes[snod..snod + 4], b"SNOD");
            assert!(
                le_u16(&bytes, snod + 6) as usize <= SNOD_MAX_ENTRIES,
                "every SNOD in the file obeys the one declared leaf_node_K"
            );
        }
    }

    /// Declaring links in descending order must not lose any of them.
    ///
    /// This is the regression that motivated sorting: a NetCDF-style file that
    /// declared `values` before `lat` lost `values` entirely, because the
    /// binary search in `H5G__node_found` walked past it.
    #[test]
    fn w1a_reverse_declaration_order_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1a_reverse.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("zzz", &[3.0], &[1]).expect("zzz");
        w.write_dataset_f64("mmm", &[2.0], &[1]).expect("mmm");
        w.write_dataset_f64("aaa", &[1.0], &[1]).expect("aaa");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        for (name, want) in [("zzz", 3.0), ("mmm", 2.0), ("aaa", 1.0)] {
            let ds = f.dataset(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(ds.as_f64().expect("as_f64"), vec![want]);
        }
        let mut names = f.dataset_names().expect("names");
        names.sort();
        assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
    }
}
