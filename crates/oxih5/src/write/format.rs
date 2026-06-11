//! Low-level HDF5 binary format writers.
//!
//! Functions in this module write fixed-structure HDF5 elements:
//! superblock, object headers, B-tree nodes, local heaps, and SNOD nodes.

// ---------------------------------------------------------------------------
// Byte-write helpers (pub(super) so mod.rs and messages.rs can use them)
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn write_u16_le(buf: &mut [u8], offset: usize, val: u16) {
    buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

#[inline]
pub(super) fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
pub(super) fn write_u64_le(buf: &mut [u8], offset: usize, val: u64) {
    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

/// Write an 8-byte object header message header.
pub(super) fn write_msg_header(buf: &mut [u8], offset: usize, msg_type: u16, body_size: u16) {
    write_u16_le(buf, offset, msg_type);
    write_u16_le(buf, offset + 2, body_size);
    // flags = 0, reserved = 0 (already zero)
}

/// Write an 8-byte object header message header with explicit flags byte.
pub(super) fn write_msg_header_flags(
    buf: &mut [u8],
    offset: usize,
    msg_type: u16,
    body_size: u16,
    flags: u8,
) {
    write_u16_le(buf, offset, msg_type);
    write_u16_le(buf, offset + 2, body_size);
    buf[offset + 4] = flags;
}

// ---------------------------------------------------------------------------
// HDF5 signature
// ---------------------------------------------------------------------------

pub(super) fn write_signature(buf: &mut [u8]) {
    buf[0..8].copy_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);
}

// ---------------------------------------------------------------------------
// Superblock v0 (bytes 8..96) — parameterized addresses
// ---------------------------------------------------------------------------

pub(super) fn write_superblock(buf: &mut [u8], btree_addr: usize, heap_addr: usize, eof_addr: u64) {
    buf[8] = 0x00; // superblock version 0
    buf[9] = 0x00; // free-space version
    buf[10] = 0x00; // root group STE version
    buf[11] = 0x00; // reserved
    buf[12] = 0x00; // shared header msg version
    buf[13] = 0x08; // size_of_offsets = 8
    buf[14] = 0x08; // size_of_lengths = 8
    buf[15] = 0x00; // reserved
    write_u16_le(buf, 16, 4); // leaf_node_K = 4
    write_u16_le(buf, 18, 16); // internal_node_K = 16
    write_u32_le(buf, 20, 0); // file consistency flags

    write_u64_le(buf, 24, 0); // base address
    write_u64_le(buf, 32, u64::MAX); // free space address (undefined)
    write_u64_le(buf, 40, eof_addr); // end of file
    write_u64_le(buf, 48, u64::MAX); // driver info block (undefined)

    // Root Group Symbol Table Entry
    write_u64_le(buf, 56, 0); // link_name_offset = 0
    write_u64_le(buf, 64, 96); // root group OH address = 96
    write_u32_le(buf, 72, 1); // cache_type = 1 (root group)
    write_u32_le(buf, 76, 0); // reserved
    write_u64_le(buf, 80, btree_addr as u64); // B-tree address
    write_u64_le(buf, 88, heap_addr as u64); // local heap address
}

// ---------------------------------------------------------------------------
// Root group object header v1 at address 96 — parameterized + root attrs
// ---------------------------------------------------------------------------

/// Compute the size (bytes) of the root group OH for the given root string attrs.
pub(super) fn compute_root_oh_size(root_str_attrs: &[(String, String)]) -> usize {
    let sym_tab_msg = super::messages::msg_total(16); // Symbol Table msg body = 16
    let attr_total: usize = root_str_attrs
        .iter()
        .map(|(name, val)| {
            super::messages::msg_total(super::messages::attr_body_size_str(name, val))
        })
        .sum();
    16 + sym_tab_msg + attr_total // 16-byte OH prefix
}

/// Write the root group OH at buf[96..96+oh_size].
pub(super) fn write_root_oh(
    buf: &mut [u8],
    btree_addr: usize,
    heap_addr: usize,
    root_str_attrs: &[(String, String)],
    oh_size: usize,
) {
    const BASE: usize = 96;
    let sym_tab_body = 16u16;
    let sym_tab_msg_total = super::messages::msg_total(16);
    let attr_total: usize = root_str_attrs
        .iter()
        .map(|(name, val)| {
            super::messages::msg_total(super::messages::attr_body_size_str(name, val))
        })
        .sum();

    let header_data_size = sym_tab_msg_total + attr_total;
    let num_messages = 1 + root_str_attrs.len();

    // OH prefix
    buf[BASE] = 0x01;
    buf[BASE + 1] = 0x00;
    write_u16_le(buf, BASE + 2, num_messages as u16);
    write_u32_le(buf, BASE + 4, 1);
    write_u32_le(buf, BASE + 8, header_data_size as u32);
    write_u32_le(buf, BASE + 12, 0);

    let mut pos = BASE + 16;

    // Symbol Table message (type 0x0011)
    write_msg_header(buf, pos, 0x0011, sym_tab_body);
    write_u64_le(buf, pos + 8, btree_addr as u64);
    write_u64_le(buf, pos + 16, heap_addr as u64);
    pos += sym_tab_msg_total;

    // Root group string attribute messages
    for (name, val) in root_str_attrs {
        let body_sz = super::messages::attr_body_size_str(name, val);
        write_msg_header(buf, pos, 0x000C, body_sz as u16);
        super::messages::write_str_attr_body(buf, pos + 8, name, val);
        pos += super::messages::msg_total(body_sz);
    }
    let _ = (pos, oh_size);
}

// ---------------------------------------------------------------------------
// Group object header (for sub-groups, W0b)
// ---------------------------------------------------------------------------

/// Write a sub-group object header (40 bytes: 16-byte prefix + Symbol Table msg).
pub(super) fn write_group_oh(
    buf: &mut [u8],
    base: usize,
    grp_btree_addr: usize,
    grp_heap_addr: usize,
) {
    buf[base] = 0x01; // version = 1
    buf[base + 1] = 0x00; // reserved
    write_u16_le(buf, base + 2, 1); // num_messages = 1
    write_u32_le(buf, base + 4, 1); // reference count = 1
    write_u32_le(buf, base + 8, 24); // header_data_size = 24 (sym table msg)
    write_u32_le(buf, base + 12, 0); // reserved

    // Symbol Table message
    write_msg_header(buf, base + 16, 0x0011, 16);
    write_u64_le(buf, base + 24, grp_btree_addr as u64);
    write_u64_le(buf, base + 32, grp_heap_addr as u64);
}

/// Fixed size of a sub-group OH = 40 bytes.
pub(super) const GROUP_OH_SIZE: usize = 40;

// ---------------------------------------------------------------------------
// B-tree v1 leaf node — group B-tree (node type 0)
// ---------------------------------------------------------------------------

/// Write a B-tree v1 group leaf node (48 bytes) at `base`.
/// `snod_addr` is the address of the single child SNOD.
/// `key1` is the last name offset in the associated heap (upper key).
pub(super) fn write_btree_leaf(buf: &mut [u8], base: usize, snod_addr: u64, key1: u64) {
    buf[base..base + 4].copy_from_slice(b"TREE");
    buf[base + 4] = 0x00; // node_type = 0 (group)
    buf[base + 5] = 0x00; // level = 0 (leaf)
    write_u16_le(buf, base + 6, 1); // entries_used = 1
    write_u64_le(buf, base + 8, u64::MAX); // left sibling (undefined)
    write_u64_le(buf, base + 16, u64::MAX); // right sibling (undefined)
    write_u64_le(buf, base + 24, 0); // key[0]
    write_u64_le(buf, base + 32, snod_addr); // child[0]
    write_u64_le(buf, base + 40, key1); // key[1]
}

/// Fixed size of a group B-tree leaf node = 48 bytes.
pub(super) const BTREE_LEAF_SIZE: usize = 48;

// ---------------------------------------------------------------------------
// Local heap — header + data segment
// ---------------------------------------------------------------------------

/// Write local heap header (32 bytes) at `base`.
/// `data_addr` is the absolute address of the heap data segment.
/// `data_size` is the total allocated size of the data segment.
/// `used_size` is how many bytes are actually used (free list starts here).
pub(super) fn write_local_heap(
    buf: &mut [u8],
    base: usize,
    data_addr: usize,
    data_size: usize,
    used_size: usize,
) {
    buf[base..base + 4].copy_from_slice(b"HEAP");
    buf[base + 4] = 0x00; // version = 0
                          // bytes 5..8 = reserved (already zero)
    write_u64_le(buf, base + 8, data_size as u64); // data segment size
    write_u64_le(buf, base + 16, used_size as u64); // first free block offset
    write_u64_le(buf, base + 24, data_addr as u64); // data segment address
}

/// Fixed size of a local heap header = 32 bytes.
pub(super) const HEAP_HEADER_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// SNOD — Symbol Table Node
// ---------------------------------------------------------------------------

/// Write a SNOD at `snod_addr` with `n` dataset entries (cache_type=0)
/// and `m` group entries (cache_type=1).
///
/// Dataset entries: name_offsets[0..n], oh_addrs[0..n]
/// Group entries: grp_name_offsets[0..m], grp_oh_addrs[0..m],
///                grp_btree_addrs[0..m], grp_heap_addrs[0..m]
#[allow(clippy::too_many_arguments)]
pub(super) fn write_snod(
    buf: &mut [u8],
    snod_addr: usize,
    name_offsets: &[u64],
    oh_addrs: &[usize],
    grp_name_offsets: &[u64],
    grp_oh_addrs: &[usize],
    grp_btree_addrs: &[usize],
    grp_heap_addrs: &[usize],
) {
    let n_ds = name_offsets.len();
    let n_grp = grp_name_offsets.len();
    let n = n_ds + n_grp;

    let base = snod_addr;
    buf[base..base + 4].copy_from_slice(b"SNOD");
    buf[base + 4] = 0x01; // version = 1
    buf[base + 5] = 0x00; // reserved
    write_u16_le(buf, base + 6, n as u16); // num_symbols

    // Dataset entries (cache_type = 0)
    for i in 0..n_ds {
        let ste = base + 8 + i * 40;
        write_u64_le(buf, ste, name_offsets[i]);
        write_u64_le(buf, ste + 8, oh_addrs[i] as u64);
        // cache_type = 0, reserved = 0, scratch = 0 (already zero)
    }

    // Group entries (cache_type = 1, scratch = btree + heap)
    for i in 0..n_grp {
        let ste = base + 8 + (n_ds + i) * 40;
        write_u64_le(buf, ste, grp_name_offsets[i]);
        write_u64_le(buf, ste + 8, grp_oh_addrs[i] as u64);
        write_u32_le(buf, ste + 16, 1); // cache_type = 1 (group)
        write_u32_le(buf, ste + 20, 0); // reserved
        write_u64_le(buf, ste + 24, grp_btree_addrs[i] as u64); // scratch: B-tree addr
        write_u64_le(buf, ste + 32, grp_heap_addrs[i] as u64); // scratch: heap addr
    }
    // Any remaining entries are zero (already zero from vec initialization).
}
