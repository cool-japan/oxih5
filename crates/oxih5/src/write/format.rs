//! Low-level HDF5 binary primitives.
//!
//! This module is a leaf of the writer: it emits fixed-shape byte structures —
//! little-endian integers, object-header message headers, the superblock,
//! local heaps and symbol table nodes — and knows nothing about datatypes,
//! attributes, or object-header contents.  Everything whose length varies with
//! user data lives in [`super::elem`] and [`super::oh`]; the shape of a group's
//! symbol table, including the `leaf_node_K` this superblock declares, lives in
//! [`super::btree_v1`].
//!
//! Every structure writer returns the number of bytes it owns, so callers can
//! feed that straight into [`super::check_size`].

use oxih5_core::OxiH5Error;

use super::btree_v1::{self, SNOD_MAX_ENTRIES, SNOD_PREFIX, SNOD_SIZE, STE_SIZE};
use super::narrow;

// ---------------------------------------------------------------------------
// Byte-write helpers
// ---------------------------------------------------------------------------

/// Write a little-endian `u16` at `offset`.
#[inline]
pub(super) fn write_u16_le(buf: &mut [u8], offset: usize, val: u16) {
    buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `u32` at `offset`.
#[inline]
pub(super) fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `u64` at `offset`.
#[inline]
pub(super) fn write_u64_le(buf: &mut [u8], offset: usize, val: u64) {
    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

/// Zero `len` bytes at `start`.
///
/// Body writers call this first so that the bytes they emit never depend on the
/// caller having handed them a pre-zeroed buffer.
#[inline]
pub(super) fn fill_zero(buf: &mut [u8], start: usize, len: usize) {
    buf[start..start + len].fill(0);
}

/// Size of an object-header message header: type, body size, flags, reserved.
pub(super) const MSG_HDR_SIZE: usize = 8;

/// Write an 8-byte object header message header.
pub(super) fn write_msg_header(
    buf: &mut [u8],
    offset: usize,
    msg_type: u16,
    body_size: u16,
    flags: u8,
) {
    fill_zero(buf, offset, MSG_HDR_SIZE);
    write_u16_le(buf, offset, msg_type);
    write_u16_le(buf, offset + 2, body_size);
    buf[offset + 4] = flags;
    // bytes 5..8 are reserved and stay zero
}

// ---------------------------------------------------------------------------
// HDF5 signature + superblock v0
// ---------------------------------------------------------------------------

/// Size of the HDF5 file signature.
pub(super) const SIGNATURE_SIZE: usize = 8;

/// Address of the root group's object header, fixed by the superblock layout.
pub(super) const ROOT_OH_ADDR: usize = 96;

/// Write the 8-byte HDF5 signature; returns bytes written.
pub(super) fn write_signature(buf: &mut [u8]) -> usize {
    buf[0..SIGNATURE_SIZE].copy_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);
    SIGNATURE_SIZE
}

/// Narrow a compile-time geometry constant to the width of its superblock
/// field.
///
/// Used only to initialise `const` items, so the assertion fails the *build*
/// rather than a write if the geometry ever stops fitting.
const fn as_u16(value: usize) -> u16 {
    assert!(value <= u16::MAX as usize);
    value as u16
}

/// `leaf_node_K` as the superblock encodes it.
///
/// libhdf5 sizes a symbol table node's on-disk image from this field alone —
/// `8 + 2*K*40` — so it must be the same K that
/// [`btree_v1::SNOD_SIZE`](super::btree_v1::SNOD_SIZE) is built from.  Reading
/// it from there rather than repeating a literal is what makes the superblock
/// and the nodes provably agree.
const SYM_LEAF_K_FIELD: u16 = as_u16(btree_v1::SYM_LEAF_K);

/// `internal_node_K` as the superblock encodes it.
const GROUP_INTERNAL_K_FIELD: u16 = as_u16(btree_v1::GROUP_INTERNAL_K);

/// Write superblock v0 into `buf[8..96]`; returns bytes written.
///
/// The trailing root-group symbol table entry points at the root object header
/// at [`ROOT_OH_ADDR`], whose symbol table message duplicates `btree_addr` and
/// `heap_addr`.
pub(super) fn write_superblock(
    buf: &mut [u8],
    btree_addr: usize,
    heap_addr: usize,
    eof_addr: u64,
) -> usize {
    fill_zero(buf, SIGNATURE_SIZE, ROOT_OH_ADDR - SIGNATURE_SIZE);
    buf[8] = 0x00; // superblock version 0
    buf[9] = 0x00; // free-space version
    buf[10] = 0x00; // root group STE version
    buf[11] = 0x00; // reserved
    buf[12] = 0x00; // shared header msg version
    buf[13] = 0x08; // size_of_offsets = 8
    buf[14] = 0x08; // size_of_lengths = 8
    buf[15] = 0x00; // reserved
    write_u16_le(buf, 16, SYM_LEAF_K_FIELD); // leaf_node_K
    write_u16_le(buf, 18, GROUP_INTERNAL_K_FIELD); // internal_node_K
    write_u32_le(buf, 20, 0); // file consistency flags

    write_u64_le(buf, 24, 0); // base address
    write_u64_le(buf, 32, u64::MAX); // free space address (undefined)
    write_u64_le(buf, 40, eof_addr); // end of file
    write_u64_le(buf, 48, u64::MAX); // driver info block (undefined)

    // Root Group Symbol Table Entry
    write_u64_le(buf, 56, 0); // link_name_offset = 0
    write_u64_le(buf, 64, ROOT_OH_ADDR as u64); // root group OH address
    write_u32_le(buf, 72, 1); // cache_type = 1 (root group)
    write_u32_le(buf, 76, 0); // reserved
    write_u64_le(buf, 80, btree_addr as u64); // B-tree address
    write_u64_le(buf, 88, heap_addr as u64); // local heap address

    ROOT_OH_ADDR - SIGNATURE_SIZE
}

// ---------------------------------------------------------------------------
// Local heap — header + data segment
// ---------------------------------------------------------------------------

/// Fixed size of a local heap header.
pub(super) const HEAP_HEADER_SIZE: usize = 32;

/// Write a local heap header at `base`; returns bytes written.
///
/// `data_addr` is the absolute address of the heap data segment, `data_size`
/// its total allocated size, and `used_size` how much of it is occupied — the
/// free list starts at that offset.
pub(super) fn write_local_heap(
    buf: &mut [u8],
    base: usize,
    data_addr: usize,
    data_size: usize,
    used_size: usize,
) -> usize {
    fill_zero(buf, base, HEAP_HEADER_SIZE);
    buf[base..base + 4].copy_from_slice(b"HEAP");
    buf[base + 4] = 0x00; // version = 0
                          // bytes 5..8 = reserved
    write_u64_le(buf, base + 8, data_size as u64); // data segment size
    write_u64_le(buf, base + 16, used_size as u64); // first free block offset
    write_u64_le(buf, base + 24, data_addr as u64); // data segment address
    HEAP_HEADER_SIZE
}

// ---------------------------------------------------------------------------
// SNOD — Symbol Table Node
// ---------------------------------------------------------------------------

/// What one symbol table entry says about its link.
///
/// A version-1 symbol table entry has no link-type field.  What it has is a
/// *cache type*, and that one number is how the format distinguishes a plain
/// object from a group from a soft link — so the three cases are one enum here
/// rather than an address plus a flag that can disagree with it.
pub(super) enum SnodValue {
    /// Cache type 0: a plain object at this header address, nothing cached.
    Object(u64),
    /// Cache type 1: a group at this header address, with its own B-tree and
    /// local heap cached in the scratch pad so a reader can descend without
    /// first parsing the group's Symbol Table message.
    Group {
        /// Object-header address of the sub-group.
        oh_addr: u64,
        /// The sub-group's symbol table B-tree root.
        btree_addr: u64,
        /// The sub-group's local heap header.
        heap_addr: u64,
    },
    /// Cache type 2: a **soft link**.  It has no object header at all, so the
    /// address field carries the undefined-address sentinel and the scratch pad
    /// carries the offset of the target path within *this* group's local heap.
    SoftLink {
        /// Byte offset of the NUL-terminated target path in the local heap.
        link_value_offset: u32,
    },
}

/// One symbol table entry.
pub(super) struct SnodEntry {
    /// Offset of the link name within the enclosing local heap.
    pub(super) name_offset: u64,
    /// What the entry links to, and how the cache type says so.
    pub(super) value: SnodValue,
}

/// Write a SNOD holding `entries`; returns bytes written, always
/// [`SNOD_SIZE`](super::btree_v1::SNOD_SIZE).
///
/// Every SNOD in the file is emitted at that one fixed width, with the slots
/// past `entries.len()` zero-filled, because libhdf5 sizes the image it reads
/// from the superblock's `leaf_node_K` and not from the node's own `nsyms` —
/// see [`super::btree_v1`].
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `entries` overruns
/// [`SNOD_MAX_ENTRIES`](super::btree_v1::SNOD_MAX_ENTRIES), or if the entry
/// count does not fit the 16-bit on-disk field.
pub(super) fn write_snod(
    buf: &mut [u8],
    snod_addr: usize,
    entries: &[SnodEntry],
) -> Result<usize, OxiH5Error> {
    if entries.len() > SNOD_MAX_ENTRIES {
        return Err(OxiH5Error::Format(format!(
            "internal writer error: symbol table node holds {} entries, capacity {SNOD_MAX_ENTRIES}",
            entries.len()
        )));
    }
    let total = SNOD_SIZE;
    fill_zero(buf, snod_addr, total);

    buf[snod_addr..snod_addr + 4].copy_from_slice(b"SNOD");
    buf[snod_addr + 4] = 0x01; // version = 1
    buf[snod_addr + 5] = 0x00; // reserved
    write_u16_le(
        buf,
        snod_addr + 6,
        narrow::<u16>("symbol table entry count", entries.len())?,
    );

    for (i, entry) in entries.iter().enumerate() {
        let ste = snod_addr + SNOD_PREFIX + i * STE_SIZE;
        write_u64_le(buf, ste, entry.name_offset);
        match entry.value {
            SnodValue::Object(oh_addr) => {
                write_u64_le(buf, ste + 8, oh_addr);
                // cache_type = 0 and an all-zero scratch pad.
            }
            SnodValue::Group {
                oh_addr,
                btree_addr,
                heap_addr,
            } => {
                write_u64_le(buf, ste + 8, oh_addr);
                write_u32_le(buf, ste + 16, 1); // cache_type = 1 (group)
                write_u32_le(buf, ste + 20, 0); // reserved
                write_u64_le(buf, ste + 24, btree_addr); // scratch: B-tree addr
                write_u64_le(buf, ste + 32, heap_addr); // scratch: local heap addr
            }
            SnodValue::SoftLink { link_value_offset } => {
                // A soft link has no object header; libhdf5 writes the
                // undefined-address sentinel here and reads a *defined* address
                // beside cache type 2 as corruption.
                write_u64_le(buf, ste + 8, u64::MAX);
                write_u32_le(buf, ste + 16, 2); // cache_type = 2 (symbolic link)
                write_u32_le(buf, ste + 20, 0); // reserved
                                                // The scratch pad's first four
                                                // bytes are the link value's
                                                // offset into the local heap;
                                                // the remaining twelve stay zero.
                write_u32_le(buf, ste + 24, link_value_offset);
            }
        }
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_and_superblock_fill_the_root_oh_prefix() {
        let mut buf = vec![0u8; ROOT_OH_ADDR];
        let sig = write_signature(&mut buf);
        let sb = write_superblock(&mut buf, 200, 300, 4096);
        assert_eq!(sig + sb, ROOT_OH_ADDR);
        assert_eq!(&buf[0..4], &[0x89, b'H', b'D', b'F']);
        assert_eq!(u64::from_le_bytes(buf[40..48].try_into().unwrap()), 4096);
        assert_eq!(
            u64::from_le_bytes(buf[64..72].try_into().unwrap()),
            ROOT_OH_ADDR as u64
        );
    }

    /// The superblock must declare the very `leaf_node_K` the SNOD writer is
    /// built from: libhdf5 sizes a node's image from the field, our writer
    /// sizes it from the constant, and they are the same number by
    /// construction.
    #[test]
    fn superblock_declares_the_geometry_the_nodes_are_built_from() {
        let mut buf = vec![0u8; ROOT_OH_ADDR];
        write_superblock(&mut buf, 200, 300, 4096);
        let leaf_k = u16::from_le_bytes([buf[16], buf[17]]) as usize;
        let internal_k = u16::from_le_bytes([buf[18], buf[19]]) as usize;
        assert_eq!(leaf_k, btree_v1::SYM_LEAF_K);
        assert_eq!(internal_k, btree_v1::GROUP_INTERNAL_K);
        // The arithmetic H5G__cache_node_deserialize performs on the field.
        assert_eq!(SNOD_PREFIX + 2 * leaf_k * STE_SIZE, SNOD_SIZE);
    }

    #[test]
    fn snod_entries_are_written_at_forty_byte_stride() {
        let mut buf = vec![0u8; SNOD_SIZE];
        let entries = vec![
            SnodEntry {
                name_offset: 8,
                value: SnodValue::Object(0x1000),
            },
            SnodEntry {
                name_offset: 16,
                value: SnodValue::Group {
                    oh_addr: 0x2000,
                    btree_addr: 0x3000,
                    heap_addr: 0x4000,
                },
            },
        ];
        let wrote = write_snod(&mut buf, 0, &entries).expect("write_snod");
        assert_eq!(wrote, SNOD_SIZE);
        assert_eq!(&buf[0..4], b"SNOD");
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 2);
        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), 8);
        // Second entry is a group: cache_type 1 plus the scratch-pad addresses.
        assert_eq!(u32::from_le_bytes(buf[64..68].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(buf[72..80].try_into().unwrap()), 0x3000);
        assert_eq!(u64::from_le_bytes(buf[80..88].try_into().unwrap()), 0x4000);
        // Unused slots stay zero — libhdf5 decodes all 2K of them regardless.
        assert!(buf[88..].iter().all(|&b| b == 0));
    }

    /// Byte-pinned against the symbol table entry libhdf5 2.0.0 wrote for
    /// `f['soft'] = h5py.SoftLink('/d')`: an undefined object header address,
    /// cache type 2, and the heap offset of the target path in the first four
    /// bytes of the scratch pad.
    #[test]
    fn a_soft_link_entry_matches_libhdf5() {
        let mut buf = vec![0u8; SNOD_SIZE];
        let entries = vec![SnodEntry {
            name_offset: 16,
            value: SnodValue::SoftLink {
                link_value_offset: 24,
            },
        }];
        write_snod(&mut buf, 0, &entries).expect("write_snod");

        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), 16);
        assert_eq!(
            u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            u64::MAX,
            "a soft link has no object header"
        );
        assert_eq!(u32::from_le_bytes(buf[24..28].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(buf[28..32].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(buf[32..36].try_into().unwrap()), 24);
        assert!(
            buf[36..48].iter().all(|&b| b == 0),
            "the rest of the scratch pad stays zero"
        );
    }

    #[test]
    fn snod_rejects_more_entries_than_one_node_can_hold() {
        let mut buf = vec![0u8; SNOD_SIZE];
        let entries: Vec<SnodEntry> = (0..SNOD_MAX_ENTRIES + 1)
            .map(|_| SnodEntry {
                name_offset: 0,
                value: SnodValue::Object(0),
            })
            .collect();
        assert!(write_snod(&mut buf, 0, &entries).is_err());
    }
}
