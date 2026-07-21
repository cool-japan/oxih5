use crate::superblock::{read_u16_le, read_u32_le, read_u64_le};
use oxih5_core::OxiH5Error;

/// The scratch-pad payload of a symbol-table entry, decoded per its cache type.
///
/// The 16-byte scratch pad at the end of every entry is interpreted differently
/// depending on the entry's 4-byte cache-type field, so the two are decoded
/// together into one value rather than exposed as raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymTabCache {
    /// Cache type 0 — nothing is cached and the scratch pad is unused.
    None,
    /// Cache type 1 — the entry names a group, and that group's B-tree and
    /// local-heap addresses are cached here (saving a read of the group's own
    /// Symbol Table message).
    Group {
        btree_address: u64,
        heap_address: u64,
    },
    /// Cache type 2 — the entry is a **soft link**.  Its
    /// `object_header_address` is the HDF5 undefined-address sentinel
    /// (`u64::MAX`) because a soft link has no target object header of its own;
    /// the link value is instead a NUL-terminated path stored in the containing
    /// group's local heap, starting at this byte offset into the heap's data
    /// segment.
    SoftLink { link_value_offset: u32 },
    /// A cache type this parser does not interpret.  Preserved verbatim so
    /// callers can name it in a diagnostic instead of silently mis-reading the
    /// scratch pad.
    Unknown(u32),
}

/// One symbol-table entry from a SNOD node.
/// For soo=8 (the only size we support in M1), each entry is exactly 40 bytes.
#[derive(Debug, Clone)]
pub struct SymTabEntry {
    /// Byte offset of the entry's name within the group's local heap data segment.
    pub name_offset: u64,
    /// Absolute file address of the object (dataset) header.
    ///
    /// For a soft link (see [`SymTabCache::SoftLink`]) this is `u64::MAX`, the
    /// HDF5 "undefined address" sentinel — **not** a usable address.  Check
    /// [`SymTabEntry::cache`] before dereferencing it.
    pub object_header_address: u64,
    /// The entry's decoded cache type and scratch-pad contents.
    pub cache: SymTabCache,
}

impl SymTabEntry {
    /// The local-heap offset of this entry's soft-link value, or `None` when
    /// the entry is not a soft link.
    pub fn soft_link_value_offset(&self) -> Option<u32> {
        match self.cache {
            SymTabCache::SoftLink { link_value_offset } => Some(link_value_offset),
            _ => None,
        }
    }

    /// `true` when this entry's object-header address is usable, i.e. it is a
    /// hard link rather than a soft link or an entry with an undefined address.
    pub fn has_object_header(&self) -> bool {
        self.object_header_address != u64::MAX
            && !matches!(self.cache, SymTabCache::SoftLink { .. })
    }
}

/// Size of one symbol table entry for size_of_offsets=8.
/// Layout: link_name_offset(8) + object_header_address(8) + cache_type(4) + reserved(4) + scratch(16) = 40
const STE_SIZE: usize = 40;

/// Parse all symbol table entries from a SNOD (symbol table node) at `snod_address`.
///
/// SNOD layout:
/// ```text
/// Offset  Size  Field
///  0       4     Signature "SNOD"
///  4       1     Version (must be 1)
///  5       1     Reserved
///  6       2     Number of symbols K (u16 LE)
///  8       K*40  Symbol table entries (40 bytes each for soo=8)
/// ```
///
/// Each 40-byte entry:
/// ```text
///  0       8     Link name offset (offset into local heap data segment)
///  8       8     Object header address (absolute file offset)
/// 16       4     Cache type (0=no cache, 1=group, 2=soft link)
/// 20       4     Reserved
/// 24      16     Scratch-pad area, interpreted per cache type:
///                  type 1: 8 bytes B-tree address + 8 bytes local-heap address
///                  type 2: 4 bytes local-heap offset of the soft-link value
/// ```
pub fn parse(file_data: &[u8], snod_address: u64) -> Result<Vec<SymTabEntry>, OxiH5Error> {
    let off = usize::try_from(snod_address).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "SNOD address {snod_address} exceeds addressable range"
        ))
    })?;
    let off8 = off
        .checked_add(8)
        .ok_or_else(|| OxiH5Error::Corrupted(format!("SNOD address {snod_address} too large")))?;

    if off8 > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "SNOD at {snod_address}: header out of bounds (file len={})",
            file_data.len()
        )));
    }

    if &file_data[off..off + 4] != b"SNOD" {
        return Err(OxiH5Error::Format(format!(
            "no SNOD signature at {snod_address}: got {:?}",
            &file_data[off..off + 4]
        )));
    }

    let version = file_data[off + 4];
    if version != 1 {
        return Err(OxiH5Error::Format(format!(
            "unsupported SNOD version: {version}"
        )));
    }

    let num_symbols = read_u16_le(file_data, off + 6)? as usize;
    let entries_start = off8;
    let required_end = num_symbols
        .checked_mul(STE_SIZE)
        .and_then(|n| entries_start.checked_add(n))
        .ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "SNOD at {snod_address}: entries overflow with {num_symbols} symbols"
            ))
        })?;

    if required_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "SNOD at {snod_address}: {num_symbols} entries require {required_end} bytes \
             but file only has {}",
            file_data.len()
        )));
    }

    let mut entries = Vec::with_capacity(num_symbols);
    for i in 0..num_symbols {
        let e = entries_start + i * STE_SIZE;
        let name_offset = read_u64_le(file_data, e)?;
        let object_header_address = read_u64_le(file_data, e + 8)?;
        // e+16 = cache_type (4), e+20 = reserved (4), e+24 = scratch pad (16).
        // Every read below stays inside the 40 bytes already bounds-checked by
        // `required_end`, so none of them can run off the end of `file_data`.
        let cache_type = read_u32_le(file_data, e + 16)?;
        let cache = match cache_type {
            0 => SymTabCache::None,
            1 => SymTabCache::Group {
                btree_address: read_u64_le(file_data, e + 24)?,
                heap_address: read_u64_le(file_data, e + 32)?,
            },
            2 => SymTabCache::SoftLink {
                link_value_offset: read_u32_le(file_data, e + 24)?,
            },
            other => SymTabCache::Unknown(other),
        };
        entries.push(SymTabEntry {
            name_offset,
            object_header_address,
            cache,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory SNOD holding the given `(name_offset,
    /// object_header_address, cache_type, scratch)` entries.
    fn snod_with(entries: &[(u64, u64, u32, [u8; 16])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SNOD");
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (name_offset, oh_addr, cache_type, scratch) in entries {
            buf.extend_from_slice(&name_offset.to_le_bytes());
            buf.extend_from_slice(&oh_addr.to_le_bytes());
            buf.extend_from_slice(&cache_type.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
            buf.extend_from_slice(scratch);
        }
        buf
    }

    #[test]
    fn decodes_all_three_cache_types() {
        let mut group_scratch = [0u8; 16];
        group_scratch[..8].copy_from_slice(&1440u64.to_le_bytes());
        group_scratch[8..].copy_from_slice(&4096u64.to_le_bytes());

        let mut soft_scratch = [0u8; 16];
        soft_scratch[..4].copy_from_slice(&32u32.to_le_bytes());

        let data = snod_with(&[
            (8, 800, 0, [0u8; 16]),
            (16, 1400, 1, group_scratch),
            (24, u64::MAX, 2, soft_scratch),
        ]);

        let entries = parse(&data, 0).expect("parse SNOD");
        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].cache, SymTabCache::None);
        assert!(entries[0].has_object_header());
        assert_eq!(entries[0].soft_link_value_offset(), None);

        assert_eq!(
            entries[1].cache,
            SymTabCache::Group {
                btree_address: 1440,
                heap_address: 4096,
            }
        );
        assert!(entries[1].has_object_header());

        assert_eq!(
            entries[2].cache,
            SymTabCache::SoftLink {
                link_value_offset: 32
            }
        );
        assert_eq!(entries[2].soft_link_value_offset(), Some(32));
        // The undefined-address sentinel must never be treated as an address.
        assert!(!entries[2].has_object_header());
    }

    #[test]
    fn unknown_cache_type_is_preserved_not_guessed() {
        let data = snod_with(&[(8, 800, 7, [0xAAu8; 16])]);
        let entries = parse(&data, 0).expect("parse SNOD");
        assert_eq!(entries[0].cache, SymTabCache::Unknown(7));
    }

    #[test]
    fn truncated_entry_table_is_rejected() {
        let mut data = snod_with(&[(8, 800, 0, [0u8; 16])]);
        data.truncate(data.len() - 1);
        assert!(parse(&data, 0).is_err());
    }
}
