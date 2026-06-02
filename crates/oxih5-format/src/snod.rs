use crate::superblock::{read_u16_le, read_u64_le};
use oxih5_core::OxiH5Error;

/// One symbol-table entry from a SNOD node.
/// For soo=8 (the only size we support in M1), each entry is exactly 40 bytes.
#[derive(Debug, Clone)]
pub struct SymTabEntry {
    /// Byte offset of the entry's name within the group's local heap data segment.
    pub name_offset: u64,
    /// Absolute file address of the object (dataset) header.
    pub object_header_address: u64,
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
/// 24      16     Scratch-pad area (8 bytes B-tree addr + 8 bytes heap addr, if cache_type=1)
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
        // e+16 = cache_type (4), e+20 = reserved (4), e+24 = scratch (16) — ignored for M1.
        entries.push(SymTabEntry {
            name_offset,
            object_header_address,
        });
    }

    Ok(entries)
}
