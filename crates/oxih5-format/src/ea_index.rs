use crate::btree_v2::ChunkRecord;
/// Extensible Array chunk index parser (HDF5 1.10+).
///
/// The Extensible Array (EA) is used for datasets with a single unlimited
/// dimension. This module implements EA header, index block inline elements,
/// EA data block (EADB) parsing, and secondary block (EASB) parsing.
///
/// For datasets whose EA has not yet allocated any data blocks, an empty
/// chunk list is returned without error.
use oxih5_core::OxiH5Error;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse an extensible array chunk index rooted at `header_address`.
///
/// Returns all chunk records stored in the extensible array, or an empty
/// `Vec` if the array has not allocated storage yet (index block address is
/// UNDEF).
///
/// `ndims` is the dataset's dimensionality; it is used to determine the
/// per-element offset field width.
pub fn parse_extensible_array(
    file_data: &[u8],
    header_address: u64,
    ndims: usize,
) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    let base = header_address as usize;

    // -----------------------------------------------------------------------
    // EA Header layout ("EAHD"):
    //  0  4   Signature "EAHD"
    //  4  1   Version (must be 0)
    //  5  1   Client ID
    //  6  1   Element size (bytes per element = bytes per chunk record)
    //  7  1   max_nelmts_bits
    //  8  1   idx_blk_elmts   (elements stored directly in the index block)
    //  9  1   data_blk_min_elmts
    // 10  1   secondary_blk_min_data_block_pointers
    // 11  1   max_dblk_page_nelmts_bits
    // 12  8   num_created_blks
    // 20  8   num_realized_blks
    // 28  8   index_block_address
    // 36  4   Checksum
    // Total: 40 bytes minimum
    // -----------------------------------------------------------------------

    let hdr_end = base
        .checked_add(40)
        .ok_or_else(|| OxiH5Error::Format("EA: header address overflow".into()))?;
    if hdr_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EA: header at {base:#x} exceeds file length {}",
            file_data.len()
        )));
    }

    let sig = &file_data[base..base + 4];
    if sig != b"EAHD" {
        return Err(OxiH5Error::Format(format!(
            "EA: bad header signature {sig:?} at {base:#x}"
        )));
    }

    let version = file_data[base + 4];
    if version != 0 {
        return Err(OxiH5Error::Format(format!(
            "EA: unsupported header version {version}"
        )));
    }

    let element_size = file_data[base + 6] as usize;
    let idx_blk_elmts = file_data[base + 8] as usize;
    // secondary_blk_min_data_block_pointers: informational; used by EASB parsing
    let _sbmin = file_data[base + 10] as usize;

    let idx_blk_addr = u64::from_le_bytes(
        file_data[base + 28..base + 36]
            .try_into()
            .map_err(|_| OxiH5Error::Format("EA: index block addr slice".into()))?,
    );

    if idx_blk_addr == u64::MAX {
        // No index block allocated — empty array.
        return Ok(Vec::new());
    }

    // -----------------------------------------------------------------------
    // EA Index Block layout ("EAIB"):
    //  0  4   Signature "EAIB"
    //  4  1   Version (must be 0)
    //  5  1   Client ID
    //  6  8   Header address (back-pointer)
    // 14  N   Inline elements: idx_blk_elmts * element_size bytes
    // 14+N M  Data block addresses (variable count, not parsed here)
    //   …     Checksum (4 bytes)
    // -----------------------------------------------------------------------

    let ib = idx_blk_addr as usize;
    let ib_hdr_end = ib
        .checked_add(14)
        .ok_or_else(|| OxiH5Error::Format("EA: index block address overflow".into()))?;
    if ib_hdr_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EA: index block at {ib:#x} truncated (file len={})",
            file_data.len()
        )));
    }

    let ib_sig = &file_data[ib..ib + 4];
    if ib_sig != b"EAIB" {
        return Err(OxiH5Error::Format(format!(
            "EA: bad index block signature {ib_sig:?} at {ib:#x}"
        )));
    }

    let ib_version = file_data[ib + 4];
    if ib_version != 0 {
        return Err(OxiH5Error::Format(format!(
            "EA: unsupported index block version {ib_version}"
        )));
    }

    // Parse inline elements starting at offset 14 in the index block.
    let elem_start = ib + 14;
    let elem_area_end = elem_start
        .checked_add(idx_blk_elmts * element_size)
        .ok_or_else(|| OxiH5Error::Format("EA: inline element area overflow".into()))?;

    if elem_area_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EA: index block inline elements truncated (need {elem_area_end}, have {})",
            file_data.len()
        )));
    }

    let mut records = parse_elements(&file_data[elem_start..elem_area_end], element_size, ndims)?;

    // -----------------------------------------------------------------------
    // EA Data Block / Secondary Block pointers follow the inline elements in
    // the index block.
    //
    // Each pointer is an 8-byte file address. We collect all non-UNDEF
    // direct data block (EADB) addresses and secondary block (EASB) addresses
    // first, then process each. EASB addresses are resolved into additional
    // EADB addresses via `parse_secondary_block`.
    //
    // Safety limit: never scan more than 1024 pointers.
    // -----------------------------------------------------------------------
    const MAX_DATA_BLOCK_PTRS: usize = 1024;
    let mut db_addresses: Vec<u64> = Vec::new();
    let mut secondary_block_addresses: Vec<u64> = Vec::new();
    {
        let mut ptr_pos = elem_area_end;
        for _ in 0..MAX_DATA_BLOCK_PTRS {
            // Need 8 bytes for the address field.
            if ptr_pos + 8 > file_data.len() {
                break;
            }
            let db_addr = u64::from_le_bytes(
                file_data[ptr_pos..ptr_pos + 8]
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("EA: data block pointer".into()))?,
            );
            ptr_pos += 8;

            // u64::MAX == UNDEF: end of the populated list.
            if db_addr == u64::MAX {
                break;
            }
            // Validate signature before accepting this address.
            let db = db_addr as usize;
            match file_data.get(db..db.saturating_add(4)) {
                Some(b"EADB") => db_addresses.push(db_addr),
                Some(b"EASB") => secondary_block_addresses.push(db_addr),
                _ => break, // Unknown signature — stop scanning.
            }
        }
    }

    // Resolve secondary blocks into direct data block addresses.
    for &sb_addr in &secondary_block_addresses {
        if let Ok(more_dbs) = parse_secondary_block(file_data, sb_addr) {
            db_addresses.extend(more_dbs);
        }
        // Silently skip failed secondary blocks — be defensive.
    }

    if db_addresses.is_empty() {
        return Ok(records);
    }

    // Build a sorted set of "known boundaries" to bound each data block's
    // element region.  Include the index block start and secondary block
    // addresses so we don't over-read into adjacent structures.
    let mut boundaries: Vec<usize> = db_addresses.iter().map(|&a| a as usize).collect();
    boundaries.push(ib); // index block start
    boundaries.push(file_data.len()); // file end
    for &sb in &secondary_block_addresses {
        boundaries.push(sb as usize);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    // -----------------------------------------------------------------------
    // EA Data Block ("EADB") layout:
    //  0  4   Signature "EADB"
    //  4  1   Version (must be 0)
    //  5  1   Client ID
    //  6  8   Header Address (back-pointer)
    // 14  M   Elements (* element_size bytes each)
    // …       Checksum (4 bytes)
    //
    // The block's element region ends at the start of the NEXT known
    // structure in the file, minus the 4-byte checksum.
    // -----------------------------------------------------------------------
    for db_addr in &db_addresses {
        let db = *db_addr as usize;

        if db + 14 > file_data.len() {
            continue; // Truncated block header.
        }

        let db_version = file_data[db + 4];
        if db_version != 0 {
            continue; // Unsupported version.
        }

        let db_elem_start = db + 14;
        if db_elem_start >= file_data.len() || element_size == 0 {
            continue;
        }

        // Find the next known boundary after db_elem_start.
        let next_boundary = boundaries
            .iter()
            .find(|&&b| b > db_elem_start)
            .copied()
            .unwrap_or(file_data.len());

        // Leave room for the 4-byte checksum at the end of the block.
        let block_data_end = next_boundary.saturating_sub(4);
        if block_data_end <= db_elem_start {
            continue;
        }

        let n_elems = (block_data_end - db_elem_start) / element_size;
        if n_elems == 0 {
            continue;
        }

        let db_elem_end = db_elem_start + n_elems * element_size;
        if db_elem_end > file_data.len() {
            continue;
        }

        let db_records =
            parse_elements(&file_data[db_elem_start..db_elem_end], element_size, ndims)?;
        records.extend(db_records);
    }

    Ok(records)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse an EA Secondary Block (EASB) and return the list of EADB addresses
/// it contains.
///
/// EASB layout:
/// ```text
///  0  4   Signature "EASB"
///  4  1   Version (must be 0)
///  5  1   Client ID
///  6  8   Header address (back-pointer to EAHD)
/// 14  8   Block offset (virtual heap offset — not used)
/// 22  …   Data block addresses (8 bytes each); terminated by u64::MAX or
///          invalid signature; capped at MAX_DB_PER_SECONDARY_BLOCK entries
/// …   4   Checksum (not verified here)
/// ```
fn parse_secondary_block(file_data: &[u8], sb_addr: u64) -> Result<Vec<u64>, OxiH5Error> {
    const MAX_DB_PER_SECONDARY_BLOCK: usize = 512;
    let base = sb_addr as usize;

    // Validate that at least the fixed 22-byte header is present.
    let header_end = base
        .checked_add(22)
        .ok_or_else(|| OxiH5Error::Format("EASB: address overflow".into()))?;
    if header_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EASB at {base:#x}: truncated (file len={})",
            file_data.len()
        )));
    }

    let sig = &file_data[base..base + 4];
    if sig != b"EASB" {
        return Err(OxiH5Error::Format(format!(
            "EASB: bad signature {sig:?} at {base:#x}"
        )));
    }

    let version = file_data[base + 4];
    if version != 0 {
        return Err(OxiH5Error::Format(format!(
            "EASB: unsupported version {version} at {base:#x}"
        )));
    }

    // Data block pointers start at byte 22:
    //   sig(4) + version(1) + client_id(1) + hdr_addr(8) + blk_offset(8) = 22
    let mut db_addrs = Vec::new();
    let mut pos = base + 22;
    for _ in 0..MAX_DB_PER_SECONDARY_BLOCK {
        if pos + 8 > file_data.len() {
            break;
        }
        let db_addr = u64::from_le_bytes(
            file_data[pos..pos + 8]
                .try_into()
                .map_err(|_| OxiH5Error::Format("EASB: db pointer read error".into()))?,
        );
        pos += 8;

        if db_addr == u64::MAX {
            break; // UNDEF = end of allocated pointers
        }

        // Validate that the target address actually starts an EADB.
        let db = db_addr as usize;
        match file_data.get(db..db.saturating_add(4)) {
            Some(b"EADB") => db_addrs.push(db_addr),
            _ => break, // Not a data block — stop scanning this secondary block.
        }
    }

    Ok(db_addrs)
}

/// Parse a flat byte slice into chunk records.
///
/// Each element is `element_size` bytes with layout:
/// ```text
/// address     : 8 bytes
/// chunk_size  : 4 bytes
/// filter_mask : 4 bytes
/// offsets     : (element_size - 16) bytes, split into `ndims` parts
/// ```
///
/// Elements whose address field equals `u64::MAX` are treated as empty slots
/// and skipped.
fn parse_elements(
    data: &[u8],
    element_size: usize,
    ndims: usize,
) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    if element_size < 16 {
        // Not enough bytes for the fixed header — cannot parse.
        return Ok(Vec::new());
    }

    let offset_bytes = element_size - 16;
    let bytes_per_dim = if ndims > 0 && offset_bytes > 0 {
        offset_bytes / ndims
    } else {
        0
    };

    let count = data.len() / element_size;
    let mut records = Vec::with_capacity(count);

    for i in 0..count {
        let e = i * element_size;
        let elem = &data[e..e + element_size];

        let address = u64::from_le_bytes(
            elem[0..8]
                .try_into()
                .map_err(|_| OxiH5Error::Format("EA element: address slice".into()))?,
        );

        if address == u64::MAX {
            continue; // Empty slot.
        }

        let size = u32::from_le_bytes(
            elem[8..12]
                .try_into()
                .map_err(|_| OxiH5Error::Format("EA element: size slice".into()))?,
        );
        let filter_mask = u32::from_le_bytes(
            elem[12..16]
                .try_into()
                .map_err(|_| OxiH5Error::Format("EA element: filter_mask slice".into()))?,
        );

        let offsets = parse_offsets(&elem[16..], ndims, bytes_per_dim)?;

        records.push(ChunkRecord {
            address,
            size,
            filter_mask,
            offsets,
        });
    }

    Ok(records)
}

/// Parse `ndims` offsets from `data`, each `bytes_per_dim` bytes wide (LE).
fn parse_offsets(data: &[u8], ndims: usize, bytes_per_dim: usize) -> Result<Vec<u64>, OxiH5Error> {
    if ndims == 0 || bytes_per_dim == 0 {
        return Ok(Vec::new());
    }
    let mut offs = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let o = d * bytes_per_dim;
        if o + bytes_per_dim > data.len() {
            return Err(OxiH5Error::Format(format!(
                "EA: offset field {d} out of bounds"
            )));
        }
        let val = match bytes_per_dim {
            8 => u64::from_le_bytes(
                data[o..o + 8]
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("EA: offset u64".into()))?,
            ),
            4 => u32::from_le_bytes(
                data[o..o + 4]
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("EA: offset u32".into()))?,
            ) as u64,
            2 => u16::from_le_bytes(
                data[o..o + 2]
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("EA: offset u16".into()))?,
            ) as u64,
            1 => data[o] as u64,
            other => {
                return Err(OxiH5Error::Format(format!(
                    "EA: unsupported bytes_per_dim {other}"
                )))
            }
        };
        offs.push(val);
    }
    Ok(offs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_ea_header(idx_blk_addr: u64, element_size: u8, idx_blk_elmts: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(b"EAHD");
        buf[4] = 0; // version
        buf[5] = 0; // client_id
        buf[6] = element_size;
        buf[7] = 8; // max_nelmts_bits
        buf[8] = idx_blk_elmts;
        buf[9] = 1; // data_blk_min_elmts
        buf[10] = 1; // secondary_blk_min_data_block_pointers
        buf[11] = 0; // max_dblk_page_nelmts_bits
                     // num_created_blks at 12 (8 bytes)
        buf[12..20].copy_from_slice(&1u64.to_le_bytes());
        // num_realized_blks at 20 (8 bytes)
        buf[20..28].copy_from_slice(&1u64.to_le_bytes());
        // index_block_address at 28 (8 bytes)
        buf[28..36].copy_from_slice(&idx_blk_addr.to_le_bytes());
        // checksum at 36 (4 bytes)
        buf
    }

    #[test]
    fn test_ea_no_index_block() {
        let buf = build_ea_header(u64::MAX, 24, 4);
        let result = parse_extensible_array(&buf, 0, 1).expect("parse failed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_ea_bad_signature() {
        let buf = vec![0u8; 64];
        let result = parse_extensible_array(&buf, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_ea_one_inline_element() {
        // element_size = 24 (1D: 8+4+4+8), idx_blk_elmts = 1
        let element_size: u8 = 24;
        let idx_blk_elmts: u8 = 1;

        // Index block will be placed at offset 64 in the buffer.
        let ib_addr: u64 = 64;

        let mut buf = vec![0u8; 256];
        // Header at 0.
        let hdr = build_ea_header(ib_addr, element_size, idx_blk_elmts);
        buf[..hdr.len()].copy_from_slice(&hdr);

        // Index block at 64: "EAIB" + version(1) + client_id(1) + hdr_addr(8) = 14 bytes header.
        let ib = ib_addr as usize;
        buf[ib..ib + 4].copy_from_slice(b"EAIB");
        buf[ib + 4] = 0; // version
        buf[ib + 5] = 0; // client_id
        buf[ib + 6..ib + 14].copy_from_slice(&0u64.to_le_bytes()); // back-pointer to header

        // Inline element at ib+14: address=0x2000, size=512, filter_mask=0, offset=0
        let e = ib + 14;
        buf[e..e + 8].copy_from_slice(&0x2000u64.to_le_bytes()); // address
        buf[e + 8..e + 12].copy_from_slice(&512u32.to_le_bytes()); // size
        buf[e + 12..e + 16].copy_from_slice(&0u32.to_le_bytes()); // filter_mask
        buf[e + 16..e + 24].copy_from_slice(&0u64.to_le_bytes()); // offset[0]

        let records = parse_extensible_array(&buf, 0, 1).expect("parse failed");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].address, 0x2000);
        assert_eq!(records[0].size, 512);
        assert_eq!(records[0].offsets, vec![0u64]);
    }

    /// Build a minimal EA with one inline element and one data block element,
    /// verifying that both are returned.
    #[test]
    fn test_ea_with_data_block() {
        // element_size = 24 (1D: 8 addr + 4 size + 4 filter_mask + 8 offset)
        let element_size: u8 = 24;
        let elem_sz = element_size as usize;
        let idx_blk_elmts: u8 = 1;

        // --- Build the EADB (data block) at offset 0 in our buffer ---
        // Layout: sig(4) + version(1) + client_id(1) + hdr_addr(8) = 14 header bytes
        //         + 1 element + 4 checksum = 14 + 24 + 4 = 42 bytes
        let eadb_offset: usize = 0;
        let eadb_len = 14 + elem_sz + 4;
        let mut buf = vec![0u8; 512];
        buf[eadb_offset..eadb_offset + 4].copy_from_slice(b"EADB");
        buf[eadb_offset + 4] = 0; // version
        buf[eadb_offset + 5] = 1; // client_id
                                  // header_addr back-pointer at [6..14] — filled in later
                                  // Element: addr=0xDEAD, size=200, filter_mask=0, offset[0]=8
        let de = eadb_offset + 14; // element start in data block
        buf[de..de + 8].copy_from_slice(&0xDEADu64.to_le_bytes()); // address
        buf[de + 8..de + 12].copy_from_slice(&200u32.to_le_bytes()); // size
        buf[de + 12..de + 16].copy_from_slice(&0u32.to_le_bytes()); // filter_mask
        buf[de + 16..de + 24].copy_from_slice(&8u64.to_le_bytes()); // offset[0] = 8

        // --- Build the EAIB (index block) immediately after EADB ---
        // Layout: sig(4) + version(1) + client_id(1) + hdr_addr(8) = 14 header bytes
        //         + 1 inline element (24 bytes)
        //         + 1 data-block pointer (8 bytes, points to eadb_offset=0)
        //         + 1 terminator (8 bytes = u64::MAX)
        //         + 4 checksum
        let ib_offset: usize = eadb_len;
        let ib_len = 14 + elem_sz + 8 + 8 + 4;
        buf[ib_offset..ib_offset + 4].copy_from_slice(b"EAIB");
        buf[ib_offset + 4] = 0; // version
        buf[ib_offset + 5] = 1; // client_id
                                // header_addr back-pointer at [6..14] — filled in later
                                // Inline element: addr=0xBEEF, size=100, filter_mask=0, offset[0]=0
        let ie = ib_offset + 14; // inline element start
        buf[ie..ie + 8].copy_from_slice(&0xBEEFu64.to_le_bytes()); // address
        buf[ie + 8..ie + 12].copy_from_slice(&100u32.to_le_bytes()); // size
        buf[ie + 12..ie + 16].copy_from_slice(&0u32.to_le_bytes()); // filter_mask
        buf[ie + 16..ie + 24].copy_from_slice(&0u64.to_le_bytes()); // offset[0] = 0
                                                                    // Data block pointer (points to EADB at offset 0)
        let ptr_pos = ie + elem_sz;
        buf[ptr_pos..ptr_pos + 8].copy_from_slice(&(eadb_offset as u64).to_le_bytes());
        // Terminator
        buf[ptr_pos + 8..ptr_pos + 16].copy_from_slice(&u64::MAX.to_le_bytes());

        // --- Build the EAHD (header) after the index block ---
        let hdr_offset: usize = ib_offset + ib_len;
        let hdr = build_ea_header(ib_offset as u64, element_size, idx_blk_elmts);
        buf[hdr_offset..hdr_offset + hdr.len()].copy_from_slice(&hdr);

        // Patch back-pointers: EADB and EAIB both point to the header.
        buf[eadb_offset + 6..eadb_offset + 14].copy_from_slice(&(hdr_offset as u64).to_le_bytes());
        buf[ib_offset + 6..ib_offset + 14].copy_from_slice(&(hdr_offset as u64).to_le_bytes());

        let records = parse_extensible_array(&buf, hdr_offset as u64, 1).expect("parse failed");

        // Expect exactly 2 records: 1 inline + 1 from the data block.
        assert_eq!(
            records.len(),
            2,
            "expected 2 chunk records, got {}",
            records.len()
        );

        // The inline element (0xBEEF) must be present.
        let inline = records.iter().find(|r| r.address == 0xBEEF);
        assert!(inline.is_some(), "inline element 0xBEEF not found");
        let inline = inline.unwrap();
        assert_eq!(inline.size, 100);
        assert_eq!(inline.offsets, vec![0u64]);

        // The data-block element (0xDEAD) must also be present.
        let db_rec = records.iter().find(|r| r.address == 0xDEAD);
        assert!(db_rec.is_some(), "data-block element 0xDEAD not found");
        let db_rec = db_rec.unwrap();
        assert_eq!(db_rec.size, 200);
        assert_eq!(db_rec.offsets, vec![8u64]);
    }

    /// Verifies that empty slots (u64::MAX addresses) inside a data block are
    /// correctly skipped and do not stop parsing of subsequent valid entries.
    #[test]
    fn test_ea_data_block_sparse() {
        let element_size: u8 = 24;
        let elem_sz = element_size as usize;
        let idx_blk_elmts: u8 = 0; // No inline elements

        // EADB with 2 elements: first is empty slot (u64::MAX), second is valid.
        let eadb_offset: usize = 0;
        let n_db_elems = 2usize;
        let eadb_len = 14 + n_db_elems * elem_sz + 4;
        let mut buf = vec![0u8; 512];

        buf[eadb_offset..eadb_offset + 4].copy_from_slice(b"EADB");
        buf[eadb_offset + 4] = 0; // version
        buf[eadb_offset + 5] = 0; // client_id
                                  // First element: empty slot
        let e0 = eadb_offset + 14;
        buf[e0..e0 + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        // Second element: valid
        let e1 = e0 + elem_sz;
        buf[e1..e1 + 8].copy_from_slice(&0xCAFEu64.to_le_bytes());
        buf[e1 + 8..e1 + 12].copy_from_slice(&42u32.to_le_bytes());
        buf[e1 + 12..e1 + 16].copy_from_slice(&0u32.to_le_bytes());
        buf[e1 + 16..e1 + 24].copy_from_slice(&16u64.to_le_bytes());

        // EAIB with 0 inline elements + 1 data-block pointer + terminator
        let ib_offset = eadb_len;
        // 0 inline elements, 1 data-block pointer (8), 1 terminator (8), checksum (4)
        let ib_len = 14 + 8 + 8 + 4;
        buf[ib_offset..ib_offset + 4].copy_from_slice(b"EAIB");
        buf[ib_offset + 4] = 0;
        buf[ib_offset + 5] = 0;
        let ptr_pos = ib_offset + 14;
        buf[ptr_pos..ptr_pos + 8].copy_from_slice(&(eadb_offset as u64).to_le_bytes());
        buf[ptr_pos + 8..ptr_pos + 16].copy_from_slice(&u64::MAX.to_le_bytes());

        // EAHD
        let hdr_offset = ib_offset + ib_len;
        let hdr = build_ea_header(ib_offset as u64, element_size, idx_blk_elmts);
        buf[hdr_offset..hdr_offset + hdr.len()].copy_from_slice(&hdr);
        buf[eadb_offset + 6..eadb_offset + 14].copy_from_slice(&(hdr_offset as u64).to_le_bytes());
        buf[ib_offset + 6..ib_offset + 14].copy_from_slice(&(hdr_offset as u64).to_le_bytes());

        let records = parse_extensible_array(&buf, hdr_offset as u64, 1).expect("parse failed");
        // Only the second element (0xCAFE) should appear; the empty slot is skipped.
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].address, 0xCAFE);
        assert_eq!(records[0].size, 42);
        assert_eq!(records[0].offsets, vec![16u64]);
    }

    #[test]
    fn test_ea_skips_empty_slots() {
        // Two inline elements, first has address = UNDEF (empty slot).
        let element_size: u8 = 24;
        let idx_blk_elmts: u8 = 2;
        let ib_addr: u64 = 64;

        let mut buf = vec![0u8; 256];
        let hdr = build_ea_header(ib_addr, element_size, idx_blk_elmts);
        buf[..hdr.len()].copy_from_slice(&hdr);

        let ib = ib_addr as usize;
        buf[ib..ib + 4].copy_from_slice(b"EAIB");
        buf[ib + 4] = 0;
        buf[ib + 5] = 0;
        buf[ib + 6..ib + 14].copy_from_slice(&0u64.to_le_bytes());

        // Element 0: UNDEF address = skip.
        let e0 = ib + 14;
        buf[e0..e0 + 8].copy_from_slice(&u64::MAX.to_le_bytes());

        // Element 1: valid.
        let e1 = e0 + 24;
        buf[e1..e1 + 8].copy_from_slice(&0x3000u64.to_le_bytes());
        buf[e1 + 8..e1 + 12].copy_from_slice(&128u32.to_le_bytes());
        buf[e1 + 12..e1 + 16].copy_from_slice(&0u32.to_le_bytes());
        buf[e1 + 16..e1 + 24].copy_from_slice(&8u64.to_le_bytes()); // offset = 8

        let records = parse_extensible_array(&buf, 0, 1).expect("parse failed");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].address, 0x3000);
        assert_eq!(records[0].offsets, vec![8u64]);
    }

    /// Build a minimal EA with one inline element and one secondary block
    /// (EASB) that contains one EADB, and verify both records are returned.
    #[test]
    fn test_ea_with_secondary_block() {
        // element_size = 24 (1D: 8 addr + 4 size + 4 filter_mask + 8 offset)
        let element_size: u8 = 24;
        let elem_sz = element_size as usize;
        let idx_blk_elmts: u8 = 1;

        let mut buf = vec![0u8; 1024];

        // ---- EADB at offset 0: contains 1 element ----
        let eadb_off = 0usize;
        buf[eadb_off..eadb_off + 4].copy_from_slice(b"EADB");
        buf[eadb_off + 4] = 0; // version
        buf[eadb_off + 5] = 0; // client_id
                               // hdr_addr at [6..14] — patched below
        let de = eadb_off + 14; // element start
        buf[de..de + 8].copy_from_slice(&0xABCDu64.to_le_bytes()); // address
        buf[de + 8..de + 12].copy_from_slice(&300u32.to_le_bytes()); // size
        buf[de + 12..de + 16].copy_from_slice(&0u32.to_le_bytes()); // filter_mask
        buf[de + 16..de + 24].copy_from_slice(&0u64.to_le_bytes()); // offset

        // ---- EASB at offset 50: points to EADB at offset 0 ----
        let easb_off = 50usize;
        buf[easb_off..easb_off + 4].copy_from_slice(b"EASB");
        buf[easb_off + 4] = 0; // version
        buf[easb_off + 5] = 0; // client_id
                               // hdr_addr at [6..14] — patched below
                               // blk_offset at [14..22] — leave as 0
        let easb_ptrs_off = easb_off + 22;
        buf[easb_ptrs_off..easb_ptrs_off + 8].copy_from_slice(&(eadb_off as u64).to_le_bytes()); // pointer to EADB
        buf[easb_ptrs_off + 8..easb_ptrs_off + 16].copy_from_slice(&u64::MAX.to_le_bytes()); // UNDEF terminator

        // ---- EAIB at offset 110: 1 inline element + EASB pointer + terminator ----
        let ib_off = 110usize;
        buf[ib_off..ib_off + 4].copy_from_slice(b"EAIB");
        buf[ib_off + 4] = 0; // version
        buf[ib_off + 5] = 0; // client_id
                             // hdr_addr at [6..14] — patched below
                             // Inline element
        let ie = ib_off + 14;
        buf[ie..ie + 8].copy_from_slice(&0x5678u64.to_le_bytes()); // address
        buf[ie + 8..ie + 12].copy_from_slice(&100u32.to_le_bytes()); // size
        buf[ie + 12..ie + 16].copy_from_slice(&0u32.to_le_bytes()); // filter_mask
        buf[ie + 16..ie + 24].copy_from_slice(&0u64.to_le_bytes()); // offset
                                                                    // EASB pointer (not a direct EADB pointer)
        let ptr_off = ie + elem_sz;
        buf[ptr_off..ptr_off + 8].copy_from_slice(&(easb_off as u64).to_le_bytes());
        // Terminator after the EASB pointer
        buf[ptr_off + 8..ptr_off + 16].copy_from_slice(&u64::MAX.to_le_bytes());

        // ---- EAHD at offset 200 ----
        let hdr_off = 200usize;
        let hdr = build_ea_header(ib_off as u64, element_size, idx_blk_elmts);
        buf[hdr_off..hdr_off + hdr.len()].copy_from_slice(&hdr);

        // Patch back-pointers so all structures point to the header.
        buf[eadb_off + 6..eadb_off + 14].copy_from_slice(&(hdr_off as u64).to_le_bytes());
        buf[easb_off + 6..easb_off + 14].copy_from_slice(&(hdr_off as u64).to_le_bytes());
        buf[ib_off + 6..ib_off + 14].copy_from_slice(&(hdr_off as u64).to_le_bytes());

        let records =
            parse_extensible_array(&buf, hdr_off as u64, 1).expect("parse_extensible_array failed");

        assert_eq!(
            records.len(),
            2,
            "expected 2 records (1 inline + 1 via EASB→EADB), got {}",
            records.len()
        );
        assert!(
            records.iter().any(|r| r.address == 0x5678),
            "inline record (0x5678) missing"
        );
        assert!(
            records.iter().any(|r| r.address == 0xABCD),
            "EASB→EADB record (0xABCD) missing"
        );
    }
}
