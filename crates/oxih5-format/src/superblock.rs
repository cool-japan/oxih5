use oxih5_core::OxiH5Error;

/// HDF5 file signature.
const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Parsed HDF5 superblock fields needed for navigation.
///
/// Supports superblock versions 0, 1, 2, and 3.  Version 1 differs from
/// version 0 only by four extra bytes ("Indexed Storage Internal Node K" +
/// "Reserved") inserted after the File Consistency Flags, so every field from
/// the Base Address onward is shifted by +4.  Versions 2 and 3 share an
/// identical on-disk layout for parsing purposes.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock version byte (0, 1, 2, or 3).
    pub version: u8,
    pub size_of_offsets: u8,
    pub size_of_lengths: u8,
    pub base_address: u64,
    pub root_object_header_address: u64,
    /// Address of the superblock extension object header, if present.
    ///
    /// `None` for versions 0 and 1 (they carry a Driver Info Block rather than
    /// a superblock extension) and `None` for versions 2/3 whose extension
    /// address field is the HDF5 "undefined address" sentinel (all-ones for the
    /// offset width).  `Some(addr)` gives the absolute byte offset of the
    /// extension object header, which can be parsed with
    /// [`crate::header::parse_messages`].
    pub superblock_extension_address: Option<u64>,
}

/// Parse a superblock from the beginning of the file bytes.
///
/// Dispatches to the v0, v1, or v2/v3 parsers based on the version byte at
/// offset 8.
///
/// **Version 0 layout** (all little-endian):
/// ```text
/// Offset  Size  Field
///  0       8     Signature: 89 48 44 46 0d 0a 1a 0a
///  8       1     Superblock version (0)
///  9       1     Free-space storage version
/// 10       1     Root group symbol table entry version
/// 11       1     Reserved
/// 12       1     Shared object header message version
/// 13       1     size_of_offsets (usually 8)
/// 14       1     size_of_lengths (usually 8)
/// 15       1     Reserved
/// 16       2     Leaf node K
/// 18       2     Internal node K
/// 20       4     Consistency flags
/// 24       8     Base address
/// 32       8     Free-space address (0xFFFF...=undefined)
/// 40       8     EOF address
/// 48       8     Driver information address (0xFFFF...=undefined)
/// 56       ?     Root Group Symbol Table Entry:
///                  link_name_offset (soo=8 bytes)
///                  object_header_address (soo=8 bytes)  <- at file offset 64
///                  cache_type (4)
///                  reserved (4)
///                  scratch (16)
/// ```
///
/// **Version 1 layout** — identical to version 0 except four extra bytes
/// ("Indexed Storage Internal Node K" u16 + "Reserved" u16) are inserted
/// immediately after the 4-byte File Consistency Flags (bytes 20..24) and
/// before the Base Address.  Every field from the Base Address onward shifts
/// by +4 (all little-endian):
/// ```text
/// Offset  Size  Field
///  0       8     Signature: 89 48 44 46 0d 0a 1a 0a
///  8       1     Superblock version (1)
///  9       1     Free-space storage version
/// 10       1     Root group symbol table entry version
/// 11       1     Reserved
/// 12       1     Shared object header message version
/// 13       1     size_of_offsets (usually 8)  <- SAME position as v0
/// 14       1     size_of_lengths (usually 8)  <- SAME position as v0
/// 15       1     Reserved
/// 16       2     Leaf node K
/// 18       2     Internal node K
/// 20       4     Consistency flags
/// 24       2     Indexed Storage Internal Node K  <- v1-only
/// 26       2     Reserved                         <- v1-only
/// 28       8     Base address
/// 36       8     Free-space address (0xFFFF...=undefined)
/// 44       8     EOF address
/// 52       8     Driver information address (0xFFFF...=undefined)
/// 60       ?     Root Group Symbol Table Entry:
///                  link_name_offset (soo=8 bytes)
///                  object_header_address (soo=8 bytes)  <- at file offset 68
///                  cache_type (4)
///                  reserved (4)
///                  scratch (16)
/// ```
///
/// **Version 2/3 layout** (v3 identical to v2 for parsing purposes):
/// ```text
/// Offset     Size  Field
///  0          8     Signature
///  8          1     Superblock version (2 or 3)
///  9          1     size_of_offsets (soo)
/// 10          1     size_of_lengths (sol)
/// 11          1     file_consistency_flags
/// 12          soo   base_address
/// 12+soo      soo   superblock_extension_address
/// 12+2*soo    soo   end_of_file_address
/// 12+3*soo    soo   root_group_object_header_address
/// 12+4*soo    4     Fletcher-32 checksum
/// ```
pub fn parse(data: &[u8]) -> Result<Superblock, OxiH5Error> {
    // Need at least the signature + version byte.
    if data.len() < 9 {
        return Err(OxiH5Error::Format(format!(
            "file too short for superblock: {} bytes",
            data.len()
        )));
    }

    // Verify signature.
    if data[0..8] != HDF5_SIGNATURE {
        return Err(OxiH5Error::BadSignature);
    }

    let version = data[8];
    match version {
        0 => parse_v0(data),
        1 => parse_v1(data),
        2 | 3 => parse_v2(data, version),
        other => Err(OxiH5Error::UnsupportedSuperblock(other)),
    }
}

/// Parse superblock version 0.
fn parse_v0(data: &[u8]) -> Result<Superblock, OxiH5Error> {
    if data.len() < 96 {
        return Err(OxiH5Error::Format(format!(
            "file too short for superblock v0: {} bytes",
            data.len()
        )));
    }

    let size_of_offsets = data[13];
    let size_of_lengths = data[14];

    // Only standard 8-byte offsets/lengths are supported.
    if size_of_offsets != 8 || size_of_lengths != 8 {
        return Err(OxiH5Error::Format(format!(
            "superblock v0: requires soo=8/sol=8, got soo={size_of_offsets}/sol={size_of_lengths}"
        )));
    }

    let base_address = read_u64_le(data, 24)?;

    // Root group symbol table entry starts at offset 56.
    // STE layout: link_name_offset(8) + object_header_address(8) + ...
    // object_header_address is at file offset 56 + 8 = 64.
    let root_object_header_address = read_u64_le(data, 64)?;

    Ok(Superblock {
        version: 0,
        size_of_offsets,
        size_of_lengths,
        base_address,
        root_object_header_address,
        // v0 has a Driver Info Block, not a superblock extension.
        superblock_extension_address: None,
    })
}

/// Parse superblock version 1.
///
/// Version 1 is identical to version 0 except for four extra bytes
/// ("Indexed Storage Internal Node K" u16 + "Reserved" u16) inserted after the
/// File Consistency Flags (bytes 20..24).  Consequently `size_of_offsets` and
/// `size_of_lengths` remain at bytes 13/14, but the Base Address moves to
/// offset 28 and the root group's object header address moves to offset 68.
fn parse_v1(data: &[u8]) -> Result<Superblock, OxiH5Error> {
    if data.len() < 100 {
        return Err(OxiH5Error::Format(format!(
            "file too short for superblock v1: {} bytes",
            data.len()
        )));
    }

    let size_of_offsets = data[13];
    let size_of_lengths = data[14];

    // Only standard 8-byte offsets/lengths are supported.
    if size_of_offsets != 8 || size_of_lengths != 8 {
        return Err(OxiH5Error::Format(format!(
            "superblock v1: requires soo=8/sol=8, got soo={size_of_offsets}/sol={size_of_lengths}"
        )));
    }

    // Base address is shifted +4 relative to v0 (24 -> 28).
    let base_address = read_u64_le(data, 28)?;

    // Root group symbol table entry starts at offset 60 (v0 had 56).
    // object_header_address is at file offset 60 + link_name_offset(8) = 68.
    let root_object_header_address = read_u64_le(data, 68)?;

    Ok(Superblock {
        version: 1,
        size_of_offsets,
        size_of_lengths,
        base_address,
        root_object_header_address,
        // v1 has a Driver Info Block, not a superblock extension.
        superblock_extension_address: None,
    })
}

/// Parse superblock version 2 or 3 (identical layout for parsing purposes).
fn parse_v2(data: &[u8], version: u8) -> Result<Superblock, OxiH5Error> {
    // Minimum size: sig(8) + version(1) + soo(1) + sol(1) + flags(1) + 4*soo + checksum(4)
    // With soo=1 the minimum is 8+4+4*1+4=20 bytes; with soo=8 it is 8+4+32+4=48 bytes.
    // We check bounds lazily when reading individual fields.

    if data.len() < 13 {
        return Err(OxiH5Error::Format(format!(
            "file too short for superblock v2/v3: {} bytes",
            data.len()
        )));
    }

    let size_of_offsets = data[9];
    let size_of_lengths = data[10];
    let soo = size_of_offsets as usize;

    // base_address at offset 12
    let base_address = read_offset_at(data, 12, soo)?;
    // superblock_extension_address at 12+soo.  A value equal to the all-ones
    // "undefined address" sentinel for this offset width means "no extension".
    let raw_ext = read_offset_at(data, 12 + soo, soo)?;
    let superblock_extension_address = if raw_ext == undefined_address(soo) {
        None
    } else {
        Some(raw_ext)
    };
    // end_of_file_address at 12+2*soo (skip it)
    // root_group_object_header_address at 12+3*soo
    let root_object_header_address = read_offset_at(data, 12 + 3 * soo, soo)?;

    Ok(Superblock {
        version,
        size_of_offsets,
        size_of_lengths,
        base_address,
        root_object_header_address,
        superblock_extension_address,
    })
}

/// The HDF5 "undefined address" sentinel for an offset field of width `soo`
/// bytes: all bits set (e.g. `0xFF` for soo=1, `u64::MAX` for soo=8).
fn undefined_address(soo: usize) -> u64 {
    if soo >= 8 {
        u64::MAX
    } else {
        (1u64 << (soo * 8)) - 1
    }
}

/// Read a variable-width unsigned integer (1/2/4/8 bytes, little-endian) from `data` at `pos`.
fn read_offset_at(data: &[u8], pos: usize, size: usize) -> Result<u64, OxiH5Error> {
    let bytes = data.get(pos..pos + size).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "superblock: offset read at pos={pos} size={size} out of bounds (data len={})",
            data.len()
        ))
    })?;
    let value = match size {
        1 => bytes[0] as u64,
        2 => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
        4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        8 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                OxiH5Error::Format("superblock: failed to convert 8 bytes to u64".into())
            })?;
            u64::from_le_bytes(arr)
        }
        _ => {
            return Err(OxiH5Error::Format(format!(
                "superblock: unsupported offset size {size}"
            )))
        }
    };
    Ok(value)
}

// ---------------------------------------------------------------------------
// Shared byte-reading helpers exported to other format modules.
// ---------------------------------------------------------------------------

/// Read a little-endian u64 from `data` at `offset`.
pub fn read_u64_le(data: &[u8], offset: usize) -> Result<u64, OxiH5Error> {
    if offset + 8 > data.len() {
        return Err(OxiH5Error::Format(format!(
            "read_u64_le: offset {offset} out of bounds (len={})",
            data.len()
        )));
    }
    let arr: [u8; 8] = data[offset..offset + 8]
        .try_into()
        .map_err(|_| OxiH5Error::Format("u64 slice conversion failed".to_string()))?;
    Ok(u64::from_le_bytes(arr))
}

/// Read a little-endian u32 from `data` at `offset`.
pub fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, OxiH5Error> {
    if offset + 4 > data.len() {
        return Err(OxiH5Error::Format(format!(
            "read_u32_le: offset {offset} out of bounds (len={})",
            data.len()
        )));
    }
    let arr: [u8; 4] = data[offset..offset + 4]
        .try_into()
        .map_err(|_| OxiH5Error::Format("u32 slice conversion failed".to_string()))?;
    Ok(u32::from_le_bytes(arr))
}

/// Read a little-endian u16 from `data` at `offset`.
pub fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, OxiH5Error> {
    if offset + 2 > data.len() {
        return Err(OxiH5Error::Format(format!(
            "read_u16_le: offset {offset} out of bounds (len={})",
            data.len()
        )));
    }
    let arr: [u8; 2] = data[offset..offset + 2]
        .try_into()
        .map_err(|_| OxiH5Error::Format("u16 slice conversion failed".to_string()))?;
    Ok(u16::from_le_bytes(arr))
}

// ---------------------------------------------------------------------------
// Superblock extension (versions 2/3 only)
// ---------------------------------------------------------------------------

/// Decoded "B-tree 'K' Values" message (type `0x0013`) from a superblock
/// extension.
///
/// These K values control the fan-out of the version-1 B-trees used for
/// chunked-storage indexing and old-style group symbol tables.  Files that
/// omit this message use the HDF5 specification defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BtreeKValues {
    /// Internal-node K for the v1 B-tree indexing chunked dataset storage.
    pub indexed_storage_internal_k: u16,
    /// Internal-node K for old-style group (symbol table) B-trees.
    pub group_internal_k: u16,
    /// Leaf-node K for old-style group (symbol table) B-trees.
    pub group_leaf_k: u16,
}

/// Interpretable messages captured from a file's superblock extension object
/// header, returned by [`read_superblock_extension`].
///
/// The superblock extension may carry any subset of these messages, so every
/// field is optional.  Messages that are not yet interpreted are ignored.
#[derive(Debug, Clone, Default)]
pub struct SuperblockExtension {
    /// Decoded "B-tree 'K' Values" message (type `0x0013`), if present.
    pub btree_k: Option<BtreeKValues>,
    /// "Shared Message Table" (type `0x000F`) table address, if present.
    pub shared_message_table_address: Option<u64>,
    /// `true` if a "File Space Info" message (type `0x0018`) is present.
    pub file_space_info_present: bool,
    /// The "File Space Info" strategy byte, when the message is present and
    /// long enough to contain it (byte index 1 in both message versions).
    pub file_space_strategy: Option<u8>,
    /// `true` if a "Driver Info" message (type `0x0014`) is present.
    pub driver_info_present: bool,
}

/// Parse a file's superblock extension, if it has one, and return the messages
/// oxih5 can currently interpret.
///
/// Returns `Ok(None)` when the superblock carries no extension (versions 0/1,
/// or a v2/v3 extension address equal to the undefined sentinel).  Otherwise
/// the extension object header is parsed via [`crate::header::parse_messages`]
/// and its B-tree 'K' Values (0x0013), Shared Message Table (0x000F), File
/// Space Info (0x0018) and Driver Info (0x0014) messages are decoded.
///
/// A malformed message body (too short to decode, or an unsupported message
/// version) yields a typed error rather than being silently ignored.
pub fn read_superblock_extension(data: &[u8]) -> Result<Option<SuperblockExtension>, OxiH5Error> {
    let sb = parse(data)?;
    let ext_addr = match sb.superblock_extension_address {
        Some(addr) => addr,
        None => return Ok(None),
    };

    let messages = crate::header::parse_messages(data, ext_addr)?;
    let mut ext = SuperblockExtension::default();
    for msg in &messages {
        match msg.msg_type {
            // B-tree 'K' Values.
            0x0013 => ext.btree_k = Some(decode_btree_k_values(&msg.data)?),
            // Shared Message Table.
            0x000F => {
                ext.shared_message_table_address =
                    Some(decode_shared_message_table(&msg.data, sb.size_of_offsets)?);
            }
            // File Space Info: capture presence and (if available) the strategy byte.
            0x0018 => {
                ext.file_space_info_present = true;
                // Both message versions place the strategy at byte index 1.
                ext.file_space_strategy = msg.data.get(1).copied();
            }
            // Driver Info.
            0x0014 => ext.driver_info_present = true,
            _ => {}
        }
    }
    Ok(Some(ext))
}

/// Decode a "B-tree 'K' Values" message body (type `0x0013`).
///
/// Field order per the HDF5 File Format Specification:
/// `version(1) + indexed_storage_internal_node_k(u16 LE)
///  + group_internal_node_k(u16 LE) + group_leaf_node_k(u16 LE)` — 7 bytes.
///
/// Returns [`OxiH5Error::Format`] if the body is too short and
/// [`OxiH5Error::NotImplemented`] for an unrecognised message version.
fn decode_btree_k_values(data: &[u8]) -> Result<BtreeKValues, OxiH5Error> {
    if data.len() < 7 {
        return Err(OxiH5Error::Format(format!(
            "superblock extension: B-tree 'K' Values message too short: {} bytes (need 7)",
            data.len()
        )));
    }
    let version = data[0];
    if version != 0 {
        return Err(OxiH5Error::NotImplemented(format!(
            "superblock extension: B-tree 'K' Values message version {version} not supported"
        )));
    }
    Ok(BtreeKValues {
        indexed_storage_internal_k: read_u16_le(data, 1)?,
        group_internal_k: read_u16_le(data, 3)?,
        group_leaf_k: read_u16_le(data, 5)?,
    })
}

/// Decode a "Shared Message Table" message body (type `0x000F`), returning the
/// Shared Object Header Message Table address.
///
/// Layout: `version(1) + shared_object_header_message_table_address(soo)
///  + num_indexes(1)`, where `soo` is the file's `size_of_offsets`.
fn decode_shared_message_table(data: &[u8], size_of_offsets: u8) -> Result<u64, OxiH5Error> {
    let soo = size_of_offsets as usize;
    let need = 1 + soo + 1;
    if data.len() < need {
        return Err(OxiH5Error::Format(format!(
            "superblock extension: Shared Message Table message too short: {} bytes (need {need})",
            data.len()
        )));
    }
    let version = data[0];
    if version != 0 {
        return Err(OxiH5Error::NotImplemented(format!(
            "superblock extension: Shared Message Table message version {version} not supported"
        )));
    }
    read_offset_at(data, 1, soo)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal superblock v0 with soo=8, sol=8.
    fn build_v0() -> Vec<u8> {
        let mut sb = vec![0u8; 96];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 0; // version 0
        sb[13] = 8; // size_of_offsets
        sb[14] = 8; // size_of_lengths
                    // base_address at 24: 0
                    // root_object_header_address at 64: 96
        sb[64..72].copy_from_slice(&96_u64.to_le_bytes());
        sb
    }

    /// Build a minimal superblock v1 with soo=8, sol=8.
    ///
    /// v1 mirrors v0 but inserts 4 bytes after the consistency flags, so the
    /// buffer is 100 bytes and the root object header address lives at offset 68.
    fn build_v1() -> Vec<u8> {
        let mut sb = vec![0u8; 100];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 1; // version 1
        sb[13] = 8; // size_of_offsets (same position as v0)
        sb[14] = 8; // size_of_lengths (same position as v0)
                    // base_address at 28: 0
                    // root_object_header_address at 68: 100
        sb[68..76].copy_from_slice(&100_u64.to_le_bytes());
        sb
    }

    #[test]
    fn test_superblock_v0_parse() {
        let sb = build_v0();
        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.size_of_lengths, 8);
        assert_eq!(parsed.base_address, 0);
        assert_eq!(parsed.root_object_header_address, 96);
        // v0 has a Driver Info Block, not a superblock extension.
        assert_eq!(parsed.superblock_extension_address, None);
    }

    #[test]
    fn test_superblock_v1_parse() {
        let sb = build_v1();
        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.size_of_lengths, 8);
        assert_eq!(parsed.base_address, 0);
        assert_eq!(parsed.root_object_header_address, 100);
        // v1 has a Driver Info Block, not a superblock extension.
        assert_eq!(parsed.superblock_extension_address, None);
    }

    #[test]
    fn test_superblock_v1_nonzero_base_and_root() {
        let mut sb = build_v1();
        // base_address at offset 28.
        sb[28..36].copy_from_slice(&2048_u64.to_le_bytes());
        // root object header address at offset 68.
        sb[68..76].copy_from_slice(&512_u64.to_le_bytes());
        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.base_address, 2048);
        assert_eq!(parsed.root_object_header_address, 512);
    }

    #[test]
    fn test_superblock_v1_too_short() {
        // 96 bytes is enough for v0 but too short for v1 (needs 100).
        let mut sb = vec![0u8; 96];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 1;
        sb[13] = 8;
        sb[14] = 8;
        assert!(parse(&sb).is_err());
    }

    #[test]
    fn test_superblock_v1_bad_offset_width() {
        let mut sb = build_v1();
        sb[13] = 4; // unsupported soo
        assert!(matches!(parse(&sb), Err(OxiH5Error::Format(_))));
    }

    #[test]
    fn test_superblock_v0_bad_signature() {
        let mut sb = build_v0();
        sb[0] = 0xFF; // corrupt signature
        assert!(matches!(parse(&sb), Err(OxiH5Error::BadSignature)));
    }

    #[test]
    fn test_superblock_v0_too_short() {
        let sb = vec![0u8; 8];
        assert!(parse(&sb).is_err());
    }

    #[test]
    fn test_superblock_v2_parse() {
        let soo: u8 = 8;
        let sol: u8 = 8;

        // Layout: sig(8) + version(1) + soo(1) + sol(1) + flags(1) + base(8)
        //         + ext(8) + eof(8) + root(8) + checksum(4)
        let total = 12 + 4 * soo as usize + 4;
        let mut sb = vec![0u8; total];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 2; // version
        sb[9] = soo;
        sb[10] = sol;
        sb[11] = 0; // file_consistency_flags

        // base_address at 12: 0 (all zeros already)
        // superblock_extension at 20: u64::MAX (undefined)
        sb[20..28].copy_from_slice(&u64::MAX.to_le_bytes());
        // eof at 28
        sb[28..36].copy_from_slice(&1024_u64.to_le_bytes());
        // root_obj_header at 36
        sb[36..44].copy_from_slice(&48_u64.to_le_bytes());
        // checksum at 44 (ignored by parser)

        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.size_of_lengths, 8);
        assert_eq!(parsed.base_address, 0);
        assert_eq!(parsed.root_object_header_address, 48);
        // ext field was set to the u64::MAX sentinel → treated as absent.
        assert_eq!(parsed.superblock_extension_address, None);
    }

    #[test]
    fn test_superblock_v2_with_extension() {
        let soo: u8 = 8;
        let sol: u8 = 8;
        let total = 12 + 4 * soo as usize + 4;
        let mut sb = vec![0u8; total];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 2; // version
        sb[9] = soo;
        sb[10] = sol;
        sb[11] = 0; // file_consistency_flags

        // base_address at 12: 0 (all zeros already)
        // superblock_extension at 20: a real address (not the sentinel).
        sb[20..28].copy_from_slice(&4096_u64.to_le_bytes());
        // eof at 28
        sb[28..36].copy_from_slice(&8192_u64.to_le_bytes());
        // root_obj_header at 36
        sb[36..44].copy_from_slice(&48_u64.to_le_bytes());

        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.root_object_header_address, 48);
        assert_eq!(parsed.superblock_extension_address, Some(4096));
    }

    #[test]
    fn test_superblock_v3_parse() {
        // v3 uses identical layout to v2; only version byte differs.
        let soo: u8 = 8;
        let total = 12 + 4 * soo as usize + 4;
        let mut sb = vec![0u8; total];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 3; // version 3
        sb[9] = soo;
        sb[10] = soo; // sol same as soo
                      // superblock_extension at 20: u64::MAX (undefined)
        sb[20..28].copy_from_slice(&u64::MAX.to_le_bytes());
        // root_obj_header at 12 + 3*8 = 36
        sb[36..44].copy_from_slice(&100_u64.to_le_bytes());

        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.root_object_header_address, 100);
        assert_eq!(parsed.superblock_extension_address, None);
    }

    #[test]
    fn test_superblock_v2_soo4() {
        // soo=4 (32-bit offset file)
        let soo: u8 = 4;
        let total = 12 + 4 * soo as usize + 4;
        let mut sb = vec![0u8; total];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 2; // version 2
        sb[9] = soo;
        sb[10] = soo; // sol
                      // base_address at 12: 0 (4 bytes)
                      // ext at 16: 0xFFFFFFFF
        sb[16..20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // eof at 20
        sb[20..24].copy_from_slice(&512_u32.to_le_bytes());
        // root at 24
        sb[24..28].copy_from_slice(&64_u32.to_le_bytes());

        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.size_of_offsets, 4);
        assert_eq!(parsed.root_object_header_address, 64);
        // ext at offset 16 was the soo=4 all-ones sentinel → absent.
        assert_eq!(parsed.superblock_extension_address, None);
    }

    #[test]
    fn test_superblock_v2_soo4_with_extension() {
        // soo=4 with a real (non-sentinel) extension address.
        let soo: u8 = 4;
        let total = 12 + 4 * soo as usize + 4;
        let mut sb = vec![0u8; total];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 2;
        sb[9] = soo;
        sb[10] = soo;
        // ext at 16: real 32-bit address.
        sb[16..20].copy_from_slice(&0x0001_0000u32.to_le_bytes());
        // root at 24
        sb[24..28].copy_from_slice(&64_u32.to_le_bytes());

        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.superblock_extension_address, Some(0x0001_0000));
    }

    #[test]
    fn test_superblock_unsupported_version() {
        let mut sb = vec![0u8; 9];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 4; // version 4 is genuinely unsupported
        assert!(matches!(
            parse(&sb),
            Err(OxiH5Error::UnsupportedSuperblock(4))
        ));
    }

    #[test]
    fn test_superblock_v2_truncated() {
        // Too short to read root header address
        let mut sb = vec![0u8; 15]; // only has sig + version + soo + sol + flags + part of base
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 2;
        sb[9] = 8; // soo=8 but not enough bytes
        sb[10] = 8;
        assert!(parse(&sb).is_err());
    }

    // -----------------------------------------------------------------------
    // Superblock extension message decoders
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_btree_k_values_ok() {
        // version(1)=0 + indexed_storage(2) + group_internal(2) + group_leaf(2)
        let mut body = vec![0u8; 7];
        body[0] = 0; // version
        body[1..3].copy_from_slice(&32u16.to_le_bytes()); // indexed_storage_internal_k
        body[3..5].copy_from_slice(&16u16.to_le_bytes()); // group_internal_k
        body[5..7].copy_from_slice(&4u16.to_le_bytes()); // group_leaf_k
        let k = decode_btree_k_values(&body).unwrap();
        assert_eq!(k.indexed_storage_internal_k, 32);
        assert_eq!(k.group_internal_k, 16);
        assert_eq!(k.group_leaf_k, 4);
    }

    #[test]
    fn test_decode_btree_k_values_too_short() {
        let body = vec![0u8; 6]; // one byte short
        assert!(matches!(
            decode_btree_k_values(&body),
            Err(OxiH5Error::Format(_))
        ));
    }

    #[test]
    fn test_decode_btree_k_values_bad_version() {
        let mut body = vec![0u8; 7];
        body[0] = 9; // unsupported version
        assert!(matches!(
            decode_btree_k_values(&body),
            Err(OxiH5Error::NotImplemented(_))
        ));
    }

    #[test]
    fn test_decode_shared_message_table_ok() {
        // version(1)=0 + table_address(soo=8) + num_indexes(1)
        let soo: u8 = 8;
        let mut body = vec![0u8; 1 + soo as usize + 1];
        body[0] = 0; // version
        body[1..9].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
        body[9] = 3; // num_indexes
        let addr = decode_shared_message_table(&body, soo).unwrap();
        assert_eq!(addr, 0xDEAD_BEEF);
    }

    #[test]
    fn test_decode_shared_message_table_soo4() {
        // 32-bit-offset file: version(1) + table_address(4) + num_indexes(1)
        let soo: u8 = 4;
        let mut body = vec![0u8; 1 + soo as usize + 1];
        body[0] = 0;
        body[1..5].copy_from_slice(&0x0001_2345u32.to_le_bytes());
        body[5] = 1;
        let addr = decode_shared_message_table(&body, soo).unwrap();
        assert_eq!(addr, 0x0001_2345);
    }

    #[test]
    fn test_decode_shared_message_table_too_short() {
        let body = vec![0u8; 5]; // too short for soo=8
        assert!(matches!(
            decode_shared_message_table(&body, 8),
            Err(OxiH5Error::Format(_))
        ));
    }

    #[test]
    fn test_decode_shared_message_table_bad_version() {
        let mut body = vec![0u8; 10];
        body[0] = 7; // unsupported version
        assert!(matches!(
            decode_shared_message_table(&body, 8),
            Err(OxiH5Error::NotImplemented(_))
        ));
    }
}
