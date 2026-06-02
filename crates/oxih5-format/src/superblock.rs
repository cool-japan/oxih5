use oxih5_core::OxiH5Error;

/// HDF5 file signature.
const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Parsed HDF5 superblock v0 fields needed for navigation.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub size_of_offsets: u8,
    pub size_of_lengths: u8,
    pub base_address: u64,
    pub root_object_header_address: u64,
}

/// Parse a superblock from the beginning of the file bytes.
///
/// Dispatches to v0 or v2/v3 parsers based on the version byte at offset 8.
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
        size_of_offsets,
        size_of_lengths,
        base_address,
        root_object_header_address,
    })
}

/// Parse superblock version 2 or 3 (identical layout for parsing purposes).
fn parse_v2(data: &[u8], _version: u8) -> Result<Superblock, OxiH5Error> {
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
    // superblock_extension_address at 12+soo (skip it)
    // end_of_file_address at 12+2*soo (skip it)
    // root_group_object_header_address at 12+3*soo
    let root_object_header_address = read_offset_at(data, 12 + 3 * soo, soo)?;

    Ok(Superblock {
        size_of_offsets,
        size_of_lengths,
        base_address,
        root_object_header_address,
    })
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

    #[test]
    fn test_superblock_v0_parse() {
        let sb = build_v0();
        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.size_of_lengths, 8);
        assert_eq!(parsed.base_address, 0);
        assert_eq!(parsed.root_object_header_address, 96);
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
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.size_of_lengths, 8);
        assert_eq!(parsed.base_address, 0);
        assert_eq!(parsed.root_object_header_address, 48);
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
                      // root_obj_header at 12 + 3*8 = 36
        sb[36..44].copy_from_slice(&100_u64.to_le_bytes());

        let parsed = parse(&sb).unwrap();
        assert_eq!(parsed.size_of_offsets, 8);
        assert_eq!(parsed.root_object_header_address, 100);
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
    }

    #[test]
    fn test_superblock_unsupported_version() {
        let mut sb = vec![0u8; 9];
        sb[0..8].copy_from_slice(&HDF5_SIGNATURE);
        sb[8] = 1; // version 1 is transitional — not supported
        assert!(matches!(
            parse(&sb),
            Err(OxiH5Error::UnsupportedSuperblock(1))
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
}
