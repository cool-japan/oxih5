use crate::context::ParseContext;
use oxih5_core::{Link, OxiH5Error};

// ---------------------------------------------------------------------------
// Parsed Link Info (message type 0x0002)
// ---------------------------------------------------------------------------

/// Parsed contents of a Link Info message (HDF5 message type 0x0002).
///
/// A Link Info message is present in new-style groups and carries the
/// addresses of the fractal heap (which stores link names and values) and
/// the B-tree v2 name/creation-order indices.
#[derive(Debug, Clone)]
pub struct ParsedLinkInfo {
    /// True when creation order for links is tracked.
    pub creation_order_tracked: bool,
    /// Address of the B-tree v2 index by creation order (None = absent / u64::MAX).
    pub creation_order_index_address: Option<u64>,
    /// Address of the B-tree v2 index by name (None = absent / u64::MAX).
    pub name_index_address: Option<u64>,
    /// Address of the fractal heap that stores link names / data (None = absent / u64::MAX).
    pub fractal_heap_address: Option<u64>,
}

/// Parse a Link Info message body (HDF5 message type 0x0002).
///
/// Layout (all little-endian, soo = `ctx.size_of_offsets`):
/// ```text
///  0   1   version = 0
///  1   1   flags
///            bit 0: creation-order tracked
///            bit 1: creation-order indexed
/// [2   8]  creation-order value (only if bit 0 set)
/// [*   soo] B-tree v2 by creation order (only if bit 1 set)
///  *   soo  B-tree v2 by name (u64::MAX = absent)
///  *   soo  fractal heap address (u64::MAX = absent)
/// ```
pub fn parse_link_info(body: &[u8], ctx: &ParseContext) -> Result<ParsedLinkInfo, OxiH5Error> {
    if body.len() < 2 {
        return Err(OxiH5Error::Format(
            "link info message too short".to_string(),
        ));
    }

    let _version = body[0];
    let flags = body[1];
    let creation_order_tracked = (flags & 0b01) != 0;
    let creation_order_indexed = (flags & 0b10) != 0;

    let mut pos = 2usize;

    // Optional 8-byte creation order value (present if bit 0 is set).
    if creation_order_tracked {
        pos += 8;
    }

    // Optional B-tree v2 address indexed by creation order (present if bit 1 is set).
    let creation_order_index_address = if creation_order_indexed {
        let addr = ctx.read_offset(body, pos)?;
        pos += ctx.size_of_offsets as usize;
        if addr == u64::MAX {
            None
        } else {
            Some(addr)
        }
    } else {
        None
    };

    // Fractal heap address is always present (comes before name index per HDF5 spec §IV.A.2.a).
    let heap_raw = ctx.read_offset(body, pos)?;
    pos += ctx.size_of_offsets as usize;
    let fractal_heap_address = if heap_raw == u64::MAX {
        None
    } else {
        Some(heap_raw)
    };

    // B-tree v2 index by name is always present (follows the fractal heap address).
    let name_index_raw = ctx.read_offset(body, pos)?;
    let name_index_address = if name_index_raw == u64::MAX {
        None
    } else {
        Some(name_index_raw)
    };

    Ok(ParsedLinkInfo {
        creation_order_tracked,
        creation_order_index_address,
        name_index_address,
        fractal_heap_address,
    })
}

// ---------------------------------------------------------------------------
// Parsed Link (message type 0x0006)
// ---------------------------------------------------------------------------

/// A parsed link — name plus link target.
#[derive(Debug, Clone)]
pub struct ParsedLink {
    /// UTF-8 link name.
    pub name: String,
    /// Link target.
    pub link: Link,
}

/// Bits 0-1 of the Link message flags encode the width of the link-name
/// length field.  Returns the number of bytes: 1, 2, 4, or 8.
fn name_len_size(flags: u8) -> usize {
    1 << (flags & 0b11)
}

/// Parse a Link message body (HDF5 message type 0x0006).
///
/// Layout (all little-endian, soo = `ctx.size_of_offsets`):
/// ```text
///  0   1   version = 1
///  1   1   flags
///            bits 0-1: name_length_size (0→1B, 1→2B, 2→4B, 3→8B)
///            bit 2:    creation-order field present (8 bytes)
///            bit 3:    link-type byte present (if 0 → hard link type 0)
///            bit 4:    charset byte present (1 byte)
/// [2   1]  link type (only if bit 3 set): 0=hard, 1=soft, 64=external, …
/// [*   8]  creation order (only if bit 2 set)
/// [*   1]  charset (only if bit 4 set)
///  *   1/2/4/8  link name length
///  *   N   link name (UTF-8, no NUL terminator)
///          --- link value ---
///  hard:   object_header_address (soo bytes)
///  soft:   target_path_length (u16) + target_path (UTF-8)
///  ext:    value_length (u16) + skip 1 byte + filename_cstr + NUL + path_cstr + NUL
/// ```
///
/// Per HDF5 file format specification §IV.A.2.c:
///   bit 3 = Link Type field is present in the message
///   bit 4 = Character Set field is present in the message
pub fn parse_link(body: &[u8], ctx: &ParseContext) -> Result<ParsedLink, OxiH5Error> {
    if body.len() < 2 {
        return Err(OxiH5Error::Format("link message too short".to_string()));
    }

    let _version = body[0];
    let flags = body[1];
    let mut pos = 2usize;

    // Link type — bit 3 signals presence per HDF5 spec §IV.A.2.c.
    // Defaults to hard link (0) when bit 3 is clear.
    let link_type = if (flags & 0b0000_1000) != 0 {
        let lt = *body.get(pos).ok_or_else(|| {
            OxiH5Error::Format("link message: missing link type byte".to_string())
        })?;
        pos += 1;
        lt
    } else {
        0u8
    };

    // Optional creation order (8 bytes, bit 2).
    if (flags & 0b0000_0100) != 0 {
        pos += 8;
    }

    // Optional charset (1 byte, bit 4 per HDF5 spec §IV.A.2.c).
    if (flags & 0b0001_0000) != 0 {
        pos += 1;
    }

    // Link name length field (1/2/4/8 bytes depending on flags bits 0-1).
    let nls = name_len_size(flags);
    let name_len = ctx.read_int_generic(body, pos, nls)?;
    pos += nls;

    // Link name bytes.
    let name_end = pos.checked_add(name_len as usize).ok_or_else(|| {
        OxiH5Error::Corrupted(format!(
            "link message: name length {name_len} causes overflow at pos {pos}"
        ))
    })?;
    let name_bytes = body.get(pos..name_end).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "link message: name of {} bytes at {} overflows body ({})",
            name_len,
            pos,
            body.len()
        ))
    })?;
    let name = String::from_utf8(name_bytes.to_vec())
        .map_err(|e| OxiH5Error::Format(format!("link name not valid UTF-8: {e}")))?;
    pos = name_end;

    // Parse link value.
    let link = match link_type {
        0 => {
            // Hard link: object header address.
            let address = ctx.read_offset(body, pos)?;
            Link::Hard { address }
        }
        1 => {
            // Soft link: u16 length + UTF-8 path.
            if pos + 2 > body.len() {
                return Err(OxiH5Error::Format(
                    "soft link: missing target path length".to_string(),
                ));
            }
            let path_len = u16::from_le_bytes([body[pos], body[pos + 1]]) as usize;
            pos += 2;
            let path_bytes = body.get(pos..pos + path_len).ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "soft link: path of {} bytes at {} overflows body ({})",
                    path_len,
                    pos,
                    body.len()
                ))
            })?;
            let path = String::from_utf8(path_bytes.to_vec())
                .map_err(|e| OxiH5Error::Format(format!("soft link path not valid UTF-8: {e}")))?;
            Link::Soft { path }
        }
        64 => {
            // External link: u16 value_length + 1 version/flags byte + filename_cstr + NUL + path_cstr + NUL.
            if pos + 2 > body.len() {
                return Err(OxiH5Error::Format(
                    "external link: missing value length".to_string(),
                ));
            }
            let value_len = u16::from_le_bytes([body[pos], body[pos + 1]]) as usize;
            pos += 2;
            let value_end = pos + value_len;
            let value_bytes = body.get(pos..value_end).ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "external link: value of {} bytes at {} overflows body ({})",
                    value_len,
                    pos,
                    body.len()
                ))
            })?;
            // Skip the leading version/flags byte.
            let inner = if value_bytes.is_empty() {
                return Err(OxiH5Error::Format(
                    "external link: empty value payload".to_string(),
                ));
            } else {
                &value_bytes[1..]
            };
            // Split on the NUL separator between filename and path.
            let sep = inner.iter().position(|&b| b == 0).ok_or_else(|| {
                OxiH5Error::Format("external link: missing NUL after filename".to_string())
            })?;
            let file = String::from_utf8(inner[..sep].to_vec()).map_err(|e| {
                OxiH5Error::Format(format!("external link filename not valid UTF-8: {e}"))
            })?;
            let path_bytes = &inner[sep + 1..];
            // Strip trailing NUL if present.
            let path_trimmed = if path_bytes.last() == Some(&0) {
                &path_bytes[..path_bytes.len() - 1]
            } else {
                path_bytes
            };
            let path = String::from_utf8(path_trimmed.to_vec()).map_err(|e| {
                OxiH5Error::Format(format!("external link path not valid UTF-8: {e}"))
            })?;
            Link::External { file, path }
        }
        other => {
            return Err(OxiH5Error::NotImplemented(format!(
                "user-defined link type {other}"
            )));
        }
    };

    Ok(ParsedLink { name, link })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic hard-link message body for a link named "data"
    /// pointing to object header address 0x1000.
    fn build_hard_link_body(name: &str, address: u64, ctx: &ParseContext) -> Vec<u8> {
        // version=1, flags=0b0000_1000 (bit 3 = link type present per HDF5 spec §IV.A.2.c,
        // bits 0-1 = 0 → 1-byte name-len).
        let flags: u8 = 0b0000_1000;
        let mut body = vec![1u8, flags]; // version, flags
        body.push(0u8); // link type = 0 (hard)
                        // No creation order, no charset
        let name_bytes = name.as_bytes();
        body.push(name_bytes.len() as u8); // 1-byte name length
        body.extend_from_slice(name_bytes);
        // Hard link value: soo-byte address LE
        body.extend_from_slice(&address.to_le_bytes()[..ctx.size_of_offsets as usize]);
        body
    }

    #[test]
    fn test_parse_link_hard() {
        let ctx = ParseContext::default_v0();
        let body = build_hard_link_body("data", 0x1000, &ctx);
        let pl = parse_link(&body, &ctx).unwrap();
        assert_eq!(pl.name, "data");
        assert_eq!(pl.link, Link::Hard { address: 0x1000 });
    }

    #[test]
    fn test_parse_link_soft() {
        let ctx = ParseContext::default_v0();
        // flags: bit 3 set (link type present per HDF5 spec §IV.A.2.c), bits 0-1 = 0 (1-byte name-len)
        let flags: u8 = 0b0000_1000;
        let name = "alias";
        let target = "/group/data";
        let mut body = vec![1u8, flags];
        body.push(1u8); // link type = 1 (soft)
        body.push(name.len() as u8);
        body.extend_from_slice(name.as_bytes());
        // Soft link value: u16 path_len + path
        let target_bytes = target.as_bytes();
        let path_len = target_bytes.len() as u16;
        body.extend_from_slice(&path_len.to_le_bytes());
        body.extend_from_slice(target_bytes);

        let pl = parse_link(&body, &ctx).unwrap();
        assert_eq!(pl.name, "alias");
        assert_eq!(
            pl.link,
            Link::Soft {
                path: "/group/data".to_string()
            }
        );
    }

    #[test]
    fn test_parse_link_external() {
        let ctx = ParseContext::default_v0();
        // flags: bit 3 set (link type present per HDF5 spec §IV.A.2.c), bits 0-1 = 0 (1-byte name-len)
        let flags: u8 = 0b0000_1000;
        let name = "ext";
        let filename = "other.h5";
        let ext_path = "/ds";
        let mut body = vec![1u8, flags];
        body.push(64u8); // link type = 64 (external)
        body.push(name.len() as u8);
        body.extend_from_slice(name.as_bytes());
        // External link value: u16 value_len + 1 version byte + filename_cstr + NUL + path_cstr + NUL
        let mut value: Vec<u8> = Vec::new();
        value.push(0x00); // version/flags byte
        value.extend_from_slice(filename.as_bytes());
        value.push(0x00); // NUL separator
        value.extend_from_slice(ext_path.as_bytes());
        value.push(0x00); // trailing NUL
        let vlen = value.len() as u16;
        body.extend_from_slice(&vlen.to_le_bytes());
        body.extend_from_slice(&value);

        let pl = parse_link(&body, &ctx).unwrap();
        assert_eq!(pl.name, "ext");
        assert_eq!(
            pl.link,
            Link::External {
                file: "other.h5".to_string(),
                path: "/ds".to_string()
            }
        );
    }

    #[test]
    fn test_parse_link_info_basic() {
        let ctx = ParseContext::default_v0();
        // flags = 0: no creation-order tracking, no creation-order index
        // body: version(1) + flags(1) + heap_addr(8) + name_index_addr(8) = 18 bytes
        // (HDF5 spec §IV.A.2.a: fractal heap address comes before name-order btree address)
        let mut body = vec![0u8, 0u8]; // version, flags
        let heap_addr: u64 = 0x3000;
        let name_index: u64 = 0x2000;
        body.extend_from_slice(&heap_addr.to_le_bytes());
        body.extend_from_slice(&name_index.to_le_bytes());

        let info = parse_link_info(&body, &ctx).unwrap();
        assert!(!info.creation_order_tracked);
        assert!(info.creation_order_index_address.is_none());
        assert_eq!(info.fractal_heap_address, Some(0x3000));
        assert_eq!(info.name_index_address, Some(0x2000));
    }

    #[test]
    fn test_parse_link_info_with_creation_order() {
        let ctx = ParseContext::default_v0();
        // flags = 0b01: creation order tracked (but not indexed)
        // body: version(1) + flags(1) + creation_order_val(8) + heap(8) + name_index(8)
        // (HDF5 spec §IV.A.2.a: fractal heap address comes before name-order btree address)
        let mut body = vec![0u8, 0b01u8]; // version=0, flags=0x01
        body.extend_from_slice(&42u64.to_le_bytes()); // creation order value
        let heap_addr: u64 = 0x5000;
        let name_index: u64 = 0x4000;
        body.extend_from_slice(&heap_addr.to_le_bytes());
        body.extend_from_slice(&name_index.to_le_bytes());

        let info = parse_link_info(&body, &ctx).unwrap();
        assert!(info.creation_order_tracked);
        assert!(info.creation_order_index_address.is_none()); // bit 1 not set
        assert_eq!(info.fractal_heap_address, Some(0x5000));
        assert_eq!(info.name_index_address, Some(0x4000));
    }

    #[test]
    fn test_parse_link_unknown_type() {
        let ctx = ParseContext::default_v0();
        // flags: bit 3 set (link type present per HDF5 spec §IV.A.2.c), 1-byte name-len
        let flags: u8 = 0b0000_1000;
        let name = "x";
        let mut body = vec![1u8, flags];
        body.push(10u8); // unknown link type
        body.push(name.len() as u8);
        body.extend_from_slice(name.as_bytes());

        let result = parse_link(&body, &ctx);
        assert!(
            matches!(result, Err(OxiH5Error::NotImplemented(_))),
            "expected NotImplemented, got {result:?}"
        );
    }
}
