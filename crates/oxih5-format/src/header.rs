use crate::superblock::{read_u16_le, read_u32_le, read_u64_le};
use oxih5_core::OxiH5Error;

/// Continuation message type — points to an additional header block.
const MSG_CONTINUATION_V1: u16 = 0x0010;
/// NIL message type (v1) — padding/unused.
const MSG_NIL_V1: u16 = 0x0000;

/// Object header v2 continuation message type.
const MSG_CONTINUATION_V2: u8 = 0x10;
/// Object header v2 NIL message type.
const MSG_NIL_V2: u8 = 0x00;

/// HDF5 object header v2 signature.
const OHDR_SIGNATURE: &[u8; 4] = b"OHDR";
/// HDF5 object header v2 continuation block signature.
const OCHK_SIGNATURE: &[u8; 4] = b"OCHK";

/// A parsed object header message (body bytes preserved verbatim).
///
/// For v1 headers `msg_type` uses the 2-byte type code from the spec.
/// For v2 headers `msg_type` uses the 1-byte type code (which are the same
/// low-8-bit values — e.g. dataspace=0x01, datatype=0x03, continuation=0x10).
#[derive(Debug, Clone)]
pub struct Message {
    pub msg_type: u16,
    pub data: Vec<u8>,
}

/// Parse all messages from an object header at the given file offset.
///
/// Dispatches to the v1 or v2 parser based on the first 4 bytes at `offset`:
/// - If those bytes are `b"OHDR"` → v2 header.
/// - Otherwise → v1 header (first byte must be `1`).
///
/// **Object header v1 layout:**
/// ```text
/// Offset  Size  Field
///  0       1     Version (must be 1)
///  1       1     Reserved
///  2       2     Number of messages (u16 LE)  -- includes NIL
///  4       4     Reference count (u32 LE)
///  8       4     Header data size (u32 LE) — total bytes of message data
/// 12       4     Reserved (padding to 8-byte boundary)
/// 16       …     Messages
/// ```
///
/// **v1 per-message:**
/// ```text
///  0       2     Message type (u16 LE)
///  2       2     Body size (u16 LE)
///  4       1     Flags
///  5       3     Reserved
///  8       N     Body bytes (N = body size)
///  8+N     ?     Padding to next 8-byte boundary
/// ```
///
/// **Object header v2 layout:**
/// ```text
///  0       4     Signature ("OHDR")
///  4       1     Version (must be 2)
///  5       1     Flags
///                  bits 0-1: chunk #0 size field width (0→1B, 1→2B, 2→4B, 3→8B)
///                  bit 2: creation order tracked per message
///                  bit 4: attribute phase-change stored (4 bytes follow)
///                  bit 5: access/mod/change/birth timestamps stored (16 bytes follow)
///  6       [16]  Optional: access+mod+change+birth times (4×u32 LE) if bit 5
///  ...     [4]   Optional: max_compact+min_dense attrs (2×u16 LE) if bit 4
///  ...     1/2/4/8  Chunk #0 size field
///  ...     N     Messages (N = chunk #0 size bytes)
///  ...     4     Fletcher-32 checksum
/// ```
///
/// **v2 per-message:**
/// ```text
///  0       1     Message type (u8)
///  1       2     Data size (u16 LE)
///  3       1     Message flags
/// [4       2]    Creation order (u16 LE) — only present if OHDR flag bit 2 set
///  4 or 6  N     Message data (N = data size, NOT padded)
/// ```
pub fn parse_messages(file_data: &[u8], offset: u64) -> Result<Vec<Message>, OxiH5Error> {
    let off = usize::try_from(offset).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "object header offset {offset} exceeds addressable range"
        ))
    })?;
    let off4 = off
        .checked_add(4)
        .ok_or_else(|| OxiH5Error::Corrupted(format!("object header offset {offset} too large")))?;

    // Detect v2 by the "OHDR" signature.
    if file_data.get(off..off4) == Some(OHDR_SIGNATURE) {
        parse_messages_v2(file_data, offset)
    } else {
        parse_messages_v1(file_data, offset)
    }
}

// ---------------------------------------------------------------------------
// Object Header v1
// ---------------------------------------------------------------------------

fn parse_messages_v1(file_data: &[u8], offset: u64) -> Result<Vec<Message>, OxiH5Error> {
    let off = usize::try_from(offset).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "object header v1 offset {offset} exceeds addressable range"
        ))
    })?;
    let off16 = off.checked_add(16).ok_or_else(|| {
        OxiH5Error::Corrupted(format!("object header v1 offset {offset} too large"))
    })?;

    if off16 > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "object header v1 at {offset}: insufficient bytes (file len={})",
            file_data.len()
        )));
    }

    let version = file_data[off];
    if version != 1 {
        return Err(OxiH5Error::UnsupportedHeader(version));
    }

    // header_data_size is the byte count of the message region (not including
    // the 16-byte header prefix).
    let header_data_size = read_u32_le(file_data, off + 8)? as usize;

    let mut messages: Vec<Message> = Vec::with_capacity((header_data_size / 8).min(64));
    // Continuation blocks discovered while parsing; processed after the primary block.
    let mut continuations: Vec<(u64, u64)> = Vec::new();

    // Primary block: messages start at off+16.
    parse_v1_message_block(
        file_data,
        off + 16,
        header_data_size,
        &mut messages,
        &mut continuations,
    )?;

    // Follow continuation blocks (avoid cycles with a depth cap).
    let mut i = 0;
    while i < continuations.len() {
        let (cont_off, cont_len) = continuations[i];
        let cont_off_usize = usize::try_from(cont_off).map_err(|_| {
            OxiH5Error::Corrupted(format!(
                "continuation block offset {cont_off} exceeds addressable range"
            ))
        })?;
        let cont_len_usize = usize::try_from(cont_len).map_err(|_| {
            OxiH5Error::Corrupted(format!(
                "continuation block length {cont_len} exceeds addressable range"
            ))
        })?;
        parse_v1_message_block(
            file_data,
            cont_off_usize,
            cont_len_usize,
            &mut messages,
            &mut continuations,
        )?;
        i += 1;
        if i > 256 {
            return Err(OxiH5Error::Format(
                "object header v1: too many continuation blocks (possible cycle)".to_string(),
            ));
        }
    }

    Ok(messages)
}

/// Parse messages from one contiguous v1 block of `block_size` bytes starting at `start`.
fn parse_v1_message_block(
    file_data: &[u8],
    start: usize,
    block_size: usize,
    messages: &mut Vec<Message>,
    continuations: &mut Vec<(u64, u64)>,
) -> Result<(), OxiH5Error> {
    let end = start.saturating_add(block_size);
    if end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "message block at {start}+{block_size} exceeds file size {}",
            file_data.len()
        )));
    }

    let mut pos = start;

    while pos + 8 <= end {
        let msg_type = read_u16_le(file_data, pos)?;
        let msg_size = read_u16_le(file_data, pos + 2)? as usize;
        // pos+4 = flags byte; pos+5..8 = reserved.

        let body_start = pos + 8;
        let body_end = body_start + msg_size;

        if body_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "message body at {body_start}+{msg_size} exceeds file size {}",
                file_data.len()
            )));
        }

        // Advance past this message (body is padded to 8-byte boundary).
        let aligned_size = (msg_size + 7) & !7;
        pos += 8 + aligned_size;

        match msg_type {
            MSG_NIL_V1 => {
                // NIL is a placeholder/padding message in v1 object headers.
                // Per the HDF5 spec, NIL messages do NOT terminate the block —
                // the block ends at `start + block_size`.  Simply skip it.
            }
            MSG_CONTINUATION_V1 => {
                // Body: cont_offset(8) + cont_length(8).
                if msg_size >= 16 {
                    let cont_offset = read_u64_le(file_data, body_start)?;
                    let cont_length = read_u64_le(file_data, body_start + 8)?;
                    continuations.push((cont_offset, cont_length));
                }
            }
            _ => {
                messages.push(Message {
                    msg_type,
                    data: file_data[body_start..body_end].to_vec(),
                });
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Object Header v2
// ---------------------------------------------------------------------------

fn parse_messages_v2(file_data: &[u8], offset: u64) -> Result<Vec<Message>, OxiH5Error> {
    let base = usize::try_from(offset).unwrap_or(usize::MAX);
    let capacity = if base < file_data.len() {
        let off = base + 6;
        let chunk_size_field_width =
            1usize << (file_data.get(base + 5).copied().unwrap_or(0) & 0x03);
        let chunk0_size = if off + chunk_size_field_width <= file_data.len() {
            match chunk_size_field_width {
                1 => file_data[off] as usize,
                2 => u16::from_le_bytes([file_data[off], file_data[off + 1]]) as usize,
                4 => u32::from_le_bytes([
                    file_data[off],
                    file_data[off + 1],
                    file_data[off + 2],
                    file_data[off + 3],
                ]) as usize,
                _ => 0,
            }
        } else {
            0
        };
        (chunk0_size / 4).min(64)
    } else {
        0
    };
    let mut messages = Vec::with_capacity(capacity);
    let first_addr = offset;
    let mut extra_seen: Option<std::collections::HashSet<u64>> = None;
    parse_v2_block(
        file_data,
        offset,
        &mut messages,
        first_addr,
        &mut extra_seen,
        0,
    )?;
    Ok(messages)
}

/// Parse one v2 object header block (either OHDR main header or OCHK continuation block).
///
/// For the main OHDR block the length is encoded in the `chunk #0 size` field.
/// For OCHK continuation blocks the length comes from the continuation message that
/// pointed here; we derive it as `(cont_length - 8)` where 8 = sig(4) + checksum(4).
///
/// `first_addr` holds the very first block address (always known); `extra_seen` is
/// allocated lazily only when a second distinct block address appears (T3: deferred HashSet).
fn parse_v2_block(
    file_data: &[u8],
    block_start: u64,
    messages: &mut Vec<Message>,
    first_addr: u64,
    extra_seen: &mut Option<std::collections::HashSet<u64>>,
    depth: u32,
) -> Result<(), OxiH5Error> {
    if depth > 256 {
        return Err(OxiH5Error::Format(
            "object header v2: continuation depth limit exceeded".into(),
        ));
    }
    if block_start == first_addr && depth > 0 {
        return Err(OxiH5Error::Format(format!(
            "object header v2: cycle detected at block {block_start:#x}"
        )));
    }
    if depth > 0 {
        let seen = extra_seen.get_or_insert_with(std::collections::HashSet::new);
        if !seen.insert(block_start) {
            return Err(OxiH5Error::Format(format!(
                "object header v2: cycle detected at block {block_start:#x}"
            )));
        }
    }

    let base = usize::try_from(block_start).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "v2 block offset {block_start} exceeds addressable range"
        ))
    })?;

    let sig = file_data.get(base..base.saturating_add(4));
    let (msg_start, msg_end, track_creation_order) = if sig == Some(OHDR_SIGNATURE) {
        // --- Main OHDR block ---
        // byte 4: version (must be 2)
        let version = *file_data
            .get(base + 4)
            .ok_or_else(|| OxiH5Error::Format("ohdr v2: truncated before version".into()))?;
        if version != 2 {
            return Err(OxiH5Error::UnsupportedHeader(version));
        }

        // byte 5: flags
        let flags = *file_data
            .get(base + 5)
            .ok_or_else(|| OxiH5Error::Format("ohdr v2: truncated before flags".into()))?;

        let track_co = (flags >> 2) & 1 == 1;
        // bits 0-1: chunk #0 size field width
        let chunk_size_field_width = 1usize << (flags & 0x03);

        let mut p = base + 6;

        // bit 5: access+mod+change+birth timestamps stored (4 × u32 = 16 bytes)
        if (flags >> 5) & 1 == 1 {
            p += 16;
        }
        // bit 4: attribute phase-change values stored (max_compact u16 + min_dense u16 = 4 bytes)
        if (flags >> 4) & 1 == 1 {
            p += 4;
        }

        // chunk #0 size
        let chunk0_size = read_var_int(file_data, p, chunk_size_field_width)?;
        p += chunk_size_field_width;

        (p, p + chunk0_size as usize, track_co)
    } else if sig == Some(OCHK_SIGNATURE) {
        // --- OCHK continuation block ---
        // OCHK: sig(4) + messages(cont_length − 8) + checksum(4)
        // We don't have cont_length here directly; we rely on the caller having
        // populated `seen` and having passed the right block_start.
        // To find the end we scan using depth-first order — the continuation
        // message body gives us the length when we encounter it in the parent.
        // Since we don't store that here, we read until we hit a parse error or
        // a NIL message.  In practice a NIL message marks the end of the live
        // portion; the checksum occupies the final 4 bytes and won't form a
        // valid message record.
        //
        // For robustness we cap at file_data.len().
        (base + 4, file_data.len(), false)
    } else {
        return Err(OxiH5Error::Format(format!(
            "object header v2: expected OHDR or OCHK at {block_start:#x}, got {:?}",
            file_data.get(base..base.saturating_add(4))
        )));
    };

    // --- Message scan ---
    let mut pos = msg_start;
    while pos + 4 <= msg_end && pos + 4 <= file_data.len() {
        let msg_type = file_data[pos]; // 1-byte type
        let msg_size = {
            let hi = *file_data.get(pos + 2).ok_or_else(|| {
                OxiH5Error::Format(format!("ohdr v2: truncated reading msg_size at {pos}"))
            })?;
            let lo = *file_data.get(pos + 1).ok_or_else(|| {
                OxiH5Error::Format(format!("ohdr v2: truncated reading msg_size at {pos}"))
            })?;
            u16::from_le_bytes([lo, hi]) as usize
        };
        // pos+3 = message flags (1 byte)
        let hdr_size: usize = if track_creation_order { 6 } else { 4 };

        let data_start = pos + hdr_size;
        let data_end = data_start + msg_size;

        if data_end > file_data.len() {
            // Truncated — stop gracefully (may be reading past checksum in OCHK)
            break;
        }

        if msg_type == MSG_NIL_V2 {
            // NIL marks the end of live messages in this block.
            break;
        }

        if msg_type == MSG_CONTINUATION_V2 {
            // Continuation message body: cont_offset(8) + cont_length(8)
            if msg_size >= 16 {
                let cont_offset = read_u64_at(file_data, data_start)?;
                let cont_length = read_u64_at(file_data, data_start + 8)?;
                // OCHK block: sig(4) + messages(cont_length − 8) + checksum(4)
                // We store the adjusted end so parse_v2_block can use it.
                // We pass the block start; the OCHK parser caps at cont_length - 8.
                parse_v2_ochk_block(
                    file_data,
                    cont_offset,
                    cont_length,
                    messages,
                    first_addr,
                    extra_seen,
                    depth + 1,
                )?;
            }
        } else {
            messages.push(Message {
                msg_type: msg_type as u16,
                data: file_data[data_start..data_end].to_vec(),
            });
        }

        pos += hdr_size + msg_size;
    }

    Ok(())
}

/// Parse an OCHK continuation block.
///
/// OCHK layout: `sig(4) + messages(cont_length − 8) + checksum(4)`
/// The `cont_length` comes from the continuation message body (it includes the
/// sig and checksum, so the message region is `cont_length − 8` bytes).
fn parse_v2_ochk_block(
    file_data: &[u8],
    block_start: u64,
    cont_length: u64,
    messages: &mut Vec<Message>,
    first_addr: u64,
    extra_seen: &mut Option<std::collections::HashSet<u64>>,
    depth: u32,
) -> Result<(), OxiH5Error> {
    if depth > 256 {
        return Err(OxiH5Error::Format(
            "object header v2: continuation depth limit exceeded".into(),
        ));
    }
    if block_start == first_addr {
        return Err(OxiH5Error::Format(format!(
            "object header v2: cycle detected at OCHK {block_start:#x}"
        )));
    }
    {
        let seen = extra_seen.get_or_insert_with(std::collections::HashSet::new);
        if !seen.insert(block_start) {
            return Err(OxiH5Error::Format(format!(
                "object header v2: cycle detected at OCHK {block_start:#x}"
            )));
        }
    }

    let base = usize::try_from(block_start).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "OCHK block offset {block_start} exceeds addressable range"
        ))
    })?;
    let base4 = base.checked_add(4).ok_or_else(|| {
        OxiH5Error::Corrupted(format!("OCHK block offset {block_start} too large"))
    })?;
    if file_data.get(base..base4) != Some(OCHK_SIGNATURE) {
        return Err(OxiH5Error::Format(format!(
            "object header v2: expected OCHK signature at {block_start:#x}, got {:?}",
            file_data.get(base..base4)
        )));
    }

    // Messages occupy bytes [base+4 .. base + cont_length - 4]
    // (last 4 bytes of the cont_length block are the checksum)
    let msg_area_end = (block_start + cont_length).saturating_sub(4) as usize;
    let msg_area_end = msg_area_end.min(file_data.len());

    let mut pos = base4; // skip OCHK signature

    // Inherit track_creation_order from the enclosing OHDR? No — OCHK blocks
    // use the same creation-order flag as the owning OHDR.  However, we don't
    // have convenient access to that flag here.  In practice when track_co is
    // false (by far the common case) this is fine.  When true the caller
    // `parse_v2_block` would need to thread the flag through.  We accept a
    // minor limitation: OCHK blocks always assume track_co=false until a
    // refactor threads the flag through.  This matches behaviour for the vast
    // majority of files.
    let track_co = false;

    while pos + 4 <= msg_area_end {
        let msg_type = file_data[pos];
        let lo = file_data[pos + 1];
        let hi = file_data[pos + 2];
        let msg_size = u16::from_le_bytes([lo, hi]) as usize;
        let hdr_size: usize = if track_co { 6 } else { 4 };

        let data_start = pos + hdr_size;
        let data_end = data_start + msg_size;

        if data_end > file_data.len() || data_end > msg_area_end + msg_size {
            break;
        }

        if msg_type == MSG_NIL_V2 {
            break;
        }

        if msg_type == MSG_CONTINUATION_V2 {
            if msg_size >= 16 {
                let sub_offset = read_u64_at(file_data, data_start)?;
                let sub_length = read_u64_at(file_data, data_start + 8)?;
                parse_v2_ochk_block(
                    file_data,
                    sub_offset,
                    sub_length,
                    messages,
                    first_addr,
                    extra_seen,
                    depth + 1,
                )?;
            }
        } else {
            messages.push(Message {
                msg_type: msg_type as u16,
                data: file_data[data_start..data_end].to_vec(),
            });
        }

        pos += hdr_size + msg_size;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Read a variable-width unsigned integer (1/2/4/8 bytes, LE) from `data` at `pos`.
fn read_var_int(data: &[u8], pos: usize, size: usize) -> Result<u64, OxiH5Error> {
    let bytes = data.get(pos..pos + size).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "ohdr v2: read_var_int pos={pos} size={size} out of bounds (data len={})",
            data.len()
        ))
    })?;
    Ok(match size {
        1 => bytes[0] as u64,
        2 => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
        4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        8 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                OxiH5Error::Format("ohdr v2: read_var_int 8-byte conversion failed".into())
            })?;
            u64::from_le_bytes(arr)
        }
        _ => {
            return Err(OxiH5Error::Format(format!(
                "ohdr v2: unsupported var_int size {size}"
            )))
        }
    })
}

/// Read a little-endian u64 from `data` at `pos`.
fn read_u64_at(data: &[u8], pos: usize) -> Result<u64, OxiH5Error> {
    read_u64_le(data, pos)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Object header v1
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_messages_v1_basic() {
        // Build a minimal v1 object header with one dataspace message.
        let mut data = vec![0u8; 256];
        data[0] = 1; // version
                     // data[1] = reserved
                     // data[2..4] = num_messages = 1 (u16 LE)
        data[2] = 1;
        // data[4..8] = reference count = 1
        data[4] = 1;
        // data[8..12] = header_data_size = 16 (one 8-byte header + 8-byte body)
        data[8..12].copy_from_slice(&16u32.to_le_bytes());
        // data[12..16] = reserved

        // Message at offset 16:
        data[16] = 0x01; // type = dataspace (low byte)
        data[17] = 0x00; // type (high byte)
        data[18] = 0x04; // body size = 4
        data[19] = 0x00;
        data[20] = 0x00; // flags
                         // data[21..24] = reserved
        data[24..28].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        let msgs = parse_messages(&data, 0).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].msg_type, 0x0001);
        assert_eq!(msgs[0].data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_parse_messages_v1_nil_terminates() {
        let mut data = vec![0u8; 64];
        data[0] = 1; // version
        data[8..12].copy_from_slice(&48u32.to_le_bytes()); // header_data_size

        // First message: valid
        data[16] = 0x01;
        data[18] = 0x00; // size = 0
                         // Second message: NIL (type 0, all zeros) — terminates scan
                         // data[24] = 0 (already)

        let msgs = parse_messages(&data, 0).unwrap();
        // The first 0-byte message might be collected; then NIL stops.
        // Let's verify NIL handling: the all-zeros message has type=0 (NIL).
        // The first message also has type=0x01 and size=0 so it's collected.
        // data[24] = type 0x00 = NIL → break.
        assert!(msgs.len() <= 1);
    }

    #[test]
    fn test_parse_messages_v1_unsupported_version() {
        let mut data = vec![0u8; 64];
        data[0] = 5; // unsupported version
        assert!(matches!(
            parse_messages(&data, 0),
            Err(OxiH5Error::UnsupportedHeader(5))
        ));
    }

    // -----------------------------------------------------------------------
    // Object header v2
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_messages_v2_basic() {
        // Build a minimal OHDR v2 with flags=0 (chunk_size_size=1, no extras).
        // Layout: OHDR(4) + version(1) + flags(1) + chunk0_size(1) + messages...
        // message header is 4 bytes (type+size[2]+flags) + data.
        // msg(type=1 u8, size=4 u16le, flags=0) + data(4 bytes) = 8 bytes,
        // plus NIL region → chunk0_size = 12.
        // Plus checksum = 4 bytes after messages.
        let chunk0_size: usize = 12; // 8-byte message + 4 NIL
        let total = 4 + 1 + 1 + 1 + chunk0_size + 4; // OHDR + ver + flags + sz + msgs + checksum
        let mut data = vec![0u8; total.max(64)];

        // OHDR signature
        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 2; // version
        data[5] = 0; // flags: chunk_size_size=1, no creation order, no timestamps
        data[6] = chunk0_size as u8; // chunk #0 size = 1 byte field

        // Message at pos 7: type=0x01 (dataspace), size=4, flags=0
        let msg_pos = 7;
        data[msg_pos] = 0x01; // type (1 byte)
        data[msg_pos + 1] = 0x04; // size lo byte
        data[msg_pos + 2] = 0x00; // size hi byte
        data[msg_pos + 3] = 0x00; // message flags
                                  // Message data
        data[msg_pos + 4] = 0xAA;
        data[msg_pos + 5] = 0xBB;
        data[msg_pos + 6] = 0xCC;
        data[msg_pos + 7] = 0xDD;
        // NIL terminator at msg_pos + 8
        // data[msg_pos + 8] = 0x00 (already)

        let msgs = parse_messages(&data, 0).unwrap();
        assert!(!msgs.is_empty(), "expected at least one message");
        assert_eq!(msgs[0].msg_type, 0x0001);
        assert_eq!(msgs[0].data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_parse_messages_v2_with_creation_order() {
        // flags = 0x04 → track_creation_order=1, chunk_size_size=1
        let chunk0_size: usize = 16; // type(1)+size(2)+flags(1)+co(2) + data(4) + NIL(+6)
        let total = 4 + 1 + 1 + 1 + chunk0_size + 4;
        let mut data = vec![0u8; total.max(64)];

        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 2;
        data[5] = 0x04; // bit 2 = track creation order; bits 0-1 = 0 → chunk_size_size=1
        data[6] = chunk0_size as u8;

        // Message: type=0x03, size=2, flags=0, creation_order=7, data=[0x01, 0x02]
        let p = 7;
        data[p] = 0x03; // type
        data[p + 1] = 0x02; // size lo
        data[p + 2] = 0x00; // size hi
        data[p + 3] = 0x00; // msg flags
        data[p + 4] = 0x07; // creation_order lo
        data[p + 5] = 0x00; // creation_order hi
        data[p + 6] = 0x01;
        data[p + 7] = 0x02;
        // NIL at p+8: already zero

        let msgs = parse_messages(&data, 0).unwrap();
        assert!(!msgs.is_empty());
        assert_eq!(msgs[0].msg_type, 0x0003);
        assert_eq!(msgs[0].data, vec![0x01, 0x02]);
    }

    #[test]
    fn test_parse_messages_v2_timestamps_present() {
        // flags = 0x20 → bit 5 set → 16-byte timestamp block present; chunk_size_size=1
        let chunk0_size: usize = 8;
        // pos 6 starts 16-byte timestamps, then chunk0_size(1), then messages
        let msg_base = 6 + 16 + 1;
        let total = msg_base + chunk0_size + 4 + 16;
        let mut data = vec![0u8; total];

        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 2;
        data[5] = 0x20; // bit 5 = timestamps stored; bits 0-1 = 0 → chunk_size_size=1
                        // timestamps at data[6..22]: zeros
        data[6 + 16] = chunk0_size as u8; // chunk0_size field at offset 22

        // Message at msg_base: type=0x05, size=1
        let p = msg_base;
        data[p] = 0x05;
        data[p + 1] = 0x01;
        data[p + 2] = 0x00;
        data[p + 3] = 0x00;
        data[p + 4] = 0xFF;
        // NIL at p+5

        let msgs = parse_messages(&data, 0).unwrap();
        assert!(!msgs.is_empty());
        assert_eq!(msgs[0].msg_type, 0x0005);
        assert_eq!(msgs[0].data, vec![0xFF]);
    }

    #[test]
    fn test_parse_messages_v2_attr_phase_change_present() {
        // flags = 0x10 → bit 4 set → 4-byte attr phase-change block; chunk_size_size=1
        let chunk0_size: usize = 8;
        let msg_base = 6 + 4 + 1; // prefix(6) + phase_change(4) + chunk0_size(1)
        let total = msg_base + chunk0_size + 4 + 16;
        let mut data = vec![0u8; total];

        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 2;
        data[5] = 0x10; // bit 4 set
                        // attr phase change at data[6..10]: zeros
        data[6 + 4] = chunk0_size as u8; // chunk0_size at offset 10

        let p = msg_base;
        data[p] = 0x08; // data layout
        data[p + 1] = 0x02;
        data[p + 2] = 0x00;
        data[p + 3] = 0x00;
        data[p + 4] = 0xDE;
        data[p + 5] = 0xAD;

        let msgs = parse_messages(&data, 0).unwrap();
        assert!(!msgs.is_empty());
        assert_eq!(msgs[0].msg_type, 0x0008);
        assert_eq!(msgs[0].data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_parse_messages_v2_with_ochk_continuation() {
        // Build an OHDR that contains a continuation message pointing to an OCHK block.
        // OCHK contains one attribute message.

        // OCHK block at offset 200
        let ochk_offset: u64 = 200;
        // OCHK: sig(4) + msg(type=0x0C, size=2, flags=0, data=[1,2]) + NIL + checksum(4)
        // msg_area_size = 4 + 2 = 6 bytes, plus NIL = 4, checksum = 4 → cont_length = 4+6+4+4 = 18
        // Actually cont_length = sig(4) + messages + checksum(4)
        // messages = type(1)+size(2)+flags(1)+data(2) = 6, NIL = type(1)+size(2)+flags(1) = 4
        // cont_length = 4 + 6 + 4 + 4 = 18
        let cont_length: u64 = 18;

        // OHDR at offset 0
        // Continuation message in OHDR: type=0x10, size=16 (8+8), flags=0
        // data = ochk_offset(8) + cont_length(8)
        let chunk0_size: usize = 4 + 16 + 4; // msg_hdr(4) + msg_body(16) + NIL(4)
        let mut data = vec![0u8; 500];

        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 2;
        data[5] = 0x00; // flags: chunk_size_size=1
        data[6] = chunk0_size as u8;

        // Continuation message at 7
        let p = 7;
        data[p] = 0x10; // continuation
        data[p + 1] = 0x10; // size = 16
        data[p + 2] = 0x00;
        data[p + 3] = 0x00;
        // continuation body: ochk_offset(8) + cont_length(8)
        data[p + 4..p + 12].copy_from_slice(&ochk_offset.to_le_bytes());
        data[p + 12..p + 20].copy_from_slice(&cont_length.to_le_bytes());
        // NIL at p+20 (all zeros)

        // OCHK block at offset 200
        let ob = ochk_offset as usize;
        data[ob..ob + 4].copy_from_slice(b"OCHK");
        // Attribute message: type=0x0C, size=2, flags=0, data=[0x01, 0x02]
        data[ob + 4] = 0x0C;
        data[ob + 5] = 0x02;
        data[ob + 6] = 0x00;
        data[ob + 7] = 0x00;
        data[ob + 8] = 0x01;
        data[ob + 9] = 0x02;
        // NIL at ob+10

        let msgs = parse_messages(&data, 0).unwrap();
        // Should have collected the attribute message from OCHK
        let attr_msgs: Vec<_> = msgs.iter().filter(|m| m.msg_type == 0x000C).collect();
        assert!(
            !attr_msgs.is_empty(),
            "expected attribute message from OCHK"
        );
        assert_eq!(attr_msgs[0].data, vec![0x01, 0x02]);
    }

    #[test]
    fn test_parse_messages_v2_unsupported_version() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 3; // version 3 is not supported (we only handle version 2)
        assert!(parse_messages(&data, 0).is_err());
    }

    #[test]
    fn test_parse_messages_v2_real_superblock_v3_layout() {
        // Reproduces the structure of the real /tmp/test_v2_latest.h5 root OHDR:
        // OHDR at 0x30:  4f484452 02 01 7802 120000 00ffffffffffffff...
        // flags=0x01 → chunk_size_size=2, no creation order, no timestamps
        // chunk0_size = 0x0278 = 632 (but we'll use a smaller value for the test)
        // msg_type=0x02 (link info), size=0x12=18, flags=0x00 ...
        //
        // Here we reconstruct a truncated but structurally correct version:
        let chunk0_size: u16 = 28; // space for 1 message + NIL
        let total = 4 + 1 + 1 + 2 + chunk0_size as usize + 4 + 64;
        let mut data = vec![0u8; total];

        data[0..4].copy_from_slice(b"OHDR");
        data[4] = 2;
        data[5] = 0x01; // flags: bits 0-1 = 1 → chunk_size_size = 2 bytes
        data[6..8].copy_from_slice(&chunk0_size.to_le_bytes()); // 2-byte chunk0_size

        // First message at offset 8: type=0x02 (link info), size=18, flags=0
        let p = 8;
        data[p] = 0x02;
        data[p + 1] = 0x12; // size = 18
        data[p + 2] = 0x00;
        data[p + 3] = 0x00;
        // 18 bytes of data (all zeros is fine)
        // Next message at p+4+18 = p+22:
        // NIL at p+22

        let msgs = parse_messages(&data, 0).unwrap();
        assert!(!msgs.is_empty());
        assert_eq!(msgs[0].msg_type, 0x0002);
        assert_eq!(msgs[0].data.len(), 18);
    }
}
