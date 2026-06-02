use crate::superblock::{read_u16_le, read_u32_le, read_u64_le};
use oxih5_core::OxiH5Error;

/// Sentinel address in HDF5 files (all-bits-1 means "undefined").
const UNDEF: u64 = u64::MAX;

/// Maximum B-tree depth before we abort to prevent infinite recursion.
const MAX_DEPTH: u16 = 64;

// ---------------------------------------------------------------------------
// Name-index (type 5) public entry point
// ---------------------------------------------------------------------------

/// Parse a B-tree v2 group name index (type 5), returning all raw heap IDs.
///
/// Each leaf record in the tree has: 4-byte name_hash (Jenkins lookup3) +
/// `heap_id_len` bytes of heap ID.  Only the heap IDs are returned (the hash
/// is ignored — we want full enumeration, not lookup).
///
/// Returns `Vec` of heap ID byte vectors (one per link in the group).
pub fn parse_name_index(
    file_data: &[u8],
    header_address: u64,
    heap_id_len: u8,
) -> Result<Vec<Vec<u8>>, OxiH5Error> {
    let base = header_address as usize;

    // B-tree v2 header ("BTHD") — identical layout to the chunk-index case.
    let hdr_end = base
        .checked_add(38)
        .ok_or_else(|| OxiH5Error::Format("BTHD (name index): header address overflow".into()))?;
    if hdr_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "BTHD (name index): header at {:#x} exceeds file length {}",
            base,
            file_data.len()
        )));
    }

    let sig = &file_data[base..base + 4];
    if sig != b"BTHD" {
        return Err(OxiH5Error::Format(format!(
            "BTHD (name index): bad signature {sig:?} at {base:#x}"
        )));
    }

    let version = file_data[base + 4];
    if version != 0 {
        return Err(OxiH5Error::Format(format!(
            "BTHD (name index): unsupported version {version}"
        )));
    }

    let btree_type = file_data[base + 5];
    if btree_type != 5 {
        return Err(OxiH5Error::NotImplemented(format!(
            "BTHD (name index): expected type 5, got {btree_type}"
        )));
    }

    let _node_size = read_u32_le(file_data, base + 6)?;
    let record_size = read_u16_le(file_data, base + 10)?;
    let tree_depth = read_u16_le(file_data, base + 12)?;

    // Validate: record_size must equal 4 (name_hash) + heap_id_len.
    let expected_record_size = 4u16.saturating_add(heap_id_len as u16);
    if record_size != expected_record_size {
        return Err(OxiH5Error::Format(format!(
            "BTHD (name index): record_size {record_size} != 4 + heap_id_len {heap_id_len} = {expected_record_size}"
        )));
    }

    if tree_depth > MAX_DEPTH {
        return Err(OxiH5Error::Format(format!(
            "BTHD (name index): tree depth {tree_depth} exceeds maximum {MAX_DEPTH}"
        )));
    }

    let root_addr = read_u64_le(file_data, base + 16)?;
    let root_nrecords = read_u16_le(file_data, base + 24)?;

    if root_addr == UNDEF {
        // Empty tree — no links.
        return Ok(Vec::new());
    }

    let mut heap_ids = Vec::new();
    parse_name_index_node(
        file_data,
        root_addr,
        tree_depth,
        root_nrecords,
        record_size,
        heap_id_len,
        &mut heap_ids,
        0,
    )?;

    Ok(heap_ids)
}

/// Recursive traversal of a B-tree v2 name-index node.
///
/// Leaf nodes (BTLF, depth=0): skip 4-byte hash, collect `heap_id_len` bytes.
/// Internal nodes (BTIN, depth>0): recurse into children.
#[allow(clippy::too_many_arguments)]
fn parse_name_index_node(
    file_data: &[u8],
    node_addr: u64,
    depth: u16,
    num_records: u16,
    record_size: u16,
    heap_id_len: u8,
    heap_ids: &mut Vec<Vec<u8>>,
    recursion: u16,
) -> Result<(), OxiH5Error> {
    if recursion > MAX_DEPTH {
        return Err(OxiH5Error::Format(
            "BTreeV2 name index: recursion limit reached".into(),
        ));
    }
    if node_addr == UNDEF {
        return Ok(());
    }

    let base = node_addr as usize;

    // Both BTIN and BTLF start with: signature(4) + version(1) + type(1) = 6 bytes.
    let sig_end = base
        .checked_add(6)
        .ok_or_else(|| OxiH5Error::Format("BTreeV2 name-index node: address overflow".into()))?;
    if sig_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "BTreeV2 name-index node at {:#x}: truncated",
            base
        )));
    }

    let expected_sig: &[u8] = if depth == 0 { b"BTLF" } else { b"BTIN" };
    let sig = &file_data[base..base + 4];
    if sig != expected_sig {
        return Err(OxiH5Error::Format(format!(
            "BTreeV2 name-index node at {base:#x} depth={depth}: expected {expected_sig:?}, got {sig:?}"
        )));
    }

    let node_version = file_data[base + 4];
    if node_version != 0 {
        return Err(OxiH5Error::Format(format!(
            "BTreeV2 name-index node: unsupported version {node_version}"
        )));
    }

    // Records start at offset 6 in the node.
    let records_start = base + 6;
    let record_count = num_records as usize;
    let rs = record_size as usize;
    let hil = heap_id_len as usize;

    if depth == 0 {
        // ---------- Leaf node (BTLF) ----------
        let records_end = records_start
            .checked_add(record_count * rs)
            .ok_or_else(|| OxiH5Error::Format("BTLF (name index): record range overflow".into()))?;
        if records_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "BTLF (name index) at {base:#x}: record data truncated (need {records_end}, have {})",
                file_data.len()
            )));
        }

        for i in 0..record_count {
            let r_off = records_start + i * rs;
            // Skip 4 bytes of name hash, take heap_id_len bytes.
            let id_off = r_off + 4;
            let id_end = id_off + hil;
            let id_bytes = file_data.get(id_off..id_end).ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "BTLF (name index) at {base:#x}: record {i} heap_id out of bounds"
                ))
            })?;
            heap_ids.push(id_bytes.to_vec());
        }
    } else {
        // ---------- Internal node (BTIN) ----------
        let child_count = record_count + 1;
        // Each child pointer: address(8) + num_records_in_child(2) — same as chunk tree.
        let child_ptr_size = 8 + 2;

        let records_end = records_start
            .checked_add(record_count * rs)
            .ok_or_else(|| OxiH5Error::Format("BTIN (name index): record range overflow".into()))?;
        let children_end = records_end
            .checked_add(child_count * child_ptr_size)
            .ok_or_else(|| {
                OxiH5Error::Format("BTIN (name index): children range overflow".into())
            })?;

        if children_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "BTIN (name index) at {base:#x}: node data truncated (need {children_end}, have {})",
                file_data.len()
            )));
        }

        for c in 0..child_count {
            let ptr_off = records_end + c * child_ptr_size;
            let child_addr = read_u64_le(file_data, ptr_off)?;
            let child_nrecords = read_u16_le(file_data, ptr_off + 8)?;

            if child_addr == UNDEF {
                continue;
            }

            parse_name_index_node(
                file_data,
                child_addr,
                depth - 1,
                child_nrecords,
                record_size,
                heap_id_len,
                heap_ids,
                recursion + 1,
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Chunk index (types 10/11) — original implementation
// ---------------------------------------------------------------------------

/// A located chunk in a chunked dataset, returned from B-tree v2 traversal.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    /// Byte offset of this chunk's data within the file.
    pub address: u64,
    /// Compressed (or raw) size of this chunk's data in bytes.
    pub size: u32,
    /// Bitmask of disabled filters (0 = all filters active).
    pub filter_mask: u32,
    /// N-dimensional chunk offset in element units.
    pub offsets: Vec<u64>,
}

/// B-tree v2 (HDF5 1.10+) chunk index.
pub struct BTreeV2 {
    records: Vec<ChunkRecord>,
}

impl BTreeV2 {
    /// Parse a B-tree v2 chunk index rooted at `header_address`.
    ///
    /// `ndims` is the number of dataset dimensions (used to size the offset array).
    pub fn parse(file_data: &[u8], header_address: u64, ndims: usize) -> Result<Self, OxiH5Error> {
        let base = header_address as usize;

        // --- B-tree v2 header ("BTHD") ---
        // Offset  Size  Field
        //  0       4     Signature "BTHD"
        //  4       1     Version
        //  5       1     Type (chunk record format selector)
        //  6       4     Node size (bytes in a node, including signature)
        // 10       2     Record size (bytes per record)
        // 12       2     Tree depth
        // 14       1     Split percent
        // 15       1     Merge percent
        // 16       8     Root node address
        // 24       2     Num records in root node
        // 26       8     Total records in tree (all levels)
        // 34       4     Checksum
        // Total: 38 bytes

        let hdr_end = base
            .checked_add(38)
            .ok_or_else(|| OxiH5Error::Format("BTHD: header address overflow".into()))?;
        if hdr_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "BTHD: header at {:#x} exceeds file length {}",
                base,
                file_data.len()
            )));
        }

        let sig = &file_data[base..base + 4];
        if sig != b"BTHD" {
            return Err(OxiH5Error::Format(format!(
                "BTHD: bad signature {sig:?} at {base:#x}"
            )));
        }

        let version = file_data[base + 4];
        if version != 0 {
            return Err(OxiH5Error::Format(format!(
                "BTHD: unsupported version {version}"
            )));
        }

        let btree_type = file_data[base + 5];
        // We support type 10 (non-filtered) and type 11 (filtered) chunk records.
        if btree_type != 10 && btree_type != 11 {
            return Err(OxiH5Error::NotImplemented(format!(
                "BTHD: unsupported record type {btree_type} (expected 10 or 11)"
            )));
        }

        let _node_size = read_u32_le(file_data, base + 6)?;
        let record_size = read_u16_le(file_data, base + 10)?;
        let tree_depth = read_u16_le(file_data, base + 12)?;

        if tree_depth > MAX_DEPTH {
            return Err(OxiH5Error::Format(format!(
                "BTHD: tree depth {tree_depth} exceeds maximum {MAX_DEPTH}"
            )));
        }

        let root_addr = read_u64_le(file_data, base + 16)?;
        let root_nrecords = read_u16_le(file_data, base + 24)?;

        if root_addr == UNDEF {
            // Empty tree — no chunks.
            return Ok(Self {
                records: Vec::new(),
            });
        }

        let mut records = Vec::new();
        parse_node(
            file_data,
            root_addr,
            tree_depth,
            root_nrecords,
            record_size,
            btree_type,
            ndims,
            &mut records,
            0,
        )?;

        Ok(Self { records })
    }

    /// Return all chunk records collected from this tree.
    pub fn records(&self) -> &[ChunkRecord] {
        &self.records
    }
}

// ---------------------------------------------------------------------------
// Internal recursive node parser
// ---------------------------------------------------------------------------

/// Parse one B-tree v2 node (internal or leaf) and accumulate chunk records.
///
/// `depth == 0` means this is a leaf node ("BTLF").
/// `depth > 0`  means this is an internal node ("BTIN").
#[allow(clippy::too_many_arguments)]
fn parse_node(
    file_data: &[u8],
    node_addr: u64,
    depth: u16,
    num_records: u16,
    record_size: u16,
    btree_type: u8,
    ndims: usize,
    records: &mut Vec<ChunkRecord>,
    recursion: u16,
) -> Result<(), OxiH5Error> {
    if recursion > MAX_DEPTH {
        return Err(OxiH5Error::Format(
            "BTreeV2: recursion limit reached".into(),
        ));
    }
    if node_addr == UNDEF {
        return Ok(());
    }

    let base = node_addr as usize;

    // Both BTIN and BTLF start with: signature(4) + version(1) + type(1) = 6 bytes.
    let sig_end = base
        .checked_add(6)
        .ok_or_else(|| OxiH5Error::Format("BTreeV2 node: address overflow".into()))?;
    if sig_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "BTreeV2 node at {:#x}: truncated",
            base
        )));
    }

    let expected_sig: &[u8] = if depth == 0 { b"BTLF" } else { b"BTIN" };
    let sig = &file_data[base..base + 4];
    if sig != expected_sig {
        return Err(OxiH5Error::Format(format!(
            "BTreeV2 node at {base:#x} depth={depth}: expected {expected_sig:?}, got {sig:?}"
        )));
    }

    let node_version = file_data[base + 4];
    if node_version != 0 {
        return Err(OxiH5Error::Format(format!(
            "BTreeV2 node: unsupported version {node_version}"
        )));
    }

    // Records start at offset 6 in the node.
    let records_start = base + 6;
    let record_count = num_records as usize;
    let rs = record_size as usize;

    if depth == 0 {
        // ---------- Leaf node (BTLF) ----------
        // Layout: sig(4) + ver(1) + type(1) + records(num_records * record_size) + checksum(4)
        let records_end = records_start
            .checked_add(record_count * rs)
            .ok_or_else(|| OxiH5Error::Format("BTLF: record range overflow".into()))?;
        if records_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "BTLF at {base:#x}: record data truncated (need {records_end}, have {})",
                file_data.len()
            )));
        }

        for i in 0..record_count {
            let r_off = records_start + i * rs;
            let rec = parse_chunk_record(&file_data[r_off..r_off + rs], btree_type, ndims)?;
            records.push(rec);
        }
    } else {
        // ---------- Internal node (BTIN) ----------
        // Layout: sig(4) + ver(1) + type(1)
        //       + records(num_records * record_size)
        //       + child_pointers((num_records+1) * (8 + child_nrecords_size))
        //       + checksum(4)
        //
        // child_nrecords_size: enough bytes to hold max records per child.
        // For simplicity we handle the common case where the child record count
        // fits in a u16 (2 bytes).  HDF5 spec says it varies but 2 bytes covers
        // most practical files.
        let child_count = record_count + 1;

        // Each child pointer: address(8) + num_records_in_child(variable).
        // We'll attempt to read each child's record count as u16 LE.
        let child_ptr_size = 8 + 2; // address + 2-byte record count (common case)

        let records_end = records_start
            .checked_add(record_count * rs)
            .ok_or_else(|| OxiH5Error::Format("BTIN: record range overflow".into()))?;
        let children_end = records_end
            .checked_add(child_count * child_ptr_size)
            .ok_or_else(|| OxiH5Error::Format("BTIN: children range overflow".into()))?;

        if children_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "BTIN at {base:#x}: node data truncated (need {children_end}, have {})",
                file_data.len()
            )));
        }

        // Parse child pointers.
        for c in 0..child_count {
            let ptr_off = records_end + c * child_ptr_size;
            let child_addr = read_u64_le(file_data, ptr_off)?;
            let child_nrecords = read_u16_le(file_data, ptr_off + 8)?;

            if child_addr == UNDEF {
                continue;
            }

            parse_node(
                file_data,
                child_addr,
                depth - 1,
                child_nrecords,
                record_size,
                btree_type,
                ndims,
                records,
                recursion + 1,
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Record parsing
// ---------------------------------------------------------------------------

/// Parse a single chunk record from a slice of exactly `record_size` bytes.
///
/// Both type 10 (non-filtered) and type 11 (filtered) share the layout:
/// ```text
/// address     : 8 bytes
/// chunk_size  : 4 bytes
/// filter_mask : 4 bytes   (= 0 for non-filtered records unless type 11)
/// offsets     : record_size - 16 bytes, split into ndims parts
/// ```
fn parse_chunk_record(rec: &[u8], btree_type: u8, ndims: usize) -> Result<ChunkRecord, OxiH5Error> {
    if rec.len() < 16 {
        return Err(OxiH5Error::Format(format!(
            "chunk record: too short ({} bytes, need at least 16)",
            rec.len()
        )));
    }

    let address = u64::from_le_bytes(
        rec[0..8]
            .try_into()
            .map_err(|_| OxiH5Error::Format("chunk record: address slice".into()))?,
    );
    let size = u32::from_le_bytes(
        rec[8..12]
            .try_into()
            .map_err(|_| OxiH5Error::Format("chunk record: size slice".into()))?,
    );

    // Type 10 (non-filtered) has no filter_mask field per se, but the layout
    // in the HDF5 spec actually still has 4 bytes here; type 11 has it too.
    let filter_mask = if btree_type == 11 {
        u32::from_le_bytes(
            rec[12..16]
                .try_into()
                .map_err(|_| OxiH5Error::Format("chunk record: filter_mask slice".into()))?,
        )
    } else {
        0u32
    };

    // Remaining bytes encode the chunk offsets.
    let offset_bytes = rec.len() - 16;
    let offsets = if ndims == 0 || offset_bytes == 0 {
        Vec::new()
    } else {
        let bytes_per_dim = offset_bytes / ndims;
        if bytes_per_dim == 0 || offset_bytes % ndims != 0 {
            return Err(OxiH5Error::Format(format!(
                "chunk record: offset bytes {offset_bytes} not divisible by ndims {ndims}"
            )));
        }

        let mut offs = Vec::with_capacity(ndims);
        for d in 0..ndims {
            let o = 16 + d * bytes_per_dim;
            let val = match bytes_per_dim {
                8 => u64::from_le_bytes(
                    rec[o..o + 8]
                        .try_into()
                        .map_err(|_| OxiH5Error::Format("chunk record: offset u64".into()))?,
                ),
                4 => u32::from_le_bytes(
                    rec[o..o + 4]
                        .try_into()
                        .map_err(|_| OxiH5Error::Format("chunk record: offset u32".into()))?,
                ) as u64,
                2 => u16::from_le_bytes(
                    rec[o..o + 2]
                        .try_into()
                        .map_err(|_| OxiH5Error::Format("chunk record: offset u16".into()))?,
                ) as u64,
                1 => rec[o] as u64,
                other => {
                    return Err(OxiH5Error::Format(format!(
                        "chunk record: unsupported bytes_per_dim {other}"
                    )))
                }
            };
            offs.push(val);
        }
        offs
    };

    Ok(ChunkRecord {
        address,
        size,
        filter_mask,
        offsets,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal single-leaf B-tree v2 in memory and verify we can
    /// parse the one chunk record it contains.
    #[test]
    fn test_btree_v2_single_leaf() {
        // One 1-D chunk at file offset 0x1000, size 256, filter_mask 0, offset 0.
        // record_size = 8 (addr) + 4 (size) + 4 (filter_mask) + 8 (offset) = 24 bytes
        // type = 11 (filtered)

        let record_size: u16 = 24;
        let ndims: usize = 1;

        // Build the header at position 0.
        let mut buf = vec![0u8; 512];
        let hdr = 0usize;
        buf[hdr..hdr + 4].copy_from_slice(b"BTHD");
        buf[hdr + 4] = 0; // version
        buf[hdr + 5] = 11; // type = filtered chunk
        buf[hdr + 6..hdr + 10].copy_from_slice(&100u32.to_le_bytes()); // node_size (dummy)
        buf[hdr + 10..hdr + 12].copy_from_slice(&record_size.to_le_bytes());
        buf[hdr + 12..hdr + 14].copy_from_slice(&0u16.to_le_bytes()); // depth = 0 (leaf)
        buf[hdr + 14] = 75; // split percent
        buf[hdr + 15] = 25; // merge percent
                            // root_addr at hdr+16: point to position 64 (leaf node)
        let leaf_pos: u64 = 64;
        buf[hdr + 16..hdr + 24].copy_from_slice(&leaf_pos.to_le_bytes());
        buf[hdr + 24..hdr + 26].copy_from_slice(&1u16.to_le_bytes()); // 1 record in root
                                                                      // total_records at hdr+26 (8 bytes)
        buf[hdr + 26..hdr + 34].copy_from_slice(&1u64.to_le_bytes());
        // checksum at hdr+34 (4 bytes) — ignored in parsing
        buf[hdr + 34..hdr + 38].copy_from_slice(&0u32.to_le_bytes());

        // Build the leaf node at position 64.
        let leaf = leaf_pos as usize;
        buf[leaf..leaf + 4].copy_from_slice(b"BTLF");
        buf[leaf + 4] = 0; // version
        buf[leaf + 5] = 11; // type
                            // Record at leaf+6:
        let r = leaf + 6;
        let chunk_addr: u64 = 0x1000;
        buf[r..r + 8].copy_from_slice(&chunk_addr.to_le_bytes()); // address
        buf[r + 8..r + 12].copy_from_slice(&256u32.to_le_bytes()); // size
        buf[r + 12..r + 16].copy_from_slice(&0u32.to_le_bytes()); // filter_mask
        buf[r + 16..r + 24].copy_from_slice(&0u64.to_le_bytes()); // offset[0] = 0
                                                                  // checksum after records (not validated, just needs to be present)
        buf[r + 24..r + 28].copy_from_slice(&0u32.to_le_bytes());

        let tree = BTreeV2::parse(&buf, 0, ndims).expect("parse failed");
        assert_eq!(tree.records().len(), 1);
        let rec = &tree.records()[0];
        assert_eq!(rec.address, 0x1000);
        assert_eq!(rec.size, 256);
        assert_eq!(rec.filter_mask, 0);
        assert_eq!(rec.offsets, vec![0u64]);
    }

    #[test]
    fn test_btree_v2_empty_undefined_root() {
        // A header pointing to UNDEF root should yield zero records.
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(b"BTHD");
        buf[4] = 0; // version
        buf[5] = 10; // type = non-filtered
        buf[6..10].copy_from_slice(&64u32.to_le_bytes()); // node_size
        buf[10..12].copy_from_slice(&24u16.to_le_bytes()); // record_size
        buf[12..14].copy_from_slice(&0u16.to_le_bytes()); // depth
        buf[14] = 75;
        buf[15] = 25;
        // root_addr = UNDEF
        buf[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[24..26].copy_from_slice(&0u16.to_le_bytes());
        buf[26..34].copy_from_slice(&0u64.to_le_bytes());
        buf[34..38].copy_from_slice(&0u32.to_le_bytes());

        let tree = BTreeV2::parse(&buf, 0, 1).expect("parse failed");
        assert!(tree.records().is_empty());
    }

    #[test]
    fn test_bad_signature_rejected() {
        let buf = vec![0u8; 64];
        // Starts with all zeros — "BTHD" not present.
        let result = BTreeV2::parse(&buf, 0, 1);
        assert!(result.is_err());
    }

    /// Build a minimal BTHD+BTLF with 2 type-5 records (heap_id_len=8:
    /// 4-byte hash + 8-byte heap_id) and verify parse_name_index returns
    /// both heap IDs with correct bytes.
    #[test]
    fn test_btree_v2_name_index_single_leaf() {
        // type 5, heap_id_len = 8 → record_size = 4 + 8 = 12 bytes
        let heap_id_len: u8 = 8;
        let record_size: u16 = 12; // 4 hash + 8 id

        // Leaf node position in the buffer.
        let leaf_pos: u64 = 64;

        let mut buf = vec![0u8; 256];

        // ----- BTHD header at offset 0 -----
        buf[0..4].copy_from_slice(b"BTHD");
        buf[4] = 0; // version
        buf[5] = 5; // type = group name index
        buf[6..10].copy_from_slice(&128u32.to_le_bytes()); // node_size (dummy)
        buf[10..12].copy_from_slice(&record_size.to_le_bytes()); // record_size = 12
        buf[12..14].copy_from_slice(&0u16.to_le_bytes()); // depth = 0 (leaf)
        buf[14] = 75; // split percent
        buf[15] = 25; // merge percent
        buf[16..24].copy_from_slice(&leaf_pos.to_le_bytes()); // root_addr
        buf[24..26].copy_from_slice(&2u16.to_le_bytes()); // 2 records in root
        buf[26..34].copy_from_slice(&2u64.to_le_bytes()); // total records
        buf[34..38].copy_from_slice(&0u32.to_le_bytes()); // checksum (ignored)

        // ----- BTLF leaf at offset 64 -----
        let leaf = leaf_pos as usize;
        buf[leaf..leaf + 4].copy_from_slice(b"BTLF");
        buf[leaf + 4] = 0; // version
        buf[leaf + 5] = 5; // type

        // Record 0: 4-byte hash + 8-byte heap_id
        let r0 = leaf + 6;
        buf[r0..r0 + 4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes()); // hash (ignored)
        let id0: [u8; 8] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        buf[r0 + 4..r0 + 12].copy_from_slice(&id0);

        // Record 1: 4-byte hash + 8-byte heap_id
        let r1 = r0 + 12;
        buf[r1..r1 + 4].copy_from_slice(&0xCAFE_BABE_u32.to_le_bytes()); // hash (ignored)
        let id1: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        buf[r1 + 4..r1 + 12].copy_from_slice(&id1);

        let ids = parse_name_index(&buf, 0, heap_id_len).expect("parse_name_index failed");
        assert_eq!(ids.len(), 2, "expected 2 heap IDs, got {}", ids.len());
        assert_eq!(ids[0], id0.to_vec(), "first heap_id mismatch");
        assert_eq!(ids[1], id1.to_vec(), "second heap_id mismatch");
    }
}
