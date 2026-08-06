//! B-tree v1 **raw-data-chunk** index (node type 1).
//!
//! Chunked datasets written with `libver='earliest'` (the h5py / HDF5 default
//! for most files) index their chunks with a *version-1* B-tree whose node type
//! is `1` (raw data chunks), as opposed to the type-`0` group B-tree handled by
//! [`crate::btree`].
//!
//! Node layout (`TREE` node, type 1):
//! ```text
//! Offset  Size  Field
//!  0       4     Signature "TREE"
//!  4       1     Node type (1 = raw data chunks)
//!  5       1     Node level (0 = leaf, >0 = internal)
//!  6       2     Entries used K (u16 LE)
//!  8       8     Left sibling address  (u64 LE)
//! 16       8     Right sibling address (u64 LE)
//! 24       …     K+1 keys interleaved with K child pointers:
//!                   key[0] child[0] key[1] child[1] … key[K-1] child[K-1] key[K]
//! ```
//! Each *chunk key* (raw-data B-tree) is:
//! ```text
//!  0       4     Size of chunk in bytes on disk (u32 LE)
//!  4       4     Filter mask (u32 LE)  — bit i set ⇒ filter i skipped
//!  8       8×(D+1)  Chunk offset per dimension (u64 LE each); the final
//!                   "extra" dimension is the element offset and is always 0.
//! ```
//! A *child* pointer is an 8-byte file address.  At level 0 it points to the
//! raw chunk bytes; above level 0 it points to a sub-node of the same type.

use crate::btree_v2::ChunkRecord;
use oxih5_core::OxiH5Error;

/// Undefined-address sentinel.
const UNDEF: u64 = u64::MAX;

/// Maximum recursion depth, guarding against cyclic / corrupt files.
const MAX_DEPTH: usize = 64;

/// Parse a B-tree v1 raw-data-chunk index rooted at `btree_address`.
///
/// `ndims` is the dataset rank (the number of *real* dimensions).  The on-disk
/// key carries `ndims + 1` offset values; the extra trailing value is ignored.
///
/// Returns every leaf [`ChunkRecord`] (address, on-disk size, filter mask and
/// per-dimension offset in *elements*).
pub fn parse(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    let mut records = Vec::new();
    collect(file_data, btree_address, ndims, &mut records, 0)?;
    Ok(records)
}

fn collect(
    file_data: &[u8],
    node_address: u64,
    ndims: usize,
    out: &mut Vec<ChunkRecord>,
    depth: usize,
) -> Result<(), OxiH5Error> {
    if node_address == UNDEF {
        return Ok(());
    }
    if depth > MAX_DEPTH {
        return Err(OxiH5Error::Format(
            "chunk B-tree v1 depth exceeds 64 (possible cycle)".into(),
        ));
    }

    let off = node_address as usize;
    // `node_address` is a raw on-disk file address — read straight from a
    // sibling/child pointer (see the `read_u64(file_data, child_off)` call
    // below) with no upstream validation, so it can be any `u64`, including
    // values near `u64::MAX`. In a release build, plain `off + 24` silently
    // wraps on overflow instead of panicking; a wrapped sum can come out
    // *smaller* than `file_data.len()`, which would make the bounds check
    // below pass even though `off` itself is far out of range. The direct
    // slice a few lines down (`file_data[off..off + 4]`) would then panic
    // with a "range start index ... out of range" instead of this function
    // returning its typed error — confirmed by fuzzing (`fuzz_btree_v1_chunk`).
    // `checked_add` closes that: an overflow is rejected here, before any
    // indexing happens.
    let node_end = off.checked_add(24).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "chunk B-tree node address {node_address:#x} overflows a file offset"
        ))
    })?;
    if node_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "chunk B-tree node at {node_address:#x}: out of bounds (file len={})",
            file_data.len()
        )));
    }
    if &file_data[off..off + 4] != b"TREE" {
        return Err(OxiH5Error::Format(format!(
            "chunk B-tree: no TREE signature at {node_address:#x}: got {:?}",
            &file_data[off..off + 4]
        )));
    }
    let node_type = file_data[off + 4];
    if node_type != 1 {
        return Err(OxiH5Error::Format(format!(
            "chunk B-tree: expected node type 1 (raw chunks), got {node_type}"
        )));
    }
    let level = file_data[off + 5];
    let entries_used = u16::from_le_bytes([file_data[off + 6], file_data[off + 7]]) as usize;

    // A chunk key is: size(4) + filter_mask(4) + (ndims+1)*8 offset bytes.
    // `ndims` is bounded to at most 255 by its single-byte on-disk encoding
    // (`parse_dataspace`'s `dimensionality: u8`), so `key_size` itself cannot
    // overflow; the per-entry offsets built from it below still need
    // checked arithmetic because `entries_used` (attacker-controlled, up to
    // `u16::MAX`) multiplies it.
    let key_size = 8 + (ndims + 1) * 8;
    let child_size = 8usize;
    let stride = key_size + child_size;

    // Keys and children are interleaved starting at off+24:
    //   key[0] child[0] key[1] child[1] … key[K-1] child[K-1] key[K]
    // child[i] sits immediately after key[i]:
    //   child[i] @ off + 24 + (i+1)*key_size + i*child_size
    for i in 0..entries_used {
        // Same wrap-then-pass-bounds-check hazard as `node_end` above: `i`
        // and `stride` are both attacker-influenced (via `entries_used` and
        // `ndims`), so build `key_off`/`child_off` with checked arithmetic.
        let key_off = i
            .checked_mul(stride)
            .and_then(|p| p.checked_add(node_end))
            .ok_or_else(|| OxiH5Error::Format(format!("chunk B-tree key[{i}] offset overflows")))?;
        let child_off = key_off.checked_add(key_size).ok_or_else(|| {
            OxiH5Error::Format(format!("chunk B-tree child[{i}] offset overflows"))
        })?;
        let child_end = child_off.checked_add(child_size).ok_or_else(|| {
            OxiH5Error::Format(format!("chunk B-tree child[{i}] offset overflows"))
        })?;
        if child_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "chunk B-tree child[{i}] at {child_off:#x}: out of bounds"
            )));
        }

        let child_addr = read_u64(file_data, child_off)?;

        if level == 0 {
            // Leaf: the key describes a chunk and the child points to its bytes.
            let record = parse_chunk_key(file_data, key_off, ndims, child_addr)?;
            if record.address != UNDEF {
                out.push(record);
            }
        } else {
            // Internal node: recurse into the sub-tree.
            collect(file_data, child_addr, ndims, out, depth + 1)?;
        }
    }

    Ok(())
}

/// Decode a single chunk key at `key_off` together with its data `address`.
fn parse_chunk_key(
    file_data: &[u8],
    key_off: usize,
    ndims: usize,
    address: u64,
) -> Result<ChunkRecord, OxiH5Error> {
    // `key_off` is derived from `collect`'s already-`checked_add`ed
    // arithmetic, but treat it as untrusted here too — `parse_chunk_key` is
    // a free function callable with any `key_off`, and a future caller must
    // not be able to reintroduce the same wrap-then-pass-bounds-check
    // hazard fixed in `collect`.
    let header_end = key_off
        .checked_add(8)
        .ok_or_else(|| OxiH5Error::Format("chunk B-tree: key offset overflows".into()))?;
    if header_end > file_data.len() {
        return Err(OxiH5Error::Format(
            "chunk B-tree: key truncated (size/mask)".into(),
        ));
    }
    let size = read_u32(file_data, key_off)?;
    let filter_mask = read_u32(file_data, key_off + 4)?;

    // Read `ndims` real offsets (skip the trailing element-offset dimension).
    // `ndims` itself is bounded to at most 255 (see `collect`), so
    // `Vec::with_capacity(ndims)` is safe; `header_end + d * 8` still needs
    // checked arithmetic since it is built from `key_off`.
    let mut offsets = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let o = d
            .checked_mul(8)
            .and_then(|p| p.checked_add(header_end))
            .ok_or_else(|| {
                OxiH5Error::Format(format!("chunk B-tree: offset[{d}] position overflows"))
            })?;
        let o_end = o.checked_add(8).ok_or_else(|| {
            OxiH5Error::Format(format!("chunk B-tree: offset[{d}] position overflows"))
        })?;
        if o_end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "chunk B-tree: offset[{d}] truncated"
            )));
        }
        offsets.push(read_u64(file_data, o)?);
    }

    Ok(ChunkRecord {
        address,
        size,
        filter_mask,
        offsets,
    })
}

#[inline]
fn read_u32(data: &[u8], off: usize) -> Result<u32, OxiH5Error> {
    data.get(off..off + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| OxiH5Error::Format(format!("read_u32 out of bounds at {off}")))
}

#[inline]
fn read_u64(data: &[u8], off: usize) -> Result<u64, OxiH5Error> {
    data.get(off..off + 8)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| OxiH5Error::Format(format!("read_u64 out of bounds at {off}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-level (all-leaf) B-tree v1 chunk node with the given
    /// chunk keys, returning the file buffer.  `chunks` is a list of
    /// (chunk_size, filter_mask, offsets, data_address).
    fn build_leaf_node(ndims: usize, chunks: &[(u32, u32, Vec<u64>, u64)]) -> (Vec<u8>, u64) {
        let key_size = 8 + (ndims + 1) * 8;
        let child_size = 8;
        let k = chunks.len();
        // node = 24 header + K*(key+child) + 1 trailing key
        let node_len = 24 + k * (key_size + child_size) + key_size;
        let node_addr = 0u64;
        let mut buf = vec![0u8; node_len.max(8)];

        buf[0..4].copy_from_slice(b"TREE");
        buf[4] = 1; // node type = raw data chunks
        buf[5] = 0; // level 0 (leaf)
        buf[6..8].copy_from_slice(&(k as u16).to_le_bytes());
        buf[8..16].copy_from_slice(&UNDEF.to_le_bytes()); // left sibling
        buf[16..24].copy_from_slice(&UNDEF.to_le_bytes()); // right sibling

        for (i, (size, mask, offsets, addr)) in chunks.iter().enumerate() {
            let key_off = 24 + i * (key_size + child_size);
            buf[key_off..key_off + 4].copy_from_slice(&size.to_le_bytes());
            buf[key_off + 4..key_off + 8].copy_from_slice(&mask.to_le_bytes());
            for (d, &v) in offsets.iter().enumerate() {
                let o = key_off + 8 + d * 8;
                buf[o..o + 8].copy_from_slice(&v.to_le_bytes());
            }
            // trailing extra dimension (element offset) left as 0
            let child_off = key_off + key_size;
            buf[child_off..child_off + 8].copy_from_slice(&addr.to_le_bytes());
        }
        (buf, node_addr)
    }

    #[test]
    fn test_single_chunk_leaf() {
        let (buf, addr) = build_leaf_node(1, &[(16, 0, vec![0], 0x1000)]);
        let recs = parse(&buf, addr, 1).expect("parse failed");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].address, 0x1000);
        assert_eq!(recs[0].size, 16);
        assert_eq!(recs[0].filter_mask, 0);
        assert_eq!(recs[0].offsets, vec![0]);
    }

    #[test]
    fn test_two_chunks_2d() {
        let (buf, addr) = build_leaf_node(
            2,
            &[(32, 0, vec![0, 0], 0x2000), (32, 1, vec![0, 4], 0x3000)],
        );
        let recs = parse(&buf, addr, 2).expect("parse failed");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].offsets, vec![0, 0]);
        assert_eq!(recs[1].offsets, vec![0, 4]);
        assert_eq!(recs[1].filter_mask, 1);
        assert_eq!(recs[1].address, 0x3000);
    }

    #[test]
    fn test_undef_root_is_empty() {
        let buf = vec![0u8; 8];
        let recs = parse(&buf, UNDEF, 1).expect("parse failed");
        assert!(recs.is_empty());
    }

    #[test]
    fn test_bad_signature_rejected() {
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(b"XXXX");
        assert!(parse(&buf, 0, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // Regression: a node/child address near `u64::MAX` must error, not panic.
    //
    // `collect`'s bounds check used to be plain `off + 24 > file_data.len()`.
    // In a release build that addition silently wraps on overflow instead of
    // panicking, so for `off` within 24 of `usize::MAX` the wrapped sum comes
    // out *smaller* than `file_data.len()` and the check falsely passes. The
    // very next line then slices `file_data[off..off + 4]` with the
    // still-huge `off`, which panics with a "range start index ... out of
    // range" message instead of this function returning its typed error.
    // Found by `fuzz/fuzz_targets/fuzz_btree_v1_chunk.rs`
    // (`crates/oxih5-format/src/btree_v1_chunk.rs:79` on a 161-byte input,
    // `off = 0xFFFFFFFFFFFFFFFC`).
    // -----------------------------------------------------------------------

    #[test]
    fn test_root_address_overflow_rejected_not_panic() {
        let buf = vec![0u8; 64];
        let evil_addr = u64::MAX - 3;
        let result = parse(&buf, evil_addr, 1);
        assert!(
            result.is_err(),
            "an overflowing root node address must return an error, not panic"
        );
    }

    /// Same hazard, but reached the way the fuzzer actually found it: an
    /// internal node's on-disk child pointer (read via `read_u64`, no
    /// upstream validation) is the huge value, not the address `parse` is
    /// called with directly.
    #[test]
    fn test_internal_node_child_address_overflow_rejected_not_panic() {
        let ndims = 1;
        let key_size = 8 + (ndims + 1) * 8;
        let child_size = 8;
        let root_addr = 0usize;

        let mut buf = vec![0u8; 64];
        buf[root_addr..root_addr + 4].copy_from_slice(b"TREE");
        buf[root_addr + 4] = 1; // node type = raw data chunks
        buf[root_addr + 5] = 1; // level 1 (internal)
        buf[root_addr + 6..root_addr + 8].copy_from_slice(&1u16.to_le_bytes()); // 1 entry
        buf[root_addr + 8..root_addr + 16].copy_from_slice(&UNDEF.to_le_bytes());
        buf[root_addr + 16..root_addr + 24].copy_from_slice(&UNDEF.to_le_bytes());

        // child[0] — an attacker-controlled on-disk pointer near `u64::MAX`.
        let c0 = root_addr + 24 + key_size;
        let evil_child_addr = u64::MAX - 3;
        buf[c0..c0 + 8].copy_from_slice(&evil_child_addr.to_le_bytes());
        let _ = child_size; // documents the interleaving; not otherwise needed

        let result = parse(&buf, root_addr as u64, ndims);
        assert!(
            result.is_err(),
            "an overflowing child node address must return an error, not panic"
        );
    }

    /// Direct regression for `parse_chunk_key`'s own overflow guard: a
    /// `key_off` near `usize::MAX` must not panic either, independent of
    /// however `collect` derived it.
    #[test]
    fn test_parse_chunk_key_offset_overflow_rejected_not_panic() {
        let buf = vec![0u8; 32];
        let evil_key_off = usize::MAX - 2;
        let result = parse_chunk_key(&buf, evil_key_off, 2, 0x1000);
        assert!(
            result.is_err(),
            "an overflowing key offset must return an error, not panic"
        );
    }

    #[test]
    fn test_wrong_node_type_rejected() {
        let (mut buf, addr) = build_leaf_node(1, &[(16, 0, vec![0], 0x1000)]);
        buf[4] = 0; // group node type, not chunk
        assert!(parse(&buf, addr, 1).is_err());
    }

    #[test]
    fn test_internal_node_two_levels() {
        // Build two leaf nodes, then an internal node pointing to both.
        let ndims = 1;
        let key_size = 8 + (ndims + 1) * 8;
        let child_size = 8;

        // Leaf A at 0x100: one chunk at 0x500.
        let (leaf_a, _) = build_leaf_node(ndims, &[(8, 0, vec![0], 0x500)]);
        // Leaf B at 0x200: one chunk at 0x600.
        let (leaf_b, _) = build_leaf_node(ndims, &[(8, 0, vec![4], 0x600)]);

        let leaf_a_addr = 0x100usize;
        let leaf_b_addr = 0x200usize;
        let root_addr = 0x10usize;

        let mut buf = vec![0u8; 0x300];
        buf[leaf_a_addr..leaf_a_addr + leaf_a.len()].copy_from_slice(&leaf_a);
        buf[leaf_b_addr..leaf_b_addr + leaf_b.len()].copy_from_slice(&leaf_b);

        // Internal root: level 1, 2 entries.
        buf[root_addr..root_addr + 4].copy_from_slice(b"TREE");
        buf[root_addr + 4] = 1;
        buf[root_addr + 5] = 1; // level 1 (internal)
        buf[root_addr + 6..root_addr + 8].copy_from_slice(&2u16.to_le_bytes());
        buf[root_addr + 8..root_addr + 16].copy_from_slice(&UNDEF.to_le_bytes());
        buf[root_addr + 16..root_addr + 24].copy_from_slice(&UNDEF.to_le_bytes());

        // child[0] points to leaf A, child[1] to leaf B.
        let c0 = root_addr + 24 + key_size;
        buf[c0..c0 + 8].copy_from_slice(&(leaf_a_addr as u64).to_le_bytes());
        let c1 = root_addr + 24 + 2 * key_size + child_size;
        buf[c1..c1 + 8].copy_from_slice(&(leaf_b_addr as u64).to_le_bytes());

        let recs = parse(&buf, root_addr as u64, ndims).expect("parse failed");
        assert_eq!(recs.len(), 2);
        let addrs: Vec<u64> = recs.iter().map(|r| r.address).collect();
        assert!(addrs.contains(&0x500));
        assert!(addrs.contains(&0x600));
    }
}
