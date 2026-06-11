//! W0c: Chunked/unlimited dataset support — B-tree v1 type-1 writer.
//!
//! Writes the HDF5 B-tree v1 type-1 (raw data chunks) node for a single-chunk
//! dataset.  For unlimited dimensions the initial data is stored as one chunk,
//! with chunk key offsets = [0, 0, …, 0].

use super::format::{write_u16_le, write_u32_le, write_u64_le};

/// Compute the byte size of a single-chunk B-tree v1 type-1 node
/// for a dataset with `ndims` real dimensions.
///
/// Layout:
/// ```text
/// header (24) | key[0] (key_size) | child[0] (8) | key[1] (key_size)
/// key_size = 8 + (ndims+1)*8
/// ```
pub(super) fn chunk_btree_size(ndims: usize) -> usize {
    let key_size = 8 + (ndims + 1) * 8;
    24 + 2 * key_size + 8
}

/// Write a minimal B-tree v1 type-1 node (single chunk) at `base`.
///
/// The node has `entries_used = 1`.  Key[0] encodes the chunk's byte size and
/// zero offsets; child[0] points to the raw data at `data_addr`; key[1] is
/// all-zero (already zero from `vec![0u8; eof_addr]` initialization).
///
/// # Layout of each key
/// ```text
///  0  4   chunk_byte_size (u32 LE)
///  4  4   filter_mask     (u32 LE, always 0)
///  8  (ndims+1)*8   chunk offsets (u64 LE each, all zero for [0,0,…,0])
/// ```
pub(super) fn write_chunk_btree(
    buf: &mut [u8],
    base: usize,
    ndims: usize,
    data_addr: u64,
    data_byte_size: usize,
) {
    let key_size = 8 + (ndims + 1) * 8;

    // Node header
    buf[base..base + 4].copy_from_slice(b"TREE");
    buf[base + 4] = 1; // node type = 1 (raw data chunks)
    buf[base + 5] = 0; // level = 0 (leaf)
    write_u16_le(buf, base + 6, 1); // entries_used = 1
    write_u64_le(buf, base + 8, u64::MAX); // left sibling: undefined
    write_u64_le(buf, base + 16, u64::MAX); // right sibling: undefined

    // Key[0]: chunk byte size + filter mask (zero offsets already zero)
    let key0 = base + 24;
    write_u32_le(buf, key0, data_byte_size as u32);
    write_u32_le(buf, key0 + 4, 0); // filter_mask = 0
                                    // offsets: (ndims+1)*8 bytes — already zero

    // Child[0]: raw data address
    let child0 = key0 + key_size;
    write_u64_le(buf, child0, data_addr);

    // Key[1]: all zero (already zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_btree_size_1d() {
        // ndims=1: key_size = 8 + 2*8 = 24; total = 24 + 48 + 8 = 80
        assert_eq!(chunk_btree_size(1), 80);
    }

    #[test]
    fn chunk_btree_size_2d() {
        // ndims=2: key_size = 8 + 3*8 = 32; total = 24 + 64 + 8 = 96
        assert_eq!(chunk_btree_size(2), 96);
    }

    #[test]
    fn write_chunk_btree_roundtrip_signature() {
        let ndims = 1usize;
        let sz = chunk_btree_size(ndims);
        let mut buf = vec![0u8; sz];
        write_chunk_btree(&mut buf, 0, ndims, 0xABCD, 256);

        // Signature
        assert_eq!(&buf[0..4], b"TREE");
        // Node type = 1
        assert_eq!(buf[4], 1);
        // Level = 0
        assert_eq!(buf[5], 0);
        // entries_used = 1
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 1);
        // Left sibling = undefined
        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), u64::MAX);
        // Right sibling = undefined
        assert_eq!(
            u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            u64::MAX
        );
        // Key[0] chunk size = 256
        assert_eq!(u32::from_le_bytes(buf[24..28].try_into().unwrap()), 256u32);
        // Key[0] filter mask = 0
        assert_eq!(u32::from_le_bytes(buf[28..32].try_into().unwrap()), 0u32);
        // Child[0] = 0xABCD
        let key_size = 8 + (ndims + 1) * 8;
        let child_off = 24 + key_size;
        assert_eq!(
            u64::from_le_bytes(buf[child_off..child_off + 8].try_into().unwrap()),
            0xABCD
        );
    }
}
