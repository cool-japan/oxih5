use crate::superblock::{read_u16_le, read_u64_le};
use oxih5_core::OxiH5Error;

/// Sentinel value for "undefined" addresses in HDF5.
const UNDEFINED_ADDR: u64 = u64::MAX;

/// All SNOD leaf node addresses collected from a B-tree v1 group subtree.
pub struct BTreeV1 {
    pub leaf_addresses: Vec<u64>,
}

/// Parse a B-tree v1 group node rooted at `btree_address`, collecting all leaf
/// (SNOD) node addresses.
pub fn parse(file_data: &[u8], btree_address: u64) -> Result<BTreeV1, OxiH5Error> {
    let mut leaf_addresses = Vec::new();
    collect_leaves(file_data, btree_address, &mut leaf_addresses, 0)?;
    Ok(BTreeV1 { leaf_addresses })
}

/// Recursively (or iteratively for level-0 nodes) collect SNOD addresses from
/// a B-tree v1 subtree.
///
/// B-tree v1 node layout:
/// ```text
/// Offset  Size  Field
///  0       4     Signature "TREE"
///  4       1     Node type (0 = group)
///  5       1     Node level (0 = leaf, >0 = internal)
///  6       2     Entries used K (u16 LE)
///  8       8     Left sibling address (u64 LE)
/// 16       8     Right sibling address (u64 LE)
/// 24       …     Keys and children interleaved:
///                  key[0](8), child[0](8), key[1](8), child[1](8), …, key[K](8)
///                  Total: K children, K+1 keys
/// ```
/// Children are at: 24 + 8 (first key) + i*16 for i in 0..K
fn collect_leaves(
    file_data: &[u8],
    node_address: u64,
    leaves: &mut Vec<u64>,
    depth: usize,
) -> Result<(), OxiH5Error> {
    // Guard against infinite recursion in pathological files.
    if depth > 64 {
        return Err(OxiH5Error::Format(
            "B-tree depth exceeds 64 (possible cycle)".to_string(),
        ));
    }

    if node_address == UNDEFINED_ADDR {
        return Ok(());
    }

    let off = node_address as usize;
    if off + 24 > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "B-tree node at {node_address}: out of bounds (file len={})",
            file_data.len()
        )));
    }

    if &file_data[off..off + 4] != b"TREE" {
        return Err(OxiH5Error::Format(format!(
            "no TREE signature at {node_address}: got {:?}",
            &file_data[off..off + 4]
        )));
    }

    let node_type = file_data[off + 4];
    if node_type != 0 {
        return Err(OxiH5Error::Format(format!(
            "expected group B-tree (type 0), got type {node_type}"
        )));
    }

    let level = file_data[off + 5];
    let entries_used = read_u16_le(file_data, off + 6)? as usize;

    // Children are interleaved with keys:
    //   key[0] at off+24, child[0] at off+32, key[1] at off+40, child[1] at off+48, …
    for i in 0..entries_used {
        // child[i] is at offset 24 + 8 (first key) + i*(8 key + 8 child)
        let child_field_offset = off + 24 + 8 + i * 16;
        if child_field_offset + 8 > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "B-tree child[{i}] at {child_field_offset}: out of bounds"
            )));
        }
        let child_addr = read_u64_le(file_data, child_field_offset)?;

        if level == 0 {
            // Leaf node: child is a SNOD address.
            if child_addr != UNDEFINED_ADDR {
                leaves.push(child_addr);
            }
        } else {
            // Internal node: recurse into the child subtree.
            collect_leaves(file_data, child_addr, leaves, depth + 1)?;
        }
    }

    Ok(())
}
