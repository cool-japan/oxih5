use std::collections::HashSet;

use crate::btree_v2::parse_name_index;
use crate::context::ParseContext;
use crate::fractal_heap::FractalHeap;
use crate::link_msg::{parse_link, parse_link_info, ParsedLink};
use crate::{btree, heap, snod};
use oxih5_core::OxiH5Error;

/// List all non-empty dataset names in the root group by traversing the B-tree
/// and reading names from the local heap.
pub fn list_datasets(
    file_data: &[u8],
    btree_address: u64,
    heap_address: u64,
) -> Result<Vec<String>, OxiH5Error> {
    let local_heap = heap::parse(file_data, heap_address)?;
    let tree = btree::parse(file_data, btree_address)?;

    // T6: pre-size with a reasonable hint (8 entries per leaf is typical).
    let mut names = Vec::with_capacity(tree.leaf_addresses.len() * 8);
    for leaf_addr in &tree.leaf_addresses {
        let entries = snod::parse(file_data, *leaf_addr)?;
        for entry in entries {
            let name = local_heap.name_at(entry.name_offset as usize)?;
            if !name.is_empty() {
                names.push(name.to_string());
            }
        }
    }

    Ok(names)
}

/// Find a dataset by name and return its object-header address.
///
/// The `name` may be given with or without a leading `/`.
/// Only top-level (root group) datasets are searched.
pub fn find_dataset(
    file_data: &[u8],
    btree_address: u64,
    heap_address: u64,
    name: &str,
) -> Result<u64, OxiH5Error> {
    let local_heap = heap::parse(file_data, heap_address)?;
    let tree = btree::parse(file_data, btree_address)?;

    // Strip a leading "/" so callers can pass either "/temperature" or "temperature".
    let target = name.trim_start_matches('/');

    for leaf_addr in &tree.leaf_addresses {
        let entries = snod::parse(file_data, *leaf_addr)?;
        for entry in entries {
            let entry_name = local_heap.name_at(entry.name_offset as usize)?;
            if entry_name == target {
                return Ok(entry.object_header_address);
            }
        }
    }

    Err(OxiH5Error::NotFound(name.to_string()))
}

/// List all links in a new-style group.
///
/// Handles both small groups (links stored directly as Link messages 0x0006
/// in the object header) and large groups (links stored in a fractal heap
/// referenced by a Link Info message 0x0002).
///
/// When a Link Info message with valid `fractal_heap_address` and
/// `name_index_address` is present, those heap-backed links take priority over
/// any directly-stored Link messages with the same name.
pub fn list_new_style_links(
    file_data: &[u8],
    object_header_addr: u64,
    ctx: &ParseContext,
) -> Result<Vec<ParsedLink>, OxiH5Error> {
    let messages = crate::header::parse_messages(file_data, object_header_addr)?;

    // Collect directly-stored Link (0x0006) messages first.
    let mut header_links: Vec<ParsedLink> = Vec::new();
    let mut link_info = None;

    for msg in &messages {
        match msg.msg_type {
            0x0006 => {
                if let Ok(pl) = parse_link(&msg.data, ctx) {
                    header_links.push(pl);
                }
            }
            0x0002 => {
                if let Ok(info) = parse_link_info(&msg.data, ctx) {
                    link_info = Some(info);
                }
            }
            _ => {}
        }
    }

    // Check whether we have a fractal-heap-backed link list.
    let heap_links: Vec<ParsedLink> = if let Some(info) = link_info {
        match (info.fractal_heap_address, info.name_index_address) {
            (Some(fh_addr), Some(ni_addr)) => {
                // T8: pass a borrow directly — no Arc or to_vec() copy.
                let fh = FractalHeap::parse(file_data, fh_addr, ctx.size_of_offsets)?;
                let hil = fh.heap_id_len();
                let raw_ids = parse_name_index(file_data, ni_addr, hil)?;

                let mut links = Vec::with_capacity(raw_ids.len());
                for id in &raw_ids {
                    let (heap_offset, obj_size) = match fh.parse_heap_id(id) {
                        Ok(pair) => pair,
                        Err(_) => continue,
                    };
                    if obj_size == 0 {
                        continue;
                    }
                    let link_bytes = match fh.read_object(heap_offset, obj_size) {
                        Ok(bytes) => bytes,
                        Err(_) => continue,
                    };
                    if let Ok(pl) = parse_link(&link_bytes, ctx) {
                        links.push(pl);
                    }
                }
                links
            }
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    // Merge: heap links take priority; then add header links not already present.
    if heap_links.is_empty() {
        return Ok(header_links);
    }

    let mut seen: HashSet<String> = heap_links.iter().map(|pl| pl.name.clone()).collect();
    let mut result = heap_links;
    for pl in header_links {
        if seen.insert(pl.name.clone()) {
            result.push(pl);
        }
    }

    Ok(result)
}

/// Check if an object header represents a new-style group (has a Link Info
/// message 0x0002 or Link messages 0x0006, and NO Symbol Table 0x0011).
pub fn is_new_style_group(file_data: &[u8], object_header_addr: u64) -> bool {
    let Ok(messages) = crate::header::parse_messages(file_data, object_header_addr) else {
        return false;
    };
    let has_symbol_table = messages.iter().any(|m| m.msg_type == 0x0011);
    if has_symbol_table {
        return false;
    }
    messages
        .iter()
        .any(|m| m.msg_type == 0x0006 || m.msg_type == 0x0002)
}
