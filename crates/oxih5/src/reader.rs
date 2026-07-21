//! Internal read helpers shared by [`crate::File`] and [`crate::Group`].
//!
//! Extracted from `lib.rs` to keep individual source files under the 2000-line
//! limit.  These functions resolve object-header addresses through old-style
//! (symbol-table B-tree) and new-style (Link message) groups, and assemble a
//! [`Dataset`] from the messages found in an object header.

use oxih5_core::{Attribute, Dataset, Dtype, OxiH5Error};
use oxih5_format::{header, message, ChunkIndexCache};

use crate::links::{
    lookup_child, read_virtual_dataset, resolve_path_target, ChildLink, GroupRef, SoftTarget,
    VisitedLinks,
};

/// Extract (btree_address, heap_address) from the first SymbolTable message
/// (0x0011) found in `messages`.
pub(crate) fn find_symbol_table_addresses(messages: &[header::Message]) -> Option<(u64, u64)> {
    for msg in messages {
        if msg.msg_type == 0x0011 {
            if let Ok(st) = message::parse_symbol_table(&msg.data) {
                return Some((st.btree_address, st.heap_address));
            }
        }
    }
    None
}

/// Navigate from `start` through a list of group-path segments and return the
/// final group.
///
/// Every segment must name a group; a segment that resolves to a dataset is
/// reported as `NotFound("'x' is not a group")`.  Soft links along the way are
/// followed, so a path may traverse an aliased group.
pub(crate) fn navigate_to_group(
    file_data: &[u8],
    start: GroupRef,
    segments: &[&str],
) -> Result<GroupRef, OxiH5Error> {
    let mut current = start;

    for segment in segments {
        let entry_addr = find_child_header(file_data, current, segment)?;
        let msgs = header::parse_messages(file_data, entry_addr)?;
        let symbol_table = find_symbol_table_addresses(&msgs);
        // An old-style group is identified by its Symbol Table message; a
        // new-style group by Link / Link Info messages.  Anything with neither
        // is a dataset (or a committed datatype), not a group.
        if symbol_table.is_none()
            && !msgs
                .iter()
                .any(|m| m.msg_type == 0x0006 || m.msg_type == 0x0002)
        {
            return Err(OxiH5Error::NotFound(format!("'{segment}' is not a group")));
        }
        current = GroupRef::new(entry_addr, symbol_table);
    }

    Ok(current)
}

/// Find a child object (dataset or group) by name in `parent`, returning its
/// object header address.
///
/// Works for groups of either storage style.  Soft links are followed (with a
/// cycle guard); external links have no address in this file and are reported
/// as `NotImplemented`.
pub(crate) fn find_child_header(
    file_data: &[u8],
    parent: GroupRef,
    name: &str,
) -> Result<u64, OxiH5Error> {
    match lookup_child(file_data, parent, name)? {
        ChildLink::Hard(address) => Ok(address),
        ChildLink::Soft(path) => {
            let mut visited = VisitedLinks::new();
            match resolve_path_target(file_data, parent, &path, &mut visited)? {
                SoftTarget::Local(address) => Ok(address),
                SoftTarget::External { file, path } => Err(OxiH5Error::NotImplemented(format!(
                    "'{name}' is a soft link to the external link '{path}' in file '{file}'; \
                     that target has no object-header address in this file"
                ))),
            }
        }
        ChildLink::External { file, .. } => Err(OxiH5Error::NotImplemented(format!(
            "external link '{name}' in file '{file}' not followed"
        ))),
    }
}

/// Resolve the object header address for `name` within `group`.
///
/// Identical to [`find_child_header`] except that an external link is reported
/// with the *resolved* filesystem path of its target file, which is what
/// callers such as in-place overwrite need in their diagnostics.
pub(crate) fn resolve_header_address_in(
    file_data: &[u8],
    group: GroupRef,
    name: &str,
    source_dir: &std::path::Path,
) -> Result<u64, OxiH5Error> {
    // An external link's target lives in a different file, so it has no address
    // in this one.  Report that rather than reading the target: callers of this
    // function want an offset, and loading the whole external dataset only to
    // discard it would be pure waste.
    let external_error = |ext_file: &str, ext_path: &str| {
        let resolved = crate::links::resolve_external_path(ext_file, source_dir);
        OxiH5Error::NotImplemented(format!(
            "'{name}' is an external link to '{ext_path}' in '{}'; that target has no \
             object-header address in this file",
            resolved.display()
        ))
    };

    match lookup_child(file_data, group, name)? {
        ChildLink::Hard(address) => Ok(address),
        // A soft link is an in-file path, so it normally still resolves to an
        // address in *this* file — follow it, with the usual cycle guard, so
        // that callers see the same objects `File::dataset` does.
        ChildLink::Soft(path) => {
            let mut visited = VisitedLinks::new();
            match resolve_path_target(file_data, group, &path, &mut visited)? {
                SoftTarget::Local(address) => Ok(address),
                SoftTarget::External {
                    file: ext_file,
                    path: ext_path,
                } => Err(external_error(&ext_file, &ext_path)),
            }
        }
        ChildLink::External {
            file: ext_file,
            path: ext_path,
        } => Err(external_error(&ext_file, &ext_path)),
    }
}

/// On-disk footprint (in bytes) of a single element of `dtype` when it is
/// stored inside a chunk or a hyperslab source region.
///
/// For fixed-size datatypes this is simply [`Dtype::size`].  For variable-length
/// datatypes (class 9 `VarLen` and variable-length strings, i.e.
/// `String { fixed_len: None, .. }`) the on-disk element is a fixed-size
/// *global-heap reference* — `length (4) + heap collection address
/// (size_of_offsets) + heap object index (4)` — so its footprint is fixed even
/// though [`Dtype::size`] returns `None`.  With the standard 8-byte offsets this
/// is 16 bytes, matching the reference layout decoded by
/// [`oxih5_format::values::decode_vlen_strings`] /
/// [`oxih5_format::values::decode_vlen_sequences`].
pub(crate) fn on_disk_elem_footprint(dtype: &Dtype, size_of_offsets: usize) -> Option<usize> {
    match dtype {
        Dtype::VarLen { .. }
        | Dtype::String {
            fixed_len: None, ..
        } => Some(4 + size_of_offsets + 4),
        other => other.size(),
    }
}

/// `true` when `dtype` is a variable-length type stored as a global-heap
/// reference on disk (vlen sequence or vlen string).
pub(crate) fn is_vlen_dtype(dtype: &Dtype) -> bool {
    matches!(
        dtype,
        Dtype::VarLen { .. }
            | Dtype::String {
                fixed_len: None,
                ..
            }
    )
}

/// Reject filter pipelines that make no sense combined with variable-length
/// elements.  Byte-oriented transform filters (shuffle id 2, fletcher32 id 3,
/// nbit id 5, scaleoffset id 6) operate on fixed-width numeric samples and
/// cannot be applied to the 16-byte global-heap references that back vlen data.
pub(crate) fn reject_vlen_incompatible_filters(
    pipeline: &oxih5_core::FilterPipeline,
    name: &str,
) -> Result<(), OxiH5Error> {
    for f in &pipeline.filters {
        if matches!(f.id, 2 | 5 | 6) {
            return Err(OxiH5Error::Format(format!(
                "dataset '{name}': filter id {} cannot be combined with variable-length elements",
                f.id
            )));
        }
    }
    Ok(())
}

/// Read a dataset directly from its object header address (new-style groups).
///
/// This bypasses the B-tree/SNOD lookup because the address was already
/// resolved from a Link message.
pub(crate) fn read_dataset_from_object_header(
    file_data: &[u8],
    header_addr: u64,
    name: &str,
    source_dir: &std::path::Path,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    let ds_messages = header::parse_messages(file_data, header_addr)?;

    let mut dataspace = None;
    let mut datatype = None;
    let mut layout = None;
    let mut filter_pipeline = None;

    for msg in &ds_messages {
        match msg.msg_type {
            0x0001 => dataspace = Some(message::parse_dataspace(&msg.data)?),
            0x0003 => datatype = Some(message::parse_datatype(&msg.data)?),
            0x0008 => layout = Some(message::parse_layout(&msg.data)?),
            0x000B => filter_pipeline = Some(message::parse_filter_pipeline(&msg.data)?),
            _ => {}
        }
    }

    let dsp = dataspace
        .ok_or_else(|| OxiH5Error::Format(format!("no dataspace message in dataset '{name}'")))?;
    let dtp = datatype
        .ok_or_else(|| OxiH5Error::Format(format!("no datatype message in dataset '{name}'")))?;
    let lay = layout
        .ok_or_else(|| OxiH5Error::Format(format!("no layout message in dataset '{name}'")))?;

    let shape: Vec<usize> = dsp.dims.iter().map(|&d| d as usize).collect();

    use oxih5_format::message::LayoutInfo;

    let raw = match &lay {
        LayoutInfo::Contiguous {
            data_address,
            data_size,
        } => {
            let data_off = *data_address as usize;
            let data_sz = *data_size as usize;
            if data_off + data_sz > file_data.len() {
                return Err(OxiH5Error::Format(format!(
                    "dataset '{name}': data at {data_off}+{data_sz} exceeds file size {}",
                    file_data.len()
                )));
            }
            file_data[data_off..data_off + data_sz].to_vec()
        }
        LayoutInfo::Compact { data } => data.clone(),
        LayoutInfo::Chunked { .. } => {
            let elem_size = on_disk_elem_footprint(&dtp.dtype, 8).ok_or_else(|| {
                OxiH5Error::NotImplemented(format!(
                    "chunked dataset '{name}': element size not supported"
                ))
            })?;
            let dataset_dims: Vec<u64> = dsp.dims.clone();
            let pipeline = filter_pipeline
                .clone()
                .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
            if is_vlen_dtype(&dtp.dtype) {
                reject_vlen_incompatible_filters(&pipeline, name)?;
            }
            oxih5_format::chunked::read_chunked(
                file_data,
                &lay,
                &pipeline,
                &dataset_dims,
                elem_size,
                None,
                cache,
            )?
        }
        LayoutInfo::VirtualDataset {
            heap_address,
            heap_index,
        } => read_virtual_dataset(
            file_data,
            *heap_address,
            *heap_index,
            &dsp,
            &dtp.dtype,
            source_dir,
            cache,
        )?,
    };

    let attributes = read_attributes_from_header(file_data, header_addr).unwrap_or_default();

    Ok(Dataset {
        data: raw,
        shape,
        dtype: dtp.dtype,
        attributes,
        max_dims: dsp.max_dims.clone(),
    })
}

/// Core dataset-reading logic: finds `name` within `group`, parses its object
/// header, and assembles a [`Dataset`].
///
/// Delegates to [`crate::links::resolve_dataset_in_group`] so that soft and
/// external links are followed exactly as they are for a direct
/// [`crate::File::dataset`] call.
pub(crate) fn read_dataset_from_group(
    file_data: &[u8],
    group: GroupRef,
    name: &str,
    source_dir: &std::path::Path,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    crate::links::resolve_dataset_in_group(file_data, group, name, source_dir, cache)
}

/// Parse all attribute messages (0x000C) from an object header and return
/// the decoded [`Attribute`] list.  Attributes that fail to parse are silently
/// skipped so that one malformed attribute does not abort the entire read.
pub(crate) fn read_attributes_from_header(
    file_data: &[u8],
    header_address: u64,
) -> Result<Vec<Attribute>, OxiH5Error> {
    let messages = header::parse_messages(file_data, header_address)?;
    let mut attrs = Vec::new();
    for msg in &messages {
        if msg.msg_type == 0x000C {
            if let Ok(attr) = message::parse_attribute(&msg.data) {
                attrs.push(attr);
            }
        }
    }
    Ok(attrs)
}
