//! Link resolution: soft links, external links, and soft→external chains.
//!
//! Extracted from `lib.rs` to keep individual source files under the 2000-line
//! limit.  These helpers navigate new-style (Link-message) groups and open
//! external HDF5 files referenced by external links.

use super::*;

/// Resolve a soft-link target path to an object header address, starting from
/// `current_header_addr` (the root).
///
/// `visited` is a cycle guard: if `target_path` is already in the set the link
/// chain is cyclic and we return an error rather than looping infinitely.
pub(crate) fn resolve_soft_link_to_header(
    file_data: &[u8],
    root_header_addr: u64,
    target_path: &str,
    visited: &mut std::collections::HashSet<String>,
) -> Result<u64, OxiH5Error> {
    if !visited.insert(target_path.to_string()) {
        return Err(OxiH5Error::Format(format!(
            "soft link cycle detected at path '{target_path}'"
        )));
    }

    // Navigate from root, following each segment.
    let normalized = target_path.trim_start_matches('/');
    let parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();

    // Empty path → root itself.
    if parts.is_empty() {
        return Ok(root_header_addr);
    }

    let sb = superblock::parse(file_data)?;
    let ctx = oxih5_format::context::ParseContext::new(
        sb.size_of_offsets,
        sb.size_of_lengths,
        sb.base_address,
    );

    let mut current_header = root_header_addr;
    for (idx, segment) in parts.iter().enumerate() {
        let is_last = idx == parts.len() - 1;
        let links = group::list_new_style_links(file_data, current_header, &ctx)?;
        let mut found = false;
        for pl in &links {
            if pl.name == *segment {
                match &pl.link {
                    oxih5_core::Link::Hard { address } => {
                        current_header = *address;
                        found = true;
                        break;
                    }
                    oxih5_core::Link::Soft { path } => {
                        if is_last {
                            // Recurse into nested soft link with cycle guard.
                            return resolve_soft_link_to_header(
                                file_data,
                                root_header_addr,
                                path,
                                visited,
                            );
                        }
                        // Mid-path soft link: resolve it then continue.
                        let addr = resolve_soft_link_to_header(
                            file_data,
                            root_header_addr,
                            path,
                            visited,
                        )?;
                        current_header = addr;
                        found = true;
                        break;
                    }
                    oxih5_core::Link::External { .. } => {
                        return Err(OxiH5Error::NotImplemented(
                            "soft link targeting external link not supported".into(),
                        ));
                    }
                }
            }
        }
        if !found {
            return Err(OxiH5Error::NotFound(format!(
                "soft link target '{target_path}': segment '{segment}' not found"
            )));
        }
    }

    Ok(current_header)
}

/// The resolved target of a soft link.
pub(crate) enum SoftTarget {
    /// A hard object header address within the same file.
    Local(u64),
    /// The soft link ultimately points at an external link — the referenced
    /// object lives in another file.
    External { file: String, path: String },
}

/// Resolve a soft link, following it through hard links, nested soft links and,
/// crucially, a terminal **external** link (soft → external).
///
/// This mirrors [`resolve_soft_link_to_header`] but can report an external
/// target instead of failing with `NotImplemented`, so callers reading a
/// dataset can open the external file and continue.
pub(crate) fn resolve_soft_link_target(
    file_data: &[u8],
    root_header_addr: u64,
    target_path: &str,
    visited: &mut std::collections::HashSet<String>,
) -> Result<SoftTarget, OxiH5Error> {
    if !visited.insert(target_path.to_string()) {
        return Err(OxiH5Error::Format(format!(
            "soft link cycle detected at path '{target_path}'"
        )));
    }

    let normalized = target_path.trim_start_matches('/');
    let parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return Ok(SoftTarget::Local(root_header_addr));
    }

    let sb = superblock::parse(file_data)?;
    let ctx = oxih5_format::context::ParseContext::new(
        sb.size_of_offsets,
        sb.size_of_lengths,
        sb.base_address,
    );

    let mut current_header = root_header_addr;
    for (idx, segment) in parts.iter().enumerate() {
        let is_last = idx == parts.len() - 1;
        let links = group::list_new_style_links(file_data, current_header, &ctx)?;
        let mut found = false;
        for pl in &links {
            if pl.name == *segment {
                match &pl.link {
                    oxih5_core::Link::Hard { address } => {
                        current_header = *address;
                        found = true;
                        break;
                    }
                    oxih5_core::Link::Soft { path } => {
                        // Recurse for both mid-path and terminal soft links; the
                        // terminal case may itself resolve to an external target.
                        let target =
                            resolve_soft_link_target(file_data, root_header_addr, path, visited)?;
                        if is_last {
                            return Ok(target);
                        }
                        match target {
                            SoftTarget::Local(addr) => {
                                current_header = addr;
                                found = true;
                                break;
                            }
                            SoftTarget::External { .. } => {
                                return Err(OxiH5Error::NotImplemented(
                                    "soft link traverses an external link mid-path".into(),
                                ));
                            }
                        }
                    }
                    oxih5_core::Link::External { file, path } => {
                        if is_last {
                            return Ok(SoftTarget::External {
                                file: file.clone(),
                                path: path.clone(),
                            });
                        }
                        return Err(OxiH5Error::NotImplemented(
                            "soft link traverses an external link mid-path".into(),
                        ));
                    }
                }
            }
        }
        if !found {
            return Err(OxiH5Error::NotFound(format!(
                "soft link target '{target_path}': segment '{segment}' not found"
            )));
        }
    }

    Ok(SoftTarget::Local(current_header))
}

/// Resolve a dataset name within a new-style group, handling both hard links
/// and external file links.
///
/// For hard links the dataset is read from the local file at the resolved
/// object header address.  For external links the referenced file is opened
/// and `File::dataset` is called with the target path stored in the link.
/// Soft links and group-type external links return `NotImplemented`.
pub(crate) fn resolve_new_style_dataset(
    file_data: &[u8],
    parent_header_addr: u64,
    name: &str,
    source_dir: &std::path::Path,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    let sb = superblock::parse(file_data)?;
    let ctx = oxih5_format::context::ParseContext::new(
        sb.size_of_offsets,
        sb.size_of_lengths,
        sb.base_address,
    );
    let links = group::list_new_style_links(file_data, parent_header_addr, &ctx)?;
    for parsed_link in &links {
        if parsed_link.name == name {
            match &parsed_link.link {
                oxih5_core::Link::Hard { address } => {
                    return read_dataset_from_object_header(
                        file_data, *address, name, source_dir, cache,
                    );
                }
                oxih5_core::Link::Soft { path } => {
                    // Follow the soft link to its target, then read the dataset.
                    // The target may be local, or (soft → external) in another file.
                    let sb = superblock::parse(file_data)?;
                    let mut visited = std::collections::HashSet::new();
                    let target = resolve_soft_link_target(
                        file_data,
                        sb.root_object_header_address,
                        path,
                        &mut visited,
                    )?;
                    return match target {
                        SoftTarget::Local(addr) => read_dataset_from_object_header(
                            file_data, addr, name, source_dir, cache,
                        ),
                        SoftTarget::External {
                            file: ext_file,
                            path: ext_path,
                        } => resolve_external_link(&ext_file, &ext_path, source_dir),
                    };
                }
                oxih5_core::Link::External {
                    file: ext_file,
                    path: ext_path,
                } => {
                    return resolve_external_link(ext_file, ext_path, source_dir);
                }
            }
        }
    }
    Err(OxiH5Error::NotFound(name.to_string()))
}

/// Open an external HDF5 file and navigate to the dataset at `ext_path`.
///
/// `ext_file` is the filename from the external link (may be relative or
/// absolute).  `source_dir` is the directory of the file that contains the
/// link, used to resolve relative `ext_file` paths.
pub(crate) fn resolve_external_link(
    ext_file: &str,
    ext_path: &str,
    source_dir: &std::path::Path,
) -> Result<Dataset, OxiH5Error> {
    let resolved = resolve_external_path(ext_file, source_dir);

    let ext = open(&resolved).map_err(|e| {
        OxiH5Error::NotFound(format!(
            "external link target file '{}': {e}",
            resolved.display()
        ))
    })?;

    // Navigate to the target path within the external file.
    let target = ext_path.trim_start_matches('/');
    ext.dataset(target).map_err(|e| {
        OxiH5Error::NotFound(format!(
            "external link {}::{ext_path}: {e}",
            resolved.display()
        ))
    })
}

/// Open an external HDF5 file and navigate to the group at `ext_path`.
///
/// Returns the group handle from the external file.
pub(crate) fn resolve_external_link_group(
    ext_file: &str,
    ext_path: &str,
    source_dir: &std::path::Path,
) -> Result<Group, OxiH5Error> {
    let resolved = resolve_external_path(ext_file, source_dir);

    let ext = open(&resolved).map_err(|e| {
        OxiH5Error::NotFound(format!(
            "external link target file '{}': {e}",
            resolved.display()
        ))
    })?;

    let target = ext_path.trim_start_matches('/');
    ext.group(if target.is_empty() { "/" } else { target })
        .map_err(|e| {
            OxiH5Error::NotFound(format!(
                "external link {}::{ext_path}: {e}",
                resolved.display()
            ))
        })
}

/// Resolve an external-link filename to an absolute `PathBuf`.
pub(crate) fn resolve_external_path(
    ext_file: &str,
    source_dir: &std::path::Path,
) -> std::path::PathBuf {
    if std::path::Path::new(ext_file).is_absolute() {
        std::path::PathBuf::from(ext_file)
    } else {
        source_dir.join(ext_file)
    }
}

/// Resolve a Virtual Dataset (layout class 3) into a contiguous data buffer.
///
/// The VDS mapping block (in the global heap) describes, per source dataset,
/// which region of the source maps onto which region of the virtual dataspace.
/// For each entry we open the source dataset (in this same file when the source
/// filename is `"."` or empty, otherwise relative to `source_dir`), read the
/// selected source region and scatter it into the virtual output buffer.
///
/// Elements of the virtual dataset not covered by any mapping keep their fill
/// value, which is zero here (non-zero fill values are not yet applied).
/// Variable-length virtual datasets are not supported (the global-heap
/// references would point into the source files' heaps, not this file's).
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_virtual_dataset(
    file_data: &[u8],
    heap_address: u64,
    heap_index: u32,
    dsp: &oxih5_format::message::DataspaceInfo,
    dtype: &Dtype,
    source_dir: &std::path::Path,
    _cache: Option<&ChunkIndexCache>,
) -> Result<Vec<u8>, OxiH5Error> {
    if is_vlen_dtype(dtype) {
        return Err(OxiH5Error::NotImplemented(
            "virtual dataset with variable-length elements is not supported".into(),
        ));
    }
    let elem_size = dtype.size().ok_or_else(|| {
        OxiH5Error::NotImplemented("virtual dataset: element size not supported".into())
    })?;

    let sb = superblock::parse(file_data)?;
    let mapping = oxih5_format::vds::parse_vds_mapping(
        file_data,
        heap_address,
        heap_index,
        sb.size_of_lengths as usize,
    )?;

    let virt_dims: Vec<u64> = dsp.dims.clone();
    let virt_nelems: u64 = virt_dims.iter().product();
    let virt_nelems = usize::try_from(virt_nelems)
        .map_err(|_| OxiH5Error::Format("virtual dataset: size overflow".into()))?;
    let total_bytes = virt_nelems
        .checked_mul(elem_size)
        .ok_or_else(|| OxiH5Error::Format("virtual dataset: size overflow".into()))?;
    let mut out = vec![0u8; total_bytes];

    for entry in &mapping.entries {
        // Open the source dataset (same file or an external file).
        let source = if entry.source_file == "." || entry.source_file.is_empty() {
            let f = File::open_from_bytes(file_data)?;
            f.dataset(&entry.source_dataset)?
        } else {
            let resolved = resolve_external_path(&entry.source_file, source_dir);
            let f = open(&resolved)?;
            f.dataset(&entry.source_dataset)?
        };

        if source.dtype.size() != Some(elem_size) {
            return Err(OxiH5Error::Format(format!(
                "virtual dataset: source '{}' element size mismatch",
                entry.source_dataset
            )));
        }

        let src_dims: Vec<u64> = source.shape.iter().map(|&s| s as u64).collect();
        let src_offsets =
            oxih5_format::vds::selection_element_offsets(&entry.source_selection, &src_dims)?;
        let virt_offsets =
            oxih5_format::vds::selection_element_offsets(&entry.virtual_selection, &virt_dims)?;

        if src_offsets.len() != virt_offsets.len() {
            return Err(OxiH5Error::Format(format!(
                "virtual dataset: source/virtual selection element count mismatch ({} vs {}) for '{}'",
                src_offsets.len(),
                virt_offsets.len(),
                entry.source_dataset
            )));
        }

        for (&src_i, &virt_i) in src_offsets.iter().zip(virt_offsets.iter()) {
            let src_byte = src_i
                .checked_mul(elem_size)
                .ok_or_else(|| OxiH5Error::Format("virtual dataset: offset overflow".into()))?;
            let virt_byte = virt_i
                .checked_mul(elem_size)
                .ok_or_else(|| OxiH5Error::Format("virtual dataset: offset overflow".into()))?;
            let src_slice = source
                .data
                .get(src_byte..src_byte + elem_size)
                .ok_or_else(|| {
                    OxiH5Error::Format("virtual dataset: source read out of bounds".into())
                })?;
            let dst_slice = out
                .get_mut(virt_byte..virt_byte + elem_size)
                .ok_or_else(|| {
                    OxiH5Error::Format("virtual dataset: virtual write out of bounds".into())
                })?;
            dst_slice.copy_from_slice(src_slice);
        }
    }

    Ok(out)
}
