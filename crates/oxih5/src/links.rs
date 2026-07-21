//! Link resolution: soft links, external links, and soft→external chains.
//!
//! Extracted from `lib.rs` to keep individual source files under the 2000-line
//! limit.  These helpers navigate groups of **either** storage style and open
//! external HDF5 files referenced by external links.
//!
//! # Why one resolver for both group styles
//!
//! HDF5 has two unrelated ways to store a group's members:
//!
//! * **New-style** (`libver='latest'`): Link messages (0x0006) in the object
//!   header, or a fractal heap referenced by a Link Info message (0x0002).
//!   Each link carries an explicit type — hard, soft or external.
//! * **Old-style** (h5py's *default* `libver`): a version-1 B-tree over SNOD
//!   symbol-table nodes plus a local heap.  A symbol-table entry has no link
//!   type field: a soft link is one whose *cache type* is 2, whose object
//!   header address is the undefined-address sentinel (`u64::MAX`), and whose
//!   value is a path stored in the group's local heap.
//!
//! A path may cross groups of both styles, and a soft link's target is just a
//! path — so following one requires navigation that does not care which style
//! each group along the way uses.  [`GroupRef`] erases that difference and
//! [`lookup_child`] is the single per-segment step every resolver here builds
//! on.

use oxih5_core::{Dataset, Dtype, OxiH5Error};
use oxih5_format::{group, header, superblock, ChunkIndexCache};

use crate::open;
use crate::reader::{find_symbol_table_addresses, is_vlen_dtype, read_dataset_from_object_header};
use crate::{File, Group};

/// A group, identified by its object header and by how its members are indexed.
///
/// Constructed either from an object-header address via [`GroupRef::at`], or
/// directly by a [`crate::Group`] handle that already knows its own addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupRef {
    /// Absolute file address of this group's object header.
    pub(crate) header: u64,
    /// `Some((btree_address, heap_address))` for an old-style symbol-table
    /// group; `None` for a new-style Link-message group.
    pub(crate) symbol_table: Option<(u64, u64)>,
}

impl GroupRef {
    /// Classify the group whose object header lives at `header_addr`.
    ///
    /// An object with a Symbol Table message (0x0011) is old-style; anything
    /// else is treated as new-style, which yields an empty child list for
    /// objects that are not groups at all (and hence a `NotFound` from
    /// [`lookup_child`] rather than a misleading parse error).
    pub(crate) fn at(file_data: &[u8], header_addr: u64) -> Result<Self, OxiH5Error> {
        let msgs = header::parse_messages(file_data, header_addr)?;
        Ok(GroupRef {
            header: header_addr,
            symbol_table: find_symbol_table_addresses(&msgs),
        })
    }

    /// Build a reference from addresses a caller already holds.
    pub(crate) fn new(header: u64, symbol_table: Option<(u64, u64)>) -> Self {
        GroupRef {
            header,
            symbol_table,
        }
    }

    /// The root group of `file_data`.
    pub(crate) fn root(file_data: &[u8]) -> Result<Self, OxiH5Error> {
        let sb = superblock::parse(file_data)?;
        GroupRef::at(file_data, sb.root_object_header_address)
    }
}

/// What a name resolves to inside a group, independent of storage style.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChildLink {
    /// A hard link: the object header address of the target.
    Hard(u64),
    /// A soft link: an in-file path, absolute (`/a/b`) or relative to the group
    /// that holds the link.
    Soft(String),
    /// An external link: a path inside another file.
    External { file: String, path: String },
}

/// Look up `name` among the children of `group`.
///
/// This is the one place that knows the difference between the two group
/// storage styles; every path-walking routine below is written in terms of it.
pub(crate) fn lookup_child(
    file_data: &[u8],
    group: GroupRef,
    name: &str,
) -> Result<ChildLink, OxiH5Error> {
    match group.symbol_table {
        Some((btree, heap)) => match group::find_entry(file_data, btree, heap, name)? {
            group::SymTabLink::Hard(address) => Ok(ChildLink::Hard(address)),
            group::SymTabLink::Soft(path) => Ok(ChildLink::Soft(path)),
        },
        None => {
            for pl in &list_links(file_data, group.header)? {
                if pl.name == name {
                    return Ok(child_link_of(&pl.link));
                }
            }
            Err(OxiH5Error::NotFound(name.to_string()))
        }
    }
}

/// List every child of `group` as `(name, link)` pairs, in either style.
pub(crate) fn list_children(
    file_data: &[u8],
    group: GroupRef,
) -> Result<Vec<(String, ChildLink)>, OxiH5Error> {
    match group.symbol_table {
        Some((btree, heap)) => Ok(group::list_entries(file_data, btree, heap)?
            .into_iter()
            .map(|(name, link)| {
                let child = match link {
                    group::SymTabLink::Hard(address) => ChildLink::Hard(address),
                    group::SymTabLink::Soft(path) => ChildLink::Soft(path),
                };
                (name, child)
            })
            .collect()),
        None => Ok(list_links(file_data, group.header)?
            .into_iter()
            .map(|pl| {
                let child = child_link_of(&pl.link);
                (pl.name, child)
            })
            .collect()),
    }
}

/// Read the Link messages of the new-style group at `header_addr`.
fn list_links(
    file_data: &[u8],
    header_addr: u64,
) -> Result<Vec<oxih5_format::link_msg::ParsedLink>, OxiH5Error> {
    let sb = superblock::parse(file_data)?;
    let ctx = oxih5_format::context::ParseContext::new(
        sb.size_of_offsets,
        sb.size_of_lengths,
        sb.base_address,
    );
    group::list_new_style_links(file_data, header_addr, &ctx)
}

fn child_link_of(link: &oxih5_core::Link) -> ChildLink {
    match link {
        oxih5_core::Link::Hard { address } => ChildLink::Hard(*address),
        oxih5_core::Link::Soft { path } => ChildLink::Soft(path.clone()),
        oxih5_core::Link::External { file, path } => ChildLink::External {
            file: file.clone(),
            path: path.clone(),
        },
    }
}

/// The resolved target of a link path.
pub(crate) enum SoftTarget {
    /// A hard object header address within the same file.
    Local(u64),
    /// The path ultimately reaches an external link — the referenced object
    /// lives in another file.
    External { file: String, path: String },
}

/// Cycle guard for link resolution.
///
/// Keyed by `(starting group's object header, path)` rather than by path alone:
/// a *relative* soft link value such as `"inner"` is a different target in
/// every group that holds one, so the path on its own does not identify a
/// resolution step.
pub(crate) type VisitedLinks = std::collections::HashSet<(u64, String)>;

/// Resolve `path` against `base`, following hard, soft and (terminal) external
/// links through groups of either storage style.
///
/// A path beginning with `/` is resolved from the file root; any other path is
/// resolved from `base`, which is how libhdf5 interprets a relative soft-link
/// value — relative to the group that holds the link.
pub(crate) fn resolve_path_target(
    file_data: &[u8],
    base: GroupRef,
    path: &str,
    visited: &mut VisitedLinks,
) -> Result<SoftTarget, OxiH5Error> {
    let start = if path.starts_with('/') {
        GroupRef::root(file_data)?
    } else {
        base
    };

    if !visited.insert((start.header, path.to_string())) {
        return Err(OxiH5Error::Format(format!(
            "soft link cycle detected at path '{path}'"
        )));
    }

    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    // An empty path ("" or "/") names the group we started from.
    if parts.is_empty() {
        return Ok(SoftTarget::Local(start.header));
    }

    let mut current = start;
    for (idx, segment) in parts.iter().enumerate() {
        let is_last = idx == parts.len() - 1;
        let child = lookup_child(file_data, current, segment).map_err(|e| match e {
            OxiH5Error::NotFound(_) => OxiH5Error::NotFound(format!(
                "link target '{path}': segment '{segment}' not found"
            )),
            other => other,
        })?;

        let next_header = match child {
            ChildLink::Hard(address) => address,
            ChildLink::Soft(nested) => {
                // The nested value is resolved relative to `current`, the group
                // that holds it — not relative to wherever this walk began.
                let target = resolve_path_target(file_data, current, &nested, visited)?;
                if is_last {
                    return Ok(target);
                }
                match target {
                    SoftTarget::Local(address) => address,
                    SoftTarget::External { .. } => {
                        return Err(OxiH5Error::NotImplemented(
                            "soft link traverses an external link mid-path".into(),
                        ))
                    }
                }
            }
            ChildLink::External { file, path: p } => {
                if is_last {
                    return Ok(SoftTarget::External { file, path: p });
                }
                return Err(OxiH5Error::NotImplemented(
                    "soft link traverses an external link mid-path".into(),
                ));
            }
        };

        if is_last {
            return Ok(SoftTarget::Local(next_header));
        }
        current = GroupRef::at(file_data, next_header)?;
    }

    Ok(SoftTarget::Local(current.header))
}

/// Resolve `path` against `base` to an object header address in this file.
///
/// Fails with `NotImplemented` when the path leads out of the file through an
/// external link, because such a target has no address here.
pub(crate) fn resolve_path_to_header(
    file_data: &[u8],
    base: GroupRef,
    path: &str,
    visited: &mut VisitedLinks,
) -> Result<u64, OxiH5Error> {
    match resolve_path_target(file_data, base, path, visited)? {
        SoftTarget::Local(address) => Ok(address),
        SoftTarget::External { file, path: p } => Err(OxiH5Error::NotImplemented(format!(
            "soft link resolves to an external link to '{p}' in '{file}'; that target has no \
             object-header address in this file"
        ))),
    }
}

/// Convenience wrapper: resolve a soft-link value held by `base`.
pub(crate) fn resolve_soft_link_to_header(
    file_data: &[u8],
    base: GroupRef,
    target_path: &str,
) -> Result<u64, OxiH5Error> {
    let mut visited = VisitedLinks::new();
    resolve_path_to_header(file_data, base, target_path, &mut visited)
}

/// Resolve a dataset name within `group`, handling hard links, soft links
/// (including soft → external chains) and external file links.
///
/// For hard links the dataset is read from the local file at the resolved
/// object header address.  For external links the referenced file is opened
/// and `File::dataset` is called with the target path stored in the link.
pub(crate) fn resolve_dataset_in_group(
    file_data: &[u8],
    group: GroupRef,
    name: &str,
    source_dir: &std::path::Path,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    match lookup_child(file_data, group, name)? {
        ChildLink::Hard(address) => {
            read_dataset_from_object_header(file_data, address, name, source_dir, cache)
        }
        ChildLink::Soft(path) => {
            // Follow the soft link to its target, then read the dataset.  The
            // target may be local, or (soft → external) in another file.
            let mut visited = VisitedLinks::new();
            match resolve_path_target(file_data, group, &path, &mut visited)? {
                SoftTarget::Local(address) => {
                    read_dataset_from_object_header(file_data, address, name, source_dir, cache)
                }
                SoftTarget::External {
                    file: ext_file,
                    path: ext_path,
                } => resolve_external_link(&ext_file, &ext_path, source_dir),
            }
        }
        ChildLink::External {
            file: ext_file,
            path: ext_path,
        } => resolve_external_link(&ext_file, &ext_path, source_dir),
    }
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
