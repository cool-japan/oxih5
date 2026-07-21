//! `Group` — a handle to a group within an open HDF5 file.
//!
//! Extracted from `lib.rs` to keep individual source files under the 2000-line
//! limit.  Handles both old-style groups (symbol-table B-tree + SNOD + local
//! heap) and new-style groups (Link messages 0x0006).

use oxih5_core::{Attribute, Dataset, OxiH5Error};
use oxih5_format::{group, header, ChunkIndexCache};

use crate::links::{
    list_children, lookup_child, resolve_external_link_group, resolve_soft_link_to_header,
    ChildLink, GroupRef,
};
use crate::reader::{
    find_symbol_table_addresses, read_attributes_from_header, read_dataset_from_group,
};
use crate::slicing::{
    read_dataset_hyperslab_from_group_internal, read_dataset_slice_lazy_from_group,
};
use crate::{AttrView, DimSelection, FileData, Hyperslab};

/// A handle to a group within an HDF5 file.
pub struct Group {
    /// Name of this group (last path segment, or `"/"` for root).
    pub name: String,
    pub(crate) object_header_address: u64,
    pub(crate) btree_address: u64,
    pub(crate) heap_address: u64,
    /// `true` when this is a new-style group that uses Link messages (0x0006)
    /// instead of the old-style Symbol Table / B-tree / SNOD mechanism.
    pub(crate) new_style: bool,
    pub(crate) file_data: FileData,
    /// Directory of the source file, used to resolve relative external link paths.
    pub(crate) source_dir: std::path::PathBuf,
    /// Pre-parsed chunk index cache shared with the parent `File`.
    pub(crate) chunk_cache: ChunkIndexCache,
}

impl Group {
    /// This group as a style-agnostic reference for the link resolvers.
    pub(crate) fn as_ref(&self) -> GroupRef {
        GroupRef::new(
            self.object_header_address,
            if self.new_style {
                None
            } else {
                Some((self.btree_address, self.heap_address))
            },
        )
    }

    /// List names of all datasets (non-group objects) in this group.
    pub fn datasets(&self) -> Result<Vec<String>, OxiH5Error> {
        self.list_entries_by_type(false)
    }

    /// List names of all sub-groups in this group.
    pub fn groups(&self) -> Result<Vec<String>, OxiH5Error> {
        self.list_entries_by_type(true)
    }

    /// Read a dataset by name from this group (one level only — no path traversal).
    pub fn dataset(&self, name: &str) -> Result<Dataset, OxiH5Error> {
        read_dataset_from_group(
            &self.file_data,
            self.as_ref(),
            name,
            &self.source_dir,
            Some(&self.chunk_cache),
        )
    }

    /// Navigate to a named sub-group within this group (one level only).
    ///
    /// Returns a `Group` handle for the child with the given name.
    /// Supports hard links and soft links.  For external links that point to a
    /// group in another file, the external file is opened and the target group
    /// is returned.
    pub fn group(&self, name: &str) -> Result<Group, OxiH5Error> {
        let me = self.as_ref();
        let child_addr = match lookup_child(&self.file_data, me, name)? {
            ChildLink::Hard(address) => address,
            ChildLink::Soft(path) => resolve_soft_link_to_header(&self.file_data, me, &path)?,
            ChildLink::External {
                file: ext_file,
                path: ext_path,
            } => {
                return resolve_external_link_group(&ext_file, &ext_path, &self.source_dir);
            }
        };

        let child_msgs = header::parse_messages(&self.file_data, child_addr)?;
        let symbol_table = find_symbol_table_addresses(&child_msgs);
        // Old-style groups must still be rejected when they name a dataset;
        // new-style ones report the same thing from their empty link list.
        if symbol_table.is_none()
            && !child_msgs
                .iter()
                .any(|m| m.msg_type == 0x0006 || m.msg_type == 0x0002)
        {
            return Err(OxiH5Error::NotFound(format!("'{name}' is not a group")));
        }
        let (btree_address, heap_address) = symbol_table.unwrap_or((0, 0));
        Ok(Group {
            name: name.to_string(),
            object_header_address: child_addr,
            btree_address,
            heap_address,
            new_style: symbol_table.is_none(),
            file_data: self.file_data.clone(),
            source_dir: self.source_dir.clone(),
            chunk_cache: self.chunk_cache.clone(),
        })
    }

    /// Read a sub-region of a dataset by name within this group using lazy chunk loading.
    ///
    /// `ranges` specifies one `Range<usize>` per dimension.
    /// For chunked datasets only the chunks overlapping `ranges` are decompressed.
    pub fn dataset_slice(
        &self,
        name: &str,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<Dataset, OxiH5Error> {
        read_dataset_slice_lazy_from_group(
            &self.file_data,
            self.object_header_address,
            self.btree_address,
            self.heap_address,
            self.new_style,
            name,
            ranges,
            &self.source_dir,
            &self.chunk_cache,
        )
    }

    /// Read a sub-region of a dataset in this group using a strided HDF5 hyperslab selection.
    ///
    /// Each [`DimSelection`] specifies `start`/`stride`/`count`/`block` for one dimension.
    pub fn dataset_hyperslab(
        &self,
        name: &str,
        selection: &[DimSelection],
    ) -> Result<Dataset, OxiH5Error> {
        let hs = Hyperslab {
            dims: selection.to_vec(),
        };
        read_dataset_hyperslab_from_group_internal(
            &self.file_data,
            self.object_header_address,
            self.btree_address,
            self.heap_address,
            self.new_style,
            name,
            &hs,
            &self.source_dir,
            &self.chunk_cache,
        )
    }

    /// List all attributes attached to this group.
    pub fn attrs(&self) -> Result<Vec<Attribute>, OxiH5Error> {
        read_attributes_from_header(&self.file_data, self.object_header_address)
    }

    /// Return `AttrView` wrappers for all attributes attached to this group.
    ///
    /// Each `AttrView` owns its `Attribute` data and borrows the file bytes
    /// from this `Group` handle for the duration of the returned views' lifetime.
    pub fn attr_views(&self) -> Result<Vec<AttrView<'_>>, OxiH5Error> {
        let attrs = read_attributes_from_header(&self.file_data, self.object_header_address)?;
        Ok(attrs
            .into_iter()
            .map(|a| AttrView::new(a, &self.file_data))
            .collect())
    }

    /// List all entries in this group, partitioned by whether they are groups.
    ///
    /// `want_groups = true`  → return sub-group names
    /// `want_groups = false` → return dataset names
    ///
    /// Works the same way for both group storage styles: soft links are
    /// resolved and then classified by what they point at, so an alias appears
    /// in the same list as its target.  Links that cannot be resolved (dangling
    /// or cyclic) are silently skipped, and external links are excluded from
    /// listings entirely.
    fn list_entries_by_type(&self, want_groups: bool) -> Result<Vec<String>, OxiH5Error> {
        let me = self.as_ref();
        let children = list_children(&self.file_data, me)?;
        let mut names = Vec::with_capacity(children.len());

        for (name, link) in children {
            let entry_name = name.trim_start_matches('/').to_string();
            if entry_name.is_empty() {
                continue;
            }
            let addr = match link {
                ChildLink::Hard(address) => address,
                ChildLink::Soft(path) => {
                    match resolve_soft_link_to_header(&self.file_data, me, &path) {
                        Ok(address) => address,
                        Err(_) => continue,
                    }
                }
                ChildLink::External { .. } => continue,
            };
            if self.is_group_at(addr) == want_groups {
                names.push(entry_name);
            }
        }

        Ok(names)
    }

    /// `true` when the object header at `addr` belongs to a group, in either
    /// storage style: old-style groups carry a Symbol Table message (0x0011),
    /// new-style ones Link (0x0006) or Link Info (0x0002) messages.
    fn is_group_at(&self, addr: u64) -> bool {
        group::is_new_style_group(&self.file_data, addr)
            || header::parse_messages(&self.file_data, addr)
                .map(|msgs| msgs.iter().any(|m| m.msg_type == 0x0011))
                .unwrap_or(false)
    }
}
