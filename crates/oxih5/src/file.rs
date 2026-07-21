//! `File` — an open HDF5 file handle.
//!
//! Extracted from `lib.rs` to keep individual source files under the 2000-line
//! limit.  Resolves dataset and group paths, exposes superblock metadata, and
//! dereferences HDF5 object references.

use oxih5_core::{Attribute, Dataset, Dtype, OxiH5Error};
use oxih5_format::superblock::SuperblockExtension;
use oxih5_format::{header, superblock, ChunkIndexCache};

use crate::links::GroupRef;
use crate::reader::{
    find_symbol_table_addresses, navigate_to_group, read_attributes_from_header,
    read_dataset_from_group, read_dataset_from_object_header, resolve_header_address_in,
};
use crate::slicing::{read_dataset_hyperslab_internal, read_dataset_slice_lazy};
use crate::{AttrView, DimSelection, FileData, FileInfo, Group, Hyperslab, ObjectKind};

/// An open HDF5 file (file bytes held in memory or memory-mapped).
pub struct File {
    pub(crate) data: FileData,
    /// Directory of the source file, used to resolve relative external link paths.
    pub(crate) source_dir: std::path::PathBuf,
    /// Pre-parsed chunk index cache shared across all reads from this file.
    pub(crate) chunk_cache: ChunkIndexCache,
}

impl File {
    /// Open an HDF5 file for reading, loading all bytes into a heap `Vec<u8>`.
    ///
    /// For large files consider [`File::open_mmap`] which uses memory-mapped
    /// I/O instead and pages in only the regions that are actually touched.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, OxiH5Error> {
        crate::open(path)
    }

    /// Open an HDF5 file for reading using memory-mapped I/O.
    ///
    /// The OS pages in only the regions that are actually touched, which makes
    /// opening large (100 MB+) HDF5 files essentially free.  The file must
    /// not be modified externally for the lifetime of this handle.
    pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> Result<Self, OxiH5Error> {
        crate::open_mmap(path)
    }

    /// Open an HDF5 file from in-memory bytes (for testing and fuzzing).
    ///
    /// The provided bytes are copied into a heap-allocated `Arc<Vec<u8>>` and
    /// parsed exactly as if the file had been loaded via [`File::open`].  This
    /// is useful in unit tests and fuzzing harnesses where no filesystem path
    /// is available.
    pub fn open_from_bytes(data: &[u8]) -> Result<Self, OxiH5Error> {
        Ok(File {
            data: FileData::Heap(std::sync::Arc::new(data.to_vec())),
            source_dir: std::path::PathBuf::from("."),
            chunk_cache: ChunkIndexCache::new(),
        })
    }

    /// List all top-level dataset names in the root group.
    ///
    /// Note: this only lists datasets at the root level, not those inside nested groups.
    pub fn dataset_names(&self) -> Result<Vec<String>, OxiH5Error> {
        self.root()?.datasets()
    }

    /// Read a dataset by path.
    ///
    /// Supports both flat names (`"temperature"`) and hierarchical paths
    /// (`"/group1/subgroup/data"` or `"group1/subgroup/data"`).
    pub fn dataset(&self, path: &str) -> Result<Dataset, OxiH5Error> {
        // Split path into group-navigation segments + final dataset name.
        let normalized = path.trim_start_matches('/');
        let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();

        let dataset_name = parts
            .pop()
            .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

        let parent = navigate_to_group(&self.data, GroupRef::root(&self.data)?, &parts)?;
        // Resolve the final name — hard link, soft link or external link.
        read_dataset_from_group(
            &self.data,
            parent,
            dataset_name,
            &self.source_dir,
            Some(&self.chunk_cache),
        )
    }

    /// Get the root group handle.
    pub fn root(&self) -> Result<Group, OxiH5Error> {
        let sb = superblock::parse(&self.data)?;
        let messages = header::parse_messages(&self.data, sb.root_object_header_address)?;
        let (btree_address, heap_address, new_style) =
            if let Some((bt, hp)) = find_symbol_table_addresses(&messages) {
                (bt, hp, false)
            } else {
                (0, 0, true)
            };
        Ok(Group {
            name: "/".to_string(),
            object_header_address: sb.root_object_header_address,
            btree_address,
            heap_address,
            new_style,
            file_data: self.data.clone(),
            source_dir: self.source_dir.clone(),
            chunk_cache: self.chunk_cache.clone(),
        })
    }

    /// Navigate to a group by hierarchical path (e.g. `"/sensors/imu"` or `"sensors/imu"`).
    ///
    /// Pass `"/"` or `""` to get the root group.
    pub fn group(&self, path: &str) -> Result<Group, OxiH5Error> {
        let segments: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if segments.is_empty() {
            return self.root();
        }

        let last_segment = segments
            .last()
            .copied()
            .unwrap_or_else(|| path.trim_start_matches('/'));

        // Navigation is identical for both group storage styles: `GroupRef`
        // resolves each segment against whichever indexing the group uses, and
        // follows soft links on the way.
        let target = navigate_to_group(&self.data, GroupRef::root(&self.data)?, &segments)?;
        let (btree_address, heap_address) = target.symbol_table.unwrap_or((0, 0));
        Ok(Group {
            name: last_segment.to_string(),
            object_header_address: target.header,
            btree_address,
            heap_address,
            new_style: target.symbol_table.is_none(),
            file_data: self.data.clone(),
            source_dir: self.source_dir.clone(),
            chunk_cache: self.chunk_cache.clone(),
        })
    }

    /// Read a sub-region of a dataset using lazy (per-chunk) loading.
    ///
    /// `ranges` specifies one `Range<usize>` per dimension.  For a 1-D dataset of
    /// length 100, `ranges = [10..20]` returns elements 10–19.
    ///
    /// For chunked datasets only the chunks overlapping `ranges` are decompressed.
    /// For contiguous/compact datasets the full data is loaded first.
    pub fn dataset_slice(
        &self,
        path: &str,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<Dataset, OxiH5Error> {
        read_dataset_slice_lazy(
            &self.data,
            path,
            ranges,
            &self.source_dir,
            &self.chunk_cache,
        )
    }

    /// Read a sub-region of a dataset using a strided HDF5 hyperslab selection.
    ///
    /// Each [`DimSelection`] specifies `start`/`stride`/`count`/`block` for one
    /// dimension.  For chunked datasets only the chunks overlapping the selection
    /// bounding box are decompressed — interior elements not passing the stride/block
    /// filter are dropped without reading.  For contiguous/compact datasets the full
    /// data is loaded and then sampled.
    pub fn dataset_hyperslab(
        &self,
        path: &str,
        selection: &[DimSelection],
    ) -> Result<Dataset, OxiH5Error> {
        let hs = Hyperslab {
            dims: selection.to_vec(),
        };
        read_dataset_hyperslab_internal(&self.data, path, &hs, &self.source_dir, &self.chunk_cache)
    }

    /// Check whether the given path (dataset or group) exists in the file.
    ///
    /// Accepts both bare names (`"temperature"`) and hierarchical paths
    /// (`"/group1/data"`).  Returns `false` for paths that cannot be navigated.
    pub fn contains(&self, path: &str) -> bool {
        self.dataset(path).is_ok() || self.group(path).is_ok()
    }

    /// Walk the entire file tree in pre-order, calling `visitor` for every
    /// dataset and group encountered.
    ///
    /// The visitor receives `(full_path: &str, is_group: bool)`.
    /// Groups are visited before their children.  Non-fatal errors while
    /// descending into sub-groups are silently skipped.
    pub fn walk(&self, visitor: &mut impl FnMut(&str, bool)) -> Result<(), OxiH5Error> {
        let root = self.root()?;
        self.walk_group(&root, "/", visitor)
    }

    fn walk_group(
        &self,
        group: &Group,
        path: &str,
        visitor: &mut impl FnMut(&str, bool),
    ) -> Result<(), OxiH5Error> {
        // Visit datasets in this group.
        if let Ok(datasets) = group.datasets() {
            for ds_name in &datasets {
                let full_path = if path == "/" {
                    format!("/{ds_name}")
                } else {
                    format!("{path}/{ds_name}")
                };
                visitor(&full_path, false);
            }
        }

        // Visit sub-groups, then recurse into each.
        if let Ok(group_names) = group.groups() {
            for grp_name in &group_names {
                let full_path = if path == "/" {
                    format!("/{grp_name}")
                } else {
                    format!("{path}/{grp_name}")
                };
                visitor(&full_path, true);
                if let Ok(sub_group) = self.group(&full_path) {
                    // Errors from deeper levels are swallowed; the walk continues.
                    let _ = self.walk_group(&sub_group, &full_path, visitor);
                }
            }
        }

        Ok(())
    }

    /// Return file-level metadata from the superblock.
    pub fn info(&self) -> Result<FileInfo, OxiH5Error> {
        let sb = superblock::parse(&self.data)?;
        Ok(FileInfo {
            superblock_version: sb.version,
            file_size: self.data.len() as u64,
            offset_size: sb.size_of_offsets,
            length_size: sb.size_of_lengths,
            superblock_extension_address: sb.superblock_extension_address,
        })
    }

    /// Parse and return the interpretable messages from this file's superblock
    /// extension, or `Ok(None)` when the file has no extension.
    ///
    /// The superblock extension is a regular object header carrying file-level
    /// metadata messages.  It exists only for superblock versions 2 and 3
    /// (versions 0/1 carry a Driver Info Block instead) and only when the
    /// superblock's extension-address field is defined.  This accessor decodes
    /// the messages oxih5 currently understands — B-tree 'K' Values (0x0013),
    /// Shared Message Table (0x000F), File Space Info (0x0018) and Driver Info
    /// (0x0014) — and surfaces them via [`SuperblockExtension`].
    ///
    /// A malformed message (too short to decode, or an unsupported message
    /// version) is reported as a typed error rather than silently ignored.
    ///
    /// Note: the returned [`crate::BtreeKValues`] are surfaced for inspection only.
    /// oxih5's old-style B-tree/group traversal currently uses the HDF5
    /// specification default K values, which is correct for all standard files.
    /// Applying non-default K values from the extension to traversal is a
    /// documented follow-up (it would require threading the values into the
    /// B-tree and group readers).
    pub fn superblock_extension(&self) -> Result<Option<SuperblockExtension>, OxiH5Error> {
        superblock::read_superblock_extension(&self.data)
    }

    /// Resolve an HDF5 object reference (absolute byte address) to a `Dataset` or `Group`.
    ///
    /// `addr` is the absolute byte offset of the target object's header, as returned
    /// by `AttrView::as_object_refs()` or `values::decode_object_refs()`.
    /// Returns `OxiH5Error::NotFound` for `u64::MAX` (undefined reference).
    pub fn object_at(&self, addr: u64) -> Result<ObjectKind, OxiH5Error> {
        if addr == u64::MAX {
            return Err(OxiH5Error::NotFound("undefined object reference".into()));
        }
        // Check whether the object at `addr` is a group by looking for a
        // SymbolTable (0x0011) or LinkInfo (0x0002) message.
        let msgs = header::parse_messages(&self.data, addr)?;
        let is_group = msgs
            .iter()
            .any(|m| m.msg_type == 0x0011 || m.msg_type == 0x0002);
        if is_group {
            // Determine old-style vs new-style.
            let (btree_address, heap_address, new_style) =
                if let Some((bt, hp)) = find_symbol_table_addresses(&msgs) {
                    (bt, hp, false)
                } else {
                    (0, 0, true)
                };
            Ok(ObjectKind::Group(Group {
                name: format!("@{addr:#x}"),
                object_header_address: addr,
                btree_address,
                heap_address,
                new_style,
                file_data: self.data.clone(),
                source_dir: self.source_dir.clone(),
                chunk_cache: self.chunk_cache.clone(),
            }))
        } else {
            // Treat as dataset.
            let ds = read_dataset_from_object_header(
                &self.data,
                addr,
                &format!("@{addr:#x}"),
                &self.source_dir,
                Some(&self.chunk_cache),
            )?;
            Ok(ObjectKind::Dataset(ds))
        }
    }

    /// Resolve an object reference directly to a `Dataset`.
    ///
    /// Convenience wrapper around `object_at` that returns `TypeMismatch` if the
    /// referenced object is a group rather than a dataset.
    pub fn dataset_at(&self, addr: u64) -> Result<Dataset, OxiH5Error> {
        match self.object_at(addr)? {
            ObjectKind::Dataset(ds) => Ok(ds),
            ObjectKind::Group(_) => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Return `AttrView` wrappers for all attributes on a dataset at `path`.
    ///
    /// Each `AttrView` owns its `Attribute` data and borrows the file bytes
    /// from `self` for the duration of the returned views' lifetime.
    pub fn attr_views(&self, path: &str) -> Result<Vec<AttrView<'_>>, OxiH5Error> {
        let header_addr = self.resolve_dataset_header_addr(path)?;
        let attrs = read_attributes_from_header(&self.data, header_addr)?;
        Ok(attrs
            .into_iter()
            .map(|a| AttrView::new(a, &self.data))
            .collect())
    }

    /// Read only the attributes from the object header at `addr`.
    ///
    /// Does NOT read dataspace, layout, or data — lightweight for metadata-only
    /// access.  Intended for use by callers (e.g. `oxinetcdf`) that already
    /// know the header address from an object reference or prior navigation and
    /// need only attribute metadata without the cost of loading dataset data.
    ///
    /// Returns `OxiH5Error::NotFound` for `u64::MAX` (undefined reference).
    pub fn attrs_of(&self, addr: u64) -> Result<Vec<Attribute>, OxiH5Error> {
        if addr == u64::MAX {
            return Err(OxiH5Error::NotFound("undefined object reference".into()));
        }
        read_attributes_from_header(&self.data, addr)
    }

    /// Returns the object-header address for the dataset at `path`.
    ///
    /// This is the raw HDF5 file byte-offset of the object header, providing a
    /// stable, cross-group identifier for an object.  Useful as a map key in
    /// dimension-scale resolution (e.g. `oxinetcdf`'s DIMENSION_LIST address map).
    ///
    /// Returns [`OxiH5Error::NotFound`] if `path` does not exist.
    pub fn header_addr_of(&self, path: &str) -> Result<u64, OxiH5Error> {
        self.resolve_dataset_header_addr(path)
    }

    /// Internal helper: resolve the object header address for the dataset at `path`.
    pub(crate) fn resolve_dataset_header_addr(&self, path: &str) -> Result<u64, OxiH5Error> {
        let normalized = path.trim_start_matches('/');
        let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
        let dataset_name = parts
            .pop()
            .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

        let parent = navigate_to_group(&self.data, GroupRef::root(&self.data)?, &parts)?;
        resolve_header_address_in(&self.data, parent, dataset_name, &self.source_dir)
    }

    /// Decode a vlen-string dataset as `Vec<String>`.
    ///
    /// A convenience shortcut for the common pattern of reading a vlen-string
    /// dataset and decoding all elements to UTF-8 `String`s.
    ///
    /// Returns `OxiH5Error::TypeMismatch` if the dataset dtype is not a vlen string.
    pub fn dataset_strings(&self, path: &str) -> Result<Vec<String>, OxiH5Error> {
        let ds = self.dataset(path)?;
        match &ds.dtype {
            Dtype::String {
                fixed_len: None, ..
            } => {
                // vlen string dataset: data contains n_elems × 16-byte heap refs
                let n_elems = ds.len();
                oxih5_format::values::decode_vlen_strings(&self.data, &ds.data, n_elems)
            }
            Dtype::String {
                fixed_len: Some(_), ..
            } => {
                // fixed-length string dataset: use Dataset::as_string
                ds.as_string()
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Decode a variable-length sequence dataset (datatype class 9 `VarLen`)
    /// into one [`oxih5_format::values::Value::Sequence`] per element.
    ///
    /// Works for both contiguous and chunked layouts: the chunked read path
    /// assembles the on-disk 16-byte global-heap references, which are then
    /// dereferenced here.  Returns [`OxiH5Error::TypeMismatch`] if the dataset
    /// is not a vlen sequence (vlen *strings* should use [`File::dataset_strings`]).
    pub fn dataset_vlen_sequences(
        &self,
        path: &str,
    ) -> Result<Vec<oxih5_format::values::Value>, OxiH5Error> {
        let ds = self.dataset(path)?;
        match &ds.dtype {
            Dtype::VarLen { base } => {
                let n_elems = ds.len();
                oxih5_format::values::decode_vlen_sequences(&self.data, &ds.data, n_elems, base)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.data.len();
        let root_datasets = self.dataset_names().map(|v| v.len()).unwrap_or(0);
        write!(
            f,
            "File {{ size: {} bytes, root_datasets: {} }}",
            size, root_datasets
        )
    }
}
