#![deny(unsafe_code)]

pub use oxih5_core::{Attribute, ByteOrder, Dataset, Dtype, OxiH5Error};
pub use oxih5_format::values::Value;
pub use oxih5_format::{DimSelection, Hyperslab};

mod write;
pub use write::FileWriter;

mod attr_view;
pub use attr_view::AttrView;

use oxih5_format::{btree, group, header, heap, message, snod, superblock, ChunkIndexCache};

// ---------------------------------------------------------------------------
// ObjectKind — returned by File::object_at
// ---------------------------------------------------------------------------

/// The kind of object referenced by an HDF5 object reference.
///
/// Returned by [`File::object_at`] and [`File::dataset_at`].
pub enum ObjectKind {
    /// A dataset at the referenced address.
    Dataset(Dataset),
    /// A group at the referenced address.
    Group(Group),
}

// ---------------------------------------------------------------------------
// FileData — backing-store abstraction
// ---------------------------------------------------------------------------

/// Backing store for an open HDF5 file.
///
/// `Heap` holds the entire file in a `Vec<u8>` (the original behaviour);
/// `Mapped` memory-maps the file so the OS pages in only the touched regions.
///
/// Both variants implement `Deref<Target = [u8]>` so all parsing code works
/// identically regardless of which variant is active.
#[derive(Clone)]
pub(crate) enum FileData {
    /// File bytes held in a heap-allocated vector.
    Heap(std::sync::Arc<Vec<u8>>),
    /// File bytes backed by a read-only memory mapping.
    Mapped(std::sync::Arc<memmap2::Mmap>),
}

impl std::ops::Deref for FileData {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            FileData::Heap(v) => v.as_slice(),
            FileData::Mapped(m) => m.as_ref(),
        }
    }
}

impl std::fmt::Debug for FileData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileData::Heap(v) => write!(f, "Heap({}b)", v.len()),
            FileData::Mapped(m) => write!(f, "Mapped({}b)", m.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Open an HDF5 file for reading (file bytes are held in memory).
pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<File, OxiH5Error> {
    let path = path.as_ref();
    let source_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let data = std::fs::read(path)?;
    Ok(File {
        data: FileData::Heap(std::sync::Arc::new(data)),
        source_dir,
        chunk_cache: ChunkIndexCache::new(),
    })
}

/// Open an HDF5 file for reading using memory-mapped I/O.
///
/// Unlike [`open`], which reads the entire file into a heap `Vec<u8>`, this
/// function maps the file into the process address space.  The OS pages in
/// only the regions that are actually touched, which makes opening large
/// (100 MB+) HDF5 files essentially free — you pay only for the data you read.
///
/// The mapping is read-only.  Concurrent external writes to the file while the
/// mapping is live would violate the safety contract of `memmap2::Mmap::map`;
/// only use this function when the file will not be modified for the lifetime
/// of the returned `File` handle.
#[allow(unsafe_code)]
pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> Result<File, OxiH5Error> {
    let path = path.as_ref();
    let source_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let file = std::fs::File::open(path)?;
    // SAFETY: the file is opened read-only and we do not mutate the mapped
    // region anywhere in this library.  The caller must not truncate or write
    // to the file for the lifetime of the returned `File`.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(OxiH5Error::Io)?;
    Ok(File {
        data: FileData::Mapped(std::sync::Arc::new(mmap)),
        source_dir,
        chunk_cache: ChunkIndexCache::new(),
    })
}

/// Read a single dataset by name from an HDF5 file (one-shot convenience wrapper).
pub fn read_dataset<P: AsRef<std::path::Path>>(path: P, name: &str) -> Result<Dataset, OxiH5Error> {
    open(path)?.dataset(name)
}

/// Read a sub-region of a named dataset in an HDF5 file using a strided hyperslab selection.
///
/// Each [`DimSelection`] specifies `start`/`stride`/`count`/`block` for one dimension.
/// For chunked datasets only the chunks overlapping the selection bounding box are decompressed.
pub fn read_dataset_hyperslab<P: AsRef<std::path::Path>>(
    path: P,
    name: &str,
    selection: &[DimSelection],
) -> Result<Dataset, OxiH5Error> {
    File::open(path)?.dataset_hyperslab(name, selection)
}

/// Returns the crate version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// File-level metadata
// ---------------------------------------------------------------------------

/// File-level metadata returned by [`File::info`].
pub struct FileInfo {
    /// Superblock version (currently always 0 — v0 is the only supported version).
    pub superblock_version: u8,
    /// Total byte size of the file as loaded into memory.
    pub file_size: u64,
    /// `size_of_offsets` field from the superblock (typically 8).
    pub offset_size: u8,
    /// `size_of_lengths` field from the superblock (typically 8).
    pub length_size: u8,
}

// ---------------------------------------------------------------------------
// File handle
// ---------------------------------------------------------------------------

/// An open HDF5 file (file bytes held in memory or memory-mapped).
pub struct File {
    data: FileData,
    /// Directory of the source file, used to resolve relative external link paths.
    source_dir: std::path::PathBuf,
    /// Pre-parsed chunk index cache shared across all reads from this file.
    chunk_cache: ChunkIndexCache,
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
        let sb = superblock::parse(&self.data)?;
        let root_msgs = header::parse_messages(&self.data, sb.root_object_header_address)?;

        // Split path into group-navigation segments + final dataset name.
        let normalized = path.trim_start_matches('/');
        let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();

        let dataset_name = parts
            .pop()
            .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

        if let Some((root_btree, root_heap)) = find_symbol_table_addresses(&root_msgs) {
            // Old-style root group.
            let (btree, heap) = if parts.is_empty() {
                (root_btree, root_heap)
            } else {
                navigate_to_group(&self.data, root_btree, root_heap, &parts)?
            };
            read_dataset_from_group(
                &self.data,
                btree,
                heap,
                dataset_name,
                Some(&self.chunk_cache),
            )
        } else {
            // New-style root group: navigate via Link messages.
            let mut current_header = sb.root_object_header_address;
            for segment in &parts {
                current_header = find_new_style_child(&self.data, current_header, segment)?;
            }
            // Resolve the final name — may be a hard link or an external link.
            resolve_new_style_dataset(
                &self.data,
                current_header,
                dataset_name,
                &self.source_dir,
                Some(&self.chunk_cache),
            )
        }
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

        let sb = superblock::parse(&self.data)?;
        let root_msgs = header::parse_messages(&self.data, sb.root_object_header_address)?;

        let last_segment = segments
            .last()
            .copied()
            .unwrap_or_else(|| path.trim_start_matches('/'));

        if let Some((root_btree, root_heap)) = find_symbol_table_addresses(&root_msgs) {
            // Old-style root group: navigate via B-tree + SNOD.
            let mut btree = root_btree;
            let mut heap = root_heap;
            let mut current_header_addr = sb.root_object_header_address;

            for segment in &segments {
                let entry_addr = group::find_dataset(&self.data, btree, heap, segment)?;
                current_header_addr = entry_addr;
                let msgs = header::parse_messages(&self.data, entry_addr)?;
                let (new_btree, new_heap) = find_symbol_table_addresses(&msgs)
                    .ok_or_else(|| OxiH5Error::NotFound(format!("'{}' is not a group", segment)))?;
                btree = new_btree;
                heap = new_heap;
            }

            Ok(Group {
                name: last_segment.to_string(),
                object_header_address: current_header_addr,
                btree_address: btree,
                heap_address: heap,
                new_style: false,
                file_data: self.data.clone(),
                source_dir: self.source_dir.clone(),
                chunk_cache: self.chunk_cache.clone(),
            })
        } else {
            // New-style root group: navigate via Link messages.
            let mut current_header = sb.root_object_header_address;
            for segment in &segments {
                current_header = find_new_style_child(&self.data, current_header, segment)?;
            }
            let child_msgs = header::parse_messages(&self.data, current_header)?;
            let (btree_address, heap_address, new_style) =
                if let Some((bt, hp)) = find_symbol_table_addresses(&child_msgs) {
                    (bt, hp, false)
                } else {
                    (0, 0, true)
                };
            Ok(Group {
                name: last_segment.to_string(),
                object_header_address: current_header,
                btree_address,
                heap_address,
                new_style,
                file_data: self.data.clone(),
                source_dir: self.source_dir.clone(),
                chunk_cache: self.chunk_cache.clone(),
            })
        }
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
            superblock_version: 0,
            file_size: self.data.len() as u64,
            offset_size: sb.size_of_offsets,
            length_size: sb.size_of_lengths,
        })
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

    /// Internal helper: resolve the object header address for the dataset at `path`.
    fn resolve_dataset_header_addr(&self, path: &str) -> Result<u64, OxiH5Error> {
        let sb = superblock::parse(&self.data)?;
        let root_msgs = header::parse_messages(&self.data, sb.root_object_header_address)?;
        let normalized = path.trim_start_matches('/');
        let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
        let dataset_name = parts
            .pop()
            .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

        if let Some((root_btree, root_heap)) = find_symbol_table_addresses(&root_msgs) {
            let (btree, heap) = if parts.is_empty() {
                (root_btree, root_heap)
            } else {
                navigate_to_group(&self.data, root_btree, root_heap, &parts)?
            };
            group::find_dataset(&self.data, btree, heap, dataset_name)
        } else {
            let mut current_header = sb.root_object_header_address;
            for segment in &parts {
                current_header = find_new_style_child(&self.data, current_header, segment)?;
            }
            resolve_new_style_header_address(
                &self.data,
                current_header,
                dataset_name,
                &self.source_dir,
            )
        }
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

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A handle to a group within an HDF5 file.
pub struct Group {
    /// Name of this group (last path segment, or `"/"` for root).
    pub name: String,
    object_header_address: u64,
    btree_address: u64,
    heap_address: u64,
    /// `true` when this is a new-style group that uses Link messages (0x0006)
    /// instead of the old-style Symbol Table / B-tree / SNOD mechanism.
    new_style: bool,
    file_data: FileData,
    /// Directory of the source file, used to resolve relative external link paths.
    source_dir: std::path::PathBuf,
    /// Pre-parsed chunk index cache shared with the parent `File`.
    chunk_cache: ChunkIndexCache,
}

impl Group {
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
        if self.new_style {
            return resolve_new_style_dataset(
                &self.file_data,
                self.object_header_address,
                name,
                &self.source_dir,
                Some(&self.chunk_cache),
            );
        }
        read_dataset_from_group(
            &self.file_data,
            self.btree_address,
            self.heap_address,
            name,
            Some(&self.chunk_cache),
        )
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
    fn list_entries_by_type(&self, want_groups: bool) -> Result<Vec<String>, OxiH5Error> {
        if self.new_style {
            return self.list_new_style_entries(want_groups);
        }

        let tree = btree::parse(&self.file_data, self.btree_address)?;
        let local_heap = heap::parse(&self.file_data, self.heap_address)?;
        // T6: pre-size with a reasonable hint (8 entries per leaf is typical).
        let mut names = Vec::with_capacity(tree.leaf_addresses.len() * 8);

        for &leaf_addr in &tree.leaf_addresses {
            let entries = snod::parse(&self.file_data, leaf_addr)?;
            for entry in entries {
                // Skip any entry with an undefined object header address.
                if entry.object_header_address == u64::MAX {
                    continue;
                }
                let raw_name = local_heap.name_at(entry.name_offset as usize)?;
                if raw_name.is_empty() {
                    continue;
                }
                let entry_name = raw_name.trim_start_matches('/').to_string();
                if entry_name.is_empty() {
                    continue;
                }

                // A group entry has a SymbolTable message (0x0011) in its header.
                let is_group = header::parse_messages(&self.file_data, entry.object_header_address)
                    .map(|msgs| msgs.iter().any(|m| m.msg_type == 0x0011))
                    .unwrap_or(false);

                if want_groups == is_group {
                    names.push(entry_name);
                }
            }
        }

        Ok(names)
    }

    /// New-style branch for `list_entries_by_type`: enumerate links from
    /// Link messages (0x0006) stored in the object header.
    fn list_new_style_entries(&self, want_groups: bool) -> Result<Vec<String>, OxiH5Error> {
        let sb = superblock::parse(&self.file_data)?;
        let ctx = oxih5_format::context::ParseContext::new(
            sb.size_of_offsets,
            sb.size_of_lengths,
            sb.base_address,
        );
        let links = group::list_new_style_links(&self.file_data, self.object_header_address, &ctx)?;
        let mut names = Vec::new();
        for pl in &links {
            if let oxih5_core::Link::Hard { address } = &pl.link {
                // Determine whether this hard link points to a group or a dataset.
                let is_group = group::is_new_style_group(&self.file_data, *address)
                    || header::parse_messages(&self.file_data, *address)
                        .map(|msgs| msgs.iter().any(|m| m.msg_type == 0x0011))
                        .unwrap_or(false);
                if is_group == want_groups {
                    names.push(pl.name.clone());
                }
            }
        }
        Ok(names)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract (btree_address, heap_address) from the first SymbolTable message
/// (0x0011) found in `messages`.
fn find_symbol_table_addresses(messages: &[header::Message]) -> Option<(u64, u64)> {
    for msg in messages {
        if msg.msg_type == 0x0011 {
            if let Ok(st) = message::parse_symbol_table(&msg.data) {
                return Some((st.btree_address, st.heap_address));
            }
        }
    }
    None
}

/// Navigate from a starting (btree, heap) through a list of group-path
/// segments and return the (btree, heap) of the final group.
fn navigate_to_group(
    file_data: &[u8],
    start_btree: u64,
    start_heap: u64,
    segments: &[&str],
) -> Result<(u64, u64), OxiH5Error> {
    let mut btree = start_btree;
    let mut heap = start_heap;

    for segment in segments {
        let entry_addr = group::find_dataset(file_data, btree, heap, segment)?;
        let msgs = header::parse_messages(file_data, entry_addr)?;
        let (new_btree, new_heap) = find_symbol_table_addresses(&msgs)
            .ok_or_else(|| OxiH5Error::NotFound(format!("'{}' is not a group", segment)))?;
        btree = new_btree;
        heap = new_heap;
    }

    Ok((btree, heap))
}

/// Find a child object (dataset or group) by name in a new-style group,
/// returning its object header address.
///
/// Scans Link messages (0x0006) in the parent's object header.
fn find_new_style_child(
    file_data: &[u8],
    parent_header_addr: u64,
    name: &str,
) -> Result<u64, OxiH5Error> {
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
                oxih5_core::Link::Hard { address } => return Ok(*address),
                oxih5_core::Link::Soft { path } => {
                    return Err(OxiH5Error::NotImplemented(format!(
                        "soft link '{}' → '{}' not followed",
                        name, path
                    )));
                }
                oxih5_core::Link::External { file, .. } => {
                    return Err(OxiH5Error::NotImplemented(format!(
                        "external link '{}' in file '{}' not followed",
                        name, file
                    )));
                }
            }
        }
    }
    Err(OxiH5Error::NotFound(name.to_string()))
}

/// Resolve a dataset name within a new-style group, handling both hard links
/// and external file links.
///
/// For hard links the dataset is read from the local file at the resolved
/// object header address.  For external links the referenced file is opened
/// and `File::dataset` is called with the target path stored in the link.
/// Soft links and group-type external links return `NotImplemented`.
fn resolve_new_style_dataset(
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
                    return read_dataset_from_object_header(file_data, *address, name, cache);
                }
                oxih5_core::Link::Soft { path } => {
                    return Err(OxiH5Error::NotImplemented(format!(
                        "soft link '{}' → '{}' not followed",
                        name, path
                    )));
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
fn resolve_external_link(
    ext_file: &str,
    ext_path: &str,
    source_dir: &std::path::Path,
) -> Result<Dataset, OxiH5Error> {
    let resolved = if std::path::Path::new(ext_file).is_absolute() {
        std::path::PathBuf::from(ext_file)
    } else {
        source_dir.join(ext_file)
    };

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

/// Lazy slice reader for `File::dataset_slice`: resolves the path, extracts
/// messages, and for chunked layouts calls `read_chunked_slice` directly.
fn read_dataset_slice_lazy(
    file_data: &FileData,
    path: &str,
    ranges: &[std::ops::Range<usize>],
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let sb = superblock::parse(file_data)?;
    let root_msgs = header::parse_messages(file_data, sb.root_object_header_address)?;

    let normalized = path.trim_start_matches('/');
    let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    let dataset_name = parts
        .pop()
        .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

    let header_addr = if let Some((root_btree, root_heap)) = find_symbol_table_addresses(&root_msgs)
    {
        let (btree, heap) = if parts.is_empty() {
            (root_btree, root_heap)
        } else {
            navigate_to_group(file_data, root_btree, root_heap, &parts)?
        };
        group::find_dataset(file_data, btree, heap, dataset_name)?
    } else {
        let mut current_header = sb.root_object_header_address;
        for segment in &parts {
            current_header = find_new_style_child(file_data, current_header, segment)?;
        }
        resolve_new_style_header_address(file_data, current_header, dataset_name, source_dir)?
    };

    slice_dataset_at_header(file_data, header_addr, dataset_name, ranges, Some(cache))
}

/// Lazy slice reader for `Group::dataset_slice`.
#[allow(clippy::too_many_arguments)]
fn read_dataset_slice_lazy_from_group(
    file_data: &FileData,
    object_header_address: u64,
    btree_address: u64,
    heap_address: u64,
    new_style: bool,
    name: &str,
    ranges: &[std::ops::Range<usize>],
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let header_addr = if new_style {
        resolve_new_style_header_address(file_data, object_header_address, name, source_dir)?
    } else {
        group::find_dataset(file_data, btree_address, heap_address, name)?
    };
    slice_dataset_at_header(file_data, header_addr, name, ranges, Some(cache))
}

/// Hyperslab reader for `File::dataset_hyperslab`.
fn read_dataset_hyperslab_internal(
    file_data: &FileData,
    path: &str,
    selection: &Hyperslab,
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let sb = superblock::parse(file_data)?;
    let root_msgs = header::parse_messages(file_data, sb.root_object_header_address)?;

    let normalized = path.trim_start_matches('/');
    let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    let dataset_name = parts
        .pop()
        .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

    let header_addr = if let Some((root_btree, root_heap)) = find_symbol_table_addresses(&root_msgs)
    {
        let (btree, heap) = if parts.is_empty() {
            (root_btree, root_heap)
        } else {
            navigate_to_group(file_data, root_btree, root_heap, &parts)?
        };
        group::find_dataset(file_data, btree, heap, dataset_name)?
    } else {
        let mut current_header = sb.root_object_header_address;
        for segment in &parts {
            current_header = find_new_style_child(file_data, current_header, segment)?;
        }
        resolve_new_style_header_address(file_data, current_header, dataset_name, source_dir)?
    };

    hyperslab_dataset_at_header(file_data, header_addr, dataset_name, selection, Some(cache))
}

/// Hyperslab reader for `Group::dataset_hyperslab`.
#[allow(clippy::too_many_arguments)]
fn read_dataset_hyperslab_from_group_internal(
    file_data: &FileData,
    object_header_address: u64,
    btree_address: u64,
    heap_address: u64,
    new_style: bool,
    name: &str,
    selection: &Hyperslab,
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let header_addr = if new_style {
        resolve_new_style_header_address(file_data, object_header_address, name, source_dir)?
    } else {
        group::find_dataset(file_data, btree_address, heap_address, name)?
    };
    hyperslab_dataset_at_header(file_data, header_addr, name, selection, Some(cache))
}

/// Parse messages at `header_addr` and perform a hyperslab read.
///
/// For chunked layouts with an in-bounds bounding box, only the overlapping
/// chunks are decompressed.  For other layouts (or when out-of-bounds) the full
/// data is loaded and then sampled with `gather_hyperslab_contiguous`.
fn hyperslab_dataset_at_header(
    file_data: &[u8],
    header_addr: u64,
    name: &str,
    selection: &Hyperslab,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    let ds_messages = header::parse_messages(file_data, header_addr)?;

    let mut dataspace = None;
    let mut datatype = None;
    let mut layout = None;
    let mut filter_pipeline = None;
    let mut fill_value: Option<Vec<u8>> = None;

    for msg in &ds_messages {
        match msg.msg_type {
            0x0001 => dataspace = Some(message::parse_dataspace(&msg.data)?),
            0x0003 => datatype = Some(message::parse_datatype(&msg.data)?),
            0x0005 => {
                if let Ok(fv) = message::parse_fill_value(&msg.data) {
                    fill_value = fv;
                }
            }
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

    use oxih5_format::message::LayoutInfo;

    // Attempt the lazy chunked-hyperslab path first.
    if let LayoutInfo::Chunked { .. } = &lay {
        let ndims = dsp.dims.len();
        if ndims >= 1 && selection.dims.len() == ndims {
            let bbox = selection.bounding_ranges();
            let all_in_bounds = bbox
                .iter()
                .zip(dsp.dims.iter())
                .all(|(r, &dim)| r.end <= dim);

            if all_in_bounds {
                let elem_size = dtp.dtype.size().ok_or_else(|| {
                    OxiH5Error::NotImplemented(format!(
                        "chunked dataset '{name}': variable-length element size not supported"
                    ))
                })?;
                let dataset_dims: Vec<u64> = dsp.dims.clone();
                let pipeline = filter_pipeline
                    .clone()
                    .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
                let out_shape: Vec<usize> = selection
                    .output_shape()
                    .iter()
                    .map(|&s| s as usize)
                    .collect();

                let raw = oxih5_format::chunked_hyperslab::read_chunked_hyperslab(
                    file_data,
                    &lay,
                    &pipeline,
                    &dataset_dims,
                    oxih5_format::chunked::ChunkSliceParams {
                        elem_size,
                        fill_value: fill_value.as_deref(),
                    },
                    selection,
                    cache,
                )?;

                let attributes =
                    read_attributes_from_header(file_data, header_addr).unwrap_or_default();
                return Ok(Dataset {
                    data: raw,
                    shape: out_shape,
                    dtype: dtp.dtype,
                    attributes,
                });
            }
        }
    }

    // Fallback: full read then gather via contiguous sampler.
    let full_ds = read_dataset_from_object_header(file_data, header_addr, name, cache)?;
    let dataset_dims_u64: Vec<u64> = full_ds.shape.iter().map(|&s| s as u64).collect();
    let elem_size = full_ds.dtype.size().ok_or_else(|| {
        OxiH5Error::NotImplemented(format!(
            "dataset '{name}': variable-length element size not supported for hyperslab fallback"
        ))
    })?;

    let raw = oxih5_format::chunked_hyperslab::gather_hyperslab_contiguous(
        &full_ds.data,
        &dataset_dims_u64,
        selection,
        elem_size,
    )?;
    let out_shape: Vec<usize> = selection
        .output_shape()
        .iter()
        .map(|&s| s as usize)
        .collect();
    Ok(Dataset {
        data: raw,
        shape: out_shape,
        dtype: full_ds.dtype,
        attributes: full_ds.attributes,
    })
}

/// Resolve the object header address for `name` within a new-style group.
fn resolve_new_style_header_address(
    file_data: &[u8],
    parent_header_addr: u64,
    name: &str,
    source_dir: &std::path::Path,
) -> Result<u64, OxiH5Error> {
    let sb = superblock::parse(file_data)?;
    let ctx = oxih5_format::context::ParseContext::new(
        sb.size_of_offsets,
        sb.size_of_lengths,
        sb.base_address,
    );
    let links = group::list_new_style_links(file_data, parent_header_addr, &ctx)?;
    for pl in &links {
        if pl.name == name {
            match &pl.link {
                oxih5_core::Link::Hard { address } => return Ok(*address),
                oxih5_core::Link::External {
                    file: ext_file,
                    path: ext_path,
                } => {
                    let resolved = if std::path::Path::new(ext_file).is_absolute() {
                        std::path::PathBuf::from(ext_file)
                    } else {
                        source_dir.join(ext_file)
                    };
                    let ext = open(&resolved).map_err(|e| {
                        OxiH5Error::NotFound(format!(
                            "external link target '{}': {e}",
                            resolved.display()
                        ))
                    })?;
                    let ds = ext.dataset(ext_path.trim_start_matches('/'))?;
                    return Err(OxiH5Error::NotImplemented(format!(
                        "lazy slice across external link not supported; loaded full dataset for '{}' (shape {:?})",
                        name, ds.shape
                    )));
                }
                _ => {
                    return Err(OxiH5Error::NotImplemented(format!(
                        "link type not supported for lazy slice of '{name}'"
                    )));
                }
            }
        }
    }
    Err(OxiH5Error::NotFound(name.to_string()))
}

/// Extract messages at `header_addr` and perform a lazy slice.
///
/// For chunked layouts only the overlapping chunks are decompressed.
/// For other layouts the full data is loaded and then sliced in memory.
fn slice_dataset_at_header(
    file_data: &[u8],
    header_addr: u64,
    name: &str,
    ranges: &[std::ops::Range<usize>],
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    let ds_messages = header::parse_messages(file_data, header_addr)?;

    let mut dataspace = None;
    let mut datatype = None;
    let mut layout = None;
    let mut filter_pipeline = None;
    let mut fill_value: Option<Vec<u8>> = None;

    for msg in &ds_messages {
        match msg.msg_type {
            0x0001 => dataspace = Some(message::parse_dataspace(&msg.data)?),
            0x0003 => datatype = Some(message::parse_datatype(&msg.data)?),
            0x0005 => {
                if let Ok(fv) = message::parse_fill_value(&msg.data) {
                    fill_value = fv;
                }
            }
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

    use oxih5_format::message::LayoutInfo;

    // Attempt lazy chunked slice first.
    if let LayoutInfo::Chunked { .. } = &lay {
        let ndims = dsp.dims.len();
        if ndims >= 1 && ranges.len() == ndims {
            let all_in_bounds = ranges
                .iter()
                .zip(dsp.dims.iter())
                .all(|(r, &dim)| r.end <= dim as usize);

            if all_in_bounds {
                let elem_size = dtp.dtype.size().ok_or_else(|| {
                    OxiH5Error::NotImplemented(format!(
                        "chunked dataset '{name}': variable-length element size not supported"
                    ))
                })?;
                let dataset_dims: Vec<u64> = dsp.dims.clone();
                let pipeline = filter_pipeline
                    .clone()
                    .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
                let ranges_u64: Vec<std::ops::Range<u64>> = ranges
                    .iter()
                    .map(|r| r.start as u64..r.end as u64)
                    .collect();
                let out_shape: Vec<usize> = ranges.iter().map(|r| r.len()).collect();

                let raw = oxih5_format::chunked::read_chunked_slice(
                    file_data,
                    &lay,
                    &pipeline,
                    &dataset_dims,
                    oxih5_format::chunked::ChunkSliceParams {
                        elem_size,
                        fill_value: fill_value.as_deref(),
                    },
                    &ranges_u64,
                    cache,
                )?;

                let attributes =
                    read_attributes_from_header(file_data, header_addr).unwrap_or_default();
                return Ok(Dataset {
                    data: raw,
                    shape: out_shape,
                    dtype: dtp.dtype,
                    attributes,
                });
            }
        }
    }

    // Fallback: full read then in-memory slice.
    let full_ds = read_dataset_from_object_header(file_data, header_addr, name, cache)?;
    full_ds.slice(ranges)
}

/// Read a dataset directly from its object header address (new-style groups).
///
/// This bypasses the B-tree/SNOD lookup because the address was already
/// resolved from a Link message.
fn read_dataset_from_object_header(
    file_data: &[u8],
    header_addr: u64,
    name: &str,
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
            let elem_size = dtp.dtype.size().ok_or_else(|| {
                OxiH5Error::NotImplemented(format!(
                    "chunked dataset '{name}': variable-length element size not supported"
                ))
            })?;
            let dataset_dims: Vec<u64> = dsp.dims.clone();
            let pipeline = filter_pipeline
                .clone()
                .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
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
        LayoutInfo::VirtualDataset { .. } => {
            return Err(OxiH5Error::NotImplemented(format!(
                "virtual dataset layout not yet supported for '{name}'"
            )));
        }
    };

    let attributes = read_attributes_from_header(file_data, header_addr).unwrap_or_default();

    Ok(Dataset {
        data: raw,
        shape,
        dtype: dtp.dtype,
        attributes,
    })
}

/// Core dataset-reading logic: finds `name` within the given group's B-tree,
/// parses its object header, and assembles a [`Dataset`].
fn read_dataset_from_group(
    file_data: &[u8],
    btree_address: u64,
    heap_address: u64,
    name: &str,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    // Locate the dataset's object header address via B-tree / SNOD.
    let ds_header_addr = group::find_dataset(file_data, btree_address, heap_address, name)?;

    // Parse the dataset's object header messages.
    let ds_messages = header::parse_messages(file_data, ds_header_addr)?;

    // Extract the essential messages: dataspace, datatype, layout, and the
    // (optional) filter pipeline needed to decode compressed chunks.
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

    // Extract raw data bytes based on layout class.
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
            // Chunked datasets need the element size to scatter chunks into the
            // output buffer; derive it from the datatype.
            let elem_size = dtp.dtype.size().ok_or_else(|| {
                OxiH5Error::NotImplemented(format!(
                    "chunked dataset '{name}': variable-length element size not supported"
                ))
            })?;
            let dataset_dims: Vec<u64> = dsp.dims.clone();
            let pipeline = filter_pipeline
                .clone()
                .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
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
        LayoutInfo::VirtualDataset { .. } => {
            return Err(OxiH5Error::NotImplemented(format!(
                "virtual dataset layout not yet supported for '{name}'"
            )));
        }
    };

    // Collect any attribute messages from the dataset's object header.
    let attributes = read_attributes_from_header(file_data, ds_header_addr).unwrap_or_default();

    Ok(Dataset {
        data: raw,
        shape,
        dtype: dtp.dtype,
        attributes,
    })
}

/// Parse all attribute messages (0x000C) from an object header and return
/// the decoded [`Attribute`] list.  Attributes that fail to parse are silently
/// skipped so that one malformed attribute does not abort the entire read.
fn read_attributes_from_header(
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
