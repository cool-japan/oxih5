#![deny(unsafe_code)]
//! # OxiH5 — Pure-Rust HDF5 reader/writer (no libhdf5 FFI)
//!
//! Open an HDF5 file with [`open`] and read datasets by path, or build a new
//! file with [`FileWriter`].
//!
//! ## Write then read a dataset
//!
//! ```
//! use oxih5::FileWriter;
//!
//! // Write a 2×3 float64 dataset to a temporary file.
//! let path = std::env::temp_dir().join("oxih5_doctest_readme.h5");
//! let mut writer = FileWriter::new();
//! writer.write_dataset_f64("matrix", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
//! writer.build(&path)?;
//!
//! // Read it back and decode the values.
//! let file = oxih5::open(&path)?;
//! let ds = file.dataset("matrix")?;
//! assert_eq!(ds.shape, vec![2, 3]);
//! assert_eq!(ds.as_f64()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! # let _ = std::fs::remove_file(&path);
//! # Ok::<(), oxih5::OxiH5Error>(())
//! ```
//!
//! ## Read a hyperslab (sub-region) of a dataset
//!
//! ```
//! use oxih5::FileWriter;
//!
//! let path = std::env::temp_dir().join("oxih5_doctest_slice.h5");
//! let mut writer = FileWriter::new();
//! writer.write_dataset_i32("row", &[10, 11, 12, 13, 14], &[5])?;
//! writer.build(&path)?;
//!
//! let file = oxih5::open(&path)?;
//! // Select elements [1, 4).
//! let ranges: Vec<std::ops::Range<usize>> = std::iter::once(1..4).collect();
//! let slice = file.dataset_slice("row", &ranges)?;
//! assert_eq!(slice.as_i32()?, vec![11, 12, 13]);
//!
//! # let _ = std::fs::remove_file(&path);
//! # Ok::<(), oxih5::OxiH5Error>(())
//! ```

pub use oxih5_core::{Attribute, ByteOrder, Dataset, Dtype, OxiH5Error};
pub use oxih5_format::values::Value;
pub use oxih5_format::{DimSelection, Hyperslab};

mod write;
pub use write::FileWriter;

mod attr_view;
pub use attr_view::AttrView;

mod links;

mod file;
pub use file::File;

mod group_handle;
pub use group_handle::Group;

mod inplace;
pub use inplace::{
    dataset_data_extent, write_dataset_in_place, write_dataset_in_place_f32,
    write_dataset_in_place_f64, write_dataset_in_place_i16, write_dataset_in_place_i32,
    write_dataset_in_place_i64, write_dataset_in_place_i8, write_dataset_in_place_u16,
    write_dataset_in_place_u32, write_dataset_in_place_u64, write_dataset_in_place_u8, DataExtent,
};

mod reader;
mod slicing;

use oxih5_format::ChunkIndexCache;

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
    /// The actual superblock version parsed from the file (0, 1, 2, or 3).
    pub superblock_version: u8,
    /// Total byte size of the file as loaded into memory.
    pub file_size: u64,
    /// `size_of_offsets` field from the superblock (typically 8).
    pub offset_size: u8,
    /// `size_of_lengths` field from the superblock (typically 8).
    pub length_size: u8,
    /// Address of the superblock extension object header, if the file has one.
    ///
    /// Only superblock versions 2 and 3 can carry a superblock extension;
    /// versions 0 and 1 always report `None`.
    pub superblock_extension_address: Option<u64>,
}

// The superblock-extension message types (`SuperblockExtension`,
// `BtreeKValues`) and their decoders live in `oxih5_format::superblock`
// alongside the rest of the superblock parsing.  They are re-exported here so
// that callers use them as `oxih5::SuperblockExtension` / `oxih5::BtreeKValues`.
pub use oxih5_format::superblock::{BtreeKValues, SuperblockExtension};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::links::{resolve_path_to_header, GroupRef, VisitedLinks};

    // -----------------------------------------------------------------------
    // Superblock extension: File-level accessor
    // -----------------------------------------------------------------------

    /// A file written by [`FileWriter`] uses superblock v0, which has no
    /// superblock extension; `info()` must report version 0 and no extension,
    /// and `superblock_extension()` must return `Ok(None)`.
    #[test]
    fn test_superblock_extension_none_on_v0_file() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxih5_test_sb_ext_none.h5");

        crate::write::FileWriter::new()
            .write_dataset_f32("d", &[1.0f32, 2.0, 3.0], &[3])
            .expect("write")
            .build(&tmp)
            .expect("build");

        let file = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let info = file.info().expect("info");
        assert_eq!(info.superblock_version, 0);
        assert_eq!(info.superblock_extension_address, None);

        let ext = file.superblock_extension().expect("superblock_extension");
        assert!(ext.is_none(), "v0 file must have no superblock extension");
    }

    // -----------------------------------------------------------------------
    // A1 — integration: DataspaceInfo max_dims parsing + Dataset::is_unlimited
    // -----------------------------------------------------------------------

    /// Verify that `parse_dataspace` extracts `max_dims` correctly.
    #[test]
    fn test_parse_dataspace_with_max_dims() {
        use oxih5_format::message::parse_dataspace;

        // Build a v1 dataspace body with flags=0x01 (max-dims present)
        // dims: [10, 20], max_dims: [u64::MAX, 50]
        let mut body = vec![0u8; 8 + 8 * 2 + 8 * 2]; // header + dims + max_dims
        body[0] = 1; // version
        body[1] = 2; // dimensionality
        body[2] = 0x01; // flags: max-dims present
                        // body[3..8] reserved
                        // dims
        body[8..16].copy_from_slice(&10u64.to_le_bytes());
        body[16..24].copy_from_slice(&20u64.to_le_bytes());
        // max_dims
        body[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        body[32..40].copy_from_slice(&50u64.to_le_bytes());

        let dsp = parse_dataspace(&body).expect("parse_dataspace failed");
        assert_eq!(dsp.dims, vec![10u64, 20u64]);
        assert_eq!(dsp.max_dims, Some(vec![u64::MAX, 50u64]));
    }

    /// Verify that `parse_dataspace` with flags=0x00 produces `max_dims: None`.
    #[test]
    fn test_parse_dataspace_without_max_dims() {
        use oxih5_format::message::parse_dataspace;

        let mut body = vec![0u8; 8 + 8]; // v1 header + 1 dim
        body[0] = 1; // version
        body[1] = 1; // dimensionality
        body[2] = 0x00; // flags: no max-dims
        body[8..16].copy_from_slice(&42u64.to_le_bytes());

        let dsp = parse_dataspace(&body).expect("parse_dataspace failed");
        assert_eq!(dsp.dims, vec![42u64]);
        assert!(dsp.max_dims.is_none());
    }

    // -----------------------------------------------------------------------
    // A2 — File::attrs_of: unit tests using in-crate write infrastructure
    // -----------------------------------------------------------------------

    /// Verify that `attrs_of(addr)` returns empty attrs for a dataset that
    /// has no attributes.
    #[test]
    fn test_attrs_of_returns_empty_when_no_attrs() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxih5_test_attrs_of_empty.h5");

        crate::write::FileWriter::new()
            .write_dataset_f32("mydata", &[1.0f32, 2.0, 3.0], &[3])
            .expect("write_dataset_f32")
            .build(&tmp)
            .expect("build");

        let file = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        // Resolve the header address using the private helper (accessible in
        // crate-internal tests).
        let addr = file
            .resolve_dataset_header_addr("mydata")
            .expect("resolve header addr");

        let attrs = file.attrs_of(addr).expect("attrs_of");
        assert!(
            attrs.is_empty(),
            "no attrs were written; expected empty, got {:?}",
            attrs.iter().map(|a| &a.name).collect::<Vec<_>>()
        );
    }

    /// Verify that `attrs_of(u64::MAX)` returns NotFound.
    #[test]
    fn test_attrs_of_undefined_ref_returns_not_found() {
        let file = File::open_from_bytes(&[0x89u8, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a])
            .unwrap_or_else(|_| File::open_from_bytes(&[0u8; 64]).unwrap());
        // Even without a valid file, the addr check happens before any parsing.
        let result = file.attrs_of(u64::MAX);
        assert!(
            matches!(result, Err(OxiH5Error::NotFound(_))),
            "expected NotFound for u64::MAX, got {:?}",
            result
        );
    }

    // Existing tests below.

    /// Test that link resolution returns an error when the same
    /// `(group, path)` pair is visited twice (cycle detection).
    #[test]
    fn test_soft_link_cycle_detection() {
        // We test the cycle-guard directly without a real HDF5 file: the guard
        // must fire before any file parsing happens.  A *relative* path is used
        // so resolution starts from `base` rather than re-reading the
        // superblock to find the root group.
        let file_data = vec![0u8; 64];
        let base = GroupRef::new(7, None);
        let mut visited = VisitedLinks::new();
        // Pre-insert the pair to simulate having already been here.
        visited.insert((base.header, "cyclic".to_string()));

        let result = resolve_path_to_header(&file_data, base, "cyclic", &mut visited);
        assert!(result.is_err(), "expected cycle-detection error, got Ok");
        let err_str = match result {
            Err(e) => e.to_string(),
            Ok(addr) => panic!("expected an error, got address {addr}"),
        };
        assert!(
            err_str.contains("cycle"),
            "error message should mention 'cycle', got: {err_str}"
        );
    }

    /// The cycle guard is keyed by `(group, path)`, not by path alone: the same
    /// *relative* link value in two different groups names two different
    /// objects and must not be mistaken for a loop.
    #[test]
    fn test_same_relative_path_in_two_groups_is_not_a_cycle() {
        let file_data = vec![0u8; 64];
        let mut visited = VisitedLinks::new();
        visited.insert((7u64, "inner".to_string()));

        // A different group with the same relative path: not a cycle, so the
        // guard lets it through and resolution proceeds (and then fails on the
        // bogus file bytes, which is a *different* error).
        let other = GroupRef::new(99, None);
        let err = match resolve_path_to_header(&file_data, other, "inner", &mut visited) {
            Err(e) => e.to_string(),
            Ok(addr) => panic!("expected an error from the dummy file, got address {addr}"),
        };
        assert!(
            !err.contains("cycle"),
            "must not be reported as a cycle, got: {err}"
        );
    }

    /// An empty target path names the group resolution starts from.
    #[test]
    fn test_soft_link_empty_path_returns_base_group() {
        // We do not need a valid file here: an empty path returns before any
        // parsing.
        let file_data = vec![0u8; 64];
        let mut visited = VisitedLinks::new();
        let base = GroupRef::new(42, None);
        let result = resolve_path_to_header(&file_data, base, "", &mut visited);
        assert_eq!(result.expect("empty path resolves to the base group"), 42);
    }
}
