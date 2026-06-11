//! HDF5 file writer — W0a/W0b/W0c attribute + group + chunked dataset support.
//!
//! Produces minimal, valid HDF5 files using superblock v0, old-style group
//! (B-tree v1 + SNOD + local heap), contiguous **and** chunked data layouts.
//!
//! Constraints:
//! - Up to 64 datasets + groups at root level; up to 32 datasets per sub-group
//! - No compression
//! - Supported element types: f32, f64, i32, i64, u8
//! - Attribute types: fixed-length string, f64, i64, i32, object-reference list

mod chunked;
mod format;
mod messages;

use oxih5_core::{Dtype, OxiH5Error};
use std::path::Path;

// ---------------------------------------------------------------------------
// SNOD capacity constants
// ---------------------------------------------------------------------------

/// Maximum number of root-level items (datasets + groups) per file.
const SNOD_CAPACITY: usize = 64;
/// Total size of the root SNOD in bytes.
const SNOD_SIZE: usize = 8 + SNOD_CAPACITY * 40; // 2568

/// Maximum number of datasets per sub-group.
const GROUP_SNOD_CAPACITY: usize = 32;
/// Total size of a group SNOD in bytes.
const GROUP_SNOD_SIZE: usize = 8 + GROUP_SNOD_CAPACITY * 40; // 1288

// ---------------------------------------------------------------------------
// Element-type enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) enum ElemType {
    F32,
    F64,
    I32,
    I64,
    U8,
    /// HDF5 variable-length string (class 9, subtype 1).
    ///
    /// Each element in the dataset is a 16-byte global-heap reference.
    VlenStr,
}

// ---------------------------------------------------------------------------
// Attribute kinds
// ---------------------------------------------------------------------------

pub(crate) enum AttrKind {
    FixedStr(String),
    F64(f64),
    I64(i64),
    I32(i32),
    ObjRefsByName(Vec<String>),
}

pub(crate) struct AttrDesc {
    pub(crate) name: String,
    pub(crate) kind: AttrKind,
}

enum ResolvedAttrKind<'a> {
    FixedStr(&'a str),
    F64(f64),
    I64(i64),
    I32(i32),
    ObjRefs(Vec<u64>),
}

struct ResolvedAttr<'a> {
    name: &'a str,
    kind: ResolvedAttrKind<'a>,
}

// ---------------------------------------------------------------------------
// Raw dataset specification (used to pass data to group helpers).
// ---------------------------------------------------------------------------

struct RawDatasetSpec {
    raw: Vec<u8>,
    shape: Vec<usize>,
    elem_type: ElemType,
    unlimited: bool,
    chunk_shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Dataset descriptor
// ---------------------------------------------------------------------------

pub(crate) struct DatasetDesc {
    pub(crate) name: String,
    pub(crate) raw: Vec<u8>,
    pub(crate) shape: Vec<usize>,
    pub(crate) elem_type: ElemType,
    pub(crate) attrs: Vec<AttrDesc>,
    /// W0c: if true, use chunked storage with max_dim[0]=u64::MAX.
    pub(crate) unlimited: bool,
    /// W0c: chunk dimensions (ndims values); used only when `unlimited = true`.
    pub(crate) chunk_shape: Vec<usize>,
    /// W0d: strings for a VLen-string dataset.  `None` for all other types.
    /// When `Some`, `elem_type` must be `ElemType::VlenStr`.
    pub(crate) vlen_strings: Option<Vec<String>>,
}

impl DatasetDesc {
    /// Byte size of the dataset's data area on disk.
    ///
    /// For VlenStr datasets this is `n_strings × 16` (one 16-byte global-heap
    /// reference per element); for all other datasets it is `raw.len()`.
    pub(crate) fn data_len(&self) -> usize {
        match &self.vlen_strings {
            Some(strings) => strings.len() * 16,
            None => self.raw.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Group descriptor (W0b)
// ---------------------------------------------------------------------------

pub(crate) struct GroupDesc {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetDesc>,
}

// ---------------------------------------------------------------------------
// FileWriter — public API
// ---------------------------------------------------------------------------

/// HDF5 file writer.
///
/// Supports:
/// - Up to 64 items (datasets + groups) at root level
/// - Contiguous and chunked (unlimited) datasets
/// - String, f64, i64, i32, and object-reference list attributes
/// - Root-group string attributes (for NetCDF-4 `_nc3_strict` etc.)
/// - Single-level sub-groups
///
/// # Example (builder pattern)
/// ```no_run
/// use oxih5::FileWriter;
/// let path = std::env::temp_dir().join("example.h5");
/// FileWriter::new()
///     .write_dataset_f32("data", &[1.0f32, 2.0, 3.0], &[3]).unwrap()
///     .build(&path)
///     .unwrap();
/// ```
pub struct FileWriter {
    datasets: Vec<DatasetDesc>,
    /// Root-group string attributes (e.g. `_nc3_strict` for NetCDF-4 classic mode).
    root_str_attrs: Vec<(String, String)>,
    /// W0b: sub-groups under the root.
    groups: Vec<GroupDesc>,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    /// Create a new, empty file writer.
    pub fn new() -> Self {
        Self {
            datasets: Vec::new(),
            root_str_attrs: Vec::new(),
            groups: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Root-level dataset methods
    // -----------------------------------------------------------------------

    /// Add a float32 dataset at root level.
    pub fn write_dataset_f32(
        &mut self,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::F32)
    }

    /// Add a float64 dataset at root level.
    pub fn write_dataset_f64(
        &mut self,
        name: &str,
        data: &[f64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::F64)
    }

    /// Add a signed int32 dataset at root level.
    pub fn write_dataset_i32(
        &mut self,
        name: &str,
        data: &[i32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::I32)
    }

    /// Add a signed int64 dataset at root level.
    pub fn write_dataset_i64(
        &mut self,
        name: &str,
        data: &[i64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::I64)
    }

    /// Add a uint8 dataset at root level.
    pub fn write_dataset_u8(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw = data.to_vec();
        self.add_dataset(name, raw, shape, ElemType::U8)
    }

    /// Add a zero-filled dataset of the given `dtype` at root level.
    pub fn create_dataset(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: &Dtype,
    ) -> Result<(), OxiH5Error> {
        let elem_size = dtype.size().ok_or_else(|| {
            OxiH5Error::Format("create_dataset: unsupported dtype (no fixed size)".to_string())
        })?;
        let n_elems: usize = shape.iter().product();
        let raw = vec![0u8; n_elems * elem_size];
        let et = dtype_to_elem_type(dtype)?;
        self.add_dataset(name, raw, shape, et)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // W0d: Variable-length string dataset
    // -----------------------------------------------------------------------

    /// Create a vlen-string (NC_STRING) dataset at root level.
    ///
    /// Each element in `strings` is stored as a NUL-terminated byte sequence
    /// in a Global Heap Collection (GCOL) appended to the file.  The dataset's
    /// data area holds one 16-byte global-heap reference per element.
    ///
    /// The written dataset has dtype = HDF5 class-9 VLen string and can be
    /// read back by [`crate::File::dataset_strings`].
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if the name is empty, contains `'/'`, or
    /// duplicates an existing dataset, or if the root SNOD capacity is full.
    pub fn create_vlen_string_dataset(
        &mut self,
        name: &str,
        strings: &[&str],
    ) -> Result<(), OxiH5Error> {
        if self.datasets.len() + self.groups.len() >= SNOD_CAPACITY {
            return Err(OxiH5Error::Format(format!(
                "FileWriter capacity exceeded: maximum {SNOD_CAPACITY} items at root"
            )));
        }
        if name.is_empty() {
            return Err(OxiH5Error::Format(
                "vlen-string dataset name must not be empty".to_string(),
            ));
        }
        if name.contains('/') {
            return Err(OxiH5Error::Format(format!(
                "vlen-string dataset name '{name}' must not contain '/'"
            )));
        }
        if self.datasets.iter().any(|d| d.name == name) {
            return Err(OxiH5Error::Format(format!(
                "duplicate dataset name '{name}'"
            )));
        }
        let n = strings.len();
        self.datasets.push(DatasetDesc {
            name: name.to_string(),
            raw: Vec::new(), // VlenStr datasets use vlen_strings, not raw
            shape: vec![n],
            elem_type: ElemType::VlenStr,
            attrs: Vec::new(),
            unlimited: false,
            chunk_shape: Vec::new(),
            vlen_strings: Some(strings.iter().map(|s| s.to_string()).collect()),
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // W0c: Unlimited / chunked dataset
    // -----------------------------------------------------------------------

    /// Create a chunked dataset with an unlimited first dimension.
    ///
    /// The initial data (provided as raw bytes) is stored as a single chunk.
    /// `shape` gives the initial dimensions; `chunk_shape` gives the chunk size
    /// (typically equal to `shape`).  `max_dims[0]` will be `u64::MAX`.
    ///
    /// Only one-dimensional or multi-dimensional datasets with the first axis
    /// unlimited are supported.  The `dtype` must be a fixed-size primitive type.
    pub fn create_dataset_unlimited(
        &mut self,
        name: &str,
        shape: &[usize],
        chunk_shape: &[usize],
        dtype: &Dtype,
        data: &[u8],
    ) -> Result<(), OxiH5Error> {
        if self.datasets.len() + self.groups.len() >= SNOD_CAPACITY {
            return Err(OxiH5Error::Format(format!(
                "FileWriter capacity exceeded: maximum {SNOD_CAPACITY} items at root"
            )));
        }
        if name.is_empty() || name.contains('/') {
            return Err(OxiH5Error::Format(format!("invalid dataset name '{name}'")));
        }
        if self.datasets.iter().any(|d| d.name == name) {
            return Err(OxiH5Error::Format(format!(
                "duplicate dataset name '{name}'"
            )));
        }
        let elem_size = dtype.size().ok_or_else(|| {
            OxiH5Error::Format("create_dataset_unlimited: unsupported dtype".to_string())
        })?;
        let n_elems: usize = shape.iter().product();
        if data.len() != n_elems * elem_size {
            return Err(OxiH5Error::Format(format!(
                "create_dataset_unlimited: data length {} != shape product {} * elem_size {}",
                data.len(),
                n_elems,
                elem_size
            )));
        }
        let et = dtype_to_elem_type(dtype)?;
        self.datasets.push(DatasetDesc {
            name: name.to_string(),
            raw: data.to_vec(),
            shape: shape.to_vec(),
            elem_type: et,
            attrs: Vec::new(),
            unlimited: true,
            chunk_shape: if chunk_shape.is_empty() {
                shape.to_vec()
            } else {
                chunk_shape.to_vec()
            },
            vlen_strings: None,
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Attribute writing
    // -----------------------------------------------------------------------

    /// Write a scalar fixed-length string attribute on an existing dataset.
    pub fn write_string_attr(
        &mut self,
        obj_path: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), OxiH5Error> {
        let ds = self.find_dataset_mut(obj_path)?;
        ds.attrs.push(AttrDesc {
            name: attr_name.to_string(),
            kind: AttrKind::FixedStr(value.to_string()),
        });
        Ok(())
    }

    /// Write a scalar float64 attribute on an existing dataset.
    pub fn write_f64_attr(
        &mut self,
        obj_path: &str,
        attr_name: &str,
        value: f64,
    ) -> Result<(), OxiH5Error> {
        let ds = self.find_dataset_mut(obj_path)?;
        ds.attrs.push(AttrDesc {
            name: attr_name.to_string(),
            kind: AttrKind::F64(value),
        });
        Ok(())
    }

    /// Write a scalar signed int64 attribute on an existing dataset.
    pub fn write_i64_attr(
        &mut self,
        obj_path: &str,
        attr_name: &str,
        value: i64,
    ) -> Result<(), OxiH5Error> {
        let ds = self.find_dataset_mut(obj_path)?;
        ds.attrs.push(AttrDesc {
            name: attr_name.to_string(),
            kind: AttrKind::I64(value),
        });
        Ok(())
    }

    /// Write a scalar signed int32 attribute on an existing dataset.
    pub fn write_i32_attr(
        &mut self,
        obj_path: &str,
        attr_name: &str,
        value: i32,
    ) -> Result<(), OxiH5Error> {
        let ds = self.find_dataset_mut(obj_path)?;
        ds.attrs.push(AttrDesc {
            name: attr_name.to_string(),
            kind: AttrKind::I32(value),
        });
        Ok(())
    }

    /// Write an object-reference list attribute on an existing dataset.
    pub fn write_obj_ref_list_attr(
        &mut self,
        obj_path: &str,
        attr_name: &str,
        target_names: &[&str],
    ) -> Result<(), OxiH5Error> {
        let ds = self.find_dataset_mut(obj_path)?;
        ds.attrs.push(AttrDesc {
            name: attr_name.to_string(),
            kind: AttrKind::ObjRefsByName(target_names.iter().map(|s| s.to_string()).collect()),
        });
        Ok(())
    }

    /// Write a string attribute on the root group.
    ///
    /// Used for NetCDF-4 global metadata such as `_nc3_strict`.
    pub fn write_root_str_attr(&mut self, name: &str, value: &str) {
        self.root_str_attrs
            .push((name.to_string(), value.to_string()));
    }

    // -----------------------------------------------------------------------
    // W0b: Sub-group creation
    // -----------------------------------------------------------------------

    /// Create a sub-group named `name` directly under the root group.
    ///
    /// Returns `OxiH5Error::Format` if the name is invalid or the group already
    /// exists.  Only single-level groups (no '/' in name) are supported.
    pub fn create_group(&mut self, name: &str) -> Result<(), OxiH5Error> {
        if name.is_empty() || name.contains('/') {
            return Err(OxiH5Error::Format(format!("invalid group name '{name}'")));
        }
        if self.datasets.len() + self.groups.len() >= SNOD_CAPACITY {
            return Err(OxiH5Error::Format(format!(
                "FileWriter capacity exceeded: maximum {SNOD_CAPACITY} items at root"
            )));
        }
        if self.groups.iter().any(|g| g.name == name) {
            return Err(OxiH5Error::Format(format!("duplicate group name '{name}'")));
        }
        if self.datasets.iter().any(|d| d.name == name) {
            return Err(OxiH5Error::Format(format!(
                "a dataset named '{name}' already exists"
            )));
        }
        self.groups.push(GroupDesc {
            name: name.to_string(),
            datasets: Vec::new(),
        });
        Ok(())
    }

    /// Add a float64 dataset to the named sub-group.
    pub fn write_group_dataset_f64(
        &mut self,
        group: &str,
        name: &str,
        data: &[f64],
        shape: &[usize],
    ) -> Result<(), OxiH5Error> {
        let spec = RawDatasetSpec {
            raw: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            elem_type: ElemType::F64,
            unlimited: false,
            chunk_shape: vec![],
        };
        self.add_dataset_to_group(group, name, spec)
    }

    /// Add an int32 dataset to the named sub-group.
    pub fn write_group_dataset_i32(
        &mut self,
        group: &str,
        name: &str,
        data: &[i32],
        shape: &[usize],
    ) -> Result<(), OxiH5Error> {
        let spec = RawDatasetSpec {
            raw: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            elem_type: ElemType::I32,
            unlimited: false,
            chunk_shape: vec![],
        };
        self.add_dataset_to_group(group, name, spec)
    }

    /// Write a string attribute on a dataset inside a named group.
    ///
    /// `group_path` is the group name; `obj_name` is the dataset name within
    /// that group.
    pub fn write_group_string_attr(
        &mut self,
        group_path: &str,
        obj_name: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), OxiH5Error> {
        let grp = self.find_group_mut(group_path)?;
        let ds = grp
            .datasets
            .iter_mut()
            .find(|d| d.name == obj_name)
            .ok_or_else(|| {
                OxiH5Error::NotFound(format!("dataset '{obj_name}' not in group '{group_path}'"))
            })?;
        ds.attrs.push(AttrDesc {
            name: attr_name.to_string(),
            kind: AttrKind::FixedStr(value.to_string()),
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    /// Write the HDF5 file to disk.
    pub fn build(&mut self, path: impl AsRef<Path>) -> Result<(), OxiH5Error> {
        let bytes = self.build_bytes()?;
        std::fs::write(path, &bytes).map_err(OxiH5Error::Io)
    }

    /// Serialize the HDF5 file into a byte vector without writing to disk.
    pub fn build_to_vec(&mut self) -> Result<Vec<u8>, OxiH5Error> {
        self.build_bytes()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn build_bytes(&mut self) -> Result<Vec<u8>, OxiH5Error> {
        let n_root_ds = self.datasets.len();
        let n_groups = self.groups.len();

        // ----------------------------------------------------------------
        // 1. Build root local heap data (root dataset names + group names).
        // ----------------------------------------------------------------
        let mut heap_used: Vec<u8> = vec![0u8; 8]; // offset 0 reserved (free block link)
        let mut root_name_offsets: Vec<u64> = Vec::with_capacity(n_root_ds + n_groups);

        for ds in &self.datasets {
            root_name_offsets.push(heap_used.len() as u64);
            heap_used.extend_from_slice(ds.name.as_bytes());
            heap_used.push(0);
            let cur = heap_used.len();
            heap_used.resize((cur + 7) & !7, 0);
        }
        let mut grp_name_offsets_root: Vec<u64> = Vec::with_capacity(n_groups);
        for grp in &self.groups {
            grp_name_offsets_root.push(heap_used.len() as u64);
            heap_used.extend_from_slice(grp.name.as_bytes());
            heap_used.push(0);
            let cur = heap_used.len();
            heap_used.resize((cur + 7) & !7, 0);
        }

        let used_size = heap_used.len();
        let heap_data_size = ((used_size + 16 + 7) & !7).max(88);
        let mut heap_data = vec![0u8; heap_data_size];
        heap_data[..used_size].copy_from_slice(&heap_used);
        let free_size = heap_data_size - used_size;
        heap_data[used_size..used_size + 8].copy_from_slice(&1u64.to_le_bytes());
        heap_data[used_size + 8..used_size + 16].copy_from_slice(&(free_size as u64).to_le_bytes());

        // ----------------------------------------------------------------
        // 2. Compute dynamic addresses.
        // ----------------------------------------------------------------
        let root_oh_size = format::compute_root_oh_size(&self.root_str_attrs);
        let btree_addr = 96 + root_oh_size;
        let heap_hdr_addr = btree_addr + format::BTREE_LEAF_SIZE;
        let heap_data_addr = heap_hdr_addr + format::HEAP_HEADER_SIZE;
        let snod_addr = heap_data_addr + heap_data_size;

        // ----------------------------------------------------------------
        // 3. Compute addresses for root datasets.
        // ----------------------------------------------------------------
        let mut root_oh_addrs: Vec<usize> = Vec::with_capacity(n_root_ds);
        let mut root_btree_addrs: Vec<usize> = Vec::with_capacity(n_root_ds);
        let mut root_data_addrs: Vec<usize> = Vec::with_capacity(n_root_ds);
        let mut current = snod_addr + SNOD_SIZE;

        for ds in &self.datasets {
            root_oh_addrs.push(current);
            let oh_sz =
                messages::compute_oh_size(ds.shape.len(), &ds.elem_type, &ds.attrs, ds.unlimited);
            current += oh_sz;
            if ds.unlimited {
                root_btree_addrs.push(current);
                current += chunked::chunk_btree_size(ds.shape.len());
            } else {
                root_btree_addrs.push(0); // unused for contiguous
            }
            root_data_addrs.push(current);
            current += ds.data_len();
            current = (current + 7) & !7;
        }

        // ----------------------------------------------------------------
        // 4. Compute addresses for groups and their datasets.
        // ----------------------------------------------------------------
        let mut grp_oh_addrs: Vec<usize> = Vec::with_capacity(n_groups);
        let mut grp_btree_addrs: Vec<usize> = Vec::with_capacity(n_groups);
        let mut grp_heap_hdr_addrs: Vec<usize> = Vec::with_capacity(n_groups);
        let mut grp_heap_data_addrs: Vec<usize> = Vec::with_capacity(n_groups);
        let mut grp_heap_sizes: Vec<usize> = Vec::with_capacity(n_groups);
        let mut grp_heap_datas: Vec<Vec<u8>> = Vec::with_capacity(n_groups);
        let mut grp_name_offs: Vec<Vec<u64>> = Vec::with_capacity(n_groups); // per-group ds name offsets
        let mut grp_snod_addrs: Vec<usize> = Vec::with_capacity(n_groups);
        let mut grp_ds_oh_addrs: Vec<Vec<usize>> = Vec::with_capacity(n_groups);
        let mut grp_ds_btree_addrs: Vec<Vec<usize>> = Vec::with_capacity(n_groups);
        let mut grp_ds_data_addrs: Vec<Vec<usize>> = Vec::with_capacity(n_groups);

        for grp in &self.groups {
            grp_oh_addrs.push(current);
            current += format::GROUP_OH_SIZE;

            grp_btree_addrs.push(current);
            current += format::BTREE_LEAF_SIZE;

            grp_heap_hdr_addrs.push(current);
            current += format::HEAP_HEADER_SIZE;

            // Build group local heap
            let mut g_heap: Vec<u8> = vec![0u8; 8];
            let mut g_name_offs: Vec<u64> = Vec::new();
            for ds in &grp.datasets {
                g_name_offs.push(g_heap.len() as u64);
                g_heap.extend_from_slice(ds.name.as_bytes());
                g_heap.push(0);
                let cur = g_heap.len();
                g_heap.resize((cur + 7) & !7, 0);
            }
            let g_used = g_heap.len();
            let g_heap_sz = ((g_used + 16 + 7) & !7).max(88);
            let mut g_heap_data = vec![0u8; g_heap_sz];
            g_heap_data[..g_used].copy_from_slice(&g_heap);
            let g_free = g_heap_sz - g_used;
            g_heap_data[g_used..g_used + 8].copy_from_slice(&1u64.to_le_bytes());
            g_heap_data[g_used + 8..g_used + 16].copy_from_slice(&(g_free as u64).to_le_bytes());

            grp_heap_data_addrs.push(current);
            current += g_heap_sz;
            grp_heap_sizes.push(g_heap_sz);
            grp_heap_datas.push(g_heap_data);
            grp_name_offs.push(g_name_offs);

            grp_snod_addrs.push(current);
            current += GROUP_SNOD_SIZE;

            // Group's datasets
            let mut ds_oh: Vec<usize> = Vec::new();
            let mut ds_bt: Vec<usize> = Vec::new();
            let mut ds_da: Vec<usize> = Vec::new();

            for ds in &grp.datasets {
                ds_oh.push(current);
                let oh_sz = messages::compute_oh_size(
                    ds.shape.len(),
                    &ds.elem_type,
                    &ds.attrs,
                    ds.unlimited,
                );
                current += oh_sz;
                if ds.unlimited {
                    ds_bt.push(current);
                    current += chunked::chunk_btree_size(ds.shape.len());
                } else {
                    ds_bt.push(0);
                }
                ds_da.push(current);
                current += ds.data_len();
                current = (current + 7) & !7;
            }
            grp_ds_oh_addrs.push(ds_oh);
            grp_ds_btree_addrs.push(ds_bt);
            grp_ds_data_addrs.push(ds_da);
        }

        // ----------------------------------------------------------------
        // 4.5  W0d — Build Global Heap Collection for vlen-string datasets.
        // ----------------------------------------------------------------
        // Collect all strings from VlenStr datasets (root + groups) into one
        // shared GCOL.  Record the 1-based GCOL object index for each string
        // so that we can write the vlen references later.
        let mut gcol_writer = oxih5_format::GlobalHeapWriter::new();

        let root_vlen_obj_idx: Vec<Vec<u32>> = self
            .datasets
            .iter()
            .map(|ds| match &ds.vlen_strings {
                Some(strings) => strings
                    .iter()
                    .map(|s| gcol_writer.write_string(s))
                    .collect(),
                None => Vec::new(),
            })
            .collect();

        let grp_vlen_obj_idx: Vec<Vec<Vec<u32>>> = self
            .groups
            .iter()
            .map(|grp| {
                grp.datasets
                    .iter()
                    .map(|ds| match &ds.vlen_strings {
                        Some(strings) => strings
                            .iter()
                            .map(|s| gcol_writer.write_string(s))
                            .collect(),
                        None => Vec::new(),
                    })
                    .collect()
            })
            .collect();

        let gcol_bytes = if gcol_writer.is_empty() {
            Vec::new()
        } else {
            gcol_writer.build()
        };
        let gcol_addr = current; // GCOL follows all structured data
        let eof_addr = current + gcol_bytes.len();

        // ----------------------------------------------------------------
        // 5. Resolve ObjRefsByName attributes.
        // ----------------------------------------------------------------
        // Build name → OH address map (root datasets only for now)
        let name_to_addr: std::collections::HashMap<&str, u64> = self
            .datasets
            .iter()
            .zip(root_oh_addrs.iter())
            .map(|(ds, &addr)| (ds.name.as_str(), addr as u64))
            .collect();

        let mut root_resolved_refs: Vec<Vec<(String, Vec<u64>)>> = vec![Vec::new(); n_root_ds];
        for (i, ds) in self.datasets.iter().enumerate() {
            for attr in &ds.attrs {
                if let AttrKind::ObjRefsByName(names) = &attr.kind {
                    let addrs: Vec<u64> = names
                        .iter()
                        .map(|nm| name_to_addr.get(nm.as_str()).copied().unwrap_or(u64::MAX))
                        .collect();
                    root_resolved_refs[i].push((attr.name.clone(), addrs));
                }
            }
        }

        // ----------------------------------------------------------------
        // 6. Allocate buffer and write everything.
        // ----------------------------------------------------------------
        let mut buf = vec![0u8; eof_addr];

        let btree_key1 = root_name_offsets
            .iter()
            .chain(grp_name_offsets_root.iter())
            .last()
            .copied()
            .unwrap_or(0);

        format::write_signature(&mut buf);
        format::write_superblock(&mut buf, btree_addr, heap_hdr_addr, eof_addr as u64);
        format::write_root_oh(
            &mut buf,
            btree_addr,
            heap_hdr_addr,
            &self.root_str_attrs,
            root_oh_size,
        );
        format::write_btree_leaf(&mut buf, btree_addr, snod_addr as u64, btree_key1);
        format::write_local_heap(
            &mut buf,
            heap_hdr_addr,
            heap_data_addr,
            heap_data_size,
            used_size,
        );
        buf[heap_data_addr..heap_data_addr + heap_data.len()].copy_from_slice(&heap_data);

        format::write_snod(
            &mut buf,
            snod_addr,
            &root_name_offsets,
            &root_oh_addrs,
            &grp_name_offsets_root,
            &grp_oh_addrs,
            &grp_btree_addrs,
            &grp_heap_hdr_addrs,
        );

        // Root datasets
        for (i, ds) in self.datasets.iter().enumerate() {
            let mut resolved_attrs: Vec<ResolvedAttr<'_>> = Vec::new();
            let mut ref_idx = 0usize;
            for attr in &ds.attrs {
                match &attr.kind {
                    AttrKind::FixedStr(s) => resolved_attrs.push(ResolvedAttr {
                        name: &attr.name,
                        kind: ResolvedAttrKind::FixedStr(s.as_str()),
                    }),
                    AttrKind::F64(v) => resolved_attrs.push(ResolvedAttr {
                        name: &attr.name,
                        kind: ResolvedAttrKind::F64(*v),
                    }),
                    AttrKind::I64(v) => resolved_attrs.push(ResolvedAttr {
                        name: &attr.name,
                        kind: ResolvedAttrKind::I64(*v),
                    }),
                    AttrKind::I32(v) => resolved_attrs.push(ResolvedAttr {
                        name: &attr.name,
                        kind: ResolvedAttrKind::I32(*v),
                    }),
                    AttrKind::ObjRefsByName(_) => {
                        if ref_idx < root_resolved_refs[i].len() {
                            resolved_attrs.push(ResolvedAttr {
                                name: &attr.name,
                                kind: ResolvedAttrKind::ObjRefs(
                                    root_resolved_refs[i][ref_idx].1.clone(),
                                ),
                            });
                            ref_idx += 1;
                        }
                    }
                }
            }

            let data_addr = root_data_addrs[i] as u64;
            let btree_a = root_btree_addrs[i] as u64;
            messages::write_dataset_oh(
                &mut buf,
                root_oh_addrs[i],
                ds,
                data_addr,
                btree_a,
                &resolved_attrs,
            );

            if let Some(strings) = &ds.vlen_strings {
                // W0d: write 16-byte vlen references into the dataset data area.
                let obj_idxs = &root_vlen_obj_idx[i];
                let base = root_data_addrs[i];
                for (j, (&obj_idx, s)) in obj_idxs.iter().zip(strings.iter()).enumerate() {
                    write_vlen_ref(
                        &mut buf,
                        base + j * 16,
                        s.len() as u32 + 1,
                        obj_idx,
                        gcol_addr as u64,
                    );
                }
            } else if ds.unlimited {
                let bt_base = root_btree_addrs[i];
                chunked::write_chunk_btree(
                    &mut buf,
                    bt_base,
                    ds.shape.len(),
                    data_addr,
                    ds.raw.len(),
                );
                let raw_end = root_data_addrs[i] + ds.raw.len();
                buf[root_data_addrs[i]..raw_end].copy_from_slice(&ds.raw);
            } else {
                let raw_end = root_data_addrs[i] + ds.raw.len();
                buf[root_data_addrs[i]..raw_end].copy_from_slice(&ds.raw);
            }
        }

        // Groups
        for (gi, grp) in self.groups.iter().enumerate() {
            let grp_snod = grp_snod_addrs[gi];
            let grp_heap_hdr = grp_heap_hdr_addrs[gi];
            let grp_heap_data = grp_heap_data_addrs[gi];
            let grp_bt = grp_btree_addrs[gi];
            let grp_oh = grp_oh_addrs[gi];

            format::write_group_oh(&mut buf, grp_oh, grp_bt, grp_heap_hdr);

            let grp_key1 = grp_name_offs[gi].last().copied().unwrap_or(0);
            format::write_btree_leaf(&mut buf, grp_bt, grp_snod as u64, grp_key1);
            format::write_local_heap(
                &mut buf,
                grp_heap_hdr,
                grp_heap_data,
                grp_heap_sizes[gi],
                grp_name_offs[gi]
                    .last()
                    .map(|&o| {
                        // used_size = offset of last name + len(last_name_bytes_aligned)
                        // We tracked grp_heap_datas; just use the heap's used portion
                        // The heap data was built with exactly `g_used` bytes used.
                        // We need to store g_used somewhere... let me recompute from heap data.
                        // Actually, we stored the free-list pointer at g_used, so:
                        // free_list_offset = used_size = position of the first free block.
                        // We wrote it in heap_data[g_used..g_used+8] = 1 (free list link)
                        // but we need to pass used_size to write_local_heap.
                        // Workaround: find the free list offset from the heap data we stored.
                        let _ = o; // suppress warning
                        0usize // will be corrected below
                    })
                    .unwrap_or(0),
            );

            // Fix: recompute grp_used_size from grp_heap_datas
            // We need to know the used portion of the group heap.
            // We stored it implicitly. Let's track it: recompute from grp_name_offs.
            // The heap data starts with 8 zero bytes (offset 0 reserved), then name entries.
            // `g_used` = heap_used.len() at the end of the name-building loop.
            // We can recompute it: initial=8, then for each ds add aligned(name+1).
            let mut g_used_recomputed = 8usize;
            for ds in &grp.datasets {
                g_used_recomputed += (ds.name.len() + 1 + 7) & !7;
            }

            // Rewrite the local heap with correct used_size
            format::write_local_heap(
                &mut buf,
                grp_heap_hdr,
                grp_heap_data,
                grp_heap_sizes[gi],
                g_used_recomputed,
            );

            // Copy group heap data
            buf[grp_heap_data..grp_heap_data + grp_heap_datas[gi].len()]
                .copy_from_slice(&grp_heap_datas[gi]);

            // Group SNOD (datasets only, no sub-groups within sub-groups)
            format::write_snod(
                &mut buf,
                grp_snod,
                &grp_name_offs[gi],
                &grp_ds_oh_addrs[gi],
                &[],
                &[],
                &[],
                &[],
            );

            // Group datasets
            for (di, ds) in grp.datasets.iter().enumerate() {
                let resolved_attrs: Vec<ResolvedAttr<'_>> = ds
                    .attrs
                    .iter()
                    .filter_map(|attr| {
                        match &attr.kind {
                            AttrKind::FixedStr(s) => Some(ResolvedAttr {
                                name: &attr.name,
                                kind: ResolvedAttrKind::FixedStr(s.as_str()),
                            }),
                            AttrKind::F64(v) => Some(ResolvedAttr {
                                name: &attr.name,
                                kind: ResolvedAttrKind::F64(*v),
                            }),
                            AttrKind::I64(v) => Some(ResolvedAttr {
                                name: &attr.name,
                                kind: ResolvedAttrKind::I64(*v),
                            }),
                            AttrKind::I32(v) => Some(ResolvedAttr {
                                name: &attr.name,
                                kind: ResolvedAttrKind::I32(*v),
                            }),
                            AttrKind::ObjRefsByName(_) => None, // not resolved in groups for simplicity
                        }
                    })
                    .collect();

                let da = grp_ds_data_addrs[gi][di] as u64;
                let bta = grp_ds_btree_addrs[gi][di] as u64;
                messages::write_dataset_oh(
                    &mut buf,
                    grp_ds_oh_addrs[gi][di],
                    ds,
                    da,
                    bta,
                    &resolved_attrs,
                );

                if let Some(strings) = &ds.vlen_strings {
                    // W0d: write vlen references for group VlenStr datasets.
                    let obj_idxs = &grp_vlen_obj_idx[gi][di];
                    let base = grp_ds_data_addrs[gi][di];
                    for (j, (&obj_idx, s)) in obj_idxs.iter().zip(strings.iter()).enumerate() {
                        write_vlen_ref(
                            &mut buf,
                            base + j * 16,
                            s.len() as u32 + 1,
                            obj_idx,
                            gcol_addr as u64,
                        );
                    }
                } else if ds.unlimited {
                    let bt_base = grp_ds_btree_addrs[gi][di];
                    chunked::write_chunk_btree(&mut buf, bt_base, ds.shape.len(), da, ds.raw.len());
                    let raw_end = grp_ds_data_addrs[gi][di] + ds.raw.len();
                    buf[grp_ds_data_addrs[gi][di]..raw_end].copy_from_slice(&ds.raw);
                } else {
                    let raw_end = grp_ds_data_addrs[gi][di] + ds.raw.len();
                    buf[grp_ds_data_addrs[gi][di]..raw_end].copy_from_slice(&ds.raw);
                }
            }
        }

        // ----------------------------------------------------------------
        // 7. Write GCOL at the end of the file (W0d).
        // ----------------------------------------------------------------
        if !gcol_bytes.is_empty() {
            let gcol_end = gcol_addr + gcol_bytes.len();
            buf[gcol_addr..gcol_end].copy_from_slice(&gcol_bytes);
        }

        Ok(buf)
    }

    fn find_dataset_mut(&mut self, obj_path: &str) -> Result<&mut DatasetDesc, OxiH5Error> {
        let name = obj_path.trim_start_matches('/');
        self.datasets
            .iter_mut()
            .find(|ds| ds.name == name)
            .ok_or_else(|| OxiH5Error::NotFound(format!("dataset '{obj_path}' not found")))
    }

    fn find_group_mut(&mut self, group_name: &str) -> Result<&mut GroupDesc, OxiH5Error> {
        let name = group_name.trim_start_matches('/');
        self.groups
            .iter_mut()
            .find(|g| g.name == name)
            .ok_or_else(|| OxiH5Error::NotFound(format!("group '{group_name}' not found")))
    }

    fn add_dataset(
        &mut self,
        name: &str,
        raw: Vec<u8>,
        shape: &[usize],
        elem_type: ElemType,
    ) -> Result<&mut Self, OxiH5Error> {
        if self.datasets.len() + self.groups.len() >= SNOD_CAPACITY {
            return Err(OxiH5Error::Format(format!(
                "FileWriter capacity exceeded: maximum {SNOD_CAPACITY} items at root"
            )));
        }
        if name.is_empty() {
            return Err(OxiH5Error::Format(
                "dataset name must not be empty".to_string(),
            ));
        }
        if name.contains('/') {
            return Err(OxiH5Error::Format(format!(
                "dataset name '{name}' must not contain '/'"
            )));
        }
        if self.datasets.iter().any(|d| d.name == name) {
            return Err(OxiH5Error::Format(format!(
                "duplicate dataset name '{name}'"
            )));
        }
        self.datasets.push(DatasetDesc {
            name: name.to_string(),
            raw,
            shape: shape.to_vec(),
            elem_type,
            attrs: Vec::new(),
            unlimited: false,
            chunk_shape: Vec::new(),
            vlen_strings: None,
        });
        Ok(self)
    }

    fn add_dataset_to_group(
        &mut self,
        group: &str,
        name: &str,
        spec: RawDatasetSpec,
    ) -> Result<(), OxiH5Error> {
        let grp = self.find_group_mut(group)?;
        if grp.datasets.len() >= GROUP_SNOD_CAPACITY {
            return Err(OxiH5Error::Format(format!(
                "group '{group}' capacity exceeded: maximum {GROUP_SNOD_CAPACITY} datasets"
            )));
        }
        if name.is_empty() || name.contains('/') {
            return Err(OxiH5Error::Format(format!("invalid dataset name '{name}'")));
        }
        if grp.datasets.iter().any(|d| d.name == name) {
            return Err(OxiH5Error::Format(format!(
                "duplicate dataset name '{name}' in group '{group}'"
            )));
        }
        grp.datasets.push(DatasetDesc {
            name: name.to_string(),
            raw: spec.raw,
            shape: spec.shape,
            elem_type: spec.elem_type,
            attrs: Vec::new(),
            unlimited: spec.unlimited,
            chunk_shape: spec.chunk_shape,
            vlen_strings: None,
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// W0d helpers
// ---------------------------------------------------------------------------

/// Write a 16-byte HDF5 vlen reference into `buf` at `offset`.
///
/// Layout:
/// ```text
/// [0–3]:  seq_len  (u32 LE) — string byte count including NUL terminator
/// [4–5]:  obj_idx  (u16 LE) — 1-based GCOL object index
/// [6–7]:  reserved (u16 LE) = 0
/// [8–15]: heap_addr (u64 LE) — absolute address of the GCOL in the file
/// ```
fn write_vlen_ref(buf: &mut [u8], offset: usize, seq_len: u32, obj_idx: u32, heap_addr: u64) {
    buf[offset..offset + 4].copy_from_slice(&seq_len.to_le_bytes());
    buf[offset + 4..offset + 6].copy_from_slice(&(obj_idx as u16).to_le_bytes());
    // bytes [6..8] = reserved, already zero in the buffer
    buf[offset + 8..offset + 16].copy_from_slice(&heap_addr.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Dtype → ElemType
// ---------------------------------------------------------------------------

fn dtype_to_elem_type(dtype: &Dtype) -> Result<ElemType, OxiH5Error> {
    match dtype {
        Dtype::Float { size: 4, .. } => Ok(ElemType::F32),
        Dtype::Float { size: 8, .. } => Ok(ElemType::F64),
        Dtype::Int {
            size: 4,
            signed: true,
            ..
        } => Ok(ElemType::I32),
        Dtype::Int {
            size: 8,
            signed: true,
            ..
        } => Ok(ElemType::I64),
        Dtype::Int {
            size: 1,
            signed: false,
            ..
        } => Ok(ElemType::U8),
        _ => Err(OxiH5Error::Format(format!("unsupported dtype {dtype:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Tests — W0a round-trip (preserved) + W0b + W0c + C11 infrastructure
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::File;

    #[test]
    fn write_string_attr_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_write_str_attr.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("x", &[1.0, 2.0, 3.0, 4.0], &[4])
            .expect("write");
        w.write_string_attr("x", "units", "km").expect("attr");
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let attrs = f.attr_views("x").expect("attr_views");
        let a = attrs.iter().find(|a| a.name() == "units").expect("units");
        assert_eq!(a.as_str_fixed().expect("as_str_fixed"), "km");
    }

    #[test]
    fn write_f64_attr_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_write_f64_attr.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f32("data", &[0.0f32; 4], &[4])
            .expect("write");
        w.write_f64_attr("data", "scale_factor", 2.5).expect("attr");
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let attrs = f.attr_views("data").expect("attr_views");
        let sf = attrs
            .iter()
            .find(|a| a.name() == "scale_factor")
            .expect("sf");
        let v = sf.as_f64().expect("as_f64");
        assert!((v - 2.5).abs() < 1e-15, "expected 2.5 got {v}");
    }

    #[test]
    fn write_i32_attr_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_write_i32_attr.h5");
        let mut w = FileWriter::new();
        w.write_dataset_i32("ds", &[1i32, 2, 3], &[3])
            .expect("write");
        w.write_i32_attr("ds", "_Netcdf4Dimid", 7).expect("attr");
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let attrs = f.attr_views("ds").expect("attr_views");
        let a = attrs
            .iter()
            .find(|a| a.name() == "_Netcdf4Dimid")
            .expect("dimid");
        assert_eq!(a.as_i64(), Some(7));
    }

    #[test]
    fn write_i64_attr_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_write_i64_attr.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("big", &[0.0; 2], &[2]).expect("write");
        w.write_i64_attr("big", "count", i64::MAX).expect("attr");
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let attrs = f.attr_views("big").expect("attr_views");
        let a = attrs.iter().find(|a| a.name() == "count").expect("count");
        assert_eq!(a.as_i64(), Some(i64::MAX));
    }

    #[test]
    fn write_multiple_attrs_on_same_dataset() {
        let tmp = std::env::temp_dir().join("oxih5_test_multi_attr.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("temp", &[20.0, 21.0], &[2])
            .expect("write");
        w.write_string_attr("temp", "units", "degC").expect("units");
        w.write_string_attr("temp", "long_name", "Surface Temperature")
            .expect("long_name");
        w.write_f64_attr("temp", "_FillValue", 9.969_209_968_386_869e36)
            .expect("fillval");
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let attrs = f.attr_views("temp").expect("attr_views");
        assert!(attrs.iter().any(|a| a.name() == "units"));
        assert!(attrs.iter().any(|a| a.name() == "long_name"));
        assert!(attrs.iter().any(|a| a.name() == "_FillValue"));
    }

    #[test]
    fn write_obj_ref_list_attr_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_objref_attr.h5");
        let mut w = FileWriter::new();
        w.write_dataset_i32("lat", &[0i32, 1, 2], &[3])
            .expect("lat");
        w.write_string_attr("lat", "CLASS", "DIMENSION_SCALE")
            .expect("CLASS");
        w.write_dataset_f64("temp", &[0.0; 3], &[3]).expect("temp");
        w.write_obj_ref_list_attr("temp", "DIMENSION_LIST", &["lat"])
            .expect("dim_list");
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let lat_addr = f.header_addr_of("lat").expect("lat addr");
        let attrs = f.attr_views("temp").expect("attr_views temp");
        let dl = attrs
            .iter()
            .find(|a| a.name() == "DIMENSION_LIST")
            .expect("dl");
        let refs = dl.as_object_refs().expect("refs");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], lat_addr);
    }

    #[test]
    fn attr_on_unknown_dataset_returns_error() {
        let mut w = FileWriter::new();
        let result = w.write_string_attr("nonexistent", "key", "val");
        assert!(result.is_err());
    }

    #[test]
    fn nc_file_writer_simulation() {
        let tmp = std::env::temp_dir().join("oxih5_test_ncfw_sim.h5");
        let mut w = FileWriter::new();
        w.write_dataset_i32("lat", &[0i32, 1, 2, 3], &[4]).unwrap();
        w.write_string_attr("lat", "CLASS", "DIMENSION_SCALE")
            .unwrap();
        w.write_string_attr("lat", "NAME", "lat").unwrap();
        w.write_i32_attr("lat", "_Netcdf4Dimid", 0).unwrap();
        w.write_dataset_i32("lon", &[0i32, 1, 2, 3, 4, 5, 6, 7], &[8])
            .unwrap();
        w.write_string_attr("lon", "CLASS", "DIMENSION_SCALE")
            .unwrap();
        w.write_string_attr("lon", "NAME", "lon").unwrap();
        w.write_i32_attr("lon", "_Netcdf4Dimid", 1).unwrap();
        w.write_dataset_f64("temp", &[0.0f64; 32], &[4, 8]).unwrap();
        w.write_obj_ref_list_attr("temp", "DIMENSION_LIST", &["lat", "lon"])
            .unwrap();
        w.build(&tmp).unwrap();
        let f = File::open(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let lat_addr = f.header_addr_of("lat").unwrap();
        let lon_addr = f.header_addr_of("lon").unwrap();
        let lat_attrs = f.attr_views("lat").unwrap();
        let class_attr = lat_attrs.iter().find(|a| a.name() == "CLASS");
        assert!(class_attr.is_some());
        assert_eq!(
            class_attr
                .unwrap()
                .as_str_fixed()
                .unwrap_or_default()
                .trim(),
            "DIMENSION_SCALE"
        );
        let temp_attrs = f.attr_views("temp").unwrap();
        let dl = temp_attrs.iter().find(|a| a.name() == "DIMENSION_LIST");
        assert!(dl.is_some());
        let refs = dl.unwrap().as_object_refs().expect("refs");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0], lat_addr);
        assert_eq!(refs[1], lon_addr);
    }

    #[test]
    fn existing_write_tests_still_pass_with_64_capacity() {
        let tmp = std::env::temp_dir().join("oxih5_test_64cap.h5");
        let mut w = FileWriter::new();
        for i in 0..10usize {
            w.write_dataset_f64(&format!("ds{i}"), &[i as f64], &[1])
                .expect("write");
        }
        w.build(&tmp).expect("build");
        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let names = f.dataset_names().expect("names");
        assert_eq!(names.len(), 10);
    }

    // -----------------------------------------------------------------------
    // C11 infrastructure: root group string attributes
    // -----------------------------------------------------------------------

    #[test]
    fn root_str_attr_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_root_str_attr.h5");
        let mut w = FileWriter::new();
        w.write_root_str_attr("_nc3_strict", "");
        w.write_dataset_f64("data", &[1.0, 2.0], &[2]).unwrap();
        w.build(&tmp).unwrap();

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        // Root group attrs via root().attr_views()
        let root = f.root().expect("root");
        let attrs = root.attr_views().expect("root_attr_views");
        let nc3 = attrs.iter().find(|a| a.name() == "_nc3_strict");
        assert!(nc3.is_some(), "_nc3_strict not found on root group");
    }

    #[test]
    fn root_str_attr_does_not_break_dataset_reads() {
        let tmp = std::env::temp_dir().join("oxih5_test_root_attr_ds.h5");
        let mut w = FileWriter::new();
        w.write_root_str_attr("convention", "CF-1.8");
        w.write_dataset_f32("pressure", &[101.3f32, 99.8], &[2])
            .unwrap();
        w.write_string_attr("pressure", "units", "hPa").unwrap();
        w.build(&tmp).unwrap();

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let ds = f.dataset("pressure").expect("pressure");
        assert_eq!(ds.shape, vec![2usize]);
        let attrs = f.attr_views("pressure").expect("pressure attrs");
        assert!(attrs.iter().any(|a| a.name() == "units"));
    }

    // -----------------------------------------------------------------------
    // W0b: Sub-group round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn w0b_create_group_and_dataset_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0b_group.h5");
        let mut w = FileWriter::new();

        // Root dataset
        w.write_dataset_f64("root_ds", &[1.0, 2.0], &[2]).unwrap();

        // Sub-group with one dataset
        w.create_group("sensors").unwrap();
        w.write_group_dataset_f64("sensors", "temperature", &[22.5, 23.0, 21.8], &[3])
            .unwrap();
        w.write_group_string_attr("sensors", "temperature", "units", "degC")
            .unwrap();

        w.build(&tmp).unwrap();

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        // Root dataset should still be readable
        let root_ds = f.dataset("root_ds").expect("root_ds");
        assert_eq!(root_ds.shape, vec![2usize]);

        // Navigate to sub-group
        let grp = f.group("sensors").expect("sensors group");
        let names = grp.datasets().expect("group datasets");
        assert!(
            names.iter().any(|n| n == "temperature"),
            "temperature not in sensors: {names:?}"
        );

        // Read dataset via path
        let temp = f
            .dataset("sensors/temperature")
            .expect("sensors/temperature");
        assert_eq!(temp.shape, vec![3usize]);
    }

    #[test]
    fn w0b_group_groups_listing() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0b_groups.h5");
        let mut w = FileWriter::new();
        w.create_group("grp1").unwrap();
        w.create_group("grp2").unwrap();
        w.write_group_dataset_i32("grp1", "x", &[1i32, 2], &[2])
            .unwrap();
        w.build(&tmp).unwrap();

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let root = f.root().expect("root");
        let group_names = root.groups().expect("root groups");
        assert!(
            group_names.iter().any(|n| n == "grp1"),
            "grp1 missing: {group_names:?}"
        );
        assert!(
            group_names.iter().any(|n| n == "grp2"),
            "grp2 missing: {group_names:?}"
        );
    }

    // -----------------------------------------------------------------------
    // W0c: Unlimited / chunked dataset round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn w0c_unlimited_dataset_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0c_unlimited.h5");

        let data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let dtype = Dtype::Float {
            size: 8,
            order: oxih5_core::ByteOrder::Little,
        };
        let mut w = FileWriter::new();
        w.create_dataset_unlimited("time_series", &[10], &[10], &dtype, &raw)
            .unwrap();
        w.build(&tmp).unwrap();

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let ds = f.dataset("time_series").expect("time_series");
        assert_eq!(ds.shape, vec![10usize]);
        assert!(ds.is_unlimited(), "expected unlimited dim 0");

        let vals = ds.as_f64().expect("as_f64");
        assert_eq!(vals.len(), 10);
        for (i, &v) in vals.iter().enumerate() {
            assert!((v - i as f64 * 0.5).abs() < 1e-15, "mismatch at {i}: {v}");
        }
    }

    #[test]
    fn w0c_2d_unlimited_roundtrip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0c_2d_unlimited.h5");

        // Shape: [3, 4] — unlimited on dim 0
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dtype = Dtype::Float {
            size: 4,
            order: oxih5_core::ByteOrder::Little,
        };
        let mut w = FileWriter::new();
        w.create_dataset_unlimited("grid", &[3, 4], &[3, 4], &dtype, &raw)
            .unwrap();
        w.build(&tmp).unwrap();

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let ds = f.dataset("grid").expect("grid");
        assert_eq!(ds.shape, vec![3usize, 4]);
        assert!(ds.is_unlimited(), "expected unlimited");
    }

    // -----------------------------------------------------------------------
    // W0d: GlobalHeap / vlen-string dataset round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn w0d_gcol_round_trip() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0d_gcol.h5");
        let mut w = FileWriter::new();
        w.create_vlen_string_dataset("strs", &["hello", "world", "foo"])
            .expect("create_vlen_string_dataset");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let result = f.dataset_strings("strs").expect("dataset_strings");
        assert_eq!(result, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn w0d_gcol_empty_string() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0d_empty_str.h5");
        let mut w = FileWriter::new();
        w.create_vlen_string_dataset("s", &["", "non-empty", ""])
            .expect("create_vlen_string_dataset");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let result = f.dataset_strings("s").expect("dataset_strings");
        assert_eq!(result, vec!["", "non-empty", ""]);
    }

    #[test]
    fn w0d_gcol_with_coexisting_numeric_dataset() {
        let tmp = std::env::temp_dir().join("oxih5_test_w0d_mixed.h5");
        let mut w = FileWriter::new();
        w.write_dataset_f64("nums", &[1.0, 2.0, 3.0], &[3])
            .expect("nums");
        w.create_vlen_string_dataset("labels", &["alpha", "beta", "gamma"])
            .expect("labels");
        w.build(&tmp).expect("build");

        let f = File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        // Numeric dataset still readable.
        let nums = f.dataset("nums").expect("nums");
        let vals = nums.as_f64().expect("as_f64");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);

        // Vlen string dataset readable.
        let labels = f.dataset_strings("labels").expect("labels");
        assert_eq!(labels, vec!["alpha", "beta", "gamma"]);
    }
}
