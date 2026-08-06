//! `FileWriter` dataset-creation entry points.
//!
//! Every method here takes a *path* and appends a [`DatasetDesc`] to the
//! plan tree; they differ only in the [`ElemType`] handed to `add_dataset`,
//! which is where all per-type knowledge lives.  See [`super`] for the
//! writer-wide overview and the shared size contract.

use oxih5_core::{ByteOrder, Dtype, OxiH5Error};

use super::elem::{
    dtype_to_elem_type, AttrDesc, AttrKind, ElemType, NumType, COMPACT_LAYOUT_SENTINEL_NAME,
    FILL_VALUE_SENTINEL_NAME,
};
use super::tree::{dataset_mut, insertion_point, DatasetDesc, Storage};
use super::{checked_byte_len, narrow, pad8, FileWriter};

impl FileWriter {
    // -----------------------------------------------------------------------
    // Dataset methods
    //
    // Every one of these takes a path: `"x"` is a root dataset, `"/a/b/x"` is a
    // dataset in `a/b` and creates both groups on the way.  They differ only in
    // the `ElemType` they hand to `add_dataset`, which is where all per-type
    // knowledge lives.
    // -----------------------------------------------------------------------

    /// Add a float32 dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is empty or malformed, if its
    /// final name is already used by a dataset or a group, or if `data.len()`
    /// does not equal the product of `shape`.
    pub fn write_dataset_f32(
        &mut self,
        path: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::F32)
    }

    /// Add a float64 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_f64(
        &mut self,
        path: &str,
        data: &[f64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::F64)
    }

    /// Add a signed int32 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_i32(
        &mut self,
        path: &str,
        data: &[i32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::I32)
    }

    /// Add a signed int64 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_i64(
        &mut self,
        path: &str,
        data: &[i64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::I64)
    }

    /// Add a uint8 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_u8(
        &mut self,
        path: &str,
        data: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw = data.to_vec();
        self.add_dataset(path, raw, shape, ElemType::U8)
    }

    /// Add a signed int8 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_i8(
        &mut self,
        path: &str,
        data: &[i8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::I8)
    }

    /// Add a signed int16 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_i16(
        &mut self,
        path: &str,
        data: &[i16],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::I16)
    }

    /// Add a uint16 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_u16(
        &mut self,
        path: &str,
        data: &[u16],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::U16)
    }

    /// Add a uint32 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_u32(
        &mut self,
        path: &str,
        data: &[u32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::U32)
    }

    /// Add a uint64 dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`].
    pub fn write_dataset_u64(
        &mut self,
        path: &str,
        data: &[u64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(path, raw, shape, ElemType::U64)
    }

    /// Add a zero-filled dataset of the given `dtype` at `path`.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `dtype` has no fixed size, is
    /// big-endian, or is otherwise outside the writer's element-type set; if the
    /// shape product or its byte size overflows `usize`; and for the same path
    /// conditions as [`Self::write_dataset_f32`].
    pub fn create_dataset(
        &mut self,
        path: &str,
        shape: &[usize],
        dtype: &Dtype,
    ) -> Result<(), OxiH5Error> {
        let elem_size = dtype.size().ok_or_else(|| {
            OxiH5Error::Format("create_dataset: unsupported dtype (no fixed size)".to_string())
        })?;
        // Overflow-checked, matching `add_dataset`: a shape whose product would
        // wrap `usize` is a typed error here, not a debug-profile multiply panic
        // (or a release wraparound that would allocate a too-small buffer).
        let byte_len = checked_byte_len(&format!("create_dataset('{path}')"), shape, elem_size)?;
        let raw = vec![0u8; byte_len];
        let et = dtype_to_elem_type(dtype)?;
        self.add_dataset(path, raw, shape, et)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // W0d: Variable-length string dataset
    // -----------------------------------------------------------------------

    /// Create a vlen-string (NC_STRING) dataset at `path`.
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
    /// Returns `OxiH5Error::Format` if `path` is empty or malformed, or if its
    /// final name is already used by a dataset or a group.
    pub fn create_vlen_string_dataset(
        &mut self,
        path: &str,
        strings: &[&str],
    ) -> Result<(), OxiH5Error> {
        let n = strings.len();
        let values: Vec<String> = strings.iter().map(|s| (*s).to_string()).collect();
        let (parent, name) = insertion_point(&mut self.root, path, "vlen-string dataset")?;
        parent.push_dataset(DatasetDesc {
            name: name.to_string(),
            raw: Vec::new(), // VlenStr datasets use vlen_strings, not raw
            shape: vec![n],
            elem_type: ElemType::VlenStr,
            dtype: None,
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: Some(values),
            vlen_seqs: None,
            creation_order: 0,
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // G003: Fixed-length string dataset
    // -----------------------------------------------------------------------

    /// Create a fixed-length string (NC_CHAR / numpy `S<width>`) dataset at
    /// `path`.
    ///
    /// Every element occupies exactly `width` bytes, NUL-padded, stored inline
    /// in the dataset's contiguous data area under an ASCII class-3 string
    /// datatype — the shape h5py reads back as a numpy `S<width>` (raw bytes)
    /// array and oxih5's own [`crate::File::dataset_strings`] decodes to
    /// `String`s.  This is the classic NetCDF/HDF5 idiom for station names,
    /// category labels and time strings.
    ///
    /// String payloads are stored as their raw (UTF-8) bytes: a multi-byte UTF-8
    /// string that fits `width` round-trips byte-for-byte, and an empty string
    /// is all-NUL.  The datatype declares ASCII purely to match libhdf5's numpy
    /// `S<width>` convention — the bytes are identical either way.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `width` is `0` or does not fit the 32-bit
    /// on-disk size field, if any string is **longer** than `width` bytes
    /// (truncation is refused, never silent), if the byte size overflows
    /// `usize`, or for the same path conditions as [`Self::write_dataset_f32`].
    pub fn create_fixed_string_dataset(
        &mut self,
        path: &str,
        width: usize,
        strings: &[&str],
    ) -> Result<(), OxiH5Error> {
        // Everything that can be rejected is rejected before the path is
        // resolved: resolving it creates intermediate groups, and a refused
        // dataset must leave no trace of itself in the file.
        if width == 0 {
            return Err(OxiH5Error::Format(format!(
                "create_fixed_string_dataset('{path}'): width must be at least 1 \
                 (a zero-width string datatype is not representable)"
            )));
        }
        let width_field: u32 = narrow("fixed-string width", width)?;
        let n = strings.len();
        let byte_len = checked_byte_len(
            &format!("create_fixed_string_dataset('{path}')"),
            &[n],
            width,
        )?;
        let mut raw = vec![0u8; byte_len];
        for (i, s) in strings.iter().enumerate() {
            let bytes = s.as_bytes();
            if bytes.len() > width {
                return Err(OxiH5Error::Format(format!(
                    "create_fixed_string_dataset('{path}'): string {i} is {} bytes, \
                     over the fixed width {width} — widen the dataset or shorten the \
                     string; truncation is refused",
                    bytes.len()
                )));
            }
            let slot = i * width;
            raw[slot..slot + bytes.len()].copy_from_slice(bytes);
        }

        let (parent, name) = insertion_point(&mut self.root, path, "fixed-string dataset")?;
        parent.push_dataset(DatasetDesc {
            name: name.to_string(),
            raw,
            shape: vec![n],
            elem_type: ElemType::FixedStr(width_field),
            dtype: None,
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: None,
            vlen_seqs: None,
            creation_order: 0,
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // G009: Boolean dataset (H5T_ENUM over i8, the h5py/netCDF-C convention)
    // -----------------------------------------------------------------------

    /// Add a boolean dataset at `path`, stored the way libhdf5/h5py store one: a
    /// class-8 enumeration `{ FALSE = 0, TRUE = 1 }` over a signed 8-bit base
    /// type, one `i8` per element.
    ///
    /// h5py reads the dataset back as a numpy `bool` array — the round-trip
    /// producers of boolean QA/quality masks (very common in geospatial and
    /// remote-sensing data) need in order to write their masks back as booleans
    /// rather than as plain integers.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is empty or malformed, if its final
    /// name is already used by a dataset or a group, or if the element count
    /// overflows `usize`.
    pub fn write_dataset_bool(
        &mut self,
        path: &str,
        data: &[bool],
    ) -> Result<&mut Self, OxiH5Error> {
        // Each element is one enum byte: FALSE = 0, TRUE = 1.
        let raw: Vec<u8> = data.iter().map(|&b| u8::from(b)).collect();
        self.add_dataset(path, raw, &[data.len()], ElemType::Bool)
    }

    // -----------------------------------------------------------------------
    // W0c: Unlimited / chunked dataset
    // -----------------------------------------------------------------------

    /// Create a chunked dataset with an unlimited first dimension at `path`.
    ///
    /// `shape` gives the initial dimensions and `max_dims[0]` becomes
    /// `u64::MAX`.  `chunk_shape` is the tile size in elements: the dataset is
    /// cut into `ceil(shape[d] / chunk_shape[d])` chunks along each dimension,
    /// and a chunk that hangs over an edge is stored full-size with the fill
    /// value in the overhang, exactly as libhdf5 stores one.
    ///
    /// A **shorter** `chunk_shape` is completed from `shape`, so `&[]` means
    /// "one chunk, the whole dataset" and `&[2]` over a shape of `[6, 4]` means
    /// `[2, 4]`.  That completion is why passing one extent for a variable of
    /// any rank — which is what `oxinetcdf` does — declares a geometry that
    /// matches what is actually stored.
    ///
    /// Only one-dimensional or multi-dimensional datasets with the first axis
    /// unlimited are supported.  The `dtype` must be a fixed-size primitive type.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `dtype` has no fixed size, is
    /// big-endian, or is otherwise outside the writer's element-type set; if the
    /// shape product or its byte size overflows `usize`; if `data.len()` does not
    /// match `shape` × the element size; if `chunk_shape`
    /// holds a 0, the dataset has no dimensions at all, or the tiling needs
    /// more than 2²⁴ chunks — the last three are reported by [`Self::build`],
    /// where the geometry is resolved; or for the same path conditions as
    /// [`Self::write_dataset_f32`].
    pub fn create_dataset_unlimited(
        &mut self,
        path: &str,
        shape: &[usize],
        chunk_shape: &[usize],
        dtype: &Dtype,
        data: &[u8],
    ) -> Result<(), OxiH5Error> {
        // Everything that can be rejected is rejected before the path is
        // resolved: resolving it creates intermediate groups, and a refused
        // dataset must leave no trace of itself in the file.
        let elem_size = dtype.size().ok_or_else(|| {
            OxiH5Error::Format("create_dataset_unlimited: unsupported dtype".to_string())
        })?;
        // Overflow-checked: without this, `shape.iter().product()` panics in a
        // debug build and wraps in release — and a wrap to a smaller `expected`
        // (e.g. 0) makes the length check below *accept* data that does not match
        // the declared shape, carrying a silently inconsistent descriptor into
        // the layout pass. The checked form rejects the overflow outright.
        let expected = checked_byte_len(
            &format!("create_dataset_unlimited('{path}')"),
            shape,
            elem_size,
        )?;
        if data.len() != expected {
            return Err(OxiH5Error::Format(format!(
                "create_dataset_unlimited('{path}'): data length {} != shape {shape:?} \
                 × elem_size {elem_size} = {expected}",
                data.len(),
            )));
        }
        let et = dtype_to_elem_type(dtype)?;

        let (parent, name) = insertion_point(&mut self.root, path, "dataset")?;
        parent.push_dataset(DatasetDesc {
            name: name.to_string(),
            raw: data.to_vec(),
            shape: shape.to_vec(),
            elem_type: et,
            dtype: None,
            attrs: Vec::new(),
            storage: Storage::Chunked {
                // Kept verbatim, short vector and all: completing it needs the
                // shape, and `chunked::chunk_shape_of` is the one place that
                // knows how.  Storing a completed copy here would be a second
                // answer to the same question.
                chunk_shape: chunk_shape.to_vec(),
                unlimited_dim0: true,
            },
            filter: None,
            vlen_strings: None,
            vlen_seqs: None,
            creation_order: 0,
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // G014: Custom dataset fill value
    //
    // A custom fill value is attached to an already-created dataset and folded
    // into its object header's fill-value message at build time (see
    // `write/oh.rs`).  Each typed entry point serialises its value little-endian
    // and states the element type it belongs to; the shared helper refuses a
    // value whose type does not match the dataset's, so an `f64` fill cannot be
    // set on an `i32` dataset.  With no custom fill the dataset keeps the
    // historic "defined, zero-length" fill message, byte-for-byte.
    // -----------------------------------------------------------------------

    /// Set a custom float32 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset,
    /// `OxiH5Error::Format` if it names a group or the root, or if its element
    /// type is not `f32`.
    pub fn set_fill_value_f32(&mut self, path: &str, value: f32) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::F32, value.to_le_bytes().to_vec())
    }

    /// Set a custom float64 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `f64` element type.
    pub fn set_fill_value_f64(&mut self, path: &str, value: f64) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::F64, value.to_le_bytes().to_vec())
    }

    /// Set a custom int8 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `i8` element type.
    pub fn set_fill_value_i8(&mut self, path: &str, value: i8) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::I8, value.to_le_bytes().to_vec())
    }

    /// Set a custom int16 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `i16` element type.
    pub fn set_fill_value_i16(&mut self, path: &str, value: i16) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::I16, value.to_le_bytes().to_vec())
    }

    /// Set a custom int32 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `i32` element type.
    pub fn set_fill_value_i32(&mut self, path: &str, value: i32) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::I32, value.to_le_bytes().to_vec())
    }

    /// Set a custom int64 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `i64` element type.
    pub fn set_fill_value_i64(&mut self, path: &str, value: i64) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::I64, value.to_le_bytes().to_vec())
    }

    /// Set a custom uint8 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `u8` element type.
    pub fn set_fill_value_u8(&mut self, path: &str, value: u8) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::U8, value.to_le_bytes().to_vec())
    }

    /// Set a custom uint16 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `u16` element type.
    pub fn set_fill_value_u16(&mut self, path: &str, value: u16) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::U16, value.to_le_bytes().to_vec())
    }

    /// Set a custom uint32 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `u32` element type.
    pub fn set_fill_value_u32(&mut self, path: &str, value: u32) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::U32, value.to_le_bytes().to_vec())
    }

    /// Set a custom uint64 fill value on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// As [`Self::set_fill_value_f32`], for the `u64` element type.
    pub fn set_fill_value_u64(&mut self, path: &str, value: u64) -> Result<&mut Self, OxiH5Error> {
        self.set_fill_value_le(path, ElemType::U64, value.to_le_bytes().to_vec())
    }

    /// Attach a typed custom fill value to the dataset at `path`.
    ///
    /// The value is carried on the dataset's attribute list as a sentinel and
    /// folded into the fill-value message by [`super::oh::dataset_oh_msgs`]; any
    /// previously set fill value is replaced.  Refusing a mismatched element
    /// type here is what stops an `f64` fill from landing in an `i32` dataset's
    /// fill message, where its extra bytes would corrupt the object header.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset,
    /// `OxiH5Error::Format` if it names a group or the root group, or if `elem`
    /// is not the dataset's element type.
    fn set_fill_value_le(
        &mut self,
        path: &str,
        elem: ElemType,
        le_bytes: Vec<u8>,
    ) -> Result<&mut Self, OxiH5Error> {
        let ds = dataset_mut(&mut self.root, path)?;
        if ds.dtype.is_some() {
            // A compound, array, opaque, bitfield or vlen-sequence dataset has no
            // `ElemType` for `elem` to be compared against, so the width check
            // below would pass on a coincidence: a one-byte fill would be
            // accepted for a twelve-byte record.  Refuse it outright.
            return Err(OxiH5Error::Format(format!(
                "set_fill_value('{path}'): a structured datatype has no scalar fill value"
            )));
        }
        if ds.elem_type != elem {
            return Err(OxiH5Error::Format(format!(
                "set_fill_value('{path}'): a {elem:?} fill value does not match the \
                 dataset's element type {:?}",
                ds.elem_type
            )));
        }
        // The typed wrappers guarantee this, but the invariant the fill message
        // relies on — value width equals element width — is asserted at the one
        // seam every fill value flows through rather than trusted.
        if le_bytes.len() != elem.byte_size() {
            return Err(OxiH5Error::Format(format!(
                "set_fill_value('{path}'): fill value is {} bytes, not the {} the \
                 element type occupies",
                le_bytes.len(),
                elem.byte_size()
            )));
        }
        // At most one fill value per dataset: drop any earlier sentinel first.
        ds.attrs
            .retain(|attr| !matches!(attr.kind, AttrKind::FillValue { .. }));
        ds.attrs.push(AttrDesc {
            name: FILL_VALUE_SENTINEL_NAME.to_string(),
            kind: AttrKind::FillValue { le_bytes },
        });
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // G017: Compact layout
    // -----------------------------------------------------------------------

    /// Store the dataset at `path` with the **compact** layout: its data inline
    /// in the object header rather than in a separate data area.
    ///
    /// libhdf5 uses this for small datasets to save a seek and a data block.
    /// The data is moved into the object header, so no separate contiguous data
    /// area is reserved — the dataset is stored exactly once.  Repeated calls are
    /// idempotent.
    ///
    /// Only a contiguous, fixed-layout dataset can be made compact: a chunked
    /// dataset (including a compressed one) and a vlen-string dataset have no
    /// single inline byte run and are refused.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset,
    /// `OxiH5Error::Format` if it names a group or the root group, if the dataset
    /// is chunked or a vlen-string dataset, or if its data exceeds the 64 KiB the
    /// HDF5 specification caps compact storage at.
    pub fn set_compact(&mut self, path: &str) -> Result<&mut Self, OxiH5Error> {
        let ds = dataset_mut(&mut self.root, path)?;
        // Idempotent: a dataset already marked compact keeps its inlined data
        // (its `raw` was emptied when the marker was attached).
        if ds
            .attrs
            .iter()
            .any(|attr| matches!(attr.kind, AttrKind::CompactLayout { .. }))
        {
            return Ok(self);
        }
        if ds.chunked().is_some() {
            return Err(OxiH5Error::Format(format!(
                "set_compact('{path}'): a chunked dataset cannot use the compact layout"
            )));
        }
        if ds.vlen_strings.is_some() || ds.vlen_seqs.is_some() {
            return Err(OxiH5Error::Format(format!(
                "set_compact('{path}'): a variable-length dataset cannot use the compact \
                 layout — its data area is global-heap references, not the values themselves"
            )));
        }
        // The compact layout message is a 4-byte prefix plus the data, padded to
        // 8 bytes and declared in the object header's 16-bit per-message size
        // field: `pad8(4 + len)` must fit 16 bits.  That is the ~64 KiB the HDF5
        // specification caps compact storage at.
        if pad8(4 + ds.raw.len()) > u16::MAX as usize {
            return Err(OxiH5Error::Format(format!(
                "set_compact('{path}'): {} bytes is over the compact-layout cap of 64 KiB — \
                 use the default contiguous layout for a dataset this large",
                ds.raw.len()
            )));
        }
        // Move the data onto the sentinel so the layout pass reserves no separate
        // data area for it: the object header carries the only copy.
        let data = std::mem::take(&mut ds.raw);
        ds.attrs.push(AttrDesc {
            name: COMPACT_LAYOUT_SENTINEL_NAME.to_string(),
            kind: AttrKind::CompactLayout { data },
        });
        Ok(self)
    }

    /// Place a contiguous dataset at `path`.
    fn add_dataset(
        &mut self,
        path: &str,
        raw: Vec<u8>,
        shape: &[usize],
        elem_type: ElemType,
    ) -> Result<&mut Self, OxiH5Error> {
        // Validate that the caller-supplied data length matches the declared
        // shape, so a mismatched `write_dataset_*(path, data, shape)` fails with
        // a clear error instead of silently producing a corrupt file (or later
        // panicking on an out-of-bounds slice during `build`).  This runs
        // *before* the path is resolved, because resolving it creates the
        // intermediate groups and a refused dataset must leave no trace.
        let byte_size = elem_type.byte_size();
        let expected = checked_byte_len(&format!("dataset '{path}'"), shape, byte_size)?;
        if raw.len() != expected {
            return Err(OxiH5Error::Format(format!(
                "dataset '{path}': data length {} does not match shape {shape:?} × {byte_size} bytes = {expected}",
                raw.len()
            )));
        }

        let (parent, name) = insertion_point(&mut self.root, path, "dataset")?;
        parent.push_dataset(DatasetDesc {
            name: name.to_string(),
            raw,
            shape: shape.to_vec(),
            elem_type,
            dtype: None,
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: None,
            vlen_seqs: None,
            creation_order: 0,
        });
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// G008 / G012: explicit byte order and half precision
// ---------------------------------------------------------------------------

/// A typed run of numeric values for [`FileWriter::write_dataset_numeric`] and
/// [`FileWriter::write_numeric_attr`].
///
/// One enum rather than a method per `(type, byte order)` pair: the byte order
/// is a separate argument, so eleven element types times two orders is one
/// entry point instead of twenty-two near-identical ones.
///
/// [`NumericValues::F16`] takes `f32` because Rust has no `f16` in stable
/// arithmetic; each value is rounded to IEEE-754 binary16 with
/// [`oxih5_core::f32_to_f16`] (round-to-nearest-ties-to-even), which is exactly
/// what `numpy.float16` does.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum NumericValues<'a> {
    /// IEEE-754 binary16, supplied as `f32` and rounded on write.
    F16(&'a [f32]),
    /// IEEE-754 binary32.
    F32(&'a [f32]),
    /// IEEE-754 binary64.
    F64(&'a [f64]),
    /// Signed 8-bit integers.
    I8(&'a [i8]),
    /// Signed 16-bit integers.
    I16(&'a [i16]),
    /// Signed 32-bit integers.
    I32(&'a [i32]),
    /// Signed 64-bit integers.
    I64(&'a [i64]),
    /// Unsigned 8-bit integers.
    U8(&'a [u8]),
    /// Unsigned 16-bit integers.
    U16(&'a [u16]),
    /// Unsigned 32-bit integers.
    U32(&'a [u32]),
    /// Unsigned 64-bit integers.
    U64(&'a [u64]),
}

impl NumericValues<'_> {
    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            NumericValues::F16(v) | NumericValues::F32(v) => v.len(),
            NumericValues::F64(v) => v.len(),
            NumericValues::I8(v) => v.len(),
            NumericValues::I16(v) => v.len(),
            NumericValues::I32(v) => v.len(),
            NumericValues::I64(v) => v.len(),
            NumericValues::U8(v) => v.len(),
            NumericValues::U16(v) => v.len(),
            NumericValues::U32(v) => v.len(),
            NumericValues::U64(v) => v.len(),
        }
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The numeric family these values belong to.
    pub(super) fn num_type(self) -> NumType {
        match self {
            NumericValues::F16(_) => NumType::F16,
            NumericValues::F32(_) => NumType::F32,
            NumericValues::F64(_) => NumType::F64,
            NumericValues::I8(_) => NumType::I8,
            NumericValues::I16(_) => NumType::I16,
            NumericValues::I32(_) => NumType::I32,
            NumericValues::I64(_) => NumType::I64,
            NumericValues::U8(_) => NumType::U8,
            NumericValues::U16(_) => NumType::U16,
            NumericValues::U32(_) => NumType::U32,
            NumericValues::U64(_) => NumType::U64,
        }
    }

    /// Serialise into on-disk bytes in `order`.
    ///
    /// Every arm goes through `to_le_bytes`/`to_be_bytes` on the on-disk width,
    /// so the payload's byte order always matches the byte-order bit the
    /// datatype message will declare.
    pub(super) fn to_bytes(self, order: ByteOrder) -> Vec<u8> {
        /// Concatenate `f(v)` over `values`, in `order`.
        macro_rules! pack {
            ($values:expr, $to_bits:expr) => {{
                let mut out = Vec::with_capacity($values.len() * core::mem::size_of_val(&0u64));
                for v in $values.iter() {
                    let bytes = $to_bits(*v);
                    match order {
                        ByteOrder::Little => out.extend_from_slice(&bytes.to_le_bytes()),
                        ByteOrder::Big => out.extend_from_slice(&bytes.to_be_bytes()),
                    }
                }
                out
            }};
        }
        match self {
            NumericValues::F16(v) => pack!(v, oxih5_core::f32_to_f16),
            // A float's byte order applies to its IEEE-754 bit pattern, so
            // going through `to_bits` is the same encoding as `f32::to_be_bytes`.
            NumericValues::F32(v) => pack!(v, f32::to_bits),
            NumericValues::F64(v) => pack!(v, f64::to_bits),
            NumericValues::I8(v) => pack!(v, core::convert::identity::<i8>),
            NumericValues::I16(v) => pack!(v, core::convert::identity::<i16>),
            NumericValues::I32(v) => pack!(v, core::convert::identity::<i32>),
            NumericValues::I64(v) => pack!(v, core::convert::identity::<i64>),
            NumericValues::U8(v) => pack!(v, core::convert::identity::<u8>),
            NumericValues::U16(v) => pack!(v, core::convert::identity::<u16>),
            NumericValues::U32(v) => pack!(v, core::convert::identity::<u32>),
            NumericValues::U64(v) => pack!(v, core::convert::identity::<u64>),
        }
    }
}

impl FileWriter {
    /// Add a numeric dataset at `path` in an explicit byte order.
    ///
    /// This is the entry point for the two element shapes the per-type
    /// `write_dataset_*` methods do not cover: **big-endian** datasets
    /// (`ByteOrder::Big`), and **half-precision** floats
    /// ([`NumericValues::F16`]).  With `ByteOrder::Little` and any other
    /// variant it is equivalent to the matching `write_dataset_*` method.
    ///
    /// The declared datatype and the payload bytes are produced from the same
    /// `order`, so the two can never disagree.
    ///
    /// ```
    /// use oxih5::{ByteOrder, FileWriter, NumericValues};
    ///
    /// let mut w = FileWriter::new();
    /// w.write_dataset_numeric("be", NumericValues::F32(&[1.0, 2.0]), ByteOrder::Big, &[2])?;
    /// w.write_dataset_numeric("half", NumericValues::F16(&[0.5, 1.5]), ByteOrder::Little, &[2])?;
    /// # Ok::<(), oxih5::OxiH5Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f32`]: `OxiH5Error::Format` if `path` is empty
    /// or malformed, if its final name is already taken, or if the value count
    /// does not equal the product of `shape`.
    pub fn write_dataset_numeric(
        &mut self,
        path: &str,
        values: NumericValues<'_>,
        order: ByteOrder,
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let elem = values.num_type().as_elem(order);
        self.add_dataset(path, values.to_bytes(order), shape, elem)
    }
}
