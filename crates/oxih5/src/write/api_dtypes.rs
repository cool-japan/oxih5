//! `FileWriter` entry points for the datatype classes beyond scalars.
//!
//! Compound records (class 6), ragged variable-length sequences (class 9
//! subtype 0), array elements (class 10), opaque blobs (class 5) and bitfields
//! (class 4).  Every one of them is a datatype whose message length depends on
//! the caller's description, so each entry point encodes that description once,
//! up front, through [`super::dtype`] — a bad member list, a zero extent or an
//! unwritable member type is a typed error at the *call*, not at build time.
//!
//! # Row bytes, not typed rows
//!
//! A compound dataset takes the record bytes the caller already has, together
//! with the member offsets that describe them.  That is deliberate: a record
//! type is chosen by the caller (packed, C-padded, reordered), and any typed
//! row struct this crate could offer would be one particular choice imposed on
//! everybody.  Handing over `&[u8]` plus offsets keeps the writer honest — the
//! datatype message describes exactly the bytes that were written.

use oxih5_core::{ByteOrder, CompoundField, Dtype, OxiH5Error};

use super::dtype::{self, member_elem_type, EncodedDtype};
use super::elem::ElemType;
use super::tree::{insertion_point, DatasetDesc, Storage};
use super::{checked_byte_len, FileWriter};

impl FileWriter {
    /// Add a dataset whose type is one of the structured classes.
    ///
    /// The single insertion point behind every entry point in this module, so
    /// the length check, the duplicate-name check and the descriptor's shape
    /// are settled once rather than five times.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `raw` does not match `shape` times the
    /// datatype's element size, if `path` is malformed, or if its final name is
    /// already used in that group.
    fn add_structured_dataset(
        &mut self,
        path: &str,
        encoded: EncodedDtype,
        raw: Vec<u8>,
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let expected = checked_byte_len(&format!("dataset '{path}'"), shape, encoded.elem_size)?;
        if raw.len() != expected {
            return Err(OxiH5Error::Format(format!(
                "dataset '{path}' ({}): data length {} does not match shape {shape:?} × {} \
                 bytes = {expected}",
                encoded.label,
                raw.len(),
                encoded.elem_size
            )));
        }
        let (parent, name) = insertion_point(&mut self.root, path, "dataset")?;
        parent.push_dataset(DatasetDesc {
            name: name.to_string(),
            raw,
            shape: shape.to_vec(),
            // Never read while `dtype` is set; see `DatasetDesc::elem_size`.
            elem_type: ElemType::U8,
            dtype: Some(encoded),
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: None,
            vlen_seqs: None,
            creation_order: 0,
        });
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // G004: compound (record) datasets
    // -----------------------------------------------------------------------

    /// Create a **compound** (record / structured) dataset at `path`.
    ///
    /// `fields` describes the record: each member's name, its byte offset
    /// within the record, and its type.  `record_size` is the stride between
    /// consecutive records — which is a caller's choice, not a derived value:
    /// a packed layout and a C-padded one over the same members are different,
    /// valid records, and the offsets say which one `rows` actually holds.
    ///
    /// `rows` is the raw record bytes, laid out exactly as `fields` and
    /// `record_size` describe, and `shape` is the dataset's extent in records.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// use oxih5_core::{ByteOrder, CompoundField, Dtype};
    ///
    /// let fields = vec![
    ///     CompoundField {
    ///         name: "id".to_string(),
    ///         offset: 0,
    ///         dtype: Dtype::Int { size: 4, signed: true, order: ByteOrder::Little },
    ///     },
    ///     CompoundField {
    ///         name: "value".to_string(),
    ///         offset: 4,
    ///         dtype: Dtype::Float { size: 8, order: ByteOrder::Little },
    ///     },
    /// ];
    /// let mut rows = Vec::new();
    /// for (id, value) in [(1i32, 2.5f64), (3, 4.5)] {
    ///     rows.extend_from_slice(&id.to_le_bytes());
    ///     rows.extend_from_slice(&value.to_le_bytes());
    /// }
    ///
    /// let path = std::env::temp_dir().join("records.h5");
    /// FileWriter::new()
    ///     .create_compound_dataset("/events", &fields, 12, &rows, &[2])
    ///     .unwrap()
    ///     .build(&path)
    ///     .unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `fields` is empty, if two members share
    /// a name or overlap, if a member runs past `record_size`, if a member's
    /// type is not writable inline, or if `rows` does not match
    /// `shape × record_size`.
    pub fn create_compound_dataset(
        &mut self,
        path: &str,
        fields: &[CompoundField],
        record_size: usize,
        rows: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let encoded = dtype::compound(fields, record_size)?;
        self.add_structured_dataset(path, encoded, rows.to_vec(), shape)
    }

    // -----------------------------------------------------------------------
    // G015: variable-length (ragged) sequence datasets
    // -----------------------------------------------------------------------

    /// Create a **variable-length sequence** (ragged) dataset at `path`.
    ///
    /// Every element is an independently sized run of `base` values, stored in
    /// the file's global heap and addressed by a 16-byte reference — the same
    /// mechanism a vlen-string dataset uses, with a sequence type instead of a
    /// string one.  An empty element is written as the null reference, which is
    /// what libhdf5 writes and what reads back as an empty sequence.
    ///
    /// `sequences` holds each element's bytes, already serialised in `base`'s
    /// byte order; a run whose length is not a whole number of `base` elements
    /// is refused.
    ///
    /// Read it back with [`crate::File::dataset_vlen_sequences`].
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `base` is not a writable fixed-size
    /// type, if a sequence's byte length is not a multiple of the base type's
    /// width, or if `path` is malformed or its name is taken.
    pub fn create_vlen_sequence_dataset(
        &mut self,
        path: &str,
        base: &Dtype,
        sequences: &[Vec<u8>],
    ) -> Result<&mut Self, OxiH5Error> {
        let base_elem = member_elem_type(base)?;
        let width = base_elem.byte_size();
        for (index, bytes) in sequences.iter().enumerate() {
            if width == 0 || bytes.len() % width != 0 {
                return Err(OxiH5Error::Format(format!(
                    "create_vlen_sequence_dataset('{path}'): sequence {index} is {} bytes, \
                     not a whole number of {width}-byte elements",
                    bytes.len()
                )));
            }
        }
        let encoded = dtype::vlen_sequence(base_elem)?;

        let (parent, name) = insertion_point(&mut self.root, path, "vlen-sequence dataset")?;
        parent.push_dataset(DatasetDesc {
            name: name.to_string(),
            raw: Vec::new(), // the data area is references, built at emit time
            shape: vec![sequences.len()],
            elem_type: ElemType::U8,
            dtype: Some(encoded),
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: None,
            vlen_seqs: Some(sequences.to_vec()),
            creation_order: 0,
        });
        Ok(self)
    }

    /// Create a ragged `int32` dataset — [`Self::create_vlen_sequence_dataset`]
    /// with the serialisation done for you.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("ragged.h5");
    /// FileWriter::new()
    ///     .create_vlen_i32_dataset("/rows", &[vec![1, 2, 3], vec![], vec![10]])
    ///     .unwrap()
    ///     .build(&path)
    ///     .unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::create_vlen_sequence_dataset`].
    pub fn create_vlen_i32_dataset(
        &mut self,
        path: &str,
        sequences: &[Vec<i32>],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<Vec<u8>> = sequences
            .iter()
            .map(|seq| seq.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect();
        self.create_vlen_sequence_dataset(
            path,
            &Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            &raw,
        )
    }

    /// Create a ragged `float64` dataset.
    ///
    /// # Errors
    ///
    /// As [`Self::create_vlen_sequence_dataset`].
    pub fn create_vlen_f64_dataset(
        &mut self,
        path: &str,
        sequences: &[Vec<f64>],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<Vec<u8>> = sequences
            .iter()
            .map(|seq| seq.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect();
        self.create_vlen_sequence_dataset(
            path,
            &Dtype::Float {
                size: 8,
                order: ByteOrder::Little,
            },
            &raw,
        )
    }

    // -----------------------------------------------------------------------
    // G016: array, opaque and bitfield datasets
    // -----------------------------------------------------------------------

    /// Create an **array-datatype** dataset at `path`.
    ///
    /// Every element of the dataset is a fixed `array_dims`-shaped block of
    /// `base` values, so an element occupies `∏ array_dims × sizeof(base)`
    /// bytes.  This is genuinely different from folding those dimensions into
    /// the dataspace: the *element type* is the array, which is what a reader
    /// sees and what a compound member of array type would need.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `base` is not a writable fixed-size
    /// type, if `array_dims` is empty or holds a zero, or if `data` does not
    /// match `shape` times the element size.
    pub fn create_array_dataset(
        &mut self,
        path: &str,
        base: &Dtype,
        array_dims: &[usize],
        data: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let encoded = dtype::array(member_elem_type(base)?, array_dims)?;
        self.add_structured_dataset(path, encoded, data.to_vec(), shape)
    }

    /// Create an **opaque** dataset at `path`: `elem_size` uninterpreted bytes
    /// per element, labelled `tag`.
    ///
    /// The tag is how a reader recognises the bytes — h5py writes `NUMPY:|V7`
    /// for a `void7` element — and is stored NUL-padded to an 8-byte boundary.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `elem_size` is zero, if `tag` holds a
    /// NUL or is over 255 bytes once padded, or if `data` does not match
    /// `shape × elem_size`.
    pub fn create_opaque_dataset(
        &mut self,
        path: &str,
        elem_size: usize,
        tag: &str,
        data: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let encoded = dtype::opaque(elem_size, tag)?;
        self.add_structured_dataset(path, encoded, data.to_vec(), shape)
    }

    /// Create a **bitfield** dataset at `path`: `elem_size` bytes of flags per
    /// element, `elem_size × 8` bits precise.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `elem_size` is not in `1..=8`, or if
    /// `data` does not match `shape × elem_size`.
    pub fn create_bitfield_dataset(
        &mut self,
        path: &str,
        elem_size: usize,
        order: ByteOrder,
        data: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let encoded = dtype::bitfield(elem_size, order)?;
        self.add_structured_dataset(path, encoded, data.to_vec(), shape)
    }
}
