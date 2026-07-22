//! `AttrView` — a facade wrapper that exposes file-context-dependent attribute
//! accessors for attributes read from an HDF5 file.
//!
//! Core `Attribute` is file-independent (it just holds bytes + dtype + dataspace).
//! Operations that require the global heap (vlen-string decode, object-ref resolve)
//! need the file bytes, which are not stored in `Attribute`.  `AttrView<'a>` owns
//! the `Attribute` data while borrowing `&'a [u8]` (the file bytes).

use oxih5_core::{Attribute, Dataspace, Dtype, OxiH5Error};
use oxih5_format::values;

/// A view of a single [`Attribute`] with access to the originating file bytes.
///
/// The `Attribute` is owned (cloned from the file at creation time).
/// The file bytes are borrowed for the lifetime `'a`.
///
/// Provides:
/// - All file-independent accessors (`as_i64`, `as_f64`, `as_str_fixed`, …)
/// - `as_strings` — decodes fixed-length or vlen-string attributes to `Vec<String>`
/// - `as_object_refs` — decodes object-reference attributes to `Vec<u64>`
/// - `as_compound` — decodes compound-type attributes to `Vec<values::Value>`
pub struct AttrView<'a> {
    /// The attribute data (owned).
    pub attr: Attribute,
    /// The originating file bytes, borrowed for vlen / object-ref resolution.
    file_data: &'a [u8],
}

impl<'a> AttrView<'a> {
    /// Create a new `AttrView` with an owned `Attribute` and borrowed file bytes.
    pub(crate) fn new(attr: Attribute, file_data: &'a [u8]) -> Self {
        Self { attr, file_data }
    }

    /// Attribute name.
    pub fn name(&self) -> &str {
        &self.attr.name
    }

    /// Attribute datatype.
    pub fn dtype(&self) -> &Dtype {
        &self.attr.dtype
    }

    /// Returns true if the attribute is a scalar (or single-element simple space).
    pub fn is_scalar(&self) -> bool {
        self.attr.is_scalar()
    }

    /// Returns the shape of the attribute's dataspace.
    pub fn shape(&self) -> Vec<u64> {
        self.attr.shape()
    }

    /// Decode as a scalar i64.
    pub fn as_i64(&self) -> Option<i64> {
        self.attr.as_i64()
    }

    /// Decode as a scalar u64 (unsigned integer dtypes only).
    pub fn as_u64(&self) -> Option<u64> {
        self.attr.as_u64()
    }

    /// Decode as a scalar f64 (f32 and f64 dtypes).
    pub fn as_f64(&self) -> Option<f64> {
        self.attr.as_f64()
    }

    /// Decode a fixed-length string attribute as a `String` (trims NUL padding).
    ///
    /// Returns `None` for variable-length (vlen) string attributes — the form
    /// h5py writes by default for `dset.attrs['units'] = 'm'`.  Use
    /// [`as_str`](Self::as_str) for the general scalar-string accessor that
    /// handles both fixed-length and vlen strings, or [`as_strings`](Self::as_strings)
    /// for the array case.
    pub fn as_str_fixed(&self) -> Option<String> {
        self.attr.as_str_fixed()
    }

    /// Decode a single-element string attribute as a `String`, handling **both**
    /// fixed-length and variable-length (vlen) string datatypes.
    ///
    /// This is the general scalar-string accessor.  h5py writes scalar string
    /// attributes (`dset.attrs['units'] = 'm'`) and netCDF-C's
    /// `var.setncattr_string(...)` as *variable-length* strings (HDF5 datatype
    /// class 9), for which [`as_str_fixed`](Self::as_str_fixed) returns `None`;
    /// this method resolves the global-heap reference and returns the decoded
    /// string.  Empty strings and multi-byte UTF-8 are handled.
    ///
    /// Returns `None` when the attribute is not a string datatype, or when its
    /// dataspace holds more than one element (use [`as_strings`](Self::as_strings)
    /// for arrays of strings).
    pub fn as_str(&self) -> Option<String> {
        match &self.attr.dtype {
            Dtype::String { .. } => {
                if self.n_elems() != 1 {
                    return None;
                }
                self.as_strings().ok()?.into_iter().next()
            }
            _ => None,
        }
    }

    /// Decode this attribute as a vector of strings.
    ///
    /// Handles both fixed-length strings and vlen-string attributes:
    /// - Fixed: splits `data` into `fixed_len`-byte chunks, strips NUL, converts UTF-8.
    /// - Vlen: resolves 16-byte global-heap references via the file bytes.
    pub fn as_strings(&self) -> Result<Vec<String>, OxiH5Error> {
        match &self.attr.dtype {
            Dtype::String {
                fixed_len: Some(n), ..
            } => {
                let n = *n;
                if n == 0 {
                    return Ok(vec![]);
                }
                // An attribute's data section is padded out to an 8-byte
                // boundary inside its object-header message — libhdf5 does this
                // too, so `data` routinely carries up to 7 trailing bytes that
                // belong to no element.  The element count therefore has to come
                // from the dataspace; deriving it from `data.len()` either
                // rejected valid attributes (`"hello"`, 5 bytes padded to 8, is
                // not divisible by 5) or invented empty ones (`"km"`, 2 bytes
                // padded to 8, yielded four strings).
                let count = self.n_elems();
                let needed = count.checked_mul(n).ok_or(OxiH5Error::DataTruncated)?;
                if self.attr.data.len() < needed {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(count);
                for chunk in self.attr.data[..needed].chunks_exact(n) {
                    let trimmed = chunk.split(|&b| b == 0).next().unwrap_or(chunk);
                    let s = String::from_utf8(trimmed.to_vec())
                        .map_err(|e| OxiH5Error::Format(format!("fixed-string attr UTF-8: {e}")))?;
                    out.push(s);
                }
                Ok(out)
            }
            Dtype::String {
                fixed_len: None, ..
            } => {
                let n_elems = self.n_elems();
                values::decode_vlen_strings(self.file_data, &self.attr.data, n_elems)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Decode this attribute as object references (one u64 per element).
    ///
    /// Each value is an absolute byte offset of the target object header.
    /// `u64::MAX` denotes an undefined/null reference.
    pub fn as_object_refs(&self) -> Result<Vec<u64>, OxiH5Error> {
        match &self.attr.dtype {
            Dtype::Reference {
                ref_type: oxih5_core::RefType::Object,
            } => {
                let n = self.n_elems();
                values::decode_object_refs(&self.attr.data, n)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Decode this attribute as compound-type values.
    pub fn as_compound(&self) -> Result<Vec<values::Value>, OxiH5Error> {
        match &self.attr.dtype {
            Dtype::Compound { fields } => {
                let n = self.n_elems();
                values::decode_compound(self.file_data, &self.attr.data, fields, n, 0)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Decode this attribute as a vlen sequence of typed values.
    pub fn as_vlen_sequence(&self) -> Result<Vec<values::Value>, OxiH5Error> {
        match &self.attr.dtype {
            Dtype::VarLen { base } => {
                let n = self.n_elems();
                values::decode_vlen_sequences(self.file_data, &self.attr.data, n, base)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn n_elems(&self) -> usize {
        match &self.attr.dataspace {
            Dataspace::Scalar => 1,
            Dataspace::Null => 0,
            Dataspace::Simple { dims, .. } => dims.iter().product::<u64>() as usize,
        }
    }
}

impl std::fmt::Debug for AttrView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttrView")
            .field("name", &self.attr.name)
            .field("dtype", &self.attr.dtype)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_core::{ByteOrder, Charset, Dataspace};

    /// Build a minimal `GCOL` global-heap collection holding `objects`
    /// (index, bytes), starting at offset 0.  Mirrors the on-disk layout
    /// libhdf5 writes: 16-byte header, then each object as a 16-byte header
    /// (index, ref-count, reserved, size) followed by data padded to an 8-byte
    /// boundary, terminated by an index-0 NIL object.
    fn build_gcol(objects: &[(u16, &[u8])]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"GCOL");
        data.push(1);
        data.extend_from_slice(&[0u8; 3]);
        let size_pos = data.len();
        data.extend_from_slice(&[0u8; 8]);
        for (idx, obj) in objects {
            data.extend_from_slice(&idx.to_le_bytes());
            data.extend_from_slice(&1u16.to_le_bytes());
            data.extend_from_slice(&[0u8; 4]);
            data.extend_from_slice(&(obj.len() as u64).to_le_bytes());
            data.extend_from_slice(obj);
            let pad = (8 - (data.len() % 8)) % 8;
            data.extend(std::iter::repeat(0u8).take(pad));
        }
        data.extend_from_slice(&[0u8; 16]);
        let total = data.len() as u64;
        data[size_pos..size_pos + 8].copy_from_slice(&total.to_le_bytes());
        data
    }

    /// Build a 16-byte on-disk vlen-string reference (length, heap addr, index).
    fn vlen_ref(seq_len: u32, heap_addr: u64, obj_idx: u32) -> Vec<u8> {
        let mut r = Vec::with_capacity(16);
        r.extend_from_slice(&seq_len.to_le_bytes());
        r.extend_from_slice(&heap_addr.to_le_bytes());
        r.extend_from_slice(&obj_idx.to_le_bytes());
        r
    }

    fn vlen_string_attr(refs: Vec<u8>, n: u64) -> Attribute {
        Attribute {
            name: "units".into(),
            dtype: Dtype::String {
                fixed_len: None,
                charset: Charset::Utf8,
            },
            dataspace: if n == 1 {
                Dataspace::Scalar
            } else {
                Dataspace::Simple {
                    dims: vec![n],
                    max_dims: None,
                }
            },
            data: refs,
        }
    }

    #[test]
    fn test_as_str_fixed_scalar() {
        let attr = Attribute {
            name: "units".into(),
            dtype: Dtype::String {
                fixed_len: Some(8),
                charset: Charset::Ascii,
            },
            dataspace: Dataspace::Scalar,
            data: b"meters\0\0".to_vec(),
        };
        let view = AttrView::new(attr, &[]);
        assert_eq!(view.as_str(), Some("meters".to_string()));
    }

    #[test]
    fn test_as_str_vlen_scalar() {
        // h5py's default form for `dset.attrs['units'] = 'm'`: a scalar,
        // variable-length string resolved through the global heap.
        let heap = build_gcol(&[(1, b"m")]);
        let attr = vlen_string_attr(vlen_ref(1, 0, 1), 1);
        let view = AttrView::new(attr, &heap);
        // as_str_fixed cannot decode a vlen string; as_str must.
        assert_eq!(view.as_str_fixed(), None);
        assert_eq!(view.as_str(), Some("m".to_string()));
    }

    #[test]
    fn test_as_str_vlen_empty_and_utf8() {
        let heap = build_gcol(&[(1, "café温度".as_bytes())]);
        let attr = vlen_string_attr(vlen_ref(11, 0, 1), 1);
        let view = AttrView::new(attr, &heap);
        assert_eq!(view.as_str(), Some("café温度".to_string()));

        // Empty vlen string: seq_len 0, no heap lookup.
        let empty = vlen_string_attr(vlen_ref(0, 0, 0), 1);
        let view = AttrView::new(empty, &[]);
        assert_eq!(view.as_str(), Some(String::new()));
    }

    #[test]
    fn test_as_str_multi_element_returns_none() {
        // A 1-D array of vlen strings is not a scalar: as_str returns None,
        // callers must use as_strings().
        let heap = build_gcol(&[(1, b"a"), (2, b"bb")]);
        let mut refs = vlen_ref(1, 0, 1);
        refs.extend(vlen_ref(2, 0, 2));
        let attr = vlen_string_attr(refs, 2);
        let view = AttrView::new(attr, &heap);
        assert_eq!(view.as_str(), None);
        assert_eq!(
            view.as_strings().unwrap(),
            vec!["a".to_string(), "bb".to_string()]
        );
    }

    #[test]
    fn test_as_str_non_string_returns_none() {
        let attr = Attribute {
            name: "count".into(),
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            dataspace: Dataspace::Scalar,
            data: 7i32.to_le_bytes().to_vec(),
        };
        let view = AttrView::new(attr, &[]);
        assert_eq!(view.as_str(), None);
    }

    #[test]
    fn test_attr_view_fixed_strings() {
        let data: Vec<u8> = b"hello\0\0\0world\0\0\0".to_vec();
        let attr = Attribute {
            name: "label".into(),
            dtype: Dtype::String {
                fixed_len: Some(8),
                charset: Charset::Utf8,
            },
            dataspace: Dataspace::Simple {
                dims: vec![2],
                max_dims: None,
            },
            data,
        };
        let view = AttrView::new(attr, &[]);
        let strings = view.as_strings().unwrap();
        assert_eq!(strings, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_attr_view_object_refs() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x1000u64.to_le_bytes());
        data.extend_from_slice(&0x2000u64.to_le_bytes());
        let attr = Attribute {
            name: "refs".into(),
            dtype: Dtype::Reference {
                ref_type: oxih5_core::RefType::Object,
            },
            dataspace: Dataspace::Simple {
                dims: vec![2],
                max_dims: None,
            },
            data,
        };
        let view = AttrView::new(attr, &[]);
        let refs = view.as_object_refs().unwrap();
        assert_eq!(refs, vec![0x1000u64, 0x2000u64]);
    }

    #[test]
    fn test_attr_view_scalar_i64() {
        let attr = Attribute {
            name: "count".into(),
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            dataspace: Dataspace::Scalar,
            data: 99i32.to_le_bytes().to_vec(),
        };
        let view = AttrView::new(attr, &[]);
        assert_eq!(view.as_i64(), Some(99));
        assert!(view.is_scalar());
    }

    #[test]
    fn test_attr_view_type_mismatch_string_on_int() {
        let attr = Attribute {
            name: "num".into(),
            dtype: Dtype::Int {
                size: 4,
                signed: false,
                order: ByteOrder::Little,
            },
            dataspace: Dataspace::Scalar,
            data: 42u32.to_le_bytes().to_vec(),
        };
        let view = AttrView::new(attr, &[]);
        assert!(view.as_strings().is_err());
    }

    #[test]
    fn test_attr_view_compound() {
        use oxih5_core::CompoundField;
        let fields = vec![
            CompoundField {
                name: "x".into(),
                offset: 0,
                dtype: Dtype::Int {
                    size: 4,
                    signed: true,
                    order: ByteOrder::Little,
                },
            },
            CompoundField {
                name: "y".into(),
                offset: 4,
                dtype: Dtype::Float {
                    size: 4,
                    order: ByteOrder::Little,
                },
            },
        ];
        let mut data = Vec::new();
        data.extend_from_slice(&5i32.to_le_bytes());
        data.extend_from_slice(&1.5f32.to_le_bytes());
        let attr = Attribute {
            name: "point".into(),
            dtype: Dtype::Compound { fields },
            dataspace: Dataspace::Scalar,
            data,
        };
        let view = AttrView::new(attr, &[]);
        let values = view.as_compound().unwrap();
        assert_eq!(values.len(), 1);
        if let oxih5_format::values::Value::Compound(ref pairs) = values[0] {
            assert_eq!(pairs[0].0, "x");
            assert_eq!(pairs[0].1, oxih5_format::values::Value::Int(5));
        } else {
            panic!("expected Compound value");
        }
    }
}
