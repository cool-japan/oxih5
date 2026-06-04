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
    /// Use `as_strings()` for vlen-string or mixed cases.
    pub fn as_str_fixed(&self) -> Option<String> {
        self.attr.as_str_fixed()
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
                if self.attr.data.len() % n != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let count = self.attr.data.len() / n;
                let mut out = Vec::with_capacity(count);
                for chunk in self.attr.data.chunks_exact(n) {
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
