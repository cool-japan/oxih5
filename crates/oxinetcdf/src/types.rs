//! NetCDF-4 logical type system (B5).
//!
//! [`NcType`] maps HDF5 datatype classes to the NetCDF-4 type vocabulary.  All
//! HDF5 types that `oxih5::Dtype` can represent are covered.  Unknown or
//! unsupported variants fall back to `NcType::Opaque`.

use oxih5::Dtype;

/// The logical type of a NetCDF-4 variable or attribute.
///
/// Derived from the underlying [`oxih5::Dtype`] by [`NcType::from`].
#[derive(Debug, Clone, PartialEq)]
pub enum NcType {
    /// NC_BYTE — signed 8-bit integer.
    Int8,
    /// NC_SHORT — signed 16-bit integer.
    Int16,
    /// NC_INT / NC_LONG — signed 32-bit integer.
    Int32,
    /// NC_INT64 — signed 64-bit integer.
    Int64,
    /// NC_UBYTE — unsigned 8-bit integer.
    UInt8,
    /// NC_USHORT — unsigned 16-bit integer.
    UInt16,
    /// NC_UINT — unsigned 32-bit integer.
    UInt32,
    /// NC_UINT64 — unsigned 64-bit integer.
    UInt64,
    /// NC_FLOAT — single-precision 32-bit float.
    Float32,
    /// NC_DOUBLE — double-precision 64-bit float.
    Float64,
    /// NC_CHAR — single-byte character (fixed-length string of size 1).
    Char,
    /// NC_STRING — variable-length UTF-8 string.
    String,
    /// NC_ENUM — enumerated integer type.
    ///
    /// `name` is the HDF5 type name (may be empty for anonymous enums).
    /// `members` maps member names to their integer values.
    Enum {
        name: std::string::String,
        members: Vec<(std::string::String, i64)>,
    },
    /// NC_VLEN — variable-length sequence of another type.
    Vlen(Box<NcType>),
    /// NC_OPAQUE — opaque blob of `size` bytes.
    Opaque(usize),
    /// NC_COMPOUND — composite record type.
    ///
    /// `name` is the HDF5 type name (may be empty for anonymous compounds).
    /// `fields` is a list of `(field_name, field_type)` pairs in declaration order.
    Compound {
        name: std::string::String,
        fields: Vec<(std::string::String, NcType)>,
    },
}

impl NcType {
    /// Returns `true` if this is a numeric scalar type (int or float, not
    /// composite or string).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            NcType::Int8
                | NcType::Int16
                | NcType::Int32
                | NcType::Int64
                | NcType::UInt8
                | NcType::UInt16
                | NcType::UInt32
                | NcType::UInt64
                | NcType::Float32
                | NcType::Float64
        )
    }

    /// Returns `true` if this is any integer type (signed or unsigned).
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            NcType::Int8
                | NcType::Int16
                | NcType::Int32
                | NcType::Int64
                | NcType::UInt8
                | NcType::UInt16
                | NcType::UInt32
                | NcType::UInt64
        )
    }

    /// Returns `true` if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, NcType::Float32 | NcType::Float64)
    }
}

impl From<&Dtype> for NcType {
    fn from(dtype: &Dtype) -> Self {
        match dtype {
            // Signed integers
            Dtype::Int {
                size: 1,
                signed: true,
                ..
            } => NcType::Int8,
            Dtype::Int {
                size: 2,
                signed: true,
                ..
            } => NcType::Int16,
            Dtype::Int {
                size: 4,
                signed: true,
                ..
            } => NcType::Int32,
            Dtype::Int {
                size: 8,
                signed: true,
                ..
            } => NcType::Int64,
            // Unsigned integers
            Dtype::Int {
                size: 1,
                signed: false,
                ..
            } => NcType::UInt8,
            Dtype::Int {
                size: 2,
                signed: false,
                ..
            } => NcType::UInt16,
            Dtype::Int {
                size: 4,
                signed: false,
                ..
            } => NcType::UInt32,
            Dtype::Int {
                size: 8,
                signed: false,
                ..
            } => NcType::UInt64,
            // Floats
            Dtype::Float { size: 4, .. } => NcType::Float32,
            Dtype::Float { size: 8, .. } => NcType::Float64,
            // String types
            Dtype::String {
                fixed_len: Some(1), ..
            } => NcType::Char,
            Dtype::String {
                fixed_len: None, ..
            } => NcType::String,
            Dtype::String {
                fixed_len: Some(_), ..
            } => NcType::String,
            // Enum
            Dtype::Enum { members, .. } => NcType::Enum {
                name: std::string::String::new(),
                members: members.clone(),
            },
            // Variable-length sequences
            Dtype::VarLen { base } => NcType::Vlen(Box::new(NcType::from(base.as_ref()))),
            // Opaque
            Dtype::Opaque { size, .. } => NcType::Opaque(*size),
            // Compound
            Dtype::Compound { fields } => NcType::Compound {
                name: std::string::String::new(),
                fields: fields
                    .iter()
                    .map(|f| (f.name.clone(), NcType::from(&f.dtype)))
                    .collect(),
            },
            // Catch-all for types that don't map cleanly (Array, Reference, Bitfield,
            // unknown Float/Int sizes, etc.)
            Dtype::Array { .. } | Dtype::Reference { .. } | Dtype::Bitfield { .. } => {
                NcType::Opaque(dtype.size().unwrap_or(0))
            }
            // Unknown integer size
            Dtype::Int { size, .. } => NcType::Opaque(*size),
            // Unknown float size
            Dtype::Float { size, .. } => NcType::Opaque(*size),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests (B5)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5::{ByteOrder, Dtype};
    use oxih5_core::{Charset, CompoundField};

    #[test]
    fn test_nc_type_from_signed_ints() {
        assert_eq!(
            NcType::from(&Dtype::Int {
                size: 1,
                signed: true,
                order: ByteOrder::Little
            }),
            NcType::Int8
        );
        assert_eq!(
            NcType::from(&Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little
            }),
            NcType::Int32
        );
        assert_eq!(
            NcType::from(&Dtype::Int {
                size: 8,
                signed: true,
                order: ByteOrder::Big
            }),
            NcType::Int64
        );
    }

    #[test]
    fn test_nc_type_from_unsigned_ints() {
        assert_eq!(
            NcType::from(&Dtype::Int {
                size: 1,
                signed: false,
                order: ByteOrder::Little
            }),
            NcType::UInt8
        );
        assert_eq!(
            NcType::from(&Dtype::Int {
                size: 2,
                signed: false,
                order: ByteOrder::Big
            }),
            NcType::UInt16
        );
    }

    #[test]
    fn test_nc_type_from_floats() {
        assert_eq!(
            NcType::from(&Dtype::Float {
                size: 4,
                order: ByteOrder::Little
            }),
            NcType::Float32
        );
        assert_eq!(
            NcType::from(&Dtype::Float {
                size: 8,
                order: ByteOrder::Little
            }),
            NcType::Float64
        );
    }

    #[test]
    fn test_nc_type_from_string() {
        assert_eq!(
            NcType::from(&Dtype::String {
                fixed_len: None,
                charset: Charset::Utf8,
            }),
            NcType::String
        );
        // single-byte fixed → Char
        assert_eq!(
            NcType::from(&Dtype::String {
                fixed_len: Some(1),
                charset: Charset::Ascii,
            }),
            NcType::Char
        );
        // multi-byte fixed → String
        assert_eq!(
            NcType::from(&Dtype::String {
                fixed_len: Some(32),
                charset: Charset::Utf8,
            }),
            NcType::String
        );
    }

    #[test]
    fn test_nc_type_from_enum() {
        let dtype = Dtype::Enum {
            base: Box::new(Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            }),
            members: vec![("RED".to_string(), 0), ("GREEN".to_string(), 1)],
        };
        match NcType::from(&dtype) {
            NcType::Enum { name, members } => {
                assert!(name.is_empty());
                assert_eq!(members.len(), 2);
                assert_eq!(members[0], ("RED".to_string(), 0));
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn test_nc_type_from_vlen() {
        let dtype = Dtype::VarLen {
            base: Box::new(Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            }),
        };
        match NcType::from(&dtype) {
            NcType::Vlen(inner) => assert_eq!(*inner, NcType::Float32),
            other => panic!("expected Vlen, got {:?}", other),
        }
    }

    #[test]
    fn test_nc_type_from_opaque() {
        let dtype = Dtype::Opaque {
            size: 16,
            tag: "uuid".to_string(),
        };
        assert_eq!(NcType::from(&dtype), NcType::Opaque(16));
    }

    #[test]
    fn test_nc_type_from_compound() {
        let dtype = Dtype::Compound {
            fields: vec![
                CompoundField {
                    name: "x".to_string(),
                    offset: 0,
                    dtype: Dtype::Float {
                        size: 4,
                        order: ByteOrder::Little,
                    },
                },
                CompoundField {
                    name: "y".to_string(),
                    offset: 4,
                    dtype: Dtype::Int {
                        size: 4,
                        signed: true,
                        order: ByteOrder::Little,
                    },
                },
            ],
        };
        match NcType::from(&dtype) {
            NcType::Compound { name, fields } => {
                assert!(name.is_empty());
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0], ("x".to_string(), NcType::Float32));
                assert_eq!(fields[1], ("y".to_string(), NcType::Int32));
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn test_nc_type_is_numeric() {
        assert!(NcType::Float64.is_numeric());
        assert!(NcType::Int32.is_numeric());
        assert!(!NcType::String.is_numeric());
        assert!(!NcType::Compound {
            name: std::string::String::new(),
            fields: vec![]
        }
        .is_numeric());
    }

    #[test]
    fn test_nc_type_is_float() {
        assert!(NcType::Float32.is_float());
        assert!(NcType::Float64.is_float());
        assert!(!NcType::Int32.is_float());
    }
}
