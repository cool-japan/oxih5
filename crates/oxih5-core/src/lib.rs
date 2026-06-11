#![forbid(unsafe_code)]

mod dataset_convert;
pub use dataset_convert::f16_to_f32;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

impl std::fmt::Display for ByteOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ByteOrder::Little => write!(f, "LE"),
            ByteOrder::Big => write!(f, "BE"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Charset {
    Ascii,
    Utf8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompoundField {
    pub name: String,
    pub offset: usize,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefType {
    Object,
    Region,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dtype {
    Int {
        size: usize,
        signed: bool,
        order: ByteOrder,
    },
    Float {
        size: usize,
        order: ByteOrder,
    },
    String {
        fixed_len: Option<usize>,
        charset: Charset,
    },
    Compound {
        fields: Vec<CompoundField>,
    },
    Array {
        base: Box<Dtype>,
        dims: Vec<usize>,
    },
    Enum {
        base: Box<Dtype>,
        members: Vec<(std::string::String, i64)>,
    },
    Opaque {
        size: usize,
        tag: std::string::String,
    },
    Reference {
        ref_type: RefType,
    },
    VarLen {
        base: Box<Dtype>,
    },
    Bitfield {
        size: usize,
        order: ByteOrder,
    },
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::Float { size, order } => write!(f, "Float{} {}", size * 8, order),
            Dtype::Int {
                size,
                signed,
                order,
            } => {
                let sign = if *signed { "signed" } else { "unsigned" };
                write!(f, "Int{} {} ({})", size * 8, order, sign)
            }
            Dtype::String { fixed_len, charset } => {
                let cs = match charset {
                    Charset::Ascii => "ASCII",
                    Charset::Utf8 => "UTF-8",
                };
                match fixed_len {
                    Some(n) => write!(f, "String ({}, fixed={})", cs, n),
                    None => write!(f, "String ({}, variable-length)", cs),
                }
            }
            Dtype::Compound { fields } => {
                let names: Vec<&str> = fields.iter().map(|fi| fi.name.as_str()).collect();
                write!(f, "Compound {{{}}}", names.join(", "))
            }
            Dtype::Array { base, dims } => write!(f, "Array[{:?}] of {}", dims, base),
            Dtype::Enum { base, .. } => write!(f, "Enum (base: {})", base),
            Dtype::Opaque { size, tag } => write!(f, "Opaque({} bytes, tag={})", size, tag),
            Dtype::Reference { ref_type } => write!(f, "Reference({:?})", ref_type),
            Dtype::VarLen { base } => write!(f, "VarLen({})", base),
            Dtype::Bitfield { size, order } => write!(f, "Bitfield{} {}", size * 8, order),
        }
    }
}

impl Dtype {
    /// In-memory size of a single element of this datatype, in bytes.
    ///
    /// For variable-length and string-with-no-fixed-length types the stored
    /// element is a global-heap reference / pointer whose on-disk footprint is
    /// not fixed by the datatype alone, so this returns `None`.
    pub fn size(&self) -> Option<usize> {
        match self {
            Dtype::Int { size, .. }
            | Dtype::Float { size, .. }
            | Dtype::Bitfield { size, .. }
            | Dtype::Opaque { size, .. } => Some(*size),
            Dtype::String { fixed_len, .. } => *fixed_len,
            Dtype::Reference { ref_type } => Some(match ref_type {
                RefType::Object => 8,
                RefType::Region => 12,
            }),
            Dtype::Enum { base, .. } => base.size(),
            Dtype::Array { base, dims } => {
                let elems: usize = dims.iter().product();
                base.size().map(|s| s * elems)
            }
            Dtype::Compound { fields } => {
                let mut total = 0usize;
                for f in fields {
                    let fs = f.dtype.size()?;
                    total = total.max(f.offset + fs);
                }
                Some(total)
            }
            Dtype::VarLen { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dataspace {
    Simple {
        dims: Vec<u64>,
        max_dims: Option<Vec<u64>>,
    },
    Null,
    Scalar,
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: std::string::String,
    pub dtype: Dtype,
    pub dataspace: Dataspace,
    pub data: Vec<u8>,
}

impl Attribute {
    /// Try to decode this attribute as a scalar i64 value.
    ///
    /// Accepts any signed or unsigned integer dtype of size 1/2/4/8 and widens
    /// to i64.  Unsigned values are reinterpreted via the bit pattern (u64 cast
    /// to i64) when they exceed i64::MAX — callers are responsible for
    /// interpreting the signedness correctly from `dtype`.
    pub fn as_i64(&self) -> Option<i64> {
        match &self.dtype {
            Dtype::Int {
                size,
                signed,
                order,
            } => {
                let sz = *size;
                if self.data.len() < sz {
                    return None;
                }
                let bytes = &self.data[..sz];
                Some(match (sz, order, signed) {
                    (1, _, true) => i64::from(bytes[0] as i8),
                    (1, _, false) => i64::from(bytes[0]),
                    (2, ByteOrder::Little, true) => {
                        i64::from(i16::from_le_bytes([bytes[0], bytes[1]]))
                    }
                    (2, ByteOrder::Big, true) => {
                        i64::from(i16::from_be_bytes([bytes[0], bytes[1]]))
                    }
                    (2, ByteOrder::Little, false) => {
                        i64::from(u16::from_le_bytes([bytes[0], bytes[1]]))
                    }
                    (2, ByteOrder::Big, false) => {
                        i64::from(u16::from_be_bytes([bytes[0], bytes[1]]))
                    }
                    (4, ByteOrder::Little, true) => {
                        i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    (4, ByteOrder::Big, true) => {
                        i64::from(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    (4, ByteOrder::Little, false) => {
                        i64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    (4, ByteOrder::Big, false) => {
                        i64::from(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    (8, ByteOrder::Little, true) => i64::from_le_bytes(bytes.try_into().ok()?),
                    (8, ByteOrder::Big, true) => i64::from_be_bytes(bytes.try_into().ok()?),
                    (8, ByteOrder::Little, false) => {
                        u64::from_le_bytes(bytes.try_into().ok()?) as i64
                    }
                    (8, ByteOrder::Big, false) => u64::from_be_bytes(bytes.try_into().ok()?) as i64,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Try to decode this attribute as a scalar u64 value.
    ///
    /// Only works for unsigned integer dtypes; signed integers return `None`.
    pub fn as_u64(&self) -> Option<u64> {
        match &self.dtype {
            Dtype::Int {
                size,
                signed: false,
                order,
            } => {
                let sz = *size;
                if self.data.len() < sz {
                    return None;
                }
                let bytes = &self.data[..sz];
                Some(match (sz, order) {
                    (1, _) => u64::from(bytes[0]),
                    (2, ByteOrder::Little) => u64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
                    (2, ByteOrder::Big) => u64::from(u16::from_be_bytes([bytes[0], bytes[1]])),
                    (4, ByteOrder::Little) => {
                        u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    (4, ByteOrder::Big) => {
                        u64::from(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    (8, ByteOrder::Little) => u64::from_le_bytes(bytes.try_into().ok()?),
                    (8, ByteOrder::Big) => u64::from_be_bytes(bytes.try_into().ok()?),
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Try to decode this attribute as a scalar f64 value.
    pub fn as_f64(&self) -> Option<f64> {
        match &self.dtype {
            Dtype::Float { size: 8, order } => {
                if self.data.len() < 8 {
                    return None;
                }
                let arr: [u8; 8] = self.data[..8].try_into().ok()?;
                Some(match order {
                    ByteOrder::Little => f64::from_le_bytes(arr),
                    ByteOrder::Big => f64::from_be_bytes(arr),
                })
            }
            Dtype::Float { size: 4, order } => {
                if self.data.len() < 4 {
                    return None;
                }
                let arr: [u8; 4] = self.data[..4].try_into().ok()?;
                Some(match order {
                    ByteOrder::Little => f64::from(f32::from_le_bytes(arr)),
                    ByteOrder::Big => f64::from(f32::from_be_bytes(arr)),
                })
            }
            _ => None,
        }
    }

    /// Try to decode a fixed-length string attribute as a `String`.
    ///
    /// Trims NUL padding.  Returns `None` for vlen-string or non-string dtypes.
    pub fn as_str_fixed(&self) -> Option<String> {
        match &self.dtype {
            Dtype::String {
                fixed_len: Some(n), ..
            } => {
                let n = *n;
                let bytes = self.data.get(..n)?;
                let trimmed = bytes.split(|&b| b == 0).next().unwrap_or(bytes);
                String::from_utf8(trimmed.to_vec()).ok()
            }
            _ => None,
        }
    }

    /// Returns true if the dataspace is a scalar or has exactly one element.
    pub fn is_scalar(&self) -> bool {
        match &self.dataspace {
            Dataspace::Scalar => true,
            Dataspace::Simple { dims, .. } => dims.iter().product::<u64>() <= 1,
            Dataspace::Null => false,
        }
    }

    /// Returns the shape of the attribute's dataspace as a `Vec<u64>`.
    ///
    /// Scalar → `[]`, Null → `[0]`, Simple → the dim vec.
    pub fn shape(&self) -> Vec<u64> {
        match &self.dataspace {
            Dataspace::Scalar => vec![],
            Dataspace::Null => vec![0],
            Dataspace::Simple { dims, .. } => dims.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilterInfo {
    pub id: u16,
    pub name: Option<std::string::String>,
    pub flags: u16,
    pub client_data: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilterPipeline {
    pub filters: Vec<FilterInfo>,
}

#[derive(Debug, Clone)]
pub struct PropertyList {
    pub chunk_dims: Option<Vec<u64>>,
    pub filters: Option<FilterPipeline>,
    pub fill_value: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Link {
    Hard {
        address: u64,
    },
    Soft {
        path: std::string::String,
    },
    External {
        file: std::string::String,
        path: std::string::String,
    },
}

#[derive(Debug, Clone)]
pub struct Group {
    pub name: std::string::String,
    pub children: Vec<(std::string::String, Link)>,
    pub attributes: Vec<Attribute>,
}

pub struct Dataset {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub attributes: Vec<Attribute>,
    /// Maximum dimensions from the HDF5 simple dataspace, if present.
    ///
    /// A value of `u64::MAX` (`H5S_UNLIMITED`) on an axis indicates that
    /// axis is extendable without bound.  `None` means the dataspace message
    /// did not carry max-dims information (older or compact files).
    pub max_dims: Option<Vec<u64>>,
}

impl Dataset {
    /// Return all attributes attached to this dataset.
    pub fn attrs(&self) -> &[Attribute] {
        &self.attributes
    }

    /// Find a single attribute by name.
    pub fn attr(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Return the maximum dimensions as a slice, if present.
    ///
    /// Returns `None` when the HDF5 dataspace message did not carry max-dims
    /// (e.g. compact/old-style files without the max-dims flag).
    pub fn max_dims(&self) -> Option<&[u64]> {
        self.max_dims.as_deref()
    }

    /// Return `true` if any axis has `H5S_UNLIMITED` maximum dimensions.
    ///
    /// An unlimited axis has `max_dims[i] == u64::MAX`.
    /// Returns `false` when `max_dims` is absent.
    pub fn is_unlimited(&self) -> bool {
        self.max_dims
            .as_ref()
            .is_some_and(|md| md.contains(&u64::MAX))
    }

    /// Return the indices of axes that are unlimited (`H5S_UNLIMITED`).
    ///
    /// Returns an empty `Vec` when no axis is unlimited or when `max_dims`
    /// is absent.
    pub fn unlimited_axes(&self) -> Vec<usize> {
        self.max_dims.as_ref().map_or_else(Vec::new, |md| {
            md.iter()
                .enumerate()
                .filter(|(_, &d)| d == u64::MAX)
                .map(|(i, _)| i)
                .collect()
        })
    }

    pub fn len(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Extract a sub-region of the dataset using index ranges.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Dataset, OxiH5Error> {
        let ndims = self.shape.len();
        if ranges.len() != ndims {
            return Err(OxiH5Error::Format(format!(
                "slice: {} ranges for {} dimensions",
                ranges.len(),
                ndims
            )));
        }

        for (dim, (range, &dim_size)) in ranges.iter().zip(self.shape.iter()).enumerate() {
            if range.start > range.end {
                return Err(OxiH5Error::Format(format!(
                    "slice: invalid range {}..{}",
                    range.start, range.end
                )));
            }
            if range.end > dim_size {
                return Err(OxiH5Error::Format(format!(
                    "slice: range {}..{} out of bounds for dimension {} (size {})",
                    range.start, range.end, dim, dim_size
                )));
            }
        }

        let elem_size = self.dtype_size()?;
        let out_shape: Vec<usize> = ranges.iter().map(|r| r.len()).collect();
        let out_elems: usize = out_shape.iter().product();
        let mut out_data = vec![0u8; out_elems * elem_size];

        if out_elems == 0 {
            return Ok(Dataset {
                data: out_data,
                shape: out_shape,
                dtype: self.dtype.clone(),
                attributes: self.attributes.clone(),
                max_dims: self.max_dims.clone(),
            });
        }

        let mut src_strides = vec![1usize; ndims];
        for d in (0..ndims.saturating_sub(1)).rev() {
            src_strides[d] = src_strides[d + 1] * self.shape[d + 1];
        }

        let mut coords = vec![0usize; ndims];
        let mut dst_flat = 0usize;
        loop {
            let src_flat: usize = coords
                .iter()
                .enumerate()
                .map(|(d, &c)| (ranges[d].start + c) * src_strides[d])
                .sum();

            let src_off = src_flat * elem_size;
            let dst_off = dst_flat * elem_size;
            if src_off + elem_size <= self.data.len() && dst_off + elem_size <= out_data.len() {
                out_data[dst_off..dst_off + elem_size]
                    .copy_from_slice(&self.data[src_off..src_off + elem_size]);
            }
            dst_flat += 1;

            let mut carry = true;
            for d in (0..ndims).rev() {
                if carry {
                    coords[d] += 1;
                    if coords[d] >= out_shape[d] {
                        coords[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        Ok(Dataset {
            data: out_data,
            shape: out_shape,
            dtype: self.dtype.clone(),
            attributes: self.attributes.clone(),
            max_dims: self.max_dims.clone(),
        })
    }

    /// Reshape the dataset to a new shape (zero-copy — validates element count only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Dataset, OxiH5Error> {
        let old_count: usize = if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        };
        let new_count: usize = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product()
        };
        if old_count != new_count {
            return Err(OxiH5Error::Format(format!(
                "reshape: cannot reshape {} elements to shape {:?} ({} elements)",
                old_count, new_shape, new_count
            )));
        }
        Ok(Dataset {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype.clone(),
            attributes: self.attributes.clone(),
            max_dims: self.max_dims.clone(),
        })
    }
}

#[derive(Debug, Error)]
pub enum OxiH5Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid HDF5 signature")]
    BadSignature,

    #[error("unsupported superblock version: {0}")]
    UnsupportedSuperblock(u8),

    #[error("unsupported object header version: {0}")]
    UnsupportedHeader(u8),

    #[error("unsupported datatype class: {0}")]
    UnsupportedDatatype(u8),

    #[error("unsupported data layout class: {0}")]
    UnsupportedLayout(u8),

    #[error("dataset not found: {0}")]
    NotFound(String),

    #[error("type mismatch")]
    TypeMismatch,

    #[error("data buffer truncated")]
    DataTruncated,

    #[error("not yet implemented: {0}")]
    NotImplemented(String),

    #[error("format error: {0}")]
    Format(String),

    #[error("unsupported filter: {0}")]
    UnsupportedFilter(String),

    #[error("corrupted data: {0}")]
    Corrupted(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(
            Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little
            }
            .size(),
            Some(4)
        );
        assert_eq!(
            Dtype::Float {
                size: 8,
                order: ByteOrder::Big
            }
            .size(),
            Some(8)
        );
        assert_eq!(
            Dtype::String {
                fixed_len: Some(10),
                charset: Charset::Ascii
            }
            .size(),
            Some(10)
        );
        assert_eq!(
            Dtype::String {
                fixed_len: None,
                charset: Charset::Utf8
            }
            .size(),
            None
        );
        assert_eq!(
            Dtype::Array {
                base: Box::new(Dtype::Int {
                    size: 2,
                    signed: false,
                    order: ByteOrder::Little
                }),
                dims: vec![3, 4]
            }
            .size(),
            Some(2 * 12)
        );
        assert_eq!(
            Dtype::Enum {
                base: Box::new(Dtype::Int {
                    size: 4,
                    signed: true,
                    order: ByteOrder::Little
                }),
                members: vec![("a".into(), 0), ("b".into(), 1)]
            }
            .size(),
            Some(4)
        );
        assert_eq!(
            Dtype::Compound {
                fields: vec![
                    CompoundField {
                        name: "x".into(),
                        offset: 0,
                        dtype: Dtype::Int {
                            size: 4,
                            signed: true,
                            order: ByteOrder::Little
                        }
                    },
                    CompoundField {
                        name: "y".into(),
                        offset: 4,
                        dtype: Dtype::Float {
                            size: 8,
                            order: ByteOrder::Little
                        }
                    }
                ]
            }
            .size(),
            Some(12)
        );
        assert_eq!(
            Dtype::Reference {
                ref_type: RefType::Object
            }
            .size(),
            Some(8)
        );
        assert_eq!(
            Dtype::VarLen {
                base: Box::new(Dtype::Int {
                    size: 4,
                    signed: true,
                    order: ByteOrder::Little
                })
            }
            .size(),
            None
        );
    }

    fn make_f32_le_dataset(values: &[f32]) -> Dataset {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Dataset {
            data,
            shape: vec![values.len()],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        }
    }

    fn make_i32_le_dataset(values: &[i32]) -> Dataset {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Dataset {
            data,
            shape: vec![values.len()],
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        }
    }

    #[test]
    fn test_byte_orders() {
        let ds = make_f32_le_dataset(&[1.0_f32, 2.0, 3.0]);
        let v = ds.as_f32().unwrap();
        assert_eq!(v, vec![1.0_f32, 2.0, 3.0]);

        let data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let ds = Dataset {
            data,
            shape: vec![2],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Big,
            },
            attributes: vec![],
            max_dims: None,
        };
        let v = ds.as_f32().unwrap();
        assert_eq!(v, vec![1.0_f32, 2.0]);

        let ds = make_i32_le_dataset(&[10, -5]);
        assert_eq!(ds.as_i32().unwrap(), vec![10, -5]);
    }

    #[test]
    fn test_type_mismatch() {
        let ds = make_f32_le_dataset(&[1.0]);
        assert!(matches!(ds.as_i32(), Err(OxiH5Error::TypeMismatch)));
        assert!(matches!(ds.as_u8(), Err(OxiH5Error::TypeMismatch)));
    }

    #[test]
    fn test_dataset_len() {
        let ds = Dataset {
            data: vec![0_u8; 4],
            shape: vec![],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        assert_eq!(ds.len(), 1);
        assert!(!ds.is_empty());

        let ds = make_f32_le_dataset(&[0.0; 5]);
        assert_eq!(ds.len(), 5);

        let data: Vec<u8> = vec![0_u8; 24];
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        assert_eq!(ds.len(), 6);
    }

    #[test]
    fn test_dtype_eq_clone() {
        let f = Dtype::Float {
            size: 4,
            order: ByteOrder::Little,
        };
        assert_eq!(f.clone(), f);
        let i = Dtype::Int {
            size: 4,
            signed: true,
            order: ByteOrder::Little,
        };
        assert_ne!(f, i);
        let s = Dtype::String {
            fixed_len: Some(10),
            charset: Charset::Utf8,
        };
        assert_eq!(s.clone(), s);
    }

    #[test]
    fn test_error_display() {
        let errors: Vec<OxiH5Error> = vec![
            OxiH5Error::BadSignature,
            OxiH5Error::UnsupportedSuperblock(3),
            OxiH5Error::TypeMismatch,
            OxiH5Error::NotFound("foo".into()),
            OxiH5Error::UnsupportedFilter("lzf".into()),
            OxiH5Error::Corrupted("bad checksum".into()),
        ];
        for e in errors {
            assert!(
                !e.to_string().is_empty(),
                "OxiH5Error display was empty for: {:?}",
                e
            );
        }
    }

    #[test]
    fn test_f16_decode() {
        let data = vec![0x3C_u8, 0x00, 0x38, 0x00];
        let ds = Dataset {
            data,
            shape: vec![2],
            dtype: Dtype::Float {
                size: 2,
                order: ByteOrder::Big,
            },
            attributes: vec![],
            max_dims: None,
        };
        let v = ds.as_f16().unwrap();
        assert!((v[0] - 1.0_f32).abs() < 1e-6);
        assert!((v[1] - 0.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_integer_converters() {
        let data: Vec<u8> = [100_u16, 200_u16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ds = Dataset {
            data,
            shape: vec![2],
            dtype: Dtype::Int {
                size: 2,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        assert_eq!(ds.as_u16().unwrap(), vec![100_u16, 200_u16]);

        let ds = Dataset {
            data: vec![0x7F, 0x80],
            shape: vec![2],
            dtype: Dtype::Int {
                size: 1,
                signed: true,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        assert_eq!(ds.as_i8().unwrap(), vec![127_i8, -128_i8]);
    }

    #[test]
    fn test_slice_1d() {
        let data: Vec<u8> = (0u8..10).collect();
        let ds = Dataset {
            data,
            shape: vec![10],
            dtype: Dtype::Int {
                size: 1,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let ranges = [std::ops::Range { start: 2, end: 5 }];
        let sliced = ds.slice(&ranges).unwrap();
        assert_eq!(sliced.shape, vec![3]);
        assert_eq!(sliced.data, vec![2u8, 3, 4]);
    }

    #[test]
    fn test_slice_2d() {
        let data: Vec<u8> = (0u8..12).collect();
        let ds = Dataset {
            data,
            shape: vec![3, 4],
            dtype: Dtype::Int {
                size: 1,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let ranges = [
            std::ops::Range { start: 1, end: 3 },
            std::ops::Range { start: 1, end: 3 },
        ];
        let sliced = ds.slice(&ranges).unwrap();
        assert_eq!(sliced.shape, vec![2, 2]);
        assert_eq!(sliced.data, vec![5u8, 6, 9, 10]);
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let ds = Dataset {
            data: vec![0u8; 4],
            shape: vec![4],
            dtype: Dtype::Int {
                size: 1,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let ranges = [std::ops::Range { start: 0, end: 10 }];
        assert!(ds.slice(&ranges).is_err());
    }

    #[test]
    fn test_slice_empty() {
        let ds = Dataset {
            data: vec![0u8; 4],
            shape: vec![4],
            dtype: Dtype::Int {
                size: 1,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let ranges = [std::ops::Range { start: 2, end: 2 }];
        let empty = ds.slice(&ranges).unwrap();
        assert_eq!(empty.shape, vec![0]);
        assert!(empty.data.is_empty());
    }

    #[test]
    fn test_reshape() {
        let ds = Dataset {
            data: vec![0u8; 12],
            shape: vec![3, 4],
            dtype: Dtype::Int {
                size: 1,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let reshaped = ds.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape, vec![2, 6]);
        assert_eq!(reshaped.data.len(), 12);

        assert!(ds.reshape(&[2, 7]).is_err());
    }

    #[test]
    fn test_attribute_scalar_accessors() {
        // Test as_i64 with i32 LE
        let attr = Attribute {
            name: "count".into(),
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            dataspace: Dataspace::Scalar,
            data: 42i32.to_le_bytes().to_vec(),
        };
        assert_eq!(attr.as_i64(), Some(42));
        assert!(attr.is_scalar());
        assert_eq!(attr.shape(), vec![]);

        // Test as_f64 with f64 LE
        let pi = std::f64::consts::PI;
        let attr_f = Attribute {
            name: "scale".into(),
            dtype: Dtype::Float {
                size: 8,
                order: ByteOrder::Little,
            },
            dataspace: Dataspace::Scalar,
            data: pi.to_le_bytes().to_vec(),
        };
        let v = attr_f.as_f64().unwrap();
        assert!((v - pi).abs() < 1e-10);

        // Test as_str_fixed
        let mut str_data = b"hello\0\0\0".to_vec();
        str_data.resize(8, 0);
        let attr_s = Attribute {
            name: "label".into(),
            dtype: Dtype::String {
                fixed_len: Some(8),
                charset: Charset::Utf8,
            },
            dataspace: Dataspace::Scalar,
            data: str_data,
        };
        assert_eq!(attr_s.as_str_fixed(), Some("hello".into()));
    }

    // -----------------------------------------------------------------------
    // A1 — Dataset::max_dims / is_unlimited / unlimited_axes unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_unlimited_true_when_any_axis_is_max() {
        let ds = Dataset {
            data: vec![0u8; 8],
            shape: vec![2],
            dtype: Dtype::Int {
                size: 4,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: Some(vec![u64::MAX, 10]),
        };
        assert!(ds.is_unlimited(), "expected is_unlimited() == true");
        assert_eq!(ds.unlimited_axes(), vec![0usize]);
        assert_eq!(ds.max_dims(), Some([u64::MAX, 10].as_slice()));
    }

    #[test]
    fn test_is_unlimited_false_when_no_unlimited_axis() {
        let ds = Dataset {
            data: vec![0u8; 8],
            shape: vec![2, 5],
            dtype: Dtype::Int {
                size: 4,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: Some(vec![100, 200]),
        };
        assert!(!ds.is_unlimited(), "fixed max_dims should not be unlimited");
        assert!(ds.unlimited_axes().is_empty());
    }

    #[test]
    fn test_is_unlimited_false_when_no_max_dims() {
        let ds = Dataset {
            data: vec![0u8; 4],
            shape: vec![1],
            dtype: Dtype::Int {
                size: 4,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        assert!(!ds.is_unlimited());
        assert!(ds.unlimited_axes().is_empty());
        assert!(ds.max_dims().is_none());
    }

    #[test]
    fn test_unlimited_axes_multiple() {
        // Both axes unlimited
        let ds = Dataset {
            data: vec![0u8; 4],
            shape: vec![1, 1],
            dtype: Dtype::Int {
                size: 4,
                signed: false,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: Some(vec![u64::MAX, u64::MAX]),
        };
        assert_eq!(ds.unlimited_axes(), vec![0usize, 1usize]);
    }
}

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod ndarray_tests {
    use super::*;

    #[test]
    fn test_to_array_f32() {
        let ds = Dataset {
            data: vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40],
            shape: vec![2],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_f32().unwrap();
        assert_eq!(arr.shape(), &[2]);
        assert!((arr[[0]] - 1.0_f32).abs() < 1e-6);
        assert!((arr[[1]] - 2.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_to_array_i32_2d() {
        let data: Vec<u8> = (0i32..6).flat_map(|v| v.to_le_bytes()).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_i32().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[1, 2]], 5);
    }

    #[test]
    fn test_to_array_u8() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: false,
                size: 1,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_u8().expect("to_array_u8");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1u8);
        assert_eq!(arr[[1, 2]], 6u8);
    }

    #[test]
    fn test_to_array_u16() {
        let data: Vec<u8> = (1u16..=6).flat_map(|v| v.to_le_bytes()).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: false,
                size: 2,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_u16().expect("to_array_u16");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1u16);
        assert_eq!(arr[[1, 2]], 6u16);
    }

    #[test]
    fn test_to_array_u32() {
        let data: Vec<u8> = (1u32..=6).flat_map(|v| v.to_le_bytes()).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: false,
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_u32().expect("to_array_u32");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1u32);
        assert_eq!(arr[[1, 2]], 6u32);
    }

    #[test]
    fn test_to_array_u64() {
        let data: Vec<u8> = (1u64..=6).flat_map(|v| v.to_le_bytes()).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: false,
                size: 8,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_u64().expect("to_array_u64");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1u64);
        assert_eq!(arr[[1, 2]], 6u64);
    }

    #[test]
    fn test_to_array_i8() {
        let data: Vec<u8> = (-3i8..=2).map(|v| v as u8).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: true,
                size: 1,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_i8().expect("to_array_i8");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], -3i8);
        assert_eq!(arr[[1, 2]], 2i8);
    }

    #[test]
    fn test_to_array_i16() {
        let data: Vec<u8> = (-3i16..=2).flat_map(|v| v.to_le_bytes()).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: true,
                size: 2,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_i16().expect("to_array_i16");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], -3i16);
        assert_eq!(arr[[1, 2]], 2i16);
    }

    #[test]
    fn test_to_array_i64() {
        let data: Vec<u8> = (1i64..=6).flat_map(|v| v.to_le_bytes()).collect();
        let ds = Dataset {
            data,
            shape: vec![2, 3],
            dtype: Dtype::Int {
                signed: true,
                size: 8,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_i64().expect("to_array_i64");
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1i64);
        assert_eq!(arr[[1, 2]], 6i64);
    }

    #[test]
    fn test_to_array_f16() {
        let data: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40];
        let ds = Dataset {
            data,
            shape: vec![2],
            dtype: Dtype::Float {
                size: 2,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        let arr = ds.to_array_f16().expect("to_array_f16");
        assert_eq!(arr.shape(), &[2]);
        assert!((arr[[0]] - 1.0_f32).abs() < 1e-3);
        assert!((arr[[1]] - 2.0_f32).abs() < 1e-3);
    }

    #[test]
    fn test_to_array_u8_type_mismatch() {
        let ds = Dataset {
            data: vec![0u8, 0, 0, 0],
            shape: vec![1],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
            max_dims: None,
        };
        assert!(ds.to_array_u8().is_err());
    }
}
