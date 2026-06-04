use crate::error::NcError;
use oxih5::{Attribute, ByteOrder, Dtype};

/// A NetCDF dimension (resolved from an HDF5 dimension-scale dataset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcDimension {
    pub name: String,
    pub len: u64,
    pub id: u32,
    pub is_unlimited: bool,
}

/// One axis of a variable: the dimension it maps to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcAxis {
    pub dim_id: u32,
    pub name: String,
    pub len: u64,
}

/// A NetCDF attribute (thin wrapper over an oxih5 Attribute).
#[derive(Debug, Clone)]
pub struct NcAttribute {
    pub name: String,
    inner: Attribute,
}

fn decode_f64_le(bytes: &[u8], attr: &str) -> Result<f64, NcError> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f64 slice wrong length".into(),
        })?;
    Ok(f64::from_le_bytes(arr))
}

fn decode_f64_be(bytes: &[u8], attr: &str) -> Result<f64, NcError> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f64 slice wrong length".into(),
        })?;
    Ok(f64::from_be_bytes(arr))
}

fn decode_f32_le(bytes: &[u8], attr: &str) -> Result<f32, NcError> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f32 slice wrong length".into(),
        })?;
    Ok(f32::from_le_bytes(arr))
}

fn decode_f32_be(bytes: &[u8], attr: &str) -> Result<f32, NcError> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f32 slice wrong length".into(),
        })?;
    Ok(f32::from_be_bytes(arr))
}

/// Decode a fixed-size integer chunk into an i64.
///
/// Both signed and unsigned integers are returned as i64; unsigned values are
/// widened (small sizes fit without loss), and signed values are reinterpreted
/// via the two's-complement bit pattern.  `size` must match the length of
/// `bytes` exactly (guaranteed by the caller's `chunks_exact(sz)`).
fn decode_int_chunk(
    bytes: &[u8],
    size: usize,
    order: &ByteOrder,
    attr: &str,
) -> Result<i64, NcError> {
    let bad = || NcError::BadConventionAttribute {
        owner: String::new(),
        attr: attr.to_owned(),
        reason: "int slice wrong length".into(),
    };
    match (size, order) {
        (1, _) => Ok(i64::from(bytes[0] as i8)),
        (2, ByteOrder::Little) => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i16::from_le_bytes(arr)))
        }
        (2, ByteOrder::Big) => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i16::from_be_bytes(arr)))
        }
        (4, ByteOrder::Little) => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i32::from_le_bytes(arr)))
        }
        (4, ByteOrder::Big) => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i32::from_be_bytes(arr)))
        }
        (8, ByteOrder::Little) => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from_le_bytes(arr))
        }
        (8, ByteOrder::Big) => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from_be_bytes(arr))
        }
        _ => Err(NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: format!("unsupported integer size {size}"),
        }),
    }
}

impl NcAttribute {
    pub(crate) fn new(inner: Attribute) -> Self {
        Self {
            name: inner.name.clone(),
            inner,
        }
    }

    pub fn dtype(&self) -> &Dtype {
        &self.inner.dtype
    }

    /// Try to decode this attribute as a text string.
    /// Works for fixed-length string attributes.
    /// For vlen-string attributes, returns `NcError::Unsupported` until
    /// `AttrView::as_strings` (oxih5 A4) is available.
    pub fn as_text(&self) -> Result<String, NcError> {
        match &self.inner.dtype {
            Dtype::String {
                fixed_len: Some(n), ..
            } => {
                let n = *n;
                let bytes =
                    self.inner
                        .data
                        .get(..n)
                        .ok_or_else(|| NcError::BadConventionAttribute {
                            owner: String::new(),
                            attr: self.name.clone(),
                            reason: "attribute data shorter than fixed string length".into(),
                        })?;
                // Trim NUL padding.
                let trimmed = bytes.split(|&b| b == 0).next().unwrap_or(bytes);
                String::from_utf8(trimmed.to_vec()).map_err(|e| NcError::BadConventionAttribute {
                    owner: String::new(),
                    attr: self.name.clone(),
                    reason: format!("UTF-8 decode: {e}"),
                })
            }
            Dtype::String {
                fixed_len: None, ..
            } => Err(NcError::Unsupported(
                "vlen-string attribute decode requires AttrView (oxih5 A4)".into(),
            )),
            _ => Err(NcError::BadConventionAttribute {
                owner: String::new(),
                attr: self.name.clone(),
                reason: "attribute dtype is not a string".into(),
            }),
        }
    }

    /// Decode this attribute as a vector of f64 values.
    pub fn as_f64(&self) -> Result<Vec<f64>, NcError> {
        match &self.inner.dtype {
            Dtype::Float { size: 8, order } => {
                let data = &self.inner.data;
                if data.len() % 8 != 0 {
                    return Err(NcError::BadConventionAttribute {
                        owner: String::new(),
                        attr: self.name.clone(),
                        reason: "f64 data length not multiple of 8".into(),
                    });
                }
                let name = self.name.as_str();
                match order {
                    ByteOrder::Little => data
                        .chunks_exact(8)
                        .map(|c| decode_f64_le(c, name))
                        .collect(),
                    ByteOrder::Big => data
                        .chunks_exact(8)
                        .map(|c| decode_f64_be(c, name))
                        .collect(),
                }
            }
            Dtype::Float { size: 4, order } => {
                let data = &self.inner.data;
                if data.len() % 4 != 0 {
                    return Err(NcError::BadConventionAttribute {
                        owner: String::new(),
                        attr: self.name.clone(),
                        reason: "f32 data length not multiple of 4".into(),
                    });
                }
                let name = self.name.as_str();
                match order {
                    ByteOrder::Little => data
                        .chunks_exact(4)
                        .map(|c| decode_f32_le(c, name).map(f64::from))
                        .collect(),
                    ByteOrder::Big => data
                        .chunks_exact(4)
                        .map(|c| decode_f32_be(c, name).map(f64::from))
                        .collect(),
                }
            }
            _ => Err(NcError::BadConventionAttribute {
                owner: String::new(),
                attr: self.name.clone(),
                reason: "attribute dtype is not a float".into(),
            }),
        }
    }

    /// Decode this attribute as a vector of i64 values.
    pub fn as_i64(&self) -> Result<Vec<i64>, NcError> {
        match &self.inner.dtype {
            Dtype::Int { size, order, .. } => {
                let sz = *size;
                let data = &self.inner.data;
                if sz == 0 || data.len() % sz != 0 {
                    return Err(NcError::BadConventionAttribute {
                        owner: String::new(),
                        attr: self.name.clone(),
                        reason: "int data length not multiple of element size".into(),
                    });
                }
                let name = self.name.as_str();
                data.chunks_exact(sz)
                    .map(|c| decode_int_chunk(c, sz, order, name))
                    .collect()
            }
            _ => Err(NcError::BadConventionAttribute {
                owner: String::new(),
                attr: self.name.clone(),
                reason: "attribute dtype is not an integer".into(),
            }),
        }
    }

    /// Return the underlying oxih5 Attribute for escape-hatch access.
    pub fn raw(&self) -> &Attribute {
        &self.inner
    }
}

/// A NetCDF variable (an HDF5 dataset with DIMENSION_LIST, or a coord var).
#[derive(Debug, Clone)]
pub struct NcVariable {
    pub name: String,
    pub dtype: Dtype,
    pub dims: Vec<NcAxis>,
    pub shape: Vec<u64>,
    pub attrs: Vec<NcAttribute>,
    pub is_coordinate: bool,
    /// HDF5 object path of the underlying dataset (e.g. `"/group/varname"`).
    pub h5_path: String,
}

impl NcVariable {
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dim_names(&self) -> Vec<&str> {
        self.dims.iter().map(|a| a.name.as_str()).collect()
    }

    pub fn attr(&self, name: &str) -> Option<&NcAttribute> {
        self.attrs.iter().find(|a| a.name == name)
    }

    pub fn fill_value(&self) -> Option<&NcAttribute> {
        self.attr("_FillValue")
    }

    pub fn units(&self) -> Option<String> {
        self.attr("units").and_then(|a| a.as_text().ok())
    }
}

/// A NetCDF group (root or subgroup). Slice 1 fully resolves the root group.
#[derive(Debug, Clone)]
pub struct NcGroup {
    pub name: String,
    pub path: String,
    pub dimensions: Vec<NcDimension>,
    pub variables: Vec<NcVariable>,
    pub attrs: Vec<NcAttribute>,
    pub subgroup_names: Vec<String>,
}

impl NcGroup {
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    pub fn dimension_by_id(&self, id: u32) -> Option<&NcDimension> {
        self.dimensions.iter().find(|d| d.id == id)
    }

    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    pub fn attr(&self, name: &str) -> Option<&NcAttribute> {
        self.attrs.iter().find(|a| a.name == name)
    }
}
