#![forbid(unsafe_code)]

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
    /// not fixed by the datatype alone, so this returns `None`.  Fixed-width
    /// numeric, opaque, bitfield, reference, compound, array and enum types all
    /// return a concrete size.
    pub fn size(&self) -> Option<usize> {
        match self {
            Dtype::Int { size, .. }
            | Dtype::Float { size, .. }
            | Dtype::Bitfield { size, .. }
            | Dtype::Opaque { size, .. } => Some(*size),
            Dtype::String { fixed_len, .. } => *fixed_len,
            // Object references are 8 bytes, region references 12 bytes in the
            // common 8-byte-offset file; we report the object-reference size
            // which is what fixed-width reference datasets use.
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
                // Compound size is the max(offset + member_size); members may be
                // padded, so this is a lower bound when trailing padding exists.
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

    pub fn as_f32(&self) -> Result<Vec<f32>, OxiH5Error> {
        match &self.dtype {
            Dtype::Float { size: 4, order } => {
                if self.data.len() % 4 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 4);
                for chunk in self.data.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => f32::from_le_bytes(arr),
                        ByteOrder::Big => f32::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_f64(&self) -> Result<Vec<f64>, OxiH5Error> {
        match &self.dtype {
            Dtype::Float { size: 8, order } => {
                if self.data.len() % 8 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 8);
                for chunk in self.data.chunks_exact(8) {
                    let arr: [u8; 8] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => f64::from_le_bytes(arr),
                        ByteOrder::Big => f64::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_i32(&self) -> Result<Vec<i32>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 4,
                signed: true,
                order,
            } => {
                if self.data.len() % 4 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 4);
                for chunk in self.data.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => i32::from_le_bytes(arr),
                        ByteOrder::Big => i32::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_u8(&self) -> Result<Vec<u8>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 1,
                signed: false,
                ..
            } => Ok(self.data.clone()),
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_u16(&self) -> Result<Vec<u16>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 2,
                signed: false,
                order,
            } => {
                if self.data.len() % 2 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 2);
                for chunk in self.data.chunks_exact(2) {
                    let arr: [u8; 2] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => u16::from_le_bytes(arr),
                        ByteOrder::Big => u16::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_u32(&self) -> Result<Vec<u32>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 4,
                signed: false,
                order,
            } => {
                if self.data.len() % 4 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 4);
                for chunk in self.data.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => u32::from_le_bytes(arr),
                        ByteOrder::Big => u32::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_u64(&self) -> Result<Vec<u64>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 8,
                signed: false,
                order,
            } => {
                if self.data.len() % 8 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 8);
                for chunk in self.data.chunks_exact(8) {
                    let arr: [u8; 8] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => u64::from_le_bytes(arr),
                        ByteOrder::Big => u64::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_i8(&self) -> Result<Vec<i8>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 1,
                signed: true,
                ..
            } => Ok(self.data.iter().map(|&b| b as i8).collect()),
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_i16(&self) -> Result<Vec<i16>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 2,
                signed: true,
                order,
            } => {
                if self.data.len() % 2 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 2);
                for chunk in self.data.chunks_exact(2) {
                    let arr: [u8; 2] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => i16::from_le_bytes(arr),
                        ByteOrder::Big => i16::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_i64(&self) -> Result<Vec<i64>, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 8,
                signed: true,
                order,
            } => {
                if self.data.len() % 8 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 8);
                for chunk in self.data.chunks_exact(8) {
                    let arr: [u8; 8] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let v = match order {
                        ByteOrder::Little => i64::from_le_bytes(arr),
                        ByteOrder::Big => i64::from_be_bytes(arr),
                    };
                    out.push(v);
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_f16(&self) -> Result<Vec<f32>, OxiH5Error> {
        match &self.dtype {
            Dtype::Float { size: 2, order } => {
                if self.data.len() % 2 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / 2);
                for chunk in self.data.chunks_exact(2) {
                    let arr: [u8; 2] = chunk.try_into().map_err(|_| OxiH5Error::DataTruncated)?;
                    let bits = match order {
                        ByteOrder::Little => u16::from_le_bytes(arr),
                        ByteOrder::Big => u16::from_be_bytes(arr),
                    };
                    out.push(f16_to_f32(bits));
                }
                Ok(out)
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    pub fn as_string(&self) -> Result<Vec<std::string::String>, OxiH5Error> {
        match &self.dtype {
            Dtype::String {
                fixed_len: Some(n), ..
            } => {
                let n = *n;
                if n == 0 {
                    return Err(OxiH5Error::Format("fixed string length is zero".into()));
                }
                if self.data.len() % n != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let mut out = Vec::with_capacity(self.data.len() / n);
                for chunk in self.data.chunks_exact(n) {
                    let trimmed: Vec<u8> = chunk.iter().copied().take_while(|&b| b != 0).collect();
                    let s = std::string::String::from_utf8(trimmed)
                        .map_err(|e| OxiH5Error::Format(format!("invalid UTF-8: {}", e)))?;
                    out.push(s);
                }
                Ok(out)
            }
            Dtype::String {
                fixed_len: None, ..
            } => Err(OxiH5Error::NotImplemented(
                "VarLen string decode".to_string(),
            )),
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Returns the byte size of a single element for fixed-width dtypes.
    fn dtype_size(&self) -> Result<usize, OxiH5Error> {
        match &self.dtype {
            Dtype::Int { size, .. }
            | Dtype::Float { size, .. }
            | Dtype::Bitfield { size, .. }
            | Dtype::Opaque { size, .. } => Ok(*size),
            Dtype::String {
                fixed_len: Some(n), ..
            } => Ok(*n),
            _ => Err(OxiH5Error::NotImplemented(format!(
                "dtype_size for {:?}",
                self.dtype
            ))),
        }
    }

    /// Extract a sub-region of the dataset using index ranges.
    ///
    /// `ranges`: a slice of `std::ops::Range<usize>`, one per dimension.
    /// Returns a new `Dataset` containing only the selected elements.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Dataset, OxiH5Error> {
        let ndims = self.shape.len();
        if ranges.len() != ndims {
            return Err(OxiH5Error::Format(format!(
                "slice: {} ranges for {} dimensions",
                ranges.len(),
                ndims
            )));
        }

        // Validate ranges
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

        // Short-circuit for empty output
        if out_elems == 0 {
            return Ok(Dataset {
                data: out_data,
                shape: out_shape,
                dtype: self.dtype.clone(),
                attributes: self.attributes.clone(),
            });
        }

        // Compute row-major strides for the source shape
        let mut src_strides = vec![1usize; ndims];
        for d in (0..ndims.saturating_sub(1)).rev() {
            src_strides[d] = src_strides[d + 1] * self.shape[d + 1];
        }

        // Walk all multi-dimensional output positions
        let mut coords = vec![0usize; ndims];
        let mut dst_flat = 0usize;
        loop {
            // Compute the flat index in the source
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

            // Increment coords in last-dimension-first (row-major) order
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
        })
    }

    // -----------------------------------------------------------------------
    // Lazy iterators — stream typed values directly from the raw byte buffer
    // without allocating an intermediate Vec.
    // -----------------------------------------------------------------------

    /// Lazily iterate over `f32` values decoded from a 32-bit float dataset.
    ///
    /// Returns `Err(OxiH5Error::TypeMismatch)` when the dtype is not a 4-byte
    /// float.  Returns `Err(OxiH5Error::DataTruncated)` when the buffer length
    /// is not a multiple of 4.
    pub fn iter_f32(&self) -> Result<impl Iterator<Item = f32> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Float { size: 4, order } => {
                if self.data.len() % 4 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(4).map(move |chunk| {
                    let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    match order {
                        ByteOrder::Little => f32::from_le_bytes(arr),
                        ByteOrder::Big => f32::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `f64` values decoded from a 64-bit float dataset.
    pub fn iter_f64(&self) -> Result<impl Iterator<Item = f64> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Float { size: 8, order } => {
                if self.data.len() % 8 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(8).map(move |chunk| {
                    let arr: [u8; 8] = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    match order {
                        ByteOrder::Little => f64::from_le_bytes(arr),
                        ByteOrder::Big => f64::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `i32` values decoded from a signed 32-bit integer dataset.
    pub fn iter_i32(&self) -> Result<impl Iterator<Item = i32> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 4,
                signed: true,
                order,
            } => {
                if self.data.len() % 4 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(4).map(move |chunk| {
                    let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    match order {
                        ByteOrder::Little => i32::from_le_bytes(arr),
                        ByteOrder::Big => i32::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `u8` values from an unsigned 8-bit integer dataset.
    pub fn iter_u8(&self) -> Result<impl Iterator<Item = u8> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 1,
                signed: false,
                ..
            } => Ok(self.data.iter().copied()),
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `i8` values from a signed 8-bit integer dataset.
    pub fn iter_i8(&self) -> Result<impl Iterator<Item = i8> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 1,
                signed: true,
                ..
            } => Ok(self.data.iter().map(|&b| b as i8)),
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `u16` values decoded from an unsigned 16-bit integer dataset.
    pub fn iter_u16(&self) -> Result<impl Iterator<Item = u16> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 2,
                signed: false,
                order,
            } => {
                if self.data.len() % 2 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(2).map(move |chunk| {
                    let arr: [u8; 2] = [chunk[0], chunk[1]];
                    match order {
                        ByteOrder::Little => u16::from_le_bytes(arr),
                        ByteOrder::Big => u16::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `i16` values decoded from a signed 16-bit integer dataset.
    pub fn iter_i16(&self) -> Result<impl Iterator<Item = i16> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 2,
                signed: true,
                order,
            } => {
                if self.data.len() % 2 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(2).map(move |chunk| {
                    let arr: [u8; 2] = [chunk[0], chunk[1]];
                    match order {
                        ByteOrder::Little => i16::from_le_bytes(arr),
                        ByteOrder::Big => i16::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `u32` values decoded from an unsigned 32-bit integer dataset.
    pub fn iter_u32(&self) -> Result<impl Iterator<Item = u32> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 4,
                signed: false,
                order,
            } => {
                if self.data.len() % 4 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(4).map(move |chunk| {
                    let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    match order {
                        ByteOrder::Little => u32::from_le_bytes(arr),
                        ByteOrder::Big => u32::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `i64` values decoded from a signed 64-bit integer dataset.
    pub fn iter_i64(&self) -> Result<impl Iterator<Item = i64> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 8,
                signed: true,
                order,
            } => {
                if self.data.len() % 8 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(8).map(move |chunk| {
                    let arr: [u8; 8] = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    match order {
                        ByteOrder::Little => i64::from_le_bytes(arr),
                        ByteOrder::Big => i64::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `u64` values decoded from an unsigned 64-bit integer dataset.
    pub fn iter_u64(&self) -> Result<impl Iterator<Item = u64> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Int {
                size: 8,
                signed: false,
                order,
            } => {
                if self.data.len() % 8 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(8).map(move |chunk| {
                    let arr: [u8; 8] = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    match order {
                        ByteOrder::Little => u64::from_le_bytes(arr),
                        ByteOrder::Big => u64::from_be_bytes(arr),
                    }
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Lazily iterate over `f32` values decoded from a 16-bit (half-precision) float dataset.
    ///
    /// Each f16 is decoded to f32 via software conversion.
    pub fn iter_f16(&self) -> Result<impl Iterator<Item = f32> + '_, OxiH5Error> {
        match &self.dtype {
            Dtype::Float { size: 2, order } => {
                if self.data.len() % 2 != 0 {
                    return Err(OxiH5Error::DataTruncated);
                }
                let order = *order;
                Ok(self.data.chunks_exact(2).map(move |chunk| {
                    let arr: [u8; 2] = [chunk[0], chunk[1]];
                    let bits = match order {
                        ByteOrder::Little => u16::from_le_bytes(arr),
                        ByteOrder::Big => u16::from_be_bytes(arr),
                    };
                    f16_to_f32(bits)
                }))
            }
            _ => Err(OxiH5Error::TypeMismatch),
        }
    }

    /// Validate and "reshape" the dataset to a new shape.
    ///
    /// This does not copy or reorder data — it just validates the total element
    /// count matches, then returns a new `Dataset` with the new shape and
    /// the same (cloned) data.
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
        })
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    let f32_bits: u32 = if exp == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            let mut m = mantissa;
            let mut e = 127u32.wrapping_sub(14);
            while m & 0x400 == 0 {
                m <<= 1;
                e = e.wrapping_sub(1);
            }
            m &= 0x3FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        let e = exp + 127 - 15;
        (sign << 31) | (e << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

// ---------------------------------------------------------------------------
// ndarray bridge (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
impl Dataset {
    /// Convert typed data to an `ndarray::ArrayD<f32>`.
    ///
    /// Requires the `ndarray` feature.
    pub fn to_array_f32(&self) -> Result<ndarray::ArrayD<f32>, OxiH5Error> {
        let values = self.as_f32()?;
        let shape: Vec<usize> = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<f64>`.
    ///
    /// Requires the `ndarray` feature.
    pub fn to_array_f64(&self) -> Result<ndarray::ArrayD<f64>, OxiH5Error> {
        let values = self.as_f64()?;
        let shape: Vec<usize> = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<i32>`.
    ///
    /// Requires the `ndarray` feature.
    pub fn to_array_i32(&self) -> Result<ndarray::ArrayD<i32>, OxiH5Error> {
        let values = self.as_i32()?;
        let shape: Vec<usize> = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
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
        // Variable-length string has no fixed element size.
        assert_eq!(
            Dtype::String {
                fixed_len: None,
                charset: Charset::Utf8
            }
            .size(),
            None
        );
        // Array: base size * product(dims).
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
        // Enum inherits its base type's size.
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
        // Compound: max(offset + member size).
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
        // Object reference is 8 bytes, variable-length has no fixed size.
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
        };
        let ranges = [std::ops::Range { start: 2, end: 5 }];
        let sliced = ds.slice(&ranges).unwrap();
        assert_eq!(sliced.shape, vec![3]);
        assert_eq!(sliced.data, vec![2u8, 3, 4]);
    }

    #[test]
    fn test_slice_2d() {
        // 3x4 matrix of u8 = [0..12]
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
        };
        // Row 1: [4,5,6,7], Row 2: [8,9,10,11]
        // Slice rows 1..3, cols 1..3 = [5,6,9,10]
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
        };
        let reshaped = ds.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape, vec![2, 6]);
        assert_eq!(reshaped.data.len(), 12);

        // Wrong element count should fail
        assert!(ds.reshape(&[2, 7]).is_err());
    }
}

#[cfg(feature = "ndarray")]
#[cfg(test)]
mod ndarray_tests {
    use super::*;

    #[test]
    fn test_to_array_f32() {
        // 1.0f32 LE bytes = [0x00, 0x00, 0x80, 0x3F], 2.0f32 LE = [0x00, 0x00, 0x00, 0x40]
        let ds = Dataset {
            data: vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40],
            shape: vec![2],
            dtype: Dtype::Float {
                size: 4,
                order: ByteOrder::Little,
            },
            attributes: vec![],
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
        };
        let arr = ds.to_array_i32().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[1, 2]], 5);
    }
}
