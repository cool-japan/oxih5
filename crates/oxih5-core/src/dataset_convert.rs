//! Typed value decoders and lazy iterators for [`Dataset`].
//!
//! This module is `pub(crate)` — all public items are re-exported from
//! the parent `lib.rs` via `impl Dataset`.

use crate::{ByteOrder, Dataset, Dtype, OxiH5Error};

/// Software decode of an IEEE 754 half-precision float (binary16) to f32.
///
/// Half-precision layout: 1 sign bit, 5 exponent bits, 10 mantissa bits.
/// Special cases: subnormals, ±infinity, NaN are all handled correctly.
pub fn f16_to_f32(bits: u16) -> f32 {
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

impl Dataset {
    /// Returns the byte size of a single element for fixed-width dtypes.
    pub(crate) fn dtype_size(&self) -> Result<usize, OxiH5Error> {
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

    pub fn as_string(&self) -> Result<Vec<String>, OxiH5Error> {
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
                    let s = String::from_utf8(trimmed)
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

    // -----------------------------------------------------------------------
    // Lazy iterators — stream typed values directly from the raw byte buffer
    // without allocating an intermediate Vec.
    // -----------------------------------------------------------------------

    /// Lazily iterate over `f32` values decoded from a 32-bit float dataset.
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
}

// ---------------------------------------------------------------------------
// ndarray bridge (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
impl Dataset {
    /// Convert typed data to an `ndarray::ArrayD<f32>`. Requires `ndarray` feature.
    pub fn to_array_f32(&self) -> Result<ndarray::ArrayD<f32>, OxiH5Error> {
        let values = self.as_f32()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<f64>`. Requires `ndarray` feature.
    pub fn to_array_f64(&self) -> Result<ndarray::ArrayD<f64>, OxiH5Error> {
        let values = self.as_f64()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<i32>`. Requires `ndarray` feature.
    pub fn to_array_i32(&self) -> Result<ndarray::ArrayD<i32>, OxiH5Error> {
        let values = self.as_i32()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<u8>`. Requires `ndarray` feature.
    pub fn to_array_u8(&self) -> Result<ndarray::ArrayD<u8>, OxiH5Error> {
        let values = self.as_u8()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<u16>`. Requires `ndarray` feature.
    pub fn to_array_u16(&self) -> Result<ndarray::ArrayD<u16>, OxiH5Error> {
        let values = self.as_u16()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<u32>`. Requires `ndarray` feature.
    pub fn to_array_u32(&self) -> Result<ndarray::ArrayD<u32>, OxiH5Error> {
        let values = self.as_u32()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<u64>`. Requires `ndarray` feature.
    pub fn to_array_u64(&self) -> Result<ndarray::ArrayD<u64>, OxiH5Error> {
        let values = self.as_u64()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<i8>`. Requires `ndarray` feature.
    pub fn to_array_i8(&self) -> Result<ndarray::ArrayD<i8>, OxiH5Error> {
        let values = self.as_i8()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<i16>`. Requires `ndarray` feature.
    pub fn to_array_i16(&self) -> Result<ndarray::ArrayD<i16>, OxiH5Error> {
        let values = self.as_i16()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert typed data to an `ndarray::ArrayD<i64>`. Requires `ndarray` feature.
    pub fn to_array_i64(&self) -> Result<ndarray::ArrayD<i64>, OxiH5Error> {
        let values = self.as_i64()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }

    /// Convert half-precision float data to an `ndarray::ArrayD<f32>`. Requires `ndarray` feature.
    pub fn to_array_f16(&self) -> Result<ndarray::ArrayD<f32>, OxiH5Error> {
        let values = self.as_f16()?;
        let shape = if self.shape.is_empty() {
            vec![1]
        } else {
            self.shape.clone()
        };
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
            .map_err(|e| OxiH5Error::Format(format!("ndarray shape error: {}", e)))
    }
}
