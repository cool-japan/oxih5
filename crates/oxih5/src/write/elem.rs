//! Element-type and attribute-type data tables.
//!
//! The writer supports a small, fixed set of types.  Each of them used to be
//! described by hand-written byte literals repeated across six `match` sites —
//! four of which were literal copy-paste duplicates — so adding a type meant
//! editing six places, and any disagreement between an "allocate" site and a
//! "write" site silently produced a corrupt file.
//!
//! This module replaces that with one table per concept:
//!
//! * [`ElemType::spec`] is the only per-dataset-dtype `match`.  On-disk element
//!   size, datatype-message body size, and the datatype bytes themselves are all
//!   derived from it.
//! * [`ResolvedAttrKind::sizes`] is the only per-attribute-dtype size `match`,
//!   and attribute datatypes reuse the very same byte encoders as dataset
//!   datatypes.
//!
//! The encoders are the exact inverse of the reader's parser in
//! `oxih5_format::datatype`: the class/version byte carries `class | version <<
//! 4`, bit field 0 carries the byte order in bit 0 and (for fixed-point) the
//! sign flag in bit 3, the element size sits at `[4..8]`, and the version-1
//! properties section carries bit offset and bit precision at `[8..12]`.
//! [`tests::generated_dtype_bodies_match_the_historic_literals`] pins that
//! equivalence against the byte literals this module replaced.

use std::collections::HashMap;

use oxih5_core::{ByteOrder, Dtype, OxiH5Error};

use super::format::{fill_zero, write_u16_le, write_u32_le, write_u64_le};
use super::{check_size, narrow, pad8};

// ---------------------------------------------------------------------------
// Datatype message body sizes
// ---------------------------------------------------------------------------

/// On-disk size of a variable-length reference (`H5T__vlen_disk_write`).
pub(super) const VLEN_REF_SIZE: usize = 16;

/// Datatype body size for a class-0 fixed-point type: 12 used + 4 padding.
const FIXED_DT_BODY: usize = 16;
/// Datatype body size for a class-1 float type: 20 used + 4 padding.
const FLOAT_DT_BODY: usize = 24;
/// Datatype body size for a class-3 fixed-length string.
const STRING_DT_BODY: usize = 8;
/// Datatype body size for a class-7 object reference.
const REF_DT_BODY: usize = 8;
/// Datatype body size for a class-9 vlen string: 8 outer + 8 base type.
const VLEN_DT_BODY: usize = STRING_DT_BODY + STRING_DT_BODY;

/// Dataspace body size for a scalar attribute.
const SCALAR_DSPACE_BODY: usize = 8;
/// Dataspace body size for a 1-D attribute with max dims present.
const VECTOR_DSPACE_BODY: usize = 24;

/// Size of the fixed prefix of an attribute v1 message body.
const ATTR_BODY_PREFIX: usize = 8;

/// The "undefined address" sentinel used for unresolvable object references.
const UNDEFINED_ADDR: u64 = u64::MAX;

// ---------------------------------------------------------------------------
// Dataset element types
// ---------------------------------------------------------------------------

/// Element type of a writable dataset.
///
/// Every variant is little-endian: the writer serialises through
/// `to_le_bytes` and has no byte-swap path, which is why
/// [`dtype_to_elem_type`] refuses big-endian dtypes outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElemType {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary64.
    F64,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// HDF5 variable-length string (class 9, subtype 1).
    ///
    /// Each element in the dataset is a 16-byte global-heap reference.
    VlenStr,
}

/// On-disk encoding family of an [`ElemType`].
///
/// This is the writer's entire dtype table: every size and every byte of a
/// datatype message body is derived from one of these three shapes, so adding
/// an element type means adding one [`ElemType::spec`] arm and nothing else.
#[derive(Debug, Clone, Copy)]
enum ElemSpec {
    /// HDF5 class 0 — little-endian fixed-point integer, no padding.
    Fixed {
        /// Width in bytes.
        size: u8,
        /// Two's-complement signed when set.
        signed: bool,
    },
    /// HDF5 class 1 — little-endian IEEE-754 binary(`size * 8`) float.
    FloatIeee {
        /// Width in bytes.
        size: u8,
    },
    /// HDF5 class 9 — variable-length sequence of class-3 characters.
    VlenStr,
}

impl ElemType {
    /// The one and only per-[`ElemType`] match in the writer.
    const fn spec(self) -> ElemSpec {
        /// Two's-complement signed fixed-point of `size` bytes.
        const fn int(size: u8) -> ElemSpec {
            ElemSpec::Fixed { size, signed: true }
        }
        /// Unsigned fixed-point of `size` bytes.
        const fn uint(size: u8) -> ElemSpec {
            ElemSpec::Fixed {
                size,
                signed: false,
            }
        }
        match self {
            ElemType::F32 => ElemSpec::FloatIeee { size: 4 },
            ElemType::F64 => ElemSpec::FloatIeee { size: 8 },
            ElemType::I8 => int(1),
            ElemType::I16 => int(2),
            ElemType::I32 => int(4),
            ElemType::I64 => int(8),
            ElemType::U8 => uint(1),
            ElemType::U16 => uint(2),
            ElemType::U32 => uint(4),
            ElemType::U64 => uint(8),
            ElemType::VlenStr => ElemSpec::VlenStr,
        }
    }

    /// Every writable element type, in declaration order.
    ///
    /// Guarded by [`tests::all_lists_every_variant_exactly_once`], whose
    /// wildcard-free `match` stops compiling the moment a variant is added.
    #[cfg(test)]
    const ALL: [ElemType; 11] = [
        ElemType::F32,
        ElemType::F64,
        ElemType::I8,
        ElemType::I16,
        ElemType::I32,
        ElemType::I64,
        ElemType::U8,
        ElemType::U16,
        ElemType::U32,
        ElemType::U64,
        ElemType::VlenStr,
    ];

    /// On-disk size, in bytes, of a single element.
    ///
    /// For `VlenStr` this is the 16-byte global-heap reference footprint, not
    /// the length of any string.
    pub(crate) const fn byte_size(self) -> usize {
        match self.spec() {
            ElemSpec::Fixed { size, .. } | ElemSpec::FloatIeee { size } => size as usize,
            ElemSpec::VlenStr => VLEN_REF_SIZE,
        }
    }

    /// Body size, in bytes, of this type's datatype message (0x0003).
    pub(crate) const fn dt_body_size(self) -> usize {
        match self.spec() {
            ElemSpec::Fixed { .. } => FIXED_DT_BODY,
            ElemSpec::FloatIeee { .. } => FLOAT_DT_BODY,
            ElemSpec::VlenStr => VLEN_DT_BODY,
        }
    }
}

// ---------------------------------------------------------------------------
// Datatype body encoders
// ---------------------------------------------------------------------------

/// Number of IEEE-754 exponent bits in a binary interchange format of `size`
/// bytes (binary16, binary32, binary64, binary128).
const fn ieee_exp_bits(size: u8) -> Option<u8> {
    match size {
        2 => Some(5),
        4 => Some(8),
        8 => Some(11),
        16 => Some(15),
        _ => None,
    }
}

/// Write a class-0 (fixed-point) datatype body; returns bytes written.
fn write_fixed_dtype(buf: &mut [u8], start: usize, size: u8, signed: bool) -> usize {
    fill_zero(buf, start, FIXED_DT_BODY);
    buf[start] = 0x10; // class 0 (fixed-point), version 1
    buf[start + 1] = u8::from(signed) << 3; // little-endian, no padding, sign flag
                                            // [2..4] remaining class bit fields = 0
    write_u32_le(buf, start + 4, u32::from(size));
    // [8..10] bit offset = 0
    write_u16_le(buf, start + 10, u16::from(size) * 8); // bit precision
                                                        // [12..16] padding to the 8-byte message boundary
    FIXED_DT_BODY
}

/// Write a class-1 (IEEE-754 float) datatype body; returns bytes written.
///
/// Everything follows from `size`: precision is `size * 8`, the sign bit sits at
/// `precision - 1`, the exponent field of [`ieee_exp_bits`] bits sits directly
/// above a mantissa of `precision - exp_bits - 1` bits based at 0, and the bias
/// is `2^(exp_bits - 1) - 1`.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `size` is not an IEEE-754 interchange width.
fn write_float_dtype(buf: &mut [u8], start: usize, size: u8) -> Result<usize, OxiH5Error> {
    let exp_bits = ieee_exp_bits(size).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "internal writer error: no IEEE-754 layout for a {size}-byte float"
        ))
    })?;
    // `size` is one of 2/4/8/16 here, so `size * 8` cannot overflow a u8.
    let precision = size * 8;
    let mant_bits = precision - exp_bits - 1;

    fill_zero(buf, start, FLOAT_DT_BODY);
    buf[start] = 0x11; // class 1 (floating-point), version 1
    buf[start + 1] = 0x20; // little-endian; mantissa normalisation 1 (implied MSB)
    buf[start + 2] = precision - 1; // sign bit location
                                    // [3] remaining class bit field = 0
    write_u32_le(buf, start + 4, u32::from(size));
    // [8..10] bit offset = 0
    write_u16_le(buf, start + 10, u16::from(precision)); // bit precision
    buf[start + 12] = mant_bits; // exponent location
    buf[start + 13] = exp_bits; // exponent size, in bits
    buf[start + 14] = 0; // mantissa location
    buf[start + 15] = mant_bits; // mantissa size, in bits
    write_u32_le(buf, start + 16, (1u32 << (exp_bits - 1)) - 1); // exponent bias
                                                                 // [20..24] padding to the 8-byte message boundary
    Ok(FLOAT_DT_BODY)
}

/// Write a class-3 (fixed-length string) datatype body; returns bytes written.
fn write_string_dtype(buf: &mut [u8], start: usize, len: u32) -> usize {
    fill_zero(buf, start, STRING_DT_BODY);
    buf[start] = 0x13; // class 3 (string), version 1
    buf[start + 1] = 0x10; // null-padded (bits 0..4), UTF-8 charset (bits 4..8)
    write_u32_le(buf, start + 4, len);
    STRING_DT_BODY
}

/// Write a class-7 (object reference) datatype body; returns bytes written.
fn write_ref_dtype(buf: &mut [u8], start: usize) -> usize {
    fill_zero(buf, start, REF_DT_BODY);
    buf[start] = 0x17; // class 7 (reference), version 1
    write_u32_le(buf, start + 4, 8); // one object reference is one 8-byte address
    REF_DT_BODY
}

/// Write a class-9 (variable-length string) datatype body; returns bytes written.
///
/// The outer type declares the size of the on-disk reference; the nested base
/// type is a single UTF-8 character.
fn write_vlen_str_dtype(buf: &mut [u8], start: usize) -> usize {
    fill_zero(buf, start, VLEN_DT_BODY);
    buf[start] = 0x19; // class 9 (vlen), version 1
    buf[start + 1] = 0x01; // vlen type 1 (string), null-terminated padding
    write_u32_le(buf, start + 4, VLEN_REF_SIZE as u32);
    write_string_dtype(buf, start + STRING_DT_BODY, 1);
    VLEN_DT_BODY
}

/// Write the datatype message body for `elem_type`; returns bytes written.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the element type has no encodable layout.
pub(super) fn write_datatype_body(
    buf: &mut [u8],
    start: usize,
    elem_type: ElemType,
) -> Result<usize, OxiH5Error> {
    match elem_type.spec() {
        ElemSpec::Fixed { size, signed } => Ok(write_fixed_dtype(buf, start, size, signed)),
        ElemSpec::FloatIeee { size } => write_float_dtype(buf, start, size),
        ElemSpec::VlenStr => Ok(write_vlen_str_dtype(buf, start)),
    }
}

// ---------------------------------------------------------------------------
// Dtype → ElemType
// ---------------------------------------------------------------------------

/// Map a public [`Dtype`] onto a writable [`ElemType`].
///
/// This is deliberately *not* folded into [`ElemType::spec`]: it narrows an
/// open, reader-side type description down to the closed set the writer can
/// emit, which is a different question from how that set is encoded.
///
/// The `order` field is load-bearing.  Every element type here is little-endian
/// and every `write_dataset_*` helper serialises with `to_le_bytes`; there is no
/// byte-swap path in the writer.  Accepting `ByteOrder::Big` would therefore
/// emit little-endian payload bytes underneath a datatype message the caller
/// asked to be big-endian — a file that reads back wrong rather than a file that
/// fails to be written.  So big-endian is rejected here, at the one place a
/// caller-supplied `Dtype` enters the writer.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `dtype` is big-endian, or for any dtype the
/// writer cannot emit.
pub(super) fn dtype_to_elem_type(dtype: &Dtype) -> Result<ElemType, OxiH5Error> {
    if matches!(
        dtype,
        Dtype::Int {
            order: ByteOrder::Big,
            ..
        } | Dtype::Float {
            order: ByteOrder::Big,
            ..
        }
    ) {
        return Err(OxiH5Error::Format(format!(
            "unsupported dtype {dtype:?}: the writer emits little-endian only, \
             big-endian is not supported"
        )));
    }
    match dtype {
        Dtype::Float { size: 4, .. } => Ok(ElemType::F32),
        Dtype::Float { size: 8, .. } => Ok(ElemType::F64),
        Dtype::Int {
            size: 1,
            signed: true,
            ..
        } => Ok(ElemType::I8),
        Dtype::Int {
            size: 2,
            signed: true,
            ..
        } => Ok(ElemType::I16),
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
        Dtype::Int {
            size: 2,
            signed: false,
            ..
        } => Ok(ElemType::U16),
        Dtype::Int {
            size: 4,
            signed: false,
            ..
        } => Ok(ElemType::U32),
        Dtype::Int {
            size: 8,
            signed: false,
            ..
        } => Ok(ElemType::U64),
        _ => Err(OxiH5Error::Format(format!("unsupported dtype {dtype:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Attribute kinds
// ---------------------------------------------------------------------------

/// A user-supplied attribute value, before object references are resolved.
pub(crate) enum AttrKind {
    /// Scalar fixed-length string.
    FixedStr(String),
    /// Scalar float64.
    F64(f64),
    /// Scalar signed int64.
    I64(i64),
    /// Scalar signed int32.
    I32(i32),
    /// A vector of float64.
    F64Array(Vec<f64>),
    /// A vector of signed int64.
    I64Array(Vec<i64>),
    /// A vector of fixed-length strings, all stored at one common width.
    StrArray(Vec<String>),
    /// A vector of object references, given by target object name.
    ObjRefsByName(Vec<String>),
}

/// A named attribute attached to an object.
pub(crate) struct AttrDesc {
    /// Attribute name.
    pub(crate) name: String,
    /// Attribute value.
    pub(crate) kind: AttrKind,
}

/// An attribute value in write-ready form.
pub(super) enum ResolvedAttrKind<'a> {
    /// Scalar fixed-length string.
    FixedStr(&'a str),
    /// Scalar float64.
    F64(f64),
    /// Scalar signed int64.
    I64(i64),
    /// Scalar signed int32.
    I32(i32),
    /// A vector of float64.
    F64Array(&'a [f64]),
    /// A vector of signed int64.
    I64Array(&'a [i64]),
    /// A vector of fixed-length strings.
    ///
    /// HDF5 has one datatype per attribute, so every element occupies the same
    /// `width` bytes, NUL-padded — which is why the width is resolved here,
    /// once, rather than recomputed at each of the three sites that need it.
    StrArray {
        /// The strings, in order.
        values: &'a [String],
        /// Common on-disk element width, in bytes.
        width: usize,
    },
    /// A vector of object references.
    ///
    /// `addrs` holds exactly one entry per name and is filled in by
    /// [`fill_obj_refs`] once object-header addresses are known.  Its *length*
    /// is fixed at resolution time, which is what makes it safe to size an
    /// object header before a single address exists.
    ObjRefs {
        /// Target object names, in order.
        names: &'a [String],
        /// Target object-header addresses, initially [`UNDEFINED_ADDR`].
        addrs: Vec<u64>,
    },
}

/// Byte sizes of the three variable sections of an attribute v1 message.
#[derive(Debug, Clone, Copy)]
pub(super) struct AttrSizes {
    /// Unpadded datatype message body size.
    pub(super) dtype: usize,
    /// Unpadded dataspace message body size.
    pub(super) dspace: usize,
    /// Raw attribute data size.
    pub(super) data: usize,
}

impl ResolvedAttrKind<'_> {
    /// Element count, or `None` for a scalar dataspace.
    ///
    /// The dataspace *shape* of an attribute follows from this one question, so
    /// asking it once is what keeps [`Self::sizes`] and
    /// [`ResolvedAttr::write_dspace_body`] from disagreeing about whether a
    /// given kind is a scalar — a disagreement that reserves 8 bytes and writes
    /// 24, straight through the middle of the next message.
    fn vector_len(&self) -> Option<usize> {
        match self {
            ResolvedAttrKind::FixedStr(_)
            | ResolvedAttrKind::F64(_)
            | ResolvedAttrKind::I64(_)
            | ResolvedAttrKind::I32(_) => None,
            ResolvedAttrKind::F64Array(values) => Some(values.len()),
            ResolvedAttrKind::I64Array(values) => Some(values.len()),
            ResolvedAttrKind::StrArray { values, .. } => Some(values.len()),
            ResolvedAttrKind::ObjRefs { addrs, .. } => Some(addrs.len()),
        }
    }

    /// The one and only per-attribute-kind size table.
    pub(super) fn sizes(&self) -> AttrSizes {
        let dspace = match self.vector_len() {
            None => SCALAR_DSPACE_BODY,
            Some(_) => VECTOR_DSPACE_BODY,
        };
        let (dtype, data) = match self {
            ResolvedAttrKind::FixedStr(s) => (STRING_DT_BODY, s.len()),
            ResolvedAttrKind::F64(_) => (FLOAT_DT_BODY, 8),
            ResolvedAttrKind::I64(_) => (FIXED_DT_BODY, 8),
            ResolvedAttrKind::I32(_) => (FIXED_DT_BODY, 4),
            ResolvedAttrKind::F64Array(values) => (FLOAT_DT_BODY, values.len() * 8),
            ResolvedAttrKind::I64Array(values) => (FIXED_DT_BODY, values.len() * 8),
            ResolvedAttrKind::StrArray { values, width } => (STRING_DT_BODY, values.len() * width),
            ResolvedAttrKind::ObjRefs { addrs, .. } => (REF_DT_BODY, addrs.len() * 8),
        };
        AttrSizes {
            dtype,
            dspace,
            data,
        }
    }
}

/// A named attribute in write-ready form.
pub(super) struct ResolvedAttr<'a> {
    /// Attribute name.
    pub(super) name: &'a str,
    /// Attribute value.
    pub(super) kind: ResolvedAttrKind<'a>,
}

impl ResolvedAttr<'_> {
    /// Unpadded body size of this attribute's v1 message (0x000C).
    pub(super) fn body_size(&self) -> usize {
        let sizes = self.kind.sizes();
        ATTR_BODY_PREFIX
            + pad8(self.name.len() + 1)
            + pad8(sizes.dtype)
            + pad8(sizes.dspace)
            + sizes.data
    }

    /// Write this attribute's v1 message body at `start`; returns bytes written.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a field overflows its on-disk width, or
    /// if a section writes a different number of bytes than [`Self::body_size`]
    /// accounted for.
    pub(super) fn write_body(&self, buf: &mut [u8], start: usize) -> Result<usize, OxiH5Error> {
        let sizes = self.kind.sizes();
        let name_size = self.name.len() + 1; // includes the NUL terminator

        fill_zero(buf, start, ATTR_BODY_PREFIX);
        buf[start] = 0x01; // attribute message version 1
                           // [1] reserved
        write_u16_le(buf, start + 2, narrow("attribute name size", name_size)?);
        write_u16_le(
            buf,
            start + 4,
            narrow("attribute datatype size", sizes.dtype)?,
        );
        write_u16_le(
            buf,
            start + 6,
            narrow("attribute dataspace size", sizes.dspace)?,
        );

        let mut pos = start + ATTR_BODY_PREFIX;

        let name_padded = pad8(name_size);
        fill_zero(buf, pos, name_padded);
        buf[pos..pos + self.name.len()].copy_from_slice(self.name.as_bytes());
        pos += name_padded;

        let dtype_padded = pad8(sizes.dtype);
        fill_zero(buf, pos, dtype_padded);
        check_size(
            "attribute datatype",
            self.write_dtype_body(buf, pos)?,
            sizes.dtype,
        )?;
        pos += dtype_padded;

        let dspace_padded = pad8(sizes.dspace);
        fill_zero(buf, pos, dspace_padded);
        check_size(
            "attribute dataspace",
            self.write_dspace_body(buf, pos),
            sizes.dspace,
        )?;
        pos += dspace_padded;

        fill_zero(buf, pos, sizes.data);
        check_size("attribute data", self.write_data(buf, pos), sizes.data)?;
        pos += sizes.data;

        Ok(pos - start)
    }

    /// Write the attribute's inline datatype message body; returns bytes written.
    ///
    /// An array attribute and its scalar counterpart share a datatype: HDF5
    /// puts the element count in the dataspace, never in the type.
    fn write_dtype_body(&self, buf: &mut [u8], start: usize) -> Result<usize, OxiH5Error> {
        match &self.kind {
            ResolvedAttrKind::FixedStr(s) => Ok(write_string_dtype(
                buf,
                start,
                narrow("attribute string length", s.len())?,
            )),
            ResolvedAttrKind::StrArray { width, .. } => Ok(write_string_dtype(
                buf,
                start,
                narrow("attribute string width", *width)?,
            )),
            ResolvedAttrKind::F64(_) | ResolvedAttrKind::F64Array(_) => {
                write_float_dtype(buf, start, 8)
            }
            ResolvedAttrKind::I64(_) | ResolvedAttrKind::I64Array(_) => {
                Ok(write_fixed_dtype(buf, start, 8, true))
            }
            ResolvedAttrKind::I32(_) => Ok(write_fixed_dtype(buf, start, 4, true)),
            ResolvedAttrKind::ObjRefs { .. } => Ok(write_ref_dtype(buf, start)),
        }
    }

    /// Write the attribute's inline dataspace message body; returns bytes written.
    fn write_dspace_body(&self, buf: &mut [u8], start: usize) -> usize {
        match self.kind.vector_len() {
            Some(n) => {
                fill_zero(buf, start, VECTOR_DSPACE_BODY);
                buf[start] = 0x01; // version = 1
                buf[start + 1] = 0x01; // dimensionality = 1
                buf[start + 2] = 0x01; // flags = max dims present
                write_u64_le(buf, start + 8, n as u64);
                write_u64_le(buf, start + 16, n as u64);
                VECTOR_DSPACE_BODY
            }
            None => {
                fill_zero(buf, start, SCALAR_DSPACE_BODY);
                buf[start] = 0x01; // version = 1, dimensionality = 0 (scalar)
                SCALAR_DSPACE_BODY
            }
        }
    }

    /// Write the attribute's raw data; returns bytes written.
    ///
    /// The caller has already zeroed the region, which is what lets the
    /// fixed-length string arms write only the characters and leave the
    /// NUL padding implicit.
    fn write_data(&self, buf: &mut [u8], start: usize) -> usize {
        match &self.kind {
            ResolvedAttrKind::FixedStr(s) => {
                buf[start..start + s.len()].copy_from_slice(s.as_bytes());
                s.len()
            }
            ResolvedAttrKind::F64(v) => {
                buf[start..start + 8].copy_from_slice(&v.to_le_bytes());
                8
            }
            ResolvedAttrKind::I64(v) => {
                buf[start..start + 8].copy_from_slice(&v.to_le_bytes());
                8
            }
            ResolvedAttrKind::I32(v) => {
                buf[start..start + 4].copy_from_slice(&v.to_le_bytes());
                4
            }
            ResolvedAttrKind::F64Array(values) => {
                for (i, v) in values.iter().enumerate() {
                    buf[start + i * 8..start + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
                values.len() * 8
            }
            ResolvedAttrKind::I64Array(values) => {
                for (i, v) in values.iter().enumerate() {
                    buf[start + i * 8..start + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
                values.len() * 8
            }
            ResolvedAttrKind::StrArray { values, width } => {
                for (i, s) in values.iter().enumerate() {
                    let slot = start + i * width;
                    buf[slot..slot + s.len()].copy_from_slice(s.as_bytes());
                }
                values.len() * width
            }
            ResolvedAttrKind::ObjRefs { addrs, .. } => {
                for (i, &addr) in addrs.iter().enumerate() {
                    write_u64_le(buf, start + i * 8, addr);
                }
                addrs.len() * 8
            }
        }
    }
}

/// Common on-disk width of a fixed-length string array, in bytes.
///
/// HDF5 stores one datatype per attribute, so every element has to fit the
/// longest.  Zero is not a legal string size, so an array of nothing but empty
/// strings still occupies one byte per element.
fn str_array_width(values: &[String]) -> usize {
    values.iter().map(String::len).max().unwrap_or(0).max(1)
}

/// Turn attribute descriptors into write-ready attributes.
///
/// This is the *only* raw → resolved conversion in the writer.  It used to have
/// a second, string-only copy for the root group, which meant the root could
/// carry nothing but strings and the two copies could drift; folding the root
/// group into the ordinary group tree removed the need for it.
///
/// Object-reference addresses start out undefined and are filled in later by
/// [`fill_obj_refs`].  Resolving *before* sizing is the invariant that makes
/// reserved-but-unwritten object-header space unrepresentable: there is no way
/// to ask for the size of an attribute the writer will not go on to emit.
pub(super) fn resolve_attrs(attrs: &[AttrDesc]) -> Vec<ResolvedAttr<'_>> {
    attrs
        .iter()
        .map(|attr| ResolvedAttr {
            name: &attr.name,
            kind: match &attr.kind {
                AttrKind::FixedStr(s) => ResolvedAttrKind::FixedStr(s.as_str()),
                AttrKind::F64(v) => ResolvedAttrKind::F64(*v),
                AttrKind::I64(v) => ResolvedAttrKind::I64(*v),
                AttrKind::I32(v) => ResolvedAttrKind::I32(*v),
                AttrKind::F64Array(values) => ResolvedAttrKind::F64Array(values),
                AttrKind::I64Array(values) => ResolvedAttrKind::I64Array(values),
                AttrKind::StrArray(values) => ResolvedAttrKind::StrArray {
                    values,
                    width: str_array_width(values),
                },
                AttrKind::ObjRefsByName(names) => ResolvedAttrKind::ObjRefs {
                    names,
                    addrs: vec![UNDEFINED_ADDR; names.len()],
                },
            },
        })
        .collect()
}

/// Fill in object-reference addresses from a path → object-header address map.
///
/// Length-preserving by construction: each name yields exactly one address, so
/// no attribute message can change size after its object header has been sized.
///
/// A target that names nothing is an **error**.  It used to fall back to the
/// undefined-address sentinel, which produced a structurally valid file whose
/// `DIMENSION_LIST` pointed at `0xffff_ffff_ffff_ffff` — a dangling reference
/// that only surfaces when something tries to dereference it, arbitrarily far
/// from the typo that caused it.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` naming the attribute and the target if a
/// reference cannot be resolved.
pub(super) fn fill_obj_refs(
    attrs: &mut [ResolvedAttr<'_>],
    path_to_addr: &HashMap<String, u64>,
) -> Result<(), OxiH5Error> {
    for attr in attrs {
        if let ResolvedAttrKind::ObjRefs { names, addrs } = &mut attr.kind {
            for (slot, name) in addrs.iter_mut().zip(names.iter()) {
                // Targets may be written with or without a leading separator;
                // the map is keyed by path from the root without one.
                let key = name.strip_prefix('/').unwrap_or(name);
                *slot = path_to_addr.get(key).copied().ok_or_else(|| {
                    OxiH5Error::Format(format!(
                        "attribute '{}': object reference target '{name}' names no dataset \
                         or group in the file",
                        attr.name
                    ))
                })?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a datatype body into a fresh, zeroed buffer.
    fn body_of(elem_type: ElemType) -> Vec<u8> {
        let size = elem_type.dt_body_size();
        let mut buf = vec![0u8; size];
        let wrote = write_datatype_body(&mut buf, 0, elem_type).expect("write_datatype_body");
        assert_eq!(wrote, size, "{elem_type:?} wrote the wrong length");
        buf
    }

    /// The proof that the parametric encoders are behaviour-preserving: the
    /// bytes they generate are identical to the hand-written literals that used
    /// to be scattered across `messages.rs`.
    #[test]
    fn generated_dtype_bodies_match_the_historic_literals() {
        let f32_literal: [u8; 24] = [
            0x11, 0x20, 0x1f, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x17, 0x08,
            0x00, 0x17, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let f64_literal: [u8; 24] = [
            0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b,
            0x00, 0x34, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let i32_literal: [u8; 16] = [
            0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let i64_literal: [u8; 16] = [
            0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let u8_literal: [u8; 16] = [
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let vlen_literal: [u8; 16] = [
            0x19, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x13, 0x10, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00,
        ];

        assert_eq!(body_of(ElemType::F32), f32_literal, "F32");
        assert_eq!(body_of(ElemType::F64), f64_literal, "F64");
        assert_eq!(body_of(ElemType::I32), i32_literal, "I32");
        assert_eq!(body_of(ElemType::I64), i64_literal, "I64");
        assert_eq!(body_of(ElemType::U8), u8_literal, "U8");
        assert_eq!(body_of(ElemType::VlenStr), vlen_literal, "VlenStr");
    }

    /// `ALL` must list every variant, exactly once.
    ///
    /// The wildcard-free `match` is the guard: adding an [`ElemType`] variant
    /// makes this arm list non-exhaustive, so the compiler drags the author to
    /// the one place that also names the table they need to extend.
    #[test]
    fn all_lists_every_variant_exactly_once() {
        for elem_type in ElemType::ALL {
            match elem_type {
                ElemType::F32
                | ElemType::F64
                | ElemType::I8
                | ElemType::I16
                | ElemType::I32
                | ElemType::I64
                | ElemType::U8
                | ElemType::U16
                | ElemType::U32
                | ElemType::U64
                | ElemType::VlenStr => {}
            }
        }
        for (i, a) in ElemType::ALL.iter().enumerate() {
            for b in &ElemType::ALL[i + 1..] {
                assert_ne!(a, b, "{a:?} appears twice in ElemType::ALL");
            }
        }
    }

    /// The five newly writable integer types must set the three fields the
    /// reader actually looks at for class 0.
    ///
    /// Cross-checked against `oxih5_format::datatype`'s `0 =>` arm: the low
    /// nibble of byte 0 is the class and the high nibble the version, bit 0 of
    /// byte 1 is the byte order and bit 3 the sign flag, `[4..8]` is the element
    /// size, and (version 1 only) `[8..10]`/`[10..12]` are bit offset and bit
    /// precision.
    #[test]
    fn new_int_dtype_bodies_carry_class_size_and_precision() {
        let cases: [(ElemType, u16, bool); 5] = [
            (ElemType::I8, 1, true),
            (ElemType::I16, 2, true),
            (ElemType::U16, 2, false),
            (ElemType::U32, 4, false),
            (ElemType::U64, 8, false),
        ];
        for (elem_type, size, signed) in cases {
            let body = body_of(elem_type);
            assert_eq!(body.len(), FIXED_DT_BODY, "{elem_type:?}: body size");
            assert_eq!(body[0] & 0x0F, 0, "{elem_type:?}: class must be 0");
            assert_eq!(body[0] >> 4, 1, "{elem_type:?}: version must be 1");
            assert_eq!(body[1] & 0x01, 0, "{elem_type:?}: must be little-endian");
            assert_eq!(body[1] & 0x08 != 0, signed, "{elem_type:?}: sign flag");
            assert_eq!(
                u32::from_le_bytes([body[4], body[5], body[6], body[7]]),
                u32::from(size),
                "{elem_type:?}: element size"
            );
            assert_eq!(
                u16::from_le_bytes([body[8], body[9]]),
                0,
                "{elem_type:?}: bit offset"
            );
            assert_eq!(
                u16::from_le_bytes([body[10], body[11]]),
                size * 8,
                "{elem_type:?}: bit precision"
            );
        }
    }

    /// The encoders must be the exact inverse of the reader's parser.
    ///
    /// `write_datatype_body` and `oxih5_format::datatype::parse_datatype` are
    /// two independently hand-written codecs for the same on-disk layout, and
    /// nothing but a test makes them agree.  This closes the loop
    /// `ElemType -> bytes -> Dtype -> ElemType`, so a wrong class nibble, sign
    /// bit, or size field cannot survive in either direction.
    #[test]
    fn every_dtype_body_parses_back_through_the_reader() {
        let le = ByteOrder::Little;
        let int = |size, signed| Dtype::Int {
            size,
            signed,
            order: le,
        };
        let expected: [(ElemType, Dtype); 11] = [
            (ElemType::F32, Dtype::Float { size: 4, order: le }),
            (ElemType::F64, Dtype::Float { size: 8, order: le }),
            (ElemType::I8, int(1, true)),
            (ElemType::I16, int(2, true)),
            (ElemType::I32, int(4, true)),
            (ElemType::I64, int(8, true)),
            (ElemType::U8, int(1, false)),
            (ElemType::U16, int(2, false)),
            (ElemType::U32, int(4, false)),
            (ElemType::U64, int(8, false)),
            (
                ElemType::VlenStr,
                Dtype::String {
                    fixed_len: None,
                    charset: oxih5_core::Charset::Utf8,
                },
            ),
        ];
        assert_eq!(
            expected.len(),
            ElemType::ALL.len(),
            "every ElemType must be covered here"
        );

        for (elem_type, want) in expected {
            let body = body_of(elem_type);
            let got = oxih5_format::datatype::parse_datatype(&body).unwrap_or_else(|e| {
                panic!("{elem_type:?}: the reader rejected the writer's own bytes: {e}")
            });
            assert_eq!(got, want, "{elem_type:?}");

            // VlenStr datasets are created through `create_vlen_string_dataset`,
            // never through a `Dtype`, so it is deliberately outside the
            // `dtype_to_elem_type` domain.
            if elem_type != ElemType::VlenStr {
                assert_eq!(
                    dtype_to_elem_type(&got).expect("dtype_to_elem_type"),
                    elem_type,
                    "{elem_type:?}: round trip"
                );
            }
        }
    }

    /// Attribute datatypes historically carried their own copies of the same
    /// literals; they must now come out of the same encoders.
    #[test]
    fn attr_dtype_bodies_match_the_historic_literals() {
        let cases: [(ResolvedAttrKind<'_>, &[u8]); 4] = [
            (
                ResolvedAttrKind::F64(0.0),
                &[
                    0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34,
                    0x0b, 0x00, 0x34, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                ],
            ),
            (
                ResolvedAttrKind::I64(0),
                &[
                    0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ],
            ),
            (
                ResolvedAttrKind::I32(0),
                &[
                    0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ],
            ),
            (
                ResolvedAttrKind::FixedStr("abc"),
                &[0x13, 0x10, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00],
            ),
        ];
        for (kind, expected) in cases {
            let attr = ResolvedAttr { name: "a", kind };
            let mut buf = vec![0u8; expected.len()];
            let wrote = attr.write_dtype_body(&mut buf, 0).expect("dtype body");
            assert_eq!(wrote, expected.len());
            assert_eq!(buf, expected);
        }

        // Object references: class 7, version 1, one 8-byte address.
        let attr = ResolvedAttr {
            name: "r",
            kind: ResolvedAttrKind::ObjRefs {
                names: &[],
                addrs: Vec::new(),
            },
        };
        let mut buf = vec![0u8; REF_DT_BODY];
        assert_eq!(attr.write_dtype_body(&mut buf, 0).expect("ref dtype"), 8);
        assert_eq!(buf, [0x17, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn byte_sizes_match_the_element_widths() {
        assert_eq!(ElemType::F32.byte_size(), 4);
        assert_eq!(ElemType::F64.byte_size(), 8);
        assert_eq!(ElemType::I8.byte_size(), 1);
        assert_eq!(ElemType::I16.byte_size(), 2);
        assert_eq!(ElemType::I32.byte_size(), 4);
        assert_eq!(ElemType::I64.byte_size(), 8);
        assert_eq!(ElemType::U8.byte_size(), 1);
        assert_eq!(ElemType::U16.byte_size(), 2);
        assert_eq!(ElemType::U32.byte_size(), 4);
        assert_eq!(ElemType::U64.byte_size(), 8);
        assert_eq!(ElemType::VlenStr.byte_size(), VLEN_REF_SIZE);
    }

    #[test]
    fn dt_body_sizes_match_the_encoders() {
        for elem_type in ElemType::ALL {
            assert_eq!(
                body_of(elem_type).len(),
                elem_type.dt_body_size(),
                "{elem_type:?}"
            );
        }
    }

    #[test]
    fn float_encoder_rejects_non_ieee_widths() {
        let mut buf = vec![0u8; FLOAT_DT_BODY];
        assert!(write_float_dtype(&mut buf, 0, 3).is_err());
        assert!(write_float_dtype(&mut buf, 0, 4).is_ok());
    }

    #[test]
    fn attr_body_size_matches_bytes_written() {
        let names = vec!["lat".to_string(), "lon".to_string()];
        let cases = [
            ResolvedAttrKind::FixedStr("degrees_north"),
            ResolvedAttrKind::F64(2.5),
            ResolvedAttrKind::I64(i64::MIN),
            ResolvedAttrKind::I32(-7),
            ResolvedAttrKind::ObjRefs {
                names: &names,
                addrs: vec![1, 2],
            },
        ];
        for kind in cases {
            let attr = ResolvedAttr {
                name: "an_attribute",
                kind,
            };
            let expected = attr.body_size();
            let mut buf = vec![0u8; expected + 64];
            assert_eq!(attr.write_body(&mut buf, 0).expect("write_body"), expected);
        }
    }

    #[test]
    fn obj_refs_keep_their_length_through_resolution() {
        let descs = vec![AttrDesc {
            name: "DIMENSION_LIST".to_string(),
            kind: AttrKind::ObjRefsByName(vec!["lat".to_string(), "grp/lon".to_string()]),
        }];
        let mut resolved = resolve_attrs(&descs);
        let before = resolved[0].body_size();

        let mut map: HashMap<String, u64> = HashMap::new();
        map.insert("lat".to_string(), 0x1234);
        map.insert("grp/lon".to_string(), 0x5678);
        fill_obj_refs(&mut resolved, &map).expect("both targets exist");

        assert_eq!(resolved[0].body_size(), before, "sizing must be stable");
        match &resolved[0].kind {
            ResolvedAttrKind::ObjRefs { addrs, .. } => {
                assert_eq!(addrs, &[0x1234, 0x5678]);
            }
            _ => panic!("expected ObjRefs"),
        }
    }

    /// An unresolvable target is an error, not `u64::MAX` written to disk.
    ///
    /// The sentinel produced a file that opened fine and carried a dangling
    /// reference, so the mistake surfaced — if at all — at dereference time,
    /// arbitrarily far from the misspelt name that caused it.
    #[test]
    fn an_unresolvable_obj_ref_is_reported_not_silently_undefined() {
        let descs = vec![AttrDesc {
            name: "DIMENSION_LIST".to_string(),
            kind: AttrKind::ObjRefsByName(vec!["lat".to_string(), "typo".to_string()]),
        }];
        let mut resolved = resolve_attrs(&descs);
        let mut map: HashMap<String, u64> = HashMap::new();
        map.insert("lat".to_string(), 0x1234);

        let err = fill_obj_refs(&mut resolved, &map).expect_err("must refuse");
        let msg = format!("{err}");
        assert!(msg.contains("'typo'"), "must name the target: {msg}");
        assert!(msg.contains("DIMENSION_LIST"), "must name the attr: {msg}");
    }

    /// A leading separator on a target is optional, as it is everywhere else.
    #[test]
    fn obj_ref_targets_may_be_written_absolutely() {
        let descs = vec![AttrDesc {
            name: "refs".to_string(),
            kind: AttrKind::ObjRefsByName(vec!["/a/b/x".to_string()]),
        }];
        let mut resolved = resolve_attrs(&descs);
        let mut map: HashMap<String, u64> = HashMap::new();
        map.insert("a/b/x".to_string(), 0x99);
        fill_obj_refs(&mut resolved, &map).expect("absolute target");
        match &resolved[0].kind {
            ResolvedAttrKind::ObjRefs { addrs, .. } => assert_eq!(addrs, &[0x99]),
            _ => panic!("expected ObjRefs"),
        }
    }

    /// Every array kind must reserve exactly what it writes, at every length.
    ///
    /// Arrays are the first attribute kind whose data size is unbounded, so the
    /// scalar-versus-vector dataspace choice and the per-element stride are both
    /// new ways for the reserve and write paths to disagree.
    #[test]
    fn array_attr_body_sizes_match_bytes_written() {
        let strings: Vec<String> = ["a", "", "much longer entry"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let empty: Vec<String> = Vec::new();
        let floats = [1.5f64, -2.5, f64::NAN];
        let ints = [i64::MIN, 0, i64::MAX];

        let cases = [
            ResolvedAttrKind::F64Array(&floats),
            ResolvedAttrKind::F64Array(&[]),
            ResolvedAttrKind::I64Array(&ints),
            ResolvedAttrKind::I64Array(&[]),
            ResolvedAttrKind::StrArray {
                values: &strings,
                width: str_array_width(&strings),
            },
            ResolvedAttrKind::StrArray {
                values: &empty,
                width: str_array_width(&empty),
            },
        ];
        for kind in cases {
            let attr = ResolvedAttr {
                name: "an_array",
                kind,
            };
            let expected = attr.body_size();
            let mut buf = vec![0u8; expected + 64];
            assert_eq!(attr.write_body(&mut buf, 0).expect("write_body"), expected);
        }
    }

    /// Every element of a string array occupies the width of the longest, and
    /// the slack is NUL — which is how the reader recovers the lengths.
    #[test]
    fn string_arrays_are_nul_padded_to_a_common_width() {
        assert_eq!(str_array_width(&["ab".to_string(), "cdef".to_string()]), 4);
        // A zero-byte string type is not representable, so all-empty is 1.
        assert_eq!(str_array_width(&[String::new(), String::new()]), 1);
        assert_eq!(str_array_width(&[]), 1);

        let values: Vec<String> = ["ab", "cdef", ""]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let attr = ResolvedAttr {
            name: "labels",
            kind: ResolvedAttrKind::StrArray {
                values: &values,
                width: 4,
            },
        };
        let total = attr.body_size();
        let mut buf = vec![0xAAu8; total];
        attr.write_body(&mut buf, 0).expect("write_body");
        let data = &buf[total - 12..];
        assert_eq!(data, b"ab\0\0cdef\0\0\0\0");
    }

    /// Scalars get a scalar dataspace and arrays get a 1-D one, whatever the
    /// element type.
    #[test]
    fn vector_len_decides_the_dataspace_shape() {
        let values = [1.0f64];
        let strings = ["x".to_string()];
        assert_eq!(ResolvedAttrKind::F64(1.0).vector_len(), None);
        assert_eq!(ResolvedAttrKind::I64(1).vector_len(), None);
        assert_eq!(ResolvedAttrKind::I32(1).vector_len(), None);
        assert_eq!(ResolvedAttrKind::FixedStr("x").vector_len(), None);
        assert_eq!(ResolvedAttrKind::F64Array(&values).vector_len(), Some(1));
        assert_eq!(ResolvedAttrKind::I64Array(&[7]).vector_len(), Some(1));
        assert_eq!(
            ResolvedAttrKind::StrArray {
                values: &strings,
                width: 1
            }
            .vector_len(),
            Some(1)
        );

        for kind in [
            ResolvedAttrKind::F64(1.0),
            ResolvedAttrKind::F64Array(&values),
        ] {
            let want = match kind.vector_len() {
                None => SCALAR_DSPACE_BODY,
                Some(_) => VECTOR_DSPACE_BODY,
            };
            assert_eq!(kind.sizes().dspace, want);
        }
    }

    /// An array shares its element's datatype: the count lives in the
    /// dataspace, so `f64` and `f64[]` must emit identical datatype bodies.
    #[test]
    fn array_datatypes_match_their_scalar_counterparts() {
        let floats = [0.0f64];
        let ints = [0i64];
        let pairs = [
            (
                ResolvedAttrKind::F64(0.0),
                ResolvedAttrKind::F64Array(&floats),
            ),
            (ResolvedAttrKind::I64(0), ResolvedAttrKind::I64Array(&ints)),
        ];
        for (scalar, array) in pairs {
            let mut a = vec![0u8; 32];
            let mut b = vec![0u8; 32];
            let scalar_attr = ResolvedAttr {
                name: "s",
                kind: scalar,
            };
            let array_attr = ResolvedAttr {
                name: "a",
                kind: array,
            };
            let n = scalar_attr.write_dtype_body(&mut a, 0).expect("scalar");
            let m = array_attr.write_dtype_body(&mut b, 0).expect("array");
            assert_eq!(n, m);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn dtype_mapping_covers_the_writable_set() {
        let le = ByteOrder::Little;
        let int = |size, signed| Dtype::Int {
            size,
            signed,
            order: le,
        };
        let cases: [(Dtype, ElemType); 10] = [
            (Dtype::Float { size: 4, order: le }, ElemType::F32),
            (Dtype::Float { size: 8, order: le }, ElemType::F64),
            (int(1, true), ElemType::I8),
            (int(2, true), ElemType::I16),
            (int(4, true), ElemType::I32),
            (int(8, true), ElemType::I64),
            (int(1, false), ElemType::U8),
            (int(2, false), ElemType::U16),
            (int(4, false), ElemType::U32),
            (int(8, false), ElemType::U64),
        ];
        for (dtype, want) in cases {
            assert_eq!(
                dtype_to_elem_type(&dtype).unwrap_or_else(|e| panic!("{dtype:?}: {e}")),
                want
            );
        }

        // Widths the writer has no encoding for stay rejected.
        assert!(dtype_to_elem_type(&Dtype::Float { size: 2, order: le }).is_err());
        assert!(dtype_to_elem_type(&int(3, true)).is_err());
        assert!(dtype_to_elem_type(&Dtype::Reference {
            ref_type: oxih5_core::RefType::Object
        })
        .is_err());
    }

    /// Big-endian dtypes must fail, not be written out as little-endian.
    ///
    /// Every element type is little-endian and no `write_dataset_*` helper has a
    /// byte-swap path, so accepting `ByteOrder::Big` here would produce a file
    /// whose datatype message and payload disagree — data that reads back wrong
    /// with no error anywhere.  Real big-endian support is a separate change;
    /// until then this is the honest failure.
    #[test]
    fn big_endian_dtypes_are_rejected() {
        let be = ByteOrder::Big;
        let candidates = [
            Dtype::Float { size: 4, order: be },
            Dtype::Float { size: 8, order: be },
            Dtype::Int {
                size: 4,
                signed: true,
                order: be,
            },
            Dtype::Int {
                size: 2,
                signed: false,
                order: be,
            },
            Dtype::Int {
                size: 8,
                signed: false,
                order: be,
            },
        ];
        for dtype in candidates {
            let err = dtype_to_elem_type(&dtype)
                .expect_err("big-endian must not be silently written as little-endian");
            let msg = err.to_string();
            assert!(
                msg.contains("big-endian"),
                "{dtype:?}: error should name big-endian, got {msg:?}"
            );
        }

        // The little-endian twin of each rejected dtype is still accepted, so
        // the guard rejects the byte order and not the type.
        let le = ByteOrder::Little;
        assert!(dtype_to_elem_type(&Dtype::Float { size: 4, order: le }).is_ok());
        assert!(dtype_to_elem_type(&Dtype::Int {
            size: 2,
            signed: false,
            order: le
        })
        .is_ok());
    }
}
