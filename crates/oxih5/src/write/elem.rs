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
use oxih5_format::{GlobalHeapWriter, HeapObjectLocation};

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
/// Datatype body size for a class-9 vlen sequence of class-7 object references
/// (`H5T_VLEN{ H5T_REFERENCE(object) }`): 8 outer + 8 base type.
const VLEN_REF_DT_BODY: usize = STRING_DT_BODY + REF_DT_BODY;
/// Datatype body size of the `REFERENCE_LIST` compound (class 6, version 1):
/// `{ dataset: objref @0, dimension: uint32 @8 }`, declared struct size 12.
const REF_INDEX_DT_BODY: usize = 116;
/// On-disk footprint of one `REFERENCE_LIST` compound element: an 8-byte object
/// reference followed by a 4-byte unsigned dimension index (the declared struct
/// size, no trailing padding).
const REF_INDEX_ELEM: usize = 12;
/// Datatype body size of the boolean enumeration (class 8, version 1):
/// `8` header + `12` inline `i8` base type + `2 × 8` member names +
/// `2 × 1` member values.  Not a multiple of 8 — the object-header message
/// padding aligns it, exactly as libhdf5 leaves it.
const BOOL_ENUM_DT_BODY: usize = 38;

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
    /// HDF5 fixed-length string (class 3) of the given byte width.
    ///
    /// Each element occupies exactly `width` bytes, NUL-padded, under the
    /// **ASCII** charset — the on-disk shape h5py reads back as numpy
    /// `S<width>` (raw bytes).  The width is the on-disk field, so it is carried
    /// as the `u32` the datatype message stores.
    FixedStr(u32),
    /// HDF5 boolean, stored as libhdf5/h5py do it: a class-8 enumeration
    /// `{ FALSE = 0, TRUE = 1 }` over a signed 8-bit base type.
    ///
    /// Each element is one `i8` (`0` or `1`); h5py reads the dataset back as a
    /// numpy `bool` array.
    Bool,
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
    /// HDF5 class 3 — fixed-length string of `width` bytes, ASCII, NUL-padded.
    FixedStr {
        /// On-disk width in bytes; the datatype message's element size.
        width: u32,
    },
    /// HDF5 class 8 — boolean enumeration `{ FALSE = 0, TRUE = 1 }` over `i8`.
    Bool,
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
            ElemType::FixedStr(width) => ElemSpec::FixedStr { width },
            ElemType::Bool => ElemSpec::Bool,
        }
    }

    /// Every writable element type, in declaration order.
    ///
    /// Guarded by [`tests::all_lists_every_variant_exactly_once`], whose
    /// wildcard-free `match` stops compiling the moment a variant is added.
    #[cfg(test)]
    const ALL: [ElemType; 13] = [
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
        // A representative width stands in for the whole `FixedStr` family; the
        // encoder derives everything from it.
        ElemType::FixedStr(4),
        ElemType::Bool,
    ];

    /// On-disk size, in bytes, of a single element.
    ///
    /// For `VlenStr` this is the 16-byte global-heap reference footprint, not
    /// the length of any string.
    pub(crate) const fn byte_size(self) -> usize {
        match self.spec() {
            ElemSpec::Fixed { size, .. } | ElemSpec::FloatIeee { size } => size as usize,
            ElemSpec::VlenStr => VLEN_REF_SIZE,
            ElemSpec::FixedStr { width } => width as usize,
            // The enum's base type is one signed byte.
            ElemSpec::Bool => 1,
        }
    }

    /// Body size, in bytes, of this type's datatype message (0x0003).
    pub(crate) const fn dt_body_size(self) -> usize {
        match self.spec() {
            ElemSpec::Fixed { .. } => FIXED_DT_BODY,
            ElemSpec::FloatIeee { .. } => FLOAT_DT_BODY,
            ElemSpec::VlenStr => VLEN_DT_BODY,
            ElemSpec::FixedStr { .. } => STRING_DT_BODY,
            ElemSpec::Bool => BOOL_ENUM_DT_BODY,
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
    buf[start + 1] = 0x20; // little-endian; mantissa normalization 2 (implied MSB, not stored)
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

/// String padding convention — the low nibble (bits 0..4) of a class-3 string
/// datatype's bit field.
///
/// The reader in `oxih5_format::datatype` recovers a string's length by trimming
/// at the first NUL and never inspects this nibble, so it is purely an
/// interoperability signal for third-party readers such as libhdf5.
#[derive(Debug, Clone, Copy)]
enum StrPad {
    /// `H5T_STR_NULLTERM` (0): the value is terminated by the first NUL, which
    /// requires a spare byte beyond the content.
    NullTerm,
    /// `H5T_STR_NULLPAD` (1): the value occupies the full declared width and any
    /// unused trailing bytes are NUL.  This is what libhdf5 emits for a
    /// fixed-length string whose declared size equals its content length —
    /// where `NULLTERM` would promise a terminator there is no room for.
    NullPad,
}

impl StrPad {
    /// This convention as it sits in bits 0..4 of a class-3 bit field.
    const fn nibble(self) -> u8 {
        match self {
            StrPad::NullTerm => 0x00,
            StrPad::NullPad => 0x01,
        }
    }
}

/// Character-set convention — the charset nibble (bits 4..8) of a class-3
/// string datatype's bit field.
///
/// The writer's own payloads are always UTF-8 bytes, so the *attribute* string
/// types (and the vlen base character) declare UTF-8, matching every string the
/// writer emitted before this parameter existed.  A fixed-length string
/// *dataset* declares ASCII instead, because that is the nibble libhdf5 sets for
/// a numpy `S<width>` type: it is what makes h5py read the dataset back as
/// `S<width>` (raw bytes) exactly.  The payload bytes are identical either way —
/// the nibble only labels how a third-party reader should interpret them, and a
/// multi-byte UTF-8 string that fits the width round-trips byte-for-byte
/// regardless.
#[derive(Debug, Clone, Copy)]
enum StrCharset {
    /// `H5T_CSET_ASCII` (0): libhdf5's charset for a numpy `S<width>` type.
    Ascii,
    /// `H5T_CSET_UTF8` (1): what the writer declares for its own UTF-8 payloads.
    Utf8,
}

impl StrCharset {
    /// This charset as it sits in bits 4..8 of a class-3 bit field.
    const fn nibble(self) -> u8 {
        match self {
            StrCharset::Ascii => 0x00,
            StrCharset::Utf8 => 0x10,
        }
    }
}

/// Write a class-3 (fixed-length string) datatype body; returns bytes written.
///
/// `pad` selects the padding convention in bits 0..4 and `charset` the charset
/// in bits 4..8.
fn write_string_dtype(
    buf: &mut [u8],
    start: usize,
    len: u32,
    pad: StrPad,
    charset: StrCharset,
) -> usize {
    fill_zero(buf, start, STRING_DT_BODY);
    buf[start] = 0x13; // class 3 (string), version 1
    buf[start + 1] = charset.nibble() | pad.nibble(); // charset (bits 4..8), padding (bits 0..4)
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
    buf[start + 1] = 0x01; // vlen type 1 (string), null-terminate padding (bits 4..8)
    buf[start + 2] = 0x01; // character set 1 = UTF-8 (bits 8..12); str payloads are UTF-8
    write_u32_le(buf, start + 4, VLEN_REF_SIZE as u32);
    // Nested base type: a single UTF-8 character.  libhdf5 tolerates this
    // class-3 base type on read, and a vlen string's length comes from the heap
    // object rather than this nibble, so it stays NULLTERM to keep the base-type
    // bytes stable.
    write_string_dtype(
        buf,
        start + STRING_DT_BODY,
        1,
        StrPad::NullTerm,
        StrCharset::Utf8,
    );
    VLEN_DT_BODY
}

/// The on-disk datatype-message body of a vlen-of-object-reference type,
/// `H5T_VLEN{ H5T_REFERENCE(object) }` — a class-9 vlen *sequence* whose base
/// type is a class-7, 8-byte object reference.
///
/// Byte-pinned against the `DIMENSION_LIST` attribute netCDF-4 writes on a
/// dimensioned variable.  A plain reference *array* — what [`write_ref_dtype`]
/// emits — makes netCDF-C's `H5DSiterate_scales` read each 8-byte reference as
/// an `hvl_t {len, ptr}` and dereference a NULL pointer; this vlen form is the
/// layout that reader requires.  h5py surfaces it as an array of
/// dereferenceable object-reference arrays.
const VLEN_REF_DT_BODY_BYTES: [u8; VLEN_REF_DT_BODY] = [
    0x19, 0x00, 0x00, 0x00, // class 9 (vlen) v1; low nibble of byte 1 = 0 = sequence
    0x10, 0x00, 0x00, 0x00, // on-disk element size = 16 (the hvl_t disk footprint)
    0x17, 0x00, 0x00, 0x00, // base type: class 7 (object reference) v1
    0x08, 0x00, 0x00, 0x00, //   one 8-byte object-header address
];

/// The on-disk datatype-message body of the `REFERENCE_LIST` compound,
/// `{ dataset: H5T_STD_REF_OBJ @0, dimension: uint32 @8 }` with struct size 12.
///
/// Byte-pinned against netCDF-4's `REFERENCE_LIST` (member names, offsets, and
/// base types), verified against `h5py`'s `H5Tencode`.  This is the datatype
/// message *version 1* form (the verbose per-member layout), rather than the
/// more compact version-3 libhdf5 emits, because oxih5's own compound reader
/// decodes the version-1 member layout — so the file round-trips through both
/// libhdf5/h5py/netCDF-C *and* oxih5.  The struct size is the declared 12 (the
/// members are contiguous), which keeps `Dtype::size()` on the read side equal
/// to the on-disk stride.
const REF_INDEX_DT_BODY_BYTES: [u8; REF_INDEX_DT_BODY] = [
    // Compound datatype message, version 1, 2 members, struct size 12.
    0x16, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    // Member 0 name "dataset\0", NUL-terminated, padded to 8 bytes.
    0x64, 0x61, 0x74, 0x61, 0x73, 0x65, 0x74, 0x00,
    // Member 0 byte offset = 0, then the 28-byte version-1 dimension block.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Member 0 base type: class 7 (object reference), size 8.
    0x17, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    // Member 1 name "dimension\0", NUL-terminated, padded to 16 bytes.
    0x64, 0x69, 0x6d, 0x65, 0x6e, 0x73, 0x69, 0x6f, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Member 1 byte offset = 8, then the 28-byte version-1 dimension block.
    0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Member 1 base type: class 0 (fixed-point), unsigned, size 4, precision 32.
    0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00,
];

/// Write the vlen-of-object-reference datatype body; returns bytes written.
fn write_vlen_ref_dtype(buf: &mut [u8], start: usize) -> usize {
    buf[start..start + VLEN_REF_DT_BODY].copy_from_slice(&VLEN_REF_DT_BODY_BYTES);
    VLEN_REF_DT_BODY
}

/// Write the `REFERENCE_LIST` compound datatype body; returns bytes written.
fn write_ref_index_dtype(buf: &mut [u8], start: usize) -> usize {
    buf[start..start + REF_INDEX_DT_BODY].copy_from_slice(&REF_INDEX_DT_BODY_BYTES);
    REF_INDEX_DT_BODY
}

/// The on-disk datatype-message body of the boolean enumeration,
/// `H5T_ENUM{ H5T_STD_I8LE; "FALSE" = 0, "TRUE" = 1 }`.
///
/// Byte-pinned against what h5py writes for a numpy `bool` array (captured from
/// `h5py`'s `H5Tencode`, minus its two-byte wrapper): a class-8 enumeration with
/// an inline class-0 signed 8-bit base type, two version-1 member names padded
/// to 8-byte boundaries, and two one-byte member values.  h5py recognises this
/// exact shape and reads such a dataset back as a numpy `bool` array; oxih5's
/// own class-8 parser decodes it to `Enum { base: i8, [("FALSE", 0), ("TRUE",
/// 1)] }`.  The base type is the natural 12-byte class-0 form (8 header + 4
/// version-1 properties, no trailing padding), because the enum parser advances
/// by the base type's *consumed* length — padding it to 16 would desynchronise
/// the member table.
const BOOL_ENUM_DT_BODY_BYTES: [u8; BOOL_ENUM_DT_BODY] = [
    // Enum header: class 8 (enum) v1, member count 2, base size 1.
    0x18, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    // Inline base type: class 0 (fixed-point) v1, signed little-endian, size 1,
    // bit offset 0, bit precision 8 — the 12-byte natural form.
    0x10, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00,
    // Member name "FALSE\0", NUL-terminated, padded to 8 bytes.
    0x46, 0x41, 0x4c, 0x53, 0x45, 0x00, 0x00, 0x00,
    // Member name "TRUE\0", NUL-terminated, padded to 8 bytes.
    0x54, 0x52, 0x55, 0x45, 0x00, 0x00, 0x00, 0x00,
    // Member values, one base-type byte each: FALSE = 0, TRUE = 1.
    0x00, 0x01,
];

/// Write the boolean enumeration datatype body; returns bytes written.
fn write_bool_enum_dtype(buf: &mut [u8], start: usize) -> usize {
    buf[start..start + BOOL_ENUM_DT_BODY].copy_from_slice(&BOOL_ENUM_DT_BODY_BYTES);
    BOOL_ENUM_DT_BODY
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
        // A fixed-length string *dataset* declares ASCII, so h5py reads it as
        // numpy `S<width>` (raw bytes); the width is the on-disk element size.
        ElemSpec::FixedStr { width } => Ok(write_string_dtype(
            buf,
            start,
            width,
            StrPad::NullPad,
            StrCharset::Ascii,
        )),
        ElemSpec::Bool => Ok(write_bool_enum_dtype(buf, start)),
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
    /// Scalar fixed-length string declared `H5T_STR_NULLTERM` (padding nibble 0)
    /// with a spare terminator byte (on-disk width = `len + 1`), rather than the
    /// `H5T_STR_NULLPAD` [`AttrKind::FixedStr`] uses.
    ///
    /// This exists for one interop reason: libnetcdf's dimension-scale API
    /// (`H5DSis_scale`) recognises a `DIMENSION_SCALE` only if the object's
    /// `CLASS` attribute is a `NULLTERM` string; a `NULLPAD` one makes netCDF-4
    /// fall back to `phony_dim_*` and lose the file's dimensions.  The payload
    /// bytes are identical — only the datatype's padding nibble and the trailing
    /// terminator byte differ — so the oxih5 reader (which trims at the first
    /// NUL) reads either form the same way.
    FixedStrNullTerm(String),
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
    /// A scalar numeric value of any writer-supported element type, carried as
    /// its little-endian on-disk bytes.  This is the generic seam every
    /// per-type `write_<t>_attr` funnels through, so a new numeric attribute
    /// type is one thin public method rather than a new variant here.
    Num {
        /// The element type the bytes encode.
        elem: ElemType,
        /// The value, already serialised little-endian; `elem.byte_size()` long.
        le_bytes: Vec<u8>,
    },
    /// A 1-D numeric array of any writer-supported element type, carried as its
    /// concatenated little-endian on-disk bytes.
    NumArray {
        /// The element type the bytes encode.
        elem: ElemType,
        /// The values, concatenated little-endian; a multiple of
        /// `elem.byte_size()` long.
        le_bytes: Vec<u8>,
    },
    /// A 1-D vlen-of-object-reference attribute (netCDF-4 `DIMENSION_LIST`): one
    /// independent sequence of target object names per element.
    VlenObjRefsByName(Vec<Vec<String>>),
    /// A 1-D compound `{ dataset: objref, dimension: u32 }` attribute (netCDF-4
    /// `REFERENCE_LIST`): one `(target object name, index)` pair per element.
    RefIndexList(Vec<(String, u32)>),
    /// **Not an attribute.**  A dataset-level custom fill value, carried on the
    /// dataset's attribute list purely as the writer's per-dataset side channel
    /// (the descriptor has no dedicated field).  [`resolve_attrs`] drops it, so
    /// it never becomes an attribute message; [`fill_value_bytes`] reads it, and
    /// the dataset object-header builder folds it into the fill-value message
    /// instead.  The value is already serialised little-endian and its width was
    /// validated against the dataset element type by
    /// [`super::FileWriter::set_fill_value_f64`] and its siblings when the
    /// sentinel was attached.
    FillValue {
        /// The fill value's little-endian bytes, the dataset element width long.
        le_bytes: Vec<u8>,
    },
    /// **Not an attribute.**  A request to store the dataset with the compact
    /// layout — its data inline in the object header rather than in a separate
    /// data area.  Like [`AttrKind::FillValue`] it rides the attribute list as
    /// the writer's per-dataset side channel: [`resolve_attrs`] drops it, and
    /// the dataset object-header builder reads it via [`compact_layout_data`]
    /// and emits a class-0 layout message carrying these bytes.  The data is
    /// *moved* here out of the descriptor's `raw` (which is emptied), so it is
    /// stored once — inline — with no separate data area reserved.
    CompactLayout {
        /// The dataset's raw little-endian data, to inline in the header.
        data: Vec<u8>,
    },
}

/// The reserved attribute name under which a [`AttrKind::FillValue`] sentinel is
/// parked on a dataset.  It never reaches disk (the sentinel is dropped before
/// any attribute is emitted), so this only has to be a name no real attribute is
/// likely to collide with when [`super::FileWriter::set_fill_value_f64`] looks
/// for an existing sentinel to replace.
pub(super) const FILL_VALUE_SENTINEL_NAME: &str = "\u{0}__oxih5_fill_value__";

/// The custom fill value carried on `attrs`, if any, as its little-endian bytes.
///
/// This is the read side of the [`AttrKind::FillValue`] side channel: the
/// dataset object-header builder calls it to decide whether the fill-value
/// message carries a defined value or the historic zero-length default.
pub(super) fn fill_value_bytes(attrs: &[AttrDesc]) -> Option<&[u8]> {
    attrs.iter().find_map(|attr| match &attr.kind {
        AttrKind::FillValue { le_bytes, .. } => Some(le_bytes.as_slice()),
        _ => None,
    })
}

/// The reserved attribute name under which a [`AttrKind::CompactLayout`] request
/// is parked on a dataset; it never reaches disk, exactly as
/// [`FILL_VALUE_SENTINEL_NAME`].
pub(super) const COMPACT_LAYOUT_SENTINEL_NAME: &str = "\u{0}__oxih5_compact_layout__";

/// The inline compact-layout data carried on `attrs`, if the dataset was marked
/// compact — the read side of the [`AttrKind::CompactLayout`] side channel.
pub(super) fn compact_layout_data(attrs: &[AttrDesc]) -> Option<&[u8]> {
    attrs.iter().find_map(|attr| match &attr.kind {
        AttrKind::CompactLayout { data } => Some(data.as_slice()),
        _ => None,
    })
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
    /// Scalar fixed-length string (`H5T_STR_NULLPAD`).
    FixedStr(&'a str),
    /// Scalar fixed-length string declared `H5T_STR_NULLTERM` with a spare
    /// terminator byte; see [`AttrKind::FixedStrNullTerm`].
    FixedStrNullTerm(&'a str),
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
    /// A scalar numeric value, as its little-endian on-disk bytes.
    Num {
        /// The element type the bytes encode.
        elem: ElemType,
        /// The value's little-endian bytes.
        le_bytes: &'a [u8],
    },
    /// A 1-D numeric array, as its concatenated little-endian on-disk bytes.
    NumArray {
        /// The element type the bytes encode.
        elem: ElemType,
        /// The values' concatenated little-endian bytes.
        le_bytes: &'a [u8],
    },
    /// A 1-D vlen-of-object-reference attribute.
    ///
    /// Each element is one independent [`VlenObjRefElem`]; the sequence's
    /// object-reference payload lives in the global heap, and the attribute's
    /// inline data is one 16-byte on-disk vlen reference per element.  The
    /// *count* of elements is fixed at resolution time, which is what makes it
    /// safe to size an object header before any heap address exists.
    VlenObjRefs {
        /// Per-element target names (each element's sequence), in order.
        names: &'a [Vec<String>],
        /// Per-element resolved sequences, one [`VlenObjRefElem`] per element.
        elems: Vec<VlenObjRefElem>,
    },
    /// A 1-D compound `{ dataset: objref, dimension: u32 }` attribute.
    RefIndexList {
        /// The `(target name, index)` pairs, in order.
        pairs: &'a [(String, u32)],
        /// Target object-header addresses, one per pair, initially
        /// [`UNDEFINED_ADDR`] and filled by [`fill_obj_refs`].
        addrs: Vec<u64>,
    },
}

/// One element of a resolved [`ResolvedAttrKind::VlenObjRefs`] attribute.
///
/// Its length (`addrs.len()`) is settled at resolution time and never changes,
/// so the attribute's on-disk size is stable before any address or heap
/// location is known.  The two placement fields are filled in two later passes:
/// [`fill_obj_refs`] resolves `addrs` once object-header addresses exist, then
/// [`register_vlen_objref_heap`] and [`fill_vlen_objref_locs`] place the
/// sequence in the global heap.
pub(super) struct VlenObjRefElem {
    /// Resolved target object-header addresses of this element's sequence, one
    /// per name; each entry starts at [`UNDEFINED_ADDR`].
    addrs: Vec<u64>,
    /// Absolute file address of the global-heap collection holding this
    /// element's sequence, once placed.  Stays [`UNDEFINED_ADDR`] until placed
    /// for a non-empty sequence — the sentinel that makes an unwired build fail
    /// loudly rather than emit a dangling vlen reference — and is `0` for an
    /// empty sequence, which needs no heap object.
    heap_addr: u64,
    /// 1-based object index within that collection, once placed; `0` otherwise.
    heap_idx: u32,
    /// 1-based *global* heap ordinal handed back by
    /// [`GlobalHeapWriter::write_bytes`]; `0` for an empty (unregistered)
    /// sequence.
    ///
    /// Written by [`register_vlen_objref_heap`] and read by
    /// [`fill_vlen_objref_locs`]; both are driven by the build-path hook that
    /// places attribute vlen payloads in the global heap.
    ordinal: u32,
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
            | ResolvedAttrKind::FixedStrNullTerm(_)
            | ResolvedAttrKind::F64(_)
            | ResolvedAttrKind::I64(_)
            | ResolvedAttrKind::I32(_)
            | ResolvedAttrKind::Num { .. } => None,
            ResolvedAttrKind::F64Array(values) => Some(values.len()),
            ResolvedAttrKind::I64Array(values) => Some(values.len()),
            ResolvedAttrKind::StrArray { values, .. } => Some(values.len()),
            ResolvedAttrKind::ObjRefs { addrs, .. } => Some(addrs.len()),
            ResolvedAttrKind::NumArray { elem, le_bytes } => {
                Some(le_bytes.len() / elem.byte_size())
            }
            ResolvedAttrKind::VlenObjRefs { elems, .. } => Some(elems.len()),
            ResolvedAttrKind::RefIndexList { pairs, .. } => Some(pairs.len()),
        }
    }

    /// The one and only per-attribute-kind size table.
    pub(super) fn sizes(&self) -> AttrSizes {
        let dspace = match self.vector_len() {
            None => SCALAR_DSPACE_BODY,
            Some(_) => VECTOR_DSPACE_BODY,
        };
        let (dtype, data) = match self {
            ResolvedAttrKind::FixedStr(s) => (STRING_DT_BODY, fixed_str_width(s)),
            ResolvedAttrKind::FixedStrNullTerm(s) => (STRING_DT_BODY, nullterm_str_width(s)),
            ResolvedAttrKind::F64(_) => (FLOAT_DT_BODY, 8),
            ResolvedAttrKind::I64(_) => (FIXED_DT_BODY, 8),
            ResolvedAttrKind::I32(_) => (FIXED_DT_BODY, 4),
            ResolvedAttrKind::F64Array(values) => (FLOAT_DT_BODY, values.len() * 8),
            ResolvedAttrKind::I64Array(values) => (FIXED_DT_BODY, values.len() * 8),
            ResolvedAttrKind::StrArray { values, width } => (STRING_DT_BODY, values.len() * width),
            ResolvedAttrKind::ObjRefs { addrs, .. } => (REF_DT_BODY, addrs.len() * 8),
            ResolvedAttrKind::Num { elem, le_bytes }
            | ResolvedAttrKind::NumArray { elem, le_bytes } => {
                (elem.dt_body_size(), le_bytes.len())
            }
            ResolvedAttrKind::VlenObjRefs { elems, .. } => {
                (VLEN_REF_DT_BODY, elems.len() * VLEN_REF_SIZE)
            }
            ResolvedAttrKind::RefIndexList { pairs, .. } => {
                (REF_INDEX_DT_BODY, pairs.len() * REF_INDEX_ELEM)
            }
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
        // A zero-length attribute name is rejected here — the one seam every
        // attribute's bytes flow through on the way to disk.  libhdf5 treats a
        // decoded name length of 0 as invalid and then cannot iterate ANY of the
        // object's attributes, so this poisons its siblings the same way an
        // empty fixed-string datatype does.
        if self.name.is_empty() {
            return Err(OxiH5Error::Format(
                "attribute name must not be empty: libhdf5 reports a zero-length \
                 attribute name as invalid and can then enumerate none of the \
                 object's attributes"
                    .to_string(),
            ));
        }
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
        check_size("attribute data", self.write_data(buf, pos)?, sizes.data)?;
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
                narrow("attribute string length", fixed_str_width(s))?,
                StrPad::NullPad,
                StrCharset::Utf8,
            )),
            ResolvedAttrKind::FixedStrNullTerm(s) => Ok(write_string_dtype(
                buf,
                start,
                narrow("attribute string length", nullterm_str_width(s))?,
                // NULLTERM + ASCII is what netCDF-C's NC_CHAR (`H5T_C_S1`) CLASS
                // attribute encodes to, and what libnetcdf's H5DS scale check
                // requires; the charset is not load-bearing (verified) but ASCII
                // keeps the bytes identical to a real netCDF-4 file.
                StrPad::NullTerm,
                StrCharset::Ascii,
            )),
            ResolvedAttrKind::StrArray { width, .. } => Ok(write_string_dtype(
                buf,
                start,
                narrow("attribute string width", *width)?,
                StrPad::NullPad,
                StrCharset::Utf8,
            )),
            ResolvedAttrKind::F64(_) | ResolvedAttrKind::F64Array(_) => {
                write_float_dtype(buf, start, 8)
            }
            ResolvedAttrKind::I64(_) | ResolvedAttrKind::I64Array(_) => {
                Ok(write_fixed_dtype(buf, start, 8, true))
            }
            ResolvedAttrKind::I32(_) => Ok(write_fixed_dtype(buf, start, 4, true)),
            ResolvedAttrKind::ObjRefs { .. } => Ok(write_ref_dtype(buf, start)),
            ResolvedAttrKind::Num { elem, .. } | ResolvedAttrKind::NumArray { elem, .. } => {
                write_datatype_body(buf, start, *elem)
            }
            ResolvedAttrKind::VlenObjRefs { .. } => Ok(write_vlen_ref_dtype(buf, start)),
            ResolvedAttrKind::RefIndexList { .. } => Ok(write_ref_index_dtype(buf, start)),
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
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` for a [`ResolvedAttrKind::VlenObjRefs`]
    /// attribute whose global-heap payload was never placed — the honest
    /// failure that stops a dangling vlen reference reaching disk when the
    /// build path forgot to call [`register_vlen_objref_heap`] /
    /// [`fill_vlen_objref_locs`].
    fn write_data(&self, buf: &mut [u8], start: usize) -> Result<usize, OxiH5Error> {
        Ok(match &self.kind {
            ResolvedAttrKind::FixedStr(s) => {
                // The region is pre-zeroed, so an empty string keeps the one
                // clamped NUL byte and a non-empty one fills its own length; in
                // both cases the count matches `fixed_str_width` in `sizes`.
                buf[start..start + s.len()].copy_from_slice(s.as_bytes());
                fixed_str_width(s)
            }
            ResolvedAttrKind::FixedStrNullTerm(s) => {
                // The region is pre-zeroed, so the trailing terminator byte (and
                // the empty-string case) is already NUL; the count matches
                // `nullterm_str_width` in `sizes`.
                buf[start..start + s.len()].copy_from_slice(s.as_bytes());
                nullterm_str_width(s)
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
            // A scalar and its array share this path: the region is pre-zeroed
            // and the bytes are already little-endian, so both just copy.
            ResolvedAttrKind::Num { le_bytes, .. }
            | ResolvedAttrKind::NumArray { le_bytes, .. } => {
                buf[start..start + le_bytes.len()].copy_from_slice(le_bytes);
                le_bytes.len()
            }
            ResolvedAttrKind::VlenObjRefs { elems, .. } => {
                for (i, elem) in elems.iter().enumerate() {
                    // A non-empty sequence must have been placed in the heap; an
                    // unplaced one is a build-path bug, reported here rather than
                    // silently written as a reference into heap address 0.
                    if !elem.addrs.is_empty() && elem.heap_addr == UNDEFINED_ADDR {
                        return Err(OxiH5Error::Format(format!(
                            "attribute '{}': vlen object-reference payload was never placed \
                             in the global heap — the build path must call \
                             register_vlen_objref_heap and fill_vlen_objref_locs",
                            self.name
                        )));
                    }
                    let base = start + i * VLEN_REF_SIZE;
                    write_u32_le(
                        buf,
                        base,
                        narrow("vlen object-reference sequence length", elem.addrs.len())?,
                    );
                    // An empty sequence writes the all-zero "null" vlen reference
                    // (heap_addr 0, index 0) libhdf5 uses for a zero-length vlen.
                    write_u64_le(buf, base + 4, elem.heap_addr);
                    write_u32_le(buf, base + 12, elem.heap_idx);
                }
                elems.len() * VLEN_REF_SIZE
            }
            ResolvedAttrKind::RefIndexList { pairs, addrs } => {
                for (i, ((_, index), &addr)) in pairs.iter().zip(addrs.iter()).enumerate() {
                    let base = start + i * REF_INDEX_ELEM;
                    write_u64_le(buf, base, addr);
                    write_u32_le(buf, base + 8, *index);
                }
                pairs.len() * REF_INDEX_ELEM
            }
        })
    }
}

/// On-disk width of a *scalar* fixed-length string, in bytes.
///
/// Zero is not a legal string-datatype size: libhdf5 cannot iterate an object's
/// attributes once one carries a size-0 string type, and that failure poisons
/// every sibling attribute on the object (it can list none of them).  So an
/// empty scalar string still occupies one byte — a single NUL — exactly as
/// [`str_array_width`] clamps an all-empty array to one byte per element.  This
/// is the single source of truth for the reserve site ([`ResolvedAttrKind::sizes`]),
/// the datatype-size site, and the data-width site, which must agree.
fn fixed_str_width(s: &str) -> usize {
    s.len().max(1)
}

/// On-disk width of a `H5T_STR_NULLTERM` fixed-length string attribute, in bytes.
///
/// A NULLTERM string reserves one byte beyond its content for the terminator, so
/// the width is `len + 1` — which is also 1 for the empty string, matching the
/// "no legal size-0 string" clamp [`fixed_str_width`] applies.
fn nullterm_str_width(s: &str) -> usize {
    s.len() + 1
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
        .filter_map(|attr| {
            let kind = match &attr.kind {
                // The fill-value sentinel is not an attribute — it rides the
                // attribute list only as a per-dataset side channel and is
                // folded into the fill-value message elsewhere, so it is dropped
                // here rather than sized or emitted as one.  That is what keeps
                // the reserve and emit passes agreeing that it is not an
                // attribute message.  The compact-layout request is the same
                // kind of side channel and is dropped for the same reason.
                AttrKind::FillValue { .. } | AttrKind::CompactLayout { .. } => return None,
                AttrKind::FixedStr(s) => ResolvedAttrKind::FixedStr(s.as_str()),
                AttrKind::FixedStrNullTerm(s) => ResolvedAttrKind::FixedStrNullTerm(s.as_str()),
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
                AttrKind::Num { elem, le_bytes } => ResolvedAttrKind::Num {
                    elem: *elem,
                    le_bytes,
                },
                AttrKind::NumArray { elem, le_bytes } => ResolvedAttrKind::NumArray {
                    elem: *elem,
                    le_bytes,
                },
                AttrKind::VlenObjRefsByName(sequences) => ResolvedAttrKind::VlenObjRefs {
                    names: sequences,
                    elems: sequences
                        .iter()
                        .map(|seq| VlenObjRefElem {
                            addrs: vec![UNDEFINED_ADDR; seq.len()],
                            // An empty sequence needs no heap object; its
                            // reference is the all-zero "null" vlen reference,
                            // so its heap address is a legitimate 0 rather than
                            // the unplaced sentinel.
                            heap_addr: if seq.is_empty() { 0 } else { UNDEFINED_ADDR },
                            heap_idx: 0,
                            ordinal: 0,
                        })
                        .collect(),
                },
                AttrKind::RefIndexList(pairs) => ResolvedAttrKind::RefIndexList {
                    pairs,
                    addrs: vec![UNDEFINED_ADDR; pairs.len()],
                },
            };
            Some(ResolvedAttr {
                name: &attr.name,
                kind,
            })
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
        let attr_name = attr.name;
        match &mut attr.kind {
            ResolvedAttrKind::ObjRefs { names, addrs } => {
                for (slot, name) in addrs.iter_mut().zip(names.iter()) {
                    *slot = resolve_ref(attr_name, name, path_to_addr)?;
                }
            }
            ResolvedAttrKind::RefIndexList { pairs, addrs } => {
                for (slot, (name, _)) in addrs.iter_mut().zip(pairs.iter()) {
                    *slot = resolve_ref(attr_name, name, path_to_addr)?;
                }
            }
            ResolvedAttrKind::VlenObjRefs { names, elems } => {
                for (elem, seq) in elems.iter_mut().zip(names.iter()) {
                    for (slot, name) in elem.addrs.iter_mut().zip(seq.iter()) {
                        *slot = resolve_ref(attr_name, name, path_to_addr)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Resolve one object-reference target name to its object-header address.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` naming the attribute and the target if the name
/// resolves to no dataset or group in the file.
fn resolve_ref(
    attr_name: &str,
    name: &str,
    path_to_addr: &HashMap<String, u64>,
) -> Result<u64, OxiH5Error> {
    // Targets may be written with or without a leading separator; the map is
    // keyed by path from the root without one.
    let key = name.strip_prefix('/').unwrap_or(name);
    path_to_addr.get(key).copied().ok_or_else(|| {
        OxiH5Error::Format(format!(
            "attribute '{attr_name}': object reference target '{name}' names no dataset \
             or group in the file"
        ))
    })
}

/// Register the global-heap payloads of every vlen-object-reference attribute in
/// `attrs` with the shared collection writer `gcol`.
///
/// Each non-empty sequence's payload is its resolved object-header addresses,
/// concatenated as 8-byte little-endian words (the on-disk representation of a
/// sequence of `H5T_STD_REF_OBJ`), stored as one heap object; the returned
/// 1-based global ordinal is recorded on the element for
/// [`fill_vlen_objref_locs`] to resolve into a concrete `(address, index)` once
/// the collections are laid out.
///
/// This must run **after** [`fill_obj_refs`] (so the addresses exist) and
/// **before** [`GlobalHeapWriter::build_collections`].  An empty sequence needs
/// no heap object and is skipped.
///
/// Called by `GroupPlan::register_vlen_objref_attrs` from the build path, which
/// places attribute vlen payloads in the global heap.
pub(super) fn register_vlen_objref_heap(
    attrs: &mut [ResolvedAttr<'_>],
    gcol: &mut GlobalHeapWriter,
) {
    for attr in attrs {
        if let ResolvedAttrKind::VlenObjRefs { elems, .. } = &mut attr.kind {
            for elem in elems.iter_mut() {
                if elem.addrs.is_empty() {
                    continue;
                }
                let mut payload = Vec::with_capacity(elem.addrs.len() * 8);
                for &addr in &elem.addrs {
                    payload.extend_from_slice(&addr.to_le_bytes());
                }
                elem.ordinal = gcol.write_bytes(&payload);
            }
        }
    }
}

/// Resolve the heap ordinals recorded by [`register_vlen_objref_heap`] into the
/// absolute collection address and 1-based local index each sequence landed at.
///
/// `collection_addrs` is the absolute file address of each collection in ordinal
/// order and `locations` is the per-object map returned by
/// [`GlobalHeapWriter::build_collections`].  This must run **after** the
/// collections are laid out.  A stale ordinal that names no location is left at
/// its unplaced sentinel, so [`ResolvedAttr::write_body`] reports it rather than
/// writing a dangling reference.
///
/// Called by `GroupPlan::fill_vlen_objref_locs` from the build path, once the
/// global-heap collections are laid out.
pub(super) fn fill_vlen_objref_locs(
    attrs: &mut [ResolvedAttr<'_>],
    collection_addrs: &[u64],
    locations: &[HeapObjectLocation],
) {
    for attr in attrs {
        if let ResolvedAttrKind::VlenObjRefs { elems, .. } = &mut attr.kind {
            for elem in elems.iter_mut() {
                if elem.ordinal == 0 {
                    continue; // empty sequence: already the null vlen reference
                }
                if let Some(loc) = (elem.ordinal as usize)
                    .checked_sub(1)
                    .and_then(|z| locations.get(z))
                {
                    if let Some(&addr) = collection_addrs.get(loc.collection as usize) {
                        elem.heap_addr = addr;
                        elem.heap_idx = loc.index;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "elem_tests.rs"]
mod tests;
