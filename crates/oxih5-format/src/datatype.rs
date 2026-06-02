//! HDF5 datatype message parser (message type 0x0003).
//!
//! Supports all 11 datatype classes (0–10):
//!   0 = Integer, 1 = Float, 2 = Time (not in HDF5 spec, skipped),
//!   3 = String, 4 = Bitfield, 5 = Opaque, 6 = Compound, 7 = Reference,
//!   8 = Enum, 9 = VarLen, 10 = Array

use oxih5_core::{ByteOrder, Charset, CompoundField, Dtype, OxiH5Error, RefType};

const MAX_NESTING_DEPTH: usize = 32;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Parse a datatype message body, returning the Dtype.
pub fn parse_datatype(body: &[u8]) -> Result<Dtype, OxiH5Error> {
    let (dtype, _) = parse_datatype_consuming(body, 0)?;
    Ok(dtype)
}

/// Parse a datatype message body, returning both the Dtype and number of bytes consumed.
///
/// This is needed for inline type messages embedded in compound, enum, array, and vlen types.
pub fn parse_datatype_consuming(body: &[u8], depth: usize) -> Result<(Dtype, usize), OxiH5Error> {
    if depth > MAX_NESTING_DEPTH {
        return Err(OxiH5Error::Format("datatype nesting too deep".into()));
    }
    if body.len() < 8 {
        return Err(OxiH5Error::Format(format!(
            "datatype body too short: {} bytes",
            body.len()
        )));
    }

    let class_and_version = body[0];
    let class = class_and_version & 0x0F;
    let version = (class_and_version >> 4) & 0x0F;
    let bit_fields_0 = body[1];

    // Element size in bytes
    let elem_size = read_u32_le(body, 4)? as usize;

    match class {
        // -----------------------------------------------------------------------
        // Class 0: Fixed-point integer
        // -----------------------------------------------------------------------
        0 => {
            let order = if bit_fields_0 & 0x01 != 0 {
                ByteOrder::Big
            } else {
                ByteOrder::Little
            };
            let signed = bit_fields_0 & 0x08 != 0;
            // Version 1: Properties section is 4 bytes (bit_offset u16 + bit_precision u16).
            // Version 2+: no separate Properties section (all encoded in class-bits).
            let consumed = if version == 1 { 8 + 4 } else { 8 };
            Ok((
                Dtype::Int {
                    size: elem_size,
                    signed,
                    order,
                },
                consumed,
            ))
        }

        // -----------------------------------------------------------------------
        // Class 1: Floating-point
        // -----------------------------------------------------------------------
        1 => {
            let order = if bit_fields_0 & 0x01 != 0 {
                ByteOrder::Big
            } else {
                ByteOrder::Little
            };
            // Version 1: Properties section is 12 bytes
            //   (bit_offset_exp u2, size_exp u1, bit_offset_mant u2, size_mant u1,
            //    exponent_bias u4, pad/sign flags u2).
            // Version 2+: no separate Properties section.
            let consumed = if version == 1 { 8 + 12 } else { 8 };
            Ok((
                Dtype::Float {
                    size: elem_size,
                    order,
                },
                consumed,
            ))
        }

        // -----------------------------------------------------------------------
        // Class 3: String
        // -----------------------------------------------------------------------
        3 => {
            // bit_fields_0 bits 4..7: charset (0=ASCII, 1=UTF-8)
            let charset = if (bit_fields_0 >> 4) & 0x0F != 0 {
                Charset::Utf8
            } else {
                Charset::Ascii
            };
            // For fixed-length strings, elem_size is the byte length.
            // HDF5 variable-length strings are represented as class 9 (VLen) wrapping a character
            // type, so here we always produce a fixed-length string.
            Ok((
                Dtype::String {
                    fixed_len: Some(elem_size),
                    charset,
                },
                8,
            ))
        }

        // -----------------------------------------------------------------------
        // Class 4: Bitfield
        // -----------------------------------------------------------------------
        4 => {
            let order = if bit_fields_0 & 0x01 != 0 {
                ByteOrder::Big
            } else {
                ByteOrder::Little
            };
            // Version 1: Properties section is 4 bytes (bit_offset u16 + bit_precision u16),
            // same layout as Int. Version 2+: no separate Properties section.
            let consumed = if version == 1 { 8 + 4 } else { 8 };
            Ok((
                Dtype::Bitfield {
                    size: elem_size,
                    order,
                },
                consumed,
            ))
        }

        // -----------------------------------------------------------------------
        // Class 5: Opaque
        // -----------------------------------------------------------------------
        5 => {
            // bit_fields_0 = tag length in bytes
            let tag_len = bit_fields_0 as usize;
            // Properties start at body[8]; tag bytes are body[8..8+tag_len]
            let tag_end = 8 + tag_len;
            let tag_bytes = body.get(8..tag_end).ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "opaque: tag at body[8..{}] out of bounds (len={})",
                    tag_end,
                    body.len()
                ))
            })?;
            let nul_pos = tag_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(tag_bytes.len());
            let tag = std::str::from_utf8(&tag_bytes[..nul_pos])
                .unwrap_or("")
                .to_string();
            // Properties section is tag_len bytes padded to 8-byte boundary
            let props_size = (tag_len + 7) & !7;
            Ok((
                Dtype::Opaque {
                    size: elem_size,
                    tag,
                },
                8 + props_size,
            ))
        }

        // -----------------------------------------------------------------------
        // Class 6: Compound
        // -----------------------------------------------------------------------
        6 => parse_compound(body, version, elem_size, depth),

        // -----------------------------------------------------------------------
        // Class 7: Reference
        // -----------------------------------------------------------------------
        7 => {
            // bit_fields_0 bits 0..3: reference type (0=object, 1=region)
            let ref_type = if bit_fields_0 & 0x0F != 0 {
                RefType::Region
            } else {
                RefType::Object
            };
            Ok((Dtype::Reference { ref_type }, 8))
        }

        // -----------------------------------------------------------------------
        // Class 8: Enumeration
        // -----------------------------------------------------------------------
        8 => parse_enum(body, version, elem_size, depth),

        // -----------------------------------------------------------------------
        // Class 9: Variable-length
        // -----------------------------------------------------------------------
        9 => {
            // bit_fields_0 bits 0..3: type (0=sequence, 1=string)
            let is_string = bit_fields_0 & 0x0F == 1;
            // Properties start at body[8]: inline base type message
            if body.len() < 9 {
                return Err(OxiH5Error::Format(
                    "vlen: body too short for base type".into(),
                ));
            }
            let (base_dtype, base_consumed) = parse_datatype_consuming(&body[8..], depth + 1)?;
            let total_consumed = 8 + base_consumed;
            if is_string {
                // VLen string: base type is a character dtype, expose as String VarLen
                Ok((
                    Dtype::String {
                        fixed_len: None,
                        charset: match &base_dtype {
                            Dtype::String { charset, .. } => charset.clone(),
                            _ => Charset::Ascii,
                        },
                    },
                    total_consumed,
                ))
            } else {
                Ok((
                    Dtype::VarLen {
                        base: Box::new(base_dtype),
                    },
                    total_consumed,
                ))
            }
        }

        // -----------------------------------------------------------------------
        // Class 10: Array
        // -----------------------------------------------------------------------
        10 => parse_array(body, version, depth),

        other => Err(OxiH5Error::UnsupportedDatatype(other)),
    }
}

// ---------------------------------------------------------------------------
// Compound datatype (class 6)
// ---------------------------------------------------------------------------

fn parse_compound(
    body: &[u8],
    version: u8,
    _struct_size: usize,
    depth: usize,
) -> Result<(Dtype, usize), OxiH5Error> {
    // Number of members = (body[1] | (body[2]<<8) | (body[3]<<16)) — 24-bit LE
    let nmembers = (body[1] as usize) | ((body[2] as usize) << 8) | ((body[3] as usize) << 16);

    let mut pos = 8usize; // properties start at byte 8
    let mut fields = Vec::with_capacity(nmembers);

    for _ in 0..nmembers {
        // Read NUL-terminated member name
        let (name, name_consumed) = read_nul_string_padded(body, pos, version)?;
        pos += name_consumed;

        // Byte offset within compound (4 bytes for v1/v2, variable for v3)
        if pos + 4 > body.len() {
            return Err(OxiH5Error::Format(
                "compound: truncated at member byte-offset".into(),
            ));
        }
        let member_offset = read_u32_le(body, pos)? as usize;
        pos += 4;

        if version == 1 {
            // Version 1: after byte_offset (4), skip the per-member legacy dimensionality info:
            //   dimensionality (1) + reserved (3) + permutation_index (4) + reserved (4)
            //   + dim_sizes (4*4 = 16) = 28 bytes.
            if pos + 28 > body.len() {
                return Err(OxiH5Error::Format(
                    "compound v1: truncated at dim info".into(),
                ));
            }
            pos += 28;
        }

        // Parse inline member type
        if pos >= body.len() {
            return Err(OxiH5Error::Format(
                "compound: truncated before member type".into(),
            ));
        }
        let (member_dtype, type_consumed) = parse_datatype_consuming(&body[pos..], depth + 1)?;
        pos += type_consumed;

        fields.push(CompoundField {
            name,
            offset: member_offset,
            dtype: member_dtype,
        });
    }

    Ok((Dtype::Compound { fields }, pos))
}

// ---------------------------------------------------------------------------
// Enum datatype (class 8)
// ---------------------------------------------------------------------------

fn parse_enum(
    body: &[u8],
    version: u8,
    base_size: usize,
    depth: usize,
) -> Result<(Dtype, usize), OxiH5Error> {
    // Number of members in lower 24 bits of class-bits (body[1..4])
    let nmembers = (body[1] as usize) | ((body[2] as usize) << 8) | ((body[3] as usize) << 16);

    // Properties at body[8]: inline base type message
    if body.len() < 9 {
        return Err(OxiH5Error::Format(
            "enum: body too short for base type".into(),
        ));
    }
    let (base_dtype, base_consumed) = parse_datatype_consuming(&body[8..], depth + 1)?;
    let mut pos = 8 + base_consumed;

    // Member names: NUL-terminated, padded to 8-byte boundary for v1, not padded for v2+
    let mut member_names: Vec<String> = Vec::with_capacity(nmembers);
    for _ in 0..nmembers {
        let (name, consumed) = read_nul_string_padded(body, pos, version)?;
        member_names.push(name);
        pos += consumed;
    }

    // Member values: each is `base_size` bytes (LE integer)
    let mut members = Vec::with_capacity(nmembers);
    for (i, name) in member_names.into_iter().enumerate() {
        if pos + base_size > body.len() {
            return Err(OxiH5Error::Format(format!(
                "enum: truncated at member value {} (pos={} base_size={} body_len={})",
                i,
                pos,
                base_size,
                body.len()
            )));
        }
        let value = read_int_as_i64(&body[pos..pos + base_size])?;
        pos += base_size;
        members.push((name, value));
    }

    Ok((
        Dtype::Enum {
            base: Box::new(base_dtype),
            members,
        },
        pos,
    ))
}

// ---------------------------------------------------------------------------
// Array datatype (class 10)
// ---------------------------------------------------------------------------

fn parse_array(body: &[u8], version: u8, depth: usize) -> Result<(Dtype, usize), OxiH5Error> {
    if version == 1 || version == 0 {
        // Version 1: body[8..12] = ndims (4 bytes), body[12..16] = reserved (4),
        //            body[16..16+4*ndims] = dim_sizes (u32 each),
        //            body[16+4*ndims..16+8*ndims] = dim_perm (u32 each),
        //            then inline base type
        if body.len() < 12 {
            return Err(OxiH5Error::Format(
                "array v1: body too short for ndims".into(),
            ));
        }
        let ndims = read_u32_le(body, 8)? as usize;
        // reserved 4 bytes at body[12]
        let dims_start = 16;
        let dims_end = dims_start + ndims * 4;
        let perm_end = dims_end + ndims * 4;
        if body.len() < perm_end {
            return Err(OxiH5Error::Format(format!(
                "array v1: body too short for dims (need {})",
                perm_end
            )));
        }
        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            dims.push(read_u32_le(body, dims_start + i * 4)? as usize);
        }
        let (base_dtype, base_consumed) = parse_datatype_consuming(&body[perm_end..], depth + 1)?;
        let total = perm_end + base_consumed;
        Ok((
            Dtype::Array {
                base: Box::new(base_dtype),
                dims,
            },
            total,
        ))
    } else {
        // Version 2: body[8] = ndims (1 byte), body[9..9+8*ndims] = dim_sizes (u64 each),
        //            then inline base type
        if body.len() < 9 {
            return Err(OxiH5Error::Format(
                "array v2: body too short for ndims".into(),
            ));
        }
        let ndims = body[8] as usize;
        let dims_start = 9;
        let dims_end = dims_start + ndims * 8;
        if body.len() < dims_end {
            return Err(OxiH5Error::Format(format!(
                "array v2: body too short for dims (need {})",
                dims_end
            )));
        }
        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            dims.push(read_u64_le(body, dims_start + i * 8)? as usize);
        }
        let (base_dtype, base_consumed) = parse_datatype_consuming(&body[dims_end..], depth + 1)?;
        let total = dims_end + base_consumed;
        Ok((
            Dtype::Array {
                base: Box::new(base_dtype),
                dims,
            },
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a NUL-terminated string from `body` at `pos`.
/// For version 1, pad to 8-byte boundary. For version 2+, no padding.
fn read_nul_string_padded(
    body: &[u8],
    pos: usize,
    version: u8,
) -> Result<(String, usize), OxiH5Error> {
    if pos >= body.len() {
        return Err(OxiH5Error::Format(format!(
            "read_nul_string: pos {} >= body len {}",
            pos,
            body.len()
        )));
    }
    let slice = &body[pos..];
    let nul_pos = slice.iter().position(|&b| b == 0).ok_or_else(|| {
        OxiH5Error::Format(format!("read_nul_string: no NUL terminator at pos {}", pos))
    })?;
    let name = std::str::from_utf8(&slice[..nul_pos])
        .unwrap_or("")
        .to_string();
    // raw length includes the NUL byte
    let raw_len = nul_pos + 1;
    let consumed = if version <= 1 {
        // Pad to 8-byte boundary
        (raw_len + 7) & !7
    } else {
        raw_len
    };
    Ok((name, consumed))
}

/// Read up to 8 bytes as a little-endian signed integer.
fn read_int_as_i64(bytes: &[u8]) -> Result<i64, OxiH5Error> {
    match bytes.len() {
        1 => Ok(bytes[0] as i8 as i64),
        2 => Ok(i16::from_le_bytes([bytes[0], bytes[1]]) as i64),
        4 => Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64),
        8 => Ok(i64::from_le_bytes(bytes.try_into().map_err(|_| {
            OxiH5Error::Format("read_int_as_i64: slice error".into())
        })?)),
        n => Err(OxiH5Error::Format(format!(
            "read_int_as_i64: unsupported size {}",
            n
        ))),
    }
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, OxiH5Error> {
    if offset + 4 > data.len() {
        return Err(OxiH5Error::Format(format!(
            "read_u32_le: offset {} out of bounds (len={})",
            offset,
            data.len()
        )));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_u64_le(data: &[u8], offset: usize) -> Result<u64, OxiH5Error> {
    if offset + 8 > data.len() {
        return Err(OxiH5Error::Format(format!(
            "read_u64_le: offset {} out of bounds (len={})",
            offset,
            data.len()
        )));
    }
    let arr: [u8; 8] = data[offset..offset + 8]
        .try_into()
        .map_err(|_| OxiH5Error::Format("read_u64_le: slice error".into()))?;
    Ok(u64::from_le_bytes(arr))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal datatype header (8 bytes).
    /// For version-1 Int/Float/Bitfield types used as *inline* nested base types, append
    /// Properties bytes with `build_int_props()` or `build_float_props()` after calling this.
    fn build_dtype_header(class: u8, version: u8, bit_fields: [u8; 3], size: u32) -> Vec<u8> {
        let mut body = vec![0u8; 8];
        body[0] = (version << 4) | (class & 0x0F);
        body[1] = bit_fields[0];
        body[2] = bit_fields[1];
        body[3] = bit_fields[2];
        body[4..8].copy_from_slice(&size.to_le_bytes());
        body
    }

    /// Build version-1 Int/Bitfield Properties section (4 bytes: bit_offset=0, bit_precision=size*8).
    fn build_int_props_v1(bit_precision: u16) -> [u8; 4] {
        let mut props = [0u8; 4];
        props[0..2].copy_from_slice(&0u16.to_le_bytes()); // bit_offset
        props[2..4].copy_from_slice(&bit_precision.to_le_bytes()); // bit_precision
        props
    }

    /// Build version-1 Float Properties section (12 bytes, minimal/placeholder values).
    fn build_float_props_v1() -> [u8; 12] {
        // bit_offset_exp(2) + size_exp(1) + bit_offset_mant(2) + size_mant(1) + exp_bias(4) + flags(2)
        [0u8; 12]
    }

    #[test]
    fn test_integer_le_signed() {
        let mut body = build_dtype_header(0, 1, [0x08, 0, 0], 4);
        body.extend_from_slice(&build_int_props_v1(32)); // 4-byte Properties for v1 Int
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little
            }
        );
    }

    #[test]
    fn test_integer_be_unsigned() {
        let mut body = build_dtype_header(0, 1, [0x01, 0, 0], 2);
        body.extend_from_slice(&build_int_props_v1(16)); // 4-byte Properties for v1 Int
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Int {
                size: 2,
                signed: false,
                order: ByteOrder::Big
            }
        );
    }

    #[test]
    fn test_float_le() {
        let mut body = build_dtype_header(1, 1, [0x00, 0, 0], 4);
        body.extend_from_slice(&build_float_props_v1()); // 12-byte Properties for v1 Float
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Float {
                size: 4,
                order: ByteOrder::Little
            }
        );
    }

    #[test]
    fn test_float_be() {
        let mut body = build_dtype_header(1, 1, [0x01, 0, 0], 8);
        body.extend_from_slice(&build_float_props_v1()); // 12-byte Properties for v1 Float
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Float {
                size: 8,
                order: ByteOrder::Big
            }
        );
    }

    #[test]
    fn test_string_ascii() {
        // class 3, version 1, bit_fields[0] = 0 (null-pad, ASCII)
        let body = build_dtype_header(3, 1, [0x00, 0, 0], 16);
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::String {
                fixed_len: Some(16),
                charset: Charset::Ascii
            }
        );
    }

    #[test]
    fn test_string_utf8() {
        // class 3, bit_fields[0] bit 4 set = UTF-8
        let body = build_dtype_header(3, 1, [0x10, 0, 0], 32);
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::String {
                fixed_len: Some(32),
                charset: Charset::Utf8
            }
        );
    }

    #[test]
    fn test_bitfield_le() {
        let mut body = build_dtype_header(4, 1, [0x00, 0, 0], 4);
        body.extend_from_slice(&build_int_props_v1(32)); // 4-byte Properties for v1 Bitfield
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Bitfield {
                size: 4,
                order: ByteOrder::Little
            }
        );
    }

    #[test]
    fn test_bitfield_be() {
        let mut body = build_dtype_header(4, 1, [0x01, 0, 0], 4);
        body.extend_from_slice(&build_int_props_v1(32)); // 4-byte Properties for v1 Bitfield
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Bitfield {
                size: 4,
                order: ByteOrder::Big
            }
        );
    }

    #[test]
    fn test_opaque_no_tag() {
        // class 5, tag length = 0
        let body = build_dtype_header(5, 1, [0x00, 0, 0], 8);
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Opaque {
                size: 8,
                tag: String::new()
            }
        );
    }

    #[test]
    fn test_opaque_with_tag() {
        // class 5, tag length = 5 ("hello")
        let mut body = build_dtype_header(5, 1, [5, 0, 0], 8);
        // properties (tag): "hello\0\0\0" (padded to 8 bytes)
        body.extend_from_slice(b"hello\x00\x00\x00");
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Opaque {
                size: 8,
                tag: "hello".into()
            }
        );
    }

    #[test]
    fn test_reference_object() {
        let body = build_dtype_header(7, 1, [0x00, 0, 0], 8);
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Reference {
                ref_type: RefType::Object
            }
        );
    }

    #[test]
    fn test_reference_region() {
        let body = build_dtype_header(7, 1, [0x01, 0, 0], 12);
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Reference {
                ref_type: RefType::Region
            }
        );
    }

    #[test]
    fn test_vlen_sequence() {
        // class 9, version 1, base type = int32 LE signed
        let mut body = build_dtype_header(9, 1, [0x00, 0, 0], 16);
        // inline base type: int32 LE signed (header + v1 Properties)
        body.extend_from_slice(&build_dtype_header(0, 1, [0x08, 0, 0], 4));
        body.extend_from_slice(&build_int_props_v1(32));
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::VarLen {
                base: Box::new(Dtype::Int {
                    size: 4,
                    signed: true,
                    order: ByteOrder::Little
                })
            }
        );
    }

    #[test]
    fn test_vlen_string() {
        // class 9, version 1, bit_fields[0] = 1 (string type), base = ASCII string
        let mut body = build_dtype_header(9, 1, [0x01, 0, 0], 16);
        // inline base type: ASCII string (class 3)
        body.extend_from_slice(&build_dtype_header(3, 1, [0x00, 0, 0], 1));
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::String {
                fixed_len: None,
                charset: Charset::Ascii
            }
        );
    }

    #[test]
    fn test_array_v2() {
        // class 10, version 2: ndims=2, dims=[3,4], base = float32 LE v1
        let mut body = build_dtype_header(10, 2, [0, 0, 0], 48);
        body.push(2u8); // ndims
        body.extend_from_slice(&3u64.to_le_bytes()); // dim[0]
        body.extend_from_slice(&4u64.to_le_bytes()); // dim[1]
                                                     // base type: float32 LE (header + v1 Properties)
        body.extend_from_slice(&build_dtype_header(1, 1, [0x00, 0, 0], 4));
        body.extend_from_slice(&build_float_props_v1());
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Array {
                base: Box::new(Dtype::Float {
                    size: 4,
                    order: ByteOrder::Little
                }),
                dims: vec![3, 4]
            }
        );
    }

    #[test]
    fn test_array_v1() {
        // class 10, version 1: ndims=1, dim=[5], perm=[0], base = int8
        let mut body = build_dtype_header(10, 1, [0, 0, 0], 5);
        // ndims (4 bytes) + reserved (4 bytes)
        body.extend_from_slice(&1u32.to_le_bytes()); // ndims
        body.extend_from_slice(&0u32.to_le_bytes()); // reserved
                                                     // dim_sizes (4 bytes each)
        body.extend_from_slice(&5u32.to_le_bytes()); // dim[0]
                                                     // dim_perm (4 bytes each)
        body.extend_from_slice(&0u32.to_le_bytes()); // perm[0]
                                                     // base type: int8 LE unsigned (header + v1 Properties)
        body.extend_from_slice(&build_dtype_header(0, 1, [0x00, 0, 0], 1));
        body.extend_from_slice(&build_int_props_v1(8));
        let dtype = parse_datatype(&body).unwrap();
        assert_eq!(
            dtype,
            Dtype::Array {
                base: Box::new(Dtype::Int {
                    size: 1,
                    signed: false,
                    order: ByteOrder::Little
                }),
                dims: vec![5]
            }
        );
    }

    #[test]
    fn test_compound_v1_simple() {
        // Compound v1 with 2 members: x (int32 @ offset 0), y (float32 @ offset 4)
        // struct_size = 8
        let nmembers: u32 = 2;
        let struct_size: u32 = 8;
        let mut body = vec![0u8; 8];
        body[0] = (1u8 << 4) | 6u8; // version=1, class=6
        body[1] = (nmembers & 0xFF) as u8;
        body[2] = ((nmembers >> 8) & 0xFF) as u8;
        body[3] = ((nmembers >> 16) & 0xFF) as u8;
        body[4..8].copy_from_slice(&struct_size.to_le_bytes());

        // Member 1: "x\0\0\0\0\0\0\0" (8-byte padded), offset=0, dim_info (28 bytes), int32 LE
        body.extend_from_slice(b"x\x00\x00\x00\x00\x00\x00\x00"); // name padded to 8
        body.extend_from_slice(&0u32.to_le_bytes()); // byte offset
        body.extend_from_slice(&[0u8; 28]); // dim info v1: dim(1)+rsvd(3)+perm(4)+rsvd(4)+dim_sizes(16)
        body.extend_from_slice(&build_dtype_header(0, 1, [0x08, 0, 0], 4)); // int32 LE signed header
        body.extend_from_slice(&build_int_props_v1(32)); // v1 Int Properties

        // Member 2: "y\0\0\0\0\0\0\0" (8-byte padded), offset=4, dim_info (28 bytes), float32 LE
        body.extend_from_slice(b"y\x00\x00\x00\x00\x00\x00\x00"); // name padded to 8
        body.extend_from_slice(&4u32.to_le_bytes()); // byte offset
        body.extend_from_slice(&[0u8; 28]); // dim info v1
        body.extend_from_slice(&build_dtype_header(1, 1, [0x00, 0, 0], 4)); // float32 LE header
        body.extend_from_slice(&build_float_props_v1()); // v1 Float Properties

        let dtype = parse_datatype(&body).unwrap();
        match dtype {
            Dtype::Compound { fields } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[0].offset, 0);
                assert_eq!(
                    fields[0].dtype,
                    Dtype::Int {
                        size: 4,
                        signed: true,
                        order: ByteOrder::Little
                    }
                );
                assert_eq!(fields[1].name, "y");
                assert_eq!(fields[1].offset, 4);
                assert_eq!(
                    fields[1].dtype,
                    Dtype::Float {
                        size: 4,
                        order: ByteOrder::Little
                    }
                );
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn test_enum_v1() {
        // Enum v1 with 2 members: base = int32 LE
        let nmembers: u32 = 2;
        let base_size: u32 = 4;
        let mut body = vec![0u8; 8];
        body[0] = (1u8 << 4) | 8u8; // version=1, class=8
        body[1] = (nmembers & 0xFF) as u8;
        body[2] = ((nmembers >> 8) & 0xFF) as u8;
        body[3] = ((nmembers >> 16) & 0xFF) as u8;
        body[4..8].copy_from_slice(&base_size.to_le_bytes());

        // Base type: int32 LE signed (header + v1 Properties)
        body.extend_from_slice(&build_dtype_header(0, 1, [0x08, 0, 0], 4));
        body.extend_from_slice(&build_int_props_v1(32));

        // Member names (v1: padded to 8-byte boundary)
        body.extend_from_slice(b"OFF\x00\x00\x00\x00\x00"); // "OFF\0" padded to 8
        body.extend_from_slice(b"ON\x00\x00\x00\x00\x00\x00"); // "ON\0" padded to 8

        // Member values (int32 LE each)
        body.extend_from_slice(&0i32.to_le_bytes()); // OFF = 0
        body.extend_from_slice(&1i32.to_le_bytes()); // ON = 1

        let dtype = parse_datatype(&body).unwrap();
        match dtype {
            Dtype::Enum { base, members } => {
                assert_eq!(
                    *base,
                    Dtype::Int {
                        size: 4,
                        signed: true,
                        order: ByteOrder::Little
                    }
                );
                assert_eq!(members.len(), 2);
                assert_eq!(members[0], ("OFF".into(), 0));
                assert_eq!(members[1], ("ON".into(), 1));
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn test_too_short_errors() {
        // body too short
        assert!(parse_datatype(&[]).is_err());
        assert!(parse_datatype(&[0u8; 7]).is_err());
    }

    #[test]
    fn test_depth_guard() {
        // Build a deeply nested VLen type (depth=33 should fail)
        fn build_vlen(base: &[u8]) -> Vec<u8> {
            let mut body = vec![0u8; 8];
            body[0] = (1u8 << 4) | 9u8; // version=1, class=9 (vlen)
            body[1] = 0; // sequence
            body[4..8].copy_from_slice(&16u32.to_le_bytes());
            body.extend_from_slice(base);
            body
        }
        let int_body = {
            let mut b = vec![0u8; 8];
            b[0] = 1u8 << 4; // version=1, class=0 (int)
            b[1] = 0x08;
            b[4..8].copy_from_slice(&4u32.to_le_bytes());
            b
        };
        // Build 40 nested vlen layers
        let mut current = int_body;
        for _ in 0..40 {
            current = build_vlen(&current);
        }
        assert!(parse_datatype(&current).is_err());
    }
}
