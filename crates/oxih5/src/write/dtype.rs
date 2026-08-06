//! Structured datatype encoders — the classes an [`ElemType`] cannot name.
//!
//! [`super::elem::ElemType`] is a closed, `Copy` set: eleven numeric families in
//! two byte orders, a fixed string, a vlen string, a boolean.  Every one of them
//! has a fixed-size datatype message, which is what lets the writer's two passes
//! size a dataset header without allocating.
//!
//! The five classes here — **compound** (6), **variable-length sequence** (9
//! subtype 0), **array** (10), **opaque** (5) and **bitfield** (4) — are not
//! like that: their message length depends on member names, dimension counts and
//! tag strings.  Rather than making `ElemType` variable-length (and non-`Copy`,
//! which would ripple through chunk geometry, the filter pipeline and the
//! payload builder), a dataset may carry an [`EncodedDtype`]: a datatype message
//! body encoded **once**, up front, together with the element stride that body
//! declares.  The object header then emits those bytes verbatim.
//!
//! Every encoder here produces the **version-1** form of its class — that is
//! the form `oxih5_format::datatype` decodes and the form libhdf5 accepts from
//! any writer — with one exception it does not get to make: the **array** class
//! was introduced *with* datatype message version 2 and libhdf5 rejects an
//! older one outright, so [`array`] emits version 2 and says so.
//!
//! Version 1 spells a compound member's now-obsolete array dimensions out in
//! full (28 bytes per member that are always zero) — see [`MEMBER_DIM_BLOCK`] —
//! which is what h5py's own `H5Tencode` output for a record dtype shows.
//!
//! Nested base types are written in their **natural** (unpadded) length by
//! [`write_nested_datatype_body`], because every reader advances by a nested
//! type's consumed length rather than by an 8-byte grid.

use oxih5_core::{ByteOrder, CompoundField, Dtype, OxiH5Error};

use super::elem::{dtype_to_elem_type, write_nested_datatype_body, ElemType, VLEN_REF_SIZE};
use super::format::{write_u16_le, write_u32_le};
use super::{check_size, narrow, pad8};

/// Bytes of obsolete per-member array description in a version-1 compound:
/// dimensionality (1), reserved (3), permutation (4), reserved (4), and four
/// 4-byte dimension sizes.  Always zero, always present.
const MEMBER_DIM_BLOCK: usize = 28;

/// Fixed prefix of every datatype message body: class/version, three bit-field
/// bytes, and the element size.
const DT_PREFIX: usize = 8;

/// Largest number of members a version-1 compound's 24-bit count can express.
const MAX_COMPOUND_MEMBERS: usize = (1 << 24) - 1;

/// Datatype message version an array (class 10) is encoded with.
///
/// The array class arrived *with* version 2 and libhdf5 rejects anything older
/// as "bad version number for datatype message"; the other classes here are all
/// version 1.
const ARRAY_DT_VERSION: u8 = 2;

/// Largest dimension count an array datatype's 32-bit field can express, capped
/// far below it at the dataspace rank limit so a bad request fails early.
const MAX_ARRAY_DIMS: usize = 32;

/// A datatype message body built from a structured description.
///
/// Carried by a dataset instead of (not beside) an [`ElemType`], so there is
/// never a question of which of the two describes the data — see
/// [`super::tree::DatasetDesc::elem_size`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedDtype {
    /// The datatype message body, ready to copy into an object header.
    pub(crate) body: Vec<u8>,
    /// On-disk size of one element, in bytes — the dataset's stride.
    pub(crate) elem_size: usize,
    /// What the type is, for diagnostics only.
    pub(crate) label: String,
    /// Set for a class-9 sequence: elements are 16-byte global-heap references
    /// and the payload lives in the global heap rather than in the data area.
    pub(crate) vlen_base: Option<ElemType>,
}

/// Class code and version, as the first byte of a datatype message body.
const fn class_version(class: u8) -> u8 {
    // Version 1 in the upper nibble, class in the lower one.
    (1 << 4) | class
}

/// Map a reader-side [`Dtype`] onto a writable member type.
///
/// Wider than [`dtype_to_elem_type`] by exactly one case — a fixed-length
/// string — because a record field is very often one, while a whole *dataset*
/// of fixed strings has its own entry point that names the width explicitly.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` for any dtype the writer cannot emit inline.
pub(crate) fn member_elem_type(dtype: &Dtype) -> Result<ElemType, OxiH5Error> {
    match dtype {
        Dtype::String {
            fixed_len: Some(width),
            ..
        } => {
            let width: u32 = narrow("fixed-length string member width", *width)?;
            if width == 0 {
                return Err(OxiH5Error::Format(
                    "a fixed-length string member must be at least one byte wide".to_string(),
                ));
            }
            Ok(ElemType::FixedStr(width))
        }
        other => dtype_to_elem_type(other),
    }
}

// ---------------------------------------------------------------------------
// Class 6 — compound
// ---------------------------------------------------------------------------

/// Encode a compound (record) datatype from its member list.
///
/// Members are written in the order given, each with the byte offset the caller
/// declared; the struct size is `struct_size`, which is the stride between
/// consecutive records on disk.  Nothing is re-packed: a caller that lays out
/// its rows with C padding declares the offsets that padding produces, and a
/// caller that packs them tightly declares those instead.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if there are no members, if a member name is
/// empty or contains a NUL, if a member's type is not writable inline, if two
/// members overlap or run past `struct_size`, or if the member count overflows
/// the 24-bit on-disk field.
pub(crate) fn compound(
    fields: &[CompoundField],
    struct_size: usize,
) -> Result<EncodedDtype, OxiH5Error> {
    if fields.is_empty() {
        return Err(OxiH5Error::Format(
            "a compound datatype must have at least one member".to_string(),
        ));
    }
    if fields.len() > MAX_COMPOUND_MEMBERS {
        return Err(OxiH5Error::Format(format!(
            "compound datatype has {} members, over the {MAX_COMPOUND_MEMBERS} a version-1 \
             message can count",
            fields.len()
        )));
    }
    if struct_size == 0 {
        return Err(OxiH5Error::Format(
            "a compound datatype must have a non-zero struct size".to_string(),
        ));
    }

    // Resolve every member type first, so a bad one is reported before any
    // bytes are laid out.
    let mut members: Vec<(&CompoundField, ElemType)> = Vec::with_capacity(fields.len());
    let mut occupied = vec![false; struct_size];
    for field in fields {
        if field.name.is_empty() {
            return Err(OxiH5Error::Format(
                "a compound member name must not be empty".to_string(),
            ));
        }
        if field.name.contains('\0') {
            return Err(OxiH5Error::Format(format!(
                "compound member name '{}' contains a NUL byte, which libhdf5 stores \
                 NUL-terminated and would silently truncate",
                field.name.escape_debug()
            )));
        }
        if fields.iter().filter(|f| f.name == field.name).count() > 1 {
            return Err(OxiH5Error::Format(format!(
                "compound datatype has two members named '{}'",
                field.name
            )));
        }
        let elem = member_elem_type(&field.dtype)
            .map_err(|e| OxiH5Error::Format(format!("compound member '{}': {e}", field.name)))?;
        let end = field
            .offset
            .checked_add(elem.byte_size())
            .ok_or_else(|| OxiH5Error::Format("compound member offset overflows".to_string()))?;
        if end > struct_size {
            return Err(OxiH5Error::Format(format!(
                "compound member '{}' occupies bytes {}..{end} of a {struct_size}-byte record",
                field.name, field.offset
            )));
        }
        for slot in &mut occupied[field.offset..end] {
            if *slot {
                return Err(OxiH5Error::Format(format!(
                    "compound member '{}' at offset {} overlaps another member",
                    field.name, field.offset
                )));
            }
            *slot = true;
        }
        members.push((field, elem));
    }

    // Size, then write: one formula, used twice, exactly as the object header
    // does it.
    let body_size = DT_PREFIX
        + members
            .iter()
            .map(|(field, elem)| {
                pad8(field.name.len() + 1) + 4 + MEMBER_DIM_BLOCK + elem.nested_dt_body_size()
            })
            .sum::<usize>();

    let mut body = vec![0u8; body_size];
    body[0] = class_version(6);
    // Member count is the low 24 bits of the class bit field.
    let count = members.len();
    body[1] = (count & 0xFF) as u8;
    body[2] = ((count >> 8) & 0xFF) as u8;
    body[3] = ((count >> 16) & 0xFF) as u8;
    write_u32_le(&mut body, 4, narrow("compound struct size", struct_size)?);

    let mut pos = DT_PREFIX;
    for (field, elem) in &members {
        // Version-1 member names are NUL-terminated and padded to 8 bytes.
        let name_span = pad8(field.name.len() + 1);
        body[pos..pos + field.name.len()].copy_from_slice(field.name.as_bytes());
        pos += name_span;
        write_u32_le(
            &mut body,
            pos,
            narrow("compound member offset", field.offset)?,
        );
        pos += 4;
        // The obsolete dimension block stays zero: dimensionality 0.
        pos += MEMBER_DIM_BLOCK;
        pos += write_nested_datatype_body(&mut body, pos, *elem)?;
    }
    check_size("compound datatype body", pos, body_size)?;

    let names: Vec<&str> = members.iter().map(|(f, _)| f.name.as_str()).collect();
    Ok(EncodedDtype {
        body,
        elem_size: struct_size,
        label: format!("compound {{{}}}", names.join(", ")),
        vlen_base: None,
    })
}

// ---------------------------------------------------------------------------
// Class 9 subtype 0 — variable-length sequence
// ---------------------------------------------------------------------------

/// Encode a variable-length **sequence** datatype over `base`.
///
/// This is the ragged-array type: every element of the dataset is a 16-byte
/// global-heap reference whose sequence length counts *elements of `base`*, not
/// bytes — verified against libhdf5, which writes `seq_len = 3` and a 12-byte
/// heap object for a three-element `int32` sequence.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `base` is itself variable-length, which
/// would need a second level of heap references this writer does not lay out.
pub(crate) fn vlen_sequence(base: ElemType) -> Result<EncodedDtype, OxiH5Error> {
    if matches!(base, ElemType::VlenStr) {
        return Err(OxiH5Error::Format(
            "a variable-length sequence of variable-length strings is not writable: its \
             elements would themselves be heap references"
                .to_string(),
        ));
    }

    let body_size = DT_PREFIX + base.nested_dt_body_size();
    let mut body = vec![0u8; body_size];
    body[0] = class_version(9);
    // Bit field: low nibble 0 = sequence (1 would be a string).  A sequence has
    // no padding or charset nibble, so bytes 1..4 stay zero.
    write_u32_le(&mut body, 4, narrow("vlen reference size", VLEN_REF_SIZE)?);
    let wrote = write_nested_datatype_body(&mut body, DT_PREFIX, base)?;
    check_size("vlen sequence datatype body", DT_PREFIX + wrote, body_size)?;

    Ok(EncodedDtype {
        body,
        elem_size: VLEN_REF_SIZE,
        label: format!("vlen sequence of {base:?}"),
        vlen_base: Some(base),
    })
}

// ---------------------------------------------------------------------------
// Class 10 — array
// ---------------------------------------------------------------------------

/// Encode an array datatype: a fixed-shape block of `base` per element.
///
/// Emitted as datatype message **version 2**, not version 1 like every other
/// class here.  That is not a style choice: the array class was introduced
/// *with* version 2, libhdf5 refuses a version-1 array outright ("bad version
/// number for datatype message"), and version 2 is what `H5T__array_encode`
/// produces — dimensionality byte, three reserved bytes, 4-byte dimension
/// sizes, then a permutation index per dimension.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `dims` is empty, holds a zero extent, is
/// deeper than [`MAX_ARRAY_DIMS`], or if the element size overflows.
pub(crate) fn array(base: ElemType, dims: &[usize]) -> Result<EncodedDtype, OxiH5Error> {
    if dims.is_empty() {
        return Err(OxiH5Error::Format(
            "an array datatype must have at least one dimension".to_string(),
        ));
    }
    if dims.len() > MAX_ARRAY_DIMS {
        return Err(OxiH5Error::Format(format!(
            "array datatype has {} dimensions, over the writer's limit of {MAX_ARRAY_DIMS}",
            dims.len()
        )));
    }
    let mut count = 1usize;
    for &dim in dims {
        if dim == 0 {
            return Err(OxiH5Error::Format(
                "an array datatype dimension must not be zero".to_string(),
            ));
        }
        count = count
            .checked_mul(dim)
            .ok_or_else(|| OxiH5Error::Format("array datatype extent overflows".to_string()))?;
    }
    let elem_size = count
        .checked_mul(base.byte_size())
        .ok_or_else(|| OxiH5Error::Format("array datatype byte size overflows".to_string()))?;

    // Version 2: dimensionality (1) + reserved (3) + dim sizes (4 each) + a
    // permutation index per dimension (4 each), then the inline base type.
    let dims_start = DT_PREFIX + 4;
    let perm_end = dims_start + dims.len() * 8;
    let body_size = perm_end + base.nested_dt_body_size();

    let mut body = vec![0u8; body_size];
    body[0] = (ARRAY_DT_VERSION << 4) | 10;
    write_u32_le(&mut body, 4, narrow("array element size", elem_size)?);
    body[DT_PREFIX] = narrow::<u8>("array dimensionality", dims.len())?;
    // [9..12] reserved.
    for (i, &dim) in dims.iter().enumerate() {
        write_u32_le(&mut body, dims_start + i * 4, narrow("array extent", dim)?);
        // The permutation is the identity: libhdf5 has never used anything else
        // and every reader ignores it, but the field must be present in v2.
        write_u32_le(
            &mut body,
            dims_start + dims.len() * 4 + i * 4,
            narrow("array permutation index", i)?,
        );
    }
    let wrote = write_nested_datatype_body(&mut body, perm_end, base)?;
    check_size("array datatype body", perm_end + wrote, body_size)?;

    Ok(EncodedDtype {
        body,
        elem_size,
        label: format!("array {dims:?} of {base:?}"),
        vlen_base: None,
    })
}

// ---------------------------------------------------------------------------
// Class 5 — opaque
// ---------------------------------------------------------------------------

/// Encode an opaque datatype: `size` uninterpreted bytes carrying `tag`.
///
/// The tag is an ASCII label a reader may use to recognise the bytes (h5py
/// writes `NUMPY:|V7` for a `void7` dtype); it is stored NUL-padded to an
/// 8-byte boundary and its *padded* length goes in the bit field, which is how
/// `oxih5_format::datatype` recovers the properties section's width.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `size` is zero, if the tag contains a NUL,
/// or if the padded tag is longer than the 8-bit field that counts it.
pub(crate) fn opaque(size: usize, tag: &str) -> Result<EncodedDtype, OxiH5Error> {
    if size == 0 {
        return Err(OxiH5Error::Format(
            "an opaque datatype must be at least one byte wide".to_string(),
        ));
    }
    if tag.contains('\0') {
        return Err(OxiH5Error::Format(
            "an opaque datatype tag must not contain a NUL byte".to_string(),
        ));
    }
    // The stored tag is NUL-padded to 8 bytes; an exact multiple still gets a
    // terminator, matching libhdf5 (`H5T__opaque_encode`).
    let tag_span = pad8(tag.len() + 1);
    let tag_len: u8 = u8::try_from(tag_span).map_err(|_| {
        OxiH5Error::Format(format!(
            "opaque datatype tag is {} bytes, over the 255-byte on-disk limit",
            tag.len()
        ))
    })?;

    let body_size = DT_PREFIX + tag_span;
    let mut body = vec![0u8; body_size];
    body[0] = class_version(5);
    body[1] = tag_len;
    write_u32_le(&mut body, 4, narrow("opaque element size", size)?);
    body[DT_PREFIX..DT_PREFIX + tag.len()].copy_from_slice(tag.as_bytes());

    Ok(EncodedDtype {
        body,
        elem_size: size,
        label: format!("opaque {size}B '{tag}'"),
        vlen_base: None,
    })
}

// ---------------------------------------------------------------------------
// Class 4 — bitfield
// ---------------------------------------------------------------------------

/// Encode a bitfield datatype: `size` bytes of flags, `size * 8` bits precise.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `size` is zero or wider than eight bytes,
/// the range whose precision fits the 16-bit field.
pub(crate) fn bitfield(size: usize, order: ByteOrder) -> Result<EncodedDtype, OxiH5Error> {
    if size == 0 || size > 8 {
        return Err(OxiH5Error::Format(format!(
            "a bitfield datatype must be 1..=8 bytes wide, not {size}"
        )));
    }

    // Version 1: 8-byte header plus a 4-byte properties section holding the bit
    // offset and the bit precision — the same shape as class 0.
    let body_size = DT_PREFIX + 4;
    let mut body = vec![0u8; body_size];
    body[0] = class_version(4);
    body[1] = match order {
        ByteOrder::Little => 0x00,
        ByteOrder::Big => 0x01,
    };
    write_u32_le(&mut body, 4, narrow("bitfield element size", size)?);
    // [8..10] bit offset = 0.
    write_u16_le(&mut body, 10, (size * 8) as u16);

    Ok(EncodedDtype {
        body,
        elem_size: size,
        label: format!("bitfield {size}B {order}"),
        vlen_base: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_core::Charset;
    use oxih5_format::datatype::parse_datatype;

    fn field(name: &str, offset: usize, dtype: Dtype) -> CompoundField {
        CompoundField {
            name: name.to_string(),
            offset,
            dtype,
        }
    }

    fn i32_dtype() -> Dtype {
        Dtype::Int {
            size: 4,
            signed: true,
            order: ByteOrder::Little,
        }
    }

    fn f64_dtype() -> Dtype {
        Dtype::Float {
            size: 8,
            order: ByteOrder::Little,
        }
    }

    /// Byte-pinned against the compound datatype message h5py 3.16 / libhdf5
    /// 2.0.0 wrote for `numpy.dtype([('a', '<i4'), ('b', '<f8')])`.
    #[test]
    fn compound_body_matches_libhdf5_byte_for_byte() {
        let encoded = compound(
            &[field("a", 0, i32_dtype()), field("b", 4, f64_dtype())],
            12,
        )
        .expect("encode");

        let mut want: Vec<u8> = vec![0x16, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00];
        // Member "a": name padded to 8, offset 0, 28 zero bytes, i32 base.
        want.extend_from_slice(b"a\0\0\0\0\0\0\0");
        want.extend_from_slice(&0u32.to_le_bytes());
        want.extend_from_slice(&[0u8; MEMBER_DIM_BLOCK]);
        want.extend_from_slice(&[
            0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0, 0, 0x20, 0x00,
        ]);
        // Member "b": name padded to 8, offset 4, 28 zero bytes, f64 base.
        want.extend_from_slice(b"b\0\0\0\0\0\0\0");
        want.extend_from_slice(&4u32.to_le_bytes());
        want.extend_from_slice(&[0u8; MEMBER_DIM_BLOCK]);
        want.extend_from_slice(&[
            0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b,
            0x00, 0x34, 0xff, 0x03, 0x00, 0x00,
        ]);
        assert_eq!(encoded.body, want);
        assert_eq!(encoded.elem_size, 12);
    }

    /// Anything this module writes must decode back through the reader that
    /// parses datatype messages, with the same members, offsets and types.
    #[test]
    fn compound_round_trips_through_the_reader() {
        let encoded = compound(
            &[
                field("id", 0, i32_dtype()),
                field("value", 4, f64_dtype()),
                field(
                    "label",
                    12,
                    Dtype::String {
                        fixed_len: Some(8),
                        charset: Charset::Ascii,
                    },
                ),
            ],
            20,
        )
        .expect("encode");

        let dtype = parse_datatype(&encoded.body).expect("parse");
        let Dtype::Compound { fields } = &dtype else {
            panic!("expected a compound, got {dtype:?}");
        };
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "id");
        assert_eq!(fields[0].offset, 0);
        assert_eq!(fields[0].dtype, i32_dtype());
        assert_eq!(fields[1].name, "value");
        assert_eq!(fields[1].offset, 4);
        assert_eq!(fields[1].dtype, f64_dtype());
        assert_eq!(fields[2].name, "label");
        assert_eq!(fields[2].offset, 12);
        assert!(matches!(
            fields[2].dtype,
            Dtype::String {
                fixed_len: Some(8),
                ..
            }
        ));
        assert_eq!(dtype.size(), Some(20));
    }

    /// A record laid out with padding declares the offsets that padding
    /// produces; the encoder must not silently re-pack them.
    #[test]
    fn compound_honours_declared_offsets_including_padding() {
        let encoded = compound(
            &[field("a", 0, i32_dtype()), field("b", 8, f64_dtype())],
            16,
        )
        .expect("encode");
        let Dtype::Compound { fields } = parse_datatype(&encoded.body).expect("parse") else {
            panic!("expected a compound");
        };
        assert_eq!(fields[1].offset, 8, "the 4-byte hole is preserved");
    }

    #[test]
    fn compound_rejects_malformed_member_lists() {
        let overlap = compound(
            &[field("a", 0, f64_dtype()), field("b", 4, i32_dtype())],
            12,
        )
        .expect_err("overlap");
        assert!(format!("{overlap}").contains("overlaps"), "{overlap}");

        let over = compound(&[field("a", 8, f64_dtype())], 12).expect_err("past the end");
        assert!(format!("{over}").contains("8..16"), "{over}");

        let dup = compound(&[field("a", 0, i32_dtype()), field("a", 4, i32_dtype())], 8)
            .expect_err("duplicate");
        assert!(format!("{dup}").contains("two members named"), "{dup}");

        let empty_name = compound(&[field("", 0, i32_dtype())], 4).expect_err("empty member name");
        assert!(format!("{empty_name}").contains("must not be empty"));

        assert!(compound(&[], 4).is_err(), "no members");
        assert!(
            compound(&[field("a", 0, i32_dtype())], 0).is_err(),
            "no size"
        );

        let bad_type = compound(
            &[field(
                "a",
                0,
                Dtype::Reference {
                    ref_type: oxih5_core::RefType::Object,
                },
            )],
            8,
        )
        .expect_err("unwritable member type");
        assert!(
            format!("{bad_type}").contains("compound member 'a'"),
            "{bad_type}"
        );
    }

    /// Byte-pinned against libhdf5's `h5py.vlen_dtype(numpy.int32)`.
    #[test]
    fn vlen_sequence_body_matches_libhdf5() {
        let encoded = vlen_sequence(ElemType::I32).expect("encode");
        assert_eq!(
            encoded.body,
            vec![
                0x19, 0x00, 0x00, 0x00, // class 9 v1, sequence
                0x10, 0x00, 0x00, 0x00, // 16-byte on-disk reference
                0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
                0x00, // i32 base
            ]
        );
        assert_eq!(encoded.elem_size, 16);
        assert_eq!(encoded.vlen_base, Some(ElemType::I32));

        let dtype = parse_datatype(&encoded.body).expect("parse");
        assert_eq!(
            dtype,
            Dtype::VarLen {
                base: Box::new(i32_dtype())
            }
        );
    }

    #[test]
    fn vlen_sequence_covers_every_fixed_base_and_refuses_vlen_ones() {
        for base in [ElemType::F64, ElemType::U8, ElemType::I16, ElemType::F16] {
            let encoded = vlen_sequence(base).expect("encode");
            let Dtype::VarLen { base: parsed } = parse_datatype(&encoded.body).expect("parse")
            else {
                panic!("expected a vlen sequence for {base:?}");
            };
            assert_eq!(parsed.size(), Some(base.byte_size()));
        }
        let err = vlen_sequence(ElemType::VlenStr).expect_err("nested vlen");
        assert!(format!("{err}").contains("heap references"), "{err}");
    }

    /// Byte-pinned against `h5t.array_create(h5t.NATIVE_INT32, (2, 3))` as
    /// **h5py 3.16 / libhdf5 2.0.0** encodes it — including the version-2
    /// class byte, without which libhdf5 refuses the file outright.
    #[test]
    fn array_body_matches_libhdf5_byte_for_byte() {
        let encoded = array(ElemType::I32, &[2, 3]).expect("encode");
        assert_eq!(
            encoded.body,
            vec![
                0x2a, 0x00, 0x00, 0x00, // class 10 (array), version 2
                0x18, 0x00, 0x00, 0x00, // element size = 24
                0x02, 0x00, 0x00, 0x00, // dimensionality = 2, then 3 reserved
                0x02, 0x00, 0x00, 0x00, // dim[0]
                0x03, 0x00, 0x00, 0x00, // dim[1]
                0x00, 0x00, 0x00, 0x00, // permutation[0]
                0x01, 0x00, 0x00, 0x00, // permutation[1]
                0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00,
            ]
        );
        assert_eq!(encoded.elem_size, 2 * 3 * 4);

        let dtype = parse_datatype(&encoded.body).expect("parse");
        let Dtype::Array { base, dims } = &dtype else {
            panic!("expected an array, got {dtype:?}");
        };
        assert_eq!(dims, &vec![2, 3]);
        assert_eq!(**base, i32_dtype());
        assert_eq!(dtype.size(), Some(24));
    }

    #[test]
    fn array_rejects_degenerate_shapes() {
        assert!(array(ElemType::I32, &[]).is_err(), "no dimensions");
        assert!(array(ElemType::I32, &[2, 0]).is_err(), "zero extent");
        let deep: Vec<usize> = vec![1; MAX_ARRAY_DIMS + 1];
        assert!(array(ElemType::I32, &deep).is_err(), "too deep");
    }

    /// Byte-pinned against libhdf5's `h5py.opaque_dtype(numpy.dtype('V7'))`,
    /// whose tag is `NUMPY:|V7` NUL-padded to 16 bytes.
    #[test]
    fn opaque_body_matches_libhdf5() {
        let encoded = opaque(7, "NUMPY:|V7").expect("encode");
        let mut want: Vec<u8> = vec![0x15, 0x10, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00];
        want.extend_from_slice(b"NUMPY:|V7\0\0\0\0\0\0\0");
        assert_eq!(encoded.body, want);
        assert_eq!(encoded.elem_size, 7);

        assert_eq!(
            parse_datatype(&encoded.body).expect("parse"),
            Dtype::Opaque {
                size: 7,
                tag: "NUMPY:|V7".to_string()
            }
        );
    }

    /// An exactly-8-byte tag still gets a terminator, so its span is 16 — the
    /// case where a naive `pad8(len)` would drop the NUL and merge the tag with
    /// whatever follows.
    #[test]
    fn an_eight_byte_opaque_tag_still_terminates() {
        let encoded = opaque(4, "12345678").expect("encode");
        assert_eq!(encoded.body[1], 16, "padded tag span");
        assert_eq!(encoded.body.len(), DT_PREFIX + 16);
        assert_eq!(
            parse_datatype(&encoded.body).expect("parse"),
            Dtype::Opaque {
                size: 4,
                tag: "12345678".to_string()
            }
        );
        assert!(opaque(0, "x").is_err(), "zero width");
        assert!(opaque(4, "a\0b").is_err(), "NUL in tag");
    }

    /// A bitfield keeps its width and byte order through the round trip.
    #[test]
    fn bitfield_round_trips_in_both_byte_orders() {
        for (size, order) in [
            (1usize, ByteOrder::Little),
            (4, ByteOrder::Little),
            (8, ByteOrder::Big),
        ] {
            let encoded = bitfield(size, order).expect("encode");
            assert_eq!(encoded.elem_size, size);
            assert_eq!(
                parse_datatype(&encoded.body).expect("parse"),
                Dtype::Bitfield { size, order }
            );
            assert_eq!(
                u16::from_le_bytes([encoded.body[10], encoded.body[11]]) as usize,
                size * 8,
                "bit precision"
            );
        }
        assert!(bitfield(0, ByteOrder::Little).is_err());
        assert!(bitfield(9, ByteOrder::Little).is_err());
    }

    /// A nested base type must be written in its natural length: the four bytes
    /// of padding a top-level class-0 message carries would shift every member
    /// after it.
    #[test]
    fn nested_base_types_are_unpadded() {
        assert_eq!(ElemType::I32.dt_body_size(), 16);
        assert_eq!(ElemType::I32.nested_dt_body_size(), 12);
        assert_eq!(ElemType::F64.dt_body_size(), 24);
        assert_eq!(ElemType::F64.nested_dt_body_size(), 20);

        // Two i32 members: if the base were padded, the second member's name
        // would land four bytes late and the reader would see garbage.
        let encoded =
            compound(&[field("a", 0, i32_dtype()), field("b", 4, i32_dtype())], 8).expect("encode");
        let Dtype::Compound { fields } = parse_datatype(&encoded.body).expect("parse") else {
            panic!("expected a compound");
        };
        assert_eq!(fields[1].name, "b");
    }
}
