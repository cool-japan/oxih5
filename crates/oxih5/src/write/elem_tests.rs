//! Unit tests for [`super`] — the element-type and attribute-type
//! machinery.  Split out of `elem.rs` to keep that file under the
//! 2000-line limit; wired back in via `#[path = "elem_tests.rs"] mod tests`.

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
        0x11, 0x20, 0x1f, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x17, 0x08, 0x00,
        0x17, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let f64_literal: [u8; 24] = [
        0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b, 0x00,
        0x34, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let i32_literal: [u8; 16] = [
        0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];
    let i64_literal: [u8; 16] = [
        0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];
    let u8_literal: [u8; 16] = [
        0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];
    // Byte 2 = 0x01 is the outer class-9 UTF-8 charset (fixed in B017); the
    // nested base type at [8..] stays `13 10` (class-3, NULLTERM, UTF-8).
    let vlen_literal: [u8; 16] = [
        0x19, 0x01, 0x01, 0x00, 0x10, 0x00, 0x00, 0x00, 0x13, 0x10, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00,
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
            | ElemType::VlenStr
            | ElemType::FixedStr(_)
            | ElemType::Bool => {}
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
    let expected: [(ElemType, Dtype); 13] = [
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
        // A fixed-length string dataset reads back as a fixed-length ASCII
        // string of its width — the numpy `S<width>` shape.
        (
            ElemType::FixedStr(4),
            Dtype::String {
                fixed_len: Some(4),
                charset: oxih5_core::Charset::Ascii,
            },
        ),
        // A boolean dataset reads back as the enum h5py stores it as.
        (
            ElemType::Bool,
            Dtype::Enum {
                base: Box::new(int(1, true)),
                members: vec![("FALSE".to_string(), 0), ("TRUE".to_string(), 1)],
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

        // VlenStr, FixedStr and Bool datasets are created through dedicated
        // entry points (`create_vlen_string_dataset`,
        // `create_fixed_string_dataset`, `write_dataset_bool`), never through a
        // `Dtype`, so they are deliberately outside the `dtype_to_elem_type`
        // domain and only the forward direction is checked for them.
        if !matches!(
            elem_type,
            ElemType::VlenStr | ElemType::FixedStr(_) | ElemType::Bool
        ) {
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
                0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b,
                0x00, 0x34, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
        ),
        (
            ResolvedAttrKind::I64(0),
            &[
                0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ],
        ),
        (
            ResolvedAttrKind::I32(0),
            &[
                0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ],
        ),
        (
            // Byte 1 = 0x11: NULLPAD (bits 0..4, fixed in B022) + UTF-8
            // charset (bits 4..8). libhdf5 uses NULLPAD for a fixed string
            // whose declared size equals its content length.
            ResolvedAttrKind::FixedStr("abc"),
            &[0x13, 0x11, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00],
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

// -----------------------------------------------------------------------
// Fix-lane ELEM regression tests (B003, B017, B022, R004)
// -----------------------------------------------------------------------

/// Render an attribute's inline datatype body into a fresh buffer.
fn attr_dtype_body(kind: ResolvedAttrKind<'_>) -> Vec<u8> {
    let attr = ResolvedAttr { name: "a", kind };
    let mut buf = vec![0u8; 32];
    let wrote = attr.write_dtype_body(&mut buf, 0).expect("dtype body");
    buf.truncate(wrote);
    buf
}

/// B003: an empty scalar fixed string must never emit a size-0 string
/// datatype — libhdf5 then cannot iterate ANY attribute on the object, which
/// poisons every sibling attribute too.  The datatype size clamps to 1.
#[test]
fn empty_scalar_fixed_string_datatype_size_is_clamped_to_one() {
    assert_eq!(fixed_str_width(""), 1, "empty string clamps to one byte");
    assert_eq!(fixed_str_width("meters"), 6, "non-empty is unchanged");

    let body = attr_dtype_body(ResolvedAttrKind::FixedStr(""));
    assert_eq!(body[0], 0x13, "class 3 string, version 1");
    let size = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
    assert_eq!(size, 1, "empty-string datatype size must be 1, never 0");
}

/// B003: reserve ([`ResolvedAttrKind::sizes`]) and write
/// ([`ResolvedAttr::write_body`]) must still agree for an empty scalar
/// string now that both are clamped to one byte.
#[test]
fn empty_scalar_fixed_string_reserve_equals_write() {
    let attr = ResolvedAttr {
        name: "empty",
        kind: ResolvedAttrKind::FixedStr(""),
    };
    assert_eq!(attr.kind.sizes().data, 1, "one NUL byte reserved");
    let expected = attr.body_size();
    let mut buf = vec![0xAAu8; expected + 64];
    assert_eq!(attr.write_body(&mut buf, 0).expect("write_body"), expected);
    // The single data byte is the clamped NUL that stands in for "".
    assert_eq!(buf[expected - 1], 0x00, "the clamped byte is a NUL");
}

/// B017: the outer class-9 vlen-string charset nibble must declare UTF-8,
/// and the reader still recovers UTF-8 (it reads the nested base type).
#[test]
fn vlen_string_datatype_declares_utf8_charset() {
    let body = body_of(ElemType::VlenStr);
    assert_eq!(body[0], 0x19, "class 9 vlen, version 1");
    assert_eq!(body[2] & 0x0F, 0x01, "outer charset nibble must be UTF-8");
    let got = oxih5_format::datatype::parse_datatype(&body).expect("parse");
    assert_eq!(
        got,
        Dtype::String {
            fixed_len: None,
            charset: oxih5_core::Charset::Utf8,
        },
    );
}

/// B022: fixed-length string datatypes use NULLPAD — libhdf5's convention
/// for a string whose declared size equals its content length, where
/// NULLTERM would promise a terminator that has no room to exist.
#[test]
fn fixed_string_datatypes_use_nullpad() {
    assert_eq!(StrPad::NullTerm.nibble(), 0x00);
    assert_eq!(StrPad::NullPad.nibble(), 0x01);

    let strings = vec!["a".to_string(), "bb".to_string()];
    let cases = [
        attr_dtype_body(ResolvedAttrKind::FixedStr("meters")),
        attr_dtype_body(ResolvedAttrKind::StrArray {
            values: &strings,
            width: str_array_width(&strings),
        }),
    ];
    for body in cases {
        assert_eq!(body[0], 0x13, "class 3 string");
        assert_eq!(body[1] & 0x0F, 0x01, "padding nibble must be NULLPAD");
        assert_eq!(body[1] >> 4, 0x01, "charset nibble must be UTF-8");
    }
}

/// R004: an empty attribute name is rejected at the write seam every
/// attribute flows through; a non-empty name still writes cleanly.
#[test]
fn empty_attribute_name_is_rejected() {
    let attr = ResolvedAttr {
        name: "",
        kind: ResolvedAttrKind::I32(7),
    };
    let mut buf = vec![0u8; attr.body_size() + 64];
    let err = attr
        .write_body(&mut buf, 0)
        .expect_err("empty name must fail");
    assert!(
        err.to_string().contains("empty"),
        "error must explain the cause: {err}"
    );

    let ok = ResolvedAttr {
        name: "n",
        kind: ResolvedAttrKind::I32(7),
    };
    let expected = ok.body_size();
    let mut ok_buf = vec![0u8; expected + 64];
    assert_eq!(ok.write_body(&mut ok_buf, 0).expect("write_body"), expected);
}

// -----------------------------------------------------------------------
// Wave 2 ATTRCORE: widened numerics (G006), vlen-objref (B005),
// ref-index-list compound (B018)
// -----------------------------------------------------------------------

/// The vlen-of-object-reference datatype body is byte-identical to the
/// `DIMENSION_LIST` type netCDF-4 writes (captured from `h5py`'s
/// `H5Tencode`): `H5T_VLEN{ H5T_REFERENCE(object) }`.
#[test]
fn vlen_ref_datatype_matches_netcdf_dimension_list() {
    let mut buf = vec![0u8; VLEN_REF_DT_BODY];
    assert_eq!(write_vlen_ref_dtype(&mut buf, 0), VLEN_REF_DT_BODY);
    assert_eq!(
        buf,
        vec![
            0x19, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x08, 0x00,
            0x00, 0x00,
        ],
    );
    // The reader must accept it: a class-9 vlen sequence over a class-7
    // object reference decodes to a VarLen of a Reference.
    let got = oxih5_format::datatype::parse_datatype(&buf).expect("reader parses vlen-ref");
    assert!(
        matches!(
            &got,
            Dtype::VarLen { base } if matches!(**base, Dtype::Reference { .. })
        ),
        "expected VarLen{{Reference}}, got {got:?}"
    );
}

/// The `REFERENCE_LIST` compound datatype body is byte-identical to what
/// netCDF-4 writes (member names `dataset`/`dimension`, offsets 0/8, struct
/// size 12), captured from `h5py`'s version-1 `H5Tencode`.
#[test]
fn ref_index_datatype_matches_netcdf_reference_list() {
    let mut buf = vec![0u8; REF_INDEX_DT_BODY];
    assert_eq!(write_ref_index_dtype(&mut buf, 0), REF_INDEX_DT_BODY);
    // Header: class 6, version 1, 2 members, struct size 12.
    assert_eq!(buf[0], 0x16, "class 6 compound, version 1");
    assert_eq!(
        (buf[1] as usize) | ((buf[2] as usize) << 8) | ((buf[3] as usize) << 16),
        2,
        "two members"
    );
    assert_eq!(
        u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        12,
        "struct size 12"
    );
    // The reader must decode both named members at their offsets.
    let got = oxih5_format::datatype::parse_datatype(&buf).expect("reader parses compound");
    match got {
        Dtype::Compound { fields } => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "dataset");
            assert_eq!(fields[0].offset, 0);
            assert!(matches!(fields[0].dtype, Dtype::Reference { .. }));
            assert_eq!(fields[1].name, "dimension");
            assert_eq!(fields[1].offset, 8);
            assert!(matches!(fields[1].dtype, Dtype::Int { size: 4, .. }));
        }
        other => panic!("expected Compound, got {other:?}"),
    }
    // `Dtype::size()` (the read-side stride) equals the on-disk element size.
    assert_eq!(
        oxih5_format::datatype::parse_datatype(&buf)
            .expect("parse")
            .size(),
        Some(REF_INDEX_ELEM),
    );
}

/// Every widened numeric scalar reuses the shared `ElemType` encoder, so its
/// attribute datatype body equals the dataset datatype body for that type.
#[test]
fn widened_numeric_attr_datatypes_reuse_the_elem_encoder() {
    let empty: Vec<u8> = Vec::new();
    for elem in [
        ElemType::F32,
        ElemType::I8,
        ElemType::I16,
        ElemType::U8,
        ElemType::U16,
        ElemType::U32,
        ElemType::U64,
    ] {
        let want = body_of(elem);
        for kind in [
            ResolvedAttrKind::Num {
                elem,
                le_bytes: &empty,
            },
            ResolvedAttrKind::NumArray {
                elem,
                le_bytes: &empty,
            },
        ] {
            let got = attr_dtype_body(kind);
            assert_eq!(got, want, "{elem:?}: attr datatype must match dataset");
        }
    }
}

/// A scalar numeric attribute gets a scalar dataspace; the array form gets a
/// 1-D one, and both reserve exactly what they write.
#[test]
fn widened_numeric_scalar_and_array_reserve_equals_write() {
    let u16le: Vec<u8> = [7u16, 8, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
    let scalar = ResolvedAttr {
        name: "flag",
        kind: ResolvedAttrKind::Num {
            elem: ElemType::U16,
            le_bytes: &u16le[..2],
        },
    };
    let array = ResolvedAttr {
        name: "flags",
        kind: ResolvedAttrKind::NumArray {
            elem: ElemType::U16,
            le_bytes: &u16le,
        },
    };
    assert_eq!(scalar.kind.vector_len(), None, "scalar dataspace");
    assert_eq!(array.kind.vector_len(), Some(3), "1-D of three");
    for attr in [&scalar, &array] {
        let expected = attr.body_size();
        let mut buf = vec![0xAAu8; expected + 64];
        assert_eq!(attr.write_body(&mut buf, 0).expect("write_body"), expected);
    }
}

/// A `REFERENCE_LIST` compound attribute reserves and writes 12 bytes per
/// entry — an 8-byte resolved address followed by the 4-byte index.
#[test]
fn ref_index_list_reserve_equals_write_and_lays_out_entries() {
    let pairs = vec![("v0".to_string(), 0u32), ("v1".to_string(), 1u32)];
    let attr = ResolvedAttr {
        name: "REFERENCE_LIST",
        kind: ResolvedAttrKind::RefIndexList {
            pairs: &pairs,
            addrs: vec![0x0123, 0x4567],
        },
    };
    assert_eq!(attr.kind.vector_len(), Some(2));
    let total = attr.body_size();
    let mut buf = vec![0u8; total];
    assert_eq!(attr.write_body(&mut buf, 0).expect("write_body"), total);
    // The data is the trailing `n * 12` bytes: [addr u64 | index u32] each.
    let data = &buf[total - 2 * REF_INDEX_ELEM..];
    assert_eq!(
        u64::from_le_bytes(data[0..8].try_into().expect("8")),
        0x0123
    );
    assert_eq!(u32::from_le_bytes(data[8..12].try_into().expect("4")), 0);
    assert_eq!(
        u64::from_le_bytes(data[12..20].try_into().expect("8")),
        0x4567
    );
    assert_eq!(u32::from_le_bytes(data[20..24].try_into().expect("4")), 1);
}

/// An unplaced non-empty vlen-object-reference sequence is a hard build
/// error, never a dangling reference into heap address 0.
#[test]
fn vlen_objref_without_heap_placement_is_reported() {
    let descs = vec![AttrDesc {
        name: "DIMENSION_LIST".to_string(),
        kind: AttrKind::VlenObjRefsByName(vec![vec!["lat".to_string()]]),
    }];
    let mut resolved = resolve_attrs(&descs);
    let mut map: HashMap<String, u64> = HashMap::new();
    map.insert("lat".to_string(), 0x200);
    fill_obj_refs(&mut resolved, &map).expect("addresses resolve");
    // Deliberately skip register/fill: the payload was never placed.
    let mut buf = vec![0u8; resolved[0].body_size() + 64];
    let err = resolved[0]
        .write_body(&mut buf, 0)
        .expect_err("must refuse an unplaced vlen reference");
    assert!(
        err.to_string().contains("never placed"),
        "error must explain the cause: {err}"
    );
}

/// The full elem-level heap pipeline: resolve addresses, register each
/// sequence's payload in the global heap, lay the collections out, fill the
/// locations, then serialise.  The inline data is one 16-byte on-disk vlen
/// reference per element and the heap object is the concatenated 8-byte
/// addresses — the layout netCDF-C dereferences.
#[test]
fn vlen_objref_heap_round_trip_lays_out_references() {
    let descs = vec![AttrDesc {
        name: "DIMENSION_LIST".to_string(),
        kind: AttrKind::VlenObjRefsByName(vec![
            vec!["lat".to_string()],
            vec!["lon".to_string()],
            Vec::new(), // an empty sequence needs no heap object
        ]),
    }];
    let mut resolved = resolve_attrs(&descs);
    let mut map: HashMap<String, u64> = HashMap::new();
    map.insert("lat".to_string(), 0x0000_0000_0000_0111);
    map.insert("lon".to_string(), 0x0000_0000_0000_0222);
    fill_obj_refs(&mut resolved, &map).expect("addresses resolve");

    let mut gcol = GlobalHeapWriter::new();
    register_vlen_objref_heap(&mut resolved, &mut gcol);
    let (collections, locations) = gcol.build_collections();
    assert_eq!(
        locations.len(),
        2,
        "two non-empty sequences, one object each"
    );
    assert_eq!(collections.len(), 1);
    let heap_base: u64 = 0x1000;
    let collection_addrs = vec![heap_base];
    fill_vlen_objref_locs(&mut resolved, &collection_addrs, &locations);

    let total = resolved[0].body_size();
    let mut buf = vec![0u8; total];
    assert_eq!(
        resolved[0].write_body(&mut buf, 0).expect("write_body"),
        total
    );

    // Inline data: the trailing 3 * 16 bytes.
    let data = &buf[total - 3 * VLEN_REF_SIZE..];
    let read_ref = |i: usize| {
        let b = &data[i * VLEN_REF_SIZE..(i + 1) * VLEN_REF_SIZE];
        (
            u32::from_le_bytes(b[0..4].try_into().expect("4")),
            u64::from_le_bytes(b[4..12].try_into().expect("8")),
            u32::from_le_bytes(b[12..16].try_into().expect("4")),
        )
    };
    // Element 0 → lat: seq_len 1, at heap_base, index 1.
    assert_eq!(read_ref(0), (1, heap_base, 1));
    // Element 1 → lon: seq_len 1, at heap_base, index 2.
    assert_eq!(read_ref(1), (1, heap_base, 2));
    // Element 2 → empty: the all-zero null vlen reference.
    assert_eq!(read_ref(2), (0, 0, 0));

    // The heap objects hold the raw 8-byte target addresses.
    assert_eq!(
        gcol_object(&collections[0], 1),
        0x0000_0000_0000_0111u64.to_le_bytes()
    );
    assert_eq!(
        gcol_object(&collections[0], 2),
        0x0000_0000_0000_0222u64.to_le_bytes()
    );
}

// -----------------------------------------------------------------------
// Wave 2 DSFEAT: fixed-length string (G003) and boolean enum (G009) datasets
// -----------------------------------------------------------------------

/// A fixed-length string dataset's datatype body is byte-identical to what
/// h5py writes for a numpy `S<width>` type (captured from `h5py`'s
/// `H5Tencode`, minus its two-byte wrapper): class 3, version 1, NULLPAD,
/// **ASCII** charset, and the width in the element-size field.
#[test]
fn fixed_string_dataset_datatype_matches_h5py_s_width() {
    // h5py `np.dtype('S4')` -> `13 01 00 00 04 00 00 00`.
    assert_eq!(
        body_of(ElemType::FixedStr(4)),
        [0x13, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00],
        "S4"
    );
    // h5py `np.dtype('S10')` -> `13 01 00 00 0a 00 00 00`.
    assert_eq!(
        body_of(ElemType::FixedStr(10)),
        [0x13, 0x01, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00],
        "S10"
    );
    // The charset nibble must be ASCII (0), unlike the UTF-8 attribute strings.
    let body = body_of(ElemType::FixedStr(4));
    assert_eq!(body[1] >> 4, 0x00, "charset nibble must be ASCII");
    assert_eq!(body[1] & 0x0F, 0x01, "padding nibble must be NULLPAD");
    // The reader recovers the width and the ASCII charset.
    assert_eq!(
        oxih5_format::datatype::parse_datatype(&body).expect("parse"),
        Dtype::String {
            fixed_len: Some(4),
            charset: oxih5_core::Charset::Ascii,
        },
    );
}

/// The boolean dataset's datatype body is byte-identical to what h5py writes
/// for a numpy `bool` array: a class-8 enumeration over `i8` with members
/// `FALSE = 0` and `TRUE = 1`.
#[test]
fn bool_dataset_datatype_matches_h5py_bool_enum() {
    let body = body_of(ElemType::Bool);
    assert_eq!(
        body,
        vec![
            0x18, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // enum v1, 2 members, base size 1
            0x10, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
            0x00, // i8 base type
            0x46, 0x41, 0x4c, 0x53, 0x45, 0x00, 0x00, 0x00, // "FALSE\0\0\0"
            0x54, 0x52, 0x55, 0x45, 0x00, 0x00, 0x00, 0x00, // "TRUE\0\0\0\0"
            0x00, 0x01, // values FALSE=0, TRUE=1
        ],
    );
    // The reader decodes it to exactly the enum h5py reads back as bool.
    let le = ByteOrder::Little;
    assert_eq!(
        oxih5_format::datatype::parse_datatype(&body).expect("parse"),
        Dtype::Enum {
            base: Box::new(Dtype::Int {
                size: 1,
                signed: true,
                order: le,
            }),
            members: vec![("FALSE".to_string(), 0), ("TRUE".to_string(), 1)],
        },
    );
    assert_eq!(ElemType::Bool.byte_size(), 1, "one i8 per element");
}

/// Extract the body of 1-based heap object `want` from a serialised GCOL
/// collection, for the vlen-heap round-trip test.
fn gcol_object(collection: &[u8], want: u16) -> Vec<u8> {
    assert_eq!(&collection[0..4], b"GCOL", "signature");
    let mut pos = 16usize;
    while pos + 16 <= collection.len() {
        let idx = u16::from_le_bytes(collection[pos..pos + 2].try_into().expect("2"));
        let size =
            u64::from_le_bytes(collection[pos + 8..pos + 16].try_into().expect("8")) as usize;
        if idx == want {
            return collection[pos + 16..pos + 16 + size].to_vec();
        }
        if idx == 0 {
            break; // reached the free-space object
        }
        pos = (pos + 16 + size + 7) & !7;
    }
    panic!("heap object {want} not found");
}
