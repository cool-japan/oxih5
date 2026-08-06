//! G004 / G015 / G016 — compound, ragged, array, opaque and bitfield writes.
//!
//! Each test writes a dataset with `FileWriter`, reads it back through this
//! crate's own reader, and asserts the datatype message's **bytes** as well as
//! the decoded values.  The byte assertions come from datatype messages **h5py
//! 3.16 / libhdf5 2.0.0** wrote for the equivalent numpy types, and every file
//! these tests produce has been read back by that libhdf5.

use oxih5::FileWriter;
use oxih5_core::{ByteOrder, Charset, CompoundField, Dtype};
use oxih5_format::values::Value;
use oxih5_format::{header, superblock};

/// Write `writer` to a scratch file and return both the path and its bytes.
fn build(writer: &mut FileWriter, name: &str) -> (std::path::PathBuf, Vec<u8>) {
    let path = std::env::temp_dir().join(format!("oxih5_wave5dt_{name}.h5"));
    writer.build(&path).expect("build");
    let bytes = std::fs::read(&path).expect("read back");
    (path, bytes)
}

/// The datatype message body of the dataset at `path` within `bytes`.
fn datatype_body(bytes: &[u8], file: &oxih5::File, path: &str) -> Vec<u8> {
    let addr = file.header_addr_of(path).expect("header address");
    let msgs = header::parse_messages(bytes, addr).expect("messages");
    msgs.iter()
        .find(|m| m.msg_type == 0x0003)
        .expect("datatype message")
        .data
        .clone()
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

fn field(name: &str, offset: usize, dtype: Dtype) -> CompoundField {
    CompoundField {
        name: name.to_string(),
        offset,
        dtype,
    }
}

// ---------------------------------------------------------------------------
// G004 — compound (record) datasets
// ---------------------------------------------------------------------------

/// A record dataset's datatype message is byte-identical to the one libhdf5
/// writes for `numpy.dtype([('id', '<i4'), ('value', '<f8')])`, and its rows
/// come back with the offsets they were written at.
#[test]
fn a_compound_dataset_matches_libhdf5_and_round_trips() {
    let fields = vec![field("id", 0, i32_dtype()), field("value", 4, f64_dtype())];
    let mut rows = Vec::new();
    for (id, value) in [(1i32, 2.5f64), (3, 4.5), (5, -1.25)] {
        rows.extend_from_slice(&id.to_le_bytes());
        rows.extend_from_slice(&value.to_le_bytes());
    }

    let mut w = FileWriter::new();
    w.create_compound_dataset("/events", &fields, 12, &rows, &[3])
        .expect("compound");
    let (path, bytes) = build(&mut w, "compound");

    let f = oxih5::open(&path).expect("open");
    let ds = f.dataset("events").expect("events");
    assert_eq!(ds.shape, vec![3]);
    assert_eq!(ds.data, rows, "the row bytes are stored verbatim");

    let Dtype::Compound { fields: parsed } = &ds.dtype else {
        panic!("expected a compound, got {:?}", ds.dtype);
    };
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].name, "id");
    assert_eq!(parsed[0].offset, 0);
    assert_eq!(parsed[0].dtype, i32_dtype());
    assert_eq!(parsed[1].name, "value");
    assert_eq!(parsed[1].offset, 4);
    assert_eq!(parsed[1].dtype, f64_dtype());
    assert_eq!(ds.dtype.size(), Some(12), "the declared record stride");

    // Byte-pinned against libhdf5's own encoding of the same record type.
    let body = datatype_body(&bytes, &f, "events");
    let mut want: Vec<u8> = vec![0x16, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00];
    want.extend_from_slice(b"id\0\0\0\0\0\0");
    want.extend_from_slice(&0u32.to_le_bytes());
    want.extend_from_slice(&[0u8; 28]);
    want.extend_from_slice(&[
        0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00,
    ]);
    want.extend_from_slice(b"value\0\0\0");
    want.extend_from_slice(&4u32.to_le_bytes());
    want.extend_from_slice(&[0u8; 28]);
    want.extend_from_slice(&[
        0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b, 0x00,
        0x34, 0xff, 0x03, 0x00, 0x00,
    ]);
    // The object header pads the message out to 8 bytes; compare the live part.
    assert_eq!(&body[..want.len()], &want[..]);
    assert!(
        body[want.len()..].iter().all(|&b| b == 0),
        "only padding beyond the encoded type"
    );
    let _ = std::fs::remove_file(&path);
}

/// A record laid out with C padding declares the offsets that padding produces;
/// the writer must not silently repack it, or the bytes and the type would
/// describe different things.
#[test]
fn a_padded_record_keeps_the_offsets_it_was_given() {
    let fields = vec![field("a", 0, i32_dtype()), field("b", 8, f64_dtype())];
    let mut rows = vec![0u8; 16];
    rows[0..4].copy_from_slice(&7i32.to_le_bytes());
    rows[8..16].copy_from_slice(&0.5f64.to_le_bytes());

    let mut w = FileWriter::new();
    w.create_compound_dataset("/rec", &fields, 16, &rows, &[1])
        .expect("compound");
    let (path, _) = build(&mut w, "padded_record");

    let f = oxih5::open(&path).expect("open");
    let ds = f.dataset("rec").expect("rec");
    let Dtype::Compound { fields: parsed } = &ds.dtype else {
        panic!("expected a compound");
    };
    assert_eq!(parsed[1].offset, 8, "the four-byte hole is preserved");
    assert_eq!(ds.dtype.size(), Some(16));
    assert_eq!(ds.data.len(), 16);
    let _ = std::fs::remove_file(&path);
}

/// A record with a fixed-length string member — the shape a pandas HDFStore
/// table or an event log actually uses.
#[test]
fn a_compound_may_hold_a_fixed_length_string_member() {
    let fields = vec![
        field("id", 0, i32_dtype()),
        field(
            "label",
            4,
            Dtype::String {
                fixed_len: Some(8),
                charset: Charset::Ascii,
            },
        ),
    ];
    let mut rows = vec![0u8; 24];
    rows[0..4].copy_from_slice(&1i32.to_le_bytes());
    rows[4..8].copy_from_slice(b"cold");
    rows[12..16].copy_from_slice(&2i32.to_le_bytes());
    rows[16..20].copy_from_slice(b"warm");

    let mut w = FileWriter::new();
    w.create_compound_dataset("/log", &fields, 12, &rows, &[2])
        .expect("compound");
    let (path, _) = build(&mut w, "string_member");

    let f = oxih5::open(&path).expect("open");
    let ds = f.dataset("log").expect("log");
    let Dtype::Compound { fields: parsed } = &ds.dtype else {
        panic!("expected a compound");
    };
    assert!(matches!(
        parsed[1].dtype,
        Dtype::String {
            fixed_len: Some(8),
            ..
        }
    ));
    assert_eq!(&ds.data[4..8], b"cold");
    assert_eq!(&ds.data[16..20], b"warm");
    let _ = std::fs::remove_file(&path);
}

/// Every way of describing an impossible record is a typed error at the call.
#[test]
fn malformed_record_descriptions_are_rejected() {
    let mut w = FileWriter::new();
    let rows = vec![0u8; 12];

    let bad = [
        // Two members overlapping.
        (
            vec![field("a", 0, f64_dtype()), field("b", 4, i32_dtype())],
            12usize,
        ),
        // A member past the declared record size.
        (vec![field("a", 8, f64_dtype())], 12),
        // Two members of one name.
        (
            vec![field("a", 0, i32_dtype()), field("a", 4, i32_dtype())],
            12,
        ),
        // An empty member name.
        (vec![field("", 0, i32_dtype())], 4),
        // No members at all.
        (Vec::new(), 4),
    ];
    for (index, (fields, size)) in bad.into_iter().enumerate() {
        assert!(
            w.create_compound_dataset("/x", &fields, size, &rows, &[1])
                .is_err(),
            "case {index} should have been refused"
        );
    }
    // A member type the writer cannot encode inline.
    assert!(w
        .create_compound_dataset(
            "/x",
            &[field(
                "r",
                0,
                Dtype::Reference {
                    ref_type: oxih5_core::RefType::Object
                }
            )],
            8,
            &rows[..8],
            &[1]
        )
        .is_err());
    // Row bytes that do not match shape × record size.
    assert!(w
        .create_compound_dataset("/x", &[field("a", 0, i32_dtype())], 4, &rows, &[1])
        .is_err());
    // …and none of those left a half-built dataset behind.
    let bytes = w.build_to_vec().expect("build");
    let f = oxih5::File::open_from_bytes(&bytes).expect("open");
    assert!(f.dataset_names().expect("names").is_empty());
}

// ---------------------------------------------------------------------------
// G015 — variable-length (ragged) sequence datasets
// ---------------------------------------------------------------------------

/// A ragged `int32` dataset round-trips through `dataset_vlen_sequences`, and
/// its datatype message is byte-identical to `h5py.vlen_dtype(numpy.int32)`.
#[test]
fn a_ragged_int32_dataset_matches_libhdf5_and_round_trips() {
    let rows = vec![vec![1i32, 2, 3], Vec::new(), vec![10], vec![-4, -5]];
    let mut w = FileWriter::new();
    w.create_vlen_i32_dataset("/rows", &rows).expect("ragged");
    let (path, bytes) = build(&mut w, "ragged_i32");

    let f = oxih5::open(&path).expect("open");
    let ds = f.dataset("rows").expect("rows");
    assert_eq!(ds.shape, vec![4]);
    assert_eq!(
        ds.dtype,
        Dtype::VarLen {
            base: Box::new(i32_dtype())
        }
    );

    let seqs = f.dataset_vlen_sequences("rows").expect("sequences");
    let decoded: Vec<Vec<i64>> = seqs
        .iter()
        .map(|value| {
            let Value::Sequence(items) = value else {
                panic!("expected a sequence, got {value:?}");
            };
            items
                .iter()
                .map(|item| match item {
                    Value::Int(v) => *v,
                    other => panic!("expected an int, got {other:?}"),
                })
                .collect()
        })
        .collect();
    assert_eq!(
        decoded,
        vec![vec![1, 2, 3], Vec::new(), vec![10], vec![-4, -5]]
    );

    // Byte-pinned against libhdf5's `vlen_dtype(int32)`.
    assert_eq!(
        datatype_body(&bytes, &f, "rows"),
        vec![
            0x19, 0x00, 0x00, 0x00, // class 9 (vlen) v1, subtype 0 = sequence
            0x10, 0x00, 0x00, 0x00, // 16-byte on-disk reference
            0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
            0x00, // int32 base
            0x00, 0x00, 0x00, 0x00, // object-header padding to 8 bytes
        ]
    );

    // Each element is a 16-byte reference whose sequence length counts
    // *elements*, not bytes — 3, not 12 — and an empty element is the all-zero
    // null reference libhdf5 writes.
    let lens: Vec<u32> = (0..4)
        .map(|i| {
            let at = i * 16;
            u32::from_le_bytes(ds.data[at..at + 4].try_into().expect("4 bytes"))
        })
        .collect();
    assert_eq!(lens, vec![3, 0, 1, 2]);
    assert!(
        ds.data[16..32].iter().all(|&b| b == 0),
        "an empty sequence is the null reference"
    );
    let _ = std::fs::remove_file(&path);
}

/// The same machinery over `float64`, and over a base type given as a `Dtype`.
#[test]
fn ragged_datasets_work_for_every_fixed_base_type() {
    let mut w = FileWriter::new();
    w.create_vlen_f64_dataset("/reals", &[vec![1.5], vec![2.5, 3.5]])
        .expect("f64");
    let raw: Vec<Vec<u8>> = vec![vec![1u8, 2, 3], vec![9]];
    w.create_vlen_sequence_dataset(
        "/bytes",
        &Dtype::Int {
            size: 1,
            signed: false,
            order: ByteOrder::Little,
        },
        &raw,
    )
    .expect("u8");
    let (path, _) = build(&mut w, "ragged_mixed");

    let f = oxih5::open(&path).expect("open");
    let reals = f.dataset_vlen_sequences("reals").expect("reals");
    let Value::Sequence(second) = &reals[1] else {
        panic!("expected a sequence");
    };
    assert_eq!(second.len(), 2);
    assert!(matches!(second[0], Value::Float(v) if (v - 2.5).abs() < f64::EPSILON));

    let raw_back = f.dataset_vlen_sequences("bytes").expect("bytes");
    let Value::Sequence(first) = &raw_back[0] else {
        panic!("expected a sequence");
    };
    assert_eq!(first.len(), 3);
    let _ = std::fs::remove_file(&path);
}

/// Ragged and vlen-string datasets share one global heap, and both still decode
/// — the sequence's references must not disturb the strings' object indices.
#[test]
fn ragged_and_string_vlen_datasets_share_one_global_heap() {
    let mut w = FileWriter::new();
    w.create_vlen_string_dataset("/names", &["alpha", "beta"])
        .expect("strings");
    w.create_vlen_i32_dataset("/rows", &[vec![1, 2], vec![3]])
        .expect("ragged");
    w.create_vlen_string_dataset("/more", &["gamma"])
        .expect("more strings");
    let (path, _) = build(&mut w, "shared_heap");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset_strings("names").expect("names"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
    assert_eq!(
        f.dataset_strings("more").expect("more"),
        vec!["gamma".to_string()]
    );
    let seqs = f.dataset_vlen_sequences("rows").expect("rows");
    let Value::Sequence(first) = &seqs[0] else {
        panic!("expected a sequence");
    };
    assert_eq!(first.len(), 2);
    let _ = std::fs::remove_file(&path);
}

/// A ragged dataset has no raw data area to tile, filter or inline, so every
/// storage-modifying entry point refuses it rather than producing a file whose
/// references a reader cannot follow.
#[test]
fn a_ragged_dataset_refuses_tiling_filtering_and_compaction() {
    let mut w = FileWriter::new();
    w.create_vlen_i32_dataset("/rows", &[vec![1, 2]])
        .expect("ragged");
    assert!(w.set_deflate("rows", 6).is_err(), "deflate");
    assert!(w.set_shuffle("rows").is_err(), "shuffle");
    assert!(w.set_fletcher32("rows").is_err(), "fletcher32");
    assert!(w.set_chunking("rows", &[1]).is_err(), "chunking");
    assert!(w.set_compact("rows").is_err(), "compact");
    // The dataset survives every refusal unchanged.
    let bytes = w.build_to_vec().expect("build");
    let f = oxih5::File::open_from_bytes(&bytes).expect("open");
    assert!(matches!(
        f.dataset("rows").expect("rows").dtype,
        Dtype::VarLen { .. }
    ));
}

// ---------------------------------------------------------------------------
// G016 — array, opaque and bitfield datasets
// ---------------------------------------------------------------------------

/// An array-datatype dataset: every element is a fixed block of the base type,
/// encoded as datatype message **version 2** — the only version the array class
/// has, and the one libhdf5 refuses to read a file without.
#[test]
fn an_array_dataset_matches_libhdf5_and_round_trips() {
    let data: Vec<u8> = (0..12i32).flat_map(i32::to_le_bytes).collect();
    let mut w = FileWriter::new();
    w.create_array_dataset("/arr", &i32_dtype(), &[2, 3], &data, &[2])
        .expect("array");
    let (path, bytes) = build(&mut w, "array");

    let f = oxih5::open(&path).expect("open");
    let ds = f.dataset("arr").expect("arr");
    assert_eq!(ds.shape, vec![2]);
    assert_eq!(
        ds.dtype,
        Dtype::Array {
            base: Box::new(i32_dtype()),
            dims: vec![2, 3]
        }
    );
    assert_eq!(ds.dtype.size(), Some(24));
    assert_eq!(ds.data, data);

    assert_eq!(
        datatype_body(&bytes, &f, "arr"),
        vec![
            0x2a, 0x00, 0x00, 0x00, // class 10 (array), version 2
            0x18, 0x00, 0x00, 0x00, // element size = 2 × 3 × 4
            0x02, 0x00, 0x00, 0x00, // dimensionality, then 3 reserved bytes
            0x02, 0x00, 0x00, 0x00, // dim[0]
            0x03, 0x00, 0x00, 0x00, // dim[1]
            0x00, 0x00, 0x00, 0x00, // permutation[0]
            0x01, 0x00, 0x00, 0x00, // permutation[1]
            0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00,
        ]
    );
    let _ = std::fs::remove_file(&path);
}

/// An opaque dataset carries its tag NUL-padded to eight bytes, exactly as
/// libhdf5 encodes `h5py.opaque_dtype(numpy.dtype('V7'))`.
#[test]
fn an_opaque_dataset_matches_libhdf5_and_round_trips() {
    let data = vec![0xABu8; 21];
    let mut w = FileWriter::new();
    w.create_opaque_dataset("/opq", 7, "NUMPY:|V7", &data, &[3])
        .expect("opaque");
    let (path, bytes) = build(&mut w, "opaque");

    let f = oxih5::open(&path).expect("open");
    let ds = f.dataset("opq").expect("opq");
    assert_eq!(
        ds.dtype,
        Dtype::Opaque {
            size: 7,
            tag: "NUMPY:|V7".to_string()
        }
    );
    assert_eq!(ds.data, data);

    let mut want: Vec<u8> = vec![0x15, 0x10, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00];
    want.extend_from_slice(b"NUMPY:|V7\0\0\0\0\0\0\0");
    assert_eq!(datatype_body(&bytes, &f, "opq"), want);
    let _ = std::fs::remove_file(&path);
}

/// A bitfield dataset, in both byte orders, byte-pinned against libhdf5's
/// `H5T_STD_B8LE`.
#[test]
fn a_bitfield_dataset_matches_libhdf5_and_round_trips() {
    let mut w = FileWriter::new();
    w.create_bitfield_dataset("/bits", 1, ByteOrder::Little, &[0x0F, 0xF0], &[2])
        .expect("bitfield");
    w.create_bitfield_dataset("/wide", 4, ByteOrder::Big, &[0, 0, 0, 1, 0, 0, 0, 2], &[2])
        .expect("wide bitfield");
    let (path, bytes) = build(&mut w, "bitfield");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("bits").expect("bits").dtype,
        Dtype::Bitfield {
            size: 1,
            order: ByteOrder::Little
        }
    );
    assert_eq!(
        f.dataset("wide").expect("wide").dtype,
        Dtype::Bitfield {
            size: 4,
            order: ByteOrder::Big
        }
    );
    assert_eq!(f.dataset("bits").expect("bits").data, vec![0x0F, 0xF0]);

    // libhdf5's own `H5T_STD_B8LE` encoding is twelve bytes; the object header
    // pads the message out to its 8-byte grid, which must be zeros and nothing
    // else.
    let bits = datatype_body(&bytes, &f, "bits");
    assert_eq!(
        &bits[..12],
        &[0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00]
    );
    assert!(bits[12..].iter().all(|&b| b == 0), "only padding beyond");
    // Only bit 0 of the bit field differs for the big-endian one.
    let wide = datatype_body(&bytes, &f, "wide");
    assert_eq!(wide[0], 0x14);
    assert_eq!(wide[1], 0x01, "big-endian");
    assert_eq!(
        u16::from_le_bytes([wide[10], wide[11]]),
        32,
        "bit precision"
    );
    let _ = std::fs::remove_file(&path);
}

/// Degenerate structured-type requests are typed errors at the call, and the
/// element size a structured type declares is the one every later size check
/// uses.
#[test]
fn degenerate_structured_types_are_rejected() {
    let mut w = FileWriter::new();
    assert!(
        w.create_array_dataset("/x", &i32_dtype(), &[], &[], &[0])
            .is_err(),
        "no dimensions"
    );
    assert!(
        w.create_array_dataset("/x", &i32_dtype(), &[2, 0], &[], &[0])
            .is_err(),
        "zero extent"
    );
    assert!(
        w.create_opaque_dataset("/x", 0, "t", &[], &[0]).is_err(),
        "zero width"
    );
    assert!(
        w.create_opaque_dataset("/x", 4, "a\0b", &[0; 4], &[1])
            .is_err(),
        "NUL in tag"
    );
    assert!(
        w.create_bitfield_dataset("/x", 0, ByteOrder::Little, &[], &[0])
            .is_err(),
        "zero-byte bitfield"
    );
    assert!(
        w.create_bitfield_dataset("/x", 9, ByteOrder::Little, &[0; 9], &[1])
            .is_err(),
        "over-wide bitfield"
    );
    // A data length that does not match the structured element size.
    assert!(
        w.create_array_dataset("/x", &i32_dtype(), &[2, 3], &[0u8; 12], &[2])
            .is_err(),
        "half an element per row"
    );
}

/// A structured datatype has no scalar fill value, so `set_fill_value_*` must
/// refuse it rather than compare against the placeholder `ElemType`.
#[test]
fn a_structured_dataset_has_no_scalar_fill_value() {
    let mut w = FileWriter::new();
    w.create_opaque_dataset("/opq", 7, "tag", &[0u8; 7], &[1])
        .expect("opaque");
    let Err(err) = w.set_fill_value_u8("opq", 0) else {
        panic!("a structured datatype has no scalar fill");
    };
    assert!(format!("{err}").contains("no scalar fill value"), "{err}");
}

/// Every structured type survives being placed in a nested group beside
/// ordinary datasets — the layout pass must size them all from the same
/// element-size definition.
#[test]
fn structured_datasets_coexist_with_ordinary_ones() {
    let fields = vec![field("id", 0, i32_dtype()), field("value", 4, f64_dtype())];
    let mut rows = Vec::new();
    rows.extend_from_slice(&11i32.to_le_bytes());
    rows.extend_from_slice(&0.5f64.to_le_bytes());

    let mut w = FileWriter::new();
    w.write_dataset_f64("/plain", &[1.0, 2.0], &[2])
        .expect("plain");
    w.create_compound_dataset("/g/events", &fields, 12, &rows, &[1])
        .expect("compound");
    w.create_vlen_i32_dataset("/g/rows", &[vec![1], vec![2, 3]])
        .expect("ragged");
    w.create_array_dataset(
        "/g/arr",
        &i32_dtype(),
        &[2],
        &(0..4i32).flat_map(i32::to_le_bytes).collect::<Vec<u8>>(),
        &[2],
    )
    .expect("array");
    w.write_string_attr("/g/events", "units", "counts")
        .expect("attr");
    let (path, bytes) = build(&mut w, "mixed");

    let f = oxih5::open(&path).expect("open");
    assert_eq!(
        f.dataset("plain").expect("plain").as_f64().expect("f64"),
        vec![1.0, 2.0]
    );
    assert_eq!(f.dataset("g/events").expect("events").data.len(), 12);
    assert_eq!(f.dataset("g/arr").expect("arr").data.len(), 16);
    assert_eq!(f.dataset_vlen_sequences("g/rows").expect("rows").len(), 2);

    // The superblock's end-of-file address is where the file actually ends,
    // which is the check that catches a structured type whose size formula and
    // its emitter disagree by even one byte.
    assert_eq!(
        u64::from_le_bytes(bytes[40..48].try_into().expect("8 bytes")),
        bytes.len() as u64
    );
    assert_eq!(
        superblock::parse(&bytes).expect("superblock").version,
        0,
        "still a version-0 superblock"
    );
    let _ = std::fs::remove_file(&path);
}
