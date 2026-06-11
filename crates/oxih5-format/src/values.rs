//! Typed value decoding for complex HDF5 datatypes.
//!
//! Handles:
//! - Variable-length (vlen) STRING and vlen-of-base SEQUENCE values via GlobalHeap
//! - Object references (8-byte on-disk addresses → target address)
//! - Compound-type element decoding (recursively dispatched through `decode_one_value`)
//!
//! # On-disk vlen reference layout (HDF5 spec, §III.B.2)
//! Each vlen element stored in the dataset/attribute data is a 16-byte global-heap
//! reference:
//! ```text
//!  0   4   sequence length (number of elements, u32 LE)
//!  4   4   reserved / padding (u32 LE)
//!  8   8   global heap collection address (absolute, u64 LE)
//! ```
//! (The object index into the collection is stored as the low 16 bits of a u32 at
//! offset 4; some implementations use slightly different encodings but the "length +
//! collection address + object index" triple is universal.)
//!
//! The on-disk layout actually used by libhdf5 is:
//! ```text
//!  0   4   sequence length (u32 LE)
//!  4   2   object index (u16 LE)
//!  6   2   reserved (u16 LE)
//!  8   8   heap collection address (u64 LE)
//! ```
//!
//! # Object references
//! An object reference is a single u64 (LE) equal to the absolute byte offset of the
//! target object's header within the HDF5 file.  `u64::MAX` denotes an undefined
//! reference.

use std::collections::HashMap;

use oxih5_core::{ByteOrder, CompoundField, Dtype, OxiH5Error, RefType};

use crate::global_heap::GlobalHeap;

// ---------------------------------------------------------------------------
// Public `Value` type — the decoded Rust representation of an HDF5 element
// ---------------------------------------------------------------------------

/// A decoded HDF5 value (one element of any supported datatype).
///
/// This is a typed envelope used by compound-decode and attribute helper APIs.
/// For simple scalar types you will usually reach for `Dataset::as_f64()` etc.
/// directly.  `Value` exists for richer contexts (compound fields, vlen
/// sequences, object references) where a single typed container is needed.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Signed integer (any width), widened to i64.
    Int(i64),
    /// Unsigned integer (any width), widened to u64.
    Uint(u64),
    /// Floating-point (f32 or f64), widened to f64.
    Float(f64),
    /// Fixed-length or vlen UTF-8 string.
    Str(String),
    /// Opaque bytes.
    Opaque(Vec<u8>),
    /// Object reference — absolute byte address of the target object header.
    /// `u64::MAX` means undefined / null reference.
    ObjectRef(u64),
    /// Region reference — points to a dataset and a selection within it.
    RegionRef {
        /// Absolute byte address of the referenced dataset's object header.
        dataset_addr: u64,
        /// The data-space selection encoded in the region reference.
        selection: RegionSelection,
    },
    /// Sequence of values (vlen array or array datatype).
    Sequence(Vec<Value>),
    /// Named fields of a compound element.
    Compound(Vec<(String, Value)>),
    /// Enum member value (raw integer).
    Enum(i64),
    /// Bitfield (raw bit pattern).
    Bitfield(u64),
}

// ---------------------------------------------------------------------------
// Region reference selection types
// ---------------------------------------------------------------------------

/// A data-space selection encoded in an HDF5 region reference.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionSelection {
    /// Point list: each inner `Vec<u64>` is one point (one index per dimension).
    Points(Vec<Vec<u64>>),
    /// Hyperslab: each tuple is `(start, count)` per dimension.
    Hyperslab(Vec<(u64, u64)>),
}

// ---------------------------------------------------------------------------
// VLen reference helpers
// ---------------------------------------------------------------------------

/// Parse a 16-byte on-disk vlen reference into (length, heap_address, object_index).
///
/// Layout:
/// ```text
///  0   4   sequence length  (u32 LE)
///  4   2   object index     (u16 LE)
///  6   2   reserved         (u16 LE, ignored)
///  8   8   heap address     (u64 LE)
/// ```
pub fn parse_vlen_ref(bytes: &[u8]) -> Result<(u32, u64, u16), OxiH5Error> {
    if bytes.len() < 16 {
        return Err(OxiH5Error::Format(format!(
            "vlen ref: need 16 bytes, got {}",
            bytes.len()
        )));
    }
    let seq_len = u32::from_le_bytes(
        bytes[0..4]
            .try_into()
            .map_err(|_| OxiH5Error::Format("vlen ref: length bytes".into()))?,
    );
    let obj_idx = u16::from_le_bytes(
        bytes[4..6]
            .try_into()
            .map_err(|_| OxiH5Error::Format("vlen ref: index bytes".into()))?,
    );
    let heap_addr = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| OxiH5Error::Format("vlen ref: heap address bytes".into()))?,
    );
    Ok((seq_len, heap_addr, obj_idx))
}

/// Fetch the data bytes for a heap object, using the per-collection parse cache.
///
/// `heap_cache` maps heap collection address → parsed `GlobalHeap`.  We parse
/// each collection at most once per decode call.
fn heap_object_bytes<'a>(
    file_data: &[u8],
    heap_addr: u64,
    obj_idx: u16,
    heap_cache: &'a mut HashMap<u64, GlobalHeap>,
) -> Result<&'a [u8], OxiH5Error> {
    if let std::collections::hash_map::Entry::Vacant(e) = heap_cache.entry(heap_addr) {
        let heap = GlobalHeap::parse(file_data, heap_addr)?;
        e.insert(heap);
    }
    let heap = heap_cache
        .get(&heap_addr)
        .ok_or_else(|| OxiH5Error::Format(format!("heap cache miss for addr {heap_addr}")))?;
    heap.object(obj_idx)
}

// ---------------------------------------------------------------------------
// VLen string decode
// ---------------------------------------------------------------------------

/// Decode a buffer containing N contiguous 16-byte vlen-string references.
///
/// Each reference points to a NUL-terminated (or unterminated) UTF-8 byte
/// sequence stored in a global-heap collection.
///
/// `n_elems` is the number of vlen elements (one 16-byte reference each).
/// An empty heap object (zero-length data) decodes as an empty `String`.
pub fn decode_vlen_strings(
    file_data: &[u8],
    data: &[u8],
    n_elems: usize,
) -> Result<Vec<String>, OxiH5Error> {
    if data.len() < n_elems * 16 {
        return Err(OxiH5Error::Format(format!(
            "vlen string buffer: need {} bytes for {} elems, got {}",
            n_elems * 16,
            n_elems,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(n_elems);
    let mut heap_cache: HashMap<u64, GlobalHeap> = HashMap::new();

    for i in 0..n_elems {
        let ref_bytes = &data[i * 16..(i + 1) * 16];
        let (seq_len, heap_addr, obj_idx) = parse_vlen_ref(ref_bytes)?;

        if seq_len == 0 {
            out.push(String::new());
            continue;
        }

        let raw = heap_object_bytes(file_data, heap_addr, obj_idx, &mut heap_cache)?;

        // Strip trailing NUL(s) — libhdf5 NUL-terminates but the sequence length
        // includes the terminator in some versions.
        let trimmed = raw.split(|&b| b == 0).next().unwrap_or(raw);
        let s = String::from_utf8(trimmed.to_vec())
            .map_err(|e| OxiH5Error::Format(format!("vlen string UTF-8: {e}")))?;
        out.push(s);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// VLen sequence decode
// ---------------------------------------------------------------------------

/// Decode a buffer containing N contiguous 16-byte vlen-of-base SEQUENCE references.
///
/// Each decoded element is a `Value::Sequence` of `seq_len` elements, each decoded
/// by the `base_dtype` dispatcher.
pub fn decode_vlen_sequences(
    file_data: &[u8],
    data: &[u8],
    n_elems: usize,
    base_dtype: &Dtype,
) -> Result<Vec<Value>, OxiH5Error> {
    if data.len() < n_elems * 16 {
        return Err(OxiH5Error::Format(format!(
            "vlen sequence buffer: need {} bytes for {} refs, got {}",
            n_elems * 16,
            n_elems,
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n_elems);
    let mut heap_cache: HashMap<u64, GlobalHeap> = HashMap::new();

    // For vlen-of-vlen (and other variable-length base types), each element in
    // the heap object is itself a 16-byte vlen reference.  We detect this by
    // checking whether the base dtype has a fixed size; if not we use 16 bytes
    // per element (the on-disk vlen reference footprint).
    let base_size_opt = base_dtype.size();
    let is_vlen_base = base_size_opt.is_none();
    // For vlen bases, each nested element is stored as a 16-byte heap reference.
    let elem_footprint = base_size_opt.unwrap_or(16);

    for i in 0..n_elems {
        let ref_bytes = &data[i * 16..(i + 1) * 16];
        let (seq_len, heap_addr, obj_idx) = parse_vlen_ref(ref_bytes)?;

        if seq_len == 0 {
            out.push(Value::Sequence(vec![]));
            continue;
        }

        let seq_len = seq_len as usize;
        let raw = heap_object_bytes(file_data, heap_addr, obj_idx, &mut heap_cache)?.to_vec();

        let needed = seq_len * elem_footprint;
        if raw.len() < needed {
            return Err(OxiH5Error::Format(format!(
                "vlen sequence elem {i}: heap object has {} bytes, expected {} ({seq_len} × {elem_footprint})",
                raw.len(),
                needed
            )));
        }

        let mut elems = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let elem_bytes = &raw[j * elem_footprint..(j + 1) * elem_footprint];
            if is_vlen_base {
                // Nested vlen: decode the 16-byte reference as a vlen element.
                elems.push(decode_one_value(
                    file_data,
                    elem_bytes,
                    base_dtype,
                    &mut heap_cache,
                    1,
                )?);
            } else {
                elems.push(decode_one_value(
                    file_data,
                    elem_bytes,
                    base_dtype,
                    &mut heap_cache,
                    0,
                )?);
            }
        }
        out.push(Value::Sequence(elems));
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Object reference decode
// ---------------------------------------------------------------------------

/// Decode a buffer containing N contiguous 8-byte object references.
///
/// Each reference is a u64 LE absolute byte offset of the target object's header.
/// `u64::MAX` is an undefined/null reference (returned as `Value::ObjectRef(u64::MAX)`).
pub fn decode_object_refs(data: &[u8], n_elems: usize) -> Result<Vec<u64>, OxiH5Error> {
    if data.len() < n_elems * 8 {
        return Err(OxiH5Error::Format(format!(
            "object refs buffer: need {} bytes, got {}",
            n_elems * 8,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let arr: [u8; 8] = data[i * 8..(i + 1) * 8]
            .try_into()
            .map_err(|_| OxiH5Error::Format("object ref: byte slice".into()))?;
        out.push(u64::from_le_bytes(arr));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Compound element decode
// ---------------------------------------------------------------------------

/// Decode a single element of a compound dtype.
///
/// `element_bytes` is a slice of exactly `dtype.size()` bytes for this element.
/// Each field is decoded by `decode_one_value` and collected into a `Value::Compound`.
pub fn decode_compound_element(
    file_data: &[u8],
    element_bytes: &[u8],
    fields: &[CompoundField],
    heap_cache: &mut HashMap<u64, GlobalHeap>,
    depth: u32,
) -> Result<Value, OxiH5Error> {
    let mut decoded = Vec::with_capacity(fields.len());
    for field in fields {
        // Variable-length fields in a compound are stored as 16-byte heap
        // references on disk.  Use 16 as the field size when dtype.size() is None.
        let field_size = field.dtype.size().unwrap_or(16);
        let field_end = field.offset.checked_add(field_size).ok_or_else(|| {
            OxiH5Error::Format(format!(
                "compound field '{}': offset+size overflows",
                field.name
            ))
        })?;
        let field_bytes = element_bytes.get(field.offset..field_end).ok_or_else(|| {
            OxiH5Error::Format(format!(
                "compound field '{}': element slice [{}..{}] out of bounds (element is {} bytes)",
                field.name,
                field.offset,
                field_end,
                element_bytes.len()
            ))
        })?;
        let val = decode_one_value(file_data, field_bytes, &field.dtype, heap_cache, depth)?;
        decoded.push((field.name.clone(), val));
    }
    Ok(Value::Compound(decoded))
}

/// Decode all elements of a compound dataset or attribute buffer.
///
/// `data` contains `n_elems` packed elements, each of `elem_stride` bytes.
/// When `elem_stride` is 0 it defaults to `dtype.size()` (no trailing padding).
pub fn decode_compound(
    file_data: &[u8],
    data: &[u8],
    fields: &[CompoundField],
    n_elems: usize,
    elem_stride: usize,
) -> Result<Vec<Value>, OxiH5Error> {
    if elem_stride == 0 && n_elems == 0 {
        return Ok(vec![]);
    }
    let stride = if elem_stride == 0 {
        // Compute from field layout: max(offset + field_size).
        let mut s = 0usize;
        for f in fields {
            // Variable-length fields are stored as 16-byte heap references on disk.
            let fsz = f.dtype.size().unwrap_or(16);
            s = s.max(f.offset + fsz);
        }
        s
    } else {
        elem_stride
    };

    if stride == 0 {
        return Err(OxiH5Error::Format(
            "compound decode: element stride is zero".into(),
        ));
    }

    let needed = stride
        .checked_mul(n_elems)
        .ok_or_else(|| OxiH5Error::Format("compound decode: stride × n_elems overflows".into()))?;
    if data.len() < needed {
        return Err(OxiH5Error::Format(format!(
            "compound decode: buffer has {} bytes, need {needed} ({n_elems} × {stride})",
            data.len()
        )));
    }

    let mut heap_cache: HashMap<u64, GlobalHeap> = HashMap::new();
    let mut out = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let elem = &data[i * stride..(i + 1) * stride];
        out.push(decode_compound_element(
            file_data,
            elem,
            fields,
            &mut heap_cache,
            0,
        )?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Central value dispatcher
// ---------------------------------------------------------------------------

/// Maximum recursion depth for nested compound/array/vlen decode.
const MAX_DECODE_DEPTH: u32 = 16;

/// Decode a single HDF5 value of any type from the raw byte slice `bytes`.
///
/// `bytes` must contain exactly `dtype.size()` bytes (for fixed-width types).
/// For vlen types `bytes` must contain exactly 16 bytes (the vlen reference).
///
/// `heap_cache` is passed through so nested vlen/compound fields share the
/// per-call collection cache.
///
/// `depth` guards against runaway recursion; returns `OxiH5Error::Format` when
/// `depth >= MAX_DECODE_DEPTH`.
pub fn decode_one_value(
    file_data: &[u8],
    bytes: &[u8],
    dtype: &Dtype,
    heap_cache: &mut HashMap<u64, GlobalHeap>,
    depth: u32,
) -> Result<Value, OxiH5Error> {
    if depth >= MAX_DECODE_DEPTH {
        return Err(OxiH5Error::Format(format!(
            "decode_one_value: recursion depth {depth} exceeds limit {MAX_DECODE_DEPTH}"
        )));
    }

    match dtype {
        // ------------------------------------------------------------------ integers
        Dtype::Int {
            size,
            signed,
            order,
        } => {
            let sz = *size;
            if bytes.len() < sz {
                return Err(OxiH5Error::Format(format!(
                    "int decode: need {sz} bytes, got {}",
                    bytes.len()
                )));
            }
            let b = &bytes[..sz];
            if *signed {
                let v: i64 = match (sz, order) {
                    (1, _) => i64::from(b[0] as i8),
                    (2, ByteOrder::Little) => i64::from(i16::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("i16 decode".into()))?,
                    )),
                    (2, ByteOrder::Big) => i64::from(i16::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("i16 decode".into()))?,
                    )),
                    (4, ByteOrder::Little) => i64::from(i32::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("i32 decode".into()))?,
                    )),
                    (4, ByteOrder::Big) => i64::from(i32::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("i32 decode".into()))?,
                    )),
                    (8, ByteOrder::Little) => i64::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("i64 decode".into()))?,
                    ),
                    (8, ByteOrder::Big) => i64::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("i64 decode".into()))?,
                    ),
                    // Odd-size signed integers (3, 5, 6, 7 bytes): pack into u64
                    // then sign-extend via arithmetic shift.
                    (sz, bo) if sz > 0 && sz <= 8 => {
                        let mut raw = 0u64;
                        match bo {
                            ByteOrder::Little => {
                                for (i, &byte) in b.iter().enumerate() {
                                    raw |= u64::from(byte) << (i * 8);
                                }
                            }
                            ByteOrder::Big => {
                                for (i, &byte) in b.iter().enumerate() {
                                    raw |= u64::from(byte) << ((sz - 1 - i) * 8);
                                }
                            }
                        }
                        let shift = 64u32.saturating_sub((sz as u32) * 8);
                        // Sign-extend: left shift to place sign bit at bit 63, then arithmetic right shift.
                        ((raw << shift) as i64) >> shift
                    }
                    _ => return Err(OxiH5Error::NotImplemented(format!("signed int size {sz}"))),
                };
                Ok(Value::Int(v))
            } else {
                let v: u64 = match (sz, order) {
                    (1, _) => u64::from(b[0]),
                    (2, ByteOrder::Little) => u64::from(u16::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("u16 decode".into()))?,
                    )),
                    (2, ByteOrder::Big) => u64::from(u16::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("u16 decode".into()))?,
                    )),
                    (4, ByteOrder::Little) => u64::from(u32::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("u32 decode".into()))?,
                    )),
                    (4, ByteOrder::Big) => u64::from(u32::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("u32 decode".into()))?,
                    )),
                    (8, ByteOrder::Little) => u64::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("u64 decode".into()))?,
                    ),
                    (8, ByteOrder::Big) => u64::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("u64 decode".into()))?,
                    ),
                    // Odd-size unsigned integers (3, 5, 6, 7 bytes): pack into u64.
                    (sz, bo) if sz > 0 && sz <= 8 => {
                        let mut raw = 0u64;
                        match bo {
                            ByteOrder::Little => {
                                for (i, &byte) in b.iter().enumerate() {
                                    raw |= u64::from(byte) << (i * 8);
                                }
                            }
                            ByteOrder::Big => {
                                for (i, &byte) in b.iter().enumerate() {
                                    raw |= u64::from(byte) << ((sz - 1 - i) * 8);
                                }
                            }
                        }
                        raw
                    }
                    _ => {
                        return Err(OxiH5Error::NotImplemented(format!(
                            "unsigned int size {sz}"
                        )))
                    }
                };
                Ok(Value::Uint(v))
            }
        }

        // ------------------------------------------------------------------ floats
        Dtype::Float { size, order } => {
            let sz = *size;
            if bytes.len() < sz {
                return Err(OxiH5Error::Format(format!(
                    "float decode: need {sz} bytes, got {}",
                    bytes.len()
                )));
            }
            let b = &bytes[..sz];
            let v: f64 = match (sz, order) {
                (2, ByteOrder::Little) => {
                    let bits = u16::from_le_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("f16 decode".into()))?,
                    );
                    f64::from(crate::global_heap::f16_to_f32_pub(bits))
                }
                (2, ByteOrder::Big) => {
                    let bits = u16::from_be_bytes(
                        b.try_into()
                            .map_err(|_| OxiH5Error::Format("f16 decode".into()))?,
                    );
                    f64::from(crate::global_heap::f16_to_f32_pub(bits))
                }
                (4, ByteOrder::Little) => f64::from(f32::from_le_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("f32 decode".into()))?,
                )),
                (4, ByteOrder::Big) => f64::from(f32::from_be_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("f32 decode".into()))?,
                )),
                (8, ByteOrder::Little) => f64::from_le_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("f64 decode".into()))?,
                ),
                (8, ByteOrder::Big) => f64::from_be_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("f64 decode".into()))?,
                ),
                _ => return Err(OxiH5Error::NotImplemented(format!("float size {sz}"))),
            };
            Ok(Value::Float(v))
        }

        // ------------------------------------------------------------------ fixed strings
        Dtype::String {
            fixed_len: Some(n), ..
        } => {
            let n = *n;
            if bytes.len() < n {
                return Err(OxiH5Error::Format(format!(
                    "fixed string: need {n} bytes, got {}",
                    bytes.len()
                )));
            }
            let trimmed = bytes[..n].split(|&b| b == 0).next().unwrap_or(&bytes[..n]);
            let s = String::from_utf8(trimmed.to_vec())
                .map_err(|e| OxiH5Error::Format(format!("fixed string UTF-8: {e}")))?;
            Ok(Value::Str(s))
        }

        // ------------------------------------------------------------------ vlen strings
        Dtype::String {
            fixed_len: None, ..
        } => {
            let (seq_len, heap_addr, obj_idx) = parse_vlen_ref(bytes)?;
            if seq_len == 0 {
                return Ok(Value::Str(String::new()));
            }
            let raw = heap_object_bytes(file_data, heap_addr, obj_idx, heap_cache)?;
            let trimmed = raw.split(|&b| b == 0).next().unwrap_or(raw);
            let s = String::from_utf8(trimmed.to_vec())
                .map_err(|e| OxiH5Error::Format(format!("vlen string UTF-8: {e}")))?;
            Ok(Value::Str(s))
        }

        // ------------------------------------------------------------------ opaque
        Dtype::Opaque { size, .. } => Ok(Value::Opaque(bytes[..*size].to_vec())),

        // ------------------------------------------------------------------ bitfield
        Dtype::Bitfield { size, order } => {
            let sz = *size;
            if bytes.len() < sz {
                return Err(OxiH5Error::Format(format!(
                    "bitfield decode: need {sz} bytes, got {}",
                    bytes.len()
                )));
            }
            let b = &bytes[..sz];
            let v: u64 = match (sz, order) {
                (1, _) => u64::from(b[0]),
                (2, ByteOrder::Little) => u64::from(u16::from_le_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("bitfield u16".into()))?,
                )),
                (2, ByteOrder::Big) => u64::from(u16::from_be_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("bitfield u16".into()))?,
                )),
                (4, ByteOrder::Little) => u64::from(u32::from_le_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("bitfield u32".into()))?,
                )),
                (4, ByteOrder::Big) => u64::from(u32::from_be_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("bitfield u32".into()))?,
                )),
                (8, ByteOrder::Little) => u64::from_le_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("bitfield u64".into()))?,
                ),
                (8, ByteOrder::Big) => u64::from_be_bytes(
                    b.try_into()
                        .map_err(|_| OxiH5Error::Format("bitfield u64".into()))?,
                ),
                // Odd-size bitfields (3, 5, 6, 7 bytes): pack bytes into u64.
                (sz, bo) if sz > 0 && sz <= 8 => {
                    let mut raw = 0u64;
                    match bo {
                        ByteOrder::Little => {
                            for (i, &byte) in b.iter().enumerate() {
                                raw |= u64::from(byte) << (i * 8);
                            }
                        }
                        ByteOrder::Big => {
                            for (i, &byte) in b.iter().enumerate() {
                                raw |= u64::from(byte) << ((sz - 1 - i) * 8);
                            }
                        }
                    }
                    raw
                }
                _ => return Err(OxiH5Error::NotImplemented(format!("bitfield size {sz}"))),
            };
            Ok(Value::Bitfield(v))
        }

        // ------------------------------------------------------------------ reference
        Dtype::Reference { ref_type } => match ref_type {
            RefType::Object => {
                if bytes.len() < 8 {
                    return Err(OxiH5Error::Format(format!(
                        "object ref: need 8 bytes, got {}",
                        bytes.len()
                    )));
                }
                let addr = u64::from_le_bytes(
                    bytes[..8]
                        .try_into()
                        .map_err(|_| OxiH5Error::Format("object ref bytes".into()))?,
                );
                Ok(Value::ObjectRef(addr))
            }
            RefType::Region => {
                // A region reference is 12 bytes on disk:
                //   bytes [0..8]  — dataset object header address (u64 LE)
                //   bytes [8..12] — heap reference offset (u32 LE), pointing into global heap
                //                   where the serialised selection lives.
                if bytes.len() < 12 {
                    return Err(OxiH5Error::Format(format!(
                        "region ref: need 12 bytes, got {}",
                        bytes.len()
                    )));
                }
                let dataset_addr = u64::from_le_bytes(
                    bytes[0..8]
                        .try_into()
                        .map_err(|_| OxiH5Error::Format("region ref: dataset addr".into()))?,
                );
                let heap_offset = u32::from_le_bytes(
                    bytes[8..12]
                        .try_into()
                        .map_err(|_| OxiH5Error::Format("region ref: heap offset".into()))?,
                ) as u64;

                let selection = parse_region_selection(file_data, heap_offset, heap_cache)?;
                Ok(Value::RegionRef {
                    dataset_addr,
                    selection,
                })
            }
        },

        // ------------------------------------------------------------------ enum
        Dtype::Enum { base, .. } => {
            // Enum is stored as its base integer type; we surface the raw integer.
            let v = decode_one_value(file_data, bytes, base, heap_cache, depth + 1)?;
            let raw = match v {
                Value::Int(i) => i,
                Value::Uint(u) => u as i64,
                other => {
                    return Err(OxiH5Error::Format(format!(
                        "enum: unexpected base value {:?}",
                        other
                    )))
                }
            };
            Ok(Value::Enum(raw))
        }

        // ------------------------------------------------------------------ array
        Dtype::Array { base, dims } => {
            // Variable-length base types are stored as 16-byte heap references.
            let base_size = base.size().unwrap_or(16);
            let n_elems: usize = dims.iter().product();
            let needed = n_elems.checked_mul(base_size).ok_or_else(|| {
                OxiH5Error::Format("array decode: n_elems × base_size overflows".into())
            })?;
            if bytes.len() < needed {
                return Err(OxiH5Error::Format(format!(
                    "array decode: need {needed} bytes, got {}",
                    bytes.len()
                )));
            }
            let mut elems = Vec::with_capacity(n_elems);
            for i in 0..n_elems {
                let elem = &bytes[i * base_size..(i + 1) * base_size];
                elems.push(decode_one_value(
                    file_data,
                    elem,
                    base,
                    heap_cache,
                    depth + 1,
                )?);
            }
            Ok(Value::Sequence(elems))
        }

        // ------------------------------------------------------------------ vlen (non-string)
        Dtype::VarLen { base } => {
            let (seq_len, heap_addr, obj_idx) = parse_vlen_ref(bytes)?;
            if seq_len == 0 {
                return Ok(Value::Sequence(vec![]));
            }
            // Variable-length base types (vlen-of-vlen) are stored as 16-byte
            // heap references; use 16 bytes as the element footprint in that case.
            let base_size = base.size().unwrap_or(16);
            let seq_len = seq_len as usize;
            let raw = heap_object_bytes(file_data, heap_addr, obj_idx, heap_cache)?.to_vec();
            let needed = seq_len * base_size;
            if raw.len() < needed {
                return Err(OxiH5Error::Format(format!(
                    "vlen sequence: heap object has {} bytes, need {needed}",
                    raw.len()
                )));
            }
            let mut elems = Vec::with_capacity(seq_len);
            for j in 0..seq_len {
                let elem = &raw[j * base_size..(j + 1) * base_size];
                elems.push(decode_one_value(
                    file_data,
                    elem,
                    base,
                    heap_cache,
                    depth + 1,
                )?);
            }
            Ok(Value::Sequence(elems))
        }

        // ------------------------------------------------------------------ compound
        Dtype::Compound { fields } => {
            decode_compound_element(file_data, bytes, fields, heap_cache, depth + 1)
        }
    }
}

// ---------------------------------------------------------------------------
// Region reference helpers
// ---------------------------------------------------------------------------

/// Parse a region selection from the global heap.
///
/// `heap_offset` is a u32 value from bytes 8–11 of the 12-byte region reference,
/// cast to u64 and used as the absolute collection address.
///
/// The selection is stored in the heap collection at that address; the first
/// object (index 1) contains the serialised selection:
/// ```text
///  0  4   version + type (type: 1=point, 2=hyperslab)
///  4  4   unused/padding
///  8  4   dimensionality (N)
/// 12  4   unused
/// 16  …   type-specific payload
/// ```
/// Point list payload: `N * ndim` × u64 point coordinates.
/// Hyperslab payload:  `2 * N * ndim` × u64 → (start, count) per dim per block.
fn parse_region_selection(
    file_data: &[u8],
    heap_offset: u64,
    heap_cache: &mut HashMap<u64, GlobalHeap>,
) -> Result<RegionSelection, OxiH5Error> {
    let sel_bytes = heap_object_bytes(file_data, heap_offset, 1, heap_cache)?.to_vec();

    if sel_bytes.len() < 16 {
        return Err(OxiH5Error::Format(format!(
            "region ref selection: need at least 16 bytes, got {}",
            sel_bytes.len()
        )));
    }

    // Bytes 0–1: version; bytes 2–3: type
    let sel_type = u16::from_le_bytes(
        sel_bytes[2..4]
            .try_into()
            .map_err(|_| OxiH5Error::Format("region ref: sel_type bytes".into()))?,
    );
    let ndim = u32::from_le_bytes(
        sel_bytes[8..12]
            .try_into()
            .map_err(|_| OxiH5Error::Format("region ref: ndim bytes".into()))?,
    ) as usize;

    let payload = &sel_bytes[16..];

    match sel_type {
        1 => {
            // Point list: first u32 = number of points, then ndim u64s per point.
            if payload.len() < 4 {
                return Err(OxiH5Error::Format(
                    "region ref point list: missing count".into(),
                ));
            }
            let n_points = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("region ref: point count bytes".into()))?,
            ) as usize;
            let pt_data = &payload[4..];
            let needed = n_points.saturating_mul(ndim).saturating_mul(8);
            if pt_data.len() < needed {
                return Err(OxiH5Error::Format(format!(
                    "region ref point list: need {needed} bytes for {n_points} points × {ndim} dims, got {}",
                    pt_data.len()
                )));
            }
            let mut points = Vec::with_capacity(n_points);
            for p in 0..n_points {
                let mut coords = Vec::with_capacity(ndim);
                for d in 0..ndim {
                    let off = (p * ndim + d) * 8;
                    let v = u64::from_le_bytes(
                        pt_data[off..off + 8]
                            .try_into()
                            .map_err(|_| OxiH5Error::Format("region ref: point coord".into()))?,
                    );
                    coords.push(v);
                }
                points.push(coords);
            }
            Ok(RegionSelection::Points(points))
        }
        2 => {
            // Hyperslab: n_blocks blocks, each block = start(ndim u64) + count(ndim u64).
            if payload.len() < 4 {
                return Err(OxiH5Error::Format(
                    "region ref hyperslab: missing block count".into(),
                ));
            }
            let n_blocks = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("region ref: block count bytes".into()))?,
            ) as usize;
            let hs_data = &payload[4..];
            let needed = n_blocks
                .saturating_mul(2)
                .saturating_mul(ndim)
                .saturating_mul(8);
            if hs_data.len() < needed {
                return Err(OxiH5Error::Format(format!(
                    "region ref hyperslab: need {needed} bytes, got {}",
                    hs_data.len()
                )));
            }
            let mut slabs = Vec::with_capacity(n_blocks * ndim);
            for b in 0..n_blocks {
                for d in 0..ndim {
                    let start_off = (b * 2 * ndim + d) * 8;
                    let count_off = (b * 2 * ndim + ndim + d) * 8;
                    let start =
                        u64::from_le_bytes(hs_data[start_off..start_off + 8].try_into().map_err(
                            |_| OxiH5Error::Format("region ref: hyperslab start".into()),
                        )?);
                    let count =
                        u64::from_le_bytes(hs_data[count_off..count_off + 8].try_into().map_err(
                            |_| OxiH5Error::Format("region ref: hyperslab count".into()),
                        )?);
                    slabs.push((start, count));
                }
            }
            Ok(RegionSelection::Hyperslab(slabs))
        }
        other => Err(OxiH5Error::NotImplemented(format!(
            "region ref: unknown selection type {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Dataset convenience methods (decode full buffer)
// ---------------------------------------------------------------------------

/// Decode all compound elements in a dataset's raw byte buffer.
///
/// This is the main entry point for reading compound datasets.  `data` is the
/// raw element bytes (from `Dataset::data`), `fields` comes from
/// `Dtype::Compound { fields }`, and `n_elems` = `Dataset::len()`.
///
/// `elem_stride = 0` means "auto-compute from field layout".
#[inline]
pub fn decode_dataset_compound(
    file_data: &[u8],
    data: &[u8],
    fields: &[CompoundField],
    n_elems: usize,
    elem_stride: usize,
) -> Result<Vec<Value>, OxiH5Error> {
    decode_compound(file_data, data, fields, n_elems, elem_stride)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_heap::build_gcol_for_test;
    use oxih5_core::{ByteOrder, Charset, CompoundField, Dtype, RefType};

    // ------------------------------------------------------------------
    // parse_vlen_ref

    #[test]
    fn test_parse_vlen_ref_basic() {
        let mut ref_bytes = [0u8; 16];
        // length = 3
        ref_bytes[0..4].copy_from_slice(&3u32.to_le_bytes());
        // object index = 7
        ref_bytes[4..6].copy_from_slice(&7u16.to_le_bytes());
        // heap address = 0x1000
        ref_bytes[8..16].copy_from_slice(&0x1000u64.to_le_bytes());

        let (len, addr, idx) = parse_vlen_ref(&ref_bytes).unwrap();
        assert_eq!(len, 3);
        assert_eq!(addr, 0x1000);
        assert_eq!(idx, 7);
    }

    #[test]
    fn test_parse_vlen_ref_too_short() {
        assert!(parse_vlen_ref(&[0u8; 10]).is_err());
    }

    // ------------------------------------------------------------------
    // decode_vlen_strings

    #[test]
    fn test_decode_vlen_strings_basic() {
        // Build a GCOL at offset 0 with one object: "hello"
        let gcol = build_gcol_for_test(&[(1, b"hello")]);

        // Build the vlen-ref buffer: one 16-byte ref pointing to obj 1
        let mut refs = [0u8; 16];
        refs[0..4].copy_from_slice(&1u32.to_le_bytes()); // seq_len = 1 (ignored for strings; we use heap data directly)
        refs[4..6].copy_from_slice(&1u16.to_le_bytes()); // obj_idx = 1
        refs[8..16].copy_from_slice(&0u64.to_le_bytes()); // heap at offset 0

        let strings = decode_vlen_strings(&gcol, &refs, 1).unwrap();
        assert_eq!(strings, vec!["hello".to_string()]);
    }

    #[test]
    fn test_decode_vlen_strings_empty_element() {
        let gcol = build_gcol_for_test(&[]);

        let mut refs = [0u8; 16];
        // seq_len = 0 → empty string, no heap lookup
        refs[0..4].copy_from_slice(&0u32.to_le_bytes());

        let strings = decode_vlen_strings(&gcol, &refs, 1).unwrap();
        assert_eq!(strings, vec![String::new()]);
    }

    #[test]
    fn test_decode_vlen_strings_multiple() {
        let gcol = build_gcol_for_test(&[(1, b"alpha"), (2, b"beta")]);

        let mut refs = [0u8; 32];
        // ref 0: obj 1 at offset 0
        refs[0..4].copy_from_slice(&1u32.to_le_bytes());
        refs[4..6].copy_from_slice(&1u16.to_le_bytes());
        // refs[8..16] = 0 (heap addr 0)
        // ref 1: obj 2 at offset 0
        refs[16..20].copy_from_slice(&1u32.to_le_bytes());
        refs[20..22].copy_from_slice(&2u16.to_le_bytes());
        // refs[24..32] = 0

        let strings = decode_vlen_strings(&gcol, &refs, 2).unwrap();
        assert_eq!(strings[0], "alpha");
        assert_eq!(strings[1], "beta");
    }

    // ------------------------------------------------------------------
    // decode_object_refs

    #[test]
    fn test_decode_object_refs_basic() {
        let mut data = [0u8; 16];
        data[0..8].copy_from_slice(&0x0800u64.to_le_bytes());
        data[8..16].copy_from_slice(&u64::MAX.to_le_bytes()); // undefined ref

        let refs = decode_object_refs(&data, 2).unwrap();
        assert_eq!(refs[0], 0x0800);
        assert_eq!(refs[1], u64::MAX);
    }

    #[test]
    fn test_decode_object_refs_too_short() {
        assert!(decode_object_refs(&[0u8; 4], 1).is_err());
    }

    // ------------------------------------------------------------------
    // decode_one_value — primitives

    #[test]
    fn test_decode_int_signed() {
        let bytes = (-42i32).to_le_bytes();
        let dtype = Dtype::Int {
            size: 4,
            signed: true,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Int(-42));
    }

    #[test]
    fn test_decode_int_unsigned() {
        let bytes = 255u8.to_le_bytes();
        let dtype = Dtype::Int {
            size: 1,
            signed: false,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Uint(255));
    }

    #[test]
    fn test_decode_float_f64() {
        let pi = std::f64::consts::PI;
        let bytes = pi.to_le_bytes();
        let dtype = Dtype::Float {
            size: 8,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        if let Value::Float(f) = v {
            assert!((f - pi).abs() < 1e-10);
        } else {
            panic!("expected Value::Float");
        }
    }

    #[test]
    fn test_decode_fixed_string() {
        let mut bytes = b"rust\0\0\0\0".to_vec();
        bytes.resize(8, 0);
        let dtype = Dtype::String {
            fixed_len: Some(8),
            charset: Charset::Utf8,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Str("rust".into()));
    }

    #[test]
    fn test_decode_object_ref() {
        let addr = 0xDEADBEEFu64;
        let bytes = addr.to_le_bytes();
        let dtype = Dtype::Reference {
            ref_type: RefType::Object,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::ObjectRef(0xDEADBEEF));
    }

    // ------------------------------------------------------------------
    // decode_compound_element

    #[test]
    fn test_decode_compound_basic() {
        // Compound: {x: i32 LE at offset 0, y: f64 LE at offset 4}
        let fields = vec![
            CompoundField {
                name: "x".into(),
                offset: 0,
                dtype: Dtype::Int {
                    size: 4,
                    signed: true,
                    order: ByteOrder::Little,
                },
            },
            CompoundField {
                name: "y".into(),
                offset: 4,
                dtype: Dtype::Float {
                    size: 8,
                    order: ByteOrder::Little,
                },
            },
        ];
        let mut elem = [0u8; 12];
        elem[0..4].copy_from_slice(&7i32.to_le_bytes());
        elem[4..12].copy_from_slice(&2.5f64.to_le_bytes());

        let v = decode_compound_element(&[], &elem, &fields, &mut HashMap::new(), 0).unwrap();
        if let Value::Compound(pairs) = v {
            assert_eq!(pairs[0].0, "x");
            assert_eq!(pairs[0].1, Value::Int(7));
            assert_eq!(pairs[1].0, "y");
            if let Value::Float(f) = pairs[1].1 {
                assert!((f - 2.5).abs() < 1e-12);
            } else {
                panic!("expected Float for y");
            }
        } else {
            panic!("expected Compound");
        }
    }

    // ------------------------------------------------------------------
    // decode_compound (multi-element)

    #[test]
    fn test_decode_compound_multiple_elems() {
        let fields = vec![CompoundField {
            name: "val".into(),
            offset: 0,
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
        }];
        let mut data = Vec::new();
        for v in [10i32, 20, 30] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = decode_compound(&[], &data, &fields, 3, 0).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0],
            Value::Compound(vec![("val".into(), Value::Int(10))])
        );
        assert_eq!(
            result[2],
            Value::Compound(vec![("val".into(), Value::Int(30))])
        );
    }

    // ------------------------------------------------------------------
    // depth guard

    #[test]
    fn test_decode_depth_limit() {
        // Build a deeply nested Array<Array<...<i32>...>> that exceeds depth limit
        let inner = Dtype::Int {
            size: 4,
            signed: true,
            order: ByteOrder::Little,
        };
        // 20 levels of Array wrapping — exceeds MAX_DECODE_DEPTH (16)
        let mut dtype = inner;
        for _ in 0..20 {
            dtype = Dtype::Array {
                base: Box::new(dtype),
                dims: vec![1],
            };
        }
        let bytes = 0i32.to_le_bytes();
        let result = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0);
        assert!(
            result.is_err(),
            "expected depth limit error, got: {:?}",
            result
        );
    }

    // ------------------------------------------------------------------
    // Slice 3a: odd-size integer decode

    #[test]
    fn test_decode_signed_int_3bytes_le_positive() {
        // 3-byte LE signed int: 0x01_00_00 = 65536
        let bytes = [0x00u8, 0x00, 0x01];
        let dtype = Dtype::Int {
            size: 3,
            signed: true,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Int(65536));
    }

    #[test]
    fn test_decode_signed_int_3bytes_le_negative() {
        // 3-byte LE signed int: 0xFF_FF_FF = -1
        let bytes = [0xFF_u8, 0xFF, 0xFF];
        let dtype = Dtype::Int {
            size: 3,
            signed: true,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Int(-1));
    }

    #[test]
    fn test_decode_signed_int_5bytes_le() {
        // 5-byte LE signed int: encode -100 (0xFFFF_FFFF_9C in LE)
        let val: i64 = -100;
        let mut bytes = [0u8; 5];
        let full = val.to_le_bytes();
        bytes.copy_from_slice(&full[..5]);
        let dtype = Dtype::Int {
            size: 5,
            signed: true,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Int(-100));
    }

    #[test]
    fn test_decode_signed_int_6bytes_be() {
        // 6-byte BE signed int: 1 = 0x000000000001
        let bytes = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x01];
        let dtype = Dtype::Int {
            size: 6,
            signed: true,
            order: ByteOrder::Big,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Int(1));
    }

    #[test]
    fn test_decode_signed_int_7bytes_le_negative() {
        // 7-byte LE signed int: -1 (all 0xFF)
        let bytes = [0xFF_u8; 7];
        let dtype = Dtype::Int {
            size: 7,
            signed: true,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Int(-1));
    }

    #[test]
    fn test_decode_unsigned_int_3bytes_le() {
        // 3-byte LE unsigned: 0x01_02_03 → 0x030201 = 197121
        let bytes = [0x01u8, 0x02, 0x03];
        let dtype = Dtype::Int {
            size: 3,
            signed: false,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Uint(0x03_02_01));
    }

    #[test]
    fn test_decode_unsigned_int_5bytes_le() {
        // 5-byte LE unsigned: 0x01_00_00_00_00 = 4294967296
        let bytes = [0x00u8, 0x00, 0x00, 0x00, 0x01];
        let dtype = Dtype::Int {
            size: 5,
            signed: false,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Uint(4_294_967_296));
    }

    #[test]
    fn test_decode_bitfield_3bytes_le() {
        // 3-byte LE bitfield: 0x01_02_03 → 0x030201
        let bytes = [0x01u8, 0x02, 0x03];
        let dtype = Dtype::Bitfield {
            size: 3,
            order: ByteOrder::Little,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Bitfield(0x03_02_01));
    }

    #[test]
    fn test_decode_bitfield_6bytes_be() {
        // 6-byte BE bitfield
        let bytes = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0xFF];
        let dtype = Dtype::Bitfield {
            size: 6,
            order: ByteOrder::Big,
        };
        let v = decode_one_value(&[], &bytes, &dtype, &mut HashMap::new(), 0).unwrap();
        assert_eq!(v, Value::Bitfield(0xFF));
    }

    // ------------------------------------------------------------------
    // Slice 3b: vlen-of-vlen

    #[test]
    fn test_decode_vlen_of_vlen() {
        // Build inner GCOL: obj 1 = 2 i32 values (LE): [10, 20]
        let mut inner_data = Vec::new();
        inner_data.extend_from_slice(&10i32.to_le_bytes());
        inner_data.extend_from_slice(&20i32.to_le_bytes());
        let inner_gcol = build_gcol_for_test(&[(1, &inner_data)]);

        // Build outer GCOL at a different offset.
        // The outer sequence's heap object at index 1 contains one 16-byte vlen ref
        // pointing into inner_gcol (at offset 0).
        let mut outer_heap_obj = [0u8; 16];
        outer_heap_obj[0..4].copy_from_slice(&2u32.to_le_bytes()); // seq_len = 2
        outer_heap_obj[4..6].copy_from_slice(&1u16.to_le_bytes()); // obj_idx = 1
                                                                   // heap address = 0 (inner_gcol is placed at offset 0 in file_data)

        // file_data = inner_gcol (at 0) followed by outer_gcol (which we build separately)
        let inner_len = inner_gcol.len();
        let outer_gcol = build_gcol_for_test(&[(1, &outer_heap_obj)]);

        let mut file_data = inner_gcol.clone();
        file_data.extend_from_slice(&outer_gcol);

        // Outer vlen ref: seq_len=1, obj_idx=1, heap_addr = inner_len (outer_gcol start)
        let mut outer_ref = [0u8; 16];
        outer_ref[0..4].copy_from_slice(&1u32.to_le_bytes()); // 1 outer element
        outer_ref[4..6].copy_from_slice(&1u16.to_le_bytes()); // obj_idx=1 in outer_gcol
        outer_ref[8..16].copy_from_slice(&(inner_len as u64).to_le_bytes()); // outer_gcol addr

        // base_dtype = VarLen<i32> — the outer sequence contains vlen-of-i32 elements
        let base_dtype = Dtype::VarLen {
            base: Box::new(Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            }),
        };
        let result = decode_vlen_sequences(&file_data, &outer_ref, 1, &base_dtype).unwrap();
        assert_eq!(result.len(), 1, "expected 1 outer element");
        // The single outer element is itself a Sequence (the VarLen<i32> decoded from the inner heap).
        // Since base_dtype is VarLen, each outer element decodes as Value::Sequence.
        if let Value::Sequence(outer_seq) = &result[0] {
            // The outer_seq has one element (seq_len=1 in outer_heap_obj), which is
            // itself a Sequence of 2 i32s.
            assert_eq!(outer_seq.len(), 1, "expected 1 nested vlen element");
            if let Value::Sequence(inner) = &outer_seq[0] {
                assert_eq!(inner.len(), 2, "expected 2 i32 values in inner sequence");
                assert_eq!(inner[0], Value::Int(10));
                assert_eq!(inner[1], Value::Int(20));
            } else {
                panic!("expected inner Sequence, got {:?}", outer_seq[0]);
            }
        } else {
            panic!("expected Value::Sequence, got {:?}", result[0]);
        }
    }

    // ------------------------------------------------------------------
    // Slice 3b: compound with vlen field

    #[test]
    fn test_decode_compound_with_vlen_field() {
        // Compound: {id: i32 at offset 0, tags: vlen<string> at offset 4}
        // The vlen field occupies 16 bytes on disk.
        let gcol = build_gcol_for_test(&[(1, b"hello")]);

        let mut elem = [0u8; 20]; // 4 (i32) + 16 (vlen ref)
        elem[0..4].copy_from_slice(&42i32.to_le_bytes()); // id = 42
                                                          // vlen ref for "hello": seq_len=1, obj_idx=1, heap_addr=0
        elem[4..8].copy_from_slice(&1u32.to_le_bytes()); // seq_len
        elem[8..10].copy_from_slice(&1u16.to_le_bytes()); // obj_idx
                                                          // elem[12..20] heap addr = 0 (all zeros)

        let fields = vec![
            CompoundField {
                name: "id".into(),
                offset: 0,
                dtype: Dtype::Int {
                    size: 4,
                    signed: true,
                    order: ByteOrder::Little,
                },
            },
            CompoundField {
                name: "tags".into(),
                offset: 4,
                dtype: Dtype::String {
                    fixed_len: None,
                    charset: Charset::Utf8,
                },
            },
        ];

        let result =
            decode_compound_element(&gcol, &elem, &fields, &mut HashMap::new(), 0).unwrap();
        if let Value::Compound(pairs) = result {
            assert_eq!(pairs[0].1, Value::Int(42));
            assert_eq!(pairs[1].1, Value::Str("hello".into()));
        } else {
            panic!("expected Compound");
        }
    }

    // ------------------------------------------------------------------
    // Slice 8a: region reference decode

    #[test]
    fn test_decode_region_ref_points() {
        // Build a GCOL at offset 0 with one object (index 1): the selection bytes
        // Selection: type=1 (point), ndim=2, 1 point at (3, 7)
        let mut sel = [0u8; 40];
        sel[0] = 1; // version
                    // sel[1] = 0
        sel[2..4].copy_from_slice(&1u16.to_le_bytes()); // type = point list
                                                        // sel[4..8] unused
        sel[8..12].copy_from_slice(&2u32.to_le_bytes()); // ndim = 2
                                                         // sel[12..16] unused
                                                         // payload starts at offset 16
        sel[16..20].copy_from_slice(&1u32.to_le_bytes()); // n_points = 1
        sel[20..28].copy_from_slice(&3u64.to_le_bytes()); // point[0][0] = 3
        sel[28..36].copy_from_slice(&7u64.to_le_bytes()); // point[0][1] = 7

        let gcol = build_gcol_for_test(&[(1, &sel)]);

        // 12-byte region reference: dataset_addr=0x1234, heap_offset pointing to gcol at 0
        let mut ref_bytes = [0u8; 12];
        ref_bytes[0..8].copy_from_slice(&0x1234u64.to_le_bytes()); // dataset addr
        ref_bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // heap at offset 0

        let dtype = Dtype::Reference {
            ref_type: RefType::Region,
        };
        let v = decode_one_value(&gcol, &ref_bytes, &dtype, &mut HashMap::new(), 0).unwrap();

        if let Value::RegionRef {
            dataset_addr,
            selection: RegionSelection::Points(pts),
        } = v
        {
            assert_eq!(dataset_addr, 0x1234);
            assert_eq!(pts.len(), 1);
            assert_eq!(pts[0], vec![3u64, 7u64]);
        } else {
            panic!("expected RegionRef with Points, got {:?}", v);
        }
    }

    #[test]
    fn test_decode_region_ref_hyperslab() {
        // Build GCOL with hyperslab selection: type=2, ndim=1, 1 block: start=5, count=10
        let mut sel = [0u8; 40];
        sel[0] = 1; // version
        sel[2..4].copy_from_slice(&2u16.to_le_bytes()); // type = hyperslab
        sel[8..12].copy_from_slice(&1u32.to_le_bytes()); // ndim = 1
                                                         // payload at offset 16
        sel[16..20].copy_from_slice(&1u32.to_le_bytes()); // n_blocks = 1
        sel[20..28].copy_from_slice(&5u64.to_le_bytes()); // start[0] = 5
        sel[28..36].copy_from_slice(&10u64.to_le_bytes()); // count[0] = 10

        let gcol = build_gcol_for_test(&[(1, &sel)]);

        let mut ref_bytes = [0u8; 12];
        ref_bytes[0..8].copy_from_slice(&0xABCD_u64.to_le_bytes());
        // heap offset = 0

        let dtype = Dtype::Reference {
            ref_type: RefType::Region,
        };
        let v = decode_one_value(&gcol, &ref_bytes, &dtype, &mut HashMap::new(), 0).unwrap();

        if let Value::RegionRef {
            dataset_addr,
            selection: RegionSelection::Hyperslab(slabs),
        } = v
        {
            assert_eq!(dataset_addr, 0xABCD);
            assert_eq!(slabs.len(), 1);
            assert_eq!(slabs[0], (5u64, 10u64));
        } else {
            panic!("expected RegionRef with Hyperslab, got {:?}", v);
        }
    }
}
