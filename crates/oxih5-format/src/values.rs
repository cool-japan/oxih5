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

    let base_size = base_dtype.size().ok_or_else(|| {
        OxiH5Error::NotImplemented("vlen sequence with variable-length base type".into())
    })?;

    let mut out = Vec::with_capacity(n_elems);
    let mut heap_cache: HashMap<u64, GlobalHeap> = HashMap::new();

    for i in 0..n_elems {
        let ref_bytes = &data[i * 16..(i + 1) * 16];
        let (seq_len, heap_addr, obj_idx) = parse_vlen_ref(ref_bytes)?;

        if seq_len == 0 {
            out.push(Value::Sequence(vec![]));
            continue;
        }

        let seq_len = seq_len as usize;
        let raw = heap_object_bytes(file_data, heap_addr, obj_idx, &mut heap_cache)?.to_vec();

        let needed = seq_len * base_size;
        if raw.len() < needed {
            return Err(OxiH5Error::Format(format!(
                "vlen sequence elem {i}: heap object has {} bytes, expected {} ({seq_len} × {base_size})",
                raw.len(),
                needed
            )));
        }

        let mut elems = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let elem_bytes = &raw[j * base_size..(j + 1) * base_size];
            elems.push(decode_one_value(
                file_data,
                elem_bytes,
                base_dtype,
                &mut heap_cache,
                0,
            )?);
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
        let field_size = field.dtype.size().ok_or_else(|| {
            OxiH5Error::NotImplemented(format!(
                "compound field '{}': variable-length field size not supported in compound",
                field.name
            ))
        })?;
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
            let fsz = f.dtype.size().ok_or_else(|| {
                OxiH5Error::NotImplemented(format!(
                    "cannot compute stride: field '{}' is variable-length",
                    f.name
                ))
            })?;
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
            RefType::Region => Err(OxiH5Error::NotImplemented("region reference decode".into())),
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
            let base_size = base.size().ok_or_else(|| {
                OxiH5Error::NotImplemented("array of variable-length base type".into())
            })?;
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
            let base_size = base.size().ok_or_else(|| {
                OxiH5Error::NotImplemented("vlen of variable-length base type".into())
            })?;
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
}
