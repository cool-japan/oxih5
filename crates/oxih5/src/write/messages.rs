//! HDF5 attribute, datatype, and dataspace message writers.
//!
//! This module contains all the helpers needed to write HDF5 attribute
//! messages (type 0x000C, v1), datatype messages (type 0x0003), dataspace
//! messages (type 0x0001), fill-value messages (type 0x0005), and
//! contiguous/chunked data-layout messages (type 0x0008).

use super::format::{
    write_msg_header, write_msg_header_flags, write_u16_le, write_u32_le, write_u64_le,
};
use super::{AttrDesc, AttrKind, DatasetDesc, ElemType, ResolvedAttr, ResolvedAttrKind};

// ---------------------------------------------------------------------------
// Size helpers
// ---------------------------------------------------------------------------

/// Total bytes consumed by one message in the object header:
/// 8-byte message header + body + padding to 8-byte boundary.
#[inline]
pub(super) fn msg_total(body_size: usize) -> usize {
    (8 + body_size + 7) & !7
}

/// (dtype_size, dspace_size, data_size) for a raw AttrKind.
pub(super) fn attr_kind_sizes_raw(kind: &AttrKind) -> (usize, usize, usize) {
    match kind {
        AttrKind::FixedStr(s) => (8, 8, s.len()),
        AttrKind::F64(_) => (24, 8, 8),
        AttrKind::I64(_) => (16, 8, 8),
        AttrKind::I32(_) => (16, 8, 4),
        AttrKind::ObjRefsByName(names) => (8, 24, names.len() * 8),
    }
}

/// (dtype_size, dspace_size, data_size) for a resolved attribute kind.
pub(super) fn resolved_attr_sizes(kind: &ResolvedAttrKind<'_>) -> (usize, usize, usize) {
    match kind {
        ResolvedAttrKind::FixedStr(s) => (8, 8, s.len()),
        ResolvedAttrKind::F64(_) => (24, 8, 8),
        ResolvedAttrKind::I64(_) => (16, 8, 8),
        ResolvedAttrKind::I32(_) => (16, 8, 4),
        ResolvedAttrKind::ObjRefs(refs) => (8, 24, refs.len() * 8),
    }
}

/// Unpadded body size of an attribute v1 message (for a raw AttrKind).
pub(super) fn attr_body_size(attr_name: &str, kind: &AttrKind) -> usize {
    let name_padded = (attr_name.len() + 1 + 7) & !7;
    let (dtype_size, dspace_size, data_size) = attr_kind_sizes_raw(kind);
    let dtype_padded = (dtype_size + 7) & !7;
    let dspace_padded = (dspace_size + 7) & !7;
    8 + name_padded + dtype_padded + dspace_padded + data_size
}

/// Unpadded body size of a resolved attribute v1 message.
pub(super) fn resolved_attr_body_size(attr_name: &str, kind: &ResolvedAttrKind<'_>) -> usize {
    let name_padded = (attr_name.len() + 1 + 7) & !7;
    let (dtype_size, dspace_size, data_size) = resolved_attr_sizes(kind);
    let dtype_padded = (dtype_size + 7) & !7;
    let dspace_padded = (dspace_size + 7) & !7;
    8 + name_padded + dtype_padded + dspace_padded + data_size
}

/// Body size of a simple fixed-length string attribute (for root group attrs).
pub(super) fn attr_body_size_str(name: &str, value: &str) -> usize {
    let name_padded = (name.len() + 1 + 7) & !7;
    let dtype_padded = (8usize + 7) & !7; // string dtype body = 8
    let dspace_padded = (8usize + 7) & !7; // scalar dspace body = 8
    let data_size = value.len();
    8 + name_padded + dtype_padded + dspace_padded + data_size
}

// ---------------------------------------------------------------------------
// Object header size computation
// ---------------------------------------------------------------------------

/// Compute total OH size (prefix + all messages) for a dataset.
///
/// `unlimited` → use chunked layout msg body instead of contiguous.
pub(super) fn compute_oh_size(
    ndims: usize,
    elem_type: &ElemType,
    attrs: &[AttrDesc],
    unlimited: bool,
) -> usize {
    // Dataspace v1 with max dims: 8 + ndims*8 dims + ndims*8 max_dims
    let ds_body_size = 8 + ndims * 8 * 2;
    let dt_body_size: usize = match elem_type {
        ElemType::F32 | ElemType::F64 => 24,
        ElemType::I32 | ElemType::I64 | ElemType::U8 => 16,
        // VLen string: 8-byte outer header + 8-byte base type (fixed char)
        ElemType::VlenStr => 16,
    };
    let fv_body_size: usize = 8;
    let lo_body_size: usize = if unlimited {
        // V3 chunked: version(1)+class(1)+dim(1)+btree_addr(8)+chunk_dims((ndims+1)*4)
        11 + (ndims + 1) * 4
    } else {
        24 // v3 contiguous (18 actual + 6 padding)
    };

    let attr_total: usize = attrs
        .iter()
        .map(|a| msg_total(attr_body_size(&a.name, &a.kind)))
        .sum();

    16 + msg_total(ds_body_size)
        + msg_total(dt_body_size)
        + msg_total(fv_body_size)
        + msg_total(lo_body_size)
        + attr_total
}

// ---------------------------------------------------------------------------
// Dataset object header writer
// ---------------------------------------------------------------------------

/// Write the full dataset OH (v1) at `oh_addr`.
///
/// `data_addr` for contiguous: the raw data address.
/// `btree_addr` for chunked: the chunk B-tree address (ignored if `!ds.unlimited`).
#[allow(clippy::too_many_arguments)]
pub(super) fn write_dataset_oh(
    buf: &mut [u8],
    oh_addr: usize,
    ds: &DatasetDesc,
    data_addr: u64,
    btree_addr: u64,
    resolved_attrs: &[ResolvedAttr<'_>],
) {
    let ndims = ds.shape.len();

    // ---- Body sizes ----
    let ds_body_size = 8 + ndims * 8 * 2;
    let dt_body_size: usize = match ds.elem_type {
        ElemType::F32 | ElemType::F64 => 24,
        ElemType::I32 | ElemType::I64 | ElemType::U8 => 16,
        ElemType::VlenStr => 16,
    };
    let fv_body_size: usize = 8;
    let lo_body_size: usize = if ds.unlimited {
        11 + (ndims + 1) * 4
    } else {
        24
    };

    let attr_total: usize = resolved_attrs
        .iter()
        .map(|a| msg_total(resolved_attr_body_size(a.name, &a.kind)))
        .sum();

    let header_data_size = msg_total(ds_body_size)
        + msg_total(dt_body_size)
        + msg_total(fv_body_size)
        + msg_total(lo_body_size)
        + attr_total;

    let num_messages: usize = 4 + resolved_attrs.len();

    // ---- OH v1 prefix (16 bytes) ----
    buf[oh_addr] = 0x01;
    buf[oh_addr + 1] = 0x00;
    write_u16_le(buf, oh_addr + 2, num_messages as u16);
    write_u32_le(buf, oh_addr + 4, 1);
    write_u32_le(buf, oh_addr + 8, header_data_size as u32);
    write_u32_le(buf, oh_addr + 12, 0);

    let mut pos = oh_addr + 16;

    // ---- Message 1: Dataspace (0x0001) ----
    write_msg_header(buf, pos, 0x0001, ds_body_size as u16);
    let body_start = pos + 8;
    buf[body_start] = 0x01; // version = 1
    buf[body_start + 1] = ndims as u8;
    buf[body_start + 2] = 0x01; // flags = 0x01 (max dims present)
    for (i, &dim) in ds.shape.iter().enumerate() {
        write_u64_le(buf, body_start + 8 + i * 8, dim as u64);
        let max_dim = if ds.unlimited && i == 0 {
            u64::MAX // unlimited first dimension
        } else {
            dim as u64
        };
        write_u64_le(buf, body_start + 8 + ndims * 8 + i * 8, max_dim);
    }
    pos += msg_total(ds_body_size);

    // ---- Message 2: Datatype (0x0003) ----
    write_msg_header_flags(buf, pos, 0x0003, dt_body_size as u16, 0x01);
    let dt_start = pos + 8;
    write_datatype_body(buf, dt_start, &ds.elem_type);
    pos += msg_total(dt_body_size);

    // ---- Message 3: Fill Value (0x0005) ----
    write_msg_header_flags(buf, pos, 0x0005, fv_body_size as u16, 0x01);
    let fv_start = pos + 8;
    buf[fv_start] = 0x02; // version = 2
    buf[fv_start + 1] = 0x02; // space_allocation_time = 2
    buf[fv_start + 2] = 0x02; // fill_write_time = 2
    buf[fv_start + 3] = 0x01; // fill_value_defined = 1
    pos += msg_total(fv_body_size);

    // ---- Message 4: Data Layout (0x0008) ----
    write_msg_header(buf, pos, 0x0008, lo_body_size as u16);
    let lo_start = pos + 8;
    if ds.unlimited {
        write_chunked_layout_body(
            buf,
            lo_start,
            ndims,
            btree_addr,
            &ds.chunk_shape,
            &ds.elem_type,
        );
    } else {
        write_contiguous_layout_body(buf, lo_start, data_addr, ds.data_len() as u64);
    }
    pos += msg_total(lo_body_size);

    // ---- Attribute messages (0x000C) ----
    for ra in resolved_attrs {
        write_attr_msg_v1(buf, pos, ra.name, &ra.kind);
        pos += msg_total(resolved_attr_body_size(ra.name, &ra.kind));
    }
    let _ = pos;
}

fn write_contiguous_layout_body(buf: &mut [u8], start: usize, data_addr: u64, data_size: u64) {
    buf[start] = 0x03; // version = 3
    buf[start + 1] = 0x01; // class = 1 (contiguous)
    write_u64_le(buf, start + 2, data_addr);
    write_u64_le(buf, start + 10, data_size);
}

fn write_chunked_layout_body(
    buf: &mut [u8],
    start: usize,
    ndims: usize,
    btree_addr: u64,
    chunk_shape: &[usize],
    elem_type: &ElemType,
) {
    let d = ndims + 1; // dimensionality including element-size dim
    buf[start] = 0x03; // version = 3
    buf[start + 1] = 0x02; // class = 2 (chunked)
    buf[start + 2] = d as u8;
    write_u64_le(buf, start + 3, btree_addr);
    // Chunk dims: chunk_shape[0..ndims] as u32, then elem_size as u32
    let elem_size: usize = match elem_type {
        ElemType::F32 => 4,
        ElemType::F64 => 8,
        ElemType::I32 => 4,
        ElemType::I64 => 8,
        ElemType::U8 => 1,
        // VlenStr datasets are never chunked/unlimited; elem_size is 16 bytes
        // (the on-disk global-heap reference size) but this branch is unreachable.
        ElemType::VlenStr => 16,
    };
    for i in 0..ndims {
        let cs = if i < chunk_shape.len() {
            chunk_shape[i]
        } else {
            1
        };
        write_u32_le(buf, start + 11 + i * 4, cs as u32);
    }
    write_u32_le(buf, start + 11 + ndims * 4, elem_size as u32);
}

// ---------------------------------------------------------------------------
// Attribute message writer (v1 format)
// ---------------------------------------------------------------------------

pub(super) fn write_attr_msg_v1(
    buf: &mut [u8],
    pos: usize,
    attr_name: &str,
    kind: &ResolvedAttrKind<'_>,
) {
    let body_size = resolved_attr_body_size(attr_name, kind);
    write_msg_header(buf, pos, 0x000C, body_size as u16);
    write_attr_body_v1(buf, pos + 8, attr_name, kind);
}

fn write_attr_body_v1(buf: &mut [u8], start: usize, attr_name: &str, kind: &ResolvedAttrKind<'_>) {
    let name_size = attr_name.len() + 1;
    let name_padded = (name_size + 7) & !7;
    let (dtype_size, dspace_size, data_size) = resolved_attr_sizes(kind);
    let dtype_padded = (dtype_size + 7) & !7;
    let dspace_padded = (dspace_size + 7) & !7;

    buf[start] = 0x01; // version = 1
    write_u16_le(buf, start + 2, name_size as u16);
    write_u16_le(buf, start + 4, dtype_size as u16);
    write_u16_le(buf, start + 6, dspace_size as u16);

    let mut p = start + 8;
    buf[p..p + attr_name.len()].copy_from_slice(attr_name.as_bytes());
    p += name_padded;
    write_attr_dtype_body(buf, p, kind);
    p += dtype_padded;
    write_attr_dspace_body(buf, p, kind);
    p += dspace_padded;
    write_attr_data(buf, p, kind, data_size);
}

/// Write a simple fixed-string attribute body (for root group attrs).
pub(super) fn write_str_attr_body(buf: &mut [u8], start: usize, name: &str, value: &str) {
    let kind = ResolvedAttrKind::FixedStr(value);
    write_attr_body_v1(buf, start, name, &kind);
}

fn write_attr_dtype_body(buf: &mut [u8], start: usize, kind: &ResolvedAttrKind<'_>) {
    match kind {
        ResolvedAttrKind::FixedStr(s) => {
            buf[start] = 0x13; // class 3 (string), version 1
            buf[start + 1] = 0x10; // UTF-8, null-padded
            write_u32_le(buf, start + 4, s.len() as u32);
        }
        ResolvedAttrKind::F64(_) => {
            let body: [u8; 24] = [
                0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b,
                0x00, 0x34, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];
            buf[start..start + 24].copy_from_slice(&body);
        }
        ResolvedAttrKind::I64(_) => {
            let body: [u8; 16] = [
                0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ResolvedAttrKind::I32(_) => {
            let body: [u8; 16] = [
                0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ResolvedAttrKind::ObjRefs(_) => {
            buf[start] = 0x17; // class 7 (reference), version 1
            write_u32_le(buf, start + 4, 8);
        }
    }
}

fn write_attr_dspace_body(buf: &mut [u8], start: usize, kind: &ResolvedAttrKind<'_>) {
    match kind {
        ResolvedAttrKind::FixedStr(_)
        | ResolvedAttrKind::F64(_)
        | ResolvedAttrKind::I64(_)
        | ResolvedAttrKind::I32(_) => {
            buf[start] = 0x01; // version = 1, ndims=0 (scalar)
        }
        ResolvedAttrKind::ObjRefs(refs) => {
            let n = refs.len() as u64;
            buf[start] = 0x01; // version = 1
            buf[start + 1] = 0x01; // ndims = 1
            buf[start + 2] = 0x01; // flags = 0x01 (max-dims present)
            write_u64_le(buf, start + 8, n);
            write_u64_le(buf, start + 16, n);
        }
    }
}

fn write_attr_data(buf: &mut [u8], start: usize, kind: &ResolvedAttrKind<'_>, data_size: usize) {
    match kind {
        ResolvedAttrKind::FixedStr(s) => {
            let n = data_size.min(s.len());
            buf[start..start + n].copy_from_slice(s.as_bytes());
        }
        ResolvedAttrKind::F64(v) => {
            buf[start..start + 8].copy_from_slice(&v.to_le_bytes());
        }
        ResolvedAttrKind::I64(v) => {
            buf[start..start + 8].copy_from_slice(&v.to_le_bytes());
        }
        ResolvedAttrKind::I32(v) => {
            buf[start..start + 4].copy_from_slice(&v.to_le_bytes());
        }
        ResolvedAttrKind::ObjRefs(refs) => {
            for (i, &addr) in refs.iter().enumerate() {
                write_u64_le(buf, start + i * 8, addr);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Datatype body encodings
// ---------------------------------------------------------------------------

pub(super) fn write_datatype_body(buf: &mut [u8], start: usize, elem_type: &ElemType) {
    match elem_type {
        ElemType::F32 => {
            let body: [u8; 24] = [
                0x11, 0x20, 0x1f, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x17, 0x08,
                0x00, 0x17, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];
            buf[start..start + 24].copy_from_slice(&body);
        }
        ElemType::F64 => {
            let body: [u8; 24] = [
                0x11, 0x20, 0x3f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0b,
                0x00, 0x34, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];
            buf[start..start + 24].copy_from_slice(&body);
        }
        ElemType::I32 => {
            let body: [u8; 16] = [
                0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ElemType::I64 => {
            let body: [u8; 16] = [
                0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ElemType::U8 => {
            let body: [u8; 16] = [
                0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ElemType::VlenStr => {
            // HDF5 datatype class 9 (VLen), subtype 1 (string).
            //
            // Layout (16 bytes = 8 outer + 8 base):
            //   [0]:    0x19  — class 9, version 1   ((1<<4)|9)
            //   [1]:    0x01  — type = 1 (string)
            //   [2-3]:  0x00  — reserved
            //   [4-7]:  0x10, 0x00, 0x00, 0x00  — element size = 16 (global heap ref)
            //   [8]:    0x13  — base class 3 (string), version 1   ((1<<4)|3)
            //   [9]:    0x10  — UTF-8 charset (bits 4-7 = 1)
            //   [10-11]: 0x00 — reserved
            //   [12-15]: 0x01, 0x00, 0x00, 0x00  — base element size = 1
            let body: [u8; 16] = [
                0x19, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x13, 0x10, 0x00, 0x00, 0x01, 0x00,
                0x00, 0x00,
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
    }
}
