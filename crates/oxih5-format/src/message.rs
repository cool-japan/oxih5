//! HDF5 object header message parsers.
//!
//! Covers message types used by the oxih5 facade:
//!   0x0001 = Dataspace
//!   0x0003 = Datatype  (delegated to datatype module)
//!   0x0005 = Fill Value
//!   0x0008 = Data Layout
//!   0x000B = Filter Pipeline
//!   0x000C = Attribute
//!   0x0011 = Symbol Table

use crate::datatype::parse_datatype as parse_datatype_from_body;
use crate::superblock::{read_u16_le, read_u32_le, read_u64_le};
use oxih5_core::{Attribute, Dataspace, Dtype, FilterInfo, FilterPipeline, OxiH5Error};

// ---------------------------------------------------------------------------
// Dataspace (message type 0x0001)
// ---------------------------------------------------------------------------

/// Parsed dataspace: shape dimensions and optional maximum dimensions.
pub struct DataspaceInfo {
    pub dims: Vec<u64>,
    /// Maximum dimensions, present when the dataspace flags bit 0 is set.
    ///
    /// A value of `u64::MAX` (`0xFFFF_FFFF_FFFF_FFFF`) signals that the
    /// corresponding axis is unlimited (`H5S_UNLIMITED`).
    pub max_dims: Option<Vec<u64>>,
}

/// Parse a dataspace message v1 or v2 body.
///
/// **V1 layout** (HDF5 `libver='earliest'`):
/// ```text
///  0    1    version (1)
///  1    1    dimensionality
///  2    1    flags (bit 0 = max-dims present, bit 1 = perm-indices present)
///  3    5    reserved (zero-padded to align to 8 bytes)
///  8    8*D  dim sizes (u64 LE each, D = dimensionality)
///  8+8D 8*D  max-dim sizes (u64 LE each) — present if flags bit 0 set
/// ```
///
/// **V2 layout** (HDF5 `libver='latest'`, compact header — no reserved region):
/// ```text
///  0    1    version (2)
///  1    1    dimensionality
///  2    1    flags (bit 0 = max-dims present)
///  3    1    type  (0=null, 1=simple, 2=scalar)
///  4    8*D  dim sizes (u64 LE each, D = dimensionality)
///  4+8D 8*D  max-dim sizes (u64 LE each) — present if flags bit 0 set
/// ```
pub fn parse_dataspace(body: &[u8]) -> Result<DataspaceInfo, OxiH5Error> {
    if body.is_empty() {
        return Err(OxiH5Error::Format(
            "dataspace body too short: 0 bytes".to_string(),
        ));
    }

    let version = body[0];
    let (header_size, min_body) = match version {
        1 => (8usize, 8usize),
        2 => (4usize, 4usize),
        _ => {
            return Err(OxiH5Error::Format(format!(
                "unsupported dataspace version: {version}"
            )))
        }
    };

    if body.len() < min_body {
        return Err(OxiH5Error::Format(format!(
            "dataspace body too short: {} bytes",
            body.len()
        )));
    }

    let dimensionality = body[1] as usize;
    // body[2] = flags (both v1 and v2)
    // v1: body[3..8] = reserved; v2: body[3] = type

    let dims_offset = header_size;
    let required = dims_offset + dimensionality * 8;
    if body.len() < required {
        return Err(OxiH5Error::Format(format!(
            "dataspace body too short for {dimensionality} dims: have {}, need {required}",
            body.len()
        )));
    }

    let flags = body[2];

    let mut dims = Vec::with_capacity(dimensionality);
    for i in 0..dimensionality {
        dims.push(read_u64_le(body, dims_offset + i * 8)?);
    }

    let max_dims = if flags & 0x01 != 0 {
        let max_offset = dims_offset + dimensionality * 8;
        let max_required = max_offset + dimensionality * 8;
        if body.len() < max_required {
            None
        } else {
            let mut md = Vec::with_capacity(dimensionality);
            for i in 0..dimensionality {
                md.push(read_u64_le(body, max_offset + i * 8)?);
            }
            Some(md)
        }
    } else {
        None
    };

    Ok(DataspaceInfo { dims, max_dims })
}

/// Parse a dataspace message body into the richer `Dataspace` core type.
///
/// Handles both v1 (8-byte header with reserved region) and v2 (4-byte compact header).
/// This variant is used internally by attribute parsing and for future API evolution.
pub fn parse_dataspace_rich(body: &[u8]) -> Result<Dataspace, OxiH5Error> {
    if body.is_empty() {
        return Err(OxiH5Error::Format(
            "dataspace body too short: 0 bytes".to_string(),
        ));
    }

    let version = body[0];
    let (header_size, min_body) = match version {
        1 => (8usize, 8usize),
        2 => (4usize, 4usize),
        _ => {
            return Err(OxiH5Error::Format(format!(
                "unsupported dataspace version: {version}"
            )))
        }
    };

    if body.len() < min_body {
        return Err(OxiH5Error::Format(format!(
            "dataspace body too short: {} bytes",
            body.len()
        )));
    }

    let dimensionality = body[1] as usize;
    let flags = body[2];

    // v2 type byte: 0=null, 1=simple, 2=scalar. Treat null/scalar as Scalar when ndims=0.
    if dimensionality == 0 {
        return Ok(Dataspace::Scalar);
    }

    let dims_offset = header_size;
    let required = dims_offset + dimensionality * 8;
    if body.len() < required {
        return Err(OxiH5Error::Format(format!(
            "dataspace body too short for {dimensionality} dims: have {}, need {required}",
            body.len()
        )));
    }

    let mut dims = Vec::with_capacity(dimensionality);
    for i in 0..dimensionality {
        dims.push(read_u64_le(body, dims_offset + i * 8)?);
    }

    let max_dims = if flags & 0x01 != 0 {
        let max_offset = dims_offset + dimensionality * 8;
        let max_required = max_offset + dimensionality * 8;
        if body.len() < max_required {
            None
        } else {
            let mut md = Vec::with_capacity(dimensionality);
            for i in 0..dimensionality {
                md.push(read_u64_le(body, max_offset + i * 8)?);
            }
            Some(md)
        }
    } else {
        None
    };

    Ok(Dataspace::Simple { dims, max_dims })
}

// ---------------------------------------------------------------------------
// Datatype (message type 0x0003)
// ---------------------------------------------------------------------------

/// Parsed datatype descriptor.
pub struct DatatypeInfo {
    pub dtype: Dtype,
}

/// Parse a datatype message body.
///
/// Delegates to the full `datatype` module which handles all 11 datatype classes.
pub fn parse_datatype(body: &[u8]) -> Result<DatatypeInfo, OxiH5Error> {
    let dtype = parse_datatype_from_body(body)?;
    Ok(DatatypeInfo { dtype })
}

// ---------------------------------------------------------------------------
// Data Layout (message type 0x0008)
// ---------------------------------------------------------------------------

/// Parsed data layout — either contiguous, compact (inline), chunked, or virtual.
#[derive(Debug, Clone)]
pub enum LayoutInfo {
    /// Raw data stored at a file offset.
    Contiguous { data_address: u64, data_size: u64 },
    /// Inline data stored in the object header message body.
    Compact { data: Vec<u8> },
    /// Chunked data with a B-tree index.
    Chunked {
        data_address: u64,
        dimensionality: u8,
        chunk_dims: Vec<u64>,
        index_type: u8,
    },
    /// Virtual dataset layout — assembled from source dataset regions.
    ///
    /// The mapping entries are stored in the Global Heap.
    VirtualDataset {
        /// Address of the global heap collection storing VDS mapping.
        heap_address: u64,
        /// Number of VDS mapping entries stored in the global heap.
        entry_count: u32,
    },
}

/// Parse a data layout message body.
///
/// Handles layout v1 and v3 for classes 0 (compact), 1 (contiguous), and 2 (chunked).
pub fn parse_layout(body: &[u8]) -> Result<LayoutInfo, OxiH5Error> {
    if body.len() < 2 {
        return Err(OxiH5Error::Format(format!(
            "layout body too short: {} bytes",
            body.len()
        )));
    }

    let version = body[0];
    let class = body[1];

    match (version, class) {
        // -------------------------------------------------------------------
        // V3 contiguous
        // -------------------------------------------------------------------
        (3, 1) => {
            if body.len() < 18 {
                return Err(OxiH5Error::Format(format!(
                    "layout v3 contiguous body too short: {} bytes",
                    body.len()
                )));
            }
            let data_address = read_u64_le(body, 2)?;
            let data_size = read_u64_le(body, 10)?;

            if data_address == u64::MAX {
                return Err(OxiH5Error::Format(
                    "layout: data address is undefined (u64::MAX)".to_string(),
                ));
            }

            Ok(LayoutInfo::Contiguous {
                data_address,
                data_size,
            })
        }

        // -------------------------------------------------------------------
        // V3 compact: body[2..4] = size (u16 LE), body[4..4+size] = data
        // -------------------------------------------------------------------
        (3, 0) => {
            if body.len() < 4 {
                return Err(OxiH5Error::Format(format!(
                    "layout v3 compact body too short: {} bytes",
                    body.len()
                )));
            }
            let data_size = read_u16_le(body, 2)? as usize;
            if body.len() < 4 + data_size {
                return Err(OxiH5Error::Format(format!(
                    "layout v3 compact: data {} bytes but only {} available",
                    data_size,
                    body.len() - 4
                )));
            }
            Ok(LayoutInfo::Compact {
                data: body[4..4 + data_size].to_vec(),
            })
        }

        // -------------------------------------------------------------------
        // V3 chunked
        // Layout v3 chunked:
        //   body[0] = 3 (version)
        //   body[1] = 2 (class)
        //   body[2] = dimensionality (D)
        //   body[3..11] = data address (u64 LE)
        //   body[11..11+4*D] = chunk dims (u32 LE each)
        //   last dim entry is the element size (implicit extra dimension in v3)
        // -------------------------------------------------------------------
        (3, 2) => {
            if body.len() < 11 {
                return Err(OxiH5Error::Format(format!(
                    "layout v3 chunked body too short: {} bytes",
                    body.len()
                )));
            }
            let dimensionality = body[2];
            let data_address = read_u64_le(body, 3)?;
            let ndims = dimensionality as usize;
            let chunk_end = 11 + ndims * 4;
            if body.len() < chunk_end {
                return Err(OxiH5Error::Format(format!(
                    "layout v3 chunked: need {} bytes for dims",
                    chunk_end
                )));
            }
            let mut chunk_dims = Vec::with_capacity(ndims);
            for i in 0..ndims {
                chunk_dims.push(read_u32_le(body, 11 + i * 4)? as u64);
            }
            Ok(LayoutInfo::Chunked {
                data_address,
                dimensionality,
                chunk_dims,
                index_type: 0,
            })
        }

        // -------------------------------------------------------------------
        // V1 contiguous (older files)
        // V1 layout: body[0]=1 (version), body[1]=class, body[2..4]=reserved,
        //            then dimensionality(4), reserved(4*4), then dims, then address, size
        // For class 1: simpler — try reading address + size at body[8..24]
        // -------------------------------------------------------------------
        (1, 1) => {
            if body.len() < 24 {
                return Err(OxiH5Error::Format(format!(
                    "layout v1 contiguous body too short: {} bytes",
                    body.len()
                )));
            }
            // V1 contiguous: data_address at body[8], data_size at body[16]
            let data_address = read_u64_le(body, 8)?;
            let data_size = read_u64_le(body, 16)?;
            if data_address == u64::MAX {
                return Err(OxiH5Error::Format(
                    "layout v1: data address is undefined (u64::MAX)".to_string(),
                ));
            }
            Ok(LayoutInfo::Contiguous {
                data_address,
                data_size,
            })
        }

        // -------------------------------------------------------------------
        // V4 and V5 chunked (layout class 2, libver='latest' / superblock v3)
        //
        // Body format discovered empirically (verified against FAHD addresses):
        //   body[0]  = version (4 or 5)
        //   body[1]  = class (2 = chunked)
        //   body[2]  = flags (reserved, usually 0x00)
        //   body[3]  = ndims_incl — chunk rank + 1 (includes element-size "dim")
        //   body[4]  = pline_size_flags — filter related, typically 0x01
        //   body[5..5+ndims_incl-1]  = chunk_dims[ndims_actual] as u8 LE each
        //              (ndims_actual = ndims_incl - 1)
        //   body[4+ndims_incl]  = elem_size_byte (low 8 bits of element size)
        //   body[5+ndims_incl]  = hdf5_idx_type (HDF5 internal: 3=fixed-array,
        //                                        4=extensible-array, 5=B-tree-v2)
        //   body[6+ndims_incl]  = max_nelmts_bits (FA/EA parameter, not used here)
        //   body[7+ndims_incl..15+ndims_incl] = chunk index address (u64 LE)
        //              — points to the FAHD (fixed array), EAHD (extensible array),
        //                or BTHD (B-tree v2) header
        //
        // HDF5 index-type values → oxih5-format ChunkIndex:
        //   3 (fixed-array)     → index_type = 1 (FixedArray)
        //   4 (extensible-array)→ index_type = 2 (ExtensibleArray)
        //   5 (B-tree v2)       → index_type = 3 (BTreeV2)
        // -------------------------------------------------------------------
        (4 | 5, 2) => {
            // Minimum: ver(1)+cls(1)+flags(1)+ndims_incl(1)+pline(1)
            //          +chunk_dims(ndims_actual bytes)+elem_size(1)+idx_type(1)
            //          +max_nelmts_bits(1)+index_addr(8)
            if body.len() < 4 {
                return Err(OxiH5Error::Format(format!(
                    "layout v{}/{} chunked body too short: {} bytes",
                    body[0],
                    body[1],
                    body.len()
                )));
            }
            let ndims_incl = body[3] as usize;
            if ndims_incl < 1 {
                return Err(OxiH5Error::Format(format!(
                    "layout v{} chunked: invalid ndims_incl=0",
                    body[0]
                )));
            }
            let ndims_actual = ndims_incl - 1;
            // Byte offset to hdf5_idx_type:  5 + ndims_actual  (= 5 + ndims_incl - 1 = 4 + ndims_incl)
            // Byte offset to max_nelmts_bits: 6 + ndims_actual  (= 5 + ndims_incl)
            // Byte offset to index_addr:      8 + ndims_actual  (= 7 + ndims_incl)
            let min_len = 8 + ndims_actual + 8; // = 16 + ndims_actual
            if body.len() < min_len {
                return Err(OxiH5Error::Format(format!(
                    "layout v{} chunked body too short: {} bytes (need {})",
                    body[0],
                    body.len(),
                    min_len,
                )));
            }
            // Chunk dims — one byte each (u8, suitable for real-world chunk sizes ≤ 255
            // elements; larger values would require a different encoding not yet seen).
            let mut chunk_dims = Vec::with_capacity(ndims_incl);
            for i in 0..ndims_actual {
                chunk_dims.push(body[5 + i] as u64);
            }
            // Append element size as the trailing "chunk dim" (v3/v4 convention).
            let elem_size_byte = body[4 + ndims_incl] as u64;
            chunk_dims.push(elem_size_byte);

            // HDF5 index type → internal ChunkIndex discriminant:
            // 3=FixedArray→1, 4=ExtensibleArray→2, 5=BTreeV2→3
            let hdf5_idx = body[5 + ndims_incl];
            let index_type: u8 = match hdf5_idx {
                3 => 1, // FixedArray
                4 => 2, // ExtensibleArray
                5 => 3, // BTreeV2
                other => {
                    return Err(OxiH5Error::Format(format!(
                        "layout v{} chunked: unknown HDF5 index type {other}",
                        body[0]
                    )))
                }
            };

            // Index address (FAHD / EAHD / BTHD pointer).
            // Layout: ..+pline(1)+chunk_dims(ndims_actual)+elem_size(1)+idx_type(1)+max_bits(1)+addr(8)
            // Base offset = 4 (ver+cls+flags+ndims_incl) + 1 (pline) = 5
            // Plus: ndims_actual (chunk_dims) + 1 (elem_size) + 1 (idx_type) + 1 (max_bits) = ndims_actual+3
            // Total offset = 5 + ndims_actual + 3 = 8 + ndims_actual
            let data_address = read_u64_le(body, 8 + ndims_actual)?;

            Ok(LayoutInfo::Chunked {
                data_address,
                dimensionality: ndims_incl as u8,
                chunk_dims,
                index_type,
            })
        }

        // -------------------------------------------------------------------
        // V4 virtual dataset (VDS, layout class 3)
        // Layout v4 VDS body (HDF5 spec §IV.A.2.q):
        //   body[0]     = 4  (version)
        //   body[1]     = 3  (class = virtual)
        //   body[2..10] = global heap collection address (u64 LE) where VDS
        //                 mapping is stored; GCOL contains source paths and
        //                 hyperslabs
        //   body[10..14]= number of VDS mapping entries (u32 LE)
        // -------------------------------------------------------------------
        (4, 3) => {
            if body.len() < 14 {
                return Err(OxiH5Error::Format(format!(
                    "layout v4 VDS body too short: {} bytes (need 14)",
                    body.len()
                )));
            }
            let heap_address = read_u64_le(body, 2)?;
            let entry_count = read_u32_le(body, 10)?;
            Ok(LayoutInfo::VirtualDataset {
                heap_address,
                entry_count,
            })
        }

        (v, c) => Err(OxiH5Error::Format(format!(
            "unsupported layout version={v} class={c}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Symbol Table (message type 0x0011)
// ---------------------------------------------------------------------------

/// Parsed symbol-table message (B-tree address + local heap address).
pub struct SymbolTableInfo {
    pub btree_address: u64,
    pub heap_address: u64,
}

/// Parse a symbol table message body.
///
/// Body: btree_address(8) + local_heap_address(8)
pub fn parse_symbol_table(body: &[u8]) -> Result<SymbolTableInfo, OxiH5Error> {
    if body.len() < 16 {
        return Err(OxiH5Error::Format(format!(
            "symbol table message body too short: {} bytes",
            body.len()
        )));
    }
    let btree_address = read_u64_le(body, 0)?;
    let heap_address = read_u64_le(body, 8)?;
    Ok(SymbolTableInfo {
        btree_address,
        heap_address,
    })
}

// ---------------------------------------------------------------------------
// Fill Value (message type 0x0005)
// ---------------------------------------------------------------------------

/// Parse a fill value message body.
///
/// Returns `Some(bytes)` if a fill value is defined, `None` otherwise.
pub fn parse_fill_value(body: &[u8]) -> Result<Option<Vec<u8>>, OxiH5Error> {
    if body.is_empty() {
        return Ok(None);
    }
    let version = body[0];
    match version {
        1 | 2 => {
            if body.len() < 4 {
                return Ok(None);
            }
            let defined = body[3];
            if defined == 0 {
                return Ok(None);
            }
            if body.len() < 8 {
                return Err(OxiH5Error::Format("fill_value: truncated".into()));
            }
            let size = read_u32_le(body, 4)? as usize;
            if body.len() < 8 + size {
                return Err(OxiH5Error::Format("fill_value: data truncated".into()));
            }
            Ok(Some(body[8..8 + size].to_vec()))
        }
        3 => {
            if body.len() < 2 {
                return Ok(None);
            }
            let flags = body[1];
            let defined = (flags >> 5) & 1;
            if defined == 0 {
                return Ok(None);
            }
            if body.len() < 6 {
                return Err(OxiH5Error::Format("fill_value v3: truncated".into()));
            }
            let size = read_u32_le(body, 2)? as usize;
            if body.len() < 6 + size {
                return Err(OxiH5Error::Format("fill_value v3: data truncated".into()));
            }
            Ok(Some(body[6..6 + size].to_vec()))
        }
        _ => Ok(None), // unknown version, skip gracefully
    }
}

// ---------------------------------------------------------------------------
// Filter Pipeline (message type 0x000B)
// ---------------------------------------------------------------------------

/// Parse a filter pipeline message body.
///
/// Returns a `FilterPipeline` containing all filter descriptors.
pub fn parse_filter_pipeline(body: &[u8]) -> Result<FilterPipeline, OxiH5Error> {
    if body.len() < 2 {
        return Err(OxiH5Error::Corrupted(
            "filter pipeline body too short".into(),
        ));
    }
    let version = body[0];
    let nfilters = body[1] as usize;
    let mut filters = Vec::with_capacity(nfilters);

    if version == 1 {
        let mut pos = 8usize; // version(1) + nfilters(1) + reserved(6)
        for _ in 0..nfilters {
            if pos + 8 > body.len() {
                break;
            }
            let filter_id = read_u16_le(body, pos)?;
            let name_len = read_u16_le(body, pos + 2)? as usize;
            let flags = read_u16_le(body, pos + 4)?;
            let ndata = read_u16_le(body, pos + 6)? as usize;
            pos += 8;

            // Name: name_len bytes padded to 8-byte boundary
            let name = if name_len > 0 && pos + name_len <= body.len() {
                let name_bytes = &body[pos..pos + name_len];
                let nul_pos = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_len);
                Some(
                    std::str::from_utf8(&name_bytes[..nul_pos])
                        .unwrap_or("")
                        .to_string(),
                )
            } else {
                None
            };
            let name_padded = name_len.div_ceil(8) * 8;
            pos += name_padded;

            // Client data
            let mut client_data = Vec::with_capacity(ndata);
            for _ in 0..ndata {
                if pos + 4 > body.len() {
                    break;
                }
                client_data.push(read_u32_le(body, pos)?);
                pos += 4;
            }
            // V1: pad client_data to 8-byte boundary (ndata must be even)
            if ndata % 2 != 0 {
                pos += 4;
            }

            filters.push(FilterInfo {
                id: filter_id,
                name,
                flags,
                client_data,
            });
        }
    } else if version == 2 {
        let mut pos = 2usize; // version(1) + nfilters(1)
        for _ in 0..nfilters {
            if pos + 2 > body.len() {
                break;
            }
            let filter_id = read_u16_le(body, pos)?;
            pos += 2;

            // In v2, name_len field only present if filter_id >= 256
            let name_len = if filter_id >= 256 {
                if pos + 2 > body.len() {
                    break;
                }
                let n = read_u16_le(body, pos)? as usize;
                pos += 2;
                n
            } else {
                0
            };

            if pos + 4 > body.len() {
                break;
            }
            let flags = read_u16_le(body, pos)?;
            pos += 2;
            let ndata = read_u16_le(body, pos)? as usize;
            pos += 2;

            let name = if name_len > 0 && pos + name_len <= body.len() {
                let name_bytes = &body[pos..pos + name_len];
                let nul_pos = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_len);
                Some(
                    std::str::from_utf8(&name_bytes[..nul_pos])
                        .unwrap_or("")
                        .to_string(),
                )
            } else {
                None
            };
            pos += name_len;

            let mut client_data = Vec::with_capacity(ndata);
            for _ in 0..ndata {
                if pos + 4 > body.len() {
                    break;
                }
                client_data.push(read_u32_le(body, pos)?);
                pos += 4;
            }

            filters.push(FilterInfo {
                id: filter_id,
                name,
                flags,
                client_data,
            });
        }
    }

    Ok(FilterPipeline { filters })
}

// ---------------------------------------------------------------------------
// Attribute (message type 0x000C)
// ---------------------------------------------------------------------------

/// Parse an attribute message body, returning a fully decoded `Attribute`.
pub fn parse_attribute(body: &[u8]) -> Result<Attribute, OxiH5Error> {
    if body.is_empty() {
        return Err(OxiH5Error::Format("attribute: empty body".into()));
    }
    let version = body[0];
    match version {
        1 => parse_attribute_v1(body),
        2 | 3 => parse_attribute_v2(body, version),
        v => Err(OxiH5Error::Format(format!(
            "attribute: unsupported version {}",
            v
        ))),
    }
}

/// Attribute message v1.
///
/// Layout:
/// ```text
///   0    1    version (1)
///   1    1    reserved
///   2    2    name size (u16 LE, includes NUL)
///   4    2    datatype message size (u16 LE)
///   6    2    dataspace message size (u16 LE)
///   8    ...  name (padded to 8-byte boundary)
///        ...  datatype message body (padded to 8-byte boundary)
///        ...  dataspace message body (padded to 8-byte boundary)
///        ...  attribute value bytes
/// ```
fn parse_attribute_v1(body: &[u8]) -> Result<Attribute, OxiH5Error> {
    if body.len() < 8 {
        return Err(OxiH5Error::Format(
            "attribute v1: body too short for header".into(),
        ));
    }
    let name_size = read_u16_le(body, 2)? as usize;
    let dtype_size = read_u16_le(body, 4)? as usize;
    let dspace_size = read_u16_le(body, 6)? as usize;

    let mut pos = 8usize;

    // Name (padded to 8-byte boundary)
    let name_padded = (name_size + 7) & !7;
    if pos + name_padded > body.len() {
        return Err(OxiH5Error::Format("attribute v1: name truncated".into()));
    }
    let name = extract_nul_string(&body[pos..pos + name_size]);
    pos += name_padded;

    // Datatype message body (padded to 8-byte boundary)
    let dtype_padded = (dtype_size + 7) & !7;
    if pos + dtype_padded > body.len() {
        return Err(OxiH5Error::Format(
            "attribute v1: datatype truncated".into(),
        ));
    }
    let dtype = parse_datatype_from_body(&body[pos..pos + dtype_size])?;
    pos += dtype_padded;

    // Dataspace message body (padded to 8-byte boundary)
    let dspace_padded = (dspace_size + 7) & !7;
    if pos + dspace_padded > body.len() {
        return Err(OxiH5Error::Format(
            "attribute v1: dataspace truncated".into(),
        ));
    }
    let dataspace = parse_dataspace_rich(&body[pos..pos + dspace_size])?;
    pos += dspace_padded;

    // Remaining bytes = attribute value data
    let data = body[pos..].to_vec();

    Ok(Attribute {
        name,
        dtype,
        dataspace,
        data,
    })
}

/// Attribute message v2/v3.
///
/// Layout:
/// ```text
///   0    1    version (2 or 3)
///   1    1    flags (bit 0=shared dtype, bit 1=shared dspace)
///   2    2    name size (u16 LE)
///   4    2    datatype message size (u16 LE)
///   6    2    dataspace message size (u16 LE)
///   8    1    name charset (v3 only)
///   ...  ...  name (NOT padded)
///        ...  datatype (NOT padded)
///        ...  dataspace (NOT padded)
///        ...  attribute value bytes
/// ```
fn parse_attribute_v2(body: &[u8], version: u8) -> Result<Attribute, OxiH5Error> {
    if body.len() < 8 {
        return Err(OxiH5Error::Format(format!(
            "attribute v{}: body too short for header",
            version
        )));
    }
    let name_size = read_u16_le(body, 2)? as usize;
    let dtype_size = read_u16_le(body, 4)? as usize;
    let dspace_size = read_u16_le(body, 6)? as usize;

    // v3 has an extra charset byte at position 8; v2 does not
    let mut pos = if version == 3 { 9usize } else { 8usize };

    // Name (NOT padded in v2/v3)
    if pos + name_size > body.len() {
        return Err(OxiH5Error::Format(format!(
            "attribute v{}: name truncated",
            version
        )));
    }
    let name = extract_nul_string(&body[pos..pos + name_size]);
    pos += name_size;

    // Datatype
    if pos + dtype_size > body.len() {
        return Err(OxiH5Error::Format(format!(
            "attribute v{}: datatype truncated",
            version
        )));
    }
    let dtype = parse_datatype_from_body(&body[pos..pos + dtype_size])?;
    pos += dtype_size;

    // Dataspace
    if pos + dspace_size > body.len() {
        return Err(OxiH5Error::Format(format!(
            "attribute v{}: dataspace truncated",
            version
        )));
    }
    let dataspace = parse_dataspace_rich(&body[pos..pos + dspace_size])?;
    pos += dspace_size;

    // Value data
    let data = body[pos..].to_vec();

    Ok(Attribute {
        name,
        dtype,
        dataspace,
        data,
    })
}

/// Extract a NUL-terminated string from a byte slice (stopping at the first NUL).
fn extract_nul_string(bytes: &[u8]) -> String {
    let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..nul]).unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Modification Time (message type 0x0012)
// ---------------------------------------------------------------------------

/// Parse a modification time message body (type `0x0012`).
///
/// Layout:
/// ```text
///   0    1    version (1)
///   1    3    reserved
///   4    4    seconds since Unix epoch (u32 LE)
/// ```
///
/// Returns the modification timestamp as a Unix second count.
pub fn parse_modification_time(body: &[u8]) -> Result<u32, OxiH5Error> {
    if body.len() < 8 {
        return Err(OxiH5Error::Format(format!(
            "modification_time: body too short ({} bytes, need 8)",
            body.len()
        )));
    }
    let version = body[0];
    if version != 1 {
        return Err(OxiH5Error::Format(format!(
            "modification_time: unsupported version {version}"
        )));
    }
    // version(1) + reserved(3) + seconds(4)
    let timestamp = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
    Ok(timestamp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_core::{ByteOrder, Charset, Dtype};

    // -----------------------------------------------------------------------
    // parse_dataspace
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_dataspace_basic() {
        let mut body = vec![0u8; 8 + 8 * 2];
        body[0] = 1; // version
        body[1] = 2; // dimensionality
                     // dim[0] = 10
        body[8..16].copy_from_slice(&10u64.to_le_bytes());
        // dim[1] = 20
        body[16..24].copy_from_slice(&20u64.to_le_bytes());
        let ds = parse_dataspace(&body).unwrap();
        assert_eq!(ds.dims, vec![10, 20]);
    }

    #[test]
    fn test_parse_dataspace_too_short() {
        assert!(parse_dataspace(&[]).is_err());
        assert!(parse_dataspace(&[1u8; 7]).is_err());
    }

    // -----------------------------------------------------------------------
    // parse_datatype (delegates to datatype module)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_datatype_int() {
        let mut body = vec![0u8; 8];
        body[0] = 1u8 << 4; // version=1, class=0 (int)
        body[1] = 0x08; // signed
        body[4..8].copy_from_slice(&4u32.to_le_bytes()); // size=4
        let dti = parse_datatype(&body).unwrap();
        assert_eq!(
            dti.dtype,
            Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little
            }
        );
    }

    #[test]
    fn test_parse_datatype_float() {
        let mut body = vec![0u8; 8];
        body[0] = (1u8 << 4) | 1u8; // version=1, class=1 (float)
        body[4..8].copy_from_slice(&4u32.to_le_bytes());
        let dti = parse_datatype(&body).unwrap();
        assert_eq!(
            dti.dtype,
            Dtype::Float {
                size: 4,
                order: ByteOrder::Little
            }
        );
    }

    #[test]
    fn test_parse_datatype_string() {
        let mut body = vec![0u8; 8];
        body[0] = (1u8 << 4) | 3u8; // class=3 (string)
        body[1] = 0x10; // UTF-8
        body[4..8].copy_from_slice(&10u32.to_le_bytes());
        let dti = parse_datatype(&body).unwrap();
        assert_eq!(
            dti.dtype,
            Dtype::String {
                fixed_len: Some(10),
                charset: Charset::Utf8
            }
        );
    }

    // -----------------------------------------------------------------------
    // parse_layout
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_layout_v3_contiguous() {
        let mut body = vec![0u8; 18];
        body[0] = 3; // version
        body[1] = 1; // class = contiguous
        body[2..10].copy_from_slice(&0x1000u64.to_le_bytes()); // address
        body[10..18].copy_from_slice(&64u64.to_le_bytes()); // size
        let layout = parse_layout(&body).unwrap();
        match layout {
            LayoutInfo::Contiguous {
                data_address,
                data_size,
            } => {
                assert_eq!(data_address, 0x1000);
                assert_eq!(data_size, 64);
            }
            _ => panic!("expected Contiguous"),
        }
    }

    #[test]
    fn test_parse_layout_v3_compact() {
        let inline = b"hello";
        let mut body = vec![0u8; 4 + inline.len()];
        body[0] = 3; // version
        body[1] = 0; // class = compact
        body[2..4].copy_from_slice(&(inline.len() as u16).to_le_bytes());
        body[4..].copy_from_slice(inline);
        let layout = parse_layout(&body).unwrap();
        match layout {
            LayoutInfo::Compact { data } => assert_eq!(data, inline),
            _ => panic!("expected Compact"),
        }
    }

    #[test]
    fn test_parse_layout_v3_chunked() {
        let ndims: u8 = 3;
        let mut body = vec![0u8; 11 + ndims as usize * 4];
        body[0] = 3;
        body[1] = 2; // chunked
        body[2] = ndims;
        body[3..11].copy_from_slice(&0x2000u64.to_le_bytes());
        body[11..15].copy_from_slice(&10u32.to_le_bytes()); // dim[0]
        body[15..19].copy_from_slice(&20u32.to_le_bytes()); // dim[1]
        body[19..23].copy_from_slice(&4u32.to_le_bytes()); // dim[2] (elem size)
        let layout = parse_layout(&body).unwrap();
        match layout {
            LayoutInfo::Chunked {
                data_address,
                dimensionality,
                chunk_dims,
                ..
            } => {
                assert_eq!(data_address, 0x2000);
                assert_eq!(dimensionality, 3);
                assert_eq!(chunk_dims, vec![10, 20, 4]);
            }
            _ => panic!("expected Chunked"),
        }
    }

    // -----------------------------------------------------------------------
    // parse_fill_value
    // -----------------------------------------------------------------------

    #[test]
    fn test_fill_value_v1_undefined() {
        let body = vec![1u8, 0, 0, 0]; // version=1, defined=0
        assert_eq!(parse_fill_value(&body).unwrap(), None);
    }

    #[test]
    fn test_fill_value_v1_defined() {
        let mut body = vec![1u8, 0, 0, 1]; // version=1, defined=1
        body.extend_from_slice(&4u32.to_le_bytes()); // size=4
        body.extend_from_slice(&42i32.to_le_bytes()); // value
        let fv = parse_fill_value(&body).unwrap().unwrap();
        assert_eq!(fv, 42i32.to_le_bytes());
    }

    #[test]
    fn test_fill_value_v3_undefined() {
        let body = vec![3u8, 0b0000_0000]; // version=3, defined=0
        assert_eq!(parse_fill_value(&body).unwrap(), None);
    }

    #[test]
    fn test_fill_value_v3_defined() {
        let mut body = vec![3u8, 0b0010_0000u8]; // version=3, defined bit set
        body.extend_from_slice(&4u32.to_le_bytes()); // size=4
        body.extend_from_slice(&99i32.to_le_bytes());
        let fv = parse_fill_value(&body).unwrap().unwrap();
        assert_eq!(fv, 99i32.to_le_bytes());
    }

    // -----------------------------------------------------------------------
    // parse_filter_pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_pipeline_empty() {
        // A valid v1 filter pipeline with zero filters:
        // version(1) + nfilters(1) + reserved(6) = 8 bytes minimum.
        let body: [u8; 8] = [1, 0, 0, 0, 0, 0, 0, 0]; // version=1, nfilters=0, reserved
        let fp = parse_filter_pipeline(&body).unwrap();
        assert!(fp.filters.is_empty());
    }

    #[test]
    fn test_filter_pipeline_too_short_returns_err() {
        // An empty slice must be rejected — the parser needs at least 2 bytes.
        assert!(parse_filter_pipeline(&[]).is_err());
    }

    #[test]
    fn test_filter_pipeline_v1_deflate() {
        // Version 1 filter pipeline with one filter: deflate (id=1)
        let name = b"deflate\x00"; // 8 bytes (already 8-byte aligned)
        let name_len = name.len() as u16;
        let ndata: u16 = 2; // 2 client data words (even, no padding)
        let mut body = vec![1u8]; // version
        body.push(1u8); // nfilters
        body.extend_from_slice(&[0u8; 6]); // reserved
                                           // filter header
        body.extend_from_slice(&1u16.to_le_bytes()); // filter_id = deflate
        body.extend_from_slice(&name_len.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes()); // flags
        body.extend_from_slice(&ndata.to_le_bytes());
        body.extend_from_slice(name);
        body.extend_from_slice(&6u32.to_le_bytes()); // cd[0] = level 6
        body.extend_from_slice(&0u32.to_le_bytes()); // cd[1]
        let fp = parse_filter_pipeline(&body).unwrap();
        assert_eq!(fp.filters.len(), 1);
        assert_eq!(fp.filters[0].id, 1);
        assert_eq!(fp.filters[0].name.as_deref(), Some("deflate"));
        assert_eq!(fp.filters[0].client_data[0], 6);
    }

    #[test]
    fn test_filter_pipeline_v2_builtin() {
        // Version 2 filter with id < 256 (no name_len field): shuffle (id=2)
        let mut body = vec![2u8]; // version
        body.push(1u8); // nfilters
        body.extend_from_slice(&2u16.to_le_bytes()); // filter_id = shuffle (< 256)
        body.extend_from_slice(&0u16.to_le_bytes()); // flags
        body.extend_from_slice(&0u16.to_le_bytes()); // ndata = 0
        let fp = parse_filter_pipeline(&body).unwrap();
        assert_eq!(fp.filters.len(), 1);
        assert_eq!(fp.filters[0].id, 2);
    }

    // -----------------------------------------------------------------------
    // parse_attribute
    // -----------------------------------------------------------------------

    fn build_int32_dtype_body() -> Vec<u8> {
        let mut b = vec![0u8; 8];
        b[0] = 1u8 << 4; // version=1, class=0 (int)
        b[1] = 0x08; // signed
        b[4..8].copy_from_slice(&4u32.to_le_bytes());
        b
    }

    fn build_scalar_dataspace_body() -> Vec<u8> {
        vec![1u8, 0, 0, 0, 0, 0, 0, 0] // version=1, dimensionality=0 => scalar
    }

    #[test]
    fn test_parse_attribute_v1_scalar() {
        let name = b"attr\x00\x00\x00\x00"; // "attr\0" padded to 8 bytes
        let dtype_body = build_int32_dtype_body(); // 8 bytes (already 8-byte multiple)
        let dspace_body = build_scalar_dataspace_body(); // 8 bytes
        let value = 42i32.to_le_bytes();

        let mut body = Vec::new();
        body.push(1u8); // version
        body.push(0u8); // reserved
        body.extend_from_slice(&(name.len() as u16).to_le_bytes()); // name size = 8
        body.extend_from_slice(&(dtype_body.len() as u16).to_le_bytes());
        body.extend_from_slice(&(dspace_body.len() as u16).to_le_bytes());
        body.extend_from_slice(name);
        body.extend_from_slice(&dtype_body);
        body.extend_from_slice(&dspace_body);
        body.extend_from_slice(&value);

        let attr = parse_attribute(&body).unwrap();
        assert_eq!(attr.name, "attr");
        assert_eq!(
            attr.dtype,
            Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little
            }
        );
        assert_eq!(attr.dataspace, Dataspace::Scalar);
        assert_eq!(attr.data, value.to_vec());
    }

    #[test]
    fn test_parse_attribute_v2_scalar() {
        let name = b"speed\x00"; // 6 bytes, not padded in v2
        let dtype_body = build_int32_dtype_body();
        let dspace_body = build_scalar_dataspace_body();
        let value = 99i32.to_le_bytes();

        let mut body = Vec::new();
        body.push(2u8); // version 2
        body.push(0u8); // flags
        body.extend_from_slice(&(name.len() as u16).to_le_bytes());
        body.extend_from_slice(&(dtype_body.len() as u16).to_le_bytes());
        body.extend_from_slice(&(dspace_body.len() as u16).to_le_bytes());
        // v2: no extra charset byte
        body.extend_from_slice(name);
        body.extend_from_slice(&dtype_body);
        body.extend_from_slice(&dspace_body);
        body.extend_from_slice(&value);

        let attr = parse_attribute(&body).unwrap();
        assert_eq!(attr.name, "speed");
        assert_eq!(attr.data, value.to_vec());
    }

    #[test]
    fn test_parse_attribute_unsupported_version() {
        let body = vec![5u8, 0, 0, 0, 0, 0, 0, 0];
        assert!(parse_attribute(&body).is_err());
    }

    // -----------------------------------------------------------------------
    // parse_symbol_table
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_symbol_table() {
        let mut body = vec![0u8; 16];
        body[0..8].copy_from_slice(&0xABCDu64.to_le_bytes());
        body[8..16].copy_from_slice(&0x1234u64.to_le_bytes());
        let sti = parse_symbol_table(&body).unwrap();
        assert_eq!(sti.btree_address, 0xABCD);
        assert_eq!(sti.heap_address, 0x1234);
    }

    // -----------------------------------------------------------------------
    // parse_modification_time
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_modification_time_basic() {
        let mut body = vec![0u8; 8];
        body[0] = 1; // version
                     // body[1..4] = reserved (zeros)
        let ts: u32 = 1_716_000_000; // some Unix timestamp
        body[4..8].copy_from_slice(&ts.to_le_bytes());
        assert_eq!(parse_modification_time(&body).unwrap(), ts);
    }

    #[test]
    fn test_parse_modification_time_epoch() {
        let mut body = vec![0u8; 8];
        body[0] = 1;
        // timestamp = 0 (Unix epoch)
        assert_eq!(parse_modification_time(&body).unwrap(), 0);
    }

    #[test]
    fn test_parse_modification_time_too_short() {
        let body = vec![1u8; 7]; // only 7 bytes
        assert!(parse_modification_time(&body).is_err());
    }

    #[test]
    fn test_parse_modification_time_wrong_version() {
        let mut body = vec![0u8; 8];
        body[0] = 2; // unsupported version
        assert!(parse_modification_time(&body).is_err());
    }

    #[test]
    fn test_parse_modification_time_max_ts() {
        let mut body = vec![0u8; 8];
        body[0] = 1;
        body[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(parse_modification_time(&body).unwrap(), u32::MAX);
    }
}
