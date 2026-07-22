//! Object-header (v1) construction.
//!
//! A v1 object header is a 16-byte prefix followed by a sequence of messages,
//! each an 8-byte header plus a body padded up to an 8-byte boundary.  The root
//! group, sub-groups, and datasets all use that same container and differ only
//! in their message list — historically they were three separate writers, two of
//! which recomputed their own sizes and one of which hard-coded them.
//!
//! [`OhMsg`] is the message list, [`oh_size`] measures it, and [`write_oh`]
//! emits it.  Both go through a single size formula, [`OhMsg::body_size`], so
//! the allocate pass and the write pass cannot disagree; [`write_oh`] also
//! re-checks every message against it as it advances.

use oxih5_core::OxiH5Error;

use super::elem::{
    compact_layout_data, fill_value_bytes, write_datatype_body, ElemType, ResolvedAttr,
};
use super::format::{
    fill_zero, write_msg_header, write_u16_le, write_u32_le, write_u64_le, MSG_HDR_SIZE,
};
use super::pipeline;
use super::tree::{DatasetDesc, Filter};
use super::{check_size, narrow, pad8};

/// Size of the v1 object-header prefix.
const OH_PREFIX: usize = 16;

/// Body size of a symbol table message: B-tree address + local heap address.
const SYMBOL_TABLE_BODY: usize = 16;
/// Body size of a fill-value message (v2, defined but zero-length).
const FILL_VALUE_BODY: usize = 8;
/// Body size of a v3 contiguous data layout message: 18 used + 6 padding.
const LAYOUT_CONTIGUOUS_BODY: usize = 24;
/// Fixed part of a v3 chunked data layout message, before the chunk dimensions.
const LAYOUT_CHUNKED_PREFIX: usize = 11;
/// Fixed part of a v3 compact data layout message, before the inline data:
/// version, layout class, and a 16-bit inline-size field.
const LAYOUT_COMPACT_PREFIX: usize = 4;
/// Fixed part of a v1 dataspace message body, before the dimension vectors.
const DATASPACE_PREFIX: usize = 8;

/// The HDF5 "undefined address" sentinel (`H5_ADDR_UNDEF`, all bits set).
///
/// A contiguous dataset with no elements has no allocated raw-data storage, so
/// its layout message must carry this sentinel rather than a concrete file
/// offset: libhdf5 reads an undefined address paired with size 0 as "empty
/// dataset", but reads a *defined* address paired with size 0 as "invalid
/// dataset size, likely file corruption".  The previous writer left the write
/// cursor in the address field, which for a 0-byte payload aliases the next
/// object in the file.
const UNDEFINED_ADDRESS: u64 = u64::MAX;

/// Total bytes one message occupies: header plus body padded to 8 bytes.
///
/// The padded body size is also what [`write_oh`] declares in the message
/// header, so the reserved region and the declared region are the same region.
pub(super) const fn msg_total(body_size: usize) -> usize {
    MSG_HDR_SIZE + pad8(body_size)
}

/// Addresses referenced from object-header messages.
///
/// Every field is a fixed-width address, so filling one in can never change the
/// size of any message — which is exactly why addresses live here rather than in
/// [`OhMsg`].
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct OhAddrs {
    /// Symbol table B-tree, or chunk index B-tree for a chunked dataset.
    pub(super) btree: u64,
    /// Local heap holding the group's link names.
    pub(super) heap: u64,
    /// Raw data address for a contiguous dataset.
    pub(super) data: u64,
}

/// One message of a v1 object header.
///
/// Each variant carries everything that affects its own length and nothing that
/// does not; addresses come from [`OhAddrs`].
pub(super) enum OhMsg<'a> {
    /// 0x0011 — symbol table; its presence is what makes an object a group.
    SymbolTable,
    /// 0x0001 — dataspace, v1, always written with max dims present.
    Dataspace {
        /// Current extent of each dimension.
        dims: &'a [usize],
        /// When set, dimension 0 has an unlimited maximum extent.
        unlimited_dim0: bool,
    },
    /// 0x0003 — datatype.
    Datatype(ElemType),
    /// 0x0005 — fill value.
    FillValue {
        /// Space-allocation time recorded in the message: libhdf5 uses
        /// `Incremental` (3) for chunked layouts, which allocate chunk storage
        /// lazily, and `Late` (2) for contiguous layouts.  Matching it keeps the
        /// message byte-identical to libhdf5 for each layout kind.
        incremental: bool,
        /// The custom fill value's little-endian bytes, or `None` for the
        /// historic "defined, zero-length" default.  When present, the message
        /// declares a defined fill value of these bytes — the version-2 form
        /// libhdf5 writes for `h5py`'s `fillvalue=`, which the reader honours in
        /// a chunked dataset's unwritten regions.  When `None`, the message is
        /// byte-for-byte what the writer emitted before custom fills existed.
        value: Option<&'a [u8]>,
    },
    /// 0x0008 — contiguous data layout.
    LayoutContiguous {
        /// Size of the raw data area in bytes.
        size: u64,
    },
    /// 0x0008 — chunked data layout.
    LayoutChunked {
        /// One entry per dataspace dimension, followed by the element size —
        /// exactly the vector HDF5 stores.
        chunk_dims: &'a [u32],
    },
    /// 0x0008 — compact data layout: the data is inlined in the object header
    /// rather than addressed elsewhere.
    LayoutCompact {
        /// The raw little-endian data, stored inline; at most 64 KiB.
        data: &'a [u8],
    },
    /// 0x000B — filter pipeline, one to three of shuffle / deflate / fletcher32.
    FilterPipeline {
        /// Which filters the dataset's chunks pass through, in choice form; the
        /// canonical on-disk order is applied by [`pipeline`].
        filter: Filter,
        /// Element size in bytes, the shuffle filter's client-data value.
        elem_size: usize,
    },
    /// 0x000C — attribute, v1.
    Attr(&'a ResolvedAttr<'a>),
}

impl OhMsg<'_> {
    /// On-disk message type code.
    fn msg_type(&self) -> u16 {
        match self {
            OhMsg::SymbolTable => 0x0011,
            OhMsg::Dataspace { .. } => 0x0001,
            OhMsg::Datatype(_) => 0x0003,
            OhMsg::FillValue { .. } => 0x0005,
            OhMsg::LayoutContiguous { .. }
            | OhMsg::LayoutChunked { .. }
            | OhMsg::LayoutCompact { .. } => 0x0008,
            OhMsg::FilterPipeline { .. } => 0x000B,
            OhMsg::Attr(_) => 0x000C,
        }
    }

    /// Message header flags byte.
    fn flags(&self) -> u8 {
        match self {
            // Datatype, fill value and filter pipeline are marked constant:
            // they may not be modified in place by a subsequent writer, which
            // is also how libhdf5 writes all three.  A pipeline that could be
            // rewritten in place would let a reader disagree with the chunk
            // lengths already recorded in the B-tree keys.
            OhMsg::Datatype(_) | OhMsg::FillValue { .. } | OhMsg::FilterPipeline { .. } => 0x01,
            _ => 0x00,
        }
    }

    /// Human-readable name, used only in size-contract diagnostics.
    fn what(&self) -> &'static str {
        match self {
            OhMsg::SymbolTable => "symbol table message",
            OhMsg::Dataspace { .. } => "dataspace message",
            OhMsg::Datatype(_) => "datatype message",
            OhMsg::FillValue { .. } => "fill value message",
            OhMsg::LayoutContiguous { .. } => "contiguous layout message",
            OhMsg::LayoutChunked { .. } => "chunked layout message",
            OhMsg::LayoutCompact { .. } => "compact layout message",
            OhMsg::FilterPipeline { .. } => "filter pipeline message",
            OhMsg::Attr(_) => "attribute message",
        }
    }

    /// Unpadded body size — the writer's single size formula.
    fn body_size(&self) -> usize {
        match self {
            OhMsg::SymbolTable => SYMBOL_TABLE_BODY,
            // Dimension vector plus max-dimension vector.
            OhMsg::Dataspace { dims, .. } => DATASPACE_PREFIX + dims.len() * 8 * 2,
            OhMsg::Datatype(elem_type) => elem_type.dt_body_size(),
            // 4 fixed header bytes + a 4-byte size field + the value bytes; the
            // default (`None`) is size 0 and no value, i.e. the historic 8.
            OhMsg::FillValue { value, .. } => FILL_VALUE_BODY + value.map_or(0, <[u8]>::len),
            OhMsg::LayoutContiguous { .. } => LAYOUT_CONTIGUOUS_BODY,
            OhMsg::LayoutChunked { chunk_dims } => LAYOUT_CHUNKED_PREFIX + chunk_dims.len() * 4,
            OhMsg::LayoutCompact { data } => LAYOUT_COMPACT_PREFIX + data.len(),
            OhMsg::FilterPipeline { filter, .. } => pipeline::body_size(filter),
            OhMsg::Attr(attr) => attr.body_size(),
        }
    }

    /// Total bytes this message occupies in the header, padding included.
    fn total(&self) -> usize {
        msg_total(self.body_size())
    }

    /// Write this message's body at `start`; returns bytes written.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a field overflows its on-disk width.
    fn write_body(
        &self,
        buf: &mut [u8],
        start: usize,
        addrs: &OhAddrs,
    ) -> Result<usize, OxiH5Error> {
        match self {
            OhMsg::SymbolTable => {
                fill_zero(buf, start, SYMBOL_TABLE_BODY);
                write_u64_le(buf, start, addrs.btree);
                write_u64_le(buf, start + 8, addrs.heap);
                Ok(SYMBOL_TABLE_BODY)
            }

            OhMsg::Dataspace {
                dims,
                unlimited_dim0,
            } => {
                let body = self.body_size();
                fill_zero(buf, start, body);
                buf[start] = 0x01; // version = 1
                buf[start + 1] = narrow("dataspace dimensionality", dims.len())?;
                buf[start + 2] = 0x01; // flags = max dims present
                let max_base = start + DATASPACE_PREFIX + dims.len() * 8;
                for (i, &dim) in dims.iter().enumerate() {
                    write_u64_le(buf, start + DATASPACE_PREFIX + i * 8, dim as u64);
                    let max_dim = if *unlimited_dim0 && i == 0 {
                        u64::MAX
                    } else {
                        dim as u64
                    };
                    write_u64_le(buf, max_base + i * 8, max_dim);
                }
                Ok(body)
            }

            OhMsg::Datatype(elem_type) => write_datatype_body(buf, start, *elem_type),

            OhMsg::FillValue { incremental, value } => {
                let body = self.body_size();
                fill_zero(buf, start, body);
                buf[start] = 0x02; // version = 2
                                   // Space allocation time: chunked storage is
                                   // allocated incrementally (3), contiguous late
                                   // (2) — matching libhdf5 for each layout kind.
                buf[start + 1] = if *incremental { 0x03 } else { 0x02 };
                buf[start + 2] = 0x02; // fill write time = never
                buf[start + 3] = 0x01; // fill value defined
                let value = value.unwrap_or(&[]);
                // Size 0 with no value bytes is the historic zero-length default;
                // a non-empty value is the typed sentinel `set_fill_value_*` set.
                write_u32_le(buf, start + 4, narrow("fill value size", value.len())?);
                buf[start + 8..start + 8 + value.len()].copy_from_slice(value);
                Ok(body)
            }

            OhMsg::LayoutContiguous { size } => {
                fill_zero(buf, start, LAYOUT_CONTIGUOUS_BODY);
                buf[start] = 0x03; // version = 3
                buf[start + 1] = 0x01; // class = 1 (contiguous)
                                       // An empty contiguous dataset has no raw
                                       // storage: write the undefined address so
                                       // libhdf5 does not read the (aliasing)
                                       // write cursor as a defined size-0 extent
                                       // and reject the file as corrupt.
                let data_addr = if *size == 0 {
                    UNDEFINED_ADDRESS
                } else {
                    addrs.data
                };
                write_u64_le(buf, start + 2, data_addr);
                write_u64_le(buf, start + 10, *size);
                Ok(LAYOUT_CONTIGUOUS_BODY)
            }

            OhMsg::LayoutChunked { chunk_dims } => {
                let body = self.body_size();
                fill_zero(buf, start, body);
                buf[start] = 0x03; // version = 3
                buf[start + 1] = 0x02; // class = 2 (chunked)
                buf[start + 2] = narrow("chunk dimensionality", chunk_dims.len())?;
                write_u64_le(buf, start + 3, addrs.btree);
                for (i, &extent) in chunk_dims.iter().enumerate() {
                    write_u32_le(buf, start + LAYOUT_CHUNKED_PREFIX + i * 4, extent);
                }
                Ok(body)
            }

            OhMsg::LayoutCompact { data } => {
                let body = self.body_size();
                fill_zero(buf, start, body);
                buf[start] = 0x03; // version = 3
                buf[start + 1] = 0x00; // class = 0 (compact)
                                       // 16-bit inline-size field; the data
                                       // immediately follows the 4-byte prefix.
                write_u16_le(buf, start + 2, narrow("compact layout size", data.len())?);
                buf[start + LAYOUT_COMPACT_PREFIX..start + LAYOUT_COMPACT_PREFIX + data.len()]
                    .copy_from_slice(data);
                Ok(body)
            }

            OhMsg::FilterPipeline { filter, elem_size } => {
                Ok(pipeline::write_body(buf, start, filter, *elem_size))
            }

            OhMsg::Attr(attr) => attr.write_body(buf, start),
        }
    }
}

/// Total size of a v1 object header holding `msgs`.
pub(super) fn oh_size(msgs: &[OhMsg<'_>]) -> usize {
    OH_PREFIX + msgs.iter().map(OhMsg::total).sum::<usize>()
}

/// Write a complete v1 object header at `addr`; returns bytes written.
///
/// The returned size always equals [`oh_size`] for the same message list, and
/// `num_messages` is `msgs.len()` by construction.  oxih5's own reader ignores
/// that field, but libhdf5's v1 deserializer cross-checks it, so an off-by-one
/// there is not a cosmetic problem.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if a message body exceeds the 16-bit on-disk
/// size field, if the message count or header size overflows its field, or if
/// any message writes a different number of bytes than [`OhMsg::body_size`]
/// reserved for it.
pub(super) fn write_oh(
    buf: &mut [u8],
    addr: usize,
    msgs: &[OhMsg<'_>],
    addrs: &OhAddrs,
) -> Result<usize, OxiH5Error> {
    let total = oh_size(msgs);
    let header_data_size = total - OH_PREFIX;

    fill_zero(buf, addr, OH_PREFIX);
    buf[addr] = 0x01; // object header version = 1
                      // [1] reserved
    write_u16_le(
        buf,
        addr + 2,
        narrow("object header message count", msgs.len())?,
    );
    write_u32_le(buf, addr + 4, 1); // reference count
    write_u32_le(
        buf,
        addr + 8,
        narrow("object header data size", header_data_size)?,
    );
    write_u32_le(buf, addr + 12, 0); // reserved

    let mut pos = addr + OH_PREFIX;
    for msg in msgs {
        let body_size = msg.body_size();
        // The size declared in the message header is the *padded* body size —
        // exactly what `msg_total` reserves.  libhdf5's `H5O__chunk_deserialize`
        // walks a v1 header by advancing `8 + declared_size` and aborts the whole
        // chunk with "message not aligned" the moment that lands off an 8-byte
        // boundary, so declaring the unpadded size makes any object header
        // containing an odd-length message (a chunked layout, or an attribute
        // whose data size is not a multiple of 8) unreadable by libhdf5.  Our own
        // reader re-aligns either way, which is why this went unnoticed.
        let declared_size = pad8(body_size);
        let size_field = u16::try_from(declared_size).map_err(|_| {
            OxiH5Error::Format(format!(
                "internal writer error: {} body is {declared_size} bytes, over the 65535-byte limit",
                msg.what()
            ))
        })?;
        write_msg_header(buf, pos, msg.msg_type(), size_field, msg.flags());

        let body_start = pos + MSG_HDR_SIZE;
        check_size(
            msg.what(),
            msg.write_body(buf, body_start, addrs)?,
            body_size,
        )?;

        // Pad the message out to its 8-byte boundary.
        let end = pos + msg.total();
        fill_zero(buf, body_start + body_size, end - body_start - body_size);
        pos = end;
    }

    check_size("object header", pos - addr, total)?;
    Ok(total)
}

// ---------------------------------------------------------------------------
// Message lists
// ---------------------------------------------------------------------------

/// The message list of a group object header: a symbol table, then attributes.
///
/// Used for the root group and for every sub-group alike — that uniformity is
/// what gives sub-groups attributes.  They were previously sized by a
/// `sub_group_oh_size()` that took no attributes at all, so the only group in
/// the file that could carry any was the root.
pub(super) fn group_oh_msgs<'a>(attrs: &'a [ResolvedAttr<'a>]) -> Vec<OhMsg<'a>> {
    let mut msgs = Vec::with_capacity(1 + attrs.len());
    msgs.push(OhMsg::SymbolTable);
    msgs.extend(attrs.iter().map(OhMsg::Attr));
    msgs
}

/// The message list of a dataset object header.
///
/// This is the single definition of what a dataset header contains; the sizing
/// pass and the writing pass both go through it, so they cannot drift apart.
///
/// The filter pipeline sits between the fill value and the layout, where the
/// format specification lists it.  Position carries no meaning to any reader —
/// both libhdf5 and oxih5 collect messages by type — but a fixed one keeps the
/// golden-bytes oracle meaningful.
pub(super) fn dataset_oh_msgs<'a>(
    ds: &'a DatasetDesc,
    chunk_dims: &'a [u32],
    attrs: &'a [ResolvedAttr<'a>],
) -> Vec<OhMsg<'a>> {
    let is_chunked = ds.chunked().is_some();
    let mut msgs = Vec::with_capacity(5 + attrs.len());
    msgs.push(OhMsg::Dataspace {
        dims: &ds.shape,
        unlimited_dim0: ds.unlimited_dim0(),
    });
    msgs.push(OhMsg::Datatype(ds.elem_type));
    // Chunked storage is allocated incrementally by libhdf5; contiguous late.
    // A custom fill value, if one was set, rides the dataset's attribute list as
    // a sentinel; folding it in here (rather than emitting it as an attribute)
    // is what turns `set_fill_value_*` into the defined fill-value message the
    // reader and h5py honour.
    msgs.push(OhMsg::FillValue {
        incremental: is_chunked,
        value: fill_value_bytes(&ds.attrs),
    });
    // A stored filter always carries at least one active stage (the `set_*`
    // entry points never store an empty one), so an empty pipeline message is
    // never emitted.
    if let Some(filter) = ds.filter.filter(Filter::is_active) {
        msgs.push(OhMsg::FilterPipeline {
            filter,
            elem_size: ds.elem_type.byte_size(),
        });
    }
    // A compact request wins over the default contiguous layout: the data was
    // moved onto the attribute list as a sentinel, so it is inlined here and no
    // separate data area is reserved.  Compact and chunked are mutually
    // exclusive (`set_compact` refuses a chunked dataset).
    msgs.push(match (compact_layout_data(&ds.attrs), is_chunked) {
        (Some(data), _) => OhMsg::LayoutCompact { data },
        (None, true) => OhMsg::LayoutChunked { chunk_dims },
        (None, false) => OhMsg::LayoutContiguous {
            size: ds.data_len() as u64,
        },
    });
    msgs.extend(attrs.iter().map(OhMsg::Attr));
    msgs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::elem::{resolve_attrs, AttrDesc, AttrKind, ResolvedAttrKind};

    /// The historic hard-coded sub-group header: 40 bytes total, 24 bytes of
    /// message data, one message.
    #[test]
    fn group_header_matches_the_historic_constants() {
        let msgs = group_oh_msgs(&[]);
        assert_eq!(msgs.len(), 1);
        assert_eq!(oh_size(&msgs), 40);

        let mut buf = vec![0u8; 40];
        let addrs = OhAddrs {
            btree: 0x1111,
            heap: 0x2222,
            data: 0,
        };
        assert_eq!(write_oh(&mut buf, 0, &msgs, &addrs).expect("write_oh"), 40);

        assert_eq!(buf[0], 0x01); // version
        assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), 1); // num_messages
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 1); // refcount
        assert_eq!(u32::from_le_bytes(buf[8..12].try_into().unwrap()), 24); // data size
        assert_eq!(u16::from_le_bytes([buf[16], buf[17]]), 0x0011); // msg type
        assert_eq!(u16::from_le_bytes([buf[18], buf[19]]), 16); // body size
        assert_eq!(u64::from_le_bytes(buf[24..32].try_into().unwrap()), 0x1111);
        assert_eq!(u64::from_le_bytes(buf[32..40].try_into().unwrap()), 0x2222);
    }

    #[test]
    fn num_messages_tracks_the_message_list() {
        let attrs = vec![
            ResolvedAttr {
                name: "a",
                kind: ResolvedAttrKind::I32(1),
            },
            ResolvedAttr {
                name: "b",
                kind: ResolvedAttrKind::F64(2.0),
            },
        ];
        let msgs = group_oh_msgs(&attrs);
        let total = oh_size(&msgs);
        let mut buf = vec![0u8; total];
        assert_eq!(
            write_oh(&mut buf, 0, &msgs, &OhAddrs::default()).expect("write_oh"),
            total
        );
        assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), 3);
        assert_eq!(
            u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize,
            total - OH_PREFIX
        );
    }

    /// A group header grows with its attributes — any group, not just the root.
    #[test]
    fn group_oh_size_counts_its_attributes() {
        let attrs = vec![AttrDesc {
            name: "Conventions".to_string(),
            kind: AttrKind::FixedStr("CF-1.8".to_string()),
        }];
        let resolved = resolve_attrs(&attrs);
        // 16 prefix + 24 symbol table + one attribute message.
        let expected = 16 + 24 + msg_total(resolved[0].body_size());
        assert_eq!(oh_size(&group_oh_msgs(&resolved)), expected);
        // An attribute-free group is still exactly the historic 40 bytes.
        assert_eq!(oh_size(&group_oh_msgs(&[])), 40);
    }

    #[test]
    fn dataspace_body_grows_with_dimensionality() {
        for ndims in 0..4usize {
            let dims = vec![2usize; ndims];
            let msg = OhMsg::Dataspace {
                dims: &dims,
                unlimited_dim0: false,
            };
            assert_eq!(msg.body_size(), 8 + ndims * 16);
        }
    }

    #[test]
    fn chunked_layout_body_matches_the_historic_formula() {
        for ndims in 1..4usize {
            let chunk_dims = vec![4u32; ndims + 1];
            let msg = OhMsg::LayoutChunked {
                chunk_dims: &chunk_dims,
            };
            assert_eq!(msg.body_size(), 11 + (ndims + 1) * 4);
        }
    }

    /// A custom fill value is the version-2 defined message h5py writes for
    /// `fillvalue=`, byte-pinned against `h5py`'s object header: `02 <alloc> 02
    /// 01 <size u32> <value LE>`.  `None` stays the historic zero-length default.
    #[test]
    fn fill_value_message_matches_h5py() {
        // Contiguous f64 fillvalue=-999.0: `02 02 02 01 08 00 00 00 <-999 LE>`.
        let neg999 = (-999.0f64).to_le_bytes();
        let msg = OhMsg::FillValue {
            incremental: false,
            value: Some(&neg999),
        };
        assert_eq!(msg.body_size(), 16, "4 header + 4 size + 8 value");
        let mut buf = vec![0xAAu8; 16];
        assert_eq!(
            msg.write_body(&mut buf, 0, &OhAddrs::default())
                .expect("write"),
            16
        );
        let mut want = vec![0x02, 0x02, 0x02, 0x01, 0x08, 0x00, 0x00, 0x00];
        want.extend_from_slice(&neg999);
        assert_eq!(buf, want, "contiguous -999.0 f64");

        // Chunked i32 fillvalue=-7: alloc time 3, size 4, value `f9 ff ff ff`.
        let neg7 = (-7i32).to_le_bytes();
        let msg = OhMsg::FillValue {
            incremental: true,
            value: Some(&neg7),
        };
        assert_eq!(msg.body_size(), 12, "4 header + 4 size + 4 value");
        let mut buf = vec![0u8; 12];
        msg.write_body(&mut buf, 0, &OhAddrs::default())
            .expect("write");
        assert_eq!(
            buf,
            vec![0x02, 0x03, 0x02, 0x01, 0x04, 0x00, 0x00, 0x00, 0xf9, 0xff, 0xff, 0xff],
            "chunked -7 i32"
        );

        // The default (no custom value) is the historic 8-byte zero-length form.
        let msg = OhMsg::FillValue {
            incremental: false,
            value: None,
        };
        assert_eq!(msg.body_size(), 8, "default fill message is unchanged");
        let mut buf = vec![0xAAu8; 8];
        msg.write_body(&mut buf, 0, &OhAddrs::default())
            .expect("write");
        assert_eq!(buf, vec![0x02, 0x02, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00]);
    }

    /// The compact layout message inlines the data behind a 4-byte prefix,
    /// byte-pinned against h5py: `03 00 <size u16> <data>` (version 3, class 0).
    #[test]
    fn compact_layout_message_matches_h5py() {
        // Six little-endian i32 [0..6], as h5py's compact `d` stores them.
        let data: Vec<u8> = (0..6i32).flat_map(|v| v.to_le_bytes()).collect();
        let msg = OhMsg::LayoutCompact { data: &data };
        assert_eq!(msg.body_size(), 4 + 24, "4 prefix + 24 data");
        let mut buf = vec![0xAAu8; 4 + 24];
        assert_eq!(
            msg.write_body(&mut buf, 0, &OhAddrs::default())
                .expect("write"),
            28
        );
        let mut want = vec![0x03, 0x00, 0x18, 0x00]; // v3, compact, inline size 24
        want.extend_from_slice(&data);
        assert_eq!(buf, want);
        // The message is flagged 0 (rewritable), like every other layout message.
        assert_eq!(msg.flags(), 0x00);
        assert_eq!(msg.msg_type(), 0x0008);
    }

    #[test]
    fn unlimited_dataspace_marks_only_dimension_zero() {
        let dims = [3usize, 4];
        let msgs = [OhMsg::Dataspace {
            dims: &dims,
            unlimited_dim0: true,
        }];
        let total = oh_size(&msgs);
        let mut buf = vec![0u8; total];
        write_oh(&mut buf, 0, &msgs, &OhAddrs::default()).expect("write_oh");
        let body = 16 + MSG_HDR_SIZE;
        assert_eq!(
            u64::from_le_bytes(buf[body + 8..body + 16].try_into().unwrap()),
            3
        );
        assert_eq!(
            u64::from_le_bytes(buf[body + 24..body + 32].try_into().unwrap()),
            u64::MAX
        );
        assert_eq!(
            u64::from_le_bytes(buf[body + 32..body + 40].try_into().unwrap()),
            4
        );
    }
}
