//! Filter pipeline message (0x000B) encoding.
//!
//! The message names the filters a chunk passed through on its way to disk, in
//! the order they were applied; a reader inverts them in reverse.  This writer
//! composes a pipeline from three independent entry points — shuffle, deflate
//! and fletcher32 — and lays them down here in the one order libhdf5/h5py write:
//! **shuffle (id 2), then deflate (id 1), then fletcher32 (id 3)**.  A dataset
//! may carry any non-empty subset, so the message holds one, two or three
//! filter descriptions.
//!
//! # Two versions, one switch
//!
//! Version 1 pads both the filter name and the client data out to 8-byte
//! boundaries; version 2 drops the padding and omits the name entirely for a
//! filter whose id is below 256 (all three of ours are).  [`PIPELINE_VERSION`]
//! picks between them and **everything else follows from it** — the message's
//! size, its body, and the constant the round-trip test checks — so flipping it
//! is a one-line change rather than an edit in three places that can half-apply.
//!
//! Version 1 is what we emit.  The file already declares superblock v0, object
//! header v1, dataspace v1 and layout v3, which is the set libhdf5 produces for
//! `libver='earliest'`; a v1 pipeline is the member of that set, and therefore
//! the combination third-party readers have seen most of.  The v2 encoder
//! exists because the alternative to having it is discovering we need it and
//! writing it under pressure.
//!
//! # The trap this module's test exists for
//!
//! `oxih5_format::message::parse_filter_pipeline` returns an **empty pipeline
//! with `Ok`** for any version byte that is neither 1 nor 2 — no error, no
//! diagnostic.  A dataset whose pipeline said version 3 would therefore read
//! back as an uncompressed dataset, and its compressed bytes would be handed to
//! the caller as if they were elements.  Nothing downstream can catch that, so
//! the encoders are round-tripped through the real parser here.

use super::format::{fill_zero, write_u16_le, write_u32_le};
use super::pad8;
use super::tree::Filter;

/// HDF5 registered filter ids.
const SHUFFLE_FILTER_ID: u16 = 2;
const DEFLATE_FILTER_ID: u16 = 1;
const FLETCHER32_FILTER_ID: u16 = 3;

/// Filter names as libhdf5 writes them, NUL-terminated.  Version 1 stores the
/// name padded up to a multiple of 8 bytes; the padded *length* is what the
/// message's name-length field records.
const SHUFFLE_NAME: &[u8] = b"shuffle\0";
const DEFLATE_NAME: &[u8] = b"deflate\0";
const FLETCHER32_NAME: &[u8] = b"fletcher32\0";

/// libhdf5 marks shuffle and deflate as optional (`H5Z_FLAG_OPTIONAL`, bit 0)
/// in the pipeline message and fletcher32 as mandatory (flags 0); h5py records
/// exactly this, so matching it keeps a written pipeline byte-identical.
const H5Z_FLAG_OPTIONAL: u16 = 0x0001;

/// Version of the filter pipeline message the writer emits.
///
/// See the module documentation; changing this changes the emitted bytes and
/// nothing else.
pub(super) const PIPELINE_VERSION: u8 = 1;

/// Version-1 message header: version(1) nfilters(1) reserved(6).
const V1_HEADER: usize = 8;

/// Version-2 message header: version(1) nfilters(1).
const V2_HEADER: usize = 2;

/// One resolved filter description, ready to encode.
struct FilterDesc {
    /// Registered filter id.
    id: u16,
    /// Message flags (`H5Z_FLAG_OPTIONAL` for shuffle/deflate, 0 for fletcher32).
    flags: u16,
    /// NUL-terminated name, unpadded; version 1 pads it on disk.
    name: &'static [u8],
    /// The single client-data value (shuffle's element size, deflate's level),
    /// or `None` for a filter that carries none (fletcher32).
    client_data: Option<u32>,
}

/// The pipeline's filters in canonical on-disk order: shuffle, deflate,
/// fletcher32 — a subset of at most three.
///
/// `elem_size` becomes the shuffle filter's client-data value; it does not
/// affect any filter's on-disk *size*, so a placeholder is fine when sizing.
fn descriptors(filter: &Filter, elem_size: u32) -> Vec<FilterDesc> {
    let mut descs = Vec::with_capacity(3);
    if filter.shuffle {
        descs.push(FilterDesc {
            id: SHUFFLE_FILTER_ID,
            flags: H5Z_FLAG_OPTIONAL,
            name: SHUFFLE_NAME,
            client_data: Some(elem_size),
        });
    }
    if let Some(level) = filter.deflate {
        descs.push(FilterDesc {
            id: DEFLATE_FILTER_ID,
            flags: H5Z_FLAG_OPTIONAL,
            name: DEFLATE_NAME,
            client_data: Some(u32::from(level)),
        });
    }
    if filter.fletcher32 {
        descs.push(FilterDesc {
            id: FLETCHER32_FILTER_ID,
            flags: 0,
            name: FLETCHER32_NAME,
            client_data: None,
        });
    }
    descs
}

/// Bytes one filter description occupies in a version-1 message: the fixed
/// 8-byte header, then the name padded to 8, then the client data padded to 8.
fn v1_desc_size(desc: &FilterDesc) -> usize {
    8 + pad8(desc.name.len()) + pad8(desc.client_data.map_or(0, |_| 4))
}

/// Bytes one filter description occupies in a version-2 message: id(2)
/// flags(2) ndata(2) and the unpadded client data — no name, for an id below
/// 256.
fn v2_desc_size(desc: &FilterDesc) -> usize {
    6 + desc.client_data.map_or(0, |_| 4)
}

/// Body size of the pipeline message for `filter`.
pub(super) fn body_size(filter: &Filter) -> usize {
    let descs = descriptors(filter, 1);
    if PIPELINE_VERSION == 1 {
        V1_HEADER + descs.iter().map(v1_desc_size).sum::<usize>()
    } else {
        V2_HEADER + descs.iter().map(v2_desc_size).sum::<usize>()
    }
}

/// Write the pipeline body for `filter` at `start`; returns bytes written,
/// always [`body_size`].
///
/// `elem_size` is the dataset's element size in bytes, the shuffle filter's
/// client-data value.
pub(super) fn write_body(buf: &mut [u8], start: usize, filter: &Filter, elem_size: usize) -> usize {
    let descs = descriptors(filter, elem_size as u32);
    if PIPELINE_VERSION == 1 {
        write_v1(buf, start, &descs)
    } else {
        write_v2(buf, start, &descs)
    }
}

/// Version-1 encoder — each filter named, name-padded and client-data-padded.
fn write_v1(buf: &mut [u8], start: usize, descs: &[FilterDesc]) -> usize {
    let total = V1_HEADER + descs.iter().map(v1_desc_size).sum::<usize>();
    fill_zero(buf, start, total);
    buf[start] = 1; // version
    buf[start + 1] = descs.len() as u8; // number of filters
                                        // bytes 2..8 reserved, left zero

    let mut pos = start + V1_HEADER;
    for desc in descs {
        let name_len = pad8(desc.name.len());
        let ndata = u16::from(desc.client_data.is_some());
        write_u16_le(buf, pos, desc.id);
        write_u16_le(buf, pos + 2, name_len as u16);
        write_u16_le(buf, pos + 4, desc.flags);
        write_u16_le(buf, pos + 6, ndata);
        buf[pos + 8..pos + 8 + desc.name.len()].copy_from_slice(desc.name);
        // The name's padding bytes are already zero from `fill_zero`.
        pos += 8 + name_len;
        if let Some(value) = desc.client_data {
            write_u32_le(buf, pos, value);
            pos += pad8(4); // one 4-byte value, padded to the 8-byte boundary
        }
    }
    total
}

/// Version-2 encoder — no name, no padding, for filter ids below 256.
fn write_v2(buf: &mut [u8], start: usize, descs: &[FilterDesc]) -> usize {
    let total = V2_HEADER + descs.iter().map(v2_desc_size).sum::<usize>();
    fill_zero(buf, start, total);
    buf[start] = 2; // version
    buf[start + 1] = descs.len() as u8; // number of filters

    let mut pos = start + V2_HEADER;
    for desc in descs {
        let ndata = u16::from(desc.client_data.is_some());
        write_u16_le(buf, pos, desc.id);
        write_u16_le(buf, pos + 2, desc.flags);
        write_u16_le(buf, pos + 4, ndata);
        pos += 6;
        if let Some(value) = desc.client_data {
            write_u32_le(buf, pos, value);
            pos += 4;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_format::message::parse_filter_pipeline;

    /// A deflate-only pipeline at `level`.
    fn deflate(level: u8) -> Filter {
        Filter {
            deflate: Some(level),
            ..Filter::default()
        }
    }

    /// Every non-empty subset of the three filters, as `(shuffle, deflate,
    /// fletcher32)` tuples paired with the ids they must encode to, in order.
    fn cases() -> Vec<(Filter, Vec<u16>)> {
        vec![
            (
                Filter {
                    shuffle: true,
                    ..Filter::default()
                },
                vec![SHUFFLE_FILTER_ID],
            ),
            (deflate(6), vec![DEFLATE_FILTER_ID]),
            (
                Filter {
                    fletcher32: true,
                    ..Filter::default()
                },
                vec![FLETCHER32_FILTER_ID],
            ),
            (
                Filter {
                    shuffle: true,
                    deflate: Some(6),
                    ..Filter::default()
                },
                vec![SHUFFLE_FILTER_ID, DEFLATE_FILTER_ID],
            ),
            (
                Filter {
                    shuffle: true,
                    fletcher32: true,
                    ..Filter::default()
                },
                vec![SHUFFLE_FILTER_ID, FLETCHER32_FILTER_ID],
            ),
            (
                Filter {
                    deflate: Some(9),
                    fletcher32: true,
                    ..Filter::default()
                },
                vec![DEFLATE_FILTER_ID, FLETCHER32_FILTER_ID],
            ),
            (
                Filter {
                    shuffle: true,
                    deflate: Some(6),
                    fletcher32: true,
                },
                vec![SHUFFLE_FILTER_ID, DEFLATE_FILTER_ID, FLETCHER32_FILTER_ID],
            ),
        ]
    }

    /// Every pipeline the writer can emit must survive the parser oxih5 reads
    /// files with, at both message versions, and read back the exact filters —
    /// in the exact order — that were asked for.
    ///
    /// The parser accepts an unknown version silently and reports *no filters*,
    /// so a mis-encoded version byte does not fail here by accident: the filter
    /// list assertions are what catch it.
    #[test]
    fn every_pipeline_parses_to_the_filters_it_encodes() {
        let elem_size = 4u32;
        for (filter, ids) in cases() {
            let descs = descriptors(&filter, elem_size);
            for (version, header, encode) in [
                (
                    1u8,
                    V1_HEADER,
                    write_v1 as fn(&mut [u8], usize, &[FilterDesc]) -> usize,
                ),
                (2, V2_HEADER, write_v2),
            ] {
                let size = header
                    + descs
                        .iter()
                        .map(if version == 1 {
                            v1_desc_size
                        } else {
                            v2_desc_size
                        })
                        .sum::<usize>();
                let mut buf = vec![0xEEu8; size + 8];
                assert_eq!(encode(&mut buf, 0, &descs), size, "v{version} body size");
                assert_eq!(buf[0], version, "v{version} version byte");
                assert_eq!(buf[1] as usize, ids.len(), "v{version} filter count");
                assert!(
                    buf[size..].iter().all(|&b| b == 0xEE),
                    "v{version} wrote past its body"
                );

                let pipeline = parse_filter_pipeline(&buf[..size]).expect("parse");
                let got: Vec<u16> = pipeline.filters.iter().map(|f| f.id).collect();
                assert_eq!(got, ids, "v{version} filter ids and order");

                // Client data must be exactly what each filter carries.
                for f in &pipeline.filters {
                    match f.id {
                        SHUFFLE_FILTER_ID => assert_eq!(f.client_data, vec![elem_size]),
                        DEFLATE_FILTER_ID => assert!(!f.client_data.is_empty()),
                        FLETCHER32_FILTER_ID => assert!(f.client_data.is_empty()),
                        other => panic!("unexpected filter id {other}"),
                    }
                }
            }
        }
    }

    /// `body_size` and the emitters must agree, at whichever version is live.
    #[test]
    fn body_size_matches_what_the_emitter_writes() {
        for (filter, _) in cases() {
            let size = body_size(&filter);
            let mut buf = vec![0u8; size];
            assert_eq!(write_body(&mut buf, 0, &filter, 4), size);
        }
    }

    /// The exact bytes of the all-three pipeline, so a reshuffle of the encoder
    /// shows up here rather than in a file libhdf5 rejects.  This is the 80-byte
    /// body h5py (libver='earliest') writes for `shuffle=True,
    /// compression='gzip', compression_opts=6, fletcher32=True` over an int32
    /// dataset, read straight out of the file.
    #[test]
    fn v1_all_three_is_byte_exact_with_h5py() {
        let filter = Filter {
            shuffle: true,
            deflate: Some(6),
            fletcher32: true,
        };
        assert_eq!(PIPELINE_VERSION, 1, "this vector is the v1 encoding");
        let size = body_size(&filter);
        assert_eq!(size, 80);
        let mut buf = vec![0u8; size];
        assert_eq!(write_body(&mut buf, 0, &filter, 4), size);

        #[rustfmt::skip]
        let want: Vec<u8> = vec![
            0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // version 1, 3 filters, reserved
            0x02, 0x00, 0x08, 0x00, 0x01, 0x00, 0x01, 0x00, // shuffle: id 2, name_len 8, flags 1, ndata 1
            b's', b'h', b'u', b'f', b'f', b'l', b'e', 0x00, // "shuffle\0"
            0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // client data = elem_size 4, padded
            0x01, 0x00, 0x08, 0x00, 0x01, 0x00, 0x01, 0x00, // deflate: id 1, name_len 8, flags 1, ndata 1
            b'd', b'e', b'f', b'l', b'a', b't', b'e', 0x00, // "deflate\0"
            0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // client data = level 6, padded
            0x03, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, // fletcher32: id 3, name_len 16, flags 0, ndata 0
            b'f', b'l', b'e', b't', b'c', b'h', b'e', b'r', //
            b'3', b'2', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // "fletcher32\0" padded to 16
        ];
        assert_eq!(buf, want);
    }

    /// A one-filter deflate body is still exactly what libhdf5 writes for it,
    /// now with the optional flag h5py sets.
    #[test]
    fn v1_single_deflate_is_byte_exact() {
        let mut buf = vec![0u8; body_size(&deflate(6))];
        write_body(&mut buf, 0, &deflate(6), 8);
        #[rustfmt::skip]
        let want: Vec<u8> = vec![
            0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // version 1, 1 filter
            0x01, 0x00, 0x08, 0x00, 0x01, 0x00, 0x01, 0x00, // deflate: id 1, name_len 8, flags 1, ndata 1
            b'd', b'e', b'f', b'l', b'a', b't', b'e', 0x00, // "deflate\0"
            0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // level 6, padded
        ];
        assert_eq!(buf, want);
    }

    /// Version 1 names each filter; version 2 omits the name for ids below 256.
    #[test]
    fn only_version_1_carries_the_filter_names() {
        let descs = descriptors(&deflate(6), 4);
        let mut buf = vec![0u8; V1_HEADER + descs.iter().map(v1_desc_size).sum::<usize>()];
        write_v1(&mut buf, 0, &descs);
        assert_eq!(
            parse_filter_pipeline(&buf).expect("parse").filters[0]
                .name
                .as_deref(),
            Some("deflate"),
            "v1 stores the name"
        );

        let mut buf = vec![0u8; V2_HEADER + descs.iter().map(v2_desc_size).sum::<usize>()];
        write_v2(&mut buf, 0, &descs);
        assert_eq!(
            parse_filter_pipeline(&buf).expect("parse").filters[0].name,
            None,
            "v2 omits it"
        );
    }
}
