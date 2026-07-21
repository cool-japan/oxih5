//! Filter pipeline message (0x000B) encoding.
//!
//! The message names the filters a chunk passed through on its way to disk, in
//! the order they were applied; a reader inverts them in reverse.  Only DEFLATE
//! is writable today, so the encoders below emit a one-filter pipeline.
//!
//! # Two versions, one switch
//!
//! Version 1 pads both the filter name and the client data out to 8-byte
//! boundaries; version 2 drops the padding and omits the name entirely for a
//! filter whose id is below 256.  [`PIPELINE_VERSION`] picks between them and
//! **everything else follows from it** — the message's size, its body, and the
//! constant the round-trip test checks — so flipping it is a one-line change
//! rather than an edit in three places that can half-apply.
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

/// HDF5 registered filter id for zlib DEFLATE.
const DEFLATE_FILTER_ID: u16 = 1;

/// The filter's name as libhdf5 writes it, NUL-terminated and already a
/// multiple of 8 bytes long, so version 1 needs no name padding.
const DEFLATE_NAME: &[u8; 8] = b"deflate\0";

/// Version of the filter pipeline message the writer emits.
///
/// See the module documentation; changing this changes the emitted bytes and
/// nothing else.
pub(super) const PIPELINE_VERSION: u8 = 1;

/// Body size of a version-1 one-filter pipeline message.
///
/// ```text
/// version(1) nfilters(1) reserved(6)
/// filter_id(2) name_len(2) flags(2) ndata(2)
/// "deflate\0"(8)
/// level(4) pad(4)          — one client-data value is odd, so v1 pads to 8
/// ```
const V1_BODY: usize = 8 + 8 + 8 + 8;

/// Body size of a version-2 one-filter pipeline message.
///
/// ```text
/// version(1) nfilters(1)
/// filter_id(2) flags(2) ndata(2)   — no name_len: DEFLATE's id is below 256
/// level(4)                         — v2 does not pad client data
/// ```
const V2_BODY: usize = 2 + 6 + 4;

/// Body size of the pipeline message this writer emits.
pub(super) const fn deflate_body_size() -> usize {
    if PIPELINE_VERSION == 1 {
        V1_BODY
    } else {
        V2_BODY
    }
}

/// Write a one-filter DEFLATE pipeline body at `start`; returns bytes written,
/// always [`deflate_body_size`].
pub(super) fn write_deflate_body(buf: &mut [u8], start: usize, level: u8) -> usize {
    if PIPELINE_VERSION == 1 {
        write_deflate_v1(buf, start, level)
    } else {
        write_deflate_v2(buf, start, level)
    }
}

/// Version-1 encoder — named, name-padded, client-data-padded.
fn write_deflate_v1(buf: &mut [u8], start: usize, level: u8) -> usize {
    fill_zero(buf, start, V1_BODY);
    buf[start] = 1; // version
    buf[start + 1] = 1; // number of filters
                        // bytes 2..8 reserved

    write_u16_le(buf, start + 8, DEFLATE_FILTER_ID);
    write_u16_le(buf, start + 10, DEFLATE_NAME.len() as u16);
    write_u16_le(buf, start + 12, 0); // flags: the filter is mandatory
    write_u16_le(buf, start + 14, 1); // one client-data value

    buf[start + 16..start + 24].copy_from_slice(DEFLATE_NAME);
    write_u32_le(buf, start + 24, u32::from(level));
    // bytes 28..32: the odd client-data value's padding.
    V1_BODY
}

/// Version-2 encoder — no name, no padding.
fn write_deflate_v2(buf: &mut [u8], start: usize, level: u8) -> usize {
    fill_zero(buf, start, V2_BODY);
    buf[start] = 2; // version
    buf[start + 1] = 1; // number of filters

    write_u16_le(buf, start + 2, DEFLATE_FILTER_ID);
    write_u16_le(buf, start + 4, 0); // flags
    write_u16_le(buf, start + 6, 1); // one client-data value
    write_u32_le(buf, start + 8, u32::from(level));
    V2_BODY
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_format::message::parse_filter_pipeline;

    /// Encode one body with each encoder and hand both to the real parser.
    fn parsed(body: &[u8]) -> oxih5_core::FilterPipeline {
        parse_filter_pipeline(body).expect("parse_filter_pipeline")
    }

    /// Both encoders must survive the parser oxih5 actually reads files with.
    ///
    /// The parser accepts an unknown version silently and reports *no filters*,
    /// so a mis-encoded version byte does not fail here by accident — the
    /// filter count assertion is what catches it.
    #[test]
    fn w1e_deflate_pipeline_msg_parses() {
        for (version, size, encode) in [
            (
                1u8,
                V1_BODY,
                write_deflate_v1 as fn(&mut [u8], usize, u8) -> usize,
            ),
            (2, V2_BODY, write_deflate_v2),
        ] {
            for level in [0u8, 1, 6, 9] {
                let mut buf = vec![0xEEu8; size + 8];
                assert_eq!(encode(&mut buf, 0, level), size, "v{version} body size");
                assert_eq!(buf[0], version, "v{version} version byte");
                assert!(
                    buf[size..].iter().all(|&b| b == 0xEE),
                    "v{version} wrote past its body"
                );

                let pipeline = parsed(&buf[..size]);
                assert_eq!(
                    pipeline.filters.len(),
                    1,
                    "v{version} level {level}: parser saw {} filters — an unknown \
                     version parses as an *empty* pipeline with Ok, so this is \
                     the assertion that catches a wrong version byte",
                    pipeline.filters.len()
                );
                let filter = &pipeline.filters[0];
                assert_eq!(filter.id, DEFLATE_FILTER_ID, "v{version} filter id");
                assert_eq!(filter.flags, 0, "v{version} flags");
                assert_eq!(
                    filter.client_data,
                    vec![u32::from(level)],
                    "v{version} client data must be the level and nothing else"
                );
            }
        }
    }

    /// Version 1 names the filter; version 2 does not, for an id below 256.
    #[test]
    fn only_version_1_carries_the_filter_name() {
        let mut buf = vec![0u8; V1_BODY];
        write_deflate_v1(&mut buf, 0, 6);
        assert_eq!(
            parsed(&buf).filters[0].name.as_deref(),
            Some("deflate"),
            "v1 stores the name"
        );

        let mut buf = vec![0u8; V2_BODY];
        write_deflate_v2(&mut buf, 0, 6);
        assert_eq!(parsed(&buf).filters[0].name, None, "v2 omits it");
    }

    /// Both bodies are exact; a stray byte would move every address after the
    /// message.
    #[test]
    fn body_sizes_are_what_the_object_header_reserves() {
        assert_eq!(V1_BODY, 32);
        assert_eq!(V2_BODY, 12);
        assert_eq!(
            deflate_body_size(),
            if PIPELINE_VERSION == 1 { 32 } else { 12 }
        );
    }

    /// The exact v1 byte string, so a reshuffle of the encoder shows up here
    /// rather than in a file libhdf5 rejects.
    #[test]
    fn v1_body_is_byte_exact() {
        let mut buf = vec![0u8; V1_BODY];
        write_deflate_v1(&mut buf, 0, 6);
        let mut want = vec![0u8; V1_BODY];
        want[0] = 1; // version
        want[1] = 1; // nfilters
        want[8] = 1; // filter id = deflate (u16 LE)
        want[10] = 8; // name length
        want[14] = 1; // one client-data value
        want[16..24].copy_from_slice(b"deflate\0");
        want[24] = 6; // level (u32 LE)
        assert_eq!(buf, want);
    }
}
